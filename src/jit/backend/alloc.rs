//! Code memory in 64 KiB segments, each dual-mapped once and sub-allocated in
//! 64-byte units by a bitmap in the segment's own first units. A block's
//! writable address masks back to its segment's header, so a `CodeBlock`
//! frees itself without a pointer to the allocator; the allocator outlives
//! every block (regions hold an `Rc` of it).
//!
//! Each segment's first data units hold its copy of `exit_common`, which the
//! region exit stubs of the segment branch to.

use std::cell::{Cell, RefCell};
use std::io;
use std::ptr::NonNull;

use crate::jit::backend::code::{map_dual, sync_icache};

pub(crate) const SEG_SIZE: usize = 64 * 1024;
const UNIT: usize = 64;
const TOTAL_UNITS: usize = SEG_SIZE / UNIT;
const BITMAP_WORDS: usize = TOTAL_UNITS / 64;
const HEADER_UNITS: usize = 3;
const DATA_UNITS: usize = TOTAL_UNITS - HEADER_UNITS;

#[repr(C)]
struct SegHeader {
    rx_base: *mut u8,
    /// The executable address of the segment's `exit_common`.
    exit_common: *const u8,
    used_units: Cell<u16>,
    bitmap: [Cell<u64>; BITMAP_WORDS],
}

const _: () = assert!(size_of::<SegHeader>() <= HEADER_UNITS * UNIT);

impl SegHeader {
    fn test(&self, i: usize) -> bool {
        (self.bitmap[i / 64].get() >> (i % 64)) & 1 != 0
    }

    fn set(&self, i: usize) {
        let w = &self.bitmap[i / 64];
        w.set(w.get() | (1 << (i % 64)));
    }

    fn clear(&self, i: usize) {
        let w = &self.bitmap[i / 64];
        w.set(w.get() & !(1 << (i % 64)));
    }

    fn find_free(&self, len: usize) -> Option<usize> {
        let mut start = HEADER_UNITS;
        let mut run = 0;
        for i in HEADER_UNITS..TOTAL_UNITS {
            if self.test(i) {
                run = 0;
                start = i + 1;
            } else {
                run += 1;
                if run == len {
                    return Some(start);
                }
            }
        }
        None
    }

    fn mark(&self, start: usize, len: usize) {
        for i in start..start + len {
            debug_assert!(!self.test(i), "double-allocated unit {i}");
            self.set(i);
        }
        self.used_units.set(self.used_units.get() + len as u16);
    }

    fn free(&self, start: usize, len: usize) {
        for i in start..start + len {
            debug_assert!(self.test(i), "double-freed unit {i}");
            self.clear(i);
        }
        self.used_units.set(self.used_units.get() - len as u16);
    }
}

struct Segment {
    rw: NonNull<u8>,
    rx: NonNull<u8>,
}

impl Segment {
    fn header(&self) -> &SegHeader {
        unsafe { &*(self.rw.as_ptr() as *const SegHeader) }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.rw.as_ptr().cast(), SEG_SIZE);
            libc::munmap(self.rx.as_ptr().cast(), SEG_SIZE);
        }
    }
}

/// One region's code in a segment.
pub(crate) struct CodeBlock {
    rx: NonNull<u8>,
    rw: NonNull<u8>,
    len_units: u16,
}

impl CodeBlock {
    /// The executable address of the block's first byte.
    pub(crate) fn rx(&self) -> *const u8 {
        self.rx.as_ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.len_units as usize * UNIT
    }

    /// The segment's `exit_common`.
    pub(crate) fn exit_common(&self) -> *const u8 {
        self.header().exit_common
    }

    fn header(&self) -> &SegHeader {
        let base = self.rw.as_ptr() as usize & !(SEG_SIZE - 1);
        unsafe { &*(base as *const SegHeader) }
    }

    /// Copy `words` in at the start of the block and publish them.
    pub(crate) fn write(&self, words: &[u32]) {
        let len = words.len() * 4;
        assert!(len <= self.len());
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), self.rw.as_ptr(), len);
            sync_icache(self.rw.as_ptr(), self.rx.as_ptr(), len);
        }
    }

    /// Overwrite the 64-bit word at byte `off` and publish it.
    pub(crate) fn patch_u64(&self, off: usize, v: u64) {
        assert!(off + 8 <= self.len() && off % 8 == 0);
        unsafe {
            self.rw.as_ptr().add(off).cast::<u64>().write(v);
            sync_icache(self.rw.as_ptr().add(off), self.rx.as_ptr().add(off), 8);
        }
    }

    /// The block's bytes, for disassembly.
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.rw.as_ptr(), self.len()) }
    }
}

impl Drop for CodeBlock {
    fn drop(&mut self) {
        let rw = self.rw.as_ptr() as usize;
        let base = rw & !(SEG_SIZE - 1);
        let start = (rw - base) / UNIT;
        self.header().free(start, self.len_units as usize);
    }
}

/// Builds a segment's `exit_common` for code at the given executable address.
pub(crate) type SegmentInit = fn(rx: usize) -> Vec<u32>;

pub(crate) struct CodeAllocator {
    inner: RefCell<Inner>,
    init: SegmentInit,
}

struct Inner {
    segments: Vec<Segment>,
    primary: usize,
    /// Mapping failed once (no address in reach): refuse from then on.
    exhausted: bool,
}

impl CodeAllocator {
    pub(crate) fn new(init: SegmentInit) -> Self {
        CodeAllocator {
            inner: RefCell::new(Inner {
                segments: Vec::new(),
                primary: usize::MAX,
                exhausted: false,
            }),
            init,
        }
    }

    /// Room for `len` bytes, uninitialized.
    pub(crate) fn reserve(&self, len: usize) -> io::Result<CodeBlock> {
        let units = len.max(1).div_ceil(UNIT);
        if units > DATA_UNITS {
            return Err(io::Error::other("region too large for a code segment"));
        }
        let mut inner = self.inner.borrow_mut();
        let (seg, start) = inner.reserve(units, self.init)?;
        let s = &inner.segments[seg];
        let off = start * UNIT;
        let rw = unsafe { s.rw.as_ptr().add(off) };
        let rx = unsafe { s.header().rx_base.add(off) };
        Ok(CodeBlock {
            rx: NonNull::new(rx).expect("segment rx"),
            rw: NonNull::new(rw).expect("segment rw"),
            len_units: units as u16,
        })
    }
}

impl Inner {
    fn new_segment(&mut self, init: SegmentInit) -> io::Result<usize> {
        if self.exhausted {
            return Err(io::Error::other("code memory exhausted"));
        }
        let (rw, rx) = match map_dual(SEG_SIZE, (SEG_SIZE - 1) as u64) {
            Ok(p) => p,
            Err(e) => {
                self.exhausted = true;
                return Err(e);
            }
        };
        let h = unsafe { &mut *(rw.as_ptr() as *mut SegHeader) };
        h.rx_base = rx.as_ptr();
        for i in 0..HEADER_UNITS {
            h.set(i);
        }
        let seg = Segment { rw, rx };
        // `exit_common` takes the first data units.
        let start = HEADER_UNITS;
        let at = start * UNIT;
        let words = init(rx.as_ptr() as usize + at);
        let units = (words.len() * 4).div_ceil(UNIT);
        seg.header().mark(start, units);
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr().cast::<u8>(),
                rw.as_ptr().add(at),
                words.len() * 4,
            );
            sync_icache(rw.as_ptr().add(at), rx.as_ptr().add(at), words.len() * 4);
            let h = &mut *(rw.as_ptr() as *mut SegHeader);
            h.exit_common = rx.as_ptr().add(at);
        }
        self.segments.push(seg);
        Ok(self.segments.len() - 1)
    }

    fn reserve(&mut self, units: usize, init: SegmentInit) -> io::Result<(usize, usize)> {
        if self.primary != usize::MAX
            && let Some(u) = self.segments[self.primary].header().find_free(units)
        {
            self.segments[self.primary].header().mark(u, units);
            return Ok((self.primary, u));
        }
        let mut found = None;
        for (i, s) in self.segments.iter().enumerate() {
            if i == self.primary {
                continue;
            }
            let h = s.header();
            if (h.used_units.get() as usize) * 2 < DATA_UNITS
                && let Some(u) = h.find_free(units)
            {
                found = Some((i, u));
                break;
            }
        }
        if let Some((i, u)) = found {
            self.segments[i].header().mark(u, units);
            self.primary = i;
            return Ok((i, u));
        }
        let i = self.new_segment(init)?;
        let u = self.segments[i]
            .header()
            .find_free(units)
            .ok_or_else(|| io::Error::other("region too large for a code segment"))?;
        self.segments[i].header().mark(u, units);
        self.primary = i;
        Ok((i, u))
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    fn no_exit(_rx: usize) -> Vec<u32> {
        vec![0xD420_0000]
    }

    fn ret_const(alloc: &CodeAllocator, imm: u16) -> CodeBlock {
        let b = alloc.reserve(8).expect("reserve");
        b.write(&[0xD280_0000 | (imm as u32) << 5, 0xD65F_03C0]);
        b
    }

    fn call(b: &CodeBlock) -> u64 {
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(b.rx()) };
        f()
    }

    #[test]
    fn allocates_runs_and_frees() {
        let alloc = CodeAllocator::new(no_exit);
        let blocks: Vec<_> = (0..64).map(|i| ret_const(&alloc, i)).collect();
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(call(b), i as u64);
            assert!(!b.exit_common().is_null());
        }
        assert_eq!(alloc.inner.borrow().segments.len(), 1);
        let used = alloc.inner.borrow().segments[0].header().used_units.get();
        drop(blocks);
        let after = alloc.inner.borrow().segments[0].header().used_units.get();
        assert_eq!(after, used - 64);
    }
}
