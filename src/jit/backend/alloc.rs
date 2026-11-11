//! Code memory in 64 KiB segments, each mapped once (`code::map_in`) and
//! sub-allocated in 64-byte units by a bitmap in the segment's own first
//! units. A block's writable address masks back to its segment's header, so
//! a `CodeBlock` frees itself without a pointer to the allocator; the
//! allocator outlives every block (regions hold an `Rc` of it). The
//! executable alias is aligned too, and a second bitmap marks where blocks
//! start, so an address in a block finds the block's first word
//! (`block_start`), where a region keeps its pointer.
//!
//! Every segment's executable alias is a slot of the process's one code
//! range (`Range`), shared by every `Lua`: code reaches other code with
//! direct branches wherever the range is, and the text too when the range
//! is near it; a far range branches to the text, and every range to shared
//! libraries (libm), through islands (`branch_target`). Encodings and reach
//! are the target's (`target::code`).

use std::cell::{Cell, RefCell};
use std::io;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard};

use crate::jit::backend::code::{map_in, reserve as reserve_space, unmap};
use crate::jit::backend::target::code::{self as tcode, sync_icache};

pub(crate) const SEG_SIZE: usize = 64 * 1024;
const UNIT: usize = 64;
const TOTAL_UNITS: usize = SEG_SIZE / UNIT;
const BITMAP_WORDS: usize = TOTAL_UNITS / 64;
const HEADER_UNITS: usize = 5;
const DATA_UNITS: usize = TOTAL_UNITS - HEADER_UNITS;

#[repr(C)]
struct SegHeader {
    rx_base: *mut u8,
    used_units: Cell<u16>,
    bitmap: [Cell<u64>; BITMAP_WORDS],
    /// The first unit of each block.
    starts: [Cell<u64>; BITMAP_WORDS],
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
        let w = &self.starts[start / 64];
        w.set(w.get() | 1 << (start % 64));
    }

    fn free(&self, start: usize, len: usize) {
        for i in start..start + len {
            debug_assert!(self.test(i), "double-freed unit {i}");
            self.clear(i);
        }
        self.used_units.set(self.used_units.get() - len as u16);
        let w = &self.starts[start / 64];
        w.set(w.get() & !(1 << (start % 64)));
    }

    /// The first unit of the block holding unit `i`.
    fn start_of(&self, i: usize) -> usize {
        let mut k = i / 64;
        let mut w = self.starts[k].get() & (u64::MAX >> (63 - i % 64));
        while w == 0 {
            k -= 1;
            w = self.starts[k].get();
        }
        k * 64 + 63 - w.leading_zeros() as usize
    }
}

/// The executable address of the first byte of the code block holding
/// `addr`.
///
/// # Safety
/// `addr` lies in a live block's executable alias.
pub(crate) unsafe fn block_start(addr: usize) -> usize {
    let seg = addr & !(SEG_SIZE - 1);
    // SAFETY: the alias maps the header too, readable.
    let h = unsafe { &*(seg as *const SegHeader) };
    seg + h.start_of((addr - seg) / UNIT) * UNIT
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
        unmap(self.rw, self.rx, SEG_SIZE);
        if let Some(r) = range().as_mut() {
            r.give(self.rx.as_ptr() as usize);
        }
    }
}

/// The process's code range: one reservation, made on first use, near the
/// text if any of `RANGE_SIZES` fits in reach, else anywhere (beyond reach
/// under `TCVM_JIT_FAR`, for tests of far code). Never given back.
struct Range {
    base: usize,
    used: Vec<u64>,
    slots: usize,
    #[cfg(test)]
    near: bool,
    /// Islands of targets out of reach: (target, island).
    islands: Vec<(usize, usize)>,
    /// The slot islands are written in: executable and writable alias, next
    /// free byte.
    island_slot: Option<(usize, usize, usize)>,
}

static RANGE: Mutex<Option<Range>> = Mutex::new(None);

fn range() -> MutexGuard<'static, Option<Range>> {
    RANGE.lock().unwrap_or_else(|e| e.into_inner())
}

impl Range {
    fn new() -> io::Result<Range> {
        #[cfg(feature = "tcvm-dev")]
        let far = std::env::var_os("TCVM_JIT_FAR").is_some_and(|v| v != "0");
        #[cfg(not(feature = "tcvm-dev"))]
        let far = false;
        let near = (!far)
            .then(|| {
                tcode::RANGE_SIZES.iter().find_map(|&size| {
                    reserve_space(size, SEG_SIZE, true, false)
                        .ok()
                        .map(|p| (p, size))
                })
            })
            .flatten();
        let (p, size) = match near {
            Some(r) => r,
            None => (
                reserve_space(tcode::RANGE_SIZES[0], SEG_SIZE, false, far)?,
                tcode::RANGE_SIZES[0],
            ),
        };
        let base = p.as_ptr() as usize;
        let slots = size / SEG_SIZE;
        Ok(Range {
            base,
            used: vec![0; slots.div_ceil(64)],
            slots,
            #[cfg(test)]
            near: crate::jit::backend::code::in_reach(base, size),
            islands: Vec::new(),
            island_slot: None,
        })
    }

    /// A free slot's executable address.
    fn take(&mut self) -> Option<usize> {
        let i = (0..self.slots).find(|&i| self.used[i / 64] >> (i % 64) & 1 == 0)?;
        self.used[i / 64] |= 1 << (i % 64);
        Some(self.base + i * SEG_SIZE)
    }

    fn give(&mut self, rx: usize) {
        let i = (rx - self.base) / SEG_SIZE;
        debug_assert!(
            self.used[i / 64] >> (i % 64) & 1 != 0,
            "a free code slot freed"
        );
        self.used[i / 64] &= !(1 << (i % 64));
    }

    fn contains(&self, a: usize) -> bool {
        (self.base..self.base + self.slots * SEG_SIZE).contains(&a)
    }

    /// The island branching on to `target`, made on first use.
    fn island(&mut self, target: usize) -> Option<usize> {
        if let Some(&(_, i)) = self.islands.iter().find(|i| i.0 == target) {
            return Some(i);
        }
        if self
            .island_slot
            .is_none_or(|(_, _, next)| next + tcode::ISLAND > SEG_SIZE)
        {
            let rx = self.take()?;
            let slot = NonNull::new(rx as *mut u8).expect("a slot at 0");
            let Ok(rw) = map_in(slot, SEG_SIZE) else {
                self.give(rx);
                return None;
            };
            self.island_slot = Some((rx, rw.as_ptr() as usize, 0));
        }
        let (rx, rw, next) = self.island_slot.as_mut().expect("an island slot");
        let bytes = tcode::island(target);
        let (at_rx, at_rw) = (*rx + *next, *rw + *next);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at_rw as *mut u8, bytes.len());
            sync_icache(at_rw as *mut u8, at_rx as *mut u8, bytes.len());
        }
        *next += tcode::ISLAND;
        self.islands.push((target, at_rx));
        Some(at_rx)
    }
}

/// Whether the code range is near the text, and its islands (tests).
#[cfg(test)]
pub(crate) fn range_state() -> Option<(bool, usize)> {
    range().as_ref().map(|r| (r.near, r.islands.len()))
}

/// Where a direct branch at `from`, code in the range, goes for `target`: the
/// target itself in reach, else its island (a far range's branch into the
/// text, or any range's into a shared library); none if neither reaches.
pub(crate) fn branch_target(from: usize, target: usize) -> Option<usize> {
    if tcode::reaches(from, target) {
        return Some(target);
    }
    let mut g = range();
    let r = g.as_mut()?;
    if !r.contains(from) {
        return None;
    }
    let i = r.island(target)?;
    tcode::reaches(from, i).then_some(i)
}

/// One region's code in a segment.
pub(crate) struct CodeBlock {
    rx: NonNull<u8>,
    rw: NonNull<u8>,
    len_units: u16,
    unwind: Option<crate::jit::backend::unwind::Fde>,
}

impl CodeBlock {
    /// The executable address of the block's first byte.
    pub(crate) fn rx(&self) -> *const u8 {
        self.rx.as_ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.len_units as usize * UNIT
    }

    fn header(&self) -> &SegHeader {
        let base = self.rw.as_ptr() as usize & !(SEG_SIZE - 1);
        unsafe { &*(base as *const SegHeader) }
    }

    /// Copy `code` in at the start of the block and publish it.
    pub(crate) fn write(&self, code: &[u8]) {
        assert!(code.len() <= self.len());
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.rw.as_ptr(), code.len());
            sync_icache(self.rw.as_ptr(), self.rx.as_ptr(), code.len());
        }
    }

    /// Overwrite the bytes at `off` and publish them.
    fn patch<const N: usize>(&self, off: usize, f: impl FnOnce(&mut [u8; N])) {
        assert!(off + N <= self.len());
        unsafe {
            f(&mut *self.rw.as_ptr().add(off).cast::<[u8; N]>());
            sync_icache(self.rw.as_ptr().add(off), self.rx.as_ptr().add(off), N);
        }
    }

    /// Tell the unwinder the block's calls are made with a frame of
    /// `frame` bytes open (`unwind::Fde`).
    pub(crate) fn describe_frame(&mut self, frame: u32) {
        self.unwind = Some(crate::jit::backend::unwind::Fde::register(
            self.rx() as usize,
            self.len(),
            frame,
        ));
    }

    /// Point the direct branch or call at byte `off` at `target` (or its
    /// island) and publish it.
    pub(crate) fn repoint_branch(&self, off: usize, target: usize) {
        let at = self.rx() as usize + off;
        let target = branch_target(at, target).expect("a branch target out of reach");
        self.patch(off, |b| tcode::retarget(b, at, target));
    }

    /// Overwrite the branch at byte `off` with a call of `target` (or its
    /// island) and publish it.
    pub(crate) fn patch_call(&self, off: usize, target: usize) {
        let at = self.rx() as usize + off;
        let target = branch_target(at, target).expect("a branch target out of reach");
        self.patch(off, |b| *b = tcode::call(at, target));
    }

    /// Overwrite the patch point at byte `off` with a jump to byte `to` of
    /// the block and publish it.
    pub(crate) fn patch_jump(&self, off: usize, to: usize) {
        assert!(to < self.len());
        let rx = self.rx() as usize;
        self.patch(off, |b| *b = tcode::jump(rx + off, rx + to));
    }

    /// Overwrite the 64-bit word at byte `off` and publish it.
    pub(crate) fn patch_u64(&self, off: usize, v: u64) {
        assert!(off.is_multiple_of(8));
        self.patch(off, |b| *b = v.to_ne_bytes());
    }
}

impl Drop for CodeBlock {
    fn drop(&mut self) {
        // Out of the unwinder's tables before the units are reused.
        drop(self.unwind.take());
        let rw = self.rw.as_ptr() as usize;
        let base = rw & !(SEG_SIZE - 1);
        let start = (rw - base) / UNIT;
        self.header().free(start, self.len_units as usize);
    }
}

pub(crate) struct CodeAllocator {
    inner: RefCell<Inner>,
}

/// Why `reserve` found no room.
#[derive(Debug)]
pub(crate) enum ReserveError {
    /// More than one segment holds.
    TooLarge,
    /// No segment has room and no new one maps: none will.
    Exhausted(io::Error),
}

impl std::fmt::Display for ReserveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReserveError::TooLarge => write!(f, "larger than a code segment"),
            ReserveError::Exhausted(e) => write!(f, "code memory exhausted: {e}"),
        }
    }
}

struct Inner {
    segments: Vec<Segment>,
    primary: usize,
    /// No segment could be had once (the code range full, or mapping
    /// failed): refuse from then on.
    exhausted: bool,
    /// Segments it may map (tests).
    max_segments: usize,
}

impl CodeAllocator {
    pub(crate) fn new() -> Self {
        CodeAllocator {
            inner: RefCell::new(Inner {
                segments: Vec::new(),
                primary: usize::MAX,
                exhausted: false,
                max_segments: usize::MAX,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn limit_segments(&self, n: usize) {
        self.inner.borrow_mut().max_segments = n;
    }

    /// Room for `len` bytes, uninitialized.
    pub(crate) fn reserve(&self, len: usize) -> Result<CodeBlock, ReserveError> {
        let units = len.max(1).div_ceil(UNIT);
        if units > DATA_UNITS {
            return Err(ReserveError::TooLarge);
        }
        let mut inner = self.inner.borrow_mut();
        let (seg, start) = inner.reserve(units)?;
        let s = &inner.segments[seg];
        let off = start * UNIT;
        let rw = unsafe { s.rw.as_ptr().add(off) };
        let rx = unsafe { s.header().rx_base.add(off) };
        Ok(CodeBlock {
            rx: NonNull::new(rx).expect("segment rx"),
            rw: NonNull::new(rw).expect("segment rw"),
            len_units: units as u16,
            unwind: None,
        })
    }
}

/// A slot of the range, mapped: its writable and executable aliases.
fn map_segment() -> io::Result<(NonNull<u8>, NonNull<u8>)> {
    let rx = {
        let mut g = range();
        if g.is_none() {
            *g = Some(Range::new()?);
        }
        let r = g.as_mut().expect("the code range");
        r.take()
            .ok_or_else(|| io::Error::other("the code range is full"))?
    };
    let rx = NonNull::new(rx as *mut u8).expect("a slot at 0");
    match map_in(rx, SEG_SIZE) {
        Ok(rw) => Ok((rw, rx)),
        Err(e) => {
            if let Some(r) = range().as_mut() {
                r.give(rx.as_ptr() as usize);
            }
            Err(e)
        }
    }
}

impl Inner {
    fn new_segment(&mut self) -> Result<usize, ReserveError> {
        if self.exhausted {
            return Err(ReserveError::Exhausted(io::Error::other(
                "an earlier mapping failed",
            )));
        }
        let mapped = if self.segments.len() < self.max_segments {
            map_segment()
        } else {
            Err(io::Error::other("segment limit"))
        };
        let (rw, rx) = match mapped {
            Ok(p) => p,
            Err(e) => {
                self.exhausted = true;
                return Err(ReserveError::Exhausted(e));
            }
        };
        // `block_start` masks executable addresses to their segment.
        assert!(
            rx.as_ptr() as usize & (SEG_SIZE - 1) == 0,
            "an unaligned code segment"
        );
        let h = unsafe { &mut *(rw.as_ptr() as *mut SegHeader) };
        h.rx_base = rx.as_ptr();
        for i in 0..HEADER_UNITS {
            h.set(i);
        }
        self.segments.push(Segment { rw, rx });
        Ok(self.segments.len() - 1)
    }

    fn reserve(&mut self, units: usize) -> Result<(usize, usize), ReserveError> {
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
        let i = self.new_segment()?;
        let u = self.segments[i]
            .header()
            .find_free(units)
            .ok_or(ReserveError::TooLarge)?;
        self.segments[i].header().mark(u, units);
        self.primary = i;
        Ok((i, u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ret_const(alloc: &CodeAllocator, imm: u16) -> CodeBlock {
        let code = tcode::ret_imm(imm);
        let b = alloc.reserve(code.len()).expect("reserve");
        b.write(&code);
        b
    }

    fn call(b: &CodeBlock) -> u64 {
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(b.rx()) };
        f()
    }

    #[test]
    fn allocates_runs_and_frees() {
        let alloc = CodeAllocator::new();
        let blocks: Vec<_> = (0..64).map(|i| ret_const(&alloc, i)).collect();
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(call(b), i as u64);
        }
        assert_eq!(alloc.inner.borrow().segments.len(), 1);
        let used = alloc.inner.borrow().segments[0].header().used_units.get();
        drop(blocks);
        let after = alloc.inner.borrow().segments[0].header().used_units.get();
        assert_eq!(after, used - 64);
    }
}
