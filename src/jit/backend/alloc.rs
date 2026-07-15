//! A bitmap allocator for JIT code, one instance per `Lua`.
//!
//! Mapping a fresh pair of pages per compiled function is several syscalls and a
//! whole (16 KiB) page of slack for a few hundred bytes of code. This allocator
//! amortizes that over **64 KiB segments**: each segment is dual-mapped once
//! (RW and RX aliases, via [`code::map_dual`]), then sub-allocated in **64-byte
//! units** tracked by a bitmap. Compiling a function copies its encoded words
//! into a free run and invalidates only that range's instruction cache — no
//! per-function syscall.
//!
//! # Metadata lives in the segment
//!
//! A segment's bookkeeping (its bitmap and a little scalar state) sits in the
//! *first* units of the segment itself, as a [`SegHeader`]. Because segments are
//! **64 KiB-aligned** (only the RW alias needs to be — see [`code::map_dual`]), a
//! block's writable pointer masks straight back to its header: `hdr = rw &
//! !(SEG-1)`. That is what lets a [`CodeBlock`] free itself on `Drop` with no back
//! pointer to the allocator and no `Rc` — the header it needs is a mask away, and
//! the segment stays mapped for the allocator's whole life (segments are recycled,
//! never unmapped mid-run), so the mask is always valid.
//!
//! # Header sizing
//!
//! 64 KiB / 64 B = 1024 units, so the bitmap is 1024 bits = 128 bytes. The header
//! is that bitmap plus scalar fields, rounded up to whole units. With `H` header
//! units we need `64·H ≥ 128 + scalars`: `H = 1` and `H = 2` can't even hold the
//! bitmap-plus-scalars, so the minimum is **`H = 3` (192 bytes)** — 128 for the
//! bitmap, up to 64 for scalars. The header's own 3 units are pre-marked
//! allocated, leaving 1021 units (~65 KB) of code per segment.
//!
//! # Placement policy
//!
//! One segment is *primary*: allocations come from it until a request no longer
//! fits. Then a non-primary segment that is less than half used is promoted to
//! primary if the request fits there; otherwise a new segment is mapped. Freed
//! space is returned to its segment's bitmap and reused.

use std::cell::{Cell, RefCell};
use std::io;
use std::ptr::NonNull;

use crate::jit::backend::code::{map_dual, sync_icache};

const SEG_SIZE: usize = 64 * 1024;
const UNIT: usize = 64;
const TOTAL_UNITS: usize = SEG_SIZE / UNIT; // 1024
const BITMAP_WORDS: usize = TOTAL_UNITS / 64; // 16
const HEADER_UNITS: usize = 3; // 192-byte header; see module docs
const DATA_UNITS: usize = TOTAL_UNITS - HEADER_UNITS; // 1021

/// In-segment bookkeeping, laid out at the segment's RW base. Every field is
/// read/written through a shared reference (`&SegHeader`) — hence the `Cell`s —
/// so a `CodeBlock` can free itself without owning the segment.
#[repr(C)]
struct SegHeader {
    /// Executable-alias base of this segment. Code for unit `u` executes at
    /// `rx_base + u*UNIT`. Written once at segment creation, read-only after.
    rx_base: *mut u8,
    /// Data units currently allocated (excludes the header's own units). Drives
    /// the half-used reuse heuristic.
    used_units: Cell<u16>,
    /// One bit per unit, `1` = allocated. The header's units are pre-marked.
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

    /// First-fit: the lowest unit index at which `len` free units run
    /// contiguously, or `None`. Scans only the data region.
    fn find_free(&self, len: usize) -> Option<usize> {
        let mut start = HEADER_UNITS;
        let mut run = 0;
        let mut i = HEADER_UNITS;
        while i < TOTAL_UNITS {
            if self.test(i) {
                run = 0;
                start = i + 1;
            } else {
                run += 1;
                if run == len {
                    return Some(start);
                }
            }
            i += 1;
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

/// One dual-mapped 64 KiB region. Owns the mapping; frees it on `Drop`.
struct Segment {
    /// RW alias base, 64 KiB-aligned and equal to the [`SegHeader`] address.
    rw: NonNull<u8>,
    /// RX alias base (arbitrary alignment).
    rx: NonNull<u8>,
}

impl Segment {
    fn new() -> io::Result<Segment> {
        let (rw, rx) = map_dual(SEG_SIZE, (SEG_SIZE - 1) as u64)?;
        // The mapping is zero-filled, so the bitmap and counters start empty.
        // Initialize the header under exclusive access, before any shared
        // reference to it can exist.
        let h = unsafe { &mut *(rw.as_ptr() as *mut SegHeader) };
        h.rx_base = rx.as_ptr();
        for i in 0..HEADER_UNITS {
            h.set(i);
        }
        Ok(Segment { rw, rx })
    }

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

/// An owning handle to one function's code within a segment.
///
/// Holds the executable entry (to run) and the writable start (to locate the
/// segment header on `Drop`). Freeing masks `rw` to the header and clears the
/// bitmap — the segment is guaranteed still mapped, so no back reference to the
/// allocator is needed.
pub struct CodeBlock {
    rx: NonNull<u8>,
    rw: NonNull<u8>,
    len_units: u16,
}

impl CodeBlock {
    /// Executable entry point. See [`code::Code::entry`] for the transmute
    /// contract; the same applies here.
    pub fn entry(&self) -> *const u8 {
        self.rx.as_ptr()
    }
}

impl Drop for CodeBlock {
    fn drop(&mut self) {
        let rw = self.rw.as_ptr() as usize;
        let base = rw & !(SEG_SIZE - 1);
        let hdr = unsafe { &*(base as *const SegHeader) };
        let start_unit = (rw - base) / UNIT;
        hdr.free(start_unit, self.len_units as usize);
    }
}

/// Per-`Lua` code allocator. Lives off the GC heap (see `lua::OffHeap`) and
/// outlives every `Region`, so a `CodeBlock`'s mask-to-header free is always
/// sound.
pub struct CodeAllocator {
    inner: RefCell<Inner>,
}

struct Inner {
    segments: Vec<Segment>,
    /// Index of the segment allocations currently come from, or `usize::MAX`
    /// before the first segment exists.
    primary: usize,
}

impl Default for CodeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeAllocator {
    pub fn new() -> Self {
        CodeAllocator {
            inner: RefCell::new(Inner {
                segments: Vec::new(),
                primary: usize::MAX,
            }),
        }
    }

    /// Copy `words` into code memory, publish them to the instruction cache, and
    /// return an owning handle. Takes `&self`: mutation is behind the `RefCell`,
    /// so the allocator is reachable through a shared `Context`.
    pub fn alloc(&self, words: &[u32]) -> io::Result<CodeBlock> {
        let len_bytes = words.len() * 4;
        let len_units = len_bytes.div_ceil(UNIT);
        assert!(
            (1..=DATA_UNITS).contains(&len_units),
            "code block of {len_units} units exceeds a {DATA_UNITS}-unit segment"
        );

        let mut inner = self.inner.borrow_mut();
        let (seg_idx, start_unit) = inner.reserve(len_units)?;
        let seg = &inner.segments[seg_idx];
        let off = start_unit * UNIT;
        let rw = unsafe { seg.rw.as_ptr().add(off) };
        let rx = unsafe { seg.header().rx_base.add(off) };

        // The RW alias is always writable, so the code goes in with a plain copy
        // (host is little-endian aarch64, matching the encoded word order). Then
        // make just this range fetchable through the RX alias.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), rw, len_bytes);
            sync_icache(rw, rx, len_bytes);
        }

        Ok(CodeBlock {
            rx: NonNull::new(rx).expect("segment rx offset non-null"),
            rw: NonNull::new(rw).expect("segment rw offset non-null"),
            len_units: len_units as u16,
        })
    }
}

impl Inner {
    /// Reserve `len_units` contiguous units and return `(segment index, start
    /// unit)`, marking them allocated. Tries the primary first; on a miss,
    /// promotes a half-empty segment or maps a new one.
    fn reserve(&mut self, len_units: usize) -> io::Result<(usize, usize)> {
        if self.primary != usize::MAX
            && let Some(u) = self.segments[self.primary].header().find_free(len_units)
        {
            self.segments[self.primary].header().mark(u, len_units);
            return Ok((self.primary, u));
        }

        // Primary is full (or none yet). Scan for a non-primary segment that is
        // under half used and has room. Kept separate from the mutation below so
        // the shared borrow of `segments` ends first.
        let mut found = None;
        for i in 0..self.segments.len() {
            if i == self.primary {
                continue;
            }
            let h = self.segments[i].header();
            if (h.used_units.get() as usize) * 2 < DATA_UNITS
                && let Some(u) = h.find_free(len_units)
            {
                found = Some((i, u));
                break;
            }
        }
        if let Some((i, u)) = found {
            self.segments[i].header().mark(u, len_units);
            self.primary = i;
            return Ok((i, u));
        }

        // Nothing suitable: map a new segment and make it primary.
        let seg = Segment::new()?;
        let u = seg
            .header()
            .find_free(len_units)
            .expect("fresh segment fits");
        seg.header().mark(u, len_units);
        self.segments.push(seg);
        let idx = self.segments.len() - 1;
        self.primary = idx;
        Ok((idx, u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode `mov x0, #imm; ret`, allocate it, and call it — the allocator's
    /// end-to-end contract in one shot.
    fn ret_const(alloc: &CodeAllocator, imm: u16) -> CodeBlock {
        // mov x0, #imm  (MOVZ x0, #imm) ; ret
        let movz = 0xD280_0000u32 | ((imm as u32) << 5);
        let ret = 0xD65F_03C0u32;
        alloc.alloc(&[movz, ret]).expect("alloc")
    }

    fn call(block: &CodeBlock) -> u64 {
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(block.entry()) };
        f()
    }

    #[test]
    fn allocates_and_runs() {
        let alloc = CodeAllocator::new();
        let b = ret_const(&alloc, 42);
        assert_eq!(call(&b), 42);
    }

    /// Many small allocations share one segment, each independently runnable.
    #[test]
    fn packs_many_into_one_segment() {
        let alloc = CodeAllocator::new();
        let blocks: Vec<_> = (0..64).map(|i| ret_const(&alloc, i)).collect();
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(call(b), i as u64);
        }
        // 64 two-word (one-unit) blocks fit far inside a single 1021-unit segment.
        assert_eq!(alloc.inner.borrow().segments.len(), 1);
    }

    /// Freeing returns units to the bitmap; a later allocation reuses them.
    #[test]
    fn free_returns_space() {
        let alloc = CodeAllocator::new();
        let used_after = |a: &CodeAllocator| a.inner.borrow().segments[0].header().used_units.get();

        let b0 = ret_const(&alloc, 1);
        let peak = used_after(&alloc);
        {
            let _b1 = ret_const(&alloc, 2);
            assert!(used_after(&alloc) > peak);
        }
        // `_b1` dropped: its unit is back.
        assert_eq!(used_after(&alloc), peak);
        assert_eq!(call(&b0), 1);
    }

    /// A run larger than one segment's capacity is rejected, not silently wrong.
    #[test]
    #[should_panic(expected = "exceeds")]
    fn oversized_block_panics() {
        let alloc = CodeAllocator::new();
        let words = vec![0xD503_201Fu32; DATA_UNITS * (UNIT / 4) + 4]; // > 1021 units
        let _ = alloc.alloc(&words);
    }
}
