//! The memory the collector manages, after LuaJIT 3.0's GC design (arenas there; chunks here,
//! as [`Arena`](crate::dmm::Arena) is the collector). A chunk is `CHUNK_SIZE` bytes aligned to
//! its size, so masking any object pointer finds its metadata: a header and two bitmaps with one
//! bit per 16-byte cell. A block is an object's run of cells, typed by its first cell:
//!
//! | block | mark | cell                  |
//! |-------|------|-----------------------|
//! | 0     | 0    | inside a block        |
//! | 0     | 1    | starts a free block   |
//! | 1     | 0    | starts a white block  |
//! | 1     | 1    | starts a black block  |
//!
//! Allocation bumps a pointer through a free run and sets the block bit; sweeping is two word
//! operations per 64 cells and never reads a dead object unless it needs dropping. An object too
//! big for a chunk gets a huge chunk of its own, with the same metadata and the object in its
//! first data cell.

use core::cell::{Cell, UnsafeCell};
use core::ptr::{self, NonNull};
use std::alloc::{self, Layout};
use std::vec::Vec;

use crate::dmm::{metrics::Metrics, types::GcBox};

pub(crate) const CELL: usize = 16;
const CHUNK_SIZE: usize = 1 << 18;
const CELLS: usize = CHUNK_SIZE / CELL;
const WORDS: usize = CELLS / 64;
/// Bitmap words covering the metadata's own cells, which hold the header instead.
const META_WORDS: usize = WORDS / 64;
const FIRST_CELL: usize = META_WORDS * 64;
/// Objects bigger than this get a huge chunk.
const HUGE: usize = CHUNK_SIZE / 8;

#[derive(Copy, Clone, Eq, PartialEq)]
enum Kind {
    /// Traced objects without drop glue.
    Plain,
    /// Untraced objects without drop glue, which marking never reads.
    Leaf,
    /// Objects with drop glue.
    Drop,
    Huge,
    /// A huge chunk whose object died, freed when the sweep ends.
    Dead,
}

#[repr(C)]
struct Header {
    /// Bytes of the blocks allocated here, as of the last run retired from this chunk.
    allocated: Cell<usize>,
    /// Bytes of the blocks marked this cycle.
    marked: Cell<usize>,
    /// The whole allocation: `CHUNK_SIZE`, or more for a huge chunk.
    size: usize,
    kind: Cell<Kind>,
}

#[repr(C)]
struct Meta {
    header: Header,
    block: [Cell<u64>; WORDS - META_WORDS],
    _mark_header: [u64; META_WORDS],
    mark: [Cell<u64>; WORDS - META_WORDS],
}

const _: () = assert!(size_of::<Header>() == META_WORDS * 8);
const _: () = assert!(size_of::<Meta>() == FIRST_CELL * CELL);

#[inline(always)]
fn meta<'a>(p: *const u8) -> &'a Meta {
    // SAFETY (for callers): `p` points into a live chunk, whose metadata starts at its base.
    unsafe { &*p.map_addr(|a| a & !(CHUNK_SIZE - 1)).cast::<Meta>() }
}

#[inline(always)]
fn cell_of(p: *const u8) -> usize {
    (p.addr() & (CHUNK_SIZE - 1)) / CELL
}

/// The word index and bit of `cell` in a bitmap.
#[inline(always)]
fn bit(cell: usize) -> (usize, u64) {
    (cell / 64 - META_WORDS, 1 << (cell % 64))
}

#[inline(always)]
fn base(meta: &Meta) -> *mut u8 {
    ptr::from_ref(meta).cast::<u8>().cast_mut()
}

/// Whether the object at `p` is marked.
///
/// # Safety
/// `p` must be a live object's start.
#[inline(always)]
pub(crate) unsafe fn is_marked(p: GcBox) -> bool {
    let p = p.as_ptr();
    let (w, b) = bit(cell_of(p));
    meta(p).mark[w].get() & b != 0
}

/// Mark the object at `p`, which is `bytes` big.
///
/// # Safety
/// `p` must be an unmarked live object's start.
#[inline(always)]
pub(crate) unsafe fn set_marked(p: GcBox, bytes: usize) {
    let p = p.as_ptr();
    let meta = meta(p);
    let (w, b) = bit(cell_of(p));
    meta.mark[w].update(|m| m | b);
    meta.header.marked.update(|n| n + bytes);
}

#[derive(Copy, Clone)]
struct Sweep {
    /// The next chunk to sweep.
    next: usize,
    /// Chunks from here on were made while sweeping; their objects are all new.
    end: usize,
}

/// Where objects of one kind are bump-allocated.
struct Space {
    kind: Kind,
    /// The run being bump-allocated: `top..limit` is free, `run_start..top` allocated since the
    /// run began. All null when there is none.
    top: Cell<*mut u8>,
    limit: Cell<*mut u8>,
    run_start: Cell<*mut u8>,
    /// Where the search for the next run resumes.
    scan_chunk: Cell<usize>,
    scan_cell: Cell<usize>,
}

impl Space {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            top: Cell::new(ptr::null_mut()),
            limit: Cell::new(ptr::null_mut()),
            run_start: Cell::new(ptr::null_mut()),
            scan_chunk: Cell::new(0),
            scan_cell: Cell::new(FIRST_CELL),
        }
    }

    #[inline(always)]
    fn bump(&self, size: usize) -> Option<NonNull<u8>> {
        let top = self.top.get();
        if size > self.limit.get().addr() - top.addr() {
            return None;
        }
        self.top.set(top.wrapping_add(size));
        let (w, b) = bit(cell_of(top));
        meta(top).block[w].update(|m| m | b);
        // SAFETY: `top` is in a chunk.
        Some(unsafe { NonNull::new_unchecked(top) })
    }

    /// Bump-allocate from `start..end` of `meta`, a free run.
    fn start_run(&self, meta: &Meta, start: usize, end: usize) {
        // Clear the free-block starts so the run reads as one block's inside until allocated.
        clear_range(&meta.mark, start, end);
        let base = base(meta);
        // SAFETY: both are within the chunk.
        let (start, end) = unsafe { (base.add(start * CELL), base.add(end * CELL)) };
        self.run_start.set(start);
        self.top.set(start);
        self.limit.set(end);
    }

    /// End the current run, leaving its unused tail a free block.
    fn retire_run(&self) {
        let top = self.top.get();
        if top.is_null() {
            return;
        }
        let meta = meta(self.run_start.get());
        if top != self.limit.get() {
            let (w, b) = bit(cell_of(top));
            meta.mark[w].update(|m| m | b);
        }
        let used = top.addr() - self.run_start.get().addr();
        meta.header.allocated.update(|n| n + used);
        self.top.set(ptr::null_mut());
        self.limit.set(ptr::null_mut());
        self.run_start.set(ptr::null_mut());
    }

    fn rescan(&self) {
        self.scan_chunk.set(0);
        self.scan_cell.set(FIRST_CELL);
    }
}

pub(crate) struct Heap {
    chunks: UnsafeCell<Vec<NonNull<Meta>>>,
    plain: Space,
    leaf: Space,
    drop: Space,
    sweep: Cell<Option<Sweep>>,
}

impl Heap {
    pub(crate) fn new() -> Self {
        Self {
            chunks: UnsafeCell::new(Vec::new()),
            plain: Space::new(Kind::Plain),
            leaf: Space::new(Kind::Leaf),
            drop: Space::new(Kind::Drop),
            sweep: Cell::new(None),
        }
    }

    /// # Safety
    /// No other reference to the list may be live; `Heap` is `!Sync`, so a method using the
    /// result within its own body is enough.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn chunks(&self) -> &mut Vec<NonNull<Meta>> {
        unsafe { &mut *self.chunks.get() }
    }

    /// Allocate `size` bytes, a multiple of `CELL`, as a white block for an object of a type
    /// that `needs_drop` and `needs_trace`.
    #[inline(always)]
    pub(crate) fn alloc(
        &self,
        size: usize,
        needs_drop: bool,
        needs_trace: bool,
        metrics: &Metrics,
    ) -> NonNull<u8> {
        debug_assert!(size.is_multiple_of(CELL) && size > 0);
        if size > HUGE {
            return self.alloc_huge(size);
        }
        let space = match (needs_drop, needs_trace) {
            (true, _) => &self.drop,
            (false, true) => &self.plain,
            (false, false) => &self.leaf,
        };
        match space.bump(size) {
            Some(p) => p,
            None => self.alloc_slow(space, size, metrics),
        }
    }

    #[cold]
    #[inline(never)]
    fn alloc_slow(&self, space: &Space, size: usize, metrics: &Metrics) -> NonNull<u8> {
        space.retire_run();
        match self.next_run(space, size / CELL, metrics) {
            Some((meta, start, end)) => space.start_run(meta, start, end),
            None => {
                let meta = self.new_chunk(CHUNK_SIZE, space.kind);
                space.scan_chunk.set(unsafe { self.chunks() }.len() - 1);
                space.scan_cell.set(CELLS);
                space.start_run(meta, FIRST_CELL, CELLS);
            }
        }
        space.bump(size).unwrap()
    }

    #[cold]
    #[inline(never)]
    fn alloc_huge(&self, size: usize) -> NonNull<u8> {
        let meta = self.new_chunk(FIRST_CELL * CELL + size, Kind::Huge);
        let (w, b) = bit(FIRST_CELL);
        meta.block[w].set(b);
        meta.header.allocated.set(size);
        // SAFETY: the object is the chunk's data.
        unsafe { NonNull::new_unchecked(base(meta).add(FIRST_CELL * CELL)) }
    }

    fn new_chunk(&self, size: usize, kind: Kind) -> &Meta {
        let layout = Layout::from_size_align(size, CHUNK_SIZE).expect("allocation too large");
        // SAFETY: `layout` is not zero-sized; only the metadata needs zeroing, as allocation
        // writes every object before it is read.
        let meta = unsafe {
            let p = alloc::alloc(layout);
            if p.is_null() {
                alloc::handle_alloc_error(layout);
            }
            p.write_bytes(0, size_of::<Meta>());
            let meta = p.cast::<Meta>();
            ptr::addr_of_mut!((*meta).header).write(Header {
                allocated: Cell::new(0),
                marked: Cell::new(0),
                size,
                kind: Cell::new(kind),
            });
            NonNull::new_unchecked(meta)
        };
        unsafe { self.chunks() }.push(meta);
        // SAFETY: just made; freed only by `finish_sweep` or `Drop`.
        unsafe { meta.as_ref() }
    }

    /// The next free run of at least `cells` cells in one of `space`'s chunks that may be
    /// allocated into, sweeping chunks on the way if a sweep is under way.
    fn next_run(
        &self,
        space: &Space,
        cells: usize,
        metrics: &Metrics,
    ) -> Option<(&Meta, usize, usize)> {
        loop {
            let i = space.scan_chunk.get();
            let chunk = *unsafe { self.chunks() }.get(i)?;
            if let Some(sweep) = self.sweep.get()
                && i >= sweep.next
                && i < sweep.end
            {
                // Sweep up to `i`; the other space may have swept past it already.
                while self.sweep.get().is_some_and(|s| s.next <= i) {
                    self.sweep_next(metrics);
                }
                continue;
            }
            // SAFETY: chunks in the list are live.
            let meta = unsafe { chunk.as_ref() };
            if meta.header.kind.get() == space.kind
                && let Some((start, end)) = find_run(meta, space.scan_cell.get(), cells)
            {
                space.scan_cell.set(end);
                return Some((meta, start, end));
            }
            space.scan_chunk.set(i + 1);
            space.scan_cell.set(FIRST_CELL);
        }
    }

    /// Mark `p` and return its size and whether it needs tracing, reading only metadata (LuaJIT
    /// 3.0's marking): the size is the cells up to the next block or free start, or up to its
    /// space's run top when it is the last object in the run (whose tail has no bits set); a huge
    /// object's is in its chunk header. Only leaf chunks hold objects that need no tracing.
    ///
    /// # Safety
    /// `p` must be an unmarked live object's start.
    #[inline]
    pub(crate) unsafe fn mark(&self, p: GcBox) -> (usize, bool) {
        let p = p.as_ptr();
        let meta = meta(p);
        let kind = meta.header.kind.get();
        let cell = cell_of(p);
        let size = if kind == Kind::Huge {
            meta.header.allocated.get()
        } else {
            let mut end = next_bit(meta, cell + 1, |b, m| b | m).unwrap_or(CELLS);
            let top = self.space(kind).top.get();
            if top.addr() & !(CHUNK_SIZE - 1) == base(meta).addr() && cell < cell_of(top) {
                end = end.min(cell_of(top));
            }
            (end - cell) * CELL
        };
        debug_assert_eq!(size, unsafe { GcBox::from_ptr(p) }.size());
        let (w, b) = bit(cell);
        meta.mark[w].update(|m| m | b);
        meta.header.marked.update(|n| n + size);
        (size, kind != Kind::Leaf)
    }

    fn space(&self, kind: Kind) -> &Space {
        match kind {
            Kind::Plain => &self.plain,
            Kind::Leaf => &self.leaf,
            Kind::Drop => &self.drop,
            Kind::Huge | Kind::Dead => unreachable!(),
        }
    }

    /// Begin sweeping every chunk that exists now. Allocation from here on only uses chunks
    /// already swept, or made after this.
    pub(crate) fn start_sweep(&self) {
        for space in [&self.plain, &self.leaf, &self.drop] {
            space.retire_run();
            space.rescan();
        }
        self.sweep.set(Some(Sweep {
            next: 0,
            end: unsafe { self.chunks() }.len(),
        }));
    }

    /// Sweep the next chunk; false once every chunk is swept.
    pub(crate) fn sweep_next(&self, metrics: &Metrics) -> bool {
        let Some(mut sweep) = self.sweep.get() else {
            return false;
        };
        if sweep.next == sweep.end {
            return false;
        }
        // SAFETY: chunks in the list are live.
        let meta = unsafe { self.chunks()[sweep.next].as_ref() };
        sweep_chunk(meta, metrics);
        sweep.next += 1;
        self.sweep.set(Some(sweep));
        true
    }

    /// End the sweep once `sweep_next` is done: free the dead huge chunks.
    pub(crate) fn finish_sweep(&self) {
        debug_assert!(self.sweep.get().is_some_and(|s| s.next == s.end));
        self.sweep.set(None);
        unsafe { self.chunks() }.retain(|chunk| {
            // SAFETY: chunks in the list are live until freed here.
            let meta = unsafe { chunk.as_ref() };
            if meta.header.kind.get() != Kind::Dead {
                return true;
            }
            unsafe { free_chunk(*chunk) };
            false
        });
        for space in [&self.plain, &self.leaf, &self.drop] {
            space.rescan();
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        for &chunk in self.chunks.get_mut().iter() {
            // SAFETY: the heap owns its chunks; every block holds an initialized object.
            unsafe {
                if matches!(chunk.as_ref().header.kind.get(), Kind::Drop | Kind::Huge) {
                    for_each_bit(chunk.as_ref(), |b, _| b, |p| drop_object(p, None));
                }
                free_chunk(chunk);
            }
        }
    }
}

/// # Safety
/// `chunk` must be live and unreferenced.
unsafe fn free_chunk(chunk: NonNull<Meta>) {
    unsafe {
        let size = chunk.as_ref().header.size;
        alloc::dealloc(
            chunk.as_ptr().cast(),
            Layout::from_size_align_unchecked(size, CHUNK_SIZE),
        );
    }
}

/// Drop the object at `p` unless it was dropped already.
///
/// # Safety
/// `p` must start a block holding an initialized object nothing will use again.
unsafe fn drop_object(p: *mut u8, metrics: Option<&Metrics>) {
    // SAFETY: as the caller promises.
    unsafe {
        let mut gc_box = GcBox::from_ptr(p);
        if gc_box.header().is_live() {
            let size = gc_box.size();
            gc_box.drop_in_place();
            if let Some(metrics) = metrics {
                metrics.mark_gc_dropped(size);
            }
        }
    }
}

/// Free the white blocks of `meta` and turn its black ones white.
fn sweep_chunk(meta: &Meta, metrics: &Metrics) {
    let header = &meta.header;
    match header.kind.get() {
        kind @ (Kind::Plain | Kind::Leaf | Kind::Drop) => {
            let base = base(meta);
            for w in 0..WORDS - META_WORDS {
                let (b, m) = (meta.block[w].get(), meta.mark[w].get());
                if kind == Kind::Drop {
                    let mut dead = b & !m;
                    while dead != 0 {
                        let cell = (w + META_WORDS) * 64 + dead.trailing_zeros() as usize;
                        // SAFETY: a white block after marking holds an unreachable object.
                        unsafe { drop_object(base.add(cell * CELL), Some(metrics)) };
                        dead &= dead - 1;
                    }
                }
                meta.block[w].set(b & m);
                meta.mark[w].set(b ^ m);
            }
            let (allocated, marked) = (header.allocated.get(), header.marked.get());
            metrics.mark_gc_freed(allocated - marked);
            metrics.mark_gc_remembered(marked);
            header.allocated.set(marked);
        }
        Kind::Huge => {
            let (w, b) = bit(FIRST_CELL);
            let size = header.allocated.get();
            if meta.mark[w].get() & b != 0 {
                meta.mark[w].set(0);
                metrics.mark_gc_remembered(size);
            } else {
                // SAFETY: an unmarked huge object is unreachable.
                unsafe { drop_object(base(meta).add(FIRST_CELL * CELL), Some(metrics)) };
                meta.block[w].set(0);
                metrics.mark_gc_freed(size);
                header.kind.set(Kind::Dead);
            }
        }
        Kind::Dead => unreachable!(),
    }
    header.marked.set(0);
}

/// Call `f` on the start of every cell of `meta` where `select(block, mark)` has a bit set.
fn for_each_bit(meta: &Meta, select: impl Fn(u64, u64) -> u64, mut f: impl FnMut(*mut u8)) {
    let base = base(meta);
    for w in 0..WORDS - META_WORDS {
        let mut bits = select(meta.block[w].get(), meta.mark[w].get());
        while bits != 0 {
            let cell = (w + META_WORDS) * 64 + bits.trailing_zeros() as usize;
            // SAFETY: within the chunk.
            f(unsafe { base.add(cell * CELL) });
            bits &= bits - 1;
        }
    }
}

/// The first cell at or after `from` where `select(block, mark)` has a bit set.
fn next_bit(meta: &Meta, from: usize, select: impl Fn(u64, u64) -> u64) -> Option<usize> {
    if from >= CELLS {
        return None;
    }
    let (mut w, _) = bit(from);
    let mut bits = select(meta.block[w].get(), meta.mark[w].get()) & (!0 << (from % 64));
    loop {
        if bits != 0 {
            return Some((w + META_WORDS) * 64 + bits.trailing_zeros() as usize);
        }
        w += 1;
        if w == WORDS - META_WORDS {
            return None;
        }
        bits = select(meta.block[w].get(), meta.mark[w].get());
    }
}

/// The first free run of at least `cells` cells at or after `from`: from a free-block start to
/// the next block start, merging the free blocks between.
fn find_run(meta: &Meta, from: usize, cells: usize) -> Option<(usize, usize)> {
    let mut from = from;
    loop {
        let start = next_bit(meta, from, |b, m| m & !b)?;
        let end = next_bit(meta, start + 1, |b, _| b).unwrap_or(CELLS);
        if end - start >= cells {
            return Some((start, end));
        }
        from = end;
    }
}

/// Clear the bits of `start..end` in `map`.
fn clear_range(map: &[Cell<u64>], start: usize, end: usize) {
    let mut cell = start;
    while cell < end {
        let (w, _) = bit(cell);
        let hi = (cell / 64 + 1) * 64;
        let lo_bit = cell % 64;
        let n = end.min(hi) - cell;
        let mask = if n == 64 { !0 } else { ((1u64 << n) - 1) << lo_bit };
        map[w].update(|m| m & !mask);
        cell += n;
    }
}
