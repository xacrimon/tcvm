//! The memory the collector manages: a single-threaded, non-moving Immix (Blackburn & McKinley,
//! 2008) after mmtk-core's `ImmixSpace` and `ImmixAllocator`. Chunks of `CHUNK` bytes, aligned to
//! their size, are cut into blocks of lines of cells; a chunk's first blocks hold the metadata of
//! all of them, found by masking an object's address:
//!
//! - a byte per line: 0 if free, else the `epoch` of the marking that last found it live, so lines
//!   never need clearing before marking (mmtk's line mark states);
//! - two bits per cell: the mark bit, and an `AUX` bit that starts an object in drop blocks and
//!   flags one larger than a line in leaf blocks.
//!
//! Objects are bump-allocated through holes, runs of free lines. Marking marks the lines of the
//! objects it reaches, and the sweep frees every line no marking reached without visiting the
//! objects in it, except to drop them. Objects larger than `HUGE` are mapped on their own, aligned
//! like a chunk, with their metadata in a table keyed by address: offset 0 of a chunk holds
//! metadata, so an aligned object pointer is a huge one.

use core::cell::{Cell, UnsafeCell};
use core::ptr::{self, NonNull};
use std::alloc::{self, Layout};
use std::vec::Vec;

use hashbrown::HashTable;

use crate::dmm::{metrics::Metrics, types::GcBox};

pub(crate) const CELL: usize = 16;
const LINE: usize = 256;
const BLOCK: usize = 1 << 15;
const CHUNK: usize = 1 << 22;
const LINES: usize = BLOCK / LINE;
const BLOCKS: usize = CHUNK / BLOCK;
/// Bitmap words per block, at two bits per cell.
const WORDS: usize = BLOCK / CELL * 2 / 64;
/// Blocks at the start of a chunk taken by its metadata.
const META_BLOCKS: usize = size_of::<Chunk>().div_ceil(BLOCK);
const DATA_BLOCKS: u128 = !0 << META_BLOCKS;
/// Objects bigger than this are huge: half a block, as in mmtk.
const HUGE: usize = BLOCK / 2;

const MARK: u64 = 0x5555_5555_5555_5555;
const AUX: u64 = MARK << 1;

const _: () = assert!(BLOCKS == 128 && META_BLOCKS < BLOCKS);
const _: () = assert!(LINES <= u16::MAX as usize);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
enum Kind {
    /// Traced objects without drop glue; zeroed metadata reads as this.
    Plain,
    /// Untraced objects without drop glue, which marking never reads.
    Leaf,
    /// Objects with drop glue.
    Drop,
}

#[repr(C)]
struct BlockInfo {
    /// The next block in its space's `recyclable` list.
    next: Cell<*mut u8>,
    /// Lines charged to allocation since the block was last swept, plus the ones it kept then.
    held: Cell<u16>,
    kind: Cell<Kind>,
    /// Swept this cycle, when equal to `Heap::swept`.
    swept: Cell<bool>,
}

#[repr(C, align(128))]
struct Lines([Cell<u8>; BLOCKS * LINES]);

#[repr(C, align(128))]
struct Bits([Cell<u64>; BLOCKS * WORDS]);

/// A chunk's metadata, at its start. Entries for the metadata's own blocks go unused.
#[repr(C)]
struct Chunk {
    /// Free blocks, one bit each.
    free: Cell<u128>,
    blocks: [BlockInfo; BLOCKS],
    lines: Lines,
    bits: Bits,
}

#[inline(always)]
fn chunk_of<'a>(p: *const u8) -> &'a Chunk {
    // SAFETY (for callers): `p` points into a live chunk, whose metadata starts at its base.
    unsafe { &*p.map_addr(|a| a & !(CHUNK - 1)).cast::<Chunk>() }
}

#[inline(always)]
fn block_index(p: *const u8) -> usize {
    (p.addr() / BLOCK) % BLOCKS
}

/// The index of `p`'s line in its chunk.
#[inline(always)]
fn line_index(p: *const u8) -> usize {
    (p.addr() % CHUNK) / LINE
}

/// The index of the bitmap word holding `p`'s cell in its chunk, and its mark bit.
#[inline(always)]
fn mark_bit(p: *const u8) -> (usize, u64) {
    let cell = (p.addr() % CHUNK) / CELL;
    (cell / 32, 1 << (cell % 32 * 2))
}

#[inline(always)]
fn block_start(chunk: &Chunk, b: usize) -> *mut u8 {
    ptr::from_ref(chunk)
        .cast::<u8>()
        .cast_mut()
        .wrapping_add(b * BLOCK)
}

#[inline(always)]
fn block_lines(chunk: &Chunk, b: usize) -> &[Cell<u8>; LINES] {
    chunk.lines.0[b * LINES..][..LINES].try_into().unwrap()
}

#[inline(always)]
fn block_bits(chunk: &Chunk, b: usize) -> &[Cell<u64>; WORDS] {
    chunk.bits.0[b * WORDS..][..WORDS].try_into().unwrap()
}

/// Whether the object at `p` is huge rather than in a chunk.
#[inline(always)]
fn is_huge(p: *const u8) -> bool {
    p.addr().is_multiple_of(CHUNK)
}

/// Whether `p` is a huge object rather than one in a chunk.
#[inline(always)]
pub(crate) fn is_huge_box(p: GcBox) -> bool {
    is_huge(p.as_ptr())
}

/// A huge object's metadata (mmtk's large object space). Its gray bit is the one in its own
/// header, as for any object.
struct Huge {
    ptr: NonNull<u8>,
    size: usize,
    marked: Cell<bool>,
    needs_drop: bool,
    needs_trace: bool,
}

#[inline(always)]
fn huge_hash(p: *const u8) -> u64 {
    // Huge objects' addresses differ only above the chunk bits.
    ((p.addr() / CHUNK) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// # Safety
/// `huge` must be unreachable.
unsafe fn free_huge(huge: &Huge) {
    unsafe {
        if huge.needs_drop {
            drop_object(huge.ptr.as_ptr());
        }
        unmap(huge.ptr.as_ptr(), huge.size);
    }
}

/// Map `size` bytes, zeroed and aligned to `CHUNK`, from the OS. Only `size` rounded up to whole
/// pages stays mapped; the slack taken to align it is unmapped again.
fn map(size: usize) -> NonNull<u8> {
    let page = page_size();
    let len = size.next_multiple_of(page);
    let span = len.checked_add(CHUNK - page).expect("allocation too large");
    // SAFETY: a new anonymous mapping, trimmed to an aligned `len` bytes.
    unsafe {
        let p = libc::mmap(
            ptr::null_mut(),
            span,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        if p == libc::MAP_FAILED {
            alloc::handle_alloc_error(Layout::from_size_align_unchecked(size, CHUNK));
        }
        let p = p.cast::<u8>();
        let start = p.map_addr(|a| a.next_multiple_of(CHUNK));
        let head = start.addr() - p.addr();
        if head != 0 {
            libc::munmap(p.cast(), head);
        }
        if span - head != len {
            libc::munmap(start.add(len).cast(), span - head - len);
        }
        NonNull::new_unchecked(start)
    }
}

/// # Safety
/// `p` and `size` must be those of a `map`, and nothing in it used again.
unsafe fn unmap(p: *mut u8, size: usize) {
    unsafe { libc::munmap(p.cast(), size.next_multiple_of(page_size())) };
}

pub(crate) fn page_size() -> usize {
    // SAFETY: no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Where objects of one kind are bump-allocated (mmtk's `ImmixAllocator`).
struct Space {
    kind: Kind,
    /// The hole being bump-allocated through: `cursor..limit`. Both null when there is none.
    cursor: Cell<*mut u8>,
    limit: Cell<*mut u8>,
    /// The block the next hole is looked for in, from line `hole_line` on; null when none.
    hole_block: Cell<*mut u8>,
    hole_line: Cell<usize>,
    /// Where objects larger than a line that missed the current hole go, in free blocks only, so
    /// they don't make it give up the rest of a hole.
    big_cursor: Cell<*mut u8>,
    big_limit: Cell<*mut u8>,
    /// Swept blocks of this kind with free lines, linked through `BlockInfo::next`.
    recyclable: Cell<*mut u8>,
}

impl Space {
    const fn new(kind: Kind) -> Self {
        Self {
            kind,
            cursor: Cell::new(ptr::null_mut()),
            limit: Cell::new(ptr::null_mut()),
            hole_block: Cell::new(ptr::null_mut()),
            hole_line: Cell::new(0),
            big_cursor: Cell::new(ptr::null_mut()),
            big_limit: Cell::new(ptr::null_mut()),
            recyclable: Cell::new(ptr::null_mut()),
        }
    }

    /// Give up every block this space allocates in or from, for the sweep to reclaim.
    fn retire(&self) {
        self.cursor.set(ptr::null_mut());
        self.limit.set(ptr::null_mut());
        self.hole_block.set(ptr::null_mut());
        self.big_cursor.set(ptr::null_mut());
        self.big_limit.set(ptr::null_mut());
        self.recyclable.set(ptr::null_mut());
    }
}

/// The next block to sweep.
#[derive(Copy, Clone)]
struct Sweep {
    chunk: usize,
    block: usize,
}

pub(crate) struct Heap {
    chunks: UnsafeCell<Vec<NonNull<Chunk>>>,
    /// No chunk before this one has a free block.
    free_from: Cell<usize>,
    free_blocks: Cell<usize>,
    /// Indexed by `Kind`.
    spaces: [Space; 3],
    /// The line mark of the current or last marking; never 0.
    epoch: Cell<u8>,
    /// Flipped as each sweep starts; see `BlockInfo::swept`.
    swept: Cell<bool>,
    sweep: Cell<Option<Sweep>>,
    huge: UnsafeCell<HashTable<Huge>>,
    /// Huge objects allocated since the last collection, the only ones a minor one can free
    /// (mmtk's large object nursery).
    young_huge: UnsafeCell<Vec<NonNull<u8>>>,
}

impl Heap {
    pub(crate) fn new() -> Self {
        Self {
            chunks: UnsafeCell::new(Vec::new()),
            free_from: Cell::new(0),
            free_blocks: Cell::new(0),
            spaces: [
                Space::new(Kind::Plain),
                Space::new(Kind::Leaf),
                Space::new(Kind::Drop),
            ],
            epoch: Cell::new(1),
            swept: Cell::new(false),
            sweep: Cell::new(None),
            huge: UnsafeCell::new(HashTable::new()),
            young_huge: UnsafeCell::new(Vec::new()),
        }
    }

    /// # Safety
    /// No other reference to the list may be live; `Heap` is `!Sync`, so a method using the
    /// result within its own body is enough.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn chunks(&self) -> &mut Vec<NonNull<Chunk>> {
        unsafe { &mut *self.chunks.get() }
    }

    /// # Safety
    /// As for `chunks`.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn huge_table(&self) -> &mut HashTable<Huge> {
        unsafe { &mut *self.huge.get() }
    }

    /// # Safety
    /// `p` must be a live huge object's start, and the result dropped before the table changes.
    unsafe fn huge(&self, p: *const u8) -> &Huge {
        unsafe { self.huge_table() }
            .find(huge_hash(p), |huge| ptr::eq(huge.ptr.as_ptr(), p))
            .expect("not a huge object")
    }

    /// Whether the object at `p` is marked.
    ///
    /// # Safety
    /// `p` must be a live object's start.
    #[inline(always)]
    pub(crate) unsafe fn is_marked(&self, p: GcBox) -> bool {
        let p = p.as_ptr();
        if is_huge(p) {
            #[cold]
            #[inline(never)]
            fn huge_marked(heap: &Heap, p: *const u8) -> bool {
                unsafe { heap.huge(p) }.marked.get()
            }
            return huge_marked(self, p);
        }
        let (w, m) = mark_bit(p);
        chunk_of(p).bits.0[w].get() & m != 0
    }

    /// Mark `p` and return whether it needs tracing, which marks its lines (`mark_lines`). A leaf
    /// is never read: its first line is marked, and the next one is kept from allocation as it may
    /// hold the rest (the Immix paper's conservative line marking); a leaf larger than a line has
    /// its `AUX` bit set and all its lines marked.
    ///
    /// # Safety
    /// `p` must be an unmarked live object's start.
    #[inline(always)]
    pub(crate) unsafe fn mark(&self, p: GcBox) -> bool {
        let p = p.as_ptr();
        if is_huge(p) {
            #[cold]
            #[inline(never)]
            fn mark_huge(heap: &Heap, p: *const u8) -> bool {
                let huge = unsafe { heap.huge(p) };
                huge.marked.set(true);
                huge.needs_trace
            }
            return mark_huge(self, p);
        }
        let chunk = chunk_of(p);
        let (w, m) = mark_bit(p);
        let word = &chunk.bits.0[w];
        let old = word.get();
        word.set(old | m);
        if chunk.blocks[block_index(p)].kind.get() != Kind::Leaf {
            return true;
        }
        debug_assert_eq!(
            old & m << 1 != 0,
            unsafe { GcBox::from_ptr(p) }.size() > LINE
        );
        if old & m << 1 != 0 {
            #[cold]
            #[inline(never)]
            fn mark_big_leaf(heap: &Heap, p: *mut u8) {
                // SAFETY: a marked object's start.
                unsafe { heap.mark_span(p, GcBox::from_ptr(p).size()) }
            }
            mark_big_leaf(self, p);
        } else {
            chunk.lines.0[line_index(p)].set(self.epoch.get());
        }
        false
    }

    /// Mark the lines of the object at `p`, of `size` bytes (mmtk marks lines as it scans).
    ///
    /// # Safety
    /// `p` must be a marked live object's start, and `size` its size.
    #[inline(always)]
    pub(crate) unsafe fn mark_lines(&self, p: GcBox, size: usize) {
        let p = p.as_ptr();
        if !is_huge(p) {
            unsafe { self.mark_span(p, size) };
        }
    }

    /// Mark an object that won't be traced, as `mark` and `mark_lines` would together.
    ///
    /// # Safety
    /// `p` must be an unmarked live object's start, and `size` its size.
    pub(crate) unsafe fn mark_untraced(&self, p: GcBox, size: usize) {
        unsafe {
            self.mark(p);
            self.mark_lines(p, size);
        }
    }

    /// # Safety
    /// `p` must start an object of `size` bytes in a chunk.
    #[inline(always)]
    unsafe fn mark_span(&self, p: *mut u8, size: usize) {
        let lines = &chunk_of(p).lines.0;
        let epoch = self.epoch.get();
        let (first, last) = (line_index(p), line_index(p.wrapping_add(size - 1)));
        // Two stores cover anything up to a line long; a loop here becomes a call to memset.
        lines[first].set(epoch);
        lines[last].set(epoch);
        if last > first + 1 {
            #[cold]
            #[inline(never)]
            fn fill(lines: &[Cell<u8>], epoch: u8) {
                for line in lines {
                    line.set(epoch);
                }
            }
            fill(&lines[first + 1..last], epoch);
        }
    }

    /// Allocate `size` bytes, a multiple of `CELL`, for an object of a type that `needs_drop` and
    /// `needs_trace`. Allocation is charged to `metrics` a hole at a time.
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
            return self.alloc_huge(size, needs_drop, needs_trace, metrics);
        }
        let kind = match (needs_drop, needs_trace) {
            (true, _) => Kind::Drop,
            (false, true) => Kind::Plain,
            (false, false) => Kind::Leaf,
        };
        let space = &self.spaces[kind as usize];
        let top = space.cursor.get();
        let p = if size <= space.limit.get().addr() - top.addr() {
            space.cursor.set(top.wrapping_add(size));
            top
        } else {
            self.alloc_slow(space, size, metrics)
        };
        // Debug builds also mark where plain objects start, to check their lines in the sweep.
        if kind == Kind::Drop
            || (kind == Kind::Leaf && size > LINE)
            || (cfg!(debug_assertions) && kind == Kind::Plain)
        {
            let (w, m) = mark_bit(p);
            chunk_of(p).bits.0[w].update(|x| x | m << 1);
        }
        // SAFETY: in a chunk.
        unsafe { NonNull::new_unchecked(p) }
    }

    #[cold]
    #[inline(never)]
    fn alloc_slow(&self, space: &Space, size: usize, metrics: &Metrics) -> *mut u8 {
        if size > LINE {
            let top = space.big_cursor.get();
            if size <= space.big_limit.get().addr() - top.addr() {
                space.big_cursor.set(top.wrapping_add(size));
                return top;
            }
            let start = self.take_free_block(space.kind, metrics);
            space.big_cursor.set(start.wrapping_add(size));
            space.big_limit.set(start.wrapping_add(BLOCK));
            return start;
        }
        let (start, end) = self.next_hole(space, metrics);
        space.cursor.set(start.wrapping_add(size));
        space.limit.set(end);
        start
    }

    /// The next hole for `space`: in its block, then in its recyclable blocks, then a free block.
    fn next_hole(&self, space: &Space, metrics: &Metrics) -> (*mut u8, *mut u8) {
        loop {
            let block = space.hole_block.get();
            if !block.is_null() {
                let chunk = chunk_of(block);
                let b = block_index(block);
                let conservative = space.kind == Kind::Leaf;
                if let Some((s, e)) = find_hole(chunk, b, space.hole_line.get(), conservative) {
                    space.hole_line.set(e);
                    chunk.blocks[b].held.update(|h| h + (e - s) as u16);
                    metrics.mark_gc_allocated((e - s) * LINE);
                    let bits = block_bits(chunk, b);
                    for l in s..e {
                        // The two lines of a word, one half each.
                        let half = AUX & (0xffff_ffff << (l % 2 * 32));
                        if conservative {
                            // Leaves larger than a line that died here.
                            bits[l / 2].update(|x| x & !half);
                        } else {
                            debug_assert_eq!(
                                bits[l / 2].get() & half,
                                0,
                                "a free line starts an object"
                            );
                        }
                    }
                    return (block.wrapping_add(s * LINE), block.wrapping_add(e * LINE));
                }
                space.hole_block.set(ptr::null_mut());
            }
            if let Some(block) = self.pop_recyclable(space, metrics) {
                space.hole_block.set(block);
                space.hole_line.set(0);
                continue;
            }
            let block = self.take_free_block(space.kind, metrics);
            return (block, block.wrapping_add(BLOCK));
        }
    }

    /// A swept block of `space`'s kind with free lines, sweeping further for one while a sweep is
    /// under way and there is no free block to take instead.
    fn pop_recyclable(&self, space: &Space, metrics: &Metrics) -> Option<*mut u8> {
        loop {
            let head = space.recyclable.get();
            if !head.is_null() {
                space
                    .recyclable
                    .set(chunk_of(head).blocks[block_index(head)].next.get());
                return Some(head);
            }
            if self.free_blocks.get() != 0 || !self.sweep_next(metrics) {
                return None;
            }
        }
    }

    /// Take a free block for `kind`, sweeping for one or else mapping a new chunk if there is none,
    /// and charge all of it.
    fn take_free_block(&self, kind: Kind, metrics: &Metrics) -> *mut u8 {
        while self.free_blocks.get() == 0 && self.sweep_next(metrics) {}
        let mut i = self.free_from.get();
        let (chunk, b) = loop {
            let Some(&chunk) = unsafe { self.chunks() }.get(i) else {
                break (self.new_chunk(), META_BLOCKS);
            };
            // SAFETY: chunks in the list are live.
            let chunk = unsafe { chunk.as_ref() };
            let free = chunk.free.get();
            if free != 0 {
                break (chunk, free.trailing_zeros() as usize);
            }
            i += 1;
        };
        self.free_from.set(i);
        self.free_blocks.update(|n| n - 1);
        chunk.free.update(|f| f & !(1 << b));
        let info = &chunk.blocks[b];
        info.next.set(ptr::null_mut());
        info.held.set(LINES as u16);
        info.kind.set(kind);
        // Nothing allocated here before the sweep began, so there's nothing for it to free.
        info.swept.set(self.swept.get());
        debug_assert!(block_lines(chunk, b).iter().all(|l| l.get() == 0));
        // No marks in a free block, but `AUX` bits of dead objects.
        for w in block_bits(chunk, b) {
            w.set(0);
        }
        metrics.mark_gc_allocated(BLOCK);
        metrics.mark_reserved(BLOCK);
        block_start(chunk, b)
    }

    fn new_chunk(&self) -> &Chunk {
        // SAFETY: the mapping is zeroed, which the rest of the metadata starts as.
        let chunk = unsafe {
            let chunk = map(CHUNK).cast::<Chunk>();
            ptr::addr_of_mut!((*chunk.as_ptr()).free).write(Cell::new(DATA_BLOCKS));
            chunk
        };
        unsafe { self.chunks() }.push(chunk);
        self.free_blocks
            .update(|n| n + DATA_BLOCKS.count_ones() as usize);
        // SAFETY: just made; freed only by `finish_sweep` or `Drop`, once free.
        unsafe { chunk.as_ref() }
    }

    #[cold]
    #[inline(never)]
    fn alloc_huge(
        &self,
        size: usize,
        needs_drop: bool,
        needs_trace: bool,
        metrics: &Metrics,
    ) -> NonNull<u8> {
        // Not rounded up to whole chunks: only its pages are mapped, and charged.
        let ptr = map(size);
        let huge = Huge {
            ptr,
            size,
            marked: Cell::new(false),
            needs_drop,
            needs_trace,
        };
        unsafe { self.huge_table() }.insert_unique(huge_hash(ptr.as_ptr()), huge, |huge| {
            huge_hash(huge.ptr.as_ptr())
        });
        unsafe { &mut *self.young_huge.get() }.push(ptr);
        let pages = size.next_multiple_of(page_size());
        metrics.mark_gc_allocated(pages);
        metrics.mark_reserved(pages);
        ptr
    }

    /// Start a full collection's marking: a new line mark, and every mark cleared. A minor one
    /// keeps the marks, so everything marked before is old and not traced again (mmtk's sticky
    /// mark bits).
    pub(crate) fn start_marking(&self) {
        self.epoch.set(if self.epoch.get() == u8::MAX {
            1
        } else {
            self.epoch.get() + 1
        });
        for &chunk in unsafe { self.chunks() }.iter() {
            // SAFETY: chunks in the list are live.
            let chunk = unsafe { chunk.as_ref() };
            let free = chunk.free.get();
            for b in META_BLOCKS..BLOCKS {
                if free & 1 << b == 0 {
                    for w in block_bits(chunk, b) {
                        w.update(|x| x & AUX);
                    }
                }
            }
        }
        for huge in unsafe { self.huge_table() }.iter() {
            huge.marked.set(false);
        }
    }

    /// Sweep the huge objects, and begin sweeping every block in use now. Allocation from here on
    /// only uses blocks already swept, or free.
    pub(crate) fn start_sweep(&self, metrics: &Metrics, full: bool) {
        // All at once, unlike blocks, so that huge objects allocated during the sweep are left
        // alone.
        let free = |huge: &Huge| {
            let size = huge.size.next_multiple_of(page_size());
            // SAFETY: an unmarked huge object is unreachable.
            unsafe { free_huge(huge) };
            metrics.mark_gc_freed(size);
            metrics.mark_released(size);
        };
        let table = unsafe { self.huge_table() };
        let young = unsafe { &mut *self.young_huge.get() };
        if full {
            table.retain(|huge| {
                if !huge.marked.get() {
                    free(huge);
                }
                huge.marked.get()
            });
        } else {
            for p in young.iter() {
                let Ok(entry) = table.find_entry(huge_hash(p.as_ptr()), |huge| huge.ptr == *p)
                else {
                    unreachable!("a young huge object outside the table")
                };
                if !entry.get().marked.get() {
                    free(entry.get());
                    entry.remove();
                }
            }
        }
        young.clear();
        for space in &self.spaces {
            space.retire();
        }
        self.swept.update(|s| !s);
        self.sweep.set(Some(Sweep {
            chunk: 0,
            block: META_BLOCKS,
        }));
    }

    /// Sweep the next block; false once every block is swept.
    pub(crate) fn sweep_next(&self, metrics: &Metrics) -> bool {
        let Some(mut sweep) = self.sweep.get() else {
            return false;
        };
        loop {
            let Some(&chunk) = unsafe { self.chunks() }.get(sweep.chunk) else {
                self.sweep.set(Some(sweep));
                return false;
            };
            // SAFETY: chunks in the list are live.
            let chunk = unsafe { chunk.as_ref() };
            if sweep.block >= BLOCKS {
                sweep.chunk += 1;
                sweep.block = META_BLOCKS;
                continue;
            }
            let b = sweep.block;
            sweep.block += 1;
            if chunk.free.get() & 1 << b != 0 || chunk.blocks[b].swept.get() == self.swept.get() {
                continue;
            }
            self.sweep.set(Some(sweep));
            self.sweep_block(chunk, sweep.chunk, b, metrics);
            return true;
        }
    }

    /// End the sweep once `sweep_next` is done, returning wholly free chunks to the OS but one.
    pub(crate) fn finish_sweep(&self) {
        debug_assert!(
            self.sweep
                .get()
                .is_some_and(|s| s.chunk == unsafe { self.chunks() }.len())
        );
        self.sweep.set(None);
        // One is kept so a heap hovering at a chunk boundary doesn't map and unmap every cycle.
        let mut spare = false;
        unsafe { self.chunks() }.retain(|&chunk| {
            // SAFETY: chunks in the list are live.
            if unsafe { chunk.as_ref() }.free.get() != DATA_BLOCKS || !spare {
                spare |= unsafe { chunk.as_ref() }.free.get() == DATA_BLOCKS;
                return true;
            }
            // SAFETY: no block of it is in use, so nothing points into it.
            unsafe { unmap(chunk.as_ptr().cast(), CHUNK) };
            self.free_blocks
                .update(|n| n - DATA_BLOCKS.count_ones() as usize);
            false
        });
        self.free_from.set(0);
    }

    /// Drop the dead objects of block `b` of `chunk`, the `i`th chunk, and free its lines that no
    /// marking reached since it was last swept (mmtk's `Block::sweep`).
    fn sweep_block(&self, chunk: &Chunk, i: usize, b: usize, metrics: &Metrics) {
        let info = &chunk.blocks[b];
        info.swept.set(self.swept.get());
        let kind = info.kind.get();
        let epoch = self.epoch.get();
        let lines = block_lines(chunk, b);
        let bits = block_bits(chunk, b);
        let start = block_start(chunk, b);
        if kind == Kind::Drop || (cfg!(debug_assertions) && kind == Kind::Plain) {
            for (w, word) in bits.iter().enumerate() {
                let x = word.get();
                #[cfg(debug_assertions)]
                {
                    let mut live = x & AUX & (x & MARK) << 1;
                    while live != 0 {
                        let cell = w * 32 + live.trailing_zeros() as usize / 2;
                        let p = start.wrapping_add(cell * CELL);
                        let size = unsafe { GcBox::from_ptr(p) }.size();
                        let (first, last) = (line_index(p), line_index(p.wrapping_add(size - 1)));
                        debug_assert!(
                            (first..=last).all(|l| chunk.lines.0[l].get() == epoch),
                            "a marked object's line is unmarked"
                        );
                        live &= live - 1;
                    }
                }
                let dead = x & AUX & !((x & MARK) << 1);
                if dead == 0 {
                    continue;
                }
                if kind == Kind::Drop {
                    let mut d = dead;
                    while d != 0 {
                        let cell = w * 32 + d.trailing_zeros() as usize / 2;
                        // SAFETY: an unmarked object after marking is unreachable.
                        unsafe { drop_object(start.wrapping_add(cell * CELL)) };
                        d &= d - 1;
                    }
                }
                word.set(x & !dead);
            }
        }
        // A byte counter keeps the loop in byte lanes.
        let mut live = 0u8;
        for line in lines {
            let kept = line.get() == epoch;
            live += kept as u8;
            line.set(if kept { epoch } else { 0 });
        }
        let live = live as usize;
        let held = info.held.get() as usize;
        debug_assert!(live <= held, "marked lines never handed out");
        metrics.mark_gc_freed((held - live) * LINE);
        info.held.set(live as u16);
        if live == 0 {
            metrics.mark_released(BLOCK);
            chunk.free.update(|f| f | 1 << b);
            self.free_blocks.update(|n| n + 1);
            self.free_from.update(|f| f.min(i));
        } else if find_hole(chunk, b, 0, kind == Kind::Leaf).is_some() {
            let space = &self.spaces[kind as usize];
            info.next.set(space.recyclable.get());
            space.recyclable.set(start);
        }
    }
}

/// The first hole at or after line `from` of block `b`: a run of free lines, less the first free
/// line after a live one when `conservative`, as it may hold the end of a small object.
fn find_hole(chunk: &Chunk, b: usize, from: usize, conservative: bool) -> Option<(usize, usize)> {
    let lines = block_lines(chunk, b);
    let mut l = from;
    loop {
        while l < LINES && lines[l].get() != 0 {
            l += 1;
        }
        if l == LINES {
            return None;
        }
        if conservative && l > 0 && lines[l - 1].get() != 0 {
            l += 1;
            continue;
        }
        break;
    }
    let start = l;
    while l < LINES && lines[l].get() == 0 {
        l += 1;
    }
    Some((start, l))
}

impl Drop for Heap {
    fn drop(&mut self) {
        for &chunk in self.chunks.get_mut().iter() {
            // SAFETY: the heap owns its chunks; every `AUX` bit in a drop block starts an
            // initialized object.
            unsafe {
                let c = chunk.as_ref();
                let free = c.free.get();
                for b in META_BLOCKS..BLOCKS {
                    if free & 1 << b != 0 || c.blocks[b].kind.get() != Kind::Drop {
                        continue;
                    }
                    let start = block_start(c, b);
                    for (w, word) in block_bits(c, b).iter().enumerate() {
                        let mut starts = word.get() & AUX;
                        while starts != 0 {
                            let cell = w * 32 + starts.trailing_zeros() as usize / 2;
                            drop_object(start.wrapping_add(cell * CELL));
                            starts &= starts - 1;
                        }
                    }
                }
                unmap(chunk.as_ptr().cast(), CHUNK);
            }
        }
        for huge in self.huge.get_mut().drain() {
            // SAFETY: as above.
            unsafe { free_huge(&huge) };
        }
    }
}

/// Drop the object at `p` unless it was dropped already.
///
/// # Safety
/// `p` must start an initialized object nothing will use again.
unsafe fn drop_object(p: *mut u8) {
    // SAFETY: as the caller promises.
    unsafe {
        let mut gc_box = GcBox::from_ptr(p);
        if gc_box.header().is_live() {
            gc_box.drop_in_place();
        }
    }
}
