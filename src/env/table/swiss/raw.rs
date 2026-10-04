use core::alloc::{Allocator, Layout};
use core::hint::{likely, unlikely};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::ptr::{self, NonNull};
use core::slice;
use std::alloc::handle_alloc_error;

use super::control::{BitMaskIter, Group, Tag, TagSliceExt};

// `RawTableInner::replace_tag_in_group` stores a group as one `u64` or `u128`.
const _: () = assert!(
    Group::WIDTH == 8 || Group::WIDTH == 16,
    "32-bit targets other than wasm32 are unsupported: their generic group is 4 bytes"
);

/// Primary hash function, used to select the initial bucket to probe from.
#[inline]
fn h1(hash: u64) -> usize {
    hash as usize
}

/// Probe sequence based on triangular numbers, which is guaranteed (since our
/// table size is a power of two) to visit every group of elements exactly once.
///
/// A triangular probe has us jump by 1 more group every time. So first we
/// jump by 1 group (meaning we just continue our linear scan), then 2 groups
/// (skipping over 1 group), then 3 groups (skipping over 2 groups), and so on.
///
/// Proof that the probe will visit every group in the table:
/// <https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/>
#[derive(Clone)]
struct ProbeSeq {
    pos: usize,
    stride: usize,
}

impl ProbeSeq {
    #[inline]
    fn move_next(&mut self, bucket_mask: usize) {
        // We should have found an empty bucket by now and ended the probe.
        debug_assert!(
            self.stride <= bucket_mask,
            "Went past end of probe sequence"
        );

        self.stride = self.stride.wrapping_add(Group::WIDTH);
        self.pos = self.pos.wrapping_add(self.stride) & bucket_mask;
    }
}

/// Returns the number of buckets needed to hold the given number of items,
/// taking the maximum load factor into account.
///
/// Returns `None` if an overflow occurs.
///
/// This ensures that `buckets * table_layout.size >= table_layout.ctrl_align`.
#[inline]
fn capacity_to_buckets(cap: usize, table_layout: TableLayout) -> Option<usize> {
    debug_assert_ne!(cap, 0);

    // For small tables we require at least 1 empty bucket so that lookups are
    // guaranteed to terminate if an element doesn't exist in the table.
    if cap < 15 {
        // In general, buckets * table_layout.size >= table_layout.ctrl_align
        // must be true to avoid wasting bytes on padding to ctrl_align for
        // small item sizes, so the minimum capacity is adjusted upwards for
        // small items.
        //
        // This is brittle, e.g. if we ever add 32 byte groups, it will select
        // 3 regardless of the table_layout.size.
        let min_cap = match (Group::WIDTH, table_layout.size) {
            (16, 0..=1) => 14,
            (16, 2..=3) | (8, 0..=1) => 7,
            _ => 3,
        };
        let cap = min_cap.max(cap);
        // We don't bother with a table size of 2 buckets since that can only
        // hold a single element. Instead, we skip directly to a 4 bucket table
        // which can hold 3 elements.
        let buckets = if cap < 4 {
            4
        } else if cap < 8 {
            8
        } else {
            16
        };
        ensure_bucket_bytes_at_least_ctrl_align(table_layout, buckets);
        return Some(buckets);
    }

    // Otherwise require 1/8 buckets to be empty (87.5% load)
    //
    // Be careful when modifying this, calculate_layout relies on the
    // overflow check here.
    let adjusted_cap = cap.checked_mul(8)? / 7;

    // Any overflows will have been caught by the checked_mul. Also, any
    // rounding errors from the division above will be cleaned up by
    // next_power_of_two (which can't overflow because of the previous division).
    let buckets = adjusted_cap.next_power_of_two();
    ensure_bucket_bytes_at_least_ctrl_align(table_layout, buckets);
    Some(buckets)
}

// `maximum_buckets_in` relies on the property that for non-ZST `T`, any
// chosen `buckets` will satisfy `buckets * table_layout.size >=
// table_layout.ctrl_align`, so `calculate_layout_for` does not need to add
// extra padding beyond `table_layout.size * buckets`. If small-table bucket
// selection or growth policy changes, revisit `maximum_buckets_in`.
#[inline]
fn ensure_bucket_bytes_at_least_ctrl_align(table_layout: TableLayout, buckets: usize) {
    if table_layout.size != 0 {
        let prod = table_layout.size.saturating_mul(buckets);
        debug_assert!(prod >= table_layout.ctrl_align);
    }
}

/// Returns the maximum effective capacity for the given bucket mask, taking
/// the maximum load factor into account.
#[inline]
fn bucket_mask_to_capacity(bucket_mask: usize) -> usize {
    if bucket_mask < 8 {
        // For tables with 1/2/4/8 buckets, we always reserve one empty slot.
        // Keep in mind that the bucket mask is one less than the bucket count.
        bucket_mask
    } else {
        // `bucket_mask` is bounded by the maximum allocation size, so it can
        // never be `usize::MAX` and the `+ 1` below cannot overflow.
        debug_assert!(bucket_mask != usize::MAX);
        // For larger tables we reserve 12.5% of the slots as empty.
        ((bucket_mask + 1) / 8) * 7
    }
}

/// Helper which allows the max calculation for `ctrl_align` to be statically computed for each `T`
/// while keeping the rest of `calculate_layout_for` independent of `T`
#[derive(Copy, Clone)]
struct TableLayout {
    size: usize,
    ctrl_align: usize,
}

impl TableLayout {
    #[inline]
    const fn new<T>() -> Self {
        let layout = Layout::new::<T>();
        Self {
            size: layout.size(),
            ctrl_align: if layout.align() > Group::WIDTH {
                layout.align()
            } else {
                Group::WIDTH
            },
        }
    }

    #[inline]
    fn calculate_layout_for(self, buckets: usize) -> Option<(Layout, usize)> {
        debug_assert!(buckets.is_power_of_two());

        let TableLayout { size, ctrl_align } = self;
        // Manual layout calculation since Layout methods are not yet stable.
        let ctrl_offset =
            size.checked_mul(buckets)?.checked_add(ctrl_align - 1)? & !(ctrl_align - 1);
        let len = ctrl_offset.checked_add(buckets + Group::WIDTH)?;

        // We need an additional check to ensure that the allocation doesn't
        // exceed `isize::MAX` (https://github.com/rust-lang/rust/pull/95295).
        if len > isize::MAX as usize - (ctrl_align - 1) {
            return None;
        }

        Some((
            unsafe { Layout::from_size_align_unchecked(len, ctrl_align) },
            ctrl_offset,
        ))
    }
}

/// A Swiss table of `Copy` entries. Deleting an entry leaves a *dead* bucket whose contents
/// stay readable, so [`position`](Self::position) can still find it, until the next rehash
/// drops it. Setting the key again revives it; other keys may take it like a tombstone.
///
/// On a key's probe sequence, its entry comes before any other bucket with its tag and key
/// bits, so a probe that also matches dead buckets finds the right one. Keys compared by
/// address can share bits across entries once an address is reused. The order still holds
/// because a new key takes the first free bucket, ahead of every dead one, and a revive only
/// passes dead buckets whose tag differs, which tag matches, being exact, never take for it.
pub(crate) struct RawTable<T: Copy, A: Allocator> {
    table: RawTableInner,
    alloc: A,
    marker: PhantomData<T>,
}

/// Non-generic part of `RawTable` which allows functions to be instantiated only once regardless
/// of how many different key-value types are used.
struct RawTableInner {
    // Mask to get an index from a hash value. The value is one less than the
    // number of buckets in the table.
    bucket_mask: usize,

    // [Padding], T_n, ..., T1, T0, C0, C1, ...
    //                              ^ points here
    ctrl: NonNull<u8>,

    // Number of elements that can be inserted before we need to grow the table
    growth_left: usize,

    // Number of live elements in the table
    items: usize,
}

impl<T: Copy, A: Allocator> RawTable<T, A> {
    const TABLE_LAYOUT: TableLayout = TableLayout::new::<T>();

    /// Creates a new empty hash table without allocating any memory, using the
    /// given allocator.
    #[inline]
    pub(crate) const fn new_in(alloc: A) -> Self {
        Self {
            table: RawTableInner::NEW,
            alloc,
            marker: PhantomData,
        }
    }

    /// Allocates a new hash table using the given allocator, with at least enough capacity for
    /// inserting the given number of elements without reallocating.
    pub(crate) fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            table: RawTableInner::with_capacity(&alloc, Self::TABLE_LAYOUT, capacity),
            alloc,
            marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying allocator.
    #[inline]
    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Number of live entries.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.table.items
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    unsafe fn bucket(&self, index: usize) -> NonNull<T> {
        unsafe { self.table.bucket(index) }
    }

    /// Bucket index of the live entry `eq` accepts.
    #[inline]
    fn find(&self, hash: u64, mut eq: impl FnMut(&T) -> bool) -> Option<usize> {
        // SAFETY: `find_inner` only offers full buckets.
        unsafe {
            self.table
                .find_inner(hash, &mut |index| eq(self.bucket(index).as_ref()))
        }
    }

    /// The live entry `eq` accepts.
    #[inline]
    pub(crate) fn get(&self, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<&T> {
        let index = self.find(hash, eq)?;
        // SAFETY: `find` returns full buckets.
        Some(unsafe { self.bucket(index).as_ref() })
    }

    /// Kills the entry `eq` accepts, if any. Whether it was live is data, not a branch, so
    /// deleting keys of unpredictable liveness doesn't mispredict.
    #[inline]
    pub(crate) fn kill(&mut self, hash: u64, mut eq: impl FnMut(&T) -> bool) {
        // SAFETY: `find_entry_inner` only offers and returns full and dead buckets, and the
        // probe position it returns is a group load's.
        unsafe {
            if let Some((pos, bit)) = self
                .table
                .find_entry_inner(hash, &mut |index| eq(self.bucket(index).as_ref()))
            {
                let old = self
                    .table
                    .replace_tag_in_group(pos, bit, Tag::full(hash).dead());
                self.table.items -= usize::from(old.is_full());
            }
        }
    }

    /// The bucket of the entry, live or dead, that `eq` accepts: the key's own, by the order
    /// on [`RawTable`].
    #[inline]
    pub(crate) fn position(&self, hash: u64, mut eq: impl FnMut(&T) -> bool) -> Option<usize> {
        // A traversal's cursor is nearly always live, and probing live tags alone keeps the
        // dead tag match off its path.
        // SAFETY: both probes only offer full and dead buckets.
        unsafe {
            let eq = &mut |index| eq(self.bucket(index).as_ref());
            self.table.find_inner(hash, eq).or_else(|| {
                let (pos, bit) = self.table.find_entry_inner(hash, eq)?;
                Some((pos + bit) & self.table.bucket_mask)
            })
        }
    }

    /// The entry, live or dead, `eq` accepts as `Ok` (its probe position and bit), or else as
    /// `Err` the first free (empty or dead) bucket on `hash`'s probe sequence, where a new entry
    /// goes. Unlike upstream this doesn't reserve room first, so it never moves an entry.
    #[inline]
    pub(crate) fn find_or_find_insert_index(
        &self,
        hash: u64,
        mut eq: impl FnMut(&T) -> bool,
    ) -> Result<(usize, usize), usize> {
        // SAFETY: `find_or_find_insert_index_inner` only offers full and dead buckets to `eq`.
        unsafe {
            self.table
                .find_or_find_insert_index_inner(hash, &mut |index| eq(self.bucket(index).as_ref()))
        }
    }

    /// Makes the entry at `pos + bit` live under `hash`, whether or not it was, returning it.
    ///
    /// # Safety
    ///
    /// `pos` and `bit` must come from [`find_or_find_insert_index`]
    /// (Self::find_or_find_insert_index) for `hash` as `Ok`, with no mutation since.
    #[inline]
    pub(crate) unsafe fn revive(&mut self, pos: usize, bit: usize, hash: u64) -> &mut T {
        unsafe {
            let old = self.table.replace_tag_in_group(pos, bit, Tag::full(hash));
            self.table.items += usize::from(!old.is_full());
            self.bucket((pos + bit) & self.table.bucket_mask).as_mut()
        }
    }

    /// Whether filling the free bucket `index` must wait for the table to grow.
    ///
    /// # Safety
    ///
    /// `index` must be a free bucket from a probe, either
    /// [`find_or_find_insert_index`](Self::find_or_find_insert_index) as `Err` or
    /// `RawTableInner::find_insert_index`, with no mutation since.
    #[inline]
    pub(crate) unsafe fn needs_growth(&self, index: usize) -> bool {
        self.table.growth_left == 0 && unsafe { *self.table.ctrl(index) }.special_is_empty()
    }

    /// Stores `value`, whose hash is `hash`, in the free bucket `index`.
    ///
    /// # Safety
    ///
    /// `index` must be a free bucket from a probe for `hash`, as for
    /// [`needs_growth`](Self::needs_growth), which must be false for it.
    #[inline]
    pub(crate) unsafe fn insert_at_index(&mut self, index: usize, hash: u64, value: T) {
        unsafe {
            debug_assert!(self.table.ctrl(index).read().is_special());
            let old_ctrl = *self.table.ctrl(index);
            self.table
                .record_item_insert_at(index, old_ctrl, Tag::full(hash));
            self.bucket(index).write(value);
        }
    }

    /// Inserts `value`, whose key has no entry in the table, making room if needed.
    pub(crate) fn insert(&mut self, hash: u64, value: T, hasher: impl Fn(&T) -> u64) {
        unsafe {
            let mut index = self.table.find_insert_index(hash);
            if unlikely(self.needs_growth(index)) {
                self.reserve_rehash(hasher);
                index = self.table.find_insert_index(hash);
            }
            self.insert_at_index(index, hash, value);
        }
    }

    /// The first live entry at bucket `from` or later.
    #[inline]
    pub(crate) fn next_full(&self, from: usize) -> Option<&T> {
        let index = self.table.next_full_scalar(from)?;
        // SAFETY: `next_full_scalar` returns full buckets.
        Some(unsafe { self.bucket(index).as_ref() })
    }

    /// Every live entry.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        // SAFETY: the iterator borrows the table, and yields full buckets.
        unsafe { self.table.full_buckets_indices() }
            .map(|index| unsafe { self.bucket(index).as_ref() })
    }

    /// Kills every live entry `f` accepts.
    pub(crate) fn kill_where(&mut self, mut f: impl FnMut(&T) -> bool) {
        // SAFETY: the iterator yields full buckets and allows killing the ones it yielded.
        unsafe {
            for index in self.table.full_buckets_indices() {
                if f(self.bucket(index).as_ref()) {
                    self.table.kill(index);
                }
            }
        }
    }

    /// Makes room for one more element: rehashes in place, dropping deleted and dead
    /// buckets, if that leaves the table at most half full, else grows it.
    #[cold]
    #[inline(never)]
    fn reserve_rehash(&mut self, hasher: impl Fn(&T) -> u64) {
        let hasher =
            |table: &RawTableInner, index| hasher(unsafe { table.bucket::<T>(index).as_ref() });
        let new_items = self.table.items + 1;
        let full_capacity = bucket_mask_to_capacity(self.table.bucket_mask);
        // SAFETY: the layout and allocator are the ones this table was allocated with.
        unsafe {
            if new_items <= full_capacity / 2 {
                self.table.rehash_in_place(&hasher, Self::TABLE_LAYOUT.size);
            } else {
                self.table.resize_inner(
                    &self.alloc,
                    usize::max(new_items, full_capacity + 1),
                    &hasher,
                    Self::TABLE_LAYOUT,
                );
            }
        }
    }
}

impl<T: Copy, A: Allocator> Drop for RawTable<T, A> {
    fn drop(&mut self) {
        if !self.table.is_empty_singleton() {
            // SAFETY: the layout and allocator are the ones this table was allocated with.
            unsafe { self.table.free_buckets(&self.alloc, Self::TABLE_LAYOUT) };
        }
    }
}

impl RawTableInner {
    /// Creates a new empty hash table without allocating any memory.
    ///
    /// In effect this returns a table with exactly 1 bucket. However we can
    /// leave the data pointer dangling since that bucket is never accessed
    /// due to our load factor forcing us to always have at least 1 free bucket.
    const NEW: Self = Self {
        // Be careful to cast the entire slice to a raw pointer.
        ctrl: unsafe { NonNull::new_unchecked(Group::static_empty().as_ptr().cast_mut().cast()) },
        bucket_mask: 0,
        items: 0,
        growth_left: 0,
    };
}

/// Find the previous power of 2. If it's already a power of 2, it's unchanged.
/// Passing zero is undefined behavior.
fn prev_pow2(z: usize) -> usize {
    let shift = usize::BITS as usize - 1;
    1 << (shift - (z.leading_zeros() as usize))
}

/// Finds the largest number of buckets that can fit in `allocation_size`
/// provided the given TableLayout.
///
/// This relies on some invariants of `capacity_to_buckets`, so only feed in
/// an `allocation_size` calculated from `capacity_to_buckets`.
fn maximum_buckets_in(
    allocation_size: usize,
    table_layout: TableLayout,
    group_width: usize,
) -> usize {
    // Given an equation like:
    //   z >= x * y + x + g
    // x can be maximized by doing:
    //   x = (z - g) / (y + 1)
    // If you squint:
    //   x is the number of buckets
    //   y is the table_layout.size
    //   z is the size of the allocation
    //   g is the group width
    // But this is ignoring the padding needed for ctrl_align.
    // If we remember these restrictions:
    //   x is always a power of 2
    //   Layout size for T must always be a multiple of T
    // Then the alignment can be ignored if we add the constraint:
    //   x * y >= table_layout.ctrl_align
    // This is taken care of by `capacity_to_buckets`.
    // It may be helpful to understand this if you remember that:
    //   ctrl_offset = align(x * y, ctrl_align)
    let x = (allocation_size - group_width) / (table_layout.size + 1);
    prev_pow2(x)
}

impl RawTableInner {
    /// Allocates a new [`RawTableInner`] with the given number of buckets.
    /// The control bytes and buckets are left uninitialized.
    ///
    /// # Safety
    ///
    /// The caller of this function must ensure that the `buckets` is power of two
    /// and also initialize all control bytes of the length `self.bucket_mask + 1 +
    /// Group::WIDTH` with the [`Tag::EMPTY`] bytes.
    unsafe fn new_uninitialized<A: Allocator>(
        alloc: &A,
        table_layout: TableLayout,
        mut buckets: usize,
    ) -> Self {
        debug_assert!(buckets.is_power_of_two());

        let Some((layout, mut ctrl_offset)) = table_layout.calculate_layout_for(buckets) else {
            panic!("Hash table capacity overflow");
        };

        let Ok(block) = alloc.allocate(layout) else {
            handle_alloc_error(layout);
        };
        // The allocator can't return a value smaller than was
        // requested, so this can be != instead of >=.
        if block.len() != layout.size() {
            // Utilize over-sized allocations.
            let x = maximum_buckets_in(block.len(), table_layout, Group::WIDTH);
            debug_assert!(x >= buckets);
            // Calculate the new ctrl_offset.
            let (oversized_layout, oversized_ctrl_offset) = {
                let option = table_layout.calculate_layout_for(x);
                unsafe { option.unwrap_unchecked() }
            };
            debug_assert!(oversized_layout.size() <= block.len());
            debug_assert!(oversized_ctrl_offset >= ctrl_offset);
            ctrl_offset = oversized_ctrl_offset;
            buckets = x;
        }

        // SAFETY: the allocation is `ctrl_offset + buckets + Group::WIDTH` bytes long.
        let ctrl = unsafe { block.cast::<u8>().add(ctrl_offset) };
        Self {
            ctrl,
            bucket_mask: buckets - 1,
            items: 0,
            growth_left: bucket_mask_to_capacity(buckets - 1),
        }
    }

    /// Allocates a new [`RawTableInner`] with at least enough capacity for inserting
    /// the given number of elements without reallocating.
    ///
    /// All the control bytes are initialized with the [`Tag::EMPTY`] bytes.
    fn with_capacity<A: Allocator>(alloc: &A, table_layout: TableLayout, capacity: usize) -> Self {
        if capacity == 0 {
            return Self::NEW;
        }
        let Some(buckets) = capacity_to_buckets(capacity, table_layout) else {
            panic!("Hash table capacity overflow");
        };
        // SAFETY: We checked that we could successfully allocate the new table, and then
        // initialized all control bytes with the constant `Tag::EMPTY` byte.
        unsafe {
            let mut result = Self::new_uninitialized(alloc, table_layout, buckets);
            result.ctrl_slice().fill_empty();
            result
        }
    }

    /// Fixes up an insertion index returned by the [`RawTableInner::find_insert_index_in_group`] method.
    ///
    /// In tables smaller than the group width (`self.num_buckets() < Group::WIDTH`), trailing control
    /// bytes outside the range of the table are filled with [`Tag::EMPTY`] entries. These will unfortunately
    /// trigger a match of [`RawTableInner::find_insert_index_in_group`] function. This is because
    /// the `Some(bit)` returned by `group.match_empty_or_deleted().lowest_set_bit()` after masking
    /// (`(probe_seq.pos + bit) & self.bucket_mask`) may point to a full bucket that is already occupied.
    /// We detect this situation here and perform a second scan starting at the beginning of the table.
    /// This second scan is guaranteed to find an empty slot (due to the load factor) before hitting the
    /// trailing control bytes (containing [`Tag::EMPTY`] bytes).
    ///
    /// # Safety
    ///
    /// The control bytes must be initialized, and `index` must come from
    /// `find_insert_index_in_group`, with no insertion since.
    #[inline]
    unsafe fn fix_insert_index(&self, mut index: usize) -> usize {
        if unlikely(unsafe { self.is_bucket_full(index) }) {
            debug_assert!(self.bucket_mask < Group::WIDTH);
            // SAFETY: the table is smaller than a group, so this scan finds a free bucket
            // (due to the load factor) before the trailing EMPTY control bytes.
            index = unsafe {
                Group::load_aligned(self.ctrl(0))
                    .match_empty_or_deleted()
                    .lowest_set_bit()
                    .unwrap_unchecked()
            };
        }
        index
    }

    /// Finds the position to insert something in a group.
    ///
    /// **This may have false positives and must be fixed up with `fix_insert_index`
    /// before it's used.**
    #[inline]
    fn find_insert_index_in_group(&self, group: &Group, probe_seq: &ProbeSeq) -> Option<usize> {
        let bit = group.match_empty_or_deleted().lowest_set_bit();

        if likely(bit.is_some()) {
            Some((probe_seq.pos + bit.unwrap()) & self.bucket_mask)
        } else {
            None
        }
    }

    /// Searches for an element in the table, live or dead, or a potential slot where that
    /// element could be inserted (an empty, deleted or dead bucket index).
    ///
    /// This uses dynamic dispatch to reduce the amount of code generated, but that is
    /// eliminated by LLVM optimizations.
    ///
    /// Unlike upstream this doesn't reserve room first, so the caller checks
    /// `RawTable::needs_growth` before inserting at an `Err` index.
    ///
    /// # Safety
    ///
    /// The control bytes must be initialized.
    #[inline]
    unsafe fn find_or_find_insert_index_inner(
        &self,
        hash: u64,
        eq: &mut dyn FnMut(usize) -> bool,
    ) -> Result<(usize, usize), usize> {
        let mut insert_index = None;

        let tag_hash = Tag::full(hash);
        let tag_dead = tag_hash.dead();
        let mut probe_seq = self.probe_seq(hash);

        loop {
            // SAFETY: `pos` is a multiple of `Group::WIDTH` no greater than `bucket_mask`, and
            // the control bytes are group-aligned and run `Group::WIDTH` past the last bucket
            // (or are `Group::static_empty()` for the unallocated table).
            let group = unsafe { Group::load_aligned(self.ctrl(probe_seq.pos)) };

            // A key's entry is its first match, live or dead, so both tags are tried together.
            for bit in group.match_tag(tag_hash) | group.match_tag(tag_dead) {
                let index = (probe_seq.pos + bit) & self.bucket_mask;

                if likely(eq(index)) {
                    return Ok((probe_seq.pos, bit));
                }
            }

            // We didn't find the element we were looking for in the group, try to get an
            // insertion slot from the group if we don't have one yet.
            if likely(insert_index.is_none()) {
                insert_index = self.find_insert_index_in_group(&group, &probe_seq);
            }

            if let Some(insert_index) = insert_index {
                // Only stop the search if the group contains at least one empty element.
                // Otherwise, the element that we are looking for might be in a following group.
                if likely(group.match_empty().any_bit_set()) {
                    // SAFETY: the index comes from `find_insert_index_in_group`.
                    unsafe {
                        return Err(self.fix_insert_index(insert_index));
                    }
                }
            }

            probe_seq.move_next(self.bucket_mask);
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting a new
    /// element and sets the hash for that slot. Returns an index of that slot and the
    /// old control byte stored in the found index.
    ///
    /// # Safety
    ///
    /// The table must be allocated with initialized control bytes and a free bucket, and the
    /// caller must write an element with this hash at the returned index and update `items`
    /// and `growth_left`.
    #[inline]
    unsafe fn prepare_insert_index(&mut self, hash: u64) -> (usize, Tag) {
        unsafe {
            let index: usize = self.find_insert_index(hash);
            let old_ctrl = *self.ctrl(index);
            self.set_ctrl_hash(index, hash);
            (index, old_ctrl)
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting
    /// a new element, returning the `index` for the new bucket.
    ///
    /// # Safety
    ///
    /// The control bytes must be initialized and the table must have a free bucket,
    /// otherwise this never returns or returns an index past the last bucket.
    #[inline]
    unsafe fn find_insert_index(&self, hash: u64) -> usize {
        let mut probe_seq = self.probe_seq(hash);
        loop {
            // SAFETY: see `find_or_find_insert_index_inner`.
            let group = unsafe { Group::load_aligned(self.ctrl(probe_seq.pos)) };

            let index = self.find_insert_index_in_group(&group, &probe_seq);
            if likely(index.is_some()) {
                // SAFETY: the index comes from `find_insert_index_in_group`.
                unsafe {
                    return self.fix_insert_index(index.unwrap_unchecked());
                }
            }
            probe_seq.move_next(self.bucket_mask);
        }
    }

    /// Searches for an element in a table, returning the `index` of the found element.
    /// This uses dynamic dispatch to reduce the amount of code generated, but it is
    /// eliminated by LLVM optimizations.
    ///
    /// # Safety
    ///
    /// The control bytes must be initialized.
    #[inline(always)]
    unsafe fn find_inner(&self, hash: u64, eq: &mut dyn FnMut(usize) -> bool) -> Option<usize> {
        let tag_hash = Tag::full(hash);
        let mut probe_seq = self.probe_seq(hash);

        loop {
            // SAFETY: see `find_or_find_insert_index_inner`.
            let group = unsafe { Group::load_aligned(self.ctrl(probe_seq.pos)) };

            for bit in group.match_tag(tag_hash) {
                // This is the same as `(probe_seq.pos + bit) % self.num_buckets()` because the number
                // of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
                let index = (probe_seq.pos + bit) & self.bucket_mask;

                if likely(eq(index)) {
                    return Some(index);
                }
            }

            if likely(group.match_empty().any_bit_set()) {
                return None;
            }

            probe_seq.move_next(self.bucket_mask);
        }
    }

    /// [`find_inner`](Self::find_inner) over full and dead buckets, returning the match's probe
    /// position and bit.
    ///
    /// # Safety
    ///
    /// The control bytes must be initialized.
    #[inline(always)]
    unsafe fn find_entry_inner(
        &self,
        hash: u64,
        eq: &mut dyn FnMut(usize) -> bool,
    ) -> Option<(usize, usize)> {
        let tag_hash = Tag::full(hash);
        let tag_dead = tag_hash.dead();
        let mut probe_seq = self.probe_seq(hash);

        loop {
            // SAFETY: see `find_or_find_insert_index_inner`.
            let group = unsafe { Group::load_aligned(self.ctrl(probe_seq.pos)) };

            for bit in group.match_tag(tag_hash) | group.match_tag(tag_dead) {
                let index = (probe_seq.pos + bit) & self.bucket_mask;

                if likely(eq(index)) {
                    return Some((probe_seq.pos, bit));
                }
            }

            if likely(group.match_empty().any_bit_set()) {
                return None;
            }

            probe_seq.move_next(self.bucket_mask);
        }
    }

    /// Sets the control byte at `pos + bit` to `tag`, returning the old one, by storing the
    /// whole group window at `pos`. Unlike a byte store at the probed index, the store's
    /// address doesn't wait on the probe, so a following lookup's load of the same group
    /// forwards from it instead of running ahead and replaying.
    ///
    /// # Safety
    ///
    /// The table must be allocated, `pos` a probe position and `bit < Group::WIDTH`.
    #[inline]
    unsafe fn replace_tag_in_group(&mut self, pos: usize, bit: usize, tag: Tag) -> Tag {
        let shift = bit * 8;
        unsafe {
            if Group::WIDTH == 8 {
                let group = self.ctrl(pos).cast::<[u8; 8]>();
                let w = u64::from_le_bytes(group.read());
                let old = (w >> shift) as u8;
                group.write((w ^ u64::from(old ^ tag.0) << shift).to_le_bytes());
                Tag(old)
            } else {
                let group = self.ctrl(pos).cast::<[u8; 16]>();
                let w = u128::from_le_bytes(group.read());
                let old = (w >> shift) as u8;
                group.write((w ^ u128::from(old ^ tag.0) << shift).to_le_bytes());
                Tag(old)
            }
        }
    }

    /// Returns an iterator over full buckets indices in the table.
    ///
    /// # Safety
    ///
    /// The table must outlive the iterator and have initialized control bytes.
    #[inline(always)]
    unsafe fn full_buckets_indices(&self) -> FullBucketsIndices {
        unsafe {
            let ctrl = NonNull::new_unchecked(self.ctrl(0).cast::<u8>());

            FullBucketsIndices {
                // Load the first group
                current_group: Group::load_aligned(ctrl.as_ptr().cast())
                    .match_full()
                    .into_iter(),
                group_first_index: 0,
                ctrl,
                items: self.items,
            }
        }
    }

    /// Index of the first full bucket at `from` or later, a bucket at a time. A `next` call
    /// usually finds one within a few, and a branchy scan lets the CPU predict past it,
    /// where a group scan would make the entry load wait on the SIMD match.
    #[inline]
    fn next_full_scalar(&self, from: usize) -> Option<usize> {
        // SAFETY: each index is a bucket.
        (from..self.num_buckets()).find(|&index| unsafe { self.is_bucket_full(index) })
    }

    /// Prepares for rehashing data in place (that is, without allocating new memory).
    /// Converts all full index `control bytes` to `Tag::DELETED` and all special control
    /// bytes, dead ones included, to `Tag::EMPTY`.
    ///
    /// # Safety
    ///
    /// The table must be allocated with initialized control bytes, and the caller must turn
    /// every `Tag::DELETED` byte back into a full one.
    #[inline]
    unsafe fn prepare_rehash_in_place(&mut self) {
        // Bulk convert all full control bytes to DELETED, and all DELETED control bytes to EMPTY.
        // This effectively frees up all buckets containing a DELETED entry.
        //
        // SAFETY: `i` steps through the buckets by `Group::WIDTH` from the aligned start of the
        // control bytes, which run `Group::WIDTH` past the last bucket.
        unsafe {
            for i in (0..self.num_buckets()).step_by(Group::WIDTH) {
                let group = Group::load_aligned(self.ctrl(i));
                let group = group.convert_special_to_empty_and_full_to_deleted();
                group.store_aligned(self.ctrl(i));
            }
        }
    }

    /// Returns a pointer to the bucket at `index`.
    ///
    /// # Safety
    ///
    /// The table must be allocated and `index < self.num_buckets()`.
    #[inline]
    unsafe fn bucket<T>(&self, index: usize) -> NonNull<T> {
        debug_assert_ne!(self.bucket_mask, 0);
        debug_assert!(index < self.num_buckets());
        // Not `sub(index + 1)`: LLVM negates that as `!index`, a dependent instruction before
        // the entry load, where this folds the `- 1` into the load's offset as upstream does.
        unsafe { self.data_end::<T>().sub(index).sub(1) }
    }

    /// Returns a raw `*mut u8` pointer to the start of the `data` element in the table
    /// (convenience for `self.data_end::<u8>().as_ptr().sub((index + 1) * size_of)`).
    ///
    /// # Safety
    ///
    /// The table must be allocated, `index < self.num_buckets()`, and `size_of` must be
    /// the size of the elements stored in the table.
    #[inline]
    unsafe fn bucket_ptr(&self, index: usize, size_of: usize) -> *mut u8 {
        debug_assert_ne!(self.bucket_mask, 0);
        debug_assert!(index < self.num_buckets());
        unsafe {
            let base: *mut u8 = self.data_end().as_ptr();
            base.sub((index + 1) * size_of)
        }
    }

    /// Returns pointer to one past last `data` element in the table as viewed from
    /// the start point of the allocation (convenience for `self.ctrl.cast()`).
    ///
    /// ```none
    ///                        `table.data_end::<T>()` returns pointer that points here
    ///                        (to the end of `T0`)
    ///                          ∨
    /// [Pad], T_n, ..., T1, T0, |CT0, CT1, ..., CT_n|, CTa_0, CTa_1, ..., CTa_m
    ///                           \________  ________/
    ///                                    \/
    ///       `n = buckets - 1`, i.e. `RawTableInner::num_buckets() - 1`
    ///
    /// where: T0...T_n  - our stored data;
    ///        CT0...CT_n - control bytes or metadata for `data`.
    ///        CTa_0...CTa_m - additional control bytes, where `m = Group::WIDTH - 1`, always
    ///                        EMPTY, so that a table smaller than a group loads as one group.
    /// ```
    #[inline]
    fn data_end<T>(&self) -> NonNull<T> {
        self.ctrl.cast()
    }

    /// Returns an iterator-like object for a probe sequence on the table.
    ///
    /// This iterator never terminates, but is guaranteed to visit each bucket
    /// group exactly once. The loop using `probe_seq` must terminate upon
    /// reaching a group containing an empty bucket.
    #[inline]
    fn probe_seq(&self, hash: u64) -> ProbeSeq {
        ProbeSeq {
            // This is the same as `hash as usize % self.num_buckets()` because the number
            // of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
            // Unlike upstream, groups are aligned, so a control byte sits in one group and
            // needs no mirror at the end of the array.
            pos: h1(hash) & self.bucket_mask & !(Group::WIDTH - 1),
            stride: 0,
        }
    }

    #[inline]
    unsafe fn record_item_insert_at(&mut self, index: usize, old_ctrl: Tag, new_ctrl: Tag) {
        self.growth_left -= usize::from(old_ctrl.special_is_empty());
        unsafe {
            self.set_ctrl(index, new_ctrl);
        }
        self.items += 1;
    }

    #[inline]
    fn is_in_same_group(&self, i: usize, new_i: usize, hash: u64) -> bool {
        let probe_seq_pos = self.probe_seq(hash).pos;
        let probe_index =
            |pos: usize| (pos.wrapping_sub(probe_seq_pos) & self.bucket_mask) / Group::WIDTH;
        probe_index(i) == probe_index(new_i)
    }

    /// Sets a control byte to the hash.
    ///
    /// # Safety
    ///
    /// The table must be allocated and `index <= self.bucket_mask`.
    #[inline]
    unsafe fn set_ctrl_hash(&mut self, index: usize, hash: u64) {
        unsafe {
            self.set_ctrl(index, Tag::full(hash));
        }
    }

    /// Replaces the hash in the control byte at the given index with the provided one,
    /// returning the old control byte.
    ///
    /// # Safety
    ///
    /// The table must be allocated and `index <= self.bucket_mask`.
    #[inline]
    unsafe fn replace_ctrl_hash(&mut self, index: usize, hash: u64) -> Tag {
        unsafe {
            let prev_ctrl = *self.ctrl(index);
            self.set_ctrl_hash(index, hash);
            prev_ctrl
        }
    }

    /// Sets a control byte.
    ///
    /// # Safety
    ///
    /// The table must be allocated and `index <= self.bucket_mask`.
    #[inline]
    unsafe fn set_ctrl(&mut self, index: usize, ctrl: Tag) {
        unsafe {
            *self.ctrl(index) = ctrl;
        }
    }

    /// Returns a pointer to a control byte.
    ///
    /// # Safety
    ///
    /// `index < self.bucket_mask + 1 + Group::WIDTH`; only reads are allowed on the
    /// unallocated table.
    #[inline]
    unsafe fn ctrl(&self, index: usize) -> *mut Tag {
        debug_assert!(index < self.num_ctrl_bytes());
        unsafe { self.ctrl.as_ptr().add(index).cast() }
    }

    /// Gets the slice of all control bytes, as possibily uninitialized tags.
    fn ctrl_slice(&mut self) -> &mut [mem::MaybeUninit<Tag>] {
        // SAFETY: We have the correct number of control bytes.
        unsafe { slice::from_raw_parts_mut(self.ctrl.as_ptr().cast(), self.num_ctrl_bytes()) }
    }

    #[inline]
    fn num_buckets(&self) -> usize {
        self.bucket_mask + 1
    }

    /// Checks whether the bucket at `index` is full.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index` is less than the number of buckets.
    #[inline]
    unsafe fn is_bucket_full(&self, index: usize) -> bool {
        debug_assert!(index < self.num_buckets());
        unsafe { (*self.ctrl(index)).is_full() }
    }

    #[inline]
    fn num_ctrl_bytes(&self) -> usize {
        self.bucket_mask + 1 + Group::WIDTH
    }

    #[inline]
    fn is_empty_singleton(&self) -> bool {
        self.bucket_mask == 0
    }

    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    ///
    /// # Safety
    ///
    /// `alloc` and `layout` must be the ones this table was allocated with, the control
    /// bytes must be initialized, and `capacity >= self.items`.
    #[inline(always)]
    unsafe fn resize_inner<A: Allocator>(
        &mut self,
        alloc: &A,
        capacity: usize,
        hasher: &dyn Fn(&Self, usize) -> u64,
        layout: TableLayout,
    ) {
        debug_assert!(self.items <= capacity);
        let mut new_table = RawTableInner::with_capacity(alloc, layout, capacity);

        for full_byte_index in unsafe { self.full_buckets_indices() } {
            let hash = hasher(self, full_byte_index);

            // SAFETY: the new table has room for every element, holds no deleted buckets, and
            // gets an element at `new_index` right away.
            unsafe {
                let (new_index, _) = new_table.prepare_insert_index(hash);
                ptr::copy_nonoverlapping(
                    self.bucket_ptr(full_byte_index, layout.size),
                    new_table.bucket_ptr(new_index, layout.size),
                    layout.size,
                );
            }
        }

        new_table.growth_left -= self.items;
        new_table.items = self.items;

        mem::swap(self, &mut new_table);
        if !new_table.is_empty_singleton() {
            // SAFETY: `new_table` now holds the old allocation, made with `alloc` and `layout`.
            unsafe { new_table.free_buckets(alloc, layout) };
        }
    }

    /// Rehashes the contents of the table in place (i.e. without changing the
    /// allocation).
    ///
    /// # Safety
    ///
    /// The table must be allocated with initialized control bytes, and `size_of` must be
    /// the size of the elements stored in the table.
    unsafe fn rehash_in_place(&mut self, hasher: &dyn Fn(&Self, usize) -> u64, size_of: usize) {
        unsafe {
            self.prepare_rehash_in_place();
        }

        // At this point, DELETED elements are elements that we haven't
        // rehashed yet. Find them and re-insert them at their ideal
        // position.
        'outer: for i in 0..self.num_buckets() {
            unsafe {
                if *self.ctrl(i) != Tag::DELETED {
                    continue;
                }
            }

            let i_p = unsafe { self.bucket_ptr(i, size_of) };

            loop {
                // Hash the current item
                let hash = hasher(self, i);

                // Search for a suitable place to put it
                let new_i = unsafe { self.find_insert_index(hash) };

                // Probing works by scanning through all of the control
                // bytes in groups. If both the new and old position fall
                // within the same group, then there is no benefit in moving
                // it and we can just continue to the next item.
                if likely(self.is_in_same_group(i, new_i, hash)) {
                    unsafe { self.set_ctrl_hash(i, hash) };
                    continue 'outer;
                }

                let new_i_p = unsafe { self.bucket_ptr(new_i, size_of) };

                // We are moving the current item to a new position. Write
                // our H2 to the control byte of the new position.
                let prev_ctrl = unsafe { self.replace_ctrl_hash(new_i, hash) };
                if prev_ctrl == Tag::EMPTY {
                    unsafe { self.set_ctrl(i, Tag::EMPTY) };
                    // If the target slot is empty, simply move the current
                    // element into the new slot and clear the old control
                    // byte.
                    unsafe {
                        ptr::copy_nonoverlapping(i_p, new_i_p, size_of);
                    }
                    continue 'outer;
                }

                // If the target slot is occupied, swap the two elements
                // and then continue processing the element that we just
                // swapped into the old slot.
                debug_assert_eq!(prev_ctrl, Tag::DELETED);
                unsafe {
                    ptr::swap_nonoverlapping(i_p, new_i_p, size_of);
                }
            }
        }

        self.growth_left = bucket_mask_to_capacity(self.bucket_mask) - self.items;
    }

    /// Deallocates the table without dropping any entries.
    ///
    /// # Safety
    ///
    /// The table must be allocated, with `alloc` and `table_layout`.
    #[inline]
    unsafe fn free_buckets<A: Allocator>(&mut self, alloc: &A, table_layout: TableLayout) {
        unsafe {
            let (ptr, layout) = self.allocation_info(table_layout);
            alloc.deallocate(ptr, layout);
        }
    }

    /// Returns a pointer to the allocated memory and the layout that was used to
    /// allocate the table.
    ///
    /// # Safety
    ///
    /// The table must be allocated, with `table_layout`.
    #[inline]
    unsafe fn allocation_info(&self, table_layout: TableLayout) -> (NonNull<u8>, Layout) {
        debug_assert!(
            !self.is_empty_singleton(),
            "this function can only be called on non-empty tables"
        );

        let (layout, ctrl_offset) = {
            let option = table_layout.calculate_layout_for(self.num_buckets());
            unsafe { option.unwrap_unchecked() }
        };
        (
            unsafe { NonNull::new_unchecked(self.ctrl.as_ptr().sub(ctrl_offset)) },
            layout,
        )
    }

    /// Marks the full bucket at `index` dead.
    ///
    /// # Safety
    ///
    /// The table must be allocated and `index` must be a full bucket.
    #[inline]
    unsafe fn kill(&mut self, index: usize) {
        unsafe {
            let tag = *self.ctrl(index);
            debug_assert!(tag.is_full());
            self.set_ctrl(index, tag.dead());
        }
        self.items -= 1;
    }
}

/// Iterator which returns an index of every full bucket in the table.
///
/// For maximum flexibility this iterator is not bound by a lifetime, but you
/// must observe several rules when using it:
/// - You must not free the hash table while iterating (including via growing/shrinking).
/// - It is fine to erase a bucket that has been yielded by the iterator.
/// - Erasing a bucket that has not yet been yielded by the iterator may still
///   result in the iterator yielding index of that bucket.
/// - It is unspecified whether an element inserted after the iterator was
///   created will be yielded by that iterator.
/// - The order in which the iterator yields indices of the buckets is unspecified
///   and may change in the future.
#[derive(Clone)]
struct FullBucketsIndices {
    // Mask of full buckets in the current group. Bits are cleared from this
    // mask as each element is processed.
    current_group: BitMaskIter,

    // Initial value of the bytes' indices of the current group (relative
    // to the start of the control bytes).
    group_first_index: usize,

    // Pointer to the current group of control bytes,
    // Must be aligned to the group size (Group::WIDTH).
    ctrl: NonNull<u8>,

    // Number of elements in the table.
    items: usize,
}

impl FullBucketsIndices {
    /// Advances the iterator and returns the next value.
    ///
    /// # Safety
    ///
    /// The table must be alive and not moved, and this must not be called after
    /// every element was yielded.
    #[inline(always)]
    unsafe fn next_impl(&mut self) -> Option<usize> {
        loop {
            if let Some(index) = self.current_group.next() {
                // The returned `self.group_first_index + index` will always
                // be in the range `0..self.num_buckets()`. See explanation below.
                return Some(self.group_first_index + index);
            }

            // SAFETY: we stop after the last element, so for tables no larger than a group we
            // never get here, and for larger ones (a multiple of the group width) the last
            // group loaded starts at `num_buckets() - Group::WIDTH`.
            unsafe {
                self.ctrl = NonNull::new_unchecked(self.ctrl.as_ptr().add(Group::WIDTH));
            }

            // SAFETY: See explanation above.
            unsafe {
                self.current_group = Group::load_aligned(self.ctrl.as_ptr().cast())
                    .match_full()
                    .into_iter();
                self.group_first_index += Group::WIDTH;
            }
        }
    }
}

impl Iterator for FullBucketsIndices {
    type Item = usize;

    /// Advances the iterator and returns the next value. It is up to
    /// the caller to ensure that the `RawTable` outlives the `FullBucketsIndices`,
    /// because we cannot make the `next` method unsafe.
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        // Return if we already yielded all items.
        if self.items == 0 {
            return None;
        }

        // SAFETY:
        // 1. We check number of items to yield using `items` field.
        // 2. The caller ensures that the table is alive and has not moved.
        let nxt = unsafe { self.next_impl() };

        debug_assert!(nxt.is_some());
        self.items -= 1;

        nxt
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.items, Some(self.items))
    }
}

impl ExactSizeIterator for FullBucketsIndices {}
impl FusedIterator for FullBucketsIndices {}
