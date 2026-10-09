use core::cell::Cell;

/// When an [`crate::Arena`] collects, after mmtk-core's StickyImmix trigger: once the memory in
/// use passes a heap limit, which each full collection sets to twice what survived it (LuaJIT's
/// pause of 200, in place of mmtk's MemBalancer). A collection is full if less than `min_nursery`
/// was left under the limit when the last one ended, and minor otherwise. Memory in use is the
/// lines and huge objects allocated and not freed, plus external allocations: not mmtk's reserved
/// pages, which count a block whole while any line of it lives, so that survivors spread thin
/// over many blocks don't inflate the limit.
#[derive(Debug, Copy, Clone)]
pub struct Pacing {
    /// The least heap limit, and the limit until the first full collection.
    pub min_heap: usize,
    /// mmtk's minimum nursery: the least room left under the limit at the end of a collection
    /// that keeps the next one minor.
    pub min_nursery: usize,
}

impl Pacing {
    pub const DEFAULT: Pacing = Pacing {
        min_heap: 1 << 20,
        min_nursery: 2 << 20,
    };
}

impl Default for Pacing {
    #[inline]
    fn default() -> Pacing {
        Self::DEFAULT
    }
}

/// Allocation between [`Metrics::gc_check_due`] firings for a host that doesn't collect when it
/// does.
const DEFERRED_GC_CHECK: usize = 64 << 10;

/// Allocation counter and the threshold the interpreter compares it with,
/// adjacent so one `ldp` loads both.
#[derive(Debug, Default)]
#[repr(C)]
struct GcCheck {
    allocated_bytes_total: Cell<usize>,
    gc_check_at: Cell<usize>,
}

impl GcCheck {
    /// Whether allocation since the last [`Metrics::arm_gc_check`] may have pushed memory in use
    /// past the heap limit. Memory in use grows by at most what is allocated, so this can fire
    /// early, never late.
    #[inline(always)]
    fn due(&self) -> bool {
        self.allocated_bytes_total.get() >= self.gc_check_at.get()
    }
}

#[derive(Debug, Default)]
struct MetricsInner {
    gc_check: GcCheck,

    pacing: Cell<Pacing>,

    /// Bytes of lines and huge objects handed to allocation and not yet freed.
    total_gc_bytes: Cell<usize>,
    total_external_bytes: Cell<usize>,

    /// Set by the last full collection; 0 before the first.
    heap_limit: Cell<usize>,
    next_full: Cell<bool>,

    minor_collections: Cell<usize>,
    full_collections: Cell<usize>,
}

/// The arena's allocation and collection counters. Lives in its own allocation, owned by the
/// arena's `Context` and freed after every object, so allocators can point at it (see
/// [`MetricsAlloc`](crate::dmm::MetricsAlloc)); handed out only by reference.
pub struct Metrics(MetricsInner);

impl Metrics {
    /// Offset of the allocation counter, followed by the threshold
    /// `gc_check_due` compares it with, for compiled code.
    pub(crate) const GC_CHECK_OFFSET: usize = std::mem::offset_of!(Metrics, 0.gc_check);
}

impl Metrics {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Sets the parameters that decide when the arena collects.
    #[inline]
    pub fn set_pacing(&self, pacing: Pacing) {
        self.0.pacing.set(pacing);
    }

    /// Returns the bytes of lines and huge objects held by live `Gc` pointers, as of the last
    /// collection, plus those allocated since. A line is held whole while any object in it lives.
    #[inline]
    pub fn total_gc_allocation(&self) -> usize {
        self.0.total_gc_bytes.get()
    }

    /// Returns the total bytes that have been marked as externally allocated.
    ///
    /// A call to [`Metrics::mark_external_allocation`] will increase this count, and a call to
    /// [`Metrics::mark_external_deallocation`] will decrease it.
    #[inline]
    pub fn total_external_allocation(&self) -> usize {
        self.0.total_external_bytes.get()
    }

    /// Returns the sum of `Metrics::total_gc_allocation()` and
    /// `Metrics::total_external_allocation()`.
    #[inline]
    pub fn total_allocation(&self) -> usize {
        self.0
            .total_gc_bytes
            .get()
            .saturating_add(self.0.total_external_bytes.get())
    }

    /// Minor and full collections finished so far.
    pub fn collections(&self) -> (usize, usize) {
        (
            self.0.minor_collections.get(),
            self.0.full_collections.get(),
        )
    }

    /// Call to mark that bytes have been externally allocated that are owned by an arena. They
    /// count as memory in use.
    #[inline]
    pub fn mark_external_allocation(&self, bytes: usize) {
        self.0
            .total_external_bytes
            .update(|b| b.saturating_add(bytes));
        self.0
            .gc_check
            .allocated_bytes_total
            .update(|b| b.wrapping_add(bytes));
    }

    /// Call to mark that bytes which have been marked as allocated with
    /// [`Metrics::mark_external_allocation`] have been since deallocated.
    ///
    /// It is safe, but may result in unspecified behavior (such as very weird GC pacing), if the
    /// amount of bytes marked for deallocation is greater than the number of bytes marked for
    /// allocation.
    #[inline]
    pub fn mark_external_deallocation(&self, bytes: usize) {
        self.0
            .total_external_bytes
            .update(|b| b.saturating_sub(bytes));
    }

    /// How far the memory in use is past the heap limit; a collection is due once this is
    /// positive.
    #[inline]
    pub fn allocation_debt(&self) -> f64 {
        if self.0.total_gc_bytes.get() == 0 {
            // If we have no live `Gc`s, then there is no possible collection to do so always
            // return zero debt.
            return 0.0;
        }

        self.raw_debt().max(0.0)
    }

    #[inline(always)]
    pub fn gc_check_due(&self) -> bool {
        self.0.gc_check.due()
    }

    /// Arm [`Metrics::gc_check_due`] to fire once memory in use could pass the heap limit.
    pub fn arm_gc_check(&self) {
        let remaining = (-self.raw_debt()).max(1.0) as usize;
        self.set_gc_check_in(remaining);
    }

    /// Fire [`Metrics::gc_check_due`] again only after `DEFERRED_GC_CHECK` more bytes, so a host
    /// that keeps stepping without collecting isn't sent back at every check.
    pub fn defer_gc_check(&self) {
        self.set_gc_check_in(DEFERRED_GC_CHECK);
    }

    fn set_gc_check_in(&self, bytes: usize) {
        let c = &self.0.gc_check;
        c.gc_check_at
            .set(c.allocated_bytes_total.get().saturating_add(bytes));
    }

    fn in_use(&self) -> usize {
        self.total_allocation()
    }

    fn heap_limit(&self) -> usize {
        self.0.heap_limit.get().max(self.0.pacing.get().min_heap)
    }

    // `allocation_debt` without the clamp: minus the room left under the limit.
    fn raw_debt(&self) -> f64 {
        self.in_use() as f64 - self.heap_limit() as f64
    }

    /// Whether the next collection is a full one.
    pub(crate) fn next_full(&self) -> bool {
        self.0.next_full.get()
    }

    pub(crate) fn finish_cycle(&self, full: bool) {
        let pacing = self.0.pacing.get();
        let in_use = self.in_use();
        // Against the limit the collection ran under, as mmtk's `StickyImmix::on_pause_end`
        // decides before the trigger sets a new one.
        self.0
            .next_full
            .set(self.heap_limit().saturating_sub(in_use) < pacing.min_nursery);
        if full {
            self.0.heap_limit.set((2 * in_use).max(pacing.min_heap));
            self.0.full_collections.update(|n| n + 1);
        } else {
            self.0.minor_collections.update(|n| n + 1);
        }
    }

    /// Lines or huge object bytes handed to allocation.
    #[inline]
    pub(crate) fn mark_gc_allocated(&self, bytes: usize) {
        self.0.total_gc_bytes.update(|b| b + bytes);
        self.0
            .gc_check
            .allocated_bytes_total
            .update(|b| b.wrapping_add(bytes));
    }

    #[inline]
    pub(crate) fn mark_gc_freed(&self, bytes: usize) {
        self.0.total_gc_bytes.update(|b| b - bytes);
    }
}
