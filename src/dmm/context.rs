use core::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::{self, ManuallyDrop},
    ops::{ControlFlow, Deref, DerefMut},
    ptr::NonNull,
};
use std::{boxed::Box, vec::Vec};

use crate::dmm::{
    Gc, GcWeak,
    collect::{Collect, Trace},
    heap::{CELL, GrayQueue, Heap},
    metrics::Metrics,
    types::{GcBox, GcBoxHeader, GcBoxInner, Invariant, TrailingBytes, trailing_layout},
};

/// Handle value given by arena callbacks during construction and mutation. Allows allocating new
/// `Gc` pointers and internally mutating values held by `Gc` pointers.
#[repr(transparent)]
pub struct Mutation<'gc> {
    context: Context,
    _invariant: Invariant<'gc>,
}

impl<'gc> Mutation<'gc> {
    #[inline]
    pub fn metrics(&self) -> &Metrics {
        self.context.metrics()
    }

    /// IF the `parent` pointer is colored black and we are in the marking phase, then change it to
    /// gray and enqueue it for tracing again. A white parent just becomes light gray (see
    /// `GcBoxHeader`), so it needs no barrier until it is next traced; `child` is not looked at.
    ///
    /// This operation is known as a "backwards write barrier". Calling this method is one of the
    /// safe ways for the value in the `parent` pointer to use internal mutability to adopt the
    /// `child` pointer without invalidating the color invariant.
    ///
    /// If the `child` parameter is given, then calling this method ensures that the `parent`
    /// pointer may safely adopt the `child` pointer. If no `child` is given, then calling this
    /// method is more general, and it ensures that the `parent` pointer may adopt *any* child
    /// pointer(s) before collection is next triggered.
    #[inline]
    pub fn backward_barrier(&self, parent: Gc<'gc, ()>, child: Option<Gc<'gc, ()>>) {
        self.context.backward_barrier(
            unsafe { GcBox::erase(parent.ptr) },
            child.map(|p| unsafe { GcBox::erase(p.ptr) }),
        )
    }

    /// Whether [`Mutation::backward_barrier`] on `parent` would enqueue it. When not, the rest of
    /// what it does is done, so `parent` may be written without it.
    #[inline]
    pub fn backward_barrier_pending(&self, parent: Gc<'gc, ()>) -> bool {
        self.context
            .backward_barrier_pending(unsafe { GcBox::erase(parent.ptr) })
    }

    /// A version of [`Mutation::backward_barrier`] that allows adopting a [`GcWeak`] child.
    #[inline]
    pub fn backward_barrier_weak(&self, parent: Gc<'gc, ()>, child: GcWeak<'gc, ()>) {
        self.context
            .backward_barrier_weak(unsafe { GcBox::erase(parent.ptr) }, unsafe {
                GcBox::erase(child.inner.ptr)
            })
    }

    /// IF we are in the marking phase AND the `parent` pointer (if given) is colored black, AND
    /// the `child` is colored white, then immediately change the `child` to gray and enqueue it
    /// for tracing.
    ///
    /// This operation is known as a "forwards write barrier". Calling this method is one of the
    /// safe ways for the value in the `parent` pointer to use internal mutability to adopt the
    /// `child` pointer without invalidating the color invariant.
    ///
    /// If the `parent` parameter is given, then calling this method ensures that the `parent`
    /// pointer may safely adopt the `child` pointer. If no `parent` is given, then calling this
    /// method is more general, and it ensures that the `child` pointer may be adopted by *any*
    /// parent pointer(s) before collection is next triggered.
    #[inline]
    pub fn forward_barrier(&self, parent: Option<Gc<'gc, ()>>, child: Gc<'gc, ()>) {
        self.context
            .forward_barrier(parent.map(|p| unsafe { GcBox::erase(p.ptr) }), unsafe {
                GcBox::erase(child.ptr)
            })
    }

    /// A version of [`Mutation::forward_barrier`] that allows adopting a [`GcWeak`] child.
    #[inline]
    pub fn forward_barrier_weak(&self, parent: Option<Gc<'gc, ()>>, child: GcWeak<'gc, ()>) {
        self.context
            .forward_barrier_weak(parent.map(|p| unsafe { GcBox::erase(p.ptr) }), unsafe {
                GcBox::erase(child.inner.ptr)
            })
    }

    #[inline]
    pub(crate) fn allocate<T: Collect<'gc> + 'gc>(&self, t: T) -> NonNull<GcBoxInner<T>> {
        self.context.allocate(t)
    }

    #[inline]
    pub(crate) fn allocate_trailing<T: Collect<'gc> + TrailingBytes + 'gc>(
        &self,
        t: T,
        init: impl FnOnce(NonNull<u8>),
    ) -> NonNull<GcBoxInner<T>> {
        self.context.allocate_trailing(t, init)
    }

    #[inline]
    pub(crate) fn upgrade(&self, gc_box: GcBox) -> bool {
        self.context.upgrade(gc_box)
    }
}

/// Handle value given to finalization callbacks in `MarkedArena`.
///
/// Derefs to `Mutation<'gc>` to allow for arbitrary mutation, but adds additional powers to examine
/// the state of the fully marked arena.
#[repr(transparent)]
pub struct Finalization<'gc> {
    context: Context,
    _invariant: Invariant<'gc>,
}

impl<'gc> Deref for Finalization<'gc> {
    type Target = Mutation<'gc>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Finalization and Mutation are #[repr(transparent)]
        unsafe { mem::transmute::<&Self, &Mutation>(self) }
    }
}

impl<'gc> Finalization<'gc> {
    #[inline]
    pub(crate) fn resurrect(&self, gc_box: GcBox) {
        self.context.resurrect(gc_box)
    }

    /// # Safety
    /// `gc_box` must be a live object.
    #[inline]
    pub(crate) unsafe fn is_marked(&self, gc_box: GcBox) -> bool {
        unsafe { self.context.heap.is_marked(gc_box) }
    }

    /// The objects whose trace called [`Trace::defer`] this cycle, each once.
    pub fn deferred(&self) -> Vec<Gc<'gc, ()>> {
        let boxes = self.context.deferred.dedup();
        boxes.into_iter().map(|b| unsafe { b.as_gc() }).collect()
    }

    /// Mark the [`Finalization::deferred`] objects handled; sweeping refuses to start until they
    /// are.
    pub fn clear_deferred(&self) {
        self.context.deferred.clear()
    }
}

impl<'gc> Trace<'gc> for Context {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        let gc_box = unsafe { GcBox::erase(gc.ptr) };
        Context::trace(self, gc_box)
    }

    fn trace_gc_weak(&mut self, gc: GcWeak<'gc, ()>) {
        let gc_box = unsafe { GcBox::erase(gc.inner.ptr) };
        Context::trace_weak(self, gc_box)
    }

    fn defer(&mut self) {
        let gc_box = self.tracing.expect("`defer` outside of an object's trace");
        self.deferred.push(gc_box);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Phase {
    Mark,
    Sweep,
    Sleep,
    Drop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum RunUntil {
    // Run collection until we reach the stop condition *or* debt is zero.
    PayDebt,
    // Run collection until we reach our stop condition.
    Stop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum Stop {
    // Don't proceed past the end of marking, *just* before the sweep phase
    FullyMarked,
    // Don't proceed past the very beginning of the sweep phase
    AtSweep,
    // Stop once we reach the end of the current cycle and are in `Phase::Sleep`.
    FinishCycle,
    // Stop once we have done an entire cycle as a single atomic unit. This is the maximum amount
    // of work that a call to `Context::do_collection` will do, since a full collection as a single
    // atomic unit means that all unreachable values *must* already be freed.
    Full,
}

pub(crate) struct Context {
    // A separate allocation, not a field: a `&mut Context` would otherwise invalidate the pointers
    // allocators hold into it. Freed in `drop`, after every object.
    metrics: NonNull<Metrics>,
    phase: Phase,

    // Every object. Dropped by hand before the metrics its objects' allocators point to.
    heap: ManuallyDrop<Heap>,

    /// Does the root needs to be traced?
    /// This should be `true` at the beginning of `Phase::Mark`.
    root_needs_trace: bool,

    /// A queue of gray objects, used during `Phase::Mark`.
    /// This holds traceable objects that have yet to be traced.
    gray: GrayQueue,

    // A queue of gray objects that became gray as a result
    // of a write barrier.
    gray_again: Queue<GcBox>,

    // The object `mark_one` is tracing, for `Trace::defer`.
    tracing: Option<GcBox>,

    // Objects queued by `Trace::defer`, with duplicates; see `Finalization::deferred`.
    deferred: Queue<GcBox>,

    // Unmarked objects a `GcWeak` reached this cycle, each once (see `settle_weak`).
    weak: Queue<GcBox>,
}

impl Drop for Context {
    fn drop(&mut self) {
        let mut cx = PhaseGuard::enter(self, Some(Phase::Drop));
        // SAFETY: dropped once, here; the heap drops every object.
        unsafe { ManuallyDrop::drop(&mut cx.heap) };
        // SAFETY: every object, and so every allocator pointing here, is gone.
        drop(unsafe { Box::from_raw(cx.metrics.as_ptr()) });
    }
}

impl Context {
    pub(crate) unsafe fn new() -> Context {
        Context {
            phase: Phase::Sleep,
            metrics: NonNull::from(Box::leak(Box::new(Metrics::new()))),
            heap: ManuallyDrop::new(Heap::new()),
            root_needs_trace: true,
            gray: GrayQueue::new(),
            gray_again: Queue::new(),
            tracing: None,
            deferred: Queue::new(),
            weak: Queue::new(),
        }
    }

    #[inline]
    pub(crate) unsafe fn mutation_context<'gc>(&self) -> &Mutation<'gc> {
        unsafe { mem::transmute::<&Self, &Mutation>(self) }
    }

    #[inline]
    pub(crate) unsafe fn finalization_context<'gc>(&self) -> &Finalization<'gc> {
        unsafe { mem::transmute::<&Self, &Finalization>(self) }
    }

    #[inline]
    pub(crate) fn metrics(&self) -> &Metrics {
        // SAFETY: allocated in `new` and freed only when `self` is dropped.
        unsafe { self.metrics.as_ref() }
    }

    #[inline]
    pub(crate) fn root_barrier(&mut self) {
        if self.phase == Phase::Mark {
            self.root_needs_trace = true;
        }
    }

    #[inline]
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    #[inline]
    pub(crate) fn gray_remaining(&self) -> bool {
        !self.gray.is_empty() || !self.gray_again.is_empty() || self.root_needs_trace
    }

    // Do some collection work until either we have achieved our `target` (paying off debt or
    // finishing a full collection) or we have reached the `stop` condition.
    //
    // In order for this to be safe, at the time of call no `Gc` pointers can be live that are not
    // reachable from the given root object.
    //
    // If we are currently in `Phase::Sleep` and have positive debt, this will immediately
    // transition the collector to `Phase::Mark`.
    #[deny(unsafe_op_in_unsafe_fn)]
    // `!(debt > 0.0)` rather than `debt <= 0.0` so a NaN debt counts as paid.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub(crate) unsafe fn do_collection<'gc, R: Collect<'gc> + ?Sized>(
        &mut self,
        root: &R,
        run_until: RunUntil,
        stop: Stop,
    ) {
        let mut cx = PhaseGuard::enter(self, None);

        if run_until == RunUntil::PayDebt && !(cx.metrics().allocation_debt() > 0.0) {
            return;
        }

        let mut has_slept = false;

        loop {
            match cx.phase {
                Phase::Sleep => {
                    has_slept = true;
                    // Immediately enter the mark phase
                    cx.switch(Phase::Mark);
                }
                Phase::Mark => {
                    if cx.mark_one(root).is_break() {
                        if stop <= Stop::FullyMarked {
                            break;
                        } else {
                            // A deferred object may hold pointers that sweeping is about to free.
                            assert!(
                                cx.deferred.is_empty(),
                                "deferred objects must be cleared in finalization before sweeping"
                            );
                            // If we have no gray objects left, we enter the sweep phase.
                            cx.switch(Phase::Sweep);
                            cx.settle_weak();
                            // Allocation from here on only uses chunks already swept, so nothing
                            // allocated during the sweep is freed by it.
                            cx.heap.start_sweep(cx.metrics());
                        }
                    }
                }
                Phase::Sweep => {
                    if stop <= Stop::AtSweep {
                        break;
                    } else if cx.sweep_one().is_break() {
                        // Begin a new cycle.
                        //
                        // We reset our debt if we have done an entire collection cycle (marking and
                        // sweeping) as a single atomic unit. This keeps inherited debt from growing
                        // without bound.
                        cx.metrics().finish_cycle(has_slept);
                        cx.root_needs_trace = true;
                        cx.switch(Phase::Sleep);

                        // We treat a stop condition of `Stop::Finish` as special for the purposes
                        // of logging, and log that we finished a cycle.
                        if stop == Stop::FinishCycle {
                            return;
                        }

                        // Otherwise we always break if we have performed a full cycle as a single
                        // atomic unit, because there cannot be any more work to do in this case.
                        if has_slept {
                            // We shouldn't be stopping here if the stop condition is something like
                            // `Stop::AtSweep`, but this should be impossible since the only way to
                            // get here is to have gone through the entire cycle.
                            assert!(stop == Stop::Full);
                            break;
                        }
                    }
                }
                Phase::Drop => unreachable!(),
            }

            if run_until == RunUntil::PayDebt && !(cx.metrics().allocation_debt() > 0.0) {
                break;
            }
        }
    }

    #[inline]
    fn allocate<'gc, T: Collect<'gc>>(&self, t: T) -> NonNull<GcBoxInner<T>> {
        const { assert!(align_of::<GcBoxInner<T>>() <= CELL) };
        let size = size_of::<GcBoxInner<T>>().next_multiple_of(CELL);
        let ptr = self
            .heap
            .alloc(size, mem::needs_drop::<T>(), T::NEEDS_TRACE, self.metrics())
            .cast::<GcBoxInner<T>>();
        // SAFETY: a fresh block of `size` bytes.
        unsafe { ptr.write(GcBoxInner::new(GcBoxHeader::new::<T>(), t)) };
        self.metrics().mark_gc_allocated(size);
        ptr
    }

    /// `init` must initialize all `t.trailing_len()` bytes it is handed.
    #[inline]
    fn allocate_trailing<'gc, T: Collect<'gc> + TrailingBytes>(
        &self,
        t: T,
        init: impl FnOnce(NonNull<u8>),
    ) -> NonNull<GcBoxInner<T>> {
        const { assert!(align_of::<GcBoxInner<T>>() <= CELL) };
        let layout = trailing_layout(Layout::new::<GcBoxInner<T>>(), t.trailing_len());
        let size = layout.size().next_multiple_of(CELL);
        // `TrailingBytes` types have no drop glue.
        let ptr = self
            .heap
            .alloc(size, false, T::NEEDS_TRACE, self.metrics())
            .cast::<GcBoxInner<T>>();
        // SAFETY: a fresh block of `size` bytes. The header and value are written before `init`
        // runs, so a panic in it leaves a whole object for the sweep.
        unsafe {
            ptr.write(GcBoxInner::new(GcBoxHeader::new_trailing::<T>(), t));
            // For `RefLock::trailing_ptr_of`, which only has a reference to the value.
            ptr.expose_provenance();
            init(ptr.cast::<u8>().add(GcBoxInner::<T>::TRAILING_OFFSET));
        }
        self.metrics().mark_gc_allocated(size);
        ptr
    }

    // LuaJIT 3.0's quad-color barrier: a gray parent (light gray: new or written since it was
    // traced, or dark gray: queued) needs nothing. The child is not looked at, as in LuaJIT 2's
    // table barrier: it is almost always white anyway.
    #[inline]
    fn backward_barrier(&self, parent: GcBox, _child: Option<GcBox>) {
        if !parent.header().is_gray() {
            #[cold]
            #[inline(never)]
            fn barrier(this: &Context, parent: GcBox) {
                if !this.lighten(parent) {
                    this.make_gray_again(parent);
                }
            }
            barrier(self, parent);
        }
    }

    /// Whether the barrier on `parent` would enqueue it; when not, whatever else it does is done.
    #[inline]
    fn backward_barrier_pending(&self, parent: GcBox) -> bool {
        !parent.header().is_gray() && !self.lighten(parent)
    }

    /// Turn a white `parent` light gray. False for a black one while marking, which has to be
    /// traced again.
    #[cold]
    #[inline(never)]
    fn lighten(&self, parent: GcBox) -> bool {
        // SAFETY: a barrier is only called on a live object.
        if self.phase == Phase::Mark && unsafe { self.heap.is_marked(parent) } {
            return false;
        }
        parent.header().set_gray(true);
        true
    }

    #[inline]
    fn backward_barrier_weak(&self, parent: GcBox, _child: GcBox) {
        self.backward_barrier(parent, None);
    }

    #[inline]
    fn forward_barrier(&self, parent: Option<GcBox>, child: GcBox) {
        // During the marking phase, if we are mutating a black object, we may add a white object
        // to it and invalidate the invariant that black objects may not point to white objects.
        // Immediately trace the child white object to turn it gray (or black) to prevent this.
        if self.phase == Phase::Mark && parent.is_none_or(|p| self.is_black(p)) {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, child: GcBox) {
                this.trace(child);
            }
            barrier(self, child);
        }
    }

    #[inline]
    fn forward_barrier_weak(&self, parent: Option<GcBox>, child: GcBox) {
        // During the marking phase, if we are mutating a black object, we may add a white object
        // to it and invalidate the invariant that black objects may not point to white objects.
        // Immediately trace the child white object to turn it gray (or black) to prevent this.
        if self.phase == Phase::Mark && parent.is_none_or(|p| self.is_black(p)) {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, child: GcBox) {
                this.trace_weak(child);
            }
            barrier(self, child);
        }
    }

    #[inline]
    fn trace(&self, gc_box: GcBox) {
        // SAFETY: a traced pointer is a live object's.
        if unsafe { self.heap.is_marked(gc_box) } {
            return;
        }
        debug_assert!(gc_box.header().is_live());
        // The object itself is first read when traversed. Until then its gray bit may be clear,
        // so a write to it takes the barrier and queues it again: traversing twice is cheaper
        // than reading every object as it is found.
        if unsafe { self.heap.mark(gc_box) } {
            // SAFETY: it was just marked, so it isn't queued.
            unsafe { self.gray.push(gc_box) };
        }
    }

    #[inline]
    fn trace_weak(&self, gc_box: GcBox) {
        let header = gc_box.header();
        if !header.is_weak() && unsafe { !self.heap.is_marked(gc_box) } {
            header.set_weak(true);
            self.weak.push(gc_box);
        }
    }

    /// Settle the objects only `GcWeak`s reached, once marking is done: drop the ones still
    /// unmarked, but keep their boxes through the sweep so their weak pointers can still see
    /// that they died. A box is freed in the first cycle no weak pointer reaches it.
    fn settle_weak(&self) {
        while let Some(mut gc_box) = self.weak.pop() {
            let header = gc_box.header();
            header.set_weak(false);
            // SAFETY: the box was live when reached and the sweep has not begun.
            unsafe {
                if !self.heap.is_marked(gc_box) {
                    let size = gc_box.size();
                    if header.is_live() {
                        header.set_live(false);
                        gc_box.drop_in_place();
                        self.metrics().mark_gc_dropped(size);
                    }
                    self.heap.mark(gc_box);
                    self.metrics().mark_gc_marked(size);
                }
            }
        }
    }

    /// Determines whether or not a Gc pointer is safe to be upgraded.
    /// This is used by weak pointers to determine if it can safely upgrade to a strong pointer.
    #[inline]
    fn upgrade(&self, gc_box: GcBox) -> bool {
        // A weakly reached object that died was dropped before the sweep began (`settle_weak`).
        // Any other object a `GcWeak` points to during the sweep was reachable when marking
        // ended, or allocated since, so it survives the sweep. In the other phases, an upgraded
        // pointer that outlives the callback must have been stored, which a barrier catches.
        gc_box.header().is_live()
    }

    #[inline]
    fn resurrect(&self, gc_box: GcBox) {
        debug_assert_eq!(self.phase, Phase::Mark);
        debug_assert!(gc_box.header().is_live());
        self.trace(gc_box);
    }

    fn mark_one<'gc, R: Collect<'gc> + ?Sized>(&mut self, root: &R) -> ControlFlow<()> {
        // We look for an object first in the normal gray queue, then the "gray again" queue.
        // Processing "gray again" objects later gives them more time to be mutated again without
        // triggering another write barrier.
        let next_gray = self.gray.pop().or_else(|| self.gray_again.pop());

        if let Some(gc_box) = next_gray {
            // Every traversal counts as work, including the second one of an object a barrier
            // queued again. Marking reads no object, so its work is counted here too, where the
            // size is at hand; objects never traversed count none, as in LuaJIT.
            let size = gc_box.size();
            self.metrics().mark_gc_marked(size);
            self.metrics().mark_gc_traced(size);
            // Black before the traversal, so writes during it are caught.
            gc_box.header().set_gray(false);
            // Drop and huge chunks also hold types with nothing to trace.
            if !gc_box.header().needs_trace() {
                return ControlFlow::Continue(());
            }

            // If we have an object in the gray queue, take one, trace it, and turn it black.

            // Our `Collect::trace` call may panic, and if it does the object will be lost from
            // the gray queue but potentially incompletely traced. By catching a panic during
            // `Arena::collect()`, this could lead to memory unsafety.
            //
            // So, if the `Collect::trace` call panics, we need to add the popped object back to the
            // `gray_again` queue. If the panic is caught, this will maybe give some time for its
            // trace method to not panic before attempting to collect it again.
            struct DropGuard<'a> {
                context: &'a mut Context,
                gc_box: GcBox,
            }

            impl<'a> Drop for DropGuard<'a> {
                fn drop(&mut self) {
                    self.context.tracing = None;
                    self.context.make_gray_again(self.gc_box);
                }
            }

            let guard = DropGuard {
                context: self,
                gc_box,
            };
            debug_assert!(gc_box.header().is_live());
            guard.context.tracing = Some(gc_box);
            unsafe { gc_box.trace_value(guard.context) }
            guard.context.tracing = None;
            mem::forget(guard);

            ControlFlow::Continue(())
        } else if self.root_needs_trace {
            // We treat the root object as gray if `root_needs_trace` is set, and we process it at
            // the end of the gray queue for the same reason as the "gray again" objects.
            root.trace(self);
            self.root_needs_trace = false;
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn sweep_one(&mut self) -> ControlFlow<()> {
        if self.heap.sweep_next(self.metrics()) {
            ControlFlow::Continue(())
        } else {
            self.heap.finish_sweep();
            ControlFlow::Break(())
        }
    }

    // Take a black pointer and turn it gray and put it in the `gray_again` queue.
    fn make_gray_again(&self, gc_box: GcBox) {
        debug_assert!(self.is_black(gc_box));
        gc_box.header().set_gray(true);
        self.gray_again.push(gc_box);
    }

    /// Marked and traced, or not in need of tracing.
    #[inline(always)]
    fn is_black(&self, gc_box: GcBox) -> bool {
        // SAFETY: only called on live objects' pointers.
        !gc_box.header().is_gray() && unsafe { self.heap.is_marked(gc_box) }
    }
}

/// Helper type for managing phase transitions.
struct PhaseGuard<'a> {
    cx: &'a mut Context,
}

impl<'a> Deref for PhaseGuard<'a> {
    type Target = Context;

    #[inline(always)]
    fn deref(&self) -> &Context {
        self.cx
    }
}

impl<'a> DerefMut for PhaseGuard<'a> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Context {
        self.cx
    }
}

impl<'a> PhaseGuard<'a> {
    fn enter(cx: &'a mut Context, phase: Option<Phase>) -> Self {
        if let Some(phase) = phase {
            cx.phase = phase;
        }

        Self { cx }
    }

    fn switch(&mut self, phase: Phase) {
        self.cx.phase = phase;
    }
}

// A shared, internally mutable `Vec<T>` that avoids the overhead of `RefCell`. Used for the "gray",
// "gray again" and deferred queues.
//
// SAFETY: We do not return any references at all to the contents of the internal `UnsafeCell`, nor
// do we provide any methods with callbacks. Since this type is `!Sync`, only one reference to the
// `UnsafeCell` contents can be alive at any given time, thus we cannot violate aliasing rules.
#[derive(Default)]
struct Queue<T> {
    vec: UnsafeCell<Vec<T>>,
}

impl<T> Queue<T> {
    fn new() -> Self {
        Self {
            vec: UnsafeCell::new(Vec::new()),
        }
    }

    fn is_empty(&self) -> bool {
        unsafe { (*self.vec.get().cast_const()).is_empty() }
    }

    fn push(&self, val: T) {
        unsafe {
            (*self.vec.get()).push(val);
        }
    }

    fn pop(&self) -> Option<T> {
        unsafe { (*self.vec.get()).pop() }
    }

    fn clear(&self) {
        unsafe { (*self.vec.get()).clear() }
    }
}

impl<T: Copy + Ord> Queue<T> {
    /// Drop duplicates in place and return a copy of the rest.
    fn dedup(&self) -> Vec<T> {
        let vec = unsafe { &mut *self.vec.get() };
        vec.sort_unstable();
        vec.dedup();
        vec.clone()
    }
}
