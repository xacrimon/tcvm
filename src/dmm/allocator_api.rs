use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};
use std::alloc::{AllocError, Allocator, Global};

use crate::dmm::{
    Gc,
    collect::{Collect, Trace},
    context::Mutation,
    metrics::Metrics,
    types::{Invariant, TrailingBytes},
};

/// An allocator that reports its allocations to the arena's [`Metrics`] as external memory.
///
/// Holds a plain pointer to the metrics, so it is `Copy` (for a `Copy` allocator) and has no drop
/// glue; the arena frees the metrics only after every object, and so every allocator stored in one.
#[derive(Clone, Copy)]
pub struct MetricsAlloc<'gc, A = Global> {
    metrics: NonNull<Metrics>,
    allocator: A,
    _marker: Invariant<'gc>,
}

impl<'gc> MetricsAlloc<'gc> {
    #[inline]
    pub fn new(mc: &Mutation<'gc>) -> Self {
        Self::new_in(mc, Global)
    }

    /// A `MetricsAlloc` with an arbitrary branding lifetime, for storage that needs `'static`.
    ///
    /// # Safety
    /// The arena that owns `metrics` must outlive the allocator and everything allocated with it,
    /// as it does when both are stored in its objects.
    #[inline]
    pub unsafe fn from_metrics(metrics: &Metrics) -> Self {
        unsafe { Self::from_metrics_in(metrics, Global) }
    }
}

impl<'gc, A> MetricsAlloc<'gc, A> {
    #[inline]
    pub fn new_in(mc: &Mutation<'gc>, allocator: A) -> Self {
        // SAFETY: the `'gc` brand keeps the allocator inside the arena that owns the metrics.
        unsafe { Self::from_metrics_in(mc.metrics(), allocator) }
    }

    /// # Safety
    /// As for [`MetricsAlloc::from_metrics`].
    #[inline]
    pub unsafe fn from_metrics_in(metrics: &Metrics, allocator: A) -> Self {
        Self {
            metrics: NonNull::from(metrics),
            allocator,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn metrics(&self) -> &Metrics {
        // SAFETY: see `from_metrics`.
        unsafe { self.metrics.as_ref() }
    }
}

unsafe impl<'gc, A: Allocator> Allocator for MetricsAlloc<'gc, A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocator.allocate(layout)?;
        self.metrics().mark_external_allocation(layout.size());
        Ok(ptr)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            self.metrics().mark_external_deallocation(layout.size());
            self.allocator.deallocate(ptr, layout);
        }
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocator.allocate_zeroed(layout)?;
        self.metrics().mark_external_allocation(layout.size());
        Ok(ptr)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let ptr = self.allocator.grow(ptr, old_layout, new_layout)?;
            self.metrics()
                .mark_external_allocation(new_layout.size() - old_layout.size());
            Ok(ptr)
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let ptr = self.allocator.grow_zeroed(ptr, old_layout, new_layout)?;
            self.metrics()
                .mark_external_allocation(new_layout.size() - old_layout.size());
            Ok(ptr)
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let ptr = self.allocator.shrink(ptr, old_layout, new_layout)?;
            self.metrics()
                .mark_external_deallocation(old_layout.size() - new_layout.size());
            Ok(ptr)
        }
    }
}

unsafe impl<'gc, A: 'static> Collect<'gc> for MetricsAlloc<'gc, A> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc> Collect<'gc> for Global {
    const NEEDS_TRACE: bool = false;
}

/// An allocator whose memory is GC cells nothing traces: the holder of an allocation must mark
/// its cell with [`GcAlloc::mark`] whenever it is traced, and trace what it stores there itself,
/// as for a table's own fields. Deallocation does nothing; the sweep frees an unmarked cell. So a
/// structure using it needs no drop glue, and its memory is only valid while its holder is traced.
#[derive(Clone, Copy)]
pub struct GcAlloc<'gc> {
    mc: NonNull<Mutation<'gc>>,
}

/// A [`GcAlloc`] cell's header; its `len` bytes follow it, aligned to [`GcAlloc::MAX_ALIGN`].
#[repr(align(16))]
struct RawCell {
    len: usize,
}

// SAFETY: only `GcAlloc::allocate` makes a `RawCell`, through `Gc::new_with_trailing` with `len`
// bytes, and it has no drop glue.
unsafe impl TrailingBytes for RawCell {
    #[inline(always)]
    fn trailing_len(&self) -> usize {
        self.len
    }
}

// SAFETY: holds no pointers it could trace; see `GcAlloc`.
unsafe impl<'gc> Collect<'gc> for RawCell {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> GcAlloc<'gc> {
    /// The largest alignment an allocation may ask for.
    pub const MAX_ALIGN: usize = align_of::<RawCell>();

    #[inline]
    pub fn new(mc: &Mutation<'gc>) -> Self {
        Self {
            mc: NonNull::from(mc),
        }
    }

    /// Mark the cell of an allocation this allocator returned.
    ///
    /// # Safety
    /// `ptr` must be the start of a `GcAlloc` allocation that has not been collected.
    #[inline]
    pub unsafe fn mark<T: Trace<'gc>>(cc: &mut T, ptr: NonNull<u8>) {
        let cell: Gc<'gc, RawCell> = unsafe { Gc::from_trailing_ptr(ptr) };
        cc.trace(&cell);
    }
}

unsafe impl<'gc> Allocator for GcAlloc<'gc> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        assert!(layout.align() <= Self::MAX_ALIGN);
        // SAFETY: the `'gc` brand keeps the allocator inside the arena whose context `mc` points
        // to, which the arena boxes and frees only after every object, as for `MetricsAlloc`.
        let mc = unsafe { self.mc.as_ref() };
        // SAFETY: the trailing bytes are the allocation, which the caller initializes.
        let cell = unsafe { Gc::new_with_trailing(mc, RawCell { len: layout.size() }, |_| {}) };
        Ok(NonNull::slice_from_raw_parts(
            Gc::trailing_ptr(cell),
            layout.size(),
        ))
    }

    #[inline]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

unsafe impl<'gc> Collect<'gc> for GcAlloc<'gc> {
    const NEEDS_TRACE: bool = false;
}
