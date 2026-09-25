use core::{alloc::Layout, marker::PhantomData, ptr::NonNull};
use std::alloc::{AllocError, Allocator, Global};

use crate::dmm::{collect::Collect, context::Mutation, metrics::Metrics, types::Invariant};

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
