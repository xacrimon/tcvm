use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::{mem, ptr};

use crate::dmm::{Gc, collect::Collect, context::Context, heap::CELL};

/// A thin-pointer-sized box containing a type-erased GC object.
/// Stores the metadata required by the GC algorithm inline (see `GcBoxInner`
/// for its typed counterpart).

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct GcBox(NonNull<GcBoxInner<()>>);

impl GcBox {
    /// Erases a pointer to a typed GC object.
    ///
    /// **SAFETY:** The pointer must point to a valid `GcBoxInner` allocated by the heap.
    #[inline(always)]
    pub(crate) unsafe fn erase<T: ?Sized>(ptr: NonNull<GcBoxInner<T>>) -> Self {
        // This cast is sound because `GcBoxInner` is `repr(C)`.
        let erased = ptr.as_ptr() as *mut GcBoxInner<()>;
        unsafe { Self(NonNull::new_unchecked(erased)) }
    }

    /// The box starting at `p`.
    ///
    /// **SAFETY:** `p` must start a block holding an initialized `GcBoxInner`.
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(p: *mut u8) -> Self {
        unsafe { Self(NonNull::new_unchecked(p.cast())) }
    }

    #[inline(always)]
    pub(crate) fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr().cast()
    }

    /// Gets a pointer to the value stored inside this box.
    /// `T` must be the same type that was used with `erase`, so that
    /// we can correctly compute the field offset.
    #[inline(always)]
    fn unerased_value<T>(&self) -> *mut T {
        unsafe {
            let ptr = self.0.as_ptr() as *mut GcBoxInner<T>;
            // Don't create a reference, to keep the full provenance.
            // Also, this gives us interior mutability "for free".
            ptr::addr_of_mut!((*ptr).value) as *mut T
        }
    }

    /// # Safety
    /// `'gc` must be the branding lifetime of the arena that owns this box.
    #[inline(always)]
    pub(crate) unsafe fn as_gc<'gc>(self) -> Gc<'gc, ()> {
        Gc {
            ptr: self.0,
            _invariant: PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn header(&self) -> &GcBoxHeader {
        unsafe { &self.0.as_ref().header }
    }

    /// Traces the stored value.
    ///
    /// **SAFETY**: `Self::drop_in_place` must not have been called.
    #[inline(always)]
    pub(crate) unsafe fn trace_value(&self, cc: &mut Context) {
        unsafe { (self.header().vtable().trace_value)(*self, cc) }
    }

    /// Drops the stored value.
    ///
    /// **SAFETY**: once called, no GC pointers should access the stored value
    /// (but accessing the `GcBox` itself is still safe).
    #[inline(always)]
    pub(crate) unsafe fn drop_in_place(&mut self) {
        unsafe { (self.header().vtable().drop_value)(*self) }
    }

    /// The (shallow) size occupied by this box in memory, trailing bytes included, in whole cells.
    #[inline(always)]
    pub(crate) fn size(&self) -> usize {
        let vtable = self.header().vtable();
        let size = match vtable.trailing_len {
            None => vtable.box_layout.size(),
            // SAFETY: only `TrailingBytes` types have a `trailing_len`, and those have no drop
            // glue, so the length is readable even after `drop_in_place`. The sum can't overflow:
            // the box was allocated with it.
            Some(len) => vtable.box_layout.size() + unsafe { len(*self) },
        };
        size.next_multiple_of(CELL)
    }
}

/// The layout of a box whose value is followed by `len` bytes; `base` is the layout without them.
/// [`GcBox::size`] computes the same size unchecked.
pub(crate) fn trailing_layout(base: Layout, len: usize) -> Layout {
    base.size()
        .checked_add(len)
        .and_then(|size| Layout::from_size_align(size, base.align()).ok())
        .expect("trailing bytes too large to allocate")
        .pad_to_align()
}

/// A type that [`Gc::new_with_bytes`](crate::dmm::Gc::new_with_bytes) allocates with a run of
/// bytes directly after it, so a variable-length payload shares its object's allocation.
///
/// # Safety
/// Values must only ever be allocated by `Gc::new_with_bytes` (keep the constructor private),
/// `trailing_len` must return the length they were allocated with, and the type must have no drop
/// glue: the collector reads the length to free the box after dropping the value.
pub unsafe trait TrailingBytes {
    fn trailing_len(&self) -> usize;
}

pub(crate) struct GcBoxHeader {
    /// A custom virtual function table for handling type-specific operations.
    ///
    /// The lower bits of the pointer are used to store GC flags:
    /// - bit 0 (`GRAY`): writes need no barrier. Set at birth (light gray), by the first write
    ///   since the object was last traced (light gray again), and while it waits to be traced
    ///   (dark gray, marked); cleared when traced, leaving it black;
    /// - bit 1 (`WEAK`): reached only through a `GcWeak` so far this cycle;
    /// - bit 2 for the `needs_trace` flag;
    /// - bit 3 for the `is_live` flag.
    tagged_vtable: Cell<*const CollectVtable>,
}

impl GcBoxHeader {
    /// A header for a freshly allocated, live `T`.
    #[inline(always)]
    pub fn new<'gc, T: Collect<'gc>>() -> Self {
        // Helper trait to materialize vtables in static memory.
        trait HasCollectVtable {
            const VTABLE: CollectVtable;
        }

        impl<'gc, T: Collect<'gc>> HasCollectVtable for T {
            const VTABLE: CollectVtable = CollectVtable::vtable_for::<T>();
        }

        Self::with_vtable(&<T as HasCollectVtable>::VTABLE, T::NEEDS_TRACE)
    }

    /// Like [`GcBoxHeader::new`], for a value allocated with trailing bytes.
    #[inline(always)]
    pub fn new_trailing<'gc, T: Collect<'gc> + TrailingBytes>() -> Self {
        trait HasTrailingVtable {
            const VTABLE: CollectVtable;
        }

        impl<'gc, T: Collect<'gc> + TrailingBytes> HasTrailingVtable for T {
            const VTABLE: CollectVtable = CollectVtable::vtable_for_trailing::<T>();
        }

        Self::with_vtable(&<T as HasTrailingVtable>::VTABLE, T::NEEDS_TRACE)
    }

    #[inline(always)]
    fn with_vtable(vtable: &'static CollectVtable, needs_trace: bool) -> Self {
        let flags = GRAY | LIVE | if needs_trace { NEEDS_TRACE } else { 0 };
        Self {
            tagged_vtable: Cell::new((vtable as *const CollectVtable).map_addr(|a| a | flags)),
        }
    }

    /// Gets a reference to the `CollectVtable` used by this box.
    #[inline(always)]
    fn vtable(&self) -> &'static CollectVtable {
        let ptr = tagged_ptr::untag(self.tagged_vtable.get());
        // SAFETY:
        // - the pointer was properly untagged.
        // - the vtable is stored in static memory.
        unsafe { &*ptr }
    }

    #[inline(always)]
    pub(crate) fn is_gray(&self) -> bool {
        tagged_ptr::get::<GRAY, _>(self.tagged_vtable.get()) != 0
    }

    #[inline(always)]
    pub(crate) fn set_gray(&self, gray: bool) {
        tagged_ptr::set_bool::<GRAY, _>(&self.tagged_vtable, gray);
    }

    #[inline(always)]
    pub(crate) fn is_weak(&self) -> bool {
        tagged_ptr::get::<WEAK, _>(self.tagged_vtable.get()) != 0
    }

    #[inline(always)]
    pub(crate) fn set_weak(&self, weak: bool) {
        tagged_ptr::set_bool::<WEAK, _>(&self.tagged_vtable, weak);
    }

    #[inline]
    pub(crate) fn needs_trace(&self) -> bool {
        tagged_ptr::get::<NEEDS_TRACE, _>(self.tagged_vtable.get()) != 0
    }

    /// Determines whether or not we've dropped the `dyn Collect` value
    /// stored in `GcBox.value`
    /// When we garbage-collect a `GcBox` that still has outstanding weak pointers,
    /// we set `alive` to false. When there are no more weak pointers remaining,
    /// we will deallocate the `GcBox`, but skip dropping the `dyn Collect` value
    /// (since we've already done it).
    #[inline]
    pub(crate) fn is_live(&self) -> bool {
        tagged_ptr::get::<LIVE, _>(self.tagged_vtable.get()) != 0
    }

    #[inline]
    pub(crate) fn set_live(&self, alive: bool) {
        tagged_ptr::set_bool::<LIVE, _>(&self.tagged_vtable, alive);
    }
}

const GRAY: usize = 0x1;
const WEAK: usize = 0x2;
const NEEDS_TRACE: usize = 0x4;
const LIVE: usize = 0x8;

/// Type-specific operations for GC'd values.
///
/// We use a custom vtable instead of `dyn Collect` for extra flexibility.
/// The type is over-aligned so that `GcBoxHeader` can store flags into the LSBs of the vtable pointer.
#[repr(align(16))]
struct CollectVtable {
    /// The layout of the `GcBox` the GC'd value is stored in, without any trailing bytes.
    box_layout: Layout,
    /// Reads the length of the value's trailing bytes, for `TrailingBytes` types.
    trailing_len: Option<unsafe fn(GcBox) -> usize>,
    /// Drops the value stored in the given `GcBox` (without deallocating the box).
    drop_value: unsafe fn(GcBox),
    /// Traces the value stored in the given `GcBox`.
    trace_value: unsafe fn(GcBox, &mut Context),
}

impl CollectVtable {
    /// Makes a vtable for a known, `Sized` type.
    /// Because `T: Sized`, we can recover a typed pointer
    /// directly from the erased `GcBox`.
    #[inline(always)]
    const fn vtable_for<'gc, T: Collect<'gc>>() -> Self {
        Self {
            box_layout: Layout::new::<GcBoxInner<T>>(),
            trailing_len: None,
            drop_value: |erased| unsafe {
                ptr::drop_in_place(erased.unerased_value::<T>());
            },
            trace_value: |erased, cc| unsafe {
                let val = &*(erased.unerased_value::<T>());
                val.trace(cc)
            },
        }
    }

    #[inline(always)]
    const fn vtable_for_trailing<'gc, T: Collect<'gc> + TrailingBytes>() -> Self {
        assert!(
            !mem::needs_drop::<T>(),
            "`TrailingBytes` types must not need drop"
        );
        Self {
            trailing_len: Some(|erased| unsafe { (*erased.unerased_value::<T>()).trailing_len() }),
            ..Self::vtable_for::<T>()
        }
    }
}

/// A typed GC'd value, together with its metadata.
/// This type is never manipulated directly by the GC algorithm, allowing
/// user-facing `Gc`s to freely cast their pointer to it.
#[repr(C)]
pub(crate) struct GcBoxInner<T: ?Sized> {
    pub(crate) header: GcBoxHeader,
    /// The typed value stored in this `GcBox`.
    pub(crate) value: mem::ManuallyDrop<T>,
}

impl<'gc, T: Collect<'gc>> GcBoxInner<T> {
    #[inline(always)]
    pub(crate) fn new(header: GcBoxHeader, t: T) -> Self {
        Self {
            header,
            value: mem::ManuallyDrop::new(t),
        }
    }
}

impl<T> GcBoxInner<T> {
    /// Offset of a `TrailingBytes` value's bytes from the start of its box: right after the box
    /// itself, so the allocation is exactly `trailing_layout(Layout::new::<Self>(), len)`.
    pub(crate) const TRAILING_OFFSET: usize = mem::size_of::<Self>();

    /// Offset of a `TrailingBytes` value's bytes from the value.
    pub(crate) const TRAILING_FROM_VALUE: usize =
        Self::TRAILING_OFFSET - mem::offset_of!(Self, value);
}

// Phantom type that holds a lifetime and ensures that it is invariant.
pub(crate) type Invariant<'a> = PhantomData<Cell<&'a ()>>;

/// Utility functions for tagging and untagging pointers.
mod tagged_ptr {
    use core::cell::Cell;

    trait ValidMask<const MASK: usize> {
        const CHECK: ();
    }

    impl<T, const MASK: usize> ValidMask<MASK> for T {
        const CHECK: () = assert!(MASK < core::mem::align_of::<T>());
    }

    /// Checks that `$mask` can be used to tag a pointer to `$type`.
    /// If this isn't true, this macro will cause a post-monomorphization error.
    macro_rules! check_mask {
        ($type:ty, $mask:expr) => {
            let _ = <$type as ValidMask<$mask>>::CHECK;
        };
    }

    #[inline(always)]
    pub(super) fn untag<T>(tagged_ptr: *const T) -> *const T {
        let mask = core::mem::align_of::<T>() - 1;
        tagged_ptr.map_addr(|addr| addr & !mask)
    }

    #[inline(always)]
    pub(super) fn get<const MASK: usize, T>(tagged_ptr: *const T) -> usize {
        check_mask!(T, MASK);
        tagged_ptr.addr() & MASK
    }

    #[inline(always)]
    pub(super) fn set_bool<const MASK: usize, T>(pcell: &Cell<*const T>, value: bool) {
        check_mask!(T, MASK);
        let ptr = pcell.get();
        let ptr = ptr.map_addr(|addr| (addr & !MASK) | if value { MASK } else { 0 });
        pcell.set(ptr)
    }
}
