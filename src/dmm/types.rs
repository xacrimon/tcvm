use core::alloc::Layout;
use core::cell::Cell;
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::{mem, ptr};

use crate::dmm::{collect::Collect, context::Context};

/// A thin-pointer-sized box containing a type-erased GC object.
/// Stores the metadata required by the GC algorithm inline (see `GcBoxInner`
/// for its typed counterpart).

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct GcBox(NonNull<GcBoxInner<()>>);

impl GcBox {
    /// Erases a pointer to a typed GC object.
    ///
    /// **SAFETY:** The pointer must point to a valid `GcBoxInner` allocated
    /// in a `Box`.
    #[inline(always)]
    pub(crate) unsafe fn erase<T: ?Sized>(ptr: NonNull<GcBoxInner<T>>) -> Self {
        // This cast is sound because `GcBoxInner` is `repr(C)`.
        let erased = ptr.as_ptr() as *mut GcBoxInner<()>;
        unsafe { Self(NonNull::new_unchecked(erased)) }
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

    /// Deallocates the box. Failing to call `Self::drop_in_place` beforehand
    /// will cause the stored value to be leaked.
    ///
    /// **SAFETY**: once called, this `GcBox` should never be accessed by any GC
    /// pointers again, and `size` must be `self.size()`.
    #[inline(always)]
    pub(crate) unsafe fn dealloc(self, size: usize) {
        unsafe {
            let align = self.header().vtable().box_layout.align();
            let ptr = self.0.as_ptr() as *mut u8;
            // SAFETY: the pointer was allocated with this size and alignment.
            std::alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, align));
        }
    }

    /// The (shallow) size occupied by this box in memory, trailing bytes included.
    #[inline(always)]
    pub(crate) fn size(&self) -> usize {
        let vtable = self.header().vtable();
        match vtable.trailing_len {
            None => vtable.box_layout.size(),
            // SAFETY: only `TrailingBytes` types have a `trailing_len`, and those have no drop
            // glue, so the length is readable even after `drop_in_place`. The sum can't overflow:
            // the box was allocated with it.
            Some(len) => (vtable.box_layout.size() + unsafe { len(*self) })
                .next_multiple_of(vtable.box_layout.align()),
        }
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
    /// The next element in the global linked list of allocated objects.
    next: Cell<Option<GcBox>>,
    /// A custom virtual function table for handling type-specific operations.
    ///
    /// The lower bits of the pointer are used to store GC flags:
    /// - bits 0 & 1 for the current `GcColor`;
    /// - bit 2 for the `needs_trace` flag;
    /// - bit 3 for the `is_live` flag.
    tagged_vtable: Cell<*const CollectVtable>,
}

impl GcBoxHeader {
    #[inline(always)]
    pub fn new<'gc, T: Collect<'gc>>() -> Self {
        // Helper trait to materialize vtables in static memory.
        trait HasCollectVtable {
            const VTABLE: CollectVtable;
        }

        impl<'gc, T: Collect<'gc>> HasCollectVtable for T {
            const VTABLE: CollectVtable = CollectVtable::vtable_for::<T>();
        }

        Self::with_vtable(&<T as HasCollectVtable>::VTABLE)
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

        Self::with_vtable(&<T as HasTrailingVtable>::VTABLE)
    }

    #[inline(always)]
    fn with_vtable(vtable: &'static CollectVtable) -> Self {
        Self {
            next: Cell::new(None),
            tagged_vtable: Cell::new(vtable as *const _),
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

    /// Gets the next element in the global linked list of allocated objects.
    #[inline(always)]
    pub(crate) fn next(&self) -> Option<GcBox> {
        self.next.get()
    }

    /// Sets the next element in the global linked list of allocated objects.
    #[inline(always)]
    pub(crate) fn set_next(&self, next: Option<GcBox>) {
        self.next.set(next)
    }

    #[inline]
    pub(crate) fn color(&self) -> GcColor {
        match tagged_ptr::get::<0x3, _>(self.tagged_vtable.get()) {
            0x0 => GcColor::White,
            0x1 => GcColor::WhiteWeak,
            0x2 => GcColor::Gray,
            _ => GcColor::Black,
        }
    }

    #[inline]
    pub(crate) fn set_color(&self, color: GcColor) {
        tagged_ptr::set::<0x3, _>(
            &self.tagged_vtable,
            match color {
                GcColor::White => 0x0,
                GcColor::WhiteWeak => 0x1,
                GcColor::Gray => 0x2,
                GcColor::Black => 0x3,
            },
        );
    }
    #[inline]
    pub(crate) fn needs_trace(&self) -> bool {
        tagged_ptr::get::<0x4, _>(self.tagged_vtable.get()) != 0x0
    }

    /// Determines whether or not we've dropped the `dyn Collect` value
    /// stored in `GcBox.value`
    /// When we garbage-collect a `GcBox` that still has outstanding weak pointers,
    /// we set `alive` to false. When there are no more weak pointers remaining,
    /// we will deallocate the `GcBox`, but skip dropping the `dyn Collect` value
    /// (since we've already done it).
    #[inline]
    pub(crate) fn is_live(&self) -> bool {
        tagged_ptr::get::<0x8, _>(self.tagged_vtable.get()) != 0x0
    }

    #[inline]
    pub(crate) fn set_needs_trace(&self, needs_trace: bool) {
        tagged_ptr::set_bool::<0x4, _>(&self.tagged_vtable, needs_trace);
    }

    #[inline]
    pub(crate) fn set_live(&self, alive: bool) {
        tagged_ptr::set_bool::<0x8, _>(&self.tagged_vtable, alive);
    }
}

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
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum GcColor {
    /// An object that has not yet been reached by tracing (if we're in a tracing phase).
    ///
    /// During `Phase::Sweep`, we will free all white objects that existed *before* the start of the
    /// current `Phase::Sweep`. Objects allocated during `Phase::Sweep` will be white, but will not
    /// be freed.
    White,
    /// Like White, but for objects weakly reachable from a Black object.
    ///
    /// These objects may drop their contents during `Phase::Sweep`, but must stay allocated so that
    /// weak references can check the alive status.
    WhiteWeak,
    /// An object reachable from a Black object, but that has not yet been traced using
    /// `Collect::trace`. We also mark black objects as gray during `Phase::Mark` in response to
    /// a write barrier, so that we re-trace and find any objects newly reachable from the mutated
    /// object.
    Gray,
    /// An object that was reached during tracing. It will not be freed during `Phase::Sweep`. At
    /// the end of `Phase::Sweep`, all black objects will be reset to white.
    Black,
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
    pub(super) fn set<const MASK: usize, T>(pcell: &Cell<*const T>, tag: usize) {
        check_mask!(T, MASK);
        let ptr = pcell.get();
        let ptr = ptr.map_addr(|addr| (addr & !MASK) | (tag & MASK));
        pcell.set(ptr)
    }

    #[inline(always)]
    pub(super) fn set_bool<const MASK: usize, T>(pcell: &Cell<*const T>, value: bool) {
        check_mask!(T, MASK);
        let ptr = pcell.get();
        let ptr = ptr.map_addr(|addr| (addr & !MASK) | if value { MASK } else { 0 });
        pcell.set(ptr)
    }
}
