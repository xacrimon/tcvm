//! Values in a GC cell of their own, owned by one table: the table's trace marks the cell and
//! traces its values, and the table's write barrier covers stores into it. The cell has no
//! drop glue, so a table that outgrows it just leaves it for the sweep.

use core::cell::UnsafeCell;
use core::ptr::NonNull;

use crate::dmm::{Collect, Gc, Mutation, Trace, TrailingBytes};
use crate::env::value::Value;

/// A cell's header; its `len` values follow it.
pub(super) struct Slots {
    len: u32,
}

// SAFETY: only `alloc` makes a `Slots`, through `Gc::new_with_trailing` with `len` values, and it
// has no drop glue.
unsafe impl TrailingBytes for Slots {
    #[inline(always)]
    fn trailing_len(&self) -> usize {
        self.len as usize * size_of::<Value>()
    }
}

// SAFETY: deliberately untraced; the owning table traces the values (see the module doc).
unsafe impl<'gc> Collect<'gc> for Slots {
    const NEEDS_TRACE: bool = false;
}

/// A new cell of `len` values, `f(i)` at `i`; returns its first value.
#[inline]
pub(super) fn alloc<'gc>(
    mc: &Mutation<'gc>,
    len: usize,
    f: impl Fn(usize) -> Value<'gc>,
) -> NonNull<Value<'gc>> {
    let len32 = u32::try_from(len).expect("table part too large");
    // SAFETY: the bytes are only read as the values written here.
    let cell = unsafe {
        Gc::new_with_trailing(mc, Slots { len: len32 }, |dst| {
            let dst = dst.cast::<Value<'gc>>();
            for i in 0..len {
                dst.add(i).write(f(i));
            }
        })
    };
    Gc::trailing_ptr(cell).cast()
}

/// Mark the cell `alloc` returned `values` for.
///
/// # Safety
/// `values` must come from `alloc`, and its cell must not have been collected.
#[inline]
pub(super) unsafe fn mark<'gc, T: Trace<'gc>>(cc: &mut T, values: NonNull<Value<'gc>>) {
    let cell: Gc<'gc, Slots> = unsafe { Gc::from_trailing_ptr(values.cast()) };
    cc.trace(&cell);
}

/// A value in a GC cell of its own that only one table refers to, owned the
/// same way as a `Slots` cell: the table marks it and traces what it holds.
pub(super) struct Owned<T>(UnsafeCell<T>);

// SAFETY: deliberately untraced, as for `Slots`.
unsafe impl<'gc, T: 'gc> Collect<'gc> for Owned<T> {
    const NEEDS_TRACE: bool = false;
}

impl<'gc, T: 'gc> Owned<T> {
    pub(super) fn new(mc: &Mutation<'gc>, value: T) -> Gc<'gc, Self> {
        Gc::new(mc, Owned(UnsafeCell::new(value)))
    }

    /// # Safety
    /// No `&mut` from [`Self::get_mut`] may be live; the owner's borrows
    /// stand in for the cell's.
    #[inline(always)]
    pub(super) unsafe fn get(this: Gc<'gc, Self>) -> &'gc T {
        unsafe { &*Gc::as_ref(this).0.get() }
    }

    /// # Safety
    /// No other reference into the cell may be live.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    pub(super) unsafe fn get_mut(this: Gc<'gc, Self>) -> &'gc mut T {
        unsafe { &mut *Gc::as_ref(this).0.get() }
    }
}
