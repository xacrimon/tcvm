//! Values in a GC cell of their own, owned by one table: the table's trace marks the cell and
//! traces its values, and the table's write barrier covers stores into it. The cell has no
//! drop glue, so a table that outgrows it just leaves it for the sweep.

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
