//! Dependencies of compiled code on runtime state (ws2 §5.3, V8's
//! `DependentCode`): an object a region assumed something of lists the
//! region, weakly, with the groups of facts it assumed; a change to one of
//! them invalidates the regions listed for it (`Region::invalidate`).

use crate::dmm::{Collect, Gc, GcWeak, Mutation, RefLock};
use crate::jit::region::Region;

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Dep<'gc> {
    region: GcWeak<'gc, Region<'gc>>,
    #[collect(require_static)]
    groups: u32,
}

/// The regions depending on one object.
pub(crate) type DepList<'gc> = RefLock<Vec<Dep<'gc>>>;

/// List `region` as assuming `groups`, dropping the dead regions.
pub(crate) fn add<'gc>(
    mc: &Mutation<'gc>,
    list: Gc<'gc, DepList<'gc>>,
    region: Gc<'gc, Region<'gc>>,
    groups: u32,
) {
    let mut l = list.borrow_mut(mc);
    l.retain(|d| d.region.upgrade(mc).is_some());
    l.push(Dep {
        region: Gc::downgrade(region),
        groups,
    });
}

/// Invalidate the regions assuming any of `groups`, and drop them from the
/// list; whether there were any. Adopts nothing, so a collector's weak clear
/// may call it (`mc` its finalization's).
pub(crate) fn invalidate<'gc>(
    mc: &Mutation<'gc>,
    list: Gc<'gc, DepList<'gc>>,
    groups: u32,
) -> bool {
    let mut any = false;
    // SAFETY: only removes entries.
    unsafe { list.as_checked_cell() }
        .borrow_mut()
        .retain(|d| match d.region.upgrade(mc) {
            Some(r) if d.groups & groups != 0 => {
                r.invalidate();
                any = true;
                false
            }
            Some(_) => true,
            None => false,
        });
    any
}
