//! Dependencies of compiled code on runtime state (ws2 §5.3, V8's
//! `DependentCode`): an object a region assumed something of lists the
//! region, weakly, with the groups of facts it assumed; a change to one of
//! them invalidates the regions listed for it (`Region::invalidate`).

use crate::dmm::{Collect, Gc, GcWeak, Lock, Mutation, RefLock};
use crate::jit::region::Region;

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Dep<'gc> {
    region: GcWeak<'gc, Region<'gc>>,
    #[collect(require_static)]
    groups: u64,
}

/// The regions depending on one object.
pub(crate) type DepList<'gc> = RefLock<Vec<Dep<'gc>>>;

/// Where an object keeps its list, made by the first dependency.
pub(crate) type DepSlot<'gc> = Lock<Option<Gc<'gc, DepList<'gc>>>>;

/// A watched holder's group for its shape: its keys, their slots, its
/// metatable (ws2 §5.1).
pub(crate) const SHAPE_GROUP: u64 = 1 << 63;

/// A watched holder's group for the value of `slot`; slots 63 apart share
/// one, as they share a mutable bit (`Shape::slot_mutable`).
pub(crate) fn slot_group(slot: u32) -> u64 {
    1 << (slot % 63)
}

/// List `region` as assuming `groups` of `owner`, whose list `slot` holds.
pub(crate) fn add<'gc, T: Collect<'gc>>(
    mc: &Mutation<'gc>,
    owner: Gc<'gc, T>,
    slot: &DepSlot<'gc>,
    region: Gc<'gc, Region<'gc>>,
    groups: u64,
) {
    let list = match slot.get() {
        Some(l) => l,
        None => {
            let l = Gc::new(mc, RefLock::new(Vec::new()));
            // Adopting a fresh `Gc` through the lock: barrier first.
            mc.backward_barrier(Gc::erase(owner), None);
            unsafe { slot.as_cell() }.set(Some(l));
            l
        }
    };
    let mut l = list.borrow_mut(mc);
    l.retain(|d| d.region.upgrade(mc).is_some());
    l.push(Dep {
        region: Gc::downgrade(region),
        groups,
    });
}

/// Invalidate the regions in `slot`'s list assuming any of `groups`, and
/// drop them from it; whether there were any. Adopts nothing, so a
/// collector's weak clear may call it (`mc` its finalization's).
pub(crate) fn invalidate<'gc>(mc: &Mutation<'gc>, slot: &DepSlot<'gc>, groups: u64) -> bool {
    let Some(list) = slot.get() else {
        return false;
    };
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
