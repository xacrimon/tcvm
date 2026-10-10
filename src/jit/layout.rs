//! Offsets of the runtime fields compiled code reads, derived with
//! `offset_of!` (`Runtime` and `JitRuntime` are not `repr(C)`, so they are per
//! build), with tests against live objects.

use std::mem::offset_of;

use crate::dmm::Gc;
use crate::dmm::lock::RefLock;
use crate::env::table::TableState;
use crate::jit::state::JitRuntime;
use crate::lua::State;
use crate::vm::dispatch::Runtime;

/// `State.jit` from the `rt` register.
const JIT: usize = offset_of!(State<'static>, jit);
/// The pointer to the exit register image.
pub(crate) const EXIT_REGS: usize = JIT + offset_of!(JitRuntime, exit_regs);
/// The pointer to the arena's `Metrics`.
pub(crate) const METRICS: usize =
    offset_of!(State<'static>, rt) + offset_of!(Runtime<'static>, metrics);
/// The allocation counter and threshold pair inside `Metrics`.
pub(crate) const GC_CHECK: usize = crate::dmm::metrics::Metrics::GC_CHECK_OFFSET;

/// A table's `TableState` from its lock (a table value's payload).
const T_STATE: usize = RefLock::<TableState<'static>>::TRAILING_FROM_LOCK
    - RefLock::<TableState<'static>>::TRAILING_FROM_INNER;
pub(crate) const T_SHAPE: usize = T_STATE + TableState::SHAPE_OFFSET;
pub(crate) const T_SPILL: usize = T_STATE + TableState::SPILL_OFFSET;
pub(crate) const T_SPILL_CAP: usize = T_STATE + TableState::SPILL_CAP_OFFSET;
pub(crate) const T_ARRAY: usize = T_STATE + TableState::ARRAY_OFFSET;
pub(crate) const T_ASIZE: usize = T_STATE + TableState::ASIZE_OFFSET;
/// A table's GC header (bit 0: gray, no barrier needed) before its lock.
pub(crate) const T_HEADER: i32 =
    -(Gc::<'static, RefLock<TableState<'static>>>::VALUE_OFFSET as i32);

// A header word is a `u64` slot of the value stack.
const _: () = assert!(size_of::<crate::env::value::Value<'static>>() == 8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lua;

    #[test]
    fn offsets_match_reality() {
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let base = ctx.jit() as *const JitRuntime as usize;
            let state = (ctx.jit() as *const JitRuntime as usize) - JIT;
            assert_eq!(base - state, JIT);
            let regs = unsafe { ((state + EXIT_REGS) as *const *mut u64).read() };
            assert_eq!(regs, ctx.jit().exit_regs);
            let metrics = unsafe { ((state + METRICS) as *const usize).read() };
            assert_ne!(metrics, 0);
        });
    }

    #[test]
    fn table_fields() {
        use crate::env::value::Value;
        use crate::env::{MetamethodBits, MtCache};
        let mut lua = Lua::new();
        lua.load_all();
        let ex = lua
            .try_enter(|ctx| {
                let src = "local t = setmetatable({10, 20, 30, x = 1}, {__index = {}}) return t";
                Ok::<_, crate::LoadError>(ctx.stash(crate::Executor::start(
                    ctx,
                    ctx.load(src, None)?,
                    (),
                )))
            })
            .unwrap();
        lua.finish(&ex).unwrap();
        lua.enter(|ctx| {
            let v: Value = ctx.fetch(&ex).take_result(ctx).unwrap();
            let t = v.get_table().unwrap();
            let lock = (v.to_raw() & ((1 << 48) - 1)) as usize;
            let word = |off: usize| unsafe { ((lock + off) as *const u64).read() };
            let shape = t.shape();
            assert_eq!(word(T_SHAPE), shape.as_word());
            let asize = unsafe { ((lock + T_ASIZE) as *const u32).read() } as usize;
            assert!(asize >= 4);
            let arr = word(T_ARRAY) as *const Value;
            assert_eq!(unsafe { arr.add(2).read() }.get_small(), Some(20));
            let class: MtCache = shape.mt_cache().unwrap();
            let cw = class.as_word() as usize;
            let bits = unsafe { ((cw + MtCache::BITS_OFFSET) as *const u32).read() };
            assert!(bits & MetamethodBits::INDEX.bits() != 0);
            let index = unsafe { ((cw + MtCache::INDEX_OFFSET) as *const Value).read() };
            assert!(index.get_table().is_some());
            let header = unsafe { ((lock as isize + T_HEADER as isize) as *const u64).read() };
            assert_eq!(header & 1 != 0, Gc::is_gray(t.inner()));
        });
    }

    #[test]
    fn upvalue_cell_value_pointer() {
        use crate::dmm::Gc;
        use crate::env::function::UpvalueCell;
        use crate::env::value::Value;
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let cell = UpvalueCell::new_closed(ctx.mutation(), Value::small(7));
            // What an upvalue slot holds: the cell's box.
            let boxed = Gc::erased_ptr(cell) as usize;
            let v = unsafe { ((boxed + UpvalueCell::V_OFFSET) as *const *const Value).read() };
            assert_eq!(unsafe { v.read() }.get_small(), Some(7));
        });
    }
}
