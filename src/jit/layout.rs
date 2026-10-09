//! Offsets of the runtime fields compiled code reads, derived with
//! `offset_of!` (`Runtime` and `JitRuntime` are not `repr(C)`, so they are per
//! build), with tests against live objects.

use std::mem::offset_of;

use crate::env::thread::ThreadState;
use crate::jit::state::JitRuntime;
use crate::lua::State;
use crate::vm::dispatch::Runtime;

/// `State.jit` from the `rt` register.
const JIT: usize = offset_of!(State<'static>, jit);
/// The pointer to the exit register image.
pub(crate) const EXIT_REGS: usize = JIT + offset_of!(JitRuntime, exit_regs);
/// `JitRuntime.epoch` (a `u32`).
pub(crate) const EPOCH: usize = JIT + offset_of!(JitRuntime, epoch);
/// The pointer to the arena's `Metrics`.
pub(crate) const METRICS: usize =
    offset_of!(State<'static>, rt) + offset_of!(Runtime<'static>, metrics);
/// The allocation counter and threshold pair inside `Metrics`.
pub(crate) const GC_CHECK: usize = crate::dmm::metrics::Metrics::GC_CHECK_OFFSET;
/// `ThreadState.stack_end` from the `thread` register.
pub(crate) const STACK_END: usize = offset_of!(ThreadState<'static>, stack_end);
pub(crate) const TOP: usize = offset_of!(ThreadState<'static>, top);

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
}
