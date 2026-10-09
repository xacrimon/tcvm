//! The method JIT (`jit-design.md`): regions compiled from a prototype's
//! function entry or a hot loop header, entered and left as dispatch targets.

pub(crate) mod feedback;
pub(crate) mod region;
pub(crate) mod state;

use crate::vm::abi::{Handler, Slot, handler};

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// A compiled function entry: tail into its region.
    op fn op_jit_entry {
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A compiled loop entry.
    op fn op_jit_loop {
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A counting instruction's counter ran out: compile its entry, or run
    /// the instruction. `insn` is the counting word, `pc` past it, and the
    /// frame stands exactly at it.
    slow fn jit_hot {
        let word = insn.as_insn();
        let jit = rt.jit();
        let counter = &jit.hot[word.hot_counter()];
        if !jit.config.enabled {
            counter.set(u16::MAX);
        } else {
            counter.set(jit.hot_start(word));
        }
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word))
    }
}

/// `code` as the dispatch target it is.
///
/// # Safety
/// `code` is the entry of a region or a handler.
#[inline(always)]
pub(crate) unsafe fn as_handler(code: *const u8) -> Handler {
    unsafe { std::mem::transmute::<*const u8, Handler>(code) }
}
