//! Calls. The callee and its arguments go to the callee's slots, the call
//! ends the block, and its results are the resume block's.

use super::Builder;
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::ir::ops::{ExitKind, ExitTag, Op};
use crate::jit::ir::types::Rep;
use crate::jit::ir::{BlockFlags, HOME, Val};

impl Builder<'_, '_> {
    /// `R[a], ... = R[a](R[a + 4], ...)`; a fused form's callee is `R[e]`.
    pub(super) fn call(&mut self, pc: u32, i: Instruction) {
        let (a, nb, c) = i.abc();
        // Arguments or results up to `top` have no code yet.
        if nb == 0 || c == 0 {
            return self.deopt(ExitTag::Unsupported);
        }
        let fused = matches!(i.op(), Bc::CALLS | Bc::CALLS_R0 | Bc::CALLS_R1);
        let callee = self.reg(if fused { i.e() as u8 } else { a });
        self.e
            .emit(Op::SlotStore, Rep::None, a as u16, callee.0, 0, 0);
        for r in a as u16 + 4..a as u16 + 3 + nb as u16 {
            let v = self.reg(r as u8);
            self.e.emit(Op::SlotStore, Rep::None, r, v.0, 0, 0);
        }
        // What the frame reads after the call, which writes from `R[a]` up:
        // kept in the native frame, or in its slot where that holds it.
        let entries: Vec<(u32, Val)> = (self.live(pc + 1).iter())
            .filter(|&r| r < a as usize && !self.reg(r as u8).is_none())
            .map(|r| {
                let v = self.reg(r as u8);
                let home = self.e.slot_holds(r as u16, v);
                (r as u32 | if home { HOME } else { 0 }, v)
            })
            .collect();
        let state = (self.e).frame_state(pc + 1, ExitKind::Before, Val::NONE, &entries);
        let imm = a as u32 | (nb as u32 - 1) << 8 | (c as u32) << 16;
        let lp = self.e.g.block(self.e.cur).lp;
        let resume = self.e.new_block();
        self.e
            .emit(Op::CallEntry, Rep::None, 0, state.0, resume.0, imm);
        self.e.begin(resume, lp, 0, BlockFlags::RESUME);
        // One result comes in a register; more land in their slots.
        let rep = if c == 2 { Rep::Val } else { Rep::None };
        let first = self.e.emit(Op::Resume, rep, 0, imm, 0, 0);
        if c == 2 {
            self.set(a, first);
        } else {
            for r in a..a + c - 1 {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.set(r, v);
            }
        }
        self.state = Val::NONE;
        // The callee may have broken what the region assumes.
        self.lazy_exit(pc + 1);
    }
}
