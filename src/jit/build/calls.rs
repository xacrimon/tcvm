//! Calls. The callee and its arguments go to the callee's slots, the call
//! ends the block, and its results are the resume block's.

use super::Builder;
use crate::instruction::UpvalSource;
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Helper, Op};
use crate::jit::ir::table::Key;
use crate::jit::ir::types::Rep;
use crate::jit::ir::{BlockFlags, HOME, Val};

impl Builder<'_, '_> {
    /// `R[a] = closure(P[d])`: the helper reads what it captures from the
    /// frame's slots, and marks the frame as having open upvalues.
    pub(super) fn closure_new(&mut self, pc: u32, i: Instruction) {
        let (a, proto) = (i.a(), i.d());
        let child = self.proto.prototypes[proto as usize];
        for d in child.upvalue_desc.iter() {
            match d.source {
                UpvalSource::ParentLocal(r) if !(d.by_value && r == a) => {
                    let v = self.reg(r);
                    self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
                }
                _ => {}
            }
        }
        let f = (self.e).call(Helper::Closure, proto as u32 | (a as u32) << 16, &[]);
        self.set(a, f);
        self.gc_check(pc);
    }

    /// `CLOSE`: the cells of the registers from `R[a]` up. A `__close` runs
    /// Lua: the interpreter does it.
    pub(super) fn close(&mut self, i: Instruction) {
        let tbc = self.frame_flags(crate::vm::frame::flag::HAS_TBC);
        let zero = self.e.k64(Rep::I64, 0);
        let none = self.e.cmp(Op::Cmp, Cc::Eq, tbc, zero);
        let state = self.state();
        let tag = ExitTag::Unsupported as u16;
        self.e
            .emit(Op::Guard, Rep::None, tag, none.0, state.0, u32::MAX);
        self.e.call(Helper::Close, i.a() as u32, &[]);
    }

    /// `R[a], ... = R[a](R[a + 4], ...)`; a fused form's callee is `R[e]`.
    pub(super) fn call(&mut self, pc: u32, i: Instruction) {
        let (a, nb, c) = i.abc();
        let fused = matches!(i.op(), Bc::CALLS | Bc::CALLS_R0 | Bc::CALLS_R1);
        let callee = self.reg(if fused { i.e() as u8 } else { a });
        self.e
            .emit(Op::SlotStore, Rep::None, a as u16, callee.0, 0, 0);
        // Arguments up to `top` (`nb == 0`): those before the ones a call
        // landed there, which have no value here.
        let end = match nb {
            0 => self.cfg.nregs as u16,
            _ => a as u16 + 3 + nb as u16,
        };
        for r in a as u16 + 4..end {
            let v = self.reg(r as u8);
            if !v.is_none() {
                self.e.emit(Op::SlotStore, Rep::None, r, v.0, 0, 0);
            }
        }
        // What the frame reads after the call, which writes from `R[a]` up:
        // kept in the native frame, or in its slot where that holds it.
        // A captured register's slot is its cell's, read or not.
        let captured = self.cfg.captured;
        for r in captured.iter().filter(|&r| r < a as usize) {
            if self.reg(r as u8).is_none() {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.e.table.set(Key::reg(r as u32), v.0);
            }
        }
        let mut kept = self.live(pc + 1);
        kept.union(&captured);
        let entries: Vec<(u32, Val)> = (kept.iter())
            .filter(|&r| r < a as usize && !self.reg(r as u8).is_none())
            .map(|r| {
                let v = self.reg(r as u8);
                let home = captured.contains(r) || self.e.slot_holds(r as u16, v);
                (r as u32 | if home { HOME } else { 0 }, v)
            })
            .collect();
        let state = (self.e).frame_state(pc + 1, ExitKind::Before, Val::NONE, &entries);
        let imm = a as u32 | (nb.wrapping_sub(1) as u32) << 8 | (c as u32) << 16;
        let lp = self.e.g.block(self.e.cur).lp;
        let resume = self.e.new_block();
        self.e
            .emit(Op::CallEntry, Rep::None, 0, state.0, resume.0, imm);
        self.e.begin(resume, lp, 0, BlockFlags::RESUME);
        // One result comes in a register; more land in their slots.
        let rep = if c == 2 { Rep::Val } else { Rep::None };
        let first = self.e.emit(Op::Resume, rep, 0, imm, 0, 0);
        // Results up to `top` (`c == 0`) stay in their slots, for the
        // instruction that takes them from there.
        for r in a as usize..self.cfg.nregs {
            self.e.table.set(Key::reg(r as u32), Val::NONE.0);
        }
        if c == 2 {
            self.set(a, first);
        } else if c != 0 {
            for r in a..a + c - 1 {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.set(r, v);
            }
        }
        // The callee may have written the captured registers.
        for r in captured.iter().filter(|&r| r < a as usize) {
            let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
            self.e.table.set(Key::reg(r as u32), v.0);
        }
        self.state = Val::NONE;
        // The callee may have broken what the region assumes.
        self.lazy_exit(pc + 1);
    }
}
