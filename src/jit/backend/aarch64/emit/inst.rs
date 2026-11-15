//! One allocated `MInst` to its machine instruction. Registers come from
//! the instruction's allocations; the pinned `BASE` and `THREAD`, and the
//! stack registers inside the frame pseudo-instructions, are the only ones
//! named here (`jit::rails`).

use super::Em;
use crate::env::value::NIL;
use crate::jit::backend::aarch64::abi::{BASE, INSN, PC, THREAD};
use crate::jit::backend::aarch64::asm::{Cond, Extend, FP, Gpr, LR, Label, SP, Sz};
use crate::jit::backend::aarch64::inst::MOp;
use crate::jit::ir::ops::{Ctx, Helper};

impl Em<'_> {
    pub(super) fn inst(&mut self, i: usize, block: usize, next: Option<Label>) {
        let m = self.v.insts[i];
        let (sz, imm) = (m.sz, m.imm);
        // A value left for the allocator to make where used has no code.
        let unmade = self.allocs(i).first().is_some_and(|a| a.is_none());
        match m.op {
            MOp::Entry => {
                self.a.sub_imm(Sz::X, SP, SP, self.frame as u64);
                self.a.stp(FP, LR, SP, 0);
                self.a.mov_sp(FP, SP);
            }
            MOp::MovImm | MOp::FImm | MOp::LdrSlot if unmade => {}
            MOp::MovImm => self.a.mov_imm(self.r(i, 0), imm),
            MOp::FImm => self.fimm(self.fr(i, 0), imm),
            MOp::LdrSlot => self.a.ldr(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::StrSlot => self.a.str(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::Ldr if sz == Sz::W => self.a.ldr_w(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::Ldr => self.a.ldr(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::Str => self.a.str(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::LdrIdx | MOp::StrIdx => {
                let ext = if m.cc == 1 { Extend::Uxtw } else { Extend::Lsl };
                let (t, n, x) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match m.op {
                    MOp::LdrIdx => self.a.ldr_idx(t, n, x, ext, true),
                    _ => self.a.str_idx(t, n, x, ext, true),
                }
            }
            MOp::MovBase => self.a.mov(self.r(i, 0), BASE),
            MOp::Uxtw => self.a.mov_w(self.r(i, 0), self.r(i, 1)),
            MOp::Cset => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a.cset(Sz::W, self.r(i, 0), cc)
            }
            MOp::Csel => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a
                    .csel(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2), cc)
            }
            MOp::AndRR => self.a.and(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::OrrRR => self.a.orr(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::EorRR => self.a.eor(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SdivRR => self.a.sdiv(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::Sxtw => self.a.sxtw(self.r(i, 0), self.r(i, 1)),
            MOp::NegR => self.a.neg(sz, self.r(i, 0), self.r(i, 1)),
            MOp::NegsR => self.a.negs(sz, self.r(i, 0), self.r(i, 1)),
            MOp::FNeg => self.a.fneg(self.fr(i, 0), self.fr(i, 1)),
            MOp::Frintm => self.a.frintm(self.fr(i, 0), self.fr(i, 1)),
            MOp::AndImm => self.a.and_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::TstImm => self.a.tst_imm(sz, self.r(i, 0), imm),
            MOp::AddRR => self.a.add(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubRR => self.a.sub(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::MulRR => self.a.mul(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddImm => self.a.add_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::SubImm => self.a.sub_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::AddsRR => self.a.adds(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubsRR => self.a.subs(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddsImm => self.a.adds_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::SubsImm => self.a.subs_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::CmpRR => self.a.cmp(sz, self.r(i, 0), self.r(i, 1)),
            MOp::CmpImm => self.a.cmp_imm(sz, self.r(i, 0), imm),
            MOp::CmnImm => self.a.cmn_imm(sz, self.r(i, 0), imm),
            MOp::CmpSxtw => self.a.cmp_ext(self.r(i, 0), self.r(i, 1), Extend::Sxtw),
            MOp::Fcmp => self.a.fcmp(self.fr(i, 0), self.fr(i, 1)),
            MOp::LslImm => self.a.lsl_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::LsrImm => self.a.lsr_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::AsrImm => self.a.asr_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::Smull => self.a.smull(self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::OrrImm => self.a.orr_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::Scvtf => self.a.scvtf(sz, self.fr(i, 0), self.r(i, 1)),
            MOp::FmovToGpr => self.a.fmov_to_gpr(self.r(i, 0), self.fr(i, 1)),
            MOp::FmovFromGpr => self.a.fmov_from_gpr(self.fr(i, 0), self.r(i, 1)),
            MOp::FAdd => self.a.fadd(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FSub => self.a.fsub(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FMul => self.a.fmul(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FDiv => self.a.fdiv(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::B => {
                let to = self.dest[self.v.succs(block)[0].index()];
                if Some(to) != next {
                    self.a.b(to);
                }
            }
            MOp::BCond => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                let ([t, f], inline) = self.edge_labels(i, block, next);
                // Unordered fails a float condition and holds its inverse,
                // so the inverted branch takes it to `f` too.
                if Some(f) == next {
                    self.a.b_cond(cc, t);
                } else {
                    self.a.b_cond(cc.invert(), f);
                    if Some(t) != next {
                        self.a.b(t);
                    }
                }
                for (_, _, e) in inline {
                    self.edit(e);
                }
            }
            MOp::BCondExit => {
                // SAFETY: as above.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a.b_cond(cc, self.exit_labels[imm as usize]);
            }
            MOp::BExit => self.a.b(self.exit_labels[imm as usize]),
            MOp::Call => {
                let h = Helper::of(imm as u16);
                let info = h.info();
                for (k, c) in info.ctx.iter().enumerate() {
                    match c {
                        Ctx::Rt => self.a.mov_imm(Gpr(k as u8), self.rt),
                        Ctx::Thread => self.a.mov(Gpr(k as u8), THREAD),
                        Ctx::Base => self.a.mov(Gpr(k as u8), BASE),
                    }
                }
                if info.imm {
                    self.a.mov_imm(Gpr(info.ctx.len() as u8), imm >> 16);
                }
                self.a.bl_far(crate::jit::helpers::address(h));
            }
            MOp::Ret => {
                // To a compiled caller, or to `level_land` for a frame
                // dispatch entered.
                let (a, n) = (imm & 0xff, imm >> 8);
                if n == 0 {
                    self.a.mov_imm(Gpr(0), NIL);
                } else {
                    self.a.ldr(Gpr(0), BASE, a as i32 * 8);
                }
                self.a.ldp(FP, LR, SP, 0);
                self.a.add_imm(Sz::X, SP, SP, self.frame as u64);
                self.a.movz(Sz::X, INSN, n as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, a * 8);
                self.a.ret();
            }
        }
    }
}
