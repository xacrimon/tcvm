//! One allocated `MInst` to its machine instruction. Registers come from
//! the instruction's allocations; the pinned `BASE` and `THREAD`, and the
//! stack registers inside the frame pseudo-instructions, are the only ones
//! named here (`jit::rails`).

use super::Em;
use crate::env::value::NIL;
use crate::jit::backend::aarch64::abi::{BASE, INSN, PC, THREAD};
use crate::jit::backend::aarch64::asm::{self, Cond, Extend, FP, Gpr, LR, Label, SP};
use crate::jit::backend::aarch64::inst::{MOp, Sz};
use crate::jit::bug;
use crate::jit::ir::ops::Helper;

impl Em<'_> {
    pub(super) fn inst(&mut self, i: usize, block: usize, next: Option<Label>) {
        let m = self.v.insts[i];
        let op = MOp::of(m.op);
        // The slot after the registers, where the row has an immediate
        // there; an operation's width, where it has one.
        let imm = m.ops[self.v.row(i).regs().min(2)];
        let sz = match Sz::of(m.sz) {
            Sz::W => asm::Sz::W,
            _ => asm::Sz::X,
        };
        // A value left for the allocator to make where used has no code.
        if self.v.row(i).remat && self.allocs(i)[0].is_none() {
            return;
        }
        match op {
            MOp::Entry => {
                self.a.sub_imm(asm::Sz::X, SP, SP, self.frame as u64);
                self.a.stp(FP, LR, SP, 0);
                self.a.mov_sp(FP, SP);
            }
            MOp::MovImm => self.a.mov_imm(self.r(i, 0), imm as u64),
            MOp::MovWide => self.a.mov_imm(self.r(i, 0), self.v.wide(imm)),
            MOp::FImm => self.fimm(self.fr(i, 0), self.v.wide(imm)),
            MOp::LdrSlot => self.a.ldr(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::StrSlot => self.a.str(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::Ldr | MOp::Str => {
                let (t, n, off) = (self.r(i, 0), self.r(i, 1), imm as i32);
                match (op, Sz::of(m.sz)) {
                    (MOp::Ldr, Sz::X) => self.a.ldr(t, n, off),
                    (MOp::Ldr, Sz::W) => self.a.ldr_w(t, n, off),
                    (_, Sz::X) => self.a.str(t, n, off),
                    (_, Sz::W) => self.a.str_w(t, n, off),
                    (_, w) => bug!("no load or store of {w:?}"),
                }
            }
            MOp::LdrIdx | MOp::StrIdx => {
                let ext = if m.cc == 1 { Extend::Uxtw } else { Extend::Lsl };
                let (t, n, x) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match op {
                    MOp::LdrIdx => self.a.ldr_idx(t, n, x, ext, true),
                    _ => self.a.str_idx(t, n, x, ext, true),
                }
            }
            MOp::LdrbIdx => self.a.ldrb_reg(self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::MovBase => self.a.mov(self.r(i, 0), BASE),
            MOp::MovThread => self.a.mov(self.r(i, 0), THREAD),
            MOp::Uxtw => self.a.mov_w(self.r(i, 0), self.r(i, 1)),
            MOp::Cset => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a.cset(asm::Sz::W, self.r(i, 0), cc)
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
            MOp::MvnR => self.a.mvn(sz, self.r(i, 0), self.r(i, 1)),
            MOp::NegsR => self.a.negs(sz, self.r(i, 0), self.r(i, 1)),
            MOp::FNeg => self.a.fneg(self.fr(i, 0), self.fr(i, 1)),
            MOp::Frintm => self.a.frintm(self.fr(i, 0), self.fr(i, 1)),
            MOp::AndImm => self.a.and_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::AndWide => {
                let k = self.v.wide(imm);
                self.a.and_imm(sz, self.r(i, 0), self.r(i, 1), k)
            }
            MOp::TstImm => self.a.tst_imm(sz, self.r(i, 0), imm as u64),
            MOp::TstWide => self.a.tst_imm(sz, self.r(i, 0), self.v.wide(imm)),
            MOp::AddRR => self.a.add(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubRR => self.a.sub(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::MulRR => self.a.mul(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddImm => self.a.add_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::SubImm => self.a.sub_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::AddsRR => self.a.adds(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubsRR => self.a.subs(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddsImm => self.a.adds_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::SubsImm => self.a.subs_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::CmpRR => self.a.cmp(sz, self.r(i, 0), self.r(i, 1)),
            MOp::CmpImm => self.a.cmp_imm(sz, self.r(i, 0), imm as u64),
            MOp::CmnImm => self.a.cmn_imm(sz, self.r(i, 0), imm as u64),
            MOp::CmpSxtw => self.a.cmp_ext(self.r(i, 0), self.r(i, 1), Extend::Sxtw),
            MOp::Fcmp => self.a.fcmp(self.fr(i, 0), self.fr(i, 1)),
            MOp::LslImm => self.a.lsl_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::LsrImm => self.a.lsr_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::AsrImm => self.a.asr_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::Smull => self.a.smull(self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::OrrImm => self.a.orr_imm(sz, self.r(i, 0), self.r(i, 1), imm as u64),
            MOp::OrrWide => {
                let k = self.v.wide(imm);
                self.a.orr_imm(sz, self.r(i, 0), self.r(i, 1), k)
            }
            MOp::Scvtf => self.a.scvtf(sz, self.fr(i, 0), self.r(i, 1)),
            MOp::Fcvtzs => self.a.fcvtzs(sz, self.r(i, 0), self.fr(i, 1)),
            MOp::LslRR => self.a.lslv(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::LsrRR => self.a.lsrv(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AsrRR => self.a.asrv(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
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
            MOp::Call => (self.a).bl_far(crate::jit::helpers::address(Helper::of(imm as u16))),
            MOp::Ret => {
                // To a compiled caller, or to `level_land` for a frame
                // dispatch entered.
                let (a, n) = (imm as u64 & 0xff, imm as u64 >> 8);
                if n == 0 {
                    self.a.mov_imm(Gpr(0), NIL);
                } else {
                    self.a.ldr(Gpr(0), BASE, a as i32 * 8);
                }
                self.a.ldp(FP, LR, SP, 0);
                self.a.add_imm(asm::Sz::X, SP, SP, self.frame as u64);
                self.a.movz(asm::Sz::X, INSN, n as u16, 0);
                self.a.add_imm(asm::Sz::X, PC, BASE, a * 8);
                self.a.ret();
            }
        }
    }
}
