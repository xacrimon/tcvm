//! Instruction selection: the low-level graph to `MInst`s, backward, so
//! that a value nothing asked for costs nothing and a compare is made where
//! its branch or guard reads the flags. An op is one instruction, except
//! the flag-coupled pairs and `MulOvf`. What a record's slots hold is its
//! opcode's row's to say (`inst::ROWS`); a list and a 64-bit constant go to
//! the pool.

use crate::env::value::{PAYLOAD_MASK, SMALL_INT, Value};
use crate::jit::backend::aarch64::asm::{Cond, logical_imm};
use crate::jit::backend::aarch64::inst::{MOp, ROWS, Sz};
use crate::jit::backend::emit::{Exit, Selected, Src};
use crate::jit::backend::reg::{Block as RBlock, RegClass, VReg};
use crate::jit::backend::vcode::{MInst, Pin, Remat, Slot};
use crate::jit::bug;
use crate::jit::compile::Bail;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Heaps, Helper, Op};
use crate::jit::ir::types::Rep;
use crate::jit::ir::{Block, Graph, HOME, Pool, Val};
use crate::jit::region::SRep;

const NONE: u32 = u32::MAX;

fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Lt,
        Cc::Le => Cond::Le,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
        Cc::Lo => Cond::Lo,
        Cc::Ls => Cond::Ls,
        Cc::Hi => Cond::Hi,
        Cc::Hs => Cond::Hs,
    }
}

/// After `fcmp`: false when unordered, except `ne`.
fn float_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Mi,
        Cc::Le => Cond::Ls,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
        Cc::Lo | Cc::Ls | Cc::Hi | Cc::Hs => bug!("an unsigned float compare"),
    }
}

fn sz(rep: Rep) -> Sz {
    match rep {
        Rep::I32 | Rep::B1 => Sz::W,
        _ => Sz::X,
    }
}

/// The bytes a load or store of field `aux` moves.
fn width(aux: u16) -> Sz {
    match Field::info(aux).width {
        4 => Sz::W,
        8 => Sz::X,
        w => bug!("no load or store of {w} bytes"),
    }
}

#[derive(Default)]
pub(crate) struct Select {
    /// Each value's virtual register.
    vreg: Vec<u32>,
    /// The instructions, last first.
    insts: Vec<MInst>,
    /// Per block of the layout: its run of `insts`, and the arguments its
    /// terminator passes each successor, a run of `args`.
    blocks: Vec<(u32, u32, u32, [u32; 2])>,
    args: Vec<VReg>,
    /// The frame slots the region stores to.
    stored: Vec<bool>,
    bail: Option<Bail>,
}

struct Sel<'a> {
    g: &'a Graph,
    pool: &'a Pool,
    s: &'a mut Select,
    out: &'a mut Selected,
}

impl Select {
    pub(crate) fn run(&mut self, g: &Graph, pool: &Pool, out: &mut Selected) -> Result<(), Bail> {
        out.layout.of(g);
        out.vcode.clear();
        out.exits.clear();
        out.entries.clear();
        self.vreg.clear();
        self.vreg.resize(g.insts.len(), NONE);
        self.insts.clear();
        self.args.clear();
        self.blocks.clear();
        self.bail = None;
        self.stored.clear();
        self.stored.resize(256, false);
        for i in &g.insts {
            match i.op {
                Op::SlotStore => self.stored[i.aux as usize] = true,
                Op::CallHelper if Helper::of(i.aux).info().writes.contains(Heaps::SLOT) => {
                    self.stored.fill(true)
                }
                _ => {}
            }
        }
        let n = out.layout.order.len();
        self.blocks.resize(n, (0, 0, 0, [0; 2]));
        let mut sel = Sel {
            g,
            pool,
            s: self,
            out,
        };
        for pos in (0..n).rev() {
            let b = sel.out.layout.order[pos];
            let d = g.block(b);
            let at = sel.s.insts.len() as u32;
            sel.s.blocks[pos].2 = sel.s.args.len() as u32;
            for v in (d.start..d.end).rev() {
                sel.inst(pos, Val(v));
            }
            (sel.s.blocks[pos].0, sel.s.blocks[pos].1) = (at, sel.s.insts.len() as u32);
            if let Some(why) = sel.s.bail {
                return Err(why);
            }
        }
        sel.finish();
        Ok(())
    }
}

impl Sel<'_> {
    fn class(&self, v: Val) -> RegClass {
        match self.g.rep(v) {
            Rep::F64 => RegClass::Float,
            _ => RegClass::Int,
        }
    }

    /// The value whose register holds `v`: a truncation reads its operand's
    /// low half, a guard's result is the value it passed, and a parameter
    /// with one input is that input.
    fn root(&self, mut v: Val) -> Val {
        loop {
            let i = self.g.inst(v);
            v = match i.op {
                Op::Trunc32 => Val(i.a),
                Op::Guard | Op::GuardNot => Val(i.c),
                Op::Param if i.b == 1 => Val(self.g.inputs(v)[0]),
                _ => return v,
            }
        }
    }

    /// The register of `v`, which asks for its instruction.
    fn reg(&mut self, v: Val) -> VReg {
        let v = self.root(v);
        let class = self.class(v);
        if self.s.vreg[v.idx()] == NONE {
            self.s.vreg[v.idx()] = self.out.vcode.new_vreg(class).vreg() as u32;
        }
        VReg::new(self.s.vreg[v.idx()] as usize, class)
    }

    /// The register of an instruction's own value, if anything asked.
    fn def(&self, v: Val) -> Option<VReg> {
        let r = self.s.vreg[v.idx()];
        (r != NONE).then(|| VReg::new(r as usize, self.class(v)))
    }

    fn fresh(&mut self, v: Val) -> VReg {
        self.def(v).unwrap_or_else(|| self.reg(v))
    }

    fn imm(&self, v: Val) -> Option<i64> {
        let i = self.g.inst(v);
        match (i.op, i.rep) {
            (Op::K32, Rep::I32) => Some(i.a as i32 as i64),
            (Op::K32, _) => Some(i.a as i64),
            (Op::K64, _) => Some(i.k64() as i64),
            _ => None,
        }
    }

    /// An instruction of `regs`, then `rest` in the slots after them.
    fn push(&mut self, op: MOp, sz: Sz, cc: Cond, regs: &[VReg], rest: &[u32]) {
        let row = &ROWS[op as usize];
        if row.regs() != regs.len() {
            bug!("{op:?} takes {} registers", row.regs());
        }
        let mut ops = [0; 3];
        for (o, v) in ops.iter_mut().zip(regs) {
            *o = v.vreg() as u32;
        }
        ops[regs.len()..regs.len() + rest.len()].copy_from_slice(rest);
        self.s.insts.push(MInst {
            op: op as u8,
            sz: sz as u8,
            cc: cc as u8,
            n: 0,
            ops,
        });
    }

    fn op(&mut self, op: MOp, sz: Sz, regs: &[VReg]) {
        self.push(op, sz, Cond::Eq, regs, &[])
    }

    /// With an immediate in the slot after the registers.
    fn imm_op(&mut self, op: MOp, sz: Sz, imm: u32, regs: &[VReg]) {
        debug_assert_eq!(ROWS[op as usize].slots[regs.len()], Slot::Imm);
        self.push(op, sz, Cond::Eq, regs, &[imm])
    }

    /// With constant `k` after the registers: in the slot where it fits 32
    /// bits zero-extended (`imm`), else in the pool (`wide`).
    fn const_op(&mut self, imm: MOp, wide: MOp, sz: Sz, k: u64, regs: &[VReg]) {
        let k = if sz == Sz::W { k & 0xffff_ffff } else { k };
        if let Ok(k) = u32::try_from(k) {
            return self.imm_op(imm, sz, k, regs);
        }
        self.wide_op(wide, sz, k, regs)
    }

    fn wide_op(&mut self, op: MOp, sz: Sz, k: u64, regs: &[VReg]) {
        debug_assert_eq!(ROWS[op as usize].slots[regs.len()], Slot::Wide);
        let at = self.out.vcode.pool.len() as u32;
        self.out.vcode.pool.extend([k as u32, (k >> 32) as u32]);
        self.push(op, sz, Cond::Eq, regs, &[at])
    }

    /// A branch to the exit that rebuilds frame state `fs`, taken where
    /// `cc` holds: the state's values ride on it as operands the allocator
    /// may leave anywhere.
    fn exit(&mut self, op: MOp, cc: Cond, fs: Val, tag: u16) {
        let g = self.g;
        let (pc, entries) = g.entries(fs);
        let at = self.out.entries.len() as u32;
        let first = self.out.vcode.pool.len() as u32;
        for &[r, v] in entries {
            if r & HOME != 0 {
                continue;
            }
            let (v, i) = (Val(v), g.inst(Val(v)));
            let rep = match i.rep {
                Rep::Val => SRep::Val,
                Rep::I32 => SRep::I32,
                Rep::I64 => SRep::I64,
                Rep::F64 => SRep::F64,
                Rep::B1 => SRep::B1,
                Rep::Ptr | Rep::None => {
                    self.s.bail = Some(Bail::Unsupported("a raw pointer in a frame state"));
                    SRep::Val
                }
            };
            // A constant is written boxed, by the exit.
            let k = match (i.op, i.rep) {
                (Op::K64, Rep::Val) => Some((SRep::Val, i.k64())),
                (Op::KObj, Rep::Val) => Some((SRep::Val, self.pool.words[i.a as usize])),
                (Op::K32, Rep::I32) => Some((SRep::Val, Value::small(i.a as i32).to_raw())),
                (Op::K32, _) => Some((SRep::Val, Value::boolean(i.a != 0).to_raw())),
                (Op::KF64, _) => Some((SRep::Val, Value::float(f64::from_bits(i.k64())).to_raw())),
                _ => None,
            };
            let src = match k {
                Some((rep, w)) => (r as u8, rep, Src::Const(w)),
                None => {
                    let reg = self.reg(v);
                    let k = self.out.vcode.pool.len() as u32 - first;
                    self.out.vcode.pool.push(reg.vreg() as u32);
                    (r as u8, rep, Src::Operand(k as u16))
                }
            };
            self.out.entries.push(src);
        }
        let exit = self.out.exits.len() as u32;
        self.out.exits.push(Exit {
            pc,
            kind: if g.inst(fs).aux == 0 {
                ExitKind::Before
            } else {
                ExitKind::After
            },
            tag: ExitTag::of(tag),
            at,
            len: self.out.entries.len() as u32 - at,
            inst: 0,
            own: 0,
        });
        let n = self.out.vcode.pool.len() as u32 - first;
        self.push(op, Sz::X, cc, &[], &[exit, first, n]);
    }

    /// The condition code a branch on `c` takes: a compare's, or for a
    /// truth value that it is not zero.
    fn cond(&mut self, c: Val) -> Cond {
        let i = self.g.inst(c);
        match i.op {
            Op::Cmp => int_cond(Cc::of(i.aux)),
            Op::FCmp => float_cond(Cc::of(i.aux)),
            _ => Cond::Ne,
        }
    }

    /// The compare that sets the flags `cond(c)` reads.
    fn compare(&mut self, c: Val) {
        let i = *self.g.inst(c);
        let (x, y) = (Val(i.a), Val(i.b));
        if i.op == Op::FCmp {
            let (x, y) = (self.reg(x), self.reg(y));
            return self.op(MOp::Fcmp, Sz::X, &[x, y]);
        }
        if i.op != Op::Cmp {
            let c = self.reg(c);
            return self.imm_op(MOp::CmpImm, Sz::W, 0, &[c]);
        }
        let sz = sz(self.g.rep(x));
        // Bits tested against zero need no value.
        let d = *self.g.inst(x);
        if d.op == Op::And
            && matches!(Cc::of(i.aux), Cc::Eq | Cc::Ne)
            && self.imm(y) == Some(0)
            && let Some(k) = self.imm(Val(d.b))
            && logical_imm(k as u64, sz.bits()).is_some()
        {
            let x = self.reg(Val(d.a));
            return self.const_op(MOp::TstImm, MOp::TstWide, sz, k as u64, &[x]);
        }
        let x = self.reg(x);
        match self.imm(y) {
            Some(k @ 0..4096) => self.imm_op(MOp::CmpImm, sz, k as u32, &[x]),
            Some(k @ -4095..0) => self.imm_op(MOp::CmnImm, sz, -k as u32, &[x]),
            _ => {
                let y = self.reg(y);
                self.op(MOp::CmpRR, sz, &[x, y])
            }
        }
    }

    /// The arguments an edge from `from` passes the parameters of `to`.
    fn pass(&mut self, pos: usize, k: usize, from: Block, to: Block) {
        let g = self.g;
        let pred = (g.preds(to).iter().position(|&p| p == from)).unwrap();
        let d = g.block(to);
        let at = self.s.args.len();
        // One predecessor: its parameters are their inputs.
        let params = if d.npreds == 1 { d.start } else { d.end };
        for p in d.start..params {
            if g.insts[p as usize].op != Op::Param {
                break;
            }
            let v = self.reg(Val(g.inputs(Val(p))[pred]));
            self.s.args.push(v);
        }
        self.s.blocks[pos].3[k] = (self.s.args.len() - at) as u32;
    }

    fn inst(&mut self, pos: usize, v: Val) {
        use Op::*;
        let i = *self.g.inst(v);
        let (a, b) = (Val(i.a), Val(i.b));
        let w = sz(i.rep);
        let here = self.out.layout.order[pos];
        match i.op {
            Param | FrameState | Trunc32 => {}
            Cmp | FCmp => {
                // Asked for as a value, beside what branches on it.
                let Some(d) = self.def(v) else { return };
                let cc = self.cond(v);
                self.push(MOp::Cset, Sz::W, cc, &[d], &[]);
                self.compare(v);
            }
            Select if i.rep != Rep::F64 => {
                let Some(d) = self.def(v) else { return };
                let cc = self.cond(a);
                let (x, y) = (self.reg(b), self.reg(Val(i.c)));
                self.push(MOp::Csel, w, cc, &[d, x, y], &[]);
                self.compare(a);
            }
            Base | Thread => {
                let Some(d) = self.def(v) else { return };
                let (op, pin) = match i.op {
                    Base => (MOp::MovBase, Pin::Base),
                    _ => (MOp::MovThread, Pin::Thread),
                };
                self.out.vcode.remat[d.vreg()] = Remat::Pinned(pin);
                self.op(op, Sz::X, &[d]);
            }
            Load => {
                let Some(d) = self.def(v) else { return };
                let base = self.reg(a);
                let off = Field::offset(i.aux, i.c) as u32;
                self.imm_op(MOp::Ldr, width(i.aux), off, &[d, base]);
            }
            Store => {
                let (x, base) = (self.reg(b), self.reg(a));
                let off = Field::offset(i.aux, i.c) as u32;
                self.imm_op(MOp::Str, width(i.aux), off, &[x, base]);
            }
            LoadIdx | StoreIdx => {
                let t = match i.op {
                    LoadIdx => match self.def(v) {
                        Some(d) => d,
                        None => return,
                    },
                    _ => self.reg(Val(i.c)),
                };
                // A zero-extended index is the 32-bit register, extended by
                // the access.
                let x = *self.g.inst(b);
                let (index, uxtw) = if x.op == ZExt32 {
                    (self.reg(Val(x.a)), Cond::Ne)
                } else {
                    (self.reg(b), Cond::Eq)
                };
                let base = self.reg(a);
                let op = if i.op == LoadIdx {
                    MOp::LdrIdx
                } else {
                    MOp::StrIdx
                };
                self.push(op, Sz::X, uxtw, &[t, base, index], &[]);
            }
            ZExt32 => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                self.op(MOp::Uxtw, Sz::X, &[d, x]);
            }
            UnboxPtr | UnboxPtrPin => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                self.const_op(MOp::AndImm, MOp::AndWide, Sz::X, PAYLOAD_MASK, &[d, x]);
            }
            And | Or => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                match self
                    .imm(b)
                    .filter(|&k| logical_imm(k as u64, w.bits()).is_some())
                {
                    Some(k) if i.op == And => {
                        self.const_op(MOp::AndImm, MOp::AndWide, w, k as u64, &[d, x])
                    }
                    Some(k) => self.const_op(MOp::OrrImm, MOp::OrrWide, w, k as u64, &[d, x]),
                    None => {
                        let y = self.reg(b);
                        let op = if i.op == And { MOp::AndRR } else { MOp::OrrRR };
                        self.op(op, w, &[d, x, y])
                    }
                }
            }
            K32 | K64 | KF64 | KObj => {
                let Some(d) = self.def(v) else { return };
                let k = match i.op {
                    K32 => i.a as u64,
                    // The heap is non-moving: the object's word is a constant,
                    // and its pointer the word's payload.
                    KObj if i.rep == Rep::Ptr => self.pool.words[i.a as usize] & PAYLOAD_MASK,
                    KObj => self.pool.words[i.a as usize],
                    _ => i.k64(),
                };
                if i.op == KF64 {
                    self.out.vcode.remat[d.vreg()] = Remat::Float(k);
                    return self.wide_op(MOp::FImm, Sz::X, k, &[d]);
                }
                self.out.vcode.remat[d.vreg()] = Remat::Int(k);
                self.const_op(MOp::MovImm, MOp::MovWide, Sz::X, k, &[d]);
            }
            SlotLoad => {
                let Some(d) = self.def(v) else { return };
                // A slot nothing stores to keeps the value: it is read
                // again from there instead of being kept.
                if !self.s.stored[i.aux as usize] {
                    self.out.vcode.remat[d.vreg()] = Remat::Slot(i.aux as u8);
                }
                self.imm_op(MOp::LdrSlot, Sz::X, i.aux as u32, &[d]);
            }
            SlotStore => {
                let x = self.reg(a);
                self.imm_op(MOp::StrSlot, Sz::X, i.aux as u32, &[x]);
            }
            Add | Sub => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                let (imm, rr, other) = match i.op {
                    Add => (MOp::AddImm, MOp::AddRR, MOp::SubImm),
                    _ => (MOp::SubImm, MOp::SubRR, MOp::AddImm),
                };
                match self.imm(b) {
                    Some(k @ 0..4096) => self.imm_op(imm, w, k as u32, &[d, x]),
                    Some(k @ -4095..0) => self.imm_op(other, w, -k as u32, &[d, x]),
                    _ => {
                        let y = self.reg(b);
                        self.op(rr, w, &[d, x, y])
                    }
                }
            }
            Mul | SDiv | Xor => {
                let Some(d) = self.def(v) else { return };
                let (x, y) = (self.reg(a), self.reg(b));
                let op = match i.op {
                    Mul => MOp::MulRR,
                    SDiv => MOp::SdivRR,
                    _ => MOp::EorRR,
                };
                self.op(op, w, &[d, x, y]);
            }
            SExt32 | Neg | FNeg | FFloor => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                let op = match i.op {
                    SExt32 => MOp::Sxtw,
                    Neg => MOp::NegR,
                    FNeg => MOp::FNeg,
                    _ => MOp::Frintm,
                };
                self.op(op, w, &[d, x]);
            }
            NegOvf => {
                self.exit(MOp::BCondExit, Cond::Vs, b, i.aux);
                let (d, x) = (self.fresh(v), self.reg(a));
                self.op(MOp::NegsR, w, &[d, x]);
            }
            Shl | Shr | Sar => {
                let Some(d) = self.def(v) else { return };
                let Some(k) = self.imm(b) else {
                    self.s.bail = Some(Bail::Unsupported("a variable shift"));
                    return;
                };
                let op = match i.op {
                    Shl => MOp::LslImm,
                    Shr => MOp::LsrImm,
                    _ => MOp::AsrImm,
                };
                let x = self.reg(a);
                self.imm_op(op, w, k as u32 & (w.bits() - 1), &[d, x]);
            }
            AddOvf | SubOvf => {
                self.exit(MOp::BCondExit, Cond::Vs, Val(i.c), i.aux);
                let (d, x) = (self.fresh(v), self.reg(a));
                let (imm, rr, other) = match i.op {
                    AddOvf => (MOp::AddsImm, MOp::AddsRR, MOp::SubsImm),
                    _ => (MOp::SubsImm, MOp::SubsRR, MOp::AddsImm),
                };
                match self.imm(b) {
                    Some(k @ 0..4096) => self.imm_op(imm, w, k as u32, &[d, x]),
                    Some(k @ -4095..0) => self.imm_op(other, w, -k as u32, &[d, x]),
                    _ => {
                        let y = self.reg(b);
                        self.op(rr, w, &[d, x, y])
                    }
                }
            }
            MulOvf if i.rep == Rep::I32 => {
                // The 64-bit product fits when it is its low half extended.
                self.exit(MOp::BCondExit, Cond::Ne, Val(i.c), i.aux);
                let d = self.fresh(v);
                self.op(MOp::CmpSxtw, Sz::X, &[d, d]);
                let (x, y) = (self.reg(a), self.reg(b));
                self.op(MOp::Smull, Sz::X, &[d, x, y]);
            }
            Guard | GuardNot => {
                let cc = self.cond(a);
                // Unordered fails every float condition but `ne`, and the
                // inverse of each holds then: a guard leaves on a NaN.
                let leave = if i.op == Guard { cc.invert() } else { cc };
                self.exit(MOp::BCondExit, leave, b, i.aux);
                self.compare(a);
            }
            Deopt => self.exit(MOp::BExit, Cond::Eq, a, i.aux),
            Jump => {
                self.op(MOp::B, Sz::X, &[]);
                self.pass(pos, 0, here, Block(i.a));
            }
            Br => {
                let cc = self.cond(a);
                self.push(MOp::BCond, Sz::X, cc, &[], &[]);
                self.compare(a);
                self.pass(pos, 0, here, Block(i.b));
                self.pass(pos, 1, here, Block(i.c));
            }
            Ret => self.imm_op(MOp::Ret, Sz::X, i.a | i.b << 8, &[]),
            CallHelper => {
                // The list: the result if anything reads it, then the
                // operands, which the row fixes to the argument registers.
                let first = self.out.vcode.pool.len() as u32;
                let d = self.def(v);
                if let Some(d) = d {
                    self.out.vcode.pool.push(d.vreg() as u32);
                }
                for k in 0..i.b {
                    let x = self.reg(Val(self.g.run(i.a, i.b)[k as usize]));
                    self.out.vcode.pool.push(x.vreg() as u32);
                }
                let n = self.out.vcode.pool.len() as u32 - first;
                self.push(MOp::Call, Sz::X, Cond::Eq, &[], &[i.aux as u32, first, n]);
                self.s.insts.last_mut().unwrap().n = d.is_some() as u8;
            }
            BoxI32 => {
                let Some(d) = self.def(v) else { return };
                // Whatever the upper half held, it is the tag's ones now.
                let x = self.reg(a);
                self.const_op(MOp::OrrImm, MOp::OrrWide, Sz::X, SMALL_INT, &[d, x]);
            }
            BoxF64 | UnboxF64 | IToF => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                let (op, sz) = match i.op {
                    BoxF64 => (MOp::FmovToGpr, Sz::X),
                    UnboxF64 => (MOp::FmovFromGpr, Sz::X),
                    _ => (MOp::Scvtf, Sz::W),
                };
                self.op(op, sz, &[d, x]);
            }
            FAdd | FSub | FMul | FDiv => {
                let Some(d) = self.def(v) else { return };
                let (x, y) = (self.reg(a), self.reg(b));
                let op = match i.op {
                    FAdd => MOp::FAdd,
                    FSub => MOp::FSub,
                    FMul => MOp::FMul,
                    _ => MOp::FDiv,
                };
                self.op(op, Sz::X, &[d, x, y]);
            }
            _ => {
                if i.op.info().dialect == crate::jit::ir::ops::Dialect::H {
                    bug!("high-level {:?} reached selection", i.op);
                }
                self.s.bail = Some(Bail::Unsupported("an op selection has no instruction for"));
            }
        }
    }

    /// The machine blocks, forward, from what was selected backward.
    fn finish(&mut self) {
        let g = self.g;
        let n = self.out.layout.order.len();
        let pos_of =
            |l: &crate::jit::backend::order::Layout, b: Block| RBlock::new(l.pos[b.idx()] as usize);
        let mut params: Vec<VReg> = Vec::new();
        for pos in 0..n {
            let b = self.out.layout.order[pos];
            let d = g.block(b);
            let start = self.out.vcode.insts.len() as u32;
            if pos == 0 {
                self.out.vcode.push(MInst {
                    op: MOp::Entry as u8,
                    sz: Sz::X as u8,
                    cc: 0,
                    n: 0,
                    ops: [0; 3],
                });
            }
            let (at, end, args, nargs) = self.s.blocks[pos];
            for k in (at..end).rev() {
                let i = self.s.insts[k as usize];
                let at = self.out.vcode.push(i);
                if matches!(MOp::of(i.op), MOp::BCondExit | MOp::BExit) {
                    self.out.exits[i.ops[0] as usize].inst = at.index() as u32;
                }
            }
            params.clear();
            for p in d.start..d.end {
                if g.insts[p as usize].op != Op::Param || d.npreds == 1 {
                    break;
                }
                params.push(self.reg(Val(p)));
            }
            let (succs, ns) = g.succs(g.term(b));
            let succs = succs.map(|s| {
                if s.is_none() {
                    RBlock::new(0)
                } else {
                    pos_of(&self.out.layout, s)
                }
            });
            let total = (nargs[0] + nargs[1]) as usize;
            let layout = &self.out.layout;
            self.out.vcode.push_block(
                start,
                self.out.vcode.insts.len() as u32,
                &succs[..ns],
                g.preds(b).iter().map(|&p| pos_of(layout, p)),
                params.iter().copied(),
                &self.s.args[args as usize..args as usize + total],
                &nargs[..ns],
                false,
            );
        }
        self.out.vcode.loop_of.clone_from(&self.out.layout.loop_of);
        (self.out.vcode.loop_parent).clone_from(&self.out.layout.loop_parent);
    }
}
