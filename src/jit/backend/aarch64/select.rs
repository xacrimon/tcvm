//! Instruction selection: the low-level graph to `MInst`s, backward, so
//! that a value nothing asked for costs nothing and a compare is made where
//! its branch or guard reads the flags. An op is one instruction, except
//! the flag-coupled pairs and `MulOvf`.

use crate::env::value::{PAYLOAD_MASK, SMALL_INT, Value};
use crate::jit::backend::aarch64::abi;
use crate::jit::backend::aarch64::asm::{Cond, Sz, logical_imm};
use crate::jit::backend::aarch64::inst::{MInst, MOp, Role};
use crate::jit::backend::emit::{CallSite, Exit, Selected, Src, TRACED};
use crate::jit::backend::reg::{
    Block as RBlock, Operand, OperandConstraint, OperandKind, OperandPos, PRegSet, RegClass, VReg,
};
use crate::jit::backend::vcode::Remat;
use crate::jit::bug;
use crate::jit::compile::Bail;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Heaps, Helper, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, BlockFlags, Graph, HOME, Pool, Val};
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

#[derive(Default)]
pub(crate) struct Select {
    /// Each value's virtual register.
    vreg: Vec<u32>,
    /// The instructions, last first, each with its run of `ops`.
    insts: Vec<(MInst, u32, u32, PRegSet)>,
    ops: Vec<Operand>,
    /// Per block of the layout: its run of `insts`, and the arguments its
    /// terminator passes each successor, a run of `args`.
    blocks: Vec<(u32, u32, u32, [u32; 2])>,
    args: Vec<VReg>,
    /// The frame slots the region stores to.
    stored: Vec<bool>,
    /// For a region that calls: the layout position of each value's block,
    /// and of the last block that may use it (`spans`); and the values the
    /// collector must find if a call runs while they live.
    defined: Vec<u32>,
    last: Vec<u32>,
    traced: Vec<(u32, u8)>,
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
        out.sites.clear();
        out.dispatches.clear();
        self.vreg.clear();
        self.vreg.resize(g.insts.len(), NONE);
        self.insts.clear();
        self.ops.clear();
        self.args.clear();
        self.blocks.clear();
        self.bail = None;
        self.stored.clear();
        self.stored.resize(256, false);
        let mut calls = false;
        for i in &g.insts {
            match i.op {
                Op::SlotStore => self.stored[i.aux as usize] = true,
                // The collector clears the slots a call's site does not name.
                Op::CallEntry => {
                    self.stored.fill(true);
                    calls = true;
                }
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
        sel.s.traced.clear();
        if calls {
            sel.spans();
        }
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

    fn push(&mut self, op: MOp, sz: Sz, cc: Cond, imm: u64, regs: &[VReg]) {
        let roles = op.info().roles;
        if roles.len() != regs.len() {
            bug!("{op:?} takes {} registers", roles.len());
        }
        let at = self.s.ops.len() as u32;
        for (r, &v) in roles.iter().zip(regs) {
            self.s.ops.push(match r {
                Role::D | Role::Fd => Operand::reg_def(v),
                Role::U | Role::Fu => Operand::reg_use(v),
            });
        }
        let i = MInst {
            op,
            cc: cc as u8,
            sz,
            imm,
        };
        (self.s.insts).push((i, at, regs.len() as u32, PRegSet::empty()));
    }

    fn op(&mut self, op: MOp, sz: Sz, imm: u64, regs: &[VReg]) {
        self.push(op, sz, Cond::Eq, imm, regs)
    }

    /// For a region that calls, where each value lives: from its block to
    /// the last block using it, in layout order, or to the end of the
    /// outermost loop that holds the use and not the definition. A value
    /// lives at no other block, and only at those its block dominates.
    fn spans(&mut self) {
        let g = self.g;
        let n = g.insts.len();
        self.s.defined.clear();
        self.s.defined.resize(n, 0);
        self.s.last.clear();
        self.s.last.resize(n, 0);
        for (pos, &b) in self.out.layout.order.iter().enumerate() {
            let d = g.block(b);
            self.s.defined[d.start as usize..d.end as usize].fill(pos as u32);
        }
        for pos in 0..self.out.layout.order.len() {
            let b = self.out.layout.order[pos];
            let d = g.block(b);
            for v in d.start..d.end {
                let i = &g.insts[v as usize];
                if i.op == Op::Param {
                    // An input is used where its edge leaves.
                    for (k, &x) in g.inputs(Val(v)).iter().enumerate() {
                        let from = self.out.layout.pos[g.preds(b)[k].idx()];
                        self.used(Val(x), from);
                    }
                } else {
                    g.uses(i, |x| self.used(x, pos as u32));
                }
            }
        }
        for v in 0..n as u32 {
            let i = &g.insts[v as usize];
            if !matches!(i.rep, Rep::Val | Rep::Ptr)
                || self.root(Val(v)) != Val(v)
                || matches!(i.op, Op::K32 | Op::K64 | Op::KF64 | Op::KObj | Op::Base)
                || self.s.last[v as usize] <= self.s.defined[v as usize]
            {
                continue;
            }
            // A pointer of unknown kind is of something its owner keeps: a
            // closure's cell, a shape's class.
            let set = g.facts[v as usize].set;
            let tag = match (i.rep, set) {
                (Rep::Val, _) if set.intersects(TypeSet::HEAP) => TRACED,
                (Rep::Ptr, TypeSet::UDATA) => crate::env::value::TAG_USERDATA as u8,
                (Rep::Ptr, TypeSet::BIGINT) => crate::env::value::TAG_BOXED_INT as u8,
                (Rep::Ptr, TypeSet::STR) => crate::env::value::TAG_STRING as u8,
                (Rep::Ptr, TypeSet::TAB) => crate::env::value::TAG_TABLE as u8,
                (Rep::Ptr, TypeSet::FUN) => (crate::env::value::FUNCTION_BITS >> 48 & 7) as u8,
                (Rep::Ptr, TypeSet::THR) => crate::env::value::TAG_THREAD as u8,
                _ => continue,
            };
            self.s.traced.push((v, tag));
        }
    }

    fn used(&mut self, x: Val, at: u32) {
        let g = self.g;
        let x = self.root(x);
        let def = self.s.defined[x.idx()];
        let mut end = at;
        let mut l = g.block(self.out.layout.order[at as usize]).lp;
        while !l.is_none() {
            let (first, last) = self.out.layout.loops[l.idx()];
            if (first..=last).contains(&def) {
                break;
            }
            end = end.max(last);
            l = g.loops[l.idx()].parent;
        }
        let last = &mut self.s.last[x.idx()];
        *last = (*last).max(end);
    }

    /// A branch to the exit that rebuilds frame state `fs`, taken where
    /// `cc` holds: the state's values ride on it as operands the allocator
    /// may leave anywhere.
    fn exit(&mut self, op: MOp, cc: Cond, fs: Val, tag: u16) {
        let g = self.g;
        let (pc, at, first) = self.state(fs, false);
        let exit = self.out.exits.len() as u64;
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
        let i = MInst {
            op,
            cc: cc as u8,
            sz: Sz::X,
            imm: exit,
        };
        let n = self.s.ops.len() as u32 - first;
        self.s.insts.push((i, first, n, PRegSet::empty()));
    }

    /// A Lua call. What its state names lives on in stack slots, or in the
    /// frame slot that holds it; so does every other value the collector
    /// must find that lives across it.
    fn call(&mut self, pos: usize, fs: Val, imm: u32) {
        let g = self.g;
        let here = self.out.layout.order[pos];
        let (pc, at, first) = self.state(fs, true);
        for k in 0..self.s.traced.len() {
            let (v, tag) = self.s.traced[k];
            let (def, last) = (self.s.defined[v as usize], self.s.last[v as usize]);
            let block = self.out.layout.order[def as usize];
            if def as usize > pos || last as usize <= pos || !g.dominates(block, here) {
                continue;
            }
            let reg = self.reg(Val(v));
            if self.s.ops[first as usize..].iter().any(|o| o.vreg() == reg) {
                continue;
            }
            let k = self.s.ops.len() as u32 - first;
            (self.s.ops).push(Operand::new(
                reg,
                OperandConstraint::Stack,
                OperandKind::Use,
                OperandPos::Late,
            ));
            let rep = if tag == TRACED { SRep::Val } else { SRep::Ptr };
            self.out.entries.push((tag, rep, Src::Operand(k as u16)));
        }
        self.out.sites.push(CallSite {
            at,
            len: self.out.entries.len() as u32 - at,
            block: pos as u32,
            inst: 0,
        });
        let i = MInst {
            op: MOp::CallLua,
            cc: 0,
            sz: Sz::X,
            imm: imm as u64 | (pc as u64) << 32,
        };
        let n = self.s.ops.len() as u32 - first;
        self.s.insts.push((i, first, n, abi::allocatable()));
    }

    /// The values of frame state `fs` as late operands, appended, with
    /// their entries: for a call's `site` in stack slots, a register whose
    /// frame slot holds its value named so; for an exit anywhere, such a
    /// register left out. Returns the state's pc, and where the entries and
    /// the operands start.
    fn state(&mut self, fs: Val, site: bool) -> (u32, u32, u32) {
        let g = self.g;
        let (pc, entries) = g.entries(fs);
        let at = self.out.entries.len() as u32;
        let first = self.s.ops.len() as u32;
        for &[r, v] in entries {
            if r & HOME != 0 {
                if site {
                    (self.out.entries).push(((r & !HOME) as u8, SRep::Val, Src::Home));
                }
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
                    let k = self.s.ops.len() as u32 - first;
                    let constraint = if site {
                        OperandConstraint::Stack
                    } else {
                        OperandConstraint::Any
                    };
                    (self.s.ops).push(Operand::new(
                        reg,
                        constraint,
                        OperandKind::Use,
                        OperandPos::Late,
                    ));
                    (r as u8, rep, Src::Operand(k as u16))
                }
            };
            self.out.entries.push(src);
        }
        (pc, at, first)
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
            return self.op(MOp::Fcmp, Sz::X, 0, &[x, y]);
        }
        if i.op != Op::Cmp {
            let c = self.reg(c);
            return self.op(MOp::CmpImm, Sz::W, 0, &[c]);
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
            return self.op(MOp::TstImm, sz, k as u64, &[x]);
        }
        let x = self.reg(x);
        match self.imm(y) {
            Some(k @ 0..4096) => self.op(MOp::CmpImm, sz, k as u64, &[x]),
            Some(k @ -4095..0) => self.op(MOp::CmnImm, sz, -k as u64, &[x]),
            _ => {
                let y = self.reg(y);
                self.op(MOp::CmpRR, sz, 0, &[x, y])
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
                self.push(MOp::Cset, Sz::W, cc, 0, &[d]);
                self.compare(v);
            }
            Select if i.rep != Rep::F64 => {
                let Some(d) = self.def(v) else { return };
                let cc = self.cond(a);
                let (x, y) = (self.reg(b), self.reg(Val(i.c)));
                self.push(MOp::Csel, w, cc, 0, &[d, x, y]);
                self.compare(a);
            }
            Base => {
                let Some(d) = self.def(v) else { return };
                self.op(MOp::MovBase, Sz::X, 0, &[d]);
            }
            Load => {
                let Some(d) = self.def(v) else { return };
                let base = self.reg(a);
                let off = Field::offset(i.aux, i.c) as i64 as u64;
                self.op(MOp::Ldr, w, off, &[d, base]);
            }
            Store => {
                let (x, base) = (self.reg(b), self.reg(a));
                let off = Field::offset(i.aux, i.c) as i64 as u64;
                self.op(MOp::Str, Sz::X, off, &[x, base]);
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
                self.push(op, Sz::X, uxtw, 0, &[t, base, index]);
            }
            ZExt32 => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                self.op(MOp::Uxtw, Sz::X, 0, &[d, x]);
            }
            UnboxPtr | UnboxPtrPin => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                self.op(MOp::AndImm, Sz::X, PAYLOAD_MASK, &[d, x]);
            }
            And | Or => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                match self
                    .imm(b)
                    .filter(|&k| logical_imm(k as u64, w.bits()).is_some())
                {
                    Some(k) if i.op == And => self.op(MOp::AndImm, w, k as u64, &[d, x]),
                    Some(k) => self.op(MOp::OrrImm, w, k as u64, &[d, x]),
                    None => {
                        let y = self.reg(b);
                        let op = if i.op == And { MOp::AndRR } else { MOp::OrrRR };
                        self.op(op, w, 0, &[d, x, y])
                    }
                }
            }
            K32 | K64 | KF64 | KObj => {
                let Some(d) = self.def(v) else { return };
                let (op, remat) = match i.op {
                    K32 => (MOp::MovImm, Remat::Int(i.a as u64)),
                    K64 => (MOp::MovImm, Remat::Int(i.k64())),
                    // The heap is non-moving: the object's word is a constant,
                    // and its pointer the word's payload.
                    KObj => {
                        let w = self.pool.words[i.a as usize];
                        let w = if i.rep == Rep::Ptr {
                            w & PAYLOAD_MASK
                        } else {
                            w
                        };
                        (MOp::MovImm, Remat::Int(w))
                    }
                    _ => (MOp::FImm, Remat::Float(i.k64())),
                };
                self.out.vcode.remat[d.vreg()] = remat;
                let (Remat::Int(imm) | Remat::Float(imm)) = remat else {
                    unreachable!()
                };
                self.op(op, Sz::X, imm, &[d]);
            }
            SlotLoad => {
                let Some(d) = self.def(v) else { return };
                // A slot nothing stores to keeps the value: it is read
                // again from there instead of being kept.
                if !self.s.stored[i.aux as usize] {
                    self.out.vcode.remat[d.vreg()] = Remat::Slot(i.aux as u8);
                }
                self.op(MOp::LdrSlot, Sz::X, i.aux as u64, &[d]);
            }
            SlotStore => {
                let x = self.reg(a);
                self.op(MOp::StrSlot, Sz::X, i.aux as u64, &[x]);
            }
            Add | Sub => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                let (imm, rr, other) = match i.op {
                    Add => (MOp::AddImm, MOp::AddRR, MOp::SubImm),
                    _ => (MOp::SubImm, MOp::SubRR, MOp::AddImm),
                };
                match self.imm(b) {
                    Some(k @ 0..4096) => self.op(imm, w, k as u64, &[d, x]),
                    Some(k @ -4095..0) => self.op(other, w, -k as u64, &[d, x]),
                    _ => {
                        let y = self.reg(b);
                        self.op(rr, w, 0, &[d, x, y])
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
                self.op(op, w, 0, &[d, x, y]);
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
                self.op(op, w, 0, &[d, x]);
            }
            NegOvf => {
                self.exit(MOp::BCondExit, Cond::Vs, b, i.aux);
                let (d, x) = (self.fresh(v), self.reg(a));
                self.op(MOp::NegsR, w, 0, &[d, x]);
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
                self.op(op, w, k as u64 & (w.bits() as u64 - 1), &[d, x]);
            }
            AddOvf | SubOvf => {
                self.exit(MOp::BCondExit, Cond::Vs, Val(i.c), i.aux);
                let (d, x) = (self.fresh(v), self.reg(a));
                let (imm, rr, other) = match i.op {
                    AddOvf => (MOp::AddsImm, MOp::AddsRR, MOp::SubsImm),
                    _ => (MOp::SubsImm, MOp::SubsRR, MOp::AddsImm),
                };
                match self.imm(b) {
                    Some(k @ 0..4096) => self.op(imm, w, k as u64, &[d, x]),
                    Some(k @ -4095..0) => self.op(other, w, -k as u64, &[d, x]),
                    _ => {
                        let y = self.reg(b);
                        self.op(rr, w, 0, &[d, x, y])
                    }
                }
            }
            MulOvf if i.rep == Rep::I32 => {
                // The 64-bit product fits when it is its low half extended.
                self.exit(MOp::BCondExit, Cond::Ne, Val(i.c), i.aux);
                let d = self.fresh(v);
                self.op(MOp::CmpSxtw, Sz::X, 0, &[d, d]);
                let (x, y) = (self.reg(a), self.reg(b));
                self.op(MOp::Smull, Sz::X, 0, &[d, x, y]);
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
            LazyExit => self.exit(MOp::LazyExit, Cond::Eq, a, ExitTag::Lazy as u16),
            Jump => {
                self.op(MOp::B, Sz::X, 0, &[]);
                self.pass(pos, 0, here, Block(i.a));
            }
            Br => {
                let cc = self.cond(a);
                self.push(MOp::BCond, Sz::X, cc, 0, &[]);
                self.compare(a);
                self.pass(pos, 0, here, Block(i.b));
                self.pass(pos, 1, here, Block(i.c));
            }
            Ret => self.op(MOp::Ret, Sz::X, i.a as u64 | (i.b as u64) << 8, &[]),
            Dispatch => {
                let word = i.a as u64 | (i.b as u64) << 32;
                self.out.dispatches.push((word, i.c));
                let k = self.out.dispatches.len() as u64 - 1;
                self.op(MOp::Dispatch, Sz::X, k, &[]);
            }
            CallEntry => {
                self.call(pos, a, i.c);
                self.pass(pos, 0, here, Block(i.b));
            }
            Resume => {
                // Nothing is in a register here: what lives across the call
                // is in the frame, and its reloads come after.
                let at = self.s.ops.len() as u32;
                let mut clobbers = abi::allocatable();
                let mut imm = i.a as u64;
                match self.def(v) {
                    Some(d) => {
                        clobbers.remove(abi::preg_int(0));
                        self.s.ops.push(Operand::reg_fixed_def(d, abi::preg_int(0)));
                        imm |= 1 << 31;
                    }
                    // One result nothing reads is none.
                    None if i.rep == Rep::Val => imm = imm & !0xff_0000 | 1 << 16,
                    None => {}
                }
                let m = MInst {
                    op: MOp::Resume,
                    cc: 0,
                    sz: Sz::X,
                    imm,
                };
                let n = self.s.ops.len() as u32 - at;
                self.s.insts.push((m, at, n, clobbers));
            }
            CallHelper => {
                let h = Helper::of(i.aux).info();
                let at = self.s.ops.len() as u32;
                let mut clobbers = abi::caller_saved();
                if let Some(d) = self.def(v) {
                    let p = match i.rep {
                        Rep::F64 => abi::preg_float(0),
                        _ => abi::preg_int(0),
                    };
                    clobbers.remove(p);
                    self.s.ops.push(Operand::reg_fixed_def(d, p));
                }
                // Integers after the pinned registers and the immediate.
                let (mut ints, mut floats) = (h.ctx.len() as u8 + h.imm as u8, 0);
                for k in 0..i.b {
                    let x = Val(self.g.run(i.a, i.b)[k as usize]);
                    let p = if self.g.rep(x) == Rep::F64 {
                        floats += 1;
                        abi::preg_float(floats - 1)
                    } else {
                        ints += 1;
                        abi::preg_int(ints - 1)
                    };
                    let x = self.reg(x);
                    self.s.ops.push(Operand::reg_fixed_use(x, p));
                }
                let m = MInst {
                    op: MOp::Call,
                    cc: 0,
                    sz: Sz::X,
                    imm: i.aux as u64 | (i.c as u64) << 16,
                };
                let n = self.s.ops.len() as u32 - at;
                self.s.insts.push((m, at, n, clobbers));
            }
            BoxI32 => {
                let Some(d) = self.def(v) else { return };
                // Whatever the upper half held, it is the tag's ones now.
                let x = self.reg(a);
                self.op(MOp::OrrImm, Sz::X, SMALL_INT, &[d, x]);
            }
            BoxF64 | UnboxF64 | IToF => {
                let Some(d) = self.def(v) else { return };
                let x = self.reg(a);
                let (op, sz) = match i.op {
                    BoxF64 => (MOp::FmovToGpr, Sz::X),
                    UnboxF64 => (MOp::FmovFromGpr, Sz::X),
                    _ => (MOp::Scvtf, Sz::W),
                };
                self.op(op, sz, 0, &[d, x]);
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
                self.op(op, Sz::X, 0, &[d, x, y]);
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
                let entry = MInst {
                    op: MOp::Entry,
                    cc: 0,
                    sz: Sz::X,
                    imm: 0,
                };
                self.out.vcode.push(entry, &[], PRegSet::empty());
            }
            let (at, end, args, nargs) = self.s.blocks[pos];
            for k in (at..end).rev() {
                let (i, ops, len, clobbers) = self.s.insts[k as usize];
                let ops = &self.s.ops[ops as usize..(ops + len) as usize];
                let at = self.out.vcode.push(i, ops, clobbers);
                if matches!(i.op, MOp::BCondExit | MOp::BExit | MOp::LazyExit) {
                    self.out.exits[i.imm as usize].inst = at.index() as u32;
                }
                if i.op == MOp::CallLua {
                    let site = self.out.sites.iter_mut().find(|s| s.block == pos as u32);
                    site.unwrap().inst = at.index() as u32;
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
                d.flags.contains(BlockFlags::RESUME),
            );
        }
        self.out.vcode.loop_of.clone_from(&self.out.layout.loop_of);
        (self.out.vcode.loop_parent).clone_from(&self.out.layout.loop_parent);
    }
}
