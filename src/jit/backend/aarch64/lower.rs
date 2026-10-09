//! IR to aarch64 `MInst`s (10.3): one VCode block per IR block, compares
//! fused into the branch, guard, select or set that is their only use,
//! small immediates folded, refining guards and i32 unboxes aliased to their
//! operand's register, and snapshot values attached as `Any` uses.

use regalloc2::{Block as RBlock, Operand, PRegSet, RegClass, VReg};

use crate::env::value::Value;
use crate::jit::backend::aarch64::abi::{caller_saved, preg_float, preg_int};
use crate::jit::backend::aarch64::asm::{Asm, Cond, Sz};
use crate::jit::backend::aarch64::inst::{AluOp, ENTRY_FAIL, FOp, FUnOp, MInst, Test};
use crate::jit::backend::vcode::VCode;
use crate::jit::ir::ops::{Cc, ExitTag, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, ExitKind, Func, Inst, NO_SNAP, Val};
use crate::jit::region::SRep;

/// An exit of the region before allocation: its snapshot's entries, those
/// with a value in a register as operands of the exiting instruction.
pub(crate) struct LExit {
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    /// `(register, rep, source)`.
    pub(crate) entries: Vec<(u8, SRep, Src)>,
    /// The VCode instruction that exits, once lowered.
    pub(crate) inst: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Src {
    /// The `k`th snapshot operand of the exiting instruction.
    Operand(u16),
    /// A raw word.
    Const(u64),
}

pub(crate) struct Lowered {
    pub(crate) vcode: VCode<MInst>,
    pub(crate) exits: Vec<LExit>,
    /// The first operand index of each instruction's snapshot uses.
    pub(crate) snap_ops: Vec<u32>,
    /// The registers the entry guards test.
    pub(crate) entry_regs: Vec<u8>,
}

struct Lower<'a, 'gc> {
    f: &'a Func<'gc>,
    v: VCode<MInst>,
    vreg: Vec<Option<VReg>>,
    /// Values computed by their single user instead of at their definition.
    fused: Vec<bool>,
    uses: Vec<u32>,
    exits: Vec<LExit>,
    snap_ops: Vec<u32>,
    entry_regs: Vec<u8>,
    block_of: Vec<u32>,
    pc_base: usize,
}

fn rep_class(rep: Rep) -> RegClass {
    if rep == Rep::F64 {
        RegClass::Float
    } else {
        RegClass::Int
    }
}

fn srep(rep: Rep) -> SRep {
    match rep {
        Rep::Val | Rep::Ptr => SRep::Val,
        Rep::I32 => SRep::I32,
        Rep::I64 => SRep::I64,
        Rep::F64 => SRep::F64,
        Rep::B1 => SRep::B1,
    }
}

fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Lt,
        Cc::Le => Cond::Le,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
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
    }
}

/// The address of `Code[pc]` of the prototype being compiled.
pub(crate) fn lower<'gc>(f: &Func<'gc>, code_base: usize) -> Result<Lowered, String> {
    let cfg = f.cfg();
    let order = &crate::jit::backend::order::layout(f);
    let mut block_of = vec![u32::MAX; f.blocks.len()];
    for (k, b) in order.iter().enumerate() {
        block_of[b.idx()] = k as u32;
    }
    let uses = f.use_counts();
    let mut lw = Lower {
        f,
        v: VCode::new(),
        vreg: vec![None; f.vals.len()],
        fused: vec![false; f.vals.len()],
        uses,
        exits: Vec::new(),
        snap_ops: Vec::new(),
        entry_regs: Vec::new(),
        block_of,
        pc_base: code_base,
    };
    lw.find_fusions(order);
    let mut params: Vec<VReg> = Vec::new();
    let mut succs: Vec<RBlock> = Vec::new();
    let mut args: Vec<VReg> = Vec::new();
    let mut nargs: Vec<u32> = Vec::new();
    for &b in order {
        let start = lw.v.insts.len() as u32;
        params.clear();
        for &p in f.params(b) {
            params.push(lw.vr(p));
        }
        if b == f.entry {
            lw.push(MInst::Prologue, &[], PRegSet::empty());
        }
        succs.clear();
        args.clear();
        nargs.clear();
        for &i in f.insts_of(b) {
            lw.lower_inst(i)?;
            if f.op(i).is_terminator() {
                // A call's resume block takes its values from the return.
                let call = matches!(f.op(i), Op::Call { .. });
                for e in f.edges(i) {
                    succs.push(RBlock::new(lw.block_of[e.target.idx()] as usize));
                    let a = if call { &[][..] } else { f.vl(e.args) };
                    for &x in a {
                        args.push(lw.vr(x));
                    }
                    nargs.push(a.len() as u32);
                }
            }
        }
        let end = lw.v.insts.len() as u32;
        let preds = cfg
            .preds(b)
            .iter()
            .map(|p| RBlock::new(lw.block_of[p.idx()] as usize));
        lw.v.push_block(
            start,
            end,
            &succs,
            preds,
            params.iter().copied(),
            &args,
            &nargs,
            f.blocks[b.idx()].resume,
        );
    }
    lw.v.entry = RBlock::new(0);
    Ok(Lowered {
        vcode: lw.v,
        exits: lw.exits,
        snap_ops: lw.snap_ops,
        entry_regs: lw.entry_regs,
    })
}

impl<'a, 'gc> Lower<'a, 'gc> {
    /// The vreg of a value, through aliases.
    fn vr(&mut self, v: Val) -> VReg {
        let v = self.alias(v);
        if let Some(r) = self.vreg[v.idx()] {
            return r;
        }
        let r = self.v.new_vreg(rep_class(self.f.ty(v).rep));
        self.vreg[v.idx()] = Some(r);
        r
    }

    /// The value whose register `v` shares: refining guards and i32
    /// unboxes/narrowings change no bits.
    fn alias(&self, mut v: Val) -> Val {
        while let Some(i) = self.f.def_inst(v) {
            match self.f.op(i) {
                Op::Guard(_) => v = self.f.args(i)[0],
                Op::Unbox(Rep::I32) => v = self.f.args(i)[0],
                Op::LToI => v = self.f.args(i)[0],
                _ => break,
            }
        }
        v
    }

    fn push(&mut self, inst: MInst, ops: &[Operand], clob: PRegSet) -> usize {
        let i = self.v.push(inst, ops, clob);
        self.snap_ops.push(ops.len() as u32);
        i.index()
    }

    /// Mark compares used once, by a branch, guard or select of the same
    /// block, to be computed there.
    fn find_fusions(&mut self, order: &[Block]) {
        for &b in order {
            for &i in self.f.insts_of(b) {
                let op = self.f.op(i);
                let cand = match op {
                    Op::Br | Op::GuardTrue | Op::GuardFalse => Some(self.f.args(i)[0]),
                    _ => None,
                };
                if let Some(c) = cand
                    && self.uses[c.idx()] == 1
                    && let Some(di) = self.f.def_inst(c)
                    && self.f.insts[di.idx()].block == b
                    && matches!(
                        self.f.op(di),
                        Op::ICmp(_)
                            | Op::LCmp(_)
                            | Op::FCmp(_)
                            | Op::IsType(_)
                            | Op::IsFalsy
                            | Op::SameBits
                    )
                {
                    self.fused[c.idx()] = true;
                }
            }
        }
    }

    /// The test and operands computing B1 `c`.
    fn test_of(&mut self, c: Val) -> (Test, Cond, Vec<Val>) {
        let i = self.f.def_inst(c).expect("a fused compare");
        let args = self.f.args(i).to_vec();
        match self.f.op(i) {
            Op::ICmp(cc) => {
                if let Some(k) = self.imm12(args[1]) {
                    (
                        Test::I32 {
                            imm: Some(k as i32),
                        },
                        int_cond(cc),
                        vec![args[0]],
                    )
                } else if let Some(k) = self.imm12(args[0]) {
                    (
                        Test::I32 {
                            imm: Some(k as i32),
                        },
                        int_cond(cc.swap()),
                        vec![args[1]],
                    )
                } else {
                    (Test::I32 { imm: None }, int_cond(cc), args)
                }
            }
            Op::LCmp(cc) => {
                if let Some(k) = self.imm12(args[1]) {
                    (Test::X { imm: Some(k) }, int_cond(cc), vec![args[0]])
                } else {
                    (Test::X { imm: None }, int_cond(cc), args)
                }
            }
            Op::FCmp(cc) => (Test::F64, float_cond(cc), args),
            Op::SameBits => (Test::X { imm: None }, Cond::Eq, args),
            Op::IsType(set) => (Test::Type(set), Cond::Eq, args),
            Op::IsFalsy => (Test::Falsy, Cond::Lo, args),
            op => panic!("test_of {op:?}"),
        }
    }

    /// An integer constant usable as a compare immediate (negative ones
    /// become `cmn`).
    fn imm12(&self, v: Val) -> Option<i64> {
        let n = match self.f.def_op(v)? {
            Op::KI32(n) => n as i64,
            Op::KI64(n) => n,
            _ => return None,
        };
        (n.unsigned_abs() < 4096).then_some(n)
    }

    fn use_(&mut self, v: Val) -> Operand {
        Operand::reg_use(self.vr(v))
    }

    fn def(&mut self, v: Val) -> Operand {
        Operand::reg_def(self.vr(v))
    }

    /// Snapshot operands and the exit record of an exiting IR instruction.
    fn exit_of(&mut self, i: Inst) -> (u32, Vec<Operand>) {
        let d = &self.f.insts[i.idx()];
        debug_assert!(d.snap != NO_SNAP, "{:?} exits without a snapshot", d.op);
        let s = self.f.snaps[d.snap as usize];
        let mut ops: Vec<Operand> = Vec::new();
        let mut seen: Vec<(VReg, u16)> = Vec::new();
        let mut entries = Vec::new();
        for &(r, v) in self.f.entries(d.snap) {
            // Boxes in snapshots are the exit's work.
            let v = match self.f.def_op(v) {
                Some(Op::Box) => self.f.args(self.f.def_inst(v).unwrap())[0],
                _ => v,
            };
            let rep = self.f.ty(v).rep;
            if let Some(k) = self.const_word(v) {
                entries.push((r, SRep::Val, Src::Const(k)));
                continue;
            }
            let vr = self.vr(v);
            let k = match seen.iter().find(|(x, _)| *x == vr) {
                Some(&(_, k)) => k,
                None => {
                    let k = ops.len() as u16;
                    // Late: the instruction may write its result or call
                    // before it exits, and the value must survive that.
                    ops.push(Operand::new(
                        vr,
                        regalloc2::OperandConstraint::Any,
                        regalloc2::OperandKind::Use,
                        regalloc2::OperandPos::Late,
                    ));
                    seen.push((vr, k));
                    k
                }
            };
            entries.push((r, srep(rep), Src::Operand(k)));
        }
        let pc = s.pc;
        let kind = s.kind;
        let tag = d.tag;
        self.exits.push(LExit {
            pc,
            kind,
            tag,
            entries,
            inst: None,
        });
        (self.exits.len() as u32 - 1, ops)
    }

    /// The boxed word of a constant value.
    fn const_word(&self, v: Val) -> Option<u64> {
        match self.f.def_op(v)? {
            Op::KVal(b) => Some(b),
            Op::KObj(p) => Some(self.f.pool[p as usize].to_raw()),
            Op::KI32(n) => Some(Value::small(n).to_raw()),
            Op::KF64(b) => Some(Value::float(f64::from_bits(b)).to_raw()),
            Op::KB1(c) => Some(Value::boolean(c).to_raw()),
            Op::KI64(n) if i32::try_from(n).is_ok() => Some(Value::small(n as i32).to_raw()),
            _ => None,
        }
    }

    /// Push an exiting instruction: `ops` then the snapshot uses.
    fn push_exit(
        &mut self,
        i: Inst,
        mk: impl FnOnce(u32) -> MInst,
        mut ops: Vec<Operand>,
        clob: PRegSet,
    ) {
        let (exit, sops) = self.exit_of(i);
        let base = ops.len() as u32;
        ops.extend(sops);
        let at = self.v.push(mk(exit), &ops, clob).index();
        self.snap_ops.push(base);
        self.exits[exit as usize].inst = Some(at);
    }

    fn lower_inst(&mut self, i: Inst) -> Result<(), String> {
        let f = self.f;
        let op = f.op(i);
        let args: Vec<Val> = f.args(i).to_vec();
        let res = (f.insts[i.idx()].rn == 1).then(|| f.result(i));
        // Unused pure results and fused compares are not computed here.
        if let Some(r) = res
            && (self.fused[r.idx()] || (op.is_pure() && self.uses[r.idx()] == 0))
        {
            return Ok(());
        }
        let none = PRegSet::empty();
        match op {
            Op::KVal(b) => {
                let d = self.def(res.unwrap());
                self.push(MInst::MovImm(b), &[d], none);
            }
            Op::KObj(p) => {
                let d = self.def(res.unwrap());
                self.push(MInst::MovImm(f.pool[p as usize].to_raw()), &[d], none);
            }
            Op::KI32(n) => {
                let d = self.def(res.unwrap());
                self.push(MInst::MovImm(n as u32 as u64), &[d], none);
            }
            Op::KI64(n) => {
                let d = self.def(res.unwrap());
                self.push(MInst::MovImm(n as u64), &[d], none);
            }
            Op::KB1(c) => {
                let d = self.def(res.unwrap());
                self.push(MInst::MovImm(c as u64), &[d], none);
            }
            Op::KF64(b) => {
                let d = self.def(res.unwrap());
                self.push(MInst::FImm(b), &[d], none);
            }
            Op::Load(r) => {
                let d = self.def(res.unwrap());
                self.push(MInst::LoadSlot(r), &[d], none);
            }
            Op::Store(r) => {
                let v = args[0];
                // A float stores straight from its d register.
                if let Some(Op::Box) = f.def_op(v)
                    && let x = f.args(f.def_inst(v).unwrap())[0]
                    && f.ty(x).rep == Rep::F64
                {
                    let u = self.use_(x);
                    self.push(MInst::StoreSlotF(r), &[u], none);
                } else {
                    let u = self.use_(v);
                    self.push(MInst::StoreSlot(r), &[u], none);
                }
            }
            Op::UpvalValue(k) => {
                let d = self.def(res.unwrap());
                self.push(MInst::LoadUpval(k), &[d], none);
            }
            Op::Box => {
                let x = args[0];
                let r = res.unwrap();
                match f.ty(x).rep {
                    Rep::I32 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::BoxI32, &[d, u], none);
                    }
                    Rep::F64 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::FmovToGpr, &[d, u], none);
                    }
                    Rep::B1 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::BoxB1, &[d, u], none);
                    }
                    Rep::I64 => {
                        let d = Operand::reg_fixed_def(self.vr(r), preg_int(0));
                        let u = Operand::reg_fixed_use(self.vr(x), preg_int(1));
                        let mut clob = caller_saved();
                        clob.remove(preg_int(0));
                        if f.insts[i.idx()].snap != NO_SNAP {
                            self.push_exit(i, |exit| MInst::BoxI64 { exit }, vec![d, u], clob);
                        } else {
                            self.push(MInst::BoxI64 { exit: u32::MAX }, &[d, u], clob);
                        }
                    }
                    Rep::Val | Rep::Ptr => return Err("box of a boxed value".into()),
                }
            }
            Op::Unbox(rep) => {
                let x = args[0];
                let r = res.unwrap();
                match rep {
                    Rep::I32 => {}
                    Rep::F64 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::FmovFromGpr, &[d, u], none);
                    }
                    Rep::I64 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::UnboxI64, &[d, u], none);
                    }
                    Rep::Ptr => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(
                            MInst::AluImm(AluOp::And, Sz::X, (1 << 48) - 1),
                            &[d, u],
                            none,
                        );
                    }
                    _ => return Err(format!("unbox to {rep:?}")),
                }
            }
            Op::IsType(_)
            | Op::IsFalsy
            | Op::SameBits
            | Op::ICmp(_)
            | Op::LCmp(_)
            | Op::FCmp(_) => {
                let r = res.unwrap();
                let (test, cond, targs) = self.test_of(r);
                let mut ops = vec![self.def(r)];
                for a in targs {
                    ops.push(self.use_(a));
                }
                self.push(MInst::Set { test, cond }, &ops, none);
            }
            Op::Select => {
                let r = res.unwrap();
                let float = f.ty(r).rep == Rep::F64;
                let ops = [
                    self.def(r),
                    self.use_(args[0]),
                    self.use_(args[1]),
                    self.use_(args[2]),
                ];
                self.push(MInst::Select { float }, &ops, none);
            }
            Op::Guard(set) if f.insts[i.idx()].tag == ExitTag::Entry => {
                let Some(Op::Load(r)) = f.def_op(args[0]) else {
                    return Err("an entry guard of no entry load".into());
                };
                self.entry_regs.push(r);
                let u = self.use_(args[0]);
                self.push(
                    MInst::Guard {
                        test: Test::Type(set),
                        cond: Cond::Eq,
                        exit: ENTRY_FAIL,
                    },
                    &[u],
                    none,
                );
            }
            Op::Guard(set) => {
                let u = self.use_(args[0]);
                self.push_exit(
                    i,
                    |exit| MInst::Guard {
                        test: Test::Type(set),
                        cond: Cond::Eq,
                        exit,
                    },
                    vec![u],
                    none,
                );
            }
            Op::GuardTrue | Op::GuardFalse => {
                let c = args[0];
                let (test, cond, targs) = if self.fused[c.idx()] {
                    self.test_of(c)
                } else {
                    (Test::B1, Cond::Ne, vec![c])
                };
                let cond = if op == Op::GuardTrue {
                    cond
                } else {
                    cond.invert()
                };
                let ops: Vec<Operand> = targs.iter().map(|&a| self.use_(a)).collect();
                self.push_exit(i, |exit| MInst::Guard { test, cond, exit }, ops, none);
            }
            Op::GuardSame => {
                let ops = vec![self.use_(args[0]), self.use_(args[1])];
                self.push_exit(
                    i,
                    |exit| MInst::Guard {
                        test: Test::X { imm: None },
                        cond: Cond::Eq,
                        exit,
                    },
                    ops,
                    none,
                );
            }
            Op::GuardNoClose => {
                self.push_exit(i, |exit| MInst::GuardNoClose { exit }, vec![], none)
            }
            Op::GcCheck => self.push_exit(i, |exit| MInst::GcCheck { exit }, vec![], none),
            Op::IAdd | Op::ISub => {
                let sub = op == Op::ISub;
                let r = res.unwrap();
                let d = self.def(r);
                if let Some(k) = self.imm12(args[1]) {
                    let (sub, k) = if k < 0 { (!sub, -k) } else { (sub, k) };
                    let n = self.use_(args[0]);
                    self.push_exit(
                        i,
                        |exit| MInst::AddImmOvf {
                            sub,
                            imm: k as u32,
                            exit,
                        },
                        vec![d, n],
                        none,
                    );
                } else {
                    let (n, m) = (self.use_(args[0]), self.use_(args[1]));
                    self.push_exit(i, |exit| MInst::AddOvf { sub, exit }, vec![d, n, m], none);
                }
            }
            Op::IMul => {
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push_exit(i, |exit| MInst::MulOvf { exit }, ops, none);
            }
            Op::INeg => {
                let ops = vec![self.def(res.unwrap()), self.use_(args[0])];
                self.push_exit(i, |exit| MInst::NegOvf { exit }, ops, none);
            }
            Op::IDivFloor | Op::IModFloor | Op::LDivFloor | Op::LModFloor => {
                let div = matches!(op, Op::IDivFloor | Op::LDivFloor);
                let sz = if matches!(op, Op::IDivFloor | Op::IModFloor) {
                    Sz::W
                } else {
                    Sz::X
                };
                let nonzero = self.imm12(args[1]).is_some_and(|k| k != 0)
                    || matches!(f.def_op(args[1]), Some(Op::KI32(n)) if n != 0)
                    || matches!(f.def_op(args[1]), Some(Op::KI64(n)) if n != 0);
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push_exit(
                    i,
                    |exit| MInst::DivMod {
                        div,
                        sz,
                        nonzero,
                        exit,
                    },
                    ops,
                    none,
                );
            }
            Op::IShl | Op::IShr => {
                let left = op == Op::IShl;
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push_exit(i, |exit| MInst::ShiftI32 { left, exit }, ops, none);
            }
            Op::LShl | Op::LShr => {
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push(
                    MInst::ShiftI64 {
                        left: op == Op::LShl,
                    },
                    &ops,
                    none,
                );
            }
            Op::IAddNo
            | Op::ISubNo
            | Op::IAnd
            | Op::IOr
            | Op::IXor
            | Op::IMulNo
            | Op::LAdd
            | Op::LSub
            | Op::LMul
            | Op::LUDiv
            | Op::LAnd
            | Op::LOr
            | Op::LXor => {
                let (alu, sz) = match op {
                    Op::IAddNo => (AluOp::Add, Sz::W),
                    Op::ISubNo => (AluOp::Sub, Sz::W),
                    Op::IMulNo => (AluOp::Mul, Sz::W),
                    Op::IAnd => (AluOp::And, Sz::W),
                    Op::IOr => (AluOp::Orr, Sz::W),
                    Op::IXor => (AluOp::Eor, Sz::W),
                    Op::LAdd => (AluOp::Add, Sz::X),
                    Op::LSub => (AluOp::Sub, Sz::X),
                    Op::LMul => (AluOp::Mul, Sz::X),
                    Op::LUDiv => (AluOp::Udiv, Sz::X),
                    Op::LAnd => (AluOp::And, Sz::X),
                    Op::LOr => (AluOp::Orr, Sz::X),
                    _ => (AluOp::Eor, Sz::X),
                };
                let r = res.unwrap();
                let d = self.def(r);
                if matches!(alu, AluOp::Add | AluOp::Sub)
                    && let Some(k) = self.imm12(args[1])
                {
                    let (alu, k) = if k < 0 {
                        (
                            if alu == AluOp::Add {
                                AluOp::Sub
                            } else {
                                AluOp::Add
                            },
                            -k,
                        )
                    } else {
                        (alu, k)
                    };
                    let n = self.use_(args[0]);
                    self.push(MInst::AluImm(alu, sz, k as u64), &[d, n], none);
                } else {
                    let (n, m) = (self.use_(args[0]), self.use_(args[1]));
                    self.push(MInst::Alu(alu, sz), &[d, n, m], none);
                }
            }
            Op::INot | Op::LNot => {
                let sz = if op == Op::INot { Sz::W } else { Sz::X };
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::Mvn(sz), &ops, none);
            }
            Op::LNeg => {
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::Neg(Sz::X), &ops, none);
            }
            Op::IToF | Op::LToF => {
                let sz = if op == Op::IToF { Sz::W } else { Sz::X };
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::Scvtf(sz), &ops, none);
            }
            Op::FToL => {
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::Fcvtzs(Sz::X), &ops, none);
            }
            Op::IToL => {
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::Sxtw, &ops, none);
            }
            Op::LToI => {
                let u = self.use_(args[0]);
                self.push_exit(i, |exit| MInst::LToI { exit }, vec![u], none);
            }
            Op::FToIExact => {
                let ops = vec![self.def(res.unwrap()), self.use_(args[0])];
                self.push_exit(i, |exit| MInst::FToIExact { exit }, ops, none);
            }
            Op::ToF64 => {
                let ops = vec![self.def(res.unwrap()), self.use_(args[0])];
                self.push_exit(i, |exit| MInst::ToF64 { exit }, ops, none);
            }
            Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
                let fop = match op {
                    Op::FAdd => FOp::Add,
                    Op::FSub => FOp::Sub,
                    Op::FMul => FOp::Mul,
                    _ => FOp::Div,
                };
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push(MInst::FAlu(fop), &ops, none);
            }
            Op::FIDiv => {
                let r = res.unwrap();
                let t = self.v.new_vreg(RegClass::Float);
                let ops = [Operand::reg_def(t), self.use_(args[0]), self.use_(args[1])];
                self.push(MInst::FAlu(FOp::Div), &ops, none);
                let ops = [self.def(r), Operand::reg_use(t)];
                self.push(MInst::FUn(FUnOp::Floor), &ops, none);
            }
            Op::FNeg | Op::FAbs | Op::FSqrt | Op::FFloor | Op::FCeil => {
                let u = match op {
                    Op::FNeg => FUnOp::Neg,
                    Op::FAbs => FUnOp::Abs,
                    Op::FSqrt => FUnOp::Sqrt,
                    Op::FFloor => FUnOp::Floor,
                    _ => FUnOp::Ceil,
                };
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::FUn(u), &ops, none);
            }
            Op::Helper(h @ (HelperId::FMod | HelperId::FPow)) => {
                let r = res.unwrap();
                let ops = [
                    Operand::reg_fixed_def(self.vr(r), preg_float(0)),
                    Operand::reg_fixed_use(self.vr(args[0]), preg_float(0)),
                    Operand::reg_fixed_use(self.vr(args[1]), preg_float(1)),
                ];
                let mut clob = caller_saved();
                clob.remove(preg_float(0));
                self.push(MInst::Helper(h), &ops, clob);
            }
            Op::Helper(h @ (HelperId::Lt | HelperId::Le | HelperId::Eq)) => {
                // `rt` goes in x0 at the call.
                let r = res.unwrap();
                let ops = [
                    Operand::reg_fixed_def(self.vr(r), preg_int(0)),
                    Operand::reg_fixed_use(self.vr(args[0]), preg_int(1)),
                    Operand::reg_fixed_use(self.vr(args[1]), preg_int(2)),
                ];
                let mut clob = caller_saved();
                clob.remove(preg_int(0));
                self.push(MInst::Helper(h), &ops, clob);
            }
            Op::Resume { c } => {
                let a = match self.f.blocks[self.f.insts[i.idx()].block.idx()].resume {
                    true => self.resume_a(i),
                    false => 0,
                };
                if c == 2 || c == 3 {
                    let ops: Vec<Operand> = f
                        .results(i)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|r| self.def(r))
                        .collect();
                    self.push(
                        MInst::Resume {
                            wanted: c - 1,
                            c,
                            a,
                        },
                        &ops,
                        none,
                    );
                } else if c == 1 {
                    self.push(MInst::Resume { wanted: 0, c, a }, &[], none);
                } else {
                    self.push(MInst::Resume { wanted: 0, c, a }, &[], caller_saved());
                }
            }
            Op::Jump => {
                self.push(MInst::Jump, &[], none);
            }
            Op::Br => {
                let c = args[0];
                let (test, cond, targs) = if self.fused[c.idx()] {
                    self.test_of(c)
                } else {
                    (Test::B1, Cond::Ne, vec![c])
                };
                let ops: Vec<Operand> = targs.iter().map(|&a| self.use_(a)).collect();
                self.push(MInst::Br { test, cond }, &ops, none);
            }
            Op::Call { a, nargs, pc, .. } => {
                let pc_after = self.pc_base + (pc as usize + 1) * 8;
                self.push(MInst::Call { a, nargs, pc_after }, &[], none);
            }
            Op::Return { a, n } => {
                self.push(MInst::Return { a, n }, &[], none);
            }
            Op::Deopt => self.push_exit(i, |exit| MInst::Deopt { exit }, vec![], none),
        }
        Ok(())
    }

    /// The `a` of the call a resume block resumes, from its predecessor's
    /// terminator.
    fn resume_a(&self, i: Inst) -> u8 {
        let b = self.f.insts[i.idx()].block;
        for &p in self.f.cfg().preds(b) {
            if let Some(t) = self.f.terminator(p)
                && let Op::Call { a, .. } = self.f.op(t)
            {
                return a;
            }
        }
        0
    }
}

#[allow(unused)]
fn unused(_: &Asm) {}

#[allow(unused)]
const _: TypeSet = TypeSet::ANY;
