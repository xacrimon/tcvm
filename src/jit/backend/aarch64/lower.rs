//! aarch64 instruction selection (`backend::lower`): compares fused into
//! the branch, guard, select or set that is their only use, small
//! immediates folded, and each IR instruction's `MInst`s.

use crate::jit::FastMap;
use crate::jit::backend::aarch64::abi::{allocatable, caller_saved, preg_float, preg_int};
use crate::jit::backend::aarch64::asm::{Cond, Shift, Sz, logical_imm};
use crate::jit::backend::aarch64::inst::{AluOp, ENTRY_FAIL, FOp, FUnOp, MInst, Test};
use crate::jit::backend::aarch64::types::is_single_test;
use crate::jit::backend::lower::{Known, Lower, Select, Tramp};
use crate::jit::backend::reg::{Operand, PRegSet, RegClass, VReg};
use crate::jit::backend::vcode::Remat;
use crate::jit::ir::ops::{Cc, ExitTag, HelperId, NO_PROTO, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, Inst, NO_SNAP, Val};

/// The selection's own state.
#[derive(Default)]
pub(crate) struct Isel {
    /// Selects that are an `abs` (`x < 0 ? -x : x`): the negation, whose
    /// exit an i32's takes (`find_abs`).
    abs_of: FastMap<Inst, Inst>,
}

pub(super) fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Lt,
        Cc::Le => Cond::Le,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
        Cc::Lo => Cond::Lo,
        Cc::Hi => Cond::Hi,
    }
}

/// After `fcmp`: false when unordered, except `ne`.
pub(super) fn float_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Mi,
        Cc::Le => Cond::Ls,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
        Cc::Lo | Cc::Hi => unreachable!("an unsigned float compare"),
    }
}

impl Select for Lower<'_, '_> {
    /// Compares fused into their users, `abs` selects, and products a
    /// `madd` or `msub` takes.
    fn find_patterns(&mut self, order: &[Block]) {
        self.find_fusions(order);
        self.find_abs(order);
        self.find_multiply_adds(order);
    }

    fn entry(&mut self) {
        self.push(MInst::Prologue, &[], PRegSet::empty());
        self.entry_params();
    }

    fn frame_closure(&mut self) {
        let c = self.v.new_vreg(RegClass::Int);
        self.push(
            MInst::FrameClosure,
            &[Operand::reg_def(c)],
            PRegSet::empty(),
        );
        self.closure = Some(c);
    }

    fn lower_inst(&mut self, i: Inst) -> Result<(), String> {
        let f = self.f;
        let op = f.op(i);
        let args: Vec<Val> = f.args(i).to_vec();
        let res = (f.insts[i.idx()].rn == 1).then(|| f.result(i));
        let none = PRegSet::empty();
        match op {
            Op::KVal(b) => self.constant(res.unwrap(), Remat::Int(b)),
            Op::KObj(p) => self.constant(res.unwrap(), Remat::Int(f.pool[p as usize].to_raw())),
            Op::KPtr(p) => self.constant(
                res.unwrap(),
                Remat::Int(f.pool[p as usize].to_raw() & crate::env::value::PAYLOAD_MASK),
            ),
            Op::KI32(n) => self.constant(res.unwrap(), Remat::Int(n as u32 as u64)),
            Op::KI64(n) => self.constant(res.unwrap(), Remat::Int(n as u64)),
            Op::KB1(c) => self.constant(res.unwrap(), Remat::Int(c as u64)),
            Op::KF64(b) => self.constant(res.unwrap(), Remat::Float(b)),
            Op::Home(r) => self.constant(res.unwrap(), Remat::Slot(r)),
            Op::Load(r) => {
                let d = self.def(res.unwrap());
                self.push(MInst::LoadSlot(r), &[d], none);
            }
            Op::Store(r) => self.store_slot(r, args[0]),
            // Defined by the entry's `Params`; nothing moves it (its effect).
            Op::Param(_) if f.insts[i.idx()].block == f.entry => {}
            Op::Param(_) => return Err("a parameter outside the entry block".into()),
            Op::UpvalValue(k) => {
                let d = self.def(res.unwrap());
                let c = Operand::reg_use(self.closure.expect("the frame's closure"));
                self.push(MInst::LoadUpvalOf(k), &[d, c], none);
            }
            Op::SetTop(at) => {
                let n = self.use_(args[0]);
                self.push(MInst::SetTop { at }, &[n], none);
            }
            Op::FrameFn => {
                let d = self.def(res.unwrap());
                let t = self.v.new_vreg(RegClass::Int);
                self.v.remat[t.vreg()] = Remat::Int(crate::env::value::FUNCTION_BITS);
                self.push(
                    MInst::MovImm(crate::env::value::FUNCTION_BITS),
                    &[Operand::reg_def(t)],
                    none,
                );
                let c = Operand::reg_use(self.closure.expect("the frame's closure"));
                self.push(
                    MInst::Alu(AluOp::Orr, Sz::X),
                    &[d, c, Operand::reg_use(t)],
                    none,
                );
            }
            Op::UpvalValueOf(k) | Op::UpvalCellOf(k) => {
                let (d, c) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::LoadUpvalOf(k), &[d, c], none);
            }
            Op::Box => {
                let x = args[0];
                let r = res.unwrap();
                match f.ty(x).rep {
                    Rep::I32 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::BoxI32, &[d, u], none);
                    }
                    Rep::F64 if self.in_d[r.idx()] => {}
                    Rep::F64 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::FmovToGpr, &[d, u], none);
                    }
                    Rep::B1 => {
                        let (d, u) = (self.def(r), self.use_(x));
                        self.push(MInst::BoxB1, &[d, u], none);
                    }
                    // The slow case calls through a trampoline: only x0
                    // changes.
                    Rep::I64 => {
                        let d = Operand::reg_fixed_def(self.vr(r), preg_int(0));
                        let u = Operand::reg_fixed_use(self.vr(x), preg_int(1));
                        let clob = none;
                        let slow = crate::jit::helpers::jit_box_i64 as *const () as usize;
                        self.needs_trampoline(slow, Tramp::Int);
                        if f.insts[i.idx()].snap != NO_SNAP {
                            self.push_exit(i, |exit| MInst::BoxI64 { exit }, vec![d, u], clob);
                        } else {
                            self.push(MInst::BoxI64 { exit: u32::MAX }, &[d, u], clob);
                        }
                    }
                    Rep::Val | Rep::Ptr => return Err("box of a boxed value".into()),
                }
            }
            // The operand's register (`alias`).
            Op::Refine(_) => {}
            Op::Unbox(rep) => {
                let x = args[0];
                let r = res.unwrap();
                match rep {
                    Rep::I32 => {}
                    Rep::F64 if self.alias(r) != r => {}
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
                        self.v.remat[d.vreg().vreg()] = Remat::AndImm {
                            src: u.vreg().vreg() as u32,
                            imm: crate::env::value::PAYLOAD_MASK,
                        };
                        self.push(
                            MInst::AluImm(AluOp::And, Sz::X, crate::env::value::PAYLOAD_MASK),
                            &[d, u],
                            none,
                        );
                    }
                    _ => return Err(format!("unbox to {rep:?}")),
                }
            }
            // Tests whose flags no branch, guard or select took: a boolean.
            Op::IsType(_)
            | Op::IsFalsy
            | Op::SameBits
            | Op::ShapeIs(_)
            | Op::ShapeIn(_)
            | Op::ICmp(_)
            | Op::LCmp(_)
            | Op::FCmp(_)
            | Op::HasFlag(_)
            | Op::NumMtSet
            | Op::TypeMtIs { .. }
            | Op::FnFixedLua
            | Op::FnNative
            | Op::FnFits
            | Op::TabHasRoom(_)
            | Op::TabNoKeys(_)
            | Op::TabHasMm(_) => {
                let r = res.unwrap();
                let (test, cond, targs) = self.test_of(r);
                let mut ops = vec![self.def(r)];
                for a in targs {
                    ops.push(self.use_(a));
                }
                self.push(MInst::Set { test, cond }, &ops, none);
            }
            Op::Select if self.isel.abs_of.contains_key(&i) => {
                let n = self.isel.abs_of[&i];
                let ops = vec![self.def(res.unwrap()), self.use_(args[2])];
                if f.op(n) == Op::INeg {
                    self.push_exit(n, |exit| MInst::AbsOvf { exit }, ops, none);
                } else {
                    self.push(MInst::Abs, &ops, none);
                }
            }
            Op::Select => {
                let r = res.unwrap();
                let float = f.ty(r).rep == Rep::F64;
                let (test, cond, targs) = self.cond_of(args[0]);
                let mut ops = vec![self.def(r), self.use_(args[1]), self.use_(args[2])];
                for a in targs {
                    ops.push(self.use_(a));
                }
                self.push(MInst::Select { test, cond, float }, &ops, none);
            }
            Op::Guard(set) if f.insts[i.idx()].tag == ExitTag::Entry => {
                let Some(Op::Load(r) | Op::Home(r) | Op::Param(r)) = f.def_op(args[0]) else {
                    return Err("an entry guard of no entry load".into());
                };
                self.entry_regs.push(r);
                if self.prefix_guards {
                    self.entry_tests.push((r, set));
                    return Ok(());
                }
                // The parameters stay where they came, for the failure.
                let mut ops = vec![self.use_(args[0])];
                for (k, &p) in self.params.iter().enumerate() {
                    ops.push(Operand::reg_fixed_use(p, preg_int(k as u8)));
                }
                self.push(
                    MInst::Guard {
                        test: Test::Type(set, false),
                        cond: Cond::Eq,
                        exit: ENTRY_FAIL,
                    },
                    &ops,
                    none,
                );
            }
            Op::Guard(set) if f.landings.contains(&i) => {
                let u = self.use_(args[0]);
                self.push_exit(i, |exit| MInst::GuardLanding { set, exit }, vec![u], none);
            }
            Op::Guard(set) => {
                let ops = args.iter().map(|&a| self.use_(a)).collect();
                self.push_exit(
                    i,
                    |exit| MInst::Guard {
                        test: Test::Type(set, args.len() > 1),
                        cond: Cond::Eq,
                        exit,
                    },
                    ops,
                    none,
                );
            }
            Op::GuardMiss => return Err("an unexpanded guard.miss".into()),
            Op::GuardTrue | Op::GuardFalse => {
                let (test, cond, targs) = self.cond_of(args[0]);
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
            Op::GuardNoFlag(mask) => self.push_exit(
                i,
                |exit| MInst::Guard {
                    test: Test::HasFlag(mask),
                    cond: Cond::Eq,
                    exit,
                },
                vec![],
                none,
            ),
            Op::ClassMm(k) => {
                let (d, t) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::ClassMm(k), &[d, t], none);
            }
            Op::StrLenOf => {
                let (d, t) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::StrLenOf, &[d, t], none);
            }
            Op::UpvalCell(k) => {
                let d = self.def(res.unwrap());
                let c = Operand::reg_use(self.closure.expect("the frame's closure"));
                self.push(MInst::LoadUpvalOf(k), &[d, c], none);
            }
            Op::CellLoad => {
                let (d, c) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::LoadCell, &[d, c], none);
            }
            Op::GuardShape(k) => {
                let shape = f.shapes[k as usize].as_word();
                let t = self.use_(args[0]);
                self.push_exit(
                    i,
                    |exit| MInst::Guard {
                        test: Test::Shape(shape),
                        cond: Cond::Eq,
                        exit,
                    },
                    vec![t],
                    none,
                );
            }
            Op::GuardShapeIn(set) => {
                let t = self.use_(args[0]);
                self.push_exit(
                    i,
                    |exit| MInst::Guard {
                        test: Test::ShapeIn(set),
                        cond: Cond::Eq,
                        exit,
                    },
                    vec![t],
                    none,
                );
            }
            Op::GuardProto(k) => {
                let proto = f.protos[k as usize].ptr.as_ptr() as u64;
                let fun = TypeSet::FUN.contains(f.ty(args[0]).set);
                let v = self.use_(args[0]);
                self.push_exit(
                    i,
                    |exit| MInst::GuardProto { proto, fun, exit },
                    vec![v],
                    none,
                );
            }
            Op::ClassBits(k) | Op::ClassIndex(k) => {
                let class = f.shapes[k as usize]
                    .mt_cache()
                    .ok_or("a class read of a shape without a metatable")?
                    .as_word();
                let d = self.def(res.unwrap());
                let m = if matches!(op, Op::ClassBits(_)) {
                    MInst::ClassBits(class)
                } else {
                    MInst::ClassIndex(class)
                };
                self.push(m, &[d], none);
            }
            Op::TabLoad { loc, .. } => {
                let (d, t) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::TabLoad(loc), &[d, t], none);
            }
            Op::TabStore { loc, .. } => {
                let (t, v) = (self.use_(args[0]), self.stored(args[1]));
                self.push(MInst::TabStore(loc), &[t, v], none);
            }
            Op::TabSetShape(k) => {
                let t = self.use_(args[0]);
                self.push(
                    MInst::TabSetShape(f.shapes[k as usize].as_word()),
                    &[t],
                    none,
                );
            }
            Op::TabASize | Op::TabArray => {
                let (d, t) = (self.def(res.unwrap()), self.use_(args[0]));
                let m = if op == Op::TabASize {
                    MInst::TabASize
                } else {
                    MInst::TabArray
                };
                self.push(m, &[d, t], none);
            }
            Op::ArrLoad => {
                let (d, a) = (self.def(res.unwrap()), self.use_(args[0]));
                match self.konst_int(args[1]) {
                    Some(k @ 0..4096) => self.push(MInst::ArrLoadAt(k as u16), &[d, a], none),
                    _ => {
                        let k = self.use_(args[1]);
                        self.push(MInst::ArrLoad, &[d, a, k], none)
                    }
                };
            }
            Op::ArrStore => {
                let a = self.use_(args[0]);
                match self.konst_int(args[1]) {
                    Some(k @ 0..4096) => {
                        let v = self.stored(args[2]);
                        self.push(MInst::ArrStoreAt(k as u16), &[a, v], none)
                    }
                    _ => {
                        let ops = [a, self.use_(args[1]), self.stored(args[2])];
                        self.push(MInst::ArrStore, &ops, none)
                    }
                };
            }
            Op::NamedNext => {
                let d = Operand::reg_def_at_start(self.vr(res.unwrap()));
                let mut ops = vec![d, self.use_(args[0]), self.use_(args[1])];
                for _ in 0..3 {
                    ops.push(Operand::reg_temp(self.v.new_vreg(RegClass::Int)));
                }
                self.push(MInst::NamedNext, &ops, none);
            }
            Op::NamedVal | Op::NamedKey => {
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                let m = if op == Op::NamedVal {
                    MInst::NamedVal
                } else {
                    MInst::NamedKey
                };
                self.push(m, &ops, none);
            }
            Op::ArrNext => {
                // Both written while the operands are still read.
                let d = Operand::reg_def_at_start(self.vr(res.unwrap()));
                let v = Operand::reg_temp(self.v.new_vreg(RegClass::Int));
                let ops = [d, self.use_(args[0]), self.use_(args[1]), v];
                self.push(MInst::ArrNext, &ops, none);
            }
            Op::Barrier => {
                let t = self.use_(args[0]);
                let slow = crate::jit::helpers::jit_barrier as *const () as usize;
                self.needs_trampoline(slow, Tramp::Args(2));
                self.push(MInst::Barrier, &[t], none);
            }
            Op::NumVarargs => {
                let d = self.def(res.unwrap());
                self.push(MInst::NumVarargs, &[d], none);
            }
            Op::VarargAt(k) => {
                let (d, nv) = (self.def(res.unwrap()), self.use_(args[0]));
                self.push(MInst::VarargAt(k), &[d, nv], none);
            }
            Op::VarargIdx => {
                let d = self.def(res.unwrap());
                let (nv, k) = (self.use_(args[0]), self.use_(args[1]));
                self.push(MInst::VarargIdx, &[d, nv, k], none);
            }
            Op::CellStore => {
                let (c, v) = (self.use_(args[0]), self.stored(args[1]));
                self.push(MInst::StoreCell, &[c, v], none);
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
                let k = self.konst_int(args[1]);
                let nonzero = k.is_some_and(|k| k != 0);
                let minus1 = k.is_none_or(|k| k == -1);
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
                        minus1,
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
            Op::LShl | Op::LShr
                if self
                    .konst_int(args[1])
                    .is_some_and(|k| (0..64).contains(&k)) =>
            {
                let k = self.konst_int(args[1]).unwrap() as u8;
                let sh = if op == Op::LShl {
                    Shift::Lsl
                } else {
                    Shift::Lsr
                };
                let ops = [self.def(res.unwrap()), self.use_(args[0])];
                self.push(MInst::ShiftImm(sh, Sz::X, k), &ops, none);
            }
            Op::IAsr | Op::LAsr => {
                let sz = if op == Op::IAsr { Sz::W } else { Sz::X };
                let d = self.def(res.unwrap());
                let n = self.use_(args[0]);
                if let Some(k) = self.konst_int(args[1]) {
                    let k = (k as u32 & (sz.bits() - 1)) as u8;
                    self.push(MInst::ShiftImm(Shift::Asr, sz, k), &[d, n], none);
                } else {
                    let m = self.use_(args[1]);
                    self.push(MInst::ShiftReg(Shift::Asr, sz), &[d, n, m], none);
                }
            }
            Op::IUMulShr(k) => {
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push(MInst::UMulShr(k), &ops, none);
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
                    && let Some((x, y, a)) = self.fused_product(&args)
                {
                    let (n, m, a) = (self.use_(x), self.use_(y), self.use_(a));
                    let sub = alu == AluOp::Sub;
                    self.push(MInst::MulAdd { sub, sz }, &[d, n, m, a], none);
                    return Ok(());
                }
                let logical = matches!(alu, AluOp::And | AluOp::Orr | AluOp::Eor)
                    .then(|| self.konst_int(args[1]))
                    .flatten()
                    .map(|k| {
                        if sz == Sz::W {
                            k as u32 as u64
                        } else {
                            k as u64
                        }
                    })
                    .filter(|&k| logical_imm(k, sz.bits()).is_some());
                if let Some(k) = logical {
                    let n = self.use_(args[0]);
                    self.push(MInst::AluImm(alu, sz, k), &[d, n], none);
                } else if matches!(alu, AluOp::Add | AluOp::Sub)
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
            Op::FToLExact => {
                let ops = vec![self.def(res.unwrap()), self.use_(args[0])];
                self.push_exit(i, |exit| MInst::FToLExact { exit }, ops, none);
            }
            Op::StrByteAt => {
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push_exit(i, |exit| MInst::StrByteAt { exit }, ops, none);
            }
            Op::ToPos(set) => {
                let t = f.ty(args[0]).set;
                let kinds = if t.is_empty() { set } else { set & t };
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Float)),
                ];
                if kinds.contains(TypeSet::STR) {
                    let slow = crate::jit::helpers::jit_str_to_i64 as *const () as usize;
                    self.needs_trampoline(slow, Tramp::Int);
                }
                self.push_exit(i, |exit| MInst::ToPos { exit, kinds }, ops, none);
            }
            Op::IFCmp(cc) => {
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Float)),
                ];
                self.push(MInst::IFCmp { cc }, &ops, none);
            }
            Op::NumCmp(cc) => {
                // Arms for the kinds the operands' types allow (numbers).
                let kinds = |v: Val| match f.ty(v).set & TypeSet::NUM {
                    k if k.is_empty() => TypeSet::NUM,
                    k => k,
                };
                let (kx, ky) = (kinds(args[0]), kinds(args[1]));
                let ops = [
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Float)),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Float)),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Int)),
                    Operand::reg_temp(self.v.new_vreg(RegClass::Int)),
                ];
                self.push(MInst::NumCmp { cc, kx, ky }, &ops, none);
            }
            Op::IRem | Op::LRem => {
                let sz = if op == Op::IRem { Sz::W } else { Sz::X };
                let nonzero = self.konst_int(args[1]).is_some_and(|k| k != 0);
                let ops = vec![
                    self.def(res.unwrap()),
                    self.use_(args[0]),
                    self.use_(args[1]),
                ];
                self.push_exit(i, |exit| MInst::Rem { sz, nonzero, exit }, ops, none);
            }
            Op::StrLt | Op::StrLe => {
                let d = Operand::reg_def_at_start(self.vr(res.unwrap()));
                let mut ops = vec![d, self.use_(args[0]), self.use_(args[1])];
                for _ in 0..5 {
                    ops.push(Operand::reg_temp(self.v.new_vreg(RegClass::Int)));
                }
                self.push(
                    MInst::StrCmp {
                        le: op == Op::StrLe,
                    },
                    &ops,
                    none,
                );
            }
            Op::ToF64(set) => {
                let mut ops = vec![self.def(res.unwrap())];
                ops.extend(args.iter().map(|&a| self.use_(a)));
                let kbox = args.len() > 1;
                // Arms only for the kinds the operand may have.
                let t = f.ty(args[0]).set;
                let kinds = if t.is_empty() { set } else { set & t };
                if kinds.contains(TypeSet::STR) {
                    let slow = crate::jit::helpers::jit_str_to_f64 as *const () as usize;
                    self.needs_trampoline(slow, Tramp::Int);
                }
                self.push_exit(i, |exit| MInst::ToF64 { exit, kbox, kinds }, ops, none);
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
            Op::Helper(HelperId::FMod) => {
                let d = Operand::reg_def_at_start(self.vr(res.unwrap()));
                let q = Operand::reg_temp(self.v.new_vreg(RegClass::Float));
                let ops = [d, q, self.use_(args[0]), self.use_(args[1])];
                self.needs_trampoline(crate::jit::helpers::address(HelperId::FMod), Tramp::Float);
                self.push(MInst::FMod, &ops, none);
            }
            Op::Helper(h) => {
                // Arguments go in order after the pinned registers and the
                // immediate, which the emitter moves in at the call. A cold
                // arm's trampoline keeps every other register.
                let sig = h.sig();
                let cold = f.blocks[f.insts[i.idx()].block.idx()].cold;
                let mut ops = Vec::with_capacity(args.len() + 1);
                let mut clob = if cold {
                    let mut s = PRegSet::empty();
                    for k in 0..sig.ctx.len() + sig.imm.is_some() as usize {
                        s.add(preg_int(k as u8));
                    }
                    s
                } else {
                    caller_saved()
                };
                if let Some(r) = res {
                    let p = if sig.ret == Some(Rep::F64) {
                        preg_float(0)
                    } else {
                        preg_int(0)
                    };
                    ops.push(Operand::reg_fixed_def(self.vr(r), p));
                    clob.remove(p);
                }
                let mut xn = (sig.ctx.len() + sig.imm.is_some() as usize) as u8;
                let mut dn = 0;
                for &a in &args {
                    let p = if f.ty(a).rep == Rep::F64 {
                        dn += 1;
                        preg_float(dn - 1)
                    } else {
                        xn += 1;
                        preg_int(xn - 1)
                    };
                    ops.push(Operand::reg_fixed_use(self.vr(a), p));
                }
                let m = if cold {
                    let kind = if sig.ret == Some(Rep::F64) {
                        Tramp::Float
                    } else {
                        Tramp::Int
                    };
                    self.needs_trampoline(crate::jit::helpers::address(h), kind);
                    MInst::HelperCold(h)
                } else {
                    MInst::Helper(h)
                };
                self.push(m, &ops, clob);
            }
            Op::Resume { c } => {
                let (a, hdr) = match self.f.blocks[self.f.insts[i.idx()].block.idx()].resume {
                    true => self.resume_a(i),
                    false => (0, 0),
                };
                // Nothing is in a register here: what lives across the call is
                // in the frame, and its reloads come after. The results land in
                // x0 and x1, which a clobber cannot name.
                let mut clob = allocatable();
                let ops: Vec<Operand> = if c == 2 || c == 3 {
                    let rs: Vec<Val> = f.results(i).collect();
                    rs.into_iter()
                        .enumerate()
                        .map(|(k, r)| {
                            let p = preg_int(k as u8);
                            clob.remove(p);
                            Operand::reg_fixed_def(self.vr(r), p)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let wanted = if c == 2 || c == 3 { c - 1 } else { 0 };
                self.push(MInst::Resume { wanted, c, a, hdr }, &ops, clob);
            }
            Op::Jump => {
                self.push(MInst::Jump, &[], none);
            }
            Op::Br => {
                // The encoder lays one edge's moves in line past the branch.
                if self.f.edges(i)[0].target == self.f.edges(i)[1].target {
                    return Err(format!("i{} branches to one block both ways", i.0));
                }
                let (test, cond, targs) = self.cond_of(args[0]);
                let ops: Vec<Operand> = targs.iter().map(|&a| self.use_(a)).collect();
                self.push(MInst::Br { test, cond }, &ops, none);
            }
            Op::Call { a, nargs, pc, .. } => {
                let pc_after = self.pc_after(i, pc);
                let known = self.known.get(&i).copied();
                // Arguments the entry guards would pass, missing ones nil.
                let typed = match known {
                    Some(Known::Lua { proto, nparams, .. }) => {
                        self.prefix_guards
                            && Some(proto) == self.own
                            && nargs != u8::MAX
                            && nparams <= 8
                            && (0..nparams as usize).all(|k| {
                                let set = args.get(k).map_or(TypeSet::NIL, |&x| f.ty(x).set);
                                self.entry_sets[k].contains(set)
                            })
                    }
                    _ => false,
                };
                // A known Lua callee takes its first arguments in x0..; any
                // other callee finds them in its slots.
                let ops: Vec<Operand> = if nargs == u8::MAX {
                    args.iter().map(|&x| self.use_(x)).collect()
                } else if matches!(known, Some(Known::Lua { .. })) {
                    (args.iter().enumerate())
                        .map(|(k, &x)| Operand::reg_fixed_use(self.vr(x), preg_int(k as u8)))
                        .collect()
                } else {
                    for (k, &x) in args.iter().enumerate() {
                        self.store_slot(a + 4 + k as u8, x);
                    }
                    Vec::new()
                };
                self.push_call(
                    i,
                    MInst::Call {
                        a,
                        nargs,
                        pc_after,
                        known,
                        typed,
                    },
                    ops,
                );
            }
            Op::Dispatch { pc, word } => {
                let pc_after = self.pc_base + (pc as usize + 1) * 8;
                let op = (word & 0xff) as u8;
                self.push(MInst::Dispatch { op, word, pc_after }, &[], none);
            }
            Op::TailEnter { nargs, proto } => {
                let p = self.use_(args[0]);
                let known = (proto != NO_PROTO).then(|| {
                    let p = f.protos[proto as usize];
                    (p.ptr.as_ptr() as u64, p.num_params)
                });
                self.push(MInst::TailEnter { nargs, known }, &[p], none);
            }
            Op::CallMm { pc, nargs, hdr, .. } => {
                let pc_after = self.pc_after(i, pc);
                let hdr = hdr as u16;
                let ops: Vec<Operand> = args.iter().map(|&a| self.use_(a)).collect();
                self.push_call(
                    i,
                    MInst::CallMm {
                        hdr,
                        nargs,
                        pc_after,
                    },
                    ops,
                );
            }
            Op::Return { a, n } => match self.returned(i) {
                Some(v) => {
                    let u = Operand::reg_fixed_use(self.vr(v), preg_int(0));
                    self.push(MInst::Return { a, n, x0: true }, &[u], none);
                }
                None => {
                    self.push(MInst::Return { a, n, x0: false }, &[], none);
                }
            },
            Op::Deopt => self.push_exit(i, |exit| MInst::Deopt { exit }, vec![], none),
            Op::LazyExit => self.push_exit(i, |exit| MInst::LazyExit { exit }, vec![], none),
        }
        Ok(())
    }
}

impl<'a, 'gc> Lower<'a, 'gc> {
    /// `R[r] = v`.
    fn store_slot(&mut self, r: u8, v: Val) {
        let u = self.stored(v);
        self.push(MInst::StoreSlot(r), &[u], PRegSet::empty());
    }

    /// Every parameter entries pass in registers, used or not: an entry
    /// guard's failure stores them all for the interpreter.
    fn entry_params(&mut self) {
        let f = self.f;
        if f.meta.params == 0 {
            return;
        }
        let regs: Vec<VReg> = (0..f.meta.params)
            .map(|_| self.v.new_vreg(RegClass::Int))
            .collect();
        for &i in f.insts_of(f.entry) {
            if let Op::Param(k) = f.op(i) {
                self.vreg[f.result(i).idx()] = Some(regs[k as usize]);
            }
        }
        self.params = regs;
        if self.params.is_empty() {
            return;
        }
        let ops: Vec<Operand> = (self.params.iter().enumerate())
            .map(|(k, &r)| Operand::reg_fixed_def(r, preg_int(k as u8)))
            .collect();
        self.push(MInst::Params, &ops, PRegSet::empty());
    }

    /// A constant: defined where the IR puts it, computed again by the
    /// allocator wherever that is cheaper than keeping it.
    fn constant(&mut self, r: Val, k: Remat) {
        let d = self.def(r);
        self.v.remat[d.vreg().vreg()] = k;
        let inst = match k {
            Remat::Int(v) => MInst::MovImm(v),
            Remat::Float(b) => MInst::FImm(b),
            Remat::Slot(r) => MInst::LoadSlot(r),
            Remat::None | Remat::AndImm { .. } => unreachable!(),
        };
        self.push(inst, &[d], PRegSet::empty());
    }

    /// Products a sum or difference in their block alone uses, which then
    /// computes them with `madd` or `msub`.
    fn find_multiply_adds(&mut self, order: &[Block]) {
        let f = self.f;
        for &b in order {
            for &i in f.insts_of(b) {
                let mul = match f.op(i) {
                    Op::IAddNo | Op::ISubNo => Op::IMulNo,
                    Op::LAdd | Op::LSub => Op::LMul,
                    _ => continue,
                };
                let sub = matches!(f.op(i), Op::ISubNo | Op::LSub);
                let args = f.args(i);
                for (k, &a) in args.iter().enumerate().rev() {
                    let other = args[1 - k];
                    if (sub && k == 0) || self.imm12(other).is_some() || self.fused[a.idx()] {
                        continue;
                    }
                    if let Some(d) = f.def_inst(a)
                        && f.op(d) == mul
                        && f.insts[d.idx()].block == b
                        && self.uses[a.idx()] == 1
                    {
                        self.fused[a.idx()] = true;
                        break;
                    }
                }
            }
        }
    }

    /// The product a sum or difference computes with it, and its other operand.
    fn fused_product(&self, args: &[Val]) -> Option<(Val, Val, Val)> {
        let f = self.f;
        (0..2).rev().find_map(|k| {
            let a = args[k];
            let d = f.def_inst(a)?;
            (self.fused[a.idx()] && matches!(f.op(d), Op::IMulNo | Op::LMul))
                .then(|| (f.args(d)[0], f.args(d)[1], args[1 - k]))
        })
    }

    /// Mark compares that only branches, guards and selects use, each
    /// computing the compare itself: flags feed a branch directly (`cmp`
    /// and `b.cond` fuse), where a boolean costs a `cset` and a test of it.
    fn find_fusions(&mut self, order: &[Block]) {
        let f = self.f;
        let mut tested = vec![0u32; f.vals.len()];
        for &b in order {
            for &i in f.insts_of(b) {
                let op = f.op(i);
                if !matches!(op, Op::Br | Op::GuardTrue | Op::GuardFalse | Op::Select) {
                    continue;
                }
                let c = f.args(i)[0];
                let Some(di) = f.def_inst(c) else {
                    continue;
                };
                let fusable = match f.op(di) {
                    Op::ICmp(_)
                    | Op::LCmp(_)
                    | Op::FCmp(_)
                    | Op::IsFalsy
                    | Op::SameBits
                    | Op::ShapeIs(_)
                    | Op::ShapeIn(_)
                    | Op::HasFlag(_)
                    | Op::NumMtSet
                    | Op::TypeMtIs { .. }
                    | Op::FnFixedLua
                    | Op::FnNative
                    | Op::FnFits
                    | Op::TabHasRoom(_)
                    | Op::TabNoKeys(_)
                    | Op::TabHasMm(_) => true,
                    // A select takes flags: no test that branches.
                    Op::IsType(set) => op != Op::Select || is_single_test(set),
                    _ => false,
                };
                debug_assert!(!fusable || f.op(di).folds());
                let mut n = 0;
                f.for_each_use(i, |v| n += (v == c) as u32);
                if fusable && n == 1 {
                    tested[c.idx()] += 1;
                }
            }
        }
        for (v, &n) in tested.iter().enumerate() {
            if n > 0 && n == self.uses[v] {
                self.fused[v] = true;
            }
        }
        // A one-bit `and` that only fused tests against zero use: they test
        // the bit (`tbz`), and it is not computed.
        let mut bit_tests = vec![0u32; f.vals.len()];
        for v in 0..f.vals.len() {
            if !self.fused[v] {
                continue;
            }
            if let Some(Op::ICmp(Cc::Eq | Cc::Ne)) = f.def_op(Val(v as u32)) {
                let args = f.args(f.def_inst(Val(v as u32)).unwrap());
                if self.imm12(args[1]) == Some(0) && self.bit_of(args[0]).is_some() {
                    bit_tests[args[0].idx()] += 1;
                }
            }
        }
        for (v, &n) in bit_tests.iter().enumerate() {
            if n > 0 && n == self.uses[v] {
                self.fused[v] = true;
            }
        }
    }

    /// Selects `x < 0 ? -x : x` whose negation only they use, in their
    /// block with nothing but pure instructions between: `cmp; cneg` (an
    /// i32's exit on `MIN` moves to the select, which no effect separates
    /// it from).
    fn find_abs(&mut self, order: &[Block]) {
        let f = self.f;
        for &b in order {
            let insts = f.insts_of(b);
            for (k, &s) in insts.iter().enumerate() {
                if f.op(s) != Op::Select {
                    continue;
                }
                let (c, neg, x) = (f.args(s)[0], f.args(s)[1], f.args(s)[2]);
                let below = match f.def_op(c) {
                    Some(Op::ICmp(Cc::Lt)) => Op::INeg,
                    Some(Op::LCmp(Cc::Lt)) => Op::LNeg,
                    _ => continue,
                };
                let ca = f.args(f.def_inst(c).unwrap());
                let zero = matches!(f.def_op(ca[1]), Some(Op::KI32(0) | Op::KI64(0)));
                let Some(n) = f.def_inst(neg) else {
                    continue;
                };
                let Some(at) = insts[..k].iter().position(|&j| j == n) else {
                    continue;
                };
                if ca[0] != x
                    || !zero
                    || f.op(n) != below
                    || f.args(n)[0] != x
                    || self.uses[neg.idx()] != 1
                    || !insts[at + 1..k].iter().all(|&j| f.op(j).is_pure())
                {
                    continue;
                }
                self.absorbed[n.idx()] = true;
                self.isel.abs_of.insert(s, n);
            }
        }
    }

    /// `v` as `x & (1 << k)`: `x` and `k`.
    fn bit_of(&self, v: Val) -> Option<(Val, u8)> {
        let d = self.f.def_inst(v)?;
        if self.f.op(d) != Op::IAnd {
            return None;
        }
        let args = self.f.args(d);
        (0..2).find_map(|k| match self.f.def_op(args[k]) {
            Some(Op::KI32(m)) if m > 0 && (m & (m - 1)) == 0 => {
                Some((args[1 - k], m.trailing_zeros() as u8))
            }
            _ => None,
        })
    }

    /// The test and operands computing B1 `c`.
    fn test_of(&mut self, c: Val) -> (Test, Cond, Vec<Val>) {
        let i = self.f.def_inst(c).expect("a fused compare");
        let args = self.f.args(i).to_vec();
        match self.f.op(i) {
            Op::ICmp(cc @ (Cc::Eq | Cc::Ne))
                if self.imm12(args[1]) == Some(0)
                    && self.fused[args[0].idx()]
                    && self.bit_of(args[0]).is_some() =>
            {
                let (x, k) = self.bit_of(args[0]).expect("a fused bit");
                (Test::Bit(k), int_cond(cc), vec![x])
            }
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
            Op::FCmp(cc) => {
                // `#0.0` compares as -0.0 does.
                let zero = |v: Val| matches!(self.f.def_op(v), Some(Op::KF64(b)) if f64::from_bits(b) == 0.0);
                if zero(args[1]) {
                    (Test::F64 { zero: true }, float_cond(cc), vec![args[0]])
                } else if zero(args[0]) {
                    (
                        Test::F64 { zero: true },
                        float_cond(cc.swap()),
                        vec![args[1]],
                    )
                } else {
                    (Test::F64 { zero: false }, float_cond(cc), args)
                }
            }
            Op::SameBits => (Test::X { imm: None }, Cond::Eq, args),
            Op::IsType(set) => (Test::Type(set, args.len() > 1), Cond::Eq, args),
            Op::IsFalsy => (Test::Falsy(args.len() > 1), Cond::Lo, args),
            Op::ShapeIs(k) => (
                Test::Shape(self.f.shapes[k as usize].as_word()),
                Cond::Eq,
                args,
            ),
            Op::ShapeIn(k) => (Test::ShapeIn(k), Cond::Eq, args),
            Op::HasFlag(mask) => (Test::HasFlag(mask), Cond::Ne, args),
            Op::NumMtSet => (Test::NumMtSet, Cond::Ne, args),
            Op::TypeMtIs { slot, word } => (Test::TypeMtIs { slot, word }, Cond::Eq, args),
            Op::FnFixedLua => (Test::FnFixedLua, Cond::Eq, args),
            Op::FnNative => (Test::FnNative, Cond::Hi, args),
            Op::FnFits => (Test::FnFits, Cond::Ls, args),
            Op::TabHasRoom(loc) => (Test::TabHasRoom(loc), Cond::Hi, args),
            Op::TabNoKeys(parts) => (Test::TabNoKeys(parts), Cond::Eq, args),
            Op::TabHasMm(k) => (Test::MmBit(k), Cond::Ne, args),
            op => panic!("test_of {op:?}"),
        }
    }

    /// The test of B1 `c`: its compare's when fused, else of the boolean.
    fn cond_of(&mut self, c: Val) -> (Test, Cond, Vec<Val>) {
        if self.fused[c.idx()] {
            self.test_of(c)
        } else {
            (Test::B1, Cond::Ne, vec![c])
        }
    }

    /// An integer constant usable as a compare immediate (negative ones
    /// become `cmn`).
    fn imm12(&self, v: Val) -> Option<i64> {
        self.konst_int(v).filter(|n| n.unsigned_abs() < 4096)
    }
}
