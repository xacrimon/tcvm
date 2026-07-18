//! The lowering driver: bytecode -> SSA, with basic-block versioning.
//!
//! Two phases over the same [`step`] transfer function (see `sink.rs`):
//!
//!  1. **Analyze.** A worklist over `(pc, TypeContext)` discovers which block
//!     versions exist and what each one's entry types are. Emits no IR, so a
//!     context that turns out to be wrong costs nothing to discard.
//!  2. **Emit.** Walk each version once with contexts already final. Every
//!     branch target resolves to a version the analysis already decided on, so
//!     there is no patching and no rebuilding.
//!
//! # Why there is no separate loop fixpoint
//!
//! Versioning subsumes it. When a back-edge reaches a header with a context the
//! existing version doesn't accept, we don't *join* the two — we mint a second
//! version. So
//!
//! ```lua
//! local s = 0                      -- INT on entry
//! for i = 1, n do s = s + 0.5 end  -- FLOAT from iteration 2
//! ```
//!
//! gets a header version typed `s: INT` (entered once, from outside) and a
//! second typed `s: FLOAT` (which self-loops, since float + float is float).
//! That is **loop peeling, for free** — strictly better than joining to a boxed
//! `NUM`, which is what a classic fixpoint would have converged to.
//!
//! Termination comes from [`VERSION_CAP`] rather than from lattice height: at
//! most `CAP` specialized versions per pc, after which every further context
//! routes to one fully-generic version that accepts anything. So the version
//! count is bounded by `(CAP + 1) * code.len()` and the worklist drains.

use std::collections::HashMap;

use foldhash::fast::RandomState;

use crate::dmm::Gc;
use crate::env::function::{InlineCache, Prototype};
use crate::env::shape::MetamethodBits;
use crate::env::value::KindSet;
use crate::env::value::{Value, ValueKind};
use crate::instruction::Instruction;
use crate::jit::frontend::cfg::{self, Cfg, Term, Unsupported};
use crate::jit::frontend::sink::{
    self, Decline, Feedback, IcFeedback, RegState, Scalar, Sink, compare, step, to_val,
};
use crate::jit::ir::op::{ArithKind, Cc, FloatOp, IntOp, Op};
use crate::jit::ir::pool::{ConstPool, ConstRef, ProtoRef, ShapeRef, StrRef};
use crate::jit::ir::ty::{Refine, Rep, Ty, TypeContext, TypeSet};
use crate::jit::ir::{Block, BlockCall, Exit, FrameState, Func, InstData, Val};

/// Specialized versions allowed per bytecode pc before we give up and route
/// everything to one generic version. Higgs used 5; the number is arbitrary,
/// and it is the knob that trades code size against specialization.
pub const VERSION_CAP: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerError {
    Cfg(Unsupported),
    Declined(Decline),
}

impl From<Unsupported> for LowerError {
    fn from(e: Unsupported) -> Self {
        LowerError::Cfg(e)
    }
}

impl From<Decline> for LowerError {
    fn from(e: Decline) -> Self {
        LowerError::Declined(e)
    }
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

/// Feedback drawn from the running VM: the inline caches the interpreter has
/// already filled, plus the concrete types of the live registers at the moment
/// compilation was triggered.
pub struct VmFeedback<'gc, 'a> {
    proto: Gc<'gc, Prototype<'gc>>,
    pool: &'a mut ConstPool<'gc>,
    /// Register types observed on the live stack at the compile trigger.
    entry: Vec<Ty>,
}

impl<'gc, 'a> VmFeedback<'gc, 'a> {
    pub fn new(
        proto: Gc<'gc, Prototype<'gc>>,
        pool: &'a mut ConstPool<'gc>,
        entry: Vec<Ty>,
    ) -> Self {
        VmFeedback { proto, pool, entry }
    }

    pub fn entry_ty(&self, reg: u8) -> Ty {
        self.entry.get(reg as usize).copied().unwrap_or(Ty::ANY)
    }
}

/// The type of a concrete constant, refined to the constant itself.
fn const_ty<'gc>(pool: &mut ConstPool<'gc>, v: Value<'gc>) -> (ConstRef, Ty) {
    let set = TypeSet::of_value(v);
    let c = pool.intern_value(v);
    (
        c,
        Ty {
            rep: Rep::Val,
            set,
            refine: Refine::Const(c),
        },
    )
}

/// Translate the interpreter's observed value kinds into the JIT's lattice.
/// Booleans widen back out to `FALSE | TRUE` — the record collapses them, and
/// nothing downstream of a field load cares which.
fn kinds_to_types(k: KindSet) -> TypeSet {
    let mut t = TypeSet::empty();
    for (bit, set) in [
        (KindSet::NIL, TypeSet::NIL),
        (KindSet::BOOLEAN, TypeSet::BOOL),
        (KindSet::INTEGER, TypeSet::INT),
        (KindSet::FLOAT, TypeSet::FLOAT),
        (KindSet::STRING, TypeSet::STR),
        (KindSet::TABLE, TypeSet::TAB),
        (KindSet::FUNCTION, TypeSet::FUN),
        (KindSet::THREAD, TypeSet::THR),
        (KindSet::USERDATA, TypeSet::UDATA),
    ] {
        if k.contains(bit) {
            t |= set;
        }
    }
    t
}

impl<'gc, 'a> Feedback for VmFeedback<'gc, 'a> {
    fn ic(&mut self, ic_idx: u16, mm: MetamethodBits) -> Option<IcFeedback> {
        let cache = self.proto.ic_table.get(ic_idx as usize)?.get();
        let InlineCache::Mono { shape, slot } = cache else {
            return None;
        };
        // An absent slot means the key isn't in this shape: a get would consult
        // `__index`, and a set would *transition* the shape. Neither compiles to
        // a slot access.
        if slot == InlineCache::ABSENT_SLOT {
            return None;
        }
        // The interpreter only bypasses the metamethod when the slot's value is
        // non-nil, which we can't know at compile time. So if the shape already
        // carries `mm`, decline — emitting `assume.no_mm` here would assert
        // something that is already false.
        if shape.has_mm(mm) {
            return None;
        }
        // `intern_shape` refuses dict sentinels, which are shared across every
        // dict table with the same metatable and so prove nothing about layout.
        let s = self.pool.intern_shape(shape)?;
        let types = self
            .proto
            .ic_types
            .get(ic_idx as usize)
            .map(|c| kinds_to_types(c.get()))
            .unwrap_or(TypeSet::ANY);
        Some(IcFeedback {
            shape: s,
            slot,
            types,
        })
    }

    fn constant(&mut self, idx: u16) -> (ConstRef, Ty) {
        let v = self.proto.constants[idx as usize];
        const_ty(self.pool, v)
    }

    fn scalar(&mut self, idx: u16) -> Option<Scalar> {
        let v = self.proto.constants[idx as usize];
        match v.kind() {
            ValueKind::Integer => Some(Scalar::Int(v.get_integer()?)),
            ValueKind::Float => Some(Scalar::Float(v.get_float()?)),
            _ => None,
        }
    }

    fn proto(&mut self, idx: u16) -> ProtoRef {
        self.pool.intern_proto(self.proto.prototypes[idx as usize])
    }

    fn key(&mut self, idx: u16) -> StrRef {
        let v = self.proto.constants[idx as usize];
        let s = v
            .get_string()
            .expect("global key must be a string constant");
        self.pool.intern_string(s)
    }

    fn for_step(&mut self, base: u8) -> Option<i64> {
        let reg = base.checked_add(2)?;
        let mut step = None;
        let mut defs = Vec::new();
        for i in self.proto.code.iter().copied() {
            defs.clear();
            cfg::reg_defs(i, &mut defs);
            if !defs.contains(&reg) {
                continue;
            }
            // The only write we accept is a LOAD of an integer constant. Anything
            // else writing the step register (including a second `for` loop that
            // reuses this base with a different step) means we can't pin the sign
            // down, so we decline the loop.
            let Instruction::LOAD { idx, .. } = i else {
                return None;
            };
            let v = self.proto.constants[idx as usize].get_integer()?;
            if step.is_some_and(|prev| prev != v) {
                return None;
            }
            step = Some(v);
        }
        step
    }
}

// ---------------------------------------------------------------------------
// Phase 1 sink: types only
// ---------------------------------------------------------------------------

/// The analysis instantiation: a "value" *is* its type, and every method is a
/// lattice computation that emits nothing.
struct TySink;

impl Sink for TySink {
    type V = Ty;

    fn ty(&self, v: Ty) -> Ty {
        v
    }
    fn begin_inst(&mut self, _pc: u32, _regs: &[Option<Ty>]) {}

    fn kconst(&mut self, _c: ConstRef, ty: Ty) -> Ty {
        ty
    }
    fn iconst(&mut self, _v: i64) -> Ty {
        Ty::I64
    }
    fn fconst(&mut self, _v: f64) -> Ty {
        Ty::F64
    }
    fn bconst(&mut self, _v: bool) -> Ty {
        Ty::B1
    }
    fn knil(&mut self) -> Ty {
        Ty::boxed(TypeSet::NIL)
    }

    fn pack_int(&mut self, _v: Ty) -> Ty {
        Ty::boxed(TypeSet::INT)
    }
    fn pack_float(&mut self, _v: Ty) -> Ty {
        Ty::boxed(TypeSet::FLOAT)
    }
    fn pack_bool(&mut self, _v: Ty) -> Ty {
        Ty::boxed(TypeSet::BOOL)
    }
    fn unpack_int(&mut self, _v: Ty) -> Ty {
        Ty::I64
    }
    fn unpack_float(&mut self, _v: Ty) -> Ty {
        Ty::F64
    }
    fn sitofp(&mut self, _v: Ty) -> Ty {
        Ty::F64
    }

    fn int_arith(&mut self, _op: IntOp, _a: Ty, _b: Ty) -> Ty {
        Ty::I64
    }
    fn float_arith(&mut self, _op: FloatOp, _a: Ty, _b: Ty) -> Ty {
        Ty::F64
    }
    fn icmp(&mut self, _cc: Cc, _a: Ty, _b: Ty) -> Ty {
        Ty::B1
    }
    fn fcmp(&mut self, _cc: Cc, _a: Ty, _b: Ty) -> Ty {
        Ty::B1
    }
    fn is_falsy(&mut self, _v: Ty) -> Ty {
        Ty::B1
    }
    fn guard_nonzero(&mut self, _v: Ty) {}

    fn guard_type(&mut self, v: Ty, set: TypeSet) -> Ty {
        v.refined_to(set)
    }
    fn guard_shape(&mut self, _v: Ty, s: ShapeRef) -> Ty {
        Ty::with_shape(s)
    }
    fn assume_no_mm(&mut self, _s: ShapeRef, _bits: MetamethodBits) {}
    fn tab_props(&mut self, _t: Ty) -> Ty {
        Ty::PTR
    }
    fn slot_get(&mut self, _p: Ty, _slot: u32) -> Ty {
        Ty::ANY
    }
    fn slot_set(&mut self, _p: Ty, _slot: u32, _v: Ty) {}
    fn tab_new(&mut self) -> Ty {
        Ty::boxed(TypeSet::TAB)
    }

    fn lua_arith(&mut self, _op: ArithKind, _a: Ty, _b: Ty) -> Ty {
        // A metamethod can return anything.
        Ty::ANY
    }
    fn lua_unary(&mut self, _op: ArithKind, _a: Ty) -> Ty {
        Ty::ANY
    }
    fn lua_cmp(&mut self, _cc: Cc, _a: Ty, _b: Ty) -> Ty {
        Ty::B1
    }
    fn lua_eq(&mut self, _a: Ty, _b: Ty) -> Ty {
        Ty::B1
    }
    fn lua_concat(&mut self, _a: Ty, _b: Ty) -> Ty {
        Ty::ANY
    }
    fn lua_len(&mut self, _a: Ty) -> Ty {
        Ty::ANY
    }
    fn lua_get_index(&mut self, _t: Ty, _k: Ty) -> Ty {
        Ty::ANY
    }
    fn lua_set_index(&mut self, _t: Ty, _k: Ty, _v: Ty) {}

    fn upval_get(&mut self, _idx: u8) -> Ty {
        Ty::ANY
    }
    fn upval_set(&mut self, _idx: u8, _v: Ty) {}
    fn upval_close(&mut self, _start: u8) {}
    fn get_global(&mut self, _upval: u8, _key: StrRef) -> Ty {
        Ty::ANY
    }
    fn set_global(&mut self, _upval: u8, _key: StrRef, _v: Ty) {}

    fn stack_get(&mut self, _reg: u8) -> Ty {
        Ty::ANY
    }
    fn stack_set(&mut self, _reg: u8, _v: Ty) {}

    fn call(&mut self, _args: &[Ty], nret: u8) -> Vec<Ty> {
        vec![Ty::ANY; nret as usize]
    }
    fn closure_new(&mut self, p: ProtoRef) -> Ty {
        Ty {
            rep: Rep::Val,
            set: TypeSet::FUN,
            refine: Refine::Proto(p),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2 sink: emit IR
// ---------------------------------------------------------------------------

struct IrSink<'f, 'gc> {
    func: &'f mut Func<'gc>,
    block: Block,
    /// Current bytecode pc and register map, kept in sync by `begin_inst` so a
    /// guard can build its `FrameState` on demand.
    pc: u32,
    regs: Vec<Option<Val>>,
    /// The version's entry context, recorded on every exit so a hot exit can
    /// tell the recompiler which specialization was wrong.
    ctx: TypeContext,
}

impl<'f, 'gc> IrSink<'f, 'gc> {
    fn emit(&mut self, op: Op, args: Vec<Val>, result: Option<Ty>) -> Option<Val> {
        let needs_fs = op.needs_frame_state();
        let fs = needs_fs.then(|| {
            let fs = FrameState {
                pc: self.pc,
                regs: self.regs.clone(),
                parent: None,
            };
            self.func.add_frame_state(fs)
        });
        let exit = (op.is_guard() || op == Op::Deopt).then(|| {
            let e = Exit {
                fs: fs.expect("a guard always carries a FrameState"),
                ctx: self.ctx.clone(),
                count: 0,
            };
            self.func.add_exit(e)
        });

        let tys: Vec<Ty> = result.into_iter().collect();
        let data = InstData {
            op,
            args,
            targets: Vec::new(),
            results: Vec::new(),
            fs,
            exit,
        };
        let (_, results) = self.func.append_inst(self.block, data, &tys);
        results.first().copied()
    }

    fn emit1(&mut self, op: Op, args: Vec<Val>, ty: Ty) -> Val {
        self.emit(op, args, Some(ty)).expect("op has one result")
    }

    fn emit0(&mut self, op: Op, args: Vec<Val>) {
        self.emit(op, args, None);
    }
}

impl<'f, 'gc> Sink for IrSink<'f, 'gc> {
    type V = Val;

    fn ty(&self, v: Val) -> Ty {
        self.func.ty(v)
    }

    fn begin_inst(&mut self, pc: u32, regs: &[Option<Val>]) {
        self.pc = pc;
        self.regs.clear();
        self.regs.extend_from_slice(regs);
    }

    fn kconst(&mut self, c: ConstRef, ty: Ty) -> Val {
        self.emit1(Op::KConst(c), vec![], ty)
    }
    fn iconst(&mut self, v: i64) -> Val {
        self.emit1(Op::IConst(v), vec![], Ty::I64)
    }
    fn fconst(&mut self, v: f64) -> Val {
        self.emit1(Op::FConst(v.to_bits()), vec![], Ty::F64)
    }
    fn bconst(&mut self, v: bool) -> Val {
        self.emit1(Op::BConst(v), vec![], Ty::B1)
    }
    fn knil(&mut self) -> Val {
        let c = self.func.pool.intern_value(Value::nil());
        self.emit1(Op::KConst(c), vec![], Ty::boxed(TypeSet::NIL))
    }

    fn pack_int(&mut self, v: Val) -> Val {
        self.emit1(Op::PackInt, vec![v], Ty::boxed(TypeSet::INT))
    }
    fn pack_float(&mut self, v: Val) -> Val {
        self.emit1(Op::PackFloat, vec![v], Ty::boxed(TypeSet::FLOAT))
    }
    fn pack_bool(&mut self, v: Val) -> Val {
        self.emit1(Op::PackBool, vec![v], Ty::boxed(TypeSet::BOOL))
    }
    fn unpack_int(&mut self, v: Val) -> Val {
        self.emit1(Op::UnpackInt, vec![v], Ty::I64)
    }
    fn unpack_float(&mut self, v: Val) -> Val {
        self.emit1(Op::UnpackFloat, vec![v], Ty::F64)
    }
    fn sitofp(&mut self, v: Val) -> Val {
        self.emit1(Op::SiToFp, vec![v], Ty::F64)
    }

    fn int_arith(&mut self, op: IntOp, a: Val, b: Val) -> Val {
        // Unary ops pass the same operand twice; drop the duplicate.
        let args = if matches!(op, IntOp::Neg | IntOp::BNot) {
            vec![a]
        } else {
            vec![a, b]
        };
        self.emit1(Op::IntArith(op), args, Ty::I64)
    }
    fn float_arith(&mut self, op: FloatOp, a: Val, b: Val) -> Val {
        let args = if matches!(op, FloatOp::Neg) {
            vec![a]
        } else {
            vec![a, b]
        };
        self.emit1(Op::FloatArith(op), args, Ty::F64)
    }
    fn icmp(&mut self, cc: Cc, a: Val, b: Val) -> Val {
        self.emit1(Op::ICmp(cc), vec![a, b], Ty::B1)
    }
    fn fcmp(&mut self, cc: Cc, a: Val, b: Val) -> Val {
        self.emit1(Op::FCmp(cc), vec![a, b], Ty::B1)
    }
    fn is_falsy(&mut self, v: Val) -> Val {
        self.emit1(Op::IsFalsy, vec![v], Ty::B1)
    }
    fn guard_nonzero(&mut self, v: Val) {
        let zero = self.iconst(0);
        let ne = self.emit1(Op::ICmp(Cc::Ne), vec![v, zero], Ty::B1);
        self.emit0(Op::GuardCond, vec![ne]);
    }

    fn guard_type(&mut self, v: Val, set: TypeSet) -> Val {
        let ty = self.func.ty(v).refined_to(set);
        self.emit1(Op::GuardType(set), vec![v], ty)
    }
    fn guard_shape(&mut self, v: Val, s: ShapeRef) -> Val {
        self.emit1(Op::GuardShape(s), vec![v], Ty::with_shape(s))
    }
    fn assume_no_mm(&mut self, s: ShapeRef, bits: MetamethodBits) {
        self.emit0(Op::AssumeNoMm(s, bits), vec![]);
    }
    fn tab_props(&mut self, t: Val) -> Val {
        self.emit1(Op::TabProps, vec![t], Ty::PTR)
    }
    fn slot_get(&mut self, p: Val, slot: u32) -> Val {
        self.emit1(Op::SlotGet(slot), vec![p], Ty::ANY)
    }
    fn slot_set(&mut self, p: Val, slot: u32, v: Val) {
        self.emit0(Op::SlotSet(slot), vec![p, v]);
    }
    fn tab_new(&mut self) -> Val {
        self.emit1(
            Op::TabNew { array_hint: 0 },
            vec![],
            Ty::boxed(TypeSet::TAB),
        )
    }

    fn lua_arith(&mut self, op: ArithKind, a: Val, b: Val) -> Val {
        self.emit1(Op::LuaArith(op), vec![a, b], Ty::ANY)
    }
    fn lua_unary(&mut self, op: ArithKind, a: Val) -> Val {
        self.emit1(Op::LuaArith(op), vec![a], Ty::ANY)
    }
    fn lua_cmp(&mut self, cc: Cc, a: Val, b: Val) -> Val {
        self.emit1(Op::LuaCmp(cc), vec![a, b], Ty::B1)
    }
    fn lua_eq(&mut self, a: Val, b: Val) -> Val {
        self.emit1(Op::LuaEq, vec![a, b], Ty::B1)
    }
    fn lua_concat(&mut self, a: Val, b: Val) -> Val {
        self.emit1(Op::LuaConcat, vec![a, b], Ty::ANY)
    }
    fn lua_len(&mut self, a: Val) -> Val {
        self.emit1(Op::LuaLen, vec![a], Ty::ANY)
    }
    fn lua_get_index(&mut self, t: Val, k: Val) -> Val {
        self.emit1(Op::LuaGetIndex, vec![t, k], Ty::ANY)
    }
    fn lua_set_index(&mut self, t: Val, k: Val, v: Val) {
        self.emit0(Op::LuaSetIndex, vec![t, k, v]);
    }

    fn upval_get(&mut self, idx: u8) -> Val {
        let cell = self.emit1(Op::UpvalCell(idx), vec![], Ty::PTR);
        self.emit1(Op::UpvalGet, vec![cell], Ty::ANY)
    }
    fn upval_set(&mut self, idx: u8, v: Val) {
        let cell = self.emit1(Op::UpvalCell(idx), vec![], Ty::PTR);
        self.emit0(Op::UpvalSet, vec![cell, v]);
    }
    fn upval_close(&mut self, start: u8) {
        self.emit0(Op::UpvalClose(start), vec![]);
    }
    fn get_global(&mut self, _upval: u8, key: StrRef) -> Val {
        self.emit1(Op::GetGlobal(key), vec![], Ty::ANY)
    }
    fn set_global(&mut self, _upval: u8, key: StrRef, v: Val) {
        self.emit0(Op::SetGlobal(key), vec![v]);
    }

    fn stack_get(&mut self, reg: u8) -> Val {
        self.emit1(Op::StackGet(reg), vec![], Ty::ANY)
    }
    fn stack_set(&mut self, reg: u8, v: Val) {
        self.emit0(Op::StackSet(reg), vec![v]);
    }

    fn call(&mut self, args: &[Val], nret: u8) -> Vec<Val> {
        let tys = vec![Ty::ANY; nret as usize];
        let fs = FrameState {
            pc: self.pc,
            regs: self.regs.clone(),
            parent: None,
        };
        let fs = self.func.add_frame_state(fs);
        let data = InstData {
            op: Op::Call { nret },
            args: args.to_vec(),
            targets: Vec::new(),
            results: Vec::new(),
            fs: Some(fs),
            exit: None,
        };
        let (_, results) = self.func.append_inst(self.block, data, &tys);
        results
    }
    fn closure_new(&mut self, p: ProtoRef) -> Val {
        self.emit1(
            Op::ClosureNew(p),
            vec![],
            Ty {
                rep: Rep::Val,
                set: TypeSet::FUN,
                refine: Refine::Proto(p),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

/// What a block's terminator does, with a distinct register state per edge —
/// `TESTSET` assigns its destination on only one of its two edges, and
/// `FORLOOP` advances the counter on only one.
enum TermOut<V: Copy> {
    Jump(u32, RegState<V>),
    Br {
        cond: V,
        t: (u32, RegState<V>),
        f: (u32, RegState<V>),
    },
    Ret(Vec<V>),
}

/// Lower a terminator. Shared by both phases, like `step`.
fn terminator<S: Sink, F: Feedback>(
    s: &mut S,
    st: &mut RegState<S::V>,
    fb: &mut F,
    code: &[Instruction],
    pc: u32,
    term: Term,
) -> Result<TermOut<S::V>, Decline> {
    let i = code[pc as usize];
    s.begin_inst(pc, &st.regs);

    Ok(match term {
        Term::Jump(t) => TermOut::Jump(t, st.clone()),

        Term::Return => {
            let Instruction::RETURN { values, count } = i else {
                // STOP, or TAILCALL (which the CFG pass admits but we don't
                // lower yet).
                return Err(Decline::Op(pc));
            };
            let mut out = Vec::new();
            for n in 0..count.saturating_sub(1) {
                let v = st.get(s, values + n);
                out.push(to_val(s, v));
            }
            TermOut::Ret(out)
        }

        Term::Branch { skip, jump, .. } => match i {
            // Every compare/test skips the paired JMP iff `result != inverted`,
            // so an inverted test just swaps the edges — no negation op needed.
            Instruction::EQ { lhs, rhs, inverted } => {
                let (a, b) = (st.get(s, lhs), st.get(s, rhs));
                let c = compare(s, Cc::Eq, a, b);
                let (t, f) = if inverted { (jump, skip) } else { (skip, jump) };
                TermOut::Br {
                    cond: c,
                    t: (t, st.clone()),
                    f: (f, st.clone()),
                }
            }
            Instruction::LT { lhs, rhs, inverted } => {
                let (a, b) = (st.get(s, lhs), st.get(s, rhs));
                let c = compare(s, Cc::Lt, a, b);
                let (t, f) = if inverted { (jump, skip) } else { (skip, jump) };
                TermOut::Br {
                    cond: c,
                    t: (t, st.clone()),
                    f: (f, st.clone()),
                }
            }
            Instruction::LE { lhs, rhs, inverted } => {
                let (a, b) = (st.get(s, lhs), st.get(s, rhs));
                let c = compare(s, Cc::Le, a, b);
                let (t, f) = if inverted { (jump, skip) } else { (skip, jump) };
                TermOut::Br {
                    cond: c,
                    t: (t, st.clone()),
                    f: (f, st.clone()),
                }
            }
            Instruction::TEST { src, inverted } => {
                let v = st.get(s, src);
                let falsy = sink::falsy(s, v);
                // `falsy` is the negation of the test's `truthy`, so the edges
                // land opposite to the compare ops.
                let (t, f) = if inverted { (skip, jump) } else { (jump, skip) };
                TermOut::Br {
                    cond: falsy,
                    t: (t, st.clone()),
                    f: (f, st.clone()),
                }
            }
            // Assigns `dst = src` only on the edge that does *not* skip. This
            // is exactly the conditional definition that block parameters make
            // expressible: we hand the assigned value along one edge and the
            // incoming value along the other.
            Instruction::TESTSET { dst, src, inverted } => {
                let v = st.get(s, src);
                let falsy = sink::falsy(s, v);

                let mut assigned = st.clone();
                assigned.set(s, dst, v);

                let (t, f) = if inverted {
                    ((skip, st.clone()), (jump, assigned))
                } else {
                    ((jump, assigned), (skip, st.clone()))
                };
                TermOut::Br { cond: falsy, t, f }
            }

            Instruction::FORPREP { base, .. } => {
                let (cond, body) = for_prep(s, st, fb, pc, base)?;
                let exit = st.clone();
                // `skip` is the fall-through into the body; `jump` skips the
                // loop entirely.
                TermOut::Br {
                    cond,
                    t: (skip, body),
                    f: (jump, exit),
                }
            }
            Instruction::FORLOOP { base, .. } => {
                let (cond, body) = for_loop(s, st, fb, pc, base)?;
                let exit = st.clone();
                // `jump` is the back-edge into the body; `skip` falls out.
                TermOut::Br {
                    cond,
                    t: (jump, body),
                    f: (skip, exit),
                }
            }
            _ => return Err(Decline::Op(pc)),
        },

        Term::Fallthrough(t) => TermOut::Jump(t, st.clone()),
    })
}

/// The loop's control registers, unpacked, plus the constant step. We require a
/// compile-time-known integer step: its *sign* decides whether the test is `<=`
/// or `>=`, and an unboxed integer loop is the entire point — a boxed fallback
/// would need a generic loop op we don't have, so we decline instead.
fn for_control<S: Sink, F: Feedback>(
    s: &mut S,
    st: &mut RegState<S::V>,
    fb: &mut F,
    pc: u32,
    base: u8,
) -> Result<(S::V, S::V, i64), Decline> {
    let init = st.get(s, base);
    let limit = st.get(s, base + 1);

    // The step comes from the bytecode, not the register state: it is invariant
    // by construction, so it need not survive a merge in the type lattice — and
    // therefore need not be part of the versioning key.
    let step_val = fb.for_step(base).ok_or(Decline::ForLoop(pc))?;
    if step_val == 0 {
        return Err(Decline::ForLoop(pc));
    }
    if s.ty(init).set != TypeSet::INT || s.ty(limit).set != TypeSet::INT {
        return Err(Decline::ForLoop(pc));
    }

    // `as_int`, not `unpack_int`: on the back-edge these are already raw i64s,
    // and unpacking a value that was never packed is nonsense.
    let i = sink::as_int(s, init);
    let l = sink::as_int(s, limit);
    Ok((i, l, step_val))
}

fn for_prep<S: Sink, F: Feedback>(
    s: &mut S,
    st: &mut RegState<S::V>,
    fb: &mut F,
    pc: u32,
    base: u8,
) -> Result<(S::V, RegState<S::V>), Decline> {
    let (i, l, step) = for_control(s, st, fb, pc, base)?;

    let cc = if step > 0 { Cc::Le } else { Cc::Ge };
    let cond = s.icmp(cc, i, l);

    // The counter and the visible loop variable (base+3) stay *unboxed*: their
    // block-parameter type becomes `i64`, so the back-edge carries a raw
    // integer and the body never re-tags it.
    let mut body = st.clone();
    body.set(s, base, i);
    body.set(s, base + 3, i);
    Ok((cond, body))
}

fn for_loop<S: Sink, F: Feedback>(
    s: &mut S,
    st: &mut RegState<S::V>,
    fb: &mut F,
    pc: u32,
    base: u8,
) -> Result<(S::V, RegState<S::V>), Decline> {
    let (i, l, step) = for_control(s, st, fb, pc, base)?;

    let k = s.iconst(step);
    let next = s.int_arith(IntOp::Add, i, k);
    let cc = if step > 0 { Cc::Le } else { Cc::Ge };
    let cond = s.icmp(cc, next, l);

    let mut body = st.clone();
    body.set(s, base, next);
    body.set(s, base + 3, next);
    Ok((cond, body))
}

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

struct Version {
    pc: u32,
    ctx: TypeContext,
    /// The fallback version for this pc: every parameter is `any`, so it accepts
    /// any edge (packing as needed). Minted once the version cap is hit, and the
    /// reason the worklist terminates.
    generic: bool,
    /// Filled by emission.
    block: Option<Block>,
    /// Successor versions, in the order the terminator produces them.
    succs: Vec<usize>,
}

struct Versions {
    all: Vec<Version>,
    /// pc -> version ids, in creation order.
    by_pc: HashMap<u32, Vec<usize>, RandomState>,
}

impl Versions {
    fn new() -> Self {
        Versions {
            all: Vec::new(),
            by_pc: HashMap::default(),
        }
    }

    /// Find the version at `pc` for exactly this `ctx`, or mint one. Past
    /// [`VERSION_CAP`] specialized versions we mint a single fully-generic
    /// version instead, which accepts everything — that, not lattice height, is
    /// what bounds the version count and makes the worklist terminate.
    ///
    /// A specialized version serves **one** entry context, matched exactly. It is
    /// tempting to reuse a version whose parameters merely *subsume* the incoming
    /// context — a `tab` parameter can certainly hold a `tab<S0>` — but that
    /// silently widens the argument on the edge and throws the refinement away.
    /// Do it to a shape and the loop header stops knowing the receiver's layout,
    /// so the body re-guards on every iteration; do it to a representation and the
    /// accumulator gets re-tagged on every back-edge. Refusing mints a second,
    /// sharper version that self-loops — the peeled first iteration pays for the
    /// fact, and the steady state runs without it.
    fn resolve(&mut self, pc: u32, ctx: TypeContext, work: &mut Vec<usize>) -> usize {
        let ids = self.by_pc.entry(pc).or_default();

        // The generic version is the fallback of last resort: routing to it throws
        // all specialization away, so it only wins when nothing else matches.
        let mut generic = None;
        for &id in ids.iter() {
            if self.all[id].generic {
                generic = Some(id);
            } else if self.all[id].ctx == ctx {
                return id;
            }
        }

        if ids.len() < VERSION_CAP {
            let id = self.all.len();
            ids.push(id);
            self.all.push(Version {
                pc,
                ctx,
                generic: false,
                block: None,
                succs: Vec::new(),
            });
            work.push(id);
            return id;
        }

        if let Some(id) = generic {
            return id;
        }

        let id = self.all.len();
        ids.push(id);
        self.all.push(Version {
            pc,
            ctx: TypeContext(ctx.0.iter().map(|_| Ty::ANY).collect()),
            generic: true,
            block: None,
            succs: Vec::new(),
        });
        work.push(id);
        id
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Registers captured as `ParentLocal` by a `CLOSURE` in this prototype. An
/// open upvalue names a stack slot by index, so these cannot live in SSA — the
/// frontend routes them through `StackGet`/`StackSet`.
fn pinned_regs(proto: &Prototype<'_>) -> Vec<bool> {
    let mut pinned = vec![false; 256];
    for i in proto.code.iter() {
        if let Instruction::CLOSURE { proto: idx, .. } = *i {
            for d in proto.prototypes[idx as usize].upvalue_desc.iter() {
                if let crate::instruction::UpValueDescriptor::ParentLocal(r) = *d {
                    pinned[r as usize] = true;
                }
            }
        }
    }
    pinned
}

/// The parameters of a block compiled from `pc`: its live-in registers, minus
/// any that are stack-pinned (those live in memory, not SSA).
fn params_of(cfg: &Cfg, pinned: &[bool], pc: u32) -> Vec<u8> {
    cfg.block_at(pc)
        .live_in
        .iter()
        .copied()
        .filter(|&r| !pinned[r as usize])
        .collect()
}

pub fn lower<'gc>(
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
    entry_types: Vec<Ty>,
) -> Result<Func<'gc>, LowerError> {
    let cfg = cfg::build(&proto.code)?;
    let pinned = pinned_regs(&proto);
    let mut pool = ConstPool::new();

    let mut versions = {
        let mut fb = VmFeedback::new(proto, &mut pool, entry_types.clone());
        analyze(&proto, &cfg, &pinned, &mut fb, entry_pc)?
    };

    let mut func = Func::new();
    {
        let mut fb = VmFeedback::new(proto, &mut pool, entry_types);
        emit(&proto, &cfg, &pinned, &mut fb, &mut func, &mut versions)?;
    }
    func.pool = pool;
    func.pinned_regs = pinned
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p)
        .map(|(r, _)| r as u8)
        .collect();
    func.entry_regs = params_of(&cfg, &pinned, entry_pc);
    Ok(func)
}

fn analyze<'gc>(
    proto: &Prototype<'gc>,
    cfg: &Cfg,
    pinned: &[bool],
    fb: &mut VmFeedback<'gc, '_>,
    entry_pc: u32,
) -> Result<Versions, LowerError> {
    let mut versions = Versions::new();
    let mut work = Vec::new();

    let entry_ctx = TypeContext(
        params_of(cfg, pinned, entry_pc)
            .iter()
            .map(|&r| fb.entry_ty(r).for_context())
            .collect(),
    );
    versions.resolve(entry_pc, entry_ctx, &mut work);

    let mut s = TySink;
    while let Some(id) = work.pop() {
        let pc = versions.all[id].pc;
        let ctx = versions.all[id].ctx.clone();
        let mut st: RegState<Ty> = RegState::new(256, pinned.to_vec());
        for (n, &r) in params_of(cfg, pinned, pc).iter().enumerate() {
            st.regs[r as usize] = Some(ctx.0[n]);
        }

        let succs = run_block(proto, cfg, fb, &mut s, &mut st, pc, |target, out_st| {
            let ctx = TypeContext(
                params_of(cfg, pinned, target)
                    .iter()
                    .map(|&r| out_st.regs[r as usize].unwrap_or(Ty::ANY).for_context())
                    .collect(),
            );
            versions.resolve(target, ctx, &mut work)
        })?;

        versions.all[id].succs = succs;
    }

    Ok(versions)
}

/// Walk one block's instructions, then its terminator, invoking `edge` for each
/// successor. Shared by both phases; `edge` is what differs (analysis resolves
/// a version, emission records the block call).
fn run_block<'gc, S: Sink, E>(
    proto: &Prototype<'gc>,
    cfg: &Cfg,
    fb: &mut VmFeedback<'gc, '_>,
    s: &mut S,
    st: &mut RegState<S::V>,
    pc: u32,
    mut edge: E,
) -> Result<Vec<usize>, LowerError>
where
    E: FnMut(u32, &RegState<S::V>) -> usize,
{
    let block = cfg.block_at(pc);
    let out = run_body(proto, cfg, fb, s, st, pc)?;

    let _ = block;
    Ok(match out {
        TermOut::Jump(t, st) => vec![edge(t, &st)],
        TermOut::Br { t, f, .. } => vec![edge(t.0, &t.1), edge(f.0, &f.1)],
        TermOut::Ret(_) => vec![],
    })
}

/// Execute a block's instructions and its terminator. Shared verbatim by both
/// phases, so they cannot disagree about what a block does.
fn run_body<'gc, S: Sink>(
    proto: &Prototype<'gc>,
    cfg: &Cfg,
    fb: &mut VmFeedback<'gc, '_>,
    s: &mut S,
    st: &mut RegState<S::V>,
    pc: u32,
) -> Result<TermOut<S::V>, LowerError> {
    let b = cfg.block_at(pc);
    let tpc = term_pc(cfg, &proto.code, pc)?;
    // Instructions up to the terminator. A compare's paired JMP sits *after* the
    // terminator pc and is consumed by it, so it never appears here either.
    let body_end = tpc.unwrap_or(b.end);

    for p in b.start..body_end {
        let i = proto.code[p as usize];
        s.begin_inst(p, &st.regs);
        step(s, st, fb, p, i)?;
    }

    Ok(match tpc {
        Some(tp) => {
            let term = cfg::terminator(&proto.code, tp)?;
            terminator(s, st, fb, &proto.code, tp, term)?
        }
        // Runs off the end into the next block.
        None => TermOut::Jump(b.end, st.clone()),
    })
}

/// The pc of the block's terminating instruction, if it has one.
///
/// `None` means the block runs off its end into the next one — every
/// instruction in it is a plain effect, and the edge is an unconditional jump to
/// `b.end`. Conflating that with "the last instruction is the terminator" would
/// silently skip executing that instruction.
fn term_pc(cfg: &Cfg, code: &[Instruction], pc: u32) -> Result<Option<u32>, Unsupported> {
    let b = cfg.block_at(pc);
    for p in b.start..b.end {
        match cfg::terminator(code, p)? {
            Term::Fallthrough(_) => continue,
            _ => return Ok(Some(p)),
        }
    }
    Ok(None)
}

fn emit<'gc>(
    proto: &Prototype<'gc>,
    cfg: &Cfg,
    pinned: &[bool],
    fb: &mut VmFeedback<'gc, '_>,
    func: &mut Func<'gc>,
    versions: &mut Versions,
) -> Result<(), LowerError> {
    // Create every block up front so edges can name blocks that aren't emitted
    // yet (back-edges, forward branches).
    for id in 0..versions.all.len() {
        let block = if id == 0 {
            func.entry
        } else {
            func.new_block()
        };
        let pc = versions.all[id].pc;
        let ctx = versions.all[id].ctx.clone();
        for &ty in &ctx.0 {
            func.append_param(block, ty);
        }
        func.block_mut(block).origin = Some(crate::jit::ir::Origin { pc, ctx });
        versions.all[id].block = Some(block);
    }

    for id in 0..versions.all.len() {
        let pc = versions.all[id].pc;
        let block = versions.all[id].block.unwrap();
        let succs = versions.all[id].succs.clone();

        let mut st: RegState<Val> = RegState::new(256, pinned.to_vec());
        let params = params_of(cfg, pinned, pc);
        let block_params = func.block(block).params.clone();
        for (n, &r) in params.iter().enumerate() {
            st.regs[r as usize] = Some(block_params[n]);
        }

        let mut s = IrSink {
            func,
            block,
            pc,
            regs: st.regs.clone(),
            ctx: versions.all[id].ctx.clone(),
        };

        // Same walk as the analysis, with the IR sink. Each edge's out-state
        // becomes a BlockCall.
        let out = run_body(proto, cfg, fb, &mut s, &mut st, pc)?;

        match out {
            TermOut::Jump(_, out_st) => {
                let call = block_call(&mut s, versions, succs[0], cfg, pinned, &out_st);
                let data = InstData {
                    op: Op::Jump,
                    args: vec![],
                    targets: vec![call],
                    results: vec![],
                    fs: None,
                    exit: None,
                };
                s.func.append_inst(block, data, &[]);
            }
            TermOut::Br { cond, t, f } => {
                let ct = block_call(&mut s, versions, succs[0], cfg, pinned, &t.1);
                let cf = block_call(&mut s, versions, succs[1], cfg, pinned, &f.1);
                let data = InstData {
                    op: Op::Br,
                    args: vec![cond],
                    targets: vec![ct, cf],
                    results: vec![],
                    fs: None,
                    exit: None,
                };
                s.func.append_inst(block, data, &[]);
            }
            TermOut::Ret(vals) => {
                let data = InstData {
                    op: Op::Ret,
                    args: vals,
                    targets: vec![],
                    results: vec![],
                    fs: None,
                    exit: None,
                };
                s.func.append_inst(block, data, &[]);
            }
        }
    }
    Ok(())
}

/// Build the argument list for an edge, coercing each value to the target
/// version's parameter type. Only ever *packs*: an unboxed target accepts only
/// its own representation (see `accepts`), so a mismatch means the target is
/// boxed.
fn block_call(
    s: &mut IrSink<'_, '_>,
    versions: &Versions,
    target: usize,
    cfg: &Cfg,
    pinned: &[bool],
    st: &RegState<Val>,
) -> BlockCall {
    let v = &versions.all[target];
    let params = params_of(cfg, pinned, v.pc);
    let mut args = Vec::with_capacity(params.len());
    for (n, &r) in params.iter().enumerate() {
        let val = st.regs[r as usize].expect("live-in register must be defined on this edge");
        let want = v.ctx.0[n];
        let have = s.ty(val);
        let coerced = if have.rep == want.rep {
            val
        } else {
            debug_assert_eq!(want.rep, Rep::Val, "unboxed target must match exactly");
            to_val(s, val)
        };
        args.push(coerced);
    }
    BlockCall {
        block: v.block.expect("blocks are created before emission"),
        args,
    }
}

impl<V: Copy> Clone for RegState<V> {
    fn clone(&self) -> Self {
        RegState {
            regs: self.regs.clone(),
            pinned: self.pinned.clone(),
        }
    }
}
