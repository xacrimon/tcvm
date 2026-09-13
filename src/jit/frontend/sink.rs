//! The per-opcode transfer function, written once.
//!
//! Lowering runs in two phases: an analysis that computes each block version's
//! entry type context to a fixpoint (emitting no IR), then an emission that
//! builds the IR with those contexts already final. The hazard in that design
//! is the two phases *disagreeing* — analysis concluding an `ADD` yields `INT`
//! while emission emits a float add. Then the versioning key is a lie and the
//! block parameter types are wrong.
//!
//! So the per-opcode logic lives in exactly one place: [`step`], generic over a
//! [`Sink`]. The analysis instantiates it with `V = Ty` (a "value" *is* its
//! type; every method just computes the result type and emits nothing), and
//! emission instantiates it with `V = ir::Val` (a real SSA value). The
//! specialization decision — int add or generic add? — is made by one branch of
//! one function, so the phases cannot diverge. Not by convention: structurally.

use crate::env::shape::MetamethodBits;
use crate::instruction::{Instruction, Op};
use crate::jit::ir::op::{ArithKind, Cc, FloatOp, IntOp};
use crate::jit::ir::pool::{ConstRef, ProtoRef, ShapeRef, StrRef};
use crate::jit::ir::ty::{Rep, Ty, TypeSet};

/// What lowering knows about the code it is compiling, before it starts.
///
/// Deliberately thin. The seed types come from the *live stack* at the moment
/// compilation is triggered — we are always compiling a running frame, so the
/// concrete type of every live register is right there — and the shapes come
/// from the inline caches the interpreter has already filled. No separate
/// profiler.
pub trait Feedback {
    /// The shape, slot, and observed value types at an IC site.
    ///
    /// `None` unless the site is monomorphic, the shape is specializable (never
    /// a dict sentinel), the slot is present, *and* the shape's metatable does
    /// not currently carry `mm`. That last condition matters: the interpreter's
    /// fast path only bypasses `__index` when the loaded value is non-nil, so a
    /// shape that already has the metamethod cannot be compiled to a bare slot
    /// access — and `assume.no_mm` would be asserting something already false.
    fn ic(&mut self, ic_idx: u16, mm: MetamethodBits) -> Option<IcFeedback>;
    /// Intern constant `idx` from the prototype's constant table.
    fn constant(&mut self, idx: u16) -> (ConstRef, Ty);
    /// Constant `idx` as an unboxed scalar, if it is one.
    ///
    /// A `LOAD` of a number puts the raw `i64`/`f64` in the register rather than
    /// the tagged `Value` the interpreter would have written. The register file is
    /// ours to represent as we like — only the *boundaries* (calls, returns,
    /// stores, deopt) have to agree with the interpreter — and loading numbers
    /// boxed means immediately unpacking them at the first use, and worse,
    /// splitting a loop header into a boxed version and an unboxed one just
    /// because a counter was initialized from a constant.
    fn scalar(&mut self, idx: u16) -> Option<Scalar>;
    /// Intern a sub-prototype for `CLOSURE`.
    fn proto(&mut self, idx: u16) -> ProtoRef;
    /// Intern a constant as a string key, for global access.
    fn key(&mut self, idx: u16) -> StrRef;
    /// The constant step of a numeric `for` loop whose control registers start
    /// at `base`, resolved from the bytecode rather than from the type lattice.
    ///
    /// Lua reserves `base..base+2` as the loop's internal control registers and
    /// never lets the body write them, so the step is invariant by construction
    /// and its defining `LOAD` is the only write. Reading it from the bytecode
    /// is why `Refine::Const` no longer has to survive a merge — and therefore
    /// no longer has to be part of the versioning key.
    fn for_step(&mut self, base: u8) -> Option<i64>;
}

/// A constant with an unboxed representation. Booleans are deliberately absent:
/// `Rep::B1` is a condition, not a Lua value, and a register that reaches a deopt
/// must hold something the frame writer can turn back into a `Value`.
#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    Int(i64),
    Float(f64),
}

/// What an inline cache tells us about one access site.
#[derive(Clone, Copy, Debug)]
pub struct IcFeedback {
    pub shape: ShapeRef,
    pub slot: u32,
    /// Value kinds this site has actually loaded. Empty when the site has never
    /// run — a shape proves *where* a field lives, never *what* it holds, so
    /// this is what lets the arithmetic consuming a field read specialize.
    pub types: TypeSet,
}

/// A construct lowering declines to compile. Refusing is always legal: guard
/// failure deopts to the interpreter anyway, so the region simply stays
/// interpreted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decline {
    /// A numeric `for` whose control values aren't all integers with a
    /// compile-time-known step sign. The unboxed loop is the whole point; a
    /// boxed fallback would need a generic loop op we don't have.
    ForLoop(u32),
    /// Lua 5.5's global-declaration check.
    ErrNNil(u32),
    /// Reached an opcode with no lowering.
    Op(u32),
}

/// Receives the ops the transfer function decides to emit.
///
/// The `V` associated type is what makes one function serve both phases:
/// analysis sets `V = Ty` and every method is a lattice computation; emission
/// sets `V = ir::Val` and every method appends an instruction.
pub trait Sink {
    type V: Copy;

    fn ty(&self, v: Self::V) -> Ty;

    /// Called before each bytecode instruction. Emission uses this to keep the
    /// `FrameState` it hands to guards in sync with the register map; analysis
    /// ignores it.
    fn begin_inst(&mut self, pc: u32, regs: &[Option<Self::V>]);

    fn kconst(&mut self, c: ConstRef, ty: Ty) -> Self::V;
    fn iconst(&mut self, v: i64) -> Self::V;
    fn fconst(&mut self, v: f64) -> Self::V;
    fn bconst(&mut self, v: bool) -> Self::V;
    fn knil(&mut self) -> Self::V;

    fn pack_int(&mut self, v: Self::V) -> Self::V;
    fn pack_float(&mut self, v: Self::V) -> Self::V;
    fn pack_bool(&mut self, v: Self::V) -> Self::V;
    fn unpack_int(&mut self, v: Self::V) -> Self::V;
    fn unpack_float(&mut self, v: Self::V) -> Self::V;
    fn sitofp(&mut self, v: Self::V) -> Self::V;

    fn int_arith(&mut self, op: IntOp, a: Self::V, b: Self::V) -> Self::V;
    fn float_arith(&mut self, op: FloatOp, a: Self::V, b: Self::V) -> Self::V;
    fn icmp(&mut self, cc: Cc, a: Self::V, b: Self::V) -> Self::V;
    fn fcmp(&mut self, cc: Cc, a: Self::V, b: Self::V) -> Self::V;
    fn is_falsy(&mut self, v: Self::V) -> Self::V;
    /// Guard a nonzero divisor: integer `//` and `%` raise on zero.
    fn guard_nonzero(&mut self, v: Self::V);

    /// Exit unless the value's type is in `set`. Produces a refined value.
    fn guard_type(&mut self, v: Self::V, set: TypeSet) -> Self::V;
    fn guard_shape(&mut self, v: Self::V, s: ShapeRef) -> Self::V;
    fn assume_no_mm(&mut self, s: ShapeRef, bits: MetamethodBits);
    fn tab_props(&mut self, t: Self::V) -> Self::V;
    fn slot_get(&mut self, p: Self::V, slot: u32) -> Self::V;
    fn slot_set(&mut self, p: Self::V, slot: u32, v: Self::V);
    fn tab_new(&mut self) -> Self::V;

    fn lua_arith(&mut self, op: ArithKind, a: Self::V, b: Self::V) -> Self::V;
    fn lua_unary(&mut self, op: ArithKind, a: Self::V) -> Self::V;
    fn lua_cmp(&mut self, cc: Cc, a: Self::V, b: Self::V) -> Self::V;
    fn lua_eq(&mut self, a: Self::V, b: Self::V) -> Self::V;
    fn lua_concat(&mut self, a: Self::V, b: Self::V) -> Self::V;
    fn lua_len(&mut self, a: Self::V) -> Self::V;
    fn lua_get_index(&mut self, t: Self::V, k: Self::V) -> Self::V;
    fn lua_set_index(&mut self, t: Self::V, k: Self::V, v: Self::V);

    fn upval_get(&mut self, idx: u8) -> Self::V;
    fn upval_set(&mut self, idx: u8, v: Self::V);
    fn upval_close(&mut self, start: u8);
    fn get_global(&mut self, upval: u8, key: StrRef) -> Self::V;
    fn set_global(&mut self, upval: u8, key: StrRef, v: Self::V);

    /// A stack-pinned register: one captured by a `CLOSURE` in this prototype.
    /// Open upvalues address slots by index, so such a register cannot live in
    /// SSA — every read and write goes through memory.
    fn stack_get(&mut self, reg: u8) -> Self::V;
    fn stack_set(&mut self, reg: u8, v: Self::V);

    fn call(&mut self, args: &[Self::V], nret: u8) -> Vec<Self::V>;
    fn closure_new(&mut self, p: ProtoRef) -> Self::V;
}

/// The symbolic register file. Pinned registers are absent from it — they live
/// in memory, and `get`/`set` route them through the sink.
pub struct RegState<V> {
    pub regs: Vec<Option<V>>,
    pub pinned: Vec<bool>,
}

impl<V: Copy> RegState<V> {
    pub fn new(size: usize, pinned: Vec<bool>) -> Self {
        RegState {
            regs: vec![None; size],
            pinned,
        }
    }

    pub fn get<S: Sink<V = V>>(&mut self, s: &mut S, r: u8) -> V {
        if self.pinned[r as usize] {
            return s.stack_get(r);
        }
        match self.regs[r as usize] {
            Some(v) => v,
            // Reading a register the frontend never defined and liveness didn't
            // list as live-in. Well-formed bytecode doesn't do this, but nil is
            // what the interpreter would see.
            None => s.knil(),
        }
    }

    /// A pinned register's home is the thread's value stack, which holds tagged
    /// `Value`s — an open upvalue or the collector may read that slot at any
    /// time. So a pinned store packs, while an SSA register keeps whatever
    /// representation specialization gave it.
    pub fn set<S: Sink<V = V>>(&mut self, s: &mut S, r: u8, v: V) {
        if self.pinned[r as usize] {
            let v = to_val(s, v);
            s.stack_set(r, v);
            return;
        }
        self.regs[r as usize] = Some(v);
    }

    /// Replace a register's value with a *refined* one — same value, narrower
    /// type, as produced by a guard. Skips pinned registers: their home is
    /// memory, and re-storing an identical value would emit a pointless write
    /// on every iteration.
    pub fn refine(&mut self, r: u8, v: V) {
        if self.pinned[r as usize] {
            return;
        }
        self.regs[r as usize] = Some(v);
    }
}

/// Coerce `v` to a boxed `Value`. Only ever needs to *pack*: the lattice's
/// `join` degrades a representation mismatch to `Rep::Val`, so a merge target
/// is boxed whenever the edges disagree.
pub fn to_val<S: Sink>(s: &mut S, v: S::V) -> S::V {
    match s.ty(v).rep {
        Rep::Val | Rep::Ptr => v,
        Rep::I64 => s.pack_int(v),
        Rep::F64 => s.pack_float(v),
        Rep::B1 => s.pack_bool(v),
    }
}

/// Get `v` as a raw `i64`. A register is allowed to *hold* an unboxed value —
/// that is the whole point of specialization — so this is a no-op whenever the
/// value is already unboxed, and only unpacks a tagged one.
pub fn as_int<S: Sink>(s: &mut S, v: S::V) -> S::V {
    if s.ty(v).rep == Rep::I64 {
        return v;
    }
    s.unpack_int(v)
}

/// Get `v` as a raw `f64`, promoting an integer the way Lua does.
pub fn as_float<S: Sink>(s: &mut S, v: S::V) -> S::V {
    match s.ty(v).rep {
        Rep::F64 => v,
        Rep::I64 => s.sitofp(v),
        _ => {
            if s.ty(v).set == TypeSet::INT {
                let i = s.unpack_int(v);
                s.sitofp(i)
            } else {
                s.unpack_float(v)
            }
        }
    }
}

/// Numeric operands, unpacked to a common representation.
enum Num<V> {
    Int(V, V),
    Float(V, V),
    /// Not provably numeric — the generic path, which may hit a metamethod.
    Neither,
}

fn numeric<S: Sink>(s: &mut S, a: S::V, b: S::V) -> Num<S::V> {
    let (ta, tb) = (s.ty(a).set, s.ty(b).set);
    if ta == TypeSet::INT && tb == TypeSet::INT {
        let x = as_int(s, a);
        let y = as_int(s, b);
        return Num::Int(x, y);
    }
    // Mixed int/float, or both float: Lua promotes to float.
    if ta.difference(TypeSet::NUM).is_empty()
        && tb.difference(TypeSet::NUM).is_empty()
        && !ta.is_empty()
        && !tb.is_empty()
    {
        let x = as_float(s, a);
        let y = as_float(s, b);
        return Num::Float(x, y);
    }
    Num::Neither
}

fn arith_ops(k: ArithKind) -> Option<(IntOp, FloatOp)> {
    Some(match k {
        ArithKind::Add => (IntOp::Add, FloatOp::Add),
        ArithKind::Sub => (IntOp::Sub, FloatOp::Sub),
        ArithKind::Mul => (IntOp::Mul, FloatOp::Mul),
        ArithKind::Mod => (IntOp::Mod, FloatOp::Mod),
        ArithKind::IDiv => (IntOp::IDiv, FloatOp::IDiv),
        // `/` and `^` are always float in Lua, even on two integers, so they
        // have no integer form to specialize into.
        ArithKind::Div | ArithKind::Pow => return None,
        _ => return None,
    })
}

/// The generic fallback. Operands must be tagged `Value`s, so pack anything we
/// had unboxed.
fn generic_arith<S: Sink>(s: &mut S, k: ArithKind, a: S::V, b: S::V) -> S::V {
    let a = to_val(s, a);
    let b = to_val(s, b);
    s.lua_arith(k, a, b)
}

/// Lower one arithmetic bytecode op.
///
/// Results stay **unboxed**. A register is free to hold a raw `i64`, and
/// packing only happens where a tagged `Value` is genuinely required — a
/// generic op, a call argument, a return, or an edge into a boxed block
/// parameter. Packing here instead would round-trip the loop counter through a
/// tagged pair every iteration, which is the exact cost we are trying to
/// delete.
fn arith<S: Sink>(s: &mut S, k: ArithKind, a: S::V, b: S::V) -> S::V {
    // Lua's `/` and `^` are float-valued even on two integers, so they have no
    // integer form to specialize into.
    if matches!(k, ArithKind::Div | ArithKind::Pow) {
        let (ta, tb) = (s.ty(a).set, s.ty(b).set);
        if ta.difference(TypeSet::NUM).is_empty()
            && tb.difference(TypeSet::NUM).is_empty()
            && !ta.is_empty()
            && !tb.is_empty()
        {
            let x = as_float(s, a);
            let y = as_float(s, b);
            let op = if k == ArithKind::Div {
                FloatOp::Div
            } else {
                FloatOp::Pow
            };
            return s.float_arith(op, x, y);
        }
        return generic_arith(s, k, a, b);
    }

    let Some((iop, fop)) = arith_ops(k) else {
        return generic_arith(s, k, a, b);
    };

    match numeric(s, a, b) {
        Num::Int(x, y) => {
            // Integer `//` and `%` raise on a zero divisor; the interpreter's
            // `op_arith` guards this before `wrapping_div`/`wrapping_rem` can
            // panic.
            if matches!(iop, IntOp::IDiv | IntOp::Mod) {
                s.guard_nonzero(y);
            }
            s.int_arith(iop, x, y)
        }
        Num::Float(x, y) => s.float_arith(fop, x, y),
        Num::Neither => generic_arith(s, k, a, b),
    }
}

/// Lower one bitwise bytecode op. Lua also accepts floats with exact integer
/// values here, but anything else raises — so only a provable integer
/// specializes, and the rest goes generic.
fn bitwise<S: Sink>(s: &mut S, k: ArithKind, a: S::V, b: S::V) -> S::V {
    let iop = match k {
        ArithKind::BAnd => IntOp::BAnd,
        ArithKind::BOr => IntOp::BOr,
        ArithKind::BXor => IntOp::BXor,
        ArithKind::Shl => IntOp::Shl,
        ArithKind::Shr => IntOp::Shr,
        _ => return generic_arith(s, k, a, b),
    };
    if s.ty(a).set == TypeSet::INT && s.ty(b).set == TypeSet::INT {
        let x = as_int(s, a);
        let y = as_int(s, b);
        return s.int_arith(iop, x, y);
    }
    generic_arith(s, k, a, b)
}

/// Guard `v`'s shape, unless its type already proves it.
///
/// This is what makes carrying `Refine::Shape` in the version key pay: once the
/// guard's refined value is written back into the register, the back-edge
/// carries `tab<S>`, the loop header is keyed on `tab<S>`, and the next
/// iteration's guard is skipped *here in the frontend* — no LICM required.
fn guard_shape_once<S: Sink>(s: &mut S, v: S::V, shape: ShapeRef) -> S::V {
    if s.ty(v).shape() == Some(shape) {
        return v;
    }
    s.guard_shape(v, shape)
}

/// Narrow `v` to the types a site has actually observed, guarding the
/// speculation. No-op when the observation is useless: an empty set (the site
/// never ran) or a set that rules nothing out.
fn speculate<S: Sink>(s: &mut S, v: S::V, observed: TypeSet) -> S::V {
    if observed.is_empty() || observed == TypeSet::ANY {
        return v;
    }
    s.guard_type(v, observed)
}

/// Truthiness as a `B1`. Folds when the type set already decides it — which is
/// why `TypeSet` splits `FALSE`/`TRUE` instead of carrying one `BOOL`.
pub fn falsy<S: Sink>(s: &mut S, v: S::V) -> S::V {
    let set = s.ty(v).set;
    if set.is_truthy() {
        return s.bconst(false);
    }
    if set.is_falsy() {
        return s.bconst(true);
    }
    let v = to_val(s, v);
    s.is_falsy(v)
}

/// Lower a comparison to a `B1`. Callers branch on the result.
pub fn compare<S: Sink>(s: &mut S, cc: Cc, a: S::V, b: S::V) -> S::V {
    match numeric(s, a, b) {
        Num::Int(x, y) => s.icmp(cc, x, y),
        Num::Float(x, y) => s.fcmp(cc, x, y),
        // Strings compare too, and tables may have `__lt`/`__eq`.
        Num::Neither => {
            let a = to_val(s, a);
            let b = to_val(s, b);
            if cc == Cc::Eq {
                s.lua_eq(a, b)
            } else {
                s.lua_cmp(cc, a, b)
            }
        }
    }
}

/// The transfer function for one non-terminator instruction.
///
/// Terminators are handled by the driver, which needs to compute a distinct
/// successor state per edge (`TESTSET` assigns on only one of its two edges,
/// and `FORLOOP` advances the counter on only one).
pub fn step<S: Sink, F: Feedback>(
    s: &mut S,
    st: &mut RegState<S::V>,
    fb: &mut F,
    pc: u32,
    i: Instruction,
) -> Result<(), Decline> {
    // Every arithmetic and bitwise opcode is `Abc`-shaped: destination, then
    // the two operands.
    macro_rules! binop {
        ($f:ident, $k:expr) => {{
            let (dst, lhs, rhs) = i.abc();
            let a = st.get(s, lhs);
            let b = st.get(s, rhs);
            let r = $f(s, $k, a, b);
            st.set(s, dst, r);
        }};
    }

    match i.op() {
        Op::MOVE => {
            let (dst, src) = i.ab();
            let v = st.get(s, src);
            st.set(s, dst, v);
        }
        // Numbers load unboxed; everything else (nil, booleans, strings) has no
        // unboxed form a register may hold, and stays a tagged `Value`.
        Op::LOAD => {
            let (dst, idx) = i.ad();
            let v = match fb.scalar(idx) {
                Some(Scalar::Int(n)) => s.iconst(n),
                Some(Scalar::Float(x)) => s.fconst(x),
                None => {
                    let (c, ty) = fb.constant(idx);
                    s.kconst(c, ty)
                }
            };
            st.set(s, dst, v);
        }

        Op::ADD => binop!(arith, ArithKind::Add),
        Op::SUB => binop!(arith, ArithKind::Sub),
        Op::MUL => binop!(arith, ArithKind::Mul),
        Op::MOD => binop!(arith, ArithKind::Mod),
        Op::POW => binop!(arith, ArithKind::Pow),
        Op::DIV => binop!(arith, ArithKind::Div),
        Op::IDIV => binop!(arith, ArithKind::IDiv),
        Op::BAND => binop!(bitwise, ArithKind::BAnd),
        Op::BOR => binop!(bitwise, ArithKind::BOr),
        Op::BXOR => binop!(bitwise, ArithKind::BXor),
        Op::SHL => binop!(bitwise, ArithKind::Shl),
        Op::SHR => binop!(bitwise, ArithKind::Shr),

        Op::UNM => {
            let (dst, src) = i.ab();
            let a = st.get(s, src);
            let set = s.ty(a).set;
            let r = if set == TypeSet::INT {
                let x = as_int(s, a);
                s.int_arith(IntOp::Neg, x, x)
            } else if set == TypeSet::FLOAT {
                let x = as_float(s, a);
                s.float_arith(FloatOp::Neg, x, x)
            } else {
                let a = to_val(s, a);
                s.lua_unary(ArithKind::Unm, a)
            };
            st.set(s, dst, r);
        }
        Op::BNOT => {
            let (dst, src) = i.ab();
            let a = st.get(s, src);
            let r = if s.ty(a).set == TypeSet::INT {
                let x = as_int(s, a);
                s.int_arith(IntOp::BNot, x, x)
            } else {
                let a = to_val(s, a);
                s.lua_unary(ArithKind::BNot, a)
            };
            st.set(s, dst, r);
        }
        // `not` never consults a metamethod — it is pure truthiness, and folds
        // outright when the type set already decides it. The result is a Lua
        // boolean, so it packs: a `B1` is a condition, and a register must hold
        // something a deopt can write back as a `Value`.
        Op::NOT => {
            let (dst, src) = i.ab();
            let a = st.get(s, src);
            let f = falsy(s, a);
            let f = s.pack_bool(f);
            st.set(s, dst, f);
        }
        Op::LEN => {
            let (dst, src) = i.ab();
            let a = st.get(s, src);
            let a = to_val(s, a);
            let r = s.lua_len(a);
            st.set(s, dst, r);
        }
        Op::CONCAT => {
            let (dst, lhs, rhs) = i.abc();
            let a = st.get(s, lhs);
            let b = st.get(s, rhs);
            let a = to_val(s, a);
            let b = to_val(s, b);
            let r = s.lua_concat(a, b);
            st.set(s, dst, r);
        }

        Op::NEWTABLE => {
            let dst = i.a();
            let t = s.tab_new();
            st.set(s, dst, t);
        }

        // The IC is the whole story here: a monomorphic site gives us the shape,
        // the shape gives us a constant slot offset, and the site's observed
        // value kinds let us speculate on what the load produces. `assume.no_mm`
        // is a watchpoint, not a check — it emits nothing.
        Op::GETFIELD => {
            let (dst, table, ic_idx, key_idx) = i.abde();
            let t = st.get(s, table);
            let t = to_val(s, t);
            let r = match fb.ic(ic_idx, MetamethodBits::INDEX) {
                Some(ic) => {
                    let t = guard_shape_once(s, t, ic.shape);
                    st.refine(table, t);
                    s.assume_no_mm(ic.shape, MetamethodBits::INDEX);
                    let p = s.tab_props(t);
                    let v = s.slot_get(p, ic.slot);
                    // A guarded load, in the LuaJIT sense: the type comes from
                    // what the site has actually observed, and the guard is what
                    // makes it sound. Without this the value is `any`, and every
                    // arithmetic op consuming it stays a metamethod-capable call.
                    speculate(s, v, ic.types)
                }
                None => {
                    let (c, ty) = fb.constant(key_idx);
                    let k = s.kconst(c, ty);
                    s.lua_get_index(t, k)
                }
            };
            st.set(s, dst, r);
        }
        Op::SETFIELD => {
            let (src, table, ic_idx, key_idx) = i.abde();
            let t = st.get(s, table);
            let t = to_val(s, t);
            let v = st.get(s, src);
            let v = to_val(s, v);
            match fb.ic(ic_idx, MetamethodBits::NEWINDEX) {
                // A slot store is legal only for a key already present in the
                // guarded shape. A *new* key transitions the shape, so an
                // absent-slot IC must take the generic path.
                Some(ic) => {
                    let t = guard_shape_once(s, t, ic.shape);
                    st.refine(table, t);
                    s.assume_no_mm(ic.shape, MetamethodBits::NEWINDEX);
                    let p = s.tab_props(t);
                    s.slot_set(p, ic.slot, v);
                }
                None => {
                    let (c, ty) = fb.constant(key_idx);
                    let k = s.kconst(c, ty);
                    s.lua_set_index(t, k, v);
                }
            }
        }
        Op::GETTABLE => {
            let (dst, table, key) = i.abc();
            let t = st.get(s, table);
            let k = st.get(s, key);
            let t = to_val(s, t);
            let k = to_val(s, k);
            let r = s.lua_get_index(t, k);
            st.set(s, dst, r);
        }
        Op::SETTABLE => {
            let (src, table, key) = i.abc();
            let t = st.get(s, table);
            let k = st.get(s, key);
            let v = st.get(s, src);
            let t = to_val(s, t);
            let k = to_val(s, k);
            let v = to_val(s, v);
            s.lua_set_index(t, k, v);
        }
        Op::SELF => {
            let (dst, object, key_idx) = i.abd();
            let obj = st.get(s, object);
            let obj = to_val(s, obj);
            let (c, ty) = fb.constant(key_idx);
            let k = s.kconst(c, ty);
            let m = s.lua_get_index(obj, k);
            st.set(s, dst, m);
            st.set(s, dst + 1, obj);
        }
        Op::SETLIST => {
            let (table, count, offset) = i.abd();
            let t = st.get(s, table);
            let t = to_val(s, t);
            for n in 0..count {
                let v = st.get(s, table + 1 + n);
                let v = to_val(s, v);
                let idx = s.iconst(offset as i64 + n as i64 + 1);
                let k = s.pack_int(idx);
                s.lua_set_index(t, k, v);
            }
        }

        Op::GETUPVAL => {
            let (dst, idx) = i.ab();
            let v = s.upval_get(idx);
            st.set(s, dst, v);
        }
        Op::SETUPVAL => {
            let (src, idx) = i.ab();
            let v = st.get(s, src);
            let v = to_val(s, v);
            s.upval_set(idx, v);
        }
        Op::GETTABUP => {
            let (dst, idx, _, key) = i.abde();
            let k = fb.key(key);
            let v = s.get_global(idx, k);
            st.set(s, dst, v);
        }
        Op::SETTABUP => {
            let (src, idx, _, key) = i.abde();
            let v = st.get(s, src);
            let v = to_val(s, v);
            let k = fb.key(key);
            s.set_global(idx, k, v);
        }
        Op::CLOSE => {
            let start = i.a();
            s.upval_close(start)
        }

        Op::CLOSURE => {
            let (dst, proto) = i.ad();
            let p = fb.proto(proto);
            let f = s.closure_new(p);
            st.set(s, dst, f);
        }

        // `args` counts the callee plus its arguments; `returns` counts the
        // results plus one. Both zero sentinels were rejected by the CFG pass.
        Op::CALL => {
            let (func, args, returns) = i.abc();
            let mut argv = Vec::with_capacity(args as usize);
            for n in 0..args {
                let v = st.get(s, func + n);
                argv.push(to_val(s, v));
            }
            let nret = returns - 1;
            let results = s.call(&argv, nret);
            for (n, r) in results.iter().enumerate() {
                st.set(s, func + n as u8, *r);
            }
        }

        Op::LFALSESKIP => {
            let src = i.a();
            let f = s.bconst(false);
            let f = s.pack_bool(f);
            st.set(s, src, f);
        }

        Op::NOP => {}
        Op::ERRNNIL => return Err(Decline::ErrNNil(pc)),

        // Terminators are the driver's business; the compare ops are consumed
        // there together with their paired JMP.
        Op::JMP
        | Op::EQ
        | Op::LT
        | Op::LE
        | Op::TEST
        | Op::TESTSET
        | Op::RETURN
        | Op::FORPREP
        | Op::FORLOOP
        | Op::TFORPREP
        | Op::TFORLOOP
        | Op::STOP => {
            debug_assert!(false, "terminator {i:?} reached step()");
            return Err(Decline::Op(pc));
        }

        Op::TAILCALL | Op::TFORCALL | Op::TBC | Op::VARARG | Op::VARARGGET | Op::VARARGPREP => {
            return Err(Decline::Op(pc));
        }
    }
    Ok(())
}
