//! IR operations, their result types and effects.

use crate::jit::ir::types::{Rep, Ty, TypeSet};

/// An integer or float comparison. Float compares are ordered: false when
/// either operand is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Cc {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cc {
    /// The comparison with its operands swapped.
    pub(crate) fn swap(self) -> Cc {
        match self {
            Cc::Eq => Cc::Eq,
            Cc::Ne => Cc::Ne,
            Cc::Lt => Cc::Gt,
            Cc::Le => Cc::Ge,
            Cc::Gt => Cc::Lt,
            Cc::Ge => Cc::Le,
        }
    }

    pub(crate) fn eval_i(self, a: i64, b: i64) -> bool {
        match self {
            Cc::Eq => a == b,
            Cc::Ne => a != b,
            Cc::Lt => a < b,
            Cc::Le => a <= b,
            Cc::Gt => a > b,
            Cc::Ge => a >= b,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Cc::Eq => "eq",
            Cc::Ne => "ne",
            Cc::Lt => "lt",
            Cc::Le => "le",
            Cc::Gt => "gt",
            Cc::Ge => "ge",
        }
    }
}

/// Rust routines a region calls with the C ABI (`jit::helpers`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum HelperId {
    /// `(f64, f64) -> f64`: Lua's float `%`.
    FMod,
    /// `(f64, f64) -> f64`: `pow`.
    FPow,
}

impl HelperId {
    pub(crate) fn name(self) -> &'static str {
        match self {
            HelperId::FMod => "fmod",
            HelperId::FPow => "pow",
        }
    }
}

/// Why an exit leaves, for counting toward recompilation (`jit-design.md`
/// 5.6): only widenable kinds count.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ExitTag {
    /// A type guard; a recompile with the feedback it records widens it.
    Type,
    /// An i32 result left i32.
    Overflow,
    /// The first execution of a block compiled as never executed.
    NeverRan,
    /// An operation compiled as a deopt: no recompile changes it.
    Unsupported,
    /// An operand the interpreter raises on, or its slow path.
    Slow,
    /// The collector's check after an allocation.
    Gc,
    /// A prologue guard on an entry value (5.2): fails to the entry-fail
    /// stub, which records the entry's kinds.
    Entry,
}

impl ExitTag {
    pub(crate) fn widenable(self) -> bool {
        matches!(self, ExitTag::Type | ExitTag::Overflow | ExitTag::NeverRan)
    }
}

// Some operations are lowered already but produced only from milestone 2.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Op {
    // --- constants: pure, no operands --------------------------------------
    /// A `Value` word with no heap object (nil, booleans, small ints, floats).
    KVal(u64),
    /// A heap `Value`, its pool index.
    KObj(u32),
    KI32(i32),
    KI64(i64),
    KB1(bool),
    /// An f64 by its bits.
    KF64(u64),

    // --- home slots -----------------------------------------------------------
    /// `R[r]` as a `Val`.
    Load(u8),
    /// `R[r] = v`, `v` a `Val`.
    Store(u8),

    // --- boxing and tests -----------------------------------------------------
    /// The `Val` of an unboxed value.
    Box,
    /// The unboxed `rep` of a `Val` whose type is within `rep`'s set.
    Unbox(Rep),
    /// Whether a `Val` is in the set, as a `B1`.
    IsType(TypeSet),
    /// Whether a `Val` is nil or false.
    IsFalsy,
    /// `c ? a : b`.
    Select,
    /// Whether two `Val`s are the same word.
    SameBits,

    // --- guards (each has a snapshot) ----------------------------------------
    /// The `Val` refined to the set; exits when it is not in it.
    Guard(TypeSet),
    /// Exits when the `B1` is false.
    GuardTrue,
    /// Exits when the `B1` is true.
    GuardFalse,
    /// Exits when the two `Val`s are not the same word.
    GuardSame,
    /// The collector check after an allocation.
    GcCheck,
    /// Exits when the frame has upvalues or to-be-closed variables to close.
    GuardNoClose,

    // --- i32 ------------------------------------------------------------------
    /// Exits on overflow.
    IAdd,
    ISub,
    IMul,
    /// Proven not to overflow.
    IAddNo,
    ISubNo,
    IMulNo,
    /// Exits on `i32::MIN`.
    INeg,
    IAnd,
    IOr,
    IXor,
    INot,
    /// Lua shift semantics on the 64-bit value; exits when the result leaves i32.
    IShl,
    IShr,
    /// Floor division and modulo; exit on a zero divisor and `MIN // -1`.
    IDivFloor,
    IModFloor,
    ICmp(Cc),
    IToF,
    IToL,

    // --- i64 (wrapping) -------------------------------------------------------
    LAdd,
    LSub,
    LMul,
    LNeg,
    LAnd,
    LOr,
    LXor,
    LNot,
    LShl,
    LShr,
    /// Unsigned division by a divisor known nonzero.
    LUDiv,
    /// Exit on a zero divisor.
    LDivFloor,
    LModFloor,
    LCmp(Cc),
    LToF,
    /// Exits when outside i32.
    LToI,

    // --- f64 ------------------------------------------------------------------
    FAdd,
    FSub,
    FMul,
    FDiv,
    FNeg,
    FAbs,
    FSqrt,
    FFloor,
    FCeil,
    /// Lua float `//`: `floor(a / b)`.
    FIDiv,
    FCmp(Cc),
    /// The i32 an f64 holds exactly; exits otherwise.
    FToIExact,
    /// The f64 of a `Val` small integer or float; exits otherwise.
    ToF64,
    /// The i64 of an f64 rounded toward zero, saturating; NaN gives 0.
    FToL,

    // --- closure --------------------------------------------------------------
    /// A by-value upvalue of the running closure.
    UpvalValue(u8),

    // --- calls ----------------------------------------------------------------
    /// A C-ABI helper.
    Helper(HelperId),
    /// The first instruction of a call's resume block, `c` the CALL's result
    /// operand: one or two results for `c` 2 or 3; otherwise they were landed
    /// in slots.
    Resume {
        c: u8,
    },

    // --- terminators ----------------------------------------------------------
    Jump,
    /// On a `B1`: edge 0 when true, edge 1 when false.
    Br,
    /// Return `n` results from `R[a..]` (already stored).
    Return {
        a: u8,
        n: u8,
    },
    /// Call `R[a]` with `nargs` arguments in `R[a+4..]` (stored), `c` the
    /// CALL's result operand; edge 0 is the resume block.
    Call {
        a: u8,
        nargs: u8,
        c: u8,
        pc: u32,
    },
    /// Leave for the interpreter (with a snapshot).
    Deopt,
}

/// Memory and control effects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Effects(u32);

impl Effects {
    pub(crate) const NONE: Effects = Effects(0);
    pub(crate) const MAY_DEOPT: Effects = Effects(1);
    pub(crate) const MAY_ALLOC: Effects = Effects(2);
    pub(crate) const TAILOUT: Effects = Effects(4);
    pub(crate) const TERMINATOR: Effects = Effects(8);
    pub(crate) const R_SLOT: Effects = Effects(16);
    pub(crate) const W_SLOT: Effects = Effects(32);
    pub(crate) const R_GC: Effects = Effects(128);
    /// Writes every class (a call: anything may run).
    pub(crate) const W_ALL: Effects = Effects(0xffff_0000);

    pub(crate) fn has(self, e: Effects) -> bool {
        self.0 & e.0 == e.0
    }

    /// Whether anything is written: a slot or a memory class.
    pub(crate) fn writes(self) -> bool {
        self.0 & (Effects::W_SLOT.0 | Effects::W_ALL.0) != 0
    }
}

impl std::ops::BitOr for Effects {
    type Output = Effects;
    fn bitor(self, o: Effects) -> Effects {
        Effects(self.0 | o.0)
    }
}

impl Op {
    pub(crate) fn effects(self) -> Effects {
        use Op::*;
        match self {
            Load(_) => Effects::R_SLOT,
            Store(_) => Effects::W_SLOT,
            Guard(_) | GuardTrue | GuardFalse | GuardSame | GuardNoClose | IAdd | ISub | IMul
            | INeg | IShl | IShr | IDivFloor | IModFloor | LDivFloor | LModFloor | LToI
            | FToIExact | ToF64 => Effects::MAY_DEOPT,
            GcCheck => Effects::MAY_DEOPT | Effects::R_GC,
            Helper(_) => Effects::NONE,
            Resume { .. } => Effects::R_SLOT,
            Jump | Br => Effects::TERMINATOR,
            Return { .. } => Effects::TERMINATOR | Effects::TAILOUT | Effects::R_SLOT,
            Call { .. } => {
                Effects::TERMINATOR
                    | Effects::TAILOUT
                    | Effects::W_ALL
                    | Effects::R_SLOT
                    | Effects::W_SLOT
            }
            Deopt => Effects::TERMINATOR | Effects::MAY_DEOPT,
            _ => Effects::NONE,
        }
    }

    pub(crate) fn is_terminator(self) -> bool {
        self.effects().has(Effects::TERMINATOR)
    }

    pub(crate) fn may_deopt(self) -> bool {
        self.effects().has(Effects::MAY_DEOPT)
    }

    /// Pure: no effect, no exit, so it may be removed when unused, numbered
    /// and moved.
    pub(crate) fn is_pure(self) -> bool {
        self.effects() == Effects::NONE && !matches!(self, Op::Helper(_))
    }

    pub(crate) fn is_const(self) -> bool {
        matches!(
            self,
            Op::KVal(_) | Op::KObj(_) | Op::KI32(_) | Op::KI64(_) | Op::KF64(_) | Op::KB1(_)
        )
    }

    /// How many values the instruction defines.
    pub(crate) fn num_results(self) -> usize {
        use Op::*;
        match self {
            Store(_)
            | GuardTrue
            | GuardFalse
            | GuardSame
            | GuardNoClose
            | GcCheck
            | Jump
            | Br
            | Return { .. }
            | Call { .. }
            | Deopt => 0,
            Resume { c } => match c {
                2 => 1,
                3 => 2,
                _ => 0,
            },
            _ => 1,
        }
    }

    /// The result type of a pure operation from its operand types.
    pub(crate) fn result_ty(self, args: &[Ty]) -> Ty {
        use Op::*;
        match self {
            KVal(bits) => Ty::val(crate::jit::ir::const_set(bits)),
            KObj(_) => Ty::ANY,
            KI32(_) => Ty::I32,
            KB1(_) => Ty::B1,
            KI64(_) => Ty::I64,
            KF64(_) => Ty::F64,
            Load(_) | UpvalValue(_) | Resume { .. } => Ty::ANY,
            Box => args[0].boxed(),
            Unbox(rep) => Ty {
                rep,
                set: if rep == Rep::Ptr {
                    args[0].set
                } else {
                    args[0].set & Ty::of_rep(rep).set
                },
                refine: args[0].refine,
            },
            IsType(_) | IsFalsy | SameBits | ICmp(_) | LCmp(_) | FCmp(_) => Ty::B1,
            Select => args[1].join(args[2]),
            Guard(set) => Ty {
                set: args[0].set & set,
                ..args[0]
            },
            IAdd | ISub | IMul | IAddNo | ISubNo | IMulNo | INeg | IAnd | IOr | IXor | INot
            | IShl | IShr | IDivFloor | IModFloor | LToI | FToIExact => Ty::I32,
            IToF | LToF | FAdd | FSub | FMul | FDiv | FNeg | FAbs | FSqrt | FFloor | FCeil
            | FIDiv | ToF64 => Ty::F64,
            IToL | FToL | LAdd | LSub | LMul | LUDiv | LNeg | LAnd | LOr | LXor | LNot | LShl
            | LShr | LDivFloor | LModFloor => Ty::I64,
            Helper(HelperId::FMod | HelperId::FPow) => Ty::F64,
            _ => Ty::ANY,
        }
    }

    pub(crate) fn name(self) -> String {
        use Op::*;
        match self {
            KVal(b) => format!("kval {:#x}", b),
            KObj(p) => format!("kobj k{p}"),
            KI32(v) => format!("ki32 {v}"),
            KI64(v) => format!("ki64 {v}"),
            KB1(v) => format!("kb1 {v}"),
            KF64(b) => format!("kf64 {:?}", f64::from_bits(b)),
            Load(r) => format!("load r{r}"),
            Store(r) => format!("store r{r}"),
            Box => "box".into(),
            Unbox(rep) => format!("unbox.{}", rep.name()),
            IsType(s) => format!("istype {}", Ty::val(s)),
            IsFalsy => "isfalsy".into(),
            Select => "select".into(),
            Guard(s) => format!("guard {}", Ty::val(s)),
            GuardTrue => "guard.true".into(),
            GuardFalse => "guard.false".into(),
            GuardSame => "guard.same".into(),
            GcCheck => "gccheck".into(),
            GuardNoClose => "guard.noclose".into(),
            SameBits => "samebits".into(),
            ICmp(c) => format!("icmp.{}", c.name()),
            LCmp(c) => format!("lcmp.{}", c.name()),
            FCmp(c) => format!("fcmp.{}", c.name()),
            UpvalValue(i) => format!("upval u{i}"),
            Helper(h) => format!("call.{}", h.name()),
            Resume { c } => format!("resume c={c}"),
            Return { a, n } => format!("return r{a} n={n}"),
            Call { a, nargs, c, pc } => format!("call r{a} nargs={nargs} c={c} pc{pc}"),
            other => format!("{other:?}").to_lowercase(),
        }
    }
}
