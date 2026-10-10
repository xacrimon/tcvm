//! IR operations, their result types and effects.

use crate::instruction::ArithKind;
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
    /// Unsigned, on integers only.
    Lo,
    Hi,
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
            Cc::Lo => Cc::Hi,
            Cc::Hi => Cc::Lo,
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
            // Sign-extended i32s order as u64s as they do as u32s.
            Cc::Lo => (a as u64) < (b as u64),
            Cc::Hi => (a as u64) > (b as u64),
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
            Cc::Lo => "lo",
            Cc::Hi => "hi",
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
    /// `f64 -> f64`.
    Sin,
    Cos,
    /// `(Val, Val) -> i32`: a generic `<`, `<=` or `==`: 1, 0, or
    /// `HELPER_FAIL` when only the interpreter can (a metamethod, an error).
    Lt,
    Le,
    Eq,
    /// `(Val, Val) -> Val`: the arithmetic of two numbers, or
    /// `HELPER_FAIL_VAL` (not numbers, a zero integer divisor).
    Arith(ArithKind),
    /// `Val -> Val`: `-v` and `~v` of a number, or `HELPER_FAIL_VAL`.
    Unm,
    BNot,
    /// `Val -> Val`: `#v` of a string or of a table without `__len`, or
    /// `HELPER_FAIL_VAL`.
    Len,
    /// `(Val, Val) -> Val`: `a .. b` of strings and numbers, or
    /// `HELPER_FAIL_VAL`.
    Concat,
    /// `->`: the barrier of a store of a heap value into upvalue `i`'s cell.
    UpvalBarrier(u8),
    /// `-> Val`: `CLOSURE` of prototype `proto` into `R[dst]`, from the
    /// captured registers' home slots.
    Closure {
        proto: u16,
        dst: u8,
    },
    /// `->`: close this frame's open upvalues from `R[a]` up, from the
    /// captured registers' home slots.
    Close(u8),
    /// `-> Val`: `NEWTABLE` from the prototype's template.
    NewTable(u16),
    /// `->`: `SETLIST` of `R[a+1..=a+n]`, in their slots, into the table in
    /// `R[a]` from index `off + 1`.
    SetList {
        a: u8,
        n: u8,
        off: u32,
    },
    /// `Val -> Val`: `VARARGGET` of a key.
    VarargGet,
    /// `-> Val`: `VARARGPREP`'s vararg table, also stored in `R[np]`.
    VarargTable(u8),
    /// `-> Val`: a `TFORCALL_NEXT` step from the table in `R[a+1]` at the
    /// position in `R[a+3]`, their slots, into the slots of the position
    /// and `count` loop variables: 1, 0 at the end, or `HELPER_FAIL_VAL`.
    TforNext {
        a: u8,
        count: u8,
    },
    /// `(Val, Val) -> Val`: an `ipairs` step past the array part, `t[k]`:
    /// the value, nil at the end, or `HELPER_FAIL_VAL` for `__index`.
    TforIpairs,
    /// `-> I32`: the arguments of a CALL at `R[a]` up to `top`.
    TopArgs(u8),
    /// `-> Val`: `VARARG` of all the extras into `R[a..]`, `top` after
    /// them: 0, or `HELPER_FAIL_VAL` when they do not fit the stack.
    VarargAll(u8),
    /// `(Val, Val) -> Val`: the metamethod of the bit (`MetamethodBits`)
    /// from the left operand, else the right, as `binop_metamethod` finds
    /// it; nil when neither has one.
    Mm(u32),
    /// `(Val, Val) -> Val`: the `__index` function `recv[key]` calls, when
    /// `recv`'s own metatable holds it; else nil.
    IndexFn,
    /// `(Val, Val) -> Val`: the same for `__newindex`.
    NewIndexFn,
    /// `(Val, Val) -> Val`: `recv[key]` through `__index` tables, or
    /// `HELPER_FAIL_VAL` (an `__index` function, an error).
    Index,
    /// `(Val, Val, Val) ->`: `recv[key] = v` through `__newindex` tables,
    /// `raw_set_keyed` when `keyed` (`t[k] = v`), or `HELPER_FAIL_VAL`.
    NewIndex {
        keyed: bool,
    },
}

/// What a compare helper returns when the interpreter must run the compare.
pub(crate) const HELPER_FAIL: i32 = 2;

/// What a `Val` helper returns when the interpreter must run the
/// instruction: the negative quiet NaN, which no `Value` is (floats are
/// canonical) and one `movz` makes.
pub(crate) const HELPER_FAIL_VAL: u64 = 0xFFF8_0000_0000_0000;

/// A pinned register a helper takes ahead of its operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Ctx {
    Rt,
    Thread,
    Base,
}

/// A helper's C signature: the pinned registers, then an immediate, then
/// the operands, each in the next argument register of its class.
pub(crate) struct Sig {
    pub(crate) ctx: &'static [Ctx],
    pub(crate) imm: Option<u64>,
    pub(crate) args: &'static [Rep],
    pub(crate) ret: Option<Rep>,
    pub(crate) effects: Effects,
}

impl HelperId {
    pub(crate) fn name(self) -> String {
        match self {
            HelperId::FMod => "fmod".into(),
            HelperId::FPow => "pow".into(),
            HelperId::Sin => "sin".into(),
            HelperId::Cos => "cos".into(),
            HelperId::Lt => "lt".into(),
            HelperId::Le => "le".into(),
            HelperId::Eq => "eq".into(),
            HelperId::Arith(k) => format!("arith.{k:?}").to_lowercase(),
            HelperId::Unm => "unm".into(),
            HelperId::BNot => "bnot".into(),
            HelperId::Len => "len".into(),
            HelperId::Concat => "concat".into(),
            HelperId::UpvalBarrier(i) => format!("upval_barrier u{i}"),
            HelperId::Closure { proto, dst } => format!("closure p{proto} r{dst}"),
            HelperId::Close(a) => format!("close r{a}"),
            HelperId::NewTable(t) => format!("newtable t{t}"),
            HelperId::SetList { a, n, off } => format!("setlist r{a} n={n} off={off}"),
            HelperId::VarargGet => "varargget".into(),
            HelperId::VarargTable(np) => format!("varargtable r{np}"),
            HelperId::TforNext { a, count } => format!("tfornext r{a} count={count}"),
            HelperId::TforIpairs => "tforipairs".into(),
            HelperId::TopArgs(a) => format!("topargs r{a}"),
            HelperId::VarargAll(a) => format!("varargall r{a}"),
            HelperId::Index => "index".into(),
            HelperId::Mm(bit) => format!("mm {bit:#x}"),
            HelperId::IndexFn => "index_fn".into(),
            HelperId::NewIndexFn => "newindex_fn".into(),
            HelperId::NewIndex { keyed } => format!("newindex keyed={keyed}"),
        }
    }

    pub(crate) fn sig(self) -> Sig {
        use Rep::*;
        const RT: &[Ctx] = &[Ctx::Rt];
        const FRAME: &[Ctx] = &[Ctx::Rt, Ctx::Thread, Ctx::Base];
        const BASE: &[Ctx] = &[Ctx::Rt, Ctx::Base];
        const STACK: &[Ctx] = &[Ctx::Thread, Ctx::Base];
        let alloc = Effects::MAY_ALLOC;
        let (ctx, imm, args, ret, effects): (_, _, &'static [Rep], _, _) = match self {
            HelperId::FMod | HelperId::FPow => {
                (&[][..], None, &[F64, F64], Some(F64), Effects::NONE)
            }
            HelperId::Sin | HelperId::Cos => (&[][..], None, &[F64], Some(F64), Effects::NONE),
            HelperId::Lt | HelperId::Le | HelperId::Eq => {
                (RT, None, &[Val, Val], Some(I32), Effects::NONE)
            }
            HelperId::Arith(k) => (RT, Some(k as u64), &[Val, Val], Some(Val), alloc),
            HelperId::Unm | HelperId::BNot | HelperId::Len => (RT, None, &[Val], Some(Val), alloc),
            HelperId::Concat => (RT, None, &[Val, Val], Some(Val), alloc),
            HelperId::UpvalBarrier(i) => (FRAME, Some(i as u64), &[], None, Effects::NONE),
            HelperId::Closure { proto, dst } => (
                FRAME,
                Some(proto as u64 | (dst as u64) << 32),
                &[],
                Some(Val),
                alloc | Effects::R_SLOT | Effects::W_FRAME,
            ),
            HelperId::Close(a) => (
                FRAME,
                Some(a as u64),
                &[],
                None,
                Effects::R_SLOT | Effects::W_UPVAL,
            ),
            HelperId::NewTable(t) => (BASE, Some(t as u64), &[], Some(Val), alloc),
            HelperId::SetList { a, n, off } => (
                FRAME,
                Some(a as u64 | (n as u64) << 8 | (off as u64) << 16),
                &[],
                None,
                alloc | Effects::R_SLOT | Effects::W_TAB | Effects::W_SHAPE,
            ),
            HelperId::VarargGet => (BASE, None, &[Val], Some(Val), alloc),
            HelperId::VarargTable(np) => (
                BASE,
                Some(np as u64),
                &[],
                Some(Val),
                alloc | Effects::W_SLOT,
            ),
            // A lookup moves a table off a stale class (`Table::meta`).
            HelperId::Mm(bit) => (
                RT,
                Some(bit as u64),
                &[Val, Val],
                Some(Val),
                alloc | Effects::W_SHAPE | Effects::W_META,
            ),
            HelperId::IndexFn | HelperId::NewIndexFn => (
                RT,
                None,
                &[Val, Val],
                Some(Val),
                alloc | Effects::W_SHAPE | Effects::W_META,
            ),
            HelperId::TforNext { a, count } => (
                BASE,
                Some(a as u64 | (count as u64) << 8),
                &[],
                Some(Val),
                alloc | Effects::R_SLOT | Effects::W_SLOT,
            ),
            HelperId::TforIpairs => (RT, None, &[Val, Val], Some(Val), alloc),
            HelperId::TopArgs(a) => (STACK, Some(a as u64), &[], Some(I32), Effects::NONE),
            HelperId::VarargAll(a) => (STACK, Some(a as u64), &[], Some(Val), Effects::W_SLOT),
            HelperId::Index => (
                RT,
                None,
                &[Val, Val],
                Some(Val),
                alloc | Effects::W_SHAPE | Effects::W_META,
            ),
            HelperId::NewIndex { keyed } => (
                RT,
                Some(keyed as u64),
                &[Val, Val, Val],
                Some(Val),
                alloc | Effects::W_TAB | Effects::W_SHAPE | Effects::W_META,
            ),
        };
        Sig {
            ctx,
            imm,
            args,
            ret,
            effects,
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
    /// A collector check (5.7): leaves through `Exit::Gc`, then resumes.
    Gc,
    /// A table's shape is not the one compiled for.
    Shape,
    /// A helper failed where the site's feedback never saw a metamethod
    /// run: the exit records one, and the recompile calls it (5.4).
    Mm,
    /// A prologue guard on an entry value (5.2): fails to the entry-fail
    /// stub, which records the entry's kinds.
    Entry,
}

impl ExitTag {
    pub(crate) fn widenable(self) -> bool {
        matches!(
            self,
            ExitTag::Type | ExitTag::Overflow | ExitTag::NeverRan | ExitTag::Shape | ExitTag::Mm
        )
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

    /// The `Val` refined to the set without a test: a guarded value
    /// rebuilt after a call (9.8), which a guard already tested.
    Refine(TypeSet),

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
    /// Exits when the frame's header has any of the flags.
    GuardNoFlag(u8),

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
    /// The cell of a shared upvalue of the running closure.
    UpvalCell(u8),
    /// The value of an upvalue cell.
    CellLoad,
    /// `cell = v`, without its barrier.
    CellStore,
    /// Whether the frame's header has the flag (`vm::frame::flag`).
    HasFlag(u8),

    // --- tables: `Ptr` operands are a table's lock (its value's payload) ----
    /// Exits unless the table's shape is `shapes[i]`.
    GuardShape(u32),
    /// Whether the table's shape is `shapes[i]`, a `B1`.
    ShapeIs(u32),
    /// The live metamethod bits of the class `shapes[i]` pins, an `I32`.
    ClassBits(u32),
    /// The live `__index` of the class `shapes[i]` pins.
    ClassIndex(u32),
    /// The named slot at `loc` (a `SlotLoc`); `key` is the pool index of its
    /// constant key: accesses with other keys never alias it.
    TabLoad {
        loc: u32,
        key: u32,
    },
    TabStore {
        loc: u32,
        key: u32,
    },
    /// Whether the spill cell has room for `loc`, the spilled slot a
    /// transition adds.
    TabHasRoom(u32),
    /// Move the table to `shapes[i]`.
    TabSetShape(u32),
    /// The array part's size (`I32`) and cell.
    TabASize,
    TabArray,
    /// `arr[i]` of an array part, `i` an `I32` below its size.
    ArrLoad,
    ArrStore,
    /// The table's backward barrier unless it is gray.
    Barrier,

    // --- varargs: the frame's extras below its header ----------------------
    /// How many extras the frame has (`nv`), an `I32`.
    NumVarargs,
    /// Extra `k` (from 0) given `nv`, or nil past the last.
    VarargAt(u8),
    /// Extra `k - 1` given `nv` and the `I32` key `k`, or nil out of range.
    VarargIdx,

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
    /// Call the first operand with the rest as arguments from a header past
    /// the frame's registers (`stage_mm!`): a metamethod or a generic `for`'s
    /// iterator. Edge 0 is the resume block, which takes `c - 1` results,
    /// landed from `R[land]` when more than two.
    CallMm {
        pc: u32,
        nargs: u8,
        land: u8,
        c: u8,
    },
    /// Exits unless a metamethod call's header and `n` arguments fit below
    /// the stack's end.
    GuardFits(u8),
    /// Leave for the interpreter's handler of the instruction at `pc`, the
    /// frame in its slots (R1): a `TAILCALL` or a `RETURN` that leaves the
    /// frame, whose interpreter work is all there is left.
    Dispatch {
        pc: u32,
        word: u64,
    },
    /// Call `R[a]` with `nargs` arguments in `R[a+4..]` (stored), `c` the
    /// CALL's result operand; edge 0 is the resume block. `nargs` is
    /// `u8::MAX` for arguments up to `top`, their count the operand.
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
    pub(crate) const R_UPVAL: Effects = Effects(1 << 8);
    pub(crate) const R_FRAME: Effects = Effects(1 << 9);
    pub(crate) const W_UPVAL: Effects = Effects(1 << 16);
    pub(crate) const W_FRAME: Effects = Effects(1 << 17);
    pub(crate) const R_TAB: Effects = Effects(1 << 10);
    pub(crate) const R_SHAPE: Effects = Effects(1 << 11);
    pub(crate) const R_META: Effects = Effects(1 << 12);
    pub(crate) const W_TAB: Effects = Effects(1 << 18);
    pub(crate) const W_SHAPE: Effects = Effects(1 << 19);
    pub(crate) const W_META: Effects = Effects(1 << 20);
    /// The collector's state: a barrier or an allocation.
    pub(crate) const W_GC: Effects = Effects(1 << 21);
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
            Guard(_) | GuardTrue | GuardFalse | GuardSame | IAdd | ISub | IMul | INeg | IShl
            | IShr | IDivFloor | IModFloor | LDivFloor | LModFloor | LToI | FToIExact | ToF64 => {
                Effects::MAY_DEOPT
            }
            GuardNoFlag(_) => Effects::MAY_DEOPT | Effects::R_FRAME,
            GuardShape(_) => Effects::MAY_DEOPT | Effects::R_SHAPE,
            ClassBits(_) | ClassIndex(_) => Effects::R_META,
            ShapeIs(_) => Effects::R_SHAPE,
            TabLoad { .. } | ArrLoad => Effects::R_TAB,
            TabStore { .. } | ArrStore => Effects::W_TAB,
            TabHasRoom(_) | TabASize | TabArray => Effects::R_SHAPE,
            TabSetShape(_) => Effects::W_SHAPE,
            Barrier => Effects::W_GC,
            HasFlag(_) => Effects::R_FRAME,
            CellLoad => Effects::R_UPVAL,
            CellStore => Effects::W_UPVAL,
            GcCheck => Effects::MAY_DEOPT | Effects::R_GC,
            Helper(h) => h.sig().effects,
            Resume { .. } => Effects::R_SLOT,
            Jump | Br => Effects::TERMINATOR,
            Return { .. } => Effects::TERMINATOR | Effects::TAILOUT | Effects::R_SLOT,
            GuardFits(_) => Effects::MAY_DEOPT,
            Call { .. } | CallMm { .. } | Dispatch { .. } => {
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

    /// How many values the instruction defines.
    pub(crate) fn num_results(self) -> usize {
        use Op::*;
        match self {
            Helper(h) => h.sig().ret.is_some() as usize,
            Store(_)
            | CellStore
            | GuardShape(_)
            | TabStore { .. }
            | TabSetShape(_)
            | ArrStore
            | Barrier
            | GuardTrue
            | GuardFalse
            | GuardSame
            | GuardNoFlag(_)
            | GcCheck
            | Jump
            | Br
            | Return { .. }
            | Call { .. }
            | CallMm { .. }
            | Dispatch { .. }
            | GuardFits(_)
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
            Load(_) | UpvalValue(_) | CellLoad | Resume { .. } => Ty::ANY,
            UpvalCell(_) => Ty::PTR,
            NumVarargs | ClassBits(_) | TabASize => Ty::I32,
            ClassIndex(_) | TabLoad { .. } | ArrLoad => Ty::ANY,
            TabArray => Ty::PTR,
            TabHasRoom(_) | ShapeIs(_) => Ty::B1,
            VarargAt(_) | VarargIdx => Ty::ANY,
            HasFlag(_) => Ty::B1,
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
            Guard(set) | Refine(set) => Ty {
                set: args[0].set & set,
                ..args[0]
            },
            IAdd | ISub | IMul | IAddNo | ISubNo | IMulNo | INeg | IAnd | IOr | IXor | INot
            | IShl | IShr | IDivFloor | IModFloor | LToI | FToIExact => Ty::I32,
            IToF | LToF | FAdd | FSub | FMul | FDiv | FNeg | FAbs | FSqrt | FFloor | FCeil
            | FIDiv | ToF64 => Ty::F64,
            IToL | FToL | LAdd | LSub | LMul | LUDiv | LNeg | LAnd | LOr | LXor | LNot | LShl
            | LShr | LDivFloor | LModFloor => Ty::I64,
            Helper(HelperId::Arith(ArithKind::Div | ArithKind::Pow)) => Ty::val(TypeSet::FLOAT),
            Helper(HelperId::Arith(ArithKind::Add | ArithKind::Sub | ArithKind::Mul))
            | Helper(HelperId::Arith(ArithKind::Mod | ArithKind::IDiv))
            | Helper(HelperId::Unm) => Ty::val(TypeSet::NUM),
            Helper(HelperId::Arith(_) | HelperId::BNot | HelperId::Len) => Ty::val(TypeSet::INT),
            Helper(HelperId::Concat) => Ty::val(TypeSet::STR),
            Helper(HelperId::Closure { .. }) => Ty::val(TypeSet::FUN),
            Helper(HelperId::NewTable(_) | HelperId::VarargTable(_)) => Ty::val(TypeSet::TAB),
            Helper(h) => Ty::of_rep(h.sig().ret.unwrap_or(Rep::Val)),
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
            GuardNoFlag(m) => format!("guard.noflag {m:#x}"),
            HasFlag(m) => format!("hasflag {m:#x}"),
            SameBits => "samebits".into(),
            ICmp(c) => format!("icmp.{}", c.name()),
            LCmp(c) => format!("lcmp.{}", c.name()),
            FCmp(c) => format!("fcmp.{}", c.name()),
            UpvalValue(i) => format!("upval u{i}"),
            UpvalCell(i) => format!("upcell u{i}"),
            Helper(h) => format!("call.{}", h.name()),
            Resume { c } => format!("resume c={c}"),
            Return { a, n } => format!("return r{a} n={n}"),
            Call { a, nargs, c, pc } => format!("call r{a} nargs={nargs} c={c} pc{pc}"),
            other => format!("{other:?}").to_lowercase(),
        }
    }
}
