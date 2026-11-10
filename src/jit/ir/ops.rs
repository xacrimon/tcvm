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

    #[cfg(feature = "tcvm-dev")]
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

/// The libm functions regions call. Their NaNs are the hardware default or an operand's,
/// both plain floats in a `Value`, so results need no canonicalizing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Libm {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Exp,
    Log,
    Log2,
    Log10,
    Pow,
    Atan2,
    Fmod,
}

impl Libm {
    pub(crate) fn binary(self) -> bool {
        matches!(self, Libm::Pow | Libm::Atan2 | Libm::Fmod)
    }
}

/// Rust routines a region calls with the C ABI (`jit::helpers`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum HelperId {
    /// `(f64, f64) -> f64`: Lua's float `%`.
    FMod,
    /// A libm function, called directly.
    Libm(Libm),
    /// `(Val, Val) -> Val`: the arithmetic of numbers and numeric strings, for
    /// the slow arm of a site that saw them, or `HELPER_FAIL_VAL` (anything
    /// else, a zero integer divisor).
    ArithCoerce(ArithKind),
    /// `Val -> Val`: `-v` and `~v` of a number, or `HELPER_FAIL_VAL`.
    Unm,
    BNot,
    /// `Val -> Val`: `#v` of a string or of a table without `__len`, or
    /// `HELPER_FAIL_VAL`.
    Len,
    /// `(Val, Val) -> Val`: `a .. b` of strings and numbers, or
    /// `HELPER_FAIL_VAL`.
    Concat,
    /// `Ptr ->`: the barrier of a store of a heap value into the cell.
    UpvalBarrier,
    /// `-> Val`: `CLOSURE` of prototype `proto` into `R[dst]`, from the
    /// captured registers' home slots.
    Closure {
        proto: u16,
        dst: u8,
    },
    /// `Ptr -> Val`: `CLOSURE` in the inlined frame at region slot `at`, of
    /// the operand closure's prototype `proto` (W3 §6). No header marks
    /// its open upvalues: exits mark the frame they write (`SnapFrame`).
    ClosureOf {
        proto: u16,
        dst: u8,
        at: u8,
    },
    /// `->`: close this frame's open upvalues from `R[a]` up, from the
    /// captured registers' home slots.
    Close(u8),
    /// `I64 -> Val`: `NEWTABLE` from the template at that address, a constant
    /// of the frame's prototype (W3 §6), which the region keeps.
    NewTable,
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
    /// `(Val, Val) -> Val`: `t[k]` for a table and a string (W2.5), or
    /// `HELPER_FAIL_VAL` where `__index` may answer.
    GetStr,
    /// `(Val, Val, Val) -> Val`: `t[k] = v` for a table and a string as
    /// `t[k] = v` stores, 0, or `HELPER_FAIL_VAL` where `__newindex` may run.
    SetStr,
    /// Builtins (12), each `-> Val` with `HELPER_FAIL_VAL` where the builtin
    /// must run: `type(v)`, `getmetatable(v)`, `setmetatable(t, mt)` on a
    /// table without one, `rawget(t, k)`, `rawset(t, k, v)`, `rawequal(a,
    /// b)` (`-> I32`), `rawlen(v)`, `tostring(v)` without `__tostring`,
    /// `tonumber(v)`, `string.len(s)`, `string.sub(s, i, j)` (`j` nil for
    /// -1), `string.byte(s, i)`'s first result (`i` nil for 1).
    Type,
    GetMetatable,
    SetMetatable,
    RawGet,
    RawSet,
    RawEqual,
    RawLen,
    ToString,
    ToNumber,
    /// `string.sub(s, i, j)` and `string.byte(s, i)`'s one result, the
    /// string its payload (`Ptr`), the positions `I32` (`ToPos`).
    StrSub,
    StrByte,
    /// `table.insert(t, v)` and `table.remove(t)` on a table whose
    /// metatable has none of `__index`, `__newindex` and `__len`, else
    /// `HELPER_FAIL_VAL`; `insert` gives nil.
    TInsert,
    TRemove,
    /// `(Val, Val, Val) ->`: `recv[key] = v` through `__newindex` tables,
    /// `raw_set_keyed` when `keyed` (`t[k] = v`), or `HELPER_FAIL_VAL`.
    NewIndex {
        keyed: bool,
    },
}

/// What a `Val` helper returns when the interpreter must run the
/// instruction: the negative quiet NaN, which no `Value` is (floats are
/// canonical) and one `movz` makes.
pub(crate) const HELPER_FAIL_VAL: u64 = 0xFFF8_0000_0000_0000;

/// A `Call` or `TailEnter` whose callee's prototype is not known.
pub(crate) const NO_PROTO: u32 = u32::MAX;

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
    #[cfg(feature = "tcvm-dev")]
    pub(crate) args: &'static [Rep],
    pub(crate) ret: Option<Rep>,
    pub(crate) effects: Effects,
}

impl HelperId {
    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn name(self) -> String {
        match self {
            HelperId::FMod => "fmod".into(),
            HelperId::Libm(f) => format!("{f:?}").to_lowercase(),
            HelperId::ArithCoerce(k) => format!("arith_coerce.{k:?}").to_lowercase(),
            HelperId::Unm => "unm".into(),
            HelperId::BNot => "bnot".into(),
            HelperId::Len => "len".into(),
            HelperId::Concat => "concat".into(),
            HelperId::Closure { proto, dst } => format!("closure p{proto} r{dst}"),
            HelperId::ClosureOf { proto, dst, at } => format!("closure.of p{proto} r{dst} @{at}"),
            HelperId::Close(a) => format!("close r{a}"),
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
            other => format!("{other:?}").to_lowercase(),
        }
    }

    pub(crate) fn sig(self) -> Sig {
        use Rep::*;
        const RT: &[Ctx] = &[Ctx::Rt];
        const FRAME: &[Ctx] = &[Ctx::Rt, Ctx::Thread, Ctx::Base];
        const BASE: &[Ctx] = &[Ctx::Rt, Ctx::Base];
        const STACK: &[Ctx] = &[Ctx::Thread, Ctx::Base];
        const THREAD: &[Ctx] = &[Ctx::Rt, Ctx::Thread];
        const ALLOC: Flags = Flags::MAY_ALLOC;
        const NO: Flags = Flags::empty();
        const NONE: Mem = Mem::empty();
        let pure = Effects::NONE;
        // A metamethod lookup moves a table off a stale class (`Table::meta`)
        // or adopts a metatable's class.
        let lookup = Effects::new(ALLOC, Mem::SHAPE | Mem::META, Mem::SHAPE | Mem::META);
        let table_write = Mem::TAB | Mem::ARR | Mem::SHAPE | Mem::META | Mem::GC;
        // Stores that touch table memory alone: a metamethod fails them
        // rather than run, and an allocation never collects (W2.7).
        let table_store = Effects::new(ALLOC, table_write, table_write);
        let e = Effects::new;
        let (ctx, imm, args, ret, effects): (_, _, &'static [Rep], _, _) = match self {
            HelperId::FMod => (&[][..], None, &[F64, F64], Some(F64), pure),
            HelperId::Libm(f) => {
                let args: &[Rep] = if f.binary() { &[F64, F64] } else { &[F64] };
                (&[][..], None, args, Some(F64), pure)
            }
            // Integers out of i32 are boxed.
            HelperId::ArithCoerce(k) => (
                RT,
                Some(k as u64),
                &[Val, Val],
                Some(Val),
                e(ALLOC, NONE, NONE),
            ),
            HelperId::Unm | HelperId::BNot => (RT, None, &[Val], Some(Val), e(ALLOC, NONE, NONE)),
            HelperId::Len => (
                RT,
                None,
                &[Val],
                Some(Val),
                e(ALLOC, Mem::TAB | Mem::ARR | lookup.reads, lookup.writes),
            ),
            HelperId::Concat => (RT, None, &[Val, Val], Some(Val), e(ALLOC, NONE, NONE)),
            HelperId::UpvalBarrier => (THREAD, None, &[Ptr], None, e(NO, NONE, Mem::GC)),
            // New open upvalues join the thread's list.
            HelperId::Closure { proto, dst } => (
                FRAME,
                Some(proto as u64 | (dst as u64) << 32),
                &[],
                Some(Val),
                e(ALLOC, Mem::SLOT, Mem::FRAME),
            ),
            HelperId::ClosureOf { proto, dst, at } => (
                FRAME,
                Some(proto as u64 | (dst as u64) << 32 | (at as u64) << 40),
                &[Ptr],
                Some(Val),
                e(ALLOC, Mem::SLOT, NONE),
            ),
            HelperId::Close(a) => (
                FRAME,
                Some(a as u64),
                &[],
                None,
                e(NO, Mem::SLOT | Mem::FRAME, Mem::UPVAL | Mem::FRAME),
            ),
            HelperId::NewTable => (RT, None, &[I64], Some(Val), e(ALLOC, NONE, NONE)),
            HelperId::SetList { a, n, off } => (
                FRAME,
                Some(a as u64 | (n as u64) << 8 | (off as u64) << 16),
                &[],
                None,
                e(ALLOC, Mem::SLOT | Mem::FRAME, Mem::TAB | Mem::ARR | Mem::GC),
            ),
            HelperId::VarargGet => (BASE, None, &[Val], Some(Val), e(ALLOC, NONE, NONE)),
            HelperId::VarargTable(np) => (
                BASE,
                Some(np as u64),
                &[],
                Some(Val),
                e(ALLOC, NONE, Mem::SLOT),
            ),
            HelperId::Mm(bit) => (RT, Some(bit as u64), &[Val, Val], Some(Val), lookup),
            HelperId::IndexFn | HelperId::NewIndexFn => (RT, None, &[Val, Val], Some(Val), lookup),
            HelperId::TforNext { a, count } => (
                BASE,
                Some(a as u64 | (count as u64) << 8),
                &[],
                Some(Val),
                e(
                    ALLOC,
                    Mem::SLOT | Mem::TAB | Mem::ARR | Mem::SHAPE,
                    Mem::SLOT,
                ),
            ),
            HelperId::TforIpairs => (
                RT,
                None,
                &[Val, Val],
                Some(Val),
                Effects::reads(Mem::ARR | Mem::TAB | Mem::META),
            ),
            HelperId::TopArgs(a) => (
                STACK,
                Some(a as u64),
                &[],
                Some(I32),
                Effects::reads(Mem::FRAME),
            ),
            HelperId::VarargAll(a) => (
                STACK,
                Some(a as u64),
                &[],
                Some(Val),
                e(NO, Mem::FRAME, Mem::SLOT | Mem::FRAME),
            ),
            // Fails rather than refresh a stale class (`jit_index`).
            HelperId::Index => (
                RT,
                None,
                &[Val, Val],
                Some(Val),
                Effects::reads(Mem::TAB | Mem::ARR | Mem::SHAPE | Mem::META),
            ),
            HelperId::GetStr => (
                &[][..],
                None,
                &[Val, Val],
                Some(Val),
                Effects::reads(Mem::TAB | Mem::SHAPE | Mem::META),
            ),
            HelperId::SetStr => (RT, None, &[Val, Val, Val], Some(Val), table_store),
            // `raw_set` may grow the table.
            HelperId::NewIndex { keyed } => (
                RT,
                Some(keyed as u64),
                &[Val, Val, Val],
                Some(Val),
                table_store,
            ),
            HelperId::Type | HelperId::ToNumber => {
                (RT, None, &[Val], Some(Val), e(ALLOC, NONE, NONE))
            }
            HelperId::RawLen => (
                RT,
                None,
                &[Val],
                Some(Val),
                e(ALLOC, Mem::TAB | Mem::ARR, NONE),
            ),
            HelperId::ToString => (RT, None, &[Val], Some(Val), lookup),
            HelperId::GetMetatable => (
                RT,
                None,
                &[Val],
                Some(Val),
                Effects::reads(Mem::TAB | Mem::SHAPE | Mem::META),
            ),
            HelperId::SetMetatable => (RT, None, &[Val, Val], Some(Val), table_store),
            HelperId::RawGet => (
                &[][..],
                None,
                &[Val, Val],
                Some(Val),
                Effects::reads(Mem::TAB | Mem::ARR | Mem::SHAPE),
            ),
            HelperId::RawSet => (RT, None, &[Val, Val, Val], Some(Val), table_store),
            HelperId::RawEqual => (&[][..], None, &[Val, Val], Some(I32), pure),
            HelperId::StrSub => (RT, None, &[Ptr, I32, I32], Some(Val), e(ALLOC, NONE, NONE)),
            HelperId::StrByte => (&[][..], None, &[Ptr, I32], Some(Val), pure),
            HelperId::TInsert => (RT, None, &[Val, Val], Some(Val), table_store),
            HelperId::TRemove => (RT, None, &[Val], Some(Val), table_store),
        };
        #[cfg(not(feature = "tcvm-dev"))]
        let _ = args;
        Sig {
            ctx,
            imm,
            #[cfg(feature = "tcvm-dev")]
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
    /// A value compiled as a constant differed (an intrinsic's callee, a
    /// by-value upvalue): the exit records `CONST_MISS`, and the recompile
    /// takes only what is known for certain as constant.
    Const,
    /// An array access left the array part, or met a nil under a class with
    /// `__index`/`__newindex`, where the site's feedback never saw one: the
    /// exit records `ARR_MISS`, and the recompile keeps the slow arm.
    Bounds,
    /// A helper failed where the site's feedback never saw a metamethod
    /// run: the exit records one, and the recompile calls it (5.4).
    Mm,
    /// A prologue guard on an entry value (5.2): fails to the entry-fail
    /// stub, which records the entry's kinds.
    Entry,
    /// A `LazyExit`: a dependency broke.
    Lazy,
    /// A guard on a builtin's arguments, at its CALL: the exit takes an
    /// argument the builtin raises on (no number, no string) to the
    /// interpreter as `Slow` does, recording and counting nothing; any other
    /// kind is a `Type` exit. One exit for both, which compiled code cannot
    /// tell apart at a guard that covers some numbers.
    Arg,
    /// `Arg` for `min`/`max`, which compare their arguments: strings mixed with
    /// numbers raise too.
    ArgCmp,
    /// A builtin's result was of a kind its inline code does not make, where
    /// the call's feedback never saw one (`floor` without an integer,
    /// `string.byte` out of range): the exit records `OTHER_RESULT`.
    OtherResult,
}

impl ExitTag {
    pub(crate) fn widenable(self) -> bool {
        matches!(
            self,
            ExitTag::Type
                | ExitTag::Overflow
                | ExitTag::NeverRan
                | ExitTag::Shape
                | ExitTag::Mm
                | ExitTag::Bounds
                | ExitTag::Const
                | ExitTag::OtherResult
                | ExitTag::Arg
                | ExitTag::ArgCmp
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
    /// A heap object's pointer (`unbox.ptr` of a `KObj`), its pool index.
    KPtr(u32),
    KI32(i32),
    KI64(i64),
    KB1(bool),
    /// An f64 by its bits.
    KF64(u64),

    // --- home slots -----------------------------------------------------------
    /// `R[r]` as a `Val`.
    Load(u8),
    /// A load whose slot holds its value wherever its register is read, and
    /// which a call's snapshot names only for that slot, kept in place
    /// (`opt::snaps`, the last pass): the allocator loads it from the slot
    /// where it is wanted, and an exit reads it there.
    Home(u8),
    /// Parameter `k` of a function region, which every entry passes in
    /// `x_k` (W4 §2): no slot holds it until an exit writes it.
    Param(u8),
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
    /// Exits when the `Val` is nil and the `B1` true: an array element whose
    /// table's class may have the metamethod (W2.3); `opt::miss` expands it.
    GuardMiss,
    /// Exits when the two `Val`s are not the same word.
    GuardSame,
    /// Exits unless the `Val` is a Lua closure of `Func::protos[k]`.
    GuardProto(u32),
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
    /// C's `%` (`fmod` of integers): the sign of the dividend, `MIN % -1`
    /// is 0; exits on a zero divisor.
    IRem,
    /// The machine's arithmetic shift right, by the amount mod 32.
    IAsr,
    /// The low half of `(a * b) >> s`, operands unsigned and the product
    /// 64-bit: a division by a constant multiplying by its magic number.
    IUMulShr(u8),
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
    /// `IRem` of i64s.
    LRem,
    /// The machine's arithmetic shift right, by the amount mod 64.
    LAsr,
    LCmp(Cc),
    LToF,
    /// Exits when outside i32.
    LToI,
    /// The f64 of an i64 within ±2^53, which converts exactly; exits
    /// otherwise.
    LToFExact,

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
    /// The i32 an f64 holds exactly; exits otherwise. -0.0 gives 0: every
    /// use is Lua's float-to-integer conversion (keys, bitwise operands,
    /// `floor`/`ceil`, loop limits).
    FToIExact,
    /// The i64 an f64 holds exactly; exits otherwise (`2^63`, NaN, inf).
    FToLExact,
    /// The f64 of a `Val` of the kinds in the set, as Lua converts them: a
    /// float inline, small and big integers and strings (`tonumber`'s
    /// rules) in out-of-line arms; exits otherwise. Arithmetic never passes
    /// `STR`: string arithmetic is the string metamethods'.
    ToF64(TypeSet),
    /// A string position: the integer `luaL_checkinteger` takes a `Val` of
    /// the kinds in the set for, clamped to i32, which changes no position's
    /// meaning as strings are shorter than 2^31. A small integer inline; a
    /// big one, a float with an i64 value and a string that converts to one
    /// (`jit_str_to_i64`) out of line; exits otherwise.
    ToPos(TypeSet),
    /// The i64 of an f64 rounded toward zero, saturating; NaN gives 0.
    FToL,

    // --- closure --------------------------------------------------------------
    /// A by-value upvalue of the running closure.
    UpvalValue(u8),
    /// The cell of a shared upvalue of the running closure.
    UpvalCell(u8),
    /// A by-value upvalue of the Lua closure (a `Ptr`): of an inlined
    /// callee known by its prototype (E1). A memory read, so that it stays
    /// behind that guard.
    UpvalValueOf(u8),
    /// The cell of a shared upvalue of the Lua closure (a `Ptr`): of an
    /// inlined callee (W3 §6).
    UpvalCellOf(u8),
    /// The region frame's function, from its header.
    FrameFn,
    /// `top` past region slot `at` and the operand's count (an i32) of
    /// values, as a call returning them up to `top` leaves it (W3 §7).
    SetTop(u8),
    /// The value of an upvalue cell. An open cell is a slot, perhaps one of
    /// this region's: it reads and writes slots too.
    CellLoad,
    /// `cell = v`, without its barrier.
    CellStore,
    /// Whether the frame's header has the flag (`vm::frame::flag`).
    HasFlag(u8),
    /// Whether numbers have a metatable.
    NumMtSet,
    /// Whether the metatable of a type (by its `State` slot) is still the
    /// one whose word that is.
    TypeMtIs {
        slot: u8,
        word: u64,
    },
    /// Whether the function (a `Ptr` to its closure) is a Lua closure
    /// without varargs.
    FnFixedLua,
    /// Whether the function (a `Ptr` to its closure) is a native.
    FnNative,
    /// Whether the Lua closure's window fits from this frame's base.
    FnFits,

    // --- tables: `Ptr` operands are a table's lock (its value's payload) ----
    /// Exits unless the table's shape is `shapes[i]`.
    GuardShape(u32),
    /// Whether the table's shape is `shapes[i]`, a `B1`.
    ShapeIs(u32),
    /// Exits unless the table's shape is one of `shape_sets[i]`.
    GuardShapeIn(u32),
    /// Whether the table's shape is one of `shape_sets[i]`, a `B1`.
    ShapeIn(u32),
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
    /// The metamethod of bit index `i` the table's class has, nil without.
    ClassMm(u8),
    /// Whether the table's class has the metamethod of bit index `i` now, a
    /// `B1`: false without a class.
    TabHasMm(u8),
    /// Whether the table's hash parts in the mask (`HASH_*`) hold no key, a
    /// `B1`.
    TabNoKeys(u8),
    /// A string's length (`I32`: strings are capped below 2^31), the operand
    /// its value's payload (`Ptr`).
    StrLenOf,
    /// `string.byte(s, i)`'s byte (`I32`) of the string whose payload is
    /// the first operand, at Lua position `i` (`I32`, negative from the end);
    /// exits out of range.
    StrByteAt,
    /// `i cc f` of an i64 and an f64 (`Eq`, `Lt`, `Le`, `Gt`, `Ge`), exact as
    /// Lua compares them (`LTintfloat` and kin; equal when the float is that
    /// integer): as floats while `|i| <= 2^53` converts exactly, else by the
    /// float's floor or ceiling as an integer, out of line.
    IFCmp(Cc),
    /// `x cc y` (`Eq`, `Lt`, `Le`) of two numbers of any kinds, exact as
    /// Lua's `LTnum`, `LEnum` and equality: two floats inline, integers and
    /// mixes out of line (a mix as `IFCmp`).
    NumCmp(Cc),
    /// Whether string `a` orders before (`StrLe`: not after) `b`, by bytes;
    /// the operands their payloads.
    StrLt,
    StrLe,
    /// The array part's size (`I32`) and cell.
    TabASize,
    TabArray,
    /// `arr[i]` of an array part, `i` an `I32` below its size.
    ArrLoad,
    ArrStore,
    /// The index of the table's first non-nil array slot at or after the
    /// `I32` operand, or the array part's size when there is none.
    ArrNext,
    /// The index (`I32`) of the table's first non-nil named slot at or after
    /// the operand, its metatable's slot skipped; -1 when there is none.
    NamedNext,
    /// The value of the table's named slot and the key of its shape's slot
    /// at an `I32` index below the shape's slot count.
    NamedVal,
    NamedKey,
    /// The table's backward barrier unless it is gray; with a second
    /// operand, the value stored, for that value alone (`simplify` drops it
    /// once the value is never a heap object).
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
    /// Call the first operand with the rest as arguments from a header at
    /// region slot `hdr`, past the calling frame's registers (`stage_mm!`): a
    /// metamethod or a generic `for`'s iterator. Edge 0 is the resume block,
    /// which takes `c - 1` results, landed from region slot `land` when more
    /// than two.
    CallMm {
        pc: u32,
        nargs: u8,
        land: u8,
        c: u8,
        hdr: u8,
    },
    /// Leave for the interpreter's handler of the instruction at `pc`, the
    /// frame in its slots (R1): a `TAILCALL` or a `RETURN` that leaves the
    /// frame, whose interpreter work is all there is left.
    Dispatch {
        pc: u32,
        word: u64,
    },
    /// `TAILCALL`'s fast arm (5.5): the Lua closure (a `Ptr`) takes over
    /// the frame, its `nargs` arguments stored to `R[0..]`; missing
    /// parameters are nil-filled, header word 0 rewritten, and its first
    /// instruction dispatched. With `proto` (`Func::protos`, else
    /// `NO_PROTO`) the closure is known to be one of it, which takes at
    /// least `nargs` arguments, stored nil past those given.
    TailEnter {
        nargs: u8,
        proto: u32,
    },
    /// Call `R[a]` with `nargs` arguments in `R[a+4..]`, `c` the CALL's
    /// result operand; edge 0 is the resume block. `nargs` is `u8::MAX`
    /// for arguments up to `top` (stored), their count the operand; else
    /// the first eight are the operands, the rest stored (W4 §2). `proto`
    /// is the callee's prototype when a guard proved it (`Func::protos`),
    /// else `NO_PROTO`.
    Call {
        a: u8,
        nargs: u8,
        c: u8,
        pc: u32,
        proto: u32,
    },
    /// Leave for the interpreter (with a snapshot).
    Deopt,
    /// Leave once the region is invalidated (ws2 §5.4): nothing until then,
    /// its place patched to the exit after. One follows every call and
    /// table write, where a dependency can break.
    LazyExit,
}

bitflags::bitflags! {
    /// Memory classes, each versioned separately by GVN (8.4). `SLOT`, `TAB`
    /// and `ARR` reads and writes may name a key within the class
    /// (`Op::mem_key`).
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct Mem: u16 {
        const SLOT = 1;
        const TAB = 2;
        /// Array parts: their items (`ARR_ITEMS`) and their size and cell
        /// (`ARR_SIZE`).
        const ARR = 4;
        const SHAPE = 8;
        /// Metatable classes: their bits and metamethods.
        const META = 16;
        const UPVAL = 32;
        const FRAME = 64;
        /// The collector's state.
        const GC = 128;
        const ALL = 0xff;
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct Flags: u8 {
        const MAY_DEOPT = 1;
        const MAY_ALLOC = 2;
        const TAILOUT = 4;
        const TERMINATOR = 8;
        /// Must stay where it is for a reason no memory class names.
        const SIDE_EFFECT = 16;
    }
}

/// `Op::TabNoKeys` parts: integer keys, dictionary-mode string keys, other
/// keys.
pub(crate) const HASH_INTS: u8 = 1;
pub(crate) const HASH_STRS: u8 = 2;
pub(crate) const HASH_MISC: u8 = 4;

/// The key of an array part's items within `Mem::ARR`.
pub(crate) const ARR_ITEMS: u32 = 0;
/// The key of an array part's size and cell, which only a resize changes.
pub(crate) const ARR_SIZE: u32 = 1;

/// What an operation or a helper may do besides computing its results.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Effects {
    pub(crate) flags: Flags,
    pub(crate) reads: Mem,
    pub(crate) writes: Mem,
}

impl Effects {
    pub(crate) const NONE: Effects = Effects {
        flags: Flags::empty(),
        reads: Mem::empty(),
        writes: Mem::empty(),
    };

    const fn new(flags: Flags, reads: Mem, writes: Mem) -> Effects {
        Effects {
            flags,
            reads,
            writes,
        }
    }

    const fn flags(flags: Flags) -> Effects {
        Effects::new(flags, Mem::empty(), Mem::empty())
    }

    const fn reads(reads: Mem) -> Effects {
        Effects::new(Flags::empty(), reads, Mem::empty())
    }

    pub(crate) fn has(self, f: Flags) -> bool {
        self.flags.contains(f)
    }

    /// No effect, no exit, no memory: removable when unused, numbered and
    /// moved freely.
    pub(crate) fn is_pure(self) -> bool {
        self == Effects::NONE
    }

    pub(crate) fn reads_memory(self) -> bool {
        !self.reads.is_empty()
    }

    pub(crate) fn writes(self) -> bool {
        !self.writes.is_empty()
    }

    /// Whether the instruction may go when its results are unused.
    pub(crate) fn removable(self) -> bool {
        !self
            .flags
            .intersects(Flags::SIDE_EFFECT | Flags::MAY_DEOPT | Flags::TERMINATOR | Flags::TAILOUT)
            && self.writes.is_empty()
    }

    /// Whether the instruction may move, its memory and exits permitting.
    pub(crate) fn movable(self) -> bool {
        !self
            .flags
            .intersects(Flags::SIDE_EFFECT | Flags::TAILOUT | Flags::TERMINATOR)
    }
}

impl Op {
    pub(crate) fn effects(self) -> Effects {
        use Op::*;
        const DEOPT: Flags = Flags::MAY_DEOPT;
        let e = Effects::new;
        match self {
            Load(_) | Home(_) => Effects::reads(Mem::SLOT),
            // Its register is the entry's: nothing may move or copy it
            // (after a call no register or slot holds it).
            Param(_) => Effects::flags(Flags::SIDE_EFFECT),
            Store(_) => e(Flags::empty(), Mem::empty(), Mem::SLOT),
            Guard(_) | GuardTrue | GuardFalse | GuardMiss | GuardSame | GuardProto(_) | IAdd
            | ISub | IMul | INeg | IShl | IShr | IDivFloor | IModFloor | IRem | LDivFloor
            | LModFloor | LRem | LToI | LToFExact | FToIExact | FToLExact | ToF64(_) | ToPos(_)
            | StrByteAt => Effects::flags(DEOPT),
            GuardNoFlag(_) => e(DEOPT, Mem::FRAME, Mem::empty()),
            GuardShape(_) | GuardShapeIn(_) => e(DEOPT, Mem::SHAPE, Mem::empty()),
            GcCheck => e(DEOPT, Mem::GC, Mem::empty()),
            ShapeIs(_) | ShapeIn(_) | TabHasRoom(_) | NamedKey => Effects::reads(Mem::SHAPE),
            // A table's class changes only through helpers, which write `META`.
            ClassBits(_)
            | ClassIndex(_)
            | ClassMm(_)
            | TabHasMm(_)
            | NumMtSet
            | TypeMtIs { .. } => Effects::reads(Mem::META),
            TabLoad { .. } | TabNoKeys(_) => Effects::reads(Mem::TAB),
            TabStore { .. } => e(Flags::empty(), Mem::empty(), Mem::TAB),
            TabASize | TabArray | ArrLoad | ArrNext => Effects::reads(Mem::ARR),
            ArrStore => e(Flags::empty(), Mem::empty(), Mem::ARR),
            NamedNext | NamedVal => Effects::reads(Mem::TAB | Mem::SHAPE),
            TabSetShape(_) => e(Flags::empty(), Mem::empty(), Mem::SHAPE),
            Barrier => e(Flags::empty(), Mem::empty(), Mem::GC),
            HasFlag(_) | FnFits => Effects::reads(Mem::FRAME),
            UpvalValueOf(_) | UpvalCellOf(_) => Effects::reads(Mem::UPVAL),
            FrameFn => Effects::reads(Mem::FRAME),
            SetTop(_) => e(Flags::empty(), Mem::empty(), Mem::FRAME),
            CellLoad => Effects::reads(Mem::UPVAL | Mem::SLOT),
            CellStore => e(Flags::empty(), Mem::empty(), Mem::UPVAL | Mem::SLOT),
            Helper(h) => h.sig().effects,
            Resume { .. } => e(Flags::SIDE_EFFECT, Mem::SLOT, Mem::empty()),
            Jump | Br => Effects::flags(Flags::TERMINATOR),
            Return { .. } => e(Flags::TERMINATOR | Flags::TAILOUT, Mem::SLOT, Mem::empty()),
            Call { .. } | CallMm { .. } | Dispatch { .. } | TailEnter { .. } => {
                e(Flags::TERMINATOR | Flags::TAILOUT, Mem::ALL, Mem::ALL)
            }
            Deopt => Effects::flags(Flags::TERMINATOR | DEOPT),
            LazyExit => Effects::flags(Flags::SIDE_EFFECT | DEOPT),
            _ => Effects::NONE,
        }
    }

    /// The key within its one memory class a keyed read or write names:
    /// accesses with other keys never alias it.
    pub(crate) fn mem_key(self) -> Option<u32> {
        match self {
            Op::Load(r) | Op::Home(r) | Op::Store(r) => Some(r as u32),
            Op::TabLoad { key, .. } | Op::TabStore { key, .. } => Some(key),
            Op::ArrLoad | Op::ArrStore => Some(ARR_ITEMS),
            Op::TabASize | Op::TabArray => Some(ARR_SIZE),
            _ => None,
        }
    }

    pub(crate) fn is_terminator(self) -> bool {
        self.effects().has(Flags::TERMINATOR)
    }

    pub(crate) fn may_deopt(self) -> bool {
        self.effects().has(Flags::MAY_DEOPT)
    }

    pub(crate) fn is_pure(self) -> bool {
        self.effects().is_pure()
    }

    /// What lowering may compute where it is used instead: a test a
    /// branch, guard or select takes, a product a sum takes, a tested bit.
    /// Its operands are read there (`Op::Home` counts on the set).
    pub(crate) fn folds(self) -> bool {
        use Op::*;
        matches!(
            self,
            ICmp(_)
                | LCmp(_)
                | FCmp(_)
                | IsType(_)
                | IsFalsy
                | SameBits
                | ShapeIs(_)
                | ShapeIn(_)
                | HasFlag(_)
                | NumMtSet
                | TypeMtIs { .. }
                | FnFixedLua
                | FnNative
                | FnFits
                | TabHasRoom(_)
                | TabNoKeys(_)
                | IMulNo
                | LMul
                | IAnd
                | TabHasMm(_)
        )
    }

    /// How many values the instruction defines.
    pub(crate) fn num_results(self) -> usize {
        use Op::*;
        match self {
            Helper(h) => h.sig().ret.is_some() as usize,
            Store(_)
            | SetTop(_)
            | CellStore
            | GuardShape(_)
            | GuardShapeIn(_)
            | TabStore { .. }
            | TabSetShape(_)
            | ArrStore
            | Barrier
            | GuardTrue
            | GuardFalse
            | GuardMiss
            | GuardSame
            | GuardProto(_)
            | GuardNoFlag(_)
            | GcCheck
            | Jump
            | Br
            | Return { .. }
            | Call { .. }
            | CallMm { .. }
            | Dispatch { .. }
            | TailEnter { .. }
            | Deopt
            | LazyExit => 0,
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
            KPtr(p) => Ty {
                refine: crate::jit::ir::types::Refine::Const(p),
                ..Ty::PTR
            },
            KI32(_) => Ty::I32,
            KB1(_) => Ty::B1,
            KI64(_) => Ty::I64,
            KF64(_) => Ty::F64,
            Load(_)
            | Home(_)
            | Param(_)
            | UpvalValue(_)
            | UpvalValueOf(_)
            | CellLoad
            | Resume { .. } => Ty::ANY,
            UpvalCell(_) | UpvalCellOf(_) => Ty::PTR,
            FrameFn => Ty::val(TypeSet::FUN),
            NumVarargs | ClassBits(_) | TabASize | ArrNext | NamedNext => Ty::I32,
            NamedVal => Ty::ANY,
            NamedKey => Ty::val(TypeSet::STR),
            ClassIndex(_) | TabLoad { .. } | ArrLoad => Ty::ANY,
            TabArray => Ty::PTR,
            TabHasRoom(_) | TabNoKeys(_) | TabHasMm(_) | ShapeIs(_) | ShapeIn(_) => Ty::B1,
            VarargAt(_) | VarargIdx => Ty::ANY,
            HasFlag(_) | FnFixedLua | FnNative | FnFits | NumMtSet | TypeMtIs { .. } => Ty::B1,
            ClassMm(_) => Ty::ANY,
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
            IsType(_) | IsFalsy | SameBits | ICmp(_) | LCmp(_) | FCmp(_) | IFCmp(_) | NumCmp(_)
            | StrLt | StrLe => Ty::B1,
            Select => args[1].join(args[2]),
            Guard(set) | Refine(set) => Ty {
                set: args[0].set & set,
                ..args[0]
            },
            IAdd | ISub | IMul | IAddNo | ISubNo | IMulNo | INeg | IAnd | IOr | IXor | INot
            | IShl | IShr | IDivFloor | IModFloor | IRem | IAsr | IUMulShr(_) | LToI
            | FToIExact => Ty::I32,
            IToF | LToF | LToFExact | FAdd | FSub | FMul | FDiv | FNeg | FAbs | FSqrt | FFloor
            | FCeil | FIDiv | ToF64(_) => Ty::F64,
            StrLenOf | StrByteAt | ToPos(_) => Ty::I32,
            IToL | FToL | FToLExact | LAdd | LSub | LMul | LUDiv | LNeg | LAnd | LOr | LXor
            | LNot | LShl | LShr | LDivFloor | LModFloor | LRem | LAsr => Ty::I64,
            Helper(HelperId::ArithCoerce(ArithKind::Div | ArithKind::Pow)) => {
                Ty::val(TypeSet::FLOAT)
            }
            Helper(HelperId::ArithCoerce(ArithKind::Add | ArithKind::Sub | ArithKind::Mul))
            | Helper(HelperId::ArithCoerce(ArithKind::Mod | ArithKind::IDiv))
            | Helper(HelperId::Unm) => Ty::val(TypeSet::NUM),
            Helper(HelperId::ArithCoerce(_) | HelperId::BNot | HelperId::Len) => {
                Ty::val(TypeSet::INT)
            }
            Helper(HelperId::Concat | HelperId::Type | HelperId::ToString | HelperId::StrSub) => {
                Ty::val(TypeSet::STR)
            }
            Helper(HelperId::RawLen) => Ty::val(TypeSet::INT),
            Helper(HelperId::StrByte) => Ty::val(TypeSet::SMALL | TypeSet::NIL),
            Helper(HelperId::Closure { .. } | HelperId::ClosureOf { .. }) => Ty::val(TypeSet::FUN),
            Helper(HelperId::NewTable | HelperId::VarargTable(_)) => Ty::val(TypeSet::TAB),
            Helper(h) => Ty::of_rep(h.sig().ret.unwrap_or(Rep::Val)),
            _ => Ty::ANY,
        }
    }

    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn name(self) -> String {
        use Op::*;
        match self {
            KVal(b) => format!("kval {:#x}", b),
            KObj(p) => format!("kobj k{p}"),
            KPtr(p) => format!("kptr k{p}"),
            KI32(v) => format!("ki32 {v}"),
            KI64(v) => format!("ki64 {v}"),
            KB1(v) => format!("kb1 {v}"),
            KF64(b) => format!("kf64 {:?}", f64::from_bits(b)),
            IUMulShr(s) => format!("iumulshr {s}"),
            Load(r) => format!("load r{r}"),
            Home(r) => format!("home r{r}"),
            Param(k) => format!("param {k}"),
            Store(r) => format!("store r{r}"),
            Box => "box".into(),
            Unbox(rep) => format!("unbox.{}", rep.name()),
            IsType(s) => format!("istype {}", Ty::val(s)),
            IsFalsy => "isfalsy".into(),
            Select => "select".into(),
            Guard(s) => format!("guard {}", Ty::val(s)),
            GuardTrue => "guard.true".into(),
            GuardFalse => "guard.false".into(),
            GuardMiss => "guard.miss".into(),
            GuardSame => "guard.same".into(),
            GuardProto(k) => format!("guard.proto p{k}"),
            GcCheck => "gccheck".into(),
            GuardNoFlag(m) => format!("guard.noflag {m:#x}"),
            HasFlag(m) => format!("hasflag {m:#x}"),
            SameBits => "samebits".into(),
            ICmp(c) => format!("icmp.{}", c.name()),
            LCmp(c) => format!("lcmp.{}", c.name()),
            FCmp(c) => format!("fcmp.{}", c.name()),
            UpvalValue(i) => format!("upval u{i}"),
            UpvalValueOf(i) => format!("upval.of u{i}"),
            UpvalCell(i) => format!("upcell u{i}"),
            UpvalCellOf(i) => format!("upcell.of u{i}"),
            FrameFn => "framefn".into(),
            SetTop(at) => format!("settop r{at}"),
            Helper(h) => format!("call.{}", h.name()),
            Resume { c } => format!("resume c={c}"),
            Return { a, n } => format!("return r{a} n={n}"),
            Call {
                a,
                nargs,
                c,
                pc,
                proto,
            } => {
                let p = if proto == NO_PROTO {
                    String::new()
                } else {
                    format!(" p{proto}")
                };
                format!("call r{a} nargs={nargs} c={c} pc{pc}{p}")
            }
            other => format!("{other:?}").to_lowercase(),
        }
    }
}
