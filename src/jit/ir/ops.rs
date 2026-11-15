//! The op table: one row per op gives its dialect, operand form, result,
//! effects and flags. The enum, the static table and the printer's names
//! come from the rows, so an op without a row does not compile. Helpers have
//! a table of their own.

use crate::jit::ir::types::Rep;

/// An integer or float comparison. On a NaN operand every float compare is
/// false except `Ne`, which is true.
#[repr(u8)]
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
    Ls,
    Hi,
    Hs,
}

impl Cc {
    const ALL: [Cc; 10] = [
        Cc::Eq,
        Cc::Ne,
        Cc::Lt,
        Cc::Le,
        Cc::Gt,
        Cc::Ge,
        Cc::Lo,
        Cc::Ls,
        Cc::Hi,
        Cc::Hs,
    ];

    pub(crate) fn of(aux: u16) -> Cc {
        Cc::ALL[aux as usize]
    }

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
            Cc::Ls => Cc::Hs,
            Cc::Hi => Cc::Lo,
            Cc::Hs => Cc::Ls,
        }
    }

    /// The comparison true where this one is false: of integers only, as a
    /// NaN fails both of a float pair.
    pub(crate) fn not(self) -> Cc {
        match self {
            Cc::Eq => Cc::Ne,
            Cc::Ne => Cc::Eq,
            Cc::Lt => Cc::Ge,
            Cc::Le => Cc::Gt,
            Cc::Gt => Cc::Le,
            Cc::Ge => Cc::Lt,
            Cc::Lo => Cc::Hs,
            Cc::Ls => Cc::Hi,
            Cc::Hi => Cc::Ls,
            Cc::Hs => Cc::Lo,
        }
    }

    /// Of 64-bit operands; 32-bit ones come sign-extended for the signed
    /// kinds and zero-extended for the unsigned.
    pub(crate) fn eval_i(self, a: i64, b: i64) -> bool {
        match self {
            Cc::Eq => a == b,
            Cc::Ne => a != b,
            Cc::Lt => a < b,
            Cc::Le => a <= b,
            Cc::Gt => a > b,
            Cc::Ge => a >= b,
            Cc::Lo => (a as u64) < (b as u64),
            Cc::Ls => (a as u64) <= (b as u64),
            Cc::Hi => (a as u64) > (b as u64),
            Cc::Hs => (a as u64) >= (b as u64),
        }
    }

    /// `None` for the unsigned kinds, which floats do not have.
    pub(crate) fn eval_f(self, a: f64, b: f64) -> Option<bool> {
        Some(match self {
            Cc::Eq => a == b,
            Cc::Ne => a != b,
            Cc::Lt => a < b,
            Cc::Le => a <= b,
            Cc::Gt => a > b,
            Cc::Ge => a >= b,
            Cc::Lo | Cc::Ls | Cc::Hi | Cc::Hs => return None,
        })
    }

    pub(crate) fn unsigned(self) -> bool {
        matches!(self, Cc::Lo | Cc::Ls | Cc::Hi | Cc::Hs)
    }

    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn name(self) -> &'static str {
        ["eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs"][self as usize]
    }
}

/// Why an exit leaves, for counting toward recompilation: only widenable kinds
/// count.
#[repr(u8)]
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
    /// A collector check: leaves through `Exit::Gc`, then resumes.
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
    /// A helper failed where the site's feedback never saw a metamethod run:
    /// the exit records one, and the recompile calls it.
    Mm,
    /// A prologue guard on an entry value: fails to the entry-fail stub, which
    /// records the entry's kinds.
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
    /// numbers raise too, and any other kind is learned, as `__lt` may order
    /// it.
    ArgCmp,
    /// A builtin's result was of a kind its inline code does not make, where
    /// the call's feedback never saw one (`floor` without an integer,
    /// `string.byte` out of range): the exit records `OTHER_RESULT`.
    OtherResult,
}

impl ExitTag {
    const ALL: [ExitTag; 15] = [
        ExitTag::Type,
        ExitTag::Overflow,
        ExitTag::NeverRan,
        ExitTag::Unsupported,
        ExitTag::Slow,
        ExitTag::Gc,
        ExitTag::Shape,
        ExitTag::Const,
        ExitTag::Bounds,
        ExitTag::Mm,
        ExitTag::Entry,
        ExitTag::Lazy,
        ExitTag::Arg,
        ExitTag::ArgCmp,
        ExitTag::OtherResult,
    ];

    pub(crate) fn of(bits: u16) -> ExitTag {
        ExitTag::ALL[bits as usize]
    }

    /// Whether an exit's feedback reads its site's operands (their kinds, a
    /// receiver's shape, a callee), which a moved exit's blame carries. The
    /// others' operands were guarded already or tell nothing.
    pub(crate) fn blames_operands(self) -> bool {
        matches!(
            self,
            ExitTag::Type
                | ExitTag::Shape
                | ExitTag::Const
                | ExitTag::Mm
                | ExitTag::NeverRan
                | ExitTag::Arg
                | ExitTag::ArgCmp
        )
    }

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

/// Whether an exit runs the instruction at its pc or continues after it.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ExitKind {
    Before,
    After,
}

/// Registers of a moved exit's blame entries: the value its guard tested and
/// the table it was read from.
pub(crate) const BLAME_TESTED: u8 = 255;
pub(crate) const BLAME_TABLE: u8 = 254;

/// Parts of a table's hash a no-keys test covers: integer keys,
/// dictionary-mode string keys, other keys.
pub(crate) const HASH_INTS: u8 = 1;
pub(crate) const HASH_STRS: u8 = 2;
pub(crate) const HASH_MISC: u8 = 4;

/// What a `Val` helper returns when the interpreter must run the
/// instruction: the negative quiet NaN, which no `Value` is (floats are
/// canonical).
pub(crate) const HELPER_FAIL_VAL: u64 = 0xFFF8_0000_0000_0000;

bitflags::bitflags! {
    /// Abstract heaps, each versioned separately. `Immutable` memory
    /// (prototype fields, string length and bytes) is none of them.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct Heaps: u16 {
        /// Memory nothing writes: no heap.
        const IMMUTABLE = 0;
        /// Frame slots, keyed by register.
        const SLOT = 1;
        /// Named table slots, keyed by their constant key.
        const FIELD = 2;
        const ARR_ITEMS = 4;
        /// An array part's size and pointer.
        const ARR_SIZE = 8;
        const SHAPE = 16;
        /// Class bits and metamethods.
        const META = 32;
        const UPVAL = 64;
        /// The frame header.
        const FRAME = 128;
        /// The collector's state.
        const GC = 256;
        const ALL = 0x1ff;
        /// The heap of the instruction's field (`aux`), not of the op.
        const BY_FIELD = 0x8000;

        const ARR = Self::ARR_ITEMS.bits() | Self::ARR_SIZE.bits();
        const LOOKUP = Self::SHAPE.bits() | Self::META.bits();
        const TABLE = Self::FIELD.bits() | Self::ARR.bits() | Self::LOOKUP.bits() | Self::GC.bits();
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct Flags: u8 {
        /// Takes a frame state.
        const EXITS = 1;
        const CALLS = 2;
        const ALLOCS = 4;
        const TERMINATOR = 8;
        /// `a` and `b` may swap.
        const COMMUTES = 16;
        /// Valid only where it stands: never moved, never numbered.
        const PINNED = 32;
        /// One bytecode meaning on boxed values, which may run a metamethod:
        /// kept until its operands' facts are final, never numbered.
        const GENERIC = 64;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Dialect {
    /// High-level: generic and unit ops, which copy 1 lowers.
    H,
    /// Low-level only.
    L,
    Both,
}

/// What an operand word holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arg {
    /// Unused.
    No,
    Imm,
    Val,
    I32,
    I64,
    F64,
    B1,
    Ptr,
    /// An integer of the instruction's width: `I32`, or any 64-bit word
    /// (`Rep::word64`). Every `Int` operand of an instruction has one width,
    /// its `Int` result's.
    Int,
    /// `Int`, or `B1` where the operands and the result all are.
    IntB,
    /// A 64-bit word.
    W64,
    /// A value of the result's representation.
    Same,
    /// A value of any representation.
    Any,
    /// A frame state.
    Fs,
    /// A frame state, or none.
    OptFs,
    /// A value of any representation, or none.
    Opt,
    /// A block.
    Blk,
    /// With `Len` in `b`: a run of values in the extra pool.
    Vals,
    /// With `Len` in `b`: a pc, then a run of (register, value) pairs in the
    /// extra pool.
    Entries,
    Len,
}

/// An op's result representation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Res {
    Is(Rep),
    /// `I32` or `I64`, the width of its `Int` operands; `B1` of `IntB` ones.
    Int,
    /// Given at emission and checked there per op.
    Dyn,
    /// That of operand `c`, the value an exit passes through when it does
    /// not leave; none without one.
    Pass,
}

/// What `aux` holds, for the printer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Aux {
    No,
    Cc,
    /// A type set.
    Set,
    /// A type set and, above it, an exit tag.
    SetTag,
    Tag,
    /// A frame slot.
    Slot,
    /// A field class.
    Field,
    /// An `ArithKind`.
    Kind,
    /// An `ExitKind`.
    Exit,
    /// A `Helper`.
    Helper,
}

pub(crate) struct OpInfo {
    pub(crate) name: &'static str,
    pub(crate) dialect: Dialect,
    pub(crate) aux: Aux,
    pub(crate) args: [Arg; 3],
    pub(crate) res: Res,
    pub(crate) reads: Heaps,
    pub(crate) writes: Heaps,
    pub(crate) flags: Flags,
}

macro_rules! res {
    (Int) => {
        Res::Int
    };
    (Dyn) => {
        Res::Dyn
    };
    (Pass) => {
        Res::Pass
    };
    ($r:ident) => {
        Res::Is(Rep::$r)
    };
}

macro_rules! ops {
    ($($name:ident $d:ident $aux:ident ($a:ident $b:ident $c:ident) $res:ident
        [$($r:ident)*] [$($w:ident)*] [$($f:ident)*];)*) => {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub(crate) enum Op { $($name),* }

        pub(crate) static OPS: [OpInfo; [$(Op::$name),*].len()] = [$(OpInfo {
            name: stringify!($name),
            dialect: Dialect::$d,
            aux: Aux::$aux,
            args: [Arg::$a, Arg::$b, Arg::$c],
            res: res!($res),
            reads: Heaps::from_bits_retain(0 $(| Heaps::$r.bits())*),
            writes: Heaps::from_bits_retain(0 $(| Heaps::$w.bits())*),
            flags: Flags::from_bits_retain(0 $(| Flags::$f.bits())*),
        }),*];
    };
}

// name, dialect, aux, (a b c), result, [reads] [writes] [flags]
ops! {
    // Constants. A `K32` is an `I32` or a `B1`; a `K64` (halves in `a`, `b`)
    // an `I64`, a `Val` with no heap object or a raw word; a `KObj` (a pool
    // index, its type in `aux`) a heap `Val` or its `Ptr`.
    K32         Both No     (Imm No No)       Dyn  [] [] [];
    K64         Both No     (Imm Imm No)      Dyn  [] [] [];
    KF64        Both No     (Imm Imm No)      F64  [] [] [];
    KObj        Both Set    (Imm No No)       Dyn  [] [] [];

    // Memory. A load's `Opt` operand is what makes it safe (a shape guard);
    // an indexed access takes the index its bounds guard returns. `c` of a
    // fixed access is the index of a parametrised field class.
    Load        Both Field  (Ptr Opt Imm)     Dyn  [BY_FIELD] [] [];
    Store       Both Field  (Ptr Any Imm)     None [] [BY_FIELD] [];
    LoadIdx     Both Field  (Ptr I64 Opt)     Dyn  [BY_FIELD] [] [];
    StoreIdx    Both Field  (Ptr I64 Any)     None [] [BY_FIELD] [];
    // A generic `for`'s steps over a table (its lock), each a declared
    // sequence. `ArrNext`: the index of the first array slot at or after
    // `b` that is not nil, or the array part's size. `NamedNext`: that of
    // the first such named slot, the metatable's skipped, or -1. `NamedKey`
    // and `NamedVal`: the key and the value of named slot `b`. `HashKeys`:
    // the key counts of the hash parts in `b` (`HASH_*`) or'd.
    ArrNext     Both No     (Ptr I32 No)      I32  [ARR_SIZE ARR_ITEMS] [] [];
    NamedNext   Both No     (Ptr I32 No)      I32  [SHAPE FIELD] [] [];
    NamedKey    Both No     (Ptr I32 No)      Val  [SHAPE] [] [];
    NamedVal    Both No     (Ptr I32 No)      Val  [SHAPE FIELD] [] [];
    HashKeys    Both No     (Ptr Imm No)      I64  [FIELD ARR_ITEMS] [] [];
    // The frame's base, which frame fields are loaded from.
    Base        Both No     (No No No)        Ptr  [] [] [];
    SlotLoad    Both Slot   (No No No)        Val  [SLOT] [] [];
    SlotStore   Both Slot   (Val No No)       None [] [SLOT] [];

    // Integers, 32 and 64 bit by the result. Shifts and divisions have the
    // machine's semantics; a division's operands are what its guards return.
    Add         Both No     (Int Int No)      Int  [] [] [COMMUTES];
    Sub         Both No     (Int Int No)      Int  [] [] [];
    Mul         Both No     (Int Int No)      Int  [] [] [COMMUTES];
    And         Both No     (IntB IntB No)    Int  [] [] [COMMUTES];
    Or          Both No     (IntB IntB No)    Int  [] [] [COMMUTES];
    Xor         Both No     (IntB IntB No)    Int  [] [] [COMMUTES];
    Not         Both No     (Int No No)       Int  [] [] [];
    Neg         Both No     (Int No No)       Int  [] [] [];
    Shl         Both No     (Int Int No)      Int  [] [] [];
    Shr         Both No     (Int Int No)      Int  [] [] [];
    Sar         Both No     (Int Int No)      Int  [] [] [];
    SDiv        Both No     (Int Int No)      Int  [] [] [];
    UDiv        Both No     (Int Int No)      Int  [] [] [];
    ZExt32      Both No     (I32 No No)       I64  [] [] [];
    SExt32      Both No     (I32 No No)       I64  [] [] [];
    Trunc32     Both No     (W64 No No)       I32  [] [] [];
    Cmp         Both Cc     (Int Int No)      B1   [] [] [];
    Select      Both No     (B1 Same Same)    Dyn  [] [] [];

    // Checked: the value, or an exit.
    AddOvf      Both Tag    (Int Int Fs)      Int  [] [] [EXITS COMMUTES];
    SubOvf      Both Tag    (Int Int Fs)      Int  [] [] [EXITS];
    MulOvf      Both Tag    (Int Int Fs)      Int  [] [] [EXITS COMMUTES];
    NegOvf      Both Tag    (Int Fs No)       Int  [] [] [EXITS];

    // Floats. `FToI` and `FToL` out of range are unspecified: a guard
    // follows.
    FAdd        Both No     (F64 F64 No)      F64  [] [] [COMMUTES];
    FSub        Both No     (F64 F64 No)      F64  [] [] [];
    FMul        Both No     (F64 F64 No)      F64  [] [] [COMMUTES];
    FDiv        Both No     (F64 F64 No)      F64  [] [] [];
    FNeg        Both No     (F64 No No)       F64  [] [] [];
    FAbs        Both No     (F64 No No)       F64  [] [] [];
    FSqrt       Both No     (F64 No No)       F64  [] [] [];
    FFloor      Both No     (F64 No No)       F64  [] [] [];
    FCeil       Both No     (F64 No No)       F64  [] [] [];
    FCmp        Both Cc     (F64 F64 No)      B1   [] [] [];
    IToF        Both No     (I32 No No)       F64  [] [] [];
    LToF        Both No     (I64 No No)       F64  [] [] [];
    FToI        Both No     (F64 No No)       I32  [] [] [];
    FToL        Both No     (F64 No No)       I64  [] [] [];

    // Boxing. Unboxing a small integer is `Trunc32`. An `UnboxPtr`'s operand
    // is typed by its definition; `UnboxPtrPin`'s by a branch, which pins it.
    BoxI32      Both No     (I32 No No)       Val  [] [] [];
    BoxF64      Both No     (F64 No No)       Val  [] [] [];
    BoxPtr      Both Set    (Ptr No No)       Val  [] [] [];
    UnboxF64    Both No     (Val No No)       F64  [] [] [];
    UnboxPtr    Both No     (Val No No)       Ptr  [] [] [];
    UnboxPtrPin Both No     (Val No No)       Ptr  [] [] [PINNED];

    // Exits. A guard returns operand `c` where it does not leave: a use of
    // the result depends on the guard.
    Guard       Both Tag    (B1 Fs Opt)       Pass [] [] [EXITS];
    GuardNot    Both Tag    (B1 Fs Opt)       Pass [] [] [EXITS];
    Deopt       Both Tag    (Fs No No)        None [] [] [EXITS TERMINATOR];
    LazyExit    Both No     (Fs No No)        None [] [] [EXITS PINNED];
    FrameState  Both Exit   (Entries Len OptFs) None [] [] [];

    // Control. A parameter's inputs are in its block's predecessor order.
    // `Ret` returns `b` results from `R[a..]`, stored already.
    Param       Both No     (Vals Len No)     Dyn  [] [] [PINNED];
    Jump        Both No     (Blk No No)       None [] [] [TERMINATOR];
    Br          Both No     (B1 Blk Blk)      None [] [] [TERMINATOR];
    Ret         Both No     (Imm Imm No)      None [SLOT] [] [TERMINATOR];
    // The Lua call protocol. The callee and its arguments are in its slots;
    // the state's values live on in the frame, and the collector clears the
    // slots it does not name. `c`, a byte each: where results land (the
    // `CALL`'s `a`), the argument count, the `CALL`'s `c`, and the header's
    // slot, which is the first but for a metamethod's call, staged past the
    // frame's registers. `Resume` is first in the call's successor: the
    // first result when it has a value, else the results in their slots.
    // Leave for the interpreter's handler of the instruction at pc `c`,
    // whose word is `a` (low) and `b`: what is live is in its slots.
    Dispatch    Both No     (Imm Imm Imm)     None [SLOT] [] [TERMINATOR];
    CallEntry   Both No     (Fs Blk Imm)      None [ALL] [ALL] [CALLS TERMINATOR];
    // Replace the frame by one of the Lua closure without varargs `a` (its
    // lock), whose `b` arguments are in `R[0..]`: nothing of this frame is
    // left to close. `FnFits`: whether such a closure's window fits the
    // stack from this frame's base.
    TailEnter   Both No     (Ptr Imm No)      None [SLOT] [] [TERMINATOR];
    FnFits      Both No     (Ptr No No)       B1   [FRAME] [] [];
    // The call that ends the block is of a Lua closure without varargs
    // whose prototype's address is pool word `c`: these are its first
    // arguments, passed in registers. `Arg`: parameter `a` of a function
    // region, which such a caller passes so and every other entry loads.
    CallArgs    Both No     (Vals Len Imm)    None [] [] [CALLS PINNED];
    Arg         Both No     (Imm No No)       Val  [] [] [PINNED];
    Resume      Both No     (Imm No No)       Dyn  [] [SLOT] [PINNED];

    // A C call of the helper in `aux` (`HELPERS`), its immediate in `c`.
    CallHelper  Both Helper (Vals Len Imm)    Dyn  [] [] [CALLS];
    // The same call from a cold block, of a helper that does not allocate:
    // it keeps every register but its result's.
    CallHelperCold Both Helper (Vals Len Imm) Dyn  [] [] [CALLS];

    // Unit ops: optimized as wholes, lowered by copy 1.
    CheckType   H    SetTag (Val Fs No)       Dyn  [] [] [EXITS];
    IsType      H    Set    (Val No No)       B1   [] [] [];
    BoxI64      H    No     (I64 No No)       Val  [] [] [ALLOCS];
    UnboxI64    H    No     (Val No No)       I64  [] [] [];
    ToF64       H    SetTag (Val Fs No)       F64  [] [] [EXITS];
    DivFloor    H    Tag    (Int Int Fs)      Int  [] [] [EXITS];
    ModFloor    H    Tag    (Int Int Fs)      Int  [] [] [EXITS];
    LuaShl      H    Tag    (Int Int OptFs)   Int  [] [] [];
    LuaShr      H    Tag    (Int Int OptFs)   Int  [] [] [];
    FToIExact   H    Tag    (F64 Fs No)       I32  [] [] [EXITS];
    FToLExact   H    Tag    (F64 Fs No)       I64  [] [] [EXITS];
    LToI        H    Tag    (I64 Fs No)       I32  [] [] [EXITS];
    AbsOvf      H    Tag    (Int Fs No)       Int  [] [] [EXITS];
    FIDiv       H    No     (F64 F64 No)      F64  [] [] [];
    FMod        H    No     (F64 F64 No)      F64  [] [] [];

    // Generic ops: the frame state's pc is the feedback site.
    Arith       H    Kind   (Val Val Fs)      Val  [ALL] [ALL] [EXITS CALLS GENERIC];
    Unm         H    No     (Val Fs No)       Val  [ALL] [ALL] [EXITS CALLS GENERIC];
    Compare     H    Cc     (Val Val Fs)      B1   [ALL] [ALL] [EXITS CALLS GENERIC];
    Truthy      H    No     (Val No No)       B1   [] [] [GENERIC];
    // Their result is no number to wait for: it is final under a pending key.
    GetIndex    H    No     (Val Val Fs)      Val  [ALL] [ALL] [EXITS CALLS];
    SetIndex    H    No     (Vals Len Fs)     None [ALL] [ALL] [EXITS CALLS];
}

impl Op {
    #[inline]
    pub(crate) fn info(self) -> &'static OpInfo {
        &OPS[self as usize]
    }

    #[inline]
    pub(crate) fn has(self, f: Flags) -> bool {
        self.info().flags.intersects(f)
    }

    /// Whether a dominating instruction with the same operands stands for
    /// this one. Memory readers join once the memory table versions their
    /// heaps.
    #[inline]
    pub(crate) fn numbered(self) -> bool {
        let i = self.info();
        const NO: Flags = Flags::CALLS
            .union(Flags::ALLOCS)
            .union(Flags::TERMINATOR)
            .union(Flags::PINNED)
            .union(Flags::GENERIC);
        !i.flags.intersects(NO) && i.reads.is_empty() && i.writes.is_empty()
    }
}

/// A pinned register a helper takes ahead of its operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Ctx {
    Rt,
    Thread,
    Base,
}

/// A helper's C signature and effects: the pinned registers, then an
/// immediate where its call has one, then the operands, each in the next
/// argument register of its class.
pub(crate) struct HelperInfo {
    pub(crate) name: &'static str,
    pub(crate) ctx: &'static [Ctx],
    /// Whether an immediate follows the pinned registers.
    pub(crate) imm: bool,
    pub(crate) args: &'static [Rep],
    pub(crate) ret: Rep,
    pub(crate) reads: Heaps,
    pub(crate) writes: Heaps,
    pub(crate) flags: Flags,
}

const NONE: &[Ctx] = &[];
const RT: &[Ctx] = &[Ctx::Rt];
const FRAME: &[Ctx] = &[Ctx::Rt, Ctx::Thread, Ctx::Base];
const BASE: &[Ctx] = &[Ctx::Rt, Ctx::Base];
const STACK: &[Ctx] = &[Ctx::Thread, Ctx::Base];
const THREAD: &[Ctx] = &[Ctx::Rt, Ctx::Thread];

macro_rules! helpers {
    ($($name:ident $ctx:ident $imm:literal ($($arg:ident)*) $ret:ident
        [$($r:ident)*] [$($w:ident)*] [$($f:ident)*];)*) => {
        /// Rust routines a region calls with the C ABI (`jit::helpers`).
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub(crate) enum Helper { $($name),* }

        pub(crate) static HELPERS: [HelperInfo; [$(Helper::$name),*].len()] = [$(HelperInfo {
            name: stringify!($name),
            ctx: $ctx,
            imm: $imm != 0,
            args: &[$(Rep::$arg),*],
            ret: Rep::$ret,
            reads: Heaps::from_bits_retain(0 $(| Heaps::$r.bits())*),
            writes: Heaps::from_bits_retain(0 $(| Heaps::$w.bits())*),
            flags: Flags::from_bits_retain(0 $(| Flags::$f.bits())*),
        }),*];
    };
}

// name, pinned registers, whether an immediate follows them, (operands),
// result, [reads] [writes] [flags].
// A `Val` result is `HELPER_FAIL_VAL` where the interpreter must run the
// instruction. The immediates are their callers': an `ArithKind`, registers,
// a prototype's index.
helpers! {
    // Lua's float `%`, and libm called directly.
    FMod         NONE   0 (F64 F64)     F64  [] [] [];
    Sin          NONE   0 (F64)         F64  [] [] [];
    Cos          NONE   0 (F64)         F64  [] [] [];
    Tan          NONE   0 (F64)         F64  [] [] [];
    Asin         NONE   0 (F64)         F64  [] [] [];
    Acos         NONE   0 (F64)         F64  [] [] [];
    Exp          NONE   0 (F64)         F64  [] [] [];
    Log          NONE   0 (F64)         F64  [] [] [];
    Log2         NONE   0 (F64)         F64  [] [] [];
    Log10        NONE   0 (F64)         F64  [] [] [];
    Pow          NONE   0 (F64 F64)     F64  [] [] [];
    Atan2        NONE   0 (F64 F64)     F64  [] [] [];
    LibmFmod     NONE   0 (F64 F64)     F64  [] [] [];
    // Arithmetic of numbers and numeric strings; integers out of i32 are
    // boxed.
    ArithCoerce  RT     1 (Val Val)     Val  [] [] [ALLOCS];
    Unm          RT     0 (Val)         Val  [] [] [ALLOCS];
    BNot         RT     0 (Val)         Val  [] [] [ALLOCS];
    // A metamethod lookup moves a table off a stale class or adopts a
    // metatable's class.
    Len          RT     0 (Val)         Val  [FIELD ARR LOOKUP] [LOOKUP] [ALLOCS];
    Concat       RT     0 (Val Val)     Val  [] [] [ALLOCS];
    UpvalBarrier THREAD 0 (Ptr)         None [] [GC] [];
    // The backward barrier of a table that is not gray.
    Barrier      RT     0 (Ptr)         None [] [GC] [];
    // New open upvalues join the thread's list.
    Closure      FRAME  1 ()            Val  [SLOT] [FRAME] [ALLOCS];
    ClosureOf    FRAME  1 (Ptr)         Val  [SLOT] [] [ALLOCS];
    Close        FRAME  1 ()            None [SLOT FRAME] [UPVAL FRAME] [];
    NewTable     RT     0 (I64)         Val  [] [] [ALLOCS];
    SetList      FRAME  1 ()            None [SLOT FRAME] [FIELD ARR GC] [ALLOCS];
    VarargGet    BASE   0 (Val)         Val  [] [] [ALLOCS];
    VarargTable  BASE   1 ()            Val  [] [SLOT] [ALLOCS];
    TforNext     BASE   1 ()            Val  [SLOT FIELD ARR SHAPE] [SLOT] [ALLOCS];
    TforIpairs   RT     0 (Val Val)     Val  [ARR FIELD META] [] [];
    TopArgs      STACK  1 ()            I32  [FRAME] [] [];
    VarargAll    STACK  1 ()            Val  [FRAME] [SLOT FRAME] [];
    Mm           RT     1 (Val Val)     Val  [LOOKUP] [LOOKUP] [ALLOCS];
    IndexFn      RT     0 (Val Val)     Val  [LOOKUP] [LOOKUP] [ALLOCS];
    NewIndexFn   RT     0 (Val Val)     Val  [LOOKUP] [LOOKUP] [ALLOCS];
    // Fails rather than refresh a stale class.
    Index        RT     0 (Val Val)     Val  [FIELD ARR LOOKUP] [] [];
    GetStr       NONE   0 (Val Val)     Val  [FIELD LOOKUP] [] [];
    // Stores that touch table memory alone: a metamethod fails them rather
    // than run, and an allocation never collects.
    SetStr       RT     0 (Val Val Val) Val  [TABLE] [TABLE] [ALLOCS];
    NewIndex     RT     1 (Val Val Val) Val  [TABLE] [TABLE] [ALLOCS];
    Type         RT     0 (Val)         Val  [] [] [ALLOCS];
    GetMetatable RT     0 (Val)         Val  [FIELD LOOKUP] [] [];
    SetMetatable RT     0 (Val Val)     Val  [TABLE] [TABLE] [ALLOCS];
    RawGet       NONE   0 (Val Val)     Val  [FIELD ARR SHAPE] [] [];
    RawSet       RT     0 (Val Val Val) Val  [TABLE] [TABLE] [ALLOCS];
    RawEqual     NONE   0 (Val Val)     I32  [] [] [];
    RawLen       RT     0 (Val)         Val  [FIELD ARR] [] [ALLOCS];
    ToString     RT     0 (Val)         Val  [LOOKUP] [LOOKUP] [ALLOCS];
    ToNumber     RT     0 (Val)         Val  [] [] [ALLOCS];
    StrSub       RT     0 (Ptr I32 I32) Val  [] [] [ALLOCS];
    StrByte      NONE   0 (Ptr I32)     Val  [] [] [];
    TInsert      RT     0 (Val Val)     Val  [TABLE] [TABLE] [ALLOCS];
    TRemove      RT     0 (Val)         Val  [TABLE] [TABLE] [ALLOCS];
}

impl Helper {
    #[inline]
    pub(crate) fn info(self) -> &'static HelperInfo {
        &HELPERS[self as usize]
    }

    pub(crate) fn of(aux: u16) -> Helper {
        assert!((aux as usize) < HELPERS.len());
        // SAFETY: a `repr(u8)` enum with a variant per row.
        unsafe { std::mem::transmute(aux as u8) }
    }
}
