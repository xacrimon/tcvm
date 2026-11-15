//! The op table: one row per op gives its dialect, operand form, result,
//! effects and flags. The enum, the static table and the printer's names
//! come from the rows, so an op without a row does not compile. Helpers have
//! a table of their own.

use crate::jit::ir::types::{Rep, TypeSet};

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
    pub(crate) struct Flags: u16 {
        /// Takes a frame state.
        const EXITS = 1;
        const CALLS = 2;
        const ALLOCS = 4;
        const TERMINATOR = 8;
        /// `a` and `b` may swap.
        const COMMUTES = 16;
        /// Valid only where it stands: never moved, never numbered.
        const PINNED = 32;
        /// A copy emits it again operand by operand; an op without this has
        /// an arm of its own in the copy.
        const PLAIN = 128;
        /// Runs for what it does, though it has no result, writes no heap
        /// and may not leave: kept when nothing uses it.
        const KEPT = 256;
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
    /// `I32` or `B1`: a 32-bit constant's.
    K32,
    /// Any 64-bit word's: a 64-bit constant's.
    Word,
    /// `Val` or `Ptr`: a heap object's word, or its pointer.
    Obj,
    /// Its field's.
    Field,
    /// Any, but some: a parameter's, a select's.
    Some,
    /// Its helper's, whose row the operands are checked against too.
    Helper,
    /// What its type set unboxes to.
    Unboxed,
    /// That of operand `c`, the value an exit passes through when it does
    /// not leave; none without one.
    Pass,
}

/// What `aux` holds, for the printer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Aux {
    No,
    /// A `Cc`; `FCc` one of those a float compare has.
    Cc,
    FCc,
    /// A type set.
    Set,
    /// A type set and, above it, an exit tag.
    SetTag,
    /// A type set and `SET_ARM`.
    SetArm,
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
    /// `access::ARRAY`, or 0.
    Array,
}

/// The bit of a `Proved`'s `aux` that says it stands in its branch's false
/// arm.
pub(crate) const SET_ARM: u16 = 1 << TypeSet::BITS;

/// How the fact of an op's `Val` or `Ptr` result is found; any other
/// result's is its representation's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FactOf {
    /// Nothing beyond its representation.
    Rep,
    /// A fixed type set.
    Set(TypeSet),
    /// The type set in `aux`.
    AuxSet,
    /// Its field's set.
    Field,
    /// Operand `c`'s fact; `MeetC` that met with the set in `aux`.
    PassC,
    MeetC,
    /// What holds of both operand `b` and operand `c`.
    JoinBC,
    /// Operand `a`'s set, of the heap types.
    HeapA,
    /// The constant's own type.
    Konst,
    /// Its helper's (`HelperInfo::fact`).
    Helper,
}

/// An op as a memory access.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mem {
    No,
    SlotLoad,
    SlotStore,
    Load,
    Store,
    LoadIdx,
    StoreIdx,
}

pub(crate) struct OpInfo {
    pub(crate) name: &'static str,
    pub(crate) dialect: Dialect,
    pub(crate) aux: Aux,
    pub(crate) args: [Arg; 3],
    pub(crate) res: Res,
    pub(crate) fact: FactOf,
    pub(crate) mem: Mem,
    pub(crate) reads: Heaps,
    pub(crate) writes: Heaps,
    pub(crate) flags: Flags,
}

macro_rules! res {
    (Int) => {
        Res::Int
    };
    (K32) => {
        Res::K32
    };
    (Word) => {
        Res::Word
    };
    (Obj) => {
        Res::Obj
    };
    (Field) => {
        Res::Field
    };
    (Some) => {
        Res::Some
    };
    (Helper) => {
        Res::Helper
    };
    (Unboxed) => {
        Res::Unboxed
    };
    (Pass) => {
        Res::Pass
    };
    ($r:ident) => {
        Res::Is(Rep::$r)
    };
}

macro_rules! fact {
    (Rep) => {
        FactOf::Rep
    };
    (AuxSet) => {
        FactOf::AuxSet
    };
    (Field) => {
        FactOf::Field
    };
    (PassC) => {
        FactOf::PassC
    };
    (MeetC) => {
        FactOf::MeetC
    };
    (JoinBC) => {
        FactOf::JoinBC
    };
    (HeapA) => {
        FactOf::HeapA
    };
    (Konst) => {
        FactOf::Konst
    };
    (Helper) => {
        FactOf::Helper
    };
    ($set:ident) => {
        FactOf::Set(TypeSet::$set)
    };
}

macro_rules! ops {
    ($($name:ident $d:ident $aux:ident ($a:ident $b:ident $c:ident) $res:ident
        $fact:ident $mem:ident [$($r:ident)*] [$($w:ident)*] [$($f:ident)*];)*) => {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub(crate) enum Op { $($name),* }

        pub(crate) static OPS: [OpInfo; [$(Op::$name),*].len()] = [$(OpInfo {
            name: stringify!($name),
            dialect: Dialect::$d,
            aux: Aux::$aux,
            args: [Arg::$a, Arg::$b, Arg::$c],
            res: res!($res),
            fact: fact!($fact),
            mem: Mem::$mem,
            reads: Heaps::from_bits_retain(0 $(| Heaps::$r.bits())*),
            writes: Heaps::from_bits_retain(0 $(| Heaps::$w.bits())*),
            flags: Flags::from_bits_retain(0 $(| Flags::$f.bits())*),
        }),*];
    };
}

// name, dialect, aux, (a b c), result, fact, memory, [reads] [writes] [flags]
ops! {
    // Constants. A `K32` is an `I32` or a `B1`; a `K64` (halves in `a`, `b`)
    // an `I64`, a `Val` with no heap object or a raw word; a `KObj` (a pool
    // index, its type in `aux`) a heap `Val` or its `Ptr`.
    K32         Both No     (Imm No No)       K32      Rep    No        [] [] [PLAIN];
    K64         Both No     (Imm Imm No)      Word     Konst  No        [] [] [PLAIN];
    KF64        Both No     (Imm Imm No)      F64      Rep    No        [] [] [PLAIN];
    KObj        Both Set    (Imm No No)       Obj      AuxSet No        [] [] [PLAIN];

    // Memory. A load's `Opt` operand is what makes it safe (a shape guard);
    // an indexed access takes the index its bounds guard returns. `c` of a
    // fixed access is the index of a parametrised field class.
    Load        Both Field  (Ptr Opt Imm)     Field    Field  Load      [BY_FIELD] [] [PLAIN];
    Store       Both Field  (Ptr Any Imm)     None     Rep    Store     [] [BY_FIELD] [PLAIN];
    LoadIdx     Both Field  (Ptr I64 Opt)     Field    Field  LoadIdx   [BY_FIELD] [] [PLAIN];
    StoreIdx    Both Field  (Ptr I64 Any)     None     Rep    StoreIdx  [] [BY_FIELD] [PLAIN];
    // The frame's base, which frame fields are loaded from.
    Base        Both No     (No No No)        Ptr      Rep    No        [] [] [PLAIN];
    Thread      Both No     (No No No)        Ptr      Rep    No        [] [] [PLAIN];
    SlotLoad    Both Slot   (No No No)        Val      Rep    SlotLoad  [SLOT] [] [PLAIN];
    SlotStore   Both Slot   (Val No No)       None     Rep    SlotStore [] [SLOT] [PLAIN];

    // Integers, 32 and 64 bit by the result. Shifts and divisions have the
    // machine's semantics; a division's operands are what its guards return.
    Add         Both No     (Int Int No)      Int      Rep    No        [] [] [COMMUTES PLAIN];
    Sub         Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    Mul         Both No     (Int Int No)      Int      Rep    No        [] [] [COMMUTES PLAIN];
    And         Both No     (IntB IntB No)    Int      Rep    No        [] [] [COMMUTES PLAIN];
    Or          Both No     (IntB IntB No)    Int      Rep    No        [] [] [COMMUTES PLAIN];
    Xor         Both No     (IntB IntB No)    Int      Rep    No        [] [] [COMMUTES PLAIN];
    Not         Both No     (Int No No)       Int      Rep    No        [] [] [PLAIN];
    Neg         Both No     (Int No No)       Int      Rep    No        [] [] [PLAIN];
    Shl         Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    Shr         Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    Sar         Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    SDiv        Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    UDiv        Both No     (Int Int No)      Int      Rep    No        [] [] [PLAIN];
    ZExt32      Both No     (I32 No No)       I64      Rep    No        [] [] [PLAIN];
    SExt32      Both No     (I32 No No)       I64      Rep    No        [] [] [PLAIN];
    Trunc32     Both No     (W64 No No)       I32      Rep    No        [] [] [PLAIN];
    Cmp         Both Cc     (Int Int No)      B1       Rep    No        [] [] [PLAIN];
    Select      Both No     (B1 Same Same)    Some     JoinBC No        [] [] [PLAIN];

    // Checked: the value, or an exit.
    AddOvf      Both Tag    (Int Int Fs)      Int      Rep    No        [] [] [EXITS COMMUTES PLAIN];
    SubOvf      Both Tag    (Int Int Fs)      Int      Rep    No        [] [] [EXITS PLAIN];
    MulOvf      Both Tag    (Int Int Fs)      Int      Rep    No        [] [] [EXITS COMMUTES PLAIN];
    NegOvf      Both Tag    (Int Fs No)       Int      Rep    No        [] [] [EXITS PLAIN];

    // Floats. `FToI` and `FToL` out of range are unspecified: a guard
    // follows.
    FAdd        Both No     (F64 F64 No)      F64      Rep    No        [] [] [COMMUTES PLAIN];
    FSub        Both No     (F64 F64 No)      F64      Rep    No        [] [] [PLAIN];
    FMul        Both No     (F64 F64 No)      F64      Rep    No        [] [] [COMMUTES PLAIN];
    FDiv        Both No     (F64 F64 No)      F64      Rep    No        [] [] [PLAIN];
    FNeg        Both No     (F64 No No)       F64      Rep    No        [] [] [PLAIN];
    FAbs        Both No     (F64 No No)       F64      Rep    No        [] [] [PLAIN];
    FSqrt       Both No     (F64 No No)       F64      Rep    No        [] [] [PLAIN];
    FFloor      Both No     (F64 No No)       F64      Rep    No        [] [] [PLAIN];
    FCeil       Both No     (F64 No No)       F64      Rep    No        [] [] [PLAIN];
    FCmp        Both FCc    (F64 F64 No)      B1       Rep    No        [] [] [PLAIN];
    IToF        Both No     (I32 No No)       F64      Rep    No        [] [] [PLAIN];
    LToF        Both No     (I64 No No)       F64      Rep    No        [] [] [PLAIN];
    FToI        Both No     (F64 No No)       I32      Rep    No        [] [] [PLAIN];
    FToL        Both No     (F64 No No)       I64      Rep    No        [] [] [PLAIN];

    // Boxing. Unboxing a small integer is `Trunc32`. An `UnboxPtr`'s operand
    // is typed by its definition (a check's, a guard's or a `Proved`'s result).
    BoxI32      Both No     (I32 No No)       Val      SMALL  No        [] [] [PLAIN];
    BoxF64      Both No     (F64 No No)       Val      FLOAT  No        [] [] [PLAIN];
    BoxPtr      Both Set    (Ptr No No)       Val      AuxSet No        [] [] [PLAIN];
    UnboxF64    Both No     (Val No No)       F64      Rep    No        [] [] [PLAIN];
    UnboxPtr    Both No     (Val No No)       Ptr      HeapA  No        [] [] [PLAIN];

    // Exits. A guard returns operand `c` where it does not leave: a use of
    // the result depends on the guard.
    Guard       Both Tag    (B1 Fs Opt)       Pass     PassC  No        [] [] [EXITS PLAIN];
    GuardNot    Both Tag    (B1 Fs Opt)       Pass     PassC  No        [] [] [EXITS PLAIN];
    // What a branch proved, in the arm it leads to: operand `c`, known to
    // be in the set in `aux` there (`SET_ARM`: its high bit for the false
    // arm). No code: a use of the result depends on the branch.
    Proved      Both SetArm (B1 No Any)       Pass     MeetC  No        [] [] [PLAIN];
    Deopt       Both Tag    (Fs No No)        None     Rep    No        [] [] [EXITS TERMINATOR PLAIN];
    LazyExit    Both No     (Fs No No)        None     Rep    No        [] [] [EXITS PINNED PLAIN];
    FrameState  Both Exit   (Entries Len OptFs) None     Rep    No        [] [] [];

    // Control. A parameter's inputs are in its block's predecessor order.
    // `Ret` returns `b` results from `R[a..]`, stored already.
    Param       Both No     (Vals Len No)     Some     Rep    No        [] [] [PINNED];
    Jump        Both No     (Blk No No)       None     Rep    No        [] [] [TERMINATOR];
    Br          Both No     (B1 Blk Blk)      None     Rep    No        [] [] [TERMINATOR];
    Ret         Both No     (Imm Imm No)      None     Rep    No        [SLOT] [] [TERMINATOR PLAIN];

    // A C call of the helper in `aux` (`HELPERS`): its context values, its
    // immediate where it has one, then its operands (`Emitter::call`).
    CallHelper  Both Helper (Vals Len No)     Helper   Helper No        [] [] [CALLS];

    // Unit ops: optimized as wholes, lowered by copy 1.
    CheckType   H    SetTag (Val Fs No)       Unboxed  AuxSet No        [] [] [EXITS PLAIN];
    IsType      H    Set    (Val No No)       B1       Rep    No        [] [] [PLAIN];
    BoxI64      H    No     (I64 No No)       Val      INT    No        [] [] [ALLOCS PLAIN];
    UnboxI64    H    No     (Val No No)       I64      Rep    No        [] [] [PLAIN];
    ToF64       H    SetTag (Val Fs No)       F64      Rep    No        [] [] [EXITS PLAIN];
    DivFloor    H    Tag    (Int Int Fs)      Int      Rep    No        [] [] [EXITS PLAIN];
    ModFloor    H    Tag    (Int Int Fs)      Int      Rep    No        [] [] [EXITS PLAIN];
    LuaShl      H    Tag    (Int Int OptFs)   Int      Rep    No        [] [] [PLAIN];
    LuaShr      H    Tag    (Int Int OptFs)   Int      Rep    No        [] [] [PLAIN];
    FToIExact   H    Tag    (F64 Fs No)       I32      Rep    No        [] [] [EXITS PLAIN];
    FToLExact   H    Tag    (F64 Fs No)       I64      Rep    No        [] [] [EXITS PLAIN];
    LToI        H    Tag    (I64 Fs No)       I32      Rep    No        [] [] [EXITS PLAIN];
    AbsOvf      H    Tag    (Int Fs No)       Int      Rep    No        [] [] [EXITS PLAIN];
    FIDiv       H    No     (F64 F64 No)      F64      Rep    No        [] [] [PLAIN];
    FMod        H    No     (F64 F64 No)      F64      Rep    No        [] [] [PLAIN];
    // A store's barrier on table `a`: nothing where one stood with no
    // collector step since (`Emitter::barrier` drops it for a value that is
    // no heap object).
    Barrier     H    No     (Ptr No No)       None     Rep    No        [GC] [] [KEPT PLAIN];
    // A table from the template whose address is `a`.
    NewTable    H    No     (I64 No No)       Val      TAB    No        [] [] [ALLOCS PLAIN];
    // Leave, for the interpreter to collect, past the collector's
    // threshold: nothing where one stood with no allocation since.
    GcCheck     H    Tag    (Fs No No)        None     Rep    No        [GC] [] [EXITS PLAIN];

    // Generic ops: the frame state's pc is the feedback site.
    Arith       H    Kind   (Val Val Fs)      Val      Rep    No        [ALL] [ALL] [EXITS CALLS GENERIC PLAIN];
    Unm         H    No     (Val Fs No)       Val      Rep    No        [ALL] [ALL] [EXITS CALLS GENERIC PLAIN];
    Compare     H    Cc     (Val Val Fs)      B1       Rep    No        [ALL] [ALL] [EXITS CALLS GENERIC PLAIN];
    Truthy      H    No     (Val No No)       B1       Rep    No        [] [] [GENERIC PLAIN];
    // Their result is no number to wait for: it is final under a pending key.
    GetIndex    H    Array  (Val Val Fs)      Val      Rep    No        [ALL] [ALL] [EXITS CALLS PLAIN];
    SetIndex    H    Array  (Vals Len Fs)     None     Rep    No        [ALL] [ALL] [EXITS CALLS];
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
        !i.flags.intersects(NO) && i.writes.is_empty()
    }
}

/// What a helper takes ahead of its operands: the runtime's address, the
/// thread, the frame's base. Each is an operand of its call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Ctx {
    Rt,
    Thread,
    Base,
}

/// A helper's C signature and effects: the context values, then an
/// immediate where its call has one, then the operands, each in the next
/// argument register of its class.
pub(crate) struct HelperInfo {
    pub(crate) name: &'static str,
    pub(crate) ctx: &'static [Ctx],
    /// Whether an immediate follows the context values.
    pub(crate) imm: bool,
    pub(crate) args: &'static [Rep],
    pub(crate) ret: Rep,
    /// What a boxed result is known to be: every type where nothing.
    pub(crate) fact: TypeSet,
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
    ($($name:ident $ctx:ident $imm:literal ($($arg:ident)*) $ret:ident $fact:ident
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
            fact: TypeSet::$fact,
            reads: Heaps::from_bits_retain(0 $(| Heaps::$r.bits())*),
            writes: Heaps::from_bits_retain(0 $(| Heaps::$w.bits())*),
            flags: Flags::from_bits_retain(0 $(| Flags::$f.bits())*),
        }),*];
    };
}

// name, context, whether an immediate follows it, (operands), result, its
// fact, [reads] [writes] [flags].
// A `Val` result is `HELPER_FAIL_VAL` where the interpreter must run the
// instruction. The immediates are their callers': an `ArithKind`, registers,
// a prototype's index.
helpers! {
    // Lua's float `%`, and libm called directly.
    FMod         NONE   0 (F64 F64)     F64  ANY [] [] [];
    Sin          NONE   0 (F64)         F64  ANY [] [] [];
    Cos          NONE   0 (F64)         F64  ANY [] [] [];
    Tan          NONE   0 (F64)         F64  ANY [] [] [];
    Asin         NONE   0 (F64)         F64  ANY [] [] [];
    Acos         NONE   0 (F64)         F64  ANY [] [] [];
    Exp          NONE   0 (F64)         F64  ANY [] [] [];
    Log          NONE   0 (F64)         F64  ANY [] [] [];
    Log2         NONE   0 (F64)         F64  ANY [] [] [];
    Log10        NONE   0 (F64)         F64  ANY [] [] [];
    Pow          NONE   0 (F64 F64)     F64  ANY [] [] [];
    Atan2        NONE   0 (F64 F64)     F64  ANY [] [] [];
    LibmFmod     NONE   0 (F64 F64)     F64  ANY [] [] [];
    // Arithmetic of numbers and numeric strings; integers out of i32 are
    // boxed.
    ArithCoerce  RT     1 (Val Val)     Val  ANY [] [] [ALLOCS];
    Unm          RT     0 (Val)         Val  ANY [] [] [ALLOCS];
    BNot         RT     0 (Val)         Val  ANY [] [] [ALLOCS];
    // A metamethod lookup moves a table off a stale class or adopts a
    // metatable's class.
    Len          RT     0 (Val)         Val  ANY [FIELD ARR LOOKUP] [LOOKUP] [ALLOCS];
    Concat       RT     0 (Val Val)     Val  ANY [] [] [ALLOCS];
    UpvalBarrier THREAD 0 (Ptr)         None ANY [] [GC] [];
    // The backward barrier of a table that is not gray.
    Barrier      RT     0 (Ptr)         None ANY [] [GC] [];
    // New open upvalues join the thread's list.
    Closure      FRAME  1 ()            Val  ANY [SLOT] [FRAME] [ALLOCS];
    ClosureOf    FRAME  1 (Ptr)         Val  ANY [SLOT] [] [ALLOCS];
    Close        FRAME  1 ()            None ANY [SLOT FRAME] [UPVAL FRAME] [];
    // An integer outside 32 bits, boxed.
    BoxInt       RT     0 (I64)         Val  INT [] [] [ALLOCS];
    NewTable     RT     0 (I64)         Val  TAB [] [] [ALLOCS];
    SetList      FRAME  1 ()            None ANY [SLOT FRAME] [FIELD ARR GC] [ALLOCS];
    VarargGet    BASE   0 (Val)         Val  ANY [] [] [ALLOCS];
    VarargTable  BASE   1 ()            Val  ANY [] [SLOT] [ALLOCS];
    TforNext     BASE   1 ()            Val  ANY [SLOT FIELD ARR SHAPE] [SLOT] [ALLOCS];
    TforIpairs   RT     0 (Val Val)     Val  ANY [ARR FIELD META] [] [];
    TopArgs      STACK  1 ()            I32  ANY [FRAME] [] [];
    VarargAll    STACK  1 ()            Val  ANY [FRAME] [SLOT FRAME] [];
    Mm           RT     1 (Val Val)     Val  ANY [LOOKUP] [LOOKUP] [ALLOCS];
    IndexFn      RT     0 (Val Val)     Val  ANY [LOOKUP] [LOOKUP] [ALLOCS];
    NewIndexFn   RT     0 (Val Val)     Val  ANY [LOOKUP] [LOOKUP] [ALLOCS];
    // Fails rather than refresh a stale class.
    Index        RT     0 (Val Val)     Val  ANY [FIELD ARR LOOKUP] [] [];
    GetStr       NONE   0 (Val Val)     Val  ANY [FIELD LOOKUP] [] [];
    // Stores that touch table memory alone: a metamethod fails them rather
    // than run, and an allocation never collects.
    SetStr       RT     0 (Val Val Val) Val  ANY [TABLE] [TABLE] [ALLOCS];
    NewIndex     RT     1 (Val Val Val) Val  ANY [TABLE] [TABLE] [ALLOCS];
    Type         RT     0 (Val)         Val  ANY [] [] [ALLOCS];
    GetMetatable RT     0 (Val)         Val  ANY [FIELD LOOKUP] [] [];
    SetMetatable RT     0 (Val Val)     Val  ANY [TABLE] [TABLE] [ALLOCS];
    RawGet       NONE   0 (Val Val)     Val  ANY [FIELD ARR SHAPE] [] [];
    RawSet       RT     0 (Val Val Val) Val  ANY [TABLE] [TABLE] [ALLOCS];
    RawEqual     NONE   0 (Val Val)     I32  ANY [] [] [];
    RawLen       RT     0 (Val)         Val  ANY [FIELD ARR] [] [ALLOCS];
    ToString     RT     0 (Val)         Val  ANY [LOOKUP] [LOOKUP] [ALLOCS];
    ToNumber     RT     0 (Val)         Val  ANY [] [] [ALLOCS];
    StrSub       RT     0 (Ptr I32 I32) Val  ANY [] [] [ALLOCS];
    StrByte      NONE   0 (Ptr I32)     Val  ANY [] [] [];
    TInsert      RT     0 (Val Val)     Val  ANY [TABLE] [TABLE] [ALLOCS];
    TRemove      RT     0 (Val)         Val  ANY [TABLE] [TABLE] [ALLOCS];
}

impl HelperInfo {
    /// The representations of a call's whole operand list.
    pub(crate) fn operands(&self) -> impl Iterator<Item = Rep> + '_ {
        let ctx = self.ctx.iter().map(|c| match c {
            Ctx::Rt => Rep::I64,
            Ctx::Thread | Ctx::Base => Rep::Ptr,
        });
        (ctx.chain(self.imm.then_some(Rep::I64))).chain(self.args.iter().copied())
    }
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
