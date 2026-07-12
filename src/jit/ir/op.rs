//! Opcodes and their effect summaries.
//!
//! The op set deliberately spans levels: `LuaGetIndex` (full `__index` chain,
//! calls, may raise) and `SlotGet` (one load at a constant displacement) are
//! both members, and lowering rewrites the former into the latter in place.
//!
//! Effects are not decoration — GVN, load elimination, and LICM are unsound
//! without them. Memory is partitioned into alias classes rather than one
//! monolithic heap edge, because otherwise every table store would kill every
//! table load and the optimizer would have nothing to work with.

use bitflags::bitflags;

use crate::env::shape::MetamethodBits;
use crate::jit::ir::pool::{ConstRef, ProtoRef, ShapeRef, StrRef};
use crate::jit::ir::ty::TypeSet;

/// Arithmetic and bitwise operators, matching the bytecode's set. Used by the
/// generic `LuaArith` escape hatch before specialization picks an int or float
/// form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArithKind {
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IDiv,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
    Unm,
    BNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    /// Raises on a zero divisor — takes a `GuardCond` on the divisor. Mirrors
    /// `num::ArithOp::INT_ZERO_DIVISOR_RAISES`.
    IDiv,
    /// Same zero-divisor rule as `IDiv`.
    Mod,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
    Neg,
    BNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
    IDiv,
    Mod,
    Pow,
    Neg,
}

/// Comparison conditions. Float compares are *ordered*: NaN compares false,
/// which is Lua's rule.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cc {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

bitflags! {
    /// Alias classes. v1 disambiguates by class only, not by base value: two
    /// `TAB_PROPS` accesses on different tables are assumed to may-alias.
    /// Escape analysis will upgrade this to per-allocation disambiguation.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Mem: u16 {
        /// String-keyed slot storage.
        const TAB_PROPS = 1 << 0;
        /// The `properties` Vec's data pointer. Killed by any shape transition
        /// (which reallocates), *not* by an in-place slot store.
        const TAB_PROPS_PTR = 1 << 1;
        const TAB_ARRAY = 1 << 2;
        /// The array Vec's data pointer. Killed by array growth.
        const TAB_ARRAY_PTR = 1 << 3;
        const TAB_HASH = 1 << 4;
        /// A table's `shape` field.
        const SHAPE = 1 << 5;
        const UPVAL = 1 << 6;
        /// The thread's value stack. Open upvalues alias this, which is why a
        /// captured local cannot live purely in SSA.
        const STACK = 1 << 7;
    }
}

impl Mem {
    pub const NONE: Self = Self::empty();
    pub const ALL: Self = Self::all();
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Flags: u8 {
        /// May allocate or otherwise let the collector run. Every live boxed
        /// value must be anchored across one of these — see `Effects`.
        const MAY_GC = 1 << 0;
        const MAY_RAISE = 1 << 1;
        /// Transfers control to arbitrary Lua or native code.
        const MAY_CALL = 1 << 2;
        /// May suspend the thread. Since compiled code cannot suspend a native
        /// frame, these deopt on suspend.
        const MAY_YIELD = 1 << 3;
        const TERMINATOR = 1 << 4;
        /// Cannot be removed even if its results are unused.
        const SIDE_EFFECT = 1 << 5;
    }
}

/// An op's full effect summary.
///
/// # The rooting rule
///
/// At any op with `MAY_GC`, every live value with rep `Val` or `Ptr` must be
/// *anchored*: either it is a Lua register named by the op's `FrameState` (and
/// so spilled to its canonical stack slot, where `ThreadState::trace` finds
/// it), or it is derived from a value that is. Derived pointers are sound to
/// leave unrooted only because the collector does not move objects. Anything
/// else spills to the traced JIT spill area. Lowering enforces this; the
/// verifier checks it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Effects {
    pub flags: Flags,
    pub reads: Mem,
    pub writes: Mem,
}

impl Effects {
    const PURE: Self = Effects {
        flags: Flags::empty(),
        reads: Mem::NONE,
        writes: Mem::NONE,
    };

    const fn reads(mem: Mem) -> Self {
        Effects {
            flags: Flags::empty(),
            reads: mem,
            writes: Mem::NONE,
        }
    }

    const fn writes(mem: Mem) -> Self {
        Effects {
            flags: Flags::SIDE_EFFECT,
            reads: Mem::NONE,
            writes: mem,
        }
    }

    /// Full clobber: an unbounded call into Lua or native code.
    const fn opaque() -> Self {
        Effects {
            flags: Flags::MAY_GC
                .union(Flags::MAY_RAISE)
                .union(Flags::MAY_CALL)
                .union(Flags::MAY_YIELD)
                .union(Flags::SIDE_EFFECT),
            reads: Mem::ALL,
            writes: Mem::ALL,
        }
    }

    pub fn is_pure(self) -> bool {
        self.flags.is_empty() && self.writes.is_empty() && self.reads.is_empty()
    }

    /// Safe to hoist out of a loop or move across unrelated code, given no
    /// aliasing conflict. Guards are *not* movable by this alone — hoisting a
    /// guard is only legal if it is known to execute on every iteration.
    pub fn is_movable(self) -> bool {
        !self
            .flags
            .intersects(Flags::SIDE_EFFECT | Flags::MAY_CALL | Flags::TERMINATOR)
    }
}

/// An IR opcode. Immediates ride the enum; SSA operands are held separately in
/// `InstData::args`, so `Op` stays `Copy + Hash` and can key GVN directly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    // --- constants and representation ------------------------------------
    KConst(ConstRef),
    IConst(i64),
    /// Held as bits so the op stays `Hash + Eq`.
    FConst(u64),
    BConst(bool),

    // Representation changes only — *not* boxing in the JS sense. A `Value` is
    // a 16-byte tagged pair, so packing an i64 is a tag write plus a payload
    // write: no allocation, no GC, no rooting. Hence `Effects::PURE`.
    PackInt,
    PackFloat,
    PackBool,
    /// Requires the operand's set ⊑ INT. Not a check — a guard must precede it.
    UnpackInt,
    UnpackFloat,
    /// Requires set ⊑ HEAP. The payload is already a pointer; this just drops
    /// the tag.
    UnpackPtr,
    /// The raw tag byte, for lowering type tests.
    TagOf,
    IsType(TypeSet),
    IsFalsy,

    // --- guards and assumptions -------------------------------------------
    /// Exit unless the value's type is in the set. Produces a refined value.
    GuardType(TypeSet),
    /// Exit unless the table's shape pointer matches. Produces a value refined
    /// to `Tab<S>`. Self-invalidating: any layout change mints a new shape.
    GuardShape(ShapeRef),
    /// Exit unless the condition holds. Bounds checks, zero divisors.
    GuardCond,
    /// A watchpoint, not a check: emits no code, records a dependency on the
    /// metatable behind this shape. Invalidation is observed at the next
    /// safepoint. `MtCacheData::bits` mutates in place, so no shape guard can
    /// substitute for this.
    AssumeNoMm(ShapeRef, MetamethodBits),

    // --- generic Lua ops (the escape hatch when types are unknown) ---------
    LuaArith(ArithKind),
    LuaCmp(Cc),
    LuaEq,
    LuaConcat,
    LuaLen,
    LuaGetIndex,
    LuaSetIndex,

    // --- specialized arithmetic -------------------------------------------
    IntArith(IntOp),
    FloatArith(FloatOp),
    ICmp(Cc),
    FCmp(Cc),
    SiToFp,
    /// Float to integer, exact or exit. Backs Lua's implicit float→int coercion
    /// in bitwise ops (`num::exact_float_to_int`).
    FpToIntExact,

    // --- tables ------------------------------------------------------------
    TabNew {
        array_hint: u32,
    },
    /// The `properties` Vec data pointer for a shape-guarded table.
    TabProps,
    /// Load a property at a constant slot. Legal only under a `GuardShape` that
    /// proves the slot's existence.
    SlotGet(u32),
    /// Store a property at a constant slot. Legal only for a key *already
    /// present* in the guarded shape — a new key transitions the shape and must
    /// go through `LuaSetIndex`.
    SlotSet(u32),
    TabArr,
    TabArrLen,
    /// Bounds must be guarded separately.
    ArrGet,
    ArrSet,
    /// Falls back to a runtime helper; no GC, no metamethods (raw).
    TabHashGet,

    GcBarrierBack,
    GcBarrierFwd,

    // --- upvalues and stack-pinned registers -------------------------------
    /// `closure->upvalues[idx]`. The upvalue array is immutable per closure.
    UpvalCell(u8),
    UpvalGet,
    UpvalSet,
    UpvalClose(u8),
    /// Read a stack-pinned register (one captured by a `CLOSURE` in this
    /// prototype). Open upvalues address slots by index, so such a register
    /// cannot live in SSA.
    StackGet(u8),
    StackSet(u8),

    // --- calls -------------------------------------------------------------
    /// Materializes the frame, transfers control, and yields `nret` results.
    /// Args are `[callee, arg0, ..]`.
    ///
    /// The suspend/error epilogue is *not* in the IR: a callee that yields
    /// cannot suspend our native frame, so lowering emits the status check
    /// after the call and exits through this op's `FrameState` (with `pc` past
    /// the `CALL`, so the interpreter resumes as if mid-call). Modelling that
    /// as an IR-level branch would put a diamond around every call site and buy
    /// the optimizer nothing — the check is a fixed epilogue.
    Call {
        nret: u8,
    },
    ClosureNew(ProtoRef),

    // --- global access -----------------------------------------------------
    /// `_ENV[k]` before specialization. Lowers to a shape guard on the globals
    /// table plus a `SlotGet`.
    GetGlobal(StrRef),
    SetGlobal(StrRef),

    // --- control -----------------------------------------------------------
    Jump,
    Br,
    Ret,
    /// Unconditional exit to the interpreter. Used for cold paths we chose not
    /// to compile.
    Deopt,
    /// GC poll *and* invalidation poll. Required at loop back-edges and after
    /// every call — a call can invalidate our own assumptions.
    Safepoint,
}

impl Op {
    pub fn effects(self) -> Effects {
        use Op::*;
        match self {
            KConst(_) | IConst(_) | FConst(_) | BConst(_) | PackInt | PackFloat | PackBool
            | UnpackInt | UnpackFloat | UnpackPtr | TagOf | IsType(_) | IsFalsy | IntArith(_)
            | FloatArith(_) | ICmp(_) | FCmp(_) | SiToFp => Effects::PURE,

            // Guards are pure computations with an exit edge: removable when
            // subsumed by a dominating guard, but not freely reorderable across
            // the code they protect.
            GuardType(_) | GuardShape(_) | GuardCond | FpToIntExact => Effects {
                flags: Flags::empty(),
                reads: Mem::NONE,
                writes: Mem::NONE,
            },

            // Emits no code; the dependency is recorded at compile time.
            AssumeNoMm(..) => Effects::PURE,

            LuaArith(_)
            | LuaCmp(_)
            | LuaEq
            | LuaConcat
            | LuaLen
            | LuaGetIndex
            | LuaSetIndex
            | Call { .. } => Effects::opaque(),

            TabNew { .. } | ClosureNew(_) => Effects {
                flags: Flags::MAY_GC.union(Flags::SIDE_EFFECT),
                reads: Mem::NONE,
                writes: Mem::NONE,
            },

            TabProps => Effects::reads(Mem::TAB_PROPS_PTR),
            SlotGet(_) => Effects::reads(Mem::TAB_PROPS),
            SlotSet(_) => Effects::writes(Mem::TAB_PROPS),
            TabArr => Effects::reads(Mem::TAB_ARRAY_PTR),
            TabArrLen => Effects::reads(Mem::TAB_ARRAY),
            ArrGet => Effects::reads(Mem::TAB_ARRAY),
            ArrSet => Effects::writes(Mem::TAB_ARRAY),
            TabHashGet => Effects::reads(Mem::TAB_HASH),

            GcBarrierBack | GcBarrierFwd => Effects {
                flags: Flags::SIDE_EFFECT,
                reads: Mem::NONE,
                writes: Mem::NONE,
            },

            UpvalCell(_) => Effects::PURE,
            UpvalGet => Effects::reads(Mem::UPVAL.union(Mem::STACK)),
            UpvalSet => Effects::writes(Mem::UPVAL.union(Mem::STACK)),
            UpvalClose(_) => Effects {
                flags: Flags::MAY_GC.union(Flags::SIDE_EFFECT),
                reads: Mem::NONE,
                writes: Mem::UPVAL.union(Mem::STACK),
            },
            StackGet(_) => Effects::reads(Mem::STACK),
            StackSet(_) => Effects::writes(Mem::STACK),

            // Pre-lowering forms of a globals-table access.
            GetGlobal(_) => Effects::opaque(),
            SetGlobal(_) => Effects::opaque(),

            Jump | Br | Ret | Deopt => Effects {
                flags: Flags::TERMINATOR.union(Flags::SIDE_EFFECT),
                reads: Mem::NONE,
                writes: Mem::NONE,
            },

            Safepoint => Effects {
                flags: Flags::MAY_GC.union(Flags::SIDE_EFFECT),
                reads: Mem::NONE,
                writes: Mem::NONE,
            },
        }
    }

    pub fn is_terminator(self) -> bool {
        self.effects().flags.contains(Flags::TERMINATOR)
    }

    /// Guards produce a refined copy of their first operand and carry an exit.
    pub fn is_guard(self) -> bool {
        matches!(self, Op::GuardType(_) | Op::GuardShape(_) | Op::GuardCond)
    }

    /// Ops that need a `FrameState`: anything that can exit to the interpreter,
    /// raise, or let the collector run.
    pub fn needs_frame_state(self) -> bool {
        let e = self.effects();
        self.is_guard()
            || matches!(self, Op::Deopt | Op::Safepoint)
            || e.flags
                .intersects(Flags::MAY_GC | Flags::MAY_RAISE | Flags::MAY_CALL)
    }
}
