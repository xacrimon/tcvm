//! The x86-64 machine instructions, before register allocation. Each
//! variant's operands (in its `VCode` operand pool, in the order its doc
//! gives) are allocated by `ra`; r10, r11 and xmm15 are scratch inside one
//! instruction, never allocated. A definition marked `= n` reuses that
//! input's register (`OperandConstraint::Reuse`), as the two-address
//! instructions need. A value word a load or store moves may be in an xmm
//! register: a float's, which lowering keeps there (`Lower::in_d`).
//!
//! An `I32` or `B1` in a register has no defined upper half: an i32 unboxed
//! from a value shares the value's register.

use crate::jit::backend::lower::Known;
use crate::jit::backend::vcode::{Flow, TargetInst};
use crate::jit::backend::x86_64::asm::{Cond, Shift, Sz};
use crate::jit::ir::ops::{Cc, HelperId};
use crate::jit::ir::types::TypeSet;

/// The exit of an entry guard: the region's entry-fail stub.
pub(crate) const ENTRY_FAIL: u32 = u32::MAX - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AluOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Mul,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FUnOp {
    Neg,
    Abs,
    Sqrt,
    Floor,
    Ceil,
}

/// How a float compare sets the flags its condition reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FCmp {
    /// `vucomisd a, b`: `A` and `Ae` are false when unordered.
    Ord,
    /// `vucomisd b, a`, for `<` and `<=`.
    Rev,
    /// An equality mask tested: `Ne` where equal and ordered.
    Eq,
}

/// A compare whose flags a branch, guard, select or `setcc` consumes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Test {
    /// `cmp` of two 32-bit registers, or one against `imm`: operands
    /// `[a, b]` or `[a]`.
    I32 { imm: Option<i32> },
    /// `cmp` of two 64-bit registers (i64 or raw `Value` words), or one
    /// against `imm`.
    X { imm: Option<i32> },
    /// A compare of two xmm registers, or one against zero: `[a, b]` or
    /// `[a]`.
    F64 { zero: bool, how: FCmp },
    /// A `Val` in a type set: `[v]`.
    Type(TypeSet),
    /// A table's shape is the one at this address: `[t]`.
    Shape(u64),
    /// A table's shape is one of `Lowered::shape_sets[i]`: `[t]`.
    ShapeIn(u32),
    /// A `B1` is true: `[c]`.
    B1,
    /// Bit `k` of a 32-bit register is set (`Ne`) or clear (`E`): `[x]`.
    Bit(u8),
    /// Bit `k` of the table's class's metamethod bits is set (`Ne`) or
    /// clear (`E`), clear without a class: `[t]`.
    MmBit(u8),
    /// The frame's header has the flag (`Ne`): `[]`.
    HasFlag(u8),
    /// Numbers have a metatable (`Ne`): `[]`.
    NumMtSet,
    /// A type's metatable is the one with that word (`E`): `[]`.
    TypeMtIs { slot: u8, word: u64 },
    /// The function is a Lua closure without varargs (`E`): `[f]`.
    FnFixedLua,
    /// The function is a native (`A`): `[f]`.
    FnNative,
    /// The Lua closure's window fits from `base` (`Be`): `[f]`.
    FnFits,
    /// The table's spill cell has room for the spilled `loc` (`A`): `[t]`.
    TabHasRoom(u32),
    /// The table's hash parts in the mask hold no key (`E`): `[t]`.
    TabNoKeys(u8),
}

#[derive(Clone, Debug)]
pub(crate) enum MInst {
    /// Open the native frame. The entry block's first instruction.
    Prologue,
    /// `[def p0, def p1, ..]`, fixed in `abi::ARGS`: a function region's
    /// parameters, as every entry passes them. No code.
    Params,
    /// `[def d]`, int.
    MovImm(u64),
    /// `[def d]`, float.
    FImm(u64),
    /// `[def d]`: `R[r]`.
    LoadSlot(u8),
    /// `[use v]`: `R[r] = v`.
    StoreSlot(u8),
    /// `[def d]`: the frame's closure, from its header (no register holds it),
    /// for its upvalues.
    FrameClosure,
    /// `[def d, use c]`: a by-value upvalue of the closure `c`.
    LoadUpvalOf(u8),
    /// `[def d, use n, use m]`: `lea`, or `add` into either operand.
    Add(Sz),
    /// `[def d, use n]`: `n + imm`, by `lea` unless in place.
    AddImm(Sz, i32),
    /// `[def d = n, use n, use m]`; never `Add`.
    Alu(AluOp, Sz),
    /// `[def d, use n]`: `and`, `or` or `xor` with a constant, a pooled
    /// literal where it is no `imm32`.
    AluImm(AluOp, Sz, u64),
    /// `[def d = n, use n]`: a shift by a constant.
    ShiftImm(Shift, Sz, u8),
    /// `[def d, use n, use m]`: a shift by `m` mod the register size.
    ShiftReg(Shift, Sz),
    /// `[def d, use n, use m]`: the 64-bit product of two u32s, shifted
    /// right by the amount.
    UMulShr(u8),
    /// `[def d = n, use n]`: `neg` or `not`.
    Neg(Sz),
    Not(Sz),
    /// `[def d = n, use n, use m, snap..]`: i32 add/sub exiting on overflow.
    AddOvf {
        sub: bool,
        exit: u32,
    },
    /// `[def d = n, use n, snap..]`.
    AddImmOvf {
        imm: i32,
        exit: u32,
    },
    /// `[def d = n, use n, use m, snap..]`: i32 multiply exiting on overflow.
    MulOvf {
        exit: u32,
    },
    /// `[def d = n, use n, snap..]`: i32 negate exiting on `MIN`.
    NegOvf {
        exit: u32,
    },
    /// `[def d, use n, snap..]`: i32 `abs` exiting on `MIN`.
    AbsOvf {
        exit: u32,
    },
    /// `[def d, use n]`: i64 `abs`, wrapping.
    Abs,
    /// `[def d (rax or rdx), use n (rax), use m (late), snap..]`: floor
    /// division (the quotient, in rax) or modulo (in rdx), clobbering the
    /// other of the two; exits on a zero divisor (unless `nonzero`) and, for
    /// i32 division, `MIN // -1`. A divisor of -1 (unless not `minus1`)
    /// goes around `idiv`, which traps on `MIN / -1`.
    DivMod {
        div: bool,
        sz: Sz,
        nonzero: bool,
        minus1: bool,
        exit: u32,
    },
    /// `[def d (rdx), use n (rax), use m (late), snap..]`: C's `%`,
    /// clobbering rax; exits on a zero divisor unless `nonzero`.
    Rem {
        sz: Sz,
        nonzero: bool,
        exit: u32,
    },
    /// `[def d (rax), use n (rax), use m (late)]`: unsigned 64-bit division
    /// by a nonzero divisor, clobbering rdx.
    UDiv,
    /// `[def d, use n, use m, snap..]`: Lua's shift of an i32 (sign-extended),
    /// exiting when the result leaves i32.
    ShiftI32 {
        left: bool,
        exit: u32,
    },
    /// `[def d, use n, use m]`: Lua's 64-bit shift.
    ShiftI64 {
        left: bool,
    },
    /// `[def d, use n]`: `movsxd`.
    Sext,
    /// `[def d (float), use n (int)]`.
    IntToF(Sz),
    /// `[def d, use n (float)]`: toward zero, saturating, NaN to 0.
    FToInt,
    /// `[def d (int), use n (float)]`: the bits.
    FmovToGpr,
    /// `[def d (float), use n (int)]`: the bits.
    FmovFromGpr,
    /// `[use n, snap..]`: the i32 of an i64, exiting outside i32.
    LToI {
        exit: u32,
    },
    /// `[def d, use n (float), snap..]`: the i32 an f64 holds exactly.
    FToIExact {
        exit: u32,
    },
    /// `[def d, use n (float), snap..]`: the i64 an f64 holds exactly.
    FToLExact {
        exit: u32,
    },
    /// `[def d, use v, temp t (float), snap..]`: `ToPos` of a `Value` of
    /// `kinds`, a small integer inline and the others out of line (a string
    /// through `jit_str_to_i64`'s trampoline).
    ToPos {
        exit: u32,
        kinds: TypeSet,
    },
    /// `[def d (early), use a, use b, temp n, temp k, temp lb]`: whether
    /// string payload `a` orders before `b` (`le`: not after) by bytes.
    StrCmp {
        le: bool,
    },
    /// `[def d (float), use v, snap..]`: the f64 of a `Value` of `kinds`, a
    /// float inline and the others out of line (a string through
    /// `jit_str_to_f64`'s trampoline).
    ToF64 {
        exit: u32,
        kinds: TypeSet,
    },
    /// `[def d = n, use n]`: box an i32.
    BoxI32,
    /// `[def d, use n]`: box a b1.
    BoxB1,
    /// `[def d (rax), use n (rsi), snap..]`: box an i64 (small inline, else
    /// the helper), then the collector check.
    BoxI64 {
        exit: u32,
    },
    /// `[def d, use v]`: the i64 of a small or boxed integer.
    UnboxI64,
    /// `[def d (float), use n, use m]`.
    FAlu(FOp),
    /// `[def d (float), use n]`.
    FUn(FUnOp),
    /// `[def d (early), temp q, use a, use b]`: Lua's float `%`, the cases the
    /// floor of the rounded quotient misses out of line.
    FMod,
    /// `[def d, test operands..]`: `setcc`.
    Set {
        test: Test,
        cond: Cond,
    },
    /// `[def d = b, use a, use b, test operands..]`: `test ? a : b`, int or
    /// float by `float`.
    Select {
        test: Test,
        cond: Cond,
        float: bool,
    },
    /// `[test operands.., snap..]`: exit unless `cond` holds after the test.
    Guard {
        test: Test,
        cond: Cond,
        exit: u32,
    },
    /// `[use v, snap..]`: a self-call result's type guard, right after its
    /// `Resume`: out of line where the ways back but the fast return land
    /// (`opt::selfcall`), else as a `Guard`.
    GuardLanding {
        set: TypeSet,
        exit: u32,
    },
    /// `[def d, use t]`: the metamethod of a bit index from the table's class.
    ClassMm(u8),
    /// `[def d, use i, use f, temp t (float)]`: `IFCmp`.
    IFCmp {
        cc: Cc,
    },
    /// `[def d, use x, use y, temp fx, temp fy (float), temp ix, temp iy]`:
    /// `NumCmp` of operands whose kinds are `kx`, `ky` (numbers).
    NumCmp {
        cc: Cc,
        kx: TypeSet,
        ky: TypeSet,
    },
    /// `[def d, use s]`: a string's length.
    StrLenOf,
    /// `[def d, use s, use i, snap..]`: the byte of string `s` at Lua
    /// position `i`, exiting out of range.
    StrByteAt {
        exit: u32,
    },
    /// `[def d, use c]`: the value of an upvalue cell.
    LoadCell,
    /// `[use c, use v]`: store into an upvalue cell.
    StoreCell,
    /// `[use f, snap..]`: exit unless the value is a Lua closure of the
    /// prototype at `proto` (its box); `fun` when it is known a function.
    GuardProto {
        proto: u64,
        fun: bool,
        exit: u32,
    },
    /// `[def d]`: a class's live bits (`I32`) or `__index`.
    ClassBits(u64),
    ClassIndex(u64),
    /// `[def d, use t]` / `[use t, use v]`: a named slot.
    TabLoad(u32),
    TabStore(u32),
    /// `[use t]`: move the table to `shape`.
    TabSetShape(u64),
    /// `[def d, use t]`.
    TabASize,
    TabArray,
    /// `[def d, use arr, use i]` / `[use arr, use i, use v]`.
    ArrLoad,
    ArrStore,
    /// `[def d, use arr]` / `[use arr, use v]`: at a constant index.
    ArrLoadAt(u16),
    ArrStoreAt(u16),
    /// `[early def d, use t, use from, temp v]`.
    ArrNext,
    /// `[early def d, use t, use from, temp n, temp cap, temp mt]`.
    NamedNext,
    /// `[def d, use t, use i]`.
    NamedVal,
    NamedKey,
    /// `[use t]`: the backward barrier unless gray; its slow path keeps
    /// every register.
    Barrier,
    /// `[def d]`: the frame's `nv`.
    NumVarargs,
    /// `[def d, use nv]`: extra `k`, or nil when `nv <= k`.
    VarargAt(u8),
    /// `[def d, use nv, use k]`: extra `k - 1`, or nil out of range.
    VarargIdx,
    /// `[snap..]`.
    GcCheck {
        exit: u32,
    },
    /// A C-ABI helper with fixed-register operands.
    Helper(HelperId),
    /// The same in a cold arm, through its trampoline: it clobbers only the
    /// pinned and immediate registers the call writes, and its result's.
    HelperCold(HelperId),
    /// The end of a block: to succ 0.
    Jump,
    /// `[test operands..]`: to succ 0 when `cond` holds, else succ 1.
    Br {
        test: Test,
        cond: Cond,
    },
    /// `[use count?, site..]`: a call, fast or through `enter`; succ 0 is
    /// the resume block. `nargs` is `u8::MAX` for the count in the operand;
    /// the site's values follow in stack slots (`LSite`).
    Call {
        a: u8,
        nargs: u8,
        pc_after: usize,
        /// A constant callee the call writes word 0 for itself.
        known: Option<Known>,
        /// A self-call whose arguments pass the entry guards: it enters past
        /// them.
        typed: bool,
    },
    /// `[use f, use args.., site..]`: a metamethod call from a header at
    /// slot `hdr` (the frame's register count); succ 0 is the resume block.
    CallMm {
        hdr: u16,
        nargs: u8,
        pc_after: usize,
    },
    /// Leave for the handler of `op` with `insn` its word and `pc` the
    /// address after it.
    Dispatch {
        op: u8,
        word: u64,
        pc_after: usize,
    },
    /// `[use f]`: the Lua closure takes over the frame (`Op::TailEnter`);
    /// `known` its prototype (its box) and parameter count when a guard
    /// proved them.
    TailEnter {
        nargs: u8,
        known: Option<(u64, u8)>,
    },
    /// `[use n]`: `top` past slot `at` and `n` values (`Op::SetTop`).
    SetTop {
        at: u8,
    },
    /// A resume block's first instruction, where a fast callee's `ret`
    /// lands: `[def rax, def rdx]` up to `wanted` (1 or 2), or the landing
    /// helper.
    Resume {
        wanted: u8,
        c: u8,
        a: u8,
        /// The call's header slot: the callee's base is `hdr + 4` past ours.
        hdr: u16,
    },
    /// Return `n` results from `R[a..]`, the first (or nil) also in rax: `[use
    /// rax]` when `reg`, else it is read from its slot.
    Return {
        a: u8,
        n: u8,
        reg: bool,
    },
    /// `[snap..]`.
    Deopt {
        exit: u32,
    },
    /// `[snap..]`: a patch point; `Region::invalidate` writes a jump to
    /// the exit there.
    LazyExit {
        exit: u32,
    },
}

impl MInst {
    /// The exit of an instruction that may take one before its effects,
    /// with the frame its snapshot describes (`TCVM_JIT_DEOPT_ALL`).
    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn exit_id(&self) -> Option<u32> {
        use MInst::*;
        let e = match *self {
            AddOvf { exit, .. }
            | AddImmOvf { exit, .. }
            | MulOvf { exit }
            | NegOvf { exit }
            | AbsOvf { exit }
            | DivMod { exit, .. }
            | ShiftI32 { exit, .. }
            | LToI { exit }
            | FToIExact { exit }
            | FToLExact { exit }
            | Rem { exit, .. }
            | StrByteAt { exit }
            | ToPos { exit, .. }
            | ToF64 { exit, .. }
            | Guard { exit, .. }
            | GuardLanding { exit, .. }
            | GuardProto { exit, .. }
            | GcCheck { exit }
            | LazyExit { exit } => exit,
            _ => return None,
        };
        (e != ENTRY_FAIL).then_some(e)
    }

    /// A `Resume` that reads at most the first result, in its register:
    /// not the result count or the values behind it.
    pub(crate) fn plain_resume(&self) -> bool {
        matches!(
            self,
            MInst::Resume { wanted: 1, .. }
                | MInst::Resume {
                    wanted: 0,
                    c: 1,
                    ..
                }
        )
    }
}

impl TargetInst for MInst {
    fn flow(&self) -> Flow {
        match *self {
            MInst::Jump => Flow::Jump,
            MInst::Br { .. } => Flow::Branch,
            MInst::Deopt { exit } => Flow::Deopt(exit),
            MInst::Resume { .. } => Flow::Resume {
                plain: self.plain_resume(),
            },
            _ => Flow::Other,
        }
    }

    fn is_remat(&self) -> bool {
        matches!(
            self,
            MInst::MovImm(_) | MInst::FImm(_) | MInst::AluImm(..) | MInst::LoadSlot(_)
        )
    }

    #[cfg(feature = "tcvm-dev")]
    fn is_branch(&self) -> bool {
        matches!(
            self,
            MInst::Jump | MInst::Br { .. } | MInst::Call { .. } | MInst::CallMm { .. }
        )
    }
}
