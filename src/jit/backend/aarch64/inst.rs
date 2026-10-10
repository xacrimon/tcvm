//! The aarch64 machine instructions, before register allocation. Each
//! variant's operands (in its `VCode` operand pool, in the order its doc
//! gives) are allocated by regalloc2; x16 and x17 are scratch inside one
//! instruction, never allocated.

use crate::jit::backend::aarch64::asm::{Cond, Sz};
use crate::jit::backend::vcode::TargetInst;
use crate::jit::ir::ops::HelperId;
use crate::jit::ir::types::TypeSet;

/// The exit of an entry guard: the region's entry-fail stub (6.4).
pub(crate) const ENTRY_FAIL: u32 = u32::MAX - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AluOp {
    Add,
    Sub,
    And,
    Orr,
    Eor,
    Mul,
    Udiv,
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

/// A compare whose flags a branch, guard, select or `cset` consumes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Test {
    /// `cmp` of two w registers, or one against `imm`: operands `[a, b]` or `[a]`.
    I32 { imm: Option<i32> },
    /// `cmp` of two x registers (i64 or raw `Value` words), or one against `imm`.
    X { imm: Option<i64> },
    /// `fcmp` of two d registers: `[a, b]`.
    F64,
    /// A `Val` in a type set: `[v]`.
    Type(TypeSet),
    /// A `Val` is nil or false: `[v]`.
    Falsy,
    /// A table's shape is the one at this address: `[t]`.
    Shape(u64),
    /// A `B1` is true: `[c]`.
    B1,
}

#[derive(Clone, Debug)]
pub(crate) enum MInst {
    /// Open the native frame. The entry block's first instruction.
    Prologue,
    /// `[def d]`, int.
    MovImm(u64),
    /// `[def d]`, float.
    FImm(u64),
    /// `[def d]`: `R[r]`.
    LoadSlot(u8),
    /// `[use v]` (int): `R[r] = v`.
    StoreSlot(u8),
    /// `[use v]` (float): `R[r] = v` as a float `Value`.
    StoreSlotF(u8),
    /// `[def d]`: a by-value upvalue of the running closure.
    LoadUpval(u8),
    /// `[def d, use n, use m]`.
    Alu(AluOp, Sz),
    /// `[def d, use n]`.
    AluImm(AluOp, Sz, u64),
    /// `[def d, use n]`: `neg` or `mvn`.
    Neg(Sz),
    Mvn(Sz),
    /// `[def d, use n, use m, snap..]`: i32 add/sub exiting on overflow.
    AddOvf {
        sub: bool,
        exit: u32,
    },
    /// `[def d, use n, snap..]`.
    AddImmOvf {
        sub: bool,
        imm: u32,
        exit: u32,
    },
    /// `[def d, use n, use m, snap..]`: i32 multiply exiting on overflow.
    MulOvf {
        exit: u32,
    },
    /// `[def d, use n, snap..]`: i32 negate exiting on `MIN`.
    NegOvf {
        exit: u32,
    },
    /// `[def d, use n, use m, snap..]`: floor division or modulo, exiting on
    /// a zero divisor (unless `nonzero`) and, for i32 division, `MIN // -1`.
    DivMod {
        div: bool,
        sz: Sz,
        nonzero: bool,
        exit: u32,
    },
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
    /// `[def d, use n]`: `sxtw`.
    Sxtw,
    /// `[def d (float), use n (int)]`.
    Scvtf(Sz),
    /// `[def d, use n (float)]`: `fcvtzs`, saturating.
    Fcvtzs(Sz),
    /// `[def d (int), use n (float)]`: the bits.
    FmovToGpr,
    /// `[def d (float), use n (int)]`: the bits.
    FmovFromGpr,
    /// `[def d, use n, snap..]`: the i32 of an i64, exiting outside i32.
    LToI {
        exit: u32,
    },
    /// `[def d, use n (float), snap..]`: the i32 an f64 holds exactly.
    FToIExact {
        exit: u32,
    },
    /// `[def d (float), use v, snap..]`: the f64 of a small integer or
    /// float `Value`.
    ToF64 {
        exit: u32,
    },
    /// `[def d, use n]`: box an i32.
    BoxI32,
    /// `[def d, use n]`: box a b1.
    BoxB1,
    /// `[def d (x0), use n (x1), snap..]`: box an i64 (small inline, else
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
    /// `[def d (w), test operands..]`: `cset`.
    Set {
        test: Test,
        cond: Cond,
    },
    /// `[def d, use c (w), use a, use b]`: `c ? a : b`, int or float by `float`.
    Select {
        float: bool,
    },
    /// `[test operands.., snap..]`: exit unless `cond` holds after the test.
    Guard {
        test: Test,
        cond: Cond,
        exit: u32,
    },
    /// `[snap..]`: exit when the frame's header has any of the flags.
    GuardNoFlag {
        mask: u8,
        exit: u32,
    },
    /// `[def d]`: whether the frame's header has the flag.
    HasFlag(u8),
    /// `[def d, use f]`: whether the function is a Lua closure without varargs.
    FnFixedLua,
    /// `[def d, use f]`: whether the Lua closure's window fits from `base`.
    FnFits,
    /// `[def d, use c]`: the value of an upvalue cell.
    LoadCell,
    /// `[use c, use v]`: store into an upvalue cell.
    StoreCell,
    /// `[use t, snap..]`: exit unless the table's shape is `shape`.
    GuardShape {
        shape: u64,
        exit: u32,
    },
    /// `[def d]`: a class's live bits (`I32`) or `__index`.
    ClassBits(u64),
    ClassIndex(u64),
    /// `[def d, use t]` / `[use t, use v]`: a named slot.
    TabLoad(u32),
    TabStore(u32),
    /// `[def d, use t]`: the spill cell has room for the spilled `loc`.
    TabHasRoom(u32),
    /// `[use t]`: move the table to `shape`.
    TabSetShape(u64),
    /// `[def d, use t]`.
    TabASize,
    TabArray,
    /// `[def d, use arr, use i]` / `[use arr, use i, use v]`.
    ArrLoad,
    ArrStore,
    /// `[early def d, use t, use from, temp v]`.
    ArrNext,
    /// `[use t (x1)]`: the backward barrier unless gray; clobbers as a call.
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
    /// The end of a block: to succ 0.
    Jump,
    /// `[test operands..]`: to succ 0 when `cond` holds, else succ 1.
    Br {
        test: Test,
        cond: Cond,
    },
    /// Tail-out into `enter`; succ 0 is the resume block. `nargs` is
    /// `u8::MAX` for the count in the operand.
    Call {
        a: u8,
        nargs: u8,
        pc_after: usize,
    },
    /// `[use f, use args..]`: a metamethod call from a header at slot `hdr`
    /// (the frame's register count); succ 0 is the resume block.
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
    /// `[use f]`: the Lua closure takes over the frame (`Op::TailEnter`).
    TailEnter {
        nargs: u8,
    },
    /// `[snap..]`: exit unless the header at `hdr` and `n` arguments fit.
    GuardFits {
        end: u16,
        exit: u32,
    },
    /// A resume block's first instruction: `[def r0, def r1]` up to `wanted`
    /// (1 or 2), or the landing helper.
    Resume {
        wanted: u8,
        c: u8,
        a: u8,
    },
    /// Return `n` results from `R[a..]`.
    Return {
        a: u8,
        n: u8,
    },
    /// `[snap..]`.
    Deopt {
        exit: u32,
    },
}

impl MInst {
    /// The exit of an instruction that may take one before its effects,
    /// with the frame its snapshot describes.
    pub(crate) fn exit_id(&self) -> Option<u32> {
        use MInst::*;
        let e = match *self {
            AddOvf { exit, .. }
            | AddImmOvf { exit, .. }
            | MulOvf { exit }
            | NegOvf { exit }
            | DivMod { exit, .. }
            | ShiftI32 { exit, .. }
            | LToI { exit }
            | FToIExact { exit }
            | ToF64 { exit }
            | Guard { exit, .. }
            | GuardNoFlag { exit, .. }
            | GuardShape { exit, .. }
            | GcCheck { exit }
            | GuardFits { exit, .. } => exit,
            _ => return None,
        };
        (e != ENTRY_FAIL).then_some(e)
    }
}

impl TargetInst for MInst {
    fn is_branch(&self) -> bool {
        matches!(
            self,
            MInst::Jump | MInst::Br { .. } | MInst::Call { .. } | MInst::CallMm { .. }
        )
    }

    fn is_ret(&self) -> bool {
        matches!(
            self,
            MInst::Return { .. }
                | MInst::Deopt { .. }
                | MInst::Dispatch { .. }
                | MInst::TailEnter { .. }
        )
    }
}
