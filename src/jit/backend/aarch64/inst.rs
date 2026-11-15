//! aarch64 machine opcodes and their table. One `MInst` is one machine
//! instruction, or one of the pseudo-instructions the table marks. A row
//! is an opcode's whole convention: what each slot of its record holds, its
//! list, what it clobbers, how control leaves it.

use crate::jit::backend::aarch64::asm;
use crate::jit::backend::reg::RegClass;
use crate::jit::backend::vcode::{Clobbers, Constraint, Flow, List, Row, Slot};
use crate::jit::bug;

/// A record's `sz`: an integer operation's width (`W`, `X`), or the bytes a
/// load or store moves.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sz {
    B = 1,
    H = 2,
    W = 4,
    X = 8,
}

impl Sz {
    pub(crate) fn of(sz: u8) -> Sz {
        match sz {
            1 => Sz::B,
            2 => Sz::H,
            4 => Sz::W,
            8 => Sz::X,
            _ => bug!("a size of {sz} bytes"),
        }
    }

    /// An integer operation's width.
    pub(crate) fn asm(self) -> asm::Sz {
        match self {
            Sz::W => asm::Sz::W,
            Sz::X => asm::Sz::X,
            Sz::B | Sz::H => bug!("an operation on less than 32 bits"),
        }
    }

    pub(crate) fn bits(self) -> u32 {
        self as u32 * 8
    }
}

const fn reg(class: RegClass, def: bool) -> Slot {
    Slot::Reg {
        class,
        def,
        late: def,
        constraint: Constraint::Any,
    }
}

// An integer register defined, used; a float register defined, used.
const D: Slot = reg(RegClass::Int, true);
const U: Slot = reg(RegClass::Int, false);
const FD: Slot = reg(RegClass::Float, true);
const FU: Slot = reg(RegClass::Float, false);
const IMM: Slot = Slot::Imm;
const WIDE: Slot = Slot::Wide;
const START: Slot = Slot::Start;
const LEN: Slot = Slot::Len;
const NO: Slot = Slot::None;

macro_rules! mops {
    ($($name:ident [$a:ident $b:ident $c:ident] $list:ident $clobbers:ident $flow:ident
        $remat:literal $pseudo:literal;)*) => {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum MOp { $($name),* }

        pub(crate) static ROWS: [Row; [$(MOp::$name),*].len()] = [$(Row {
            name: stringify!($name),
            slots: [$a, $b, $c],
            list: List::$list,
            clobbers: Clobbers::$clobbers,
            flow: Flow::$flow,
            remat: $remat,
            pseudo: $pseudo,
        }),*];
    };
}

// name [slots] list clobbers flow remat pseudo
mops! {
    // Opens the native frame.
    Entry     [NO NO NO]       None None Fall false true;
    // Constants: one that fits 32 bits zero-extended, and one in the pool.
    MovImm    [D IMM NO]       None None Fall true false;
    MovWide   [D WIDE NO]      None None Fall true false;
    FImm      [FD WIDE NO]     None None Fall true false;
    // A frame slot, by its index.
    LdrSlot   [D IMM NO]       None None Fall true false;
    StrSlot   [U IMM NO]       None None Fall false false;
    // The immediate is the byte offset from the base, the last register;
    // `sz` is the bytes moved, and a load of fewer than 8 zero-extends.
    Ldr       [D U IMM]        None None Fall false false;
    Str       [U U IMM]        None None Fall false false;
    // The base, then an index of 8-byte elements: `cc` 1 for a 32-bit index
    // zero-extended (`uxtw`).
    LdrIdx    [D U U]          None None Fall false false;
    StrIdx    [U U U]          None None Fall false false;
    // A pinned register's value.
    MovBase   [D NO NO]        None None Fall true false;
    MovThread [D NO NO]        None None Fall true false;
    Uxtw      [D U NO]         None None Fall false false;
    Cset      [D NO NO]        None None Fall false false;
    // The first register where `cc` holds, else the second.
    Csel      [D U U]          None None Fall false false;
    AndRR     [D U U]          None None Fall false false;
    EorRR     [D U U]          None None Fall false false;
    SdivRR    [D U U]          None None Fall false false;
    Sxtw      [D U NO]         None None Fall false false;
    NegR      [D U NO]         None None Fall false false;
    // `mvn`: every bit inverted.
    MvnR      [D U NO]         None None Fall false false;
    // Flag-setting, as `AddsRR`.
    NegsR     [D U NO]         None None Fall false false;
    FNeg      [FD FU NO]       None None Fall false false;
    // Round toward negative infinity.
    Frintm    [FD FU NO]       None None Fall false false;
    OrrRR     [D U U]          None None Fall false false;
    // A mask, as its value: one that fits 32 bits zero-extended, and one in
    // the pool.
    AndImm    [D U IMM]        None None Fall false false;
    AndWide   [D U WIDE]       None None Fall false false;
    OrrImm    [D U IMM]        None None Fall false false;
    OrrWide   [D U WIDE]       None None Fall false false;
    TstImm    [U IMM NO]       None None Fall false false;
    TstWide   [U WIDE NO]      None None Fall false false;
    AddRR     [D U U]          None None Fall false false;
    SubRR     [D U U]          None None Fall false false;
    MulRR     [D U U]          None None Fall false false;
    AddImm    [D U IMM]        None None Fall false false;
    SubImm    [D U IMM]        None None Fall false false;
    // Flag-setting: the branch that reads the flags follows, with only the
    // allocator's moves between.
    AddsRR    [D U U]          None None Fall false false;
    SubsRR    [D U U]          None None Fall false false;
    AddsImm   [D U IMM]        None None Fall false false;
    SubsImm   [D U IMM]        None None Fall false false;
    CmpRR     [U U NO]         None None Fall false false;
    CmpImm    [U IMM NO]       None None Fall false false;
    CmnImm    [U IMM NO]       None None Fall false false;
    // `cmp xn, wm, sxtw`: whether a product fits 32 bits.
    CmpSxtw   [U U NO]         None None Fall false false;
    Fcmp      [FU FU NO]       None None Fall false false;
    // A shift by a register's low bits.
    LslRR     [D U U]          None None Fall false false;
    LsrRR     [D U U]          None None Fall false false;
    AsrRR     [D U U]          None None Fall false false;
    LslImm    [D U IMM]        None None Fall false false;
    LsrImm    [D U IMM]        None None Fall false false;
    AsrImm    [D U IMM]        None None Fall false false;
    Smull     [D U U]          None None Fall false false;
    Scvtf     [FD U NO]        None None Fall false false;
    // A float to an integer of `sz`, toward zero.
    Fcvtzs    [D FU NO]        None None Fall false false;
    FmovToGpr [D FU NO]        None None Fall false false;
    FmovFromGpr [FD U NO]      None None Fall false false;
    FAdd      [FD FU FU]       None None Fall false false;
    FSub      [FD FU FU]       None None Fall false false;
    FMul      [FD FU FU]       None None Fall false false;
    FDiv      [FD FU FU]       None None Fall false false;
    B         [NO NO NO]       None None Jump false false;
    // To successor 0 where `cc` holds, else successor 1.
    BCond     [NO NO NO]       None None Branch false false;
    // To the exit in the immediate where `cc` holds; `BExit` always. The
    // list is their frame state's values.
    BCondExit [IMM START LEN]  AnyLoc None CondExit false false;
    BExit     [IMM START LEN]  AnyLoc None Exit false false;
    // A C call of the helper in the immediate: the list is its result, if
    // used, and its operands.
    Call      [IMM START LEN]  CArgs CallerSaved Fall false false;
    // Return `imm >> 8` results from `R[imm & 0xff..]`.
    Ret       [IMM NO NO]      None None Fall false true;
}

impl MOp {
    pub(crate) fn of(op: u8) -> MOp {
        assert!((op as usize) < ROWS.len());
        // SAFETY: a `repr(u8)` enum with a variant per row.
        unsafe { std::mem::transmute(op) }
    }
}

#[cfg(test)]
mod tests {
    use super::{MOp, ROWS, Sz};
    use crate::jit::backend::aarch64::abi::{c_abi, preg_float, preg_int};
    use crate::jit::backend::reg::{
        Operand, OperandConstraint, OperandKind, OperandPos, RegClass, VReg,
    };
    use crate::jit::backend::vcode::{MInst, VCode};

    #[test]
    fn every_row_is_one_the_decoder_reads() {
        for row in &ROWS {
            row.check().unwrap_or_else(|e| panic!("{}: {e}", row.name));
        }
    }

    fn rec(op: MOp, n: u8, ops: [u32; 3]) -> MInst {
        MInst {
            op: op as u8,
            sz: Sz::X as u8,
            cc: 0,
            n,
            ops,
        }
    }

    /// Records by hand: each operand as the allocator takes it, and where
    /// each instruction's allocations start.
    #[test]
    fn records_decode_to_their_operands() {
        use OperandKind::{Def, Use};
        use OperandPos::{Early, Late};
        let abi = c_abi();
        let mut v = VCode::new(&ROWS);
        let i: Vec<VReg> = (0..6).map(|_| v.new_vreg(RegClass::Int)).collect();
        let f: Vec<VReg> = (0..3).map(|_| v.new_vreg(RegClass::Float)).collect();
        let n = |r: VReg| r.vreg() as u32;
        // 0: i0 = i1 + i2.
        v.push(rec(MOp::AddRR, 0, [n(i[0]), n(i[1]), n(i[2])]));
        // 1: f0 = call(i1, f1, i2, f2): a result, integers and floats mixed.
        v.pool.extend([f[0], i[1], f[1], i[2], f[2]].map(n));
        v.push(rec(MOp::Call, 1, [7, 0, 5]));
        // 2: call(i3, i4) with no result.
        v.pool.extend([i[3], i[4]].map(n));
        v.push(rec(MOp::Call, 0, [7, 5, 2]));
        // 3: an exit with three values of its frame state.
        v.pool.extend([i[0], f[0], i[5]].map(n));
        v.push(rec(MOp::BCondExit, 0, [0, 7, 3]));
        assert_eq!(v.offs, [0, 3, 8, 10, 13]);
        let ops = |k: usize| -> Vec<Operand> {
            (0..v.operands_of(k).1)
                .map(|o| v.operand(&abi, k, o))
                .collect()
        };
        let reg = OperandConstraint::Reg;
        let op = Operand::new;
        assert_eq!(
            ops(0),
            [
                op(i[0], reg, Def, Late),
                op(i[1], reg, Use, Early),
                op(i[2], reg, Use, Early)
            ]
        );
        let fixed = OperandConstraint::FixedReg;
        assert_eq!(
            ops(1),
            [
                op(f[0], fixed(preg_float(0)), Def, Late),
                op(i[1], fixed(preg_int(0)), Use, Early),
                op(f[1], fixed(preg_float(0)), Use, Early),
                op(i[2], fixed(preg_int(1)), Use, Early),
                op(f[2], fixed(preg_float(1)), Use, Early),
            ]
        );
        assert_eq!(
            ops(2),
            [
                op(i[3], fixed(preg_int(0)), Use, Early),
                op(i[4], fixed(preg_int(1)), Use, Early)
            ]
        );
        let any = OperandConstraint::Any;
        assert_eq!(
            ops(3),
            [
                op(i[0], any, Use, Late),
                op(f[0], any, Use, Late),
                op(i[5], any, Use, Late)
            ]
        );
        assert_eq!(v.operand_at(&abi, 9), ops(2)[1]);
        // A call's result is not among what it clobbers.
        let clob = |k| v.inst_clobbers(&abi, crate::jit::backend::reg::Inst::new(k));
        assert!(!clob(1).contains(preg_float(0)) && clob(1).contains(preg_int(0)));
        assert!(clob(2).contains(preg_float(0)) && clob(2).contains(preg_int(0)));
        assert_eq!(clob(0), crate::jit::backend::reg::PRegSet::empty());
    }
}
