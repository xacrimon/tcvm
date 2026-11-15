//! aarch64 machine instructions before register allocation: one `MInst` is
//! one machine instruction, or one of the declared pseudo-instructions
//! (entry, return). The table gives each opcode's register operands in
//! order, which is the order of its operands in the `VCode` and of the
//! allocations the encoder reads.

use crate::jit::backend::aarch64::asm::Sz;
use crate::jit::backend::vcode::{Flow, TargetInst};

/// A register operand's part.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// An integer register written, read.
    D,
    U,
    /// A float register written, read.
    Fd,
    Fu,
}

pub(crate) struct MInfo {
    pub(crate) name: &'static str,
    pub(crate) roles: &'static [Role],
}

macro_rules! mops {
    ($($name:ident ($($r:ident)*);)*) => {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum MOp { $($name),* }

        pub(crate) static MOPS: [MInfo; [$(MOp::$name),*].len()] = [$(MInfo {
            name: stringify!($name),
            roles: &[$(Role::$r),*],
        }),*];
    };
}

// name (register operands). `imm` is the immediate where there is one: a
// constant, a shift, a frame slot, an exit's index.
mops! {
    // Opens the native frame.
    Entry     ();
    // Values the allocator may make again where used: left without an
    // allocation they have no code.
    MovImm    (D);
    FImm      (Fd);
    LdrSlot   (D);
    StrSlot   (U);
    // `imm` is the byte offset from the base, the last register; a `W`
    // load is of 32 bits.
    Ldr       (D U);
    Str       (U U);
    // The base, then an index of 8-byte elements: `cc` 1 for a 32-bit index
    // zero-extended (`uxtw`).
    LdrIdx    (D U U);
    StrIdx    (U U U);
    MovBase   (D);
    Uxtw      (D U);
    Cset      (D);
    // The first register where `cc` holds, else the second.
    Csel      (D U U);
    AndRR     (D U U);
    OrrRR     (D U U);
    AndImm    (D U);
    TstImm    (U);
    AddRR     (D U U);
    SubRR     (D U U);
    MulRR     (D U U);
    AddImm    (D U);
    SubImm    (D U);
    // Flag-setting: the branch that reads the flags follows, with only the
    // allocator's moves between.
    AddsRR    (D U U);
    SubsRR    (D U U);
    AddsImm   (D U);
    SubsImm   (D U);
    CmpRR     (U U);
    CmpImm    (U);
    CmnImm    (U);
    // `cmp xn, wm, sxtw`: whether a product fits 32 bits.
    CmpSxtw   (U U);
    Fcmp      (Fu Fu);
    LslImm    (D U);
    LsrImm    (D U);
    AsrImm    (D U);
    Smull     (D U U);
    OrrImm    (D U);
    Scvtf     (Fd U);
    FmovToGpr (D Fu);
    FmovFromGpr (Fd U);
    FAdd      (Fd Fu Fu);
    FSub      (Fd Fu Fu);
    FMul      (Fd Fu Fu);
    FDiv      (Fd Fu Fu);
    // To successor 0.
    B         ();
    // To successor 0 where `cc` holds, else successor 1.
    BCond     ();
    // To exit `imm` where `cc` holds; `BExit` always. Their frame state's
    // values follow as operands the allocator may leave anywhere.
    BCondExit ();
    BExit     ();
    // A C call of helper `imm & 0xffff` with immediate `imm >> 16`: its
    // operands are fixed to the argument registers past the pinned ones.
    Call      ();
    // Return `imm >> 8` results from `R[imm & 0xff..]`.
    Ret       ();
}

/// 16 bytes; the registers are the `VCode`'s operands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MInst {
    pub(crate) op: MOp,
    /// An `asm::Cond`.
    pub(crate) cc: u8,
    pub(crate) sz: Sz,
    pub(crate) imm: u64,
}

const _: () = assert!(size_of::<MInst>() == 16);

impl MOp {
    pub(crate) fn info(self) -> &'static MInfo {
        &MOPS[self as usize]
    }
}

impl TargetInst for MInst {
    #[cfg(feature = "tcvm-dev")]
    fn is_branch(&self) -> bool {
        matches!(self.op, MOp::B | MOp::BCond)
    }

    fn flow(&self) -> Flow {
        match self.op {
            MOp::B => Flow::Jump,
            MOp::BCond => Flow::Branch,
            MOp::BExit => Flow::Deopt(self.imm as u32),
            _ => Flow::Other,
        }
    }

    fn is_remat(&self) -> bool {
        matches!(self.op, MOp::MovImm | MOp::FImm | MOp::LdrSlot)
    }
}
