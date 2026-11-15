//! What every target's selection and encoder share: a region's machine code
//! before allocation with its exits (`Selected`), and its code after
//! (`Emitted`).

use crate::jit::backend::order::Layout;
use crate::jit::backend::target::asm::Asm;
use crate::jit::backend::target::inst::MInst;
use crate::jit::backend::vcode::VCode;
use crate::jit::ir::ops::{ExitKind, ExitTag};
use crate::jit::region::{SRep, SnapEntry};

/// Where an exit finds an entry's value.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Src {
    /// The exiting instruction's `k`-th operand past its own.
    Operand(u16),
    /// A boxed constant.
    Const(u64),
}

/// An exit of the code: the frame it rebuilds, from its frame state.
pub(crate) struct Exit {
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    /// Its entries, `Selected::entries[at..at + len]`.
    pub(crate) at: u32,
    pub(crate) len: u32,
    /// The instruction that leaves, and how many operands are its own.
    pub(crate) inst: u32,
    pub(crate) own: u32,
}

pub(crate) struct Selected {
    pub(crate) vcode: VCode<MInst>,
    pub(crate) exits: Vec<Exit>,
    pub(crate) entries: Vec<(u8, SRep, Src)>,
    pub(crate) layout: Layout,
}

impl Default for Selected {
    fn default() -> Self {
        Selected {
            vcode: VCode::new(),
            exits: Vec::new(),
            entries: Vec::new(),
            layout: Layout::default(),
        }
    }
}

/// An exit after allocation.
pub(crate) struct ExitOut {
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
}

/// The machine code of a region, before its address is known. Places are
/// byte offsets into it.
pub(crate) struct Emitted {
    pub(crate) asm: Asm,
    /// Where the interpreter enters (`JIT_ENTRY`/`JIT_LOOP`).
    pub(crate) entry: u32,
    pub(crate) num_spills: u32,
    /// Bytes of the native frame.
    pub(crate) frame_size: u32,
    /// The 8-byte word holding the region pointer.
    pub(crate) region_word: u32,
    pub(crate) exits: Vec<ExitOut>,
}
