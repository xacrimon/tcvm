//! The x86-64 side of a code block's unwind information
//! (`backend::unwind`): its DWARF registers and frame rules, with data
//! alignment -8.

use crate::jit::backend::unwind::{DW_CFA_DEF_CFA, DW_CFA_OFFSET, uleb};

const RBP: u8 = 6;
const RSP: u8 = 7;
const RA: u8 = 16;

pub(crate) const CODE_ALIGN: u64 = 1;
pub(crate) const RETURN_COLUMN: u8 = RA;

/// The rules at a function's entry: the return address pushed.
pub(crate) fn entry_rules(b: &mut Vec<u8>) {
    b.extend([DW_CFA_DEF_CFA, RSP, 8]);
    b.extend([DW_CFA_OFFSET | RA, 1]);
}

/// The rules with a frame open, whatever its size: `rbp` points at the
/// caller's, pushed below the return address.
pub(crate) fn frame_rules(b: &mut Vec<u8>, _frame: u32) {
    b.extend([DW_CFA_DEF_CFA, RBP]);
    uleb(b, 16);
    b.extend([DW_CFA_OFFSET | RBP, 2]);
}
