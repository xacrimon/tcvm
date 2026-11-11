//! The aarch64 side of a code block's unwind information
//! (`backend::unwind`): its DWARF registers and frame rules, with data
//! alignment -8.

use crate::jit::backend::unwind::{DW_CFA_DEF_CFA, DW_CFA_OFFSET, uleb};

const X29: u8 = 29;
const X30: u8 = 30;
const SP: u8 = 31;

pub(crate) const CODE_ALIGN: u64 = 4;
pub(crate) const RETURN_COLUMN: u8 = X30;

/// The rules at a function's entry: nothing pushed.
pub(crate) fn entry_rules(b: &mut Vec<u8>) {
    b.extend([DW_CFA_DEF_CFA, SP, 0]);
}

/// The rules with a frame of `frame` bytes open: its frame record at the
/// bottom and x29 pointing at it.
pub(crate) fn frame_rules(b: &mut Vec<u8>, frame: u32) {
    b.extend([DW_CFA_DEF_CFA, X29]);
    uleb(b, frame as u64);
    b.push(DW_CFA_OFFSET | X29);
    uleb(b, frame as u64 / 8);
    b.push(DW_CFA_OFFSET | X30);
    uleb(b, frame as u64 / 8 - 1);
}
