//! A compiled region: its code, the constants it embeds, and its exits with
//! the snapshots that rebuild the frame (Appendix C).

use std::cell::Cell;
use std::rc::Rc;

use crate::dmm::{Collect, Gc};
use crate::env::function::Prototype;
use crate::env::value::Value;
use crate::jit::backend::alloc::{CodeAllocator, CodeBlock};
use crate::jit::ir::ExitKind;
use crate::jit::ir::ops::ExitTag;

/// How a snapshot entry's value is held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SRep {
    Val,
    I32,
    I64,
    F64,
    B1,
}

/// Where a snapshot entry's value is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Loc {
    /// Register image index (`exit_common`'s order).
    Reg(u8),
    /// Spill slot.
    Spill(u16),
    /// A raw word in the region's constant table.
    Const(u32),
}

/// One home-slot write of an exit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapEntry {
    pub(crate) reg: u8,
    pub(crate) rep: SRep,
    pub(crate) loc: Loc,
}

pub(crate) struct ExitInfo {
    /// The pc the interpreter resumes at.
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    pub(crate) snap_start: u32,
    pub(crate) snap_len: u32,
    pub(crate) count: Cell<u32>,
}

/// A region of compiled code, owned by its prototype's `JitState`.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Region<'gc> {
    /// Freed before the allocator it points into.
    #[collect(require_static)]
    pub(crate) code: CodeBlock,
    #[collect(require_static)]
    pub(crate) alloc: Rc<CodeAllocator>,
    #[collect(require_static)]
    pub(crate) entry: *const u8,
    /// Every heap value the code or a snapshot embeds.
    pub(crate) pool: Box<[Value<'gc>]>,
    #[collect(require_static)]
    pub(crate) exits: Box<[ExitInfo]>,
    #[collect(require_static)]
    pub(crate) snaps: Box<[SnapEntry]>,
    /// Raw words of snapshot constants (heap ones are in `pool` too).
    #[collect(require_static)]
    pub(crate) consts: Box<[u64]>,
    pub(crate) proto: Gc<'gc, Prototype<'gc>>,
    pub(crate) entry_pc: u32,
    /// The native frame's size, which `exit_common` pops.
    #[collect(require_static)]
    pub(crate) frame_size: u32,
    #[collect(require_static)]
    pub(crate) num_spills: u32,
    #[collect(require_static)]
    pub(crate) retired: Cell<bool>,
    #[collect(require_static)]
    pub(crate) entry_fails: Cell<u32>,
}

impl<'gc> Region<'gc> {
    pub(crate) fn snap(&self, e: &ExitInfo) -> &[SnapEntry] {
        &self.snaps[e.snap_start as usize..(e.snap_start + e.snap_len) as usize]
    }
}

pub(crate) mod layout {
    use super::Region;

    pub(crate) const FRAME_SIZE: usize = std::mem::offset_of!(Region<'static>, frame_size);
    pub(crate) const NUM_SPILLS: usize = std::mem::offset_of!(Region<'static>, num_spills);
}
