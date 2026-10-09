//! A compiled region: its code, the constants it embeds, and its exits with
//! the snapshots that rebuild the frame, encoded as Appendix C gives.

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
    Nil,
    False,
    True,
}

/// One home-slot write of an exit; `reg` counts from the region's base.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SnapEntry {
    pub(crate) reg: u8,
    pub(crate) rep: SRep,
    pub(crate) loc: Loc,
}

/// An inlined call's frame an exit rebuilds, outermost first: the callee
/// (a pool index), the `CALL`'s `a` and `c`, its base relative to the
/// enclosing frame's, and the `CALL`'s pc.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SnapFrame {
    pub(crate) func: u32,
    pub(crate) a: u8,
    pub(crate) c: u8,
    pub(crate) delta: u16,
    pub(crate) caller_pc: u32,
}

/// Append a snapshot to `out`: `kind:2 frames:2 entries:12`, the pc, three
/// words per frame, a word per entry (`reg:8 rep:3 loc:3 index:18`).
pub(crate) fn encode_snap(
    out: &mut Vec<u32>,
    kind: ExitKind,
    pc: u32,
    frames: &[SnapFrame],
    entries: &[SnapEntry],
) {
    assert!(frames.len() < 4 && entries.len() < 1 << 12);
    let kind = match kind {
        ExitKind::Before => 0,
        ExitKind::After => 1,
        ExitKind::Gc => 2,
    };
    out.push(kind | (frames.len() as u32) << 2 | (entries.len() as u32) << 4);
    out.push(pc);
    for f in frames {
        out.push(f.func);
        out.push(f.a as u32 | (f.c as u32) << 8 | (f.delta as u32) << 16);
        out.push(f.caller_pc);
    }
    for e in entries {
        let (loc, index) = match e.loc {
            Loc::Reg(i) => (0, i as u32),
            Loc::Spill(i) => (1, i as u32),
            Loc::Const(i) => (2, i),
            Loc::Nil => (3, 0),
            Loc::False => (4, 0),
            Loc::True => (5, 0),
        };
        assert!(index < 1 << 18);
        out.push(e.reg as u32 | (e.rep as u32) << 8 | loc << 11 | index << 14);
    }
}

/// A snapshot read back from a region's arena.
pub(crate) struct SnapView<'a> {
    pub(crate) kind: ExitKind,
    pub(crate) pc: u32,
    frames: &'a [u32],
    entries: &'a [u32],
}

impl<'a> SnapView<'a> {
    pub(crate) fn decode(words: &'a [u32]) -> Self {
        let h = words[0];
        let kind = match h & 3 {
            0 => ExitKind::Before,
            1 => ExitKind::After,
            _ => ExitKind::Gc,
        };
        let nf = (h >> 2 & 3) as usize;
        let ne = (h >> 4 & 0xfff) as usize;
        let frames = &words[2..2 + 3 * nf];
        SnapView {
            kind,
            pc: words[1],
            frames,
            entries: &words[2 + 3 * nf..2 + 3 * nf + ne],
        }
    }

    pub(crate) fn frames(&self) -> impl Iterator<Item = SnapFrame> + 'a {
        self.frames.chunks_exact(3).map(|w| SnapFrame {
            func: w[0],
            a: w[1] as u8,
            c: (w[1] >> 8) as u8,
            delta: (w[1] >> 16) as u16,
            caller_pc: w[2],
        })
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = SnapEntry> + 'a {
        self.entries.iter().map(|&w| {
            let index = w >> 14;
            let rep = match w >> 8 & 7 {
                0 => SRep::Val,
                1 => SRep::I32,
                2 => SRep::I64,
                3 => SRep::F64,
                _ => SRep::B1,
            };
            let loc = match w >> 11 & 7 {
                0 => Loc::Reg(index as u8),
                1 => Loc::Spill(index as u16),
                2 => Loc::Const(index),
                3 => Loc::Nil,
                4 => Loc::False,
                _ => Loc::True,
            };
            SnapEntry {
                reg: w as u8,
                rep,
                loc,
            }
        })
    }
}

pub(crate) struct ExitInfo {
    /// The exit's snapshot: an offset into the region's arena.
    pub(crate) snap: u32,
    pub(crate) tag: ExitTag,
    pub(crate) count: Cell<u32>,
}

/// A region of compiled code, owned by its prototype's `JitState`.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Region<'gc> {
    /// Freed before the allocator it points into.
    #[collect(require_static)]
    pub(crate) code: CodeBlock,
    /// Keeps the segment `code` lies in mapped: dropping `code` writes its
    /// header.
    #[collect(require_static)]
    pub(crate) _alloc: Rc<CodeAllocator>,
    #[collect(require_static)]
    pub(crate) entry: *const u8,
    /// Every heap value the code or a snapshot embeds.
    pub(crate) pool: Box<[Value<'gc>]>,
    #[collect(require_static)]
    pub(crate) exits: Box<[ExitInfo]>,
    /// The snapshots, which exits with equal ones share.
    #[collect(require_static)]
    pub(crate) snaps: Box<[u32]>,
    /// Raw words of snapshot constants (heap ones are in `pool` too).
    #[collect(require_static)]
    pub(crate) consts: Box<[u64]>,
    /// The registers the prologue's entry guards test.
    #[collect(require_static)]
    pub(crate) entry_regs: Box<[u8]>,
    #[collect(require_static)]
    pub(crate) entry_fails: Cell<u32>,
    pub(crate) proto: Gc<'gc, Prototype<'gc>>,
    pub(crate) entry_pc: u32,
    /// The native frame's size, which `exit_common` pops.
    #[collect(require_static)]
    pub(crate) frame_size: u32,
    #[collect(require_static)]
    pub(crate) num_spills: u32,
    #[collect(require_static)]
    pub(crate) retired: Cell<bool>,
}

impl<'gc> Region<'gc> {
    pub(crate) fn snap(&self, e: &ExitInfo) -> SnapView<'_> {
        SnapView::decode(&self.snaps[e.snap as usize..])
    }
}

pub(crate) mod layout {
    use super::Region;

    pub(crate) const FRAME_SIZE: usize = std::mem::offset_of!(Region<'static>, frame_size);
    pub(crate) const NUM_SPILLS: usize = std::mem::offset_of!(Region<'static>, num_spills);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_round_trip() {
        let frames = [
            SnapFrame {
                func: 3,
                a: 7,
                c: 2,
                delta: 11,
                caller_pc: 40,
            },
            SnapFrame {
                func: 0,
                a: 0,
                c: 0,
                delta: 4,
                caller_pc: 1 << 20,
            },
        ];
        let entries = [
            SnapEntry {
                reg: 255,
                rep: SRep::I64,
                loc: Loc::Spill(63),
            },
            SnapEntry {
                reg: 0,
                rep: SRep::Val,
                loc: Loc::Const((1 << 18) - 1),
            },
            SnapEntry {
                reg: 9,
                rep: SRep::B1,
                loc: Loc::Reg(53),
            },
            SnapEntry {
                reg: 1,
                rep: SRep::Val,
                loc: Loc::True,
            },
        ];
        let mut out = vec![0xdead];
        encode_snap(&mut out, ExitKind::Gc, 1234, &frames, &entries);
        let v = SnapView::decode(&out[1..]);
        assert_eq!(v.kind, ExitKind::Gc);
        assert_eq!(v.pc, 1234);
        assert_eq!(v.frames().collect::<Vec<_>>(), frames);
        assert_eq!(v.entries().collect::<Vec<_>>(), entries);
    }
}
