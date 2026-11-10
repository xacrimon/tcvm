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
    /// The frame slot of this register holds the value: the entry's own at
    /// a call site, its home; at an exit, as the exit finds it (before any
    /// entry is written).
    Slot(u8),
}

/// One home-slot write of an exit; `reg` counts from the region's base.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SnapEntry {
    pub(crate) reg: u8,
    pub(crate) rep: SRep,
    pub(crate) loc: Loc,
}

/// An inlined call's frame an exit rebuilds, outermost first: the callee
/// (a pool index, or `DYNAMIC`), the `CALL`'s `a` and `c`, its base
/// relative to the enclosing frame's, and the `CALL`'s pc.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SnapFrame {
    pub(crate) func: u32,
    pub(crate) a: u8,
    pub(crate) c: u8,
    /// Its base past the enclosing frame's; 0 for a callee that replaced
    /// the region's frame by a tail call, whose header it takes.
    pub(crate) delta: u16,
    pub(crate) caller_pc: u32,
    /// Its callee captures locals: the header an exit writes has
    /// `HAS_OPEN`, so the frame's return closes what a `CLOSURE` opened.
    pub(crate) open: bool,
}

/// `SnapFrame::open` in the encoded caller pc.
const OPEN: u32 = 1 << 31;

/// A `SnapFrame::func` of a closure known by its prototype, `Region::protos`
/// by the low bits: the snapshot's entry for the header's word 0 (register
/// `delta - 4` of the enclosing frame) holds it.
pub(crate) const DYNAMIC: u32 = 1 << 31;

/// Append a snapshot to `out`: `kind:1 frames:2 entries:12`, the pc, three
/// words per frame, a word per entry (`reg:8 rep:3 loc:3 index:18`).
pub(crate) fn encode_snap(
    out: &mut Vec<u32>,
    kind: ExitKind,
    pc: u32,
    frames: &[SnapFrame],
    entries: &[SnapEntry],
) {
    assert!(frames.len() < 4 && entries.len() < 1 << 12);
    let kind = (kind == ExitKind::After) as u32;
    out.push(kind | (frames.len() as u32) << 1 | (entries.len() as u32) << 3);
    out.push(pc);
    for f in frames {
        out.push(f.func);
        out.push(f.a as u32 | (f.c as u32) << 8 | (f.delta as u32) << 16);
        out.push(f.caller_pc | if f.open { OPEN } else { 0 });
    }
    for e in entries {
        let (loc, index) = match e.loc {
            Loc::Reg(i) => (0, i as u32),
            Loc::Spill(i) => (1, i as u32),
            Loc::Const(i) => (2, i),
            Loc::Nil => (3, 0),
            Loc::False => (4, 0),
            Loc::True => (5, 0),
            Loc::Slot(r) => (6, r as u32),
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
        let kind = if h & 1 != 0 {
            ExitKind::After
        } else {
            ExitKind::Before
        };
        let nf = (h >> 1 & 3) as usize;
        let ne = (h >> 3 & 0xfff) as usize;
        let frames = &words[2..2 + 3 * nf];
        SnapView {
            kind,
            pc: words[1],
            frames,
            entries: &words[2 + 3 * nf..2 + 3 * nf + ne],
        }
    }

    pub(crate) fn frames(&self) -> impl Iterator<Item = SnapFrame> + 'a {
        self.frames
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[func, w, caller_pc]| SnapFrame {
                func,
                a: w as u8,
                c: (w >> 8) as u8,
                delta: (w >> 16) as u16,
                caller_pc: caller_pc & !OPEN,
                open: caller_pc & OPEN != 0,
            })
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = SnapEntry> + 'a {
        self.entries.iter().map(|&w| Self::entry_of(w))
    }

    /// The entry of register `reg`.
    pub(crate) fn entry(&self, reg: usize) -> Option<SnapEntry> {
        let w = self.entries.iter().find(|&&w| w as u8 as usize == reg)?;
        Some(Self::entry_of(*w))
    }

    fn entry_of(w: u32) -> SnapEntry {
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
            5 => Loc::True,
            _ => Loc::Slot(index as u8),
        };
        SnapEntry {
            reg: w as u8,
            rep,
            loc,
        }
    }
}

pub(crate) struct ExitInfo {
    /// The exit's snapshot: an offset into the region's arena.
    pub(crate) snap: u32,
    pub(crate) tag: ExitTag,
    pub(crate) count: Cell<u32>,
    /// `FeedbackVector::seen` of the exit's site when compiled: one the exit
    /// changed means the recompile learns something.
    pub(crate) seen: u64,
    /// The site's pc when not the snapshot's (`ExitInfo::site`), else
    /// `NO_BLAME`.
    site: u32,
    /// The `Region::site_protos` index of the site's function when it is
    /// not the snapshot's (a check moved out of an inlined body), or
    /// `u16::MAX`.
    pub(crate) site_proto: u16,
    /// A moved exit's blame entries, the site's operands by its registers:
    /// a snapshot without frames in the arena, or `NO_BLAME`.
    blame: u32,
}

pub(crate) const NO_BLAME: u32 = u32::MAX;
const _: () = assert!(size_of::<ExitInfo>() == 32);

impl ExitInfo {
    pub(crate) fn new(
        snap: u32,
        tag: ExitTag,
        seen: u64,
        site: Option<u32>,
        site_proto: u16,
        blame: u32,
    ) -> Self {
        ExitInfo {
            snap,
            tag,
            count: Cell::new(0),
            seen,
            site: site.unwrap_or(NO_BLAME),
            site_proto,
            blame,
        }
    }

    /// The site, when not the snapshot's pc: a check hoisted out of a loop
    /// records its access's miss (E2), one at a checkpoint its instruction.
    #[inline]
    pub(crate) fn site(&self) -> Option<u32> {
        (self.site != NO_BLAME).then_some(self.site)
    }
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
    /// A function region's entry for compiled callers (5.4), or null.
    #[collect(require_static)]
    pub(crate) fast_entry: *const u8,
    /// Every heap value the code or a snapshot embeds.
    pub(crate) pool: Box<[Value<'gc>]>,
    /// Every shape the code embeds.
    pub(crate) shapes: Box<[crate::env::Shape<'gc>]>,
    #[collect(require_static)]
    pub(crate) exits: Box<[ExitInfo]>,
    /// Under `TCVM_JIT_DEOPT_ALL`, whether each exit fired: the code takes
    /// one that has not (14); empty otherwise.
    #[collect(require_static)]
    pub(crate) fired: Box<[Cell<u32>]>,
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
    /// The prototypes of known callees (`Func::protos`), whose states
    /// repoint the code's direct calls of them.
    pub(crate) protos: Box<[Gc<'gc, Prototype<'gc>>]>,
    /// The functions of exits' sites outside their snapshots' frames.
    pub(crate) site_protos: Box<[Gc<'gc, Prototype<'gc>>]>,
    pub(crate) entry_pc: u32,
    /// The slots past its base the region's inlined frames take, past the
    /// frame's own window (`Lowered::window`), else 0.
    pub(crate) window: u16,
    /// The entry's loop nest, innermost first (`Entry::nest`).
    #[collect(require_static)]
    pub(crate) nest: Box<[u32]>,
    #[collect(require_static)]
    pub(crate) num_spills: u32,
    /// Bytes of the region's native frame, which `exit_common` pops.
    #[collect(require_static)]
    pub(crate) frame_size: u32,
    #[collect(require_static)]
    pub(crate) retired: Cell<bool>,
    /// Takings of one exit that ask for a recompile: `EXIT_HOT`, doubled
    /// per retry of the entry.
    #[collect(require_static)]
    pub(crate) exit_hot: Cell<u32>,
    /// The calls a frame of this region can be in while others run above
    /// it, by the code offset of the return addresses naming them.
    #[collect(require_static)]
    pub(crate) sites: Box<[Site]>,
    /// For a region with dependencies (ws2 §5.3), what `invalidate`
    /// writes: (code offset, branch), the entry's and each lazy exit's.
    #[collect(require_static)]
    pub(crate) lazy: Box<[(u32, u32)]>,
    /// Code offsets of self-calls' return addresses whose result guard runs
    /// out of line, and of the landing that runs it: where a level started
    /// for their callee returns instead (`jit::jit_exit`).
    #[collect(require_static)]
    pub(crate) landings: Box<[(u32, u32)]>,
    #[collect(require_static)]
    pub(crate) invalidated: Cell<bool>,
}

/// A call of the region's: its return address, and what a walk of the
/// frame needs there (H7, H11).
#[derive(Clone, Copy)]
pub(crate) struct Site {
    /// The code offset of the return address.
    pub(crate) key: u32,
    /// The snapshot naming the values the frame keeps in spill slots
    /// across the call, in `snaps`.
    pub(crate) snap: u32,
    /// The callee's header slot, and the `pc` after the call, whose header
    /// words a fast call does not write (D4).
    pub(crate) hdr: u16,
    pub(crate) pc_after: u64,
    /// The return address is a level's (`jit_level`): `BASE` there is this
    /// frame's. Otherwise it is a fast callee's, and `BASE` the callee's.
    pub(crate) level: bool,
    /// The code offset of the call's slow path, where a fast callee whose
    /// entry checks failed goes (`abi::call_miss`).
    pub(crate) miss: u32,
}

impl<'gc> Region<'gc> {
    pub(crate) fn snap(&self, e: &ExitInfo) -> SnapView<'_> {
        SnapView::decode(&self.snaps[e.snap as usize..])
    }

    pub(crate) fn blame(&self, e: &ExitInfo) -> Option<SnapView<'_>> {
        (e.blame != NO_BLAME).then(|| SnapView::decode(&self.snaps[e.blame as usize..]))
    }

    /// A dependency broke (ws2 §5.4): from now on its entry goes to
    /// `jit_dead_entry`, and a frame leaves at the first lazy exit it
    /// reaches, each past a call or a table write, where any break happens.
    /// No context is needed, so a collector's weak clear may break one; the
    /// entry is retired by whichever comes first.
    pub(crate) fn invalidate(&self) {
        if self.invalidated.replace(true) {
            return;
        }
        for &(off, w) in &self.lazy {
            self.code.patch_u32(off as usize, w);
        }
    }

    /// The region whose code holds the return address `ra`: its pointer is
    /// the code block's first word.
    ///
    /// # Safety
    /// `ra` points into a live region's code.
    pub(crate) unsafe fn of_return(ra: usize) -> *const Region<'static> {
        unsafe { *(crate::jit::backend::alloc::block_start(ra) as *const *const Region<'static>) }
    }

    /// The site whose return address is `ra`, and its snapshot.
    pub(crate) fn site(&self, ra: usize) -> Option<(&Site, SnapView<'_>)> {
        let off = ra.checked_sub(self.code.rx() as usize)? as u32;
        let k = self.sites.binary_search_by_key(&off, |s| s.key).ok()?;
        let s = &self.sites[k];
        Some((s, SnapView::decode(&self.snaps[s.snap as usize..])))
    }
}

pub(crate) mod layout {
    use super::Region;

    pub(crate) const NUM_SPILLS: usize = std::mem::offset_of!(Region<'static>, num_spills);
    pub(crate) const FRAME_SIZE: usize = std::mem::offset_of!(Region<'static>, frame_size);
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
                open: true,
            },
            SnapFrame {
                func: 0,
                a: 0,
                c: 0,
                delta: 4,
                caller_pc: 1 << 20,
                open: false,
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
            SnapEntry {
                reg: 2,
                rep: SRep::I32,
                loc: Loc::Slot(255),
            },
        ];
        let mut out = vec![0xdead];
        encode_snap(&mut out, ExitKind::After, 1234, &frames, &entries);
        let v = SnapView::decode(&out[1..]);
        assert_eq!(v.kind, ExitKind::After);
        assert_eq!(v.pc, 1234);
        assert_eq!(v.frames().collect::<Vec<_>>(), frames);
        assert_eq!(v.entries().collect::<Vec<_>>(), entries);
    }
}
