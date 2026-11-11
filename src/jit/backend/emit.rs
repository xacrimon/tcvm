//! What every target's emitter shares (10.5, 10.6): the record of a
//! region's code (`Emitted`) with its exits and call sites placed after
//! allocation, what the code links against, and the block layout's last
//! steps (jump threading, loop rotation). The instruction sequences, the
//! frame and the stubs are the target's (`target::emit`).

use crate::env::value::Value;
use crate::jit::backend::lower::{LExit, LSite, Lowered, Src};
use crate::jit::backend::ra::{InstOrEdit, Output};
use crate::jit::backend::reg::{Allocation, Block, Inst};
use crate::jit::backend::target::abi::image_index;
use crate::jit::backend::target::asm::Asm;
use crate::jit::backend::vcode::{Flow, TargetInst};
use crate::jit::ir::ops::ExitTag;
use crate::jit::ir::{ExitKind, ExitSite};
use crate::jit::region::{Loc, SRep, SnapEntry, SnapFrame};

/// The machine code of a region, before its address is known. Places are
/// byte offsets into it.
pub(crate) struct Emitted {
    pub(crate) asm: Asm,
    /// Where the interpreter enters (`JIT_ENTRY`/`JIT_LOOP`), and where a
    /// compiled caller does (5.4), when the region has a fast entry.
    pub(crate) entry: u32,
    pub(crate) fast: Option<u32>,
    pub(crate) num_spills: u32,
    /// Bytes of the native frame.
    pub(crate) frame_size: u32,
    /// The 8-byte word holding the region pointer.
    pub(crate) region_word: u32,
    pub(crate) exits: Vec<ExitOut>,
    /// The calls, for walks of the frame (H7).
    pub(crate) sites: Vec<SiteOut>,
    /// The direct branches to other regions' known entries, by the callee's
    /// prototype (its box), which `jit::set_fast` repoints: to the register
    /// entry for a call passing arguments there, else the slot one.
    pub(crate) direct: Vec<(u32, u64, bool)>,
    /// What `Region::invalidate` patches: each place and the branch target
    /// written there, the entry's first (ws2 §5.4).
    pub(crate) lazy: Vec<(u32, u32)>,
    /// A self-call's return addresses whose result guard runs out of line,
    /// and its landing for a level returning there (`jit::jit_exit`).
    pub(crate) landings: Vec<(u32, u32)>,
    /// Where each hot block starts (its IR block), in layout order, and
    /// where the hot ones end: deopt blocks, cold arms and stubs after (the
    /// asm dump's `blocks` and `cold`).
    #[cfg_attr(not(feature = "tcvm-dev"), allow(dead_code))]
    pub(crate) hot: Vec<(u32, u32)>,
    #[cfg_attr(not(feature = "tcvm-dev"), allow(dead_code))]
    pub(crate) cold: u32,
    /// With `trace`: where each run of code starts and what it is, a VCode
    /// instruction's index or one of `Origin`'s, in address order.
    #[cfg_attr(not(feature = "tcvm-dev"), allow(dead_code))]
    pub(crate) origins: Vec<(u32, u32)>,
}

/// Code that is no VCode instruction, for `Emitted::origins`.
#[repr(u32)]
pub(crate) enum Origin {
    Move = u32::MAX - 4,
    Remat,
    Entry,
    OutOfLine,
    Stubs,
}

impl Origin {
    /// The dump's name for origin `o`, unless it is an instruction's index.
    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn name(o: u32) -> Option<&'static str> {
        const MOVE: u32 = Origin::Move as u32;
        const REMAT: u32 = Origin::Remat as u32;
        const ENTRY: u32 = Origin::Entry as u32;
        const OUT_OF_LINE: u32 = Origin::OutOfLine as u32;
        const STUBS: u32 = Origin::Stubs as u32;
        Some(match o {
            MOVE => "move",
            REMAT => "remat",
            ENTRY => "entry",
            OUT_OF_LINE => "out-of-line",
            STUBS => "stubs",
            _ => return None,
        })
    }
}

/// A call of the region: its return addresses, whether each is a level's
/// (`jit_level`), the values the frame keeps in spill slots across it, its
/// header slot and the `pc` after it. `L` is a place in the code: the
/// target's label while it emits, then a byte offset.
pub(crate) struct SiteOut<L = u32> {
    pub(crate) keys: Vec<(L, bool)>,
    /// The call's slow path, where a fast callee that missed goes.
    pub(crate) miss: L,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
    pub(crate) hdr: u16,
    pub(crate) pc_after: u64,
    /// The inlined frames it is made from (`LSite::frames`).
    pub(crate) frames: Vec<SnapFrame>,
}

impl<L: Copy> SiteOut<L> {
    /// With its places as byte offsets (`at`), each return address once: a
    /// fast callee's is its fast resume's when the resume block follows
    /// the call.
    pub(crate) fn placed(self, at: impl Fn(L) -> u32) -> SiteOut {
        let mut keys: Vec<(u32, bool)> = self.keys.iter().map(|&(k, l)| (at(k), l)).collect();
        keys.sort_unstable();
        keys.dedup();
        SiteOut {
            keys,
            miss: at(self.miss),
            entries: self.entries,
            consts: self.consts,
            hdr: self.hdr,
            pc_after: self.pc_after,
            frames: self.frames,
        }
    }
}

pub(crate) struct ExitOut {
    pub(crate) pc: u32,
    pub(crate) frames: Vec<SnapFrame>,
    pub(crate) site: Option<ExitSite>,
    pub(crate) frame: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) blame: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
}

/// A function region's fast entry: what `enter` would check of the callee.
#[derive(Clone, Copy)]
pub(crate) struct FastEntry {
    pub(crate) max_stack: u8,
    pub(crate) num_params: u8,
    /// The region's prototype, whose calls enter here directly (D2).
    pub(crate) proto: u64,
}

/// What a region's code embeds of its runtime: addresses that stay put for
/// the runtime's life, and what other code it reaches now.
#[derive(Clone)]
pub(crate) struct Links {
    /// The runtime (the handler ABI's), and what code reads through it:
    /// the arena's `Metrics` and the type metatables' cells.
    pub(crate) rt: u64,
    pub(crate) metrics: u64,
    pub(crate) type_mt_cells: [u64; 6],
    /// The trampolines of `Lowered::trampolines`: (helper, trampoline).
    pub(crate) trampolines: Vec<(usize, usize)>,
    /// Known callees' prototypes (their boxes), their parameter counts and
    /// fast entries now (0 for none; `jit::direct_target`).
    pub(crate) direct: Vec<(u64, u8, usize)>,
}

impl Links {
    /// The trampoline of the helper at `target`.
    pub(crate) fn trampoline_of(&self, target: usize) -> usize {
        let t = self.trampolines.iter().find(|t| t.0 == target);
        t.expect("a trampoline for the helper").1
    }
}

/// The runtime record of an exit: each entry's location after allocation.
pub(crate) fn exit_out(l: &Lowered, out: &Output, e: &LExit) -> ExitOut {
    let mut consts = Vec::new();
    let inst = e.inst.expect("an exit without an instruction");
    let allocs = out.inst_allocs(&l.vcode, Inst::new(inst));
    let base = l.snap_ops[inst] as usize;
    let (nil, f, t) = (
        Value::nil().to_raw(),
        Value::boolean(false).to_raw(),
        Value::boolean(true).to_raw(),
    );
    let mut placed = |&(reg, rep, src): &(u8, SRep, Src)| {
        let loc = match src {
            Src::Const(w) if w == nil => Loc::Nil,
            Src::Const(w) if w == f => Loc::False,
            Src::Const(w) if w == t => Loc::True,
            Src::Const(w) => {
                consts.push(w);
                Loc::Const(consts.len() as u32 - 1)
            }
            Src::Slot(r) => Loc::Slot(r),
            Src::Operand(k) => {
                let al = allocs[base + k as usize];
                if let Some(p) = al.as_reg() {
                    Loc::Reg(image_index(p))
                } else if let Some(s) = al.as_stack() {
                    Loc::Spill(s.index() as u16)
                } else {
                    panic!("snapshot operand without an allocation");
                }
            }
        };
        SnapEntry { reg, rep, loc }
    };
    let entries = e.entries.iter().map(&mut placed).collect();
    let blame = e.blame.iter().map(&mut placed).collect();
    ExitOut {
        pc: e.pc,
        frames: e.frames.clone(),
        site: e.site,
        frame: e.frame,
        kind: e.kind,
        tag: e.tag,
        entries,
        blame,
        consts,
    }
}

/// The values a call keeps in the frame: its stack-slot snapshot operands
/// (`snaps`, the call's allocations from its first snapshot operand), and
/// the registers below its header `hdr` whose home slots hold theirs.
pub(crate) fn site_entries(
    site: &LSite,
    snaps: &[Allocation],
    hdr: u16,
) -> (Vec<SnapEntry>, Vec<u64>) {
    let mut entries = Vec::new();
    let mut consts = Vec::new();
    for &(reg, rep, src) in &site.entries {
        let loc = match src {
            Src::Const(w) => {
                consts.push(w);
                Loc::Const(consts.len() as u32 - 1)
            }
            Src::Operand(k) => Loc::Spill(
                snaps[k as usize]
                    .as_stack()
                    .expect("a kept value outside the frame")
                    .index() as u16,
            ),
            Src::Slot(_) => unreachable!("a call's entry in a frame slot"),
        };
        entries.push(SnapEntry { reg, rep, loc });
    }
    for r in site.homes.iter().filter(|&r| r < hdr as usize) {
        entries.push(SnapEntry {
            reg: r as u8,
            rep: SRep::Val,
            loc: Loc::Slot(r as u8),
        });
    }
    (entries, consts)
}

/// An instruction with no code: a value left for the allocator to make
/// where used.
pub(crate) fn emits_nothing(l: &Lowered, out: &Output, i: usize) -> bool {
    l.vcode.insts[i].is_remat() && out.inst_allocs(&l.vcode, Inst::new(i))[0].is_none()
}

/// Loop rotation (V8's `ComputeAssemblyOrder`, with any exiting block as the
/// cut): a hot loop whose last block jumps back to its first is laid out
/// from the block after one that branches out to the block following the
/// loop, so that branch becomes the back edge and the jump a fall-through
/// into the header; entering then takes a jump to the header. Returns the
/// rotated loops' new first blocks.
pub(crate) fn rotate_loops<L: Copy + PartialEq>(
    l: &Lowered,
    labels: &[L],
    dest: &[L],
    laid: &mut [usize],
    hot: usize,
) -> Vec<usize> {
    let mut tops = Vec::new();
    let term = |b: usize| l.vcode.insts[l.vcode.blocks[b].end as usize - 1].flow();
    for &(first, last) in &l.loops {
        let inside = |b: usize| (first as usize..=last as usize).contains(&b);
        let Some(s) = laid[..hot].iter().position(|&b| inside(b)) else {
            continue;
        };
        let e = s + laid[s..hot].iter().take_while(|&&b| inside(b)).count();
        if e == laid.len() || laid[e..hot].iter().any(|&b| inside(b)) {
            continue;
        }
        let latch = laid[e - 1];
        if term(latch) != Flow::Jump || dest[l.vcode.succs(latch)[0].index()] != labels[laid[s]] {
            continue;
        }
        let after = labels[laid[e]];
        // The cut nearest the latch.
        let cut = (s..e - 1).rev().find(|&k| {
            let b = laid[k];
            if term(b) != Flow::Branch {
                return false;
            }
            let ss = l.vcode.succs(b);
            let (t, f) = (dest[ss[0].index()], dest[ss[1].index()]);
            let next = labels[laid[k + 1]];
            (t, f) == (after, next) || (t, f) == (next, after)
        });
        if let Some(k) = cut {
            laid[s..e].rotate_left(k + 1 - s);
            tops.push(laid[s]);
        }
    }
    tops
}

/// Where branches to each block go, whether it is left out, and where a
/// resume block's other returns go when not into it. Branches to a block
/// that is only a deopt go to its exit (`exit_of`). A resume block that is
/// only a plain resume (`Flow::Resume`) and a jump has its slow and native
/// returns go where the jump does.
pub(crate) fn thread_jumps<L: Copy>(
    l: &Lowered,
    out: &Output,
    labels: &[L],
    exit_of: impl Fn(u32) -> L,
) -> (Vec<L>, Vec<bool>, Vec<Option<L>>) {
    let nb = l.vcode.blocks.len();
    // A block's `n` instructions that emit code, if it has no moves.
    let only = |b: usize, n: usize| -> Option<Vec<Flow>> {
        if b == 0 {
            return None;
        }
        let insts: Option<Vec<Flow>> = out
            .block_insts_and_edits(&l.vcode, Block::new(b))
            .filter(|ie| !matches!(ie, InstOrEdit::Inst(i) if emits_nothing(l, out, i.index())))
            .map(|ie| match ie {
                InstOrEdit::Inst(i) => Some(l.vcode.insts[i.index()].flow()),
                InstOrEdit::Edit(_) => None,
            })
            .collect();
        insts.filter(|v| v.len() == n)
    };
    let mut dest = labels.to_vec();
    let mut skip = vec![false; nb];
    for b in 0..nb {
        if !l.vcode.blocks[b].resume
            && let Some(i) = only(b, 1)
            && let Flow::Deopt(exit) = i[0]
        {
            dest[b] = exit_of(exit);
            skip[b] = true;
        }
    }
    let resume_to = (0..nb)
        .map(|b| {
            if !l.vcode.blocks[b].resume {
                return None;
            }
            let i = only(b, 2)?;
            (i[..] == [Flow::Resume { plain: true }, Flow::Jump])
                .then(|| dest[l.vcode.succs(b)[0].index()])
        })
        .collect();
    (dest, skip, resume_to)
}
