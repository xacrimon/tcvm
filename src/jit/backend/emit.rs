//! What every target's selection and encoder share: a region's machine code
//! before allocation with its exits (`Selected`), and its code after
//! (`Emitted`).

use crate::jit::backend::order::Layout;
use crate::jit::backend::ra::{InstOrEdit, Output};
use crate::jit::backend::reg::{Block, Inst};
use crate::jit::backend::target::asm::Asm;
use crate::jit::backend::target::inst::MInst;
use crate::jit::backend::vcode::{Flow, TargetInst, VCode};
use crate::jit::ir::ops::{ExitKind, ExitTag};
use crate::jit::region::{SRep, SnapEntry};

/// Where an exit finds an entry's value.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Src {
    /// The exiting instruction's `k`-th operand past its own.
    Operand(u16),
    /// A boxed constant.
    Const(u64),
    /// A call site's: the register's frame slot holds it.
    Home,
}

/// A call of the code: what the frame keeps across it, for the collector.
pub(crate) struct CallSite {
    /// Its entries, `Selected::entries[at..at + len]`; one with register
    /// `TRACED` is a value no register of the frame names.
    pub(crate) at: u32,
    pub(crate) len: u32,
    /// The call's block and instruction.
    pub(crate) block: u32,
    pub(crate) inst: u32,
}

pub(crate) const TRACED: u8 = 255;

/// A call site after allocation: its return addresses (with whether each
/// is a level's), its slow path, and its entries.
pub(crate) struct SiteOut {
    pub(crate) keys: Vec<(u32, bool)>,
    pub(crate) miss: u32,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
    pub(crate) hdr: u16,
    pub(crate) pc_after: u64,
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
    pub(crate) sites: Vec<CallSite>,
    pub(crate) layout: Layout,
}

impl Default for Selected {
    fn default() -> Self {
        Selected {
            vcode: VCode::new(),
            exits: Vec::new(),
            entries: Vec::new(),
            sites: Vec::new(),
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
    pub(crate) sites: Vec<SiteOut>,
}

/// Loop rotation (V8's `ComputeAssemblyOrder`, with any exiting block as the
/// cut): a hot loop whose last block jumps back to its first is laid out
/// from the block after one that branches out to the block following the
/// loop, so that branch becomes the back edge and the jump a fall-through
/// into the header; entering then takes a jump to the header. `loops` and
/// `hot` are the layout's, `laid` its emit order less the blocks left out.
/// Returns the rotated loops' new first blocks.
pub(crate) fn rotate_loops<I: TargetInst, L: Copy + PartialEq>(
    v: &VCode<I>,
    loops: &[(u32, u32)],
    labels: &[L],
    dest: &[L],
    laid: &mut [u32],
    hot: usize,
) -> Vec<u32> {
    let mut tops = Vec::new();
    let term = |b: u32| v.insts[v.blocks[b as usize].end as usize - 1].flow();
    for &(first, last) in loops {
        let inside = |b: u32| (first..=last).contains(&b);
        let Some(s) = laid[..hot].iter().position(|&b| inside(b)) else {
            continue;
        };
        let e = s + laid[s..hot].iter().take_while(|&&b| inside(b)).count();
        if e == laid.len() || laid[e..hot].iter().any(|&b| inside(b)) {
            continue;
        }
        let latch = laid[e - 1];
        if term(latch) != Flow::Jump
            || dest[v.succs(latch as usize)[0].index()] != labels[laid[s] as usize]
        {
            continue;
        }
        let after = labels[laid[e] as usize];
        // The cut nearest the latch.
        let cut = (s..e - 1).rev().find(|&k| {
            let b = laid[k];
            if term(b) != Flow::Branch {
                return false;
            }
            let ss = v.succs(b as usize);
            let (t, f) = (dest[ss[0].index()], dest[ss[1].index()]);
            let next = labels[laid[k + 1] as usize];
            (t, f) == (after, next) || (t, f) == (next, after)
        });
        if let Some(k) = cut {
            laid[s..e].rotate_left(k + 1 - s);
            tops.push(laid[s]);
        }
    }
    tops
}

/// Where branches to each block go, and whether it is left out: a block
/// whose only code is a deopt is its exit (`exit_of`), and one that is only
/// a jump is where the jump goes.
pub(crate) fn thread_jumps<I: TargetInst, L: Copy>(
    v: &VCode<I>,
    out: &Output,
    labels: &[L],
    exit_of: impl Fn(u32) -> L,
) -> (Vec<L>, Vec<bool>) {
    let mut dest = labels.to_vec();
    let mut skip = vec![false; labels.len()];
    let mut jumps = vec![usize::MAX; labels.len()];
    for b in 1..labels.len() {
        if v.blocks[b].resume {
            continue;
        }
        // Its instructions that emit code: a move is one, a value left for
        // the allocator to make where used is not.
        let mut code = out
            .block_insts_and_edits(v, Block::new(b))
            .filter(|ie| match ie {
                InstOrEdit::Edit(_) => true,
                &InstOrEdit::Inst(i) => {
                    !(v.insts[i.index()].is_remat() && out.inst_allocs(v, i)[0].is_none())
                }
            });
        let (Some(InstOrEdit::Inst(i)), None) = (code.next(), code.next()) else {
            continue;
        };
        match v.insts[Inst::index(i)].flow() {
            Flow::Deopt(exit) => {
                dest[b] = exit_of(exit);
                skip[b] = true;
            }
            Flow::Jump => jumps[b] = v.succs(b)[0].index(),
            _ => {}
        }
    }
    for b in 1..labels.len() {
        // The end of a chain of jumps; one that comes around is left alone.
        let (mut to, mut steps) = (b, 0);
        while jumps[to] != usize::MAX && steps <= labels.len() {
            (to, steps) = (jumps[to], steps + 1);
        }
        if to != b && steps <= labels.len() {
            dest[b] = dest[to];
            skip[b] = true;
        }
    }
    (dest, skip)
}
