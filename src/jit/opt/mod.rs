//! The optimizer (`jit-design.md` 9): passes over a `Func`, each verified in
//! checking builds.

pub(crate) mod gvn;
pub(crate) mod infer;
pub(crate) mod simplify;

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, BlockCall, Func, NO_SNAP};

/// Delete pure instructions whose results are unused, to a fixpoint.
pub(crate) fn dce(f: &mut Func<'_>) {
    loop {
        let uses = f.use_counts();
        let mut changed = false;
        for b in 0..f.blocks.len() {
            if f.blocks[b].dead {
                continue;
            }
            let before = f.blocks[b].insts.len();
            let insts = std::mem::take(&mut f.blocks[b].insts);
            let kept: Vec<_> = insts
                .into_iter()
                .filter(|&i| {
                    let d = &f.insts[i.idx()];
                    let removable = (d.op.is_pure() || matches!(d.op, Op::Load(_)))
                        && d.rn > 0
                        && f.results(i).all(|r| uses[r.idx()] == 0);
                    !removable
                })
                .collect();
            changed |= kept.len() != before;
            f.blocks[b].insts = kept;
        }
        if !changed {
            break;
        }
    }
}

/// Split every edge from a block with several successors to a block with
/// several predecessors, so the register allocator has a place for the
/// edge's moves.
pub(crate) fn split_critical_edges(f: &mut Func<'_>) {
    let preds = f.preds();
    let nb = f.blocks.len();
    for b in 0..nb {
        if f.blocks[b].dead {
            continue;
        }
        let Some(t) = f.terminator(Block(b as u32)) else {
            continue;
        };
        let n = f.edges(t).len();
        if n < 2 {
            continue;
        }
        for k in 0..n {
            let target = f.edges(t)[k].target;
            if preds[target.idx()].len() < 2 {
                continue;
            }
            let mid = f.new_block();
            let args = std::mem::take(&mut f.edges_mut(t)[k].args);
            let j = f.make_inst(Op::Jump, &[], None, crate::jit::ir::ops::ExitTag::Type);
            f.append(mid, j);
            f.set_edges(j, vec![BlockCall { target, args }]);
            f.edges_mut(t)[k].target = mid;
        }
    }
}

/// Snapshots referenced by no live instruction are left in place; nothing
/// reads them.
pub(crate) fn snapshot_users(f: &Func<'_>) -> Vec<bool> {
    let mut used = vec![false; f.snaps.len()];
    for bd in &f.blocks {
        if bd.dead {
            continue;
        }
        for &i in &bd.insts {
            let s = f.insts[i.idx()].snap;
            if s != NO_SNAP {
                used[s as usize] = true;
            }
        }
    }
    used
}
