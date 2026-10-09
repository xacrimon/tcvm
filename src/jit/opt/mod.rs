//! The optimizer (`jit-design.md` 9): passes over a `Func`, each verified in
//! checking builds.

pub(crate) mod gvn;
pub(crate) mod infer;
pub(crate) mod peel;
pub(crate) mod simplify;
pub(crate) mod speculate;

use crate::jit::ir::ops::Op;
use crate::jit::ir::verify::{dominates, dominators};
use crate::jit::ir::{Block, BlockCall, Func, NO_SNAP};

/// Whether some path from the entry does work in compiled code before it
/// leaves: a loop's back edge, a return from a function entry, or an exit a
/// recompile may widen (inside a loop, for a loop entry, whose exit and
/// return happen once per run of the loop); a call continues in its resume
/// block. A region whose every path ends otherwise only adds an exit to the
/// interpreter's work.
pub(crate) fn completes(f: &Func<'_>) -> bool {
    let rpo = f.rpo();
    let idom = dominators(f, &rpo, &f.preds());
    let loop_entry = f.meta.loop_entry;
    let mut in_loop = vec![false; f.blocks.len()];
    for (_, body) in natural_loops(f) {
        for b in body {
            in_loop[b.idx()] = true;
        }
    }
    let mut done = vec![false; f.blocks.len()];
    // Successors before predecessors, but for back edges, which complete.
    for &b in rpo.iter().rev() {
        let Some(t) = f.terminator(b) else {
            continue;
        };
        let d = &f.insts[t.idx()];
        done[b.idx()] = match d.op {
            Op::Return { .. } => !loop_entry,
            Op::Deopt => d.tag.widenable() && (in_loop[b.idx()] || !loop_entry),
            _ => f.succs(b).any(|s| done[s.idx()] || dominates(&idom, s, b)),
        };
    }
    done[f.entry.idx()]
}

/// Each back edge's header and the blocks of its natural loop.
pub(crate) fn natural_loops(f: &Func<'_>) -> Vec<(Block, Vec<Block>)> {
    let rpo = f.rpo();
    let preds = f.preds();
    let idom = dominators(f, &rpo, &preds);
    let mut loops = Vec::new();
    for &t in &rpo {
        for h in f.succs(t) {
            if !dominates(&idom, h, t) {
                continue;
            }
            let mut body = vec![false; f.blocks.len()];
            body[h.idx()] = true;
            let mut work = vec![t];
            while let Some(b) = work.pop() {
                if !body[b.idx()] {
                    body[b.idx()] = true;
                    work.extend(preds[b.idx()].iter().copied());
                }
            }
            let blocks = (0..f.blocks.len() as u32)
                .map(Block)
                .filter(|b| body[b.idx()])
                .collect();
            loops.push((h, blocks));
        }
    }
    loops
}

/// Drop the collector checks of loop headers whose loops neither box an
/// integer (which may allocate) nor call out.
pub(crate) fn prune_gc_checks(f: &mut Func<'_>) {
    use crate::jit::ir::types::Rep;
    let mut keep = vec![false; f.blocks.len()];
    for (h, body) in natural_loops(f) {
        let allocates = body.iter().any(|b| {
            f.blocks[b.idx()].insts.iter().any(|&i| match f.op(i) {
                Op::Box => f.ty(f.args(i)[0]).rep == Rep::I64,
                Op::Call { .. } => true,
                op => op.effects().has(crate::jit::ir::ops::Effects::MAY_ALLOC),
            })
        });
        keep[h.idx()] |= allocates;
    }
    for b in 0..f.blocks.len() {
        if keep[b] {
            continue;
        }
        f.blocks[b]
            .insts
            .retain(|&i| f.insts[i.idx()].op != Op::GcCheck);
    }
}

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
