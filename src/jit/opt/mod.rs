//! The optimizer (`jit-design.md` 9): passes over a `Func`, each verified in
//! checking builds.

pub(crate) mod calls;
pub(crate) mod gvn;
pub(crate) mod infer;
pub(crate) mod peel;
pub(crate) mod simplify;
pub(crate) mod speculate;

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, Func, Inst};

/// Whether some path from the entry does work in compiled code before it
/// leaves: a loop's back edge, a return from a function entry, or an exit a
/// recompile may widen (inside a loop, for a loop entry, whose exit and
/// return happen once per run of the loop); a call continues in its resume
/// block. A region whose every path ends otherwise only adds an exit to the
/// interpreter's work.
pub(crate) fn completes(f: &Func<'_>) -> bool {
    let cfg = f.cfg();
    let loop_entry = f.meta.loop_entry;
    let mut in_loop = vec![false; f.blocks.len()];
    for (_, body) in natural_loops(f) {
        for b in body {
            in_loop[b.idx()] = true;
        }
    }
    let mut done = vec![false; f.blocks.len()];
    // Successors before predecessors, but for back edges, which complete.
    for &b in cfg.rpo.iter().rev() {
        let Some(t) = f.terminator(b) else {
            continue;
        };
        let d = &f.insts[t.idx()];
        done[b.idx()] = match d.op {
            Op::Return { .. } => !loop_entry,
            Op::Deopt => d.tag.widenable() && (in_loop[b.idx()] || !loop_entry),
            _ => f.succs(b).any(|s| done[s.idx()] || cfg.dominates(s, b)),
        };
    }
    done[f.entry.idx()]
}

/// Each back edge's header and the blocks of its natural loop.
pub(crate) fn natural_loops(f: &Func<'_>) -> Vec<(Block, Vec<Block>)> {
    let cfg = f.cfg();
    let mut loops = Vec::new();
    let mut work = Vec::new();
    for &t in &cfg.rpo {
        for h in f.succs(t) {
            if !cfg.dominates(h, t) {
                continue;
            }
            let mut body = vec![false; f.blocks.len()];
            body[h.idx()] = true;
            work.push(t);
            while let Some(b) = work.pop() {
                if !body[b.idx()] {
                    body[b.idx()] = true;
                    work.extend_from_slice(cfg.preds(b));
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

/// Keep one collector check per section, as LuaJIT does (`asm_gc_check`):
/// at the header of each loop that boxes an integer (which may allocate),
/// allocates or calls out, and at the entry when code outside every loop
/// does; garbage then waits at most one iteration or one entry.
pub(crate) fn prune_gc_checks(f: &mut Func<'_>) {
    use crate::jit::ir::types::Rep;
    let allocates = |f: &Func<'_>, b: Block| {
        f.insts_of(b).iter().any(|&i| match f.op(i) {
            Op::Box => f.ty(f.args(i)[0]).rep == Rep::I64,
            Op::Call { .. } => true,
            op => op.effects().has(crate::jit::ir::ops::Effects::MAY_ALLOC),
        })
    };
    let mut keep = vec![false; f.blocks.len()];
    let mut in_loop = vec![false; f.blocks.len()];
    for (h, body) in natural_loops(f) {
        for &b in &body {
            in_loop[b.idx()] = true;
        }
        keep[h.idx()] |= body.iter().any(|&b| allocates(f, b));
    }
    keep[f.entry.idx()] = f
        .cfg()
        .rpo
        .iter()
        .any(|&b| !in_loop[b.idx()] && allocates(f, b));
    for b in 0..f.blocks.len() {
        if keep[b] {
            continue;
        }
        f.retain(Block(b as u32), |f, i| f.op(i) != Op::GcCheck);
    }
}

/// Delete pure instructions whose results are unused, and then those only
/// they used.
pub(crate) fn dce(f: &mut Func<'_>) {
    let mut uses = f.use_counts();
    let removable = |f: &Func<'_>, uses: &[u32], i: Inst| {
        let d = &f.insts[i.idx()];
        (d.op.is_pure() || matches!(d.op, Op::Load(_)))
            && d.rn > 0
            && f.results(i).all(|r| uses[r.idx()] == 0)
    };
    let mut dead = vec![false; f.insts.len()];
    let mut work: Vec<Inst> = Vec::new();
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead {
            continue;
        }
        for &i in f.insts_of(Block(b as u32)) {
            if removable(f, &uses, i) {
                dead[i.idx()] = true;
                work.push(i);
            }
        }
    }
    if work.is_empty() {
        return;
    }
    while let Some(i) = work.pop() {
        f.for_each_use(i, |v| uses[v.idx()] -= 1);
        for k in 0..f.insts[i.idx()].an as usize {
            let a = f.args(i)[k];
            if let Some(d) = f.def_inst(a)
                && !dead[d.idx()]
                && f.insts[d.idx()].block.0 != u32::MAX
                && !f.blocks[f.insts[d.idx()].block.idx()].dead
                && removable(f, &uses, d)
            {
                dead[d.idx()] = true;
                work.push(d);
            }
        }
    }
    for b in 0..f.blocks.len() {
        f.retain(Block(b as u32), |_, i| !dead[i.idx()]);
    }
}

/// Split every edge from a block with several successors to a block with
/// several predecessors, so the register allocator has a place for the
/// edge's moves.
pub(crate) fn split_critical_edges(f: &mut Func<'_>) {
    let cfg = f.cfg();
    for &b in &cfg.rpo {
        let Some(t) = f.terminator(b) else {
            continue;
        };
        let n = f.edges(t).len();
        if n < 2 {
            continue;
        }
        for k in 0..n {
            let e = f.edges(t)[k];
            if cfg.preds(e.target).len() < 2 {
                continue;
            }
            let mid = f.new_block();
            let j = f.make_inst(Op::Jump, &[], None, crate::jit::ir::ops::ExitTag::Type);
            f.append(mid, j);
            f.set_edges(j, &[e]);
            f.set_target(t, k, mid);
            f.set_edge_args(t, k, &[]);
        }
    }
}
