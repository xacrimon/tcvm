//! Jump threading: a block that only jumps on is left out, each edge into it
//! going where the jump went with the arguments it would pass, unless that
//! edge would then come from a branch into a join (a critical edge, which
//! the allocator has no place to put moves on; `split_critical_edges`); a
//! block whose jump is its successor's only entry takes the successor's
//! instructions. Inlined callees' returns funnel through such blocks, and
//! folds leave arms empty.

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Val, ValDef};

pub(crate) fn thread(f: &mut Func<'_>) -> bool {
    let mut changed = false;
    loop {
        let cfg = f.cfg();
        let mut npreds: Vec<u32> = (0..f.blocks.len())
            .map(|b| cfg.preds(Block(b as u32)).len() as u32)
            .collect();
        let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
        let uses = f.use_counts();
        let mut round = false;
        let mut merged = false;
        // Successors first: a chain goes in one round.
        for &b in cfg.rpo.iter().rev() {
            let bd = f.blocks[b.idx()];
            let Some(t) = f.terminator(b).filter(|&t| f.op(t) == Op::Jump) else {
                continue;
            };
            if bd.dead || b == f.entry {
                continue;
            }
            // Its parameters go on as its arguments; used elsewhere, they
            // are definitions the block stays for.
            let passed_on = |p: Val| {
                uses[p.idx()] == f.edge_args(t, 0).iter().filter(|&&v| v == p).count() as u32
            };
            if f.insts_of(b).len() == 1 && !bd.resume && f.params(b).iter().all(|&p| passed_on(p)) {
                round |= thread_block(f, &cfg, &mut npreds, b, t);
                if f.blocks[b.idx()].dead {
                    continue;
                }
            }
            let s = f.edges(t)[0].target;
            let sd = f.blocks[s.idx()];
            let same_reps = (f.params(s).iter().zip(f.edge_args(t, 0)))
                .all(|(&p, &a)| f.ty(p).rep == f.ty(a).rep);
            if s == b
                || s == f.entry
                || npreds[s.idx()] != 1
                || sd.resume
                || sd.cold != bd.cold
                || !same_reps
            {
                continue;
            }
            for (&p, &a) in f.params(s).iter().zip(f.edge_args(t, 0)) {
                map[p.idx()] = a;
            }
            // The jump is last.
            f.blocks[b.idx()].insts.len -= 1;
            for i in f.insts_of(s).to_vec() {
                f.append(b, i);
            }
            f.blocks[s.idx()].dead = true;
            merged = true;
        }
        if merged {
            f.apply_replacements(&mut map);
        }
        if !(round || merged) {
            return changed;
        }
        f.cfg_changed();
        changed = true;
    }
}

/// Edges into `b`, which only jumps, go where it does; `b` dies with its last.
fn thread_block(f: &mut Func<'_>, cfg: &CfgInfo, npreds: &mut [u32], b: Block, t: Inst) -> bool {
    let e = f.edges(t)[0];
    let s = e.target;
    if s == b {
        return false;
    }
    let mut any = false;
    for &(src, k) in cfg.incoming(b) {
        let k = k as usize;
        // A call keeps its resume block. From a branch only where the join
        // gains no second entry: `b` dies and was its one.
        if !matches!(f.op(src), Op::Jump | Op::Br)
            || f.edges(src)[k].target != b
            || (f.edges(src).len() > 1 && !(npreds[s.idx()] == 1 && npreds[b.idx()] == 1))
        {
            continue;
        }
        let args: Vec<Val> = (f.vl(e.args).iter())
            .map(|&v| match f.vals[v.idx()].def {
                ValDef::Param(pb, i) if pb == b => f.edge_args(src, k)[i as usize],
                _ => v,
            })
            .collect();
        f.set_target(src, k, s);
        f.set_edge_args(src, k, &args);
        npreds[b.idx()] -= 1;
        npreds[s.idx()] += 1;
        any = true;
    }
    if npreds[b.idx()] == 0 {
        f.blocks[b.idx()].dead = true;
    }
    any
}
