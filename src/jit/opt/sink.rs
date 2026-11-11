//! Code sinking: a pure instruction whose every use, snapshot entries included,
//! is in one block it dominates moves to the start of that block when it is
//! cold, so the hot path neither computes it nor keeps it live across the arm's
//! call, or when it runs no more often: in no loop the instruction's block is
//! not in (a loop's exit edge, an arm). Into a hot block only what has at most
//! one operand that is no constant moves, so no two live ranges grow for one.

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, Func, Inst, NO_SNAP};
use crate::jit::opt::natural_loops;

pub(crate) fn sink(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    // The headers of the loops around each block.
    let mut loops: Vec<Vec<Block>> = vec![Vec::new(); f.blocks.len()];
    for (h, body) in natural_loops(f) {
        for b in body {
            if !loops[b.idx()].contains(&h) {
                loops[b.idx()].push(h);
            }
        }
    }
    let konst = |f: &Func<'_>, v| {
        matches!(
            f.def_op(v),
            Some(
                Op::KI32(_)
                    | Op::KI64(_)
                    | Op::KF64(_)
                    | Op::KVal(_)
                    | Op::KObj(_)
                    | Op::KPtr(_)
                    | Op::KB1(_)
            )
        )
    };
    let mut changed = false;
    loop {
        // The one block each value is used in, or `MANY`.
        const NONE: u32 = u32::MAX;
        const MANY: u32 = u32::MAX - 1;
        let mut used_in = vec![NONE; f.vals.len()];
        for &b in &cfg.rpo {
            for &i in f.insts_of(b) {
                f.for_each_use(i, |v| {
                    let u = &mut used_in[v.idx()];
                    *u = match *u {
                        NONE => b.0,
                        x if x == b.0 => x,
                        _ => MANY,
                    };
                });
            }
        }
        let mut moves: Vec<(Inst, Block)> = Vec::new();
        for &b in &cfg.rpo {
            if f.blocks[b.idx()].cold {
                continue;
            }
            for &i in f.insts_of(b) {
                let d = &f.insts[i.idx()];
                if !d.op.is_pure() || d.rn != 1 {
                    continue;
                }
                let u = used_in[f.result(i).idx()];
                if u == NONE || u == MANY || u == b.0 {
                    continue;
                }
                let c = Block(u);
                let fits = f.blocks[c.idx()].cold
                    || (d.snap == NO_SNAP
                        && loops[c.idx()].iter().all(|h| loops[b.idx()].contains(h))
                        && f.args(i).iter().filter(|&&a| !konst(f, a)).count() <= 1);
                // Its operands are defined in `b` or above, which dominates
                // `c` since the use there does.
                if fits && cfg.dominates(b, c) {
                    moves.push((i, c));
                }
            }
        }
        if moves.is_empty() {
            return changed;
        }
        changed = true;
        let mut moved = vec![false; f.insts.len()];
        for &(i, _) in &moves {
            moved[i.idx()] = true;
        }
        for &b in &cfg.rpo {
            f.retain(b, |_, i| !moved[i.idx()]);
        }
        // In reverse, so an operand moved with its use lands before it.
        for &(i, c) in moves.iter().rev() {
            let at = f
                .insts_of(c)
                .iter()
                .take_while(|&&x| matches!(f.op(x), Op::Resume { .. }))
                .count();
            f.insert(c, at, i);
        }
    }
}
