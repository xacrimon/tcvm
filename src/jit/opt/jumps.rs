//! Jump threading: a block that only jumps on is left out, each edge into it
//! going where the jump went with the arguments it would pass; a block whose
//! jump is its successor's only entry takes the successor's instructions.
//! Inlined callees' returns funnel through such blocks, and folds leave arms
//! empty. A branch's edge into a join carries its own moves
//! (`ra::Output::edge_edits`), so nothing stays for them.

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, BlockCall, CfgInfo, Func, Inst, Val, ValDef};

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
        // A call keeps its resume block. A branch this round made a jump has
        // lost the edges `cfg` names past its first; the next round sees it.
        if !matches!(f.op(src), Op::Jump | Op::Br)
            || f.edges(src).get(k).is_none_or(|e| e.target != b)
        {
            continue;
        }
        let mut args: Vec<Val> = (f.vl(e.args).iter())
            .map(|&v| match f.vals[v.idx()].def {
                ValDef::Param(pb, i) if pb == b => f.edge_args(src, k)[i as usize],
                _ => v,
            })
            .collect();
        npreds[b.idx()] -= 1;
        any = true;
        // A branch whose other edge reaches `s` too decides only arguments:
        // a select for each the edges disagree on, and a jump.
        if let Some(j) = (0..f.edges(src).len()).find(|&j| j != k && f.edges(src)[j].target == s) {
            let (c, tag, sb) = (
                f.args(src)[0],
                f.insts[src.idx()].tag,
                f.insts[src.idx()].block,
            );
            for (n, a) in args.iter_mut().enumerate() {
                let o = f.edge_args(src, j)[n];
                if *a != o {
                    let (x, y) = if k == 0 { (*a, o) } else { (o, *a) };
                    let sel = f.make_inst(Op::Select, &[c, x, y], None, tag);
                    f.insert_before_term(sb, sel);
                    *a = f.result(sel);
                }
            }
            f.insts[src.idx()].op = Op::Jump;
            f.insts[src.idx()].an = 0;
            let args = f.new_vlist(&args);
            f.set_edges(src, &[BlockCall { target: s, args }]);
            continue;
        }
        f.set_target(src, k, s);
        f.set_edge_args(src, k, &args);
        npreds[s.idx()] += 1;
    }
    if npreds[b.idx()] == 0 {
        f.blocks[b.idx()].dead = true;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::ir::ops::ExitTag;
    use crate::jit::ir::{List, RegionMeta};

    fn term(f: &mut Func<'_>, b: Block, op: Op, args: &[Val], to: &[Block]) {
        let i = f.make_inst(op, args, None, ExitTag::Type);
        f.append(b, i);
        let calls: Vec<BlockCall> = (to.iter())
            .map(|&target| BlockCall {
                target,
                args: List::default(),
            })
            .collect();
        f.set_edges(i, &calls);
    }

    /// `x: br c, b0, h` with `b0: jump h` and a jump-only loop header `h`:
    /// threading `b0` first turns the branch into a jump, which the round's
    /// edges into `h` still name by its second edge.
    #[test]
    fn branch_made_a_jump_earlier_in_the_round() {
        let mut f = Func::new(RegionMeta {
            entry_pc: 0,
            loop_entry: false,
            params: 0,
        });
        let [e, h, x, b0] = [(); 4].map(|_| f.new_block());
        f.entry = e;
        let c = f.make_inst(Op::KB1(true), &[], None, ExitTag::Type);
        f.append(e, c);
        let c = f.result(c);
        term(&mut f, e, Op::Jump, &[], &[h]);
        term(&mut f, h, Op::Jump, &[], &[x]);
        let k = f.make_inst(Op::KI64(1), &[], None, ExitTag::Type);
        f.append(x, k);
        term(&mut f, x, Op::Br, &[c], &[b0, h]);
        term(&mut f, b0, Op::Jump, &[], &[h]);
        assert!(thread(&mut f));
        crate::jit::ir::verify::verify(&f).unwrap();
        let t = f.terminator(x).unwrap();
        assert_eq!(f.op(t), Op::Jump);
        assert_eq!(f.edges(t)[0].target, x);
    }
}
