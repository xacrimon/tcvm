//! Bounds checks against a loop's range (E2): a `FORLOOP_I` counter walks
//! from its entry value to `last` by an invariant step exactly (`IAddNo`
//! under `idx != last`), so an array guard `idx < n` (unsigned) with `n`
//! from outside the loop holds on every iteration when it holds at both
//! ends, which the preheader checks once. The checks exit `Before` the loop header and
//! record the access's miss, so the next compile keeps the guard.

use crate::jit::ir::ops::{Cc, ExitTag, Op};
use crate::jit::ir::types::Rep;
use crate::jit::ir::{Block, Func, Inst, Val};
use crate::jit::opt::calls::same_reg;
use crate::jit::opt::natural_loops;

pub(crate) fn loop_bounds(f: &mut Func<'_>) -> bool {
    let mut loops: Vec<(Block, Vec<Block>)> = Vec::new();
    for (h, body) in natural_loops(f) {
        match loops.iter_mut().find(|(x, _)| *x == h) {
            Some((_, b)) => {
                for x in body {
                    if !b.contains(&x) {
                        b.push(x);
                    }
                }
            }
            None => loops.push((h, body)),
        }
    }
    loops.sort_by_key(|(_, b)| b.len());
    let mut changed = false;
    for (h, body) in loops {
        changed |= hoist_bounds(f, h, &body);
    }
    changed
}

/// A loop counter: its header parameter's index and its last value.
struct Counter {
    param: usize,
    last: Val,
}

/// The counters of the loop at `h`: parameters every back edge passes
/// `IAddNo(p, step)`, `step` invariant, on the edge where `p != last`.
fn counters(f: &Func<'_>, h: Block, in_body: &[bool]) -> Vec<Counter> {
    let cfg = f.cfg();
    let mut out = Vec::new();
    'params: for (n, &p) in f.params(h).iter().enumerate() {
        if f.ty(p).rep != Rep::I32 {
            continue;
        }
        let mut found: Option<(Val, Val)> = None;
        for &(t, k) in cfg.incoming(h) {
            if !in_body[f.insts[t.idx()].block.idx()] {
                continue;
            }
            let a = f.edge_args(t, k as usize)[n];
            let Some(d) = f.def_inst(a) else {
                continue 'params;
            };
            if f.op(d) != Op::IAddNo || same_reg(f, f.args(d)[0]) != p {
                continue 'params;
            }
            let step = f.args(d)[1];
            if f.def_block(step).is_some_and(|b| in_body[b.idx()]) {
                continue 'params;
            }
            // The add's block is the taken side of `p != last`.
            let inc = cfg.incoming(f.insts[d.idx()].block);
            let [(br, 0)] = inc[..] else {
                continue 'params;
            };
            if f.op(br) != Op::Br {
                continue 'params;
            }
            let c = f.args(br)[0];
            let Some(ci) = f.def_inst(c) else {
                continue 'params;
            };
            let [x, last] = f.args(ci)[..] else {
                continue 'params;
            };
            if f.op(ci) != Op::ICmp(Cc::Ne)
                || same_reg(f, x) != p
                || f.def_block(last).is_some_and(|b| in_body[b.idx()])
            {
                continue 'params;
            }
            match found {
                None => found = Some((step, last)),
                Some(s) if s == (step, last) => {}
                Some(_) => continue 'params,
            }
        }
        if let Some((_, last)) = found {
            out.push(Counter { param: n, last });
        }
    }
    out
}

fn hoist_bounds(f: &mut Func<'_>, h: Block, body: &[Block]) -> bool {
    let mut in_body = vec![false; f.blocks.len()];
    for &b in body {
        in_body[b.idx()] = true;
    }
    let cfg = f.cfg();
    let outside: Vec<(Inst, u32)> = cfg
        .incoming(h)
        .iter()
        .copied()
        .filter(|&(t, _)| !in_body[f.insts[t.idx()].block.idx()])
        .collect();
    let [(t, k)] = outside[..] else {
        return false;
    };
    if !matches!(f.op(t), Op::Jump | Op::Br) {
        return false;
    }
    let Some(entry_snap) = f.insts_of(h).iter().find_map(|&i| {
        (f.op(i) == Op::GcCheck).then(|| crate::jit::ir::Snap(f.insts[i.idx()].snap))
    }) else {
        return false;
    };
    let ivs = counters(f, h, &in_body);
    if ivs.is_empty() {
        return false;
    }
    let params: Vec<Val> = f.params(h).to_vec();
    // Each guard `idx < n` of a counter, with the access's pc.
    let mut found: Vec<(Inst, usize, Val, u32)> = Vec::new();
    for &b in body {
        for &i in f.insts_of(b) {
            if f.op(i) != Op::GuardTrue || f.insts[i.idx()].tag != ExitTag::Bounds {
                continue;
            }
            let Some(ci) = f.def_inst(f.args(i)[0]) else {
                continue;
            };
            let [x, n] = f.args(ci)[..] else {
                continue;
            };
            if f.op(ci) != Op::ICmp(Cc::Lo) || f.def_block(n).is_some_and(|b| in_body[b.idx()]) {
                continue;
            }
            let x = same_reg(f, x);
            let Some(c) = ivs.iter().position(|iv| params[iv.param] == x) else {
                continue;
            };
            let pc = f.snaps[f.insts[i.idx()].snap as usize].pc;
            found.push((i, c, n, pc));
        }
    }
    if found.is_empty() {
        return false;
    }
    let edge_args: Vec<Val> = f.edge_args(t, k as usize).to_vec();
    let mut pre = f.insts[t.idx()].block;
    if f.edges(t).len() > 1 {
        let e = f.edges(t)[k as usize];
        let mid = f.new_block();
        f.blocks[mid.idx()].cold = f.blocks[pre.idx()].cold;
        let j = f.make_inst(Op::Jump, &[], None, ExitTag::Type);
        f.append(mid, j);
        f.set_edges(j, &[e]);
        f.set_target(t, k as usize, mid);
        f.set_edge_args(t, k as usize, &[]);
        pre = mid;
    }
    let snap = f.map_snap(entry_snap, |v| match params.iter().position(|&p| p == v) {
        Some(n) => edge_args[n],
        None => v,
    });
    let mut gone = vec![false; f.insts.len()];
    for (g, c, n, pc) in found {
        let iv = &ivs[c];
        for x in [edge_args[iv.param], iv.last] {
            let cmp = f.make_inst(Op::ICmp(Cc::Lo), &[x, n], None, ExitTag::Type);
            f.insert_before_term(pre, cmp);
            let r = f.result(cmp);
            let guard = f.make_inst(Op::GuardTrue, &[r], Some(snap), ExitTag::Bounds);
            f.insert_before_term(pre, guard);
            f.exit_sites.insert(guard, pc);
        }
        gone[g.idx()] = true;
    }
    for &b in body {
        f.retain(b, |_, i| !gone[i.idx()]);
    }
    true
}
