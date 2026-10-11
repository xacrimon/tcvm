//! Loop-invariant code motion (9.5): instructions whose operands come from
//! outside a loop, and whose memory the loop does not write, move to its
//! preheader. Pure instructions move from anywhere in the loop. Exiting
//! instructions and loads move only from blocks every entered iteration
//! runs, and a load or a memory guard only until an exiting instruction
//! stays: it may rely on the guards before it (a table's shape for a slot's
//! offset). A moved exit leaves `Before` the loop header with the
//! preheader's frame.

use crate::jit::ir::ops::{Flags, Mem, Op};
use crate::jit::ir::{Block, Func, Inst, Val};
use crate::jit::opt::calls::rematerializable;
use crate::jit::opt::natural_loops;

pub(crate) fn licm(f: &mut Func<'_>) -> bool {
    // One entry per header, innermost first, so what leaves an inner loop
    // may leave the outer one too.
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
        changed |= hoist(f, h, &body);
    }
    changed
}

fn hoist(f: &mut Func<'_>, h: Block, body: &[Block]) -> bool {
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
    // The loop's memory writes: whole classes, and keys within a class.
    let mut wclass = Mem::empty();
    let mut wkeys: Vec<(Mem, u32)> = Vec::new();
    for &b in body {
        for &i in f.insts_of(b) {
            let op = f.op(i);
            let w = op.effects().writes;
            match op.mem_key() {
                Some(k) if !w.is_empty() => wkeys.push((w, k)),
                _ => wclass |= w,
            }
        }
    }
    // Blocks every entered iteration runs: they dominate each latch and
    // each block leaving the loop.
    let ends: Vec<Block> = body
        .iter()
        .copied()
        .filter(|&b| f.succs(b).any(|s| s == h || !in_body[s.idx()]))
        .collect();
    let spine: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|&b| in_body[b.idx()] && ends.iter().all(|&e| cfg.dominates(b, e)))
        .collect();
    // The header's own exit, for a moved one, with the header's parameters
    // as the preheader passes them; none, and exits stay.
    let entry_snap = f.insts_of(h).iter().find_map(|&i| {
        (f.op(i) == Op::GcCheck).then(|| crate::jit::ir::Snap(f.insts[i.idx()].snap))
    });
    let params: Vec<Val> = f.params(h).to_vec();
    let edge_args: Vec<Val> = f.edge_args(t, k as usize).to_vec();
    let memory_kept = |op: Op| {
        let r = op.effects().reads;
        !r.intersects(wclass)
            && match op.mem_key() {
                Some(k) => !wkeys.contains(&(r, k)),
                None => !wkeys.iter().any(|&(c, _)| r.intersects(c)),
            }
    };
    // Across a call only what can be computed again may live (R6, 9.8).
    let calls = body.iter().any(|&b| {
        f.terminator(b)
            .is_some_and(|i| matches!(f.op(i), Op::Call { .. } | Op::CallMm { .. }))
    });
    let mut moves: Vec<Inst> = Vec::new();
    let mut moved = vec![false; f.insts.len()];
    // Pure instructions anywhere; the spine's exits and loads in order.
    let mut ordered = true;
    for &b in cfg.rpo.iter().filter(|&&b| in_body[b.idx()]) {
        let on_spine = spine.contains(&b);
        for &i in f.insts_of(b) {
            let op = f.op(i);
            if op.is_terminator() || f.insts[i.idx()].block != b {
                continue;
            }
            let e = op.effects();
            let exits = e.has(Flags::MAY_DEOPT);
            let reads_memory = e.reads_memory();
            let movable = match op {
                // Slots are loaded only where a region (re)starts.
                Op::Load(_) | Op::GcCheck => false,
                _ if e.is_pure() => true,
                _ if !e.movable() || e.writes() || e.has(Flags::MAY_ALLOC) => false,
                // A test of values alone needs no earlier guard; a read of
                // memory may (a slot's offset is right for one shape).
                _ => {
                    (exits || reads_memory)
                        && on_spine
                        && (ordered || !reads_memory)
                        && memory_kept(op)
                        && (!exits || entry_snap.is_some())
                }
            };
            let ok = movable
                && f.args(i).iter().all(|&a| {
                    f.def_inst(a).is_some_and(|d| moved[d.idx()])
                        || f.def_block(a).is_none_or(|d| !in_body[d.idx()])
                })
                && (!calls
                    || (op.is_pure() || matches!(op, Op::Guard(_)))
                        && f.args(i).iter().all(|&a| rematerializable(f, a)));
            if ok {
                moved[i.idx()] = true;
                moves.push(i);
            } else if exits && on_spine && op != Op::GcCheck {
                // What it tests may be what a later load relies on.
                ordered = false;
            }
        }
    }
    if moves.is_empty() {
        return false;
    }
    // A preheader of the loop's own: split an edge from a branch.
    let mut pre = f.insts[t.idx()].block;
    if f.edges(t).len() > 1 {
        let e = f.edges(t)[k as usize];
        let mid = f.new_block();
        let j = f.make_inst(Op::Jump, &[], None, crate::jit::ir::ops::ExitTag::Type);
        f.append(mid, j);
        f.set_edges(j, &[e]);
        f.set_target(t, k as usize, mid);
        f.set_edge_args(t, k as usize, &[]);
        pre = mid;
    }
    let snap = entry_snap.map(|s| {
        f.map_snap(s, |v| match params.iter().position(|&p| p == v) {
            Some(n) => edge_args[n],
            None => v,
        })
    });
    for &i in &moves {
        if f.op(i).may_deopt() {
            f.insts[i.idx()].snap = snap.unwrap().0;
        }
    }
    for &b in body {
        f.retain(b, |_, i| !moved[i.idx()]);
    }
    for &i in &moves {
        f.insert_before_term(pre, i);
    }
    true
}
