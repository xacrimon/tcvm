//! Array misses (W2.3): `GuardMiss(v, has)`, an element `v` nil under a
//! class with the metamethod (`has`), expands once the loops are optimized.
//! One a guard that excludes nil from `v` dominates goes, as does one such a
//! guard follows under its checkpoint, whose exit runs the load again
//! ([`covered`]); the rest become the cheapest of:
//! - the type guard on `v` of its use, as a branch whose failing arm tests
//!   nil, then the class; the guards between with the use's snapshot (the
//!   other operand's) follow the branch, as an exit there is the same exit;
//! - a guard on `has` alone where `has` is computed on the hot path anyway:
//!   outside the loop (LICM moved it), or for a guard of this form sharing
//!   it; LICM moves it as well, recording the miss at the access (E2);
//! - a nil test whose arm tests the class.
//!
//! Once LICM and GVN have run again, each arm computes the class test again
//! ([`recompute`]), so the hot copy may die: the class read there is the
//! one `has` read, as no pass moves a read across a write of what it reads.

use crate::jit::ir::ops::{ExitTag, Flags, Op};
use crate::jit::ir::types::TypeSet;
use crate::jit::ir::{Block, BlockCall, Func, Inst, List, NO_SNAP, Snap, Val};
use crate::jit::opt::natural_loops;

/// Expand every `GuardMiss`; `None` without any, else the arms' guards.
pub(crate) fn expand(f: &mut Func<'_>) -> Option<Vec<Inst>> {
    let mut guards: Vec<Inst> = Vec::new();
    let mut typed: Vec<(Val, Inst)> = Vec::new();
    for (b, bd) in f.blocks.iter().enumerate() {
        if bd.dead {
            continue;
        }
        for &i in f.insts_of(Block(b as u32)) {
            match f.op(i) {
                Op::GuardMiss => guards.push(i),
                Op::Guard(set) if !set.intersects(TypeSet::NIL) => typed.push((f.args(i)[0], i)),
                _ => {}
            }
        }
    }
    if guards.is_empty() {
        return None;
    }
    let cfg = f.cfg();
    let pos = |f: &Func<'_>, i: Inst| {
        let b = f.insts[i.idx()].block;
        f.insts_of(b).iter().position(|&x| x == i).unwrap()
    };
    let dominated = |f: &Func<'_>, g: Inst| {
        let (v, b) = (f.args(g)[0], f.insts[g.idx()].block);
        typed.iter().any(|&(x, w)| {
            let wb = f.insts[w.idx()].block;
            x == v
                && if wb == b {
                    pos(f, w) < pos(f, g)
                } else {
                    cfg.dominates(wb, b)
                }
        })
    };
    let gone: Vec<Inst> = guards
        .iter()
        .copied()
        .filter(|&g| dominated(f, g))
        .collect();
    for &g in &gone {
        let b = f.insts[g.idx()].block;
        f.retain(b, |_, i| i != g);
    }
    guards.retain(|g| !gone.contains(g));
    guards.retain(|&g| !covered(f, g, pos(f, g)));
    let loops = natural_loops(f);
    let uses = f.use_counts();
    let fusions: Vec<Option<(usize, Vec<usize>)>> = guards
        .iter()
        .map(|&g| {
            let b = f.insts[g.idx()].block;
            fusion(f, f.insts_of(b), pos(f, g), f.args(g)[0])
        })
        .collect();
    // The class tests some guard needs on the hot path.
    let mut hot: Vec<Val> = Vec::new();
    for (&g, fused) in guards.iter().zip(&fusions) {
        let (b, has) = (f.insts[g.idx()].block, f.args(g)[1]);
        let outside = match loops
            .iter()
            .filter(|(_, body)| body.contains(&b))
            .min_by_key(|(_, body)| body.len())
        {
            Some((_, body)) => f.def_block(has).is_none_or(|d| !body.contains(&d)),
            None => uses[has.idx()] > 1,
        };
        if fused.is_none() && outside && f.def_op(has) != Some(Op::KB1(true)) {
            hot.push(has);
        }
    }
    let mut arms = Vec::new();
    // Last first: a split moves what follows into a block no loop lists.
    for (g, fused) in guards.into_iter().zip(fusions).rev() {
        let b = f.insts[g.idx()].block;
        let insts: Vec<Inst> = f.insts_of(b).to_vec();
        let k = pos(f, g);
        let (v, has) = (f.args(g)[0], f.args(g)[1]);
        let miss = Snap(f.insts[g.idx()].snap);
        let blame = f.insts[g.idx()].blame;
        let mut guard = |f: &mut Func<'_>, arm: Block| {
            let x = f.make_inst(Op::GuardFalse, &[has], Some(miss), ExitTag::Bounds);
            f.insts[x.idx()].blame = blame;
            f.append(arm, x);
            arms.push(x);
        };
        if let Some((m, moved)) = fused {
            let w = insts[m];
            let Op::Guard(set) = f.op(w) else {
                unreachable!()
            };
            let (kind, tag) = (Snap(f.insts[w.idx()].snap), f.insts[w.idx()].tag);
            let cont = f.new_block();
            let fail = cold_block(f);
            let mm = cold_block(f);
            let other = cold_block(f);
            let after: Vec<Inst> = moved
                .iter()
                .map(|&x| insts[x])
                .chain(insts[m..].iter().copied())
                .collect();
            f.retain(b, |_, i| i != g && !after.contains(&i));
            for &i in &after {
                f.append(cont, i);
            }
            // `w` keeps its result, now refined where the test passed.
            f.insts[w.idx()].op = Op::Refine(set);
            // Its exit leaves before what moves after the test.
            let kind_blame = match f.insts[w.idx()].blame {
                NO_SNAP => NO_SNAP,
                _ => f.moved_blame(w, |f, v| f.def_inst(v).is_none_or(|d| !after.contains(&d))),
            };
            f.drop_exit(w);
            let c = f.make_inst(Op::IsType(set), &[v], None, ExitTag::Type);
            f.append(b, c);
            branch(f, b, f.result(c), cont, fail);
            let n = f.make_inst(Op::IsType(TypeSet::NIL), &[v], None, ExitTag::Type);
            f.append(fail, n);
            branch(f, fail, f.result(n), mm, other);
            // A nil the class does not answer fails the use's guard too: one
            // exit for both.
            guard(f, mm);
            let j = f.make_inst(Op::Jump, &[], None, ExitTag::Type);
            f.append(mm, j);
            f.set_edges(j, &[edge(other)]);
            let d = f.make_inst(Op::Deopt, &[], Some(kind), tag);
            f.insts[d.idx()].blame = kind_blame;
            f.append(other, d);
        } else if hot.contains(&has) {
            let x = f.make_inst(Op::GuardFalse, &[has], Some(miss), ExitTag::Bounds);
            f.insts[x.idx()].blame = blame;
            f.retain(b, |_, i| i != g);
            f.insert(b, k, x);
        } else {
            let cont = f.new_block();
            let mm = cold_block(f);
            f.retain(b, |_, i| !insts[k..].contains(&i));
            for &i in &insts[k + 1..] {
                f.append(cont, i);
            }
            let n = f.make_inst(Op::IsType(TypeSet::NIL), &[v], None, ExitTag::Type);
            f.append(b, n);
            branch(f, b, f.result(n), mm, cont);
            guard(f, mm);
            let j = f.make_inst(Op::Jump, &[], None, ExitTag::Type);
            f.append(mm, j);
            f.set_edges(j, &[edge(cont)]);
        }
    }
    Some(arms)
}

/// Whether a type guard on its element excluding nil takes miss guard `g`'s
/// place, `g` at `k` of its block: one after it on the only path on, with
/// its snapshot (a shared checkpoint, so its exit runs the load again) and
/// no side effect between. That guard's blame then names the element and
/// the table, for the exit to tell a miss from another type (`jit::on_exit`:
/// the interpreter records the miss as it runs the load again), and `g` goes.
fn covered(f: &mut Func<'_>, g: Inst, k: usize) -> bool {
    use crate::jit::ir::{BLAME_TABLE, BLAME_TESTED};
    let (v, t) = (f.args(g)[0], f.args(g)[2]);
    let snap = f.insts[g.idx()].snap;
    let (mut b, mut from) = (f.insts[g.idx()].block, k + 1);
    let w = 'found: loop {
        for &i in &f.insts_of(b)[from..] {
            let op = f.op(i);
            if let Op::Guard(set) = op
                && f.args(i)[0] == v
                && !set.intersects(TypeSet::NIL)
            {
                break 'found i;
            }
            if op == Op::Jump
                && let [e] = f.edges(i)
                && f.cfg().preds(e.target) == [b]
            {
                (b, from) = (e.target, 0);
                continue 'found;
            }
            if op.side_effect() || op.is_terminator() {
                return false;
            }
        }
        return false;
    };
    let d = f.insts[w.idx()];
    if d.blame == NO_SNAP
        || d.tag != ExitTag::Type
        || !f.same_snap(d.snap, snap)
        || f.blame_entries(w).iter().any(|e| e.0 == BLAME_TABLE)
    {
        return false;
    }
    let at = f.exit_site(w);
    let from = f.snap_pool.len();
    let mut ents = f.entries(d.blame).to_vec();
    ents.extend([(BLAME_TESTED, v), (BLAME_TABLE, t)]);
    f.snap_pool.extend_from_slice(&ents);
    let s = f.finish_snap(at.frame, at.pc, crate::jit::ir::ExitKind::Before, from);
    f.insts[w.idx()].blame = s.0;
    let gb = f.insts[g.idx()].block;
    f.retain(gb, |_, i| i != g);
    true
}

/// The type guard on `v` excluding nil that the miss guard at `k` can take
/// its test from, at `m`, and what moves after the test: the guards between,
/// which must share `m`'s snapshot, and the pure instructions using them.
/// Reads between stay before the test, so they must precede those guards.
fn fusion(f: &Func<'_>, insts: &[Inst], k: usize, v: Val) -> Option<(usize, Vec<usize>)> {
    let mut moved: Vec<usize> = Vec::new();
    let mut exits: Vec<Inst> = Vec::new();
    let mut results: Vec<Val> = Vec::new();
    for (m, &i) in insts.iter().enumerate().skip(k + 1) {
        let op = f.op(i);
        if let Op::Guard(set) = op
            && f.args(i)[0] == v
        {
            let snap = f.insts[i.idx()].snap;
            let ok = !set.intersects(TypeSet::NIL)
                && f.insts[i.idx()].tag != ExitTag::Entry
                && exits.iter().all(|&x| f.insts[x.idx()].snap == snap)
                && f.snap_entries(i).iter().all(|e| !results.contains(&e.1));
            return ok.then_some((m, moved));
        }
        let e = op.effects();
        let uses_moved = f.args(i).iter().any(|a| results.contains(a));
        if e.is_pure() && !uses_moved {
            continue;
        }
        let tests = e.flags == Flags::MAY_DEOPT && !e.reads_memory() && !e.writes();
        let reads = !e.writes()
            && !e.flags.intersects(
                Flags::MAY_DEOPT
                    | Flags::MAY_ALLOC
                    | Flags::SIDE_EFFECT
                    | Flags::TERMINATOR
                    | Flags::TAILOUT,
            );
        if reads && !uses_moved && exits.is_empty() {
            continue;
        }
        if e.is_pure() || tests && op != Op::GuardMiss {
            moved.push(m);
            if tests {
                exits.push(i);
            }
            results.extend(f.results(i));
            continue;
        }
        return None;
    }
    None
}

/// Compute the class test of each of `arms`' guards GVN left again before
/// it, where the test is a hot one.
pub(crate) fn recompute(f: &mut Func<'_>, arms: &[Inst]) {
    for &x in arms {
        let b = f.insts[x.idx()].block;
        let Some(at) = f.insts_of(b).iter().position(|&i| i == x) else {
            continue;
        };
        let has = f.args(x)[0];
        if f.def_block(has).is_none_or(|d| f.blocks[d.idx()].cold) {
            continue;
        }
        let again = class_again(f, has, b, at);
        f.args_mut(x)[0] = again;
    }
}

/// `has` computed again in `b` at `at`, where it is the class test
/// `build::tables` emits.
fn class_again(f: &mut Func<'_>, has: Val, b: Block, at: usize) -> Val {
    let Some(d) = f.def_inst(has) else {
        return has;
    };
    let op = f.op(d);
    if !matches!(op, Op::TabHasMm(_)) {
        return has;
    }
    let t = f.args(d)[0];
    let c = f.make_inst(op, &[t], None, ExitTag::Type);
    f.insert(b, at, c);
    f.result(c)
}

fn cold_block(f: &mut Func<'_>) -> Block {
    let b = f.new_block();
    f.blocks[b.idx()].cold = true;
    b
}

fn edge(target: Block) -> BlockCall {
    BlockCall {
        target,
        args: List::default(),
    }
}

/// End `b` with a branch on `c` to `t` or `e`.
fn branch(f: &mut Func<'_>, b: Block, c: Val, t: Block, e: Block) {
    let br = f.make_inst(Op::Br, &[c], None, ExitTag::Type);
    f.append(b, br);
    f.set_edges(br, &[edge(t), edge(e)]);
}
