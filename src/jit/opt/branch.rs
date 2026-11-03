//! Branch elimination (W1 5.4, after Turboshaft's
//! `branch-elimination-reducer.h`): a merge block ending in a `br` whose
//! outcome an incoming edge decides is copied into that edge's source, the
//! copy's `br` a jump. The edge decides it when the condition is known
//! there (a dominating branch edge or guard, or the edge itself), or is a
//! test of the block's parameters its arguments answer.

use crate::jit::FastMap;
use crate::jit::ir::ops::Op;
use crate::jit::ir::types::TypeSet;
use crate::jit::ir::{Block, BlockCall, Func, Inst, NO_SNAP, Snap, Val};
use crate::jit::opt::DomTree;

/// Merge blocks of more instructions than this (the terminator counted,
/// parameters not: the copies drop them) stay: Turboshaft's
/// `kMaxOpCountForCloning`.
const CLONE_LIMIT: usize = 13;

/// A branch on a select of two booleans it alone uses branches on the
/// select's condition, then in each arm on the boolean chosen there: no
/// select, and a pure compare of the block only the select used moves to
/// its arm, where the branch takes its flags. The float `for` loop's test,
/// `select(step > 0, up, down)`, so branches on the step's sign, which
/// never changes, then compares once. Not where it enters a loop: the
/// loop would have two entries, which LICM and `range` take none of.
fn split_selects(f: &mut Func<'_>) -> bool {
    let uses = f.use_counts();
    let cfg = f.cfg();
    let enters_loop = |b: Block, t: Block| {
        cfg.preds(t).iter().any(|&p| cfg.dominates(t, p)) && !cfg.dominates(t, b)
    };
    let mut changed = false;
    for b in 0..f.blocks.len() as u32 {
        let b = Block(b);
        let Some(t) = f.terminator(b).filter(|&t| f.op(t) == Op::Br) else {
            continue;
        };
        let s = f.args(t)[0];
        let Some(d) = f.def_inst(s).filter(|&d| f.op(d) == Op::Select) else {
            continue;
        };
        if f.blocks[b.idx()].dead
            || uses[s.idx()] != 1
            || f.edges(t).iter().any(|e| enters_loop(b, e.target))
        {
            continue;
        }
        let (c, chosen) = (f.args(d)[0], [f.args(d)[1], f.args(d)[2]]);
        // Dead once the branch tests `c`, and its operands may move below it.
        let db = f.insts[d.idx()].block;
        f.retain(db, |_, j| j != d);
        let targets: Vec<(Block, Vec<Val>)> = (f.edges(t).iter())
            .map(|e| (e.target, f.vl(e.args).to_vec()))
            .collect();
        let tag = f.insts[t.idx()].tag;
        let mut arms = Vec::new();
        for v in chosen {
            let nb = f.new_block();
            f.blocks[nb.idx()].cold = f.blocks[b.idx()].cold;
            if let Some(cmp) = f.def_inst(v)
                && f.insts[cmp.idx()].block == b
                && f.op(cmp).is_pure()
                && uses[v.idx()] == 1
                && v != c
            {
                f.retain(b, |_, j| j != cmp);
                f.append(nb, cmp);
            }
            let br = f.make_inst(Op::Br, &[v], None, tag);
            f.append(nb, br);
            let edges: Vec<BlockCall> = (targets.iter())
                .map(|(target, args)| BlockCall {
                    target: *target,
                    args: f.new_vlist(args),
                })
                .collect();
            f.set_edges(br, &edges);
            arms.push(BlockCall {
                target: nb,
                args: f.new_vlist(&[]),
            });
        }
        f.args_mut(t)[0] = c;
        f.set_edges(t, &arms);
        changed = true;
    }
    changed
}

pub(crate) fn eliminate(f: &mut Func<'_>) -> bool {
    let split = split_selects(f);
    split | eliminate_merges(f)
}

fn eliminate_merges(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    let shaped = |f: &Func<'_>, m: Block| {
        let bd = &f.blocks[m.idx()];
        !bd.resume
            && !bd.cold
            && m != f.entry
            && cfg.incoming(m).len() > 1
            && bd.insts.len() <= CLONE_LIMIT
            && f.terminator(m).is_some_and(|t| f.op(t) == Op::Br)
    };
    if !cfg.rpo.iter().any(|&m| shaped(f, m)) {
        return false;
    }
    let tree = DomTree::new(f, &cfg);
    let mut header = vec![false; f.blocks.len()];
    for &t in &cfg.rpo {
        for h in f.succs(t) {
            header[h.idx()] |= cfg.dominates(h, t);
        }
    }
    let merge = |f: &Func<'_>, m: Block| shaped(f, m) && !header[m.idx()];
    // (terminator, edge, merge block, its outcome on the edge), found on the
    // blocks as they are: copies are not considered again.
    let mut copies: Vec<(Inst, usize, Block, bool)> = Vec::new();
    let mut known: FastMap<Val, bool> = FastMap::default();
    let mut undo: Vec<(Val, Option<bool>)> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut stack: Vec<(Block, bool)> = vec![(f.entry, false)];
    while let Some((b, done)) = stack.pop() {
        if done {
            let m = marks.pop().unwrap();
            while undo.len() > m {
                let (c, old) = undo.pop().unwrap();
                match old {
                    Some(o) => known.insert(c, o),
                    None => known.remove(&c),
                };
            }
            continue;
        }
        marks.push(undo.len());
        if let &[(t, k)] = cfg.incoming(b)
            && f.op(t) == Op::Br
        {
            let c = f.args(t)[0];
            undo.push((c, known.insert(c, k == 0)));
        }
        for &i in f.insts_of(b) {
            if let op @ (Op::GuardTrue | Op::GuardFalse) = f.op(i) {
                let c = f.args(i)[0];
                undo.push((c, known.insert(c, op == Op::GuardTrue)));
            }
        }
        if !f.blocks[b.idx()].cold
            && let Some(t) = f.terminator(b)
        {
            for k in 0..f.edges(t).len() {
                let m = f.edges(t)[k].target;
                if !merge(f, m) {
                    continue;
                }
                // A `br` here decides its own condition on the edge.
                let own = (f.op(t) == Op::Br).then(|| f.args(t)[0]);
                let old = own.map(|c| known.insert(c, k == 0));
                if let Some(c) = outcome(f, m, f.edge_args(t, k), &known) {
                    copies.push((t, k, m, c));
                }
                if let (Some(c), Some(old)) = (own, old) {
                    match old {
                        Some(o) => known.insert(c, o),
                        None => known.remove(&c),
                    };
                }
            }
        }
        stack.push((b, true));
        for &c in tree.children(b).iter().rev() {
            stack.push((c, false));
        }
    }
    if copies.is_empty() {
        return false;
    }
    // Each value of a copied block: the blocks defining it, and as what.
    let mut defs: FastMap<Val, Vec<(Block, Val)>> = FastMap::default();
    for (t, k, m, c) in copies {
        let nb = f.new_block();
        let mut vmap: FastMap<Val, Val> = FastMap::default();
        let args = f.edge_args(t, k).to_vec();
        for (j, &p) in f.params(m).to_vec().iter().enumerate() {
            // An argument may not be its parameter's representation yet
            // (`legalize` makes it so): uses of the parameter expect it.
            let a = crate::jit::opt::infer::convert(f, args[j], f.ty(p).rep, nb);
            vmap.insert(p, a);
            defs.entry(p).or_insert_with(|| vec![(m, p)]).push((nb, a));
        }
        let map = |vmap: &FastMap<Val, Val>, v: Val| vmap.get(&v).copied().unwrap_or(v);
        let insts = f.insts_of(m).to_vec();
        let (&br, body) = insts.split_last().unwrap();
        for &i in body {
            let d = f.insts[i.idx()];
            let a: Vec<Val> = f.args(i).iter().map(|&x| map(&vmap, x)).collect();
            let snap = (d.snap != NO_SNAP).then(|| f.map_snap(Snap(d.snap), |v| map(&vmap, v)));
            let ni = f.make_inst(d.op, &a, snap, d.tag);
            if let Some(&s) = f.exit_sites.get(&i) {
                f.exit_sites.insert(ni, s);
            }
            for (r, nr) in f.results(i).zip(f.results(ni)) {
                f.vals[nr.idx()].ty = f.vals[r.idx()].ty;
                vmap.insert(r, nr);
                defs.entry(r).or_insert_with(|| vec![(m, r)]).push((nb, nr));
            }
            f.append(nb, ni);
        }
        let e = f.edges(br)[!c as usize];
        let a: Vec<Val> = f.vl(e.args).iter().map(|&x| map(&vmap, x)).collect();
        let j = f.make_inst(Op::Jump, &[], None, f.insts[br.idx()].tag);
        f.append(nb, j);
        let a = f.new_vlist(&a);
        f.set_edges(
            j,
            &[BlockCall {
                target: e.target,
                args: a,
            }],
        );
        f.set_target(t, k, nb);
        f.set_edge_args(t, k, &[]);
    }
    crate::jit::opt::simplify::mark_unreachable(f);
    repair_ssa(f, &defs);
    true
}

/// The outcome of `m`'s branch entered with `args`, where `known` holds.
fn outcome(f: &Func<'_>, m: Block, args: &[Val], known: &FastMap<Val, bool>) -> Option<bool> {
    let c = f.args(f.terminator(m)?)[0];
    let params = f.params(m);
    let at_edge = |v: Val| params.iter().position(|&p| p == v).map_or(v, |j| args[j]);
    let b1 = |v: Val| match f.def_op(v) {
        Some(Op::KB1(x)) => Some(x),
        _ => known.get(&v).copied(),
    };
    let c = at_edge(c);
    let d = match f.def_inst(c) {
        Some(d) if f.insts[d.idx()].block == m => d,
        _ => return b1(c),
    };
    let a: Vec<Val> = f.args(d).iter().map(|&x| at_edge(x)).collect();
    // An operand computed in `m` has no type of its own on the edge.
    if a.iter().any(|&x| f.def_block(x) == Some(m)) {
        return None;
    }
    let decided = |t: crate::jit::ir::types::Ty, set: TypeSet| {
        if t.set.is_empty() {
            None
        } else if t.within(set) {
            Some(true)
        } else if !t.set.intersects(set) {
            Some(false)
        } else {
            None
        }
    };
    let kint = |v: Val| match f.def_op(v) {
        Some(Op::KI32(n)) => Some(n as i64),
        Some(Op::KI64(n)) => Some(n),
        _ => None,
    };
    let kval = |v: Val| match f.def_op(v) {
        Some(Op::KVal(bits)) => Some(bits),
        _ => None,
    };
    match f.op(d) {
        Op::IsType(set) => decided(f.ty(a[0]), set),
        Op::IsFalsy => decided(f.ty(a[0]), TypeSet::FALSY),
        Op::SameBits if a[0] == a[1] => Some(true),
        Op::SameBits => Some(kval(a[0])? == kval(a[1])?),
        Op::ICmp(cc) | Op::LCmp(cc) => Some(cc.eval_i(kint(a[0])?, kint(a[1])?)),
        _ => None,
    }
}

/// Uses of a value `defs` gives several definitions now take the one that
/// reaches them, through block parameters where they meet.
fn repair_ssa(f: &mut Func<'_>, defs: &FastMap<Val, Vec<(Block, Val)>>) {
    let mut sites: Vec<(Val, Block)> = Vec::new();
    for b in 0..f.blocks.len() as u32 {
        let b = Block(b);
        if f.blocks[b.idx()].dead {
            continue;
        }
        for &i in f.insts_of(b) {
            f.for_each_use(i, |v| {
                if let Some(d) = defs.get(&v)
                    && d.iter().all(|&(x, _)| x != b)
                {
                    sites.push((v, b));
                }
            });
        }
    }
    sites.sort_unstable();
    sites.dedup();
    let cfg = f.cfg();
    let mut memo: FastMap<(Val, Block), Val> = FastMap::default();
    for (v, b) in sites {
        let r = reaching(f, &cfg, defs, &mut memo, v, b);
        for k in 0..f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            f.replace_uses(i, v, r);
        }
    }
}

/// `v` at the start of `b`, or at the end of a block defining it (Braun et
/// al.'s lookup, every block sealed).
fn reaching(
    f: &mut Func<'_>,
    cfg: &crate::jit::ir::CfgInfo,
    defs: &FastMap<Val, Vec<(Block, Val)>>,
    memo: &mut FastMap<(Val, Block), Val>,
    v: Val,
    b: Block,
) -> Val {
    if let Some(&(_, r)) = defs[&v].iter().find(|&&(x, _)| x == b) {
        return r;
    }
    if let Some(&r) = memo.get(&(v, b)) {
        return r;
    }
    let inc = cfg.incoming(b);
    let r = match inc {
        [] => v,
        &[(t, _)] => {
            let p = f.insts[t.idx()].block;
            reaching(f, cfg, defs, memo, v, p)
        }
        _ => {
            let p = f.add_param(b, f.ty(v));
            memo.insert((v, b), p);
            for &(t, k) in inc {
                let pred = f.insts[t.idx()].block;
                let a = reaching(f, cfg, defs, memo, v, pred);
                f.push_edge_arg(t, k as usize, a);
            }
            p
        }
    };
    memo.insert((v, b), r);
    r
}
