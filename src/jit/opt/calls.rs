//! The call-boundary pass (9.8, C1, H11): a call clobbers every register,
//! so a value live across it stays in the region's frame, and must then be
//! one the call-site snapshot names: the collector traces those spill slots
//! while the call runs. A value one of the call's home slots holds is loaded
//! from it again; constants, and pure computations of such values (raw
//! pointers among them, which the collector does not see), are computed
//! again after the resume. Later uses get whichever definition reaches them.

use crate::jit::FastMap;
use crate::jit::ir::ops::{ExitTag, Mem, Op};
use crate::jit::ir::types::Rep;
use crate::jit::ir::verify::live_in;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Val};

pub(crate) fn call_boundaries(f: &mut Func<'_>) -> Result<(), String> {
    f.boundaries = true;
    let cfg = f.cfg();
    if !cfg.rpo.iter().any(|b| f.blocks[b.idx()].resume) {
        return Ok(());
    }
    // Copies and repairing parameters by the value they stand for: equal to
    // it wherever both are defined.
    let mut origin: FastMap<Val, Val> = FastMap::default();
    // A copy may make a value another call kept live across it instead.
    for _ in 0..16 {
        if !boundaries_round(f, &mut origin)? {
            // Repairs add a parameter wherever definitions might meet; most
            // join one value.
            crate::jit::build::remove_trivial_params(f);
            fold_constant_params(f);
            return Ok(());
        }
    }
    Err("the call boundaries did not settle".into())
}

/// Parameters that only join copies of one argument-free value (a constant
/// made again after each call) are that value, made where the block starts:
/// a loop's callee stays a constant (`find_known_callees`) and costs no
/// moves.
fn fold_constant_params(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut changed = false;
    for &b in &cfg.rpo {
        let incoming = cfg.incoming(b);
        if f.blocks[b.idx()].params.len() == 0 || incoming.is_empty() {
            continue;
        }
        let params = f.params(b).to_vec();
        let mut kept = Vec::with_capacity(params.len());
        for (k, &p) in params.iter().enumerate() {
            let mut op = None;
            let same = incoming.iter().all(|&(t, e)| {
                let a = map[f.edge_args(t, e as usize)[k].idx()];
                if a == p {
                    return true;
                }
                let o = f
                    .def_inst(a)
                    .filter(|&d| f.insts[d.idx()].an == 0 && f.op(d).is_pure())
                    .map(|d| f.op(d));
                o.is_some() && *op.get_or_insert(o) == o
            });
            let Some(Some(op)) = op.filter(|_| same) else {
                kept.push(k);
                continue;
            };
            let at = usize::from(
                f.insts_of(b)
                    .first()
                    .is_some_and(|&i| matches!(f.op(i), Op::Resume { .. })),
            );
            let i = f.make_inst(op, &[], None, ExitTag::Type);
            f.insert(b, at, i);
            let r = f.result(i);
            f.vals[r.idx()].ty = f.vals[p.idx()].ty;
            map.push(r);
            map[p.idx()] = r;
            changed = true;
        }
        if kept.len() == params.len() {
            continue;
        }
        for &(t, e) in incoming {
            let args: Vec<Val> = kept
                .iter()
                .map(|&k| f.edge_args(t, e as usize)[k])
                .collect();
            f.set_edge_args(t, e as usize, &args);
        }
        let ps: Vec<Val> = kept.iter().map(|&k| params[k]).collect();
        f.set_params(b, &ps);
    }
    if changed {
        f.apply_replacements(&mut map);
    }
}

/// Keep or copy what is live across each call, and give later uses the
/// copies; whether anything changed.
fn boundaries_round(f: &mut Func<'_>, origin: &mut FastMap<Val, Val>) -> Result<bool, String> {
    let cfg = f.cfg();
    let live = live_in(f, &cfg.rpo);
    // What the homes hold matters only where a call does not keep all.
    let needed = cfg.rpo.iter().any(|&b| {
        f.blocks[b.idx()].resume && {
            let kept = kept_across(f, &cfg, b);
            live[b.idx()]
                .iter()
                .any(|v| !kept.contains(&same_reg(f, Val(v as u32))))
        }
    });
    let contents = if needed {
        home_contents(f, &cfg)
    } else {
        FastMap::default()
    };
    let mut changed = false;
    // Each value live across a call, with its copy in each resume block it
    // is live into.
    let mut across: Vec<(Val, FastMap<Block, Val>)> = Vec::new();
    for &b in &cfg.rpo {
        if !f.blocks[b.idx()].resume {
            continue;
        }
        let mut cx = After {
            b,
            kept: kept_across(f, &cfg, b),
            homes: contents.get(&b).map_or(&[][..], |h| &h[..]),
            origin: &mut *origin,
            memo: FastMap::default(),
            // The copies go after the resume block's `Resume`.
            at: 1,
        };
        for v in live[b.idx()].iter().collect::<Vec<_>>() {
            let v = Val(v as u32);
            if cx.kept.contains(&same_reg(f, v)) {
                continue;
            }
            let c = match remat(f, v, &mut cx) {
                Some(c) => c,
                None => {
                    keep_in_free_reg(f, &cfg, b, v)?;
                    cx.kept.push(same_reg(f, v));
                    changed = true;
                    continue;
                }
            };
            if c != v {
                let o = origin_of(cx.origin, v);
                cx.origin.insert(c, o);
            }
            match across.iter_mut().find(|(x, _)| *x == v) {
                Some((_, cs)) => {
                    cs.insert(b, c);
                }
                None => {
                    let mut cs = FastMap::default();
                    cs.insert(b, c);
                    across.push((v, cs));
                }
            }
        }
    }
    changed |= !across.is_empty();
    let cfg = f.cfg();
    // A value and its refinements share a register (`same_reg`): one
    // repaired with them refines the other's definitions again, so a call
    // keeping one keeps both. Roots first.
    across.sort_by_key(|&(v, _)| refinement_depth(f, v));
    let mut reps: Vec<Repair> = Vec::with_capacity(across.len());
    for (v, copies) in across {
        let root = same_reg(f, v);
        let from = reps
            .iter()
            .position(|r| r.v == root && root != v)
            .map(|k| (k, refining_ops(f, v, root)));
        reps.push(Repair {
            v,
            def_block: f.def_block(v).unwrap(),
            copies,
            memo: FastMap::default(),
            from,
        });
    }
    // The blocks using each repaired value; repairs add no uses of another.
    let rep_of: FastMap<Val, usize> = reps.iter().enumerate().map(|(k, r)| (r.v, k)).collect();
    let mut used_in: Vec<Vec<Block>> = vec![Vec::new(); reps.len()];
    for &b in &cfg.rpo {
        for &i in f.insts_of(b) {
            f.for_each_use(i, |x| {
                if let Some(&k) = rep_of.get(&x)
                    && used_in[k].last() != Some(&b)
                {
                    used_in[k].push(b);
                }
            });
        }
    }
    for k in 0..reps.len() {
        // A parameter is defined where its block starts.
        let (v, d) = (reps[k].v, f.def_inst(reps[k].v));
        for b in std::mem::take(&mut used_in[k]) {
            let mut before = (d.is_none() && b == reps[k].def_block).then_some(v);
            // A kept other name of the value is there from the start.
            if let Some(&c) = reps[k].copies.get(&b)
                && f.def_block(c) != Some(b)
            {
                before = Some(c);
            }
            let mut j = 0;
            while j < f.insts_of(b).len() {
                let i = f.insts_of(b)[j];
                j += 1;
                if Some(i) == d {
                    before = Some(v);
                    continue;
                }
                if let Some(&c) = reps[k].copies.get(&b)
                    && f.def_inst(c) == Some(i)
                {
                    before = Some(c);
                    continue;
                }
                if !uses(f, i, v) {
                    continue;
                }
                // The definition reaching instruction `i`; one made here goes
                // where the block starts, before it.
                let r = match before {
                    Some(x) => x,
                    None => {
                        let n = f.insts_of(b).len();
                        let x = start(&mut reps, k, f, &cfg, b, origin);
                        j += f.insts_of(b).len() - n;
                        before = Some(x);
                        x
                    }
                };
                if r != v {
                    f.replace_uses(i, v, r);
                }
            }
        }
    }
    Ok(changed)
}

/// Name `v`, which no register holds after the call into `b`, in the call's
/// snapshot under a register dead there, so the collector traces its spill
/// slot.
fn keep_in_free_reg(f: &mut Func<'_>, cfg: &CfgInfo, b: Block, v: Val) -> Result<(), String> {
    let none = || {
        format!(
            "v{} ({:?} {}) is live across the call resuming at b{}",
            v.0,
            f.def_op(v),
            f.ty(v),
            b.0
        )
    };
    if f.ty(v).rep == crate::jit::ir::types::Rep::Ptr {
        return Err(none());
    }
    let p = *cfg.preds(b).first().ok_or_else(none)?;
    let t = f.terminator(p).ok_or_else(none)?;
    let pc = match f.op(t) {
        Op::Call { pc, .. } | Op::CallMm { pc, .. } => pc,
        _ => return Err(none()),
    };
    let used: Vec<u8> = f.snap_entries(t).iter().map(|e| e.0).collect();
    let r = f
        .call_free
        .get(&pc)
        .and_then(|free| free.iter().find(|&r| !used.contains(&(r as u8))))
        .ok_or_else(none)?;
    let d = f.snaps[f.insts[t.idx()].snap as usize];
    let from = f.snap_pool.len();
    f.snap_pool
        .extend_from_within(d.entries.start as usize..(d.entries.start + d.entries.len) as usize);
    f.snap_pool.push((r as u8, v));
    let s = f.finish_snap(d.frame, d.pc, d.kind, from);
    if let Some(&homes) = f.call_homes.get(&f.insts[t.idx()].snap) {
        f.call_homes.insert(s.0, homes);
    }
    f.insts[t.idx()].snap = s.0;
    Ok(())
}

/// The value whose register `v` shares, as the lowering aliases them:
/// through refining guards and i32 unboxes, whose bits are the operand's
/// (not `LToI`, whose operand holds an i64).
pub(crate) fn same_reg(f: &Func<'_>, mut v: Val) -> Val {
    while let Some(d) = f.def_inst(v) {
        match f.op(d) {
            Op::Guard(_) | Op::Refine(_) | Op::Unbox(crate::jit::ir::types::Rep::I32) => {
                v = f.args(d)[0]
            }
            _ => break,
        }
    }
    v
}

/// Whether `v` is a constant, which needs no spill slot across a call.
pub(crate) fn is_const(f: &Func<'_>, v: Val) -> bool {
    matches!(
        f.def_op(v),
        Some(Op::KVal(_) | Op::KObj(_) | Op::KI32(_) | Op::KI64(_) | Op::KB1(_) | Op::KF64(_))
    )
}

/// The values the call into resume block `b` keeps in the frame: its
/// snapshot's, but raw pointers and constants, which are computed again
/// (the collector traces only values), each as `same_reg` gives it.
pub(crate) fn kept_across(f: &Func<'_>, cfg: &CfgInfo, b: Block) -> Vec<Val> {
    let Some(&p) = cfg.preds(b).first() else {
        return Vec::new();
    };
    let Some(t) = f.terminator(p) else {
        return Vec::new();
    };
    f.snap_entries(t)
        .iter()
        .map(|e| e.1)
        .filter(|&v| f.ty(v).rep != crate::jit::ir::types::Rep::Ptr)
        .map(|v| same_reg(f, v))
        .filter(|&v| !is_const(f, v))
        .collect()
}

/// Whether `v` can be computed again after a call: a constant, a pure
/// instruction of such operands, or a type guard's refinement of one,
/// which holds again since the value is the same.
pub(crate) fn rematerializable(f: &Func<'_>, v: Val) -> bool {
    let Some(d) = f.def_inst(v) else {
        return false;
    };
    let op = f.op(d);
    // A pure helper is a call again.
    (op.is_pure() && !matches!(op, Op::Helper(_)) || matches!(op, Op::Guard(_) | Op::Refine(_)))
        && f.args(d).iter().all(|&a| rematerializable(f, a))
}

/// What is there after the call into resume block `b`, for `remat`.
struct After<'a> {
    b: Block,
    /// Values the call keeps in the frame, as they are.
    kept: Vec<Val>,
    /// Values its home slots hold, by register.
    homes: &'a [(u8, Val)],
    origin: &'a mut FastMap<Val, Val>,
    /// Copies made so far.
    memo: FastMap<Val, Val>,
    /// Where the next copy goes.
    at: usize,
}

impl After<'_> {
    fn insert(&mut self, f: &mut Func<'_>, i: Inst) {
        f.insert(self.b, self.at, i);
        self.at += 1;
    }
}

/// `v` computed again after the call, its operands first.
fn remat(f: &mut Func<'_>, v: Val, cx: &mut After<'_>) -> Option<Val> {
    if let Some(&c) = cx.memo.get(&v) {
        return Some(c);
    }
    if cx.kept.contains(&same_reg(f, v)) {
        return Some(v);
    }
    // Another name of the value the call keeps; a refinement's too (a
    // repair parameter of a refined load is kept, not the load).
    let o = origin_of(cx.origin, v);
    if let Some(&k) = cx
        .kept
        .iter()
        .find(|&&k| origin_of(cx.origin, k) == o && f.ty(k).rep == f.ty(v).rep)
    {
        return Some(k);
    }
    // A slot holds `v`, or `v` boxed.
    let rep = f.ty(v).rep;
    let home = cx.homes.iter().find(|&&(_, x)| {
        x == v
            || matches!(rep, Rep::I32 | Rep::I64 | Rep::F64)
                && f.def_inst(x)
                    .is_some_and(|d| f.op(d) == Op::Box && f.args(d)[0] == v)
    });
    if let Some(&(r, _)) = home {
        let l = f.make_inst(Op::Load(r), &[], None, ExitTag::Type);
        let lv = f.result(l);
        f.vals[lv.idx()].ty = f.ty(v).boxed();
        cx.insert(f, l);
        let c = if rep == Rep::Val {
            lv
        } else {
            let u = f.make_inst(Op::Unbox(rep), &[lv], None, ExitTag::Type);
            let (uv, ty) = (f.result(u), f.ty(v));
            f.vals[uv.idx()].ty = ty;
            cx.insert(f, u);
            uv
        };
        cx.memo.insert(v, c);
        return Some(c);
    }
    let d = f.def_inst(v)?;
    let op = match f.op(d) {
        Op::Guard(set) => Op::Refine(set),
        op if op.is_pure() && !matches!(op, Op::Helper(_)) || matches!(op, Op::Refine(_)) => op,
        _ => return None,
    };
    let args: Vec<Val> = f.args(d).to_vec();
    let mut copied = Vec::with_capacity(args.len());
    for a in args {
        copied.push(remat(f, a, cx)?);
    }
    let c = f.make_inst(op, &copied, None, f.insts[d.idx()].tag);
    let cv = f.result(c);
    f.vals[cv.idx()].ty = f.ty(v);
    cx.insert(f, c);
    cx.memo.insert(v, cv);
    Some(cv)
}

/// What each call's home slots (`Func::call_homes`) hold when it is made,
/// by its resume block: the value last loaded from or stored to each, or
/// where paths bring different ones, the block parameter joining exactly
/// those. A call keeps only its homes, but for the captured registers,
/// which it may write through their cells; a helper that writes slots keeps
/// none.
fn home_contents(f: &Func<'_>, cfg: &CfgInfo) -> FastMap<Block, Vec<(u8, Val)>> {
    const NONE: Val = Val(u32::MAX);
    let homes_of = |t: Inst| {
        let mut h = f
            .call_homes
            .get(&f.insts[t.idx()].snap)
            .copied()
            .unwrap_or_default();
        h.minus(&f.captured);
        h
    };
    // Only the slots some call keeps are tracked, by their index here.
    let mut tracked = crate::jit::build::cfg::RegSet::EMPTY;
    for &b in &cfg.rpo {
        if let Some(t) = f.terminator(b)
            && matches!(f.op(t), Op::Call { .. } | Op::CallMm { .. })
        {
            tracked.union(&homes_of(t));
        }
    }
    let slots: Vec<usize> = tracked.iter().collect();
    if slots.is_empty() {
        return FastMap::default();
    }
    let mut idx = [usize::MAX; 256];
    for (k, &r) in slots.iter().enumerate() {
        idx[r] = k;
    }
    let n = slots.len();
    // What each block passes its successors, once reached.
    let mut out: Vec<Option<Vec<Val>>> = vec![None; f.blocks.len()];
    let mut at_call: FastMap<Block, Vec<(u8, Val)>> = FastMap::default();
    let mut s: Vec<Val> = Vec::with_capacity(n);
    for _ in 0..50 {
        let mut changed = false;
        for &b in &cfg.rpo {
            s.clear();
            if b == f.entry {
                s.resize(n, NONE);
            } else if !meet(f, cfg, &out, b, n, &mut s) {
                continue;
            }
            for &i in f.insts_of(b) {
                match f.op(i) {
                    // An untracked slot's load or store changes no tracked one.
                    Op::Load(r) => {
                        if let Some(x) = s.get_mut(idx[r as usize]) {
                            *x = f.result(i);
                        }
                    }
                    Op::Store(r) => {
                        if let Some(x) = s.get_mut(idx[r as usize]) {
                            *x = f.args(i)[0];
                        }
                    }
                    Op::Call { .. } | Op::CallMm { .. } => {
                        let h = homes_of(i);
                        let here: Vec<(u8, Val)> = slots
                            .iter()
                            .zip(&s)
                            .filter(|&(&r, &x)| h.contains(r) && x != NONE)
                            .map(|(&r, &x)| (r as u8, x))
                            .collect();
                        at_call.insert(f.edges(i)[0].target, here);
                        for (&r, x) in slots.iter().zip(s.iter_mut()) {
                            if !h.contains(r) {
                                *x = NONE;
                            }
                        }
                    }
                    op if op.effects().writes.contains(Mem::SLOT) && !op.is_terminator() => {
                        s.fill(NONE)
                    }
                    _ => {}
                }
            }
            if out[b.idx()].as_deref() != Some(&s[..]) {
                out[b.idx()] = Some(s.clone());
                changed = true;
            }
        }
        if !changed {
            return at_call;
        }
    }
    // Not settled: claim nothing.
    FastMap::default()
}

/// The slots' contents at the start of `b` from its reached predecessors
/// into `s`; false when none is reached yet.
fn meet(
    f: &Func<'_>,
    cfg: &CfgInfo,
    out: &[Option<Vec<Val>>],
    b: Block,
    n: usize,
    s: &mut Vec<Val>,
) -> bool {
    const NONE: Val = Val(u32::MAX);
    let inc = cfg.incoming(b);
    let reached: Vec<(Inst, u32, &Vec<Val>)> = inc
        .iter()
        .filter_map(|&(t, e)| Some((t, e, out[f.insts[t.idx()].block.idx()].as_ref()?)))
        .collect();
    let Some(&(_, _, first)) = reached.first() else {
        return false;
    };
    s.extend_from_slice(first);
    let params = f.params(b);
    for r in 0..n {
        if reached.iter().all(|x| x.2[r] == s[r]) {
            continue;
        }
        // A parameter every reached edge passes the slot's value to; an
        // edge not reached yet passes it too, or the next round finds out.
        s[r] = (0..params.len())
            .find(|&k| {
                reached
                    .iter()
                    .all(|&(t, e, o)| o[r] != NONE && f.edge_args(t, e as usize)[k] == o[r])
            })
            .map_or(NONE, |k| params[k]);
    }
    true
}

fn uses(f: &Func<'_>, i: Inst, v: Val) -> bool {
    let mut u = false;
    f.for_each_use(i, |x| u |= x == v);
    u
}

/// How many refining steps (`same_reg`) `v` is above its root.
fn refinement_depth(f: &Func<'_>, v: Val) -> usize {
    let mut n = 0;
    let mut x = v;
    while same_reg(f, x) != x {
        x = f.args(f.def_inst(x).unwrap())[0];
        n += 1;
    }
    n
}

/// The operations making `v` from `root` again, first first: a guard as
/// the refinement it proved.
fn refining_ops(f: &Func<'_>, v: Val, root: Val) -> Vec<Op> {
    let mut ops = Vec::new();
    let mut x = v;
    while x != root {
        let d = f.def_inst(x).unwrap();
        ops.push(match f.op(d) {
            Op::Guard(set) => Op::Refine(set),
            op => op,
        });
        x = f.args(d)[0];
    }
    ops.reverse();
    ops
}

/// The definition of `v` reaching each block (Braun et al.'s lookup, every
/// block sealed), with parameters where definitions meet; for a refinement
/// of a value repaired too, that value's refined again.
struct Repair {
    v: Val,
    def_block: Block,
    copies: FastMap<Block, Val>,
    memo: FastMap<Block, Val>,
    from: Option<(usize, Vec<Op>)>,
}

/// The value `v` stands for (`call_boundaries`).
fn origin_of(origin: &FastMap<Val, Val>, v: Val) -> Val {
    origin.get(&v).copied().unwrap_or(v)
}

/// The definition of repair `k`'s value at the end of `b`.
fn end(
    reps: &mut [Repair],
    k: usize,
    f: &mut Func<'_>,
    cfg: &CfgInfo,
    b: Block,
    origin: &mut FastMap<Val, Val>,
) -> Val {
    if let Some(&c) = reps[k].copies.get(&b) {
        return c;
    }
    if b == reps[k].def_block {
        return reps[k].v;
    }
    start(reps, k, f, cfg, b, origin)
}

/// The definition of repair `k`'s value at the start of `b`.
fn start(
    reps: &mut [Repair],
    k: usize,
    f: &mut Func<'_>,
    cfg: &CfgInfo,
    b: Block,
    origin: &mut FastMap<Val, Val>,
) -> Val {
    if let Some(&r) = reps[k].memo.get(&b) {
        return r;
    }
    let inc = cfg.incoming(b);
    let r = match inc.len() {
        // Every use is dominated by the definition, so only its own block
        // can start without one.
        0 => reps[k].v,
        1 => {
            let p = f.insts[inc[0].0.idx()].block;
            end(reps, k, f, cfg, p, origin)
        }
        _ => match reps[k].from.clone() {
            Some((root, ops)) => {
                let mut x = start(reps, root, f, cfg, b, origin);
                let at = usize::from(
                    f.insts_of(b)
                        .first()
                        .is_some_and(|&i| matches!(f.op(i), Op::Resume { .. })),
                );
                for (n, op) in ops.into_iter().enumerate() {
                    let i = f.make_inst(op, &[x], None, ExitTag::Type);
                    f.insert(b, at + n, i);
                    x = f.result(i);
                }
                origin.insert(x, origin_of(origin, reps[k].v));
                x
            }
            None => {
                let p = f.add_param(b, f.ty(reps[k].v));
                origin.insert(p, origin_of(origin, reps[k].v));
                reps[k].memo.insert(b, p);
                for &(t, e) in inc {
                    let pred = f.insts[t.idx()].block;
                    let a = end(reps, k, f, cfg, pred, origin);
                    f.push_edge_arg(t, e as usize, a);
                }
                p
            }
        },
    };
    reps[k].memo.insert(b, r);
    r
}
