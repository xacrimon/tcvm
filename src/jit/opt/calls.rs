//! The call-boundary pass: a call clobbers every register, so a value live
//! across it stays in the region's frame, and must then be one the call-site
//! snapshot names: the collector traces those spill slots while the call runs.
//! A value one of the call's home slots holds is loaded from it again; pure
//! computations of such values are computed again after the resume. Constants
//! stay live (the allocator makes them where used), and so does a table's
//! pointer, the call naming its table (the collector does not see raw
//! pointers). Later uses get whichever definition reaches them. A copy with all
//! its uses in one later block goes there when no call comes between
//! (`place_copies`).

use crate::jit::FastMap;
use crate::jit::ir::ops::{ExitTag, Mem, Op};
use crate::jit::ir::types::{Rep, Ty, TypeSet};
use crate::jit::ir::verify::live_in;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Val};
use crate::jit::opt::natural_loops;

pub(crate) fn call_boundaries(f: &mut Func<'_>) -> Result<(), String> {
    f.boundaries = true;
    fold_ptr_params(f);
    let cfg = f.cfg();
    if !cfg.rpo.iter().any(|b| f.blocks[b.idx()].resume) {
        return Ok(());
    }
    // Copies and repairing parameters by the value they stand for: equal to
    // it wherever both are defined.
    let mut origin: FastMap<Val, Val> = FastMap::default();
    let mut made: Vec<Inst> = Vec::new();
    // A copy may make a value another call kept live across it instead.
    for _ in 0..16 {
        if !boundaries_round(f, &mut origin, &mut made)? {
            // Repairs add a parameter wherever definitions might meet; most
            // join one value.
            crate::jit::build::remove_trivial_params(f);
            fold_constant_params(f);
            place_copies(f, &made);
            return Ok(());
        }
    }
    Err("the call boundaries did not settle".into())
}

/// Whether `v` is a constant or a pure computation of constants alone.
fn const_chain(f: &Func<'_>, v: Val) -> bool {
    let Some(d) = f.def_inst(v) else {
        return false;
    };
    let op = f.op(d);
    op.is_pure() && !matches!(op, Op::Helper(_)) && f.args(d).iter().all(|&a| const_chain(f, a))
}

/// Pointer parameters that join unboxings are an unboxing of a table value,
/// made where their block starts: a call keeps no raw pointer, and only a
/// definition is made again after one. The table is one value that
/// dominates the block (the receiver of a call dispatched by shape,
/// across its bodies' calls), or a parameter joining the unboxed
/// tables edge by edge, made when none does (branch elimination's repairs
/// join a cloned block's table and its unboxing).
fn fold_ptr_params(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut changed = false;
    for &b in &cfg.rpo {
        let ps = f.params(b).to_vec();
        let inc = cfg.incoming(b);
        if inc.is_empty() || !ps.iter().any(|&p| f.ty(p).rep == Rep::Ptr) {
            continue;
        }
        let mut at = usize::from(
            (f.insts_of(b).first()).is_some_and(|&i| matches!(f.op(i), Op::Resume { .. })),
        );
        let mut params = ps.clone();
        let mut args: Vec<Vec<Val>> = (inc.iter())
            .map(|&(t, e)| f.edge_args(t, e as usize).to_vec())
            .collect();
        let mut folded = vec![false; ps.len()];
        for (k, &p) in ps.iter().enumerate() {
            if f.ty(p).rep != Rep::Ptr {
                continue;
            }
            let x = match unboxed_from(f, &cfg, b, k, p) {
                Some(x) => x,
                None => {
                    let Some(tables) = unboxed_tables(f, &args, k, p) else {
                        continue;
                    };
                    match sibling(f, &ps, &args, &tables) {
                        Some(q) => q,
                        None => {
                            let set = (tables.iter().flatten())
                                .fold(TypeSet::empty(), |s, &t| s | f.ty(t).set);
                            let q = f.add_param(b, Ty::val(set));
                            for (a, t) in args.iter_mut().zip(&tables) {
                                a.push(t.unwrap_or(q));
                            }
                            params.push(q);
                            q
                        }
                    }
                }
            };
            let i = f.make_inst(Op::Unbox(Rep::Ptr), &[x], None, ExitTag::Type);
            f.insert(b, at, i);
            at += 1;
            let u = f.result(i);
            f.vals[u.idx()].ty = f.vals[p.idx()].ty;
            map.push(Val(map.len() as u32));
            map[p.idx()] = u;
            folded[k] = true;
        }
        if !folded.contains(&true) {
            continue;
        }
        let keep = |k: &usize| folded.get(*k).is_none_or(|&f| !f);
        for (&(t, e), a) in inc.iter().zip(&args) {
            let a: Vec<Val> = (0..a.len()).filter(keep).map(|k| a[k]).collect();
            f.set_edge_args(t, e as usize, &a);
        }
        let ps2: Vec<Val> = (0..params.len()).filter(keep).map(|k| params[k]).collect();
        f.set_params(b, &ps2);
        changed = true;
    }
    if changed {
        map.resize(f.vals.len(), Val(0));
        for (x, m) in map.iter_mut().enumerate() {
            if m.0 == 0 && x != 0 {
                *m = Val(x as u32);
            }
        }
        f.apply_replacements(&mut map);
    }
}

/// The table value whose unboxing every edge into `b` passes as its `k`th
/// parameter `p` (through guards and refinements), when it dominates `b`.
fn unboxed_from(f: &Func<'_>, cfg: &CfgInfo, b: Block, k: usize, p: Val) -> Option<Val> {
    let mut x = None;
    for &(t, e) in cfg.incoming(b) {
        let a = f.edge_args(t, e as usize)[k];
        if a == p {
            continue;
        }
        let d = f.def_inst(a)?;
        if f.op(d) != Op::Unbox(Rep::Ptr) {
            return None;
        }
        let r = same_reg(f, f.args(d)[0]);
        match x {
            None => x = Some(r),
            Some(y) if y == r => {}
            Some(_) => return None,
        }
    }
    let x = x?;
    let home = match f.vals[x.idx()].def {
        crate::jit::ir::ValDef::Param(blk, _) => blk,
        crate::jit::ir::ValDef::Inst(i, _) => f.insts[i.idx()].block,
    };
    (home != b && cfg.dominates(home, b)).then_some(x)
}

/// By edge, the table whose unboxing the edge passes as the `k`th
/// parameter `p`, or none for an edge passing `p` itself.
fn unboxed_tables(f: &Func<'_>, args: &[Vec<Val>], k: usize, p: Val) -> Option<Vec<Option<Val>>> {
    (args.iter())
        .map(|a| {
            if a[k] == p {
                return Some(None);
            }
            let d = f.def_inst(a[k])?;
            (f.op(d) == Op::Unbox(Rep::Ptr)).then(|| Some(f.args(d)[0]))
        })
        .collect()
}

/// The parameter of `ps` each edge passes the table of `tables` as (itself
/// where the pointer goes round unchanged).
fn sibling(f: &Func<'_>, ps: &[Val], args: &[Vec<Val>], tables: &[Option<Val>]) -> Option<Val> {
    let joins = |j: usize, q: Val| {
        args.iter().zip(tables).all(|(a, t)| match t {
            Some(t) => same_reg(f, a[j]) == same_reg(f, *t),
            None => a[j] == q,
        })
    };
    (ps.iter().enumerate())
        .find(|&(j, &q)| f.ty(q).rep == Rep::Val && joins(j, q))
        .map(|(_, &q)| q)
}

/// `v`'s computation again in block `b` at `at` (advanced past it), once
/// per block.
fn copy_chain(
    f: &mut Func<'_>,
    v: Val,
    b: Block,
    at: &mut usize,
    copies: &mut FastMap<(Block, Val), Val>,
) -> Val {
    if let Some(&c) = copies.get(&(b, v)) {
        return c;
    }
    let d = f.def_inst(v).expect("a computation");
    let args: Vec<Val> = f.args(d).to_vec();
    let args: Vec<Val> = args
        .into_iter()
        .map(|a| copy_chain(f, a, b, at, copies))
        .collect();
    let i = f.make_inst(f.op(d), &args, None, ExitTag::Type);
    f.insert(b, *at, i);
    *at += 1;
    let c = f.result(i);
    f.vals[c.idx()].ty = f.vals[v.idx()].ty;
    copies.insert((b, v), c);
    c
}

/// Parameters that only join copies of one constant computation (the
/// call-boundary pass's copies after calls, or other such parameters
/// around nested loops) are that computation, made again where the block
/// starts: no loop carries a constant around its back edge in a register
/// the allocator may spill on every iteration.
fn fold_constant_params(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let mut params: Vec<(Block, usize, Val)> = Vec::new();
    for &b in &cfg.rpo {
        if !cfg.incoming(b).is_empty() {
            for (k, &p) in f.params(b).iter().enumerate() {
                params.push((b, k, p));
            }
        }
    }
    // The computation each parameter stands for (a value of it), or none;
    // absent until known, optimistically around cycles.
    let mut of: FastMap<Val, Option<Val>> = FastMap::default();
    loop {
        let mut changed = false;
        for &(b, k, p) in &params {
            if of.get(&p) == Some(&None) {
                continue;
            }
            let mut rep: Option<Val> = None;
            let mut bad = false;
            for &(t, e) in cfg.incoming(b) {
                let a = f.edge_args(t, e as usize)[k];
                if a == p {
                    continue;
                }
                let r = match of.get(&a) {
                    Some(&Some(r)) => r,
                    Some(&None) => {
                        bad = true;
                        break;
                    }
                    None if const_chain(f, a) => a,
                    None if f.def_inst(a).is_none() => continue,
                    None => {
                        bad = true;
                        break;
                    }
                };
                match rep {
                    None => rep = Some(r),
                    Some(x) if chain_eq(f, x, r) => {}
                    Some(_) => {
                        bad = true;
                        break;
                    }
                }
            }
            let new = if bad {
                None
            } else if let Some(r) = rep {
                Some(r)
            } else {
                continue;
            };
            if of.get(&p) != Some(&new) {
                of.insert(p, new);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut copies: FastMap<(Block, Val), Val> = FastMap::default();
    let mut repl: Vec<(Val, Val)> = Vec::new();
    for &b in &cfg.rpo {
        let ps = f.params(b).to_vec();
        let folded: Vec<(usize, Val)> = (ps.iter().enumerate())
            .filter_map(|(k, &p)| of.get(&p).copied().flatten().map(|r| (k, r)))
            .collect();
        if folded.is_empty() {
            continue;
        }
        let mut at = usize::from(
            f.insts_of(b)
                .first()
                .is_some_and(|&i| matches!(f.op(i), Op::Resume { .. })),
        );
        for &(k, r) in &folded {
            let c = copy_chain(f, r, b, &mut at, &mut copies);
            repl.push((ps[k], c));
        }
        let kept: Vec<usize> = (0..ps.len())
            .filter(|k| !folded.iter().any(|&(j, _)| j == *k))
            .collect();
        for &(t, e) in cfg.incoming(b) {
            let args: Vec<Val> = kept
                .iter()
                .map(|&k| f.edge_args(t, e as usize)[k])
                .collect();
            f.set_edge_args(t, e as usize, &args);
        }
        let ps2: Vec<Val> = kept.iter().map(|&k| ps[k]).collect();
        f.set_params(b, &ps2);
    }
    if !repl.is_empty() {
        let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
        for (p, c) in repl {
            map[p.idx()] = c;
        }
        f.apply_replacements(&mut map);
    }
}

/// Whether `a` and `b` compute the same constant.
fn chain_eq(f: &Func<'_>, a: Val, b: Val) -> bool {
    if a == b {
        return true;
    }
    let (Some(x), Some(y)) = (f.def_inst(a), f.def_inst(b)) else {
        return false;
    };
    f.op(x) == f.op(y)
        && f.args(x).len() == f.args(y).len()
        && (0..f.args(x).len()).all(|k| chain_eq(f, f.args(x)[k], f.args(y)[k]))
}

/// Keep or copy what is live across each call, and give later uses the
/// copies; whether anything changed.
fn boundaries_round(
    f: &mut Func<'_>,
    origin: &mut FastMap<Val, Val>,
    made: &mut Vec<Inst>,
) -> Result<bool, String> {
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
            made: &mut *made,
            memo: FastMap::default(),
            // The copies go after the resume block's `Resume`.
            at: 1,
        };
        // A table's pointer stays live across the call with its table named
        // by the call: the collector sees the table, not the pointer.
        for v in live[b.idx()].iter().collect::<Vec<_>>() {
            if let Some(t) = table_of(f, Val(v as u32))
                && !rooted(f, &cx.kept, t)
                && keep_unnamed(f, &cfg, b, t).is_ok()
            {
                cx.kept.push(same_reg(f, t));
                changed = true;
            }
        }
        for v in live[b.idx()].iter().collect::<Vec<_>>() {
            let v = Val(v as u32);
            // Constants and rooted pointers stay live: the allocator makes
            // them again.
            if cx.kept.contains(&same_reg(f, v))
                || is_const(f, v)
                || table_of(f, v).is_some_and(|t| rooted(f, &cx.kept, t))
            {
                continue;
            }
            let c = match remat(f, v, &mut cx) {
                Some(c) => c,
                None => {
                    keep_unnamed(f, &cfg, b, v)?;
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

/// Move each pure copy whose uses, snapshot entries included, are all in one
/// later block its resume block dominates to where that block starts, when
/// no path between passes a call (nothing new is then live across one) and
/// the block is in no loop the resume block is not.
fn place_copies(f: &mut Func<'_>, made: &[Inst]) {
    const NONE: u32 = u32::MAX;
    const MANY: u32 = u32::MAX - 1;
    let cfg = f.cfg();
    let mut loops: Vec<Vec<Block>> = vec![Vec::new(); f.blocks.len()];
    for (h, body) in natural_loops(f) {
        for b in body {
            if !loops[b.idx()].contains(&h) {
                loops[b.idx()].push(h);
            }
        }
    }
    // Whether a block between `b` and `c` (reached from one, reaching the
    // other, neither passing `b`) resumes from a call.
    let call_between = |f: &Func<'_>, b: Block, c: Block| {
        let mut fwd = vec![false; f.blocks.len()];
        let mut work: Vec<Block> = f.succs(b).collect();
        while let Some(x) = work.pop() {
            if x != b && !fwd[x.idx()] {
                fwd[x.idx()] = true;
                if x != c {
                    work.extend(f.succs(x));
                }
            }
        }
        let mut seen = vec![false; f.blocks.len()];
        let mut work: Vec<Block> = cfg.preds(c).to_vec();
        while let Some(x) = work.pop() {
            if x == b || seen[x.idx()] {
                continue;
            }
            seen[x.idx()] = true;
            if fwd[x.idx()] && f.blocks[x.idx()].resume {
                return true;
            }
            work.extend_from_slice(cfg.preds(x));
        }
        false
    };
    loop {
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
        for &i in made {
            let d = &f.insts[i.idx()];
            let b = d.block;
            if f.blocks[b.idx()].dead || !d.op.is_pure() || d.rn != 1 || !f.insts_of(b).contains(&i)
            {
                continue;
            }
            let u = used_in[f.result(i).idx()];
            if u == NONE || u == MANY || u == b.0 {
                continue;
            }
            let c = Block(u);
            if !f.blocks[c.idx()].resume
                && cfg.dominates(b, c)
                && loops[c.idx()].iter().all(|h| loops[b.idx()].contains(h))
                && !call_between(f, b, c)
            {
                moves.push((i, c));
            }
        }
        if moves.is_empty() {
            return;
        }
        let mut moved = vec![false; f.insts.len()];
        for &(i, _) in &moves {
            moved[i.idx()] = true;
        }
        for &b in &cfg.rpo {
            f.retain(b, |_, i| !moved[i.idx()]);
        }
        // In reverse, so an operand moved with its use lands before it.
        for &(i, c) in moves.iter().rev() {
            f.insert(c, 0, i);
        }
    }
}

/// Name `v`, which no register holds after the call into `b`, in the call's
/// snapshot as `KEPT`, so the collector traces its spill slot.
fn keep_unnamed(f: &mut Func<'_>, cfg: &CfgInfo, b: Block, v: Val) -> Result<(), String> {
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
    if !matches!(f.op(t), Op::Call { .. } | Op::CallMm { .. }) {
        return Err(none());
    }
    let d = f.snaps[f.insts[t.idx()].snap as usize];
    let from = f.snap_pool.len();
    f.snap_pool
        .extend_from_within(d.entries.start as usize..(d.entries.start + d.entries.len) as usize);
    f.snap_pool.push((crate::jit::ir::KEPT, v));
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

/// The table whose pointer `v` is: live across a call when the call names
/// the table (the rooting rule).
pub(crate) fn table_of(f: &Func<'_>, v: Val) -> Option<Val> {
    let d = f.def_inst(v)?;
    (f.op(d) == Op::Unbox(Rep::Ptr)).then(|| f.args(d)[0])
}

/// Whether table `t` stays reachable across a call keeping `kept`: named
/// by it, or a constant (the code holds those).
pub(crate) fn rooted(f: &Func<'_>, kept: &[Val], t: Val) -> bool {
    kept.contains(&same_reg(f, t)) || is_const(f, t)
}

/// Whether `v` is a constant or shares one's register (`same_reg`), which
/// needs no spill slot across a call.
pub(crate) fn is_const(f: &Func<'_>, v: Val) -> bool {
    matches!(
        f.def_op(same_reg(f, v)),
        Some(
            Op::KVal(_)
                | Op::KObj(_)
                | Op::KPtr(_)
                | Op::KI32(_)
                | Op::KI64(_)
                | Op::KB1(_)
                | Op::KF64(_)
        )
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
    /// Every copy made, for `place_copies`.
    made: &'a mut Vec<Inst>,
    /// Copies made so far.
    memo: FastMap<Val, Val>,
    /// Where the next copy goes.
    at: usize,
}

impl After<'_> {
    fn insert(&mut self, f: &mut Func<'_>, i: Inst) {
        f.insert(self.b, self.at, i);
        self.at += 1;
        self.made.push(i);
    }
}

/// `v` computed again after the call, its operands first.
fn remat(f: &mut Func<'_>, v: Val, cx: &mut After<'_>) -> Option<Val> {
    if let Some(&c) = cx.memo.get(&v) {
        return Some(c);
    }
    if is_const(f, v) || table_of(f, v).is_some_and(|t| rooted(f, &cx.kept, t)) {
        return Some(v);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::ir::{BlockCall, ExitKind, List, RegionMeta};

    fn inst(f: &mut Func<'_>, b: Block, op: Op, args: &[Val]) -> Val {
        let i = f.make_inst(op, args, None, ExitTag::Type);
        f.append(b, i);
        f.result(i)
    }

    /// A call from `b` to resume block `r`, its snapshot naming `named`.
    fn call(f: &mut Func<'_>, b: Block, r: Block, named: &[(u8, Val)]) -> Inst {
        let from = f.snap_pool.len();
        f.snap_pool.extend_from_slice(named);
        let s = f.finish_snap(0, 0, ExitKind::After, from);
        let op = Op::Call {
            a: 0,
            nargs: 0,
            c: 1,
            pc: 0,
            proto: 0,
        };
        let t = f.make_inst(op, &[], Some(s), ExitTag::Type);
        f.append(b, t);
        let args = List::default();
        f.set_edges(t, &[BlockCall { target: r, args }]);
        f.blocks[r.idx()].resume = true;
        let i = f.make_inst(Op::Resume { c: 1 }, &[], None, ExitTag::Type);
        f.append(r, i);
        t
    }

    /// The pointer of a table that is a guarded constant is live across a
    /// call that names nothing: naming the table, which `kept_across` drops
    /// as a constant, repeated every round.
    #[test]
    fn pointers_of_guarded_constant_tables_cross_calls() {
        let mut f = Func::new(RegionMeta {
            entry_pc: 0,
            loop_entry: false,
            params: 0,
        });
        let [e, r] = [(); 2].map(|_| f.new_block());
        f.entry = e;
        let k = inst(&mut f, e, Op::KObj(0), &[]);
        let s = f.finish_snap(0, 0, ExitKind::Before, f.snap_pool.len());
        let g = f.make_inst(Op::Guard(TypeSet::TAB), &[k], Some(s), ExitTag::Type);
        f.append(e, g);
        let t = f.result(g);
        let p = inst(&mut f, e, Op::Unbox(Rep::Ptr), &[t]);
        let c = call(&mut f, e, r, &[]);
        inst(&mut f, r, Op::TabASize, &[p]);
        let ret = f.make_inst(Op::Return { a: 0, n: 0 }, &[], None, ExitTag::Type);
        f.append(r, ret);
        call_boundaries(&mut f).unwrap();
        assert!(f.snap_entries(c).is_empty());
        crate::jit::ir::verify::verify(&f).unwrap();
    }

    fn term(f: &mut Func<'_>, b: Block, op: Op, args: &[Val], to: &[(Block, &[Val])]) {
        let i = f.make_inst(op, args, None, ExitTag::Type);
        f.append(b, i);
        let calls: Vec<BlockCall> = (to.iter())
            .map(|&(target, args)| BlockCall {
                target,
                args: f.new_vlist(args),
            })
            .collect();
        f.set_edges(i, &calls);
    }

    /// A loop's parameter joining copies of one refined constant is that
    /// constant made again where the loop starts, after the last round: live
    /// across the loop's call as the constant whose register it shares.
    #[test]
    fn folded_refined_constants_cross_calls() {
        let mut f = Func::new(RegionMeta {
            entry_pc: 0,
            loop_entry: false,
            params: 0,
        });
        let [e, h, r, x] = [(); 4].map(|_| f.new_block());
        f.entry = e;
        let five = |f: &mut Func<'_>, b: Block| {
            let k = inst(f, b, Op::KVal(0xffff_ffff_0000_0005), &[]);
            let k = inst(f, b, Op::Refine(TypeSet::SMALL), &[k]);
            inst(f, b, Op::Unbox(Rep::I32), &[k])
        };
        let u = five(&mut f, e);
        term(&mut f, e, Op::Jump, &[], &[(h, &[u])]);
        let p = f.add_param(h, Ty::I32);
        call(&mut f, h, r, &[]);
        let v = inst(&mut f, r, Op::Box, &[p]);
        let st = f.make_inst(Op::Store(0), &[v], None, ExitTag::Type);
        f.append(r, st);
        let u = five(&mut f, r);
        let c = inst(&mut f, r, Op::KB1(true), &[]);
        term(&mut f, r, Op::Br, &[c], &[(h, &[u]), (x, &[])]);
        term(&mut f, x, Op::Return { a: 0, n: 0 }, &[], &[]);
        call_boundaries(&mut f).unwrap();
        assert!(f.params(h).is_empty());
        crate::jit::ir::verify::verify(&f).unwrap();
    }
}
