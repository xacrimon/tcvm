//! Use-driven speculation (9.1): the type guards of a value's uses agree on a
//! set, and the value is guarded once where it is defined instead, so
//! inference carries the type to every use.

use crate::jit::FastMap;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, CfgInfo, ExitKind, Func, Inst, Snap, Val, ValDef};

/// What is known of the values a region is entered with.
#[derive(Default)]
pub(crate) struct EntryKinds {
    /// For a loop entry, the kind of each live register's value in the frame
    /// the compile was triggered from (J7), by register; empty when unknown.
    pub(crate) frame: Vec<TypeSet>,
    /// Kinds failed entry guards saw, by register.
    pub(crate) seen: Vec<(u8, TypeSet)>,
}

/// A value to guard at its definition.
struct Cand {
    v: Val,
    set: TypeSet,
    /// The register of an entry value; guarded in the prologue.
    entry: Option<u8>,
    snap: Option<Snap>,
}

/// Guard values at their definitions; whether any was. Types must be
/// inferred.
pub(crate) fn speculate(f: &mut Func<'_>, kinds: &EntryKinds) -> bool {
    let exp = expectations(f);
    let mut cands = Vec::new();
    for (k, vd) in f.vals.iter().enumerate() {
        let ValDef::Inst(i, _) = vd.def else {
            continue;
        };
        let (e, t) = (exp[k], vd.ty);
        let v = Val(k as u32);
        let block = f.insts[i.idx()].block;
        if t.rep != Rep::Val || block.0 == u32::MAX || f.blocks[block.idx()].dead {
            continue;
        }
        if block == f.entry
            && let Op::Load(r) = f.op(i)
        {
            if let Some(set) = entry_set(f, kinds, r, e)
                && t.set.intersects(set)
                && !t.within(set)
            {
                cands.push(Cand {
                    v,
                    set,
                    entry: Some(r),
                    snap: None,
                });
            }
            continue;
        }
        if e == TypeSet::ANY || !t.set.intersects(e) || t.within(e) {
            continue;
        }
        if let Some(&(_, s)) = f.def_snaps.iter().find(|(x, _)| *x == v) {
            cands.push(Cand {
                v,
                set: e,
                entry: None,
                snap: Some(s),
            });
        }
    }
    if cands.is_empty() {
        return false;
    }
    let ant = anticipated(f, &cands);
    let cfg = f.cfg();
    // Where each guard goes: in the prologue for an entry value; after the
    // definition when every path from there reaches a use expecting the set;
    // else at the earliest blocks the definition dominates from which every
    // path does, so no guard deopts an execution a use's guard would not
    // have (9.1). Values without any keep their uses' guards.
    let mut places: Vec<(usize, Option<Block>)> = Vec::new();
    for (ci, c) in cands.iter().enumerate() {
        if c.entry.is_some() || ant.after_def[ci] {
            places.push((ci, None));
            continue;
        }
        let d = f.def_inst(c.v).unwrap();
        let db = f.insts[d.idx()].block;
        for &b in &cfg.rpo {
            if b == db || !ant.at(b, ci) || !cfg.dominates(db, b) {
                continue;
            }
            let mut x = cfg.idom[b.idx()].unwrap();
            while x != db && !ant.at(x, ci) {
                x = cfg.idom[x.idx()].unwrap();
            }
            if x == db && !barrier(f, &cfg, d, b) {
                places.push((ci, Some(b)));
            }
        }
    }
    if places.is_empty() {
        return false;
    }
    // Each guard tests the value itself and exits with the frame of the
    // value's definition, which the replacements below must not touch.
    let frames: Vec<(u32, ExitKind, Vec<(u8, Val)>)> = cands
        .iter()
        .map(|c| match c.snap {
            Some(s) => (
                f.snaps[s.idx()].pc,
                f.snaps[s.idx()].kind,
                f.entries(s.0).to_vec(),
            ),
            None => (f.meta.entry_pc, ExitKind::Before, Vec::new()),
        })
        .collect();
    let mut guards = Vec::with_capacity(places.len());
    for &(ci, at) in &places {
        let c = &cands[ci];
        let tag = if c.entry.is_some() {
            ExitTag::Entry
        } else {
            ExitTag::Type
        };
        let g = f.make_inst(Op::Guard(c.set), &[c.v], None, tag);
        match at {
            None if c.entry.is_some() => f.insert_before_term(f.entry, g),
            None => {
                let d = f.def_inst(c.v).unwrap();
                let b = f.insts[d.idx()].block;
                let k = f.insts_of(b).iter().position(|&x| x == d).unwrap();
                f.insert(b, k + 1, g);
            }
            Some(b) => f.insert(b, 0, g),
        }
        guards.push(g);
    }
    // A guard after the definition dominates every use; one in a block,
    // the uses that block dominates.
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    for (&(ci, at), &g) in places.iter().zip(&guards) {
        if at.is_none() {
            map[cands[ci].v.idx()] = f.result(g);
        }
    }
    f.apply_replacements(&mut map);
    for (&(ci, at), &g) in places.iter().zip(&guards) {
        let Some(b) = at else {
            continue;
        };
        let (v, r) = (cands[ci].v, f.result(g));
        for &x in &cfg.rpo {
            if !cfg.dominates(b, x) {
                continue;
            }
            for k in 0..f.blocks[x.idx()].insts.len() {
                let i = f.insts_of(x)[k];
                if i != g {
                    f.replace_uses(i, v, r);
                }
            }
        }
    }
    for (&(ci, _), &g) in places.iter().zip(&guards) {
        f.args_mut(g)[0] = cands[ci].v;
        let (pc, kind, entries) = &frames[ci];
        let from = f.snap_pool.len();
        f.snap_pool.extend_from_slice(entries);
        let s = f.finish_snap(*pc, *kind, from);
        f.insts[g.idx()].snap = s.0;
    }
    true
}

/// The set every informative use of each value expects: the intersection
/// of its type guards' sets and of what the block parameters it flows into
/// expect. `ANY` is no expectation, the empty set a conflict.
fn expectations(f: &Func<'_>) -> Vec<TypeSet> {
    let mut exp = vec![TypeSet::ANY; f.vals.len()];
    let cfg = f.cfg();
    for &b in &cfg.rpo {
        for &i in f.insts_of(b) {
            if let Some((a, set)) = informative(f, i) {
                exp[a.idx()] &= set;
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            for (k, &p) in f.params(b).iter().enumerate() {
                let e = exp[p.idx()];
                if e == TypeSet::ANY {
                    continue;
                }
                for &(term, j) in cfg.incoming(b) {
                    let a = f.edge_args(term, j as usize)[k];
                    let n = exp[a.idx()] & e;
                    if n != exp[a.idx()] {
                        exp[a.idx()] = n;
                        changed = true;
                    }
                }
            }
        }
    }
    exp
}

/// The value a type guard tests and the set it expects.
fn informative(f: &Func<'_>, i: Inst) -> Option<(Val, TypeSet)> {
    match f.op(i) {
        Op::Guard(set) if f.insts[i.idx()].tag == ExitTag::Type => Some((f.args(i)[0], set)),
        _ => None,
    }
}

/// The set the register's entry value is guarded to, if any. An
/// expectation of its uses holds when the kinds failed entry guards saw and,
/// at a loop entry, the frame's value fit it; a register without one is
/// typed at a loop entry by the frame's kind and the seen ones (J7).
fn entry_set(f: &Func<'_>, kinds: &EntryKinds, r: u8, e: TypeSet) -> Option<TypeSet> {
    let seen = kinds
        .seen
        .iter()
        .filter(|&&(x, _)| x == r)
        .fold(TypeSet::empty(), |a, &(_, s)| a | s);
    let frame = (f.meta.loop_entry && !kinds.frame.is_empty()).then(|| {
        kinds
            .frame
            .get(r as usize)
            .copied()
            .unwrap_or(TypeSet::empty())
    });
    if e == TypeSet::ANY {
        let set = frame.filter(|k| !k.is_empty())? | seen;
        return (set != TypeSet::ANY).then_some(set);
    }
    let fits = e.contains(seen) && frame.is_none_or(|k| !k.is_empty() && e.contains(k));
    fits.then_some(e)
}

/// Per block and candidate, whether every path from the block's start
/// reaches a use expecting a subset of the candidate's set before leaving
/// the region other than by an exit (the anticipability, or down-safety, of
/// partial redundancy elimination), and the same just after each
/// candidate's definition: one bit-vector dataflow for all candidates. A
/// value's names are itself and the parameters it flows into.
struct Ant {
    words: usize,
    at_start: Vec<u64>,
    after_def: Vec<bool>,
}

impl Ant {
    fn at(&self, b: Block, c: usize) -> bool {
        self.at_start[b.idx() * self.words + c / 64] >> (c % 64) & 1 != 0
    }
}

fn anticipated(f: &Func<'_>, cands: &[Cand]) -> Ant {
    let cfg = f.cfg();
    let rpo = &cfg.rpo;
    let w = cands.len().div_ceil(64);
    // The candidates each value names, and those each instruction defines.
    let mut names: FastMap<Val, Vec<usize>> = FastMap::default();
    let mut defs: FastMap<Inst, Vec<usize>> = FastMap::default();
    for (ci, c) in cands.iter().enumerate() {
        if c.entry.is_some() {
            continue;
        }
        defs.entry(f.def_inst(c.v).unwrap()).or_default().push(ci);
        let mut mine = vec![c.v];
        let mut grew = true;
        while grew {
            grew = false;
            for &b in rpo {
                for (k, &p) in f.params(b).iter().enumerate() {
                    if !mine.contains(&p)
                        && cfg
                            .incoming(b)
                            .iter()
                            .any(|&(t, j)| mine.contains(&f.edge_args(t, j as usize)[k]))
                    {
                        mine.push(p);
                        grew = true;
                    }
                }
            }
        }
        for v in mine {
            names.entry(v).or_default().push(ci);
        }
    }
    let mut at_start = vec![!0u64; f.blocks.len() * w];
    let mut after_def = vec![false; cands.len()];
    let mut cur = vec![0u64; w];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().rev() {
            let insts = f.insts_of(b);
            let term = *insts.last().unwrap();
            match f.op(term) {
                Op::Deopt => cur.fill(!0),
                Op::Jump | Op::Br => {
                    cur.fill(!0);
                    for s in f.succs(b) {
                        for (x, y) in cur.iter_mut().zip(&at_start[s.idx() * w..]) {
                            *x &= y;
                        }
                    }
                }
                _ => cur.fill(0),
            }
            for &i in insts.iter().rev() {
                if let Some(cs) = defs.get(&i) {
                    for &c in cs {
                        after_def[c] = cur[c / 64] >> (c % 64) & 1 != 0;
                    }
                }
                if let Some((a, set)) = informative(f, i)
                    && let Some(cs) = names.get(&a)
                {
                    for &c in cs {
                        if cands[c].set.contains(set) {
                            cur[c / 64] |= 1 << (c % 64);
                        }
                    }
                }
            }
            let row = &mut at_start[b.idx() * w..(b.idx() + 1) * w];
            if row != cur.as_slice() {
                row.copy_from_slice(&cur);
                changed = true;
            }
        }
    }
    Ant {
        words: w,
        at_start,
        after_def,
    }
}

/// Whether a path from `d` to the start of `b` writes or calls: a guard at
/// `b` exits with `d`'s frame, and the interpreter runs what lies between
/// again.
fn barrier(f: &Func<'_>, cfg: &CfgInfo, d: Inst, b: Block) -> bool {
    let writes = |i: &Inst| f.op(*i).effects().writes();
    let db = f.insts[d.idx()].block;
    let k = f.insts_of(db).iter().position(|&x| x == d).unwrap();
    if f.insts_of(db)[k + 1..].iter().any(writes) {
        return true;
    }
    let mut seen = vec![false; f.blocks.len()];
    let mut work = cfg.preds(b).to_vec();
    while let Some(x) = work.pop() {
        if x == db || std::mem::replace(&mut seen[x.idx()], true) {
            continue;
        }
        if f.blocks[x.idx()].resume || f.insts_of(x).iter().any(writes) {
            return true;
        }
        work.extend_from_slice(cfg.preds(x));
    }
    false
}
