//! Use-driven speculation (9.1): the type guards of a value's uses agree on a
//! set, and the value is guarded once where it is defined instead, so
//! inference carries the type to every use.

use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{ExitKind, Func, Inst, Snap, Val, ValDef};

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
        if t.rep != Rep::Val || e == TypeSet::ANY || !t.set.intersects(e) || t.within(e) {
            continue;
        }
        let v = Val(k as u32);
        let block = f.insts[i.idx()].block;
        if block.0 == u32::MAX || f.blocks[block.idx()].dead {
            continue;
        }
        if block == f.entry
            && let Op::Load(r) = f.op(i)
        {
            if entry_agrees(f, kinds, r, e) {
                cands.push(Cand {
                    v,
                    set: e,
                    entry: Some(r),
                    snap: None,
                });
            }
        } else if let Some(&(_, s)) = f.def_snaps.iter().find(|(x, _)| *x == v) {
            cands.push(Cand {
                v,
                set: e,
                entry: None,
                snap: Some(s),
            });
        }
    }
    let safe = down_safe(f, &cands);
    let cands: Vec<Cand> = cands
        .into_iter()
        .zip(safe)
        .filter(|(c, s)| c.entry.is_some() || *s)
        .map(|(c, _)| c)
        .collect();
    if cands.is_empty() {
        return false;
    }
    let mut guards = Vec::with_capacity(cands.len());
    for c in &cands {
        let tag = c.entry.map_or(ExitTag::Type, ExitTag::Entry);
        let g = f.make_inst(Op::Guard(c.set), &[c.v], None, tag);
        if c.entry.is_some() {
            f.insert_before_term(f.entry, g);
        } else {
            let d = f.def_inst(c.v).unwrap();
            let b = f.insts[d.idx()].block;
            let at = f.insts_of(b).iter().position(|&x| x == d).unwrap();
            f.insert(b, at + 1, g);
        }
        guards.push(g);
    }
    // Each guard tests the value itself and exits with the frame of the
    // value's definition, which the replacement must not touch.
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
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    for (c, &g) in cands.iter().zip(&guards) {
        map[c.v.idx()] = f.result(g);
    }
    f.apply_replacements(&mut map);
    for ((c, &g), (pc, kind, entries)) in cands.into_iter().zip(&guards).zip(frames) {
        f.args_mut(g)[0] = c.v;
        let from = f.snap_pool.len();
        f.snap_pool.extend_from_slice(&entries);
        let s = f.finish_snap(pc, kind, from);
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

/// Whether the register's entry value may be speculated at `e`: the kinds
/// failed entry guards saw and, at a loop entry, the frame's value agree.
fn entry_agrees(f: &Func<'_>, kinds: &EntryKinds, r: u8, e: TypeSet) -> bool {
    if kinds.seen.iter().any(|&(x, s)| x == r && !e.contains(s)) {
        return false;
    }
    if f.meta.loop_entry && !kinds.frame.is_empty() {
        let k = kinds
            .frame
            .get(r as usize)
            .copied()
            .unwrap_or(TypeSet::empty());
        return !k.is_empty() && e.contains(k);
    }
    true
}

/// For each candidate, whether every path from just after its definition
/// reaches a use expecting a subset of its set before leaving the region
/// other than by an exit: a guard there deopts no execution that would not
/// have deopted anyway (the anticipability of partial redundancy
/// elimination). A value's names are itself and the parameters it flows
/// into.
fn down_safe(f: &Func<'_>, cands: &[Cand]) -> Vec<bool> {
    let cfg = f.cfg();
    let rpo = &cfg.rpo;
    let mut out = vec![false; cands.len()];
    for (ci, c) in cands.iter().enumerate() {
        if c.entry.is_some() {
            continue;
        }
        let mut names = vec![false; f.vals.len()];
        names[c.v.idx()] = true;
        let mut grew = true;
        while grew {
            grew = false;
            for &b in rpo {
                for (k, &p) in f.params(b).iter().enumerate() {
                    if !names[p.idx()]
                        && cfg
                            .incoming(b)
                            .iter()
                            .any(|&(t, j)| names[f.edge_args(t, j as usize)[k].idx()])
                    {
                        names[p.idx()] = true;
                        grew = true;
                    }
                }
            }
        }
        let def = f.def_inst(c.v).unwrap();
        let mut ds_in = vec![true; f.blocks.len()];
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().rev() {
                let insts = f.insts_of(b);
                let term = *insts.last().unwrap();
                let mut cur = match f.op(term) {
                    Op::Deopt => true,
                    Op::Jump | Op::Br => f.succs(b).all(|s| ds_in[s.idx()]),
                    _ => false,
                };
                for &i in insts.iter().rev() {
                    if i == def {
                        out[ci] = cur;
                    }
                    if let Some((a, set)) = informative(f, i)
                        && names[a.idx()]
                        && c.set.contains(set)
                    {
                        cur = true;
                    }
                }
                if cur != ds_in[b.idx()] {
                    ds_in[b.idx()] = cur;
                    changed = true;
                }
            }
        }
    }
    out
}
