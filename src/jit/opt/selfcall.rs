//! Self-calls' results: a region's call of its own prototype gets what one of
//! its returns gives, so a guard on the result that every return's value passes
//! is no test on the fast return. One landing guard tests it where anything
//! else comes back: the call's slow path and a level the callee went on in (an
//! exit, a dispatch).

use crate::dmm::Gc;
use crate::env::function::Prototype;
use crate::jit::ir::ops::{ExitTag, NO_PROTO, Op};
use crate::jit::ir::types::TypeSet;
use crate::jit::ir::{Block, Func, Inst};

/// Trust the results of the region's calls of `own` that want one; whether
/// any guard changed.
pub(crate) fn trust_results(f: &mut Func<'_>, own: Gc<'_, Prototype<'_>>) -> bool {
    if f.meta.loop_entry {
        return false;
    }
    let Some(gives) = returns_set(f) else {
        return false;
    };
    let mut sites: Vec<Inst> = Vec::new();
    let mut guards: Vec<(usize, Inst)> = Vec::new();
    for b in (0..f.blocks.len() as u32).map(Block) {
        for &i in f.insts_of(b) {
            if let Op::Call { proto, c: 2, .. } = f.op(i)
                && proto != NO_PROTO
                && std::ptr::addr_eq(Gc::as_ptr(f.protos[proto as usize]), Gc::as_ptr(own))
            {
                sites.push(i);
            }
        }
    }
    let results: Vec<_> = (sites.iter())
        .map(|&i| {
            let r = f.insts_of(f.edges(i)[0].target)[0];
            f.result(r)
        })
        .collect();
    for b in (0..f.blocks.len() as u32).map(Block) {
        for &i in f.insts_of(b) {
            if let Op::Guard(set) = f.op(i)
                && f.insts[i.idx()].tag == ExitTag::Type
                && set.contains(gives)
                && let Some(k) = results.iter().position(|&v| v == f.args(i)[0])
            {
                guards.push((k, i));
            }
        }
    }
    let mut changed = false;
    for (k, &v) in results.iter().enumerate() {
        let mine: Vec<Inst> = (guards.iter()).filter(|g| g.0 == k).map(|g| g.1).collect();
        let Some(&(_, snap)) = f.def_snaps.iter().find(|d| d.0 == v) else {
            continue;
        };
        if mine.is_empty() {
            continue;
        }
        let mut land = TypeSet::ANY;
        for &g in &mine {
            let Op::Guard(set) = f.op(g) else {
                unreachable!()
            };
            land &= set;
            f.insts[g.idx()].op = Op::Refine(set);
        }
        // The definition's snapshot names what the guards gave since.
        let given: Vec<_> = mine.iter().map(|&g| f.result(g)).collect();
        let snap = f.map_snap(snap, |x| if given.contains(&x) { v } else { x });
        let g = f.make_inst(Op::Guard(land), &[v], Some(snap), ExitTag::Type);
        f.insert(f.insts[f.def_inst(v).unwrap().idx()].block, 1, g);
        f.landings.push(g);
        changed = true;
    }
    changed
}

/// The set of every return's first value; none if one is unknown, or the
/// region tail-calls, whose callee returns to the caller itself.
fn returns_set(f: &Func<'_>) -> Option<TypeSet> {
    let mut set = TypeSet::empty();
    for b in (0..f.blocks.len() as u32).map(Block) {
        let insts = f.insts_of(b);
        for (k, &i) in insts.iter().enumerate() {
            match f.op(i) {
                Op::TailEnter { .. } => return None,
                Op::Return { n: 0, .. } => set |= TypeSet::NIL,
                Op::Return { a, .. } => {
                    let stored = insts[..k].iter().rev().find_map(|&j| match f.op(j) {
                        Op::Store(r) if r == a => Some(Some(f.args(j)[0])),
                        op if op.effects().writes.contains(crate::jit::ir::ops::Mem::SLOT) => {
                            Some(None)
                        }
                        _ => None,
                    });
                    set |= f.ty(stored.flatten()?).set;
                }
                _ => {}
            }
        }
    }
    Some(set)
}
