//! Shape checks on tables known to be in a stable shape (ws2 §6.2). A load
//! from a slot that tracks its tables' shape is typed `Refine::Shape`, which
//! inference carries through block parameters whose arguments all have it.
//! A table stays in a stable shape: the first to leave it invalidates the
//! region, whose frames leave at their next lazy exit. So the fact holds
//! wherever no table may have moved since the last lazy exit.

use crate::jit::deps::SHAPE_GROUP;
use crate::jit::ir::ops::{Mem, Op};
use crate::jit::ir::types::Refine;
use crate::jit::ir::{DepOn, Func, Inst};

/// Whether a table may have moved since the last lazy exit, after `i`.
fn after(f: &Func<'_>, i: Inst, moved: bool) -> bool {
    match f.op(i) {
        Op::LazyExit => false,
        // Compiled moves leave only unstable shapes (`tables::trans_store`).
        Op::TabSetShape(_) => moved,
        op => moved || op.effects().writes.contains(Mem::SHAPE),
    }
}

/// Fold the shape checks of such tables; returns whether any went.
pub(crate) fn fold(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    let mut out = vec![false; f.blocks.len()];
    let moved_in = |out: &[bool], b| cfg.preds(b).iter().any(|p| out[p.idx()]);
    loop {
        let mut changed = false;
        for &b in &cfg.rpo {
            let mut m = moved_in(&out, b);
            for &i in f.insts_of(b) {
                m = after(f, i, m);
            }
            if m != out[b.idx()] {
                out[b.idx()] = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut removed = Vec::new();
    let mut used = Vec::new();
    for &b in &cfg.rpo {
        let mut m = moved_in(&out, b);
        for k in 0..f.insts_of(b).len() {
            let i = f.insts_of(b)[k];
            let op = f.op(i);
            let known = match f.args(i).first() {
                Some(&a) if !m => match f.vals[a.idx()].ty.refine {
                    Refine::Shape(s) => Some(s),
                    _ => None,
                },
                _ => None,
            };
            m = after(f, i, m);
            let Some(s) = known else {
                continue;
            };
            let holds = match op {
                Op::GuardShape(k) => {
                    if k == s {
                        removed.push(i);
                    }
                    continue;
                }
                Op::GuardShapeIn(set) => {
                    if f.shape_sets[set as usize].contains(&s) {
                        removed.push(i);
                        used.push(s);
                    }
                    continue;
                }
                Op::ShapeIs(k) => k == s,
                Op::ShapeIn(set) => f.shape_sets[set as usize].contains(&s),
                _ => continue,
            };
            f.insts[i.idx()].op = Op::KB1(holds);
            f.insts[i.idx()].an = 0;
            used.push(s);
        }
    }
    for &i in &removed {
        if let Op::GuardShape(k) = f.op(i) {
            used.push(k);
        }
    }
    used.sort_unstable();
    used.dedup();
    for &k in &used {
        let shape = f.shapes[k as usize];
        f.depend(DepOn::Shape(shape), SHAPE_GROUP);
    }
    if !removed.is_empty() {
        for b in 0..f.blocks.len() {
            f.retain(crate::jit::ir::Block(b as u32), |_, i| {
                !removed.contains(&i)
            });
        }
    }
    !used.is_empty()
}
