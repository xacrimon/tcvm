//! Global value numbering (9.4) over the dominator tree: pure instructions
//! and type guards (a value's type never changes, so a dominating equal guard
//! makes a later one redundant). Scopes restart at resume blocks, so no value
//! is shared across a call.

use std::collections::HashMap;

use crate::jit::ir::ops::Op;
use crate::jit::ir::verify::dominators;
use crate::jit::ir::{Block, Func, Inst, Val};

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    op: Op,
    args: Vec<Val>,
}

fn key_of(f: &Func<'_>, i: Inst) -> Option<Key> {
    let op = f.op(i);
    let numberable = op.is_pure() && f.insts[i.idx()].rn == 1 || matches!(op, Op::Guard(_));
    if !numberable {
        return None;
    }
    Some(Key {
        op,
        args: f.args(i).to_vec(),
    })
}

pub(crate) fn gvn(f: &mut Func<'_>) -> bool {
    let rpo = f.rpo();
    let preds = f.preds();
    let idom = dominators(f, &rpo, &preds);
    let mut children: Vec<Vec<Block>> = vec![Vec::new(); f.blocks.len()];
    for &b in &rpo {
        if b == f.entry {
            continue;
        }
        if let Some(p) = idom[b.idx()] {
            children[p.idx()].push(b);
        }
    }
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut remove: Vec<Inst> = Vec::new();
    let mut table: Vec<HashMap<Key, Val>> = Vec::new();
    // Iterative dominator-tree walk with a scope per block.
    // (block, done, lowest visible scope)
    let mut stack: Vec<(Block, bool, usize)> = vec![(f.entry, false, 0)];
    while let Some((b, done, barrier)) = stack.pop() {
        if done {
            table.pop();
            continue;
        }
        table.push(HashMap::new());
        let scope = table.len() - 1;
        let lo = if f.blocks[b.idx()].resume {
            scope
        } else {
            barrier
        };
        let insts = f.blocks[b.idx()].insts.clone();
        for i in insts {
            // Arguments through earlier replacements.
            for a in f.args_mut(i) {
                while map[a.idx()] != *a {
                    *a = map[a.idx()];
                }
            }
            let Some(k) = key_of(f, i) else {
                continue;
            };
            let found = table[lo..=scope]
                .iter()
                .rev()
                .find_map(|t| t.get(&k).copied());
            let r = f.result(i);
            match found {
                Some(v) => {
                    map[r.idx()] = v;
                    remove.push(i);
                }
                None => {
                    table[scope].insert(k, r);
                }
            }
        }
        stack.push((b, true, lo));
        for &c in children[b.idx()].iter().rev() {
            stack.push((c, false, lo));
        }
    }
    let changed = !remove.is_empty();
    if changed {
        for b in 0..f.blocks.len() {
            f.blocks[b].insts.retain(|i| !remove.contains(i));
        }
        f.apply_replacements(&mut map);
    }
    changed
}
