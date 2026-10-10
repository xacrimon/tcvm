//! Global value numbering (9.4) over the dominator tree: pure instructions
//! and type guards (a value's type never changes, so a dominating equal guard
//! makes a later one redundant). Scopes restart at resume blocks, so no value
//! is shared across a call.

use crate::jit::FastMap;
use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, Func, Inst, Val};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    op: Op,
    n: u8,
    args: [Val; 3],
}

fn key_of(f: &Func<'_>, i: Inst) -> Option<Key> {
    let op = f.op(i);
    let numberable = op.is_pure() && f.insts[i.idx()].rn == 1 || matches!(op, Op::Guard(_));
    let a = f.args(i);
    if !numberable || a.len() > 3 {
        return None;
    }
    let mut args = [Val(0); 3];
    args[..a.len()].copy_from_slice(a);
    Some(Key {
        op,
        n: a.len() as u8,
        args,
    })
}

pub(crate) fn gvn(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    // The dominator tree's children, by block.
    let n = f.blocks.len();
    let mut at = vec![0u32; n + 1];
    for &b in &cfg.rpo {
        if b != f.entry
            && let Some(p) = cfg.idom[b.idx()]
        {
            at[p.idx() + 1] += 1;
        }
    }
    for k in 0..n {
        at[k + 1] += at[k];
    }
    let mut fill = at.clone();
    let mut children = vec![Block(0); at[n] as usize];
    for &b in &cfg.rpo {
        if b != f.entry
            && let Some(p) = cfg.idom[b.idx()]
        {
            children[fill[p.idx()] as usize] = b;
            fill[p.idx()] += 1;
        }
    }
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut removed = vec![false; f.insts.len()];
    let mut changed = false;
    // Each number with the scope that made it; an undo log restores the
    // table when a scope closes.
    let mut table: FastMap<Key, (Val, u32)> = FastMap::default();
    let mut undo: Vec<(Key, Option<(Val, u32)>)> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    // (block, done, lowest visible scope)
    let mut stack: Vec<(Block, bool, usize)> = vec![(f.entry, false, 0)];
    while let Some((b, done, barrier)) = stack.pop() {
        if done {
            let m = marks.pop().unwrap();
            while undo.len() > m {
                let (k, old) = undo.pop().unwrap();
                match old {
                    Some(o) => table.insert(k, o),
                    None => table.remove(&k),
                };
            }
            continue;
        }
        marks.push(undo.len());
        let scope = marks.len() - 1;
        let lo = if f.blocks[b.idx()].resume {
            scope
        } else {
            barrier
        };
        for k in 0..f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            // Arguments through earlier replacements.
            for a in f.args_mut(i) {
                while map[a.idx()] != *a {
                    *a = map[a.idx()];
                }
            }
            let Some(key) = key_of(f, i) else {
                continue;
            };
            let r = f.result(i);
            match table.get(&key) {
                Some(&(v, s)) if s as usize >= lo => {
                    map[r.idx()] = v;
                    removed[i.idx()] = true;
                    changed = true;
                }
                _ => {
                    let old = table.insert(key, (r, scope as u32));
                    undo.push((key, old));
                }
            }
        }
        stack.push((b, true, lo));
        for &c in children[at[b.idx()] as usize..at[b.idx() + 1] as usize]
            .iter()
            .rev()
        {
            stack.push((c, false, lo));
        }
    }
    if changed {
        for b in 0..f.blocks.len() {
            f.retain(Block(b as u32), |_, i| !removed[i.idx()]);
        }
        f.apply_replacements(&mut map);
    }
    changed
}
