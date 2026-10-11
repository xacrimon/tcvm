//! Global value numbering (9.4) over the dominator tree: pure instructions,
//! type guards (a value's type never changes, so a dominating equal guard
//! makes a later one redundant), and loads and memory guards keyed by the
//! version of the memory they read (8.4); a call writes every class, and
//! scopes restart after one.

use crate::jit::FastMap;
use crate::jit::ir::ops::{Flags, Mem, Op};
use crate::jit::ir::{Block, Func, Inst, Val};

const CLASSES: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    op: Op,
    n: u8,
    args: [Val; 3],
    /// The versions of the memory a read depends on.
    mem: [u32; 2],
}

/// The one class of a keyed access.
fn class_of(m: Mem) -> usize {
    debug_assert_eq!(m.bits().count_ones(), 1);
    m.bits().trailing_zeros() as usize
}

/// The versions at a point: per class, renewed by its unkeyed writes; per
/// class, renewed by any write to it; and per key within a class. Each
/// renewal takes a number above all before it, so a later point sees the
/// same versions as a dominating one exactly when nothing was renewed.
#[derive(Clone, Default)]
struct Versions {
    class: [u32; CLASSES],
    any: [u32; CLASSES],
    keys: FastMap<(u8, u32), u32>,
}

impl Versions {
    /// What a read of `reads` (keyed by `key`) depends on: the key's and
    /// its class's for a keyed read, the newest of any write to each class
    /// otherwise.
    fn version(&self, reads: Mem, key: Option<u32>) -> [u32; 2] {
        match key {
            Some(k) => {
                let c = class_of(reads);
                let kv = self.keys.get(&(c as u8, k)).copied().unwrap_or(0);
                [self.class[c], kv]
            }
            None => {
                let mut v = 0;
                for c in 0..CLASSES {
                    if reads.bits() >> c & 1 != 0 {
                        v = v.max(self.any[c]);
                    }
                }
                [v, 0]
            }
        }
    }

    fn renew(&mut self, mask: Mem, fresh: &mut u32) {
        for c in 0..CLASSES {
            if mask.bits() >> c & 1 != 0 {
                *fresh += 1;
                self.class[c] = *fresh;
                self.any[c] = *fresh;
                self.keys.retain(|&(x, _), _| x as usize != c);
            }
        }
    }

    fn renew_key(&mut self, m: Mem, key: u32, fresh: &mut u32) {
        let c = class_of(m);
        *fresh += 1;
        self.keys.insert((c as u8, key), *fresh);
        self.any[c] = *fresh;
    }

    fn write(&mut self, op: Op, fresh: &mut u32) {
        let w = op.effects().writes;
        if w.is_empty() {
            return;
        }
        match op.mem_key() {
            Some(k) => self.renew_key(w, k, fresh),
            None => self.renew(w, fresh),
        }
    }
}

fn key_of(f: &Func<'_>, i: Inst, mem: &Versions) -> Option<Key> {
    let op = f.op(i);
    let a = f.args(i);
    if a.len() > 3 {
        return None;
    }
    let e = op.effects();
    let numberable = match op {
        // Allocations between two checks are not memory it reads.
        Op::GcCheck => false,
        // What these test of values never changes.
        Op::Guard(_) | Op::GuardSame | Op::GuardTrue | Op::GuardFalse => true,
        _ => {
            e.is_pure() && f.insts[i.idx()].rn == 1
                || e.reads_memory()
                    && !e.writes()
                    && !e.flags.intersects(
                        Flags::MAY_ALLOC | Flags::SIDE_EFFECT | Flags::TERMINATOR | Flags::TAILOUT,
                    )
        }
    };
    if !numberable {
        return None;
    }
    let mut args = [Val(0); 3];
    args[..a.len()].copy_from_slice(a);
    Some(Key {
        op,
        n: a.len() as u8,
        args,
        mem: if e.reads_memory() {
            mem.version(e.reads, op.mem_key())
        } else {
            [0, 0]
        },
    })
}

/// The load a store makes redundant, and the value it would read.
fn forwarded(f: &Func<'_>, i: Inst) -> Option<(Op, Vec<Val>, Val)> {
    let a = f.args(i);
    match f.op(i) {
        Op::Store(r) => Some((Op::Load(r), vec![], a[0])),
        Op::TabStore { loc, key } => Some((Op::TabLoad { loc, key }, vec![a[0]], a[1])),
        Op::ArrStore => Some((Op::ArrLoad, vec![a[0], a[1]], a[2])),
        Op::CellStore => Some((Op::CellLoad, vec![a[0]], a[1])),
        _ => None,
    }
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
    // The classes each block writes.
    let mut wmask = vec![Mem::empty(); n];
    for &b in &cfg.rpo {
        for &i in f.insts_of(b) {
            wmask[b.idx()] |= f.op(i).effects().writes;
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
    // The versions at each block's end, for its dominator-tree children.
    let mut end: Vec<Option<Versions>> = vec![None; n];
    let mut fresh = 0u32;
    let mut seen = vec![false; n];
    let mut work: Vec<Block> = Vec::new();
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
        // Scopes restart after a call: a value shared across one must then
        // stay in the frame or be computed again (C1).
        let lo = if f.blocks[b.idx()].resume {
            scope
        } else {
            barrier
        };
        // The dominator's versions, renewed for what any path from it
        // here writes (a loop header: its body).
        let mut mem = match cfg.idom[b.idx()].filter(|_| b != f.entry) {
            Some(d) => {
                let mut m = end[d.idx()].clone().unwrap_or_default();
                seen.fill(false);
                seen[d.idx()] = true;
                work.clear();
                work.extend_from_slice(cfg.preds(b));
                let mut w = Mem::empty();
                while let Some(x) = work.pop() {
                    if std::mem::replace(&mut seen[x.idx()], true) {
                        continue;
                    }
                    w |= wmask[x.idx()];
                    work.extend_from_slice(cfg.preds(x));
                }
                m.renew(w, &mut fresh);
                m
            }
            None => Versions::default(),
        };
        for k in 0..f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            // Arguments through earlier replacements.
            for a in f.args_mut(i) {
                while map[a.idx()] != *a {
                    *a = map[a.idx()];
                }
            }
            if let Some(key) = key_of(f, i, &mem) {
                match table.get(&key) {
                    Some(&(v, s)) if s as usize >= lo => {
                        if f.insts[i.idx()].rn == 1 {
                            map[f.result(i).idx()] = v;
                        }
                        removed[i.idx()] = true;
                        changed = true;
                        continue;
                    }
                    _ => {
                        let r = if f.insts[i.idx()].rn == 1 {
                            f.result(i)
                        } else {
                            Val(u32::MAX)
                        };
                        let old = table.insert(key, (r, scope as u32));
                        undo.push((key, old));
                    }
                }
            }
            let op = f.op(i);
            if !op.effects().writes() {
                continue;
            }
            mem.write(op, &mut fresh);
            // A store answers the load of what it stored.
            if let Some((lop, largs, v)) = forwarded(f, i) {
                let mut args = [Val(0); 3];
                args[..largs.len()].copy_from_slice(&largs);
                let key = Key {
                    op: lop,
                    n: largs.len() as u8,
                    args,
                    mem: mem.version(lop.effects().reads, lop.mem_key()),
                };
                let old = table.insert(key, (v, scope as u32));
                undo.push((key, old));
            }
        }
        end[b.idx()] = Some(mem);
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
