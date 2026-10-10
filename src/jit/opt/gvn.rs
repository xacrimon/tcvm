//! Global value numbering (9.4) over the dominator tree: pure instructions,
//! type guards (a value's type never changes, so a dominating equal guard
//! makes a later one redundant), and loads and memory guards keyed by the
//! version of the memory they read (8.4). Scopes restart at resume blocks,
//! so no value is shared across a call.

use crate::jit::FastMap;
use crate::jit::ir::ops::{Effects, Op};
use crate::jit::ir::{Block, Func, Inst, Val};

/// Memory classes, each with a version that every write to it renews.
const SLOT: usize = 0;
pub(crate) const TAB: usize = 1;
const ARR: usize = 2;
const SHAPE: usize = 3;
const META: usize = 4;
const UPVAL: usize = 5;
const FRAME: usize = 6;
const CLASSES: usize = 7;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    op: Op,
    n: u8,
    args: [Val; 3],
    /// The class's version and the key's within it, for a read of memory.
    mem: [u32; 2],
}

/// The memory a read depends on: its class and, for a home slot or a named
/// slot, the key within it.
pub(crate) fn reads(op: Op) -> Option<(usize, Option<u32>)> {
    Some(match op {
        Op::Load(r) => (SLOT, Some(r as u32)),
        Op::TabLoad { key, .. } => (TAB, Some(key)),
        Op::ArrLoad => (ARR, None),
        Op::GuardShape(_) | Op::ShapeIs(_) | Op::TabASize | Op::TabArray | Op::TabHasRoom(_) => {
            (SHAPE, None)
        }
        Op::ClassBits(_) | Op::ClassIndex(_) => (META, None),
        Op::CellLoad => (UPVAL, None),
        Op::HasFlag(_) | Op::GuardNoFlag(_) => (FRAME, None),
        _ => return None,
    })
}

/// The classes an instruction writes, a bit each; a keyed write also renews
/// its key, which `write_key` names.
pub(crate) fn writes(op: Op) -> u32 {
    let e = op.effects();
    match op {
        Op::Store(_) => 1 << SLOT,
        Op::TabStore { .. } => 1 << TAB,
        Op::ArrStore => 1 << ARR,
        _ => {
            let mut m = 0;
            if e.has(Effects::W_SLOT) {
                m |= 1 << SLOT;
            }
            if e.has(Effects::W_TAB) {
                m |= 1 << TAB | 1 << ARR;
            }
            if e.has(Effects::W_SHAPE) {
                m |= 1 << SHAPE;
            }
            if e.has(Effects::W_META) {
                m |= 1 << META;
            }
            if e.has(Effects::W_UPVAL) {
                m |= 1 << UPVAL;
            }
            if e.has(Effects::W_FRAME) {
                m |= 1 << FRAME;
            }
            m
        }
    }
}

/// The key a keyed write renews within its class.
pub(crate) fn write_key(op: Op) -> Option<u32> {
    match op {
        Op::Store(r) => Some(r as u32),
        Op::TabStore { key, .. } => Some(key),
        _ => None,
    }
}

/// The versions at a point: per class, and per key within one.
#[derive(Clone, Default)]
struct Mem {
    class: [u32; CLASSES],
    keys: FastMap<(u8, u32), u32>,
}

impl Mem {
    fn version(&self, c: usize, key: Option<u32>) -> [u32; 2] {
        let k = key.and_then(|k| self.keys.get(&(c as u8, k)).copied());
        [self.class[c], k.unwrap_or(0)]
    }

    fn renew(&mut self, mask: u32, fresh: &mut u32) {
        for c in 0..CLASSES {
            if mask >> c & 1 != 0 {
                *fresh += 1;
                self.class[c] = *fresh;
                self.keys.retain(|&(x, _), _| x as usize != c);
            }
        }
    }

    fn renew_key(&mut self, c: usize, key: u32, fresh: &mut u32) {
        *fresh += 1;
        self.keys.insert((c as u8, key), *fresh);
    }
}

fn key_of(f: &Func<'_>, i: Inst, mem: &Mem) -> Option<Key> {
    let op = f.op(i);
    let a = f.args(i);
    if a.len() > 3 {
        return None;
    }
    let read = reads(op);
    let numberable =
        op.is_pure() && f.insts[i.idx()].rn == 1 || matches!(op, Op::Guard(_)) || read.is_some();
    if !numberable {
        return None;
    }
    let mut args = [Val(0); 3];
    args[..a.len()].copy_from_slice(a);
    Some(Key {
        op,
        n: a.len() as u8,
        args,
        mem: read.map_or([0, 0], |(c, k)| mem.version(c, k)),
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
    let mut wmask = vec![0u32; n];
    for &b in &cfg.rpo {
        for &i in f.insts_of(b) {
            wmask[b.idx()] |= writes(f.op(i));
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
    let mut end: Vec<Option<Mem>> = vec![None; n];
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
                let mut w = 0;
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
            None => Mem::default(),
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
            let w = writes(op);
            if w == 0 {
                continue;
            }
            match write_key(op) {
                Some(k) => mem.renew_key(w.trailing_zeros() as usize, k, &mut fresh),
                None => mem.renew(w, &mut fresh),
            }
            // A store answers the load of what it stored.
            if let Some((lop, largs, v)) = forwarded(f, i) {
                let (c, kk) = reads(lop).unwrap();
                let mut args = [Val(0); 3];
                args[..largs.len()].copy_from_slice(&largs);
                let key = Key {
                    op: lop,
                    n: largs.len() as u8,
                    args,
                    mem: mem.version(c, kk),
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
