//! Global value numbering (9.4) over the dominator tree: pure instructions,
//! type guards (a value's type never changes, so a dominating equal guard
//! makes a later one redundant), and loads and memory guards keyed by the
//! version of the memory they read (8.4); a call writes every class, and
//! scopes restart after one. Branch outcomes (W1 5.3): a `B1` is known in
//! the blocks an edge of a `br` on it dominates and after a guard on it, and
//! a `br`, guard or `select` on a known one folds.

use std::hash::BuildHasher;

use foldhash::fast::FixedState;
use hashbrown::hash_table::{Entry, HashTable};

use crate::jit::FastMap;
use crate::jit::ir::ops::{Flags, Mem, Op};
use crate::jit::ir::{Block, Func, Inst, Val};

const CLASSES: usize = 8;
const NONE: Val = Val(u32::MAX);
/// A known `B1`'s mark; its low bit is the outcome.
const KNOWN: u8 = 2;

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
        if mask.is_empty() {
            return;
        }
        for c in 0..CLASSES {
            if mask.bits() >> c & 1 != 0 {
                *fresh += 1;
                self.class[c] = *fresh;
                self.any[c] = *fresh;
            }
        }
        self.keys.retain(|&(x, _), _| mask.bits() >> x & 1 == 0);
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

/// `i`'s number, if it has one. Equal argument-free values (constants,
/// by-value upvalues) count as one argument, `consts` holding the first of
/// each and `canon` each value's once looked up: they are made again after
/// every call (`calls`), which keeps them apart as values.
fn key_of(
    f: &Func<'_>,
    i: Inst,
    mem: &Versions,
    consts: &mut FastMap<Op, Val>,
    canon: &mut [Val],
) -> Option<Key> {
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
        Op::Guard(_)
        | Op::GuardSame
        | Op::GuardProto(_)
        | Op::GuardTrue
        | Op::GuardFalse
        | Op::GuardMiss
        | Op::ToF64(_)
        | Op::ToPos(_)
        | Op::FToIExact
        | Op::FToLExact
        | Op::StrByteAt => true,
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
    for (k, &v) in a.iter().enumerate() {
        if canon[v.idx()] == NONE {
            canon[v.idx()] = match f.def_inst(v) {
                Some(d) if f.insts[d.idx()].an == 0 && f.op(d).is_pure() => {
                    *consts.entry(f.op(d)).or_insert(v)
                }
                _ => v,
            };
        }
        args[k] = canon[v.idx()];
    }
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

/// The load a store makes redundant (its operation and operands), and the
/// value it would read.
fn forwarded<'a>(f: &'a Func<'_>, i: Inst) -> Option<(Op, &'a [Val], Val)> {
    let a = f.args(i);
    match f.op(i) {
        Op::Store(r) => Some((Op::Load(r), &[], a[0])),
        Op::TabStore { loc, key } => Some((Op::TabLoad { loc, key }, &a[..1], a[1])),
        Op::ArrStore => Some((Op::ArrLoad, &a[..2], a[2])),
        Op::CellStore => Some((Op::CellLoad, &a[..1], a[1])),
        _ => None,
    }
}

/// A number's hash, key, and the number and scope it replaced.
type Undo = (u64, Key, Option<(Val, u32)>);

/// The value numbers in scope: entries with their hash and the scope that
/// made them, and an undo log restoring the table when a scope closes.
struct Numbers {
    table: HashTable<(u64, Key, Val, u32)>,
    undo: Vec<Undo>,
    marks: Vec<usize>,
    state: FixedState,
}

impl Numbers {
    /// `key`'s number made at or above scope `lo`, if any; else `key` is
    /// numbered `v` by scope `scope`.
    fn find_or_insert(&mut self, key: Key, lo: u32, any: bool, v: Val, scope: u32) -> Option<Val> {
        let h = self.state.hash_one(key);
        match self.table.entry(h, |e| e.1 == key, |e| e.0) {
            Entry::Occupied(mut o) => {
                let e = o.get_mut();
                if e.3 >= lo || any {
                    return Some(e.2);
                }
                self.undo.push((h, key, Some((e.2, e.3))));
                (e.2, e.3) = (v, scope);
            }
            Entry::Vacant(x) => {
                x.insert((h, key, v, scope));
                self.undo.push((h, key, None));
            }
        }
        None
    }

    /// Number `key` `v` by scope `scope`.
    fn set(&mut self, key: Key, v: Val, scope: u32) {
        let h = self.state.hash_one(key);
        match self.table.entry(h, |e| e.1 == key, |e| e.0) {
            Entry::Occupied(mut o) => {
                let e = o.get_mut();
                self.undo.push((h, key, Some((e.2, e.3))));
                (e.2, e.3) = (v, scope);
            }
            Entry::Vacant(x) => {
                x.insert((h, key, v, scope));
                self.undo.push((h, key, None));
            }
        }
    }

    fn close(&mut self) {
        let m = self.marks.pop().unwrap();
        while self.undo.len() > m {
            let (h, k, old) = self.undo.pop().unwrap();
            let Ok(e) = self.table.find_entry(h, |e| e.1 == k) else {
                unreachable!("an undone number not in the table")
            };
            match old {
                Some((v, s)) => {
                    let x = e.into_mut();
                    (x.2, x.3) = (v, s);
                }
                None => drop(e.remove()),
            }
        }
    }
}

pub(crate) fn gvn(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    let n = f.blocks.len();
    // The classes each block writes.
    let mut wmask = vec![Mem::empty(); n];
    let mut live = 0;
    for &b in &cfg.rpo {
        live += f.insts_of(b).len();
        for &i in f.insts_of(b) {
            wmask[b.idx()] |= f.op(i).effects().writes;
        }
    }
    let nv = f.vals.len();
    let mut map: Vec<Val> = (0..nv as u32).map(Val).collect();
    let mut removed = vec![false; f.insts.len()];
    let mut changed = false;
    let mut numbers = Numbers {
        table: HashTable::with_capacity(live),
        undo: Vec::with_capacity(live),
        marks: Vec::new(),
        state: FixedState::default(),
    };
    let mut consts: FastMap<Op, Val> = FastMap::default();
    let mut canon = vec![NONE; nv];
    // The versions at each block's end, for its dominator-tree children.
    let mut end: Vec<Option<Versions>> = vec![None; n];
    let mut fresh = 0u32;
    // Known `B1`s (`KNOWN | outcome`, 0 unknown), with their own undo log,
    // marked per scope.
    let mut known = vec![0u8; nv];
    let mut kundo: Vec<(Val, u8)> = Vec::new();
    let mut kmarks: Vec<usize> = Vec::new();
    let mut folded = false;
    // The walk that last reached each block, for the writes on paths from a
    // dominator.
    let mut seen = vec![0u32; n];
    let mut walk = 0u32;
    let mut work: Vec<Block> = Vec::new();
    // (block, done, lowest visible scope)
    let mut stack: Vec<(Block, bool, usize)> = vec![(f.entry, false, 0)];
    while let Some((b, done, barrier)) = stack.pop() {
        if done {
            numbers.close();
            let m = kmarks.pop().unwrap();
            while kundo.len() > m {
                let (c, old) = kundo.pop().unwrap();
                known[c.idx()] = old;
            }
            continue;
        }
        numbers.marks.push(numbers.undo.len());
        kmarks.push(kundo.len());
        // A fact crosses calls: folding on it adds no use.
        if let &[(t, k)] = cfg.incoming(b)
            && f.op(t) == Op::Br
        {
            let c = f.args(t)[0];
            kundo.push((
                c,
                std::mem::replace(&mut known[c.idx()], KNOWN | (k == 0) as u8),
            ));
        }
        let scope = numbers.marks.len() - 1;
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
                // The last child to start takes the dominator's versions.
                let mut m = if cfg.children(d).last() == Some(&b) {
                    end[d.idx()].take().unwrap_or_default()
                } else {
                    end[d.idx()].clone().unwrap_or_default()
                };
                if cfg.preds(b) != [d] {
                    walk += 1;
                    seen[d.idx()] = walk;
                    work.clear();
                    work.extend_from_slice(cfg.preds(b));
                    let mut w = Mem::empty();
                    while let Some(x) = work.pop() {
                        if std::mem::replace(&mut seen[x.idx()], walk) == walk {
                            continue;
                        }
                        w |= wmask[x.idx()];
                        work.extend_from_slice(cfg.preds(x));
                    }
                    m.renew(w, &mut fresh);
                }
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
            let op = f.op(i);
            if let Op::Br | Op::GuardTrue | Op::GuardFalse | Op::Select = op
                && let x = known[f.args(i)[0].idx()]
                && x != 0
            {
                let c = x == KNOWN | 1;
                match op {
                    Op::Br => {
                        let e = f.edges(i)[!c as usize];
                        f.insts[i.idx()].op = Op::Jump;
                        f.insts[i.idx()].an = 0;
                        f.set_edges(i, &[e]);
                        folded = true;
                        changed = true;
                        continue;
                    }
                    Op::Select => {
                        map[f.result(i).idx()] = f.args(i)[2 - c as usize];
                        removed[i.idx()] = true;
                        changed = true;
                        continue;
                    }
                    // One that always fails stays: its exit is the path.
                    _ if c == (op == Op::GuardTrue) => {
                        removed[i.idx()] = true;
                        changed = true;
                        continue;
                    }
                    _ => {}
                }
            }
            if matches!(op, Op::GuardTrue | Op::GuardFalse) {
                let c = f.args(i)[0];
                let x = KNOWN | (op == Op::GuardTrue) as u8;
                kundo.push((c, std::mem::replace(&mut known[c.idx()], x)));
            }
            if let Some(key) = key_of(f, i, &mem, &mut consts, &mut canon) {
                let rn = f.insts[i.idx()].rn;
                let r = if rn == 1 { f.result(i) } else { NONE };
                // A check without a result keeps nothing alive across a
                // call.
                if let Some(v) = numbers.find_or_insert(key, lo as u32, rn == 0, r, scope as u32) {
                    if rn == 1 {
                        map[f.result(i).idx()] = v;
                    }
                    removed[i.idx()] = true;
                    changed = true;
                    continue;
                }
            }
            if !op.effects().writes() {
                continue;
            }
            mem.write(op, &mut fresh);
            // A store answers the load of what it stored.
            if let Some((lop, largs, v)) = forwarded(f, i) {
                let mut args = [Val(0); 3];
                args[..largs.len()].copy_from_slice(largs);
                let key = Key {
                    op: lop,
                    n: largs.len() as u8,
                    args,
                    mem: mem.version(lop.effects().reads, lop.mem_key()),
                };
                numbers.set(key, v, scope as u32);
            }
        }
        end[b.idx()] = Some(mem);
        stack.push((b, true, lo));
        for &c in cfg.children(b).iter().rev() {
            stack.push((c, false, lo));
        }
    }
    if changed {
        for b in 0..f.blocks.len() {
            f.retain(Block(b as u32), |_, i| !removed[i.idx()]);
        }
        f.apply_replacements(&mut map);
    }
    if folded {
        crate::jit::opt::simplify::mark_unreachable(f);
    }
    changed
}
