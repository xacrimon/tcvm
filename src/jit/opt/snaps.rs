//! Snapshot re-pruning (`jit-design.md` 9.8): an exit need not restore a
//! register whose slot holds the value its snapshot names, so a value only
//! exits use dies at its last other use. The slots' contents are followed
//! through the region: a load tells what a slot holds, a store sets it, and
//! a call clears what it keeps neither in its homes (`Func::call_homes`)
//! nor in a slot its snapshot names with that very value, which joins its
//! homes (the collector traces the slot in place while the call runs,
//! `ThreadState::trace`). The same walk finds the home loads (`Op::Home`):
//! a load whose slot holds its value wherever its register is read, which
//! the allocator may load from the slot again instead of spilling.

use crate::jit::FastMap;
use crate::jit::build::cfg::RegSet;
use crate::jit::ir::ops::{HelperId, Mem, Op};
use crate::jit::ir::{Block, CfgInfo, Func, Inst, KEPT, List, NO_SNAP, SnapData, Val};

/// A slot's contents when nothing is known.
const UNKNOWN: Val = Val(u32::MAX);
const NONE: u16 = u16::MAX;

/// `v` without its boxing, guards and refinements: the slot value it is.
fn canon(f: &Func<'_>, mut v: Val) -> Val {
    while let Some(i) = f.def_inst(v)
        && matches!(
            f.op(i),
            Op::Box | Op::Unbox(_) | Op::Guard(_) | Op::Refine(_)
        )
    {
        v = f.args(i)[0];
    }
    v
}

/// What an instruction does to the slots followed: sets one (a load tells,
/// a store puts), is a call, writes the captured ones (a cell store) or
/// those from a slot up, or exits with a snapshot, read before the
/// instruction acts. And where a load's register is read.
#[derive(Clone, Copy)]
enum Ev {
    Set(u16, Val),
    Call(Inst),
    Cells,
    From(u8),
    Exit(Inst),
    Read(Val),
}

/// The loads an instruction using `v` reads: `v`'s own (`Slots::load_of`),
/// or those of the operands of what lowering computes at its use.
fn reads(f: &Func<'_>, load_of: &[Val], v: Val, evs: &mut Vec<Ev>) {
    let l = load_of[v.idx()];
    if l != UNKNOWN {
        evs.push(Ev::Read(l));
    } else if let Some(d) = f.def_inst(v)
        && f.op(d).folds()
    {
        for &a in f.args(d) {
            reads(f, load_of, a, evs);
        }
    }
}

/// The contents of the slots snapshots name and the region loads or
/// stores, through it.
struct Slots<'a, 'gc> {
    f: &'a Func<'gc>,
    cfg: &'a CfgInfo,
    /// The slots followed, and each slot's index among them.
    regs: Vec<u8>,
    idx: [u16; 256],
    /// Each block's events, a range of `evs`.
    evs: Vec<Ev>,
    range: Vec<(u32, u32)>,
    /// Registers a call's snapshot names with the slot's own value that it
    /// does not keep after all: another call of the snapshot does not.
    forbid: FastMap<u32, RegSet>,
    /// Each block's contents at its end, once reached.
    out: Vec<Val>,
    reached: Vec<bool>,
}

impl<'a, 'gc> Slots<'a, 'gc> {
    /// And the load each value is, unchanged (`canon`, or narrowed):
    /// lowering has those share its register or reads it for them.
    fn new(f: &'a Func<'gc>, cfg: &'a CfgInfo) -> Option<(Self, Vec<Val>)> {
        let (mut named, mut touched, mut loaded) = (RegSet::EMPTY, RegSet::EMPTY, RegSet::EMPTY);
        for &b in &cfg.rpo {
            for &i in f.insts_of(b) {
                match f.op(i) {
                    Op::Load(r) => {
                        touched.insert(r as usize);
                        loaded.insert(r as usize);
                    }
                    Op::Store(r) => touched.insert(r as usize),
                    _ => {}
                }
                for &(r, _) in f.snap_entries(i) {
                    if r != KEPT {
                        named.insert(r as usize);
                    }
                }
            }
        }
        for ((n, t), l) in named.0.iter_mut().zip(&touched.0).zip(&loaded.0) {
            *n = *n & t | l;
        }
        let regs: Vec<u8> = named.iter().map(|r| r as u8).collect();
        if regs.is_empty() {
            return None;
        }
        let mut idx = [NONE; 256];
        for (k, &r) in regs.iter().enumerate() {
            idx[r as usize] = k as u16;
        }
        let cells = regs.iter().any(|&r| f.captured.contains(r as usize));
        let (mut evs, mut range) = (Vec::new(), vec![(0, 0); f.blocks.len()]);
        let mut load_of = vec![UNKNOWN; f.vals.len()];
        for &b in &cfg.rpo {
            let start = evs.len() as u32;
            for &i in f.insts_of(b) {
                let op = f.op(i);
                if let Op::Box | Op::Unbox(_) | Op::Guard(_) | Op::Refine(_) | Op::LToI = op {
                    load_of[f.result(i).idx()] = load_of[f.args(i)[0].idx()];
                }
                f.for_each_use(i, |v| reads(f, &load_of, v, &mut evs));
                match op {
                    Op::Call { .. } | Op::CallMm { .. } => evs.push(Ev::Call(i)),
                    _ if f.insts[i.idx()].snap != NO_SNAP => evs.push(Ev::Exit(i)),
                    _ => {}
                }
                let written = evs.len();
                match op {
                    Op::Load(r) => {
                        let v = f.result(i);
                        evs.push(Ev::Set(idx[r as usize], v));
                        load_of[v.idx()] = v;
                    }
                    Op::Store(r) if idx[r as usize] != NONE => {
                        evs.push(Ev::Set(idx[r as usize], canon(f, f.args(i)[0])));
                    }
                    Op::Store(_) | Op::Call { .. } | Op::CallMm { .. } => {}
                    // A cell is a slot of the frame only when a closure captured it.
                    Op::CellStore if cells => evs.push(Ev::Cells),
                    // Its control and loop variables (`jit_tfor_next`).
                    Op::Helper(HelperId::TforNext { a, .. }) => evs.push(Ev::From(a)),
                    op if op.effects().writes.contains(Mem::SLOT) => evs.push(Ev::From(0)),
                    _ => {}
                }
                // An exit reads its snapshot after what it wrote itself.
                if evs.len() > written {
                    for &(_, v) in f.snap_entries(i).iter().chain(f.blame_entries(i)) {
                        reads(f, &load_of, v, &mut evs);
                    }
                }
            }
            range[b.idx()] = (start, evs.len() as u32);
        }
        let slots = Slots {
            f,
            cfg,
            out: vec![UNKNOWN; f.blocks.len() * regs.len()],
            reached: vec![false; f.blocks.len()],
            regs,
            idx,
            evs,
            range,
            forbid: FastMap::default(),
        };
        Some((slots, load_of))
    }

    fn events(&self, b: Block) -> &[Ev] {
        let (lo, hi) = self.range[b.idx()];
        &self.evs[lo as usize..hi as usize]
    }

    /// The index of slot `r` among those followed, or `NONE`.
    #[inline]
    fn at(&self, r: u8) -> u16 {
        self.idx[r as usize]
    }

    /// The contents at `b`'s start, from the predecessors reached so far.
    fn meet(&self, b: Block, st: &mut [Val]) {
        let n = self.regs.len();
        let mut any = false;
        for &p in self.cfg.preds(b) {
            if b == self.f.entry || !self.reached[p.idx()] {
                continue;
            }
            let o = &self.out[p.idx() * n..][..n];
            if !std::mem::replace(&mut any, true) {
                st.copy_from_slice(o);
            } else {
                for (x, &y) in st.iter_mut().zip(o) {
                    if *x != y {
                        *x = UNKNOWN;
                    }
                }
            }
        }
        if !any {
            st.fill(UNKNOWN);
        }
    }

    /// `st` after `ev`; at a call, the registers it newly keeps in place go
    /// to `kept`.
    fn step(&self, st: &mut [Val], ev: Ev, mut kept: Option<&mut RegSet>) {
        let f = self.f;
        match ev {
            Ev::Set(k, v) => st[k as usize] = v,
            Ev::Call(i) => {
                let (Op::Call { a: hdr, .. } | Op::CallMm { hdr, .. }) = f.op(i) else {
                    unreachable!()
                };
                let s = f.insts[i.idx()].snap;
                let homes = f.call_homes.get(&s);
                let forbid = self.forbid.get(&s);
                for (x, &r) in st.iter_mut().zip(&self.regs) {
                    let r = r as usize;
                    if r >= hdr as usize || f.captured.contains(r) {
                        *x = UNKNOWN;
                    } else if !homes.is_some_and(|h| h.contains(r)) {
                        let holds = !forbid.is_some_and(|h| h.contains(r))
                            && (f.entries(s).iter())
                                .any(|&(er, v)| er as usize == r && canon(f, v) == *x);
                        match kept.as_deref_mut() {
                            Some(kept) if holds => kept.insert(r),
                            _ if holds => {}
                            _ => *x = UNKNOWN,
                        }
                    }
                }
            }
            Ev::Cells => {
                for (x, &r) in st.iter_mut().zip(&self.regs) {
                    if f.captured.contains(r as usize) {
                        *x = UNKNOWN;
                    }
                }
            }
            // `regs` ascends.
            Ev::From(lo) => st[self.regs.partition_point(|&r| r < lo)..].fill(UNKNOWN),
            Ev::Exit(_) | Ev::Read(_) => {}
        }
    }

    /// The contents at every block's end: a fixpoint, each slot's contents
    /// only ever becoming unknown.
    fn flow(&mut self) {
        let n = self.regs.len();
        let mut st = vec![UNKNOWN; n];
        self.reached.fill(false);
        loop {
            let mut changed = false;
            for &b in &self.cfg.rpo {
                self.meet(b, &mut st);
                for &ev in self.events(b) {
                    self.step(&mut st, ev, None);
                }
                let o = &mut self.out[b.idx() * n..][..n];
                if !self.reached[b.idx()] || o != &st[..] {
                    o.copy_from_slice(&st);
                    self.reached[b.idx()] = true;
                    changed = true;
                }
            }
            if !changed {
                return;
            }
        }
    }

    /// What each call keeps in place; a snapshot several calls share keeps
    /// what all of them do, the rest forbidden. Whether anything was.
    fn settle(&mut self) -> bool {
        let n = self.regs.len();
        let mut st = vec![UNKNOWN; n];
        let mut by_snap: FastMap<u32, (RegSet, RegSet)> = FastMap::default();
        for &b in &self.cfg.rpo {
            self.meet(b, &mut st);
            for &ev in self.events(b) {
                let mut kept = RegSet::EMPTY;
                self.step(&mut st, ev, Some(&mut kept));
                if let Ev::Call(i) = ev {
                    let s = self.f.insts[i.idx()].snap;
                    let e = by_snap.entry(s).or_insert((kept, kept));
                    for w in 0..4 {
                        e.0.0[w] &= kept.0[w];
                        e.1.0[w] |= kept.0[w];
                    }
                }
            }
        }
        let mut more = false;
        for (s, (all, mut any)) in by_snap {
            any.minus(&all);
            if any != RegSet::EMPTY {
                self.forbid.entry(s).or_default().union(&any);
                more = true;
            }
        }
        more
    }
}

pub(crate) fn prune_snapshots(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let Some((mut slots, mut load_of)) = Slots::new(f, &cfg) else {
        return;
    };
    loop {
        slots.flow();
        if !slots.settle() {
            break;
        }
    }
    // Each exit's entries whose slot holds the value, what each call keeps
    // in place, and the loads read where their slot does not hold them (or
    // that a call names other than for their own slot, kept): those stay
    // ordinary values. An exit naming a home load for another slot copies
    // it from its own (`Loc::Slot`).
    let mut pruned: Vec<(Inst, u32, u32)> = Vec::new();
    let mut bits: Vec<bool> = Vec::new();
    let mut homes: Vec<(u32, RegSet)> = Vec::new();
    let mut st = vec![UNKNOWN; slots.regs.len()];
    for &b in &cfg.rpo {
        slots.meet(b, &mut st);
        for &ev in slots.events(b) {
            let mut kept = RegSet::EMPTY;
            match ev {
                Ev::Read(v) => {
                    if let Some(Op::Load(r)) = f.def_op(v)
                        && st[slots.at(r) as usize] != v
                    {
                        load_of[v.idx()] = UNKNOWN;
                    }
                }
                Ev::Exit(i) => {
                    let from = bits.len();
                    bits.extend((f.snap_entries(i).iter()).map(|&(r, v)| {
                        let k = slots.at(r);
                        k != NONE && st[k as usize] == canon(f, v)
                    }));
                    if bits[from..].contains(&true) {
                        pruned.push((i, from as u32, bits.len() as u32));
                    } else {
                        bits.truncate(from);
                    }
                }
                Ev::Call(i) => {
                    slots.step(&mut st, ev, Some(&mut kept));
                    let s = f.insts[i.idx()].snap;
                    if kept != RegSet::EMPTY {
                        homes.push((s, kept));
                    }
                    for &(r, e) in f.entries(s) {
                        let v = load_of[e.idx()];
                        if v != UNKNOWN
                            && (r == KEPT
                                || f.def_op(v) != Some(Op::Load(r))
                                || st[slots.at(r) as usize] != v)
                        {
                            load_of[v.idx()] = UNKNOWN;
                        }
                    }
                }
                ev => slots.step(&mut st, ev, None),
            }
        }
    }
    drop(slots);
    for (v, &l) in load_of.iter().enumerate() {
        if l == Val(v as u32)
            && let i = f.def_inst(l).unwrap()
            && let Op::Load(r) = f.op(i)
        {
            f.insts[i.idx()].op = Op::Home(r);
        }
    }
    for (s, kept) in homes {
        f.call_homes.entry(s).or_default().union(&kept);
    }
    // The same snapshot less the same entries is made once.
    let mut done: FastMap<(u32, Vec<bool>), u32> = FastMap::default();
    let mut key = (0, Vec::new());
    for (i, from, to) in pruned {
        let s = f.insts[i.idx()].snap;
        key.0 = s;
        key.1.clear();
        key.1.extend_from_slice(&bits[from as usize..to as usize]);
        let ns = match done.get(&key) {
            Some(&ns) => ns,
            None => {
                let d = f.snaps[s as usize];
                let start = f.snap_pool.len() as u32;
                for (k, &gone) in key.1.iter().enumerate() {
                    if !gone {
                        let e = f.snap_pool[d.entries.start as usize + k];
                        f.snap_pool.push(e);
                    }
                }
                let len = f.snap_pool.len() as u32 - start;
                f.snaps.push(SnapData {
                    entries: List { start, len },
                    ..d
                });
                done.insert(key.clone(), f.snaps.len() as u32 - 1);
                f.snaps.len() as u32 - 1
            }
        };
        f.insts[i.idx()].snap = ns;
    }
}
