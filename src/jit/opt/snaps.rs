//! Snapshot re-pruning (`jit-design.md` 9.8): an exit need not restore a
//! register whose slot holds the value its snapshot names, so a value only
//! exits use dies at its last other use. The slots' contents are followed
//! through the region: a load tells what a slot holds, a store sets it, and
//! a call clears what it keeps neither in its homes (`Func::call_homes`)
//! nor in a slot its snapshot names with that very value, which joins its
//! homes (the collector traces the slot in place while the call runs,
//! `ThreadState::trace`). The same walk finds the home loads
//! (`Func::home_loads`): a load whose slot holds its value wherever the
//! value is live, which the allocator may load from the slot again instead
//! of spilling.

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
/// a store puts), is a call, writes the captured ones (a cell store),
/// those from a slot up or any, or exits with a snapshot, read before the
/// instruction acts.
#[derive(Clone, Copy)]
enum Ev {
    Set(u16, Val),
    Call(Inst),
    Cells,
    From(u8),
    All,
    Exit(Inst),
}

/// The contents of the slots snapshots name and the region loads or
/// stores, through it.
struct Slots<'a, 'gc> {
    f: &'a Func<'gc>,
    cfg: &'a CfgInfo,
    /// The slots followed, and each slot's index among them.
    regs: Vec<u8>,
    idx: [u16; 256],
    /// Each block's events, a range of `evs`, and each event's instruction
    /// index in its block.
    evs: Vec<Ev>,
    pos: Vec<u32>,
    range: Vec<(u32, u32)>,
    /// The loads (the first 65535), each with its slot's index; each
    /// value's load or `NONE`, and each slot's loads.
    loads: Vec<(Val, u16)>,
    load_of: Vec<u16>,
    of_slot: Vec<Vec<u16>>,
    /// Registers a call's snapshot names with the slot's own value that it
    /// does not keep after all: another call of the snapshot does not.
    forbid: FastMap<u32, RegSet>,
    /// Each block's contents at its end, once reached.
    out: Vec<Val>,
    reached: Vec<bool>,
}

impl<'a, 'gc> Slots<'a, 'gc> {
    fn new(f: &'a Func<'gc>, cfg: &'a CfgInfo) -> Option<Self> {
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
        let (mut evs, mut pos, mut range) = (Vec::new(), Vec::new(), vec![(0, 0); f.blocks.len()]);
        let mut loads = Vec::new();
        let mut load_of = vec![NONE; f.vals.len()];
        let mut of_slot = vec![Vec::new(); regs.len()];
        for &b in &cfg.rpo {
            let start = evs.len() as u32;
            for (j, &i) in f.insts_of(b).iter().enumerate() {
                let op = f.op(i);
                match op {
                    Op::Call { .. } | Op::CallMm { .. } => evs.push(Ev::Call(i)),
                    _ if f.insts[i.idx()].snap != NO_SNAP => evs.push(Ev::Exit(i)),
                    _ => {}
                }
                match op {
                    Op::Load(r) => {
                        let k = idx[r as usize];
                        let v = f.result(i);
                        evs.push(Ev::Set(k, v));
                        if loads.len() < NONE as usize {
                            load_of[v.idx()] = loads.len() as u16;
                            of_slot[k as usize].push(loads.len() as u16);
                            loads.push((v, k));
                        }
                    }
                    Op::Store(r) if idx[r as usize] != NONE => {
                        evs.push(Ev::Set(idx[r as usize], canon(f, f.args(i)[0])));
                    }
                    Op::Store(_) | Op::Call { .. } | Op::CallMm { .. } => {}
                    // A cell is a slot of the frame only when a closure captured it.
                    Op::CellStore if cells => evs.push(Ev::Cells),
                    // Its control and loop variables (`jit_tfor_next`).
                    Op::Helper(HelperId::TforNext { a, .. }) => evs.push(Ev::From(a)),
                    op if op.effects().writes.contains(Mem::SLOT) => evs.push(Ev::All),
                    _ => {}
                }
                pos.resize(evs.len(), j as u32);
            }
            range[b.idx()] = (start, evs.len() as u32);
        }
        Some(Slots {
            f,
            cfg,
            out: vec![UNKNOWN; f.blocks.len() * regs.len()],
            reached: vec![false; f.blocks.len()],
            regs,
            idx,
            evs,
            pos,
            range,
            loads,
            load_of,
            of_slot,
            forbid: FastMap::default(),
        })
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
            Ev::From(lo) => {
                for (x, &r) in st.iter_mut().zip(&self.regs) {
                    if r >= lo {
                        *x = UNKNOWN;
                    }
                }
            }
            Ev::All => st.fill(UNKNOWN),
            Ev::Exit(_) => {}
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

/// Where the loads are live, for the home loads: those live into each
/// block (a bitset of `words`), and each one's last use in each block that
/// has any (`uses`, a range per block).
struct Live {
    words: usize,
    live_in: Vec<u64>,
    uses: Vec<(u32, u32)>,
    range: Vec<(u32, u32)>,
}

impl Live {
    fn new(f: &Func<'_>, cfg: &CfgInfo, load_of: &[u16], loads: usize) -> Live {
        let (words, nb) = (loads.div_ceil(64), f.blocks.len());
        let (mut live_in, mut kill) = (vec![0u64; nb * words], vec![0u64; nb * words]);
        let (mut uses, mut range) = (Vec::new(), vec![(0, 0); nb]);
        // Each load's entry of `uses` in the block walked.
        const NO: u32 = u32::MAX;
        let mut at = vec![NO; loads];
        for &b in &cfg.rpo {
            let (start, w) = (uses.len(), b.idx() * words);
            for (j, &i) in f.insts_of(b).iter().enumerate() {
                f.for_each_use(i, |v| match load_of[v.idx()] {
                    NONE => {}
                    l if at[l as usize] == NO => {
                        at[l as usize] = uses.len() as u32;
                        uses.push((l as u32, j as u32));
                    }
                    l => uses[at[l as usize] as usize].1 = j as u32,
                });
                if let Op::Load(_) = f.op(i) {
                    let l = load_of[f.result(i).idx()];
                    if l != NONE {
                        kill[w + l as usize / 64] |= 1 << (l % 64);
                    }
                }
            }
            // Used here and defined elsewhere: live in.
            for &(l, _) in &uses[start..] {
                let l = l as usize;
                at[l] = NO;
                live_in[w + l / 64] |= !kill[w + l / 64] & 1 << (l % 64);
            }
            range[b.idx()] = (start as u32, uses.len() as u32);
        }
        loop {
            let mut changed = false;
            for &b in cfg.rpo.iter().rev() {
                let w = b.idx() * words;
                for s in f.succs(b) {
                    for k in 0..words {
                        let x = live_in[s.idx() * words + k] & !kill[w + k];
                        changed |= x & !live_in[w + k] != 0;
                        live_in[w + k] |= x;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Live {
            words,
            live_in,
            uses,
            range,
        }
    }

    fn live_in(&self, b: Block) -> &[u64] {
        &self.live_in[b.idx() * self.words..][..self.words]
    }
}

pub(crate) fn prune_snapshots(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let Some(mut slots) = Slots::new(f, &cfg) else {
        f.home_loads.clear();
        return;
    };
    loop {
        slots.flow();
        if !slots.settle() {
            break;
        }
    }
    let live = Live::new(f, &cfg, &slots.load_of, slots.loads.len());
    // Each exit's entries whose slot holds the value, what each call keeps
    // in place, and the loads whose slot does not hold them somewhere they
    // are live, or that a call still names (for another slot, or for their
    // own without keeping it): those stay ordinary values. An exit naming
    // one for another slot copies it from its own (`Loc::Slot`).
    let mut pruned: Vec<(Inst, u32, u32)> = Vec::new();
    let mut bits: Vec<bool> = Vec::new();
    let mut homes: Vec<(u32, RegSet)> = Vec::new();
    let mut dirty = vec![false; slots.loads.len()];
    // In the block walked: each load's last use (its instruction's index
    // and one, or none), and those live out.
    let mut last = vec![0u32; slots.loads.len()];
    let mut out = vec![0u64; live.words];
    let n = slots.regs.len();
    let mut st = vec![UNKNOWN; n];
    for &b in &cfg.rpo {
        slots.meet(b, &mut st);
        for (w, &x) in live.live_in(b).iter().enumerate() {
            let mut x = x;
            while x != 0 {
                let l = w * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                let (v, k) = slots.loads[l];
                dirty[l] |= st[k as usize] != v;
            }
        }
        out.fill(0);
        for s in f.succs(b) {
            for (o, &x) in out.iter_mut().zip(live.live_in(s)) {
                *o |= x;
            }
        }
        let (ulo, uhi) = live.range[b.idx()];
        for &(l, j) in &live.uses[ulo as usize..uhi as usize] {
            last[l as usize] = j + 1;
        }
        let (lo, hi) = slots.range[b.idx()];
        // The exiting instruction last seen: its snapshot is read after
        // what it writes itself.
        let mut exit = None;
        for e in lo as usize..hi as usize {
            let (ev, j) = (slots.evs[e], slots.pos[e]);
            let mut kept = RegSet::EMPTY;
            // The slots written: one, or those left unknown.
            let one = match ev {
                Ev::Exit(i) => {
                    let from = bits.len();
                    for &(r, v) in f.snap_entries(i) {
                        let k = slots.at(r);
                        bits.push(k != NONE && st[k as usize] == canon(f, v));
                    }
                    if bits[from..].contains(&true) {
                        pruned.push((i, from as u32, bits.len() as u32));
                    } else {
                        bits.truncate(from);
                    }
                    exit = Some((i, j));
                    continue;
                }
                Ev::Call(i) => {
                    slots.step(&mut st, ev, Some(&mut kept));
                    let s = f.insts[i.idx()].snap;
                    if kept != RegSet::EMPTY {
                        homes.push((s, kept));
                    }
                    if let Some(h) = f.call_homes.get(&s) {
                        kept.union(h);
                    }
                    // Named for its own slot, which the call keeps: a home.
                    for &(r, v) in f.entries(s) {
                        let l = slots.load_of[canon(f, v).idx()];
                        if l != NONE {
                            let k = slots.loads[l as usize].1;
                            dirty[l as usize] |= r == KEPT
                                || slots.regs[k as usize] != r
                                || !kept.contains(r as usize);
                        }
                    }
                    None
                }
                Ev::Set(k, _) => {
                    slots.step(&mut st, ev, None);
                    Some(k as usize)
                }
                Ev::Cells | Ev::From(_) | Ev::All => {
                    slots.step(&mut st, ev, None);
                    None
                }
            };
            let named = |v: Val| {
                exit.is_some_and(|(i, at)| {
                    at == j && f.snap_entries(i).iter().any(|&(_, e)| canon(f, e) == v)
                })
            };
            let written = match one {
                Some(k) => k..k + 1,
                None => 0..n,
            };
            for k in written {
                if one.is_none() && st[k] != UNKNOWN {
                    continue;
                }
                for &l in &slots.of_slot[k] {
                    let (l, v) = (l as usize, slots.loads[l as usize].0);
                    if st[k] != v
                        && !dirty[l]
                        && (last[l] > j + 1 || out[l / 64] >> (l % 64) & 1 != 0 || named(v))
                    {
                        dirty[l] = true;
                    }
                }
            }
        }
        for &(l, _) in &live.uses[ulo as usize..uhi as usize] {
            last[l as usize] = 0;
        }
    }
    let mut home = vec![false; f.vals.len()];
    for (&(v, _), &d) in slots.loads.iter().zip(&dirty) {
        home[v.idx()] = !d;
    }
    drop(slots);
    f.home_loads = home;
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
