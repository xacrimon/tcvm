//! After the scan: each operand's allocation and the moves between pieces,
//! across edges, into fixed registers, and the stores after definitions,
//! each gap's moves sequenced as one parallel move.

use super::build::{merge_tail, push_coalesced};
use super::{Edit, Ivs, Loc, NONE, Need, Output, Pos, Ra, Want, intersect};
use crate::jit::backend::reg::{Allocation, Inst, ProgPoint, RegClass, SpillSlot, VReg};
use crate::jit::backend::vcode::{Remat, TargetInst, VCode};

/// A move of a value of class `cls`, or with `remat` its computation
/// again from `from` (`Allocation::none()` for a constant).
#[derive(Clone, Copy, Debug)]
struct Mv {
    from: Allocation,
    to: Allocation,
    cls: u8,
    vreg: u32,
    remat: bool,
}

/// The order of the moves sharing a gap: stores after the definition
/// before it, then moves between pieces, across the edge into the block,
/// stores of its parameters, across the edges out of it (a block of one
/// instruction has both), cold stores of a piece ending at the
/// instruction's late point (from where the moves before left it), and
/// copies into the instruction's fixed registers. Moves on an edge of a
/// branch into a join go on that edge (`Output::edge_edits`), the edge's
/// index above the phase.
const DEF_STORE: u8 = 0;
const CONNECT: u8 = 1;
const EDGE_IN: u8 = 2;
const PARAM_STORE: u8 = 3;
const EDGE_OUT: u8 = 4;
const LATE_STORE: u8 = 5;
const FIXED_COPY: u8 = 6;
const EDGE: u8 = 7;

/// `resolve`'s buffers: the values of split groups; the moves as
/// `(linear index, phase, move)`, then sorted, and one gap's; slot
/// coloring's groups with their run of `all` (`busy`), a group's padded
/// intervals and each slot's; `edge_moves`'s constants made once, piece
/// cursors and live values.
#[derive(Default)]
pub(super) struct Resolve {
    split: Vec<u64>,
    moves: Vec<(u32, u8, Mv)>,
    sorted: Vec<(u32, u8, Mv)>,
    keys: Vec<u64>,
    group: Vec<Mv>,
    busy: Vec<(u32, u32, u32)>,
    all: Vec<(Pos, Pos)>,
    ivs: Vec<(Pos, Pos)>,
    slots: Vec<Vec<(Pos, Pos)>>,
    once: Vec<u32>,
    cur: Vec<u32>,
    live: Vec<(u32, u32)>,
}

impl Resolve {
    pub(super) fn clear(&mut self) {
        clear!(self; split, moves, sorted, keys, group, busy, all, ivs, once, cur, live);
        for s in &mut self.slots {
            s.clear();
        }
    }
}

impl Ra {
    pub(super) fn resolve<I: TargetInst>(
        &mut self,
        f: &VCode<I>,
        t: &mut Resolve,
    ) -> Result<Output, String> {
        self.index_pieces(&mut t.split);
        self.slot_derived(f);
        self.color_slots(t);
        self.snapshot_allocs();
        // A group is stored where it goes from a register to the stack when
        // that runs no more often than storing after its definitions (V8's
        // `DecideSpillingMode`; cold-only spills always), its slot uses
        // falling in stack pieces; else once after each definition, its slot
        // holding it from there on.
        for g in 0..self.vregs.len() {
            let d = self.stores_at_transitions(g);
            self.deferred.push(d);
        }
        for x in 0..self.pieces.len() as u32 {
            let p = self.pieces[x as usize];
            if p.dead() {
                continue;
            }
            if p.loc == Loc::None {
                return Err(format!("v{} left unallocated", p.g));
            }
            for k in p.ulo..p.uhi {
                let u = self.uses[k as usize];
                let a = match u.need {
                    Need::Reg => match p.loc {
                        Loc::Reg(r) => Allocation::reg(r),
                        _ => return Err(format!("v{} needs a register at {}", p.g, u.pos)),
                    },
                    Need::Any => self.alloc_of(x),
                    Need::Fixed(r) => Allocation::reg(r),
                };
                self.allocs[u.alloc as usize] = a;
            }
        }
        let moves = &mut t.moves;
        for v in 0..self.vregs.len() as u32 {
            let gr = self.rep[v as usize];
            let g = &self.vregs[gr as usize];
            if g.slot == NONE || g.head == NONE || self.deferred[gr as usize] {
                continue;
            }
            let (cls, slot) = (g.class, self.slot(gr));
            let vi = &self.vregs[v as usize];
            if vi.def != NONE {
                let l = vi.def / 2;
                if self.lin_block[l as usize + 1] != self.lin_block[l as usize] {
                    return Err(format!("v{v} defined by a block's last instruction"));
                }
                let from = self.alloc_at(v, vi.def).expect("the definition's piece");
                debug_assert!(from.is_reg());
                moves.push((l + 1, DEF_STORE, mv(from, slot, cls, v, false)));
            } else if vi.param_of != NONE && !vi.args_in_group {
                let b = vi.param_of;
                if let Some(from) = self.alloc_at(v, self.block_start(b))
                    && from.is_reg()
                {
                    let at = self.first[b as usize];
                    moves.push((at, PARAM_STORE, mv(from, slot, cls, v, false)));
                }
            }
        }
        for g in 0..self.vregs.len() as u32 {
            let gi = &self.vregs[g as usize];
            if self.rep[g as usize] != g || gi.head == NONE {
                continue;
            }
            let (cls, deferred) = (gi.class, self.deferred[g as usize]);
            let ps = &self.flat[gi.flat.0 as usize..gi.flat.1 as usize];
            // Between pieces inside a block; a stack piece needs a store only
            // in cold code, before the instruction it starts at.
            for w in ps.windows(2) {
                let (a, b) = (&self.pieces[w[0] as usize], &self.pieces[w[1] as usize]);
                let at = b.start;
                if a.end != at
                    || a.loc == b.loc
                    || self.is_def(g, at)
                    || b.loc == Loc::Remat
                    || b.loc == Loc::Stack && !deferred
                {
                    continue;
                }
                if at == self.block_start(self.block_of_pos(at)) {
                    continue;
                }
                if at & 1 != 0 && b.loc != Loc::Stack {
                    return Err(format!("v{g} reloaded inside instruction {}", at / 2));
                }
                let phase = if at & 1 != 0 { LATE_STORE } else { CONNECT };
                moves.push((
                    at / 2,
                    phase,
                    self.mv(f, g, at - 1, self.alloc_of(w[1]), cls),
                ));
            }
            // Into fixed registers, from wherever the value is when the
            // instruction starts.
            for &x in ps {
                let p = &self.pieces[x as usize];
                let mut k = p.ulo;
                while let Some(j) = self.want_from(k, p.uhi, Want::Fixed) {
                    let u = self.uses[j as usize];
                    let Need::Fixed(r) = u.need else {
                        unreachable!("a fixed use")
                    };
                    let n = u.pos / 2;
                    let m = self.mv(f, g, 2 * n, Allocation::reg(r), cls);
                    if m.remat || m.from != m.to {
                        moves.push((n, FIXED_COPY, m));
                    }
                    k = j + 1;
                }
            }
        }
        self.edge_moves(f, t)?;
        // By instruction and phase, in the order pushed.
        let Resolve {
            moves,
            sorted,
            keys,
            group,
            ..
        } = t;
        keys.clear(); // `color_slots` sorted with them too.
        keys.extend((moves.iter().enumerate()).map(|(k, &(l, phase, _))| {
            ((self.at_lin[l as usize].index() as u64) << 8 | phase as u64) << 32 | k as u64
        }));
        keys.sort_unstable();
        sorted.extend(keys.iter().map(|&k| moves[k as u32 as usize]));
        let moves = sorted;
        let mut edits: Vec<(ProgPoint, Edit)> = Vec::with_capacity(moves.len());
        let mut edge_edits: Vec<(Inst, u8, Edit)> = Vec::new();
        let mut k = 0;
        while k < moves.len() {
            let (l, phase) = (moves[k].0, moves[k].1);
            group.clear();
            while k < moves.len() && moves[k].0 == l && moves[k].1 == phase {
                group.push(moves[k].2);
                k += 1;
            }
            let i = self.at_lin[l as usize];
            if phase & 7 == EDGE {
                self.sequence(group, |e| edge_edits.push((i, phase >> 3, e)))?;
            } else {
                let at = ProgPoint::before(i);
                self.sequence(group, |e| edits.push((at, e)))?;
            }
        }
        Ok(Output {
            num_spillslots: self.num_slots as usize,
            edits,
            edge_edits,
            allocs: std::mem::take(&mut self.allocs),
        })
    }

    /// Each group's pieces by start as a run of `flat`, for lookups; the
    /// values whose group's pieces are not all in one place, as a set over
    /// vregs (only those can be in two places at an edge's ends).
    fn index_pieces(&mut self, split: &mut Vec<u64>) {
        split.resize(self.words, 0);
        for g in 0..self.vregs.len() {
            let s = self.flat.len() as u32;
            let mut x = self.vregs[g].head;
            while x != NONE {
                self.flat.push(x);
                self.flat_start.push(self.start(x));
                x = self.pieces[x as usize].next;
            }
            let e = self.flat.len() as u32;
            self.vregs[g].flat = (s, e);
            let ps = &self.flat[s as usize..e as usize];
            let loc = |x: &u32| self.pieces[*x as usize].loc;
            if ps.iter().any(|x| loc(x) != loc(&ps[0])) {
                let (a, b) = self.vregs[g].members;
                for &m in &self.members[a as usize..b as usize] {
                    split[m as usize / 64] |= 1 << (m % 64);
                }
            }
        }
    }

    fn slot(&self, g: u32) -> Allocation {
        Allocation::stack(SpillSlot::new(self.vregs[g as usize].slot as usize))
    }

    fn pieces_of(&self, g: u32) -> &[u32] {
        let (a, b) = self.vregs[g as usize].flat;
        &self.flat[a as usize..b as usize]
    }

    fn stores_at_transitions(&self, g: usize) -> bool {
        let gi = &self.vregs[g];
        if self.rep[g] != g as u32 || gi.slot == NONE {
            return false;
        }
        let mut at_transitions = 0f32;
        for &x in self.pieces_of(g as u32) {
            let p = &self.pieces[x as usize];
            if p.loc == Loc::Stack {
                at_transitions += self.freq_at(p.start);
            } else if p.stack {
                return false;
            }
        }
        let mut at_defs = 0f32;
        for &v in &self.members[gi.members.0 as usize..gi.members.1 as usize] {
            let vi = &self.vregs[v as usize];
            if vi.def != NONE {
                at_defs += self.freq_at(vi.def);
            } else if vi.param_of != NONE && !vi.args_in_group {
                at_defs += self.freq_at(self.block_start(vi.param_of));
            }
        }
        at_transitions <= at_defs
    }

    /// Each snapshot operand's allocation: its group's slot, or where the
    /// piece at its point is. Pieces come by start, the operands by point.
    fn snapshot_allocs(&mut self) {
        for v in 0..self.vregs.len() {
            let (a, b) = (self.snap_off[v] as usize, self.snap_off[v + 1] as usize);
            if a == b {
                continue;
            }
            let g = self.rep[v];
            let (fa, fb) = self.vregs[g as usize].flat;
            let mut c = fa as usize;
            for k in a..b {
                let (pos, op) = self.snaps[k];
                while c + 1 < fb as usize && self.flat_start[c + 1] <= pos {
                    c += 1;
                }
                let x = self.flat[c];
                debug_assert!(self.ivs_of(x).covers(pos));
                self.allocs[(op >> 1) as usize] = if op & 1 != 0 {
                    self.pieces[x as usize].stack = true;
                    self.slot(g)
                } else {
                    self.alloc_of(x)
                };
            }
        }
    }

    /// A value computed from one on the stack costs a load and the
    /// computation where reloading it costs the load alone: one spilled
    /// where its source is on the stack gets a slot, stored after its
    /// definition, when that is in a register.
    fn slot_derived<I: TargetInst>(&mut self, f: &VCode<I>) {
        for v in 0..self.vregs.len() {
            let Remat::AndImm { src, .. } = f.remat[v] else {
                continue;
            };
            let vi = &self.vregs[v];
            if !vi.remat || vi.head == NONE {
                continue;
            }
            let def_in_reg = self.alloc_at(v as u32, vi.def).is_some_and(|a| a.is_reg());
            let src_on_stack = self.pieces_of(v as u32).iter().any(|&x| {
                let iv = self.ivs_of(x);
                self.pieces[x as usize].loc == Loc::Remat
                    && (0..iv.len()).any(|k| {
                        self.alloc_at(src, iv.get(k).0)
                            .is_some_and(|l| l.is_stack())
                    })
            });
            if def_in_reg && src_on_stack {
                let (a, b) = self.vregs[v].flat;
                for k in a..b {
                    let x = self.flat[k as usize] as usize;
                    if self.pieces[x].loc == Loc::Remat {
                        self.pieces[x].loc = Loc::Stack;
                    }
                }
                self.vregs[v].remat = false;
                self.slot_of(v as u32);
            }
        }
    }

    /// Give spilled groups slots by coloring (JSC's stack coloring): a slot
    /// is busy over its group's whole range (stores in cold mode may sit at
    /// a hot predecessor's end); groups take the first slot free over
    /// theirs, in start order.
    fn color_slots(&mut self, t: &mut Resolve) {
        let Resolve {
            busy,
            all,
            ivs,
            keys,
            slots,
            ..
        } = t;
        for g in 0..self.vregs.len() {
            if self.rep[g] != g as u32 || self.vregs[g].slot == NONE {
                continue;
            }
            ivs.clear();
            for &x in self.pieces_of(g as u32) {
                // Stores land in the gaps around: a point either side.
                let iv = self.ivs_of(x);
                ivs.extend(
                    (0..iv.len())
                        .map(|k| iv.get(k))
                        .map(|(a, b)| (a.saturating_sub(1), b + 1)),
                );
            }
            // Pieces by start: already in order.
            let s = all.len();
            for &x in ivs.iter() {
                push_coalesced(all, s, x);
            }
            busy.push((g as u32, s as u32, all.len() as u32));
        }
        // By first point, in group order.
        keys.extend((busy.iter().enumerate()).map(|(k, &(_, s, e))| {
            (if s < e { all[s as usize].0 } else { 0 } as u64) << 32 | k as u64
        }));
        keys.sort_unstable();
        let mut used = 0;
        for (g, s, e) in keys.iter().map(|&k| busy[k as u32 as usize]) {
            let ivs = &all[s as usize..e as usize];
            let k = match (slots[..used].iter())
                .position(|x| intersect(Ivs::plain(x), Ivs::plain(ivs), 0).is_none())
            {
                Some(k) => k,
                None => {
                    if used == slots.len() {
                        slots.push(Vec::new());
                    }
                    used += 1;
                    used - 1
                }
            };
            let x = &mut slots[k];
            let mid = x.len();
            x.extend_from_slice(ivs);
            merge_tail(x, 0, mid, |iv| iv.0, push_coalesced);
            self.vregs[g as usize].slot = k as u32;
        }
        self.num_slots = used as u32;
    }

    /// The move of `v` to `to` from where it is at `at`: computed again when
    /// it is there (a constant from nothing, a derived value from its
    /// source's location at `at`).
    fn mv<I: TargetInst>(&self, f: &VCode<I>, v: u32, at: Pos, to: Allocation, cls: u8) -> Mv {
        let from = self.alloc_at(v, at).expect("live where it moves");
        self.mv_from(f, v, from, at, to, cls)
    }

    /// `mv` with `v`'s location at `at` known.
    fn mv_from<I: TargetInst>(
        &self,
        f: &VCode<I>,
        v: u32,
        from: Allocation,
        at: Pos,
        to: Allocation,
        cls: u8,
    ) -> Mv {
        match (from.is_none(), f.remat[v as usize]) {
            (true, Remat::AndImm { src, .. }) => {
                let src = self.alloc_at(src, at).expect("the source live");
                mv(src, to, cls, v, true)
            }
            (true, _) => mv(from, to, cls, v, true),
            (false, _) => mv(from, to, cls, v, false),
        }
    }

    /// Moves on every edge whose ends hold a value in different places: on
    /// the successor's side when it has one predecessor (not a resume block,
    /// which a return enters past its moves), else on the predecessor's.
    /// Only values of split groups can be in two places.
    #[inline(never)]
    fn edge_moves<I: TargetInst>(&self, f: &VCode<I>, t: &mut Resolve) -> Result<(), String> {
        let Resolve {
            split,
            moves,
            once,
            cur,
            live,
            ..
        } = t;
        // The piece of each live value at the block's start, found by a
        // cursor per group: blocks come in point order.
        cur.resize(self.vregs.len(), 0);
        for &s in &self.order {
            let preds = f.block_preds(s);
            let si = s.index();
            let resume = f.blocks[si].resume;
            let ss = self.block_start(si as u32);
            live.clear();
            // Live in, so some piece covers the start and every
            // predecessor's end: the last one starting by there. One whose
            // span holds every predecessor's end needs nothing here.
            let pes = preds.iter().map(|p| 2 * self.last[p.index()] + 1);
            let (lo, hi) = (pes.clone().min().unwrap_or(ss), pes.max().unwrap_or(ss));
            for (k, (&w, &m)) in self.live_in(si).iter().zip(split.iter()).enumerate() {
                for v in bits(k, w & m) {
                    let g = self.rep[v as usize];
                    let (a, b) = self.vregs[g as usize].flat;
                    let starts = &self.flat_start[a as usize..b as usize];
                    let mut c = cur[g as usize] as usize;
                    while c + 1 < starts.len() && starts[c + 1] <= ss {
                        c += 1;
                    }
                    cur[g as usize] = c as u32;
                    let x = self.flat[a as usize + c];
                    let (xs, xe) = (self.start(x), self.end(x));
                    if !(xs <= ss && ss < xe) {
                        return Err(format!("v{v} not live into b{si}"));
                    }
                    debug_assert!(self.ivs_of(x).covers(ss));
                    if xs <= lo && hi < xe {
                        continue;
                    }
                    live.push((v, x));
                }
            }
            // The piece at a predecessor's end: the one at the start when
            // its span holds it (the value is live there), its previous one
            // when the predecessor falls through to a piece starting here,
            // else looked up.
            let from = |v: u32, x: u32, pi: usize| {
                let pe = 2 * self.last[pi] + 1;
                let p = &self.pieces[x as usize];
                let y = if p.start <= pe && pe < p.end {
                    x
                } else if pe + 1 == ss {
                    p.prev
                } else {
                    let (a, b) = self.vregs[self.rep[v as usize] as usize].flat;
                    let k = self.flat_start[a as usize..b as usize].partition_point(|&s| s <= pe);
                    if k == 0 {
                        return Err(format!("v{v} not live out of b{pi}"));
                    }
                    self.flat[a as usize + k - 1]
                };
                debug_assert_eq!(Some(y), self.piece_at(self.rep[v as usize], pe));
                Ok(y)
            };
            // A constant no predecessor has where the join wants it: made
            // once at the join's start, not on every edge.
            once.clear();
            if preds.len() > 1 && !resume {
                for &(v, x) in live.iter() {
                    if !matches!(f.remat[v as usize], Remat::Int(_) | Remat::Float(_)) {
                        continue;
                    }
                    let to = self.alloc_of(x);
                    if !to.is_reg() {
                        continue;
                    }
                    let mut held = false;
                    for p in preds {
                        held |= self.alloc_of(from(v, x, p.index())?) == to;
                    }
                    if !held {
                        let cls = self.vregs[v as usize].class;
                        let m = mv(Allocation::none(), to, cls, v, true);
                        moves.push((self.first[si], EDGE_IN, m));
                        once.push(v);
                    }
                }
            }
            // A branch with both edges here is in `preds` once per edge.
            for (n, &p) in preds.iter().enumerate() {
                if preds[..n].contains(&p) {
                    continue;
                }
                let pi = p.index();
                let pe = 2 * self.last[pi] + 1;
                let succs = f.block_succs(p);
                for k in 0..succs.len() {
                    if succs[k] != s {
                        continue;
                    }
                    // Into the block's start, before the predecessor's jump,
                    // or on the edge itself when both have company.
                    let (at, phase) = if preds.len() == 1 && !resume {
                        (self.first[si], EDGE_IN)
                    } else if succs.len() == 1 {
                        (self.last[pi], EDGE_OUT)
                    } else {
                        (self.last[pi], EDGE | (k as u8) << 3)
                    };
                    let before = moves.len();
                    for &(v, y) in live.iter() {
                        let x = from(v, y, pi)?;
                        if x == y {
                            continue;
                        }
                        let (from, to) = (self.alloc_of(x), self.alloc_of(y));
                        // The slot has held it since its definition; a
                        // constant is computed again where it is needed.
                        let g = self.rep[v as usize] as usize;
                        if from != to
                            && !to.is_none()
                            && (!to.is_stack() || self.deferred[g])
                            && !once.contains(&v)
                        {
                            let cls = self.vregs[v as usize].class;
                            moves.push((at, phase, self.mv_from(f, v, from, pe, to, cls)));
                        }
                    }
                    let args = f.branch_blockparams(p, k);
                    for (q, a) in f.block_params(s).iter().zip(args) {
                        let q = q.vreg() as u32;
                        let Some(to) = self.alloc_at(q, ss) else {
                            continue;
                        };
                        let from = (self.alloc_at(a.vreg() as u32, pe)).ok_or_else(|| {
                            format!("argument v{} not live at b{pi}'s end", a.vreg())
                        })?;
                        if from != to {
                            let cls = self.vregs[q as usize].class;
                            let m = self.mv_from(f, a.vreg() as u32, from, pe, to, cls);
                            moves.push((at, phase, m));
                        }
                    }
                    if resume && moves.len() != before {
                        return Err(format!(
                            "moves into resume block b{si}: {:?}",
                            &moves[before..]
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Emit a parallel move as a sequence: a move goes once nothing else
    /// still reads its target; a cycle parks one value in the scratch
    /// register; stack-to-stack goes through the transport register.
    fn sequence(&self, ms: &mut Vec<Mv>, mut emit: impl FnMut(Edit)) -> Result<(), String> {
        ms.retain(|m| m.remat || m.from != m.to);
        let transport = Allocation::reg(self.env.transport);
        let mut put = |m: Mv| {
            if m.remat {
                let class = if m.cls == 0 {
                    RegClass::Int
                } else {
                    RegClass::Float
                };
                let vreg = VReg::new(m.vreg as usize, class);
                let to = if m.to.is_stack() { transport } else { m.to };
                emit(Edit::Remat {
                    to,
                    vreg,
                    src: m.from,
                });
                if m.to.is_stack() {
                    emit(Edit::Move {
                        from: transport,
                        to: m.to,
                    });
                }
            } else if m.from.is_stack() && m.to.is_stack() {
                emit(Edit::Move {
                    from: m.from,
                    to: transport,
                });
                emit(Edit::Move {
                    from: transport,
                    to: m.to,
                });
            } else {
                emit(Edit::Move {
                    from: m.from,
                    to: m.to,
                });
            }
        };
        while !ms.is_empty() {
            if let Some(k) = (0..ms.len()).find(|&k| !ms.iter().any(|m| m.from == ms[k].to)) {
                put(ms.swap_remove(k));
                continue;
            }
            // Every target is still read: a cycle. Park the first target's
            // value and read it from the scratch register.
            let t = ms[0].to;
            let reader = ms.iter().find(|m| m.from == t).expect("a reader");
            let scratch = Allocation::reg(self.env.scratch[reader.cls as usize]);
            if ms.iter().any(|m| m.from == scratch || m.to == scratch) {
                return Err("the scratch register in a parallel move".into());
            }
            put(mv(t, scratch, reader.cls, reader.vreg, false));
            for m in ms.iter_mut() {
                if m.from == t {
                    m.from = scratch;
                }
            }
        }
        Ok(())
    }
}

fn mv(from: Allocation, to: Allocation, cls: u8, vreg: u32, remat: bool) -> Mv {
    Mv {
        from,
        to,
        cls,
        vreg,
        remat,
    }
}

/// The values of word `k` of a set.
fn bits(k: usize, mut w: u64) -> impl Iterator<Item = u32> {
    std::iter::from_fn(move || {
        (w != 0).then(|| {
            let v = (k * 64) as u32 + w.trailing_zeros();
            w &= w - 1;
            v
        })
    })
}
