//! After the scan: each operand's allocation and the moves between pieces,
//! across edges, into fixed registers, and the stores after definitions,
//! each gap's moves sequenced as one parallel move.

use super::build::{merge_tail, push_coalesced};
use super::{Edit, Ivs, Loc, NONE, Need, Output, Pos, Ra, STACK, Want, class_index, intersect};
use crate::jit::backend::reg::{
    Allocation, Block, Inst, OperandKind, ProgPoint, RegClass, SpillSlot, VReg,
};
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

/// `cleanup`'s effect entries after the locations (a definition's, with
/// its vreg): a clobber of the integer or the float registers of a mask, a
/// register use, a move or an edge's move by its index in `sorted`.
const CLOBBER: u32 = u32::MAX - 1;
const CLOBBER1: u32 = u32::MAX - 2;
const USE: u32 = u32::MAX - 3;
const MOVE: u32 = u32::MAX - 4;
const EDGE_MOVE: u32 = u32::MAX - 5;

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
    /// `cleanup`'s: the moves changing nothing, each block's contents at
    /// its start and the two being worked on, the worklist.
    dead: Vec<bool>,
    held: Vec<u32>,
    now: Vec<u32>,
    next: Vec<u32>,
    reached: Vec<bool>,
    work: std::collections::VecDeque<u32>,
    queued: Vec<bool>,
    keep: Vec<bool>,
    genkill: Vec<u32>,
    mloc: Vec<(u32, u32)>,
    eff: Vec<(u32, u32)>,
    eoff: Vec<u32>,
    held16: Vec<u16>,
    now16: Vec<u16>,
    next16: Vec<u16>,
}

impl Resolve {
    pub(super) fn clear(&mut self) {
        clear!(self; split, moves, sorted, keys, group, busy, all, ivs, once, cur, live, dead, held,
            now, next, reached, work, queued, keep, genkill, mloc, eff, eoff, held16, now16, next16);
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
                let a = match u.need() {
                    Need::Reg => match p.loc {
                        Loc::Reg(r) => Allocation::reg(r),
                        _ => return Err(format!("v{} needs a register at {}", p.g, u.pos)),
                    },
                    Need::Any => self.alloc_of(x),
                    Need::Fixed(r) => Allocation::reg(r),
                };
                self.allocs[u.alloc() as usize] = a;
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
            let ps = self.pieces_of(g);
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
                let m = self.mv_member(f, g, at - 1, at - 1, self.alloc_of(w[1]), cls);
                moves.push((at / 2, phase, m));
            }
            // Into fixed registers, from wherever the value is when the
            // instruction starts.
            for &x in ps {
                let p = &self.pieces[x as usize];
                let mut k = p.ulo;
                while let Some(j) = self.want_from(k, p.uhi, Want::Fixed) {
                    let u = self.uses[j as usize];
                    let Need::Fixed(r) = u.need() else {
                        unreachable!("a fixed use")
                    };
                    let n = u.pos / 2;
                    let m = self.mv_member(f, g, u.pos, 2 * n, Allocation::reg(r), cls);
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
            ..
        } = t;
        keys.clear(); // `color_slots` sorted with them too.
        keys.extend((moves.iter().enumerate()).map(|(k, &(l, phase, _))| {
            ((self.at_lin[l as usize].index() as u64) << 8 | phase as u64) << 32 | k as u64
        }));
        keys.sort_unstable();
        sorted.extend(keys.iter().map(|&k| moves[k as u32 as usize]));
        self.cleanup(f, t);
        let Resolve {
            sorted: moves,
            dead,
            group,
            ..
        } = t;
        let mut edits: Vec<(ProgPoint, Edit)> = Vec::with_capacity(moves.len());
        let mut edge_edits: Vec<(Inst, u8, Edit)> = Vec::new();
        let mut k = 0;
        while k < moves.len() {
            let (l, phase) = (moves[k].0, moves[k].1);
            group.clear();
            while k < moves.len() && moves[k].0 == l && moves[k].1 == phase {
                if !dead[k] {
                    group.push(moves[k].2);
                }
                k += 1;
            }
            if group.is_empty() {
                continue;
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
            let s = self.flat.len();
            self.flat_off.push(s as u32);
            let mut x = self.vregs[g].head;
            while x != NONE {
                self.flat.push(x);
                self.flat_start.push(self.start(x));
                x = self.pieces[x as usize].next;
            }
            let ps = &self.flat[s..];
            let loc = |x: &u32| self.pieces[*x as usize].loc;
            if ps.iter().any(|x| loc(x) != loc(&ps[0])) {
                for &m in self.members_of(g as u32) {
                    split[m as usize / 64] |= 1 << (m % 64);
                }
            }
        }
        self.flat_off.push(self.flat.len() as u32);
    }

    /// The member of group `g` live at `p`, whose own intervals cover it
    /// (members' ranges are disjoint); none where no member is.
    fn member_at(&self, g: u32, p: Pos) -> Option<u32> {
        let ms = self.members_of(g);
        if ms.len() == 1 {
            return Some(g);
        }
        ms.iter().copied().find(|&m| {
            let run = &self.ivs[self.own[m as usize] as usize..self.own[m as usize + 1] as usize];
            let k = run.partition_point(|iv| iv.1 <= p);
            k < run.len() && run[k].0 <= p
        })
    }

    /// A move of group `g`'s value at `p` (`mv`), named for the member live
    /// there; for none, it names no value.
    fn mv_member<I: TargetInst>(
        &self,
        f: &VCode<I>,
        g: u32,
        p: Pos,
        at: Pos,
        to: Allocation,
        cls: u8,
    ) -> Mv {
        let v = self.member_at(g, p);
        let mut m = self.mv(f, v.unwrap_or(g), at, to, cls);
        if v.is_none() {
            debug_assert!(!m.remat);
            m.vreg = UNKNOWN as u32;
        }
        m
    }

    /// The location index of an allocation: allocatable registers by class,
    /// then the slots.
    fn loc_index(&self, a: Allocation) -> Option<usize> {
        if let Some(r) = a.as_reg() {
            let k = self.reg_idx[r.index()];
            (k != NONE).then(|| k as usize + class_index(r.class()) * self.env.regs[0].len())
        } else {
            a.as_stack()
                .map(|s| s.index() + self.env.regs[0].len() + self.env.regs[1].len())
        }
    }

    /// Drop the moves the code does not need: those that change nothing (a
    /// reload or remat into a register still holding the value, a store into
    /// a slot that kept it) and those into a register nothing reads before it
    /// is written again. The scan's intervals are linear, so a register is
    /// taken, or a piece reaches a block, at positions control flow need not
    /// pass through; what each location holds on every path into a point
    /// (forward), and which are read on some path from it (backward), are
    /// followed instead, values by vreg so a group's members stay apart.
    /// Dropping a dead write can leave an older value in place, so the passes
    /// repeat while the backward one finds more.
    #[inline(never)]
    fn cleanup<I: TargetInst>(&self, f: &VCode<I>, t: &mut Resolve) {
        t.dead.resize(t.sorted.len(), false);
        if f.num_vregs() >= UNKNOWN as usize {
            return;
        }
        let width = self.env.regs[0].len() + self.env.regs[1].len() + self.num_slots as usize;
        let Resolve {
            sorted,
            mloc,
            eff,
            eoff,
            ..
        } = t;
        let loc = |a: Allocation| self.loc_index(a).map_or(NONE, |l| l as u32);
        mloc.extend(sorted.iter().map(|m| (loc(m.2.to), loc(m.2.from))));
        // Each instruction's effects in order (`eoff`): the moves before it
        // (an edge's tagged apart), in phase order, its register uses,
        // clobbers and definitions. `sorted` is by instruction, so a cursor
        // finds the moves; a block's instructions are a range, so its effects
        // are.
        eoff.resize(f.num_insts() + 1, 0);
        let mut k = 0;
        for (n, off) in eoff[..f.num_insts()].iter_mut().enumerate() {
            let i = Inst::new(n);
            *off = eff.len() as u32;
            while k < sorted.len() && self.at_lin[sorted[k].0 as usize].index() == n {
                let tag = if sorted[k].1 & 7 == EDGE {
                    EDGE_MOVE
                } else {
                    MOVE
                };
                eff.push((tag, k as u32));
                k += 1;
            }
            // Reads before writes: backward, the writes kill first.
            let base = f.ops[i.index()].0 as usize;
            let ops = f.inst_operands(i);
            for (k, op) in ops.iter().enumerate() {
                if op.kind() == OperandKind::Use
                    && self.allocs[base + k].is_reg()
                    && let Some(l) = self.loc_index(self.allocs[base + k])
                {
                    eff.push((USE, l as u32));
                }
            }
            let mut m = [0u32; 2];
            for r in f.inst_clobbers(i) {
                let k = self.reg_idx[r.index()];
                if k != NONE {
                    m[class_index(r.class())] |= 1 << k;
                }
            }
            for (tag, &m) in [CLOBBER, CLOBBER1].iter().zip(&m) {
                if m != 0 {
                    eff.push((*tag, m));
                }
            }
            for (k, op) in ops.iter().enumerate() {
                if op.kind() == OperandKind::Def
                    && let Some(l) = self.loc_index(self.allocs[base + k])
                {
                    eff.push((l as u32, op.vreg().vreg() as u32));
                }
            }
        }
        debug_assert_eq!(k, sorted.len());
        eoff[f.num_insts()] = eff.len() as u32;
        self.redundant(f, t, width);
        while self.unread(f, t, width) && self.redundant(f, t, width) {}
    }

    /// Block `b`'s effects, and its last instruction's.
    fn effects<I: TargetInst>(
        &self,
        f: &VCode<I>,
        eoff: &[u32],
        b: usize,
    ) -> ((usize, usize), (usize, usize)) {
        let r = f.block_insns(Block::new(b));
        let (first, last) = (r.first().index(), r.last().index());
        (
            (eoff[first] as usize, eoff[last + 1] as usize),
            (eoff[last] as usize, eoff[last + 1] as usize),
        )
    }

    /// Forward over effects `a..e`, `edge`'s moves if any else a block's:
    /// what each location holds after them. A gap's run of moves of one phase is a
    /// parallel move, read before written; those changing nothing are marked
    /// when `mark`, dead ones skipped.
    #[inline(never)]
    fn hold(
        &self,
        ms: &mut Marks,
        st: &mut [u16],
        (a, e): (usize, usize),
        edge: Option<u8>,
        mark: bool,
    ) {
        let n0 = self.env.regs[0].len();
        let Marks {
            sorted,
            dead,
            eff,
            mloc,
            marked,
            ..
        } = ms;
        let mut k = a;
        while k < e {
            let (tag, j) = eff[k];
            if tag == MOVE || tag == EDGE_MOVE {
                // One gap's: an instruction between two may have no effects.
                let (l, phase) = (sorted[j as usize].0, sorted[j as usize].1);
                let same = |x: (u32, u32)| {
                    x.0 == tag && (sorted[x.1 as usize].0, sorted[x.1 as usize].1) == (l, phase)
                };
                let mut e2 = k + 1;
                while e2 < e && same(eff[e2]) {
                    e2 += 1;
                }
                let mine = (phase & 7 == EDGE).then_some(phase >> 3) == edge;
                if mine && mark {
                    for &(_, j) in &eff[k..e2] {
                        let j = j as usize;
                        let (l, v) = (mloc[j].0, sorted[j].2.vreg as u16);
                        if !dead[j] && l != NONE && v != UNKNOWN && st[l as usize] == v {
                            dead[j] = true;
                            *marked += 1;
                        }
                    }
                }
                if mine {
                    for &(_, j) in &eff[k..e2] {
                        let j = j as usize;
                        if !dead[j] && mloc[j].0 != NONE {
                            st[mloc[j].0 as usize] = sorted[j].2.vreg as u16;
                        }
                    }
                }
                k = e2;
                continue;
            }
            if edge.is_none() {
                match tag {
                    USE => {}
                    CLOBBER => clobber(st, 0, j),
                    CLOBBER1 => clobber(st, n0, j),
                    _ => st[tag as usize] = j as u16,
                }
            }
            k += 1;
        }
    }

    /// Forward: what each location holds on every path (`NONE` for
    /// unknown), a fixpoint from the entry; then a pass marking the moves
    /// changing nothing. Whether any was.
    fn redundant<I: TargetInst>(&self, f: &VCode<I>, t: &mut Resolve, width: usize) -> bool {
        let Resolve {
            sorted,
            dead,
            held16: held,
            now16: now,
            next16: next,
            reached,
            work,
            queued,
            keep,
            mloc,
            eff,
            eoff,
            ..
        } = t;
        let ms = &mut Marks {
            sorted,
            dead,
            keep,
            mloc,
            eff,
            marked: 0,
        };
        let nb = f.num_blocks();
        held.clear();
        held.resize(nb * width, UNKNOWN);
        reached.clear();
        reached.resize(nb, false);
        queued.clear();
        queued.resize(nb, false);
        now.resize(width, UNKNOWN);
        next.resize(width, UNKNOWN);
        // Blocks in allocation order (predecessors first but for back
        // edges), those not reached yet left for when they are.
        reached[f.entry_block().index()] = true;
        queued.fill(true);
        work.extend(self.order.iter().map(|b| b.index() as u32));
        let mut mark = false;
        loop {
            let b = match work.pop_front() {
                Some(b) => b as usize,
                // Settled: once more over every block, marking.
                None if !mark => {
                    mark = true;
                    work.extend((0..nb as u32).filter(|&b| reached[b as usize]));
                    continue;
                }
                None => break,
            };
            queued[b] = false;
            if !reached[b] {
                continue;
            }
            let (all, last) = self.effects(f, eoff, b);
            now.copy_from_slice(&held[b * width..(b + 1) * width]);
            self.hold(ms, now, all, None, mark);
            // Only a branch has moves on its edges: a jump's successor takes
            // the state as it is.
            let edges = eff[last.0..last.1].iter().any(|x| x.0 == EDGE_MOVE);
            for (k, s) in f.block_succs(Block::new(b)).iter().enumerate() {
                let si = s.index();
                let out: &[u16] = if edges {
                    next.copy_from_slice(now);
                    self.hold(ms, next, last, Some(k as u8), mark);
                    next
                } else {
                    now
                };
                if mark {
                    continue;
                }
                let into = &mut held[si * width..(si + 1) * width];
                let changed = if !reached[si] {
                    reached[si] = true;
                    into.copy_from_slice(out);
                    true
                } else {
                    let mut c = false;
                    for (h, &n) in into.iter_mut().zip(out.iter()) {
                        if *h != n && *h != UNKNOWN {
                            *h = UNKNOWN;
                            c = true;
                        }
                    }
                    c
                };
                if changed && !std::mem::replace(&mut queued[si], true) {
                    work.push_back(si as u32);
                }
            }
        }
        ms.marked != 0
    }

    /// Backward over effects `a..e` (`edge`'s moves, or a block's): the
    /// locations live before them. A move into a register not read after it
    /// is unread, marked dead when `mark`, and reads nothing; with `exact`
    /// off every move reads its source (a block summary). Stores stay: a
    /// slot's readers are explicit, but the saving is small and the frame
    /// image copies them all.
    #[inline(never)]
    fn read(
        &self,
        ms: &mut Marks,
        live: &mut [u32],
        kill: &mut [u32],
        (a, e): (usize, usize),
        edge: Option<u8>,
        mode: Mode,
    ) {
        let (exact, mark) = (mode != Mode::Summary, mode == Mode::Mark);
        let write = |live: &mut [u32], kill: &mut [u32], l: usize| {
            clr(live, l);
            if !kill.is_empty() {
                set(kill, l);
            }
        };
        let n0 = self.env.regs[0].len();
        let Marks {
            sorted,
            dead,
            keep,
            eff,
            mloc,
            marked,
        } = ms;
        let mut e = e;
        while e > a {
            let (tag, j) = eff[e - 1];
            if tag == MOVE || tag == EDGE_MOVE {
                let (l, phase) = (sorted[j as usize].0, sorted[j as usize].1);
                let same = |x: (u32, u32)| {
                    x.0 == tag && (sorted[x.1 as usize].0, sorted[x.1 as usize].1) == (l, phase)
                };
                let mut k = e - 1;
                while k > a && same(eff[k - 1]) {
                    k -= 1;
                }
                if (phase & 7 == EDGE).then_some(phase >> 3) == edge {
                    // Unread before any target is cleared: a parallel move.
                    keep.clear();
                    for &(_, j) in &eff[k..e] {
                        let j = j as usize;
                        let to = mloc[j].0;
                        let unread = exact
                            && !dead[j]
                            && !sorted[j].2.to.is_stack()
                            && to != NONE
                            && !has(live, to as usize);
                        if unread && mark {
                            dead[j] = true;
                            *marked += 1;
                        }
                        keep.push(!dead[j] && !unread);
                    }
                    for (n, &(_, j)) in eff[k..e].iter().enumerate() {
                        if keep[n] && mloc[j as usize].0 != NONE {
                            write(live, kill, mloc[j as usize].0 as usize);
                        }
                    }
                    for (n, &(_, j)) in eff[k..e].iter().enumerate() {
                        if keep[n] && mloc[j as usize].1 != NONE {
                            set(live, mloc[j as usize].1 as usize);
                        }
                    }
                }
                e = k;
                continue;
            }
            if edge.is_none() {
                match tag {
                    USE => set(live, j as usize),
                    CLOBBER => {
                        for k in bits(0, j as u64) {
                            write(live, kill, k as usize);
                        }
                    }
                    CLOBBER1 => {
                        for k in bits(0, j as u64) {
                            write(live, kill, n0 + k as usize);
                        }
                    }
                    _ => write(live, kill, tag as usize),
                }
            }
            e -= 1;
        }
    }

    /// Backward: the locations read on some path from each point. Each
    /// block's reads before any write (`reads`) and writes (`kill`) are
    /// summarized once, the fixpoint from the end runs on the sets, and one
    /// exact walk marks the unread moves. Whether any was.
    fn unread<I: TargetInst>(&self, f: &VCode<I>, t: &mut Resolve, width: usize) -> bool {
        let Resolve {
            sorted,
            dead,
            held,
            now,
            next,
            work,
            queued,
            keep,
            genkill,
            mloc,
            eff,
            eoff,
            ..
        } = t;
        let ms = &mut Marks {
            sorted,
            dead,
            keep,
            mloc,
            eff,
            marked: 0,
        };
        let nb = f.num_blocks();
        let words = width.div_ceil(32);
        // `held` holds each block's live-in set, `now`/`next` one set each.
        held.clear();
        held.resize(nb * words, 0);
        now.clear();
        now.resize(words, 0);
        next.clear();
        next.resize(words, 0);
        queued.clear();
        queued.resize(nb, true);
        genkill.clear();
        genkill.resize(2 * nb * words, 0);
        // A block's summary: what it reads before writing (`reads`, the
        // live-in of nothing live out) and what it writes (`kill`).
        for b in 0..nb {
            let (reads, kill) = genkill[2 * b * words..2 * (b + 1) * words].split_at_mut(words);
            let (all, _) = self.effects(f, eoff, b);
            self.read(ms, reads, kill, all, None, Mode::Summary);
        }
        work.extend(self.order.iter().rev().map(|b| b.index() as u32));
        let mut mark = false;
        loop {
            let b = match work.pop_front() {
                Some(b) => b as usize,
                None if !mark => {
                    mark = true;
                    work.extend(0..nb as u32);
                    continue;
                }
                None => break,
            };
            queued[b] = false;
            let (all, last) = self.effects(f, eoff, b);
            // Live out: what each successor needs after that edge's moves.
            now.fill(0);
            for (k, s) in f.block_succs(Block::new(b)).iter().enumerate() {
                let si = s.index();
                next.copy_from_slice(&held[si * words..(si + 1) * words]);
                let mode = if mark { Mode::Mark } else { Mode::Exact };
                self.read(ms, next, &mut [], last, Some(k as u8), mode);
                for (o, &x) in now.iter_mut().zip(next.iter()) {
                    *o |= x;
                }
            }
            if mark {
                self.read(ms, now, &mut [], all, None, Mode::Mark);
                continue;
            }
            let (reads, kill) = genkill[2 * b * words..2 * (b + 1) * words].split_at(words);
            for ((o, &g), &k) in now.iter_mut().zip(reads).zip(kill) {
                *o = g | (*o & !k);
            }
            let into = &mut held[b * words..(b + 1) * words];
            if into != &now[..] {
                into.copy_from_slice(now);
                for p in f.block_preds(Block::new(b)) {
                    if !std::mem::replace(&mut queued[p.index()], true) {
                        work.push_back(p.index() as u32);
                    }
                }
            }
        }
        ms.marked != 0
    }

    fn slot(&self, g: u32) -> Allocation {
        Allocation::stack(SpillSlot::new(self.vregs[g as usize].slot as usize))
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
            } else if p.flags & STACK != 0 {
                return false;
            }
        }
        let mut at_defs = 0f32;
        for &v in self.members_of(g as u32) {
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
            let (fa, fb) = self.flat_run(g);
            let mut c = fa as usize;
            for k in a..b {
                let (pos, op) = self.snaps[k];
                while c + 1 < fb as usize && self.flat_start[c + 1] <= pos {
                    c += 1;
                }
                let x = self.flat[c];
                debug_assert!(self.ivs_of(x).covers(pos));
                self.allocs[(op >> 1) as usize] = if op & 1 != 0 {
                    self.pieces[x as usize].flags |= STACK;
                    self.slot(g)
                } else {
                    self.alloc_of(x)
                };
            }
        }
    }

    /// A value computed from one on the stack (or in its frame slot) costs
    /// a load and the computation where reloading it costs the load alone:
    /// one spilled where its source is on the stack gets a slot, stored
    /// after its definition, when that is in a register.
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
                            .is_some_and(|l| l.is_stack() || l.is_none())
                    })
            });
            if def_in_reg && src_on_stack {
                let (a, b) = self.flat_run(v as u32);
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
            // The source nowhere is in its frame slot (`Remat::Slot`).
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
                    let (a, b) = self.flat_run(g);
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
                    let (a, b) = self.flat_run(self.rep[v as usize]);
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
                    if !f.remat[v as usize].is_free() {
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

/// `cleanup`'s moves and marks: the sorted moves, the ones dropped, and a
/// parallel move's kept ones.
struct Marks<'a> {
    sorted: &'a [(u32, u8, Mv)],
    dead: &'a mut [bool],
    keep: &'a mut Vec<bool>,
    /// Each move's locations (`NONE` for none), the effects, and how many
    /// moves this pass marked.
    mloc: &'a [(u32, u32)],
    eff: &'a [(u32, u32)],
    marked: u32,
}

/// How `read` treats moves: every one reads its source (a block summary,
/// also collecting what the block writes), only those whose target is read
/// after (exact), and marking the others dead.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Summary,
    Exact,
    Mark,
}

/// Registers `mask` (indices from `base`) hold nothing known.
fn clobber(st: &mut [u16], base: usize, mask: u32) {
    for k in bits(0, mask as u64) {
        st[base + k as usize] = UNKNOWN;
    }
}

/// `cleanup`'s forward state: what a location holds, by vreg (so a region of
/// more values than fit gets no cleanup).
const UNKNOWN: u16 = u16::MAX;

fn set(s: &mut [u32], l: usize) {
    s[l / 32] |= 1 << (l % 32);
}

fn clr(s: &mut [u32], l: usize) {
    s[l / 32] &= !(1 << (l % 32));
}

fn has(s: &[u32], l: usize) -> bool {
    s[l / 32] >> (l % 32) & 1 != 0
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
