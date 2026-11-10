//! Before the scan: which values get the registers calls keep
//! (`call-protocol.md`, decision 6). A value live across calls costs a
//! reload or a computation after each; the ones that cost most take those
//! registers for their whole range, and the scan allocates the rest around
//! them. In a function region each such register first holds the caller's
//! value, which goes to a slot when the register is taken and comes back
//! where the region leaves.

use super::build::push_coalesced;
use super::{Ivs, Loc, NONE, PINNED, Pos, Ra, STACK, Want, Why, intersect};
use crate::jit::backend::vcode::{Remat, TargetInst, VCode};

/// What a value must save for a function region to take a register for
/// it. The cost is the caller's value's store and its reload at the return
/// taken (3 in `benefit`'s units), on every path through the region and
/// against frequencies that are guesses: measured, 3 loses to never taking
/// (list +7%), and 24 (a reload in a loop and a half) is level with it.
const PRICE: f32 = 24.0;

impl Ra {
    pub(super) fn keep<I: TargetInst>(&mut self, f: &VCode<I>) {
        let kept = self.env.kept;
        if kept == [0, 0] {
            return;
        }
        for c in 0..2 {
            let n = self.env.regs[c].len();
            self.pin[c].resize_with(n, Vec::new);
        }
        self.callers.extend(f.kept.iter().map(|&(v, _)| v));
        let function = !f.kept.is_empty();
        // Per class: the calls, which take every other register, and the
        // points fixed operands reserve the kept ones over (the caller's
        // values where the region leaves).
        let mut calls: [Vec<Pos>; 2] = [Vec::new(), Vec::new()];
        let mut tails: [Vec<(Pos, Pos)>; 2] = [Vec::new(), Vec::new()];
        for c in 0..2 {
            let others = self.env.regs[c].len() as u32 - kept[c].count_ones();
            calls[c].extend(
                (self.clobs.iter())
                    .filter(|x| 2 * (x.1[c] & !kept[c]).count_ones() >= others)
                    .map(|x| x.0),
            );
            let mut t: Vec<(Pos, Pos)> = (self.fixres[c].iter())
                .filter(|x| kept[c] >> x.3 & 1 != 0)
                .map(|x| (x.0, x.1))
                .collect();
            t.sort_unstable();
            for x in t {
                push_coalesced(&mut tails[c], 0, x);
            }
        }
        // A function region pays per invocation, so its uses weigh by how
        // often one invocation reaches them: a branch's weight split over
        // its hot arms, times the loops'. (The scan's frequencies count a
        // loop's depth alone: every arm of a function without loops at 1.)
        let mut reach: Vec<f32> = Vec::new();
        if function {
            reach.resize(self.freq.len(), 0.0);
            let hot = |ra: &Ra, b: usize| ra.block_start(b as u32) < ra.cold_start;
            reach[self.order[0].index()] = 1.0;
            for &b in &self.order {
                let bi = b.index();
                if !hot(self, bi) {
                    continue;
                }
                let fwd = |s: &&crate::jit::backend::reg::Block| {
                    hot(self, s.index()) && self.rank[s.index()] > self.rank[bi]
                };
                let n = f.block_succs(b).iter().filter(fwd).count();
                for s in f.block_succs(b).iter().filter(fwd) {
                    reach[s.index()] += reach[bi] / n as f32;
                }
            }
            for (r, &q) in reach.iter_mut().zip(&self.freq) {
                *r = if q < 1.0 { q } else { *r * q };
            }
        }
        let mut cands: Vec<(f32, u32)> = Vec::new();
        let mut charged: Vec<u32> = Vec::new();
        for g in 0..self.vregs.len() as u32 {
            let gi = &self.vregs[g as usize];
            if self.rep[g as usize] != g || gi.head == NONE || self.callers.contains(&g) {
                continue;
            }
            let c = gi.class as usize;
            if calls[c].is_empty() {
                continue;
            }
            let b = self.benefit(f, g, &calls[c], &reach, &mut charged);
            if b > 0.0 {
                cands.push((b, g));
            }
        }
        cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut used = [0u32; 2];
        for &(b, g) in &cands {
            let c = self.vregs[g as usize].class as usize;
            let (lo, hi) = self.group_run(g);
            let free = |ra: &Ra, k: usize| {
                let iv = Ivs::plain(&ra.ivs[lo as usize..hi as usize]);
                intersect(Ivs::plain(&ra.pin[c][k]), iv, 0).is_none()
            };
            // One already paid for, else a new one when it pays.
            let mut pick = None;
            let mut m = kept[c] & used[c];
            while m != 0 && pick.is_none() {
                let k = m.trailing_zeros() as usize;
                pick = free(self, k).then_some(k);
                m &= m - 1;
            }
            if pick.is_none() && (!function || b > PRICE) {
                let m = kept[c] & !used[c];
                pick = (m != 0).then(|| m.trailing_zeros() as usize);
            }
            let Some(k) = pick else {
                #[cfg(feature = "tcvm-dev")]
                {
                    let kind = match f.remat[g as usize] {
                        r if !self.vregs[g as usize].remat || matches!(r, Remat::Slot(_)) => 2,
                        r if weight(r) == 1.0 => 0,
                        _ => 1,
                    };
                    self.refused[kind] += 1;
                }
                continue;
            };
            used[c] |= 1 << k;
            self.take(g, c, k, &tails[c]);
        }
        // A function region's caller's values: in its slot through the body
        // when the register was taken, else left where it is, the register
        // no one's.
        for &(v, r) in &f.kept {
            let (c, k) = (
                self.vregs[v as usize].class as usize,
                self.reg_idx[r.index()] as usize,
            );
            let head = self.vregs[v as usize].head;
            let rest = self.pieces[head as usize].next;
            if used[c] >> k & 1 != 0 {
                self.pieces[head as usize].flags |= STACK;
                self.slot_of(v);
                if rest != NONE {
                    self.pieces[rest as usize].loc = Loc::Stack;
                }
            } else {
                if rest != NONE {
                    self.pieces[rest as usize].loc = Loc::Reg(r);
                    self.pieces[rest as usize].flags |= PINNED;
                }
                self.pin[c][k].clear();
                self.pin[c][k].push((0, Pos::MAX));
                self.pinned[c] |= 1 << k;
            }
        }
    }

    /// The run of `ivs` holding group `g`'s intervals.
    fn group_run(&self, g: u32) -> (u32, u32) {
        let mut x = self.vregs[g as usize].head;
        let lo = self.pieces[x as usize].lo;
        while self.pieces[x as usize].next != NONE {
            x = self.pieces[x as usize].next;
        }
        (lo, self.pieces[x as usize].hi)
    }

    /// What keeping group `g` out of a register across the calls it spans
    /// costs: per call, making it again at the use that comes next (later
    /// in line, or earlier in a loop the value is live around), by the
    /// frequency there (`reach`'s in a function region); and its store,
    /// when nothing else needs its slot.
    fn benefit<I: TargetInst>(
        &self,
        f: &VCode<I>,
        g: u32,
        calls: &[Pos],
        reach: &[f32],
        charged: &mut Vec<u32>,
    ) -> f32 {
        let freq = |p: Pos| match reach {
            [] => self.freq_at(p),
            _ => reach[self.block_of_pos(p) as usize],
        };
        let (lo, hi) = self.group_run(g);
        let iv = Ivs::plain(&self.ivs[lo as usize..hi as usize]);
        let (s, e) = (iv.raw[0].0, iv.raw[iv.len() - 1].1);
        let head = &self.pieces[self.vregs[g as usize].head as usize];
        let mut last = head;
        while last.next != NONE {
            last = &self.pieces[last.next as usize];
        }
        let (ulo, uhi) = (head.ulo, last.uhi);
        let us = &self.uses[ulo as usize..uhi as usize];
        let from = |p: Pos| {
            let k = ulo + us.partition_point(|u| u.pos < p) as u32;
            self.want_from(k, uhi, Want::Beneficial)
        };
        charged.clear();
        let a = calls.partition_point(|&q| q < s);
        for &q in &calls[a..] {
            if q >= e {
                break;
            }
            // A call's own result is defined where it clobbers.
            if !iv.covers(q) || self.is_def(g, q) {
                continue;
            }
            let b = self.block_of_pos(q);
            let h0 = self.inner[b as usize];
            let next = from(q + 1);
            let in_loop = |j: u32| {
                h0 == NONE || self.in_loop(self.block_of_pos(self.uses[j as usize].pos), h0)
            };
            if let Some(j) = next.filter(|&j| in_loop(j)) {
                charged.push(j);
                continue;
            }
            // Around a loop: the first use from its header on, before the
            // call.
            let mut h = h0;
            let mut around = None;
            while h != NONE && around.is_none() {
                let hs = self.block_start(h);
                if hs < q && iv.covers(hs) {
                    around = from(hs).filter(|&j| self.uses[j as usize].pos < q);
                }
                h = self.outer[h as usize];
            }
            if let Some(j) = around.or(next) {
                charged.push(j);
            }
        }
        if charged.is_empty() {
            return 0.0;
        }
        charged.sort_unstable();
        charged.dedup();
        let gi = &self.vregs[g as usize];
        let w = if gi.remat {
            weight(f.remat[g as usize])
        } else {
            2.0
        };
        let mut b: f32 = (charged.iter())
            .map(|&j| w * freq(self.uses[j as usize].pos))
            .sum();
        if !gi.remat && gi.slot == NONE {
            for &m in self.members_of(g) {
                let mi = &self.vregs[m as usize];
                if mi.def != NONE {
                    b += freq(mi.def);
                } else if mi.param_of != NONE && !mi.args_in_group {
                    b += freq(self.block_start(mi.param_of));
                }
            }
        }
        b
    }

    /// Group `g` takes register `k` of class `c`: every piece of it not
    /// placed yet, but where a fixed operand of another value reserves the
    /// register (`tails`), which the scan allocates.
    fn take(&mut self, g: u32, c: usize, k: usize, tails: &[(Pos, Pos)]) {
        let r = self.env.regs[c][k];
        let mut x = self.vregs[g as usize].head;
        while x != NONE {
            let next = self.pieces[x as usize].next;
            if self.pieces[x as usize].loc == Loc::None {
                let mut cur = x;
                let from = tails.partition_point(|t| t.1 <= self.start(cur));
                for &(a, b) in &tails[from..] {
                    if a >= self.end(cur) {
                        break;
                    }
                    if self.ivs_of(cur).first_in(a, b).is_none() {
                        continue;
                    }
                    // The part before the instruction keeps the register;
                    // the part at it is the scan's.
                    let at = match self.split_at(cur, a & !1, Why::Hold) {
                        Some(t) if t != cur => {
                            self.pin_piece(cur, c, k, r);
                            t
                        }
                        Some(t) => t,
                        None => unreachable!("a reservation inside the piece"),
                    };
                    match self.split_at(at, (b + 1) & !1, Why::Hold) {
                        Some(t) if t == at => cur = at,
                        Some(t) => {
                            self.queue(at);
                            cur = t;
                        }
                        None => {
                            self.queue(at);
                            cur = NONE;
                            break;
                        }
                    }
                }
                if cur != NONE {
                    self.pin_piece(cur, c, k, r);
                }
            }
            x = next;
        }
        self.pin[c][k].sort_unstable();
        self.pinned[c] |= 1 << k;
    }

    fn pin_piece(&mut self, x: u32, c: usize, k: usize, r: crate::jit::backend::reg::PReg) {
        let p = &mut self.pieces[x as usize];
        p.loc = Loc::Reg(r);
        p.flags |= PINNED;
        let iv = p.ivs(&self.ivs);
        self.pin[c][k].extend((0..iv.len()).map(|j| iv.get(j)));
    }
}

/// What making a value again costs, in instructions: one for a constant a
/// single move makes, two for another constant or a load.
fn weight(r: Remat) -> f32 {
    match r {
        Remat::Int(k) => {
            let chunks = |k: u64| (0..4).filter(|i| k >> (16 * i) & 0xffff != 0).count();
            if chunks(k) <= 1 || chunks(!k) <= 1 {
                1.0
            } else {
                2.0
            }
        }
        _ => 2.0,
    }
}
