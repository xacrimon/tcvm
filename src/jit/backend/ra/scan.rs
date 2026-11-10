//! The scan: pieces in start order take a free register, evict a holder
//! whose next register use comes later, or spill until they need one
//! (V8's `LinearScanAllocator`, `register-allocator.cc`).

use std::cmp::Reverse;

use super::{FIXED, Held, Loc, MAX_REGS, NONE, Piece, Pos, Ra, Want, Why, intersect};
use crate::jit::backend::reg::PReg;
use crate::jit::backend::vcode::{TargetInst, VCode};

type Regs = [Pos; MAX_REGS];

/// Per register of a piece's class, from the piece's start: where it is
/// free until, where an inactive holder's interval next meets the piece,
/// and where a reservation (`block_pos`) takes it; the holders' next uses
/// (`use_pos`) once asked for. Fixed through one `process`: the piece and
/// the sets do not change before it decides.
struct Avail {
    free: Regs,
    inact: Regs,
    fixed: Regs,
    use_pos: Option<Regs>,
}

impl Ra {
    pub(super) fn scan<I: TargetInst>(&mut self, f: &VCode<I>) -> Result<(), String> {
        // Every step moves forward, so a function of `n` instructions takes
        // far fewer; past this the scan is going round in circles.
        let mut budget = 64 * self.at_lin.len() + 10_000;
        while let Some((s, c)) = self.pop() {
            budget = budget
                .checked_sub(1)
                .ok_or_else(|| format!("the scan made no progress at {s}"))?;
            let p = &self.pieces[c as usize];
            // Spilled while queued (`maybe_spill_previous`) or split since.
            if p.loc != Loc::None || p.start != s {
                if p.loc == Loc::None {
                    self.queue(c);
                }
                continue;
            }
            self.advance(s);
            // A piece starting at a late point (not a definition) holds its
            // value where the cut left it: the stack, until a gap can reload.
            if s & 1 != 0 && !self.is_def(self.pieces[c as usize].g, s) {
                self.spill_tail(c)?;
                continue;
            }
            self.process(f, c)?;
        }
        Ok(())
    }

    /// The unallocated piece first by start (`build::key`).
    fn pop(&mut self) -> Option<(Pos, u32)> {
        let a = self.initial.get(self.next_initial).copied();
        let k = match (a, self.unhandled.peek()) {
            (Some(a), Some(&Reverse(b))) if b < a => self.unhandled.pop()?.0,
            (Some(a), _) => {
                self.next_initial += 1;
                a
            }
            (None, _) => self.unhandled.pop()?.0,
        };
        Some(((k >> 32) as Pos, k as u32))
    }

    fn queue(&mut self, x: u32) {
        let k = super::build::key(self.start(x), x);
        self.unhandled.push(Reverse(k));
    }

    /// Retire pieces that end by `pos`; move the rest between active (they
    /// cover `pos`) and inactive (a hole there). Only entries whose
    /// `until` has come look at their piece.
    #[inline(never)]
    fn advance(&mut self, pos: Pos) {
        if pos < self.next_change {
            return;
        }
        let (pieces, ivs) = (&self.pieces, &self.ivs);
        let mut back = std::mem::take(&mut self.scratch);
        back.clear();
        let mut next = Pos::MAX;
        // Listed pieces start by `pos`: the interval ending after it covers
        // it unless it starts later. Kept entries stay in order.
        let mut keep = 0;
        for k in 0..self.active.len() {
            let mut h = self.active[k];
            if h.until <= pos {
                let p = &pieces[h.x as usize];
                if p.end <= pos {
                    continue;
                }
                let iv = p.ivs(ivs);
                let (s, e) = iv.get(iv.after(pos));
                if s > pos {
                    h.until = s;
                    next = next.min(s);
                    back.push(h);
                    continue;
                }
                h.until = e;
            }
            next = next.min(h.until);
            self.active[keep] = h;
            keep += 1;
        }
        self.active.truncate(keep);
        let mut keep = 0;
        for k in 0..self.inactive.len() {
            let mut h = self.inactive[k];
            if h.until <= pos {
                let p = &pieces[h.x as usize];
                if p.end <= pos {
                    continue;
                }
                let iv = p.ivs(ivs);
                let (s, e) = iv.get(iv.after(pos));
                if s <= pos {
                    h.until = e;
                    next = next.min(e);
                    self.active.push(h);
                    continue;
                }
                h.until = s;
            }
            next = next.min(h.until);
            self.inactive[keep] = h;
            keep += 1;
        }
        self.inactive.truncate(keep);
        self.inactive.extend_from_slice(&back);
        self.scratch = back;
        self.next_change = next;
    }

    /// `c`, starting at the scan's position, holds its register from here.
    fn activate(&mut self, c: u32) {
        let r = self.reg_of(c).expect("a register");
        let until = self.ivs_of(c).get(0).1;
        self.active.push(Held {
            x: c,
            until,
            cls: self.pieces[c as usize].class,
            k: self.reg_idx[r.index()] as u8,
        });
        self.next_change = self.next_change.min(until);
    }

    fn class_of(&self, x: u32) -> usize {
        self.pieces[x as usize].class as usize
    }

    fn reg_of(&self, x: u32) -> Option<PReg> {
        match self.pieces[x as usize].loc {
            Loc::Reg(r) => Some(r),
            _ => None,
        }
    }

    /// Per register of `c`'s class: the first point of `c` a reservation
    /// not its own takes (a clobber, another value's fixed operand).
    fn fixed_conflicts(&self, c: u32) -> Regs {
        let cls = self.class_of(c);
        let p = &self.pieces[c as usize];
        let (s, e, iv) = (p.start, p.end, p.ivs(&self.ivs));
        let mut out = [Pos::MAX; MAX_REGS];
        // A clobber takes every register it names at once: the first one
        // `c` covers settles them.
        let mut open = self.clobregs[cls];
        let mut j = 0;
        let from = cursor(&self.clobcur, &self.clobs, |x| x.0 < s);
        for &(q, m) in &self.clobs[from..] {
            if open == 0 || q >= e {
                break;
            }
            while iv.get(j).1 <= q {
                j += 1;
            }
            if iv.get(j).0 <= q {
                let mut b = m[cls] & open;
                open &= !b;
                while b != 0 {
                    out[b.trailing_zeros() as usize] = q;
                    b &= b - 1;
                }
            }
        }
        // Reservations by point (each spans at most two): one at or past a
        // register's best settles nothing more for it.
        let mut pending = self.fixregs[cls];
        let list = &self.fixres[cls];
        let from = cursor(&self.fixcur[cls], list, |x| x.0 + 2 <= s);
        for &(a, b, o, k) in &list[from..] {
            if pending == 0 || a >= e {
                break;
            }
            let k = k as usize;
            if a >= out[k] {
                pending &= !(1 << k);
            } else if self.rep[o as usize] != p.g
                && let Some(x) = iv.first_in(a.max(s), b)
            {
                out[k] = out[k].min(x);
            }
        }
        out
    }

    #[inline(never)]
    fn avail(&self, c: u32) -> Avail {
        let cls = self.class_of(c);
        let n = self.env.regs[cls].len();
        let iv = self.ivs_of(c);
        let mut free = [Pos::MAX; MAX_REGS];
        let mut inact = [Pos::MAX; MAX_REGS];
        // `% MAX_REGS` only spares the bounds checks.
        for h in &self.active {
            if h.cls as usize == cls {
                free[h.k as usize % MAX_REGS] = 0;
            }
        }
        let e = self.end(c);
        for h in &self.inactive {
            let k = h.k as usize % MAX_REGS;
            // Not before its next interval: past `c` or the best so far
            // settles nothing.
            if h.cls as usize != cls || h.until >= e.min(inact[k]) {
                continue;
            }
            // They meet where the holder's next interval starts when `c`
            // covers it; the holder has nothing before there.
            let x = if iv.covers(h.until) {
                Some(h.until)
            } else {
                intersect(self.ivs_of(h.x), iv, h.until)
            };
            if let Some(x) = x {
                inact[k] = inact[k].min(x);
            }
        }
        let fixed = self.fixed_conflicts(c);
        for k in 0..n {
            free[k] = free[k].min(inact[k]).min(fixed[k]);
        }
        Avail {
            free,
            inact,
            fixed,
            use_pos: None,
        }
    }

    /// V8's `use_pos`: per register, where its holders next want it.
    fn use_pos(&self, c: u32, a: &mut Avail) -> Regs {
        if let Some(u) = a.use_pos {
            return u;
        }
        let cls = self.class_of(c);
        let s = self.start(c);
        let mut use_pos = [Pos::MAX; MAX_REGS];
        for h in &self.active {
            if h.cls as usize == cls {
                let next = (self.next_use(h.x, s, Want::Beneficial)).map_or(Pos::MAX, |u| u.pos);
                use_pos[h.k as usize] = use_pos[h.k as usize].min(next);
            }
        }
        let n = self.env.regs[cls].len();
        for (k, u) in use_pos[..n].iter_mut().enumerate() {
            *u = (*u).min(a.inact[k]).min(a.fixed[k]);
        }
        a.use_pos = Some(use_pos);
        use_pos
    }

    /// The register `c` would like: that of its first fixed use, else its
    /// previous piece's, else that of the fixed definition right after it.
    fn hint(&self, c: u32) -> Option<usize> {
        let p = &self.pieces[c as usize];
        let mut k = p.ulo;
        let fixed = loop {
            let Some(j) = self.want_from(k, p.uhi, Want::Fixed) else {
                break None;
            };
            if let super::Need::Fixed(r) = self.uses[j as usize].need() {
                break Some(r);
            }
            k = j + 1;
        };
        let prev = || (p.prev != NONE).then(|| self.reg_of(p.prev)).flatten();
        let next = || {
            (p.next != NONE && self.pieces[p.next as usize].flags & FIXED != 0)
                .then(|| self.reg_of(p.next))
                .flatten()
        };
        let r = fixed.or_else(prev).or_else(next)?;
        let k = self.reg_idx[r.index()];
        (k != NONE && r.class() == self.env.regs[self.class_of(c)][k as usize].class())
            .then_some(k as usize)
    }

    /// V8's `PickRegisterThatIsAvailableLongest`: by whole instructions,
    /// the hint winning ties.
    fn pick_longest(&self, hint: Option<usize>, until: &[Pos]) -> usize {
        let mut reg = hint.unwrap_or(0);
        let mut cur = until[reg] / 2;
        for (k, &u) in until.iter().enumerate() {
            if u / 2 > cur {
                reg = k;
                cur = u / 2;
            }
        }
        reg
    }

    /// Whether `c` wants a register in `p`'s block before `p`: what a
    /// register held up to an instruction's late point is for.
    fn used_in_block_before(&self, c: u32, p: Pos) -> bool {
        let from = self.block_start(self.block_of_pos(p));
        (self.next_use(c, from, Want::Beneficial)).is_some_and(|u| u.pos < p)
    }

    /// Where a piece holding its register up to `until` (a late point)
    /// can let it go: the gap after its last use wanting it before there.
    fn cut_after_last_use(&self, c: u32, until: Pos) -> Pos {
        let us = self.uses_of(c);
        let k = us.partition_point(|u| u.pos < until);
        us[..k]
            .iter()
            .rev()
            .find(|u| u.need() != super::Need::Any)
            .map_or(until, |u| ((u.pos & !1) + 2).min(until))
    }

    fn needs_reg_at(&self, c: u32, p: Pos) -> bool {
        self.next_use(c, p, Want::Reg).is_some_and(|u| u.pos == p)
    }

    fn assign(&mut self, c: u32, k: usize) {
        let r = self.env.regs[self.class_of(c)][k];
        self.pieces[c as usize].loc = Loc::Reg(r);
    }

    fn process<I: TargetInst>(&mut self, f: &VCode<I>, c: u32) -> Result<(), String> {
        // A value computed again is made where a register first serves it
        // (V8 spills a range with a spill operand until then): holding it
        // before costs a materialization on each path into the stretch
        // that lacks it.
        if self.vregs[self.pieces[c as usize].g as usize].remat {
            let s = self.start(c);
            match self.next_use(c, s, Want::Beneficial) {
                None => {
                    self.spill(c);
                    return Ok(());
                }
                // Already made in a loop's preheader for a use in the loop,
                // and kept around it.
                Some(u) if u.pos > s + 1 && !self.hoisted_to(c, u.pos) => {
                    return self.spill_between(c, s, u.pos);
                }
                Some(_) => {}
            }
            // Nor held into a join a path enters without it.
            if let Some(j) = self.entered_without(f, c) {
                let t = self
                    .split_at(c, j, Why::Join)
                    .expect("a split inside the piece");
                self.queue(t);
            }
        }
        let cls = self.class_of(c);
        let n = self.env.regs[cls].len();
        let (s, e) = (self.start(c), self.end(c));
        let mut a = self.avail(c);
        let hint = self.hint(c);
        if let Some(h) = hint
            && a.free[h] >= e
        {
            self.assign(c, h);
            self.activate(c);
            return Ok(());
        }
        // The hinted register held by pieces that want no register again,
        // or that are made again where they next want one less often than
        // `c` starts: take it from them (they would be spilled by the next
        // clobber; making one there costs less than moving `c` here).
        if let Some(h) = hint {
            let here = self.freq_at(s);
            let idle = |x: &Held| {
                x.cls as usize != cls
                    || x.k as usize != h
                    || self.next_use(x.x, s, Want::Beneficial).is_none_or(|u| {
                        self.vregs[self.pieces[x.x as usize].g as usize].remat
                            && u.pos > s
                            && self.freq_at(u.pos) < here
                    })
            };
            if a.free[h] < e
                && a.fixed[h] >= e
                && self.active.iter().all(idle)
                && self.inactive.iter().all(idle)
            {
                self.assign(c, h);
                self.split_and_spill_intersecting(c)?;
                self.activate(c);
                return Ok(());
            }
        }
        // The fixed register a use wants first, free through that use: hold
        // it to there (an argument made in its register), the rest after.
        if let Some(h) = hint
            && let Some(u) = (self.uses_of(c).iter())
                .find(|u| u.need() == super::Need::Fixed(self.env.regs[cls][h]))
                .copied()
        {
            let cut = (u.pos & !1) + 2;
            if a.free[h] >= cut && cut < e {
                let tail = self
                    .split_at(c, cut, Why::Hold)
                    .expect("a split inside the piece");
                self.queue(tail);
                self.assign(c, h);
                self.activate(c);
                return Ok(());
            }
        }
        let k = self.pick_longest(hint, &a.free[..n]);
        let until = a.free[k];
        // Free only part of the way: losing it costs a reload at the next
        // use after; evicting a holder of a register free longer costs its
        // reload. Take whichever runs less often.
        if until > s && until < e && self.evicting_is_cheaper(c, until, &mut a) {
            self.allocate_blocked(c, &mut a)?;
            if self.reg_of(c).is_some() {
                self.activate(c);
            }
            return Ok(());
        }
        if until > s {
            if until >= e {
                self.assign(c, k);
                self.activate(c);
                return Ok(());
            }
            // Taken at an instruction's late point (a clobber, a def): the
            // register still serves its early uses, the stack the rest.
            if until & 1 != 0 && !self.needs_reg_at(c, until) && self.used_in_block_before(c, until)
            {
                let tail = self
                    .split_at(c, self.cut_after_last_use(c, until), Why::Taken)
                    .expect("a split inside the piece");
                self.assign(c, k);
                self.activate(c);
                return self.spill_tail(tail);
            }
            // Free at the start, taken before the end: split in the gap
            // before it is taken.
            let gap = until & !1;
            if gap > s {
                // A reload nothing reads before a clobber takes every
                // register (one placed at a loop's header, the loop calling
                // before its use): out of a register until that use.
                let p = &self.pieces[c as usize];
                if a.fixed[..n].iter().all(|&x| x <= until)
                    && p.prev != NONE
                    && self.pieces[p.prev as usize].loc.spilled()
                    && !self.is_def(p.g, s)
                    && (self.next_use(c, s, Want::Beneficial)).is_none_or(|u| u.pos >= gap)
                {
                    return self.spill_tail(c);
                }
                let tail = self
                    .split_at(c, gap, Why::Gap)
                    .expect("a split inside the piece");
                self.queue(tail);
                let k = match hint {
                    Some(h) if a.free[h] >= gap => h,
                    _ => k,
                };
                self.assign(c, k);
                self.activate(c);
                return Ok(());
            }
        }
        self.allocate_blocked(c, &mut a)?;
        if self.reg_of(c).is_some() {
            self.activate(c);
        }
        Ok(())
    }

    /// Whether piece `c` starts a loop header whose loop holds position `p`
    /// and a register is free for `c` throughout the loop's hot blocks.
    /// Else, made there, it would be made again on every back edge.
    fn hoisted_to(&self, c: u32, p: Pos) -> bool {
        let s = self.start(c);
        let h = self.block_of_pos(s);
        if !self.is_header(h) || self.block_start(h) != s || !self.in_loop(self.block_of_pos(p), h)
        {
            return false;
        }
        let end = match self.hot_end[h as usize] {
            0 => s,
            e => e,
        };
        let n = self.env.regs[self.class_of(c)].len();
        self.fixed_conflicts(c)[..n].iter().any(|&x| x >= end)
    }

    fn in_loop(&self, b: u32, h: u32) -> bool {
        let mut x = self.inner[b as usize];
        while x != NONE {
            if x == h {
                return true;
            }
            x = self.outer[x as usize];
        }
        false
    }

    /// The first block inside piece `c` that a hot edge from outside it
    /// enters, back edges aside.
    fn entered_without<I: TargetInst>(&self, f: &VCode<I>, c: u32) -> Option<Pos> {
        let (s, e) = (self.start(c), self.end(c));
        for &b in &self.order[self.rank[self.block_of_pos(s) as usize] as usize + 1..] {
            let j = b.index() as u32;
            if self.block_start(j) >= e {
                break;
            }
            let outside = f.block_preds(b).iter().any(|p| {
                let p = p.index() as u32;
                let end = self.block_end(p) - 1;
                end < self.cold_start && !(s..e).contains(&end) && !self.in_loop(p, j)
            });
            if outside {
                return Some(self.block_start(j));
            }
        }
        None
    }

    fn evicting_is_cheaper(&self, c: u32, until: Pos, a: &mut Avail) -> bool {
        let Some(after) = self.next_use(c, until, Want::Beneficial) else {
            return false;
        };
        let n = self.env.regs[self.class_of(c)].len();
        let use_pos = self.use_pos(c, a);
        let k = self.pick_longest(self.hint(c), &use_pos[..n]);
        if use_pos[k] <= until || a.fixed[k] <= until {
            return false;
        }
        let evict = if use_pos[k] == Pos::MAX {
            0.0
        } else {
            self.freq_at(use_pos[k])
        };
        evict < self.freq_at(after.pos)
    }

    fn allocate_blocked(&mut self, c: u32, a: &mut Avail) -> Result<(), String> {
        let (s, e) = (self.start(c), self.end(c));
        let Some(ru) = self.next_use(c, s, Want::Reg) else {
            // Nothing here needs a register.
            let (begin, at) = self.optimal_spilling_pos(c, s);
            self.maybe_spill_previous(begin, at, c);
            self.spill(c);
            return Ok(());
        };
        let n = self.env.regs[self.class_of(c)].len();
        let use_pos = self.use_pos(c, a);
        let k = self.pick_longest(self.hint(c), &use_pos[..n]);
        // Everyone needs their register before this piece does: spill it
        // until its use, if a gap before the use can take the reload.
        let g = self.pieces[c as usize].g;
        if use_pos[k] < ru.pos && self.reload_room(g, s, ru.pos) {
            return self.spill_between(c, s, ru.pos);
        }
        let mut new_end = e.min(a.fixed[k]);
        if new_end < e
            && new_end & 1 != 0
            && new_end > s
            && ru.pos < new_end
            && !self.needs_reg_at(c, new_end)
            && self.used_in_block_before(c, new_end)
        {
            // Blocked at a late point after its register use: hold the
            // register to its last use before there, the stack after.
            let t = self
                .split_at(c, self.cut_after_last_use(c, new_end), Why::Blocked)
                .expect("a split inside the piece");
            self.assign(c, k);
            self.split_and_spill_intersecting(c)?;
            return self.spill_tail(t);
        }
        if new_end < e {
            new_end &= !1;
        }
        if new_end <= s {
            if self.reload_room(g, s, ru.pos) {
                return self.spill_between(c, s, ru.pos);
            }
            return Err(format!("no register for v{g} at {s}"));
        }
        if new_end < e {
            let mut t = self.split_between(c, s, new_end, Why::Blocked);
            if t == Some(c) {
                t = self.split_at(c, new_end, Why::Blocked);
            }
            let t = t.expect("a split inside the piece");
            self.queue(t);
        }
        self.assign(c, k);
        self.split_and_spill_intersecting(c)
    }

    /// Evict the pieces holding `c`'s register where `c` needs it: spill them
    /// until their next register use.
    fn split_and_spill_intersecting(&mut self, c: u32) -> Result<(), String> {
        let r = self.reg_of(c).expect("a register");
        let (cls, reg) = (self.pieces[c as usize].class, self.reg_idx[r.index()] as u8);
        let mine = |h: &Held| h.cls == cls && h.k == reg;
        let s = self.start(c);
        let mut k = 0;
        while k < self.active.len() {
            let a = self.active[k].x;
            if !mine(&self.active[k]) {
                k += 1;
                continue;
            }
            self.active.swap_remove(k);
            let next = self.next_use(a, s, Want::Reg);
            let (begin, at) = self.optimal_spilling_pos(a, s);
            self.maybe_spill_previous(begin, at, a);
            match next {
                None => self.spill_after(a, at, Why::Evicted(s)),
                Some(u) if u.pos > s => {
                    self.spill_between_until(a, at, s, u.pos, Why::Evicted(s))?
                }
                Some(_) => {
                    return Err(format!(
                        "v{} and v{} need one register at {s}",
                        self.pieces[a as usize].g, self.pieces[c as usize].g
                    ));
                }
            }
        }
        let mut k = 0;
        while k < self.inactive.len() {
            let i = self.inactive[k].x;
            let hit = (mine(&self.inactive[k]))
                .then(|| intersect(self.ivs_of(i), self.ivs_of(c), s))
                .flatten();
            let Some(x) = hit else {
                k += 1;
                continue;
            };
            self.inactive.swap_remove(k);
            match self.next_use(i, s, Want::Reg) {
                None => self.spill_after(i, s, Why::Evicted(s)),
                Some(u) => self.spill_between_until(i, s, s, x.min(u.pos), Why::Evicted(s))?,
            }
        }
        Ok(())
    }

    /// Split `x` at `pos`: `None` when nothing of it lies at or after `pos`,
    /// `x` itself when all of it does, else the new piece from `pos`.
    pub(super) fn split_at(&mut self, x: u32, pos: Pos, why: Why) -> Option<u32> {
        #[cfg(not(feature = "tcvm-dev"))]
        let _ = why;
        let p = self.pieces[x as usize];
        debug_assert!(
            !(self.active.iter().chain(&self.inactive)).any(|h| h.x == x),
            "splitting a listed piece"
        );
        if pos >= p.end {
            return None;
        }
        if pos <= p.start {
            return Some(x);
        }
        let iv = p.ivs(&self.ivs);
        let k = iv.after(pos);
        // Inside interval `k`, the two share it; in the hole before it, the
        // head ends where `k - 1` does.
        let (head_hi, head_end, tail_start) = if iv.get(k).0 < pos {
            (k + 1, pos, pos)
        } else {
            (k, iv.get(k - 1).1, iv.get(k).0)
        };
        let u = p.uses(&self.uses).partition_point(|u| u.pos < pos) as u32;
        let n = self.pieces.len() as u32;
        self.pieces.push(Piece {
            start: tail_start,
            lo: p.lo + k as u32,
            ulo: p.ulo + u,
            prev: x,
            loc: Loc::None,
            flags: 0,
            #[cfg(feature = "tcvm-dev")]
            why,
            ..p
        });
        if p.next != NONE {
            self.pieces[p.next as usize].prev = n;
        }
        let h = &mut self.pieces[x as usize];
        h.hi = p.lo + head_hi as u32;
        h.end = head_end;
        h.uhi = p.ulo + u;
        h.next = n;
        Some(n)
    }

    /// Undo the split of `x` that made `y`, its next piece.
    fn merge_back(&mut self, x: u32, y: u32) {
        let t = self.pieces[y as usize];
        debug_assert_eq!(self.pieces[x as usize].next, y);
        let p = &mut self.pieces[x as usize];
        p.hi = t.hi;
        p.end = t.end;
        p.uhi = t.uhi;
        p.next = t.next;
        if t.next != NONE {
            self.pieces[t.next as usize].prev = x;
        }
        let d = &mut self.pieces[y as usize];
        d.lo = d.hi;
        d.prev = NONE;
        d.next = NONE;
    }

    fn split_between(&mut self, x: u32, start: Pos, end: Pos, why: Why) -> Option<u32> {
        let at = self.optimal_split_pos(start, end);
        self.split_at(x, at, why)
    }

    /// V8's `FindOptimalSplitPos`: across blocks, at the header of the
    /// outermost loop entered after `start`, else as late as possible.
    fn optimal_split_pos(&self, start: Pos, end: Pos) -> Pos {
        if start / 2 == end / 2 {
            return end;
        }
        let (sb, eb) = (self.block_of_pos(start), self.block_of_pos(end));
        if sb == eb {
            return end;
        }
        // Headers inside (start, end] only: cold blocks laid out last can
        // put a loop's header after blocks of its body.
        let mut b = eb;
        loop {
            let h = self.containing_loop(b);
            if h == NONE
                || self.rank[h as usize] <= self.rank[sb as usize]
                || !(start + 1..=end).contains(&self.block_start(h))
            {
                break;
            }
            b = h;
        }
        if b == eb && !self.is_header(eb) {
            return end;
        }
        self.block_start(b)
    }

    /// V8's `FindOptimalSpillingPos`: a spill inside a loop moves to the
    /// header when nothing in the loop before it needs the register, outward
    /// through enclosing loops. Returns the piece live there with it.
    fn optimal_spilling_pos(&self, x: u32, pos: Pos) -> (u32, Pos) {
        let v = self.pieces[x as usize].g;
        let first = self.start(self.vregs[v as usize].head);
        let (mut begin, mut pos) = (x, pos);
        // Spilled in cold code: it stays there.
        if pos >= self.cold_start {
            return (begin, pos);
        }
        let b = self.block_of_pos(pos);
        let mut h = if self.is_header(b) {
            b
        } else {
            self.containing_loop(b)
        };
        while h != NONE {
            let ls = self.block_start(h);
            // Defined at or after the header: spilling there buys nothing;
            // a header laid out after the spill is no earlier place.
            if first >= ls || ls > pos {
                return (begin, pos);
            }
            if let Some(live) = self.piece_at(v, ls)
                && !self.pieces[live as usize].loc.spilled()
            {
                let mut y = live;
                while y != NONE && self.start(y) < pos {
                    if (self.next_use(y, ls, Want::Beneficial)).is_some_and(|u| u.pos <= pos) {
                        return (begin, pos);
                    }
                    y = self.pieces[y as usize].next;
                }
                begin = live;
                pos = ls;
            }
            h = self.containing_loop(h);
        }
        (begin, pos)
    }

    /// V8's `MaybeSpillPreviousRanges`: spill `begin` from `pos` and every
    /// piece after it up to `end`.
    fn maybe_spill_previous(&mut self, begin: u32, pos: Pos, end: u32) {
        if begin == end {
            return;
        }
        let keep = |p: &Piece| p.loc.spilled() || p.flags & super::FIXED != 0;
        if !keep(&self.pieces[begin as usize]) {
            self.spill_after(begin, pos, Why::Spilled);
        }
        let mut y = self.pieces[begin as usize].next;
        while y != NONE && y != end {
            if !keep(&self.pieces[y as usize]) {
                self.spill(y);
            }
            y = self.pieces[y as usize].next;
        }
    }

    fn spill(&mut self, x: u32) {
        debug_assert!(
            self.next_use(x, 0, Want::Reg).is_none(),
            "spilling v{} over a register use",
            self.pieces[x as usize].g
        );
        let v = self.pieces[x as usize].g;
        if self.vregs[v as usize].remat {
            self.pieces[x as usize].loc = Loc::Remat;
        } else {
            self.pieces[x as usize].loc = Loc::Stack;
            self.slot_of(v);
        }
        self.active.retain(|h| h.x != x);
        self.inactive.retain(|h| h.x != x);
    }

    /// Spill `t`, a piece starting at a late point, until its next register
    /// use, reloaded in the best gap before it.
    fn spill_tail(&mut self, t: u32) -> Result<(), String> {
        let (s, g) = (self.start(t), self.pieces[t as usize].g);
        match self.next_use(t, s, Want::Reg) {
            None => {
                self.spill(t);
                Ok(())
            }
            Some(u) if u.pos > s => self.spill_between(t, s, u.pos),
            // A member defined right there: a piece of its own.
            Some(_) if self.is_def(g, s) => {
                self.queue(t);
                Ok(())
            }
            Some(u) => Err(format!(
                "v{g} needs a register at {} where none is free",
                u.pos
            )),
        }
    }

    fn spill_after(&mut self, x: u32, pos: Pos, why: Why) {
        if let Some(t) = self.split_at(x, pos, why) {
            self.spill(t);
        }
    }

    fn spill_between(&mut self, x: u32, start: Pos, end: Pos) -> Result<(), String> {
        self.spill_between_until(x, start, start, end, Why::Spilled)
    }

    /// V8's `SpillBetweenUntil`: spill `x` from `start` (`why`), staying
    /// spilled until `until` at least, and reload it for `end` (a register
    /// use) in the best gap before it.
    fn spill_between_until(
        &mut self,
        x: u32,
        start: Pos,
        until: Pos,
        end: Pos,
        why: Why,
    ) -> Result<(), String> {
        debug_assert!(start < end);
        let Some(second) = self.split_at(x, start, why) else {
            return Ok(());
        };
        let s2 = self.start(second);
        if s2 >= end {
            self.queue(second);
            return Ok(());
        }
        // A member defined at `end` starts its own piece, with no reload.
        let g = self.pieces[second as usize].g;
        if self.is_def(g, end) && end >= until {
            let third = self
                .split_at(second, end, Why::Reload)
                .expect("a split inside the piece");
            self.queue(third);
            if third != second {
                self.spill(second);
            }
            return Ok(());
        }
        let split_start = (s2 + 1).max(until);
        let mut third_end = split_start.max(end & !1);
        if end == self.block_start(self.block_of_pos(end)) {
            third_end = split_start.max(end);
        }
        let third = match self.split_between(second, split_start, third_end, Why::Reload) {
            Some(t) => t,
            None => {
                self.spill(second);
                return Ok(());
            }
        };
        let t = self.start(third);
        if t & 1 != 0 && !self.is_def(g, t) {
            // No gap to reload in before `end`: a piece not yet reached goes
            // back unallocated instead.
            if s2 >= until {
                if third != second {
                    self.merge_back(second, third);
                }
                self.queue(second);
                return Ok(());
            }
            return Err(format!("no gap to reload v{g} before {end}"));
        }
        self.queue(third);
        if third != second {
            self.spill(second);
        }
        Ok(())
    }
}

/// `list.partition_point(before)` from where `at` left it: forward from
/// there while the point grows, by search when it went back.
fn cursor<T>(at: &std::cell::Cell<u32>, list: &[T], before: impl Fn(&T) -> bool) -> usize {
    let mut k = at.get() as usize;
    if k > list.len() || k > 0 && !before(&list[k - 1]) {
        k = list.partition_point(&before);
    } else {
        while k < list.len() && before(&list[k]) {
            k += 1;
        }
    }
    at.set(k as u32);
    k
}
