//! The scan: pieces in start order take a free register, evict a holder
//! whose next register use comes later, or spill until they need one
//! (V8's `LinearScanAllocator`, `register-allocator.cc`).

use std::cmp::Reverse;

use super::{Loc, NONE, Piece, Pos, Ra, intersect, needs_reg, reg_beneficial};
use crate::jit::backend::reg::{Block, PReg};
use crate::jit::backend::vcode::TargetInst;

impl<I: TargetInst> Ra<'_, I> {
    pub(super) fn scan(&mut self) -> Result<(), String> {
        // Every step moves forward, so a function of `n` instructions takes
        // far fewer; past this the scan is going round in circles.
        let mut budget = 64 * self.at_lin.len() + 10_000;
        while let Some(Reverse((s, c))) = self.unhandled.pop() {
            budget = budget
                .checked_sub(1)
                .ok_or_else(|| format!("the scan made no progress at {s}"))?;
            let p = &self.pieces[c as usize];
            // Spilled while queued (`maybe_spill_previous`) or split since.
            if p.loc != Loc::None || p.start() != s {
                if p.loc == Loc::None {
                    self.queue(c);
                }
                continue;
            }
            self.advance(s);
            // A piece starting at a late point (not a definition) holds its
            // value where the cut left it: the stack, until a gap can reload.
            if s & 1 != 0 && !self.is_def(self.pieces[c as usize].vreg, s) {
                self.spill_tail(c)?;
                continue;
            }
            self.process(c)?;
        }
        Ok(())
    }

    fn queue(&mut self, x: u32) {
        let s = self.pieces[x as usize].start();
        self.unhandled.push(Reverse((s, x)));
    }

    /// Retire pieces that end by `pos`; move the rest between active (they
    /// cover `pos`) and inactive (a hole there).
    fn advance(&mut self, pos: Pos) {
        let pieces = &self.pieces;
        let mut back: Vec<u32> = Vec::new();
        self.active.retain(|&a| {
            let p = &pieces[a as usize];
            if p.end() <= pos {
                false
            } else if !p.covers(pos) {
                back.push(a);
                false
            } else {
                true
            }
        });
        let mut fwd: Vec<u32> = Vec::new();
        self.inactive.retain(|&i| {
            let p = &pieces[i as usize];
            if p.end() <= pos {
                false
            } else if p.covers(pos) {
                fwd.push(i);
                false
            } else {
                true
            }
        });
        self.active.extend(fwd);
        self.inactive.extend(back);
    }

    fn class_of(&self, x: u32) -> usize {
        self.vregs[self.pieces[x as usize].vreg as usize].class as usize
    }

    fn reg_of(&self, x: u32) -> Option<PReg> {
        match self.pieces[x as usize].loc {
            Loc::Reg(r) => Some(r),
            _ => None,
        }
    }

    /// The first point of `c` a reservation of `r` not its own takes.
    fn fixed_conflict(&self, r: PReg, c: u32) -> Option<Pos> {
        let p = &self.pieces[c as usize];
        let (s, e) = (p.start(), p.end());
        let list = &self.fixed[r.index()];
        // Reservations span at most two points.
        let mut k = list.partition_point(|x| x.0 + 2 <= s);
        let mut best: Option<Pos> = None;
        while k < list.len() && list[k].0 < e && best.is_none_or(|b| list[k].0 < b) {
            let (a, b, o) = list[k];
            if o != p.vreg
                && let Some(x) = intersect(&[(a, b)], &p.ivs, s)
            {
                best = Some(best.map_or(x, |y| y.min(x)));
            }
            k += 1;
        }
        best
    }

    /// The register `c` would like: that of its first fixed use, else its
    /// previous piece's.
    fn hint(&self, c: u32) -> Option<usize> {
        let p = &self.pieces[c as usize];
        let fixed = p.uses.iter().find_map(|u| match u.need {
            super::Need::Fixed(r) => Some(r),
            _ => None,
        });
        let prev = || {
            let ps = &self.vregs[p.vreg as usize].pieces;
            let k = ps.iter().position(|&x| x == c)?;
            (k > 0).then(|| self.reg_of(ps[k - 1])).flatten()
        };
        let r = fixed.or_else(prev)?;
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
        self.pieces[c as usize]
            .next_use(from, reg_beneficial)
            .is_some_and(|u| u.pos < p)
    }

    /// Where a piece holding its register up to `until` (a late point)
    /// can let it go: the gap after its last use wanting it before there.
    fn cut_after_last_use(&self, c: u32, until: Pos) -> Pos {
        let p = &self.pieces[c as usize];
        let k = p.uses.partition_point(|u| u.pos < until);
        p.uses[..k]
            .iter()
            .rev()
            .find(|u| reg_beneficial(u.need))
            .map_or(until, |u| ((u.pos & !1) + 2).min(until))
    }

    fn needs_reg_at(&self, c: u32, p: Pos) -> bool {
        self.pieces[c as usize]
            .next_use(p, needs_reg)
            .is_some_and(|u| u.pos == p)
    }

    fn assign(&mut self, c: u32, k: usize) {
        let r = self.env.regs[self.class_of(c)][k];
        self.pieces[c as usize].loc = Loc::Reg(r);
    }

    fn process(&mut self, c: u32) -> Result<(), String> {
        // A value computed again is made where a register first serves it
        // (V8 spills a range with a spill operand until then): holding it
        // before costs a materialization on each path into the stretch
        // that lacks it.
        if self.vregs[self.pieces[c as usize].vreg as usize].remat {
            let s = self.pieces[c as usize].start();
            match self.pieces[c as usize].next_use(s, reg_beneficial) {
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
            if let Some(j) = self.entered_without(c) {
                let t = self.split_at(c, j).expect("a split inside the piece");
                self.queue(t);
            }
        }
        let cls = self.class_of(c);
        let n = self.env.regs[cls].len();
        let (s, e) = {
            let p = &self.pieces[c as usize];
            (p.start(), p.end())
        };
        let mut free = vec![Pos::MAX; n];
        for &a in &self.active {
            if let Some(r) = self.reg_of(a)
                && self.class_of(a) == cls
            {
                free[self.reg_idx[r.index()] as usize] = 0;
            }
        }
        for &i in &self.inactive {
            if let Some(r) = self.reg_of(i)
                && self.class_of(i) == cls
                && let Some(x) = intersect(
                    &self.pieces[i as usize].ivs,
                    &self.pieces[c as usize].ivs,
                    s,
                )
            {
                let k = self.reg_idx[r.index()] as usize;
                free[k] = free[k].min(x);
            }
        }
        for (k, fk) in free.iter_mut().enumerate() {
            if let Some(x) = self.fixed_conflict(self.env.regs[cls][k], c) {
                *fk = (*fk).min(x);
            }
        }
        let hint = self.hint(c);
        if let Some(h) = hint
            && free[h] >= e
        {
            self.assign(c, h);
            self.active.push(c);
            return Ok(());
        }
        // The hinted register held by pieces that want no register again:
        // take it from them (they would be spilled by the next clobber).
        if let Some(h) = hint {
            let r = self.env.regs[cls][h];
            let idle = |x: &u32| {
                self.reg_of(*x) != Some(r)
                    || self.pieces[*x as usize]
                        .next_use(s, reg_beneficial)
                        .is_none()
            };
            let (_, block_pos) = self.blocked_pos(c);
            if free[h] < e
                && block_pos[h] >= e
                && self.active.iter().all(idle)
                && self.inactive.iter().all(idle)
            {
                self.assign(c, h);
                self.split_and_spill_intersecting(c)?;
                self.active.push(c);
                return Ok(());
            }
        }
        // The fixed register a use wants first, free through that use: hold
        // it to there (an argument made in its register), the rest after.
        if let Some(h) = hint
            && let Some(u) = (self.pieces[c as usize].uses.iter())
                .find(|u| u.need == super::Need::Fixed(self.env.regs[cls][h]))
                .copied()
        {
            let cut = (u.pos & !1) + 2;
            if free[h] >= cut && cut < e {
                let tail = self.split_at(c, cut).expect("a split inside the piece");
                self.queue(tail);
                self.assign(c, h);
                self.active.push(c);
                return Ok(());
            }
        }
        let k = self.pick_longest(hint, &free);
        let until = free[k];
        // Free only part of the way: losing it costs a reload at the next
        // use after; evicting a holder of a register free longer costs its
        // reload. Take whichever runs less often.
        if until > s && until < e && self.evicting_is_cheaper(c, until) {
            self.allocate_blocked(c)?;
            if self.reg_of(c).is_some() {
                self.active.push(c);
            }
            return Ok(());
        }
        if until > s {
            if until >= e {
                self.assign(c, k);
                self.active.push(c);
                return Ok(());
            }
            // Taken at an instruction's late point (a clobber, a def): the
            // register still serves its early uses, the stack the rest.
            if until & 1 != 0 && !self.needs_reg_at(c, until) && self.used_in_block_before(c, until)
            {
                let tail = self
                    .split_at(c, self.cut_after_last_use(c, until))
                    .expect("a split inside the piece");
                self.assign(c, k);
                self.active.push(c);
                return self.spill_tail(tail);
            }
            // Free at the start, taken before the end: split in the gap
            // before it is taken.
            let gap = until & !1;
            if gap > s {
                let tail = self.split_at(c, gap).expect("a split inside the piece");
                self.queue(tail);
                let k = match hint {
                    Some(h) if free[h] >= gap => h,
                    _ => k,
                };
                self.assign(c, k);
                self.active.push(c);
                return Ok(());
            }
        }
        self.allocate_blocked(c)?;
        if self.reg_of(c).is_some() {
            self.active.push(c);
        }
        Ok(())
    }

    /// Whether piece `c` starts a loop header whose loop holds position `p`
    /// and a register is free for `c` throughout the loop's hot blocks.
    /// Else, made there, it would be made again on every back edge.
    fn hoisted_to(&self, c: u32, p: Pos) -> bool {
        let s = self.pieces[c as usize].start();
        let h = self.block_of_pos(s);
        if !self.is_header(h) || self.block_start(h) != s || !self.in_loop(self.block_of_pos(p), h)
        {
            return false;
        }
        let end = (0..self.first.len() as u32)
            .filter(|&b| self.in_loop(b, h) && self.block_start(b) < self.cold_start)
            .map(|b| self.block_end(b))
            .max()
            .unwrap_or(s);
        self.env.regs[self.class_of(c)]
            .iter()
            .any(|&r| self.fixed_conflict(r, c).is_none_or(|x| x >= end))
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
    fn entered_without(&self, c: u32) -> Option<Pos> {
        let (s, e) = {
            let p = &self.pieces[c as usize];
            (p.start(), p.end())
        };
        let mut i = s / 2 + 1;
        while 2 * i < e {
            let j = self.lin_block[i as usize];
            if self.first[j as usize] == i {
                let outside = self.f.block_preds(Block::new(j as usize)).iter().any(|p| {
                    let p = p.index() as u32;
                    let end = self.block_end(p) - 1;
                    end < self.cold_start && !(s..e).contains(&end) && !self.in_loop(p, j)
                });
                if outside {
                    return Some(2 * i);
                }
            }
            i += 1;
        }
        None
    }

    /// Per register of `c`'s class: where its holders next want it
    /// (V8's `use_pos`), and where a fixed reservation takes it
    /// (`block_pos`).
    fn blocked_pos(&self, c: u32) -> (Vec<Pos>, Vec<Pos>) {
        let cls = self.class_of(c);
        let n = self.env.regs[cls].len();
        let s = self.pieces[c as usize].start();
        let mut use_pos = vec![Pos::MAX; n];
        let mut block_pos = vec![Pos::MAX; n];
        for &a in &self.active {
            if let Some(r) = self.reg_of(a)
                && self.class_of(a) == cls
            {
                let k = self.reg_idx[r.index()] as usize;
                let next = self.pieces[a as usize]
                    .next_use(s, reg_beneficial)
                    .map_or(Pos::MAX, |u| u.pos);
                use_pos[k] = use_pos[k].min(next);
            }
        }
        for &i in &self.inactive {
            if let Some(r) = self.reg_of(i)
                && self.class_of(i) == cls
                && let Some(x) = intersect(
                    &self.pieces[i as usize].ivs,
                    &self.pieces[c as usize].ivs,
                    s,
                )
            {
                let k = self.reg_idx[r.index()] as usize;
                use_pos[k] = use_pos[k].min(x);
            }
        }
        for k in 0..n {
            if let Some(x) = self.fixed_conflict(self.env.regs[cls][k], c) {
                block_pos[k] = block_pos[k].min(x);
                use_pos[k] = use_pos[k].min(block_pos[k]);
            }
        }
        (use_pos, block_pos)
    }

    fn evicting_is_cheaper(&self, c: u32, until: Pos) -> bool {
        let Some(after) = self.pieces[c as usize].next_use(until, reg_beneficial) else {
            return false;
        };
        let (use_pos, block_pos) = self.blocked_pos(c);
        let k = self.pick_longest(self.hint(c), &use_pos);
        if use_pos[k] <= until || block_pos[k] <= until {
            return false;
        }
        let evict = if use_pos[k] == Pos::MAX {
            0.0
        } else {
            self.freq_at(use_pos[k])
        };
        evict < self.freq_at(after.pos)
    }

    fn allocate_blocked(&mut self, c: u32) -> Result<(), String> {
        let (s, e) = {
            let p = &self.pieces[c as usize];
            (p.start(), p.end())
        };
        let Some(ru) = self.pieces[c as usize].next_use(s, needs_reg) else {
            // Nothing here needs a register.
            let (begin, at) = self.optimal_spilling_pos(c, s);
            self.maybe_spill_previous(begin, at, c);
            self.spill(c);
            return Ok(());
        };
        let (use_pos, block_pos) = self.blocked_pos(c);
        let k = self.pick_longest(self.hint(c), &use_pos);
        // Everyone needs their register before this piece does: spill it
        // until its use, if a gap before the use can take the reload.
        let g = self.pieces[c as usize].vreg;
        if use_pos[k] < ru.pos && self.reload_room(g, s, ru.pos) {
            return self.spill_between(c, s, ru.pos);
        }
        let mut new_end = e.min(block_pos[k]);
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
                .split_at(c, self.cut_after_last_use(c, new_end))
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
            return Err(format!(
                "no register for v{} at {s}",
                self.pieces[c as usize].vreg
            ));
        }
        if new_end < e {
            let mut t = self.split_between(c, s, new_end);
            if t == Some(c) {
                t = self.split_at(c, new_end);
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
        let s = self.pieces[c as usize].start();
        let mut k = 0;
        while k < self.active.len() {
            let a = self.active[k];
            if self.reg_of(a) != Some(r) {
                k += 1;
                continue;
            }
            self.active.swap_remove(k);
            let next = self.pieces[a as usize].next_use(s, needs_reg);
            let (begin, at) = self.optimal_spilling_pos(a, s);
            self.maybe_spill_previous(begin, at, a);
            match next {
                None => self.spill_after(a, at),
                Some(u) if u.pos > s => self.spill_between_until(a, at, s, u.pos)?,
                Some(_) => {
                    return Err(format!(
                        "v{} and v{} need one register at {s}",
                        self.pieces[a as usize].vreg, self.pieces[c as usize].vreg
                    ));
                }
            }
        }
        let mut k = 0;
        while k < self.inactive.len() {
            let i = self.inactive[k];
            let hit = (self.reg_of(i) == Some(r))
                .then(|| {
                    intersect(
                        &self.pieces[i as usize].ivs,
                        &self.pieces[c as usize].ivs,
                        s,
                    )
                })
                .flatten();
            let Some(x) = hit else {
                k += 1;
                continue;
            };
            self.inactive.swap_remove(k);
            match self.pieces[i as usize].next_use(s, needs_reg) {
                None => self.spill_after(i, s),
                Some(u) => self.spill_between(i, s, x.min(u.pos))?,
            }
        }
        Ok(())
    }

    /// Split `x` at `pos`: `None` when nothing of it lies at or after `pos`,
    /// `x` itself when all of it does, else the new piece from `pos`.
    pub(super) fn split_at(&mut self, x: u32, pos: Pos) -> Option<u32> {
        let p = &mut self.pieces[x as usize];
        if pos >= p.end() {
            return None;
        }
        if pos <= p.start() {
            return Some(x);
        }
        let k = p.ivs.partition_point(|iv| iv.1 <= pos);
        let mut tail: Vec<(Pos, Pos)> = Vec::with_capacity(p.ivs.len() - k);
        if p.ivs[k].0 < pos {
            tail.push((pos, p.ivs[k].1));
            tail.extend_from_slice(&p.ivs[k + 1..]);
            p.ivs[k].1 = pos;
            p.ivs.truncate(k + 1);
        } else {
            tail.extend_from_slice(&p.ivs[k..]);
            p.ivs.truncate(k);
        }
        let u = p.uses.partition_point(|u| u.pos < pos);
        let uses = p.uses.split_off(u);
        let v = p.vreg;
        let n = self.pieces.len() as u32;
        self.pieces.push(Piece {
            vreg: v,
            ivs: tail,
            uses,
            loc: Loc::None,
            fixed: false,
        });
        let ps = &mut self.vregs[v as usize].pieces;
        let at = ps.iter().position(|&y| y == x).expect("the piece's value") + 1;
        ps.insert(at, n);
        Some(n)
    }

    /// Undo the split of `x` that made `y`, its next piece.
    fn merge_back(&mut self, x: u32, y: u32) {
        let tail = std::mem::take(&mut self.pieces[y as usize]);
        let p = &mut self.pieces[x as usize];
        if p.end() == tail.ivs[0].0 {
            p.ivs.last_mut().expect("an interval").1 = tail.ivs[0].1;
            p.ivs.extend_from_slice(&tail.ivs[1..]);
        } else {
            p.ivs.extend_from_slice(&tail.ivs);
        }
        p.uses.extend_from_slice(&tail.uses);
        let ps = &mut self.vregs[p.vreg as usize].pieces;
        ps.retain(|&z| z != y);
    }

    fn split_between(&mut self, x: u32, start: Pos, end: Pos) -> Option<u32> {
        let at = self.optimal_split_pos(start, end);
        self.split_at(x, at)
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
        let v = self.pieces[x as usize].vreg;
        let first = self.pieces[self.vregs[v as usize].pieces[0] as usize].start();
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
                let ps = &self.vregs[v as usize].pieces;
                let from = ps.iter().position(|&y| y == live).expect("a piece");
                for &y in &ps[from..] {
                    let p = &self.pieces[y as usize];
                    if p.start() >= pos {
                        break;
                    }
                    if p.next_use(ls, reg_beneficial).is_some_and(|u| u.pos <= pos) {
                        return (begin, pos);
                    }
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
        let v = self.pieces[begin as usize].vreg;
        if !self.pieces[begin as usize].loc.spilled() && !self.pieces[begin as usize].fixed {
            self.spill_after(begin, pos);
        }
        let ps = self.vregs[v as usize].pieces.clone();
        let from = ps.iter().position(|&y| y == begin).expect("a piece") + 1;
        for &y in &ps[from..] {
            if y == end {
                break;
            }
            if !self.pieces[y as usize].loc.spilled() && !self.pieces[y as usize].fixed {
                self.spill(y);
            }
        }
    }

    fn spill(&mut self, x: u32) {
        debug_assert!(
            self.pieces[x as usize].next_use(0, needs_reg).is_none(),
            "spilling v{} over a register use",
            self.pieces[x as usize].vreg
        );
        let v = self.pieces[x as usize].vreg;
        if self.vregs[v as usize].remat {
            self.pieces[x as usize].loc = Loc::Remat;
        } else {
            self.pieces[x as usize].loc = Loc::Stack;
            self.slot_of(v);
        }
        self.active.retain(|&a| a != x);
        self.inactive.retain(|&a| a != x);
    }

    /// Spill `t`, a piece starting at a late point, until its next register
    /// use, reloaded in the best gap before it.
    fn spill_tail(&mut self, t: u32) -> Result<(), String> {
        let p = &self.pieces[t as usize];
        match p.next_use(p.start(), needs_reg) {
            None => {
                self.spill(t);
                Ok(())
            }
            Some(u) if u.pos > p.start() => self.spill_between(t, p.start(), u.pos),
            // A member defined right there: a piece of its own.
            Some(_) if self.is_def(p.vreg, p.start()) => {
                self.queue(t);
                Ok(())
            }
            Some(u) => Err(format!(
                "v{} needs a register at {} where none is free",
                p.vreg, u.pos
            )),
        }
    }

    fn spill_after(&mut self, x: u32, pos: Pos) {
        if let Some(t) = self.split_at(x, pos) {
            self.spill(t);
        }
    }

    fn spill_between(&mut self, x: u32, start: Pos, end: Pos) -> Result<(), String> {
        self.spill_between_until(x, start, start, end)
    }

    /// V8's `SpillBetweenUntil`: spill `x` from `start`, staying spilled
    /// until `until` at least, and reload it for `end` (a register use) in
    /// the best gap before it.
    fn spill_between_until(
        &mut self,
        x: u32,
        start: Pos,
        until: Pos,
        end: Pos,
    ) -> Result<(), String> {
        debug_assert!(start < end);
        let Some(second) = self.split_at(x, start) else {
            return Ok(());
        };
        let s2 = self.pieces[second as usize].start();
        if s2 >= end {
            self.queue(second);
            return Ok(());
        }
        // A member defined at `end` starts its own piece, with no reload.
        let g = self.pieces[second as usize].vreg;
        if self.is_def(g, end) && end >= until {
            let third = self
                .split_at(second, end)
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
        let third = match self.split_between(second, split_start, third_end) {
            Some(t) => t,
            None => {
                self.spill(second);
                return Ok(());
            }
        };
        let t = self.pieces[third as usize].start();
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
            return Err(format!(
                "no gap to reload v{} before {end}",
                self.pieces[x as usize].vreg
            ));
        }
        self.queue(third);
        if third != second {
            self.spill(second);
        }
        Ok(())
    }
}
