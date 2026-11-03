//! The scan: pieces in start order take a free register, evict a holder
//! whose next register use comes later, or spill until they need one
//! (V8's `LinearScanAllocator`, `register-allocator.cc`).

use std::cmp::Reverse;

use regalloc2::PReg;

use super::{Loc, NONE, Piece, Pos, Ra, intersect, needs_reg, reg_beneficial};
use crate::jit::backend::vcode::TargetInst;

impl<I: TargetInst> Ra<'_, I> {
    pub(super) fn scan(&mut self) -> Result<(), String> {
        while let Some(Reverse((s, c))) = self.unhandled.pop() {
            let p = &self.pieces[c as usize];
            // Spilled while queued (`maybe_spill_previous`) or split since.
            if p.loc != Loc::None || p.start() != s {
                if p.loc == Loc::None {
                    self.queue(c);
                }
                continue;
            }
            self.advance(s);
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

    /// The register `c` would like: that of the fixed use it first benefits
    /// from, else its previous piece's.
    fn hint(&self, c: u32) -> Option<usize> {
        let p = &self.pieces[c as usize];
        let fixed = match p.next_use(p.start(), reg_beneficial) {
            Some(u) => match u.need {
                super::Need::Fixed(r) => Some(r),
                _ => None,
            },
            None => None,
        };
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

    fn assign(&mut self, c: u32, k: usize) {
        let r = self.env.regs[self.class_of(c)][k];
        self.pieces[c as usize].loc = Loc::Reg(r);
    }

    fn process(&mut self, c: u32) -> Result<(), String> {
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
        let k = self.pick_longest(hint, &free);
        let until = free[k];
        if until > s {
            if until >= e {
                self.assign(c, k);
                self.active.push(c);
                return Ok(());
            }
            // Taken at an instruction's late point (a clobber, a def): the
            // register still serves its early uses, the stack the rest.
            if until & 1 != 0 {
                let tail = self.split_at(c, until).expect("a split inside the piece");
                self.assign(c, k);
                self.active.push(c);
                return self.spill_tail(tail);
            }
            // Free at the start, taken before the end: split in the gap
            // before it is taken.
            if until > s {
                let tail = self.split_at(c, until).expect("a split inside the piece");
                self.queue(tail);
                let k = match hint {
                    Some(h) if free[h] >= until => h,
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

    fn allocate_blocked(&mut self, c: u32) -> Result<(), String> {
        let cls = self.class_of(c);
        let n = self.env.regs[cls].len();
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
        let k = self.pick_longest(self.hint(c), &use_pos);
        // Everyone needs their register before this piece does: spill it
        // until its use, if a gap before the use can take the reload.
        if use_pos[k] < ru.pos && (ru.pos & !1) > s {
            return self.spill_between(c, s, ru.pos);
        }
        let mut new_end = e.min(block_pos[k]);
        if new_end < e && new_end & 1 != 0 && new_end > s && ru.pos < new_end {
            // Blocked at a late point after its register use: hold the
            // register to there, the stack after.
            let t = self.split_at(c, new_end).expect("a split inside the piece");
            self.assign(c, k);
            self.split_and_spill_intersecting(c)?;
            return self.spill_tail(t);
        }
        if new_end < e {
            new_end &= !1;
        }
        if new_end <= s {
            if (ru.pos & !1) > s {
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
        });
        let ps = &mut self.vregs[v as usize].pieces;
        let at = ps.iter().position(|&y| y == x).expect("the piece's value") + 1;
        ps.insert(at, n);
        Some(n)
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
        let mut b = eb;
        loop {
            let h = self.containing_loop(b);
            if h == NONE || self.rank[h as usize] <= self.rank[sb as usize] {
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
        let b = self.block_of_pos(pos);
        let mut h = if self.is_header(b) {
            b
        } else {
            self.containing_loop(b)
        };
        while h != NONE {
            let ls = self.block_start(h);
            // Defined at or after the header: spilling there buys nothing.
            if first >= ls {
                return (begin, pos);
            }
            if let Some(live) = self.piece_at(v, ls)
                && self.pieces[live as usize].loc != Loc::Stack
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
        if self.pieces[begin as usize].loc != Loc::Stack {
            self.spill_after(begin, pos);
        }
        let ps = self.vregs[v as usize].pieces.clone();
        let from = ps.iter().position(|&y| y == begin).expect("a piece") + 1;
        for &y in &ps[from..] {
            if y == end {
                break;
            }
            if self.pieces[y as usize].loc != Loc::Stack {
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
        self.pieces[x as usize].loc = Loc::Stack;
        let v = self.pieces[x as usize].vreg;
        self.slot_of(v);
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
        if self.pieces[third as usize].start() & 1 != 0 {
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
