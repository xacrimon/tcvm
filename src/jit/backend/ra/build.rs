//! Live ranges: each group's intervals and use positions, the points fixed
//! operands reserve registers over, and the clobbers.

use super::{Ivs, Loc, NONE, Need, Piece, Pos, Ra, Use, VInfo, Want, class_index, intersect};
use crate::jit::backend::reg::{
    Allocation, Block, OperandConstraint, OperandKind, OperandPos, PReg, PRegSet,
};
use crate::jit::backend::vcode::{Remat, TargetInst, VCode};

impl Ra {
    pub(super) fn build<I: TargetInst>(
        &mut self,
        f: &VCode<I>,
        w: &mut Build,
    ) -> Result<(), String> {
        let nv = f.num_vregs();
        let n = f.operands.len() as u32;
        self.allocs = vec![Allocation::none(); n as usize];
        self.vregs.extend((0..nv).map(|v| VInfo {
            class: class_index(f.vreg_classes[v]) as u8,
            remat: f.remat[v] != Remat::None,
            args_in_group: false,
            def: NONE,
            param_of: NONE,
            head: NONE,
            slot: NONE,
            defs: (0, 0),
            members: (0, 0),
            flat: (0, 0),
        }));
        // A constant some use takes from wherever it is (a snapshot, a
        // call's slots) is spilled as any value: such a use has no register.
        for (k, &w) in self.anyuse.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                self.vregs[k * 64 + w.trailing_zeros() as usize].remat = false;
                w &= w - 1;
            }
        }
        self.count(f, w);
        self.walk(f, n, w)?;
        self.reservations(f, w);
        self.interval_runs(w);
        self.group(f, &mut w.span, &w.uoff);
        self.members_and_defs(w);
        self.make_pieces(w);
        Ok(())
    }

    /// Each value's run of uses (`uoff`) and of snapshot operands
    /// (`snap_off`), counted ahead for the walk to fill from the end.
    fn count<I: TargetInst>(&mut self, f: &VCode<I>, w: &mut Build) {
        let nv = self.vregs.len();
        let Build { uoff, at, sat, .. } = w;
        uoff.resize(nv + 1, 0);
        self.snap_off.resize(nv + 1, 0);
        for op in &f.operands {
            let v = op.vreg().vreg();
            match (op.kind(), op.constraint()) {
                (OperandKind::Use, OperandConstraint::Any | OperandConstraint::Stack) => {
                    self.snap_off[v + 1] += 1;
                }
                (OperandKind::Def, OperandConstraint::FixedReg(_)) => {}
                // An early register definition is also a use at the late point.
                (OperandKind::Def, _)
                    if op.pos() == OperandPos::Early
                        && def_need(&self.vregs[v], f.remat[v]) == Need::Reg =>
                {
                    uoff[v + 1] += 2;
                }
                _ => uoff[v + 1] += 1,
            }
        }
        prefix(uoff);
        prefix(&mut self.snap_off);
        at.extend_from_slice(&uoff[1..]);
        sat.extend_from_slice(&self.snap_off[1..]);
        let none = Use {
            pos: 0,
            alloc: 0,
            need: Need::Any,
        };
        self.uses.reserve_exact(uoff[nv] as usize);
        self.uses.resize(uoff[nv] as usize, none);
        self.snaps.reserve_exact(self.snap_off[nv] as usize);
        self.snaps.resize(self.snap_off[nv] as usize, (0, 0));
    }

    /// The backward walk: interval, use and reservation records.
    #[inline(never)]
    fn walk<I: TargetInst>(&mut self, f: &VCode<I>, n: u32, w: &mut Build) -> Result<(), String> {
        let nv = f.num_vregs();
        // Intervals as `(vreg, ..)` records, uses and snapshot operands into
        // their runs from the end: blocks last to first, an instruction's
        // late point before its early one, operands last to first, so each
        // value's come in descending order. An interval joins the value's
        // previous one when they touch (`last`).
        let Build {
            ivr,
            fixtmp,
            fixed_def,
            stack_use,
            open,
            touched,
            out,
            defined,
            defs,
            uoff,
            at: uat,
            sat,
            ..
        } = w;
        ivr.recs.reserve(n as usize);
        ivr.last.resize(nv, NONE);
        fixed_def.resize(nv, None);
        stack_use.resize(nv, false);
        open.resize(nv, NONE);
        out.resize(self.words, 0);
        defined.resize(self.words, 0);
        let has = |s: &[u64], v: usize| s[v / 64] >> (v % 64) & 1 != 0;
        for bi in (0..self.order.len()).rev() {
            let b = self.order[bi].index();
            let (start, end) = (self.block_start(b as u32), self.block_end(b as u32));
            out.fill(0);
            for s in f.block_succs(Block::new(b)) {
                for (o, x) in out.iter_mut().zip(self.live_in(s.index())) {
                    *o |= x;
                }
            }
            for s in 0..f.block_succs(Block::new(b)).len() {
                for a in f.branch_blockparams(Block::new(b), s) {
                    let a = a.vreg();
                    if !has(out, a) && open[a] == NONE {
                        open[a] = end;
                        touched.push(a as u32);
                    }
                }
            }
            for l in (self.first[b]..=self.last[b]).rev() {
                let i = self.at_lin[l as usize];
                let base = f.ops[i.index()].0;
                let ops = f.inst_operands(i);
                let (early, late) = (2 * l, 2 * l + 1);
                for pos in [late, early] {
                    for (k, op) in ops.iter().enumerate().rev() {
                        let v = op.vreg().vreg();
                        let alloc = base + k as u32;
                        let at = if op.pos() == OperandPos::Late {
                            late
                        } else {
                            early
                        };
                        if at != pos {
                            // An early register definition keeps its
                            // register through the instruction.
                            if pos == late
                                && op.kind() == OperandKind::Def
                                && matches!(
                                    op.constraint(),
                                    OperandConstraint::Reg | OperandConstraint::Any
                                )
                                && def_need(&self.vregs[v], f.remat[v]) == Need::Reg
                            {
                                let need = Need::Reg;
                                put(&mut self.uses, uat, v, Use { pos, alloc, need });
                            }
                            continue;
                        }
                        if op.kind() == OperandKind::Use {
                            if has(out, v) {
                            } else if open[v] == NONE {
                                open[v] = pos + 1;
                                touched.push(v as u32);
                            } else {
                                open[v] = open[v].max(pos + 1);
                            }
                            let need = match op.constraint() {
                                OperandConstraint::Reg => Need::Reg,
                                OperandConstraint::Any | OperandConstraint::Stack => {
                                    let stack = op.constraint() == OperandConstraint::Stack;
                                    stack_use[v] |= stack;
                                    put(&mut self.snaps, sat, v, (pos, alloc << 1 | stack as u32));
                                    continue;
                                }
                                OperandConstraint::FixedReg(r) => {
                                    fixtmp.push((r.index() as u32, early, pos + 1, v as u32));
                                    Need::Fixed(r)
                                }
                                c => return Err(format!("unsupported use constraint {c:?}")),
                            };
                            put(&mut self.uses, uat, v, Use { pos, alloc, need });
                            continue;
                        }
                        // A definition holds its location through the
                        // instruction at least.
                        let to = if has(out, v) {
                            defined[v / 64] |= 1 << (v % 64);
                            defs.push(v as u32);
                            end
                        } else {
                            match std::mem::replace(&mut open[v], NONE) {
                                NONE => 2 * l + 2,
                                e => e.max(2 * l + 2),
                            }
                        };
                        let vi = &mut self.vregs[v];
                        if vi.def != NONE || vi.param_of != NONE {
                            return Err(format!("v{v} defined twice"));
                        }
                        vi.def = pos;
                        ivr.add(v, pos, to);
                        match op.constraint() {
                            OperandConstraint::FixedReg(r) => {
                                fixed_def[v] = Some(r);
                                self.allocs[alloc as usize] = Allocation::reg(r);
                                fixtmp.push((r.index() as u32, pos, 2 * l + 2, v as u32));
                            }
                            OperandConstraint::Reg | OperandConstraint::Any => {
                                let need = def_need(&self.vregs[v], f.remat[v]);
                                put(&mut self.uses, uat, v, Use { pos, alloc, need });
                            }
                            c => return Err(format!("unsupported def constraint {c:?}")),
                        }
                    }
                }
            }
            for p in f.block_params(Block::new(b)) {
                let v = p.vreg();
                let vi = &mut self.vregs[v];
                if vi.def != NONE || vi.param_of != NONE {
                    return Err(format!("v{v} defined twice"));
                }
                vi.param_of = b as u32;
                if open[v] != NONE {
                    ivr.add(v, start, open[v]);
                    open[v] = NONE;
                }
            }
            for v in touched.drain(..) {
                let v = v as usize;
                if open[v] != NONE {
                    ivr.add(v, start, open[v]);
                    open[v] = NONE;
                }
            }
            ivr.live_through(out, defined, start, end);
            for v in defs.drain(..) {
                defined[v as usize / 64] = 0;
            }
        }
        debug_assert!(uat[..] == uoff[..nv] && sat[..] == self.snap_off[..nv]);
        debug_assert!((0..nv).all(|v| {
            let us = &self.uses[uoff[v] as usize..uoff[v + 1] as usize];
            us.windows(2)
                .all(|w| (w[0].pos, w[0].alloc) < (w[1].pos, w[1].alloc))
        }));
        Ok(())
    }

    /// Each value's intervals ascending, runs of `ivs` (`span`).
    fn interval_runs(&mut self, w: &mut Build) {
        let nv = self.vregs.len();
        let Build {
            ivr,
            ioff,
            at,
            span,
            ..
        } = w;
        let ivr = &ivr.recs;
        runs(ioff, nv, ivr.iter().map(|r| r.0));
        self.ivs.resize(ivr.len(), (0, 0));
        at.clear();
        at.extend_from_slice(&ioff[1..]);
        for &(v, a, b) in ivr {
            at[v as usize] -= 1;
            self.ivs[at[v as usize] as usize] = (a, b);
        }
        span.extend((0..nv).map(|v| (ioff[v], ioff[v + 1])));
        debug_assert!((0..nv).all(|v| {
            let ivs = &self.ivs[span[v].0 as usize..span[v].1 as usize];
            ivs.windows(2).all(|w| w[0].1 < w[1].0)
        }));
    }

    /// Values computed again, and the groups of parameters and arguments.
    fn group<I: TargetInst>(&mut self, f: &VCode<I>, span: &mut [(u32, u32)], uoff: &[u32]) {
        let nv = self.vregs.len();
        // The definition of a value that loses its remat needs a register.
        let need_reg_at_def = |uses: &mut [Use], def: Pos, v: usize| {
            for u in &mut uses[uoff[v] as usize..uoff[v + 1] as usize] {
                if u.pos == def {
                    u.need = Need::Reg;
                }
            }
        };
        // A value computed again from another only where that one is live
        // (always holding it at the gap before a reload), not from a value
        // itself computed again.
        for v in 0..nv {
            if let Remat::AndImm { src, .. } = f.remat[v]
                && self.vregs[v].remat
            {
                let src = src as usize;
                let si = run_ivs(&self.ivs, span[src]);
                let within = |&(a, b): &(Pos, Pos)| {
                    let k = si.raw.partition_point(|iv| iv.1 < b);
                    k < si.len() && si.raw[k].0 <= a
                };
                let mine = &self.ivs[span[v].0 as usize..span[v].1 as usize];
                if self.vregs[src].remat || !mine.iter().all(within) {
                    self.vregs[v].remat = false;
                    need_reg_at_def(&mut self.uses, self.vregs[v].def, v);
                }
            }
        }
        // Groups: a block parameter shares one allocation unit with each
        // argument passed to it whose range stays clear of the group's
        // (V8's `BundleBuilder`, allocated as one range), so the edge needs
        // no move unless a split moves one of them. A merged group's
        // intervals are written after the rest of `ivs`.
        self.rep.extend(0..nv as u32);
        for bi in 0..self.order.len() {
            let b = self.order[bi];
            for (k, q) in f.block_params(b).iter().enumerate() {
                let mut all = true;
                for &p in f.block_preds(b) {
                    let pk = (f.block_succs(p).iter())
                        .position(|&x| x == b)
                        .expect("an edge");
                    let a = f.branch_blockparams(p, pk)[k];
                    let (gq, ga) = (self.find(q.vreg() as u32), self.find(a.vreg() as u32));
                    if gq == ga {
                        continue;
                    }
                    let (gq, ga) = (gq as usize, ga as usize);
                    // A constant stays out (it is made again on the edge); a
                    // derived value joins and is spilled as any value.
                    let konst = |v: usize| {
                        matches!(f.remat[v], Remat::Int(_) | Remat::Float(_)) && self.vregs[v].remat
                    };
                    if self.vregs[gq].class != self.vregs[ga].class
                        || konst(gq)
                        || konst(ga)
                        || span[gq].0 == span[gq].1
                        || span[ga].0 == span[ga].1
                        || intersect(
                            run_ivs(&self.ivs, span[gq]),
                            run_ivs(&self.ivs, span[ga]),
                            0,
                        )
                        .is_some()
                    {
                        all = false;
                        continue;
                    }
                    self.rep[ga] = gq as u32;
                    for x in [gq, ga] {
                        if self.vregs[x].remat {
                            self.vregs[x].remat = false;
                            need_reg_at_def(&mut self.uses, self.vregs[x].def, x);
                        }
                    }
                    let s = self.ivs.len();
                    let (a, b) = (span[gq], span[ga]);
                    merge(&mut self.ivs, s, a, b, |x| x.0, push_coalesced);
                    span[gq] = (s as u32, self.ivs.len() as u32);
                    span[ga] = (s as u32, s as u32);
                }
                self.vregs[q.vreg()].args_in_group = all;
            }
        }
        for v in 0..nv as u32 {
            let r = self.find(v);
            self.rep[v as usize] = r;
        }
    }

    /// Each group's members, then their definitions.
    fn members_and_defs(&mut self, w: &mut Build) {
        let nv = self.vregs.len();
        self.defs.reserve(nv);
        let Build { moff, at, .. } = w;
        runs(moff, nv, self.rep.iter().copied());
        self.members.resize(nv, 0);
        at.clear();
        at.extend_from_slice(moff);
        for v in 0..nv {
            let r = self.rep[v] as usize;
            self.members[at[r] as usize] = v as u32;
            at[r] += 1;
        }
        for g in 0..nv {
            let ms = (moff[g], moff[g + 1]);
            self.vregs[g].members = ms;
            let s = self.defs.len() as u32;
            for &m in &self.members[ms.0 as usize..ms.1 as usize] {
                let d = self.vregs[m as usize].def;
                if d != NONE {
                    self.defs.push(d);
                }
            }
            let e = self.defs.len() as u32;
            self.defs[s as usize..].sort_unstable();
            self.vregs[g].defs = (s, e);
        }
    }

    /// Each group's uses and first piece, its fixed definitions' pieces,
    /// and the queue.
    fn make_pieces(&mut self, w: &mut Build) {
        let nv = self.vregs.len();
        let Build {
            fixed_def,
            stack_use,
            span,
            uoff,
            fx,
            keys,
            ..
        } = w;
        // Scans split about half as many pieces again.
        self.pieces.reserve(2 * nv);
        self.initial.reserve(nv);
        let merged = |v: usize| {
            let (a, b) = self.vregs[self.rep[v] as usize].members;
            b - a > 1
        };
        let extra: u32 = (0..nv)
            .filter(|&v| merged(v))
            .map(|v| uoff[v + 1] - uoff[v])
            .sum();
        self.uses.reserve_exact(extra as usize);
        self.next_of.reserve_exact(self.uses.len() + extra as usize);
        for (v, &(lo, hi)) in span.iter().enumerate() {
            if self.rep[v] != v as u32 || lo == hi {
                continue;
            }
            // The group's uses by point and operand: its own run, or its
            // members' merged after every value's (a cursor per run).
            let (ma, mb) = self.vregs[v].members;
            let (mut ulo, mut uhi) = (uoff[v] as usize, uoff[v + 1] as usize);
            if mb - ma > 1 {
                ulo = self.uses.len();
                keys.clear();
                keys.extend((ma..mb).map(|k| {
                    let m = self.members[k as usize] as usize;
                    (uoff[m] as u64) << 32 | uoff[m + 1] as u64
                }));
                while let Some(c) = (keys.iter_mut().filter(|c| **c >> 32 != **c & 0xffff_ffff))
                    .min_by_key(|c| {
                        let u = self.uses[(**c >> 32) as usize];
                        (u.pos, u.alloc)
                    })
                {
                    let u = self.uses[(*c >> 32) as usize];
                    *c += 1 << 32;
                    self.uses.push(u);
                }
                uhi = self.uses.len();
            }
            let mut stack = false;
            fx.clear();
            for k in ma..mb {
                let m = self.members[k as usize] as usize;
                stack |= stack_use[m];
                if let Some(r) = fixed_def[m] {
                    fx.push((self.vregs[m].def as u64) << 32 | r.index() as u64);
                }
            }
            self.next_of.resize(self.uses.len(), [0; 3]);
            let uhi = uhi as u32;
            let mut next = [uhi; 3];
            for k in (ulo as u32..uhi).rev() {
                for (n, w) in next.iter_mut().zip(Want::of(self.uses[k as usize].need)) {
                    if w {
                        *n = k;
                    }
                }
                self.next_of[k as usize] = next;
            }
            if stack {
                self.slot_of(v as u32);
            }
            let x = self.pieces.len() as u32;
            self.pieces.push(Piece {
                g: v as u32,
                class: self.vregs[v].class,
                start: self.ivs[lo as usize].0,
                end: self.ivs[hi as usize - 1].1,
                lo,
                hi,
                ulo: ulo as u32,
                uhi,
                prev: NONE,
                next: NONE,
                loc: Loc::None,
                fixed: false,
                stack: false,
            });
            self.vregs[v].head = x;
            // Each fixed definition's own point in its register; the rest is
            // allocated like any value.
            fx.sort_unstable();
            let mut rest = Some(x);
            for &d in fx.iter() {
                let (def, r) = ((d >> 32) as Pos, PReg::from_index(d as u32 as usize));
                let Some(cur) = rest else { break };
                let at = match self.split_at(cur, def) {
                    Some(y) if y != cur => {
                        self.initial.push(key(self.start(cur), cur));
                        y
                    }
                    Some(y) => y,
                    None => unreachable!("a definition past its range"),
                };
                self.pieces[at as usize].loc = Loc::Reg(r);
                self.pieces[at as usize].fixed = true;
                rest = self.split_at(at, (def & !1) + 2);
            }
            if let Some(cur) = rest {
                self.initial.push(key(self.start(cur), cur));
            }
        }
        self.initial.sort_unstable();
    }

    /// Fixed operands' reservations per class by point, and each clobbering
    /// instruction's registers, in point order.
    fn reservations<I: TargetInst>(&mut self, f: &VCode<I>, w: &mut Build) {
        let Build {
            fixtmp, keys, seen, ..
        } = w;
        // By class and point; only allocatable registers are asked about.
        keys.extend(
            (fixtmp.iter().enumerate())
                .filter(|(_, e)| self.reg_idx[e.0 as usize] != NONE)
                .map(|(k, e)| {
                    let c = class_index(PReg::from_index(e.0 as usize).class()) as u64;
                    c << 56 | (e.1 as u64) << 24 | k as u64
                }),
        );
        assert!(fixtmp.len() < 1 << 24, "too many fixed operands");
        keys.sort_unstable();
        for &key in keys.iter() {
            let (r, a, b, o) = fixtmp[(key & 0xff_ffff) as usize];
            let (c, k) = ((key >> 56) as usize, self.reg_idx[r as usize]);
            self.fixregs[c] |= 1 << k;
            self.fixres[c].push((a, b, o, k as u8));
        }
        // Calls clobber one of a few sets: each translated once.
        for (l, &i) in self.at_lin.iter().enumerate() {
            let set = f.inst_clobbers(i);
            if set == PRegSet::empty() {
                continue;
            }
            let m = match seen.iter().find(|x| x.0 == set) {
                Some(&(_, m)) => m,
                None => {
                    let mut m = [0u32; 2];
                    for r in set {
                        let k = self.reg_idx[r.index()];
                        if k != NONE {
                            m[class_index(r.class())] |= 1 << k;
                        }
                    }
                    seen.push((set, m));
                    m
                }
            };
            if m != [0, 0] {
                self.clobs.push((2 * l as u32 + 1, m));
                self.clobregs[0] |= m[0];
                self.clobregs[1] |= m[1];
            }
        }
    }

    /// The group of `v`, compressing the path.
    fn find(&mut self, v: u32) -> u32 {
        let mut r = v;
        while self.rep[r as usize] != r {
            r = self.rep[r as usize];
        }
        let mut x = v;
        while self.rep[x as usize] != r {
            let n = self.rep[x as usize];
            self.rep[x as usize] = r;
            x = n;
        }
        r
    }
}

/// `build`'s buffers. The backward walk records intervals as `(vreg, ..)`,
/// fixed operands' reservations as `(register, from, to, owner)`, and per
/// value its fixed definition and whether a use takes it from the stack.
#[derive(Default)]
pub(super) struct Build {
    ivr: Recs,
    fixtmp: Vec<(u32, Pos, Pos, u32)>,
    fixed_def: Vec<Option<PReg>>,
    stack_use: Vec<bool>,
    /// Per value not live out, the end of the interval open above the
    /// walk's point; the values with one.
    open: Vec<u32>,
    touched: Vec<u32>,
    /// Live out of the block, and of those the ones defined in it.
    out: Vec<u64>,
    defined: Vec<u64>,
    defs: Vec<u32>,
    /// Each value's run of intervals (`span`, by `ioff`) and of uses
    /// (`Ra::uses`, by `uoff`), each group's run of members (`moff`), and
    /// the cursors filling the runs (`sat` the snapshot operands').
    ioff: Vec<u32>,
    uoff: Vec<u32>,
    moff: Vec<u32>,
    at: Vec<u32>,
    sat: Vec<u32>,
    span: Vec<(u32, u32)>,
    /// Sort keys, a group's fixed definitions, and the clobber sets
    /// translated so far.
    keys: Vec<u64>,
    fx: Vec<u64>,
    seen: Vec<(PRegSet, [u32; 2])>,
}

impl Build {
    pub(super) fn clear(&mut self) {
        clear!(self; fixtmp, fixed_def, stack_use, open, touched, out, defined, defs,
            ioff, uoff, moff, at, sat, span, keys, fx, seen);
        clear!(self.ivr; recs, last);
    }
}

/// A register definition's need: a constant's only where it is used; a
/// value derived from another is made where it is defined, and only
/// computed again where a reload would be.
fn def_need(vi: &VInfo, r: Remat) -> Need {
    if vi.remat && !matches!(r, Remat::AndImm { .. }) {
        Need::Any
    } else {
        Need::Reg
    }
}

/// `x` into `v`'s run, filled from its end (`at`).
#[inline]
fn put<T>(run: &mut [T], at: &mut [u32], v: usize, x: T) {
    at[v] -= 1;
    run[at[v] as usize] = x;
}

/// Interval records as the walk makes them, each value's last (lowest).
#[derive(Default)]
struct Recs {
    recs: Vec<(u32, Pos, Pos)>,
    last: Vec<u32>,
}

impl Recs {
    /// `v` holds `a..b`: joined to its last interval when that starts at
    /// `b`.
    #[inline]
    fn add(&mut self, v: usize, a: Pos, b: Pos) {
        match self.last[v] {
            k if k != NONE && self.recs[k as usize].1 == b => self.recs[k as usize].1 = a,
            _ => {
                self.last[v] = self.recs.len() as u32;
                self.recs.push((v as u32, a, b));
            }
        }
    }

    /// Values live out of a block and not defined in it hold all of it
    /// (a parameter is defined at its start).
    #[inline(never)]
    fn live_through(&mut self, out: &[u64], defined: &[u64], start: Pos, end: Pos) {
        for (k, (&w, &d)) in out.iter().zip(defined).enumerate() {
            let mut w = w & !d;
            while w != 0 {
                self.add(k * 64 + w.trailing_zeros() as usize, start, end);
                w &= w - 1;
            }
        }
    }
}

/// A queue entry: `start`, then piece `x`, in one comparable word.
pub(super) fn key(start: Pos, x: u32) -> u64 {
    (start as u64) << 32 | x as u64
}

fn run_ivs(ivs: &[(Pos, Pos)], (a, b): (u32, u32)) -> Ivs<'_> {
    Ivs::plain(&ivs[a as usize..b as usize])
}

/// Offsets of the runs of a counting sort by `keys` (each below `n`), into
/// `off`: key `k`'s run is `off[k]..off[k + 1]`.
fn runs(off: &mut Vec<u32>, n: usize, keys: impl Iterator<Item = u32>) {
    off.clear();
    off.resize(n + 1, 0);
    for k in keys {
        off[k as usize + 1] += 1;
    }
    prefix(off);
}

/// Run offsets from the counts in `off[1..]`.
fn prefix(off: &mut [u32]) {
    for k in 1..off.len() {
        off[k] += off[k - 1];
    }
}

/// Append to `v` (from `s`, by `put`) the merge of its sorted runs `a` and
/// `b`, by `key`.
fn merge<T: Copy, K: Ord>(
    v: &mut Vec<T>,
    s: usize,
    a: (u32, u32),
    b: (u32, u32),
    key: impl Fn(&T) -> K,
    put: impl Fn(&mut Vec<T>, usize, T),
) {
    let (mut i, mut j) = (a.0 as usize, b.0 as usize);
    let (ae, be) = (a.1 as usize, b.1 as usize);
    while i < ae || j < be {
        let x = if j == be || i < ae && key(&v[i]) <= key(&v[j]) {
            i += 1;
            v[i - 1]
        } else {
            j += 1;
            v[j - 1]
        };
        put(v, s, x);
    }
}

/// Merge the sorted `v[lo..mid]` and `v[mid..]` in their place, by `put`.
pub(super) fn merge_tail<T: Copy, K: Ord>(
    v: &mut Vec<T>,
    lo: usize,
    mid: usize,
    key: impl Fn(&T) -> K,
    put: impl Fn(&mut Vec<T>, usize, T),
) {
    let end = v.len();
    merge(
        v,
        end,
        (lo as u32, mid as u32),
        (mid as u32, end as u32),
        key,
        put,
    );
    let n = v.len() - end;
    v.copy_within(end.., lo);
    v.truncate(lo + n);
}

/// Append `x` to the run of `ivs` from `s`, joined with its last interval
/// when they overlap or touch.
pub(super) fn push_coalesced(ivs: &mut Vec<(Pos, Pos)>, s: usize, x: (Pos, Pos)) {
    match ivs.len() {
        n if n > s && x.0 <= ivs[n - 1].1 => ivs[n - 1].1 = ivs[n - 1].1.max(x.1),
        _ => ivs.push(x),
    }
}
