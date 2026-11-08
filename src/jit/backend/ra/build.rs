//! Live ranges: each value's intervals, its use positions, and the points
//! fixed operands and clobbers reserve registers over.

use super::{NONE, Need, Piece, Pos, Ra, Use, VInfo, class_index};
use crate::jit::backend::reg::{
    Allocation, Block, Inst, OperandConstraint, OperandKind, OperandPos,
};
use crate::jit::backend::vcode::TargetInst;

impl<I: TargetInst> Ra<'_, I> {
    pub(super) fn build(&mut self) -> Result<(), String> {
        let f = self.f;
        let nv = f.num_vregs();
        let mut n = 0u32;
        self.alloc_base = Vec::with_capacity(f.num_insts());
        for i in 0..f.num_insts() {
            self.alloc_base.push(n);
            n += f.inst_operands(Inst::new(i)).len() as u32;
        }
        self.allocs = vec![Allocation::none(); n as usize];
        self.vregs = (0..nv)
            .map(|v| VInfo {
                class: class_index(f.vreg_classes[v]) as u8,
                pieces: Vec::new(),
                slot: NONE,
                def: NONE,
                param_of: NONE,
                args_in_group: false,
                defs: Vec::new(),
                remat: f.remat[v] != crate::jit::backend::vcode::Remat::None,
            })
            .collect();
        // A constant some use takes from wherever it is (a snapshot, a
        // call's slots) is spilled as any value: such a use has no register.
        for i in 0..f.num_insts() {
            for op in f.inst_operands(Inst::new(i)) {
                if op.kind() == OperandKind::Use
                    && matches!(
                        op.constraint(),
                        OperandConstraint::Stack | OperandConstraint::Any
                    )
                {
                    self.vregs[op.vreg().vreg()].remat = false;
                }
            }
        }
        let mut ivs: Vec<Vec<(Pos, Pos)>> = vec![Vec::new(); nv];
        let mut uses: Vec<Vec<Use>> = vec![Vec::new(); nv];
        let mut fixed_def = vec![None; nv];
        let mut stack_use = vec![false; nv];
        // Per value, the end of the interval open above the current point.
        let mut open = vec![NONE; nv];
        let mut touched: Vec<u32> = Vec::new();
        for bi in 0..self.order.len() {
            let b = self.order[bi].index();
            let (start, end) = (self.block_start(b as u32), self.block_end(b as u32));
            let out = self.live_out(b);
            for (k, &w) in out.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let v = k * 64 + w.trailing_zeros() as usize;
                    w &= w - 1;
                    open[v] = end;
                    touched.push(v as u32);
                }
            }
            for s in 0..f.block_succs(Block::new(b)).len() {
                for a in f.branch_blockparams(Block::new(b), s) {
                    if open[a.vreg()] == NONE {
                        open[a.vreg()] = end;
                        touched.push(a.vreg() as u32);
                    }
                }
            }
            for l in (self.first[b]..=self.last[b]).rev() {
                let i = self.at_lin[l as usize];
                let base = self.alloc_base[i.index()];
                let ops = f.inst_operands(i);
                let (early, late) = (2 * l, 2 * l + 1);
                for (k, op) in ops.iter().enumerate() {
                    if op.kind() != OperandKind::Def {
                        continue;
                    }
                    let v = op.vreg().vreg();
                    let pos = if op.pos() == OperandPos::Late {
                        late
                    } else {
                        early
                    };
                    // A definition holds its location through the instruction
                    // at least.
                    let to = match std::mem::replace(&mut open[v], NONE) {
                        NONE => 2 * l + 2,
                        e => e.max(2 * l + 2),
                    };
                    let vi = &mut self.vregs[v];
                    if vi.def != NONE || vi.param_of != NONE {
                        return Err(format!("v{v} defined twice"));
                    }
                    vi.def = pos;
                    ivs[v].push((pos, to));
                    let alloc = base + k as u32;
                    match op.constraint() {
                        OperandConstraint::FixedReg(r) => {
                            fixed_def[v] = Some(r);
                            self.allocs[alloc as usize] = Allocation::reg(r);
                            self.fixed[r.index()].push((pos, 2 * l + 2, v as u32));
                        }
                        // A constant needs a register only where it is used;
                        // a value derived from another is made where it is
                        // defined, and only computed again where a reload
                        // would be.
                        OperandConstraint::Reg | OperandConstraint::Any
                            if self.vregs[v].remat
                                && !matches!(
                                    f.remat[v],
                                    crate::jit::backend::vcode::Remat::AndImm { .. }
                                ) =>
                        {
                            uses[v].push(Use {
                                pos,
                                alloc,
                                need: Need::Any,
                            });
                        }
                        OperandConstraint::Reg | OperandConstraint::Any => {
                            uses[v].push(Use {
                                pos,
                                alloc,
                                need: Need::Reg,
                            });
                            // An early definition keeps its register through
                            // the instruction.
                            if pos == early {
                                uses[v].push(Use {
                                    pos: late,
                                    alloc,
                                    need: Need::Reg,
                                });
                            }
                        }
                        c => return Err(format!("unsupported def constraint {c:?}")),
                    }
                }
                for (k, op) in ops.iter().enumerate() {
                    if op.kind() != OperandKind::Use {
                        continue;
                    }
                    let v = op.vreg().vreg();
                    let pos = if op.pos() == OperandPos::Late {
                        late
                    } else {
                        early
                    };
                    if open[v] == NONE {
                        open[v] = pos + 1;
                        touched.push(v as u32);
                    } else {
                        open[v] = open[v].max(pos + 1);
                    }
                    let need = match op.constraint() {
                        OperandConstraint::Reg => Need::Reg,
                        OperandConstraint::Any => Need::Any,
                        OperandConstraint::Stack => {
                            stack_use[v] = true;
                            Need::Stack
                        }
                        OperandConstraint::FixedReg(r) => {
                            self.fixed[r.index()].push((early, pos + 1, v as u32));
                            Need::Fixed(r)
                        }
                        c => return Err(format!("unsupported use constraint {c:?}")),
                    };
                    uses[v].push(Use {
                        pos,
                        alloc: base + k as u32,
                        need,
                    });
                }
                for r in f.inst_clobbers(i) {
                    if self.reg_idx[r.index()] != NONE {
                        self.fixed[r.index()].push((late, late + 1, NONE));
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
                    ivs[v].push((start, open[v]));
                    open[v] = NONE;
                }
            }
            for v in touched.drain(..) {
                let v = v as usize;
                if open[v] != NONE {
                    ivs[v].push((start, open[v]));
                    open[v] = NONE;
                }
            }
        }
        for list in &mut self.fixed {
            list.sort_unstable_by_key(|e| e.0);
        }
        let mut merged: Vec<Vec<(Pos, Pos)>> = ivs
            .into_iter()
            .map(|mut iv| {
                iv.sort_unstable_by_key(|x| x.0);
                coalesce(iv)
            })
            .collect();
        // A value computed again from another only where that one is live
        // (always holding it at the gap before a reload), not from a value
        // itself computed again.
        for v in 0..nv {
            if let crate::jit::backend::vcode::Remat::AndImm { src, .. } = f.remat[v]
                && self.vregs[v].remat
            {
                let src = src as usize;
                let within = |&(a, b): &(Pos, Pos)| {
                    let k = merged[src].partition_point(|iv| iv.1 < b);
                    k < merged[src].len() && merged[src][k].0 <= a
                };
                if self.vregs[src].remat || !merged[v].iter().all(within) {
                    self.vregs[v].remat = false;
                    let def = self.vregs[v].def;
                    for u in &mut uses[v] {
                        if u.pos == def {
                            u.need = Need::Reg;
                        }
                    }
                }
            }
        }
        // Groups: a block parameter shares one allocation unit with each
        // argument passed to it whose range stays clear of the group's
        // (V8's `BundleBuilder`, allocated as one range), so
        // the edge needs no move unless a split moves one of them.
        self.rep = (0..nv as u32).collect();
        let mut fixed: Vec<Vec<(Pos, crate::jit::backend::reg::PReg)>> = fixed_def
            .iter()
            .enumerate()
            .map(|(v, r)| r.map(|r| (self.vregs[v].def, r)).into_iter().collect())
            .collect();
        for bi in 0..self.order.len() {
            let b = self.order[bi];
            for (k, q) in f.block_params(b).iter().enumerate() {
                let mut all = true;
                for &p in f.block_preds(b) {
                    let pk = f
                        .block_succs(p)
                        .iter()
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
                        matches!(
                            f.remat[v],
                            crate::jit::backend::vcode::Remat::Int(_)
                                | crate::jit::backend::vcode::Remat::Float(_)
                        ) && self.vregs[v].remat
                    };
                    if self.vregs[gq].class != self.vregs[ga].class
                        || konst(gq)
                        || konst(ga)
                        || merged[gq].is_empty()
                        || merged[ga].is_empty()
                        || super::intersect(&merged[gq], &merged[ga], 0).is_some()
                    {
                        all = false;
                        continue;
                    }
                    self.rep[ga] = gq as u32;
                    for x in [gq, ga] {
                        if self.vregs[x].remat {
                            self.vregs[x].remat = false;
                            let def = self.vregs[x].def;
                            for u in uses[x].iter_mut() {
                                if u.pos == def {
                                    u.need = Need::Reg;
                                }
                            }
                        }
                    }
                    let mut iv = std::mem::take(&mut merged[ga]);
                    iv.extend_from_slice(&merged[gq]);
                    iv.sort_unstable_by_key(|x| x.0);
                    merged[gq] = coalesce(iv);
                    let u = std::mem::take(&mut uses[ga]);
                    uses[gq].extend(u);
                    let fx = std::mem::take(&mut fixed[ga]);
                    fixed[gq].extend(fx);
                    stack_use[gq] |= stack_use[ga];
                }
                self.vregs[q.vreg()].args_in_group = all;
            }
        }
        for v in 0..nv as u32 {
            let r = self.find(v);
            self.rep[v as usize] = r;
            let d = self.vregs[v as usize].def;
            if d != NONE {
                self.vregs[r as usize].defs.push(d);
            }
        }
        for vi in &mut self.vregs {
            vi.defs.sort_unstable();
        }
        let mut queue = Vec::new();
        for v in 0..nv {
            if self.rep[v] != v as u32 || merged[v].is_empty() {
                continue;
            }
            let mut us = std::mem::take(&mut uses[v]);
            us.sort_by_key(|u| u.pos);
            if stack_use[v] {
                self.slot_of(v as u32);
            }
            let x = self.pieces.len() as u32;
            self.pieces.push(Piece {
                vreg: v as u32,
                ivs: std::mem::take(&mut merged[v]),
                uses: us,
                loc: super::Loc::None,
                fixed: false,
            });
            self.vregs[v].pieces.push(x);
            // Each fixed definition's own point in its register; the rest is
            // allocated like any value.
            let mut fx = std::mem::take(&mut fixed[v]);
            fx.sort_unstable_by_key(|d| d.0);
            let mut rest = Some(x);
            for (def, r) in fx {
                let Some(cur) = rest else { break };
                let at = match self.split_at(cur, def) {
                    Some(y) if y != cur => {
                        queue.push(cur);
                        y
                    }
                    Some(y) => y,
                    None => unreachable!("a definition past its range"),
                };
                self.pieces[at as usize].loc = super::Loc::Reg(r);
                self.pieces[at as usize].fixed = true;
                rest = self.split_at(at, (def & !1) + 2);
            }
            if let Some(cur) = rest {
                queue.push(cur);
            }
        }
        for x in queue {
            let s = self.pieces[x as usize].start();
            self.unhandled.push(std::cmp::Reverse((s, x)));
        }
        Ok(())
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

/// Sorted intervals with overlapping and adjacent ones joined.
fn coalesce(iv: Vec<(Pos, Pos)>) -> Vec<(Pos, Pos)> {
    let mut out: Vec<(Pos, Pos)> = Vec::with_capacity(iv.len());
    for (a, b) in iv {
        match out.last_mut() {
            Some(m) if a <= m.1 => m.1 = m.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}
