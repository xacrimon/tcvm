//! After the scan: each operand's allocation and the moves between pieces,
//! across edges, into fixed registers, and the stores after definitions,
//! each gap's moves sequenced as one parallel move.

use regalloc2::{Allocation, Function, ProgPoint, RegClass, VReg};

use super::{Edit, Loc, NONE, Need, Output, Pos, Ra, intersect};
use crate::jit::backend::vcode::TargetInst;

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
/// copies into the instruction's fixed registers.
const DEF_STORE: u8 = 0;
const CONNECT: u8 = 1;
const EDGE_IN: u8 = 2;
const PARAM_STORE: u8 = 3;
const EDGE_OUT: u8 = 4;
const LATE_STORE: u8 = 5;
const FIXED_COPY: u8 = 6;

impl<I: TargetInst> Ra<'_, I> {
    pub(super) fn resolve(&mut self) -> Result<Output, String> {
        self.slot_derived();
        // A group is stored where it goes from a register to the stack when
        // that runs no more often than storing after its definitions (V8's
        // `DecideSpillingMode`; cold-only spills always), its slot uses
        // falling in stack pieces; else once after each definition, its slot
        // holding it from there on.
        self.deferred = (0..self.vregs.len())
            .map(|g| self.stores_at_transitions(g))
            .collect();
        self.color_slots();
        for x in 0..self.pieces.len() as u32 {
            let p = &self.pieces[x as usize];
            // Merged back into the piece it was split from.
            if p.ivs.is_empty() {
                continue;
            }
            if p.loc == Loc::None {
                return Err(format!("v{} left unallocated", p.vreg));
            }
            for u in &p.uses {
                let a = match u.need {
                    Need::Reg => match p.loc {
                        Loc::Reg(r) => Allocation::reg(r),
                        _ => return Err(format!("v{} needs a register at {}", p.vreg, u.pos)),
                    },
                    Need::Any => self.alloc_of(x),
                    Need::Stack => Allocation::stack(regalloc2::SpillSlot::new(
                        self.vregs[p.vreg as usize].slot as usize,
                    )),
                    Need::Fixed(r) => Allocation::reg(r),
                };
                self.allocs[u.alloc as usize] = a;
            }
        }
        let mut moves: Vec<(u32, u8, Mv)> = Vec::new();
        for v in 0..self.vregs.len() as u32 {
            let gr = self.rep[v as usize];
            let g = &self.vregs[gr as usize];
            if g.slot == NONE || g.pieces.is_empty() || self.deferred[gr as usize] {
                continue;
            }
            let (cls, slot) = (
                g.class,
                Allocation::stack(regalloc2::SpillSlot::new(g.slot as usize)),
            );
            let vi = &self.vregs[v as usize];
            if vi.def != NONE {
                let l = vi.def / 2;
                if self.lin_block[l as usize + 1] != self.lin_block[l as usize] {
                    return Err(format!("v{v} defined by a block's last instruction"));
                }
                let from = self.alloc_at(v, vi.def).expect("the definition's piece");
                debug_assert!(from.is_reg());
                moves.push((
                    l + 1,
                    DEF_STORE,
                    Mv {
                        from,
                        to: slot,
                        cls,
                        vreg: v,
                        remat: false,
                    },
                ));
            } else if vi.param_of != NONE && !vi.args_in_group {
                let b = vi.param_of;
                if let Some(from) = self.alloc_at(v, self.block_start(b))
                    && from.is_reg()
                {
                    moves.push((
                        self.first[b as usize],
                        PARAM_STORE,
                        Mv {
                            from,
                            to: slot,
                            cls,
                            vreg: v,
                            remat: false,
                        },
                    ));
                }
            }
        }
        for g in 0..self.vregs.len() as u32 {
            let gi = &self.vregs[g as usize];
            if self.rep[g as usize] != g || gi.pieces.is_empty() {
                continue;
            }
            let cls = gi.class;
            // Between pieces inside a block; a stack piece needs a store only
            // in cold code, before the instruction it starts at.
            let deferred = self.deferred[g as usize];
            for w in gi.pieces.windows(2) {
                let (a, b) = (&self.pieces[w[0] as usize], &self.pieces[w[1] as usize]);
                let at = b.start();
                if a.end() != at
                    || a.loc == b.loc
                    || self.is_def(g, at)
                    || b.loc == Loc::Remat
                    || b.loc == Loc::Stack && !deferred
                {
                    continue;
                }
                let bb = self.block_of_pos(at);
                if at == self.block_start(bb) {
                    continue;
                }
                if at & 1 != 0 && b.loc != Loc::Stack {
                    return Err(format!("v{g} reloaded inside instruction {}", at / 2));
                }
                moves.push((
                    at / 2,
                    if at & 1 != 0 { LATE_STORE } else { CONNECT },
                    self.mv(g, at - 1, self.alloc_of(w[1]), cls),
                ));
            }
            // Into fixed registers, from wherever the value is when the
            // instruction starts.
            for &x in &gi.pieces {
                for u in &self.pieces[x as usize].uses {
                    if let Need::Fixed(r) = u.need {
                        let n = u.pos / 2;
                        let m = self.mv(g, 2 * n, Allocation::reg(r), cls);
                        if m.remat || m.from != m.to {
                            moves.push((n, FIXED_COPY, m));
                        }
                    }
                }
            }
        }
        self.edge_moves(&mut moves)?;
        moves.sort_by_key(|&(l, phase, _)| (self.at_lin[l as usize].index(), phase));
        let mut edits: Vec<(ProgPoint, Edit)> = Vec::new();
        let mut k = 0;
        while k < moves.len() {
            let (l, phase) = (moves[k].0, moves[k].1);
            let mut group = Vec::new();
            while k < moves.len() && moves[k].0 == l && moves[k].1 == phase {
                group.push(moves[k].2);
                k += 1;
            }
            let at = ProgPoint::before(self.at_lin[l as usize]);
            self.sequence(group, |e| edits.push((at, e)))?;
        }
        Ok(Output {
            num_spillslots: self.num_slots as usize,
            edits,
            allocs: std::mem::take(&mut self.allocs),
            inst_alloc_offsets: std::mem::take(&mut self.alloc_base),
        })
    }

    fn stores_at_transitions(&self, g: usize) -> bool {
        let gi = &self.vregs[g];
        if self.rep[g] != g as u32 || gi.slot == NONE {
            return false;
        }
        let mut at_transitions = 0f32;
        for &x in &gi.pieces {
            let p = &self.pieces[x as usize];
            if p.ivs.is_empty() {
                continue;
            }
            if p.loc == Loc::Stack {
                at_transitions += self.freq_at(p.start());
            } else if p.uses.iter().any(|u| u.need == Need::Stack) {
                return false;
            }
        }
        let mut at_defs = 0f32;
        for v in 0..self.vregs.len() {
            if self.rep[v] != g as u32 {
                continue;
            }
            let vi = &self.vregs[v];
            if vi.def != NONE {
                at_defs += self.freq_at(vi.def);
            } else if vi.param_of != NONE && !vi.args_in_group {
                at_defs += self.freq_at(self.block_start(vi.param_of));
            }
        }
        at_transitions <= at_defs
    }

    /// A value computed from one on the stack costs a load and the
    /// computation where reloading it costs the load alone: one spilled
    /// where its source is on the stack gets a slot, stored after its
    /// definition, when that is in a register.
    fn slot_derived(&mut self) {
        for v in 0..self.vregs.len() {
            let crate::jit::backend::vcode::Remat::AndImm { src, .. } = self.f.remat[v] else {
                continue;
            };
            let vi = &self.vregs[v];
            if !vi.remat || vi.pieces.is_empty() {
                continue;
            }
            let def_in_reg = self.alloc_at(v as u32, vi.def).is_some_and(|a| a.is_reg());
            let src_on_stack = vi.pieces.iter().any(|&x| {
                let p = &self.pieces[x as usize];
                p.loc == Loc::Remat
                    && p.ivs
                        .iter()
                        .any(|&(a, _)| self.alloc_at(src, a).is_some_and(|l| l.is_stack()))
            });
            if def_in_reg && src_on_stack {
                let ps = self.vregs[v].pieces.clone();
                for x in ps {
                    if self.pieces[x as usize].loc == Loc::Remat {
                        self.pieces[x as usize].loc = Loc::Stack;
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
    fn color_slots(&mut self) {
        let mut busy: Vec<(u32, Vec<(Pos, Pos)>)> = Vec::new();
        for g in 0..self.vregs.len() {
            let gi = &self.vregs[g];
            if self.rep[g] != g as u32 || gi.slot == NONE {
                continue;
            }
            let mut ivs: Vec<(Pos, Pos)> = Vec::new();
            for &x in &gi.pieces {
                let p = &self.pieces[x as usize];
                // Stores land in the gaps around: a point either side.
                for &(a, b) in &p.ivs {
                    ivs.push((a.saturating_sub(1), b + 1));
                }
            }
            ivs.sort_unstable_by_key(|iv| iv.0);
            let mut merged: Vec<(Pos, Pos)> = Vec::with_capacity(ivs.len());
            for (a, b) in ivs {
                match merged.last_mut() {
                    Some(m) if a <= m.1 => m.1 = m.1.max(b),
                    _ => merged.push((a, b)),
                }
            }
            busy.push((g as u32, merged));
        }
        busy.sort_by_key(|(_, iv)| iv.first().map_or(0, |x| x.0));
        let mut slots: Vec<Vec<(Pos, Pos)>> = Vec::new();
        for (g, ivs) in busy {
            let k = match slots.iter().position(|s| intersect(s, &ivs, 0).is_none()) {
                Some(k) => k,
                None => {
                    slots.push(Vec::new());
                    slots.len() - 1
                }
            };
            let s = &mut slots[k];
            s.extend_from_slice(&ivs);
            s.sort_unstable_by_key(|iv| iv.0);
            self.vregs[g as usize].slot = k as u32;
        }
        self.num_slots = slots.len() as u32;
    }

    /// The move of `v` to `to` from where it is at `at`: computed again when
    /// it is there (a constant from nothing, a derived value from its
    /// source's location at `at`).
    fn mv(&self, v: u32, at: Pos, to: Allocation, cls: u8) -> Mv {
        let from = self.alloc_at(v, at).expect("live where it moves");
        let (from, remat) = match (from.is_none(), self.f.remat[v as usize]) {
            (true, crate::jit::backend::vcode::Remat::AndImm { src, .. }) => {
                (self.alloc_at(src, at).expect("the source live"), true)
            }
            (true, _) => (from, true),
            (false, _) => (from, false),
        };
        Mv {
            from,
            to,
            cls,
            vreg: v,
            remat,
        }
    }

    /// Moves on every edge whose ends hold a value in different places: on
    /// the successor's side when it has one predecessor (not a resume block,
    /// which a return enters past its moves), else on the predecessor's.
    fn edge_moves(&self, moves: &mut Vec<(u32, u8, Mv)>) -> Result<(), String> {
        let f = self.f;
        for &s in &self.order {
            let preds = f.block_preds(s);
            let si = s.index();
            let resume = f.blocks[si].resume;
            let ss = self.block_start(si as u32);
            for &p in preds {
                let pi = p.index();
                let pe = 2 * self.last[pi] + 1;
                let (at, phase) = if preds.len() == 1 && !resume {
                    (self.first[si], EDGE_IN)
                } else {
                    if f.block_succs(p).len() != 1 {
                        return Err(format!("critical edge b{pi} -> b{si}"));
                    }
                    (self.last[pi], EDGE_OUT)
                };
                let before = moves.len();
                for (k, &w) in self.live_in(si).iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        let v = (k * 64 + w.trailing_zeros() as usize) as u32;
                        w &= w - 1;
                        let from = self.alloc_at(v, pe);
                        let to = self.alloc_at(v, ss);
                        let (Some(from), Some(to)) = (from, to) else {
                            return Err(format!("v{v} not live across b{pi} -> b{si}"));
                        };
                        // The slot has held it since its definition; a
                        // constant is computed again where it is needed.
                        let g = self.rep[v as usize] as usize;
                        if from != to && !to.is_none() && (!to.is_stack() || self.deferred[g]) {
                            let cls = self.vregs[v as usize].class;
                            moves.push((at, phase, self.mv(v, pe, to, cls)));
                        }
                    }
                }
                let k = f
                    .block_succs(p)
                    .iter()
                    .position(|&x| x == s)
                    .expect("an edge");
                let args = f.branch_blockparams(p, self.at_lin[self.last[pi] as usize], k);
                for (q, a) in f.block_params(s).iter().zip(args) {
                    let q = q.vreg() as u32;
                    let Some(to) = self.alloc_at(q, ss) else {
                        continue;
                    };
                    let from = self
                        .alloc_at(a.vreg() as u32, pe)
                        .ok_or_else(|| format!("argument v{} not live at b{pi}'s end", a.vreg()))?;
                    if from != to {
                        let cls = self.vregs[q as usize].class;
                        moves.push((at, phase, self.mv(a.vreg() as u32, pe, to, cls)));
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
        Ok(())
    }

    /// Emit a parallel move as a sequence: a move goes once nothing else
    /// still reads its target; a cycle parks one value in the scratch
    /// register; stack-to-stack goes through the transport register.
    fn sequence(&self, mut ms: Vec<Mv>, mut emit: impl FnMut(Edit)) -> Result<(), String> {
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
            put(Mv {
                from: t,
                to: scratch,
                cls: reader.cls,
                vreg: reader.vreg,
                remat: false,
            });
            for m in &mut ms {
                if m.from == t {
                    m.from = scratch;
                }
            }
        }
        Ok(())
    }
}
