//! After the scan: each operand's allocation and the moves between pieces,
//! across edges, into fixed registers, and the stores after definitions,
//! each gap's moves sequenced as one parallel move.

use regalloc2::{Allocation, Edit, Function, Output, ProgPoint};

use super::{Loc, NONE, Need, Ra};
use crate::jit::backend::vcode::TargetInst;

/// A move of a value of class `cls`.
#[derive(Clone, Copy, Debug)]
struct Mv {
    from: Allocation,
    to: Allocation,
    cls: u8,
}

/// The order of the moves sharing a gap: stores after the definition
/// before it, then moves between pieces, across the edge into the block,
/// stores of its parameters, across the edges out of it (a block of one
/// instruction has both), and copies into the instruction's fixed
/// registers.
const DEF_STORE: u8 = 0;
const CONNECT: u8 = 1;
const EDGE_IN: u8 = 2;
const PARAM_STORE: u8 = 3;
const EDGE_OUT: u8 = 4;
const FIXED_COPY: u8 = 5;

impl<I: TargetInst> Ra<'_, I> {
    pub(super) fn resolve(&mut self) -> Result<Output, String> {
        for x in 0..self.pieces.len() as u32 {
            let p = &self.pieces[x as usize];
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
        // The stores after definitions: a spilled group's slot holds each
        // member from its definition on.
        for v in 0..self.vregs.len() as u32 {
            let g = &self.vregs[self.rep[v as usize] as usize];
            if g.slot == NONE || g.pieces.is_empty() {
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
            // Between pieces inside a block; a stack piece needs no store.
            for w in gi.pieces.windows(2) {
                let (a, b) = (&self.pieces[w[0] as usize], &self.pieces[w[1] as usize]);
                let at = b.start();
                if a.end() != at || b.loc == Loc::Stack || a.loc == b.loc {
                    continue;
                }
                let bb = self.block_of_pos(at);
                if at == self.block_start(bb) {
                    continue;
                }
                if at & 1 != 0 {
                    return Err(format!("v{g} reloaded inside instruction {}", at / 2));
                }
                moves.push((
                    at / 2,
                    CONNECT,
                    Mv {
                        from: self.alloc_of(w[0]),
                        to: self.alloc_of(w[1]),
                        cls,
                    },
                ));
            }
            // Into fixed registers, from wherever the value is when the
            // instruction starts.
            for &x in &gi.pieces {
                for u in &self.pieces[x as usize].uses {
                    if let Need::Fixed(r) = u.need {
                        let n = u.pos / 2;
                        let from = self.alloc_at(g, 2 * n).expect("live at its use");
                        let to = Allocation::reg(r);
                        if from != to {
                            moves.push((n, FIXED_COPY, Mv { from, to, cls }));
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
            debug_locations: Vec::new(),
            stats: Default::default(),
        })
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
                        // The slot has held it since its definition.
                        if from != to && !to.is_stack() {
                            let cls = self.vregs[v as usize].class;
                            moves.push((at, phase, Mv { from, to, cls }));
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
                        moves.push((at, phase, Mv { from, to, cls }));
                    }
                }
                if resume && moves.len() != before {
                    return Err(format!("moves into resume block b{si}"));
                }
            }
        }
        Ok(())
    }

    /// Emit a parallel move as a sequence: a move goes once nothing else
    /// still reads its target; a cycle parks one value in the scratch
    /// register; stack-to-stack goes through the transport register.
    fn sequence(&self, mut ms: Vec<Mv>, mut emit: impl FnMut(Edit)) -> Result<(), String> {
        ms.retain(|m| m.from != m.to);
        let transport = Allocation::reg(self.env.transport);
        let mut put = |m: Mv| {
            if m.from.is_stack() && m.to.is_stack() {
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
