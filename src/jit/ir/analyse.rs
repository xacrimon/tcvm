//! The analyses a copy reads, each over its finished input graph: which
//! instructions are live and how often each value is used, and the types
//! and representations of the high-level graph's parameters.

use super::ops::{Flags, Op};
use super::types::{Rep, TypeSet};
use super::{Block, BlockFlags, Graph, Inst, Val};
use crate::jit::build::{SPEC_SHIFT, Spec};

#[derive(Default)]
pub(crate) struct Analysis {
    /// Uses of each value by live instructions, saturating; an instruction
    /// is live when it has an effect or a use.
    pub(crate) uses: Vec<u8>,
    work: Vec<u32>,
    /// High-level graphs: the type set each value has once the loops'
    /// parameters are resolved.
    pub(crate) ty: Vec<TypeSet>,
    /// The loop parameters that get their type from a check on the loop's
    /// entry edge.
    pub(crate) guarded: Vec<Val>,
    block_of: Vec<Block>,
}

/// Whether an instruction stays with no use of its value.
pub(crate) fn has_effect(i: &Inst) -> bool {
    let info = i.op.info();
    !info.writes.is_empty()
        || info
            .flags
            .intersects(Flags::EXITS | Flags::TERMINATOR | Flags::CALLS)
}

impl Analysis {
    pub(crate) fn live(&self, v: Val, i: &Inst) -> bool {
        self.uses[v.idx()] != 0 || has_effect(i)
    }

    /// Liveness and use counts.
    pub(crate) fn uses(&mut self, g: &Graph) {
        self.uses.clear();
        self.uses.resize(g.insts.len(), 0);
        self.work.clear();
        self.work
            .extend((0..g.insts.len() as u32).filter(|&v| has_effect(&g.insts[v as usize])));
        // A loop's entry state is where what copy 1 puts on its entry edge
        // leaves to.
        for l in &g.loops {
            self.uses[l.entry_state.idx()] = 1;
            self.work.push(l.entry_state.0);
        }
        while let Some(v) = self.work.pop() {
            let i = &g.insts[v as usize];
            g.uses(i, |u| {
                let n = &mut self.uses[u.idx()];
                if *n == 0 && !has_effect(g.inst(u)) {
                    self.work.push(u.0);
                }
                *n = n.saturating_add(1);
            });
        }
    }

    /// The kind a generic op takes operand `k` as, given its type.
    fn kind(&self, i: &Inst, k: usize) -> Spec {
        let set = self.ty[i.args()[k] as usize];
        if !set.is_empty() && TypeSet::SMALL.contains(set) {
            Spec::I32
        } else if !set.is_empty() && TypeSet::FLOAT.contains(set) {
            Spec::F64
        } else {
            Spec::of(i.aux >> SPEC_SHIFT[k])
        }
    }

    /// Resolve the loop parameters of a high-level graph, once: a parameter
    /// takes the type its uses demand when they agree, one of them runs on
    /// every iteration, and every value coming around the loop has it; its
    /// entry value is then checked on the entry edge unless it has the type
    /// already. Demands only drop, so this ends.
    pub(crate) fn types(&mut self, g: &Graph) {
        let n = g.insts.len();
        self.block_of.clear();
        self.block_of.resize(n, Block::NONE);
        for &b in &g.order {
            let d = g.block(b);
            self.block_of[d.start as usize..d.end as usize].fill(b);
        }
        // What each pending parameter's uses demand: empty for none yet,
        // `ANY` once they disagree or nothing demands on every iteration.
        let mut demand = vec![TypeSet::empty(); n];
        let mut every = vec![false; n];
        // A loop's parameter; one that only passes a loop's value on is typed
        // by what it passes.
        let pending = |v: usize| {
            let header = g
                .block(self.block_of[v])
                .flags
                .contains(BlockFlags::LOOP_HEADER);
            g.insts[v].op == Op::Param && g.facts[v].pending() && header
        };
        for v in 0..n {
            let i = &g.insts[v];
            if !self.live(Val(v as u32), i) {
                continue;
            }
            let b = self.block_of[v];
            let mut want = |p: u32, set: TypeSet| {
                if !pending(p as usize) {
                    return;
                }
                let d = &mut demand[p as usize];
                *d = if d.is_empty() || *d == set {
                    set
                } else {
                    TypeSet::ANY
                };
                let header = self.block_of[p as usize];
                let hd = g.block(header);
                every[p as usize] |= g.preds(header)[hd.nfwd as usize..]
                    .iter()
                    .all(|&latch| g.dominates(b, latch));
            };
            match i.op {
                Op::CheckType => want(i.a, TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits())),
                Op::Arith | Op::Compare => {
                    for (k, shift) in SPEC_SHIFT.into_iter().enumerate() {
                        want(i.args()[k], Spec::of(i.aux >> shift).set());
                    }
                }
                Op::Unm => want(i.a, Spec::of(i.aux >> SPEC_SHIFT[0]).set()),
                _ => {}
            }
        }
        for v in 0..n {
            // Only a type a parameter is held unboxed for is checked on entry.
            if !pending(v) || !every[v] || !matches!(demand[v].unboxed(), Rep::I32 | Rep::F64) {
                demand[v] = TypeSet::ANY;
            }
        }
        // Types forward under the demands, dropping a demand a value coming
        // around does not meet, until none drops.
        self.ty.clear();
        self.ty.resize(n, TypeSet::ANY);
        loop {
            for v in 0..n {
                let i = &g.insts[v];
                self.ty[v] = match i.op {
                    Op::Param if pending(v) => demand[v],
                    Op::Param if i.rep == Rep::Val => (g.inputs(Val(v as u32)).iter())
                        .fold(TypeSet::empty(), |s, &x| s | self.ty[x as usize]),
                    Op::Arith if g.facts[v].pending() => {
                        let (a, b) = (self.kind(i, 0), self.kind(i, 1));
                        if crate::jit::build::int_result(i.aux & 15, a, b) {
                            TypeSet::SMALL
                        } else {
                            TypeSet::FLOAT
                        }
                    }
                    Op::Unm if g.facts[v].pending() => self.kind(i, 0).set(),
                    _ => g.facts[v].set,
                };
            }
            let mut dropped = false;
            for v in 0..n {
                if !pending(v) || demand[v] == TypeSet::ANY {
                    continue;
                }
                let nfwd = g.block(self.block_of[v]).nfwd as usize;
                let around = &g.inputs(Val(v as u32))[nfwd..];
                if nfwd != 1
                    || around
                        .iter()
                        .any(|&x| !demand[v].contains(self.ty[x as usize]))
                {
                    demand[v] = TypeSet::ANY;
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
        }
        self.guarded.clear();
        for (v, &want) in demand.iter().enumerate() {
            if pending(v) && want != TypeSet::ANY {
                let entry = g.inputs(Val(v as u32))[0] as usize;
                if !want.contains(self.ty[entry]) {
                    self.guarded.push(Val(v as u32));
                }
            }
        }
    }
}
