//! The analyses a copy reads, each over its finished input graph: which
//! instructions are live and how often each value is used, and the types
//! and representations of the high-level graph's parameters.

use super::access::ARRAY;
use super::ops::{Flags, Op};
use super::types::{Rep, TypeSet};
use super::{Block, BlockFlags, Graph, Inst, Val};
use super::{arith, compare};
use crate::jit::build::Spec;

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
    /// The loop parameters held as `I64`: integers of either size.
    pub(crate) wide: Vec<Val>,
    block_of: Vec<Block>,
}

/// Whether an instruction stays with no use of its value.
pub(crate) fn has_effect(i: &Inst) -> bool {
    let info = i.op.info();
    !info.writes.is_empty()
        || info
            .flags
            .intersects(Flags::EXITS | Flags::TERMINATOR | Flags::CALLS | Flags::KEPT)
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

    /// The kind of the numeric loop an op is of, as its rule will decide
    /// it (`forloop::kind_of`, `Emitter::for_step`).
    fn loop_kind(&self, g: &Graph, i: &Inst) -> Spec {
        let by = |v: u32| {
            let set = self.ty[v as usize];
            if set.is_empty() {
                None
            } else if TypeSet::SMALL.contains(set) {
                Some(Spec::I32)
            } else if TypeSet::FLOAT.contains(set) {
                Some(Spec::F64)
            } else {
                None
            }
        };
        let speculated = Spec::of(i.aux);
        match i.op {
            Op::ForStart | Op::ForStep => super::forloop::kind_of(by(i.a), by(i.b), speculated),
            // The step's kind is the loop's.
            Op::ForNext => by(i.b).unwrap_or(speculated),
            _ => {
                let r = g.run(i.a, i.b);
                super::forloop::kind_of(by(r[0]), by(r[2]), speculated)
            }
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
                Op::Compare => {
                    if let Some(set) = compare::demand(i.aux) {
                        want(i.a, set);
                        want(i.b, set);
                    }
                }
                // An operand the site saw as one kind is held as it; one it
                // saw only as integers, 64-bit ones among them, as `I64`.
                Op::Arith | Op::Unm => {
                    let n = if i.op == Op::Unm { 1 } else { 2 };
                    for (k, shift) in arith::SEEN_SHIFT.into_iter().enumerate().take(n) {
                        match (i.aux >> shift) as u8 & 15 {
                            arith::SMALL => want(i.args()[k], TypeSet::SMALL),
                            arith::FLOAT => want(i.args()[k], TypeSet::FLOAT),
                            s if s & arith::BIG != 0 && TypeSet::INT.contains(arith::set_of(s)) => {
                                want(i.args()[k], TypeSet::INT)
                            }
                            _ => {}
                        }
                    }
                }
                Op::GetIndex if i.aux & ARRAY != 0 => want(i.b, TypeSet::SMALL),
                Op::SetIndex if i.aux & ARRAY != 0 => want(g.run(i.a, i.b)[1], TypeSet::SMALL),
                Op::GetIndex | Op::SetIndex => {}
                // Only number demands are resolved: a receiver's is none.
                Op::GetField | Op::SetField | Op::Concat | Op::Len => {}
                // An integer loop takes its operands as small integers; a
                // float loop converts what it is given.
                // The counter is held as the loop's kind.
                Op::ForMore => want(g.run(i.a, i.b)[0], Spec::of(i.aux).set()),
                Op::ForNext => want(i.a, Spec::of(i.aux).set()),
                Op::ForStart | Op::ForStep | Op::ForLimit | Op::ForSkip => {
                    if Spec::of(i.aux) == Spec::I32 {
                        g.uses(i, |u| {
                            if g.rep(u) == Rep::Val {
                                want(u.0, TypeSet::SMALL)
                            }
                        });
                    }
                }
                // No other op asks a type of an operand it would check.
                Op::K32
                | Op::K64
                | Op::KF64
                | Op::KObj
                | Op::Load
                | Op::Store
                | Op::LoadIdx
                | Op::StoreIdx
                | Op::Base
                | Op::Thread
                | Op::SlotLoad
                | Op::SlotStore
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::And
                | Op::Or
                | Op::Xor
                | Op::Not
                | Op::Neg
                | Op::Shl
                | Op::Shr
                | Op::Sar
                | Op::SDiv
                | Op::UDiv
                | Op::ZExt32
                | Op::SExt32
                | Op::Trunc32
                | Op::Cmp
                | Op::Select
                | Op::AddOvf
                | Op::SubOvf
                | Op::MulOvf
                | Op::NegOvf
                | Op::FAdd
                | Op::FSub
                | Op::FMul
                | Op::FDiv
                | Op::FNeg
                | Op::FAbs
                | Op::FSqrt
                | Op::FFloor
                | Op::FCeil
                | Op::FCmp
                | Op::IToF
                | Op::LToF
                | Op::FToI
                | Op::FToL
                | Op::BoxI32
                | Op::BoxF64
                | Op::BoxPtr
                | Op::UnboxF64
                | Op::UnboxPtr
                | Op::Guard
                | Op::GuardNot
                | Op::Proved
                | Op::Deopt
                | Op::LazyExit
                | Op::FrameState
                | Op::Param
                | Op::Jump
                | Op::Br
                | Op::Ret
                | Op::CallHelper
                | Op::IsType
                | Op::BoxI64
                | Op::UnboxI64
                | Op::ToF64
                | Op::DivFloor
                | Op::ModFloor
                | Op::LuaShl
                | Op::LuaShr
                | Op::FToIExact
                | Op::FToLExact
                | Op::LToI
                | Op::AbsOvf
                | Op::FIDiv
                | Op::FMod
                | Op::Barrier
                | Op::NewTable
                | Op::GcCheck
                | Op::Truthy => {}
            }
        }
        for v in 0..n {
            // Only a type a parameter is held unboxed for is checked on entry.
            let held =
                matches!(demand[v].unboxed(), Rep::I32 | Rep::F64) || demand[v] == TypeSet::INT;
            if !pending(v) || !every[v] || !held {
                demand[v] = TypeSet::ANY;
            }
        }
        // Types forward under the demands, dropping a demand a value coming
        // around does not meet, until none drops.
        loop {
            // A parameter nothing holds unboxed has the types its inputs
            // have: from nothing, growing, until nothing grows.
            self.ty.clear();
            self.ty.resize(n, TypeSet::empty());
            let mut grew = true;
            while std::mem::take(&mut grew) {
                for v in 0..n {
                    let i = &g.insts[v];
                    let was = self.ty[v];
                    self.ty[v] = match i.op {
                        Op::Param if pending(v) && demand[v] != TypeSet::ANY => demand[v],
                        Op::Param if i.rep == Rep::Val => (g.inputs(Val(v as u32)).iter())
                            .filter(|&&x| x != u32::MAX)
                            .fold(TypeSet::empty(), |s, &x| s | self.ty[x as usize]),
                        Op::Arith if g.facts[v].pending() => {
                            arith::result(i.aux, self.ty[i.a as usize], Some(self.ty[i.b as usize]))
                        }
                        Op::Unm if g.facts[v].pending() => {
                            arith::result(i.aux, self.ty[i.a as usize], None)
                        }
                        Op::ForStart | Op::ForStep | Op::ForLimit | Op::ForNext
                            if g.facts[v].pending() =>
                        {
                            self.loop_kind(g, i).set()
                        }
                        Op::Param | Op::Arith | Op::Unm => g.facts[v].set,
                        // Every other op's type is its fact's.
                        Op::K32
                        | Op::K64
                        | Op::KF64
                        | Op::KObj
                        | Op::Load
                        | Op::Store
                        | Op::LoadIdx
                        | Op::StoreIdx
                        | Op::Base
                        | Op::Thread
                        | Op::SlotLoad
                        | Op::SlotStore
                        | Op::Add
                        | Op::Sub
                        | Op::Mul
                        | Op::And
                        | Op::Or
                        | Op::Xor
                        | Op::Not
                        | Op::Neg
                        | Op::Shl
                        | Op::Shr
                        | Op::Sar
                        | Op::SDiv
                        | Op::UDiv
                        | Op::ZExt32
                        | Op::SExt32
                        | Op::Trunc32
                        | Op::Cmp
                        | Op::Select
                        | Op::AddOvf
                        | Op::SubOvf
                        | Op::MulOvf
                        | Op::NegOvf
                        | Op::FAdd
                        | Op::FSub
                        | Op::FMul
                        | Op::FDiv
                        | Op::FNeg
                        | Op::FAbs
                        | Op::FSqrt
                        | Op::FFloor
                        | Op::FCeil
                        | Op::FCmp
                        | Op::IToF
                        | Op::LToF
                        | Op::FToI
                        | Op::FToL
                        | Op::BoxI32
                        | Op::BoxF64
                        | Op::BoxPtr
                        | Op::UnboxF64
                        | Op::UnboxPtr
                        | Op::Guard
                        | Op::GuardNot
                        | Op::Proved
                        | Op::Deopt
                        | Op::LazyExit
                        | Op::FrameState
                        | Op::Jump
                        | Op::Br
                        | Op::Ret
                        | Op::CallHelper
                        | Op::CheckType
                        | Op::IsType
                        | Op::BoxI64
                        | Op::UnboxI64
                        | Op::ToF64
                        | Op::DivFloor
                        | Op::ModFloor
                        | Op::LuaShl
                        | Op::LuaShr
                        | Op::FToIExact
                        | Op::FToLExact
                        | Op::LToI
                        | Op::AbsOvf
                        | Op::FIDiv
                        | Op::FMod
                        | Op::Barrier
                        | Op::NewTable
                        | Op::GcCheck
                        | Op::Compare
                        | Op::Truthy
                        | Op::GetIndex
                        | Op::SetIndex
                        | Op::GetField
                        | Op::SetField
                        | Op::Concat
                        | Op::Len
                        | Op::ForStart
                        | Op::ForStep
                        | Op::ForLimit
                        | Op::ForSkip
                        | Op::ForMore
                        | Op::ForNext => g.facts[v].set,
                    };
                    grew |= self.ty[v] != was;
                }
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
        self.wide.clear();
        for (v, &want) in demand.iter().enumerate() {
            if pending(v) && want == TypeSet::INT {
                self.wide.push(Val(v as u32));
            }
            if pending(v) && want != TypeSet::ANY {
                let entry = g.inputs(Val(v as u32))[0] as usize;
                if !want.contains(self.ty[entry]) {
                    self.guarded.push(Val(v as u32));
                }
            }
        }
    }
}
