//! The emitter: the one way an instruction enters a graph. The builder and
//! every copy go through `emit`, which checks the operands against the op
//! table, applies the rules (`rules`), returns a dominating identical
//! instruction where there is one, and otherwise appends and records what
//! the instruction makes known.

use super::fields::Field;
use super::ops::{
    Arg, Aux, Cc, Ctx, Dialect, ExitKind, ExitTag, FactOf, Flags, Heaps, Helper, HelperFact, Mem,
    Op, SET_ARM,
};
use super::table::{self, Key, Snap, Table};
use super::types::{Fact, FactFlags, Rep, TypeSet};
use super::{Block, BlockData, BlockFlags, Graph, Inst, Loop, LoopData, Val};
use crate::jit::bug;
use crate::jit::compile::Bail;

const NONE: u32 = u32::MAX;

/// What of `aux` tells two instructions apart: not an exit's tag, which
/// says why it would leave, not what it computes.
fn aux_key(i: &Inst) -> u16 {
    match i.op.info().aux {
        Aux::Tag => 0,
        Aux::SetTag => i.aux & TypeSet::ANY.bits(),
        Aux::No
        | Aux::Cc
        | Aux::FCc
        | Aux::Set
        | Aux::SetArm
        | Aux::Slot
        | Aux::Field
        | Aux::Kind
        | Aux::Exit
        | Aux::Helper
        | Aux::Array => i.aux,
    }
}

/// A heap key's index among the versions: the heaps by their bit, then
/// the frame slots, then the indexed fields' slots.
const SLOT_KEYS: u32 = 16;
const FIELD_KEYS: u32 = SLOT_KEYS + 256;

/// A memory access as the number table keys it: its kind, field or slot,
/// base and index, and for a store the value it leaves there. A load and
/// the store that feeds it have one key.
fn mem_key(i: &Inst) -> Option<([u32; 4], Option<u32>)> {
    Some(match i.op.info().mem {
        Mem::No => return None,
        Mem::SlotLoad => ([0, i.aux as u32, 0, 0], None),
        Mem::SlotStore => ([0, i.aux as u32, 0, 0], Some(i.a)),
        Mem::Load => ([1, i.aux as u32, i.a, i.c], None),
        Mem::Store => ([1, i.aux as u32, i.a, i.c], Some(i.b)),
        Mem::LoadIdx => ([2, i.aux as u32, i.a, i.b], None),
        Mem::StoreIdx => ([2, i.aux as u32, i.a, i.b], Some(i.c)),
    })
}

/// The heap a memory access is in, and its key's index where the heap is
/// keyed.
fn heap_of(i: &Inst) -> (Heaps, Option<u32>) {
    match i.op.info().mem {
        Mem::No => bug!("{:?} is no memory access", i.op),
        Mem::SlotLoad | Mem::SlotStore => (Heaps::SLOT, Some(SLOT_KEYS + i.aux as u32)),
        Mem::Load | Mem::Store | Mem::LoadIdx | Mem::StoreIdx => {
            let f = Field::info(i.aux);
            let spilled = (i.aux == Field::Spilled as u16) as u32;
            let keyed = f.indexed && !f.heap.is_empty();
            (f.heap, keyed.then(|| FIELD_KEYS + 2 * i.c + spilled))
        }
    }
}

/// How many arms one join takes.
const MAX_ARMS: usize = 4;

/// A block a branch leads to, before it is opened (`Emitter::arm`).
#[derive(Clone, Copy)]
pub(crate) struct Arm {
    block: Block,
    lp: Loop,
    cold: bool,
}

/// A continuation before it is opened, with the values each arm that left
/// for it brings.
pub(crate) struct Join<const N: usize> {
    block: Block,
    lp: Loop,
    arms: [[Val; N]; MAX_ARMS],
    n: usize,
    /// An arm left from a loop inside the continuation's.
    from_loop: bool,
}

/// `reset` before the first graph, and between graphs.
#[derive(Default)]
pub(crate) struct Emitter {
    pub(crate) g: Graph,
    pub(crate) table: Table,
    /// Each sealed block's end state.
    ends: Vec<Snap>,
    /// Value numbers, open-addressed: an instruction, its block, and for a
    /// memory access the versions of its heap and key it holds at; empty
    /// where the instruction is `NONE`. An entry whose block does not
    /// dominate, or whose versions have passed, is stale and gets
    /// overwritten.
    num: Vec<[u32; 4]>,
    num_len: usize,
    /// The last heap version handed out.
    version: u32,
    /// Edges into blocks not opened yet: each with the next of its target's,
    /// newest first from `pending[target]`.
    edges: Vec<(Block, u32)>,
    pending: Vec<u32>,
    /// The open block.
    pub(crate) cur: Block,
    /// The output is low-level: a high-level op with no lowering is a bug.
    pub(crate) lowered: bool,
    /// Sticky: once set, `emit` appends nothing and its results mean
    /// nothing; the caller stops at its next block.
    pub(crate) bail: Option<Bail>,
    /// The runtime's address, which helpers take (`call`), and that of
    /// the arena's metrics, where the collector's counters are.
    pub(crate) rt: u64,
    pub(crate) metrics: u64,
    scratch: Vec<Snap>,
    /// A call's operand list while `call` builds it.
    list: Vec<Val>,
}

impl Default for Block {
    fn default() -> Block {
        Block::NONE
    }
}

impl Emitter {
    /// Start a graph over, keeping the allocations.
    pub(crate) fn reset(&mut self, lowered: bool) {
        self.g.reset();
        self.table.reset();
        self.ends.clear();
        self.num.clear();
        self.num.resize(256, [NONE; 4]);
        self.version = 0;
        self.num_len = 0;
        self.edges.clear();
        self.pending.clear();
        self.cur = Block::NONE;
        self.lowered = lowered;
        self.bail = None;
    }

    pub(crate) fn bail(&mut self, why: Bail) {
        self.bail.get_or_insert(why);
    }

    // --- blocks -------------------------------------------------------------

    pub(crate) fn new_block(&mut self) -> Block {
        self.g.blocks.push(BlockData {
            start: NONE,
            end: NONE,
            preds: 0,
            npreds: 0,
            nfwd: 0,
            idom: Block::NONE,
            jump: Block::NONE,
            depth: 0,
            lp: Loop::NONE,
            flags: BlockFlags::empty(),
        });
        self.pending.push(NONE);
        self.ends.push(Snap::ROOT);
        Block(self.g.blocks.len() as u32 - 1)
    }

    pub(crate) fn new_loop(
        &mut self,
        header: Block,
        parent: Loop,
        writes: Heaps,
        entry_state: Val,
    ) -> Loop {
        self.g.loops.push(LoopData {
            header,
            parent,
            writes,
            entry_state,
        });
        Loop(self.g.loops.len() as u32 - 1)
    }

    /// Open `b` in loop `lp`, after every forward predecessor's terminator.
    /// `late` predecessors come after it: a loop header's back edges, a
    /// continuation's cold arms. False for a block nothing reaches, which
    /// stays unopened.
    pub(crate) fn begin(&mut self, b: Block, lp: Loop, late: u32, flags: BlockFlags) -> bool {
        if !self.cur.is_none() || self.g.block(b).start != NONE {
            bug!("{b:?} opened twice or inside {:?}", self.cur);
        }
        let g = &mut self.g;
        let at = g.preds.len();
        let mut e = self.pending[b.idx()];
        while e != NONE {
            let (from, next) = self.edges[e as usize];
            g.preds.push(from);
            e = next;
        }
        g.preds[at..].reverse();
        let nfwd = g.preds.len() - at;
        let entry = g.order.is_empty();
        if nfwd == 0 && !entry {
            return false;
        }
        let (idom, depth, jump) = if entry {
            (b, 0, b)
        } else {
            let mut idom = g.preds[at];
            for k in at + 1..at + nfwd {
                idom = g.common_dom(idom, g.preds[k]);
            }
            // Skew-binary jumps: an ancestor query takes logarithmic steps.
            let (p, j) = (g.block(idom), g.block(g.block(idom).jump));
            let jump = if p.depth - j.depth == j.depth - g.block(j.jump).depth {
                j.jump
            } else {
                idom
            };
            (idom, p.depth + 1, jump)
        };
        self.scratch.clear();
        (self.scratch).extend(g.preds[at..].iter().map(|p| self.ends[p.idx()]));
        g.preds.resize(at + nfwd + late as usize, Block::NONE);
        let start = g.insts.len() as u32;
        let d = &mut g.blocks[b.idx()];
        (d.start, d.preds, d.nfwd, d.npreds) = (start, at as u32, nfwd as u32, nfwd as u32 + late);
        (d.idom, d.depth, d.jump, d.lp, d.flags) = (idom, depth, jump, lp, flags);
        g.order.push(b);
        self.cur = b;
        self.table.start(&self.scratch);
        // What every predecessor knows of a value stays known.
        for i in 0..self.table.merges() {
            let (key, at_preds) = self.table.merge(i);
            if key.is_reg() {
                continue;
            }
            // A value the predecessors' branches typed differently has no
            // one branch to name.
            if key.is_proof() {
                let v = at_preds[0];
                let same = at_preds.iter().all(|&w| w == v);
                self.table.set(key, if same { v } else { table::NONE });
                continue;
            }
            if !key.is_fact() {
                // A heap the predecessors left at different versions is at
                // a new one: nothing known of it survives the join.
                let v = at_preds[0];
                let v = if at_preds.iter().all(|&w| w == v) {
                    v
                } else {
                    self.version += 1;
                    self.version
                };
                self.table.set(key, v);
                continue;
            }
            let def = self.g.facts[key.index() as usize];
            let of = |w: u32| {
                if w == table::NONE {
                    def
                } else {
                    Fact::from_bits(w)
                }
            };
            let f = (at_preds[1..].iter()).fold(of(at_preds[0]), |f, &w| f.join(of(w)));
            self.table
                .set(key, if f == def { table::NONE } else { f.bits() });
        }
        // What the loop's body may write is unknown from its header on.
        if flags.contains(BlockFlags::LOOP_HEADER) {
            let writes = self.g.loops[lp.idx()].writes;
            self.kill(writes);
        }
        // A branch's condition is known on its edges. The fact is of an SSA
        // value, so one forward predecessor is enough: back edges come from
        // where it already held.
        if nfwd == 1 {
            let t = *self.g.term(self.g.preds[at]);
            if t.op == Op::Br && t.b != t.c {
                self.assume(Val(t.a), t.b == b.0, true);
            }
        }
        true
    }

    /// End the graph: a late predecessor that never came (its block left
    /// some other way) goes from its block and the block's parameters.
    pub(crate) fn finish(&mut self) {
        let g = &mut self.g;
        for k in 0..g.order.len() {
            let d = &mut g.blocks[g.order[k].idx()];
            let preds = &g.preds[d.preds as usize..(d.preds + d.npreds) as usize];
            let came = preds.iter().take_while(|p| !p.is_none()).count() as u32;
            if came == d.npreds {
                continue;
            }
            d.npreds = came;
            for i in &mut g.insts[d.start as usize..d.end as usize] {
                if i.op != Op::Param {
                    break;
                }
                i.b = came;
            }
        }
    }

    /// A parameter of the open block, before its other instructions:
    /// `inputs` by forward predecessor, the late ones set as their edges are
    /// emitted (`next_pred`, `set_input`). Without `fact`, its fact is what
    /// holds of every input.
    pub(crate) fn param(&mut self, rep: Rep, fact: Option<Fact>, inputs: &[Val]) -> Val {
        let d = self.g.block(self.cur);
        let (nfwd, npreds) = (d.nfwd as usize, d.npreds as usize);
        let params_only = self.g.insts[d.start as usize..]
            .iter()
            .all(|i| i.op == Op::Param);
        if inputs.len() != nfwd || !params_only || rep == Rep::None {
            bug!("a parameter of {:?} out of place", self.cur);
        }
        let mut joined: Option<Fact> = None;
        for (k, &v) in inputs.iter().enumerate() {
            if self.g.rep(v) != rep {
                bug!("parameter input {v:?} is not {rep:?}");
            }
            let w = self.table.pred_value(Key::fact(v.0), k);
            let f = if w == table::NONE {
                self.g.facts[v.idx()]
            } else {
                Fact::from_bits(w)
            };
            joined = Some(joined.map_or(f, |j| j.join(f)));
        }
        let at = self.g.extra.len();
        self.g.extra.extend(inputs.iter().map(|v| v.0));
        self.g.extra.resize(at + npreds, NONE);
        let fact = fact.or(joined).unwrap_or(Fact::of_rep(rep));
        let i = Inst {
            op: Op::Param,
            rep,
            aux: 0,
            a: at as u32,
            b: npreds as u32,
            c: 0,
        };
        self.push(i, fact)
    }

    /// The predecessor index the next edge into the open block `b` takes.
    pub(crate) fn next_pred(&self, b: Block) -> usize {
        let preds = self.g.preds(b);
        (preds.iter().position(|p| p.is_none()))
            .unwrap_or_else(|| bug!("{b:?} has no late predecessor left"))
    }

    /// A parameter's input from late predecessor `k`: written once.
    pub(crate) fn set_input(&mut self, param: Val, k: usize, v: Val) {
        let i = *self.g.inst(param);
        let slot = &mut self.g.extra[i.a as usize + k];
        if i.op != Op::Param || *slot != NONE || self.g.insts[v.idx()].rep != i.rep {
            bug!("input {k} of {param:?} set twice or to another representation");
        }
        *slot = v.0;
    }

    fn edge(&mut self, to: Block) {
        if self.g.block(to).start == NONE {
            self.edges.push((self.cur, self.pending[to.idx()]));
            self.pending[to.idx()] = self.edges.len() as u32 - 1;
        } else {
            let k = self.next_pred(to);
            let at = self.g.block(to).preds as usize + k;
            self.g.preds[at] = self.cur;
        }
    }

    // --- facts --------------------------------------------------------------

    /// What is known of `v` here.
    #[inline]
    pub(crate) fn fact(&self, v: Val) -> Fact {
        match self.table.get(Key::fact(v.0)) {
            table::NONE => self.g.facts[v.idx()],
            w => Fact::from_bits(w),
        }
    }

    pub(super) fn refine(&mut self, v: Val, with: Fact) {
        let was = self.fact(v);
        let f = was.meet(with);
        if f != was {
            self.table.set(Key::fact(v.0), f.bits());
        }
    }

    /// The `B1` `c` is `truth` from here on, and with it what it tested;
    /// `branch` where an edge of a branch on it says so, which is then what
    /// types the tested value (`typed`).
    fn assume(&mut self, c: Val, truth: bool, branch: bool) {
        let flag = if truth {
            FactFlags::TRUE
        } else {
            FactFlags::FALSE
        };
        self.refine(
            c,
            Fact {
                set: TypeSet::ANY,
                flags: flag,
            },
        );
        let i = *self.g.inst(c);
        if i.op == Op::IsType {
            let set = TypeSet::from_bits_retain(i.aux);
            let set = if truth { set } else { !set };
            self.refine(Val(i.a), Fact::of(set));
            if branch {
                let arm = if truth { 0 } else { table::PROOF_FALSE };
                self.table.set(Key::proof(i.a), c.0 | arm);
            }
        }
    }

    /// Whether the `B1` is known here: a constant, or tested on the way.
    pub(crate) fn known(&self, c: Val) -> Option<bool> {
        let i = self.g.inst(c);
        if i.op == Op::K32 {
            return Some(i.a != 0);
        }
        let f = self.fact(c).flags;
        if f.contains(FactFlags::TRUE) {
            Some(true)
        } else if f.contains(FactFlags::FALSE) {
            Some(false)
        } else {
            None
        }
    }

    /// The integer a constant holds: an `I32` sign-extended, a `B1` 0 or 1.
    pub(crate) fn ki(&self, v: Val) -> Option<i64> {
        let i = self.g.inst(v);
        match (i.op, i.rep) {
            (Op::K32, Rep::I32) => Some(i.a as i32 as i64),
            (Op::K32, _) => Some(i.a as i64),
            (Op::K64, _) => Some(i.k64() as i64),
            _ => None,
        }
    }

    pub(crate) fn kf(&self, v: Val) -> Option<f64> {
        let i = self.g.inst(v);
        (i.op == Op::KF64).then(|| f64::from_bits(i.k64()))
    }

    // --- emission -----------------------------------------------------------

    pub(crate) fn emit(&mut self, op: Op, rep: Rep, aux: u16, a: u32, b: u32, c: u32) -> Val {
        if self.bail.is_some() {
            return Val(0);
        }
        let mut i = Inst {
            op,
            rep,
            aux,
            a,
            b,
            c,
        };
        self.check(&mut i);
        #[cfg(feature = "tcvm-dev")]
        let at = (self.cur, self.g.blocks.len());
        let v = match self.rule(&mut i) {
            Some(v) => v,
            None => self.append(i),
        };
        // A rule makes no block and leaves the open one open, or closed.
        #[cfg(feature = "tcvm-dev")]
        if at.1 != self.g.blocks.len() || !(self.cur == at.0 || self.cur.is_none()) {
            bug!("the rule of {op:?} left {:?} for {:?}", at.0, self.cur);
        }
        v
    }

    /// `emit`, and for an op its rule leaves high-level, its rewrite: the
    /// lowering that needs blocks. Returns in the continuation block.
    pub(crate) fn lower(&mut self, op: Op, rep: Rep, aux: u16, a: u32, b: u32, c: u32) -> Val {
        if self.bail.is_some() {
            return Val(0);
        }
        let mut i = Inst {
            op,
            rep,
            aux,
            a,
            b,
            c,
        };
        self.check(&mut i);
        if let Some(v) = self.rule(&mut i) {
            return v;
        }
        match i.op {
            Op::GetIndex | Op::SetIndex => self.index(&i),
            Op::Barrier => self.barrier_rewrite(&i),
            Op::BoxI64 => self.box_rewrite(&i),
            Op::UnboxI64 => self.unbox_rewrite(&i),
            Op::ToF64 => self.float_rewrite(&i),
            // No rewrite: a rule lowers each, or it is not lowered yet.
            Op::CheckType
            | Op::IsType
            | Op::NewTable
            | Op::GcCheck
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
            | Op::Arith
            | Op::Unm
            | Op::Compare
            | Op::Truthy => self.append(i),
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
            | Op::CallHelper => self.append(i),
        }
    }

    /// Number and append, past the rules.
    pub(super) fn append(&mut self, i: Inst) -> Val {
        if self.lowered && i.op.info().dialect == Dialect::H {
            bug!("high-level {:?} in a low-level graph", i.op);
        }
        let info = i.op.info();
        let mem = mem_key(&i);
        let numbered = i.op.numbered() || mem.is_some();
        let mut slot = 0;
        if numbered {
            match (self.number(&i), mem) {
                // A store of what is there already is nothing.
                (Ok(v), Some((_, Some(stored)))) if v.0 == stored => return Val::NONE,
                (Ok(v), Some((_, None)) | None) => return v,
                _ => {}
            }
        }
        // A write makes its heap's other contents unknown; a store then
        // records what it left.
        if let Some((_, Some(_))) = mem {
            match heap_of(&i) {
                (_, Some(key)) => {
                    self.version += 1;
                    self.table.set(Key::heap(key), self.version);
                }
                (heap, None) => self.kill(heap),
            }
        } else if !info.writes.is_empty() {
            self.kill(info.writes);
        } else if i.op == Op::CallHelper {
            // An allocation moves the collector's counter.
            let h = Helper::of(i.aux).info();
            let gc = if h.flags.contains(Flags::ALLOCS) {
                Heaps::GC
            } else {
                Heaps::empty()
            };
            self.kill(h.writes | gc);
        } else if info.flags.contains(Flags::ALLOCS) {
            self.kill(Heaps::GC);
        }
        if numbered && let Err(s) = self.number(&i) {
            slot = s;
        }
        let fact = self.def_fact(&i);
        let v = self.push(i, fact);
        if numbered {
            if self.num[slot][0] == NONE {
                self.num_len += 1;
            }
            let (heap, key) = self.versions(&i);
            self.num[slot] = [v.0, self.cur.0, heap, key];
            if self.num_len * 2 > self.num.len() {
                self.grow();
            }
        }
        // What standing past it tells.
        match i.op {
            Op::Guard => self.assume(Val(i.a), true, false),
            Op::GuardNot => self.assume(Val(i.a), false, false),
            Op::CheckType => {
                let set = TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits());
                self.refine(Val(i.a), Fact::of(set))
            }
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
            | Op::Arith
            | Op::Unm
            | Op::Compare
            | Op::Truthy
            | Op::GetIndex
            | Op::SetIndex => {}
        }
        if i.op.has(Flags::TERMINATOR) {
            let (succs, n) = self.g.succs(&i);
            for &s in &succs[..n] {
                self.edge(s);
            }
            let d = &mut self.g.blocks[self.cur.idx()];
            d.end = self.g.insts.len() as u32;
            d.flags |= BlockFlags::SEALED;
            self.ends[self.cur.idx()] = self.table.seal();
            self.cur = Block::NONE;
        }
        v
    }

    fn push(&mut self, i: Inst, fact: Fact) -> Val {
        self.g.insts.push(i);
        self.g.facts.push(fact);
        Val(self.g.insts.len() as u32 - 1)
    }

    /// The fact a new instruction's result has by its definition.
    fn def_fact(&self, i: &Inst) -> Fact {
        let of = |w: u32| self.fact(Val(w));
        let by_rep = Fact::of_rep(i.rep);
        let aux_set = TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits());
        // Only a boxed value and a pointer have more to say than their
        // representation.
        let held = matches!(i.rep, Rep::Val | Rep::Ptr);
        let mut f = match i.op.info().fact {
            FactOf::Rep => by_rep,
            _ if !held => by_rep,
            FactOf::Small => Fact::of(TypeSet::SMALL),
            FactOf::Float => Fact::of(TypeSet::FLOAT),
            FactOf::IntSet => Fact::of(TypeSet::INT),
            FactOf::Tab => Fact::of(TypeSet::TAB),
            FactOf::AuxSet => Fact::of(aux_set),
            FactOf::Field if i.rep == Rep::Val => Fact::of(Field::info(i.aux).set),
            FactOf::Field => by_rep,
            FactOf::PassC => of(i.c),
            FactOf::MeetC => of(i.c).meet(Fact::of(aux_set)),
            FactOf::JoinBC => of(i.b).join(of(i.c)),
            FactOf::HeapA => Fact::of(of(i.a).set & TypeSet::HEAP),
            // SAFETY: a `K64` `Val` names no heap object.
            FactOf::Konst if i.rep == Rep::Val => Fact::of(TypeSet::of(unsafe {
                crate::env::value::Value::from_raw(i.k64())
            })),
            FactOf::Konst => by_rep,
            FactOf::Helper => match Helper::of(i.aux).info().fact {
                HelperFact::Tab if i.rep == Rep::Val => Fact::of(TypeSet::TAB),
                HelperFact::Int if i.rep == Rep::Val => Fact::of(TypeSet::INT),
                HelperFact::Tab | HelperFact::Int | HelperFact::Rep => by_rep,
            },
        };
        // A generic op's result is as unresolved as its operands.
        if i.op.has(Flags::GENERIC) {
            let mut pending = false;
            self.g.uses(i, |v| pending |= self.fact(v).pending());
            if pending {
                f.flags |= FactFlags::PENDING;
            }
        }
        f
    }

    /// The instruction against its op's row, always on. Unused words are
    /// zeroed, so that equal instructions are equal words.
    fn check(&self, i: &mut Inst) {
        let info = i.op.info();
        if self.cur.is_none() {
            bug!("{:?} outside a block", i.op);
        }
        for (k, w) in [&mut i.a, &mut i.b, &mut i.c].into_iter().enumerate() {
            if info.args[k] == Arg::No {
                *w = 0;
            }
        }
        if let Err(what) = self.g.form(i) {
            bug!("{:?} breaks its form: {what}", i.op);
        }
    }

    // --- numbering ----------------------------------------------------------

    fn hash(&self, i: &Inst) -> usize {
        let mix = |h: u64, x: u32| {
            (h ^ x as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(29)
        };
        if let Some((key, _)) = mem_key(i) {
            let h = key.iter().fold(0xff, |h, &x| mix(h, x));
            return (h ^ h >> 32) as usize;
        }
        let info = i.op.info();
        let exits = info.flags.contains(Flags::EXITS);
        let mut h = mix(i.op as u64 | (i.rep as u64) << 8, aux_key(i) as u32);
        for (k, kind) in info.args.iter().enumerate() {
            match kind {
                // Where it would leave to is not what it computes.
                Arg::Fs | Arg::OptFs if exits => {}
                Arg::Vals | Arg::Entries => {
                    for &x in self.g.run_of(i) {
                        h = mix(h, x);
                    }
                }
                _ => h = mix(h, i.args()[k]),
            }
        }
        (h ^ h >> 32) as usize
    }

    fn same(&self, i: &Inst, j: &Inst) -> bool {
        if let (Some((x, _)), y) = (mem_key(i), mem_key(j)) {
            return y.is_some_and(|(y, _)| x == y);
        }
        if (i.op, i.rep, aux_key(i)) != (j.op, j.rep, aux_key(j)) {
            return false;
        }
        let info = i.op.info();
        let exits = info.flags.contains(Flags::EXITS);
        (info.args.iter().enumerate()).all(|(k, kind)| match kind {
            Arg::Fs | Arg::OptFs if exits => true,
            Arg::Vals | Arg::Entries => self.g.run_of(i) == self.g.run_of(j),
            _ => i.args()[k] == j.args()[k],
        })
    }

    /// The versions a memory access's heap and key are at; zeros for
    /// anything else, and for memory nothing writes.
    fn versions(&self, i: &Inst) -> (u32, u32) {
        if mem_key(i).is_none() {
            let info = i.op.info();
            return if info.flags.contains(Flags::VERSIONED) {
                let class = info.reads.bits().trailing_zeros();
                (self.table.get(Key::heap(class)), 0)
            } else {
                (0, 0)
            };
        }
        let (heap, key) = heap_of(i);
        if heap.is_empty() {
            return (0, 0);
        }
        let class = heap.bits().trailing_zeros();
        (
            self.table.get(Key::heap(class)),
            key.map_or(0, |k| self.table.get(Key::heap(k))),
        )
    }

    /// Everything in `heaps` may have been written.
    pub(crate) fn kill(&mut self, heaps: Heaps) {
        let mut bits = (heaps & Heaps::ALL).bits();
        while bits != 0 {
            self.version += 1;
            self.table
                .set(Key::heap(bits.trailing_zeros()), self.version);
            bits &= bits - 1;
        }
    }

    /// The dominating value equal to `i`, or the slot `i` takes: for a
    /// memory access, what its place holds, a load's result or a store's
    /// value.
    pub(super) fn number(&self, i: &Inst) -> Result<Val, usize> {
        let mask = self.num.len() - 1;
        let mut s = self.hash(i) & mask;
        let now = self.versions(i);
        loop {
            let [v, b, heap, key] = self.num[s];
            if v == NONE {
                return Err(s);
            }
            let j = &self.g.insts[v as usize];
            if self.same(i, j) {
                let b = Block(b);
                let holds = (heap, key) == now && (b == self.cur || self.g.dominates(b, self.cur));
                return match (holds, mem_key(j)) {
                    (false, _) => Err(s),
                    (true, Some((_, Some(stored)))) => Ok(Val(stored)),
                    (true, _) => Ok(Val(v)),
                };
            }
            s = (s + 1) & mask;
        }
    }

    /// Whether frame slot `r` holds `v` here.
    pub(crate) fn slot_holds(&self, r: u16, v: Val) -> bool {
        let load = Inst {
            op: Op::SlotLoad,
            rep: Rep::Val,
            aux: r,
            a: 0,
            b: 0,
            c: 0,
        };
        self.number(&load) == Ok(v)
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.num);
        self.num.resize(old.len() * 2, [NONE; 4]);
        let mask = self.num.len() - 1;
        for e in old {
            if e[0] != NONE {
                let mut s = self.hash(&self.g.insts[e[0] as usize]) & mask;
                while self.num[s][0] != NONE {
                    s = (s + 1) & mask;
                }
                self.num[s] = e;
            }
        }
    }

    // --- what a branch proved -----------------------------------------------

    /// `v` in the arm of a branch on `cond` (`arm`: its true one), where it
    /// is known to be in `set` because of the branch: what relies on that
    /// uses the result.
    pub(crate) fn proved(&mut self, cond: Val, arm: bool, v: Val, set: TypeSet) -> Val {
        let aux = (set & TypeSet::ANY).bits() | if arm { 0 } else { SET_ARM };
        self.emit(Op::Proved, self.g.rep(v), aux, cond.0, 0, v.0)
    }

    /// `v` as a value whose definition says it is in `set`: itself, the
    /// result of the guard that checked it, or of a `Proved` under the
    /// branch that typed it. None where its fact holds some other way.
    pub(crate) fn typed(&mut self, v: Val, set: TypeSet) -> Option<Val> {
        if set.contains(self.g.facts[v.idx()].set) {
            return Some(v);
        }
        if self.lowered
            && let Some(cond) = self.type_test(v, set)
        {
            let guard = Inst {
                op: Op::Guard,
                rep: self.g.rep(v),
                aux: 0,
                a: cond.0,
                b: 0,
                c: v.0,
            };
            if let Ok(r) = self.number(&guard) {
                return Some(r);
            }
        }
        let proof = self.table.get(Key::proof(v.0));
        if proof == table::NONE {
            return None;
        }
        let cond = Val(proof & !table::PROOF_FALSE);
        let c = *self.g.inst(cond);
        let arm = proof & table::PROOF_FALSE == 0;
        let by = TypeSet::from_bits_retain(c.aux);
        let by = if arm { by } else { !by & TypeSet::ANY };
        // The branch tested this set, not what else narrowed the value since.
        (c.op == Op::IsType && set.contains(by)).then(|| self.proved(cond, arm, v, by))
    }

    // --- arms, joins and loops ----------------------------------------------

    /// Branch on `cond` into two arms, each cold or not. A condition known
    /// here leaves one of them unreached.
    pub(crate) fn fork(&mut self, cond: Val, cold_t: bool, cold_f: bool) -> (Arm, Arm) {
        let lp = self.loop_here();
        let arm = |e: &mut Self, cold| Arm {
            block: e.new_block(),
            lp,
            cold,
        };
        let (t, f) = (arm(self, cold_t), arm(self, cold_f));
        self.br(cond, t.block, f.block);
        (t, f)
    }

    /// Where `cond` holds, a new arm; else on to `join` with `values`.
    pub(crate) fn fork_or_leave<const N: usize>(
        &mut self,
        cond: Val,
        cold: bool,
        join: &mut Join<N>,
        values: [Val; N],
    ) -> Arm {
        let arm = Arm {
            block: self.new_block(),
            lp: self.loop_here(),
            cold,
        };
        // A condition known to hold sends nothing on.
        if self.known(cond) != Some(true) {
            self.left(join, values);
        }
        self.br(cond, arm.block, join.block);
        arm
    }

    /// Open `arm`; false where nothing reaches it.
    pub(crate) fn arm(&mut self, arm: Arm) -> bool {
        if self.bail.is_some() {
            return false;
        }
        let flags = if arm.cold {
            BlockFlags::COLD
        } else {
            BlockFlags::empty()
        };
        self.begin(arm.block, arm.lp, 0, flags)
    }

    /// A continuation of the open block's loop, opened by `join` once its
    /// arms have left for it.
    pub(crate) fn new_join<const N: usize>(&mut self) -> Join<N> {
        Join {
            block: self.new_block(),
            lp: self.loop_here(),
            arms: [[Val::NONE; N]; MAX_ARMS],
            n: 0,
            from_loop: false,
        }
    }

    fn left<const N: usize>(&mut self, join: &mut Join<N>, values: [Val; N]) {
        if self.bail.is_some() {
            return;
        }
        if join.n == MAX_ARMS {
            bug!("more arms than a join takes");
        }
        join.arms[join.n] = values;
        join.n += 1;
        join.from_loop |= self.loop_here() != join.lp;
    }

    /// End the open arm: on to `join` with `values`.
    pub(crate) fn leave<const N: usize>(&mut self, join: &mut Join<N>, values: [Val; N]) {
        self.left(join, values);
        self.jump(join.block);
    }

    /// Open `join` as the merge of the arms that left for it: a parameter
    /// where their values differ, and for every value where an arm left from
    /// a loop inside the join's (the loop-closed form keeps those). None
    /// where no arm left: nothing reaches what follows.
    pub(crate) fn join<const N: usize>(&mut self, join: Join<N>) -> Option<[Val; N]> {
        if self.bail.is_some() {
            return Some([Val(0); N]);
        }
        if join.n == 0 || !self.begin(join.block, join.lp, 0, BlockFlags::empty()) {
            return None;
        }
        let came = self.g.block(self.cur).nfwd as usize;
        if came != join.n {
            bug!("{} arms left for a join {came} reach", join.n);
        }
        let arms = &join.arms[..came];
        let mut out = [Val::NONE; N];
        for (k, o) in out.iter_mut().enumerate() {
            let first = arms[0][k];
            *o = if !join.from_loop && arms.iter().all(|a| a[k] == first) {
                first
            } else {
                let mut inputs = [Val::NONE; MAX_ARMS];
                for (i, a) in inputs.iter_mut().zip(arms) {
                    *i = a[k];
                }
                self.param(self.g.rep(first), None, &inputs[..came])
            };
        }
        Some(out)
    }

    /// Enter a loop of its own header, nested in the open block's, carrying
    /// `values` as the header's parameters. `writes` is what its body may
    /// write, unknown from the header on; `state` what an exit hoisted to its
    /// entry restores.
    pub(crate) fn loop_open<const N: usize>(
        &mut self,
        writes: Heaps,
        state: Val,
        values: [Val; N],
    ) -> (Loop, [Val; N]) {
        if self.bail.is_some() {
            return (Loop::NONE, values);
        }
        let header = self.new_block();
        let lp = self.new_loop(header, self.loop_here(), writes, state);
        self.jump(header);
        self.begin(header, lp, 1, BlockFlags::LOOP_HEADER);
        // A parameter says nothing its back edge's value might not.
        let params = values.map(|v| {
            let rep = self.g.rep(v);
            self.param(rep, Some(Fact::of_rep(rep)), &[v])
        });
        (lp, params)
    }

    /// The back edge of `lp` from the open block, with the header's
    /// parameters' next values.
    pub(crate) fn loop_back<const N: usize>(&mut self, lp: Loop, values: [Val; N]) {
        if self.bail.is_some() {
            return;
        }
        let header = self.g.loops[lp.idx()].header;
        let k = self.next_pred(header);
        let first = self.g.block(header).start;
        for (j, v) in values.into_iter().enumerate() {
            self.set_input(Val(first + j as u32), k, v);
        }
        self.jump(header);
    }

    fn loop_here(&self) -> Loop {
        self.g.block(self.cur).lp
    }

    // --- shorthands ---------------------------------------------------------

    pub(crate) fn op1(&mut self, op: Op, rep: Rep, a: Val) -> Val {
        self.emit(op, rep, 0, a.0, 0, 0)
    }

    pub(crate) fn op2(&mut self, op: Op, rep: Rep, a: Val, b: Val) -> Val {
        self.emit(op, rep, 0, a.0, b.0, 0)
    }

    pub(crate) fn cmp(&mut self, op: Op, cc: Cc, a: Val, b: Val) -> Val {
        self.emit(op, Rep::B1, cc as u16, a.0, b.0, 0)
    }

    pub(crate) fn k32(&mut self, rep: Rep, x: u32) -> Val {
        self.emit(Op::K32, rep, 0, x, 0, 0)
    }

    pub(crate) fn k64(&mut self, rep: Rep, x: u64) -> Val {
        self.emit(Op::K64, rep, 0, x as u32, (x >> 32) as u32, 0)
    }

    pub(crate) fn kf64(&mut self, x: f64) -> Val {
        let x = x.to_bits();
        self.emit(Op::KF64, Rep::F64, 0, x as u32, (x >> 32) as u32, 0)
    }

    /// An integer constant of `rep`'s width.
    pub(crate) fn kint(&mut self, rep: Rep, x: i64) -> Val {
        match rep {
            Rep::I32 | Rep::B1 => self.k32(rep, x as u32),
            _ => self.k64(rep, x as u64),
        }
    }

    /// A C call of `h` with `args`, and `imm` where it takes an immediate.
    /// A call of `h`: its context values and immediate, by its row, ahead
    /// of `args`.
    pub(crate) fn call(&mut self, h: Helper, imm: u32, args: &[Val]) -> Val {
        let info = h.info();
        let mut list = std::mem::take(&mut self.list);
        list.clear();
        for c in info.ctx {
            list.push(match c {
                Ctx::Rt => self.k64(Rep::I64, self.rt),
                Ctx::Thread => self.emit(Op::Thread, Rep::Ptr, 0, 0, 0, 0),
                Ctx::Base => self.emit(Op::Base, Rep::Ptr, 0, 0, 0, 0),
            });
        }
        if info.imm {
            list.push(self.k64(Rep::I64, imm as u64));
        }
        list.extend_from_slice(args);
        let v = self.call_with(h, &list);
        self.list = list;
        v
    }

    /// A call of `h` with its whole operand list.
    pub(crate) fn call_with(&mut self, h: Helper, list: &[Val]) -> Val {
        let at = self.g.extra.len() as u32;
        self.g.extra.extend(list.iter().map(|v| v.0));
        let n = list.len() as u32;
        self.emit(Op::CallHelper, h.info().ret, h as u16, at, n, 0)
    }

    /// Check that the `Val` is in `set`, leaving with `tag` to `fs`
    /// otherwise: its unboxed value where the set has one representation,
    /// else the value itself, now known to be in the set.
    pub(crate) fn check_type(&mut self, v: Val, set: TypeSet, tag: ExitTag, fs: Val) -> Val {
        let aux = set.bits() | (tag as u16) << TypeSet::BITS;
        let rep = set.unboxed();
        let r = self.emit(Op::CheckType, rep, aux, v.0, fs.0, 0);
        if rep == Rep::None { v } else { r }
    }

    pub(crate) fn jump(&mut self, to: Block) {
        self.emit(Op::Jump, Rep::None, 0, to.0, 0, 0);
    }

    pub(crate) fn br(&mut self, c: Val, t: Block, f: Block) {
        self.emit(Op::Br, Rep::None, 0, c.0, t.0, f.0);
    }

    /// The state an exit restores: `entries` are (register, value) by
    /// ascending register, a register word with `HOME` where its frame slot
    /// holds the value already; `parent` is the caller's state for an
    /// inlined frame.
    pub(crate) fn frame_state(
        &mut self,
        pc: u32,
        kind: ExitKind,
        parent: Val,
        entries: &[(u32, Val)],
    ) -> Val {
        let at = self.g.extra.len();
        self.g.extra.push(pc);
        let mut last = None;
        for &(r, v) in entries {
            let reg = r & !super::HOME;
            if last.is_some_and(|l| l >= reg) || self.g.rep(v) == Rep::None {
                bug!("frame state entry r{reg} out of order or without a value");
            }
            last = Some(reg);
            // An exit boxes by representation: the entry needs no box.
            let d = self.g.inst(v);
            let v = if matches!(d.op, Op::BoxI32 | Op::BoxF64) {
                Val(d.a)
            } else {
                v
            };
            self.g.extra.extend([r, v.0]);
        }
        let n = entries.len() as u32;
        let v = self.emit(
            Op::FrameState,
            Rep::None,
            kind as u16,
            at as u32,
            n,
            parent.0,
        );
        // An equal state stood already: the run is not needed.
        if self.bail.is_some() || self.g.inst(v).a != at as u32 {
            self.g.extra.truncate(at);
        }
        v
    }
}
