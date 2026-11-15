//! The emitter: the one way an instruction enters a graph. The builder and
//! every copy go through `emit`, which checks the operands against the op
//! table, applies the rules (`rules`), returns a dominating identical
//! instruction where there is one, and otherwise appends and records what
//! the instruction makes known.

use super::ops::{Arg, Aux, Cc, Dialect, ExitKind, ExitTag, Flags, Heaps, Op};
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
        _ => i.aux,
    }
}

/// `reset` before the first graph, and between graphs.
#[derive(Default)]
pub(crate) struct Emitter {
    pub(crate) g: Graph,
    pub(crate) table: Table,
    /// Each sealed block's end state.
    ends: Vec<Snap>,
    /// Value numbers, open-addressed: a value and its block, empty where the
    /// value is `NONE`. An entry whose block does not dominate is stale and
    /// gets overwritten.
    num: Vec<[u32; 2]>,
    num_len: usize,
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
    scratch: Vec<Snap>,
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
        self.num.resize(256, [NONE; 2]);
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

    pub(crate) fn new_loop(&mut self, header: Block, parent: Loop, writes: Heaps) -> Loop {
        self.g.loops.push(LoopData {
            header,
            parent,
            writes,
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
            if !key.is_fact() {
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
        // A branch's condition is known on its edges. The fact is of an SSA
        // value, so one forward predecessor is enough: back edges come from
        // where it already held.
        if nfwd == 1 {
            let t = *self.g.term(self.g.preds[at]);
            if t.op == Op::Br && t.b != t.c {
                self.assume(Val(t.a), t.b == b.0);
            }
        }
        true
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

    fn refine(&mut self, v: Val, with: Fact) {
        let was = self.fact(v);
        let f = was.meet(with);
        if f != was {
            self.table.set(Key::fact(v.0), f.bits());
        }
    }

    /// The `B1` `c` is `truth` from here on, and with it what it tested.
    fn assume(&mut self, c: Val, truth: bool) {
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
        if let Some(v) = self.rule(&mut i) {
            return v;
        }
        self.append(i)
    }

    /// Number and append, past the rules.
    pub(super) fn append(&mut self, i: Inst) -> Val {
        let numbered = i.op.numbered();
        let mut slot = 0;
        if numbered {
            match self.number(&i) {
                Ok(v) => return v,
                Err(s) => slot = s,
            }
        }
        let fact = self.def_fact(&i);
        let v = self.push(i, fact);
        if numbered {
            if self.num[slot][0] == NONE {
                self.num_len += 1;
            }
            self.num[slot] = [v.0, self.cur.0];
            if self.num_len * 2 > self.num.len() {
                self.grow();
            }
        }
        match i.op {
            Op::Guard => self.assume(Val(i.a), true),
            Op::GuardNot => self.assume(Val(i.a), false),
            Op::CheckType => {
                let set = TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits());
                self.refine(Val(i.a), Fact::of(set))
            }
            _ => {}
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
        let mut f = match (i.op, i.rep) {
            // SAFETY: a `K64` `Val` names no heap object.
            (Op::K64, Rep::Val) => Fact::of(TypeSet::of(unsafe {
                crate::env::value::Value::from_raw(i.k64())
            })),
            (Op::KObj | Op::BoxPtr, _) => Fact::of(TypeSet::from_bits_retain(i.aux)),
            (Op::BoxI32, _) => Fact::of(TypeSet::SMALL),
            (Op::BoxF64, _) => Fact::of(TypeSet::FLOAT),
            (Op::BoxI64, _) => Fact::of(TypeSet::INT),
            (Op::Select, Rep::Val | Rep::Ptr) => of(i.b).join(of(i.c)),
            (Op::Guard | Op::GuardNot, Rep::Val | Rep::Ptr) => of(i.c),
            (Op::UnboxPtr | Op::UnboxPtrPin, _) => Fact::of(of(i.a).set & TypeSet::HEAP),
            (Op::CheckType, Rep::Ptr) => {
                Fact::of(TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits()))
            }
            _ => Fact::of_rep(i.rep),
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
        if self.lowered && info.dialect == Dialect::H {
            bug!("high-level {:?} in a low-level graph", i.op);
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
        let info = i.op.info();
        let exits = info.flags.contains(Flags::EXITS);
        let mix = |h: u64, x: u32| {
            (h ^ x as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(29)
        };
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

    /// The dominating instruction equal to `i`, or the slot `i` takes.
    pub(super) fn number(&self, i: &Inst) -> Result<Val, usize> {
        let mask = self.num.len() - 1;
        let mut s = self.hash(i) & mask;
        loop {
            let [v, b] = self.num[s];
            if v == NONE {
                return Err(s);
            }
            if self.same(i, &self.g.insts[v as usize]) {
                let b = Block(b);
                return if b == self.cur || self.g.dominates(b, self.cur) {
                    Ok(Val(v))
                } else {
                    Err(s)
                };
            }
            s = (s + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.num);
        self.num.resize(old.len() * 2, [NONE; 2]);
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
