//! A copy: the input graph emitted again into another emitter, which
//! rewrites as it goes. Copy 1 gives each parameter the representation its
//! resolved type has, converts what flows into it on the edge, and emits
//! into a low-level graph, so every high-level op meets its lowering rule
//! or its rewrite (`Emitter::lower`).

use super::analyse::Analysis;
use super::arith;
use super::emit::Emitter;
use super::ops::{Arg, Dialect, ExitKind, ExitTag, Flags, Op};
use super::types::{Fact, Rep, TypeSet};
use super::{Block, BlockFlags, Graph, Inst, Val};
use crate::jit::bug;

const NONE: u32 = u32::MAX;

#[derive(Default)]
pub(crate) struct Copy {
    /// Output values and blocks by input ones.
    map: Vec<Val>,
    blocks: Vec<Block>,
    /// Forward edges into blocks not opened yet: the values passed to the
    /// target's parameters at `vals[at..]`, with the target's next edge,
    /// newest first from `pending[target]`.
    edges: Vec<(u32, u32)>,
    pending: Vec<u32>,
    vals: Vec<Val>,
    scratch: Vec<Val>,
    entries: Vec<(u32, Val)>,
    came: Vec<u32>,
}

/// The representation a parameter of this type has.
fn rep_of(a: &Analysis, p: Val, i: &Inst, set: TypeSet) -> Rep {
    match set.unboxed() {
        r @ (Rep::I32 | Rep::F64) if i.rep == Rep::Val => r,
        // Integers of either size, as the analysis decided.
        _ if a.wide.contains(&p) => Rep::I64,
        _ => i.rep,
    }
}

impl Copy {
    /// `v` as `rep`, a value of type `set`: boxed, or unboxed where its
    /// fact allows, else through a check that leaves to `state`.
    fn convert(e: &mut Emitter, v: Val, rep: Rep, set: TypeSet, state: Val) -> Val {
        match (e.g.rep(v), rep) {
            (have, want) if have == want => v,
            (Rep::I32, Rep::Val) => e.op1(Op::BoxI32, Rep::Val, v),
            (Rep::F64, Rep::Val) => e.op1(Op::BoxF64, Rep::Val, v),
            (Rep::Val, Rep::I32 | Rep::F64) if !state.is_none() => {
                e.check_type(v, set, ExitTag::Type, state)
            }
            (Rep::Val, Rep::I32 | Rep::F64) if set.contains(e.fact(v).set) => e.unboxed(v, set),
            // Integers of either size: each of its arms, and with a state
            // one that leaves for anything else. These make blocks.
            (Rep::Val, Rep::I64) => {
                let tag = ExitTag::Type as u16;
                e.lower(Op::UnboxI64, Rep::I64, tag, v.0, state.0, 0)
            }
            (Rep::I32, Rep::I64) => e.op1(Op::SExt32, Rep::I64, v),
            (Rep::I64, Rep::Val) => e.lower(Op::BoxI64, Rep::Val, 0, v.0, 0, 0),
            (have, want) => bug!(
                "no conversion of {v:?} ({:?}, {:?}) from {have:?} to {want:?} of {set:?}",
                e.g.op(v),
                e.fact(v).set
            ),
        }
    }

    /// The values an edge to input block `to` passes its parameters, each
    /// in its parameter's representation, appended to `scratch`.
    fn pass(&mut self, g: &Graph, a: &Analysis, e: &mut Emitter, to: Block, from_pred: usize) {
        let d = g.block(to);
        let entry = from_pred < d.nfwd as usize && d.flags.contains(BlockFlags::LOOP_HEADER);
        self.scratch.clear();
        for p in d.start..d.end {
            let i = &g.insts[p as usize];
            if i.op != Op::Param {
                break;
            }
            let set = a.ty[p as usize];
            let v = self.map[g.inputs(Val(p))[from_pred] as usize];
            // A loop's entry edge carries the checks its parameters need.
            let state = if entry && a.guarded.contains(&Val(p)) {
                self.map[g.loops[d.lp.idx()].entry_state.idx()]
            } else {
                Val::NONE
            };
            let v = Copy::convert(e, v, rep_of(a, Val(p), i, set), set, state);
            self.scratch.push(v);
        }
    }

    /// An edge from the open block, which is input block `from`, to input
    /// block `to`: the output block to branch to. `split` puts what the edge
    /// computes in a block of its own.
    fn edge(
        &mut self,
        g: &Graph,
        a: &Analysis,
        e: &mut Emitter,
        from: Block,
        to: Block,
        nth: usize,
    ) -> Block {
        let pred = (g.preds(to).iter().enumerate())
            .filter(|&(_, &p)| p == from)
            .nth(nth)
            .unwrap_or_else(|| bug!("{from:?} is no predecessor of {to:?}"))
            .0;
        self.pass(g, a, e, to, pred);
        let out = self.blocks[to.idx()];
        if e.g.block(out).start != NONE {
            // A late edge: the target is open and takes its inputs now.
            let k = e.next_pred(out);
            let first = e.g.block(out).start;
            for (j, &v) in self.scratch.iter().enumerate() {
                e.set_input(Val(first + j as u32), k, v);
            }
        } else {
            let at = self.vals.len() as u32;
            self.vals.extend_from_slice(&self.scratch);
            self.edges.push((at, self.pending[to.idx()]));
            self.pending[to.idx()] = self.edges.len() as u32 - 1;
        }
        out
    }

    /// Copy `g`, analysed by `a`, into `e`, which was reset.
    pub(crate) fn run(&mut self, g: &Graph, a: &Analysis, e: &mut Emitter) {
        self.map.clear();
        self.map.resize(g.insts.len(), Val::NONE);
        self.blocks.clear();
        self.blocks
            .extend((0..g.blocks.len()).map(|_| e.new_block()));
        self.pending.clear();
        self.pending.resize(g.blocks.len(), NONE);
        self.edges.clear();
        self.vals.clear();
        for l in &g.loops {
            e.new_loop(self.blocks[l.header.idx()], l.parent, l.writes, Val::NONE);
        }
        for &b in &g.order {
            if e.bail.is_some() {
                return;
            }
            let d = g.block(b);
            let header = d.flags.contains(BlockFlags::LOOP_HEADER);
            if header {
                let state = self.map[g.loops[d.lp.idx()].entry_state.idx()];
                e.g.loops[d.lp.idx()].entry_state = state;
            }
            if !e.begin(
                self.blocks[b.idx()],
                d.lp,
                d.npreds - d.nfwd,
                d.flags - BlockFlags::SEALED,
            ) {
                continue;
            }
            // The forward edges that came, oldest first.
            let mut came = std::mem::take(&mut self.came);
            came.clear();
            let mut at = self.pending[b.idx()];
            while at != NONE {
                came.push(self.edges[at as usize].0);
                at = self.edges[at as usize].1;
            }
            came.reverse();
            for v in d.start..d.end {
                let i = &g.insts[v as usize];
                self.map[v as usize] = match i.op {
                    Op::Param => {
                        let j = (v - d.start) as usize;
                        self.scratch.clear();
                        (self.scratch).extend(came.iter().map(|&at| self.vals[at as usize + j]));
                        let set = a.ty[v as usize];
                        e.param(
                            rep_of(a, Val(v), i, set),
                            Some(Fact::of(set)),
                            &self.scratch,
                        )
                    }
                    _ if !a.live(Val(v), i) => continue,
                    Op::FrameState => {
                        let (pc, entries) = g.entries(Val(v));
                        self.entries.clear();
                        (self.entries)
                            .extend(entries.iter().map(|x| (x[0], self.map[x[1] as usize])));
                        let parent = if i.c == NONE {
                            Val::NONE
                        } else {
                            self.map[i.c as usize]
                        };
                        let kind = if i.aux == 0 {
                            ExitKind::Before
                        } else {
                            ExitKind::After
                        };
                        e.frame_state(pc, kind, parent, &self.entries)
                    }
                    Op::Jump => {
                        let to = self.edge(g, a, e, b, Block(i.a), 0);
                        e.jump(to);
                        break;
                    }
                    Op::Br => {
                        let c = self.map[i.a as usize];
                        let (t, f) = (Block(i.b), Block(i.c));
                        // Decided first: an edge not taken passes nothing.
                        match e.known(c) {
                            Some(taken) => {
                                let to = self.edge(g, a, e, b, if taken { t } else { f }, 0);
                                e.jump(to);
                            }
                            None => self.branch(g, a, e, b, c, t, f),
                        }
                        break;
                    }
                    Op::CallHelper => {
                        self.scratch.clear();
                        for &x in g.run(i.a, i.b) {
                            let v = self.map[x as usize];
                            let v = Copy::convert(e, v, g.rep(Val(x)), TypeSet::ANY, Val::NONE);
                            self.scratch.push(v);
                        }
                        e.call_with(super::ops::Helper::of(i.aux), &self.scratch)
                    }
                    // An integer held as `I64` goes to its op as it is:
                    // boxed, it would be tested and unboxed again.
                    Op::Arith | Op::Unm => {
                        let mut w = i.args();
                        let n = if i.op == Op::Unm { 1 } else { 2 };
                        let fs = self.map[w[n] as usize];
                        w[n] = fs.0;
                        for (k, x) in w[..n].iter_mut().enumerate() {
                            let v = self.map[*x as usize];
                            let wide = arith::takes_i64(i.aux, k);
                            *x = match e.g.rep(v) {
                                Rep::I64 if wide => v.0,
                                // The site saw this operand small: an
                                // integer that is not leaves, as its type
                                // told the analysis.
                                Rep::I64 => {
                                    let tag = ExitTag::Type as u16;
                                    let x = e.emit(Op::LToI, Rep::I32, tag, v.0, fs.0, 0);
                                    e.op1(Op::BoxI32, Rep::Val, x).0
                                }
                                _ => Copy::convert(e, v, Rep::Val, TypeSet::ANY, Val::NONE).0,
                            };
                        }
                        e.lower(i.op, i.rep, i.aux, w[0], w[1], w[2])
                    }
                    // Three boxed operands in the extra pool, and a state.
                    Op::SetIndex | Op::ForLimit | Op::ForSkip | Op::ForMore => {
                        let run = [0, 1, 2].map(|k| {
                            let x = self.map[g.run(i.a, i.b)[k] as usize];
                            Copy::convert(e, x, Rep::Val, TypeSet::ANY, Val::NONE).0
                        });
                        let at = e.g.extra.len() as u32;
                        e.g.extra.extend(run);
                        e.lower(i.op, i.rep, i.aux, at, 3, self.map[i.c as usize].0)
                    }
                    _ if i.op.has(Flags::PLAIN) => self.inst(g, e, i),
                    _ => bug!("{:?} has no copy", i.op),
                };
                if e.cur.is_none() {
                    break;
                }
            }
            self.came = came;
        }
        e.finish();
    }

    /// A two-way branch. An edge that computes something (a conversion into
    /// a parameter) gets a block of its own, so the other does not pay.
    #[allow(clippy::too_many_arguments)]
    fn branch(
        &mut self,
        g: &Graph,
        a: &Analysis,
        e: &mut Emitter,
        b: Block,
        c: Val,
        t: Block,
        f: Block,
    ) {
        let same = t == f;
        let needs = |s: &Self, e: &Emitter, to: Block, nth: usize| {
            let pred = (g.preds(to).iter().enumerate())
                .filter(|&(_, &p)| p == b)
                .nth(nth)
                .unwrap()
                .0;
            let d = g.block(to);
            (d.start..d.end)
                .take_while(|&p| g.insts[p as usize].op == Op::Param)
                .any(|p| {
                    let v = s.map[g.inputs(Val(p))[pred] as usize];
                    e.g.rep(v) != rep_of(a, Val(p), &g.insts[p as usize], a.ty[p as usize])
                })
        };
        let split = [needs(self, e, t, 0), needs(self, e, f, same as usize)];
        let lp = e.g.block(e.cur).lp;
        let mut outs = [Block::NONE; 2];
        for (k, &to) in [t, f].iter().enumerate() {
            outs[k] = if split[k] {
                e.new_block()
            } else {
                self.edge(g, a, e, b, to, k * same as usize)
            };
        }
        e.br(c, outs[0], outs[1]);
        for (k, &to) in [t, f].iter().enumerate() {
            if split[k] {
                e.begin(outs[k], lp, 0, BlockFlags::empty());
                let out = self.edge(g, a, e, b, to, k * same as usize);
                e.jump(out);
            }
        }
    }

    /// Any other instruction: its operands mapped, each in the
    /// representation it had: a value a parameter now holds unboxed is boxed
    /// for a use that took it boxed, which the rules undo where they can.
    fn inst(&mut self, g: &Graph, e: &mut Emitter, i: &Inst) -> Val {
        let mut w = i.args();
        for (k, kind) in i.op.info().args.iter().enumerate() {
            match kind {
                Arg::No | Arg::Imm | Arg::Len => {}
                Arg::Opt | Arg::OptFs if w[k] == NONE => {}
                Arg::Blk | Arg::Vals | Arg::Entries => bug!("{:?} is not copied plainly", i.op),
                _ => {
                    let was = g.rep(Val(w[k]));
                    let v = self.map[w[k] as usize];
                    w[k] = Copy::convert(e, v, was, TypeSet::ANY, Val::NONE).0;
                }
            }
        }
        // A high-level op meets its rule, or its rewrite.
        match i.op.info().dialect {
            Dialect::H => e.lower(i.op, i.rep, i.aux, w[0], w[1], w[2]),
            Dialect::Both | Dialect::L => e.emit(i.op, i.rep, i.aux, w[0], w[1], w[2]),
        }
    }
}
