//! Bytecode to the high-level graph. Registers are boxed SSA values in the
//! emitter's table: a join where predecessors disagree makes a parameter, a
//! loop header one for each live register its body assigns, pending until
//! the loop's types are resolved (`ir::phis`). A site whose operands' facts
//! are final specialises as it is emitted; one that hangs on a loop
//! parameter stays generic, carrying what its feedback speculates.

mod calls;
pub(crate) mod cfg;
mod tables;

use cfg::{Cfg, RegSet};

use crate::dmm::Gc;
use crate::env::function::Prototype;
use crate::env::value::Value;
use crate::instruction::{ArithKind, Family, Instruction, Op as Bc};
use crate::jit::compile::Bail;
use crate::jit::feedback as fb;
use crate::jit::ir::emit::Emitter;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, HELPER_FAIL_VAL, Heaps, Helper, Op};
use crate::jit::ir::table::{self, Key};
use crate::jit::ir::types::{Fact, FactFlags, Rep, TypeSet};
use crate::jit::ir::{Block, BlockFlags, HOME, Loop, Pool, Val};

/// The number kind a generic op takes an operand as where its fact does not
/// say: what the site's form or feedback saw. In the op's `aux`, three bits
/// an operand.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Spec {
    I32,
    F64,
}

impl Spec {
    pub(crate) fn of(bits: u16) -> Spec {
        if bits & 7 == 0 { Spec::I32 } else { Spec::F64 }
    }

    pub(crate) fn set(self) -> TypeSet {
        match self {
            Spec::I32 => TypeSet::SMALL,
            Spec::F64 => TypeSet::FLOAT,
        }
    }

    /// The kind a fact decides, where it does.
    pub(crate) fn by_fact(f: Fact) -> Option<Spec> {
        if f.pending() || f.set.is_empty() {
            None
        } else if TypeSet::SMALL.contains(f.set) {
            Some(Spec::I32)
        } else if TypeSet::FLOAT.contains(f.set) {
            Some(Spec::F64)
        } else {
            None
        }
    }
}

/// Whether an arithmetic op of `kind` (an `ArithKind`) on operands of these
/// kinds makes an integer: the bitwise ones always (their operands are
/// checked to be integers), a division or a power never.
pub(crate) fn int_result(kind: u16, a: Spec, b: Spec) -> bool {
    use ArithKind::*;
    let is = |k: ArithKind| kind == k as u16;
    if is(BAnd) || is(BOr) || is(BXor) {
        true
    } else {
        a == Spec::I32 && b == Spec::I32 && !is(Div) && !is(Pow)
    }
}

/// A generic op's `aux`: its low four bits (an `ArithKind`, a `Cc`), then
/// each operand's speculated kind.
pub(crate) const SPEC_SHIFT: [u32; 2] = [4, 7];

/// A loop of the region's control flow, which is the function's from the
/// region's entry: a loop region's entry heads everything that comes back
/// to it.
struct RLoop {
    header: u32,
    parent: Option<usize>,
    body: Vec<bool>,
    /// Edges back to the header.
    latches: u32,
    /// The registers the body writes.
    assigned: RegSet,
    ir: Loop,
    /// The header's parameters, by register.
    params: Vec<(u8, Val)>,
}

/// What a region assumes, and of whom (`jit::deps`).
#[derive(Clone, Copy)]
pub(crate) enum DepOn<'gc> {
    /// A class keeps the metamethod bits as they are.
    Class(crate::env::shape::MtCache<'gc>),
    /// A watched holder keeps its shape, or a slot its value
    /// (`deps::SHAPE_GROUP`, `deps::slot_group`).
    Shape(crate::env::Shape<'gc>),
}

/// What `build` hands on besides the graph: the shapes the code compares
/// with, what it assumes, and the instructions it has no code for.
pub(crate) struct Built<'gc> {
    pub(crate) shapes: Vec<crate::env::Shape<'gc>>,
    pub(crate) deps: Vec<(DepOn<'gc>, u64)>,
    pub(crate) missing: Vec<Bc>,
}

struct Builder<'a, 'gc> {
    e: &'a mut Emitter,
    pool: &'a mut Pool,
    cfg: &'a Cfg,
    proto: Gc<'gc, Prototype<'gc>>,
    loops: Vec<RLoop>,
    /// A bytecode block's innermost loop.
    loop_of: Vec<Option<usize>>,
    /// The loop a bytecode block heads.
    heads: Vec<Option<usize>>,
    /// The graph block forward edges into a bytecode block enter.
    blocks: Vec<Block>,
    /// The bytecode block and instruction being emitted.
    bb: u32,
    pc: u32,
    /// The frame state before the instruction at `pc`, once an exit needs it.
    state: Val,
    /// The shapes the code compares with, which the region keeps alive.
    shapes: Vec<crate::env::Shape<'gc>>,
    deps: Vec<(DepOn<'gc>, u64)>,
    /// The address of the arena's metrics, where the collector's counters
    /// are.
    metrics: u64,
    /// The instructions compiled as a deopt for want of code.
    missing: Vec<Bc>,
}

/// Build the region of `proto` entered at `entry_pc` into `e`, which was
/// reset. A region the builder cannot make sets the emitter's bail.
pub(crate) fn build<'gc>(
    e: &mut Emitter,
    pool: &mut Pool,
    cfg: &Cfg,
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
    metrics: u64,
) -> Built<'gc> {
    let nb = cfg.blocks.len();
    let root = cfg.block_of[entry_pc as usize];
    let (rpo, order) = cfg::rpo_of(&cfg.blocks, root);
    let mut b = Builder {
        e,
        pool,
        cfg,
        proto,
        loops: Vec::new(),
        loop_of: vec![None; nb],
        heads: vec![None; nb],
        blocks: Vec::new(),
        bb: root,
        pc: entry_pc,
        state: Val::NONE,
        shapes: Vec::new(),
        deps: Vec::new(),
        metrics,
        missing: Vec::new(),
    };
    if !b.find_loops(root, &rpo, &order) {
        b.e.bail(Bail::Unsupported("irreducible control flow"));
    }
    b.blocks = (0..nb).map(|_| b.e.new_block()).collect();
    for &bb in &rpo {
        if b.e.bail.is_some() {
            break;
        }
        b.block(bb, bb == root, entry_pc);
    }
    b.e.g.invalidable = !b.deps.is_empty();
    Built {
        shapes: b.shapes,
        deps: b.deps,
        missing: b.missing,
    }
}

impl<'gc> Builder<'_, 'gc> {
    /// The loops of the control flow from `root`; false where it is
    /// irreducible.
    fn find_loops(&mut self, root: u32, rpo: &[u32], order: &[usize]) -> bool {
        let blocks = &self.cfg.blocks;
        let nb = blocks.len();
        let mut idom = vec![u32::MAX; nb];
        idom[root as usize] = root;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo[1..] {
                let mut new = u32::MAX;
                for &p in &blocks[b as usize].preds {
                    if idom[p as usize] == u32::MAX {
                        continue;
                    }
                    let (mut x, mut y) = (if new == u32::MAX { p } else { new }, p);
                    while x != y {
                        while order[x as usize] > order[y as usize] {
                            x = idom[x as usize];
                        }
                        while order[y as usize] > order[x as usize] {
                            y = idom[y as usize];
                        }
                    }
                    new = x;
                }
                if idom[b as usize] != new {
                    idom[b as usize] = new;
                    changed = true;
                }
            }
        }
        for &b in rpo {
            for &h in &blocks[b as usize].succs {
                if order[h as usize] > order[b as usize] {
                    continue;
                }
                let mut d = b;
                while d != h && d != root {
                    d = idom[d as usize];
                }
                if d != h {
                    return false;
                }
                let li = *self.heads[h as usize].get_or_insert_with(|| {
                    let mut body = vec![false; nb];
                    body[h as usize] = true;
                    self.loops.push(RLoop {
                        header: h,
                        parent: None,
                        body,
                        latches: 0,
                        assigned: RegSet::EMPTY,
                        ir: Loop::NONE,
                        params: Vec::new(),
                    });
                    self.loops.len() - 1
                });
                self.loops[li].latches += 1;
                let mut stack = vec![b];
                while let Some(x) = stack.pop() {
                    if std::mem::replace(&mut self.loops[li].body[x as usize], true) {
                        continue;
                    }
                    let preds = &blocks[x as usize].preds;
                    stack.extend(preds.iter().filter(|&&p| order[p as usize] != usize::MAX));
                }
            }
        }
        // A block's innermost loop is the smallest holding it; a loop's
        // parent the smallest other holding its header.
        let size = |l: &RLoop| l.body.iter().filter(|&&x| x).count();
        let sizes: Vec<usize> = self.loops.iter().map(size).collect();
        let smallest = |b: usize, but: Option<usize>| {
            (0..sizes.len())
                .filter(|&l| Some(l) != but && self.loops[l].body[b])
                .filter(|&l| but.is_none_or(|i| sizes[l] > sizes[i]))
                .min_by_key(|&l| sizes[l])
        };
        let inner: Vec<Option<usize>> = (0..nb).map(|b| smallest(b, None)).collect();
        self.loop_of = inner;
        let parents: Vec<Option<usize>> = (0..sizes.len())
            .map(|l| smallest(self.loops[l].header as usize, Some(l)))
            .collect();
        for (l, parent) in parents.into_iter().enumerate() {
            self.loops[l].parent = parent;
        }
        for l in 0..sizes.len() {
            let mut assigned = RegSet::EMPTY;
            for (b, _) in (self.loops[l].body.iter().enumerate()).filter(|(_, x)| **x) {
                for pc in blocks[b].start..blocks[b].end {
                    let i = self.cfg.code[pc as usize];
                    let ud = cfg::use_def(&self.proto, i, self.cfg.nregs, &self.cfg.captured);
                    assigned.union(&ud.defs);
                    // Written on their jump's edge only.
                    let a = i.a() as usize;
                    match i.op() {
                        Bc::FORLOOP | Bc::FORLOOP_I | Bc::FORLOOP_F => {
                            assigned.insert(a + 2);
                            assigned.insert(a + 3);
                        }
                        Bc::JTSET | Bc::JFSET => assigned.insert(a),
                        // A callee may write the captured registers, which
                        // are read again after it.
                        Bc::CALL
                        | Bc::CALL_R0
                        | Bc::CALL_R1
                        | Bc::CALLS
                        | Bc::CALLS_R0
                        | Bc::CALLS_R1 => assigned.union(&self.cfg.captured),
                        _ => {}
                    }
                }
            }
            self.loops[l].assigned = assigned;
        }
        true
    }

    // --- registers and states ---------------------------------------------

    fn reg(&self, r: u8) -> Val {
        Val(self.e.table.get(Key::reg(r as u32)))
    }

    /// A captured register's slot holds its value at every moment: the
    /// closures holding its cell read it there, and write it there while
    /// they run, which is during a call (`calls.rs` reads it again after).
    fn set(&mut self, r: u8, v: Val) {
        self.e.table.set(Key::reg(r as u32), v.0);
        if self.cfg.captured.contains(r as usize) {
            self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
        }
    }

    /// Leave for the interpreter past its threshold, after the instruction
    /// at `pc`, which allocated: it collects.
    fn gc_check(&mut self, pc: u32) {
        let metrics = self.e.k64(Rep::Ptr, self.metrics);
        let used = self.e.load(Field::GcUsed, metrics, 0);
        let limit = self.e.load(Field::GcLimit, metrics, 0);
        let below = self.e.cmp(Op::Cmp, Cc::Lo, used, limit);
        let after = self.state_at(pc + 1);
        let tag = ExitTag::Gc as u16;
        self.e
            .emit(Op::Guard, Rep::None, tag, below.0, after.0, u32::MAX);
    }

    /// The frame's flags in `mask`.
    fn frame_flags(&mut self, mask: u64) -> Val {
        let base = self.e.emit(Op::Base, Rep::Ptr, 0, 0, 0, 0);
        let flags = self.e.load(Field::FrameFlags, base, 0);
        let mask = self.e.k64(Rep::I64, mask);
        self.e.op2(Op::And, Rep::I64, flags, mask)
    }

    fn live(&self, pc: u32) -> RegSet {
        self.cfg.live_in[pc as usize]
    }

    /// The frame state before the instruction being emitted: the live
    /// registers, those whose slot still holds its value marked.
    fn state(&mut self) -> Val {
        if self.state.is_none() {
            self.state = self.state_at(self.pc);
        }
        self.state
    }

    fn state_at(&mut self, pc: u32) -> Val {
        let entries: Vec<(u32, Val)> = (self.live(pc).iter())
            .filter(|&r| !self.reg(r as u8).is_none())
            .map(|r| {
                let v = self.reg(r as u8);
                let home = self.e.slot_holds(r as u16, v);
                (r as u32 | if home { HOME } else { 0 }, v)
            })
            .collect();
        self.e
            .frame_state(pc, ExitKind::Before, Val::NONE, &entries)
    }

    /// Assume `class` keeps `bits` as they are now.
    fn depend_class(
        &mut self,
        class: crate::env::shape::MtCache<'gc>,
        bits: crate::env::MetamethodBits,
    ) {
        self.deps.push((DepOn::Class(class), bits.bits() as u64));
    }

    /// A lazy exit to before the instruction at `pc`, a successor of the
    /// one being emitted.
    fn lazy_exit(&mut self, pc: u32) {
        let s = self.state_at(pc);
        self.e.emit(Op::LazyExit, Rep::None, 0, s.0, 0, 0);
    }

    /// Leave for the interpreter's handler of the instruction being
    /// emitted, with what is live stored: for one that ends the frame, or
    /// replaces it.
    fn dispatch(&mut self) {
        for r in self.live(self.pc).iter() {
            let v = self.reg(r as u8);
            if !v.is_none() && !self.e.slot_holds(r as u16, v) {
                self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
            }
        }
        let word = self.cfg.code[self.pc as usize].raw();
        let (lo, hi) = (word as u32, (word >> 32) as u32);
        self.e.emit(Op::Dispatch, Rep::None, 0, lo, hi, self.pc);
    }

    fn deopt(&mut self, tag: ExitTag) {
        if tag == ExitTag::Unsupported {
            self.missing.push(self.cfg.code[self.pc as usize].op());
        }
        let s = self.state();
        self.e.emit(Op::Deopt, Rep::None, tag as u16, s.0, 0, 0);
    }

    fn konst(&mut self, v: Value<'_>) -> Val {
        let set = TypeSet::of(v);
        if TypeSet::HEAP.contains(set) {
            let k = self.pool.intern(v.to_raw());
            self.e.emit(Op::KObj, Rep::Val, set.bits(), k, 0, 0)
        } else {
            self.e.k64(Rep::Val, v.to_raw())
        }
    }

    /// A helper's result, or an exit where it failed: the interpreter runs
    /// the instruction, which the helper left untouched.
    fn helped(&mut self, h: Helper, imm: u32, args: &[Val]) -> Val {
        let state = self.state();
        self.e.helped(h, imm, args, state)
    }

    // --- blocks -----------------------------------------------------------

    /// The graph block an edge from the block being emitted to bytecode
    /// block `to` enters. A back edge passes the header's parameters their
    /// values first.
    fn target(&mut self, to: u32) -> Block {
        if let Some(l) = self.heads[to as usize]
            && self.loops[l].body[self.bb as usize]
        {
            let head = self.e.g.loops[self.loops[l].ir.idx()].header;
            let k = self.e.next_pred(head);
            for n in 0..self.loops[l].params.len() {
                let (r, p) = self.loops[l].params[n];
                let v = self.reg(r);
                self.e.set_input(p, k, v);
            }
            return head;
        }
        self.blocks[to as usize]
    }

    fn jump(&mut self, to: u32) {
        let t = self.target(to);
        self.e.jump(t);
    }

    /// Branch on `c` to bytecode block `yes`, else `no`.
    fn br(&mut self, c: Val, yes: u32, no: u32) {
        // Decided first: an edge that is not taken passes nothing.
        match self.e.known(c) {
            Some(taken) => self.jump(if taken { yes } else { no }),
            None if yes == no => self.jump(yes),
            None => {
                let (t, f) = (self.target(yes), self.target(no));
                self.e.br(c, t, f);
            }
        }
    }

    fn ir_loop(&self, bb: u32) -> Loop {
        self.loop_of[bb as usize].map_or(Loop::NONE, |l| self.loops[l].ir)
    }

    fn block(&mut self, bb: u32, root: bool, entry_pc: u32) {
        let blk = &self.cfg.blocks[bb as usize];
        let (start, end) = (if root { entry_pc } else { blk.start }, blk.end);
        let head = self.heads[bb as usize];
        // What forward edges enter is outside the loop the block heads.
        let outer = match head {
            Some(l) => (self.loops[l].parent).map_or(Loop::NONE, |p| self.loops[p].ir),
            None => self.ir_loop(bb),
        };
        if !self
            .e
            .begin(self.blocks[bb as usize], outer, 0, BlockFlags::empty())
        {
            return;
        }
        (self.bb, self.pc, self.state) = (bb, start, Val::NONE);
        if root {
            for r in self.live(start).iter() {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.set(r as u8, v);
            }
        } else {
            self.join(start, outer);
        }
        if let Some(l) = head {
            // The loop's entry state, for what moves in front of it.
            let state = self.state();
            let header = self.e.new_block();
            let lp = self.e.new_loop(header, outer, Heaps::ALL, state);
            self.loops[l].ir = lp;
            self.e.jump(header);
            let latches = self.loops[l].latches;
            self.e.begin(header, lp, latches, BlockFlags::LOOP_HEADER);
            let pending = Fact {
                set: TypeSet::ANY,
                flags: FactFlags::PENDING,
            };
            let live = self.live(start);
            let assigned = self.loops[l].assigned;
            for r in live.iter().filter(|&r| assigned.contains(r)) {
                let v = self.reg(r as u8);
                let p = self.e.param(Rep::Val, Some(pending), &[v]);
                self.e.table.set(Key::reg(r as u32), p.0);
                self.loops[l].params.push((r as u8, p));
            }
            self.state = Val::NONE;
        }
        if self.cfg.never_ran[bb as usize] {
            return self.deopt(ExitTag::NeverRan);
        }
        for pc in start..end {
            (self.pc, self.state) = (pc, Val::NONE);
            self.inst(pc, self.cfg.code[pc as usize]);
            if self.e.cur.is_none() || self.e.bail.is_some() {
                return;
            }
        }
        match self.cfg.blocks[bb as usize].succs[..] {
            [next] => self.jump(next),
            _ => self.e.bail(Bail::Unsupported("code runs off its end")),
        }
    }

    /// A join's registers: one whose predecessors disagree becomes a
    /// parameter, and so does one a loop that ends here assigned
    /// (loop-closed form).
    fn join(&mut self, pc: u32, lp: Loop) {
        let b = self.e.cur;
        let npreds = self.e.g.block(b).nfwd as usize;
        let mut closed = RegSet::EMPTY;
        for k in 0..npreds {
            let mut from = self.e.g.block(self.e.g.preds(b)[k]).lp;
            while !from.is_none() && !self.e.g.loop_within(lp, from) {
                let l = self.loops.iter().find(|l| l.ir == from).unwrap();
                closed.union(&l.assigned);
                from = self.e.g.loops[from.idx()].parent;
            }
        }
        let live = self.live(pc);
        let mut vals = Vec::with_capacity(npreds);
        // A register the predecessors changed alike holds that value.
        for i in 0..self.e.table.merges() {
            let (key, at) = self.e.table.merge(i);
            if key.is_reg() && at.iter().all(|&v| v == at[0]) {
                let v = at[0];
                self.e.table.set(key, v);
            }
        }
        for r in live.iter() {
            let key = Key::reg(r as u32);
            vals.clear();
            vals.extend((0..npreds).map(|k| Val(self.e.table.pred_value(key, k))));
            // Liveness takes a conditional write (`JTSET`) for a read: a
            // register some path never gave a value has none to read.
            if vals.iter().any(|v| v.is_none()) {
                self.e.table.set(key, Val::NONE.0);
                continue;
            }
            if closed.contains(r) || vals.iter().any(|&v| v != vals[0]) {
                let p = self.e.param(Rep::Val, None, &vals);
                self.e.table.set(key, p.0);
            }
        }
        // A register that is not live has no value here.
        for i in 0..self.e.table.merges() {
            let (key, _) = self.e.table.merge(i);
            if key.is_reg() && !live.contains(key.index() as usize) {
                self.e.table.set(key, table::NONE);
            }
        }
    }

    // --- instructions -------------------------------------------------------

    /// The kind the site at `pc` takes each operand as, by its quickened
    /// form or else its feedback; none where the slice has no code for what
    /// it saw.
    fn spec(&self, pc: u32, i: Instruction) -> Option<[Spec; 2]> {
        let bits = self.proto.feedback.bits(pc as usize);
        let kinds = bits & fb::KINDS;
        let (op, generic) = (i.op(), i.generic_op());
        let info = generic.info();
        if bits & (fb::MM | fb::OVERFLOW) != 0 || kinds & !(fb::SMALL | fb::FLOAT) != 0 {
            return None;
        }
        use Spec::*;
        if op != generic && bits & fb::KIND_EXIT == 0 {
            let form = info.forms.iter().position(|f| *f == Some(op))?;
            let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
            return match (info.family, form) {
                (Family::CmpReg, _) => Some([I32, I32]),
                (Family::CmpImm, _) => Some([F64, F64]),
                (_, 0) => Some([I32, I32]),
                (_, 1) => Some([F64, F64]),
                (_, 2) if imm => Some([I32, I32]),
                (_, 2) => Some([I32, F64]),
                (_, 3) => Some([F64, I32]),
                _ => None,
            };
        }
        match (kinds & fb::SMALL != 0, kinds & fb::FLOAT != 0) {
            (true, false) => Some([I32, I32]),
            (false, true) => Some([F64, F64]),
            _ => None,
        }
    }

    /// A generic op on `x` and `y`, or a deopt where neither their facts
    /// nor the site's speculation give both a kind.
    fn generic(&mut self, op: Op, rep: Rep, low: u16, x: Val, y: Val) -> Option<Val> {
        let spec = self.spec(self.pc, self.cfg.code[self.pc as usize]);
        let mut aux = low;
        let own = [x, y].map(|v| Spec::by_fact(self.e.fact(v)));
        // An equality nothing says is of numbers is of anything.
        if op == Op::Compare && low == Cc::Eq as u16 && spec.is_none() && own == [None; 2] {
            return Some(self.equal(x, y));
        }
        for k in 0..2 {
            // A site that recorded nothing takes an operand as the other is.
            let Some(s) = own[k].or(spec.map(|s| s[k])).or(own[1 - k]) else {
                self.deopt(ExitTag::Unsupported);
                return None;
            };
            aux |= (s as u16) << SPEC_SHIFT[k];
        }
        let s = self.state();
        Some(self.e.emit(op, rep, aux, x.0, y.0, s.0))
    }

    fn inst(&mut self, pc: u32, i: Instruction) {
        use Bc::*;
        let (a, b) = (i.a(), i.b());
        let cfg = self.cfg;
        let succs = &cfg.blocks[self.bb as usize].succs;
        // Of a branch: where it falls through and where it jumps.
        let (fall, jump) = (succs.first().copied(), succs.get(1).copied());
        let op = i.op();
        let info = i.generic_op().info();
        match op {
            NOP | FUNC | LOOP => {}
            MOVE => {
                let v = self.reg(b);
                self.set(a, v);
            }
            LOAD => {
                let v = self.konst(self.proto.constants[i.d() as usize]);
                self.set(a, v);
            }
            CONCAT => {
                let (x, y) = (self.reg(b), self.reg(i.c()));
                let v = self.helped(Helper::Concat, 0, &[x, y]);
                self.set(a, v);
            }
            NEWTABLE => self.newtable(pc, i),
            SETLIST => self.setlist(i),
            GETTABLE => self.gettable(pc, i),
            SETTABLE => self.settable(pc, i),
            GETUPVAL | GETUPVAL_REF => self.getupval(i),
            SETUPVAL => self.setupval(i),
            UNM => {
                let x = self.reg(b);
                // The fast path of a small integer records nothing.
                let seen = self.proto.feedback.bits(pc as usize);
                let spec = (self.spec(pc, i).map(|s| s[0])).or((seen == 0).then_some(Spec::I32));
                let Some(k) = Spec::by_fact(self.e.fact(x)).or(spec) else {
                    return self.deopt(ExitTag::Unsupported);
                };
                let s = self.state();
                let aux = (k as u16) << SPEC_SHIFT[0];
                let v = self.e.emit(Op::Unm, Rep::Val, aux, x.0, s.0, 0);
                self.set(a, v);
            }
            LEN => {
                let x = self.reg(b);
                let v = self.helped(Helper::Len, 0, &[x]);
                self.set(a, v);
            }
            LOADI => {
                let v = self.e.k64(Rep::Val, Value::small(i.imm()).to_raw());
                self.set(a, v);
            }
            LOADNIL => {
                let nil = self.e.k64(Rep::Val, Value::nil().to_raw());
                for r in a..a + b {
                    self.set(r, nil);
                }
            }
            JMP | JMP_BACK => self.jump(fall.unwrap()),
            LFALSESKIP => {
                let v = self.konst(Value::boolean(false));
                self.set(a, v);
                self.jump(fall.unwrap());
            }
            NOT => {
                let v = self.reg(b);
                let t = self.e.emit(Op::Truthy, Rep::B1, 0, v.0, 0, 0);
                let (no, yes) = (
                    self.konst(Value::boolean(false)),
                    self.konst(Value::boolean(true)),
                );
                let r = self.e.emit(Op::Select, Rep::Val, 0, t.0, no.0, yes.0);
                self.set(a, r);
            }
            JT | JF => {
                let v = self.reg(a);
                let t = self.e.emit(Op::Truthy, Rep::B1, 0, v.0, 0, 0);
                let (fall, jump) = (fall.unwrap(), jump.unwrap());
                if op == JT {
                    self.br(t, jump, fall)
                } else {
                    self.br(t, fall, jump)
                }
            }
            // The register is written on the edge taken only.
            JTSET | JFSET => {
                let v = self.reg(b);
                let t = self.e.emit(Op::Truthy, Rep::B1, 0, v.0, 0, 0);
                let (fall, jump) = (fall.unwrap(), jump.unwrap());
                match self.e.known(t) {
                    Some(k) if k == (op == JTSET) => {
                        self.set(a, v);
                        self.jump(jump);
                    }
                    Some(_) => self.jump(fall),
                    None => {
                        let edge = self.e.new_block();
                        let out = self.target(fall);
                        if op == JTSET {
                            self.e.br(t, edge, out);
                        } else {
                            self.e.br(t, out, edge);
                        }
                        let lp = self.ir_loop(self.bb);
                        self.e.begin(edge, lp, 0, BlockFlags::empty());
                        self.set(a, v);
                        self.jump(jump);
                    }
                }
            }
            JEQS | JNEQS => {
                // A string is equal to nothing but itself.
                let k = self.konst(self.proto.constants[i.h() as usize]);
                let x = self.reg(a);
                let c = self.e.cmp(Op::Cmp, Cc::Eq, x, k);
                let (fall, jump) = (fall.unwrap(), jump.unwrap());
                if op == JEQS {
                    self.br(c, jump, fall)
                } else {
                    self.br(c, fall, jump)
                }
            }
            RETURN if b != 0 => {
                // What a return would close is the interpreter's to close: a
                // frame entered mid-way may have it.
                // Open upvalues of its own registers it closes itself.
                use crate::vm::frame::flag::{HAS_OPEN, HAS_TBC};
                let captures = self.cfg.captured != RegSet::EMPTY;
                let mask = if captures {
                    HAS_TBC
                } else {
                    HAS_OPEN | HAS_TBC
                };
                let set = self.frame_flags(mask);
                let zero = self.e.k64(Rep::I64, 0);
                let none = self.e.cmp(Op::Cmp, Cc::Eq, set, zero);
                let state = self.state();
                let tag = ExitTag::Unsupported as u16;
                self.e
                    .emit(Op::Guard, Rep::None, tag, none.0, state.0, u32::MAX);
                for r in a..a + b - 1 {
                    let v = self.reg(r);
                    self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
                }
                if captures {
                    let open = self.frame_flags(HAS_OPEN);
                    let any = self.e.cmp(Op::Cmp, Cc::Ne, open, zero);
                    self.e.unless(any, |e| {
                        e.call(Helper::Close, 0, &[]);
                    });
                }
                self.e
                    .emit(Op::Ret, Rep::None, 0, a as u32, b as u32 - 1, 0);
            }
            RETURN0 => {
                self.e.emit(Op::Ret, Rep::None, 0, 0, 0, 0);
            }
            // Nothing to close: the compiler emits `RETURN` where there is.
            RETURN1 => {
                // Nothing where the slot holds it already.
                let v = self.reg(a);
                self.e.emit(Op::SlotStore, Rep::None, a as u16, v.0, 0, 0);
                self.e.emit(Op::Ret, Rep::None, 0, a as u32, 1, 0);
            }
            FORPREP => self.forprep(pc, i, fall.unwrap(), jump.unwrap()),
            // A loop's kind is its form's, unless its site or its `FORPREP`'s
            // saw the other since (a compiled float loop that met integers
            // records them: no `FORPREP` rewrites a form that is a JIT word).
            // A site that saw both kinds has no one representation.
            FORLOOP_I | FORLOOP_F => {
                let prep = (pc as i64 + i.branch_offset() as i64) as usize;
                let seen = (self.proto.feedback.bits(pc as usize) | self.proto.feedback.bits(prep))
                    & fb::KINDS;
                let (ints, floats) = (seen & (fb::SMALL | fb::BIGINT) != 0, seen & fb::FLOAT != 0);
                let (exit, back) = (fall.unwrap(), jump.unwrap());
                if (ints && floats) || seen & fb::BIGINT != 0 {
                    self.deopt(ExitTag::Unsupported)
                } else if floats || (op == FORLOOP_F && !ints) {
                    self.forloop_f(a, exit, back)
                } else {
                    self.forloop_i(a, exit, back)
                }
            }
            _ => match info.family {
                Family::RegArith | Family::ImmArith | Family::RegBit | Family::ImmBit
                    if !matches!(info.kind, ArithKind::Shl | ArithKind::Shr | ArithKind::None) =>
                {
                    let reg = self.reg(b);
                    let regs = matches!(info.family, Family::RegArith | Family::RegBit);
                    let (x, y) = if regs {
                        (reg, self.reg(i.c()))
                    } else {
                        let k = if i.imm_is_int() {
                            i32::try_from(i.imm_int()).ok().map(Value::small)
                        } else {
                            Some(Value::float(i.imm_float()))
                        };
                        let Some(k) = k.map(|k| self.konst(k)) else {
                            return self.deopt(ExitTag::Unsupported);
                        };
                        if info.reversed { (k, reg) } else { (reg, k) }
                    };
                    if let Some(v) = self.generic(Op::Arith, Rep::Val, info.kind as u16, x, y) {
                        self.set(a, v);
                    }
                }
                f if matches!(f, Family::CmpReg | Family::CmpImm)
                    || matches!(i.generic_op(), JEQI | JNEQI) =>
                {
                    let x = self.reg(a);
                    let (fall, jump) = (fall.unwrap(), jump.unwrap());
                    if matches!(i.generic_op(), JEQ | JNEQ) {
                        // Nil, booleans, strings, functions and threads are
                        // equal only to themselves.
                        let y = self.reg(b);
                        let bits = TypeSet::NIL
                            | TypeSet::BOOL
                            | TypeSet::STR
                            | TypeSet::FUN
                            | TypeSet::THR;
                        let by_bits = [x, y].iter().any(|&v| {
                            let f = self.e.fact(v);
                            !f.pending() && bits.contains(f.set)
                        });
                        if by_bits {
                            let c = self.e.cmp(Op::Cmp, Cc::Eq, x, y);
                            return match i.generic_op() {
                                JEQ => self.br(c, jump, fall),
                                _ => self.br(c, fall, jump),
                            };
                        }
                    }
                    let (cc, y) = match i.generic_op() {
                        JLT | JNLT => (Cc::Lt, self.reg(b)),
                        JLE | JNLE => (Cc::Le, self.reg(b)),
                        JEQ | JNEQ => (Cc::Eq, self.reg(b)),
                        g => {
                            let cc = match g {
                                JEQI | JNEQI => Cc::Eq,
                                JLTI | JNLTI => Cc::Lt,
                                JLEI | JNLEI => Cc::Le,
                                JGTI | JNGTI => Cc::Gt,
                                JGEI | JNGEI => Cc::Ge,
                                _ => return self.deopt(ExitTag::Unsupported),
                            };
                            let k = i.cmp_imm_int();
                            let k = if i.cmp_imm().is_float() {
                                Some(Value::float(k as f64))
                            } else {
                                i32::try_from(k).ok().map(Value::small)
                            };
                            match k.map(|k| self.konst(k)) {
                                Some(k) => (cc, k),
                                None => return self.deopt(ExitTag::Unsupported),
                            }
                        }
                    };
                    let Some(c) = self.generic(Op::Compare, Rep::B1, cc as u16, x, y) else {
                        return;
                    };
                    match i.generic_op().branch_sense() {
                        Some(false) => self.br(c, fall, jump),
                        _ => self.br(c, jump, fall),
                    }
                }
                _ => match i.op().unquickened() {
                    GETFIELD | GETTABUP | GETTABUP_REF => self.getfield(i),
                    SELF => self.self_(i),
                    CLOSURE => self.closure_new(pc, i),
                    CLOSE => self.close(i),
                    // The interpreter's, until the frame can be replaced
                    // in compiled code.
                    TAILCALL => self.dispatch(),
                    SETFIELD | SETTABUP | SETTABUP_REF => self.setfield(i),
                    _ if matches!(
                        i.op(),
                        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1
                    ) =>
                    {
                        self.call(pc, i)
                    }
                    _ => self.deopt(ExitTag::Unsupported),
                },
            },
        }
    }

    /// The f64 of a loop value in a float loop: a small integer converts.
    fn loop_f64(&mut self, v: Val, state: Val) -> Val {
        if Spec::by_fact(self.e.fact(v)) == Some(Spec::I32) {
            let x = self.e.check_type(v, TypeSet::SMALL, ExitTag::Type, state);
            self.e.op1(Op::IToF, Rep::F64, x)
        } else {
            self.e.check_type(v, TypeSet::FLOAT, ExitTag::Type, state)
        }
    }

    /// `FORPREP`, as the kind its loop has been: after it `R[a]` is the last
    /// value the control variable takes (the limit, in a float loop),
    /// `R[a+1]` the step, `R[a+2]` the variable and `R[a+3]` its copy.
    fn forprep(&mut self, pc: u32, i: Instruction, body: u32, skip: u32) {
        let a = i.a();
        let at = (pc as i64 + i.branch_offset() as i64) as usize;
        let fb_of = |pc: usize| self.proto.feedback.bits(pc);
        let seen = (fb_of(pc as usize) | fb_of(at)) & fb::KINDS;
        let (init, limit, step) = (self.reg(a), self.reg(a + 1), self.reg(a + 2));
        let state = self.state();
        let form = self.cfg.code[at].op();
        let zero_step = ExitTag::Slow as u16;
        if form == Bc::FORLOOP_I && seen & !fb::SMALL == 0 {
            let small =
                |s: &mut Self, v: Val| s.e.check_type(v, TypeSet::SMALL, ExitTag::Type, state);
            let (i0, l, st) = (small(self, init), small(self, limit), small(self, step));
            let zero = self.e.k32(Rep::I32, 0);
            // A zero step raises.
            let moves = self.e.cmp(Op::Cmp, Cc::Ne, st, zero);
            (self.e).emit(Op::Guard, Rep::None, zero_step, moves.0, state.0, u32::MAX);
            // last = init + (limit - init) / step * step: exact where the
            // loop runs, when the quotient is not negative; unused where it
            // does not.
            let wide = |s: &mut Self, v: Val| s.e.op1(Op::SExt32, Rep::I64, v);
            let (i64_, l64, s64) = (wide(self, i0), wide(self, l), wide(self, st));
            let span = self.e.op2(Op::Sub, Rep::I64, l64, i64_);
            let count = self.e.op2(Op::SDiv, Rep::I64, span, s64);
            let off = self.e.op2(Op::Mul, Rep::I64, count, s64);
            let last = self.e.op2(Op::Add, Rep::I64, i64_, off);
            let last = self.e.op1(Op::Trunc32, Rep::I32, last);
            let last = self.e.op1(Op::BoxI32, Rep::Val, last);
            let up = self.e.cmp(Op::Cmp, Cc::Gt, st, zero);
            let (over, under) = (
                self.e.cmp(Op::Cmp, Cc::Gt, i0, l),
                self.e.cmp(Op::Cmp, Cc::Lt, i0, l),
            );
            let past = self.e.emit(Op::Select, Rep::B1, 0, up.0, over.0, under.0);
            self.set(a, last);
            self.set(a + 1, step);
            self.set(a + 2, init);
            self.set(a + 3, init);
            self.br(past, skip, body);
        } else if form == Bc::FORLOOP_F && seen & !(fb::FLOAT | fb::SMALL) == 0 {
            // Integer init and step are an integer loop, whatever ran before.
            let int = |s: &Self, v: Val| Spec::by_fact(s.e.fact(v)) == Some(Spec::I32);
            if int(self, init) && int(self, step) {
                return self.deopt(ExitTag::Type);
            }
            let (i0, l, st) = (
                self.loop_f64(init, state),
                self.loop_f64(limit, state),
                self.loop_f64(step, state),
            );
            let zero = self.e.kf64(0.0);
            let moves = self.e.cmp(Op::FCmp, Cc::Ne, st, zero);
            (self.e).emit(Op::Guard, Rep::None, zero_step, moves.0, state.0, u32::MAX);
            let up = self.e.cmp(Op::FCmp, Cc::Gt, st, zero);
            let (over, under) = (
                self.e.cmp(Op::FCmp, Cc::Lt, l, i0),
                self.e.cmp(Op::FCmp, Cc::Lt, i0, l),
            );
            let past = self.e.emit(Op::Select, Rep::B1, 0, up.0, over.0, under.0);
            let boxed = |s: &mut Self, v: Val| s.e.op1(Op::BoxF64, Rep::Val, v);
            let (lv, sv, iv) = (boxed(self, l), boxed(self, st), boxed(self, i0));
            self.set(a, lv);
            self.set(a + 1, sv);
            self.set(a + 2, iv);
            self.set(a + 3, iv);
            self.br(past, skip, body);
        } else if form == Bc::FORLOOP && seen == 0 {
            // The loop never ran: its kind is not known yet.
            self.deopt(ExitTag::NeverRan)
        } else {
            self.deopt(ExitTag::Unsupported)
        }
    }

    /// `FORLOOP_I`: around while the control variable is not the last value.
    fn forloop_i(&mut self, a: u8, exit: u32, back: u32) {
        let state = self.state();
        let small = |s: &mut Self, r: u8| {
            let v = s.reg(r);
            s.e.check_type(v, TypeSet::SMALL, ExitTag::Type, state)
        };
        let (last, step, idx) = (small(self, a), small(self, a + 1), small(self, a + 2));
        let go = self.e.cmp(Op::Cmp, Cc::Ne, idx, last);
        if self.e.known(go).is_some() {
            return self.deopt(ExitTag::Unsupported);
        }
        // The step is taken on the edge back: `idx` walks to `last` exactly,
        // so it stays in range.
        let edge = self.e.new_block();
        let out = self.target(exit);
        self.e.br(go, edge, out);
        let lp = self.ir_loop(self.bb);
        self.e.begin(edge, lp, 0, BlockFlags::empty());
        let next = self.e.op2(Op::Add, Rep::I32, idx, step);
        let next = self.e.op1(Op::BoxI32, Rep::Val, next);
        self.set(a + 2, next);
        self.set(a + 3, next);
        self.jump(back);
    }

    /// `FORLOOP_F`: the float loop steps, and goes around while the control
    /// variable is within the limit, on the step's side.
    fn forloop_f(&mut self, a: u8, exit: u32, back: u32) {
        let state = self.state();
        let float = |s: &mut Self, r: u8| {
            let v = s.reg(r);
            s.e.check_type(v, TypeSet::FLOAT, ExitTag::Type, state)
        };
        let (lim, step, idx) = (float(self, a), float(self, a + 1), float(self, a + 2));
        let next = self.e.op2(Op::FAdd, Rep::F64, idx, step);
        let zero = self.e.kf64(0.0);
        let up = self.e.cmp(Op::FCmp, Cc::Gt, step, zero);
        let (below, above) = (
            self.e.cmp(Op::FCmp, Cc::Le, next, lim),
            self.e.cmp(Op::FCmp, Cc::Le, lim, next),
        );
        let go = self.e.emit(Op::Select, Rep::B1, 0, up.0, below.0, above.0);
        match self.e.known(go) {
            Some(true) => {}
            Some(false) => return self.jump(exit),
            None => {
                let edge = self.e.new_block();
                let out = self.target(exit);
                self.e.br(go, edge, out);
                let lp = self.ir_loop(self.bb);
                self.e.begin(edge, lp, 0, BlockFlags::empty());
            }
        }
        let next = self.e.op1(Op::BoxF64, Rep::Val, next);
        self.set(a + 2, next);
        self.set(a + 3, next);
        self.jump(back);
    }
}
