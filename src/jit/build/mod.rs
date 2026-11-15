//! Bytecode to the high-level graph. Registers are boxed SSA values in the
//! emitter's table: a join where predecessors disagree makes a parameter, a
//! loop header one for each live register its body assigns, pending until
//! the loop's types are resolved (`ir::phis`). A site whose operands' facts
//! are final specialises as it is emitted; one that hangs on a loop
//! parameter stays generic, carrying what its feedback speculates.

pub(crate) mod cfg;

use cfg::{Cfg, RegSet};

use crate::dmm::Gc;
use crate::env::function::Prototype;
use crate::env::value::Value;
use crate::instruction::{ArithKind, Family, Instruction, Op as Bc};
use crate::jit::compile::Bail;
use crate::jit::feedback as fb;
use crate::jit::ir::emit::Emitter;
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Heaps, Op};
use crate::jit::ir::table::{self, Key};
use crate::jit::ir::types::{Fact, FactFlags, Rep, TypeSet};
use crate::jit::ir::{Block, BlockFlags, HOME, Loop, Val};

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

struct Builder<'a, 'gc> {
    e: &'a mut Emitter,
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
}

/// Build the region of `proto` entered at `entry_pc` into `e`, which was
/// reset. A region the builder cannot make sets the emitter's bail.
pub(crate) fn build<'gc>(
    e: &mut Emitter,
    cfg: &Cfg,
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
) {
    let nb = cfg.blocks.len();
    let root = cfg.block_of[entry_pc as usize];
    let (rpo, order) = cfg::rpo_of(&cfg.blocks, root);
    let mut b = Builder {
        e,
        cfg,
        proto,
        loops: Vec::new(),
        loop_of: vec![None; nb],
        heads: vec![None; nb],
        blocks: Vec::new(),
        bb: root,
        pc: entry_pc,
        state: Val::NONE,
    };
    // A captured register lives in its slot, which the closures holding it
    // read and write: registers here are values only.
    if cfg.captured != RegSet::EMPTY {
        return b.e.bail(Bail::Unsupported("captured registers"));
    }
    if !b.find_loops(root, &rpo, &order) {
        return b.e.bail(Bail::Unsupported("irreducible control flow"));
    }
    b.blocks = (0..nb).map(|_| b.e.new_block()).collect();
    for &bb in &rpo {
        if b.e.bail.is_some() {
            return;
        }
        b.block(bb, bb == root, entry_pc);
    }
}

impl Builder<'_, '_> {
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

    fn set(&mut self, r: u8, v: Val) {
        self.e.table.set(Key::reg(r as u32), v.0);
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
            .map(|r| {
                let v = self.reg(r as u8);
                let home = self.e.slot_holds(r as u16, v);
                (r as u32 | if home { HOME } else { 0 }, v)
            })
            .collect();
        self.e
            .frame_state(pc, ExitKind::Before, Val::NONE, &entries)
    }

    fn deopt(&mut self, tag: ExitTag) {
        let s = self.state();
        self.e.emit(Op::Deopt, Rep::None, tag as u16, s.0, 0, 0);
    }

    fn konst(&mut self, v: Value<'_>) -> Option<Val> {
        let heap = TypeSet::HEAP.contains(TypeSet::of(v));
        (!heap).then(|| self.e.k64(Rep::Val, v.to_raw()))
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
                self.set(r as u8, p);
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
            if !key.is_fact() && at.iter().all(|&v| v == at[0]) {
                let v = at[0];
                self.e.table.set(key, v);
            }
        }
        for r in live.iter() {
            let key = Key::reg(r as u32);
            vals.clear();
            vals.extend((0..npreds).map(|k| Val(self.e.table.pred_value(key, k))));
            if vals.iter().any(|v| v.is_none()) {
                crate::jit::bug!("r{r} is live into pc {pc} without a value");
            }
            if closed.contains(r) || vals.iter().any(|&v| v != vals[0]) {
                let p = self.e.param(Rep::Val, None, &vals);
                self.e.table.set(key, p.0);
            }
        }
        // A register that is not live has no value here.
        for i in 0..self.e.table.merges() {
            let (key, _) = self.e.table.merge(i);
            if !key.is_fact() && !live.contains(key.index() as usize) {
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
            LOAD => match self.konst(self.proto.constants[i.d() as usize]) {
                Some(v) => self.set(a, v),
                None => self.deopt(ExitTag::Unsupported),
            },
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
            FORLOOP_I => {
                let state = self.state();
                let small = |s: &mut Self, r: u8| {
                    let v = s.reg(r);
                    s.e.check_type(v, TypeSet::SMALL, ExitTag::Type, state)
                };
                let (last, step, idx) = (small(self, a), small(self, a + 1), small(self, a + 2));
                let go = self.e.cmp(Op::Cmp, Cc::Ne, idx, last);
                let (exit, back) = (fall.unwrap(), jump.unwrap());
                if self.e.known(go).is_some() {
                    return self.deopt(ExitTag::Unsupported);
                }
                // The step is taken on the edge back: `idx` walks to `last`
                // exactly, so it stays in range.
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
            _ => match info.family {
                Family::RegArith | Family::ImmArith
                    if matches!(
                        info.kind,
                        ArithKind::Add | ArithKind::Sub | ArithKind::Mul | ArithKind::Div
                    ) =>
                {
                    let reg = self.reg(b);
                    let (x, y) = if info.family == Family::RegArith {
                        (reg, self.reg(i.c()))
                    } else {
                        let k = if i.imm_is_int() {
                            i32::try_from(i.imm_int()).ok().map(Value::small)
                        } else {
                            Some(Value::float(i.imm_float()))
                        };
                        let Some(k) = k.and_then(|k| self.konst(k)) else {
                            return self.deopt(ExitTag::Unsupported);
                        };
                        if info.reversed { (k, reg) } else { (reg, k) }
                    };
                    if let Some(v) = self.generic(Op::Arith, Rep::Val, info.kind as u16, x, y) {
                        self.set(a, v);
                    }
                }
                Family::CmpReg | Family::CmpImm => {
                    let x = self.reg(a);
                    let (cc, y) = match i.generic_op() {
                        JLT | JNLT => (Cc::Lt, self.reg(b)),
                        JLE | JNLE => (Cc::Le, self.reg(b)),
                        JEQ | JNEQ => (Cc::Eq, self.reg(b)),
                        g => {
                            let cc = match g {
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
                            match k.and_then(|k| self.konst(k)) {
                                Some(k) => (cc, k),
                                None => return self.deopt(ExitTag::Unsupported),
                            }
                        }
                    };
                    let Some(c) = self.generic(Op::Compare, Rep::B1, cc as u16, x, y) else {
                        return;
                    };
                    let (fall, jump) = (fall.unwrap(), jump.unwrap());
                    match i.generic_op().branch_sense() {
                        Some(false) => self.br(c, fall, jump),
                        _ => self.br(c, jump, fall),
                    }
                }
                _ => self.deopt(ExitTag::Unsupported),
            },
        }
    }

    /// An integer loop stepping by a constant 1 or -1, which the loop's
    /// form says it has been; anything else is left to the interpreter.
    fn forprep(&mut self, pc: u32, i: Instruction, body: u32, skip: u32) {
        let a = i.a();
        let at = (pc as i64 + i.branch_offset() as i64) as usize;
        let seen =
            (self.proto.feedback.bits(pc as usize) | self.proto.feedback.bits(at)) & fb::KINDS;
        let (init, limit, step) = (self.reg(a), self.reg(a + 1), self.reg(a + 2));
        let up = match self.e.g.inst(step).k64() {
            _ if self.e.g.op(step) != Op::K64 => None,
            w if w == Value::small(1).to_raw() => Some(true),
            w if w == Value::small(-1).to_raw() => Some(false),
            _ => None,
        };
        let (Some(up), true) = (
            up,
            self.cfg.code[at].op() == Bc::FORLOOP_I && seen & !fb::SMALL == 0,
        ) else {
            return self.deopt(ExitTag::Unsupported);
        };
        let state = self.state();
        let i0 = self
            .e
            .check_type(init, TypeSet::SMALL, ExitTag::Type, state);
        let l = self
            .e
            .check_type(limit, TypeSet::SMALL, ExitTag::Type, state);
        let past = self.e.cmp(Op::Cmp, if up { Cc::Gt } else { Cc::Lt }, i0, l);
        // The last value, the step, the control variable and its copy.
        self.set(a, limit);
        self.set(a + 1, step);
        self.set(a + 2, init);
        self.set(a + 3, init);
        self.br(past, skip, body);
    }
}
