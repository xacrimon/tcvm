//! Bytecode to the high-level graph. Registers are boxed SSA values in the
//! emitter's table: a join where predecessors disagree makes a parameter, a
//! loop header one for each live register its body assigns, pending until
//! the loop's types are resolved (`ir::phis`). A site whose operands' facts
//! are final specialises as it is emitted; one that hangs on a loop
//! parameter stays generic, carrying what its feedback speculates.

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
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Heaps, Helper, Op};
use crate::jit::ir::table::{self, Key};
use crate::jit::ir::types::{Fact, FactFlags, Rep, TypeSet};
use crate::jit::ir::{Block, BlockFlags, HOME, Loop, Pool, Val};
use crate::jit::ir::{arith, compare};

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
    pool: &'a mut Pool,
    /// For looking a shape's transitions up; nothing is written.
    mc: &'a crate::dmm::Mutation<'gc>,
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
    /// The shapes the site records name: those the code compares with
    /// (`Sites::kept`) the region keeps alive.
    shapes: Vec<crate::env::Shape<'gc>>,
    /// The instructions compiled as a deopt for want of code.
    missing: Vec<Bc>,
}

/// Build the region of `proto` entered at `entry_pc` into `e`, which was
/// reset. A region the builder cannot make sets the emitter's bail.
pub(crate) fn build<'gc>(
    e: &mut Emitter,
    pool: &mut Pool,
    mc: &crate::dmm::Mutation<'gc>,
    cfg: &Cfg,
    proto: Gc<'gc, Prototype<'gc>>,
    entry_pc: u32,
) -> (Vec<crate::env::Shape<'gc>>, Vec<Bc>) {
    let nb = cfg.blocks.len();
    let root = cfg.block_of[entry_pc as usize];
    let (rpo, order) = cfg::rpo_of(&cfg.blocks, root);
    let mut b = Builder {
        e,
        pool,
        mc,
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
        missing: Vec::new(),
    };
    // A captured register lives in its slot, which the closures holding it
    // read and write: registers here are values only.
    if cfg.captured != RegSet::EMPTY {
        b.e.bail(Bail::Unsupported("captured registers"));
    } else if !b.find_loops(root, &rpo, &order) {
        b.e.bail(Bail::Unsupported("irreducible control flow"));
    }
    b.blocks = (0..nb).map(|_| b.e.new_block()).collect();
    for &bb in &rpo {
        if b.e.bail.is_some() {
            break;
        }
        b.block(bb, bb == root, entry_pc);
    }
    (b.shapes, b.missing)
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
            if key.is_reg() && at.iter().all(|&v| v == at[0]) {
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
            if key.is_reg() && !live.contains(key.index() as usize) {
                self.e.table.set(key, table::NONE);
            }
        }
    }

    // --- instructions -------------------------------------------------------

    /// What the compare site at `pc` saw, as `Compare`'s `aux` above its
    /// condition: by its quickened form (two small integers, or a float
    /// register) while no compiled code left it on kinds; else what its
    /// slow path and the exits recorded, with the small integers and floats
    /// its fast path leaves no record of.
    fn compared(&self, pc: u32, i: Instruction) -> u16 {
        let bits = self.proto.feedback.bits(pc as usize);
        let (op, generic) = (i.op(), i.generic_op());
        let form = op != generic && bits & fb::KIND_EXIT == 0;
        let kinds = match generic.info().family {
            Family::CmpReg if form => fb::SMALL,
            Family::CmpImm if form => fb::FLOAT,
            _ => bits & fb::KINDS | fb::SMALL | fb::FLOAT,
        };
        let mm = if bits & fb::MM != 0 { arith::MM } else { 0 };
        (kinds as u16) << compare::SEEN_SHIFT | mm
    }

    /// The kinds the arithmetic site at `pc` saw for each of its operands,
    /// and whether it overflowed or ran a metamethod, as `Arith`'s `aux`
    /// above the op: by its quickened form, else its feedback; for a site
    /// that never ran, by what the operands are known to be. `unrecorded`
    /// is what a site that ran leaves no record of. None, after a deopt,
    /// where nothing says anything.
    fn seen(&mut self, pc: u32, i: Instruction, vals: [Val; 2], unrecorded: u8) -> Option<u16> {
        let bits = self.proto.feedback.bits(pc as usize);
        let (op, generic) = (i.op(), i.generic_op());
        let info = generic.info();
        let i_adaptive = i.adaptive();
        let site = [
            (fb::SMALL, arith::SMALL),
            (fb::BIGINT, arith::BIG),
            (fb::FLOAT, arith::FLOAT),
            (fb::STR, arith::STR),
        ]
        .iter()
        .fold(0, |k, &(bit, kind)| {
            k | if bits & bit != 0 { kind } else { 0 }
        });
        let mm = matches!(op, Bc::ARITH_MM | Bc::ARITH_MM_R | Bc::ARITH_MMI);
        // A form no compiled code left on kinds says each operand's.
        let form = (op != generic && !mm && bits & fb::KIND_EXIT == 0)
            .then(|| info.forms.iter().position(|f| *f == Some(op)))
            .flatten();
        let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
        let (i, f) = (arith::SMALL, arith::FLOAT);
        let wider = site & (arith::BIG | arith::STR);
        // A site whose kinds kept changing is locked to its any-numbers
        // form, or to its generic opcode, and records no more of them.
        let (misses, locked) = match generic.info().adaptive_shift {
            0 => (0, false),
            _ => i_adaptive,
        };
        let mut kinds = match form {
            Some(0) => [i, i],
            Some(1) => [f, f],
            Some(2) if imm => [i, i],
            Some(2) => [i, f],
            Some(3) => [f, i],
            Some(_) => [i | f; 2],
            None if !mm && (locked || misses != 0) => [site | i | f; 2],
            None => [site | unrecorded; 2],
        };
        if form.is_some() {
            kinds = kinds.map(|k| k | wider);
        }
        // A site that never ran: what the operands are known to be, each
        // as itself or else as the other.
        if kinds == [0; 2] && !mm && bits & fb::MM == 0 {
            let own = vals.map(|v| {
                let f = self.e.fact(v);
                let known = !f.pending() && (TypeSet::NUM | TypeSet::STR).contains(f.set);
                if known { arith::kinds_of(f.set) } else { 0 }
            });
            kinds = [own[0] | own[1]; 2];
            if kinds == [0; 2] {
                self.deopt(ExitTag::NeverRan);
                return None;
            }
        }
        let mut aux =
            (kinds[0] as u16) << arith::SEEN_SHIFT[0] | (kinds[1] as u16) << arith::SEEN_SHIFT[1];
        if bits & fb::OVERFLOW != 0 {
            aux |= arith::OVERFLOWED;
        }
        if mm || bits & fb::MM != 0 {
            aux |= arith::MM;
        }
        Some(aux)
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
                let s = self.state();
                let v = self.e.emit(Op::Concat, Rep::Val, 0, x.0, y.0, s.0);
                self.set(a, v);
                // It allocates: past the collector's threshold the code
                // leaves after the instruction.
                let after = self.state_at(pc + 1);
                let tag = ExitTag::Gc as u16;
                self.e.emit(Op::GcCheck, Rep::None, tag, after.0, 0, 0);
            }
            NEWTABLE => self.newtable(pc, i),
            SETLIST => self.setlist(i),
            GETTABLE => self.gettable(pc, i),
            SETTABLE => self.settable(pc, i),
            GETUPVAL | GETUPVAL_REF => self.getupval(i),
            SETUPVAL => self.setupval(i),
            // The fast paths of a small integer record nothing.
            UNM => {
                let x = self.reg(b);
                if let Some(aux) = self.seen(pc, i, [x, x], arith::SMALL) {
                    let s = self.state();
                    let v = self
                        .e
                        .emit(Op::Unm, Rep::Val, aux & !(15 << 8), x.0, s.0, 0);
                    self.set(a, v);
                }
            }
            // `~x` is `x ~ -1`: the same coercions.
            BNOT => {
                let x = self.reg(b);
                let ones = self.e.k64(Rep::Val, Value::small(-1).to_raw());
                if let Some(aux) = self.seen(pc, i, [x, ones], arith::SMALL) {
                    let s = self.state();
                    let aux = aux | ArithKind::BXor as u16;
                    let v = self.e.emit(Op::Arith, Rep::Val, aux, x.0, ones.0, s.0);
                    self.set(a, v);
                }
            }
            LEN => {
                let x = self.reg(b);
                let s = self.state();
                // Lowered here where the operand's fact is final.
                let v = self.e.lower(Op::Len, Rep::Val, 0, x.0, s.0, 0);
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
                let base = self.e.emit(Op::Base, Rep::Ptr, 0, 0, 0, 0);
                let field = crate::jit::ir::fields::Field::FrameFlags as u16;
                let flags = self.e.emit(Op::Load, Rep::I64, field, base.0, u32::MAX, 0);
                let mask = crate::vm::frame::flag::HAS_OPEN | crate::vm::frame::flag::HAS_TBC;
                let mask = self.e.k64(Rep::I64, mask);
                let set = self.e.op2(Op::And, Rep::I64, flags, mask);
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
                    self.forloop(a, Spec::F64, exit, back)
                } else {
                    self.forloop(a, Spec::I32, exit, back)
                }
            }
            _ => match info.family {
                Family::RegArith | Family::ImmArith | Family::RegBit | Family::ImmBit
                    if info.kind != ArithKind::None =>
                {
                    // A metamethod form keeps the op it was in another
                    // field: read the operands of the instruction as made.
                    let mm = matches!(op, ARITH_MM | ARITH_MM_R | ARITH_MMI);
                    let i = if mm { i.unquickened() } else { i };
                    let (a, b) = (i.a(), i.b());
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
                    if let Some(aux) = self.seen(pc, self.cfg.code[pc as usize], [x, y], 0) {
                        let s = self.state();
                        let aux = aux | info.kind as u16;
                        let v = self.e.emit(Op::Arith, Rep::Val, aux, x.0, y.0, s.0);
                        self.set(a, v);
                    }
                }
                f if matches!(f, Family::CmpReg | Family::CmpImm)
                    || matches!(i.generic_op(), JEQI | JNEQI) =>
                {
                    let x = self.reg(a);
                    let (fall, jump) = (fall.unwrap(), jump.unwrap());
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
                    let aux = cc as u16 | self.compared(pc, i);
                    let s = self.state();
                    let c = self.e.emit(Op::Compare, Rep::B1, aux, x.0, y.0, s.0);
                    // No kind the site saw is possible here: it left.
                    if self.e.cur.is_none() {
                        return;
                    }
                    match i.generic_op().branch_sense() {
                        Some(false) => self.br(c, fall, jump),
                        _ => self.br(c, jump, fall),
                    }
                }
                _ => match i.op().unquickened() {
                    GETFIELD | GETTABUP | GETTABUP_REF => self.getfield(i),
                    SETFIELD | SETTABUP | SETTABUP_REF => self.setfield(i),
                    _ => self.deopt(ExitTag::Unsupported),
                },
            },
        }
    }

    /// `FORPREP`: after it `R[a]` is the last value the control variable
    /// takes (the limit, in a float loop), `R[a+1]` the step, `R[a+2]` the
    /// variable and `R[a+3]` its copy. The loop's kind is its values' where
    /// they say; this is the one its form and its feedback speculate.
    fn forprep(&mut self, pc: u32, i: Instruction, body: u32, skip: u32) {
        let a = i.a();
        let at = (pc as i64 + i.branch_offset() as i64) as usize;
        let fb_of = |pc: usize| self.proto.feedback.bits(pc);
        let seen = (fb_of(pc as usize) | fb_of(at)) & fb::KINDS;
        let kind = match self.cfg.code[at].op() {
            Bc::FORLOOP_I if seen & !fb::SMALL == 0 => Spec::I32,
            Bc::FORLOOP_F if seen & !(fb::FLOAT | fb::SMALL) == 0 => Spec::F64,
            // The loop never ran: its kind is not known yet.
            Bc::FORLOOP if seen == 0 => return self.deopt(ExitTag::NeverRan),
            _ => return self.deopt(ExitTag::Unsupported),
        };
        let (init, limit, step) = (self.reg(a), self.reg(a + 1), self.reg(a + 2));
        let (state, aux) = (self.state(), kind as u16);
        let start = (self.e).emit(Op::ForStart, Rep::Val, aux, init.0, step.0, state.0);
        let by = (self.e).emit(Op::ForStep, Rep::Val, aux, init.0, step.0, state.0);
        let three = |e: &mut Emitter, op: Op, rep: Rep| {
            let at = e.g.extra.len() as u32;
            e.g.extra.extend([init.0, limit.0, step.0]);
            e.emit(op, rep, aux, at, 3, state.0)
        };
        let last = three(self.e, Op::ForLimit, Rep::Val);
        let past = three(self.e, Op::ForSkip, Rep::B1);
        self.set(a, last);
        self.set(a + 1, by);
        self.set(a + 2, start);
        self.set(a + 3, start);
        self.br(past, skip, body);
    }

    /// `FORLOOP`: around while the loop has more to do, the step taken on
    /// the edge back. `kind` is what its form and feedback speculate.
    fn forloop(&mut self, a: u8, kind: Spec, exit: u32, back: u32) {
        let (state, aux) = (self.state(), kind as u16);
        let (last, step, idx) = (self.reg(a), self.reg(a + 1), self.reg(a + 2));
        let at = self.e.g.extra.len() as u32;
        self.e.g.extra.extend([idx.0, last.0, step.0]);
        let more = self.e.emit(Op::ForMore, Rep::B1, aux, at, 3, state.0);
        match self.e.known(more) {
            Some(true) => {}
            Some(false) => return self.jump(exit),
            None => {
                let edge = self.e.new_block();
                let out = self.target(exit);
                self.e.br(more, edge, out);
                let lp = self.ir_loop(self.bb);
                self.e.begin(edge, lp, 0, BlockFlags::empty());
            }
        }
        let next = self.e.emit(Op::ForNext, Rep::Val, aux, idx.0, step.0, 0);
        self.set(a + 2, next);
        self.set(a + 3, next);
        self.jump(back);
    }
}
