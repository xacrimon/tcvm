//! The builder (`jit-design.md` 8.7): one pass over the bytecode from the
//! entry, Braun-style SSA over the registers, home-slot tracking for the
//! stores at tail-outs and the entries of snapshots, and the outer-loop
//! peeling of a loop entry inside a nest.

pub(crate) mod cfg;
mod emit;

use std::collections::HashMap;

use crate::dmm::Gc;
use crate::env::function::{LuaFn, Prototype};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::cfg::{Cfg, RegSet};
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Refine, Rep, Ty, TypeSet};
use crate::jit::ir::{
    Block, BlockCall, ExitKind, Func, Inst, RegionMeta, Snap, SnapData, Val, ValDef,
};
use crate::lua::Context;

#[derive(Debug)]
pub(crate) enum BuildError {
    Irreducible,
    TooLarge,
}

/// A copy of a bytecode block in an outer-loop peeling version (8.7).
struct Instance {
    bc: u32,
    version: u32,
    ir: Block,
    succs: Vec<usize>,
    npreds: u32,
    done_preds: u32,
    started: bool,
    processed: bool,
}

/// What a home slot holds, as far as the region knows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotState {
    Unknown,
    /// The boxed form of this value.
    Holds(Val),
    /// The value at the root block's start, not yet loaded.
    Lazy(Block),
}

/// Per IR block SSA state.
struct BlockState {
    defs: Vec<Option<Val>>,
    slots: Vec<SlotState>,
    sealed: bool,
    incomplete: Vec<(u8, Val)>,
    preds: Vec<Block>,
    /// A block values do not flow into (the prologue, a resume block):
    /// registers read there load their home slot.
    root: bool,
    /// Where a root's loads are inserted.
    root_at: usize,
    /// Types of the slots of a root (what was stored before the call).
    root_ty: Vec<TypeSet>,
    /// The frame at a resume block's start, for guards of its loads.
    after: Option<Snap>,
    terminated: bool,
}

pub(crate) struct Builder<'a, 'gc> {
    pub(crate) f: Func<'gc>,
    pub(crate) ctx: Context<'gc>,
    pub(crate) closure: LuaFn<'gc>,
    pub(crate) proto: Gc<'gc, Prototype<'gc>>,
    pub(crate) cfg: &'a Cfg,
    nregs: usize,
    bs: Vec<BlockState>,
    cur: Block,
    instances: Vec<Instance>,
    inst_of: HashMap<(u32, u32), usize>,
    /// Writes of the instruction being emitted, applied at its end so its
    /// exits see the frame before it.
    pending: Vec<(u8, Val)>,
    /// The `Before` snapshot of the instruction being emitted.
    cur_snap: Option<Snap>,
    pub(crate) pc: u32,
    pub(crate) deopt_all: bool,
}

impl<'a, 'gc> Builder<'a, 'gc> {
    pub(crate) fn new(
        ctx: Context<'gc>,
        closure: LuaFn<'gc>,
        cfg: &'a Cfg,
        entry_pc: u32,
        loop_entry: bool,
    ) -> Self {
        let proto = closure.proto;
        let meta = RegionMeta {
            entry_pc,
            loop_entry,
            max_stack: proto.max_stack_size,
        };
        Builder {
            f: Func::new(meta),
            ctx,
            closure,
            proto,
            cfg,
            nregs: cfg.nregs,
            bs: Vec::new(),
            cur: Block(0),
            instances: Vec::new(),
            inst_of: HashMap::new(),
            pending: Vec::new(),
            cur_snap: None,
            pc: entry_pc,
            deopt_all: false,
        }
    }

    // --- blocks ---------------------------------------------------------------

    pub(crate) fn new_block(&mut self) -> Block {
        let b = self.f.new_block();
        self.bs.push(BlockState {
            defs: vec![None; self.nregs],
            slots: vec![SlotState::Unknown; self.nregs],
            sealed: false,
            incomplete: Vec::new(),
            preds: Vec::new(),
            root: false,
            root_at: 0,
            root_ty: Vec::new(),
            after: None,
            terminated: false,
        });
        b
    }

    /// A block whose registers load their home slots, with `tys` their
    /// known types.
    fn new_root(&mut self, tys: Vec<TypeSet>) -> Block {
        let b = self.new_block();
        let s = &mut self.bs[b.idx()];
        s.root = true;
        s.sealed = true;
        s.root_ty = tys;
        s.slots = vec![SlotState::Lazy(b); self.nregs];
        b
    }

    pub(crate) fn switch_to(&mut self, b: Block) {
        self.cur = b;
    }

    pub(crate) fn cur(&self) -> Block {
        self.cur
    }

    pub(crate) fn terminated(&self) -> bool {
        self.bs[self.cur.idx()].terminated
    }

    /// A block entered only from the current one, sealed.
    pub(crate) fn new_sealed_succ(&mut self) -> Block {
        let b = self.new_block();
        self.bs[b.idx()].sealed = true;
        b
    }

    fn seal(&mut self, b: Block) {
        if self.bs[b.idx()].sealed {
            return;
        }
        self.bs[b.idx()].sealed = true;
        let inc = std::mem::take(&mut self.bs[b.idx()].incomplete);
        for (r, p) in inc {
            self.add_param_args(b, r, p);
        }
        // The slot state of a loop header was assumed unknown; nothing to fix.
    }

    /// Append, on every edge into `b`, the argument for its new parameter
    /// `p` of register `r`.
    fn add_param_args(&mut self, b: Block, r: u8, _p: Val) {
        let preds = self.bs[b.idx()].preds.clone();
        // One argument per edge, in the order the edges were added.
        let mut seen: HashMap<Block, usize> = HashMap::new();
        for pred in preds {
            let v = self.read_var_at(r, pred);
            let k = seen.entry(pred).or_insert(0);
            let nth = *k;
            *k += 1;
            let term = self
                .f
                .terminator(pred)
                .expect("a predecessor is terminated");
            let mut found = 0;
            for e in self.f.edges_mut(term) {
                if e.target == b {
                    if found == nth {
                        e.args.push(v);
                        break;
                    }
                    found += 1;
                }
            }
        }
    }

    // --- SSA over registers ---------------------------------------------------

    pub(crate) fn read_var_at(&mut self, r: u8, b: Block) -> Val {
        if let Some(v) = self.bs[b.idx()].defs[r as usize] {
            return v;
        }
        // Walk up single-predecessor chains iteratively.
        let mut chain = vec![b];
        let mut x = b;
        let v = loop {
            let s = &self.bs[x.idx()];
            if let Some(v) = s.defs[r as usize] {
                break v;
            }
            if s.root {
                break self.root_load(x, r);
            }
            if !s.sealed {
                let p = self.f.add_param(x, Ty::ANY);
                self.bs[x.idx()].incomplete.push((r, p));
                self.bs[x.idx()].defs[r as usize] = Some(p);
                break p;
            }
            if s.preds.len() == 1 {
                x = s.preds[0];
                chain.push(x);
                continue;
            }
            if s.preds.is_empty() {
                // Unreachable code reads nothing meaningful.
                let v = self.konst_in(x, Value::nil());
                break v;
            }
            let p = self.f.add_param(x, Ty::ANY);
            self.bs[x.idx()].defs[r as usize] = Some(p);
            self.add_param_args(x, r, p);
            break p;
        };
        for c in chain {
            self.bs[c.idx()].defs[r as usize].get_or_insert(v);
        }
        v
    }

    fn root_load(&mut self, b: Block, r: u8) -> Val {
        let set = self.bs[b.idx()]
            .root_ty
            .get(r as usize)
            .copied()
            .unwrap_or(TypeSet::ANY);
        let inst = self.f.make_inst(Op::Load(r), &[], None, ExitTag::Type);
        self.f.insts[inst.idx()].block = b;
        let at = self.bs[b.idx()].root_at;
        self.f.blocks[b.idx()].insts.insert(at, inst);
        self.bs[b.idx()].root_at += 1;
        let v = self.f.result(inst);
        self.f.vals[v.idx()].ty = Ty::val(if set.is_empty() { TypeSet::ANY } else { set });
        self.bs[b.idx()].defs[r as usize] = Some(v);
        self.bs[b.idx()].slots[r as usize] = SlotState::Holds(v);
        if let Some(s) = self.bs[b.idx()].after {
            self.f.def_snaps.push((v, s));
        }
        v
    }

    fn konst_in(&mut self, b: Block, v: Value<'gc>) -> Val {
        let inst = self
            .f
            .make_inst(Op::KVal(v.to_raw()), &[], None, ExitTag::Type);
        self.f.insts[inst.idx()].block = b;
        self.f.blocks[b.idx()].insts.insert(0, inst);
        if self.bs[b.idx()].root {
            self.bs[b.idx()].root_at += 1;
        }
        self.f.result(inst)
    }

    /// The register's value before the instruction being emitted.
    pub(crate) fn reg(&mut self, r: u8) -> Val {
        self.read_var_at(r, self.cur)
    }

    /// Write `r` at the end of the instruction.
    pub(crate) fn set(&mut self, r: u8, v: Val) {
        if (r as usize) < self.nregs {
            self.pending.push((r, v));
        }
    }

    /// Apply the instruction's writes now.
    pub(crate) fn flush(&mut self) {
        for (r, v) in std::mem::take(&mut self.pending) {
            let s = &mut self.bs[self.cur.idx()];
            s.defs[r as usize] = Some(v);
            // A slot whose root value is unchanged still holds it.
            let keep = match s.slots[r as usize] {
                SlotState::Holds(h) => h == self.root_of(v),
                _ => false,
            };
            if !keep {
                self.bs[self.cur.idx()].slots[r as usize] = SlotState::Unknown;
            }
        }
        self.cur_snap = None;
    }

    /// The value `v` is a representation of: through guards, boxes and unboxes.
    pub(crate) fn root_of(&self, mut v: Val) -> Val {
        loop {
            match self.f.vals[v.idx()].def {
                ValDef::Inst(i, _) => match self.f.op(i) {
                    Op::Guard(_) | Op::Box | Op::Unbox(_) => v = self.f.args(i)[0],
                    _ => return v,
                },
                ValDef::Param(..) => return v,
            }
        }
    }

    /// Whether `r`'s home slot holds `v` now.
    fn slot_holds(&mut self, r: u8, v: Val) -> bool {
        match self.bs[self.cur.idx()].slots[r as usize] {
            SlotState::Holds(h) => h == self.root_of(v),
            SlotState::Lazy(root) => {
                // Untouched since the root: `v` is the root's load of `r`.
                let rv = self.root_of(v);
                matches!(self.f.op_of_val(rv), Some(Op::Load(lr)) if lr == r)
                    && self.f.def_block(rv) == Some(root)
            }
            SlotState::Unknown => false,
        }
    }

    // --- instructions -------------------------------------------------------

    pub(crate) fn push(&mut self, op: Op, args: &[Val]) -> Inst {
        let i = self.f.make_inst(op, args, None, ExitTag::Type);
        self.f.append(self.cur, i);
        i
    }

    /// A pure instruction's single result.
    pub(crate) fn ins(&mut self, op: Op, args: &[Val]) -> Val {
        let i = self.push(op, args);
        self.f.result(i)
    }

    /// An instruction that may exit, with the instruction's `Before`
    /// snapshot.
    pub(crate) fn ins_exit(&mut self, op: Op, args: &[Val], tag: ExitTag) -> Inst {
        let s = self.snap_before();
        let i = self.f.make_inst(op, args, Some(s), tag);
        self.f.append(self.cur, i);
        i
    }

    pub(crate) fn ins_snap(&mut self, op: Op, args: &[Val], snap: Snap, tag: ExitTag) -> Inst {
        let i = self.f.make_inst(op, args, Some(snap), tag);
        self.f.append(self.cur, i);
        i
    }

    pub(crate) fn set_ty(&mut self, v: Val, ty: Ty) {
        self.f.vals[v.idx()].ty = ty;
    }

    pub(crate) fn ty(&self, v: Val) -> Ty {
        self.f.ty(v)
    }

    /// End the current block with `op` and these edges.
    pub(crate) fn terminate(
        &mut self,
        op: Op,
        args: &[Val],
        edges: Vec<Block>,
        snap: Option<(Snap, ExitTag)>,
    ) -> Inst {
        debug_assert!(!self.terminated());
        let (s, tag) = match snap {
            Some((s, t)) => (Some(s), t),
            None => (None, ExitTag::Type),
        };
        let i = self.f.make_inst(op, args, s, tag);
        self.f.append(self.cur, i);
        let calls = edges
            .iter()
            .map(|&t| BlockCall {
                target: t,
                args: Vec::new(),
            })
            .collect();
        self.f.set_edges(i, calls);
        for &t in &edges {
            self.bs[t.idx()].preds.push(self.cur);
        }
        self.bs[self.cur.idx()].terminated = true;
        // Edges into blocks already sealed get their arguments now.
        for (k, &t) in edges.iter().enumerate() {
            if self.bs[t.idx()].sealed && !self.f.blocks[t.idx()].params.is_empty() {
                let params = self.f.blocks[t.idx()].params.clone();
                let regs: Vec<u8> = params.iter().map(|&p| self.param_reg(t, p)).collect();
                let mut args = Vec::new();
                for r in regs {
                    args.push(self.read_var_at(r, self.cur));
                }
                self.f.edges_mut(i)[k].args = args;
            }
        }
        i
    }

    /// The register a block parameter stands for.
    fn param_reg(&self, b: Block, p: Val) -> u8 {
        let defs = &self.bs[b.idx()].defs;
        if let Some(r) = self.bs[b.idx()]
            .incomplete
            .iter()
            .find(|(_, q)| *q == p)
            .map(|&(r, _)| r)
        {
            return r;
        }
        for (r, d) in defs.iter().enumerate() {
            if *d == Some(p) {
                return r as u8;
            }
        }
        panic!("block parameter without a register");
    }

    pub(crate) fn jump(&mut self, target: Block) {
        self.terminate(Op::Jump, &[], vec![target], None);
    }

    pub(crate) fn br(&mut self, cond: Val, t: Block, f: Block) {
        self.terminate(Op::Br, &[cond], vec![t, f], None);
    }

    // --- snapshots and tail-outs ----------------------------------------------

    /// Registers whose values an exit or a tail-out at `live` must provide:
    /// the live ones and every captured one.
    fn exit_regs(&self, live: RegSet) -> RegSet {
        let mut s = live;
        s.union(&self.cfg.captured);
        s
    }

    fn snapshot(&mut self, pc: u32, kind: ExitKind, live: RegSet) -> Snap {
        let regs = self.exit_regs(live);
        let mut entries = Vec::new();
        for r in regs.iter() {
            if r >= self.nregs || self.untouched(r) {
                continue;
            }
            let v = self.reg(r as u8);
            if !self.slot_holds(r as u8, v) {
                entries.push((r as u8, v));
            }
        }
        self.f.add_snap(SnapData { pc, kind, entries })
    }

    /// The snapshot that re-runs the instruction being emitted.
    pub(crate) fn snap_before(&mut self) -> Snap {
        if let Some(s) = self.cur_snap {
            return s;
        }
        let live = self.cfg.live_in[self.pc as usize];
        let s = self.snapshot(self.pc, ExitKind::Before, live);
        self.cur_snap = Some(s);
        s
    }

    /// The snapshot that continues after the instruction at `pc`, once its
    /// writes are flushed.
    pub(crate) fn snap_after(&mut self, pc: u32) -> Snap {
        let live = match self.cfg.live_in.get(pc as usize + 1) {
            Some(&l) => l,
            None => RegSet::EMPTY,
        };
        self.snapshot(pc, ExitKind::After, live)
    }

    /// Whether `r`'s home slot still holds what it held at the root: the
    /// register is the root's (perhaps not yet created) load.
    fn untouched(&self, r: usize) -> bool {
        matches!(self.bs[self.cur.idx()].slots[r], SlotState::Lazy(_))
    }

    /// Note the `After` snapshot of `v`'s definition, the instruction at the
    /// current pc, and of the loads of the block it starts when that is a
    /// resume block (9.1).
    pub(crate) fn note_def(&mut self, vals: &[Val], resume: Option<Block>) {
        self.flush();
        let s = self.snap_after(self.pc);
        for &v in vals {
            self.f.def_snaps.push((v, s));
        }
        if let Some(b) = resume {
            self.bs[b.idx()].after = Some(s);
        }
    }

    /// Store the registers in `live` (and the captured ones) whose home
    /// slots do not hold their values, before a tail-out. Returns the types
    /// the slots hold.
    pub(crate) fn store_live(&mut self, live: RegSet) -> Vec<TypeSet> {
        let regs = self.exit_regs(live);
        let mut tys = vec![TypeSet::ANY; self.nregs];
        for r in regs.iter() {
            if r >= self.nregs {
                continue;
            }
            if let SlotState::Lazy(root) = self.bs[self.cur.idx()].slots[r] {
                let t = self.bs[root.idx()].root_ty.get(r).copied();
                tys[r] = t.filter(|t| !t.is_empty()).unwrap_or(TypeSet::ANY);
                continue;
            }
            let v = self.reg(r as u8);
            if !self.slot_holds(r as u8, v) {
                let b = self.boxed(v);
                self.push(Op::Store(r as u8), &[b]);
                self.bs[self.cur.idx()].slots[r as usize] = SlotState::Holds(self.root_of(v));
            }
            tys[r] = self.ty(v).set;
        }
        for r in self.cfg.captured.iter() {
            if r < self.nregs {
                tys[r] = TypeSet::ANY;
            }
        }
        tys
    }

    /// Leave for the interpreter at the instruction being emitted.
    pub(crate) fn deopt(&mut self, tag: ExitTag) {
        let s = self.snap_before();
        self.terminate(Op::Deopt, &[], vec![], Some((s, tag)));
    }

    /// End the block with a call: `Call` to a fresh resume block, which
    /// becomes current.
    pub(crate) fn call(&mut self, op: Op, slot_tys: Vec<TypeSet>, c: u8) -> Block {
        let resume = self.new_root(slot_tys);
        self.f.blocks[resume.idx()].resume = true;
        self.terminate(op, &[], vec![resume], None);
        self.bs[resume.idx()].preds.clear();
        self.switch_to(resume);
        let r = self.push(Op::Resume { c }, &[]);
        self.bs[resume.idx()].root_at = 1;
        let _ = r;
        resume
    }

    // --- values -------------------------------------------------------------

    pub(crate) fn konst(&mut self, v: Value<'gc>) -> Val {
        if let Some(i) = v.get_small() {
            let k = self.ins(Op::KVal(v.to_raw()), &[]);
            let _ = i;
            return k;
        }
        if v.is_float() || v.is_nil() || v.get_boolean().is_some() {
            return self.ins(Op::KVal(v.to_raw()), &[]);
        }
        let idx = self.f.intern(v);
        let k = self.ins(Op::KObj(idx), &[]);
        let set = value_set(v);
        self.set_ty(
            k,
            Ty {
                rep: Rep::Val,
                set,
                refine: Refine::Const(idx),
            },
        );
        k
    }

    pub(crate) fn ki32(&mut self, n: i32) -> Val {
        self.ins(Op::KI32(n), &[])
    }

    pub(crate) fn kf64(&mut self, x: f64) -> Val {
        self.ins(Op::KF64(x.to_bits()), &[])
    }

    /// `v` as a `Val`.
    pub(crate) fn boxed(&mut self, v: Val) -> Val {
        let ty = self.ty(v);
        if ty.rep == Rep::Val {
            return v;
        }
        if let Some(Op::Unbox(_)) = self.f.def_op(v) {
            let src = self.f.args(self.f.def_inst(v).unwrap())[0];
            return src;
        }
        match self.f.def_op(v) {
            Some(Op::KI32(n)) => return self.konst(Value::small(n)),
            Some(Op::KF64(b)) => return self.konst(Value::float(f64::from_bits(b))),
            _ => {}
        }
        self.ins(Op::Box, &[v])
    }

    /// `v` guarded to `set`, exiting at the instruction being emitted.
    pub(crate) fn guard(&mut self, v: Val, set: TypeSet) -> Val {
        let ty = self.ty(v);
        if ty.within(set) && !(self.deopt_all && ty.rep == Rep::Val) {
            return v;
        }
        let i = self.ins_exit(Op::Guard(set), &[v], ExitTag::Type);
        self.f.result(i)
    }

    /// The i32 of `v`, guarded small.
    pub(crate) fn as_i32(&mut self, v: Val) -> Val {
        let ty = self.ty(v);
        match ty.rep {
            Rep::I32 => return v,
            Rep::I64 => {
                let i = self.ins_exit(Op::LToI, &[v], ExitTag::Type);
                return self.f.result(i);
            }
            Rep::F64 => {
                let i = self.ins_exit(Op::FToIExact, &[v], ExitTag::Type);
                return self.f.result(i);
            }
            _ => {}
        }
        if let Some(Op::KVal(bits)) = self.f.def_op(v) {
            let k: Value<'_> = unsafe { std::mem::transmute::<u64, Value<'static>>(bits) };
            if let Some(n) = k.get_small() {
                return self.ki32(n);
            }
        }
        if let Some(Op::Box) = self.f.def_op(v) {
            let src = self.f.args(self.f.def_inst(v).unwrap())[0];
            if self.ty(src).rep == Rep::I32 {
                return src;
            }
        }
        let g = self.guard(v, TypeSet::SMALL);
        self.ins(Op::Unbox(Rep::I32), &[g])
    }

    /// The f64 of `v`, guarded float.
    pub(crate) fn as_f64(&mut self, v: Val) -> Val {
        let ty = self.ty(v);
        match ty.rep {
            Rep::F64 => return v,
            Rep::I32 => return self.ins(Op::IToF, &[v]),
            Rep::I64 => return self.ins(Op::LToF, &[v]),
            _ => {}
        }
        if let Some(Op::KVal(bits)) = self.f.def_op(v) {
            let k: Value<'_> = unsafe { std::mem::transmute::<u64, Value<'static>>(bits) };
            if let Some(x) = k.get_float() {
                return self.kf64(x);
            }
        }
        if let Some(Op::Box) = self.f.def_op(v) {
            let src = self.f.args(self.f.def_inst(v).unwrap())[0];
            if self.ty(src).rep == Rep::F64 {
                return src;
            }
        }
        let g = self.guard(v, TypeSet::FLOAT);
        self.ins(Op::Unbox(Rep::F64), &[g])
    }

    /// The f64 of a number: a float as is, a small integer converted.
    pub(crate) fn num_f64(&mut self, v: Val) -> Val {
        let ty = self.ty(v);
        if ty.rep == Rep::Val && ty.within(TypeSet::SMALL) {
            let i = self.as_i32(v);
            return self.ins(Op::IToF, &[i]);
        }
        self.as_f64(v)
    }

    /// The i64 of `v`, guarded to an integer.
    pub(crate) fn as_i64(&mut self, v: Val) -> Val {
        let ty = self.ty(v);
        match ty.rep {
            Rep::I64 => return v,
            Rep::I32 => return self.ins(Op::IToL, &[v]),
            _ => {}
        }
        if ty.rep == Rep::Val && ty.within(TypeSet::SMALL) {
            let i = self.as_i32(v);
            return self.ins(Op::IToL, &[i]);
        }
        if let Some(Op::Box) = self.f.def_op(v) {
            let src = self.f.args(self.f.def_inst(v).unwrap())[0];
            match self.ty(src).rep {
                Rep::I64 => return src,
                Rep::I32 => return self.ins(Op::IToL, &[src]),
                _ => {}
            }
        }
        let g = self.guard(v, TypeSet::INT);
        self.ins(Op::Unbox(Rep::I64), &[g])
    }

    // --- the instance graph ---------------------------------------------------

    fn instance(&mut self, bc: u32, version: u32) -> usize {
        if let Some(&i) = self.inst_of.get(&(bc, version)) {
            return i;
        }
        let ir = self.new_block();
        self.instances.push(Instance {
            bc,
            version,
            ir,
            succs: Vec::new(),
            npreds: 0,
            done_preds: 0,
            started: false,
            processed: false,
        });
        let i = self.instances.len() - 1;
        self.inst_of.insert((bc, version), i);
        i
    }

    /// Discover the instances reachable from the entry, versioned for
    /// outer-loop peeling, and return them in reverse post-order.
    fn discover(&mut self, entry_bc: u32) -> Vec<usize> {
        let cfg = self.cfg;
        // The loops around the entry, outermost first.
        let mut nest = Vec::new();
        let mut l = if self.f.meta.loop_entry {
            cfg.loop_of[entry_bc as usize]
        } else {
            None
        };
        while let Some(i) = l {
            nest.push(i);
            l = cfg.loops[i].parent;
        }
        nest.reverse();
        let n = nest.len() as u32;
        // The deepest nest level holding a block, plus one (0 outside the nest).
        let level = |b: u32| -> u32 {
            let mut m = 0;
            for (j, &li) in nest.iter().enumerate() {
                if cfg.loops[li].body[b as usize] {
                    m = j as u32 + 1;
                }
            }
            m
        };
        let succ_version = |a: u32, k: u32, b: u32| -> u32 {
            for (j, &li) in nest.iter().enumerate() {
                let j = j as u32;
                if j < k && cfg.loops[li].header == b && cfg.loops[li].body[a as usize] {
                    return j;
                }
            }
            k.min(level(b))
        };
        let start = self.instance(entry_bc, n.min(level(entry_bc)));
        let mut work = vec![start];
        let mut seen = vec![start];
        while let Some(i) = work.pop() {
            let (bc, k) = (self.instances[i].bc, self.instances[i].version);
            let succs = if self.block_deopts(bc) {
                Vec::new()
            } else {
                cfg.blocks[bc as usize].succs.clone()
            };
            let mut out = Vec::new();
            for s in succs {
                let v = succ_version(bc, k, s);
                let si = self.instance(s, v);
                out.push(si);
                self.instances[si].npreds += 1;
                if !seen.contains(&si) {
                    seen.push(si);
                    work.push(si);
                }
            }
            self.instances[i].succs = out;
        }
        // RPO over instances.
        let mut visited = vec![false; self.instances.len()];
        let mut post = Vec::new();
        let mut stack = vec![(start, 0usize)];
        visited[start] = true;
        while let Some(&mut (i, ref mut k)) = stack.last_mut() {
            if *k < self.instances[i].succs.len() {
                let s = self.instances[i].succs[*k];
                *k += 1;
                if !visited[s] {
                    visited[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(i);
                stack.pop();
            }
        }
        post.reverse();
        for (j, &li) in nest.iter().enumerate() {
            if let Some(&ii) = self.inst_of.get(&(cfg.loops[li].header, j as u32)) {
                let b = self.instances[ii].ir;
                self.f.blocks[b.idx()].peeled = true;
            }
        }
        post
    }

    /// Whether the block is compiled as one deopt: it never ran (6.3), or
    /// an instruction in it always deopts, cutting its successors.
    fn block_deopts(&self, bc: u32) -> bool {
        if self.cfg.never_ran[bc as usize] {
            return true;
        }
        let blk = &self.cfg.blocks[bc as usize];
        (blk.start..blk.end).any(|pc| emit::always_deopts(self, pc))
    }

    // --- driver -------------------------------------------------------------

    pub(crate) fn build(mut self) -> Result<Func<'gc>, BuildError> {
        let entry_pc = self.f.meta.entry_pc;
        let entry_bc = self.cfg.block_of[entry_pc as usize];
        let prologue = self.new_root(vec![TypeSet::ANY; self.nregs]);
        self.f.entry = prologue;
        let order = self.discover(entry_bc);
        let first = self.instances[order[0]].ir;
        self.switch_to(prologue);
        self.jump(first);
        self.instances[order[0]].npreds += 1;
        self.instances[order[0]].done_preds += 1;
        for &ii in &order {
            let (bc, ir) = (self.instances[ii].bc, self.instances[ii].ir);
            let reached = self.instances[ii].done_preds > 0 || !self.bs[ir.idx()].preds.is_empty();
            self.instances[ii].started = true;
            if self.instances[ii].done_preds == self.instances[ii].npreds {
                self.seal(ir);
            }
            if reached {
                self.switch_to(ir);
                self.init_slots(ir);
                self.emit_instance(ii, bc);
            } else {
                self.f.blocks[ir.idx()].dead = true;
                self.bs[ir.idx()].terminated = true;
            }
            self.instances[ii].processed = true;
            let succs = self.instances[ii].succs.clone();
            for s in succs {
                self.instances[s].done_preds += 1;
                if self.instances[s].started
                    && self.instances[s].done_preds == self.instances[s].npreds
                {
                    let sb = self.instances[s].ir;
                    self.seal(sb);
                }
            }
            if self.f.insts.len() > 4000 {
                return Err(BuildError::TooLarge);
            }
        }
        // Any block left unsealed (its unreached predecessors) is sealed now.
        for b in 0..self.bs.len() {
            if !self.bs[b].sealed {
                self.seal(Block(b as u32));
            }
        }
        // Blocks no edge reaches.
        let reach = self.f.rpo();
        let mut live = vec![false; self.f.blocks.len()];
        for b in reach {
            live[b.idx()] = true;
        }
        for (b, l) in live.iter().enumerate() {
            if !l {
                self.f.blocks[b].dead = true;
            }
        }
        remove_trivial_params(&mut self.f);
        Ok(self.f)
    }

    /// The slot state at the start of a block: the meet over its emitted
    /// predecessors, unknown at a loop header.
    fn init_slots(&mut self, b: Block) {
        let preds = self.bs[b.idx()].preds.clone();
        if !self.bs[b.idx()].sealed || preds.is_empty() {
            if !self.bs[b.idx()].root {
                self.bs[b.idx()].slots = vec![SlotState::Unknown; self.nregs];
            }
            return;
        }
        let mut st = self.bs[preds[0].idx()].slots.clone();
        for p in &preds[1..] {
            for (r, s) in st.iter_mut().enumerate() {
                if *s != self.bs[p.idx()].slots[r] {
                    *s = SlotState::Unknown;
                }
            }
        }
        // A value that differs between predecessors became a parameter, and
        // `Holds` names the old one: only keep states whose value is still
        // the register's here.
        self.bs[b.idx()].slots = st;
    }

    fn emit_instance(&mut self, ii: usize, bc: u32) {
        let blk = &self.cfg.blocks[bc as usize];
        let (start, end) = (blk.start, blk.end);
        self.cur_snap = None;
        if self.cfg.never_ran[bc as usize] {
            self.pc = start;
            self.deopt(ExitTag::NeverRan);
            return;
        }
        if self.cfg.loops.iter().any(|l| l.header == bc) {
            // The collector's chance once per iteration; `prune_gc_checks`
            // keeps it only in loops that allocate or call.
            self.pc = start;
            let s = self.snap_before();
            self.ins_snap(Op::GcCheck, &[], s, ExitTag::Gc);
        }
        for pc in start..end {
            self.pc = pc;
            self.cur_snap = None;
            let last = pc + 1 == end;
            let succs: Vec<Block> = if last {
                self.instances[ii]
                    .succs
                    .iter()
                    .map(|&s| self.instances[s].ir)
                    .collect()
            } else {
                Vec::new()
            };
            emit::emit(self, pc, &succs);
            self.flush();
            if self.terminated() {
                return;
            }
            if last {
                // Fallthrough into the single successor.
                match succs.as_slice() {
                    [s] => self.jump(*s),
                    [] => self.deopt(ExitTag::Unsupported),
                    _ => unreachable!("a branch instruction did not terminate its block"),
                }
            }
        }
    }

    pub(crate) fn code(&self, pc: u32) -> Instruction {
        self.cfg.code[pc as usize]
    }

    pub(crate) fn cur_op(&self) -> BcOp {
        self.code(self.pc).op()
    }

    pub(crate) fn feedback(&self, pc: u32) -> u8 {
        self.proto.feedback[pc as usize].get()
    }
}

/// The type set of a constant value.
pub(crate) fn value_set(v: Value<'_>) -> TypeSet {
    if v.is_nil() {
        TypeSet::NIL
    } else if let Some(b) = v.get_boolean() {
        if b { TypeSet::TRUE } else { TypeSet::FALSE }
    } else if v.get_small().is_some() {
        TypeSet::SMALL
    } else if v.get_integer().is_some() {
        TypeSet::BIGINT
    } else if v.is_float() {
        TypeSet::FLOAT
    } else if v.get_string().is_some() {
        TypeSet::STR
    } else if v.get_table().is_some() {
        TypeSet::TAB
    } else if v.get_function().is_some() {
        TypeSet::FUN
    } else if v.get_thread().is_some() {
        TypeSet::THR
    } else {
        TypeSet::UDATA
    }
}

/// Remove parameters whose incoming arguments are all one other value (or
/// the parameter itself), to a fixpoint.
pub(crate) fn remove_trivial_params(f: &mut Func<'_>) {
    loop {
        let preds_edges = incoming(f);
        let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
        let mut changed = false;
        for b in 0..f.blocks.len() {
            if f.blocks[b].dead {
                continue;
            }
            let params = f.blocks[b].params.clone();
            for (k, &p) in params.iter().enumerate() {
                let mut same: Option<Val> = None;
                let mut trivial = true;
                for &(term, e) in &preds_edges[b] {
                    let a = f.edges(term)[e].args[k];
                    let a = resolve(&map, a);
                    if a == p {
                        continue;
                    }
                    match same {
                        None => same = Some(a),
                        Some(s) if s == a => {}
                        Some(_) => {
                            trivial = false;
                            break;
                        }
                    }
                }
                if trivial && let Some(s) = same {
                    map[p.idx()] = s;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        // A replacement of another representation enters through a
        // conversion at the parameter's block.
        for b in 0..f.blocks.len() {
            let params = f.blocks[b].params.clone();
            let mut at = 0;
            for &p in &params {
                let s = resolve(&map, p);
                if s == p || f.vals[s.idx()].ty.rep == f.vals[p.idx()].ty.rep {
                    continue;
                }
                let rep = f.vals[p.idx()].ty.rep;
                let op = conversion(f.vals[s.idx()].ty.rep, rep);
                let i = f.make_inst(op, &[s], None, ExitTag::Type);
                f.insts[i.idx()].block = Block(b as u32);
                f.blocks[b].insts.insert(at, i);
                at += 1;
                let r = f.result(i);
                f.vals[r.idx()].ty = Ty {
                    rep,
                    ..f.vals[p.idx()].ty
                };
                while map.len() < f.vals.len() {
                    map.push(Val(map.len() as u32));
                }
                map[p.idx()] = r;
            }
        }
        f.apply_replacements(&mut map);
        // Drop the replaced parameters and their arguments.
        for b in 0..f.blocks.len() {
            let params = f.blocks[b].params.clone();
            let keep: Vec<bool> = params.iter().map(|&p| map[p.idx()] == p).collect();
            if keep.iter().all(|&k| k) {
                continue;
            }
            f.blocks[b].params = params
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p)
                .collect();
            for (k, &p) in f.blocks[b].params.clone().iter().enumerate() {
                f.vals[p.idx()].def = ValDef::Param(Block(b as u32), k as u16);
            }
            for &(term, e) in &preds_edges[b] {
                let args = std::mem::take(&mut f.edges_mut(term)[e].args);
                f.edges_mut(term)[e].args = args
                    .into_iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(a, _)| a)
                    .collect();
            }
        }
    }
}

/// The operation converting `from` to `to`.
pub(crate) fn conversion(from: Rep, to: Rep) -> Op {
    match (from, to) {
        (_, Rep::Val) => Op::Box,
        (Rep::I32, Rep::I64) => Op::IToL,
        (Rep::I32, Rep::F64) => Op::IToF,
        (Rep::I64, Rep::I32) => Op::LToI,
        (_, rep) => Op::Unbox(rep),
    }
}

fn resolve(map: &[Val], mut v: Val) -> Val {
    while map[v.idx()] != v {
        v = map[v.idx()];
    }
    v
}

/// For each block, the (terminator, edge index) pairs entering it.
pub(crate) fn incoming(f: &Func<'_>) -> Vec<Vec<(Inst, usize)>> {
    let mut inc = vec![Vec::new(); f.blocks.len()];
    for bd in &f.blocks {
        if bd.dead {
            continue;
        }
        if let Some(&t) = bd.insts.last()
            && f.op(t).is_terminator()
        {
            for (k, e) in f.edges(t).iter().enumerate() {
                inc[e.target.idx()].push((t, k));
            }
        }
    }
    inc
}

impl Func<'_> {
    pub(crate) fn op_of_val(&self, v: Val) -> Option<Op> {
        self.def_op(v)
    }

    pub(crate) fn def_block(&self, v: Val) -> Option<Block> {
        match self.vals[v.idx()].def {
            ValDef::Inst(i, _) => Some(self.insts[i.idx()].block),
            ValDef::Param(b, _) => Some(b),
        }
    }
}
