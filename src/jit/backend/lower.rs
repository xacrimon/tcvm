//! IR to a target's `MInst`s, the part every target shares: one VCode block per
//! IR block in layout order, the vreg of each value (refining guards and
//! unboxes that change no bits share their operand's), snapshot values attached
//! to exiting instructions as `Any` uses and to calls as stack uses, and the
//! call protocol's choices (known callees, results in the result register,
//! floats kept in float registers).
//!
//! The target selects the instructions: it implements `Select` for `Lower`
//! (`target::lower`), with `Isel` its own state.

use crate::env::value::Value;
use crate::jit::FastMap;
use crate::jit::backend::order::{Layout, layout};
use crate::jit::backend::reg::{
    Block as RBlock, Operand, OperandConstraint, OperandKind, OperandPos, PRegSet, RegClass, VReg,
};
use crate::jit::backend::target::abi::allocatable;
use crate::jit::backend::target::inst::MInst;
use crate::jit::backend::target::lower::Isel;
use crate::jit::backend::vcode::VCode;
use crate::jit::build::cfg::RegSet;
use crate::jit::ir::ops::{ExitTag, Mem, NO_PROTO, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{
    Block, CfgInfo, ExitKind, ExitSite, Func, Inst, KEPT, NO_FRAME, NO_SNAP, Val,
};
use crate::jit::region::{SRep, SnapFrame};

/// An exit of the region before allocation: its snapshot's entries, those
/// with a value in a register as operands of the exiting instruction.
pub(crate) struct LExit {
    pub(crate) pc: u32,
    /// The inlined frames the exit rebuilds, outermost first.
    pub(crate) frames: Vec<SnapFrame>,
    /// The pc whose feedback the exit records, when not `pc`.
    pub(crate) site: Option<ExitSite>,
    /// The snapshot's inlined frame, or `NO_FRAME`.
    pub(crate) frame: u32,
    pub(crate) kind: ExitKind,
    pub(crate) tag: ExitTag,
    /// `(register, rep, source)`.
    pub(crate) entries: Vec<(u8, SRep, Src)>,
    /// A moved exit's blame entries, by the blamed instruction's registers.
    pub(crate) blame: Vec<(u8, SRep, Src)>,
    /// The VCode instruction that exits, once lowered.
    pub(crate) inst: Option<usize>,
    /// Whether `TCVM_JIT_DEOPT_ALL` may take it where it is first reached:
    /// not a test that a helper that writes failed (and did nothing), as
    /// the exit's frame would do the helper's work again.
    #[cfg(feature = "tcvm-dev")]
    pub(crate) forced: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Src {
    /// The `k`th snapshot operand of the exiting instruction.
    Operand(u16),
    /// A raw word.
    Const(u64),
    /// At an exit: the frame slot of this register (a home load's).
    Slot(u8),
}

/// A call's site: the values its snapshot names, which live in the frame's
/// spill slots across it, as operands of the call, and the registers whose home
/// slots hold theirs (`Func::call_homes`).
pub(crate) struct LSite {
    /// `(register, rep, source)`.
    pub(crate) entries: Vec<(u8, SRep, Src)>,
    pub(crate) homes: RegSet,
    /// The inlined frames the call is made from, outermost first, which the
    /// frame walk shows.
    pub(crate) frames: Vec<SnapFrame>,
}

/// A call's callee, known at compile time.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Known {
    /// A closure without varargs of the prototype at `proto` (its box), which
    /// takes `nparams`: `closure`, a constant the call writes word 0 for itself
    /// (the closure word on the fast path, the value on the slow one), or one a
    /// guard proved to be of the prototype.
    Lua {
        closure: Option<u64>,
        proto: u64,
        nparams: u8,
    },
    /// A native: its closure, which `jit_call_native` runs.
    Native(u64),
}

/// What a trampoline's call hands back: an integer or a float result, or
/// nothing from a call whose first `args` integer argument registers the
/// caller set up (and keeps itself). It keeps every other register.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tramp {
    Int,
    Float,
    Args(u8),
}

pub(crate) struct Lowered {
    pub(crate) vcode: VCode<MInst>,
    pub(crate) exits: Vec<LExit>,
    /// The calls' sites, by their resume block's VCode block.
    pub(crate) sites: Vec<(usize, LSite)>,
    /// The first operand index of each instruction's snapshot uses.
    pub(crate) snap_ops: Vec<u32>,
    /// The registers the entry guards test.
    pub(crate) entry_regs: Vec<u8>,
    /// `RegionMeta::params`: an entry guard's failure stores them, as only
    /// a caller using slots wrote them there.
    pub(crate) params: u8,
    /// `Func::shape_sets` as the shapes' addresses.
    pub(crate) shape_sets: Vec<Vec<u64>>,
    /// The region has dependencies: its entry is patchable.
    pub(crate) invalidable: bool,
    /// The VCode blocks in the order the allocator and the encoder take them.
    pub(crate) emit_order: Vec<u32>,
    /// How many of `emit_order` are hot: the rest end in a deopt or are
    /// cold.
    pub(crate) hot_blocks: usize,
    /// Each loop's first and last VCode block (its blocks lie between, the
    /// header first), innermost loops first.
    pub(crate) loops: Vec<(u32, u32)>,
    /// The IR block of each VCode block (the asm dump's names).
    #[cfg(feature = "tcvm-dev")]
    pub(crate) ir_blocks: Vec<u32>,
    /// The entry guards, all on parameters, as tests the fast entry makes
    /// before the entry block (the parameter and the set), which self-calls
    /// passing them enter; none in the block's code.
    pub(crate) prefix_guards: bool,
    pub(crate) entry_tests: Vec<(u8, TypeSet)>,
    /// `Func::window`, when past the frame's own window: every entry makes sure
    /// of it, once, before the guards.
    pub(crate) window: Option<u16>,
    /// The helpers the code calls through trampolines, by address.
    pub(crate) trampolines: Vec<(usize, Tramp)>,
}

pub(crate) struct Lower<'a, 'gc> {
    pub(crate) f: &'a Func<'gc>,
    pub(crate) v: VCode<MInst>,
    pub(crate) vreg: Vec<Option<VReg>>,
    /// Values computed by their single user instead of at their definition.
    pub(crate) fused: Vec<bool>,
    /// Values lowering keeps in a float register as their float's word
    /// (`find_float_words`).
    pub(crate) in_d: Vec<bool>,
    pub(crate) uses: Vec<u32>,
    exits: Vec<LExit>,
    sites: Vec<(usize, LSite)>,
    snap_ops: Vec<u32>,
    pub(crate) entry_regs: Vec<u8>,
    block_of: Vec<u32>,
    pub(crate) pc_base: usize,
    /// Instructions another computes with its own: a store a known call writes
    /// itself and its callee, a result's store to its slot, and what the
    /// target's patterns take in.
    pub(crate) absorbed: Vec<bool>,
    pub(crate) known: FastMap<Inst, Known>,
    /// The frame's closure, defined in the entry block when the region
    /// reads its upvalues.
    pub(crate) closure: Option<VReg>,
    /// The parameters entries pass in registers, by index (`Params`).
    pub(crate) params: Vec<VReg>,
    /// The region's prototype (its box) when a compiled caller can enter
    /// it, and the set each parameter's entry guard admits.
    pub(crate) own: Option<u64>,
    pub(crate) entry_sets: Vec<TypeSet>,
    pub(crate) prefix_guards: bool,
    pub(crate) entry_tests: Vec<(u8, TypeSet)>,
    trampolines: Vec<(usize, Tramp)>,
    /// The target's own selection state.
    pub(crate) isel: Isel,
}

/// A target's instruction selection, over the shared context: what the
/// driver (`lower`) asks of it. It emits with `push`, `push_exit` and
/// `push_call`.
pub(crate) trait Select {
    /// Mark what the target's instructions compute together, before any
    /// is lowered: values their single user computes (`fused`) and
    /// instructions another takes in (`absorbed`).
    fn find_patterns(&mut self, order: &[Block]);
    /// The entry block's head: the native frame, then the parameters
    /// entries pass in registers (`params`, and each one's `vreg`).
    fn entry(&mut self);
    /// Load the frame's closure for the upvalue reads, into `closure`.
    fn frame_closure(&mut self);
    /// Lower IR instruction `i`, which the driver did not skip.
    fn lower_inst(&mut self, i: Inst) -> Result<(), String>;
}

pub(crate) fn rep_class(rep: Rep) -> RegClass {
    if rep == Rep::F64 {
        RegClass::Float
    } else {
        RegClass::Int
    }
}

fn srep(rep: Rep) -> SRep {
    match rep {
        Rep::Val | Rep::Ptr => SRep::Val,
        Rep::I32 => SRep::I32,
        Rep::I64 => SRep::I64,
        Rep::F64 => SRep::F64,
        Rep::B1 => SRep::B1,
    }
}

/// The address of `Code[pc]` of the prototype being compiled; `frame` its
/// frame's window, which every entry has.
pub(crate) fn lower<'gc>(
    f: &Func<'gc>,
    code_base: usize,
    own: Option<u64>,
    frame: u16,
) -> Result<Lowered, String> {
    let cfg = f.cfg();
    let Layout {
        order,
        emit: emit_order,
        hot: hot_blocks,
        loops,
        loop_of,
        loop_parent,
    } = layout(f);
    let order = &order;
    let mut block_of = vec![u32::MAX; f.blocks.len()];
    for (k, b) in order.iter().enumerate() {
        block_of[b.idx()] = k as u32;
    }
    let uses = f.use_counts();
    let mut lw = Lower {
        f,
        v: VCode::new(),
        vreg: vec![None; f.vals.len()],
        fused: vec![false; f.vals.len()],
        in_d: vec![false; f.vals.len()],
        uses,
        exits: Vec::new(),
        sites: Vec::new(),
        snap_ops: Vec::new(),
        entry_regs: Vec::new(),
        block_of,
        pc_base: code_base,
        absorbed: vec![false; f.insts.len()],
        known: Default::default(),
        closure: None,
        params: Vec::new(),
        own,
        entry_sets: vec![TypeSet::ANY; f.meta.params as usize],
        prefix_guards: false,
        entry_tests: Vec::new(),
        trampolines: Vec::new(),
        isel: Isel::default(),
    };
    // The entry's parameters and their guards lead its block (speculation
    // puts them there); in a region a compiled caller enters, the fast
    // entry tests them before the block, unless an entry guard comes later
    // or tests something else.
    let entry_insts = f.insts_of(f.entry);
    let entry_guard = |i: Inst| {
        matches!(f.op(i), Op::Guard(_))
            && f.insts[i.idx()].tag == ExitTag::Entry
            && matches!(f.def_op(f.args(i)[0]), Some(Op::Param(_)))
    };
    let lead = (entry_insts.iter())
        .position(|&i| !matches!(f.op(i), Op::Param(_)) && !entry_guard(i))
        .unwrap_or(entry_insts.len());
    lw.prefix_guards = own.is_some()
        && !(f.insts.iter().enumerate()).any(|(k, d)| {
            d.tag == ExitTag::Entry
                && matches!(d.op, Op::Guard(_))
                && !entry_insts[..lead].contains(&Inst(k as u32))
        });
    for &i in &entry_insts[..lead] {
        if let Op::Guard(set) = f.op(i)
            && let Some(Op::Param(r)) = f.def_op(f.args(i)[0])
        {
            lw.entry_sets[r as usize] &= set;
        }
    }
    let closure_at = closure_block(f, &cfg, order);
    lw.find_patterns(order);
    lw.find_known_callees(order);
    lw.find_register_returns(order);
    lw.find_float_words(order);
    let mut params: Vec<VReg> = Vec::new();
    let mut succs: Vec<RBlock> = Vec::new();
    let mut args: Vec<VReg> = Vec::new();
    let mut nargs: Vec<u32> = Vec::new();
    for &b in order {
        let start = lw.v.insts.len() as u32;
        params.clear();
        for &p in f.params(b) {
            params.push(lw.vr(p));
        }
        if b == f.entry {
            lw.entry();
        }
        // A resume block's first instruction is where a fast callee returns.
        let mut closure_due = closure_at == Some(b);
        if closure_due && !f.blocks[b.idx()].resume {
            lw.frame_closure();
            closure_due = false;
        }
        succs.clear();
        args.clear();
        nargs.clear();
        for &i in f.insts_of(b) {
            if !lw.skipped(i) {
                lw.lower_inst(i)?;
            }
            if std::mem::take(&mut closure_due) {
                lw.frame_closure();
            }
            if f.op(i).is_terminator() {
                // A call's resume block takes its values from the return.
                let call = matches!(f.op(i), Op::Call { .. } | Op::CallMm { .. });
                for e in f.edges(i) {
                    succs.push(RBlock::new(lw.block_of[e.target.idx()] as usize));
                    let a = if call { &[][..] } else { f.vl(e.args) };
                    for &x in a {
                        args.push(lw.vr(x));
                    }
                    nargs.push(a.len() as u32);
                }
            }
        }
        let end = lw.v.insts.len() as u32;
        let preds = cfg
            .preds(b)
            .iter()
            .map(|p| RBlock::new(lw.block_of[p.idx()] as usize));
        lw.v.push_block(
            start,
            end,
            &succs,
            preds,
            params.iter().copied(),
            &args,
            &nargs,
            f.blocks[b.idx()].resume,
        );
    }
    lw.v.entry = RBlock::new(0);
    lw.v.loop_of = loop_of;
    lw.v.loop_parent = loop_parent;
    Ok(Lowered {
        vcode: lw.v,
        exits: lw.exits,
        sites: lw.sites,
        snap_ops: lw.snap_ops,
        entry_regs: lw.entry_regs,
        params: f.meta.params,
        shape_sets: f
            .shape_sets
            .iter()
            .map(|set| {
                set.iter()
                    .map(|&k| f.shapes[k as usize].as_word())
                    .collect()
            })
            .collect(),
        invalidable: !f.deps.is_empty(),
        emit_order,
        hot_blocks,
        loops,
        #[cfg(feature = "tcvm-dev")]
        ir_blocks: order.iter().map(|b| b.0).collect(),
        prefix_guards: lw.prefix_guards,
        entry_tests: lw.entry_tests,
        window: (f.window > frame).then_some(f.window),
        trampolines: lw.trampolines,
    })
}

/// Where the frame's closure is loaded for the upvalue reads: the nearest block
/// dominating them all, or the entry when that lies in a loop.
fn closure_block(f: &Func<'_>, cfg: &CfgInfo, order: &[Block]) -> Option<Block> {
    let mut at: Option<Block> = None;
    for &b in order {
        let reads = f
            .insts_of(b)
            .iter()
            .any(|&i| matches!(f.op(i), Op::UpvalValue(_) | Op::UpvalCell(_) | Op::FrameFn));
        if !reads {
            continue;
        }
        let mut a = at.unwrap_or(b);
        while !cfg.dominates(a, b) {
            a = cfg.idom[a.idx()].expect("a dominator");
        }
        at = Some(a);
    }
    let at = at?;
    let in_loop = order.iter().any(|&p| {
        f.succs(p)
            .any(|h| cfg.dominates(h, p) && cfg.dominates(h, at))
    });
    Some(if in_loop { f.entry } else { at })
}

impl<'a, 'gc> Lower<'a, 'gc> {
    /// A stored value's operand: a float box stores its float's register,
    /// whether or not other uses compute the box.
    pub(crate) fn stored(&mut self, v: Val) -> Operand {
        let f = self.f;
        match f.def_inst(v) {
            Some(i) if f.op(i) == Op::Box && f.ty(f.args(i)[0]).rep == Rep::F64 => {
                self.use_(f.args(i)[0])
            }
            _ => self.use_(v),
        }
    }

    /// The vreg of a value, through aliases.
    pub(crate) fn vr(&mut self, v: Val) -> VReg {
        let v = self.alias(v);
        if let Some(r) = self.vreg[v.idx()] {
            return r;
        }
        let r = self.v.new_vreg(self.class(v));
        self.vreg[v.idx()] = Some(r);
        r
    }

    /// The register class of a value `alias` returns.
    fn class(&self, v: Val) -> RegClass {
        if self.in_d[v.idx()] {
            RegClass::Float
        } else {
            rep_class(self.f.ty(v).rep)
        }
    }

    /// The rep of `v` as its register holds it: `F64` for a word in a float
    /// register.
    fn reg_rep(&self, v: Val) -> Rep {
        match self.class(self.alias(v)) {
            RegClass::Float => Rep::F64,
            _ => self.f.ty(v).rep,
        }
    }

    /// The value whose register `v` shares: refining guards and i32
    /// unboxes/narrowings change no bits, nor do f64 boxes and unboxes of a
    /// word in a float register (`in_d`).
    pub(crate) fn alias(&self, mut v: Val) -> Val {
        while let Some(i) = self.f.def_inst(v) {
            match self.f.op(i) {
                Op::Guard(_) | Op::Refine(_) => v = self.f.args(i)[0],
                Op::Unbox(Rep::I32) => v = self.f.args(i)[0],
                Op::LToI | Op::LTrunc => v = self.f.args(i)[0],
                Op::Box if self.in_d[v.idx()] => v = self.f.args(i)[0],
                Op::Unbox(Rep::F64) => {
                    let x = self.alias(self.f.args(i)[0]);
                    if self.class(x) != RegClass::Float {
                        break;
                    }
                    v = x;
                }
                _ => break,
            }
        }
        v
    }

    /// The slot of the home load `v` shares a register with.
    fn home_load(&self, v: Val) -> Option<u8> {
        match self.f.def_op(self.alias(v)) {
            Some(Op::Home(r)) => Some(r),
            _ => None,
        }
    }

    pub(crate) fn push(&mut self, inst: MInst, ops: &[Operand], clob: PRegSet) -> usize {
        let i = self.v.push(inst, ops, clob);
        self.snap_ops.push(ops.len() as u32);
        i.index()
    }

    /// Calls of a constant Lua closure without varargs or native, whose store
    /// to the call's slot nothing between reads, and of a Lua closure of a
    /// prototype a guard proved.
    fn find_known_callees(&mut self, order: &[Block]) {
        for &b in order {
            let Some(t) = self.f.terminator(b) else {
                continue;
            };
            let Op::Call { a, proto, .. } = self.f.op(t) else {
                continue;
            };
            if proto != NO_PROTO {
                let p = self.f.protos[proto as usize];
                self.known.insert(
                    t,
                    Known::Lua {
                        closure: None,
                        proto: p.ptr.as_ptr() as u64,
                        nparams: p.num_params,
                    },
                );
            }
            for &j in self.f.insts_of(b).iter().rev().skip(1) {
                let op = self.f.op(j);
                match op {
                    Op::Store(r) if r == a => {
                        let Some(Op::KObj(p)) = self.f.def_op(self.f.args(j)[0]) else {
                            break;
                        };
                        use crate::env::function::FunctionKind;
                        let v = self.f.pool[p as usize];
                        let known = match v.get_function().map(|f| f.inner().as_ref()) {
                            // The prototype word the fast path loads.
                            Some(FunctionKind::Lua(c)) if !c.is_vararg => Some(Known::Lua {
                                closure: Some(v.to_raw()),
                                proto: c.proto.ptr.as_ptr() as u64,
                                nparams: c.num_params,
                            }),
                            Some(FunctionKind::Native(nc)) => {
                                Some(Known::Native(nc as *const _ as u64))
                            }
                            _ => None,
                        };
                        if let Some(k) = known {
                            // The constant is then made for nothing else.
                            self.absorbed[j.idx()] = true;
                            self.uses[self.f.args(j)[0].idx()] -= 1;
                            self.known.insert(t, k);
                        }
                        break;
                    }
                    Op::Store(_) => {}
                    op if op.effects().reads.intersects(Mem::SLOT)
                        || op.effects().writes.intersects(Mem::SLOT) =>
                    {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    /// A return of one result passes it in the result register alone: its store
    /// to the result slot goes, when nothing between them can exit or read the
    /// slot. A caller reads it from the register, `level_land` stores it for
    /// the interpreter.
    fn find_register_returns(&mut self, order: &[Block]) {
        for &b in order {
            let Some(t) = self.f.terminator(b) else {
                continue;
            };
            let Op::Return { a, n: 1 } = self.f.op(t) else {
                continue;
            };
            for &j in self.f.insts_of(b).iter().rev().skip(1) {
                if self.f.insts[j.idx()].snap != NO_SNAP {
                    break;
                }
                match self.f.op(j) {
                    Op::Store(r) if r == a => {
                        if self.result_in_reg(self.f.args(j)[0]) {
                            self.absorbed[j.idx()] = true;
                        }
                        break;
                    }
                    op if op.effects().reads.intersects(Mem::SLOT)
                        || op.effects().writes.intersects(Mem::SLOT) =>
                    {
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    /// The value return `i` passes in the result register: what its block stored to `R[a]`,
    /// if a register has it.
    pub(crate) fn returned(&self, i: Inst) -> Option<Val> {
        let f = self.f;
        let Op::Return { a, n } = f.op(i) else {
            return None;
        };
        let b = f.insts[i.idx()].block;
        let mut stored = None;
        for &j in f.insts_of(b).iter().rev().skip(1) {
            match f.op(j) {
                Op::Store(r) if r == a => {
                    stored = Some(f.args(j)[0]);
                    break;
                }
                op if op.effects().writes.contains(Mem::SLOT) => break,
                _ => {}
            }
        }
        stored.filter(|&v| n > 0 && self.result_in_reg(v))
    }

    /// The values whose float's word lowering keeps in a float register
    /// (`in_d`): f64 boxes and loads of floats that only stores, snapshots
    /// and f64 unboxes use. Such a load loads into the float register, its
    /// unboxes and a box's share the register (`alias`), and stores store
    /// it, all where the IR has them (`opt::snaps` follows reads through
    /// boxes and unboxes).
    fn find_float_words(&mut self, order: &[Block]) {
        let f = self.f;
        // Per alias root: the uses that need the word in an integer register,
        // and whether an f64 unbox reads it.
        let mut gpr = vec![0u32; f.vals.len()];
        let mut unboxed = vec![false; f.vals.len()];
        for &b in order {
            for &i in f.insts_of(b) {
                let op = f.op(i);
                if self.absorbed[i.idx()]
                    || (f.insts[i.idx()].rn == 1
                        && op.is_pure()
                        && self.uses[f.result(i).idx()] == 0
                        && !self.fused[f.result(i).idx()])
                {
                    continue;
                }
                let args = f.args(i);
                // The operands only an integer register serves; the others are
                // stored values, and snapshot entries take either too.
                let xs = match op {
                    Op::Unbox(Rep::F64) => {
                        unboxed[self.alias(args[0]).idx()] = true;
                        &[][..]
                    }
                    Op::Refine(_) | Op::Store(_) => &[],
                    Op::TabStore { .. } | Op::CellStore => &args[..1],
                    Op::ArrStore => &args[..2],
                    Op::Call { nargs, .. }
                        if nargs != u8::MAX
                            && !matches!(self.known.get(&i), Some(Known::Lua { .. })) =>
                    {
                        &[]
                    }
                    _ => args,
                };
                for &a in xs {
                    gpr[self.alias(a).idx()] += 1;
                }
                if let Some(v) = self.returned(i) {
                    gpr[self.alias(v).idx()] += 1;
                }
                for e in f.edges(i) {
                    for &a in f.vl(e.args) {
                        gpr[self.alias(a).idx()] += 1;
                    }
                }
            }
        }
        for (k, d) in f.vals.iter().enumerate() {
            let Some(i) = f.def_inst(Val(k as u32)) else {
                continue;
            };
            self.in_d[k] = gpr[k] == 0
                && match f.op(i) {
                    Op::Box => f.ty(f.args(i)[0]).rep == Rep::F64,
                    Op::Load(_)
                    | Op::Home(_)
                    | Op::TabLoad { .. }
                    | Op::ArrLoad
                    | Op::CellLoad
                    | Op::UpvalValue(_)
                    | Op::UpvalValueOf(_) => unboxed[k] && d.ty.set == TypeSet::FLOAT,
                    _ => false,
                };
        }
    }

    /// Whether a value stored as a result can be returned in the result
    /// register: a boxed value (a float's box is then computed there, not
    /// stored and loaded).
    fn result_in_reg(&self, v: Val) -> bool {
        self.f.ty(v).rep == Rep::Val
    }

    pub(crate) fn konst_int(&self, v: Val) -> Option<i64> {
        match self.f.def_op(v)? {
            Op::KI32(n) => Some(n as i64),
            Op::KI64(n) => Some(n),
            _ => None,
        }
    }

    pub(crate) fn use_(&mut self, v: Val) -> Operand {
        Operand::reg_use(self.vr(v))
    }

    pub(crate) fn def(&mut self, v: Val) -> Operand {
        Operand::reg_def(self.vr(v))
    }

    /// Snapshot operands and the exit record of an exiting IR instruction.
    fn exit_of(&mut self, i: Inst) -> (u32, Vec<Operand>) {
        let d = &self.f.insts[i.idx()];
        debug_assert!(d.snap != NO_SNAP, "{:?} exits without a snapshot", d.op);
        let s = self.f.snaps[d.snap as usize];
        let f = self.f;
        let mut ops: Vec<Operand> = Vec::new();
        let mut entries = Vec::new();
        for &(r, v) in f.entries(d.snap) {
            // A call's kept value, in a snapshot an exit shares.
            if r == KEPT {
                continue;
            }
            let (rep, src) = self.exit_src(v, &mut ops);
            // A home load's slot holds its own register already.
            if src != Src::Slot(r) {
                entries.push((r, rep, src));
            }
        }
        let mut blame = Vec::new();
        for &(r, v) in f.blame_entries(i) {
            let (rep, src) = self.exit_src(v, &mut ops);
            blame.push((r, rep, src));
        }
        let pc = s.pc;
        let kind = s.kind;
        let tag = d.tag;
        let frames = self.snap_frames(s.frame);
        #[cfg(feature = "tcvm-dev")]
        let store_failed = d.op == Op::GuardFalse
            && f.def_inst(f.args(i)[0]).is_some_and(|c| {
                f.op(c) == Op::SameBits
                    && f.args(c).iter().any(
                        |&a| matches!(f.def_op(a), Some(Op::Helper(h)) if h.sig().effects.writes()),
                    )
            });
        self.exits.push(LExit {
            pc,
            frames,
            site: (d.blame != NO_SNAP).then(|| self.f.exit_site(i)),
            frame: s.frame,
            kind,
            tag,
            entries,
            blame,
            inst: None,
            #[cfg(feature = "tcvm-dev")]
            forced: !store_failed,
        });
        (self.exits.len() as u32 - 1, ops)
    }

    /// The value a snapshot entry of `v` keeps: boxes are the exit's work.
    fn snap_value(&self, v: Val) -> Val {
        match self.f.def_op(v) {
            Some(Op::Box) => self.f.args(self.f.def_inst(v).unwrap())[0],
            _ => v,
        }
    }

    /// The index of `vr` among the snapshot operands `ops`, added with
    /// `constraint` where new. Late: the instruction may write its result
    /// or call before it exits, and the value must survive that.
    fn snap_operand(ops: &mut Vec<Operand>, vr: VReg, constraint: OperandConstraint) -> u16 {
        let k = ops.iter().position(|o| o.vreg() == vr).unwrap_or_else(|| {
            ops.push(Operand::new(
                vr,
                constraint,
                OperandKind::Use,
                OperandPos::Late,
            ));
            ops.len() - 1
        });
        k as u16
    }

    /// Where an exit finds `v`: a constant, a home load's slot, or `v`'s
    /// register as one of the exit's snapshot operands `ops`.
    fn exit_src(&mut self, v: Val, ops: &mut Vec<Operand>) -> (SRep, Src) {
        let v = self.snap_value(v);
        if let Some(k) = self.const_word(v) {
            return (SRep::Val, Src::Const(k));
        }
        let rep = srep(self.reg_rep(v));
        if let Some(s) = self.home_load(v) {
            return (rep, Src::Slot(s));
        }
        let vr = self.vr(v);
        let k = Self::snap_operand(ops, vr, OperandConstraint::Any);
        (rep, Src::Operand(k))
    }

    /// The boxed word of a constant value.
    fn const_word(&self, v: Val) -> Option<u64> {
        match self.f.def_op(v)? {
            Op::KVal(b) => Some(b),
            // A pointer constant in a blame stands for the object.
            Op::KObj(p) | Op::KPtr(p) => Some(self.f.pool[p as usize].to_raw()),
            Op::KI32(n) => Some(Value::small(n).to_raw()),
            Op::KF64(b) => Some(Value::float(f64::from_bits(b)).to_raw()),
            Op::KB1(c) => Some(Value::boolean(c).to_raw()),
            Op::KI64(n) if i32::try_from(n).is_ok() => Some(Value::small(n as i32).to_raw()),
            _ => None,
        }
    }

    /// The inlined frames up to `frame`, outermost first.
    fn snap_frames(&self, mut frame: u32) -> Vec<SnapFrame> {
        let mut frames = Vec::new();
        while frame != NO_FRAME {
            let x = self.f.inline_frames[frame as usize];
            frames.push(SnapFrame {
                func: x.func,
                a: x.a,
                c: x.c,
                delta: if x.replaces { 0 } else { x.a as u16 + 4 },
                caller_pc: x.caller_pc,
                open: x.open,
            });
            frame = x.parent;
        }
        frames.reverse();
        frames
    }

    /// The address of the instruction past `pc` in the code of the frame
    /// call `i` is made from.
    pub(crate) fn pc_after(&self, i: Inst, pc: u32) -> usize {
        let frame = self.f.snaps[self.f.insts[i.idx()].snap as usize].frame;
        let code = match frame {
            NO_FRAME => self.pc_base,
            fr => self.f.inline_frames[fr as usize].code,
        };
        code + (pc as usize + 1) * 8
    }

    /// Push a call: `ops`, then its snapshot's values as uses in stack slots,
    /// which the collector traces while the call runs.
    pub(crate) fn push_call(&mut self, i: Inst, mk: MInst, mut ops: Vec<Operand>) {
        let snap = self.f.insts[i.idx()].snap;
        let frames = self.snap_frames(self.f.snaps[snap as usize].frame);
        let mut entries = Vec::new();
        let mut sops: Vec<Operand> = Vec::new();
        let base = ops.len();
        let homes = self.f.call_homes.get(&snap).copied().unwrap_or_default();
        for &(r, v) in self.f.entries(snap) {
            let v = self.snap_value(v);
            if let Some(k) = self.const_word(v) {
                entries.push((r, SRep::Val, Src::Const(k)));
                continue;
            }
            // The slot holds it, and the call keeps it there: a home entry.
            if let Some(s) = self.home_load(v) {
                debug_assert!(s == r && homes.contains(r as usize), "v{} spilled", v.0);
                continue;
            }
            let vr = self.vr(v);
            let k = Self::snap_operand(&mut sops, vr, OperandConstraint::Stack);
            entries.push((r, srep(self.reg_rep(v)), Src::Operand(k)));
        }
        ops.extend(sops);
        self.v.push(mk, &ops, allocatable());
        self.snap_ops.push(base as u32);
        let resume = self.f.edges(i)[0].target;
        self.sites.push((
            self.block_of[resume.idx()] as usize,
            LSite {
                entries,
                homes,
                frames,
            },
        ));
    }

    /// Push an exiting instruction: `ops` then the snapshot uses.
    pub(crate) fn push_exit(
        &mut self,
        i: Inst,
        mk: impl FnOnce(u32) -> MInst,
        mut ops: Vec<Operand>,
        clob: PRegSet,
    ) {
        let (exit, sops) = self.exit_of(i);
        let base = ops.len() as u32;
        ops.extend(sops);
        let at = self.v.push(mk(exit), &ops, clob).index();
        self.snap_ops.push(base);
        self.exits[exit as usize].inst = Some(at);
    }

    /// The `a` and the header slot of the call a resume block resumes,
    /// from its predecessor's terminator.
    pub(crate) fn resume_a(&self, i: Inst) -> (u8, u16) {
        let b = self.f.insts[i.idx()].block;
        for &p in self.f.cfg().preds(b) {
            if let Some(t) = self.f.terminator(p) {
                match self.f.op(t) {
                    Op::Call { a, .. } => return (a, a as u16),
                    Op::CallMm { land, hdr, .. } => return (land, hdr as u16),
                    _ => {}
                }
            }
        }
        (0, 0)
    }

    /// Whether `i` gets no instructions of its own: an unused pure result,
    /// a value its user computes (`fused`), or one another instruction took
    /// in (`absorbed`).
    fn skipped(&self, i: Inst) -> bool {
        let d = &self.f.insts[i.idx()];
        let unused = d.rn == 1 && {
            let r = self.f.result(i);
            self.fused[r.idx()] || (d.op.is_pure() && self.uses[r.idx()] == 0)
        };
        unused || self.absorbed[i.idx()]
    }

    /// The region calls the helper at `addr` through a trampoline of `kind`.
    pub(crate) fn needs_trampoline(&mut self, addr: usize, kind: Tramp) {
        if !self.trampolines.iter().any(|t| t.0 == addr) {
            self.trampolines.push((addr, kind));
        }
    }
}
