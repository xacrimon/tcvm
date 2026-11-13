//! The builder: one pass over the bytecode from the entry, Braun-style SSA over
//! the registers, home-slot tracking for the stores at tail-outs and the
//! entries of snapshots, and the outer-loop peeling of a loop entry inside a
//! nest.

pub(crate) mod cfg;
mod emit;
mod inline;
mod intrinsics;
mod iter;
mod tables;

use crate::dmm::Gc;
use crate::env::MetamethodBits;
use crate::env::function::{LuaFn, Prototype};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::cfg::{Cfg, RegSet};
use crate::jit::ir::ops::{ExitTag, HelperId, Op};
use crate::jit::ir::types::{Refine, Rep, Ty, TypeSet};
use crate::jit::ir::{
    Block, BlockCall, DepOn, ExitKind, Func, Inst, List, RegionMeta, Snap, Val, ValDef,
};

#[derive(Debug)]
pub(crate) enum BuildError {
    TooLarge,
}

/// Bytecode instructions the outer-loop peeling of a loop entry may copy, about
/// 400 IR instructions; past it the outermost levels stay merged into the
/// entry's loop.
const OSR_PEEL_LIMIT: u32 = 150;

/// A copy of a bytecode block in an outer-loop peeling version.
struct Instance {
    bc: u32,
    version: u32,
    ir: Block,
    /// A run of `Builder::inst_succs`.
    succs: List,
    npreds: u32,
    done_preds: u32,
    started: bool,
    discovered: bool,
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

/// The shapes a table may have (`Func::shapes` indices), at most
/// `MAX_SHAPES`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Shapes {
    n: u8,
    k: [u32; crate::jit::feedback::MAX_SHAPES],
}

impl Shapes {
    pub(crate) fn one(k: u32) -> Shapes {
        let mut s = Shapes {
            n: 1,
            k: [0; crate::jit::feedback::MAX_SHAPES],
        };
        s.k[0] = k;
        s
    }

    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.k[..self.n as usize]
    }

    pub(crate) fn contains(&self, k: u32) -> bool {
        self.as_slice().contains(&k)
    }

    /// These and `o`'s, when they fit.
    fn union(self, o: Shapes) -> Option<Shapes> {
        let mut s = self;
        for &k in o.as_slice() {
            if !s.contains(k) {
                *s.k.get_mut(s.n as usize)? = k;
                s.n += 1;
            }
        }
        Some(s)
    }

    /// The set of `ks`, when it fits.
    pub(crate) fn of(ks: &[u32]) -> Option<Shapes> {
        let mut s = Shapes {
            n: 0,
            k: [0; crate::jit::feedback::MAX_SHAPES],
        };
        for &k in ks {
            if !s.contains(k) {
                *s.k.get_mut(s.n as usize)? = k;
                s.n += 1;
            }
        }
        Some(s)
    }

    /// These, but those also in `ks`.
    fn only(self, ks: &[u32]) -> Shapes {
        let mut s = self;
        for &x in self.as_slice() {
            if !ks.contains(&x) {
                s = s.without(x);
            }
        }
        s
    }

    fn without(self, k: u32) -> Shapes {
        let mut s = Shapes {
            n: 0,
            k: [0; crate::jit::feedback::MAX_SHAPES],
        };
        for &x in self.as_slice().iter().filter(|&&x| x != k) {
            s.k[s.n as usize] = x;
            s.n += 1;
        }
        s
    }
}

/// What a branch or guard tells of a value on one of its edges.
#[derive(Clone, Copy, Debug)]
enum Fact {
    /// Its type set narrowed to this.
    Narrow(TypeSet),
    /// It is this constant.
    Is(Val),
    /// The table it unboxes to has one of these shapes.
    Shapes(Shapes),
}

/// Per IR block SSA state; its registers' definitions and slot states are
/// rows of `Builder::defs` and `Builder::slots`.
struct BlockState {
    sealed: bool,
    incomplete: Vec<(u8, Val)>,
    preds: Vec<Block>,
    /// A block values do not flow into (the prologue, after a helper that
    /// wrote slots): registers read there load their home slot.
    root: bool,
    /// Registers a call's resume block reads from their home slots: the
    /// call wrote them, a closure may have through a cell, or the slot held
    /// the value already. The others flow in from before the call.
    reload: RegSet,
    /// Where a root's loads are inserted.
    root_at: usize,
    /// Types of the slots of a root (what was stored before the call).
    root_ty: Vec<TypeSet>,
    /// The frame at a resume block's start, for guards of its loads.
    after: Option<Snap>,
    terminated: bool,
    /// Values this block refined, in order (`Builder::refine`): a read
    /// through it of a register holding the first gets the second.
    refines: Vec<(Val, Val)>,
    /// What the branch into it tells, bound once it is entered from that
    /// branch alone.
    on_entry: Vec<(Val, Fact)>,
    facts: Facts,
}

/// What the builder knows of values on a path. A block starts with the meet
/// over its predecessors'; an inlined body with its caller's at the call, the
/// code after the call with the meet over its returns'. Refinements are the
/// registers' (`BlockState::refines`): a body's of its arguments reach the
/// caller through the parameters it never writes.
#[derive(Clone, Default)]
struct Facts {
    /// The shapes tables have here, by fact key (`fact_key`): the shape guards
    /// and tests passed since an instruction that may change shapes, and
    /// whether a fact holds only while its shapes stay stable: one kept across
    /// such an instruction, or a load from a slot that tracks its tables'
    /// shape.
    shapes: Vec<(Val, Shapes, bool)>,
    /// The facts of stable shapes an instruction that may change shapes
    /// left, back at the lazy exit after it: no frame passes that exit
    /// once one of their shapes is left.
    stash: Vec<(Val, Shapes)>,
}

impl Facts {
    /// What holds where a path with `o` joins: a table's shapes are those
    /// it may have on either. A join follows no instruction's lazy exit, so
    /// nothing is stashed.
    fn meet(&mut self, o: &Facts) {
        self.shapes.retain_mut(|(v, s, stable)| {
            let Some(x) = o.shapes.iter().find(|x| x.0 == *v) else {
                return false;
            };
            match s.union(x.1) {
                Some(u) => {
                    (*s, *stable) = (u, *stable || x.2);
                    true
                }
                None => false,
            }
        });
        self.stash.clear();
    }
}

/// What holds of values wherever they are, shared by a region's builders with
/// its block states: an inlined body's builder takes them at the call and gives
/// them back.
#[derive(Default)]
pub(crate) struct Known<'gc> {
    /// The receiver and constant key of each direct field load, for values
    /// the builder can know now (intrinsic callees).
    pub(crate) field_loads: crate::jit::FastMap<Val, (Val, Value<'gc>)>,
    /// The shapes field sites compiled tests for, by receiver value.
    pub(crate) recv_shapes: crate::jit::FastMap<Val, Vec<crate::env::Shape<'gc>>>,
    /// `fact_key` of parameters of sealed blocks, whose arguments are final.
    fact_keys: std::cell::RefCell<crate::jit::FastMap<Val, Val>>,
    /// The first constant of each pool entry, which facts of every other
    /// constant of it are kept under.
    konst_keys: crate::jit::FastMap<u32, Val>,
    /// Callees' code as their bodies are built, by prototype.
    pub(crate) callees: crate::jit::FastMap<usize, Option<std::rc::Rc<inline::InlineCode>>>,
    /// Closures a `CLOSURE` of the compile made, with their prototypes.
    pub(crate) made: crate::jit::FastMap<Val, Gc<'gc, Prototype<'gc>>>,
    /// The inlining plan.
    pub(crate) plan: inline::Plan,
}

pub(crate) struct Builder<'a, 'gc> {
    pub(crate) ctx: crate::lua::Context<'gc>,
    pub(crate) f: Func<'gc>,
    /// The frame's closure, but for a callee known by its prototype, whose
    /// upvalues come from `Inline::closure`.
    pub(crate) closure: Option<LuaFn<'gc>>,
    /// Frame 0's closure, whose upvalues `UpvalValue` and `UpvalCell` read:
    /// an inlined body's builder has its callee as `closure`.
    pub(crate) region: LuaFn<'gc>,
    pub(crate) proto: Gc<'gc, Prototype<'gc>>,
    pub(crate) cfg: &'a Cfg,
    nregs: usize,
    bs: Vec<BlockState>,
    defs: Vec<Option<Val>>,
    slots: Vec<SlotState>,
    cur: Block,
    instances: Vec<Instance>,
    inst_succs: Vec<usize>,
    /// The instance of each (bytecode block, version), by block then version.
    inst_of: Vec<Vec<u32>>,
    scratch: Vec<Val>,
    /// Writes of the instruction being emitted, applied at its end so its
    /// exits see the frame before it.
    pending: Vec<(u8, Val)>,
    /// The `Before` snapshot of the instruction being emitted and the block
    /// it read the registers at: an arm's own is not a sibling arm's.
    cur_snap: Option<(Snap, Block)>,
    pub(crate) pc: u32,
    pub(crate) deopt_all: bool,
    pub(crate) known: Known<'gc>,
    /// The callee and first argument of the call at a pc, known before a
    /// `VARARG` up to `top` that reloads their registers.
    pub(crate) callee_hint: Option<(u32, Val, Val)>,
    /// The shape a `setmetatable` call's result has after it, guarded once
    /// its arms join (`intrinsics::moved_shape`).
    pub(crate) moved_shape: Option<u32>,
    /// What an inlined call left up to `top` from a register, for the next
    /// instruction; `cur_top` is the current instruction's.
    pub(crate) top: Option<(u8, Top)>,
    pub(crate) cur_top: Option<(u8, Top)>,
    /// Blocks made now are cold arms.
    cold: bool,
    /// A loop entry's frame, by register: what `intrinsics::value_now`
    /// guesses a loop header's value of a register is.
    pub(crate) entry_frame: Vec<Value<'gc>>,
    loop_entry: bool,
    /// Set when this builds an inlined callee's body.
    pub(crate) inline: Option<Inline>,
    /// Scratch for `read_var_at`'s walk.
    chain: Vec<Block>,
    /// The refinements the instruction being emitted made, in order: its
    /// `Before` snapshot, shared by all its arms, names what they refined.
    pc_refines: Vec<(Val, Val)>,
    /// The block the instruction being emitted started in: it dominates
    /// every arm of the instruction and what follows it.
    pc_block: Block,
    /// The `Before` snapshot exits take until a side effect (Maglev's
    /// checkpointed frame): every instruction from its pc on can run again.
    ckpt: Option<Snap>,
    /// The instruction being emitted has a side effect: `ckpt` ends with it.
    pc_effect: bool,
    /// What the next instance starts `ckpt` with: an inlined body's entry
    /// takes its call's.
    entry_ckpt: Option<Snap>,
    /// The site alone of the instruction being emitted, the blame of its
    /// exits whose feedback reads no operands, once one needs it.
    pc_site: Option<Snap>,
    /// The registers the instruction being emitted has written so far.
    pc_written: RegSet,
    /// Where the instruction being emitted starts: the block, and its last
    /// instruction then.
    #[cfg(feature = "tcvm-dev")]
    pc_start: (Block, Option<Inst>),
}

/// What the builder of an inlined callee knows of its call.
pub(crate) struct Inline {
    /// Its frame in `Func::inline_frames`.
    pub(crate) frame: u32,
    /// Its register 0, counted from the region's base.
    pub(crate) base: usize,
    /// The enclosing frames' snapshot entries at the call.
    pub(crate) outer: Vec<(u8, Val)>,
    /// The enclosing frames' slots that hold what they are read as after
    /// it, by region slot: a call in the body keeps them (`call_homes`).
    pub(crate) outer_homes: RegSet,
    /// Where its returns jump, with the `wanted` results as arguments.
    pub(crate) cont: Block,
    pub(crate) wanted: u8,
    /// How many inlined frames enclose it, itself included.
    pub(crate) depth: u8,
    /// It emitted an instruction that needs its frame on the stack.
    pub(crate) failed: bool,
    /// It, or a body inlined into it, makes a call, which may write its
    /// callers' captured registers through their cells.
    pub(crate) called: bool,
    /// It replaces the region's own frame: its returns and tail calls are the
    /// region's, and `cont` is none.
    pub(crate) region_return: bool,
    /// Its call wants every result: its returns pass `wanted` values, nil past
    /// theirs, and their count.
    pub(crate) counted: bool,
    /// Its call is a metamethod's, or the one of the body it replaces was.
    pub(crate) mm: bool,
    /// Parameters the body never writes whose arguments the caller's
    /// registers hold after the call, reported at each return.
    pub(crate) watch: Vec<u8>,
    /// Each return's jump and what `watch` holds there; none for a body
    /// that only exits, which is no inline.
    pub(crate) returns: Vec<(Inst, Vec<Val>)>,
    /// The callee closure (a `Ptr`) when known by its prototype only, whose
    /// upvalues are loaded from it; else the builder's closure is it.
    pub(crate) closure: Option<Val>,
    /// The window what is built in it needs, which counts toward the
    /// region's once it is inlined (`Builder::need_window`).
    pub(crate) window: u16,
    /// The caller's checkpoint at the call, which the body's exits take
    /// until its first side effect.
    pub(crate) ckpt: Option<Snap>,
    /// The body (or one inlined into it) has a side effect.
    pub(crate) effects: bool,
}

/// How many values an inlined call left up to `top`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Top {
    Fixed(u8),
    /// An i32, at most the second many.
    Count(Val, u8),
}

impl Inline {
    /// The enclosing frames' entries without its own header's (a callee
    /// known by its prototype), for what replaces its frame. One in the
    /// region frame's place (base 0) has no header slot.
    pub(crate) fn outer_entries(&self) -> Vec<(u8, Val)> {
        let hdr = self.base.checked_sub(4);
        self.outer
            .iter()
            .copied()
            .filter(|&(r, _)| Some(r as usize) != hdr)
            .collect()
    }
}

impl BlockState {
    fn placeholder() -> Self {
        BlockState {
            sealed: false,
            incomplete: Vec::new(),
            preds: Vec::new(),
            root: false,
            reload: RegSet::EMPTY,
            root_at: 0,
            root_ty: Vec::new(),
            after: None,
            terminated: false,
            refines: Vec::new(),
            on_entry: Vec::new(),
            facts: Facts::default(),
        }
    }
}

impl<'a, 'gc> Builder<'a, 'gc> {
    pub(crate) fn new(
        ctx: crate::lua::Context<'gc>,
        closure: LuaFn<'gc>,
        cfg: &'a Cfg,
        entry_pc: u32,
        loop_entry: bool,
    ) -> Self {
        let proto = closure.proto;
        let function = entry_pc == 0 && !loop_entry && !proto.is_vararg;
        let meta = RegionMeta {
            entry_pc,
            loop_entry,
            params: if function {
                proto.num_params.min(crate::jit::REG_ARGS)
            } else {
                0
            },
        };
        let mut f = Func::new(meta);
        f.captured = cfg.captured;
        Builder {
            ctx,
            f,
            closure: Some(closure),
            region: closure,
            proto,
            cfg,
            nregs: cfg.nregs,
            bs: Vec::new(),
            defs: Vec::new(),
            slots: Vec::new(),
            cur: Block(0),
            instances: Vec::new(),
            inst_succs: Vec::new(),
            inst_of: vec![Vec::new(); cfg.blocks.len()],
            scratch: Vec::new(),
            pending: Vec::new(),
            cur_snap: None,
            pc: entry_pc,
            deopt_all: false,
            known: Known::default(),
            callee_hint: None,
            moved_shape: None,
            top: None,
            cur_top: None,
            cold: false,
            entry_frame: Vec::new(),
            loop_entry,
            inline: None,
            chain: Vec::new(),
            pc_refines: Vec::new(),
            ckpt: None,
            pc_effect: false,
            entry_ckpt: None,
            pc_site: None,
            pc_written: RegSet::EMPTY,
            #[cfg(feature = "tcvm-dev")]
            pc_start: (Block(0), None),
            pc_block: Block(0),
        }
    }

    /// The builder of an inlined callee's body, which takes `outer`'s
    /// `Func`, block states and value facts until `finish`.
    pub(crate) fn new_inline(
        outer: &mut Builder<'_, 'gc>,
        closure: Option<LuaFn<'gc>>,
        proto: Gc<'gc, Prototype<'gc>>,
        cfg: &'a Cfg,
        inline: Inline,
    ) -> Self {
        let meta = outer.f.meta;
        let f = std::mem::replace(&mut outer.f, Func::new(meta));
        let n = f.blocks.len();
        Builder {
            ctx: outer.ctx,
            f,
            closure,
            region: outer.region,
            proto,
            cfg,
            nregs: cfg.nregs,
            bs: std::mem::take(&mut outer.bs),
            defs: vec![None; n * cfg.nregs],
            slots: vec![SlotState::Unknown; n * cfg.nregs],
            cur: Block(0),
            instances: Vec::new(),
            inst_succs: Vec::new(),
            inst_of: vec![Vec::new(); cfg.blocks.len()],
            scratch: Vec::new(),
            pending: Vec::new(),
            cur_snap: None,
            pc: 0,
            deopt_all: outer.deopt_all,
            known: std::mem::take(&mut outer.known),
            callee_hint: None,
            moved_shape: None,
            top: None,
            cur_top: None,
            cold: outer.cold,
            entry_frame: Vec::new(),
            loop_entry: false,
            entry_ckpt: inline.ckpt,
            inline: Some(inline),
            chain: Vec::new(),
            pc_refines: Vec::new(),
            ckpt: None,
            pc_effect: false,
            pc_site: None,
            pc_written: RegSet::EMPTY,
            #[cfg(feature = "tcvm-dev")]
            pc_start: (Block(0), None),
            pc_block: Block(0),
        }
    }

    /// Give `outer` back what `new_inline` took; the body's call state.
    pub(crate) fn finish(mut self, outer: &mut Builder<'_, 'gc>) -> Inline {
        let meta = self.f.meta;
        outer.f = std::mem::replace(&mut self.f, Func::new(meta));
        outer.bs = std::mem::take(&mut self.bs);
        outer.known = std::mem::take(&mut self.known);
        outer.pad_blocks();
        self.inline.take().expect("an inlined body")
    }

    /// Cover blocks another builder made since this one's last.
    fn pad_blocks(&mut self) {
        let n = self.f.blocks.len();
        self.defs.resize(n * self.nregs, None);
        self.slots.resize(n * self.nregs, SlotState::Unknown);
    }

    /// The inlined frame snapshots name, or `NO_FRAME`.
    fn frame_id(&self) -> u32 {
        self.inline
            .as_ref()
            .map_or(crate::jit::ir::NO_FRAME, |i| i.frame)
    }

    /// Register 0 counted from the region's base.
    pub(crate) fn reg_base(&self) -> usize {
        self.inline.as_ref().map_or(0, |i| i.base)
    }

    /// Register `r`'s home slot, counted from the region's base.
    pub(crate) fn slot_of(&self, r: u8) -> u8 {
        (self.reg_base() + r as usize) as u8
    }

    /// An inlined body's closure as a `Ptr`, which its upvalues and
    /// closures are read from; none for the region's own frame, whose
    /// closure the lowering has.
    pub(crate) fn body_closure(&mut self) -> Option<Val> {
        let i = self.inline.as_ref()?;
        if let Some(c) = i.closure {
            return Some(c);
        }
        let c = self.closure.expect("a constant callee");
        let k = self.konst(Value::function(c.function()));
        Some(self.ins(Op::Unbox(Rep::Ptr), &[k]))
    }

    /// The cell of the frame's shared upvalue `k`.
    pub(crate) fn upval_cell(&mut self, k: u8) -> Val {
        match self.body_closure() {
            Some(c) => self.ins(Op::UpvalCellOf(k), &[c]),
            None => self.ins(Op::UpvalCell(k), &[]),
        }
    }

    /// An inlined body cannot hold what needs its frame on the stack: its
    /// closure's upvalues as the frame's, a tail-out, a frame-relative
    /// helper. Its slots, calls and slot-reading helpers are the region's.
    fn inline_check(&mut self, op: Op) {
        use crate::jit::ir::ops::{Ctx, HelperId, Mem};
        let Some(i) = &mut self.inline else {
            return;
        };
        if i.region_return && matches!(op, Op::Return { .. } | Op::TailEnter { .. }) {
            return;
        }
        if matches!(
            op,
            Op::Load(_)
                | Op::Store(_)
                | Op::CellLoad
                | Op::CellStore
                | Op::SetTop(_)
                | Op::Call { .. }
                | Op::CallMm { .. }
                | Op::Resume { .. }
                | Op::Helper(
                    HelperId::TopArgs(_)
                        | HelperId::SetList { .. }
                        | HelperId::ClosureOf { .. }
                        | HelperId::Close(_)
                        | HelperId::TforNext { .. }
                )
        ) {
            return;
        }
        let e = op.effects();
        let frame = (e.reads | e.writes).intersects(Mem::SLOT | Mem::FRAME);
        let bad = frame
            || match op {
                Op::UpvalValue(_)
                | Op::UpvalCell(_)
                | Op::Dispatch { .. }
                | Op::TailEnter { .. }
                | Op::Return { .. } => true,
                Op::Helper(h) => h.sig().ctx.contains(&Ctx::Base),
                _ => false,
            };
        i.failed |= bad;
    }

    // --- blocks ---------------------------------------------------------------

    pub(crate) fn new_block(&mut self) -> Block {
        let b = self.f.new_block();
        self.f.blocks[b.idx()].cold = self.cold;
        self.defs.extend(std::iter::repeat_n(None, self.nregs));
        self.slots
            .extend(std::iter::repeat_n(SlotState::Unknown, self.nregs));
        self.bs.push(BlockState::placeholder());
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
        self.slots_of(b).fill(SlotState::Lazy(b));
        b
    }

    #[inline]
    fn def(&self, b: Block, r: u8) -> Option<Val> {
        self.defs[b.idx() * self.nregs + r as usize]
    }

    #[inline]
    fn set_def(&mut self, b: Block, r: u8, v: Val) {
        self.defs[b.idx() * self.nregs + r as usize] = Some(v);
    }

    #[inline]
    fn slot(&self, b: Block, r: usize) -> SlotState {
        self.slots[b.idx() * self.nregs + r]
    }

    #[inline]
    fn set_slot(&mut self, b: Block, r: usize, s: SlotState) {
        self.slots[b.idx() * self.nregs + r] = s;
    }

    fn slots_of(&mut self, b: Block) -> &mut [SlotState] {
        &mut self.slots[b.idx() * self.nregs..(b.idx() + 1) * self.nregs]
    }

    /// Continue in `b`; entered for the first time from the branch that
    /// alone reaches it, what the branch tells holds there.
    pub(crate) fn switch_to(&mut self, b: Block) {
        self.cur = b;
        if self.bs[b.idx()].on_entry.is_empty() {
            return;
        }
        let facts = std::mem::take(&mut self.bs[b.idx()].on_entry);
        let s = &self.bs[b.idx()];
        if s.sealed && s.preds.len() == 1 && self.f.blocks[b.idx()].insts.is_empty() {
            for (v, fact) in facts {
                self.bind_fact(v, fact);
            }
        }
    }

    /// Make the current block, and the blocks made until `end_cold`, a cold
    /// arm; the previous setting, for `end_cold`.
    pub(crate) fn begin_cold(&mut self) -> bool {
        self.f.blocks[self.cur.idx()].cold = true;
        std::mem::replace(&mut self.cold, true)
    }

    pub(crate) fn end_cold(&mut self, was: bool) {
        self.cold = was;
    }

    pub(crate) fn terminated(&self) -> bool {
        self.bs[self.cur.idx()].terminated
    }

    /// A block entered only from the current one, sealed.
    pub(crate) fn new_sealed_succ(&mut self) -> Block {
        let b = self.new_block();
        self.bs[b.idx()].sealed = true;
        // Its slots and facts start as the current block's end with them.
        let (n, c) = (self.nregs, self.cur.idx());
        self.slots.copy_within(c * n..(c + 1) * n, b.idx() * n);
        self.bs[b.idx()].facts = self.bs[c].facts.clone();
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
        // One argument per edge, in the order the edges were added.
        for j in 0..self.bs[b.idx()].preds.len() {
            let pred = self.bs[b.idx()].preds[j];
            let nth = self.bs[b.idx()].preds[..j]
                .iter()
                .filter(|&&x| x == pred)
                .count();
            let v = self.read_var_at(r, pred);
            let term = self
                .f
                .terminator(pred)
                .expect("a predecessor is terminated");
            let k = (0..self.f.edges(term).len())
                .filter(|&k| self.f.edges(term)[k].target == b)
                .nth(nth)
                .expect("an edge per predecessor");
            self.f.push_edge_arg(term, k, v);
        }
    }

    // --- SSA over registers ---------------------------------------------------

    pub(crate) fn read_var_at(&mut self, r: u8, b: Block) -> Val {
        if let Some(v) = self.def(b, r) {
            return v;
        }
        // Walk up single-predecessor chains iteratively.
        let mut x = b;
        let v = loop {
            if let Some(v) = self.def(x, r) {
                break v;
            }
            let s = &self.bs[x.idx()];
            if s.root || s.reload.contains(r as usize) {
                break self.root_load(x, r);
            }
            if !s.sealed {
                let p = self.f.add_param(x, Ty::ANY);
                self.bs[x.idx()].incomplete.push((r, p));
                self.set_def(x, r, p);
                break p;
            }
            if s.preds.len() == 1 {
                x = s.preds[0];
                continue;
            }
            if s.preds.is_empty() {
                // Unreachable code reads nothing meaningful.
                let v = self.konst_in(x, Value::nil());
                break v;
            }
            let p = self.f.add_param(x, Ty::ANY);
            self.set_def(x, r, p);
            self.add_param_args(x, r, p);
            break p;
        };
        // The chain walked again, top down, with the refinements of the
        // blocks below `x` (its own are in its definition).
        if self.def(x, r).is_none() {
            self.set_def(x, r, v);
        }
        let mut chain = std::mem::take(&mut self.chain);
        chain.clear();
        let mut c = b;
        while c != x {
            chain.push(c);
            c = self.bs[c.idx()].preds[0];
        }
        let mut v = v;
        for &c in chain.iter().rev() {
            for &(from, to) in &self.bs[c.idx()].refines {
                if v == from {
                    v = to;
                }
            }
            if self.def(c, r).is_none() {
                self.set_def(c, r, v);
            }
        }
        self.chain = chain;
        self.def(b, r).unwrap_or(v)
    }

    fn root_load(&mut self, b: Block, r: u8) -> Val {
        // A parameter its entry passed in a register; its slot holds
        // nothing until an exit writes it.
        let param = b == self.f.entry && r < self.f.meta.params;
        let op = if param {
            Op::Param(r)
        } else {
            Op::Load(self.slot_of(r))
        };
        self.inline_check(op);
        let set = self.bs[b.idx()]
            .root_ty
            .get(r as usize)
            .copied()
            .unwrap_or(TypeSet::ANY);
        let inst = self.f.make_inst(op, &[], None, ExitTag::Type);
        let at = self.bs[b.idx()].root_at;
        self.f.insert(b, at, inst);
        self.bs[b.idx()].root_at += 1;
        let v = self.f.result(inst);
        self.f.vals[v.idx()].ty = Ty::val(if set.is_empty() { TypeSet::ANY } else { set });
        self.set_def(b, r, v);
        if !param {
            self.set_slot(b, r as usize, SlotState::Holds(v));
        }
        if let Some(s) = self.bs[b.idx()].after {
            self.f.def_snaps.push((v, s));
        }
        v
    }

    fn konst_in(&mut self, b: Block, v: Value<'gc>) -> Val {
        let inst = self
            .f
            .make_inst(Op::KVal(v.to_raw()), &[], None, ExitTag::Type);
        // A resume block starts with its `Resume`.
        let first = self.f.insts_of(b).first().map(|&i| self.f.op(i));
        let at = matches!(first, Some(Op::Resume { .. })) as usize;
        self.f.insert(b, at, inst);
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
        for k in 0..self.pending.len() {
            let (r, v) = self.pending[k];
            self.pc_written.insert(r as usize);
            self.set_def(self.cur, r, v);
            // A slot whose root value is unchanged still holds it.
            let keep = match self.slot(self.cur, r as usize) {
                SlotState::Holds(h) => h == self.root_of(v),
                _ => false,
            };
            if !keep {
                self.set_slot(self.cur, r as usize, SlotState::Unknown);
            }
        }
        self.pending.clear();
        self.cur_snap = None;
    }

    /// The value `v` is a representation of: through guards, refinements,
    /// boxes and unboxes.
    pub(crate) fn root_of(&self, mut v: Val) -> Val {
        loop {
            match self.f.vals[v.idx()].def {
                ValDef::Inst(i, _) => match self.f.op(i) {
                    Op::Guard(_) | Op::Refine(_) | Op::Box | Op::Unbox(_) => v = self.f.args(i)[0],
                    _ => return v,
                },
                ValDef::Param(..) => return v,
            }
        }
    }

    /// Whether `r`'s home slot holds `v` now.
    fn slot_holds(&mut self, r: u8, v: Val) -> bool {
        match self.slot(self.cur, r as usize) {
            SlotState::Holds(h) => h == self.root_of(v),
            SlotState::Lazy(root) => {
                // Untouched since the root: `v` is the root's load of `r`.
                let rv = self.root_of(v);
                let s = self.slot_of(r);
                matches!(self.f.op_of_val(rv), Some(Op::Load(lr)) if lr == s)
                    && self.f.def_block(rv) == Some(root)
            }
            SlotState::Unknown => false,
        }
    }

    // --- refinements -----------------------------------------------------------

    /// From here on in the current block, and in blocks a read reaches it
    /// through, registers holding `v` hold `to`: a guard's result, a
    /// refinement, or the constant `v` was proven to be. A slot holding `v`
    /// holds that constant.
    pub(crate) fn refine(&mut self, v: Val, to: Val) {
        if v == to {
            return;
        }
        let cur = self.cur;
        if self.root_of(to) != self.root_of(v) {
            for r in 0..self.nregs {
                if self.slot_holds(r as u8, v) {
                    self.set_slot(cur, r, SlotState::Holds(to));
                }
            }
        }
        for r in 0..self.nregs as u8 {
            if self.def(cur, r) == Some(v) {
                self.set_def(cur, r, to);
            }
        }
        self.bs[cur.idx()].refines.push((v, to));
        self.pc_refines.push((v, to));
    }

    fn is_konst(&self, v: Val) -> bool {
        matches!(self.f.def_op(v), Some(Op::KVal(_) | Op::KObj(_)))
    }

    /// What `cond` being `holds` tells of a value: a narrower type set, or
    /// (an equality with a constant holding) the constant.
    fn fact_of(&self, cond: Val, holds: bool) -> Option<(Val, Fact)> {
        let i = self.f.def_inst(cond)?;
        let a = self.f.args(i);
        let (v, narrow) = match self.f.op(i) {
            Op::ShapeIs(k) => {
                let set = match self.known_shapes(a[0]) {
                    _ if holds => Shapes::one(k),
                    Some(s) => s.without(k),
                    None => return None,
                };
                return Some((a[0], Fact::Shapes(set)));
            }
            Op::ShapeIn(i) => {
                let ks = &self.f.shape_sets[i as usize];
                let set = match self.known_shapes(a[0]) {
                    Some(s) if holds => s.only(ks),
                    None if holds => Shapes::of(ks)?,
                    Some(s) => ks.iter().fold(s, |s, &k| s.without(k)),
                    None => return None,
                };
                return Some((a[0], Fact::Shapes(set)));
            }
            Op::IsType(set) => (a[0], if holds { set } else { TypeSet::ANY - set }),
            Op::IsFalsy if holds => (a[0], TypeSet::FALSY),
            Op::IsFalsy => (a[0], TypeSet::ANY - TypeSet::FALSY),
            Op::SameBits if self.is_konst(a[1]) => {
                if holds {
                    return Some((a[0], Fact::Is(a[1])));
                }
                let Some(Op::KVal(bits)) = self.f.def_op(a[1]) else {
                    return None;
                };
                let one = [
                    (Value::nil(), TypeSet::NIL),
                    (Value::boolean(false), TypeSet::FALSE),
                    (Value::boolean(true), TypeSet::TRUE),
                ]
                .into_iter()
                .find(|(k, _)| k.to_raw() == bits)?
                .1;
                (a[0], TypeSet::ANY - one)
            }
            _ => return None,
        };
        let t = self.ty(v);
        if t.rep != Rep::Val {
            return None;
        }
        let have = if t.set.is_empty() {
            TypeSet::ANY
        } else {
            t.set
        };
        let set = have & narrow;
        (set != have && !set.is_empty()).then_some((v, Fact::Narrow(set)))
    }

    /// Bind `fact` of `v` in the current block from here on.
    fn bind_fact(&mut self, v: Val, fact: Fact) {
        let to = match fact {
            Fact::Is(k) => k,
            Fact::Narrow(set) => self.ins(Op::Refine(set), &[v]),
            Fact::Shapes(s) => return self.set_shapes(v, s),
        };
        self.refine(v, to);
    }

    /// The value facts about `v` are kept under: its root, through
    /// parameters every edge passes the same root (a register read where
    /// arms that guarded it join).
    fn fact_key(&self, v: Val) -> Val {
        self.fact_key_at(v, 8)
    }

    fn fact_key_at(&self, v: Val, depth: u32) -> Val {
        let r = self.root_of(v);
        let ValDef::Param(blk, k) = self.f.vals[r.idx()].def else {
            return match self.f.def_op(r) {
                Some(Op::KObj(idx)) => self.known.konst_keys.get(&idx).copied().unwrap_or(r),
                _ => r,
            };
        };
        if depth == 0 || !self.bs[blk.idx()].sealed {
            return r;
        }
        if let Some(&x) = self.known.fact_keys.borrow().get(&r) {
            return x;
        }
        let x = self.param_fact_key(r, blk, k, depth);
        self.known.fact_keys.borrow_mut().insert(r, x);
        x
    }

    /// The fact key of parameter `r`, the `k`th of sealed block `blk`.
    fn param_fact_key(&self, r: Val, blk: Block, k: u16, depth: u32) -> Val {
        let s = &self.bs[blk.idx()];
        let mut same = None;
        for &pred in &s.preds {
            let Some(term) = self.f.terminator(pred) else {
                return r;
            };
            for e in self.f.edges(term).iter().filter(|e| e.target == blk) {
                let Some(&a) = self.f.vl(e.args).get(k as usize) else {
                    return r;
                };
                if a == r {
                    continue;
                }
                let ka = self.fact_key_at(a, depth - 1);
                match same {
                    None => same = Some(ka),
                    Some(x) if x == ka => {}
                    Some(_) => return r,
                }
            }
        }
        same.unwrap_or(r)
    }

    /// The shapes the table `t` unboxes from may have here, if known.
    pub(crate) fn known_shapes(&self, t: Val) -> Option<Shapes> {
        let r = self.fact_key(t);
        let s = &self.bs[self.cur.idx()].facts.shapes;
        let &(_, set, stable) = s.iter().find(|x| x.0 == r)?;
        if stable {
            self.f
                .stable_used
                .borrow_mut()
                .extend_from_slice(set.as_slice());
        }
        Some(set)
    }

    /// Know that the table `t` unboxes from has one of `s` here.
    pub(crate) fn set_shapes(&mut self, t: Val, s: Shapes) {
        self.put_shapes(t, s, false);
    }

    /// Know that the table `t` unboxes from has `s` while `s` stays stable.
    pub(crate) fn set_stable_shapes(&mut self, t: Val, s: Shapes) {
        self.put_shapes(t, s, true);
    }

    fn put_shapes(&mut self, t: Val, s: Shapes, stable: bool) {
        let r = self.fact_key(t);
        let facts = &mut self.bs[self.cur.idx()].facts.shapes;
        match facts.iter_mut().find(|x| x.0 == r) {
            Some(x) => (x.1, x.2) = (s, stable),
            None => facts.push((r, s, stable)),
        }
    }

    /// Whether code may rely on the tables of every shape in `s` staying.
    fn all_stable(&self, s: Shapes) -> bool {
        s.as_slice().iter().all(|&k| {
            let shape = self.f.shapes[k as usize];
            shape.is_stable() && !shape.is_dict()
        })
    }

    /// The facts of stable shapes, for after an instruction that may change
    /// shapes and is followed by a lazy exit.
    fn stable_facts(&self, b: Block) -> Vec<(Val, Shapes)> {
        let facts = &self.bs[b.idx()].facts.shapes;
        facts
            .iter()
            .filter(|x| self.all_stable(x.1))
            .map(|x| (x.0, x.1))
            .collect()
    }

    /// An instruction that may change shapes ends what is known of them,
    /// but for stable shapes past a lazy exit after it: calls (`call`), and
    /// the helpers followed by one.
    fn note_effects(&mut self, op: Op) {
        if !op
            .effects()
            .writes
            .contains(crate::jit::ir::ops::Mem::SHAPE)
        {
            return;
        }
        let lazy = matches!(
            op,
            Op::Helper(HelperId::NewIndex { .. } | HelperId::RawSet | HelperId::SetMetatable)
        );
        let stash = match lazy {
            true => self.stable_facts(self.cur),
            false => Vec::new(),
        };
        let s = &mut self.bs[self.cur.idx()].facts;
        s.shapes.clear();
        s.stash = stash;
    }

    /// `t` (of one of the shapes known for it, none stable) moves to shape
    /// `k`: facts of tables that cannot be `t`, their shapes disjoint from
    /// its, stay.
    pub(crate) fn move_shape(&mut self, t: Val, k: u32) {
        let from = self.known_shapes(t);
        let i = self
            .f
            .make_inst(Op::TabSetShape(k), &[t], None, ExitTag::Type);
        self.inline_check(Op::TabSetShape(k));
        self.note_effect(Op::TabSetShape(k));
        self.f.append(self.cur, i);
        let r = self.fact_key(t);
        let s = &mut self.bs[self.cur.idx()].facts;
        s.stash.clear();
        match from {
            Some(f) => s
                .shapes
                .retain(|x| x.0 != r && !x.1.as_slice().iter().any(|&j| f.contains(j))),
            None => s.shapes.clear(),
        }
        self.set_shapes(t, Shapes::one(k));
    }

    /// What an exiting check just emitted proves from here on.
    fn note_check(&mut self, i: Inst) {
        let op = self.f.op(i);
        let a0 = self.f.args(i).first().copied();
        match op {
            Op::Guard(_) => {
                let g = self.f.result(i);
                self.refine(a0.unwrap(), g);
            }
            Op::GuardSame if self.is_konst(self.f.args(i)[1]) => {
                let k = self.f.args(i)[1];
                self.refine(a0.unwrap(), k);
            }
            Op::GuardTrue | Op::GuardFalse => {
                if let Some((v, fact)) = self.fact_of(a0.unwrap(), op == Op::GuardTrue) {
                    self.bind_fact(v, fact);
                }
            }
            _ => {}
        }
    }

    // --- instructions -------------------------------------------------------

    pub(crate) fn push(&mut self, op: Op, args: &[Val]) -> Inst {
        // A cold arm's allocation has its own collector check, so the loop
        // around it needs none: garbage waits for the arm's next run.
        if let Op::Helper(h) = op
            && self.f.blocks[self.cur.idx()].cold
            && h.sig().effects.has(crate::jit::ir::ops::Flags::MAY_ALLOC)
        {
            let s = self.snap_before();
            self.ins_snap(Op::GcCheck, &[], s, ExitTag::Gc);
        }
        self.inline_check(op);
        self.note_effects(op);
        self.note_effect(op);
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
        self.inline_check(op);
        self.note_effects(op);
        self.note_effect(op);
        let s = self.snap_before();
        let i = self.f.make_inst(op, args, Some(s), tag);
        self.f.append(self.cur, i);
        self.note_site(i, s);
        self.note_check(i);
        i
    }

    pub(crate) fn ins_snap(&mut self, op: Op, args: &[Val], snap: Snap, tag: ExitTag) -> Inst {
        self.inline_check(op);
        self.note_effects(op);
        self.note_effect(op);
        let i = self.f.make_inst(op, args, Some(snap), tag);
        self.f.append(self.cur, i);
        self.note_site(i, snap);
        self.note_check(i);
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
        edges: &[Block],
        snap: Option<(Snap, ExitTag)>,
    ) -> Inst {
        debug_assert!(!self.terminated());
        self.inline_check(op);
        self.note_effect(op);
        let (s, tag) = match snap {
            Some((s, t)) => (Some(s), t),
            None => (None, ExitTag::Type),
        };
        let i = self.f.make_inst(op, args, s, tag);
        self.f.append(self.cur, i);
        if let Some(s) = s {
            self.note_site(i, s);
        }
        let none = BlockCall {
            target: Block(0),
            args: List::default(),
        };
        let mut calls = [none; 2];
        for (c, &t) in calls.iter_mut().zip(edges) {
            c.target = t;
        }
        self.f.set_edges(i, &calls[..edges.len()]);
        for &t in edges {
            self.bs[t.idx()].preds.push(self.cur);
        }
        self.bs[self.cur.idx()].terminated = true;
        // Edges into blocks already sealed get their arguments now.
        for (k, &t) in edges.iter().enumerate() {
            if self.bs[t.idx()].sealed && !self.f.blocks[t.idx()].params.is_empty() {
                let mut args = std::mem::take(&mut self.scratch);
                args.clear();
                for j in 0..self.f.blocks[t.idx()].params.len() {
                    let r = self.param_reg(t, self.f.params(t)[j]);
                    args.push(self.read_var_at(r, self.cur));
                }
                self.f.set_edge_args(i, k, &args);
                self.scratch = args;
            }
        }
        i
    }

    /// The one value a sealed block's parameter receives on every edge
    /// (through such parameters of earlier joins), or the register an
    /// unsealed one stands for.
    pub(crate) fn param_source(&self, p: Val) -> Option<Result<Val, u8>> {
        self.param_source_at(p, 8)
    }

    fn param_source_at(&self, p: Val, depth: u32) -> Option<Result<Val, u8>> {
        let ValDef::Param(blk, k) = self.f.vals[p.idx()].def else {
            return None;
        };
        let s = &self.bs[blk.idx()];
        if !s.sealed {
            return s
                .incomplete
                .iter()
                .find(|(_, q)| *q == p)
                .map(|&(r, _)| Err(r));
        }
        let mut same = None;
        for &pred in &s.preds {
            let term = self.f.terminator(pred)?;
            for e in self.f.edges(term).iter().filter(|e| e.target == blk) {
                let mut a = *self.f.vl(e.args).get(k as usize)?;
                if depth > 0
                    && a != p
                    && let Some(Ok(x)) = self.param_source_at(a, depth - 1)
                {
                    a = x;
                }
                match same {
                    _ if a == p => {}
                    None => same = Some(a),
                    Some(x) if x == a => {}
                    Some(_) => return None,
                }
            }
        }
        same.map(Ok)
    }

    /// The register a block parameter stands for.
    fn param_reg(&self, b: Block, p: Val) -> u8 {
        if let Some(r) = self.bs[b.idx()]
            .incomplete
            .iter()
            .find(|(_, q)| *q == p)
            .map(|&(r, _)| r)
        {
            return r;
        }
        (0..self.nregs as u8)
            .find(|&r| self.def(b, r) == Some(p))
            .expect("block parameter without a register")
    }

    pub(crate) fn jump(&mut self, target: Block) {
        self.terminate(Op::Jump, &[], &[target], None);
    }

    /// Branch on `cond`; what it tells each edge holds in the block it
    /// enters when that is reached from here alone.
    pub(crate) fn br(&mut self, cond: Val, t: Block, f: Block) {
        self.terminate(Op::Br, &[cond], &[t, f], None);
        if t == f {
            return;
        }
        for (b, holds) in [(t, true), (f, false)] {
            if let Some(fact) = self.fact_of(cond, holds) {
                self.bs[b.idx()].on_entry.push(fact);
            }
        }
    }

    // --- snapshots and tail-outs ----------------------------------------------

    /// Registers whose values an exit or a tail-out at `live` must provide:
    /// the live ones and every captured one.
    fn exit_regs(&self, live: RegSet) -> RegSet {
        let mut s = live;
        s.union(&self.cfg.captured);
        s
    }

    /// `at`: the block whose register values the snapshot names: the current
    /// one, or for a checkpoint, which other arms and later instructions
    /// take too, the one its instruction started in.
    fn snapshot(&mut self, pc: u32, kind: ExitKind, live: RegSet, at: Block) -> Snap {
        let regs = self.exit_regs(live);
        // Reading a register creates no snapshot, so the entries are the
        // pool's tail.
        let from = self.f.snap_pool.len();
        // An inlined frame's registers follow its callers', by register.
        if let Some(i) = &self.inline {
            self.f.snap_pool.extend_from_slice(&i.outer);
        }
        let base = self.reg_base();
        for r in regs.iter() {
            if r >= self.nregs || self.untouched(r) {
                continue;
            }
            let mut v = self.read_var_at(r as u8, at);
            if kind == ExitKind::Before {
                for &(from, to) in self.pc_refines.iter().rev() {
                    if v == to {
                        v = from;
                    }
                }
            }
            if !self.slot_holds(r as u8, v) {
                self.f.snap_pool.push(((base + r) as u8, v));
            }
        }
        let frame = self.frame_id();
        self.f.finish_snap(frame, pc, kind, from)
    }

    /// This frame's and its callers' snapshot entries at a call at the
    /// current pc, for a callee inlined there: the registers `live` after
    /// it below `a`, where its frame starts.
    pub(crate) fn call_entries(&mut self, live: RegSet, a: u8) -> Vec<(u8, Val)> {
        let regs = self.exit_regs(live);
        let mut out = self.inline.as_ref().map_or(Vec::new(), |i| i.outer.clone());
        let base = self.reg_base();
        for r in regs.iter() {
            if r >= a as usize || r >= self.nregs || self.untouched(r) {
                continue;
            }
            let v = self.reg(r as u8);
            if !self.slot_holds(r as u8, v) {
                out.push(((base + r) as u8, v));
            }
        }
        out
    }

    /// The snapshot that re-runs the instruction being emitted.
    pub(crate) fn snap_before(&mut self) -> Snap {
        if !self.pc_effect
            && let Some(s) = self.ckpt
        {
            return s;
        }
        if let Some((s, at)) = self.cur_snap
            && (at == self.pc_block || at == self.cur)
        {
            return s;
        }
        let live = self.cfg.live_in[self.pc as usize];
        let at = if self.pc_effect {
            self.cur
        } else {
            self.pc_block
        };
        let s = self.snapshot(self.pc, ExitKind::Before, live, at);
        self.cur_snap = Some((s, at));
        if !self.pc_effect {
            self.ckpt = Some(s);
            #[cfg(feature = "tcvm-dev")]
            self.f.ckpts.push((s.0, self.pc_start.0, self.pc_start.1));
        }
        s
    }

    /// The checkpoint an exit here would take, made if there is none yet.
    pub(crate) fn live_ckpt(&mut self) -> Option<Snap> {
        (!self.pc_effect).then(|| self.snap_before())
    }

    /// A side effect ends the checkpoint with the instruction making it.
    fn note_effect(&mut self, op: Op) {
        if op.side_effect() {
            self.side_effect();
        }
    }

    pub(crate) fn side_effect(&mut self) {
        self.pc_effect = true;
        if let Some(i) = self.inline.as_mut() {
            i.effects = true;
        }
    }

    /// An exit taking the checkpoint of an earlier instruction is blamed on
    /// this one.
    fn note_site(&mut self, i: Inst, s: Snap) {
        let d = self.f.snaps[s.idx()];
        if self.ckpt != Some(s)
            || d.kind != ExitKind::Before
            || (d.pc, d.frame) == (self.pc, self.frame_id())
        {
            return;
        }
        if !self.f.insts[i.idx()].tag.blames_operands() {
            let site = match self.pc_site {
                Some(b) => b,
                None => {
                    let from = self.f.snap_pool.len();
                    let b = (self.f).finish_snap(self.frame_id(), self.pc, ExitKind::Before, from);
                    *self.pc_site.insert(b)
                }
            };
            self.f.insts[i.idx()].blame = site.0;
            return;
        }
        let mut ents = self.blame();
        // A shape guard names the table it tested: an upvalue's is in no
        // register.
        if self.f.insts[i.idx()].tag == ExitTag::Shape
            && let Some(&t) = self.f.args(i).first()
        {
            let t = self.root_of(t);
            if self.ty(t).rep == Rep::Val || matches!(self.f.def_op(t), Some(Op::KPtr(_))) {
                ents.push((crate::jit::ir::BLAME_TESTED, t));
            }
        }
        let from = self.f.snap_pool.len();
        self.f.snap_pool.extend_from_slice(&ents);
        let b = (self.f).finish_snap(self.frame_id(), self.pc, ExitKind::Before, from);
        self.f.insts[i.idx()].blame = b.0;
    }

    /// The blame entries of an exit of the instruction being emitted: the
    /// registers it reads (but those it wrote). One whose kind is known
    /// here, by its type or a guard of the instruction that passed, is
    /// named by a constant of the kind, which is all its feedback reads;
    /// the others, the one the exit's guard tests among them, by their
    /// values.
    fn blame(&mut self) -> Vec<(u8, Val)> {
        let insn = self.code(self.pc);
        let uses = cfg::use_def(&self.proto, insn, self.nregs, &self.cfg.captured).uses;
        let mut ents = Vec::new();
        for r in uses.iter() {
            if self.pc_written.contains(r) {
                continue;
            }
            let v = self.reg(r as u8);
            let ty = self.ty(v);
            // At the block's start: the exit's instruction is in already.
            let v = if ty.rep == Rep::F64 || ty.set == TypeSet::FLOAT {
                self.konst_in(self.cur, Value::float(0.0))
            } else if ty.rep == Rep::I32 || ty.set == TypeSet::SMALL {
                self.konst_in(self.cur, Value::small(0))
            } else {
                v
            };
            ents.push((r as u8, v));
        }
        ents
    }

    /// The snapshot that continues after the instruction at `pc`, once its
    /// writes are flushed.
    pub(crate) fn snap_after(&mut self, pc: u32) -> Snap {
        let live = match self.cfg.live_in.get(pc as usize + 1) {
            Some(&l) => l,
            None => RegSet::EMPTY,
        };
        self.snapshot(pc, ExitKind::After, live, self.cur)
    }

    /// Whether `r`'s home slot still holds what it held at the root: the
    /// register is the root's (perhaps not yet created) load.
    fn untouched(&self, r: usize) -> bool {
        matches!(self.slot(self.cur, r), SlotState::Lazy(_))
    }

    /// Note the `After` snapshot of `v`'s definition, the instruction at the
    /// current pc, and of the loads of the block it starts when that is a
    /// resume block.
    pub(crate) fn note_def(&mut self, vals: &[Val], resume: Option<Block>) -> Snap {
        self.flush();
        let s = self.snap_after(self.pc);
        for &v in vals {
            self.f.def_snaps.push((v, s));
        }
        if let Some(b) = resume {
            self.bs[b.idx()].after = Some(s);
        }
        s
    }

    /// A lazy exit with `s`, for a point right after a call or a table write,
    /// before anything that assumes what a dependency names.
    pub(crate) fn lazy_exit(&mut self, s: Snap) {
        self.ins_snap(Op::LazyExit, &[], s, ExitTag::Lazy);
        let st = &mut self.bs[self.cur.idx()].facts;
        for (v, set) in std::mem::take(&mut st.stash) {
            if !st.shapes.iter().any(|x| x.0 == v) {
                st.shapes.push((v, set, true));
            }
        }
    }

    /// A lazy exit past the instruction, once its writes are flushed.
    pub(crate) fn lazy_exit_after(&mut self) {
        debug_assert!(self.pending.is_empty());
        let s = self.snap_after(self.pc);
        self.lazy_exit(s);
    }

    /// The snapshot that runs the instruction at `pc` from the current
    /// state: a successor of the one being emitted.
    pub(crate) fn snap_at(&mut self, pc: u32) -> Snap {
        let live = self.cfg.live_in[pc as usize];
        self.snapshot(pc, ExitKind::Before, live, self.cur)
    }

    /// Assume `class` keeps `bits` as they are now.
    pub(crate) fn depend_class(&mut self, class: crate::env::MtCache<'gc>, bits: MetamethodBits) {
        self.depend(DepOn::Class(class), bits.bits() as u64);
    }

    /// Assume the watched holder of `shape` keeps it, and `groups` of it.
    pub(crate) fn depend_holder(&mut self, shape: crate::env::Shape<'gc>, groups: u64) {
        self.depend(DepOn::Shape(shape), groups);
    }

    /// Assume `field`'s kind stays.
    pub(crate) fn depend_field(&mut self, field: crate::env::shape::Field<'gc>) {
        self.depend(DepOn::Field(field), 1);
    }

    fn depend(&mut self, on: DepOn<'gc>, groups: u64) {
        self.f.depend(on, groups);
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
            if let SlotState::Lazy(root) = self.slot(self.cur, r) {
                let t = self.bs[root.idx()].root_ty.get(r).copied();
                tys[r] = t.filter(|t| !t.is_empty()).unwrap_or(TypeSet::ANY);
                continue;
            }
            let v = self.reg(r as u8);
            if !self.slot_holds(r as u8, v) {
                let b = self.boxed(v);
                self.push(Op::Store(self.slot_of(r as u8)), &[b]);
                let root = self.root_of(v);
                self.set_slot(self.cur, r, SlotState::Holds(root));
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

    /// Store the registers in `regs` whose home slots do not hold their
    /// values, for a helper that reads them there.
    pub(crate) fn store_regs(&mut self, regs: impl IntoIterator<Item = usize>) {
        for r in regs {
            if r >= self.nregs || self.untouched(r) {
                continue;
            }
            let v = self.reg(r as u8);
            if !self.slot_holds(r as u8, v) {
                let bv = self.boxed(v);
                self.push(Op::Store(self.slot_of(r as u8)), &[bv]);
                let root = self.root_of(v);
                self.set_slot(self.cur, r, SlotState::Holds(root));
            }
        }
    }

    /// Continue in a block whose registers are read from their home slots,
    /// where a helper wrote some; `tys` the types of the stored ones.
    pub(crate) fn sync_root(&mut self, tys: Vec<TypeSet>) {
        let root = self.new_root(tys);
        self.terminate(Op::Jump, &[], &[root], None);
        self.bs[root.idx()].preds.clear();
        self.switch_to(root);
    }

    /// `r` loaded from its home slot, which a helper wrote.
    pub(crate) fn reload(&mut self, r: u8) -> Val {
        let v = self.ins(Op::Load(self.slot_of(r)), &[]);
        self.set_def(self.cur, r, v);
        self.set_slot(self.cur, r as usize, SlotState::Holds(v));
        v
    }

    /// Leave for the interpreter at the instruction being emitted.
    pub(crate) fn deopt(&mut self, tag: ExitTag) {
        let s = self.snap_before();
        self.terminate(Op::Deopt, &[], &[], Some((s, tag)));
    }

    /// End the block with a call (`Call`, `CallMm`) to a fresh resume block,
    /// which becomes current. Of the registers `live` after it, those it
    /// `written` and the captured ones are read from their slots after it;
    /// those whose slots hold their values keep them there (its homes: the
    /// call-boundary pass reloads what is used after it, typed as before); the
    /// others keep their values in the region's frame across it, the call-site
    /// snapshot naming them for the collector.
    pub(crate) fn call(
        &mut self,
        op: Op,
        args: &[Val],
        live: RegSet,
        written: RegSet,
        c: u8,
    ) -> Block {
        if let Some(i) = self.inline.as_mut() {
            i.called = true;
        }
        let mut reload = written;
        reload.union(&self.cfg.captured);
        // By region slot: the captured registers are stored and read through
        // their cells. A body's call keeps its callers' entries and homes.
        let base = self.reg_base();
        let mut homes = RegSet::EMPTY;
        for r in self.cfg.captured.iter() {
            homes.insert(base + r);
        }
        let from = self.f.snap_pool.len();
        if let Some(i) = &self.inline {
            homes.union(&i.outer_homes);
            self.f.snap_pool.extend_from_slice(&i.outer);
        }
        for r in live.iter() {
            if r >= self.nregs || reload.contains(r) {
                continue;
            }
            if self.untouched(r) {
                homes.insert(base + r);
                continue;
            }
            let v = self.reg(r as u8);
            if self.slot_holds(r as u8, v) {
                homes.insert(base + r);
            } else {
                self.f.snap_pool.push(((base + r) as u8, v));
            }
        }
        self.inline_check(op);
        let frame = self.frame_id();
        let snap = self.f.finish_snap(frame, self.pc, ExitKind::After, from);
        // A snapshot equal to an earlier one is shared: every call of it
        // traces what any of them keeps in its slots.
        self.f.call_homes.entry(snap.0).or_default().union(&homes);
        let before = self.cur;
        let resume = self.new_block();
        self.f.blocks[resume.idx()].resume = true;
        let stash = self.stable_facts(before);
        let s = &mut self.bs[resume.idx()];
        s.sealed = true;
        s.reload = reload;
        s.facts.stash = stash;
        self.terminate(op, args, &[resume], Some((snap, ExitTag::Type)));
        for r in 0..self.nregs {
            let state = if reload.contains(r) {
                SlotState::Lazy(resume)
            } else if homes.contains(base + r) {
                self.slot(before, r)
            } else {
                SlotState::Unknown
            };
            self.set_slot(resume, r, state);
        }
        self.switch_to(resume);
        self.push(Op::Resume { c }, &[]);
        self.bs[resume.idx()].root_at = 1;
        resume
    }

    /// The value stack holds `end` slots past the region's base for what is
    /// built here (`Func::window`): a body's need counts once it inlines.
    pub(crate) fn need_window(&mut self, end: u16) {
        match &mut self.inline {
            Some(i) => i.window = i.window.max(end),
            None => self.f.window = self.f.window.max(end),
        }
    }

    /// End an inlined body with a real call that replaces its frame (its
    /// `TAILCALL`): made from the body's header slot, as its caller made the
    /// call the body was inlined at, so its site is the caller's frame there,
    /// keeping what the caller kept. Continues in the resume block, where the
    /// body's registers are dead.
    pub(crate) fn tail_call(&mut self, op: Op, args: &[Val], c: u8) -> Block {
        let i = self.inline.as_mut().expect("an inlined body");
        i.called = true;
        let x = self.f.inline_frames[i.frame as usize];
        let homes = i.outer_homes;
        let from = self.f.snap_pool.len();
        let outer = self.inline.as_ref().unwrap().outer_entries();
        self.f.snap_pool.extend_from_slice(&outer);
        self.inline_check(op);
        let snap = self
            .f
            .finish_snap(x.parent, x.caller_pc, ExitKind::After, from);
        self.f.call_homes.entry(snap.0).or_default().union(&homes);
        let before = self.cur;
        let resume = self.new_block();
        self.f.blocks[resume.idx()].resume = true;
        let stash = self.stable_facts(before);
        let s = &mut self.bs[resume.idx()];
        s.sealed = true;
        s.facts.stash = stash;
        self.terminate(op, args, &[resume], Some((snap, ExitTag::Type)));
        for r in 0..self.nregs {
            self.set_slot(resume, r, SlotState::Unknown);
        }
        self.switch_to(resume);
        self.push(Op::Resume { c }, &[]);
        self.bs[resume.idx()].root_at = 1;
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
        self.known.konst_keys.entry(idx).or_insert(k);
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

    pub(crate) fn intern_shape(&mut self, s: crate::env::Shape<'gc>) -> u32 {
        match self
            .f
            .shapes
            .iter()
            .position(|&x| crate::env::Shape::ptr_eq(x, s))
        {
            Some(k) => k as u32,
            None => {
                self.f.shapes.push(s);
                self.f.shapes.len() as u32 - 1
            }
        }
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
        // An unboxed value outside `set` (a float where a table is wanted)
        // is tested boxed, like any other.
        let v = self.boxed(v);
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
        if let Some(i) = self.instance_of(bc, version) {
            return i;
        }
        let ir = self.new_block();
        self.instances.push(Instance {
            bc,
            version,
            ir,
            succs: List::default(),
            npreds: 0,
            done_preds: 0,
            started: false,
            discovered: false,
        });
        let i = self.instances.len() - 1;
        let row = &mut self.inst_of[bc as usize];
        if row.len() <= version as usize {
            row.resize(version as usize + 1, u32::MAX);
        }
        row[version as usize] = i as u32;
        i
    }

    fn instance_of(&self, bc: u32, version: u32) -> Option<usize> {
        let &i = self.inst_of[bc as usize].get(version as usize)?;
        (i != u32::MAX).then_some(i as usize)
    }

    fn succ_at(&self, ii: usize, k: usize) -> usize {
        self.inst_succs[self.instances[ii].succs.start as usize + k]
    }

    /// Discover the instances reachable from the entry, versioned for
    /// outer-loop peeling, and return them in reverse post-order.
    fn discover(&mut self, entry_bc: u32) -> Vec<usize> {
        let cfg = self.cfg;
        // The loops around the entry, outermost first.
        let mut nest = Vec::new();
        let mut l = if self.loop_entry {
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
        // A back edge of level `j` past the `d` merged outermost levels
        // starts a fresh copy of that level's header.
        let succ_version = |a: u32, k: u32, b: u32, d: u32| -> u32 {
            for (j, &li) in nest.iter().enumerate() {
                let j = j as u32;
                if j >= d && j < k && cfg.loops[li].header == b && cfg.loops[li].body[a as usize] {
                    return j;
                }
            }
            k.min(level(b))
        };
        // Merge outermost levels until the copies fit `OSR_PEEL_LIMIT`.
        let nv = n as usize + 1;
        let mut d = 0;
        while d < n {
            let mut seen = vec![false; cfg.blocks.len() * nv];
            let mut dup = 0;
            let v0 = n.min(level(entry_bc));
            let mut work = vec![(entry_bc, v0)];
            seen[entry_bc as usize * nv + v0 as usize] = true;
            let mut first = vec![true; cfg.blocks.len()];
            while let Some((bc, k)) = work.pop() {
                let blk = &cfg.blocks[bc as usize];
                if !std::mem::take(&mut first[bc as usize]) {
                    dup += blk.end - blk.start;
                }
                if self.block_deopts(bc) {
                    continue;
                }
                for &s in &blk.succs {
                    let v = succ_version(bc, k, s, d);
                    if !std::mem::replace(&mut seen[s as usize * nv + v as usize], true) {
                        work.push((s, v));
                    }
                }
            }
            if dup <= OSR_PEEL_LIMIT {
                break;
            }
            d += 1;
        }
        let start = self.instance(entry_bc, n.min(level(entry_bc)));
        let mut work = vec![start];
        self.instances[start].discovered = true;
        while let Some(i) = work.pop() {
            let (bc, k) = (self.instances[i].bc, self.instances[i].version);
            let succs: &[u32] = if self.block_deopts(bc) {
                &[]
            } else {
                &cfg.blocks[bc as usize].succs
            };
            let mut out = [0usize; 4];
            for (o, &s) in out.iter_mut().zip(succs) {
                let v = succ_version(bc, k, s, d);
                let si = self.instance(s, v);
                *o = si;
                self.instances[si].npreds += 1;
                if !self.instances[si].discovered {
                    self.instances[si].discovered = true;
                    work.push(si);
                }
            }
            let start = self.inst_succs.len();
            self.inst_succs.extend_from_slice(&out[..succs.len()]);
            self.instances[i].succs = List {
                start: start as u32,
                len: succs.len() as u32,
            };
        }
        // RPO over instances.
        let mut visited = vec![false; self.instances.len()];
        let mut post = Vec::new();
        let mut stack = vec![(start, 0usize)];
        visited[start] = true;
        while let Some(&mut (i, ref mut k)) = stack.last_mut() {
            if *k < self.instances[i].succs.len() {
                let s = self.succ_at(i, *k);
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
        // An entry at the entry loop's header ran a whole iteration before
        // the loop proper; one at its latch (a FORLOOP, a goto's LOOP) only
        // the latch, and the loop is peeled as any other.
        if let Some(&li) = nest.last() {
            let h = cfg.loops[li].header;
            if h == entry_bc
                && cfg.blocks[h as usize].start == self.f.meta.entry_pc
                && let Some(ii) = self.instance_of(h, n - 1)
            {
                let b = self.instances[ii].ir;
                self.f.blocks[b.idx()].peeled = true;
            }
        }
        post
    }

    /// Whether the block is compiled as one deopt: it never ran, or an
    /// instruction in it always deopts, cutting its successors.
    fn block_deopts(&self, bc: u32) -> bool {
        if self.cfg.never_ran[bc as usize] {
            return true;
        }
        let blk = &self.cfg.blocks[bc as usize];
        (blk.start..blk.end).any(|pc| emit::always_deopts(self, pc))
    }

    // --- driver -------------------------------------------------------------

    pub(crate) fn build(mut self) -> Result<Func<'gc>, BuildError> {
        inline::make_plan(&mut self);
        let entry_pc = self.f.meta.entry_pc;
        let entry_bc = self.cfg.block_of[entry_pc as usize];
        let prologue = self.new_root(vec![TypeSet::ANY; self.nregs]);
        self.f.entry = prologue;
        for r in 0..self.f.meta.params as usize {
            self.set_slot(prologue, r, SlotState::Unknown);
        }
        let order = self.discover(entry_bc);
        let first = self.instances[order[0]].ir;
        self.switch_to(prologue);
        // The collector's chance at each entry; `prune_gc_checks` keeps it
        // when code outside loops allocates.
        self.pc = entry_pc;
        self.pc_block = prologue;
        let s = self.snap_before();
        self.ins_snap(Op::GcCheck, &[], s, ExitTag::Gc);
        self.jump(first);
        self.emit_order(&order, 4000)?;
        // Any block left unsealed (its unreached predecessors) is sealed now.
        for b in 0..self.bs.len() {
            if !self.bs[b].sealed {
                self.seal(Block(b as u32));
            }
        }
        // Blocks no edge reaches.
        let mut live = vec![false; self.f.blocks.len()];
        for &b in &self.f.cfg().rpo {
            live[b.idx()] = true;
        }
        for (b, l) in live.iter().enumerate() {
            if !l {
                self.f.blocks[b].dead = true;
            }
        }
        remove_trivial_params(&mut self.f);
        let mut used = std::mem::take(self.f.stable_used.get_mut());
        used.sort_unstable();
        used.dedup();
        for k in used {
            let s = self.f.shapes[k as usize];
            self.depend(DepOn::Shape(s), crate::jit::deps::SHAPE_GROUP);
        }
        // Nothing can invalidate a region without dependencies.
        if self.f.deps.is_empty() {
            for b in 0..self.f.blocks.len() {
                self.f
                    .retain(Block(b as u32), |f, i| f.op(i) != Op::LazyExit);
            }
        }
        Ok(self.f)
    }

    /// Build an inlined callee's body from `args`, its parameters, and the
    /// facts of the caller's block `from`: its entry block, unreached until
    /// the caller jumps there, or none when it needs its frame
    /// (`Inline::failed`) or grew past `limit` instructions.
    pub(crate) fn build_inline(
        &mut self,
        args: &[Val],
        from: Block,
        limit: usize,
    ) -> Option<Block> {
        let first_block = self.f.blocks.len();
        let entry = self.new_block();
        self.bs[entry.idx()].sealed = true;
        self.bs[entry.idx()].facts = self.bs[from.idx()].facts.clone();
        for r in 0..self.proto.num_params as usize {
            let v = match args.get(r) {
                Some(&v) => v,
                None => self.konst_in(entry, Value::nil()),
            };
            self.set_def(entry, r as u8, v);
        }
        let entry_bc = self.cfg.block_of[0];
        let order = self.discover(entry_bc);
        let first = self.instances[order[0]].ir;
        self.switch_to(entry);
        self.jump(first);
        let built = self.emit_order(&order, limit).is_ok();
        for b in first_block..self.bs.len() {
            if !self.bs[b].sealed {
                self.seal(Block(b as u32));
            }
        }
        let ok = built
            && self
                .inline
                .as_ref()
                .is_some_and(|i| !i.failed && !i.returns.is_empty());
        if !ok {
            for b in first_block..self.f.blocks.len() {
                self.f.blocks[b].dead = true;
            }
        }
        ok.then_some(entry)
    }

    /// Emit the instances in `order`, whose first the current block jumped
    /// to; past `limit` instructions, give up.
    fn emit_order(&mut self, order: &[usize], limit: usize) -> Result<(), BuildError> {
        self.instances[order[0]].npreds += 1;
        self.instances[order[0]].done_preds += 1;
        for &ii in order {
            let (bc, ir) = (self.instances[ii].bc, self.instances[ii].ir);
            let reached = self.instances[ii].done_preds > 0 || !self.bs[ir.idx()].preds.is_empty();
            self.instances[ii].started = true;
            if self.instances[ii].done_preds == self.instances[ii].npreds {
                self.seal(ir);
            }
            if reached {
                self.init_slots(ir);
                self.switch_to(ir);
                self.emit_instance(ii, bc);
            } else {
                self.f.blocks[ir.idx()].dead = true;
                self.bs[ir.idx()].terminated = true;
            }
            for k in 0..self.instances[ii].succs.len() {
                let s = self.succ_at(ii, k);
                self.instances[s].done_preds += 1;
                if self.instances[s].started
                    && self.instances[s].done_preds == self.instances[s].npreds
                {
                    let sb = self.instances[s].ir;
                    self.seal(sb);
                }
            }
            if self.f.built() > limit || self.inline.as_ref().is_some_and(|i| i.failed) {
                return Err(BuildError::TooLarge);
            }
        }
        Ok(())
    }

    /// The slot state and facts at the start of a block: the meet over its
    /// emitted predecessors, unknown at a loop header.
    fn init_slots(&mut self, b: Block) {
        self.init_facts(b);
        let n = self.nregs;
        if !self.bs[b.idx()].sealed || self.bs[b.idx()].preds.is_empty() {
            if !self.bs[b.idx()].root {
                self.slots_of(b).fill(SlotState::Unknown);
            }
            return;
        }
        let p0 = self.bs[b.idx()].preds[0];
        self.slots
            .copy_within(p0.idx() * n..(p0.idx() + 1) * n, b.idx() * n);
        for j in 1..self.bs[b.idx()].preds.len() {
            let p = self.bs[b.idx()].preds[j];
            for r in 0..n {
                if self.slot(b, r) != self.slot(p, r) {
                    self.set_slot(b, r, SlotState::Unknown);
                }
            }
        }
    }

    /// The facts at the start of a block: the meet over its predecessors',
    /// none at a loop header.
    fn init_facts(&mut self, b: Block) {
        let s = &self.bs[b.idx()];
        let facts = match s.preds.split_first() {
            Some((p0, rest)) if s.sealed => {
                let mut x = self.bs[p0.idx()].facts.clone();
                x.stash.clear();
                for p in rest {
                    x.meet(&self.bs[p.idx()].facts);
                }
                x
            }
            _ => Facts::default(),
        };
        self.bs[b.idx()].facts = facts;
    }

    /// What the builder keeps of one instruction starts over at `pc`.
    fn start_pc(&mut self, pc: u32) {
        self.pc = pc;
        self.pc_block = self.cur;
        self.cur_snap = None;
        self.pc_refines.clear();
        self.pc_site = None;
        self.pc_written = RegSet::EMPTY;
        #[cfg(feature = "tcvm-dev")]
        {
            // Not a lazy exit, which goes when the region has no dependency.
            let insts = self.f.insts_of(self.cur).iter().rev();
            let last = insts.copied().find(|&i| self.f.op(i) != Op::LazyExit);
            self.pc_start = (self.cur, last);
        }
    }

    fn emit_instance(&mut self, ii: usize, bc: u32) {
        let blk = &self.cfg.blocks[bc as usize];
        let (start, end) = (blk.start, blk.end);
        self.pc_effect = false;
        self.start_pc(start);
        // An inlined body's entry continues its call's checkpoint, unless a
        // loop's back edge enters it too.
        self.ckpt = self
            .entry_ckpt
            .take()
            .filter(|_| !self.cfg.loops.iter().any(|l| l.header == bc));
        if self.cfg.never_ran[bc as usize] {
            self.deopt(ExitTag::NeverRan);
            return;
        }
        if self.cfg.loops.iter().any(|l| l.header == bc) {
            // The collector's chance once per iteration; `prune_gc_checks`
            // keeps it only in loops that allocate or call.
            let s = self.snap_before();
            self.ins_snap(Op::GcCheck, &[], s, ExitTag::Gc);
        }
        let mut sb = [Block(0); 4];
        let ns = self.instances[ii].succs.len();
        for (k, s) in sb.iter_mut().enumerate().take(ns) {
            *s = self.instances[self.succ_at(ii, k)].ir;
        }
        for pc in start..end {
            self.start_pc(pc);
            let last = pc + 1 == end;
            // A `next` or `ipairs` step branches as its TFORLOOP would.
            let fused = pc + 2 == end
                && matches!(
                    self.code(pc).op(),
                    BcOp::TFORCALL_NEXT | BcOp::TFORCALL_IPAIRS
                )
                && self.code(pc + 1).op() == BcOp::TFORLOOP;
            let succs = if last || fused { &sb[..ns] } else { &[][..] };
            emit::emit(self, pc, succs);
            self.flush();
            if std::mem::take(&mut self.pc_effect) {
                self.ckpt = None;
            }
            if self.terminated() {
                return;
            }
            if last {
                // Fallthrough into the single successor.
                match succs {
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

    pub(crate) fn feedback(&self, pc: u32) -> crate::jit::feedback::Bits {
        self.proto.feedback.bits(pc as usize)
    }

    /// Whether compiled code exited at the site at `pc` on operand kinds its
    /// form does not cover. Its form may have changed since, and may change
    /// again: the interpreter's forms do not only widen, a region's speculation
    /// must.
    pub(crate) fn kind_exited(&self, pc: u32) -> bool {
        self.feedback(pc) & crate::jit::feedback::KIND_EXIT != 0
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

/// The value a boxed value refines: through guards and refinements.
fn refined_from(f: &Func<'_>, mut v: Val) -> Val {
    while let Some(i) = f.def_inst(v) {
        match f.op(i) {
            Op::Guard(_) | Op::Refine(_) => v = f.args(i)[0],
            _ => break,
        }
    }
    v
}

/// Remove parameters whose incoming arguments are all one other value (or
/// the parameter itself, refined or not), to a fixpoint. Arguments that are
/// refinements of one value (a register guarded differently on each path)
/// join to that value, refined to their union where that is narrower.
pub(crate) fn remove_trivial_params(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let mut map: Vec<Val> = Vec::new();
    let mut kept: Vec<Val> = Vec::new();
    // Parameters replaced by a refinement of a value, made where the block
    // starts once the replacements settle.
    let mut refined: Vec<(Block, Val, Val, TypeSet)> = Vec::new();
    // Parameters replaced by a constant (op, one argument for its type).
    let mut konsts: Vec<(Block, Val, Op, Val)> = Vec::new();
    let mut heads: Vec<usize> = Vec::new();
    loop {
        map.clear();
        map.extend((0..f.vals.len() as u32).map(Val));
        let mut changed = false;
        for &b in &cfg.rpo {
            for k in 0..f.blocks[b.idx()].params.len() {
                let p = f.params(b)[k];
                let mut same: Option<Val> = None;
                let mut trivial = true;
                // The one value the arguments refine, and their types' union.
                let mut base: Option<(Val, TypeSet)> = None;
                for &(term, e) in cfg.incoming(b) {
                    let a = resolve(&map, f.edge_args(term, e as usize)[k]);
                    // A refinement of the parameter is its value.
                    if a == p || refined_from(f, a) == p {
                        continue;
                    }
                    let (ra, ta) = (refined_from(f, a), f.vals[a.idx()].ty);
                    base = match base {
                        _ if ta.rep != Rep::Val || f.vals[p.idx()].ty.rep != Rep::Val => None,
                        None if same.is_none() => Some((ra, ta.set)),
                        Some((x, set)) if x == ra => Some((x, set | ta.set)),
                        _ => None,
                    };
                    match same {
                        None => same = Some(a),
                        Some(s) if s == a => {}
                        Some(_) => {
                            trivial = false;
                        }
                    }
                    if !trivial && base.is_none() {
                        break;
                    }
                }
                // Copies of one constant (made again after calls): a copy
                // where the block starts, as no argument need dominate it.
                let mut konst: Option<(Op, Val)> = None;
                let copies = !trivial
                    && cfg.incoming(b).iter().all(|&(term, e)| {
                        let a = resolve(&map, f.edge_args(term, e as usize)[k]);
                        let op = match f.def_op(a) {
                            Some(op @ (Op::KVal(_) | Op::KObj(_) | Op::KI32(_))) => op,
                            Some(op @ (Op::KI64(_) | Op::KF64(_) | Op::KB1(_))) => op,
                            _ => return a == p,
                        };
                        f.vals[a.idx()].ty.rep == f.vals[p.idx()].ty.rep
                            && konst.get_or_insert((op, a)).0 == op
                    });
                if copies && let Some((op, a)) = konst {
                    konsts.push((b, p, op, a));
                    changed = true;
                    continue;
                }
                if !trivial
                    && let Some((x, set)) = base
                    && x != p
                {
                    let whole = f.vals[x.idx()].ty.set;
                    if set.is_empty() || whole.is_empty() || set.contains(whole) {
                        map[p.idx()] = x;
                    } else {
                        refined.push((b, p, x, set & whole));
                    }
                    changed = true;
                    continue;
                }
                // An i64 not known small into an i32 parameter would need
                // an exit, which has no snapshot here: the parameter stays.
                let narrows = |s: Val| {
                    let t = f.vals[s.idx()].ty;
                    t.rep == Rep::I64
                        && f.vals[p.idx()].ty.rep == Rep::I32
                        && !t.within(TypeSet::SMALL)
                };
                if trivial
                    && let Some(s) = same
                    && !narrows(s)
                {
                    map[p.idx()] = s;
                    changed = true;
                    continue;
                }
                // A parameter every edge passes what it passes an earlier
                // one is that one: a loop's index and its variable.
                for j in 0..k {
                    let q = f.params(b)[j];
                    if resolve(&map, q) != q || f.vals[q.idx()].ty != f.vals[p.idx()].ty {
                        continue;
                    }
                    let dup = cfg.incoming(b).iter().all(|&(term, e)| {
                        let args = f.edge_args(term, e as usize);
                        let (x, y) = (resolve(&map, args[j]), resolve(&map, args[k]));
                        x == y || (x == q && y == p)
                    });
                    if dup {
                        map[p.idx()] = q;
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
        // What this round put where each block starts: a conversion of one
        // of them goes after it.
        heads.clear();
        heads.resize(f.blocks.len(), 0);
        for (b, p, x, set) in refined.drain(..) {
            let x = resolve(&map, x);
            let i = f.make_inst(Op::Refine(set), &[x], None, ExitTag::Type);
            f.insert(b, heads[b.idx()], i);
            heads[b.idx()] += 1;
            let r = f.result(i);
            while map.len() < f.vals.len() {
                map.push(Val(map.len() as u32));
            }
            map[p.idx()] = r;
        }
        for (b, p, op, a) in konsts.drain(..) {
            let i = f.make_inst(op, &[], None, ExitTag::Type);
            f.insert(b, heads[b.idx()], i);
            heads[b.idx()] += 1;
            let r = f.result(i);
            f.vals[r.idx()].ty = f.vals[a.idx()].ty;
            while map.len() < f.vals.len() {
                map.push(Val(map.len() as u32));
            }
            map[p.idx()] = r;
        }
        // A replacement of another representation enters through a
        // conversion at the parameter's block.
        for &b in &cfg.rpo {
            let mut at = heads[b.idx()];
            for k in 0..f.blocks[b.idx()].params.len() {
                let p = f.params(b)[k];
                let s = resolve(&map, p);
                if s == p || f.vals[s.idx()].ty.rep == f.vals[p.idx()].ty.rep {
                    continue;
                }
                let rep = f.vals[p.idx()].ty.rep;
                let op = conversion(f.vals[s.idx()].ty, rep);
                let i = f.make_inst(op, &[s], None, ExitTag::Type);
                f.insert(b, at, i);
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
        for &b in &cfg.rpo {
            let n = f.blocks[b.idx()].params.len();
            if f.params(b).iter().all(|&p| map[p.idx()] == p) {
                continue;
            }
            for &(term, e) in cfg.incoming(b) {
                kept.clear();
                for k in 0..n {
                    if map[f.params(b)[k].idx()] == f.params(b)[k] {
                        kept.push(f.edge_args(term, e as usize)[k]);
                    }
                }
                f.set_edge_args(term, e as usize, &kept);
            }
            kept.clear();
            kept.extend(f.params(b).iter().copied().filter(|&p| map[p.idx()] == p));
            f.set_params(b, &kept);
        }
    }
}

/// The operation converting a value of type `from` to `to`, which exits
/// nowhere; a pair without one fails the compile.
pub(crate) fn conversion(from: Ty, to: Rep) -> Op {
    match (from.rep, to) {
        (Rep::I32 | Rep::I64 | Rep::F64 | Rep::B1, Rep::Val) => Op::Box,
        (Rep::I32, Rep::I64) => Op::IToL,
        (Rep::I32, Rep::F64) => Op::IToF,
        (Rep::I64, Rep::F64) => Op::LToF,
        (Rep::I64, Rep::I32) if from.within(TypeSet::SMALL) => Op::LTrunc,
        (Rep::Val, rep) if rep != Rep::Val => Op::Unbox(rep),
        // Contained: the compile fails (`tier::compile_caught`).
        (from, to) => panic!("no conversion from {} to {}", from.name(), to.name()),
    }
}

fn resolve(map: &[Val], mut v: Val) -> Val {
    while map[v.idx()] != v {
        v = map[v.idx()];
    }
    v
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
