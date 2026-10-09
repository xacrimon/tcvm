//! The region IR: a CFG of blocks with parameters, instructions in SSA form,
//! snapshots in a side table, and a constant pool of the heap values the
//! code embeds. Every list (a block's instructions and parameters, operands,
//! edge arguments, snapshot entries) is a run of a flat pool (8.1).

pub(crate) mod ops;
pub(crate) mod print;
pub(crate) mod types;
pub(crate) mod verify;

use std::cell::RefCell;
use std::rc::Rc;

use crate::env::value::Value;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Ty, TypeSet};

macro_rules! idx {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        pub(crate) struct $name(pub(crate) u32);

        impl $name {
            #[inline]
            pub(crate) fn idx(self) -> usize {
                self.0 as usize
            }
        }
    };
}

idx!(Block);
idx!(Inst);
idx!(Val);
idx!(Snap);

pub(crate) const NO_SNAP: u32 = u32::MAX;

/// A run `pool[start..start + len]` of one of a `Func`'s pools. Growing a run
/// that does not end its pool moves it to the end, so a copy of the old
/// handle keeps reading the old elements.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct List {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

impl List {
    #[inline]
    pub(crate) fn len(self) -> usize {
        self.len as usize
    }

    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self.len == 0
    }

    #[inline]
    fn range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

fn list_push<T: Copy>(pool: &mut Vec<T>, l: &mut List, x: T) {
    if l.range().end != pool.len() {
        let s = pool.len();
        pool.extend_from_within(l.range());
        l.start = s as u32;
    }
    pool.push(x);
    l.len += 1;
}

fn list_insert<T: Copy>(pool: &mut Vec<T>, l: &mut List, at: usize, x: T) {
    let s = pool.len();
    let r = l.range();
    pool.extend_from_within(r.start..r.start + at);
    pool.push(x);
    pool.extend_from_within(r.start + at..r.end);
    *l = List {
        start: s as u32,
        len: l.len + 1,
    };
}

fn list_new<T: Copy>(pool: &mut Vec<T>, xs: &[T]) -> List {
    let s = pool.len();
    pool.extend_from_slice(xs);
    List {
        start: s as u32,
        len: xs.len() as u32,
    }
}

/// Where a value is defined.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ValDef {
    Inst(Inst, u8),
    Param(Block, u16),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValData {
    pub(crate) ty: Ty,
    pub(crate) def: ValDef,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct InstData {
    pub(crate) op: Op,
    /// Operands: `vpool[a0..a0 + an]`.
    pub(crate) a0: u32,
    pub(crate) an: u16,
    /// Results: `Val(r0)..Val(r0 + rn)`.
    pub(crate) r0: u32,
    pub(crate) rn: u8,
    /// Successor edges of a terminator: `edges[e0..e0 + en]`.
    pub(crate) e0: u32,
    pub(crate) en: u8,
    pub(crate) snap: u32,
    pub(crate) tag: ExitTag,
    pub(crate) block: Block,
}

/// A branch target with the arguments for its parameters (a run of
/// `vpool`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockCall {
    pub(crate) target: Block,
    pub(crate) args: List,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BlockData {
    /// A run of `ppool`.
    pub(crate) params: List,
    /// A run of `layout`.
    pub(crate) insts: List,
    /// The resume block of a call: entered by a callee's return, not by a
    /// branch.
    pub(crate) resume: bool,
    /// Deleted by a pass; skipped by everything.
    pub(crate) dead: bool,
    /// A loop header the region enters after a whole iteration of the loop
    /// (an entry at its header, 8.7), so peeling it again buys nothing.
    pub(crate) peeled: bool,
}

/// How an exit resumes the interpreter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ExitKind {
    /// Run the instruction at `pc`.
    Before,
    /// The instruction at `pc` is done: continue at `pc + 1`.
    After,
    /// As `After`, through the collector.
    Gc,
}

/// The frame an exit rebuilds: home slots to write and where to resume.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapData {
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    /// `(register, value)` by register: a run of `snap_pool`.
    pub(crate) entries: List,
}

/// What a region was compiled from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegionMeta {
    pub(crate) entry_pc: u32,
    pub(crate) loop_entry: bool,
    pub(crate) max_stack: u8,
}

/// The control-flow facts passes share, computed on demand and dropped by
/// any change to an edge's target or the set of blocks.
pub(crate) struct CfgInfo {
    /// Blocks reachable from the entry, in reverse post-order.
    pub(crate) rpo: Vec<Block>,
    preds_at: Vec<u32>,
    preds: Vec<Block>,
    inc_at: Vec<u32>,
    inc: Vec<(Inst, u32)>,
    pub(crate) idom: Vec<Option<Block>>,
}

impl CfgInfo {
    /// The predecessors of `b`, once per edge.
    #[inline]
    pub(crate) fn preds(&self, b: Block) -> &[Block] {
        &self.preds[self.preds_at[b.idx()] as usize..self.preds_at[b.idx() + 1] as usize]
    }

    /// The (terminator, edge index) pairs entering `b`.
    #[inline]
    pub(crate) fn incoming(&self, b: Block) -> &[(Inst, u32)] {
        &self.inc[self.inc_at[b.idx()] as usize..self.inc_at[b.idx() + 1] as usize]
    }

    pub(crate) fn dominates(&self, a: Block, b: Block) -> bool {
        verify::dominates(&self.idom, a, b)
    }
}

pub(crate) struct Func<'gc> {
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) insts: Vec<InstData>,
    pub(crate) vals: Vec<ValData>,
    /// Operands and edge arguments.
    pub(crate) vpool: Vec<Val>,
    /// Block parameters.
    pub(crate) ppool: Vec<Val>,
    /// Blocks' instructions.
    pub(crate) layout: Vec<Inst>,
    pub(crate) edges: Vec<BlockCall>,
    pub(crate) snaps: Vec<SnapData>,
    pub(crate) snap_pool: Vec<(u8, Val)>,
    pub(crate) pool: Vec<Value<'gc>>,
    pub(crate) entry: Block,
    pub(crate) meta: RegionMeta,
    /// The `After` snapshot of a value's definition, where a definition
    /// guard (9.1) may exit: call results, loads after a call, upvalues.
    pub(crate) def_snaps: Vec<(Val, Snap)>,
    /// The call-boundary pass ran: nothing is live across a call (R6).
    pub(crate) boundaries: bool,
    cfg: RefCell<Option<Rc<CfgInfo>>>,
}

/// The type set of a constant `Value` word.
pub(crate) fn const_set(bits: u64) -> TypeSet {
    let v: Value<'static> = unsafe { std::mem::transmute::<u64, Value<'static>>(bits) };
    if v.is_nil() {
        TypeSet::NIL
    } else if let Some(b) = v.get_boolean() {
        if b { TypeSet::TRUE } else { TypeSet::FALSE }
    } else if v.get_small().is_some() {
        TypeSet::SMALL
    } else if v.is_float() {
        TypeSet::FLOAT
    } else {
        TypeSet::ANY
    }
}

impl<'gc> Func<'gc> {
    pub(crate) fn new(meta: RegionMeta) -> Self {
        Func {
            blocks: Vec::new(),
            insts: Vec::new(),
            vals: Vec::new(),
            vpool: Vec::new(),
            ppool: Vec::new(),
            layout: Vec::new(),
            edges: Vec::new(),
            snaps: Vec::new(),
            snap_pool: Vec::new(),
            pool: Vec::new(),
            entry: Block(0),
            meta,
            def_snaps: Vec::new(),
            boundaries: false,
            cfg: RefCell::new(None),
        }
    }

    // --- control flow ---------------------------------------------------------

    /// The shared control-flow facts.
    pub(crate) fn cfg(&self) -> Rc<CfgInfo> {
        if let Some(c) = &*self.cfg.borrow() {
            return c.clone();
        }
        let c = Rc::new(self.compute_cfg());
        *self.cfg.borrow_mut() = Some(c.clone());
        c
    }

    /// Drop the control-flow facts after a change they depend on.
    #[inline]
    pub(crate) fn cfg_changed(&mut self) {
        *self.cfg.get_mut() = None;
    }

    fn compute_cfg(&self) -> CfgInfo {
        let rpo = self.rpo();
        let n = self.blocks.len();
        let mut preds_at = vec![0u32; n + 1];
        for &b in &rpo {
            for s in self.succs(b) {
                preds_at[s.idx() + 1] += 1;
            }
        }
        for k in 0..n {
            preds_at[k + 1] += preds_at[k];
        }
        let inc_at = preds_at.clone();
        let mut fill = preds_at.clone();
        let mut preds = vec![Block(0); preds_at[n] as usize];
        let mut inc = vec![(Inst(0), 0u32); preds_at[n] as usize];
        for &b in &rpo {
            let Some(t) = self.terminator(b) else {
                continue;
            };
            for (k, e) in self.edges(t).iter().enumerate() {
                let at = &mut fill[e.target.idx()];
                preds[*at as usize] = b;
                inc[*at as usize] = (t, k as u32);
                *at += 1;
            }
        }
        let mut info = CfgInfo {
            rpo,
            preds_at,
            preds,
            inc_at,
            inc,
            idom: Vec::new(),
        };
        info.idom = verify::dominators(self, &info);
        info
    }

    /// Blocks in reverse post-order from the entry.
    pub(crate) fn rpo(&self) -> Vec<Block> {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(Block, usize)> = vec![(self.entry, 0)];
        seen[self.entry.idx()] = true;
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            let succs = match self.terminator(b) {
                Some(t) => self.edges(t),
                None => &[],
            };
            if *i < succs.len() {
                let s = succs[*i].target;
                *i += 1;
                if !seen[s.idx()] {
                    seen[s.idx()] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        post
    }

    pub(crate) fn terminator(&self, b: Block) -> Option<Inst> {
        let &last = self.insts_of(b).last()?;
        self.op(last).is_terminator().then_some(last)
    }

    pub(crate) fn succs(&self, b: Block) -> impl Iterator<Item = Block> + '_ {
        self.terminator(b)
            .into_iter()
            .flat_map(move |t| self.edges(t).iter().map(|e| e.target))
    }

    // --- blocks ---------------------------------------------------------------

    pub(crate) fn new_block(&mut self) -> Block {
        self.blocks.push(BlockData::default());
        self.cfg_changed();
        Block(self.blocks.len() as u32 - 1)
    }

    #[inline]
    pub(crate) fn insts_of(&self, b: Block) -> &[Inst] {
        &self.layout[self.blocks[b.idx()].insts.range()]
    }

    #[inline]
    pub(crate) fn params(&self, b: Block) -> &[Val] {
        &self.ppool[self.blocks[b.idx()].params.range()]
    }

    pub(crate) fn add_param(&mut self, b: Block, ty: Ty) -> Val {
        let k = self.blocks[b.idx()].params.len as u16;
        let v = Val(self.vals.len() as u32);
        self.vals.push(ValData {
            ty,
            def: ValDef::Param(b, k),
        });
        list_push(&mut self.ppool, &mut self.blocks[b.idx()].params, v);
        v
    }

    /// Replace `b`'s parameters, renumbering their definitions.
    pub(crate) fn set_params(&mut self, b: Block, params: &[Val]) {
        self.blocks[b.idx()].params = list_new(&mut self.ppool, params);
        for (k, &p) in params.iter().enumerate() {
            self.vals[p.idx()].def = ValDef::Param(b, k as u16);
        }
    }

    pub(crate) fn append(&mut self, b: Block, inst: Inst) {
        self.insts[inst.idx()].block = b;
        list_push(&mut self.layout, &mut self.blocks[b.idx()].insts, inst);
    }

    /// Insert `inst` at position `at` of `b`.
    pub(crate) fn insert(&mut self, b: Block, at: usize, inst: Inst) {
        self.insts[inst.idx()].block = b;
        list_insert(&mut self.layout, &mut self.blocks[b.idx()].insts, at, inst);
    }

    /// Insert `inst` into `b` before its terminator.
    pub(crate) fn insert_before_term(&mut self, b: Block, inst: Inst) {
        let n = self.blocks[b.idx()].insts.len();
        let at = match self.terminator(b) {
            Some(_) => n - 1,
            None => n,
        };
        self.insert(b, at, inst);
    }

    /// Keep the instructions of `b` that `keep` accepts, in place.
    pub(crate) fn retain(&mut self, b: Block, mut keep: impl FnMut(&Self, Inst) -> bool) {
        let r = self.blocks[b.idx()].insts.range();
        let mut out = r.start;
        for k in r.clone() {
            let i = self.layout[k];
            if keep(self, i) {
                self.layout[out] = i;
                out += 1;
            }
        }
        self.blocks[b.idx()].insts.len = (out - r.start) as u32;
    }

    // --- values -----------------------------------------------------------------

    /// The pool index of a heap value.
    pub(crate) fn intern(&mut self, v: Value<'gc>) -> u32 {
        if let Some(i) = self.pool.iter().position(|p| p.same_bits(&v)) {
            return i as u32;
        }
        self.pool.push(v);
        self.pool.len() as u32 - 1
    }

    #[inline]
    pub(crate) fn vl(&self, l: List) -> &[Val] {
        &self.vpool[l.range()]
    }

    #[inline]
    pub(crate) fn vl_mut(&mut self, l: List) -> &mut [Val] {
        &mut self.vpool[l.range()]
    }

    pub(crate) fn new_vlist(&mut self, vals: &[Val]) -> List {
        list_new(&mut self.vpool, vals)
    }

    #[inline]
    pub(crate) fn op(&self, inst: Inst) -> Op {
        self.insts[inst.idx()].op
    }

    #[inline]
    pub(crate) fn ty(&self, v: Val) -> Ty {
        self.vals[v.idx()].ty
    }

    /// The instruction defining `v`, unless it is a parameter.
    #[inline]
    pub(crate) fn def_inst(&self, v: Val) -> Option<Inst> {
        match self.vals[v.idx()].def {
            ValDef::Inst(i, _) => Some(i),
            ValDef::Param(..) => None,
        }
    }

    pub(crate) fn def_op(&self, v: Val) -> Option<Op> {
        self.def_inst(v).map(|i| self.op(i))
    }

    // --- instructions -----------------------------------------------------------

    /// Create an instruction (not placed in a block).
    pub(crate) fn make_inst(
        &mut self,
        op: Op,
        args: &[Val],
        snap: Option<Snap>,
        tag: ExitTag,
    ) -> Inst {
        let inst = Inst(self.insts.len() as u32);
        let a0 = self.vpool.len() as u32;
        self.vpool.extend_from_slice(args);
        let nres = op.num_results();
        let r0 = self.vals.len() as u32;
        if nres > 0 {
            let ty = self.result_ty(op, args);
            for k in 0..nres {
                self.vals.push(ValData {
                    ty,
                    def: ValDef::Inst(inst, k as u8),
                });
            }
        }
        self.insts.push(InstData {
            op,
            a0,
            an: args.len() as u16,
            r0,
            rn: nres as u8,
            e0: 0,
            en: 0,
            snap: snap.map_or(NO_SNAP, |s| s.0),
            tag,
            block: Block(u32::MAX),
        });
        inst
    }

    /// `op`'s result type from its operands' (only the first three matter).
    pub(crate) fn result_ty(&self, op: Op, args: &[Val]) -> Ty {
        let mut tys = [Ty::ANY; 3];
        let n = args.len().min(3);
        for (t, &a) in tys.iter_mut().zip(&args[..n]) {
            *t = self.vals[a.idx()].ty;
        }
        op.result_ty(&tys[..n])
    }

    pub(crate) fn set_edges(&mut self, inst: Inst, edges: &[BlockCall]) {
        let e0 = self.edges.len() as u32;
        self.edges.extend_from_slice(edges);
        let d = &mut self.insts[inst.idx()];
        d.e0 = e0;
        d.en = edges.len() as u8;
        self.cfg_changed();
    }

    /// Point edge `k` of `inst` at `target`.
    pub(crate) fn set_target(&mut self, inst: Inst, k: usize, target: Block) {
        let e0 = self.insts[inst.idx()].e0 as usize;
        self.edges[e0 + k].target = target;
        self.cfg_changed();
    }

    #[inline]
    pub(crate) fn args(&self, inst: Inst) -> &[Val] {
        let d = &self.insts[inst.idx()];
        &self.vpool[d.a0 as usize..d.a0 as usize + d.an as usize]
    }

    #[inline]
    pub(crate) fn args_mut(&mut self, inst: Inst) -> &mut [Val] {
        let d = self.insts[inst.idx()];
        &mut self.vpool[d.a0 as usize..d.a0 as usize + d.an as usize]
    }

    #[inline]
    pub(crate) fn result(&self, inst: Inst) -> Val {
        let d = &self.insts[inst.idx()];
        debug_assert!(d.rn >= 1, "{:?} has no result", d.op);
        Val(d.r0)
    }

    pub(crate) fn results(&self, inst: Inst) -> impl Iterator<Item = Val> + use<> {
        let d = &self.insts[inst.idx()];
        (d.r0..d.r0 + d.rn as u32).map(Val)
    }

    #[inline]
    pub(crate) fn edges(&self, inst: Inst) -> &[BlockCall] {
        let d = &self.insts[inst.idx()];
        &self.edges[d.e0 as usize..d.e0 as usize + d.en as usize]
    }

    /// The arguments of edge `k` of `inst`.
    #[inline]
    pub(crate) fn edge_args(&self, inst: Inst, k: usize) -> &[Val] {
        self.vl(self.edges(inst)[k].args)
    }

    #[inline]
    pub(crate) fn edge_args_mut(&mut self, inst: Inst, k: usize) -> &mut [Val] {
        let l = self.edges(inst)[k].args;
        self.vl_mut(l)
    }

    pub(crate) fn push_edge_arg(&mut self, inst: Inst, k: usize, v: Val) {
        let e = self.insts[inst.idx()].e0 as usize + k;
        list_push(&mut self.vpool, &mut self.edges[e].args, v);
    }

    pub(crate) fn set_edge_args(&mut self, inst: Inst, k: usize, args: &[Val]) {
        let e = self.insts[inst.idx()].e0 as usize + k;
        self.edges[e].args = list_new(&mut self.vpool, args);
    }

    // --- snapshots --------------------------------------------------------------

    /// A snapshot of the entries pushed to `snap_pool` since `from`; the
    /// previous one when it is equal, as most guards of one instruction
    /// share their snapshot.
    pub(crate) fn finish_snap(&mut self, pc: u32, kind: ExitKind, from: usize) -> Snap {
        let entries = List {
            start: from as u32,
            len: (self.snap_pool.len() - from) as u32,
        };
        if let Some(last) = self.snaps.last()
            && last.pc == pc
            && last.kind == kind
            && self.snap_pool[last.entries.range()] == self.snap_pool[entries.range()]
        {
            self.snap_pool.truncate(from);
            return Snap(self.snaps.len() as u32 - 1);
        }
        self.snaps.push(SnapData { pc, kind, entries });
        Snap(self.snaps.len() as u32 - 1)
    }

    /// A new snapshot with `s`'s frame and its values mapped by `map`.
    pub(crate) fn map_snap(&mut self, s: Snap, mut map: impl FnMut(Val) -> Val) -> Snap {
        let d = self.snaps[s.idx()];
        let start = self.snap_pool.len();
        for k in d.entries.range() {
            let (r, v) = self.snap_pool[k];
            self.snap_pool.push((r, map(v)));
        }
        self.snaps.push(SnapData {
            pc: d.pc,
            kind: d.kind,
            entries: List {
                start: start as u32,
                len: d.entries.len,
            },
        });
        Snap(self.snaps.len() as u32 - 1)
    }

    #[inline]
    pub(crate) fn entries(&self, s: u32) -> &[(u8, Val)] {
        &self.snap_pool[self.snaps[s as usize].entries.range()]
    }

    /// The snapshot entries of `inst`, if it has a snapshot.
    #[inline]
    pub(crate) fn snap_entries(&self, inst: Inst) -> &[(u8, Val)] {
        match self.insts[inst.idx()].snap {
            NO_SNAP => &[],
            s => self.entries(s),
        }
    }

    // --- uses -------------------------------------------------------------------

    /// Make `inst` use `r` where it uses `v`; a snapshot it may share is
    /// copied first.
    pub(crate) fn replace_uses(&mut self, inst: Inst, v: Val, r: Val) {
        for a in self.args_mut(inst) {
            if *a == v {
                *a = r;
            }
        }
        for e in 0..self.edges(inst).len() {
            for a in self.edge_args_mut(inst, e) {
                if *a == v {
                    *a = r;
                }
            }
        }
        if self.snap_entries(inst).iter().any(|e| e.1 == v) {
            let s = self.insts[inst.idx()].snap;
            let ns = self.map_snap(Snap(s), |x| if x == v { r } else { x });
            self.insts[inst.idx()].snap = ns.0;
        }
    }

    /// Every value `inst` uses: operands, edge arguments, snapshot entries.
    pub(crate) fn for_each_use(&self, inst: Inst, mut f: impl FnMut(Val)) {
        for &a in self.args(inst) {
            f(a);
        }
        for e in self.edges(inst) {
            for &a in self.vl(e.args) {
                f(a);
            }
        }
        for &(_, v) in self.snap_entries(inst) {
            f(v);
        }
    }

    /// Use counts of every value: operands, edge arguments and snapshot
    /// entries of live instructions.
    pub(crate) fn use_counts(&self) -> Vec<u32> {
        let mut uses = vec![0u32; self.vals.len()];
        for (b, bd) in self.blocks.iter().enumerate() {
            if bd.dead {
                continue;
            }
            for &i in self.insts_of(Block(b as u32)) {
                self.for_each_use(i, |v| uses[v.idx()] += 1);
            }
        }
        uses
    }

    /// Replace uses through a map from each value to its replacement (or
    /// itself), resolving chains.
    pub(crate) fn apply_replacements(&mut self, map: &mut [Val]) {
        fn find(map: &mut [Val], v: Val) -> Val {
            let mut r = v;
            while map[r.idx()] != r {
                r = map[r.idx()];
            }
            let mut c = v;
            while map[c.idx()] != r {
                let n = map[c.idx()];
                map[c.idx()] = r;
                c = n;
            }
            r
        }
        for v in &mut self.vpool {
            if v.idx() < map.len() {
                *v = find(map, *v);
            }
        }
        for e in &mut self.snap_pool {
            if e.1.idx() < map.len() {
                e.1 = find(map, e.1);
            }
        }
    }
}
