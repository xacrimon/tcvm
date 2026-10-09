//! The region IR: a CFG of blocks with parameters, instructions in SSA form
//! whose operands and results live in flat pools, snapshots in a side table,
//! and a constant pool of the heap values the code embeds.

pub(crate) mod ops;
pub(crate) mod print;
pub(crate) mod types;
pub(crate) mod verify;

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
    /// Operands: `args[a0..a0 + an]`.
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

/// A branch target with the arguments for its parameters.
#[derive(Clone, Debug)]
pub(crate) struct BlockCall {
    pub(crate) target: Block,
    pub(crate) args: Vec<Val>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BlockData {
    pub(crate) params: Vec<Val>,
    pub(crate) insts: Vec<Inst>,
    /// The resume block of a call: entered by a callee's return, not by a
    /// branch.
    pub(crate) resume: bool,
    /// Deleted by a pass; skipped by everything.
    pub(crate) dead: bool,
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SnapData {
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    /// `(register, value)`, by register.
    pub(crate) entries: Vec<(u8, Val)>,
}

/// What a region was compiled from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegionMeta {
    pub(crate) entry_pc: u32,
    pub(crate) loop_entry: bool,
    pub(crate) max_stack: u8,
}

pub(crate) struct Func<'gc> {
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) insts: Vec<InstData>,
    pub(crate) vals: Vec<ValData>,
    pub(crate) args: Vec<Val>,
    pub(crate) edges: Vec<BlockCall>,
    pub(crate) snaps: Vec<SnapData>,
    pub(crate) pool: Vec<Value<'gc>>,
    pub(crate) entry: Block,
    pub(crate) meta: RegionMeta,
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
            args: Vec::new(),
            edges: Vec::new(),
            snaps: Vec::new(),
            pool: Vec::new(),
            entry: Block(0),
            meta,
        }
    }

    pub(crate) fn new_block(&mut self) -> Block {
        self.blocks.push(BlockData::default());
        Block(self.blocks.len() as u32 - 1)
    }

    pub(crate) fn add_param(&mut self, b: Block, ty: Ty) -> Val {
        let k = self.blocks[b.idx()].params.len() as u16;
        let v = Val(self.vals.len() as u32);
        self.vals.push(ValData {
            ty,
            def: ValDef::Param(b, k),
        });
        self.blocks[b.idx()].params.push(v);
        v
    }

    /// The pool index of a heap value.
    pub(crate) fn intern(&mut self, v: Value<'gc>) -> u32 {
        if let Some(i) = self.pool.iter().position(|p| p.same_bits(&v)) {
            return i as u32;
        }
        self.pool.push(v);
        self.pool.len() as u32 - 1
    }

    pub(crate) fn add_snap(&mut self, s: SnapData) -> Snap {
        // Most guards of one instruction share their snapshot.
        if let Some(last) = self.snaps.last()
            && *last == s
        {
            return Snap(self.snaps.len() as u32 - 1);
        }
        self.snaps.push(s);
        Snap(self.snaps.len() as u32 - 1)
    }

    /// Create an instruction (not placed in a block).
    pub(crate) fn make_inst(
        &mut self,
        op: Op,
        args: &[Val],
        snap: Option<Snap>,
        tag: ExitTag,
    ) -> Inst {
        let inst = Inst(self.insts.len() as u32);
        let a0 = self.args.len() as u32;
        self.args.extend_from_slice(args);
        let nres = op.num_results();
        let r0 = self.vals.len() as u32;
        let arg_tys: smallvec_ty::Tys = args.iter().map(|&a| self.vals[a.idx()].ty).collect();
        for k in 0..nres {
            let ty = op.result_ty(&arg_tys);
            self.vals.push(ValData {
                ty,
                def: ValDef::Inst(inst, k as u8),
            });
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

    pub(crate) fn set_edges(&mut self, inst: Inst, edges: Vec<BlockCall>) {
        let e0 = self.edges.len() as u32;
        let en = edges.len() as u8;
        self.edges.extend(edges);
        let d = &mut self.insts[inst.idx()];
        d.e0 = e0;
        d.en = en;
    }

    pub(crate) fn append(&mut self, b: Block, inst: Inst) {
        self.insts[inst.idx()].block = b;
        self.blocks[b.idx()].insts.push(inst);
    }

    /// Insert `inst` into `b` before its terminator.
    pub(crate) fn insert_before_term(&mut self, b: Block, inst: Inst) {
        self.insts[inst.idx()].block = b;
        let insts = &mut self.blocks[b.idx()].insts;
        let at = match insts.last() {
            Some(&last) if self.insts[last.idx()].op.is_terminator() => insts.len() - 1,
            _ => insts.len(),
        };
        insts.insert(at, inst);
    }

    #[inline]
    pub(crate) fn args(&self, inst: Inst) -> &[Val] {
        let d = &self.insts[inst.idx()];
        &self.args[d.a0 as usize..d.a0 as usize + d.an as usize]
    }

    #[inline]
    pub(crate) fn args_mut(&mut self, inst: Inst) -> &mut [Val] {
        let d = self.insts[inst.idx()];
        &mut self.args[d.a0 as usize..d.a0 as usize + d.an as usize]
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

    #[inline]
    pub(crate) fn edges_mut(&mut self, inst: Inst) -> &mut [BlockCall] {
        let d = self.insts[inst.idx()];
        &mut self.edges[d.e0 as usize..d.e0 as usize + d.en as usize]
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

    pub(crate) fn terminator(&self, b: Block) -> Option<Inst> {
        let &last = self.blocks[b.idx()].insts.last()?;
        self.op(last).is_terminator().then_some(last)
    }

    pub(crate) fn succs(&self, b: Block) -> impl Iterator<Item = Block> + '_ {
        self.terminator(b)
            .into_iter()
            .flat_map(move |t| self.edges(t).iter().map(|e| e.target))
    }

    /// Predecessor blocks of each block, by edge (a block appears once per
    /// edge to the target).
    pub(crate) fn preds(&self) -> Vec<Vec<Block>> {
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for (b, bd) in self.blocks.iter().enumerate() {
            if bd.dead {
                continue;
            }
            for s in self.succs(Block(b as u32)) {
                preds[s.idx()].push(Block(b as u32));
            }
        }
        preds
    }

    /// Blocks in reverse post-order from the entry.
    pub(crate) fn rpo(&self) -> Vec<Block> {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(Block, usize)> = vec![(self.entry, 0)];
        seen[self.entry.idx()] = true;
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            let succs: smallvec_ty::Blocks = self.succs(b).collect();
            if *i < succs.len() {
                let s = succs[*i];
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

    /// Use counts of every value: operands, edge arguments and snapshot
    /// entries of live instructions.
    pub(crate) fn use_counts(&self) -> Vec<u32> {
        let mut uses = vec![0u32; self.vals.len()];
        for bd in &self.blocks {
            if bd.dead {
                continue;
            }
            for &i in &bd.insts {
                for &a in self.args(i) {
                    uses[a.idx()] += 1;
                }
                for e in self.edges(i) {
                    for &a in &e.args {
                        uses[a.idx()] += 1;
                    }
                }
                let s = self.insts[i.idx()].snap;
                if s != NO_SNAP {
                    for &(_, v) in &self.snaps[s as usize].entries {
                        uses[v.idx()] += 1;
                    }
                }
            }
        }
        uses
    }

    /// Replace every use of `from` with `to`.
    pub(crate) fn replace_uses(&mut self, from: Val, to: Val) {
        for a in &mut self.args {
            if *a == from {
                *a = to;
            }
        }
        for e in &mut self.edges {
            for a in &mut e.args {
                if *a == from {
                    *a = to;
                }
            }
        }
        for s in &mut self.snaps {
            for (_, v) in &mut s.entries {
                if *v == from {
                    *v = to;
                }
            }
        }
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
        for i in 0..self.args.len() {
            self.args[i] = find(map, self.args[i]);
        }
        for e in &mut self.edges {
            for a in &mut e.args {
                *a = find(map, *a);
            }
        }
        for s in &mut self.snaps {
            for (_, v) in &mut s.entries {
                *v = find(map, *v);
            }
        }
    }
}

/// Small fixed-capacity collections, to keep hot paths allocation-free.
pub(crate) mod smallvec_ty {
    pub(crate) type Tys = Vec<super::Ty>;
    pub(crate) type Blocks = Vec<super::Block>;
}
