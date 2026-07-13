//! The IR verifier.
//!
//! Every invariant a later pass is allowed to *assume* is checked here, so that
//! a broken pass fails at the point of breakage rather than as wrong machine
//! code three stages downstream. Run it after the frontend and after every
//! optimization.
//!
//! Four families, in order — each depends on the previous one holding, so the
//! run aborts as soon as a family reports:
//!
//!  1. **Referential.** Every index is in range and every value is defined
//!     exactly once. Nothing below can even be evaluated until this holds; a
//!     dangling `Val` would panic the type lookups.
//!  2. **Structural.** One terminator, last, with the right target count; every
//!     block reachable; edge arity and edge types match the target's parameters.
//!  3. **Typing and metadata.** Operand and result representations per op, guard
//!     results being genuine refinements of their operand, and the `FrameState` /
//!     `Exit` presence rules. Plus SSA dominance: every use dominated by its def.
//!  4. **The rooting rule.** At every `MAY_GC` op, each live value that holds a
//!     `Gc` pointer must be anchored — see [`Effects`](super::op::Effects). This
//!     is the one the backend cannot recover from silently: an unanchored table
//!     across an allocation is a use-after-free that shows up as corruption in
//!     unrelated code.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::jit::ir::op::{ArithKind, Flags, FloatOp, IntOp, Op};
use crate::jit::ir::ty::{Refine, Rep, Ty, TypeSet};
use crate::jit::ir::{Block, Def, FsRef, Func, Inst, Val};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub block: Option<Block>,
    pub inst: Option<Inst>,
    pub msg: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.block, self.inst) {
            (Some(b), Some(i)) => write!(f, "block{} inst{}: {}", b.0, i.0, self.msg),
            (Some(b), None) => write!(f, "block{}: {}", b.0, self.msg),
            _ => write!(f, "{}", self.msg),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Errors(pub Vec<Error>);

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.0 {
            writeln!(f, "{e}")?;
        }
        Ok(())
    }
}

pub fn verify(func: &Func<'_>) -> Result<(), Errors> {
    let mut v = Verifier::new(func);

    v.check_references();
    if v.errs.is_empty() {
        v.index();
        v.check_structure();
    }
    if v.errs.is_empty() {
        v.compute_dominators();
        v.check_insts();
    }
    if v.errs.is_empty() {
        v.check_rooting();
    }

    if v.errs.is_empty() {
        Ok(())
    } else {
        Err(Errors(v.errs))
    }
}

struct Verifier<'a, 'gc> {
    f: &'a Func<'gc>,
    errs: Vec<Error>,
    at: (Option<Block>, Option<Inst>),
    /// Where each instruction sits: its block and its position within it.
    pos: HashMap<Inst, (Block, usize)>,
    preds: Vec<Vec<Block>>,
    /// Reverse postorder from the entry. Unreachable blocks are absent.
    rpo: Vec<Block>,
    rpo_num: Vec<Option<usize>>,
    idom: Vec<Option<Block>>,
}

impl<'a, 'gc> Verifier<'a, 'gc> {
    fn new(f: &'a Func<'gc>) -> Self {
        let n = f.num_blocks();
        Verifier {
            f,
            errs: Vec::new(),
            at: (None, None),
            pos: HashMap::new(),
            preds: vec![Vec::new(); n],
            rpo: Vec::new(),
            rpo_num: vec![None; n],
            idom: vec![None; n],
        }
    }

    fn err(&mut self, msg: impl Into<String>) {
        self.errs.push(Error {
            block: self.at.0,
            inst: self.at.1,
            msg: msg.into(),
        });
    }

    fn check(&mut self, cond: bool, msg: impl Into<String>) {
        if !cond {
            self.err(msg);
        }
    }

    // -- 1. referential integrity -------------------------------------------

    /// Nothing else may run until this passes: `Func::ty` and friends index
    /// directly, so a stale `Val` is a panic rather than a diagnostic.
    fn check_references(&mut self) {
        let (nb, ni, nv) = (self.f.num_blocks(), self.f.num_insts(), self.f.num_values());
        let (nfs, nex) = (self.f.num_frame_states(), self.f.num_exits());

        self.at = (None, None);
        if self.f.entry.index() >= nb {
            self.err("entry block is out of range");
            return;
        }

        // An instruction must be listed by exactly one block.
        let mut owner: Vec<Option<Block>> = vec![None; ni];

        for b in self.f.blocks() {
            self.at = (Some(b), None);
            let data = self.f.block(b);

            for (n, &p) in data.params.iter().enumerate() {
                if p.index() >= nv {
                    self.err(format!("param {n} is out of range"));
                    continue;
                }
                if self.f.def(p) != Def::Param(b, n as u32) {
                    self.err(format!("param {n} (v{}) is not defined here", p.0));
                }
            }

            for &i in &data.insts {
                if i.index() >= ni {
                    self.err(format!("inst{} is out of range", i.0));
                    continue;
                }
                self.at = (Some(b), Some(i));
                if let Some(prev) = owner[i.index()] {
                    self.err(format!("also listed in block{}", prev.0));
                    continue;
                }
                owner[i.index()] = Some(b);

                let d = self.f.inst(i);
                for &v in d.args.iter().chain(&d.results) {
                    if v.index() >= nv {
                        self.err(format!("v{} is out of range", v.0));
                    }
                }
                for (n, &r) in d.results.iter().enumerate() {
                    if r.index() < nv && self.f.def(r) != Def::Inst(i) {
                        self.err(format!("result {n} (v{}) is not defined here", r.0));
                    }
                }
                for t in &d.targets {
                    if t.block.index() >= nb {
                        self.err(format!("target block{} is out of range", t.block.0));
                    }
                    for &v in &t.args {
                        if v.index() >= nv {
                            self.err(format!("edge argument v{} is out of range", v.0));
                        }
                    }
                }
                if let Some(fs) = d.fs
                    && fs.index() >= nfs
                {
                    self.err(format!("fs{} is out of range", fs.0));
                }
                if let Some(e) = d.exit
                    && e.index() >= nex
                {
                    self.err(format!("exit{} is out of range", e.0));
                }
            }
        }

        self.at = (None, None);
        for r in (0..nfs).map(|n| FsRef(n as u32)) {
            let fs = self.f.frame_state(r);
            for v in fs.regs.iter().flatten() {
                if v.index() >= nv {
                    self.err(format!("fs{}: v{} is out of range", r.0, v.0));
                }
            }
            if let Some(p) = fs.parent
                && p.index() >= nfs
            {
                self.err(format!("fs{}: parent fs{} is out of range", r.0, p.0));
            }
        }
        for n in 0..nex {
            let e = self.f.exit(crate::jit::ir::ExitRef(n as u32));
            if e.fs.index() >= nfs {
                self.err(format!("exit{n}: fs{} is out of range", e.fs.0));
            }
        }
        self.at = (None, None);
    }

    fn index(&mut self) {
        for b in self.f.blocks() {
            for (n, &i) in self.f.block(b).insts.iter().enumerate() {
                self.pos.insert(i, (b, n));
            }
        }
    }

    // -- 2. structure --------------------------------------------------------

    fn check_structure(&mut self) {
        for b in self.f.blocks() {
            self.at = (Some(b), None);
            let insts = &self.f.block(b).insts;

            let Some((&last, rest)) = insts.split_last() else {
                self.err("block is empty");
                continue;
            };
            for &i in rest {
                if self.f.inst(i).op.is_terminator() {
                    self.at = (Some(b), Some(i));
                    self.err("terminator in the middle of a block");
                    self.at = (Some(b), None);
                }
            }

            let d = self.f.inst(last);
            self.at = (Some(b), Some(last));
            if !d.op.is_terminator() {
                self.err(format!("block does not end in a terminator ({:?})", d.op));
                continue;
            }

            let want = match d.op {
                Op::Jump => 1,
                Op::Br => 2,
                _ => 0,
            };
            if d.targets.len() != want {
                self.err(format!(
                    "{:?} wants {want} target(s), has {}",
                    d.op,
                    d.targets.len()
                ));
                continue;
            }

            for t in &d.targets {
                let params = &self.f.block(t.block).params;
                if t.args.len() != params.len() {
                    self.err(format!(
                        "edge to block{} passes {} argument(s), it takes {}",
                        t.block.0,
                        t.args.len(),
                        params.len()
                    ));
                    continue;
                }
                for (n, (&arg, &p)) in t.args.iter().zip(params).enumerate() {
                    let (have, want) = (self.f.ty(arg), self.f.ty(p));
                    if !want.accepts(have) {
                        self.err(format!(
                            "edge to block{}: argument {n} (v{}) is {have:?}, parameter v{} is {want:?}",
                            t.block.0, arg.0, p.0
                        ));
                    }
                }
            }
        }

        // Non-terminators must not carry targets, or a pass walking the CFG by
        // terminator alone would miss an edge.
        for b in self.f.blocks() {
            for &i in &self.f.block(b).insts {
                let d = self.f.inst(i);
                if !d.op.is_terminator() && !d.targets.is_empty() {
                    self.at = (Some(b), Some(i));
                    self.err("non-terminator carries branch targets");
                }
            }
        }

        self.at = (None, None);
        if self.errs.is_empty() {
            self.build_cfg();
        }
    }

    fn build_cfg(&mut self) {
        for b in self.f.blocks() {
            for &i in &self.f.block(b).insts {
                for t in &self.f.inst(i).targets {
                    self.preds[t.block.index()].push(b);
                }
            }
        }

        // Postorder DFS, then reverse.
        let mut seen = vec![false; self.f.num_blocks()];
        let mut post = Vec::new();
        let mut stack = vec![(self.f.entry, 0usize)];
        seen[self.f.entry.index()] = true;
        while let Some((b, n)) = stack.pop() {
            let succs = self.succs(b);
            if n < succs.len() {
                stack.push((b, n + 1));
                let s = succs[n];
                if !seen[s.index()] {
                    seen[s.index()] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
            }
        }
        post.reverse();
        self.rpo = post;
        for (n, &b) in self.rpo.iter().enumerate() {
            self.rpo_num[b.index()] = Some(n);
        }

        for b in self.f.blocks() {
            if self.rpo_num[b.index()].is_none() {
                self.at = (Some(b), None);
                self.err("block is unreachable from the entry");
            }
        }
        self.at = (None, None);
    }

    fn succs(&self, b: Block) -> Vec<Block> {
        self.f
            .block(b)
            .insts
            .last()
            .map(|&i| {
                self.f
                    .inst(i)
                    .targets
                    .iter()
                    .map(|t| t.block)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    /// Cooper-Harvey-Kennedy over the RPO. Unreachable blocks have no idom, and
    /// no check below consults them (they were already reported).
    fn compute_dominators(&mut self) {
        let entry = self.f.entry;
        self.idom[entry.index()] = Some(entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &b in self.rpo.clone().iter().skip(1) {
                let mut new: Option<Block> = None;
                for p in self.preds[b.index()].clone() {
                    if self.idom[p.index()].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(cur) => self.intersect(p, cur),
                    });
                }
                if new.is_some() && self.idom[b.index()] != new {
                    self.idom[b.index()] = new;
                    changed = true;
                }
            }
        }
    }

    fn intersect(&self, mut a: Block, mut b: Block) -> Block {
        while a != b {
            while self.rpo_num[a.index()] > self.rpo_num[b.index()] {
                a = self.idom[a.index()].expect("a processed block has an idom");
            }
            while self.rpo_num[b.index()] > self.rpo_num[a.index()] {
                b = self.idom[b.index()].expect("a processed block has an idom");
            }
        }
        a
    }

    fn dominates(&self, a: Block, b: Block) -> bool {
        let mut cur = b;
        loop {
            if cur == a {
                return true;
            }
            match self.idom[cur.index()] {
                Some(p) if p != cur => cur = p,
                _ => return false,
            }
        }
    }

    /// Is `v` available at instruction `at`, which sits at `at_pos` in `block`?
    fn available(&self, v: Val, block: Block, at_pos: usize) -> bool {
        match self.f.def(v) {
            Def::Param(b, _) => self.dominates(b, block),
            Def::Inst(i) => {
                let Some(&(db, dpos)) = self.pos.get(&i) else {
                    return false;
                };
                if db == block {
                    dpos < at_pos
                } else {
                    self.dominates(db, block)
                }
            }
        }
    }

    // -- 3. per-instruction typing, metadata, and dominance ------------------

    fn check_insts(&mut self) {
        for b in self.f.blocks() {
            if self.rpo_num[b.index()].is_none() {
                continue;
            }
            for (n, &i) in self.f.block(b).insts.iter().enumerate() {
                self.at = (Some(b), Some(i));
                self.check_dominance(i, b, n);
                self.check_metadata(i);
                self.check_types(i);
            }
        }
        self.at = (None, None);
    }

    fn check_dominance(&mut self, i: Inst, b: Block, n: usize) {
        let d = self.f.inst(i);
        let uses: Vec<Val> = d
            .args
            .iter()
            .copied()
            .chain(d.targets.iter().flat_map(|t| t.args.iter().copied()))
            .chain(
                d.fs.into_iter()
                    .flat_map(|fs| self.fs_values(fs).into_iter()),
            )
            .collect();
        for v in uses {
            if !self.available(v, b, n) {
                self.err(format!("v{} is used before its definition dominates", v.0));
            }
        }
    }

    fn check_metadata(&mut self, i: Inst) {
        let d = self.f.inst(i);
        let op = d.op;

        self.check(
            d.fs.is_some() == op.needs_frame_state(),
            format!(
                "{op:?} {} a FrameState",
                if op.needs_frame_state() {
                    "requires"
                } else {
                    "must not carry"
                }
            ),
        );

        let wants_exit = op.is_guard() || op == Op::Deopt;
        self.check(
            d.exit.is_some() == wants_exit,
            format!(
                "{op:?} {} an Exit",
                if wants_exit {
                    "requires"
                } else {
                    "must not carry"
                }
            ),
        );

        // An exit resumes the interpreter from the state captured at the
        // instruction that failed. Anything else and we would deopt into a frame
        // that never existed.
        if let (Some(fs), Some(e)) = (d.fs, d.exit)
            && self.f.exit(e).fs != fs
        {
            self.err(format!(
                "exit{} refers to fs{}, but the instruction carries fs{}",
                e.0,
                self.f.exit(e).fs.0,
                fs.0
            ));
        }

        // A frame state names Lua registers, and a Lua register holds a `Value`
        // or an unboxed number the deopt writer can re-pack. A raw pointer or a
        // condition flag has no `Value` encoding at all.
        if let Some(fs) = d.fs {
            for v in self.fs_values(fs) {
                if matches!(self.f.ty(v).rep, Rep::Ptr | Rep::B1) {
                    self.err(format!(
                        "fs{}: v{} is {:?}, which has no Value encoding to deopt into",
                        fs.0,
                        v.0,
                        self.f.ty(v).rep
                    ));
                }
            }
        }
    }

    fn check_types(&mut self, i: Inst) {
        let d = self.f.inst(i);
        let op = d.op;
        let args = d.args.clone();
        let res = d.results.clone();

        // Arity, then per-operand representation. `sig` covers the ops whose
        // operands are a fixed list of reps; the rest are checked by hand below.
        let val = Rep::Val;
        let sig: Option<(&[Rep], usize)> = match op {
            Op::KConst(_) | Op::IConst(_) | Op::FConst(_) | Op::BConst(_) => Some((&[], 1)),
            Op::PackInt => Some((&[Rep::I64], 1)),
            Op::PackFloat => Some((&[Rep::F64], 1)),
            Op::PackBool => Some((&[Rep::B1], 1)),
            Op::UnpackInt
            | Op::UnpackFloat
            | Op::UnpackPtr
            | Op::TagOf
            | Op::IsType(_)
            | Op::IsFalsy => Some((&[val], 1)),
            Op::GuardType(_) | Op::GuardShape(_) => Some((&[val], 1)),
            Op::GuardCond => Some((&[Rep::B1], 0)),
            Op::AssumeNoMm(..) => Some((&[], 0)),
            Op::LuaCmp(_) | Op::LuaEq | Op::LuaConcat | Op::LuaGetIndex => Some((&[val, val], 1)),
            Op::LuaLen => Some((&[val], 1)),
            Op::LuaSetIndex => Some((&[val, val, val], 0)),
            Op::ICmp(_) => Some((&[Rep::I64, Rep::I64], 1)),
            Op::FCmp(_) => Some((&[Rep::F64, Rep::F64], 1)),
            Op::SiToFp => Some((&[Rep::I64], 1)),
            Op::FpToIntExact => Some((&[Rep::F64], 1)),
            Op::TabNew { .. } | Op::ClosureNew(_) | Op::GetGlobal(_) => Some((&[], 1)),
            Op::TabProps | Op::TabArr | Op::TabArrLen => Some((&[val], 1)),
            Op::SlotGet(_) | Op::UpvalGet => Some((&[Rep::Ptr], 1)),
            Op::SlotSet(_) | Op::UpvalSet => Some((&[Rep::Ptr, val], 0)),
            Op::ArrGet => Some((&[Rep::Ptr, Rep::I64], 1)),
            Op::ArrSet => Some((&[Rep::Ptr, Rep::I64, val], 0)),
            Op::TabHashGet => Some((&[val, val], 1)),
            Op::GcBarrierBack | Op::SetGlobal(_) | Op::StackSet(_) => Some((&[val], 0)),
            Op::GcBarrierFwd => Some((&[val, val], 0)),
            Op::UpvalCell(_) | Op::StackGet(_) => Some((&[], 1)),
            Op::UpvalClose(_) | Op::Jump | Op::Deopt | Op::Safepoint => Some((&[], 0)),
            Op::Br => Some((&[Rep::B1], 0)),
            // Variadic in their operands: checked below.
            Op::LuaArith(_) | Op::IntArith(_) | Op::FloatArith(_) | Op::Call { .. } | Op::Ret => {
                None
            }
        };

        if let Some((reps, nres)) = sig {
            if args.len() != reps.len() {
                self.err(format!(
                    "{op:?} takes {} operand(s), has {}",
                    reps.len(),
                    args.len()
                ));
                return;
            }
            if res.len() != nres {
                self.err(format!(
                    "{op:?} produces {nres} result(s), has {}",
                    res.len()
                ));
                return;
            }
            for (n, (&a, &want)) in args.iter().zip(reps).enumerate() {
                self.want_rep(a, want, &format!("{op:?} operand {n}"));
            }
        }

        match op {
            Op::KConst(c) => {
                let want = TypeSet::of_value(self.f.pool.value(c));
                let got = self.f.ty(res[0]);
                self.want_rep(res[0], val, "kconst result");
                self.check(
                    got.set == want,
                    format!("kconst result is {:?}, the constant is {want:?}", got.set),
                );
            }
            Op::IConst(_) => self.want_rep(res[0], Rep::I64, "iconst result"),
            Op::FConst(_) => self.want_rep(res[0], Rep::F64, "fconst result"),
            Op::BConst(_) => self.want_rep(res[0], Rep::B1, "bconst result"),

            Op::PackInt => self.want_exact(res[0], Ty::boxed(TypeSet::INT), "pack.int result"),
            Op::PackFloat => {
                self.want_exact(res[0], Ty::boxed(TypeSet::FLOAT), "pack.float result")
            }
            Op::PackBool => self.want_exact(res[0], Ty::boxed(TypeSet::BOOL), "pack.bool result"),

            // The unpacks are *not* checks. A preceding guard must already have
            // proven the tag, or the payload read is garbage.
            Op::UnpackInt => {
                self.want_set(args[0], TypeSet::INT, "unpack.int operand");
                self.want_rep(res[0], Rep::I64, "unpack.int result");
            }
            Op::UnpackFloat => {
                self.want_set(args[0], TypeSet::FLOAT, "unpack.float operand");
                self.want_rep(res[0], Rep::F64, "unpack.float result");
            }
            Op::UnpackPtr => {
                self.want_set(args[0], TypeSet::HEAP, "unpack.ptr operand");
                self.want_rep(res[0], Rep::Ptr, "unpack.ptr result");
            }
            Op::TagOf => self.want_rep(res[0], Rep::I64, "tag.of result"),
            Op::IsType(_) | Op::IsFalsy => self.want_rep(res[0], Rep::B1, "result"),

            // A guard narrows; it never widens and never changes representation.
            // If this fails, some pass is laundering a type through a guard.
            Op::GuardType(set) => {
                let want = self.f.ty(args[0]).refined_to(set);
                self.want_exact(res[0], want, "guard.type result");
            }
            // Subsumes a type check: the operand need only *possibly* be a table,
            // and the backend tests the tag before dereferencing the shape.
            Op::GuardShape(s) => {
                let got = self.f.ty(args[0]).set;
                self.check(
                    got.intersects(TypeSet::TAB),
                    format!(
                        "guard.shape operand (v{}) is {got:?} and can never be a table",
                        args[0].0
                    ),
                );
                self.want_exact(res[0], Ty::with_shape(s), "guard.shape result");
            }

            Op::LuaArith(k) => {
                let n = if matches!(k, ArithKind::Unm | ArithKind::BNot) {
                    1
                } else {
                    2
                };
                self.want_arity(&args, n, op);
                self.want_results(&res, 1, op);
                for (j, &a) in args.iter().enumerate() {
                    self.want_rep(a, val, &format!("lua.arith operand {j}"));
                }
                if res.len() == 1 {
                    self.want_rep(res[0], val, "lua.arith result");
                }
            }
            Op::IntArith(k) => {
                let n = if matches!(k, IntOp::Neg | IntOp::BNot) {
                    1
                } else {
                    2
                };
                self.want_arity(&args, n, op);
                self.want_results(&res, 1, op);
                for (j, &a) in args.iter().enumerate() {
                    self.want_rep(a, Rep::I64, &format!("{op:?} operand {j}"));
                }
                if res.len() == 1 {
                    self.want_rep(res[0], Rep::I64, "int arith result");
                }
            }
            Op::FloatArith(k) => {
                let n = if matches!(k, FloatOp::Neg) { 1 } else { 2 };
                self.want_arity(&args, n, op);
                self.want_results(&res, 1, op);
                for (j, &a) in args.iter().enumerate() {
                    self.want_rep(a, Rep::F64, &format!("{op:?} operand {j}"));
                }
                if res.len() == 1 {
                    self.want_rep(res[0], Rep::F64, "float arith result");
                }
            }
            Op::ICmp(_) | Op::FCmp(_) => self.want_rep(res[0], Rep::B1, "compare result"),
            Op::SiToFp => self.want_rep(res[0], Rep::F64, "sitofp result"),
            Op::FpToIntExact => self.want_rep(res[0], Rep::I64, "fp_to_int_exact result"),

            // `slot.get`'s displacement is only meaningful under a proven layout,
            // and the layout is proven by the shape refinement on this operand.
            // Without it the load reads whatever happens to be at that offset.
            Op::TabProps => {
                self.want_set(args[0], TypeSet::TAB, "tab.props operand");
                self.check(
                    self.f.ty(args[0]).shape().is_some(),
                    "tab.props operand is not shape-refined; the slot displacement is unproven"
                        .to_string(),
                );
                self.want_rep(res[0], Rep::Ptr, "tab.props result");
            }
            Op::TabArr => {
                self.want_set(args[0], TypeSet::TAB, "tab.arr operand");
                self.want_rep(res[0], Rep::Ptr, "tab.arr result");
            }
            Op::TabArrLen => {
                self.want_set(args[0], TypeSet::TAB, "tab.arr_len operand");
                self.want_rep(res[0], Rep::I64, "tab.arr_len result");
            }
            Op::TabHashGet => self.want_set(args[0], TypeSet::TAB, "tab.hash_get operand"),
            Op::TabNew { .. } => self.want_exact(res[0], Ty::boxed(TypeSet::TAB), "tab.new result"),
            Op::SlotGet(_) | Op::ArrGet | Op::UpvalGet | Op::GetGlobal(_) => {
                self.want_rep(res[0], val, "load result")
            }

            Op::ClosureNew(p) => self.want_exact(
                res[0],
                Ty {
                    rep: val,
                    set: TypeSet::FUN,
                    refine: Refine::Proto(p),
                },
                "closure.new result",
            ),

            // The whole point of pinning: an open upvalue names a stack slot by
            // index, so a register it can observe must not be routed through SSA.
            // A stack access to an *unpinned* register means some pass has
            // confused the two homes.
            Op::StackGet(r) | Op::StackSet(r) => {
                self.check(
                    self.f.pinned_regs.contains(&r),
                    format!("stack access to r{r}, which is not pinned"),
                );
                if let Op::StackGet(_) = op {
                    self.want_rep(res[0], val, "stack.get result");
                }
            }

            Op::Call { nret } => {
                self.check(!args.is_empty(), "call has no callee");
                for (j, &a) in args.iter().enumerate() {
                    self.want_rep(a, val, &format!("call operand {j}"));
                }
                self.want_results(&res, nret as usize, op);
                for &r in &res {
                    self.want_rep(r, val, "call result");
                }
            }
            Op::Ret => {
                self.want_results(&res, 0, op);
                for (j, &a) in args.iter().enumerate() {
                    self.want_rep(a, val, &format!("ret operand {j}"));
                }
            }

            _ => {}
        }
    }

    fn want_arity(&mut self, args: &[Val], n: usize, op: Op) {
        self.check(
            args.len() == n,
            format!("{op:?} takes {n} operand(s), has {}", args.len()),
        );
    }

    fn want_results(&mut self, res: &[Val], n: usize, op: Op) {
        self.check(
            res.len() == n,
            format!("{op:?} produces {n} result(s), has {}", res.len()),
        );
    }

    fn want_rep(&mut self, v: Val, rep: Rep, what: &str) {
        let got = self.f.ty(v).rep;
        self.check(
            got == rep,
            format!("{what} (v{}) is {got:?}, wanted {rep:?}", v.0),
        );
    }

    fn want_set(&mut self, v: Val, set: TypeSet, what: &str) {
        let got = self.f.ty(v).set;
        self.check(
            set.contains(got),
            format!("{what} (v{}) is {got:?}, wanted a subset of {set:?}", v.0),
        );
    }

    fn want_exact(&mut self, v: Val, ty: Ty, what: &str) {
        let got = self.f.ty(v);
        self.check(
            got == ty,
            format!("{what} (v{}) is {got:?}, wanted {ty:?}", v.0),
        );
    }

    // -- 4. the rooting rule -------------------------------------------------

    fn fs_values(&self, fs: FsRef) -> Vec<Val> {
        let mut out = Vec::new();
        let mut cur = Some(fs);
        while let Some(r) = cur {
            let state = self.f.frame_state(r);
            out.extend(state.regs.iter().flatten().copied());
            cur = state.parent;
        }
        out
    }

    /// Values needing a root: anything whose payload the collector must see.
    /// A packed integer has rep `Val` but no pointer in it, so it does not.
    fn needs_root(t: Ty) -> bool {
        match t.rep {
            Rep::Ptr => true,
            Rep::Val => t.set.intersects(TypeSet::HEAP),
            _ => false,
        }
    }

    /// Is `v` reachable by the collector across a GC point?
    ///
    /// Either it is a Lua register named by the frame state — the deopt writer
    /// spills those to their canonical stack slots, where `ThreadState::trace`
    /// finds them — or it is *derived* from something that is. Derived pointers
    /// are sound to leave unrooted only because the collector never moves
    /// objects; the base keeps the allocation alive and the interior pointer
    /// stays valid.
    fn anchored(&self, v: Val, roots: &HashSet<Val>) -> bool {
        let mut cur = v;
        // A use-def walk in an already-dominance-checked SSA graph cannot cycle;
        // the bound is belt and braces.
        for _ in 0..64 {
            if roots.contains(&cur) {
                return true;
            }
            let Def::Inst(i) = self.f.def(cur) else {
                return false;
            };
            let d = self.f.inst(i);
            match d.op {
                // The constant pool traces its values, and the running closure
                // traces its upvalue cells. Both outlive any GC point here.
                Op::KConst(_) | Op::UpvalCell(_) => return true,
                // Interior pointers and refinements: same object, so ask the base.
                Op::TabProps
                | Op::TabArr
                | Op::UnpackPtr
                | Op::GuardType(_)
                | Op::GuardShape(_) => {
                    cur = d.args[0];
                }
                _ => return false,
            }
        }
        false
    }

    fn check_rooting(&mut self) {
        let live_in = self.liveness();

        for b in self.f.blocks() {
            if self.rpo_num[b.index()].is_none() {
                continue;
            }
            let mut live: HashSet<Val> = self
                .succs(b)
                .iter()
                .flat_map(|s| live_in[s.index()].iter().copied())
                .collect();

            for &i in self.f.block(b).insts.iter().rev() {
                let d = self.f.inst(i);

                // Kill the definitions first: a result is produced *after* the
                // collector ran, so it is not live across its own op.
                for r in &d.results {
                    live.remove(r);
                }

                if d.op.effects().flags.contains(Flags::MAY_GC) {
                    let fs = d.fs.expect("a MAY_GC op carries a FrameState");
                    let roots: HashSet<Val> = self.fs_values(fs).into_iter().collect();
                    // What survives the op, plus what the op holds while it runs.
                    let mut bad: Vec<Val> = live
                        .iter()
                        .copied()
                        .chain(d.args.iter().copied())
                        .filter(|&v| Self::needs_root(self.f.ty(v)))
                        .filter(|&v| !self.anchored(v, &roots))
                        .collect();
                    bad.sort();
                    bad.dedup();
                    for v in bad {
                        self.at = (Some(b), Some(i));
                        self.err(format!(
                            "v{} ({:?}) is live across a may-gc op but is not anchored by fs{}",
                            v.0,
                            self.f.ty(v),
                            fs.0
                        ));
                    }
                }

                for v in self.inst_uses(i) {
                    live.insert(v);
                }
            }
        }
        self.at = (None, None);
    }

    fn inst_uses(&self, i: Inst) -> Vec<Val> {
        let d = self.f.inst(i);
        d.args
            .iter()
            .copied()
            .chain(d.targets.iter().flat_map(|t| t.args.iter().copied()))
            .chain(d.fs.into_iter().flat_map(|fs| self.fs_values(fs)))
            .collect()
    }

    /// Backward liveness to a fixpoint. Block parameters are definitions, so a
    /// value handed along an edge is a use in the *predecessor's* terminator and
    /// does not escape into the successor's live-in.
    fn liveness(&self) -> Vec<HashSet<Val>> {
        let mut live_in: Vec<HashSet<Val>> = vec![HashSet::new(); self.f.num_blocks()];
        let mut changed = true;
        while changed {
            changed = false;
            for &b in self.rpo.iter().rev() {
                let mut live: HashSet<Val> = self
                    .succs(b)
                    .iter()
                    .flat_map(|s| live_in[s.index()].iter().copied())
                    .collect();
                for &i in self.f.block(b).insts.iter().rev() {
                    for r in &self.f.inst(i).results {
                        live.remove(r);
                    }
                    live.extend(self.inst_uses(i));
                }
                for p in &self.f.block(b).params {
                    live.remove(p);
                }
                if live != live_in[b.index()] {
                    live_in[b.index()] = live;
                    changed = true;
                }
            }
        }
        live_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::shape::MetamethodBits;
    use crate::jit::ir::pool::ShapeRef;
    use crate::jit::ir::ty::TypeContext;
    use crate::jit::ir::{BlockCall, Exit, FrameState, InstData};

    fn inst(op: Op, args: Vec<Val>) -> InstData {
        InstData {
            op,
            args,
            targets: Vec::new(),
            results: Vec::new(),
            fs: None,
            exit: None,
        }
    }

    fn errors(f: &Func<'_>) -> String {
        verify(f)
            .expect_err("expected verification to fail")
            .to_string()
    }

    /// `t.x + 1` under a shape guard, in one block. Valid by construction; each
    /// negative test below breaks exactly one thing about it.
    ///
    /// Instruction indices, which the tests reach for directly:
    /// 0 guard.shape, 1 assume.no_mm, 2 tab.props, 3 slot.get, 4 guard.type,
    /// 5 unpack.int, 6 iconst, 7 add.i64, 8 pack.int, 9 ret.
    fn shaped_add() -> Func<'static> {
        let mut f = Func::new();
        let b = f.entry;
        let s0 = ShapeRef(0);
        let t = f.append_param(b, Ty::boxed(TypeSet::TAB));

        let fs = f.add_frame_state(FrameState {
            pc: 0,
            regs: vec![Some(t)],
            parent: None,
        });
        let exit = f.add_exit(Exit {
            fs,
            ctx: TypeContext::default(),
            count: 0,
        });
        let mut g = inst(Op::GuardShape(s0), vec![t]);
        g.fs = Some(fs);
        g.exit = Some(exit);
        let (_, r) = f.append_inst(b, g, &[Ty::with_shape(s0)]);
        let t1 = r[0];

        f.append_inst(
            b,
            inst(Op::AssumeNoMm(s0, MetamethodBits::INDEX), vec![]),
            &[],
        );
        let (_, r) = f.append_inst(b, inst(Op::TabProps, vec![t1]), &[Ty::PTR]);
        let props = r[0];
        let (_, r) = f.append_inst(b, inst(Op::SlotGet(0), vec![props]), &[Ty::ANY]);
        let x = r[0];

        let fs2 = f.add_frame_state(FrameState {
            pc: 1,
            regs: vec![Some(t1)],
            parent: None,
        });
        let e2 = f.add_exit(Exit {
            fs: fs2,
            ctx: TypeContext::default(),
            count: 0,
        });
        let mut gt = inst(Op::GuardType(TypeSet::INT), vec![x]);
        gt.fs = Some(fs2);
        gt.exit = Some(e2);
        let (_, r) = f.append_inst(b, gt, &[Ty::ANY.refined_to(TypeSet::INT)]);
        let xi = r[0];

        let (_, r) = f.append_inst(b, inst(Op::UnpackInt, vec![xi]), &[Ty::I64]);
        let a = r[0];
        let (_, r) = f.append_inst(b, inst(Op::IConst(1), vec![]), &[Ty::I64]);
        let one = r[0];
        let (_, r) = f.append_inst(b, inst(Op::IntArith(IntOp::Add), vec![a, one]), &[Ty::I64]);
        let sum = r[0];
        let (_, r) = f.append_inst(b, inst(Op::PackInt, vec![sum]), &[Ty::boxed(TypeSet::INT)]);
        f.append_inst(b, inst(Op::Ret, vec![r[0]]), &[]);
        f
    }

    /// An edge that packs an `i64` into a boxed parameter.
    fn two_block() -> Func<'static> {
        let mut f = Func::new();
        let b0 = f.entry;
        let a = f.append_param(b0, Ty::I64);
        let b1 = f.new_block();
        let p = f.append_param(b1, Ty::boxed(TypeSet::INT));

        let (_, r) = f.append_inst(b0, inst(Op::PackInt, vec![a]), &[Ty::boxed(TypeSet::INT)]);
        let mut j = inst(Op::Jump, vec![]);
        j.targets = vec![BlockCall {
            block: b1,
            args: vec![r[0]],
        }];
        f.append_inst(b0, j, &[]);
        f.append_inst(b1, inst(Op::Ret, vec![p]), &[]);
        f
    }

    /// Two allocations, with the first table still live across the second. The
    /// only thing keeping it alive is the second `tab.new`'s frame state.
    fn two_allocs(rooted: bool) -> Func<'static> {
        let mut f = Func::new();
        let b = f.entry;

        let fs0 = f.add_frame_state(FrameState {
            pc: 0,
            regs: vec![],
            parent: None,
        });
        let mut a0 = inst(Op::TabNew { array_hint: 0 }, vec![]);
        a0.fs = Some(fs0);
        let (_, r) = f.append_inst(b, a0, &[Ty::boxed(TypeSet::TAB)]);
        let t0 = r[0];

        let fs1 = f.add_frame_state(FrameState {
            pc: 1,
            regs: if rooted { vec![Some(t0)] } else { vec![] },
            parent: None,
        });
        let mut a1 = inst(Op::TabNew { array_hint: 0 }, vec![]);
        a1.fs = Some(fs1);
        let (_, r) = f.append_inst(b, a1, &[Ty::boxed(TypeSet::TAB)]);
        let t1 = r[0];

        f.append_inst(b, inst(Op::Ret, vec![t0, t1]), &[]);
        f
    }

    #[test]
    fn accepts_well_formed_ir() {
        verify(&shaped_add()).expect("valid");
        verify(&two_block()).expect("valid");
    }

    #[test]
    fn rejects_missing_terminator() {
        let mut f = shaped_add();
        f.block_mut(Block(0)).insts.pop();
        assert!(errors(&f).contains("does not end in a terminator"));
    }

    #[test]
    fn rejects_unreachable_block() {
        let mut f = shaped_add();
        let b = f.new_block();
        f.append_inst(b, inst(Op::Ret, vec![]), &[]);
        assert!(errors(&f).contains("unreachable"));
    }

    #[test]
    fn rejects_use_before_def() {
        let mut f = shaped_add();
        // `add.i64` now precedes the `iconst` it consumes.
        f.block_mut(Block(0)).insts.swap(6, 7);
        assert!(errors(&f).contains("used before its definition dominates"));
    }

    /// The unpacks are unchecked payload reads. Feed `unpack.int` the raw
    /// `slot.get` result instead of the guarded one and the tag is unproven.
    #[test]
    fn rejects_unpack_of_unproven_tag() {
        let mut f = shaped_add();
        let raw = f.inst(Inst(3)).results[0];
        f.inst_mut(Inst(5)).args[0] = raw;
        assert!(errors(&f).contains("unpack.int operand"));
    }

    /// A constant slot displacement is meaningless without a proven layout.
    #[test]
    fn rejects_slot_access_without_shape() {
        let mut f = shaped_add();
        let unguarded = f.block(Block(0)).params[0];
        f.inst_mut(Inst(2)).args[0] = unguarded;
        assert!(errors(&f).contains("not shape-refined"));
    }

    #[test]
    fn rejects_guard_without_exit() {
        let mut f = shaped_add();
        f.inst_mut(Inst(0)).exit = None;
        assert!(errors(&f).contains("requires an Exit"));
    }

    /// A guard may only narrow. Widening one launders a type the code never
    /// proved.
    #[test]
    fn rejects_guard_that_widens() {
        let mut f = shaped_add();
        let r = f.inst(Inst(4)).results[0];
        f.set_ty(r, Ty::ANY);
        assert!(errors(&f).contains("guard.type result"));
    }

    #[test]
    fn rejects_edge_arity_mismatch() {
        let mut f = two_block();
        f.inst_mut(Inst(1)).targets[0].args.clear();
        assert!(errors(&f).contains("it takes 1"));
    }

    /// The edge check is what stops a boxed parameter from silently swallowing a
    /// raw `i64` — which is how a specialized loop accumulator gets re-tagged.
    #[test]
    fn rejects_edge_type_mismatch() {
        let mut f = two_block();
        let raw = f.block(Block(0)).params[0];
        f.inst_mut(Inst(1)).targets[0].args[0] = raw;
        assert!(errors(&f).contains("parameter"));
    }

    #[test]
    fn accepts_value_rooted_across_gc() {
        verify(&two_allocs(true)).expect("valid");
    }

    #[test]
    fn rejects_unrooted_value_across_gc() {
        assert!(errors(&two_allocs(false)).contains("not anchored"));
    }
}
