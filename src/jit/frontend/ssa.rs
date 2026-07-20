//! On-the-fly SSA construction, after Braun et al. (CC 2013).
//!
//! Block parameters are discovered by *reading* variables rather than placed
//! from liveness. `read_var` looks the variable up locally; failing that it walks
//! predecessors, minting a parameter only where the incoming values genuinely
//! disagree. The result is pruned SSA for any program and minimal SSA for
//! reducible control flow, with no dominance frontiers and no liveness analysis —
//! neither of which this compiler would otherwise have a use for.
//!
//! # Why a parameter is created before its operands are known
//!
//! Reading a variable at a loop header sends the lookup around the loop and back
//! to the same header, which would not terminate. So the parameter is appended
//! *first* and recorded as the block's definition; the recursion then finds it
//! and stops. Once the operands are in, a parameter whose operands are all one
//! value (ignoring references to itself) was never needed, and is dropped. That
//! is the whole of the minimality argument, and it is why parameters get created
//! and then removed rather than never created.
//!
//! A block with exactly one predecessor is the exception: its value simply *is*
//! the predecessor's, so no parameter is created at all — not created and then
//! removed. `read_var` follows whole chains of such blocks, which is where most
//! of the saving comes from on real control flow.
//!
//! # Removal, given no aliasing in this IR
//!
//! Cranelift can retire a parameter with `change_to_alias`. Here a dropped
//! parameter is recorded in a replacement map, resolved on every subsequent read,
//! and its column deleted in one pass at the end — which is exactly what
//! [`Func::simplify_params`](crate::jit::ir::Func::simplify_params)'s `apply`
//! already does. So operands are appended to the predecessor edges for *every*
//! parameter, including ones about to be dropped: the column and its arguments
//! are then deleted together, and the block's parameter list stays parallel with
//! every edge's argument list throughout.
//!
//! # Sealing
//!
//! A block may only be sealed once every predecessor edge has been declared, and
//! an edge may only be declared once its source block is *filled* — no further
//! definitions will be added to it. Reading through an unfilled predecessor would
//! see a definition that is not yet final. Until a block is sealed, a read that
//! reaches it parks an *incomplete* parameter, completed when the seal arrives;
//! that is what makes loop headers work.

use std::collections::HashMap;

use foldhash::fast::RandomState;

use crate::jit::ir::ty::Ty;
use crate::jit::ir::{Block, Func, Inst, Val};

/// A variable undergoing SSA construction: a Lua register.
pub type Var = u8;

/// One incoming edge: the source block, its terminator, and which of that
/// terminator's targets this edge is.
#[derive(Clone, Copy)]
struct Pred {
    from: Block,
    inst: Inst,
    target: usize,
}

#[derive(Clone, Default)]
struct BlockState {
    preds: Vec<Pred>,
    sealed: bool,
    /// Parameters minted while unsealed, awaiting operands.
    incomplete: Vec<(Var, Val)>,
    /// Set at seal time when there is exactly one predecessor. The fast path.
    single_pred: Option<Block>,
}

/// Work items for the explicit stack.
///
/// The lookup is recursive by nature, and its depth is bounded by the control
/// flow rather than by anything we choose, so it runs on a heap stack instead of
/// the machine one. Inlining will make these chains longer.
enum Task {
    Read { var: Var, block: Block },
    Finish { var: Var, block: Block, phi: Val },
}

pub struct SsaBuilder {
    /// `currentDef`: what each variable holds in each block. Dense because the
    /// register file is small and this is read on every variable access.
    defs: Vec<Vec<Option<Val>>>,
    blocks: Vec<BlockState>,
    /// Parameters found redundant, and what they turned out to be.
    repl: HashMap<Val, Val, RandomState>,
    tasks: Vec<Task>,
    results: Vec<Val>,
    /// Scratch for cycle detection in the single-predecessor walk.
    seen: Vec<Block>,
    /// The type each block declares for each variable. A parameter is typed by
    /// the block it lands on, which the chain walk may pick rather than the
    /// caller — so the table is consulted here rather than passed per read.
    reg_ty: Vec<Vec<Option<Ty>>>,
}

impl SsaBuilder {
    pub fn new(num_blocks: usize, num_vars: usize, reg_ty: Vec<Vec<Option<Ty>>>) -> Self {
        debug_assert_eq!(reg_ty.len(), num_blocks);
        SsaBuilder {
            defs: vec![vec![None; num_vars]; num_blocks],
            blocks: vec![BlockState::default(); num_blocks],
            repl: HashMap::default(),
            tasks: Vec::new(),
            results: Vec::new(),
            seen: Vec::new(),
            reg_ty,
        }
    }

    /// Declare an incoming edge. `inst` is `from`'s terminator and `target` its
    /// index among that terminator's targets.
    ///
    /// Only legal once `from` is filled: a read that follows this edge takes the
    /// definitions in `from` to be final.
    pub fn add_pred(&mut self, block: Block, from: Block, inst: Inst, target: usize) {
        debug_assert!(
            !self.blocks[block.index()].sealed,
            "b{} is already sealed",
            block.index()
        );
        self.blocks[block.index()]
            .preds
            .push(Pred { from, inst, target });
    }

    /// Record that `var` holds `val` from here on in `block`.
    pub fn write_var(&mut self, block: Block, var: Var, val: Val) {
        self.defs[block.index()][var as usize] = Some(val);
    }

    /// Whether `var` has a definition in `block` itself, without consulting
    /// predecessors.
    pub fn has_local(&self, block: Block, var: Var) -> bool {
        self.defs[block.index()][var as usize].is_some()
    }

    /// The value `var` holds at this point in `block`, minting parameters as
    /// needed.
    pub fn read_var(&mut self, func: &mut Func<'_>, block: Block, var: Var) -> Val {
        debug_assert!(self.tasks.is_empty() && self.results.is_empty());
        self.tasks.push(Task::Read { var, block });
        self.run(func);
        let v = self.results.pop().expect("a read yields exactly one value");
        debug_assert!(self.results.is_empty(), "unbalanced read");
        self.resolve(v)
    }

    /// No further predecessors will be declared for `block`; complete anything
    /// that was waiting on that.
    pub fn seal(&mut self, func: &mut Func<'_>, block: Block) {
        let st = &mut self.blocks[block.index()];
        debug_assert!(!st.sealed, "b{} sealed twice", block.index());
        st.sealed = true;
        if st.preds.len() == 1 {
            st.single_pred = Some(st.preds[0].from);
        }

        let incomplete = std::mem::take(&mut self.blocks[block.index()].incomplete);
        for (var, phi) in incomplete {
            self.schedule_operands(var, block, phi);
            self.run(func);
            self.results.pop().expect("completing yields one value");
            debug_assert!(self.results.is_empty(), "unbalanced seal");
        }
    }

    pub fn is_sealed(&self, block: Block) -> bool {
        self.blocks[block.index()].sealed
    }

    /// The replacement map for parameters that turned out to be redundant. The
    /// caller rewrites uses and deletes the columns.
    pub fn into_replacements(self) -> HashMap<Val, Val, RandomState> {
        debug_assert!(
            self.blocks.iter().all(|b| b.incomplete.is_empty()),
            "a block was never sealed",
        );
        self.repl
    }

    /// The same map, without consuming the builder.
    pub fn replacements(&self) -> &HashMap<Val, Val, RandomState> {
        &self.repl
    }

    /// Follow a value through any replacements made since it was handed out.
    ///
    /// A read of an unsealed block yields a parameter that may later turn out to
    /// be redundant — the loop-invariant case — so a value held across a `seal`
    /// is not necessarily still current. Uses recorded in the IR are fixed up by
    /// the caller's final rewrite; this is for asking directly.
    pub fn current(&self, v: Val) -> Val {
        self.resolve(v)
    }

    // --- the lookup ---------------------------------------------------------

    fn run(&mut self, func: &mut Func<'_>) {
        while let Some(t) = self.tasks.pop() {
            match t {
                Task::Read { var, block } => self.step_read(func, var, block),
                Task::Finish { var, block, phi } => self.step_finish(func, var, block, phi),
            }
        }
    }

    fn step_read(&mut self, func: &mut Func<'_>, var: Var, block: Block) {
        if let Some(v) = self.defs[block.index()][var as usize] {
            let v = self.resolve(v);
            self.results.push(v);
            return;
        }

        // Walk chains of sealed single-predecessor blocks. None of them needs a
        // parameter, and memoizing along the way makes the next read anywhere on
        // the chain a hit.
        self.seen.clear();
        let mut at = block;
        let found = loop {
            let Some(pred) = self.blocks[at.index()].single_pred else {
                break None;
            };
            // A cycle of single-predecessor blocks cannot be entered, so it is
            // dead — but it would still spin here forever.
            if self.seen.contains(&at) {
                break None;
            }
            self.seen.push(at);
            at = pred;
            if let Some(v) = self.defs[at.index()][var as usize] {
                break Some(self.resolve(v));
            }
        };

        // Either a definition, or a parameter on the block where the chain ran
        // out. Recording it *before* the operands are read is what stops a loop
        // sending the lookup around forever.
        let val = match found {
            Some(v) => v,
            None => {
                let ty = self.reg_ty[at.index()][var as usize]
                    .expect("a variable read through a block is live there");
                let phi = func.append_param(at, ty);
                self.defs[at.index()][var as usize] = Some(phi);
                if self.blocks[at.index()].sealed {
                    self.schedule_operands(var, at, phi);
                } else {
                    self.blocks[at.index()].incomplete.push((var, phi));
                    self.results.push(phi);
                }
                phi
            }
        };

        // Memoize onto every block the chain walked through.
        let mut b = block;
        while b != at {
            self.defs[b.index()][var as usize] = Some(val);
            b = self.blocks[b.index()]
                .single_pred
                .expect("the chain was walked through here");
        }

        // A sealed miss pushed its own tasks and will leave the result; anything
        // else resolves now.
        if found.is_some() {
            self.results.push(val);
        }
    }

    /// Queue a read on every predecessor, then the decision.
    fn schedule_operands(&mut self, var: Var, block: Block, phi: Val) {
        self.tasks.push(Task::Finish { var, block, phi });
        // Reversed, so the reads pop in predecessor order and their results line
        // up with the edges.
        for i in (0..self.blocks[block.index()].preds.len()).rev() {
            let from = self.blocks[block.index()].preds[i].from;
            self.tasks.push(Task::Read { var, block: from });
        }
    }

    fn step_finish(&mut self, func: &mut Func<'_>, var: Var, block: Block, phi: Val) {
        let n = self.blocks[block.index()].preds.len();
        let ops: Vec<Val> = self.results.split_off(self.results.len() - n);
        let ops: Vec<Val> = ops.into_iter().map(|v| self.resolve(v)).collect();

        // Every operand becomes an argument on its edge whether or not the
        // parameter survives, so the columns stay parallel; a dropped one is
        // deleted along with its arguments in the final pass.
        for (i, &op) in ops.iter().enumerate() {
            debug_assert_eq!(
                func.ty(op).rep,
                func.ty(phi).rep,
                "an edge operand must already have the parameter's representation: \
                 versions are keyed on exactly that, so a mismatch means analyze and \
                 emit disagreed",
            );
            let p = self.blocks[block.index()].preds[i];
            func.inst_mut(p.inst).targets[p.target].args.push(op);
        }

        // Redundant exactly when every operand other than the parameter itself is
        // the same value. All-self-references means unreachable code; leave it.
        let mut others = ops.iter().copied().filter(|&o| o != phi);
        let same = match others.next() {
            Some(v) if others.all(|o| o == v) => Some(v),
            _ => None,
        };

        match same {
            Some(v) => {
                self.repl.insert(phi, v);
                self.defs[block.index()][var as usize] = Some(v);
                self.results.push(v);
            }
            None => self.results.push(phi),
        }
    }

    fn resolve(&self, mut v: Val) -> Val {
        while let Some(&next) = self.repl.get(&v) {
            v = next;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::ir::op::Op;
    use crate::jit::ir::ty::{Rep, TypeSet};
    use crate::jit::ir::{BlockCall, InstData};

    const INT: Ty = Ty::new(Rep::I64, TypeSet::INT);

    /// A `Func` plus its builder, with just enough scaffolding to write CFGs by
    /// hand: blocks, jumps, and a way to mint an ordinary (non-parameter) value.
    struct T<'gc> {
        f: Func<'gc>,
        b: SsaBuilder,
    }

    impl<'gc> T<'gc> {
        fn new(nblocks: usize) -> Self {
            let mut f = Func::new();
            // Block 0 is the entry, already present.
            for _ in 1..nblocks {
                f.new_block();
            }
            let b = SsaBuilder::new(nblocks, 8, vec![vec![Some(INT); 8]; nblocks]);
            T { f, b }
        }

        fn blk(&self, i: usize) -> Block {
            Block(i as u32)
        }

        /// An ordinary definition in `block` — stands in for any computation.
        fn def(&mut self, block: Block, var: Var) -> Val {
            let (_, vals) = self.f.append_inst(
                block,
                InstData {
                    op: Op::IConst(0),
                    args: vec![],
                    targets: vec![],
                    results: vec![],
                    fs: None,
                    exit: None,
                },
                &[INT],
            );
            let v = vals[0];
            self.b.write_var(block, var, v);
            v
        }

        /// Terminate `from` with a jump to each of `targets`, and declare the
        /// edges. Filling the block first is the discipline `add_pred` requires.
        fn jump(&mut self, from: Block, targets: &[Block]) {
            let op = if targets.len() == 1 { Op::Jump } else { Op::Br };
            let (inst, _) = self.f.append_inst(
                from,
                InstData {
                    op,
                    args: vec![],
                    targets: targets
                        .iter()
                        .map(|&block| BlockCall {
                            block,
                            args: vec![],
                        })
                        .collect(),
                    results: vec![],
                    fs: None,
                    exit: None,
                },
                &[],
            );
            for (i, &t) in targets.iter().enumerate() {
                self.b.add_pred(t, from, inst, i);
            }
        }

        fn read(&mut self, block: Block, var: Var) -> Val {
            self.b.read_var(&mut self.f, block, var)
        }

        fn seal(&mut self, block: Block) {
            self.b.seal(&mut self.f, block);
        }

        fn params(&self, block: Block) -> usize {
            self.f.block(block).params.len()
        }

        /// Parameters that survived — the ones not recorded as redundant.
        fn live_params(&self, block: Block, repl: &HashMap<Val, Val, RandomState>) -> usize {
            self.f
                .block(block)
                .params
                .iter()
                .filter(|v| !repl.contains_key(v))
                .count()
        }
    }

    /// A straight line needs no parameters at all: every block has one
    /// predecessor, so the value is simply the one from before.
    #[test]
    fn straight_line_creates_no_parameters() {
        let mut t = T::new(3);
        let (b0, b1, b2) = (t.blk(0), t.blk(1), t.blk(2));
        t.b.seal(&mut t.f, b0);

        let v = t.def(b0, 0);
        t.jump(b0, &[b1]);
        t.seal(b1);
        t.jump(b1, &[b2]);
        t.seal(b2);

        assert_eq!(t.read(b2, 0), v, "the value carries through unchanged");
        assert_eq!(t.params(b1), 0);
        assert_eq!(t.params(b2), 0);
    }

    /// A diamond whose arms write different values needs one parameter at the
    /// join; one whose arms leave the value alone needs none.
    #[test]
    fn a_join_takes_a_parameter_only_when_the_arms_disagree() {
        let mut t = T::new(4);
        let (b0, l, r, j) = (t.blk(0), t.blk(1), t.blk(2), t.blk(3));
        t.b.seal(&mut t.f, b0);

        let outer = t.def(b0, 0);
        let outer1 = t.def(b0, 1);
        t.jump(b0, &[l, r]);
        t.seal(l);
        t.seal(r);

        // var 0 is rewritten on both arms; var 1 is left alone.
        t.def(l, 0);
        t.jump(l, &[j]);
        t.def(r, 0);
        t.jump(r, &[j]);
        t.seal(j);

        let joined = t.read(j, 0);
        let untouched = t.read(j, 1);
        let repl = t.b.replacements().clone();

        assert_ne!(joined, outer, "var 0 differs per arm: needs a parameter");
        assert!(!repl.contains_key(&joined));
        assert_eq!(
            t.b.current(untouched),
            outer1,
            "var 1 is the same on both arms: no parameter"
        );
        assert!(repl.contains_key(&t.f.block(j).params[1]));
        assert_eq!(t.live_params(j, &repl), 1);
    }

    /// The case the whole algorithm exists for: a loop header must mint its
    /// parameter before the operands are known, or the lookup goes round forever.
    #[test]
    fn a_loop_carried_variable_gets_one_parameter() {
        let mut t = T::new(4);
        let (b0, head, body, exit) = (t.blk(0), t.blk(1), t.blk(2), t.blk(3));
        t.b.seal(&mut t.f, b0);

        let init = t.def(b0, 0);
        t.jump(b0, &[head]);
        // `head` stays unsealed: the back edge has not been declared.
        t.jump(head, &[body, exit]);
        t.seal(body);

        // The body reads the loop-carried value and writes a new one.
        let read_in_body = t.read(body, 0);
        t.def(body, 0);
        t.jump(body, &[head]);
        t.seal(head); // now every predecessor is in
        t.seal(exit);

        let repl = t.b.replacements().clone();
        assert_eq!(t.live_params(head, &repl), 1, "one loop-carried parameter");
        assert_ne!(read_in_body, init, "the body sees the parameter, not init");
    }

    /// A variable defined before a loop and never written inside it needs no
    /// parameter at the header — the parameter is created to break the cycle and
    /// then found redundant. This is the `mix2` inner-loop case, where 32 of 36
    /// placed parameters were waste.
    #[test]
    fn a_loop_invariant_variable_needs_no_parameter() {
        let mut t = T::new(4);
        let (b0, head, body, exit) = (t.blk(0), t.blk(1), t.blk(2), t.blk(3));
        t.b.seal(&mut t.f, b0);

        let invariant = t.def(b0, 0);
        t.jump(b0, &[head]);
        t.jump(head, &[body, exit]);
        t.seal(body);

        // Read but never written in the loop.
        let inside = t.read(body, 0);
        t.jump(body, &[head]);
        t.seal(head);
        t.seal(exit);

        let after = t.read(exit, 0);
        let repl = t.b.replacements().clone();
        let _ = &repl;

        // `inside` was read before the header sealed, so it is the parameter that
        // sealing then found redundant — the caller's copy is updated by the
        // final rewrite, and `current` is how to ask now.
        assert_eq!(
            t.b.current(inside),
            invariant,
            "reads resolve to the original definition",
        );
        assert_eq!(t.b.current(after), invariant);
        assert_eq!(
            t.live_params(head, &repl),
            0,
            "the header parameter was redundant and dropped",
        );
    }

    /// Nested loops, both carrying one variable and both threading an invariant
    /// one — `mix2`'s shape in miniature.
    #[test]
    fn nested_loops_place_parameters_only_where_values_differ() {
        let mut t = T::new(6);
        let (b0, outer, inner, ibody, obody, exit) =
            (t.blk(0), t.blk(1), t.blk(2), t.blk(3), t.blk(4), t.blk(5));
        t.b.seal(&mut t.f, b0);

        let inv = t.def(b0, 0); // never written again
        t.def(b0, 1); // rewritten by the inner loop
        t.jump(b0, &[outer]);
        t.jump(outer, &[inner, exit]);
        t.jump(inner, &[ibody, obody]);
        t.seal(ibody);

        let seen = t.read(ibody, 0);
        t.read(ibody, 1);
        t.def(ibody, 1);
        t.jump(ibody, &[inner]);
        t.seal(inner);

        t.seal(obody);
        t.jump(obody, &[outer]);
        t.seal(outer);
        t.seal(exit);

        let repl = t.b.replacements().clone();
        assert_eq!(
            t.b.current(seen),
            inv,
            "the invariant reaches the inner body unchanged",
        );
        assert_eq!(
            t.live_params(inner, &repl),
            1,
            "only the rewritten variable needs an inner parameter",
        );
        assert_eq!(
            t.live_params(outer, &repl),
            1,
            "and it is carried round the outer loop too",
        );
    }
}
