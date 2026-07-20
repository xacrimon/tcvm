//! Redundant block-parameter elimination.
//!
//! Two kinds of redundant phi, either of which a minimal SSA would not have
//! formed:
//!
//! * **trivial** — every incoming arg is the same value `v` (ignoring the phi's
//!   own back-edge self-reference); the phi *is* `v`. A loop-invariant threaded
//!   through the header collapses this way, back to its dominating def.
//! * **congruent** — two columns of one block receive an identical arg on every
//!   edge; they are the same value. A numeric `for` hits this: the counter and the
//!   visible loop variable are one value carried in two columns.
//!
//! Removing them turns the two back-edge copies of `i` into one (which the
//! coalescer may then erase) and, more generally, shrinks the live set handed to
//! the allocator. This is Braun et al.'s trivial-phi removal plus phi congruence,
//! run as a worklist so a collapse that exposes another is chased without
//! re-scanning untouched blocks.
//!
//! The frontend now builds SSA on the fly
//! ([`ssa`](crate::jit::frontend::ssa)), so neither kind survives construction —
//! the builder drops a parameter the moment its operands agree, and congruence is
//! collected by the seeded call below. What this pass is *for* is being run
//! again: any transformation that proves two values equal can make a
//! parameter's operands agree and leave a trivial one behind that did not exist
//! when the IR was built.
//!
//! Soundness rides on a standard SSA property, which the verifier re-checks: a
//! phi all of whose operands are one value `v` has `v` dominating the phi's block,
//! so replacing the phi with `v` cannot outrun a definition. Congruence merges are
//! trivially safe — both columns are defined at the same block entry.

use std::collections::HashMap;

use foldhash::fast::RandomState;

use super::{Block, Def, Func, Inst, Val};

impl<'gc> Func<'gc> {
    /// Eliminate trivial and congruent block parameters to a fixpoint, rewriting
    /// their uses (instruction args, edge args, and frame-state registers, so a
    /// deopt still reconstructs every Lua register) and deleting the columns.
    ///
    /// Re-runnable, and meant to be re-run: any pass that proves two values equal
    /// can make a parameter's operands agree and leave a trivial one behind that
    /// did not exist when the IR was built. GVN is the obvious case.
    pub fn simplify_params(&mut self) {
        self.simplify_params_seeded(HashMap::default());
    }

    /// As [`simplify_params`](Self::simplify_params), starting from parameters an
    /// earlier stage already proved redundant.
    ///
    /// This exists for exactly one caller: on-the-fly SSA construction retires
    /// the sentinels it minted to break a lookup cycle, and those replacements
    /// have to reach the IR. Seeding rather than applying them separately is what
    /// keeps the rewrite to a single pass — `apply` walks every instruction,
    /// every edge argument and every frame state, and a frame state carries a
    /// slot per Lua register, so doing it twice costs more than either discovery
    /// pass does.
    ///
    /// Seeding is sound because nothing is compacted until `apply`: every block's
    /// parameter list and every edge's argument list stay parallel throughout
    /// discovery, so the column indices still line up, and an already-retired
    /// parameter is skipped by the `repl` check at the top of the worklist.
    pub fn simplify_params_seeded(&mut self, seed: HashMap<Val, Val, RandomState>) {
        let preds = self.pred_edges();

        // The value a param was proven equal to. Targets are stored already
        // resolved, so `resolve` chases chains at most one deep and cannot cycle.
        let mut repl: HashMap<Val, Val, RandomState> = seed;
        // Which params name a value as a *raw* operand. When that value is
        // eliminated, those params may in turn become trivial or congruent.
        let mut users: HashMap<Val, Vec<Val>, RandomState> = HashMap::default();
        // The canonical param registered for a `(block, resolved operands)`
        // signature, plus each param's currently-registered signature so a
        // reprocess can retract a stale one before it wrongly attracts others.
        let mut canon: HashMap<(Block, Vec<Val>), Val, RandomState> = HashMap::default();
        let mut sig_of_param: HashMap<Val, Vec<Val>, RandomState> = HashMap::default();

        let mut work: Vec<Val> = Vec::new();
        for (b, pred_edges) in preds.iter().enumerate() {
            if pred_edges.is_empty() {
                continue; // entry / unreachable: no phis to resolve
            }
            for p in self.blocks[b].params.clone() {
                // Seeded columns are already retired and still present — nothing
                // compacts until `apply`. Scanning them would mean building the
                // operand vectors for every parameter construction *created*
                // rather than every one it kept, which on a loop-heavy function
                // is twice the work for no result.
                if repl.contains_key(&p) {
                    continue;
                }
                for o in self.raw_operands(pred_edges, p) {
                    users.entry(o).or_default().push(p);
                }
                work.push(p);
            }
        }

        while let Some(p) = work.pop() {
            if repl.contains_key(&p) {
                continue; // already eliminated
            }
            let Def::Param(b, pos) = self.def(p) else {
                continue;
            };
            let pos = pos as usize;

            // A stale signature this param registered on an earlier visit no
            // longer describes it — retract before recomputing.
            if let Some(old) = sig_of_param.remove(&p) {
                let old_key = (b, old);
                if canon.get(&old_key) == Some(&p) {
                    canon.remove(&old_key);
                }
            }

            let ops: Vec<Val> = preds[b.index()]
                .iter()
                .map(|&(inst, t)| resolve(&repl, self.insts[inst.index()].targets[t].args[pos]))
                .collect();

            // Trivial: one distinct operand once the phi's self-reference is
            // dropped. That operand becomes the phi's value.
            //
            // Construction never leaves one of these — the builder compares the
            // operands at the moment it creates the parameter, and measurement
            // agrees: zero trivial eliminations across the corpus on the seeded
            // call. The check earns its keep on a *later* run, after a pass that
            // made two operands equal.
            let mut distinct = ops.iter().copied().filter(|&o| o != p);
            let first = distinct.next();
            let trivial = match first {
                None => None, // only self-references: leave it be
                Some(v) if distinct.all(|o| o == v) => Some(v),
                Some(_) => None,
            };
            if let Some(v) = trivial {
                eliminate(p, v, &mut repl, &users, &mut work);
                continue;
            }

            // Congruent: another column of this block already carries this exact
            // operand vector. Merge into it (same-block, so always dominance-safe).
            let key = (b, ops.clone());
            match canon.get(&key) {
                Some(&q) if q != p && !repl.contains_key(&q) => {
                    eliminate(p, q, &mut repl, &users, &mut work);
                }
                _ => {
                    canon.insert(key, p);
                    sig_of_param.insert(p, ops);
                }
            }
        }

        if repl.is_empty() {
            return;
        }
        self.apply(&repl, &preds);
    }

    /// Predecessor edges per block, in a fixed order: each is a (terminator,
    /// target index) that carries one arg per column of the target block.
    fn pred_edges(&self) -> Vec<Vec<(Inst, usize)>> {
        let mut preds: Vec<Vec<(Inst, usize)>> = vec![Vec::new(); self.blocks.len()];
        for i in 0..self.insts.len() {
            let inst = Inst(i as u32);
            for (t, call) in self.insts[i].targets.iter().enumerate() {
                preds[call.block.index()].push((inst, t));
            }
        }
        preds
    }

    /// The raw (unresolved) operands of param `p` across its predecessor edges.
    fn raw_operands(&self, preds: &[(Inst, usize)], p: Val) -> Vec<Val> {
        let Def::Param(_, pos) = self.def(p) else {
            return Vec::new();
        };
        let pos = pos as usize;
        preds
            .iter()
            .map(|&(inst, t)| self.insts[inst.index()].targets[t].args[pos])
            .collect()
    }

    /// Rewrite every use of an eliminated param to its replacement, then delete
    /// the dead columns from each block and the matching arg from every edge.
    fn apply(&mut self, repl: &HashMap<Val, Val, RandomState>, preds: &[Vec<(Inst, usize)>]) {
        // 1. Uses: instruction args, edge args, and frame-state registers. Defs
        //    (results, params) are left alone; a deleted param's column goes in
        //    step 2.
        for inst in &mut self.insts {
            for a in &mut inst.args {
                *a = resolve(repl, *a);
            }
            for call in &mut inst.targets {
                for a in &mut call.args {
                    *a = resolve(repl, *a);
                }
            }
        }
        for fs in &mut self.states {
            for slot in fs.regs.iter_mut().flatten() {
                *slot = resolve(repl, *slot);
            }
        }

        // 2. Columns: keep the survivors, in order, in both the block's params and
        //    every predecessor edge's args. Renumber the survivors' `Def::Param`.
        for (b, pred_edges) in preds.iter().enumerate() {
            let params = &self.blocks[b].params;
            let kept: Vec<usize> = (0..params.len())
                .filter(|&c| !repl.contains_key(&params[c]))
                .collect();
            if kept.len() == params.len() {
                continue;
            }

            let new_params: Vec<Val> = kept.iter().map(|&c| params[c]).collect();
            for (n, &v) in new_params.iter().enumerate() {
                self.values[v.index()].def = Def::Param(Block(b as u32), n as u32);
            }
            self.blocks[b].params = new_params;

            for &(inst, t) in pred_edges {
                let args = &self.insts[inst.index()].targets[t].args;
                let new_args: Vec<Val> = kept.iter().map(|&c| args[c]).collect();
                self.insts[inst.index()].targets[t].args = new_args;
            }
        }
    }
}

/// Record `p == v` and re-queue every param that named `p` as a raw operand;
/// keyed on the raw operand, reprocessing recomputes resolved operands and sees
/// the collapse.
fn eliminate(
    p: Val,
    v: Val,
    repl: &mut HashMap<Val, Val, RandomState>,
    users: &HashMap<Val, Vec<Val>, RandomState>,
    work: &mut Vec<Val>,
) {
    let v = resolve(repl, v);
    if v == p {
        return; // never point a value at itself
    }
    repl.insert(p, v);
    if let Some(list) = users.get(&p) {
        work.extend_from_slice(list);
    }
}

fn resolve(repl: &HashMap<Val, Val, RandomState>, mut v: Val) -> Val {
    while let Some(&n) = repl.get(&v) {
        if n == v {
            break;
        }
        v = n;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::ir::op::Op;
    use crate::jit::ir::ty::{Rep, Ty, TypeSet};
    use crate::jit::ir::{BlockCall, InstData};

    const INT: Ty = Ty::new(Rep::I64, TypeSet::INT);

    fn iconst(f: &mut Func<'static>, b: Block, n: i64) -> Val {
        let (_, vals) = f.append_inst(
            b,
            InstData {
                op: Op::IConst(n),
                args: vec![],
                targets: vec![],
                results: vec![],
                fs: None,
                exit: None,
            },
            &[INT],
        );
        vals[0]
    }

    fn term(f: &mut Func<'static>, b: Block, op: Op, args: Vec<Val>, t: Vec<BlockCall>) -> Inst {
        f.append_inst(
            b,
            InstData {
                op,
                args,
                targets: t,
                results: vec![],
                fs: None,
                exit: None,
            },
            &[],
        )
        .0
    }

    /// A diamond whose arms pass different values, so the join needs its
    /// parameter — until something proves the two equal.
    ///
    /// This is the case `simplify_params` exists for now that construction
    /// leaves nothing trivial behind: GVN (or any pass that rewrites an edge
    /// argument) can make the operands agree long after the IR was built, and
    /// re-running has to notice. Without a caller in the compiler yet, this is
    /// the only thing keeping that path honest.
    #[test]
    fn a_parameter_that_becomes_trivial_later_is_still_collected() {
        let mut f = Func::new();
        let (b0, b1, b2, b3) = (f.entry, f.new_block(), f.new_block(), f.new_block());

        let a = iconst(&mut f, b0, 1);
        let b = iconst(&mut f, b0, 2);
        let cond = iconst(&mut f, b0, 0);
        term(
            &mut f,
            b0,
            Op::Br,
            vec![cond],
            vec![
                BlockCall {
                    block: b1,
                    args: vec![],
                },
                BlockCall {
                    block: b2,
                    args: vec![],
                },
            ],
        );

        term(
            &mut f,
            b1,
            Op::Jump,
            vec![],
            vec![BlockCall {
                block: b3,
                args: vec![a],
            }],
        );
        let from_b2 = term(
            &mut f,
            b2,
            Op::Jump,
            vec![],
            vec![BlockCall {
                block: b3,
                args: vec![b],
            }],
        );

        let p = f.append_param(b3, INT);
        term(&mut f, b3, Op::Ret, vec![p], vec![]);

        // The arms disagree, so the parameter is load-bearing and stays.
        f.simplify_params();
        assert_eq!(
            f.block(b3).params.len(),
            1,
            "the arms pass different values"
        );

        // Now something proves them equal — the shape GVN leaves behind.
        f.inst_mut(from_b2).targets[0].args[0] = a;
        f.simplify_params();

        assert!(
            f.block(b3).params.is_empty(),
            "both edges now pass the same value: the parameter is trivial",
        );
        let ret = f.block(b3).insts.last().copied().expect("b3 has a return");
        assert_eq!(
            f.inst(ret).args[0],
            a,
            "the return should have been rewritten to the value itself",
        );
    }
}
