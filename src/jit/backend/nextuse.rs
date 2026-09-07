//! Global next-use distances — Braun & Hack, CC'09 §4.1.
//!
//! Belady's rule is "evict the value whose next use is furthest away", and on a
//! single basic block that is easy: scan forward and read off the distance. Across
//! a CFG it is not. At the end of a block two live-out values both look infinitely
//! far away, because neither has a use *in this block*, and the choice between them
//! is then arbitrary — which is exactly the choice that matters.
//!
//! So this replaces liveness with something strictly richer: instead of a live set
//! per block, a map from each value to *how far away its next use is*. Liveness
//! falls out of it — a value is live exactly when its distance is finite — so this
//! subsumes the analysis it replaces rather than adding to it.
//!
//! # The loop trick
//!
//! One detail carries most of the benefit. Every control-flow edge gets a length,
//! and **edges leaving a loop get a very large one** ([`M`]); every other edge gets
//! zero. Edge lengths add into the distances of values crossing them.
//!
//! The effect is that a use *after* a loop ranks as further away than any use
//! *inside* it, however many instructions actually separate them. So when the
//! spiller has to evict something, it evicts the value the loop does not touch, and
//! the reload for that value lands outside the loop instead of in its body. That is
//! the whole of Braun09's advantage over a linear-scan splitter: Wimmer05 recovers
//! the same effect by special-casing split positions against a flattened block
//! order, and Braun09 gets it structurally from the CFG (their §6).
//!
//! [`M`] is a sentinel, not a frequency estimate — it only has to exceed the
//! longest path through any loop so that the ordering comes out right. Do not
//! confuse it with the execution-frequency weight in [`super::spillcost`], which is
//! a different number for a different purpose.
//!
//! # Where this departs from the paper
//!
//! **Edge lengths are kept on edges.** The paper folds `ℓ` into the block transfer
//! function and notes (their footnote 5) that this is sound only because critical
//! edges are split, so an edge length can be attributed uniquely to a block. That
//! is a concession to presenting the analysis in standard dataflow form. Writing
//! the fixpoint directly, as here, there is no reason to launder edge lengths
//! through blocks — the join takes the minimum over successors of
//! `entry[S] + ℓ(B,S)` and the attribution question never arises.
//!
//! **The transfer function kills definitions.** As printed, the paper's `f_B` has
//! two cases — a use in the block, or `|B| + a(v)` otherwise — and no third case
//! for a value *defined* in the block. Taken literally that makes a value defined
//! in `B` and live out of `B` come out live-*in* at `B`, which is wrong, and would
//! be wrong the same way in plain liveness without the `− def` term. Killed
//! explicitly below.
//!
//! **Pressure is per register class.** Braun09 assumes one register file. Distances
//! are class-agnostic and stay shared, but the maximum-pressure figure that §4.2
//! feeds to `p_L` is meaningless pooled across classes, since a float value in a
//! register costs nothing against the integer budget.

use super::order::Layout;
use super::regalloc::{Block, RegClass, RegallocFunc, VReg};

/// No further use on any path — the value is dead. Saturating, so arithmetic on it
/// stays put.
pub const INF: u32 = u32::MAX;

/// The length of an edge leaving a loop.
///
/// Has to exceed the longest path through any loop, so that a use after the loop
/// outranks every use inside it; the paper suggests 100000 and notes that in
/// practice it "works nicely". Nothing here depends on the exact value.
pub const M: u32 = 100_000;

/// Next-use distances at block boundaries, plus the pressure figures §4.2 needs.
pub struct NextUse {
    /// Per block, the distance from the block's entry to each value's next use.
    /// A value is live-in exactly where this is below [`INF`].
    pub entry: Vec<Vec<u32>>,
    /// Per block, the same measured from the block's exit — the join over
    /// successors, edge lengths included.
    pub exit: Vec<Vec<u32>>,
    /// Per block and register class, the most values simultaneously live at any
    /// point in the block.
    pub pressure: Vec<[u32; RegClass::COUNT]>,
}

impl NextUse {
    /// The distance to `v`'s next use from the entry of `b`.
    pub fn at_entry(&self, b: Block, v: VReg) -> u32 {
        self.entry[b.0 as usize][v.0 as usize]
    }

    /// The distance to `v`'s next use from the exit of `b`.
    pub fn at_exit(&self, b: Block, v: VReg) -> u32 {
        self.exit[b.0 as usize][v.0 as usize]
    }

    /// Whether `v` is live into `b`. Liveness is a shadow of the distances: finite
    /// means some path reaches a use.
    pub fn live_in(&self, b: Block, v: VReg) -> bool {
        self.at_entry(b, v) < INF
    }

    /// The greatest pressure anywhere in loop `l`, over the blocks the layout
    /// assigns to it — Braun09's `p_L`, which §4.2 uses to guess how many
    /// live-through values can survive a loop without being evicted.
    pub fn loop_pressure(&self, layout: &Layout, l: Block, class: RegClass) -> u32 {
        (0..self.pressure.len())
            .filter(|&b| in_loop(layout, Block(b as u32), l))
            .map(|b| self.pressure[b][class as usize])
            .max()
            .unwrap_or(0)
    }
}

/// Whether `b` lies within the loop headed by `l`, at any nesting depth.
pub fn in_loop(layout: &Layout, b: Block, l: Block) -> bool {
    let mut h = layout.loop_header[b.0 as usize];
    while let Some(x) = h {
        if x == l {
            return true;
        }
        h = layout.loop_parent[x.0 as usize];
    }
    false
}

/// How many loops the edge `b -> s` leaves.
///
/// Zero for an edge that stays put or enters a loop. The paper gives every
/// loop-leaving edge the same length `M`; counting them instead means that breaking
/// out of two nested loops ranks as further than breaking out of one, which is the
/// same idea applied consistently and costs nothing.
fn loops_exited(layout: &Layout, b: Block, s: Block) -> u32 {
    let mut n = 0;
    let mut h = layout.loop_header[b.0 as usize];
    while let Some(x) = h {
        if !in_loop(layout, s, x) {
            n += 1;
        }
        h = layout.loop_parent[x.0 as usize];
    }
    n
}

/// Distance from the entry of `b` to the first use of each value that is not
/// preceded by that value's definition, plus which values `b` defines.
///
/// Returns `(nu, defined)`. A jump argument counts as a use at the terminator, and
/// a block parameter as a definition at the entry, since neither is an operand of
/// any instruction.
fn local(f: &impl RegallocFunc, b: Block) -> (Vec<u32>, Vec<bool>) {
    let n = f.num_vregs();
    let mut nu = vec![INF; n];
    let mut defined = vec![false; n];

    for &p in f.block_params(b) {
        defined[p.0 as usize] = true;
    }

    let insts = f.block_insts(b);
    for (d, &i) in insts.iter().enumerate() {
        // Temps are deliberately absent. A temp is scratch belonging to one
        // instruction, not a value that flows between them — counting it as a use
        // would make it look live-in at the block, since no instruction defines it.
        for o in f.uses(i) {
            let v = o.vreg.0 as usize;
            // First use wins, and only if the value was not defined earlier in this
            // block — past its own definition the value is a different thing, and
            // its distance from the block entry is meaningless.
            if !defined[v] && nu[v] == INF {
                nu[v] = d as u32;
            }
        }
        for o in f.defs(i).iter().chain(f.temps(i)) {
            defined[o.vreg.0 as usize] = true;
        }
    }

    // The arguments this block passes are read by its terminator.
    let end = insts.len().saturating_sub(1) as u32;
    for &a in f.jump_args(b) {
        let v = a.0 as usize;
        if !defined[v] && nu[v] == INF {
            nu[v] = end;
        }
    }

    (nu, defined)
}

/// Solve for next-use distances over `f`.
///
/// A backward fixpoint, like liveness, over distances rather than bits. It
/// converges for the reason liveness does: the join is a minimum, every transfer is
/// monotone, and the domain has no infinite descending chain (`INF` down to 0).
pub fn analyze(f: &impl RegallocFunc, layout: &Layout) -> NextUse {
    let nb = f.num_blocks();
    let nv = f.num_vregs();

    let locals: Vec<(Vec<u32>, Vec<bool>)> = (0..nb).map(|b| local(f, Block(b as u32))).collect();
    let succs: Vec<Vec<Block>> = (0..nb).map(|b| f.succs(Block(b as u32))).collect();
    let len: Vec<u32> = (0..nb)
        .map(|b| f.block_insts(Block(b as u32)).len() as u32)
        .collect();

    let mut entry = vec![vec![INF; nv]; nb];
    let mut exit = vec![vec![INF; nv]; nb];

    // Reverse layout order approximates a backward sweep, so most information
    // travels one block per iteration and the loop turns few times.
    let mut rev: Vec<Block> = layout.order.clone();
    rev.reverse();

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rev {
            let bi = b.0 as usize;

            // Join: the nearest use over any successor, each charged the length of
            // the edge that reaches it.
            let mut out = vec![INF; nv];
            for &s in &succs[bi] {
                let edge = M.saturating_mul(loops_exited(layout, b, s));
                let ent = &entry[s.0 as usize];
                for v in 0..nv {
                    let d = ent[v].saturating_add(edge);
                    if d < out[v] {
                        out[v] = d;
                    }
                }
            }

            let (nu, defined) = &locals[bi];
            let mut ent = vec![INF; nv];
            for v in 0..nv {
                ent[v] = if nu[v] != INF {
                    // Used in this block before any definition of it.
                    nu[v]
                } else if defined[v] {
                    // Defined here and not used before that, so not live in — the
                    // kill the paper's printed transfer function leaves out.
                    INF
                } else {
                    // Passes straight through: cross the block, then continue.
                    len[bi].saturating_add(out[v])
                };
            }

            if ent != entry[bi] || out != exit[bi] {
                entry[bi] = ent;
                exit[bi] = out;
                changed = true;
            }
        }
    }

    let pressure = pressures(f, &exit, nb, nv);
    NextUse {
        entry,
        exit,
        pressure,
    }
}

/// The maximum simultaneous live values per block, per class.
///
/// Braun09 notes this comes free during the liveness pass; it is separate here only
/// because the fixpoint above iterates and this must be counted once, at the end.
fn pressures(
    f: &impl RegallocFunc,
    exit: &[Vec<u32>],
    nb: usize,
    nv: usize,
) -> Vec<[u32; RegClass::COUNT]> {
    let mut out = vec![[0u32; RegClass::COUNT]; nb];

    for b in 0..nb {
        let blk = Block(b as u32);
        let mut live: Vec<bool> = exit[b].iter().map(|&d| d < INF).collect();
        // The arguments this block passes are read by the edge, which runs before
        // the block's exit — so they are not live *out*, but they are certainly
        // occupying registers at the branch. Omitting them makes `p_L` undercount
        // by the width of every back edge, which is precisely where it is consulted.
        for &a in f.jump_args(blk) {
            live[a.0 as usize] = true;
        }
        let mut count = [0u32; RegClass::COUNT];
        for v in 0..nv {
            if live[v] {
                count[f.class(VReg(v as u32)) as usize] += 1;
            }
        }
        let mut peak = count;

        // Backward: a def kills, a use makes live. The peak is taken after each
        // instruction's effect, which is where pressure is highest.
        for &i in f.block_insts(blk).iter().rev() {
            for o in f.defs(i) {
                let v = o.vreg.0 as usize;
                if live[v] {
                    live[v] = false;
                    count[f.class(o.vreg) as usize] -= 1;
                }
            }
            for o in f.uses(i) {
                let v = o.vreg.0 as usize;
                if !live[v] {
                    live[v] = true;
                    count[f.class(o.vreg) as usize] += 1;
                }
            }
            // A temp needs a register *at this instruction* and nowhere else, so it
            // raises the peak here without joining the live set. Adding it to `live`
            // would leave it there for the whole rest of the backward walk, which on
            // `mix` inflated the peak from 20 to 27 — enough to claim a function that
            // allocates with zero spill traffic could not possibly fit.
            let mut here = count;
            for o in f.temps(i) {
                here[f.class(o.vreg) as usize] += 1;
            }
            for c in 0..RegClass::COUNT {
                peak[c] = peak[c].max(here[c]);
            }
        }
        out[b] = peak;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::order;
    use crate::jit::backend::regalloc::Operand;
    use crate::jit::backend::testfunc::TestFunc;

    fn analyze_of(f: &TestFunc) -> (NextUse, Layout) {
        let layout = order::compute(f).expect("reducible");
        let nu = analyze(f, &layout);
        (nu, layout)
    }

    /// The distance really is a count of instructions to the next read.
    #[test]
    fn distance_counts_instructions_to_the_next_use() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();

        f.inst(b, vec![Operand::any(v)], vec![]); // 0: def
        f.inst(b, vec![], vec![]); // 1
        f.inst(b, vec![], vec![]); // 2
        f.inst(b, vec![], vec![Operand::any(v)]); // 3: use

        let (nu, _) = analyze_of(&f);
        // Defined here, so not live in, however near the use is.
        assert_eq!(nu.at_entry(b, v), INF);
        assert!(!nu.live_in(b, v));
    }

    /// A value passing through a block is charged for crossing it.
    #[test]
    fn a_live_through_value_pays_the_block_it_crosses() {
        let mut f = TestFunc::default();
        let (entry, mid, tail) = (f.block(), f.block(), f.block());
        let v = f.int();

        f.inst(entry, vec![Operand::any(v)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[mid]);

        // Three instructions, none of them touching `v`.
        f.inst(mid, vec![], vec![]);
        f.inst(mid, vec![], vec![]);
        f.inst(mid, vec![], vec![]);
        f.goto(mid, &[tail]);

        f.inst(tail, vec![], vec![Operand::any(v)]);

        let (nu, _) = analyze_of(&f);
        assert_eq!(nu.at_entry(tail, v), 0, "used by the first instruction");
        assert_eq!(
            nu.at_entry(mid, v),
            3,
            "three instructions to cross, then used at once"
        );
        assert!(nu.live_in(mid, v), "liveness falls out of the distance");
    }

    /// The point of the whole analysis. Two values are both live out of a block with
    /// no use in it, so a block-local view calls both infinitely far away and picks
    /// between them arbitrarily. Globally they are plainly different: one is read in
    /// the very next block, the other only much later.
    #[test]
    fn a_global_view_separates_two_values_a_local_one_cannot() {
        let mut f = TestFunc::default();
        let (entry, near, far) = (f.block(), f.block(), f.block());
        let (soon, later) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(soon)], vec![]);
        f.inst(entry, vec![Operand::any(later)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[near]);

        f.inst(near, vec![], vec![Operand::any(soon)]);
        f.inst(near, vec![], vec![]);
        f.goto(near, &[far]);

        for _ in 0..5 {
            f.inst(far, vec![], vec![]);
        }
        f.inst(far, vec![], vec![Operand::any(later)]);

        let (nu, _) = analyze_of(&f);
        let (a, b) = (nu.at_exit(entry, soon), nu.at_exit(entry, later));
        assert!(a < INF && b < INF, "both are live out of the entry block");
        assert!(
            a < b,
            "the value used in the next block must rank nearer than the one used later \
             ({a} vs {b}) — this is the comparison Belady cannot make block-locally"
        );
    }

    /// The loop trick, which is where Braun09's advantage comes from. A value used
    /// *inside* a loop and one used only *after* it must not compare by raw
    /// instruction count, or the spiller evicts the wrong one and the reload lands
    /// in the loop body.
    #[test]
    fn a_use_after_a_loop_outranks_a_use_inside_it() {
        let mut f = TestFunc::default();
        let (entry, header, body, exit) = (f.block(), f.block(), f.block(), f.block());
        let (in_loop_v, after_loop_v) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(in_loop_v)], vec![]);
        f.inst(entry, vec![Operand::any(after_loop_v)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        // The header branches to the body or out; the body loops back.
        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        // A long body, so that raw instruction counts would favour the wrong value.
        for _ in 0..20 {
            f.inst(body, vec![], vec![]);
        }
        f.inst(body, vec![], vec![Operand::any(in_loop_v)]);
        f.goto(body, &[header]);

        f.inst(exit, vec![], vec![Operand::any(after_loop_v)]);

        let (nu, layout) = analyze_of(&f);
        assert!(layout.is_header(header), "the test needs a real loop");

        let inside = nu.at_entry(header, in_loop_v);
        let outside = nu.at_entry(header, after_loop_v);
        assert!(
            inside < outside,
            "the in-loop value ({inside}) must rank nearer than the post-loop one \
             ({outside}), even though the loop body is 21 instructions long and the \
             post-loop use is 1 instruction past the header"
        );
        assert!(
            outside >= M,
            "the post-loop use must have been charged the loop-exit edge length, \
             got {outside}"
        );
    }

    /// Pressure is counted per register file: a float in a register costs nothing
    /// against the integer budget, so pooling them would make `p_L` meaningless.
    #[test]
    fn pressure_is_counted_per_register_class() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (i1, i2) = (f.int(), f.int());
        let g = f.vreg(RegClass::Float);

        f.inst(b, vec![Operand::any(i1)], vec![]);
        f.inst(b, vec![Operand::any(i2)], vec![]);
        f.inst(b, vec![Operand::any(g)], vec![]);
        f.inst(b, vec![], vec![Operand::any(i1), Operand::any(i2)]);
        f.inst(b, vec![], vec![Operand::any(g)]);

        let (nu, _) = analyze_of(&f);
        let p = nu.pressure[b.0 as usize];
        assert_eq!(p[RegClass::Int as usize], 2, "two integers live at once");
        assert_eq!(p[RegClass::Float as usize], 1, "one float, counted apart");
    }

    /// A block parameter is defined at its block's entry, so it is not live *into*
    /// that block however soon it is read.
    #[test]
    fn a_block_parameter_is_not_live_into_its_own_block() {
        let mut f = TestFunc::default();
        let (entry, target) = (f.block(), f.block());
        let (arg, param) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(arg)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[target]);
        f.pass(entry, &[arg]);

        f.params(target, &[param]);
        f.inst(target, vec![], vec![Operand::any(param)]);

        let (nu, _) = analyze_of(&f);
        assert_eq!(
            nu.at_entry(target, param),
            INF,
            "a parameter is defined at the entry, not live into it"
        );
        assert_eq!(
            nu.at_exit(entry, arg),
            INF,
            "an argument is read by the edge, which runs *before* the exit, so it \
             is not live out — the pressure count picks it up separately"
        );
    }

    /// A value whose only use in a block is being passed along its edge is still
    /// live to the branch, and still occupies a register there. Jump arguments are
    /// operands of no instruction, so nothing counts them unless it is made to.
    #[test]
    fn a_jump_argument_counts_toward_pressure_and_is_live_in() {
        let mut f = TestFunc::default();
        let (entry, mid, target) = (f.block(), f.block(), f.block());
        let (a, b, param_a, param_b) = (f.int(), f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(a)], vec![]);
        f.inst(entry, vec![Operand::any(b)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[mid]);

        // `mid` neither reads nor writes either value — it only forwards them.
        f.inst(mid, vec![], vec![]);
        f.inst(mid, vec![], vec![]);
        f.goto(mid, &[target]);
        f.pass(mid, &[a, b]);

        f.params(target, &[param_a, param_b]);
        f.inst(target, vec![], vec![Operand::any(param_a)]);
        f.inst(target, vec![], vec![Operand::any(param_b)]);

        let (nu, _) = analyze_of(&f);
        assert_eq!(
            nu.at_entry(mid, a),
            1,
            "live in, with its use at the terminator that carries the edge"
        );
        assert_eq!(
            nu.pressure[mid.0 as usize][RegClass::Int as usize],
            2,
            "both arguments hold a register at the branch"
        );
    }
}
