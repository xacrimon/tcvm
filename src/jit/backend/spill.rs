//! Belady spilling over a CFG — Braun & Hack, CC'09 §2 and §4.
//!
//! This decides *what lives in a register where*, before any register is assigned.
//! That ordering is the whole idea. On SSA form a program's register demand equals
//! its maximum register pressure, so once a transformation has lowered pressure to
//! `k` everywhere, assignment provably needs no further spills and can be done
//! greedily. Spilling stops being driven by an allocator running out of registers
//! and starts being driven by the shape of the program.
//!
//! The core is Belady's rule — when room is needed, evict the value whose next use
//! is furthest away — which is optimal on straight-line code and, per Farach &
//! Liberatore, a 2C-approximation of the local problem. [`super::nextuse`] supplies
//! the distances, and supplies them CFG-globally, which is what lifts the rule off
//! a single block.
//!
//! # The three things that make it global (§4)
//!
//! Running Algorithm 1 per block would go wrong in three ways, and the fixes are
//! most of this module:
//!
//! 1. **Distances are block-local.** Fixed by [`super::nextuse`].
//! 2. **`W` would start empty**, so every live-in value reloads on its first use.
//!    Fixed by processing blocks in layout order and deriving each block's entry
//!    set from its predecessors' exit sets ([`init_usual`]).
//! 3. **A loop body cannot have two versions**, one with the reload and one
//!    without. Putting the reload inside the loop executes it every iteration when
//!    it was needed only on the first. Fixed by treating loop headers specially —
//!    ignoring the predecessors entirely and seeding `W` from what the loop *uses*,
//!    which hoists the reload onto the edge into the loop ([`init_loop_header`],
//!    the paper's Fig. 2a versus 2b).
//!
//! # Per register class
//!
//! Braun09 assumes one register file; we have two with different sizes. The whole
//! algorithm runs once per class over only that class's values, with `k` the size
//! of that class's pool. Nothing about the algorithm cares, but it does mean `W`,
//! `S` and the pressure figures are all per class.
//!
//! # What this does *not* do
//!
//! It does not reconstruct SSA, and does not need to. The paper inserts real reload
//! instructions, which are second definitions of a value, which breaks SSA and
//! forces their §4.4 repair (Sastry & Ju, walking the dominance tree and inserting
//! φ-functions lazily). Here a reload is an edit and a split is a split: the value
//! keeps one definition and merely changes location, so the question §4.4 answers —
//! "which definition reaches this use" — never arises. What replaces it is
//! [`super::regalloc::Locations`]' position-indexed lookup.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;

use super::nextuse::{INF, NextUse, in_loop};
use super::order::Layout;
use super::regalloc::{Block, Inst, MachineEnv, RegClass, RegallocFunc, VReg};

/// Where the spiller decided values should live.
///
/// `W` is the paper's register set: the values that are in registers at a point.
/// Everything live but absent from `W` is in memory there.
#[derive(Debug, Default)]
pub struct SpillPlan {
    /// Values in registers at each block's entry.
    pub w_entry: Vec<Vec<VReg>>,
    /// Values in registers at each block's exit.
    pub w_exit: Vec<Vec<VReg>>,
    /// Values to reload into a register before each instruction.
    pub reload_before: Vec<Vec<VReg>>,
    /// Values to store to their slot before each instruction.
    pub spill_before: Vec<Vec<VReg>>,
    /// Coupling code (§4.3): what an edge must reload to make its successor's entry
    /// set true. Keyed by `(predecessor, successor)`.
    pub edge_reload: HashMap<(Block, Block), Vec<VReg>, RandomState>,
    /// Coupling code: what an edge must store, so that a value counted as already
    /// spilled at the successor really is spilled on this path too.
    pub edge_spill: HashMap<(Block, Block), Vec<VReg>, RandomState>,
}

impl SpillPlan {
    fn new(f: &impl RegallocFunc) -> Self {
        SpillPlan {
            w_entry: vec![Vec::new(); f.num_blocks()],
            w_exit: vec![Vec::new(); f.num_blocks()],
            reload_before: vec![Vec::new(); f.num_insts()],
            spill_before: vec![Vec::new(); f.num_insts()],
            edge_reload: HashMap::default(),
            edge_spill: HashMap::default(),
        }
    }

    /// Whether `v` is in a register at the entry of `b`.
    pub fn in_reg_at_entry(&self, b: Block, v: VReg) -> bool {
        self.w_entry[b.0 as usize].contains(&v)
    }

    /// Whether `v` is in a register at the exit of `b`.
    pub fn in_reg_at_exit(&self, b: Block, v: VReg) -> bool {
        self.w_exit[b.0 as usize].contains(&v)
    }

    /// Every reload the plan calls for, wherever it sits. For measurement.
    pub fn total_reloads(&self) -> usize {
        self.reload_before.iter().map(Vec::len).sum::<usize>()
            + self.edge_reload.values().map(Vec::len).sum::<usize>()
    }

    /// Every store the plan calls for.
    pub fn total_spills(&self) -> usize {
        self.spill_before.iter().map(Vec::len).sum::<usize>()
            + self.edge_spill.values().map(Vec::len).sum::<usize>()
    }
}

/// Distances from each point in a block to the next use of each value.
///
/// `d[j][v]` is measured from just before the block's `j`th instruction, and
/// `d[len]` is measured from the block's exit — which is where the global analysis
/// takes over, so a value with no further use in this block still compares sensibly
/// against one that has.
struct BlockDistances {
    d: Vec<Vec<u32>>,
}

impl BlockDistances {
    fn build(f: &impl RegallocFunc, b: Block, nu: &NextUse, nv: usize) -> Self {
        let insts = f.block_insts(b);
        let mut d = vec![vec![INF; nv]; insts.len() + 1];
        d[insts.len()].clone_from(&nu.exit[b.0 as usize]);

        // A jump argument is read by the edge, which runs at the terminator.
        if let Some(last) = insts.len().checked_sub(1) {
            for &a in f.jump_args(b) {
                d[last][a.0 as usize] = 0;
            }
        }

        for j in (0..insts.len()).rev() {
            let i = insts[j];
            let (before, after) = d.split_at_mut(j + 1);
            for (slot, &next) in before[j].iter_mut().zip(after[0].iter()) {
                // Already zero means a jump argument read by the edge at this
                // instruction; leave it.
                if *slot != 0 {
                    *slot = next.saturating_add(1);
                }
            }
            for o in f.uses(i) {
                d[j][o.vreg.0 as usize] = 0;
            }
        }
        BlockDistances { d }
    }

    fn at(&self, j: usize, v: VReg) -> u32 {
        self.d[j][v.0 as usize]
    }
}

/// Sort `w` by increasing next-use distance measured from point `j`.
///
/// Ties break on value number so a plan is reproducible; a spiller that shuffled
/// under equal distances would make every downstream measurement noisy.
fn sort_by_distance(w: &mut [VReg], dist: &BlockDistances, j: usize) {
    w.sort_by_key(|&v| (dist.at(j, v), v.0));
}

/// The paper's `limit`: shrink `W` to at most `m` values, storing whatever leaves
/// and is not already in memory.
///
/// `S` records what has already been stored. On SSA form a value has one
/// definition, so it need be stored at most once however often it is evicted —
/// which is why `S` is worth keeping rather than recomputing.
#[allow(clippy::too_many_arguments)]
fn limit(
    f: &impl RegallocFunc,
    w: &mut Vec<VReg>,
    s: &mut Vec<VReg>,
    dist: &BlockDistances,
    j: usize,
    m: usize,
    at: Inst,
    plan: &mut SpillPlan,
) {
    if w.len() <= m {
        return;
    }
    sort_by_distance(w, dist, j);
    for &v in &w[m..] {
        // Nothing to store if it is already in memory, if it is never read again —
        // the value is simply dead — or if it will be replayed rather than reloaded.
        if !s.contains(&v) && dist.at(j, v) != INF && f.remat(v).is_none() {
            plan.spill_before[at].push(v);
            s.push(v);
        }
        // It is out of registers now, so a later eviction must not store it again;
        // but it stays in `s` if it was stored, since the slot still holds it.
    }
    w.truncate(m);
}

/// Algorithm 1, over one block, for one register class.
#[allow(clippy::too_many_arguments)]
fn min_algorithm(
    f: &impl RegallocFunc,
    b: Block,
    class: RegClass,
    k: usize,
    dist: &BlockDistances,
    w: &mut Vec<VReg>,
    s: &mut Vec<VReg>,
    plan: &mut SpillPlan,
) {
    let insts = f.block_insts(b);
    for (j, &i) in insts.iter().enumerate() {
        // Operands not in registers have to be brought back.
        let mut reloads: Vec<VReg> = Vec::new();
        for o in f.uses(i) {
            let v = o.vreg;
            if f.class(v) == class && !w.contains(&v) && !reloads.contains(&v) {
                reloads.push(v);
            }
        }
        for &v in &reloads {
            w.push(v);
            // A value being reloaded was necessarily stored earlier, so the slot is
            // live and a later eviction need not store it again.
            if !s.contains(&v) {
                s.push(v);
            }
        }

        // Room for the operands, measured from this instruction...
        limit(f, w, s, dist, j, k, i, plan);

        // ...then room for the results, measured from the *next* instruction,
        // because once this one writes its results its own operands stop mattering.
        // Getting this second call wrong is what makes defs collide with uses.
        let ndefs = f
            .defs(i)
            .iter()
            .chain(f.temps(i))
            .filter(|o| f.class(o.vreg) == class)
            .count();
        limit(f, w, s, dist, j + 1, k.saturating_sub(ndefs), i, plan);

        for o in f.defs(i) {
            if f.class(o.vreg) == class && !w.contains(&o.vreg) {
                w.push(o.vreg);
            }
        }

        plan.reload_before[i].extend_from_slice(&reloads);
    }
}

/// `W_entry` for an ordinary block (§4.2, `initUsual`).
///
/// Values in registers on *every* incoming edge are taken unconditionally — no
/// edge has to do anything for them. Values in registers on only some edges compete
/// for what room is left, nearest use first.
fn init_usual(
    preds: &[Block],
    w_exit: &[Vec<VReg>],
    nu: &NextUse,
    b: Block,
    k: usize,
    processed: &[bool],
) -> Vec<VReg> {
    let mut freq: HashMap<VReg, usize, RandomState> = HashMap::default();
    let seen: Vec<Block> = preds
        .iter()
        .copied()
        .filter(|p| processed[p.0 as usize])
        .collect();
    for &p in &seen {
        for &v in &w_exit[p.0 as usize] {
            *freq.entry(v).or_insert(0) += 1;
        }
    }

    let mut take: Vec<VReg> = Vec::new();
    let mut cand: Vec<VReg> = Vec::new();
    for (&v, &n) in &freq {
        // A value a predecessor still had in a register but that is dead here must
        // not take a slot. `W` is filled to `k`, so every dead value admitted evicts
        // a live one, which then reloads — on `mix`, whose peak pressure of 19 fits
        // the 20-register file outright, this alone produced 52 reloads against 0
        // stores. Nothing in the paper says this because its `W_exit` is implicitly
        // live; ours retains whatever was in registers when the block ended.
        if nu.at_entry(b, v) == INF {
            continue;
        }
        if n == seen.len() {
            take.push(v);
        } else {
            cand.push(v);
        }
    }
    take.sort_by_key(|v| v.0);
    cand.sort_by_key(|&v| (nu.at_entry(b, v), v.0));

    take.truncate(k);
    let room = k - take.len();
    take.extend(cand.into_iter().take(room));
    take
}

/// `W_entry` for a loop header (§4.2, `initLoopHeader`).
///
/// Deliberately ignores the predecessors. The point of Fig. 2a→2b is that a value
/// spilled *before* the loop and used *inside* it must be counted as in a register
/// at the header, so that the reload lands on the edge into the loop rather than in
/// its body where it would run every iteration.
///
/// Values that merely live *through* the loop without being used in it are the
/// opposite case (Fig. 2c): admitting one costs a reload on the back edge, executed
/// every iteration, to serve a single use after the loop. They are admitted only if
/// the loop looks like it has room to carry them — estimated from the loop's peak
/// pressure, per the paper.
fn init_loop_header(
    f: &impl RegallocFunc,
    layout: &Layout,
    nu: &NextUse,
    b: Block,
    class: RegClass,
    k: usize,
) -> Vec<VReg> {
    // Live-in values plus the block's own parameters — the paper's `I_B`, which is
    // "live-in at B as well as defined by φ-functions in B".
    let mut alive: Vec<VReg> = (0..f.num_vregs() as u32)
        .map(VReg)
        .filter(|&v| f.class(v) == class && nu.live_in(b, v))
        .collect();
    for &p in f.block_params(b) {
        if f.class(p) == class && !alive.contains(&p) {
            alive.push(p);
        }
    }

    let used = used_in_loop(f, layout, b, &alive);
    let mut cand: Vec<VReg> = alive.iter().copied().filter(|v| used.contains(v)).collect();
    let mut live_through: Vec<VReg> = alive
        .iter()
        .copied()
        .filter(|v| !used.contains(v))
        .collect();

    cand.sort_by_key(|&v| (nu.at_entry(b, v), v.0));
    if cand.len() >= k {
        cand.truncate(k);
        return cand;
    }

    // Room left over. `p_L - |T_B|` estimates the pressure from values the loop
    // actually uses, so `k - that` is how many live-through values can plausibly
    // survive the loop without being evicted.
    //
    // That estimate counts only values live *across* instructions, and none of the
    // registers an instruction needs for its own results. Admit right up to it and
    // the first `limit(.., k - |defs|)` inside the loop tips over `k` and evicts one
    // of the very values just admitted — which then reloads on the back edge, every
    // iteration, exactly the case Fig. 2d warns about. On mix2's inner loop the
    // paper's figure admitted 13 live-through values into 20 registers against a
    // working set of 7, and one came straight back out.
    //
    // So keep back what the widest instruction in the loop needs. The paper's
    // footnote 6 already concedes this estimate "might be an under-approximation";
    // this is the part of the shortfall that is cheap to see.
    let p_l = nu.loop_pressure(layout, b, class);
    let headroom = loop_headroom(f, layout, b, class);
    let free = (k + live_through.len()).saturating_sub(p_l as usize + headroom);
    let room = free.min(k.saturating_sub(cand.len()));
    live_through.sort_by_key(|&v| (nu.at_entry(b, v), v.0));
    cand.extend(live_through.into_iter().take(room));
    cand
}

/// The most registers any single instruction in the loop needs for its own results
/// — its definitions and temps, which have to be somewhere the moment it executes
/// and cannot share with anything live across it.
fn loop_headroom(f: &impl RegallocFunc, layout: &Layout, b: Block, class: RegClass) -> usize {
    let mut worst = 0;
    for blk in 0..f.num_blocks() {
        let blk = Block(blk as u32);
        if !in_loop(layout, blk, b) {
            continue;
        }
        for &i in f.block_insts(blk) {
            let n = f
                .defs(i)
                .iter()
                .chain(f.temps(i))
                .filter(|o| f.class(o.vreg) == class)
                .count();
            worst = worst.max(n);
        }
    }
    worst
}

/// Which of `vals` are read anywhere inside the loop headed by `b`.
fn used_in_loop(f: &impl RegallocFunc, layout: &Layout, b: Block, vals: &[VReg]) -> Vec<VReg> {
    let mut out = Vec::new();
    for blk in 0..f.num_blocks() {
        let blk = Block(blk as u32);
        if !in_loop(layout, blk, b) {
            continue;
        }
        for &i in f.block_insts(blk) {
            for o in f.uses(i) {
                if vals.contains(&o.vreg) && !out.contains(&o.vreg) {
                    out.push(o.vreg);
                }
            }
        }
        for &a in f.jump_args(blk) {
            if vals.contains(&a) && !out.contains(&a) {
                out.push(a);
            }
        }
    }
    out
}

/// Run the spiller over `f`, lowering register pressure to the size of each class's
/// pool.
pub fn plan(f: &impl RegallocFunc, layout: &Layout, nu: &NextUse, env: &MachineEnv) -> SpillPlan {
    let nb = f.num_blocks();
    let nv = f.num_vregs();
    let mut plan = SpillPlan::new(f);

    let succs: Vec<Vec<Block>> = (0..nb).map(|b| f.succs(Block(b as u32))).collect();
    let mut preds: Vec<Vec<Block>> = vec![Vec::new(); nb];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s.0 as usize].push(Block(b as u32));
        }
    }

    for class in RegClass::ALL {
        let k = env.order(class).len();
        if k == 0 {
            continue;
        }

        // `W` and `S` at each block's exit, for this class.
        let mut w_exit: Vec<Vec<VReg>> = vec![Vec::new(); nb];
        let mut s_exit: Vec<Vec<VReg>> = vec![Vec::new(); nb];
        // Kept because the back-edge fix-up below needs each block's *entry* `S`,
        // long after the forward pass has moved on.
        let mut s_entry: Vec<Vec<VReg>> = vec![Vec::new(); nb];
        let mut processed = vec![false; nb];

        for &b in &layout.order {
            let bi = b.0 as usize;
            let dist = BlockDistances::build(f, b, nu, nv);

            let inherited = if layout.is_header(b) {
                init_loop_header(f, layout, nu, b, class, k)
            } else {
                init_usual(&preds[bi], &w_exit, nu, b, k, &processed)
            };

            // Block parameters go in first. A parameter is defined at the entry and
            // is usually read soon after, so it is the last thing worth evicting;
            // appending it and truncating instead dropped one on `mix2` and then
            // reloaded it from a slot the edge had never written. Nearest use first,
            // measured inside this block, since a parameter's distance *from* the
            // entry is meaningless — it is born there.
            let mut params: Vec<VReg> = f
                .block_params(b)
                .iter()
                .copied()
                .filter(|&p| f.class(p) == class)
                .collect();
            params.sort_by_key(|&p| (dist.at(0, p), p.0));

            let mut w: Vec<VReg> = params.iter().copied().take(k).collect();
            for v in inherited {
                if w.len() >= k {
                    break;
                }
                if !w.contains(&v) {
                    w.push(v);
                }
            }

            // `S` invariant: v is in `S` at a point iff it was spilled on *every*
            // path to that point. Union over predecessors, then narrowed to what is
            // actually in registers here.
            let mut s: Vec<VReg> = Vec::new();
            for &p in preds[bi].iter().filter(|p| processed[p.0 as usize]) {
                for &v in &s_exit[p.0 as usize] {
                    if !s.contains(&v) {
                        s.push(v);
                    }
                }
            }
            s.retain(|v| w.contains(v));

            plan.w_entry[bi].extend(w.iter().copied());
            s_entry[bi] = s.clone();

            // Coupling code, per §4.3. A predecessor not yet processed is a back
            // edge; the paper says to skip it and fix it up once that block has been
            // seen, which the second pass below does.
            for &p in &preds[bi] {
                if !processed[p.0 as usize] {
                    continue;
                }
                couple(
                    f,
                    nu,
                    class,
                    p,
                    b,
                    &w,
                    &s,
                    &w_exit[p.0 as usize],
                    &s_exit[p.0 as usize],
                    &mut plan,
                );
            }

            min_algorithm(f, b, class, k, &dist, &mut w, &mut s, &mut plan);

            // Values dead at the exit are holding nothing anyone will read. Leaving
            // them in `W` would carry them into successors' entry sets, where the
            // fill-to-`k` would spend real registers on them.
            //
            // A jump argument is the exception: it is read by the edge itself, so it
            // is dead *after* the exit yet must be in a register *at* it. Dropping it
            // makes every block parameter look absent from its predecessor, and the
            // coupling code then reloads all of them — 50 of mix's edge reloads.
            let args = f.jump_args(b);
            w.retain(|&v| nu.at_exit(b, v) != INF || args.contains(&v));
            s.retain(|v| w.contains(v));

            plan.w_exit[bi].extend(w.iter().copied());
            w_exit[bi] = w;
            s_exit[bi] = s;
            processed[bi] = true;
        }

        // Back edges, now that their sources have been processed. Braun09 §4.3 says
        // to skip an unprocessed predecessor and add its coupling code once that
        // block has been seen; this is that.
        let mut index = vec![usize::MAX; nb];
        for (k, &b) in layout.order.iter().enumerate() {
            index[b.0 as usize] = k;
        }
        for &b in &layout.order {
            let bi = b.0 as usize;
            let w: Vec<VReg> = plan.w_entry[bi]
                .iter()
                .copied()
                .filter(|&v| f.class(v) == class)
                .collect();
            let s: Vec<VReg> = s_entry[bi].clone();
            for &p in &preds[bi] {
                // Only the edges skipped above: those whose source comes no earlier
                // in the layout, which is what a back edge is.
                if index[p.0 as usize] < index[bi] {
                    continue;
                }
                couple(
                    f,
                    nu,
                    class,
                    p,
                    b,
                    &w,
                    &s,
                    &w_exit[p.0 as usize],
                    &s_exit[p.0 as usize],
                    &mut plan,
                );
            }
        }
    }

    plan
}

/// Coupling code for one edge (§4.3), plus the case the paper's two rules miss.
///
/// Shared by the forward pass and the back-edge fix-up deliberately: they are the
/// same computation, and keeping two copies of it in step is what went wrong twice
/// while this was being written.
#[allow(clippy::too_many_arguments)]
fn couple(
    f: &impl RegallocFunc,
    nu: &NextUse,
    class: RegClass,
    p: Block,
    b: Block,
    w: &[VReg],
    s: &[VReg],
    wp: &[VReg],
    sp: &[VReg],
    plan: &mut SpillPlan,
) {
    // A block parameter is never reloaded from a slot: it is *delivered* by the
    // edge, from the argument the predecessor passes. What it needs is for that
    // argument to be in a register at the branch; where it travels from is edge
    // resolution's business, not the spiller's.
    let params: Vec<VReg> = f
        .block_params(b)
        .iter()
        .copied()
        .filter(|&v| f.class(v) == class)
        .collect();
    let reload: Vec<VReg> = w
        .iter()
        .copied()
        .filter(|v| !wp.contains(v) && !params.contains(v))
        .collect();

    // The paper's rule: a value this block treats as already in memory, because
    // some other path spilled it, must really be in memory on this path too.
    let mut store: Vec<VReg> = s
        .iter()
        .copied()
        .filter(|v| !sp.contains(v) && wp.contains(v))
        .collect();

    // And the case §4.3 does not cover. A value the predecessor held in a register,
    // that is still live, and that `b` has no room for is being evicted *by the
    // block boundary itself* — it leaves registers without passing through `limit`,
    // so nothing has stored it, and reading it later would read a slot no one ever
    // wrote.
    for &v in wp {
        if !w.contains(&v) && !sp.contains(&v) && !store.contains(&v) && nu.at_entry(b, v) != INF {
            store.push(v);
        }
    }

    // A parameter with no room in `W` lives in its slot, and the only thing that
    // ever writes that slot is this edge — the argument goes to memory rather than
    // to a register. Without this the parameter is read from a slot nobody wrote.
    for &prm in &params {
        if !w.contains(&prm) && !store.contains(&prm) {
            store.push(prm);
        }
    }

    if !reload.is_empty() {
        plan.edge_reload.entry((p, b)).or_default().extend(reload);
    }
    if !store.is_empty() {
        plan.edge_spill.entry((p, b)).or_default().extend(store);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::nextuse;
    use crate::jit::backend::order;
    use crate::jit::backend::regalloc::{Operand, PReg};
    use crate::jit::backend::testfunc::TestFunc;

    /// A machine with `n` integer registers and no float ones, so `k` is exactly `n`
    /// and the arithmetic in the tests is the paper's arithmetic.
    fn env(n: u8) -> MachineEnv {
        MachineEnv {
            allocation_order: [
                (0..n).map(|i| PReg::new(RegClass::Int, i)).collect(),
                Vec::new(),
            ],
        }
    }

    fn plan_of(f: &TestFunc, n: u8) -> SpillPlan {
        let layout = order::compute(f).expect("reducible");
        let nu = nextuse::analyze(f, &layout);
        plan(f, &layout, &nu, &env(n))
    }

    /// The property the whole pass exists to establish: nowhere does the plan ask
    /// for more values in registers than the machine has. Everything downstream —
    /// the claim that assignment never has to spill — rests on this.
    fn assert_within_k(f: &TestFunc, p: &SpillPlan, k: usize) {
        for b in 0..f.num_blocks() {
            assert!(
                p.w_entry[b].len() <= k,
                "mb{b} enters with {} values in registers, k = {k}",
                p.w_entry[b].len()
            );
            assert!(
                p.w_exit[b].len() <= k,
                "mb{b} exits with {} values in registers, k = {k}",
                p.w_exit[b].len()
            );
        }
    }

    /// Belady's rule itself: under pressure, the value whose next use is furthest
    /// away is the one that goes to memory.
    #[test]
    fn the_furthest_next_use_is_the_one_evicted() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (soon, later, third) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(soon)], vec![]);
        f.inst(b, vec![Operand::any(later)], vec![]);
        // With k = 2 this third definition forces one of the two out.
        f.inst(b, vec![Operand::any(third)], vec![]);
        f.inst(b, vec![], vec![Operand::any(third)]);
        f.inst(b, vec![], vec![Operand::any(soon)]);
        f.inst(b, vec![], vec![Operand::any(later)]);

        let p = plan_of(&f, 2);
        assert_within_k(&f, &p, 2);

        let stored: Vec<VReg> = p.spill_before.iter().flatten().copied().collect();
        assert!(
            stored.contains(&later),
            "`later` is read last, so it is the one to evict; stored {stored:?}"
        );
        assert!(
            !stored.contains(&soon),
            "`soon` is read before `later` and must keep its register"
        );
    }

    /// A value already in memory before a loop, and used inside it, must be counted
    /// as in a register at the loop header — so the reload lands on the edge *into*
    /// the loop instead of in the body, where it would run every iteration. This is
    /// the paper's Fig. 2a versus 2b, and is the reason loop headers ignore their
    /// predecessors.
    #[test]
    fn a_reload_for_a_loop_use_is_hoisted_out_of_the_loop() {
        let mut f = TestFunc::default();
        let (entry, header, body, exit) = (f.block(), f.block(), f.block(), f.block());
        let (x, filler) = (f.int(), f.int());

        // `x` is defined early, then enough other values are defined and used that
        // `x` is evicted before the loop is reached.
        f.inst(entry, vec![Operand::any(x)], vec![]);
        f.inst(entry, vec![Operand::any(filler)], vec![]);
        f.inst(entry, vec![], vec![Operand::any(filler)]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        f.inst(body, vec![], vec![Operand::any(x)]);
        f.inst(body, vec![], vec![]);
        f.goto(body, &[header]);

        f.inst(exit, vec![], vec![]);

        let p = plan_of(&f, 2);
        assert_within_k(&f, &p, 2);

        assert!(
            p.in_reg_at_entry(header, x),
            "the loop header must claim `x`, so that the reload is forced onto the \
             edge into the loop rather than into the body"
        );
        let in_body: Vec<VReg> = f
            .block_insts(body)
            .iter()
            .flat_map(|&i| p.reload_before[i].iter().copied())
            .collect();
        assert!(
            !in_body.contains(&x),
            "no reload of `x` may sit inside the loop body; found {in_body:?}"
        );
    }

    /// The opposite case, Fig. 2c/2d. A value that lives *through* a loop without
    /// being used in it should not be forced into a register at the header when the
    /// loop has no room for it: keeping it there costs a reload on the back edge,
    /// executed every iteration, to serve a single use after the loop.
    #[test]
    fn a_live_through_value_is_not_forced_to_survive_a_crowded_loop() {
        let mut f = TestFunc::default();
        let (entry, header, body, exit) = (f.block(), f.block(), f.block(), f.block());
        let through = f.int();
        let a = f.int();
        let b = f.int();

        f.inst(entry, vec![Operand::any(through)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        // The loop uses two values of its own, filling a two-register machine, so
        // `through` cannot also survive in a register.
        f.inst(body, vec![Operand::any(a)], vec![]);
        f.inst(body, vec![Operand::any(b)], vec![]);
        f.inst(body, vec![], vec![Operand::any(a), Operand::any(b)]);
        f.inst(body, vec![], vec![]);
        f.goto(body, &[header]);

        // `through` is read only here, after the loop.
        f.inst(exit, vec![], vec![Operand::any(through)]);

        let p = plan_of(&f, 2);
        assert_within_k(&f, &p, 2);

        assert!(
            !p.in_reg_at_entry(header, through),
            "a value the loop never reads must not be held in a register across a \
             loop that has no room for it — that buys a back-edge reload every \
             iteration to serve one use after the loop"
        );
    }

    /// A value used inside the loop wins the register over one that is not, whatever
    /// the raw instruction counts say. This is the loop-exit edge length doing its
    /// job through the spiller.
    #[test]
    fn the_loop_prefers_the_value_it_actually_uses() {
        let mut f = TestFunc::default();
        let (entry, header, body, exit) = (f.block(), f.block(), f.block(), f.block());
        let (used, unused) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(used)], vec![]);
        f.inst(entry, vec![Operand::any(unused)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        for _ in 0..10 {
            f.inst(body, vec![], vec![]);
        }
        f.inst(body, vec![], vec![Operand::any(used)]);
        f.goto(body, &[header]);

        f.inst(exit, vec![], vec![Operand::any(unused)]);

        // One register: exactly one of the two can be held.
        let p = plan_of(&f, 1);
        assert_within_k(&f, &p, 1);

        assert!(
            p.in_reg_at_entry(header, used),
            "the loop's own value takes the register"
        );
        assert!(
            !p.in_reg_at_entry(header, unused),
            "the post-loop value does not"
        );
    }

    /// A value is stored at most once however often it is evicted — SSA gives it one
    /// definition, so one slot write suffices and the rest are redundant.
    #[test]
    fn a_value_is_stored_at_most_once() {
        let mut f = TestFunc::default();
        let b = f.block();
        let victim = f.int();
        let others: Vec<VReg> = (0..4).map(|_| f.int()).collect();

        f.inst(b, vec![Operand::any(victim)], vec![]);
        // Repeatedly fill and drain the register file, reading `victim` in between
        // so it is reloaded and then evicted again.
        for &o in &others {
            f.inst(b, vec![Operand::any(o)], vec![]);
            f.inst(b, vec![], vec![Operand::any(o)]);
            f.inst(b, vec![], vec![Operand::any(victim)]);
        }

        let p = plan_of(&f, 2);
        assert_within_k(&f, &p, 2);

        let stores = p
            .spill_before
            .iter()
            .flatten()
            .filter(|&&v| v == victim)
            .count();
        assert!(
            stores <= 1,
            "one definition means one store, got {stores} — S is not doing its job"
        );
    }

    /// A dead value is never stored: there is no one left to read the slot.
    #[test]
    fn a_dead_value_is_evicted_without_being_stored() {
        let mut f = TestFunc::default();
        let b = f.block();
        let dead = f.int();
        let (a, c) = (f.int(), f.int());

        f.inst(b, vec![Operand::any(dead)], vec![]); // never read again
        f.inst(b, vec![Operand::any(a)], vec![]);
        f.inst(b, vec![Operand::any(c)], vec![]);
        f.inst(b, vec![], vec![Operand::any(a), Operand::any(c)]);

        let p = plan_of(&f, 2);
        let stored: Vec<VReg> = p.spill_before.iter().flatten().copied().collect();
        assert!(
            !stored.contains(&dead),
            "a value with no further use must be dropped, not stored"
        );
    }

    /// The spiller, on real functions.
    ///
    /// Three properties, each of which caught a real bug while this was being written:
    ///
    /// - **Pressure is within the register file everywhere.** Every claim downstream —
    ///   above all that assignment can then proceed without spilling at all — rests on
    ///   this.
    /// - **Nothing is reloaded that was never stored.** A reload reads a slot; if no
    ///   store ever wrote that slot it reads garbage. This is the invariant that
    ///   exposed dead values occupying `W`, jump arguments being dropped at block
    ///   exits, and block parameters being reloaded rather than delivered by the edge.
    /// - **A function that fits needs no spill code.** `is_prime` and `mix` both peak
    ///   below the register file, so a correct spiller must leave them completely
    ///   alone — and does, which is a claim about the whole pipeline agreeing with an
    ///   allocator that reached the same answer by an entirely different route.
    #[test]
    fn the_spill_plan_is_within_k_and_coherent() {
        use crate::Lua;
        use crate::jit::backend::isel::select;
        use crate::jit::backend::regalloc::allocate;
        use crate::jit::backend::target::{annotate, machine_env};
        use crate::jit::backend::{nextuse, order, spillcost};
        use crate::jit::frontend::lower::lower;
        use crate::jit::ir::ty::{Rep, Ty, TypeSet};

        const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

        eprintln!("\n=== spill plan ===");
        for (file, chunk) in [("primes", "primes"), ("mix", "mix"), ("mix2", "mix2")] {
            let source = std::fs::read_to_string(format!("test-files/{file}.lua")).unwrap();
            let mut lua = Lua::new();
            lua.load_all();
            lua.enter(|ctx| {
                let c = ctx.load(&source, Some(chunk)).expect("compile");
                let proto = c.as_lua().expect("lua closure").proto.prototypes[0];
                let func = lower(proto, 0, vec![INT]).expect("lower");
                let mut m = select(&func).expect("isel");
                annotate(&mut m);

                let env = machine_env();
                let layout = order::compute(&m).expect("reducible");
                let nu = nextuse::analyze(&m, &layout);
                let plan = plan(&m, &layout, &nu, &env);

                for class in RegClass::ALL {
                    let k = env.order(class).len();
                    for b in 0..m.num_blocks() {
                        for (what, set) in [("entry", &plan.w_entry[b]), ("exit", &plan.w_exit[b])]
                        {
                            let n = set.iter().filter(|&&v| m.class(v) == class).count();
                            assert!(
                                n <= k,
                                "{file} mb{b} {what}: {n} {class:?} values in registers, k = {k}"
                            );
                        }
                    }
                }

                let stored: Vec<VReg> = plan
                    .spill_before
                    .iter()
                    .flatten()
                    .chain(plan.edge_spill.values().flatten())
                    .copied()
                    .collect();
                for v in plan
                    .reload_before
                    .iter()
                    .flatten()
                    .chain(plan.edge_reload.values().flatten())
                {
                    assert!(
                        stored.contains(v) || m.remat(*v).is_some(),
                        "{file}: v{} is reloaded but never stored — the slot it reads \
                         was never written",
                        v.0
                    );
                }

                // The plan's own cost, weighted the same way `spillcost` weights the
                // allocator's output, so the two are comparable. Two conventions have to
                // match for that: an edge's code lands in the predecessor, so it is
                // charged at the predecessor's depth; and a rematerialized value is
                // replayed rather than loaded, so like `spillcost` it is counted apart
                // from memory traffic rather than as some of it.
                let mem = |v: &&VReg| m.remat(**v).is_none();
                let ops = plan
                    .reload_before
                    .iter()
                    .chain(&plan.spill_before)
                    .chain(plan.edge_reload.values())
                    .chain(plan.edge_spill.values())
                    .flatten()
                    .filter(mem)
                    .count();
                let mut weighted = 0u64;
                let mut depth_of_inst = vec![0u32; m.num_insts()];
                for &b in &layout.order {
                    for &i in m.block_insts(b) {
                        depth_of_inst[i] = layout.depth[b.0 as usize];
                    }
                }
                for (i, (r, sp)) in plan
                    .reload_before
                    .iter()
                    .zip(&plan.spill_before)
                    .enumerate()
                {
                    let n = r.iter().chain(sp).filter(mem).count();
                    weighted += n as u64 * 10u64.pow(depth_of_inst[i]);
                }
                for (&(p, _), vs) in plan.edge_reload.iter().chain(plan.edge_spill.iter()) {
                    let n = vs.iter().filter(mem).count();
                    weighted += n as u64 * 10u64.pow(layout.depth[p.0 as usize]);
                }

                // Fig. 2d, as a property. A value the loop never reads, that the
                // header nonetheless chose to keep in a register, must not then be
                // evicted *inside* the loop: that buys a store in the body and a
                // reload on the back edge, both executed every iteration, to serve a
                // use after the loop. It is strictly worse than never admitting it.
                //
                // This is what the `p_L` estimate gets wrong when it is not given
                // headroom for instruction results, and it cost mix2 two operations
                // in its innermost loop — 200 of 1759 weighted, enough on its own to
                // turn the whole aarch64 result from a 7.7% win into a 2% loss.
                for h in 0..m.num_blocks() {
                    let h = Block(h as u32);
                    if !layout.is_header(h) {
                        continue;
                    }
                    let used = used_in_loop(&m, &layout, h, &plan.w_entry[h.0 as usize]);
                    for &v in &plan.w_entry[h.0 as usize] {
                        if used.contains(&v) {
                            continue;
                        }
                        for b in 0..m.num_blocks() {
                            let b = Block(b as u32);
                            if !in_loop(&layout, b, h) {
                                continue;
                            }
                            for &i in m.block_insts(b) {
                                assert!(
                                    !plan.spill_before[i].contains(&v),
                                    "{file}: v{} is held in a register at loop header \
                                     mb{}, is never read in that loop, and is then \
                                     spilled inside it at inst {i}",
                                    v.0,
                                    h.0
                                );
                            }
                        }
                    }
                }

                let now = spillcost::measure(&m, &allocate(&m, &env).expect("allocate"))
                    .expect("reducible");
                eprintln!(
                    "  {file:6} plan {} ops (weighted {weighted})   vs today {} ops \
                     (weighted {})",
                    ops,
                    now.total(),
                    now.weighted,
                );

                // A function whose peak pressure fits must come out untouched. Keyed
                // on the measured peak rather than on which benchmark it is, because
                // that differs by target: mix peaks at 19, which fits aarch64's 20
                // registers and does not fit x86-64's 13.
                let fits = RegClass::ALL.iter().all(|&c| {
                    let k = env.order(c).len() as u32;
                    nu.pressure.iter().all(|p| p[c as usize] <= k)
                });
                if fits {
                    assert_eq!(
                        ops, 0,
                        "{file} peaks below the register file, so it needs no spill code"
                    );
                }
            });
        }
    }
}
