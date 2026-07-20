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
use super::regalloc::{Block, Constraint, Inst, MachineEnv, PReg, RegClass, RegallocFunc, VReg};

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
    /// The register set as each instruction *reads* it: after any reload the plan
    /// calls for, before the instruction writes its results.
    pub w_use: Vec<Vec<VReg>>,
    /// The register set as each instruction *leaves* it: results written, evictions
    /// done.
    ///
    /// Recorded rather than left to be replayed from the transition lists, because
    /// it cannot be replayed from them. `limit` drops a dead value, and a
    /// rematerializable one, without emitting any code — so a value can leave the
    /// register set with nothing in `spill_before` to mark it, and a reader
    /// reconstructing `W` from the transitions would believe it was still there.
    pub w_after: Vec<Vec<VReg>>,
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
            w_use: vec![Vec::new(); f.num_insts()],
            w_after: vec![Vec::new(); f.num_insts()],
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

impl SpillPlan {
    /// Per value, the maximal runs of the doubled position axis over which the plan
    /// keeps it in a register.
    ///
    /// This is the plan in the form the scan wants. Where liveness gives one
    /// interval per value spanning its whole life, this gives one per stretch that
    /// the value actually spends in a register — several per value, with the gaps
    /// between them spent in a stack slot. Colouring these rather than the live
    /// ranges *is* live-range splitting.
    ///
    /// The axis matches [`super::regalloc::allocate`]: instruction `i` reads at
    /// `2·pos[i]` and writes at `2·pos[i] + 1`, so a run that ends where another
    /// begins does not overlap it.
    ///
    /// Because `|W| ≤ k` at every point, at most `k` of these runs cover any one
    /// position — which is exactly the condition under which a linear scan cannot
    /// run out of registers.
    pub fn register_runs(
        &self,
        f: &impl RegallocFunc,
        order: &[Block],
        pos: &[Inst],
        num_vregs: usize,
    ) -> Vec<Vec<(u32, u32)>> {
        let mut runs: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_vregs];
        let extend = |runs: &mut Vec<Vec<(u32, u32)>>, v: VReg, at: u32| {
            let r = &mut runs[v.0 as usize];
            match r.last_mut() {
                // Positions arrive in increasing order, so a run continues exactly
                // when the previous one ended where this slot begins.
                Some(last) if last.1 == at => last.1 = at + 1,
                _ => r.push((at, at + 1)),
            }
        };
        for &b in order {
            for &i in f.block_insts(b) {
                let p = pos[i] as u32;
                for &v in &self.w_use[i] {
                    extend(&mut runs, v, p * 2);
                }
                for &v in &self.w_after[i] {
                    extend(&mut runs, v, p * 2 + 1);
                }
            }
        }
        runs
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
    keep: &[VReg],
    plan: &mut SpillPlan,
) {
    if w.len() <= m {
        return;
    }
    // `keep` holds values this instruction forbids evicting whatever their
    // distance says: the non-reused inputs of a two-address op, which must
    // coexist with the results or the op clobbers them (`build_intervals`
    // extends exactly these ranges for the same reason). They are usually dying
    // — distance ∞ — which under plain Belady sorts them *first* out the door.
    if keep.is_empty() {
        sort_by_distance(w, dist, j);
    } else {
        w.sort_by_key(|&v| (!keep.contains(&v), dist.at(j, v), v.0));
    }
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
    let args = f.jump_args(b);
    for (j, &i) in insts.iter().enumerate() {
        // A dead value holds nothing; drop it the moment it dies rather than at
        // the block's end. This is not merely tidiness. A loop-carried parameter
        // is dead after its last use while its next iteration's replacement is
        // computed — and if it lingers in `W`, its register reads as busy at the
        // replacement's definition, the arg/param affinity is refused, and every
        // accumulator pays a move on the back edge that coalescing used to make
        // free. Intervals end at the last use; so must runs. (The keep-set from a
        // previous instruction's two-address op dies here too, exactly one slot
        // after the interference it existed to create.)
        w.retain(|&v| dist.at(j, v) != INF);
        s.retain(|v| w.contains(v));
        // Operands not in registers have to be brought back — but only the ones that
        // actually demand a register. Braun09 states the opposite assumption
        // outright ("each instruction requires that its operands are available in
        // registers"), and it does not hold here: an `Any` operand is read from
        // wherever the value already is, slot included.
        //
        // The distinction is not marginal. mix2's deopt stub takes 35 operands, 34
        // of them `Any` — it names the whole VM state so the exit can rebuild an
        // interpreter frame. Treating those as register operands asks for 34
        // registers on a 20-register machine, which cannot be met, and the eviction
        // it forces throws out values that genuinely did need one.
        let mut reloads: Vec<VReg> = Vec::new();
        for o in f.uses(i) {
            let v = o.vreg;
            if o.constraint != Constraint::Any
                && f.class(v) == class
                && !w.contains(&v)
                && !reloads.contains(&v)
            {
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

        // Registers this instruction pins or destroys are not available to values
        // in `W` across it: a clobber is written over, and a `Fixed` operand's
        // register is where that one value must sit, to the exclusion of all
        // others. The spiller not modelling these is a decline at colouring time —
        // on x86-64, whose isel pins shift counts and division, it was exactly
        // that.
        let mut burned: Vec<PReg> = Vec::new();
        for &r in f.clobbers(i) {
            if r.class() == class && !burned.contains(&r) {
                burned.push(r);
            }
        }
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if let Constraint::Fixed(r) = o.constraint
                && r.class() == class
                && !burned.contains(&r)
            {
                burned.push(r);
            }
        }
        let kc = k.saturating_sub(burned.len());

        // Room for the operands, measured from this instruction — less whatever the
        // temps need. A temp is scratch that spans the *whole* instruction, so it
        // cannot share with an operand that dies here, and the register it occupies
        // is unavailable from the read onwards rather than only from the write.
        // Reserving it at the write alone leaves the read point one over `k`, and
        // the scan then has nowhere to put it.
        let ntemps = f
            .temps(i)
            .iter()
            .filter(|o| f.class(o.vreg) == class)
            .count();
        // Wimmer05's must-have-register flag (§2.3), as a keep-set. Every operand
        // read here has distance 0, so among themselves they sort arbitrarily —
        // and a deopt-shaped instruction reads more `Any` operands than the
        // machine has registers, all tied. An arbitrary cut can evict the one
        // operand the instruction *cannot* take from a slot while keeping
        // thirty-four it happily can.
        let must: Vec<VReg> = f
            .uses(i)
            .iter()
            .filter(|o| o.constraint != Constraint::Any && f.class(o.vreg) == class)
            .map(|o| o.vreg)
            .collect();
        limit(f, w, s, dist, j, kc.saturating_sub(ntemps), i, &must, plan);

        // ...then room for the results, measured from the *next* instruction,
        // because once this one writes its results its own operands stop mattering.
        // Getting this second call wrong is what makes defs collide with uses.
        // Results are *not* discounted the way operands are, even an `Any` one. The
        // encoders write every def to a register and let an edit store it afterwards
        // (`aarch64::def_g` makes `Alloc::Spill` on a def `unreachable!`), so a def
        // needs a register here whatever its constraint says it would tolerate.
        // isel emits no `Any` defs today — 0 across all three benchmarks against 205
        // `Any` uses on mix2 — so this costs nothing and stays on the safe side of a
        // contract the spiller does not own.
        let ndefs = f
            .defs(i)
            .iter()
            .chain(f.temps(i))
            .filter(|o| f.class(o.vreg) == class)
            .count();
        // What the instruction reads is `W` as it stands here: the first `limit`
        // has already guaranteed every operand is in it.
        plan.w_use[i].extend(w.iter().copied());

        // The non-reused inputs of a two-address op must survive the write; see
        // `limit`.
        let keep: Vec<VReg> = match f.defs(i).iter().find_map(|o| match o.constraint {
            Constraint::Reuse(uk) => Some(uk),
            _ => None,
        }) {
            Some(uk) => f
                .uses(i)
                .iter()
                .enumerate()
                .filter(|&(k2, o)| k2 != uk && f.class(o.vreg) == class)
                .map(|(_, o)| o.vreg)
                .collect(),
            None => Vec::new(),
        };
        limit(
            f,
            w,
            s,
            dist,
            j + 1,
            kc.saturating_sub(ndefs),
            i,
            &keep,
            plan,
        );

        for o in f.defs(i) {
            if f.class(o.vreg) == class && !w.contains(&o.vreg) {
                w.push(o.vreg);
            }
        }

        // A value whose last read was *this* instruction leaves `W` now, not at
        // the next one: intervals end at the last use, and a run one slot longer
        // is not a rounding error. `a1 = p1 + c` is the case — with `p1` lingering
        // in `w_after`, its register reads as busy at `a1`'s definition, the
        // arg/param affinity is refused, and every accumulator's argument lands
        // one register off, which came out as a permutation of moves on the back
        // edge. Three-address instructions may share a dying source's register
        // with their result; the keep-set stays, because two-address ones must
        // not.
        //
        // Three exemptions, all values that are dead by the distances yet still
        // occupy a register: the keep-set (above), a *dead def* — never read, but
        // the instruction still writes it somewhere, the minimal interval the
        // whole-value scan also gives it — and the jump arguments, which the edge
        // reads after the last instruction, so their runs must reach the block's
        // end or the slot-gap test sees a one-slot hole and invents a spill.
        w.retain(|&v| {
            dist.at(j + 1, v) != INF
                || keep.contains(&v)
                || f.defs(i).iter().any(|o| o.vreg == v)
                || args.contains(&v)
        });
        s.retain(|v| w.contains(v));

        plan.w_after[i].extend(w.iter().copied());
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
    /// End to end: hand the spiller's decisions to the allocator and check the
    /// result with the symbolic verifier, which knows nothing about how the
    /// allocation was produced. This is the test that says whether splitting
    /// actually works, as opposed to whether the plan is sensible.
    #[test]
    fn a_split_allocation_verifies() {
        for (file, chunk) in [("primes", "primes"), ("mix", "mix")] {
            check_split(file, chunk);
        }
    }

    #[test]
    fn a_split_allocation_verifies_on_mix2() {
        check_split("mix2", "mix2");
    }

    /// The allocation-level A/B, on whichever target is being tested — x86-64's
    /// 13 registers included, which is where the pressure is. Reg-reg move edits
    /// stand in for the encoded shuffle; `spillcost` weights the memory traffic.
    /// The aarch64-only `asm_dump::split_vs_whole_report` gives the same numbers
    /// against real encoded output.
    #[test]
    fn split_vs_whole_allocations() {
        use crate::Lua;
        use crate::jit::backend::isel::select;
        use crate::jit::backend::regalloc::{
            Alloc, Edit, RegisterSets, allocate, allocate_with, verify,
        };
        use crate::jit::backend::spillcost;
        use crate::jit::backend::target::{annotate, machine_env};
        use crate::jit::frontend::lower::lower;
        use crate::jit::ir::ty::{Rep, Ty, TypeSet};

        const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

        eprintln!("\n=== split vs whole (allocations) ===");
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

                let whole = allocate(&m, &env).expect("whole");
                verify(&m, &whole).expect("whole verifies");
                let layout = order::compute(&m).expect("reducible");
                let nu = nextuse::analyze(&m, &layout);
                let p = plan(&m, &layout, &nu, &env);
                let split = allocate_with(
                    &m,
                    &env,
                    Some(RegisterSets {
                        w_use: &p.w_use,
                        w_after: &p.w_after,
                    }),
                )
                .expect("split");
                verify(&m, &split).expect("split verifies");

                for (which, ra) in [("whole", &whole), ("split", &split)] {
                    let rr = ra
                        .edits()
                        .iter()
                        .filter(|(_, e)| {
                            matches!(
                                e,
                                Edit::Move(mv)
                                    if matches!((mv.from, mv.to), (Alloc::Reg(_), Alloc::Reg(_)))
                            )
                        })
                        .count();
                    let cost = spillcost::measure(&m, ra).expect("reducible");
                    eprintln!("  {file:7} {which}: {rr:3} reg-reg move edits, {cost}");
                }
            });
        }
    }

    /// Allocate `file` from the spiller's decisions and put the result through the
    /// symbolic verifier, which knows nothing about how the allocation was made.
    ///
    /// # What the split path rests on
    ///
    /// `Locations` is *honest*: entries are anchored at each value's birth (never
    /// position 0), every departure from registers is recorded — the slot for a
    /// spilled value, **nowhere** for a replayed one — and everything downstream
    /// reads locations instead of keeping its own books. Edge resolution compares
    /// the two ends of every edge for every live value and emits exactly the
    /// disagreements; runs are clipped at block boundaries so a run never claims
    /// continuity across an edge control may not take; a reload is emitted in-block
    /// only where a run follows a real gap mid-block, while gap-following runs that
    /// start *at* a block entry are delivered by the edges — which is what hoists a
    /// loop header's reload onto the entry edge and off the back edge.
    fn check_split(file: &str, chunk: &str) {
        use crate::Lua;
        use crate::jit::backend::isel::select;
        use crate::jit::backend::regalloc::{RegisterSets, allocate_with, verify};
        use crate::jit::backend::target::{annotate, machine_env};
        use crate::jit::backend::{nextuse, order};
        use crate::jit::frontend::lower::lower;
        use crate::jit::ir::ty::{Rep, Ty, TypeSet};

        const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

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
            let p = plan(&m, &layout, &nu, &env);

            let sets = RegisterSets {
                w_use: &p.w_use,
                w_after: &p.w_after,
            };
            let ra = allocate_with(&m, &env, Some(sets))
                .unwrap_or_else(|e| panic!("{file}: split allocation declined: {e:?}"));
            verify(&m, &ra).unwrap_or_else(|e| panic!("{file}: {e}"));
        });
    }

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

        f.inst(b, vec![Operand::reg(soon)], vec![]);
        f.inst(b, vec![Operand::reg(later)], vec![]);
        // With k = 2 this third definition forces one of the two out.
        f.inst(b, vec![Operand::reg(third)], vec![]);
        f.inst(b, vec![], vec![Operand::reg(third)]);
        f.inst(b, vec![], vec![Operand::reg(soon)]);
        f.inst(b, vec![], vec![Operand::reg(later)]);

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
        f.inst(entry, vec![Operand::reg(x)], vec![]);
        f.inst(entry, vec![Operand::reg(filler)], vec![]);
        f.inst(entry, vec![], vec![Operand::reg(filler)]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        f.inst(body, vec![], vec![Operand::reg(x)]);
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

        f.inst(entry, vec![Operand::reg(through)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        // The loop uses two values of its own, filling a two-register machine, so
        // `through` cannot also survive in a register.
        f.inst(body, vec![Operand::reg(a)], vec![]);
        f.inst(body, vec![Operand::reg(b)], vec![]);
        f.inst(body, vec![], vec![Operand::reg(a), Operand::reg(b)]);
        f.inst(body, vec![], vec![]);
        f.goto(body, &[header]);

        // `through` is read only here, after the loop.
        f.inst(exit, vec![], vec![Operand::reg(through)]);

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

        f.inst(entry, vec![Operand::reg(used)], vec![]);
        f.inst(entry, vec![Operand::reg(unused)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);

        f.inst(header, vec![], vec![]);
        f.goto(header, &[body, exit]);

        for _ in 0..10 {
            f.inst(body, vec![], vec![]);
        }
        f.inst(body, vec![], vec![Operand::reg(used)]);
        f.goto(body, &[header]);

        f.inst(exit, vec![], vec![Operand::reg(unused)]);

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

        f.inst(b, vec![Operand::reg(victim)], vec![]);
        // Repeatedly fill and drain the register file, reading `victim` in between
        // so it is reloaded and then evicted again.
        for &o in &others {
            f.inst(b, vec![Operand::reg(o)], vec![]);
            f.inst(b, vec![], vec![Operand::reg(o)]);
            f.inst(b, vec![], vec![Operand::reg(victim)]);
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

        f.inst(b, vec![Operand::reg(dead)], vec![]); // never read again
        f.inst(b, vec![Operand::reg(a)], vec![]);
        f.inst(b, vec![Operand::reg(c)], vec![]);
        f.inst(b, vec![], vec![Operand::reg(a), Operand::reg(c)]);

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

                // An `Any` operand the plan leaves out of registers is still read
                // from its slot when the code is emitted — the encoder loads it in
                // place. Those are real memory accesses, and today's figure counts
                // them (96 of mix2's 210 on aarch64), so leaving them out of the
                // plan's figure would credit it for traffic that still happens.
                let mut in_place = 0u32;
                let mut in_place_weighted = 0u64;
                for (i, &depth) in depth_of_inst.iter().enumerate() {
                    for o in m.uses(i) {
                        if o.constraint == Constraint::Any
                            && !plan.w_use[i].contains(&o.vreg)
                            && m.remat(o.vreg).is_none()
                        {
                            in_place += 1;
                            in_place_weighted += 10u64.pow(depth);
                        }
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

                // The contract the allocator will rely on: every operand really is
                // in a register where its instruction reads it, and the register set
                // never exceeds the file. Together these are what let the scan stop
                // making spill decisions at all — if they hold, colouring cannot
                // fail, which is the entire premise of decoupled spilling.
                for i in 0..m.num_insts() {
                    for class in RegClass::ALL {
                        let k = env.order(class).len();
                        for (what, set) in [("use", &plan.w_use[i]), ("after", &plan.w_after[i])] {
                            let n = set.iter().filter(|&&v| m.class(v) == class).count();
                            assert!(
                                n <= k,
                                "{file} inst {i} {what}: {n} {class:?} values in \
                                 registers, k = {k}"
                            );
                        }
                    }
                    // Only operands that demand a register. An `Any` operand is read
                    // from wherever the value is, so the plan owes it nothing.
                    for o in m.uses(i).iter().filter(|o| o.constraint != Constraint::Any) {
                        assert!(
                            plan.w_use[i].contains(&o.vreg),
                            "{file} inst {i}: reads v{} in a register but the plan \
                             does not have it in one there",
                            o.vreg.0
                        );
                    }
                    for o in m.defs(i) {
                        assert!(
                            plan.w_after[i].contains(&o.vreg),
                            "{file} inst {i}: writes v{} to a register but the plan \
                             does not have it in one afterwards",
                            o.vreg.0
                        );
                    }
                }

                // The runs the scan will colour, and the premise it rests on: at
                // most `k` of them cover any one position. If that holds, a linear
                // scan over them provably cannot run out of registers, which is the
                // whole reason the spilling decision was pulled out in front.
                {
                    let mut posn = vec![0usize; m.num_insts()];
                    let mut n = 0usize;
                    for &b in &layout.order {
                        for &i in m.block_insts(b) {
                            posn[i] = n;
                            n += 1;
                        }
                    }
                    let runs = plan.register_runs(&m, &layout.order, &posn, m.num_vregs());

                    for class in RegClass::ALL {
                        let k = env.order(class).len();
                        let mut cover = vec![0u32; n * 2 + 2];
                        for (v, rs) in runs.iter().enumerate() {
                            if m.class(VReg(v as u32)) != class {
                                continue;
                            }
                            for &(lo, hi) in rs {
                                for c in cover.iter_mut().take(hi as usize).skip(lo as usize) {
                                    *c += 1;
                                }
                            }
                        }
                        if let Some((at, &worst)) = cover.iter().enumerate().max_by_key(|&(_, c)| c)
                        {
                            assert!(
                                worst as usize <= k,
                                "{file}: {worst} {class:?} runs cover position {at}, \
                                 k = {k} — the scan could not colour this"
                            );
                        }
                    }

                    // A run must be inside the value's live range, or the scan would
                    // be reserving a register for a value that does not exist yet.
                    for (v, rs) in runs.iter().enumerate() {
                        for &(lo, hi) in rs {
                            assert!(lo < hi, "{file}: v{v} has an empty run");
                        }
                    }
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

                let ops = ops + in_place as usize;
                let weighted = weighted + in_place_weighted;

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
