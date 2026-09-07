//! What the spill code actually costs, weighted by how often it runs.
//!
//! A static count of spill instructions cannot judge a spiller. Braun & Hack's
//! central claim (CC'09) is about how often a reload *executes*: their whole
//! contribution is moving reloads out of loops, which leaves the static count
//! unchanged — or slightly worse — while cutting the executed count by half. Judge
//! that with a static count and it looks like a regression.
//!
//! They counted for real, with marked NOPs under Valgrind. The cheap equivalent,
//! and the one PLAN.md settles on, is to weight each edit by the loop depth of the
//! block it lands in: a reload in a doubly-nested loop is worth a hundred outside
//! one. That is a proxy for an execution count, not an execution count — it assumes
//! every loop runs [`TRIP`] times and every branch is taken — but it is on the right
//! side of the only distinction that matters here, which is inside-the-loop versus
//! outside it.
//!
//! The *counts* are exact rather than heuristic, and were checked against an
//! independent measurement: on `mix2` this reports 210 memory operations (61 reload
//! edits, 53 store edits, 96 loads the encoder emits in place) and `disasm_mix2`'s
//! objdump marks 210 `sp`-relative loads and stores, agreeing on loads and stores
//! separately. Only the weighting is an estimate.
//!
//! The weight is deliberately *not* Braun09's `M ≈ 100000` loop-exit edge length.
//! That number is a sentinel inside their next-use lattice, chosen to exceed the
//! longest path through a loop so that any use after the loop ranks as further away
//! than every use inside it. It is not an execution-frequency estimate and reusing
//! it here would make a single in-loop reload swamp the entire rest of the report.

use super::order::{self, Layout};
use super::regalloc::{Alloc, Allocation, Edit, RegallocFunc};

/// Assumed iterations per loop nesting level. The classic static estimate.
const TRIP: u64 = 10;

/// Spill traffic, counted statically and weighted by loop depth.
///
/// Traffic comes from two disjoint places and both are counted. An edit is traffic
/// the *allocator* asked for; an operand left at [`Alloc::Spill`] is traffic the
/// *encoder* emits on its own, loading the slot into a scratch register at the
/// mention (see `aarch64::read_g`). They cannot overlap: a `Reg`-constrained
/// operand gets a reload edit and a register, an `Any` one keeps its slot and gets
/// no edit. Counting only the first misses most of the loads on `mix2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpillCost {
    /// Memory-to-register edits.
    pub reloads: u32,
    /// Register-to-memory edits.
    pub stores: u32,
    /// Operands the encoder loads from their slot in place, having been given no
    /// register to find them in.
    pub operand_loads: u32,
    /// Defs written straight to a slot, likewise.
    pub operand_stores: u32,
    /// Slot-to-slot moves. Counted apart because no machine has the instruction:
    /// each is a reload *and* a store, and each is a resolution failure worth
    /// seeing rather than a spill decision.
    pub slot_to_slot: u32,
    /// Constants replayed instead of reloaded. Not memory traffic — tracked so a
    /// fall in `reloads` that is really a shift into remat is visible as one.
    pub remats: u32,
    /// Register-to-register move edits. Not memory traffic either, but the same
    /// static-count trap applies: a shuffle on a once-executed entry edge is
    /// nearly free while one at a loop join runs every iteration, so these carry
    /// their own weighted figures below.
    pub moves: u32,
    /// `Σ TRIP^depth` over reloads, stores, and twice each slot-to-slot move.
    pub weighted: u64,
    /// The same sum restricted to edits at depth ≥ 1 — the number Braun09 is
    /// actually trying to move, isolated from flat-code traffic that no amount of
    /// hoisting can help.
    pub weighted_in_loops: u64,
    /// `Σ TRIP^depth` over register-to-register moves, kept apart from the memory
    /// figures because a move costs a fraction of a load.
    pub moves_weighted: u64,
    /// The move sum restricted to depth ≥ 1.
    pub moves_in_loops: u64,
}

impl SpillCost {
    pub fn total(&self) -> u32 {
        self.reloads + self.stores + self.slot_to_slot + self.operand_loads + self.operand_stores
    }
}

impl std::fmt::Display for SpillCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} memory ops ({} reload, {} store, {} slot-to-slot, \
             {} in-place load, {} in-place store), {} remat; \
             weighted {} ({} in loops); \
             {} moves, weighted {} ({} in loops)",
            self.total(),
            self.reloads,
            self.stores,
            self.slot_to_slot,
            self.operand_loads,
            self.operand_stores,
            self.remats,
            self.weighted,
            self.weighted_in_loops,
            self.moves,
            self.moves_weighted,
            self.moves_in_loops,
        )
    }
}

/// Measure `ra`'s spill traffic against the loop structure of `f`.
///
/// Returns `None` for a function whose control flow the layout pass rejects; there
/// is no allocation to measure in that case either.
pub fn measure(f: &impl RegallocFunc, ra: &Allocation) -> Option<SpillCost> {
    let layout = order::compute(f).ok()?;
    Some(measure_with(f, ra, &layout))
}

/// As [`measure`], for a caller that already has the layout in hand.
pub fn measure_with(f: &impl RegallocFunc, ra: &Allocation, layout: &Layout) -> SpillCost {
    // Edits are positioned at instructions, so the depth of an edit is the depth of
    // the block owning its instruction.
    let mut depth_of_inst = vec![0u32; f.num_insts()];
    for &b in &layout.order {
        let d = layout.depth[b.0 as usize];
        for &i in f.block_insts(b) {
            depth_of_inst[i] = d;
        }
    }

    let mut c = SpillCost::default();
    for &(p, e) in ra.edits() {
        let depth = depth_of_inst[p.inst];
        let weight = TRIP.saturating_pow(depth);
        let mut charge = |n: u64| {
            c.weighted += n * weight;
            if depth > 0 {
                c.weighted_in_loops += n * weight;
            }
        };
        match e {
            Edit::Move(m) => match (m.from, m.to) {
                (Alloc::Spill(_), Alloc::Reg(_)) => {
                    c.reloads += 1;
                    charge(1);
                }
                (Alloc::Reg(_), Alloc::Spill(_)) => {
                    c.stores += 1;
                    charge(1);
                }
                (Alloc::Spill(_), Alloc::Spill(_)) => {
                    c.slot_to_slot += 1;
                    charge(2);
                }
                (Alloc::Reg(_), Alloc::Reg(_)) => {
                    c.moves += 1;
                    c.moves_weighted += weight;
                    if depth > 0 {
                        c.moves_in_loops += weight;
                    }
                }
            },
            Edit::Remat { .. } => c.remats += 1,
        }
    }

    // Operands the allocator answered with a slot. The encoder loads each one at
    // the mention, so a value read three times in a loop pays three times — which
    // is exactly the whole-live-range spilling PLAN.md §2.1 is about.
    for (i, &depth) in depth_of_inst.iter().enumerate() {
        let weight = TRIP.saturating_pow(depth);
        let loads = (0..f.uses(i).len())
            .filter(|&k| matches!(ra.use_(i, k), Alloc::Spill(_)))
            .count() as u32;
        let stores = (0..f.defs(i).len())
            .filter(|&k| matches!(ra.def(i, k), Alloc::Spill(_)))
            .count() as u32;
        c.operand_loads += loads;
        c.operand_stores += stores;
        c.weighted += (loads + stores) as u64 * weight;
        if depth > 0 {
            c.weighted_in_loops += (loads + stores) as u64 * weight;
        }
    }
    c
}
