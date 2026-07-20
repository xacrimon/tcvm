# Plan: SSA-form register allocation with live-range splitting

**Status: Stages 1, 2, 2b done. The spilling yardstick now exists and the
allocation interface is position-indexed. What remains is the spiller itself,
and the decision is to take it straight from Braun09 rather than build
Wimmer05's in-scan splitting first — see §2.**

Papers are in `papers/` (gitignored) with a distilled algorithm crib sheet at
`papers/NOTES.md` — pseudocode, measured numbers, and correctness traps. Read
that before implementing; this file holds staging, decisions, and what has
actually been measured.

- **Wimmer & Mössenböck**, *Optimized Interval Splitting in a Linear Scan Register
  Allocator*, VEE 2005 — the splitting scan.
- **Wimmer & Franz**, *Linear Scan Register Allocation on SSA Form*, CGO 2010 —
  SSA on top of that scan.
- **Braun & Hack**, *Register Spilling and Live-Range Splitting for SSA-Form
  Programs*, CC 2009 — decoupled Belady spilling.

---

## 1. Done

### Stage 1 — allocate on SSA form (`31fb43d`, `b3fab3c`)

`isel` keeps block parameters; the allocator resolves the edges. Three pieces:

- **`order.rs`** — block layout where dominators precede their blocks and a loop's
  blocks stay contiguous, plus the loop forest. The contiguity is what lets
  interval construction cover a loop with one range add per live value. Rejects
  irreducible control flow rather than miscompiling it (Wimmer10 §4.3).
- **`build_intervals`** — Wimmer10 Fig. 4, one reverse pass, no dataflow fixpoint.
- **`resolve_edges`** — Wimmer10 Fig. 7, SSA deconstruction folded into edge
  resolution, over physical locations rather than virtual registers.

`Allocation` records where block parameters live so `verify` can check that
resolution delivered each argument rather than assume it.

**Measured on `mix`:** isel −24%, allocate −27%, full lower..encode 97.3 → 76.4 µs
(**−21%**). Machine code byte-identical on `is_prime` (57 instructions, 0 moves).
Exec unchanged.

The win is where Wimmer10 said it would be — the deleted passes, not the scan.
Almost all of it is allocation traffic: `isel` was materializing 50 extra `MInst`s
on `mix` (126 → 76 instructions), each costing four heap allocations.

### Stage 2 — real coalescing (`2ebe37e`)

Union-find over values with disjoint live ranges, merged into one allocation unit
the scan places once. A hint is consulted while placing a value and dropped when
the register it wanted is taken — exactly what happens under the pressure that
makes coalescing matter. A merged set has one location and cannot come apart.
Merge order is by loop depth, since an early merge can block a later one.

**Measured on `mix2`:** back edge 29/29, branch join 14/14, inner loop 4/4, all
zero moves; the only unmerged pairs are on paths that execute once. `mix` is 4/4.
Compile time *improved* (fewer intervals to scan). x86-64 `mix` spills 10 → 8.

**Every coalescing miss is an excluded value, not interference** — so the merge
order is leaving nothing on the table. 43 of 44 are rematerializable constants on
once-executed paths.

### Stage 2b — edge resolution made total (`aa60789`)

A cycle breaks through a stack slot (no register needed); a slot-to-slot move or a
replay into a slot bounces a register — saved to a fresh slot, borrowed, restored.
An edge has no operands of its own, so there is always something to bounce, which
makes resolution total. x86-64 no longer declines `mix2`.

---

### Stage 2c — the spilling yardstick (`0e3ab0b`)

`spillcost.rs` weights every memory operation by the loop depth of the block it
lands in. This was §3's "missing dynamic reload metric", and it had to come first:
Braun09's whole contribution is hoisting reloads out of loops, which leaves a
static count unchanged while halving the executed one. Judged statically, the
thing we are building looks like a no-op.

Traffic comes from two disjoint places and both are counted — an `Edit` the
allocator asked for, and an operand left at `Alloc::Spill`, which the encoder
loads at the mention on its own (`aarch64::read_g`). Counting only edits misses 96
of mix2's 157 loads.

The counts are exact, not heuristic: mix2 reports 210 memory ops and `disasm_mix2`
independently marks 210 `sp`-relative accesses, agreeing on loads and stores
separately. Only the weighting is an estimate (`TRIP = 10` per nesting level).

**Baseline:** is_prime 0, mix 0, **mix2 210 ops / weighted 1722, of which 1680
(97.6%) is inside loops.** That last number is the target.

### Stage 2d — position-indexed locations (`d60f1ad`)

`loc: Vec<Option<Alloc>>` became a `Locations` table keyed by value *and*
position. Callers must now say which mention they mean; one that has not been
taught no longer compiles. Pure plumbing, verified neutral (mix2 210/1722
unchanged, is_prime 57/0 unchanged).

This is the precondition for everything below, and it is the *whole* of the
interface change — the scan still assigns one location per value, so every entry
starts at position 0.

---

## 2. What is left, and why it is all one thing

**Splitting.** The allocator assigns one location per value for its whole life. Three
separate problems trace to that, all measured on `mix2`:

1. **Over-spilling.** 29 values spilled where the pressure peak needs ~16 in
   memory. A value live across the peak but used mostly outside it is spilled
   *entirely*, so it lives in memory for its whole life and pays a reload at every
   mention — including the ones where a register was free.
2. **In-loop reload traffic.** 44 reloads + 28 spill-stores per iteration, 50 of
   them in one 14-parameter join block. Against **0 reg-reg moves in the loop** —
   the shuffle problem is solved, this is all spill traffic.
3. **`Reuse` coalescing is a net loss.** Merging two-address pairs was tried and
   cost x86-64's `mix` loop 12 extra memory operations to save 7 register moves.
   Merging raises pressure nowhere, but yields one *longer* interval, and without
   splitting a longer interval must find a register free for its whole extent or
   spill entirely — where the two halves could each have taken one (Wimmer05 §4.2).

So splitting is not merely Stage 3's spilling improvement; it is the precondition
that makes the rest of the coalescing worth having.

### Splitting does not make Stage 2b redundant

Worth stating because it is counter-intuitive. All three scratch cases survive
splitting: cycles are a property of the assignment, slot-to-slot moves are
explicitly discussed as *arising* from phi handling (Wimmer10 §6), and a replayed
constant still needs a register before it can be stored. What changes is
frequency — fewer values spilled means more registers free, so the bounce fires
less. Meanwhile resolution does *more* work under splitting, since a value split
differently on two paths disagrees at every join, so the sequencer, cycle
breaking and slot routing all get busier.

### Decision: go straight to Braun–Hack, skip Wimmer05's splitting scan

Taken deliberately, and it contradicts an earlier version of this file which had
Wimmer05's splitting scan as Stage 3 with Braun09 layered on top. NOTES.md §4 had
it right: *"C subsumes most of Wimmer05's spill machinery. If C is the
destination, build the minimum viable spiller in B."*

Under Braun09 the spiller runs **before** assignment and lowers max pressure to
*k* everywhere; on SSA form register demand then equals max pressure, so the scan
provably never spills again. That makes ALLOCATEBLOCKEDREG's whole apparatus —
`nextUsePos`, spill-current-itself, loop pseudo-uses, out-of-loop split positions
— dead weight the day it lands. They are all approximations of what the Belady
pass does exactly.

So the scan never needs to split. It only ever *colors already-split intervals*,
which is a much smaller change than Stage 3 was.

### Two things reading the paper changed (the crib sheet compressed them)

- **§4.4's SSA reconstruction drops out entirely for us.** The paper inserts real
  reload *instructions*, which is a second definition of `x0`, which breaks SSA and
  forces a Sastry & Ju dominance-tree walk with lazy φ insertion (their Fig. 4).
  We cannot mutate the IR and do not want to: a reload here is an `Edit` and a
  split is an interval split, so there is never a second *definition*, only a
  second *location*. `Locations::get(v, p)` answers the exact question SSA
  reconstruction exists to answer. This is the payoff of the interface being
  per-operand rather than per-vreg.
- **§4.4's input requirement is real, and §4 below files it as optional.** The
  paper *demands* conventional SSA — every φ-congruence class interference-free —
  so a class shares one spill slot, "else spilled φ-functions result in memory
  copy instructions". That is the "spill-slot coalescing" bullet in §4. Under
  Braun09 it is a **precondition**, not a nicety: without it every spilled block
  parameter degenerates into exactly the stack-to-stack moves that `resolve_edges`'
  bounce exists to paper over.

### The remaining tension

Stage 1's win came from *deleting* a dataflow fixpoint, and Braun09 reintroduces
one (richer, over `Var → N ∪ {∞}`). Neither paper measures the combination.
Benchmark it; do not assume it. Braun09 reports 430 instructions/ms for the
spilling phase alone and never measures a whole allocator against linear scan.

---

## 3. Benchmarks

- **`test-files/mix.lua`** — 12 accumulators, one loop. Fits aarch64's 20-register
  pool with room to spare (0 spills), so it cannot judge spilling there; spills 8
  on x86-64's 13.
- **`test-files/mix2.lua`** — 28 accumulators, a nested loop, and a branch whose
  arms write disjoint sets so the join must reconcile two live sets. 29 spill
  slots on aarch64, 44 on x86-64. This is the case that measures splitting.
- **`benches/jit_pipeline.rs`** — per-stage compile time on `mix`.
- **`asm_dump`** — reg-reg move density on `is_prime` (aarch64 only).
- **`spillcost.rs`** — loop-depth-weighted spill traffic, the spilling yardstick.
  Run `cargo test --lib asm_dump::spill_traffic -- --nocapture` before and after
  any change to spill policy. **Judge by `weighted in loops`, not by the op
  count**: a successful hoist moves a reload out of a loop without deleting it, so
  the static count can hold still or rise while the real cost falls tenfold.

---

## 4. Smaller things

- **Fallthroughs (2 instructions).** Loop-contiguous ordering costs 2 extra
  branches on `is_prime` (55 → 57) because placement is driven by contiguity, not
  fallthrough. Wimmer05 §2.1 accepts this trade for locality. Recovering it is a
  greedy chain-formation pass, not a tweak — real work for 2 instructions.
- **Spill-slot coalescing — promoted out of this section.** Non-merged spilled
  values each take a fresh slot (`spills += 1`). Giving a parameter and its
  argument the same slot when their ranges do not overlap would cut both the slot
  count and the slot-to-slot moves the bounce exists for (Wimmer10 §6 does exactly
  this). **This is no longer optional:** Braun09 §4.4 requires it as an input
  condition. See §2.
- **`MInst` allocation traffic.** Each instruction holds four `Vec`s and `defs`/
  `uses` always allocate. This was most of Stage 1's isel win; a `SmallVec` or a
  flat operand arena would attack the same cost across the whole MIR. Unprofiled.

---

## 5. Corrections to earlier versions of this plan

Recorded because each was believed and acted on:

- **`is_prime` was never 65 instructions / 2 moves.** That figure predated the
  branch-fusion and IR-simplification commits; the baseline was 55/0 before this
  work began. There were no "2 surviving loop-carried moves" to remove, so the
  codegen argument for Stage 1 was weaker than first stated — the case rests on
  compile time, which held up.
- **`build_intervals` is not safe on non-SSA input.** It is sound but a
  pessimization: the loop rule's justification is SSA dominance, and a multi-def
  value defined *inside* a loop is live-in at the header yet dead through the
  tail. Extending it fills the hole that lets the two ends share a register.
- **Splitting, not SSA, is what removes stack traffic.** The papers bundle them;
  for us they are separable, and Stage 1 delivered compile time while leaving
  every spilling defect exactly as it was.
