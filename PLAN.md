# Plan: SSA-form register allocation with live-range splitting

**Status: Stage 1 is done and measured. Splitting is the whole of what remains,
and three separate things are now blocked on it.**

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

### Then Braun–Hack

Once pressure can be lowered to *k* by splitting, Braun09's decoupled spilling
decides *where* the splits go: global next-use analysis with loop-exit edge
lengths (M ≈ 100000), loop-aware `W_entry` so reloads hoist out of loops, and
coupling code on edges. Their −54.5% executed-reloads figure is measured against
precisely the situation `mix2` is in now.

Note the tension: Stage 1's win came from *deleting* a dataflow fixpoint, and
Braun09 reintroduces one (richer, over `Var → N ∪ {∞}`). Neither paper measures
the combination. Benchmark it; do not assume it.

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

**Missing: a dynamic reload metric.** Static spill counts cannot see Braun09's
central claim, which is about how often a reload *executes* — their loop-hoisting
wins are invisible statically. They counted with marked NOPs under Valgrind; the
cheap equivalent here is to weight edits by loop depth, or instrument the encoder
to count. Decide this before Stage 3, because it is the yardstick for all of it.

---

## 4. Smaller things

- **Fallthroughs (2 instructions).** Loop-contiguous ordering costs 2 extra
  branches on `is_prime` (55 → 57) because placement is driven by contiguity, not
  fallthrough. Wimmer05 §2.1 accepts this trade for locality. Recovering it is a
  greedy chain-formation pass, not a tweak — real work for 2 instructions.
- **Spill-slot coalescing.** Non-merged spilled values each take a fresh slot
  (`spills += 1`). Giving a parameter and its argument the same slot when their
  ranges do not overlap would cut both the slot count and the slot-to-slot moves
  the bounce exists for (Wimmer10 §6 does exactly this).
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
