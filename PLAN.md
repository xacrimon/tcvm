# Plan: SSA-form register allocation with live-range splitting

Rework the JIT register allocator to run on SSA form with live-range splitting.
Two goals, in priority order:

1. **Compile time** — delete the dataflow liveness fixpoint and the SSA
   deconstruction pass in `isel`. This is the primary motivation.
2. **Code quality under pressure** — remove the reload-to-stack "bounce" and the
   retroactive-cold-spill defect, both of which are symptoms of whole-value
   assignment; then decouple spilling entirely (Belady/Min).

Papers are in `papers/` (gitignored) with a distilled algorithm crib sheet at
`papers/NOTES.md` — pseudocode, measured numbers, and correctness traps. Read
that before implementing; this file only holds staging and decisions.

- **Wimmer & Mössenböck**, *Optimized Interval Splitting in a Linear Scan Register
  Allocator*, VEE 2005 — the splitting scan.
- **Wimmer & Franz**, *Linear Scan Register Allocation on SSA Form*, CGO 2010 —
  SSA on top of that scan.
- **Braun & Hack**, *Register Spilling and Live-Range Splitting for SSA-Form
  Programs*, CC 2009 — decoupled Belady spilling.

---

## 1. What the papers actually deliver

Corrected against a close read; the earlier version of this plan overstated some
of it.

### The compile-time win is in the deleted passes, not the scan

Wimmer10 Fig. 10 breaks the back end down:

| phase | change |
|---|---|
| LIR construction | −19% to −27% — SSA deconstruction deleted |
| lifetime analysis | −25% to −31% — dataflow fixpoint deleted |
| **linear scan itself** | **~unchanged** |
| resolution | **+1% to +10%** — it absorbs deconstruction |
| back end total | −13% to −19% |

So the entire win comes from removing work, not from the scan getting smarter.
For us the two deleted passes are `isel::parallel_copy` (isel.rs:993) plus
critical-edge materialization, and `liveness` (regalloc.rs:1352). That maps
cleanly onto our code, which is the main reason to be optimistic.

**Do not budget anything for the interval-intersection optimization.** SSA lets
you skip 59–79% of intersection tests, and Wimmer10 §7.1 measures the speedup
from that as *not measurable*.

### Codegen is roughly a wash from SSA alone

Machine code size −0% to −1%; run time within noise (one statistically
significant 1% on SciMark FFT). The codegen case rests on **splitting**
(Wimmer05) and later on **Belady spilling** (Braun09), not on SSA per se.

### Braun09's numbers are a pressure regime we don't currently have

−54.5% reloads / −61.5% spills, measured at **7 GP registers** on x86 CINT2000,
against a Wimmer-style linear scan. aarch64 gives us a 20-register pool and
`mix`/`is_prime` spill zero — we cannot observe this without a higher-pressure
benchmark. (Agreed to add one later; C should not be evaluated before it exists.)

### B and C pull in opposite directions on compile time

Wimmer10's win comes from *deleting* a dataflow fixpoint. Braun09 *reintroduces*
one — richer, over `Var → N ∪ {∞}` with pointwise-min join and loop-exit edge
lengths of M ≈ 100000. It buys that back by making assignment trivial (pressure
is already ≤ k, so the scan makes no spill decisions and could be a linear-time
coloring), but **neither paper measures the combination**. Braun09's 430 insns/ms
is the spilling pass measured in isolation.

Treat C's compile-time effect as unmeasured. Benchmark it; don't assume it.

---

## 2. Where our allocator sits today

The arrangement both Wimmer papers argue against, plus patches.

1. **SSA destructed before regalloc.** `isel` lowers IR block-params into vreg
   `Mov` copies on edges, sequentializes the parallel move itself
   (`parallel_copy`, isel.rs:993, including a scratch register for permutation
   cycles) and materializes critical-edge blocks. The allocator then sees a
   conventional multi-def vreg CFG and **recomputes liveness by dataflow
   fixpoint** (`liveness`, regalloc.rs:1352).
2. **Whole-value assignment, no splitting.** Each vreg gets one location for its
   whole life. Under pressure: whole-spill plus a separate reload phase that
   re-loads at each mention.
3. **Patches that exist because we don't split:**
   - **reload-to-stack "bounce"** (`reload_reg`, regalloc.rs:1287) — when no
     register is free at a reload point, save a live-through value to a scratch
     stack slot for one instruction and restore after.
   - **whole-victim eviction → retroactive cold-spill** — evicting an active
     value turns its already-scanned (often cold) past into stack traffic too.
   - **remat** partly exists to soften whole-spill cost.

Wimmer05 addresses (3) directly: **use positions carry a must-have-register vs
should-have-register flag**, which is exactly the guarantee we lack — we
discover at reload time that nothing is free, and bounce. And
`ALLOCATEBLOCKEDREG` splits the victim *at the current position* rather than
whole-spilling it, which is the retroactive-cold-spill fix.

What survives the rework: the doubled position axis, segmented intervals with
holes (`Segs`), `Edit`/`Allocation` result shapes (regalloc.rs:238, 389 — designed
for this), and `verify` (regalloc.rs:1410, symbolic, already handles split
allocations).

---

## 3. Coalescing: what to do instead of the papers' answer

Both Wimmer papers **refuse coalescing** and use register hints instead —
Wimmer05 §4.2 on cost grounds (it mutates the IR after allocator structures are
built, forcing iterative rebuild), Wimmer10 §3 because coalescing two values
would violate SSA.

**We should not follow them here.** Our scan does real move coalescing today and
it is why `is_prime` sits at 2 moves / 65 instructions (3.1%). Wimmer10's
baseline was the HotSpot product allocator; ours is already tuned on exactly this
metric. Hints-only is a plausible regression.

### regalloc3's approach — the one to copy

regalloc3 allocates on SSA *and* coalesces, resolving Wimmer10's objection
cleanly: it never merges live ranges, it merges **values into a `ValueSet`**, a
group of values whose live ranges provably do not overlap. The input function's
SSA is untouched; the grouping is purely allocator-internal, and assigning the
whole set one register is safe by construction. (`src/internal/coalescing.rs`,
366 lines; DESIGN.md §Coalescing.)

Mechanism:

- `Value → ValueSet` tracked in a **union-find**; every value starts in its own set.
- **Two sources of implied moves in SSA: block parameters and `Reuse` operands.**
  Both are exactly what we have.
- Merge test: walk the two sorted `ValueSegment` vectors in lockstep checking for
  overlap; if clean, merge the lists and union the sets. Fast path first — if one
  set lies entirely before the other, concatenate without the lockstep walk.
- **Order matters**, because merging A–B can block A–C. Blocks are processed by
  priority: hot blocks first, then **critical-edge blocks holding only a jump**
  (a move-free such block can be deleted outright by jump threading), then more
  connected blocks. Heuristic borrowed from LLVM's `compareMBBPriority`.

Hints remain, complementary rather than a substitute: fixed-register hints
weighted by the frequency of the move they would eliminate, plus a weak
"last register allocated for this value set" hint to keep split pieces together.

Note this is separable from the rest of regalloc3. regalloc3 is LLVM-Greedy —
priority queue, evict/split stages, spill weights, a B-tree register matrix —
which is more machinery than a linear scan should need, and is why we're not
adopting it wholesale. But **union-find coalescing over non-overlapping SSA
values is independent of Greedy** and would feed a Wimmer-style scan fine.

---

## 4. Staging

### Stage 1 — SSA-form MIR + splitting scan (the "B" work)

These are not separable in the target design. The papers' allocator *is* a
splitting linear scan on SSA; splitting-without-SSA was only ever a shortcut to
avoid the MIR rework, and we're doing the MIR rework.

- Carry block params from the optimizing IR (already SSA) into `MFunc`; stop
  destructing in `isel`. `parallel_copy` largely moves to resolution.
- Replace `liveness` with `BUILDINTERVALS` (Wimmer10 Fig. 4) — one reverse pass,
  no fixpoint.
- Rebuild the scan around `TRYALLOCATEFREEREG` / `ALLOCATEBLOCKEDREG` with
  splitting (Wimmer05 Figs. 4, 5).
- Add **use positions** with the must/should-have-register flag. This is what
  retires the bounce.
- Fuse SSA deconstruction into resolution (Wimmer10 Fig. 7) — the `if it starts
  at start of successor` branch, ~20 lines replacing our deconstruction.

**Build the minimum viable spiller here.** Under C the scan reaches every point
with pressure already ≤ k and makes no spill decisions, so `ALLOCATEBLOCKEDREG`'s
apparatus — nextUsePos, spill-current-itself, loop pseudo-uses, out-of-loop split
positions — becomes dead weight. Each is an approximation of what Belady will
later do exactly. Don't build what C deletes.

Preconditions to satisfy, both required by `BUILDINTERVALS` and later by
Braun09's sweep:
- block order with **all dominators before the block** and **all blocks of a loop
  contiguous**;
- **critical edges split** (Braun09 attributes edge lengths to blocks).

### Stage 2 — coalescing

Port regalloc3's union-find `ValueSet` scheme (§3). Deferred to its own stage so
Stage 1's codegen can be measured against today's coalescing scan and the
regression, if any, is attributable.

### Stage 3 — decoupled Belady spilling (the "C" work)

Braun09: global next-use analysis with loop-exit edge lengths, loop-aware
`W_entry` so reloads hoist out of loops, coupling code on edges, Min/`limit` per
block, then SSA reconstruction (Sastry–Ju) because reloads are second defs.
Requires **conventional SSA** — every φ-congruence class interference-free so the
class shares one spill slot.

Gate on the higher-pressure benchmark existing. Measure compile time explicitly
(see §1) rather than assuming the fixpoint pays for itself.

---

## 5. Open questions

- **Are our loops guaranteed reducible?** `BUILDINTERVALS` is silently wrong on
  irreducible (multi-entry) loops — Wimmer10 §4.3. Lua's structured loops should
  guarantee it, but confirm rather than assume, and consider an assert.
- **Stack-to-stack moves become reachable.** Once phi intervals can be assigned a
  stack slot at their definition (unavoidable when a block has more phis than
  registers), resolution can emit stack→stack. Wimmer10 handles it by borrowing
  any free register — including a float register for an integer — without
  reserving a scratch. Decide our story; we have no scratch register either.
- **Does `verify` survive?** It is symbolic and already handles split
  allocations, but it could not model values defined by an edge in the earlier
  SSA prototype. Extending it matters — it is the safety net that proved
  eviction/bounce/remat correct.
- **x64 deopt stubs.** The earlier regalloc3 prototype hit a real limitation:
  keeping more values in registers left too few free of a guard's keepalives for
  `stub_scratch` (3 gprs, 13-reg pool). Any allocator that holds more in
  registers will hit the same wall; the stub encoder likely needs to write a
  keepalive back and reuse its register. aarch64's 20-reg pool has slack.
