# Plan: SSA-form register allocation with live-range splitting

**Status: the split path works end to end — allocates, verifies on both
targets, encodes on aarch64 — behind `allocate_with`, with `allocate` unchanged
as the default. The Belady policy wins under pressure (x86-64 mix −36%
weighted); the assignment gives much of it back at joins for want of
coalescing. §2 has the numbers and the two named fixes.**

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

### Stage 3 — next-use analysis and the Belady spiller (`4e93e2a`, `f66dffc`, `5c6cbb3`)

`nextuse.rs` is Braun09 §4.1: distances rather than live sets, joined by pointwise
minimum, with loop-exit edges charged `M` so a use after a loop outranks any use
inside it. `spill.rs` is §2 + §4.2 + §4.3: Algorithm 1 per block, `W_entry` from
Algorithm 2, coupling code on edges.

**Measured, plan against today's emitted code:**

| target | | plan | today | Δ ops | Δ weighted |
|---|---|---|---|---|---|
| x86-64 (13 regs) | mix | 26 (w 224) | 69 (w 483) | **−62%** | **−54%** |
| x86-64 (13 regs) | mix2 | 278 (w 2159) | 330 (w 2715) | **−16%** | **−20%** |
| aarch64 (20 regs) | mix2 | 182 (w 1469) | 210 (w 1722) | **−13%** | **−15%** |

**These include the in-place loads the plan implies**, and must. Leaving an `Any`
operand out of registers does not make its load disappear — the encoder reads it
from its slot at the mention. Counting only the plan's own reloads and stores gives
mix2 −61% weighted, which is an accounting artifact: today's figure counts those
96 in-place loads and the plan's would not have.

**Read the two targets together, not separately.** The benefit scales with
pressure, which is the shape Braun09's own setup implies — they measured on x86
with **7** registers, picked precisely to stress spilling. aarch64's 20 leaves
mix2 only just over the line; x86-64's 13 is nearer the paper's setting and shows
roughly twice the win.

`is_prime` and `mix` on aarch64 both come out at exactly **zero** spill code,
agreeing with an allocator that reached the same answer by an entirely different
route. That is the strongest correctness signal available short of wiring it in.

**Four things the papers get wrong or do not apply here**, all found by property
tests or by a number that did not fit, none by reading the code:

- §4.3's two coupling rules do not cover a value the predecessor held in a
  register, still live, that the successor has no room for. It leaves registers at
  the block boundary without passing through `limit`, so nothing stores it.
- The printed transfer function `f_B` in §4.1 has no case for a value *defined* in
  the block, which makes a value defined in `B` and live out of `B` come out
  live-*in* at `B`.
- §4.2's `p_L` estimate for admitting live-through values to a loop counts only
  values live *across* instructions and nothing for the registers an instruction
  needs for its own results, so it admits right up to `k` and the first
  `limit(.., k - |defs|)` in the loop evicts one of them — a store in the body and a
  reload on the back edge, every iteration, which is the Fig. 2d case the rule
  exists to prevent. Their footnote 6 concedes the estimate "might be an
  under-approximation". Keeping back the loop's widest instruction fixes it and is
  worth 170 weighted on mix2, the difference between −8% and +2% on aarch64.

- Braun09 assumes a load/store architecture where "each instruction requires that
  its operands are available in registers" (§2). We have `Any` operands read
  straight from a slot, and mix2's deopt stub takes 35 operands, 34 of them `Any` —
  it names the whole VM state so an exit can rebuild an interpreter frame. Treating
  those as register operands asks for 34 registers on a 20-register machine. Only
  register-demanding operands may drive spilling; this was worth roughly half the
  total win on both targets.

**The invariant that found most of the bugs**: *nothing is reloaded that was never
stored*. Zero stores against fifty reloads is incoherent on its face, and three
separate defects presented as exactly that — dead values occupying `W`, jump
arguments dropped from `w_exit`, and block parameters reloaded from slots the edge
was supposed to fill. Keep it.

---

### Stage 4 — colouring pre-split runs (`d0370a6`, `99019a8`)

`allocate_with(f, env, Some(sets))` takes the spiller's decisions and colours the
runs they imply. **`primes` and `mix` pass the symbolic verifier as split
allocations**; `mix2` does not, for the reason in §2. `allocate` itself is
unchanged and still makes its own whole-value decisions, so nothing in the shipped
pipeline has moved yet.

The scan turned out to be the whole-value one with its hardest part deleted. At
most `k` runs cover any position, so a free register always exists: no eviction
heuristic, no spill-versus-evict decision, no retroactive re-spilling. Runs are
contiguous, so there is no `inactive` list either — splitting *before* the scan
turns holes into separate intervals rather than gaps to reason about.

Three things the verifier caught, none of which unit tests would have:

- **Temps get no run**, because no spiller tracks scratch. They must span the whole
  instruction slot (as `build_intervals` has them) so they collide with every
  operand — which means the spiller must reserve them from the *read* onwards, not
  only the write.
- **The slot test cannot compare totals.** A value stays in `W` after its last use
  until something evicts it, so a run overhanging one end exactly offsets a real gap
  in the middle; that reads as fully covered, allocates no slot, emits no reload,
  and the register is read having never been written. Per position, not per sum.
- **One store per value, after the definition** — Wimmer05 §4c, exact rather than
  heuristic here, since SSA gives one definition and the slot never goes stale.

---

### Stage 5 — edge resolution over split values (`ba357cd`)

`resolve_edges` now resolves **every value live at the successor's entry**, not only
block parameters, and places moves at whichever end of the edge is private to it.
Inert for the whole-value path — a value with one location for life agrees at both
ends by construction — which its 289 unchanged tests confirm. Braun09's coupling
code falls out of it for free: a value in a register on one side and a slot on the
other *is* a reload, so there is no second list to keep in step.

Runs are also clipped at block boundaries, and a reload is emitted only where a run
follows a real gap rather than at every run start.

**`primes` and `mix` passing at `99019a8` was luck**, not a regression since: runs
were unclipped there and happened not to span a join on those two functions.

---

### Stage 6 — the split path works end to end (`922a3b6`)

All three benchmarks allocate, **verify on both targets**, and encode on aarch64
as genuinely split allocations; every prior test passes unchanged. The batch that
got there, each item forced by the verifier or by x86-64's constraints:

- **`Locations` is honest**: entries anchor at each value's birth, a spilled
  value's gaps point at its slot, a replayed value's gaps say *nowhere* (a new
  entry state). Everything downstream reads the table; nothing keeps its own
  books.
- **Runs clip at clobber/`Fixed` sites**, not only block boundaries — one register
  per run means a clobber anywhere bans it everywhere, and a run crossing three
  call-shaped sites accumulated every ban and declined. Wimmer05's split-at-calls,
  before the scan. The boundary is a register hop, one move.
- **Hops at one barrier are a parallel copy**, same as edge moves — emitted in
  value order, two hops through one register clobber each other. The sequencing
  moved out of `resolve_edges` into `emit_parallel`; both callers share it.
- **The spiller models the machine**: `k − burned` at pinned/clobbering
  instructions; a two-address op's non-reused inputs kept across the def (dying
  inputs sort first under plain Belady — exactly wrong); Wimmer05's
  must-have-register flag as a keep-set, because a deopt-shaped instruction reads
  35 operands all at distance 0 and an arbitrary cut evicted the one that cannot
  be read from a slot.
- **Dying values leave `W` at their last read**, not block end — a lingering dead
  parameter's register reads as busy at its own replacement's def, refusing the
  affinity and rotating every accumulator one register off. Exempt: keep-sets,
  dead defs, jump args.
- **Guards whose keepalives could fill the register file declare two temps** for
  their stub's scratch (`stub_temps`, aarch64). `stub_scratch`'s hunt only ever
  worked because whole-value over-spilling left slack. Small guards pay nothing.
  **x86-64's encoder has the same latent assumption and no fix yet** — its split
  allocations verify but encoding them is unexercised.

## 2. What is left: the measurement says coalescing, twice

**Emitted aarch64** (`asm_dump::split_vs_whole_report`):

| | whole | split |
|---|---|---|
| is_prime | 57 insts, 0 moves | **57 insts, 0 moves — identical** |
| mix | 273, 0 moves | 279, 6 moves |
| mix2 | 1088, 11 moves, 210 ops (w 1722) | 1170, **64 moves**, 254 ops (**w 1937**) |

**x86-64, allocation level** (`spill::tests::split_vs_whole_allocations`):

| | whole | split |
|---|---|---|
| mix | 26 moves, 69 ops (w 483) | 38 moves, **41 ops (w 311, −36%)** |
| mix2 | 31 moves, 330 ops (w 2715) | **85 moves**, 358 ops (w 2653, −2%) |

Read together: **the Belady policy wins where pressure is real** — x64 mix −36%
weighted is the paper's promise showing up — **and the assignment gives it back at
joins**, because presplit colouring has a pairwise affinity where the whole-value
path has transitive coalescing and shared spill slots. mix2's 14-parameter join
pays per-edge slot deliveries (+24 stores over whole) and register shuffles (+53
moves) that coalescing made free. The old §2.3 said splitting is the precondition
that makes coalescing worth having; the measurement says the converse too.

Two named fixes, then re-measure:

1. **Spill-slot sharing across arg/param chains** — Braun09 §4.4's CSSA
   precondition, deferred earlier because `slot_to_slot` measured zero; the cost
   now shows as per-edge stores instead. One slot per phi-congruence class makes
   the arg's store-at-def *be* the edge delivery.
2. **Transitive affinity over runs** — union-find over non-interfering
   arg/param/`Reuse` pairs, as `coalesce()` does over intervals, so a value
   feeding two joins lands where both expect it.

`allocate()` is unchanged and remains the default everywhere.

### Superseded

`resolve_edges` walks an edge's arguments against its successor's parameters and
nothing else. Without splitting that is *complete*: every other value has a single
location for its whole life, so both ends of an edge agree by construction and
there is nothing to reconcile. With splitting it is not — Wimmer10 Fig. 7 resolves
**every interval live at the successor's entry**, because a value split differently
on two paths disagrees at the join exactly as a parameter would.

This is the risk this file predicted, and it is where `mix2` stops. It presents as
a register read that was never written: runs are built over the linearized position
axis, so one can span from the end of one block into the next merely because they
are adjacent *in layout*, and resolution then sees the same register at both ends
of the real edge and emits nothing. Two halves:

1. **Clip runs to block boundaries**, so a run never implies continuity across an
   edge that control does not take.
2. **Resolve every live value, not just parameters** — with the reload/store then
   falling out of the two ends disagreeing, which is what makes coupling code and
   split moves the same mechanism (Wimmer10 §6).

Then re-measure, and only then consider making it the default. The rest:

1. **Build intervals from the plan, not from liveness.** A value is currently one
   interval spanning its whole live range. Under the plan it is one interval per
   maximal run where the plan says it is in a register — several per value, which
   is what `Locations` was made position-indexed for.
2. **Delete the scan's spill decisions.** With pressure already at `k` everywhere,
   `TRYALLOCATEFREEREG` never fails, so the eviction heuristic, `EvictKey`, and the
   `remat_spilled` bookkeeping all go. This should be a net *deletion*.
3. **Emit the plan's edits.** A reload becomes `Edit::Move` slot→reg (or
   `Edit::Remat`), a store reg→slot, at the points the plan names. Edge coupling
   code joins the existing parallel copy in `resolve_edges`.
4. **Re-measure.** The plan's own count is an estimate of what this will emit, not
   a promise: the current allocator leaves `Any` operands in slots and lets the
   encoder load them in place (96 of mix2's 157 aarch64 loads), and the plan has no
   notion of that. Expect the real number to differ from Stage 3's table.

**The risk is concentrated in `resolve_edges`.** A value split differently on two
paths disagrees at every join, so the sequencer, cycle breaking and slot routing
all get much busier than they are today. Stage 2b made resolution total, which is
what makes this survivable — but "total" was established against a workload where
almost nothing was split.

### Why the old §2 no longer applies

It read: *"Splitting. The allocator assigns one location per value for its whole
life. Three separate problems trace to that."* Two of those three are now
addressed by the pass above rather than by an in-scan splitter, and the third
(`Reuse` coalescing being a net loss) is still open but is downstream of the
integration, not of the algorithm. Kept below for the measurements, which stand:

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

Note that (1)'s "~16" was a guess when written and has since been confirmed
independently: `nextuse` puts mix2's aarch64 peak at 36 against a 20-register
pool, and 36 − 20 = 16 exactly.

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
