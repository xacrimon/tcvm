# Property-access fast-path optimization

After the shape / inline-cache landing, profiling nbody (`hyperfine`,
release build, borrow checks disabled) showed only ~3% wall-clock
improvement. Investigation against `flamegraph.svg` and
`getfield.asm` (disassembly of `op_getfield`) identified the
remaining bottleneck: dependent-load latency through the IC fast
path. The IC is hitting (no slow-path frames in the flamegraph), but
each hit pays for a long chain of pointer chases.

## What the assembly shows

`op_getfield` on a hit does **~12 dependent loads** end-to-end. The
chain is the killer — each load waits for the previous, so nothing
parallelises:

```
read_ic chain (5 dependent loads to find the IC slot):
  ldp [x23, #32]       thread.frames.{ptr,len}     — 1 load
  ldur [x10, #-32]     frame.closure (Gc ptr)      — 2
  ldr  [x10, #16]      closure.proto (Gc ptr)      — 3
  ldr  [x10, #88]      proto.ic_table (Gc ptr)     — 4
  ldr  [x10, #16]      Box<[InlineCache]>.data     — 5
  add  + ldr [x10]     ic_table[ic_idx].shape      — 6

ic_check chain (2 more dependent loads for the gen check):
  ldr  [x9]            value.data → TableState     — 7
  ldr  [x9, #64]       state.shape                 — 8
  ldr  [x12, #56]      shape.mt_token              — 9
  ldr  [x12, #16]      token.generation            — 10

property_at chain (2 more):
  ldr  [x9, #8]        state.properties.data       — 11
  ldr  [x9 + slot*16]  the Value itself            — 12
```

Each load on Apple Silicon is 4-5 cycles min at L1 (more on miss),
and they're sequential. That's **40-60+ cycles of pointer chasing
per cache hit**, plus the dispatch overhead and register/value
writes.

`op_setfield` is structurally the same and shows the same pattern at
~26% of total runtime alongside `op_getfield`'s ~23%. Same fixes
apply symmetrically.

## Action items, ordered by impact-to-invasiveness

### 1. Drop the `mt_gen` check from `ic_check`

Saves loads 9 and 10 on every access.

`InlineCache::Mono::mt_gen` is redundant. The IC compares
`live_shape.mt_token.gen() == ic.mt_gen`. But `shape.maybe_has_mm()`,
which the fast path *also* calls when the value is nil, does its own
staleness check via `mm_cache.gen_at_compute`. Two independent
staleness mechanisms guarding the same condition.

Drop `mt_gen` from the IC entry. The fast path becomes:

- shape ptr eq
- read property
- if value is nil and `maybe_has_mm(INDEX)` → slow

For nbody, none of the bodies have metatables, so
`shape.mt_token == None`, `maybe_has_mm` short-circuits with one
load (the `None` check). No `token.generation` read.

This eliminates 2 dependent loads on every fast-path access and
shrinks `InlineCache::Mono` from 16 to 12 bytes (or pad to 16 with
free space for a future poly slot).

### 3. Pin `proto` in a register across handler tail-calls

Saves loads 2–4 (the closure / proto / ic_table chain).

The `read_ic` chain pays for `frames.last().closure.proto` on every
opcode that touches an IC. But `proto` only changes at CALL /
RETURN / continuation entry. Two shapes:

**3a (recommended)**: Add
`current_ic_base: *const Lock<InlineCache>` to `ThreadState`,
updated on push_frame / pop_frame. `read_ic` becomes
`*current_ic_base.add(ic_idx)`. **One load instead of five.**
Touches the call/return handlers and `read_ic`; nothing else.

**3b**: Add `proto` (or `ic_base`) to the handler signature.
`extern "rust-preserve-none"` already preserves all registers across
the tail call, so it's essentially free at runtime — `proto` is
already in a fixed register at handler entry. Cost: every handler
signature changes, every `dispatch!()` and `become` site updates.
Much bigger churn.

Pick 3a unless something else forces 3b.

### 4. Inline a few "in-object" property slots in `TableState`

Saves load 11.

V8's "in-object properties" idea: for shapes with ≤N slots, store
the values inline in the table struct rather than in a `Vec`.
Eliminates the Vec data-pointer load on every read for small objects
(nbody bodies have 7 fields each — fits in 8 inline slots).

Cost: `TableState` becomes a DST, or accept a fixed inline buffer of
`[Lock<Value>; N]` plus a `Vec` for overflow. The IC's `slot` field
would need to encode in-object vs. overflow, but that fits trivially
in the high bit.

### 5. Embed the IC slot adjacent to the bytecode

The V8 / JSC feedback-vector approach.

Instead of `proto.ic_table[ic_idx]`, store IC slots in a parallel
array indexed by the same offset as the instruction itself — or even
interleave them. With a per-frame `ip_to_ic_offset` constant in a
register, `ic_slot_addr = ic_base + (ip - code_base) * sizeof(slot)`,
computable from `ip` directly. No proto deref at all on hits.

Most invasive of the lot. Skip unless 1–3 don't get the numbers
where they need to be.

## Suggested rollout

**PR 1: item 1.** Small, mechanical, no architectural change. The
`ic_check` body shrinks to a single `cmp` of shape pointers. With
~50% of runtime in `op_getfield` + `op_setfield`, this should
noticeably move the nbody benchmark.

**PR 2: item 3a** (`current_ic_base` on `ThreadState`). The bigger
win — cuts `read_ic` from a 5-load chain to a 1-load chain. Scoped
to the call/return handlers and `read_ic`.

**PR 3: item 4.** Once IC overhead is in the noise, property access
becomes Vec-deref-bound and inline slots is the right next step.

**PR 4 (optional): item 5**, only if the numbers still need more.

## Resolved

- ~~Drop the `Gc<RefLock<...>>` wrapper around `ic_table`~~ — the
  field is now `Box<[Lock<InlineCache>]>` directly on `Prototype`.
  Reads are counter-free `Lock::get()`; writes emit the backward
  barrier on the parent `Prototype` and use `Lock::as_cell()`. Saves
  one dependent load on every fast-path access (the Gc-deref of the
  separate `ic_table` allocation is gone; the slice now lives inline
  in the prototype).
