# Shape system findings (#217)

Investigation of tcvm's shape (hidden class) system: what it costs the
collector, creating transitions, following them in inline caches, and moving
tables between shapes. Report only. Nothing here is implemented, and PR #218 is
unchanged.

Inputs: issue #217 ("optimize shape/hidden class system … also, improve shape
transitions so they don't keep strings alive") and `interp-perf-survey.md` on
branch `nanbox-perf-survey`. The second half of #217 (edges keeping key strings
alive) is already fixed on `gc-interp-exit` (commit `f03aa06`, "env: prune dead
shape transitions and stop pinning their keys").

## Method

- **tcvm:** branch `gc-interp-exit` at `1b013bd` (the PR #218 head), release build.
- **Counters:** a throwaway worktree added atomic counters to
  `transition_add_prop`, `transition_set_metatable`, `Shape::find_slot`,
  `TableState::migrate_to_dict` and `TableState::next`, printed when `Lua` is
  dropped. The worktree has been removed.
- **Profiles:** xctrace *Time Profiler* (1 ms sampling) on the unmodified release
  binary, symbolized top frames. "Self" is the top frame; "inclusive" means the
  function appears anywhere on the sampled stack.
- **Machine:** M4 Pro, macOS 26.
- **Reference engines**, read at: V8 `e4c54cbf078` (`~/repos/v8`), JavaScriptCore
  `7429477a` (`~/repos/jsc`).

Benchmarks (sources at the end):

- `particles_pm`: `test-files/particles_bench.lua` with a Park–Miller generator
  (identical output on tcvm, PUC Lua 5.5 and LuaJIT). `particles_long` is the
  same with 5× the frames, for profiling.
- `ctor`: 1 M `{x = i, y = i + 1, z = i + 2}` plus 1 M `Vec.new(i, 1):len2()`
  (a `setmetatable({}, Vec)` class). `ctor_long` is the same at 4 M each.
- `dynkeys`: a map-like table filled through computed string keys, rebuilt 200
  times (a spatial hash).
- `nbody` (`test-files/nbody.lua`) and `binary_trees` as controls.

## 1. Measurements

### 1.1 Counters

| benchmark | property adds | … reusing an existing transition | new shapes (property) | `setmetatable` calls / new shapes | `find_slot` calls | … misses | descriptors scanned | descriptor bytes copied | dict migrations | `next` calls |
|---|---|---|---|---|---|---|---|---|---|---|
| particles_pm | 1,675,633 | 1,658,002 (99 %) | **17,631** | 816,863 / 3 | 7,776,738 | **5,805,838** | 14,622,648 | **10,021,552** | **600** | 166,943 |
| ctor | 5,000,168 | 4,999,999 (100 %) | 169 | 1,000,000 / 1 | **18,000,213** | **16,000,196** | 18,001,593 | 25,984 | 0 | 0 |
| dynkeys | 12,965 | 9,812 (76 %) | 3,153 | 0 / 0 | 26,169 | 26,165 | 833,511 | 1,648,080 | 200 | 44,400 |
| nbody | 200 | 31 | 169 | 0 / 0 | 404 | 270 | 2,755 | 26,208 | 0 | 0 |
| binary_trees | 165 | 3 | 162 | 0 / 0 | 167 | 165 | 1,467 | 25,760 | 0 | 0 |

- **Copies:** "descriptor bytes copied" is the size of the descriptor list each new
  shape allocates (it copies its parent's list and appends one entry).
- **Baseline:** the ~160 shapes in every run come from the standard library tables.
- **ctor:** 18 M lookups for 2 M objects is **9 `find_slot` scans per object**, almost
  all misses.
- **particles_pm:** three quarters of all lookups miss. It creates a fresh shape chain
  of up to 64 keys every frame and then migrates it to dictionary mode (600 frames,
  600 migrations).

### 1.2 Profiles

`ctor_long` (1,007 samples), self time:

| function | self |
|---|---|
| `_xzm_malloc_tc` / `_xzm_free_tc` / `__xzm_xzone_free_to_freelist_chunk` / `_malloc_zone_malloc` / `_free` / `DYLD-STUB$$free` | 8.9 / 7.4 / 7.3 / 4.2 / 3.1 / 2.3 % |
| `Context::do_collection` | 6.2 % |
| `transition_add_prop` | **5.2 %** |
| `Context::sweep_one` | 5.1 % |
| `setfield_slow` | **4.9 %** |
| `TableState::raw_get` (includes the inlined `find_slot`) | **4.7 %** |
| `op_add_slow` | 2.8 % |
| `TableState::set_string_key` (includes `find_slot`) | **2.6 %** |
| `Value::boxed_integer` | 2.4 % |
| `CollectVtable::vtable_for::<ShapeData>` trace, i.e. tracing shapes | **2.3 %** |
| `walk_newindex_chain` | **2.2 %** |
| `op_setfield` | 1.5 % |

Inclusive: `setfield_slow` **24.4 %**, `set_string_key` 13.6 %,
`transition_add_prop` 5.2 %, `walk_newindex_chain` 4.6 %, `op_self` 3.1 %,
collection 37.7 %.

`particles_long` (889 samples), self time:

| function | self |
|---|---|
| `TableState::raw_get` (includes `find_slot`) | **7.4 %** |
| `op_getfield` | 6.1 % |
| `_xzm_free_tc` / `_xzm_malloc_tc` / … | 4.2 / 3.6 / 3.0 % |
| `walk_index_chain` | **3.8 %** |
| `Context::do_collection` | 2.5 % |
| `Formatter::run` (string.format) | 2.5 % |
| `transition_add_prop` | **2.0 %** |
| `setfield_slow` | **1.9 %** |
| `walk_newindex_chain` | **1.8 %** |
| `TableState::set_string_key` | **1.7 %** |

Inclusive: `setfield_slow` **11.9 %**, `set_string_key` 9.8 %, `op_self` **7.8 %**,
`walk_index_chain` 6.0 %, `transition_add_prop` 2.9 %, `walk_newindex_chain`
2.8 %, collection 14.7 %.

`find_slot` has no symbol in the release binary; it is inlined into `raw_get`,
`set_string_key` and the IC fill.

## 2. How the system works today

- **`ShapeData`** (`src/env/shape/mod.rs:168`) holds:
  - `parent`, `last_key`, `slot_count`, `mt_cache`, `is_dict`;
  - `transitions: RefLock<TransitionTable>`: two `hashbrown` tables, `by_prop`
    keyed by the key's pointer and `by_mt` keyed by `MtCache` identity, children
    held as `GcWeak`;
  - `descriptors: Box<[Descriptor]>`: the **full** ordered key list, copied from
    the parent plus one (`transition_add_prop`, `shape/mod.rs:447`).
- **`find_slot`** (`shape/mod.rs:385`) is a linear scan over `descriptors`,
  capped by `MAX_PROPERTIES_FAST = 64`.
- **Adding a key:** `TableState::set_string_key` (`src/env/table/mod.rs:281`) scans
  for the key; on a miss it calls `transition_add_prop` and pushes onto
  `properties`. At 64 keys it migrates to dictionary mode (`migrate_to_dict`,
  `table/mod.rs:328`) with a per-metatable sentinel shape, one way.
- **Deleting a key** leaves its slot in place with a nil value (`table/mod.rs:287`).
- **`setmetatable`** calls `transition_set_metatable` (`shape/mod.rs:505`), which
  copies the descriptor list again for the new shape.
- **`InlineCache`** (`src/env/function.rs:83`) has two states, `Empty` and
  `Mono { shape, slot }`, with `slot == ABSENT_SLOT` meaning "key not in shape".
  It holds the shape strongly.
- **`SELF`** (`src/vm/interp.rs:1209`) has no inline cache.

## 3. Findings

### F1. Adding a property does the same lookup three times, and the inline cache never learns it

**Evidence:** `ctor` does 9 `find_slot` scans per object; `setfield_slow` is 24 %
inclusive on `ctor_long` and 12 % on `particles_long`.

For `p.x = v` on a table that doesn't have `x` yet:

1. `op_setfield` (`interp.rs:~1140`) reads the IC. It misses, because the table's
   shape isn't the cached one or the slot is `ABSENT_SLOT`.
2. `setfield_slow` (`interp.rs:1178`):
   - calls `fill_ic_for_constant_key` (`interp.rs:622`): scan 1, a full-length
     miss. It then fills the IC with *(old shape, ABSENT_SLOT)* and emits a
     backward barrier on the prototype;
   - calls `set_slow_body!` → `walk_newindex_chain` → `raw_get`: scan 2;
   - calls `raw_set` → `set_string_key`: scan 3, then `transition_add_prop`
     (`RefLock` borrow, hash probe, `GcWeak::upgrade`), then `properties.push`.
3. The next time the same site runs on another fresh table, the IC matches the
   old shape with `ABSENT_SLOT`, which the handler treats as a miss
   (`slot != ABSENT_SLOT`), so every step repeats.

**What the others do:**
- **JSC `op_put_by_id`:** its metadata holds `oldStructureID, offset,
  newStructureID`. A hit compares the old structure, stores the new structure ID
  and writes the value (survey §4.1; `LowLevelInterpreter64.asm:1757–1815`).
- **V8:** has a `StoreTransition` handler for the same.

**Proposal:**
- **Transition IC.** A `Transition { old_shape, new_shape, slot }` entry. The hit
  path compares the shape, checks the `NEWINDEX` bit (the existing value is nil
  by definition), checks capacity, pushes the value and stores `new_shape`.
  Assert `new_shape.slot_count == slot + 1`.
- **One lookup on the slow path.** Compute the slot once and pass it to the IC
  fill, the `__newindex` decision and the store. Today the IC fill's scan is
  thrown away.
- **Next step:** the survey's shape-carrying `NEWTABLE` (§4.2, LuaJIT `TDUP`,
  V8 boilerplates). A constructor then allocates with its final shape and every
  following `SETFIELD` is a plain slot write.

### F2. Tables used as maps build shape chains that nothing will ever cache

**Evidence:** `particles_pm` creates 17,631 shapes and does 600 dictionary
migrations; `dynkeys` creates 3,153 shapes and does 200. Between them that's
11.7 MB of descriptor copies (more in F3).

`particles_bench.lua` rebuilds a bucket table every frame, keyed by computed
strings (cell coordinates). Each new key is a shape transition from the table's
current shape. The keys differ between frames, so most transitions are new, and
the table walks up to 64 of them before `migrate_to_dict` throws the chain away.
No inline cache ever sees these shapes: all the accesses are `GETTABLE`/`SETTABLE`
with a register key.

**What the others do:**
- **V8 `Map::TooManyFastProperties`** (`src/objects/map-inl.h:319`): for stores
  whose origin is `kMaybeKeyed` (computed keys), an object goes to dictionary
  mode once its out-of-object fields exceed `fast_properties_soft_limit`
  (default 12, `src/flags/flag-definitions.h:3295`), or its in-object count if
  that's larger. Constant-key stores don't count against this.
- **JSC** switches to a cacheable dictionary once the transition chain passes
  `s_maxTransitionLength = 128`, or 512 for plain non-eval `put_by_id`
  (`Structure.h:206–257`).

**Proposal:**
- Track how many properties were added through a computed key (`SETTABLE`,
  `rawset`, `table.*`), for example a small counter in `TableState`. Go to
  dictionary mode past a low limit, V8 uses 12. `SETFIELD`/`SETTABUP` with
  constant keys keep building shapes as today.
- Dictionary lookups got cheaper in PR #218, because string keys now carry their
  hash. So the cost of going to dictionary mode early is lower than when the
  64-key cap was chosen.
- **Validate** with `particles_bench`: its `World` and `Particle` objects must stay
  in shape mode while the buckets go to dictionary mode. Also check `_ENV`:
  module-level `name = value` assignments are `SETTABUP`, so they aren't
  affected.

### F3. Every shape copies its whole key list

**Evidence:** particles copies 10.0 MB of descriptor lists in total; F2's churn
multiplies it.

`ShapeData::descriptors` is a fresh `Box<[Descriptor]>` per shape, built with
`parent.descriptors().to_vec()` plus the new key (`shape/mod.rs:447`). A chain of
*n* keys allocates 1 + 2 + … + n = n(n+1)/2 descriptors. At 16 bytes each, a
64-key chain is about 33 KB. The collector traces every copy: each shape's trace
walks its full list and marks every key. `transition_set_metatable` makes one
more full copy for each (shape, metatable) pair (`shape/mod.rs:538`).

**What V8 does:** a `Map` points at a `DescriptorArray` and records
`NumberOfOwnDescriptors`. When a child extends its parent with the next key and
the parent owns its array, the array is extended in place and shared; the
parent's `owns_descriptors` flag is cleared (`Map::ConnectTransition`,
`src/objects/map.cc:1620–1631`; `ShareDescriptor`). Only a child that *branches*
from a non-tip position copies.

**Proposal:**
- **Share one array along a chain.** Make the descriptor storage a shared GC
  object: a growable array with interior mutability, since ancestors read only
  their prefix. Each shape records its prefix length, which is already
  `slot_count`, plus an "owns the tip" bit. Extending the owning tip appends and
  shares; branching copies the prefix.
- **Share across metatable siblings.** `transition_set_metatable` can share the
  parent's array outright: same keys, same slots.
- **Effect:** per-chain memory and tracing go from O(n²) to O(n), and the shape
  itself shrinks by a pointer plus length.

### F4. Every shape carries two hash tables for its transitions, although most have at most one child

**Evidence:**
- `TransitionTable` (`shape/mod.rs:115`) is two `hashbrown` tables with a
  `MetricsAlloc` each, plus the `RefLock` flag: about 80–90 bytes inside every
  `ShapeData` whether or not it has children.
- The first inserted child allocates the table's heap storage.
- The collector's trace of every shape runs `retain` over `by_prop` and traces
  `by_mt`.
- Tracing shapes is 2.3 % self time on `ctor_long` with only 169 shapes, because
  the run does thousands of cycles and each re-traces every shape and its
  descriptor list.

In a linear chain (a constructor, a class's fields) every shape has exactly one
child; the particles run's 17,262 first-edge insertions were almost all like
that.

**What the others do:**
- **V8:** a map with one transition stores it as an inline weak reference, and
  only a second transition allocates a `TransitionArray`
  (`src/objects/transitions.h:66–72`: "a Map's field either holds an in-place
  weak reference to a transition target … or a TransitionArray").
- **JSC:** `StructureTransitionTable` works the same way with a single-slot flag
  (`isUsingSingleSlot`, `StructureTransitionTable.h:155, 287`).

**Proposal:**
- Replace `TransitionTable` with an enum: none / one inline edge / boxed
  `hashbrown` table for two or more.
- Property and metatable edges can share the representation with a tag, or keep
  one inline slot each.
- Together with F3 this shrinks a shape to roughly its header plus six words and
  makes tracing it cheap.

### F5. Lookups mostly miss, and misses scan linearly through the receiver, the metatable and the class

**Evidence:**
- 75 % of `find_slot` calls miss on particles; `raw_get` is the top self-time
  function there (7.4 %).
- `op_self` is 7.8 % inclusive and `walk_index_chain` 6.0 %.

A method call `obj:m()`:
- `op_self` does `raw_get` on the receiver, a full scan to a miss.
- It sees the `INDEX` bit and goes to `op_self_slow` → `walk_index_chain`, which
  does `raw_get("__index")` on the metatable (a scan), then `raw_get(key)` on the
  class (a scan).
- `SELF` has no IC at all; `GETFIELD` through `__index` fills an IC with
  `ABSENT_SLOT` that never hits (survey §2.2, §3.1, §3.2).

**What the others do:**
- **JSC:** `GetByIdMode::ProtoLoad` / `Unset` (survey §3.1–3.2).
- **V8:** LoadIC prototype handlers with a validity cell. For slow-path name
  lookups V8 also keeps a small global `DescriptorLookupCache` keyed by
  (map, name), 64 entries (`src/objects/lookup-cache.h:19–46`).

**Proposal:**
- **More cache modes:** the survey's §3.5 tagged-union entry with proto-load and
  absent modes on `GETFIELD`, `GETTABUP` and `SELF`, validated by the receiver
  shape plus an `__index` epoch on `MtCache` (survey §3.1 option b).
- **For slow paths that remain:** a small (shape, key) → slot cache, which is cheap
  since both are pointers. For the rare large shape, a lazily built hash map
  instead of the scan (JSC materializes a `PropertyTable` on demand,
  `Structure.h:933–944`).

### F6. Smaller items

- **`pairs` is O(n²) in shape mode.** `TableState::next`
  (`table/mod.rs:502`) finds the current key's position with `find_slot` on every
  call (166,943 calls on particles). The next key lives at the next descriptor
  index (`descriptors[i].slot == i`), so a per-table "last position" hint (like
  `len_hint`), checked by comparing `descriptors[hint].key` to the given key,
  makes the common in-order traversal O(1). LuaJIT's `ITERN` gets the same by
  carrying the index in the loop's control slot (survey §6).
- **Deleted keys keep their slot** (nil-valued, `table/mod.rs:287`), so a table
  that churns keys climbs to the cap. With F2 such tables go to dictionary mode
  early anyway. JSC also caps removal transitions (`s_maxTransitionLengthForRemove
  = 4096`) before going to dictionary mode.
- **`MtEdge` keeps its `MtCache` alive** the way property edges used to keep their
  key strings alive: a shape's metatable edges pin every `MtCache` that was ever
  set on a table of that shape. The fix is the same as `f03aa06`: don't trace the
  key, prune edges whose child is dropped.
- **Cold-path IC fills emit a backward barrier** (`fill_ic`, `interp.rs:587`)
  every time, including the useless (old shape, `ABSENT_SLOT`) fills in F1.
- **Dictionary-mode cliff at 64 keys for `_ENV` and module tables** (survey §3.4)
  is unchanged.

### Survey items already done

- **§5.1 hash caching:** done in PR #218 (`a6cfe13`). Strings store their interner
  hash, and `lua_string_hash` returns it instead of rehashing the bytes.
- **§4.5 array-part sizing:** done in #96.
- **#217 "don't keep strings alive":** done in PR #218 (`f03aa06`).

## 4. Suggested order

1. **F1: transition IC plus a single lookup on the property-add path.** The
   largest measured cost (`setfield_slow` 24 % inclusive on constructors), and it
   touches only the interpreter. Then the survey's shape-carrying `NEWTABLE`
   (§4.2) on top.
2. **F2: dictionary mode for computed-key stores.** Small, and removes the shape
   churn in map-like code (17.6 K shapes and 600 migrations on particles).
3. **F3 + F4: shared key arrays and an inline single transition.** Memory and
   collector cost; both are internal to `shape/mod.rs` and `table/mod.rs`.
4. **F5: proto-load, absent and `SELF` inline caches** (survey §3.1, §3.2, §3.5),
   plus a (shape, key) lookup cache for the slow paths that remain.
5. **F6 items** as they come up; the `next` hint is small and independent.

Each step should be measured with the same counters and profiles: `ctor`,
`particles_pm`, `dynkeys`, plus the survey's `ctor_f`, `setfield_new`,
`field_proto*` and `method_call` benchmarks.

## Appendix: benchmark sources

`dynkeys.lua`:

```lua
-- map-like tables with computed string keys (e.g. a spatial hash rebuilt per frame)
local total = 0
for frame = 1, 200 do
  local buckets = {}
  for i = 1, 300 do
    local key = (i % 17) .. "," .. (i % 13)
    local b = buckets[key]
    if not b then b = {}; buckets[key] = b end
    b[#b + 1] = i
  end
  for _, b in pairs(buckets) do total = total + #b end
end
print(total)
```

`ctor.lua` (`ctor_long` replaces `1000000` with `4000000`):

```lua
local sum = 0
for i = 1, 1000000 do
  local p = { x = i, y = i + 1, z = i + 2 }
  sum = sum + p.y
end
local Vec = {}; Vec.__index = Vec
function Vec.new(x, y) local v = setmetatable({}, Vec); v.x = x; v.y = y; return v end
function Vec:len2() return self.x * self.x + self.y * self.y end
for i = 1, 1000000 do sum = sum + Vec.new(i, 1):len2() end
print(sum)
```

`particles_pm.lua`: `test-files/particles_bench.lua` with `rand` replaced by

```lua
local function rand()
    -- Park-Miller: every product stays below 2^53, so doubles (LuaJIT) agree.
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647
end
```

`particles_long.lua` is `particles_pm.lua` with the last loop as
`for _ = 1, frames * 5 do world:step(0.1) end`.

The counters were a temporary patch: an atomic counter per event listed in §1.1,
incremented in `shape/mod.rs` and `table/mod.rs`, and printed from a `Drop` impl
on `Lua` when `TCVM_SHAPE_STATS` is set.
