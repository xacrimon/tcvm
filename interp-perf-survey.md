# Interpreter performance survey: what JSC, V8 and LuaJIT do that tcvm doesn't

Scope: the bytecode interpreter, excluding the call/return path (#171 / PR #172).
Weighted toward table- and field-heavy Lua ("realistic" OOP-style code), with
numeric loops second. Survey only — nothing here has been prototyped; every
"expected gain" is an estimate unless a measurement is cited.

Sources, all read at the stated revisions:

| | checkout | rev | files |
|---|---|---|---|
| tcvm | this repo, branch `8-byte-value` | `7c0bf4b` | `src/vm/interp.rs`, `src/env/{table,shape,value,string,function}.rs`, `src/dmm/*` |
| LuaJIT 2.1 | `../LuaJIT` | `c6ffc141` (2026-09-08) | `src/vm_arm64.dasc`, `lj_bc.h`, `lj_parse.c`, `lj_str.c`, `lj_obj.h` |
| V8 | `../v8` | `e4c54cbf078` (2026-09-15) | `src/ic/*`, `src/interpreter/interpreter-generator.cc`, `src/common/globals.h`, `src/builtins/builtins-constructor-gen.cc` |
| JavaScriptCore | `../jsc` (sparse: `Source/JavaScriptCore`) | `7429477a` (2026-09-16) | `llint/LowLevelInterpreter64.asm`, `llint/LLIntSlowPaths.cpp`, `bytecode/{BytecodeList.rb,GetByIdMetadata.h,ArithProfile.h,ArrayProfile.h}`, `jit/JITMathIC.h` |
| PUC Lua 5.5.1 | `../lua` | `7579fc9d` | `lvm.c`, `lvm.h`, `lstring.h`, `lobject.h` |

Measurements: M4 Pro, macOS 26, `cargo build --release`, `hyperfine -N -w 1 -r 5`,
against Homebrew `luajit -j off` and `lua5.5`. Benchmark sources are in
appendix A; they are deliberately single-pattern microbenchmarks, not programs.

---

## 1. Summary and ranking

tcvm's *monomorphic own-property* fast path is already the best of the three
interpreters (`p.x` reads: 0.77× LuaJIT, 0.61× PUC). Everything below is about
the cases that fall off that path. Ranked by (measured gap × how common the
pattern is in ordinary Lua):

| # | Technique | Pattern it fixes | Today vs `luajit -j off` | Who does it | Est. gain on pattern | Depends on |
|---|---|---|---|---|---|---|
| 1 | **Transition IC + shape-carrying `NEWTABLE`** (§4.1, §4.2) | `{x=..,y=..}`, `self.x = v` in constructors | 2.1–2.7× slower | JSC put_by_id transition mode; V8 StoreTransition handler + literal boilerplates; LuaJIT `TDUP` | 1.5–2× on constructors | — |
| 2 | **`__index`-chain (proto-load) IC on `GETFIELD`/`SELF`** (§3.1) | `obj:method()`, `obj.method`, class-based OOP | 1.3× (small tables) to 2.1× (12 fields / 30 methods) | JSC `GetByIdMode::ProtoLoad`; V8 LoadIC handler + prototype validity cell | 1.5–2× on method lookup; removes the O(fields+methods) linear scans | — |
| 3 | **Boxed-integer allocation** (§7) | any `integer` accumulator > 2³¹ | 3.4× LuaJIT, **6.1× PUC**, 12× tcvm's own small-int path | V8 HeapNumber via bump allocation; JSC/LuaJIT avoid it (doubles) | up to 5× on affected loops | GC allocator (§4.3) or a value-repr change |
| 4 | **Dictionary-mode caching / global cells** (§3.4) | > 64 globals, > 64-key module tables | 3–4.5× slower; 3.7× slower than the *same code* with < 64 globals | V8 `PropertyCell` in feedback slot; JSC `GlobalVar` direct slot pointer | 3× on affected programs | — |
| 5 | **String representation** (§5) | `a..b..c`, string keys in dict tables, interner | concat 5.3× slower | LuaJIT: cached hash + sparse hashing + inline bytes; V8/JSC ropes | 2–3× on string building | — |
| 6 | **`pairs`/`next` specialization** (§6) | `for k,v in pairs(t)` | 6.2× LuaJIT (beats PUC) | LuaJIT `ISNEXT`/`ITERN` despecialization; V8 `ForInNext` enum cache | 3–4× on iteration | interacts with #172 |
| 7 | **Type-specialized (quickened) bytecodes** (§8) | arithmetic/compare sites, `t[i]` vs `t[k]`, mixed int/float | int/float mixes go out of line; `t[i]` calls out-of-line `raw_get` | LuaJIT compile-time operand-typed ops + runtime despecialization; JSC `ArithProfile`→`JITMathIC` (baseline JIT only); V8 feedback only | 5–15% on numeric loops; more where mixes are common | mutable `code` |
| 8 | **Polymorphic IC (2–4 entries)** (§3.3) | one site seeing several shapes | 1.6× slower than the mono control | V8 (4 maps), JSC baseline `PolymorphicAccess` (LLInt is mono) | 1.5× on poly sites | — |
| 9 | **Absent-key ("Unset") IC mode** (§3.2) | `if t.field == nil`, optional fields | 1.8× slower | JSC `GetByIdMode::Unset` | ~1.8× on the pattern | — |
| 10 | **Inline property storage / smaller table object** (§4.4) | every field access, every allocation | 184 B + 1–2 mallocs per empty table (LuaJIT: 64 B) | V8 in-object slack tracking; JSC inline capacity; LuaJIT colocated array | one dependent load off every IC hit; halves constructor mallocs | pairs with #1 |
| 11 | **Pointer-based upvalues** (§9) | closures, module-level `local` functions | `GETUPVAL` 2.8× PUC (#130); 13–19 instr / 1–2 branches | PUC & LuaJIT `UpVal.v` pointer | −8 instr, −2 branches per upvalue access | — |
| 12 | Small handler fixes (§10) | `LEN`, `GETTABLE` int arm, `SETTABLE` double lookup, `SETFIELD` bounds check | 1.5× PUC on `t[i]` | — | a few % each | — |

Items 1, 2, 4, 9 share one prerequisite: extending `InlineCache` from a
two-variant enum to a small tagged union with more modes (§3.5). Item 7 (and 6,
if done LuaJIT-style) needs the bytecode to be patchable at runtime (§8.4).

---

## 2. Where tcvm stands (measured)

`tcvm / luajit -j off / lua5.5` wall time, 5 runs, ±3 % or better. Ratios > 1
mean tcvm is slower.

| bench | pattern | tcvm | luajit | lua5.5 | vs luajit | vs lua |
|---|---|---|---|---|---|---|
| `field_own` | `acc + p.x + p.y` (IC hit) | 0.125 | 0.162 | 0.206 | **0.77** | **0.61** |
| `setfield_own` | `p.x = i; p.y = i` (IC hit) | 0.118 | 0.137 | 0.198 | **0.86** | **0.59** |
| `field_mono_ctl` | `objs[i%2+1].x`, same shape | 0.147 | 0.143 | 0.224 | 1.03 | 0.66 |
| `field_poly2` | same, two shapes alternating | 0.238 | 0.145 | 0.228 | 1.64 | 1.04 |
| `field_poly4` | four shapes | 0.236 | 0.147 | 0.237 | 1.61 | 1.00 |
| `field_absent` | `p.z == nil` (key absent) | 0.195 | 0.107 | 0.201 | 1.83 | 0.97 |
| `field_proto` | `obj.area` via `__index` (3 fields, 1 method) | 0.277 | 0.212 | 0.235 | 1.31 | 1.18 |
| `field_proto2` | two-level `__index` chain | 0.372 | 0.289 | 0.349 | 1.29 | 1.07 |
| `field_proto_big` | `obj.m29` (12 fields, 30 methods) | 0.451 | 0.213 | 0.238 | **2.12** | 1.89 |
| `method_call` | `obj:get()` (includes CALL, see #171) | 0.217 | 0.095 | 0.137 | 2.27 | 1.58 |
| `ctor_f` | `{x=i, y=i+1, z=i+2}` | 0.417 | 0.201 | 0.280 | **2.07** | 1.49 |
| `ctor_arr_f` | `{i, i+1, i+2}` | 0.325 | 0.144 | 0.188 | **2.25** | 1.72 |
| `setfield_new` | `p={}; p.x=i; p.y=i; p.z=i` | 0.474 | 0.373 | 0.645 | 1.27 | 0.73 |
| `globals_small` | `g = g + 1`, 1 user global | 0.122 | 0.097 | 0.174 | 1.25 | 0.70 |
| `globals_dict` | same, after 100 user globals | 0.451 | 0.099 | 0.170 | **4.55** | 2.66 |
| `dict_field_f` | `t.k40 + t.k70` on an 80-key table | 0.456 | 0.217 | 0.220 | **2.10** | 2.07 |
| `str_key_var` | `t[k1] + t[k2]`, string in register | 0.189 | 0.163 | 0.242 | 1.16 | 0.78 |
| `array_get_f` | `acc + t[i]`, 1000-element array | 0.147 | 0.148 | 0.095 | 0.99 | 1.55 |
| `ipairs_f` | `for i,v in ipairs(t)` | 0.292 | 0.114 | 0.290 | 2.56 | 1.01 |
| `pairs50` | `for k,v in pairs(t)`, 50 string keys | 0.191 | 0.031 | 0.231 | **6.24** | 0.83 |
| `concat` | `a .. b .. c .. d` | 0.371 | 0.070 | 0.088 | **5.30** | 4.22 |
| `smallint_add` | `acc = acc + 1` from 0 | 0.043 | 0.064 | 0.086 | **0.67** | **0.50** |
| `bigint_add` | `acc = acc + 1` from 3·10⁹ | 0.518 | 0.151 | 0.084 | **3.42** | **6.14** |

Two of these change the reading of everything else:

- **Boxed integers.** The first round of these benchmarks used integer
  accumulators that crossed 2³¹ (`array_get`, `ctor`, `dict_field`, `ipairs`),
  and those all showed 2–5× *worse* ratios than the `_f` (float accumulator)
  variants above. On the 8-byte `Value` every integer outside `i32` is
  `Gc::new(mc, i64)` — a `Box::new` (`src/dmm/context.rs:360`) plus linked-list
  insert — on every arithmetic result (`src/env/value.rs:133–146`). See §7.
- **Calls.** `method_call`, `ipairs_f` and `pairs50` include one native or Lua
  call per iteration; #171 measured ~300 instructions per call, which is most
  of their gap. Only the non-call part is attributed to the techniques here.

### 2.1 What the IC hit path costs today

`op_getfield` (`src/vm/interp.rs:1077`) fast path, arm64 release, 25
instructions, no frame:

```
ldr   x8, [x24, x8, lsl #3]        ; R[table]
asr/cmn/b.ne                       ; table tag
ldr   x10, [x27, #0x30]            ; ds.ic_table
ldr   w9, [x11] ; tbz              ; InlineCache discriminant (Mono?)
ldr   w10, [x11, #4]               ; slot
ldr   x9, [x8, #0x40]              ; TableState.shape
ldr   x11, [x11, #8]               ; cached shape
cmp / ccmn (slot != ABSENT) / b.ne
ldr   x8, [x8, #0x8]               ; properties.ptr        <- extra chase (§4.4)
ldr   x8, [x8, x10, lsl #3]        ; properties[slot]
cmp nil / b.ne                     ; nil -> check __index bit
str ; ldr ; and ; ldr ; br         ; dispatch
```

That is already tighter than LuaJIT's `BC_TGETS` (`vm_arm64.dasc:3213`: `sid & hmask`,
node walk, `nomm` check — no IC, a hash probe every time), which is why the hit
benchmarks win. The dependent-load depth is 4 (reg → table → shape/properties →
value); JSC's `GetByIdMode::Default` is 3 because the property lives in the
object's inline storage.

### 2.2 What a miss costs today

`fill_ic_for_constant_key` (`interp.rs:639`) + `table_get_slow_body`
(`interp.rs:460`) + `walk_index_chain` (`interp.rs:3493`) on `obj.method`:

1. `read_ic` → `ic_check` succeeds with `slot == ABSENT_SLOT` → treated as a miss
2. `getfield_slow`: `fill_ic_for_constant_key` → `shape.find_slot(key)` — **linear
   scan over all descriptors** (`shape/mod.rs:363`), full length on a miss
3. `table_get_slow_body`: `t.raw_get(k)` → `get_string_key` → `find_slot` **again**
4. `has_mm(INDEX)` → `walk_index_chain`: `mt.raw_get("__index")` → `find_slot` on the
   metatable; then `Class.raw_get(key)` → `find_slot` on the class (or, past 64
   keys, a dict-mode probe that **re-hashes the key's bytes** — §5.1)

So `obj.m29` with 12 fields and 30 methods is 12 + 12 + ~2 + 30 pointer
compares plus five dependent object hops, every time. LuaJIT does three O(1)
hash probes (`vmeta_tgets` → `lj_meta_tget`), and its cost does not grow with
the table. Profile of `ctor_long` (appendix B) shows the same shape for stores:
32 % malloc/free, 14 % `transition_add_prop`, 32 % in
`setfield_slow`/`set_string_key`/`raw_get`/`raw_set`/`maybe_update_mt_bit`, 8 % in
the handlers doing the actual work.

---

## 3. Property-access inline caches beyond monomorphic-own

tcvm's `InlineCache` (`src/env/function.rs:80`) has two states, `Empty` and
`Mono { shape, slot }`, with `slot == ABSENT_SLOT` meaning "key not in shape".
The engines below have between four and nine states per site. The point of
each extra state is to turn a specific class of miss into a hit.

### 3.1 Prototype-chain load (`__index` table) — JSC ProtoLoad, V8 LoadIC handlers

**JSC LLInt** (`bytecode/GetByIdMetadata.h:33–48`, `LowLevelInterpreter64.asm:1653–1696`):
`op_get_by_id` metadata is a 16-byte union with modes `Default` (structure +
offset on the receiver), `ProtoLoad` (structure + offset + `cachedSlot` — a
pointer to the *holder* object up the prototype chain), `Unset`, `ArrayLength`.
`ProtoLoad` fast path: compare receiver structure, load `cachedOffset` from the
cached holder, done — **no walk, no holder structure check**. Validity comes
from watchpoints: the slow path only installs `ProtoLoad` when every structure
on the chain is watchable and the holder's property has a replacement
watchpoint; any write fires the watchpoint, which clears the LLInt cache. It
is also gated by `hitCountForLLIntCaching` (`Options::prototypeHitCountForLLIntCaching`,
default 2, `runtime/OptionsList.h:558`) so a site must resolve through the
prototype a couple of times before the cache is built.

**V8** (`src/ic/accessor-assembler.cc:1283–1340`, `HandleLoadICHandlerCase`):
the feedback slot holds `(map, handler)`; a load-from-prototype handler
carries a `validity_cell` (`Map::prototype_validity_cell_`) that is
invalidated when any prototype map on the chain changes, plus the holder
(weak) and the field index. Fast path: map compare, validity-cell compare,
load from holder.

**tcvm design.** The receiver's `Shape` already fixes the metatable identity
(`ShapeData::mt_cache`, `shape/mod.rs`), so a receiver-shape compare proves
"same metatable". What it does not prove is that `mt.__index` still points at
the same holder table and that the holder still has the key at the same slot.
Two ways to get a validity check without watchpoints:

- *(a) Re-validate the holder.* Cache `{recv_shape, holder: Table, holder_shape,
  holder_slot}`. Hit path: receiver shape compare; load
  `recv.metatable.properties[__index_slot]` and compare to `holder` (needs the
  metatable's `__index` slot in the entry too, or a `MtCache`-level cached
  `__index` value); holder shape compare; load `holder.properties[holder_slot]`.
  ~7 loads, 3 compares. Handles method reassignment (value is re-read) and
  `__index` reassignment (holder compare fails).
- *(b) An epoch on `MtCache`.* `MtCacheData` already sees every write of a
  metamethod-named key (`TableState::maybe_update_mt_bit`, `table/mod.rs`) —
  bump a `u32 index_epoch` there whenever `__index` is written. Cache
  `{recv_shape, epoch, holder, holder_shape, holder_slot}`; hit path:
  receiver shape compare, `shape.mt_cache.epoch` compare (2 loads), holder
  shape compare, load. This is V8's validity cell with the cell hung off the
  metatable identity, which is exactly where Lua's "prototype" lives. Holder
  shape compare covers a method being added/removed on the class; a method
  *replaced* (same shape, same slot, different value) is handled by the value
  being re-read from the holder. Multi-level chains (`field_proto2`) cache
  the final holder and every intermediate `MtCache` epoch, or just cap at two
  levels like JSC caps chain length.

Either way the miss path should also stop scanning: `fill_ic_for_constant_key`
and `table_get_slow_body` each call `find_slot`; one lookup should feed both.

`SELF` (`interp.rs:1204`) has no IC at all ("the `e` slot is free for one"),
and it is *the* method-call instruction, so it should get the same entry
format. Expected: `field_proto_big` from 2.1× to ≈1× LuaJIT; `method_call`'s
non-call portion likewise.

### 3.2 Absent key — JSC `Unset` mode

`GetByIdMode::Unset` (`GetByIdMetadata.h:36`, asm `:1689`): structure compare
→ return `undefined`, with the chain guaranteed absent by watchpoints. tcvm
stores `ABSENT_SLOT` but then treats it as a miss (`interp.rs:1097`,
`if slot != ABSENT_SLOT`) and falls into the full slow path every time
(`field_absent`: 1.83× LuaJIT). With no metatable, or an `MtCache` whose
`INDEX` bit is clear, an `ABSENT_SLOT` hit can return nil directly: one
`has_mm` check that the handler already does on the nil-value path. With an
`__index` *table*, it becomes a §3.1 proto-load whose holder also lacks the
key — cache "absent through the chain" with the same epoch guard.

### 3.3 Polymorphic entries — V8 (4), JSC baseline

V8 goes `MONOMORPHIC → POLYMORPHIC` (up to 4 `(map, handler)` pairs, linear
scan) `→ MEGAMORPHIC` (`InlineCacheState`, `src/common/globals.h:1896`); the
megamorphic state probes a global `StubCache` keyed by `(map, name)`
(`stub-cache.h:88–91`, 4096 + 1024 entries) so even megamorphic sites avoid
a dictionary lookup. JSC's LLInt is monomorphic; the baseline JIT's
`PolymorphicAccess` stubs handle several structures.

tcvm's `field_poly2` is 1.6× the mono control because every access refills
the entry (`fill_ic` also emits a backward barrier on the prototype). Two
entries per site would cover the common Lua case of two constructors
producing the same "class" with keys in different order — which also argues
for the compiler emitting constructor keys in a canonical order so that
case doesn't arise (LuaJIT's `TDUP` makes both shapes identical for free).
Cost: the entry grows from 16 to ~32 bytes; the hit path gains one compare on
the second entry only. Worth doing after §3.1 since proto-load entries are
where polymorphism actually shows up (a method called on two subclasses).

### 3.4 Dictionary-mode tables and globals — V8 `PropertyCell`, JSC `GlobalVar`

Every table past `MAX_PROPERTIES_FAST = 64` string keys (`shape/mod.rs:75`)
migrates one-way to dict mode (`table/mod.rs:335`) with a per-metatable
sentinel shape, so no IC ever hits it again. That includes `_ENV` once a
program has ~30 user globals on top of the ~35 standard ones — `globals_dict`
is 3.7× slower than `globals_small` for identical loop code, and 4.5× LuaJIT.
Module tables with > 64 functions hit the same cliff (`dict_field_f`: 2.1×).
#131 already notes "in table dict-mode, we seem to be slower than PUC".

V8 keeps the global object in dictionary mode *and* makes global loads the
cheapest access in the system: the feedback slot holds a weak pointer to the
`PropertyCell` (`accessor-assembler.cc:3836–3858`, `LoadGlobalIC_TryPropertyCellCase`)
— load the cell's value, one hole check, no map compare. JSC's
`op_get_from_scope` with `GlobalVar` resolution (`LowLevelInterpreter64.asm:2911–2917`)
stores a raw pointer to the variable slot in the metadata and does a single
`loadq [t0]`; correctness comes from the `WatchpointSet` on the symbol table
entry.

Options for tcvm, cheapest first:

1. *Raise the cap for `_ENV`* (or generally): the cap exists to bound the
   transition tree; a table that is *itself* the only one with its shape
   (globals) costs nothing extra in the tree. 256 would clear most programs.
   Doesn't fix module tables shared across many shapes... but those are also
   singletons in practice.
2. *Cells in dict mode*: dict entries store `Gc<Lock<Value>>`; the IC caches
   `(dict_identity, cell)`; hit path is a pointer compare + load, identical
   to V8. Costs one allocation per dict key and an indirection on non-IC
   paths (`next`, `raw_get`).
3. *Position IC*: cache `(dict generation, bucket index)`, bump the generation
   on rehash/reap. Cheap to add given `hash_part::position` exists, but a
   generation bump on every insert past capacity makes it fragile.

Option 1 plus a proper hash for dict mode (§5.1) probably covers 90 %.

### 3.5 Entry format

All of the above want the entry to be a 32-byte tagged union rather than a
2-variant enum: `{kind: u8, slot: u32, shape: Shape, holder: Option<Table>,
holder_shape/epoch}`. JSC's union (`GetByIdModeMetadata`, 16 bytes with the
mode byte overlapping the holder pointer's high bits) shows how to keep it
small; tcvm can afford 32 bytes since `ic_table` is per prototype, not per
instruction. Keep the `Mono` fast path's check order (discriminant, shape,
slot) — it is one `ldr w9 / tbz` today.

---

## 4. Table construction and allocation

### 4.1 Transition IC on `SETFIELD` — JSC transition mode, V8 `StoreTransition`

JSC `op_put_by_id` metadata (`BytecodeList.rb:294`): `oldStructureID, offset,
newStructureID, structureChain`. Fast path (`LowLevelInterpreter64.asm:1757–1815`):
compare old structure; if `newStructureID != 0` it is a cached *transition*:
walk the cached prototype chain checking each structure (for setters), store
the new structure ID into the object, store the value at `offset`. The slow
path (`LLIntSlowPaths.cpp:1118–1140`) only caches a transition when the new
structure's out-of-line capacity equals the old one's — i.e. **no storage
reallocation is needed** — and it isn't a dictionary. V8's
`StoreTransitionHandler` is the same idea with the transition map's validity
cell.

tcvm today: a `SETFIELD` on a key absent from the shape always takes
`setfield_slow` → `fill_ic_for_constant_key` (scan) → `walk_newindex_chain` →
`raw_set` → `set_string_key` (scan again) → `transition_add_prop` (hash probe
in the parent's transition table + `GcWeak::upgrade`) → `Vec::push`. That's the
32 % + 14 % in the constructor profile. A `Transition { old_shape, new_shape,
slot }` entry makes the hit path: shape compare, `has_mm(NEWINDEX)` bit check
(only when the existing value is nil, which it is by definition here — so
check the bit unconditionally), capacity check on `properties`, push, store
new shape. `new_shape.slot_count == slot + 1` is the invariant to assert.

### 4.2 Shape-carrying `NEWTABLE` — LuaJIT `TDUP`, V8 boilerplates, JSC `ObjectAllocationProfile`

LuaJIT (`lj_parse.c:1850–1905`): any constructor with at least one constant
key builds a *template table* at compile time containing every constant key
(with a placeholder value for non-constant ones) and emits `TDUP` — a
`lj_tab_dup` memcpy of the template — followed by `TSETS` for the non-constant
values into keys that already exist. Constant-only tables need no stores at
all. `TNEW` carries `asize | hbits << 11` so even the dynamic case allocates
the final size once (`vm_arm64.dasc:3128–3155`).

V8 `CreateObjectLiteral` (`interpreter-generator.cc:2693`) allocates from an
`AllocationSite` boilerplate — a fully built object whose map already has
every literal property — and `CreateShallowObjectLiteral`
(`builtins-constructor-gen.cc:600`) copies it. JSC `op_new_object`
(`BytecodeList.rb:597`) carries `inlineCapacity` from the parser's count of
`this.x = ` assignments in the constructor, and `ObjectAllocationProfile`
caches the structure + allocator so repeated `new C()` get identical
structures.

For tcvm the compiler already knows the constant string keys of a constructor
in order (#33 notes `compile_expr_table` counts entries). Emit
`NEWTABLE dst, K[shape_template]` where the constant is a *shape* (or a list
of keys resolved to a shape at load time), allocate `properties` at
`slot_count` filled with nil, and let the following `SETFIELD`s hit the plain
`Mono` IC — every store becomes a slot write. With §4.1 in place this is a
further 14 % (`transition_add_prop`) plus the `Vec` regrowths
(`finish_grow`, 44 samples). Array constructors (`ctor_arr_f`, 2.25×) get the
same from an array-size hint: `SETLIST` today does `Value::integer` +
`raw_set` + `resize` per element (`interp.rs:2847`).

Doing this well also fixes the polymorphism-by-key-order problem in §3.3.

### 4.3 Allocation fast path — V8/JSC bump allocation, LuaJIT `lj_alloc`

dmm's `Context::allocate` (`src/dmm/context.rs:360`) is `Box::new` per object
— a system `malloc` (`_xzm_xzone_thread_cache_fill_and_malloc` is the top
symbol in the constructor profile at 15 %, malloc+free together 32 %). V8
bump-allocates in the new space (~10 instructions inline in the interpreter),
JSC uses per-size-class `LocalAllocator` free lists (`allocateCell` is ~15
instructions inline in `op_new_object`), LuaJIT uses its own dlmalloc
derivative. A size-class free-list allocator inside the arena (segregated
pages, sweep frees onto the list) would make table, closure, boxed-int and
string allocation each ~10 instructions instead of a malloc round trip, and
would also fix §7 without touching the value representation. This is really
GC-roadmap work (#160's GC item) but it is the single largest line in the
constructor profile, so it belongs in this ranking.

### 4.4 Inline property storage and object size — V8 in-object properties, JSC inline capacity

`size_of::<TableState>() == 168` (measured), plus the 16-byte `GcBox` header:
184 bytes and one malloc before any content, then a second malloc for
`properties` on the first string key and a third for `array`. A LuaJIT
`GCtab` is 64 bytes with the array part colocated for constructors; PUC's
`Table` is 56. Much of the 168 is three separate hashbrown/`Vec` headers with
their own `MetricsAlloc` (an `Rc` clone each) and a 40-byte `Option<DictState>`
that is `None` for almost every table.

V8 objects carry N in-object property slots sized by slack tracking on the
constructor; JSC's `JSFinalObject` has `inlineCapacity` slots before the
butterfly. Two ways to get there in tcvm: (a) make `TableState` a DST with
`N` trailing `Value` slots (N from the §4.2 template, default 4–6) and spill
to `properties` beyond N — this removes the `properties.ptr` load from the IC
hit path (§2.1) and the second malloc from every small object; (b) at minimum,
box the dict state and the misc hash behind one `Option<Box<Slow>>` so the
common table is ~80 bytes. (a) changes `property_at`/IC slot addressing
(`slot < N ? inline : spill[slot - N]`, or JSC's negative/positive offset
trick); the shape already knows the total count.

### 4.5 Array part sizing

`raw_set` on an integer key does `self.array.resize(index, nil)`
(`table/mod.rs:269`) for *any* positive index — `t[1000000] = 1` on an empty
table allocates 8 MB, and sparse integer keys never go to `misc_hash`. PUC's
`luaH_resize`/`rehash` computes the array size as the largest `n` such that
more than half of `1..n` is used (`ltable.c:446`, `computesizes`) and puts the rest
in the hash part; LuaJIT does the same in `lj_tab_resize`/`rehashtab`. This is
a correctness-adjacent memory issue more than a speed one, but it should be on
the list before anyone benchmarks sparse tables.

---

## 5. Strings

### 5.1 Hash caching and identity hashing

`impl Hash for LuaString` hashes the **bytes** (`src/env/string.rs:67`), and
`lua_string_hash` (`table/hash_part.rs:76`) is what dict-mode tables use for
every probe — so every `t.k40` on an 80-key table re-hashes `"k40"` with
foldhash, and every interner probe (`string.rs:95`) hashes the full input.
CLAUDE.md says "interned strings with cached hash"; `StringData` has no hash
field, so that's stale. Because strings are interned, the *pointer* is a
valid identity hash for dict tables (the shape transition table already keys
on it, `shape/mod.rs:398`); for the interner itself, cache the hash in
`StringData` (LuaJIT `GCstr.hash`/`sid`, `lj_obj.h:306–313`; PUC
`TString.hash`, `lobject.h:410`).

LuaJIT's `hash_sparse` (`lj_str.c:76–96`) hashes only 4 samples of any string
regardless of length, so interning a 1 MB string costs O(1) hashing (with a
dense fallback under `LUAJIT_SECURITY_STRHASH` when a chain gets long). PUC
interns only strings ≤ `LUAI_MAXSHORTLEN = 40` (`lstring.h:29`); long strings
are plain allocations with a lazily computed hash. tcvm interns everything and
hashes everything fully.

### 5.2 Concatenation

`op_concat` (`interp.rs:1781`) is binary and builds into a fresh `Vec::new()`
that regrows (the `concat` profile, appendix B: `realloc`/`finish_grow`/`memmove`
≈ 40 %, malloc/free ≈ 25 %, interner probe + `memcmp` ≈ 12 %). `a..b..c..d` is
three interned intermediates. LuaJIT `BC_CAT` (`lj_bc.h:121`) and PUC
`OP_CONCAT` take a register range and build once in a preallocated buffer
(`lj_meta_cat` computes the total length first); #33 already lists the n-ary
form. V8 (`ConsString`) and JSC (`JSRopeString`) go further and make
concatenation O(1) by building a rope, flattened on first byte access — a big
win for string-building loops, but it makes every string consumer check for
ropes; for a Lua VM the buffer approach plus non-interned long strings gets
most of the benefit with none of the intrusion.

`StringData` is `Box<[u8]>` behind a `GcBox` (16 + 16 bytes + separate
buffer): two mallocs per string. LuaJIT/PUC put the bytes inline after the
header. Same DST trick as §4.4(a).

---

## 6. `pairs`/`next` and `ipairs`

LuaJIT (`vm_arm64.dasc:3653–3690`, `BC_ISNEXT`): the parser emits
`ISNEXT`+`ITERN` for any `for k,v in <expr>` whose iterator is `pairs(t)` or
`next, t`. At loop entry `ISNEXT` checks that the iterator is the builtin
`next` (`ffid == FF_next_N`), the state is a table, the control is nil; if so
it seeds the control with a raw index and `ITERN` (`:3600–3651`) walks the
array then the hash nodes directly in ~15 instructions, no call. If any check
fails, `ISNEXT` **rewrites itself to `JMP` and the `ITERN` to `ITERC`** — the
canonical despecialization pattern. V8's `ForInPrepare`/`ForInNext`
(`interpreter-generator.cc:3340–3375`) cache the map's enum-cache key array and
check only `receiver.map == cache_type` per step.

tcvm's `TFORCALL` (`interp.rs:2795`) runs `invoke_metamethod!` →
`schedule_meta_call` for every step, and `TableState::next` (`table/mod.rs:380`)
re-locates the previous key with `shape.find_slot` — O(n) per step, so
`pairs` over an n-key shape table is O(n²) (bounded by 64², but 50 keys ×
200k loops is what `pairs50` measures). The V8 approach maps directly onto
shapes: `TFORPREP` recognises `(next, table, nil)`, stores `(shape, slot
cursor)` in the hidden control slot, and a `TFORNEXT` opcode walks
`descriptors[cursor..]` while the shape is unchanged, then the array and misc
parts by index. Falls back to the generic call on any mismatch (LuaJIT's
despecialization, or simply a branch in the handler since the check is
cheap). `ipairs` is the same shape with a `ipairs_aux` identity check.
Expected: `pairs50` from 6.2× to ≈1.5× LuaJIT. Requires the iterator
identity to be checkable — the builtins are `NativeFn` pointers today, which is
enough.

---

## 7. Boxed integers on the 8-byte `Value`

Not a "technique the others use" but the largest single cliff measured, and it
interacts with §4.3. `Value::integer` (`value.rs:133`) boxes anything outside
`i32`; `bigint_add` runs 12× slower than `smallint_add` and 6× slower than
PUC, because every `ADD` result is a `malloc` + GC-list insert, and every
operand read is a pointer chase. Realistic triggers: byte offsets and sizes,
`os.time()`-derived values, hashes and PRNG state (`math.random` seeds), IDs,
anything using `<<`/`|` on 32-bit quantities, `string.len` sums.

What the others do: V8 has the same Smi cliff (31-bit on pointer-compressed
builds) but falls to `HeapNumber` via bump allocation, and Maglev/TurboFan
unbox; JSC's NaN-box holds a full `int32` and promotes to `double` — legal in
JS, not in Lua 5.5 where integers are 64-bit. LuaJIT is 5.1 (doubles only).

Options, in order of cost:

1. **Free-list / bump allocation for `i64` boxes** (§4.3 applied to one
   type) — turns ~100 ns of malloc/free/sweep into ~10 instructions; still a
   GC object per result.
2. **Widen the inline integer.** The immediate tag leaves a 48-bit payload
   (`value.rs:33–49`); the top word is all-ones only because that makes
   `both_small` one `and`+`cmp`. A 47-bit inline integer (`asr #17` decode)
   would cover offsets, timestamps in µs, and most hashes, and keep the
   allocation for genuinely 64-bit values. Costs the `both_small` trick and
   the `w`-register decode; needs measuring on `primes`/`collatz`.
3. **Per-thread box cache** for the loop-carried case (the result box is
   reused when the destination register already holds a box with refcount
   one) — fragile with a tracing GC, not recommended.

Whichever is chosen, the survey's other numbers (§2) should be re-run with
integer accumulators afterwards, since they currently hide behind `_f`.

---

## 8. Type-specialized bytecodes (quickening)

### 8.1 What the engines actually do

- **LuaJIT** specializes at *compile time* by operand kind: `ADDVN/ADDNV/ADDVV`
  (`lj_bc.h:102–118`), `ISEQV/ISEQS/ISEQN/ISEQP`, `TGETV/TGETS/TGETB`,
  `USETV/USETS/USETN/USETP`. Within a handler the *number* check is fixed
  order (integer first on dual-number builds). Runtime rewriting is used for
  control, not types: `ISNEXT→JMP`/`ITERN→ITERC` (§6), and the hot-loop /
  hot-function counters turn `FORL→IFORL/JFORL`, `LOOP→ILOOP/JLOOP`,
  `FUNCF→IFUNCF/JFUNCF` to hand off to the JIT or blacklist.
- **V8 Ignition** does *not* rewrite bytecode. `Add` calls the generic
  `BinaryOpAssembler` with the Smi path first and records
  `BinaryOperationFeedback` (`globals.h:2511`: SignedSmall → Number →
  NumberOrOddball → String → Any, a lattice) in the feedback vector for
  Sparkplug/Maglev/TurboFan. Its interpreter-level type specialization is all
  in ICs (property access, `GetKeyedProperty` with elements-kind handlers,
  `ForInNext`).
- **JSC LLInt** likewise keeps fixed-order fast paths (`binaryOpCustomStore`,
  `LowLevelInterpreter64.asm:1230–1275`: int/int, then not-int/int mixes into
  doubles) and records an `ArithProfile` (`bytecode/ArithProfile.h`,
  `ObservedType` bits Int32/Number/NonNumber per operand). The **`JITMathIC`**
  (`jit/JITMathIC.h`, `JITAddGenerator` etc.) is the "IC for non-property
  ops" you remembered: in the *baseline JIT* each arithmetic site gets a
  repatchable inline snippet generated from its `ArithProfile`; on a type
  miss the out-of-line path regenerates the snippet with the wider profile
  (`generateOutOfLine`, `m_generateFastPathOnRepatch`). It is code-generation,
  not an interpreter technique — the interpreter analog is quickening.
  Other non-property LLInt caches: `op_get_from_scope` resolve mode (§3.4),
  `op_new_object` allocation profile (§4.2), `op_instanceof` (two
  `GetByIdModeMetadata`), `op_to_this` cached structure, `op_get_by_val`
  `ArrayProfile` (observed indexing types, `ArrayProfile.h:214`),
  `op_enumerator_*` for-in modes, and the call-link infos (excluded here).

So among the three, only LuaJIT rewrites interpreter bytecode at runtime, and
only for control. Type-quickening in an interpreter is best known from
CPython 3.11+ (`BINARY_OP_ADD_INT`, `LOAD_ATTR_INSTANCE_VALUE`) and from the
JSC/V8 baseline tiers. That is a reason to be modest about its payoff here: it
removes tag tests and taken branches, which on the M4 Pro are cheap unless
they sit on the loop's latency chain (the `csel` finding in the perf memory).

### 8.2 Where it would pay in tcvm

- **Arithmetic mixes.** `arith_handler` (`interp.rs:1356`) inlines float/float
  and small/small and sends everything else — including int+float, which
  `acc = 0.0; acc = acc + t[i]` produces on every iteration — to
  `op_*_slow` → `num::op_arith_slow` (40 samples in the constructor profile
  for `acc + v.z`). An `ADD_IF`/`ADD_FI` quickened form, or just an inline
  mixed arm in the generic handler, is the cheapest fix. Per-site
  `ADD_FF`/`ADD_II` would remove one tag test and the fall-through branch
  from the type the site doesn't use — measurable (memory: "the int/int arm
  first with `likely`" was worth 21–46 % on the immediate ops, but that was
  branch *placement*, which quickening also gives you).
- **`GETTABLE`/`SETTABLE` by key kind.** `op_gettable` (`interp.rs:960`) calls
  out-of-line `TableState::raw_get` (a 0x2c0-byte function with the dict-mode
  hasher inlined; disassembly in appendix B), with a frame. A `GETTABLE_I`
  form that inlines only `small-int → array bounds → load`, and a
  `GETTABLE_S` form with a shape IC keyed by the *runtime* string (JSC's
  baseline `GetByVal` string-identity IC; V8 `KeyedLoadIC` with name
  feedback), would bring `array_get_f` from 1.55× PUC to parity and give
  `t[k]` sites (`str_key_var`, 1.16×) the same IC as `GETFIELD`. This is the
  Lua analog of V8's elements-kind specialization.
- **Comparisons** `EQ`/`LT`/`LE`: same fixed-order structure; `EQ` int≠int
  "detours through the string and userdata tag checks" (#131). Quickened
  `EQ_II`/`LT_FF` or just a same-tag early exit.
- **`FORLOOP`** already branches on the step's type each iteration; a
  `FORLOOP_I`/`FORLOOP_F` split chosen by `FORPREP` is the LuaJIT
  `FORL`/`IFORL` pattern applied to types, and removes the small-int check
  triple (`interp.rs:2734`).

### 8.3 Superinstructions the compiler could emit

Cheaper than runtime quickening and independent of it:

- `SETFIELD` with a constant value (`LOAD K; SETFIELD` is every field of every
  constructor in `compile_nbody.snap`; PUC has RK operands, LuaJIT `TDUP`
  bakes them in). Listed in #33 as `SETTABUP K K`; the same applies to
  `SETFIELD`/`SETI`.
- `GETTABUP` + `GETFIELD` for `math.sqrt`/`string.format` (two IC hits →
  one cached value guarded by both shapes; JSC/V8 don't do this since
  `Math.sqrt` resolves through the same two ICs, so it's a Lua-specific
  micro-win, low priority).
- `GETI`/`SETI` with an 8-bit immediate index (#33; LuaJIT `TGETB`).

### 8.4 Mechanism

`Prototype.code` is `Box<[Instruction]>` in an immutable `Gc<Prototype>`. In-place
opcode rewriting needs it to be `Box<[Cell<Instruction>]>` (or an `UnsafeCell`
slice; instructions hold no GC pointers so no barrier is involved) with the
rewrite done as a single 8-byte store from the slow path. Rules from
LuaJIT/CPython experience: rewrite generic → specialized on the *first*
execution; on a specialized miss rewrite back to generic (or to a
"stable-generic" form that never re-specializes, to avoid ping-pong at truly
polymorphic sites); never rewrite from a metamethod continuation. The
`Handler` table grows by one entry per specialized form; handler placement
is a known noise source (memory: ±3–4 %), so measure with counters.

---

## 9. Upvalues

`op_getupval` is 13 instructions and one branch for a closed upvalue, 19 and
two branches for an open one (`interp.rs:789`, disassembly appendix B): load
the upvalue `Gc`, load the `UpvalueState` discriminant, branch open/closed,
compare the owning thread with the running thread, branch, then index into
whichever stack. PUC and LuaJIT
keep a pointer `v` in the upvalue that points at the stack slot while open and
at the upvalue's own `value` field once closed (`UpVal.v`, moved by
`luaF_close`/`lj_func_closeuv`; PUC fixes them up in `correctstack` when the
stack reallocates). One load, no branch. #130 covers the frame-chase part of
this (now fixed by `DispatchState`); the branchy representation is the
remaining cost. tcvm's `Vec` stack reallocates too, so it needs the same
fix-up in `ensure_slots` — the open-upvalue list already exists for
`op_close`.

---

## 10. Small things found on the way

- `op_len` (`interp.rs:1739`) does `t.get_metamethod(mm_len)` — a metatable
  `raw_get` — instead of `shape().has_mm(MetamethodBits::LEN)` like every other
  handler.
- `op_settable` (`interp.rs:1020`) does `raw_get(k)` then `raw_set(k)` when the
  `NEWINDEX` bit is set: two lookups where one slot lookup would do.
- `op_setfield` hit path indexes `state.properties[slot as usize]` with a bounds
  check (`interp.rs:1163`) where `property_at`'s unchecked variant is the
  intent; and `maybe_update_mt_bit(constant!(key_idx), v)` loads the constant on
  every store to check a usually-`None` `mt_cache`. Reorder: check `mt_cache`
  first.
- `raw_get` checks the string tag before the integer tag; for `GETTABLE` from a
  numeric `for` that's a wasted compare on every access (§8.2 fixes it
  properly).
- `Table::new` clones the metrics `Rc` three times and builds three empty
  containers (§4.4).
- `op_closure` (`interp.rs:2898`) allocates a `Vec`, converts to `Box<[_]>`,
  then `Gc::new`s — two mallocs plus the closure; the open-upvalue search is a
  linear scan of `open_upvalues` per captured local.

---

## 11. Suggested order

1. §3.5 entry format + §3.1 proto-load on `GETFIELD`/`GETTABUP`/`SELF` + §3.2
   absent mode. One design, three modes, pure interpreter change, no compiler
   work. Re-measure `field_proto*`, `field_absent`, `method_call`.
2. §4.1 transition IC on `SETFIELD`/`SETTABUP`, then §4.2 shape-carrying
   `NEWTABLE` (compiler + `Prototype` constant kind). Re-measure `ctor_*`,
   `setfield_new`, nbody's constructors.
3. §3.4 option 1 (raise the cap, at least for `_ENV`) + §5.1 pointer/cached
   hash. Trivial and removes the two worst cliffs after §7.
4. §7 — decide between free-list boxes (which becomes §4.3, the allocator) and
   widening the inline integer; this is a value-representation decision on the
   branch you just merged, so it needs your call rather than a survey's.
5. §6 `pairs` specialization once #172 lands (the call fast path changes the
   baseline).
6. §8 quickening: `GETTABLE_I`/`_S` first (biggest measured gap), then
   arithmetic mixes; `ADD_FF`/`ADD_II` only with counters showing the branch
   on the chain.
7. §4.4 inline storage and §9 upvalues as representation changes when the
   above have settled, since both touch every handler.

---

## Appendix A — benchmark sources

Full sources are in `interp-perf-survey.bench/` next to this file (untracked);
each prints a checksum that matches on all three interpreters. `_f` variants
replace `local acc = 0` with `0.0`. Run: `hyperfine -N -w 1 -r 5
'target/release/tcvm-cli --file X.lua' 'luajit -j off X.lua' 'lua5.5 X.lua'`.

```lua
-- field_own            local p = {x=1.5,y=2.5,z=3.5,w=4.5}; for i=1,3e7 do acc = acc + p.x + p.y end
-- setfield_own         local p = {x=0,y=0}; for i=1,3e7 do p.x = i; p.y = i end
-- field_mono_ctl       local objs = {{x=1,y=2},{x=1,y=2}}; ... local o = objs[(i%2)+1]; acc = acc + o.x
-- field_poly2          objs = {{x=1,y=2},{y=2,x=1}}   (same loop)
-- field_poly4          objs = {{x=1,y=2},{y=2,x=1},{x=1,z=2},{z=2,x=1}}, i%4
-- field_absent         local p = {x=1,y=2}; ... if p.z == nil then acc = acc + 1 end
-- field_proto          Class.__index = Class; Class.area = fn; obj = setmetatable({w=2,h=3,id=7}, Class)
--                      for i=1,3e7 do local f = obj.area; if f then acc = acc + 1 end end
-- field_proto2         Base <- Derived (Derived.__index = Derived, setmetatable(Derived, Base)); obj.area
-- field_proto_big      12 fields f0..f11, 30 methods m0..m29, lookup obj.m29
-- method_call          Class.get = function(s) return s.w end; for i=1,1e7 do acc = acc + obj:get() end
-- ctor                 for i=1,5e6 do local v = {x=i, y=i+1, z=i+2}; acc = acc + v.z end
-- ctor_arr             for i=1,5e6 do local v = {i, i+1, i+2}; acc = acc + v[3] end
-- setfield_new         for i=1,5e6 do local p = {}; p.x=i; p.y=i; p.z=i; acc = acc + p.z end
-- globals_small        g = 0; for i=1,3e7 do g = g + 1 end
-- globals_dict         v0..v99 = 0..99 (100 globals) then the same loop
-- dict_field           local t = {k0=0,...,k79=79}; for i=1,3e7 do acc = acc + t.k40 + t.k70 end
-- str_key_var          local t = {alpha=1,...,eps=5}; local k1,k2 = "gamma","eps"; acc = acc + t[k1] + t[k2]
-- array_get            t[1..1000] = i; for r=1,3e4 do for i=1,1000 do acc = acc + t[i] end end
-- ipairs               same table; for r=1,2e4 do for i,v in ipairs(t) do acc = acc + v end end
-- pairs50              local t = {k0=0,...,k49=49}; for r=1,2e5 do for k,v in pairs(t) do acc = acc + v end end
-- concat               local a,b,c,d = "alpha","beta","gamma","delta"; for i=1,3e6 do local s = a..b..c..d; n = n + #s end
-- smallint_add         local acc = 0; for i=1,3e7 do acc = acc + 1 end
-- bigint_add           local acc = 3000000000; (same loop)
```

## Appendix B — evidence

**Constructor profile** (`sample`, 2 s of `ctor_long` = `ctor_f` × 8, 1601
samples, top of stack):

```
 235  _xzm_xzone_thread_cache_fill_and_malloc   (libsystem_malloc)
 230  shape::transition_add_prop
 172  interp::setfield_slow
 140  TableState::set_string_key
 110  TableState::raw_get
 101  _xzm_malloc_tc
  56  _malloc_zone_malloc
  49  Table::raw_set
  48  TableState::maybe_update_mt_bit
  44  RawVec::finish_grow (properties Vec regrowth)
  43  op_newtable
  42  Context::allocate<RefLock<TableState>>
  40  op_add_slow            (acc float + v.z int  -> out-of-line mix)
  40  op_setfield
  34  op_forloop
  27  _xzm_xzone_free_to_freelist_chunk
  ... malloc/free total ≈ 507 (32 %)
```

**Concat profile** (2 s of `concat_long`, top of stack): `memmove` 152,
`realloc`-family 136+82+68+16, `RawVec::finish_grow` 116, interner
`HashTable::entry` 102, `coerce_to_str` 101, `free`-family 90+41+29+14+7,
`Interner::intern` 68, `memcmp` 62+9, `op_concat` 46, `foldhash::hash_bytes_long` 13.

**`op_gettable`** (release, `7c0bf4b`): frame push, table tag check,
`bl TableState::raw_get` (out of line, 0x2c0 bytes, own frame, dict hasher
inlined), nil + `has_mm(INDEX)` check, store, dispatch. `raw_get`'s integer
arm is reached after the string-tag compare and runs
`cmp x1, #-0x100000000; b.lo` (small int) → `cmp/b.le` (≥ 1) → bounds against
`array.len` → load.

**`op_getupval`** (release): `ldr` upvalue ptr → `ldr` discriminant → `cbz`
(closed?) → `ldr` thread → `cmp`/`b.eq` running thread → `ldr` stack base →
`ldr` value → store → dispatch: 13 instructions / 1 branch closed, 19 / 2 open.

**Object sizes** (`size_of`, this branch): `TableState` 168, `ShapeData` 136,
`Prototype` 152, `LuaClosure` 24, `UpvalueState` 16, `InlineCache` 16,
`Value` 8, `StringData` 16 (+ separate byte buffer). `GcBoxHeader` adds 16.

**Reference line index**

| what | where |
|---|---|
| tcvm IC read/fill/check | `src/vm/interp.rs:578, 598, 623, 639` |
| tcvm `GETFIELD` / `SETFIELD` / `SELF` / `GETTABLE` / `CONCAT` / `TFORCALL` / `LEN` / `CLOSURE` | `interp.rs:1077, 1138, 1204, 960, 1781, 2795, 1739, 2898` |
| tcvm `walk_index_chain`, `read_upvalue` | `interp.rs:3493, 3221` |
| tcvm `InlineCache`, `Prototype.ic_table` | `src/env/function.rs:80, 60` |
| tcvm shape cap, `find_slot`, `transition_add_prop` | `src/env/shape/mod.rs:75, 363, 398` |
| tcvm `raw_get`, `set_string_key`, `migrate_to_dict`, `next`, array `resize` | `src/env/table/mod.rs:224, 288, 335, 380, 269` |
| tcvm string hash by bytes | `src/env/string.rs:67`, `table/hash_part.rs:76` |
| tcvm boxed integer | `src/env/value.rs:133–146` |
| tcvm allocation | `src/dmm/context.rs:360` |
| LuaJIT `TGETS`, `TNEW/TDUP`, `ITERN`, `ISNEXT` | `vm_arm64.dasc:3213, 3128, 3600, 3653` |
| LuaJIT template tables | `lj_parse.c:1850–1905` |
| LuaJIT string hash / `GCstr` | `lj_str.c:76`, `lj_obj.h:306` |
| V8 IC states, binary-op feedback | `src/common/globals.h:1896, 2511` |
| V8 validity cell, global property cell, stub cache | `src/ic/accessor-assembler.cc:1283, 3836`; `src/ic/stub-cache.h:88` |
| V8 object literal boilerplate, `ForInNext` | `interpreter-generator.cc:2693, 3340`; `builtins-constructor-gen.cc:600` |
| JSC get_by_id modes | `bytecode/GetByIdMetadata.h:33`, `llint/LowLevelInterpreter64.asm:1653` |
| JSC put_by_id transition | `BytecodeList.rb:294`, `LowLevelInterpreter64.asm:1757`, `LLIntSlowPaths.cpp:1118` |
| JSC get_from_scope, new_object, get_by_val | `BytecodeList.rb:542, 597, 657`; asm `:2901` |
| JSC arithmetic profile / math IC | `LowLevelInterpreter64.asm:1230`, `bytecode/ArithProfile.h:37`, `jit/JITMathIC.h:70–150` |
| PUC short-string limit, `TString.hash`, `NEWTABLE` sizing | `lstring.h:29`, `lobject.h:410`, `lvm.c:1416` |

---

## 12. Instruction-set comparison on `particles.lua` (added after review)

Static listings: `tcvm-cli -l -f particles.lua` (438 instructions),
`luajit -bl` (436), `luac5.5 -l -l -p` (482, of which 35 are `MMBIN*` metamethod
shadows and 10 `EXTRAARG`, so ≈437 real). tcvm's density already matches
LuaJIT's; the instruction set is not where the big gaps are. What differs is a
handful of specific forms, listed by how often they fire in this file's hot
functions (`Vec2.new/add/sub/scale/len2`, `Particle.new`, the `update` closure,
`rand`, `World:step`'s inner loops).

Fusable pairs counted in tcvm's listing:

| pair in tcvm output | count | what the others emit |
|---|---|---|
| `GETUPVAL u; GETFIELD r r K` (upvalue table, constant key) | 11 (3 hot: `Vec2.new` in add/sub/scale) | PUC: one `GETTABUP` — tcvm only emits `GETTABUP` for `_ENV` (`compiler/rules.rs:2007,2167,2829`). LuaJIT: `UGET`+`TGETS`, same as tcvm |
| `LOAD r K; SETFIELD r t K` (constant value store) | 7 (`age = 0`, `alive = true`, `frame = 0`, …) | PUC: `SETFIELD t K vK` (RK value). LuaJIT: baked into the `TDUP` template, zero instructions |
| `LOAD r K; ADD/MUL/MOD/DIV` (constant that doesn't fit `Imm`) | 3, all in `rand` (`1103515245`, `2147483648` ×2) | PUC `MULK/MODK/DIVK`, LuaJIT `MULVN/MODVN/DIVVN`: `rand` is 8 instructions there vs 11 here |
| `LOAD r K; GETTABLE`/`SETTABLE` (small integer index) | 3 (cold: `ps[1]`, `arg[1]`) | PUC `GETI/SETI`, LuaJIT `TGETB/TSETB` (#33) |
| `NOT r; TEST r; JMP` | 2 (`if not b`, `if not p.alive`, both in loops) | PUC `jumponcond` (`lcode.c:1160`) deletes the `OP_NOT` and flips `TEST`'s k; LuaJIT `ISF` |
| `LOAD r K; EQ` for `x == nil` / `x == "str"` / `x == true` | 0 here, ubiquitous elsewhere (`type(x) == "table"`, `node ~= nil`) | LuaJIT `ISEQS/ISNES/ISEQP/ISNEP`; PUC `EQK` — one instruction, and for nil/bool/interned-string constants a single 64-bit compare with no `__eq` possible |
| `SELF; CALL`, `GETFIELD; ADDI; SETFIELD` | 5, 4 | nobody fuses these; the win is the IC on `SELF` (§3.1), not fusion |

Side by side, `rand` and `Particle.new`:

```
tcvm                              luajit                        puc 5.5
GETUPVAL R0 U0                    UGET  0 0 ; seed              GETUPVAL 0 0
LOAD     R1 K0 ; 1103515245       MULVN 0 0 0 ; 1103515245      MULK 0 0 0
MUL      R0 R0 R1                 ADDVN 0 0 1 ; 12345           ADDK 0 0 1
ADDI     R0 R0 #12345             MODVN 0 0 2 ; 2147483648      MODK 0 0 2
LOAD     R1 K1 ; 2147483648       USETV 0 0                     SETUPVAL 0 0
MOD      R0 R0 R1                 UGET  0 0                     GETUPVAL 0 0
SETUPVAL R0 U0                    DIVVN 0 0 2                   DIVK 0 0 2
GETUPVAL R0 U0                    RET1  0 2                     RETURN1 0
LOAD     R1 K1
DIV      R0 R0 R1
RETURN   R0 count=2               (8)                           (8, +4 MMBINK shadows)
(11)

NEWTABLE R3                       TDUP  4 1   ; {id=,pos=,vel=,age=0,alive=true}
GETUPVAL R4 U0                    UGET  5 0
SETFIELD R4 R3 "id"               TSETS 5 4 "id"
SETFIELD R0 R3 "pos"              TSETS 0 4 "pos"
SETFIELD R1 R3 "vel"              TSETS 1 4 "vel"
LOAD     R4 K5 ; 0                (age, alive came with the template)
SETFIELD R4 R3 "age"
LOAD     R4 K7 ; true
SETFIELD R4 R3 "alive"
(9)                               (5)
```

Note `rand` also boxes: `seed * 1103515245` is ≈2⁶¹, so on the 8-byte `Value`
every call allocates two `i64` boxes (§7) before `% 2147483648` brings it back
under 2³¹.

### 12.1 Recommendations, in order

1. **Emit `GETTABUP`/`SETTABUP` for any upvalue table with a constant key**, not
   just `_ENV` — compiler-only, no new opcode, the IC slot already exists.
   Removes one dispatch and a register round-trip from every `Vec2.new(...)`,
   `Particle.new(...)`, `insert(...)`-style call inside methods.
2. **`EQK`/`NEK` for nil/boolean/string constants** (`Abd { src, inverted, key: KIdx }`):
   handler is `skip_if!((reg.bits == k.bits) != inverted)` — no tag dispatch,
   no metamethod check, because nil/bool/interned strings are bit-unique in the
   NaN-box. Numbers stay on `EQI`/generic `EQ` (1 == 1.0 needs the slow
   compare). This is LuaJIT's `ISEQS/ISEQP` and the most common comparison
   shape in real Lua.
3. **K-form arithmetic** (`ADDK/SUBK/MULK/DIVK/MODK/IDIVK/POWK` + bitwise, `AbcK`
   with `d: KIdx`) for constants the 32-bit `Imm` can't hold (any float that
   isn't f32-exact — `0.1`, `0.995`, `PI` — and ints ≥ 2³⁰). Put the constant's
   kind in the flag byte at compile time so the handler tests only the
   register operand (LuaJIT's `*VN` forms assume a number constant).
4. **Constant-value stores** — either `SETFIELD/SETI/SETTABUP` with an RK value
   (PUC's `k` bit; the `e` slot is free once the value is a `KIdx`), or, for
   constructor sites, the shape-carrying `NEWTABLE` of §4.2 which makes the
   constant fields free like `TDUP`. Do §4.2 first; add the RK store for the
   `self.count = 0` cases outside constructors.
5. **`GETI`/`SETI`** (#33) and the **`NOT` → `TEST` peephole** (`codenot` on a
   `Reg` operand feeding a condition; mirror `jumponcond`).
6. **n-ary `CONCAT`** (#33, §5.2).

Not worth an opcode: `SELF+CALL`, read-modify-write field ops, `RETURN0/1`
(same dispatch count; #171 territory), `LOADI/LOADF/LOADTRUE` (the pool is one
load off `ds` now, so they only save the constant fetch).

Expected effect on this file's hot functions: `Vec2.add/sub` 9→8, `scale` 7→6,
`Particle.new` 19→14, `rand` 11→8 dispatches — roughly 10 % fewer instructions
in the parts that matter, which is the ceiling for ISA work here; the per-
instruction costs in §3–§7 are worth several times that.
