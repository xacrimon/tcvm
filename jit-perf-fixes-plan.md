# JIT performance fixes: implementation plan

This is the implementation plan for `jit-perf-fixes.md` (the report). The
report says what is wrong and what the code should look like afterwards; this
document says which functions change, in what order, and how each step is
verified. Every file and line reference below was read against the tree at
5e587c6 (branch `jit2`). Paths are relative to `src/` unless they start with
`tests/` or `tools/`.

Scope decisions already taken with the user:

- `debug.setlocal` and `debug.sethook` are not implemented (`builtin/debug.rs`
  lists them as stubs) and nothing is done to accommodate them.
- String ordering compiles to a byte loop, not LuaJIT's word loop, so the
  string allocator does not change.
- Changes on the interpreter and runtime side (frame header, `ff.rs`, `coro.rs`,
  `enter`, helpers, feedback recording) are in scope.

Contents

1. Corrections to the report, from reading the source
2. The code as it is: the pieces the plan touches
3. Milestone A: speculate or exit (P1, P6)
4. Milestone B: types on values (P2)
5. Milestone C: values across calls, builtins through cores (P3 steps 1–2, P4)
6. Milestone D: known callees, results in registers, lazy headers (P3 steps 3–4)
7. Milestone E: the missing optimizations (P5)
8. Milestone F: compile time and the harness (P7, P9)
9. Testing and tooling
10. Order, dependencies and the numbers each milestone is measured against
11. Open points

---

## 1. Corrections to the report, from reading the source

The plan is grounded in the code, and reading it in full corrected a few of the
report's premises. They change details of the fixes, not the fixes themselves.

1. **Float ops never accept `I32` or `I64` operands; the builder converts.**
   `ir/verify.rs:285-286` admits only `F64` for every `F*` op, and
   `build/mod.rs:757-779` (`as_f64`) emits `IToF` / `LToF`. The intrinsic path
   already converts an `I64` argument with `LToF` (`build/mod.rs:762`). The
   partialsums regression therefore comes from site decisions, not from the
   IR: `arith_mode` (`build/emit.rs:33-100`) returns `Generic` for a mixed
   `I64`/`F64` site, and `emit_math` (`build/intrinsics.rs:339-343`) refuses an
   intrinsic whose feedback byte has any kind outside `SMALL | FLOAT`, which a
   `BIGINT` bit is. P2's "`I64` is a representation" item is two small changes.
2. **After a call the slot's type set survives; only the representation is
   lost.** `root_load` (`build/mod.rs:303-321`) types a resume block's `Load`
   from `root_ty`, which `store_live` (`:566-593`) filled with the type of the
   value stored before the call, and `guard()` (`:716-723`) short-circuits on
   a known set. The guards the report counted after calls are on result
   registers and captured registers, which are forced to `ANY`
   (`build/emit.rs:1590-1592`, `build/mod.rs:587-591`), and on speculation
   guards placed at resume loads. P3 step 1 as written ("trust the slots")
   has no work of its own; it is subsumed by step 2.
3. **`PEEL_LIMIT = 48` is in `opt/peel.rs:10`; `OSR_PEEL_LIMIT = 150` is in
   `build/mod.rs:29`** and counts bytecodes of outer-loop copies, not IR.
4. **Float selection on a block parameter must preserve Lua 5.5 semantics.**
   `math.type`, `//`, string conversion and integer overflow distinguish `0`
   from `0.0`, so a parameter that joins an integer `0` and a float cannot be
   made `F64` and deopt with a float where the interpreter had an integer. The
   report's zero-trip argument covers the loop never running, not an exit in
   the first iteration. Peeling is what makes spectral's accumulator `F64`
   (the loop proper then joins only floats), and the parameter rule stays
   exact (B2).
5. **Helpers are never pure, whatever they declare.** `Op::is_pure`
   (`ir/ops.rs:744-746`) is `effects() == NONE && !Helper`, GVN's `key_of`
   (`opt/gvn.rs:118-138`) and LICM (`opt/licm.rs:119`) skip every `Helper`,
   and the `R_*` effect flags are declared but never consumed
   (`ir/ops.rs:647-671`). A6 replaces the three paths with one summary
   struct, the shape the pre-rewrite JIT already had.
6. **`TabMmBits` is not in GVN's memory model** (`opt/gvn.rs:32-45`), so it is
   neither shared nor hoisted today; `TabASize` and `TabArray` are classed
   `SHAPE`, so an `ArrStore` does not invalidate them.
7. **Nothing in the interpreter records `MM` for `__index` or `__newindex`**,
   nor anything for `UNM`, `BNOT`, `LEN`, `CONCAT`; `get_slow`
   (`vm/ops/field.rs:543`) records only the receiver's kind, and the GETTABLE
   array fast path records nothing. `MM` on a table site only ever comes from a
   JIT exit (`jit/mod.rs:449-522`).
8. **`MAX_ENTRIES = 4`** (`jit/state.rs:12`) is regions per prototype, not a
   snapshot limit; snapshots take up to 4095 entries (`region.rs:58-88`).
9. **`tbz`/`tbnz` reach ±32 KiB** (`backend/aarch64/asm.rs:131-140`) and a
   region can be 64 KiB, so the lowering uses `cbz`/`cbnz` (±1 MiB) for
   boolean branches, and `type_flags` (`backend/aarch64/emit.rs:421-459`)
   panics on a multi-part set, which a `Set` of `IsType(NUM)` would hit.
10. **`exit_common` has its own inline copy of `leave`**
    (`backend/aarch64/abi.rs:149-154`), and `fast_call` leaves two forms of
    return address (`b fast_resume` after the `blr`, or `fast_resume | 1` on
    deep chains, `emit.rs:311-341`). Both matter for C1's frame walk.
11. **The regalloc2 `Function` has no cold-block or spill-weight hints**
    (`backend/vcode.rs:127-187`); block order (`backend/order.rs:14-78`) is the
    only lever besides clobbers.
12. **`remove_trivial_params` creates `LToI` with no snapshot**
    (`build/mod.rs:1218`) while `LToI` is `MAY_DEOPT`; a latent verifier
    failure that B2 fixes in passing.

---

## 2. The code as it is: the pieces the plan touches

A compact map so the later sections can name things without re-explaining.

**Pipeline** (`jit/compile.rs:72-140`, `build_ir`): `Cfg::build` →
`Builder::build` → `infer` → `speculate(EntryKinds)` → `peel` →
`optimize` (up to 4 rounds of `infer`, `narrow`, `simplify`, `gvn`, `licm`,
`dce`; then `legalize`) → `prune_gc_checks` → `dce` → `call_boundaries` →
`prune_snapshots` → `completes`. Then `split_critical_edges`, `lower`,
regalloc2 (Ion), `emit`, install (`compile.rs:224-416`).

**Builder** (`jit/build/`): `Builder` (`mod.rs:72-103`) keeps per block a
`defs` row (SSA value per register) and a `slots` row (`SlotState::Unknown |
Holds(Val) | Lazy(Block)`, `:45-52`). Reads go through `reg(r)` → `read_var_at`
(`:255-301`), writes through `set` and the deferred `pending` list applied by
`flush` (`:347-362`). Block parameters are created as `Ty::ANY` (`:270, :284`);
kinds reach them only in `opt::infer`. Snapshots: `snapshot(pc, kind, live)`
(`:505-520`) lists live registers whose slot does not hold their value;
`snap_before` (`:523`), `snap_after` (`:535`), `note_def` (`:552-561`) for
definition snapshots that `speculate` uses. Conversions: `boxed` (`:698`),
`guard(v, set)` (`:716`), `as_i32` (`:726`), `as_f64` (`:757`), `num_f64`
(`:782`), `as_i64` (`:792`). Calls: `Builder::call` (`:637-647`) makes the
resume block a root (`new_root`, `:166-174`) with no SSA predecessors.
Emitters: `emit.rs` (`arith_mode` `:33-100`, `emit_arith` `:567-667`,
`emit_generic` `:829-877`, `emit_unm` `:739`, `emit_bnot` `:790`, `helper_arms`
`:1054-1078`, `calls_mm` `:1083`, `emit_len` `:1090`, `emit_cmp_reg`
`:1259-1325`, `generic_cmp` `:1337`, `generic_eq` `:1380`, `helper_branch`
`:1411`, `emit_cmp_imm` `:1451`, `emit_eqi` `:1501`, `emit_call` `:1543`,
`call` `:1577-1607`, `emit_forprep` `:1611`), `tables.rs` (`index` `:78-98`,
`newindex` `:102-121`, `emit_get` `:163-228`, `emit_set` `:452-517`,
`table_site` `:613`, `int_key` `:621`, `emit_gettable` `:634-685`,
`emit_settable` `:689-741`), `intrinsics.rs` (`intrinsic_of` `:39-95`,
`value_now` `:100-118`, `builtin` `:157-224`, `emit_math` `:326-461`),
`iter.rs`, `cfg.rs` (`never_ran` `:606-629`).

**IR** (`jit/ir/`): `Func` (`mod.rs:206-230`), `InstData` (`:110-124`),
`BlockData { params, insts, resume, dead, peeled }` (`:135-148`), `SnapData {
pc, kind, entries }` (`:161-166`). `Ty { rep, set, refine }` (`types.rs:72`),
`Rep` (`:8-20`), `TypeSet` (`:38-58`). `Op` (`ops.rs:388-643`), `Effects`
(`:647-671`), `HelperId` + `Sig` (`:66-191`, table `:228-347`), `ExitTag`
(`:353-382`), `result_ty` (`:784-837`). Verifier rep rules
(`verify.rs:269-313`), R6 check (`:163`).

**Passes** (`jit/opt/`): `infer` + `narrow` + `convert` + `legalize`
(`infer.rs`), `speculate` (`speculate.rs:31-185`, `expectations` `:190`,
`anticipated` `:278`, `barrier` `:396`), `peel` (`peel.rs`), `licm`
(`licm.rs:16-180`, spine rule `:72-82`, `memory_kept` `:90`, `entry_snap`
`:85`), `gvn` (`gvn.rs`, `reads` `:32`, `writes` `:49`, `key_of` `:118`,
`forwarded` `:141`), `simplify`, `calls::call_boundaries` (`calls.rs:12-91`,
`rematerializable` `:96`, `remat` `:106`), `snaps::prune_snapshots`,
`prune_gc_checks` (`mod.rs:81-111`), `dce` (`:115`), `split_critical_edges`
(`:162`).

**Backend** (`jit/backend/aarch64/`): registers (`abi.rs:9-28`: x20 INSN, x21
PC, x22 BASE, x23 RT, x24 CLOSURE, x25 THREAD; x16/x17 scratch; `caller_saved`
`:59-69`); `dispatch_entry` (`:180-209`) and `exit_common` (`:104-164`);
lowering (`lower.rs`: `alias` `:197`, branch fusion `:217-290`, `exit_of`
`:312-360`, `Box` `:450-505`, `Helper` `:954-982`, `Resume` `:983-1009`,
`Br` `:1013`, `Call` `:1023`, `CallMm` `:1037`); emission (`emit.rs`: frame
`:94-108, 295-304`, fast entry `:128-147`, stubs `:165-203`, `exit_out`
`:222-262`, `fast_call` `:311-341`, `leave` `:343-351`, `Helper` `:1072-1085`,
`Call` `:1107-1127`, `Dispatch` `:1128`, `TailEnter` `:1135`, `Resume`
`:1176-1217`, `CallMm` `:1218-1242`, `Return` `:1250-1272`, `GcCheck`
`:1279`); `vcode.rs` (`push` `:71`, `Function` impl `:127-187`); `order.rs`.
`region.rs`: `encode_snap` (`:58-88`), `SnapView::decode` (`:99-152`),
`Region` (`:162-211`). `jit/mod.rs`: `jit_exit` (`:84-152`), `install`
(`:339-422`), `on_exit` (`:449-522`), `record_site` (`:563-617`),
`request_recompile` (`:663`). `jit/layout.rs`: runtime offsets.

**Runtime**: frame header (`vm/frame.rs:3-10`: word 0 func, 1 continuation +
5 flag bits, 2 caller base, 3 caller pc; `write_hdr` `:223`); `enter`
(`vm/ops/call.rs:297-325`), `native_enter` (`vm/native.rs:469-496`),
`fast_entry!` (`vm/ff.rs:18-70`), `resume_switch!` (`vm/coro.rs:161-233`),
`ff_resume` (`:250`), `ff_yield` (`:268-333`), `ff_wrap` (`:239`); feedback
(`jit/feedback.rs:9-23`; writers `vm/ops/arith.rs:551-606, 642`,
`compare.rs:294-363`, `field.rs:543, 579`, `control.rs:362, 486`); values
(`env/value.rs:27-53`: NaN-boxed, small ints `0xFFFF_FFFF << 32 | i`, floats
raw, heap `0xFFF8 | tag`); strings (`env/string.rs:18-37`: `hash`, `len`,
bytes at payload + 16, all interned); tables (`env/table/mod.rs:491-516`,
offsets in `jit/layout.rs:49-60`); closures (`env/function.rs:330-377`,
`UpValueDescriptor::by_value` set by the compiler when the variable is never
assigned, `compiler/defs.rs:346-352`); helpers (`jit/helpers.rs`).

**Tests**: `tests/suite/jit_regions.rs` (`agrees(src)` runs a chunk with the
JIT on and off and compares, `:24-26`), `src/jit/tests.rs` (IR building),
`src/jit/layout.rs:91-300` (offsets), `src/vm/tests/`. Env: `TCVM_JIT_CHECK`
(verifier after each pass, regalloc2 checker), `TCVM_JIT_DEOPT_ALL` (every
exit fires once), `TCVM_JIT_DUMP=ir,asm`, `TCVM_JIT_LOG`, `TCVM_JIT_HOT_*`.

---

## 3. Milestone A: speculate or exit (P1, P6)

Goal: no arithmetic, compare, equality or array site has a helper call or a
type dispatch on its hot path; the only calls left at such sites are in cold
arms that feedback asked for, and those arms do not disturb the hot path's
register allocation. Kernels: queen 65 → about 22 instructions per iteration,
linear-sieve 315 → about 150, every array-reading loop in the suite.

### A1. Feedback: what the interpreter and the exits record

The kind tables of P1 are chosen from each operand's kind, which the builder
knows from the value's `Ty` or, for a value of unknown type, from the site's
feedback byte. Two gaps in the byte make that impossible today (correction 7).

*Files:* `jit/feedback.rs`, `vm/ops/table.rs:31-48, 72-92` (GETTABLE /
SETTABLE handlers), `vm/ops/field.rs:395-446, 530-592`, `jit/mod.rs:563-617`
(`record_site`).

1. **Array sites get a miss bit.** The byte's eight bits are all assigned, and
   the meaning of a bit is already per opcode (`FLOAT_LIMIT = OTHER` at
   FORPREP, `feedback.rs:20`). At GETTABLE and SETTABLE sites define
   `fb::ARR_MISS = BIGINT` ("the small-int key was not in the array part, or
   the array part held nil and the class has `__index`/`__newindex`") and keep
   `SMALL`/`FLOAT`/`STR`/`TAB`/`OTHER` as the key's kind (the receiver's kind
   is in the IC already, and a non-table receiver sets `OTHER`). Writers:
   `gettable_general` and `settable_general` (the slow paths behind
   `vm/ops/table.rs:31-48, 72-92`) OR the bits in once per change, as
   `record` does (`feedback.rs:46-57`, store-on-change). The fast path stays
   untouched, so the interpreter's cost is one byte load on a path that
   already does a hash lookup.
2. **Metamethod sites record `MM` in the interpreter.** `index_chain` /
   `newindex_chain` (`field.rs:395-446`) record `MM` when they call a
   function; `arith_slow` already does (`arith.rs:642`). UNM, BNOT, LEN and
   CONCAT slow paths record `MM` the same way; their byte was never written.
3. **Exits record what the site needs.** `record_site` gains arms for
   GETTABLE / SETTABLE (key kind from the rebuilt frame, plus `ARR_MISS` for
   the new `ExitTag::Bounds`), UNM / BNOT / LEN (operand kind), CONCAT (both
   operand kinds). `ExitTag` gains `Bounds` (widenable, like `Type`).
4. **One compare byte covers two operands.** `record_site` for `CmpReg` ORs
   both operands' kinds into one byte, so `TAB | OTHER` cannot say which side
   was the table. A3 needs only "numbers only", "strings only", "one kind that
   compares by bits", or "mixed"; the joint byte answers all of those, so the
   byte's layout does not change.

*Verification:* a unit test in `jit/feedback.rs` for the per-opcode bit names;
an interpreter-only run (`TCVM_JIT=off`) of the LJR suite before and after,
cycles within noise (the bytes are written on slow paths only).

### A2. Arithmetic: the op from the operand kinds

*Files:* `build/emit.rs` (`arith_mode` `:33-100`, `Mode` `:19-31`,
`emit_arith` `:567-667`, `emit_generic` `:829-877`, `int_op` `:950`,
`float_op` `:977`, `small_or_float` `:992`, `emit_unm` `:739-788`,
`emit_bnot` `:790-821`), `build/mod.rs` (`guard` `:716`, `as_i32` `:726`,
`as_f64` `:757`, `num_f64` `:782`, `as_i64` `:792`), `ir/ops.rs`
(`HelperId::Arith`, `Unm`, `BNot`), `jit/helpers.rs` (`jit_arith`).

**Replace `Mode` with a per-operand kind.** `arith_mode` decides one mode for
the site; the new code decides a kind for each operand and derives the op:

```rust
/// What the hot path assumes about one operand.
enum NumKind { I32, I64, F64, /// SMALL or FLOAT, decided by a tag test at the use.
               Num }

fn operand_kind(b: &mut Builder, pc: u32, v: &Operand, byte: u8) -> Option<NumKind> {
    // 1. the value's own type (num_kind, emit.rs:1206) wins;
    // 2. else the site's byte: INT only → I32 unless BIGINT|OVERFLOW → I64;
    //    FLOAT only → F64; SMALL|FLOAT (no BIGINT) → Num;
    //    any of STR|TAB|OTHER → None (string coercion or an error).
}
```

`emit_arith` then:

1. Establishes each operand's kind with one guard: `guard(v, SMALL)` for
   `I32`, `guard(v, INT)` for `I64`, `guard(v, FLOAT)` for `F64`,
   `guard(v, NUM)` for `Num` (`build/mod.rs:716` already folds a guard the
   type satisfies). The guard's tag is `Type`, so a miss recompiles the site
   with the kind it saw (`on_exit`, `jit/mod.rs:449-522`).
2. Converts and emits per the P1 arithmetic table:

| left, right | result | code |
|---|---|---|
| `I32`, `I32`; `+ - *` | `I32` | `IAdd`/`ISub`/`IMul` with `ExitTag::Overflow` (`emit.rs:589-604` today) |
| `I64` with any int kind; `+ - *` | `I64` | `IToL` the `I32` side, `LAdd`/`LSub`/`LMul` (wrapping, pure) |
| any int kind with `F64` or `Num` | `F64` | `IToF`/`LToF` the int, `ToF64` the `Num` (a tag test and a `scvtf`/`fmov` select, `ir/ops.rs:500`), float op |
| `Num`, `Num` | `F64` | `ToF64` both, float op (a `SMALL\|FLOAT` byte, B3) |
| `/`, `^` | `F64` | `num_f64` both, `FDiv`; `^` with a constant exponent folds as LuaJIT does (`lj_opt_fold.c:1103-1114`): `x ^ 0` is `1.0`, `x ^ 1` is `x`, `x ^ 2` is `FMul`; otherwise `Helper(FPow)` (pure, a libm `pow` call today, `helpers.rs:27`); never `sqrt` for `x ^ 0.5` |
| `//`, `%` on `I32`/`I64` | int | `IDivFloor`/`IModFloor` (exit on zero, `Slow`), `LDivFloor`/`LModFloor` |
| `//`, `%` with a float | `F64` | `FIDiv`; `%` through `Helper(FMod)` (pure, a libm `fmod` call today) until E7's inline form |
| bitwise, int kinds | int | `IAnd`.. / `LAnd`.. ; shifts `IShl`/`IShr`/`LShl`/`LShr` |
| bitwise with `F64` | int | `FToIExact` (`Type`) then the int op; the exit is the "no integer representation" error |
| `None` (STR/TAB/OTHER, no `MM`) | – | guard both `NUM` and the numeric path; a byte that keeps seeing strings after the recompile gets the cold arm below |
| `MM` in the byte | – | `IsType(NUM)` both; the false arm is a cold metamethod arm (`metamethod` + `mm_call`, `emit.rs:1044, 1004`) |

3. **`emit_generic`, `small_or_float`, `both` and the `Helper(Arith)` arms are
   deleted.** `int_op` and `float_op` stay as the typed emitters.
   `HelperId::Arith` is kept under a new name, `ArithCoerce`, for one use: the
   cold arm of a site whose byte has `STR` (string to number coercion,
   `"10" + 1`); `jit_arith` (`helpers.rs`) keeps its numeric and string arms
   and returns `HELPER_FAIL_VAL` for the rest. `Unm` and `BNot` helpers go the
   same way (`emit_unm` and `emit_bnot` get the kind table: `FNeg`, `INeg`
   with the `Overflow` exit or `LNeg`, `INot`/`LNot`; a cold `ArithCoerce`
   arm for `STR`; the metamethod arm for `MM`).
4. **Immediates** (`Operand::Int`/`Float`, `emit.rs:670-737`) have a kind by
   construction; the `_IF`/`_FI` reversed forms keep their operand order.
5. **The quickened forms** (`info.forms`, `instruction.rs:1193-1197`) remain
   the first source of kinds (`arith_mode`'s form arm, `:41-60`, becomes the
   form-to-kind mapping in `operand_kind`), and `kind_exited(pc)`
   (`build/mod.rs:1135`) still means "do not trust the form".

*Verification:* `tests/suite/jit_regions.rs` gets a generated differential
test, `arith_kinds`, that runs every operator over every pair from
`{1, 2^40, 1.5, 2^53 + 1 (int), "10", "x", {}, nil, true}` in a loop hot
enough to compile, in both JIT states (`agrees`), and the same chunk against
`lua5.5` through the script of section 9. `TCVM_JIT_DUMP=asm` on
`luabench/queen.lua`: no `bl` to `jit_arith`/`jit_lt` in the loop region.
`TCVM_JIT_DEOPT_ALL=1` run of the suite.

### A3. Compares and equality

*Files:* `build/emit.rs` (`emit_cmp_reg` `:1259-1325`, `generic_cmp` `:1337`,
`generic_eq` `:1380`, `helper_branch` `:1411`, `eq_by_bits` `:1444`,
`emit_cmp_imm` `:1451`, `emit_eqi` `:1501`, `byte_kind` `:1221`), `ir/ops.rs`,
`ir/verify.rs:269-313`, `backend/aarch64/lower.rs` (`test_of` `:247-290`,
branch fusion `:217-244`), `backend/aarch64/emit.rs`, `jit/layout.rs`,
`jit/helpers.rs` (`jit_lt`, `jit_le`, `jit_eq` deleted), `ir/ops.rs:240-241`
(`HelperId::Lt/Le/Eq` deleted).

**New IR ops** (`ir/ops.rs`, with `result_ty`, `effects`, `verify` rules and
`print` names):

- `LToFExact`: `I64 → F64`, `MAY_DEOPT`, tag `Type`. Lowered as
  `cls x16, xs; cmp x16, #10; b.lo exit; scvtf d, xs`. A `cls` of 10 or more
  means bits 63..53 all equal the sign bit, so |i| ≤ 2^53 and the cast is
  exact (2^53 itself exits, which is conservative and harmless).
- `StrLt`, `StrLe`: `(Ptr, Ptr) → B1`, pure. Operands are string payload
  pointers (`Unbox(Ptr)` of a `STR`-typed value, as `StrLenOf` takes,
  `ir/ops.rs:559`). Lowered inline as the byte loop of the report, with six
  `reg_temp` operands (the pattern `NamedNext` uses, `lower.rs:692-697`);
  `jit/layout.rs` gains `STR_LEN = LuaString::LEN_OFFSET` and `STR_DATA = 16`
  (`GcBoxInner::<StringData>::TRAILING_FROM_VALUE`, `env/string.rs:26-37`)
  with an offset test next to the others (`layout.rs:91-300`). The loop
  fuses with a following `Br` through `test_of` like `ICmp` does.

**`emit_cmp_reg`** becomes: establish both kinds (A2's `operand_kind`, with
the compare byte's joint kinds), then:

| x, y | code |
|---|---|
| `I32`, `I32` / `I64`, `I64` | `ICmp(cc)` / `LCmp(cc)` (`cmp_vals`, `:1242`) |
| `I32`, `I64` | `IToL`, `LCmp` |
| `F64`, `F64` | `FCmp(cc)`; the backend's condition codes `mi/ls/gt/ge/eq/ne` are already NaN-correct (`FCmp` is "ordered; false on NaN", `ir/ops.rs:495`) |
| `I32`, `F64` | `IToF`, `FCmp` |
| `I64`, `F64` | `LToFExact`, `FCmp`; equality the same (an int that fits equals the float iff the doubles are equal) |
| `Num` with anything numeric | `ToF64` the `Num` side (a `SMALL\|FLOAT` byte), `FCmp`; for `I64` on the other side `LToFExact` |
| `STR`, `STR`: `==`/`~=` | `SameBits` (all strings interned, `env/string.rs:70-78`) |
| `STR`, `STR`: `<`/`<=` | `guard(STR)` both, `Unbox(Ptr)`, `StrLt`/`StrLe`; `>`/`>=` swap |
| `==`/`~=`, both types known, sets disjoint, not int/float | `KB1(false)` / `KB1(true)` (`__eq` only fires for two tables or two userdata) |
| `==`/`~=`, one side within `NIL\|BOOL\|STR\|FUN\|THR` | `SameBits` (`eq_by_bits`, today) |
| `==`/`~=`, unknown kinds, no `MM` | `SameBits` → true; else `IsType(NUM)` both → numeric compare as above; else `IsType(TAB)` both → `GuardFalse(TabMmBits & EQ)` (`Slow`) → false; else false. No helper, about 8 instructions on the longest path |
| `<`/`<=` with a non-number, non-string kind, no `MM` | the numeric path; the `Type` guard's exit is where the interpreter raises |
| `MM` in the byte | `IsType` dispatch to the number/string path, the false arm a cold metamethod arm (`helper_branch`'s arm, `:1431-1439`, kept; its helper call removed) |

`generic_cmp`, `generic_eq`, `helper_branch`'s helper call, `jit_lt`, `jit_le`,
`jit_eq`, `HelperId::{Lt, Le, Eq}` are deleted. `emit_cmp_imm` and `emit_eqi`
use the same table with the immediate's kind fixed. JEQS/JNEQS stay
`SameBits`. `byte_kind` and `byte_within` fold into `operand_kind`.

*Verification:* `cmp_kinds` differential test generated over the same value
set as A2 with all six operators; the `cmp_sem.lua` checks from the report
(`1 < "a"` raises in both JIT states, `9007199254740993 > 2^53` is true,
`"a\0b" < "a\0c"`, NaN comparisons) as `agrees` cases plus a `lua5.5` run.
Hot-path `bl` count zero in the queen and linear-sieve loop regions.

### A4. Array reads and writes

*Files:* `build/tables.rs` (`emit_gettable` `:634-685`, `emit_settable`
`:689-741`, `class_bit` `:60-66`, `index` `:78-98`, `newindex` `:102-121`),
`build/iter.rs:80-84, 142-146`, `opt/gvn.rs:32-45` (`reads`), `ir/ops.rs`
(`TabMmBits` effects), `jit/mod.rs:563-617`.

1. **Bounds miss is an exit unless feedback saw it.** The `ICmp(Lo) i n`
   branch (`tables.rs:660-662`) becomes `GuardTrue(inside)` tagged
   `ExitTag::Bounds` when the site's byte lacks `ARR_MISS`; with `ARR_MISS`
   the out-of-bounds arm stays, as a cold arm (A5) calling `Helper(Index)`.
2. **The nil result is the result when the class has no `__index`.** Today
   the nil arm tests `class_bit(t, INDEX)` live inside the arm (`:665-678`).
   New: `has = TabMmBits(t) & INDEX` is evaluated before the load and
   `GuardFalse(has)` tagged `Shape` when the byte lacks `MM`; then `ArrLoad`'s
   value is the result, nil or not, with no test. With `MM`, the nil arm
   stays and is cold. `TabMmBits` reads `META` (A6), so GVN shares it per
   receiver and LICM hoists it out of loops whose body has no helper table
   write; inline `TabStore`s cannot change class bits (section 11, point 1). The same guard replaces `class_bit(NEWINDEX)` in
   `emit_settable` (`:714-740`) and the iterator fast paths in `iter.rs`.
3. **`TabASize`/`TabArray` read `ARR`** as size and pointer (A6); an
   `ArrStore` cannot resize, so it leaves them valid, and any helper that
   writes `ARR` invalidates them, which is right.
4. **Array writes** mirror reads: `GuardTrue(inside)` with `Bounds`, the nil
   old-value test only when the class has `__newindex` (guarded away
   otherwise), `barrier_for` + `ArrStore`. A write to `asize` exactly (append)
   is A5 of the design (17.3 A5) and is left to milestone E.

*Verification:* `agrees` cases for sparse arrays (`t[1000000] = 1` then reads
of `t[i]` in a loop), arrays with `__index` metatables, key 0 and negative
keys, float keys `t[1.0]`; `TCVM_JIT_DEOPT_ALL`; queen's loop region shows one
`TabMmBits` load hoisted and no `bl jit_index`.

### A5. Cold arms and the helper trampolines

*Files:* `ir/mod.rs:135-148` (`BlockData`), `build/emit.rs:1166-1193` (`arm`,
`branch_arm`, `arm_end`), `backend/aarch64/lower.rs:954-982` (`Helper`),
`backend/aarch64/inst.rs`, `backend/aarch64/emit.rs:1072-1085`,
`backend/aarch64/abi.rs:59-69`, `backend/order.rs:14-78`, `jit/helpers.rs`
(address table `:24-67`), `jit/state.rs` (a new `trampolines` code block
owned by `JitRuntime`), a new pass `opt/sink.rs`.

1. **Cold blocks.** `BlockData` gains `cold: bool`. The builder sets it on
   every arm it creates for a helper fallback or a metamethod call
   (`helper_arms`'s metamethod arm, `index`/`newindex`'s arm, A2's
   `ArithCoerce` arm, A4's out-of-bounds and nil arms). `peel` and
   `repair_ssa` copy the flag; `order::layout` places cold blocks after the
   `Deopt` blocks (`order.rs:74-77`), which keeps the hot loop contiguous.
2. **Trampolines.** `JitRuntime` builds, once at JIT creation, one trampoline
   per `HelperId` in its own code block (the `CodeAllocator` of
   `backend/alloc.rs`): `sub sp, sp, #N; stp` of every register in
   `caller_saved()` that is not in the helper's declared clobber set, `bl
   helpers::address(h)`, `ldp` them back, `add sp`, `ret`. For a helper with
   two `Val` args and `Rt` context the saved set is x4–x15 (12), d0–d7 and
   d16–d31 (24): 18 `stp`/`ldp` pairs each way, about 40 instructions,
   executed only on the arm. x29/x30 are saved by the usual `stp x29, x30`
   so the frame chain stays walkable (C1 relies on it).
3. **Lowering.** `Op::Helper(h)` in a cold block lowers to `MInst::HelperCold(h)`
   whose clobber set is exactly the registers the emission writes: the ctx
   and immediate registers x0..x(k-1) (`emit.rs:1074-1080` moves them before
   the `bl`), the fixed argument registers, x0/d0 for the result, and x16,
   x17. Everything else is preserved by the trampoline, so regalloc2 keeps the
   hot path's values where they are. Emission is `bl trampoline[h]`.
4. **Sink what only the arm uses.** `opt/sink.rs` (run after GVN in
   `optimize`) moves a pure, single-result instruction whose uses are all in
   one cold block, and whose operands dominate that block, to the start of
   that block (the boxed key of a GETTABLE's miss arm, the `Box` of an
   operand for a metamethod call). Snapshots count as uses, so a value an
   exit needs stays. Iterated to a fixpoint per block; linear.
5. **`Box` of `I64`** keeps its `jit_box_i64` call (`lower.rs:466-476`); it
   is rare on hot paths once A2 keeps small ints `I32`. It is given the same
   trampoline treatment so its clobber set shrinks to {x0, x1, x16, x17}.
6. **Collector checks follow the allocation into the cold block.** A
   `MAY_ALLOC` instruction in a cold block (`NewIndex`, `Index`'s metamethod
   arm, `Concat`, `ArithCoerce`, `Box` of `I64`) gets a `GcCheck` placed right
   after it in that block, with the arm's `After` snapshot (`snap_after`,
   `build/mod.rs:535`), and `prune_gc_checks` (`opt/mod.rs:81-111`) counts
   only allocations in non-cold blocks when deciding whether a loop header or
   the entry keeps its check. The hot path of a loop that stores into an
   array pays no check; the arm pays one only when it runs; garbage still
   waits at most one iteration or one arm, which is the collector rule.

*Verification:* a lowering test that a region with a cold `Helper(Index)`
arm has no spill moves on its hot path (`TCVM_JIT_DUMP=asm`, count `str`/`ldr`
to `[x29, ...]` in the loop); ray's `vsub` region spill count (18 stores, 38
loads today) and qt's `clip` (41 and 55) re-measured; `TCVM_JIT_CHECK=1`
(the regalloc2 checker validates the reduced clobber set against the
operands).

### A6. One effect summary for ops and helpers

*Files:* `ir/ops.rs:647-671` (`Effects`), `:691-746` (`effects`, `is_pure`),
`:185-191, 228-347` (`Sig`, `sig`), `opt/gvn.rs:12-19, 32-78, 118-138`
(memory classes, `reads`, `writes`, `key_of`), `opt/licm.rs:90-125`
(`memory_kept`, `movable`), `opt/mod.rs:81-157` (`prune_gc_checks`, `dce`),
`ir/verify.rs:126`, `jit/helpers.rs:437-450` (`jit_index`).

The pre-rewrite JIT (commit 58a7de9, `src/jit/ir/op.rs`) had the shape we
want: one `Effects { flags, reads: Mem, writes: Mem }` per op with alias
classes, and `is_pure`/`is_movable` derived from it; nothing consumed it then.
The rewrite split it into write bits in `Effects(u32)`, a read table in
`gvn::reads`, and a third path for helpers (`Sig.effects`, write bits only,
with `is_pure` excluding every helper). LuaJIT gives ops and C calls the same
mode letters (`lj_ir.h:287-298`, `lj_ircall.h:153`); Turboshaft's `OpEffects`
(`turboshaft/operations.h:546`) is `produces`/`consumes` over effect
dimensions plus `can_allocate` and `required_when_unused`, with purity and
hoistability derived. This item unifies ours the same way.

1. **The struct.** In `ir/ops.rs`:

   ```rust
   bitflags Mem: u16 { SLOT, TAB, ARR, SHAPE, META, UPVAL, FRAME, GC }
   bitflags Flags: u8 { MAY_DEOPT, MAY_ALLOC, TAILOUT, TERMINATOR, SIDE_EFFECT }
   struct Effects { flags: Flags, reads: Mem, writes: Mem }
   ```

   `TAB` reads and writes stay keyed by the constant key where one exists
   (`TabLoad`/`TabStore{key}`), as GVN's `key_of` and `write_key` do today
   (`gvn.rs:81, 118-138`); the key is on the op, not in the summary.
   `Op::effects()` returns the struct; `HelperId::sig()` returns the same
   struct in place of its `effects` bits. Derived predicates, used by every
   pass: `is_pure()` (no flags, no reads, no writes; true for a helper that
   qualifies), `reads_memory()`, `writes()`, `removable()` (no `SIDE_EFFECT`,
   no `MAY_DEOPT`, no writes), `movable()` (no `SIDE_EFFECT`, no `TAILOUT`,
   no `TERMINATOR`). The `R_*`/`W_*` bits, `W_ALL`, `gvn::reads`, `gvn::writes`
   and `Sig.effects` are deleted; `gvn::MemClass` becomes `Mem`.
2. **Assignments** (the only entries that change meaning):
   - `TabMmBits`, `ClassMm`: reads `META`. An inline `TabStore` through an IC
     entry never changes a class's bits, because the interpreter does not
     cache a store of a metamethod key into a table whose shape is a
     metatable (`vm/ops/field.rs:353`, `Shape::as_mt`,
     `env/shape/mod.rs:1048`), so `TabStore` writes `TAB` only and the guard
     of A4.2 hoists out of any loop without a helper table write.
   - `TabASize`, `TabArray`: read `ARR` (an `ArrStore` writes `ARR` too, but
     with a key of "contents" versus "size/pointer": `ArrStore` cannot
     resize, so `key_of` keeps `TabASize`/`TabArray` valid across it; any
     helper writing `ARR` invalidates both).
   - `Index` (`jit_index`): reads `TAB | SHAPE | META`, no flags: it does
     `raw_get` and the `__index` walk and returns `HELPER_FAIL_VAL` where a
     function would run (`helpers.rs:437-450`), allocating nothing. `RawGet`,
     `RawEqual`, `GetMetatable`, `Type`, `StrByte`, `FMod`, `FPow`, `Sin`,
     `Cos` are pure. `NewIndex`, `RawSet`, `SetMetatable`: `writes TAB | ARR
     | SHAPE | META` and `MAY_ALLOC`: `jit_newindex` ends in `raw_set`
     (`helpers.rs:508-510`) whose `set_int_key` and `rehash_ints`
     (`env/table/mod.rs:1221-1277`) call `slots::alloc`. The other
     `MAY_ALLOC` declarations are audited the same way, with the verdict in a
     comment on each `sig` arm; A5.6 keeps their checks off hot paths.
     `UpvalBarrier`: `writes GC`, `SIDE_EFFECT`.
   - `Call`, `CallMm`, `Dispatch`, `TailEnter`: `reads ALL, writes ALL,
     TAILOUT | TERMINATOR`.
3. **Consumers.** `gvn::key_of` numbers any `reads_memory()` instruction,
   helpers included, by `(op, args, version of each read class)`; `dce`
   removes any `removable()` instruction with no used result; LICM's
   `movable` rule becomes `e.movable() && (e.is_pure() || (reads_memory &&
   memory_kept && on_spine ...))` with the existing spine and `ordered`
   conditions unchanged; `prune_gc_checks` tests `MAY_ALLOC`; the verifier
   requires a snapshot iff `MAY_DEOPT` (`verify.rs:126`).

*Verification:* `TCVM_JIT_DUMP=opt` on queen: one `call.index`-free loop with
one hoisted `TabMmBits`; the IR snapshot tests of `src/jit/tests.rs` gain a
case per changed assignment; an `agrees` case where `__index` is a function
that mutates the table (the helper must still `FAIL` so the mutation happens
in the interpreter); an `agrees` case that assigns `mt.__index` inside a loop
over tables of that metatable (the store goes through a helper, which
invalidates the hoisted guard's class on the next entry).

### A7. Lowering

*Files:* `backend/aarch64/lower.rs:217-290` (fusion, `test_of`), `:1013-1022`
(`Br`), `backend/aarch64/emit.rs:399-403, 421-497, 756-771, 1092-1106`.

1. `Br` and `GuardTrue/False` on a `B1` that is not a fusable compare emit
   `cbz`/`cbnz` (`Test::B1` today emits `cmp w, #0; b.cond`, `:399-403`).
   `cbz` reaches ±1 MiB, so no veneer logic is needed (correction 9).
2. `IsType` with a multi-part set (`NUM`, `NIL | BOOL | STR...`) fuses into a
   `Br` through `type_branch` (`:462-497`), and an unfused one lowers to a
   `cset` sequence instead of panicking in `type_flags` (`:421-459`).
3. `Select` on the result of an `IsType` whose only other use is the same
   block's `Br` is folded by `simplify` into the two arms' edge arguments (the
   `select(c, c, false)` pattern of S2 disappears with A2 anyway).
4. Shared constants: within one block, `KVal`/`KI32`/`KF64` operands that the
   lowering materializes into x16 for a `test` (`NIL`, `BOX`, `FALSE`,
   `asm.rs:337-384`) are cached in a register for the block when used more
   than twice. The general rematerialization of constants is E6.

### A8. Constants the builder does not see (P6)

*Files:* `build/intrinsics.rs:100-118` (`value_now`), `build/tables.rs:355-372`
(`own_load`, `field_loads`), `:547` (`SlotLoc::spilled`), `build/emit.rs:274-278`
(GETUPVAL), `jit/compile.rs:108-116` (`EntryKinds`), `opt/speculate.rs:31-185`,
`env/function.rs:703-725` (`UpValueDescriptor::by_value`).

1. **`_AUX` slots refine like `_INL` ones.** `value_now` looks a field load up
   in `field_loads` and calls `raw_get` on the receiver's current value; an
   `own_load` through a spilled location (`SlotLoc::spilled`) is registered
   in `field_loads` the same way, so array3d's `assert` from `_ENV` is found
   and guarded with `GuardSame` like any intrinsic callee (`:142-150`).
2. **By-value upvalues are constants of the closure.** An upvalue the compiler
   marked `by_value` (`compiler/defs.rs:346-352`: the variable is never
   assigned) holds one value for the closure's whole life. GETUPVAL of such an
   upvalue emits `konst(closure.upvalue(k))` guarded once by
   `GuardSame(UpvalValue(k), konst)` at the region entry (hoisted by GVN
   since `UpvalValue` is pure, `ir/ops.rs:506`), for every value kind, not
   only functions, strings and tables as `value_now` does today. This is
   LuaJIT's `rec_upvalue_constify` (`lj_record.c:1760`) and it is what makes
   richard's `NO_TASK` compare `SameBits` against a constant nil and qt's
   constants immediate.
3. **Live-frame functions at a loop entry.** `EntryKinds` (`compile.rs:108-116`)
   carries the frame's kinds; it gains the frame's values for registers whose
   kind is `FUN`, and `speculate` turns the prologue `Load(r)` of such a
   register into `GuardSame(load, konst)` tagged `Entry` (next to the kind
   guards it already places, `speculate.rs:31-185`). partialsums' `sin`,
   `cos`, `sqrt` then reach `intrinsic_of`.
4. **Intrinsics accept `BIGINT` feedback** (`emit_math`, `intrinsics.rs:339-343`):
   the refusal test becomes "any kind outside `SMALL | BIGINT | FLOAT`";
   `number_f64` already converts `I64` with `LToF`.

*Verification:* `agrees` cases for an upvalue assigned after capture (must not
constify), a loop entered with a function-valued register that is reassigned
inside the loop (the entry guard holds, the loop's own write is SSA), and a
global `assert` replaced by a user function between two runs of the loop
(the `GuardSame` exits and the recompile sees the new value).

### A9. Recompiles: learn for free, back off, let feedback settle

*Files:* `jit/mod.rs:449-522` (`on_exit`), `:563-617` (`record_site`),
`:622-661` (`on_entry_fail`), `:663-689` (`request_recompile`), `:693-772`
(`recompile`), `:431-445` (`retire_entry`), `jit/state.rs:12-20`
(`EXIT_HOT`, `MAX_RECOMPILES`, `MAX_STRIKES`), `:247-291` (`Entry`),
`jit/compile.rs:108-116` (`EntryKinds.seen`).

Milestone A adds widenable exits (`Bounds`, callee `GuardSame` misses) to
`Type`, `Shape`, `Overflow` and `Mm`, so a site can learn its kinds one
exit at a time; with `MAX_RECOMPILES = 4` and a blacklist after it, a site
could be removed from compilation before it is stable. The engines treat
this as a tiering policy, not a count:

- V8 has no deopt limit. A deopt discards the code, resets the function's
  interrupt budget and marks the feedback vector (`deoptimizer.cc:1895-1910`);
  the function is reoptimized only after running a full budget with its
  feedback unchanged (`minimum_invocations_after_ic_update`,
  `tiering-manager.cc:486-540`). Only compile failures and oversized
  bytecode disable optimization (`ShouldOptimize`, `:412-450`).
- JSC counts exits against a threshold (`osrExitCountForReoptimization =
  100`, 5 inside a loop, `OptionsList.h:394-395`), doubles both the warm-up
  and the exit thresholds per reoptimization (`reoptimizationRetryCounter`,
  `CodeBlock.cpp:2618, 2743-2748, 2880-2891`, capped only so they do not
  overflow), and records every jettisoning exit as a `FrequentExitSite`
  that the DFG consults before speculating (`DFGOSRExitBase.cpp:35-60`,
  `DFGGraph.h:386-506`).

We already have the exit profile (`kind_exits`, the feedback bytes, B5's
kinds per exit) and the `NeverRan` free recompile. Three changes complete
it:

1. **A recompile that learned something is free.** `on_exit` compares the
   site's feedback byte, its `shapes_seen` list and its callee slot before
   and after `record_site`; a recompile requested by an exit that changed
   any of them does not count against `MAX_RECOMPILES` (the `NeverRan`
   rule, `jit/mod.rs:497-521`, generalized). A recompile whose exit
   learned nothing counts: that is the flapping case the budget is for.
2. **Backoff instead of a cliff.** `Entry` gains `retries: u8`; each counted
   recompile increments it and the entry's exit threshold and hot threshold
   become `EXIT_HOT << retries` and `HOT << retries`. The blacklist
   (`retire_entry`) comes at `MAX_RETRIES = 6`, replacing `MAX_RECOMPILES`
   as the cliff; `MAX_STRIKES` for compile failures is unchanged.
3. **Let feedback settle.** After a learned exit, `request_recompile`
   returns the entry to counting at `HOT / 4` instead of compiling on the
   next entry (today the slot points at `jit_recompile`, which compiles
   immediately, `:663-689`). Kinds that arrive within a few hundred
   executions of each other are then folded into one compile. This is the
   direct fix for T3's churn (deltablue: 50 recompiles of 122 compiles,
   nearly all from shape exits that arrive in bursts).

*Verification:* `TCVM_JIT_LOG=1` counts of compiles and recompiles on
deltablue, havlak and json before and after (targets: recompiles under a
third of compiles, no blacklists on the suite); a unit test in `jit/tests.rs`
that drives an entry through six learning exits and checks it is still
compiled, then through six non-learning exits and checks the thresholds
doubled and the entry retired.

### Milestone A exit criteria

- `tests/suite` passes with the JIT on, off, `TCVM_JIT_CHECK=1` and
  `TCVM_JIT_DEOPT_ALL=1`.
- `luabench/queen.lua`'s inner loop region: no `bl` on the hot path, ≤ 25
  instructions per `isplaceok` iteration (`tools/hot.py`-style count of the
  loop's straight-line path from `TCVM_JIT_DUMP=asm`).
- Suite cycles: scimark-*, heapsort, nsieve, towers, list, bounce level with
  or ahead of LuaJIT; queen and linear-sieve within 1.3×.

---

## 4. Milestone B: types on values (P2)

Goal: a loop whose steady state is typed compiles typed code even when the
first iteration or a rare path has another kind. Kernels: spectral's add
becomes one `fmadd` once P3 removes the call around it; partialsums 1260 →
about 400 instructions per iteration; series stops recompiling.

### B1. Peel more

*Files:* `opt/peel.rs:10-37`, `build/mod.rs:29` (`OSR_PEEL_LIMIT`),
`build/mod.rs:1034` (`TooLarge` at 4000 instructions).

- `PEEL_LIMIT: usize = 48` → `400`, counted in IR instructions as today. The
  condition "innermost only" stays for this limit; an outer loop whose inner
  loop body is under 100 instructions is peeled too (nested loops with a
  small inner loop are the scimark shape).
- Peeling runs before `optimize` (`compile.rs:121`), so the peeled iteration
  is built from the entry's kinds: spectral's `a` is the `I32` constant `0`
  in the copy, and the loop proper joins only the copy's back edge and its
  own, both `F64` once A2 types the add as `IToF + FAdd` in the copy and
  `ToF64 + FAdd` in the body. `narrow` (`infer.rs:109-144`) then makes the
  parameter `F64` through its existing `within(FLOAT)` rule and `simplify`
  folds `ToF64(Box(F64))` (`simplify.rs:70-97`). No new representation
  machinery is needed for this case.
- Compile-time guard: a region over 2500 IR instructions after peeling skips
  the peel (the budget of E1 counts the same way).

### B2. Representation selection on parameters, kept exact

*Files:* `opt/infer.rs:93-144` (`narrow_rep`, `narrow`), `:186-231`
(`convert`), `build/mod.rs:1170-1264` (`remove_trivial_params`,
`conversion`).

- `narrow_rep` keeps its three rules (`SMALL` → `I32`, `FLOAT` → `F64`, `INT`
  with an `I64` argument → `I64`) and gains none that changes a value's Lua
  type (correction 4). What it gains is *use-driven* `F64` for a parameter
  whose incoming values are all `F64` already but whose set is wider because
  an edge carries a `Val` of set `FLOAT` (a `Load` from a resume block or an
  `UpvalValue`): today the `Val` edge keeps the parameter `Val`; new,
  `convert` unboxes that edge with `Unbox(F64)` (safe because the set is
  within `FLOAT`), as it does for `narrow`'s own targets (`infer.rs:136-140`).
- `conversion` in `remove_trivial_params` (`build/mod.rs:1218`) attaches the
  current `Before` snapshot to the `LToI` it creates (correction 12), or
  refuses the merge when none is available.
- `Ty::join` (`types.rs:134`) is unchanged; `Select` is its only user.

### B3. `SMALL | FLOAT` feedback is a number, not polymorphism

*Files:* `build/emit.rs` (A2's `operand_kind`), `build/intrinsics.rs:339-343`.

A byte of `SMALL | FLOAT` with no `BIGINT` gives the operand kind `Num`: a
`guard(NUM)` and a `ToF64` (tag test, `scvtf`/`fmov` select: 4 instructions)
feeding the float op. Both arms of today's `small_or_float` dispatch
(`emit.rs:992-998`) produced the same float op, so this is strictly less code.
When the value's type is known the guard and the select fold, which is what
happens for every loop-carried value after B1.

### B4. `I64` operands of float consumers

*Files:* `build/emit.rs` (A2 table rows "any int kind with `F64`"),
`build/intrinsics.rs:339-343, 475-485`.

Covered by A2 (an `I64` operand of a float op is one `LToF`; the cast rounds,
which is Lua's own semantics, verified in the report) and A8.4 (`emit_math`
accepts `BIGINT`). Range analysis (E3) later narrows an `I64` whose bounds
fit back to `I32`, so a counter that overflowed once in a dead branch does not
stay wide.

### B5. Feedback with a sense of time

*Files:* `jit/mod.rs:449-522` (`on_exit`), `build/mod.rs:1135-1140`
(`kind_exited`), `jit/state.rs:282` (`kind_exits`).

- `kind_exits` records, per pc, the kinds the exit saw (a byte), not only the
  fact of an exit; `operand_kind` prefers the exit's byte over the site's
  historical byte when both exist, since the exit's byte is what the
  recompiled region will meet.
- A `Type` exit at `EXIT_HOT` recompiles with the site widened to the union;
  an exit that fires once on a `NeverRan` block keeps its free recompile
  (`:497-521`).

### Milestone B exit criteria

- `agrees` cases: a loop whose accumulator starts integer and becomes float,
  printed with `math.type` after 0, 1 and n iterations; a counter that
  overflows i32 mid-loop and feeds `math.sin` and `^`.
- partialsums' loop region: `sin`, `cos`, `sqrt` inline (`fsqrt`, `bl sin`),
  no `call.arith.*`; series compiles once; spectral's inner loop has one
  `fmadd` and no `ToF64` once B1's peel runs.
- Suite: partialsums and series within 1.2× of LuaJIT.

---

## 5. Milestone C: values across calls, builtins through cores (P3 steps 1–2, P4)

Goal: a call from compiled code to compiled code costs the call itself:
nothing is boxed, stored, reloaded or re-guarded around it. Builtins reached
from compiled code run their fast cores. Kernels: spectral 222 → about 110
instructions per inner iteration, fib 79 → about 55 per call, array3d 838 →
about 300, coroutine-ring 97.6 → about 70 cycles per round trip.

### C1. Values stay in the native frame across fast calls

The design: a fast call keeps the caller's native frame (J10, in place), so
values live across the call stay in the caller's spill slots. Home slots are
written only when compiled code is actually left, by a runtime routine that
walks the native frame chain and materializes every frame's values from a
per-call-site snapshot. The resume point has two entries: the fast one (a
`ret` lands, the frame and its spill slots intact) and a slow one, reached
through header word 1 from the interpreter, that re-opens the frame and
reloads the spill slots from the home slots the walk wrote.

*Files:* `ir/ops.rs:588-640` (`Call`, `CallMm`, `Resume`), `ir/verify.rs:163`
(`check_r6`), `build/mod.rs:166-174, 303-321, 566-647` (`new_root`,
`root_load`, `store_live`, `call`), `build/emit.rs:1577-1607` (`call`),
`:1013-1040` (`staged_call`), `opt/calls.rs` (`call_boundaries`),
`opt/snaps.rs`, `opt/gvn.rs:213-217`, `opt/licm.rs:96-99`,
`backend/aarch64/lower.rs:312-389, 983-1050`, `backend/aarch64/emit.rs:94-108,
165-203, 311-351, 1107-1127, 1176-1242`, `backend/aarch64/abi.rs:104-164`
(`exit_common`), `backend/vcode.rs:127-187`, `region.rs`, `jit/mod.rs:84-152`,
`jit/helpers.rs`, `jit/layout.rs`, `jit/compile.rs:351-384`.

**C1.1 The IR: a call is a clobber, not a boundary.**

- `Call { a, nargs, c, pc }` and `CallMm` stay terminators with edge 0 the
  resume block, but the resume block is an ordinary successor: `Builder::call`
  (`build/mod.rs:637-647`) no longer clears its predecessors or makes it a
  root; `defs` flow across the edge as SSA values (edge arguments), and
  `read_var_at` works unchanged. `BlockData.resume` stays true (the backend
  needs it for the two entries).
- `store_live` before a call stores only what the callee's frame needs: the
  arguments (`a + 4 ..`) and the callee slot `a` (word 0). Captured registers
  (open upvalues, `cfg.captured`) are stored too, since the callee may read
  them through the cell, and after the call they are reloaded (`root_load`
  stays for captured registers and for result registers, both `ANY`).
- `Resume { c }` keeps defining the 1 or 2 landed results.
- `Call` gets a snapshot (`InstData.snap`), the *call-site snapshot*: the
  live Lua registers after the call whose value is not in its home slot, like
  an exit's `After(pc)` snapshot (`snap_after`, `build/mod.rs:535`), minus the
  result and captured registers. Its `ExitKind` is `After`.
- R6 is replaced: `check_r6` (`verify.rs:163`) now rejects only a `Ptr`-typed
  value live into a resume block (table array pointers and slot cells are
  recomputed after any call, since the callee may resize), and `call_boundaries`
  (`opt/calls.rs`) rematerializes only those (`Unbox(Ptr)` and `TabArray`
  chains are rematerializable by the existing rule). `f.boundaries` and the
  "nothing at all once boundaries is set" rule go away.
- GVN scopes no longer restart at resume blocks (`gvn.rs:213-217`); the
  memory versions do, because a call writes every class. LICM's call-in-loop rule
  (`licm.rs:96-99`) relaxes to the same: pure values and memory-independent
  guards hoist across calls, memory reads do not.

**C1.2 Register allocation.** The `Call`/`CallMm` MInsts declare a clobber of
every allocatable register (`vcode.rs:176-178` returns the per-inst set;
today it is empty for calls, `lower.rs:1026`). regalloc2 then keeps every
value live across the call in a spill slot, which is the caller's frame
(`[x29 + 16 + 8s]`, `emit.rs:94-96`). The call-site snapshot's entries are
lowered like an exit's (`exit_of`, `lower.rs:312-360`) but with
`OperandConstraint::Stack` instead of `Any`, so each entry's location is a
spill slot or a constant, never a register. Later (D) a fast entry can save
callee-saved registers so values may stay in x19–x28/d8–d15; C1 does not.

**C1.3 Frame layout.** Every region frame gets one more fixed word:
`[x29 + 16]` holds the region's `region_word` (already a per-region
constant patched at install, `emit.rs:200-203`); spill slots start at
`[x29 + 24]`. The `Prologue` and `open_frame` (`emit.rs:295-299`) store it.
The trampolines of A5 and `jit_barrier`/`jit_box_i64` calls push ordinary
`x29, x30` pairs, so the chain from any point in compiled code to
`NATIVE_FP` is a sequence of region frames and C frames, each with its
return address at `[fp + 8]`.

**C1.4 Site table.** Each region records, for every point where control can
leave compiled code while this frame is live, a *site*: `Site { ret_off: u32,
snap: u32, pc_after: u32, resume_off: u32 }` keyed by the code offset of the
return address that the chain will hold for it:

- a fast call: the address after the `blr` (the `b fast_resume` instruction)
  and, for deep chains, `fast_resume | 1` (`emit.rs:329-340`); the site
  stores the resume label and the walker masks bit 0;
- the slow call path, `Dispatch`, `TailEnter`'s dispatch arm, `CallMm`'s slow
  path, `fast_miss`, the entry-fail stub: each becomes `bl jit_leave_stub`
  (so `x30` is the site key) instead of the inline `leave()` sequence;
- exits: `exit_common` handles the innermost frame itself (below) and the
  walker handles the callers.

`region.rs` stores `sites: Box<[Site]>` sorted by `ret_off`, and the
snapshot arena gains the call-site snapshots (same `encode_snap`, `Loc::Spill`
and `Loc::Const` only). `compile.rs:351-384` encodes them with the exits'.

**C1.5 `jit_leave`.** A Rust `extern "C" fn jit_leave(rt, fp: *const u64, lr:
u64, base: *mut Value)` in `jit/helpers.rs`:

```
frame = (fp, lr, base)
loop:
    region = *(fp + 16)                      ; C1.3
    site   = region.sites.find(lr & !1 - region.code.start)
    for e in snapshot(site.snap): base[e.reg] = materialize(e, spill = fp + 24)
    write header words 1–3 of the Lua frame *below* this one if the site is a
      call (D4 makes them lazy; until D4 they are written at the call and this
      is a no-op)
    if *fp == NATIVE_FP: break
    lr   = *(fp + 8); fp = *fp               ; the caller's frame
    base = *(base - 2)                       ; callee header word 2 = caller base
```

The innermost frame's `base` is `BASE` at the leave site. Materialization
boxes `I64` entries with `Value::integer(mc, ..)`, which allocates without a
collector check, as `jit_exit` does today (`jit/mod.rs:129-131`); the next
check happens in the interpreter. Each tail-out site does `bl
jit_leave_stub` and then its own continuation (`b enter`, `br x16`, `b core`);
`x30` at the stub is the site key. The stub, one per region in the cold
section, replaces the six inline `leave()` sequences:

```
jit_leave_stub:
  mov  x19, x30                 ; the site address, kept across the C call
  mov  x0, RT
  mov  x1, x29                  ; the region frame: the walk starts here
  mov  x2, x30                  ; the site key
  mov  x3, BASE
  bl   jit_leave                ; a C call made from the region frame
  ldr  x17, [RT, #NATIVE_SP]    ; today's leave(): drop every compiled frame
  mov  sp, x17
  ldr  x29, [RT, #NATIVE_FP]
  ldr  x30, [RT, #NATIVE_LR]
  br   x19                      ; back to the site's continuation
```

`exit_common` (`abi.rs:104-164`): after saving the register image and the
innermost frame's spill words (as today), and *before* resetting `sp`, it
calls `jit_leave_callers(rt, *x29, *(x29 + 8), *(BASE - 2))`, the same walk
starting at the caller. The innermost frame's own values come from the exit
snapshot as today. `jit_exit` is unchanged except that `Loc::Spill` indexes
shift by one word (C1.3).

**C1.6 The two resume entries.** For each resume block the emitter already
binds `fast_resume[block]` after `open_frame` (`emit.rs:1176-1217`). New:

- the fast entry (`ret` lands): `BASE = [BASE - 16]`, `CLOSURE` reload (D4
  makes this lazy), land the results, continue; spill slots are intact.
- the slow entry (header word 1, from the interpreter): `open_frame`, store
  the region word, then a *reload stub* generated from the call-site
  snapshot: for each entry `ldr x16, [BASE, #reg*8]`, unbox per `rep`
  (`I32`: no-op alias; `F64`: `fmov d16, x16` then `str d16`; `B1`: `bits -
  FALSE`; `I64`: `ldr` through the boxed int's payload, which is a `Gc<i64>`
  deref, `env/value.rs:139-161`), `str` to the entry's spill slot; then `b
  fast_entry_path`. The stub lives in the cold section, is 32-byte aligned
  (header word 1 carries flags in the low 5 bits, `vm/frame.rs:32-44`), and
  is what the `Call` writes into word 1 (today the resume label itself).

**C1.7 What a fast call writes now.** Word 0 (the callee, already stored by
`Store(a)`), the arguments, `INSN = nargs`; words 1–3 as today until D4. The
17 stores and 11 loads per spectral iteration become spill traffic only for
values regalloc2 cannot keep in registers across the call, which is all of
them in C1 (12 live values → 12 `str`/`ldr` pairs of raw words, no box, no
unbox, no guard) and few after D's callee-saved registers. Measured target
after C1 alone: spectral about 110 instructions per inner iteration.

**C1.8 Invariants, stated for the verifier and the tests.**

- Every frame between a leave site and `NATIVE_FP` is a region frame. This
  holds because helpers are leaves (R5: they run no Lua and never call back
  into compiled code), so the C frames of the trampolines, `jit_barrier` and
  `jit_box_i64` are never on the chain when `jit_leave` runs; the only
  frames a region pushes under itself are fast callees' region frames.
  `jit_leave` asserts in debug builds that `[fp + 16]` of every frame it
  visits is a known region word (`JitState::regions`), and `TCVM_JIT_CHECK`
  makes the assertion unconditional.
- A `TailEnter` to a callee with a fast entry closes this frame and branches
  with the frame's return address (`emit.rs:1135-1175`), so the callee's
  frame takes this one's place in the chain and its return address is the
  tail-caller's caller's site. The site table therefore covers that case by
  construction (the key is the caller's return address, whichever region
  returns through it), and a tail-caller has nothing live, so no entry of
  its own is needed. The `TailEnter` dispatch arm for a callee without a
  fast entry is a leave site like the others.
- A `Ptr` value never lives across a call (verifier).
- No collection runs while a region frame holds unmaterialized values: every
  path to the collector or the unwinder goes through `jit_leave` (the slow
  call path, `Dispatch`, `TailEnter`, `fast_miss`, exits, metamethod slow
  paths), and a fast callee that reaches any of them walks the whole chain.
  A debug assertion in `Lua::enter`'s collector check that `rt.native.fp`
  equals the current frame pointer (no compiled frames below) guards this.
- Stack growth (`grow_slots`, `env/thread.rs:307-318`) rebases headers and
  open upvalues; spill slots hold no stack pointers, so they need no rebase.
- A frame materialized by `jit_leave` and then resumed through the slow entry
  reads the home slots the walk wrote; a frame resumed through the fast entry
  never had its slots written, and its spill slots were never read by anyone
  else. The two are exclusive because `leave` destroys the native frames.

*Verification:* `src/vm/tests/protocol.rs` gets: a region calling a function
that yields, errors, triggers a collection, and grows the stack, with values
of every rep live across the call, resumed through both entries (force the
slow entry by calling a native in between; force the fast entry with a
compiled callee). `TCVM_JIT_DEOPT_ALL=1` exercises every exit's caller walk.
`agrees` cases with deep recursion (fixpoint-fact's 100-deep chain crosses
`NATIVE_DEEP`, so the `| 1` return addresses are covered). A `TCVM_JIT_CHECK`
extension: after `jit_leave`, assert every frame's home slots equal what a
full snapshot replay would give (debug builds).

### C2. Builtins through their cores (P4)

*Files:* `vm/ff.rs:18-70` (`fast_entry!`), `:82-241` (the entries),
`vm/coro.rs:161-333` (`resume_switch!`, `ff_resume`, `ff_yield`, `ff_wrap`),
`vm/native.rs:469-496, 620-729` (`native_enter`, `ff_pcall`, `ff_xpcall`,
`ff_pairs`, `ff_ipairs`), `vm/ops/call.rs:297-325` (`enter`),
`env/function.rs:365-377` (`NativeClosure`), `:852-875`
(`new_native_with_entry`), `build/intrinsics.rs:39-95, 157-224`,
`build/emit.rs:1543-1607`, `backend/aarch64/emit.rs:1107-1127`.

1. **The core contract.** A core is a `Handler` (`vm/abi.rs:100-107`) entered
   with `pc` = the header, `insn` = `nargs`, `base` = the caller's window,
   `closure` = the native closure, word 0 = the callee value, word 1 = the
   continuation plus flags (a region's slow resume entry, or the
   interpreter's `rt.ret(c)`), words 2–3 as `enter` expects. It returns by
   `tail!(frame::ret(hdr), pc = values, insn = nret, base = win)` like
   `native_enter` (`native.rs:485-492`), keeping word 1 as written and ORing
   `NATIVE` (and `PROTECTED` for `resume`). On a miss it tails
   `native_enter` with the same state.
2. **`NativeClosure` gets `core: Option<Handler>`**, set beside `entry` by
   `new_native_with_entry`. `enter`'s native arm (`call.rs:305-306`) becomes
   `match nc.core { Some(c) => tail!(c, closure = Slot::native(nc)), None =>
   tail!(native_enter, ..) }`: one load and one branch for every native call
   from the interpreter too.
3. **`fast_core!`**, a twin of `fast_entry!` (`ff.rs:18-70`) that binds `a`
   as `(pc - base) / 8`, `nargs` from `insn`, has no `c` (results are
   delivered by count), and whose `land1!` is the `tail!(ret, ..)` above. The
   existing entries are rewritten as cores, and `fast_entry!` becomes the
   CALL-word adapter in front of them (it computes `a`, `nargs`, writes the
   header the way `call_body!` does for the slow path, and tails the core),
   so the interpreter's CALL keeps its current cost.
4. **`resume_switch!`** (`coro.rs:161-233`) takes the continuation from the
   header instead of `handler_bits($rt.ret(c))` (`:207`): the core variant
   passes `frame::ret(hdr)`. `ff_yield` (`:301-307`) likewise keeps the
   resumer's word 1. Both already return through word 1 of the other side,
   so a compiled resumer and a compiled coroutine stay compiled across the
   switch. `pcall` / `xpcall` cores write their own continuation
   (`ret_pcall`, `ret_xpcall`) as today but take the caller's continuation
   from the header rather than `rt.ret(c)`.
5. **Compiled callers.** `build/intrinsics.rs` learns a third outcome for a
   known native callee: `Core`. `emit_call` then emits `Call` as today; the
   backend's `Call` emission (`emit.rs:1107-1127`) for a callee the builder
   marked `known_native` skips `fast_call`'s tests and does `bl
   jit_leave_stub; b core` directly (`enter`'s checks are the core's own).
   The region resumes through its slow entry (C1.6), because every core may
   switch threads or run Lua. Non-switching builtins that already have
   helpers (`StrSub`, `StrByte`, `Type`, ...) keep the helper path, which is
   cheaper than a core since the frame stays.
6. **`select`** gets a fast core (`basic.rs:624-650` allocates a `Vec` per
   call; a core copies `nargs - n` values directly); `tostring` and
   `string.format` stay continuation natives.
7. **`pcall` and `xpcall` cores, traced.** `ff_pcall` (`native.rs:650-663`)
   builds a second header: the protected callee's, overlaid on the CALL's
   hidden slots at `base + a + 1`, with word 1 = `ret_pcall` (`:620`) and
   `PROTECTED`, then tails `enter` for the callee. From a region the path
   is: the region writes the `pcall` call's own header at `a` (word 1 = the
   slow resume entry of this site, since the core may run Lua that yields
   or errors), `bl jit_leave_stub`, `b ff_pcall_core`. The core reads
   `nargs` from `insn` and `a` from `pc - base`, writes the inner header at
   `a + 1` with `ret_pcall` and the outer header untouched, and tails
   `enter`. On a normal return, `ret_pcall` (`native.rs:620-634`) prepends
   `true`, pops the inner frame and returns through the *outer* header's
   word 1, which is the region's slow resume entry: the frame re-opens and
   the reload stub restores the spill slots. On an error, the unwinder
   finds the `PROTECTED` inner frame, delivers `false, err` through the
   same `ret_pcall` path, and lands on the same entry. `xpcall` is the
   same with the handler moved to `R[a + 1]` and the inner header at
   `a + 2` (`:667-682`). The protocol test for this is a compiled loop
   calling `pcall(f)` where `f` returns normally, errors, and yields across
   the `pcall` from a coroutine, each resumed through the slow entry with
   values of every rep live across the call.

*Verification:* `agrees` cases for `coroutine.wrap` generators driven from a
compiled loop, `resume` of a dead coroutine, a yield across a `pcall` inside
the coroutine, an error inside `resume` from compiled code (the `PROTECTED`
flag and `resume_cont`), `select('#', ...)` and `select(n, ...)` from
compiled code; `src/vm/tests/calls_lua.rs` extended with a core called from
the interpreter with `nargs` below and above the arity. Measurement:
coroutine-ring per-round-trip cycles ≤ 72 (LuaJIT 68.6), chameneos ≤ 72 per
meeting.

### Milestone C exit criteria

- spectral inner iteration ≤ 115 instructions, fib ≤ 58 per call, array3d
  `array_set` ≤ 320, coroutine-ring ≤ 72 cycles per round trip.
- The protocol tests above pass under `TCVM_JIT_CHECK=1` and
  `TCVM_JIT_DEOPT_ALL=1`.

---

## 6. Milestone D: known callees, results in registers, lazy headers (P3 steps 3–4)

Goal: a call whose callee is known at compile time costs what a C call costs
plus one store: no dynamic callee test, no repeated entry checks, results in
registers, header words written only when something leaves compiled code.
Kernel: fib 55 → about 20 instructions per call.

### D1. Callee identity

*Files:* `instruction.rs:889, 1119-1121` (`CALL` is `Abc`, `CALLS*` are
`AbcImm`), `compiler/defs.rs:257-308` (`next_ic_idx`, IC index allocation),
`env/function.rs:31-78` (`Prototype`), `:330-350` (`LuaClosure`),
`vm/ops/call.rs:139-220` (`call_body!`, the `CALLS` forms), `:380-420`
(`op_tailcall`), `jit/feedback.rs`, `build/intrinsics.rs:100-118`
(`value_now`), `build/emit.rs:1543-1607`, `jit/mod.rs:449-522` (`on_exit`).

Call sites record their callee in the interpreter, as V8's Ignition
(`CollectCallFeedback` into the feedback vector, with a call count) and JSC's
LLInt (`CallLinkInfo`) do. The interpreter's `CALL` fast path is ten
instructions today, so the write is designed to cost three on a hit.

1. **A callee slot per call site.** The compiler allocates a slot index per
   `CALL`, `CALL_R0/R1`, `CALLS*` and `TAILCALL` site, like the field IC
   index (`defs.rs:305-308`), in the instruction's `d` field (`CALL` is
   `Abc`, so `d` and `e` are free; `CALLS*` carry `src` in the 32-bit `imm`,
   `instruction.rs:446`, and become `Abcde { src: Reg in d, slot in e }`).
   `Prototype` gains `callees: Box<[CalleeSlot]>` with `CalleeSlot { target:
   Cell<u64>, count: Cell<u32> }`, one per site, zero-initialised; the
   closure keeps a pointer to it beside `ic_table` (`function.rs:338`).
2. **The interpreter write**, in `call_body!` after `get_function` and in
   `op_tailcall`:

   ```rust
   let slot = &callees[insn.d()];
   if slot.target.get() != fv.raw() {                // 3 instructions on a hit
       slot.target.set(if slot.target.get() == 0 { fv.raw() } else { MEGAMORPHIC });
   }
   slot.count.set(slot.count.get().wrapping_add(1)); // 3 more; see below
   ```

   `MEGAMORPHIC` is a non-value bit pattern (`0xFFF8_0000_0000_0001`, no
   `Value` has it since floats are canonical). The target is the closure's
   raw value: `GuardSame` needs the closure, and the builder may choose to
   guard the prototype instead when the closures differ per object (the
   `FN_PROTO` load and compare, `layout.rs:65-86`). The count feeds E1's
   budget; if the measurement below shows it costs more than 2% on
   interpreted call-heavy benchmarks, it moves to a per-prototype invocation
   count in `start!` (`vm/ops/call.rs:123-134`), which V8 also keeps, and
   sites are ranked by their callee's count.
3. **Measurement before merge.** `tools/ljr.py` with `TCVM_JIT=off`, the
   current binary against the one with the slot write, three interleaved
   rounds, cycles: the budget is 3% on recursive-fib-uv and fixpoint-fact
   (the most call-bound interpreted runs) and noise elsewhere. The hot
   counters were accepted on the same basis (design 6.1).
4. **Readers.** The builder's known-callee lookup, in order: a by-value
   upvalue (A8.2), a global through `_ENV`'s IC (A8.1), a `_PROTO` holder's
   or an own slot (`field_loads`), a closure the region created
   (`Helper(Closure)`'s result carries its prototype in `Refine`, new), the
   live frame's register at either entry kind (`jit_hot` passes the frame
   for function entries too, `jit/mod.rs:68`; `compile.rs:108-116` reads it
   only for loop entries today), and the site's `CalleeSlot` when it is
   monomorphic. A megamorphic slot means "do not guard": the site keeps the
   dynamic fast call, with no `GuardSame` and no recompile on a miss.
   Compiled code does not write the slot (the interpreter already did while
   the site warmed up, and a region with a known callee has guarded it).
5. **Exits.** A `GuardSame` miss on a callee (tag `Type`) records the new
   callee into the slot through `on_exit` (`record_site` gains the CALL arm),
   promoting it to megamorphic on the second distinct value, so the
   recompile stops guarding.

A known Lua callee gives the builder its `Prototype`: `num_params`,
`max_stack_size`, `is_vararg`, whether every `RETURN` yields exactly one
value (a one-time scan cached in the prototype, `returns_one: Cell<u8>`),
and whether a function region with a fast entry exists (`jit_fast`).

### D2. The typed entry and the window check

*Files:* `backend/aarch64/emit.rs:128-147` (fast entry), `opt/speculate.rs`
(entry guards, `ExitTag::Entry`), `backend/aarch64/lower.rs:532-547`
(`ENTRY_FAIL`), `region.rs:162-211`, `jit/mod.rs:339-422` (`install`),
`backend/aarch64/emit.rs:1107-1127`.

1. **Three entry labels per function region**: `fast_entry` (today's: window,
   arity, native stack checks, then the prologue's entry guards), `checked`
   (past the three checks, at the prologue: for a caller that proved the
   window and arity), and `typed` (past the entry kind guards, for a caller
   whose stored arguments are proven to have the guards' kinds). `Region`
   records all three; `Prototype` gains `jit_checked` and `jit_typed` beside
   `jit_fast`, cleared together by `request_recompile` (`jit/mod.rs:663`).
   The typed entry's argument kinds are the entry guards' sets, stored in
   `Region::entry_regs` (`region.rs`), so a caller can compare its
   arguments' types against them at compile time.
2. **The caller's checks.** For a known callee the `Call` lowering skips
   `fast_call`'s tag, vararg and `PROTO_FAST` tests and does `ldr x17,
   [proto, #PROTO_TYPED]` (or `_CHECKED`, `_FAST`), `cbz x17, slow`, `blr
   x17`: the load is kept so a retired callee region needs no patching of its
   callers (an embedded `bl` with caller-list invalidation is a later
   refinement, noted in section 11). The window check for the callee is
   folded into the caller's own entry check: the region's prologue checks
   `base + (max_stack + deepest) * 8 <= stack_end`, where `deepest` is the
   maximum over known callees of `a + 4 + callee.max_stack_size`, recorded by
   the builder in `RegionMeta` (`ir/mod.rs:170`). Arity is static: the
   builder nil-fills missing arguments in the caller (`TailEnter`'s nil fill,
   `emit.rs:1135-1175`, becomes a shared routine) and uses `checked`.
3. **Recursion.** A call of the region's own prototype uses the region's own
   `typed` label directly (`bl typed_entry`), the one case where the address
   is embedded, since the region cannot outlive itself; the native stack
   check (`sp >= NATIVE_LIMIT`) stays in the typed entry for self-recursive
   regions only.

### D3. Results in registers

*Files:* `backend/aarch64/emit.rs:1250-1272` (`Return`), `:1176-1217`
(`Resume`), `backend/aarch64/lower.rs:983-1009`, `ir/ops.rs:592-594`
(`Resume { c }`), `build/emit.rs:1577-1607`.

- `Return { a, n }` with `n == 1` (and `RETURN1`) loads the result into `x0`
  before `close_frame` (it is in a register already; the slot store stays for
  interpreter callers), and the fast-return path `ret`s with it. A float
  result is also placed in `d0` (`fmov`), so a caller that knows the result
  is `F64` reads `d0` with no unbox.
- `Resume { c: 2 }` for a known callee with `returns_one` defines its result
  from `x0` (lowering: `reg_fixed_def(x0)`), with no `cbz INSN` nil default
  and no `[PC]` load; the result's type is the callee's return type when the
  callee's region proved it (the `Return`'s operand type, recorded in
  `Region` as `ret_ty`), so fib's `n - 1` result is `I32` without a guard.
- Two results (`c == 3`) use `x0`/`x1`; more go through memory as today.

### D4. Lazy headers and the resume's registers

*Files:* `backend/aarch64/emit.rs:1107-1127, 1176-1217`, `jit/helpers.rs`
(`jit_leave`), `region.rs` (`Site` gains `hdr_off`).

- A fast call to a known callee writes word 0 only (`Store(a)` of the closure
  value in its final `lua_func_word` form, `vm/frame.rs:184`, so `enter`'s
  rewrite is skipped on the slow path too). Words 1–3 are constants of the
  site (the slow resume stub address, the caller base which is `BASE`, and
  `pc_after`), so `jit_leave` writes them for every frame it walks (its
  `Site` has `hdr_off = a * 8` and `pc_after`); until then no walker exists
  that could read them, because every walker runs after a `leave`.
  `fast_miss` (`emit.rs:184-199`) and the slow call path write them
  themselves before `enter`, as today.
- The fast resume entry computes `BASE` by `sub BASE, BASE, #(a + 4) * 8`
  (the callee base is the caller base plus the header offset plus 32,
  `fast_call`, `emit.rs:327`) instead of loading word 2, and reloads
  `CLOSURE` only if a later instruction in the region reads it (the lowering
  marks `CLOSURE` dead until its next use; `UpvalValue`, `Closure`,
  `TailEnter` and every tail-out use it).
- The GC tracer reads word 0 only (`env/thread.rs:203`), so a frame with
  stale words 1–3 is safe to trace; it never traces while compiled frames
  exist anyway (C1.8).

### D5. The fib target, checked against the code

With D1–D4 fib's inner node is the report's 20-instruction sequence: `w0 = n`
arrives typed, the window was checked once at the region entry, `n - 2` and
`n - 1` keep their overflow exits until E3's range analysis removes them on
the `n >= 2` edge, word 0 is one `str`, the two calls are `bl fib_typed`, the
result is `w0`, and the frame holds `n` and the first result in x19/x20 once
the typed entry saves callee-saved registers (D2's prologue for a region that
uses them: `stp x19, x20, [sp, #16]` and the matching `ldp`, only when
regalloc2 assigned them, which it does for values live across a `bl` once
`Call`'s clobber set excludes x19–x28/d8–d15 for known callees).

### Milestone D exit criteria

- fib ≤ 24 instructions per call (LuaJIT 36), spectral ≤ 85 per inner
  iteration, array3d `array_set` ≤ 160.
- `agrees` cases: a known callee replaced at run time through its upvalue
  (not by value: must not be known), through `_ENV` (the `GuardSame` exits),
  by `debug.setupvalue` (not implemented; skip); a callee that returns 0, 1,
  2 and `...` results into `c = 2` and `c = 3` sites; mutual recursion deeper
  than `NATIVE_DEEP`; a callee whose region is retired mid-run (hot exit in
  the callee) while a caller region is live.

---

## 7. Milestone E: the missing optimizations (P5)

Each item is small once A–D hold; they are ordered by the per-kernel gain the
report measured.

### E1. Inlining

*Files:* `build/mod.rs` (`Builder`, `snapshot` `:505-520`, `feedback` `:1127`,
`emit_instance` `:1081-1121`, `discover` `:852-979`), `build/cfg.rs`
(`Cfg::build`, `live_in`, `captured`, `never_ran`), `build/emit.rs:1543-1607`,
`region.rs:49-56, 58-88` (`SnapFrame`, `encode_snap`), `jit/mod.rs:99-117`
(`jit_exit`'s frame rebuild, already implemented), `jit/compile.rs:367`
(frames always empty today), `jit/mod.rs:449-522, 563-617` (`on_exit`,
`record_site`: the innermost frame's prototype).

Inlining is done in the builder, at bytecode level, because feedback, live
ranges and never-ran blocks are all per prototype and the `Cfg` machinery
already versions blocks (`Instance`, `build/mod.rs:32-42`).

1. **Frames.** `Builder` gains `frames: Vec<InlineFrame { proto, cfg: Cfg,
   closure: Value, delta: u8 /* caller slot of the callee's register 0 */,
   call_a: u8, c: u8, caller_pc: u32, cont: Block, results: Vec<Val> }>`.
   Register `r` of the innermost frame is caller register `delta + r`; `reg`,
   `set`, `snapshot`, `feedback(pc)`, `kind_exited(pc)`, `live_in` and
   `captured` go through the frame stack (`feedback` reads the innermost
   frame's prototype bytes; `live_in[pc]` of an inner frame is its own
   `Cfg`'s, shifted by `delta`, unioned with the outer frames' live sets at
   their `caller_pc`).
2. **Decision** in `emit_call`: the callee is known (D1), is a Lua closure,
   not vararg, has no `TBC`, no `CLOSURE` that captures a parameter by
   reference, no `RETURN` up to `top`, at most `INLINE_LIMIT = 60` bytecodes,
   the nest depth is under 4, and the region's instruction count plus the
   callee's estimated IR size (its bytecodes × 6) stays under `MAX_INSTS =
   1500`. Sites are considered hottest first within a region (the callee's
   call count is not recorded; use the loop depth of the site as the proxy).
   A call of the region's own prototype inlines up to `RECURSION_UNROLL = 2`
   levels, the innermost level a D2 direct call.
3. **Body.** The callee's `Cfg` is built once per prototype (cached in
   `JitState`), its blocks become new `Instance`s with the frame pushed, the
   arguments are the callee's parameter registers (already stored by the
   caller's `set` of `a + 4 + i`, no `Store`), missing parameters are
   `konst(nil)`. The callee's `RETURN { a, n }` becomes `jump(cont, results)`;
   `cont` has `wanted` parameters (`c - 1`, nil-filled when `n` is smaller).
   `GuardFits` for the inlined frame's window is one check before the body
   (`a + 4 + callee.max_stack_size` folded into D2's region-entry check when
   the site is not under a loop that grows the stack).
4. **Snapshots** list every frame: `snapshot()` emits a `SnapFrame { func:
   pool index of the callee closure, a: call_a, c, delta, caller_pc }` per
   inline frame, outermost first, and entries for each frame's live registers
   at its pc with `reg = delta + r` (the encoding allows 256 registers,
   `region.rs:58-88`). `jit_exit` already rebuilds headers from frames
   (`jit/mod.rs:99-117`); `on_exit` records feedback on the innermost frame's
   prototype at the snapshot's `pc`.
5. **Metamethod callees**: an `ARITH_MM`/`__index`-function site whose
   metamethod value is known at compile time (`ClassMm` on a known class, or
   D1's callee word for the `CallMm`) is inlined like a call with the
   operands as arguments; the `GuardSame` on the loaded metamethod is the
   identity guard. This is mandel-metatable's `__add`/`__mul`.
6. **Upvalues of the inlinee** read through its closure constant
   (`UpvalValue` on `konst(closure)` → constant for by-value upvalues, A8.2).
7. **A real call inside an inlined body.** A `Call` the inlined callee makes
   (one that is not itself inlined) needs the inlined frame to exist on the
   Lua stack before it runs, because the callee of the callee returns into
   it through its header and walkers read it: the builder writes the
   inlined frame's header words before the call, as `jit_exit` would from a
   two-frame snapshot (`jit/mod.rs:99-117`: `func` from the pool, `ret`
   = the continuation, `caller` = the enclosing frame's base, `pc` = the
   inlined `CALL`'s pc after), with word 1 the region's slow resume entry
   for that site (C1.6) so a return through the interpreter lands in the
   region. The call-site snapshot (C1.1) then records both frames: the
   enclosing frame's live registers at `caller_pc` and the inlined frame's
   at its own pc, using the same `SnapFrame` records as exits. `jit_leave`
   writes the headers it finds in the snapshot's frame list outermost first,
   so a chain walk through a region with an inlined frame rebuilds the Lua
   frames in order. After the call the inlined frame's registers are
   reloaded from its slots like the enclosing frame's (`root_load` with
   `delta`), and its header stays in place until the inlined `RETURN`,
   which needs no pop since the enclosing region never read it as a frame.

*Verification:* IR snapshot tests (`src/jit/tests.rs`) for a two-level inline
with an exit in the inner frame (the printed snapshot shows two frames);
`agrees` cases for an error raised inside an inlined callee (the traceback
line numbers must match the interpreter's: `frame::pc_index` reads word 3,
which `jit_exit` writes), a `pcall` around an inlined callee, a yield from
inside an inlined callee (the callee has a call → the inlined body contains a
`Call` whose header is written inline, design 13), recursion unrolled with a
base case hit at each level.

### E2. Bounds checks against the loop range, and range analysis

*Files:* new `opt/range.rs`, `opt/licm.rs` (entry snapshot, `:85-87`),
`build/emit.rs:1611-1683, 1795-1809` (`FORLOOP_I`'s `last` and `IAddNo`),
`ir/ops.rs` (`IAddNo`, `ISubNo`, `IMulNo` exist, `:442`).

1. `range.rs` computes an interval per `I32` value over the dominator tree:
   `KI32`, `FORLOOP_I` counters (`[init, last]`, from `emit_forprep`'s exact
   `last`), `IAdd`/`ISub`/`IMul` of ranged operands, `ICmp` on a dominating
   edge (`n >= 2` on fib's edge), `TabASize` as an upper bound for an index
   proven `Lo` against it. Results: `IAdd` → `IAddNo` when the result range
   fits; `LToI` dropped when the `I64`'s range fits (B4); `GuardTrue(ICmp Lo
   i, asize)` dropped when `i`'s range is within `[0, asize)` by a dominating
   compare.
2. The loop-range form: an `ArrLoad` index that is the loop counter or the
   counter plus a constant, in a loop whose body has no `SHAPE`/`ARR`
   write to that table (LICM's `memory_kept` rule), has its bounds guard
   replaced by `GuardTrue(last + k < asize)` in the preheader with the
   header's snapshot, and `TabArray` hoisted with it. The header snapshot
   comes from a new pure `LoopHeader` marker instruction the builder places
   at every loop header carrying `snap_before(header pc)`, so hoisting no
   longer depends on a `GcCheck` surviving `prune_gc_checks` (`licm.rs:85`).
3. Runs after GVN in `optimize`'s loop and once more after inlining.

### E3. Invariant arithmetic with exits

*Files:* `opt/licm.rs:40-180` (`hoist`, the spine rule `:72-82`).

An `IAdd`/`ISub`/`IMul` whose operands are loop invariant but whose block is
off the spine (queen's `c - n`, `c + n` under the `or` chain) is hoisted as
`LAdd`/`LSub`/`LMul` on `IToL` operands (no exit) in the preheader, plus one
`LToI` there with the `LoopHeader` snapshot (E2). The guard can only fire
where the interpreter would have produced a big integer, so it changes where
the loop runs, never a result. With E2 narrowing the result's range, the
`LToI` often folds.

### E4. Rematerialization

*Files:* `backend/aarch64/lower.rs` (operand creation, `:303-309`; `Src::Const`
for exits `:363-373`), `opt/calls.rs:96-134` (`rematerializable`, `remat`).

Constants (`KVal`, `KI32`, `KI64`, `KF64`, `KObj`), `Unbox(I32)` aliases and
`NIL`/`BOX` test constants are not given a live range across blocks: the
lowering materializes a constant at each use in a different block from its
definition, or at each use after a `Helper` in the same block, as a fresh
vreg. regalloc2 then never spills a constant (queen's spilled `1`, nbody's
spilled limit and `BOX` bound). The call-boundary pass's `remat` list is the
same set and is reused. Over-materialization is bounded: a constant used more
than 4 times in one block keeps one vreg there.

### E5. Allocation sinking

*Files:* new `opt/sink_alloc.rs`, `region.rs:58-88` (object entries in
`encode_snap`), `jit/mod.rs:84-152` (`jit_exit` allocates sunk objects),
`ir/ops.rs` (`Helper(NewTable)`, `Helper(Closure)`, `TabStore`, `TabLoad`,
`SetMetatable`).

As designed (9.10, quoted in the report): after E1, a `NewTable`/`Closure`
result whose uses are constant-slot `TabStore`/`TabLoad`, a `SetMetatable`
with a constant metatable, identity tests against constants, and snapshots,
and which is not stored anywhere else, passed to a call or helper, returned,
or carried around a back edge, is not allocated. Each slot becomes an SSA
value (GVN's store-to-load forwarding, `gvn.rs:141-150`, already gives loads
their stored values), the snapshot gains an object entry (`kind:2 slots:14
template_or_proto:16` then one entry per slot, `Loc::Object` for a register
naming it), and `jit_exit` allocates and fills the objects before the
entries. Requires A6's effect summaries (a `MAY_ALLOC` helper in between does
not escape the object, but one that writes `TAB` does) and C1 (a sunk value live
across a call escapes; it must then be allocated before the call).

### E6. Collector checks and layout

*Files:* `opt/mod.rs:81-111` (`prune_gc_checks`), `build/mod.rs:1004-1005,
1090-1096` (where `GcCheck`s are placed), `backend/order.rs`, `opt/simplify.rs`.

- With A5.6 and A6, `prune_gc_checks` keeps a header check only in loops
  that allocate on a hot path; the only change is that it must keep the check of the peeled
  copy's loop when the copy allocates (the copy's header is neither a loop
  header nor the entry today, `opt/mod.rs:81-111`): treat a `peeled` block
  as a header for this purpose.
- Jump threading: `simplify` forwards an edge into a block whose only
  instruction is a `Jump` (parameters forwarded as edge arguments) and a
  `Br` on a parameter that is constant on an incoming edge; `order::layout`
  then has no empty join blocks (queen's six branch-to-branch jumps).

### E7. Float modulo inline

*Files:* `build/emit.rs:657, 985` (`Helper(FMod)`), `ir/ops.rs` (a new
`FModFloor` op or an expansion in the builder), `backend/aarch64/lower.rs`,
`vm/num.rs` (the interpreter's `luai_nummod`).

LuaJIT inlines float `%` as `b - floor(b / c) * c` (`lj_opt_narrow_mod`,
`lj_opt_narrow.c:558-578`; four instructions, `frintm` for the floor), but
that is Lua 5.1 semantics and differs from 5.5's `fmod` plus sign
adjustment in three verified ways: `-4.0 % 2` is `-0.0` in 5.5 and `0` in
LuaJIT, `5.5 % -math.huge` is `-inf` against `nan`, `1e308 % 3` is `2.0`
against `0`. The inline form therefore follows 5.5: `q = frintm(a / b)`,
`r = fmsub(q, b, a)` (one rounding, exact when `q` is the true floor), a
one-step fix-up when `r` lands outside `[0, b)` or `(b, 0]` (the case where
`a / b` rounded up to an integer), `copysign(0, a)` for a zero result, and
the `FMod` helper when `b` is not finite or `|q| >= 2^52`. About ten
instructions on the common path. It ships only after a generated edge-case
set (signed zeros, infinities, NaN, huge and tiny quotients, exact multiples,
values one ulp either side of a multiple) agrees with `fmod`-based
`vm/num.rs` and with `lua5.5` through `tools/difflua.sh`.

### Milestone E exit criteria

- spectral ≤ 28 instructions per inner iteration (LuaJIT 24), queen ≤ 18
  (17), array3d ≤ 65 (59), ray ≤ 330 per sphere test (273), mandel-metatable
  ≤ 60 cycles per iteration (135), fixpoint-fact's closures sunk.
- The suite's numeric and call-heavy halves within 1.1× of LuaJIT or ahead;
  binary-trees, havlak, storage, fixpoint-fact, deltablue keep their lead
  (P8): compile time share under 2% on each.

---

## 8. Milestone F: compile time and the harness (P7, P9)

### F1. Compile time

*Files:* `jit/compile.rs:142-160` (`optimize`), `jit/state.rs` (`TCVM_JIT_TIME`),
a compile-time bench under `benches/`.

A and C shrink the IR 3–5× on loop regions and remove the hot-path clobbers,
which is the compile-time fix. Add the per-stage timing the design's section
14 describes to `TCVM_JIT_TIME` (it exists; make it print per pass), and a
`benches/jit_pipeline.rs` that compiles the five kernels' regions from fixed
feedback and reports per-stage microseconds, so E1's budget (`MAX_INSTS`) is
tuned on numbers. The T1–T4 items of the design stay as they are.

### F2. The harness

*Files:* `tools/` (new), `cli/` (source loading).

- `tools/ljr.py`: the runner from this investigation (44 benchmarks,
  interleaved rounds, `tcvm-cli`, `TCVM_JIT=off`, `luajit`, cycles and
  instructions from `/usr/bin/time -l`, JSON out) with a prelude file passed
  through `-f` first: `unpack = unpack or table.unpack; loadstring =
  loadstring or load`.
- life.lua is Latin-1 (`ALIVE="¥"  DEAD="þ"` on line 8) and `tcvm-cli`
  rejects non-UTF-8 source (`lua/context.rs:377`). Not planned: the runner
  skips life.lua and the suite is 43 benchmarks (section 11, point 5).
- `tools/perop.sh` and `tools/hot.py` from the investigation (per-operation
  deltas between two sizes; the hot-path instruction count of a region's
  loop from `TCVM_JIT_DUMP=asm`) move into `tools/` so the exit criteria of
  sections 3–7 are reproducible.

---

## 9. Testing and tooling

Every milestone ships with all of these green:

| Check | Command | What it proves |
|---|---|---|
| unit and suite | `cargo test` and `cargo test --test suite` | nothing regressed |
| JIT on/off agreement | `tests/suite/jit_regions.rs` (`agrees`) with the generated kind tables of A2/A3 and the cases listed per milestone | the compiled code's results equal the interpreter's |
| reference agreement | `tools/difflua.sh FILE`: runs `lua5.5` and `tcvm-cli` (JIT on, `TCVM_JIT_HOT_CALL=1 TCVM_JIT_HOT_LOOP=1`) on the file and diffs stdout | the interpreter's semantics, which the kind tables encode, match reference Lua (`cmp_sem.lua`, `arith_sem.lua` from the report are the first two files; the LJR benchmarks' outputs the rest) |
| every exit once | `TCVM_JIT_DEOPT_ALL=1 cargo test --test suite` | every snapshot, including call-site snapshots and inlined frames, replays correctly |
| verifier and allocator checker | `TCVM_JIT_CHECK=1` (default in debug builds) | IR invariants after every pass; clobber sets consistent with operands (A5) |
| protocol | `src/vm/tests/protocol.rs`, extended in C1 and C2 | frames, yields, errors, collections and growth across compiled calls |
| measurements | `tools/ljr.py`, `tools/perop.sh`, `tools/hot.py` | the per-kernel targets of sections 3–7 |

Tooling added by the plan: the `cold` block flag in the IR printer, the
site table in `TCVM_JIT_DUMP=asm` (one line per site: offset, snapshot
entries), `TCVM_JIT_LOG` lines for inlining decisions and sunk objects.

---

## 10. Order, dependencies and the numbers

| Step | Items | Depends on | Measured by | Target |
|---|---|---|---|---|
| 1 | A1–A7, A9 | – | queen, linear-sieve, ray/qt spill counts, array loops | queen ≤ 25 instr/iter; no hot-path `bl`; scimark-*, heapsort, nsieve, towers, list, bounce level with LuaJIT |
| 2 | A8, D1.1–D1.3 (the interpreter's callee slots, so every later region sees them) | A2 (intrinsic kinds) | array3d `assert`, partialsums intrinsics, richard `NO_TASK`; interpreter cycles with the JIT off | intrinsics inline in all three; callee slots within 3% on fib-uv and fixpoint-fact interpreted |
| 3 | B1–B5 | A2, A3 | spectral add, partialsums, series | partialsums ≤ 400 instr/iter, series compiles once |
| 4 | C1 | A5 (trampolines keep x29 chains), A6 | spectral, fib, array3d | spectral ≤ 115, fib ≤ 58 |
| 5 | C2 | C1 (slow resume entry) | coroutine-ring, chameneos, json | ring ≤ 72 cycles/round trip |
| 6 | D1–D4 | C1, A8 | fib, spectral, array3d, ray, richard, deltablue | fib ≤ 24, spectral ≤ 85 |
| 7 | E2, E3, E4, E6 | A, B | queen, nbody, linear-sieve | queen ≤ 18, nbody ≤ 170 |
| 8 | E1 | D1, D2, C1 | spectral, ray, richard, mandel-metatable, json | spectral ≤ 28, ray ≤ 330 |
| 9 | E5 | E1, A6 | mandel-metatable, fixpoint-fact, ray | mandel ≤ 60 cycles/iter |
| 10 | F1, F2 | – (F2's prelude first in practice, so mandelbrot and fannkuch are measured from step 1) | compile shares, the six missing benchmarks | compile share < 2% on deltablue, havlak, json |

Step 1 is the largest and the one every later step builds on; it is also the
one with the clearest verification (the kind tables are exhaustive and
checkable against reference Lua). Steps 4 and 6 are the only ones that touch
the runtime protocol, and they are bracketed by the protocol tests.

---

## 11. Open points

Decisions the plan takes a position on and the user may want to overrule:

1. *(Resolved.)* **Metatable stores and `TabMmBits` (A4, A6).** A shape
   records that its table is a metatable (`Shape::as_mt`) and the interpreter
   never caches a store of a metamethod key into one (`vm/ops/field.rs:353`),
   so inline `TabStore`s cannot change class bits and the `__index` guard
   hoists past them; only helper writes invalidate it.
2. *(Resolved.)* **`NewIndex`'s `MAY_ALLOC` (A6)** stays: `set_int_key`
   can rehash and allocate. The cost moves with the allocation into the cold
   arm (A5.6), so the loop header's check goes.
3. *(Resolved: deferred.)* **Embedded `bl` to a callee region (D2).** A
   known callee is called through `ldr x17, [proto, #PROTO_TYPED]; cbz x17,
   slow; blr x17`; only self-recursion embeds the address. This is what
   TurboFan and Maglev do (`CallJSDispatchEntry`: a load through the dispatch
   table slot the function's handle names, `macro-assembler-arm64.cc:2506`;
   `CallSelf` the one direct call), with `JSDispatchTable::SetCode` the only
   invalidation; LuaJIT embeds trace links and flushes every trace instead.
   Revisit only if a kernel is call-bound after milestone D.
4. *(Resolved.)* **Callee feedback (D1)** is recorded by the interpreter
   per call site, as V8 and JSC do: a target slot with a megamorphic
   sentinel and a call count, three instructions on a hit, measured with the
   JIT off before merging (budget 3% on the most call-bound interpreted
   runs). The first region of a prototype then knows every hot callee,
   polymorphic sites are never guarded, and E1's budget has real counts.
5. *(Not planned.)* **Non-UTF-8 sources (F2).** life.lua stays out of the
   suite. If wanted later: `load_bytes` decodes the source as Latin-1 and
   string-literal decoding in `parser/lit.rs` emits `ch as u8` for chars
   below U+0100, which round-trips every byte sequence exactly with no
   change to the logos lexer or the tree.
6. *(Resolved.)* **`x ^ 0.5`** stays `pow`: tcvm's `^` is `powf`
   (`vm/num.rs:324-334`) like reference Lua, 1290 of 2M doubles differ from
   `sqrt`, and LuaJIT folds only the exponents 0, 1 and 2, which A2 now does
   too. Today both `^` and float `%` are libm calls on every execution
   (`jit_pow`, `jit_fmod`), with the full caller-saved clobber.
7. *(Resolved: deferred to E7.)* **Float `%` inline** follows 5.5's `fmod`
   semantics, not LuaJIT's floor formula, and ships behind the edge-case
   test; `FMod` is pure meanwhile (A6), so it no longer disturbs allocation.

Risks the plan carries knowingly:

8. **The recompile budget** with the new widenable exits is handled by A9
   (learning recompiles are free, thresholds back off, feedback settles
   before a recompile); if the suite still shows blacklists after step 1,
   the retry cap moves, not the rule.
9. **`CONCAT` and string-building loops** are untouched: `CONCAT` goes
   through `helper_or_mm` with the full clobber (`build/emit.rs:292-303`)
   and A2–A7 do not change it. None of the measured kernels is
   concat-bound, so it is a known gap, to be taken up with the
   string-heavy benchmarks (k-nucleotide, fasta once the harness runs it).
