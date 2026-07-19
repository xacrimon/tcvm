# JIT code-quality analysis: `is_prime` on x86-64

> **Status (2026-07-19):** #1 (compare/branch fusion) is **done** — implemented in
> isel as `ICmp`→`Br`/`GuardCond` fusion (`MOp::BrCmp`, and `GuardCmp` reused for
> `GuardCond`). Every boolean-materialization chain in the hot loop is gone: the
> three per-iteration compare sites each shed `setcc; movzx; test` (~9 insns/iter),
> and `is_prime` shrank 361→319 bytes on x86-64 (65→61 instructions on aarch64).
> The residual `mov eax,0` before the fused compares is the deferred imm-fold
> (#2/#4). Remaining work: #2/#3/#4/#5. Updated disassembly at the end of this file.

Analysis of the JIT's output for `is_prime` (from `test-files/primes.lua`), lowered
through the real backend (`lower` → `select` → `annotate` → `allocate` → `encode`)
and disassembled with `objdump`. The question this answers: the output is still
poor after the register-allocator rewrite — what can we reasonably do about it,
given we're committed to linear scan as the base algorithm like LuaJIT and
HotSpot C1?

## Verdict

The allocator rewrite did its job. `regalloc.rs` already has copy-coalescing
(`is_copy`), physical-register hints (`phys_hint`), constant rematerialization,
interval splitting, and loop-weighted Belady eviction — this is a Wimmer/C1-shaped
linear scan, not a naive one. The residual ugliness in the dump is almost entirely
handed to the allocator by the layers *above* it (isel and the frontend).

LuaJIT and C1 aren't beating us with a cleverer allocator; their allocators are the
same class and roughly the same quality now that we have hints + coalescing + remat
+ Belady. Where they pull ahead **without being slower** is that they never ask the
allocator to clean up a mess the IR shouldn't contain: compares are fused into
guards before allocation, snapshots hold constants by reference so loop-invariant
values are never live, and induction variables aren't duplicated across phi columns.
The allocator then sees short, single-home live ranges and a nearly move-free result
falls out. The leverage has shifted upstream.

## The IR the backend is handed

```
block0(v0: int):
      v10 = iconst 2
      v11 = iconst 1
      v12 = unpack.int v0
      v13 = sub.i64 v12, v11
      v14 = iconst 1                          ; second, un-CSE'd `1`
      v15 = icmp le v10, v13
            br v15 block1(v0, v10, v13, v14, v10), block2
block1(v1: int, v2: i64, v3: i64, v4: i64, v5: i64):   ; v2 and v5 are both `i`
      v16 = unpack.int v1
      v17 = iconst 0
      v18 = icmp ne v5, v17
            guard.cond v18  ; exit0
      v19 = mod.i64 v16, v5
      v20 = iconst 0
      v21 = icmp eq v19, v20
            br v21 block3, block4(v1, v2, v3, v4)
block2:
      v22 = kconst true
            ret v22
block3:
      v23 = kconst false
            ret v23
block4(v6: int, v7: i64, v8: i64, v9: i64):
      v24 = iconst 1
      v25 = add.i64 v7, v24
      v26 = icmp le v25, v8
            br v26 block1(v6, v25, v8, v9, v25), block2   ; v25 fed to two columns
```

## Where the hot loop bleeds

The per-iteration path is `0x40`→`0xc5` plus the back-edge at `0xed`. Each source of
waste is attributed to the layer that owns it.

### 1. Boolean materialization — isel, biggest single win (~10–12 insns/iter)

The IR keeps comparisons as SSA boolean *values* that a separate `br`/`guard`
consumes:

```
v18 = icmp ne v5, v17
     guard.cond v18
```

isel lowers `ICmp`→`ICmpSet` (`cmp; setcc; movzx`, `x64.rs:590`) and
`GuardCond`→`GuardNz` (`test; jcc`, `x64.rs:642`) *independently*, so every compare
that feeds only a branch becomes the five-instruction chain seen four times:

```
40: mov eax,0 ; 45: cmp r10,rax ; 48: setne al ; 4b: movzx rax,al ; 4f: test rax,rax ; 52: je
```

…when it should be `test r10,r10; je`. The backend *already has* the fused form —
`MOp::GuardCmp` emits `cmp; jcc` directly (`x64.rs:622`) — but only
`GuardType`/`GuardShape` use it.

**Fix (isel peephole):** when an `ICmp`'s sole use is a `Br` or `GuardCond`, emit a
flags-consuming branch instead of materializing the boolean. This is what LuaJIT
does at IR-gen time (comparison ops *are* guards) and what C1 does via its `if-cmp`
canonicalization. Four sites × ~3 insns each. Nothing to do with the allocator.

### 2. `cmp reg, 0` instead of `test` — isel peephole (~2 insns/iter)

`mov eax,0; cmp r10,rax` (`0x40`, `0x92`) materializes a zero into a register to
compare against. `test r10,r10` needs no register and no `mov`. Folds into #1 once
compares fuse to branches, but worth a standalone peephole.

### 3. Loop-carried `i` lives in two registers — frontend, not fixable in regalloc

The back-edge pays two copies every iteration:

```
ed: mov rbx,rax ; f0: mov r10,rax ; f3: jmp
```

This is *not* an allocator failure. `block1` has five parameters and `i` is threaded
through **two** of them (`v2` and `v5`), both fed `v25` on the back-edge. `v5` is used
for the zero-guard and the `mod` divisor; `v2` survives to the increment in `block4`.
They are **simultaneously live** at `block1` entry, so no allocator — coalescing or
not — can put them in one register.

**Fix (numeric-`for` lowering):** thread `i` through one phi column, not two. The
coalescer already in place then collapses the back-edge to a single `mov` (or zero,
if the register lines up).

### 4. Constant `1` pinned in `r11` across the whole loop — frontend/snapshot design

`r11` is set at `0x1c` and not touched again until the *cold* deopt stub at `0x143`.
It holds the loop step `1` alive around the entire hot loop purely so the guard's
frame-state snapshot can write it back. The allocator *cannot* rematerialize it:
inside the loop it's a phi parameter (`v4`/`v9`), not a constant — its def is the
phi, not `iconst`.

This is **the** LuaJIT design lesson: LuaJIT snapshots reference constants and
rematerializable values *directly*, so they're never live SSA values threaded
through the loop. Two ways to fix:

- **Frontend:** don't route deopt-only constants through phi columns; let the
  FrameState reference the `iconst` directly.
- **Backend:** teach the exit-stub emitter that a keepalive whose value is a
  constant should be `mov`-immediate'd *in the stub* rather than kept live. This is
  the one genuine allocator/backend-adjacent win here.

Note also `iconst 1` appears **twice** in the IR (`v11` for `x-1`, `v14` for the
step) — no constant CSE — which is the literal cause of the double `mov r11d,1` at
`0x10`/`0x1c`.

### 5. Two provably-dead branches per iteration — frontend range analysis

`i` ranges `2..x-1`, so `i != 0` (the idiv-zero guard, `0x45`) and `i == -1` (the
`floor_mod` special case, `0x58`) are both always-false. LuaJIT's narrowing/range
analysis kills exactly these. Out of scope for the allocator, but real per-iteration
cost.

## The cold deopt stub (`0x11a`–`0x15f`)

`mov r12d,2; mov byte[r8+off],r12b` repeated six times. Two isel wins:
**store-immediate** (`mov byte[r8+off], 2` — no register; x86 has `mov r/m8, imm8`)
and not re-materializing the tag each time. Cold path, so runtime-irrelevant, but
it's roughly half the function's code size.

## What's reasonable, ranked by value/effort

| # | Change | Owner | Hot-path win | Effort |
|---|--------|-------|-------------|--------|
| 1 | ~~Fuse `ICmp`→`Br`/`GuardCond` into flags-consuming branches~~ **DONE** | isel | ~3 triples/iter realized | low — `GuardCmp` path existed |
| 2 | Single phi column for the induction variable | frontend `for` lowering | 1 back-edge move/iter | low–med |
| 3 | Deopt keepalives reference constants directly (or remat in-stub) | frontend snapshot / backend stub | frees a register across the loop | med |
| 4 | `cmp x,0`→`test`, `lea`/`inc` for `i+1`, store-immediate | isel peepholes | ~2 insns/iter + stub size | low |
| 5 | Range analysis to drop dead guard + `-1` special case | frontend | 2 branches/iter | high |

## Bottom line

The allocator itself is basically done. The highest-value, lowest-risk next move is
**#1 (isel compare/branch fusion)**: the machinery already exists in
`GuardCmp`/`jcc`, so it's mostly a single-use check in isel plus a fused two-way
branch `MOp`. #2 and #3 are the two frontend changes that make the induction
variable and the loop-invariant constant stop being problems the allocator is asked
to paper over — which is precisely the gap between this output and LuaJIT/C1.

## Post-fusion disassembly (after #1)

The three per-iteration boolean chains are gone; each compare now lowers to
`cmp; jcc` (a `br.cmp`) or `cmp; je` to the exit (a fused `guard.cond`). The
`mov $0, eax` that still precedes each compare-against-zero is the imm-fold
(#2/#4), deliberately deferred. Function is 319 bytes vs 361 before.

```asm
   0: push   %rbx
   1: push   %rbp
   2: push   %r12
   4: mov    %rsi, %r8
   7: mov    (%r8), %r9
   a: mov    $2, %r10d
  10: mov    $1, %r11d
  16: mov    %r9, %rax
  19: sub    %r11, %rax            ; rax = x - 1
  1c: mov    $1, %r11d             ; step (still kept live for deopt — #3)
  22: cmp    %rax, %r10            ; fused: 2 <= x-1 ?
  25: jle    0x30
  2b: jmp    0xa1                  ; empty loop -> return true
; --- loop header ---
  30: mov    %r10, %rbx
  33: mov    %rax, %rbp
  36: mov    $0, %eax              ; imm-fold target (#2/#4)
  3b: cmp    %rax, %r10            ; fused guard: i != 0 ?
  3e: je     0xf0                  ; -> deopt
  44: cmp    $-1, %r10             ; -1 special case (#5, still present)
  48: jne    0x58
  4e: mov    $0, %eax
  53: jmp    0x7b
  58: mov    %r9, %rax
  5b: cqto
  5d: idiv   %r10
  60: mov    %rdx, %rax
  63: test   %rdx, %rdx
  66: je     0x7b
  6c: mov    %r9, %rcx
  6f: xor    %r10, %rcx
  72: jns    0x7b
  78: add    %r10, %rax
  7b: mov    %rax, %r10            ; r10 = x % i
  7e: mov    $0, %eax
  83: cmp    %rax, %r10            ; fused: x % i == 0 ?
  86: je     0xce                  ; -> return false
  8c: mov    $1, %r10d
  92: mov    %rbx, %rax
  95: add    %r10, %rax            ; i + 1
  98: cmp    %rbp, %rax            ; fused: (i+1) <= (x-1) ?
  9b: jle    0xc3                  ; more iterations
; --- return true ---
  a1: mov    $1, %r9d
  a7: mov    %r9, (%r8)
  aa: mov    $1, %r10d
  b0: mov    %r10b, 8(%r8)
  b4: movabs $0x100000001, %rax
  be: jmp    0x13a
; --- back-edge ---
  c3: mov    %rax, %rbx
  c6: mov    %rax, %r10
  c9: jmp    0x36
; --- return false ---
  ce: mov    $0, %r9d
  d4: mov    %r9, (%r8)
  d7: mov    $1, %r10d
  dd: mov    %r10b, 8(%r8)
  e1: movabs $0x100000001, %rax
  eb: jmp    0x13a
; --- deopt stub (unchanged; #4 store-imm still applies) ---
  f0: mov    %r9, (%r8)
  f3: mov    $2, %r12d
  f9: mov    %r12b, 8(%r8)
 ... (six tag stores) ...
 135: mov    $0, %eax
; --- epilogue ---
 13a: pop    %r12
 13c: pop    %rbp
 13d: pop    %rbx
 13e: ret
```

## Reference: disassembly analyzed (baseline, before #1)

```asm
; --- prologue ---
  0:  push   rbx
  1:  push   rbp
  2:  push   r12
  4:  mov    r8, rsi               ; r8 = frame base
  7:  mov    r9, [r8]              ; r9 = x
  a:  mov    r10d, 0x2             ; i = 2
 10:  mov    r11d, 0x1
 16:  mov    rax, r9
 19:  sub    rax, r11              ; rax = x - 1  (loop limit)
 1c:  mov    r11d, 0x1             ; second const 1 (step; kept live for deopt)
 22:  cmp    r10, rax              ; i <= x-1 ?
 25:  setle  cl
 28:  movzx  rcx, cl
 2c:  test   rcx, rcx
 2f:  jne    0x3a
 35:  jmp    0xcb                  ; empty loop -> return true
; --- loop preheader ---
 3a:  mov    rbx, r10              ; rbx = i
 3d:  mov    rbp, rax              ; rbp = x-1
; --- loop header (back-edge target) ---
 40:  mov    eax, 0x0
 45:  cmp    r10, rax              ; i != 0 ? (idiv-by-zero guard)
 48:  setne  al
 4b:  movzx  rax, al
 4f:  test   rax, rax
 52:  je     0x11a                 ; i == 0 -> deopt
; --- x % i (inline floor_mod) ---
 58:  cmp    r10, -1               ; i == -1 special case
 5c:  jne    0x6c
 62:  mov    eax, 0x0
 67:  jmp    0x8f
 6c:  mov    rax, r9
 6f:  cqo
 71:  idiv   r10                   ; rax = x/i, rdx = x%i (trunc)
 74:  mov    rax, rdx
 77:  test   rdx, rdx
 7a:  je     0x8f
 80:  mov    rcx, r9
 83:  xor    rcx, r10
 86:  jns    0x8f
 8c:  add    rax, r10              ; correct trunc rem to floor rem
; --- if x % i == 0: return false ---
 8f:  mov    r10, rax              ; r10 = x % i
 92:  mov    eax, 0x0
 97:  cmp    r10, rax
 9a:  sete   r10b
 9e:  movzx  r10, r10b
 a2:  test   r10, r10
 a5:  jne    0xf8                  ; mod == 0 -> return false
; --- i++; i <= x-1 ? ---
 ab:  mov    r10d, 0x1
 b1:  mov    rax, rbx
 b4:  add    rax, r10             ; rax = i+1
 b7:  cmp    rax, rbp             ; (i+1) <= (x-1) ?
 ba:  setle  r10b
 be:  movzx  r10, r10b
 c2:  test   r10, r10
 c5:  jne    0xed                 ; more iterations -> loop
; --- return true (0xcb) ---
 cb:  mov    r9d, 0x1
 d1:  mov    [r8], r9
 d4:  mov    r10d, 0x1
 da:  mov    byte [r8+0x8], r10b
 de:  movabs rax, 0x100000001     ; Status::Return(nret=1)
 e8:  jmp    0x164
; --- back-edge (0xed) ---
 ed:  mov    rbx, rax             ; rbx = i (new)
 f0:  mov    r10, rax             ; r10 = i (new)
 f3:  jmp    0x40
; --- return false (0xf8) ---
 f8:  mov    r9d, 0x0
 fe:  mov    [r8], r9
101:  mov    r10d, 0x1
107:  mov    byte [r8+0x8], r10b
10b:  movabs rax, 0x100000001
115:  jmp    0x164
; --- deopt exit stub: i == 0 guard failed (0x11a) ---
11a:  mov    [r8], r9             ; slot0 = x
11d:  mov    r12d, 0x2
123:  mov    byte [r8+0x8], r12b
127:  mov    [r8+0x10], rbx       ; slot1 = i
12b:  mov    r12d, 0x2
131:  mov    byte [r8+0x18], r12b
135:  mov    [r8+0x20], rbp       ; slot2 = x-1
139:  mov    r12d, 0x2
13f:  mov    byte [r8+0x28], r12b
143:  mov    [r8+0x30], r11       ; slot3 = 1 (step)
147:  mov    r12d, 0x2
14d:  mov    byte [r8+0x38], r12b
151:  mov    [r8+0x40], r10       ; slot4 = i (current)
155:  mov    r12d, 0x2
15b:  mov    byte [r8+0x48], r12b
15f:  mov    eax, 0x0             ; Status::Deopt(exit_id=0)
; --- epilogue (0x164) ---
164:  pop    r12
166:  pop    rbp
167:  pop    rbx
168:  ret
```
