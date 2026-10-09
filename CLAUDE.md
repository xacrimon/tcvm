# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Verification Standards
- Never claim Lua semantics or reference compiler (luac) behavior without actually running it to verify. If asserting behavior, show the command and output.
- When comparing against a reference implementation, cite the actual file/line or run the binary; do not fabricate output.
Add as a new section near the top of CLAUDE.md, before tool-specific guidance.

## Planning & Brainstorming
- When the user invokes brainstorming or planning, engage in collaborative dialogue and ask clarifying questions ONE AT A TIME. Do not jump straight to creating tasks, invoking subagents, or writing plan files.
- Wait for explicit approval before transitioning from discussion to execution.
Add under a ## Code Style section (create if missing).

## Code Style
- Prefer inline match expressions over extracted helper functions for small primitive operations (e.g., comparisons). Do not introduce helpers like `primitive_lt`/`primitive_le` unless asked.
- Skip confirmations for routine sed/file edits once a refactor is approved.
- Comments: comment the non-obvious — invariants, "why this and not the obvious alternative", subtle safety/GC/ordering constraints. Do not narrate what the code plainly does, restate the function name, or annotate each step of a self-evident sequence. Prefer one tight sentence over a paragraph. Doc comments on public items are fine but should be terse.

## Git & PR Conventions
- Never add `Co-Authored-By: Claude` trailers, Claude Code footers, or any Claude attribution to commits or PRs.

## Build Commands

```bash
cargo build          # Build the project
cargo check          # Type-check without full compilation
cargo test           # Run tests (none exist yet)
cargo clippy         # Lint
cargo fmt            # Format code
```

This is a Cargo workspace with two crates: `tcvm` (main) and `tcvm-derive` (proc-macro at `derive/`).

## Toolchain

Requires **nightly Rust** (nightly-2026-04-05, pinned in `rust-toolchain.toml`). Uses these nightly features:
- `explicit_tail_calls` (`become` keyword) — core to the interpreter dispatch loop
- `rust_preserve_none_cc` — calling convention for VM opcode handlers
- `fn_align` — `#[rustc_align(32)]` on dispatch targets
- `try_trait_v2` — `?` in natives that return `NativeOut`
- `macro_metavar_expr`
- `likely_unlikely`
- `allocator_api`
- `variant_count`
- `int_format_into`
- `offset_of_enum`

Edition 2024.

## What This Is

TCVM is a Lua 5.5 bytecode virtual machine written in Rust. It interprets Lua bytecode (does not compile from source yet — the parser is incomplete/early-stage). Reference projects: [piccolo](https://github.com/kyren/piccolo), [dolang](https://github.com/bkoropoff/dolang).

## Architecture

### DMM — Garbage Collector (`src/dmm/`)

A custom tracing GC ("Dioxygen Memory Management") using invariant lifetimes (`'gc`) for safety. Key concepts:

- **`Gc<'gc, T>`** (`gc.rs`): The GC smart pointer. Copy, lifetime-bound.
- **`Arena`** (`arena.rs`): Owns all GC-managed objects. Requires a `Rootable<'a>` type parameter for the root set.
- **`Mutation<'gc>`** / **`Finalization<'gc>`** (`context.rs`): Context handles for mutating or finalizing GC objects.
- **`Collect` trait** (`collect.rs`): Types must implement this to be GC-managed. Derive it with `#[derive(Collect)]` from `tcvm-derive`.
- **Write barriers** (`barrier.rs`): Forward and backward barriers maintain GC invariants when mutating object graphs.
- **Heap** (`heap.rs`): A non-moving Immix (blocks of lines in mmap'd chunks) with sticky-mark minor collections; every collection is stop-the-world.

The derive macro (`derive/src/lib.rs`) supports attributes: `#[collect(no_drop)]`, `#[collect(require_static)]`, `#[collect(unsafe_drop)]`.

### Environment / Runtime Types (`src/env/`)

Lua runtime value types, all GC-aware:

- **`Value<'gc>`** (`value.rs`): Core enum — Nil, Boolean, Integer, Float, String, Table, Function, Thread, Userdata. Small wrapper types (Function, Thread, etc.) are Copy wrappers around `Gc` pointers.
- **`Table`** (`table/mod.rs`): Array + hash map, metamethod support.
- **`Function`** (`function.rs`): `LuaClosure` (bytecode + upvalues) and `NativeClosure`. `Prototype` holds bytecode, constants, and sub-prototypes.
- **`Thread`** (`thread.rs`): Coroutine with value stack, call frames, and open upvalues.
- **`LuaString`** (`string.rs`): Interned strings with cached hash.

### VM Interpreter (`src/vm/`)

- **Handlers** (`abi.rs`, `ops/`): Direct-threaded interpreter. Each handler is an `extern "rust-preserve-none"` function declared with `handler!` and dispatched through the runtime's tables (`dispatch.rs`) with explicit tail calls (`become`); `ops/` holds the opcode handlers by family. Debug builds assert register indices; release uses raw pointer arithmetic.
- **Frames** (`frame.rs`): Four header words below each base in the value stack; every callee returns through its header's continuation.
- **Natives** (`native.rs`, `async_native.rs`, `ff.rs`): Plain natives, continuation natives that return a `NativeOut` naming a `CONT_TABLE` continuation, async natives, and fast entries for hot builtins.
- **`coro.rs`** / **`unwind.rs`**: Coroutine switches and error unwinding, both inside dispatch.
- **`num.rs`**: Arithmetic and bitwise operation helpers.

### Instruction Set (`src/instruction.rs`)

Lua 5.5's operations as 64-bit instructions, plus forms the compiler emits (compare-and-branch, CALL by result count, fused MOVE + CALL) and forms adaptive sites rewrite themselves to (arithmetic, compares, loops, constant-key table access). `OP_INFO` describes each opcode's family and forms.

### Lua reference

A copy of the raw Lua 5.5 reference manual exists in `manual.of`; reference it to verify language specifics.
