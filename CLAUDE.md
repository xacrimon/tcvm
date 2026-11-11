# CLAUDE.md

TCVM is a Lua 5.5 implementation in Rust: parser, compiler, direct-threaded interpreter, tracing GC, and a method JIT for aarch64 and x86-64 (System V, x86-64-v3). The goal is the best Lua implementation possible, not a luac clone: match observable semantics, not luac's bytecode or register layout.

## Verification
- Never claim Lua or luac behavior without running it; show the command and output. Reference `lua`/`luac` 5.5.1 and `luajit` are on PATH; the manual is `lua_manual.of`.
- Differential runs: `lua x.lua` vs `target/release/tcvm-cli -f x.lua`, `luac -l -p x.lua` vs `tcvm-cli -l -f x.lua`. `tools/difflua.sh` does the same with every JIT region compiled at its first count.
- macOS has no `timeout`: cap runs with `perl -e 'alarm 20; exec @ARGV' …`, and confirm a check produced output before reading silence as success.
- Tests never shell out to `lua`. Derive expected output locally, then hardcode it in assertions or insta snapshots.
- x86-64 runs in Docker (`--platform linux/amd64`, the image `rust:1-bookworm`).
- Some differences from PUC Lua are deliberate (yielding across natives and message handlers, main-thread yield to the host, `%.14g` float tostring, buffered non-flushing `print`, text-only chunks, 0x7fffff00 string cap, `global` always reserved, PCG `math.random`). Don't report or "fix" them.

## Build and test
```sh
cargo build --release --workspace   # plain `cargo build` does not rebuild cli/, so tcvm-cli goes stale
cargo build --release --workspace --features tcvm/tcvm-dev   # with the development knobs below
cargo test --workspace               # tests always have `tcvm-dev`
INSTA_UPDATE=always cargo test       # accept new snapshots, then check them against `lua`
cargo clippy --workspace --all-targets --features tcvm/tcvm-dev
cargo fmt
```
The `tcvm-dev` feature holds everything that exists for development only: the IR verifier and
allocation checker, dumps, timing, the test modes and every `TCVM_*` knob but `TCVM_JIT=off`, plus
the `Lua` test hooks (`set_jit_inline`, `stop_gc_stress`, `jit_internal_errors`). A plain release
build has none of it; the tools under `tools/` and `.claude/docs/tools` need the dev build.
Workspace: `tcvm` (root), `derive/` (`#[derive(Collect)]`), `cli/` (`tcvm-cli`). Toolchain is pinned nightly (`rust-toolchain.toml`), edition 2024; nightly features are listed in `src/lib.rs`.

Tests: `tests/suite/*.rs` (runtime), `test-files/*.lua` (snapshots), `src/compiler/snapshots`, `cli/tests`. Fix a behavior bug together with a regression test. Never reach the tested path through a limitation that planned work removes; force it explicitly (e.g. `Lua::set_jit_inline(false)`).

## Layout
- `src/parser`, `src/compiler`: source → bytecode (`src/instruction.rs`; `OP_INFO` describes each opcode's family and forms).
- `src/vm`: interpreter. `extern "rust-preserve-none"` handlers dispatched with `become`; `ops/` holds handlers by family, `frame.rs` the frame layout, `native.rs`/`async_native.rs`/`ff.rs` the native protocols, `coro.rs`/`unwind.rs` the coroutine switches and unwinding.
- `src/env`: values, tables, shapes, strings, functions, threads.
- `src/dmm`: GC. `Gc<'gc, T>` with invariant lifetimes, `Collect` trait, barriers, non-moving Immix heap (`heap.rs`).
- `src/jit`: method JIT. Regions start at function entry or a hot loop, run through IR → `opt/` → `backend/` (own register allocator; targets `aarch64/` and `x86_64/`, each exporting what `backend/mod.rs` lists).
- `src/builtin`: standard library. `src/lua`: embedding API and executor.
- `tools/`: perf and differential scripts (`gate.sh`, `opcost.py`, `ljr.py`, `corpus/`).

Use rust-analyzer for anything the module docs don't answer; don't rely on this file for details.

## JIT knobs (env, `tcvm-dev` builds)
`TCVM_JIT=off` (every build), `TCVM_JIT_HOT_CALL=N`/`TCVM_JIT_HOT_LOOP=N` (1 = compile at first count), `TCVM_JIT_DEOPT_ALL=1`, `TCVM_JIT_INLINE=0`, `TCVM_JIT_DUMP=ir,opt,vcode,ra,asm`, `TCVM_JIT_LOG=1`, `TCVM_JIT_CHECK=1`, `TCVM_JIT_TIME=1`, `TCVM_JIT_ONLY=…`, `TCVM_JIT_FAR=1`, `TCVM_JIT_RA_CAPTURE=dir`, `TCVM_GC_STRESS=N`, `TCVM_HASH_SEED=N` (fixed string hash seed: `pairs` orders repeat run to run). The Bash tool runs zsh: `env $m` with several variables in `$m` sets only the first, so use `env ${=m}`.

## Working style
- Planning or brainstorming: ask clarifying questions one at a time; don't create tasks, spawn subagents, or write plan files until explicitly approved.
- Disclose every deviation from an agreed design when you make it, with its cost.
- Don't run unbounded-recursion scripts on builds without the stack limit, and never switch the user's checkout to compare builds; use a worktree.

## Code style
- Inline match expressions over small helpers for primitive operations (no `primitive_lt`-style helpers unless asked).
- Comments: only the non-obvious (invariants, why-not-the-alternative, GC/safety/ordering constraints). One-line contracts; no narration of code or generated asm, no restated signatures. Match the density of neighboring comments.
- When basing code on an upstream implementation, vendor it verbatim and change only what the design requires.
- Skip confirmations for routine edits once a refactor is approved.

## Git
- No Claude attribution in commits or PRs (no `Co-Authored-By`, no footers).
- `main` is linear with signed commits. Land PRs by fast-forward (`git push origin <sha>:refs/heads/main`), not GitHub's rebase merge, which strips signatures. Rebase onto `origin/main`, not local `main`.
- Label every new issue: run `gh label list` first and apply every label that fits.
