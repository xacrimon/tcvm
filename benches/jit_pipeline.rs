//! Timings for each stage of the JIT pipeline, from source text to running
//! native code, on `test-files/mix.lua`'s register-heavy `mix` function.
//!
//! The stages are timed in isolation so a regression can be attributed to the
//! phase that caused it: parse -> compile (bytecode) -> lower (IR) -> select
//! (machine IR) -> annotate (temps/constraints) -> allocate (registers) ->
//! encode (bytes) -> exec (run it).
//!
//! `mix` is chosen because it is the largest thing the JIT currently compiles
//! end to end: a numeric loop over twelve live integer accumulators, which puts
//! real pressure on lowering, selection, and the register allocator without
//! leaning on any op the backend declines.

use std::fs;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use tcvm::bench_support;
use tcvm::env::value::Value;
use tcvm::jit::backend::code::Code;
use tcvm::jit::backend::isel::select;
use tcvm::jit::backend::regalloc::allocate;
use tcvm::jit::backend::target::{Status, annotate, encode, machine_env};
use tcvm::jit::frontend::lower::lower;
use tcvm::jit::ir::ty::{Rep, Ty, TypeSet};
use tcvm::Lua;

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

/// The compiled region's entry: `(thread, frame base) -> status word`. `mix`
/// neither calls nor collects, so the thread pointer is unused.
type Region = extern "C" fn(*mut (), *mut Value<'static>) -> u64;

/// How many loop iterations the `exec` benchmark drives `mix` through. Big
/// enough that the steady-state loop, not the one-time entry, dominates.
const EXEC_N: i64 = 2000;

fn jit_pipeline(c: &mut Criterion) {
    let source = fs::read_to_string("test-files/mix.lua").expect("read mix.lua");
    let mut group = c.benchmark_group("jit_pipeline");

    // Stage 1 — parse. No arena needed, so it stands alone.
    group.bench_function("parse", |b| {
        b.iter(|| black_box(bench_support::parse(black_box(&source))));
    });

    // Stage 2 — compile (bytecode generation). It allocates in the arena, so each
    // iteration runs in its own `enter` and the garbage is swept periodically to
    // keep the working set bounded.
    {
        let mut lua = Lua::new();
        lua.load_all();
        let parsed = bench_support::parse(&source);
        let mut since_collect = 0u32;
        group.bench_function("compile", |b| {
            b.iter(|| {
                lua.enter(|ctx| {
                    let proto = bench_support::compile(ctx, black_box(&parsed)).expect("compile");
                    black_box(proto);
                });
                since_collect += 1;
                if since_collect == 512 {
                    lua.collect_all();
                    since_collect = 0;
                }
            });
        });
    }

    // Stages 3-7 all operate on the `mix` prototype. None of `lower`, `select`,
    // `allocate`, or `encode` touches the arena, so they share one `enter` with no
    // growth — and each takes the previous stage's output, precomputed once here.
    let mut lua = Lua::new();
    lua.load_all();
    let parsed = bench_support::parse(&source);
    lua.enter(|ctx| {
        let chunk = bench_support::compile(ctx, &parsed).expect("compile");
        let mix = chunk.prototypes[0];

        let env = machine_env();
        let func = lower(mix, 0, vec![INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        annotate(&mut m);
        let ra = allocate(&m, &env).expect("regalloc");
        let words = encode(&m, &func.pool, &ra).expect("encode");
        let code = Code::from_words(&words).expect("map code");
        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];

        // Fail loudly rather than benchmark a deopt stub: confirm the region runs
        // `mix(EXEC_N)` to a native return before timing it.
        stack[0] = Value::integer(EXEC_N);
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));
        assert_eq!(status, Status::Return(1), "exec benchmark must run natively");

        // Stage 3 — lower to IR.
        group.bench_function("frontend::lower", |b| {
            b.iter(|| black_box(lower(black_box(mix), 0, vec![INT]).expect("lower")));
        });

        // Stage 4 — instruction selection.
        group.bench_function("isel::select", |b| {
            b.iter(|| black_box(select(black_box(&func)).expect("isel")));
        });

        // Stage 5a — target annotation (temps + register constraints). It mutates
        // the MFunc in place and appends on each call, so it can't be re-run on one
        // function; every iteration anneals a fresh `select` output.
        group.bench_function("target::annotate", |b| {
            b.iter_batched_ref(
                || select(&func).expect("isel"),
                |m| annotate(black_box(m)),
                BatchSize::SmallInput,
            );
        });

        // Stage 5b — register allocation. `allocate` reads `m` without mutating it,
        // so the once-annotated `m` can be reused across iterations.
        group.bench_function("regalloc::allocate", |b| {
            b.iter(|| black_box(allocate(black_box(&m), &env).expect("regalloc")));
        });

        // Stage 6 — encode to machine bytes.
        group.bench_function("target::encode", |b| {
            b.iter(|| black_box(encode(black_box(&m), &func.pool, &ra).expect("encode")));
        });

        // The whole backend pipeline as one unit: prototype -> machine bytes, each
        // stage feeding the next. This is the number to watch for end-to-end JIT
        // compile latency; the per-stage lines above only say where it went. All of
        // it is pure over the arena, so nothing needs sweeping between iterations.
        group.bench_function("full (lower..encode)", |b| {
            b.iter(|| {
                let func = lower(black_box(mix), 0, vec![INT]).expect("lower");
                let mut m = select(&func).expect("isel");
                annotate(&mut m);
                let ra = allocate(&m, &env).expect("regalloc");
                black_box(encode(&m, &func.pool, &ra).expect("encode"))
            });
        });

        // Stage 7 — run the native code. `mix` reads its argument from R0 and
        // writes the result back there, so only R0 is reset each iteration.
        group.bench_function("exec", |b| {
            b.iter(|| {
                stack[0] = Value::integer(EXEC_N);
                black_box(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()))
            });
        });

        // Keep the executable mapping alive until every benchmark above has run.
        drop(code);
    });

    group.finish();
}

criterion_group!(benches, jit_pipeline);
criterion_main!(benches);
