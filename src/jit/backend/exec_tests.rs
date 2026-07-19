//! The first native code this JIT has ever run.
//!
//! `sum_field` compiled end to end — lowered, selected, allocated, encoded — then
//! entered directly and compared against the interpreter's answer for the same
//! inputs. Any disagreement is a bug somewhere in that pipeline; the allocation is
//! independently screened by `verify` in the encoder, so a failure here points at
//! isel or the encoder rather than the allocator.
//!
//! Entering compiled code here is a test hook, not the real thing. There is no
//! hotness counter, no OSR, and the executor does not yet know these regions
//! exist; the point is to prove the generated code is correct before any of that
//! is built on top of it.

use std::fs;

use crate::env::string::LuaString;
use crate::env::value::{Value, ValueKind};
use crate::jit::backend::code::Code;
use crate::jit::backend::isel::select;
use crate::jit::backend::mach::MFunc;
use crate::jit::backend::regalloc::{Allocation, RegallocFunc, allocate as run_alloc};
use crate::jit::backend::target::machine_env;
use crate::jit::backend::target::{Status, encode};
use crate::jit::frontend::lower::lower;
use crate::jit::ir::ty::{Rep, Ty, TypeSet};
use crate::{Executor, Lua, StashedFunction, StashedTable};

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);
const TAB: Ty = Ty::new(Rep::Val, TypeSet::TAB);

/// The compiled region's entry: `(thread, frame base) -> status word`.
///
/// The thread pointer is unused by anything the region currently compiles to —
/// nothing calls, allocates, or collects — so these tests pass null and the
/// signature stays honest about what will eventually be needed.
type Region = extern "C" fn(*mut (), *mut Value<'static>) -> u64;

/// Allocate `m` with the host target's registers. The backend constrains no
/// operand and clobbers nothing, so a decline here is a bug, not a legal answer.
///
/// Correctness rests on two independent checks that need no second allocator: the
/// encoder runs `verify` on the allocation, and every test below compares native
/// output against the interpreter.
fn allocate(m: &mut MFunc) -> Allocation {
    crate::jit::backend::target::annotate(m);
    run_alloc(m, &machine_env()).expect("the backend asks for nothing the allocator declines")
}

/// Run `jit_loop_warm.lua`, which leaves `sum_field`'s inline caches warm, and
/// hand back the function.
fn warm_sum_field(lua: &mut Lua) -> StashedFunction {
    let source = fs::read_to_string("test-files/jit_loop_warm.lua").unwrap();
    let ex = lua
        .try_enter(|ctx| {
            let chunk = ctx.load(&source, Some("warm")).expect("compile");
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let v = ctx
            .fetch(&ex)
            .take_result::<Value>(ctx)
            .expect("chunk result");
        ctx.stash(v.get_function().expect("chunk returns a function"))
    })
}

/// Build a table by *running Lua*, so its shape is whatever the constructor
/// bytecode produces — which is the shape the warm inline cache recorded. Poking
/// fields in from Rust would take a different transition path and might land on a
/// different shape, and then the guard would fail for reasons that have nothing
/// to do with the code under test.
fn table_from(lua: &mut Lua, src: &str) -> StashedTable {
    let ex = lua
        .try_enter(|ctx| {
            let chunk = ctx.load(src, Some("tab")).expect("compile");
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let v = ctx
            .fetch(&ex)
            .take_result::<Value>(ctx)
            .expect("chunk result");
        ctx.stash(v.get_table().expect("chunk returns a table"))
    })
}

/// What the interpreter says `sum_field(t, n)` is.
fn interpret(lua: &mut Lua, f: &StashedFunction, t: &StashedTable, n: i64) -> i64 {
    let ex = lua
        .try_enter(|ctx| {
            let args = (Value::table(ctx.fetch(t)), Value::integer(n));
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, ctx.fetch(f), args)))
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        ctx.fetch(&ex)
            .take_result::<Value>(ctx)
            .expect("result")
            .get_integer()
            .expect("an integer")
    })
}

/// `sum_field` as native code, on a table whose shape the region was specialized
/// for: it should run to the Lua `return` without a single guard failing.
#[test]
fn native_sum_field_matches_interpreter() {
    let mut lua = Lua::new();
    lua.load_all();
    let f = warm_sum_field(&mut lua);
    let t = table_from(&mut lua, "return { x = 7 }");

    for n in [1i64, 3, 10] {
        let want = interpret(&mut lua, &f, &t, n);
        assert_eq!(want, 7 * n, "the interpreter itself disagrees");

        lua.enter(|ctx| {
            let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
            let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
            let mut m = select(&func).expect("isel");
            let ra = allocate(&mut m);
            let code =
                Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");

            // A stand-in Lua frame: `t` and `n` where the region's entry context
            // says they are, and room for every register its exits write back.
            let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
            stack[0] = Value::table(ctx.fetch(&t));
            stack[1] = Value::integer(n);

            let region: Region = unsafe { std::mem::transmute(code.entry()) };
            let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

            assert_eq!(
                status,
                Status::Return(1),
                "expected a native return, not a deopt — a guard failed"
            );
            assert_eq!(
                stack[0].get_integer(),
                Some(want),
                "native result disagrees with the interpreter for n = {n}"
            );
        });
    }
}

/// `is_prime(x)` end to end: a numeric `for` loop, `%`, an early `return false`,
/// and a `return true`. It exercises what `sum_field` does not — the floor-mod
/// expansion's allocator temps and a loop-carried block parameter — so a bug in
/// the temp mechanism or the reload phase shows up as a wrong answer here.
#[test]
fn native_is_prime_matches_interpreter() {
    let mut lua = Lua::new();
    lua.load_all();
    let source = fs::read_to_string("test-files/primes.lua").unwrap();

    for (x, want) in [
        (2i64, true),
        (3, true),
        (4, false),
        (5, true),
        (7, true),
        (9, false),
        (11, true),
        (12, false),
    ] {
        lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("primes")).expect("compile");
            let is_prime = chunk.as_lua().expect("closure").proto.prototypes[0];
            let func = lower(is_prime, 0, vec![INT]).expect("lower");
            let mut m = select(&func).expect("isel");
            let ra = allocate(&mut m);
            let code =
                Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");

            let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
            stack[0] = Value::integer(x);

            let region: Region = unsafe { std::mem::transmute(code.entry()) };
            let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

            assert_eq!(
                status,
                Status::Return(1),
                "is_prime({x}) should run natively"
            );
            assert_eq!(stack[0].get_boolean(), Some(want), "is_prime({x})");
        });
    }
}

/// `mix(n)` end to end: a numeric `for` loop over twelve simultaneously-live
/// integer accumulators mixed with add/sub/mul/mod/shift/xor. Nothing boxes,
/// guards, or calls — it is pure unboxed integer arithmetic — so this is the
/// register-allocation stress test the other cases are not: far more live values
/// than hardware registers, forcing spills and reloads. A wrong answer here
/// points at the allocator's spill/reload machinery rather than at any guard.
///
/// It is also the only case that exercises shift selection: `mix` uses `<<`/`>>`
/// with constant counts, which isel folds to a single machine shift-by-immediate.
#[test]
fn native_mix_matches_interpreter() {
    let mut lua = Lua::new();
    lua.load_all();
    let source = fs::read_to_string("test-files/mix.lua").unwrap();

    for n in [1i64, 2, 5, 13, 50, 137] {
        let want = interpret_unary(&mut lua, &source, "mix", n);

        lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("mix")).expect("compile");
            let mix = chunk.as_lua().expect("closure").proto.prototypes[0];
            let func = lower(mix, 0, vec![INT]).expect("lower");
            let mut m = select(&func).expect("isel");
            let ra = allocate(&mut m);
            let code =
                Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");

            let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
            stack[0] = Value::integer(n);

            let region: Region = unsafe { std::mem::transmute(code.entry()) };
            let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

            assert_eq!(status, Status::Return(1), "mix({n}) should run natively");
            assert_eq!(stack[0].get_integer(), Some(want), "mix({n})");
        });
    }
}

/// Shift selection's edge cases, which `mix` (only `<< 1`, `<< 2`, `>> 2`) never
/// reaches: a right shift that must zero-fill rather than sign-extend, a count of
/// exactly 64 that clears every bit, and a negative count that reverses the
/// direction. Each `f(x)` is compiled and its native answer checked against the
/// interpreter — itself verified against reference Lua in `test-files/mix.lua`.
#[test]
fn native_shift_edge_cases_match_interpreter() {
    let mut lua = Lua::new();
    lua.load_all();

    let cases = [
        "local function f(x) return x >> 2 end",  // logical: zero-fills negatives
        "local function f(x) return x >> 64 end", // |count| >= 64 -> 0
        "local function f(x) return x << 64 end", // |count| >= 64 -> 0
        "local function f(x) return x >> -3 end", // negative count reverses to << 3
        "local function f(x) return x << -3 end", // negative count reverses to >> 3
        "local function f(x) return x << 0 end",  // identity
    ];

    for src in cases {
        for x in [-100i64, -1, 0, 3, 12345, i64::MAX, i64::MIN] {
            let want = interpret_unary(&mut lua, src, "f", x);

            lua.enter(|ctx| {
                let chunk = ctx.load(src, Some("shift")).expect("compile");
                let f = chunk.as_lua().expect("closure").proto.prototypes[0];
                let func = lower(f, 0, vec![INT]).expect("lower");
                let mut m = select(&func).expect("isel");
                let ra = allocate(&mut m);
                let code = Code::from_words(&encode(&m, &func.pool, &ra).expect("encode"))
                    .expect("map code");

                let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
                stack[0] = Value::integer(x);

                let region: Region = unsafe { std::mem::transmute(code.entry()) };
                let status =
                    Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

                assert_eq!(status, Status::Return(1), "`{src}` x={x} should run natively");
                assert_eq!(stack[0].get_integer(), Some(want), "`{src}` x={x}");
            });
        }
    }
}

/// Run `src`'s first nested function (named for readability only) on a single
/// integer argument through the interpreter and return its integer result. The
/// nested prototype is wrapped in a closure with fresh nil upvalue cells — `mix`
/// captures nothing, so their contents never matter, only their count.
fn interpret_unary(lua: &mut Lua, src: &str, _name: &str, arg: i64) -> i64 {
    use crate::dmm::{Gc, RefLock};
    use crate::env::function::{Function, UpvalueState};

    let ex = lua
        .try_enter(|ctx| {
            let chunk = ctx.load(src, Some("interp")).expect("compile");
            let proto = chunk.as_lua().expect("closure").proto.prototypes[0];
            let upvalues = (0..proto.num_upvalues)
                .map(|_| Gc::new(ctx.mutation(), RefLock::new(UpvalueState::Closed(Value::nil()))))
                .collect();
            let closure = Function::new_lua(ctx.mutation(), proto, upvalues);
            Ok::<_, crate::RuntimeError>(
                ctx.stash(Executor::start(ctx, closure, (Value::integer(arg),))),
            )
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        ctx.fetch(&ex)
            .take_result::<Value>(ctx)
            .expect("result")
            .get_integer()
            .expect("an integer")
    })
}

/// The same code, a table of a different shape. Every path out of compiled code
/// that is not a `return` is a deopt, and the stub it lands in has to leave the
/// frame exactly as the interpreter would have had it at that pc — otherwise
/// resuming there is nonsense.
#[test]
fn shape_guard_deopts_with_a_resumable_frame() {
    let mut lua = Lua::new();
    lua.load_all();
    let f = warm_sum_field(&mut lua);

    // Same field, reached by a different transition, so a different shape — and
    // the guard is on the shape, not on whether `x` happens to exist.
    let other = table_from(&mut lua, "return { y = 1, x = 7 }");

    lua.enter(|ctx| {
        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        let ra = allocate(&mut m);
        let code =
            Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");

        let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
        stack[0] = Value::table(ctx.fetch(&other));
        stack[1] = Value::integer(4);

        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

        let Status::Deopt(exit) = status else {
            panic!("a foreign shape should have failed the shape guard, got {status:?}");
        };

        // The state the interpreter must find, one iteration into the loop it was
        // about to run: the loop has not executed a body yet.
        let fs = func.exit(crate::jit::ir::ExitRef(exit));
        assert_eq!(func.frame_state(fs.fs).pc, 5, "resume pc");

        assert_eq!(stack[0].kind(), ValueKind::Table, "R0: the table");
        assert_eq!(
            stack[2].get_integer(),
            Some(0),
            "R2: the accumulator, s = 0"
        );
        assert_eq!(
            stack[3].get_integer(),
            Some(1),
            "R3: the loop counter, i = 1"
        );
        assert_eq!(stack[4].get_integer(), Some(4), "R4: the limit, n");
        assert_eq!(stack[5].get_integer(), Some(1), "R5: the step");
    });
}

/// A guard on the *value's type*, not its shape: `t.x` holding a float takes the
/// same shape but fails the tag check inside the loop, which is a deopt from a
/// mid-loop state rather than from the peeled entry.
#[test]
fn type_guard_deopts_mid_loop() {
    let mut lua = Lua::new();
    lua.load_all();
    let f = warm_sum_field(&mut lua);
    let t = table_from(&mut lua, "return { x = 7 }");

    lua.enter(|ctx| {
        // Same shape — only the value in the slot changes.
        ctx.fetch(&t).raw_set(
            ctx,
            Value::string(LuaString::new(ctx, b"x")),
            Value::float(1.5),
        );
    });

    lua.enter(|ctx| {
        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        let ra = allocate(&mut m);
        let code =
            Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");

        let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
        stack[0] = Value::table(ctx.fetch(&t));
        stack[1] = Value::integer(4);

        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

        // Exit 1 is the *type* guard on the peeled first iteration; exit 0 is the
        // shape guard ahead of it (see the MIR snapshot). Naming it is the point:
        // writing a float through `raw_set` must not have moved the table's shape,
        // so the shape guard has to pass and the tag check has to be what fails.
        assert_eq!(
            status,
            Status::Deopt(1),
            "expected the integer tag guard to fail, not the shape guard"
        );
        assert_eq!(stack[2].get_integer(), Some(0), "s is still 0");
        assert_eq!(stack[3].get_integer(), Some(1), "i is still 1");
    });

    // And to be sure the table really did keep its shape, the interpreter agrees
    // this is a float sum now.
    let ex = lua
        .try_enter(|ctx| {
            let args = (Value::table(ctx.fetch(&t)), Value::integer(4));
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, ctx.fetch(&f), args)))
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    let got = lua.enter(|ctx| {
        ctx.fetch(&ex)
            .take_result::<Value>(ctx)
            .expect("result")
            .get_float()
    });
    assert_eq!(got, Some(6.0));
}

/// Not an assertion — a way to look at the code. `cargo test -- --ignored
/// --nocapture dump_native` prints the encoded bytes; pipe them through a
/// disassembler to read what the machine will actually run.
#[test]
#[ignore]
fn dump_native() {
    let mut lua = Lua::new();
    lua.load_all();
    let f = warm_sum_field(&mut lua);
    lua.enter(|ctx| {
        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        let ra = allocate(&mut m);
        let code =
            Code::from_words(&encode(&m, &func.pool, &ra).expect("encode")).expect("map code");
        eprintln!(
            "{} vregs, {} spilled; {} instructions",
            m.num_vregs(),
            ra.num_spills,
            code.bytes().len() / 4,
        );
        for b in code.bytes() {
            print!("{b:02x}");
        }
        println!();
    });
}
