//! The first native code this JIT has ever run.
//!
//! `sum_field` compiled end to end — lowered, selected, allocated, encoded — then
//! entered directly and compared against the interpreter's answer for the same
//! inputs. The register allocator underneath is [`spill_everything`], which is
//! deliberate: it makes these tests a check on the *encoder*, so that when linear
//! scan lands, anything it breaks is a register allocation bug and nothing else.
//!
//! Entering compiled code here is a test hook, not the real thing. There is no
//! hotness counter, no OSR, and the executor does not yet know these regions
//! exist; the point is to prove the generated code is correct before any of that
//! is built on top of it.

use std::fs;

use crate::env::string::LuaString;
use crate::env::value::{Value, ValueKind};
use crate::jit::backend::aarch64::{Status, encode};
use crate::jit::backend::isel::select;
use crate::jit::backend::regalloc::spill_everything;
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
            let m = select(&func).expect("isel");
            let ra = spill_everything(&m);
            let code = encode(&m, &func.pool, &ra).expect("encode");

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
        let m = select(&func).expect("isel");
        let ra = spill_everything(&m);
        let code = encode(&m, &func.pool, &ra).expect("encode");

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
        let table = ctx.fetch(&t);
        table.raw_set(
            ctx,
            Value::string(LuaString::new(ctx, b"x")),
            Value::float(1.5),
        );

        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![TAB, INT]).expect("lower");
        let m = select(&func).expect("isel");
        let ra = spill_everything(&m);
        let code = encode(&m, &func.pool, &ra).expect("encode");

        let mut stack = vec![Value::nil(); m.max_lua_reg as usize + 1];
        stack[0] = Value::table(table);
        stack[1] = Value::integer(4);

        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

        // Exit 1 is the *type* guard on the peeled first iteration; exit 0 is the
        // shape guard ahead of it (see the MIR snapshot). Naming it is the whole
        // point of the test: writing a float through `raw_set` must not have moved
        // the table's shape, so the shape guard has to pass and the tag check has
        // to be what fails.
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
        let m = select(&func).expect("isel");
        let ra = spill_everything(&m);
        let code = encode(&m, &func.pool, &ra).expect("encode");
        for b in code.bytes() {
            print!("{b:02x}");
        }
        println!();
    });
}
