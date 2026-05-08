//! Host-side resume round-trip: a Lua chunk yields integers to the host,
//! the host inspects them via `Executor::step`, then calls `Lua::resume`
//! with new values and the chunk uses them to compute its return value.

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua, RuntimeError, StepResult};

fn yielder<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    // Whatever args were passed in are already on the stack — the yield
    // forwards them to the host as the yielded values.
    Ok(CallbackAction::Yield { then: None })
}

/// Yield (1, 2, 3) to the host; on resume, return `a + b` of the
/// resume-args. Verifies (a) host sees the yielded values, (b) resume
/// args reach Lua.
#[test]
fn host_resume_round_trip() {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let y = Function::new_native(ctx.mutation(), yielder as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"yielder"));
            ctx.globals().raw_set(ctx, key, Value::function(y));
            let chunk = ctx.load(
                "local a, b = yielder(1, 2, 3)\n\
                 return a + b",
                Some("host_resume"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");

    // Drive until the chunk yields. Inspect the yielded values inside
    // an enter-closure (they're 'gc and can't cross the boundary).
    let yielded_sum = lua.try_enter(|ctx| -> Result<i64, RuntimeError> {
        let executor = ctx.fetch(&ex);
        match executor.step(ctx)? {
            StepResult::Yielded(values) => {
                assert_eq!(values.len(), 3, "expected 3 yielded values");
                let mut sum = 0i64;
                for v in &values {
                    sum += v.get_integer().expect("yielded value is an integer");
                }
                Ok(sum)
            }
            StepResult::Done => panic!("expected Yielded, got Done"),
            StepResult::Pending => panic!("expected Yielded, got Pending"),
        }
    });
    assert_eq!(yielded_sum.unwrap(), 6, "yielded values should be (1,2,3)");

    // Feed the host's chosen values back. Chunk computes 10 + 20 = 30.
    lua.resume(&ex, (10i64, 20i64)).expect("resume");
    let result: i64 = lua
        .try_enter(|ctx| {
            let executor = ctx.fetch(&ex);
            executor.take_result::<i64>(ctx)
        })
        .expect("take_result");
    assert_eq!(result, 30, "chunk should sum the resume-args");
}
