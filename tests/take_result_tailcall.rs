use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua, RuntimeError, StepResult};

fn yielder<'gc>(
    _n: NativeContext<'gc, '_>,
    _s: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Yield { then: None })
}

/// Tail-call a native that suspends. The Lua frame is popped at TAILCALL time,
/// so when the resume lands the thread terminates via `land_call_results` with
/// `Result { bottom }` where bottom is the *tailcallee's* slot, not 0.
#[test]
fn take_result_after_suspended_tailcall() {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let y = Function::new_native(ctx.mutation(), yielder as NativeFn, Box::new([]));
            let k = Value::string(LuaString::new(ctx, b"yielder"));
            ctx.globals().raw_set(ctx, k, Value::function(y));
            let chunk = ctx.load(
                "local a, b, c = 111, 222, 333\n\
             return yielder(a + b + c)",
                Some("tr"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");

    lua.try_enter(|ctx| -> Result<(), RuntimeError> {
        match ctx.fetch(&ex).step(ctx)? {
            StepResult::Yielded(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].get_integer(), Some(666));
                Ok(())
            }
            _ => panic!("expected yield"),
        }
    })
    .expect("step");

    lua.resume(&ex, (42i64,)).expect("resume");
    let got: i64 = lua
        .try_enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx))
        .expect("take_result");
    assert_eq!(
        got, 42,
        "take_result must return only the tailcall's results"
    );
}
