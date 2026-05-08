//! A native callback uses `CallbackAction::Call { function, then }` to call
//! a Lua function and consume its results via a follow-up sequence.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

/// A trivial sequence that, on poll, takes the result at slot 0 and adds 1.
struct AddOneSequence;

unsafe impl<'gc> Collect<'gc> for AddOneSequence {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for AddOneSequence {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let v = stack.get(0).get_integer().unwrap_or(0);
        stack.replace(&[Value::integer(v + 1)]);
        Ok(SequencePoll::Return)
    }
}

/// Native callback `bumper(f)` calls `f()` then adds 1 to the result.
fn bumper<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = stack
        .get(0)
        .get_function()
        .ok_or_else(|| Error::from_str(nctx.ctx, "bumper expects a function"))?;
    // The Call action's window starts at the callback's bottom (no
    // function slot inserted yet — apply_pending_action does that). So we
    // clear our window first to leave just the args (none).
    stack.replace(&[]);
    let then = BoxSequence::new(nctx.ctx.mutation(), AddOneSequence);
    Ok(CallbackAction::Call {
        function: f,
        then: Some(then),
    })
}

#[test]
fn native_calls_lua_then_post_processes() {
    let mut lua = Lua::new();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let bumper_fn = Function::new_native(ctx.mutation(), bumper as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"bumper"));
            ctx.globals().raw_set(ctx, key, Value::function(bumper_fn));
            let chunk = ctx.load(
                "local function f() return 41 end\n\
                 return bumper(f)",
                Some("native_calls_lua"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 42);
}
