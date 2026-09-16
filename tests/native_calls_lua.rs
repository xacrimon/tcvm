//! A native callback uses `CallbackAction::Call` to call a Lua function,
//! either consuming its results via a follow-up sequence or handing them
//! straight to its own caller.

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
        ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let v = stack.get(0).get_integer().unwrap_or(0);
        stack.replace(&[Value::integer(ctx.mutation(), v + 1)]);
        Ok(SequencePoll::Return)
    }
}

/// Native callback `bumper(f)` calls `f()` then adds 1 to the result.
fn bumper<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.get(0).get_function().is_none() {
        return Err(Error::from_str(nctx.ctx, "bumper expects a function"));
    }
    // The callee at stack[0] with no arguments is already `Call` layout.
    let then = BoxSequence::new(nctx.ctx.mutation(), AddOneSequence);
    Ok(CallbackAction::Call { then: Some(then) })
}

/// Native callback `forward(f, ...)`: `f(...)`'s results are the caller's.
fn forward<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Call { then: None })
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

#[test]
fn native_call_without_then_returns_to_caller() {
    let mut lua = Lua::new();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), forward as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"forward"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load(
                "local a, b, c = forward(function(x) return x, x + 1 end, 1)\n\
                 return (a == 1 and b == 2 and c == nil) and 1 or 0",
                Some("native_calls_lua"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
