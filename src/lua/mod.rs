//! Outer Lua API. Wraps the GC arena and exposes a small surface for
//! instantiating an interpreter, loading Lua source, and calling functions
//! from Rust.

mod context;
mod convert;
mod error;
mod executor;
mod stash;

use crate::Rootable;
use crate::builtin;
use crate::dmm::{Arena, Collect, DynamicRootSet, Mutation};
use crate::env::{Table, Thread};

pub use context::Context;
pub use convert::{FromMultiValue, FromValue, IntoMultiValue, IntoValue};
pub use error::{LoadError, RuntimeError, TypeError};
pub use executor::{Executor, ExecutorMode};
pub use stash::{
    Fetchable, Stashable, StashedExecutor, StashedFunction, StashedTable, StashedThread,
};

/// Root object of the GC arena. Holds the globals table, the main thread,
/// and the dynamic root set used to stash values across `enter` boundaries.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct State<'gc> {
    pub(crate) globals: Table<'gc>,
    pub(crate) main_thread: Thread<'gc>,
    pub(crate) roots: DynamicRootSet<'gc>,
}

/// A Lua runtime instance.
pub struct Lua {
    arena: Arena<Rootable![State<'_>]>,
}

impl Default for Lua {
    fn default() -> Self {
        Self::new()
    }
}

impl Lua {
    pub fn new() -> Self {
        let arena = Arena::<Rootable![State<'_>]>::new(|mc: &Mutation<'_>| State {
            globals: Table::new(mc),
            main_thread: Thread::new(mc),
            roots: DynamicRootSet::new(mc),
        });
        Lua { arena }
    }

    /// Run `f` inside the arena's mutation context.
    pub fn enter<F, T>(&mut self, f: F) -> T
    where
        F: for<'gc> FnOnce(Context<'gc>) -> T,
    {
        self.arena.mutate(|mc, state| f(Context::new(mc, state)))
    }

    /// `enter` variant that threads a `Result` through.
    pub fn try_enter<F, T, E>(&mut self, f: F) -> Result<T, E>
    where
        F: for<'gc> FnOnce(Context<'gc>) -> Result<T, E>,
    {
        self.enter(f)
    }

    /// Run the given executor until completion.
    ///
    /// MVP: single-shot (no fuel). Returns when the executor's thread reaches
    /// `Dead` or an error is raised.
    pub fn finish(&mut self, ex: &StashedExecutor) -> Result<(), RuntimeError> {
        self.try_enter(|ctx| {
            let executor = ctx.fetch(ex);
            executor.step(ctx)
        })
    }

    /// `finish` then take typed results from the executor.
    pub fn execute<R>(&mut self, ex: &StashedExecutor) -> Result<R, RuntimeError>
    where
        R: for<'gc> FromMultiValue<'gc>,
    {
        self.finish(ex)?;
        self.try_enter(|ctx| {
            let executor = ctx.fetch(ex);
            executor.take_result::<R>(ctx)
        })
    }

    pub fn load_all(&mut self) {
        self.enter(|ctx| {
            builtin::load_basic(ctx);
            builtin::load_coroutine(ctx);
            builtin::load_debug(ctx);
            builtin::load_io(ctx);
            builtin::load_math(ctx);
            builtin::load_os(ctx);
            builtin::load_package(ctx);
            builtin::load_string(ctx);
            builtin::load_table(ctx);
            builtin::load_utf8(ctx);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::function::{Function, NativeFn};
    use crate::env::{LuaString, Value};

    #[test]
    fn roundtrip_add_function() {
        let mut lua = Lua::new();

        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let chunk = ctx.load("function add(a,b) return a+b end", Some("test"))?;
                let executor = Executor::start(ctx, chunk, ());
                Ok(ctx.stash(executor))
            })
            .expect("load + start");
        lua.execute::<()>(&ex).expect("run top-level");

        let ex = lua
            .try_enter(|ctx| -> Result<_, RuntimeError> {
                let key = Value::String(LuaString::new(ctx.mutation(), b"add"));
                let add = ctx
                    .globals()
                    .raw_get(key)
                    .get_function()
                    .ok_or(TypeError::Mismatch {
                        expected: "function",
                        got: "nil",
                    })?;
                let executor = Executor::start(ctx, add, (2i64, 3i64));
                Ok(ctx.stash(executor))
            })
            .expect("look up + start");

        let result: i64 = lua.execute(&ex).expect("run add");
        assert_eq!(result, 5);
    }

    #[test]
    fn empty_return_unit() {
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let chunk = ctx.load("local x = 1 + 2", None)?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load");
        lua.execute::<()>(&ex).expect("run");
    }

    // Native callback used by the native_* tests below. Integer add with an
    // arity/type check so we can also exercise the NativeError path.
    fn native_add<'gc>(
        _ctx: crate::env::NativeContext<'gc, '_>,
        mut stack: crate::env::Stack<'gc, '_>,
    ) -> Result<(), crate::env::NativeError> {
        let (a, b) = (stack.get(0), stack.get(1));
        let sum = match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => Value::Integer(x + y),
            _ => return Err(crate::env::NativeError::new("bad args")),
        };
        stack.replace(&[sum]);
        Ok(())
    }

    #[test]
    fn native_call_from_lua() {
        // Lua calls a Rust-native `add(2, 3)` via CALL; expects 5.
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let add =
                    Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
                let key = Value::String(LuaString::new(ctx.mutation(), b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::Function(add));

                let chunk = ctx.load("return add(2, 3)", Some("native_call"))?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load + register");
        let result: i64 = lua.execute(&ex).expect("run");
        assert_eq!(result, 5);
    }

    #[test]
    fn tailcall_into_lua_closure() {
        // Exercise op_tailcall's Lua branch: `return outer(41)` is a
        // tail call into the Lua closure `outer`, whose body is
        // `return inner(x)` — another tail call into the Lua closure
        // `inner`. Both compile to TAILCALL.
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let chunk = ctx.load(
                    "local function inner(x) return x + 1 end \
                     local function outer(x) return inner(x) end \
                     return outer(41)",
                    Some("lua_tailcall"),
                )?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load");
        let result: i64 = lua.execute(&ex).expect("run");
        assert_eq!(result, 42);
    }

    #[test]
    fn nested_function_reads_global() {
        // Previously panicked in GETTABUP because the nested function's
        // `upvalue_desc` didn't include _ENV. With _ENV routed through
        // `resolve_or_capture`, the nested closure picks up _ENV from
        // the enclosing chunk's upvalue 0.
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let add =
                    Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
                let key = Value::String(LuaString::new(ctx.mutation(), b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::Function(add));

                let chunk = ctx.load(
                    "local function f() return add(2, 3) end return f()",
                    Some("nested_global"),
                )?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load + register");
        let result: i64 = lua.execute(&ex).expect("run");
        assert_eq!(result, 5);
    }

    #[test]
    fn doubly_nested_function_reads_global() {
        // _ENV capture must cascade: inner reads a global, so the
        // middle function captures _ENV from the main chunk and the
        // inner captures it from the middle.
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let add =
                    Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
                let key = Value::String(LuaString::new(ctx.mutation(), b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::Function(add));

                let chunk = ctx.load(
                    "local function outer() \
                       local function inner() return add(1, 2) end \
                       return inner() \
                     end \
                     return outer()",
                    Some("cascade_env"),
                )?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load + register");
        let result: i64 = lua.execute(&ex).expect("run");
        assert_eq!(result, 3);
    }

    #[test]
    fn native_entry_via_executor_start() {
        // Top-level native entry: Executor::start with a native function
        // directly, step runs the callback and take_result reads results.
        let mut lua = Lua::new();
        let ex = lua.enter(|ctx| {
            let add = Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
            ctx.stash(Executor::start(ctx, add, (10i64, 32i64)))
        });
        let result: i64 = lua.execute(&ex).expect("run native entry");
        assert_eq!(result, 42);
    }

    #[test]
    fn native_error_becomes_runtime_error() {
        // A NativeError should surface as RuntimeError::Opcode (message
        // dropped at the VM boundary by design in this MVP).
        let mut lua = Lua::new();
        let ex = lua.enter(|ctx| {
            let add = Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
            // Pass a float to trigger native_add's Err path (it requires Integer).
            ctx.stash(Executor::start(ctx, add, (1i64, 2.5f64)))
        });
        let err = lua.execute::<i64>(&ex).expect_err("should error");
        assert!(matches!(err, RuntimeError::Opcode { .. }));
    }
}
