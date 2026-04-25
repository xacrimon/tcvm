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

    // Native identity callback used as a `print`-shaped probe: returns its
    // first integer arg unchanged so we can assert results from the VM.
    fn native_id<'gc>(
        _ctx: crate::env::NativeContext<'gc, '_>,
        mut stack: crate::env::Stack<'gc, '_>,
    ) -> Result<(), crate::env::NativeError> {
        let v = stack.get(0);
        let out = match v {
            Value::Integer(_) => v,
            _ => return Err(crate::env::NativeError::new("bad args")),
        };
        stack.replace(&[out]);
        Ok(())
    }

    fn run_returning_int(src: &str) -> i64 {
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let probe =
                    Function::new_native(ctx.mutation(), native_id as NativeFn, Box::new([]));
                let key = Value::String(LuaString::new(ctx.mutation(), b"probe"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::Function(probe));
                let chunk = ctx.load(src, Some("test"))?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load");
        lua.execute::<i64>(&ex).expect("run")
    }

    #[test]
    fn local_call_inside_native_call() {
        // Direct repro of the register-clobber bug: `probe(f(5))` where `f` is
        // a low-register local and `probe` is a global. Before the fix, the
        // inner f(5) setup overwrote probe's register.
        let n = run_returning_int(
            "local function f(n) return n + 1 end \
             return probe(f(5))",
        );
        assert_eq!(n, 6);
    }

    #[test]
    fn triple_nested_local_calls() {
        // Three locals at consecutive low registers, each calling the next.
        let n = run_returning_int(
            "local function h(x) return x + 1 end \
             local function g(x) return h(x) + 1 end \
             local function f(x) return g(x) + 1 end \
             return f(0)",
        );
        assert_eq!(n, 3);
    }

    #[test]
    fn recursive_fib() {
        // Self-recursive upvalue capture combined with the patched call setup.
        let n = run_returning_int(
            "local function fib(n) \
                if n < 2 then return n \
                else return fib(n - 1) + fib(n - 2) end \
             end \
             return fib(10)",
        );
        assert_eq!(n, 55);
    }

    #[test]
    fn call_in_table_constructor_with_local_target() {
        // Table-array entries with a low-register-local function target;
        // fix in emit_func_call_setup also protects the table register.
        let n = run_returning_int(
            "local function f() return 7 end \
             local t = {f(), f(), f()} \
             return t[1] + t[2] + t[3]",
        );
        assert_eq!(n, 21);
    }

    #[test]
    fn local_func_called_with_itself_as_arg() {
        // `f(f(2))` — both calls resolve to the same local register, so the
        // inner CALL would overwrite f if it weren't preserved first.
        let n = run_returning_int(
            "local function f(n) return n + 1 end \
             return f(f(2))",
        );
        assert_eq!(n, 4);
    }

    #[test]
    fn call_as_for_num_bound() {
        // for-num bound expression as a call to a low-register local.
        let n = run_returning_int(
            "local function lim() return 3 end \
             local s = 0 \
             for i = 1, lim() do s = s + i end \
             return s",
        );
        assert_eq!(n, 6);
    }

    #[test]
    fn upvalue_captured_after_temps() {
        // Two-level closure where the inner captures a parent local that was
        // defined AFTER heavy temp usage. The freereg refactor must keep
        // the parent's `x` register stable — `free_reg`'s nactvar guard is
        // the structural pin here.
        let n = run_returning_int(
            "local function outer() \
               local heavy = (1+2)*(3+4) \
               local x = heavy + 1 \
               return (function() return x end)() \
             end \
             return outer()",
        );
        assert_eq!(n, 22); // (1+2)*(3+4) = 21, +1 = 22
    }

    #[test]
    fn deeply_nested_calls() {
        // f(g(h(i(j(1))))) five-deep; each returns arg+1. Stresses that
        // freereg discipline gives bounded stack growth per call depth.
        let n = run_returning_int(
            "local function j(x) return x + 1 end \
             local function i(x) return j(x) + 1 end \
             local function h(x) return i(x) + 1 end \
             local function g(x) return h(x) + 1 end \
             local function f(x) return g(x) + 1 end \
             return f(1)",
        );
        assert_eq!(n, 6);
    }

    #[test]
    fn and_call_local_preservation() {
        // `a and probe(99)` — short-circuit into a call. The LHS local `a`
        // must survive; the call's result lands in the destination slot.
        let n = run_returning_int(
            "local a = 10 \
             local b = a and probe(99) \
             return a + b",
        );
        assert_eq!(n, 109);
    }

    #[test]
    fn local_decl_self_shadowing() {
        // `local x = x` must resolve the RHS `x` to the OUTER binding,
        // not to the one being defined. The new compile_decl delays
        // adjust_locals until after all initializers compile.
        let n = run_returning_int(
            "local x = 7 \
             do \
               local x = x \
               return x \
             end",
        );
        assert_eq!(n, 7);
    }

    #[test]
    fn local_decl_later_refs_earlier() {
        // Across two separate decl statements, the second must see
        // locals bound by the first.
        let n = run_returning_int(
            "local a = 3 \
             local b = a + 1 \
             return a + b",
        );
        assert_eq!(n, 7);
    }

    #[test]
    fn local_decl_multi_return_call() {
        // `local a, b = f()` with f returning 2 values places them in
        // consecutive locals starting at base.
        let n = run_returning_int(
            "local function f() return 10, 20 end \
             local a, b = f() \
             return a + b",
        );
        assert_eq!(n, 30);
    }

    #[test]
    fn short_circuit_and_preserves_local_b() {
        // Issue #3: `local x = a and b` must not clobber the RHS local `b`
        // on the falsy short-circuit edge (TESTSET dst patched into b's reg).
        let n = run_returning_int(
            "local a = false \
             local b = 5 \
             local x = a and b \
             return b",
        );
        assert_eq!(n, 5);
    }

    #[test]
    fn short_circuit_or_preserves_local_b() {
        // Issue #3: `local x = a or b` must not clobber the RHS local `b`
        // on the truthy short-circuit edge.
        let n = run_returning_int(
            "local a = 7 \
             local b = 5 \
             local x = a or b \
             return b",
        );
        assert_eq!(n, 5);
    }

    #[test]
    fn short_circuit_cmp_preserves_local_b() {
        // Issue #3: `local x = (b == 5) and b` — falsy CMP edge would emit
        // LFALSESKIP src=b's reg, clobbering the local. Result of the falsy
        // path must be the boolean `false` while `b` survives.
        let n = run_returning_int(
            "local b = 7 \
             local x = (b == 5) and b \
             return b",
        );
        assert_eq!(n, 7);
    }

    #[test]
    fn short_circuit_and_x_value_truthy_lhs() {
        // Truthy-lhs path: `local x = a and b` evaluates to `b` when `a` is
        // truthy. Sanity-check that the fresh-temp routing didn't break the
        // value-correct fall-through.
        let n = run_returning_int(
            "local a = 10 \
             local b = 5 \
             local x = a and b \
             return x",
        );
        assert_eq!(n, 5);
    }
}
