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
use crate::env::shape::{METAMETHOD_TABLE, MetamethodBits, Shape};
use crate::env::{LuaString, Table, Thread};

use crate::env::string::Interner;
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
    /// Shared root shape for all freshly created tables. Anchors the
    /// transition tree so two tables that grow through the same key
    /// sequence converge on the same shape pointer.
    pub(crate) empty_shape: Shape<'gc>,
    /// Pre-interned metamethod-name LuaStrings paired with their
    /// `MetamethodBits`. Used by `Shape::recompute_mm_cache` (slow
    /// path) to walk a metatable in identity-equality lookups instead
    /// of allocating per-name LuaStrings every time.
    pub(crate) metamethod_names: Box<[(LuaString<'gc>, MetamethodBits)]>,
    pub(crate) globals: Table<'gc>,
    pub(crate) main_thread: Thread<'gc>,
    pub(crate) roots: DynamicRootSet<'gc>,
    pub(crate) interner: Interner<'gc>,
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
        let arena = Arena::<Rootable![State<'_>]>::new(|mc: &Mutation<'_>| {
            let empty_shape = Shape::root_empty(mc);
            let interner = Interner::new(mc);
            let metamethod_names = METAMETHOD_TABLE
                .iter()
                .map(|(name, bit)| (interner.intern(mc, name), *bit))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            State {
                empty_shape,
                metamethod_names,
                globals: Table::new_with_shape(mc, empty_shape),
                main_thread: Thread::new(mc),
                roots: DynamicRootSet::new(mc),
                interner,
            }
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
                let key = Value::string(LuaString::new(ctx, b"add"));
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
        let sum = match (a.get_integer(), b.get_integer()) {
            (Some(x), Some(y)) => Value::integer(x + y),
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
                let key = Value::string(LuaString::new(ctx, b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::function(add));

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
                let key = Value::string(LuaString::new(ctx, b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::function(add));

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
                let key = Value::string(LuaString::new(ctx, b"add"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::function(add));

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
        let out = if v.get_integer().is_some() {
            v
        } else {
            return Err(crate::env::NativeError::new("bad args"));
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
                let key = Value::string(LuaString::new(ctx, b"probe"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::function(probe));
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

    #[test]
    fn shape_transition_basic() {
        // Set two string-keyed properties on a fresh table, read them back.
        // Exercises shape::transition_add_prop + property slot routing.
        let n = run_returning_int(
            "local t = {} \
             t.x = 10 \
             t.y = 32 \
             return t.x + t.y",
        );
        assert_eq!(n, 42);
    }

    #[test]
    fn shape_sharing_across_tables() {
        // Two tables grown through the same key sequence must share a shape
        // (otherwise IC monomorphism in Phase 4 will fail). Observable here
        // only as identical functional behaviour, but the shape pointer
        // identity is checked indirectly: if sharing failed, no IC would
        // help us in Phase 4. This locks in correctness of the read paths.
        let n = run_returning_int(
            "local function pt() local t = {} t.x = 3 t.y = 4 return t end \
             local a = pt() \
             local b = pt() \
             return a.x + a.y + b.x + b.y",
        );
        assert_eq!(n, 14);
    }

    #[test]
    fn shape_overwrite_existing_slot() {
        // Re-assigning an existing string key updates the slot in place;
        // no new transition.
        let n = run_returning_int(
            "local t = {} \
             t.k = 1 \
             t.k = 5 \
             t.k = 9 \
             return t.k",
        );
        assert_eq!(n, 9);
    }

    #[test]
    fn array_part_unchanged_by_shape() {
        // Integer-keyed inserts go through the array part, NOT the shape.
        // Confirms the routing split between string keys and array keys.
        let n = run_returning_int(
            "local t = {} \
             t[1] = 10 \
             t[2] = 20 \
             t[3] = 30 \
             return t[1] + t[2] + t[3]",
        );
        assert_eq!(n, 60);
    }

    fn run_returning_int_with_basic(src: &str) -> i64 {
        // Same as run_returning_int but loads the basic builtins
        // (setmetatable, rawget, rawset, ...) into _G first.
        let mut lua = Lua::new();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                builtin::load_basic(ctx);
                let probe =
                    Function::new_native(ctx.mutation(), native_id as NativeFn, Box::new([]));
                let key = Value::string(LuaString::new(ctx, b"probe"));
                ctx.globals()
                    .raw_set(ctx.mutation(), key, Value::function(probe));
                let chunk = ctx.load(src, Some("test"))?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load");
        lua.execute::<i64>(&ex).expect("run")
    }

    #[test]
    fn metamethod_index_table_fallback() {
        // __index as a table: missing key on `t` falls back to `proto`.
        let n = run_returning_int_with_basic(
            "local proto = {} \
             proto.fallback = 42 \
             local t = {} \
             setmetatable(t, {__index = proto}) \
             return t.fallback",
        );
        assert_eq!(n, 42);
    }

    #[test]
    fn metamethod_index_function_fallback() {
        // __index as a function: missing key calls __index(t, k).
        let n = run_returning_int_with_basic(
            "local mt = {__index = function(_t, _k) return 7 end} \
             local t = setmetatable({}, mt) \
             return t.anything",
        );
        assert_eq!(n, 7);
    }

    #[test]
    fn metamethod_existing_key_skips_index() {
        // Present key skips __index entirely.
        let n = run_returning_int_with_basic(
            "local mt = {__index = function() error('should not fire') end} \
             local t = setmetatable({}, mt) \
             t.x = 99 \
             return t.x",
        );
        assert_eq!(n, 99);
    }

    #[test]
    fn metamethod_newindex_function_intercept() {
        // __newindex as a function: writing a fresh key calls
        // __newindex(t, k, v) instead of writing on the table.
        let n = run_returning_int_with_basic(
            "local stored \
             local mt = {__newindex = function(_t, _k, v) stored = v end} \
             local t = setmetatable({}, mt) \
             t.fresh = 17 \
             return stored",
        );
        assert_eq!(n, 17);
    }

    #[test]
    fn metamethod_newindex_existing_key_skips() {
        // __newindex doesn't fire when the key already has a value.
        let n = run_returning_int_with_basic(
            "local mt = {__newindex = function() error('should not fire') end} \
             local t = setmetatable({}, mt) \
             rawset(t, 'x', 5) \
             t.x = 13 \
             return t.x",
        );
        assert_eq!(n, 13);
    }

    #[test]
    fn metamethod_mt_mutation_invalidates_cache() {
        // After setmetatable, install __index lazily on the metatable,
        // confirming that the MtToken generation bump triggers a cache
        // refresh on the next access. This exercises Phase 2's
        // generation-bump-on-metamethod-named-write path.
        let n = run_returning_int_with_basic(
            "local mt = {} \
             local t = setmetatable({}, mt) \
             mt.__index = function(_t, _k) return 99 end \
             return t.missing",
        );
        assert_eq!(n, 99);
    }

    #[test]
    fn ic_hot_loop() {
        // Repeated reads of the same string-keyed property in a tight
        // loop. Phase 4: after the first slow-path miss, the IC fills
        // and subsequent iterations take the fast path. Functional
        // result is independent of hit/miss; this confirms correctness
        // under sustained IC use.
        let n = run_returning_int(
            "local t = {} \
             t.x = 1 \
             local s = 0 \
             for i = 1, 1000 do s = s + t.x end \
             return s",
        );
        assert_eq!(n, 1000);
    }

    #[test]
    fn ic_set_hot_loop() {
        // Repeated writes to the same string-keyed property in a tight
        // loop. After the first slow-path miss, the SET IC hits and
        // the property is updated in place via direct slot access.
        let n = run_returning_int(
            "local t = {} \
             t.counter = 0 \
             for i = 1, 500 do t.counter = t.counter + 1 end \
             return t.counter",
        );
        assert_eq!(n, 500);
    }

    #[test]
    fn dict_mode_after_string_key_deletion() {
        // `t.x = nil` for an existing slot triggers dict-mode
        // migration. The table still works correctly afterward.
        let n = run_returning_int_with_basic(
            "local t = {} \
             t.a = 1 \
             t.b = 2 \
             t.c = 3 \
             t.b = nil \
             rawset(t, 'b', 20) \
             return t.a + t.b + t.c",
        );
        assert_eq!(n, 24);
    }

    #[test]
    fn dict_mode_preserves_metatable_chain() {
        // Even after dict migration, __index lookups on the metatable
        // still work — the dict sentinel keeps `mt_token`, so
        // `mm_cache` lookup paths keep functioning.
        let n = run_returning_int_with_basic(
            "local mt = {__index = function(_, _) return 99 end} \
             local t = setmetatable({}, mt) \
             t.a = 1 \
             t.b = 2 \
             t.a = nil \
             return t.missing",
        );
        assert_eq!(n, 99);
    }

    #[test]
    fn dict_mode_via_slot_cap() {
        // Adding more than MAX_PROPERTIES_FAST string keys forces
        // dict-mode migration. Subsequent reads still resolve.
        // MAX_PROPERTIES_FAST is 64; we go past that.
        let mut prog = String::from("local t = {}\n");
        for i in 0..70 {
            prog.push_str(&format!("t.k{} = {}\n", i, i));
        }
        prog.push_str("return t.k0 + t.k60 + t.k69");
        let n = run_returning_int(&prog);
        assert_eq!(n, 0 + 60 + 69);
    }

    #[test]
    fn ic_polymorphic_thrashes_correctly() {
        // Two tables with different shapes hit the same SETFIELD/
        // GETFIELD instruction. Monomorphic IC will thrash but stay
        // correct — values must not leak between tables.
        let n = run_returning_int(
            "local function box(v) local t = {} t.v = v return t end \
             local function read(t) return t.v end \
             local a = box(3) \
             local b = box(4) \
             local c = box(5) \
             return read(a) + read(b) + read(c)",
        );
        assert_eq!(n, 12);
    }

    #[test]
    fn shape_pointer_sharing() {
        // Two empty tables share the empty shape; after identical key
        // sequences they end up at the same shape pointer (transition
        // tree dedup).
        use crate::env::shape::Shape;
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let a = Table::new(ctx);
            let b = Table::new(ctx);
            assert!(
                Shape::ptr_eq(a.shape(), b.shape()),
                "fresh tables should share the empty shape"
            );

            let kx = Value::string(LuaString::new(ctx, b"x"));
            let ky = Value::string(LuaString::new(ctx, b"y"));

            a.raw_set(ctx.mutation(), kx, Value::integer(1));
            a.raw_set(ctx.mutation(), ky, Value::integer(2));
            b.raw_set(ctx.mutation(), kx, Value::integer(10));
            b.raw_set(ctx.mutation(), ky, Value::integer(20));

            assert!(
                Shape::ptr_eq(a.shape(), b.shape()),
                "tables with the same key sequence should share a shape"
            );

            // Different ordering -> different shape pointer.
            let c = Table::new(ctx);
            c.raw_set(ctx.mutation(), ky, Value::integer(2));
            c.raw_set(ctx.mutation(), kx, Value::integer(1));
            assert!(
                !Shape::ptr_eq(a.shape(), c.shape()),
                "tables grown through different key orders should have distinct shapes"
            );

            // Read-back consistency.
            assert_eq!(a.raw_get(kx).get_integer(), Some(1));
            assert_eq!(b.raw_get(ky).get_integer(), Some(20));
            assert_eq!(c.raw_get(kx).get_integer(), Some(1));
        });
    }

    #[test]
    fn mixed_string_and_int_keys() {
        // String keys on the shape, integer keys in the array part —
        // they must not collide.
        let n = run_returning_int(
            "local t = {} \
             t.x = 100 \
             t[1] = 7 \
             t.y = 200 \
             t[2] = 3 \
             return t.x + t.y + t[1] + t[2]",
        );
        assert_eq!(n, 310);
    }
}
