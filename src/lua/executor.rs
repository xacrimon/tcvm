use crate::dmm::{Collect, Gc, RefLock};
use crate::env::function::Function;
use crate::env::thread::{CallFrame, ThreadStatus};
use crate::env::{Thread, Value};
use crate::lua::RuntimeError;
use crate::lua::context::Context;
use crate::lua::convert::{FromMultiValue, IntoMultiValue};
use crate::vm;

#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ExecutorMode {
    /// Thread has no seeded call — cannot step.
    Stopped,
    /// Thread is running and can be stepped.
    Normal,
    /// Thread has returned; results are available on its stack.
    Result,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ExecutorInner<'gc> {
    pub(crate) thread: Thread<'gc>,
    pub(crate) mode: ExecutorMode,
}

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Executor<'gc>(Gc<'gc, RefLock<ExecutorInner<'gc>>>);

impl<'gc> Executor<'gc> {
    pub(crate) fn inner(self) -> Gc<'gc, RefLock<ExecutorInner<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<ExecutorInner<'gc>>>) -> Self {
        Executor(g)
    }

    pub fn mode(self) -> ExecutorMode {
        self.0.borrow().mode
    }

    /// Seed `main_thread` with `function(args...)` and return a Normal-mode
    /// executor. Any previous state on the main thread is cleared.
    ///
    /// Accepts both Lua closures and native functions as the entry point.
    /// For Lua entry, a call frame is pushed and the body runs on `step`.
    /// For native entry, no call frame is pushed — `step` invokes the
    /// native callback directly.
    pub fn start<A: IntoMultiValue<'gc>>(
        ctx: Context<'gc>,
        function: Function<'gc>,
        args: A,
    ) -> Self {
        let thread = ctx.main_thread();
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.stack.clear();
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();

            ts.stack.push(Value::Function(function));
            args.push_into(&mut ts.stack);

            if let Some(closure) = function.as_lua() {
                let base = 1usize;
                let needed = base + closure.proto.max_stack_size as usize;
                if ts.stack.len() < needed {
                    ts.stack.resize(needed, Value::Nil);
                }

                ts.frames.push(CallFrame {
                    closure,
                    base,
                    pc: 0,
                    num_results: 0, // accept all returns
                    continuation: None,
                });
            }
            // Native entry: leave frames empty; `step` will detect this and
            // invoke the callback directly.
            ts.status = ThreadStatus::Running;
        }

        Executor(Gc::new(
            ctx.mutation(),
            RefLock::new(ExecutorInner {
                thread,
                mode: ExecutorMode::Normal,
            }),
        ))
    }

    /// Drive the underlying thread to completion.
    ///
    /// MVP: no fuel — runs until the top-level frame returns or an error is
    /// raised. On success, flips the executor's mode to `Result`.
    pub fn step(self, ctx: Context<'gc>) -> Result<(), RuntimeError> {
        let thread = {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Normal {
                return Err(RuntimeError::BadMode);
            }
            inner.thread
        };

        let mc = ctx.mutation();
        let native_entry = thread.borrow().frames.is_empty();

        if native_entry {
            let mut ts = thread.borrow_mut(mc);
            let entry_fn = ts.stack[0]
                .get_function()
                .expect("native entry: stack[0] must be a Function");
            let nc = entry_fn
                .as_native()
                .expect("native entry: function must be native");
            let argc = ts.stack.len() - 1;
            let retc = vm::interp::invoke_native(mc, &mut *ts, nc, 1, argc)
                .map_err(|_| RuntimeError::Opcode { pc: 0 })?;
            // Move results from stack[1..1+retc] down to stack[0..retc].
            for i in 0..retc {
                ts.stack[i] = ts.stack[1 + i];
            }
            ts.stack.truncate(retc);
            ts.status = ThreadStatus::Dead;
        } else {
            vm::interp::run_thread(mc, thread).map_err(|e| RuntimeError::Opcode { pc: e.pc })?;
        }

        let mut inner = self.0.borrow_mut(mc);
        inner.mode = ExecutorMode::Result;
        Ok(())
    }

    /// Extract typed results from the finished thread's stack and reset the
    /// executor to `Stopped`.
    pub fn take_result<R: FromMultiValue<'gc>>(self, ctx: Context<'gc>) -> Result<R, RuntimeError> {
        let (thread, mode) = {
            let inner = self.0.borrow();
            (inner.thread, inner.mode)
        };
        if mode != ExecutorMode::Result {
            return Err(RuntimeError::BadMode);
        }

        let values: Vec<Value<'gc>> = {
            let ts = thread.borrow();
            ts.stack.clone()
        };
        let result = R::from_multi_value(&values).map_err(RuntimeError::from);

        // Clear the thread so it can be reused.
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.stack.clear();
            ts.frames.clear();
            ts.open_upvalues.clear();
            ts.tbc_slots.clear();
            ts.status = ThreadStatus::Suspended;
        }
        {
            let mut inner = self.0.borrow_mut(ctx.mutation());
            inner.mode = ExecutorMode::Stopped;
        }

        result
    }
}
