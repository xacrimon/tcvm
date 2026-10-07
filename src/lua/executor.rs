use crate::dmm::{Collect, Gc, RefLock};
use crate::env::error::Exit as ExitKind;
use crate::env::thread::ThreadStatus;
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
    /// The main thread yielded to the host; values were drained on the last
    /// `step`. Feed resume args back via [`Executor::resume`] (or
    /// [`crate::Lua::resume`]) to flip the executor back to `Normal`.
    Yielded,
}

/// Outcome of a single `Executor::step` invocation.
pub enum StepResult<'gc> {
    /// Top thread reached terminal `Result` state. Caller may `take_result`.
    Done,
    /// The main thread yielded these values to the host. Feed resume args via
    /// [`Executor::resume`] (mode → `Normal`) and call `step` again.
    Yielded(Vec<Value<'gc>>),
    /// The host gets to run before stepping on: the collector is owed work,
    /// or an async native waits on the host. Mode stays `Normal`; call `step`
    /// again to keep going, once woken (see [`Executor::step_waker`]).
    Pending,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ExecutorInner<'gc> {
    /// The main thread of this executor: the entry point seeded by `start`.
    pub(crate) thread: Thread<'gc>,
    /// The thread dispatch last ran: the main thread, or a coroutine it
    /// resumed when a GC check or an async native interrupted dispatch.
    pub(crate) current: Thread<'gc>,
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

    /// Seed a fresh main thread with `function(args...)` and return a
    /// Normal-mode executor. A non-function is called through its `__call`
    /// chain; a native is entered like any other callee.
    ///
    /// The thread's stack becomes `[f, _, _, _, args...]`: the call's
    /// frame header is written when dispatch first enters it.
    pub fn start<A: IntoMultiValue<'gc>>(
        ctx: Context<'gc>,
        function: impl Into<Value<'gc>>,
        args: A,
    ) -> Self {
        let thread = Thread::new(ctx.mutation());
        {
            let mc = ctx.mutation();
            let mut buf: Vec<Value<'gc>> = Vec::new();
            args.push_into(mc, &mut buf);
            let mut ts = thread.borrow_mut(mc);
            ts.main = true;
            ts.seed(function.into());
            ts.set_window(vm::frame::HDR, buf);
        }
        Executor(Gc::new(
            ctx.mutation(),
            RefLock::new(ExecutorInner {
                thread,
                current: thread,
                mode: ExecutorMode::Normal,
            }),
        ))
    }

    /// Drive the executor until something terminal happens or values cross
    /// the host boundary.
    ///
    /// Returns:
    /// - [`StepResult::Done`] — the main thread completed; call `take_result`.
    /// - [`StepResult::Yielded(values)`] — the main thread yielded to the
    ///   host. Feed args back via [`Executor::resume`] then `step` again.
    /// - [`StepResult::Pending`] — the collector is owed work, or an async
    ///   native waits on the host. The waker of [`step_waker`](Self::step_waker)
    ///   is woken once stepping can go on, right away for the collector.
    ///   Leave the current [`Lua::enter`](crate::Lua::enter), which collects
    ///   on its way out, and call `step` again in a new one.
    pub fn step(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
        self.step_waker(ctx, std::task::Waker::noop())
    }

    /// [`step`](Self::step), with the waker an async native's future that
    /// waits on the host gets: `Pending` then means stepping again once it
    /// is woken.
    pub fn step_waker(
        self,
        ctx: Context<'gc>,
        waker: &std::task::Waker,
    ) -> Result<StepResult<'gc>, RuntimeError> {
        let prev = ctx.set_waker(waker);
        let r = self.step_inner(ctx);
        ctx.set_waker(prev);
        r
    }

    fn step_inner(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
        let (main, current) = {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Normal {
                return Err(RuntimeError::BadMode);
            }
            (inner.thread, inner.current)
        };
        let mc = ctx.mutation();
        match main.status() {
            ThreadStatus::Result { .. } => {
                self.0.borrow_mut(mc).mode = ExecutorMode::Result;
                return Ok(StepResult::Done);
            }
            ThreadStatus::Stopped => return Err(RuntimeError::BadMode),
            _ => {}
        }

        let exit = vm::dispatch::enter(ctx, current);
        // Dispatch may have ended on a coroutine `current` resumed.
        let ended_on = unsafe { (*ctx.thread_ptr()).handle() };
        self.0.borrow_mut(mc).current = ended_on;
        match exit {
            vm::Exit::Gc => {
                ctx.mutation().metrics().defer_gc_check();
                // Ready again as soon as the host has collected.
                ctx.waker().wake_by_ref();
                return Ok(StepResult::Pending);
            }
            vm::Exit::Pending => return Ok(StepResult::Pending),
            vm::Exit::End => {}
        }

        let uncaught = ended_on.borrow_mut(mc).uncaught.take();
        if let Some(err) = uncaught {
            if let ExitKind::Process(code) = err.exit_kind() {
                // No `__close` runs, now or on a later `coroutine.close`.
                // Upvalues are closed first so closures that outlive the
                // reset keep their values.
                let mut t = Some(ended_on);
                while let Some(thread) = t {
                    let mut ts = thread.borrow_mut(mc);
                    vm::ops::control::close_upvalues(mc, &mut ts, 0);
                    t = ts.resumer.take();
                    ts.reset();
                    ts.status = ThreadStatus::Stopped;
                }
                self.0.borrow_mut(mc).mode = ExecutorMode::Stopped;
                return Err(RuntimeError::Exit(code));
            }
            // Only the main thread can leave an error uncaught: a coroutine's
            // resumer always waits in dispatch.
            debug_assert!(ended_on.ptr_eq(main));
            return Err(RuntimeError::Lua(crate::lua::Stashable::stash(
                err,
                mc,
                ctx.roots(),
            )));
        }

        let ts = main.borrow();
        match ts.status {
            ThreadStatus::Result { .. } => {
                drop(ts);
                self.0.borrow_mut(mc).mode = ExecutorMode::Result;
                Ok(StepResult::Done)
            }
            ThreadStatus::Suspended => {
                let yb = ts
                    .yield_bottom
                    .expect("suspended main thread without a yield");
                let values = ts.window(yb).to_vec();
                drop(ts);
                self.0.borrow_mut(mc).mode = ExecutorMode::Yielded;
                Ok(StepResult::Yielded(values))
            }
            _ => unreachable!("dispatch ended with the main thread still running"),
        }
    }

    /// Re-arm a `Yielded` executor with fresh resume arguments: they replace
    /// the yielded values, and the next `step` delivers them to the yield's
    /// call site.
    pub fn resume<A: IntoMultiValue<'gc>>(
        self,
        ctx: Context<'gc>,
        args: A,
    ) -> Result<(), RuntimeError> {
        let mc = ctx.mutation();
        let main = {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Yielded {
                return Err(RuntimeError::BadMode);
            }
            inner.thread
        };
        let mut buf: Vec<Value<'gc>> = Vec::new();
        args.push_into(mc, &mut buf);
        {
            let mut ts = main.borrow_mut(mc);
            let yb = ts
                .yield_bottom
                .expect("Yielded mode must have a yield bottom");
            ts.set_window(yb, buf);
        }
        let mut inner = self.0.borrow_mut(mc);
        inner.current = main;
        inner.mode = ExecutorMode::Normal;
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
        // Results are the window `stack[bottom..top]` recorded by the
        // `Result` status; the vec itself may run past `top` (grow-not-shrink).
        let result = {
            let ts = thread.borrow();
            let ThreadStatus::Result { bottom } = ts.status else {
                return Err(RuntimeError::BadMode);
            };
            R::from_multi_value(ts.window(bottom)).map_err(RuntimeError::from)
        };
        // Clear the thread so it can be reused.
        {
            let mc = ctx.mutation();
            let mut ts = thread.borrow_mut(mc);
            ts.reset();
            ts.status = ThreadStatus::Stopped;
        }
        self.0.borrow_mut(ctx.mutation()).mode = ExecutorMode::Stopped;
        result
    }
}
