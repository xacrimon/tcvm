use crate::dmm::{Collect, Gc, RefLock};
use crate::env::function::Function;
use crate::env::thread::{Frame, LuaFrame, PendingAction, ThreadStatus};
use crate::env::{Thread, Value};
use crate::lua::RuntimeError;
use crate::lua::context::Context;
use crate::lua::convert::{FromMultiValue, IntoMultiValue};
use crate::vm;
use crate::vm::sequence::CallbackAction;

#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ExecutorMode {
    /// Thread has no seeded call — cannot step.
    Stopped,
    /// Thread is running and can be stepped.
    Normal,
    /// Thread has returned; results are available on its stack.
    Result,
    /// Top thread yielded; values were drained on the last `step`. Caller
    /// may `resume` (Phase 8 public API) or treat as terminal.
    Yielded,
}

/// Outcome of a single `Executor::step` invocation.
pub enum StepResult<'gc> {
    /// Top thread reached terminal `Result` state. Caller may `take_result`.
    Done,
    /// Top thread yielded these values to the host. Caller may `resume`
    /// the executor with new arguments.
    Yielded(Vec<Value<'gc>>),
    /// Reserved for future fuel-based time-slicing — caller should call
    /// `step` again to continue. Phase 4 never produces this variant.
    Pending,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ExecutorInner<'gc> {
    /// The "main" thread for this executor — the entry point seeded by
    /// `start`. Always equals `thread_stack[0]`.
    pub(crate) thread: Thread<'gc>,
    /// Stack of currently-active threads. The top is the thread the driver
    /// is pumping; lower entries are `WaitThread`-suspended resumers.
    /// Coroutine `resume` pushes onto this stack in P7+.
    pub(crate) thread_stack: Vec<Thread<'gc>>,
    pub(crate) mode: ExecutorMode,
    /// Set when the main thread yielded to the host. Used by the driver to
    /// distinguish "main yielded" from "main is running and happens to be
    /// in Suspended status mid-pump". Cleared by `resume`.
    #[collect(require_static)]
    pub(crate) main_yielded: bool,
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

            ts.stack.push(Value::function(function));
            args.push_into(&mut ts.stack);

            if let Some(closure) = function.as_lua() {
                let base = 1usize;
                let needed = base + closure.proto.max_stack_size as usize;
                if ts.stack.len() < needed {
                    ts.stack.resize(needed, Value::nil());
                }

                ts.push_lua(LuaFrame {
                    closure,
                    base,
                    pc: 0,
                    num_results: 0, // accept all returns
                    continuation: None,
                });
            }
            // Native entry: leave frames empty; `step` will detect this and
            // invoke the callback directly. `Suspended` here means
            // "primed-and-ready"; `step` flips to `Result` on completion.
            ts.status = ThreadStatus::Suspended;
        }

        Executor(Gc::new(
            ctx.mutation(),
            RefLock::new(ExecutorInner {
                thread,
                thread_stack: vec![thread],
                mode: ExecutorMode::Normal,
                main_yielded: false,
            }),
        ))
    }

    /// Drive the executor by pumping the top frame of the top thread until
    /// the executor reaches a terminal state.
    ///
    /// Returns:
    /// - [`StepResult::Done`] — the main thread completed; call `take_result`.
    /// - [`StepResult::Yielded(values)`] — the main thread yielded to the
    ///   host. Use `resume` to continue (Phase 8 surface).
    /// - [`StepResult::Pending`] — reserved for fuel-based slicing.
    ///
    /// The hot path (Lua-only execution, sync natives) makes a single call
    /// into `run_thread` and exits. Coroutine resume / sequence pump cycles
    /// loop here until something terminal happens or values cross the host
    /// boundary. There is no fuel limit yet — see plan §P10.
    pub fn step(self, ctx: Context<'gc>) -> Result<StepResult<'gc>, RuntimeError> {
        {
            let inner = self.0.borrow();
            if inner.mode != ExecutorMode::Normal {
                return Err(RuntimeError::BadMode);
            }
        }

        let mc = ctx.mutation();

        loop {
            // Snapshot the top thread under a short borrow so we can release
            // the executor lock before re-borrowing the thread itself.
            let top = {
                let inner = self.0.borrow();
                *inner
                    .thread_stack
                    .last()
                    .expect("executor thread_stack is empty")
            };

            // (1) Inner-thread propagation. If the top thread terminated
            // (Result) or yielded (Suspended) and we're not the bottom of
            // the executor stack, hand its values off to the resumer's
            // `Frame::WaitThread` and continue.
            let stack_len = self.0.borrow().thread_stack.len();
            if stack_len > 1 {
                match top.status() {
                    ThreadStatus::Result {
                        bottom: result_bottom,
                    } => {
                        propagate_inner_to_resumer(self, ctx, top, result_bottom, false)?;
                        continue;
                    }
                    ThreadStatus::Suspended => {
                        // Inner yielded. The yielded values live at
                        // stack[bottom..] where `bottom` is the inner's
                        // pending_action.bottom (yield action stashed it).
                        let bottom = {
                            let ts = top.borrow();
                            ts.pending_action.as_ref().map(|p| p.bottom).unwrap_or(0)
                        };
                        propagate_inner_to_resumer(self, ctx, top, bottom, true)?;
                        continue;
                    }
                    _ => {}
                }
            }

            // (2) Terminal check on the main thread.
            match top.status() {
                ThreadStatus::Result { .. } => {
                    let mut inner = self.0.borrow_mut(mc);
                    inner.mode = ExecutorMode::Result;
                    return Ok(StepResult::Done);
                }
                ThreadStatus::Suspended if stack_len == 1 && self.0.borrow().main_yielded => {
                    // P8: surface as StepResult::Yielded once the yield-to-
                    // host channel is wired. For now, treat as Done.
                    let mut inner = self.0.borrow_mut(mc);
                    inner.mode = ExecutorMode::Yielded;
                    return Ok(StepResult::Yielded(Vec::new()));
                }
                ThreadStatus::Stopped => return Err(RuntimeError::BadMode),
                _ => {}
            }

            // (3) Drain a pending native-callback action from the top thread.
            let pending = {
                let mut ts = top.borrow_mut(mc);
                ts.pending_action.take()
            };
            if let Some(p) = pending {
                apply_pending_action(self, ctx, top, p)?;
                continue;
            }

            // (4) Inspect and pump the top frame.
            #[derive(Clone, Copy)]
            enum FrameKind {
                Lua,
                Sequence,
                Start,
                WaitThread,
                Error,
                Empty,
            }
            let kind = {
                let ts = top.borrow();
                match ts.frames.last() {
                    Some(Frame::Lua(_)) => FrameKind::Lua,
                    Some(Frame::Sequence { .. }) => FrameKind::Sequence,
                    Some(Frame::Start(_)) => FrameKind::Start,
                    Some(Frame::WaitThread { .. }) => FrameKind::WaitThread,
                    Some(Frame::Error(_)) => FrameKind::Error,
                    None => FrameKind::Empty,
                }
            };

            match kind {
                FrameKind::Lua => {
                    vm::interp::run_thread(ctx, top)
                        .map_err(|e| RuntimeError::Opcode { pc: e.pc })?;
                }
                FrameKind::Empty => {
                    // Native-entry shortcut: pre-P7 path; eventually folded
                    // into Frame::Start. Stack[0] = native fn, [1..] = args.
                    let mut ts = top.borrow_mut(mc);
                    let entry_fn = ts.stack[0]
                        .get_function()
                        .expect("native entry: stack[0] must be a Function");
                    let nc = entry_fn
                        .as_native()
                        .expect("native entry: function must be native");
                    let argc = ts.stack.len() - 1;
                    let action =
                        vm::interp::invoke_native(ctx, &mut *ts, nc, 1, argc).map_err(|e| {
                            RuntimeError::Lua(crate::lua::Stashable::stash(e, mc, ctx.roots()))
                        })?;
                    match action {
                        CallbackAction::Return => {
                            let retc = ts.stack.len() - 1;
                            for i in 0..retc {
                                ts.stack[i] = ts.stack[1 + i];
                            }
                            ts.stack.truncate(retc);
                            ts.status = ThreadStatus::Result { bottom: 0 };
                        }
                        _ => return Err(RuntimeError::BadMode),
                    }
                }
                FrameKind::Sequence => {
                    pump_sequence(self, ctx, top)?;
                }
                FrameKind::Start => {
                    // First-resume of a coroutine. Pop Frame::Start(f),
                    // build a one-shot call frame for `f` with the args
                    // currently on the stack as its parameters.
                    let mut ts = top.borrow_mut(mc);
                    let f = match ts.frames.pop() {
                        Some(Frame::Start(f)) => f,
                        _ => unreachable!(),
                    };
                    // Args were placed by the resumer at stack[0..].
                    // For a Lua entry: push function value at slot 0, then
                    // a LuaFrame with base=1.
                    if let Some(closure) = f.as_lua() {
                        // Re-arrange: put function at slot 0, args at [1..].
                        let argc = ts.stack.len();
                        ts.stack.insert(0, Value::function(f));
                        let _ = argc;
                        let base = 1usize;
                        let needed = base + closure.proto.max_stack_size as usize;
                        if ts.stack.len() < needed {
                            ts.stack.resize(needed, Value::nil());
                        }
                        // Nil-fill missing params.
                        let num_params = closure.proto.num_params as usize;
                        let provided = ts.stack.len() - 1; // already grew
                        let _ = provided;
                        // Actually we already grew with nils; nothing more to do.
                        let _ = num_params;
                        ts.push_lua(LuaFrame {
                            closure,
                            base,
                            pc: 0,
                            num_results: 0, // accept all returns
                            continuation: None,
                        });
                    } else {
                        // Native entry — invoke directly next iteration via
                        // the Empty branch. We re-create the empty-stack
                        // shape: stack[0] = function, [1..] = args.
                        let cur_args: Vec<Value<'gc>> = ts.stack.drain(..).collect();
                        ts.stack.push(Value::function(f));
                        ts.stack.extend(cur_args);
                    }
                    ts.status = ThreadStatus::Normal;
                }
                FrameKind::WaitThread => {
                    unreachable!("WaitThread on top of the *active* thread is invariant-violating");
                }
                FrameKind::Error => {
                    unwind_error(self, ctx, top)?;
                }
            }
        }
    }

    /// Re-arm a `Yielded` executor with fresh resume arguments. Phase 4
    /// stub: full implementation lands with coroutine support in P7/P8.
    pub fn resume<A: IntoMultiValue<'gc>>(
        self,
        ctx: Context<'gc>,
        _args: A,
    ) -> Result<(), RuntimeError> {
        let inner = self.0.borrow();
        if inner.mode != ExecutorMode::Yielded {
            return Err(RuntimeError::BadMode);
        }
        let _ = ctx;
        // Real impl: push args onto top thread's stack at the yielded
        // bottom, flip executor mode back to Normal, return Ok(()).
        Err(RuntimeError::BadMode)
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
            ts.status = ThreadStatus::Stopped;
        }
        {
            let mut inner = self.0.borrow_mut(ctx.mutation());
            inner.mode = ExecutorMode::Stopped;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Driver helpers
// ---------------------------------------------------------------------------

/// Translate a [`PendingAction`] (deposited by `op_call`/`op_tailcall` after
/// a non-`Return` `CallbackAction`) into frame-stack operations.
fn apply_pending_action<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
    p: PendingAction<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let PendingAction {
        action,
        bottom,
        func_idx,
        returns,
    } = p;
    match action {
        CallbackAction::Return => {
            // op_call/tailcall handle Return inline; should never get here.
            unreachable!("apply_pending_action: Return should be handled inline");
        }
        CallbackAction::Sequence(seq) => {
            let mut ts = top.borrow_mut(mc);
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
        }
        CallbackAction::Call { function, then } => {
            let mut ts = top.borrow_mut(mc);
            // If `then` provided, the sequence is the call's "completion
            // handler"; it inherits the caller's expected_returns. The
            // sequence sees the called function's results at stack[bottom..].
            if let Some(seq) = then {
                ts.frames.push(Frame::Sequence {
                    seq,
                    bottom,
                    func_idx,
                    returns,
                    pending_error: None,
                });
            }
            // Now schedule the call. For a Lua target, push a LuaFrame.
            // For a Native target, push a Frame::Start hand-off (driver
            // pumps it next iteration).
            // The function and args layout: stack[bottom] becomes the
            // function slot, args at [bottom+1..]. But the callback that
            // pushed `Call` already left its desired args at stack[bottom..]
            // (per piccolo convention), with no function slot in front. We
            // need to insert the function in front.
            ts.stack.insert(bottom, Value::function(function));
            // Now stack[bottom] = function, stack[bottom+1..] = args.
            schedule_call_at(&mut ts, ctx, bottom, function, returns)?;
        }
        CallbackAction::Yield { to_thread: _, then } => {
            // Phase 7 ships `to_thread = None` only (yield to immediate
            // resumer). `to_thread = Some(_)` (cross-coroutine yield)
            // requires extra thread-stack walking; deferred to P8.
            let mut ts = top.borrow_mut(mc);
            if let Some(seq) = then {
                ts.frames.push(Frame::Sequence {
                    seq,
                    bottom,
                    func_idx,
                    returns,
                    pending_error: None,
                });
            }
            // Yielded values live at stack[bottom..]. Mark thread suspended;
            // the next driver iteration sees Suspended + (we're below the
            // top of thread_stack) → propagate to resumer.
            // Re-stash the bottom on a fresh `pending_action` so the driver
            // can find it when propagating yields.
            ts.pending_action = Some(PendingAction {
                action: CallbackAction::Return, // sentinel; only `bottom` matters
                bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;

            let stack_len = exec.0.borrow().thread_stack.len();
            if stack_len == 1 {
                exec.0.borrow_mut(mc).main_yielded = true;
            }
        }
        CallbackAction::Resume {
            thread: target,
            then,
        } => {
            // Optional `then` fires when target yields/returns; install on
            // the resumer first so it's seen *after* the WaitThread frame.
            {
                let mut ts = top.borrow_mut(mc);
                if let Some(seq) = then {
                    ts.frames.push(Frame::Sequence {
                        seq,
                        bottom,
                        func_idx,
                        returns,
                        pending_error: None,
                    });
                }
                ts.frames.push(Frame::WaitThread {
                    bottom,
                    func_idx,
                    returns,
                });
                ts.status = ThreadStatus::Normal;
            }
            // Move the args (currently at resumer's stack[bottom..]) onto
            // target's stack at [0..]; the target's `Frame::Start(f)` will
            // consume them as parameters on first pump.
            let args: Vec<Value<'gc>> = {
                let mut rs = top.borrow_mut(mc);
                rs.stack.drain(bottom..).collect()
            };
            {
                let mut ts = target.borrow_mut(mc);
                if matches!(ts.status, ThreadStatus::Suspended)
                    && matches!(ts.frames.last(), Some(Frame::Start(_)))
                {
                    // First-resume: stash args at the bottom of the stack;
                    // the `Frame::Start` handler will set up the call frame.
                    ts.stack.clear();
                    ts.stack.extend(args);
                    ts.status = ThreadStatus::Normal;
                } else if matches!(ts.status, ThreadStatus::Suspended) {
                    // Mid-coroutine resume: the inner thread yielded from
                    // its own native CALL site. Drain the yield's
                    // pending_action to recover where the CALL expected
                    // results to land, place the resume-args there per the
                    // standard Lua call-return convention, then re-enter
                    // dispatch (next pump).
                    let yp = ts.pending_action.take();
                    let (inner_bottom, inner_func_idx, inner_returns) = match yp {
                        Some(p) => (p.bottom, p.func_idx, p.returns),
                        None => return Err(RuntimeError::BadMode),
                    };
                    ts.stack.truncate(inner_bottom);
                    ts.stack.extend(args);
                    land_call_results(&mut ts, inner_bottom, inner_func_idx, inner_returns);
                    ts.status = ThreadStatus::Normal;
                } else {
                    return Err(RuntimeError::BadMode);
                }
            }
            // Push target onto the executor's thread stack.
            exec.0.borrow_mut(mc).thread_stack.push(target);
        }
    }
    Ok(())
}

/// Push a Lua/Native call frame for `function` whose function-slot is at
/// `slot` (so args live at `slot+1..`). For Lua: a `LuaFrame` with `base =
/// slot+1`. For Native: invoke synchronously and either land Return values
/// at `slot..` or stash a pending action.
fn schedule_call_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    slot: usize,
    function: Function<'gc>,
    caller_returns: u8,
) -> Result<(), RuntimeError> {
    if let Some(closure) = function.as_lua() {
        let base = slot + 1;
        let needed = base + closure.proto.max_stack_size as usize;
        if ts.stack.len() < needed {
            ts.stack.resize(needed, Value::nil());
        }
        ts.push_lua(LuaFrame {
            closure,
            base,
            pc: 0,
            num_results: caller_returns,
            continuation: None,
        });
        Ok(())
    } else {
        // Native target. Drive synchronously; if it returns Return we land
        // values at [slot..]; otherwise stash a new pending_action.
        let nc = function
            .as_native()
            .expect("function is neither Lua nor Native");
        let args_base = slot + 1;
        let argc = ts.stack.len() - args_base;
        let action = vm::interp::invoke_native(ctx, ts, nc, args_base, argc).map_err(|e| {
            RuntimeError::Lua(crate::lua::Stashable::stash(e, ctx.mutation(), ctx.roots()))
        })?;
        match action {
            CallbackAction::Return => {
                // Move stack[args_base..] down to stack[slot..]. (Slot is
                // where the function used to sit; the function itself is
                // stored back in [slot] before the call by the caller.)
                let retc = ts.stack.len() - args_base;
                for i in 0..retc {
                    ts.stack[slot + i] = ts.stack[args_base + i];
                }
                ts.stack.truncate(slot + retc);
                Ok(())
            }
            other => {
                ts.pending_action = Some(PendingAction {
                    action: other,
                    bottom: args_base,
                    func_idx: slot,
                    returns: caller_returns,
                });
                Ok(())
            }
        }
    }
}

/// Pump a `Frame::Sequence` on top of `top`. Pops the frame, invokes
/// `seq.poll()` (or `seq.error()` if `pending_error.is_some()`), and
/// translates the [`SequencePoll`] back to frame ops.
fn pump_sequence<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    use crate::vm::sequence::{Execution, SequencePoll};
    let mc = ctx.mutation();
    // Pop the sequence frame and call poll/error.
    let (mut seq, bottom, func_idx, returns, pending_error) = {
        let mut ts = top.borrow_mut(mc);
        match ts.frames.pop() {
            Some(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error,
            }) => (seq, bottom, func_idx, returns, pending_error),
            _ => unreachable!("pump_sequence: top wasn't Frame::Sequence"),
        }
    };
    let poll_result = {
        let mut ts = top.borrow_mut(mc);
        let stack_view = crate::env::function::Stack::new(&mut ts.stack, bottom);
        let exec = Execution::new(top);
        if let Some(err) = pending_error {
            seq.error(ctx, exec, err, stack_view)
        } else {
            seq.poll(ctx, exec, stack_view)
        }
    };
    match poll_result {
        Ok(SequencePoll::Pending) => {
            top.borrow_mut(mc).frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
        }
        Ok(SequencePoll::Return) => {
            // Sequence finished. Land values at the original CALL's expected
            // window: stack[func_idx..func_idx+wanted].
            let mut ts = top.borrow_mut(mc);
            land_call_results(&mut ts, bottom, func_idx, returns);
        }
        Ok(SequencePoll::Call {
            function,
            bottom: rel,
        }) => {
            let abs_bottom = bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with results at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
            // Schedule the call: insert function at abs_bottom, args after.
            ts.stack.insert(abs_bottom, Value::function(function));
            schedule_call_at(&mut ts, ctx, abs_bottom, function, 0)?;
        }
        Ok(SequencePoll::TailCall(function)) => {
            // Sequence is done; the call's results must land at the
            // original CALL site `func_idx`. Args sit at stack[bottom..],
            // which is adjacent to `func_idx` after a normal CALL but not
            // after a TAILCALL→native→sequence chain (where bottom lives
            // inside the popped tail-callee's window). Compact the args
            // down to func_idx+1, then place the function at func_idx.
            let mut ts = top.borrow_mut(mc);
            let argc = ts.stack.len() - bottom;
            let new_args_base = func_idx + 1;
            if new_args_base < bottom {
                for i in 0..argc {
                    ts.stack[new_args_base + i] = ts.stack[bottom + i];
                }
                ts.stack.truncate(new_args_base + argc);
            }
            ts.stack[func_idx] = Value::function(function);
            schedule_call_at(&mut ts, ctx, func_idx, function, returns)?;
        }
        Ok(SequencePoll::Yield {
            to_thread: _,
            bottom: rel,
        }) => {
            let abs_bottom = bottom + rel;
            let mut ts = top.borrow_mut(mc);
            // Re-push self to be re-polled with resume-args at stack[bottom..].
            ts.frames.push(Frame::Sequence {
                seq,
                bottom,
                func_idx,
                returns,
                pending_error: None,
            });
            ts.pending_action = Some(PendingAction {
                action: CallbackAction::Return,
                bottom: abs_bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;
            let stack_len = exec.0.borrow().thread_stack.len();
            if stack_len == 1 {
                exec.0.borrow_mut(mc).main_yielded = true;
            }
        }
        Ok(SequencePoll::TailYield(_to_thread)) => {
            let mut ts = top.borrow_mut(mc);
            ts.pending_action = Some(PendingAction {
                action: CallbackAction::Return,
                bottom,
                func_idx,
                returns,
            });
            ts.status = ThreadStatus::Suspended;
            let stack_len = exec.0.borrow().thread_stack.len();
            if stack_len == 1 {
                exec.0.borrow_mut(mc).main_yielded = true;
            }
        }
        Ok(SequencePoll::Resume {
            thread: _target,
            bottom: _rel,
        }) => {
            // Resume from inside a sequence: defer to P8 polish.
            return Err(RuntimeError::BadMode);
        }
        Ok(SequencePoll::TailResume(_target)) => {
            return Err(RuntimeError::BadMode);
        }
        Err(err) => {
            top.borrow_mut(mc).frames.push(Frame::Error(err));
        }
    }
    Ok(())
}

/// After an inner thread terminated or yielded, transfer its result-bottom
/// values to the resumer's `Frame::WaitThread`, pop both, and let the
/// resumer continue (the next driver pump finds either a `Frame::Sequence`
/// or a `Frame::Lua` ready to resume).
fn propagate_inner_to_resumer<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    inner: Thread<'gc>,
    inner_bottom: usize,
    inner_yielded: bool,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let values: Vec<Value<'gc>> = {
        let ts = inner.borrow();
        ts.stack[inner_bottom..].to_vec()
    };
    // Pop the inner thread.
    exec.0.borrow_mut(mc).thread_stack.pop();
    // If the inner is just yielded (not terminated), keep its state for
    // future resumes; the inner's pending_action retained the bottom.
    if !inner_yielded {
        // After Result, clear the inner's stack so subsequent fetches see
        // an empty thread.
        let mut ts = inner.borrow_mut(mc);
        ts.stack.clear();
    }
    let resumer = *exec.0.borrow().thread_stack.last().unwrap();
    let mut rs = resumer.borrow_mut(mc);
    let (wt_bottom, wt_func_idx, wt_returns) = match rs.frames.pop() {
        Some(Frame::WaitThread {
            bottom,
            func_idx,
            returns,
        }) => (bottom, func_idx, returns),
        _ => unreachable!("propagate_inner_to_resumer: resumer top is not WaitThread"),
    };
    // Land values at resumer.stack[wt_bottom..] for the next sequence /
    // call frame to consume. If a `then` sequence sits underneath, it'll
    // pick them up at its own `bottom == wt_bottom`. If not, do the
    // standard CALL-landing right now.
    rs.stack.truncate(wt_bottom);
    rs.stack.extend(values);
    let next_is_sequence = matches!(rs.frames.last(), Some(Frame::Sequence { .. }));
    if !next_is_sequence {
        // No follow-up sequence: deliver results directly to the original
        // Lua CALL window.
        land_call_results(&mut rs, wt_bottom, wt_func_idx, wt_returns);
    }
    rs.status = ThreadStatus::Normal;
    Ok(())
}

/// Move values at `stack[bottom..]` into the original CALL's expected
/// landing slot per the standard Lua convention. If `frames` is empty
/// after the move (e.g., a tailcalled native suspended and the calling
/// Lua frame was already popped at TAILCALL time), terminate the thread
/// with `ThreadStatus::Result { bottom: func_idx }`.
fn land_call_results<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    bottom: usize,
    func_idx: usize,
    returns: u8,
) {
    let retc = ts.stack.len() - bottom;
    let wanted = if returns == 0 {
        retc
    } else {
        returns as usize - 1
    };
    let to_copy = retc.min(wanted);
    for i in 0..to_copy {
        ts.stack[func_idx + i] = ts.stack[bottom + i];
    }
    for i in to_copy..wanted {
        ts.stack[func_idx + i] = Value::nil();
    }
    if returns == 0 {
        ts.stack.truncate(func_idx + retc);
    } else {
        // Keep stack >= func_idx + wanted; restore caller's max_stack_size
        // window if a Lua frame is on top.
        if let Some(frame) = ts.top_lua() {
            let needed = frame.base + frame.closure.proto.max_stack_size as usize;
            if ts.stack.len() < needed {
                ts.stack.resize(needed, Value::nil());
            }
        }
    }
    if ts.frames.is_empty() {
        ts.status = ThreadStatus::Result { bottom: func_idx };
    }
}

/// Walk a thread's frame stack popping Lua/Wait frames (closing upvalues
/// at each `bottom`) until a `Sequence` frame can catch the error.
///
/// On no-catcher: if the thread isn't the bottom of the executor's
/// thread stack, route the error to the resumer's `Frame::WaitThread`
/// and pop the inner thread. This lets a coroutine error propagate to
/// the resumer's `PCallSequence::error`. If the thread *is* the bottom,
/// surface as `RuntimeError::Lua` to the host.
fn unwind_error<'gc>(
    exec: Executor<'gc>,
    ctx: Context<'gc>,
    top: Thread<'gc>,
) -> Result<(), RuntimeError> {
    let mc = ctx.mutation();
    let err = {
        let mut ts = top.borrow_mut(mc);
        let err = match ts.frames.pop() {
            Some(Frame::Error(e)) => e,
            _ => unreachable!(),
        };
        loop {
            match ts.frames.last() {
                Some(Frame::Lua(lf)) => {
                    let base = lf.base;
                    ts.frames.pop();
                    close_upvalues_at(&mut ts, ctx, base);
                    ts.stack.truncate(base);
                }
                Some(Frame::Sequence { .. }) => {
                    if let Some(Frame::Sequence { pending_error, .. }) = ts.frames.last_mut() {
                        *pending_error = Some(err);
                    }
                    return Ok(());
                }
                Some(Frame::WaitThread { .. }) => {
                    ts.frames.pop();
                }
                Some(Frame::Start(_)) | Some(Frame::Error(_)) => {
                    ts.frames.pop();
                }
                None => break err,
            }
        }
    };

    // No catcher on this thread. If we're an inner coroutine, propagate to
    // the resumer's WaitThread → its next Sequence can catch.
    let stack_len = exec.0.borrow().thread_stack.len();
    if stack_len > 1 {
        exec.0.borrow_mut(mc).thread_stack.pop();
        let resumer = *exec.0.borrow().thread_stack.last().unwrap();
        let mut rs = resumer.borrow_mut(mc);
        // Pop the WaitThread.
        match rs.frames.pop() {
            Some(Frame::WaitThread { .. }) => {}
            _ => unreachable!("inner-thread error: resumer top isn't WaitThread"),
        }
        rs.frames.push(Frame::Error(err));
        rs.status = ThreadStatus::Normal;
        return Ok(());
    }

    // Bottom of the thread stack — surface to host.
    Err(RuntimeError::Lua(crate::lua::Stashable::stash(
        err,
        mc,
        ctx.roots(),
    )))
}

/// Close open upvalues and TBC variables whose stack indices are >=
/// `start_idx`. Mirrors `interp::close_upvalues` but lives in executor.rs
/// to avoid an interp-private dependency in the unwinder.
fn close_upvalues_at<'gc>(
    ts: &mut crate::env::thread::ThreadState<'gc>,
    ctx: Context<'gc>,
    start_idx: usize,
) {
    use crate::env::function::UpvalueState;
    let mc = ctx.mutation();
    ts.open_upvalues.retain(|uv| {
        let should_close = {
            let b = uv.borrow();
            match &*b {
                UpvalueState::Open { index, .. } => *index >= start_idx,
                UpvalueState::Closed(_) => false,
            }
        };
        if should_close {
            let val = ts.stack[{
                let b = uv.borrow();
                match &*b {
                    UpvalueState::Open { index, .. } => *index,
                    _ => unreachable!(),
                }
            }];
            *uv.borrow_mut(mc) = UpvalueState::Closed(val);
            false
        } else {
            true
        }
    });
    ts.tbc_slots.retain(|&i| i < start_idx);
}
