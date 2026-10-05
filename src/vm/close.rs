//! Calling `__close` on to-be-closed variables (`luaF_close`).

use crate::env::error::{Error, Exit};
use crate::env::function::Stack;
use crate::env::thread::{ExecKind, TbcEntry, ThreadState, ThreadStatus};
use crate::env::{Function, MetamethodBits, NativeClosure, Value};
use crate::lua::Context;
use crate::vm::native::{CallbackAction, OnOk, Protect};

/// Entry of a coroutine seeded by [`seed_thread_close`]: close its variables
/// for `coroutine.close` and `coroutine.wrap` (`luaE_resetthread`), which
/// cannot yield meanwhile. An error in a `__close` replaces the error for the
/// rest. Returns `true`, or `false` and the last error. It is the thread's
/// base level, where a self-close from a `__close` lands
/// (`lua_closethread`'s outer `luaD_closeprotected`).
fn thread_close_entry<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction, Error<'gc>> {
    let err = stack.thread_mut().death_error.take();
    stack.replace(&[Value::boolean(err.is_some()), err.unwrap_or(Value::nil())]);
    close_step(ctx, stack, false)
}

/// `coroutine.close` on the running coroutine: close all its variables, above
/// its still-live frames, then end it (`lua_closethread` on itself).
pub(crate) fn close_running<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction, Error<'gc>> {
    stack.replace(&[Value::boolean(false), Value::nil()]);
    close_step(ctx, stack, true)
}

/// Call the next variable's `__close` with the error so far, the window
/// being `[has error, error]`; with all closed, return as the thread's close
/// or, with `exit`, end the thread.
fn close_step<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    exit: bool,
) -> Result<CallbackAction, Error<'gc>> {
    let has_err = stack.get(0).get_boolean() == Some(true);
    let ts = stack.thread_mut();
    let Some(entry) = ts.tbc_list.pop() else {
        ts.no_yield = false;
        let err = has_err.then(|| Error::new(ctx, stack.get(1)));
        if exit {
            return Err(Error::exit(ctx, err));
        }
        match err {
            Some(err) => stack.replace(&[Value::boolean(false), err.value()]),
            None => stack.replace(&[Value::boolean(true)]),
        }
        return Ok(CallbackAction::Return);
    };
    ts.no_yield = true;
    let v = entry.value(&ts.stack);
    let errv = stack.get(1);
    stack.truncate(2);
    stack.extend([ctx.mm_of(v, MetamethodBits::CLOSE), v]);
    if has_err {
        stack.push(errv);
    }
    Ok(CallbackAction::CallThen {
        at: 2,
        protect: if exit { Protect::Errors } else { Protect::Base },
        ok: OnOk::Cont,
        cont: if exit {
            close_running_cont
        } else {
            close_entry_cont
        },
    })
}

fn close_entry_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    close_resumed(ctx, stack, status, false)
}

fn close_running_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    close_resumed(ctx, stack, status, true)
}

fn close_resumed<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
    exit: bool,
) -> Result<CallbackAction, Error<'gc>> {
    stack.truncate(2);
    if let Err(err) = status {
        let slots = stack.as_mut_slice();
        match err.exit_kind() {
            Exit::No | Exit::Failed => {
                slots[0] = Value::boolean(true);
                slots[1] = err.value();
            }
            // Keeps the status, but the error object is lost with the stack
            // the inner reset cleared, as in Lua.
            Exit::Clean => slots[1] = Value::nil(),
            Exit::Process(_) => unreachable!("a process exit is never caught"),
        }
    }
    close_step(ctx, stack, exit)
}

/// Reset the suspended or dead `ts`, leaving it to close its open variables
/// when next resumed; `false` if it had none, leaving it merely reset. Its
/// death error, if any, goes to the first `__close` and is kept on reset.
pub(crate) fn seed_thread_close<'gc>(ctx: Context<'gc>, ts: &mut ThreadState<'gc>) -> bool {
    crate::vm::interp::close_upvalues(ctx.mutation(), ts, 0);
    let tbc_list: Vec<_> = ts
        .tbc_list
        .iter()
        .map(|e| TbcEntry::Detached {
            level: 0,
            value: e.value(&ts.stack),
        })
        .collect();
    let death_error = ts.death_error;
    ts.reset();
    ts.death_error = death_error;
    ts.status = ThreadStatus::Stopped;
    if tbc_list.is_empty() {
        return false;
    }
    ts.tbc_list = tbc_list;
    let entry = Function::new_action(ctx.mutation(), thread_close_entry, &[]);
    ts.push_exec(ExecKind::Start(Value::function(entry)));
    ts.status = ThreadStatus::Suspended;
    true
}
