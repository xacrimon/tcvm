//! The protocol between the VM and natives that act once they return: call
//! a function and continue in a [`NativeCont`], resume or yield a coroutine,
//! or wait as an async native.

use crate::env::Thread;
use crate::env::error::Error;
use crate::env::function::Stack;

/// What an [`ActionFn`](crate::env::ActionFn) native or a [`NativeCont`]
/// asks of the VM on return.
pub enum CallbackAction {
    /// Plain synchronous return. Stack values above `bottom` are the results.
    Return,
    /// Call `stack[at]` with the values above it, then run `cont` with the
    /// call's results in their place, `stack[at..]`. The native keeps its
    /// state in `stack[..at]` meanwhile: it gets a frame of its own, and the
    /// call and `cont` run without leaving the interpreter.
    CallThen {
        at: u32,
        protect: Protect,
        ok: OnOk,
        cont: NativeCont,
    },
    /// Resume the coroutine at `stack[at]` with the values above it, then run
    /// `cont` with what it yields or returns in their place, or with the
    /// error that killed it. Switches threads without leaving the
    /// interpreter.
    Resume { at: u32, ok: OnOk, cont: NativeCont },
    /// Yield the window to the resumer; the values it resumes with are the
    /// native's results.
    Yield,
    /// Yield `stack[at..]` to the resumer, then run `cont` with the values it
    /// resumes with in their place.
    YieldThen { at: u32, cont: NativeCont },
    /// The async native's future was spawned (`Stack::spawn`): poll it from a
    /// frame of its own.
    Async,
    /// Leave the native's frame for the host to come back to: its future
    /// waits on the host.
    Pending,
}

const _: () = assert!(std::mem::size_of::<Result<CallbackAction, Error<'static>>>() == 16);

/// Whether a [`CallbackAction::CallThen`]'s continuation receives the
/// errors its call raises, instead of them unwinding past it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Protect {
    No,
    /// `pcall`.
    Errors,
    /// `xpcall`: as `Errors`, after running the message handler at
    /// `stack[0]` on top of the failing frames.
    Handler,
    /// As `Errors`, and an exit too: the thread's base level
    /// (`luaD_throwbaselevel`).
    Base,
}

/// What `cont` does with the results of a [`CallbackAction::CallThen`] or
/// [`CallbackAction::Resume`] that succeeded, when that is simple enough for
/// the VM to do it instead of calling `cont`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OnOk {
    Cont,
    /// They are the native's results.
    Return,
    /// `true` and then them (`pcall`).
    ReturnTrue,
}

/// The continuation of a [`CallbackAction::CallThen`]: the native's window
/// with the call's results at `at`, or with nothing above `at` and the error
/// when a protected call failed.
pub type NativeCont = for<'gc, 'a> fn(
    ctx: crate::lua::Context<'gc>,
    closure: &'a crate::env::NativeClosure<'gc>,
    stack: Stack<'gc, 'a>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>>;

impl CallbackAction {
    /// [`CallbackAction::CallThen`] of `stack[at]`, unprotected.
    pub fn call_then(at: usize, cont: NativeCont) -> Self {
        CallbackAction::CallThen {
            at: at as u32,
            protect: Protect::No,
            ok: OnOk::Cont,
            cont,
        }
    }
}

/// Read-only view of the executor, reached from a native through
/// [`Stack::exec`]. Carries the currently-running
/// thread; richer fields (full `&[Thread<'gc>]` thread stack, fuel handle)
/// aren't implemented yet. The running thread's frames are reached through
/// [`Stack::lua_frames`], which already borrows the thread.
#[derive(Clone, Copy)]
pub struct Execution<'gc> {
    current_thread: Thread<'gc>,
    is_main: bool,
}

impl<'gc> Execution<'gc> {
    pub fn new(current_thread: Thread<'gc>, is_main: bool) -> Self {
        Execution {
            current_thread,
            is_main,
        }
    }

    /// Thread the native is running on.
    pub fn current_thread(self) -> Thread<'gc> {
        self.current_thread
    }

    /// Whether the running thread is the executor's main (entry) thread.
    pub fn is_main(self) -> bool {
        self.is_main
    }
}
