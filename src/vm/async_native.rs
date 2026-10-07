//! Async natives: a native that calls Lua, yields, or waits on the host
//! written as a Rust `async` block.
//!
//! An [`AsyncFn`] reads its arguments and [`spawn`](Stack::spawn)s a future,
//! which the VM polls from the native's own frame inside dispatch: each
//! `cx.call(..).await` makes the call and polls again with its results, with
//! no executor round trip. A future that awaits something else (a host
//! future) suspends the executor, which `step` reports as
//! [`StepResult::Pending`](crate::lua::StepResult) until the waker fires.
//!
//! Lua values are branded with `'gc` and can't be held across an `.await`;
//! [`Local`] handles keep them instead, rooted on the thread until the native
//! returns.

use std::alloc::{AllocError, Allocator, Layout};
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::ptr::NonNull;
use std::rc::Rc;
use std::task::{Context as TaskCx, Poll};

use crate::env::function::{NativeClosure, Stack};
use crate::env::thread::ThreadState;
use crate::env::{Error, Value};
use crate::lua::Context;
use crate::vm::native::{CallbackAction, OnOk, Protect};

/// An async native: reads its arguments from `stack` and returns
/// [`Stack::spawn`]'s token, or fails right away.
pub type AsyncFn = for<'gc, 'a> fn(
    ctx: Context<'gc>,
    closure: &'a NativeClosure<'gc>,
    stack: Stack<'gc, 'a>,
) -> Result<Spawned, Error<'gc>>;

/// Proof that an [`AsyncFn`] spawned its future.
pub struct Spawned(());

/// What a spawned future returns: `Ok` with the native's results left in its
/// window (see [`Cx::enter`]), or the error it raises.
pub type TaskResult = Result<(), AsyncError>;

/// A value kept across awaits: rooted on the thread until the async native
/// that made it returns. Using it after that, or from another native's
/// thread, panics.
#[derive(Clone, Copy, Debug)]
pub struct Local {
    index: u32,
    epoch: u32,
}

/// A Lua error held by an async native's future.
#[derive(Debug)]
pub struct AsyncError {
    value: Local,
    level: u32,
}

/// An async native's handle to the VM, valid while its future is polled.
#[derive(Clone)]
pub struct Cx {
    env: Rc<EnvCell>,
}

/// Where a poll's [`Env`] is, null between polls (and inside [`Cx::enter`]).
pub(crate) struct EnvCell(Cell<*mut Env<'static>>);

impl EnvCell {
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(EnvCell(Cell::new(std::ptr::null_mut())))
    }
}

/// What a future sees while it is polled, on `async_cont`'s stack.
pub(crate) struct Env<'gc> {
    ctx: Context<'gc>,
    thread: *mut ThreadState<'gc>,
    base: usize,
    /// How a protected call ended, for the `pcall` being awaited.
    status: Option<Result<(), Error<'gc>>>,
    request: Request,
}

/// What the future, returning `Pending`, wants done before its next poll.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Request {
    /// Nothing: it waits on the host.
    None,
    Call {
        at: usize,
        protect: Protect,
    },
    Yield {
        at: usize,
    },
    /// Let the host run, then poll again.
    Pending,
}

impl Cx {
    /// Run `f` on the native's window and context.
    pub fn enter<R>(&self, f: impl for<'gc> FnOnce(Context<'gc>, Stack<'gc, '_>) -> R) -> R {
        let env = self.env.0.get();
        assert!(!env.is_null(), "Cx used outside its native's poll");
        // Null meanwhile, so a nested `enter` can't alias the thread.
        struct Restore<'a>(&'a EnvCell, *mut Env<'static>);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.0.set(self.1);
            }
        }
        let _restore = Restore(&self.env, env);
        self.env.0.set(std::ptr::null_mut());
        let env = unsafe { &mut *env };
        f(env.ctx, Stack::new(unsafe { &mut *env.thread }, env.base))
    }

    /// [`enter`](Self::enter) for a body that may fail with a Lua error.
    pub fn try_enter<R>(
        &self,
        f: impl for<'gc> FnOnce(Context<'gc>, &mut Stack<'gc, '_>) -> Result<R, Error<'gc>>,
    ) -> Result<R, AsyncError> {
        self.enter(|ctx, mut stack| f(ctx, &mut stack).map_err(|e| AsyncError::new(&mut stack, e)))
    }

    /// Call `window[at]` with the values above it; its results then replace
    /// them. Unprotected: an error unwinds past the native, dropping its
    /// future.
    pub fn call(&self, at: usize) -> impl Future<Output = ()> + '_ {
        let request = Request::Call {
            at,
            protect: Protect::No,
        };
        self.request(request, |_| ())
    }

    /// [`call`](Self::call), catching the error it raises: then nothing is
    /// left above `at`.
    pub fn pcall(&self, at: usize) -> impl Future<Output = TaskResult> + '_ {
        let request = Request::Call {
            at,
            protect: Protect::Errors,
        };
        self.request(request, |env| match env.status.take().unwrap_or(Ok(())) {
            Ok(()) => Ok(()),
            Err(e) => {
                let mut stack = Stack::new(unsafe { &mut *env.thread }, env.base);
                Err(AsyncError::new(&mut stack, e))
            }
        })
    }

    /// Yield `window[at..]` to the resumer; the values it resumes with
    /// replace them.
    pub fn yield_(&self, at: usize) -> impl Future<Output = ()> + '_ {
        self.request(Request::Yield { at }, |_| ())
    }

    /// Let the host run before going on, as a host future that is ready
    /// again right away would.
    pub fn pending(&self) -> impl Future<Output = ()> + '_ {
        self.request(Request::Pending, |_| ())
    }

    /// A future that asks for `request` once, then finishes with `done`.
    fn request<R, F: FnOnce(&mut Env<'static>) -> R>(
        &self,
        request: Request,
        done: F,
    ) -> RequestFut<'_, F> {
        RequestFut {
            env: &self.env,
            request,
            done: Some(done),
            sent: false,
        }
    }

    /// An error with `msg` at `level`, as [`Error::from_str`] makes.
    pub fn error(&self, msg: &str) -> AsyncError {
        self.enter(|ctx, mut stack| AsyncError::new(&mut stack, Error::from_str(ctx, msg)))
    }
}

struct RequestFut<'a, F> {
    env: &'a EnvCell,
    request: Request,
    done: Option<F>,
    sent: bool,
}

impl<F: FnOnce(&mut Env<'static>) -> R, R> Future for RequestFut<'_, F> {
    type Output = R;

    fn poll(self: Pin<&mut Self>, _cx: &mut TaskCx<'_>) -> Poll<R> {
        // SAFETY: nothing is pinned structurally.
        let this = unsafe { self.get_unchecked_mut() };
        let env = this.env.0.get();
        assert!(!env.is_null(), "awaited outside its native's poll");
        let env = unsafe { &mut *env };
        if this.sent {
            let done = this.done.take().expect("polled after completion");
            return Poll::Ready(done(env));
        }
        assert!(
            env.request == Request::None,
            "a second Lua request before the first was awaited"
        );
        env.request = this.request;
        this.sent = true;
        Poll::Pending
    }
}

impl AsyncError {
    /// The error value.
    pub fn value(&self) -> Local {
        self.value
    }

    fn new<'gc>(stack: &mut Stack<'gc, '_>, e: Error<'gc>) -> Self {
        AsyncError {
            value: stack.local(e.value()),
            level: e.level() as u32,
        }
    }

    /// The error raised from `stack`'s native, as `Error::new(v).with_level`.
    pub fn from_value<'gc>(stack: &mut Stack<'gc, '_>, value: Value<'gc>, level: usize) -> Self {
        AsyncError {
            value: stack.local(value),
            level: level as u32,
        }
    }
}

impl<'gc> Stack<'gc, '_> {
    /// Root `v` until the async native running returns.
    ///
    /// # Panics
    /// Outside an async native.
    pub fn local(&mut self, v: Value<'gc>) -> Local {
        let ts = self.thread_mut();
        assert!(ts.local_epoch != 0, "Stack::local outside an async native");
        let index = ts.locals.len() as u32;
        ts.locals.push((v, ts.local_epoch));
        Local {
            index,
            epoch: ts.local_epoch,
        }
    }

    /// The value `l` roots.
    pub fn get_local(&mut self, l: Local) -> Value<'gc> {
        let ts = self.thread_mut();
        match ts.locals.get(l.index as usize) {
            Some(&(v, epoch)) if epoch == l.epoch => v,
            _ => panic!("a Local used after its native returned"),
        }
    }

    /// Spawn the future an [`AsyncFn`] runs as, built by `f` with its `Cx`.
    ///
    /// # Panics
    /// Outside an `AsyncFn`, or a second time in one.
    pub fn spawn<Fut>(&mut self, f: impl FnOnce(Cx) -> Fut) -> Spawned
    where
        Fut: Future<Output = TaskResult> + 'static,
    {
        let ts = self.thread_mut();
        let header = ts.spawning.take().expect("Stack::spawn outside an AsyncFn");
        let env = ts.async_env.get_or_insert_with(EnvCell::new).clone();
        let alloc = TaskAlloc(&raw mut ts.task_arena);
        let fut: Pin<Box<dyn Future<Output = TaskResult>, TaskAlloc>> =
            Box::into_pin(Box::new_in(f(Cx { env }), alloc)
                as Box<dyn Future<Output = TaskResult>, TaskAlloc>);
        let ts = self.thread_mut();
        ts.tasks.push(Task { fut, header });
        Spawned(())
    }
}

impl<'gc> Stack<'gc, '_> {
    /// [`spawn`](Stack::spawn) from an action native that turns async: it
    /// returns what to return. Its locals must be made in the future.
    pub(crate) fn spawn_action<Fut>(
        &mut self,
        ctx: Context<'gc>,
        f: impl FnOnce(Cx) -> Fut,
    ) -> CallbackAction
    where
        Fut: Future<Output = TaskResult> + 'static,
    {
        let ts = self.thread_mut();
        debug_assert!(ts.spawning.is_none());
        ts.spawning = Some(TaskHeader {
            local_base: ts.locals.len() as u32,
            prev_epoch: ts.local_epoch,
        });
        ts.local_epoch = ctx.next_epoch();
        let Spawned(()) = self.spawn(f);
        CallbackAction::Async
    }
}

/// An async native's future, with where its locals start and the epoch to
/// restore once it is done.
pub(crate) struct Task {
    fut: Pin<Box<dyn Future<Output = TaskResult>, TaskAlloc>>,
    header: TaskHeader,
}

#[derive(Clone, Copy)]
pub(crate) struct TaskHeader {
    local_base: u32,
    prev_epoch: u32,
}

impl<'gc> ThreadState<'gc> {
    /// Drop the innermost task and its locals.
    #[inline]
    pub(crate) fn drop_task(&mut self) {
        let task = self.tasks.pop().expect("no task to drop");
        drop(task.fut);
        self.locals.truncate(task.header.local_base as usize);
        self.local_epoch = task.header.prev_epoch;
    }
}

/// Run async native `f` of `nc` on the window `win .. top`.
pub(crate) fn invoke_async<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    f: AsyncFn,
    nc: &NativeClosure<'gc>,
    win: usize,
) -> Result<CallbackAction, Error<'gc>> {
    let header = TaskHeader {
        local_base: thread.locals.len() as u32,
        prev_epoch: thread.local_epoch,
    };
    thread.local_epoch = ctx.next_epoch();
    thread.spawning = Some(header);
    let r = f(ctx, nc, Stack::new(thread, win));
    let spawned = thread.spawning.take().is_none();
    match r {
        Ok(Spawned(())) => {
            assert!(spawned, "an AsyncFn returned another native's Spawned");
            Ok(CallbackAction::Async)
        }
        Err(e) => {
            if spawned {
                thread.drop_task();
            } else {
                thread.locals.truncate(header.local_base as usize);
                thread.local_epoch = header.prev_epoch;
            }
            Err(e)
        }
    }
}

/// The continuation of an async native's frame: poll its future, with how
/// the call it waited for went.
pub(crate) fn async_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    let (thread, base) = stack.into_parts();
    let mut env = Env {
        ctx,
        thread,
        base,
        status: Some(status),
        request: Request::None,
    };
    // Through a pointer to the arena, not a borrow of the thread, which the
    // future's `enter`s take whole.
    let task = unsafe { thread.tasks.last_mut().unwrap_unchecked() };
    let fut: *mut dyn Future<Output = TaskResult> =
        unsafe { task.fut.as_mut().get_unchecked_mut() };
    let envp = (&raw mut env).cast::<Env<'static>>();
    // The thread keeps it, so it outlives the poll.
    let cell: *const EnvCell = &**unsafe { thread.async_env.as_ref().unwrap_unchecked() };
    let cell = unsafe { &*cell };
    let poll = {
        // Restored after, for a poll nested in another (a native running a
        // second executor from `enter`).
        let prev = cell.0.replace(envp);
        let fut = unsafe { Pin::new_unchecked(&mut *fut) };
        let r = fut.poll(&mut TaskCx::from_waker(ctx.waker()));
        cell.0.set(prev);
        r
    };
    let thread = unsafe { &mut *env.thread };
    match poll {
        Poll::Ready(r) => {
            assert!(env.request == Request::None, "a Lua request not awaited");
            let r = r.map_err(|e| {
                let mut stack = Stack::new(thread, base);
                let value = stack.get_local(e.value);
                Error::new(ctx, value).with_level(e.level as usize)
            });
            thread.drop_task();
            r.map(|()| CallbackAction::Return)
        }
        Poll::Pending => Ok(match env.request {
            Request::Call { at, protect } => CallbackAction::CallThen {
                at: at as u32,
                protect,
                ok: OnOk::Cont,
                cont: async_cont,
            },
            Request::Yield { at } => CallbackAction::YieldThen {
                at: at as u32,
                cont: async_cont,
            },
            Request::Pending => {
                ctx.waker().wake_by_ref();
                CallbackAction::Pending
            }
            Request::None => CallbackAction::Pending,
        }),
    }
}

/// A thread's LIFO bump arena for its tasks' futures, which nest like the
/// frames that own them. Chunks never move; a free that isn't the last
/// allocation (a future dropped out of order) is only reclaimed once the
/// arena empties.
pub(crate) struct TaskArena {
    chunks: Vec<(NonNull<u8>, usize)>,
    /// Current chunk and the free space in it.
    top: usize,
    end: usize,
    live: usize,
}

const CHUNK: usize = 4096;

impl TaskArena {
    pub(crate) const fn new() -> Self {
        TaskArena {
            chunks: Vec::new(),
            top: 0,
            end: 0,
            live: 0,
        }
    }

    #[inline(always)]
    fn alloc(&mut self, l: Layout) -> NonNull<u8> {
        let p = (self.top + l.align() - 1) & !(l.align() - 1);
        if self.top != 0 && p + l.size() <= self.end {
            self.top = p + l.size();
            self.live += 1;
            return unsafe { NonNull::new_unchecked(p as *mut u8) };
        }
        self.alloc_chunk(l)
    }

    #[cold]
    #[inline(never)]
    fn alloc_chunk(&mut self, l: Layout) -> NonNull<u8> {
        let size = (l.size() + l.align()).max(CHUNK);
        let layout = Layout::from_size_align(size, 16).unwrap();
        let base = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        self.chunks.push((base, size));
        self.end = base.as_ptr() as usize + size;
        let p = (base.as_ptr() as usize + l.align() - 1) & !(l.align() - 1);
        self.top = p + l.size();
        self.live += 1;
        unsafe { NonNull::new_unchecked(p as *mut u8) }
    }

    #[inline(always)]
    fn free(&mut self, p: NonNull<u8>, l: Layout) {
        self.live -= 1;
        if p.as_ptr() as usize + l.size() == self.top {
            self.top = p.as_ptr() as usize;
        }
        if self.live == 0 {
            // Back to the first chunk; later ones stay for the next burst.
            if let Some(&(base, size)) = self.chunks.first() {
                self.top = base.as_ptr() as usize;
                self.end = self.top + size;
            }
        }
    }
}

impl Drop for TaskArena {
    fn drop(&mut self) {
        for &(base, size) in &self.chunks {
            unsafe {
                std::alloc::dealloc(base.as_ptr(), Layout::from_size_align_unchecked(size, 16))
            };
        }
    }
}

/// [`TaskArena`] as an allocator, for the task boxes. The arena is a field of
/// the thread state, which a GC allocation keeps in place, and outlives the
/// tasks (`ThreadState::tasks` is declared, so dropped, first).
#[derive(Clone, Copy)]
pub(crate) struct TaskAlloc(*mut TaskArena);

unsafe impl Allocator for TaskAlloc {
    #[inline(always)]
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let p = unsafe { (*self.0).alloc(l) };
        Ok(NonNull::slice_from_raw_parts(p, l.size()))
    }

    #[inline(always)]
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        unsafe { (*self.0).free(p, l) }
    }
}
