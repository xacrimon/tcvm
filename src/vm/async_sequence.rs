//! `async fn` callbacks for the VM. Wraps user-written `async move` blocks
//! so they can drive a [`Sequence`] state machine via `.await`.
//!
//! ## Lifetime story
//!
//! Futures created from `async {}` blocks bake in their captured-reference
//! lifetimes at suspend time and don't implement [`Collect`], which means
//! they cannot directly hold `'gc`-branded values across `.await` points.
//!
//! The standard workaround (mirroring piccolo's `async_callback.rs`) is:
//!
//! 1. Each [`async_sequence`] gets a per-sequence [`DynamicRootSet`].
//! 2. The user's `async move` block is given two handles: a [`Locals`] (to
//!    stash GC values into the per-sequence root set, returning `'static`
//!    handles), and an [`AsyncSequence`] (the proxy used to call back into
//!    the VM via `.await`).
//! 3. During each `Sequence::poll`, we install a [`Shared`] live with the
//!    *current* `'gc` brand into a [`SharedSlot`], drive the future once,
//!    then null the slot. Inside `AsyncSequence::enter` / `try_enter`, the
//!    slot is fetched and the lifetimes are re-projected — the closure is
//!    HRTB so the user can't hold references past the `enter`.
//!
//! `SharedSlot::with`/`visit` use `mem::transmute` to erase / restore the
//! lifetimes; soundness depends on the slot only being read inside `with`'s
//! drop guard. See piccolo `async_callback.rs:468-515` for the same
//! pattern with full safety commentary.
//!
//! ## Improvements vs piccolo
//!
//! TCVM's [`MetricsAlloc`] is already `'gc`-branded
//! (`src/dmm/allocator_api.rs:8`), so [`BoxSequence`] doesn't need the
//! `'static`-erasure dance piccolo uses on its allocator. Sequences boxed
//! via [`async_sequence`] still allocate through the metered allocator.
//!
//! ## Constraints (panics if violated)
//!
//! * Methods on `AsyncSequence` may *only* be called from inside the
//!   returned future. Calling them outside (or after the sequence has been
//!   collected) panics.
//! * Every `.await` inside the future must be on a method of the held
//!   `AsyncSequence`. Awaiting an arbitrary external future will panic
//!   inside `poll_fut` — the noop waker can't drive it.

use std::cell::Cell;
use std::future::{Future, poll_fn};
use std::marker::PhantomData;
use std::mem;
use std::pin::Pin;
use std::ptr;
use std::rc::Rc;
use std::task::{self, Poll, RawWaker, RawWakerVTable, Waker};

use crate::dmm::{Collect, DynamicRootSet, Mutation, Trace};
use crate::env::error::Error;
use crate::env::function::Stack;
use crate::env::value::Value;
use crate::env::{Function, Thread};
use crate::lua::Context;
use crate::lua::stash::{Fetchable, Stashable, StashedError, StashedFunction, StashedThread};
use crate::vm::sequence::{BoxSequence, Execution, Sequence, SequencePoll};

/// Build a [`Sequence`] from a Rust `async move` block.
///
/// `create` is called immediately to build the future; it receives a
/// short-lived [`Locals`] and a `'static` [`AsyncSequence`] proxy. Move the
/// proxy into the returned future and call `.await` on its async methods to
/// suspend the sequence.
pub fn async_sequence<'gc, F>(
    mc: &Mutation<'gc>,
    create: impl FnOnce(Locals<'gc, '_>, AsyncSequence) -> F,
) -> BoxSequence<'gc>
where
    F: Future<Output = Result<SequenceReturn, StashedError>> + 'static,
{
    let shared = Rc::new(SharedSlot::new());
    let roots = DynamicRootSet::new(mc);
    let fut = create(
        Locals {
            roots,
            _marker: PhantomData,
        },
        AsyncSequence {
            shared: shared.clone(),
        },
    );
    BoxSequence::new(mc, SequenceImpl { shared, roots, fut })
}

/// Terminal action produced by an `async_sequence` future. Mirrors the
/// "tail" variants of [`SequencePoll`] but takes stashed values since the
/// future runs across multiple `'gc` brands.
pub enum SequenceReturn {
    /// Stack values are the sequence's results — return to caller.
    Return,
    /// Tail-call this function with stack values as args.
    Call(StashedFunction),
    /// Tail-yield stack values to the resumer (or to `to_thread`).
    Yield(Option<StashedThread>),
    /// Tail-resume `thread` with stack values as args.
    Resume(StashedThread),
}

/// Proxy held inside an `async_sequence` future. `'static` — must not be
/// stored or moved out of the future.
pub struct AsyncSequence {
    shared: Rc<SharedSlot>,
}

impl AsyncSequence {
    /// Re-acquire the GC context for a synchronous block. Use this to
    /// stash `'gc` values into [`Locals`] or to inspect the call stack.
    pub fn enter<F, R>(&mut self, f: F) -> R
    where
        F: for<'gc> FnOnce(Context<'gc>, Locals<'gc, '_>, Execution<'gc, '_>, Stack<'gc, '_>) -> R,
    {
        self.shared.visit(move |shared| {
            // SAFETY: Re-borrowing the stack vec for the closure's
            // duration. `visit` already enforces the slot's lifetime.
            let stack = Stack::new(shared.stack_buf, shared.stack_bottom);
            f(
                shared.ctx,
                Locals {
                    roots: shared.roots,
                    _marker: PhantomData,
                },
                shared.exec,
                stack,
            )
        })
    }

    /// `enter` variant that turns an `Error<'gc>` into a `StashedError`
    /// owned by this sequence's root set.
    pub fn try_enter<F, R>(&mut self, f: F) -> Result<R, StashedError>
    where
        F: for<'gc> FnOnce(
            Context<'gc>,
            Locals<'gc, '_>,
            Execution<'gc, '_>,
            Stack<'gc, '_>,
        ) -> Result<R, Error<'gc>>,
    {
        self.shared.visit(move |shared| {
            let mc = shared.ctx.mutation();
            let stack = Stack::new(shared.stack_buf, shared.stack_bottom);
            f(
                shared.ctx,
                Locals {
                    roots: shared.roots,
                    _marker: PhantomData,
                },
                shared.exec,
                stack,
            )
            .map_err(|e| Stashable::stash(e, mc, shared.roots))
        })
    }

    /// Pause this sequence and let the executor make progress on other
    /// work. Resumes on the next pump.
    pub async fn pending(&mut self) {
        self.shared.visit(|shared| {
            shared.set_next_op(SequenceOp::Pending);
        });
        wait_once().await;
        self.shared.visit(|shared| {
            assert!(
                shared.error.is_none(),
                "SequencePoll::Pending cannot be followed by an error"
            );
        });
    }

    /// Call `func` (relative `bottom` from current stack base). When the
    /// call returns, the sequence is re-polled and stack values at
    /// `bottom..` are the call's results.
    pub async fn call(
        &mut self,
        func: &StashedFunction,
        bottom: usize,
    ) -> Result<(), StashedError> {
        self.shared.visit(|shared| {
            shared.set_next_op(SequenceOp::Call {
                function: func.fetch(shared.roots),
                bottom,
            });
        });
        wait_once().await;
        self.shared.visit(|shared| {
            let mc = shared.ctx.mutation();
            if let Some(err) = shared.error.take() {
                Err(Stashable::stash(err, mc, shared.roots))
            } else {
                Ok(())
            }
        })
    }

    /// Yield stack values starting at `bottom`. On resumption, resume-args
    /// are placed at `bottom..` for the sequence to consume.
    pub async fn lua_yield(
        &mut self,
        to_thread: Option<&StashedThread>,
        bottom: usize,
    ) -> Result<(), StashedError> {
        self.shared.visit(|shared| {
            shared.set_next_op(SequenceOp::Yield {
                to_thread: to_thread.map(|t| t.fetch(shared.roots)),
                bottom,
            });
        });
        wait_once().await;
        self.shared.visit(|shared| {
            let mc = shared.ctx.mutation();
            if let Some(err) = shared.error.take() {
                Err(Stashable::stash(err, mc, shared.roots))
            } else {
                Ok(())
            }
        })
    }

    /// Resume `thread` with stack values starting at `bottom`. On the
    /// thread yielding/returning, those values appear at `bottom..`.
    pub async fn resume(
        &mut self,
        thread: &StashedThread,
        bottom: usize,
    ) -> Result<(), StashedError> {
        self.shared.visit(|shared| {
            shared.set_next_op(SequenceOp::Resume {
                thread: thread.fetch(shared.roots),
                bottom,
            });
        });
        wait_once().await;
        self.shared.visit(|shared| {
            let mc = shared.ctx.mutation();
            if let Some(err) = shared.error.take() {
                Err(Stashable::stash(err, mc, shared.roots))
            } else {
                Ok(())
            }
        })
    }
}

/// Per-sequence stash handle. Use `stash` from inside `AsyncSequence::enter`
/// to convert a `'gc` value into a `'static` handle owned by the future.
#[derive(Copy, Clone)]
pub struct Locals<'gc, 'a> {
    roots: DynamicRootSet<'gc>,
    /// Bind a sham lifetime so handles can't escape an `enter` call.
    _marker: PhantomData<&'a ()>,
}

impl<'gc, 'a> Locals<'gc, 'a> {
    /// Stash a `'gc` value into the sequence's root set, returning a
    /// `'static` handle that lives as long as the sequence.
    pub fn stash<S: Stashable<'gc>>(&self, mc: &Mutation<'gc>, s: S) -> S::Stashed {
        s.stash(mc, self.roots)
    }

    /// Recover the live `'gc` value behind a previously-stashed handle.
    pub fn fetch<F: Fetchable>(&self, local: &F) -> F::Fetched<'gc> {
        local.fetch(self.roots)
    }
}

// ---------------------------------------------------------------------------
// SequenceImpl
// ---------------------------------------------------------------------------

struct SequenceImpl<'gc, F> {
    shared: Rc<SharedSlot>,
    roots: DynamicRootSet<'gc>,
    fut: F,
}

unsafe impl<'gc, F: 'static> Collect<'gc> for SequenceImpl<'gc, F> {
    const NEEDS_TRACE: bool = true;
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.roots);
    }
}

impl<'gc, F> SequenceImpl<'gc, F>
where
    F: Future<Output = Result<SequenceReturn, StashedError>> + 'static,
{
    fn poll_fut(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        exec: Execution<'gc, '_>,
        stack: Stack<'gc, '_>,
        error: Option<Error<'gc>>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        // SAFETY: structural pinning for `fut`; we don't move it. The
        // other fields (`shared`, `roots`) we touch only by value/copy.
        let this = unsafe { self.get_unchecked_mut() };
        let SequenceImpl { shared, roots, fut } = this;
        let roots_local = *roots;

        let (stack_buf, stack_bottom) = stack.into_parts();

        let mut next_op: Option<SequenceOp<'gc>> = None;

        let res = shared.with(
            &mut Shared {
                roots: roots_local,
                ctx,
                exec,
                stack_buf,
                stack_bottom,
                error,
                next_op: &mut next_op,
            },
            || {
                // SAFETY: pinning is structural for `fut`; we don't move
                // out of the reference. The user's future may not be
                // `Unpin`, so we use `Pin::new_unchecked`.
                unsafe {
                    Pin::new_unchecked(fut).poll(&mut task::Context::from_waker(&noop_waker()))
                }
            },
        );

        match res {
            Poll::Ready(res) => {
                assert!(
                    next_op.is_none(),
                    "AsyncSequence async method not `await`ed"
                );
                match res {
                    Ok(SequenceReturn::Return) => Ok(SequencePoll::Return),
                    Ok(SequenceReturn::Call(function)) => {
                        Ok(SequencePoll::TailCall(function.fetch(roots_local)))
                    }
                    Ok(SequenceReturn::Yield(to_thread)) => Ok(SequencePoll::TailYield(
                        to_thread.map(|t| t.fetch(roots_local)),
                    )),
                    Ok(SequenceReturn::Resume(thread)) => {
                        Ok(SequencePoll::TailResume(thread.fetch(roots_local)))
                    }
                    Err(stashed) => Err(stashed.fetch(roots_local)),
                }
            }
            Poll::Pending => Ok(
                match next_op.expect("`await` of a future other than AsyncSequence methods") {
                    SequenceOp::Pending => SequencePoll::Pending,
                    SequenceOp::Call { function, bottom } => {
                        SequencePoll::Call { function, bottom }
                    }
                    SequenceOp::Yield { to_thread, bottom } => {
                        SequencePoll::Yield { to_thread, bottom }
                    }
                    SequenceOp::Resume { thread, bottom } => {
                        SequencePoll::Resume { thread, bottom }
                    }
                },
            ),
        }
    }
}

impl<'gc, F> Sequence<'gc> for SequenceImpl<'gc, F>
where
    F: Future<Output = Result<SequenceReturn, StashedError>> + 'static,
{
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        // Delegate to DynamicRootSet's own trace through the standard
        // Collect impl. We hop through a Sized adapter so we can call
        // the generic Collect::trace method.
        struct Adapter<'a, 'gc>(&'a mut dyn Trace<'gc>);
        impl<'a, 'gc> Trace<'gc> for Adapter<'a, 'gc> {
            fn trace_gc(&mut self, gc: crate::dmm::Gc<'gc, ()>) {
                self.0.trace_gc(gc)
            }
            fn trace_gc_weak(&mut self, gc: crate::dmm::GcWeak<'gc, ()>) {
                self.0.trace_gc_weak(gc)
            }
        }
        let mut a = Adapter(cc);
        Collect::trace(&self.roots, &mut a);
    }

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        exec: Execution<'gc, '_>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        self.poll_fut(ctx, exec, stack, None)
    }

    fn error(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        exec: Execution<'gc, '_>,
        error: Error<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        self.poll_fut(ctx, exec, stack, Some(error))
    }
}

// ---------------------------------------------------------------------------
// SharedSlot — lifetime re-projection for poll
// ---------------------------------------------------------------------------

enum SequenceOp<'gc> {
    Pending,
    Call {
        function: Function<'gc>,
        bottom: usize,
    },
    Yield {
        to_thread: Option<Thread<'gc>>,
        bottom: usize,
    },
    Resume {
        thread: Thread<'gc>,
        bottom: usize,
    },
}

struct Shared<'gc, 'a> {
    roots: DynamicRootSet<'gc>,
    ctx: Context<'gc>,
    exec: Execution<'gc, 'a>,
    /// Live mutable view of the underlying value-stack vec; used to
    /// reconstruct a `Stack<'gc, '_>` per `enter` call.
    stack_buf: &'a mut Vec<Value<'gc>>,
    stack_bottom: usize,
    error: Option<Error<'gc>>,
    next_op: &'a mut Option<SequenceOp<'gc>>,
}

impl<'gc, 'a> Shared<'gc, 'a> {
    fn set_next_op(&mut self, op: SequenceOp<'gc>) {
        assert!(
            self.next_op.is_none(),
            "AsyncSequence async method not `await`ed before next op"
        );
        *self.next_op = Some(op);
    }
}

struct SharedSlot(Cell<*mut Shared<'static, 'static>>);

impl SharedSlot {
    fn new() -> Self {
        Self(Cell::new(ptr::null_mut()))
    }

    /// Install `shared` into the slot for the duration of `f()`. The slot
    /// is unconditionally cleared on exit (drop guard).
    ///
    /// # Safety
    /// The transmuted `'static` lifetimes here are an erasure trick. Inside
    /// `f` we may only access the slot via [`SharedSlot::visit`], which
    /// re-introduces the original lifetimes via HRTB.
    fn with<'gc, 'a, R>(&self, shared: &mut Shared<'gc, 'a>, f: impl FnOnce() -> R) -> R {
        unsafe {
            self.0.set(mem::transmute::<
                *mut Shared<'_, '_>,
                *mut Shared<'static, 'static>,
            >(shared));
        }

        struct Guard<'a>(&'a SharedSlot);
        impl<'a> Drop for Guard<'a> {
            fn drop(&mut self) {
                self.0.0.set(ptr::null_mut());
            }
        }
        let _guard = Guard(self);
        f()
    }

    /// Re-project the slot's lifetimes back into a fresh `Shared<'gc, 'a>`
    /// and call `f` with it. Panics if the slot is unset (called outside
    /// of a poll).
    fn visit<R>(&self, f: impl for<'gc, 'a> FnOnce(&'a mut Shared<'gc, 'a>) -> R) -> R {
        unsafe {
            let shared =
                mem::transmute::<*mut Shared<'static, 'static>, *mut Shared<'_, '_>>(self.0.get());
            assert!(
                !shared.is_null(),
                "AsyncSequence shared slot unset (called outside of poll?)"
            );
            f(&mut *shared)
        }
    }
}

// ---------------------------------------------------------------------------
// noop waker + one-shot wait
// ---------------------------------------------------------------------------

fn noop_waker() -> Waker {
    const NOOP_RAW_WAKER: RawWaker = {
        const VTABLE: RawWakerVTable =
            RawWakerVTable::new(|_| NOOP_RAW_WAKER, |_| {}, |_| {}, |_| {});
        RawWaker::new(ptr::null(), &VTABLE)
    };
    // SAFETY: the vtable's clone/wake/wake_by_ref/drop are no-ops.
    unsafe { Waker::from_raw(NOOP_RAW_WAKER) }
}

async fn wait_once() {
    let mut done = false;
    poll_fn(move |_| {
        if done {
            Poll::Ready(())
        } else {
            done = true;
            Poll::Pending
        }
    })
    .await;
}
