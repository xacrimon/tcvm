//! Multi-step / suspendable native callback machinery.
//!
//! A `Sequence<'gc>` is a heap-allocated state machine driven by the VM
//! executor. It is the foundation for both async-suspending native callbacks
//! (called by Lua) and host-side coroutine helpers like `pcall`/`xpcall`.
//!
//! The invariant lifetime story: `BoxSequence<'gc>` is rooted on a thread's
//! frame stack and traced as part of the normal GC walk. Each `poll` (or
//! `error`) call gets a fresh `'gc` brand, mirroring how a `Mutation<'gc>`
//! flows through `Lua::enter`.

use std::alloc::Allocator;
use std::pin::Pin;

use crate::dmm::{Collect, Gc, GcWeak, MetricsAlloc, Mutation, Trace};
use crate::env::error::Error;
use crate::env::function::Stack;
use crate::env::{Function, Thread};

/// What a [`Sequence::poll`] (or `error`) call requests of the executor next.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum SequencePoll<'gc> {
    /// Tick again on the next executor step. The driver loop returns to its
    /// scheduler so unrelated work (other coroutines, fuel checks) can run.
    Pending,
    /// Sequence is finished. Values at `bottom..` are its return values.
    Return,
    /// Call `function`; on completion, this sequence is polled again with the
    /// returned values placed at `bottom..` in its window.
    Call {
        function: Function<'gc>,
        bottom: usize,
    },
    /// Yield values at `bottom..` to `to_thread` (or to the resumer, if `None`).
    /// On resumption, this sequence is polled again with the resume-args at
    /// `bottom..`.
    Yield {
        to_thread: Option<Thread<'gc>>,
        bottom: usize,
    },
    /// Resume `thread` with values at `bottom..`. On the resumed thread
    /// yielding/returning, this sequence is polled with those values at
    /// `bottom..`.
    Resume { thread: Thread<'gc>, bottom: usize },
    /// Tail call; this sequence is consumed and the call's results go to the
    /// sequence's caller, not back to this sequence.
    TailCall(Function<'gc>),
    /// Tail yield; sequence consumed.
    TailYield(Option<Thread<'gc>>),
    /// Tail resume; sequence consumed.
    TailResume(Thread<'gc>),
}

/// What a native callback requests of the executor on return.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub enum CallbackAction<'gc> {
    /// Plain synchronous return. Stack values above `bottom` are the results.
    Return,
    /// Become a multi-step sequence. The pushed sequence will be polled
    /// repeatedly until it completes / yields / resumes.
    Sequence(BoxSequence<'gc>),
    /// Call `function`; on its return, optionally hand control to a follow-up
    /// sequence. Without `then`, the callback's caller receives the call's
    /// results directly.
    Call {
        function: Function<'gc>,
        then: Option<BoxSequence<'gc>>,
    },
    /// Yield to the resumer (or to a specific thread). Optional follow-up
    /// sequence runs on the resumption with the resume-args on the stack.
    Yield {
        to_thread: Option<Thread<'gc>>,
        then: Option<BoxSequence<'gc>>,
    },
    /// Resume `thread`. Optional follow-up sequence runs on the inner thread
    /// yielding/returning with those values on the stack.
    Resume {
        thread: Thread<'gc>,
        then: Option<BoxSequence<'gc>>,
    },
}

/// Read-only view of the executor that is passed to a native callback or
/// sequence poll. Carries the currently-running thread; richer fields
/// (full `&[Thread<'gc>]` thread stack, fuel handle) aren't implemented
/// yet.
#[derive(Clone, Copy)]
pub struct Execution<'gc, 'a> {
    current_thread: Thread<'gc>,
    _marker: std::marker::PhantomData<&'a Thread<'gc>>,
}

impl<'gc, 'a> Execution<'gc, 'a> {
    pub fn new(current_thread: Thread<'gc>) -> Self {
        Execution {
            current_thread,
            _marker: std::marker::PhantomData,
        }
    }

    /// Thread the native callback / sequence is running on top of.
    pub fn current_thread(self) -> Thread<'gc> {
        self.current_thread
    }
}

/// A multi-step / suspendable native callback.
///
/// Implementations must trace any `Gc` pointers they hold via
/// [`Sequence::trace_pointers`]; the [`seq_trace_pointers!`] macro provides the
/// usual delegation to a derived `Collect` impl.
///
/// `poll` is invoked by the executor once per scheduler step. Returning
/// [`SequencePoll::Pending`] yields the executor; any of the action variants
/// transfer control elsewhere (calling, yielding, resuming) and the sequence
/// is re-polled when control returns. The terminal variants `Return`,
/// `TailCall`, `TailYield`, `TailResume` consume the sequence.
pub trait Sequence<'gc>: 'gc {
    /// Trace held `Gc` pointers. `Collect::trace` is generic over the
    /// `Trace` impl which precludes vtable dispatch, so we expose this
    /// concrete-`dyn` shim and call it from the manual `Collect` impl on
    /// [`BoxSequence`].
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>);

    fn poll(
        self: Pin<&mut Self>,
        ctx: crate::lua::Context<'gc>,
        exec: Execution<'gc, '_>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>>;

    fn error(
        self: Pin<&mut Self>,
        _ctx: crate::lua::Context<'gc>,
        _exec: Execution<'gc, '_>,
        err: Error<'gc>,
        _stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        Err(err)
    }
}

/// Helper macro for `Sequence::trace_pointers`: wraps a `&mut dyn Trace<'gc>`
/// into a `Sized` adapter, then delegates to the type's derived
/// `Collect::trace`. Call as `seq_trace_pointers!(self, cc)`.
#[macro_export]
macro_rules! seq_trace_pointers {
    ($self:expr, $cc:expr) => {{
        struct __DynTraceAdapter<'__a, '__gc>(&'__a mut dyn $crate::dmm::Trace<'__gc>);
        impl<'__a, '__gc> $crate::dmm::Trace<'__gc> for __DynTraceAdapter<'__a, '__gc> {
            #[inline]
            fn trace_gc(&mut self, gc: $crate::dmm::Gc<'__gc, ()>) {
                self.0.trace_gc(gc)
            }
            #[inline]
            fn trace_gc_weak(&mut self, gc: $crate::dmm::GcWeak<'__gc, ()>) {
                self.0.trace_gc_weak(gc)
            }
        }
        let mut __adapter = __DynTraceAdapter($cc);
        $crate::dmm::Collect::trace($self, &mut __adapter);
    }};
}

/// Owning, pinned, GC-traced handle to a [`Sequence`]. Stored on a thread's
/// frame stack as part of `Frame::Sequence`.
///
/// The allocator is intentionally `MetricsAlloc<'static>` (not `'gc`-branded):
/// `MetricsAlloc`'s `'gc` brand is artificial — it carries only a `Metrics`
/// handle (`Clone`, no lifetime) and a `PhantomData<Invariant<'gc>>` lint —
/// while `Box::into_pin` requires `A: 'static`. We pin the static-erased
/// allocator on the inside; metering is unaffected.
pub struct BoxSequence<'gc>(Pin<Box<dyn Sequence<'gc> + 'gc, MetricsAlloc<'static>>>);

unsafe impl<'gc> Collect<'gc> for BoxSequence<'gc> {
    const NEEDS_TRACE: bool = true;

    #[inline]
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        // Bridge generic `T: Trace<'gc>` -> `dyn Trace<'gc>`.
        let dyn_cc: &mut dyn Trace<'gc> = cc;
        self.0.as_ref().get_ref().trace_pointers(dyn_cc);
    }
}

impl<'gc> BoxSequence<'gc> {
    /// Construct a new boxed sequence in the GC arena's metered allocator.
    pub fn new<S: Sequence<'gc> + 'gc>(mc: &Mutation<'gc>, seq: S) -> Self {
        let alloc: MetricsAlloc<'static> = MetricsAlloc::from_metrics(mc.metrics().clone());
        let b: Box<S, MetricsAlloc<'static>> = Box::new_in(seq, alloc);
        // SAFETY: rust-stable `CoerceUnsized` on `Box<T, A>` doesn't yet support
        // alternate allocators; we coerce the raw pointer manually. The
        // resulting `Box<dyn ..., MetricsAlloc>` owns the same allocation
        // produced by `Box::new_in` above and reuses the same allocator.
        let (ptr, alloc): (*mut S, MetricsAlloc<'static>) = Box::into_raw_with_allocator(b);
        let dyn_box: Box<dyn Sequence<'gc> + 'gc, MetricsAlloc<'static>> =
            unsafe { Box::from_raw_in(ptr as *mut (dyn Sequence<'gc> + 'gc), alloc) };
        BoxSequence(Box::into_pin(dyn_box))
    }

    /// Drive one step of the sequence.
    pub fn poll(
        &mut self,
        ctx: crate::lua::Context<'gc>,
        exec: Execution<'gc, '_>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        self.0.as_mut().poll(ctx, exec, stack)
    }

    /// Hand a pending error to the sequence; it may convert it into a
    /// `SequencePoll::Return` or rethrow.
    pub fn error(
        &mut self,
        ctx: crate::lua::Context<'gc>,
        exec: Execution<'gc, '_>,
        err: Error<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        self.0.as_mut().error(ctx, exec, err, stack)
    }
}

// Suppress unused-warnings on imports kept for the public surface.
#[allow(dead_code)]
fn _hold_imports<'gc>(_: Gc<'gc, ()>, _: GcWeak<'gc, ()>, _: Box<(), MetricsAlloc<'gc>>) {
    let _ = std::alloc::Layout::new::<()>();
    let _ = std::marker::PhantomData::<dyn Allocator>;
}
