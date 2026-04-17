use crate::dmm::{Collect, Gc, Mutation, RefLock};
use crate::env::function::{LuaClosure, Upvalue};
use crate::env::value::Value;
use crate::vm::interp::Continuation;

/// Copy wrapper stored in Value.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Thread<'gc>(Gc<'gc, RefLock<ThreadState<'gc>>>);

#[derive(Clone, Copy, PartialEq, Eq, Collect)]
#[collect(internal, require_static)]
pub enum ThreadStatus {
    Suspended,
    Running,
    Normal,
    Dead,
}

/// A single call frame on the call stack.
/// Only represents Lua function execution — native calls don't push frames.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct CallFrame<'gc> {
    pub closure: Gc<'gc, LuaClosure<'gc>>,
    pub base: usize,
    pub pc: usize,
    pub num_results: u8,
    /// Fixup dispatched by `op_return` when this frame unwinds. `None` for
    /// normal calls; set by metamethod/iterator helpers that need
    /// post-return processing.
    #[collect(require_static)]
    pub continuation: Option<Continuation>,
}

/// The mutable state of a thread/coroutine.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ThreadState<'gc> {
    // TODO: custom collect impl to avoid unused stack slots keeping values alive.
    pub stack: Vec<Value<'gc>>,
    pub frames: Vec<CallFrame<'gc>>,
    pub open_upvalues: Vec<Upvalue<'gc>>,
    pub tbc_slots: Vec<usize>,
    pub status: ThreadStatus,
    /// Back-reference to the owning Thread handle, needed for creating open upvalues.
    pub thread_handle: Option<Thread<'gc>>,
}

impl<'gc> Thread<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let state = ThreadState {
            stack: Vec::new(),
            frames: Vec::new(),
            open_upvalues: Vec::new(),
            tbc_slots: Vec::new(),
            status: ThreadStatus::Suspended,
            thread_handle: None,
        };
        let thread = Thread(Gc::new(mc, RefLock::new(state)));
        // Store the back-reference
        thread.borrow_mut(mc).thread_handle = Some(thread);
        thread
    }

    pub fn borrow(self) -> std::cell::Ref<'gc, ThreadState<'gc>> {
        self.0.borrow()
    }

    pub fn borrow_mut(self, mc: &Mutation<'gc>) -> std::cell::RefMut<'gc, ThreadState<'gc>> {
        self.0.borrow_mut(mc)
    }

    pub fn status(self) -> ThreadStatus {
        self.0.borrow().status
    }

    pub fn inner(self) -> Gc<'gc, RefLock<ThreadState<'gc>>> {
        self.0
    }
}
