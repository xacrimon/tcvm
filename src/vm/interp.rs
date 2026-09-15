use crate::dmm::{Gc, Lock, Mutation, RefLock};
use crate::env::function::{
    FastCall, Function, FunctionKind, InlineCache, LuaClosure, NativeClosure, NativeContext, Stack,
    Upvalue, UpvalueState,
};
use crate::env::shape::{MetamethodBits, Shape};
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::{
    CallSite, Frame, LuaFrame, PendingAction, Thread, ThreadState, ThreadStatus,
};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, Op, UpValueDescriptor};
use crate::lua::Context;
use crate::vm::num::{self, op_arith_float, op_arith_int, op_bit_int};

static HANDLERS: [Handler; Op::COUNT] = Op::table([
    (Op::MOVE, op_move),
    (Op::LOAD, op_load),
    (Op::LFALSESKIP, op_lfalseskip),
    (Op::GETUPVAL, op_getupval),
    (Op::SETUPVAL, op_setupval),
    (Op::GETTABUP, op_gettabup),
    (Op::SETTABUP, op_settabup),
    (Op::GETTABLE, op_gettable),
    (Op::SETTABLE, op_settable),
    (Op::GETFIELD, op_getfield),
    (Op::SETFIELD, op_setfield),
    (Op::SELF, op_self),
    (Op::NEWTABLE, op_newtable),
    (Op::ADD, op_add),
    (Op::SUB, op_sub),
    (Op::MUL, op_mul),
    (Op::MOD, op_mod),
    (Op::POW, op_pow),
    (Op::DIV, op_div),
    (Op::IDIV, op_idiv),
    (Op::BAND, op_band),
    (Op::BOR, op_bor),
    (Op::BXOR, op_bxor),
    (Op::SHL, op_shl),
    (Op::SHR, op_shr),
    (Op::UNM, op_unm),
    (Op::BNOT, op_bnot),
    (Op::NOT, op_not),
    (Op::LEN, op_len),
    (Op::CONCAT, op_concat),
    (Op::CLOSE, op_close),
    (Op::TBC, op_tbc),
    (Op::JMP, op_jmp),
    (Op::EQ, op_eq),
    (Op::LT, op_lt),
    (Op::LE, op_le),
    (Op::TEST, op_test),
    (Op::TESTSET, op_testset),
    (Op::CALL, op_call),
    (Op::TAILCALL, op_tailcall),
    (Op::RETURN, op_return),
    (Op::FORLOOP, op_forloop),
    (Op::FORPREP, op_forprep),
    (Op::TFORPREP, op_tforprep),
    (Op::TFORCALL, op_tforcall),
    (Op::TFORLOOP, op_tforloop),
    (Op::SETLIST, op_setlist),
    (Op::CLOSURE, op_closure),
    (Op::VARARG, op_vararg),
    (Op::VARARGGET, op_varargget),
    (Op::VARARGPREP, op_varargprep),
    (Op::ERRNNIL, op_errnnil),
    (Op::NOP, op_nop),
    (Op::STOP, op_stop),
    (Op::ADDI, op_addi),
    (Op::SUBI, op_subi),
    (Op::MULI, op_muli),
    (Op::MODI, op_modi),
    (Op::POWI, op_powi),
    (Op::DIVI, op_divi),
    (Op::IDIVI, op_idivi),
    (Op::BANDI, op_bandi),
    (Op::BORI, op_bori),
    (Op::BXORI, op_bxori),
    (Op::SHLI, op_shli),
    (Op::SHRI, op_shri),
    (Op::RSUBI, op_rsubi),
    (Op::RMODI, op_rmodi),
    (Op::RPOWI, op_rpowi),
    (Op::RDIVI, op_rdivi),
    (Op::RIDIVI, op_ridivi),
    (Op::RSHLI, op_rshli),
    (Op::RSHRI, op_rshri),
    (Op::EQI, op_eqi),
    (Op::LTI, op_lti),
    (Op::LEI, op_lei),
    (Op::GTI, op_gti),
    (Op::GEI, op_gei),
]);

/// Why an opcode faulted. `impl_error` renders the reference message for
/// it and raises it as a Lua error on the current frame.
pub(crate) enum OpError<'gc> {
    Index(Value<'gc>),
    Call(Value<'gc>),
    Arith(Value<'gc>, Value<'gc>),
    Bitwise(Value<'gc>, Value<'gc>),
    Concat(Value<'gc>, Value<'gc>),
    Compare(Value<'gc>, Value<'gc>),
    Len(Value<'gc>),
    DivByZero,
    ModByZero,
    IndexChainLoop,
    NewIndexChainLoop,
    /// ERRNNIL: constant index of the global's name.
    GlobalRedefined(u16),
    ForStepZero,
    /// Which `for` control value (`"limit"`, `"step"`, `"initial value"`)
    /// failed to coerce, and the offending value.
    ForNotNumber(&'static str, Value<'gc>),
    NilIndex,
    NanIndex,
    Internal(&'static str),
}

pub(crate) type Registers<'gc, 'a> = *mut Value<'gc>;

/// Scratch state for one dispatch run. Lives on `run_thread`'s frame and is
/// threaded through every handler by `&mut`, so handlers can hand values to a
/// cold tail-callee without a stack object of their own (which would force a
/// frame onto the fast path). Nothing here outlives dispatch, so it holds
/// `Value`s without being traced: no collection can run while the `Context`
/// is live.
pub(crate) struct DispatchState<'gc> {
    /// Set by `raise!` right before it tail-calls `impl_error`, which takes it.
    fault: Option<OpError<'gc>>,
    /// The running closure's constant pool, upvalue array and inline-cache
    /// table, so `constant!`, `upvalue!` and `read_ic` are one load off `ds`
    /// instead of a five-deep chase through `thread.frames`. Valid because
    /// the frame keeps the closure alive; every site that changes the top
    /// Lua frame and keeps dispatching must `bind` the new closure alongside
    /// `registers`.
    constants: *const Value<'gc>,
    upvalues: *const Upvalue<'gc>,
    ic_table: *const Lock<InlineCache<'gc>>,
    /// The running Lua frame, i.e. `thread.frames.last()`, which is always a
    /// `Frame::Lua` while the interpreter runs. Points into `thread.frames`'
    /// buffer, so it is rebound (`bind_frame`) wherever the top frame changes
    /// — every push can reallocate the vec.
    frame: *mut LuaFrame<'gc>,
}

impl<'gc> DispatchState<'gc> {
    fn bind(&mut self, closure: &LuaClosure<'gc>) {
        self.constants = closure.proto.constants.as_ptr();
        self.upvalues = closure.upvalues.as_ptr();
        self.ic_table = closure.proto.ic_table.as_ptr();
    }

    /// Bind to the thread's top frame, which must be a Lua frame.
    #[inline(always)]
    fn bind_frame(&mut self, thread: &mut ThreadState<'gc>) {
        let frame: *mut LuaFrame<'gc> = unsafe { thread.top_lua_unchecked_mut() };
        self.frame = frame;
        self.bind(unsafe { &(*frame).closure });
    }

    #[inline(always)]
    fn base(&self) -> usize {
        unsafe { (*self.frame).base }
    }

    /// Persist the resume address of the running frame.
    #[inline(always)]
    fn save_pc(&mut self, ip: *const Instruction) {
        unsafe { (*self.frame).pc = ip }
    }

    /// Debug check that the cache matches the top frame's closure: catches a
    /// frame change that forgot to `bind`, which the bounds checks alone
    /// would not (a stale pool is usually long enough for the index).
    #[inline(always)]
    fn debug_assert_bound(&self, thread: &ThreadState<'gc>) {
        if cfg!(debug_assertions) {
            let frame = thread.top_lua().unwrap();
            debug_assert!(std::ptr::eq(self.frame, frame));
            let closure = &frame.closure;
            debug_assert!(std::ptr::eq(
                self.constants,
                closure.proto.constants.as_ptr()
            ));
            debug_assert!(std::ptr::eq(self.upvalues, closure.upvalues.as_ptr()));
            debug_assert!(std::ptr::eq(self.ic_table, closure.proto.ic_table.as_ptr()));
        }
    }
}

/// Every dispatch target (handler, slow path, continuation) carries
/// `#[rustc_align(32)]`: they are reached only by indirect `br`, and an
/// unaligned entry can leave the fetch unit starved for the whole handler —
/// measured as a 10x rise in dispatch bubbles and ~4% on primes when a code
/// change shifted the layout.
pub(crate) type Handler = for<'gc> extern "rust-preserve-none" fn(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
);

/// A pending fixup attached to a callee frame. When `op_return` sees this on
/// the current frame, it fills in `results_base` and `nret`, then tail-calls
/// `cont_resume`, which reads its own data from `thread.top_lua()`, pops the
/// frame, restores caller state, applies the payload-specific fixup, and
/// dispatches. `payload` fully determines the fixup, so no per-variant
/// function pointer is stored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Continuation {
    pub payload: ContinuationPayload,
    /// Stack index of the first returned value — written by `op_return`.
    pub results_base: usize,
    /// Number of values returned — written by `op_return`.
    pub nret: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ContinuationPayload {
    /// Place the first returned value (or Nil) into `R[dst]` of the caller.
    StoreResult { dst: u8 },
    /// Discard results; used by `__newindex`, `__close`.
    IgnoreResult,
    /// Coerce the first result to bool; if it matches (`!= inverted`), take a
    /// jump of `offset` from the caller's resumed ip.
    CondJump { offset: i32, inverted: bool },
    /// Generic-for: copy up to `count` results into `R[base+4..]`, nil-filling
    /// the shortfall.
    TForCall { base: u8, count: u8 },
}

macro_rules! helpers {
    ($instruction:expr, $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident) => {
        #[allow(unused_macros)]
        macro_rules! dispatch {
            () => {{
                unsafe {
                    #[cfg(debug_assertions)]
                    {
                        let frame = $thread.top_lua_unchecked();
                        debug_assert!(
                            $ip.offset_from_unsigned(frame.closure.proto.code.as_ptr())
                                < frame.closure.proto.code.len()
                        );
                    }
                    let _ = $instruction;
                    let instruction = *$ip;
                    let pos = instruction.opcode() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = *$handlers.cast::<Handler>().add(pos);
                    let ip = $ip.add(1);
                    become handler(instruction, $ctx, $thread, $registers, ip, $handlers, $ds);
                }
            }};
        }

        /// Keys a raw table write must refuse (`luaH_set`); an absent key
        /// only errors once it is actually stored, so `__newindex` still
        /// sees nil/NaN keys first.
        #[allow(unused_macros)]
        macro_rules! check_index_key {
            ($$k:expr) => {{
                let __k: Value<'gc> = $$k;
                if std::hint::unlikely(__k.is_nil()) {
                    raise!(OpError::NilIndex);
                }
                if std::hint::unlikely(__k.get_float().is_some_and(f64::is_nan)) {
                    raise!(OpError::NanIndex);
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! raise {
            ($$kind:expr) => {{
                std::hint::cold_path();
                $ds.fault = Some($$kind);
                become impl_error($instruction, $ctx, $thread, $registers, $ip, $handlers, $ds);
            }};
        }

        #[allow(unused_macros)]
        macro_rules! reg {
            ($$idx:expr) => {{ unsafe { $registers.add($$idx as usize).read() } }};

            (ref $$idx:expr) => {{ unsafe { &*$registers.add($$idx as usize) } }};

            (ref mut $$idx:expr) => {{ unsafe { &mut *$registers.add($$idx as usize) } }};
        }

        #[allow(unused_macros)]
        macro_rules! constant {
            ($$idx:expr) => {{
                unsafe {
                    $ds.debug_assert_bound($thread);
                    debug_assert!(
                        ($$idx as usize)
                            < $thread.top_lua_unchecked().closure.proto.constants.len()
                    );
                    *$ds.constants.add($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                unsafe {
                    $ds.debug_assert_bound($thread);
                    debug_assert!(
                        ($$idx as usize) < $thread.top_lua_unchecked().closure.upvalues.len()
                    );
                    *$ds.upvalues.add($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! skip {
            () => {{
                $ip = unsafe { $ip.add(1) };
            }};
        }

        /// `if $$cond { skip!() }` compiled as a *branch*. Left to itself
        /// LLVM if-converts the skip into a `csel` on `ip`, which makes the
        /// next instruction-word load — and so the whole next handler —
        /// data-dependent on the compared registers: ~15 cycles of exposed
        /// latency on every execution, versus a predicted branch that
        /// costs nothing when the outcome is stable and half a flush when
        /// it isn't. The opaque asm can't be speculated, so the path stays
        /// a real branch and `ip` merges through a phi, not a select.
        #[allow(unused_macros)]
        macro_rules! skip_if {
            ($$cond:expr) => {{
                if $$cond {
                    $ip = unsafe { $ip.add(1) };
                    // The pointer is only threaded through, never read.
                    #[allow(clippy::pointers_in_nomem_asm_block)]
                    unsafe {
                        core::arch::asm!("/* {0} */", inout(reg) $ip, options(nomem, nostack, preserves_flags));
                    }
                }
            }};
        }

        /// Schedule a metamethod (or iterator) call and dispatch into it.
        /// A Lua target dispatches into a fresh frame whose `op_return` resumes
        /// the continuation; a native target runs inline — synchronously the
        /// continuation payload fires immediately, otherwise the call suspends
        /// through the executor (`schedule_meta_call` installs the pending
        /// action / error frame). A non-callable target raises `$$err`, an
        /// `OpError` variant applied to the target: `Call` by default, but
        /// `__index`/`__newindex` chains pass `Index`, since the reference
        /// indexes a non-function metamethod value rather than calling it.
        #[allow(unused_macros)]
        macro_rules! invoke_metamethod {
            ($$meta:expr, $$args:expr, $$cont:expr) => {
                invoke_metamethod!($$meta, $$args, $$cont, Call)
            };
            ($$meta:expr, $$args:expr, $$cont:expr, $$err:ident) => {{
                let __mm_meta: Value<'gc> = $$meta;
                let __mm_cont: Continuation = $$cont;
                match schedule_meta_call($ctx, $thread, __mm_meta, $$args, __mm_cont, $ip) {
                    MetaDispatch::Lua { new_ip, new_base } => {
                        $ip = new_ip;
                        $ds.bind_frame($thread);
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
                        dispatch!();
                    }
                    MetaDispatch::NativeReturn { results_base, nret } => {
                        // The native ran inline with no pushed frame, so the
                        // caller is still on top — but staging / `invoke_native`
                        // may have reallocated the stack, so rebind `registers`
                        // (the Lua arm rebinds for the same reason) before the
                        // payload writes through it.
                        let (__cb, __ms) = {
                            let __f = $thread.top_lua().unwrap();
                            (__f.base, __f.closure.proto.max_stack_size as usize)
                        };
                        // The args/results were staged *above* the caller window
                        // (at `__cb + __ms + 1 ..`), and `invoke_native` never
                        // shrinks the shared stack, so the window the payload
                        // writes through is always in bounds. Assert that.
                        debug_assert!(
                            $thread.stack.len() >= __cb + __ms,
                            "native return left caller register window out of bounds"
                        );
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(__cb) };
                        apply_cont_payload!(
                            __mm_cont,
                            results_base,
                            nret,
                            $ctx,
                            $thread,
                            $registers,
                            $ip,
                            $handlers,
                            $ds
                        );
                    }
                    // Native target suspended (or errored): the executor will
                    // resume / unwind from the installed frame state.
                    MetaDispatch::Suspended => return,
                    MetaDispatch::Unresolvable => raise!(OpError::$$err(__mm_meta)),
                }
            }};
        }
    };
}

/// Applies a [`Continuation`]'s payload given its returned values at
/// `stack[results_base .. results_base + nret]`, then dispatches. Shared by
/// the Lua-return path (`cont_resume`, after popping the callee frame) and the
/// synchronous-native path (`invoke_metamethod!`, with the caller frame still
/// current). Expects `helpers!(...)` to have run in the enclosing handler so
/// `reg!` / `dispatch!` resolve; `$registers` / `$ip` must already be bound to
/// the *caller's* window.
macro_rules! apply_cont_payload {
    ($cont:expr, $results_base:expr, $nret:expr,
     $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident) => {{
        let __cont: Continuation = $cont;
        let __results_base: usize = $results_base;
        let __nret: usize = $nret as usize;
        match __cont.payload {
            ContinuationPayload::StoreResult { dst } => {
                let __r = if __nret > 0 {
                    $thread.stack[__results_base]
                } else {
                    Value::nil()
                };
                *reg!(ref mut dst) = __r;
                dispatch!();
            }
            ContinuationPayload::IgnoreResult => {
                dispatch!();
            }
            ContinuationPayload::CondJump { offset, inverted } => {
                let __r = if __nret > 0 {
                    $thread.stack[__results_base]
                } else {
                    Value::nil()
                };
                if !__r.is_falsy() != inverted {
                    $ip = unsafe { $ip.offset(offset as isize) };
                }
                dispatch!();
            }
            ContinuationPayload::TForCall { base, count } => {
                // Destination registers are `base+3 .. base+3+count`; they must
                // fit u8 register space, which bounds `count <= 253` (so the
                // `count + 1` in the suspend path can't overflow `u8`).
                debug_assert!(
                    base as usize + 3 + count as usize <= u8::MAX as usize + 1,
                    "TFORCALL destination range exceeds u8 register space",
                );
                let __to_copy = __nret.min(count as usize);
                for i in 0..__to_copy {
                    *reg!(ref mut base + 3 + i as u8) = $thread.stack[__results_base + i];
                }
                for i in __to_copy..count as usize {
                    *reg!(ref mut base + 3 + i as u8) = Value::nil();
                }
                dispatch!();
            }
        }
    }};
}

/// Inflates the slow-path body for "table get with metamethod".
/// Expects `helpers!(...)` to have been invoked in the enclosing handler
/// so `dispatch!`, `raise!`, `invoke_metamethod!`, and `reg!` resolve.
///
/// Steps:
///   1. Try a direct `raw_get`. Non-nil result is the answer.
///   2. Nil result + no `__index` → answer is nil.
///   3. Nil result + `__index` → walk the chain. Resolved values
///      land in `$dst`; functions fire via `invoke_metamethod!`.
macro_rules! table_get_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $t:expr, $k:expr, $dst:expr) => {{
        let __t: Table<'gc> = $t;
        let __k: Value<'gc> = $k;
        let __dst_reg: u8 = $dst;

        let __v = __t.raw_get(__k);
        if !__v.is_nil() {
            *reg!(ref mut __dst_reg) = __v;
            dispatch!();
        }

        if !__t.shape().has_mm(MetamethodBits::INDEX) {
            *reg!(ref mut __dst_reg) = Value::nil();
            dispatch!();
        }

        // INDEX bit implies metatable is Some.
        let __mt = unsafe { __t.metatable().unwrap_unchecked() };
        match walk_index_chain(Value::table(__t), __mt, __k, $ctx.symbols().mm_index) {
            IndexChain::Resolved(__rv) => {
                *reg!(ref mut __dst_reg) = __rv;
                dispatch!();
            }
            IndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => {
                let __cont = Continuation {
                    payload: ContinuationPayload::StoreResult { dst: __dst_reg },
                    results_base: 0,
                    nret: 0,
                };
                invoke_metamethod!(__mm_func, &[__mm_recv, __k], __cont, Index);
            }
            IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
        }
    }};
}

/// Inflates the slow-path body for "userdata get via `__index`". Unlike
/// the table case, a userdata with no metatable or a nil `__index` is an
/// "attempt to index" error (`userdata_index_chain` returns `Err`), since
/// userdata has no raw indexing to fall back to. A key absent from the
/// `__index` table still resolves to nil.
macro_rules! userdata_get_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $u:expr, $recv:expr, $k:expr, $dst:expr) => {{
        let __u = $u;
        let __recv: Value<'gc> = $recv;
        let __k: Value<'gc> = $k;
        let __dst_reg: u8 = $dst;

        match userdata_index_chain(__u, __recv, __k, $ctx.symbols().mm_index) {
            Err(()) => raise!(OpError::Index(__recv)),
            Ok(IndexChain::Resolved(__rv)) => {
                *reg!(ref mut __dst_reg) = __rv;
                dispatch!();
            }
            Ok(IndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            }) => {
                let __cont = Continuation {
                    payload: ContinuationPayload::StoreResult { dst: __dst_reg },
                    results_base: 0,
                    nret: 0,
                };
                invoke_metamethod!(__mm_func, &[__mm_recv, __k], __cont, Index);
            }
            Ok(IndexChain::Exhausted) => raise!(OpError::IndexChainLoop),
        }
    }};
}

/// Inflates the slow-path body for "table set with metamethod".
///
/// Walks the `__newindex` chain via `walk_newindex_chain`, which
/// returns either the table to raw-write into, or a callable to invoke.
macro_rules! table_set_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $t:expr, $k:expr, $v:expr) => {{
        let __t: Table<'gc> = $t;
        let __k: Value<'gc> = $k;
        let __new_val: Value<'gc> = $v;

        match walk_newindex_chain(__t, __k, $ctx.symbols().mm_newindex) {
            NewIndexChain::RawSet(__target) => {
                check_index_key!(__k);
                __target.raw_set($ctx, __k, __new_val);
                dispatch!();
            }
            NewIndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => {
                let __cont = Continuation {
                    payload: ContinuationPayload::IgnoreResult,
                    results_base: 0,
                    nret: 0,
                };
                invoke_metamethod!(__mm_func, &[__mm_recv, __k, __new_val], __cont, Index);
            }
            NewIndexChain::Exhausted => raise!(OpError::NewIndexChainLoop),
        }
    }};
}

// ---------------------------------------------------------------------------
// Inline cache helpers
// ---------------------------------------------------------------------------

/// Read the IC entry for the current call site. The handler must have
/// validated `ic_idx` came from a `GETFIELD`/`SETFIELD`/`GETTABUP`/
/// `SETTABUP` instruction whose prototype was assembled with a matching
/// `ic_table` length.
#[inline(always)]
fn read_ic<'gc>(
    ds: &DispatchState<'gc>,
    thread: &ThreadState<'gc>,
    ic_idx: u16,
) -> InlineCache<'gc> {
    // SAFETY: ic_idx is allocated at compile-time within the prototype's
    // IC count; debug-asserted in alloc_ic_slot's saturating_add.
    ds.debug_assert_bound(thread);
    debug_assert!(
        (ic_idx as usize)
            < unsafe { &thread.top_lua_unchecked().closure.proto }
                .ic_table
                .len()
    );
    unsafe { (*ds.ic_table.add(ic_idx as usize)).get() }
}

/// Refill the IC entry. Called by slow paths after they've done a full
/// shape lookup; subsequent same-shape accesses skip the slow path.
#[inline(always)]
fn fill_ic<'gc>(
    ctx: Context<'gc>,
    thread: &ThreadState<'gc>,
    ic_idx: u16,
    shape: Shape<'gc>,
    slot: u32,
) {
    let proto_gc = unsafe { thread.top_lua_unchecked().closure.proto };
    let value = InlineCache::Mono { shape, slot };
    if let Some(slot_lock) = proto_gc.ic_table.get(ic_idx as usize) {
        // We're adopting a fresh `Shape` Gc pointer through this slot
        // (transitively reachable from the parent `Prototype`), so emit
        // the backward barrier on the Prototype manually before writing
        // through `as_cell()` — `Lock::as_cell` is `unsafe` precisely
        // because it skips the automatic barrier `Lock::set` on
        // `Gc<Lock<T>>` would emit.
        ctx.mutation().backward_barrier(Gc::erase(proto_gc), None);
        unsafe { slot_lock.as_cell() }.set(value);
    }
}

/// Verify a cached IC entry against the live shape. Returns `Some(slot)`
/// on a fresh hit, `None` on miss. Metatable-mutation staleness is
/// handled downstream by `Shape::has_mm` — see `InlineCache`.
#[inline(always)]
fn ic_check<'gc>(cache: InlineCache<'gc>, live_shape: Shape<'gc>) -> Option<u32> {
    if let InlineCache::Mono { shape, slot } = cache
        && Shape::ptr_eq(live_shape, shape)
    {
        return Some(slot);
    }
    None
}

/// Fill the IC entry from the table's *current* shape + slot for the
/// given constant key. Called at the start of constant-key slow paths
/// so subsequent same-shape accesses can take the fast path. For SET
/// paths that end up transitioning `t`'s shape (fresh-key write with no
/// `__newindex`), this leaves a one-step-stale IC entry that the next
/// access fixes up — acceptable on cold paths.
#[inline(always)]
fn fill_ic_for_constant_key<'gc>(
    ctx: Context<'gc>,
    thread: &ThreadState<'gc>,
    ic_idx: u16,
    t: Table<'gc>,
    k: Value<'gc>,
) {
    // GETFIELD/SETFIELD/GETTABUP/SETTABUP only carry constant string keys.
    debug_assert!(
        k.get_string().is_some(),
        "IC fill on non-string key — compiler invariant violation"
    );
    let Some(key_str) = k.get_string() else {
        return;
    };
    let state = t.inner().borrow();
    let shape = state.shape();
    let slot = shape.find_slot(key_str).unwrap_or(InlineCache::ABSENT_SLOT);
    drop(state);
    fill_ic(ctx, thread, ic_idx, shape, slot);
}

/// Drive the VM on `thread` until the top-level frame returns.
///
/// The caller must have seeded the thread with at least one `LuaFrame`,
/// sized `stack` to at least `base + max_stack_size`, and placed
/// the callee + arguments at `stack[base-1..]`. See `Executor::start`.
#[inline(never)]
pub(crate) fn run_thread<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) {
    let mut ts = thread.borrow_mut(ctx.mutation());
    let mut ds = DispatchState {
        fault: None,
        constants: std::ptr::null(),
        upvalues: std::ptr::null(),
        ic_table: std::ptr::null(),
        frame: std::ptr::null_mut(),
    };
    ts.top_lua()
        .expect("run_thread requires a seeded Lua frame");
    ds.bind_frame(&mut ts);
    let (ip, base) = unsafe { ((*ds.frame).pc, (*ds.frame).base) };
    let registers = unsafe { ts.stack.as_mut_ptr().add(base) };
    let handlers = HANDLERS.as_ptr() as *const ();
    op_nop(
        Instruction::nop(),
        ctx,
        &mut ts,
        registers,
        ip,
        handlers,
        &mut ds,
    );
    // The fast paths leave dead scratch above the live region (`set_top_dirty`);
    // cut it off before anything outside the interpreter can trace this thread.
    ts.trim_dead();
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Cold tail of `raise!`: publish the faulting frame's pc, then raise the
/// reference-formatted message at level 1 (the faulting Lua frame itself)
/// so the executor's unwinder can route it to a catcher. A handler so it can
/// be `become`d: a plain call here would put a frame on every raising
/// handler's fast path.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn impl_error<'gc>(
    _instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    ip: *const Instruction,
    _handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    let kind = ds.fault.take().expect("impl_error without a pending fault");
    // `ip` already points past the faulting instruction (see `dispatch!`),
    // which is the convention `LuaFrame::pc` uses.
    save_pc(thread, ip);
    let msg = crate::vm::debug::op_error_message(ctx, thread, kind);
    thread.raise(ctx, crate::env::Error::from_str(ctx, &msg));
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_move<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src) = instruction.ab();
    *reg!(ref mut dst) = reg!(src);
    dispatch!();
}

/// Load constant from the current prototype's constant pool.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_load<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, idx) = instruction.ad();
    *reg!(ref mut dst) = constant!(idx);
    dispatch!();
}

/// Set register to false and skip the next instruction.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_lfalseskip<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let src = instruction.a();
    *reg!(ref mut src) = Value::boolean(false);
    skip!();
    dispatch!();
}

// ---------------------------------------------------------------------------
// Upvalue access
// ---------------------------------------------------------------------------

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_getupval<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, idx) = instruction.ab();
    let uv = upvalue!(idx);
    *reg!(ref mut dst) = read_upvalue(thread, uv);
    dispatch!();
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_setupval<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, idx) = instruction.ab();
    let val = reg!(src);
    let uv = upvalue!(idx);
    write_upvalue(ctx.mutation(), thread, uv, val);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table access via upvalue
// ---------------------------------------------------------------------------

/// R[dst] = UpValue[idx][K[key]]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_gettabup<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, idx, ic_idx, _key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        raise!(OpError::Index(t_val));
    };

    let cache = read_ic(ds, thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let v = unsafe { t_state.property_at(slot) };
            if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
                drop(t_state);
                *reg!(ref mut dst) = v;
                dispatch!();
            }
        }
    }
    drop(t_state);
    become gettabup_slow(instruction, ctx, thread, registers, ip, handlers, ds);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn gettabup_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let Some(t) = t_val.get_table() else {
        raise!(OpError::Index(t_val));
    };
    let k = constant!(key);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_get_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, dst);
}

/// UpValue[idx][K[key]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_settabup<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        raise!(OpError::Index(t_val));
    };

    let v = reg!(src);
    let cache = read_ic(ds, thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let existing = unsafe { t_state.property_at(slot) };
            // __newindex fires only on currently-nil keys.
            if !(existing.is_nil() && t_state.shape().has_mm(MetamethodBits::NEWINDEX)) {
                drop(t_state);
                let mut state = t.inner().borrow_mut(ctx.mutation());
                state.properties[slot as usize] = v;
                state.maybe_update_mt_bit(constant!(key), v);
                dispatch!()
            }
        }
    }
    drop(t_state);
    become settabup_slow(instruction, ctx, thread, registers, ip, handlers, ds);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn settabup_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let Some(t) = t_val.get_table() else {
        raise!(OpError::Index(t_val));
    };
    let k = constant!(key);
    let v = reg!(src);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, v);
}

// ---------------------------------------------------------------------------
// Table access via register
// ---------------------------------------------------------------------------

/// R[dst] = R[table][R[key]]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_gettable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        become gettable_slow(instruction, ctx, thread, registers, ip, handlers, ds);
    };

    let k = reg!(key);
    let (v, need_index) = {
        let t_state = t.inner().borrow();
        let v = t_state.raw_get(k);
        let need = v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        become gettable_slow(instruction, ctx, thread, registers, ip, handlers, ds);
    }

    *reg!(ref mut dst) = v;
    dispatch!();
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn gettable_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, table, key) = instruction.abc();
    let recv = reg!(table);
    let k = reg!(key);
    if let Some(t) = recv.get_table() {
        table_get_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, dst);
    }
    let Some(u) = recv.get_userdata() else {
        raise!(OpError::Index(recv));
    };
    userdata_get_slow_body!(ctx, thread, registers, ip, handlers, ds, u, recv, k, dst);
}

/// R[table][R[key]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_settable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Index(reg!(table)));
    };

    let k = reg!(key);
    let v = reg!(src);
    let needs_newindex = {
        let t_state = t.inner().borrow();
        t_state.shape().has_mm(MetamethodBits::NEWINDEX) && t_state.raw_get(k).is_nil()
    };

    if needs_newindex {
        become settable_slow(instruction, ctx, thread, registers, ip, handlers, ds);
    }

    check_index_key!(k);
    let mut t_state = t.inner().borrow_mut(ctx.mutation());
    t_state.raw_set(ctx, k, v);
    dispatch!()
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn settable_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, table, key) = instruction.abc();
    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Index(reg!(table)));
    };
    let k = reg!(key);
    let v = reg!(src);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, v);
}

/// R[dst] = R[table][K[key_idx]]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_getfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, table, ic_idx, _key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        become getfield_slow(instruction, ctx, thread, registers, ip, handlers, ds);
    };

    let cache = read_ic(ds, thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let v = unsafe { t_state.property_at(slot) };
            if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
                drop(t_state);
                *reg!(ref mut dst) = v;
                dispatch!();
            }
        }
    }
    drop(t_state);
    become getfield_slow(instruction, ctx, thread, registers, ip, handlers, ds);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn getfield_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    if let Some(t) = recv.get_table() {
        fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
        table_get_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, dst);
    }
    let Some(u) = recv.get_userdata() else {
        raise!(OpError::Index(recv));
    };
    userdata_get_slow_body!(ctx, thread, registers, ip, handlers, ds, u, recv, k, dst);
}

/// R[table][K[key_idx]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_setfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, table, ic_idx, key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Index(reg!(table)));
    };

    let v = reg!(src);
    let cache = read_ic(ds, thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let existing = unsafe { t_state.property_at(slot) };
            if !(existing.is_nil() && t_state.shape().has_mm(MetamethodBits::NEWINDEX)) {
                drop(t_state);
                let mut state = t.inner().borrow_mut(ctx.mutation());
                state.properties[slot as usize] = v;
                state.maybe_update_mt_bit(constant!(key_idx), v);
                dispatch!()
            }
        }
    }
    drop(t_state);
    become setfield_slow(instruction, ctx, thread, registers, ip, handlers, ds);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn setfield_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, table, ic_idx, key_idx) = instruction.abde();
    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Index(reg!(table)));
    };
    let k = constant!(key_idx);
    let v = reg!(src);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, ds, t, k, v);
}

// ---------------------------------------------------------------------------
// SELF — method-call setup
// ---------------------------------------------------------------------------

/// Backs `obj:m(...)`. Writes the method into `R[dst]` and the
/// receiver into `R[dst+1]`. No inline cache for now — see the
/// instruction definition.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_self<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        // Non-table receiver (userdata method dispatch, or an error).
        become op_self_nontable(instruction, ctx, thread, registers, ip, handlers, ds);
    };

    let key = constant!(key_idx);
    let (method, need_index) = {
        let recv_state = recv.inner().borrow();
        let v = recv_state.raw_get(key);
        let need = v.is_nil() && recv_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        become op_self_slow(instruction, ctx, thread, registers, ip, handlers, ds);
    }

    *reg!(ref mut dst) = method;
    *reg!(ref mut (dst + 1)) = recv_val;
    dispatch!();
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_self_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        raise!(OpError::Index(recv_val));
    };
    let key = constant!(key_idx);

    // Reachable only when raw_get returned nil and the INDEX bit is set,
    // so we go straight to the chain walk.
    let mt = unsafe { recv.metatable().unwrap_unchecked() };
    match walk_index_chain(recv_val, mt, key, ctx.symbols().mm_index) {
        IndexChain::Resolved(method) => {
            *reg!(ref mut dst) = method;
            *reg!(ref mut (dst + 1)) = recv_val;
            dispatch!();
        }
        IndexChain::Invoke { func, receiver } => {
            // Functional __index. Pre-place self at dst+1; the
            // continuation writes the resolved method into dst.
            *reg!(ref mut (dst + 1)) = recv_val;
            let cont = Continuation {
                payload: ContinuationPayload::StoreResult { dst },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(func, &[receiver, key], cont, Index);
        }
        IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
    }
}

/// SELF on a non-table receiver. Userdata dispatches through its
/// metatable's `__index`; unlike GETTABLE, a missing method is *not* an
/// error here — we write `nil` into `R[dst]` and let the subsequent CALL
/// raise "attempt to call a nil value (method ...)", matching Lua. A
/// userdata with no metatable / nil `__index` likewise yields a nil
/// method (the CALL raises). Any other non-table receiver raises.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_self_nontable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(u) = recv_val.get_userdata() else {
        raise!(OpError::Index(recv_val));
    };
    let key = constant!(key_idx);

    let chain = match userdata_index_chain(u, recv_val, key, ctx.symbols().mm_index) {
        // No metatable or nil `__index`: nil method, CALL raises.
        Err(()) => IndexChain::Resolved(Value::nil()),
        Ok(c) => c,
    };
    match chain {
        IndexChain::Resolved(method) => {
            *reg!(ref mut dst) = method;
            *reg!(ref mut (dst + 1)) = recv_val;
            dispatch!();
        }
        IndexChain::Invoke { func, receiver } => {
            *reg!(ref mut (dst + 1)) = recv_val;
            let cont = Continuation {
                payload: ContinuationPayload::StoreResult { dst },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(func, &[receiver, key], cont, Index);
        }
        IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
    }
}

/// R[dst] = {}
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_newtable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let dst = instruction.a();
    *reg!(ref mut dst) = Value::table(Table::new(ctx));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Arithmetic and bitwise (register-register)
// ---------------------------------------------------------------------------

/// `R[dst] = R[lhs] <op> R[rhs]` for the arithmetic opcodes.
macro_rules! arith_handler {
    ($fn_name:ident, $slow_name:ident, $instr:ident, $num_kind:ty, $mm:ident) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, lhs, rhs) = instruction.abc();

            // Testing that the tags match *before* either arm lets the
            // second operand's check collapse into the first: inside the gate
            // `rhs.get_integer()` is implied by `lhs.get_integer()`. Halves the
            // integer path's tag compares (four to two).
            let lk = reg!(ref lhs).kind();
            if std::hint::likely(lk == reg!(ref rhs).kind()) {
                if std::hint::likely(lk == ValueKind::Integer) {
                    if let (Some(lhs), Some(rhs)) =
                        (reg!(ref lhs).get_integer(), reg!(ref rhs).get_integer())
                    {
                        if let Some(v) = op_arith_int::<$num_kind>(lhs, rhs) {
                            *reg!(ref mut dst) = v;
                            dispatch!();
                        }

                        raise!(if Op::$instr == Op::MOD {
                            OpError::ModByZero
                        } else {
                            OpError::DivByZero
                        });
                    }
                } else if lk == ValueKind::Float {
                    if let (Some(lhs), Some(rhs)) =
                        (reg!(ref lhs).get_float(), reg!(ref rhs).get_float())
                    {
                        *reg!(ref mut dst) = op_arith_float::<$num_kind>(lhs, rhs);
                        dispatch!();
                    }
                }
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers, ds);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_arith_mixed, $mm, Arith);
    };
}

/// `R[dst] = R[lhs] <op> R[rhs]` for the bitwise opcodes.
macro_rules! bit_handler {
    ($fn_name:ident, $slow_name:ident, $instr:ident, $num_kind:ty, $mm:ident) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, lhs, rhs) = instruction.abc();

            // same-type int/float
            if let (Some(lhs), Some(rhs)) =
                (reg!(ref lhs).get_integer(), reg!(ref rhs).get_integer())
            {
                *reg!(ref mut dst) = op_bit_int::<$num_kind>(lhs, rhs);
                dispatch!();
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers, ds);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_bit_mixed, $mm, Bitwise);
    };
}

macro_rules! binop_slow_handler {
    ($slow_name:ident, $instr:ident, $num_kind:ty, $num_mix_h:ident, $mm:ident, $err:ident) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $slow_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, lhs, rhs) = instruction.abc();
            let (lhs, rhs) = (reg!(lhs), reg!(rhs));
            binop_slow_body!(
                dst, lhs, rhs, $num_kind, $num_mix_h, $mm, $err, ctx, thread, registers, ip,
                handlers, ds
            );
        }
    };
}

/// The tail every binary-op slow path shares, given the operand *values*:
/// the int/float mixed arm, then the metamethod, then the type error.
/// Expects `helpers!` to have run in the enclosing handler.
macro_rules! binop_slow_body {
    ($dst:expr, $lhs:expr, $rhs:expr, $num_kind:ty, $num_mix_h:ident, $mm:ident, $err:ident,
     $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident) => {{
        let dst = $dst;
        let (lhs, rhs): (Value<'gc>, Value<'gc>) = ($lhs, $rhs);
        {
            let mixed = num::$num_mix_h::<$num_kind>(&lhs, &rhs);
            if std::hint::unlikely(mixed.is_some()) {
                if let Some(v) = mixed {
                    *reg!(ref mut dst) = v;
                    dispatch!();
                }
            }
        }

        let meta_fn = binop_metamethod(lhs, rhs, $ctx.symbols().$mm);
        if meta_fn.is_nil() {
            raise!(OpError::$err(lhs, rhs));
        }

        let cont = Continuation {
            payload: ContinuationPayload::StoreResult { dst },
            results_base: 0,
            nret: 0,
        };
        invoke_metamethod!(meta_fn, &[lhs, rhs], cont);
    }};
}

arith_handler!(op_add, op_add_slow, ADD, num::Add, mm_add);
arith_handler!(op_sub, op_sub_slow, SUB, num::Sub, mm_sub);
arith_handler!(op_mul, op_mul_slow, MUL, num::Mul, mm_mul);
arith_handler!(op_mod, op_mod_slow, MOD, num::Mod, mm_mod);
arith_handler!(op_pow, op_pow_slow, POW, num::Pow, mm_pow);
arith_handler!(op_div, op_div_slow, DIV, num::Div, mm_div);
arith_handler!(op_idiv, op_idiv_slow, IDIV, num::IDiv, mm_idiv);
bit_handler!(op_band, op_band_slow, BAND, num::BAnd, mm_band);
bit_handler!(op_bor, op_bor_slow, BOR, num::BOr, mm_bor);
bit_handler!(op_bxor, op_bxor_slow, BXOR, num::BXor, mm_bxor);
bit_handler!(op_shl, op_shl_slow, SHL, num::Shl, mm_shl);
bit_handler!(op_shr, op_shr_slow, SHR, num::Shr, mm_shr);

// ---------------------------------------------------------------------------
// Arithmetic and bitwise (register-immediate)
// ---------------------------------------------------------------------------

/// `R[dst] = R[src] <op> imm`, or `imm <op> R[src]` when `$swap`. The
/// immediate's kind is tested first (one `tbnz` on the instruction word) so
/// each arm has a single register tag check and a decode of one or two
/// instructions; the int/float mixes are handled inline, unlike the
/// register form, since the constant side is free to convert.
macro_rules! arith_imm_handler {
    ($fn_name:ident, $slow_name:ident, $instr:ident, $num_kind:ty, $mm:ident, $swap:expr) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, src, _) = instruction.abc_imm();
            let v = reg!(ref src);

            if std::hint::likely(instruction.imm_is_int()) {
                let k = instruction.imm_int();
                if std::hint::likely(v.kind() == ValueKind::Integer)
                    && let Some(i) = v.get_integer()
                {
                    let out = if $swap {
                        op_arith_int::<$num_kind>(k, i)
                    } else {
                        // The compiler never emits a zero integer divisor
                        // in this position, so no `n % 0` check.
                        debug_assert!(
                            !<$num_kind as num::ArithOp>::INT_ZERO_DIVISOR_INVALID || k != 0
                        );
                        Some(<$num_kind as num::ArithOp>::int(i, k))
                    };
                    if let Some(out) = out {
                        *reg!(ref mut dst) = out;
                        dispatch!();
                    }
                    raise!(if Op::$instr == Op::RMODI {
                        OpError::ModByZero
                    } else {
                        OpError::DivByZero
                    });
                } else if let Some(f) = v.get_float() {
                    let k = k as f64;
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    *reg!(ref mut dst) = op_arith_float::<$num_kind>(l, r);
                    dispatch!();
                }
            } else {
                let k = instruction.imm_float();
                if let Some(f) = v.get_float() {
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    *reg!(ref mut dst) = op_arith_float::<$num_kind>(l, r);
                    dispatch!();
                } else if let Some(i) = v.get_integer() {
                    let f = i as f64;
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    *reg!(ref mut dst) = op_arith_float::<$num_kind>(l, r);
                    dispatch!();
                }
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers, ds);
        }

        binop_imm_slow_handler!($slow_name, $num_kind, op_arith_mixed, $mm, Arith, $swap);
    };
}

/// `R[dst] = R[src] <op> imm` for the bitwise opcodes. The immediate is
/// always an integer; a float register goes through the slow path's exact
/// conversion.
macro_rules! bit_imm_handler {
    ($fn_name:ident, $slow_name:ident, $instr:ident, $num_kind:ty, $mm:ident, $swap:expr) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, src, _) = instruction.abc_imm();
            debug_assert!(instruction.imm_is_int());
            let k = instruction.imm_int();

            if let Some(i) = reg!(ref src).get_integer() {
                let (l, r) = if $swap { (k, i) } else { (i, k) };
                *reg!(ref mut dst) = op_bit_int::<$num_kind>(l, r);
                dispatch!();
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers, ds);
        }

        binop_imm_slow_handler!($slow_name, $num_kind, op_bit_mixed, $mm, Bitwise, $swap);
    };
}

macro_rules! binop_imm_slow_handler {
    ($slow_name:ident, $num_kind:ty, $num_mix_h:ident, $mm:ident, $err:ident, $swap:expr) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $slow_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (dst, src, flipped) = instruction.abc_imm();
            let (v, k) = (reg!(src), instruction.imm_value());
            let (lhs, rhs) = if $swap || flipped { (k, v) } else { (v, k) };
            binop_slow_body!(
                dst, lhs, rhs, $num_kind, $num_mix_h, $mm, $err, ctx, thread, registers, ip,
                handlers, ds
            );
        }
    };
}

arith_imm_handler!(op_addi, op_addi_slow, ADDI, num::Add, mm_add, false);
arith_imm_handler!(op_subi, op_subi_slow, SUBI, num::Sub, mm_sub, false);
arith_imm_handler!(op_muli, op_muli_slow, MULI, num::Mul, mm_mul, false);
arith_imm_handler!(op_modi, op_modi_slow, MODI, num::Mod, mm_mod, false);
arith_imm_handler!(op_powi, op_powi_slow, POWI, num::Pow, mm_pow, false);
arith_imm_handler!(op_divi, op_divi_slow, DIVI, num::Div, mm_div, false);
arith_imm_handler!(op_idivi, op_idivi_slow, IDIVI, num::IDiv, mm_idiv, false);
arith_imm_handler!(op_rsubi, op_rsubi_slow, RSUBI, num::Sub, mm_sub, true);
arith_imm_handler!(op_rmodi, op_rmodi_slow, RMODI, num::Mod, mm_mod, true);
arith_imm_handler!(op_rpowi, op_rpowi_slow, RPOWI, num::Pow, mm_pow, true);
arith_imm_handler!(op_rdivi, op_rdivi_slow, RDIVI, num::Div, mm_div, true);
arith_imm_handler!(op_ridivi, op_ridivi_slow, RIDIVI, num::IDiv, mm_idiv, true);
bit_imm_handler!(op_bandi, op_bandi_slow, BANDI, num::BAnd, mm_band, false);
bit_imm_handler!(op_bori, op_bori_slow, BORI, num::BOr, mm_bor, false);
bit_imm_handler!(op_bxori, op_bxori_slow, BXORI, num::BXor, mm_bxor, false);
bit_imm_handler!(op_shli, op_shli_slow, SHLI, num::Shl, mm_shl, false);
bit_imm_handler!(op_shri, op_shri_slow, SHRI, num::Shr, mm_shr, false);
bit_imm_handler!(op_rshli, op_rshli_slow, RSHLI, num::Shl, mm_shl, true);
bit_imm_handler!(op_rshri, op_rshri_slow, RSHRI, num::Shr, mm_shr, true);

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

/// R[dst] = -R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_unm<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(i.wrapping_neg());
        dispatch!();
    }
    if let Some(f) = val.get_float() {
        *reg!(ref mut dst) = Value::float(-f);
        dispatch!();
    }
    let meta_fn = unop_metamethod(val, ctx.symbols().mm_unm);
    if meta_fn.is_nil() {
        raise!(OpError::Arith(val, val));
    }
    let cont = Continuation {
        payload: ContinuationPayload::StoreResult { dst },
        results_base: 0,
        nret: 0,
    };
    // Lua passes the operand twice for unary metamethods (spec quirk).
    invoke_metamethod!(meta_fn, &[val, val], cont);
}

/// R[dst] = ~R[src]  (bitwise NOT)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_bnot<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(!i);
        dispatch!();
    }
    let meta_fn = unop_metamethod(val, ctx.symbols().mm_bnot);
    if meta_fn.is_nil() {
        raise!(OpError::Bitwise(val, val));
    }
    let cont = Continuation {
        payload: ContinuationPayload::StoreResult { dst },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(meta_fn, &[val, val], cont);
}

/// R[dst] = not R[src]  (logical NOT — always produces boolean)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_not<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    *reg!(ref mut dst) = Value::boolean(val.is_falsy());
    dispatch!();
}

/// R[dst] = #R[src]  (length)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_len<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src) = instruction.ab();
    let val = reg!(src);

    // Strings never consult __len; return byte length directly.
    if let Some(s) = val.get_string() {
        *reg!(ref mut dst) = Value::integer(s.len() as i64);
        dispatch!();
    }

    // Tables consult __len first; fall back to raw_len only if absent.
    let meta_fn = if let Some(t) = val.get_table() {
        let mm = t.get_metamethod(ctx.symbols().mm_len);
        if mm.is_nil() {
            *reg!(ref mut dst) = Value::integer(t.raw_len() as i64);
            dispatch!();
        }
        mm
    } else {
        raise!(OpError::Len(val))
    };

    let cont = Continuation {
        payload: ContinuationPayload::StoreResult { dst },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(meta_fn, &[val], cont);
}

/// R[dst] = R[lhs] .. R[rhs]  (string concatenation)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_concat<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, lhs, rhs) = instruction.abc();
    let a = reg!(lhs);
    let b = reg!(rhs);
    // Fast path: both coerce to strings/numbers.
    let mut buf = Vec::new();
    if num::coerce_to_str(&mut buf, a) && num::coerce_to_str(&mut buf, b) {
        *reg!(ref mut dst) = Value::string(LuaString::new(ctx, &buf));
        dispatch!();
    }
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_concat);
    if meta_fn.is_nil() {
        raise!(OpError::Concat(a, b));
    }
    let cont = Continuation {
        payload: ContinuationPayload::StoreResult { dst },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(meta_fn, &[a, b], cont);
}

// ---------------------------------------------------------------------------
// Upvalue / resource management
// ---------------------------------------------------------------------------

/// Close all upvalues >= R[start].
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_close<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let start = instruction.a();
    let base = ds.base();
    let start_idx = base + start as usize;
    close_upvalues(ctx.mutation(), thread, start_idx);
    close_tbc_vars(ctx.mutation(), thread, start_idx);
    dispatch!();
}

/// Mark R[val] as to-be-closed.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tbc<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let val = instruction.a();
    let base = ds.base();
    thread.tbc_slots.push(base + val as usize);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Jumps and conditionals
// ---------------------------------------------------------------------------

/// pc += offset
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_jmp<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let offset = instruction.imm();
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// if (R[lhs] == R[rhs]) != inverted then skip next instruction
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_eq<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let a = reg!(lhs);
    let b = reg!(rhs);
    if num::raw_eq(a, b) {
        // Primitive or pointer-equal — no metamethod consultation.
        skip_if!(!inverted);
        dispatch!();
    }

    // Lua 5.5: __eq fires only when both operands are the same non-primitive
    // type (tables or userdata) and raw equality fails.
    let try_meta = (a.kind() == ValueKind::Table && b.kind() == ValueKind::Table)
        || (a.kind() == ValueKind::Userdata && b.kind() == ValueKind::Userdata);
    if try_meta {
        let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_eq);
        if !meta_fn.is_nil() {
            let cont = Continuation {
                payload: ContinuationPayload::CondJump {
                    offset: 1,
                    inverted,
                },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(meta_fn, &[a, b], cont);
        }
    }

    // Not equal and no applicable metamethod.
    skip_if!(inverted);
    dispatch!();
}

/// if (R[lhs] < R[rhs]) != inverted then skip next instruction
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_lt<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some(num::lt_int_float(x, y))
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(num::lt_float_int(x, y))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x < y)
        } else {
            None
        }
    };

    if let Some(r) = primitive {
        skip_if!(r != inverted);
        dispatch!();
    }

    let (a, b) = (reg!(lhs), reg!(rhs));
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_lt);
    if meta_fn.is_nil() {
        raise!(OpError::Compare(a, b));
    }
    let cont = Continuation {
        payload: ContinuationPayload::CondJump {
            offset: 1,
            inverted,
        },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(meta_fn, &[a, b], cont);
}

/// if (R[lhs] <= R[rhs]) != inverted then skip next instruction
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_le<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
            Some(x <= y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(x <= y)
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some(num::le_int_float(x, y))
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(num::le_float_int(x, y))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x <= y)
        } else {
            None
        }
    };

    if let Some(r) = primitive {
        skip_if!(r != inverted);
        dispatch!();
    }

    let (a, b) = (reg!(lhs), reg!(rhs));
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_le);
    if meta_fn.is_nil() {
        raise!(OpError::Compare(a, b));
    }
    let cont = Continuation {
        payload: ContinuationPayload::CondJump {
            offset: 1,
            inverted,
        },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(meta_fn, &[a, b], cont);
}

/// `if (R[src] <cmp> imm) != inverted then skip`, with `$swap` putting the
/// immediate on the left. The four number pairings each get their exact
/// comparison (`lt_int_float` & co. for the mixed ones); anything else goes
/// to the metamethod with the immediate materialised as a `Value`.
macro_rules! cmp_imm_handler {
    ($fn_name:ident, $slow_name:ident, $mm:ident, $swap:expr,
     $ii:expr, $ff:expr, $if_:expr, $fi:expr) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (src, inverted) = instruction.ab_imm_flag();
            let v = reg!(ref src);

            let primitive: Option<bool> = if std::hint::likely(instruction.imm_is_int()) {
                let k = instruction.imm_int();
                if std::hint::likely(v.kind() == ValueKind::Integer)
                    && let Some(i) = v.get_integer()
                {
                    Some(if $swap { $ii(k, i) } else { $ii(i, k) })
                } else if let Some(f) = v.get_float() {
                    Some(if $swap { $if_(k, f) } else { $fi(f, k) })
                } else {
                    None
                }
            } else {
                let k = instruction.imm_float();
                if let Some(f) = v.get_float() {
                    Some(if $swap { $ff(k, f) } else { $ff(f, k) })
                } else if let Some(i) = v.get_integer() {
                    Some(if $swap { $fi(k, i) } else { $if_(i, k) })
                } else {
                    None
                }
            };

            if let Some(r) = primitive {
                skip_if!(r != inverted);
                dispatch!();
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers, ds);
        }

        /// Kept out of line so the fast path needs no stack frame for the
        /// metamethod call machinery.
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $slow_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
        ) {
            helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
            let (src, inverted) = instruction.ab_imm_flag();
            let (v, k) = (reg!(src), instruction.imm_value());
            let (a, b) = if $swap { (k, v) } else { (v, k) };
            let meta_fn = binop_metamethod(a, b, ctx.symbols().$mm);
            if meta_fn.is_nil() {
                raise!(OpError::Compare(a, b));
            }
            let cont = Continuation {
                payload: ContinuationPayload::CondJump {
                    offset: 1,
                    inverted,
                },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(meta_fn, &[a, b], cont);
        }
    };
}

cmp_imm_handler!(
    op_lti,
    op_lti_slow,
    mm_lt,
    false,
    |a, b| a < b,
    |a: f64, b: f64| a < b,
    num::lt_int_float,
    num::lt_float_int
);
cmp_imm_handler!(
    op_lei,
    op_lei_slow,
    mm_le,
    false,
    |a, b| a <= b,
    |a: f64, b: f64| a <= b,
    num::le_int_float,
    num::le_float_int
);
cmp_imm_handler!(
    op_gti,
    op_gti_slow,
    mm_lt,
    true,
    |a, b| a < b,
    |a: f64, b: f64| a < b,
    num::lt_int_float,
    num::lt_float_int
);
cmp_imm_handler!(
    op_gei,
    op_gei_slow,
    mm_le,
    true,
    |a, b| a <= b,
    |a: f64, b: f64| a <= b,
    num::le_int_float,
    num::le_float_int
);

/// `if (R[src] == imm) != inverted then skip`. Raw equality only: `__eq`
/// requires two tables or two userdata, and the immediate is a number.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_eqi<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, inverted) = instruction.ab_imm_flag();
    let v = reg!(ref src);

    let eq = if std::hint::likely(instruction.imm_is_int()) {
        let k = instruction.imm_int();
        if std::hint::likely(v.kind() == ValueKind::Integer)
            && let Some(i) = v.get_integer()
        {
            i == k
        } else if let Some(f) = v.get_float() {
            num::exact_float_to_int(f) == Some(k)
        } else {
            false
        }
    } else {
        let k = instruction.imm_float();
        if let Some(f) = v.get_float() {
            f == k
        } else if let Some(i) = v.get_integer() {
            num::exact_float_to_int(k) == Some(i)
        } else {
            false
        }
    };

    skip_if!(eq != inverted);
    dispatch!();
}

/// if (not R[src]) == inverted then skip next instruction
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_test<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, inverted) = instruction.ab_flag();
    let truthy = !reg!(src).is_falsy();
    skip_if!(truthy != inverted);
    dispatch!();
}

/// If (truthy(R[src]) == inverted) then skip next instruction;
/// otherwise R[dst] := R[src] and fall through. Matches Lua 5.5 TESTSET.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_testset<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, src, inverted) = instruction.abc_flag();
    let val = reg!(src);
    let truthy = !val.is_falsy();
    if truthy == inverted {
        skip_if!(true);
    } else {
        *reg!(ref mut dst) = val;
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Function calls
// ---------------------------------------------------------------------------

/// The native arm of CALL: run the callback inline and land its results at
/// `func_idx`. Expands inside a handler body (needs its `dispatch!`).
macro_rules! call_native {
    ($nc:expr, $func_idx:expr, $nargs:expr, $returns:expr, $base:expr,
     $ctx:ident, $thread:ident, $registers:ident, $ip:ident, $ds:ident) => {{
        let nc = $nc;
        let func_idx = $func_idx;
        let nargs = $nargs;
        let returns = $returns;
        let base = $base;
        {
            let args_base = func_idx + 1;
            let argc = if nargs == 0 {
                $thread.top - args_base
            } else {
                nargs as usize - 1
            };
            let action = match invoke_native($ctx, $thread, nc, args_base, argc) {
                Ok(a) => a,
                Err(err) => {
                    // Push Frame::Error so the executor's unwinder finds
                    // the nearest catching `Frame::Sequence` (e.g. the
                    // PCallSequence under coroutine.resume). Persist
                    // caller's pc first so re-entry would work if anything
                    // catches and resumes.
                    save_pc($thread, $ip);
                    $thread.raise($ctx, err);
                    return;
                }
            };
            match action {
                crate::vm::sequence::CallbackAction::Return => {
                    // Result count comes via the logical top, not Vec::len:
                    // `invoke_native` never shrinks the shared stack, so the
                    // caller's register window is still fully covered.
                    let retc = $thread.top - args_base;
                    // Place results at stack[func_idx..] following Lua convention.
                    let wanted = if returns == 0 {
                        retc
                    } else {
                        returns as usize - 1
                    };
                    let to_copy = retc.min(wanted);
                    for i in 0..to_copy {
                        $thread.stack[func_idx + i] = $thread.stack[args_base + i];
                    }
                    for i in to_copy..wanted {
                        $thread.stack[func_idx + i] = Value::nil();
                    }
                    // Publish the logical top. For MULTRET this is the dynamic
                    // count the next consumer reads. The stale donor copies the
                    // down-shift left above the results are dead scratch (a
                    // call's function always sits at the caller's first free
                    // register) and are dropped by `trim_dead` on exit.
                    $thread.set_top_dirty(func_idx + wanted);
                    $registers = unsafe { $thread.stack.as_mut_ptr().add(base) };
                    dispatch!();
                }
                action => {
                    // Suspension path: persist caller's pc, stash the
                    // action on the $thread for the executor to translate
                    // into frame ops, then exit the dispatch chain.
                    $ds.save_pc($ip);
                    $thread.pending_action = Some(PendingAction {
                        action,
                        call_site: CallSite {
                            bottom: args_base,
                            func_idx,
                            returns,
                            cont: None,
                        },
                    });
                    return;
                }
            }
        }
    }};
}

/// The Lua arm of CALL: push the callee's frame and continue in it. Shared by
/// `op_call` and the `__call` slow path; expands inside a handler body so it can
/// use that handler's `dispatch!`.
macro_rules! call_lua {
    ($closure:expr, $func_idx:expr, $nargs:expr, $returns:expr,
     $thread:ident, $registers:ident, $ip:ident, $ds:ident) => {{
        let closure = $closure;
        let new_base = $func_idx + 1;
        $ds.save_pc($ip);
        $thread.ensure_slots(new_base + closure.proto.max_stack_size as usize);
        // `nargs == 0` is the MULTRET sentinel: read the count from `thread.top`.
        let caller_provided = if $nargs == 0 {
            $thread.top - new_base
        } else {
            $nargs as usize - 1
        };
        let num_params = closure.proto.num_params as usize;
        for i in caller_provided..num_params {
            $thread.stack[new_base + i] = Value::nil();
        }
        let num_extras = if closure.proto.is_vararg {
            caller_provided.saturating_sub(num_params) as u32
        } else {
            0
        };
        $ip = closure.proto.code.as_ptr();
        $thread.push_lua(LuaFrame {
            closure,
            base: new_base,
            pc: $ip,
            num_results: $returns,
            num_extras,
            continuation: None,
        });
        $ds.bind_frame($thread);
        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
        dispatch!();
    }};
}

/// R[func], ..., R[func+returns-2] = R[func](R[func+1], ..., R[func+args-1])
///
/// Only the plain-function cases live here: a Lua closure is entered inline,
/// a native one is handed to `op_call_fast`/`op_call_native`, and anything
/// that needs the `__call` chain goes to `op_call_meta`. Keeping every call
/// out of this handler keeps it frameless.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_call<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (func, nargs, returns) = instruction.abc();
    if let Some(f) = reg!(func).get_function() {
        match f.inner().as_ref() {
            FunctionKind::Lua(closure) => {
                let func_idx = ds.base() + func as usize;
                call_lua!(
                    *closure, func_idx, nargs, returns, thread, registers, ip, ds
                );
            }
            FunctionKind::Native(nc) => {
                if nc.fast != FastCall::None {
                    become op_call_fast(instruction, ctx, thread, registers, ip, handlers, ds);
                }
                become op_call_native(instruction, ctx, thread, registers, ip, handlers, ds);
            }
        }
    }
    become op_call_meta(instruction, ctx, thread, registers, ip, handlers, ds);
}

/// CALL of a builtin with a `FastCall` kind: run the common shape inline,
/// writing the result straight into the function slot. Any other shape (arg
/// count, type) falls through to the full implementation via `op_call_native`.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_fast<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (func, nargs, returns) = instruction.abc();
    let fast = match reg!(func).get_function().map(|f| f.inner().as_ref()) {
        Some(FunctionKind::Native(nc)) => nc.fast,
        _ => FastCall::None,
    };
    if nargs == 2 {
        let a = reg!(func + 1);
        let result = match fast {
            FastCall::Sqrt => a.get_float().map(|x| Value::float(x.sqrt())),
            FastCall::Abs => match a.kind() {
                ValueKind::Integer => a.get_integer().map(|i| Value::integer(i.wrapping_abs())),
                ValueKind::Float => a.get_float().map(|x| Value::float(x.abs())),
                _ => None,
            },
            FastCall::Floor | FastCall::Ceil => match a.kind() {
                ValueKind::Integer => Some(a),
                ValueKind::Float => a.get_float().map(|x| {
                    let r = if fast == FastCall::Floor {
                        x.floor()
                    } else {
                        x.ceil()
                    };
                    crate::builtin::util::num_to_value(r)
                }),
                _ => None,
            },
            FastCall::None => None,
        };
        if let Some(result) = result {
            *reg!(ref mut func) = result;
            if returns == 0 {
                thread.set_top_dirty(ds.base() + func as usize + 1);
            } else {
                for i in 1..returns as usize - 1 {
                    *reg!(ref mut func as usize + i) = Value::nil();
                }
            }
            dispatch!();
        }
    }
    become op_call_native(instruction, ctx, thread, registers, ip, handlers, ds);
}

/// CALL of a plain native function (no `__call` chain involved).
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_native<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (func, nargs, returns) = instruction.abc();
    let base = ds.base();
    let func_idx = base + func as usize;
    let nc: &NativeClosure<'gc> = match reg!(func).get_function().map(|f| f.inner().as_ref()) {
        Some(FunctionKind::Native(nc)) => nc,
        _ => unreachable!("op_call_native on a non-native callee"),
    };
    call_native!(
        nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, ds
    );
}

/// The `__call` slow path of CALL: walk the metamethod chain, then enter
/// whatever it resolves to.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_meta<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (func, nargs, returns) = instruction.abc();
    let base = ds.base();
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!(OpError::Call(thread.stack[func_idx]));
    };

    match target {
        CallTarget::Lua(closure) => {
            call_lua!(closure, func_idx, nargs, returns, thread, registers, ip, ds);
        }
        CallTarget::Native(nc) => {
            call_native!(
                nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, ds
            );
        }
    }
}

/// return R[func](R[func+1], ..., R[func+args-1])  — tail call
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tailcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (func, nargs) = instruction.ab();
    let base = ds.base();
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!(OpError::Call(thread.stack[func_idx]));
    };

    match target {
        CallTarget::Lua(closure) => {
            // Results must land in this frame's *original* func slot in its
            // caller. A vararg frame's VARARGPREP shifted base past the extras
            // at `[base - num_extras .. base]`, so that slot is below them.
            let (cur_base, cur_num_extras) =
                unsafe { ((*ds.frame).base, (*ds.frame).num_extras as usize) };
            let caller_func_idx = cur_base - 1 - cur_num_extras;
            let new_base = caller_func_idx + 1;
            // Close upvalues before overwriting these slots with args, so an
            // open upvalue keeps referencing the local, not the arg value.
            close_upvalues(ctx.mutation(), thread, new_base);
            // `nargs == 0` is MULTRET: read the count from `thread.top`.
            let src_start = func_idx + 1;
            let nargs = if nargs == 0 {
                thread.top - src_start
            } else {
                nargs as usize - 1
            };
            for i in 0..nargs {
                thread.stack[new_base + i] = thread.stack[src_start + i];
            }
            ip = closure.proto.code.as_ptr();
            let frame = unsafe { &mut *ds.frame };
            frame.closure = closure;
            frame.pc = ip;
            frame.base = new_base;
            // num_extras feeds the new frame's own VARARGPREP; num_results is
            // left as-is (still the original caller's expectation).
            let num_params = closure.proto.num_params as usize;
            frame.num_extras = if closure.proto.is_vararg {
                nargs.saturating_sub(num_params) as u32
            } else {
                0
            };
            thread.ensure_slots(new_base + closure.proto.max_stack_size as usize);
            for i in nargs..num_params {
                thread.stack[new_base + i] = Value::nil();
            }
            ds.bind(&closure);
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        CallTarget::Native(nc) => {
            let args_base = func_idx + 1;
            let argc = if nargs == 0 {
                thread.top - args_base
            } else {
                nargs as usize - 1
            };
            let action = match invoke_native(ctx, thread, nc, args_base, argc) {
                Ok(a) => a,
                Err(err) => {
                    // Tailcall + native error: the message still names the
                    // tailcalling Lua frame (a native never really tail
                    // calls in the reference either), so locate it before
                    // popping that frame and installing the unwind marker.
                    save_pc(thread, ip);
                    let err = crate::vm::debug::locate(ctx, thread, err);
                    let cur_base = thread.top_lua().unwrap().base;
                    close_upvalues(ctx.mutation(), thread, cur_base);
                    close_tbc_vars(ctx.mutation(), thread, cur_base);
                    thread.frames.pop();
                    thread.frames.push(Frame::Error(err));
                    return;
                }
            };
            match action {
                crate::vm::sequence::CallbackAction::Return => {
                    // Result count via the logical top (the shared stack was
                    // never shrunk by the native call).
                    let retc = thread.top - args_base;
                    match frame_return(ctx.mutation(), thread, ds, args_base, retc) {
                        FrameReturn::Continuation => {
                            become cont_resume(
                                instruction,
                                ctx,
                                thread,
                                registers,
                                ip,
                                handlers,
                                ds,
                            );
                        }
                        FrameReturn::TopLevel => return,
                        FrameReturn::ToNonLua => return,
                        FrameReturn::Caller { new_base, new_ip } => {
                            ip = new_ip;
                            ds.bind_frame(thread);
                            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
                            dispatch!();
                        }
                    }
                }
                action => {
                    // Tailcall + suspension: pop the tailcalling Lua frame
                    // (close upvalues / TBC vars) so any subsequent action
                    // lands on the caller's frame window. Capture the
                    // popped frame's `num_results` — it carries the
                    // original caller's expectation across the tail call.
                    let (cur_base, num_results) = {
                        let f = thread.top_lua().unwrap();
                        (f.base, f.num_results)
                    };
                    close_upvalues(ctx.mutation(), thread, cur_base);
                    close_tbc_vars(ctx.mutation(), thread, cur_base);
                    thread.frames.pop();
                    thread.pending_action = Some(PendingAction {
                        action,
                        call_site: CallSite {
                            bottom: args_base,
                            func_idx: cur_base - 1,
                            returns: num_results,
                            cont: None,
                        },
                    });
                    return;
                }
            }
        }
    }
}

/// return R[values], ..., R[values+count-2]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (values, count) = instruction.ab();

    let cur_base = ds.base();
    let values_base = cur_base + values as usize;
    // `count == 0` is MULTRET: read the count from `thread.top`.
    let nret = if count == 0 {
        thread.top - values_base
    } else {
        count as usize - 1
    };

    match frame_return(ctx.mutation(), thread, ds, values_base, nret) {
        FrameReturn::Continuation => {
            become cont_resume(instruction, ctx, thread, registers, ip, handlers, ds);
        }
        FrameReturn::TopLevel => return,
        FrameReturn::ToNonLua => return,
        FrameReturn::Caller { new_base, new_ip } => {
            ip = new_ip;
            ds.bind_frame(thread);
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric for loop
// ---------------------------------------------------------------------------

/// Lua's `forlimit`: the limit of an integer loop, floored for an ascending
/// loop and ceiled for a descending one. `Some(None)` means the loop must not
/// run: either the limit already excludes `init`, or it lies beyond the i64
/// range on the side no integer can reach. A limit beyond the range on the
/// other side is clamped, since the loop then runs to the boundary. `None`
/// is a limit that isn't a number at all.
fn for_limit(init: i64, limit: Value, step: i64) -> Option<Option<i64>> {
    // Numeric strings become numbers first (`luaV_tointeger`'s `l_strton`).
    let num = match limit.get_string() {
        Some(s) => crate::builtin::util::str_to_number(s.as_bytes())?,
        None => limit,
    };
    let lim = if let Some(i) = num.get_integer() {
        i
    } else if let Some(f) = num.get_float() {
        let rounded = if step < 0 { f.ceil() } else { f.floor() };
        match num::exact_float_to_int(rounded) {
            Some(i) => i,
            // Out of range, infinite, or NaN. NaN fails `0 < f` and so is
            // treated as too negative, like the reference.
            None if 0.0 < f => {
                if step < 0 {
                    return Some(None);
                }
                i64::MAX
            }
            None => {
                if step > 0 {
                    return Some(None);
                }
                i64::MIN
            }
        }
    } else {
        return None;
    };
    let runs = if step > 0 { init <= lim } else { init >= lim };
    Some(runs.then_some(lim))
}

/// Prepare a numeric for. Before: R[base] = init, R[base+1] = limit,
/// R[base+2] = step. After: R[base] = the last value the control variable
/// takes (integer loop) or the limit (float loop), R[base+1] = step,
/// R[base+2] = the hidden control variable, R[base+3] = its visible copy.
/// Jumps past the body if the loop won't run.
///
/// The reference collapses this to two hidden slots by making the visible
/// variable the control variable. That is measurably slower here: the
/// control slot is a loop-carried load/add/store chain through memory, and
/// as soon as the body also loads that slot (any body that reads `i`) the
/// chain stops forwarding cheaply and costs several extra cycles per
/// iteration (`for i = 1, 2e8 do s = i end` on an M4 Pro: 0.39 s with the
/// reference layout, matching lua 5.5.1 itself, against 0.26 s here). A
/// store-only copy keeps the body off the chain, and comparing against a
/// precomputed last value instead of decrementing a count keeps the chain
/// to one slot.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_forprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (base, offset) = instruction.a_imm();

    let init = reg!(base);
    let limit = reg!(base + 1);
    let step = reg!(base + 2);

    // An integer loop needs only an integer init and step; the limit is
    // folded into the iteration count, so it is never compared again.
    let skip = if let (Some(i), Some(s)) = (init.get_integer(), step.get_integer()) {
        if s == 0 {
            raise!(OpError::ForStepZero);
        }
        match for_limit(i, limit, s) {
            None => raise!(OpError::ForNotNumber("limit", limit)),
            Some(None) => true,
            Some(Some(lim)) => {
                // The reference's unsigned iteration count, turned back into
                // the final value of the control variable: `last` lies in
                // [init, lim], so the wrapping arithmetic is exact and
                // `op_forloop` can stop on equality with no overflow check.
                let count = if s > 0 {
                    let span = (lim as u64).wrapping_sub(i as u64);
                    if s == 1 { span } else { span / s as u64 }
                } else {
                    // `-(s + 1) + 1` avoids negating `i64::MIN`.
                    (i as u64).wrapping_sub(lim as u64) / ((-(s + 1)) as u64 + 1)
                };
                let last = (i as u64).wrapping_add(count.wrapping_mul(s as u64)) as i64;
                *reg!(ref mut base) = Value::integer(last);
                *reg!(ref mut base + 1) = Value::integer(s);
                *reg!(ref mut base + 2) = Value::integer(i);
                *reg!(ref mut base + 3) = Value::integer(i);
                false
            }
        }
    } else {
        // Same coercion and check order as the reference `forprep`: strings
        // that name numbers are accepted, everything else is a `bad 'for'`
        // error, and the coerced floats replace the control registers.
        use crate::builtin::util::to_number;
        let Some(lim) = to_number(limit) else {
            raise!(OpError::ForNotNumber("limit", limit));
        };
        let Some(s) = to_number(step) else {
            raise!(OpError::ForNotNumber("step", step));
        };
        let Some(i) = to_number(init) else {
            raise!(OpError::ForNotNumber("initial value", init));
        };
        if s == 0.0 {
            raise!(OpError::ForStepZero);
        }
        let skip = if 0.0 < s { lim < i } else { i < lim };
        if !skip {
            *reg!(ref mut base) = Value::float(lim);
            *reg!(ref mut base + 1) = Value::float(s);
            *reg!(ref mut base + 2) = Value::float(i);
            *reg!(ref mut base + 3) = Value::float(i);
        }
        skip
    };

    if skip {
        ip = unsafe { ip.offset(offset as isize) };
    }

    dispatch!();
}

/// Numeric for loop step: advance the control variable and jump back while
/// iterations remain. Reads the layout `op_forprep` leaves behind.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_forloop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (base, offset) = instruction.a_imm();

    // The step's type tells the loop kind, and the hidden slots match it:
    // `op_forprep` wrote them and nothing else can (they are unnamed, and
    // the visible copy is `<const>` and never read here).
    let step = reg!(base + 1);
    if let Some(s) = step.get_integer() {
        let last = unsafe { reg!(base).get_integer().unwrap_unchecked() };
        let idx = unsafe { reg!(base + 2).get_integer().unwrap_unchecked() };
        // `idx` walks init, init+step, ..., last exactly, so `idx != last`
        // also guarantees `idx + step` stays in range.
        if idx != last {
            let idx = Value::integer(idx.wrapping_add(s));
            *reg!(ref mut base + 2) = idx;
            *reg!(ref mut base + 3) = idx;
            ip = unsafe { ip.offset(offset as isize) };
        }
    } else {
        let s = unsafe { step.get_float().unwrap_unchecked() };
        let lim = unsafe { reg!(base).get_float().unwrap_unchecked() };
        let idx = unsafe { reg!(base + 2).get_float().unwrap_unchecked() } + s;
        if if 0.0 < s { idx <= lim } else { lim <= idx } {
            let idx = Value::float(idx);
            *reg!(ref mut base + 2) = idx;
            *reg!(ref mut base + 3) = idx;
            ip = unsafe { ip.offset(offset as isize) };
        }
    }

    dispatch!();
}

// ---------------------------------------------------------------------------
// Generic for loop
// ---------------------------------------------------------------------------

/// Generic for preparation: jump to the loop test.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tforprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (_base, offset) = instruction.a_imm();
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: R[base+3], ... = R[base](R[base+1], R[base+2])
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tforcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (base, count) = instruction.ab();
    let iter = reg!(base);
    let state = reg!(base + 1);
    let control = reg!(base + 2);
    let cont = Continuation {
        payload: ContinuationPayload::TForCall { base, count },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(iter, &[state, control], cont);
}

/// Generic for loop test: if the first result R[base+3] != nil, copy it into
/// the control R[base+2] and jump back to the body.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tforloop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (base, offset) = instruction.a_imm();
    let first = reg!(base + 3);
    if !first.is_nil() {
        *reg!(ref mut base + 2) = first;
        ip = unsafe { ip.offset(offset as isize) };
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table initialization
// ---------------------------------------------------------------------------

/// R[table][offset+i] = R[table+i] for i in 1..=count
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_setlist<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (table, count, offset) = instruction.abd();
    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Internal("SETLIST on a non-table"));
    };
    // `count == 0` is MULTRET: element count comes from `thread.top`. It can
    // exceed u8 (a `VARARG count=0` spread), so index the stack by usize.
    let (base, max_stack) = {
        let f = thread.top_lua().unwrap();
        (f.base, f.closure.proto.max_stack_size as usize)
    };
    let elements_start = base + table as usize + 1;
    let n = if count == 0 {
        thread.top - elements_start
    } else {
        count as usize
    };
    let off = offset as i64;
    for i in 1..=n {
        let val = thread.stack[elements_start + i - 1];
        let key = Value::integer(off + i as i64);
        t.raw_set(ctx, key, val);
    }
    if count == 0 {
        // A MULTRET spread leaves `thread.stack` truncated to `thread.top` by
        // the producer (e.g. a native call's variadic return). Restore the
        // frame's register window so subsequent fixed-register ops stay in
        // bounds — mirrors `op_call`'s post-return restore (PUC's
        // `L->top = ci->top`). A resize may reallocate, so refresh `registers`.
        thread.ensure_slots(base + max_stack);
        registers = unsafe { thread.stack.as_mut_ptr().add(base) };
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Closures
// ---------------------------------------------------------------------------

/// R[dst] = closure(proto[idx])
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_closure<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, proto_idx) = instruction.ad();
    let frame = thread.top_lua().unwrap();
    let parent_closure = frame.closure;
    let base = frame.base;
    let proto = parent_closure.proto.prototypes[proto_idx as usize];

    let thread_handle = thread.thread_handle.expect("thread must have a handle");
    let mut upvalues_vec = Vec::with_capacity(proto.upvalue_desc.len());
    for desc in proto.upvalue_desc.iter() {
        let uv = match desc {
            UpValueDescriptor::ParentLocal(idx) => {
                let stack_idx = base + *idx as usize;
                // Check if there's already an open upvalue for this stack slot
                let existing = thread.open_upvalues.iter().find(|uv| {
                    matches!(&*uv.borrow(), UpvalueState::Open { index, .. } if *index == stack_idx)
                });
                if let Some(uv) = existing {
                    *uv
                } else {
                    let uv: Upvalue<'gc> = Gc::new(
                        ctx.mutation(),
                        RefLock::new(UpvalueState::Open {
                            thread: thread_handle,
                            index: stack_idx,
                        }),
                    );
                    thread.open_upvalues.push(uv);
                    uv
                }
            }
            UpValueDescriptor::ParentUpvalue(idx) => parent_closure.upvalues[*idx as usize],
        };
        upvalues_vec.push(uv);
    }
    let upvalues: Box<[Upvalue<'gc>]> = upvalues_vec.into_boxed_slice();

    let func = Function::new_lua(ctx.mutation(), proto, upvalues);
    *reg!(ref mut dst) = Value::function(func);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Varargs
// ---------------------------------------------------------------------------

/// Copy varargs into `R[dst..]`. `count == 0` is MULTRET (copy all, set
/// `thread.top`); `count > 0` copies `count - 1`, nil-padding short.
///
/// Source depends on `Prototype::needs_vararg_table` (Lua's `OP_VARARG` `k`
/// flag): optimized reads the below-base region; materialized reads `1..=t.n`
/// from the table in `R[num_params]`, so mutations to it are visible.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_vararg<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, count) = instruction.ab();
    // Copy scalars/`Gc` out so the frame borrow ends before touching the stack.
    let (base, num_extras, proto) = {
        let frame = thread.top_lua().unwrap();
        (frame.base, frame.num_extras as usize, frame.closure.proto)
    };
    let target = base + dst as usize;

    if proto.needs_vararg_table {
        // Materialized: read elements `1..=t.n` from the vararg table.
        let table = thread.stack[base + proto.num_params as usize]
            .get_table()
            .expect("materialized vararg slot must hold a table");
        // Read `t.n` in a tight borrow so it's released before the resize.
        let navail = table
            .inner()
            .borrow()
            .raw_get(Value::string(LuaString::new(ctx, b"n")))
            .get_integer()
            .filter(|n| *n >= 0)
            .unwrap_or(0) as usize;
        let wanted = if count == 0 {
            navail
        } else {
            count as usize - 1
        };
        let new_top = target + wanted;
        thread.ensure_slots(new_top);
        registers = unsafe { thread.stack.as_mut_ptr().add(base) };
        let t = table.inner().borrow();
        for i in 0..wanted {
            thread.stack[target + i] = t.raw_get(Value::integer(i as i64 + 1));
        }
        if count == 0 {
            thread.top = new_top;
        }
        dispatch!();
    }

    // Optimized: read the below-base region directly.
    let extras_start = base - num_extras;
    if count == 0 {
        // MULTRET: copy all extras and publish the new dynamic top.
        let new_top = target + num_extras;
        thread.ensure_slots(new_top);
        registers = unsafe { thread.stack.as_mut_ptr().add(base) };
        if num_extras > 0 {
            thread
                .stack
                .copy_within(extras_start..extras_start + num_extras, target);
        }
        thread.top = new_top;
    } else {
        let wanted = count as usize - 1;
        let to_copy = num_extras.min(wanted);
        if to_copy > 0 {
            thread
                .stack
                .copy_within(extras_start..extras_start + to_copy, target);
        }
        for i in to_copy..wanted {
            *reg!(ref mut dst + i as u8) = Value::nil();
        }
    }
    dispatch!();
}

/// Optimized below-base read of an un-escaped named vararg: integer key
/// `1..=num_extras`, `"n"` = count, else nil. Escaped varargs are rewritten
/// to `GETTABLE` at compile time, so `base` is unused here (kept only as that
/// rewrite's table operand). Lua 5.5 `OP_GETVARG`, which also ignores `B`.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_varargget<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (dst, _base, key) = instruction.abc();
    let key_val = reg!(key);
    let frame = thread.top_lua().unwrap();
    let num_extras = frame.num_extras as usize;
    let extras_start = frame.base - num_extras;
    // Normalize integral float keys (`args[1.0]` == `args[1]`) so the optimized
    // path agrees with the GETTABLE an escaped vararg would use.
    let int_key = key_val.get_integer().or_else(|| {
        let f = key_val.get_float()?;
        let i = f as i64;
        (i as f64 == f).then_some(i)
    });
    let v = if let Some(k) = int_key {
        if k >= 1 && (k as usize) <= num_extras {
            thread.stack[extras_start + (k as usize) - 1]
        } else {
            Value::nil()
        }
    } else if let Some(s) = key_val.get_string() {
        if s.as_bytes() == b"n" {
            Value::integer(num_extras as i64)
        } else {
            Value::nil()
        }
    } else {
        Value::nil()
    };
    *reg!(ref mut dst) = v;
    dispatch!();
}

/// Adjust the stack on entry to a vararg function: rotate the leading
/// `num_params` slots past the extras (layout becomes `[extras... fixed...]`)
/// and advance `frame.base` past the extras, so `VARARG` can read
/// `stack[base - num_extras .. base]`.
///
/// When `needs_vararg_table` is set, also materializes `{ extras...; n=count }`
/// into `R[num_params]` (Lua's `createvarargtab`); `VARARG` / `GETTABLE` then
/// read the table, so mutations to it are observed.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_varargprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let _num_fixed = instruction.a();
    let (num_extras, num_params, base, max_stack, needs_table) = {
        let frame = thread.top_lua().unwrap();
        (
            frame.num_extras as usize,
            frame.closure.proto.num_params as usize,
            frame.base,
            frame.closure.proto.max_stack_size as usize,
            frame.closure.proto.needs_vararg_table,
        )
    };
    let new_base = if num_extras > 0 {
        let total = num_extras + num_params;
        // [fixed..., extras...].rotate_left(num_params) => [extras..., fixed...]
        thread.stack[base..base + total].rotate_left(num_params);
        let new_base = base + num_extras;
        thread.ensure_slots(new_base + max_stack);
        thread.top_lua_mut().unwrap().base = new_base;
        registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
        new_base
    } else {
        base
    };
    if needs_table {
        // Store into the stack slot before filling so a mid-fill alloc can't
        // collect the table (mirrors `op_newtable`).
        let extras_start = new_base - num_extras;
        let table = Table::new(ctx);
        thread.stack[new_base + num_params] = Value::table(table);
        for i in 0..num_extras {
            let v = thread.stack[extras_start + i];
            table.raw_set(ctx, Value::integer(i as i64 + 1), v);
        }
        table.raw_set(
            ctx,
            Value::string(LuaString::new(ctx, b"n")),
            Value::integer(num_extras as i64),
        );
    }
    dispatch!();
}

/// Lua 5.5 ERRNNIL: raise if `R[src]` is **not** nil.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_errnnil<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    let (src, name_key) = instruction.ad();
    if std::hint::unlikely(!reg!(src).is_nil()) {
        raise!(OpError::GlobalRedefined(name_key));
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_nop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    helpers!(instruction, ctx, thread, registers, ip, handlers, ds);
    dispatch!();
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_stop<'gc>(
    _instruction: Instruction,
    _ctx: Context<'gc>,
    _thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
    _ds: &mut DispatchState<'gc>,
) {
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Close all TBC variables at stack indices >= `start_idx`.
/// Removes them from the tracking list; __close invocation is pending (see #45).
pub(crate) fn close_tbc_vars<'gc>(
    _mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    start_idx: usize,
) {
    thread.tbc_slots.retain(|&slot| {
        if slot >= start_idx {
            // See #45: invoke __close metamethod on thread.stack[slot].
            false
        } else {
            true
        }
    });
}

#[inline(always)]
fn read_upvalue<'gc>(thread: &ThreadState<'gc>, uv: Upvalue<'gc>) -> Value<'gc> {
    match &*uv.borrow() {
        UpvalueState::Closed(v) => *v,
        UpvalueState::Open { thread: t, index } => unsafe {
            let running = thread.thread_handle.unwrap_unchecked().inner();

            if Gc::ptr_eq(t.inner(), running) {
                *thread.stack.get_unchecked(*index)
            } else {
                *t.borrow().stack.get_unchecked(*index)
            }
        },
    }
}

#[inline(always)]
fn write_upvalue<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    uv: Upvalue<'gc>,
    val: Value<'gc>,
) {
    unsafe {
        let running = thread.thread_handle.unwrap_unchecked().inner();

        let mut uv_ref = uv.borrow_mut(mc);
        match &mut *uv_ref {
            UpvalueState::Closed(v) => *v = val,
            UpvalueState::Open { thread: t, index } => {
                if Gc::ptr_eq(t.inner(), running) {
                    *thread.stack.get_unchecked_mut(*index) = val;
                } else {
                    *t.borrow_mut(mc).stack.get_unchecked_mut(*index) = val;
                }
            }
        }
    }
}

/// Persist the top Lua frame's resume point before leaving the dispatch
/// chain (call, suspension, error), so the executor re-enters at the
/// instruction after the one at `ip` and `debug::frame_line` can locate it.
///
/// SAFETY: `ip` points into the proto's `code` (the handler chain only ever
/// advances it within that slice), so the offset is in bounds.
#[inline]
fn save_pc<'gc>(thread: &mut ThreadState<'gc>, ip: *const Instruction) {
    if let Some(frame) = thread.top_lua_mut() {
        frame.pc = ip;
    }
}

/// Invoke a native callback. Presents the callback a window whose logical
/// length is `argc` (`thread.top` seeded to `args_base + argc`); the backing
/// vec is grown to physically cover the window but is NEVER shrunk, so a
/// native call cannot truncate the shared stack below an outer frame's
/// register window. The callback signals its result count through the logical
/// top: after the `Return` path, the results count is `thread.top - args_base`.
pub(crate) fn invoke_native<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<crate::vm::sequence::CallbackAction<'gc>, crate::env::Error<'gc>> {
    let end = args_base + argc;
    // Grow-not-shrink: cover the arg window physically, but leave the slots
    // above it — which may be an outer frame's registers — alone.
    thread.ensure_slots(end);
    thread.top = end;
    let current_thread = thread
        .thread_handle
        .expect("ThreadState missing back-reference");
    let nctx = NativeContext {
        ctx,
        upvalues: &nc.upvalues,
        exec: crate::vm::sequence::Execution::new(current_thread),
    };
    let stack = Stack::new(&mut thread.stack, &mut thread.top, args_base);
    // The stack is grown-not-shrunk and may leave dead scratch above the logical
    // top; that's fine because `ThreadState`'s `Collect` traces only the live
    // high-water (derived from the frames + `top`), so dead slots never retain.
    (nc.function)(nctx, stack)
}

/// What should happen after a frame returns with values at
/// `stack[values_base .. values_base + nret]`. Produced by [`frame_return`],
/// consumed by `op_return` and the native-tailcall path in `op_tailcall`.
pub(crate) enum FrameReturn {
    /// A continuation was attached to the departing frame; caller must
    /// tail-call `cont_resume`. The continuation's `results_base` / `nret`
    /// have already been written back into the top frame.
    Continuation,
    /// The departing frame was the outermost one; thread is now `Result`.
    /// Caller should return from the handler.
    TopLevel,
    /// Normal return to the caller frame, which has been restored to the
    /// top of the frame stack. Caller rebinds `ip` / `registers` /
    /// `DispatchState` to it and dispatches.
    Caller {
        new_base: usize,
        new_ip: *const Instruction,
    },
    /// The popped Lua frame's parent is a non-Lua frame (Sequence /
    /// WaitThread / Start / Error). The values have been left at
    /// `stack[bottom..]` for the executor's driver loop to consume on the
    /// next pump. `op_return` returns to exit dispatch.
    ToNonLua,
}

/// Unwind the top-of-stack frame assuming it returned the values at
/// `stack[values_base .. values_base + nret]`. Shared by the bytecode
/// `RETURN` handler and the native-tailcall path.
#[inline(always)]
pub(crate) fn frame_return<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    ds: &mut DispatchState<'gc>,
    values_base: usize,
    nret: usize,
) -> FrameReturn {
    let (cur_base, num_results, num_extras, continuation) = {
        let f = unsafe { &*ds.frame };
        (f.base, f.num_results, f.num_extras as usize, f.continuation)
    };

    if let Some(mut cont) = continuation {
        cont.results_base = values_base;
        cont.nret = nret as u8;
        unsafe { (*ds.frame).continuation = Some(cont) };
        return FrameReturn::Continuation;
    }

    // The func slot sits at `cur_base - 1 - num_extras`: VARARGPREP shifted
    // base past the extras at `[cur_base - num_extras .. cur_base]` (0 for
    // non-vararg frames).
    close_upvalues(mc, thread, cur_base);
    close_tbc_vars(mc, thread, cur_base);
    // The top frame is the `Frame::Lua` read above and `LuaFrame` is `Copy`,
    // so there is nothing to drop; `Vec::pop` would copy the 96-byte frame out
    // and run the enum's drop glue.
    const { assert!(!std::mem::needs_drop::<LuaFrame<'_>>()) };
    unsafe { thread.frames.set_len(thread.frames.len() - 1) };

    let dst_start = cur_base - 1 - num_extras;

    let (new_base, new_ip) = match thread.frames.last() {
        Some(Frame::Lua(caller)) => (caller.base, caller.pc),
        None => {
            thread
                .stack
                .copy_within(values_base..values_base + nret, dst_start);
            // The thread is done, so nothing above the results is live: pair
            // `Result { bottom }` with a `top` marking the result end, and release
            // the rest. Consumers read `stack[bottom..top]`.
            thread.discard_above(dst_start + nret);
            thread.status = ThreadStatus::Result { bottom: dst_start };
            return FrameReturn::TopLevel;
        }
        // The parent isn't a Lua frame (Sequence/WaitThread/etc.), so the
        // executor driver picks up here: place all `nret` values at
        // `stack[dst_start..]` for the parent's window. Since no register
        // window sits above the results the shrink is legal, and it publishes
        // the parent's input window as `stack[dst_start..top]`.
        Some(_) => {
            thread
                .stack
                .copy_within(values_base..values_base + nret, dst_start);
            thread.discard_above(dst_start + nret);
            return FrameReturn::ToNonLua;
        }
    };

    // `num_results == 0` is the CALL's MULTRET: deliver all `nret` and publish `thread.top`.
    if num_results == 0 {
        thread
            .stack
            .copy_within(values_base..values_base + nret, dst_start);
        // The donor copies the down-shift left in the popped callee's
        // registers are dead scratch above `top`; `trim_dead` drops them.
        thread.set_top_dirty(dst_start + nret);
    } else {
        let wanted = num_results as usize - 1;
        let to_copy = nret.min(wanted);
        // `dst_start < values_base` (the func slot is below the callee's
        // registers) and both ranges lie inside the callee's window, which
        // `op_call` sized the vec for; a forward element copy is in bounds and
        // never reads a slot it already overwrote. Indexing here re-checks the
        // vec length against every store, so use raw pointers.
        debug_assert!(values_base + to_copy <= thread.stack.len());
        debug_assert!(dst_start + wanted <= thread.stack.len());
        let stack = thread.stack.as_mut_ptr();
        for i in 0..to_copy {
            unsafe { *stack.add(dst_start + i) = *stack.add(values_base + i) };
        }
        for i in to_copy..wanted {
            unsafe { *stack.add(dst_start + i) = Value::nil() };
        }
        // Publish the landing end. Without this, a `top` left high by a multires
        // producer *inside the callee* would still be the high-water long after
        // the callee popped, so `live_top` would keep tracing its dead registers
        // — the exact #43 leak, just via a stale `top` instead of the vec length.
        // The callee's registers themselves are dead scratch that `trim_dead`
        // releases on exit.
        thread.set_top_dirty(dst_start + wanted);
    }

    FrameReturn::Caller { new_base, new_ip }
}

/// Close all open upvalues pointing at stack indices >= `start_idx`.
/// Each open upvalue is converted to Closed by capturing the current stack value.
/// Inlined so the usual no-open-upvalues return costs one load, not a call.
#[inline(always)]
pub(crate) fn close_upvalues<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    start_idx: usize,
) {
    if !thread.open_upvalues.is_empty() {
        close_upvalues_slow(mc, thread, start_idx);
    }
}

#[inline(never)]
fn close_upvalues_slow<'gc>(mc: &Mutation<'gc>, thread: &mut ThreadState<'gc>, start_idx: usize) {
    thread.open_upvalues.retain(|uv| {
        let should_close = {
            let borrowed = uv.borrow();
            match &*borrowed {
                UpvalueState::Open { index, .. } => *index >= start_idx,
                UpvalueState::Closed(_) => false,
            }
        };
        if should_close {
            let val = thread.stack[{
                let b = uv.borrow();
                match &*b {
                    UpvalueState::Open { index, .. } => *index,
                    _ => unreachable!(),
                }
            }];
            *uv.borrow_mut(mc) = UpvalueState::Closed(val);
            false // remove from open list
        } else {
            true // keep
        }
    });
}

// ---------------------------------------------------------------------------
// Metamethod invocation / continuations
// ---------------------------------------------------------------------------

/// Maximum depth of `__index` / `__newindex` / `__call` chains before we
/// give up and raise (matches Lua 5.4's `MAXTAGLOOP`).
const MAX_TAG_LOOP: usize = 2000;

/// Result of walking an `__index` chain.
pub(crate) enum IndexChain<'gc> {
    /// The chain resolved synchronously to a value (possibly `Nil`).
    Resolved(Value<'gc>),
    /// The chain ended in a callable that must be invoked with
    /// `(receiver, key)`. `receiver` is the table that owned the function
    /// `__index`, matching Lua's `luaV_finishget` behavior.
    Invoke {
        func: Value<'gc>,
        receiver: Value<'gc>,
    },
    /// Chain depth exceeded `MAX_TAG_LOOP`; caller should raise.
    Exhausted,
}

/// Walk the `__index` chain starting from `start_metatable`'s `__index`
/// slot. `start_receiver` is the value originally indexed — a `Table`
/// for the table paths, a `Userdata` for the userdata path — and is the
/// `receiver` handed to a functional `__index` resolved at depth 0
/// (deeper hops use the intermediate `__index` table as receiver, per
/// `luaV_finishget`). `mm_index_name` is the pre-interned `__index`
/// LuaString — the function never re-interns it.
#[inline]
pub(crate) fn walk_index_chain<'gc>(
    start_receiver: Value<'gc>,
    start_metatable: Table<'gc>,
    key: Value<'gc>,
    mm_index_name: LuaString<'gc>,
) -> IndexChain<'gc> {
    let mut current_receiver = start_receiver;
    let mut mm = start_metatable.raw_get(Value::string(mm_index_name));
    for _ in 0..MAX_TAG_LOOP {
        if mm.is_nil() {
            return IndexChain::Resolved(Value::nil());
        }
        let next = match mm.get_table() {
            Some(t) => t,
            None => {
                return IndexChain::Invoke {
                    func: mm,
                    receiver: current_receiver,
                };
            }
        };
        let v = next.raw_get(key);
        if !v.is_nil() {
            return IndexChain::Resolved(v);
        }
        if !next.shape().has_mm(MetamethodBits::INDEX) {
            return IndexChain::Resolved(Value::nil());
        }
        // INDEX bit implies metatable is Some.
        let next_mt = unsafe { next.metatable().unwrap_unchecked() };
        mm = next_mt.raw_get(Value::string(mm_index_name));
        current_receiver = Value::table(next);
    }
    IndexChain::Exhausted
}

/// Result of resolving an index on a *userdata* receiver: walk the
/// userdata metatable's `__index` chain, or signal "raise" when the
/// userdata has no metatable or a nil `__index` (no raw indexing exists
/// for userdata, so this is the "attempt to index" case — see
/// `luaV_finishget`). Callers that tolerate a missing method (SELF) must
/// not use this; they treat the absent `__index` as a nil result so the
/// subsequent CALL raises instead.
#[inline]
fn userdata_index_chain<'gc>(
    u: crate::env::Userdata<'gc>,
    recv_val: Value<'gc>,
    key: Value<'gc>,
    mm_index_name: LuaString<'gc>,
) -> Result<IndexChain<'gc>, ()> {
    let mt = u.metatable().ok_or(())?;
    if mt.raw_get(Value::string(mm_index_name)).is_nil() {
        return Err(());
    }
    Ok(walk_index_chain(recv_val, mt, key, mm_index_name))
}

/// Result of walking a `__newindex` chain.
enum NewIndexChain<'gc> {
    /// Raw-assign `value` into this table.
    RawSet(Table<'gc>),
    /// The chain ended in a callable; invoke with `(receiver, key, value)`.
    Invoke {
        func: Value<'gc>,
        receiver: Value<'gc>,
    },
    /// Chain depth exceeded `MAX_TAG_LOOP`; caller should raise.
    Exhausted,
}

/// Walk the `__newindex` chain. If the key already exists in `table`, do
/// a raw set there. Otherwise follow `__newindex` tables; terminate at
/// the first callable or at a table that has the key (or has no
/// `__newindex`). `mm_newindex_name` is the pre-interned `__newindex`
/// LuaString — the function never re-interns it.
#[inline]
fn walk_newindex_chain<'gc>(
    table: Table<'gc>,
    key: Value<'gc>,
    mm_newindex_name: LuaString<'gc>,
) -> NewIndexChain<'gc> {
    let mut t = table;
    for _ in 0..MAX_TAG_LOOP {
        // If the key already has a value, skip __newindex and raw_set here.
        if !t.raw_get(key).is_nil() {
            return NewIndexChain::RawSet(t);
        }
        if !t.shape().has_mm(MetamethodBits::NEWINDEX) {
            return NewIndexChain::RawSet(t);
        }
        // NEWINDEX bit implies metatable is Some.
        let mt = unsafe { t.metatable().unwrap_unchecked() };
        let mm = mt.raw_get(Value::string(mm_newindex_name));
        if mm.is_nil() {
            return NewIndexChain::RawSet(t);
        }
        if let Some(next) = mm.get_table() {
            t = next;
            continue;
        }
        return NewIndexChain::Invoke {
            func: mm,
            receiver: Value::table(t),
        };
    }
    NewIndexChain::Exhausted
}

/// The resolved target of a call: either a Lua bytecode closure (which the
/// caller must push a frame for) or a native Rust callback (which the caller
/// invokes inline).
pub(crate) enum CallTarget<'gc> {
    Lua(Gc<'gc, LuaClosure<'gc>>),
    Native(&'gc NativeClosure<'gc>),
}

/// Outcome of [`schedule_meta_call`], consumed by the `invoke_metamethod!`
/// macro. Captures the three ways a continuation-driven call can proceed.
pub(crate) enum MetaDispatch {
    /// Resolved to a Lua closure; a frame carrying the continuation was
    /// pushed. The caller rebinds `ip`/`registers` and `DispatchState` to
    /// the new frame and dispatches.
    Lua {
        new_ip: *const Instruction,
        new_base: usize,
    },
    /// Resolved to a native callback that returned synchronously. Its results
    /// sit at `stack[results_base .. results_base + nret]`; the caller applies
    /// the continuation payload inline.
    NativeReturn { results_base: usize, nret: u8 },
    /// Native callback suspended (Call/Yield/Resume/Sequence) or errored — a
    /// `pending_action` or `Frame::Error` was installed for the executor. The
    /// caller returns to exit dispatch.
    Suspended,
    /// Target is not callable (or a suspending comparison metamethod, which we
    /// don't support). The caller raises.
    Unresolvable,
}

/// Walk the `__call` chain at `thread.stack[func_idx]` until we hit a
/// callable target, shifting args right by one on each hop to prepend the
/// current callee as the first argument (Lua 5.5 `tryfuncTM` behavior).
/// Returns the resolved target and the (possibly adjusted) `nargs`, or
/// `None` if the chain is unresolvable: non-callable value, `nargs`
/// overflow, or `MAX_TAG_LOOP` exhaustion. Callers raise on `None`.
#[inline(always)]
fn resolve_call_chain<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    func_idx: usize,
    nargs: u8,
) -> Option<(CallTarget<'gc>, u8)> {
    // Plain functions are the overwhelmingly common case; keep the `__call`
    // walk (and its stack frame) out of the handler.
    if let Some(f) = thread.stack[func_idx].get_function() {
        return match f.inner().as_ref() {
            FunctionKind::Lua(c) => Some((CallTarget::Lua(*c), nargs)),
            FunctionKind::Native(nc) => Some((CallTarget::Native(nc), nargs)),
        };
    }
    resolve_call_chain_slow(ctx, thread, func_idx, nargs)
}

#[cold]
#[inline(never)]
fn resolve_call_chain_slow<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    func_idx: usize,
    mut nargs: u8,
) -> Option<(CallTarget<'gc>, u8)> {
    for _ in 0..MAX_TAG_LOOP {
        let func_val = thread.stack[func_idx];
        if let Some(f) = func_val.get_function() {
            return match f.inner().as_ref() {
                FunctionKind::Lua(c) => Some((CallTarget::Lua(*c), nargs)),
                FunctionKind::Native(nc) => Some((CallTarget::Native(nc), nargs)),
            };
        }
        let mm = match func_val.get_table() {
            Some(t) => t.get_metamethod(ctx.symbols().mm_call),
            None => return None,
        };
        if mm.is_nil() {
            return None;
        }
        // A MULTRET call (`nargs == 0`) carries its count in `thread.top`.
        let actual_args = if nargs == 0 {
            thread.top - (func_idx + 1)
        } else {
            nargs as usize - 1
        };
        thread.ensure_slots(func_idx + 2 + actual_args);
        for i in (0..actual_args).rev() {
            thread.stack[func_idx + 2 + i] = thread.stack[func_idx + 1 + i];
        }
        thread.stack[func_idx + 1] = func_val;
        thread.stack[func_idx] = mm;
        if nargs == 0 {
            thread.set_top(thread.top + 1);
        } else {
            nargs = nargs.checked_add(1)?;
        }
    }
    None
}

/// Look up a binary metamethod on `lhs` first, then `rhs`. Only checks
/// metatables on tables; userdata metatables and the string metatable
/// are pending those subsystems (see #47). `name` is the pre-interned
/// LuaString from `Context::symbols()`.
#[inline]
fn binop_metamethod<'gc>(lhs: Value<'gc>, rhs: Value<'gc>, name: LuaString<'gc>) -> Value<'gc> {
    if let Some(t) = lhs.get_table() {
        let m = t.get_metamethod(name);
        if !m.is_nil() {
            return m;
        }
    }
    if let Some(t) = rhs.get_table() {
        return t.get_metamethod(name);
    }
    Value::nil()
}

/// Look up a unary metamethod on `val`. Same caveat as `binop_metamethod`.
#[inline]
fn unop_metamethod<'gc>(val: Value<'gc>, name: LuaString<'gc>) -> Value<'gc> {
    if let Some(t) = val.get_table() {
        return t.get_metamethod(name);
    }
    Value::nil()
}

/// Invoke a metamethod / iterator (or other helper) with a post-return
/// continuation. The function + args are staged above the caller's
/// `max_stack_size` so no live register is clobbered, then `resolve_call_chain`
/// walks any `__call` hops to the callable target.
///
/// A Lua target gets a frame carrying the continuation pushed and returns
/// [`MetaDispatch::Lua`]. A native target is invoked inline: a synchronous
/// `Return` surfaces its results via [`MetaDispatch::NativeReturn`] for the
/// caller to apply the payload to; a suspending action (Call/Yield/Resume/
/// Sequence) is mapped to a `pending_action` whose `CallSite` lands the
/// results exactly where the continuation would, then re-enters the caller
/// frame at the saved pc — so `op_return`-style continuation logic isn't
/// needed for the suspend case. A native error installs a `Frame::Error`.
///
/// The native suspend path replays all four continuation payloads uniformly via
/// `apply_native_continuation`: `StoreResult`, `IgnoreResult`, `TForCall`, and
/// `CondJump` (which bumps the caller frame's saved `pc` by the payload's offset
/// to select the branch once the suspended comparison's result arrives).
#[inline(never)]
fn schedule_meta_call<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    meta_fn: Value<'gc>,
    args: &[Value<'gc>],
    cont: Continuation,
    caller_ip: *const Instruction,
) -> MetaDispatch {
    // Save caller's pc; no decisions here depend on knowing the final target.
    // The native suspend path relies on this so the executor re-enters the
    // caller frame at the instruction following the one that scheduled us.
    save_pc(thread, caller_ip);

    // schedule_meta_call is only reachable from inside a handler via
    // invoke_metamethod!, so an active caller frame is always present.
    let caller_base = thread
        .top_lua()
        .expect("schedule_meta_call called without an active Lua frame")
        .base;
    let scratch_func =
        caller_base + thread.top_lua().unwrap().closure.proto.max_stack_size as usize;
    let new_base = scratch_func + 1;

    // Stage meta_fn + args so resolve_call_chain sees them in op_call layout.
    thread.ensure_slots(new_base + args.len());
    thread.stack[scratch_func] = meta_fn;
    for (i, &a) in args.iter().enumerate() {
        thread.stack[new_base + i] = a;
    }

    // Walk any __call chain. nargs follows op_call's convention (includes the
    // function slot), so `args.len() + 1`.
    debug_assert!(args.len() < u8::MAX as usize);
    let nargs = (args.len() + 1) as u8;
    let Some((target, final_nargs)) = resolve_call_chain(ctx, thread, scratch_func, nargs) else {
        return MetaDispatch::Unresolvable;
    };
    let actual_args = final_nargs as usize - 1;

    let closure = match target {
        CallTarget::Lua(c) => c,
        CallTarget::Native(nc) => {
            return schedule_native_meta_call(
                ctx,
                thread,
                nc,
                cont,
                caller_base,
                new_base,
                actual_args,
            );
        }
    };

    // Grow stack to fit the resolved closure's full frame.
    thread.ensure_slots(new_base + closure.proto.max_stack_size as usize);

    // Nil-fill any parameter slots not covered by the (possibly shifted) args.
    let num_params = closure.proto.num_params as usize;
    for i in actual_args..num_params {
        thread.stack[new_base + i] = Value::nil();
    }
    let num_extras = if closure.proto.is_vararg {
        actual_args.saturating_sub(num_params) as u32
    } else {
        0
    };

    thread.push_lua(LuaFrame {
        closure,
        base: new_base,
        // Ignored by op_return when a continuation is set — the continuation
        // reads return values directly from the stack via `cont.results_base`.
        pc: closure.proto.code.as_ptr(),
        num_results: 0,
        num_extras,
        continuation: Some(cont),
    });

    let new_ip = closure.proto.code.as_ptr();
    MetaDispatch::Lua { new_ip, new_base }
}

/// Native arm of [`schedule_meta_call`]. The callable native `nc` and its
/// `actual_args` arguments are already staged at `new_base - 1` / `new_base..`.
/// Drives the callback and translates its [`CallbackAction`] into a
/// [`MetaDispatch`].
fn schedule_native_meta_call<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    nc: &'gc NativeClosure<'gc>,
    cont: Continuation,
    caller_base: usize,
    new_base: usize,
    actual_args: usize,
) -> MetaDispatch {
    use crate::vm::sequence::CallbackAction;

    let args_base = new_base;
    let action = match invoke_native(ctx, thread, nc, args_base, actual_args) {
        Ok(a) => a,
        Err(err) => {
            // Mirror op_call's native-error path: a Frame::Error lets the
            // executor's unwinder route to the nearest catcher (e.g. pcall).
            thread.raise(ctx, err);
            return MetaDispatch::Suspended;
        }
    };

    match action {
        CallbackAction::Return => {
            // Result count via the logical top; the staged scratch window sits
            // above the caller frame, so nothing shrank the caller's window.
            let nret = (thread.top - args_base) as u8;
            MetaDispatch::NativeReturn {
                results_base: args_base,
                nret,
            }
        }
        // A native that wants to call back into Lua (`Call`) delivers its
        // result through the normal frame-return path, which doesn't run
        // `land_call_results` and so can't replay our continuation. It's not a
        // shape any current metamethod/iterator native produces; reject it
        // cleanly rather than silently dropping the continuation.
        CallbackAction::Call { .. } => {
            let err = crate::env::Error::from_str(
                ctx,
                "metamethod/iterator native cannot tail-call into Lua across the continuation",
            );
            thread.raise(ctx, err);
            MetaDispatch::Suspended
        }
        // Suspending native (`Yield`/`Resume`/`Sequence`). Park the full
        // continuation on the `CallSite`; when the suspension resolves, its
        // results funnel through `land_call_results`, which applies the
        // continuation against the caller frame (`apply_native_continuation`).
        // This replays every payload uniformly — including `CondJump`, whose
        // branch decision (a `pc` bump) can't be expressed as a plain landing.
        other => {
            thread.pending_action = Some(PendingAction {
                action: other,
                call_site: CallSite {
                    bottom: args_base,
                    // Unused while `cont` is set (the continuation drives the
                    // landing), but kept in-bounds / meaningful as a fallback.
                    func_idx: caller_base,
                    returns: 0,
                    cont: Some(cont),
                },
            });
            MetaDispatch::Suspended
        }
    }
}

/// Shared skeleton used by every `cont_*` function: extract the continuation
/// from the callee frame, cleanup, pop, restore caller state, then expose
/// `$cont_out` to the caller's scope for payload-specific fixup. The
/// continuation's `results_base` and `nret` remain valid post-pop because
/// nothing is pushed to the stack during cleanup.
macro_rules! finalize_return {
    (
        $instruction:expr, $ctx:expr, $thread:expr,
        $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
        cont: $cont_out:ident
    ) => {
        helpers!($instruction, $ctx, $thread, $registers, $ip, $handlers, $ds);

        let $cont_out: Continuation = $thread.top_lua().unwrap().continuation.unwrap();
        let __cur_base = $thread.top_lua().unwrap().base;

        close_upvalues($ctx.mutation(), $thread, __cur_base);
        close_tbc_vars($ctx.mutation(), $thread, __cur_base);
        $thread.frames.pop();

        $ds.bind_frame($thread);
        let __caller_base = unsafe {
            $ip = (*$ds.frame).pc;
            (*$ds.frame).base
        };
        $registers = unsafe { $thread.stack.as_mut_ptr().add(__caller_base) };
    };
}

/// The single continuation entry point, tail-called by `op_return` /
/// `op_tailcall` once a frame carrying a [`Continuation`] returns. Pops the
/// callee frame, restores the caller's `ip`/`registers`, then applies the
/// payload to the returned values (`stack[results_base .. +nret]`, written by
/// `frame_return`). The synchronous-native path in `invoke_metamethod!`
/// applies the same payload via `apply_cont_payload!` without this frame
/// teardown, since no callee frame exists there.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cont_resume<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
) {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, ds, cont: cont);
    apply_cont_payload!(
        cont,
        cont.results_base,
        cont.nret,
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds
    );
}
