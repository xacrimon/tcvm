use crate::dmm::{Gc, Mutation, RefLock};
use crate::env::function::{
    Function, FunctionKind, InlineCache, LuaFn, NativeClosure, Stack, Upvalue, UpvalueState,
};
use crate::env::shape::{MetamethodBits, Shape};
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::{
    CallSite, ExecKind, LuaFrame, PendingAction, Thread, ThreadState, ThreadStatus, frame_flags,
};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, Op, UpValueDescriptor};
use crate::lua::Context;
use crate::vm::num;

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
    (Op::RETURN0, op_return0),
    (Op::RETURN1, op_return1),
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
    CallChainTooLong,
    /// ERRNNIL: constant index of the global's name.
    GlobalRedefined(u16),
    ForStepZero,
    /// Which `for` control value (`"limit"`, `"step"`, `"initial value"`)
    /// failed to coerce, and the offending value.
    ForNotNumber(&'static str, Value<'gc>),
    NilIndex,
    NanIndex,
    /// A call's register window would cross `ThreadState::stack_limit`.
    StackOverflow,
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
    /// The native in `R[func]`, set by CALL and TAILCALL for the entry they
    /// jump to. Reading the slot again there races the store of a MOVE that
    /// just filled it: the CPU sometimes runs the load first and replays it,
    /// which made `x = max(i, 3)` cost up to twice as much.
    native: *const NativeClosure<'gc>,
}

/// The thread's top frame, which must be a Lua frame, and its closure. Both
/// travel with the handler arguments (registers), so every site that changes
/// the top frame and keeps dispatching reassigns `frame` and `closure` from
/// this. The pointer is into `thread.frames`' buffer, which any push may
/// reallocate.
#[inline(always)]
fn top_frame<'gc>(thread: &mut ThreadState<'gc>) -> (*mut LuaFrame<'gc>, LuaFn<'gc>) {
    let frame = unsafe { thread.top_lua_ptr() };
    (frame, unsafe { (*frame).closure })
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
);

/// What the caller does with a metamethod's (or generic-for iterator's)
/// results. Carried by the callee's frame (`HAS_CONT`), or by the `CallSite`
/// of a native one that suspended. `cont_resume` finds the results at the
/// callee's function slot, up to `top`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Continuation {
    /// `R[dst]` = the first result, or nil.
    StoreResult { dst: u8 },
    /// Discard the results (`__newindex`).
    IgnoreResult,
    /// Skip the comparison's following JMP when the first result's truthiness
    /// differs from `inverted`.
    CondJump { inverted: bool },
    /// Generic for: `R[base+3 .. base+3+count]` = the results, nil-padded.
    TForCall { base: u8, count: u8 },
}

// Small enough to sit in a frame record's spare bytes.
const _: () = assert!(std::mem::size_of::<Option<Continuation>>() == 3);

macro_rules! helpers {
    ($instruction:expr, $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident, $frame:ident, $closure:ident) => {
        // The running frame and its closure travel in registers with the
        // other handler arguments; rebinding them (calls, returns) reassigns
        // these locals.
        #[allow(unused_mut)]
        let mut $frame: *mut LuaFrame<'gc> = $frame;
        #[allow(unused_mut)]
        let mut $closure: LuaFn<'gc> = $closure;
        #[allow(unused_macros)]
        macro_rules! dispatch {
            () => {{
                unsafe {
                    // Catches a frame change that forgot to rebind the
                    // dispatch registers.
                    #[cfg(debug_assertions)]
                    {
                        let top = $thread.top_lua_unchecked();
                        debug_assert!(std::ptr::eq($frame, top));
                        debug_assert!(std::ptr::eq(&*$closure, &*top.closure));
                        debug_assert!(std::ptr::eq(
                            $registers,
                            $thread.stack.as_ptr().add(top.base())
                        ));
                        debug_assert!(
                            $ip.offset_from_unsigned(top.closure.proto.code.as_ptr())
                                < top.closure.proto.code.len()
                        );
                    }
                    let _ = $instruction;
                    let instruction = *$ip;
                    let pos = instruction.opcode() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = *$handlers.cast::<Handler>().add(pos);
                    let ip = $ip.add(1);
                    become handler(
                        instruction,
                        $ctx,
                        $thread,
                        $registers,
                        ip,
                        $handlers,
                        $ds,
                        $frame,
                        $closure,
                    );
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
                become impl_error(
                    $instruction,
                    $ctx,
                    $thread,
                    $registers,
                    $ip,
                    $handlers,
                    $ds,
                    $frame,
                    $closure,
                );
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
                    debug_assert!(
                        ($$idx as usize)
                            < $thread.top_lua_unchecked().closure.proto.constants.len()
                    );
                    *$closure.constants.add($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                unsafe {
                    debug_assert!(
                        ($$idx as usize) < $thread.top_lua_unchecked().closure.upvalues.len()
                    );
                    *$closure.upvalues.as_ptr().add($$idx as usize)
                }
            }};
        }

        /// Tail-call handler `f` with this handler's arguments, optionally
        /// with a different instruction word.
        #[allow(unused_macros)]
        macro_rules! tail {
            ($$f:ident) => {
                tail!($$f, $instruction)
            };
            ($$f:ident, $$instruction:expr) => {
                become $$f(
                    $$instruction,
                    $ctx,
                    $thread,
                    $registers,
                    $ip,
                    $handlers,
                    $ds,
                    $frame,
                    $closure,
                )
            };
        }

        #[allow(unused_macros)]
        macro_rules! skip {
            () => {{
                $ip = unsafe { $ip.add(1) };
            }};
        }

        /// `if $$cond { skip!() }`, forced to compile as a branch: LLVM otherwise
        /// if-converts it to a `csel` on `ip`, making the next instruction load
        /// data-dependent on the compare instead of predicted. The asm is opaque
        /// so the select can't be re-formed.
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
                        ($frame, $closure) = top_frame($thread);
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
                        dispatch!();
                    }
                    MetaDispatch::NativeReturn { results_base, nret } => {
                        // The native ran inline with no pushed frame, so the
                        // caller is still on top — but staging / `invoke_native`
                        // may have reallocated the stack, so rebind `registers`
                        // (the Lua arm rebinds for the same reason) before the
                        // payload writes through it.
                        let (__cb, __ms) = (
                            unsafe { (*$frame).base() },
                            $closure.max_stack_size as usize,
                        );
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
                    MetaDispatch::CallChainTooLong => raise!(OpError::CallChainTooLong),
                    MetaDispatch::StackOverflow => raise!(OpError::StackOverflow),
                }
            }};
        }
    };
}

/// Applies a [`Continuation`]'s payload given its returned values at
/// `stack[results_base .. results_base + nret]`, then dispatches. Shared by
/// the Lua-return paths (`op_return_cont` and `cont_resume`, after popping the
/// callee frame) and the synchronous-native path (`invoke_metamethod!`, with
/// the caller frame still current). Expects `helpers!(...)` to have run in the
/// enclosing handler so `reg!` / `dispatch!` resolve; `$registers` / `$ip` must
/// already be bound to the *caller's* window.
macro_rules! apply_cont_payload {
    ($cont:expr, $results_base:expr, $nret:expr,
     $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident) => {{
        let __cont: Continuation = $cont;
        let __nret: usize = $nret;
        // The results sit in scratch above the caller's window, below `top`.
        debug_assert!($results_base + __nret <= $thread.stack.len());
        let __results = unsafe { $thread.stack.as_mut_ptr().add($results_base) };
        let __first = if __nret > 0 {
            unsafe { __results.read() }
        } else {
            Value::nil()
        };
        match __cont {
            Continuation::StoreResult { dst } => {
                *reg!(ref mut dst) = __first;
                dispatch!();
            }
            Continuation::IgnoreResult => {
                dispatch!();
            }
            Continuation::CondJump { inverted } => {
                if !__first.is_falsy() != inverted {
                    $ip = unsafe { $ip.add(1) };
                }
                dispatch!();
            }
            Continuation::TForCall { base, count } => {
                // Destination registers are `base+3 .. base+3+count`; they must
                // fit u8 register space.
                debug_assert!(
                    base as usize + 3 + count as usize <= u8::MAX as usize + 1,
                    "TFORCALL destination range exceeds u8 register space",
                );
                let __to_copy = __nret.min(count as usize);
                unsafe {
                    let dst = $registers.add(base as usize + 3);
                    copy_values(dst, __results, __to_copy);
                    fill_nil(dst.add(__to_copy), count as usize - __to_copy);
                }
                dispatch!();
            }
        }
    }};
}

/// Inflates the slow-path body for `R[dst] = recv[k]` on any receiver.
/// Expects `helpers!(...)` to have been invoked in the enclosing handler
/// so `dispatch!`, `raise!`, `invoke_metamethod!`, and `reg!` resolve.
macro_rules! get_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $recv:expr, $k:expr, $dst:expr) => {{
        let __recv: Value<'gc> = $recv;
        let __k: Value<'gc> = $k;
        let __dst_reg: u8 = $dst;

        if let Some(__t) = __recv.get_table() {
            let __v = __t.raw_get(__k);
            if !__v.is_nil() {
                *reg!(ref mut __dst_reg) = __v;
                dispatch!();
            }
        }
        match walk_index_chain($ctx, __recv, __k) {
            IndexChain::Resolved(__rv) => {
                *reg!(ref mut __dst_reg) = __rv;
                dispatch!();
            }
            IndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => {
                let __cont = Continuation::StoreResult { dst: __dst_reg };
                invoke_metamethod!(Value::function(__mm_func), &[__mm_recv, __k], __cont, Index);
            }
            IndexChain::NotIndexable(__v) => raise!(OpError::Index(__v)),
            IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
        }
    }};
}

/// Inflates the slow-path body for `recv[k] = v` on any receiver.
macro_rules! set_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $recv:expr, $k:expr, $v:expr) => {{
        let __k: Value<'gc> = $k;
        let __new_val: Value<'gc> = $v;

        match walk_newindex_chain($ctx, $recv, __k) {
            NewIndexChain::RawSet(__target) => {
                check_index_key!(__k);
                __target.raw_set($ctx, __k, __new_val);
                dispatch!();
            }
            NewIndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => {
                let __cont = Continuation::IgnoreResult;
                invoke_metamethod!(
                    Value::function(__mm_func),
                    &[__mm_recv, __k, __new_val],
                    __cont,
                    Index
                );
            }
            NewIndexChain::NotIndexable(__v) => raise!(OpError::Index(__v)),
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
fn read_ic<'gc>(closure: LuaFn<'gc>, ic_idx: u16) -> InlineCache<'gc> {
    // SAFETY: ic_idx is allocated at compile-time within the prototype's
    // IC count; debug-asserted in alloc_ic_slot's saturating_add.
    debug_assert!((ic_idx as usize) < closure.proto.ic_table.len());
    unsafe { (*closure.ic_table.add(ic_idx as usize)).get() }
}

/// Refill the IC entry. Called by slow paths after they've done a full
/// shape lookup; subsequent same-shape accesses skip the slow path.
#[inline(always)]
fn fill_ic<'gc>(ctx: Context<'gc>, closure: LuaFn<'gc>, ic_idx: u16, shape: Shape<'gc>, slot: u32) {
    let proto_gc = closure.proto;
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
    closure: LuaFn<'gc>,
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
    fill_ic(ctx, closure, ic_idx, shape, slot);
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
        native: std::ptr::null(),
    };
    ts.top_lua()
        .expect("run_thread requires a seeded Lua frame");
    let (frame, closure) = top_frame(&mut ts);
    let (ip, base) = unsafe { ((*frame).pc, (*frame).base()) };
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
        frame,
        closure,
    );
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
    _frame: *mut LuaFrame<'gc>,
    _closure: LuaFn<'gc>,
) {
    let kind = ds.fault.take().expect("impl_error without a pending fault");
    // `ip` already points past the faulting instruction (see `dispatch!`),
    // which is the convention `LuaFrame::pc` uses.
    save_pc(thread, ip);
    let err = match kind {
        OpError::StackOverflow => crate::vm::debug::stack_overflow(ctx, thread),
        kind => {
            let msg = crate::vm::debug::op_error_message(ctx, thread, kind);
            crate::env::Error::from_str(ctx, &msg)
        }
    };
    thread.raise(ctx, err);
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, idx, ic_idx, _key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        tail!(gettabup_slow);
    };

    let cache = read_ic(closure, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape())
        && slot != InlineCache::ABSENT_SLOT
    {
        let v = unsafe { t_state.property_at(slot) };
        if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
            drop(t_state);
            *reg!(ref mut dst) = v;
            dispatch!();
        }
    }
    drop(t_state);
    tail!(gettabup_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let k = constant!(key);
    if let Some(t) = t_val.get_table() {
        fill_ic_for_constant_key(ctx, closure, ic_idx, t, k);
    }
    get_slow_body!(ctx, thread, registers, ip, handlers, ds, t_val, k, dst);
}

/// UpValue[idx][K[key]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_settabup<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        tail!(settabup_slow);
    };

    let v = reg!(src);
    let cache = read_ic(closure, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape())
        && slot != InlineCache::ABSENT_SLOT
    {
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
    drop(t_state);
    tail!(settabup_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let k = constant!(key);
    let v = reg!(src);
    if let Some(t) = t_val.get_table() {
        fill_ic_for_constant_key(ctx, closure, ic_idx, t, k);
    }
    set_slow_body!(ctx, thread, registers, ip, handlers, ds, t_val, k, v);
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        tail!(gettable_slow);
    };

    let k = reg!(key);
    let (v, need_index) = {
        let t_state = t.inner().borrow();
        let v = t_state.raw_get(k);
        let need = v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        tail!(gettable_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, key) = instruction.abc();
    let recv = reg!(table);
    let k = reg!(key);
    get_slow_body!(ctx, thread, registers, ip, handlers, ds, recv, k, dst);
}

/// R[table][R[key]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_settable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        tail!(settable_slow);
    };

    let k = reg!(key);
    let v = reg!(src);
    let needs_newindex = {
        let t_state = t.inner().borrow();
        t_state.shape().has_mm(MetamethodBits::NEWINDEX) && t_state.raw_get(k).is_nil()
    };

    if needs_newindex {
        tail!(settable_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, key) = instruction.abc();
    let (recv, k, v) = (reg!(table), reg!(key), reg!(src));
    set_slow_body!(ctx, thread, registers, ip, handlers, ds, recv, k, v);
}

/// R[dst] = R[table][K[key_idx]]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_getfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, ic_idx, _key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        tail!(getfield_slow);
    };

    let cache = read_ic(closure, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape())
        && slot != InlineCache::ABSENT_SLOT
    {
        let v = unsafe { t_state.property_at(slot) };
        if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
            drop(t_state);
            *reg!(ref mut dst) = v;
            dispatch!();
        }
    }
    drop(t_state);
    tail!(getfield_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    if let Some(t) = recv.get_table() {
        fill_ic_for_constant_key(ctx, closure, ic_idx, t, k);
    }
    get_slow_body!(ctx, thread, registers, ip, handlers, ds, recv, k, dst);
}

/// R[table][K[key_idx]] = R[src]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_setfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, ic_idx, key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        tail!(setfield_slow);
    };

    let v = reg!(src);
    let cache = read_ic(closure, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape())
        && slot != InlineCache::ABSENT_SLOT
    {
        let existing = unsafe { t_state.property_at(slot) };
        if !(existing.is_nil() && t_state.shape().has_mm(MetamethodBits::NEWINDEX)) {
            drop(t_state);
            let mut state = t.inner().borrow_mut(ctx.mutation());
            state.properties[slot as usize] = v;
            state.maybe_update_mt_bit(constant!(key_idx), v);
            dispatch!()
        }
    }
    drop(t_state);
    tail!(setfield_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    let v = reg!(src);
    if let Some(t) = recv.get_table() {
        fill_ic_for_constant_key(ctx, closure, ic_idx, t, k);
    }
    set_slow_body!(ctx, thread, registers, ip, handlers, ds, recv, k, v);
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        tail!(op_self_slow);
    };

    let key = constant!(key_idx);
    let (method, need_index) = {
        let recv_state = recv.inner().borrow();
        let v = recv_state.raw_get(key);
        let need = v.is_nil() && recv_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        tail!(op_self_slow);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let key = constant!(key_idx);

    // A table receiver gets here only after `op_self`'s raw miss.
    match walk_index_chain(ctx, recv_val, key) {
        IndexChain::Resolved(method) => {
            *reg!(ref mut dst) = method;
            *reg!(ref mut (dst + 1)) = recv_val;
            dispatch!();
        }
        IndexChain::Invoke { func, receiver } => {
            // Functional __index. Pre-place self at dst+1; the
            // continuation writes the resolved method into dst.
            *reg!(ref mut (dst + 1)) = recv_val;
            let cont = Continuation::StoreResult { dst };
            invoke_metamethod!(Value::function(func), &[receiver, key], cont, Index);
        }
        IndexChain::NotIndexable(v) => raise!(OpError::Index(v)),
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (dst, lhs, rhs) = instruction.abc();

            // Inline ints and floats only; boxed ints, overflow, zero divisors and mixes
            // all go to the slow handler so no call (and no stack frame) lands here.
            // Floats are the fall-through arm: whichever type loses pays a taken branch
            // per handler, which costs latency-bound float code (mandel ~15%) far more
            // than width-bound int code (primes ~7%).
            let (l, r) = (reg!(ref lhs), reg!(ref rhs));
            if std::hint::likely(l.is_float() && r.is_float()) {
                let (lf, rf) = (l.read_float(), r.read_float());
                reg!(ref mut dst).write_float(<$num_kind as num::ArithOp>::float_raw(lf, rf));
                dispatch!();
            } else if let Some((li, ri)) = Value::both_small(l, r) {
                if let Some(v) = <$num_kind as num::ArithOp>::small(li, ri) {
                    *reg!(ref mut dst) = v;
                    dispatch!();
                }
            }

            tail!($slow_name);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_arith_slow, $mm, Arith);
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (dst, lhs, rhs) = instruction.abc();

            let (l, r) = (reg!(ref lhs), reg!(ref rhs));
            if let Some(li) = l.get_small()
                && let Some(ri) = r.get_small()
                && let Some(v) = <$num_kind as num::BitOp>::small(li, ri)
            {
                *reg!(ref mut dst) = v;
                dispatch!();
            }

            tail!($slow_name);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_bit_slow, $mm, Bitwise);
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
        match num::$num_mix_h::<$num_kind>($ctx.mutation(), lhs, rhs) {
            num::SlowNum::Value(v) => {
                *reg!(ref mut dst) = v;
                dispatch!();
            }
            num::SlowNum::ModByZero => raise!(OpError::ModByZero),
            num::SlowNum::DivByZero => raise!(OpError::DivByZero),
            num::SlowNum::NotNumbers => {}
        }

        let meta_fn = binop_metamethod($ctx, lhs, rhs, $ctx.symbols().$mm);
        if meta_fn.is_nil() {
            raise!(OpError::$err(lhs, rhs));
        }

        let cont = Continuation::StoreResult { dst };
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

/// `R[dst] = R[src] <op> imm`, or `imm <op> R[src]` when `$swap`. Unlike the
/// register form the int/float mixes are inline: the constant side converts
/// for free.
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (dst, src, _) = instruction.abc_imm();
            let v = reg!(ref src);

            if std::hint::likely(instruction.imm_is_int()) {
                let k = instruction.imm_int();
                if let Some(i) = v.get_small() {
                    // Immediates are 31-bit, so both operands are i32.
                    let (l, r) = if $swap { (k as i32, i) } else { (i, k as i32) };
                    if let Some(out) = <$num_kind as num::ArithOp>::small(l, r) {
                        *reg!(ref mut dst) = out;
                        dispatch!();
                    }
                } else if v.is_float() {
                    let (f, k) = (v.read_float(), k as f64);
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    reg!(ref mut dst).write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                    dispatch!();
                }
            } else {
                let k = instruction.imm_float();
                if v.is_float() {
                    let f = v.read_float();
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    reg!(ref mut dst).write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                    dispatch!();
                } else if let Some(i) = v.get_small() {
                    let f = i as f64;
                    let (l, r) = if $swap { (k, f) } else { (f, k) };
                    reg!(ref mut dst).write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                    dispatch!();
                }
            }

            tail!($slow_name);
        }

        binop_imm_slow_handler!($slow_name, $num_kind, op_arith_slow, $mm, Arith, $swap);
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (dst, src, _) = instruction.abc_imm();
            debug_assert!(instruction.imm_is_int());
            let k = instruction.imm_int();

            if let Some(i) = reg!(ref src).get_small() {
                let (l, r) = if $swap { (k as i32, i) } else { (i, k as i32) };
                if let Some(out) = <$num_kind as num::BitOp>::small(l, r) {
                    *reg!(ref mut dst) = out;
                    dispatch!();
                }
            }

            tail!($slow_name);
        }

        binop_imm_slow_handler!($slow_name, $num_kind, op_bit_slow, $mm, Bitwise, $swap);
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (dst, src, flipped) = instruction.abc_imm();
            let (v, k) = (reg!(src), instruction.imm_value(ctx.mutation()));
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_small()
        && let Some(n) = i.checked_neg()
    {
        *reg!(ref mut dst) = Value::small(n);
        dispatch!();
    }
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), i.wrapping_neg());
        dispatch!();
    }
    if let Some(f) = val.get_float() {
        *reg!(ref mut dst) = Value::float(-f);
        dispatch!();
    }
    let meta_fn = ctx.metamethod_of(val, ctx.symbols().mm_unm);
    if meta_fn.is_nil() {
        raise!(OpError::Arith(val, val));
    }
    let cont = Continuation::StoreResult { dst };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_small() {
        *reg!(ref mut dst) = Value::small(!i);
        dispatch!();
    }
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), !i);
        dispatch!();
    }
    let meta_fn = ctx.metamethod_of(val, ctx.symbols().mm_bnot);
    if meta_fn.is_nil() {
        raise!(OpError::Bitwise(val, val));
    }
    let cont = Continuation::StoreResult { dst };
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);

    // Strings never consult __len; return byte length directly.
    if let Some(s) = val.get_string() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), s.len() as i64);
        dispatch!();
    }

    // Without `__len`, only a table has a length to fall back on.
    let meta_fn = ctx.metamethod_of(val, ctx.symbols().mm_len);
    if meta_fn.is_nil() {
        let Some(t) = val.get_table() else {
            raise!(OpError::Len(val));
        };
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), t.raw_len() as i64);
        dispatch!();
    }

    let cont = Continuation::StoreResult { dst };
    // Like the other unary metamethods, `__len` gets its operand twice.
    invoke_metamethod!(meta_fn, &[val, val], cont);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, lhs, rhs) = instruction.abc();
    let a = reg!(lhs);
    let b = reg!(rhs);
    // Fast path: both coerce to strings/numbers.
    let mut buf = Vec::new();
    if num::coerce_to_str(&mut buf, a) && num::coerce_to_str(&mut buf, b) {
        *reg!(ref mut dst) = Value::string(LuaString::new(ctx, &buf));
        dispatch!();
    }
    let meta_fn = binop_metamethod(ctx, a, b, ctx.symbols().mm_concat);
    if meta_fn.is_nil() {
        raise!(OpError::Concat(a, b));
    }
    let cont = Continuation::StoreResult { dst };
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let start = instruction.a();
    let base = unsafe { (*frame).base() };
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let val = instruction.a();
    let base = unsafe { (*frame).base() };
    thread.tbc_slots.push(base + val as usize);
    unsafe { (*frame).flags |= frame_flags::TBC };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
        let meta_fn = binop_metamethod(ctx, a, b, ctx.symbols().mm_eq);
        if !meta_fn.is_nil() {
            let cont = Continuation::CondJump { inverted };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if std::hint::likely(a.is_float() && b.is_float()) {
            Some(a.read_float() < b.read_float())
        } else if let Some((x, y)) = Value::both_small(a, b) {
            Some(x < y)
        } else if let Some(x) = a.get_integer()
            && let Some(y) = b.get_integer()
        {
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
    let meta_fn = binop_metamethod(ctx, a, b, ctx.symbols().mm_lt);
    if meta_fn.is_nil() {
        raise!(OpError::Compare(a, b));
    }
    let cont = Continuation::CondJump { inverted };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if std::hint::likely(a.is_float() && b.is_float()) {
            Some(a.read_float() <= b.read_float())
        } else if let Some((x, y)) = Value::both_small(a, b) {
            Some(x <= y)
        } else if let Some(x) = a.get_integer()
            && let Some(y) = b.get_integer()
        {
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
    let meta_fn = binop_metamethod(ctx, a, b, ctx.symbols().mm_le);
    if meta_fn.is_nil() {
        raise!(OpError::Compare(a, b));
    }
    let cont = Continuation::CondJump { inverted };
    invoke_metamethod!(meta_fn, &[a, b], cont);
}

/// `if (R[src] <cmp> imm) != inverted then skip`; `$swap` puts the immediate
/// on the left.
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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (src, inverted) = instruction.ab_imm_flag();
            let v = reg!(ref src);

            let primitive: Option<bool> = if std::hint::likely(instruction.imm_is_int()) {
                let k = instruction.imm_int();
                if let Some(i) = v.get_small() {
                    let i = i as i64;
                    Some(if $swap { $ii(k, i) } else { $ii(i, k) })
                } else if let Some(i) = v.get_integer() {
                    Some(if $swap { $ii(k, i) } else { $ii(i, k) })
                } else if v.is_float() {
                    let f = v.read_float();
                    Some(if $swap { $if_(k, f) } else { $fi(f, k) })
                } else {
                    None
                }
            } else {
                let k = instruction.imm_float();
                if v.is_float() {
                    let f = v.read_float();
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

            tail!($slow_name);
        }

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
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (src, inverted) = instruction.ab_imm_flag();
            let (v, k) = (reg!(src), instruction.imm_value(ctx.mutation()));
            let (a, b) = if $swap { (k, v) } else { (v, k) };
            let meta_fn = binop_metamethod(ctx, a, b, ctx.symbols().$mm);
            if meta_fn.is_nil() {
                raise!(OpError::Compare(a, b));
            }
            let cont = Continuation::CondJump { inverted };
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

/// `if (R[src] == imm) != inverted then skip`.
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, inverted) = instruction.ab_imm_flag();
    let v = reg!(ref src);

    let eq = if std::hint::likely(instruction.imm_is_int()) {
        let k = instruction.imm_int();
        if let Some(i) = v.get_small() {
            i == k as i32
        } else if let Some(i) = v.get_integer() {
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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

/// Write nil to `n` slots starting at `p`. The opaque pointer keeps this a
/// loop: as a `memset_pattern16` call it would give every handler using it a
/// stack frame, on the fast path too.
///
/// # Safety
/// `p..p + n` must be in bounds.
#[inline(always)]
unsafe fn fill_nil<'gc>(mut p: *mut Value<'gc>, n: usize) {
    for _ in 0..n {
        unsafe {
            p.write(Value::nil());
            p = p.add(1);
        }
        // The pointer is only threaded through, never read.
        #[allow(clippy::pointers_in_nomem_asm_block)]
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) p, options(nomem, nostack, preserves_flags));
        }
    }
}

/// Copy `n` values from `src` to `dst`, element by element in ascending
/// order, so `dst` may overlap `src` from below. Kept a loop like `fill_nil`.
///
/// # Safety
/// Both ranges must be in bounds, and `dst <= src` if they overlap.
#[inline(always)]
unsafe fn copy_values<'gc>(mut dst: *mut Value<'gc>, mut src: *const Value<'gc>, n: usize) {
    for _ in 0..n {
        unsafe {
            dst.write(src.read());
            dst = dst.add(1);
            src = src.add(1);
        }
        #[allow(clippy::pointers_in_nomem_asm_block)]
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) dst, options(nomem, nostack, preserves_flags));
        }
    }
}

/// Land `nret` values from `src` as `wanted` values at `dst`: the first
/// `min(nret, wanted)` copied, the rest nil. The copy is a plain loop, which
/// LLVM vectorizes for long result lists; the padding is `fill_nil`.
///
/// # Safety
/// `dst..dst + wanted` and `src..src + min(nret, wanted)` must be in bounds,
/// and `dst <= src` if they overlap.
#[inline(always)]
pub(crate) unsafe fn land_results<'gc>(
    dst: *mut Value<'gc>,
    src: *const Value<'gc>,
    nret: usize,
    wanted: usize,
) {
    let n = nret.min(wanted);
    for i in 0..n {
        unsafe { dst.add(i).write(src.add(i).read()) };
    }
    unsafe { fill_nil(dst.add(n), wanted - n) };
}

/// The native arm of CALL: run the callback inline and land its results at
/// `func_idx`. Expands inside a handler body (needs its `dispatch!`).
macro_rules! call_native {
    ($nc:expr, $func_idx:expr, $nargs:expr, $returns:expr, $base:expr,
     $ctx:ident, $thread:ident, $registers:ident, $ip:ident, $ds:ident, $frame:ident, $closure:ident) => {{
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
                    // Push `ExecKind::Error` so the executor's unwinder finds
                    // the nearest catching `ExecKind::Sequence` (e.g. the
                    // PCallSequence under coroutine.resume). Persist
                    // caller's pc first so re-entry would work if anything
                    // catches and resumes.
                    unsafe { (*$frame).pc = $ip };
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
                    // Results sit at `args_base = func_idx + 1`, one slot above
                    // where they land, inside the window `invoke_native` covered.
                    // The nil padding stays inside the caller's frame, which the
                    // CALL that entered it sized the vec for.
                    debug_assert!(args_base + retc.min(wanted) <= $thread.stack.len());
                    debug_assert!(func_idx + wanted <= $thread.stack.len());
                    let stack = $thread.stack.as_mut_ptr();
                    unsafe {
                        land_results(stack.add(func_idx), stack.add(args_base), retc, wanted)
                    };
                    // Publish the logical top. For MULTRET this is the dynamic
                    // count the next consumer reads.
                    $thread.set_top_unchecked(func_idx + wanted);
                    $registers = unsafe { $thread.stack.as_mut_ptr().add(base) };
                    dispatch!();
                }
                crate::vm::sequence::CallbackAction::Suspend(action) => {
                    // Suspension path: persist caller's pc, stash the
                    // action on the thread for the executor to translate
                    // into frame ops, then exit the dispatch chain.
                    unsafe { (*$frame).pc = $ip };
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
    // `grow`: make room for the callee's window and frame here (may call out).
    // `nogrow`: the caller has already checked both (`op_call` tail-calls
    // `op_call_grow` otherwise), so this arm stays call-free.
    (grow, $($rest:tt)*) => {
        call_lua!(@inner true, $($rest)*)
    };
    (nogrow, $($rest:tt)*) => {
        call_lua!(@inner false, $($rest)*)
    };
    (@inner $grow:literal, $callee:expr, $func_idx:expr, $nargs:expr, $returns:expr,
     $thread:ident, $registers:ident, $ip:ident, $ds:ident, $frame:ident, $closure:ident) => {{
        let callee: LuaFn<'gc> = $callee;
        let new_base = $func_idx + 1;
        unsafe { (*$frame).pc = $ip };
        if $grow {
            if !$thread.ensure_frame_slots(new_base + callee.max_stack_size as usize) {
                raise!(OpError::StackOverflow);
            }
        } else {
            debug_assert!(
                $thread.stack.len() >= new_base + callee.max_stack_size as usize
            );
        }
        let num_params = callee.num_params as usize;
        // Fixed-arg call with every parameter supplied is the common shape and
        // needs no counting; the rest (MULTRET via `thread.top`, missing
        // parameters, varargs) goes through the general accounting.
        let num_extras = if std::hint::likely(
            $nargs as usize > num_params && !callee.is_vararg,
        ) {
            0
        } else {
            // `nargs == 0` is the MULTRET sentinel: read the count from `thread.top`.
            let caller_provided = if $nargs == 0 {
                $thread.top - new_base
            } else {
                $nargs as usize - 1
            };
            // Inside the window sized above (grown here or checked by the
            // caller), so no bounds check — its panic path is the only call
            // this arm would otherwise make.
            debug_assert!(new_base + num_params <= $thread.stack.len());
            let stack = $thread.stack.as_mut_ptr();
            unsafe {
                fill_nil(
                    stack.add(new_base + caller_provided),
                    num_params.saturating_sub(caller_provided),
                )
            };
            if callee.is_vararg {
                caller_provided.saturating_sub(num_params) as u32
            } else {
                0
            }
        };
        $ip = callee.code;
        let frame = LuaFrame {
            closure: callee,
            base: new_base as u32,
            pc: $ip,
            num_results: $returns,
            flags: 0,
            num_extras,
            continuation: None,
        };
        if $grow {
            $thread.push_lua(frame);
            ($frame, $closure) = top_frame($thread);
        } else {
            // The new frame sits one slot above the running one, and the
            // closure is already in hand: no trip through `thread.frames`
            // (whose length we just stored) to find either.
            unsafe { $thread.push_lua_unchecked(frame) };
            $frame = unsafe { $frame.add(1) };
            $closure = callee;
        }
        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
        dispatch!();
    }};
}

/// R[func], ..., R[func+returns-2] = R[func](R[func+1], ..., R[func+args-1])
///
/// Only the plain-function cases live here: a Lua closure is entered inline,
/// a native one jumps to its `entry`, and anything
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs, returns) = instruction.abc();
    if let Some(f) = reg!(func).get_function() {
        match f.inner().as_ref() {
            FunctionKind::Lua(target) => {
                let func_idx = unsafe { (*frame).base() } + func as usize;
                let needed = func_idx + 1 + target.max_stack_size as usize;
                if std::hint::unlikely(thread.stack.len() < needed || thread.frames_full()) {
                    tail!(op_call_grow);
                }
                let callee = unsafe { LuaFn::from_function_unchecked(f) };
                call_lua!(
                    nogrow, callee, func_idx, nargs, returns, thread, registers, ip, ds, frame,
                    closure
                );
            }
            FunctionKind::Native(nc) => {
                ds.native = nc;
                let entry = nc.entry;
                tail!(entry);
            }
        }
    }
    tail!(op_call_meta);
}

/// CALL of a Lua closure that needs the value stack or the frame stack grown
/// first. Grows both (the only thing `op_call` cannot do without a stack
/// frame), then re-enters `op_call`, which has not modified anything yet; or
/// raises a stack overflow if the callee's window would cross the limit.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_grow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let func = instruction.a();
    let max_stack = match reg!(func).get_function().map(|f| f.inner().as_ref()) {
        Some(FunctionKind::Lua(closure)) => closure.max_stack_size as usize,
        _ => unreachable!("op_call_grow on a non-Lua callee"),
    };
    if !thread.ensure_frame_slots(unsafe { (*frame).base() } + func as usize + 1 + max_stack) {
        raise!(OpError::StackOverflow);
    }
    thread.reserve_frames(1);
    // Both vecs may have moved: rebind the frame pointer and the register window.
    (frame, closure) = top_frame(thread);
    registers = unsafe { thread.stack.as_mut_ptr().add((*frame).base()) };
    tail!(op_call);
}

/// Whether `v` is the native closure `nc`.
fn holds_native<'gc>(v: Value<'gc>, nc: &NativeClosure<'gc>) -> bool {
    matches!(
        v.get_function().map(|f| f.inner().as_ref()),
        Some(FunctionKind::Native(n)) if std::ptr::eq(n, nc)
    )
}

/// CALL or TAILCALL of a plain native function (no `__call` chain involved);
/// the default `NativeClosure::entry`.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn op_call_native<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    if instruction.op() == Op::TAILCALL {
        tail!(op_tailcall_native);
    }
    let (func, nargs, returns) = instruction.abc();
    let base = unsafe { (*frame).base() };
    let func_idx = base + func as usize;
    let nc = unsafe { &*ds.native };
    debug_assert!(holds_native(reg!(func), nc));
    call_native!(
        nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, ds, frame, closure
    );
}

/// What a one-argument math entry made of its argument.
enum Math1 {
    Float(f64),
    Small(i32),
    /// Not the common shape: leave the call to the full builtin.
    Miss,
}

/// Defines the entry (`NativeClosure::entry`) of a one-argument math builtin.
/// `$float`/`$small` map a float or inline-integer argument to a `Math1`;
/// every other shape, including extra arguments, goes to `op_call_native`.
/// The result lands straight in the function slot, and a TAILCALL returns it
/// from the frame.
macro_rules! math1_entry {
    ($name:ident, |$x:ident| $float:expr, |$i:ident| $small:expr) => {
        #[inline(never)]
        #[rustc_align(32)]
        pub(crate) extern "rust-preserve-none" fn $name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            // Raw reads: a TAILCALL is `Ab`-shaped, with `c` (unused then) zero.
            let (func, nargs, returns) = (instruction.a(), instruction.b(), instruction.c());
            let result = if nargs != 2 {
                Math1::Miss
            } else {
                // Read in place: `read_float` is a volatile load, and on a
                // copy of the slot it would round-trip through the stack.
                let arg = reg!(ref func + 1);
                if arg.is_float() {
                    let $x = arg.read_float();
                    $float
                } else if let Some($i) = arg.get_small() {
                    $small
                } else {
                    Math1::Miss
                }
            };
            let dst = reg!(ref mut func);
            match result {
                Math1::Float(f) => dst.write_float(f),
                Math1::Small(i) => *dst = Value::small(i),
                Math1::Miss => {
                    tail!(op_call_native);
                }
            }
            if instruction.op() == Op::TAILCALL {
                tail!(op_return1, Instruction::ret1(crate::instruction::Reg(func)));
            }
            if returns == 0 {
                thread.set_top_unchecked(unsafe { (*frame).base() } + func as usize + 1);
            } else {
                unsafe {
                    fill_nil(
                        registers.add(func as usize + 1),
                        (returns as usize).saturating_sub(2),
                    )
                };
            }
            dispatch!();
        }
    };
}

math1_entry!(ff_sqrt, |x| Math1::Float(x.sqrt()), |i| Math1::Float(
    f64::from(i).sqrt()
));
math1_entry!(ff_sin, |x| Math1::Float(x.sin()), |i| Math1::Float(
    f64::from(i).sin()
));
math1_entry!(ff_cos, |x| Math1::Float(x.cos()), |i| Math1::Float(
    f64::from(i).cos()
));
// `abs(i32::MIN)` leaves the inline range.
math1_entry!(ff_abs, |x| Math1::Float(x.abs()), |i| i
    .checked_abs()
    .map_or(Math1::Miss, Math1::Small));
// A rounded float becomes an integer when it fits inline; the boxed range is
// left to the builtin. `as` saturates and maps NaN to 0, so the round trip only
// holds for an integral value in i32 range (-0.0 becomes 0, as in Lua).
math1_entry!(
    ff_floor,
    |x| {
        let r = x.floor();
        if r as i32 as f64 == r {
            Math1::Small(r as i32)
        } else {
            Math1::Miss
        }
    },
    |i| Math1::Small(i)
);
math1_entry!(
    ff_ceil,
    |x| {
        let r = x.ceil();
        if r as i32 as f64 == r {
            Math1::Small(r as i32)
        } else {
            Math1::Miss
        }
    },
    |i| Math1::Small(i)
);

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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs, returns) = instruction.abc();
    let base = unsafe { (*frame).base() };
    let func_idx = base + func as usize;
    let (target, nargs) = match resolve_call_chain(ctx, thread, func_idx, nargs) {
        Ok(r) => r,
        Err(e) => raise!(e),
    };

    match target {
        CallTarget::Lua(callee) => {
            call_lua!(
                grow, callee, func_idx, nargs, returns, thread, registers, ip, ds, frame, closure
            );
        }
        CallTarget::Native(nc) => {
            call_native!(
                nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, ds, frame, closure
            );
        }
    }
}

/// Replace the running frame with a Lua `callee` whose arguments sit at
/// `stack[func_idx + 1..]`, and continue in it. The frame keeps its
/// `num_results`, continuation and flags: the callee returns to this frame's
/// caller on its behalf. Expects the callee's window to fit the stack.
macro_rules! tailcall_lua {
    ($callee:expr, $func_idx:expr, $nargs:expr,
     $thread:ident, $registers:ident, $ip:ident, $frame:ident, $closure:ident) => {{
        let callee: LuaFn<'gc> = $callee;
        // The results go to this frame's function slot in its caller, below
        // any varargs VARARGPREP moved `base` past.
        let new_base = unsafe { (*$frame).base() - (*$frame).num_extras as usize };
        debug_assert!($thread.stack.len() >= new_base + callee.max_stack_size as usize);
        let src = $func_idx + 1;
        // `nargs == 0` is MULTRET: read the count from `thread.top`.
        let nargs = if $nargs == 0 {
            $thread.top - src
        } else {
            $nargs as usize - 1
        };
        let num_params = callee.num_params as usize;
        let stack = $thread.stack.as_mut_ptr();
        // `new_base < src`, so a forward copy is safe for the overlap.
        unsafe {
            copy_values(stack.add(new_base), stack.add(src), nargs);
            fill_nil(
                stack.add(new_base + nargs),
                num_params.saturating_sub(nargs),
            );
        }
        $ip = callee.code;
        let frame = unsafe { &mut *$frame };
        frame.closure = callee;
        frame.pc = $ip;
        frame.set_base(new_base);
        frame.num_extras = if callee.is_vararg {
            nargs.saturating_sub(num_params) as u32
        } else {
            0
        };
        $closure = callee;
        $registers = unsafe { stack.add(new_base) };
        dispatch!();
    }};
}

/// return R[func](R[func+1], ..., R[func+args-1])  — tail call
///
/// Fast paths: a plain Lua callee whose window fits, from a frame with nothing
/// to close; and a plain native, which gets this TAILCALL as its `entry`'s
/// instruction and returns its results from this frame (as LuaJIT's fast
/// functions return through the frame's saved PC). Every other shape goes to
/// `op_tailcall_slow`.
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs) = instruction.ab();
    if let Some(f) = reg!(func).get_function() {
        match f.inner().as_ref() {
            FunctionKind::Lua(target) => {
                let (base, num_extras, flags) = unsafe {
                    (
                        (*frame).base(),
                        (*frame).num_extras as usize,
                        (*frame).flags,
                    )
                };
                let needed = base - num_extras + target.max_stack_size as usize;
                if std::hint::likely(
                    flags & (frame_flags::OPEN_UPVALUES | frame_flags::TBC) == 0
                        && thread.stack.len() >= needed,
                ) {
                    let callee = unsafe { LuaFn::from_function_unchecked(f) };
                    tailcall_lua!(
                        callee,
                        base + func as usize,
                        nargs,
                        thread,
                        registers,
                        ip,
                        frame,
                        closure
                    );
                }
            }
            FunctionKind::Native(nc) => {
                ds.native = nc;
                let entry = nc.entry;
                tail!(entry);
            }
        }
    }
    tail!(op_tailcall_slow);
}

/// TAILCALL of a plain native, reached through `op_call_native` or, after a
/// `__call` chain, `op_tailcall_slow`: run it, then return its results from
/// this frame. The frame is popped before a suspension or an error so that
/// lands on the caller's frame, as it would after a Lua tail call.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tailcall_native<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs) = instruction.ab();
    let base = unsafe { (*frame).base() };
    let func_idx = base + func as usize;
    let nc = unsafe { &*ds.native };
    debug_assert!(holds_native(reg!(func), nc));
    let args_base = func_idx + 1;
    let argc = if nargs == 0 {
        thread.top - args_base
    } else {
        nargs as usize - 1
    };
    let action = match invoke_native(ctx, thread, nc, args_base, argc) {
        Ok(a) => a,
        Err(err) => {
            // The message still names the tailcalling Lua frame (a native
            // never really tail calls in the reference either), so locate it
            // before popping that frame and installing the unwind marker.
            save_pc(thread, ip);
            let err = crate::vm::debug::locate(ctx, thread, err);
            close_upvalues(ctx.mutation(), thread, base);
            close_tbc_vars(ctx.mutation(), thread, base);
            thread.pop_lua();
            thread.push_exec(ExecKind::Error(err));
            return;
        }
    };
    match action {
        crate::vm::sequence::CallbackAction::Return => {
            // The results sit at `func + 1` up to `top`, as a RETURN of them
            // would find them; past its 8-bit count, MULTRET reads `top`.
            let retc = thread.top - args_base;
            registers = unsafe { thread.stack.as_mut_ptr().add(base) };
            if retc == 1 {
                tail!(
                    op_return1,
                    Instruction::ret1(crate::instruction::Reg(func + 1))
                );
            }
            let count = if retc < u8::MAX as usize {
                retc as u8 + 1
            } else {
                0
            };
            tail!(
                op_return,
                Instruction::ret(crate::instruction::Reg(func + 1), count)
            );
        }
        crate::vm::sequence::CallbackAction::Suspend(action) => {
            // The popped frame's function slot, `num_results` and
            // continuation carry the original caller's expectation across
            // the tail call.
            let (func_idx, num_results, cont) = {
                let f = unsafe { &*frame };
                let func_idx = f.base() - 1 - f.num_extras as usize;
                (func_idx, f.num_results, f.continuation)
            };
            if cont.is_some()
                && let Some(err) = lost_continuation(ctx, &action)
            {
                save_pc(thread, ip);
                thread.raise(ctx, err);
                return;
            }
            close_upvalues(ctx.mutation(), thread, base);
            close_tbc_vars(ctx.mutation(), thread, base);
            thread.pop_lua();
            thread.pending_action = Some(PendingAction {
                action,
                call_site: CallSite {
                    bottom: args_base,
                    func_idx,
                    returns: num_results,
                    cont,
                },
            });
        }
    }
}

/// TAILCALL through a `__call` chain, or into a Lua callee from a frame that
/// must close upvalues or grow the stack first.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tailcall_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs) = instruction.ab();
    let base = unsafe { (*frame).base() };
    let func_idx = base + func as usize;
    let (target, nargs) = match resolve_call_chain(ctx, thread, func_idx, nargs) {
        Ok(r) => r,
        Err(e) => raise!(e),
    };

    match target {
        CallTarget::Lua(callee) => {
            let new_base = base - unsafe { (*frame).num_extras as usize };
            if !thread.ensure_frame_slots(new_base + callee.max_stack_size as usize) {
                raise!(OpError::StackOverflow);
            }
            // Close upvalues before overwriting these slots with args, so an
            // open upvalue keeps referencing the local, not the arg value.
            // `TBC` stays set: luac never emits TAILCALL in a `<close>` scope;
            // ours still does (#188), which this can't paper over.
            close_upvalues(ctx.mutation(), thread, new_base);
            unsafe { (*frame).flags &= !frame_flags::OPEN_UPVALUES };
            tailcall_lua!(
                callee, func_idx, nargs, thread, registers, ip, frame, closure
            );
        }
        // The chain left the native in the function slot, and `nargs`
        // counts the callable objects it inserted as arguments; growing the
        // stack for them may have moved it.
        CallTarget::Native(nc) => {
            ds.native = nc;
            registers = unsafe { thread.stack.as_mut_ptr().add(base) };
            tail!(
                op_tailcall_native,
                Instruction::tailcall(crate::instruction::Reg(func), nargs)
            );
        }
    }
}

/// The tail shared by the RETURN fast paths once the results are in place
/// and `thread.top` is published: pop the frame and resume the Lua parent.
macro_rules! return_to_parent {
    ($thread:ident, $registers:ident, $ip:ident, $frame:ident, $closure:ident) => {{
        pop_to_parent!($thread, $registers, $ip, $frame, $closure);
        dispatch!();
    }};
}

/// Pop the running frame and rebind the handler state to its Lua parent.
macro_rules! pop_to_parent {
    ($thread:ident, $registers:ident, $ip:ident, $frame:ident, $closure:ident) => {{
        // `LuaFrame` is `Copy`, so nothing needs dropping.
        let n = $thread.frames.len();
        unsafe { $thread.frames.set_len(n - 1) };
        // The parent is the frame below (`flags` guaranteed it is a Lua
        // frame), so it is reached from the frame register rather than
        // through the length just stored.
        $frame = unsafe { $frame.sub(1) };
        let (new_base, new_ip, parent) =
            unsafe { ((*$frame).base(), (*$frame).pc, (*$frame).closure) };
        $closure = parent;
        $ip = new_ip;
        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
    }};
}

/// return R[values], ..., R[values+count-2]
///
/// Fast path: a fixed number of results, no continuation, nothing to close,
/// and a Lua caller. It makes no calls, so it needs no stack frame; every
/// other shape goes to `op_return_slow`.
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (values, count) = instruction.ab();

    let (cur_base, num_results, num_extras, flags) = {
        let f = unsafe { &*frame };
        (f.base(), f.num_results, f.num_extras as usize, f.flags)
    };
    // `flags` covers continuation, open upvalues, TBC slots and a non-Lua
    // parent (see `frame_flags`); `count == 0` is MULTRET.
    if std::hint::unlikely(count == 0 || flags != 0) {
        if count != 0 && flags == frame_flags::HAS_CONT {
            tail!(op_return_cont);
        }
        tail!(op_return_slow);
    }
    debug_assert!(!frame_has_open_upvalues(thread, cur_base));
    debug_assert!(thread.frames.len() >= 2 && thread.exec_depth() < thread.frames.len() - 1);

    let nret = count as usize - 1;
    let values_base = cur_base + values as usize;
    let dst_start = cur_base - 1 - num_extras;
    // `num_results == 0` is the CALL's MULTRET: deliver all `nret` and publish `thread.top`.
    let wanted = if num_results == 0 {
        nret
    } else {
        num_results as usize - 1
    };
    // `dst_start < values_base` and both ranges lie inside the callee's
    // window, which the CALL that entered it sized the vec for.
    debug_assert!(values_base + nret.min(wanted) <= thread.stack.len());
    debug_assert!(dst_start + wanted <= thread.stack.len());
    let stack = thread.stack.as_mut_ptr();
    unsafe { land_results(stack.add(dst_start), stack.add(values_base), nret, wanted) };
    // Without this a `top` left high by a multires producer inside the callee
    // would keep its dead registers traced (#43).
    thread.set_top_unchecked(dst_start + wanted);
    return_to_parent!(thread, registers, ip, frame, closure);
}

/// return
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return0<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (cur_base, num_results, num_extras, flags) = {
        let f = unsafe { &*frame };
        (f.base(), f.num_results, f.num_extras as usize, f.flags)
    };
    if std::hint::unlikely(flags != 0) {
        let generic = Instruction::ret(crate::instruction::Reg(0), 1);
        if flags == frame_flags::HAS_CONT {
            tail!(op_return_cont, generic);
        }
        tail!(op_return_slow, generic);
    }
    let dst_start = cur_base - 1 - num_extras;
    // `num_results == 0` is the CALL's MULTRET: zero results, publish `top`.
    let wanted = if num_results == 0 {
        0
    } else {
        num_results as usize - 1
    };
    debug_assert!(dst_start + wanted <= thread.stack.len());
    unsafe { fill_nil(thread.stack.as_mut_ptr().add(dst_start), wanted) };
    thread.set_top_unchecked(dst_start + wanted);
    return_to_parent!(thread, registers, ip, frame, closure);
}

/// return R[value]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return1<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let value = instruction.a();
    let (cur_base, num_results, num_extras, flags) = {
        let f = unsafe { &*frame };
        (f.base(), f.num_results, f.num_extras as usize, f.flags)
    };
    if std::hint::unlikely(flags != 0) {
        let generic = Instruction::ret(crate::instruction::Reg(value), 2);
        if flags == frame_flags::HAS_CONT {
            tail!(op_return_cont, generic);
        }
        tail!(op_return_slow, generic);
    }
    let dst_start = cur_base - 1 - num_extras;
    // `num_results == 0` is the CALL's MULTRET: one result, publish `top`.
    let wanted = if num_results == 0 {
        1
    } else {
        num_results as usize - 1
    };
    debug_assert!(dst_start + wanted.max(1) <= thread.stack.len());
    let stack = thread.stack.as_mut_ptr();
    let first = reg!(value);
    // Written even when the caller wants nothing: `dst_start` is the caller's
    // function slot, dead once the call returns.
    unsafe { *stack.add(dst_start) = first };
    unsafe { fill_nil(stack.add(dst_start + 1), wanted.saturating_sub(1)) };
    thread.set_top_unchecked(dst_start + wanted);
    return_to_parent!(thread, registers, ip, frame, closure);
}

/// RETURN of a fixed number of results from a metamethod or iterator frame
/// with nothing to close. Such a frame's parent is always the Lua frame that
/// invoked it, so this lands the results where `cont_resume` would read them,
/// pops back to the parent and applies the continuation, without
/// `frame_return`'s cleanup or a trip through `thread.frames`.
#[inline(never)]
#[rustc_align(32)]
// The incoming `ip` and `closure` belong to the frame this pops.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn op_return_cont<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (values, count) = instruction.ab();
    debug_assert!(count != 0);
    let nret = count as usize - 1;
    let (cont, func_slot) = {
        let f = unsafe { &*frame };
        debug_assert!(f.flags == frame_flags::HAS_CONT);
        (f.continuation, f.base() - 1 - f.num_extras as usize)
    };
    // The function slot is below the values, inside the frame's window.
    unsafe {
        let stack = thread.stack.as_mut_ptr();
        copy_values(stack.add(func_slot), registers.add(values as usize), nret);
    }
    thread.set_top_unchecked(func_slot + nret);
    pop_to_parent!(thread, registers, ip, frame, closure);
    // `HAS_CONT` is set exactly when `continuation` is.
    let cont = unsafe { cont.unwrap_unchecked() };
    apply_cont_payload!(
        cont, func_slot, nret, ctx, thread, registers, ip, handlers, ds
    );
}

/// The general RETURN: MULTRET, continuations, open upvalues / to-be-closed
/// variables, and returns into a non-Lua parent or out of the thread.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (values, count) = instruction.ab();

    let cur_base = unsafe { (*frame).base() };
    let values_base = cur_base + values as usize;
    // `count == 0` is MULTRET: read the count from `thread.top`.
    let nret = if count == 0 {
        thread.top - values_base
    } else {
        count as usize - 1
    };

    match frame_return(ctx.mutation(), thread, frame, values_base, nret) {
        FrameReturn::Continuation => {
            tail!(cont_resume);
        }
        FrameReturn::TopLevel | FrameReturn::ToNonLua => {}
        FrameReturn::Caller { new_base, new_ip } => {
            ip = new_ip;
            (frame, closure) = top_frame(thread);
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
    use crate::builtin::util::Number;
    let num = match limit.get_string() {
        Some(s) => crate::builtin::util::str_to_number(s.as_bytes())?,
        None => match (limit.get_integer(), limit.get_float()) {
            (Some(i), _) => Number::Int(i),
            (_, Some(f)) => Number::Float(f),
            _ => return None,
        },
    };
    let lim = match num {
        Number::Int(i) => i,
        Number::Float(f) => {
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
        }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
                *reg!(ref mut base) = Value::integer(ctx.mutation(), last);
                *reg!(ref mut base + 1) = Value::integer(ctx.mutation(), s);
                *reg!(ref mut base + 2) = Value::integer(ctx.mutation(), i);
                *reg!(ref mut base + 3) = Value::integer(ctx.mutation(), i);
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, offset) = instruction.a_imm();

    // The step's type tells the loop kind, and the hidden slots match it:
    // `op_forprep` wrote them and nothing else can (they are unnamed, and
    // the visible copy is `<const>` and never read here).
    let step = reg!(base + 1);
    if let Some(s) = step.get_small()
        && let Some((last, idx)) = Value::both_small(reg!(ref base), reg!(ref base + 2))
    {
        // `idx` walks init, init+step, ..., last exactly, so `idx != last`
        // also guarantees `idx + step` stays in range.
        if idx != last {
            let idx = Value::small(idx.wrapping_add(s));
            *reg!(ref mut base + 2) = idx;
            *reg!(ref mut base + 3) = idx;
            ip = unsafe { ip.offset(offset as isize) };
        }
    } else if let Some(s) = step.get_float() {
        let (lim, idx) = (reg!(base), reg!(base + 2));
        let lim = unsafe { lim.get_float().unwrap_unchecked() };
        let idx = unsafe { idx.get_float().unwrap_unchecked() } + s;
        if if 0.0 < s { idx <= lim } else { lim <= idx } {
            let idx = Value::float(idx);
            *reg!(ref mut base + 2) = idx;
            *reg!(ref mut base + 3) = idx;
            ip = unsafe { ip.offset(offset as isize) };
        }
    } else {
        // Kept out of line: boxing the new index allocates, which would give
        // this handler a stack frame.
        tail!(forloop_slow);
    }

    dispatch!();
}

/// `op_forloop` for an integer loop whose values don't all fit a small int.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn forloop_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, offset) = instruction.a_imm();
    let (s, last, idx) = (reg!(base + 1), reg!(base), reg!(base + 2));
    let (s, last, idx) = unsafe {
        (
            s.get_integer().unwrap_unchecked(),
            last.get_integer().unwrap_unchecked(),
            idx.get_integer().unwrap_unchecked(),
        )
    };
    if idx != last {
        let idx = Value::integer(ctx.mutation(), idx.wrapping_add(s));
        *reg!(ref mut base + 2) = idx;
        *reg!(ref mut base + 3) = idx;
        ip = unsafe { ip.offset(offset as isize) };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, count) = instruction.ab();
    let iter = reg!(base);
    let state = reg!(base + 1);
    let control = reg!(base + 2);
    let cont = Continuation::TForCall { base, count };
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (table, count, offset) = instruction.abd();
    let Some(t) = reg!(table).get_table() else {
        raise!(OpError::Internal("SETLIST on a non-table"));
    };
    // `count == 0` is MULTRET: element count comes from `thread.top`. It can
    // exceed u8 (a `VARARG count=0` spread), so index the stack by usize.
    let (base, max_stack) = (unsafe { (*frame).base() }, closure.max_stack_size as usize);
    let elements_start = base + table as usize + 1;
    let n = if count == 0 {
        thread.top - elements_start
    } else {
        count as usize
    };
    let off = offset as i64;
    for i in 1..=n {
        let val = thread.stack[elements_start + i - 1];
        let key = Value::integer(ctx.mutation(), off + i as i64);
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, proto_idx) = instruction.ad();
    let (parent_closure, base) = unsafe { ((*frame).closure, (*frame).base()) };
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
                    unsafe { (*frame).flags |= frame_flags::OPEN_UPVALUES };
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, count) = instruction.ab();
    let (base, num_extras) = unsafe { ((*frame).base(), (*frame).num_extras as usize) };
    let proto = closure.proto;
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
            thread.stack[target + i] = t.raw_get(Value::integer(ctx.mutation(), i as i64 + 1));
        }
        if count == 0 {
            thread.top = new_top;
        }
        dispatch!();
    }

    // Optimized: read the below-base region directly. The extras end at `base`
    // and the target starts at or above it, so the ranges don't overlap.
    let extras_start = base - num_extras;
    // `count == 0` is MULTRET: all extras, published through `top`.
    let wanted = if count == 0 {
        num_extras
    } else {
        count as usize - 1
    };
    if count == 0 {
        thread.ensure_slots(target + wanted);
        registers = unsafe { thread.stack.as_mut_ptr().add(base) };
        thread.top = target + wanted;
    }
    debug_assert!(target + wanted <= thread.stack.len());
    let stack = thread.stack.as_mut_ptr();
    unsafe {
        land_results(
            stack.add(target),
            stack.add(extras_start),
            num_extras,
            wanted,
        )
    };
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, _base, key) = instruction.abc();
    let key_val = reg!(key);
    let num_extras = unsafe { (*frame).num_extras as usize };
    let extras_start = unsafe { (*frame).base() } - num_extras;
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
            Value::integer(ctx.mutation(), num_extras as i64)
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let _num_fixed = instruction.a();
    let (num_extras, base) = unsafe { ((*frame).num_extras as usize, (*frame).base()) };
    let num_params = closure.num_params as usize;
    let max_stack = closure.max_stack_size as usize;
    let new_base = if num_extras > 0 {
        let new_base = base + num_extras;
        if !thread.ensure_frame_slots(new_base + max_stack) {
            raise!(OpError::StackOverflow);
        }
        let total = num_extras + num_params;
        // [fixed..., extras...].rotate_left(num_params) => [extras..., fixed...]
        thread.stack[base..base + total].rotate_left(num_params);
        unsafe { (*frame).set_base(new_base) };
        registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
        new_base
    } else {
        base
    };
    if closure.proto.needs_vararg_table {
        // Store into the stack slot before filling so a mid-fill alloc can't
        // collect the table (mirrors `op_newtable`).
        let extras_start = new_base - num_extras;
        let table = Table::new(ctx);
        thread.stack[new_base + num_params] = Value::table(table);
        for i in 0..num_extras {
            let v = thread.stack[extras_start + i];
            table.raw_set(ctx, Value::integer(ctx.mutation(), i as i64 + 1), v);
        }
        table.raw_set(
            ctx,
            Value::string(LuaString::new(ctx, b"n")),
            Value::integer(ctx.mutation(), num_extras as i64),
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
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
    _frame: *mut LuaFrame<'gc>,
    _closure: LuaFn<'gc>,
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
/// `ip` points into the proto's `code` (the handler chain only ever advances
/// it within that slice), which `LuaFrame::pc_index` relies on.
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
    let stack = Stack::new(thread, args_base);
    // The stack is grown-not-shrunk and may leave dead scratch above the logical
    // top; that's fine because `ThreadState`'s `Collect` traces only the live
    // high-water (derived from the frames + `top`), so dead slots never retain.
    (nc.function)(ctx, nc, stack)
}

/// What should happen after a frame returns with values at
/// `stack[values_base .. values_base + nret]`. Produced by [`frame_return`],
/// consumed by `op_return_slow`.
pub(crate) enum FrameReturn {
    /// A continuation was attached to the departing frame, which is still on
    /// top with its results at its function slot, up to `top`; caller must
    /// tail-call `cont_resume`.
    Continuation,
    /// The departing frame was the outermost one; thread is now `Result`.
    /// Caller should return from the handler.
    TopLevel,
    /// Normal return to the caller frame, which is now the top frame. Caller
    /// rebinds its dispatch registers to it and dispatches.
    Caller {
        new_base: usize,
        new_ip: *const Instruction,
    },
    /// The popped Lua frame's parent is a non-Lua frame (Sequence /
    /// WaitThread / Start / Error). The values have been left at the frame's
    /// function slot, up to `top`, for the executor's driver loop to consume
    /// on the next pump. `op_return_slow` returns to exit dispatch.
    ToNonLua,
}

/// Unwind the top-of-stack frame assuming it returned the values at
/// `stack[values_base .. values_base + nret]`; the general case of RETURN.
#[inline]
pub(crate) fn frame_return<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    frame: *mut LuaFrame<'gc>,
    values_base: usize,
    nret: usize,
) -> FrameReturn {
    let (cur_base, num_results, num_extras, has_cont) = {
        let f = unsafe { &*frame };
        (
            f.base(),
            f.num_results,
            f.num_extras as usize,
            f.continuation.is_some(),
        )
    };

    // Before the results move: they may land on the frame's own registers.
    if frame_has_open_upvalues(thread, cur_base) {
        close_upvalues_slow(mc, thread, cur_base);
    }
    close_tbc_vars(mc, thread, cur_base);

    // The func slot sits at `cur_base - 1 - num_extras`: VARARGPREP shifted
    // base past the extras at `[cur_base - num_extras .. cur_base]` (0 for
    // non-vararg frames).
    let dst_start = cur_base - 1 - num_extras;

    if has_cont {
        thread
            .stack
            .copy_within(values_base..values_base + nret, dst_start);
        thread.set_top_unchecked(dst_start + nret);
        return FrameReturn::Continuation;
    }
    thread.pop_lua();

    let (new_base, new_ip) = match thread.top_lua() {
        Some(caller) => (caller.base(), caller.pc),
        None if thread.frames_empty() => {
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
        // executor driver picks up here with all `nret` values as its input
        // window `stack[dst_start..top]`. The slots above stay allocated for
        // the next call; the collector clears them.
        None => {
            thread
                .stack
                .copy_within(values_base..values_base + nret, dst_start);
            thread.set_top_unchecked(dst_start + nret);
            return FrameReturn::ToNonLua;
        }
    };

    // `num_results == 0` is the CALL's MULTRET: deliver all `nret` and publish `thread.top`.
    if num_results == 0 {
        thread
            .stack
            .copy_within(values_base..values_base + nret, dst_start);
        thread.set_top_unchecked(dst_start + nret);
    } else {
        let wanted = num_results as usize - 1;
        // `dst_start < values_base` (the func slot is below the callee's
        // registers) and both ranges lie inside the callee's window, which
        // `op_call` sized the vec for.
        debug_assert!(values_base + nret.min(wanted) <= thread.stack.len());
        debug_assert!(dst_start + wanted <= thread.stack.len());
        let stack = thread.stack.as_mut_ptr();
        unsafe { land_results(stack.add(dst_start), stack.add(values_base), nret, wanted) };
        // Publish the landing end. Without this, a `top` left high by a multires
        // producer *inside the callee* would still be the high-water long after
        // the callee popped, so `live_top` would keep tracing its dead registers
        // — the exact #43 leak, just via a stale `top` instead of the vec length.
        thread.set_top_unchecked(dst_start + wanted);
    }

    FrameReturn::Caller { new_base, new_ip }
}

/// Whether the frame based at `base` still has open upvalues. Open upvalues
/// are appended in creation order and every deeper frame closes its own before
/// returning, so only the tail of the list can belong to the returning frame.
/// (Not valid for a partial `CLOSE` inside a frame, whose entries are not
/// ordered by index.)
#[inline(always)]
fn frame_has_open_upvalues<'gc>(thread: &ThreadState<'gc>, base: usize) -> bool {
    thread.open_upvalues.last().is_some_and(
        |uv| matches!(&*uv.borrow(), UpvalueState::Open { index, .. } if *index >= base),
    )
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
pub(crate) const MAX_TAG_LOOP: usize = 2000;

/// Result of walking an `__index` chain.
pub(crate) enum IndexChain<'gc> {
    /// The chain resolved synchronously to a value (possibly `Nil`).
    Resolved(Value<'gc>),
    /// The chain ended in a function that must be invoked with
    /// `(receiver, key)`, `receiver` being the value whose metatable held it.
    Invoke {
        func: Function<'gc>,
        receiver: Value<'gc>,
    },
    /// A non-table in the chain has no `__index`; the caller raises
    /// "attempt to index" on it.
    NotIndexable(Value<'gc>),
    /// Chain depth exceeded `MAX_TAG_LOOP`; caller should raise.
    Exhausted,
}

/// Resolve `receiver[key]` through `__index` (`luaV_finishget`), given that
/// `receiver`'s own raw lookup, if it is a table, already missed.
pub(crate) fn walk_index_chain<'gc>(
    ctx: Context<'gc>,
    mut receiver: Value<'gc>,
    key: Value<'gc>,
) -> IndexChain<'gc> {
    let index = ctx.symbols().mm_index;
    for _ in 0..MAX_TAG_LOOP {
        let mm = match receiver.get_table() {
            Some(t) => {
                if !t.shape().has_mm(MetamethodBits::INDEX) {
                    return IndexChain::Resolved(Value::nil());
                }
                // INDEX bit implies metatable is Some.
                let mt = unsafe { t.metatable().unwrap_unchecked() };
                mt.raw_get(Value::string(index))
            }
            None => {
                let mm = ctx.metamethod_of(receiver, index);
                if mm.is_nil() {
                    return IndexChain::NotIndexable(receiver);
                }
                mm
            }
        };
        if let Some(func) = mm.get_function() {
            return IndexChain::Invoke { func, receiver };
        }
        if let Some(t) = mm.get_table() {
            let v = t.raw_get(key);
            if !v.is_nil() {
                return IndexChain::Resolved(v);
            }
        }
        receiver = mm;
    }
    IndexChain::Exhausted
}

/// Result of walking a `__newindex` chain.
enum NewIndexChain<'gc> {
    /// Raw-assign `value` into this table.
    RawSet(Table<'gc>),
    /// The chain ended in a function; invoke with `(receiver, key, value)`.
    Invoke {
        func: Function<'gc>,
        receiver: Value<'gc>,
    },
    /// A non-table in the chain has no `__newindex`; the caller raises.
    NotIndexable(Value<'gc>),
    /// Chain depth exceeded `MAX_TAG_LOOP`; caller should raise.
    Exhausted,
}

/// Find where `t[key] = v` lands (`luaV_finishset`): the first table that
/// already has `key` or lacks `__newindex`, or a function `__newindex`.
#[inline]
fn walk_newindex_chain<'gc>(
    ctx: Context<'gc>,
    mut t: Value<'gc>,
    key: Value<'gc>,
) -> NewIndexChain<'gc> {
    let newindex = ctx.symbols().mm_newindex;
    for _ in 0..MAX_TAG_LOOP {
        let mm = match t.get_table() {
            Some(tbl) => {
                if !tbl.raw_get(key).is_nil() || !tbl.shape().has_mm(MetamethodBits::NEWINDEX) {
                    return NewIndexChain::RawSet(tbl);
                }
                // NEWINDEX bit implies metatable is Some.
                let mt = unsafe { tbl.metatable().unwrap_unchecked() };
                let mm = mt.raw_get(Value::string(newindex));
                if mm.is_nil() {
                    return NewIndexChain::RawSet(tbl);
                }
                mm
            }
            None => {
                let mm = ctx.metamethod_of(t, newindex);
                if mm.is_nil() {
                    return NewIndexChain::NotIndexable(t);
                }
                mm
            }
        };
        if let Some(func) = mm.get_function() {
            return NewIndexChain::Invoke { func, receiver: t };
        }
        t = mm;
    }
    NewIndexChain::Exhausted
}

/// The resolved target of a call: either a Lua bytecode closure (which the
/// caller must push a frame for) or a native Rust callback (which the caller
/// invokes inline).
pub(crate) enum CallTarget<'gc> {
    Lua(LuaFn<'gc>),
    Native(&'gc NativeClosure<'gc>),
}

/// Outcome of [`schedule_meta_call`], consumed by the `invoke_metamethod!`
/// macro. Captures the three ways a continuation-driven call can proceed.
pub(crate) enum MetaDispatch {
    /// Resolved to a Lua closure; a frame carrying the continuation was
    /// pushed. The caller rebinds its dispatch registers to it and dispatches.
    Lua {
        new_ip: *const Instruction,
        new_base: usize,
    },
    /// Resolved to a native callback that returned synchronously. Its results
    /// sit at `stack[results_base .. results_base + nret]`; the caller applies
    /// the continuation payload inline.
    NativeReturn { results_base: usize, nret: usize },
    /// Native callback suspended (Call/Yield/Resume/Sequence) or errored — a
    /// `pending_action` or `ExecKind::Error` was installed for the executor. The
    /// caller returns to exit dispatch.
    Suspended,
    /// Target is not callable (or a suspending comparison metamethod, which we
    /// don't support). The caller raises.
    Unresolvable,
    /// The target's `__call` chain is too long. The caller raises.
    CallChainTooLong,
    /// The call would cross the stack limit. The caller raises.
    StackOverflow,
}

/// `__call` hops a call may take before "'__call' chain too long", like the
/// reference's 4-bit `CIST_CCMT` counter.
const MAX_CALL_CHAIN: usize = 15;

/// Walk the `__call` chain at `thread.stack[func_idx]` until we hit a
/// callable target, shifting args right by one on each hop to prepend the
/// current callee as the first argument (Lua 5.5 `tryfuncTM` behavior).
/// Returns the resolved target and the (possibly adjusted) `nargs`, or the
/// error to raise.
#[inline(always)]
pub(crate) fn resolve_call_chain<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    func_idx: usize,
    nargs: u8,
) -> Result<(CallTarget<'gc>, u8), OpError<'gc>> {
    // Plain functions are the overwhelmingly common case; keep the `__call`
    // walk (and its stack frame) out of the handler.
    if let Some(f) = thread.stack[func_idx].get_function() {
        return Ok(match f.inner().as_ref() {
            FunctionKind::Lua(_) => (
                CallTarget::Lua(unsafe { LuaFn::from_function_unchecked(f) }),
                nargs,
            ),
            FunctionKind::Native(nc) => (CallTarget::Native(nc), nargs),
        });
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
) -> Result<(CallTarget<'gc>, u8), OpError<'gc>> {
    let mut hops = 0;
    loop {
        let func_val = thread.stack[func_idx];
        if let Some(f) = func_val.get_function() {
            return Ok(match f.inner().as_ref() {
                FunctionKind::Lua(_) => (
                    CallTarget::Lua(unsafe { LuaFn::from_function_unchecked(f) }),
                    nargs,
                ),
                FunctionKind::Native(nc) => (CallTarget::Native(nc), nargs),
            });
        }
        let mm = ctx.metamethod_of(func_val, ctx.symbols().mm_call);
        if mm.is_nil() {
            return Err(OpError::Call(func_val));
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
        if hops == MAX_CALL_CHAIN {
            return Err(OpError::CallChainTooLong);
        }
        hops += 1;
        // A count that no longer fits in `nargs` moves to `thread.top`, as
        // MULTRET's does.
        if nargs == 0 || nargs == u8::MAX {
            thread.set_top(func_idx + 2 + actual_args);
            nargs = 0;
        } else {
            nargs += 1;
        }
    }
}

/// Binary metamethod `name`, taken from `lhs` first, then `rhs`.
#[inline]
pub(crate) fn binop_metamethod<'gc>(
    ctx: Context<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
    name: LuaString<'gc>,
) -> Value<'gc> {
    let m = ctx.metamethod_of(lhs, name);
    if !m.is_nil() {
        return m;
    }
    ctx.metamethod_of(rhs, name)
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
/// needed for the suspend case. A native error installs an `ExecKind::Error`.
///
/// The native suspend path replays all four continuation payloads uniformly via
/// `apply_native_continuation`: `StoreResult`, `IgnoreResult`, `TForCall`, and
/// `CondJump` (which skips the caller's next instruction when the suspended
/// comparison's result selects that branch).
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
    let caller = thread
        .top_lua()
        .expect("schedule_meta_call called without an active Lua frame");
    let caller_base = caller.base();
    let scratch_func = caller_base + caller.closure.max_stack_size as usize;
    let new_base = scratch_func + 1;

    // Stage meta_fn + args so resolve_call_chain sees them in op_call layout.
    if !thread.ensure_frame_slots(new_base + args.len()) {
        return MetaDispatch::StackOverflow;
    }
    thread.stack[scratch_func] = meta_fn;
    for (i, &a) in args.iter().enumerate() {
        thread.stack[new_base + i] = a;
    }

    // Walk any __call chain. nargs follows op_call's convention (includes the
    // function slot), so `args.len() + 1`.
    debug_assert!(args.len() < u8::MAX as usize);
    let nargs = (args.len() + 1) as u8;
    let (target, final_nargs) = match resolve_call_chain(ctx, thread, scratch_func, nargs) {
        Ok(r) => r,
        Err(OpError::CallChainTooLong) => return MetaDispatch::CallChainTooLong,
        Err(_) => return MetaDispatch::Unresolvable,
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
    if !thread.ensure_frame_slots(new_base + closure.max_stack_size as usize) {
        return MetaDispatch::StackOverflow;
    }

    // Nil-fill any parameter slots not covered by the (possibly shifted) args.
    let num_params = closure.num_params as usize;
    for i in actual_args..num_params {
        thread.stack[new_base + i] = Value::nil();
    }
    let num_extras = if closure.is_vararg {
        actual_args.saturating_sub(num_params) as u32
    } else {
        0
    };

    thread.push_lua(LuaFrame {
        closure,
        base: new_base as u32,
        pc: closure.code,
        // Unused: RETURN hands the results to the continuation instead.
        num_results: 0,
        flags: frame_flags::HAS_CONT,
        num_extras,
        continuation: Some(cont),
    });

    let new_ip = closure.code;
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
            // Mirror op_call's native-error path: an `ExecKind::Error` lets the
            // executor's unwinder route to the nearest catcher (e.g. pcall).
            thread.raise(ctx, err);
            return MetaDispatch::Suspended;
        }
    };

    match action {
        CallbackAction::Return => {
            // Result count via the logical top; the staged scratch window sits
            // above the caller frame, so nothing shrank the caller's window.
            let nret = thread.top - args_base;
            MetaDispatch::NativeReturn {
                results_base: args_base,
                nret,
            }
        }
        // Suspending native. Park the full continuation on the `CallSite`; when
        // the suspension resolves, its results funnel through
        // `land_call_results`, which applies the continuation against the
        // caller frame (`apply_native_continuation`). This replays every
        // payload uniformly — including `CondJump`, whose branch decision (a
        // `pc` bump) can't be expressed as a plain landing.
        CallbackAction::Suspend(action) => {
            if let Some(err) = lost_continuation(ctx, &action) {
                thread.raise(ctx, err);
                return MetaDispatch::Suspended;
            }
            thread.pending_action = Some(PendingAction {
                action,
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

/// The error for a native that answers a call whose results a `Continuation`
/// must receive with a `Call` and no follow-up sequence: its callee returns
/// straight to the caller, so nothing is left to apply the continuation. A
/// follow-up sequence's results land through `land_call_results`, which does.
fn lost_continuation<'gc>(
    ctx: Context<'gc>,
    action: &crate::vm::sequence::Suspend<'gc>,
) -> Option<crate::env::Error<'gc>> {
    matches!(action, crate::vm::sequence::Suspend::Call { then: None }).then(|| {
        crate::env::Error::from_str(
            ctx,
            "metamethod/iterator native cannot tail-call into Lua across the continuation",
        )
    })
}

/// Where `op_return_slow` goes once `frame_return` has handled a frame carrying
/// a [`Continuation`] (MULTRET, or something to close); `op_return_cont`
/// covers the rest. Pops the callee frame, restores the caller's
/// `ip`/`registers`, then applies the payload to the results `frame_return`
/// left at the callee's function slot. The synchronous-native path in
/// `invoke_metamethod!` applies the same payload via `apply_cont_payload!`
/// without this frame teardown, since no callee frame exists there.
#[inline(never)]
#[rustc_align(32)]
// The incoming `ip`, `registers` and `closure` belong to the popped frame.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn cont_resume<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (cont, results_base) = {
        let f = unsafe { &*frame };
        let func_slot = f.base() - 1 - f.num_extras as usize;
        (f.continuation.unwrap(), func_slot)
    };
    // `frame_return` already closed upvalues and to-be-closed variables.
    thread.pop_lua();
    (frame, closure) = top_frame(thread);
    let caller_base = unsafe {
        ip = (*frame).pc;
        (*frame).base()
    };
    registers = unsafe { thread.stack.as_mut_ptr().add(caller_base) };
    apply_cont_payload!(
        cont,
        results_base,
        thread.top - results_base,
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds
    );
}
