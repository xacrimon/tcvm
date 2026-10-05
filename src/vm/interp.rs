use crate::dmm::{Gc, Mutation, RefLock};
use crate::env::function::{
    Function, FunctionKind, InlineCache, LuaFn, NativeClosure, Stack, Upvalue, UpvalueState,
};
use crate::env::shape::{MAX_PROPERTIES_FAST, MetamethodBits, Shape, mirrored};
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, Step, Table, TableState};
use crate::env::thread::{
    CallSite, ExecKind, LuaFrame, PendingAction, TbcEntry, Thread, ThreadState, ThreadStatus,
    frame_flags,
};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, Op, TFOR_VARS, UpValueDescriptor};
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
    /// A named vararg table's `n` isn't an integer in `0..=i32::MAX / 2`.
    VarargN,
    /// TBC: the register holds a value without `__close`.
    NonClosable(u8),
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
    /// The continuation `meta_call` gives the call it makes.
    ret: Handler,
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
) -> Exit;

/// Why the dispatch chain returned to `run_thread`.
pub(crate) enum Exit {
    /// The executor takes over from the thread state (return from the entry
    /// frame, suspension, error).
    End,
    /// Allocation crossed the GC check threshold; the host should collect
    /// before stepping again.
    Gc,
}

/// The arguments a continuation (`LuaFrame::ret`) gets: the `nret` results at
/// `values`, the finished call's function slot `func_slot` (passed in the `ip`
/// register), and `frame`, the frame on top again. The caller's state comes
/// from `frame`; its instruction at `pc - 1` is the one that made the call.
macro_rules! ret_args {
    ($instruction:ident, $registers:ident, $ip:ident) => {
        (
            $instruction.raw() as usize,
            $registers,
            $ip as *mut Value<'gc>,
        )
    };
}

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

        /// Exit to the executor once allocation may owe the collector work;
        /// execution resumes at the next instruction.
        #[allow(unused_macros)]
        macro_rules! gc_check {
            () => {{
                if std::hint::unlikely($ctx.mutation().metrics().gc_check_due()) {
                    unsafe { (*$frame).pc = $ip };
                    return Exit::Gc;
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
                tail!($$f, $$instruction, $ip)
            };
            ($$f:ident, $$instruction:expr, $$ip:expr) => {
                become $$f(
                    $$instruction,
                    $ctx,
                    $thread,
                    $registers,
                    $$ip,
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

        /// Continue as a [`NativeStep`] says: in the Lua frame it pushed, in the
        /// continuation of the native frame that returned, or out of dispatch.
        #[allow(unused_macros)]
        macro_rules! native_step {
            ($$step:expr) => {{
                match $$step {
                    NativeStep::EnterLua => {
                        ($frame, $closure) = top_frame($thread);
                        let __base = unsafe { (*$frame).base() };
                        $ip = unsafe { (*$frame).pc };
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(__base) };
                        dispatch!();
                    }
                    NativeStep::Return {
                        ret: __ret,
                        func_slot: __fs,
                        values: __vals,
                    } => {
                        let __nret = $thread.top - __vals;
                        let __stack = $thread.stack.as_mut_ptr();
                        // The frame below, or one before the first when there
                        // is none (the continuation is then `ret_exit`).
                        let __top = $thread
                            .frames
                            .as_mut_ptr()
                            .wrapping_add($thread.frames.len())
                            .wrapping_sub(1);
                        become __ret(
                            Instruction::from_raw(__nret as u64),
                            $ctx,
                            $thread,
                            unsafe { __stack.add(__vals) },
                            unsafe { __stack.add(__fs) } as *const Instruction,
                            $handlers,
                            $ds,
                            __top,
                            $closure,
                        );
                    }
                    NativeStep::Exit => return Exit::End,
                }
            }};
        }

        /// Call `$$f` with `$$args` for the running instruction, which `$$ret`
        /// finishes once the call returns. The call is staged above the
        /// frame's registers, where `meta_call` makes it.
        #[allow(unused_macros)]
        macro_rules! call_mm {
            ($$ret:ident, $$f:expr, [$$($$arg:expr),* $$(,)?]) => {{
                let __scratch =
                    unsafe { (*$frame).base() } + $closure.max_stack_size as usize;
                call_mm!(@at __scratch, $$ret, $$f, [$$($$arg),*]);
            }};
            // Staged at `$$scratch`, at or above the frame's registers.
            (@at $$scratch:expr, $$ret:ident, $$f:expr, [$$($$arg:expr),* $$(,)?]) => {{
                let __f: Value<'gc> = $$f;
                let __args: [Value<'gc>; _] = [$$($$arg),*];
                let __scratch: usize = $$scratch;
                let __end = __scratch + 1 + __args.len();
                $thread.ensure_slots(__end);
                let __stack = $thread.stack.as_mut_ptr();
                unsafe {
                    __stack.add(__scratch).write(__f);
                    for (__i, __a) in __args.into_iter().enumerate() {
                        __stack.add(__scratch + 1 + __i).write(__a);
                    }
                    (*$frame).pc = $ip;
                }
                $thread.set_top_unchecked(__end);
                $ds.ret = $$ret;
                become meta_call(
                    $instruction,
                    $ctx,
                    $thread,
                    $registers,
                    unsafe { __stack.add(__scratch) } as *const Instruction,
                    $handlers,
                    $ds,
                    $frame,
                    $closure,
                );
            }};
        }
    };
}

/// Inflates the slow-path body for `R[dst] = recv[k]` on any receiver.
/// Expects `helpers!(...)` to have been invoked in the enclosing handler
/// so `dispatch!`, `raise!`, `call_mm!`, and `reg!` resolve.
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
        index_chain_body!(
            $ctx, $thread, $registers, $ip, $handlers, $ds, __recv, __k, __dst_reg
        );
    }};
}

/// The `__index` half of [`get_slow_body!`], for a receiver whose own raw
/// lookup (if it is a table) already missed.
macro_rules! index_chain_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $recv:expr, $k:expr, $dst:expr) => {{
        let __recv: Value<'gc> = $recv;
        let __k: Value<'gc> = $k;
        let __dst_reg: u8 = $dst;

        match walk_index_chain($ctx, __recv, __k) {
            IndexChain::Resolved(__rv) => {
                *reg!(ref mut __dst_reg) = __rv;
                dispatch!();
            }
            IndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => call_mm!(ret_store_a, Value::function(__mm_func), [__mm_recv, __k]),
            IndexChain::NotIndexable(__v) => raise!(OpError::Index(__v)),
            IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
        }
    }};
}

/// The position slot's mark for an `ipairs` loop (see `op_tforprep`).
const TFOR_IPAIRS: i32 = -1;

/// Finish a TFORCALL step taken without calling the iterator: store `(k, v)`
/// in the loop's variables and take the following TFORLOOP's jump, or once
/// the walk is done, nil them and step past it. Expects `helpers!(...)` to
/// have been invoked in the enclosing handler.
macro_rules! tfor_finish {
    ($step:expr, $vars:expr, $count:expr, $ip:ident) => {{
        let __vars: u8 = $vars;
        let __count: u8 = $count;
        match $step {
            Some((__k, __v)) => {
                *reg!(ref mut __vars) = __k;
                if __count > 1 {
                    *reg!(ref mut __vars + 1) = __v;
                }
                for __i in 2..__count {
                    *reg!(ref mut __vars + __i) = Value::nil();
                }
                debug_assert_eq!(unsafe { (*$ip).op() }, Op::TFORLOOP);
                let (_, __offset) = unsafe { *$ip }.a_imm();
                $ip = unsafe { $ip.add(1).offset(__offset as isize) };
            }
            None => {
                for __i in 0..__count {
                    *reg!(ref mut __vars + __i) = Value::nil();
                }
                $ip = unsafe { $ip.add(1) };
            }
        }
        dispatch!();
    }};
}

/// Inflates the slow-path body for `recv[k] = v` on any receiver.
macro_rules! set_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr, $ds:ident,
     $recv:expr, $k:expr, $v:expr, $raw_set:ident) => {{
        let __k: Value<'gc> = $k;
        let __new_val: Value<'gc> = $v;

        match walk_newindex_chain($ctx, $recv, __k) {
            NewIndexChain::RawSet(__target) => {
                check_index_key!(__k);
                __target.$raw_set($ctx, __k, __new_val);
                dispatch!();
            }
            NewIndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => call_mm!(
                ret_discard,
                Value::function(__mm_func),
                [__mm_recv, __k, __new_val]
            ),
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
/// `SETTABUP`/`SELF` instruction whose prototype was assembled with a matching
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
fn fill_ic<'gc>(ctx: Context<'gc>, closure: LuaFn<'gc>, ic_idx: u16, entry: InlineCache<'gc>) {
    // Every dict table with the same metatable shares one sentinel shape,
    // so an entry on it would answer for keys it never saw.
    debug_assert!(match entry {
        InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => !shape.is_dict(),
        InlineCache::Transition { from, to, .. } => !from.is_dict() && !to.is_dict(),
        InlineCache::ProtoLoad {
            recv, holder_shape, ..
        } => !recv.is_dict() && !holder_shape.is_dict(),
        InlineCache::Empty => true,
    });
    let proto_gc = closure.proto;
    if let Some(slot_lock) = proto_gc.ic_table.get(ic_idx as usize) {
        // We're adopting a fresh `Shape` Gc pointer through this slot
        // (transitively reachable from the parent `Prototype`), so emit
        // the backward barrier on the Prototype manually before writing
        // through `as_cell()` — `Lock::as_cell` is `unsafe` precisely
        // because it skips the automatic barrier `Lock::set` on
        // `Gc<Lock<T>>` would emit.
        ctx.mutation().backward_barrier(Gc::erase(proto_gc), None);
        unsafe { slot_lock.as_cell() }.set(entry);
    }
}

/// The entry for a lookup of a key in `shape` that found `slot`.
#[inline(always)]
fn shape_entry<'gc>(shape: Shape<'gc>, slot: Option<u32>) -> InlineCache<'gc> {
    match slot {
        Some(slot) => InlineCache::Own {
            shape,
            loc: SlotLoc::new(shape, slot),
        },
        None => InlineCache::Absent { shape },
    }
}

/// The cached result of a constant-key load from `t`, whose state is
/// `state`, or `None` when the slow path must run.
#[inline(always)]
fn ic_get<'gc>(
    cache: InlineCache<'gc>,
    t: Table<'gc>,
    state: &TableState<'gc>,
) -> Option<Value<'gc>> {
    let live = state.shape();
    // Without the hint LLVM tests the rarer variants' tags first.
    let v = if std::hint::likely(matches!(cache, InlineCache::Own { .. }))
        && let InlineCache::Own { shape, loc } = cache
        && Shape::ptr_eq(live, shape)
    {
        unsafe { t.load(state, loc) }
    } else if let InlineCache::Absent { shape } = cache
        && Shape::ptr_eq(live, shape)
    {
        Value::nil()
    } else if let InlineCache::ProtoLoad {
        recv,
        holder,
        holder_shape,
        loc,
    } = cache
        && Shape::ptr_eq(live, recv)
    {
        // `recv` was filled with a metatable whose `__index` was `holder`.
        let mt = unsafe { live.mt_cache().unwrap_unchecked() };
        if mt.index_table() != holder.as_ptr() as usize {
            return None;
        }
        // SAFETY: `__index` is still `holder`, so the receiver keeps it alive.
        // The address can't name another table: `holder` reserves it while
        // this entry lives, even once dropped.
        let holder = Table::from_inner(unsafe { Gc::from_ptr(holder.as_ptr()) });
        let h = holder.inner().borrow();
        if !Shape::ptr_eq(h.shape(), holder_shape) {
            return None;
        }
        // A nil slot means the walk goes on past `holder`.
        let v = unsafe { holder.load(&h, loc) };
        return (!v.is_nil()).then_some(v);
    } else {
        return None;
    };
    (!(v.is_nil() && live.has_mm(MetamethodBits::INDEX))).then_some(v)
}

/// Store `v` through the entry for a constant-key store to `t`. Returns
/// false, having stored nothing, when the slow path must run.
#[inline(always)]
fn ic_set<'gc>(ctx: Context<'gc>, cache: InlineCache<'gc>, t: Table<'gc>, v: Value<'gc>) -> bool {
    let state = t.inner().borrow();
    let live = state.shape();
    // See `ic_get` for the hint.
    if std::hint::likely(matches!(cache, InlineCache::Own { .. }))
        && let InlineCache::Own { shape, loc } = cache
        && Shape::ptr_eq(live, shape)
    {
        // __newindex fires only on currently-nil keys.
        let existing = unsafe { t.load(&state, loc) };
        if existing.is_nil() && live.has_mm(MetamethodBits::NEWINDEX) {
            return false;
        }
        // Barrier work goes to the slow path, keeping calls (and with them a
        // stack frame) out of this handler.
        drop(state);
        let Some(w) = Gc::write_if_clean(ctx.mutation(), t.inner()) else {
            return false;
        };
        let state = w.unlock().borrow();
        // In range: the live shape is the one `loc` was cached against.
        unsafe { t.store(&state, loc, v) };
        return true;
    }
    if let InlineCache::Transition { from, to, loc } = cache
        && Shape::ptr_eq(live, from)
    {
        if live.has_mm(MetamethodBits::NEWINDEX) {
            return false;
        }
        // Storing nil to an absent key adds nothing.
        if v.is_nil() {
            return true;
        }
        // As above, and so does growing a full spill cell.
        if !state.has_room(loc) {
            return false;
        }
        drop(state);
        let Some(w) = Gc::write_if_clean(ctx.mutation(), t.inner()) else {
            return false;
        };
        let mut state = w.unlock().borrow_mut();
        // SAFETY: the live shape is `from`, and there is room.
        unsafe { t.push(&mut state, to, loc, v) };
        return true;
    }
    false
}

/// GETFIELD/SETFIELD/GETTABUP/SETTABUP/SELF only carry constant string keys.
#[inline(always)]
fn constant_key<'gc>(k: Value<'gc>) -> LuaString<'gc> {
    debug_assert!(k.get_string().is_some(), "IC site with a non-string key");
    unsafe { k.get_string().unwrap_unchecked() }
}

/// `t[k]` for a constant-key IC miss, resolved as far as an entry can cache
/// it: an own slot, or a slot in the table `t`'s metatable names as
/// `__index`. Returns the value, or the receiver to continue the `__index`
/// walk from, its own raw lookup having missed.
#[inline(always)]
fn get_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    t: Table<'gc>,
    k: Value<'gc>,
) -> Result<Value<'gc>, Value<'gc>> {
    let state = t.inner().borrow();
    let shape = state.shape();
    if shape.is_dict() {
        let v = state.raw_get(k);
        return if v.is_nil() && shape.has_mm(MetamethodBits::INDEX) {
            Err(Value::table(t))
        } else {
            Ok(v)
        };
    }
    // The entry already says `t` lacks the key, and an `__index` function
    // leaves nothing to cache: refilling would cost a barrier on every miss.
    if let InlineCache::Absent { shape: cached } = read_ic(closure, ic_idx)
        && Shape::ptr_eq(cached, shape)
        && let Some(mt) = shape.mt_cache()
        && mt.mm(MetamethodBits::INDEX).get_function().is_some()
    {
        return Err(Value::table(t));
    }
    let slot = shape.find_slot(constant_key(k));
    fill_ic(ctx, closure, ic_idx, shape_entry(shape, slot));
    let v = slot.map_or(Value::nil(), |s| state.named_get(s));
    if !v.is_nil() || !shape.has_mm(MetamethodBits::INDEX) {
        return Ok(v);
    }
    // INDEX bit implies a metatable.
    let index = unsafe { shape.mt_cache().unwrap_unchecked() }.mm(MetamethodBits::INDEX);
    drop(state);
    let recv = slot.is_none().then_some(shape);
    get_index_fill_ic(ctx, closure, ic_idx, t, index, recv, k)
}

/// The `__index` half of [`get_fill_ic`], out of line to keep the own-key
/// miss lean: `t`'s raw lookup missed and its metatable's `__index` is `index`.
/// `recv` is `t`'s shape when that shape lacks the key, which a hit in an
/// `__index` table can then be cached against.
#[inline(never)]
fn get_index_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    t: Table<'gc>,
    index: Value<'gc>,
    recv: Option<Shape<'gc>>,
    k: Value<'gc>,
) -> Result<Value<'gc>, Value<'gc>> {
    let Some(holder) = index.get_table() else {
        // A function is called with `t`; anything else is indexed in turn.
        return Err(if index.get_function().is_some() {
            Value::table(t)
        } else {
            index
        });
    };
    let h = holder.inner().borrow();
    let holder_shape = h.shape();
    if holder_shape.is_dict() {
        let v = h.raw_get(k);
        return if v.is_nil() { Err(index) } else { Ok(v) };
    }
    let Some(holder_slot) = holder_shape.find_slot(constant_key(k)) else {
        return Err(index);
    };
    let v = h.named_get(holder_slot);
    if v.is_nil() {
        return Err(index);
    }
    if let Some(recv) = recv {
        debug_assert_eq!(
            unsafe { recv.mt_cache().unwrap_unchecked() }.index_table(),
            Gc::as_ptr(holder.inner()) as usize
        );
        let entry = InlineCache::ProtoLoad {
            recv,
            holder: Gc::downgrade(holder.inner()),
            holder_shape,
            loc: SlotLoc::new(holder_shape, holder_slot),
        };
        fill_ic(ctx, closure, ic_idx, entry);
    }
    Ok(v)
}

/// `t[k] = v` for a constant-key IC miss, with one slot lookup that both
/// stores and refills the entry. Returns false, having stored nothing, when
/// `__newindex` may fire.
#[inline(always)]
fn set_own_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    t: Table<'gc>,
    k: Value<'gc>,
    v: Value<'gc>,
) -> bool {
    let state = t.inner().borrow();
    let shape = state.shape();
    let newindex = shape.has_mm(MetamethodBits::NEWINDEX);
    if shape.is_dict() {
        if newindex {
            return false;
        }
        drop(state);
        t.raw_set(ctx, k, v);
        return true;
    }
    // As in `get_fill_ic`: the entry already says only `__newindex` is left.
    if newindex
        && let InlineCache::Absent { shape: cached } = read_ic(closure, ic_idx)
        && Shape::ptr_eq(cached, shape)
    {
        return false;
    }
    let key = constant_key(k);
    // A hit stores without telling a metatable's cache, so the keys it
    // mirrors are left uncached.
    let cache = !mirrored(key);
    let slot = shape.find_slot(key);
    let existing = slot.map_or(Value::nil(), |s| state.named_get(s));
    if existing.is_nil() && newindex {
        if cache {
            fill_ic(ctx, closure, ic_idx, shape_entry(shape, slot));
        }
        return false;
    }
    drop(state);
    let mut state = t.inner().borrow_mut(ctx.mutation());
    let entry = match slot {
        Some(slot) => {
            state.named_set(slot, v);
            state.maybe_update_mt_bit(k, v);
            shape_entry(shape, Some(slot))
        }
        None => {
            state.add_string_key(ctx, key, v, MAX_PROPERTIES_FAST);
            let to = state.shape();
            // A nil store adds nothing; past the cap the table went dict.
            if Shape::ptr_eq(to, shape) || to.is_dict() {
                InlineCache::Absent { shape }
            } else {
                let loc = SlotLoc::new(to, shape.slot_count());
                InlineCache::Transition {
                    from: shape,
                    to,
                    loc,
                }
            }
        }
    };
    drop(state);
    if cache {
        fill_ic(ctx, closure, ic_idx, entry);
    }
    true
}

/// Drive the VM on `thread` until the top-level frame returns.
///
/// The caller must have seeded the thread with at least one `LuaFrame`,
/// sized `stack` to at least `base + max_stack_size`, and placed
/// the callee + arguments at `stack[base-1..]`. See `Executor::start`.
#[inline(never)]
pub(crate) fn run_thread<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) -> Exit {
    let mut ts = thread.borrow_mut(ctx.mutation());
    let mut ds = DispatchState {
        fault: None,
        native: std::ptr::null(),
        ret: ret_exit,
    };
    ts.top_lua()
        .expect("run_thread requires a seeded Lua frame");
    let (frame, closure) = top_frame(&mut ts);
    let handlers = HANDLERS.as_ptr() as *const ();
    if let Some(err) = ts.native_error.take() {
        return native_entry(ctx, &mut ts, handlers, &mut ds, closure, err);
    }
    if let Some(p) = ts.pending_ret.take() {
        let nret = ts.top - p.values;
        let stack = ts.stack.as_mut_ptr();
        let (values, func_slot) = unsafe { (stack.add(p.values), stack.add(p.func_slot)) };
        return (p.ret)(
            Instruction::from_raw(nret as u64),
            ctx,
            &mut ts,
            values,
            func_slot as *const Instruction,
            handlers,
            &mut ds,
            frame,
            closure,
        );
    }
    let (ip, base) = unsafe { ((*frame).pc, (*frame).base()) };
    let registers = unsafe { ts.stack.as_mut_ptr().add(base) };
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
    )
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
) -> Exit {
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
    Exit::End
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, idx, ic_idx, _key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        tail!(gettabup_slow);
    };

    let t_state = t.inner().borrow();
    if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &t_state) {
        drop(t_state);
        *reg!(ref mut dst) = v;
        dispatch!();
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let k = constant!(key);
    let Some(t) = t_val.get_table() else {
        get_slow_body!(ctx, thread, registers, ip, handlers, ds, t_val, k, dst);
    };
    match get_fill_ic(ctx, closure, ic_idx, t, k) {
        Ok(v) => {
            *reg!(ref mut dst) = v;
            dispatch!();
        }
        Err(from) => index_chain_body!(ctx, thread, registers, ip, handlers, ds, from, k, dst),
    }
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, idx, ic_idx, _key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        tail!(settabup_slow);
    };

    if ic_set(ctx, read_ic(closure, ic_idx), t, reg!(src)) {
        dispatch!();
    }
    tail!(settabup_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn settabup_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let k = constant!(key);
    let v = reg!(src);
    if let Some(t) = t_val.get_table()
        && set_own_fill_ic(ctx, closure, ic_idx, t, k, v)
    {
        dispatch!();
    }
    set_slow_body!(
        ctx, thread, registers, ip, handlers, ds, t_val, k, v, raw_set
    );
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        tail!(gettable_slow);
    };

    // An integer key inside the array part, handled without calls so this
    // handler needs no stack frame; any other key goes to `gettable_general`.
    if let Some(i) = reg!(key).get_small() {
        let t_state = t.inner().borrow();
        if let Some(v) = t_state.array_get(i as usize)
            && !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX))
        {
            *reg!(ref mut dst) = v;
            dispatch!();
        }
    }
    tail!(gettable_general);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn gettable_general<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, key) = instruction.abc();
    let Some(t) = reg!(table).get_table() else {
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        tail!(settable_slow);
    };

    // As in `op_gettable`; barrier work also goes to `settable_general`.
    if let Some(i) = reg!(key).get_small() {
        let t_state = t.inner().borrow();
        if let Some(old) = t_state.array_get(i as usize)
            && !(old.is_nil() && t_state.shape().has_mm(MetamethodBits::NEWINDEX))
        {
            drop(t_state);
            if let Some(w) = Gc::write_if_clean(ctx.mutation(), t.inner()) {
                let v = reg!(src);
                // In range: checked above, and nothing ran in between.
                unsafe { w.unlock().borrow_mut().set_array_at(i as usize, v) };
                dispatch!();
            }
        }
    }
    tail!(settable_general);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn settable_general<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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
    t_state.raw_set_keyed(ctx, k, v);
    dispatch!()
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn settable_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, key) = instruction.abc();
    let (recv, k, v) = (reg!(table), reg!(key), reg!(src));
    set_slow_body!(
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds,
        recv,
        k,
        v,
        raw_set_keyed
    );
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, ic_idx, _key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        tail!(getfield_slow);
    };

    let t_state = t.inner().borrow();
    if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &t_state) {
        drop(t_state);
        *reg!(ref mut dst) = v;
        dispatch!();
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    let Some(t) = recv.get_table() else {
        get_slow_body!(ctx, thread, registers, ip, handlers, ds, recv, k, dst);
    };
    match get_fill_ic(ctx, closure, ic_idx, t, k) {
        Ok(v) => {
            *reg!(ref mut dst) = v;
            dispatch!();
        }
        Err(from) => index_chain_body!(ctx, thread, registers, ip, handlers, ds, from, k, dst),
    }
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, ic_idx, _key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        tail!(setfield_slow);
    };

    if ic_set(ctx, read_ic(closure, ic_idx), t, reg!(src)) {
        dispatch!();
    }
    tail!(setfield_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn setfield_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (src, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    let v = reg!(src);
    if let Some(t) = recv.get_table()
        && set_own_fill_ic(ctx, closure, ic_idx, t, k, v)
    {
        dispatch!();
    }
    set_slow_body!(
        ctx, thread, registers, ip, handlers, ds, recv, k, v, raw_set
    );
}

// ---------------------------------------------------------------------------
// SELF — method-call setup
// ---------------------------------------------------------------------------

/// Backs `obj:m(...)`. Writes the method into `R[dst]` and the
/// receiver into `R[dst+1]`.
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, object, ic_idx, _key_idx) = instruction.abde();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        tail!(op_self_slow);
    };

    let recv_state = recv.inner().borrow();
    if let Some(method) = ic_get(read_ic(closure, ic_idx), recv, &recv_state) {
        drop(recv_state);
        *reg!(ref mut dst) = method;
        *reg!(ref mut (dst + 1)) = recv_val;
        dispatch!();
    }
    drop(recv_state);
    tail!(op_self_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_self_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, object, ic_idx, key_idx) = instruction.abde();

    let recv_val = reg!(object);
    let key = constant!(key_idx);

    let from = match recv_val.get_table() {
        Some(recv) => match get_fill_ic(ctx, closure, ic_idx, recv, key) {
            Ok(method) => {
                *reg!(ref mut dst) = method;
                *reg!(ref mut (dst + 1)) = recv_val;
                dispatch!();
            }
            Err(from) => from,
        },
        None => recv_val,
    };
    match walk_index_chain(ctx, from, key) {
        IndexChain::Resolved(method) => {
            *reg!(ref mut dst) = method;
            *reg!(ref mut (dst + 1)) = recv_val;
            dispatch!();
        }
        IndexChain::Invoke { func, receiver } => {
            // Functional __index. Pre-place self at dst+1; the
            // continuation writes the resolved method into dst.
            *reg!(ref mut (dst + 1)) = recv_val;
            call_mm!(ret_store_a, Value::function(func), [receiver, key]);
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, template) = instruction.ad();
    // SAFETY: the compiler gives each NEWTABLE a template.
    let t = unsafe { closure.proto.templates.get_unchecked(template as usize) };
    *reg!(ref mut dst) = Value::table(Table::from_template(ctx.mutation(), t));
    gc_check!();
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
        ) -> Exit {
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
        ) -> Exit {
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
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
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

        let meta_fn = binop_metamethod($ctx, lhs, rhs, MetamethodBits::$mm);
        if meta_fn.is_nil() {
            raise!(OpError::$err(lhs, rhs));
        }

        call_mm!(ret_store_a, meta_fn, [lhs, rhs]);
    }};
}

arith_handler!(op_add, op_add_slow, ADD, num::Add, ADD);
arith_handler!(op_sub, op_sub_slow, SUB, num::Sub, SUB);
arith_handler!(op_mul, op_mul_slow, MUL, num::Mul, MUL);
arith_handler!(op_mod, op_mod_slow, MOD, num::Mod, MOD);
arith_handler!(op_pow, op_pow_slow, POW, num::Pow, POW);
arith_handler!(op_div, op_div_slow, DIV, num::Div, DIV);
arith_handler!(op_idiv, op_idiv_slow, IDIV, num::IDiv, IDIV);
bit_handler!(op_band, op_band_slow, BAND, num::BAnd, BAND);
bit_handler!(op_bor, op_bor_slow, BOR, num::BOr, BOR);
bit_handler!(op_bxor, op_bxor_slow, BXOR, num::BXor, BXOR);
bit_handler!(op_shl, op_shl_slow, SHL, num::Shl, SHL);
bit_handler!(op_shr, op_shr_slow, SHR, num::Shr, SHR);

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
        ) -> Exit {
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
        ) -> Exit {
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
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
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

arith_imm_handler!(op_addi, op_addi_slow, ADDI, num::Add, ADD, false);
arith_imm_handler!(op_subi, op_subi_slow, SUBI, num::Sub, SUB, false);
arith_imm_handler!(op_muli, op_muli_slow, MULI, num::Mul, MUL, false);
arith_imm_handler!(op_modi, op_modi_slow, MODI, num::Mod, MOD, false);
arith_imm_handler!(op_powi, op_powi_slow, POWI, num::Pow, POW, false);
arith_imm_handler!(op_divi, op_divi_slow, DIVI, num::Div, DIV, false);
arith_imm_handler!(op_idivi, op_idivi_slow, IDIVI, num::IDiv, IDIV, false);
arith_imm_handler!(op_rsubi, op_rsubi_slow, RSUBI, num::Sub, SUB, true);
arith_imm_handler!(op_rmodi, op_rmodi_slow, RMODI, num::Mod, MOD, true);
arith_imm_handler!(op_rpowi, op_rpowi_slow, RPOWI, num::Pow, POW, true);
arith_imm_handler!(op_rdivi, op_rdivi_slow, RDIVI, num::Div, DIV, true);
arith_imm_handler!(op_ridivi, op_ridivi_slow, RIDIVI, num::IDiv, IDIV, true);
bit_imm_handler!(op_bandi, op_bandi_slow, BANDI, num::BAnd, BAND, false);
bit_imm_handler!(op_bori, op_bori_slow, BORI, num::BOr, BOR, false);
bit_imm_handler!(op_bxori, op_bxori_slow, BXORI, num::BXor, BXOR, false);
bit_imm_handler!(op_shli, op_shli_slow, SHLI, num::Shl, SHL, false);
bit_imm_handler!(op_shri, op_shri_slow, SHRI, num::Shr, SHR, false);
bit_imm_handler!(op_rshli, op_rshli_slow, RSHLI, num::Shl, SHL, true);
bit_imm_handler!(op_rshri, op_rshri_slow, RSHRI, num::Shr, SHR, true);

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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(ref src);
    if val.is_float() {
        let f = val.read_float();
        reg!(ref mut dst).write_float(-f);
        dispatch!();
    }
    if let Some(i) = val.get_small()
        && let Some(n) = i.checked_neg()
    {
        *reg!(ref mut dst) = Value::small(n);
        dispatch!();
    }
    tail!(op_unm_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_unm_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), i.wrapping_neg());
        dispatch!();
    }
    let meta_fn = ctx.mm_of(val, MetamethodBits::UNM);
    if meta_fn.is_nil() {
        raise!(OpError::Arith(val, val));
    }
    // Lua passes the operand twice for unary metamethods (spec quirk).
    call_mm!(ret_store_a, meta_fn, [val, val]);
}

/// R[dst] = ~R[src]  (bitwise NOT)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_bnot<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    if let Some(i) = reg!(ref src).get_small() {
        *reg!(ref mut dst) = Value::small(!i);
        dispatch!();
    }
    tail!(op_bnot_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_bnot_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), !i);
        dispatch!();
    }
    let meta_fn = ctx.mm_of(val, MetamethodBits::BNOT);
    if meta_fn.is_nil() {
        raise!(OpError::Bitwise(val, val));
    }
    call_mm!(ret_store_a, meta_fn, [val, val]);
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
) -> Exit {
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
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(t) = val.get_table()
        && !t.shape().has_mm(MetamethodBits::LEN)
    {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), t.raw_len() as i64);
        dispatch!();
    }
    tail!(op_len_slow);
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_len_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, src) = instruction.ab();
    let val = reg!(src);

    // Strings never consult __len; return byte length directly.
    if let Some(s) = val.get_string() {
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), s.len() as i64);
        dispatch!();
    }

    // Without `__len`, only a table has a length to fall back on.
    let meta_fn = ctx.mm_of(val, MetamethodBits::LEN);
    if meta_fn.is_nil() {
        let Some(t) = val.get_table() else {
            raise!(OpError::Len(val));
        };
        *reg!(ref mut dst) = Value::integer(ctx.mutation(), t.raw_len() as i64);
        dispatch!();
    }
    // Like the other unary metamethods, `__len` gets its operand twice.
    call_mm!(ret_store_a, meta_fn, [val, val]);
}

/// R[dst] = R[lhs] .. R[rhs]  (string concatenation)
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_concat<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, lhs, rhs) = instruction.abc();
    let a = reg!(lhs);
    let b = reg!(rhs);
    // Fast path: both coerce to strings/numbers.
    let mut buf = Vec::new();
    if num::coerce_to_str(&mut buf, a) && num::coerce_to_str(&mut buf, b) {
        *reg!(ref mut dst) = Value::string(LuaString::new(ctx, &buf));
        gc_check!();
        dispatch!();
    }
    let meta_fn = binop_metamethod(ctx, a, b, MetamethodBits::CONCAT);
    if meta_fn.is_nil() {
        raise!(OpError::Concat(a, b));
    }
    call_mm!(ret_store_a, meta_fn, [a, b]);
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let start = instruction.a();
    let base = unsafe { (*frame).base() };
    let start_idx = base + start as usize;
    close_upvalues(ctx.mutation(), thread, start_idx);
    if has_tbc_from(thread, start_idx) {
        tail!(close_tbc);
    }
    dispatch!();
}

/// CLOSE of a to-be-closed variable: call its `__close`, taking it off the
/// list first so a failing one leaves the rest to the unwinder, then run the
/// CLOSE again for the next.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn close_tbc<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (v, tm) = match pop_tbc(ctx, thread) {
        Ok(c) => c,
        Err(err) => {
            save_pc(thread, ip);
            thread.raise(ctx, err);
            return Exit::End;
        }
    };
    call_mm!(ret_close, tm, [v]);
}

/// Take the innermost to-be-closed variable off the list: its value and
/// `__close`, or the error for a `__close` that can't be called, raised as
/// the closing frame's rather than as the call's.
fn pop_tbc<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
) -> Result<(Value<'gc>, Value<'gc>), crate::env::Error<'gc>> {
    let entry = thread
        .tbc_list
        .pop()
        .expect("no to-be-closed variable to close");
    let v = entry.value(&thread.stack);
    let tm = ctx.mm_of(v, MetamethodBits::CLOSE);
    if let Some(e) = call_chain_error(ctx, tm) {
        let suffix = match e {
            OpError::Call(_) => " (metamethod 'close')",
            _ => "",
        };
        let msg = crate::vm::debug::op_error_message(ctx, thread, e) + suffix;
        return Err(crate::env::Error::from_str(ctx, &msg));
    }
    Ok((v, tm))
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let val = instruction.a();
    let v = reg!(val);
    // `false` and `nil` need no closing.
    if v.is_falsy() {
        dispatch!();
    }
    if ctx.mm_of(v, MetamethodBits::CLOSE).is_nil() {
        raise!(OpError::NonClosable(val));
    }
    let base = unsafe { (*frame).base() };
    thread.tbc_list.push(TbcEntry::Slot(base + val as usize));
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
) -> Exit {
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
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (lhs, rhs, inverted) = instruction.abc_flag();
    let (a, b) = (reg!(ref lhs), reg!(ref rhs));
    // Floats first: a NaN has the same bits as itself.
    let eq = if a.is_float() && b.is_float() {
        a.read_float() == b.read_float()
    } else if a.same_bits(b) {
        true
    } else {
        tail!(op_eq_slow);
    };
    skip_if!(eq != inverted);
    dispatch!();
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_eq_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (lhs, rhs, inverted) = instruction.abc_flag();
    let (a, b) = (reg!(lhs), reg!(rhs));
    if num::raw_eq(a, b) {
        skip_if!(!inverted);
        dispatch!();
    }
    // Lua 5.5: __eq fires only when both operands are the same non-primitive
    // type (tables or userdata) and raw equality fails.
    let try_meta = (a.kind() == ValueKind::Table && b.kind() == ValueKind::Table)
        || (a.kind() == ValueKind::Userdata && b.kind() == ValueKind::Userdata);
    if try_meta {
        let meta_fn = binop_metamethod(ctx, a, b, MetamethodBits::EQ);
        if !meta_fn.is_nil() {
            call_mm!(ret_cond_c, meta_fn, [a, b]);
        }
    }
    skip_if!(inverted);
    dispatch!();
}

/// `if (R[lhs] <op> R[rhs]) != inverted then skip next instruction` for LT and
/// LE: two floats or two inline integers here, everything else in `$slow`.
macro_rules! cmp_handler {
    ($name:ident, $slow:ident, $op:tt, $mm:ident, $int_float:path, $float_int:path) => {
#[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (lhs, rhs, inverted) = instruction.abc_flag();
            let (a, b) = (reg!(ref lhs), reg!(ref rhs));
            let r = if std::hint::likely(a.is_float() && b.is_float()) {
                a.read_float() $op b.read_float()
            } else if let Some((x, y)) = Value::both_small(a, b) {
                x $op y
            } else {
                tail!($slow);
            };
            skip_if!(r != inverted);
            dispatch!();
        }

        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $slow<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (lhs, rhs, inverted) = instruction.abc_flag();
            let (a, b) = (reg!(lhs), reg!(rhs));
            let primitive = if let Some(x) = a.get_integer()
                && let Some(y) = b.get_integer()
            {
                Some(x $op y)
            } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
                Some($int_float(x, y))
            } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
                Some($float_int(x, y))
            } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
                Some(x $op y)
            } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
                Some(x $op y)
            } else {
                None
            };
            if let Some(r) = primitive {
                skip_if!(r != inverted);
                dispatch!();
            }
            let meta_fn = binop_metamethod(ctx, a, b, MetamethodBits::$mm);
            if meta_fn.is_nil() {
                raise!(OpError::Compare(a, b));
            }
            call_mm!(ret_cond_c, meta_fn, [a, b]);
        }
    };
}

cmp_handler!(op_lt, op_lt_slow, <, LT, num::lt_int_float, num::lt_float_int);
cmp_handler!(op_le, op_le_slow, <=, LE, num::le_int_float, num::le_float_int);

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
        ) -> Exit {
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
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let (src, _) = instruction.ab_imm_flag();
            let (v, k) = (reg!(src), instruction.imm_value(ctx.mutation()));
            let (a, b) = if $swap { (k, v) } else { (v, k) };
            let meta_fn = binop_metamethod(ctx, a, b, MetamethodBits::$mm);
            if meta_fn.is_nil() {
                raise!(OpError::Compare(a, b));
            }
            call_mm!(ret_cond_b, meta_fn, [a, b]);
        }
    };
}

cmp_imm_handler!(
    op_lti,
    op_lti_slow,
    LT,
    false,
    |a, b| a < b,
    |a: f64, b: f64| a < b,
    num::lt_int_float,
    num::lt_float_int
);
cmp_imm_handler!(
    op_lei,
    op_lei_slow,
    LE,
    false,
    |a, b| a <= b,
    |a: f64, b: f64| a <= b,
    num::le_int_float,
    num::le_float_int
);
cmp_imm_handler!(
    op_gti,
    op_gti_slow,
    LT,
    true,
    |a, b| a < b,
    |a: f64, b: f64| a < b,
    num::lt_int_float,
    num::lt_float_int
);
cmp_imm_handler!(
    op_gei,
    op_gei_slow,
    LE,
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
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
                    return Exit::End;
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
                    gc_check!();
                    dispatch!();
                }
                action => {
                    unsafe { (*$frame).pc = $ip };
                    let f = unsafe { $thread.stack[func_idx].get_function().unwrap_unchecked() };
                    // The common shape, a native calling a Lua function, without
                    // `run_natives`' generality.
                    if let crate::vm::sequence::CallbackAction::CallThen { at, protect, cont } =
                        action
                        && let Some(callee) = native_calls_lua($thread, args_base + at as usize)
                    {
                        push_native_frame($thread, f, args_base, at, protect, cont, ret_call);
                        let new_base = args_base + at as usize + 1;
                        enter_from_native($thread, callee, new_base);
                        ($frame, $closure) = (unsafe { $thread.top_lua_ptr() }, callee);
                        $ip = callee.code;
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
                        dispatch!();
                    }
                    let step = run_natives(
                        $ctx,
                        $thread,
                        NativeState::Acted {
                            r: Ok(action),
                            framed: false,
                            f,
                            base: args_base,
                            ret: ret_call,
                        },
                    );
                    native_step!(step);
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
                caller_provided.saturating_sub(num_params) as u16
            } else {
                0
            }
        };
        $ip = callee.code;
        let _ = $returns;
        let frame = LuaFrame {
            closure: callee,
            pc: $ip,
            ret: ret_call,
            base: new_base as u32,
            num_extras,
            flags: 0,
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
) -> Exit {
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
) -> Exit {
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
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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
        ) -> Exit {
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

/// The entry of `pairs` (see `NativeClosure::entry`): for a table without
/// `__pairs`, `(next, t, nil, nil)` straight into the call's result slots.
/// Other arguments, a TAILCALL and a call keeping all results go to the
/// builtin.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_pairs<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs, returns) = (instruction.a(), instruction.b(), instruction.c());
    if instruction.op() == Op::TAILCALL || nargs != 2 || returns == 0 {
        tail!(op_call_native);
    }
    let Some(t) = reg!(func + 1).get_table() else {
        tail!(op_call_native);
    };
    if t.shape().has_mm(MetamethodBits::PAIRS) {
        tail!(op_call_native);
    }
    *reg!(ref mut func) = Value::function(ctx.next_fn());
    // `t` is already in R[func+1].
    unsafe {
        fill_nil(
            registers.add(func as usize + 2),
            (returns as usize - 1).saturating_sub(2),
        )
    };
    dispatch!();
}

/// The entry of `ipairs`: `(iterator, v, 0)` straight into the call's
/// result slots, for one argument, like [`ff_pairs`].
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_ipairs<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (func, nargs, returns) = (instruction.a(), instruction.b(), instruction.c());
    if instruction.op() == Op::TAILCALL || nargs != 2 || returns == 0 {
        tail!(op_call_native);
    }
    *reg!(ref mut func) = Value::function(ctx.ipairs_iter());
    // The argument is already in R[func+1].
    let wanted = returns as usize - 1;
    if wanted > 2 {
        *reg!(ref mut func + 2) = Value::small(0);
        unsafe { fill_nil(registers.add(func as usize + 3), wanted - 3) };
    }
    dispatch!();
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
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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
            nargs.saturating_sub(num_params) as u16
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
) -> Exit {
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
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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
            debug_assert!(!has_tbc_from(thread, base));
            thread.pop_lua();
            thread.push_exec(ExecKind::Error(err));
            return Exit::End;
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
        action => {
            // The native takes the popped frame's place: its function slot
            // and continuation carry the original caller's expectation across
            // the tail call, so its window moves down to that slot.
            let (orig_func, ret) = {
                let f = unsafe { &*frame };
                (f.base() - 1 - f.num_extras as usize, f.ret)
            };
            let f = reg!(func).get_function();
            let f = unsafe { f.unwrap_unchecked() };
            close_upvalues(ctx.mutation(), thread, base);
            debug_assert!(!has_tbc_from(thread, base));
            thread.pop_lua();
            let top = thread.top;
            thread.stack.copy_within(args_base..top, orig_func + 1);
            thread.set_top_unchecked(orig_func + 1 + (top - args_base));
            let step = run_natives(
                ctx,
                thread,
                NativeState::Acted {
                    r: Ok(action),
                    framed: false,
                    f,
                    base: orig_func + 1,
                    ret,
                },
            );
            native_step!(step);
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
) -> Exit {
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
            // No TAILCALL is emitted inside a `<close>` scope, so `TBC` only
            // marks scopes that already closed.
            debug_assert!(!has_tbc_from(thread, new_base));
            close_upvalues(ctx.mutation(), thread, new_base);
            unsafe { (*frame).flags &= !(frame_flags::OPEN_UPVALUES | frame_flags::TBC) };
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

/// Pop the running frame, whose results are the `$nret` values at `$values`,
/// and hand them to its continuation.
macro_rules! return_to_ret {
    ($nret:expr, $values:expr, $ctx:ident, $thread:ident, $registers:ident, $handlers:ident,
     $ds:ident, $frame:ident, $closure:ident) => {{
        let (__ret, __func_slot) = unsafe {
            let f = &*$frame;
            (f.ret, $registers.sub(1 + f.num_extras as usize))
        };
        // `LuaFrame` is `Copy`, so nothing needs dropping.
        let __n = $thread.frames.len();
        unsafe { $thread.frames.set_len(__n - 1) };
        become __ret(
            Instruction::from_raw($nret as u64),
            $ctx,
            $thread,
            $values,
            __func_slot as *const Instruction,
            $handlers,
            $ds,
            unsafe { $frame.sub(1) },
            $closure,
        );
    }};
}

/// return R[values], ..., R[values+count-2]
///
/// Fast path: nothing to close. It makes no calls, so it needs no stack
/// frame; `op_return_slow` closes, then returns the same way.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (values, count) = instruction.ab();
    if std::hint::unlikely(unsafe { (*frame).flags } != 0) {
        tail!(op_return_slow);
    }
    let values = unsafe { registers.add(values as usize) };
    // `count == 0` is MULTRET: read the count from `thread.top`.
    let nret = if count == 0 {
        thread.top - unsafe { values.offset_from_unsigned(thread.stack.as_ptr()) }
    } else {
        count as usize - 1
    };
    return_to_ret!(
        nret, values, ctx, thread, registers, handlers, ds, frame, closure
    );
}

/// return
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return0<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let _ = instruction;
    if std::hint::unlikely(unsafe { (*frame).flags } != 0) {
        tail!(
            op_return_slow,
            Instruction::ret(crate::instruction::Reg(0), 1)
        );
    }
    return_to_ret!(
        0, registers, ctx, thread, registers, handlers, ds, frame, closure
    );
}

/// return R[value]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return1<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let value = instruction.a();
    if std::hint::unlikely(unsafe { (*frame).flags } != 0) {
        tail!(
            op_return_slow,
            Instruction::ret(crate::instruction::Reg(value), 2)
        );
    }
    let values = unsafe { registers.add(value as usize) };
    return_to_ret!(
        1, values, ctx, thread, registers, handlers, ds, frame, closure
    );
}

/// RETURN from a frame with upvalues or to-be-closed variables to close.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
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

    if has_tbc_from(thread, cur_base) {
        let (v, tm) = match pop_tbc(ctx, thread) {
            Ok(c) => c,
            Err(err) => {
                save_pc(thread, ip);
                thread.raise(ctx, err);
                return Exit::End;
            }
        };
        // Past the frame and the results, below the call: where they end,
        // which a MULTRET RETURN finds through `top` again (`ret_return`).
        let values_end = values_base + nret;
        let mark = values_end.max(cur_base + closure.max_stack_size as usize);
        thread.ensure_slots(mark + 1);
        thread.stack[mark] = Value::small(values_end as i32);
        call_mm!(@at mark + 1, ret_return, tm, [v]);
    }
    // Before the results move: they may land on the frame's own registers.
    close_upvalues(ctx.mutation(), thread, cur_base);
    let values = unsafe { registers.add(values as usize) };
    return_to_ret!(
        nret, values, ctx, thread, registers, handlers, ds, frame, closure
    );
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
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

/// Generic for preparation: move the closing value from R[base+3] to
/// R[base+2], mark it to be closed, move the initial control from R[base+2]
/// to the first loop variable, set the position slot R[base+3] (see
/// `op_tforcall`), and jump to the loop test.
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, offset) = instruction.a_imm();
    let control = reg!(base + 2);
    let closing = reg!(base + 3);
    *reg!(ref mut base + TFOR_VARS) = control;
    *reg!(ref mut base + 2) = closing;
    // Decided once per loop, as LuaJIT's `ISNEXT` does: the iterator and
    // state are hidden, so only `debug.setlocal` can change them.
    let (iter, state) = (reg!(base), reg!(base + 1));
    *reg!(ref mut base + 3) = if state.get_table().is_none() {
        Value::nil()
    } else if iter.same_bits(&Value::function(ctx.next_fn())) && control.is_nil() {
        Value::small(0)
    } else if iter.same_bits(&Value::function(ctx.ipairs_iter())) {
        Value::small(TFOR_IPAIRS)
    } else {
        Value::nil()
    };
    if !closing.is_falsy() {
        if ctx.mm_of(closing, MetamethodBits::CLOSE).is_nil() {
            raise!(OpError::NonClosable(base + 2));
        }
        let frame_base = unsafe { (*frame).base() };
        thread
            .tbc_list
            .push(TbcEntry::Slot(frame_base + base as usize + 2));
        unsafe { (*frame).flags |= frame_flags::TBC };
    }
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: the loop variables = R[base](R[base+1], first variable).
///
/// When TFORPREP found the iterator to be the `next` `pairs` returns, or
/// `ipairs`'s, and the state a table, it set the position slot R[base+3],
/// and the step is taken without a call, and so is the following TFORLOOP's
/// jump. This handler takes the steps it can without a stack frame;
/// `tfor_next` and `tfor_ipairs` the rest. `next`'s walk keeps its position
/// in R[base+3], so a step doesn't look up the previous key; like LuaJIT's
/// `ITERN`, it then doesn't follow a key or iterator `debug.setlocal`
/// changes.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_tforcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, count) = instruction.ab();
    let vars = base + TFOR_VARS;
    let Some(t) = reg!(base + 1).get_table() else {
        tail!(tforcall_generic);
    };
    let step = match reg!(base + 3).get_small() {
        Some(TFOR_IPAIRS) if let Some(i) = reg!(vars).get_small() => {
            let state = t.inner().borrow();
            match i.checked_add(1) {
                Some(k)
                    if let Some(v) = state.array_get(k as usize)
                        && !v.is_nil() =>
                {
                    Some((Value::small(k), v))
                }
                _ => {
                    drop(state);
                    tail!(tfor_ipairs);
                }
            }
        }
        Some(pos) if pos >= 0 => match t.inner().borrow().next_at_inline(pos as u32) {
            Step::Entry(next, k, v) => {
                *reg!(ref mut base + 3) = Value::small(next as i32);
                Some((k, v))
            }
            Step::End => None,
            Step::Slow => tail!(tfor_next),
        },
        _ => tail!(tforcall_generic),
    };
    tfor_finish!(step, vars, count, ip);
}

/// [`op_tforcall`]'s `next` step through a part of integer or other keys.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn tfor_next<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, count) = instruction.ab();
    let vars = base + TFOR_VARS;
    // `op_tforcall` checked both.
    let (t, pos) = (reg!(base + 1).get_table(), reg!(base + 3).get_small());
    let (t, pos) = unsafe { (t.unwrap_unchecked(), pos.unwrap_unchecked() as u32) };
    let step = t.inner().borrow().next_at(ctx.mutation(), pos);
    // Past a position that doesn't fit, `next` resumes from the key.
    let step = step.map(|(next, k, v)| {
        *reg!(ref mut base + 3) = next.map_or(Value::nil(), |p| Value::small(p as i32));
        (k, v)
    });
    tfor_finish!(step, vars, count, ip);
}

/// [`op_tforcall`]'s `ipairs` step past the array part: through the integer
/// keys' hash part, to the end, or to `__index`.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn tfor_ipairs<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, count) = instruction.ab();
    let vars = base + TFOR_VARS;
    // `op_tforcall` checked it.
    let t = reg!(base + 1).get_table();
    let t = unsafe { t.unwrap_unchecked() };
    let i = reg!(vars).get_small();
    let i = unsafe { i.unwrap_unchecked() } as i64 + 1;
    let state = t.inner().borrow();
    let v = state.get_int(i);
    let step = if !v.is_nil() {
        Some((Value::integer(ctx.mutation(), i), v))
    } else if state.shape().has_mm(MetamethodBits::INDEX) {
        drop(state);
        tail!(tforcall_generic);
    } else {
        None
    };
    tfor_finish!(step, vars, count, ip);
}

/// [`op_tforcall`] by calling the iterator.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn tforcall_generic<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let base = instruction.a();
    let iter = reg!(base);
    let state = reg!(base + 1);
    let control = reg!(base + TFOR_VARS);
    call_mm!(ret_tfor, iter, [state, control]);
}

/// Generic for loop test: if the first loop variable != nil, jump back to
/// the body.
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (base, offset) = instruction.a_imm();
    if !reg!(base + TFOR_VARS).is_nil() {
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
) -> Exit {
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
    let items = &thread.stack[elements_start..elements_start + n];
    t.inner()
        .borrow_mut(ctx.mutation())
        .set_list(ctx.mutation(), offset as usize, items);
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, proto_idx) = instruction.ad();
    let (parent_closure, base) = unsafe { ((*frame).closure, (*frame).base()) };
    let proto = parent_closure.proto.prototypes[proto_idx as usize];

    let thread_handle = thread.thread_handle.expect("thread must have a handle");
    let mut upvalues_vec = Vec::with_capacity_in(
        proto.upvalue_desc.len(),
        crate::dmm::allocator_api::MetricsAlloc::new(ctx.mutation()),
    );
    for desc in proto.upvalue_desc.iter() {
        let uv = match desc {
            UpValueDescriptor::ParentLocal(idx) => {
                let stack_idx = base + *idx as usize;
                // Sorted by slot, so this frame's are at the end.
                let open = &thread.open_upvalues;
                let below = open.iter().rposition(|&uv| open_index(uv) <= stack_idx);
                match below {
                    Some(i) if open_index(open[i]) == stack_idx => open[i],
                    _ => {
                        let uv: Upvalue<'gc> = Gc::new(
                            ctx.mutation(),
                            RefLock::new(UpvalueState::Open {
                                thread: thread_handle,
                                index: stack_idx,
                            }),
                        );
                        let at = below.map_or(0, |i| i + 1);
                        thread.open_upvalues.insert(at, uv);
                        unsafe { (*frame).flags |= frame_flags::OPEN_UPVALUES };
                        uv
                    }
                }
            }
            UpValueDescriptor::ParentUpvalue(idx) => parent_closure.upvalues[*idx as usize],
        };
        upvalues_vec.push(uv);
    }
    let upvalues = upvalues_vec.into_boxed_slice();

    let func = Function::new_lua(ctx.mutation(), proto, upvalues);
    *reg!(ref mut dst) = Value::function(func);
    gc_check!();
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
) -> Exit {
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let (dst, count) = instruction.ab();
    let (base, num_extras) = unsafe { ((*frame).base(), (*frame).num_extras as usize) };
    let proto = closure.proto;
    let target = base + dst as usize;

    if proto.needs_vararg_table {
        // Materialized: read elements `1..=t.n` from the vararg table, nil past `n`.
        let table = thread.stack[base + proto.num_params as usize]
            .get_table()
            .expect("materialized vararg slot must hold a table");
        // Read `t.n` in a tight borrow so it's released before the resize.
        // PUC's `getnumargs` bound, checked even when `count` is fixed.
        let Some(navail) = table
            .inner()
            .borrow()
            .raw_get(Value::string(ctx.symbols().n))
            .get_integer()
            .filter(|n| (0..=i64::from(i32::MAX / 2)).contains(n))
        else {
            raise!(OpError::VarargN);
        };
        let navail = navail as usize;
        let wanted = if count == 0 {
            navail
        } else {
            count as usize - 1
        };
        let new_top = target + wanted;
        if !thread.ensure_frame_slots(new_top) {
            raise!(OpError::StackOverflow);
        }
        registers = unsafe { thread.stack.as_mut_ptr().add(base) };
        let filled = wanted.min(navail);
        let t = table.inner().borrow();
        for i in 0..filled {
            thread.stack[target + i] = t.raw_get(Value::integer(ctx.mutation(), i as i64 + 1));
        }
        thread.stack[target + filled..new_top].fill(Value::nil());
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
        if !thread.ensure_frame_slots(target + wanted) {
            raise!(OpError::StackOverflow);
        }
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
) -> Exit {
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
) -> Exit {
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
            Value::string(ctx.symbols().n),
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
) -> Exit {
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
) -> Exit {
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
) -> Exit {
    Exit::End
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether a to-be-closed variable is registered at or above `level`.
#[inline]
fn has_tbc_from(thread: &ThreadState<'_>, level: usize) -> bool {
    thread.tbc_list.last().is_some_and(|e| e.pos() >= level)
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
#[inline]
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
    let r = (nc.function)(ctx, nc, stack);
    if r.is_ok() && thread.native_overflowed() {
        return Err(native_overflow(ctx));
    }
    r
}

/// A native or sequence pushed without `Stack::check_stack`.
#[cold]
#[inline(never)]
pub(crate) fn native_overflow(ctx: Context<'_>) -> crate::env::Error<'_> {
    crate::env::Error::from_str(ctx, "stack overflow")
}

/// The stack slot of an upvalue in `ThreadState::open_upvalues`, which are
/// all open, sorted by slot (as Lua's `openupval` list is).
#[inline(always)]
fn open_index(uv: Upvalue<'_>) -> usize {
    match &*uv.borrow() {
        UpvalueState::Open { index, .. } => *index,
        UpvalueState::Closed(_) => unreachable!("closed upvalue in the open list"),
    }
}

/// Whether any open upvalue points at stack index `base` or above.
#[inline(always)]
fn frame_has_open_upvalues<'gc>(thread: &ThreadState<'gc>, base: usize) -> bool {
    thread
        .open_upvalues
        .last()
        .is_some_and(|&uv| open_index(uv) >= base)
}

/// Close all open upvalues pointing at stack indices >= `start_idx`.
/// Each open upvalue is converted to Closed by capturing the current stack value.
/// Inlined so the usual nothing-to-close case costs a load or two, not a call.
#[inline(always)]
pub(crate) fn close_upvalues<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    start_idx: usize,
) {
    if frame_has_open_upvalues(thread, start_idx) {
        close_upvalues_slow(mc, thread, start_idx);
    }
}

#[inline(never)]
fn close_upvalues_slow<'gc>(mc: &Mutation<'gc>, thread: &mut ThreadState<'gc>, start_idx: usize) {
    // Sorted by slot, so the ones to close are the tail.
    while let Some(&uv) = thread.open_upvalues.last() {
        let index = open_index(uv);
        if index < start_idx {
            break;
        }
        *uv.borrow_mut(mc) = UpvalueState::Closed(thread.stack[index]);
        thread.open_upvalues.pop();
    }
}

// ---------------------------------------------------------------------------
// Metamethod invocation / continuations
// ---------------------------------------------------------------------------

/// Maximum depth of `__index` / `__newindex` chains before we give up and
/// raise (Lua's `MAXTAGLOOP`).
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
    for _ in 0..MAX_TAG_LOOP {
        let mm = match receiver.get_table() {
            Some(t) => {
                let mm = t
                    .shape()
                    .mt_cache()
                    .map_or(Value::nil(), |c| c.mm(MetamethodBits::INDEX));
                if mm.is_nil() {
                    return IndexChain::Resolved(Value::nil());
                }
                mm
            }
            None => {
                let mm = ctx.mm_of(receiver, MetamethodBits::INDEX);
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
pub(crate) enum NewIndexChain<'gc> {
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
pub(crate) fn walk_newindex_chain<'gc>(
    ctx: Context<'gc>,
    mut t: Value<'gc>,
    key: Value<'gc>,
) -> NewIndexChain<'gc> {
    for _ in 0..MAX_TAG_LOOP {
        let mm = match t.get_table() {
            Some(tbl) => {
                let mm = tbl
                    .shape()
                    .mt_cache()
                    .map_or(Value::nil(), |c| c.mm(MetamethodBits::NEWINDEX));
                if mm.is_nil() || !tbl.raw_get(key).is_nil() {
                    return NewIndexChain::RawSet(tbl);
                }
                mm
            }
            None => {
                let mm = ctx.mm_of(t, MetamethodBits::NEWINDEX);
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
        let mm = ctx.mm_of(func_val, MetamethodBits::CALL);
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

/// The error [`resolve_call_chain`] would raise for calling `v`, found
/// without touching the stack.
pub(crate) fn call_chain_error<'gc>(ctx: Context<'gc>, mut v: Value<'gc>) -> Option<OpError<'gc>> {
    for _ in 0..=MAX_CALL_CHAIN {
        if v.get_function().is_some() {
            return None;
        }
        let mm = ctx.mm_of(v, MetamethodBits::CALL);
        if mm.is_nil() {
            return Some(OpError::Call(v));
        }
        v = mm;
    }
    Some(OpError::CallChainTooLong)
}

/// Binary metamethod `name`, taken from `lhs` first, then `rhs`.
#[inline]
pub(crate) fn binop_metamethod<'gc>(
    ctx: Context<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
    bit: MetamethodBits,
) -> Value<'gc> {
    let m = ctx.mm_of(lhs, bit);
    if !m.is_nil() {
        return m;
    }
    ctx.mm_of(rhs, bit)
}

// ---------------------------------------------------------------------------
// Continuations (`LuaFrame::ret`)
// ---------------------------------------------------------------------------

/// Make the call `call_mm!` staged above the running frame's registers: the
/// function at the slot `ip` points to (the frame's `pc` is already saved),
/// its arguments above it up to `top`, finished by `ds.ret`. A Lua function
/// gets a frame returning to it, a native one runs here and hands its results
/// straight to it.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn meta_call<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let scratch = unsafe { (ip as *const Value<'gc>).offset_from_unsigned(thread.stack.as_ptr()) };
    ip = unsafe { (*frame).pc };
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let ret = ds.ret;
    let new_base = scratch + 1;
    let nargs = thread.top - new_base;
    let f = thread.stack[scratch];
    match f.get_function().map(|f| (f, f.inner().as_ref())) {
        Some((f, FunctionKind::Lua(_))) => {
            let callee = unsafe { LuaFn::from_function_unchecked(f) };
            if !thread.ensure_frame_slots(new_base + callee.max_stack_size as usize) {
                raise!(OpError::StackOverflow);
            }
            let num_params = callee.num_params as usize;
            let stack = thread.stack.as_mut_ptr();
            unsafe {
                fill_nil(
                    stack.add(new_base + nargs),
                    num_params.saturating_sub(nargs),
                )
            };
            let num_extras = if callee.is_vararg {
                nargs.saturating_sub(num_params) as u16
            } else {
                0
            };
            ip = callee.code;
            thread.push_lua(LuaFrame {
                closure: callee,
                pc: ip,
                ret,
                base: new_base as u32,
                num_extras,
                flags: 0,
            });
            (frame, closure) = top_frame(thread);
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        Some((f, FunctionKind::Native(nc))) => {
            use crate::vm::sequence::CallbackAction;
            match invoke_native(ctx, thread, nc, new_base, nargs) {
                Ok(CallbackAction::Return) => {
                    let nret = thread.top - new_base;
                    let stack = thread.stack.as_mut_ptr();
                    let (values, func_slot) = unsafe { (stack.add(new_base), stack.add(scratch)) };
                    become ret(
                        Instruction::from_raw(nret as u64),
                        ctx,
                        thread,
                        values,
                        func_slot as *const Instruction,
                        handlers,
                        ds,
                        frame,
                        closure,
                    );
                }
                r => {
                    let step = run_natives(
                        ctx,
                        thread,
                        NativeState::Acted {
                            r,
                            framed: false,
                            f,
                            base: new_base,
                            ret,
                        },
                    );
                    native_step!(step);
                }
            }
        }
        None => {
            // Every hop adds one argument, and the count must still fit.
            debug_assert!(nargs + 1 + MAX_CALL_CHAIN < u8::MAX as usize);
            match resolve_call_chain(ctx, thread, scratch, nargs as u8 + 1) {
                // A count that outgrew `n` is already at `top`.
                Ok((_, n)) if n != 0 => thread.set_top(scratch + n as usize),
                Ok(_) => {}
                Err(e) => raise!(e),
            }
            let func_slot = unsafe { thread.stack.as_ptr().add(scratch) };
            tail!(meta_call, instruction, func_slot as *const Instruction);
        }
    }
}

/// Rebind the handler state to `frame`, the Lua frame a continuation
/// resumes, and evaluate to the instruction that made the call.
macro_rules! resume_caller {
    ($thread:ident, $registers:ident, $ip:ident, $frame:ident, $closure:ident) => {{
        let (__pc, __base, __closure) =
            unsafe { ((*$frame).pc, (*$frame).base(), (*$frame).closure) };
        $ip = __pc;
        $closure = __closure;
        $registers = unsafe { $thread.stack.as_mut_ptr().add(__base) };
        unsafe { *__pc.sub(1) }
    }};
}

/// Continuation of a CALL: land the results in its window, as many as its
/// `c` asks for.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_call<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (nret, values, dst) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let call = resume_caller!(thread, registers, ip, frame, closure);
    let returns = call.c();
    // `returns == 0` is MULTRET: deliver all `nret` and publish `thread.top`.
    let wanted = if returns == 0 {
        nret
    } else {
        returns as usize - 1
    };
    // The function slot is below the values, and the landing inside the
    // callee's window, which the CALL sized the vec for.
    unsafe { land_results(dst, values, nret, wanted) };
    // Without this a `top` left high by a multires producer inside the callee
    // would keep its dead registers traced (#43).
    let dst = unsafe { dst.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(dst + wanted);
    dispatch!();
}

/// Continuation of a frame the executor made: leave the results at the
/// function slot, up to `top`, and leave dispatch.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_exit<'gc>(
    instruction: Instruction,
    _ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    _handlers: *const (),
    _ds: &mut DispatchState<'gc>,
    _frame: *mut LuaFrame<'gc>,
    _closure: LuaFn<'gc>,
) -> Exit {
    let (nret, values, dst) = ret_args!(instruction, registers, ip);
    unsafe { copy_values(dst, values, nret) };
    let dst = unsafe { dst.offset_from_unsigned(thread.stack.as_ptr()) };
    if thread.frames_empty() {
        // The thread is done, so nothing above the results is live: pair
        // `Result { bottom }` with a `top` marking the result end, and release
        // the rest. Consumers read `stack[bottom..top]`.
        thread.discard_above(dst + nret);
        thread.status = ThreadStatus::Result { bottom: dst };
    } else {
        thread.set_top_unchecked(dst + nret);
    }
    Exit::End
}

/// The first of a call's results, nil if it has none.
#[inline(always)]
fn first_result<'gc>(nret: usize, values: *const Value<'gc>) -> Value<'gc> {
    if nret > 0 {
        unsafe { values.read() }
    } else {
        Value::nil()
    }
}

/// Continuation of a metamethod whose result goes to `R[A]`: arithmetic,
/// unary, concatenation and `__index`.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_store_a<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (nret, values, func_slot) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let v = first_result(nret, values);
    let op = resume_caller!(thread, registers, ip, frame, closure);
    *reg!(ref mut op.a()) = v;
    let func_slot = unsafe { func_slot.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(func_slot);
    dispatch!();
}

/// Continuation of a `__newindex` call: the results are dropped.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_discard<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (_, _, func_slot) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let _ = resume_caller!(thread, registers, ip, frame, closure);
    let func_slot = unsafe { func_slot.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(func_slot);
    dispatch!();
}

/// Continuation of a comparison metamethod: skip the following JMP unless
/// the result's truthiness matches the flag, in `c` (`EQ`/`LT`/`LE`) or in
/// `b` (`EQI`/`LTI`/...).
macro_rules! ret_cond {
    ($name:ident, $flag:ident) => {
        #[inline(never)]
        #[rustc_align(32)]
        // The incoming `closure` belongs to the finished call.
        #[allow(unused_assignments)]
        extern "rust-preserve-none" fn $name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
            ds: &mut DispatchState<'gc>,
            frame: *mut LuaFrame<'gc>,
            closure: LuaFn<'gc>,
        ) -> Exit {
            let (nret, values, func_slot) = ret_args!(instruction, registers, ip);
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let truthy = !first_result(nret, values).is_falsy();
            let op = resume_caller!(thread, registers, ip, frame, closure);
            let func_slot = unsafe { func_slot.offset_from_unsigned(thread.stack.as_ptr()) };
            thread.set_top_unchecked(func_slot);
            skip_if!(truthy != (op.$flag() != 0));
            dispatch!();
        }
    };
}

ret_cond!(ret_cond_c, c);
ret_cond!(ret_cond_b, b);

/// Continuation of a generic for's iterator: its results are the loop's
/// variables, nil-padded.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_tfor<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (nret, values, func_slot) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let op = resume_caller!(thread, registers, ip, frame, closure);
    let (base, count) = op.ab();
    // The variables are in the caller's window, below the results.
    unsafe {
        land_results(
            registers.add((base + TFOR_VARS) as usize),
            values,
            nret,
            count as usize,
        )
    };
    let func_slot = unsafe { func_slot.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(func_slot);
    dispatch!();
}

/// Continuation of a CLOSE's `__close`: run the CLOSE again, for the next
/// variable or to move on.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_close<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (_, _, func_slot) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let _ = resume_caller!(thread, registers, ip, frame, closure);
    ip = unsafe { ip.sub(1) };
    let func_slot = unsafe { func_slot.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(func_slot);
    dispatch!();
}

/// Continuation of a RETURN's `__close`: run the RETURN again, with `top`
/// back at the end of its results, which the slot below the call recorded.
#[inline(never)]
#[rustc_align(32)]
// The incoming `closure` belongs to the finished call.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn ret_return<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (_, _, func_slot) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let _ = resume_caller!(thread, registers, ip, frame, closure);
    ip = unsafe { ip.sub(1) };
    let values_end = unsafe { func_slot.sub(1).read() }.get_small();
    thread.set_top_unchecked(unsafe { values_end.unwrap_unchecked() } as usize);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Native frames
// ---------------------------------------------------------------------------

/// Where [`run_natives`] starts.
pub(crate) enum NativeState<'gc> {
    /// Native `f`, with its window at `base`, asked for `r`: without a frame
    /// (it ran inline, its results going to `ret`), or as the top frame's
    /// continuation.
    Acted {
        r: Result<crate::vm::sequence::CallbackAction<'gc>, crate::env::Error<'gc>>,
        framed: bool,
        f: Function<'gc>,
        base: usize,
        ret: Handler,
    },
    /// Run the top (native) frame's continuation.
    Resume(Result<(), crate::env::Error<'gc>>),
    /// Make the call the top (native) frame waits for.
    Call,
}

/// What [`run_natives`] left to do.
pub(crate) enum NativeStep {
    /// A Lua frame was pushed: dispatch into it.
    EnterLua,
    /// A native frame (or a native without one) returned `stack[values..top]`,
    /// its function slot `func_slot`: run `ret` with the frame below on top.
    Return {
        ret: Handler,
        func_slot: usize,
        values: usize,
    },
    /// A suspension or error is installed for the executor.
    Exit,
}

/// Drive natives that call: push a native's frame, make its call, and run
/// its continuation on the results, for as long as no Lua code has to run
/// and the native doesn't return.
pub(crate) fn run_natives<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut state: NativeState<'gc>,
) -> NativeStep {
    use crate::vm::sequence::{CallbackAction, NativeCont};
    loop {
        state = match state {
            NativeState::Resume(status) => {
                let frame = unsafe { thread.frames.last_mut().unwrap_unchecked() };
                debug_assert!(frame.is_native());
                // Errors the continuation itself raises unwind past it.
                frame.flags &= !(frame_flags::PROTECTED | frame_flags::HANDLER);
                let (f, base, ret) = (frame.closure.function(), frame.base(), frame.ret);
                let cont: NativeCont = unsafe { std::mem::transmute(frame.pc) };
                let nc = unsafe { f.as_native().unwrap_unchecked() };
                let mut r = cont(ctx, nc, Stack::new(thread, base), status);
                if r.is_ok() && thread.native_overflowed() {
                    r = Err(native_overflow(ctx));
                }
                NativeState::Acted {
                    r,
                    framed: true,
                    f,
                    base,
                    ret,
                }
            }
            NativeState::Acted {
                r,
                framed,
                f,
                base,
                mut ret,
            } => match r {
                Ok(CallbackAction::Return) => {
                    if framed {
                        thread.pop_lua();
                    }
                    return NativeStep::Return {
                        ret,
                        func_slot: base - 1,
                        values: base,
                    };
                }
                Ok(CallbackAction::CallThen { at, protect, cont }) => {
                    if framed {
                        thread.pop_lua();
                    }
                    push_native_frame(thread, f, base, at, protect, cont, ret);
                    NativeState::Call
                }
                Ok(CallbackAction::Suspend(action)) => {
                    if framed {
                        ret = thread.frames.pop().map(|f| f.ret).unwrap_or(ret);
                    }
                    thread.pending_action = Some(PendingAction {
                        action,
                        call_site: CallSite {
                            bottom: base,
                            func_idx: base - 1,
                            ret,
                        },
                    });
                    return NativeStep::Exit;
                }
                Err(err) => {
                    // Raised at the native's caller, as from a native that
                    // has no frame.
                    if framed {
                        thread.pop_lua();
                    }
                    thread.raise(ctx, err);
                    return NativeStep::Exit;
                }
            },
            NativeState::Call => {
                let frame = unsafe { thread.frames.last().unwrap_unchecked() };
                let slot = frame.base() + frame.num_extras as usize;
                let new_base = slot + 1;
                let nargs = thread.top - new_base;
                let fv = thread.stack[slot];
                match fv.get_function().map(|f| (f, f.inner().as_ref())) {
                    Some((f, FunctionKind::Lua(_))) => {
                        let callee = unsafe { LuaFn::from_function_unchecked(f) };
                        if !thread.ensure_frame_slots(new_base + callee.max_stack_size as usize) {
                            let err = crate::vm::debug::stack_overflow(ctx, thread);
                            thread.raise(ctx, err);
                            return NativeStep::Exit;
                        }
                        enter_from_native(thread, callee, new_base);
                        return NativeStep::EnterLua;
                    }
                    Some((f, FunctionKind::Native(nc))) => {
                        match invoke_native(ctx, thread, nc, new_base, nargs) {
                            Ok(CallbackAction::Return) => {
                                let nret = thread.top - new_base;
                                thread.stack.copy_within(new_base..new_base + nret, slot);
                                thread.set_top_unchecked(slot + nret);
                                NativeState::Resume(Ok(()))
                            }
                            r => NativeState::Acted {
                                r,
                                framed: false,
                                f,
                                base: new_base,
                                ret: ret_native,
                            },
                        }
                    }
                    None => {
                        // MULTRET: the count is at `top`, where the chain
                        // keeps it.
                        match resolve_call_chain(ctx, thread, slot, 0) {
                            Ok(_) => {}
                            Err(e) => {
                                let msg = crate::vm::debug::op_error_message(ctx, thread, e);
                                thread.raise(ctx, crate::env::Error::from_str(ctx, &msg));
                                return NativeStep::Exit;
                            }
                        }
                        NativeState::Call
                    }
                }
            }
        }
    }
}

/// Continuation of a call a native frame made: its results go to the
/// frame's window, where it said, and its continuation runs.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_native<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    let (nret, values, _) = ret_args!(instruction, registers, ip);
    helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
    let nf = unsafe { &mut *frame };
    debug_assert!(nf.is_native());
    let (base, slot) = (nf.base(), nf.base() + nf.num_extras as usize);
    // The results sit above the slot, in the finished call's window.
    unsafe { copy_values(thread.stack.as_mut_ptr().add(slot), values, nret) };
    thread.set_top_unchecked(slot + nret);
    // As `run_natives` would, with a returning continuation inline.
    nf.flags &= !(frame_flags::PROTECTED | frame_flags::HANDLER);
    let (f, ret) = (nf.closure.function(), nf.ret);
    let cont: crate::vm::sequence::NativeCont = unsafe { std::mem::transmute(nf.pc) };
    let nc = unsafe { f.as_native().unwrap_unchecked() };
    let r = cont(ctx, nc, Stack::new(thread, base), Ok(()));
    if let Ok(crate::vm::sequence::CallbackAction::Return) = r
        && !thread.native_overflowed()
    {
        let n = thread.frames.len();
        unsafe { thread.frames.set_len(n - 1) };
        let stack = thread.stack.as_mut_ptr();
        let nret = thread.top - base;
        become ret(
            Instruction::from_raw(nret as u64),
            ctx,
            thread,
            unsafe { stack.add(base) },
            unsafe { stack.add(base - 1) } as *const Instruction,
            handlers,
            ds,
            unsafe { frame.sub(1) },
            closure,
        );
    }
    let r = match r {
        Ok(_) if thread.native_overflowed() => Err(native_overflow(ctx)),
        r => r,
    };
    let step = run_natives(
        ctx,
        thread,
        NativeState::Acted {
            r,
            framed: true,
            f,
            base,
            ret,
        },
    );
    native_step!(step);
}

/// The Lua function at `slot` when a native's call of it can be entered
/// straight away: room for its window and for two more frames.
#[inline(always)]
fn native_calls_lua<'gc>(thread: &ThreadState<'gc>, slot: usize) -> Option<LuaFn<'gc>> {
    let f = thread.stack[slot].get_function()?;
    let FunctionKind::Lua(callee) = f.inner().as_ref() else {
        return None;
    };
    let fits = thread.stack.len() >= slot + 1 + callee.max_stack_size as usize
        && thread.frames.capacity() - thread.frames.len() >= 2;
    fits.then(|| unsafe { LuaFn::from_function_unchecked(f) })
}

/// Push the frame of native `f`, whose window is at `base`, for its
/// `CallThen`.
#[inline(always)]
fn push_native_frame<'gc>(
    thread: &mut ThreadState<'gc>,
    f: Function<'gc>,
    base: usize,
    at: u32,
    protect: crate::vm::sequence::Protect,
    cont: crate::vm::sequence::NativeCont,
    ret: Handler,
) {
    use crate::vm::sequence::Protect;
    let flags = frame_flags::NATIVE
        | match protect {
            Protect::No => 0,
            Protect::Errors => frame_flags::PROTECTED,
            Protect::Handler => frame_flags::PROTECTED | frame_flags::HANDLER,
        };
    thread.push_lua(LuaFrame {
        closure: unsafe { LuaFn::native_frame(f) },
        pc: cont as *const Instruction,
        ret,
        base: base as u32,
        num_extras: at as u16,
        flags,
    });
}

/// Push the frame of a native's call of Lua `callee`, its arguments at
/// `new_base` up to `top` and its window known to fit.
#[inline(always)]
fn enter_from_native<'gc>(thread: &mut ThreadState<'gc>, callee: LuaFn<'gc>, new_base: usize) {
    let nargs = thread.top - new_base;
    let num_params = callee.num_params as usize;
    let stack = thread.stack.as_mut_ptr();
    unsafe {
        fill_nil(
            stack.add(new_base + nargs),
            num_params.saturating_sub(nargs),
        )
    };
    let num_extras = if callee.is_vararg {
        nargs.saturating_sub(num_params) as u16
    } else {
        0
    };
    thread.push_lua(LuaFrame {
        closure: callee,
        pc: callee.code,
        ret: ret_native,
        base: new_base as u32,
        num_extras,
        flags: 0,
    });
}

/// `run_thread`'s start when the unwinder hands an error to the protected
/// native frame on top: run its continuation with it. Not a handler: the
/// handlers it enters are called, from `run_thread`'s frame.
#[inline(never)]
fn native_entry<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    closure: LuaFn<'gc>,
    err: crate::env::Error<'gc>,
) -> Exit {
    match run_natives(ctx, thread, NativeState::Resume(Err(err))) {
        NativeStep::EnterLua => {
            let (frame, closure) = top_frame(thread);
            let (ip, base) = unsafe { ((*frame).pc, (*frame).base()) };
            let registers = unsafe { thread.stack.as_mut_ptr().add(base) };
            op_nop(
                Instruction::nop(),
                ctx,
                thread,
                registers,
                ip,
                handlers,
                ds,
                frame,
                closure,
            )
        }
        NativeStep::Return {
            ret,
            func_slot,
            values,
        } => {
            let nret = thread.top - values;
            let stack = thread.stack.as_mut_ptr();
            let frame = thread
                .frames
                .as_mut_ptr()
                .wrapping_add(thread.frames.len())
                .wrapping_sub(1);
            ret(
                Instruction::from_raw(nret as u64),
                ctx,
                thread,
                unsafe { stack.add(values) },
                unsafe { stack.add(func_slot) } as *const Instruction,
                handlers,
                ds,
                frame,
                closure,
            )
        }
        NativeStep::Exit => Exit::End,
    }
}
