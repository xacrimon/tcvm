use crate::dmm::{Gc, Mutation};
use crate::env::function::{
    Function, FunctionKind, InlineCache, LuaFn, NativeClosure, NativeKind, Stack, UpvalueCell,
    UpvalueSlot,
};
use crate::env::shape::{MAX_PROPERTIES_FAST, MetamethodBits, Shape, mirrored};
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, Step, Table, TableState};
use crate::env::thread::{
    CallSite, ExecKind, LuaFrame, PendingAction, PendingKind, TbcEntry, Thread, ThreadState,
    ThreadStatus, frame_flags,
};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, Op, TFOR_VARS, UpvalSource};
use crate::lua::Context;
use crate::vm::num;
use crate::vm::unwind::Unwound;

/// A handler per opcode, then CALL's continuation per C operand, so that
/// CALL reads it off the `handlers` register (`call_ret`).
#[repr(C)]
struct Dispatch {
    ops: [Handler; Op::COUNT],
    rets: [Handler; 256],
}

static DISPATCH: Dispatch = Dispatch {
    ops: HANDLERS,
    rets: {
        let mut t: [Handler; 256] = [ret_call; 256];
        t[1] = ret_call0;
        t[2] = ret_call1;
        t[3] = ret_call2;
        t
    },
};

const HANDLERS: [Handler; Op::COUNT] = Op::table([
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
    (Op::GETFIELD_OWN, getfield_own),
    (Op::GETFIELD_ABSENT, getfield_absent),
    (Op::GETFIELD_PROTO, getfield_proto),
    (Op::GETTABUP_OWN, gettabup_own),
    (Op::GETTABUP_ABSENT, gettabup_absent),
    (Op::GETTABUP_PROTO, gettabup_proto),
    (Op::SELF_OWN, self_own),
    (Op::SELF_ABSENT, self_absent),
    (Op::SELF_PROTO, self_proto),
    (Op::SETFIELD_OWN, setfield_own),
    (Op::SETFIELD_TRANS, setfield_trans),
    (Op::SETTABUP_OWN, settabup_own),
    (Op::SETTABUP_TRANS, settabup_trans),
    (Op::SETFIELD_ABSENT, setfield_absent),
    (Op::SETTABUP_ABSENT, settabup_absent),
    (Op::GETUPVAL_REF, op_getupval_ref),
    (Op::GETTABUP_REF, op_gettabup_ref),
    (Op::SETTABUP_REF, op_settabup_ref),
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
    /// Raised by a native or a metamethod, the running frame's pc already
    /// saved (see `throw!`).
    Thrown(crate::env::Error<'gc>),
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
    /// The thread dispatch switched to last, if it did.
    current: Option<Thread<'gc>>,
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
    /// An async native waits on the host; dispatch resumes by polling it.
    Pending,
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
    ($instruction:expr, $ctx:expr, $thread:ident, $registers:ident, $ip:ident, $handlers:expr, $ds:ident, $frame:ident, $closure:ident) => {
        // The running frame and its closure travel in registers with the
        // other handler arguments; rebinding them (calls, returns) reassigns
        // these locals, and switching coroutines the thread.
        #[allow(unused_mut)]
        let mut $thread: &mut ThreadState<'gc> = $thread;
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
                    debug_assert!(pos < Op::COUNT);
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

        /// Raise `err` from the frames as they are, the running one's `pc`
        /// saved: unwind to the frame that catches it and continue there.
        /// Through `impl_error`, so the unwinder's call stays off the
        /// raising handler's fast path.
        #[allow(unused_macros)]
        macro_rules! throw {
            ($$err:expr) => {{
                std::hint::cold_path();
                $ds.fault = Some(OpError::Thrown($$err));
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

        /// The closure's upvalue slot `idx`, or the field its descriptor
        /// says it holds.
        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                unsafe {
                    debug_assert!(($$idx as usize) < $closure.upvalues().len());
                    *$closure.upvalue_ptr().add($$idx as usize)
                }
            }};
            (value $$idx:expr) => {{
                let slot = upvalue!($$idx);
                debug_assert!($closure.proto.upvalue_desc[$$idx as usize].by_value);
                unsafe { slot.value }
            }};
            (cell $$idx:expr) => {{
                let slot = upvalue!($$idx);
                debug_assert!(!$closure.proto.upvalue_desc[$$idx as usize].by_value);
                unsafe { slot.cell }
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

        /// [`run_natives`] from a handler, which may switch coroutines.
        #[allow(unused_macros)]
        macro_rules! run_natives {
            ($$state:expr $$(,)?) => {{
                let __before: *const ThreadState<'gc> = &*$thread;
                let __step = run_natives($ctx, &mut $thread, true, $$state);
                if !std::ptr::eq(__before, &*$thread) {
                    $ds.current = Some($thread.handle());
                }
                __step
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
                        let __top = $thread.frames.top_ptr();
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
                    NativeStep::Pending => return Exit::Pending,
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
fn fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    site: *const Instruction,
    entry: InlineCache<'gc>,
) {
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
        let first = matches!(slot_lock.get(), InlineCache::Empty);
        unsafe { slot_lock.as_cell() }.set(entry);
        quicken(site, &entry, first);
    }
}

/// Refills a table access site takes before it counts as megamorphic and
/// stops refilling (see [`site_fills`]).
const MEGAMORPHIC: u8 = 16;

/// Whether a miss at the table access `insn` should refill its cache: not
/// once it has been refilled [`MEGAMORPHIC`] times, which its unused `c`
/// slot counts. Refilling a site that sees a new shape each time (a
/// metatable per object) costs a barrier and a rewrite on every miss.
#[inline(always)]
fn site_fills(insn: Instruction) -> bool {
    insn.c() < MEGAMORPHIC
}

/// Rewrite the table access at `site` to the form for `entry`, its cache's
/// new contents (see `Op::unquickened`), and count a refill. Only a `first`
/// fill quickens: a site whose entry changes kind goes back to the generic
/// form for good, rather than flipping its opcode (and its dispatch target)
/// on every miss.
#[inline]
fn quicken(site: *const Instruction, entry: &InlineCache<'_>, first: bool) {
    let mut insn = unsafe { *site };
    if !first {
        insn.set_c(insn.c().saturating_add(1));
    }
    let op = match (insn.op().unquickened(), entry) {
        (Op::GETFIELD, InlineCache::Own { .. }) => Op::GETFIELD_OWN,
        (Op::GETFIELD, InlineCache::Absent { .. }) => Op::GETFIELD_ABSENT,
        (Op::GETFIELD, InlineCache::ProtoLoad { .. }) => Op::GETFIELD_PROTO,
        (Op::GETTABUP, InlineCache::Own { .. }) => Op::GETTABUP_OWN,
        (Op::GETTABUP, InlineCache::Absent { .. }) => Op::GETTABUP_ABSENT,
        (Op::GETTABUP, InlineCache::ProtoLoad { .. }) => Op::GETTABUP_PROTO,
        (Op::SELF, InlineCache::Own { .. }) => Op::SELF_OWN,
        (Op::SELF, InlineCache::Absent { .. }) => Op::SELF_ABSENT,
        (Op::SELF, InlineCache::ProtoLoad { .. }) => Op::SELF_PROTO,
        (Op::SETFIELD, InlineCache::Own { .. }) => Op::SETFIELD_OWN,
        (Op::SETFIELD, InlineCache::Transition { .. }) => Op::SETFIELD_TRANS,
        (Op::SETTABUP, InlineCache::Own { .. }) => Op::SETTABUP_OWN,
        (Op::SETTABUP, InlineCache::Transition { .. }) => Op::SETTABUP_TRANS,
        (Op::SETFIELD, InlineCache::Absent { .. }) => Op::SETFIELD_ABSENT,
        (Op::SETTABUP, InlineCache::Absent { .. }) => Op::SETTABUP_ABSENT,
        (op, _) => op,
    };
    let op = if first || op == insn.op() {
        op
    } else {
        insn.op().unquickened()
    };
    // SAFETY: `Code` keeps instructions in cells, and `site` came from one.
    unsafe { site.cast_mut().write(insn.with_op(op)) };
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
    site: *const Instruction,
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
    // The entry already says `t` lacks the key: refilling it would cost a
    // barrier on every miss. Only `__index` is left.
    if let InlineCache::Absent { shape: cached } = read_ic(closure, ic_idx)
        && Shape::ptr_eq(cached, shape)
        && let Some(mt) = shape.mt_cache()
    {
        let index = mt.mm(MetamethodBits::INDEX);
        drop(state);
        if index.get_function().is_some() {
            return Err(Value::table(t));
        }
        return get_index_fill_ic(ctx, closure, ic_idx, site, t, index, Some(shape), k);
    }
    let slot = shape.find_slot(constant_key(k));
    if site_fills(unsafe { *site }) {
        fill_ic(ctx, closure, ic_idx, site, shape_entry(shape, slot));
    }
    let v = slot.map_or(Value::nil(), |s| state.named_get(s));
    if !v.is_nil() || !shape.has_mm(MetamethodBits::INDEX) {
        return Ok(v);
    }
    // INDEX bit implies a metatable.
    let index = unsafe { shape.mt_cache().unwrap_unchecked() }.mm(MetamethodBits::INDEX);
    drop(state);
    let recv = slot.is_none().then_some(shape);
    get_index_fill_ic(ctx, closure, ic_idx, site, t, index, recv, k)
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
    site: *const Instruction,
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
        if site_fills(unsafe { *site }) {
            fill_ic(ctx, closure, ic_idx, site, entry);
        }
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
    site: *const Instruction,
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
            if site_fills(unsafe { *site }) {
                fill_ic(ctx, closure, ic_idx, site, shape_entry(shape, slot));
            }
        }
        return false;
    }
    drop(state);
    let mut state = t.inner().borrow_mut(ctx.mutation());
    let entry = match slot {
        Some(slot) => {
            state.named_set(slot, v);
            state.maybe_update_mt_bit(ctx.mutation(), k, v);
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
        if site_fills(unsafe { *site }) {
            fill_ic(ctx, closure, ic_idx, site, entry);
        }
    }
    true
}

/// Drive the VM on `thread` until the top-level frame returns.
///
/// The caller must have seeded the thread with at least one `LuaFrame`,
/// sized `stack` to at least `base + max_stack_size`, and placed
/// the callee + arguments at `stack[base-1..]`. See `Executor::start`.
#[inline(never)]
pub(crate) fn run_thread<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) -> (Exit, Thread<'gc>) {
    // SAFETY: the executor runs one thread at a time and holds no borrow of
    // it; dispatch may switch to coroutines it resumes the same way.
    let ts = unsafe { thread.state_mut(ctx.mutation()) };
    let mut ds = DispatchState {
        fault: None,
        native: std::ptr::null(),
        ret: ret_exit,
        current: None,
    };
    let exit = enter(ctx, ts, &mut ds);
    (exit, ds.current.unwrap_or(thread))
}

/// Start dispatch on `ts`: run what the executor delivered, or the top
/// frame from its `pc`.
fn enter<'gc>(ctx: Context<'gc>, ts: &mut ThreadState<'gc>, ds: &mut DispatchState<'gc>) -> Exit {
    let handlers = &DISPATCH as *const Dispatch as *const ();
    if let Some(p) = ts.pending_ret.take() {
        // The call site's frame, if it has one: a coroutine's body has none.
        let frame = ts.frames.top_ptr();
        let closure = match ts.frames.last() {
            Some(f) => f.closure,
            // Never dereferenced: no continuation reads the closure.
            None => unsafe { LuaFn::native_frame(ctx.next_fn()) },
        };
        let nret = ts.top - p.values;
        let stack = ts.stack.as_mut_ptr();
        let (values, func_slot) = unsafe { (stack.add(p.values), stack.add(p.func_slot)) };
        return (p.ret)(
            Instruction::from_raw(nret as u64),
            ctx,
            ts,
            values,
            func_slot as *const Instruction,
            handlers,
            ds,
            frame,
            closure,
        );
    }
    let top = ts
        .top_lua()
        .expect("run_thread requires a seeded Lua frame");
    if top.is_native() {
        // An async native that waited on the host: poll it again.
        debug_assert!(std::ptr::eq(
            top.pc,
            crate::vm::async_native::async_cont as *const Instruction
        ));
        return native_entry(ctx, ts, handlers, ds, NativeState::Resume(Ok(())));
    }
    let (frame, closure) = top_frame(ts);
    let (ip, base) = unsafe { ((*frame).pc, (*frame).base()) };
    let registers = unsafe { ts.stack.as_mut_ptr().add(base) };
    op_nop(
        Instruction::nop(),
        ctx,
        ts,
        registers,
        ip,
        handlers,
        ds,
        frame,
        closure,
    )
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Cold tail of `raise!`: publish the faulting frame's pc, then raise the
/// reference-formatted message at level 1 (the faulting Lua frame itself)
/// and unwind it to the frame that catches it. A handler so it can
/// be `become`d: a plain call here would put a frame on every raising
/// handler's fast path.
#[inline(never)]
#[rustc_align(32)]
// The incoming frame state is only rebound, to wherever the error is caught.
#[allow(unused_assignments)]
extern "rust-preserve-none" fn impl_error<'gc>(
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
    let err = match ds.fault.take().expect("impl_error without a pending fault") {
        OpError::Thrown(err) => err,
        kind => {
            // `ip` already points past the faulting instruction (see
            // `dispatch!`), which is the convention `LuaFrame::pc` uses.
            save_pc(thread, ip);
            match kind {
                OpError::StackOverflow => crate::vm::debug::stack_overflow(ctx, thread),
                kind => {
                    let msg = crate::vm::debug::op_error_message(ctx, thread, kind);
                    crate::env::Error::from_str(ctx, &msg)
                }
            }
        }
    };
    let step = run_natives!(NativeState::Raise(err));
    native_step!(step);
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

/// R[dst] = UpValue[idx], a by-value upvalue
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
    *reg!(ref mut dst) = upvalue!(value idx);
    dispatch!();
}

/// R[dst] = UpValue[idx], a shared cell
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_getupval_ref<'gc>(
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
    *reg!(ref mut dst) = upvalue!(cell idx).get();
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
    let cell = upvalue!(cell idx);
    UpvalueCell::set(cell, ctx.mutation(), thread.handle(), val);
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
    let t_val = upvalue!(value idx);
    let Some(t) = t_val.get_table() else {
        tail!(get_slow);
    };
    let t_state = t.inner().borrow();
    if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &t_state) {
        drop(t_state);
        *reg!(ref mut dst) = v;
        dispatch!();
    }
    drop(t_state);
    tail!(get_slow);
}

/// `op_gettabup` on a shared cell
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_gettabup_ref<'gc>(
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
    let t_val = upvalue!(cell idx).get();
    let Some(t) = t_val.get_table() else {
        tail!(get_slow);
    };
    let t_state = t.inner().borrow();
    if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &t_state) {
        drop(t_state);
        *reg!(ref mut dst) = v;
        dispatch!();
    }
    drop(t_state);
    tail!(get_slow);
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
    let t_val = upvalue!(value idx);
    let Some(t) = t_val.get_table() else {
        tail!(set_slow);
    };
    if ic_set(ctx, read_ic(closure, ic_idx), t, reg!(src)) {
        dispatch!();
    }
    tail!(set_slow);
}

/// `op_settabup` on a shared cell
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_settabup_ref<'gc>(
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
    let t_val = upvalue!(cell idx).get();
    let Some(t) = t_val.get_table() else {
        tail!(set_slow);
    };
    if ic_set(ctx, read_ic(closure, ic_idx), t, reg!(src)) {
        dispatch!();
    }
    tail!(set_slow);
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
        tail!(get_slow);
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
        tail!(get_slow);
    };

    let k = reg!(key);
    let (v, need_index) = {
        let t_state = t.inner().borrow();
        let v = t_state.raw_get(k);
        let need = v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        tail!(get_slow);
    }

    *reg!(ref mut dst) = v;
    dispatch!();
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
        tail!(set_slow);
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
        tail!(set_slow);
    };

    let k = reg!(key);
    let v = reg!(src);
    let needs_newindex = {
        let t_state = t.inner().borrow();
        t_state.shape().has_mm(MetamethodBits::NEWINDEX) && t_state.raw_get(k).is_nil()
    };

    if needs_newindex {
        tail!(set_slow);
    }

    check_index_key!(k);
    let mut t_state = t.inner().borrow_mut(ctx.mutation());
    t_state.raw_set_keyed(ctx, k, v);
    dispatch!()
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
        tail!(get_slow);
    };

    let t_state = t.inner().borrow();
    if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &t_state) {
        drop(t_state);
        *reg!(ref mut dst) = v;
        dispatch!();
    }
    drop(t_state);
    tail!(get_slow);
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
        tail!(set_slow);
    };

    if ic_set(ctx, read_ic(closure, ic_idx), t, reg!(src)) {
        dispatch!();
    }
    tail!(set_slow);
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
        tail!(get_slow);
    };

    let recv_state = recv.inner().borrow();
    if let Some(method) = ic_get(read_ic(closure, ic_idx), recv, &recv_state) {
        drop(recv_state);
        *reg!(ref mut dst) = method;
        *reg!(ref mut (dst + 1)) = recv_val;
        dispatch!();
    }
    drop(recv_state);
    tail!(get_slow);
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

            tail!(binop_slow);
        }
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

            tail!(binop_slow);
        }
    };
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

            tail!(binop_slow);
        }
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

            tail!(binop_slow);
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
            throw!(err);
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
                tail!(cmp_slow);
            };
            skip_if!(r != inverted);
            dispatch!();
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

            tail!(cmp_slow);
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

/// Land the results a native left at `args_base..top` at `func_idx`, as
/// many as `returns` asks for, and continue after the CALL.
macro_rules! land_native {
    ($args_base:expr, $func_idx:expr, $returns:expr, $base:expr,
     $thread:ident, $registers:ident) => {{
        let (args_base, func_idx, returns) = ($args_base, $func_idx, $returns);
        // The count comes via the logical top: a native never shrinks the
        // shared stack, so the caller's register window is still covered.
        let retc = $thread.top - args_base;
        let wanted = if returns == 0 {
            retc
        } else {
            returns as usize - 1
        };
        // The nil padding stays inside the caller's frame, which the CALL
        // that entered it sized the vec for.
        debug_assert!(args_base + retc.min(wanted) <= $thread.stack.len());
        debug_assert!(func_idx + wanted <= $thread.stack.len());
        let stack = $thread.stack.as_mut_ptr();
        unsafe { land_results(stack.add(func_idx), stack.add(args_base), retc, wanted) };
        // For MULTRET this is the dynamic count the next consumer reads.
        $thread.set_top_unchecked(func_idx + wanted);
        $registers = unsafe { $thread.stack.as_mut_ptr().add($base) };
        gc_check!();
        dispatch!();
    }};
}

/// The arguments of the CALL of a native at `func_idx`: their first slot and
/// their count.
macro_rules! native_args {
    ($func_idx:expr, $nargs:expr, $thread:ident) => {{
        let args_base = $func_idx + 1;
        let argc = if $nargs == 0 {
            $thread.top - args_base
        } else {
            $nargs as usize - 1
        };
        (args_base, argc)
    }};
}

/// The plain-native arm of CALL: run `$f` inline and land its results at
/// `func_idx`. Expands inside a handler body (needs its `dispatch!`).
macro_rules! call_plain {
    ($f:expr, $nc:expr, $func_idx:expr, $nargs:expr, $returns:expr, $base:expr,
     $ctx:ident, $thread:ident, $registers:ident, $ip:ident, $frame:ident) => {{
        let func_idx = $func_idx;
        let (args_base, argc) = native_args!(func_idx, $nargs, $thread);
        if let Err(err) = invoke_plain($ctx, $thread, $f, $nc, args_base, argc) {
            // Saved so the unwinder locates the error at this CALL.
            unsafe { (*$frame).pc = $ip };
            throw!(err);
        }
        land_native!(args_base, func_idx, $returns, $base, $thread, $registers);
    }};
}

/// The action- or async-native arm of CALL: run the native of kind `$kind`
/// inline and do what it asks. With `$async` false, the kind is known not to
/// be async.
macro_rules! call_action {
    ($async:literal, $kind:expr, $nc:expr, $func_idx:expr, $nargs:expr, $returns:expr, $base:expr,
     $ctx:ident, $thread:ident, $registers:ident, $ip:ident, $ds:ident, $frame:ident, $closure:ident) => {{
        let func_idx = $func_idx;
        let returns = $returns;
        let (args_base, argc) = native_args!(func_idx, $nargs, $thread);
        let r = match $kind {
            NativeKind::Action(f) if !$async => {
                invoke_action($ctx, $thread, f, $nc, args_base, argc)
            }
            NativeKind::Async(f) if $async => {
                crate::vm::async_native::invoke_async($ctx, $thread, f, $nc, args_base, argc)
            }
            _ => {
                debug_assert!(false, "call_action! of another kind of native");
                unsafe { std::hint::unreachable_unchecked() }
            }
        };
        let action = match r {
            Ok(a) => a,
            Err(err) => {
                unsafe { (*$frame).pc = $ip };
                throw!(err);
            }
        };
        if let crate::vm::native::CallbackAction::Return = action {
            land_native!(args_base, func_idx, returns, $base, $thread, $registers);
        }
        unsafe { (*$frame).pc = $ip };
        let f = unsafe { $thread.stack[func_idx].get_function().unwrap_unchecked() };
        if $async && let crate::vm::native::CallbackAction::Async = action {
            // The future's first poll, from the frame it gets, and the call
            // of a Lua function it asks for, without `run_natives` either.
            let cont = crate::vm::async_native::async_cont;
            let ret = call_ret(returns);
            let (no, cont_ok) = (
                crate::vm::native::Protect::No,
                crate::vm::native::OnOk::Cont,
            );
            push_native_frame($thread, f, args_base, 0, no, cont_ok, cont, ret);
            let r = cont($ctx, $nc, Stack::new($thread, args_base), Ok(()));
            match r {
                Ok(crate::vm::native::CallbackAction::CallThen {
                    at,
                    protect,
                    ok,
                    cont,
                }) if !$thread.native_overflowed()
                    && let Some(callee) = native_calls_lua($thread, args_base + at as usize, 1) =>
                {
                    let nf = native_frame(f, args_base, at, protect, ok, cont, ret);
                    unsafe { *$thread.top_lua_ptr() = nf };
                    let new_base = args_base + at as usize + 1;
                    enter_from_native($thread, callee, new_base, ret_native);
                    ($frame, $closure) = (unsafe { $thread.top_lua_ptr() }, callee);
                    $ip = callee.code;
                    $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
                    dispatch!();
                }
                Ok(crate::vm::native::CallbackAction::Return) if !$thread.native_overflowed() => {
                    $thread.pop_lua();
                    land_native!(args_base, func_idx, returns, $base, $thread, $registers);
                }
                r => {
                    let r = match r {
                        Ok(_) if $thread.native_overflowed() => Err(native_overflow($ctx)),
                        r => r,
                    };
                    let step = run_natives!(NativeState::Acted {
                        r,
                        framed: true,
                        f,
                        base: args_base,
                        ret,
                    },);
                    native_step!(step);
                }
            }
        }
        // The common shape, a native calling a Lua function, without
        // `run_natives`' generality.
        if let crate::vm::native::CallbackAction::CallThen {
            at,
            protect,
            ok,
            cont,
        } = action
            && let Some(callee) = native_calls_lua($thread, args_base + at as usize, 2)
        {
            push_native_frame(
                $thread,
                f,
                args_base,
                at,
                protect,
                ok,
                cont,
                call_ret(returns),
            );
            let new_base = args_base + at as usize + 1;
            enter_from_native($thread, callee, new_base, ret_native);
            ($frame, $closure) = (unsafe { $thread.top_lua_ptr() }, callee);
            $ip = callee.code;
            $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        let step = run_natives!(NativeState::Acted {
            r: Ok(action),
            framed: false,
            f,
            base: args_base,
            ret: call_ret(returns),
        },);
        native_step!(step);
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
    (@inner $grow:literal, $callee:expr, $func_idx:expr, $nargs:expr, $ret:expr,
     $thread:ident, $registers:ident, $ip:ident, $frame:ident, $closure:ident) => {{
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
        let num_extras = if std::hint::likely($nargs > callee.fixed_arity) {
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
        let ret: Handler = $ret;
        if $grow {
            $thread.push_lua(LuaFrame {
                closure: callee,
                pc: $ip,
                ret,
                base: new_base as u32,
                num_extras,
                flags: 0,
            });
            ($frame, $closure) = top_frame($thread);
        } else {
            // The new frame sits one slot above the running one, and the
            // closure is already in hand: no trip through `thread.frames` to
            // find either.
            $frame = unsafe {
                $thread.push_lua_above($frame, callee, $ip, ret, new_base as u32, num_extras)
            };
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
                if std::hint::unlikely(thread.call_limit < needed) {
                    tail!(op_call_grow);
                }
                let callee = unsafe { LuaFn::from_function_unchecked(f) };
                call_lua!(
                    nogrow,
                    callee,
                    func_idx,
                    nargs,
                    call_ret_at(handlers, returns),
                    thread,
                    registers,
                    ip,
                    frame,
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
    thread.grow_frames_to_stack();
    // Both stacks may have moved: rebind the frame pointer and the register window.
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
    let NativeKind::Plain(f) = nc.function else {
        debug_assert!(false, "op_call_native entry of an action native");
        unsafe { std::hint::unreachable_unchecked() }
    };
    call_plain!(
        f, nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, frame
    );
}

/// CALL or TAILCALL of an action native: the default `NativeClosure::entry`
/// of one.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn op_call_action<'gc>(
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
    call_action!(
        false,
        nc.function,
        nc,
        func_idx,
        nargs,
        returns,
        base,
        ctx,
        thread,
        registers,
        ip,
        ds,
        frame,
        closure
    );
}

/// CALL or TAILCALL of an async native: the default `NativeClosure::entry`
/// of one.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn op_call_async<'gc>(
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
    call_action!(
        true,
        nc.function,
        nc,
        func_idx,
        nargs,
        returns,
        base,
        ctx,
        thread,
        registers,
        ip,
        ds,
        frame,
        closure
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
                // Not RETURN1's handler: the frame may have varargs or something
                // to close.
                tail!(
                    op_return,
                    Instruction::ret(crate::instruction::Reg(func), 2)
                );
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
        tail!(op_call_action);
    }
    let Some(t) = reg!(func + 1).get_table() else {
        tail!(op_call_action);
    };
    if t.shape().has_mm(MetamethodBits::PAIRS) {
        tail!(op_call_action);
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
                grow,
                callee,
                func_idx,
                nargs,
                call_ret_at(handlers, returns),
                thread,
                registers,
                ip,
                frame,
                closure
            );
        }
        CallTarget::Native(nc) => match nc.function {
            NativeKind::Plain(f) => {
                call_plain!(
                    f, nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, frame
                );
            }
            kind @ NativeKind::Action(_) => {
                call_action!(
                    false, kind, nc, func_idx, nargs, returns, base, ctx, thread, registers, ip,
                    ds, frame, closure
                );
            }
            kind @ NativeKind::Async(_) => {
                call_action!(
                    true, kind, nc, func_idx, nargs, returns, base, ctx, thread, registers, ip, ds,
                    frame, closure
                );
            }
        },
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

/// TAILCALL of a native, reached through its kind's generic entry or, after a
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
    let r = match nc.function {
        // A plain native's result comes back in a register: keep its path
        // free of `CallbackAction`.
        NativeKind::Plain(f) => match invoke_plain(ctx, thread, f, nc, args_base, argc) {
            Ok(()) => Ok(crate::vm::native::CallbackAction::Return),
            Err(e) => Err(e),
        },
        _ => invoke_native(ctx, thread, nc, args_base, argc),
    };
    let action = match r {
        Ok(a) => a,
        Err(err) => {
            // The message still names the tailcalling Lua frame (a native
            // never really tail calls in the reference either), so locate it
            // before popping that frame and installing the unwind marker.
            save_pc(thread, ip);
            let err = crate::vm::debug::locate(ctx, thread, err);
            close_upvalues(ctx.mutation(), thread, base);
            debug_assert!(!has_tbc_from(thread, base));
            let popped = thread.pop_lua();
            restore_protected(thread, &popped);
            throw!(err);
        }
    };
    match action {
        crate::vm::native::CallbackAction::Return => {
            // The results sit at `func + 1` up to `top`, as a RETURN of them
            // would find them; past its 8-bit count, MULTRET reads `top`.
            let retc = thread.top - args_base;
            registers = unsafe { thread.stack.as_mut_ptr().add(base) };
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
            let (orig_func, mut ret) = {
                let f = unsafe { &*frame };
                (f.base() - 1 - f.num_extras as usize, f.ret)
            };
            // Not through `registers`: the native may have grown the stack.
            let f = unsafe { thread.stack[func_idx].get_function().unwrap_unchecked() };
            close_upvalues(ctx.mutation(), thread, base);
            debug_assert!(!has_tbc_from(thread, base));
            let popped = thread.pop_lua();
            if restore_protected(thread, &popped) {
                ret = ret_native;
            }
            let top = thread.top;
            thread.stack.copy_within(args_base..top, orig_func + 1);
            thread.set_top_unchecked(orig_func + 1 + (top - args_base));
            let step = run_natives!(NativeState::Acted {
                r: Ok(action),
                framed: false,
                f,
                base: orig_func + 1,
                ret,
            },);
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
/// and hand them to its continuation. `fixed`: the frame has no varargs
/// below it.
macro_rules! return_to_ret {
    (@pop $ret:expr, $func_slot:expr, $nret:expr, $values:expr, $ctx:ident, $thread:ident,
     $handlers:ident, $ds:ident, $frame:ident, $closure:ident) => {{
        let (__ret, __func_slot) = ($ret, $func_slot);
        unsafe { $thread.frames.set_top($frame.sub(1)) };
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
    (fixed, $nret:expr, $values:expr, $ctx:ident, $thread:ident, $registers:ident,
     $handlers:ident, $ds:ident, $frame:ident, $closure:ident) => {{
        debug_assert_eq!(unsafe { (*$frame).num_extras }, 0);
        let __ret = unsafe { (*$frame).ret };
        return_to_ret!(@pop __ret, unsafe { $registers.sub(1) }, $nret, $values, $ctx, $thread, $handlers,
                       $ds, $frame, $closure)
    }};
    ($nret:expr, $values:expr, $ctx:ident, $thread:ident, $registers:ident, $handlers:ident,
     $ds:ident, $frame:ident, $closure:ident) => {{
        let (__ret, __func_slot) = unsafe {
            let f = &*$frame;
            (f.ret, $registers.sub(1 + f.num_extras as usize))
        };
        return_to_ret!(@pop __ret, __func_slot, $nret, $values, $ctx, $thread, $handlers, $ds,
                       $frame, $closure)
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

/// return, from a function the assembler found has nothing to close and no
/// varargs
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return0<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, _ip, handlers, ds, frame, closure }
    let _ = instruction;
    debug_assert_eq!(unsafe { (*frame).flags }, 0);
    return_to_ret!(
        fixed, 0, registers, ctx, thread, registers, handlers, ds, frame, closure
    );
}

/// return R[value], as RETURN0
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return1<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    frame: *mut LuaFrame<'gc>,
    closure: LuaFn<'gc>,
) -> Exit {
    helpers! { instruction, ctx, thread, registers, _ip, handlers, ds, frame, closure }
    debug_assert_eq!(unsafe { (*frame).flags }, 0);
    let values = unsafe { registers.add(instruction.a() as usize) };
    return_to_ret!(
        fixed, 1, values, ctx, thread, registers, handlers, ds, frame, closure
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
                throw!(err);
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

    let mut self_slot = None;
    let func = Function::new_lua(ctx.mutation(), proto, |slots| {
        for (i, desc) in proto.upvalue_desc.iter().enumerate() {
            let slot = match desc.source {
                // A local function's capture of itself: its register is
                // only written below.
                UpvalSource::ParentLocal(idx) if desc.by_value && idx == dst => {
                    self_slot = Some(i);
                    UpvalueSlot {
                        value: Value::nil(),
                    }
                }
                UpvalSource::ParentLocal(idx) if desc.by_value => UpvalueSlot { value: reg!(idx) },
                UpvalSource::ParentLocal(idx) => {
                    let slot = unsafe { thread.stack.as_mut_ptr().add(base + idx as usize) };
                    // Sorted by slot, so this frame's are at the end.
                    let open = &thread.open_upvalues;
                    let below = open.iter().rposition(|uv| uv.slot() <= slot);
                    let cell = match below {
                        Some(i) if open[i].slot() == slot => open[i],
                        _ => {
                            let uv = UpvalueCell::new_open(ctx.mutation(), thread.handle(), slot);
                            let at = below.map_or(0, |i| i + 1);
                            thread.open_upvalues.insert(at, uv);
                            unsafe { (*frame).flags |= frame_flags::OPEN_UPVALUES };
                            uv
                        }
                    };
                    UpvalueSlot { cell }
                }
                UpvalSource::ParentUpvalue(idx) => upvalue!(idx),
            };
            unsafe { slots.add(i).write(slot) };
        }
    });
    if let Some(i) = self_slot {
        // The fresh closure pointing at itself needs no barrier.
        unsafe {
            let f = LuaFn::from_function_unchecked(func);
            (*f.upvalue_ptr().add(i)).value = Value::function(func);
        }
    }
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

/// Present a native the window `args_base..args_base + argc` as its
/// arguments. The backing vec is grown to cover it but never shrunk, so a
/// native cannot truncate the shared stack below an outer frame's registers;
/// it signals its result count through the logical top instead. Dead scratch
/// it leaves above that top is harmless: `ThreadState`'s trace covers only
/// the live high-water.
#[inline(always)]
fn native_window<'gc, 'a>(
    thread: &'a mut ThreadState<'gc>,
    args_base: usize,
    argc: usize,
) -> Stack<'gc, 'a> {
    let end = args_base + argc;
    thread.ensure_slots(end);
    thread.top = end;
    Stack::new(thread, args_base)
}

/// Run plain native `f` of `nc` on its arguments; its results are then at
/// `args_base..top`.
#[inline(always)]
pub(crate) fn invoke_plain<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    f: crate::env::NativeFn,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<(), crate::env::Error<'gc>> {
    let r = f(ctx, nc, native_window(thread, args_base, argc));
    if r.is_ok() && thread.native_overflowed() {
        return Err(native_overflow(ctx));
    }
    r
}

/// Run action native `f` of `nc` on its arguments.
#[inline(always)]
pub(crate) fn invoke_action<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    f: crate::env::ActionFn,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<crate::vm::native::CallbackAction, crate::env::Error<'gc>> {
    let r = f(ctx, nc, native_window(thread, args_base, argc));
    if r.is_ok() && thread.native_overflowed() {
        return Err(native_overflow(ctx));
    }
    r
}

/// Run native `nc` of either kind on its arguments.
#[inline]
pub(crate) fn invoke_native<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<crate::vm::native::CallbackAction, crate::env::Error<'gc>> {
    match nc.function {
        NativeKind::Plain(f) => invoke_plain(ctx, thread, f, nc, args_base, argc)
            .map(|()| crate::vm::native::CallbackAction::Return),
        NativeKind::Action(f) => invoke_action(ctx, thread, f, nc, args_base, argc),
        NativeKind::Async(f) => {
            crate::vm::async_native::invoke_async(ctx, thread, f, nc, args_base, argc)
        }
    }
}

/// A native pushed without `Stack::check_stack`.
#[cold]
#[inline(never)]
pub(crate) fn native_overflow(ctx: Context<'_>) -> crate::env::Error<'_> {
    crate::env::Error::from_str(ctx, "stack overflow")
}

/// Whether any open upvalue points at stack index `base` or above.
#[inline(always)]
fn frame_has_open_upvalues<'gc>(thread: &ThreadState<'gc>, base: usize) -> bool {
    thread
        .open_upvalues
        .last()
        .is_some_and(|uv| uv.slot().addr() >= thread.stack.as_ptr().wrapping_add(base).addr())
}

/// Close all open upvalues pointing at stack indices >= `start_idx`.
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
    let level = thread.stack.as_ptr().wrapping_add(start_idx).addr();
    // Sorted by slot, so the ones to close are the tail.
    while let Some(&uv) = thread.open_upvalues.last() {
        if uv.slot().addr() < level {
            break;
        }
        UpvalueCell::close(uv, mc);
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
            use crate::vm::native::CallbackAction;
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
                    let step = run_natives!(NativeState::Acted {
                        r,
                        framed: false,
                        f,
                        base: new_base,
                        ret,
                    },);
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
        r: Result<crate::vm::native::CallbackAction, crate::env::Error<'gc>>,
        framed: bool,
        f: Function<'gc>,
        base: usize,
        ret: Handler,
    },
    /// Run the top (native) frame's continuation.
    Resume(Result<(), crate::env::Error<'gc>>),
    /// Raise the error, from the frames as they are.
    Raise(crate::env::Error<'gc>),
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
    /// The native frame on top waits on the host.
    Pending,
}

/// `LUAI_MAXCCALLS`: threads resumed inside one another.
pub(crate) const MAX_RESUME_DEPTH: u16 = 200;

/// Drive natives that call: push a native's frame, make its call, and run
/// its continuation on the results, for as long as no Lua code has to run
/// and the native doesn't return. With `switch`, a resume or yield switches
/// `thread` to the other coroutine here; the executor (which can't follow)
/// gets them as suspensions otherwise.
#[inline(always)]
pub(crate) fn run_natives<'gc>(
    ctx: Context<'gc>,
    thread: &mut &mut ThreadState<'gc>,
    switch: bool,
    state: NativeState<'gc>,
) -> NativeStep {
    // Unpacked here, at the call site: passed whole, the state went through
    // memory, stored piecewise and read back wider, stalling on store
    // forwarding.
    match state {
        NativeState::Acted {
            r,
            framed,
            f,
            base,
            ret,
        } => drive_natives(
            ctx,
            thread,
            switch,
            Phase::Act,
            r,
            Ok(()),
            framed,
            f,
            base,
            ret,
        ),
        NativeState::Resume(status) => drive_natives(
            ctx,
            thread,
            switch,
            Phase::Resume,
            Ok(crate::vm::native::CallbackAction::Return),
            status,
            true,
            ctx.next_fn(),
            0,
            ret_native,
        ),
        NativeState::Raise(err) => drive_natives(
            ctx,
            thread,
            switch,
            Phase::Raise,
            Ok(crate::vm::native::CallbackAction::Return),
            Err(err),
            true,
            ctx.next_fn(),
            0,
            ret_native,
        ),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Act,
    Resume,
    Start,
    /// Unwind `status`'s error to the frame that catches it.
    Raise,
}

/// [`run_natives`], starting in `phase` with its state in scalars.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn drive_natives<'gc>(
    ctx: Context<'gc>,
    thread: &mut &mut ThreadState<'gc>,
    switch: bool,
    mut phase: Phase,
    mut r: Result<crate::vm::native::CallbackAction, crate::env::Error<'gc>>,
    mut status: Result<(), crate::env::Error<'gc>>,
    mut framed: bool,
    mut f: Function<'gc>,
    mut base: usize,
    mut ret: Handler,
) -> NativeStep {
    use crate::vm::native::{CallbackAction, NativeCont, OnOk, Protect};
    let mut slot = 0;
    loop {
        match phase {
            Phase::Resume => {
                let frame = unsafe { thread.frames.last_mut().unwrap_unchecked() };
                debug_assert!(frame.is_native());
                if status.is_ok()
                    && frame.flags & (frame_flags::PASS | frame_flags::PASS_TRUE) != 0
                    && thread.top < thread.stack_limit
                {
                    // What `cont` would do: the results at the call's slot,
                    // after a `true` in the slot below with `PASS_TRUE`.
                    let (base, ret) = (frame.base(), frame.ret);
                    let mut values = base + frame.num_extras as usize;
                    if frame.flags & frame_flags::PASS_TRUE != 0 {
                        values -= 1;
                        thread.stack[values] = Value::boolean(true);
                    }
                    thread.pop_lua();
                    return NativeStep::Return {
                        ret,
                        func_slot: base - 1,
                        values,
                    };
                }
                // Errors the continuation itself raises unwind past it.
                frame.flags &= !(frame_flags::PROTECTED | frame_flags::HANDLER);
                (f, base, ret) = (frame.closure.function(), frame.base(), frame.ret);
                let cont: NativeCont = unsafe { std::mem::transmute(frame.pc) };
                let nc = unsafe { f.as_native().unwrap_unchecked() };
                let st = std::mem::replace(&mut status, Ok(()));
                r = cont(ctx, nc, Stack::new(thread, base), st);
                if r.is_ok() && thread.native_overflowed() {
                    r = Err(native_overflow(ctx));
                }
                framed = true;
                phase = Phase::Act;
            }
            Phase::Act => match std::mem::replace(&mut r, Ok(CallbackAction::Return)) {
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
                Ok(CallbackAction::CallThen {
                    at,
                    protect,
                    ok,
                    cont,
                }) => {
                    let nf = native_frame(f, base, at, protect, ok, cont, ret);
                    set_native_frame(thread, framed, nf);
                    (slot, ret) = (base + at as usize, ret_native);
                    phase = Phase::Start;
                }
                Ok(CallbackAction::Resume { at, ok, cont }) => {
                    // The resumer waits in a frame of its own, protected so
                    // the error that kills the coroutine comes back to `cont`.
                    let nf = native_frame(f, base, at, Protect::Errors, ok, cont, ret);
                    set_native_frame(thread, framed, nf);
                    let at = base + at as usize;
                    let co = thread.stack[at]
                        .get_thread()
                        .expect("resume of a non-thread");
                    if thread.resume_depth + 1 >= MAX_RESUME_DEPTH {
                        // On the resumer, leaving the coroutine untouched
                        // (`lua_resume`'s `resume_error`).
                        let msg = LuaString::new(ctx, b"C stack overflow");
                        status = Err(crate::env::Error::new(ctx, Value::string(msg)));
                        phase = Phase::Raise;
                        continue;
                    }
                    if !switch {
                        thread.pending_action = Some(PendingAction {
                            kind: PendingKind::Resume(co),
                            call_site: CallSite {
                                bottom: at + 1,
                                func_idx: at,
                                ret: ret_native,
                            },
                        });
                        return NativeStep::Exit;
                    }
                    match resume_into(ctx, thread, co, at + 1) {
                        Some(site) => {
                            return NativeStep::Return {
                                ret: site.ret,
                                func_slot: site.func_idx,
                                values: site.bottom,
                            };
                        }
                        None => {
                            (slot, ret) = (0, ret_coroutine_end);
                            phase = Phase::Start;
                        }
                    }
                }
                Ok(CallbackAction::Yield) => {
                    if framed {
                        ret = thread.frames.pop().map(|f| f.ret).unwrap_or(ret);
                    }
                    let site = CallSite {
                        bottom: base,
                        func_idx: base - 1,
                        ret,
                    };
                    if !(switch && yield_to_resumer(ctx, thread, site)) {
                        thread.pending_action = Some(PendingAction {
                            kind: PendingKind::Yield,
                            call_site: site,
                        });
                        return NativeStep::Exit;
                    }
                    phase = Phase::Resume;
                }
                Ok(CallbackAction::YieldThen { at, cont }) => {
                    let nf = native_frame(f, base, at, Protect::No, OnOk::Cont, cont, ret);
                    set_native_frame(thread, framed, nf);
                    // The values it resumes with land where the yielded ones
                    // were, for `cont`.
                    let bottom = base + at as usize;
                    let site = CallSite {
                        bottom,
                        func_idx: bottom - 1,
                        ret: ret_native,
                    };
                    if !(switch && yield_to_resumer(ctx, thread, site)) {
                        thread.pending_action = Some(PendingAction {
                            kind: PendingKind::Yield,
                            call_site: site,
                        });
                        return NativeStep::Exit;
                    }
                    phase = Phase::Resume;
                }
                Ok(CallbackAction::Async) => {
                    let cont = crate::vm::async_native::async_cont;
                    let nf = native_frame(f, base, 0, Protect::No, OnOk::Cont, cont, ret);
                    set_native_frame(thread, framed, nf);
                    phase = Phase::Resume;
                }
                Ok(CallbackAction::Pending) => {
                    debug_assert!(framed);
                    return NativeStep::Pending;
                }
                Err(err) => {
                    // Raised at the native's caller, as from a native that
                    // has no frame.
                    if framed {
                        thread.pop_lua();
                    }
                    status = Err(err);
                    phase = Phase::Raise;
                }
            },
            Phase::Raise => {
                let err = unsafe { std::mem::replace(&mut status, Ok(())).unwrap_err_unchecked() };
                let err = crate::vm::debug::locate(ctx, thread, err);
                match crate::vm::unwind::unwind(ctx, thread, err, switch) {
                    Unwound::Catch(err) => {
                        status = Err(err);
                        phase = Phase::Resume;
                    }
                    Unwound::Call(at) => {
                        (slot, ret) = (at, ret_native);
                        phase = Phase::Start;
                    }
                    Unwound::Exit => return NativeStep::Exit,
                }
            }
            Phase::Start => {
                let new_base = slot + 1;
                let nargs = thread.top - new_base;
                let fv = thread.stack[slot];
                match fv.get_function().map(|f| (f, f.inner().as_ref())) {
                    Some((callee, FunctionKind::Lua(_))) => {
                        let callee = unsafe { LuaFn::from_function_unchecked(callee) };
                        if !thread.ensure_frame_slots(new_base + callee.max_stack_size as usize) {
                            status = Err(crate::vm::debug::stack_overflow(ctx, thread));
                            phase = Phase::Raise;
                            continue;
                        }
                        enter_from_native(thread, callee, new_base, ret);
                        return NativeStep::EnterLua;
                    }
                    Some((callee, FunctionKind::Native(nc))) => {
                        r = invoke_native(ctx, thread, nc, new_base, nargs);
                        if !matches!(r, Ok(CallbackAction::Return)) {
                            (framed, f, base) = (false, callee, new_base);
                            phase = Phase::Act;
                        } else if std::ptr::fn_addr_eq(ret, ret_native as Handler) {
                            let nret = thread.top - new_base;
                            let stack = thread.stack.as_mut_ptr();
                            unsafe { copy_values(stack.add(slot), stack.add(new_base), nret) };
                            thread.set_top_unchecked(slot + nret);
                            phase = Phase::Resume;
                        } else {
                            return NativeStep::Return {
                                ret,
                                func_slot: slot,
                                values: new_base,
                            };
                        }
                    }
                    None => {
                        // MULTRET: the count is at `top`, where the chain
                        // keeps it.
                        if let Err(e) = resolve_call_chain(ctx, thread, slot, 0) {
                            let msg = crate::vm::debug::op_error_message(ctx, thread, e);
                            status = Err(crate::env::Error::from_str(ctx, &msg));
                            phase = Phase::Raise;
                        }
                    }
                }
            }
        }
    }
}

/// Whether the top frame of `ts` is a native's waiting for a coroutine it
/// resumed in dispatch, as opposed to the executor's `WaitThread`.
fn waits_in_dispatch(ts: &ThreadState<'_>) -> bool {
    ts.top_lua().is_some_and(|f| f.is_native())
}

/// Switch `thread` to the suspended coroutine `co`, resumed with
/// `thread.stack[args..top]`. Returns the call site its last yield left
/// for the values, or `None` on a first resume, its body then at slot 0.
fn resume_into<'gc>(
    ctx: Context<'gc>,
    thread: &mut &mut ThreadState<'gc>,
    co: Thread<'gc>,
    args: usize,
) -> Option<CallSite> {
    let n = thread.top - args;
    // SAFETY: `co` is suspended, so nothing else uses its state.
    let cs = unsafe { co.state_mut(ctx.mutation()) };
    let site = if let Some(ExecKind::Start(_)) = cs.top_exec() {
        let Some(ExecKind::Start(f)) = cs.pop_exec() else {
            unreachable!()
        };
        cs.discard_above(0);
        cs.ensure_slots(1 + n);
        cs.stack[0] = f;
        unsafe {
            copy_values(
                cs.stack.as_mut_ptr().add(1),
                thread.stack.as_ptr().add(args),
                n,
            )
        };
        cs.top = 1 + n;
        None
    } else {
        let y = cs
            .yield_bottom
            .take()
            .expect("resumed a coroutine that didn't yield");
        cs.ensure_slots(y.bottom + n);
        unsafe {
            copy_values(
                cs.stack.as_mut_ptr().add(y.bottom),
                thread.stack.as_ptr().add(args),
                n,
            )
        };
        cs.top = y.bottom + n;
        Some(y)
    };
    thread.set_top_unchecked(args - 1);
    thread.status = ThreadStatus::Normal;
    cs.status = ThreadStatus::Normal;
    cs.resumer = Some(thread.handle());
    cs.resume_depth = thread.resume_depth + 1;
    *thread = cs;
    site
}

/// Hand the values a coroutine yields, `thread.stack[site.bottom..top]`, to
/// a resumer waiting in dispatch and switch `thread` to it; `false` (having
/// done nothing) when the executor must take the yield instead.
fn yield_to_resumer<'gc>(
    ctx: Context<'gc>,
    thread: &mut &mut ThreadState<'gc>,
    site: CallSite,
) -> bool {
    if thread.main || thread.no_yield {
        return false;
    }
    let Some(r) = thread.resumer else {
        return false;
    };
    // SAFETY: the resumer waits, so nothing else uses its state.
    let rs = unsafe { r.state_mut(ctx.mutation()) };
    if !waits_in_dispatch(rs) {
        return false;
    }
    hand_back(thread, rs, site.bottom);
    thread.yield_bottom = Some(site);
    thread.status = ThreadStatus::Suspended;
    thread.set_top_unchecked(site.bottom);
    *thread = rs;
    true
}

/// Move `from.stack[values..top]` into the slot of the native frame waiting
/// on top of `to`, and make `to` the running thread.
fn hand_back<'gc>(from: &mut ThreadState<'gc>, to: &mut ThreadState<'gc>, values: usize) {
    let n = from.top - values;
    let nf = unsafe { to.frames.last().unwrap_unchecked() };
    let slot = nf.base() + nf.num_extras as usize;
    to.ensure_slots(slot + n);
    unsafe {
        copy_values(
            to.stack.as_mut_ptr().add(slot),
            from.stack.as_ptr().add(values),
            n,
        )
    };
    to.set_top_unchecked(slot + n);
    to.status = ThreadStatus::Normal;
    from.resumer = None;
}

/// Continuation of a coroutine's body: the coroutine is dead, and its
/// results go to the resumer, in dispatch when it waits there.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_coroutine_end<'gc>(
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
    let resumer = thread
        .resumer
        .map(|r| unsafe { r.state_mut(ctx.mutation()) })
        .filter(|rs| waits_in_dispatch(rs));
    let Some(rs) = resumer else {
        tail!(ret_exit);
    };
    let values = unsafe { values.offset_from_unsigned(thread.stack.as_ptr()) };
    thread.set_top_unchecked(values + nret);
    hand_back(thread, rs, values);
    thread.discard_above(0);
    thread.status = ThreadStatus::Result { bottom: 0 };
    thread = rs;
    ds.current = Some(thread.handle());
    let step = run_natives!(NativeState::Resume(Ok(())));
    native_step!(step);
}

/// Resume `$co` from the CALL running this entry, its arguments from
/// `R[func + $skip]` on, the caller waiting in a frame for native `$f` that
/// takes the results by `$ok` and gets errors in `$cont`: move them to where
/// `$co` last yielded and continue its call site there, LuaJIT Remake's
/// `CoroSwitch`. Any case the switch doesn't cover (a first resume, a
/// non-suspended coroutine, no room) goes to the builtin.
macro_rules! resume_switch {
    ($co:expr, $skip:literal, $f:expr, $ok:expr, $cont:expr,
     $ctx:ident, $thread:ident, $registers:ident, $ip:ident, $handlers:ident, $ds:ident,
     $frame:ident, $closure:ident, $instruction:ident) => {{
        let co: Thread<'gc> = $co;
        if co.ptr_eq($thread.handle())
            || co.peer_status() != ThreadStatus::Suspended
            || $thread.resume_depth + 1 >= MAX_RESUME_DEPTH
            || $thread.frames_full()
        {
            tail!(op_call_action);
        }
        let (func, nargs, returns) = $instruction.abc();
        // SAFETY: `co` is suspended, so nothing else uses its state.
        let Some(cs) = (unsafe { co.state_mut_if_clean($ctx.mutation()) }) else {
            tail!(op_call_action);
        };
        // A yield the executor took resumes through it.
        let Some(y) = cs.yield_bottom.filter(|_| cs.top_is_lua()) else {
            tail!(op_call_action);
        };
        let base = unsafe { (*$frame).base() };
        let args = base + func as usize + $skip;
        let n = if nargs == 0 {
            $thread.top - args
        } else {
            nargs as usize - $skip
        };
        if y.bottom + n > cs.stack.len() || y.bottom + n > cs.stack_limit {
            tail!(op_call_action);
        }
        unsafe { (*$frame).pc = $ip };
        let nf = native_frame(
            $f,
            base + func as usize + 1,
            0,
            crate::vm::native::Protect::Errors,
            $ok,
            $cont,
            unsafe {
                *$handlers
                    .cast::<Handler>()
                    .add(Op::COUNT + returns as usize)
            },
        );
        unsafe { $thread.push_unchecked(nf) };
        $thread.set_top_unchecked(base + func as usize + 1);
        $thread.status = ThreadStatus::Normal;
        cs.yield_bottom = None;
        cs.status = ThreadStatus::Normal;
        cs.resumer = Some($thread.handle());
        cs.resume_depth = $thread.resume_depth + 1;
        // Copied rather than passed by pointer: continuations take values
        // in their own thread's stack.
        let values = unsafe { cs.stack.as_mut_ptr().add(y.bottom) };
        unsafe { copy_values(values, $thread.stack.as_ptr().add(args), n) };
        $thread = cs;
        $ds.current = Some(co);
        let (frame, closure) = top_frame($thread);
        let func_slot = unsafe { $thread.stack.as_mut_ptr().add(y.func_idx) };
        become (y.ret)(
            Instruction::from_raw(n as u64),
            $ctx,
            $thread,
            values,
            func_slot as *const Instruction,
            $handlers,
            $ds,
            frame,
            closure,
        );
    }};
}

/// The entry of a `coroutine.wrap` function; see [`resume_switch`].
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_wrap<'gc>(
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
    if instruction.op() == Op::TAILCALL {
        tail!(op_call_action);
    }
    let f = reg!(instruction.a()).get_function();
    let f = unsafe { f.unwrap_unchecked() };
    let nc = unsafe { &*ds.native };
    let co = unsafe {
        nc.upvalues()
            .get_unchecked(0)
            .get_thread()
            .unwrap_unchecked()
    };
    resume_switch!(
        co,
        1,
        f,
        crate::vm::native::OnOk::Return,
        crate::builtin::wrap_cont,
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds,
        frame,
        closure,
        instruction
    );
}

/// The entry of `coroutine.resume`; see [`resume_switch`].
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_resume<'gc>(
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
    if instruction.op() == Op::TAILCALL {
        tail!(op_call_action);
    }
    let (func, nargs) = (instruction.a(), instruction.b());
    let f = reg!(func).get_function();
    let f = unsafe { f.unwrap_unchecked() };
    // Without the thread argument the builtin raises.
    let co = if nargs != 1 {
        reg!(func + 1).get_thread()
    } else {
        None
    };
    let Some(co) = co else {
        tail!(op_call_action);
    };
    resume_switch!(
        co,
        2,
        f,
        crate::vm::native::OnOk::ReturnTrue,
        crate::builtin::resume_cont,
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds,
        frame,
        closure,
        instruction
    );
}

/// The entry of `coroutine.yield`: with the resumer waiting in dispatch on
/// a frame that takes the values as they are (`PASS`, `PASS_TRUE`), move
/// them into that frame's window and continue its call site, as
/// [`resume_switch`] does the other way. Anything else goes to the builtin.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_yield<'gc>(
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
    let Some(r) = thread.resumer else {
        tail!(op_call_action);
    };
    if instruction.op() == Op::TAILCALL || thread.no_yield {
        tail!(op_call_action);
    }
    let (func, nargs, returns) = instruction.abc();
    // SAFETY: the resumer waits, so nothing else uses its state.
    let Some(rs) = (unsafe { r.state_mut_if_clean(ctx.mutation()) }) else {
        tail!(op_call_action);
    };
    let nf = match rs.top_lua() {
        Some(nf) if nf.flags & (frame_flags::PASS | frame_flags::PASS_TRUE) != 0 => *nf,
        _ => tail!(op_call_action),
    };
    let func_idx = unsafe { (*frame).base() } + func as usize;
    let n = if nargs == 0 {
        thread.top - (func_idx + 1)
    } else {
        nargs as usize - 1
    };
    let pass_true = nf.flags & frame_flags::PASS_TRUE != 0;
    // Where `hand_back` puts them; `resume` checks its stack.
    let slot = nf.base() + nf.num_extras as usize;
    if slot + n > rs.stack.len() || slot + n >= rs.stack_limit {
        tail!(op_call_action);
    }
    unsafe { (*frame).pc = ip };
    thread.yield_bottom = Some(CallSite {
        bottom: func_idx + 1,
        func_idx,
        ret: call_ret(returns),
    });
    thread.status = ThreadStatus::Suspended;
    thread.resumer = None;
    thread.set_top_unchecked(func_idx + 1);
    let stack = rs.stack.as_mut_ptr();
    unsafe { copy_values(stack.add(slot), thread.stack.as_ptr().add(func_idx + 1), n) };
    // With `PASS_TRUE`, the `true` goes in the slot below, the native's own
    // or its function slot.
    let values = unsafe { stack.add(slot - pass_true as usize) };
    if pass_true {
        unsafe { values.write(Value::boolean(true)) };
    }
    let nret = n + pass_true as usize;
    rs.set_top_unchecked(slot + n);
    rs.status = ThreadStatus::Normal;
    unsafe { rs.frames.set_top(rs.frames.top_ptr().sub(1)) };
    thread = rs;
    ds.current = Some(r);
    let frame = thread.frames.top_ptr();
    let func_slot = unsafe { thread.stack.as_mut_ptr().add(nf.base() - 1) };
    become (nf.ret)(
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

/// Continuation of the call a `pcall` (`$k` = 1) or `xpcall` (2) entry made
/// without a frame of its own: `true` and the results, to the call site of
/// the `pcall`, `$k` slots below the finished call's. As LuaJIT Remake's
/// `OnProtectedCallSuccessReturn`, it also marks the catch point
/// ([`LuaFrame::elided_protect`]).
macro_rules! ret_protected {
    ($name:ident, $k:literal) => {
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
            let (nret, values, func_slot) = ret_args!(instruction, registers, ip);
            // The slot below the results is the finished callee's.
            let values = unsafe { values.sub(1) };
            unsafe { values.write(Value::boolean(true)) };
            // The entry runs only for a CALL in a Lua frame, now on top.
            let call = unsafe { *(*frame).pc.sub(1) };
            let ret = call_ret_at(handlers, call.c());
            become ret(
                Instruction::from_raw(nret as u64 + 1),
                ctx,
                thread,
                values,
                unsafe { func_slot.sub($k) } as *const Instruction,
                handlers,
                ds,
                frame,
                closure,
            );
        }
    };
}

ret_protected!(ret_pcall, 1);
ret_protected!(ret_xpcall, 2);

/// Put back the frame of the `pcall` that called `popped` without one, if
/// one did, for what takes the popped frame's place (a native it tail
/// called, an error) to return to; `false` if none did.
fn restore_protected<'gc>(thread: &mut ThreadState<'gc>, popped: &LuaFrame<'gc>) -> bool {
    let Some(k) = popped.elided_protect() else {
        return false;
    };
    let nf = protected_frame(thread, popped.func_slot() - k, k);
    thread.push_lua(nf);
    true
}

/// The native frame a `pcall` (`k` = 1) or `xpcall` (2) entry left out, for
/// its call at `slot` in the Lua frame on top of `ts`: the unwinder puts it
/// back when an error reaches the call it made.
pub(crate) fn protected_frame<'gc>(ts: &ThreadState<'gc>, slot: usize, k: usize) -> LuaFrame<'gc> {
    use crate::vm::native::{OnOk, Protect};
    let f = ts.stack[slot]
        .get_function()
        .expect("pcall in its call slot");
    let caller = ts.top_lua().expect("pcall's caller");
    let call = unsafe { *caller.pc.sub(1) };
    let (protect, cont): (_, crate::vm::native::NativeCont) = if k == 1 {
        (Protect::Errors, crate::builtin::pcall_cont)
    } else {
        (Protect::Handler, crate::builtin::xpcall_cont)
    };
    native_frame(
        f,
        slot + 1,
        k as u32 - 1,
        protect,
        OnOk::ReturnTrue,
        cont,
        call_ret(call.c()),
    )
}

/// Call the Lua function at `R[func + 1]`, from `R[func + $k]` once `$prep`
/// has run, its arguments above it, with `$ret` taking its results: a
/// `pcall` (`$k` = 1) or `xpcall` (2) that pushes no frame for itself.
/// Other callees and a window that doesn't fit go to the builtin.
macro_rules! protected_call {
    ($k:literal, $ret:expr, $prep:block, $ctx:ident, $thread:ident, $registers:ident,
     $ip:ident, $handlers:ident, $ds:ident, $frame:ident, $closure:ident,
     $instruction:ident) => {{
        let (func, nargs) = ($instruction.a(), $instruction.b());
        let func_idx = unsafe { (*$frame).base() } + func as usize;
        // The callee and, for `xpcall`, the handler are there.
        let present = if nargs == 0 {
            $thread.top - (func_idx + 1)
        } else {
            nargs as usize - 1
        };
        if present < $k {
            tail!(op_call_action);
        }
        let callee = match reg!(func + 1).get_function() {
            Some(f) if matches!(f.inner().as_ref(), FunctionKind::Lua(_)) => unsafe {
                LuaFn::from_function_unchecked(f)
            },
            _ => tail!(op_call_action),
        };
        let callee_idx = func_idx + $k;
        if $thread.call_limit < callee_idx + 1 + callee.max_stack_size as usize {
            tail!(op_call_action);
        }
        $prep
        // A fixed count, shifted down to the callee's arguments.
        let callee_nargs = if nargs == 0 { 0 } else { nargs - $k };
        call_lua!(
            nogrow,
            callee,
            callee_idx,
            callee_nargs,
            $ret,
            $thread,
            $registers,
            $ip,
            $frame,
            $closure
        );
    }};
}

/// The entry of `pcall`; see [`protected_call`].
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_pcall<'gc>(
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
        tail!(op_call_action);
    }
    protected_call!(
        1,
        ret_pcall,
        {},
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds,
        frame,
        closure,
        instruction
    );
}

/// The entry of `xpcall`: [`protected_call`], the handler moved to the slot
/// below the callee, where the unwinder finds it.
#[inline(never)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ff_xpcall<'gc>(
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
    let (func, nargs) = (instruction.a(), instruction.b());
    // A missing or non-function handler is the builtin's error.
    if instruction.op() == Op::TAILCALL || nargs < 3 || reg!(func + 2).get_function().is_none() {
        tail!(op_call_action);
    }
    protected_call!(
        2,
        ret_xpcall,
        {
            let (f, h) = (reg!(func + 1), reg!(func + 2));
            *reg!(ref mut func + 1) = h;
            *reg!(ref mut func + 2) = f;
        },
        ctx,
        thread,
        registers,
        ip,
        handlers,
        ds,
        frame,
        closure,
        instruction
    );
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
    if nf.flags & (frame_flags::PASS | frame_flags::PASS_TRUE) != 0 {
        // What the continuation would do, without copying the results into
        // the native's window first: the slot below them is the finished
        // callee's, free for the `true`.
        let (ret, func_slot) = (nf.ret, unsafe {
            thread.stack.as_mut_ptr().add(nf.base() - 1)
        });
        let (values, nret) = if nf.flags & frame_flags::PASS_TRUE != 0 {
            let v = unsafe { values.sub(1) };
            unsafe { v.write(Value::boolean(true)) };
            (v, nret + 1)
        } else {
            (values, nret)
        };
        unsafe { thread.frames.set_top(thread.frames.top_ptr().sub(1)) };
        become ret(
            Instruction::from_raw(nret as u64),
            ctx,
            thread,
            values,
            func_slot as *const Instruction,
            handlers,
            ds,
            unsafe { frame.sub(1) },
            closure,
        );
    }
    let (base, slot) = (nf.base(), nf.base() + nf.num_extras as usize);
    // The results sit above the slot, in the finished call's window.
    unsafe { copy_values(thread.stack.as_mut_ptr().add(slot), values, nret) };
    thread.set_top_unchecked(slot + nret);
    // As `run_natives` would, with a returning continuation inline.
    nf.flags &= !(frame_flags::PROTECTED | frame_flags::HANDLER);
    let (f, ret) = (nf.closure.function(), nf.ret);
    let cont: crate::vm::native::NativeCont = unsafe { std::mem::transmute(nf.pc) };
    let nc = unsafe { f.as_native().unwrap_unchecked() };
    let r = cont(ctx, nc, Stack::new(thread, base), Ok(()));
    match r {
        Ok(crate::vm::native::CallbackAction::Return) if !thread.native_overflowed() => {
            unsafe { thread.frames.set_top(thread.frames.top_ptr().sub(1)) };
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
        // The next call of a Lua function, from this same frame.
        Ok(crate::vm::native::CallbackAction::CallThen {
            at,
            protect,
            ok,
            cont,
        }) if !thread.native_overflowed()
            && let Some(callee) = native_calls_lua(thread, base + at as usize, 1) =>
        {
            unsafe { *frame = native_frame(f, base, at, protect, ok, cont, ret) };
            let new_base = base + at as usize + 1;
            enter_from_native(thread, callee, new_base, ret_native);
            (frame, closure) = (unsafe { thread.top_lua_ptr() }, callee);
            ip = callee.code;
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        _ => {}
    }
    let r = match r {
        Ok(_) if thread.native_overflowed() => Err(native_overflow(ctx)),
        r => r,
    };
    let step = run_natives!(NativeState::Acted {
        r,
        framed: true,
        f,
        base,
        ret,
    },);
    native_step!(step);
}

/// The Lua function at `slot` when a native's call of it can be entered
/// straight away: room for its window and for `frames` more frames.
#[inline(always)]
fn native_calls_lua<'gc>(
    thread: &ThreadState<'gc>,
    slot: usize,
    frames: usize,
) -> Option<LuaFn<'gc>> {
    let f = thread.stack[slot].get_function()?;
    let FunctionKind::Lua(callee) = f.inner().as_ref() else {
        return None;
    };
    let fits = thread.stack.len() >= slot + 1 + callee.max_stack_size as usize
        && thread.frames.capacity() - thread.frames.len() >= frames;
    fits.then(|| unsafe { LuaFn::from_function_unchecked(f) })
}

/// Push the frame of native `f`, whose window is at `base`, for its
/// `CallThen`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn push_native_frame<'gc>(
    thread: &mut ThreadState<'gc>,
    f: Function<'gc>,
    base: usize,
    at: u32,
    protect: crate::vm::native::Protect,
    ok: crate::vm::native::OnOk,
    cont: crate::vm::native::NativeCont,
    ret: Handler,
) {
    thread.push_lua(native_frame(f, base, at, protect, ok, cont, ret));
}

/// Make `nf` the native's frame: in place of the one it has when `framed`.
#[inline(always)]
fn set_native_frame<'gc>(thread: &mut ThreadState<'gc>, framed: bool, nf: LuaFrame<'gc>) {
    if framed {
        unsafe { *thread.frames.last_mut().unwrap_unchecked() = nf };
    } else {
        thread.push_lua(nf);
    }
}

/// The frame [`push_native_frame`] pushes.
#[inline(always)]
pub(crate) fn native_frame<'gc>(
    f: Function<'gc>,
    base: usize,
    at: u32,
    protect: crate::vm::native::Protect,
    ok: crate::vm::native::OnOk,
    cont: crate::vm::native::NativeCont,
    ret: Handler,
) -> LuaFrame<'gc> {
    use crate::vm::native::{OnOk, Protect};
    let flags = frame_flags::NATIVE
        | match protect {
            Protect::No => 0,
            Protect::Errors => frame_flags::PROTECTED,
            Protect::Handler => frame_flags::PROTECTED | frame_flags::HANDLER,
            Protect::Base => frame_flags::PROTECTED | frame_flags::BASE,
        }
        | match ok {
            OnOk::Cont => 0,
            OnOk::Return => frame_flags::PASS,
            OnOk::ReturnTrue => frame_flags::PASS_TRUE,
        };
    LuaFrame {
        closure: unsafe { LuaFn::native_frame(f) },
        pc: cont as *const Instruction,
        ret,
        base: base as u32,
        num_extras: at as u16,
        flags,
    }
}

/// Push the frame of a native's call of Lua `callee`, its arguments at
/// `new_base` up to `top` and its window known to fit.
#[inline(always)]
fn enter_from_native<'gc>(
    thread: &mut ThreadState<'gc>,
    callee: LuaFn<'gc>,
    new_base: usize,
    ret: Handler,
) {
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
        ret,
        base: new_base as u32,
        num_extras,
        flags: 0,
    });
}

/// `run_thread`'s start with natives to run first. Not a handler: the
/// handlers it enters are called, from `run_thread`'s frame.
#[inline(never)]
fn native_entry<'gc>(
    ctx: Context<'gc>,
    mut thread: &mut ThreadState<'gc>,
    handlers: *const (),
    ds: &mut DispatchState<'gc>,
    state: NativeState<'gc>,
) -> Exit {
    let before: *const ThreadState<'gc> = &*thread;
    let step = run_natives(ctx, &mut thread, true, state);
    if !std::ptr::eq(before, &*thread) {
        ds.current = Some(thread.handle());
    }
    let frame = thread.frames.top_ptr();
    match step {
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
            // Never dereferenced: no continuation reads the closure.
            let closure = unsafe { LuaFn::native_frame(ctx.next_fn()) };
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
        NativeStep::Pending => Exit::Pending,
    }
}

/// The slow path of every binary arithmetic and bitwise opcode, register and
/// immediate forms alike: mixed and boxed numbers, division by zero, and the
/// metamethods. The opcode says which operation it is and where its operands
/// are.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn binop_slow<'gc>(
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
    use Op::*;
    let op = instruction.op();
    let dst = instruction.a();
    let mc = ctx.mutation();
    let (lhs, rhs) = match op {
        ADD | SUB | MUL | MOD | POW | DIV | IDIV | BAND | BOR | BXOR | SHL | SHR => {
            (reg!(instruction.b()), reg!(instruction.c()))
        }
        _ => {
            let (_, src, flipped) = instruction.abc_imm();
            let (v, k) = (reg!(src), instruction.imm_value(mc));
            let swap = matches!(op, RSUBI | RMODI | RPOWI | RDIVI | RIDIVI | RSHLI | RSHRI);
            if swap || flipped { (k, v) } else { (v, k) }
        }
    };
    let (r, bit) = match op {
        ADD | ADDI => (
            num::op_arith_slow::<num::Add>(mc, lhs, rhs),
            MetamethodBits::ADD,
        ),
        SUB | SUBI | RSUBI => (
            num::op_arith_slow::<num::Sub>(mc, lhs, rhs),
            MetamethodBits::SUB,
        ),
        MUL | MULI => (
            num::op_arith_slow::<num::Mul>(mc, lhs, rhs),
            MetamethodBits::MUL,
        ),
        MOD | MODI | RMODI => (
            num::op_arith_slow::<num::Mod>(mc, lhs, rhs),
            MetamethodBits::MOD,
        ),
        POW | POWI | RPOWI => (
            num::op_arith_slow::<num::Pow>(mc, lhs, rhs),
            MetamethodBits::POW,
        ),
        DIV | DIVI | RDIVI => (
            num::op_arith_slow::<num::Div>(mc, lhs, rhs),
            MetamethodBits::DIV,
        ),
        IDIV | IDIVI | RIDIVI => (
            num::op_arith_slow::<num::IDiv>(mc, lhs, rhs),
            MetamethodBits::IDIV,
        ),
        BAND | BANDI => (
            num::op_bit_slow::<num::BAnd>(mc, lhs, rhs),
            MetamethodBits::BAND,
        ),
        BOR | BORI => (
            num::op_bit_slow::<num::BOr>(mc, lhs, rhs),
            MetamethodBits::BOR,
        ),
        BXOR | BXORI => (
            num::op_bit_slow::<num::BXor>(mc, lhs, rhs),
            MetamethodBits::BXOR,
        ),
        SHL | SHLI | RSHLI => (
            num::op_bit_slow::<num::Shl>(mc, lhs, rhs),
            MetamethodBits::SHL,
        ),
        SHR | SHRI | RSHRI => (
            num::op_bit_slow::<num::Shr>(mc, lhs, rhs),
            MetamethodBits::SHR,
        ),
        _ => unreachable!("binop_slow on {op:?}"),
    };
    match r {
        num::SlowNum::Value(v) => {
            *reg!(ref mut dst) = v;
            dispatch!();
        }
        num::SlowNum::ModByZero => raise!(OpError::ModByZero),
        num::SlowNum::DivByZero => raise!(OpError::DivByZero),
        num::SlowNum::NotNumbers => {}
    }
    let meta_fn = binop_metamethod(ctx, lhs, rhs, bit);
    if meta_fn.is_nil() {
        let bitwise = MetamethodBits::BAND | MetamethodBits::BOR | MetamethodBits::BXOR;
        raise!(
            if (bitwise | MetamethodBits::SHL | MetamethodBits::SHR).contains(bit) {
                OpError::Bitwise(lhs, rhs)
            } else {
                OpError::Arith(lhs, rhs)
            }
        );
    }
    call_mm!(ret_store_a, meta_fn, [lhs, rhs]);
}

/// The slow path of the table reads (GETTABUP, GETTABLE, GETFIELD, SELF): a
/// receiver that isn't a table, a cache miss, a key the table lacks, and
/// `__index`. The opcode says where the receiver and key are.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn get_slow<'gc>(
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
    let op = instruction.op().unquickened();
    let dst = instruction.a();
    let recv = match op {
        Op::GETTABUP => upvalue!(value instruction.b()),
        Op::GETTABUP_REF => upvalue!(cell instruction.b()).get(),
        _ => reg!(instruction.b()),
    };
    if op == Op::SELF {
        *reg!(ref mut (dst + 1)) = recv;
    }
    if op == Op::GETTABLE {
        get_slow_body!(
            ctx,
            thread,
            registers,
            ip,
            handlers,
            ds,
            recv,
            reg!(instruction.c()),
            dst
        );
    }
    let k = constant!(instruction.e());
    let Some(t) = recv.get_table() else {
        index_chain_body!(ctx, thread, registers, ip, handlers, ds, recv, k, dst);
    };
    match get_fill_ic(ctx, closure, instruction.d(), unsafe { ip.sub(1) }, t, k) {
        Ok(v) => {
            *reg!(ref mut dst) = v;
            dispatch!();
        }
        Err(from) => index_chain_body!(ctx, thread, registers, ip, handlers, ds, from, k, dst),
    }
}

/// The slow path of the table writes (SETTABUP, SETTABLE, SETFIELD), like
/// [`get_slow`]'s.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn set_slow<'gc>(
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
    let op = instruction.op().unquickened();
    let v = reg!(instruction.a());
    let recv = match op {
        Op::SETTABUP => upvalue!(value instruction.b()),
        Op::SETTABUP_REF => upvalue!(cell instruction.b()).get(),
        _ => reg!(instruction.b()),
    };
    if op == Op::SETTABLE {
        let k = reg!(instruction.c());
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
    let k = constant!(instruction.e());
    if let Some(t) = recv.get_table()
        && set_own_fill_ic(ctx, closure, instruction.d(), unsafe { ip.sub(1) }, t, k, v)
    {
        dispatch!();
    }
    set_slow_body!(
        ctx, thread, registers, ip, handlers, ds, recv, k, v, raw_set
    );
}

/// The slow path of LT, LE and their immediate forms: boxed and mixed
/// numbers, strings, and the metamethods.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cmp_slow<'gc>(
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
    let op = instruction.op();
    let imm = !matches!(op, Op::LT | Op::LE);
    let (a, b, inverted) = if imm {
        let (src, inverted) = instruction.ab_imm_flag();
        let (v, k) = (reg!(src), instruction.imm_value(ctx.mutation()));
        let (a, b) = if matches!(op, Op::GTI | Op::GEI) {
            (k, v)
        } else {
            (v, k)
        };
        (a, b, inverted)
    } else {
        let (lhs, rhs, inverted) = instruction.abc_flag();
        (reg!(lhs), reg!(rhs), inverted)
    };
    let le = matches!(op, Op::LE | Op::LEI | Op::GEI);
    let primitive = if let Some(x) = a.get_integer()
        && let Some(y) = b.get_integer()
    {
        Some(if le { x <= y } else { x < y })
    } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
        Some(if le {
            num::le_int_float(x, y)
        } else {
            num::lt_int_float(x, y)
        })
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
        Some(if le {
            num::le_float_int(x, y)
        } else {
            num::lt_float_int(x, y)
        })
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
        Some(if le { x <= y } else { x < y })
    } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
        Some(if le { x <= y } else { x < y })
    } else {
        None
    };
    if let Some(r) = primitive {
        skip_if!(r != inverted);
        dispatch!();
    }
    let bit = if le {
        MetamethodBits::LE
    } else {
        MetamethodBits::LT
    };
    let meta_fn = binop_metamethod(ctx, a, b, bit);
    if meta_fn.is_nil() {
        raise!(OpError::Compare(a, b));
    }
    if imm {
        call_mm!(ret_cond_b, meta_fn, [a, b]);
    }
    call_mm!(ret_cond_c, meta_fn, [a, b]);
}

/// The continuation for a CALL that keeps `returns - 1` results, MULTRET
/// for 0: specialized for the common fixed counts, as LuaJIT Remake's
/// call variants are.
#[inline(always)]
fn call_ret(returns: u8) -> Handler {
    DISPATCH.rets[returns as usize]
}

/// [`call_ret`] off the dispatch pointer a handler holds in a register.
#[inline(always)]
fn call_ret_at(handlers: *const (), returns: u8) -> Handler {
    unsafe { *handlers.cast::<Handler>().add(Op::COUNT + returns as usize) }
}

/// [`ret_call`] for a CALL that keeps `$n` results, nil-padded.
macro_rules! ret_call_n {
    ($name:ident, $n:literal) => {
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
            let (nret, values, dst) = ret_args!(instruction, registers, ip);
            helpers! { instruction, ctx, thread, registers, ip, handlers, ds, frame, closure }
            let base = unsafe { (*frame).base() };
            let _ = resume_caller!(thread, registers, ip, frame, closure);
            for i in 0..$n {
                let v = if i < nret {
                    unsafe { values.add(i).read() }
                } else {
                    Value::nil()
                };
                unsafe { dst.add(i).write(v) };
            }
            // Nothing above the caller's window is pending, and the window
            // is live anyway: `base` is a valid `top` that drops whatever
            // the callee left above it (#43), one store instead of a count.
            thread.set_top_unchecked(base);
            dispatch!();
        }
    };
}

ret_call_n!(ret_call0, 0);
ret_call_n!(ret_call1, 1);
ret_call_n!(ret_call2, 2);

// ---------------------------------------------------------------------------
// Quickened table accesses
// ---------------------------------------------------------------------------

/// A quickened read: the receiver `$recv` names (`reg` or `upval` operand
/// `b`), its cache entry known to be of kind `$kind`, and with `$self_` the
/// receiver also stored above the result (SELF). Any miss goes to
/// `get_slow`, which refills the entry and requickens.
macro_rules! get_quick {
    ($name:ident, $recv:ident, $kind:ident, $self_:literal) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $name<'gc>(
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
            let (dst, b, ic_idx, _) = instruction.abde();
            let recv = get_quick!(@recv $recv, b, thread);
            let Some(t) = recv.get_table() else {
                tail!(get_slow);
            };
            let state = t.inner().borrow();
            let Some(v) = get_quick!(@load $kind, read_ic(closure, ic_idx), t, &state) else {
                drop(state);
                // An absent key on a shape with an `__index` function: call
                // it from here, as `get_slow` would.
                if get_quick!(@absent $kind)
                    && let InlineCache::Absent { shape } = read_ic(closure, ic_idx)
                    && Shape::ptr_eq(t.shape(), shape)
                    && let Some(mt) = shape.mt_cache()
                    && let index = mt.mm(MetamethodBits::INDEX)
                    && index.get_function().is_some()
                {
                    if $self_ {
                        *reg!(ref mut (dst + 1)) = recv;
                    }
                    call_mm!(ret_store_a, index, [recv, constant!(instruction.e())]);
                }
                tail!(get_slow);
            };
            drop(state);
            if $self_ {
                *reg!(ref mut (dst + 1)) = recv;
            }
            *reg!(ref mut dst) = v;
            dispatch!();
        }
    };
    (@absent Absent) => {
        true
    };
    (@absent $kind:ident) => {
        false
    };
    (@recv reg, $b:ident, $thread:ident) => {
        reg!($b)
    };
    (@recv upval, $b:ident, $thread:ident) => {
        upvalue!(value $b)
    };
    // Own and Absent entries only change through `fill_ic`, which
    // requickens; the collector may empty a ProtoLoad one, so that kind is
    // checked.
    (@load Own, $cache:expr, $t:ident, $state:expr) => {{
        let InlineCache::Own { shape, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        let state: &TableState<'gc> = $state;
        if Shape::ptr_eq(state.shape(), shape) {
            let v = unsafe { $t.load(state, loc) };
            (!(v.is_nil() && shape.has_mm(MetamethodBits::INDEX))).then_some(v)
        } else {
            None
        }
    }};
    (@load Absent, $cache:expr, $t:ident, $state:expr) => {{
        let InlineCache::Absent { shape } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        let state: &TableState<'gc> = $state;
        (Shape::ptr_eq(state.shape(), shape) && !shape.has_mm(MetamethodBits::INDEX))
            .then_some(Value::nil())
    }};
    (@load Proto, $cache:expr, $t:ident, $state:expr) => {{
        let cache = $cache;
        if matches!(cache, InlineCache::ProtoLoad { .. }) {
            ic_get(cache, $t, $state)
        } else {
            None
        }
    }};
}

get_quick!(getfield_own, reg, Own, false);
get_quick!(getfield_absent, reg, Absent, false);
get_quick!(getfield_proto, reg, Proto, false);
get_quick!(gettabup_own, upval, Own, false);
get_quick!(gettabup_absent, upval, Absent, false);
get_quick!(gettabup_proto, upval, Proto, false);
get_quick!(self_own, reg, Own, true);
get_quick!(self_absent, reg, Absent, true);
get_quick!(self_proto, reg, Proto, true);

/// A quickened write, as [`get_quick!`]: `$kind` is `Own` or `Transition`.
macro_rules! set_quick {
    ($name:ident, $recv:ident, $kind:ident) => {
        #[inline(never)]
        #[rustc_align(32)]
        extern "rust-preserve-none" fn $name<'gc>(
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
            let (src, b, ic_idx, _) = instruction.abde();
            let recv = get_quick!(@recv $recv, b, thread);
            let Some(t) = recv.get_table() else {
                tail!(set_slow);
            };
            let cache = read_ic(closure, ic_idx);
            if !matches!(cache, InlineCache::$kind { .. }) {
                unsafe { std::hint::unreachable_unchecked() }
            }
            set_quick!(@store $kind, ctx, instruction, cache, t, recv, src);
            tail!(set_slow);
        }
    };
    (@store Absent, $ctx:ident, $instruction:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        // Filled for a shape with `__newindex`, which a function may answer
        // from here.
        if let InlineCache::Absent { shape } = $cache
            && Shape::ptr_eq($t.shape(), shape)
            && let Some(mt) = shape.mt_cache()
            && let newindex = mt.mm(MetamethodBits::NEWINDEX)
            && newindex.get_function().is_some()
        {
            call_mm!(ret_discard, newindex, [$recv, constant!($instruction.e()), reg!($src)]);
        }
    };
    (@store $kind:ident, $ctx:ident, $instruction:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        if ic_set($ctx, $cache, $t, reg!($src)) {
            dispatch!();
        }
    };
}

set_quick!(setfield_own, reg, Own);
set_quick!(setfield_trans, reg, Transition);
set_quick!(settabup_own, upval, Own);
set_quick!(settabup_trans, upval, Transition);
set_quick!(setfield_absent, reg, Absent);
set_quick!(settabup_absent, upval, Absent);
