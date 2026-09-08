use crate::dmm::{Gc, Mutation, RefLock};
use crate::env::function::{
    Function, FunctionKind, InlineCache, LuaClosure, NativeClosure, NativeContext, Stack, Upvalue,
    UpvalueState,
};
use crate::env::shape::{MetamethodBits, Shape};
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::{
    CallSite, Frame, LuaFrame, PendingAction, Thread, ThreadState, ThreadStatus,
};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, Op, UpValueDescriptor};
#[cfg(jit_enabled)]
use crate::jit;
use crate::lua::Context;
use crate::vm::num::{self, op_arith_float, op_arith_int, op_bit_int};

/// Opcode handlers, indexed by opcode number.
///
/// Entries are assigned by `Op` rather than written in positional order: the
/// ISA table owns the numbering, so reordering or inserting an opcode there
/// can't silently point one at its neighbour's handler.
static HANDLERS: [Handler; Op::COUNT] = handler_table();

const fn handler_table() -> [Handler; Op::COUNT] {
    let mut t: [Handler; Op::COUNT] = [op_unimplemented; Op::COUNT];
    t[Op::MOVE as usize] = op_move;
    t[Op::LOAD as usize] = op_load;
    t[Op::LFALSESKIP as usize] = op_lfalseskip;
    t[Op::GETUPVAL as usize] = op_getupval;
    t[Op::SETUPVAL as usize] = op_setupval;
    t[Op::GETTABUP as usize] = op_gettabup;
    t[Op::SETTABUP as usize] = op_settabup;
    t[Op::GETTABLE as usize] = op_gettable;
    t[Op::SETTABLE as usize] = op_settable;
    t[Op::GETFIELD as usize] = op_getfield;
    t[Op::SETFIELD as usize] = op_setfield;
    t[Op::SELF as usize] = op_self;
    t[Op::NEWTABLE as usize] = op_newtable;
    t[Op::ADD as usize] = op_add;
    t[Op::SUB as usize] = op_sub;
    t[Op::MUL as usize] = op_mul;
    t[Op::MOD as usize] = op_mod;
    t[Op::POW as usize] = op_pow;
    t[Op::DIV as usize] = op_div;
    t[Op::IDIV as usize] = op_idiv;
    t[Op::BAND as usize] = op_band;
    t[Op::BOR as usize] = op_bor;
    t[Op::BXOR as usize] = op_bxor;
    t[Op::SHL as usize] = op_shl;
    t[Op::SHR as usize] = op_shr;
    t[Op::UNM as usize] = op_unm;
    t[Op::BNOT as usize] = op_bnot;
    t[Op::NOT as usize] = op_not;
    t[Op::LEN as usize] = op_len;
    t[Op::CONCAT as usize] = op_concat;
    t[Op::CLOSE as usize] = op_close;
    t[Op::TBC as usize] = op_tbc;
    t[Op::JMP as usize] = op_jmp;
    t[Op::EQ as usize] = op_eq;
    t[Op::LT as usize] = op_lt;
    t[Op::LE as usize] = op_le;
    t[Op::TEST as usize] = op_test;
    t[Op::TESTSET as usize] = op_testset;
    t[Op::CALL as usize] = op_call;
    t[Op::TAILCALL as usize] = op_tailcall;
    t[Op::RETURN as usize] = op_return;
    t[Op::FORLOOP as usize] = op_forloop;
    t[Op::FORPREP as usize] = op_forprep;
    t[Op::TFORPREP as usize] = op_tforprep;
    t[Op::TFORCALL as usize] = op_tforcall;
    t[Op::TFORLOOP as usize] = op_tforloop;
    t[Op::SETLIST as usize] = op_setlist;
    t[Op::CLOSURE as usize] = op_closure;
    t[Op::VARARG as usize] = op_vararg;
    t[Op::VARARGGET as usize] = op_varargget;
    t[Op::VARARGPREP as usize] = op_varargprep;
    t[Op::ERRNNIL as usize] = op_errnnil;
    t[Op::NOP as usize] = op_nop;
    t[Op::STOP as usize] = op_stop;
    t
}

#[derive(Debug)]
pub(crate) struct Error {
    pub pc: usize,
}

pub(crate) type Registers<'gc, 'a> = *mut Value<'gc>;

pub(crate) type Handler = for<'gc> extern "rust-preserve-none" fn(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

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
    ($instruction:expr, $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr) => {
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
                    become handler(instruction, $ctx, $thread, $registers, ip, $handlers);
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! raise {
            () => {{
                become impl_error($instruction, $ctx, $thread, $registers, $ip, $handlers);
            }};
        }

        #[allow(unused_macros)]
        macro_rules! check {
            ($$cond:expr) => {{
                if std::hint::unlikely(!$$cond) {
                    raise!();
                }
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
                    let frame = $thread.top_lua_unchecked();
                    *frame.closure.proto.constants.get_unchecked($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                unsafe {
                    let frame = $thread.top_lua_unchecked();
                    *frame.closure.upvalues.get_unchecked($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! skip {
            () => {{
                $ip = unsafe { $ip.add(1) };
            }};
        }

        /// Schedule a metamethod (or iterator) call and dispatch into it.
        /// A Lua target dispatches into a fresh frame whose `op_return` resumes
        /// the continuation; a native target runs inline — synchronously the
        /// continuation payload fires immediately, otherwise the call suspends
        /// through the executor (`schedule_meta_call` installs the pending
        /// action / error frame). A non-callable target raises.
        #[allow(unused_macros)]
        macro_rules! invoke_metamethod {
            ($$meta:expr, $$args:expr, $$cont:expr) => {{
                let __mm_cont: Continuation = $$cont;
                match schedule_meta_call($ctx, $thread, $$meta, $$args, __mm_cont, $ip) {
                    MetaDispatch::Lua { new_ip, new_base } => {
                        $ip = new_ip;
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
                            $handlers
                        );
                    }
                    // Native target suspended (or errored): the executor will
                    // resume / unwind from the installed frame state.
                    MetaDispatch::Suspended => return Ok(()),
                    MetaDispatch::Unresolvable => raise!(),
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
     $ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr) => {{
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
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr,
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
                invoke_metamethod!(__mm_func, &[__mm_recv, __k], __cont);
            }
            IndexChain::Exhausted => raise!(),
        }
    }};
}

/// Inflates the slow-path body for "userdata get via `__index`". Unlike
/// the table case, a userdata with no metatable or a nil `__index` is an
/// "attempt to index" error (`userdata_index_chain` returns `Err`), since
/// userdata has no raw indexing to fall back to. A key absent from the
/// `__index` table still resolves to nil.
macro_rules! userdata_get_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr,
     $u:expr, $recv:expr, $k:expr, $dst:expr) => {{
        let __u = $u;
        let __recv: Value<'gc> = $recv;
        let __k: Value<'gc> = $k;
        let __dst_reg: u8 = $dst;

        match userdata_index_chain(__u, __recv, __k, $ctx.symbols().mm_index) {
            Err(()) => raise!(),
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
                invoke_metamethod!(__mm_func, &[__mm_recv, __k], __cont);
            }
            Ok(IndexChain::Exhausted) => raise!(),
        }
    }};
}

/// Inflates the slow-path body for "table set with metamethod".
///
/// Walks the `__newindex` chain via `walk_newindex_chain`, which
/// returns either the table to raw-write into, or a callable to invoke.
macro_rules! table_set_slow_body {
    ($ctx:expr, $thread:expr, $registers:ident, $ip:ident, $handlers:expr,
     $t:expr, $k:expr, $v:expr) => {{
        let __t: Table<'gc> = $t;
        let __k: Value<'gc> = $k;
        let __new_val: Value<'gc> = $v;

        match walk_newindex_chain(__t, __k, $ctx.symbols().mm_newindex) {
            NewIndexChain::RawSet(__target) => {
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
                invoke_metamethod!(__mm_func, &[__mm_recv, __k, __new_val], __cont);
            }
            NewIndexChain::Exhausted => raise!(),
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
fn observe_ic_type<'gc>(thread: &ThreadState<'gc>, ic_idx: u16, v: Value<'gc>) {
    let proto = unsafe { &thread.top_lua_unchecked().closure.proto };
    if let Some(cell) = proto.ic_types.get(ic_idx as usize) {
        crate::env::value::KindSet::observe(cell, v);
    }
}

fn read_ic<'gc>(thread: &ThreadState<'gc>, ic_idx: u16) -> InlineCache<'gc> {
    // SAFETY: ic_idx is allocated at compile-time within the prototype's
    // IC count; debug-asserted in alloc_ic_slot's saturating_add.
    let proto = unsafe { &thread.top_lua_unchecked().closure.proto };
    debug_assert!((ic_idx as usize) < proto.ic_table.len());
    unsafe { proto.ic_table.get_unchecked(ic_idx as usize) }.get()
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
pub(crate) fn run_thread<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) -> Result<(), Box<Error>> {
    let mut ts = thread.borrow_mut(ctx.mutation());
    let (ip, base) = {
        let frame = ts
            .top_lua()
            .expect("run_thread requires a seeded Lua frame");
        let code_ptr = frame.closure.proto.code.as_ptr();
        let ip = unsafe { code_ptr.add(frame.pc) };
        (ip, frame.base)
    };
    let registers = unsafe { ts.stack.as_mut_ptr().add(base) };
    let handlers = HANDLERS.as_ptr() as *const ();
    op_nop(Instruction::nop(), ctx, &mut ts, registers, ip, handlers)
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Placeholder for any opcode number the ISA table declares but
/// `handler_table` never assigns. Unreachable unless the two drift apart.
#[cold]
#[inline(never)]
extern "rust-preserve-none" fn op_unimplemented<'gc>(
    instruction: Instruction,
    _ctx: Context<'gc>,
    _thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    unreachable!("no handler installed for {:?}", instruction.op())
}

#[cold]
#[inline(never)]
extern "rust-preserve-none" fn impl_error<'gc>(
    _instruction: Instruction,
    _ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    // See #44: compute proper PC from current frame's prototype code base.
    Err(Box::new(Error { pc: 0 }))
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_move<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, src) = instruction.ab();
    *reg!(ref mut dst) = reg!(src);
    dispatch!();
}

/// Load constant from the current prototype's constant pool.
#[inline(never)]
extern "rust-preserve-none" fn op_load<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, idx) = instruction.ad();
    *reg!(ref mut dst) = constant!(idx);
    dispatch!();
}

/// Set register to false and skip the next instruction.
#[inline(never)]
extern "rust-preserve-none" fn op_lfalseskip<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let src = instruction.a();
    *reg!(ref mut src) = Value::boolean(false);
    skip!();
    dispatch!();
}

// ---------------------------------------------------------------------------
// Upvalue access
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_getupval<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, idx) = instruction.ab();
    let uv = upvalue!(idx);
    *reg!(ref mut dst) = read_upvalue(thread, uv);
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_setupval<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_gettabup<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, idx, ic_idx, _key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        raise!();
    };

    let cache = read_ic(thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let v = unsafe { t_state.property_at(slot) };
            if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
                drop(t_state);
                observe_ic_type(thread, ic_idx, v);
                *reg!(ref mut dst) = v;
                dispatch!();
            }
        }
    }
    drop(t_state);
    become gettabup_slow(instruction, ctx, thread, registers, ip, handlers);
}

#[inline(never)]
extern "rust-preserve-none" fn gettabup_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let Some(t) = t_val.get_table() else {
        raise!();
    };
    let k = constant!(key);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_get_slow_body!(ctx, thread, registers, ip, handlers, t, k, dst);
}

/// UpValue[idx][K[key]] = R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_settabup<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);

    let Some(t) = t_val.get_table() else {
        raise!();
    };

    let v = reg!(src);
    let cache = read_ic(thread, ic_idx);
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
    become settabup_slow(instruction, ctx, thread, registers, ip, handlers);
}

#[inline(never)]
extern "rust-preserve-none" fn settabup_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, idx, ic_idx, key) = instruction.abde();
    let uv = upvalue!(idx);
    let t_val = read_upvalue(thread, uv);
    let Some(t) = t_val.get_table() else {
        raise!();
    };
    let k = constant!(key);
    let v = reg!(src);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, t, k, v);
}

// ---------------------------------------------------------------------------
// Table access via register
// ---------------------------------------------------------------------------

/// R[dst] = R[table][R[key]]
#[inline(never)]
extern "rust-preserve-none" fn op_gettable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        become gettable_slow(instruction, ctx, thread, registers, ip, handlers);
    };

    let k = reg!(key);
    let (v, need_index) = {
        let t_state = t.inner().borrow();
        let v = t_state.raw_get(k);
        let need = v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        become gettable_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    *reg!(ref mut dst) = v;
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn gettable_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, table, key) = instruction.abc();
    let recv = reg!(table);
    let k = reg!(key);
    if let Some(t) = recv.get_table() {
        table_get_slow_body!(ctx, thread, registers, ip, handlers, t, k, dst);
    }
    let Some(u) = recv.get_userdata() else {
        raise!();
    };
    userdata_get_slow_body!(ctx, thread, registers, ip, handlers, u, recv, k, dst);
}

/// R[table][R[key]] = R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_settable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, table, key) = instruction.abc();

    let Some(t) = reg!(table).get_table() else {
        raise!();
    };

    let k = reg!(key);
    let v = reg!(src);
    let needs_newindex = {
        let t_state = t.inner().borrow();
        t_state.shape().has_mm(MetamethodBits::NEWINDEX) && t_state.raw_get(k).is_nil()
    };

    if needs_newindex {
        become settable_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let mut t_state = t.inner().borrow_mut(ctx.mutation());
    t_state.raw_set(ctx, k, v);
    dispatch!()
}

#[inline(never)]
extern "rust-preserve-none" fn settable_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, table, key) = instruction.abc();
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let k = reg!(key);
    let v = reg!(src);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, t, k, v);
}

/// R[dst] = R[table][K[key_idx]]
#[inline(never)]
extern "rust-preserve-none" fn op_getfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, table, ic_idx, _key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        // Non-table (userdata `__index`, or error) — handled by the slow path.
        become getfield_slow(instruction, ctx, thread, registers, ip, handlers);
    };

    let cache = read_ic(thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let v = unsafe { t_state.property_at(slot) };
            if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
                drop(t_state);
                observe_ic_type(thread, ic_idx, v);
                *reg!(ref mut dst) = v;
                dispatch!();
            }
        }
    }
    drop(t_state);
    become getfield_slow(instruction, ctx, thread, registers, ip, handlers);
}

#[inline(never)]
extern "rust-preserve-none" fn getfield_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, table, ic_idx, key_idx) = instruction.abde();
    let recv = reg!(table);
    let k = constant!(key_idx);
    if let Some(t) = recv.get_table() {
        fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
        table_get_slow_body!(ctx, thread, registers, ip, handlers, t, k, dst);
    }
    let Some(u) = recv.get_userdata() else {
        raise!();
    };
    userdata_get_slow_body!(ctx, thread, registers, ip, handlers, u, recv, k, dst);
}

/// R[table][K[key_idx]] = R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_setfield<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, table, ic_idx, key_idx) = instruction.abde();

    let Some(t) = reg!(table).get_table() else {
        raise!();
    };

    let v = reg!(src);
    let cache = read_ic(thread, ic_idx);
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
    become setfield_slow(instruction, ctx, thread, registers, ip, handlers);
}

#[inline(never)]
extern "rust-preserve-none" fn setfield_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, table, ic_idx, key_idx) = instruction.abde();
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let k = constant!(key_idx);
    let v = reg!(src);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, t, k, v);
}

// ---------------------------------------------------------------------------
// SELF — method-call setup
// ---------------------------------------------------------------------------

/// Backs `obj:m(...)`. Writes the method into `R[dst]` and the
/// receiver into `R[dst+1]`. No inline cache for now — see the
/// instruction definition.
#[inline(never)]
extern "rust-preserve-none" fn op_self<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        // Non-table receiver (userdata method dispatch, or an error).
        become op_self_nontable(instruction, ctx, thread, registers, ip, handlers);
    };

    let key = constant!(key_idx);
    let (method, need_index) = {
        let recv_state = recv.inner().borrow();
        let v = recv_state.raw_get(key);
        let need = v.is_nil() && recv_state.shape().has_mm(MetamethodBits::INDEX);
        (v, need)
    };

    if need_index {
        become op_self_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    *reg!(ref mut dst) = method;
    *reg!(ref mut (dst + 1)) = recv_val;
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_self_slow<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(recv) = recv_val.get_table() else {
        raise!();
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
            invoke_metamethod!(func, &[receiver, key], cont);
        }
        IndexChain::Exhausted => raise!(),
    }
}

/// SELF on a non-table receiver. Userdata dispatches through its
/// metatable's `__index`; unlike GETTABLE, a missing method is *not* an
/// error here — we write `nil` into `R[dst]` and let the subsequent CALL
/// raise "attempt to call a nil value (method ...)", matching Lua. A
/// userdata with no metatable / nil `__index` likewise yields a nil
/// method (the CALL raises). Any other non-table receiver raises.
#[inline(never)]
extern "rust-preserve-none" fn op_self_nontable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, object, key_idx) = instruction.abd();

    let recv_val = reg!(object);
    let Some(u) = recv_val.get_userdata() else {
        raise!();
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
            invoke_metamethod!(func, &[receiver, key], cont);
        }
        IndexChain::Exhausted => raise!(),
    }
}

/// R[dst] = {}
#[inline(never)]
extern "rust-preserve-none" fn op_newtable<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
        ) -> Result<(), Box<Error>> {
            helpers!(instruction, ctx, thread, registers, ip, handlers);
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

                        raise!(); // handle e.g. div by zero
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

            become $slow_name(instruction, ctx, thread, registers, ip, handlers);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_arith_mixed, $mm);
    };
}

/// `R[dst] = R[lhs] <op> R[rhs]` for the bitwise opcodes.
macro_rules! bit_handler {
    ($fn_name:ident, $slow_name:ident, $instr:ident, $num_kind:ty, $mm:ident) => {
        #[inline(never)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            registers: Registers<'gc, '_>,
            ip: *const Instruction,
            handlers: *const (),
        ) -> Result<(), Box<Error>> {
            helpers!(instruction, ctx, thread, registers, ip, handlers);
            let (dst, lhs, rhs) = instruction.abc();

            // same-type int/float
            if let (Some(lhs), Some(rhs)) =
                (reg!(ref lhs).get_integer(), reg!(ref rhs).get_integer())
            {
                *reg!(ref mut dst) = op_bit_int::<$num_kind>(lhs, rhs);
                dispatch!();
            }

            become $slow_name(instruction, ctx, thread, registers, ip, handlers);
        }

        binop_slow_handler!($slow_name, $instr, $num_kind, op_bit_mixed, $mm);
    };
}

macro_rules! binop_slow_handler {
    ($slow_name:ident, $instr:ident, $num_kind:ty, $num_mix_h:ident, $mm:ident) => {
        #[inline(never)]
        extern "rust-preserve-none" fn $slow_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
        ) -> Result<(), Box<Error>> {
            helpers!(instruction, ctx, thread, registers, ip, handlers);
            let (dst, lhs, rhs) = instruction.abc();

            {
                let (lhs, rhs) = (reg!(ref lhs), reg!(ref rhs));
                debug_assert_ne!(lhs.kind(), rhs.kind());
                let mixed = num::$num_mix_h::<$num_kind>(lhs, rhs);
                if std::hint::unlikely(mixed.is_some()) {
                    if let Some(v) = mixed {
                        *reg!(ref mut dst) = v;
                        dispatch!();
                    }
                }
            }

            let (lhs, rhs) = (reg!(lhs), reg!(rhs));
            let meta_fn = binop_metamethod(lhs, rhs, ctx.symbols().$mm);
            if meta_fn.is_nil() {
                raise!();
            }

            let cont = Continuation {
                payload: ContinuationPayload::StoreResult { dst },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(meta_fn, &[lhs, rhs], cont);
        }
    };
}

arith_handler!(op_add, op_add_meta, ADD, num::Add, mm_add);
arith_handler!(op_sub, op_sub_meta, SUB, num::Sub, mm_sub);
arith_handler!(op_mul, op_mul_meta, MUL, num::Mul, mm_mul);
arith_handler!(op_mod, op_mod_meta, MOD, num::Mod, mm_mod);
arith_handler!(op_pow, op_pow_meta, POW, num::Pow, mm_pow);
arith_handler!(op_div, op_div_meta, DIV, num::Div, mm_div);
arith_handler!(op_idiv, op_idiv_meta, IDIV, num::IDiv, mm_idiv);
bit_handler!(op_band, op_band_meta, BAND, num::BAnd, mm_band);
bit_handler!(op_bor, op_bor_meta, BOR, num::BOr, mm_bor);
bit_handler!(op_bxor, op_bxor_meta, BXOR, num::BXor, mm_bxor);
bit_handler!(op_shl, op_shl_meta, SHL, num::Shl, mm_shl);
bit_handler!(op_shr, op_shr_meta, SHR, num::Shr, mm_shr);

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

/// R[dst] = -R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_unm<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
        raise!();
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
extern "rust-preserve-none" fn op_bnot<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(ref mut dst) = Value::integer(!i);
        dispatch!();
    }
    let meta_fn = unop_metamethod(val, ctx.symbols().mm_bnot);
    if meta_fn.is_nil() {
        raise!();
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
extern "rust-preserve-none" fn op_not<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, src) = instruction.ab();
    let val = reg!(src);
    *reg!(ref mut dst) = Value::boolean(val.is_falsy());
    dispatch!();
}

/// R[dst] = #R[src]  (length)
#[inline(never)]
extern "rust-preserve-none" fn op_len<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
        raise!()
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
extern "rust-preserve-none" fn op_concat<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
        raise!();
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
extern "rust-preserve-none" fn op_close<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let start = instruction.a();
    let base = thread.top_lua().map_or(0, |f| f.base);
    let start_idx = base + start as usize;
    close_upvalues(ctx.mutation(), thread, start_idx);
    close_tbc_vars(ctx.mutation(), thread, start_idx);
    dispatch!();
}

/// Mark R[val] as to-be-closed.
#[inline(never)]
extern "rust-preserve-none" fn op_tbc<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let val = instruction.a();
    let base = thread.top_lua().map_or(0, |f| f.base);
    thread.tbc_slots.push(base + val as usize);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Jumps and conditionals
// ---------------------------------------------------------------------------

/// pc += offset
#[inline(never)]
extern "rust-preserve-none" fn op_jmp<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let offset = instruction.imm();
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// if (R[lhs] == R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_eq<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let a = reg!(lhs);
    let b = reg!(rhs);
    if a == b {
        // Primitive or pointer-equal — no metamethod consultation.
        if !inverted {
            skip!();
        }
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
    if inverted {
        skip!();
    }
    dispatch!();
}

/// if (R[lhs] < R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_lt<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some((x as f64) < y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(x < (y as f64))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x < y)
        } else {
            None
        }
    };

    if let Some(r) = primitive {
        if r != inverted {
            skip!();
        }
        dispatch!();
    }

    let (a, b) = (reg!(lhs), reg!(rhs));
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_lt);
    if meta_fn.is_nil() {
        raise!();
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
extern "rust-preserve-none" fn op_le<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = instruction.abc_flag();

    let primitive = {
        let (a, b) = (reg!(ref lhs), reg!(ref rhs));
        if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
            Some(x <= y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(x <= y)
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some((x as f64) <= y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(x <= (y as f64))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x <= y)
        } else {
            None
        }
    };

    if let Some(r) = primitive {
        if r != inverted {
            skip!();
        }
        dispatch!();
    }

    let (a, b) = (reg!(lhs), reg!(rhs));
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_le);
    if meta_fn.is_nil() {
        raise!();
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

/// if (not R[src]) == inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_test<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (src, inverted) = instruction.ab_flag();
    let truthy = !reg!(src).is_falsy();
    if truthy != inverted {
        skip!();
    }
    dispatch!();
}

/// If (truthy(R[src]) == inverted) then skip next instruction;
/// otherwise R[dst] := R[src] and fall through. Matches Lua 5.5 TESTSET.
#[inline(never)]
extern "rust-preserve-none" fn op_testset<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, src, inverted) = instruction.abc_flag();
    let val = reg!(src);
    let truthy = !val.is_falsy();
    if truthy == inverted {
        skip!();
    } else {
        *reg!(ref mut dst) = val;
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Function calls
// ---------------------------------------------------------------------------

/// R[func], ..., R[func+returns-2] = R[func](R[func+1], ..., R[func+args-1])
#[inline(never)]
extern "rust-preserve-none" fn op_call<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (func, nargs, returns) = instruction.abc();
    let base = thread.top_lua().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            let new_base = func_idx + 1;
            if let Some(frame) = thread.top_lua_mut() {
                let code_start = frame.closure.proto.code.as_ptr();
                frame.pc = unsafe { ip.offset_from_unsigned(code_start) };
            }
            thread.ensure_slots(new_base + closure.proto.max_stack_size as usize);
            // `nargs == 0` is the MULTRET sentinel: read the count from `thread.top`.
            let caller_provided = if nargs == 0 {
                thread.top - new_base
            } else {
                nargs as usize - 1
            };
            let num_params = closure.proto.num_params as usize;
            for i in caller_provided..num_params {
                thread.stack[new_base + i] = Value::nil();
            }
            let num_extras = if closure.proto.is_vararg {
                caller_provided.saturating_sub(num_params) as u32
            } else {
                0
            };
            thread.push_lua(LuaFrame {
                closure,
                base: new_base,
                pc: 0,
                num_results: returns,
                num_extras,
                continuation: None,
            });

            // The JIT's only entry point. The frame is already pushed and its
            // registers are in place, so a region can run over it as-is, and a
            // deopt out of one leaves a frame the interpreter can simply pick up.
            #[cfg(jit_enabled)]
            match jit::region::on_call(ctx, thread, closure.proto, new_base) {
                jit::region::Outcome::Interpret => {}
                jit::region::Outcome::Deopt(pc) => {
                    thread.top_lua_mut().unwrap().pc = pc;
                    ip = unsafe { closure.proto.code.as_ptr().add(pc) };
                    registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
                    dispatch!();
                }
                jit::region::Outcome::Returned(nret) => {
                    // Native code lands its results at `base + 0` — which is what
                    // makes this the same unwind `op_return` does, just with the
                    // count coming from the status word instead of the bytecode.
                    match frame_return(ctx.mutation(), thread, new_base, nret) {
                        FrameReturn::TopLevel | FrameReturn::ToNonLua => return Ok(()),
                        FrameReturn::Caller {
                            new_base: caller_base,
                            new_ip,
                        } => {
                            ip = new_ip;
                            registers = unsafe { thread.stack.as_mut_ptr().add(caller_base) };
                            dispatch!();
                        }
                        // Continuations are attached by the metamethod helpers to
                        // frames *they* push; this one was pushed a dozen lines up
                        // with `continuation: None`.
                        FrameReturn::Continuation => unreachable!("op_call pushed no continuation"),
                    }
                }
            }

            ip = closure.proto.code.as_ptr();
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
                    // Push Frame::Error so the executor's unwinder finds
                    // the nearest catching `Frame::Sequence` (e.g. the
                    // PCallSequence under coroutine.resume). Persist
                    // caller's pc first so re-entry would work if anything
                    // catches and resumes.
                    if let Some(frame) = thread.top_lua_mut() {
                        let code_start = frame.closure.proto.code.as_ptr();
                        frame.pc = unsafe { ip.offset_from_unsigned(code_start) };
                    }
                    thread.frames.push(Frame::Error(err));
                    return Ok(());
                }
            };
            match action {
                crate::vm::sequence::CallbackAction::Return => {
                    // Result count comes via the logical top, not Vec::len:
                    // `invoke_native` never shrinks the shared stack, so the
                    // caller's register window is still fully covered.
                    let retc = thread.top - args_base;
                    // Place results at stack[func_idx..] following Lua convention.
                    let wanted = if returns == 0 {
                        retc
                    } else {
                        returns as usize - 1
                    };
                    let to_copy = retc.min(wanted);
                    for i in 0..to_copy {
                        thread.stack[func_idx + i] = thread.stack[args_base + i];
                    }
                    for i in to_copy..wanted {
                        thread.stack[func_idx + i] = Value::nil();
                    }
                    // Publish the logical top. For MULTRET this is the dynamic
                    // count the next consumer reads; either way it nils the
                    // function slot and the stale donor copies the down-shift
                    // left above the results, which are dead scratch (a call's
                    // function always sits at the caller's first free register).
                    thread.set_top(func_idx + wanted);
                    registers = unsafe { thread.stack.as_mut_ptr().add(base) };
                    dispatch!();
                }
                action => {
                    // Suspension path: persist caller's pc, stash the
                    // action on the thread for the executor to translate
                    // into frame ops, then exit the dispatch chain.
                    if let Some(frame) = thread.top_lua_mut() {
                        let code_start = frame.closure.proto.code.as_ptr();
                        frame.pc = unsafe { ip.offset_from_unsigned(code_start) };
                    }
                    thread.pending_action = Some(PendingAction {
                        action,
                        call_site: CallSite {
                            bottom: args_base,
                            func_idx,
                            returns,
                            cont: None,
                        },
                    });
                    return Ok(());
                }
            }
        }
    }
}

/// return R[func](R[func+1], ..., R[func+args-1])  — tail call
#[inline(never)]
extern "rust-preserve-none" fn op_tailcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (func, nargs) = instruction.ab();
    let base = thread.top_lua().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            // Results must land in this frame's *original* func slot in its
            // caller. A vararg frame's VARARGPREP shifted base past the extras
            // at `[base - num_extras .. base]`, so that slot is below them.
            let (cur_base, cur_num_extras) = {
                let f = thread.top_lua().unwrap();
                (f.base, f.num_extras as usize)
            };
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
            let frame = thread.top_lua_mut().unwrap();
            frame.closure = closure;
            frame.pc = 0;
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
            ip = closure.proto.code.as_ptr();
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
                    // Tailcall + native error: pop the tailcalling Lua
                    // frame first (it's morally already gone), then push
                    // Frame::Error onto the now-top frame for the
                    // executor's unwinder.
                    let cur_base = thread.top_lua().unwrap().base;
                    close_upvalues(ctx.mutation(), thread, cur_base);
                    close_tbc_vars(ctx.mutation(), thread, cur_base);
                    thread.frames.pop();
                    thread.frames.push(Frame::Error(err));
                    return Ok(());
                }
            };
            match action {
                crate::vm::sequence::CallbackAction::Return => {
                    // Result count via the logical top (the shared stack was
                    // never shrunk by the native call).
                    let retc = thread.top - args_base;
                    match frame_return(ctx.mutation(), thread, args_base, retc) {
                        FrameReturn::Continuation => {
                            become cont_resume(instruction, ctx, thread, registers, ip, handlers);
                        }
                        FrameReturn::TopLevel => return Ok(()),
                        FrameReturn::ToNonLua => return Ok(()),
                        FrameReturn::Caller { new_base, new_ip } => {
                            ip = new_ip;
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
                    return Ok(());
                }
            }
        }
    }
}

/// return R[values], ..., R[values+count-2]
#[inline(never)]
extern "rust-preserve-none" fn op_return<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (values, count) = instruction.ab();

    let cur_base = thread.top_lua().unwrap().base;
    let values_base = cur_base + values as usize;
    // `count == 0` is MULTRET: read the count from `thread.top`.
    let nret = if count == 0 {
        thread.top - values_base
    } else {
        count as usize - 1
    };

    match frame_return(ctx.mutation(), thread, values_base, nret) {
        FrameReturn::Continuation => {
            become cont_resume(instruction, ctx, thread, registers, ip, handlers);
        }
        FrameReturn::TopLevel => return Ok(()),
        FrameReturn::ToNonLua => return Ok(()),
        FrameReturn::Caller { new_base, new_ip } => {
            ip = new_ip;
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric for loop
// ---------------------------------------------------------------------------

/// Prepare numeric for: validate and set up counter.
/// R[base] = initial value, R[base+1] = limit, R[base+2] = step
/// If loop won't execute, jump forward by offset.
#[inline(never)]
extern "rust-preserve-none" fn op_forprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (base, offset) = instruction.a_imm();

    let init = reg!(base);
    let limit = reg!(base + 1);
    let step = reg!(base + 2);

    let should_run = if let (Some(i), Some(lim), Some(s)) =
        (init.get_integer(), limit.get_integer(), step.get_integer())
    {
        if s > 0 { i <= lim } else { i >= lim }
    } else {
        let i = to_number(init).unwrap_or(0.0);
        let lim = to_number(limit).unwrap_or(0.0);
        let s = to_number(step).unwrap_or(0.0);
        if s > 0.0 { i <= lim } else { i >= lim }
    };

    if !should_run {
        ip = unsafe { ip.offset(offset as isize) };
    }

    // R[base+3] is the visible loop variable (copy of init)
    *reg!(ref mut base + 3) = init;

    dispatch!();
}

/// Numeric for loop step: update counter and test.
/// If loop continues, jump back by offset.
#[inline(never)]
extern "rust-preserve-none" fn op_forloop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (base, offset) = instruction.a_imm();

    let step = reg!(base + 2);

    let cur = reg!(base);
    let lim_v = reg!(base + 1);
    if let (Some(i), Some(lim), Some(s)) =
        (cur.get_integer(), lim_v.get_integer(), step.get_integer())
    {
        let next = i.wrapping_add(s);
        let cont = if s > 0 { next <= lim } else { next >= lim };
        if cont {
            *reg!(ref mut base) = Value::integer(next);
            *reg!(ref mut base + 3) = Value::integer(next);
            ip = unsafe { ip.offset(offset as isize) };
        }
    } else {
        let i = to_number(cur).unwrap_or(0.0);
        let lim = to_number(lim_v).unwrap_or(0.0);
        let s = to_number(step).unwrap_or(0.0);
        let next = i + s;
        let cont = if s > 0.0 { next <= lim } else { next >= lim };
        if cont {
            *reg!(ref mut base) = Value::float(next);
            *reg!(ref mut base + 3) = Value::float(next);
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
extern "rust-preserve-none" fn op_tforprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (_base, offset) = instruction.a_imm();
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: R[base+3], ... = R[base](R[base+1], R[base+2])
#[inline(never)]
extern "rust-preserve-none" fn op_tforcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_tforloop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_setlist<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (table, count, offset) = instruction.abd();
    let Some(t) = reg!(table).get_table() else {
        raise!();
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
extern "rust-preserve-none" fn op_closure<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_vararg<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_varargget<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_varargprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
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
extern "rust-preserve-none" fn op_errnnil<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    // TODO: surface the name (see #28).
    let (src, _name_key) = instruction.ad();
    check!(reg!(src).is_nil());
    dispatch!();
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_nop<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_stop<'gc>(
    _instruction: Instruction,
    _ctx: Context<'gc>,
    _thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_number(v: Value) -> Option<f64> {
    if let Some(i) = v.get_integer() {
        return Some(i as f64);
    }
    if let Some(f) = v.get_float() {
        return Some(f);
    }
    None
}

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
    /// Caller should return `Ok(())` from the handler.
    TopLevel,
    /// Normal return to the caller frame, which has been restored to the
    /// top of the frame stack. Caller updates `ip` / `registers` and
    /// dispatches.
    Caller {
        new_base: usize,
        new_ip: *const Instruction,
    },
    /// The popped Lua frame's parent is a non-Lua frame (Sequence /
    /// WaitThread / Start / Error). The values have been left at
    /// `stack[bottom..]` for the executor's driver loop to consume on the
    /// next pump. `op_return` returns `Ok(())` to exit dispatch.
    ToNonLua,
}

/// Unwind the top-of-stack frame assuming it returned the values at
/// `stack[values_base .. values_base + nret]`. Shared by the bytecode
/// `RETURN` handler and the native-tailcall path.
pub(crate) fn frame_return<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    values_base: usize,
    nret: usize,
) -> FrameReturn {
    let (cur_base, num_results, num_extras, continuation) = {
        let f = thread.top_lua().unwrap();
        (f.base, f.num_results, f.num_extras as usize, f.continuation)
    };

    if let Some(mut cont) = continuation {
        cont.results_base = values_base;
        cont.nret = nret as u8;
        thread.top_lua_mut().unwrap().continuation = Some(cont);
        return FrameReturn::Continuation;
    }

    // The func slot sits at `cur_base - 1 - num_extras`: VARARGPREP shifted
    // base past the extras at `[cur_base - num_extras .. cur_base]` (0 for
    // non-vararg frames).
    close_upvalues(mc, thread, cur_base);
    close_tbc_vars(mc, thread, cur_base);
    thread.frames.pop();

    let dst_start = cur_base - 1 - num_extras;

    if thread.frames.is_empty() {
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

    // If the parent isn't a Lua frame (Sequence/WaitThread/etc.), the executor
    // driver picks up here: place all `nret` values at `stack[dst_start..]` for
    // the parent's window. This MUST be handled before the `num_results`
    // branch below — doing both copies would run two overlapping forward
    // moves of the same source range, and the first would clobber the source
    // of the second whenever `nret` is large enough that `dst_start + nret`
    // reaches into `[values_base..]` (e.g. a sort comparator returning 4+
    // values). A single copy is safe because `dst_start < values_base`, so each
    // write lands on a slot already consumed.
    if thread.top_lua().is_none() {
        thread
            .stack
            .copy_within(values_base..values_base + nret, dst_start);
        // Parent is a Sequence / WaitThread, so no register window sits above
        // the results — the shrink is legal, and it publishes the parent's
        // input window as `stack[dst_start..top]`.
        thread.discard_above(dst_start + nret);
        return FrameReturn::ToNonLua;
    }

    // `num_results == 0` is the CALL's MULTRET: deliver all `nret` and publish `thread.top`.
    if num_results == 0 {
        thread
            .stack
            .copy_within(values_base..values_base + nret, dst_start);
        // Publishing through `set_top` also nils the donor copies the down-shift
        // left in the popped callee's registers, which would otherwise stay
        // traced as the caller's dead scratch.
        thread.set_top(dst_start + nret);
    } else {
        let wanted = num_results as usize - 1;
        let to_copy = nret.min(wanted);
        for i in 0..to_copy {
            thread.stack[dst_start + i] = thread.stack[values_base + i];
        }
        for i in to_copy..wanted {
            thread.stack[dst_start + i] = Value::nil();
        }
        // Publish the landing end. Without this, a `top` left high by a multires
        // producer *inside the callee* would still be the high-water long after
        // the callee popped, so `live_top` would keep tracing its dead registers
        // — the exact #43 leak, just via a stale `top` instead of the vec length.
        // Also nils those registers, which are dead scratch: `dst_start` is the
        // caller's function slot, and everything from there up is free.
        thread.set_top(dst_start + wanted);
    }

    let caller = thread.top_lua().unwrap();
    let new_base = caller.base;
    let new_ip = unsafe { caller.closure.proto.code.as_ptr().add(caller.pc) };
    FrameReturn::Caller { new_base, new_ip }
}

/// Close all open upvalues pointing at stack indices >= `start_idx`.
/// Each open upvalue is converted to Closed by capturing the current stack value.
pub(crate) fn close_upvalues<'gc>(
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    start_idx: usize,
) {
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
    /// pushed. The caller rebinds `ip`/`registers` to this and dispatches.
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
    /// caller returns `Ok(())` to exit dispatch.
    Suspended,
    /// Target is not callable (or a suspending comparison metamethod, which we
    /// don't support). The caller raises.
    Unresolvable,
}

/// Walk the `__call` chain at `thread.stack[func_idx]` until we hit a
/// callable target, shifting args right by one on each hop to prepend the
/// current callee as the first argument (Lua 5.5 `tryfuncTM` behavior).
/// Returns the resolved target and the (possibly adjusted) `nargs`, or
/// `None` if the chain is unresolvable: non-callable value, variadic call
/// with `__call` (see #46), or `MAX_TAG_LOOP` exhaustion. Callers raise on
/// `None`.
#[inline]
fn resolve_call_chain<'gc>(
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
        if nargs == 0 {
            // See #46: variadic + __call not yet supported.
            return None;
        }
        let actual_args = nargs as usize - 1;
        thread.ensure_slots(func_idx + 2 + actual_args);
        for i in (0..actual_args).rev() {
            thread.stack[func_idx + 2 + i] = thread.stack[func_idx + 1 + i];
        }
        thread.stack[func_idx + 1] = func_val;
        thread.stack[func_idx] = mm;
        nargs += 1;
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
    if let Some(frame) = thread.top_lua_mut() {
        let code_start = frame.closure.proto.code.as_ptr();
        frame.pc = unsafe { caller_ip.offset_from_unsigned(code_start) };
    }

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
        pc: 0,
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
            thread.frames.push(Frame::Error(err));
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
            thread.frames.push(Frame::Error(err));
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
        $registers:ident, $ip:ident, $handlers:expr,
        cont: $cont_out:ident
    ) => {
        helpers!($instruction, $ctx, $thread, $registers, $ip, $handlers);

        let $cont_out: Continuation = $thread.top_lua().unwrap().continuation.unwrap();
        let __cur_base = $thread.top_lua().unwrap().base;

        close_upvalues($ctx.mutation(), $thread, __cur_base);
        close_tbc_vars($ctx.mutation(), $thread, __cur_base);
        $thread.frames.pop();

        let __caller_base = {
            let caller = $thread.top_lua().unwrap();
            $ip = unsafe { caller.closure.proto.code.as_ptr().add(caller.pc) };
            caller.base
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
extern "rust-preserve-none" fn cont_resume<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, cont: cont);
    apply_cont_payload!(
        cont,
        cont.results_base,
        cont.nret,
        ctx,
        thread,
        registers,
        ip,
        handlers
    );
}
