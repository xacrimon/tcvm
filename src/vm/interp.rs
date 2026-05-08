use crate::dmm::{Gc, Mutation, RefLock};
use crate::env::function::{
    Function, FunctionKind, InlineCache, LuaClosure, NativeClosure, NativeContext, Stack, Upvalue,
    UpvalueState,
};
use crate::env::shape::{MetamethodBits, Shape};
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::{Frame, LuaFrame, PendingAction, Thread, ThreadState, ThreadStatus};
use crate::env::value::{Value, ValueKind};
use crate::instruction::{Instruction, UpValueDescriptor};
use crate::lua::Context;
use crate::vm::num::{self, op_arith, op_bit};

static HANDLERS: &[Handler] = &[
    op_move,
    op_load,
    op_lfalseskip,
    op_getupval,
    op_setupval,
    op_gettabup,
    op_settabup,
    op_gettable,
    op_settable,
    op_getfield,
    op_setfield,
    op_newtable,
    op_add,
    op_sub,
    op_mul,
    op_mod,
    op_pow,
    op_div,
    op_idiv,
    op_band,
    op_bor,
    op_bxor,
    op_shl,
    op_shr,
    op_unm,
    op_bnot,
    op_not,
    op_len,
    op_concat,
    op_close,
    op_tbc,
    op_jmp,
    op_eq,
    op_lt,
    op_le,
    op_test,
    op_testset,
    op_call,
    op_tailcall,
    op_return,
    op_forloop,
    op_forprep,
    op_tforprep,
    op_tforcall,
    op_tforloop,
    op_setlist,
    op_closure,
    op_vararg,
    op_varargprep,
    op_errnnil,
    op_nop,
    op_stop,
];

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

/// Function pointer called by `op_return` when a frame's continuation is set.
/// Uses the same signature as `Handler` so it can be tail-called via `become`.
pub(crate) type ContinuationFn = Handler;

/// A pending fixup attached to a callee frame. When `op_return` sees this on
/// the current frame, it fills in `results_base` and `nret`, then tail-calls
/// `func`. The continuation reads its own data from `thread.top_lua()`,
/// pops the frame, restores caller state, does its payload-specific fixup,
/// and dispatches.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Continuation {
    pub func: ContinuationFn,
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
                        debug_assert!($ip.offset_from_unsigned(frame.closure.proto.code.as_ptr()) < frame.closure.proto.code.len());
                    }
                    let _ = $instruction;
                    let instruction = *$ip;
                    let pos = instruction.discriminant() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = *$handlers.cast::<Handler>().add(pos);
                    let ip = $ip.add(1);
                    become handler(instruction, $ctx, $thread, $registers, ip, $handlers);
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! args {
            ($$kind:path { $$($$field:ident),* }) => {{
                match $instruction {
                    $$kind { $$($$field),* } => ( $$($$field),* ),
                    _ => unsafe { std::hint::unreachable_unchecked() },
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
            ($$idx:expr) => {{
                unsafe {
                    $registers.add($$idx as usize).read()
                }
            }};

            (mut $$idx:expr) => {{
                unsafe {
                    &mut *$registers.add($$idx as usize)
                }
            }};
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

        /// Schedule a Lua metamethod call and dispatch into it. On native
        /// metamethods (not yet supported) or non-function metamethods, raises.
        #[allow(unused_macros)]
        macro_rules! invoke_metamethod {
            ($$meta:expr, $$args:expr, $$cont:expr) => {{
                match schedule_meta_call($ctx, $thread, $$meta, $$args, $$cont, $ip) {
                    Some((new_ip, new_base)) => {
                        $ip = new_ip;
                        $registers = unsafe { $thread.stack.as_mut_ptr().add(new_base) };
                        dispatch!();
                    }
                    None => raise!(),
                }
            }};
        }
    };
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
            *reg!(mut __dst_reg) = __v;
            dispatch!();
        }

        if !__t.shape().has_mm(MetamethodBits::INDEX) {
            *reg!(mut __dst_reg) = Value::nil();
            dispatch!();
        }

        // INDEX bit implies metatable is Some.
        let __mt = unsafe { __t.metatable().unwrap_unchecked() };
        match walk_index_chain(__t, __mt, __k, $ctx.symbols().mm_index) {
            IndexChain::Resolved(__rv) => {
                *reg!(mut __dst_reg) = __rv;
                dispatch!();
            }
            IndexChain::Invoke {
                func: __mm_func,
                receiver: __mm_recv,
            } => {
                let __cont = Continuation {
                    func: cont_store_result,
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
                    func: cont_ignore_result,
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
    op_nop(Instruction::NOP, ctx, &mut ts, registers, ip, handlers)
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

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
    let (dst, src) = args!(Instruction::MOVE { dst, src });
    *reg!(mut dst) = reg!(src);
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
    let (dst, idx) = args!(Instruction::LOAD { dst, idx });
    *reg!(mut dst) = constant!(idx);
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
    let src = args!(Instruction::LFALSESKIP { src });
    *reg!(mut src) = Value::boolean(false);
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
    let (dst, idx) = args!(Instruction::GETUPVAL { dst, idx });
    let uv = upvalue!(idx);
    *reg!(mut dst) = read_upvalue(thread, uv);
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
    let (src, idx) = args!(Instruction::SETUPVAL { src, idx });
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
    let (dst, idx, ic_idx, _key) = args!(Instruction::GETTABUP {
        dst,
        idx,
        ic_idx,
        key
    });
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
                *reg!(mut dst) = v;
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
    let (dst, idx, ic_idx, key) = args!(Instruction::GETTABUP {
        dst,
        idx,
        ic_idx,
        key
    });
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
    let (src, idx, ic_idx, key) = args!(Instruction::SETTABUP {
        src,
        idx,
        ic_idx,
        key
    });
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
    let (src, idx, ic_idx, key) = args!(Instruction::SETTABUP {
        src,
        idx,
        ic_idx,
        key
    });
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
    let (dst, table, key) = args!(Instruction::GETTABLE { dst, table, key });

    let Some(t) = reg!(table).get_table() else {
        raise!();
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

    *reg!(mut dst) = v;
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
    let (dst, table, key) = args!(Instruction::GETTABLE { dst, table, key });
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let k = reg!(key);
    table_get_slow_body!(ctx, thread, registers, ip, handlers, t, k, dst);
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
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });

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
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });
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
    let (dst, table, ic_idx, _key_idx) = args!(Instruction::GETFIELD {
        dst,
        table,
        ic_idx,
        key_idx
    });

    let Some(t) = reg!(table).get_table() else {
        raise!();
    };

    let cache = read_ic(thread, ic_idx);
    let t_state = t.inner().borrow();
    if let Some(slot) = ic_check(cache, t_state.shape()) {
        if slot != InlineCache::ABSENT_SLOT {
            let v = unsafe { t_state.property_at(slot) };
            if !(v.is_nil() && t_state.shape().has_mm(MetamethodBits::INDEX)) {
                drop(t_state);
                *reg!(mut dst) = v;
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
    let (dst, table, ic_idx, key_idx) = args!(Instruction::GETFIELD {
        dst,
        table,
        ic_idx,
        key_idx
    });
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let k = constant!(key_idx);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_get_slow_body!(ctx, thread, registers, ip, handlers, t, k, dst);
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
    let (src, table, ic_idx, key_idx) = args!(Instruction::SETFIELD {
        src,
        table,
        ic_idx,
        key_idx
    });

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
    let (src, table, ic_idx, key_idx) = args!(Instruction::SETFIELD {
        src,
        table,
        ic_idx,
        key_idx
    });
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let k = constant!(key_idx);
    let v = reg!(src);
    fill_ic_for_constant_key(ctx, thread, ic_idx, t, k);
    table_set_slow_body!(ctx, thread, registers, ip, handlers, t, k, v);
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
    let dst = args!(Instruction::NEWTABLE { dst });
    *reg!(mut dst) = Value::table(Table::new(ctx));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Arithmetic and bitwise (register-register)
// ---------------------------------------------------------------------------

macro_rules! binop_handler {
    ($fn_name:ident, $instr:ident, $op:ident, $num_kind:ty, $mm:ident) => {
        #[inline(never)]
        extern "rust-preserve-none" fn $fn_name<'gc>(
            instruction: Instruction,
            ctx: Context<'gc>,
            thread: &mut ThreadState<'gc>,
            mut registers: Registers<'gc, '_>,
            mut ip: *const Instruction,
            handlers: *const (),
        ) -> Result<(), Box<Error>> {
            helpers!(instruction, ctx, thread, registers, ip, handlers);
            let (dst, lhs, rhs) = args!(Instruction::$instr { dst, lhs, rhs });
            let (a, b) = (reg!(lhs), reg!(rhs));
            if let Some(v) = $op::<$num_kind>(a, b) {
                *reg!(mut dst) = v;
                dispatch!();
            }
            let meta_fn = binop_metamethod(a, b, ctx.symbols().$mm);
            if meta_fn.is_nil() {
                raise!();
            }
            let cont = Continuation {
                func: cont_store_result,
                payload: ContinuationPayload::StoreResult { dst },
                results_base: 0,
                nret: 0,
            };
            invoke_metamethod!(meta_fn, &[a, b], cont);
        }
    };
}

binop_handler!(op_add, ADD, op_arith, num::Add, mm_add);
binop_handler!(op_sub, SUB, op_arith, num::Sub, mm_sub);
binop_handler!(op_mul, MUL, op_arith, num::Mul, mm_mul);
binop_handler!(op_mod, MOD, op_arith, num::Mod, mm_mod);
binop_handler!(op_pow, POW, op_arith, num::Pow, mm_pow);
binop_handler!(op_div, DIV, op_arith, num::Div, mm_div);
binop_handler!(op_idiv, IDIV, op_arith, num::IDiv, mm_idiv);
binop_handler!(op_band, BAND, op_bit, num::BAnd, mm_band);
binop_handler!(op_bor, BOR, op_bit, num::BOr, mm_bor);
binop_handler!(op_bxor, BXOR, op_bit, num::BXor, mm_bxor);
binop_handler!(op_shl, SHL, op_bit, num::Shl, mm_shl);
binop_handler!(op_shr, SHR, op_bit, num::Shr, mm_shr);

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
    let (dst, src) = args!(Instruction::UNM { dst, src });
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(mut dst) = Value::integer(i.wrapping_neg());
        dispatch!();
    }
    if let Some(f) = val.get_float() {
        *reg!(mut dst) = Value::float(-f);
        dispatch!();
    }
    let meta_fn = unop_metamethod(val, ctx.symbols().mm_unm);
    if meta_fn.is_nil() {
        raise!();
    }
    let cont = Continuation {
        func: cont_store_result,
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
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    let val = reg!(src);
    if let Some(i) = val.get_integer() {
        *reg!(mut dst) = Value::integer(!i);
        dispatch!();
    }
    let meta_fn = unop_metamethod(val, ctx.symbols().mm_bnot);
    if meta_fn.is_nil() {
        raise!();
    }
    let cont = Continuation {
        func: cont_store_result,
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
    let (dst, src) = args!(Instruction::NOT { dst, src });
    let val = reg!(src);
    *reg!(mut dst) = Value::boolean(val.is_falsy());
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
    let (dst, src) = args!(Instruction::LEN { dst, src });
    let val = reg!(src);

    // Strings never consult __len; return byte length directly.
    if let Some(s) = val.get_string() {
        *reg!(mut dst) = Value::integer(s.len() as i64);
        dispatch!();
    }

    // Tables consult __len first; fall back to raw_len only if absent.
    let meta_fn = if let Some(t) = val.get_table() {
        let mm = t.get_metamethod(ctx.symbols().mm_len);
        if mm.is_nil() {
            *reg!(mut dst) = Value::integer(t.raw_len() as i64);
            dispatch!();
        }
        mm
    } else {
        raise!()
    };

    let cont = Continuation {
        func: cont_store_result,
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
    let (dst, lhs, rhs) = args!(Instruction::CONCAT { dst, lhs, rhs });
    let a = reg!(lhs);
    let b = reg!(rhs);
    // Fast path: both coerce to strings/numbers.
    let mut buf = Vec::new();
    if num::coerce_to_str(&mut buf, a) && num::coerce_to_str(&mut buf, b) {
        *reg!(mut dst) = Value::string(LuaString::new(ctx, &buf));
        dispatch!();
    }
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_concat);
    if meta_fn.is_nil() {
        raise!();
    }
    let cont = Continuation {
        func: cont_store_result,
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
    let start = args!(Instruction::CLOSE { start });
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
    let val = args!(Instruction::TBC { val });
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
    let offset = args!(Instruction::JMP { offset });
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
    let (lhs, rhs, inverted) = args!(Instruction::EQ { lhs, rhs, inverted });
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
                func: cont_cond_jump,
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
    let (lhs, rhs, inverted) = args!(Instruction::LT { lhs, rhs, inverted });
    let a = reg!(lhs);
    let b = reg!(rhs);
    let primitive = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
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
    };
    if let Some(r) = primitive {
        if r != inverted {
            skip!();
        }
        dispatch!();
    }
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_lt);
    if meta_fn.is_nil() {
        raise!();
    }
    let cont = Continuation {
        func: cont_cond_jump,
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
    let (lhs, rhs, inverted) = args!(Instruction::LE { lhs, rhs, inverted });
    let a = reg!(lhs);
    let b = reg!(rhs);
    let primitive = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
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
    };
    if let Some(r) = primitive {
        if r != inverted {
            skip!();
        }
        dispatch!();
    }
    let meta_fn = binop_metamethod(a, b, ctx.symbols().mm_le);
    if meta_fn.is_nil() {
        raise!();
    }
    let cont = Continuation {
        func: cont_cond_jump,
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
    let (src, inverted) = args!(Instruction::TEST { src, inverted });
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
    let (dst, src, inverted) = args!(Instruction::TESTSET { dst, src, inverted });
    let val = reg!(src);
    let truthy = !val.is_falsy();
    if truthy == inverted {
        skip!();
    } else {
        *reg!(mut dst) = val;
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
    let (func, nargs, returns) = args!(Instruction::CALL {
        func,
        args,
        returns
    });
    let base = thread.top_lua().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            let new_base = func_idx + 1;
            // Save caller's PC (offset from current code start)
            if let Some(frame) = thread.top_lua_mut() {
                let code_start = frame.closure.proto.code.as_ptr();
                frame.pc = unsafe { ip.offset_from_unsigned(code_start) };
            }
            // Ensure stack is large enough
            let needed = new_base + closure.proto.max_stack_size as usize;
            if thread.stack.len() < needed {
                thread.stack.resize(needed, Value::nil());
            }
            // Nil-fill parameter slots the caller didn't supply.
            // args == 0 means variable arg count (top-based, not yet supported).
            if nargs > 0 {
                let caller_provided = nargs as usize - 1;
                let num_params = closure.proto.num_params as usize;
                for i in caller_provided..num_params {
                    thread.stack[new_base + i] = Value::nil();
                }
            }
            // Push new call frame
            thread.push_lua(LuaFrame {
                closure,
                base: new_base,
                pc: 0,
                num_results: returns,
                continuation: None,
            });
            // Rebind ip and registers to new frame
            ip = closure.proto.code.as_ptr();
            registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        CallTarget::Native(nc) => {
            let args_base = func_idx + 1;
            let argc = if nargs == 0 {
                thread.stack.len() - args_base
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
                    let retc = thread.stack.len() - args_base;
                    // Place results at stack[func_idx..] following Lua convention.
                    let wanted = if returns == 0 {
                        retc
                    } else {
                        returns as usize - 1
                    };
                    let to_copy = retc.min(wanted);
                    // `invoke_native` truncates the stack to `args_base + retc`.
                    // For a fixed-results call, restore the caller frame's
                    // working window so the result-write loop and subsequent
                    // register accesses (through the raw `registers` pointer)
                    // stay within `Vec::len()`.
                    if returns != 0 {
                        if let Some(frame) = thread.top_lua() {
                            let needed = frame.base + frame.closure.proto.max_stack_size as usize;
                            if thread.stack.len() < needed {
                                thread.stack.resize(needed, Value::nil());
                            }
                        }
                    }
                    for i in 0..to_copy {
                        thread.stack[func_idx + i] = thread.stack[args_base + i];
                    }
                    for i in to_copy..wanted {
                        thread.stack[func_idx + i] = Value::nil();
                    }
                    if returns == 0 {
                        thread.stack.truncate(func_idx + retc);
                    }
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
                        bottom: args_base,
                        func_idx,
                        returns,
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
    let (func, nargs) = args!(Instruction::TAILCALL { func, args });
    let base = thread.top_lua().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            let cur_base = thread.top_lua().unwrap().base;
            // Close upvalues for the current frame BEFORE we overwrite
            // its stack slots with the tail-call's arguments; otherwise
            // an open upvalue pointing into this frame captures the
            // arg value instead of the local it used to reference.
            close_upvalues(ctx.mutation(), thread, cur_base);
            // Move function + arguments down to current frame's base.
            let nargs = if nargs == 0 { 0 } else { nargs as usize - 1 };
            let src_start = func_idx + 1;
            for i in 0..nargs {
                thread.stack[cur_base + i] = thread.stack[src_start + i];
            }
            // Replace current frame
            let frame = thread.top_lua_mut().unwrap();
            frame.closure = closure;
            frame.pc = 0;
            // num_results stays the same (caller's expectation)
            // Ensure stack is large enough
            let needed = cur_base + closure.proto.max_stack_size as usize;
            if thread.stack.len() < needed {
                thread.stack.resize(needed, Value::nil());
            }
            // Nil-fill parameter slots the caller didn't supply.
            let num_params = closure.proto.num_params as usize;
            for i in nargs..num_params {
                thread.stack[cur_base + i] = Value::nil();
            }
            // Rebind ip and registers
            ip = closure.proto.code.as_ptr();
            registers = unsafe { thread.stack.as_mut_ptr().add(cur_base) };
            dispatch!();
        }
        CallTarget::Native(nc) => {
            let args_base = func_idx + 1;
            let argc = if nargs == 0 {
                thread.stack.len() - args_base
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
                    let retc = thread.stack.len() - args_base;
                    match frame_return(ctx.mutation(), thread, args_base, retc) {
                        FrameReturn::Continuation(func) => {
                            become func(instruction, ctx, thread, registers, ip, handlers);
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
                        bottom: args_base,
                        func_idx: cur_base - 1,
                        returns: num_results,
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
    let (values, count) = args!(Instruction::RETURN { values, count });

    let cur_base = thread.top_lua().unwrap().base;
    let nret = if count == 0 { 0 } else { count as usize - 1 };
    let values_base = cur_base + values as usize;

    match frame_return(ctx.mutation(), thread, values_base, nret) {
        FrameReturn::Continuation(func) => {
            become func(instruction, ctx, thread, registers, ip, handlers);
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
    let (base, offset) = args!(Instruction::FORPREP { base, offset });

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
    *reg!(mut base + 3) = init;

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
    let (base, offset) = args!(Instruction::FORLOOP { base, offset });

    let step = reg!(base + 2);

    let cur = reg!(base);
    let lim_v = reg!(base + 1);
    if let (Some(i), Some(lim), Some(s)) =
        (cur.get_integer(), lim_v.get_integer(), step.get_integer())
    {
        let next = i.wrapping_add(s);
        let cont = if s > 0 { next <= lim } else { next >= lim };
        if cont {
            *reg!(mut base) = Value::integer(next);
            *reg!(mut base + 3) = Value::integer(next);
            ip = unsafe { ip.offset(offset as isize) };
        }
    } else {
        let i = to_number(cur).unwrap_or(0.0);
        let lim = to_number(lim_v).unwrap_or(0.0);
        let s = to_number(step).unwrap_or(0.0);
        let next = i + s;
        let cont = if s > 0.0 { next <= lim } else { next >= lim };
        if cont {
            *reg!(mut base) = Value::float(next);
            *reg!(mut base + 3) = Value::float(next);
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
    let (_base, offset) = args!(Instruction::TFORPREP { base, offset });
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: R[base+4], ... = R[base](R[base+1], R[base+2])
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
    let (base, count) = args!(Instruction::TFORCALL { base, count });
    let iter = reg!(base);
    let state = reg!(base + 1);
    let control = reg!(base + 2);
    let cont = Continuation {
        func: cont_tforcall,
        payload: ContinuationPayload::TForCall { base, count },
        results_base: 0,
        nret: 0,
    };
    invoke_metamethod!(iter, &[state, control], cont);
}

/// Generic for loop test: if R[base+2] != nil then R[base] = R[base+2] and jump back.
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
    let (base, offset) = args!(Instruction::TFORLOOP { base, offset });
    let control = reg!(base + 2);
    if !control.is_nil() {
        *reg!(mut base) = control;
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
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (table, count, offset) = args!(Instruction::SETLIST {
        table,
        count,
        offset
    });
    let Some(t) = reg!(table).get_table() else {
        raise!();
    };
    let n = count as usize;
    let off = offset as i64;
    for i in 1..=n {
        let val = reg!(table + i as u8);
        let key = Value::integer(off + i as i64);
        t.raw_set(ctx, key, val);
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
    let (dst, proto_idx) = args!(Instruction::CLOSURE { dst, proto });
    let frame = thread.top_lua().unwrap();
    let parent_closure = frame.closure;
    let base = frame.base;
    let proto = parent_closure.proto.prototypes[proto_idx as usize];

    // Capture upvalues based on the prototype's upvalue descriptors
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
    *reg!(mut dst) = Value::function(func);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Varargs
// ---------------------------------------------------------------------------

/// Copy varargs into R[dst], R[dst+1], ...
#[inline(never)]
extern "rust-preserve-none" fn op_vararg<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let (dst, count) = args!(Instruction::VARARG { dst, count });
    let frame = thread.top_lua().unwrap();
    let base = frame.base;
    let num_fixed = frame.closure.proto.num_params as usize;
    // Varargs are stored below base: stack[base - num_varargs .. base - num_fixed]
    // Actually they're at stack[base - num_extra .. base] where the caller put them
    // For now: the varargs sit in the slots between (base - num_extra) and base
    // The exact number of varargs depends on how many args were actually passed.
    // See #26: track actual arg count to properly copy varargs.
    let wanted = if count == 0 { 0 } else { count as usize - 1 };
    for i in 0..wanted {
        *reg!(mut dst + i as u8) = Value::nil();
    }
    dispatch!();
}

/// Adjust stack for varargs on function entry.
#[inline(never)]
extern "rust-preserve-none" fn op_varargprep<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, ctx, thread, registers, ip, handlers);
    let _num_fixed = args!(Instruction::VARARGPREP { num_fixed });
    // VARARGPREP is the first instruction of a vararg function.
    // In Lua 5.5, this adjusts the stack so that fixed params are in the
    // right place and extra args are accessible by VARARG.
    // See #26: implement vararg stack adjustment when we track actual arg count.
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
    let (src, _name_key) = args!(Instruction::ERRNNIL { src, name_key });
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
    args!(Instruction::NOP {});
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_stop<'gc>(
    instruction: Instruction,
    _ctx: Context<'gc>,
    _thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, _ctx, _thread, _registers, _ip, _handlers);
    args!(Instruction::STOP {});
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
fn close_tbc_vars<'gc>(_mc: &Mutation<'gc>, thread: &mut ThreadState<'gc>, start_idx: usize) {
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

/// Invoke a native callback. Clips the thread's stack so the callback sees
/// exactly `[args_base .. args_base + argc]` as its arguments, constructs a
/// `Stack` / `NativeContext`, and calls the function. Returns the requested
/// [`CallbackAction`]; for the hot `Return` path, results count is
/// `thread.stack.len() - args_base` after the call.
pub(crate) fn invoke_native<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<crate::vm::sequence::CallbackAction<'gc>, crate::env::Error<'gc>> {
    let end = args_base + argc;
    if thread.stack.len() > end {
        thread.stack.truncate(end);
    } else if thread.stack.len() < end {
        thread.stack.resize(end, Value::nil());
    }
    let nctx = NativeContext {
        ctx,
        upvalues: &nc.upvalues,
        exec: crate::vm::sequence::Execution::new(),
    };
    let stack = Stack::new(&mut thread.stack, args_base);
    (nc.function)(nctx, stack)
}

/// What should happen after a frame returns with values at
/// `stack[values_base .. values_base + nret]`. Produced by [`frame_return`],
/// consumed by `op_return` and the native-tailcall path in `op_tailcall`.
pub(crate) enum FrameReturn {
    /// A continuation was attached to the departing frame; caller must
    /// tail-call `func`. The continuation's `results_base` / `nret` have
    /// already been written back into the top frame.
    Continuation(ContinuationFn),
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
    let (cur_base, num_results, continuation) = {
        let f = thread.top_lua().unwrap();
        (f.base, f.num_results, f.continuation)
    };

    if let Some(mut cont) = continuation {
        cont.results_base = values_base;
        cont.nret = nret as u8;
        thread.top_lua_mut().unwrap().continuation = Some(cont);
        return FrameReturn::Continuation(cont.func);
    }

    close_upvalues(mc, thread, cur_base);
    close_tbc_vars(mc, thread, cur_base);
    thread.frames.pop();

    if thread.frames.is_empty() {
        let dst_start = cur_base - 1;
        for i in 0..nret {
            thread.stack[dst_start + i] = thread.stack[values_base + i];
        }
        thread.stack.truncate(dst_start + nret);
        thread.status = ThreadStatus::Result { bottom: dst_start };
        return FrameReturn::TopLevel;
    }

    let dst_start = cur_base - 1;
    let wanted = if num_results == 0 {
        0
    } else {
        num_results as usize - 1
    };
    let to_copy = nret.min(wanted);
    for i in 0..to_copy {
        thread.stack[dst_start + i] = thread.stack[values_base + i];
    }
    for i in to_copy..wanted {
        thread.stack[dst_start + i] = Value::nil();
    }

    // If the parent isn't a Lua frame (Sequence/WaitThread/etc.), the
    // executor driver picks up here. Place all returned values at
    // `stack[dst_start..]` for the parent's window to see and exit.
    if thread.top_lua().is_none() {
        // Use the full `nret` since the non-Lua parent doesn't have a
        // `num_results`-style truncation expectation; the driver/sequence
        // gets all the returned values.
        for i in 0..nret {
            thread.stack[dst_start + i] = thread.stack[values_base + i];
        }
        thread.stack.truncate(dst_start + nret);
        return FrameReturn::ToNonLua;
    }

    let caller = thread.top_lua().unwrap();
    let new_base = caller.base;
    let new_ip = unsafe { caller.closure.proto.code.as_ptr().add(caller.pc) };
    FrameReturn::Caller { new_base, new_ip }
}

/// Close all open upvalues pointing at stack indices >= `start_idx`.
/// Each open upvalue is converted to Closed by capturing the current stack value.
fn close_upvalues<'gc>(mc: &Mutation<'gc>, thread: &mut ThreadState<'gc>, start_idx: usize) {
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
enum IndexChain<'gc> {
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

/// Walk the `__index` chain. Caller has already verified that
/// `start.raw_get(key)` was nil and `start_metatable.has_mm(INDEX)` is
/// set, so the walk begins at `start_metatable.raw_get(__index)` and
/// follows `__index` hops from there. `mm_index_name` is the
/// pre-interned `__index` LuaString — the function never re-interns it.
#[inline]
fn walk_index_chain<'gc>(
    start: Table<'gc>,
    start_metatable: Table<'gc>,
    key: Value<'gc>,
    mm_index_name: LuaString<'gc>,
) -> IndexChain<'gc> {
    let mut current_table = start;
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
                    receiver: Value::table(current_table),
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
        current_table = next;
    }
    IndexChain::Exhausted
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
        let end = func_idx + 2 + actual_args;
        if thread.stack.len() < end {
            thread.stack.resize(end, Value::nil());
        }
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

/// Set up a Lua call frame to invoke a metamethod (or other helper function),
/// attaching a post-return continuation. The function + args are placed above
/// the caller's max_stack_size so no live register is clobbered, then
/// `resolve_call_chain` walks any `__call` hops until it reaches a Lua
/// closure. On success, returns `(new_ip, new_base)` which the caller should
/// use to rebind `ip`/`registers` before dispatching. Returns `None` when the
/// chain is unresolvable — non-callable value, native target (see #32), or
/// depth exhaustion — in which case callers raise.
#[inline(never)]
fn schedule_meta_call<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    meta_fn: Value<'gc>,
    args: &[Value<'gc>],
    cont: Continuation,
    caller_ip: *const Instruction,
) -> Option<(*const Instruction, usize)> {
    // Save caller's pc; no decisions here depend on knowing the final closure.
    if let Some(frame) = thread.top_lua_mut() {
        let code_start = frame.closure.proto.code.as_ptr();
        frame.pc = unsafe { caller_ip.offset_from_unsigned(code_start) };
    }

    // schedule_meta_call is only reachable from inside a handler via
    // invoke_metamethod!, so an active caller frame is always present.
    let caller = thread
        .top_lua()
        .expect("schedule_meta_call called without an active Lua frame");
    let scratch_func = caller.base + caller.closure.proto.max_stack_size as usize;
    let new_base = scratch_func + 1;

    // Stage meta_fn + args so resolve_call_chain sees them in op_call layout.
    let staged_end = new_base + args.len();
    if thread.stack.len() < staged_end {
        thread.stack.resize(staged_end, Value::nil());
    }
    thread.stack[scratch_func] = meta_fn;
    for (i, &a) in args.iter().enumerate() {
        thread.stack[new_base + i] = a;
    }

    // Walk any __call chain. nargs follows op_call's convention (includes the
    // function slot), so `args.len() + 1`.
    debug_assert!(args.len() < u8::MAX as usize);
    let nargs = (args.len() + 1) as u8;
    let (target, final_nargs) = resolve_call_chain(ctx, thread, scratch_func, nargs)?;
    // See #32: support native metamethod targets. Today `__call`/`__index`/etc.
    // resolving to a native callback is rejected — handling it requires either
    // a Callback-kind frame or a per-continuation native dispatch path.
    let closure = match target {
        CallTarget::Lua(c) => c,
        CallTarget::Native(_) => return None,
    };
    let actual_args = final_nargs as usize - 1;

    // Grow stack to fit the resolved closure's full frame.
    let needed = new_base + closure.proto.max_stack_size as usize;
    if thread.stack.len() < needed {
        thread.stack.resize(needed, Value::nil());
    }

    // Nil-fill any parameter slots not covered by the (possibly shifted) args.
    let num_params = closure.proto.num_params as usize;
    for i in actual_args..num_params {
        thread.stack[new_base + i] = Value::nil();
    }

    thread.push_lua(LuaFrame {
        closure,
        base: new_base,
        // Ignored by op_return when a continuation is set — the continuation
        // reads return values directly from the stack via `cont.results_base`.
        pc: 0,
        num_results: 0,
        continuation: Some(cont),
    });

    let new_ip = closure.proto.code.as_ptr();
    Some((new_ip, new_base))
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

/// Continuation for unary and binary metamethods that produce a single result
/// written into the scheduling handler's `R[dst]`. Pops the metamethod's
/// frame, restores the caller, stores the result, and dispatches.
#[inline(never)]
extern "rust-preserve-none" fn cont_store_result<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, cont: cont);

    let dst = match cont.payload {
        ContinuationPayload::StoreResult { dst } => dst,
        _ => unsafe { std::hint::unreachable_unchecked() },
    };

    let result = if cont.nret > 0 {
        thread.stack[cont.results_base]
    } else {
        Value::nil()
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// Continuation that discards results — used by `__newindex` and `__close`,
/// which are invoked for their side effects.
#[inline(never)]
extern "rust-preserve-none" fn cont_ignore_result<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, cont: _cont);
    dispatch!();
}

/// Continuation for comparison metamethods (`__eq`, `__lt`, `__le`). Coerces
/// the result to a bool and, if it matches the expected sense, advances `ip`
/// by `offset` — which for the current comparison ops is `1`, effectively
/// skipping the adjacent `JMP` (matching the fast-path `skip!()` behavior).
#[inline(never)]
extern "rust-preserve-none" fn cont_cond_jump<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, cont: cont);

    let (offset, inverted) = match cont.payload {
        ContinuationPayload::CondJump { offset, inverted } => (offset, inverted),
        _ => unsafe { std::hint::unreachable_unchecked() },
    };

    let result = if cont.nret > 0 {
        thread.stack[cont.results_base]
    } else {
        Value::nil()
    };
    let truthy = !result.is_falsy();
    if truthy != inverted {
        ip = unsafe { ip.offset(offset as isize) };
    }
    dispatch!();
}

/// Continuation for generic-for (`TFORCALL`): copy up to `count` results
/// into `R[base+4..]`, nil-filling the shortfall. `TFORLOOP` follows
/// immediately after and handles the termination check.
#[inline(never)]
extern "rust-preserve-none" fn cont_tforcall<'gc>(
    instruction: Instruction,
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    mut registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    finalize_return!(instruction, ctx, thread, registers, ip, handlers, cont: cont);

    let (base, count) = match cont.payload {
        ContinuationPayload::TForCall { base, count } => (base, count),
        _ => unsafe { std::hint::unreachable_unchecked() },
    };
    debug_assert!(
        base as usize + 4 + count as usize <= u8::MAX as usize + 1,
        "TFORCALL destination range exceeds u8 register space",
    );

    let to_copy = (cont.nret as usize).min(count as usize);
    for i in 0..to_copy {
        *reg!(mut base + 4 + i as u8) = thread.stack[cont.results_base + i];
    }
    for i in to_copy..count as usize {
        *reg!(mut base + 4 + i as u8) = Value::nil();
    }
    dispatch!();
}
