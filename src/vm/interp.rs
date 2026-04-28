use crate::dmm::{Gc, Mutation, RefLock};
use crate::env::function::{
    Function, FunctionKind, LuaClosure, NativeClosure, NativeContext, NativeError, Stack, Upvalue,
    UpvalueState,
};
use crate::env::string::LuaString;
use crate::env::table::{Metamethod, Table};
use crate::env::thread::{CallFrame, Thread, ThreadState, ThreadStatus};
use crate::env::{Value, ValueKind};
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
/// `func`. The continuation reads its own data from `thread.frames.last()`,
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
                        let frame = $thread.frames.last().unwrap_unchecked();
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
                    let frame = $thread.frames.last().unwrap_unchecked();
                    *frame.closure.proto.constants.get_unchecked($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! field_constant {
            ($$idx:expr) => {{
                unsafe {
                    let frame = $thread.frames.last().unwrap_unchecked();
                    *frame
                        .closure
                        .proto
                        .field_constants
                        .get_unchecked($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                unsafe {
                    let frame = $thread.frames.last().unwrap_unchecked();
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

/// Drive the VM on `thread` until the top-level frame returns.
///
/// The caller must have seeded the thread with at least one `CallFrame`,
/// sized `stack` to at least `base + max_stack_size`, and placed
/// the callee + arguments at `stack[base-1..]`. See `Executor::start`.
#[inline(never)]
pub(crate) fn run_thread<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) -> Result<(), Box<Error>> {
    let mut ts = thread.borrow_mut(ctx.mutation());
    let (ip, base) = {
        let frame = ts
            .frames
            .last()
            .expect("run_thread requires a seeded frame");
        let code_ptr = frame.closure.proto.code.as_ptr();
        let ip = unsafe { code_ptr.add(frame.pc) };
        (ip, frame.base)
    };
    let registers = unsafe { ts.stack.as_mut_ptr().add(base) };
    let handlers = HANDLERS.as_ptr() as *const ();
    op_nop(Instruction::NOP, ctx, &mut *ts, registers, ip, handlers)
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
    let (dst, idx, key) = args!(Instruction::GETTABUP { dst, idx, key });
    let uv = upvalue!(idx);
    let t = read_upvalue(thread, uv);

    let Some(t) = t.get_table() else {
        raise!();
    };

    let t = t.inner().borrow();

    if t.has_metamethod(Metamethod::INDEX) {
        become gettabup_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let fc = field_constant!(key);
    let k = Value::string(fc.name);
    let v = t.raw_get(k);
    *reg!(mut dst) = v;
    dispatch!();
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
    todo!()
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
    let (src, idx, key) = args!(Instruction::SETTABUP { src, idx, key });
    let uv = upvalue!(idx);
    let t = read_upvalue(thread, uv);

    let Some(t) = t.get_table() else {
        raise!();
    };

    let mut t = t.inner().borrow_mut(ctx.mutation());

    if t.has_metamethod(Metamethod::NEWINDEX) {
        become settabup_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let fc = field_constant!(key);
    let k = Value::string(fc.name);
    let v = reg!(src);
    t.raw_set(k, v);
    dispatch!()
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
    todo!()
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

    let t = t.inner().borrow();

    if t.has_metamethod(Metamethod::INDEX) {
        become gettable_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let k = reg!(key);
    let v = t.raw_get(k);
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
    todo!()
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

    let mut t = t.inner().borrow_mut(ctx.mutation());

    if t.has_metamethod(Metamethod::NEWINDEX) {
        become settable_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let k = reg!(key);
    let v = reg!(src);
    t.raw_set(k, v);
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
    todo!()
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
    let (dst, table, key_idx) = args!(Instruction::GETFIELD {
        dst,
        table,
        key_idx
    });

    let Some(t) = reg!(table).get_table() else {
        raise!();
    };

    let t = t.inner().borrow();

    if t.has_metamethod(Metamethod::INDEX) {
        become getfield_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let fc = field_constant!(key_idx);
    let k = Value::string(fc.name);
    let v = t.raw_get(k);
    *reg!(mut dst) = v;
    dispatch!();
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
    todo!()
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
    let (src, table, key_idx) = args!(Instruction::SETFIELD {
        src,
        table,
        key_idx
    });

    let Some(t) = reg!(table).get_table() else {
        raise!();
    };

    let mut t = t.inner().borrow_mut(ctx.mutation());

    if t.has_metamethod(Metamethod::NEWINDEX) {
        become setfield_slow(instruction, ctx, thread, registers, ip, handlers);
    }

    let fc = field_constant!(key_idx);
    let k = Value::string(fc.name);
    let v = reg!(src);
    t.raw_set(k, v);
    dispatch!()
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
    todo!()
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
    *reg!(mut dst) = Value::table(Table::new(ctx.mutation()));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Arithmetic and bitwise (register-register)
// ---------------------------------------------------------------------------

macro_rules! binop_handler {
    ($fn_name:ident, $instr:ident, $op:ident, $num_kind:ty, $mm:expr) => {
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
            let meta_fn = binop_metamethod(ctx, a, b, $mm);
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

binop_handler!(op_add, ADD, op_arith, num::Add, b"__add");
binop_handler!(op_sub, SUB, op_arith, num::Sub, b"__sub");
binop_handler!(op_mul, MUL, op_arith, num::Mul, b"__mul");
binop_handler!(op_mod, MOD, op_arith, num::Mod, b"__mod");
binop_handler!(op_pow, POW, op_arith, num::Pow, b"__pow");
binop_handler!(op_div, DIV, op_arith, num::Div, b"__div");
binop_handler!(op_idiv, IDIV, op_arith, num::IDiv, b"__idiv");
binop_handler!(op_band, BAND, op_bit, num::BAnd, b"__band");
binop_handler!(op_bor, BOR, op_bit, num::BOr, b"__bor");
binop_handler!(op_bxor, BXOR, op_bit, num::BXor, b"__bxor");
binop_handler!(op_shl, SHL, op_bit, num::Shl, b"__shl");
binop_handler!(op_shr, SHR, op_bit, num::Shr, b"__shr");

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
    let meta_fn = unop_metamethod(ctx, val, b"__unm");
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
    let meta_fn = unop_metamethod(ctx, val, b"__bnot");
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
        let mm = t.get_metamethod(ctx, b"__len");
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
    let meta_fn = binop_metamethod(ctx, a, b, b"__concat");
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
    let base = thread.frames.last().map_or(0, |f| f.base);
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
    let base = thread.frames.last().map_or(0, |f| f.base);
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
        let meta_fn = binop_metamethod(ctx, a, b, b"__eq");
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
    let meta_fn = binop_metamethod(ctx, a, b, b"__lt");
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
    let meta_fn = binop_metamethod(ctx, a, b, b"__le");
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
    let base = thread.frames.last().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            let new_base = func_idx + 1;
            // Save caller's PC (offset from current code start)
            if let Some(frame) = thread.frames.last_mut() {
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
            thread.frames.push(CallFrame {
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
            let retc = match invoke_native(ctx, thread, nc, args_base, argc) {
                Ok(n) => n,
                Err(_) => raise!(),
            };
            // Place results at stack[func_idx..] following Lua convention.
            let wanted = if returns == 0 {
                retc
            } else {
                returns as usize - 1
            };
            let to_copy = retc.min(wanted);
            // `invoke_native` truncates the stack to `args_base + retc`. For a
            // fixed-results call, restore the caller frame's working window so
            // the result-write loop and subsequent register accesses (through
            // the raw `registers` pointer) stay within `Vec::len()`.
            if returns != 0 {
                if let Some(frame) = thread.frames.last() {
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
    let base = thread.frames.last().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let Some((target, nargs)) = resolve_call_chain(ctx, thread, func_idx, nargs) else {
        raise!();
    };

    match target {
        CallTarget::Lua(closure) => {
            let cur_base = thread.frames.last().unwrap().base;
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
            let frame = thread.frames.last_mut().unwrap();
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
            let retc = match invoke_native(ctx, thread, nc, args_base, argc) {
                Ok(n) => n,
                Err(_) => raise!(),
            };
            match frame_return(ctx.mutation(), thread, args_base, retc) {
                FrameReturn::Continuation(func) => {
                    become func(instruction, ctx, thread, registers, ip, handlers);
                }
                FrameReturn::TopLevel => return Ok(()),
                FrameReturn::Caller { new_base, new_ip } => {
                    ip = new_ip;
                    registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
                    dispatch!();
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

    let cur_base = thread.frames.last().unwrap().base;
    let nret = if count == 0 { 0 } else { count as usize - 1 };
    let values_base = cur_base + values as usize;

    match frame_return(ctx.mutation(), thread, values_base, nret) {
        FrameReturn::Continuation(func) => {
            become func(instruction, ctx, thread, registers, ip, handlers);
        }
        FrameReturn::TopLevel => return Ok(()),
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
    let t = reg!(table).get_table().unwrap();
    let n = count as usize;
    let off = offset as i64;
    for i in 1..=n {
        let val = reg!(table + i as u8);
        let key = Value::integer(off + i as i64);
        t.raw_set(ctx.mutation(), key, val);
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
    let frame = thread.frames.last().unwrap();
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
    let frame = thread.frames.last().unwrap();
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
/// `Stack` / `NativeContext`, and calls the function. On return, any values
/// the callback left on the stack above `args_base` are its return values;
/// the count is `thread.stack.len() - args_base`.
pub(crate) fn invoke_native<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    nc: &NativeClosure<'gc>,
    args_base: usize,
    argc: usize,
) -> Result<usize, NativeError> {
    let end = args_base + argc;
    if thread.stack.len() > end {
        thread.stack.truncate(end);
    } else if thread.stack.len() < end {
        thread.stack.resize(end, Value::nil());
    }
    let ctx = NativeContext {
        ctx,
        upvalues: &nc.upvalues,
    };
    let stack = Stack::new(&mut thread.stack, args_base);
    (nc.function)(ctx, stack)?;
    Ok(thread.stack.len() - args_base)
}

/// What should happen after a frame returns with values at
/// `stack[values_base .. values_base + nret]`. Produced by [`frame_return`],
/// consumed by `op_return` and the native-tailcall path in `op_tailcall`.
pub(crate) enum FrameReturn {
    /// A continuation was attached to the departing frame; caller must
    /// tail-call `func`. The continuation's `results_base` / `nret` have
    /// already been written back into the top frame.
    Continuation(ContinuationFn),
    /// The departing frame was the outermost one; thread is now `Dead`.
    /// Caller should return `Ok(())` from the handler.
    TopLevel,
    /// Normal return to the caller frame, which has been restored to the
    /// top of the frame stack. Caller updates `ip` / `registers` and
    /// dispatches.
    Caller {
        new_base: usize,
        new_ip: *const Instruction,
    },
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
        let f = thread.frames.last().unwrap();
        (f.base, f.num_results, f.continuation)
    };

    if let Some(mut cont) = continuation {
        cont.results_base = values_base;
        cont.nret = nret as u8;
        thread.frames.last_mut().unwrap().continuation = Some(cont);
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
        thread.status = ThreadStatus::Dead;
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

    let caller = thread.frames.last().unwrap();
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
/// give up and raise (matches Lua's `MAXTAGLOOP`).
const MAX_TAG_LOOP: usize = 200;

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
}

/// Walk the `__index` chain starting from `table`. Returns the resolved
/// value or a pending function call. `None` means the chain exceeded
/// `MAX_TAG_LOOP` and the caller should raise.
#[inline]
fn resolve_index_chain<'gc>(
    ctx: Context<'gc>,
    table: Table<'gc>,
    key: Value<'gc>,
) -> Option<IndexChain<'gc>> {
    let mut t = table;
    for _ in 0..MAX_TAG_LOOP {
        let v = t.raw_get(key);
        if !v.is_nil() {
            return Some(IndexChain::Resolved(v));
        }
        let mm = t.get_metamethod(ctx, b"__index");
        if mm.is_nil() {
            return Some(IndexChain::Resolved(Value::nil()));
        }
        if let Some(next) = mm.get_table() {
            t = next;
            continue;
        }
        return Some(IndexChain::Invoke {
            func: mm,
            receiver: Value::table(t),
        });
    }
    None
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
}

/// Walk the `__newindex` chain. If the key already exists in `table`, do a
/// raw set there. Otherwise follow `__newindex` tables; terminate at the
/// first callable or at a table that has the key (or has no `__newindex`).
/// `None` means the chain exceeded `MAX_TAG_LOOP` and the caller should raise.
#[inline]
fn resolve_newindex_chain<'gc>(
    ctx: Context<'gc>,
    table: Table<'gc>,
    key: Value<'gc>,
) -> Option<NewIndexChain<'gc>> {
    let mut t = table;
    for _ in 0..MAX_TAG_LOOP {
        // If the key already has a value, skip __newindex and raw_set here.
        if !t.raw_get(key).is_nil() {
            return Some(NewIndexChain::RawSet(t));
        }
        let mm = t.get_metamethod(ctx, b"__newindex");
        if mm.is_nil() {
            return Some(NewIndexChain::RawSet(t));
        }
        if let Some(next) = mm.get_table() {
            t = next;
            continue;
        }
        return Some(NewIndexChain::Invoke {
            func: mm,
            receiver: Value::table(t),
        });
    }
    None
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
            Some(t) => t.get_metamethod(ctx, b"__call"),
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
/// metatables on tables; userdata metatables and the string metatable are
/// pending those subsystems (see #47).
#[inline]
fn binop_metamethod<'gc>(
    ctx: Context<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
    name: &[u8],
) -> Value<'gc> {
    if let Some(t) = lhs.get_table() {
        let m = t.get_metamethod(ctx, name);
        if !m.is_nil() {
            return m;
        }
    }
    if let Some(t) = rhs.get_table() {
        return t.get_metamethod(ctx, name);
    }
    Value::nil()
}

/// Look up a unary metamethod on `val`. Same caveat as `binop_metamethod`.
#[inline]
fn unop_metamethod<'gc>(ctx: Context<'gc>, val: Value<'gc>, name: &[u8]) -> Value<'gc> {
    if let Some(t) = val.get_table() {
        return t.get_metamethod(ctx, name);
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
    if let Some(frame) = thread.frames.last_mut() {
        let code_start = frame.closure.proto.code.as_ptr();
        frame.pc = unsafe { caller_ip.offset_from_unsigned(code_start) };
    }

    // schedule_meta_call is only reachable from inside a handler via
    // invoke_metamethod!, so an active caller frame is always present.
    let caller = thread
        .frames
        .last()
        .expect("schedule_meta_call called without an active frame");
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

    thread.frames.push(CallFrame {
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

        let $cont_out: Continuation = $thread.frames.last().unwrap().continuation.unwrap();
        let __cur_base = $thread.frames.last().unwrap().base;

        close_upvalues($ctx.mutation(), $thread, __cur_base);
        close_tbc_vars($ctx.mutation(), $thread, __cur_base);
        $thread.frames.pop();

        let __caller_base = {
            let caller = $thread.frames.last().unwrap();
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
