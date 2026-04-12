use crate::instruction::Instruction;
use crate::vm::num::{self, op_arith, op_bit};
use crate::env::Value;
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::function::{Function, FunctionKind, UpvalueState};
use crate::env::thread::{CallFrame, ThreadState, ThreadStatus};
use crate::dmm::Mutation;

const HANDLERS: &[Handler] = &[
    op_move,
    op_load,
    op_lfalseskip,
    op_getupval,
    op_setupval,
    op_gettabup,
    op_settabup,
    op_gettable,
    op_settable,
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
    op_mmbin,
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
    op_nop,
    op_stop,
];

#[derive(Debug)]
struct Error {
    pc: usize,
}

#[cfg(debug_assertions)]
type Registers<'gc, 'a> = &'a mut [Value<'gc>];

#[cfg(not(debug_assertions))]
type Registers<'gc, 'a> = *mut Value<'gc>;

type Handler = extern "rust-preserve-none" fn(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'_, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

macro_rules! helpers {
    ($instruction:expr, $thread:expr, $registers:expr, $ip:expr, $handlers:expr) => {
        #[allow(unused_macros)]
        macro_rules! dispatch {
            () => {{
                unsafe {
                    let _ = $instruction;
                    debug_assert!($ip.offset_from_unsigned($thread.tape.as_ptr()) < $thread.tape.len());
                    let instruction = *$ip;
                    let pos = instruction.discriminant() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = *$handlers.cast::<Handler>().add(pos);
                    let ip = $ip.add(1);
                    become handler(instruction, $thread, $registers, ip, $handlers);
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
                become impl_error($instruction, $thread, $registers, $ip, $handlers);
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
                #[cfg(debug_assertions)]
                {
                    $registers[$$idx as usize]
                }

                #[cfg(not(debug_assertions))]
                unsafe {
                    $registers.add($$idx as usize).read()
                }
            }};

            (mut $$idx:expr) => {{
                #[cfg(debug_assertions)]
                {
                    &mut $registers[$$idx as usize]
                }

                #[cfg(not(debug_assertions))]
                unsafe {
                    &mut *$registers.add($$idx as usize)
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! constant {
            ($$idx:expr) => {{
                let frame = $thread.frames.last().unwrap();
                match &*frame.function.inner() {
                    FunctionKind::Lua(cl) => cl.proto.constants[$$idx as usize],
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                let frame = $thread.frames.last().unwrap();
                match &*frame.function.inner() {
                    FunctionKind::Lua(cl) => cl.upvalues[$$idx as usize],
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }};
        }

        #[allow(unused_macros)]
        macro_rules! skip {
            () => {{
                $ip = unsafe { $ip.add(1) };
            }};
        }
    };
}

#[inline(never)]
pub fn run(tape: &[Instruction], thread: &mut ThreadState<'_>) {
    let ip = tape.as_ptr();
    let handlers = HANDLERS.as_ptr() as *const ();

    #[cfg(debug_assertions)]
    let registers = &mut [];

    #[cfg(not(debug_assertions))]
    let registers = std::ptr::null_mut();

    op_nop(Instruction::NOP, thread, registers, ip, handlers).unwrap();
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[cold]
#[inline(never)]
extern "rust-preserve-none" fn impl_error<'gc>(
    _instruction: Instruction,
    thread: &mut ThreadState<'_>,
    _registers: Registers<'gc, '_>,
    ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    // TODO: compute proper PC from current frame's prototype code base
    Err(Box::new(Error { pc: 0 }))
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_move<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::MOVE { dst, src });
    *reg!(mut dst) = reg!(src);
    dispatch!();
}

/// Load constant from the current prototype's constant pool.
#[inline(never)]
extern "rust-preserve-none" fn op_load<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::LOAD { dst, idx });
    *reg!(mut dst) = constant!(idx);
    dispatch!();
}

/// Set register to false and skip the next instruction.
#[inline(never)]
extern "rust-preserve-none" fn op_lfalseskip<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let src = args!(Instruction::LFALSESKIP { src });
    *reg!(mut src) = Value::Boolean(false);
    skip!();
    dispatch!();
}

// ---------------------------------------------------------------------------
// Upvalue access
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_getupval<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::GETUPVAL { dst, idx });
    let uv = upvalue!(idx);
    let val = match &*uv.borrow() {
        UpvalueState::Open { thread: t, index } => t.borrow().stack[*index],
        UpvalueState::Closed(v) => *v,
    };
    *reg!(mut dst) = val;
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_setupval<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, idx) = args!(Instruction::SETUPVAL { src, idx });
    let val = reg!(src);
    let uv = upvalue!(idx);
    // TODO: needs &Mutation for borrow_mut
    // let mut uv_ref = uv.borrow_mut(mc);
    // match &mut *uv_ref {
    //     UpvalueState::Open { thread: t, index } => { t.borrow_mut(mc).stack[*index] = val; }
    //     UpvalueState::Closed(v) => *v = val,
    // }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table access via upvalue
// ---------------------------------------------------------------------------

/// R[dst] = UpValue[idx][K[key]]
#[inline(never)]
extern "rust-preserve-none" fn op_gettabup<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx, key) = args!(Instruction::GETTABUP { dst, idx, key });
    let uv = upvalue!(idx);
    let table_val = match &*uv.borrow() {
        UpvalueState::Open { thread: t, index } => t.borrow().stack[*index],
        UpvalueState::Closed(v) => *v,
    };
    let table = table_val.get_table().unwrap();
    let key = constant!(key);
    *reg!(mut dst) = table.raw_get(key);
    dispatch!();
}

/// UpValue[idx][K[key]] = R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_settabup<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, idx, key) = args!(Instruction::SETTABUP { src, idx, key });
    let uv = upvalue!(idx);
    let table_val = match &*uv.borrow() {
        UpvalueState::Open { thread: t, index } => t.borrow().stack[*index],
        UpvalueState::Closed(v) => *v,
    };
    let table = table_val.get_table().unwrap();
    let key = constant!(key);
    let val = reg!(src);
    // TODO: needs &Mutation for raw_set
    // table.raw_set(mc, key, val);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table access via register
// ---------------------------------------------------------------------------

/// R[dst] = R[table][R[key]]
#[inline(never)]
extern "rust-preserve-none" fn op_gettable<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, table, key) = args!(Instruction::GETTABLE { dst, table, key });
    let t = reg!(table).get_table().unwrap();
    let k = reg!(key);
    // TODO: __index metamethod fallback
    *reg!(mut dst) = t.raw_get(k);
    dispatch!();
}

/// R[table][R[key]] = R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_settable<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });
    let _t = reg!(table).get_table().unwrap();
    let _k = reg!(key);
    let _v = reg!(src);
    // TODO: needs &Mutation for raw_set
    // t.raw_set(mc, k, v);
    dispatch!();
}

/// R[dst] = {}
#[inline(never)]
extern "rust-preserve-none" fn op_newtable<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let _dst = args!(Instruction::NEWTABLE { dst });
    // TODO: needs &Mutation for Table::new
    // *reg!(mut dst) = Value::Table(Table::new(mc));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Arithmetic (register-register)
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_add<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::ADD { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Add>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    // TODO: if None, next instruction should be MMBIN for metamethod fallback
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_sub<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::SUB { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Sub>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_mul<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::MUL { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Mul>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_mod<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::MOD { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Mod>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_pow<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::POW { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Pow>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_div<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::DIV { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::Div>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_idiv<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::IDIV { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_arith::<num::IDiv>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Bitwise (register-register)
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_band<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::BAND { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_bit::<num::BAnd>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_bor<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::BOR { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_bit::<num::BOr>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_bxor<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::BXOR { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_bit::<num::BXor>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_shl<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::SHL { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_bit::<num::Shl>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_shr<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::SHR { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if let Some(v) = op_bit::<num::Shr>(lhs, rhs) {
        *reg!(mut dst) = v;
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Metamethod fallback
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_mmbin<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_lhs, _rhs, _metamethod) = args!(Instruction::MMBIN {
        lhs,
        rhs,
        metamethod
    });
    // TODO: look up metamethod on the operands' metatables and invoke it.
    raise!();
}

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

/// R[dst] = -R[src]
#[inline(never)]
extern "rust-preserve-none" fn op_unm<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::UNM { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::Integer(i) => Value::Integer(i.wrapping_neg()),
        Value::Float(f) => Value::Float(-f),
        // TODO: __unm metamethod
        _ => { raise!(); }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = ~R[src]  (bitwise NOT)
#[inline(never)]
extern "rust-preserve-none" fn op_bnot<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::Integer(i) => Value::Integer(!i),
        // TODO: __bnot metamethod
        _ => { raise!(); }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = not R[src]  (logical NOT — always produces boolean)
#[inline(never)]
extern "rust-preserve-none" fn op_not<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::NOT { dst, src });
    let val = reg!(src);
    *reg!(mut dst) = Value::Boolean(val.is_falsy());
    dispatch!();
}

/// R[dst] = #R[src]  (length)
#[inline(never)]
extern "rust-preserve-none" fn op_len<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::LEN { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::String(s) => Value::Integer(s.len() as i64),
        Value::Table(t) => Value::Integer(t.raw_len() as i64),
        // TODO: __len metamethod
        _ => { raise!(); }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = R[lhs] .. R[rhs]  (string concatenation)
#[inline(never)]
extern "rust-preserve-none" fn op_concat<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::CONCAT { dst, lhs, rhs });
    let _a = reg!(lhs);
    let _b = reg!(rhs);
    // TODO: needs &Mutation for LuaString::new
    // Also: number-to-string coercion, __concat metamethod
    // match (a, b) {
    //     (Value::String(a), Value::String(b)) => {
    //         let mut buf = Vec::with_capacity(a.len() + b.len());
    //         buf.extend_from_slice(a.as_bytes());
    //         buf.extend_from_slice(b.as_bytes());
    //         *reg!(mut dst) = Value::String(LuaString::new(mc, &buf));
    //     }
    //     _ => { raise!(); }
    // }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Upvalue / resource management
// ---------------------------------------------------------------------------

/// Close all upvalues >= R[start].
#[inline(never)]
extern "rust-preserve-none" fn op_close<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let _start = args!(Instruction::CLOSE { start });
    // TODO: needs &Mutation for upvalue borrow_mut
    // let base = thread.frames.last().map_or(0, |f| f.base);
    // let start_idx = base + start as usize;
    // Close all open upvalues pointing at stack indices >= start_idx
    // by capturing the current stack value into UpvalueState::Closed.
    dispatch!();
}

/// Mark R[val] as to-be-closed.
#[inline(never)]
extern "rust-preserve-none" fn op_tbc<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let _val = args!(Instruction::TBC { val });
    // TODO: mark variable as to-be-closed for __close metamethod
    dispatch!();
}

// ---------------------------------------------------------------------------
// Jumps and conditionals
// ---------------------------------------------------------------------------

/// pc += offset
#[inline(never)]
extern "rust-preserve-none" fn op_jmp<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let offset = args!(Instruction::JMP { offset });
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// if (R[lhs] == R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_eq<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::EQ { lhs, rhs, inverted });
    let equal = reg!(lhs) == reg!(rhs);
    if equal != inverted {
        skip!();
    }
    dispatch!();
}

/// if (R[lhs] < R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_lt<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::LT { lhs, rhs, inverted });
    let a = reg!(lhs);
    let b = reg!(rhs);
    let result = match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => a < b,
        (Value::Float(a), Value::Float(b)) => a < b,
        (Value::Integer(a), Value::Float(b)) => (a as f64) < b,
        (Value::Float(a), Value::Integer(b)) => a < (b as f64),
        (Value::String(a), Value::String(b)) => a < b,
        // TODO: __lt metamethod
        _ => { raise!(); }
    };
    if result != inverted {
        skip!();
    }
    dispatch!();
}

/// if (R[lhs] <= R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_le<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::LE { lhs, rhs, inverted });
    let a = reg!(lhs);
    let b = reg!(rhs);
    let result = match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => a <= b,
        (Value::Float(a), Value::Float(b)) => a <= b,
        (Value::Integer(a), Value::Float(b)) => (a as f64) <= b,
        (Value::Float(a), Value::Integer(b)) => a <= (b as f64),
        (Value::String(a), Value::String(b)) => a <= b,
        // TODO: __le metamethod
        _ => { raise!(); }
    };
    if result != inverted {
        skip!();
    }
    dispatch!();
}

/// if (not R[src]) == inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_test<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, inverted) = args!(Instruction::TEST { src, inverted });
    let truthy = !reg!(src).is_falsy();
    if truthy != inverted {
        skip!();
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
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_func, _args, _returns) = args!(Instruction::CALL {
        func,
        args,
        returns
    });
    // TODO: full call implementation requires &Mutation and careful
    // register/ip rebinding. Rough logic:
    //
    // 1. let func_val = reg!(func).get_function().unwrap()
    // 2. For Lua closures:
    //    - Save caller PC in current frame
    //    - Push new CallFrame { function, base: func_idx+1, pc: 0, num_results: returns }
    //    - Ensure stack has room (resize if needed)
    //    - Set ip = proto.code.as_ptr()
    //    - Rebind registers to new base
    //    - Dispatch first instruction
    // 3. For native closures:
    //    - Call the native fn with arguments
    //    - Place results in registers
    dispatch!();
}

/// return R[func](R[func+1], ..., R[func+args-1])  — tail call
#[inline(never)]
extern "rust-preserve-none" fn op_tailcall<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_func, _args) = args!(Instruction::TAILCALL { func, args });
    // TODO: like CALL but reuses the current frame:
    //
    // 1. Move arguments down to current frame's base
    // 2. Replace current frame's function/pc
    // 3. Set ip to new proto's code, rebind registers
    // 4. Dispatch
    dispatch!();
}

/// return R[values], ..., R[values+count-2]
#[inline(never)]
extern "rust-preserve-none" fn op_return<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_values, _count) = args!(Instruction::RETURN { values, count });
    // TODO: full return implementation. Rough logic:
    //
    // 1. Pop current frame
    // 2. If no caller frame -> set status = Dead, return Ok(())
    // 3. Copy nresults return values into caller's expected slots
    // 4. Restore caller's ip and registers base
    // 5. Dispatch next instruction in caller
    dispatch!();
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
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (base, offset) = args!(Instruction::FORPREP { base, offset });

    let init = reg!(base);
    let limit = reg!(base + 1);
    let step = reg!(base + 2);

    let should_run = match (init, limit, step) {
        (Value::Integer(i), Value::Integer(lim), Value::Integer(s)) => {
            if s > 0 { i <= lim } else { i >= lim }
        }
        _ => {
            let i = to_number(init).unwrap_or(0.0);
            let lim = to_number(limit).unwrap_or(0.0);
            let s = to_number(step).unwrap_or(0.0);
            if s > 0.0 { i <= lim } else { i >= lim }
        }
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
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (base, offset) = args!(Instruction::FORLOOP { base, offset });

    let step = reg!(base + 2);

    match (reg!(base), reg!(base + 1), step) {
        (Value::Integer(i), Value::Integer(lim), Value::Integer(s)) => {
            let next = i.wrapping_add(s);
            let cont = if s > 0 { next <= lim } else { next >= lim };
            if cont {
                *reg!(mut base) = Value::Integer(next);
                *reg!(mut base + 3) = Value::Integer(next);
                ip = unsafe { ip.offset(offset as isize) };
            }
        }
        _ => {
            let i = to_number(reg!(base)).unwrap_or(0.0);
            let lim = to_number(reg!(base + 1)).unwrap_or(0.0);
            let s = to_number(step).unwrap_or(0.0);
            let next = i + s;
            let cont = if s > 0.0 { next <= lim } else { next >= lim };
            if cont {
                *reg!(mut base) = Value::Float(next);
                *reg!(mut base + 3) = Value::Float(next);
                ip = unsafe { ip.offset(offset as isize) };
            }
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
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_base, offset) = args!(Instruction::TFORPREP { base, offset });
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: R[base+4], ... = R[base](R[base+1], R[base+2])
#[inline(never)]
extern "rust-preserve-none" fn op_tforcall<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_base, _count) = args!(Instruction::TFORCALL { base, count });
    // TODO: Call R[base](R[base+1], R[base+2]) and store `count` results
    // starting at R[base+4]. Requires the function call machinery.
    dispatch!();
}

/// Generic for loop test: if R[base+2] != nil then R[base] = R[base+2] and jump back.
#[inline(never)]
extern "rust-preserve-none" fn op_tforloop<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
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
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_table, _count, _offset) = args!(Instruction::SETLIST {
        table,
        count,
        offset
    });
    // TODO: needs &Mutation for table.raw_set
    // let t = reg!(table).get_table().unwrap();
    // let n = count as usize;
    // let off = offset as i64;
    // for i in 1..=n {
    //     let val = reg!(table + i as u8);
    //     let key = Value::Integer(off + i as i64);
    //     t.raw_set(mc, key, val);
    // }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Closures
// ---------------------------------------------------------------------------

/// R[dst] = closure(proto[idx])
#[inline(never)]
extern "rust-preserve-none" fn op_closure<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_dst, _proto_idx) = args!(Instruction::CLOSURE { dst, proto });
    // TODO: needs &Mutation for Function::new_lua / Gc::new
    // let frame = thread.frames.last().unwrap();
    // match &*frame.function.inner() {
    //     FunctionKind::Lua(closure) => {
    //         let proto = closure.proto.prototypes[proto_idx as usize];
    //         let func = Function::new_lua(mc, proto, Box::new([]));
    //         *reg!(mut dst) = Value::Function(func);
    //     }
    //     _ => { raise!(); }
    // }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Varargs
// ---------------------------------------------------------------------------

/// Copy varargs into R[dst], R[dst+1], ...
#[inline(never)]
extern "rust-preserve-none" fn op_vararg<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (_dst, _count) = args!(Instruction::VARARG { dst, count });
    // TODO: Copy vararg values from below the current frame's base into
    // R[dst]..R[dst+count-2]. Varargs are the arguments beyond num_fixed
    // that were passed to the current function.
    dispatch!();
}

/// Adjust stack for varargs on function entry.
#[inline(never)]
extern "rust-preserve-none" fn op_varargprep<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let _num_fixed = args!(Instruction::VARARGPREP { num_fixed });
    // TODO: Move fixed params into proper positions and save extra args
    // as varargs below the frame base.
    dispatch!();
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_nop<'gc>(
    instruction: Instruction,
    thread: &mut ThreadState<'_>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    args!(Instruction::NOP {});
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_stop<'gc>(
    instruction: Instruction,
    _thread: &mut ThreadState<'_>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, _thread, _registers, _ip, _handlers);
    args!(Instruction::STOP {});
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_number(v: Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(i as f64),
        Value::Float(f) => Some(f),
        _ => None,
    }
}
