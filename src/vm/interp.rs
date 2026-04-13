use crate::dmm::Mutation;
use crate::env::Value;
use crate::env::function::{Function, FunctionKind, UpvalueState};
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::{CallFrame, ThreadState, ThreadStatus};
use crate::instruction::Instruction;
use crate::vm::num::{self, op_arith, op_bit};

macro_rules! handler_array {
    ($gc:lifetime) => {{
        type H<'a> = Handler<'a>;
        super let handlers: &[Handler<$gc>] = &[
                    op_move as H,
                    op_load as H,
                    op_lfalseskip as H,
                    op_getupval as H,
                    op_setupval as H,
                    op_gettabup as H,
                    op_settabup as H,
                    op_gettable as H,
                    op_settable as H,
                    op_newtable as H,
                    op_add as H,
                    op_sub as H,
                    op_mul as H,
                    op_mod as H,
                    op_pow as H,
                    op_div as H,
                    op_idiv as H,
                    op_band as H,
                    op_bor as H,
                    op_bxor as H,
                    op_shl as H,
                    op_shr as H,
                    op_mmbin as H,
                    op_unm as H,
                    op_bnot as H,
                    op_not as H,
                    op_len as H,
                    op_concat as H,
                    op_close as H,
                    op_tbc as H,
                    op_jmp as H,
                    op_eq as H,
                    op_lt as H,
                    op_le as H,
                    op_test as H,
                    op_call as H,
                    op_tailcall as H,
                    op_return as H,
                    op_forloop as H,
                    op_forprep as H,
                    op_tforprep as H,
                    op_tforcall as H,
                    op_tforloop as H,
                    op_setlist as H,
                    op_closure as H,
                    op_vararg as H,
                    op_varargprep as H,
                    op_nop as H,
                    op_stop as H,
                ];
        handlers
    }};
}

#[derive(Debug)]
struct Error {
    pc: usize,
}

#[cfg(debug_assertions)]
type Registers<'gc, 'a> = &'a mut [Value<'gc>];

#[cfg(not(debug_assertions))]
type Registers<'gc, 'a> = *mut Value<'gc>;

type Handler<'gc> = extern "rust-preserve-none" fn(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

macro_rules! helpers {
    ($instruction:expr, $mc:expr, $thread:expr, $registers:expr, $ip:expr, $handlers:expr) => {
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
                    debug_assert!(pos < handler_array!('static).len());
                    let handler = *$handlers.cast::<Handler<'gc>>().add(pos);
                    let ip = $ip.add(1);
                    become handler(instruction, $mc, $thread, $registers, ip, $handlers);
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
                become impl_error($instruction, $mc, $thread, $registers, $ip, $handlers);
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
                frame.closure.proto.constants[$$idx as usize]
            }};
        }

        #[allow(unused_macros)]
        macro_rules! upvalue {
            ($$idx:expr) => {{
                let frame = $thread.frames.last().unwrap();
                frame.closure.upvalues[$$idx as usize]
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
pub fn run<'gc>(mc: &Mutation<'gc>, tape: &[Instruction], thread: &mut ThreadState<'gc>) {
    let ip = tape.as_ptr();
    let handlers = handler_array!('gc);
    let handlers = handlers.as_ptr() as *const ();

    #[cfg(debug_assertions)]
    let registers = &mut [];

    #[cfg(not(debug_assertions))]
    let registers = std::ptr::null_mut();

    op_nop(Instruction::NOP, mc, thread, registers, ip, handlers).unwrap();
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[cold]
#[inline(never)]
extern "rust-preserve-none" fn impl_error<'gc>(
    _instruction: Instruction,
    _mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::MOVE { dst, src });
    *reg!(mut dst) = reg!(src);
    dispatch!();
}

/// Load constant from the current prototype's constant pool.
#[inline(never)]
extern "rust-preserve-none" fn op_load<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::LOAD { dst, idx });
    *reg!(mut dst) = constant!(idx);
    dispatch!();
}

/// Set register to false and skip the next instruction.
#[inline(never)]
extern "rust-preserve-none" fn op_lfalseskip<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (src, idx) = args!(Instruction::SETUPVAL { src, idx });
    let val = reg!(src);
    let uv = upvalue!(idx);
    let mut uv_ref = uv.borrow_mut(mc);
    match &mut *uv_ref {
        UpvalueState::Open { thread: t, index } => {
            t.borrow_mut(mc).stack[*index] = val;
        }
        UpvalueState::Closed(v) => *v = val,
    }
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table access via upvalue
// ---------------------------------------------------------------------------

/// R[dst] = UpValue[idx][K[key]]
#[inline(never)]
extern "rust-preserve-none" fn op_gettabup<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (src, idx, key) = args!(Instruction::SETTABUP { src, idx, key });
    let uv = upvalue!(idx);
    let table_val = match &*uv.borrow() {
        UpvalueState::Open { thread: t, index } => t.borrow().stack[*index],
        UpvalueState::Closed(v) => *v,
    };
    let table = table_val.get_table().unwrap();
    let key = constant!(key);
    let val = reg!(src);
    table.raw_set(mc, key, val);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Table access via register
// ---------------------------------------------------------------------------

/// R[dst] = R[table][R[key]]
#[inline(never)]
extern "rust-preserve-none" fn op_gettable<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });
    let t = reg!(table).get_table().unwrap();
    let k = reg!(key);
    let v = reg!(src);
    // TODO: __newindex metamethod fallback
    t.raw_set(mc, k, v);
    dispatch!();
}

/// R[dst] = {}
#[inline(never)]
extern "rust-preserve-none" fn op_newtable<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let dst = args!(Instruction::NEWTABLE { dst });
    *reg!(mut dst) = Value::Table(Table::new(mc));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Arithmetic (register-register)
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_add<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::UNM { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::Integer(i) => Value::Integer(i.wrapping_neg()),
        Value::Float(f) => Value::Float(-f),
        // TODO: __unm metamethod
        _ => {
            raise!();
        }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = ~R[src]  (bitwise NOT)
#[inline(never)]
extern "rust-preserve-none" fn op_bnot<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::Integer(i) => Value::Integer(!i),
        // TODO: __bnot metamethod
        _ => {
            raise!();
        }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = not R[src]  (logical NOT — always produces boolean)
#[inline(never)]
extern "rust-preserve-none" fn op_not<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::NOT { dst, src });
    let val = reg!(src);
    *reg!(mut dst) = Value::Boolean(val.is_falsy());
    dispatch!();
}

/// R[dst] = #R[src]  (length)
#[inline(never)]
extern "rust-preserve-none" fn op_len<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::LEN { dst, src });
    let val = reg!(src);
    let result = match val {
        Value::String(s) => Value::Integer(s.len() as i64),
        Value::Table(t) => Value::Integer(t.raw_len() as i64),
        // TODO: __len metamethod
        _ => {
            raise!();
        }
    };
    *reg!(mut dst) = result;
    dispatch!();
}

/// R[dst] = R[lhs] .. R[rhs]  (string concatenation)
#[inline(never)]
#[unsafe(no_mangle)]
extern "rust-preserve-none" fn op_concat<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::CONCAT { dst, lhs, rhs });
    let a = reg!(lhs);
    let b = reg!(rhs);
    // TODO: __concat metamethod
    let mut buf = Vec::new();
    check!(num::coerce_to_str(&mut buf, a));
    check!(num::coerce_to_str(&mut buf, b));
    *reg!(mut dst) = Value::String(LuaString::new(mc, &buf));
    dispatch!();
}

// ---------------------------------------------------------------------------
// Upvalue / resource management
// ---------------------------------------------------------------------------

/// Close all upvalues >= R[start].
#[inline(never)]
extern "rust-preserve-none" fn op_close<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let start = args!(Instruction::CLOSE { start });
    let base = thread.frames.last().map_or(0, |f| f.base);
    let start_idx = base + start as usize;
    close_upvalues(mc, thread, start_idx);
    dispatch!();
}

/// Mark R[val] as to-be-closed.
#[inline(never)]
extern "rust-preserve-none" fn op_tbc<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let offset = args!(Instruction::JMP { offset });
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// if (R[lhs] == R[rhs]) != inverted then skip next instruction
#[inline(never)]
extern "rust-preserve-none" fn op_eq<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
        _ => {
            raise!();
        }
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
        _ => {
            raise!();
        }
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (func, nargs, returns) = args!(Instruction::CALL {
        func,
        args,
        returns
    });
    let base = thread.frames.last().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let func_val = thread.stack[func_idx].get_function().unwrap();
    match &*func_val.inner() {
        FunctionKind::Lua(closure) => {
            let closure = *closure;
            let new_base = func_idx + 1;
            // Save caller's PC (offset from current code start)
            if let Some(frame) = thread.frames.last_mut() {
                let code_start = frame.closure.proto.code.as_ptr();
                frame.pc = unsafe { ip.offset_from_unsigned(code_start) };
            }
            // Ensure stack is large enough
            let needed = new_base + closure.proto.max_stack_size as usize;
            if thread.stack.len() < needed {
                thread.stack.resize(needed, Value::Nil);
            }
            // Push new call frame
            thread.frames.push(CallFrame {
                closure,
                base: new_base,
                pc: 0,
                num_results: returns,
            });
            // Rebind ip and registers to new frame
            ip = closure.proto.code.as_ptr();
            #[cfg(debug_assertions)]
            let registers = &mut thread.stack[new_base..];
            #[cfg(not(debug_assertions))]
            let registers = unsafe { thread.stack.as_mut_ptr().add(new_base) };
            dispatch!();
        }
        FunctionKind::Native(_native) => {
            // TODO: native function calls
            raise!();
        }
    }
}

/// return R[func](R[func+1], ..., R[func+args-1])  — tail call
#[inline(never)]
extern "rust-preserve-none" fn op_tailcall<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (func, nargs) = args!(Instruction::TAILCALL { func, args });
    let base = thread.frames.last().map_or(0, |f| f.base);
    let func_idx = base + func as usize;
    let func_val = thread.stack[func_idx].get_function().unwrap();
    match &*func_val.inner() {
        FunctionKind::Lua(closure) => {
            let closure = *closure;
            let cur_base = thread.frames.last().unwrap().base;
            // Move function + arguments down to current frame's base - 1
            let nargs = if nargs == 0 { 0 } else { nargs as usize - 1 };
            let src_start = func_idx + 1;
            for i in 0..nargs {
                thread.stack[cur_base + i] = thread.stack[src_start + i];
            }
            // Close upvalues for the current frame
            close_upvalues(mc, thread, cur_base);
            // Replace current frame
            let frame = thread.frames.last_mut().unwrap();
            frame.closure = closure;
            frame.pc = 0;
            // num_results stays the same (caller's expectation)
            // Ensure stack is large enough
            let needed = cur_base + closure.proto.max_stack_size as usize;
            if thread.stack.len() < needed {
                thread.stack.resize(needed, Value::Nil);
            }
            // Rebind ip and registers
            ip = closure.proto.code.as_ptr();
            #[cfg(debug_assertions)]
            let registers = &mut thread.stack[cur_base..];
            #[cfg(not(debug_assertions))]
            let registers = unsafe { thread.stack.as_mut_ptr().add(cur_base) };
            dispatch!();
        }
        FunctionKind::Native(_native) => {
            // TODO: native tail calls
            raise!();
        }
    }
}

/// return R[values], ..., R[values+count-2]
#[inline(never)]
extern "rust-preserve-none" fn op_return<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (values, count) = args!(Instruction::RETURN { values, count });
    let frame = thread.frames.last().unwrap();
    let cur_base = frame.base;
    let num_results = frame.num_results;
    let nret = if count == 0 { 0 } else { count as usize - 1 };

    // Close upvalues for the departing frame
    close_upvalues(mc, thread, cur_base);

    // Pop current frame
    thread.frames.pop();

    if thread.frames.is_empty() {
        // Top-level return — copy results to stack base and finish
        let dst_start = cur_base - 1; // func slot
        for i in 0..nret {
            thread.stack[dst_start + i] = thread.stack[cur_base + values as usize + i];
        }
        thread.status = ThreadStatus::Dead;
        return Ok(());
    }

    // Copy return values into caller's expected slots
    let dst_start = cur_base - 1; // func slot in caller's frame
    let wanted = if num_results == 0 {
        0
    } else {
        num_results as usize - 1
    };
    let to_copy = nret.min(wanted);
    for i in 0..to_copy {
        thread.stack[dst_start + i] = thread.stack[cur_base + values as usize + i];
    }
    // Nil-fill remaining expected results
    for i in to_copy..wanted {
        thread.stack[dst_start + i] = Value::Nil;
    }

    // Restore caller's ip and registers
    let caller = thread.frames.last().unwrap();
    let caller_base = caller.base;
    ip = unsafe { caller.closure.proto.code.as_ptr().add(caller.pc) };
    #[cfg(debug_assertions)]
    let registers = &mut thread.stack[caller_base..];
    #[cfg(not(debug_assertions))]
    let registers = unsafe { thread.stack.as_mut_ptr().add(caller_base) };
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (base, offset) = args!(Instruction::FORPREP { base, offset });

    let init = reg!(base);
    let limit = reg!(base + 1);
    let step = reg!(base + 2);

    let should_run = match (init, limit, step) {
        (Value::Integer(i), Value::Integer(lim), Value::Integer(s)) => {
            if s > 0 {
                i <= lim
            } else {
                i >= lim
            }
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (_base, offset) = args!(Instruction::TFORPREP { base, offset });
    ip = unsafe { ip.offset(offset as isize) };
    dispatch!();
}

/// Generic for call: R[base+4], ... = R[base](R[base+1], R[base+2])
#[inline(never)]
extern "rust-preserve-none" fn op_tforcall<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (_base, _count) = args!(Instruction::TFORCALL { base, count });
    // TODO: Call R[base](R[base+1], R[base+2]) and store `count` results
    // starting at R[base+4]. Requires the function call machinery.
    dispatch!();
}

/// Generic for loop test: if R[base+2] != nil then R[base] = R[base+2] and jump back.
#[inline(never)]
extern "rust-preserve-none" fn op_tforloop<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
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
        let key = Value::Integer(off + i as i64);
        t.raw_set(mc, key, val);
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
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, proto_idx) = args!(Instruction::CLOSURE { dst, proto });
    let frame = thread.frames.last().unwrap();
    let proto = frame.closure.proto.prototypes[proto_idx as usize];
    // TODO: capture upvalues from enclosing scope based on proto.upvalue_desc
    let func = Function::new_lua(mc, proto, Box::new([]));
    *reg!(mut dst) = Value::Function(func);
    dispatch!();
}

// ---------------------------------------------------------------------------
// Varargs
// ---------------------------------------------------------------------------

/// Copy varargs into R[dst], R[dst+1], ...
#[inline(never)]
extern "rust-preserve-none" fn op_vararg<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let (dst, count) = args!(Instruction::VARARG { dst, count });
    let frame = thread.frames.last().unwrap();
    let base = frame.base;
    let num_fixed = frame.closure.proto.num_params as usize;
    // Varargs are stored below base: stack[base - num_varargs .. base - num_fixed]
    // Actually they're at stack[base - num_extra .. base] where the caller put them
    // For now: the varargs sit in the slots between (base - num_extra) and base
    // The exact number of varargs depends on how many args were actually passed.
    // TODO: track actual arg count to properly copy varargs
    let wanted = if count == 0 { 0 } else { count as usize - 1 };
    for i in 0..wanted {
        *reg!(mut dst + i as u8) = Value::Nil;
    }
    dispatch!();
}

/// Adjust stack for varargs on function entry.
#[inline(never)]
extern "rust-preserve-none" fn op_varargprep<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    let _num_fixed = args!(Instruction::VARARGPREP { num_fixed });
    // VARARGPREP is the first instruction of a vararg function.
    // In Lua 5.4, this adjusts the stack so that fixed params are in the
    // right place and extra args are accessible by VARARG.
    // TODO: implement vararg stack adjustment when we track actual arg count
    dispatch!();
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

#[inline(never)]
extern "rust-preserve-none" fn op_nop<'gc>(
    instruction: Instruction,
    mc: &Mutation<'gc>,
    thread: &mut ThreadState<'gc>,
    registers: Registers<'gc, '_>,
    mut ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, mc, thread, registers, ip, handlers);
    args!(Instruction::NOP {});
    dispatch!();
}

#[inline(never)]
extern "rust-preserve-none" fn op_stop<'gc>(
    instruction: Instruction,
    _mc: &Mutation<'gc>,
    _thread: &mut ThreadState<'gc>,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, _mc, _thread, _registers, _ip, _handlers);
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
