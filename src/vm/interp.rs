use crate::instruction::Instruction;
use crate::vm::num::{self, op_arith, op_bit};
use crate::env::Value;

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

pub struct Thread {
    tape: Vec<Instruction>,
}

#[cfg(debug_assertions)]
type Registers<'gc, 'a> = &'a mut [Value<'gc>];

#[cfg(not(debug_assertions))]
type Registers<'gc, 'a> = *mut Value<'gc>;

type Handler = fn(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'_, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

macro_rules! helpers {
    ($instruction:expr, $thread:expr, $registers:expr, $ip:expr, $handlers:expr) => {
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

        macro_rules! args {
            ($$kind:path { $$($$field:ident),* }) => {{
                match $instruction {
                    $$kind { $$($$field),* } => ( $$($$field),* ),
                    _ => unsafe { std::hint::unreachable_unchecked() },
                }
            }};
        }

        macro_rules! raise {
            () => {{
                become impl_error($instruction, $thread, $registers, $ip, $handlers);
            }};
        }

        macro_rules! check {
            ($$cond:expr) => {{
                if std::hint::unlikely(!$$cond) {
                    raise!();
                }
            }};
        }

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
    };
}

#[inline(never)]
pub fn run(tape: &[Instruction], thread: &mut Thread) {
    let ip = tape.as_ptr();
    let handlers = HANDLERS.as_ptr() as *const ();

    #[cfg(debug_assertions)]
    let registers = &mut [];

    #[cfg(not(debug_assertions))]
    let registers = std::ptr::null_mut();

    op_nop(Instruction::NOP, thread, registers, ip, handlers).unwrap();
}

#[cold]
#[inline(never)]
fn impl_error<'gc>(
    _instruction: Instruction,
    thread: &mut Thread,
    _registers: Registers<'gc, '_>,
    ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    let pc = unsafe { ip.offset_from_unsigned(thread.tape.as_ptr()) };
    let error = Error { pc };
    Err(Box::new(error))
}

#[inline(never)]
fn op_move<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::MOVE { dst, src });
    *reg!(mut dst) = reg!(src);
    dispatch!();
}

#[inline(never)]
fn op_load<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::LOAD { dst, idx });
    dispatch!();
}

#[inline(never)]
fn op_lfalseskip<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let src = args!(Instruction::LFALSESKIP { src });
    dispatch!();
}

#[inline(never)]
fn op_getupval<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::GETUPVAL { dst, idx });
    dispatch!();
}

#[inline(never)]
fn op_setupval<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, idx) = args!(Instruction::SETUPVAL { src, idx });
    dispatch!();
}

#[inline(never)]
fn op_gettabup<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx, key) = args!(Instruction::GETTABUP { dst, idx, key });
    dispatch!();
}

#[inline(never)]
fn op_settabup<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, idx, key) = args!(Instruction::SETTABUP { src, idx, key });
    dispatch!();
}

#[inline(never)]
fn op_gettable<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, table, key) = args!(Instruction::GETTABLE { dst, table, key });
    dispatch!();
}

#[inline(never)]
fn op_settable<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });
    dispatch!();
}

#[inline(never)]
fn op_newtable<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let dst = args!(Instruction::NEWTABLE { dst });
    dispatch!();
}

#[inline(never)]
fn op_add<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::ADD { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));

    if let Some(v) = op_arith::<num::Add>(lhs, rhs) {
        *reg!(mut dst) = v;
    }

    dispatch!();
}

#[inline(never)]
fn op_sub<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_mul<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_mod<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_pow<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_div<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_idiv<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::IDIV { dst, lhs, rhs });
    let (lhs, rhs) = (reg!(lhs), reg!(rhs));

    if let Some(v) = op_arith::<num::Sub>(lhs, rhs) {
        *reg!(mut dst) = v;
    }

    dispatch!();
}

#[inline(never)]
fn op_band<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_bor<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_bxor<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_shl<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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
fn op_shr<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
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

#[inline(never)]
fn op_mmbin<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, metamethod) = args!(Instruction::MMBIN {
        lhs,
        rhs,
        metamethod
    });
    dispatch!();
}

#[inline(never)]
fn op_unm<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::UNM { dst, src });
    dispatch!();
}

#[inline(never)]
fn op_bnot<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    dispatch!();
}

#[inline(never)]
fn op_not<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    dispatch!();
}

#[inline(never)]
fn op_len<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::BNOT { dst, src });
    dispatch!();
}

#[inline(never)]
fn op_concat<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, lhs, rhs) = args!(Instruction::CONCAT { dst, lhs, rhs });
    dispatch!();
}

#[inline(never)]
fn op_close<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let start = args!(Instruction::CLOSE { start });
    dispatch!();
}

#[inline(never)]
fn op_tbc<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let val = args!(Instruction::TBC { val });
    dispatch!();
}

#[inline(never)]
fn op_jmp<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let offset = args!(Instruction::JMP { offset });
    dispatch!();
}

#[inline(never)]
fn op_eq<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::EQ { lhs, rhs, inverted });
    dispatch!();
}

#[inline(never)]
fn op_lt<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::LT { lhs, rhs, inverted });
    dispatch!();
}

#[inline(never)]
fn op_le<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (lhs, rhs, inverted) = args!(Instruction::LE { lhs, rhs, inverted });
    dispatch!();
}

#[inline(never)]
fn op_test<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (src, inverted) = args!(Instruction::TEST { src, inverted });
    dispatch!();
}

#[inline(never)]
fn op_call<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (func, args, returns) = args!(Instruction::CALL {
        func,
        args,
        returns
    });
    dispatch!();
}

#[inline(never)]
fn op_tailcall<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (func, args) = args!(Instruction::TAILCALL { func, args });
    dispatch!();
}

#[inline(never)]
fn op_return<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (values, count) = args!(Instruction::RETURN { values, count });
    dispatch!();
}

#[inline(never)]
fn op_forloop<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_forprep<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforprep<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforcall<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforloop<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_setlist<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_closure<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_vararg<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_varargprep<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_nop<'gc>(
    instruction: Instruction,
    thread: &mut Thread,
    registers: Registers<'gc, '_>,
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    args!(Instruction::NOP {});
    dispatch!();
}

#[inline(never)]
fn op_stop<'gc>(
    instruction: Instruction,
    _thread: &mut Thread,
    _registers: Registers<'gc, '_>,
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, _thread, _registers, _ip, _handlers);
    args!(Instruction::STOP {});
    Ok(())
}
