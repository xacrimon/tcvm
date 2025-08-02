use std::panic;

use crate::instruction::Instruction;
use crate::value::{Value, ValueType};

const HANDLERS: &[Handler] = &[
    op_move,
    op_loadi,
    op_loadk,
    op_loadfalse,
    op_lfalseskip,
    op_loadtrue,
    op_loadnil,
    op_getupval,
    op_setupval,
    op_gettabup,
    op_gettable,
    op_geti,
    op_getfield,
    op_settabup,
    op_settable,
    op_seti,
    op_setfield,
    op_newtable,
    op_self,
    op_addi,
    op_addk,
    op_subk,
    op_mulk,
    op_modk,
    op_powk,
    op_divk,
    op_idivk,
    op_bandk,
    op_bork,
    op_bxork,
    op_shri,
    op_shli,
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
    op_mmbini,
    op_mmbink,
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
    op_eqk,
    op_eqi,
    op_lti,
    op_lei,
    op_gti,
    op_gei,
    op_test,
    op_testset,
    op_call,
    op_tailcall,
    op_return,
    op_return0,
    op_return1,
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

type Handler = fn(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

macro_rules! helpers {
    ($instruction:expr, $thread:expr, $registers:expr, $ip:expr, $handlers:expr) => {
        macro_rules! dispatch {
            () => {{
                dispatch!(@impl: $instruction, $thread, $registers, $ip, $handlers);
            }};

            (@impl: $$instruction:expr, $$thread:expr, $$registers:expr, $$ip:expr, $$handlers:expr) => {{
                unsafe {
                    let _ = $$instruction;
                    debug_assert!($$ip.offset_from_unsigned($$thread.tape.as_ptr()) < $$thread.tape.len());
                    let instruction = *$$ip;
                    let pos = instruction.discriminant() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = *$$handlers.cast::<Handler>().add(pos);
                    let ip = $$ip.add(1);
                    become handler(instruction, $$thread, $$registers, ip, $$handlers);
                }
            }};
        }

        macro_rules! args {
            ($$kind:path { $$($$field:ident),* }) => {{
                match $instruction {
                    $$kind { $$($$field),* } => {
                        ( $$($$field),* )
                    },
                    #[cfg(debug_assertions)]
                    _ => unreachable!(),
                    #[cfg(not(debug_assertions))]
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
                unsafe {
                    debug_assert!(($$idx as usize) < $registers.len());
                    *$registers.get_unchecked($$idx as usize)
                }
            }};

            (mut $$idx:expr) => {{
                unsafe {
                    debug_assert!(($$idx as usize) < $registers.len());
                    $registers.get_unchecked_mut($$idx as usize)
                }
            }};
        }
    };
}

pub fn run(tape: &[Instruction], thread: &mut Thread) {
    let ip = tape.as_ptr();
    let handlers = HANDLERS.as_ptr() as *const ();
    op_nop(Instruction::NOP, thread, &mut [], ip, handlers).unwrap();
}


#[cold]
#[inline(never)]
fn impl_error(
    _instruction: Instruction,
    thread: &mut Thread,
    _registers: &mut [Value],
    ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    let pc = unsafe { ip.offset_from_unsigned(thread.tape.as_ptr()) };
    let error = Error { pc };
    Err(Box::new(error))
}

#[inline(never)]
fn op_move(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, src) = args!(Instruction::MOVE { dst, src });
    *reg!(mut dst) = reg!(src);
    dispatch!();
}

#[inline(never)]
fn op_loadi(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, imm) = args!(Instruction::LOADI { dst, imm });
    *reg!(mut dst) = Value::new_integer(imm as i64);
    dispatch!();
}

#[inline(never)]
fn op_loadk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_loadfalse(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_lfalseskip(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_loadtrue(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_loadnil(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_getupval(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_setupval(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_gettabup(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_gettable(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_geti(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_getfield(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_settabup(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_settable(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_seti(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_setfield(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_newtable(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_self(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_addi(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_addk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_subk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mulk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_modk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_powk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_divk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_idivk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bandk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bork(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bxork(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_shri(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_shli(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_add(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_sub(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mul(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mod(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_pow(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_div(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_idiv(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_band(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bor(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bxor(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_shl(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_shr(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mmbin(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mmbini(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_mmbink(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_unm(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_bnot(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_not(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_len(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_concat(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_close(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tbc(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_jmp(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_eq(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_lt(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_le(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_eqk(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_eqi(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_lti(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_lei(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_gti(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_gei(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_test(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_testset(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_call(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tailcall(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_return(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_return0(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_return1(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_forloop(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_forprep(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforprep(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforcall(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_tforloop(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_setlist(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_closure(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_vararg(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_varargprep(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_nop(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    dispatch!();
}

#[inline(never)]
fn op_stop(
    _instruction: Instruction,
    _thread: &mut Thread,
    _registers: &mut [Value],
    _ip: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    Ok(())
}
