use crate::instruction::Instruction;
use crate::value::Value;

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

#[inline(never)]
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
fn op_load(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    ip: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, ip, handlers);
    let (dst, idx) = args!(Instruction::LOAD { dst, idx });
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
    let src = args!(Instruction::LFALSESKIP { src });
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
    let (dst, idx) = args!(Instruction::GETUPVAL { dst, idx });
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
    let (src, idx) = args!(Instruction::SETUPVAL { src, idx });
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
    let (dst, idx, key) = args!(Instruction::GETTABUP { dst, idx, key });
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
    let (src, idx, key) = args!(Instruction::SETTABUP { src, idx, key });
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
    let (dst, table, key) = args!(Instruction::GETTABLE { dst, table, key });
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
    let (src, table, key) = args!(Instruction::SETTABLE { src, table, key });
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
    let dst = args!(Instruction::NEWTABLE { dst });
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
    let (dst, lhs, rhs) = args!(Instruction::ADD { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::SUB { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::MUL { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::MOD { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::POW { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::DIV { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::IDIV { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::BAND { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::BOR { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::BXOR { dst, lhs, rhs });
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
    let (dst, lhs, rhs) = args!(Instruction::SHR { dst, lhs, rhs });
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
    let (lhs, rhs, metamethod) = args!(Instruction::MMBIN { lhs, rhs, metamethod });
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
    let (dst, src) = args!(Instruction::UNM { dst, src });
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
    let (dst, src) = args!(Instruction::BNOT { dst, src });
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
    let (dst, src) = args!(Instruction::BNOT { dst, src });
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
    let (dst, src) = args!(Instruction::BNOT { dst, src });
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
    let (dst, lhs, rhs) = args!(Instruction::CONCAT { dst, lhs, rhs });
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
    let start = args!(Instruction::CLOSE { start });
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
    let val = args!(Instruction::TBC { val });
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
    let offset = args!(Instruction::JMP { offset });
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
    let (lhs, rhs, inverted) = args!(Instruction::EQ { lhs, rhs, inverted });
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
    let (lhs, rhs, inverted) = args!(Instruction::LT { lhs, rhs, inverted });
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
    let (lhs, rhs, inverted) = args!(Instruction::LE { lhs, rhs, inverted });
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
    let (src, inverted) = args!(Instruction::TEST { src, inverted });
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
    let (func, args, returns) = args!(Instruction::CALL { func, args, returns });
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
    let (func, args) = args!(Instruction::TAILCALL { func, args });
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
    let (values, count) = args!(Instruction::RETURN { values, count });
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
    args!(Instruction::NOP {});
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
    helpers!(_instruction, _thread, _registers, _ip, _handlers);
    args!(Instruction::STOP {});
    Ok(())
}
