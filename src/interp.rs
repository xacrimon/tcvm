use std::panic;

use crate::instruction::{self, Instruction};
use crate::value::{Value, ValueType};

const HANDLERS: &[Handler] = &[op_stop, op_noop, op_add, op_load];

#[derive(Debug)]
struct Error {
    pc: usize,
    caller: &'static panic::Location<'static>,
}

pub struct Thread {
    state: (),
    tape: Vec<Instruction>,
}

type Handler = fn(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    tape: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>>;

#[track_caller]
#[cold]
#[inline(never)]
fn impl_error(
    _instruction: Instruction,
    thread: &mut Thread,
    _registers: &mut [Value],
    tape: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    let pc = unsafe { tape.offset_from_unsigned(thread.tape.as_ptr()) };
    let caller = panic::Location::caller();
    let error = Error { pc, caller };
    Err(Box::new(error))
}

macro_rules! helpers {
    ($instruction:expr, $thread:expr, $registers:expr, $tape:expr, $handlers:expr) => {
        macro_rules! dispatch {
            () => {{
                let tape = unsafe { $tape.add(1) };
                dispatch!($instruction, $thread, $registers, tape, $handlers, impl);
            }};

            (start) => {{
                dispatch!($instruction, $thread, $registers, $tape, $handlers, impl);
            }};

            ($$instruction:expr, $$thread:expr, $$registers:expr, $$tape:expr, $$handlers:expr, impl) => {{
                unsafe {
                    let _ = $$instruction;
                    debug_assert!($$tape.offset_from_unsigned($$thread.tape.as_ptr()) < $$thread.tape.len());
                    let instruction = *$$tape;
                    let pos = instruction.discriminant() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = $$handlers.cast::<Handler>().add(pos).read();
                    return handler(instruction, $$thread, $$registers, $$tape, $$handlers); // TODO: use become
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
                return impl_error($instruction, $thread, $registers, $tape, $handlers); // TODO: use become
            }};
        }

        macro_rules! check {
            ($$cond:expr) => {{
                if std::hint::unlikely(!$$cond) {
                    raise!();
                }
            }};
        }

        macro_rules! ok {
            ($expr:expr) => {
                match $expr {
                    Ok(val) => val,
                    Err(_) => raise!(),
                }
            };
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
    #[inline(never)]
    fn start(
        instruction: Instruction,
        thread: &mut Thread,
        registers: &mut [Value],
        tape: *const Instruction,
        handlers: *const (),
    ) -> Result<(), Box<Error>> {
        helpers!(instruction, thread, registers, tape, handlers);
        dispatch!(start);
    }

    let tape = tape.as_ptr();
    let handlers = HANDLERS.as_ptr() as *const ();
    start(Instruction::Nop, thread, &mut [], tape, handlers).unwrap();
}

#[inline(never)]
fn op_stop(
    _instruction: Instruction,
    _thread: &mut Thread,
    _registers: &mut [Value],
    _tape: *const Instruction,
    _handlers: *const (),
) -> Result<(), Box<Error>> {
    Ok(())
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn op_noop(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    tape: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, tape, handlers);
    dispatch!();
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn op_add(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    tape: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, tape, handlers);
    let (out, lhs, rhs) = args!(Instruction::Add { out, lhs, rhs });

    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    check!((lhs.ty(), rhs.ty()) == (ValueType::Integer, ValueType::Integer));

    let result = lhs.as_integer() + rhs.as_integer();
    *reg!(mut out) = Value::new_integer(result);

    dispatch!();
}

#[inline(never)]
fn op_load(
    instruction: Instruction,
    thread: &mut Thread,
    registers: &mut [Value],
    tape: *const Instruction,
    handlers: *const (),
) -> Result<(), Box<Error>> {
    helpers!(instruction, thread, registers, tape, handlers);
    dispatch!();
}
