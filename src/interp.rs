use std::panic;
use std::result;

use crate::instruction::Instruction;
use crate::value::{Value, ValueType};

const HANDLERS: &[Handler] = &[op_stop, op_noop, op_add, op_load];

#[derive(Debug)]
struct Error {
    pc: usize,
    caller: &'static panic::Location<'static>,
}

type Result = result::Result<(), Box<Error>>;

pub struct Thread {
    state: (),
}

type Handler = fn(
    thread: &mut Thread,
    registers: &mut [Value],
    pc: usize,
    tape: &[Instruction],
    handlers: *const (),
) -> Result;

#[track_caller]
#[inline(never)]
fn impl_error(
    _thread: &mut Thread,
    _registers: &mut [Value],
    pc: usize,
    _tape: &[Instruction],
    _handlers: *const (),
) -> Result {
    let caller = panic::Location::caller();
    let error = Error { pc, caller };
    Err(Box::new(error))
}

macro_rules! helpers {
    ($thread:expr, $registers:expr, $pc:expr, $tape:expr, $handlers:expr) => {
        macro_rules! dispatch {
            () => {{
                dispatch!($thread, $registers, $pc + 1, $tape, $handlers, impl);
            }};

            (start) => {{
                dispatch!($thread, $registers, $pc, $tape, $handlers, impl);
            }};

            ($$thread:expr, $$registers:expr, $$pc:expr, $$tape:expr, $$handlers:expr, impl) => {{
                unsafe {
                    debug_assert!($$pc < $$tape.len());
                    let pos = (*$$tape.get_unchecked($pc)).discriminant() as usize;
                    debug_assert!(pos < HANDLERS.len());
                    let handler = $$handlers.cast::<Handler>().add(pos).read();
                    return handler($$thread, $$registers, $$pc, $$tape, $$handlers); // TODO: use become
                }
            }};
        }

        macro_rules! args {
            ($$kind:path { $$($$field:ident),* }) => {{
                unsafe {
                    debug_assert!($pc < $tape.len());
                    match *$tape.get_unchecked($pc) {
                        $$kind { $$($$field),* } => {
                            ( $$($$field),* )
                        },
                        #[cfg(debug_assertions)]
                        _ => unreachable!(),
                        #[cfg(not(debug_assertions))]
                        _ => std::hint::unreachable_unchecked(),
                    }
                }
            }};
        }

        macro_rules! raise {
            () => {{
                return impl_error($thread, $registers, $pc, $tape, $handlers); // TODO: use become
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
        thread: &mut Thread,
        registers: &mut [Value],
        pc: usize,
        tape: &[Instruction],
        handlers: *const (),
    ) -> Result {
        helpers!(thread, registers, pc, tape, handlers);
        dispatch!(start);
    }

    let handlers = HANDLERS.as_ptr() as *const ();
    start(thread, &mut [], 0, tape, handlers).unwrap();
}

#[inline(never)]
fn op_stop(
    _thread: &mut Thread,
    _registers: &mut [Value],
    _pc: usize,
    _tape: &[Instruction],
    _handlers: *const (),
) -> Result {
    Ok(())
}

#[inline(never)]
fn op_noop(
    thread: &mut Thread,
    registers: &mut [Value],
    pc: usize,
    tape: &[Instruction],
    handlers: *const (),
) -> Result {
    helpers!(thread, registers, pc, tape, handlers);
    dispatch!();
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn op_add(
    thread: &mut Thread,
    registers: &mut [Value],
    pc: usize,
    tape: &[Instruction],
    handlers: *const (),
) -> Result {
    helpers!(thread, registers, pc, tape, handlers);
    let (out, lhs, rhs) = args!(Instruction::Add { out, lhs, rhs });

    let (lhs, rhs) = (reg!(lhs), reg!(rhs));
    if !matches!(
        (lhs.ty(), rhs.ty()),
        (ValueType::Integer, ValueType::Integer)
    ) {
        raise!();
    }

    let result = lhs.as_integer() + rhs.as_integer();
    *reg!(mut out) = Value::new_integer(result);

    dispatch!();
}

#[inline(never)]
fn op_load(
    thread: &mut Thread,
    registers: &mut [Value],
    pc: usize,
    tape: &[Instruction],
    handlers: *const (),
) -> Result {
    helpers!(thread, registers, pc, tape, handlers);
    dispatch!();
}
