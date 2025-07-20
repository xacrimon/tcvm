#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum OpCode {
    Stop = 0,
    NoOp = 1,
    Incr = 2,
}

const HANDLERS: &[Handler] = &[insn_stop, insn_noop, insn_incr];

type Handler = fn(pc: usize, value: &mut i32, tape: &[OpCode], handlers: *const ());

macro_rules! dispatch {
    ($pc:expr, $value:expr, $tape:expr, $handlers:expr) => {{
        let mut pc = $pc;
        pc += 1;
        dispatch!(pc, $value, $tape, $handlers, impl);
    }};

    ($pc:expr, $value:expr, $tape:expr, $handlers:expr, start) => {{
        dispatch!($pc, $value, $tape, $handlers, impl);
    }};

    ($pc:expr, $value:expr, $tape:expr, $handlers:expr, impl) => {{
        unsafe {
            let op = *$tape.get_unchecked($pc);
            let handlers = $handlers as *const Handler;
            let handler = *handlers.add(op as usize);
            become handler($pc, $value, $tape, $handlers);
        }
    }};
}

fn insn_stop(_pc: usize, _value: &mut i32, _tape: &[OpCode], _handlers: *const ()) {
    return;
}

fn insn_noop(pc: usize, value: &mut i32, tape: &[OpCode], handlers: *const ()) {
    dispatch!(pc, value, tape, handlers);
}

fn insn_incr(pc: usize, value: &mut i32, tape: &[OpCode], handlers: *const ()) {
    *value += 1;
    dispatch!(pc, value, tape, handlers);
}

fn run(tape: &[OpCode], value: &mut i32) {
    fn start(pc: usize, value: &mut i32, tape: &[OpCode], handlers: *const ()) {
        dispatch!(pc, value, tape, handlers, start);
    }

    let handlers = HANDLERS.as_ptr() as *const ();
    start(0, value, tape, handlers);
}

fn main() {
    let mut tape = vec![OpCode::Incr, OpCode::NoOp, OpCode::Incr, OpCode::NoOp];

    for _ in 0..100000 {
        tape.push(OpCode::Incr);
    }

    tape.push(OpCode::Stop);

    let mut value = 0;
    run(&tape, &mut value);
    println!("value: {}", value);
}
