#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

#[derive(Clone, Copy)]
#[repr(u8)]
enum OpCode {
    Stop = 0,
    NoOp = 1,
    Incr = 2,
}

const HANDLERS: &[Handler] = &[insn_stop, insn_noop, insn_incr];

struct State {
    pc: usize,
    value: i32,
    tape: Vec<OpCode>,
    handlers: &'static [Handler],
}

type Handler = fn(&mut State);

macro_rules! dispatch {
    ($state:expr) => {{
        $state.pc += 1;
        dispatch!($state, impl);
    }};

    ($state:expr, start) => {{
        dispatch!($state, impl);
    }};

    ($state:expr, impl) => {{
        unsafe {
            let op = *$state.tape.get_unchecked($state.pc);
            let handler = *$state.handlers.get_unchecked(op as usize);
            become handler($state);
        }
    }};
}

#[inline(never)]
fn insn_stop(_state: &mut State) {
    return;
}

#[inline(never)]
fn insn_noop(state: &mut State) {
    dispatch!(state);
}

#[inline(never)]
fn insn_incr(state: &mut State) {
    state.value += 1;
    dispatch!(state);
}

#[inline(never)]
fn run(state: &mut State) {
    dispatch!(state, start);
}

fn main() {
    let mut tape = vec![OpCode::Incr, OpCode::NoOp, OpCode::Incr, OpCode::NoOp];

    for _ in 0..100000 {
        tape.push(OpCode::Incr);
    }

    tape.push(OpCode::Stop);

    let mut state = State {
        pc: 0,
        value: 0,
        tape,
        handlers: HANDLERS,
    };

    run(&mut state);
    println!("value: {}", state.value);
}
