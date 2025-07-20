#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

#[derive(Clone, Copy)]
enum OpCode {
    Stop,
    NoOp,
    Incr
}

struct State {
    pc: usize,
    tape: Vec<OpCode>,
    value: i32,
}

type Handler = fn(&mut State);

const HANDLERS: [Handler; 3] = [
    insn_stop,
    insn_noop,
    insn_incr,
];

macro_rules! dispatch {
    ($state:expr) => {{
        dispatch!($state, false);
    }};

    ($state:expr, start) => {{
        dispatch!($state, true);
    }};

    ($state:expr, $start:expr) => {{
        if !$start {
            $state.pc += 1;
        }

        let pc = $state.pc;
        let op = $state.tape[pc];
        let handler = HANDLERS[op as usize];
        become handler($state);
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
    let mut tape = vec![
        OpCode::Incr,
        OpCode::NoOp,
        OpCode::Incr,
        OpCode::NoOp,
    ];

    for _ in 0..100000 {
        tape.push(OpCode::Incr);
    }

    tape.push(OpCode::Stop);

    let mut state = State {
        pc: 0,
        tape,
        value: 0,
    };

    run(&mut state);
    println!("value: {}", state.value);
}
