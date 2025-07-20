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
    ($state:expr, $start:expr) => {{
        if !$start {
            $state.pc += 1;
        }

        let pc = $state.pc;
        let op = $state.tape[pc];
        let handler = HANDLERS[op as usize];
        handler($state);
    }};
}

fn insn_stop(state: &mut State) {
    return;
}

fn insn_noop(state: &mut State) {
    dispatch!(state, false);
}

fn insn_incr(state: &mut State) {
    state.value += 1;
    dispatch!(state, false);
}

fn main() {
    let tape = vec![
        OpCode::Incr,
        OpCode::NoOp,
        OpCode::Incr,
        OpCode::Stop,
    ];

    let mut state = State {
        pc: 0,
        tape,
        value: 0,
    };

    dispatch!(&mut state, true);
    println!("value: {}", state.value);
}
