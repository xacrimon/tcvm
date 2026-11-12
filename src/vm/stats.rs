//! `TCVM_STATS=FILE`: what a run reached, written when its `Lua` drops. One
//! line per count, sorted: `op NAME N` for the interpreter's dispatches of an
//! opcode (compiled code dispatches none), and `jit …` for the JIT's events.
//! The fuzzer reads it to see what its programs exercise.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use crate::instruction::Op;
use crate::vm::abi::{Handler, handler};

static REAL: [Handler; 256] = crate::vm::dispatch::TABLE;

pub(crate) struct Stats {
    ops: [Cell<u64>; 256],
    events: RefCell<BTreeMap<String, u64>>,
    on: Cell<bool>,
    file: Option<std::path::PathBuf>,
}

impl Stats {
    pub(crate) fn from_env() -> Stats {
        let file = std::env::var_os("TCVM_STATS")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from);
        Stats {
            ops: std::array::from_fn(|_| Cell::new(0)),
            events: RefCell::new(BTreeMap::new()),
            on: Cell::new(file.is_some()),
            file,
        }
    }

    pub(crate) fn on(&self) -> bool {
        self.on.get()
    }

    pub(crate) fn set_on(&self, on: bool) {
        self.on.set(on);
    }

    /// Count one `key` event.
    pub(crate) fn event(&self, key: impl FnOnce() -> String) {
        if self.on.get() {
            *self.events.borrow_mut().entry(key()).or_insert(0) += 1;
        }
    }

    pub(crate) fn report(&self) -> String {
        let mut lines: Vec<String> = (Op::ALL.iter())
            .map(|&op| (op, self.ops[op as u8 as usize].get()))
            .filter(|&(_, n)| n > 0)
            .map(|(op, n)| format!("op {} {n}", op.name()))
            .collect();
        lines.extend((self.events.borrow().iter()).map(|(key, n)| format!("{key} {n}")));
        lines.sort();
        lines.push(String::new());
        lines.join("\n")
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        if let Some(file) = &self.file
            && let Err(e) = std::fs::write(file, self.report())
        {
            eprintln!("tcvm: TCVM_STATS: {}: {e}", file.display());
        }
    }
}

/// The dispatch table of a runtime that counts (`on`) or not.
pub(crate) fn table(on: bool) -> [Handler; 256] {
    if on { [counted; 256] } else { REAL }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// Every opcode's handler while counting: counts the dispatch, then runs
    /// the opcode's own.
    slow fn counted {
        let op = insn.as_insn().opcode() as usize;
        let n = &rt.runtime().stats.ops[op];
        n.set(n.get() + 1);
        tail!(REAL[op])
    }
}
