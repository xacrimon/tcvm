//! The ways into a region and the stubs after its blocks: the fast entry
//! with its fixups and misses, the dispatch entry, and the exit stubs with
//! the region's exit trampoline.

use crate::env::value::NIL;
use crate::jit::backend::emit::FastEntry;
use crate::jit::backend::x86_64::abi::{self, ARGS, BASE, INSN, RT, S0, S1, THREAD};
use crate::jit::backend::x86_64::asm::{Arith, Cond, Label, Mem, RAX, RSP, Shift, Sz};
use crate::jit::backend::x86_64::emit::{Em, Entries, arity_check, known_offset, param_loads};
use crate::jit::backend::x86_64::types::type_fail;
use crate::jit::layout;

impl<'a> Em<'a> {
    /// The fast entry, where the region has one, then the entry guards
    /// (on `ARGS`) ahead of the entry block, where typed self-calls enter.
    ///
    /// A compiled caller has written the header, set base and `insn` (the
    /// argument count) as `enter` would, and its `call` pushed the return
    /// address; what `enter` would check of a Lua callee is checked here, and a
    /// miss goes to the caller's slow path (`abi::direct_miss`). Missing
    /// arguments are filled in. The native stack and the window are checked
    /// past both known entries: no caller vouches for them.
    pub(super) fn entries(&mut self, fast: Option<FastEntry>) -> Entries {
        let l = self.l;
        let nreg = l.params;
        let e = Entries {
            fast: fast.map(|_| self.a.new_label()),
            known: self.a.new_label(),
            fixup: self.a.new_label(),
            fast_miss: self.a.new_label(),
            param_miss: self.a.new_label(),
            guards: self.a.new_label(),
        };
        if let (Some(fe), Some(entry)) = (fast, e.fast) {
            self.a.bind(entry);
            arity_check(&mut self.a, fe.num_params, e.fixup);
            self.a.bind(e.known);
            assert_eq!(
                self.a.label_offset(e.known) - self.a.label_offset(entry),
                known_offset(false, fe.num_params)
            );
            // The parameters a caller left in slots to their registers, where a
            // call passing them there enters.
            param_loads(&mut self.a, nreg);
            let regs = self.a.new_label();
            self.a.bind(regs);
            assert_eq!(
                self.a.label_offset(regs) - self.a.label_offset(entry),
                known_offset(true, fe.num_params)
            );
            self.own_entry = Some((fe.proto, e.known, regs));
            self.own_window = l.window.unwrap_or(fe.max_stack as u16);
            // A miss from here leaves the parameters in their slots for the
            // caller's slow path (from the slot entry, again).
            let miss = if nreg > 0 { e.param_miss } else { e.fast_miss };
            self.a
                .cmp(Sz::Q, RSP, Mem::B(THREAD, layout::NATIVE_LIMIT as i32));
            self.a.jcc(Cond::B, miss);
            self.window_check(self.own_window);
            self.a.jcc(Cond::A, miss);
        }
        self.a.bind(e.guards);
        for &(r, set) in &l.entry_tests {
            type_fail(&mut self.a, ARGS[r as usize], set, self.entry_fail);
        }
        if l.prefix_guards {
            self.typed_entry = Some(self.labels[0]);
        }
        e
    }

    /// Where dispatch enters, from a handler: the thread to its register,
    /// `level_return`'s address off the stack, then through `level_enter`,
    /// which returns to `level_land`, on to the entry guards.
    pub(super) fn dispatch_entry(&mut self, guards: Label) -> Label {
        let l = self.l;
        let dispatch = self.a.new_label();
        self.a.bind(dispatch);
        self.a
            .load(Sz::Q, THREAD, Mem::B(RT, layout::RT_THREAD as i32));
        param_loads(&mut self.a, l.params);
        self.a.alu_imm(Arith::Add, Sz::Q, RSP, 8);
        // The interpreter made sure of the frame's window, not of the inlined
        // frames' past it (`Lowered::window`): `jit_window` grows the stack.
        match l.window {
            Some(w) => {
                let check = self.a.new_label();
                self.a.lea(Sz::Q, S0, Mem::L(check));
                self.a.jmp_far(addr!(abi::level_enter));
                self.a.bind(check);
                self.window_check(w);
                self.a.jcc(Cond::Be, guards);
                self.a.load(Sz::Q, INSN, Mem::L(self.region_word));
                self.a.mov_imm(S0, addr!(crate::jit::jit_window) as u64);
                self.tail_level();
            }
            None => {
                self.a.lea(Sz::Q, S0, Mem::L(guards));
                self.a.jmp_far(addr!(abi::level_enter));
            }
        }
        dispatch
    }

    /// One stub per exit and the region's exit trampoline, which they
    /// share.
    pub(super) fn exit_stubs(&mut self) {
        for k in 0..self.exit_labels.len() {
            self.a.bind(self.exit_labels[k]);
            self.a.mov_imm(S0, k as u64);
            self.a.jmp(self.exit_region);
        }
        self.a.bind(self.exit_region);
        self.a.shift_imm(Shift::Shl, Sz::Q, S0, 48);
        self.a.alu(Arith::Or, Sz::Q, S0, Mem::L(self.region_word));
        self.a.jmp_far(addr!(abi::exit_common));
    }

    /// The stubs of the entries: an invalidated region's, the entry guards'
    /// failure (which needs no register image), and the fast entry's arity
    /// fixup and misses.
    pub(super) fn entry_stubs(&mut self, e: &Entries, fast: Option<FastEntry>) {
        let l = self.l;
        let nreg = l.params;
        if l.invalidable {
            // An invalidated region's entry, its frame not open yet: on to
            // `jit_dead_entry` as the entry-fail stub goes on, the parameters a
            // register call passed stored for the interpreter.
            let dead = self.a.new_label();
            self.a.bind(dead);
            self.store_args(0, l.params);
            self.a.load(Sz::Q, INSN, Mem::L(self.region_word));
            self.a.mov_imm(S0, addr!(crate::jit::jit_dead_entry) as u64);
            self.tail_level();
            self.lazy.insert(0, (self.labels[0], dead));
        }
        if !l.entry_regs.is_empty() {
            // The interpreter runs the function from its slots, which a call
            // passing the parameters in registers did not write.
            self.a.bind(self.entry_fail);
            self.store_args(0, l.params);
            if !l.prefix_guards {
                self.close_frame();
            }
            self.a.load(Sz::Q, INSN, Mem::L(self.region_word));
            self.a.mov_imm(S0, addr!(crate::jit::jit_entry_fail) as u64);
            self.tail_level();
        }
        if let Some(fe) = fast {
            // The arguments short of the parameters are nil, once the window
            // fits; then on to the checks past the arity one. No parameter
            // is in a register yet.
            self.a.bind(e.fixup);
            self.window_check(self.own_window);
            self.a.jcc(Cond::A, e.fast_miss);
            self.a.lea(Sz::Q, S0, Mem::X(BASE, INSN, 3, 0));
            self.a
                .lea(Sz::Q, S1, Mem::B(BASE, fe.num_params as i32 * 8));
            self.a.mov_imm(RAX, NIL);
            let fill = self.a.new_label();
            self.a.bind(fill);
            self.a.store(Sz::Q, Mem::B(S0, 0), RAX);
            self.a.alu_imm(Arith::Add, Sz::Q, S0, 8);
            self.a.cmp(Sz::Q, S0, S1);
            self.a.jcc(Cond::B, fill);
            self.a.jmp(e.known);
        }
        if e.fast.is_some() && nreg > 0 {
            self.a.bind(e.param_miss);
            self.store_args(0, nreg);
        }
        if e.fast.is_some() {
            self.a.bind(e.fast_miss);
            self.a.jmp_far(addr!(abi::direct_miss));
        }
    }
}
