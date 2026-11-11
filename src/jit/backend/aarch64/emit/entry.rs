//! The ways into a region and the stubs after its blocks: the fast entry
//! with its fixups and misses, the dispatch entry, and the exit stubs with
//! the region's exit trampoline.

use crate::env::value::NIL;
use crate::jit::backend::aarch64::abi::{self, BASE, INSN, THREAD, X16, X17};
use crate::jit::backend::aarch64::asm::{Cond, Gpr, Label, SP, Shift, Sz};
use crate::jit::backend::aarch64::emit::{Em, Entries, known_offset};
use crate::jit::backend::aarch64::types::type_fail;
use crate::jit::backend::emit::FastEntry;
use crate::jit::layout;

impl<'a> Em<'a> {
    /// The fast entry, where the region has one, then the entry guards
    /// (on `x0`..) ahead of the entry block, where typed self-calls enter.
    ///
    /// A compiled caller has written the header, set base, closure and `insn`
    /// (the argument count) as `enter` would, and its `blr` left the return
    /// address in x30; what `enter` would check of a Lua callee is checked
    /// here, and a miss goes to the caller's slow path (`abi::direct_miss`).
    /// Missing arguments are filled in.
    pub(super) fn entries(&mut self, fast: Option<FastEntry>) -> Entries {
        let l = self.l;
        let nreg = l.params;
        let e = Entries {
            fast: fast.map(|_| self.a.new_label()),
            budget: self.a.new_label(),
            fixup: self.a.new_label(),
            fast_miss: self.a.new_label(),
            param_miss: self.a.new_label(),
            guards: self.a.new_label(),
        };
        if let (Some(fe), Some(entry)) = (fast, e.fast) {
            self.a.bind(entry);
            self.a.cmp_imm(Sz::X, INSN, fe.num_params as u64);
            self.a.b_cond(Cond::Lo, e.fixup);
            self.a.bind(e.budget);
            self.a.mov_sp(X16, SP);
            self.a.ldr(X17, THREAD, layout::NATIVE_LIMIT as i32);
            self.a.cmp(Sz::X, X16, X17);
            self.a.b_cond(Cond::Lo, e.fast_miss);
            let (known, regs) = (self.a.new_label(), self.a.new_label());
            self.a.bind(known);
            assert_eq!(
                self.a.label_offset(known) - self.a.label_offset(entry),
                known_offset(false, fe.num_params)
            );
            // The parameters a caller left in slots to their registers, where a
            // call passing them there enters.
            for k in 0..nreg {
                self.a.ldr(Gpr(k), BASE, 8 * k as i32);
            }
            self.a.bind(regs);
            assert_eq!(
                self.a.label_offset(regs) - self.a.label_offset(entry),
                known_offset(true, fe.num_params)
            );
            self.own_entry = Some((fe.proto, known, regs));
            self.own_window = l.window.unwrap_or(fe.max_stack as u16);
            self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
            self.window_end(self.own_window);
            self.a.cmp(Sz::X, X17, X16);
            // A miss from the register entry leaves the parameters in their
            // slots for the caller's slow path (from the slot one, again).
            self.a
                .b_cond(Cond::Hi, if nreg > 0 { e.param_miss } else { e.fast_miss });
        }
        self.a.bind(e.guards);
        for &(r, set) in &l.entry_tests {
            type_fail(&mut self.a, Gpr(r), set, self.entry_fail, &[]);
        }
        if l.prefix_guards {
            self.typed_entry = Some(self.labels[0]);
        }
        e
    }

    /// Where dispatch enters: through `level_enter`, which returns to
    /// `level_land`, on to the entry guards.
    pub(super) fn dispatch_entry(&mut self, guards: Label) -> Label {
        let l = self.l;
        let dispatch = self.a.new_label();
        self.a.bind(dispatch);
        for k in 0..l.params {
            self.a.ldr(Gpr(k), BASE, 8 * k as i32);
        }
        // The interpreter made sure of the frame's window, not of the inlined
        // frames' past it (`Lowered::window`): `jit_window` grows the stack.
        match l.window {
            Some(w) => {
                let check = self.a.new_label();
                self.a.adr(X16, check);
                self.a.b_far(addr!(abi::level_enter));
                self.a.bind(check);
                self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
                self.window_end(w);
                self.a.cmp(Sz::X, X17, X16);
                self.a.b_cond(Cond::Ls, guards);
                self.a.ldr_label(INSN, self.region_word);
                self.a.ldr_lit(X16, addr!(crate::jit::jit_window) as u64);
                self.tail_level();
            }
            None => {
                self.a.adr(X16, guards);
                self.a.b_far(addr!(abi::level_enter));
            }
        }
        dispatch
    }

    /// One stub per exit and the region's exit trampoline, which they
    /// share.
    pub(super) fn exit_stubs(&mut self) {
        for k in 0..self.exit_labels.len() {
            self.a.bind(self.exit_labels[k]);
            self.a.movz(Sz::W, X16, k as u16, 0);
            self.a.b(self.exit_region);
        }
        self.a.align(8);
        self.a.bind(self.exit_region);
        self.a.ldr_label(X17, self.region_word);
        self.a.orr_shift(Sz::X, X16, X17, X16, Shift::Lsl, 48);
        self.a.b_far(addr!(abi::exit_common));
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
            for k in 0..l.params {
                self.a.str(Gpr(k), BASE, 8 * k as i32);
            }
            self.a.ldr_label(INSN, self.region_word);
            self.a
                .ldr_lit(X16, addr!(crate::jit::jit_dead_entry) as u64);
            self.tail_level();
            self.lazy.insert(0, (self.labels[0], dead));
        }
        if !l.entry_regs.is_empty() {
            // The interpreter runs the function from its slots, which a call
            // passing the parameters in registers did not write.
            self.a.bind(self.entry_fail);
            for k in 0..l.params {
                self.a.str(Gpr(k), BASE, 8 * k as i32);
            }
            if !l.prefix_guards {
                self.close_frame();
            }
            self.a.ldr_label(INSN, self.region_word);
            self.a
                .ldr_lit(X16, addr!(crate::jit::jit_entry_fail) as u64);
            self.tail_level();
        }
        if let Some(fe) = fast {
            // The arguments short of the parameters are nil, once the window
            // fits; then on to the checks past the arity one.
            self.a.bind(e.fixup);
            self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
            self.window_end(self.own_window);
            self.a.cmp(Sz::X, X17, X16);
            self.a.b_cond(Cond::Hi, e.fast_miss);
            self.a.add_shift(Sz::X, X17, BASE, INSN, Shift::Lsl, 3);
            self.a
                .add_imm(Sz::X, Gpr(9), BASE, fe.num_params as u64 * 8);
            self.a.mov_imm(X16, NIL);
            let fill = self.a.new_label();
            self.a.bind(fill);
            self.a.str(X16, X17, 0);
            self.a.add_imm(Sz::X, X17, X17, 8);
            self.a.cmp(Sz::X, X17, Gpr(9));
            self.a.b_cond(Cond::Lo, fill);
            self.a.b(e.budget);
        }
        if e.fast.is_some() && nreg > 0 {
            self.a.bind(e.param_miss);
            for k in 0..nreg {
                self.a.str(Gpr(k), BASE, 8 * k as i32);
            }
        }
        if e.fast.is_some() {
            self.a.bind(e.fast_miss);
            self.a.b_far(addr!(abi::direct_miss));
        }
    }
}
