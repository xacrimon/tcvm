//! Calls and the ways into and out of a region's frame: the fast call of
//! a compiled callee, its miss and native paths, the slow path through a
//! level, and tails into handlers.

use crate::env::function::NativeClosure;
use crate::env::value::{NIL, PAYLOAD_MASK as PTR_MASK};
use crate::jit::backend::emit::{SiteOut, site_entries};
use crate::jit::backend::lower::Known;
use crate::jit::backend::x86_64::abi::{
    self, ARGS, BASE, CLOSURE, INSN, PC, RESULTS, RT, S0, S1, THREAD,
};
use crate::jit::backend::x86_64::asm::{
    Arith, Cond, Label, Mem, R8, R9, RAX, RCX, RDI, RDX, RSI, RSP, Shift, Sz,
};
use crate::jit::backend::x86_64::emit::{Em, known_offset};
use crate::jit::backend::x86_64::inst::MInst;
use crate::jit::layout;
use crate::jit::layout::FUNCTION_TAG;

/// Where a known call enters its callee (D2).
#[derive(Clone, Copy)]
pub(super) enum Target {
    /// This region's own entry.
    Own(Label),
    /// A known entry of the prototype's fast entry, by a branch that
    /// `jit::set_fast` repoints (`Emitted::direct`): the prototype's box,
    /// the branch's first target, and whether it is the register entry.
    Direct(u64, usize, bool),
    /// The fast entry in `S1`.
    Reg,
}

impl<'a> Em<'a> {
    /// The fast path of a call whose header is at `base + hdr`, `insn` the
    /// argument count: a Lua callee without varargs whose prototype has a
    /// fast entry is called, this frame kept, and its `ret` lands past the
    /// resume block's frame push (5.4); a native goes to `native`
    /// (`native_path`), anything else to `slow`. Every register is free
    /// here: what lives across the call is in the frame. Returns the
    /// call's return address.
    pub(super) fn fast_call(&mut self, hdr: i32, slow: Label, native: Label) -> Label {
        self.a.load(Sz::Q, S0, Mem::B(BASE, hdr));
        self.a.mov(S1, S0);
        self.a.shift_imm(Shift::Shr, Sz::Q, S1, 48);
        self.a.cmp_imm(Sz::D, S1, FUNCTION_TAG as i32);
        self.a.jcc(Cond::Ne, slow);
        self.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
        self.a.load8(S1, Mem::B(S0, layout::FN_IS_VARARG as i32));
        self.a.test(Sz::D, S1, S1);
        self.a.jcc(Cond::Ne, native);
        self.a.load(Sz::Q, S1, Mem::B(S0, layout::FN_PROTO as i32));
        self.a
            .load(Sz::Q, S1, Mem::B(S1, layout::PROTO_FAST as i32));
        self.a.test(Sz::Q, S1, S1);
        self.a.jcc(Cond::E, slow);
        self.fast_enter(hdr, Target::Reg, None)
    }

    /// `fast_call` for a known callee (D2), word 0 its closure word: the
    /// constant `closure`'s, or the stored value's. With a static argument
    /// count it enters past the arity check, the parameters left out
    /// stored nil (I3), by a direct branch (`direct_target`); else it loads
    /// the fast entry, `slow` without one. Returns the call's return
    /// address, whether it enters past the check, and where.
    pub(super) fn known_call(
        &mut self,
        hdr: i32,
        slow: Label,
        (closure, proto, nparams): (Option<u64>, u64, u8),
        nargs: u8,
        window: Option<Label>,
    ) -> (Label, bool, Target) {
        // A static count passes the first arguments in `ARGS` (W4 §2), the
        // parameters left out nil; past them they are in the callee's slots,
        // inside its window, which its known entry checks (the value stack's
        // mapping runs past any window that fits).
        let known = nargs != u8::MAX;
        if known && nargs < nparams {
            self.a.mov_imm(S0, NIL);
            for k in nargs..nparams {
                match ARGS.get(k as usize) {
                    Some(&r) => self.a.mov(r, S0),
                    None => self
                        .a
                        .store(Sz::Q, Mem::B(BASE, hdr + 32 + 8 * k as i32), S0),
                }
            }
        }
        let target = match self.direct_target(proto, nparams, known) {
            // A typed self-call enters past the entry guards (W4 §2).
            Some(_) if window.is_some() => Target::Own(self.typed_entry.unwrap()),
            Some(t) if known => t,
            _ => {
                self.a.mov_imm(S1, proto);
                self.a
                    .load(Sz::Q, S1, Mem::B(S1, layout::PROTO_FAST as i32));
                // Without a fast entry, the slow path, which finds every
                // argument in the callee's slots.
                let to = if known {
                    let stores = self.a.new_label();
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(stores);
                        em.store_args(hdr + 32, nargs);
                        em.a.jmp(slow);
                    }));
                    stores
                } else {
                    slow
                };
                self.a.test(Sz::Q, S1, S1);
                self.a.jcc(Cond::E, to);
                if known {
                    let off = known_offset(true, nparams);
                    self.a.alu_imm(Arith::Add, Sz::Q, S1, off as i32);
                }
                Target::Reg
            }
        };
        match closure {
            Some(f) => self.a.mov_imm(S0, f & PTR_MASK),
            None => {
                self.a.load(Sz::Q, S0, Mem::B(BASE, hdr));
                self.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
            }
        }
        let ret = self.fast_enter(hdr, target, window);
        (ret, known, target)
    }

    /// A known entry of `proto` (its box, `nparams` parameters) a branch
    /// can take, the register one when `regs`: this region's own, or one
    /// `jit::set_fast` keeps the branch pointed at.
    pub(super) fn direct_target(&self, proto: u64, nparams: u8, regs: bool) -> Option<Target> {
        match self.own_entry {
            // The region cannot outlive itself (I4).
            Some((p, slots, r)) if p == proto => Some(Target::Own(if regs { r } else { slots })),
            _ => (self.links.direct.iter().find(|d| d.0 == proto)).map(|&(p, _, fast)| {
                Target::Direct(p, crate::jit::direct_target(fast, regs, nparams), regs)
            }),
        }
    }

    /// `call` (`jmp` unless `link`) to `target`, a direct one recorded for
    /// `jit::set_fast`.
    pub(super) fn branch_to(&mut self, target: Target, link: bool) {
        match (target, link) {
            (Target::Own(l), true) => self.a.call(l),
            (Target::Own(l), false) => self.a.jmp(l),
            (Target::Direct(proto, t, regs), _) => {
                let at = self.a.new_label();
                self.a.bind(at);
                self.direct.push((at, proto, regs));
                if link {
                    self.a.call_far(t);
                } else {
                    self.a.jmp_far(t);
                }
            }
            (Target::Reg, true) => self.a.call_reg(S1),
            (Target::Reg, false) => self.a.jmp_reg(S1),
        }
    }

    /// Enter `target` (`Target::Reg`: the fast entry in `S1`) with the
    /// closure in `S0` (`fast_call`). Returns the return address the call
    /// leaves, where the fast resume is when the resume block is laid out
    /// next (else a branch to it, `fall_into_resume`).
    pub(super) fn fast_enter(&mut self, hdr: i32, target: Target, window: Option<Label>) -> Label {
        // Word 1 zero: the frame's flags, which helpers set there (a closure
        // capturing a register, `HAS_OPEN`) and `HasFlag` reads. Left stale,
        // a flag would land on a value of the caller's that the collector
        // traces once the call returns.
        self.a.store(Sz::Q, Mem::B(BASE, hdr), S0);
        self.a.store_imm(Sz::Q, Mem::B(BASE, hdr + 8), 0);
        self.a.alu_imm(Arith::Add, Sz::Q, BASE, hdr + 32);
        // The window check the typed entry leaves to its caller.
        if let Some(w) = window {
            self.window_check(self.own_window);
            self.a.jcc(Cond::A, w);
        }
        self.branch_to(target, true);
        let ret = self.a.new_label();
        self.a.bind(ret);
        ret
    }

    /// After a fast call: on into the fast resume, which is there unless
    /// the resume block is laid out elsewhere.
    pub(super) fn fall_into_resume(&mut self, block: usize, next: Option<Label>) {
        let r = self.v.succs(block)[0].index();
        if Some(self.labels[r]) != next {
            self.a.jmp(self.resume[r].fast);
        }
    }

    /// Where a fast callee whose entry checks failed goes (its site's
    /// `miss`): the caller's `BASE` and word 0's callee value back, then its
    /// slow path at `slow`.
    pub(super) fn miss_path(&mut self, hdr: i32, block: usize, slow: Label) {
        self.a
            .bind(self.resume[self.v.succs(block)[0].index()].miss);
        self.a.alu_imm(Arith::Sub, Sz::Q, BASE, hdr + 32);
        self.a.mov_imm(S0, (FUNCTION_TAG as u64) << 48);
        self.a.alu_mr(Arith::Or, Sz::Q, Mem::B(BASE, hdr), S0);
        self.a.jmp(slow);
    }

    /// A native callee, run by `jit_call_native` on this stack (I1): from
    /// `check`, `fast_call`'s test with the function in `S0` and its
    /// `FN_IS_VARARG` byte in `S1` (1 for a vararg Lua callee, which goes to
    /// `slow`, the enum's tag above that), or the constant closure `nc`,
    /// whose argument count is in rdx already (else in `INSN`). A plain
    /// return lands as a fast callee's; anything else goes on in a level
    /// running `native_act` on the frame, which returns through
    /// `ret_to_native`. Returns the return addresses for `call_site`.
    pub(super) fn native_path(
        &mut self,
        hdr: i32,
        block: usize,
        from: Result<(Label, Label), u64>,
    ) -> Vec<(Label, bool)> {
        let resume = self.resume[self.v.succs(block)[0].index()].based;
        let mut entry = addr!(abi::native_entry);
        let mut switches = false;
        match from {
            Ok((check, slow)) => {
                self.a.bind(check);
                self.a.cmp_imm(Sz::D, S1, 1);
                self.a.jcc(Cond::E, slow);
                self.a.lea(Sz::Q, RCX, Mem::B(S0, layout::FN_NATIVE as i32));
            }
            Err(nc) => {
                // SAFETY: the region's constants keep the closure alive.
                if let Some(k) =
                    crate::vm::coro::switch_kind(unsafe { &*(nc as *const NativeClosure) })
                {
                    entry = [
                        addr!(abi::coro_resume),
                        addr!(abi::coro_wrap),
                        addr!(abi::coro_yield),
                    ][k as usize];
                    switches = true;
                }
                self.a.mov_imm(RCX, nc);
            }
        }
        let win = hdr + 32;
        self.a.mov_imm(RDI, self.links.rt);
        self.a.lea(Sz::Q, RSI, Mem::B(BASE, win));
        if from.is_ok() {
            self.a.mov(RDX, INSN);
        }
        let ret = self.a.new_label();
        if switches {
            // A stub that switches returns with a branch (the return
            // address stack's top is the other thread's call), so it is
            // entered with one: neither touches that stack.
            self.a.lea(Sz::Q, S0, Mem::L(ret));
            self.a.push(S0);
            self.a.jmp_far(entry);
        } else {
            self.a.call_far(entry);
        }
        self.a.bind(ret);
        let outcome = self.a.new_label();
        self.a.cmp_imm(Sz::Q, RDX, -1);
        if self.resume_reads_count(block) {
            self.a.jcc(Cond::E, outcome);
            self.a.mov(INSN, RDX);
            self.a.lea(Sz::Q, PC, Mem::B(BASE, win));
            self.a.jmp(resume);
        } else {
            self.a.jcc(Cond::Ne, resume);
        }
        self.a.bind(outcome);
        self.a.load(Sz::Q, S1, Mem::B(BASE, hdr + 8));
        self.a
            .alu_imm(Arith::And, Sz::Q, S1, crate::vm::frame::flag::MASK as i32);
        self.a.alu(
            Arith::Or,
            Sz::Q,
            S1,
            Mem::K(addr!(abi::ret_to_native) as u64),
        );
        self.a.store(Sz::Q, Mem::B(BASE, hdr + 8), S1);
        self.a.mov(INSN, RAX);
        self.a.lea(Sz::Q, PC, Mem::B(BASE, hdr));
        self.a.lea(Sz::Q, S1, Mem::B(BASE, win));
        self.handler_regs();
        self.a
            .mov_imm(S0, addr!(crate::vm::native::native_act) as u64);
        self.a.call_far(addr!(abi::jit_level));
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.jmp(resume);
        vec![(ret, false), (back, true)]
    }

    /// Whether the resume block after `block` reads the result count and
    /// the values (`INSN`, `PC`), not just the first result in rax.
    pub(super) fn resume_reads_count(&self, block: usize) -> bool {
        let r = self.v.succs(block)[0].index();
        !self.v.insts[self.v.blocks[r].start as usize].plain_resume()
    }

    /// A register call's arguments (W4 §2) to the callee's slots, the first
    /// at `off` from `BASE`.
    pub(super) fn store_args(&mut self, off: i32, nargs: u8) {
        for k in 0..nargs.min(ARGS.len() as u8) {
            self.a
                .store(Sz::Q, Mem::B(BASE, off + 8 * k as i32), ARGS[k as usize]);
        }
    }

    /// The slow path of a call whose header at `hdr` holds the callee's
    /// value: a level runs it through `enter` (H6.2), with `ret_to_native`
    /// to come back. Returns the address the level returns to, from where
    /// the resume block goes on past its `BASE` (a level's return restores
    /// this frame's).
    pub(super) fn level_call(&mut self, hdr: i32, block: usize, pc_after: usize) -> Label {
        self.a.mov_imm(S0, addr!(abi::ret_to_native) as u64);
        self.a.store(Sz::Q, Mem::B(BASE, hdr + 8), S0);
        self.a.store(Sz::Q, Mem::B(BASE, hdr + 16), BASE);
        self.a.mov_imm(S0, pc_after as u64);
        self.a.store(Sz::Q, Mem::B(BASE, hdr + 24), S0);
        self.a.lea(Sz::Q, PC, Mem::B(BASE, hdr));
        self.a.lea(Sz::Q, S1, Mem::B(BASE, hdr + 32));
        self.handler_regs();
        self.a
            .mov_imm(S0, addr!(crate::vm::ops::call::enter) as u64);
        self.a.call_far(addr!(abi::jit_level));
        let back = self.a.new_label();
        self.a.bind(back);
        self.a
            .jmp(self.resume[self.v.succs(block)[0].index()].based);
        back
    }

    /// The frame is closed and `S0` holds the handler that goes on: one
    /// dispatch entered goes on in its level, `level_return` for its
    /// return address again; a fast-called one starts a level returning to
    /// its caller, its header made to return through it.
    pub(super) fn tail_level(&mut self) {
        self.handler_regs();
        self.a.mov_imm(S1, abi::level_land() as u64);
        self.a.cmp(Sz::Q, S1, Mem::B(RSP, 0));
        self.a.jcc(Cond::Ne, self.level_tail);
        self.a.mov_imm(S1, abi::level_return_addr() as u64);
        self.a.store(Sz::Q, Mem::B(RSP, 0), S1);
        self.a.jmp_reg(S0);
    }

    /// `tail_level`'s level, once: a typed return address goes back as its
    /// landing, which tests the result (the level's is any value).
    pub(super) fn level_tail_stub(&mut self) {
        self.a.bind(self.level_tail);
        if !self.landings.is_empty() {
            let done = self.a.new_label();
            for (ret, boxed) in self.landings.clone() {
                let next = self.a.new_label();
                self.a.lea(Sz::Q, S1, Mem::L(ret));
                self.a.cmp(Sz::Q, S1, Mem::B(RSP, 0));
                self.a.jcc(Cond::Ne, next);
                self.a.lea(Sz::Q, S1, Mem::L(boxed));
                self.a.store(Sz::Q, Mem::B(RSP, 0), S1);
                self.a.jmp(done);
                self.a.bind(next);
            }
            self.a.bind(done);
        }
        self.a.load(Sz::Q, S1, Mem::B(BASE, -24));
        self.a
            .alu_imm(Arith::And, Sz::Q, S1, crate::vm::frame::flag::MASK as i32);
        self.a.alu(
            Arith::Or,
            Sz::Q,
            S1,
            Mem::K(addr!(abi::ret_to_native) as u64),
        );
        self.a.store(Sz::Q, Mem::B(BASE, -24), S1);
        self.a.mov(S1, BASE);
        self.a.jmp_far(addr!(abi::jit_level));
    }

    /// The handler ABI's runtime and closure registers, for code going on
    /// into a handler or a level: the frame's closure from its header.
    pub(super) fn handler_regs(&mut self) {
        self.a.mov_imm(RT, self.links.rt);
        self.a.load(Sz::Q, CLOSURE, Mem::B(BASE, -32));
        self.a.alu(Arith::And, Sz::Q, CLOSURE, Mem::K(PTR_MASK));
    }

    /// Record the call at VCode instruction `i` and its return addresses:
    /// past the call, the fast resume, and a level's.
    pub(super) fn call_site(
        &mut self,
        keys: Vec<(Label, bool)>,
        i: usize,
        block: usize,
        hdr: u16,
        pc_after: usize,
    ) {
        let resume = self.v.succs(block)[0].index();
        let site = self.l.sites.iter().find(|(b, _)| *b == resume);
        let site = &site.expect("a call's site").1;
        let snaps = &self.allocs(i)[self.l.snap_ops[i] as usize..];
        let (entries, consts) = site_entries(site, snaps, hdr);
        let frames = site.frames.clone();
        let miss = self.resume[resume].miss;
        self.sites.push(SiteOut {
            keys,
            miss,
            entries,
            consts,
            hdr,
            pc_after: pc_after as u64,
            frames,
        });
    }

    /// `MInst::Call`: the fast call in line, its other paths out of line.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn call(
        &mut self,
        i: usize,
        block: usize,
        next: Option<Label>,
        a: u8,
        nargs: u8,
        pc_after: usize,
        known: Option<Known>,
        typed: bool,
    ) {
        let hdr = a as i32 * 8;
        // A known native takes the count in rdx (`native_path`).
        let count = if matches!(known, Some(Known::Native(_))) {
            RDX
        } else {
            INSN
        };
        if nargs == u8::MAX {
            let n = self.r(i, 0);
            self.a.mov32(count, n);
        } else {
            self.a.mov_imm(count, nargs as u64);
        }
        if let Some(Known::Native(nc)) = known {
            let keys = self.native_path(hdr, block, Err(nc));
            // No fast callee misses to a native's site.
            self.a
                .bind(self.resume[self.v.succs(block)[0].index()].miss);
            self.a.ud2();
            self.call_site(keys, i, block, a as u16, pc_after);
            return;
        }
        let (slow, common) = (self.a.new_label(), self.a.new_label());
        let native = self.a.new_label();
        let window = (typed && self.typed_entry.is_some()).then(|| self.a.new_label());
        let (ret, target) = match known {
            Some(Known::Lua {
                closure,
                proto,
                nparams,
            }) => {
                let (r, _, t) =
                    self.known_call(hdr, slow, (closure, proto, nparams), nargs, window);
                (r, t)
            }
            _ => (self.fast_call(hdr, slow, native), Target::Reg),
        };
        self.fall_into_resume(block, next);
        self.ool.push(Box::new(move |em: &mut Em<'_>| {
            let resume = em.v.succs(block)[0].index();
            if let Some(w) = window {
                // The callee's window does not fit: its slow path
                // takes the arguments from its slots.
                em.a.bind(w);
                em.store_args(0, nargs);
                em.a.jmp(em.resume[resume].miss);
            }
            em.miss_path(hdr, block, common);
            let mut keys = vec![(ret, false)];
            if known.is_none() {
                keys.extend(em.native_path(hdr, block, Ok((native, slow))));
            }
            em.a.bind(slow);
            // Only a call through the loaded fast entry branches here
            // before writing word 0.
            if let (
                Some(Known::Lua {
                    closure: Some(f), ..
                }),
                Target::Reg,
            ) = (known, target)
            {
                em.a.mov_imm(S0, f);
                em.a.store(Sz::Q, Mem::B(BASE, hdr), S0);
            }
            em.a.bind(common);
            let back = em.level_call(hdr, block, pc_after);
            let fast = em.resume[resume].fast;
            keys.extend([(fast, false), (back, true)]);
            if let Some(boxed) = em.resume[resume].boxed {
                keys.push((boxed, false));
                em.landings.extend([(ret, boxed), (fast, boxed)]);
            }
            em.call_site(keys, i, block, a as u16, pc_after);
        }));
    }

    /// `MInst::Dispatch`.
    pub(super) fn dispatch(&mut self, op: u8, word: u64, pc_after: usize) {
        self.close_frame();
        self.a.mov_imm(INSN, word);
        self.a.mov_imm(PC, pc_after as u64);
        self.a.mov_imm(S0, self.links.rt + op as u64 * 8);
        self.a.load(Sz::Q, S0, Mem::B(S0, 0));
        self.tail_level();
    }

    /// `MInst::TailEnter`.
    pub(super) fn tail_enter(&mut self, i: usize, nargs: u8, known: Option<(u64, u8)>) {
        // The callee is the frame's closure from here, which frees
        // `insn` and `pc` (allocatable in the body) as scratch.
        let f = self.r(i, 0);
        self.a.mov(CLOSURE, f);
        let target = known.and_then(|(proto, nparams)| {
            Some((self.direct_target(proto, nparams, false)?, nparams))
        });
        if let Some((target, nparams)) = target {
            // A known callee (D2): the parameters left out nil, then
            // its entry past the arity check.
            if nargs < nparams {
                self.a.mov_imm(S0, NIL);
                for k in nargs..nparams {
                    self.a.store(Sz::Q, Mem::B(BASE, 8 * k as i32), S0);
                }
            }
            self.set_frame_closure();
            self.a.mov_imm(INSN, nargs.max(nparams) as u64);
            self.close_frame();
            self.branch_to(target, false);
            return;
        }
        let (top, done) = (self.a.new_label(), self.a.new_label());
        self.a
            .load8(S0, Mem::B(CLOSURE, layout::FN_NUM_PARAMS as i32));
        self.a.mov_imm(S1, nargs as u64);
        self.a.mov_imm(INSN, NIL);
        self.a.bind(top);
        self.a.cmp(Sz::Q, S1, S0);
        self.a.jcc(Cond::Ae, done);
        self.a.store(Sz::Q, Mem::X(BASE, S1, 3, 0), INSN);
        self.a.alu_imm(Arith::Add, Sz::Q, S1, 1);
        self.a.jmp(top);
        self.a.bind(done);
        // The frame keeps its vararg count, so its results still land
        // where its caller expects.
        self.set_frame_closure();
        // A callee with a fast entry is entered there with this
        // frame popped, so it returns where this one would have: to a
        // compiled caller's `ret` or through the header (5.5).
        let dispatch = self.a.new_label();
        self.a
            .load(Sz::Q, S0, Mem::B(CLOSURE, layout::FN_PROTO as i32));
        self.a
            .load(Sz::Q, S0, Mem::B(S0, layout::PROTO_FAST as i32));
        self.a.test(Sz::Q, S0, S0);
        self.a.jcc(Cond::E, dispatch);
        self.a.mov(INSN, S1);
        self.close_frame();
        self.a.jmp_reg(S0);
        self.a.bind(dispatch);
        self.close_frame();
        self.a
            .load(Sz::Q, S0, Mem::B(CLOSURE, layout::FN_CODE as i32));
        self.a.load(Sz::Q, INSN, Mem::B(S0, 0));
        self.a.lea(Sz::Q, PC, Mem::B(S0, 8));
        self.a.mov32(S1, INSN);
        self.a.alu_imm(Arith::And, Sz::D, S1, 0xff);
        self.a.mov_imm(S0, self.links.rt);
        self.a.load(Sz::Q, S0, Mem::X(S0, S1, 3, 0));
        self.tail_level();
    }

    /// `MInst::Resume`: where the call's returns land, and its results.
    pub(super) fn on_resume(&mut self, i: usize, block: usize, wanted: u8, c: u8, a: u8, hdr: u16) {
        // A fast callee returns here (`fall_into_resume`).
        self.a.bind(self.resume[block].fast);
        // The callee's base is past the header at `hdr`.
        self.a.alu_imm(Arith::Sub, Sz::Q, BASE, hdr as i32 * 8 + 32);
        if matches!(self.v.insts.get(i + 1), Some(MInst::GuardLanding { .. })) {
            self.pending_based = Some((block, hdr));
        } else if !(self.v.succs(block).first())
            .is_some_and(|s| self.resume[block].based == self.dest[s.index()])
        {
            // Else `thread_jumps` sent its returns where it jumps.
            self.a.bind(self.resume[block].based);
        }
        let first = RESULTS[0];
        match wanted {
            0 if c == 1 => {}
            0 => {
                // The results to `R[a..]` (I5): all of them, `top`
                // past them, or `c - 1` filled out with nil. Up to
                // three here, more through the helper. They come from
                // at least a header above `R[a]`, so copying up is
                // safe, and reading past them stays in the stack.
                // The first is in rax: a compiled callee returning one
                // stores none (W4 §4).
                let (helper, top, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                let at = a as i32 * 8;
                if c == 0 {
                    self.a.cmp_imm(Sz::Q, INSN, 3);
                    self.a.jcc(Cond::A, helper);
                    for k in 0..3 {
                        self.a.cmp_imm(Sz::Q, INSN, k);
                        self.a.jcc(Cond::E, top);
                        if k == 0 {
                            self.a.store(Sz::Q, Mem::B(BASE, at), first);
                        } else {
                            self.a.load(Sz::Q, S0, Mem::B(PC, 8 * k));
                            self.a.store(Sz::Q, Mem::B(BASE, at + 8 * k), S0);
                        }
                    }
                    self.a.bind(top);
                    self.set_top(INSN, at);
                    self.a.jmp(done);
                } else if c <= 4 {
                    // All `c - 1` there (a callee's count rarely
                    // varies): copied; else filled out of line.
                    let n = (c - 1) as i32;
                    let fill = self.a.new_label();
                    self.a.cmp_imm(Sz::Q, INSN, n);
                    self.a.jcc(Cond::B, fill);
                    for k in 0..n {
                        if k == 0 {
                            self.a.store(Sz::Q, Mem::B(BASE, at), first);
                        } else {
                            self.a.load(Sz::Q, S0, Mem::B(PC, 8 * k));
                            self.a.store(Sz::Q, Mem::B(BASE, at + 8 * k), S0);
                        }
                    }
                    self.a.jmp(done);
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(fill);
                        em.a.mov_imm(S1, NIL);
                        for k in 0..n {
                            if k == 0 {
                                em.a.mov(S0, first);
                            } else {
                                em.a.load(Sz::Q, S0, Mem::B(PC, 8 * k));
                            }
                            em.a.cmp_imm(Sz::Q, INSN, k);
                            em.a.cmov(Cond::Be, Sz::Q, S0, S1);
                            em.a.store(Sz::Q, Mem::B(BASE, at + 8 * k), S0);
                        }
                        em.a.jmp(done);
                    }));
                }
                self.a.bind(helper);
                // The helper reads the first result from memory too.
                let stored = self.a.new_label();
                self.a.test(Sz::Q, INSN, INSN);
                self.a.jcc(Cond::E, stored);
                self.a.store(Sz::Q, Mem::B(PC, 0), first);
                self.a.bind(stored);
                // Land through the helper: (thread, base, a, values, nret, c).
                self.a.mov(RDI, THREAD);
                self.a.mov(RSI, BASE);
                self.a.mov_imm(RDX, a as u64);
                self.a.mov(RCX, PC);
                self.a.mov(R8, INSN);
                self.a.mov_imm(R9, c as u64);
                self.a.call_far(addr!(crate::jit::helpers::jit_land));
                self.a.bind(done);
            }
            // The first result is in rax (D3).
            1 => {}
            _ => {
                let done = self.a.new_label();
                let d1 = self.r(i, 1);
                self.a.mov_imm(d1, NIL);
                self.a.cmp_imm(Sz::Q, INSN, 2);
                self.a.jcc(Cond::B, done);
                self.a.load(Sz::Q, d1, Mem::B(PC, 8));
                self.a.bind(done);
            }
        }
    }

    /// `MInst::CallMm`.
    pub(super) fn call_mm(
        &mut self,
        i: usize,
        block: usize,
        next: Option<Label>,
        hdr: u16,
        nargs: u8,
        pc_after: usize,
    ) {
        let h = hdr as i32 * 8;
        for k in 0..=nargs as usize {
            let v = self.r(i, k);
            let at = if k == 0 { h } else { h + 24 + 8 * k as i32 };
            self.a.store(Sz::Q, Mem::B(BASE, at), v);
        }
        self.a.mov_imm(INSN, nargs as u64);
        let (slow, native) = (self.a.new_label(), self.a.new_label());
        let ret = self.fast_call(h, slow, native);
        self.fall_into_resume(block, next);
        self.ool.push(Box::new(move |em: &mut Em<'_>| {
            em.miss_path(h, block, slow);
            let mut keys = vec![(ret, false)];
            keys.extend(em.native_path(h, block, Ok((native, slow))));
            em.a.bind(slow);
            let back = em.level_call(h, block, pc_after);
            let fast = em.resume[em.v.succs(block)[0].index()].fast;
            keys.extend([(fast, false), (back, true)]);
            em.call_site(keys, i, block, hdr, pc_after);
        }));
    }

    /// `MInst::Return`.
    pub(super) fn ret(&mut self, a: u8, n: u8, reg: bool) {
        // To a compiled caller, or to `level_land` for a frame dispatch
        // entered.
        if n == 0 {
            self.a.mov_imm(RESULTS[0], NIL);
        } else if !reg {
            self.a.load(Sz::Q, RESULTS[0], Mem::B(BASE, a as i32 * 8));
        }
        self.close_frame();
        self.a.mov_imm(INSN, n as u64);
        self.a.lea(Sz::Q, PC, Mem::B(BASE, a as i32 * 8));
        self.a.ret();
    }
}
