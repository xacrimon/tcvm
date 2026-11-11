//! Calls and the ways into and out of a region's frame: the fast call of
//! a compiled callee, its miss, native and deep paths, the slow path
//! through a level, and tails into handlers.

use crate::env::function::NativeClosure;
use crate::env::value::{NIL, PAYLOAD_MASK as PTR_MASK};
use crate::jit::backend::aarch64::abi::{self, BASE, CLOSURE, INSN, PC, RT, THREAD, X16, X17};
use crate::jit::backend::aarch64::asm::{Asm, Cond, Extend, Gpr, LR, Label, SP, Sz, ZR};
use crate::jit::backend::aarch64::emit::{Em, known_offset};
use crate::jit::backend::aarch64::inst::MInst;
use crate::jit::backend::emit::{SiteOut, site_entries};
use crate::jit::backend::lower::Known;
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
    /// The fast entry in x17.
    Reg,
}

impl<'a> Em<'a> {
    /// The fast path of a call whose header is at `base + hdr`, `insn` the
    /// argument count: a Lua callee without varargs whose prototype has a
    /// fast entry is called with `blr`, this frame kept, and its `ret` lands
    /// past the resume block's frame push (5.4); a native goes to `native`
    /// (`native_path`), anything else to `slow`. Every register is free
    /// here: what lives across the call is in the frame. Returns
    /// `fast_enter`'s labels.
    pub(super) fn fast_call(&mut self, hdr: i32, slow: Label, native: Label) -> (Label, Label) {
        self.a.ldr(X16, BASE, hdr);
        self.a.lsr_imm(Sz::X, X17, X16, 48);
        self.a.movz(Sz::W, Gpr(9), FUNCTION_TAG, 0);
        self.a.cmp(Sz::W, X17, Gpr(9));
        self.a.b_cond(Cond::Ne, slow);
        self.a.and_imm(Sz::X, X16, X16, PTR_MASK);
        self.a.ldrb(X17, X16, layout::FN_IS_VARARG as i32);
        self.a.cbnz(Sz::W, X17, native);
        self.a.ldr(X17, X16, layout::FN_PROTO as i32);
        self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
        self.a.cbz(Sz::X, X17, slow);
        self.fast_enter(hdr, Target::Reg, None)
    }

    /// `fast_call` for a known callee (D2), word 0 its closure word: the
    /// constant `closure`'s, or the stored value's. With a static argument
    /// count it enters past the arity and native stack checks, the
    /// parameters left out stored nil (I3), by a direct branch
    /// (`direct_target`); else it loads the fast entry, `slow` without
    /// one. Returns `fast_enter`'s labels, whether it enters past the
    /// checks, and where.
    pub(super) fn known_call(
        &mut self,
        hdr: i32,
        slow: Label,
        (closure, proto, nparams): (Option<u64>, u64, u8),
        nargs: u8,
        window: Option<Label>,
    ) -> (Label, Label, bool, Target) {
        // A static count passes the first arguments in x0.. (W4 §2), the
        // parameters left out nil; past eight they are in the callee's slots,
        // inside its window, which its known entry checks (the value stack's
        // mapping runs past any window that fits).
        let known = nargs != u8::MAX;
        if known && nargs < nparams {
            self.a.mov_imm(Gpr(9), NIL);
            for k in nargs..nparams {
                if k < 8 {
                    self.a.mov(Gpr(k), Gpr(9));
                } else {
                    self.a.str(Gpr(9), BASE, hdr + 32 + 8 * k as i32);
                }
            }
        }
        let target = match self.direct_target(proto, nparams, known) {
            // A typed self-call enters past the entry guards (W4 §2).
            Some(_) if window.is_some() => Target::Own(self.typed_entry.unwrap()),
            Some(t) if known => t,
            _ => {
                self.a.mov_imm(X17, proto);
                self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
                // Without a fast entry, the slow path, which finds every
                // argument in the callee's slots.
                let to = if known {
                    let stores = self.a.new_label();
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(stores);
                        em.store_args(hdr + 32, nargs);
                        em.a.b(slow);
                    }));
                    stores
                } else {
                    slow
                };
                self.a.cbz(Sz::X, X17, to);
                if known {
                    let off = known_offset(true, nparams);
                    self.a.add_imm(Sz::X, X17, X17, off as u64);
                }
                Target::Reg
            }
        };
        match closure {
            Some(f) => self.a.mov_imm(X16, f & PTR_MASK),
            None => {
                self.a.ldr(X16, BASE, hdr);
                self.a.and_imm(Sz::X, X16, X16, PTR_MASK);
            }
        }
        let (ret, deep) = self.fast_enter(hdr, target, window);
        (ret, deep, known, target)
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

    /// `bl` (`b` unless `link`) to `target`, a direct one recorded for
    /// `jit::set_fast`.
    pub(super) fn branch_to(&mut self, target: Target, link: bool) {
        match (target, link) {
            (Target::Own(l), true) => self.a.bl(l),
            (Target::Own(l), false) => self.a.b(l),
            (Target::Direct(proto, t, regs), _) => {
                let at = self.a.new_label();
                self.a.bind(at);
                self.direct.push((at, proto, regs));
                if link {
                    self.a.bl_far(t);
                } else {
                    self.a.b_far(t);
                }
            }
            (Target::Reg, true) => self.a.blr(X17),
            (Target::Reg, false) => self.a.br(X17),
        }
    }

    /// Enter `target` (`Target::Reg`: the fast entry in x17) with the
    /// closure in x16 (`fast_call`). Returns the return address the call
    /// leaves, where the fast resume is when the resume block is laid out
    /// next (else a branch to it, `fall_into_resume`), and the deep path's
    /// label, for `deep_path`.
    pub(super) fn fast_enter(
        &mut self,
        hdr: i32,
        target: Target,
        window: Option<Label>,
    ) -> (Label, Label) {
        // Word 1 zero: the frame's flags, which helpers set there (a closure
        // capturing a register, `HAS_OPEN`) and `HasFlag` reads. Left stale,
        // a flag would land on a value of the caller's that the collector
        // traces once the call returns.
        if hdr <= 504 {
            self.a.stp(X16, ZR, BASE, hdr);
        } else {
            self.a.str(X16, BASE, hdr);
            self.a.str(ZR, BASE, hdr + 8);
        }
        self.a.add_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
        // The window check the typed entry leaves to its caller.
        if let Some(w) = window {
            self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
            self.window_end(self.own_window);
            self.a.cmp(Sz::X, X17, X16);
            self.a.b_cond(Cond::Hi, w);
        }
        let deep = self.a.new_label();
        self.a.mov_sp(Gpr(9), SP);
        self.a.ldr(Gpr(10), THREAD, layout::NATIVE_DEEP as i32);
        self.a.cmp(Sz::X, Gpr(9), Gpr(10));
        self.a.b_cond(Cond::Lo, deep);
        self.branch_to(target, true);
        let ret = self.a.new_label();
        self.a.bind(ret);
        (ret, deep)
    }

    /// After a fast call's `bl`: on into the fast resume, which is there
    /// unless the resume block is laid out elsewhere.
    pub(super) fn fall_into_resume(&mut self, block: usize, next: Option<Label>) {
        let r = self.v.succs(block)[0].index();
        if Some(self.labels[r]) != next {
            self.a.b(self.resume[r].fast);
        }
    }

    /// Where a fast callee whose entry checks failed goes (its site's
    /// `miss`): the caller's `BASE` and word 0's callee value back, then its
    /// slow path at `slow`.
    pub(super) fn miss_path(&mut self, hdr: i32, block: usize, slow: Label) {
        self.a
            .bind(self.resume[self.v.succs(block)[0].index()].miss);
        self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
        self.a.ldr(X16, BASE, hdr);
        self.a.movz(Sz::X, X17, FUNCTION_TAG, 3);
        self.a.orr(Sz::X, X16, X16, X17);
        self.a.str(X16, BASE, hdr);
        self.a.b(slow);
    }

    /// A native callee, run by `jit_call_native` on this stack (I1): from
    /// `check`, `fast_call`'s test with the function in x16 and its
    /// `FN_IS_VARARG` byte in w17 (1 for a vararg Lua callee, which goes to
    /// `slow`, the enum's tag above that), or the constant closure `nc`,
    /// whose argument count is in x2 already (else in `INSN`). A
    /// plain return lands as a fast callee's; anything else goes on in a
    /// level running `native_act` on the frame, which returns through
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
                self.a.cmp_imm(Sz::W, X17, 1);
                self.a.b_cond(Cond::Eq, slow);
                self.a.add_imm(Sz::X, Gpr(3), X16, layout::FN_NATIVE as u64);
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
                self.a.mov_imm(Gpr(3), nc);
            }
        }
        let win = hdr as u64 + 32;
        self.a.ldr_lit(Gpr(0), self.links.rt);
        self.a.add_imm(Sz::X, Gpr(1), BASE, win);
        if from.is_ok() {
            self.a.mov(Gpr(2), INSN);
        }
        let ret = self.a.new_label();
        if switches {
            // A stub that switches returns with a branch (the return
            // address stack's top is the other thread's call), so it is
            // entered with one: neither touches that stack.
            self.a.adr(LR, ret);
            self.a.b_far(entry);
        } else {
            self.a.bl_far(entry);
        }
        self.a.bind(ret);
        let outcome = self.a.new_label();
        self.a.cmn_imm(Sz::X, Gpr(1), 1);
        if self.resume_reads_count(block) {
            self.a.b_cond(Cond::Eq, outcome);
            self.a.mov(INSN, Gpr(1));
            self.a.add_imm(Sz::X, PC, BASE, win);
            self.a.b(resume);
        } else {
            self.a.b_cond(Cond::Ne, resume);
        }
        self.a.bind(outcome);
        self.a.ldr(Gpr(9), BASE, hdr + 8);
        self.a
            .and_imm(Sz::X, Gpr(9), Gpr(9), crate::vm::frame::flag::MASK);
        self.a.ldr_lit(Gpr(10), addr!(abi::ret_to_native) as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, hdr + 8);
        self.a.mov(INSN, Gpr(0));
        self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
        self.a.add_imm(Sz::X, X17, BASE, win);
        self.handler_regs();
        self.a
            .ldr_lit(X16, addr!(crate::vm::native::native_act) as u64);
        self.a.bl_far(addr!(abi::jit_level));
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.b(resume);
        vec![(ret, false), (back, true)]
    }

    /// Whether the resume block after `block` reads the result count and
    /// the values (`INSN`, `PC`), not just the first result in x0.
    pub(super) fn resume_reads_count(&self, block: usize) -> bool {
        let r = self.v.succs(block)[0].index();
        !self.v.insts[self.v.blocks[r].start as usize].plain_resume()
    }

    /// Deep in a chain the return address goes in tagged (bit 0), so the
    /// callee returns with `br` and the return address stack stays as it
    /// was, right for the shallower frames' `ret`s. It names the fast resume
    /// itself, before which a branch to the miss path sits.
    pub(super) fn deep_path(
        &mut self,
        deep: Label,
        block: usize,
        known: Option<u8>,
        nargs: u8,
        target: Target,
    ) {
        self.a.bind(deep);
        // A register call misses with its arguments in the callee's slots.
        let miss = match known {
            Some(_) => self.a.new_label(),
            None => self.resume[self.v.succs(block)[0].index()].miss,
        };
        match target {
            Target::Own(l) => self.a.adr(X17, l),
            Target::Direct(proto, _, regs) => {
                // The fast entry, or the miss path without one.
                self.a.mov_imm(X17, proto);
                self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
                self.a.cbz(Sz::X, X17, miss);
                let off = known_offset(regs, known.unwrap_or(0));
                self.a.add_imm(Sz::X, X17, X17, off as u64);
            }
            Target::Reg => {}
        }
        if known.is_some() {
            // The known entry trusts the native stack, which is deep enough
            // here to check: below the limit the callee would miss, and its
            // full entry read arguments from slots this call did not write.
            self.a.ldr(Gpr(10), THREAD, layout::NATIVE_LIMIT as i32);
            self.a.cmp(Sz::X, Gpr(9), Gpr(10));
            self.a.b_cond(Cond::Lo, miss);
        }
        self.a
            .adr(LR, self.resume[self.v.succs(block)[0].index()].fast);
        self.a.orr_imm(Sz::X, LR, LR, 1);
        self.a.br(X17);
        if known.is_some() {
            self.a.bind(miss);
            self.store_args(0, nargs);
            self.a.b(self.resume[self.v.succs(block)[0].index()].miss);
        }
    }

    /// A register call's arguments (W4 §2) to the callee's slots, the first
    /// at `off` from `BASE`.
    pub(super) fn store_args(&mut self, off: i32, nargs: u8) {
        for k in 0..nargs.min(8) {
            self.a.str(Gpr(k), BASE, off + 8 * k as i32);
        }
    }

    /// The slow path of a call whose header at `hdr` holds the callee's
    /// value: a level runs it through `enter` (H6.2), with `ret_to_native`
    /// to come back. Returns the address the level returns to, from where
    /// the resume block goes on past its `BASE` (a level's return restores
    /// this frame's).
    pub(super) fn level_call(&mut self, hdr: i32, block: usize, pc_after: usize) -> Label {
        self.a.ldr_lit(X16, addr!(abi::ret_to_native) as u64);
        self.a.str(X16, BASE, hdr + 8);
        self.a.str(BASE, BASE, hdr + 16);
        self.a.ldr_lit(X16, pc_after as u64);
        self.a.str(X16, BASE, hdr + 24);
        self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
        self.a.add_imm(Sz::X, X17, BASE, hdr as u64 + 32);
        self.handler_regs();
        self.a
            .ldr_lit(X16, addr!(crate::vm::ops::call::enter) as u64);
        self.a.bl_far(addr!(abi::jit_level));
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.b(self.resume[self.v.succs(block)[0].index()].based);
        back
    }

    /// The frame is closed and `x16` holds the handler that goes on: one
    /// dispatch entered goes on in its level; a fast-called one starts a
    /// level returning to its caller, its header made to return through it.
    pub(super) fn tail_level(&mut self) {
        self.handler_regs();
        self.a.ldr_lit(X17, abi::level_land() as u64);
        self.a.cmp(Sz::X, LR, X17);
        self.a.b_cond(Cond::Ne, self.level_tail);
        self.a.ldr_lit(LR, abi::level_return_addr() as u64);
        self.a.br(X16);
    }

    /// `tail_level`'s level, once: a typed return address goes back as its
    /// landing, which tests the result (the level's is any value).
    pub(super) fn level_tail_stub(&mut self) {
        self.a.bind(self.level_tail);
        if !self.landings.is_empty() {
            let done = self.a.new_label();
            self.a.and_imm(Sz::X, Gpr(10), LR, !1);
            for (ret, boxed) in self.landings.clone() {
                let next = self.a.new_label();
                self.a.adr(Gpr(9), ret);
                self.a.cmp(Sz::X, Gpr(10), Gpr(9));
                self.a.b_cond(Cond::Ne, next);
                self.a.adr(Gpr(9), boxed);
                self.a.and_imm(Sz::X, LR, LR, 1);
                self.a.orr(Sz::X, LR, LR, Gpr(9));
                self.a.b(done);
                self.a.bind(next);
            }
            self.a.bind(done);
        }
        self.a.ldr(Gpr(9), BASE, -24);
        self.a
            .and_imm(Sz::X, Gpr(9), Gpr(9), crate::vm::frame::flag::MASK);
        self.a.ldr_lit(Gpr(10), addr!(abi::ret_to_native) as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, -24);
        self.a.mov(X17, BASE);
        self.a.b_far(addr!(abi::jit_level));
    }

    /// The handler ABI's runtime and closure registers, for code going on
    /// into a handler or a level: the frame's closure from its header.
    pub(super) fn handler_regs(&mut self) {
        self.a.ldr_lit(RT, self.links.rt);
        self.a.ldr(CLOSURE, BASE, -32);
        self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
    }

    /// Record the call at VCode instruction `i` and its return addresses:
    /// past `blr`, the fast resume (a tagged one's), and a level's.
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

    /// `x17 = BASE + 8 * slots`, the end of a window to check.
    pub(super) fn window_end(&mut self, slots: u16) {
        let off = slots as u64 * 8;
        if Asm::addsub_imm(off).is_some() {
            self.a.add_imm(Sz::X, X17, BASE, off);
        } else {
            self.a.mov_imm(X17, off);
            self.a.add(Sz::X, X17, BASE, X17);
        }
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
        // A known native takes the count in x2 (`native_path`).
        let count = if matches!(known, Some(Known::Native(_))) {
            Gpr(2)
        } else {
            INSN
        };
        if nargs == u8::MAX {
            let n = self.r(i, 0);
            self.a.mov(count, n);
        } else {
            self.a.movz(Sz::X, count, nargs as u16, 0);
        }
        if let Some(Known::Native(nc)) = known {
            let keys = self.native_path(hdr, block, Err(nc));
            // No fast callee misses to a native's site.
            self.a
                .bind(self.resume[self.v.succs(block)[0].index()].miss);
            self.a.brk(0xf2);
            self.call_site(keys, i, block, a as u16, pc_after);
            return;
        }
        let (slow, common) = (self.a.new_label(), self.a.new_label());
        let native = self.a.new_label();
        let window = (typed && self.typed_entry.is_some()).then(|| self.a.new_label());
        let (ret, deep, entered_known, target) = match known {
            Some(Known::Lua {
                closure,
                proto,
                nparams,
            }) => self.known_call(hdr, slow, (closure, proto, nparams), nargs, window),
            _ => {
                let (r, d) = self.fast_call(hdr, slow, native);
                (r, d, false, Target::Reg)
            }
        };
        self.fall_into_resume(block, next);
        let nparams = match known {
            Some(Known::Lua { nparams, .. }) if entered_known => Some(nparams),
            _ => None,
        };
        self.ool.push(Box::new(move |em: &mut Em<'_>| {
            let resume = em.v.succs(block)[0].index();
            if let Some(w) = window {
                // The callee's window does not fit: its slow path
                // takes the arguments from its slots.
                em.a.bind(w);
                em.store_args(0, nargs);
                em.a.b(em.resume[resume].miss);
            }
            em.miss_path(hdr, block, common);
            em.deep_path(deep, block, nparams, nargs, target);
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
                em.a.mov_imm(X16, f);
                em.a.str(X16, BASE, hdr);
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
        self.a.ldr_lit(INSN, word);
        self.a.ldr_lit(PC, pc_after as u64);
        self.a.ldr_lit(X16, self.links.rt);
        self.a.ldr(X16, X16, op as i32 * 8);
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
            // its entry past the arity and native stack checks, the
            // stack no deeper than this frame's caller left it.
            if nargs < nparams {
                self.a.mov_imm(X16, NIL);
                for k in nargs..nparams {
                    self.a.str(X16, BASE, 8 * k as i32);
                }
            }
            self.set_frame_closure();
            self.a.movz(Sz::X, INSN, nargs.max(nparams) as u16, 0);
            self.close_frame();
            self.branch_to(target, false);
            return;
        }
        let (top, done) = (self.a.new_label(), self.a.new_label());
        self.a.ldrb(X16, CLOSURE, layout::FN_NUM_PARAMS as i32);
        self.a.movz(Sz::X, X17, nargs as u16, 0);
        self.a.mov_imm(INSN, NIL);
        self.a.bind(top);
        self.a.cmp(Sz::X, X17, X16);
        self.a.b_cond(Cond::Hs, done);
        self.a.str_idx(INSN, BASE, X17, Extend::Lsl, true);
        self.a.add_imm(Sz::X, X17, X17, 1);
        self.a.b(top);
        self.a.bind(done);
        // The frame keeps its vararg count, so its results still land
        // where its caller expects.
        self.set_frame_closure();
        // A callee with a fast entry is entered there with this
        // frame popped, so it returns where this one would have: to a
        // compiled caller's `ret` or through the header (5.5).
        let dispatch = self.a.new_label();
        self.a.ldr(X16, CLOSURE, layout::FN_PROTO as i32);
        self.a.ldr(X16, X16, layout::PROTO_FAST as i32);
        self.a.cbz(Sz::X, X16, dispatch);
        self.a.mov(INSN, X17);
        self.close_frame();
        self.a.br(X16);
        self.a.bind(dispatch);
        self.close_frame();
        self.a.ldr(X16, CLOSURE, layout::FN_CODE as i32);
        self.a.ldr(INSN, X16, 0);
        self.a.add_imm(Sz::X, PC, X16, 8);
        self.a.and_imm(Sz::X, X17, INSN, 0xff);
        self.a.ldr_lit(X16, self.links.rt);
        self.a.ldr_idx(X16, X16, X17, Extend::Lsl, true);
        self.tail_level();
    }

    /// `MInst::Resume`: where the call's returns land, and its results.
    pub(super) fn on_resume(&mut self, i: usize, block: usize, wanted: u8, c: u8, a: u8, hdr: u16) {
        // A fast callee returns here (`fall_into_resume`).
        self.a.bind(self.resume[block].fast);
        // The callee's base is past the header at `hdr`.
        self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 * 8 + 32);
        if matches!(self.v.insts.get(i + 1), Some(MInst::GuardLanding { .. })) {
            self.pending_based = Some((block, hdr));
        } else if !(self.v.succs(block).first())
            .is_some_and(|s| self.resume[block].based == self.dest[s.index()])
        {
            // Else `thread_jumps` sent its returns where it jumps.
            self.a.bind(self.resume[block].based);
        }
        match wanted {
            0 if c == 1 => {}
            0 => {
                // The results to `R[a..]` (I5): all of them, `top`
                // past them, or `c - 1` filled out with nil. Up to
                // three here, more through the helper. They come from
                // at least a header above `R[a]`, so copying up is
                // safe, and reading past them stays in the stack.
                // The first is in x0: a compiled callee returning one
                // stores none (W4 §4).
                let (helper, top, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                let at = a as i32 * 8;
                if c == 0 {
                    self.a.cmp_imm(Sz::X, INSN, 3);
                    self.a.b_cond(Cond::Hi, helper);
                    for k in 0..3 {
                        self.a.cmp_imm(Sz::X, INSN, k);
                        self.a.b_cond(Cond::Eq, top);
                        if k == 0 {
                            self.a.str(Gpr(0), BASE, at);
                        } else {
                            self.a.ldr(X16, PC, 8 * k as i32);
                            self.a.str(X16, BASE, at + 8 * k as i32);
                        }
                    }
                    self.a.bind(top);
                    self.set_top(INSN, at as u64);
                    self.a.b(done);
                } else if c <= 4 {
                    // All `c - 1` there (a callee's count rarely
                    // varies): copied; else filled out of line.
                    let n = (c - 1) as u64;
                    let fill = self.a.new_label();
                    self.a.cmp_imm(Sz::X, INSN, n);
                    self.a.b_cond(Cond::Lo, fill);
                    for k in 0..n {
                        if k == 0 {
                            self.a.str(Gpr(0), BASE, at);
                        } else {
                            self.a.ldr(X16, PC, 8 * k as i32);
                            self.a.str(X16, BASE, at + 8 * k as i32);
                        }
                    }
                    self.a.b(done);
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(fill);
                        em.a.mov_imm(X17, NIL);
                        for k in 0..n {
                            if k == 0 {
                                em.a.mov(X16, Gpr(0));
                            } else {
                                em.a.ldr(X16, PC, 8 * k as i32);
                            }
                            em.a.cmp_imm(Sz::X, INSN, k);
                            em.a.csel(Sz::X, X16, X16, X17, Cond::Hi);
                            em.a.str(X16, BASE, at + 8 * k as i32);
                        }
                        em.a.b(done);
                    }));
                }
                self.a.bind(helper);
                // The helper reads the first result from memory too.
                let first = self.a.new_label();
                self.a.cbz(Sz::X, INSN, first);
                self.a.str(Gpr(0), PC, 0);
                self.a.bind(first);
                // Land through the helper: (rt, thread, base, a, values, nret, c).
                self.a.ldr_lit(Gpr(0), self.links.rt);
                self.a.mov(Gpr(1), THREAD);
                self.a.mov(Gpr(4), PC);
                self.a.mov(Gpr(5), INSN);
                self.a.mov(Gpr(2), BASE);
                self.a.movz(Sz::X, Gpr(3), a as u16, 0);
                self.a.movz(Sz::X, Gpr(6), c as u16, 0);
                self.a.bl_far(addr!(crate::jit::helpers::jit_land));
                self.a.bind(done);
            }
            // The first result is in x0 (D3).
            1 => {}
            _ => {
                let done = self.a.new_label();
                let d1 = self.r(i, 1);
                self.a.mov_imm(d1, NIL);
                self.a.cmp_imm(Sz::X, INSN, 2);
                self.a.b_cond(Cond::Lo, done);
                self.a.ldr(d1, PC, 8);
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
            self.a.str(v, BASE, at);
        }
        self.a.movz(Sz::X, INSN, nargs as u16, 0);
        let (slow, native) = (self.a.new_label(), self.a.new_label());
        let (ret, deep) = self.fast_call(h, slow, native);
        self.fall_into_resume(block, next);
        self.ool.push(Box::new(move |em: &mut Em<'_>| {
            em.miss_path(h, block, slow);
            em.deep_path(deep, block, None, 0, Target::Reg);
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
    pub(super) fn ret(&mut self, a: u8, n: u8, x0: bool) {
        // To a compiled caller, or to `level_land` for a frame
        // dispatch entered; a tagged return address is a deep one's
        // (`deep_path`).
        if n == 0 {
            self.a.mov_imm(Gpr(0), NIL);
        } else if !x0 {
            self.a.ldr(Gpr(0), BASE, a as i32 * 8);
        }
        self.close_frame();
        self.a.movz(Sz::X, INSN, n as u16, 0);
        self.a.add_imm(Sz::X, PC, BASE, a as u64 * 8);
        let deep = self.a.new_label();
        self.a.tbnz(LR, 0, deep);
        self.a.ret();
        self.a.bind(deep);
        self.a.and_imm(Sz::X, LR, LR, !1u64);
        self.a.br(LR);
    }
}
