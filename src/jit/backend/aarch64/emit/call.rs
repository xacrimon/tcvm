//! Lua calls: the fast call of a compiled callee in line, and out of line
//! its miss, native and deep paths and the slow path through a level; where
//! the returns land; and the call's site for the collector and frame walks.

use crate::env::value::{NIL, PAYLOAD_MASK as PTR_MASK};
use crate::jit::backend::aarch64::abi::{self, BASE, CLOSURE, INSN, PC, RT, THREAD, X16, X17};
use crate::jit::backend::aarch64::asm::{Asm, Cond, Gpr, LR, Label, SP, Shift, Sz, ZR};
use crate::jit::backend::aarch64::emit::Em;
use crate::jit::backend::emit::{FastEntry, SiteOut, Src};
use crate::jit::bug;
use crate::jit::layout::{self, FUNCTION_TAG};
use crate::jit::region::{Loc, SnapEntry};

/// A resume block's labels.
#[derive(Clone, Copy)]
pub(super) struct ResumeLabels {
    /// Where a fast callee's `ret` lands.
    fast: Label,
    /// Its call's miss path, which its site names (`abi::call_miss`).
    miss: Label,
    /// Past the fast resume's `BASE`, which a native call keeps and a
    /// level's return restores.
    based: Label,
}

/// A call's site, with its return addresses (and whether each is a
/// level's) and its miss path.
pub(super) type Site = (Vec<(Label, bool)>, Label, SiteOut);

/// Code out of the way of the fast paths, emitted after the blocks.
pub(super) enum Ool {
    /// A call's paths besides its fast one, so that its callee returns into
    /// the resume block.
    Call {
        i: usize,
        block: usize,
        hdr: i32,
        slow: Label,
        native: Label,
        ret: Label,
        deep: Label,
        pc_after: u64,
    },
    /// `n` results from slot offset `at`, those the callee left out nil.
    Fill {
        fill: Label,
        done: Label,
        n: u64,
        at: i32,
    },
}

/// A fast entry's labels: the entry, its arity check passed, the arity
/// fixup, and the miss.
#[derive(Clone, Copy)]
pub(super) struct Entries {
    pub(super) fast: Label,
    budget: Label,
    fixup: Label,
    miss: Label,
}

impl<'a> Em<'a> {
    /// The fast entry, ahead of the entry block, which follows it when
    /// `falls`. A compiled caller has written the header's first words, set
    /// the base and `insn` (the argument count), and its `blr` left the
    /// return address in x30; what `enter` would check of a Lua callee is
    /// checked here, and a miss goes to the caller's slow path
    /// (`abi::direct_miss`). Missing arguments are filled in.
    pub(super) fn fast_entry(&mut self, fe: FastEntry, falls: bool) -> Entries {
        let e = Entries {
            fast: self.a.new_label(),
            budget: self.a.new_label(),
            fixup: self.a.new_label(),
            miss: self.a.new_label(),
        };
        self.a.bind(e.fast);
        self.a.cmp_imm(Sz::X, INSN, fe.num_params as u64);
        self.a.b_cond(Cond::Lo, e.fixup);
        self.a.bind(e.budget);
        self.a.mov_sp(X16, SP);
        self.a.ldr(X17, THREAD, layout::NATIVE_LIMIT as i32);
        self.a.cmp(Sz::X, X16, X17);
        self.a.b_cond(Cond::Lo, e.miss);
        let known = self.a.new_label();
        self.a.bind(known);
        let off = self.a.label_offset(known) - self.a.label_offset(e.fast);
        if off != super::known_offset(false, fe.num_params) {
            bug!("the known entry is {off} bytes past the fast one");
        }
        self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
        self.window_end(fe.max_stack);
        self.a.cmp(Sz::X, X17, X16);
        self.a.b_cond(Cond::Hi, e.miss);
        if !falls {
            self.a.b(self.labels[0]);
        }
        e
    }

    /// The fast entry's arity fixup and miss.
    pub(super) fn fast_stubs(&mut self, fe: FastEntry, e: Entries) {
        // The arguments short of the parameters are nil, once the window
        // fits; then on to the checks past the arity one.
        self.a.bind(e.fixup);
        self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
        self.window_end(fe.max_stack);
        self.a.cmp(Sz::X, X17, X16);
        self.a.b_cond(Cond::Hi, e.miss);
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
        self.a.bind(e.miss);
        self.a.b_far(addr!(abi::direct_miss));
    }

    /// An invalidated region's entry, its frame not open yet: on to
    /// `jit_dead_entry`, which runs the function in the interpreter.
    pub(super) fn dead_entry(&mut self) {
        let dead = self.a.new_label();
        self.a.bind(dead);
        self.a.ldr_label(INSN, self.region_word);
        self.a
            .ldr_lit(X16, addr!(crate::jit::jit_dead_entry) as u64);
        self.tail_level();
        self.lazy.insert(0, (self.labels[0], dead));
    }

    /// The frame is closed and `x16` holds the handler that goes on: one
    /// dispatch entered goes on in its level; a fast-called one starts a
    /// level returning to its caller, its header made to return through it.
    pub(super) fn tail_level(&mut self) {
        self.handler_regs();
        self.a.ldr_lit(X17, abi::level_land() as u64);
        self.a.cmp(Sz::X, LR, X17);
        self.a.b_cond(Cond::Ne, self.level_tail.0);
        self.level_tail.1 = true;
        self.a.ldr_lit(LR, abi::level_return_addr() as u64);
        self.a.br(X16);
    }

    /// `tail_level`'s level, once.
    pub(super) fn level_tail_stub(&mut self) {
        if !self.level_tail.1 {
            return;
        }
        self.a.bind(self.level_tail.0);
        self.a.ldr(Gpr(9), BASE, -24);
        self.a
            .and_imm(Sz::X, Gpr(9), Gpr(9), crate::vm::frame::flag::MASK);
        self.a.ldr_lit(Gpr(10), addr!(abi::ret_to_native) as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, -24);
        self.a.mov(X17, BASE);
        self.a.b_far(addr!(abi::jit_level));
    }

    /// `x17 = BASE + 8 * slots`, the end of a window to check.
    fn window_end(&mut self, slots: u16) {
        let off = slots as u64 * 8;
        if Asm::addsub_imm(off).is_some() {
            self.a.add_imm(Sz::X, X17, BASE, off);
        } else {
            self.a.mov_imm(X17, off);
            self.a.add(Sz::X, X17, BASE, X17);
        }
    }

    pub(super) fn resume_labels(&mut self) -> ResumeLabels {
        ResumeLabels {
            fast: self.a.new_label(),
            miss: self.a.new_label(),
            based: self.a.new_label(),
        }
    }

    /// The labels of the block a call in `block` resumes in.
    fn resume_of(&self, block: usize) -> ResumeLabels {
        match self.resume[self.v.succs(block)[0].index()] {
            Some(r) => r,
            None => bug!("a call into no resume block"),
        }
    }

    /// `MOp::CallLua`. A Lua callee without varargs whose prototype has a
    /// fast entry is called with `blr`, this frame kept, and its `ret` lands
    /// in the resume block; a native goes to the native path, anything else
    /// to the slow one. Every register is free here: what lives across the
    /// call is in the frame.
    pub(super) fn call(&mut self, i: usize, block: usize, next: Option<Label>) {
        let imm = self.v.insts[i].imm;
        let (a, nargs, pc) = (imm & 0xff, imm >> 8 & 0xff, imm >> 32);
        let hdr = a as i32 * 8;
        let (slow, native, deep) = (self.a.new_label(), self.a.new_label(), self.a.new_label());
        if nargs == 255 {
            // Up to `top`.
            self.a.ldr(X16, THREAD, layout::THREAD_STACK as i32);
            self.a.ldr(X17, THREAD, layout::THREAD_TOP as i32);
            self.a.add_shift(Sz::X, X16, X16, X17, Shift::Lsl, 3);
            self.a.add_imm(Sz::X, X17, BASE, hdr as u64 + 32);
            self.a.sub(Sz::X, X16, X16, X17);
            self.a.lsr_imm(Sz::X, INSN, X16, 3);
        } else {
            self.a.movz(Sz::X, INSN, nargs as u16, 0);
        }
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
        // Word 1 zero: the frame's flags, which helpers set there and read.
        // Left stale, a flag would land on a value of the caller's that the
        // collector traces once the call returns.
        if hdr <= 504 {
            self.a.stp(X16, ZR, BASE, hdr);
        } else {
            self.a.str(X16, BASE, hdr);
            self.a.str(ZR, BASE, hdr + 8);
        }
        self.a.add_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
        self.a.mov_sp(Gpr(9), SP);
        self.a.ldr(Gpr(10), THREAD, layout::NATIVE_DEEP as i32);
        self.a.cmp(Sz::X, Gpr(9), Gpr(10));
        self.a.b_cond(Cond::Lo, deep);
        self.a.blr(X17);
        let ret = self.a.new_label();
        self.a.bind(ret);
        // On into the fast resume, which is there unless the resume block
        // is laid out elsewhere.
        let r = self.v.succs(block)[0].index();
        if Some(self.labels[r]) != next {
            self.a.b(self.resume_of(block).fast);
        }
        self.ool.push(Ool::Call {
            i,
            block,
            hdr,
            slow,
            native,
            ret,
            deep,
            pc_after: self.code + pc * 8,
        });
    }

    pub(super) fn out_of_line(&mut self, o: Ool) {
        match o {
            Ool::Call {
                i,
                block,
                hdr,
                slow,
                native,
                ret,
                deep,
                pc_after,
            } => {
                let r = self.resume_of(block);
                // Where a fast callee whose entry checks failed goes: the
                // caller's `BASE` and word 0's callee value back, then the
                // slow path.
                self.a.bind(r.miss);
                self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
                self.a.ldr(X16, BASE, hdr);
                self.a.movz(Sz::X, X17, FUNCTION_TAG, 3);
                self.a.orr(Sz::X, X16, X16, X17);
                self.a.str(X16, BASE, hdr);
                self.a.b(slow);
                // Deep in a chain the return address goes in tagged (bit 0),
                // so the callee returns with `br` and the return address
                // stack stays right for the shallower frames' `ret`s.
                self.a.bind(deep);
                self.a.adr(LR, r.fast);
                self.a.orr_imm(Sz::X, LR, LR, 1);
                self.a.br(X17);
                let mut keys = vec![(ret, false)];
                keys.extend(self.native_path(hdr, block, native, slow));
                self.a.bind(slow);
                let back = self.level_call(hdr, block, pc_after);
                keys.extend([(r.fast, false), (back, true)]);
                self.call_site(keys, i, block, hdr as u16 / 8, pc_after);
            }
            Ool::Fill { fill, done, n, at } => {
                self.a.bind(fill);
                self.a.mov_imm(X17, NIL);
                for k in 0..n {
                    if k == 0 {
                        self.a.mov(X16, Gpr(0));
                    } else {
                        self.a.ldr(X16, PC, 8 * k as i32);
                    }
                    self.a.cmp_imm(Sz::X, INSN, k);
                    self.a.csel(Sz::X, X16, X16, X17, Cond::Hi);
                    self.a.str(X16, BASE, at + 8 * k as i32);
                }
                self.a.b(done);
            }
        }
    }

    /// A native callee, run by `jit_call_native` on this stack: from
    /// `check`, the call's test with the function in x16 and its
    /// `FN_IS_VARARG` byte in w17 (1 for a vararg Lua callee, which goes to
    /// `slow`, the enum's tag above that). A plain return lands as a fast
    /// callee's; anything else goes on in a level running `native_act` on
    /// the frame, which returns through `ret_to_native`. Returns the return
    /// addresses, for the site.
    fn native_path(
        &mut self,
        hdr: i32,
        block: usize,
        check: Label,
        slow: Label,
    ) -> [(Label, bool); 2] {
        let resume = self.resume_of(block).based;
        self.a.bind(check);
        self.a.cmp_imm(Sz::W, X17, 1);
        self.a.b_cond(Cond::Eq, slow);
        self.a.add_imm(Sz::X, Gpr(3), X16, layout::FN_NATIVE as u64);
        let win = hdr as u64 + 32;
        self.a.ldr_lit(Gpr(0), self.rt);
        self.a.add_imm(Sz::X, Gpr(1), BASE, win);
        self.a.mov(Gpr(2), INSN);
        self.a.bl_far(addr!(abi::native_entry));
        let ret = self.a.new_label();
        self.a.bind(ret);
        let outcome = self.a.new_label();
        self.a.cmn_imm(Sz::X, Gpr(1), 1);
        let r = self.v.succs(block)[0].index();
        let plain = self.v.insts[self.v.blocks[r].start as usize].plain_resume();
        if plain {
            self.a.b_cond(Cond::Ne, resume);
        } else {
            self.a.b_cond(Cond::Eq, outcome);
            self.a.mov(INSN, Gpr(1));
            self.a.add_imm(Sz::X, PC, BASE, win);
            self.a.b(resume);
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
        [(ret, false), (back, true)]
    }

    /// The slow path of a call whose header at `hdr` holds the callee's
    /// value: a level runs it through `enter`, with `ret_to_native` to come
    /// back. Returns the address the level returns to, from where the
    /// resume block goes on past its `BASE` (a level's return restores this
    /// frame's).
    fn level_call(&mut self, hdr: i32, block: usize, pc_after: u64) -> Label {
        self.a.ldr_lit(X16, addr!(abi::ret_to_native) as u64);
        self.a.str(X16, BASE, hdr + 8);
        self.a.str(BASE, BASE, hdr + 16);
        self.a.ldr_lit(X16, pc_after);
        self.a.str(X16, BASE, hdr + 24);
        self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
        self.a.add_imm(Sz::X, X17, BASE, hdr as u64 + 32);
        self.handler_regs();
        self.a
            .ldr_lit(X16, addr!(crate::vm::ops::call::enter) as u64);
        self.a.bl_far(addr!(abi::jit_level));
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.b(self.resume_of(block).based);
        back
    }

    /// The handler ABI's runtime and closure registers, for code going on
    /// into a handler or a level: the frame's closure from its header.
    fn handler_regs(&mut self) {
        self.a.ldr_lit(RT, self.rt);
        self.a.ldr(CLOSURE, BASE, -32);
        self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
    }

    /// Record the call at instruction `i` with its return addresses: what
    /// it keeps in stack slots, and the registers whose frame slots hold
    /// theirs.
    fn call_site(
        &mut self,
        keys: Vec<(Label, bool)>,
        i: usize,
        block: usize,
        hdr: u16,
        pc_after: u64,
    ) {
        let Some(site) = self.s.sites.iter().find(|s| s.inst as usize == i) else {
            bug!("a call without its site")
        };
        let allocs = self.allocs(i);
        let mut consts = Vec::new();
        let entries = self.s.entries[site.at as usize..(site.at + site.len) as usize]
            .iter()
            .map(|&(reg, rep, src)| {
                let loc = match src {
                    Src::Const(w) => {
                        consts.push(w);
                        Loc::Const(consts.len() as u32 - 1)
                    }
                    Src::Operand(k) => match allocs[k as usize].as_stack() {
                        Some(slot) => Loc::Spill(slot.index() as u16),
                        None => bug!("a value kept across a call outside the frame"),
                    },
                    Src::Home => Loc::Slot(reg),
                };
                SnapEntry { reg, rep, loc }
            })
            .collect();
        let site = SiteOut {
            keys: Vec::new(),
            miss: 0,
            entries,
            consts,
            hdr,
            pc_after,
        };
        self.sites.push((keys, self.resume_of(block).miss, site));
    }

    /// `MOp::Resume`: where the call's returns land, and its results.
    pub(super) fn on_resume(&mut self, i: usize, block: usize) {
        let m = self.v.insts[i];
        let (a, c) = (m.imm & 0xff, m.imm >> 16 & 0xff);
        let Some(r) = self.resume[block] else {
            bug!("a resume outside a resume block")
        };
        // A fast callee returns here; its base is past the header.
        self.a.bind(r.fast);
        self.a.sub_imm(Sz::X, BASE, BASE, a * 8 + 32);
        self.a.bind(r.based);
        if m.plain_resume() {
            return;
        }
        // `c - 1` results to `R[a..]`, filled out with nil. They come from
        // at least a header above `R[a]`, so copying up is safe, and reading
        // past them stays in the stack. The first is in x0: a compiled
        // callee returning one stores none.
        let (at, n) = (a as i32 * 8, c.wrapping_sub(1));
        let done = self.a.new_label();
        let helper = self.a.new_label();
        if c == 0 {
            // All of them, `top` past them: up to three here.
            let top = self.a.new_label();
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
            self.a.ldr(X16, THREAD, layout::THREAD_STACK as i32);
            self.a.add_imm(Sz::X, X17, BASE, at as u64);
            self.a.sub(Sz::X, X17, X17, X16);
            self.a.add_shift(Sz::X, X17, INSN, X17, Shift::Lsr, 3);
            self.a.str(X17, THREAD, layout::THREAD_TOP as i32);
            self.a.b(done);
        }
        if (1..=4).contains(&c) {
            // All there (a callee's count rarely varies): copied.
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
            self.ool.push(Ool::Fill { fill, done, n, at });
        } else {
            self.a.bind(helper);
            // Through the helper, which reads the first result from memory
            // too: (thread, base, a, values, nret, c).
            let first = self.a.new_label();
            self.a.cbz(Sz::X, INSN, first);
            self.a.str(Gpr(0), PC, 0);
            self.a.bind(first);
            self.a.mov(Gpr(0), THREAD);
            self.a.mov(Gpr(3), PC);
            self.a.mov(Gpr(4), INSN);
            self.a.mov(Gpr(1), BASE);
            self.a.movz(Sz::X, Gpr(2), a as u16, 0);
            self.a.movz(Sz::X, Gpr(5), c as u16, 0);
            self.a.bl_far(addr!(crate::jit::helpers::jit_land));
        }
        self.a.bind(done);
    }
}
