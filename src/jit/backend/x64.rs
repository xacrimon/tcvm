//! Machine IR to x86-64.
//!
//! The structure mirrors the aarch64 encoder deliberately — same block layout,
//! same exit-stub-as-location-map idea, same return convention — so the two read
//! side by side. What differs is forced by the ISA, and only that:
//!
//!   - **Two-address arithmetic.** `add d, s` is `d = d + s`; the machine IR is
//!     three-address. So `d = a op b` becomes `mov acc, a; op acc, b; mov d, acc`,
//!     computed through a fixed accumulator (`rax`) rather than into the
//!     destination, because the destination may alias `b`.
//!   - **Fixed-register instructions.** `idiv` divides `rdx:rax`; a variable shift
//!     counts in `cl`. `rax`, `rcx`, and `rdx` are therefore never handed to the
//!     allocator — reserving them lets those instructions use them freely.
//!   - **Fewer free registers.** aarch64 allocates only caller-saved registers
//!     because it has enough; x86-64 does not, so the allocatable set includes the
//!     callee-saved `rbx`/`r12`–`r15`, which the prologue saves and the epilogue
//!     restores.
//!   - **Float compares set only the unsigned flags.** `comisd` reports an
//!     unordered result as `CF=ZF=PF=1`, so the Lua-correct condition for `<` is
//!     `seta` after swapping the operands, and `==`/`~=` need the parity flag
//!     folded in.
//!
//! # Shape of the output
//!
//! ```text
//!   prologue                                     <- save callee-saved, open frame
//!   entry block, then the rest in index order    <- the fast path
//!   ...
//!   exit stubs                                    <- cold; one per IR Exit
//!   epilogue                                      <- restore, ret
//! ```
//!
//! Every guard falls through on success and jumps to its stub on failure, so the
//! fast path is a straight line and the stubs sit out of the way.
//!
//! # Return convention
//!
//! `rax` carries a packed status word back to the executor: see [`Status`]. The
//! epilogue's pops do not touch `rax`, so the word set just before jumping there
//! survives to the `ret`.

use crate::env::value::ValueKind;
use crate::jit::backend::layout;
use crate::jit::backend::mach::{
    AluOp, ExitId, ExitSrc, FAluOp, MBlock, MFunc, MOp, RegClass, Tag, VReg, Width,
};
use crate::jit::backend::regalloc::{
    Alloc, Allocation, Inst, MachineEnv, Move, PReg, ProgPoint, RegallocFunc,
};
use crate::jit::backend::x64_asm::{Asm, Cond, Gpr, Label, RAX, RBP, RCX, RDX, RSP, Xmm};
use crate::jit::ir::op::Cc;
use crate::jit::ir::pool::ConstPool;

/// What compiled code reports back, packed into `rax` as `(tag << 32) | payload`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// A guard failed or an uncompiled path was reached. The payload is the exit
    /// id; the frame has already been written back.
    Deopt(u32),
    /// The region ran to a Lua `return`. The payload is the number of results,
    /// which sit at `base + 0..nret`.
    Return(u32),
}

const TAG_DEOPT: u64 = 0;
const TAG_RETURN: u64 = 1;

impl Status {
    fn packed(tag: u64, payload: u32) -> i64 {
        ((tag << 32) | payload as u64) as i64
    }

    /// Decode what compiled code left in `rax`.
    pub fn unpack(word: u64) -> Status {
        let payload = word as u32;
        match word >> 32 {
            TAG_DEOPT => Status::Deopt(payload),
            TAG_RETURN => Status::Return(payload),
            other => panic!("compiled code returned an unknown status tag {other}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncodeError {
    /// More spill slots than a 32-bit frame displacement can address. Unreachable
    /// in practice — it needs hundreds of millions of live values.
    FrameTooLarge(u32),
}

// --- scratch registers ------------------------------------------------------
//
// Never handed to the allocator (see `machine_env`), so an instruction may
// clobber them between one value's load and the next. Compiled code makes no
// calls, so nothing else clobbers them either.

/// Operand-reload scratch: a spilled use is reloaded into one of these.
const S0: Gpr = Gpr(10); // r10
const S1: Gpr = Gpr(11); // r11

/// The accumulator every two-address op computes through, and the status word on
/// the way out. `rax`, so `idiv`'s quotient lands here for free.
const ACC: Gpr = RAX;

/// The incoming arguments: the thread (unused today) and the Lua frame base.
const ARG: [Gpr; 2] = [Gpr(7), Gpr(6)]; // rdi, rsi

/// Callee-saved registers the allocator may hand out; saved in the prologue and
/// restored in the epilogue. `rbx` plus `r12`–`r15`.
const CALLEE_SAVED: [Gpr; 5] = [Gpr(3), Gpr(12), Gpr(13), Gpr(14), Gpr(15)];

/// Float scratch. `FACC` is the two-address accumulator; `F0`/`F1` reload spilled
/// float operands (and `F0` doubles as the stack-to-stack edit path).
const FACC: Xmm = Xmm(13);
const F0: Xmm = Xmm(14);
const F1: Xmm = Xmm(15);

/// One spill slot is one machine word.
const SLOT: i32 = 8;

/// The registers the allocator may hand out.
///
/// What is *missing* is the load-bearing part, and it is missing for reasons that
/// live in this file:
///
///   rax        `ACC`, the accumulator and the status word
///   rcx        the `cl` shift count, and the exit stub's base pointer
///   rdx        `idiv`'s high dividend / remainder
///   rsi, rdi   incoming arguments (`rsi` is the frame base at entry)
///   rsp, rbp   stack and frame pointer
///   r10, r11   `S0`/`S1`, the operand-reload scratch
///   xmm13-15   `FACC`/`F0`/`F1`, the float scratch
///
/// The integer set is smaller than aarch64's, so it reaches into the callee-saved
/// registers; `xmm` has no callee-saved registers in the System V ABI, so all of
/// `xmm0`–`xmm12` are free.
pub fn machine_env() -> MachineEnv {
    // Caller-saved first (free to use), then callee-saved (a save/restore each).
    const INT: &[u8] = &[8, 9, 3, 12, 13, 14, 15]; // r8, r9, rbx, r12-r15
    const FLOAT: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    MachineEnv {
        allocation_order: [
            INT.iter().map(|&r| PReg::new(RegClass::Int, r)).collect(),
            FLOAT
                .iter()
                .map(|&r| PReg::new(RegClass::Float, r))
                .collect(),
        ],
    }
}

pub struct Encoder<'a, 'gc> {
    m: &'a MFunc,
    pool: &'a ConstPool<'gc>,
    ra: &'a Allocation,
    a: Asm,
    blocks: Vec<Label>,
    exits: Vec<Label>,
    epilogue: Label,
    /// Bytes of stack reserved for spill slots, 16-byte aligned.
    frame: i32,
}

/// Encode `m` to a flat byte stream, packed into little-endian words. Placing
/// those words in executable memory is the allocator's job (or, in tests,
/// [`Code::from_words`](crate::jit::backend::code::Code::from_words)).
pub fn encode(m: &MFunc, pool: &ConstPool<'_>, ra: &Allocation) -> Result<Vec<u32>, EncodeError> {
    debug_assert!(
        crate::jit::backend::regalloc::verify(m, ra).is_ok(),
        "{}",
        crate::jit::backend::regalloc::verify(m, ra).unwrap_err()
    );

    // Slot `s` lives at `[rsp + s*SLOT]`, so the top slot's displacement must fit
    // the 32-bit field. Only an absurd live set reaches this.
    if (ra.num_spills as u64) * (SLOT as u64) > i32::MAX as u64 {
        return Err(EncodeError::FrameTooLarge(ra.num_spills));
    }
    let frame = ((ra.num_spills * SLOT as u32).next_multiple_of(16)) as i32;

    let mut a = Asm::new();
    let blocks = (0..m.blocks.len()).map(|_| a.new_label()).collect();
    let exits = (0..m.exits.len()).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();

    let mut e = Encoder {
        m,
        pool,
        ra,
        a,
        blocks,
        exits,
        epilogue,
        frame,
    };
    e.run();

    Ok(e.a.finish())
}

impl Encoder<'_, '_> {
    fn run(&mut self) {
        self.prologue();

        // Reverse postorder puts the entry first so it falls through from the
        // prologue. The order is load-bearing, not cosmetic: the allocator numbered
        // its live intervals as spans of *this* linearization.
        for b in self.m.block_order() {
            self.block(b);
        }

        for e in 0..self.m.exits.len() as u32 {
            self.exit_stub(ExitId(e));
        }

        let epilogue = self.epilogue;
        self.a.bind(epilogue);
        // Undo the frame, restore callee-saved in reverse, return. `rax` (the
        // status word) is untouched by any of this.
        if self.frame > 0 {
            assert!(self.a.try_add_imm(RSP, self.frame as i64));
        }
        for &r in CALLEE_SAVED.iter().rev() {
            self.a.pop(r);
        }
        self.a.pop(RBP);
        self.a.ret();
    }

    fn prologue(&mut self) {
        self.a.push(RBP);
        self.a.mov(RBP, RSP);
        for &r in &CALLEE_SAVED {
            self.a.push(r);
        }
        if self.frame > 0 {
            assert!(self.a.try_sub_imm(RSP, self.frame as i64));
        }
    }

    // --- operand access -----------------------------------------------------

    fn read_g(&mut self, a: Alloc, scratch: Gpr) -> Gpr {
        match a {
            Alloc::Reg(r) => Gpr(r.num()),
            Alloc::Spill(s) => {
                self.a.load(scratch, RSP, s as i32 * SLOT);
                scratch
            }
        }
    }

    fn read_f(&mut self, a: Alloc, scratch: Xmm) -> Xmm {
        match a {
            Alloc::Reg(r) => Xmm(r.num()),
            Alloc::Spill(s) => {
                self.a.load_f(scratch, RSP, s as i32 * SLOT);
                scratch
            }
        }
    }

    fn use_g(&mut self, i: Inst, k: usize, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Int);
        let a = self.ra.use_(i, k);
        self.read_g(a, scratch)
    }

    fn use_f(&mut self, i: Inst, k: usize, scratch: Xmm) -> Xmm {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Float);
        let a = self.ra.use_(i, k);
        self.read_f(a, scratch)
    }

    /// Where to write def `k`, for a single-instruction op that writes its
    /// destination directly. Pair with [`Self::def_done`].
    fn def_g(&mut self, i: Inst, k: usize, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Int);
        match self.ra.def(i, k) {
            Alloc::Reg(r) => Gpr(r.num()),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_f(&mut self, i: Inst, k: usize, scratch: Xmm) -> Xmm {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Float);
        match self.ra.def(i, k) {
            Alloc::Reg(r) => Xmm(r.num()),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_done(&mut self, i: Inst, k: usize, from: Gpr) {
        if let Alloc::Spill(s) = self.ra.def(i, k) {
            self.a.store(RSP, s as i32 * SLOT, from);
        }
    }

    fn def_done_f(&mut self, i: Inst, k: usize, from: Xmm) {
        if let Alloc::Spill(s) = self.ra.def(i, k) {
            self.a.store_f(RSP, s as i32 * SLOT, from);
        }
    }

    /// Commit the accumulator (`rax`) to def `k`'s location — the tail of every
    /// two-address op, whose result lands in `ACC`.
    fn commit_g(&mut self, i: Inst, k: usize) {
        match self.ra.def(i, k) {
            Alloc::Reg(r) => self.a.mov(Gpr(r.num()), ACC),
            Alloc::Spill(s) => self.a.store(RSP, s as i32 * SLOT, ACC),
        }
    }

    /// Commit the float accumulator (`FACC`) to def `k`'s location.
    fn commit_f(&mut self, i: Inst, k: usize) {
        match self.ra.def(i, k) {
            Alloc::Reg(r) => self.a.movsd(Xmm(r.num()), FACC),
            Alloc::Spill(s) => self.a.store_f(RSP, s as i32 * SLOT, FACC),
        }
    }

    /// Where a value the exit stub for `e` needs lives at the guard that branches
    /// there. See the aarch64 encoder's twin for why the stub asks through the
    /// guard's operand list.
    fn at_exit(&self, e: ExitId, v: VReg) -> Alloc {
        let i = self.m.exits[e.0 as usize].inst;
        let k = self
            .m
            .inst(i)
            .uses
            .iter()
            .position(|o| o.vreg == v)
            .unwrap_or_else(|| {
                panic!(
                    "exit{} reads r{} but its guard does not keep it alive",
                    e.0, v.0
                )
            });
        self.ra.use_(i, k)
    }

    /// A move the allocator asked for. Neither allocator in `regalloc` asks — but a
    /// splitting one does nothing else, and this is the path it needs.
    fn emit_move(&mut self, m: Move) {
        match m.class {
            RegClass::Int => match (m.from, m.to) {
                (Alloc::Reg(f), Alloc::Reg(t)) => self.a.mov(Gpr(t.num()), Gpr(f.num())),
                (Alloc::Reg(f), Alloc::Spill(s)) => {
                    self.a.store(RSP, s as i32 * SLOT, Gpr(f.num()))
                }
                (Alloc::Spill(s), Alloc::Reg(t)) => self.a.load(Gpr(t.num()), RSP, s as i32 * SLOT),
                (Alloc::Spill(f), Alloc::Spill(t)) => {
                    self.a.load(ACC, RSP, f as i32 * SLOT);
                    self.a.store(RSP, t as i32 * SLOT, ACC);
                }
            },
            RegClass::Float => match (m.from, m.to) {
                (Alloc::Reg(f), Alloc::Reg(t)) => self.a.movsd(Xmm(t.num()), Xmm(f.num())),
                (Alloc::Reg(f), Alloc::Spill(s)) => {
                    self.a.store_f(RSP, s as i32 * SLOT, Xmm(f.num()))
                }
                (Alloc::Spill(s), Alloc::Reg(t)) => {
                    self.a.load_f(Xmm(t.num()), RSP, s as i32 * SLOT)
                }
                (Alloc::Spill(f), Alloc::Spill(t)) => {
                    self.a.load_f(F0, RSP, f as i32 * SLOT);
                    self.a.store_f(RSP, t as i32 * SLOT, F0);
                }
            },
        }
    }

    fn edits_at(&mut self, p: ProgPoint) {
        for m in self.ra.edits_at(p).copied().collect::<Vec<_>>() {
            self.emit_move(m);
        }
    }

    // --- the floor-rounding divides -----------------------------------------
    //
    // `idiv` truncates toward zero; Lua rounds toward negative infinity. The
    // correction applies exactly when the remainder is non-zero *and* the operand
    // signs differ. Unlike aarch64's branchless `ccmp`/`csel`, this uses two short
    // forward jumps, which is fewer instructions here and just as correct.
    //
    // No zero-divisor check: the frontend has already emitted the `guard.cond`
    // that deopts before division (Lua raises on `x % 0`). The *other* hazard,
    // `i64::MIN / -1`, is handled here and cannot be skipped: unlike aarch64's
    // `sdiv`, which wraps that case to `i64::MIN`, x86 `idiv` raises `#DE` on the
    // quotient overflow. Both divides special-case a `-1` divisor to the wrapping
    // answer the interpreter's `wrapping_div`/`wrapping_rem` produce — `-n` and `0`
    // — which dodges the trap for *every* `n`, not just `i64::MIN`.
    //
    // Both operands are plain registers (never `rax`/`rdx`, which are reserved), so
    // `idiv` never aliases its inputs.

    /// `rax = floor(n / m)`, in the sign convention of Lua's `//`.
    fn floor_div(&mut self, n: Gpr, m: Gpr) {
        let done = self.a.new_label();
        let divide = self.a.new_label();

        // `m == -1`: `n // -1 == -n` (exact), computed wrapping to dodge `#DE`.
        assert!(self.a.try_cmp_imm(m, -1));
        self.a.jcc(Cond::Ne, divide);
        self.a.mov(ACC, n);
        self.a.neg(ACC);
        self.a.jmp(done);

        self.a.bind(divide);
        self.a.mov(ACC, n);
        self.a.cqo();
        self.a.idiv(m); // rax = q (trunc), rdx = r
        self.a.test(RDX, RDX);
        self.a.jcc(Cond::E, done); // r == 0: exact, no correction
        self.a.mov(RCX, n);
        self.a.xor(RCX, m); // sign bit set iff n and m disagree
        self.a.jcc(Cond::Ns, done); // same sign: truncation already floors
        assert!(self.a.try_sub_imm(ACC, 1)); // q - 1
        self.a.bind(done);
    }

    /// `rax = n - floor(n / m) * m`, in the sign convention of Lua's `%`.
    fn floor_mod(&mut self, n: Gpr, m: Gpr) {
        let done = self.a.new_label();
        let divide = self.a.new_label();

        // `m == -1`: `n % -1 == 0` (exact), so skip the trapping `idiv`.
        assert!(self.a.try_cmp_imm(m, -1));
        self.a.jcc(Cond::Ne, divide);
        self.a.mov_imm(ACC, 0);
        self.a.jmp(done);

        self.a.bind(divide);
        self.a.mov(ACC, n);
        self.a.cqo();
        self.a.idiv(m); // rdx = r (trunc, sign of n)
        self.a.mov(ACC, RDX);
        self.a.test(RDX, RDX);
        self.a.jcc(Cond::E, done);
        self.a.mov(RCX, n);
        self.a.xor(RCX, m);
        self.a.jcc(Cond::Ns, done);
        self.a.add(ACC, m); // r + m
        self.a.bind(done);
    }

    // --- blocks -------------------------------------------------------------

    fn block(&mut self, b: MBlock) {
        let label = self.blocks[b.0 as usize];
        self.a.bind(label);
        for &i in &self.m.block(b).insts.clone() {
            self.inst(i);
        }
    }

    fn inst(&mut self, i: Inst) {
        self.edits_at(ProgPoint::before(i));

        match self.m.inst(i).op {
            MOp::EntryArg(n) => {
                let d = self.def_g(i, 0, ACC);
                self.a.mov(d, ARG[n as usize]);
                self.def_done(i, 0, d);
            }
            MOp::Imm(v) => {
                let d = self.def_g(i, 0, ACC);
                self.a.mov_imm(d, v);
                self.def_done(i, 0, d);
            }
            MOp::ShapeAddr(s) => {
                let d = self.def_g(i, 0, ACC);
                self.a.mov_imm(d, self.shape_word(s));
                self.def_done(i, 0, d);
            }
            MOp::ConstPayload(c) => {
                let d = self.def_g(i, 0, ACC);
                let bits = self.pool.value(c).raw_payload() as i64;
                self.a.mov_imm(d, bits);
                self.def_done(i, 0, d);
            }
            MOp::Mov => match self.m.class(self.m.inst(i).def_vreg(0)) {
                RegClass::Int => {
                    let s = self.use_g(i, 0, S0);
                    let d = self.def_g(i, 0, ACC);
                    self.a.mov(d, s);
                    self.def_done(i, 0, d);
                }
                RegClass::Float => {
                    let s = self.use_f(i, 0, F0);
                    let d = self.def_f(i, 0, FACC);
                    self.a.movsd(d, s);
                    self.def_done_f(i, 0, d);
                }
            },
            MOp::Load { off, width } => {
                let base = self.use_g(i, 0, S0);
                let d = self.def_g(i, 0, ACC);
                match width {
                    Width::U64 => self.a.load(d, base, off),
                    Width::U8 => self.a.load8(d, base, off),
                }
                self.def_done(i, 0, d);
            }
            MOp::Store { off, width } => {
                let base = self.use_g(i, 0, S0);
                let val = self.use_g(i, 1, S1);
                match width {
                    Width::U64 => self.a.store(base, off, val),
                    Width::U8 => self.a.store8(base, off, val),
                }
            }
            MOp::Alu(o) => self.alu(i, o, None),
            MOp::AluImm(o, imm) => self.alu(i, o, Some(imm)),
            MOp::FAlu(o) => {
                match o {
                    FAluOp::Neg => {
                        let n = self.use_f(i, 0, F0);
                        self.a.movsd(FACC, n);
                        // Flip the sign bit with a mask in a scratch xmm.
                        self.a.mov_imm(ACC, i64::MIN);
                        self.a.movq_to_xmm(F1, ACC);
                        self.a.xorpd(FACC, F1);
                    }
                    _ => {
                        let n = self.use_f(i, 0, F0);
                        let m = self.use_f(i, 1, F1);
                        self.a.movsd(FACC, n);
                        match o {
                            FAluOp::Add => self.a.addsd(FACC, m),
                            FAluOp::Sub => self.a.subsd(FACC, m),
                            FAluOp::Mul => self.a.mulsd(FACC, m),
                            FAluOp::Div => self.a.divsd(FACC, m),
                            FAluOp::Neg => unreachable!("handled above"),
                        }
                    }
                }
                self.commit_f(i, 0);
            }
            MOp::ICmpSet(cc) => {
                let n = self.use_g(i, 0, S0);
                let m = self.use_g(i, 1, S1);
                self.a.cmp(n, m);
                self.a.setcc(int_cond(cc), ACC);
                self.a.movzx8(ACC, ACC);
                self.commit_g(i, 0);
            }
            MOp::FCmpSet(cc) => {
                let n = self.use_f(i, 0, F0);
                let m = self.use_f(i, 1, F1);
                self.fcmp_set(cc, n, m);
                self.commit_g(i, 0);
            }
            MOp::BitsToFloat => {
                let s = self.use_g(i, 0, S0);
                let d = self.def_f(i, 0, FACC);
                self.a.movq_to_xmm(d, s);
                self.def_done_f(i, 0, d);
            }
            MOp::FloatToBits => {
                let s = self.use_f(i, 0, F0);
                let d = self.def_g(i, 0, ACC);
                self.a.movq_to_gpr(d, s);
                self.def_done(i, 0, d);
            }
            MOp::SiToFp => {
                let s = self.use_g(i, 0, S0);
                let d = self.def_f(i, 0, FACC);
                self.a.cvtsi2sd(d, s);
                self.def_done_f(i, 0, d);
            }

            // A guard jumps to its stub when the condition it asserts is *false*.
            MOp::GuardCmp { cc, exit } => {
                let n = self.use_g(i, 0, S0);
                let m = self.use_g(i, 1, S1);
                self.a.cmp(n, m);
                let target = self.exits[exit.0 as usize];
                self.a.jcc(int_cond(cc).invert(), target);
            }
            MOp::GuardCmpImm { cc, imm, exit } => {
                let n = self.use_g(i, 0, S0);
                if !self.a.try_cmp_imm(n, imm) {
                    self.a.mov_imm(S1, imm);
                    self.a.cmp(n, S1);
                }
                let target = self.exits[exit.0 as usize];
                self.a.jcc(int_cond(cc).invert(), target);
            }
            MOp::GuardNz { exit } => {
                let n = self.use_g(i, 0, S0);
                self.a.test(n, n);
                let target = self.exits[exit.0 as usize];
                self.a.jcc(Cond::E, target);
            }

            MOp::Jump(b) => {
                let target = self.blocks[b.0 as usize];
                self.a.jmp(target);
            }
            MOp::BrNz { then_, else_ } => {
                let c = self.use_g(i, 0, S0);
                self.a.test(c, c);
                let t = self.blocks[then_.0 as usize];
                let e = self.blocks[else_.0 as usize];
                self.a.jcc(Cond::Ne, t);
                self.a.jmp(e);
            }
            MOp::Ret { nret } => {
                self.a.mov_imm(ACC, Status::packed(TAG_RETURN, nret as u32));
                let ep = self.epilogue;
                self.a.jmp(ep);
            }
            MOp::ExitTo(e) => {
                let target = self.exits[e.0 as usize];
                self.a.jmp(target);
            }
        }

        debug_assert!(
            !self.m.inst(i).op.is_terminator()
                || self.ra.edits_at(ProgPoint::after(i)).next().is_none(),
            "inst {i} is a terminator; an edit after it is unreachable"
        );
        self.edits_at(ProgPoint::after(i));
    }

    /// `d = a op b` (or `a op imm`), through the accumulator. Two-address means the
    /// result is computed in `rax` and then committed, because the destination
    /// register may be one the allocator also gave to `b`.
    fn alu(&mut self, i: Inst, o: AluOp, imm: Option<i64>) {
        // Unary ops never take an immediate.
        if matches!(o, AluOp::Neg | AluOp::Not) {
            let n = self.use_g(i, 0, S0);
            self.a.mov(ACC, n);
            match o {
                AluOp::Neg => self.a.neg(ACC),
                _ => self.a.not(ACC),
            }
            return self.commit_g(i, 0);
        }

        let n = self.use_g(i, 0, S0);

        // The second operand is either use1 or a materialized immediate. Add/Sub
        // fold a small immediate straight into the accumulator; everything else
        // lands the operand in a register and shares the register path.
        if let Some(v) = imm {
            self.a.mov(ACC, n);
            let folded = match o {
                AluOp::Add => self.a.try_add_imm(ACC, v),
                AluOp::Sub => self.a.try_sub_imm(ACC, v),
                _ => false,
            };
            if folded {
                return self.commit_g(i, 0);
            }
            self.a.mov_imm(S1, v);
            self.alu_rr(o, ACC, S1, n);
            return self.commit_g(i, 0);
        }

        let m = self.use_g(i, 1, S1);
        // `floor_div`/`floor_mod` produce their result in `rax` directly and read
        // `n`/`m` themselves; the rest compute `mov acc, n; op acc, m`.
        match o {
            AluOp::IDiv => self.floor_div(n, m),
            AluOp::Mod => self.floor_mod(n, m),
            _ => {
                self.a.mov(ACC, n);
                self.alu_rr(o, ACC, m, n);
            }
        }
        self.commit_g(i, 0);
    }

    /// Apply `acc op= m`, where `acc` already holds the first operand. `n` is the
    /// original first operand, needed only by the floor divides (unreachable here,
    /// since those are handled before this is called with a register operand).
    fn alu_rr(&mut self, o: AluOp, acc: Gpr, m: Gpr, n: Gpr) {
        debug_assert_eq!(acc, ACC);
        match o {
            AluOp::Add => self.a.add(acc, m),
            AluOp::Sub => self.a.sub(acc, m),
            AluOp::Mul => self.a.imul(acc, m),
            AluOp::And => self.a.and(acc, m),
            AluOp::Or => self.a.or(acc, m),
            AluOp::Xor => self.a.xor(acc, m),
            AluOp::Shl => {
                self.a.mov(RCX, m);
                self.a.shl_cl(acc);
            }
            AluOp::Sar => {
                self.a.mov(RCX, m);
                self.a.sar_cl(acc);
            }
            AluOp::Lsr => {
                self.a.mov(RCX, m);
                self.a.shr_cl(acc);
            }
            // Reached only via the AluImm fallback; the register-operand caller
            // routes these to `floor_*` instead.
            AluOp::Mod => {
                self.floor_mod(n, m);
            }
            AluOp::IDiv => {
                self.floor_div(n, m);
            }
            AluOp::Neg | AluOp::Not => unreachable!("unary handled in `alu`"),
        }
    }

    /// `rax = (n cc m) ? 1 : 0` for a float compare, Lua's NaN rule baked in.
    ///
    /// `comisd` sets only the unsigned flags and reports unordered as `CF=ZF=PF=1`.
    /// `seta`/`setae` (CF clear) are therefore the tests that read false against a
    /// NaN, which is what `<`/`<=`/`>`/`>=` want after orienting the operands; only
    /// `==` and `~=` have to fold in the parity flag.
    fn fcmp_set(&mut self, cc: Cc, n: Xmm, m: Xmm) {
        match cc {
            Cc::Lt => {
                self.a.comisd(m, n); // m > n  ⇔  n < m
                self.a.setcc(Cond::A, ACC);
                self.a.movzx8(ACC, ACC);
            }
            Cc::Le => {
                self.a.comisd(m, n);
                self.a.setcc(Cond::Ae, ACC);
                self.a.movzx8(ACC, ACC);
            }
            Cc::Gt => {
                self.a.comisd(n, m);
                self.a.setcc(Cond::A, ACC);
                self.a.movzx8(ACC, ACC);
            }
            Cc::Ge => {
                self.a.comisd(n, m);
                self.a.setcc(Cond::Ae, ACC);
                self.a.movzx8(ACC, ACC);
            }
            Cc::Eq => {
                // Ordered and equal: ZF set *and* PF clear.
                self.a.comisd(n, m);
                self.a.setcc(Cond::E, ACC);
                self.a.movzx8(ACC, ACC);
                self.a.setcc(Cond::Np, RCX);
                self.a.movzx8(RCX, RCX);
                self.a.and(ACC, RCX);
            }
            Cc::Ne => {
                // Not-equal or unordered: ZF clear *or* PF set — the one comparison
                // a NaN satisfies.
                self.a.comisd(n, m);
                self.a.setcc(Cond::Ne, ACC);
                self.a.movzx8(ACC, ACC);
                self.a.setcc(Cond::P, RCX);
                self.a.movzx8(RCX, RCX);
                self.a.or(ACC, RCX);
            }
        }
    }

    // --- exit stubs ---------------------------------------------------------

    /// Materialize the interpreter's view of the frame, then return. Every store
    /// is a `Value`: a payload word and a tag byte, at `base + reg * 16`.
    fn exit_stub(&mut self, e: ExitId) {
        let label = self.exits[e.0 as usize];
        self.a.bind(label);

        let stub = self.m.exits[e.0 as usize].clone();

        // The base survives every value the stub loads, so it takes `rcx` — a
        // scratch the per-slot loads below (which use `S0`/`S1`/`rax`) never touch.
        let base_at = self.at_exit(e, self.m.frame_base);
        let base = self.read_g(base_at, RCX);

        for (reg, src) in stub.slots {
            let slot = reg as i32 * layout::val::SIZE as i32;
            let (payload_off, kind_off) = (
                slot + layout::val::DATA as i32,
                slot + layout::val::KIND as i32,
            );

            let (payload, tag) = match src {
                ExitSrc::Boxed { payload, tag } => {
                    let a = self.at_exit(e, payload);
                    (self.read_g(a, S0), tag)
                }
                ExitSrc::Int(v) => {
                    let a = self.at_exit(e, v);
                    (self.read_g(a, S0), Tag::Const(ValueKind::Integer))
                }
                ExitSrc::Float(v) => {
                    let a = self.at_exit(e, v);
                    let f = self.read_f(a, F0);
                    self.a.movq_to_gpr(S0, f);
                    (S0, Tag::Const(ValueKind::Float))
                }
                ExitSrc::Const { payload, tag } => {
                    self.a.mov_imm(S0, payload as i64);
                    (S0, Tag::Const(tag))
                }
            };
            self.a.store(base, payload_off, payload);

            let tag_reg = match tag {
                Tag::Const(k) => {
                    self.a.mov_imm(S1, layout::kind(k) as i64);
                    S1
                }
                Tag::Dyn(t) => {
                    let a = self.at_exit(e, t);
                    self.read_g(a, S1)
                }
            };
            self.a.store8(base, kind_off, tag_reg);
        }

        self.a.mov_imm(ACC, Status::packed(TAG_DEOPT, e.0));
        let ep = self.epilogue;
        self.a.jmp(ep);
    }

    /// The word a `Shape` field actually holds: the collector's box address. Sound
    /// to bake in only because the collector never moves an object and the pool
    /// roots the shape for as long as the code that names it can run.
    fn shape_word(&self, s: crate::jit::ir::pool::ShapeRef) -> i64 {
        crate::dmm::Gc::box_addr(self.pool.shape(s).inner()) as i64
    }
}

/// The integer condition for an IR compare. Signed, since Lua integers are.
fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::E,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::L,
        Cc::Le => Cond::Le,
        Cc::Gt => Cond::G,
        Cc::Ge => Cond::Ge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::value::Value;
    use crate::jit::backend::code::Code;
    use crate::jit::backend::mach::{AluOp, MInst, Width};
    use crate::jit::backend::regalloc::{Alloc, AllocationBuilder, VReg};

    type Region = extern "C" fn(*mut (), *mut Value<'static>) -> u64;

    fn x(n: u8) -> PReg {
        PReg::new(RegClass::Int, n)
    }

    /// `base[0] = 41 + 1`, returned as one result — the smallest region that
    /// computes something and leaves it where the interpreter looks. The `junk`
    /// instruction overwrites whatever register the sum was computed into, so an
    /// allocation that moves the sum and an encoder that drops the move produce a
    /// *wrong answer* rather than the right one by luck.
    fn add_and_store() -> (MFunc, VReg, Inst, Inst) {
        let mut m = MFunc::new();
        let b = m.new_block();
        m.entry = b;

        let base = m.new_vreg(RegClass::Int);
        m.frame_base = base;
        let (lhs, rhs, sum, junk, tag) = (
            m.new_vreg(RegClass::Int),
            m.new_vreg(RegClass::Int),
            m.new_vreg(RegClass::Int),
            m.new_vreg(RegClass::Int),
            m.new_vreg(RegClass::Int),
        );

        m.push(b, MInst::new(MOp::EntryArg(1), vec![base], vec![]));
        m.push(b, MInst::new(MOp::Imm(41), vec![lhs], vec![]));
        m.push(b, MInst::new(MOp::Imm(1), vec![rhs], vec![]));
        let def_sum = m.push(
            b,
            MInst::new(MOp::Alu(AluOp::Add), vec![sum], vec![lhs, rhs]),
        );
        m.push(b, MInst::new(MOp::Imm(0x7fff_dead), vec![junk], vec![]));
        let store = m.push(
            b,
            MInst::new(
                MOp::Store {
                    off: layout::val::DATA as i32,
                    width: Width::U64,
                },
                vec![],
                vec![base, sum],
            ),
        );
        m.push(
            b,
            MInst::new(
                MOp::Imm(layout::kind(ValueKind::Integer) as i64),
                vec![tag],
                vec![],
            ),
        );
        m.push(
            b,
            MInst::new(
                MOp::Store {
                    off: layout::val::KIND as i32,
                    width: Width::U8,
                },
                vec![],
                vec![base, tag],
            ),
        );
        m.push(b, MInst::new(MOp::Ret { nret: 1 }, vec![], vec![]));

        (m, sum, def_sum, store)
    }

    /// Drive the edit path by hand: compute the sum in r8, move it out, and let the
    /// very next instruction (`junk`) take r8 over. Drop the move and the region
    /// returns `0x7fffdead`. Neither allocator in `regalloc` emits edits, so this
    /// is the only coverage the encoder's `emit_move` gets.
    fn split_sum_through(dest: Alloc, num_spills: u32) -> i64 {
        let (m, sum, def_sum, store) = add_and_store();

        let mut b = AllocationBuilder::new(&m);
        b.assign(m.frame_base, Alloc::Reg(x(12)));
        b.assign(VReg(1), Alloc::Reg(x(9))); // lhs
        b.assign(VReg(2), Alloc::Reg(x(13))); // rhs
        b.assign(VReg(5), Alloc::Reg(x(14))); // tag

        b.assign(sum, Alloc::Reg(x(8)));
        b.assign(VReg(4), Alloc::Reg(x(8))); // junk, which takes r8 over
        b.set_use(store, 1, dest);
        b.edit(
            ProgPoint::after(def_sum),
            Move {
                from: Alloc::Reg(x(8)),
                to: dest,
                class: RegClass::Int,
            },
        );

        let ra = b.finish(num_spills);
        let words = encode(&m, &ConstPool::new(), &ra).expect("encode");
        let code = Code::from_words(&words).expect("map code");

        let mut stack = vec![Value::nil(); 1];
        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

        assert_eq!(status, Status::Return(1));
        stack[0].get_integer().expect("an integer result")
    }

    #[test]
    fn a_split_into_a_register_emits_its_move() {
        assert_eq!(split_sum_through(Alloc::Reg(x(15)), 0), 42);
    }

    #[test]
    fn a_split_onto_the_stack_emits_its_move() {
        assert_eq!(split_sum_through(Alloc::Spill(0), 1), 42);
    }
}
