//! Machine IR to aarch64.
//!
//! # Shape of the output
//!
//! ```text
//!   prologue
//!   entry block, then the rest in index order   <- the fast path
//!   ...
//!   exit stubs                                   <- cold; one per IR Exit
//!   epilogue
//! ```
//!
//! Every guard falls through on success and branches to its stub on failure, so
//! the fast path is a straight line and the stubs sit out of the way of the
//! instruction prefetcher.
//!
//! # Exit stubs are the location map
//!
//! There is no side table saying "at exit 3, Lua register 2 lives in x11". The
//! stub *is* that map, compiled: a run of stores that puts each live value where
//! the interpreter expects it, tagging the unboxed ones on the way. It is
//! generated after allocation, when where-each-value-landed is known, and so it
//! costs nothing on the fast path and nothing in memory beyond the cold code
//! itself.
//!
//! # Return convention
//!
//! `x0` carries a packed status word back to the executor: see [`Status`].
//! Compiled code never returns values in registers — the Lua stack is where the
//! interpreter looks, so `Ret` has already stored them there.

use crate::env::value::ValueKind;
use crate::jit::backend::aarch64_asm::{Asm, Cond, FP, Fpr, Gpr, LR, Label, SP};
use crate::jit::backend::layout;
use crate::jit::backend::mach::{
    AluOp, ExitId, ExitSrc, FAluOp, MBlock, MFunc, MOp, RegClass, Tag, VReg, Width,
};
use crate::jit::backend::regalloc::{
    Alloc, Allocation, Inst, MachineEnv, Move, PReg, ProgPoint, RegallocFunc,
};
use crate::jit::ir::op::Cc;
use crate::jit::ir::pool::ConstPool;

/// What compiled code reports back, packed into `x0` as `(tag << 32) | payload`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// A guard failed or an uncompiled path was reached. The payload is the exit
    /// id; the frame has already been written back and the interpreter can resume
    /// at that exit's pc.
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

    /// Decode what compiled code left in `x0`.
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
    /// More spill slots than the frame's immediate offsets can address. Only
    /// reachable with an absurd number of live values; the fix is a second frame
    /// pointer, not a bigger immediate.
    FrameTooLarge(u32),
}

/// Scratch registers.
///
/// Caller-saved and never allocated to a virtual register, so an instruction may
/// clobber them freely between one value's load and the next. Compiled code makes
/// no calls, so nothing else can clobber them either.
///
/// Three integer scratches is the most any one instruction needs: two operands
/// and a destination.
const S0: Gpr = Gpr(9);
const S1: Gpr = Gpr(10);
const S2: Gpr = Gpr(11);

/// Two more, for the macro-ops that expand to a sequence and need somewhere to
/// keep an intermediate. Held apart from `S0..S2` because those may already be
/// standing in for a spilled operand by the time the sequence starts. `x16`/`x17`
/// are the platform's own scratch (IP0/IP1).
const T0: Gpr = Gpr(16);
const T1: Gpr = Gpr(17);
const F0: Fpr = Fpr(16);
const F1: Fpr = Fpr(17);
const F2: Fpr = Fpr(18);

/// Where an edit's move goes through when it is stack-to-stack. Free between
/// instructions by construction: an edit sits *between* two instructions, and
/// every scratch is dead at that point.
const E0: Gpr = Gpr(16);
const E1: Fpr = Fpr(16);

/// The two incoming arguments: the thread, and the Lua frame base.
const ARG: [Gpr; 2] = [Gpr(0), Gpr(1)];

/// One spill slot is one machine word.
const SLOT: i32 = 8;

/// The registers the allocator may hand out.
///
/// Caller-saved only. Compiled regions are leaves — they call nothing — so a
/// caller-saved register costs nothing to use, while a callee-saved one would have
/// to be spilled and restored in the prologue for no gain at the register counts
/// these regions actually reach. If pressure ever justifies it, x19–x28 and d8–d15
/// are sitting there.
///
/// What is *missing* from this list is the load-bearing part, and it is missing for
/// reasons that live in this file — which is why the list does too, rather than in
/// the allocator:
///
///   x0, x1   incoming arguments, and x0 is the status word on the way out
///   x9-x11   `S0`–`S2`, the encoder's integer scratch
///   x16, x17 `T0`/`T1`, and the linker may insert a veneer that clobbers them
///   x18      reserved by the platform on Darwin
///   x29, x30 frame pointer and link register
///   d16-d18  `F0`–`F2`, the encoder's float scratch
pub fn machine_env() -> MachineEnv {
    const INT: &[u8] = &[2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15];
    const FLOAT: &[u8] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

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
    frame: u32,
}

/// Encode `m` to a flat little-endian instruction stream. Placing those words in
/// executable memory is the allocator's job (or, in tests, [`Code::from_words`]);
/// the encoder does not know the final size until it is done, so it builds into a
/// plain `Vec` first.
pub fn encode(m: &MFunc, pool: &ConstPool<'_>, ra: &Allocation) -> Result<Vec<u32>, EncodeError> {
    debug_assert!(
        crate::jit::backend::regalloc::verify(m, ra).is_ok(),
        "{}",
        crate::jit::backend::regalloc::verify(m, ra).unwrap_err()
    );

    let mut a = Asm::new();
    let blocks = (0..m.blocks.len()).map(|_| a.new_label()).collect();
    let exits = (0..m.exits.len()).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();

    let frame = (ra.num_spills * SLOT as u32).next_multiple_of(16);
    // `try_sub_imm` reaches 4095, or a multiple of 4096 up to 2^24. The scaled
    // load offsets that address the slots run out first, at 4095 words.
    if ra.num_spills >= 4096 {
        return Err(EncodeError::FrameTooLarge(ra.num_spills));
    }

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

        // Reverse postorder, which puts the entry first so it falls through from
        // the prologue. The order is not merely cosmetic: the register allocator
        // computed its live intervals as spans of *this* linearization, so laying
        // the blocks out differently would free registers that still hold live
        // values here.
        for b in self.m.block_order() {
            self.block(b);
        }

        for e in 0..self.m.exits.len() as u32 {
            self.exit_stub(ExitId(e));
        }

        let epilogue = self.epilogue;
        self.a.bind(epilogue);
        self.a.mov_sp(SP, FP);
        self.a.ldp_post(FP, LR, SP, 16);
        self.a.ret();
    }

    fn prologue(&mut self) {
        self.a.stp_pre(FP, LR, SP, -16);
        self.a.mov_sp(FP, SP);
        if self.frame > 0 {
            assert!(
                self.a.try_sub_imm(SP, SP, self.frame as i64),
                "frame size checked at entry"
            );
        }
    }

    // --- operand access -----------------------------------------------------
    //
    // Everything below asks the allocator where *this operand of this instruction*
    // lives, never where a value lives. The distinction is invisible under the
    // allocators in `regalloc` — neither splits a live range, so every mention of a
    // value answers the same — and it is the whole difference under one that does.
    //
    // When an operand is in a register these are free; when it is spilled they cost
    // the load or store, at the scratch register the caller nominates.

    fn read_g(&mut self, a: Alloc, scratch: Gpr) -> Gpr {
        match a {
            Alloc::Reg(r) => Gpr(r.num()),
            Alloc::Spill(s) => {
                assert!(self.a.try_ldr(scratch, SP, s as i32 * SLOT));
                scratch
            }
        }
    }

    fn read_f(&mut self, a: Alloc, scratch: Fpr) -> Fpr {
        match a {
            Alloc::Reg(r) => Fpr(r.num()),
            Alloc::Spill(s) => {
                assert!(self.a.try_ldr_f(scratch, SP, s as i32 * SLOT));
                scratch
            }
        }
    }

    /// The register holding use `k` of instruction `i`.
    fn use_g(&mut self, i: Inst, k: usize, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Int);
        let a = self.ra.use_(i, k);
        self.read_g(a, scratch)
    }

    fn use_f(&mut self, i: Inst, k: usize, scratch: Fpr) -> Fpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Float);
        let a = self.ra.use_(i, k);
        self.read_f(a, scratch)
    }

    /// Where to write def `k` of instruction `i`. Pair every call with
    /// [`Self::def_done`], which is a no-op for a value in a register and the spill
    /// store otherwise.
    fn def_g(&mut self, i: Inst, k: usize, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Int);
        match self.ra.def(i, k) {
            Alloc::Reg(r) => Gpr(r.num()),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_f(&mut self, i: Inst, k: usize, scratch: Fpr) -> Fpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Float);
        match self.ra.def(i, k) {
            Alloc::Reg(r) => Fpr(r.num()),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_done(&mut self, i: Inst, k: usize, from: Gpr) {
        if let Alloc::Spill(s) = self.ra.def(i, k) {
            assert!(self.a.try_str(from, SP, s as i32 * SLOT));
        }
    }

    fn def_done_f(&mut self, i: Inst, k: usize, from: Fpr) {
        if let Alloc::Spill(s) = self.ra.def(i, k) {
            assert!(self.a.try_str_f(from, SP, s as i32 * SLOT));
        }
    }

    /// Where a value the exit stub for `e` needs lives at the guard that branches
    /// there.
    ///
    /// A stub is not an instruction and has no operands, so the allocator has
    /// nothing to say about it directly. It answers through the guard: isel lists
    /// every value the stub reads among the guard's uses — which is what kept them
    /// alive in the first place — so the stub's question is that guard's operand.
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

    /// A move the allocator asked for. Neither allocator in `regalloc` asks — they
    /// assign whole values, so a value never has to move — but a splitting one does
    /// nothing else, and this is what it would need the encoder to be able to do.
    fn emit_move(&mut self, m: Move) {
        match m.class {
            RegClass::Int => match (m.from, m.to) {
                (Alloc::Reg(f), Alloc::Reg(t)) => self.a.mov(Gpr(t.num()), Gpr(f.num())),
                (Alloc::Reg(f), Alloc::Spill(s)) => {
                    assert!(self.a.try_str(Gpr(f.num()), SP, s as i32 * SLOT));
                }
                (Alloc::Spill(s), Alloc::Reg(t)) => {
                    assert!(self.a.try_ldr(Gpr(t.num()), SP, s as i32 * SLOT));
                }
                (Alloc::Spill(f), Alloc::Spill(t)) => {
                    assert!(self.a.try_ldr(E0, SP, f as i32 * SLOT));
                    assert!(self.a.try_str(E0, SP, t as i32 * SLOT));
                }
            },
            RegClass::Float => match (m.from, m.to) {
                (Alloc::Reg(f), Alloc::Reg(t)) => self.a.fmov(Fpr(t.num()), Fpr(f.num())),
                (Alloc::Reg(f), Alloc::Spill(s)) => {
                    assert!(self.a.try_str_f(Fpr(f.num()), SP, s as i32 * SLOT));
                }
                (Alloc::Spill(s), Alloc::Reg(t)) => {
                    assert!(self.a.try_ldr_f(Fpr(t.num()), SP, s as i32 * SLOT));
                }
                (Alloc::Spill(f), Alloc::Spill(t)) => {
                    assert!(self.a.try_ldr_f(E1, SP, f as i32 * SLOT));
                    assert!(self.a.try_str_f(E1, SP, t as i32 * SLOT));
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
    // `sdiv` truncates toward zero; Lua rounds toward negative infinity. The
    // correction applies exactly when the remainder is non-zero *and* the operand
    // signs differ, which is two conditions — hence `ccmp`, which evaluates the
    // second only if the first held and otherwise stamps the flags with a value
    // the `csel` reads as false. That keeps the whole thing branchless.
    //
    // Nothing here needs a zero-divisor check: `sdiv` by zero yields zero rather
    // than trapping, and the frontend has already emitted the `guard.cond` that
    // deopts before we get here (Lua raises on `x % 0`). Both of these also stay
    // correct at `i64::MIN op -1`, where `sdiv` wraps to `i64::MIN` and the
    // remainder is 0 — the same answers `wrapping_div`/`wrapping_rem` give the
    // interpreter.

    /// `d = n - floor(n / m) * m`, in the sign convention of Lua's `%`.
    fn floor_mod(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.a.sdiv(T0, n, m); // q
        self.a.msub(T0, T0, m, n); // r = n - q*m, truncated
        self.a.eor(T1, T0, m); // sign bit set iff r and m disagree
        assert!(self.a.try_cmp_imm(T0, 0)); // Z = (r == 0)
        self.a.ccmp_imm(T1, 0, 0, Cond::Ne); // r != 0 ? flags of (r^m) : clear
        self.a.add(T1, T0, m); // the corrected value — `add` leaves flags alone
        self.a.csel(d, T1, T0, Cond::Mi);
    }

    /// `d = floor(n / m)`, in the sign convention of Lua's `//`.
    fn floor_div(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.a.sdiv(T0, n, m); // q
        self.a.msub(T1, T0, m, n); // r
        assert!(self.a.try_cmp_imm(T1, 0)); // Z = (r == 0)
        self.a.eor(T1, n, m); // r is dead; reuse it for the sign test
        self.a.ccmp_imm(T1, 0, 0, Cond::Ne);
        assert!(self.a.try_sub_imm(T1, T0, 1)); // q - 1
        self.a.csel(d, T1, T0, Cond::Mi);
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
                // The incoming argument registers are still live here: the entry
                // block runs before anything can clobber them, and the scratch
                // registers deliberately do not overlap `x0`/`x1`.
                let d = self.def_g(i, 0, S0);
                self.a.mov(d, ARG[n as usize]);
                self.def_done(i, 0, d);
            }
            MOp::Imm(v) => {
                let d = self.def_g(i, 0, S0);
                self.a.mov_imm(d, v);
                self.def_done(i, 0, d);
            }
            MOp::ShapeAddr(s) => {
                let d = self.def_g(i, 0, S0);
                self.a.mov_imm(d, self.shape_word(s));
                self.def_done(i, 0, d);
            }
            MOp::ConstPayload(c) => {
                let d = self.def_g(i, 0, S0);
                let bits = self.pool.value(c).raw_payload() as i64;
                self.a.mov_imm(d, bits);
                self.def_done(i, 0, d);
            }
            MOp::Mov => match self.m.class(self.m.inst(i).def_vreg(0)) {
                RegClass::Int => {
                    let s = self.use_g(i, 0, S0);
                    let d = self.def_g(i, 0, S1);
                    self.a.mov(d, s);
                    self.def_done(i, 0, d);
                }
                RegClass::Float => {
                    let s = self.use_f(i, 0, F0);
                    let d = self.def_f(i, 0, F1);
                    self.a.fmov(d, s);
                    self.def_done_f(i, 0, d);
                }
            },
            MOp::Load { off, width } => {
                let base = self.use_g(i, 0, S0);
                let d = self.def_g(i, 0, S1);
                let ok = match width {
                    Width::U64 => self.a.try_ldr(d, base, off),
                    Width::U8 => self.a.try_ldrb(d, base, off),
                };
                assert!(ok, "load offset {off} out of range");
                self.def_done(i, 0, d);
            }
            MOp::Store { off, width } => {
                let base = self.use_g(i, 0, S0);
                let val = self.use_g(i, 1, S1);
                let ok = match width {
                    Width::U64 => self.a.try_str(val, base, off),
                    Width::U8 => self.a.try_strb(val, base, off),
                };
                assert!(ok, "store offset {off} out of range");
            }
            MOp::Alu(o) => {
                let d = match o {
                    AluOp::Neg | AluOp::Not => {
                        let n = self.use_g(i, 0, S0);
                        let d = self.def_g(i, 0, S2);
                        match o {
                            AluOp::Neg => self.a.neg(d, n),
                            _ => self.a.mvn(d, n),
                        }
                        d
                    }
                    _ => {
                        let n = self.use_g(i, 0, S0);
                        let m = self.use_g(i, 1, S1);
                        let d = self.def_g(i, 0, S2);
                        match o {
                            AluOp::Add => self.a.add(d, n, m),
                            AluOp::Sub => self.a.sub(d, n, m),
                            AluOp::Mul => self.a.mul(d, n, m),
                            AluOp::And => self.a.and(d, n, m),
                            AluOp::Or => self.a.orr(d, n, m),
                            AluOp::Xor => self.a.eor(d, n, m),
                            AluOp::Shl => self.a.lslv(d, n, m),
                            AluOp::Sar => self.a.asrv(d, n, m),
                            AluOp::Lsr => self.a.lsrv(d, n, m),
                            AluOp::Mod => self.floor_mod(d, n, m),
                            AluOp::IDiv => self.floor_div(d, n, m),
                            AluOp::Neg | AluOp::Not => unreachable!("handled above"),
                        }
                        d
                    }
                };
                self.def_done(i, 0, d);
            }
            MOp::AluImm(o, imm) => {
                let n = self.use_g(i, 0, S0);
                let d = self.def_g(i, 0, S2);
                let folded = match o {
                    AluOp::Add => self.a.try_add_imm(d, n, imm),
                    AluOp::Sub => self.a.try_sub_imm(d, n, imm),
                    _ => false,
                };
                if !folded {
                    self.a.mov_imm(S1, imm);
                    match o {
                        AluOp::Add => self.a.add(d, n, S1),
                        AluOp::Sub => self.a.sub(d, n, S1),
                        AluOp::Mul => self.a.mul(d, n, S1),
                        AluOp::And => self.a.and(d, n, S1),
                        AluOp::Or => self.a.orr(d, n, S1),
                        AluOp::Xor => self.a.eor(d, n, S1),
                        AluOp::Shl => self.a.lslv(d, n, S1),
                        AluOp::Sar => self.a.asrv(d, n, S1),
                        AluOp::Lsr => self.a.lsrv(d, n, S1),
                        AluOp::Mod => self.floor_mod(d, n, S1),
                        AluOp::IDiv => self.floor_div(d, n, S1),
                        AluOp::Neg | AluOp::Not => panic!("{o:?} takes no immediate"),
                    }
                }
                self.def_done(i, 0, d);
            }
            MOp::FAlu(o) => {
                let d = match o {
                    FAluOp::Neg => {
                        let n = self.use_f(i, 0, F0);
                        let d = self.def_f(i, 0, F2);
                        self.a.fneg(d, n);
                        d
                    }
                    _ => {
                        let n = self.use_f(i, 0, F0);
                        let m = self.use_f(i, 1, F1);
                        let d = self.def_f(i, 0, F2);
                        match o {
                            FAluOp::Add => self.a.fadd(d, n, m),
                            FAluOp::Sub => self.a.fsub(d, n, m),
                            FAluOp::Mul => self.a.fmul(d, n, m),
                            FAluOp::Div => self.a.fdiv(d, n, m),
                            FAluOp::Neg => unreachable!("handled above"),
                        }
                        d
                    }
                };
                self.def_done_f(i, 0, d);
            }
            MOp::ICmpSet(cc) => {
                let n = self.use_g(i, 0, S0);
                let m = self.use_g(i, 1, S1);
                let d = self.def_g(i, 0, S2);
                self.a.cmp(n, m);
                self.a.cset(d, int_cond(cc));
                self.def_done(i, 0, d);
            }
            MOp::FCmpSet(cc) => {
                let n = self.use_f(i, 0, F0);
                let m = self.use_f(i, 1, F1);
                let d = self.def_g(i, 0, S2);
                self.a.fcmp(n, m);
                self.a.cset(d, float_cond(cc));
                self.def_done(i, 0, d);
            }
            MOp::BitsToFloat => {
                let s = self.use_g(i, 0, S0);
                let d = self.def_f(i, 0, F0);
                self.a.fmov_to_fpr(d, s);
                self.def_done_f(i, 0, d);
            }
            MOp::FloatToBits => {
                let s = self.use_f(i, 0, F0);
                let d = self.def_g(i, 0, S0);
                self.a.fmov_to_gpr(d, s);
                self.def_done(i, 0, d);
            }
            MOp::SiToFp => {
                let s = self.use_g(i, 0, S0);
                let d = self.def_f(i, 0, F0);
                self.a.scvtf(d, s);
                self.def_done_f(i, 0, d);
            }

            // A guard branches to its stub when the condition it asserts is
            // *false*, and falls through otherwise. Uses past the operands are the
            // frame-state keepalives; they exist to hold registers open for the
            // stub and produce no code here.
            MOp::GuardCmp { cc, exit } => {
                let n = self.use_g(i, 0, S0);
                let m = self.use_g(i, 1, S1);
                self.a.cmp(n, m);
                let target = self.exits[exit.0 as usize];
                self.a.b_cond(int_cond(cc).invert(), target);
            }
            MOp::GuardCmpImm { cc, imm, exit } => {
                let n = self.use_g(i, 0, S0);
                if !self.a.try_cmp_imm(n, imm) {
                    self.a.mov_imm(S1, imm);
                    self.a.cmp(n, S1);
                }
                let target = self.exits[exit.0 as usize];
                self.a.b_cond(int_cond(cc).invert(), target);
            }
            MOp::GuardNz { exit } => {
                let n = self.use_g(i, 0, S0);
                let target = self.exits[exit.0 as usize];
                self.a.cbz(n, target);
            }

            MOp::Jump(b) => {
                let target = self.blocks[b.0 as usize];
                self.a.b(target);
            }
            MOp::BrNz { then_, else_ } => {
                let c = self.use_g(i, 0, S0);
                let t = self.blocks[then_.0 as usize];
                let e = self.blocks[else_.0 as usize];
                self.a.cbnz(c, t);
                self.a.b(e);
            }
            MOp::Ret { nret } => {
                self.a
                    .mov_imm(ARG[0], Status::packed(TAG_RETURN, nret as u32));
                let ep = self.epilogue;
                self.a.b(ep);
            }
            MOp::ExitTo(e) => {
                let target = self.exits[e.0 as usize];
                self.a.b(target);
            }
        }

        // An edit after a terminator would land past the branch, where nothing
        // executes it. A splitting allocator that wants to move a value *on an
        // edge* has to put the move in the edge's own block — which the machine IR
        // always has, since isel splits critical edges to hold the parameter copies.
        debug_assert!(
            !self.m.inst(i).op.is_terminator()
                || self.ra.edits_at(ProgPoint::after(i)).next().is_none(),
            "inst {i} is a terminator; an edit after it is unreachable"
        );
        self.edits_at(ProgPoint::after(i));
    }

    // --- exit stubs ---------------------------------------------------------

    /// Materialize the interpreter's view of the frame, then return.
    ///
    /// Every store here is a `Value`: a payload word and a tag byte, at
    /// `base + reg * 16`. The tag is an immediate wherever the type was known
    /// statically, which after specialization is nearly always.
    fn exit_stub(&mut self, e: ExitId) {
        let label = self.exits[e.0 as usize];
        self.a.bind(label);

        let stub = self.m.exits[e.0 as usize].clone();

        // The base has to survive every value the stub loads, so it takes the one
        // scratch register the loads below never touch.
        let base_at = self.at_exit(e, self.m.frame_base);
        let base = self.read_g(base_at, S2);

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
                    self.a.fmov_to_gpr(S0, f);
                    (S0, Tag::Const(ValueKind::Float))
                }
                ExitSrc::Const { payload, tag } => {
                    self.a.mov_imm(S0, payload as i64);
                    (S0, Tag::Const(tag))
                }
            };
            assert!(self.a.try_str(payload, base, payload_off));

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
            assert!(self.a.try_strb(tag_reg, base, kind_off));
        }

        self.a.mov_imm(ARG[0], Status::packed(TAG_DEOPT, e.0));
        let ep = self.epilogue;
        self.a.b(ep);
    }

    /// The word a `Shape` field actually holds: the collector's box address.
    ///
    /// Baking a heap address into code is sound only because this collector never
    /// moves an object, and the constant pool roots the shape for as long as the
    /// code that names it can run.
    fn shape_word(&self, s: crate::jit::ir::pool::ShapeRef) -> i64 {
        crate::dmm::Gc::box_addr(self.pool.shape(s).inner()) as i64
    }
}

fn int_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Lt,
        Cc::Le => Cond::Le,
        Cc::Gt => Cond::Gt,
        Cc::Ge => Cond::Ge,
    }
}

/// The float mapping is not the integer one, and the difference is NaN.
///
/// An unordered compare sets `V`, so the signed `lt` (`N != V`) would report true
/// for `NaN < 1`. `mi` (`N == 1`) and `ls` (`!C || Z`) are the forms that stay
/// false when either operand is NaN — which is Lua's rule, where every comparison
/// but `~=` is false against a NaN.
fn float_cond(cc: Cc) -> Cond {
    match cc {
        Cc::Eq => Cond::Eq,
        Cc::Ne => Cond::Ne,
        Cc::Lt => Cond::Mi,
        Cc::Le => Cond::Ls,
        Cc::Gt => Cond::Gt,
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

    /// `base[0] = 41 + 1`, returned as one result. The smallest region that
    /// computes something and leaves it where the interpreter looks for it.
    ///
    /// The `junk` instruction exists to overwrite whatever register the sum was
    /// computed into, so that an allocation which moves the sum elsewhere and an
    /// encoder which fails to emit that move produce a *wrong answer* rather than
    /// the right one by luck.
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

    /// The encoder has to be able to insert a move the allocator asked for — and
    /// nothing in `regalloc` asks. Neither allocator splits a live range, so the
    /// edit list is always empty and the code path would otherwise never run.
    ///
    /// It is the path a splitting allocator (regalloc3) would live or die on, so it
    /// is driven by hand: an allocation that computes the sum in x2, moves it out,
    /// and lets the very next instruction take x2 over. Drop the move and the
    /// region returns `0x7fffdead`.
    fn split_sum_through(dest: Alloc, num_spills: u32) -> i64 {
        let (m, sum, def_sum, store) = add_and_store();

        let mut b = AllocationBuilder::new(&m);
        b.assign(m.frame_base, Alloc::Reg(x(3)));
        b.assign(VReg(1), Alloc::Reg(x(4))); // lhs
        b.assign(VReg(2), Alloc::Reg(x(6))); // rhs
        b.assign(VReg(5), Alloc::Reg(x(7))); // tag

        b.assign(sum, Alloc::Reg(x(2)));
        b.assign(VReg(4), Alloc::Reg(x(2))); // junk, which takes x2 over
        b.set_use(store, 1, dest);
        b.edit(
            ProgPoint::after(def_sum),
            Move {
                from: Alloc::Reg(x(2)),
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
        assert_eq!(split_sum_through(Alloc::Reg(x(5)), 0), 42);
    }

    #[test]
    fn a_split_onto_the_stack_emits_its_move() {
        assert_eq!(split_sum_through(Alloc::Spill(0), 1), 42);
    }
}
