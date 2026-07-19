//! Machine IR to x86-64.
//!
//! The structure mirrors the aarch64 encoder deliberately — same block layout,
//! same exit-stub-as-location-map idea, same return convention — so the two read
//! side by side. What differs is forced by the ISA, and only that:
//!
//!   - **Two-address arithmetic.** `add d, s` is `d = d + s`; the machine IR is
//!     three-address. `annotate` marks the def `Reuse(0)`, so the allocator puts the
//!     result in its first source's register (copying it in only when that source is
//!     still live) and the op writes the destination in place — no accumulator.
//!   - **Fixed-register instructions.** `idiv` divides `rdx:rax`. Its result is a
//!     `Fixed(rax)` def and it clobbers `rdx`/`rcx`; otherwise those are ordinary
//!     allocatable registers. `rax` doubles as the return word, but that write is
//!     terminal, so it costs the allocator nothing (as `x0` does on aarch64).
//!   - **Fewer free registers.** aarch64 allocates only caller-saved registers
//!     because it has enough; x86-64 does not, so the pool reaches into the
//!     callee-saved `rbx`/`rbp`/`r12`–`r15` — and the prologue saves only the ones a
//!     region actually used, so a small region pays nothing. `rbp` is an ordinary
//!     register, not a frame pointer: spills are addressed through `rsp`.
//!   - **Float compares set only the unsigned flags.** `comisd` reports an
//!     unordered result as `CF=ZF=PF=1`, so the Lua-correct condition for `<` is
//!     `seta` after swapping the operands, and `==`/`~=` need the parity flag
//!     folded in.
//!
//! # Shape of the output
//!
//! ```text
//!   prologue                                     <- save used callee-saved, open frame
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
    Alloc, Allocation, Constraint, Edit, Inst, MachineEnv, Move, Operand, PReg, ProgPoint,
    RegallocFunc,
};
use crate::jit::backend::x64_asm::{Asm, Cond, Gpr, Label, RAX, RCX, RDX, RSP, Xmm};
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

// --- registers --------------------------------------------------------------

/// `rax`. The status word leaves the region here (the return ABI), and `idiv`
/// leaves its quotient here — but only the terminal status write is special, so
/// `rax` is otherwise an ordinary allocatable register (`idiv` claims it with a
/// `Fixed` operand, the way `x0` works on aarch64).
const ACC: Gpr = RAX;

/// The incoming arguments: the thread (unused today) and the Lua frame base.
const ARG: [Gpr; 2] = [Gpr(7), Gpr(6)]; // rdi, rsi

/// Callee-saved registers the allocator may hand out. The prologue saves *only the
/// ones a region actually used* and the epilogue restores them (see
/// [`Encoder::used_callee_saved`]), so a region that fits in the caller-saved set
/// pays nothing. `rbx`, `rbp` — which is not used as a frame pointer, so it is an
/// ordinary register — and `r12`–`r15`. SysV has no callee-saved `xmm`.
const CALLEE_SAVED: [Gpr; 6] = [Gpr(3), Gpr(5), Gpr(12), Gpr(13), Gpr(14), Gpr(15)];

/// One spill slot is one machine word.
const SLOT: i32 = 8;

/// The allocatable integer registers, the target's preference order: caller-saved
/// first (a compiled region is a leaf, so nothing clobbers them and they cost
/// nothing to use), then callee-saved (`rbx`/`rbp`/`r12`–`r15`, saved only when
/// used). `rax`/`rcx`/`rdx` sit at the back of the caller-saved run because
/// `idiv`/the return word want them, so the allocator reaches for them last. Only
/// three registers stay out: `rsp` (the stack) and `rsi`/`rdi` (the incoming args).
const INT_POOL: &[u8] = &[8, 9, 10, 11, 0, 1, 2, 3, 5, 12, 13, 14, 15];

/// Every `xmm` register is allocatable — the System V ABI has no callee-saved ones,
/// and the encoder keeps no float scratch of its own (it asks for temps).
const FLOAT_POOL: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

pub fn machine_env() -> MachineEnv {
    MachineEnv {
        allocation_order: [
            INT_POOL
                .iter()
                .map(|&r| PReg::new(RegClass::Int, r))
                .collect(),
            FLOAT_POOL
                .iter()
                .map(|&r| PReg::new(RegClass::Float, r))
                .collect(),
        ],
    }
}

/// Does `imm` fit x86's sign-extended 32-bit immediate form?
fn fits_i32(imm: i64) -> bool {
    i32::try_from(imm).is_ok()
}

/// The scratch registers, by class, that the encoder needs for an op but does not
/// take as an operand — a wide immediate's holder, the mask a float negate builds.
fn temp_classes(op: MOp) -> &'static [RegClass] {
    match op {
        // A compare against a wide immediate materializes it into a register first.
        MOp::GuardCmpImm { imm, .. } | MOp::BrCmpImm { imm, .. } if !fits_i32(imm) => {
            &[RegClass::Int]
        }
        // Float negate flips the sign bit with a mask built in a gpr and an xmm.
        MOp::FAlu(FAluOp::Neg) => &[RegClass::Int, RegClass::Float],
        // `==`/`~=` on floats fold the parity flag through a second setcc register.
        MOp::FCmpSet(Cc::Eq | Cc::Ne) => &[RegClass::Int],
        _ => &[],
    }
}

/// Attach the ISA's operand constraints, clobbers, and scratch temps to the machine
/// IR, so the allocator can hand out `rax`/`rcx`/`rdx` and the former scratch
/// registers as ordinary registers. Runs after isel, before allocation.
pub fn annotate(m: &mut MFunc) {
    let rax = PReg::new(RegClass::Int, 0);
    let rcx = PReg::new(RegClass::Int, 1);
    let rdx = PReg::new(RegClass::Int, 2);

    for i in 0..m.insts.len() {
        for &class in temp_classes(m.insts[i].op) {
            let t = m.new_vreg(class);
            m.insts[i].temps.push(Operand::reg(t));
        }

        match m.insts[i].op {
            // `idiv` divides `rdx:rax`; the quotient (and, after `floor_mod`'s
            // correction, the remainder) ends up in `rax`. `rdx` is overwritten and
            // the sign correction works through `rcx`.
            MOp::Alu(AluOp::IDiv | AluOp::Mod) => {
                m.insts[i].defs[0].constraint = Constraint::Fixed(rax);
                m.insts[i].clobbers = vec![rcx, rdx];
            }
            // Every other integer op is two-address (`d = d op b`) or unary in place
            // (`neg d`): the result reuses its first source's register.
            MOp::Alu(_) => m.insts[i].defs[0].constraint = Constraint::Reuse(0),
            // Shift-by-immediate is two-address too: it rewrites its source in place.
            MOp::AluImm(AluOp::Shl | AluOp::Lsr | AluOp::Sar, _) => {
                m.insts[i].defs[0].constraint = Constraint::Reuse(0)
            }
            // Float arithmetic is two-address the same way.
            MOp::FAlu(_) => m.insts[i].defs[0].constraint = Constraint::Reuse(0),
            _ => {}
        }
    }

    // Rematerializable constants: a value defined once by a pure constant can be
    // replayed at a use instead of spilled. `EntryArg` is excluded — it reads an
    // argument register the pool may have reused. Integer class only (`mov_imm`).
    let mut def_count = vec![0u32; m.classes.len()];
    for inst in &m.insts {
        for o in &inst.defs {
            def_count[o.vreg.0 as usize] += 1;
        }
    }
    for i in 0..m.insts.len() {
        let const_op = matches!(
            m.insts[i].op,
            MOp::Imm(_) | MOp::ShapeAddr(_) | MOp::ConstPayload(_)
        );
        if !const_op || m.insts[i].defs.len() != 1 {
            continue;
        }
        let v = m.insts[i].defs[0].vreg;
        if def_count[v.0 as usize] == 1 && m.classes[v.0 as usize] == RegClass::Int {
            m.remat.insert(v, i);
        }
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
    /// The callee-saved registers this region touched, computed before the prologue;
    /// only these are pushed and popped.
    saved: Vec<Gpr>,
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
        saved: Vec::new(),
    };
    e.run();

    Ok(e.a.finish())
}

impl Encoder<'_, '_> {
    fn run(&mut self) {
        self.saved = self.used_callee_saved();
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
        // Undo the frame, restore the saved registers in reverse, return. `rax` (the
        // status word) is untouched by any of this. Leaf region, so the body's stack
        // alignment is nobody's concern — only that `rsp` comes back to where it was.
        if self.frame > 0 {
            assert!(self.a.try_add_imm(RSP, self.frame as i64));
        }
        for r in self.saved.clone().into_iter().rev() {
            self.a.pop(r);
        }
        self.a.ret();
    }

    /// The callee-saved registers the allocation put to use anywhere — as an operand,
    /// a temp, a reload/bounce target, or an exit stub's scratch — so the prologue
    /// saves exactly those and nothing more. SysV has no callee-saved `xmm`, so only
    /// integer registers matter.
    fn used_callee_saved(&self) -> Vec<Gpr> {
        fn mark(used: &mut [bool; 16], a: Alloc) {
            if let Alloc::Reg(r) = a
                && r.class() == RegClass::Int
            {
                used[r.num() as usize] = true;
            }
        }

        let mut used = [false; 16];
        for i in 0..self.m.num_insts() {
            for k in 0..self.m.inst(i).defs.len() {
                mark(&mut used, self.ra.def(i, k));
            }
            for k in 0..self.m.inst(i).uses.len() {
                mark(&mut used, self.ra.use_(i, k));
            }
            for k in 0..self.m.inst(i).temps.len() {
                mark(&mut used, self.ra.temp(i, k));
            }
            for p in [ProgPoint::before(i), ProgPoint::after(i)] {
                for e in self.ra.edits_at(p) {
                    match *e {
                        Edit::Move(m) => {
                            mark(&mut used, m.from);
                            mark(&mut used, m.to);
                        }
                        Edit::Remat { to, .. } => mark(&mut used, Alloc::Reg(to)),
                    }
                }
            }
        }
        // Exit stubs borrow scratch that can reach the callee-saved registers; the
        // same choice `exit_stub` makes, so the saves match what the stubs clobber.
        for e in 0..self.m.exits.len() {
            let (base, s0, _) = self.stub_scratch(self.m.exits[e].inst);
            for g in [base, s0] {
                used[g.0 as usize] = true;
            }
        }

        CALLEE_SAVED
            .iter()
            .copied()
            .filter(|r| used[r.0 as usize])
            .collect()
    }

    fn prologue(&mut self) {
        for r in self.saved.clone() {
            self.a.push(r);
        }
        if self.frame > 0 {
            assert!(self.a.try_sub_imm(RSP, self.frame as i64));
        }
    }

    // --- operand access -----------------------------------------------------

    /// Read a value from where the allocator put it, loading a spilled one into
    /// `scratch`. Only the exit stubs need this — their keepalives are `Any`, so
    /// one may be on the stack; every fast-path operand is a register.
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

    fn reg(a: Alloc) -> u8 {
        match a {
            Alloc::Reg(r) => r.num(),
            Alloc::Spill(_) => unreachable!("a Reg operand is never on the stack"),
        }
    }

    fn use_g(&self, i: Inst, k: usize) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Int);
        Gpr(Self::reg(self.ra.use_(i, k)))
    }

    fn use_f(&self, i: Inst, k: usize) -> Xmm {
        debug_assert_eq!(self.m.class(self.m.inst(i).use_vreg(k)), RegClass::Float);
        Xmm(Self::reg(self.ra.use_(i, k)))
    }

    fn def_g(&self, i: Inst, k: usize) -> Gpr {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Int);
        Gpr(Self::reg(self.ra.def(i, k)))
    }

    fn def_f(&self, i: Inst, k: usize) -> Xmm {
        debug_assert_eq!(self.m.class(self.m.inst(i).def_vreg(k)), RegClass::Float);
        Xmm(Self::reg(self.ra.def(i, k)))
    }

    fn temp_g(&self, i: Inst, k: usize) -> Gpr {
        Gpr(Self::reg(self.ra.temp(i, k)))
    }

    fn temp_f(&self, i: Inst, k: usize) -> Xmm {
        Xmm(Self::reg(self.ra.temp(i, k)))
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

    /// A move the allocator's reload phase asked for: a spill store, a reload, a
    /// register-to-register shuffle, or a bounce to and from a scratch slot. Never
    /// stack-to-stack — the allocator always routes through a register.
    fn emit_move(&mut self, m: Move) {
        match m.class {
            RegClass::Int => match (m.from, m.to) {
                (Alloc::Reg(f), Alloc::Reg(t)) => self.a.mov(Gpr(t.num()), Gpr(f.num())),
                (Alloc::Reg(f), Alloc::Spill(s)) => {
                    self.a.store(RSP, s as i32 * SLOT, Gpr(f.num()))
                }
                (Alloc::Spill(s), Alloc::Reg(t)) => self.a.load(Gpr(t.num()), RSP, s as i32 * SLOT),
                (Alloc::Spill(_), Alloc::Spill(_)) => {
                    unreachable!("no allocator edit is stack to stack")
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
                (Alloc::Spill(_), Alloc::Spill(_)) => {
                    unreachable!("no allocator edit is stack to stack")
                }
            },
        }
    }

    /// Replay a spilled constant into `to` instead of loading a slot it never got.
    /// The source is a pure, input-free constant op — `annotate` guarantees that.
    fn emit_remat(&mut self, src: Inst, to: Gpr) {
        match self.m.inst(src).op {
            MOp::Imm(v) => self.a.mov_imm(to, v),
            MOp::ShapeAddr(s) => self.a.mov_imm(to, self.shape_word(s)),
            MOp::ConstPayload(c) => {
                let bits = self.pool.value(c).raw_payload() as i64;
                self.a.mov_imm(to, bits);
            }
            op => unreachable!("remat source is a pure constant, got {op:?}"),
        }
    }

    fn edits_at(&mut self, p: ProgPoint) {
        for e in self.ra.edits_at(p).copied().collect::<Vec<_>>() {
            match e {
                Edit::Move(m) => self.emit_move(m),
                Edit::Remat { src, to, .. } => self.emit_remat(src, Gpr(to.num())),
            }
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
                let d = self.def_g(i, 0);
                self.a.mov(d, ARG[n as usize]);
            }
            MOp::Imm(v) => {
                let d = self.def_g(i, 0);
                self.a.mov_imm(d, v);
            }
            MOp::ShapeAddr(s) => {
                let d = self.def_g(i, 0);
                self.a.mov_imm(d, self.shape_word(s));
            }
            MOp::ConstPayload(c) => {
                let d = self.def_g(i, 0);
                let bits = self.pool.value(c).raw_payload() as i64;
                self.a.mov_imm(d, bits);
            }
            MOp::Mov => match self.m.class(self.m.inst(i).def_vreg(0)) {
                RegClass::Int => {
                    let s = self.use_g(i, 0);
                    let d = self.def_g(i, 0);
                    if d != s {
                        self.a.mov(d, s);
                    }
                }
                RegClass::Float => {
                    let s = self.use_f(i, 0);
                    let d = self.def_f(i, 0);
                    if d != s {
                        self.a.movsd(d, s);
                    }
                }
            },
            MOp::Load { off, width } => {
                let base = self.use_g(i, 0);
                let d = self.def_g(i, 0);
                match width {
                    Width::U64 => self.a.load(d, base, off),
                    Width::U8 => self.a.load8(d, base, off),
                }
            }
            MOp::Store { off, width } => {
                let base = self.use_g(i, 0);
                let val = self.use_g(i, 1);
                match width {
                    Width::U64 => self.a.store(base, off, val),
                    Width::U8 => self.a.store8(base, off, val),
                }
            }
            MOp::Alu(o) => self.alu(i, o),
            // Only shifts arrive as `AluImm`: x86 has a shift-by-immediate form,
            // and using it dodges the `cl`-only variable shift (and its fixed-`rcx`
            // constraint). Every other immediate is still lowered to `Imm` + `Alu`.
            MOp::AluImm(o @ (AluOp::Shl | AluOp::Lsr | AluOp::Sar), imm) => {
                // Two-address: `annotate` marked the def `Reuse(0)`, so it already
                // holds the value to shift; the op rewrites it in place.
                let d = self.def_g(i, 0);
                let amt = imm as u8;
                match o {
                    AluOp::Shl => self.a.shl_imm(d, amt),
                    AluOp::Lsr => self.a.shr_imm(d, amt),
                    AluOp::Sar => self.a.sar_imm(d, amt),
                    _ => unreachable!("outer match restricts these"),
                }
            }
            MOp::AluImm(..) => {
                unreachable!("isel lowers non-shift immediates into an `Imm` + `Alu`")
            }
            // Two-address: the def already holds its first source (`annotate` marked
            // it `Reuse(0)`, the allocator copied it in), so the op writes the def.
            MOp::FAlu(o) => match o {
                FAluOp::Neg => {
                    // Flip the sign bit with a mask built in a gpr temp and an xmm temp.
                    let d = self.def_f(i, 0);
                    let t0 = self.temp_g(i, 0);
                    let t1 = self.temp_f(i, 1);
                    self.a.mov_imm(t0, i64::MIN);
                    self.a.movq_to_xmm(t1, t0);
                    self.a.xorpd(d, t1);
                }
                _ => {
                    let d = self.def_f(i, 0);
                    let m = self.use_f(i, 1);
                    match o {
                        FAluOp::Add => self.a.addsd(d, m),
                        FAluOp::Sub => self.a.subsd(d, m),
                        FAluOp::Mul => self.a.mulsd(d, m),
                        FAluOp::Div => self.a.divsd(d, m),
                        FAluOp::Neg => unreachable!("handled above"),
                    }
                }
            },
            MOp::ICmpSet(cc) => {
                let n = self.use_g(i, 0);
                let m = self.use_g(i, 1);
                let d = self.def_g(i, 0);
                self.a.cmp(n, m);
                self.a.setcc(int_cond(cc), d);
                self.a.movzx8(d, d);
            }
            MOp::FCmpSet(cc) => {
                let n = self.use_f(i, 0);
                let m = self.use_f(i, 1);
                let d = self.def_g(i, 0);
                let fold = matches!(cc, Cc::Eq | Cc::Ne).then(|| self.temp_g(i, 0));
                self.fcmp_set(cc, n, m, d, fold);
            }
            MOp::BitsToFloat => {
                let s = self.use_g(i, 0);
                let d = self.def_f(i, 0);
                self.a.movq_to_xmm(d, s);
            }
            MOp::FloatToBits => {
                let s = self.use_f(i, 0);
                let d = self.def_g(i, 0);
                self.a.movq_to_gpr(d, s);
            }
            MOp::SiToFp => {
                let s = self.use_g(i, 0);
                let d = self.def_f(i, 0);
                self.a.cvtsi2sd(d, s);
            }

            // A guard jumps to its stub when the condition it asserts is *false*.
            MOp::GuardCmp { cc, exit } => {
                let n = self.use_g(i, 0);
                let m = self.use_g(i, 1);
                self.a.cmp(n, m);
                let target = self.exits[exit.0 as usize];
                self.a.jcc(int_cond(cc).invert(), target);
            }
            MOp::GuardCmpImm { cc, imm, exit } => {
                self.cmp_imm(i, cc, imm);
                let target = self.exits[exit.0 as usize];
                self.a.jcc(int_cond(cc).invert(), target);
            }
            MOp::GuardNz { exit } => {
                let n = self.use_g(i, 0);
                self.a.test(n, n);
                let target = self.exits[exit.0 as usize];
                self.a.jcc(Cond::E, target);
            }

            MOp::Jump(b) => {
                let target = self.blocks[b.0 as usize];
                self.a.jmp(target);
            }
            MOp::BrNz { then_, else_ } => {
                let c = self.use_g(i, 0);
                self.a.test(c, c);
                let t = self.blocks[then_.0 as usize];
                let e = self.blocks[else_.0 as usize];
                self.a.jcc(Cond::Ne, t);
                self.a.jmp(e);
            }
            // The compare fused in: taken on the condition itself, not its inverse
            // (a guard inverts because it branches on *failure*; this branches on
            // success to `then_`).
            MOp::BrCmp { cc, then_, else_ } => {
                let n = self.use_g(i, 0);
                let m = self.use_g(i, 1);
                self.a.cmp(n, m);
                let t = self.blocks[then_.0 as usize];
                let e = self.blocks[else_.0 as usize];
                self.a.jcc(int_cond(cc), t);
                self.a.jmp(e);
            }
            MOp::BrCmpImm {
                cc,
                imm,
                then_,
                else_,
            } => {
                self.cmp_imm(i, cc, imm);
                let t = self.blocks[then_.0 as usize];
                let e = self.blocks[else_.0 as usize];
                self.a.jcc(int_cond(cc), t);
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

    /// Set flags for `use0 cc imm`, for a guard or branch that folded a constant
    /// operand. `cc` is passed only to spot the `== 0`/`!= 0` case, where `test`
    /// is a byte shorter than `cmp $0` and needs no immediate; the caller still
    /// issues the conditional jump.
    fn cmp_imm(&mut self, i: Inst, cc: Cc, imm: i64) {
        let n = self.use_g(i, 0);
        if imm == 0 && matches!(cc, Cc::Eq | Cc::Ne) {
            self.a.test(n, n);
        } else if fits_i32(imm) {
            let ok = self.a.try_cmp_imm(n, imm);
            debug_assert!(ok, "a 32-bit immediate must encode");
        } else {
            let t = self.temp_g(i, 0);
            self.a.mov_imm(t, imm);
            self.a.cmp(n, t);
        }
    }

    /// A two-address integer op. Every form but the divides writes its result in
    /// place, into the def register the allocator has already loaded with the first
    /// source (`Reuse(0)`). The divides are macros that leave their result in `rax`
    /// (a `Fixed` def) and read the operands themselves.
    fn alu(&mut self, i: Inst, o: AluOp) {
        match o {
            AluOp::IDiv => {
                let n = self.use_g(i, 0);
                let m = self.use_g(i, 1);
                self.floor_div(n, m);
            }
            AluOp::Mod => {
                let n = self.use_g(i, 0);
                let m = self.use_g(i, 1);
                self.floor_mod(n, m);
            }
            AluOp::Neg => {
                let d = self.def_g(i, 0);
                self.a.neg(d);
            }
            AluOp::Not => {
                let d = self.def_g(i, 0);
                self.a.not(d);
            }
            AluOp::Add | AluOp::Sub | AluOp::Mul | AluOp::And | AluOp::Or | AluOp::Xor => {
                let d = self.def_g(i, 0);
                let m = self.use_g(i, 1);
                match o {
                    AluOp::Add => self.a.add(d, m),
                    AluOp::Sub => self.a.sub(d, m),
                    AluOp::Mul => self.a.imul(d, m),
                    AluOp::And => self.a.and(d, m),
                    AluOp::Or => self.a.or(d, m),
                    AluOp::Xor => self.a.xor(d, m),
                    _ => unreachable!("outer match restricts these"),
                }
            }
            // Shifts are only ever emitted by immediate (`AluImm`), never as a
            // register-register `Alu`, so this variant cannot appear.
            AluOp::Shl | AluOp::Sar | AluOp::Lsr => unreachable!("shifts are emitted as AluImm"),
        }
    }

    /// `d = (n cc m) ? 1 : 0` for a float compare, Lua's NaN rule baked in.
    ///
    /// `comisd` sets only the unsigned flags and reports unordered as `CF=ZF=PF=1`.
    /// `seta`/`setae` (CF clear) are therefore the tests that read false against a
    /// NaN, which is what `<`/`<=`/`>`/`>=` want after orienting the operands; only
    /// `==` and `~=` have to fold in the parity flag.
    fn fcmp_set(&mut self, cc: Cc, n: Xmm, m: Xmm, d: Gpr, fold: Option<Gpr>) {
        match cc {
            Cc::Lt => {
                self.a.comisd(m, n); // m > n  ⇔  n < m
                self.a.setcc(Cond::A, d);
                self.a.movzx8(d, d);
            }
            Cc::Le => {
                self.a.comisd(m, n);
                self.a.setcc(Cond::Ae, d);
                self.a.movzx8(d, d);
            }
            Cc::Gt => {
                self.a.comisd(n, m);
                self.a.setcc(Cond::A, d);
                self.a.movzx8(d, d);
            }
            Cc::Ge => {
                self.a.comisd(n, m);
                self.a.setcc(Cond::Ae, d);
                self.a.movzx8(d, d);
            }
            Cc::Eq => {
                // Ordered and equal: ZF set *and* PF clear.
                let t = fold.expect("float `==` needs a fold temp");
                self.a.comisd(n, m);
                self.a.setcc(Cond::E, d);
                self.a.movzx8(d, d);
                self.a.setcc(Cond::Np, t);
                self.a.movzx8(t, t);
                self.a.and(d, t);
            }
            Cc::Ne => {
                // Not-equal or unordered: ZF clear *or* PF set — the one comparison
                // a NaN satisfies.
                let t = fold.expect("float `~=` needs a fold temp");
                self.a.comisd(n, m);
                self.a.setcc(Cond::Ne, d);
                self.a.movzx8(d, d);
                self.a.setcc(Cond::P, t);
                self.a.movzx8(t, t);
                self.a.or(d, t);
            }
        }
    }

    // --- exit stubs ---------------------------------------------------------

    /// Scratch for an exit stub: registers no keepalive of `guard` occupies, so
    /// loading spilled keepalives into them destroys nothing the stub still has to
    /// write back. The stub is cold, so any non-keepalive register holds a dead
    /// fast-path value. One gpr holds the frame base across the run, a second
    /// ferries each slot's words, and one xmm unpacks a spilled float.
    ///
    /// Only *two* gprs, not one per word: a slot's payload is stored before its tag
    /// is materialized, so the ferry register is dead again by the time the tag
    /// needs it. That matters — the pool is 13 registers and a guard in a hot loop
    /// can keep 11 of them alive, so every scratch register this does not demand is
    /// one the fast path gets to use.
    fn stub_scratch(&self, guard: Inst) -> (Gpr, Gpr, Xmm) {
        let mut busy_i = [false; 16];
        let mut busy_f = [false; 16];
        for k in 0..self.m.inst(guard).uses.len() {
            match self.ra.use_(guard, k) {
                Alloc::Reg(r) if r.class() == RegClass::Int => busy_i[r.num() as usize] = true,
                Alloc::Reg(r) => busy_f[r.num() as usize] = true,
                _ => {}
            }
        }
        let mut ints = INT_POOL.iter().copied().filter(|&r| !busy_i[r as usize]);
        let mut next = || {
            Gpr(ints
                .next()
                .expect("the pool outnumbers a guard's keepalives"))
        };
        let (base, s0) = (next(), next());
        let f0 = FLOAT_POOL
            .iter()
            .copied()
            .find(|&r| !busy_f[r as usize])
            .expect("an xmm register is free of the keepalives");
        (base, s0, Xmm(f0))
    }

    /// Materialize the interpreter's view of the frame, then return. Every store
    /// is a `Value`: a payload word and a tag byte, at `base + reg * 16`.
    fn exit_stub(&mut self, e: ExitId) {
        let label = self.exits[e.0 as usize];
        self.a.bind(label);

        let stub = self.m.exits[e.0 as usize].clone();
        let (sc_base, sc0, sc_f) = self.stub_scratch(stub.inst);

        // The base survives every value the stub loads; a keepalive already in a
        // register stays there, otherwise it comes off the stack into `sc_base`.
        let base_at = self.at_exit(e, self.m.frame_base);
        let base = self.read_g(base_at, sc_base);

        for (reg, src) in stub.slots {
            let slot = reg as i32 * layout::val::SIZE as i32;
            let (payload_off, kind_off) = (
                slot + layout::val::DATA as i32,
                slot + layout::val::KIND as i32,
            );

            let (payload, tag) = match src {
                ExitSrc::Boxed { payload, tag } => {
                    let a = self.at_exit(e, payload);
                    (self.read_g(a, sc0), tag)
                }
                ExitSrc::Int(v) => {
                    let a = self.at_exit(e, v);
                    (self.read_g(a, sc0), Tag::Const(ValueKind::Integer))
                }
                ExitSrc::Float(v) => {
                    let a = self.at_exit(e, v);
                    let f = self.read_f(a, sc_f);
                    self.a.movq_to_gpr(sc0, f);
                    (sc0, Tag::Const(ValueKind::Float))
                }
                ExitSrc::Const { payload, tag } => {
                    self.a.mov_imm(sc0, payload as i64);
                    (sc0, Tag::Const(tag))
                }
            };
            self.a.store(base, payload_off, payload);

            // `sc0` is free again: whatever it held has just been stored.
            let tag_reg = match tag {
                Tag::Const(k) => {
                    self.a.mov_imm(sc0, layout::kind(k) as i64);
                    sc0
                }
                Tag::Dyn(t) => {
                    let a = self.at_exit(e, t);
                    self.read_g(a, sc0)
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

        m.set_layout().expect("single block is reducible");
        (m, sum, def_sum, store)
    }

    /// Drive the edit path by hand: compute the sum in r8, move it out through
    /// `edits`, and let the very next instruction (`junk`) take r8 over, with the
    /// store reading it from `store_use`. Drop the moves and the region returns
    /// `0x7fffdead`. It is the path the allocator's reload phase drives constantly,
    /// but easier to read driven by hand.
    fn split_sum_through(store_use: Alloc, edits: Vec<(ProgPoint, Move)>, num_spills: u32) -> i64 {
        let (m, sum, _def_sum, store) = add_and_store();

        let mut b = AllocationBuilder::new(&m);
        b.assign(m.frame_base, Alloc::Reg(x(12)));
        // `add` is two-address, so the sum's register must already hold lhs: lhs and
        // sum share r8 (lhs dies into the add), exactly what a `Reuse(0)` allocation
        // produces. This test builds the allocation by hand, without `annotate`.
        b.assign(VReg(1), Alloc::Reg(x(8))); // lhs, reused as the sum's register
        b.assign(VReg(2), Alloc::Reg(x(13))); // rhs
        b.assign(VReg(5), Alloc::Reg(x(14))); // tag

        b.assign(sum, Alloc::Reg(x(8)));
        b.assign(VReg(4), Alloc::Reg(x(8))); // junk, which takes r8 over
        b.set_use(store, 1, store_use);
        for (p, mv) in edits {
            b.edit(p, Edit::Move(mv));
        }

        let ra = b.finish(num_spills);
        let words = encode(&m, &ConstPool::new(), &ra).expect("encode");
        let code = Code::from_words(&words).expect("map code");

        let mut stack = vec![Value::nil(); 1];
        let region: Region = unsafe { std::mem::transmute(code.entry()) };
        let status = Status::unpack(region(std::ptr::null_mut(), stack.as_mut_ptr().cast()));

        assert_eq!(status, Status::Return(1));
        stack[0].get_integer().expect("an integer result")
    }

    fn mv(from: Alloc, to: Alloc) -> Move {
        Move {
            from,
            to,
            class: RegClass::Int,
        }
    }

    /// One move: r8 to r15, so the store reads the sum from r15.
    #[test]
    fn a_split_into_a_register_emits_its_move() {
        let (_, _, def_sum, _) = add_and_store();
        let edits = vec![(
            ProgPoint::after(def_sum),
            mv(Alloc::Reg(x(8)), Alloc::Reg(x(15))),
        )];
        assert_eq!(split_sum_through(Alloc::Reg(x(15)), edits, 0), 42);
    }

    /// Two moves through a slot — spill then reload — the exact shape the reload
    /// phase emits for a value that lives on the stack between its def and a use.
    #[test]
    fn a_split_onto_the_stack_emits_its_moves() {
        let (_, _, def_sum, store) = add_and_store();
        let edits = vec![
            (
                ProgPoint::after(def_sum),
                mv(Alloc::Reg(x(8)), Alloc::Spill(0)),
            ),
            (
                ProgPoint::before(store),
                mv(Alloc::Spill(0), Alloc::Reg(x(15))),
            ),
        ];
        assert_eq!(split_sum_through(Alloc::Reg(x(15)), edits, 1), 42);
    }
}
