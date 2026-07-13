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
use crate::jit::backend::asm::{Asm, Cond, FP, Fpr, Gpr, LR, Label, SP};
use crate::jit::backend::code::{Code, CodeBuf};
use crate::jit::backend::layout;
use crate::jit::backend::mach::{
    AluOp, ExitId, ExitSrc, FAluOp, MBlock, MFunc, MOp, RegClass, Tag, VReg, Width,
};
use crate::jit::backend::regalloc::{Alloc, Allocation};
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
const F0: Fpr = Fpr(16);
const F1: Fpr = Fpr(17);
const F2: Fpr = Fpr(18);

/// The two incoming arguments: the thread, and the Lua frame base.
const ARG: [Gpr; 2] = [Gpr(0), Gpr(1)];

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

pub fn encode(m: &MFunc, pool: &ConstPool<'_>, ra: &Allocation) -> Result<Code, EncodeError> {
    let mut a = Asm::new();
    let blocks = (0..m.blocks.len()).map(|_| a.new_label()).collect();
    let exits = (0..m.exits.len()).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();

    let frame = (ra.num_spills * 8).next_multiple_of(16);
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

    let words = e.a.finish();
    let mut buf = CodeBuf::new(words.len() * 4).expect("mmap code");
    buf.write(|code| {
        for (i, w) in words.iter().enumerate() {
            code[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    });
    Ok(buf.finalize())
}

impl Encoder<'_, '_> {
    fn run(&mut self) {
        self.prologue();

        // The entry block first, so it falls through from the prologue; the rest in
        // index order. Order is otherwise free — every edge is an explicit branch.
        let entry = self.m.entry;
        self.block(entry);
        for b in 0..self.m.blocks.len() as u32 {
            if MBlock(b) != entry {
                self.block(MBlock(b));
            }
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
    // Written against `Alloc` rather than against "everything is spilled", so a
    // real allocator drops in without touching a line below. When a value is
    // already in a register these are free; when it is spilled they cost the load
    // or store, at the scratch register the caller nominates.

    /// The register holding `v`, loading it into `scratch` if it is spilled.
    fn use_g(&mut self, v: VReg, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(v), RegClass::Int);
        match self.ra.of(v) {
            Alloc::Reg(r) => Gpr(r.0),
            Alloc::Spill(s) => {
                assert!(self.a.try_ldr(scratch, SP, (s * 8) as i32));
                scratch
            }
        }
    }

    fn use_f(&mut self, v: VReg, scratch: Fpr) -> Fpr {
        debug_assert_eq!(self.m.class(v), RegClass::Float);
        match self.ra.of(v) {
            Alloc::Reg(r) => Fpr(r.0),
            Alloc::Spill(s) => {
                assert!(self.a.try_ldr_f(scratch, SP, (s * 8) as i32));
                scratch
            }
        }
    }

    /// Where to write `v`. Pair every call with [`Self::def_done`], which is a
    /// no-op for a value in a register and the spill store otherwise.
    fn def_g(&mut self, v: VReg, scratch: Gpr) -> Gpr {
        debug_assert_eq!(self.m.class(v), RegClass::Int);
        match self.ra.of(v) {
            Alloc::Reg(r) => Gpr(r.0),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_f(&mut self, v: VReg, scratch: Fpr) -> Fpr {
        debug_assert_eq!(self.m.class(v), RegClass::Float);
        match self.ra.of(v) {
            Alloc::Reg(r) => Fpr(r.0),
            Alloc::Spill(_) => scratch,
        }
    }

    fn def_done(&mut self, v: VReg, from: Gpr) {
        if let Alloc::Spill(s) = self.ra.of(v) {
            assert!(self.a.try_str(from, SP, (s * 8) as i32));
        }
    }

    fn def_done_f(&mut self, v: VReg, from: Fpr) {
        if let Alloc::Spill(s) = self.ra.of(v) {
            assert!(self.a.try_str_f(from, SP, (s * 8) as i32));
        }
    }

    // --- blocks -------------------------------------------------------------

    fn block(&mut self, b: MBlock) {
        let label = self.blocks[b.0 as usize];
        self.a.bind(label);
        for &i in &self.m.block(b).insts.clone() {
            self.inst(i);
        }
    }

    fn inst(&mut self, i: usize) {
        let inst = self.m.inst(i).clone();
        let (op, defs, uses) = (inst.op, inst.defs, inst.uses);

        match op {
            MOp::EntryArg(n) => {
                // The incoming argument registers are still live here: the entry
                // block runs before anything can clobber them, and the scratch
                // registers deliberately do not overlap `x0`/`x1`.
                let d = self.def_g(defs[0], S0);
                self.a.mov(d, ARG[n as usize]);
                self.def_done(defs[0], d);
            }
            MOp::Imm(v) => {
                let d = self.def_g(defs[0], S0);
                self.a.mov_imm(d, v);
                self.def_done(defs[0], d);
            }
            MOp::ShapeAddr(s) => {
                let d = self.def_g(defs[0], S0);
                self.a.mov_imm(d, self.shape_word(s));
                self.def_done(defs[0], d);
            }
            MOp::ConstPayload(c) => {
                let d = self.def_g(defs[0], S0);
                let bits = self.pool.value(c).raw_payload() as i64;
                self.a.mov_imm(d, bits);
                self.def_done(defs[0], d);
            }
            MOp::Mov => match self.m.class(defs[0]) {
                RegClass::Int => {
                    let s = self.use_g(uses[0], S0);
                    let d = self.def_g(defs[0], S1);
                    self.a.mov(d, s);
                    self.def_done(defs[0], d);
                }
                RegClass::Float => {
                    let s = self.use_f(uses[0], F0);
                    let d = self.def_f(defs[0], F1);
                    self.a.fmov(d, s);
                    self.def_done_f(defs[0], d);
                }
            },
            MOp::Load { off, width } => {
                let base = self.use_g(uses[0], S0);
                let d = self.def_g(defs[0], S1);
                let ok = match width {
                    Width::U64 => self.a.try_ldr(d, base, off),
                    Width::U8 => self.a.try_ldrb(d, base, off),
                };
                assert!(ok, "load offset {off} out of range");
                self.def_done(defs[0], d);
            }
            MOp::Store { off, width } => {
                let base = self.use_g(uses[0], S0);
                let val = self.use_g(uses[1], S1);
                let ok = match width {
                    Width::U64 => self.a.try_str(val, base, off),
                    Width::U8 => self.a.try_strb(val, base, off),
                };
                assert!(ok, "store offset {off} out of range");
            }
            MOp::Alu(o) => {
                let d = match o {
                    AluOp::Neg | AluOp::Not => {
                        let n = self.use_g(uses[0], S0);
                        let d = self.def_g(defs[0], S2);
                        match o {
                            AluOp::Neg => self.a.neg(d, n),
                            _ => self.a.mvn(d, n),
                        }
                        d
                    }
                    _ => {
                        let n = self.use_g(uses[0], S0);
                        let m = self.use_g(uses[1], S1);
                        let d = self.def_g(defs[0], S2);
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
                            AluOp::Neg | AluOp::Not => unreachable!("handled above"),
                        }
                        d
                    }
                };
                self.def_done(defs[0], d);
            }
            MOp::AluImm(o, imm) => {
                let n = self.use_g(uses[0], S0);
                let d = self.def_g(defs[0], S2);
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
                        AluOp::Neg | AluOp::Not => panic!("{o:?} takes no immediate"),
                    }
                }
                self.def_done(defs[0], d);
            }
            MOp::FAlu(o) => {
                let d = match o {
                    FAluOp::Neg => {
                        let n = self.use_f(uses[0], F0);
                        let d = self.def_f(defs[0], F2);
                        self.a.fneg(d, n);
                        d
                    }
                    _ => {
                        let n = self.use_f(uses[0], F0);
                        let m = self.use_f(uses[1], F1);
                        let d = self.def_f(defs[0], F2);
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
                self.def_done_f(defs[0], d);
            }
            MOp::ICmpSet(cc) => {
                let n = self.use_g(uses[0], S0);
                let m = self.use_g(uses[1], S1);
                let d = self.def_g(defs[0], S2);
                self.a.cmp(n, m);
                self.a.cset(d, int_cond(cc));
                self.def_done(defs[0], d);
            }
            MOp::FCmpSet(cc) => {
                let n = self.use_f(uses[0], F0);
                let m = self.use_f(uses[1], F1);
                let d = self.def_g(defs[0], S2);
                self.a.fcmp(n, m);
                self.a.cset(d, float_cond(cc));
                self.def_done(defs[0], d);
            }
            MOp::BitsToFloat => {
                let s = self.use_g(uses[0], S0);
                let d = self.def_f(defs[0], F0);
                self.a.fmov_to_fpr(d, s);
                self.def_done_f(defs[0], d);
            }
            MOp::FloatToBits => {
                let s = self.use_f(uses[0], F0);
                let d = self.def_g(defs[0], S0);
                self.a.fmov_to_gpr(d, s);
                self.def_done(defs[0], d);
            }
            MOp::SiToFp => {
                let s = self.use_g(uses[0], S0);
                let d = self.def_f(defs[0], F0);
                self.a.scvtf(d, s);
                self.def_done_f(defs[0], d);
            }

            // A guard branches to its stub when the condition it asserts is
            // *false*, and falls through otherwise. Uses past the operands are the
            // frame-state keepalives; they exist to hold registers open for the
            // stub and produce no code here.
            MOp::GuardCmp { cc, exit } => {
                let n = self.use_g(uses[0], S0);
                let m = self.use_g(uses[1], S1);
                self.a.cmp(n, m);
                let target = self.exits[exit.0 as usize];
                self.a.b_cond(int_cond(cc).invert(), target);
            }
            MOp::GuardCmpImm { cc, imm, exit } => {
                let n = self.use_g(uses[0], S0);
                if !self.a.try_cmp_imm(n, imm) {
                    self.a.mov_imm(S1, imm);
                    self.a.cmp(n, S1);
                }
                let target = self.exits[exit.0 as usize];
                self.a.b_cond(int_cond(cc).invert(), target);
            }
            MOp::GuardNz { exit } => {
                let n = self.use_g(uses[0], S0);
                let target = self.exits[exit.0 as usize];
                self.a.cbz(n, target);
            }

            MOp::Jump(b) => {
                let target = self.blocks[b.0 as usize];
                self.a.b(target);
            }
            MOp::BrNz { then_, else_ } => {
                let c = self.use_g(uses[0], S0);
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
        let base = self.use_g(self.m.frame_base, S2);

        for (reg, src) in stub.slots {
            let slot = reg as i32 * layout::val::SIZE as i32;
            let (payload_off, kind_off) = (
                slot + layout::val::DATA as i32,
                slot + layout::val::KIND as i32,
            );

            let (payload, tag) = match src {
                ExitSrc::Boxed { payload, tag } => (self.use_g(payload, S0), tag),
                ExitSrc::Int(v) => (self.use_g(v, S0), Tag::Const(ValueKind::Integer)),
                ExitSrc::Float(v) => {
                    let f = self.use_f(v, F0);
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
                Tag::Dyn(t) => self.use_g(t, S1),
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
