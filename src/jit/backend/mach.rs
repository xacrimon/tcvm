//! The machine IR: a flat, deliberately dumb list of near-machine instructions
//! over virtual registers.
//!
//! # Why a second IR at all
//!
//! The optimizing IR is in SSA with block parameters and values whose
//! representation (`Rep::Val`) does not fit in a machine register. Register
//! allocation and encoding want neither. This level fixes both: every operand is
//! a virtual register holding exactly one machine word, block parameters have
//! been destroyed into copies, and each instruction corresponds to roughly one
//! encoded instruction.
//!
//! # Boxed values
//!
//! A `Value` is a 16-byte tagged pair, so a `Rep::Val` needs *two* words. But the
//! tag is only worth a register when it is not already known: whenever the IR
//! type's set is monomorphic (`tab`, `int`, `tab<S0>` — nearly everything, after
//! specialization) the tag is a compile-time `ValueKind` and only the payload is
//! live. Hence [`Tag`]. Modelling every boxed value as a register *pair* would
//! double the live set of a specialized loop for facts the compiler already
//! knows.
//!
//! # Flags
//!
//! Not modelled. A comparison materializes its result into a general register
//! with `cset`, and a branch tests that register. That costs one instruction per
//! branch relative to fusing the compare into the branch; the fusion is a
//! peephole over adjacent instructions and is deliberately left for later, so
//! that nothing here has to reason about a live flags register.
//!
//! Guards are the exception, and they are *not* terminators: a guard compares and
//! branches out-of-line to its exit stub, falling through on success. Keeping
//! them inside blocks is what lets a block with six guards stay one block.

use std::collections::HashMap;
use std::fmt::{self, Write};

use foldhash::fast::RandomState;

use crate::env::value::ValueKind;
// The register allocator owns this vocabulary, not this module: it is the leaf
// that neither the machine IR nor the encoder is allowed to reach into, so the
// types a program is *described in* live down there. Re-exported so the rest of
// the backend still says `mach::VReg`.
pub use crate::jit::backend::regalloc::{Block as MBlock, RegClass, VReg};
use crate::jit::backend::regalloc::{Inst, Operand, PReg, RegallocFunc};
use crate::jit::ir::op::Cc;
use crate::jit::ir::pool::{ConstRef, ShapeRef};

/// An exit stub: one per IR `Exit`, plus the unconditional `Deopt`s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExitId(pub u32);

/// Where a boxed value's type tag lives.
///
/// `Const` is the common case after specialization and costs nothing: the tag is
/// an immediate the encoder writes directly. `Dyn` is needed only where the type
/// set is genuinely polymorphic — a `slot.get` result before its guard, say.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tag {
    Const(ValueKind),
    Dyn(VReg),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AluOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    /// Arithmetic (sign-propagating) shift right — Lua's `>>` is logical, but
    /// the IR distinguishes; this is the machine-level pair.
    Sar,
    Lsr,
    Neg,
    Not,

    // The only two that are not a single machine instruction. Lua's `%` and `//`
    // round toward *negative infinity*; every hardware divide truncates toward
    // zero. The difference is a correction term that the encoder expands inline,
    // so these stay one MIR op — splitting them into the seven real instructions
    // here would mean teaching the allocator about temporaries that only exist
    // between them.
    Mod,
    IDiv,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FAluOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

/// The width of a memory access. `Value`'s tag is one byte; everything else the
/// backend touches is a word.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Width {
    U8,
    U64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MOp {
    /// `def0 <- the nth incoming argument register`.
    ///
    /// The entry pointers (thread, frame base) are modelled as ordinary virtual
    /// registers rather than pinned physical ones, so the allocator is free to
    /// park the frame base in a callee-saved register — which is what it wants,
    /// since the base is live across the entire region.
    EntryArg(u8),
    /// `def0 <- imm`.
    Imm(i64),
    /// `def0 <- the address of a pool shape`.
    ///
    /// Symbolic rather than an `Imm`, because the address is not knowable here:
    /// it is a heap pointer, so it differs on every run. Baking it into the
    /// machine IR would make this level non-deterministic for no gain — the
    /// encoder is the only stage that should ever hold a raw address. Sound to
    /// bake in *there* only because the collector never moves an object and the
    /// pool keeps it alive.
    ShapeAddr(ShapeRef),
    /// `def0 <- the raw payload of a pool constant`: an integer, a float's bits,
    /// or a `Gc` address. Symbolic for the same reason.
    ConstPayload(ConstRef),
    /// `def0 <- use0`. Same register class.
    Mov,
    /// `def0 <- [use0 + off]`.
    Load {
        off: i32,
        width: Width,
    },
    /// `[use0 + off] <- use1`. Note both operands are *uses*.
    Store {
        off: i32,
        width: Width,
    },

    /// `def0 <- use0 op use1`, general registers.
    Alu(AluOp),
    /// `def0 <- use0 op imm`.
    AluImm(AluOp, i64),
    /// `def0 <- use0 op use1`, float registers.
    FAlu(FAluOp),

    /// `def0 <- (use0 cc use1) ? 1 : 0`, general registers.
    ICmpSet(Cc),
    /// Same, float operands, integer result. Ordered: NaN compares false, which
    /// is Lua's rule and aarch64's `b.mi`/`b.gt` family gives it directly.
    FCmpSet(Cc),

    /// `def0 (float) <- bits of use0 (int)`. A representation change, not a
    /// conversion — this is `unpack.float`.
    BitsToFloat,
    /// `def0 (int) <- bits of use0 (float)`. This is `pack.float`'s payload.
    FloatToBits,
    /// `def0 (float) <- use0 (int) as f64`. A real conversion.
    SiToFp,

    // --- guards: branch out-of-line on failure, fall through on success -----
    /// Exit unless `use0 cc use1`.
    GuardCmp {
        cc: Cc,
        exit: ExitId,
    },
    /// Exit unless `use0 cc imm`.
    GuardCmpImm {
        cc: Cc,
        imm: i64,
        exit: ExitId,
    },
    /// Exit unless `use0 != 0`.
    GuardNz {
        exit: ExitId,
    },

    // --- terminators --------------------------------------------------------
    Jump(MBlock),
    /// `use0 != 0 ? then_ : else_`.
    BrNz {
        then_: MBlock,
        else_: MBlock,
    },
    /// `use0 cc use1 ? then_ : else_`. A comparison fused into its sole-consuming
    /// branch: the boolean is never materialized into a register, it flows through
    /// the flags straight into the branch. isel emits this in place of an
    /// `ICmpSet` + `BrNz` pair when the compare feeds nothing but this branch.
    BrCmp {
        cc: Cc,
        then_: MBlock,
        else_: MBlock,
    },
    /// Return to the executor. The results have already been stored to the Lua
    /// stack by preceding `Store`s; this just reports how many.
    Ret {
        nret: u8,
    },
    /// Unconditional deopt — a path we declined to compile.
    ExitTo(ExitId),
}

impl MOp {
    pub fn is_terminator(self) -> bool {
        matches!(
            self,
            MOp::Jump(_) | MOp::BrNz { .. } | MOp::BrCmp { .. } | MOp::Ret { .. } | MOp::ExitTo(_)
        )
    }

    /// The blocks this instruction may transfer control to *within* the function.
    /// Exit stubs are not blocks: they leave.
    pub fn targets(self) -> Vec<MBlock> {
        match self {
            MOp::Jump(b) => vec![b],
            MOp::BrNz { then_, else_ } | MOp::BrCmp { then_, else_, .. } => vec![then_, else_],
            _ => vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub struct MInst {
    pub op: MOp,
    pub defs: Vec<Operand>,
    /// Operands, *plus* — on a guard — every register the exit stub will need to
    /// write back, the frame base among them. That is not bookkeeping: it is what
    /// keeps those values alive through register allocation, and it is how the
    /// stub finds them afterwards. A value the interpreter needs after a deopt but
    /// that nothing on the fast path reads would otherwise die at its last
    /// fast-path use, and the stub would read a register holding something else.
    ///
    /// The real operands come first and carry a `Reg` constraint; the guard
    /// keepalives follow and are `Any`, so a spilled one stays on the stack for the
    /// stub to read rather than wasting a register.
    pub uses: Vec<Operand>,
    /// Scratch registers the encoder needs to realize this instruction — a wide
    /// immediate's holder, a macro-op's working registers. Filled in after isel by
    /// the target's `annotate` pass, since how many are needed is an encoding fact.
    pub temps: Vec<Operand>,
    /// Physical registers this instruction destroys without defining — `idiv`'s
    /// `rdx`, a macro-op's internal scratch. The allocator keeps live values out of
    /// them across the instruction. Filled in by `annotate`.
    pub clobbers: Vec<PReg>,
}

impl MInst {
    /// A real machine instruction: every operand takes a register (aarch64 is
    /// load/store, so a spilled one is reloaded by the allocator before the use).
    /// A target with two-address instructions or fixed registers builds its
    /// operands with [`Operand::fixed`]/[`Operand::reuse`] instead.
    pub fn new(op: MOp, defs: Vec<VReg>, uses: Vec<VReg>) -> Self {
        MInst {
            op,
            defs: defs.into_iter().map(Operand::reg).collect(),
            uses: uses.into_iter().map(Operand::reg).collect(),
            temps: Vec::new(),
            clobbers: Vec::new(),
        }
    }

    /// A guard: real operands take registers, but the trailing exit-stub keepalives
    /// are `Any` — a spilled keepalive is read from its slot by the stub.
    pub fn guard(op: MOp, uses: Vec<VReg>, keepalives: Vec<VReg>) -> Self {
        let mut m = MInst::new(op, vec![], uses);
        m.uses.extend(keepalives.into_iter().map(Operand::any));
        m
    }

    /// The virtual register a use names — the common thing to want, since aarch64
    /// never constrains one.
    pub fn use_vreg(&self, k: usize) -> VReg {
        self.uses[k].vreg
    }

    pub fn def_vreg(&self, k: usize) -> VReg {
        self.defs[k].vreg
    }
}

#[derive(Clone, Debug, Default)]
pub struct MBlockData {
    pub insts: Vec<usize>,
}

/// How one Lua register is reconstructed on the way out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExitSrc {
    /// A boxed value: payload in a general register, tag as given.
    Boxed { payload: VReg, tag: Tag },
    /// An unboxed integer: store the payload and write an `Integer` tag.
    Int(VReg),
    /// An unboxed float: move the bits out of the float register, then store with
    /// a `Float` tag.
    Float(VReg),
    /// A constant: both halves are immediates, nothing needs to stay live.
    Const { payload: u64, tag: ValueKind },
}

/// A deopt landing pad. Materializes the interpreter's view of the frame into
/// the Lua stack and returns.
///
/// This *is* the location map — as code rather than as a side table. After
/// register allocation the encoder knows where each `VReg` landed, so the stub
/// becomes a straight run of stores.
#[derive(Clone, Debug)]
pub struct ExitStub {
    /// Bytecode pc the interpreter resumes at.
    pub pc: u32,
    /// `(lua register, source)`, one per live register in the frame state.
    pub slots: Vec<(u8, ExitSrc)>,
    /// The guard that branches here. Exactly one does — the frontend mints a fresh
    /// `Exit` per guard — and naming it is what lets the stub ask *where a value
    /// lives at that guard*, which is the only question a register allocator can
    /// answer. A stub is not an instruction and has no operands of its own; every
    /// value it reads is a use of this instruction, which is why they are there.
    pub inst: Inst,
}

pub struct MFunc {
    pub blocks: Vec<MBlockData>,
    pub insts: Vec<MInst>,
    pub classes: Vec<RegClass>,
    pub entry: MBlock,
    /// The register holding the Lua frame base, defined by `EntryArg(1)`.
    ///
    /// Named here because the exit stubs address the Lua stack through it. It is
    /// kept alive to each stub the same way every other value the stub reads is:
    /// by being a use of the guard that branches there. It gets no special
    /// treatment in the allocator, and must not — a value that the allocator is
    /// told is live everywhere is a value it can never reuse the register of.
    pub frame_base: VReg,
    pub exits: Vec<ExitStub>,
    /// Shapes each `assume.no_mm` depends on. No code is emitted for those, but
    /// the compiled artifact must record the dependency so a metatable write can
    /// invalidate it.
    pub watchpoints: Vec<ShapeRef>,
    /// Lua registers this region reads or writes on the stack directly (pinned
    /// by an open upvalue). The prologue does not cache these.
    pub pinned_regs: Vec<u8>,
    /// Highest Lua register the region touches; the deopt stubs and the entry
    /// sequence bound their stack traffic by this.
    pub max_lua_reg: u8,
    /// Soft register preferences the target's `annotate` pass fills in — an entry
    /// argument would rather stay in the register it arrived in, eliding the copy
    /// off it. The allocator honours one when it fits and ignores it otherwise.
    pub phys_hints: HashMap<VReg, PReg, RandomState>,
    /// Values the target's `annotate` pass found rematerializable: defined once, by
    /// a pure constant that depends on no register, so a spill reload can replay the
    /// defining instruction instead of touching a slot. Maps the value to that
    /// instruction. The allocator reads it through [`RegallocFunc::remat`].
    pub remat: HashMap<VReg, Inst, RandomState>,
}

impl MFunc {
    pub fn new() -> Self {
        MFunc {
            blocks: Vec::new(),
            insts: Vec::new(),
            classes: Vec::new(),
            entry: MBlock(0),
            frame_base: VReg(0),
            exits: Vec::new(),
            watchpoints: Vec::new(),
            pinned_regs: Vec::new(),
            max_lua_reg: 0,
            phys_hints: HashMap::default(),
            remat: HashMap::default(),
        }
    }

    pub fn new_vreg(&mut self, class: RegClass) -> VReg {
        let v = VReg(self.classes.len() as u32);
        self.classes.push(class);
        v
    }

    pub fn new_block(&mut self) -> MBlock {
        let b = MBlock(self.blocks.len() as u32);
        self.blocks.push(MBlockData::default());
        b
    }

    pub fn push(&mut self, b: MBlock, inst: MInst) -> Inst {
        let i = self.insts.len();
        self.insts.push(inst);
        self.blocks[b.0 as usize].insts.push(i);
        i
    }

    pub fn block(&self, b: MBlock) -> &MBlockData {
        &self.blocks[b.0 as usize]
    }

    pub fn inst(&self, i: Inst) -> &MInst {
        &self.insts[i]
    }
}

/// The machine IR, as the register allocator sees it: a CFG, and the values each
/// instruction reads and writes. Nothing above this line mentions aarch64, and
/// nothing below it mentions a `MOp`.
///
/// `block_order` is inherited, deliberately. The encoder lays blocks out in the
/// order this returns and the allocator numbers live intervals in it; two
/// implementations that agree today would be a bug waiting for a CFG shape nobody
/// has written yet.
impl RegallocFunc for MFunc {
    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry(&self) -> MBlock {
        self.entry
    }

    fn block_insts(&self, b: MBlock) -> &[Inst] {
        &self.block(b).insts
    }

    /// The successors of a block: whatever its terminator names.
    ///
    /// Exit stubs are deliberately absent. A guard's branch to its stub is not an
    /// edge — the stub does not rejoin, it leaves — and the values the stub needs
    /// are already on the guard's `uses` list, so liveness picks them up there.
    fn succs(&self, b: MBlock) -> Vec<MBlock> {
        match self.block(b).insts.last() {
            Some(&i) => self.inst(i).op.targets(),
            None => vec![],
        }
    }

    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn defs(&self, i: Inst) -> &[Operand] {
        &self.insts[i].defs
    }

    fn uses(&self, i: Inst) -> &[Operand] {
        &self.insts[i].uses
    }

    fn num_vregs(&self) -> usize {
        self.classes.len()
    }

    fn class(&self, v: VReg) -> RegClass {
        self.classes[v.0 as usize]
    }

    /// A `Mov` is the only pure copy the machine IR has: `def0 <- use0`, same
    /// class. Block-parameter resolution emits these by the dozen on edges, and
    /// coalescing them is the whole point of telling the allocator they exist.
    fn is_copy(&self, i: Inst) -> Option<(usize, usize)> {
        matches!(self.insts[i].op, MOp::Mov).then_some((0, 0))
    }

    fn temps(&self, i: Inst) -> &[Operand] {
        &self.insts[i].temps
    }

    fn clobbers(&self, i: Inst) -> &[PReg] {
        &self.insts[i].clobbers
    }

    fn phys_hint(&self, v: VReg) -> Option<PReg> {
        self.phys_hints.get(&v).copied()
    }

    fn remat(&self, v: VReg) -> Option<Inst> {
        self.remat.get(&v).copied()
    }
}

impl Default for MFunc {
    fn default() -> Self {
        Self::new()
    }
}

pub fn print_mfunc(f: &MFunc) -> String {
    let mut s = String::new();
    for (bi, blk) in f.blocks.iter().enumerate() {
        let _ = writeln!(s, "mb{bi}:");
        for &i in &blk.insts {
            let inst = &f.insts[i];
            let _ = write!(s, "    ");
            if !inst.defs.is_empty() {
                let ds: Vec<String> = inst.defs.iter().map(|d| format!("r{}", d.vreg.0)).collect();
                let _ = write!(s, "{} = ", ds.join(", "));
            }
            let _ = write!(s, "{}", fmt_op(inst.op));
            if !inst.uses.is_empty() {
                let us: Vec<String> = inst.uses.iter().map(|u| format!("r{}", u.vreg.0)).collect();
                let _ = write!(s, " {}", us.join(", "));
            }
            let _ = writeln!(s);
        }
    }
    for (ei, stub) in f.exits.iter().enumerate() {
        let _ = writeln!(s, "exit{ei}: -> pc {}", stub.pc);
        for (r, src) in &stub.slots {
            let _ = writeln!(s, "    R{r} <- {}", fmt_exit_src(*src));
        }
    }
    s
}

fn fmt_op(op: MOp) -> String {
    match op {
        MOp::EntryArg(n) => format!("entryarg {n}"),
        MOp::Imm(v) => format!("imm {v}"),
        MOp::ShapeAddr(s) => format!("shapeaddr S{}", s.0),
        MOp::ConstPayload(c) => format!("constpayload K{}", c.0),
        MOp::Mov => "mov".into(),
        MOp::Load { off, width } => format!("load.{} [{off}]", fmt_width(width)),
        MOp::Store { off, width } => format!("store.{} [{off}]", fmt_width(width)),
        MOp::Alu(o) => format!("{o:?}").to_lowercase(),
        MOp::AluImm(o, i) => format!("{} #{i}", format!("{o:?}").to_lowercase()),
        MOp::FAlu(o) => format!("f{}", format!("{o:?}").to_lowercase()),
        MOp::ICmpSet(cc) => format!("icmpset.{}", fmt_cc(cc)),
        MOp::FCmpSet(cc) => format!("fcmpset.{}", fmt_cc(cc)),
        MOp::BitsToFloat => "bits->f64".into(),
        MOp::FloatToBits => "f64->bits".into(),
        MOp::SiToFp => "sitofp".into(),
        MOp::GuardCmp { cc, exit } => format!("guard.{} -> exit{}", fmt_cc(cc), exit.0),
        MOp::GuardCmpImm { cc, imm, exit } => {
            format!("guard.{} #{imm} -> exit{}", fmt_cc(cc), exit.0)
        }
        MOp::GuardNz { exit } => format!("guard.nz -> exit{}", exit.0),
        MOp::Jump(b) => format!("jump mb{}", b.0),
        MOp::BrNz { then_, else_ } => format!("brnz mb{}, mb{}", then_.0, else_.0),
        MOp::BrCmp { cc, then_, else_ } => {
            format!("brcmp.{} mb{}, mb{}", fmt_cc(cc), then_.0, else_.0)
        }
        MOp::Ret { nret } => format!("ret {nret}"),
        MOp::ExitTo(e) => format!("deopt -> exit{}", e.0),
    }
}

fn fmt_width(w: Width) -> &'static str {
    match w {
        Width::U8 => "u8",
        Width::U64 => "u64",
    }
}

fn fmt_cc(cc: Cc) -> &'static str {
    match cc {
        Cc::Eq => "eq",
        Cc::Ne => "ne",
        Cc::Lt => "lt",
        Cc::Le => "le",
        Cc::Gt => "gt",
        Cc::Ge => "ge",
    }
}

fn fmt_exit_src(src: ExitSrc) -> String {
    match src {
        ExitSrc::Boxed {
            payload,
            tag: Tag::Const(k),
        } => format!("r{}:{k:?}", payload.0),
        ExitSrc::Boxed {
            payload,
            tag: Tag::Dyn(t),
        } => format!("r{}:r{}", payload.0, t.0),
        ExitSrc::Int(v) => format!("int r{}", v.0),
        ExitSrc::Float(v) => format!("float r{}", v.0),
        ExitSrc::Const { payload, tag } => format!("const {payload:#x}:{tag:?}"),
    }
}

impl fmt::Display for MFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&print_mfunc(self))
    }
}
