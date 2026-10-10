//! Allocated `MInst`s to machine code (10.5, 10.6): the native frame, the
//! allocator's moves, every instruction's sequence, the exit stubs and the
//! region's exit trampoline in the cold section.

use regalloc2::{Allocation, Edit, Output, PReg, RegClass};

use crate::env::MtCache;
use crate::env::table::SlotLoc;
use crate::env::value::Value;
use crate::jit::backend::aarch64::abi::{
    BASE, CLOSURE, D31, IMAGE_SPILLS, INSN, MAX_SPILLS, PC, RT, THREAD, X16, X17, image_index,
};
use crate::jit::backend::aarch64::asm::{
    Asm, Cond, Extend, FP, Fpr, Gpr, LR, Label, SP, Shift, Sz, ZR,
};
use crate::jit::backend::aarch64::inst::{AluOp, ENTRY_FAIL, FOp, FUnOp, MInst, Test};
use crate::jit::backend::aarch64::lower::{LExit, Lowered, Src};
use crate::jit::backend::vcode::VCode;
use crate::jit::ir::ops::Ctx;
use crate::jit::ir::types::TypeSet;
use crate::jit::layout;
use crate::jit::region::{Loc, SnapEntry};

const NIL: u64 = 0xFFFF_FFFE_0000_0000;
const FALSE: u64 = NIL | 1;
const BOX: u64 = 0xFFF9_0000_0000_0000;
const PTR_MASK: u64 = (1 << 48) - 1;
/// The machine code of a region, before its address is known.
pub(crate) struct Emitted {
    pub(crate) asm: Asm,
    pub(crate) frame_size: u32,
    pub(crate) num_spills: u32,
    /// The 8-byte word holding the region pointer.
    pub(crate) region_word: Label,
    pub(crate) exits: Vec<ExitOut>,
}

pub(crate) struct ExitOut {
    pub(crate) pc: u32,
    pub(crate) kind: crate::jit::ir::ExitKind,
    pub(crate) tag: crate::jit::ir::ops::ExitTag,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
}

struct Em<'a> {
    a: Asm,
    v: &'a VCode<MInst>,
    out: &'a Output,
    frame: u32,
    labels: Vec<Label>,
    exit_labels: Vec<Label>,
    exit_region: Label,
    entry_fail: Label,
    helpers: Helpers,
    /// The `TCVM_JIT_DEOPT_ALL` flags of the exits, by address.
    fired: Option<usize>,
}

/// Addresses of the Rust routines a region branches to.
#[derive(Clone, Copy)]
pub(crate) struct Helpers {
    pub(crate) exit_common: usize,
    pub(crate) entry_fail: usize,
    pub(crate) enter: usize,
    pub(crate) box_i64: usize,
    pub(crate) land: usize,
}

fn gpr(p: PReg) -> Gpr {
    debug_assert_eq!(p.class(), RegClass::Int);
    Gpr(p.hw_enc() as u8)
}

fn fpr(p: PReg) -> Fpr {
    debug_assert_eq!(p.class(), RegClass::Float);
    Fpr(p.hw_enc() as u8)
}

fn spill_off(slot: usize) -> i32 {
    16 + 8 * slot as i32
}

pub(crate) fn emit(
    l: &Lowered,
    out: &Output,
    helpers: Helpers,
    fired: Option<usize>,
) -> Result<Emitted, String> {
    if out.num_spillslots > MAX_SPILLS {
        return Err(format!("{} spill slots", out.num_spillslots));
    }
    let frame = (16 + 8 * out.num_spillslots as u32).next_multiple_of(16);
    let mut a = Asm::new();
    let labels: Vec<Label> = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let exit_labels: Vec<Label> = (0..l.exits.len()).map(|_| a.new_label()).collect();
    let exit_region = a.new_label();
    let entry_fail = a.new_label();
    let mut em = Em {
        a,
        v: &l.vcode,
        out,
        frame,
        labels,
        exit_labels,
        exit_region,
        entry_fail,
        helpers,
        fired,
    };
    let nb = l.vcode.blocks.len();
    for b in 0..nb {
        let blk = &l.vcode.blocks[b];
        if blk.resume {
            em.a.align(32);
        }
        em.a.bind(em.labels[b]);
        for ie in out.block_insts_and_edits(&l.vcode, regalloc2::Block::new(b)) {
            match ie {
                regalloc2::InstOrEdit::Edit(Edit::Move { from, to }) => em.mov(*from, *to),
                regalloc2::InstOrEdit::Inst(i) => {
                    let next = (b + 1 < nb).then(|| em.labels[b + 1]);
                    em.inst(i.index(), b, next)?;
                }
            }
        }
    }
    // Cold section: one stub per exit, the trampoline, and the prologue's
    // entry-fail stub, which needs no register image (6.4).
    for (k, _) in l.exits.iter().enumerate() {
        em.a.bind(em.exit_labels[k]);
        em.a.movz(Sz::W, X16, k as u16, 0);
        em.a.b(em.exit_region);
    }
    em.a.align(8);
    let region_word = em.a.new_label();
    em.a.bind(em.exit_region);
    em.a.ldr_label(X17, region_word);
    em.a.orr_shift(Sz::X, X16, X17, X16, Shift::Lsl, 48);
    em.a.b_far(em.helpers.exit_common);
    if !l.entry_regs.is_empty() {
        em.a.bind(em.entry_fail);
        em.close_frame();
        em.a.ldr_label(INSN, region_word);
        em.a.b_far(em.helpers.entry_fail);
    }
    em.a.align(8);
    em.a.bind(region_word);
    em.a.emit(0);
    em.a.emit(0);
    let exits = l
        .exits
        .iter()
        .enumerate()
        .map(|(k, e)| exit_out(l, out, e, k))
        .collect();
    Ok(Emitted {
        asm: em.a,
        frame_size: frame,
        num_spills: out.num_spillslots as u32,
        region_word,
        exits,
    })
}

/// The runtime record of an exit: each entry's location after allocation.
fn exit_out(l: &Lowered, out: &Output, e: &LExit, _k: usize) -> ExitOut {
    let mut consts = Vec::new();
    let mut entries = Vec::new();
    let inst = e.inst.expect("an exit without an instruction");
    let allocs = out.inst_allocs(regalloc2::Inst::new(inst));
    let base = l.snap_ops[inst] as usize;
    let (nil, f, t) = (
        Value::nil().to_raw(),
        Value::boolean(false).to_raw(),
        Value::boolean(true).to_raw(),
    );
    for &(reg, rep, src) in &e.entries {
        let loc = match src {
            Src::Const(w) if w == nil => Loc::Nil,
            Src::Const(w) if w == f => Loc::False,
            Src::Const(w) if w == t => Loc::True,
            Src::Const(w) => {
                consts.push(w);
                Loc::Const(consts.len() as u32 - 1)
            }
            Src::Operand(k) => {
                let al = allocs[base + k as usize];
                if let Some(p) = al.as_reg() {
                    Loc::Reg(image_index(p))
                } else if let Some(s) = al.as_stack() {
                    Loc::Spill(s.index() as u16)
                } else {
                    panic!("snapshot operand without an allocation");
                }
            }
        };
        entries.push(SnapEntry { reg, rep, loc });
    }
    ExitOut {
        pc: e.pc,
        kind: e.kind,
        tag: e.tag,
        entries,
        consts,
    }
}

impl Em<'_> {
    fn allocs(&self, i: usize) -> &[Allocation] {
        self.out.inst_allocs(regalloc2::Inst::new(i))
    }

    fn r(&self, i: usize, k: usize) -> Gpr {
        gpr(self.allocs(i)[k].as_reg().expect("a register operand"))
    }

    fn fr(&self, i: usize, k: usize) -> Fpr {
        fpr(self.allocs(i)[k].as_reg().expect("a register operand"))
    }

    fn mov(&mut self, from: Allocation, to: Allocation) {
        match (from.as_reg(), to.as_reg(), from.as_stack(), to.as_stack()) {
            (Some(f), Some(t), _, _) => match f.class() {
                RegClass::Int => self.a.mov(gpr(t), gpr(f)),
                _ => self.a.fmov(fpr(t), fpr(f)),
            },
            (Some(f), None, _, Some(s)) => match f.class() {
                RegClass::Int => self.a.str(gpr(f), FP, spill_off(s.index())),
                _ => self.a.str_d(fpr(f), FP, spill_off(s.index())),
            },
            (None, Some(t), Some(s), _) => match t.class() {
                RegClass::Int => self.a.ldr(gpr(t), FP, spill_off(s.index())),
                _ => self.a.ldr_d(fpr(t), FP, spill_off(s.index())),
            },
            _ => panic!("stack-to-stack move"),
        }
    }

    fn open_frame(&mut self) {
        self.a.sub_imm(Sz::X, SP, SP, self.frame as u64);
        self.a.stp(FP, LR, SP, 0);
        self.a.mov_sp(FP, SP);
    }

    fn close_frame(&mut self) {
        self.a.ldp(FP, LR, SP, 0);
        self.a.add_imm(Sz::X, SP, SP, self.frame as u64);
    }

    /// Set the flags for `test` on instruction `i`'s operands from `k`;
    /// returns the condition that means the test holds, given `cond` (the
    /// IR's condition for compares, ignored for type tests).
    fn test(&mut self, i: usize, k: usize, test: Test, cond: Cond) -> Cond {
        match test {
            Test::I32 { imm: Some(n) } => {
                let x = self.r(i, k);
                if n < 0 {
                    self.a.cmn_imm(Sz::W, x, (-n) as u64);
                } else {
                    self.a.cmp_imm(Sz::W, x, n as u64);
                }
                cond
            }
            Test::I32 { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::W, x, y);
                cond
            }
            Test::X { imm: Some(n) } => {
                let x = self.r(i, k);
                if n < 0 {
                    self.a.cmn_imm(Sz::X, x, (-n) as u64);
                } else {
                    self.a.cmp_imm(Sz::X, x, n as u64);
                }
                cond
            }
            Test::X { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::X, x, y);
                cond
            }
            Test::F64 => {
                let (x, y) = (self.fr(i, k), self.fr(i, k + 1));
                self.a.fcmp(x, y);
                cond
            }
            // `cond` is `Lo` for falsy, `Hs` for truthy.
            Test::Falsy => {
                let v = self.r(i, k);
                self.a.mov_imm(X16, NIL);
                self.a.sub(Sz::X, X16, v, X16);
                self.a.cmp_imm(Sz::X, X16, 2);
                cond
            }
            Test::B1 => {
                let c = self.r(i, k);
                self.a.cmp_imm(Sz::W, c, 0);
                cond
            }
            Test::Shape(word) => {
                let t = self.r(i, k);
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_lit(X17, word);
                self.a.cmp(Sz::X, X16, X17);
                cond
            }
            // `cond` is `Eq` for in the set, `Ne` for not.
            Test::Type(set) => {
                let v = self.r(i, k);
                let c = self.type_flags(v, set);
                if cond == Cond::Eq { c } else { c.invert() }
            }
        }
    }

    /// Flags for a single-test type set; the condition meaning "in the set".
    fn type_flags(&mut self, v: Gpr, set: TypeSet) -> Cond {
        if set == TypeSet::SMALL {
            self.a.lsr_imm(Sz::X, X16, v, 32);
            self.a.cmn_imm(Sz::W, X16, 1);
            Cond::Eq
        } else if set == TypeSet::FLOAT {
            self.a.mov_imm(X16, BOX);
            self.a.cmp(Sz::X, v, X16);
            Cond::Lo
        } else if set == TypeSet::SMALL | TypeSet::FLOAT {
            // Below `BOX` (a float), else the small-int high word.
            self.a.mov_imm(X16, BOX);
            self.a.cmp(Sz::X, v, X16);
            self.a.lsr_imm(Sz::X, X17, v, 32);
            self.a.ccmn_imm(Sz::W, X17, 1, 0b0100, Cond::Hs);
            Cond::Eq
        } else if set == TypeSet::NIL {
            self.a.mov_imm(X16, NIL);
            self.a.cmp(Sz::X, v, X16);
            Cond::Eq
        } else if set == TypeSet::FALSY {
            self.a.mov_imm(X16, NIL);
            self.a.sub(Sz::X, X16, v, X16);
            self.a.cmp_imm(Sz::X, X16, 2);
            Cond::Lo
        } else if set == TypeSet::BOOL {
            self.a.mov_imm(X16, FALSE);
            self.a.sub(Sz::X, X16, v, X16);
            self.a.cmp_imm(Sz::X, X16, 2);
            Cond::Lo
        } else if let Some(tag) = single_tag(set) {
            self.a.lsr_imm(Sz::X, X16, v, 48);
            self.a.movz(Sz::W, X17, 0xfff8 | tag, 0);
            self.a.cmp(Sz::W, X16, X17);
            Cond::Eq
        } else {
            panic!("type test of {set:?} needs more than one compare")
        }
    }

    /// Branch to `ok` when `v` is in `set`, falling through otherwise.
    fn type_branch(&mut self, v: Gpr, set: TypeSet, ok: Label) {
        let mut rest = set;
        let parts = [
            TypeSet::SMALL,
            TypeSet::FLOAT,
            TypeSet::NIL,
            TypeSet::BOOL,
            TypeSet::BIGINT,
            TypeSet::STR,
            TypeSet::TAB,
            TypeSet::FUN,
            TypeSet::THR,
            TypeSet::UDATA,
        ];
        for p in parts {
            if rest.contains(p) {
                let c = self.type_flags(v, p);
                self.a.b_cond(c, ok);
                rest.remove(p);
            }
        }
        for p in [TypeSet::FALSE, TypeSet::TRUE] {
            if rest.contains(p) {
                self.a.mov_imm(
                    X16,
                    if p == TypeSet::FALSE {
                        FALSE
                    } else {
                        FALSE + 1
                    },
                );
                self.a.cmp(Sz::X, v, X16);
                self.a.b_cond(Cond::Eq, ok);
            }
        }
    }

    fn exit(&mut self, exit: u32) -> Label {
        if exit == ENTRY_FAIL {
            return self.entry_fail;
        }
        self.exit_labels[exit as usize]
    }

    fn inst(&mut self, i: usize, block: usize, next: Option<Label>) -> Result<(), String> {
        let inst = self.v.insts[i].clone();
        if let Some(fired) = self.fired
            && let Some(k) = inst.exit_id()
        {
            // An exit that never fired fires now.
            let l = self.exit(k);
            self.a.ldr_lit(X16, (fired + 4 * k as usize) as u64);
            self.a.ldr_w(X16, X16, 0);
            self.a.cbz(Sz::W, X16, l);
        }
        use MInst::*;
        match inst {
            Prologue => self.open_frame(),
            MovImm(v) => {
                let d = self.r(i, 0);
                self.a.mov_imm(d, v);
            }
            FImm(bits) => {
                let d = self.fr(i, 0);
                if bits == 0 {
                    self.a.fmov_from_gpr(d, ZR);
                } else if let Some(imm8) = Asm::fmov_imm8(f64::from_bits(bits)) {
                    self.a.fmov_imm(d, imm8);
                } else {
                    self.a.ldr_lit_f(d, bits);
                }
            }
            LoadSlot(r) => {
                let d = self.r(i, 0);
                self.a.ldr(d, BASE, r as i32 * 8);
            }
            StoreSlot(r) => {
                let v = self.r(i, 0);
                self.a.str(v, BASE, r as i32 * 8);
            }
            StoreSlotF(r) => {
                let v = self.fr(i, 0);
                self.a.str_d(v, BASE, r as i32 * 8);
            }
            LoadUpval(k) => {
                let d = self.r(i, 0);
                let off = crate::jit::compile::upvalues_offset() + k as usize * 8;
                self.a.ldr(d, CLOSURE, off as i32);
            }
            Alu(op, sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match op {
                    AluOp::Add => self.a.add(sz, d, n, m),
                    AluOp::Sub => self.a.sub(sz, d, n, m),
                    AluOp::And => self.a.and(sz, d, n, m),
                    AluOp::Orr => self.a.orr(sz, d, n, m),
                    AluOp::Eor => self.a.eor(sz, d, n, m),
                    AluOp::Mul => self.a.mul(sz, d, n, m),
                    AluOp::Udiv => self.a.udiv(sz, d, n, m),
                }
            }
            AluImm(op, sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                match op {
                    AluOp::Add => self.a.add_imm(sz, d, n, k),
                    AluOp::Sub => self.a.sub_imm(sz, d, n, k),
                    AluOp::And => self.a.and_imm(sz, d, n, k),
                    AluOp::Orr => self.a.orr_imm(sz, d, n, k),
                    AluOp::Eor => self.a.eor_imm(sz, d, n, k),
                    _ => return Err(format!("{op:?} with an immediate")),
                }
            }
            Neg(sz) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.neg(sz, d, n);
            }
            Mvn(sz) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.mvn(sz, d, n);
            }
            AddOvf { sub, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                if sub {
                    self.a.subs(Sz::W, d, n, m);
                } else {
                    self.a.adds(Sz::W, d, n, m);
                }
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            AddImmOvf { sub, imm, exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                if sub {
                    self.a.subs_imm(Sz::W, d, n, imm as u64);
                } else {
                    self.a.adds_imm(Sz::W, d, n, imm as u64);
                }
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            MulOvf { exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.smull(X16, n, m);
                self.a.cmp_ext(X16, X16, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov_w(d, X16);
            }
            NegOvf { exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.negs(Sz::W, d, n);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            DivMod {
                div,
                sz,
                nonzero,
                exit,
            } => self.divmod(i, div, sz, nonzero, exit),
            ShiftI32 { left, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.sxtw(X16, n);
                self.a.sxtw(X17, m);
                self.lua_shift(X16, X16, X17, left);
                self.a.cmp_ext(X16, X16, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov_w(d, X16);
            }
            ShiftI64 { left } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.lua_shift(X16, n, m, left);
                self.a.mov(d, X16);
            }
            Sxtw => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.sxtw(d, n);
            }
            Fcvtzs(sz) => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fcvtzs(sz, d, n);
            }
            Scvtf(sz) => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                self.a.scvtf(sz, d, n);
            }
            FmovToGpr => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fmov_to_gpr(d, n);
            }
            FmovFromGpr => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                self.a.fmov_from_gpr(d, n);
            }
            LToI { exit } => {
                let n = self.r(i, 0);
                self.a.cmp_ext(n, n, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
            }
            FToIExact { exit } => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fcvtzs(Sz::W, X16, n);
                self.a.scvtf(Sz::W, D31, X16);
                self.a.fcmp(n, D31);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                // -0.0 converts to 0 but is a float zero only by sign: keep it a float.
                let ok = self.a.new_label();
                self.a.cmp_imm(Sz::W, X16, 0);
                self.a.b_cond(Cond::Ne, ok);
                self.a.fmov_to_gpr(X17, n);
                self.a.cmp_imm(Sz::X, X17, 0);
                self.a.b_cond(Cond::Lt, l);
                self.a.bind(ok);
                self.a.mov_w(d, X16);
            }
            ToF64 { exit } => {
                let (d, v) = (self.fr(i, 0), self.r(i, 1));
                let done = self.a.new_label();
                let not_small = self.a.new_label();
                self.a.lsr_imm(Sz::X, X16, v, 32);
                self.a.cmn_imm(Sz::W, X16, 1);
                self.a.b_cond(Cond::Ne, not_small);
                self.a.scvtf(Sz::W, d, v);
                self.a.b(done);
                self.a.bind(not_small);
                self.a.mov_imm(X16, BOX);
                self.a.cmp(Sz::X, v, X16);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Hs, l);
                self.a.fmov_from_gpr(d, v);
                self.a.bind(done);
            }
            BoxI32 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.orr_imm(Sz::X, d, n, 0xFFFF_FFFF_0000_0000);
            }
            BoxB1 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                // false + c
                self.a.mov_imm(X16, FALSE);
                self.a.add_ext(d, X16, n, Extend::Uxtw, 0);
            }
            BoxI64 { exit } => {
                // x1 holds the value; the result goes to x0.
                let done = self.a.new_label();
                let slow = self.a.new_label();
                self.a.cmp_ext(Gpr(1), Gpr(1), Extend::Sxtw);
                self.a.b_cond(Cond::Ne, slow);
                self.a.orr_imm(Sz::X, Gpr(0), Gpr(1), 0xFFFF_FFFF_0000_0000);
                self.a.b(done);
                self.a.bind(slow);
                self.a.mov(Gpr(0), RT);
                self.a.bl_far(self.helpers.box_i64);
                if exit != u32::MAX {
                    self.gc_check(exit);
                }
                self.a.bind(done);
            }
            UnboxI64 => {
                let (d, v) = (self.r(i, 0), self.r(i, 1));
                let big = self.a.new_label();
                let done = self.a.new_label();
                self.a.lsr_imm(Sz::X, X16, v, 32);
                self.a.cmn_imm(Sz::W, X16, 1);
                self.a.b_cond(Cond::Ne, big);
                self.a.sxtw(d, v);
                self.a.b(done);
                self.a.bind(big);
                self.a.and_imm(Sz::X, X16, v, PTR_MASK);
                self.a.ldr(d, X16, 0);
                self.a.bind(done);
            }
            FAlu(op) => {
                let (d, n, m) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2));
                match op {
                    FOp::Add => self.a.fadd(d, n, m),
                    FOp::Sub => self.a.fsub(d, n, m),
                    FOp::Mul => self.a.fmul(d, n, m),
                    FOp::Div => self.a.fdiv(d, n, m),
                }
            }
            FUn(op) => {
                let (d, n) = (self.fr(i, 0), self.fr(i, 1));
                match op {
                    FUnOp::Neg => self.a.fneg(d, n),
                    FUnOp::Abs => self.a.fabs(d, n),
                    FUnOp::Sqrt => self.a.fsqrt(d, n),
                    FUnOp::Floor => self.a.frintm(d, n),
                    FUnOp::Ceil => self.a.frintp(d, n),
                }
            }
            Set { test, cond } => {
                let c = self.test(i, 1, test, cond);
                let d = self.r(i, 0);
                self.a.cset(Sz::W, d, c);
            }
            Select { float } => {
                let c = self.r(i, 1);
                self.a.cmp_imm(Sz::W, c, 0);
                if float {
                    let (d, x, y) = (self.fr(i, 0), self.fr(i, 2), self.fr(i, 3));
                    self.a.fcsel(d, x, y, Cond::Ne);
                } else {
                    let (d, x, y) = (self.r(i, 0), self.r(i, 2), self.r(i, 3));
                    self.a.csel(Sz::X, d, x, y, Cond::Ne);
                }
            }
            Guard { test, cond, exit } => {
                let l = self.exit(exit);
                if let Test::Type(set) = test
                    && !is_single_test(set)
                {
                    let v = self.r(i, 0);
                    if cond == Cond::Eq {
                        let ok = self.a.new_label();
                        self.type_branch(v, set, ok);
                        self.a.b(l);
                        self.a.bind(ok);
                    } else {
                        self.type_branch(v, set, l);
                    }
                } else {
                    let c = self.test(i, 0, test, cond);
                    self.a.b_cond(c.invert(), l);
                }
            }
            GuardNoFlag { mask, exit } => {
                self.a.ldr(X16, BASE, -24);
                self.a.tst_imm(Sz::X, X16, mask as u64);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
            }
            HasFlag(mask) => {
                let d = self.r(i, 0);
                self.a.ldr(X16, BASE, -24);
                self.a.tst_imm(Sz::X, X16, mask as u64);
                self.a.cset(Sz::W, d, Cond::Ne);
            }
            FnFixedLua => {
                // A native function has the niche value of `is_vararg` there.
                let (d, f) = (self.r(i, 0), self.r(i, 1));
                self.a.ldrb(X16, f, layout::FN_IS_VARARG as i32);
                self.a.cmp_imm(Sz::W, X16, 0);
                self.a.cset(Sz::W, d, Cond::Eq);
            }
            FnFits => {
                let (d, f) = (self.r(i, 0), self.r(i, 1));
                self.a.ldrb(X16, f, layout::FN_MAX_STACK as i32);
                self.a.add_shift(Sz::X, X16, BASE, X16, Shift::Lsl, 3);
                self.a.ldr(X17, THREAD, layout::THREAD_STACK_END as i32);
                self.a.cmp(Sz::X, X16, X17);
                self.a.cset(Sz::W, d, Cond::Ls);
            }
            LoadCell => {
                let (d, c) = (self.r(i, 0), self.r(i, 1));
                self.a
                    .ldr(X16, c, crate::env::function::UpvalueCell::V_OFFSET as i32);
                self.a.ldr(d, X16, 0);
            }
            GuardShape { shape, exit } => {
                let t = self.r(i, 0);
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_lit(X17, shape);
                self.a.cmp(Sz::X, X16, X17);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
            }
            ClassBits(class) => {
                let d = self.r(i, 0);
                self.a.ldr_lit(X16, class);
                self.a.ldr_w(d, X16, MtCache::BITS_OFFSET as i32);
            }
            ClassIndex(class) => {
                let d = self.r(i, 0);
                self.a.ldr_lit(X16, class);
                self.a.ldr(d, X16, MtCache::INDEX_OFFSET as i32);
            }
            TabLoad(loc) => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let loc = SlotLoc::from_raw(loc);
                if loc.spilled() {
                    self.a.ldr(X16, t, layout::T_SPILL as i32);
                    self.a.ldr(d, X16, loc.offset() as i32);
                } else {
                    self.a.ldr(d, t, loc.offset() as i32);
                }
            }
            TabStore(loc) => {
                let (t, v) = (self.r(i, 0), self.r(i, 1));
                let loc = SlotLoc::from_raw(loc);
                if loc.spilled() {
                    self.a.ldr(X16, t, layout::T_SPILL as i32);
                    self.a.str(v, X16, loc.offset() as i32);
                } else {
                    self.a.str(v, t, loc.offset() as i32);
                }
            }
            TabHasRoom(loc) => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let slot = SlotLoc::from_raw(loc).offset() / 8;
                self.a.ldr_w(X16, t, layout::T_SPILL_CAP as i32);
                self.a.mov_imm(X17, slot as u64);
                self.a.cmp(Sz::W, X16, X17);
                self.a.cset(Sz::W, d, Cond::Hi);
            }
            TabSetShape(shape) => {
                let t = self.r(i, 0);
                self.a.ldr_lit(X16, shape);
                self.a.str(X16, t, layout::T_SHAPE as i32);
            }
            TabASize => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.ldr_w(d, t, layout::T_ASIZE as i32);
            }
            TabArray => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.ldr(d, t, layout::T_ARRAY as i32);
            }
            ArrLoad => {
                let (d, a, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.ldr_idx(d, a, k, Extend::Uxtw, true);
            }
            ArrStore => {
                let (a, k, v) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.str_idx(v, a, k, Extend::Uxtw, true);
            }
            ArrNext => {
                let (d, t, from, v) = (self.r(i, 0), self.r(i, 1), self.r(i, 2), self.r(i, 3));
                let (top, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr_w(X16, t, layout::T_ASIZE as i32);
                self.a.ldr(X17, t, layout::T_ARRAY as i32);
                self.a.mov_w(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::W, d, X16);
                self.a.b_cond(Cond::Hs, done);
                self.a.ldr_idx(v, X17, d, Extend::Uxtw, true);
                // Nil is one run of ones, a logical immediate.
                self.a.eor_imm(Sz::X, v, v, NIL);
                self.a.cbnz(Sz::X, v, done);
                self.a.add_imm(Sz::W, d, d, 1);
                self.a.b(top);
                self.a.bind(done);
            }
            Barrier => {
                let skip = self.a.new_label();
                self.a.ldr(X16, Gpr(1), layout::T_HEADER);
                self.a.tbnz(X16, 0, skip);
                self.a.mov(Gpr(0), RT);
                self.a
                    .bl_far(crate::jit::helpers::jit_barrier as *const () as usize);
                self.a.bind(skip);
            }
            NumVarargs => {
                let d = self.r(i, 0);
                self.a.ldr(X16, BASE, -32);
                self.a.lsr_imm(Sz::X, d, X16, 48);
            }
            VarargAt(k) => {
                // The extras end at the header: extra k is at
                // base - 32 - 8 * nv + 8 * k, read only when k < nv.
                let (d, nv) = (self.r(i, 0), self.r(i, 1));
                self.a.cmp_imm(Sz::W, nv, k as u64);
                self.a.lsl_imm(Sz::W, X16, nv, 3);
                self.a.sub(Sz::X, X16, BASE, X16);
                self.a.ldr(d, X16, 8 * k as i32 - 32);
                self.a.mov_imm(X17, NIL);
                self.a.csel(Sz::X, d, d, X17, Cond::Hi);
            }
            VarargIdx => {
                // The index clamped to 0 out of range, so the load stays in
                // the frame.
                let (d, nv, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.sub_imm(Sz::W, X16, k, 1);
                self.a.cmp(Sz::W, X16, nv);
                self.a.csel(Sz::W, X16, X16, ZR, Cond::Lo);
                self.a.lsl_imm(Sz::W, X17, nv, 3);
                self.a.sub(Sz::X, X17, BASE, X17);
                self.a.add_shift(Sz::X, X17, X17, X16, Shift::Lsl, 3);
                self.a.ldr(d, X17, -32);
                self.a.mov_imm(X16, NIL);
                self.a.csel(Sz::X, d, d, X16, Cond::Lo);
            }
            StoreCell => {
                let (c, v) = (self.r(i, 0), self.r(i, 1));
                self.a
                    .ldr(X16, c, crate::env::function::UpvalueCell::V_OFFSET as i32);
                self.a.str(v, X16, 0);
            }
            GcCheck { exit } => self.gc_check(exit),
            Helper(h) => {
                let sig = h.sig();
                for (k, c) in sig.ctx.iter().enumerate() {
                    match c {
                        Ctx::Rt => self.a.mov(Gpr(k as u8), RT),
                        Ctx::Thread => self.a.mov(Gpr(k as u8), THREAD),
                        Ctx::Base => self.a.mov(Gpr(k as u8), BASE),
                    }
                }
                if let Some(imm) = sig.imm {
                    self.a.mov_imm(Gpr(sig.ctx.len() as u8), imm);
                }
                self.a.bl_far(crate::jit::helpers::address(h));
            }
            Jump => {
                let t = self.v.succs(block)[0].index();
                if Some(self.labels[t]) != next {
                    self.a.b(self.labels[t]);
                }
            }
            Br { test, cond } => {
                let c = self.test(i, 0, test, cond);
                let (t, f) = (
                    self.v.succs(block)[0].index(),
                    self.v.succs(block)[1].index(),
                );
                if Some(self.labels[t]) == next {
                    self.a.b_cond(c.invert(), self.labels[f]);
                } else {
                    self.a.b_cond(c, self.labels[t]);
                    if Some(self.labels[f]) != next {
                        self.a.b(self.labels[f]);
                    }
                }
            }
            Call { a, nargs, pc_after } => {
                let resume = self.labels[self.v.succs(block)[0].index()];
                let hdr = a as i32 * 8;
                self.a.adr(X16, resume);
                self.a.str(X16, BASE, hdr + 8);
                self.a.str(BASE, BASE, hdr + 16);
                self.a.ldr_lit(X16, pc_after as u64);
                self.a.str(X16, BASE, hdr + 24);
                if nargs == u8::MAX {
                    let n = self.r(i, 0);
                    self.a.mov(INSN, n);
                } else {
                    self.a.movz(Sz::X, INSN, nargs as u16, 0);
                }
                self.close_frame();
                self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
                self.a.b_far(self.helpers.enter);
            }
            Dispatch { op, word, pc_after } => {
                self.close_frame();
                self.a.ldr_lit(INSN, word);
                self.a.ldr_lit(PC, pc_after as u64);
                self.a.ldr(X16, RT, op as i32 * 8);
                self.a.br(X16);
            }
            TailEnter { nargs } => {
                // The callee is the frame's closure from here, which frees
                // `insn` and `pc` (allocatable in the body) as scratch.
                let f = self.r(i, 0);
                self.a.mov(CLOSURE, f);
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
                self.a.ldr(X16, BASE, -32);
                self.a.and_imm(Sz::X, X16, X16, !crate::vm::frame::PTR_MASK);
                self.a.orr(Sz::X, X16, X16, CLOSURE);
                self.a.str(X16, BASE, -32);
                self.a.ldr(X16, CLOSURE, layout::FN_CODE as i32);
                self.close_frame();
                self.a.ldr(INSN, X16, 0);
                self.a.add_imm(Sz::X, PC, X16, 8);
                self.a.and_imm(Sz::X, X17, INSN, 0xff);
                self.a.ldr_idx(X16, RT, X17, Extend::Lsl, true);
                self.a.br(X16);
            }
            Resume { wanted, c, a } => {
                // The callee's header word 2 is this frame's base; its word 0
                // our closure.
                self.a.ldr(BASE, BASE, -16);
                self.a.ldr(CLOSURE, BASE, -32);
                self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
                self.open_frame();
                match wanted {
                    0 if c == 1 => {}
                    0 => {
                        // Land through the helper: (rt, thread, base, a, values, nret, c).
                        self.a.mov(Gpr(0), RT);
                        self.a.mov(Gpr(1), THREAD);
                        self.a.mov(Gpr(4), PC);
                        self.a.mov(Gpr(5), INSN);
                        self.a.mov(Gpr(2), BASE);
                        self.a.movz(Sz::X, Gpr(3), a as u16, 0);
                        self.a.movz(Sz::X, Gpr(6), c as u16, 0);
                        self.a.bl_far(self.helpers.land);
                    }
                    _ => {
                        let done = self.a.new_label();
                        self.a.mov_imm(X16, NIL);
                        self.a.mov(X17, X16);
                        self.a.cbz(Sz::X, INSN, done);
                        self.a.ldr(X16, PC, 0);
                        if wanted == 2 {
                            self.a.cmp_imm(Sz::X, INSN, 1);
                            self.a.b_cond(Cond::Eq, done);
                            self.a.ldr(X17, PC, 8);
                        }
                        self.a.bind(done);
                        let d0 = self.r(i, 0);
                        self.a.mov(d0, X16);
                        if wanted == 2 {
                            let d1 = self.r(i, 1);
                            self.a.mov(d1, X17);
                        }
                    }
                }
            }
            CallMm {
                hdr,
                nargs,
                pc_after,
            } => {
                let resume = self.labels[self.v.succs(block)[0].index()];
                let h = hdr as i32 * 8;
                for k in 0..=nargs as usize {
                    let v = self.r(i, k);
                    let at = if k == 0 { h } else { h + 24 + 8 * k as i32 };
                    self.a.str(v, BASE, at);
                }
                self.a.adr(X16, resume);
                self.a.str(X16, BASE, h + 8);
                self.a.str(BASE, BASE, h + 16);
                self.a.ldr_lit(X16, pc_after as u64);
                self.a.str(X16, BASE, h + 24);
                self.close_frame();
                self.a.movz(Sz::X, INSN, nargs as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, h as u64);
                self.a.b_far(self.helpers.enter);
            }
            GuardFits { end, exit } => {
                self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
                self.a.add_imm(Sz::X, X17, BASE, end as u64 * 8);
                self.a.cmp(Sz::X, X17, X16);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Hi, l);
            }
            Return { a, n } => {
                self.a.ldr(X16, BASE, -24);
                self.a.and_imm(Sz::X, X16, X16, !31u64);
                self.close_frame();
                self.a.movz(Sz::X, INSN, n as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, a as u64 * 8);
                self.a.br(X16);
            }
            Deopt { exit } => {
                let l = self.exit(exit);
                self.a.b(l);
            }
        }
        Ok(())
    }

    fn gc_check(&mut self, exit: u32) {
        self.a.ldr(X16, RT, crate::jit::layout::METRICS as i32);
        self.a
            .ldp(X16, X17, X16, crate::jit::layout::GC_CHECK as i32);
        self.a.cmp(Sz::X, X16, X17);
        let l = self.exit(exit);
        self.a.b_cond(Cond::Hs, l);
    }

    /// Lua's shift: `n << m` (or `>>`) on 64 bits, logical, 0 past 63.
    fn lua_shift(&mut self, d: Gpr, n: Gpr, m: Gpr, left: bool) {
        // A right shift is a left shift by `-m`.
        let cnt = X17;
        if left {
            self.a.mov(cnt, m);
        } else {
            self.a.neg(Sz::X, cnt, m);
        }
        let neg = self.a.new_label();
        let zero = self.a.new_label();
        let done = self.a.new_label();
        self.a.cmp_imm(Sz::X, cnt, 0);
        self.a.b_cond(Cond::Lt, neg);
        self.a.cmp_imm(Sz::X, cnt, 64);
        self.a.b_cond(Cond::Ge, zero);
        self.a.lslv(Sz::X, d, n, cnt);
        self.a.b(done);
        self.a.bind(neg);
        self.a.neg(Sz::X, cnt, cnt);
        self.a.cmp_imm(Sz::X, cnt, 64);
        self.a.b_cond(Cond::Ge, zero);
        self.a.lsrv(Sz::X, d, n, cnt);
        self.a.b(done);
        self.a.bind(zero);
        self.a.mov(d, ZR);
        self.a.bind(done);
    }

    /// Floor division or modulo of `n` by `m`.
    fn divmod(&mut self, i: usize, div: bool, sz: Sz, nonzero: bool, exit: u32) {
        let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
        let l = self.exit(exit);
        if !nonzero {
            self.a.cbz(sz, m, l);
        }
        if div && sz == Sz::W {
            // MIN // -1 leaves i32.
            let ok = self.a.new_label();
            self.a.cmn_imm(Sz::W, m, 1);
            self.a.b_cond(Cond::Ne, ok);
            self.a.movz(Sz::W, X16, 0x8000, 1);
            self.a.cmp(Sz::W, n, X16);
            self.a.b_cond(Cond::Eq, l);
            self.a.bind(ok);
        }
        let (q, r) = (X16, X17);
        self.a.sdiv(sz, q, n, m);
        self.a.msub(sz, r, q, m, n);
        if div {
            // q - 1 when the remainder is nonzero and the signs differ.
            let done = self.a.new_label();
            self.a.cbz(sz, r, done);
            self.a.eor(sz, r, n, m);
            self.a.tbz(r, sz.bits() - 1, done);
            self.a.sub_imm(sz, q, q, 1);
            self.a.bind(done);
            self.a.mov(d, q);
        } else {
            // r + m when the remainder is nonzero and its sign differs from m's.
            let done = self.a.new_label();
            self.a.cbz(sz, r, done);
            self.a.eor(sz, q, r, m);
            self.a.tbz(q, sz.bits() - 1, done);
            self.a.add(sz, r, r, m);
            self.a.bind(done);
            self.a.mov(d, r);
        }
    }
}

fn single_tag(set: TypeSet) -> Option<u16> {
    Some(match set {
        s if s == TypeSet::UDATA => 1,
        s if s == TypeSet::BIGINT => 2,
        s if s == TypeSet::STR => 3,
        s if s == TypeSet::TAB => 4,
        s if s == TypeSet::FUN => 5,
        s if s == TypeSet::THR => 6,
        _ => return None,
    })
}

fn is_single_test(set: TypeSet) -> bool {
    set == TypeSet::SMALL
        || set == TypeSet::FLOAT
        || set == TypeSet::SMALL | TypeSet::FLOAT
        || set == TypeSet::NIL
        || set == TypeSet::FALSY
        || set == TypeSet::BOOL
        || single_tag(set).is_some()
}

#[allow(unused)]
fn unused(_: Value<'_>) {}

#[allow(unused)]
const _: usize = IMAGE_SPILLS;
