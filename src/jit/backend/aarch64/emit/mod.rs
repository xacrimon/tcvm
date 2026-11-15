//! Allocated `MInst`s to machine code: the native frame, the allocator's
//! moves, one instruction per `MInst`, the edges' move stubs, the dispatch
//! entry, the exit stubs and the region's exit trampoline.

use crate::env::value::NIL;
use crate::jit::backend::aarch64::abi::{
    self, BASE, FRAME_SPILLS, INSN, MAX_SPILLS, PC, THREAD, X16, X17,
};
use crate::jit::backend::aarch64::asm::{
    Asm, Cond, Extend, FP, Fpr, Gpr, LR, Label, SP, Shift, Sz, ZR,
};
use crate::jit::backend::aarch64::inst::{MInst, MOp};
use crate::jit::backend::emit::{
    Emitted, ExitOut, FastEntry, Selected, SiteOut, Src, rotate_loops, thread_jumps,
};
use crate::jit::backend::ra::{Edit, InstOrEdit, Output};
use crate::jit::backend::reg::{Allocation, Block, Inst, PReg, RegClass};
use crate::jit::backend::vcode::{Remat, VCode};
use crate::jit::bug;
use crate::jit::compile::Bail;
use crate::jit::ir::ops::{Ctx, Helper};
use crate::jit::region::{Loc, SnapEntry};

/// The address of a routine the code branches to.
macro_rules! addr {
    ($f:path) => {
        $f as *const () as usize
    };
}

mod call;

struct Em<'a> {
    a: Asm,
    s: &'a Selected,
    v: &'a VCode<MInst>,
    out: &'a Output,
    frame: u32,
    labels: Vec<Label>,
    /// Where a branch to each block goes (`thread_jumps`).
    dest: Vec<Label>,
    exit_labels: Vec<Label>,
    /// The code's first word, the region pointer (`Region::of_return`).
    region_word: Label,
    /// The moves of the edges that do not fall through: where the stub is,
    /// the branch and its edge, and where the stub goes on to.
    stubs: Vec<(Label, usize, usize, Label)>,
    /// By block, for the resume blocks.
    resume: Vec<Option<call::ResumeLabels>>,
    ool: Vec<call::Ool>,
    /// The calls' sites, with their return addresses and miss paths.
    sites: Vec<call::Site>,
    rt: u64,
    /// The address of the function's bytecode.
    code: u64,
}

fn gpr(p: PReg) -> Gpr {
    debug_assert_eq!(p.class(), RegClass::Int);
    Gpr(p.hw_enc() as u8)
}

fn fpr(p: PReg) -> Fpr {
    debug_assert_eq!(p.class(), RegClass::Float);
    Fpr(p.hw_enc() as u8)
}

/// A spill slot, from the frame pointer.
fn spill_off(slot: usize) -> i32 {
    (FRAME_SPILLS + 8 * slot) as i32
}

/// The fast entry past its arity and native stack checks, for a caller that
/// proved them.
const KNOWN_ENTRY: usize = 24;

/// A fast entry's known entry, from it. Parameters are not passed in
/// registers yet, so the entry for a call that would is the same.
pub(crate) fn known_offset(regs: bool, nparams: u8) -> usize {
    let _ = (regs, nparams);
    KNOWN_ENTRY
}

/// `rt` is the runtime's address, which helpers take, and `code` that of
/// the function's bytecode.
pub(crate) fn emit(
    s: &Selected,
    out: &Output,
    rt: u64,
    code: u64,
    fast: Option<FastEntry>,
) -> Result<Emitted, Bail> {
    if out.num_spillslots > MAX_SPILLS {
        return Err(Bail::TooLarge);
    }
    let mut a = Asm::new();
    let labels: Vec<Label> = (0..s.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let exit_labels: Vec<Label> = (0..s.exits.len()).map(|_| a.new_label()).collect();
    let (dest, skip) = thread_jumps(&s.vcode, out, &labels, |e| exit_labels[e as usize]);
    let mut em = Em {
        labels,
        dest,
        exit_labels,
        region_word: a.new_label(),
        frame: ((FRAME_SPILLS + 8 * out.num_spillslots) as u32).next_multiple_of(16),
        stubs: Vec::new(),
        resume: Vec::new(),
        ool: Vec::new(),
        sites: Vec::new(),
        rt,
        code,
        s,
        v: &s.vcode,
        out,
        a,
    };
    em.a.bind(em.region_word);
    em.a.emit(0);
    em.a.emit(0);
    for b in &s.vcode.blocks {
        let r = b.resume.then(|| em.resume_labels());
        em.resume.push(r);
    }
    let kept = |b: &u32| !skip[*b as usize];
    let mut laid: Vec<u32> = s.layout.emit.iter().copied().filter(kept).collect();
    let hot = s.layout.emit[..s.layout.hot]
        .iter()
        .filter(|b| kept(b))
        .count();
    let loops = &s.layout.loops;
    let tops = rotate_loops(&s.vcode, loops, &em.labels, &em.dest, &mut laid, hot);
    let entries = fast.map(|fe| em.fast_entry(fe, laid.first() == Some(&0)));
    for (at, &b) in laid.iter().enumerate() {
        let next = laid.get(at + 1).map(|&x| em.labels[x as usize]);
        // A rotated loop's top, past the jump into its header: padding
        // there never runs.
        if tops.contains(&b) && em.a.after_jump() {
            em.a.align(16);
        }
        em.a.bind(em.labels[b as usize]);
        let resume = s.vcode.blocks[b as usize].resume;
        let insts = out.block_insts_and_edits(&s.vcode, Block::new(b as usize));
        for (k, ie) in insts.enumerate() {
            match ie {
                // A fast callee's `ret` lands at the `Resume`, past anything
                // before it.
                InstOrEdit::Edit(_) if resume && k == 0 => bug!("a move before a call's resume"),
                InstOrEdit::Edit(e) => em.edit(e),
                InstOrEdit::Inst(i) => em.inst(i.index(), b as usize, next),
            }
        }
    }
    for k in 0..em.stubs.len() {
        let (stub, i, edge, to) = em.stubs[k];
        em.a.bind(stub);
        for (_, _, e) in out.edge_edits(Inst::new(i), edge) {
            em.edit(e);
        }
        em.a.b(to);
    }
    for o in std::mem::take(&mut em.ool) {
        em.out_of_line(o);
    }
    if let (Some(fe), Some(e)) = (fast, entries) {
        em.fast_stubs(fe, e);
    }
    // Where dispatch enters: through `level_enter`, which returns to
    // `level_land`, on to the entry block.
    let dispatch = em.a.new_label();
    em.a.bind(dispatch);
    em.a.adr(X16, em.labels[0]);
    em.a.b_far(addr!(abi::level_enter));
    // One stub per exit and the region's exit trampoline, which they share.
    let exit_region = em.a.new_label();
    for k in 0..em.exit_labels.len() {
        em.a.bind(em.exit_labels[k]);
        em.a.movz(Sz::W, X16, k as u16, 0);
        em.a.b(exit_region);
    }
    em.a.align(8);
    em.a.bind(exit_region);
    em.a.ldr_label(X17, em.region_word);
    em.a.orr_shift(Sz::X, X16, X17, X16, Shift::Lsl, 48);
    em.a.b_far(addr!(abi::exit_common));
    let at = |l: Label| em.a.label_offset(l) as u32;
    let sites = std::mem::take(&mut em.sites);
    let sites = (sites.into_iter())
        .map(|(keys, miss, site)| SiteOut {
            keys: keys.iter().map(|&(l, level)| (at(l), level)).collect(),
            miss: at(miss),
            ..site
        })
        .collect();
    Ok(Emitted {
        sites,
        fast: entries.map(|e| at(e.fast)),
        entry: at(dispatch),
        num_spills: out.num_spillslots as u32,
        frame_size: em.frame,
        region_word: at(em.region_word),
        exits: (0..s.exits.len()).map(|k| exit_out(s, out, k)).collect(),
        asm: em.a,
    })
}

/// The runtime record of an exit: each entry's location after allocation.
fn exit_out(s: &Selected, out: &Output, k: usize) -> ExitOut {
    let e = &s.exits[k];
    let allocs = out.inst_allocs(&s.vcode, Inst::new(e.inst as usize));
    let mut consts = Vec::new();
    let entries = s.entries[e.at as usize..(e.at + e.len) as usize]
        .iter()
        .map(|&(reg, rep, src)| {
            let loc = match src {
                Src::Const(w) => {
                    consts.push(w);
                    Loc::Const(consts.len() as u32 - 1)
                }
                Src::Home => bug!("an exit's entry in its frame slot"),
                Src::Operand(k) => {
                    let al = allocs[(e.own + k as u32) as usize];
                    match (al.as_reg(), al.as_stack()) {
                        (Some(p), _) => Loc::Reg(abi::image_index(p)),
                        (_, Some(slot)) => Loc::Spill(slot.index() as u16),
                        _ => bug!("a frame state's value without a location"),
                    }
                }
            };
            SnapEntry { reg, rep, loc }
        })
        .collect();
    ExitOut {
        pc: e.pc,
        kind: e.kind,
        tag: e.tag,
        entries,
        consts,
    }
}

impl<'a> Em<'a> {
    fn allocs(&self, i: usize) -> &[Allocation] {
        self.out.inst_allocs(self.v, Inst::new(i))
    }

    fn r(&self, i: usize, k: usize) -> Gpr {
        match self.allocs(i)[k].as_reg() {
            Some(p) => gpr(p),
            None => bug!("operand {k} of instruction {i} is in no register"),
        }
    }

    fn fr(&self, i: usize, k: usize) -> Fpr {
        match self.allocs(i)[k].as_reg() {
            Some(p) => fpr(p),
            None => bug!("operand {k} of instruction {i} is in no register"),
        }
    }

    fn edit(&mut self, e: &Edit) {
        match *e {
            Edit::Move { from, to } => self.mov(from, to),
            Edit::Remat { to, vreg, .. } => {
                let Some(r) = to.as_reg() else {
                    bug!("a value made again into no register")
                };
                match self.v.remat[vreg.vreg()] {
                    Remat::Int(x) => self.a.mov_imm(gpr(r), x),
                    // The transport register on the way to a stack slot
                    // (`ra::resolve`): the word.
                    Remat::Float(bits) if r.class() == RegClass::Int => {
                        self.a.mov_imm(gpr(r), bits)
                    }
                    Remat::Float(bits) => self.fimm(fpr(r), bits),
                    Remat::Slot(slot) => self.a.ldr(gpr(r), BASE, slot as i32 * 8),
                    Remat::AndImm { .. } | Remat::None => bug!("{vreg:?} cannot be made again"),
                }
            }
        }
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
            _ => bug!("a stack-to-stack move"),
        }
    }

    fn fimm(&mut self, d: Fpr, bits: u64) {
        if bits == 0 {
            self.a.fmov_from_gpr(d, ZR);
        } else if let Some(imm8) = Asm::fmov_imm8(f64::from_bits(bits)) {
            self.a.fmov_imm(d, imm8);
        } else {
            self.a.ldr_lit_f(d, bits);
        }
    }

    /// A branch's targets, with the allocator's moves on its edges: the
    /// edge falling through has them in line after the branch (returned),
    /// any other in a stub that jumps on.
    fn edge_labels(
        &mut self,
        i: usize,
        block: usize,
        next: Option<Label>,
    ) -> ([Label; 2], &'a [(Inst, u8, Edit)]) {
        let succs = self.v.succs(block);
        let mut labels = [self.dest[succs[0].index()], self.dest[succs[1].index()]];
        let mut inline: &'a [(Inst, u8, Edit)] = &[];
        for (k, l) in labels.iter_mut().enumerate() {
            let edits = self.out.edge_edits(Inst::new(i), k);
            if edits.is_empty() {
                continue;
            }
            if Some(*l) == next && inline.is_empty() {
                inline = edits;
                continue;
            }
            let stub = self.a.new_label();
            self.stubs.push((stub, i, k, *l));
            *l = stub;
        }
        (labels, inline)
    }

    fn inst(&mut self, i: usize, block: usize, next: Option<Label>) {
        let m = self.v.insts[i];
        let (sz, imm) = (m.sz, m.imm);
        // A value left for the allocator to make where used has no code.
        let unmade = self.allocs(i).first().is_some_and(|a| a.is_none());
        match m.op {
            MOp::CallLua => self.call(i, block, next),
            MOp::Resume => self.on_resume(i, block),
            MOp::Entry => {
                self.a.sub_imm(Sz::X, SP, SP, self.frame as u64);
                self.a.stp(FP, LR, SP, 0);
                self.a.mov_sp(FP, SP);
            }
            MOp::MovImm | MOp::FImm | MOp::LdrSlot if unmade => {}
            MOp::MovImm => self.a.mov_imm(self.r(i, 0), imm),
            MOp::FImm => self.fimm(self.fr(i, 0), imm),
            MOp::LdrSlot => self.a.ldr(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::StrSlot => self.a.str(self.r(i, 0), BASE, imm as i32 * 8),
            MOp::Ldr if sz == Sz::W => self.a.ldr_w(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::Ldr => self.a.ldr(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::Str => self.a.str(self.r(i, 0), self.r(i, 1), imm as i32),
            MOp::LdrIdx | MOp::StrIdx => {
                let ext = if m.cc == 1 { Extend::Uxtw } else { Extend::Lsl };
                let (t, n, x) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match m.op {
                    MOp::LdrIdx => self.a.ldr_idx(t, n, x, ext, true),
                    _ => self.a.str_idx(t, n, x, ext, true),
                }
            }
            MOp::MovBase => self.a.mov(self.r(i, 0), BASE),
            MOp::Uxtw => self.a.mov_w(self.r(i, 0), self.r(i, 1)),
            MOp::Cset => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a.cset(Sz::W, self.r(i, 0), cc)
            }
            MOp::Csel => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a
                    .csel(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2), cc)
            }
            MOp::AndRR => self.a.and(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::OrrRR => self.a.orr(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::EorRR => self.a.eor(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SdivRR => self.a.sdiv(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::Sxtw => self.a.sxtw(self.r(i, 0), self.r(i, 1)),
            MOp::NegR => self.a.neg(sz, self.r(i, 0), self.r(i, 1)),
            MOp::NegsR => self.a.negs(sz, self.r(i, 0), self.r(i, 1)),
            MOp::FNeg => self.a.fneg(self.fr(i, 0), self.fr(i, 1)),
            MOp::Frintm => self.a.frintm(self.fr(i, 0), self.fr(i, 1)),
            MOp::AndImm => self.a.and_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::TstImm => self.a.tst_imm(sz, self.r(i, 0), imm),
            MOp::AddRR => self.a.add(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubRR => self.a.sub(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::MulRR => self.a.mul(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddImm => self.a.add_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::SubImm => self.a.sub_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::AddsRR => self.a.adds(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::SubsRR => self.a.subs(sz, self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::AddsImm => self.a.adds_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::SubsImm => self.a.subs_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::CmpRR => self.a.cmp(sz, self.r(i, 0), self.r(i, 1)),
            MOp::CmpImm => self.a.cmp_imm(sz, self.r(i, 0), imm),
            MOp::CmnImm => self.a.cmn_imm(sz, self.r(i, 0), imm),
            MOp::CmpSxtw => self.a.cmp_ext(self.r(i, 0), self.r(i, 1), Extend::Sxtw),
            MOp::Fcmp => self.a.fcmp(self.fr(i, 0), self.fr(i, 1)),
            MOp::LslImm => self.a.lsl_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::LsrImm => self.a.lsr_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::AsrImm => self.a.asr_imm(sz, self.r(i, 0), self.r(i, 1), imm as u32),
            MOp::Smull => self.a.smull(self.r(i, 0), self.r(i, 1), self.r(i, 2)),
            MOp::OrrImm => self.a.orr_imm(sz, self.r(i, 0), self.r(i, 1), imm),
            MOp::Scvtf => self.a.scvtf(sz, self.fr(i, 0), self.r(i, 1)),
            MOp::FmovToGpr => self.a.fmov_to_gpr(self.r(i, 0), self.fr(i, 1)),
            MOp::FmovFromGpr => self.a.fmov_from_gpr(self.fr(i, 0), self.r(i, 1)),
            MOp::FAdd => self.a.fadd(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FSub => self.a.fsub(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FMul => self.a.fmul(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::FDiv => self.a.fdiv(self.fr(i, 0), self.fr(i, 1), self.fr(i, 2)),
            MOp::B => {
                let to = self.dest[self.v.succs(block)[0].index()];
                if Some(to) != next {
                    self.a.b(to);
                }
            }
            MOp::BCond => {
                // SAFETY: selection wrote an `asm::Cond`.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                let ([t, f], inline) = self.edge_labels(i, block, next);
                // Unordered fails a float condition and holds its inverse,
                // so the inverted branch takes it to `f` too.
                if Some(f) == next {
                    self.a.b_cond(cc, t);
                } else {
                    self.a.b_cond(cc.invert(), f);
                    if Some(t) != next {
                        self.a.b(t);
                    }
                }
                for (_, _, e) in inline {
                    self.edit(e);
                }
            }
            MOp::BCondExit => {
                // SAFETY: as above.
                let cc: Cond = unsafe { std::mem::transmute(m.cc) };
                self.a.b_cond(cc, self.exit_labels[imm as usize]);
            }
            MOp::BExit => self.a.b(self.exit_labels[imm as usize]),
            MOp::Call => {
                let h = Helper::of(imm as u16);
                let info = h.info();
                for (k, c) in info.ctx.iter().enumerate() {
                    match c {
                        Ctx::Rt => self.a.mov_imm(Gpr(k as u8), self.rt),
                        Ctx::Thread => self.a.mov(Gpr(k as u8), THREAD),
                        Ctx::Base => self.a.mov(Gpr(k as u8), BASE),
                    }
                }
                if info.imm {
                    self.a.mov_imm(Gpr(info.ctx.len() as u8), imm >> 16);
                }
                self.a.bl_far(crate::jit::helpers::address(h));
            }
            MOp::Ret => {
                // To a compiled caller, or to `level_land` for a frame
                // dispatch entered; a tagged return address is a deep one's.
                let (a, n) = (imm & 0xff, imm >> 8);
                if n == 0 {
                    self.a.mov_imm(Gpr(0), NIL);
                } else {
                    self.a.ldr(Gpr(0), BASE, a as i32 * 8);
                }
                self.a.ldp(FP, LR, SP, 0);
                self.a.add_imm(Sz::X, SP, SP, self.frame as u64);
                self.a.movz(Sz::X, INSN, n as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, a * 8);
                let deep = self.a.new_label();
                self.a.tbnz(LR, 0, deep);
                self.a.ret();
                self.a.bind(deep);
                self.a.and_imm(Sz::X, LR, LR, !1u64);
                self.a.br(LR);
            }
        }
    }
}
