//! Allocated `MInst`s to machine code: the allocator's moves, one
//! instruction per `MInst` (`inst`), the edges' move stubs, the dispatch
//! entry, the exit stubs and the region's exit trampoline.

use crate::jit::backend::aarch64::abi::{self, BASE, FRAME_SPILLS, MAX_SPILLS, THREAD, X16, X17};
use crate::jit::backend::aarch64::asm::{Asm, FP, Fpr, Gpr, Label, Shift, Sz, ZR};
use crate::jit::backend::emit::{Emitted, ExitOut, Selected, Src, rotate_loops, thread_jumps};
use crate::jit::backend::ra::{Edit, InstOrEdit, Output};
use crate::jit::backend::reg::{Allocation, Block, Inst, PReg, RegClass};
use crate::jit::backend::vcode::{Pin, Remat, VCode};
use crate::jit::bug;
use crate::jit::compile::Bail;
use crate::jit::region::{Loc, SnapEntry};

mod inst;

/// The address of a routine the code branches to.
macro_rules! addr {
    ($f:path) => {
        $f as *const () as usize
    };
}

struct Em<'a> {
    a: Asm,
    v: &'a VCode,
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
    rt: u64,
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

/// `rt` is the runtime's address, which helpers take.
pub(crate) fn emit(s: &Selected, out: &Output, rt: u64) -> Result<Emitted, Bail> {
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
        rt,
        v: &s.vcode,
        out,
        a,
    };
    em.a.bind(em.region_word);
    em.a.emit(0);
    em.a.emit(0);
    let kept = |b: &u32| !skip[*b as usize];
    let mut laid: Vec<u32> = s.layout.emit.iter().copied().filter(kept).collect();
    let hot = s.layout.emit[..s.layout.hot]
        .iter()
        .filter(|b| kept(b))
        .count();
    let loops = &s.layout.loops;
    let tops = rotate_loops(&s.vcode, loops, &em.labels, &em.dest, &mut laid, hot);
    for (at, &b) in laid.iter().enumerate() {
        let next = laid.get(at + 1).map(|&x| em.labels[x as usize]);
        // A rotated loop's top, past the jump into its header: padding
        // there never runs.
        if tops.contains(&b) && em.a.after_jump() {
            em.a.align(16);
        }
        em.a.bind(em.labels[b as usize]);
        for ie in out.block_insts_and_edits(&s.vcode, Block::new(b as usize)) {
            match ie {
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
    Ok(Emitted {
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
                    Remat::Pinned(Pin::Base) => self.a.mov(gpr(r), BASE),
                    Remat::Pinned(Pin::Thread) => self.a.mov(gpr(r), THREAD),
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
}
