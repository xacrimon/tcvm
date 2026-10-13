//! Allocated `MInst`s to machine code (10.5, 10.6): the native frame, the
//! allocator's moves, every instruction's sequence, the exit stubs and the
//! region's exit trampoline in the cold section.

use regalloc2::{Allocation, Edit, Output, PReg, RegClass};

use crate::env::table::SlotLoc;
use crate::env::value::Value;
use crate::env::{MtCache, Shape};
use crate::jit::backend::aarch64::abi::{
    BASE, CLOSURE, D31, IMAGE_SPILLS, INSN, MAX_SPILLS, PC, RT, THREAD, X16, X17, image_index,
};
use crate::jit::backend::aarch64::asm::{
    Asm, Cond, Extend, FP, Fpr, Gpr, LR, Label, SP, Shift, Sz, ZR,
};
use crate::jit::backend::aarch64::inst::{AluOp, ENTRY_FAIL, FOp, FUnOp, Known, MInst, Test};
use crate::jit::backend::aarch64::lower::{LExit, Lowered, Src};
use crate::jit::backend::vcode::VCode;
use crate::jit::ir::ops::Ctx;
use crate::jit::ir::types::TypeSet;
use crate::jit::layout;
use crate::jit::region::{Loc, SRep, SnapEntry};

const NIL: u64 = 0xFFFF_FFFE_0000_0000;
const FALSE: u64 = NIL | 1;
pub(crate) const BOX: u64 = 0xFFF9_0000_0000_0000;
const PTR_MASK: u64 = (1 << 48) - 1;
use crate::jit::layout::FUNCTION_TAG;
/// The machine code of a region, before its address is known.
pub(crate) struct Emitted {
    pub(crate) asm: Asm,
    /// Where the interpreter enters (`JIT_ENTRY`/`JIT_LOOP`), and where a
    /// compiled caller does (5.4), when the region has a fast entry.
    pub(crate) entry: Label,
    pub(crate) fast: Option<Label>,
    pub(crate) num_spills: u32,
    /// Bytes of the native frame.
    pub(crate) frame_size: u32,
    /// The 8-byte word holding the region pointer.
    pub(crate) region_word: Label,
    pub(crate) exits: Vec<ExitOut>,
    /// The calls, for walks of the frame (H7).
    pub(crate) sites: Vec<SiteOut>,
}

/// A call of the region: its return addresses, whether each is a level's
/// (`jit_level`), the values the frame keeps in spill slots across it, its
/// header slot and the `pc` after it.
pub(crate) struct SiteOut {
    pub(crate) keys: Vec<(Label, bool)>,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
    pub(crate) hdr: u16,
    pub(crate) pc_after: u64,
}

pub(crate) struct ExitOut {
    pub(crate) pc: u32,
    pub(crate) frames: Vec<crate::jit::region::SnapFrame>,
    pub(crate) site: Option<u32>,
    pub(crate) kind: crate::jit::ir::ExitKind,
    pub(crate) tag: crate::jit::ir::ops::ExitTag,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
}

/// Each block's branch target and whether it is left out: a block that is
/// only a `Jump` once allocated (an edge split for moves that needed none)
/// is branched past, unless the chain is a loop.
fn thread_jumps(l: &Lowered, out: &Output, labels: &[Label]) -> (Vec<Label>, Vec<bool>) {
    let nb = l.vcode.blocks.len();
    let bare: Vec<Option<usize>> = (0..nb)
        .map(|b| {
            let blk = &l.vcode.blocks[b];
            if b == 0 || blk.resume || blk.end - blk.start != 1 {
                return None;
            }
            let mut it = out.block_insts_and_edits(&l.vcode, regalloc2::Block::new(b));
            match (it.next(), it.next()) {
                (Some(regalloc2::InstOrEdit::Inst(i)), None)
                    if matches!(l.vcode.insts[i.index()], MInst::Jump) =>
                {
                    Some(l.vcode.succs(b)[0].index())
                }
                _ => None,
            }
        })
        .collect();
    let mut dest = labels.to_vec();
    let mut skip = vec![false; nb];
    for b in 0..nb {
        let mut t = b;
        let mut n = 0;
        while let Some(s) = bare[t] {
            t = s;
            n += 1;
            if n > nb {
                t = b;
                break;
            }
        }
        dest[b] = labels[t];
        skip[b] = t != b;
    }
    (dest, skip)
}

struct Em<'a> {
    a: Asm,
    v: &'a VCode<MInst>,
    out: &'a Output,
    frame: u32,
    labels: Vec<Label>,
    /// Where a branch to each block goes: past blocks that only jump (E6).
    dest: Vec<Label>,
    exit_labels: Vec<Label>,
    exit_region: Label,
    entry_fail: Label,
    helpers: Helpers,
    /// The `TCVM_JIT_DEOPT_ALL` flags of the exits, by address.
    fired: Option<usize>,
    /// Per resume block, where a fast callee's `ret` lands.
    fast_resume: Vec<Label>,
    /// Per resume block, its call's miss path (`miss_path`), branched to
    /// from just before `fast_resume`.
    miss: Vec<Label>,
    /// Per resume block, past the fast resume's `BASE` and `CLOSURE`, which
    /// a native call keeps and a level's return restores.
    resume_based: Vec<Label>,
    /// The word holding the region pointer, which each frame records.
    region_word: Label,
    sites: Vec<SiteOut>,
    sites_of: &'a [(usize, crate::jit::backend::aarch64::lower::LSite)],
    snap_ops: &'a [u32],
    /// This region's prototype and its known entry, when it has one.
    own_entry: Option<(u64, Label)>,
    /// `FMod`'s out-of-line paths: (entry, return, d, a, b).
    fmods: Vec<(Label, Label, Fpr, Fpr, Fpr)>,
}

/// A function region's fast entry: what `enter` would check of the callee.
#[derive(Clone, Copy)]
pub(crate) struct FastEntry {
    pub(crate) max_stack: u8,
    pub(crate) num_params: u8,
    /// The region's prototype, whose calls enter here directly (D2).
    pub(crate) proto: u64,
}

/// The fast entry past its arity and native stack checks, for a caller
/// that proved them (D2).
pub(crate) const KNOWN_ENTRY: u64 = 24;

/// Addresses of the Rust routines a region branches to.
#[derive(Clone)]
pub(crate) struct Helpers {
    pub(crate) exit_common: usize,
    pub(crate) entry_fail: usize,
    pub(crate) enter: usize,
    /// `jit_box_i64`'s trampoline.
    pub(crate) box_i64: usize,
    pub(crate) land: usize,
    /// `abi::jit_level` and `abi::ret_to_native`.
    pub(crate) level: usize,
    pub(crate) ret_to_native: usize,
    /// `abi::native_entry`, and `native_act` for what a native asks for
    /// besides a plain return.
    pub(crate) native: usize,
    pub(crate) native_act: usize,
    /// The trampolines of the helpers cold arms call: (helper, trampoline).
    pub(crate) trampolines: Vec<(usize, usize)>,
}

/// A trampoline (A5): a C call of `target` that keeps x0-x15, d0-d7 and
/// d16-d30 but the result's register, behind a frame record that keeps the
/// frame chain walkable; and its frame's size.
pub(crate) fn trampoline(target: usize, ret_float: bool) -> (Asm, u32) {
    let ints: Vec<Gpr> = (0..16u8)
        .filter(|&r| ret_float || r != 0)
        .map(Gpr)
        .collect();
    let floats: Vec<Fpr> = (0..8u8)
        .chain(16..31)
        .filter(|&r| !ret_float || r != 0)
        .map(Fpr)
        .collect();
    let frame = (16 + 8 * (ints.len() + floats.len()) as u64).next_multiple_of(16);
    let mut a = Asm::new();
    a.sub_imm(Sz::X, SP, SP, frame);
    a.stp(FP, LR, SP, 0);
    a.mov_sp(FP, SP);
    let save = |a: &mut Asm, load: bool| {
        let mut off = 16;
        for p in ints.chunks(2) {
            match (p, load) {
                ([x, y], false) => a.stp(*x, *y, SP, off),
                ([x, y], true) => a.ldp(*x, *y, SP, off),
                (_, false) => a.str(p[0], SP, off),
                (_, true) => a.ldr(p[0], SP, off),
            }
            off += 8 * p.len() as i32;
        }
        for p in floats.chunks(2) {
            match (p, load) {
                ([x, y], false) => a.stp_d(*x, *y, SP, off),
                ([x, y], true) => a.ldp_d(*x, *y, SP, off),
                (_, false) => a.str_d(p[0], SP, off),
                (_, true) => a.ldr_d(p[0], SP, off),
            }
            off += 8 * p.len() as i32;
        }
    };
    save(&mut a, false);
    a.bl_far(target);
    save(&mut a, true);
    a.ldp(FP, LR, SP, 0);
    a.add_imm(Sz::X, SP, SP, frame);
    a.ret();
    (a, frame as u32)
}

fn gpr(p: PReg) -> Gpr {
    debug_assert_eq!(p.class(), RegClass::Int);
    Gpr(p.hw_enc() as u8)
}

fn fpr(p: PReg) -> Fpr {
    debug_assert_eq!(p.class(), RegClass::Float);
    Fpr(p.hw_enc() as u8)
}

/// The frame: the frame record, the region word, then the spill slots.
pub(crate) const FRAME_REGION: i32 = crate::jit::FRAME_REGION as i32;
pub(crate) const FRAME_SPILLS: i32 = crate::jit::FRAME_SPILLS as i32;

fn spill_off(slot: usize) -> i32 {
    FRAME_SPILLS + 8 * slot as i32
}

pub(crate) fn emit(
    l: &Lowered,
    out: &Output,
    helpers: Helpers,
    fired: Option<usize>,
    fast: Option<FastEntry>,
) -> Result<Emitted, String> {
    if out.num_spillslots > MAX_SPILLS {
        return Err(format!("{} spill slots", out.num_spillslots));
    }
    let frame = (FRAME_SPILLS as u32 + 8 * out.num_spillslots as u32).next_multiple_of(16);
    let mut a = Asm::new();
    let labels: Vec<Label> = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let (dest, skip) = thread_jumps(l, out, &labels);
    let exit_labels: Vec<Label> = (0..l.exits.len()).map(|_| a.new_label()).collect();
    let exit_region = a.new_label();
    let entry_fail = a.new_label();
    let fast_resume = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let miss = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let resume_based = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let region_word = a.new_label();
    let mut em = Em {
        a,
        v: &l.vcode,
        out,
        frame,
        labels,
        dest,
        exit_labels,
        exit_region,
        entry_fail,
        helpers,
        fired,
        fast_resume,
        miss,
        resume_based,
        region_word,
        sites: Vec::new(),
        sites_of: &l.sites,
        snap_ops: &l.snap_ops,
        fmods: Vec::new(),
        own_entry: None,
    };
    // A compiled caller has written the header, set base, closure and `insn`
    // (the argument count) as `enter` would, and its `blr` left the return
    // address in x30; what `enter` would check of a Lua callee is checked
    // here, and a miss goes to the caller's slow path (`fast_miss`). Missing
    // arguments are filled in (`arity_fixup`, I3).
    let fast_miss = em.a.new_label();
    let (fixup, budget) = (em.a.new_label(), em.a.new_label());
    let fast_label = fast.map(|fe| {
        let l = em.a.new_label();
        em.a.bind(l);
        em.a.cmp_imm(Sz::X, INSN, fe.num_params as u64);
        em.a.b_cond(Cond::Lo, fixup);
        em.a.bind(budget);
        em.a.mov_sp(X16, SP);
        em.a.ldr(X17, THREAD, layout::NATIVE_LIMIT as i32);
        em.a.cmp(Sz::X, X16, X17);
        em.a.b_cond(Cond::Lo, fast_miss);
        let known = em.a.new_label();
        em.a.bind(known);
        assert_eq!(
            (em.a.label_offset(known) - em.a.label_offset(l)) as u64,
            KNOWN_ENTRY
        );
        em.own_entry = Some((fe.proto, known));
        em.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
        em.a.add_imm(Sz::X, X17, BASE, fe.max_stack as u64 * 8);
        em.a.cmp(Sz::X, X17, X16);
        em.a.b_cond(Cond::Hi, fast_miss);
        l
    });
    let nb = l.vcode.blocks.len();
    for b in 0..nb {
        if skip[b] {
            continue;
        }
        let blk = &l.vcode.blocks[b];
        em.a.bind(em.labels[b]);
        for (k, ie) in out
            .block_insts_and_edits(&l.vcode, regalloc2::Block::new(b))
            .enumerate()
        {
            match ie {
                // A fast callee's `ret` lands at the `Resume`, past anything
                // before it.
                regalloc2::InstOrEdit::Edit(_) if blk.resume && k == 0 => {
                    return Err("a move before a call's resume".into());
                }
                regalloc2::InstOrEdit::Edit(Edit::Move { from, to }) => em.mov(*from, *to),
                regalloc2::InstOrEdit::Inst(i) => {
                    let next = (b + 1..nb).find(|&x| !skip[x]).map(|x| em.labels[x]);
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
    for (slow, back, d, x, y) in std::mem::take(&mut em.fmods) {
        // A zero remainder is exact: fmod's zero has the dividend's sign.
        let call = em.a.new_label();
        em.a.bind(slow);
        em.a.fcmp_zero(d);
        em.a.b_cond(Cond::Ne, call);
        em.a.fmov_to_gpr(X16, x);
        em.a.and_imm(Sz::X, X16, X16, 1 << 63);
        em.a.fmov_from_gpr(d, X16);
        em.a.b(back);
        em.a.bind(call);
        let t = crate::jit::helpers::address(crate::jit::ir::ops::HelperId::FMod);
        let t = em.helpers.trampolines.iter().find(|(h, _)| *h == t);
        em.a.fmov_to_gpr(X16, x);
        em.a.fmov_to_gpr(X17, y);
        em.a.sub_imm(Sz::X, SP, SP, 16);
        em.a.stp_d(Fpr(0), Fpr(1), SP, 0);
        em.a.fmov_from_gpr(Fpr(0), X16);
        em.a.fmov_from_gpr(Fpr(1), X17);
        em.a.bl_far(t.expect("fmod's trampoline").1);
        em.a.fmov_to_gpr(X16, Fpr(0));
        em.a.ldp_d(Fpr(0), Fpr(1), SP, 0);
        em.a.add_imm(Sz::X, SP, SP, 16);
        em.a.fmov_from_gpr(d, X16);
        em.a.b(back);
    }
    em.a.align(8);
    let region_word = em.region_word;
    em.a.bind(em.exit_region);
    em.a.ldr_label(X17, region_word);
    em.a.orr_shift(Sz::X, X16, X17, X16, Shift::Lsl, 48);
    em.a.b_far(em.helpers.exit_common);
    if !l.entry_regs.is_empty() {
        em.a.bind(em.entry_fail);
        em.close_frame();
        em.a.ldr_label(INSN, region_word);
        em.a.ldr_lit(X16, em.helpers.entry_fail as u64);
        em.tail_level();
    }
    if let Some(fe) = fast {
        // The arguments short of the parameters are nil, once the window
        // fits; then on to the checks past the arity one.
        em.a.bind(fixup);
        em.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
        em.a.add_imm(Sz::X, X17, BASE, fe.max_stack as u64 * 8);
        em.a.cmp(Sz::X, X17, X16);
        em.a.b_cond(Cond::Hi, fast_miss);
        em.a.add_shift(Sz::X, X17, BASE, INSN, Shift::Lsl, 3);
        em.a.add_imm(Sz::X, Gpr(9), BASE, fe.num_params as u64 * 8);
        em.a.mov_imm(X16, NIL);
        let fill = em.a.new_label();
        em.a.bind(fill);
        em.a.str(X16, X17, 0);
        em.a.add_imm(Sz::X, X17, X17, 8);
        em.a.cmp(Sz::X, X17, Gpr(9));
        em.a.b_cond(Cond::Lo, fill);
        em.a.b(budget);
    }
    if fast_label.is_some() {
        // Back to the caller's slow path (`miss_path`): 4 bytes past its
        // return address, or before its fast resume, which a tagged one
        // names (`deep_path`). A tail call from a frame dispatch entered has
        // the level's return address instead: `enter` runs the call from the
        // header the tail call wrote.
        em.a.bind(fast_miss);
        let (tagged, dispatched) = (em.a.new_label(), em.a.new_label());
        em.a.ldr(X17, RT, layout::LEVEL_RET as i32);
        em.a.cmp(Sz::X, LR, X17);
        em.a.b_cond(Cond::Eq, dispatched);
        em.a.tbnz(LR, 0, tagged);
        em.a.add_imm(Sz::X, LR, LR, 4);
        em.a.ret();
        em.a.bind(tagged);
        em.a.and_imm(Sz::X, LR, LR, !1u64);
        em.a.sub_imm(Sz::X, LR, LR, 4);
        em.a.br(LR);
        em.a.bind(dispatched);
        em.a.movz(Sz::X, X16, FUNCTION_TAG, 3);
        em.a.orr(Sz::X, X16, X16, CLOSURE);
        em.a.str(X16, BASE, -32);
        em.a.sub_imm(Sz::X, PC, BASE, 32);
        em.a.mov(BASE, ZR);
        em.a.b_far(em.helpers.enter);
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
    let entry = em.labels[0];
    let frame_size = em.frame;
    Ok(Emitted {
        asm: em.a,
        entry,
        fast: fast_label,
        num_spills: out.num_spillslots as u32,
        frame_size,
        region_word,
        exits,
        sites: em.sites,
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
        frames: e.frames.clone(),
        site: e.site,
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
        self.a.ldr_label(X16, self.region_word);
        self.a.str(X16, FP, FRAME_REGION);
    }

    fn close_frame(&mut self) {
        self.a.ldp(FP, LR, SP, 0);
        self.a.add_imm(Sz::X, SP, SP, self.frame as u64);
    }

    /// The fast path of a call whose header is at `base + hdr`, `insn` the
    /// argument count: a Lua callee without varargs whose prototype has a
    /// fast entry is called with `blr`, this frame kept, and its `ret` lands
    /// past the resume block's frame push (5.4); a native goes to `native`
    /// (`native_path`), anything else to `slow`. Every register is free
    /// here: what lives across the call is in the frame. Returns
    /// `fast_enter`'s labels.
    fn fast_call(&mut self, hdr: i32, block: usize, slow: Label, native: Label) -> (Label, Label) {
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
        self.fast_enter(hdr, block, None)
    }

    /// `fast_call` for a known callee (D2): its prototype's fast entry, or
    /// `slow`, past the arity check when `nargs` is static, the parameters it
    /// leaves out stored nil (I3), and this region's own when it is the
    /// callee's; word 0 the closure word. Also returns whether it enters past
    /// the checks.
    fn known_call(
        &mut self,
        hdr: i32,
        block: usize,
        slow: Label,
        (f, proto, nparams): (u64, u64, u8),
        nargs: u8,
    ) -> (Label, Label, bool, Option<Label>) {
        let known = nargs != u8::MAX;
        if known && nargs < nparams {
            // Inside the callee's window, which its known entry checks; the
            // value stack's mapping runs past any window that fits.
            self.a.mov_imm(Gpr(9), NIL);
            for k in nargs..nparams {
                self.a.str(Gpr(9), BASE, hdr + 32 + 8 * k as i32);
            }
        }
        // A call of this region's own function enters its known entry by a
        // direct `bl` (I4): the region cannot outlive itself.
        let own = match self.own_entry {
            Some((p, l)) if p == proto && known => Some(l),
            _ => {
                self.a.mov_imm(X17, proto);
                self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
                self.a.cbz(Sz::X, X17, slow);
                if known {
                    self.a.add_imm(Sz::X, X17, X17, KNOWN_ENTRY);
                }
                None
            }
        };
        self.a.mov_imm(X16, f & PTR_MASK);
        let (ret, deep) = self.fast_enter(hdr, block, own);
        (ret, deep, known, own)
    }

    /// Enter the fast entry in x17, or `own`, of the closure in x16
    /// (`fast_call`). Returns the return address `blr` leaves, where a
    /// branch to the fast resume sits; a callee whose entry checks fail
    /// returns 4 bytes past it, where `miss_path` must follow. Also returns
    /// the deep path's label, for `deep_path`.
    fn fast_enter(&mut self, hdr: i32, block: usize, own: Option<Label>) -> (Label, Label) {
        let fast_resume = self.fast_resume[self.v.succs(block)[0].index()];
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
        self.a.mov(CLOSURE, X16);
        let deep = self.a.new_label();
        self.a.mov_sp(Gpr(9), SP);
        self.a.ldr(Gpr(10), THREAD, layout::NATIVE_DEEP as i32);
        self.a.cmp(Sz::X, Gpr(9), Gpr(10));
        self.a.b_cond(Cond::Lo, deep);
        match own {
            Some(l) => self.a.bl(l),
            None => self.a.blr(X17),
        }
        let ret = self.a.new_label();
        self.a.bind(ret);
        self.a.b(fast_resume);
        (ret, deep)
    }

    /// Where a fast callee whose entry checks failed returns (4 bytes past
    /// `fast_enter`'s return address, or from before the fast resume): the
    /// caller's `BASE` and `CLOSURE` and word 0's callee value back, then
    /// its slow path at `slow`, whose level keeps that `CLOSURE`.
    fn miss_path(&mut self, hdr: i32, block: usize, slow: Label) {
        self.a.bind(self.miss[self.v.succs(block)[0].index()]);
        self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
        self.a.ldr(CLOSURE, BASE, -32);
        self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
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
    fn native_path(
        &mut self,
        hdr: i32,
        block: usize,
        from: Result<(Label, Label), u64>,
    ) -> Vec<(Label, bool)> {
        let resume = self.resume_based[self.v.succs(block)[0].index()];
        match from {
            Ok((check, slow)) => {
                self.a.bind(check);
                self.a.cmp_imm(Sz::W, X17, 1);
                self.a.b_cond(Cond::Eq, slow);
                self.a.add_imm(Sz::X, Gpr(3), X16, layout::FN_NATIVE as u64);
            }
            Err(nc) => self.a.mov_imm(Gpr(3), nc),
        }
        let win = hdr as u64 + 32;
        self.a.mov(Gpr(0), RT);
        self.a.add_imm(Sz::X, Gpr(1), BASE, win);
        if from.is_ok() {
            self.a.mov(Gpr(2), INSN);
        }
        self.a.bl_far(self.helpers.native);
        let ret = self.a.new_label();
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
        self.a.ldr_lit(Gpr(10), self.helpers.ret_to_native as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, hdr + 8);
        self.a.mov(INSN, Gpr(0));
        self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
        self.a.add_imm(Sz::X, X17, BASE, win);
        self.a.ldr_lit(X16, self.helpers.native_act as u64);
        self.a.bl_far(self.helpers.level);
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.b(resume);
        vec![(ret, false), (back, true)]
    }

    /// Whether the resume block after `block` reads the result count and
    /// the values (`INSN`, `PC`), not just the first result in x0.
    fn resume_reads_count(&self, block: usize) -> bool {
        let r = self.v.succs(block)[0].index();
        !matches!(
            self.v.insts[self.v.blocks[r].start as usize],
            MInst::Resume { wanted: 1, .. }
                | MInst::Resume {
                    wanted: 0,
                    c: 1,
                    ..
                }
        )
    }

    /// Deep in a chain the return address goes in tagged (bit 0), so the
    /// callee returns with `br` and the return address stack stays as it
    /// was, right for the shallower frames' `ret`s. It names the fast resume
    /// itself, before which a branch to the miss path sits.
    fn deep_path(&mut self, deep: Label, block: usize, known: bool, own: Option<Label>) {
        self.a.bind(deep);
        if let Some(l) = own {
            self.a.adr(X17, l);
        }
        if known {
            // The known entry trusts the native stack, which is deep enough
            // here to check: below the limit the full entry misses.
            let ok = self.a.new_label();
            self.a.ldr(Gpr(10), THREAD, layout::NATIVE_LIMIT as i32);
            self.a.cmp(Sz::X, Gpr(9), Gpr(10));
            self.a.b_cond(Cond::Hs, ok);
            self.a.sub_imm(Sz::X, X17, X17, KNOWN_ENTRY);
            self.a.bind(ok);
        }
        self.a
            .adr(LR, self.fast_resume[self.v.succs(block)[0].index()]);
        self.a.orr_imm(Sz::X, LR, LR, 1);
        self.a.br(X17);
    }

    /// The slow path of a call whose header at `hdr` holds the callee's
    /// value: a level runs it through `enter` (H6.2), with `ret_to_native`
    /// to come back. Returns the address the level returns to, from where
    /// the resume block goes on past its `BASE` (a level's return restores
    /// this frame's).
    fn level_call(&mut self, hdr: i32, block: usize, pc_after: usize) -> Label {
        self.a.ldr_lit(X16, self.helpers.ret_to_native as u64);
        self.a.str(X16, BASE, hdr + 8);
        self.a.str(BASE, BASE, hdr + 16);
        self.a.ldr_lit(X16, pc_after as u64);
        self.a.str(X16, BASE, hdr + 24);
        self.a.add_imm(Sz::X, PC, BASE, hdr as u64);
        self.a.add_imm(Sz::X, X17, BASE, hdr as u64 + 32);
        self.a.ldr_lit(X16, self.helpers.enter as u64);
        self.a.bl_far(self.helpers.level);
        let back = self.a.new_label();
        self.a.bind(back);
        self.a.b(self.resume_based[self.v.succs(block)[0].index()]);
        back
    }

    /// The frame is closed and `x16` holds the handler that goes on: one
    /// dispatch entered goes on in its level; a fast-called one starts a
    /// level returning to its caller, its header made to return through it.
    fn tail_level(&mut self) {
        let level = self.a.new_label();
        self.a.ldr(X17, RT, layout::LEVEL_RET as i32);
        self.a.cmp(Sz::X, LR, X17);
        self.a.b_cond(Cond::Ne, level);
        self.a.br(X16);
        self.a.bind(level);
        self.a.ldr(Gpr(9), BASE, -24);
        self.a
            .and_imm(Sz::X, Gpr(9), Gpr(9), crate::vm::frame::flag::MASK);
        self.a.ldr_lit(Gpr(10), self.helpers.ret_to_native as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, -24);
        self.a.mov(X17, BASE);
        self.a.b_far(self.helpers.level);
    }

    /// Record the call at VCode instruction `i` and its return addresses:
    /// past `blr`, the fast resume (a tagged one's), and a level's.
    fn call_site(
        &mut self,
        keys: Vec<(Label, bool)>,
        i: usize,
        block: usize,
        hdr: u16,
        pc_after: usize,
    ) {
        let (entries, consts) = self.site_entries(i, block, hdr);
        self.sites.push(SiteOut {
            keys,
            entries,
            consts,
            hdr,
            pc_after: pc_after as u64,
        });
    }

    /// The values a call at VCode instruction `i` keeps in the frame: its
    /// stack-slot snapshot operands (`LSite`), and the registers below its
    /// header `hdr` whose home slots hold theirs.
    fn site_entries(&self, i: usize, block: usize, hdr: u16) -> (Vec<SnapEntry>, Vec<u64>) {
        let resume = self.v.succs(block)[0].index();
        let Some((_, site)) = self.sites_of.iter().find(|(b, _)| *b == resume) else {
            return (Vec::new(), Vec::new());
        };
        let allocs = self.allocs(i);
        let base = self.snap_ops[i] as usize;
        let mut entries = Vec::new();
        let mut consts = Vec::new();
        for &(reg, rep, src) in &site.entries {
            let loc = match src {
                Src::Const(w) => {
                    consts.push(w);
                    Loc::Const(consts.len() as u32 - 1)
                }
                Src::Operand(k) => Loc::Spill(
                    allocs[base + k as usize]
                        .as_stack()
                        .expect("a kept value outside the frame")
                        .index() as u16,
                ),
            };
            entries.push(SnapEntry { reg, rep, loc });
        }
        for r in site.homes.iter().filter(|&r| r < hdr as usize) {
            entries.push(SnapEntry {
                reg: r as u8,
                rep: SRep::Val,
                loc: Loc::Home,
            });
        }
        (entries, consts)
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
            // The high 16 bits, sign extended, are `tag - 8`.
            self.a.asr_imm(Sz::X, X16, v, 48);
            self.a.cmn_imm(Sz::X, X16, 8 - tag as u64);
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
            LToFExact { exit } => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                // 10 or more leading bits equal to the sign: |n| <= 2^53.
                self.a.cls(X16, n);
                self.a.cmp_imm(Sz::X, X16, 10);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Lo, l);
                self.a.scvtf(Sz::X, d, n);
            }
            StrCmp { le } => {
                use crate::env::string::LuaString;
                let (d, a, b) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, k, lb, ca, cb) = (
                    self.r(i, 3),
                    self.r(i, 4),
                    self.r(i, 5),
                    self.r(i, 6),
                    self.r(i, 7),
                );
                let (top, tail, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                self.a.ldr(d, a, LuaString::LEN_OFFSET as i32);
                self.a.ldr(lb, b, LuaString::LEN_OFFSET as i32);
                self.a.cmp(Sz::X, d, lb);
                self.a.csel(Sz::X, n, d, lb, Cond::Lo);
                self.a.add_imm(Sz::X, X16, a, LuaString::DATA_OFFSET as u64);
                self.a.add_imm(Sz::X, X17, b, LuaString::DATA_OFFSET as u64);
                self.a.movz(Sz::X, k, 0, 0);
                self.a.bind(top);
                self.a.cmp(Sz::X, k, n);
                self.a.b_cond(Cond::Hs, tail);
                self.a.ldrb_reg(ca, X16, k);
                self.a.ldrb_reg(cb, X17, k);
                self.a.add_imm(Sz::X, k, k, 1);
                self.a.cmp(Sz::W, ca, cb);
                self.a.b_cond(Cond::Eq, top);
                // The first differing byte decides.
                self.a.cset(Sz::W, d, Cond::Lo);
                self.a.b(done);
                // A common prefix: the shorter orders first.
                self.a.bind(tail);
                self.a.cmp(Sz::X, d, lb);
                self.a.cset(Sz::W, d, if le { Cond::Ls } else { Cond::Lo });
                self.a.bind(done);
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
            FMod => {
                // r = a - floor(a / b) * b. A rounded quotient whose floor is
                // off leaves r outside (0, b), as do a NaN, an infinity, a
                // zero or negative divisor and a zero remainder.
                let (d, q, x, y) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2), self.fr(i, 3));
                let (slow, back) = (self.a.new_label(), self.a.new_label());
                self.a.fdiv(q, x, y);
                self.a.frintm(q, q);
                self.a.fmsub(d, q, y, x);
                self.a.fcmp_zero(d);
                self.a.fccmp(d, y, 0, Cond::Gt);
                self.a.b_cond(Cond::Ge, slow);
                self.a.bind(back);
                self.fmods.push((slow, back, d, x, y));
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
                let d = self.r(i, 0);
                if let Test::Type(set) = test
                    && !is_single_test(set)
                {
                    // `d` may share the tested value's register: written last.
                    let v = self.r(i, 1);
                    let (yes, done) = (self.a.new_label(), self.a.new_label());
                    self.type_branch(v, set, yes);
                    self.a.movz(Sz::W, d, (cond != Cond::Eq) as u16, 0);
                    self.a.b(done);
                    self.a.bind(yes);
                    self.a.movz(Sz::W, d, (cond == Cond::Eq) as u16, 0);
                    self.a.bind(done);
                } else {
                    let c = self.test(i, 1, test, cond);
                    self.a.cset(Sz::W, d, c);
                }
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
                if test == Test::B1 {
                    let c = self.r(i, 0);
                    if cond == Cond::Ne {
                        self.a.cbz(Sz::W, c, l);
                    } else {
                        self.a.cbnz(Sz::W, c, l);
                    }
                } else if let Test::Type(set) = test
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
            NumMtSet => {
                let d = self.r(i, 0);
                self.a.ldr(X16, RT, layout::NUMBER_MT_CELL as i32);
                self.a.ldr(X16, X16, layout::NUMBER_MT as i32);
                self.a.cmp_imm(Sz::X, X16, 0);
                self.a.cset(Sz::W, d, Cond::Ne);
            }
            TypeMtIs { slot, word } => {
                let d = self.r(i, 0);
                self.a.ldr(X16, RT, layout::type_mt_cell(slot) as i32);
                self.a.ldr(X16, X16, layout::NUMBER_MT as i32);
                self.a.ldr_lit(X17, word);
                self.a.cmp(Sz::X, X16, X17);
                self.a.cset(Sz::W, d, Cond::Eq);
            }
            ClassMm(k) => {
                // Shape, class, the class's other metamethods: each may be
                // missing (null), and then so is the metamethod.
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let (none, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a
                    .ldr(X16, X16, crate::env::Shape::MT_CACHE_OFFSET as i32);
                self.a.cbz(Sz::X, X16, none);
                self.a.ldr(X16, X16, MtCache::REST_OFFSET as i32);
                self.a.cbz(Sz::X, X16, none);
                self.a.ldr(
                    d,
                    X16,
                    (MtCache::REST_VALUES_OFFSET + 8 * k as usize) as i32,
                );
                self.a.b(done);
                self.a.bind(none);
                self.a.mov_imm(d, NIL);
                self.a.bind(done);
            }
            StrLenOf => {
                let (d, s) = (self.r(i, 0), self.r(i, 1));
                self.a
                    .ldr(d, s, crate::env::string::LuaString::LEN_OFFSET as i32);
            }
            TabMmBits => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let (none, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a
                    .ldr(X16, X16, crate::env::Shape::MT_CACHE_OFFSET as i32);
                self.a.cbz(Sz::X, X16, none);
                self.a.ldr_w(d, X16, MtCache::BITS_OFFSET as i32);
                self.a.b(done);
                self.a.bind(none);
                self.a.movz(Sz::W, d, 0, 0);
                self.a.bind(done);
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
            NamedNext => {
                let (d, t, from) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, cap, mt) = (self.r(i, 3), self.r(i, 4), self.r(i, 5));
                let [top, spilled, check, step, none, done] = [(); 6].map(|_| self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_w(n, X16, Shape::SLOT_COUNT_OFFSET as i32);
                self.a.ldrb(cap, X16, Shape::INLINE_CAP_OFFSET as i32);
                self.a.ldr_w(mt, X16, Shape::MT_SLOT_OFFSET as i32);
                self.a.mov_w(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::W, d, n);
                self.a.b_cond(Cond::Hs, none);
                self.a.cmp(Sz::W, d, mt);
                self.a.b_cond(Cond::Eq, step);
                self.a.cmp(Sz::W, d, cap);
                self.a.b_cond(Cond::Hs, spilled);
                self.a.add_imm(Sz::X, X17, t, layout::T_INLINE as u64);
                self.a.ldr_idx(X16, X17, d, Extend::Uxtw, true);
                self.a.b(check);
                self.a.bind(spilled);
                self.a.ldr(X17, t, layout::T_SPILL as i32);
                self.a.sub(Sz::W, X16, d, cap);
                self.a.ldr_idx(X16, X17, X16, Extend::Uxtw, true);
                self.a.bind(check);
                self.a.eor_imm(Sz::X, X16, X16, NIL);
                self.a.cbnz(Sz::X, X16, done);
                self.a.bind(step);
                self.a.add_imm(Sz::W, d, d, 1);
                self.a.b(top);
                self.a.bind(none);
                self.a.movn(Sz::W, d, 0, 0);
                self.a.bind(done);
            }
            NamedVal => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (spilled, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldrb(X16, X16, Shape::INLINE_CAP_OFFSET as i32);
                self.a.cmp(Sz::W, k, X16);
                self.a.b_cond(Cond::Hs, spilled);
                self.a.add_imm(Sz::X, X17, t, layout::T_INLINE as u64);
                self.a.ldr_idx(d, X17, k, Extend::Uxtw, true);
                self.a.b(done);
                self.a.bind(spilled);
                self.a.sub(Sz::W, X16, k, X16);
                self.a.ldr(X17, t, layout::T_SPILL as i32);
                self.a.ldr_idx(d, X17, X16, Extend::Uxtw, true);
                self.a.bind(done);
            }
            NamedKey => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr(X16, X16, Shape::KEYS_OFFSET as i32);
                self.a.ldr(X16, X16, Shape::KEYS_SLOTS_OFFSET as i32);
                self.a.ldr_idx(X16, X16, k, Extend::Uxtw, true);
                self.a.add_imm(
                    Sz::X,
                    X16,
                    X16,
                    crate::env::string::LuaString::VALUE_OFFSET as u64,
                );
                self.a.ldr_lit(X17, crate::env::value::STRING_BITS);
                self.a.orr(Sz::X, d, X16, X17);
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
            Helper(h) | HelperCold(h) => {
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
                let target = crate::jit::helpers::address(h);
                if matches!(inst, HelperCold(_)) {
                    let t = self.helpers.trampolines.iter().find(|(x, _)| *x == target);
                    self.a.bl_far(t.expect("a trampoline for a cold helper").1);
                } else {
                    self.a.bl_far(target);
                }
            }
            Jump => {
                let t = self.dest[self.v.succs(block)[0].index()];
                if Some(t) != next {
                    self.a.b(t);
                }
            }
            Br { test, cond } => {
                let (t, f) = (
                    self.v.succs(block)[0].index(),
                    self.v.succs(block)[1].index(),
                );
                let (lt, lf) = (self.dest[t], self.dest[f]);
                let multi = matches!(test, Test::Type(set) if !is_single_test(set));
                if test == Test::B1 || multi {
                    // Where `cond` holds, and where it does not.
                    let (yes, no) = if cond == Cond::Eq || (test == Test::B1 && cond == Cond::Ne) {
                        (lt, lf)
                    } else {
                        (lf, lt)
                    };
                    let v = self.r(i, 0);
                    match test {
                        Test::Type(set) => self.type_branch(v, set, yes),
                        _ => self.a.cbnz(Sz::W, v, yes),
                    }
                    if Some(no) != next {
                        self.a.b(no);
                    }
                    return Ok(());
                }
                let c = self.test(i, 0, test, cond);
                if Some(lt) == next {
                    self.a.b_cond(c.invert(), lf);
                } else {
                    self.a.b_cond(c, lt);
                    if Some(lf) != next {
                        self.a.b(lf);
                    }
                }
            }
            Call {
                a,
                nargs,
                pc_after,
                known,
            } => {
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
                    // Nothing returns past the resume block's branch to it.
                    self.a.bind(self.miss[self.v.succs(block)[0].index()]);
                    self.a.brk(0xf2);
                    self.call_site(keys, i, block, a as u16, pc_after);
                    return Ok(());
                }
                let (slow, common) = (self.a.new_label(), self.a.new_label());
                let native = self.a.new_label();
                let (ret, deep, entered_known, own) = match known {
                    Some(Known::Lua(f, proto, nparams)) => {
                        self.known_call(hdr, block, slow, (f, proto, nparams), nargs)
                    }
                    _ => {
                        let (r, d) = self.fast_call(hdr, block, slow, native);
                        (r, d, false, None)
                    }
                };
                self.miss_path(hdr, block, common);
                self.deep_path(deep, block, entered_known, own);
                let mut keys = vec![(ret, false)];
                if known.is_none() {
                    keys.extend(self.native_path(hdr, block, Ok((native, slow))));
                }
                self.a.bind(slow);
                if let Some(Known::Lua(f, _, _)) = known {
                    self.a.mov_imm(X16, f);
                    self.a.str(X16, BASE, hdr);
                }
                self.a.bind(common);
                let back = self.level_call(hdr, block, pc_after);
                let fast = self.fast_resume[self.v.succs(block)[0].index()];
                keys.extend([(fast, false), (back, true)]);
                self.call_site(keys, i, block, a as u16, pc_after);
            }
            Dispatch { op, word, pc_after } => {
                self.close_frame();
                self.a.ldr_lit(INSN, word);
                self.a.ldr_lit(PC, pc_after as u64);
                self.a.ldr(X16, RT, op as i32 * 8);
                self.tail_level();
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
                self.a.ldr_idx(X16, RT, X17, Extend::Lsl, true);
                self.tail_level();
            }
            Resume { wanted, c, a, hdr } => {
                // Nothing falls through into a resume block: its call's
                // tagged return address names the fast resume, 4 bytes past
                // the way to the miss path (`fast_miss`).
                self.a.b(self.miss[block]);
                self.a.bind(self.fast_resume[block]);
                // The callee's base is past the header at `hdr`; our header's
                // word 0 is our closure.
                self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 * 8 + 32);
                self.a.ldr(CLOSURE, BASE, -32);
                self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
                self.a.bind(self.resume_based[block]);
                match wanted {
                    0 if c == 1 => {}
                    0 => {
                        // The results to `R[a..]` (I5): all of them, `top`
                        // past them, or `c - 1` filled out with nil. Up to
                        // three here, more through the helper. They come from
                        // at least a header above `R[a]`, so copying up is
                        // safe, and reading past them stays in the stack.
                        let (helper, top, done) =
                            (self.a.new_label(), self.a.new_label(), self.a.new_label());
                        let at = a as i32 * 8;
                        if c == 0 {
                            self.a.cmp_imm(Sz::X, INSN, 3);
                            self.a.b_cond(Cond::Hi, helper);
                            for k in 0..3 {
                                self.a.cmp_imm(Sz::X, INSN, k);
                                self.a.b_cond(Cond::Eq, top);
                                self.a.ldr(X16, PC, 8 * k as i32);
                                self.a.str(X16, BASE, at + 8 * k as i32);
                            }
                            self.a.bind(top);
                            self.a.ldr(X16, THREAD, layout::THREAD_STACK as i32);
                            self.a.add_imm(Sz::X, X17, BASE, at as u64);
                            self.a.sub(Sz::X, X17, X17, X16);
                            self.a.add_shift(Sz::X, X17, INSN, X17, Shift::Lsr, 3);
                            self.a.str(X17, THREAD, layout::THREAD_TOP as i32);
                            self.a.b(done);
                        } else if c <= 4 {
                            self.a.mov_imm(X17, NIL);
                            for k in 0..(c - 1) as u64 {
                                self.a.ldr(X16, PC, 8 * k as i32);
                                self.a.cmp_imm(Sz::X, INSN, k);
                                self.a.csel(Sz::X, X16, X16, X17, Cond::Hi);
                                self.a.str(X16, BASE, at + 8 * k as i32);
                            }
                            self.a.b(done);
                        }
                        self.a.bind(helper);
                        // Land through the helper: (rt, thread, base, a, values, nret, c).
                        self.a.mov(Gpr(0), RT);
                        self.a.mov(Gpr(1), THREAD);
                        self.a.mov(Gpr(4), PC);
                        self.a.mov(Gpr(5), INSN);
                        self.a.mov(Gpr(2), BASE);
                        self.a.movz(Sz::X, Gpr(3), a as u16, 0);
                        self.a.movz(Sz::X, Gpr(6), c as u16, 0);
                        self.a.bl_far(self.helpers.land);
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
            CallMm {
                hdr,
                nargs,
                pc_after,
            } => {
                let h = hdr as i32 * 8;
                for k in 0..=nargs as usize {
                    let v = self.r(i, k);
                    let at = if k == 0 { h } else { h + 24 + 8 * k as i32 };
                    self.a.str(v, BASE, at);
                }
                self.a.movz(Sz::X, INSN, nargs as u16, 0);
                let (slow, native) = (self.a.new_label(), self.a.new_label());
                let (ret, deep) = self.fast_call(h, block, slow, native);
                self.miss_path(h, block, slow);
                self.deep_path(deep, block, false, None);
                let mut keys = vec![(ret, false)];
                keys.extend(self.native_path(h, block, Ok((native, slow))));
                self.a.bind(slow);
                let back = self.level_call(h, block, pc_after);
                let fast = self.fast_resume[self.v.succs(block)[0].index()];
                keys.extend([(fast, false), (back, true)]);
                self.call_site(keys, i, block, hdr, pc_after);
            }
            GuardFits { end, exit } => {
                self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
                self.a.add_imm(Sz::X, X17, BASE, end as u64 * 8);
                self.a.cmp(Sz::X, X17, X16);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Hi, l);
            }
            Return { a, n, x0 } => {
                // A frame a compiled caller entered holds that caller's
                // return address, one dispatch entered its level's.
                let slow = self.a.new_label();
                if n == 0 {
                    self.a.mov_imm(Gpr(0), NIL);
                } else if !x0 {
                    self.a.ldr(Gpr(0), BASE, a as i32 * 8);
                }
                self.close_frame();
                self.a.movz(Sz::X, INSN, n as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, a as u64 * 8);
                self.a.ldr(X17, RT, layout::LEVEL_RET as i32);
                self.a.cmp(Sz::X, LR, X17);
                self.a.b_cond(Cond::Eq, slow);
                let deep = self.a.new_label();
                self.a.tbnz(LR, 0, deep);
                self.a.ret();
                self.a.bind(deep);
                self.a.and_imm(Sz::X, LR, LR, !1u64);
                self.a.br(LR);
                self.a.bind(slow);
                self.a.ldr(X16, BASE, -24);
                self.a.and_imm(Sz::X, X16, X16, !31u64);
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
        // Unsigned: `-i64::MIN` is still negative.
        self.a.neg(Sz::X, cnt, cnt);
        self.a.cmp_imm(Sz::X, cnt, 64);
        self.a.b_cond(Cond::Hs, zero);
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
