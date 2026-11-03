//! Allocated `MInst`s to machine code (10.5, 10.6): the native frame, the
//! allocator's moves, every instruction's sequence, the exit stubs and the
//! region's exit trampoline in the cold section.

use regalloc2::{Allocation, PReg, RegClass};

use crate::env::function::NativeClosure;
use crate::env::table::SlotLoc;
pub(crate) use crate::env::value::BOX;
use crate::env::value::Value;
use crate::env::value::{
    FALSE, NIL, PAYLOAD_MASK as PTR_MASK, SMALL_INT as SMALL_BITS, TRUE, TYPE_WORDS,
};
use crate::env::{MtCache, Shape};
use crate::jit::backend::aarch64::abi::{
    BASE, CLOSURE, D31, IMAGE_SPILLS, INSN, MAX_SPILLS, PC, RT, THREAD, X16, X17, image_index,
};
use crate::jit::backend::aarch64::asm::{
    Asm, Cond, Extend, FP, Fpr, Gpr, LR, Label, SP, Shift, Sz, ZR,
};
use crate::jit::backend::aarch64::inst::{AluOp, ENTRY_FAIL, FOp, FUnOp, Known, MInst, Test};
use crate::jit::backend::aarch64::lower::{LExit, Lowered, Src};
use crate::jit::backend::ra::{Edit, InstOrEdit, Output};
use crate::jit::backend::vcode::{Remat, VCode};
use crate::jit::ir::ops::Ctx;
use crate::jit::ir::types::TypeSet;
use crate::jit::layout;
use crate::jit::layout::FUNCTION_TAG;
use crate::jit::region::{Loc, SRep, SnapEntry};
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
    /// The direct branches to other regions' known entries, by the callee's
    /// prototype (its box), which `jit::set_fast` repoints: to the register
    /// entry for a call passing arguments there, else the slot one.
    pub(crate) direct: Vec<(Label, u64, bool)>,
    /// What `Region::invalidate` patches: each place and the branch target
    /// written there, the entry's first (ws2 §5.4).
    pub(crate) lazy: Vec<(Label, Label)>,
    /// A self-call's return addresses whose result guard runs out of line,
    /// and its landing for a level returning there (`jit::jit_exit`).
    pub(crate) landings: Vec<(Label, Label)>,
    /// Where each hot block starts, in layout order, and where the hot ones
    /// end: deopt blocks, cold arms and stubs after (the asm dump's
    /// `blocks` and `cold`).
    pub(crate) hot: Vec<(u32, Label)>,
    pub(crate) cold: Label,
    /// With `trace`: where each instruction's words start (word index) and
    /// what it is, a VCode instruction's index or one of `Origin`'s.
    pub(crate) origins: Vec<(u32, u32)>,
}

/// Code that is no VCode instruction, for `Emitted::origins`.
#[repr(u32)]
pub(crate) enum Origin {
    Move = u32::MAX - 4,
    Remat,
    Entry,
    OutOfLine,
    Stubs,
}

/// A call of the region: its return addresses, whether each is a level's
/// (`jit_level`), the values the frame keeps in spill slots across it, its
/// header slot and the `pc` after it.
pub(crate) struct SiteOut {
    pub(crate) keys: Vec<(Label, bool)>,
    /// The call's slow path, where a fast callee that missed goes.
    pub(crate) miss: Label,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
    pub(crate) hdr: u16,
    pub(crate) pc_after: u64,
    /// The inlined frames it is made from (`LSite::frames`).
    pub(crate) frames: Vec<crate::jit::region::SnapFrame>,
}

pub(crate) struct ExitOut {
    pub(crate) pc: u32,
    pub(crate) frames: Vec<crate::jit::region::SnapFrame>,
    pub(crate) site: Option<crate::jit::ir::ExitSite>,
    pub(crate) frame: u32,
    pub(crate) kind: crate::jit::ir::ExitKind,
    pub(crate) tag: crate::jit::ir::ops::ExitTag,
    pub(crate) entries: Vec<SnapEntry>,
    pub(crate) consts: Vec<u64>,
}

/// An instruction with no code: a value left for the allocator to make
/// where used.
fn emits_nothing(l: &Lowered, out: &Output, i: usize) -> bool {
    match l.vcode.insts[i] {
        MInst::MovImm(_) | MInst::FImm(_) | MInst::AluImm(..) => {
            out.inst_allocs(regalloc2::Inst::new(i))[0].is_none()
        }
        _ => false,
    }
}

/// The operand size of a test of one register against zero for equality,
/// which `cbz`/`cbnz` branch on without flags; a B1 is a W.
fn zero_test(test: Test, cond: Cond) -> Option<Sz> {
    if !matches!(cond, Cond::Eq | Cond::Ne) {
        return None;
    }
    match test {
        Test::B1 | Test::I32 { imm: Some(0) } => Some(Sz::W),
        Test::X { imm: Some(0) } => Some(Sz::X),
        _ => None,
    }
}

/// Loop rotation (V8's `ComputeAssemblyOrder`, with any exiting block as the
/// cut): a hot loop whose last block jumps back to its first is laid out
/// from the block after one that branches out to the block following the
/// loop, so that branch becomes the back edge and the jump a fall-through
/// into the header; entering then takes a jump to the header. Returns the
/// rotated loops' new first blocks.
fn rotate_loops(
    l: &Lowered,
    labels: &[Label],
    dest: &[Label],
    laid: &mut [usize],
    hot: usize,
) -> Vec<usize> {
    let mut tops = Vec::new();
    let term = |b: usize| &l.vcode.insts[l.vcode.blocks[b].end as usize - 1];
    for &(first, last) in &l.loops {
        let inside = |b: usize| (first as usize..=last as usize).contains(&b);
        let Some(s) = laid[..hot].iter().position(|&b| inside(b)) else {
            continue;
        };
        let e = s + laid[s..hot].iter().take_while(|&&b| inside(b)).count();
        if e == laid.len() || laid[e..hot].iter().any(|&b| inside(b)) {
            continue;
        }
        let latch = laid[e - 1];
        if !matches!(term(latch), MInst::Jump)
            || dest[l.vcode.succs(latch)[0].index()] != labels[laid[s]]
        {
            continue;
        }
        let after = labels[laid[e]];
        // The cut nearest the latch.
        let cut = (s..e - 1).rev().find(|&k| {
            let b = laid[k];
            if !matches!(term(b), MInst::Br { .. }) {
                return false;
            }
            let ss = l.vcode.succs(b);
            let (t, f) = (dest[ss[0].index()], dest[ss[1].index()]);
            let next = labels[laid[k + 1]];
            (t, f) == (after, next) || (t, f) == (next, after)
        });
        if let Some(k) = cut {
            laid[s..e].rotate_left(k + 1 - s);
            tops.push(laid[s]);
        }
    }
    tops
}

/// Where branches to each block go, whether it is left out, and where a
/// resume block's other returns (`resume_based`) go when not into it.
/// Branches go past a block that is only a `Jump` once allocated (an edge
/// split for moves that needed none), unless the chain is a loop, and to
/// the exit of one that is only a `Deopt`. A resume block that is only the
/// `Resume` and a `Jump`, emitting nothing past its `BASE` fixup, has its
/// slow and native returns go where the jump does.
fn thread_jumps(
    l: &Lowered,
    out: &Output,
    labels: &[Label],
    exit_of: impl Fn(u32) -> Label,
) -> (Vec<Label>, Vec<bool>, Vec<Option<Label>>) {
    let nb = l.vcode.blocks.len();
    // A block's `n` instructions that emit code, if it has no moves.
    let only = |b: usize, n: usize| -> Option<Vec<usize>> {
        if b == 0 {
            return None;
        }
        let insts: Option<Vec<usize>> = out
            .block_insts_and_edits(&l.vcode, regalloc2::Block::new(b))
            .filter(|ie| !matches!(ie, InstOrEdit::Inst(i) if emits_nothing(l, out, i.index())))
            .map(|ie| match ie {
                InstOrEdit::Inst(i) => Some(i.index()),
                InstOrEdit::Edit(_) => None,
            })
            .collect();
        insts.filter(|v| v.len() == n)
    };
    enum End {
        Jump(usize),
        Exit(u32),
    }
    let ends: Vec<Option<End>> = (0..nb)
        .map(|b| {
            if l.vcode.blocks[b].resume {
                return None;
            }
            match l.vcode.insts[only(b, 1)?[0]] {
                MInst::Jump => Some(End::Jump(l.vcode.succs(b)[0].index())),
                MInst::Deopt { exit } => Some(End::Exit(exit)),
                _ => None,
            }
        })
        .collect();
    let mut dest = labels.to_vec();
    let mut skip = vec![false; nb];
    for b in 0..nb {
        let mut t = b;
        let mut n = 0;
        while let Some(End::Jump(s)) = ends[t] {
            t = s;
            n += 1;
            if n > nb {
                t = b;
                break;
            }
        }
        dest[b] = match ends[t] {
            Some(End::Exit(exit)) => exit_of(exit),
            _ => labels[t],
        };
        skip[b] = t != b || matches!(ends[b], Some(End::Exit(_)));
    }
    let resume_to = (0..nb)
        .map(|b| {
            if !l.vcode.blocks[b].resume {
                return None;
            }
            let i = only(b, 2)?;
            let empty = matches!(
                l.vcode.insts[i[0]],
                MInst::Resume {
                    wanted: 0,
                    c: 1,
                    ..
                } | MInst::Resume { wanted: 1, .. }
            );
            (empty && matches!(l.vcode.insts[i[1]], MInst::Jump))
                .then(|| dest[l.vcode.succs(b)[0].index()])
        })
        .collect();
    (dest, skip, resume_to)
}

type OolPath<'a> = Box<dyn FnOnce(&mut Em<'a>) + 'a>;

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
    /// The exits that mode may take where first reached (`LExit::forced`).
    forced: Vec<bool>,
    /// Per resume block, where a fast callee's `ret` lands.
    fast_resume: Vec<Label>,
    /// Per resume block, its call's miss path (`miss_path`), which its site
    /// names (`abi::call_miss`).
    miss: Vec<Label>,
    /// Per resume block, past the fast resume's `BASE`, which
    /// a native call keeps and a level's return restores.
    resume_based: Vec<Label>,
    /// The code's first word, the region pointer (`Region::of_return`).
    region_word: Label,
    sites: Vec<SiteOut>,
    sites_of: &'a [(usize, crate::jit::backend::aarch64::lower::LSite)],
    snap_ops: &'a [u32],
    shape_sets: &'a [Vec<u64>],
    /// This region's prototype and its known entries, when it has them:
    /// arguments in slots, and in registers (W4 §2).
    own_entry: Option<(u64, Label, Label)>,
    /// The direct branches to known entries, by the callee's prototype.
    direct: Vec<(Label, u64, bool)>,
    /// `FMod`'s out-of-line paths: (entry, return, d, a, b).
    fmods: Vec<(Label, Label, Fpr, Fpr, Fpr)>,
    lazy: Vec<(Label, Label)>,
    /// Where self-calls whose arguments pass the entry guards enter (the
    /// entry block, past the fast entry's tests), checking the window
    /// themselves.
    typed_entry: Option<Label>,
    own_max_stack: u8,
    /// A resume block whose `GuardLanding` follows its `Resume` (with the
    /// call's header slot): its slow resume is bound out of line, unless a
    /// move comes between.
    pending_based: Option<(usize, u16)>,
    /// By resume block: the landing of a level returning to its call.
    boxed_of: Vec<Option<Label>>,
    landings: Vec<(Label, Label)>,
    /// `tail_level`'s level path, shared (it rewrites typed return
    /// addresses, all known only at the end).
    level_tail: Label,
    /// Code out of the way of the fast paths, emitted after the blocks: a
    /// call's paths besides its fast one, so its callee returns into the
    /// resume block.
    ool: Vec<OolPath<'a>>,
    /// Bit tests branch with `tbz`/`tbnz` (±32 KB), else `tst` and `b.cond`.
    bit_branches: bool,
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
    /// `abi::native_entry`, `abi::coro_resume` and its siblings by
    /// `coro::Switch`, and `native_act` for what a native asks for besides
    /// a plain return.
    pub(crate) native: usize,
    pub(crate) coro: [usize; 3],
    pub(crate) native_act: usize,
    /// `jit::jit_dead_entry`, where an invalidated region's entry goes.
    pub(crate) dead_entry: usize,
    /// `abi::level_enter` and `abi::call_miss`.
    pub(crate) level_enter: usize,
    pub(crate) call_miss: usize,
    /// The runtime (the handler ABI's x23), and what code reads through it
    /// that stays put for the runtime's life: the arena's `Metrics`, the
    /// type metatables' cells, and the level's return and landing.
    pub(crate) rt: u64,
    pub(crate) metrics: u64,
    pub(crate) type_mt_cells: [u64; 6],
    pub(crate) level_ret: u64,
    pub(crate) level_land: u64,
    /// The trampolines of the helpers cold arms call: (helper, trampoline).
    pub(crate) trampolines: Vec<(usize, usize)>,
    /// Known callees' prototypes (their boxes), their parameter counts and
    /// fast entries now (0 for none; `jit::direct_target`).
    pub(crate) direct: Vec<(u64, u8, usize)>,
}

/// Where a known call enters its callee (D2).
#[derive(Clone, Copy)]
enum Target {
    /// This region's own entry.
    Own(Label),
    /// A known entry of the prototype's fast entry, by a branch that
    /// `jit::set_fast` repoints (`Emitted::direct`): the prototype's box,
    /// the branch's first target, and whether it is the register entry.
    Direct(u64, usize, bool),
    /// The fast entry in x17.
    Reg,
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

/// The frame: the frame record, then the spill slots.
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
    trace: bool,
) -> Result<Emitted, String> {
    let e = emit_with(l, out, helpers.clone(), fired, fast, trace, true)?;
    if e.asm.short_branches_reach() {
        return Ok(e);
    }
    // A region past `tbz`'s reach: bit tests as `tst` and `b.cond`.
    emit_with(l, out, helpers, fired, fast, trace, false)
}

fn emit_with(
    l: &Lowered,
    out: &Output,
    helpers: Helpers,
    fired: Option<usize>,
    fast: Option<FastEntry>,
    trace: bool,
    bit_branches: bool,
) -> Result<Emitted, String> {
    let mut origins: Vec<(u32, u32)> = Vec::new();
    let mut origin = |a: &Asm, o: u32| {
        if trace {
            origins.push(((a.offset() / 4) as u32, o));
        }
    };
    if out.num_spillslots > MAX_SPILLS {
        return Err(format!("{} spill slots", out.num_spillslots));
    }
    let frame = (FRAME_SPILLS as u32 + 8 * out.num_spillslots as u32).next_multiple_of(16);
    let mut a = Asm::new();
    let labels: Vec<Label> = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let exit_labels: Vec<Label> = (0..l.exits.len()).map(|_| a.new_label()).collect();
    let exit_region = a.new_label();
    let entry_fail = a.new_label();
    let exit_of = |exit: u32| {
        if exit == ENTRY_FAIL {
            entry_fail
        } else {
            exit_labels[exit as usize]
        }
    };
    let (dest, skip, resume_to) = thread_jumps(l, out, &labels, exit_of);
    let fast_resume = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    let miss = (0..l.vcode.blocks.len()).map(|_| a.new_label()).collect();
    // Bound in their block, but where its returns go elsewhere (`thread_jumps`).
    let resume_based = (resume_to.iter())
        .map(|t| t.unwrap_or_else(|| a.new_label()))
        .collect();
    let region_word = a.new_label();
    let level_tail = a.new_label();
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
        forced: l.exits.iter().map(|e| e.forced).collect(),
        fast_resume,
        miss,
        resume_based,
        region_word,
        sites: Vec::new(),
        sites_of: &l.sites,
        snap_ops: &l.snap_ops,
        shape_sets: &l.shape_sets,
        fmods: Vec::new(),
        own_entry: None,
        direct: Vec::new(),
        lazy: Vec::new(),
        typed_entry: None,
        own_max_stack: 0,
        pending_based: None,
        boxed_of: vec![None; l.vcode.blocks.len()],
        landings: Vec::new(),
        level_tail,
        ool: Vec::new(),
        bit_branches,
    };
    origin(&em.a, Origin::Entry as u32);
    em.a.bind(em.region_word);
    em.a.emit(0);
    em.a.emit(0);
    // A compiled caller has written the header, set base, closure and `insn`
    // (the argument count) as `enter` would, and its `blr` left the return
    // address in x30; what `enter` would check of a Lua callee is checked
    // here, and a miss goes to the caller's slow path (`fast_miss`). Missing
    // arguments are filled in (`arity_fixup`, I3).
    let fast_miss = em.a.new_label();
    let (fixup, budget) = (em.a.new_label(), em.a.new_label());
    let nreg = l.params;
    let param_miss = em.a.new_label();
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
        let (known, regs) = (em.a.new_label(), em.a.new_label());
        em.a.bind(known);
        assert_eq!(
            (em.a.label_offset(known) - em.a.label_offset(l)) as u64,
            KNOWN_ENTRY
        );
        // The parameters a caller left in slots to their registers, where
        // a call passing them there enters (W4 §2).
        for k in 0..nreg {
            em.a.ldr(Gpr(k), BASE, 8 * k as i32);
        }
        em.a.bind(regs);
        assert_eq!(
            (em.a.label_offset(regs) - em.a.label_offset(l)) as usize,
            crate::jit::known_offset(true, fe.num_params)
        );
        em.own_entry = Some((fe.proto, known, regs));
        em.own_max_stack = fe.max_stack;
        em.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
        em.a.add_imm(Sz::X, X17, BASE, fe.max_stack as u64 * 8);
        em.a.cmp(Sz::X, X17, X16);
        // A miss from the register entry leaves the parameters in their
        // slots for the caller's slow path (from the slot one, again).
        em.a.b_cond(Cond::Hi, if nreg > 0 { param_miss } else { fast_miss });
        l
    });
    // The entry guards (on `x0`..), then the entry block, where typed
    // self-calls enter; dispatch enters at the guards.
    let guards = em.a.new_label();
    em.a.bind(guards);
    for &(r, set) in &l.entry_tests {
        type_fail(&mut em.a, Gpr(r), set, em.entry_fail, &[]);
    }
    if l.prefix_guards {
        em.typed_entry = Some(em.labels[0]);
    }
    let mut laid: Vec<usize> = l
        .emit_order
        .iter()
        .map(|&b| b as usize)
        .filter(|&b| !skip[b])
        .collect();
    let mut hot = vec![false; l.vcode.blocks.len()];
    for &b in &l.emit_order[..l.hot_blocks] {
        hot[b as usize] = true;
    }
    let hot_laid = laid.iter().take_while(|&&b| hot[b]).count();
    let tops = rotate_loops(l, &em.labels, &em.dest, &mut laid, hot_laid);
    let cold = em.a.new_label();
    let mut cold_bound = false;
    let mut hot_starts = Vec::new();
    for (at, &b) in laid.iter().enumerate() {
        if !cold_bound && !hot[b] {
            em.a.bind(cold);
            cold_bound = true;
        }
        if !cold_bound {
            hot_starts.push((l.ir_blocks[b], em.labels[b]));
        }
        let blk = &l.vcode.blocks[b];
        // A rotated loop's top, past the jump into its header: padding
        // there never runs (V8 aligns these too; 16 measured as well as 64).
        if tops.contains(&b) && em.a.after_jump() {
            em.a.align(16);
        }
        em.a.bind(em.labels[b]);
        for (k, ie) in out
            .block_insts_and_edits(&l.vcode, regalloc2::Block::new(b))
            .enumerate()
        {
            match ie {
                // A fast callee's `ret` lands at the `Resume`, past anything
                // before it.
                InstOrEdit::Edit(_) if blk.resume && k == 0 => {
                    return Err("a move before a call's resume".into());
                }
                InstOrEdit::Edit(e) => {
                    if let Some((r, _)) = em.pending_based.take() {
                        em.a.bind(em.resume_based[r]);
                    }
                    match *e {
                        Edit::Move { from, to } => {
                            origin(&em.a, Origin::Move as u32);
                            em.mov(from, to)
                        }
                        Edit::Remat { to, vreg, src } => {
                            origin(&em.a, Origin::Remat as u32);
                            em.remat(to, vreg.vreg(), src)
                        }
                    }
                }
                InstOrEdit::Inst(i) => {
                    let next = laid.get(at + 1).map(|&x| em.labels[x]);
                    origin(&em.a, i.index() as u32);
                    em.inst(i.index(), b, next)?;
                }
            }
        }
    }
    if !cold_bound {
        em.a.bind(cold);
    }
    origin(&em.a, Origin::OutOfLine as u32);
    for f in std::mem::take(&mut em.ool) {
        f(&mut em);
    }
    origin(&em.a, Origin::Stubs as u32);
    // Dispatch enters through `level_enter`, which returns to `level_land`.
    let dispatch = em.a.new_label();
    em.a.bind(dispatch);
    for k in 0..l.params {
        em.a.ldr(Gpr(k), BASE, 8 * k as i32);
    }
    em.a.adr(X16, guards);
    em.a.b_far(em.helpers.level_enter);
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
    if l.invalidable {
        // An invalidated region's entry, its frame not open yet: on to
        // `jit_dead_entry` as the entry-fail stub goes on (ws2 §5.4), the
        // parameters a register call passed stored for the interpreter.
        let dead = em.a.new_label();
        em.a.bind(dead);
        for k in 0..l.params {
            em.a.str(Gpr(k), BASE, 8 * k as i32);
        }
        em.a.ldr_label(INSN, region_word);
        em.a.ldr_lit(X16, em.helpers.dead_entry as u64);
        em.tail_level();
        em.lazy.insert(0, (em.labels[0], dead));
    }
    if !l.entry_regs.is_empty() {
        // The interpreter runs the function from its slots, which a call
        // passing the parameters in registers did not write.
        em.a.bind(em.entry_fail);
        for k in 0..l.params {
            em.a.str(Gpr(k), BASE, 8 * k as i32);
        }
        if !l.prefix_guards {
            em.close_frame();
        }
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
    if fast_label.is_some() && nreg > 0 {
        em.a.bind(param_miss);
        for k in 0..nreg {
            em.a.str(Gpr(k), BASE, 8 * k as i32);
        }
        em.a.b(fast_miss);
    }
    if fast_label.is_some() {
        // Back to the caller's slow path, which its site names by the return
        // address (`abi::call_miss`). A tail call from a frame dispatch
        // entered has `level_land` for one instead: `enter` runs the call
        // from the header the tail call wrote.
        em.a.bind(fast_miss);
        let dispatched = em.a.new_label();
        em.a.ldr_lit(X17, em.helpers.level_land);
        em.a.cmp(Sz::X, LR, X17);
        em.a.b_cond(Cond::Eq, dispatched);
        em.a.b_far(em.helpers.call_miss);
        em.a.bind(dispatched);
        // The tail call left its callee in `CLOSURE`.
        em.a.movz(Sz::X, X16, FUNCTION_TAG, 3);
        em.a.orr(Sz::X, X16, X16, CLOSURE);
        em.a.str(X16, BASE, -32);
        em.a.sub_imm(Sz::X, PC, BASE, 32);
        em.a.mov(BASE, ZR);
        em.a.ldr_lit(RT, em.helpers.rt);
        em.a.ldr_lit(LR, em.helpers.level_ret);
        em.a.b_far(em.helpers.enter);
    }
    em.level_tail_stub();
    let exits = l
        .exits
        .iter()
        .enumerate()
        .map(|(k, e)| exit_out(l, out, e, k))
        .collect();
    let entry = dispatch;
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
        direct: em.direct,
        lazy: em.lazy,
        landings: em.landings,
        hot: hot_starts,
        cold,
        origins,
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
        frame: e.frame,
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

    /// The registers of instruction `i`'s words for a test of `set` whose
    /// value is operand `k`, where it has them (`Test::Type`).
    fn words(&self, i: usize, k: usize, set: TypeSet, has: bool) -> Vec<Gpr> {
        if !has {
            return Vec::new();
        }
        (0..test_words(set).len())
            .map(|j| self.r(i, k + 1 + j))
            .collect()
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

    fn fimm(&mut self, d: Fpr, bits: u64) {
        if bits == 0 {
            self.a.fmov_from_gpr(d, ZR);
        } else if let Some(imm8) = Asm::fmov_imm8(f64::from_bits(bits)) {
            self.a.fmov_imm(d, imm8);
        } else {
            self.a.ldr_lit_f(d, bits);
        }
    }

    /// A value computed again into `to` (`VCode::remat`).
    fn remat(&mut self, to: Allocation, vreg: usize, src: Allocation) {
        let r = to.as_reg().expect("a register to compute into");
        match self.v.remat[vreg] {
            Remat::Int(v) => self.a.mov_imm(gpr(r), v),
            Remat::Float(bits) => self.fimm(fpr(r), bits),
            Remat::AndImm { imm, .. } => {
                let s = match (src.as_reg(), src.as_stack()) {
                    (Some(s), _) => gpr(s),
                    (None, Some(slot)) => {
                        self.a.ldr(gpr(r), FP, spill_off(slot.index()));
                        gpr(r)
                    }
                    _ => panic!("v{vreg} computed again from nowhere"),
                };
                self.a.and_imm(Sz::X, gpr(r), s, imm);
            }
            Remat::None => panic!("v{vreg} is not rematerializable"),
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

    /// The fast path of a call whose header is at `base + hdr`, `insn` the
    /// argument count: a Lua callee without varargs whose prototype has a
    /// fast entry is called with `blr`, this frame kept, and its `ret` lands
    /// past the resume block's frame push (5.4); a native goes to `native`
    /// (`native_path`), anything else to `slow`. Every register is free
    /// here: what lives across the call is in the frame. Returns
    /// `fast_enter`'s labels.
    fn fast_call(&mut self, hdr: i32, slow: Label, native: Label) -> (Label, Label) {
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
        self.fast_enter(hdr, Target::Reg, None)
    }

    /// `fast_call` for a known callee (D2), word 0 its closure word: the
    /// constant `closure`'s, or the stored value's. With a static argument
    /// count it enters past the arity and native stack checks, the
    /// parameters left out stored nil (I3), by a direct branch
    /// (`direct_target`); else it loads the fast entry, `slow` without
    /// one. Returns `fast_enter`'s labels, whether it enters past the
    /// checks, and where.
    fn known_call(
        &mut self,
        hdr: i32,
        slow: Label,
        (closure, proto, nparams): (Option<u64>, u64, u8),
        nargs: u8,
        window: Option<Label>,
    ) -> (Label, Label, bool, Target) {
        // A static count passes the first arguments in x0.. (W4 §2), the
        // parameters left out nil; past eight they are in the callee's slots,
        // inside its window, which its known entry checks (the value stack's
        // mapping runs past any window that fits).
        let known = nargs != u8::MAX;
        if known && nargs < nparams {
            self.a.mov_imm(Gpr(9), NIL);
            for k in nargs..nparams {
                if k < 8 {
                    self.a.mov(Gpr(k), Gpr(9));
                } else {
                    self.a.str(Gpr(9), BASE, hdr + 32 + 8 * k as i32);
                }
            }
        }
        let target = match self.direct_target(proto, nparams, known) {
            // A typed self-call enters past the entry guards (W4 §2).
            Some(_) if window.is_some() => Target::Own(self.typed_entry.unwrap()),
            Some(t) if known => t,
            _ => {
                self.a.mov_imm(X17, proto);
                self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
                // Without a fast entry, the slow path, which finds every
                // argument in the callee's slots.
                let to = if known {
                    let stores = self.a.new_label();
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(stores);
                        em.store_args(hdr + 32, nargs);
                        em.a.b(slow);
                    }));
                    stores
                } else {
                    slow
                };
                self.a.cbz(Sz::X, X17, to);
                if known {
                    let off = crate::jit::known_offset(true, nparams);
                    self.a.add_imm(Sz::X, X17, X17, off as u64);
                }
                Target::Reg
            }
        };
        match closure {
            Some(f) => self.a.mov_imm(X16, f & PTR_MASK),
            None => {
                self.a.ldr(X16, BASE, hdr);
                self.a.and_imm(Sz::X, X16, X16, PTR_MASK);
            }
        }
        let (ret, deep) = self.fast_enter(hdr, target, window);
        (ret, deep, known, target)
    }

    /// A known entry of `proto` (its box, `nparams` parameters) a branch
    /// can take, the register one when `regs`: this region's own, or one
    /// `jit::set_fast` keeps the branch pointed at.
    fn direct_target(&self, proto: u64, nparams: u8, regs: bool) -> Option<Target> {
        match self.own_entry {
            // The region cannot outlive itself (I4).
            Some((p, slots, r)) if p == proto => Some(Target::Own(if regs { r } else { slots })),
            _ => (self.helpers.direct.iter().find(|d| d.0 == proto)).map(|&(p, _, fast)| {
                Target::Direct(p, crate::jit::direct_target(fast, regs, nparams), regs)
            }),
        }
    }

    /// `bl` (`b` unless `link`) to `target`, a direct one recorded for
    /// `jit::set_fast`.
    fn branch_to(&mut self, target: Target, link: bool) {
        match (target, link) {
            (Target::Own(l), true) => self.a.bl(l),
            (Target::Own(l), false) => self.a.b(l),
            (Target::Direct(proto, t, regs), _) => {
                let at = self.a.new_label();
                self.a.bind(at);
                self.direct.push((at, proto, regs));
                if link {
                    self.a.bl_far(t);
                } else {
                    self.a.b_far(t);
                }
            }
            (Target::Reg, true) => self.a.blr(X17),
            (Target::Reg, false) => self.a.br(X17),
        }
    }

    /// Enter `target` (`Target::Reg`: the fast entry in x17) with the
    /// closure in x16 (`fast_call`). Returns the return address the call
    /// leaves, where the fast resume is when the resume block is laid out
    /// next (else a branch to it, `fall_into_resume`), and the deep path's
    /// label, for `deep_path`.
    fn fast_enter(&mut self, hdr: i32, target: Target, window: Option<Label>) -> (Label, Label) {
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
        // The window check the typed entry leaves to its caller.
        if let Some(w) = window {
            self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
            self.a
                .add_imm(Sz::X, X17, BASE, self.own_max_stack as u64 * 8);
            self.a.cmp(Sz::X, X17, X16);
            self.a.b_cond(Cond::Hi, w);
        }
        let deep = self.a.new_label();
        self.a.mov_sp(Gpr(9), SP);
        self.a.ldr(Gpr(10), THREAD, layout::NATIVE_DEEP as i32);
        self.a.cmp(Sz::X, Gpr(9), Gpr(10));
        self.a.b_cond(Cond::Lo, deep);
        self.branch_to(target, true);
        let ret = self.a.new_label();
        self.a.bind(ret);
        (ret, deep)
    }

    /// After a fast call's `bl`: on into the fast resume, which is there
    /// unless the resume block is laid out elsewhere.
    fn fall_into_resume(&mut self, block: usize, next: Option<Label>) {
        let r = self.v.succs(block)[0].index();
        if Some(self.labels[r]) != next {
            self.a.b(self.fast_resume[r]);
        }
    }

    /// Where a fast callee whose entry checks failed goes (its site's
    /// `miss`): the caller's `BASE` and word 0's callee value back, then its
    /// slow path at `slow`.
    fn miss_path(&mut self, hdr: i32, block: usize, slow: Label) {
        self.a.bind(self.miss[self.v.succs(block)[0].index()]);
        self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 + 32);
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
        let mut entry = self.helpers.native;
        let mut switches = false;
        match from {
            Ok((check, slow)) => {
                self.a.bind(check);
                self.a.cmp_imm(Sz::W, X17, 1);
                self.a.b_cond(Cond::Eq, slow);
                self.a.add_imm(Sz::X, Gpr(3), X16, layout::FN_NATIVE as u64);
            }
            Err(nc) => {
                // SAFETY: the region's constants keep the closure alive.
                if let Some(k) =
                    crate::vm::coro::switch_kind(unsafe { &*(nc as *const NativeClosure) })
                {
                    entry = self.helpers.coro[k as usize];
                    switches = true;
                }
                self.a.mov_imm(Gpr(3), nc);
            }
        }
        let win = hdr as u64 + 32;
        self.a.ldr_lit(Gpr(0), self.helpers.rt);
        self.a.add_imm(Sz::X, Gpr(1), BASE, win);
        if from.is_ok() {
            self.a.mov(Gpr(2), INSN);
        }
        let ret = self.a.new_label();
        if switches {
            // A stub that switches returns with a branch (the return
            // address stack's top is the other thread's call), so it is
            // entered with one: neither touches that stack.
            self.a.adr(LR, ret);
            self.a.b_far(entry);
        } else {
            self.a.bl_far(entry);
        }
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
        self.handler_regs();
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
    fn deep_path(
        &mut self,
        deep: Label,
        block: usize,
        known: Option<u8>,
        nargs: u8,
        target: Target,
    ) {
        self.a.bind(deep);
        // A register call misses with its arguments in the callee's slots.
        let miss = match known {
            Some(_) => self.a.new_label(),
            None => self.miss[self.v.succs(block)[0].index()],
        };
        match target {
            Target::Own(l) => self.a.adr(X17, l),
            Target::Direct(proto, _, regs) => {
                // The fast entry, or the miss path without one.
                self.a.mov_imm(X17, proto);
                self.a.ldr(X17, X17, layout::PROTO_FAST as i32);
                self.a.cbz(Sz::X, X17, miss);
                let off = crate::jit::known_offset(regs, known.unwrap_or(0));
                self.a.add_imm(Sz::X, X17, X17, off as u64);
            }
            Target::Reg => {}
        }
        if known.is_some() {
            // The known entry trusts the native stack, which is deep enough
            // here to check: below the limit the callee would miss, and its
            // full entry read arguments from slots this call did not write.
            self.a.ldr(Gpr(10), THREAD, layout::NATIVE_LIMIT as i32);
            self.a.cmp(Sz::X, Gpr(9), Gpr(10));
            self.a.b_cond(Cond::Lo, miss);
        }
        self.a
            .adr(LR, self.fast_resume[self.v.succs(block)[0].index()]);
        self.a.orr_imm(Sz::X, LR, LR, 1);
        self.a.br(X17);
        if known.is_some() {
            self.a.bind(miss);
            self.store_args(0, nargs);
            self.a.b(self.miss[self.v.succs(block)[0].index()]);
        }
    }

    /// A register call's arguments (W4 §2) to the callee's slots, the first
    /// at `off` from `BASE`.
    fn store_args(&mut self, off: i32, nargs: u8) {
        for k in 0..nargs.min(8) {
            self.a.str(Gpr(k), BASE, off + 8 * k as i32);
        }
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
        self.handler_regs();
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
        self.handler_regs();
        self.a.ldr_lit(X17, self.helpers.level_land);
        self.a.cmp(Sz::X, LR, X17);
        self.a.b_cond(Cond::Ne, self.level_tail);
        self.a.ldr_lit(LR, self.helpers.level_ret);
        self.a.br(X16);
    }

    /// `tail_level`'s level, once: a typed return address goes back as its
    /// landing, which tests the result (the level's is any value).
    fn level_tail_stub(&mut self) {
        self.a.bind(self.level_tail);
        if !self.landings.is_empty() {
            let done = self.a.new_label();
            self.a.and_imm(Sz::X, Gpr(10), LR, !1);
            for (ret, boxed) in self.landings.clone() {
                let next = self.a.new_label();
                self.a.adr(Gpr(9), ret);
                self.a.cmp(Sz::X, Gpr(10), Gpr(9));
                self.a.b_cond(Cond::Ne, next);
                self.a.adr(Gpr(9), boxed);
                self.a.and_imm(Sz::X, LR, LR, 1);
                self.a.orr(Sz::X, LR, LR, Gpr(9));
                self.a.b(done);
                self.a.bind(next);
            }
            self.a.bind(done);
        }
        self.a.ldr(Gpr(9), BASE, -24);
        self.a
            .and_imm(Sz::X, Gpr(9), Gpr(9), crate::vm::frame::flag::MASK);
        self.a.ldr_lit(Gpr(10), self.helpers.ret_to_native as u64);
        self.a.orr(Sz::X, Gpr(9), Gpr(9), Gpr(10));
        self.a.str(Gpr(9), BASE, -24);
        self.a.mov(X17, BASE);
        self.a.b_far(self.helpers.level);
    }

    /// The handler ABI's runtime and closure registers, for code going on
    /// into a handler or a level: the frame's closure from its header.
    fn handler_regs(&mut self) {
        self.a.ldr_lit(RT, self.helpers.rt);
        self.a.ldr(CLOSURE, BASE, -32);
        self.a.and_imm(Sz::X, CLOSURE, CLOSURE, PTR_MASK);
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
        let resume = self.v.succs(block)[0].index();
        let frames = (self.sites_of.iter().find(|(b, _)| *b == resume))
            .map_or(Vec::new(), |(_, s)| s.frames.clone());
        let miss = self.miss[resume];
        self.sites.push(SiteOut {
            keys,
            miss,
            entries,
            consts,
            hdr,
            pc_after: pc_after as u64,
            frames,
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

    /// Flags `Eq` when table `t`'s shape is one of `shape_sets[set]`: one
    /// load, then a compare per shape, each after the first only while none
    /// matched (`ccmp`, `Z` set otherwise).
    fn shape_in(&mut self, t: Gpr, set: u32) {
        self.a.ldr(X16, t, layout::T_SHAPE as i32);
        for (n, &word) in self.shape_sets[set as usize].iter().enumerate() {
            self.a.ldr_lit(X17, word);
            if n == 0 {
                self.a.cmp(Sz::X, X16, X17);
            } else {
                self.a.ccmp(Sz::X, X16, X17, 0b0100, Cond::Ne);
            }
        }
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
            Test::Falsy(words) => {
                let v = self.r(i, k);
                let w = self.words(i, k, TypeSet::FALSY, words);
                type_flags(&mut self.a, v, TypeSet::FALSY, &w);
                cond
            }
            Test::B1 => {
                let c = self.r(i, k);
                self.a.cmp_imm(Sz::W, c, 0);
                cond
            }
            Test::Bit(b) => {
                let x = self.r(i, k);
                self.a.tst_imm(Sz::W, x, 1 << b);
                cond
            }
            Test::Shape(word) => {
                let t = self.r(i, k);
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_lit(X17, word);
                self.a.cmp(Sz::X, X16, X17);
                cond
            }
            Test::ShapeIn(set) => {
                let t = self.r(i, k);
                self.shape_in(t, set);
                cond
            }
            // `cond` is `Eq` for in the set, `Ne` for not.
            Test::Type(set, words) => {
                let v = self.r(i, k);
                let w = self.words(i, k, set, words);
                let c = type_flags(&mut self.a, v, set, &w);
                if cond == Cond::Eq { c } else { c.invert() }
            }
        }
    }

    /// Exit unless instruction `i`'s operand 0 is in `set`.
    fn type_guard(&mut self, i: usize, set: TypeSet, exit: u32) {
        let (l, v) = (self.exit(exit), self.r(i, 0));
        type_fail(&mut self.a, v, set, l, &[]);
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
            && self.forced[k as usize]
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
            Params => {}
            // A value the allocator computes again where used.
            MovImm(_) | FImm(_) | AluImm(..) if self.allocs(i)[0].is_none() => {}
            MovImm(v) => {
                let d = self.r(i, 0);
                self.a.mov_imm(d, v);
            }
            FImm(bits) => {
                let d = self.fr(i, 0);
                self.fimm(d, bits);
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
            FrameClosure => {
                let d = self.r(i, 0);
                self.a.ldr(d, BASE, -32);
                self.a.and_imm(Sz::X, d, d, PTR_MASK);
            }
            LoadUpvalOf(k) => {
                let (d, c) = (self.r(i, 0), self.r(i, 1));
                let off = crate::jit::compile::upvalues_offset() + k as usize * 8;
                self.a.ldr(d, c, off as i32);
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
            ShiftImm(sh, sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                match sh {
                    Shift::Lsl => self.a.lsl_imm(sz, d, n, k as u32),
                    Shift::Lsr => self.a.lsr_imm(sz, d, n, k as u32),
                    Shift::Asr => self.a.asr_imm(sz, d, n, k as u32),
                }
            }
            ShiftReg(sh, sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match sh {
                    Shift::Lsl => self.a.lslv(sz, d, n, m),
                    Shift::Lsr => self.a.lsrv(sz, d, n, m),
                    Shift::Asr => self.a.asrv(sz, d, n, m),
                }
            }
            MulAdd { sub, sz } => {
                let (d, n, m, a) = (self.r(i, 0), self.r(i, 1), self.r(i, 2), self.r(i, 3));
                if sub {
                    self.a.msub(sz, d, n, m, a);
                } else {
                    self.a.madd(sz, d, n, m, a);
                }
            }
            UMulShr(k) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.umull(d, n, m);
                self.a.lsr_imm(Sz::X, d, d, k as u32);
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
                minus1,
                exit,
            } => self.divmod(i, div, sz, nonzero, minus1, exit),
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
            // A float inline, a small integer out of line.
            ToF64 { exit, kbox } => {
                let (d, v) = (self.fr(i, 0), self.r(i, 1));
                if kbox {
                    let k = self.r(i, 2);
                    self.a.cmp(Sz::X, v, k);
                } else {
                    self.a.mov_imm(X16, BOX);
                    self.a.cmp(Sz::X, v, X16);
                }
                let (small, done) = (self.a.new_label(), self.a.new_label());
                self.a.b_cond(Cond::Hs, small);
                self.a.fmov_from_gpr(d, v);
                self.a.bind(done);
                let l = self.exit(exit);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(small);
                    em.a.lsr_imm(Sz::X, X16, v, 32);
                    em.a.cmn_imm(Sz::W, X16, 1);
                    em.a.b_cond(Cond::Ne, l);
                    em.a.scvtf(Sz::W, d, v);
                    em.a.b(done);
                }));
            }
            BoxI32 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.orr_imm(Sz::X, d, n, SMALL_BITS);
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
                self.a.orr_imm(Sz::X, Gpr(0), Gpr(1), SMALL_BITS);
                self.a.b(done);
                self.a.bind(slow);
                self.a.ldr_lit(Gpr(0), self.helpers.rt);
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
                if let Test::Type(set, words) = test
                    && !is_single_test(set)
                {
                    // `d` may share the tested value's register: written last.
                    let v = self.r(i, 1);
                    let w = self.words(i, 1, set, words);
                    let (yes, done) = (self.a.new_label(), self.a.new_label());
                    type_branch(&mut self.a, v, set, yes, &w);
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
            Select { test, cond, float } => {
                let c = self.test(i, 3, test, cond);
                if float {
                    let (d, x, y) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2));
                    self.a.fcsel(d, x, y, c);
                } else {
                    let (d, x, y) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                    self.a.csel(Sz::X, d, x, y, c);
                }
            }
            Guard { test, cond, exit } => {
                let l = self.exit(exit);
                if let Test::Bit(b) = test
                    && self.bit_branches
                {
                    // Passes where `cond` holds (`Ne`: the bit set).
                    let x = self.r(i, 0);
                    if cond == Cond::Ne {
                        self.a.tbz(x, b as u32, l);
                    } else {
                        self.a.tbnz(x, b as u32, l);
                    }
                } else if let Some(sz) = zero_test(test, cond) {
                    // Passes where `cond` holds: out where it does not.
                    let c = self.r(i, 0);
                    if cond == Cond::Ne {
                        self.a.cbz(sz, c, l);
                    } else {
                        self.a.cbnz(sz, c, l);
                    }
                } else if let Test::Type(set, words) = test
                    && !is_single_test(set)
                {
                    let v = self.r(i, 0);
                    let w = self.words(i, 0, set, words);
                    if cond == Cond::Eq {
                        type_fail(&mut self.a, v, set, l, &w);
                    } else {
                        type_branch(&mut self.a, v, set, l, &w);
                    }
                } else {
                    let c = self.test(i, 0, test, cond);
                    self.a.b_cond(c.invert(), l);
                }
            }
            GuardLanding { set, exit } => match self.pending_based.take() {
                // The fast return skips it: a self-call returns what its
                // returns give. A level returning to the call (its callee's
                // `BASE`, `jit_exit`) and the slow path land on it.
                Some((r, hdr)) if r == block => {
                    let (cont, boxed) = (self.a.new_label(), self.a.new_label());
                    self.boxed_of[block] = Some(boxed);
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(boxed);
                        em.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 * 8 + 32);
                        em.a.bind(em.resume_based[block]);
                        em.type_guard(i, set, exit);
                        em.a.b(cont);
                    }));
                    self.a.bind(cont);
                }
                _ => self.type_guard(i, set, exit),
            },
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
                self.a.ldr_lit(
                    X16,
                    self.helpers.type_mt_cells[layout::NUMBER_MT_SLOT as usize],
                );
                self.a.ldr(X16, X16, layout::NUMBER_MT as i32);
                self.a.cmp_imm(Sz::X, X16, 0);
                self.a.cset(Sz::W, d, Cond::Ne);
            }
            TypeMtIs { slot, word } => {
                let d = self.r(i, 0);
                self.a
                    .ldr_lit(X16, self.helpers.type_mt_cells[slot as usize]);
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
            GuardShapeIn { set, exit } => {
                let t = self.r(i, 0);
                self.shape_in(t, set);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
            }
            GuardProto { proto, fun, exit } => {
                let v = self.r(i, 0);
                let l = self.exit(exit);
                if !fun {
                    self.a.lsr_imm(Sz::X, X16, v, 48);
                    self.a.movz(Sz::W, X17, FUNCTION_TAG, 0);
                    self.a.cmp(Sz::W, X16, X17);
                    self.a.b_cond(Cond::Ne, l);
                }
                // A native's bytes past its variant may be anything: the
                // tag byte first (`FN_IS_VARARG`, 0 for a fixed Lua closure).
                self.a.and_imm(Sz::X, X16, v, PTR_MASK);
                self.a.ldrb(X17, X16, layout::FN_IS_VARARG as i32);
                self.a.cbnz(Sz::W, X17, l);
                self.a.ldr(X16, X16, layout::FN_PROTO as i32);
                self.a.ldr_lit(X17, proto);
                self.a.cmp(Sz::X, X16, X17);
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
                self.a.ldr_lit(Gpr(0), self.helpers.rt);
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
                        Ctx::Rt => self.a.ldr_lit(Gpr(k as u8), self.helpers.rt),
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
                let multi = matches!(test, Test::Type(set, _) if !is_single_test(set));
                if test == Test::B1 || multi {
                    // Where `cond` holds, and where it does not.
                    let (yes, no) = if cond == Cond::Eq || (test == Test::B1 && cond == Cond::Ne) {
                        (lt, lf)
                    } else {
                        (lf, lt)
                    };
                    let v = self.r(i, 0);
                    let w = match test {
                        Test::Type(set, words) => self.words(i, 0, set, words),
                        _ => Vec::new(),
                    };
                    if Some(yes) == next {
                        match test {
                            Test::Type(set, _) => type_fail(&mut self.a, v, set, no, &w),
                            _ => self.a.cbz(Sz::W, v, no),
                        }
                        return Ok(());
                    }
                    match test {
                        Test::Type(set, _) => type_branch(&mut self.a, v, set, yes, &w),
                        _ => self.a.cbnz(Sz::W, v, yes),
                    }
                    if Some(no) != next {
                        self.a.b(no);
                    }
                    return Ok(());
                }
                if let Test::Bit(b) = test
                    && self.bit_branches
                {
                    let x = self.r(i, 0);
                    let (set, clear) = if cond == Cond::Ne { (lt, lf) } else { (lf, lt) };
                    if Some(set) == next {
                        self.a.tbz(x, b as u32, clear);
                    } else {
                        self.a.tbnz(x, b as u32, set);
                        if Some(clear) != next {
                            self.a.b(clear);
                        }
                    }
                    return Ok(());
                }
                if let Some(sz) = zero_test(test, cond) {
                    // `cbz` where the operand is zero (`Eq` holding).
                    let v = self.r(i, 0);
                    let (on_zero, on_other) = if cond == Cond::Eq { (lt, lf) } else { (lf, lt) };
                    if Some(on_zero) == next {
                        self.a.cbnz(sz, v, on_other);
                    } else {
                        self.a.cbz(sz, v, on_zero);
                        if Some(on_other) != next {
                            self.a.b(on_other);
                        }
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
                typed,
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
                    // No fast callee misses to a native's site.
                    self.a.bind(self.miss[self.v.succs(block)[0].index()]);
                    self.a.brk(0xf2);
                    self.call_site(keys, i, block, a as u16, pc_after);
                    return Ok(());
                }
                let (slow, common) = (self.a.new_label(), self.a.new_label());
                let native = self.a.new_label();
                let window = (typed && self.typed_entry.is_some()).then(|| self.a.new_label());
                let (ret, deep, entered_known, target) = match known {
                    Some(Known::Lua {
                        closure,
                        proto,
                        nparams,
                    }) => self.known_call(hdr, slow, (closure, proto, nparams), nargs, window),
                    _ => {
                        let (r, d) = self.fast_call(hdr, slow, native);
                        (r, d, false, Target::Reg)
                    }
                };
                self.fall_into_resume(block, next);
                let nparams = match known {
                    Some(Known::Lua { nparams, .. }) if entered_known => Some(nparams),
                    _ => None,
                };
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    let resume = em.v.succs(block)[0].index();
                    if let Some(w) = window {
                        // The callee's window does not fit: its slow path
                        // takes the arguments from its slots.
                        em.a.bind(w);
                        em.store_args(0, nargs);
                        em.a.b(em.miss[resume]);
                    }
                    em.miss_path(hdr, block, common);
                    em.deep_path(deep, block, nparams, nargs, target);
                    let mut keys = vec![(ret, false)];
                    if known.is_none() {
                        keys.extend(em.native_path(hdr, block, Ok((native, slow))));
                    }
                    em.a.bind(slow);
                    // Only a call through the loaded fast entry branches here
                    // before writing word 0.
                    if let (
                        Some(Known::Lua {
                            closure: Some(f), ..
                        }),
                        Target::Reg,
                    ) = (known, target)
                    {
                        em.a.mov_imm(X16, f);
                        em.a.str(X16, BASE, hdr);
                    }
                    em.a.bind(common);
                    let back = em.level_call(hdr, block, pc_after);
                    let fast = em.fast_resume[resume];
                    keys.extend([(fast, false), (back, true)]);
                    if let Some(boxed) = em.boxed_of[resume] {
                        keys.push((boxed, false));
                        em.landings.extend([(ret, boxed), (fast, boxed)]);
                    }
                    em.call_site(keys, i, block, a as u16, pc_after);
                }));
            }
            Dispatch { op, word, pc_after } => {
                self.close_frame();
                self.a.ldr_lit(INSN, word);
                self.a.ldr_lit(PC, pc_after as u64);
                self.a.ldr_lit(X16, self.helpers.rt);
                self.a.ldr(X16, X16, op as i32 * 8);
                self.tail_level();
            }
            TailEnter { nargs, known } => {
                // The callee is the frame's closure from here, which frees
                // `insn` and `pc` (allocatable in the body) as scratch.
                let f = self.r(i, 0);
                self.a.mov(CLOSURE, f);
                let target = known.and_then(|(proto, nparams)| {
                    Some((self.direct_target(proto, nparams, false)?, nparams))
                });
                if let Some((target, nparams)) = target {
                    // A known callee (D2): the parameters left out nil, then
                    // its entry past the arity and native stack checks, the
                    // stack no deeper than this frame's caller left it.
                    if nargs < nparams {
                        self.a.mov_imm(X16, NIL);
                        for k in nargs..nparams {
                            self.a.str(X16, BASE, 8 * k as i32);
                        }
                    }
                    self.a.ldr(X16, BASE, -32);
                    self.a.and_imm(Sz::X, X16, X16, !crate::vm::frame::PTR_MASK);
                    self.a.orr(Sz::X, X16, X16, CLOSURE);
                    self.a.str(X16, BASE, -32);
                    self.a.movz(Sz::X, INSN, nargs.max(nparams) as u16, 0);
                    self.close_frame();
                    self.branch_to(target, false);
                    return Ok(());
                }
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
                self.a.ldr_lit(X16, self.helpers.rt);
                self.a.ldr_idx(X16, X16, X17, Extend::Lsl, true);
                self.tail_level();
            }
            SetTop { at } => {
                let n = self.r(i, 0);
                self.a.ldr(X16, THREAD, layout::THREAD_STACK as i32);
                self.a.add_imm(Sz::X, X17, BASE, at as u64 * 8);
                self.a.sub(Sz::X, X17, X17, X16);
                self.a.add_shift(Sz::X, X17, n, X17, Shift::Lsr, 3);
                self.a.str(X17, THREAD, layout::THREAD_TOP as i32);
            }
            Resume { wanted, c, a, hdr } => {
                // A fast callee returns here (`fall_into_resume`).
                self.a.bind(self.fast_resume[block]);
                // The callee's base is past the header at `hdr`.
                self.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 * 8 + 32);
                if matches!(self.v.insts.get(i + 1), Some(GuardLanding { .. })) {
                    self.pending_based = Some((block, hdr));
                } else if !(self.v.succs(block).first())
                    .is_some_and(|s| self.resume_based[block] == self.dest[s.index()])
                {
                    // Else `thread_jumps` sent its returns where it jumps.
                    self.a.bind(self.resume_based[block]);
                }
                match wanted {
                    0 if c == 1 => {}
                    0 => {
                        // The results to `R[a..]` (I5): all of them, `top`
                        // past them, or `c - 1` filled out with nil. Up to
                        // three here, more through the helper. They come from
                        // at least a header above `R[a]`, so copying up is
                        // safe, and reading past them stays in the stack.
                        // The first is in x0: a compiled callee returning one
                        // stores none (W4 §4).
                        let (helper, top, done) =
                            (self.a.new_label(), self.a.new_label(), self.a.new_label());
                        let at = a as i32 * 8;
                        if c == 0 {
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
                        } else if c <= 4 {
                            self.a.mov_imm(X17, NIL);
                            for k in 0..(c - 1) as u64 {
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
                        self.a.bind(helper);
                        // The helper reads the first result from memory too.
                        let first = self.a.new_label();
                        self.a.cbz(Sz::X, INSN, first);
                        self.a.str(Gpr(0), PC, 0);
                        self.a.bind(first);
                        // Land through the helper: (rt, thread, base, a, values, nret, c).
                        self.a.ldr_lit(Gpr(0), self.helpers.rt);
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
                let (ret, deep) = self.fast_call(h, slow, native);
                self.fall_into_resume(block, next);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.miss_path(h, block, slow);
                    em.deep_path(deep, block, None, 0, Target::Reg);
                    let mut keys = vec![(ret, false)];
                    keys.extend(em.native_path(h, block, Ok((native, slow))));
                    em.a.bind(slow);
                    let back = em.level_call(h, block, pc_after);
                    let fast = em.fast_resume[em.v.succs(block)[0].index()];
                    keys.extend([(fast, false), (back, true)]);
                    em.call_site(keys, i, block, hdr, pc_after);
                }));
            }
            GuardFits { end, exit } => {
                self.a.ldr(X16, THREAD, layout::THREAD_STACK_END as i32);
                self.a.add_imm(Sz::X, X17, BASE, end as u64 * 8);
                self.a.cmp(Sz::X, X17, X16);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Hi, l);
            }
            Return { a, n, x0 } => {
                // To a compiled caller, or to `level_land` for a frame
                // dispatch entered; a tagged return address is a deep one's
                // (`deep_path`).
                if n == 0 {
                    self.a.mov_imm(Gpr(0), NIL);
                } else if !x0 {
                    self.a.ldr(Gpr(0), BASE, a as i32 * 8);
                }
                self.close_frame();
                self.a.movz(Sz::X, INSN, n as u16, 0);
                self.a.add_imm(Sz::X, PC, BASE, a as u64 * 8);
                let deep = self.a.new_label();
                self.a.tbnz(LR, 0, deep);
                self.a.ret();
                self.a.bind(deep);
                self.a.and_imm(Sz::X, LR, LR, !1u64);
                self.a.br(LR);
            }
            Deopt { exit } => {
                let l = self.exit(exit);
                self.a.b(l);
            }
            LazyExit { exit } => {
                let (at, l) = (self.a.new_label(), self.exit(exit));
                self.a.patch_point(at);
                self.lazy.push((at, l));
            }
        }
        Ok(())
    }

    fn gc_check(&mut self, exit: u32) {
        self.a.ldr_lit(X16, self.helpers.metrics);
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
    fn divmod(&mut self, i: usize, div: bool, sz: Sz, nonzero: bool, minus1: bool, exit: u32) {
        let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
        let l = self.exit(exit);
        if !nonzero {
            self.a.cbz(sz, m, l);
        }
        if div && sz == Sz::W && minus1 {
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

/// The tag of a boxed type alone, its index in `TYPE_WORDS`.
fn single_tag(set: TypeSet) -> Option<u16> {
    (1..=THR_K)
        .find(|&k| TypeSet::BY_WORDS[k] == set)
        .map(|k| k as u16)
}

pub(crate) fn is_single_test(set: TypeSet) -> bool {
    set == TypeSet::SMALL
        || set == TypeSet::FLOAT
        || set == TypeSet::SMALL | TypeSet::FLOAT
        || set == TypeSet::NIL
        || set == TypeSet::FALSY
        || set == TypeSet::BOOL
        || set == TypeSet::FALSE
        || set == TypeSet::TRUE
        || single_tag(set).is_some()
        || kind_runs(set).len() == 1
}

/// The kinds in the order of their words (`TYPE_WORDS`), each with its
/// first word. Nothing has the words between threads and nil or between
/// true and the small integers, so a run of kinds is one range.
const KIND_ORDER: [(TypeSet, u64); 11] = {
    let mut o = [(TypeSet::empty(), 0); 11];
    let mut k = 0;
    while k < 11 {
        o[k] = (TypeSet::BY_WORDS[k], TYPE_WORDS[k].0);
        k += 1;
    }
    o
};
const THR_K: usize = 6;
const TRUE_K: usize = 9;
const SMALL_K: usize = 10;
const _: () = assert!(
    KIND_ORDER[THR_K].0.bits() == TypeSet::THR.bits()
        && KIND_ORDER[TRUE_K].0.bits() == TypeSet::TRUE.bits()
        && KIND_ORDER[SMALL_K].0.bits() == TypeSet::SMALL.bits()
);

/// The register holding word `k` of a test's words (`part_words`), else
/// `value` made in `scratch`.
fn word(asm: &mut Asm, w: &[Gpr], k: usize, scratch: Gpr, value: u64) -> Gpr {
    match w.get(k) {
        Some(&r) => r,
        None => {
            asm.mov_imm(scratch, value);
            scratch
        }
    }
}

/// The words the test of single-test `set` compares with, in the order
/// `type_flags` reads them.
fn part_words(set: TypeSet) -> Vec<u64> {
    if set == TypeSet::FLOAT || set == TypeSet::SMALL | TypeSet::FLOAT {
        vec![BOX]
    } else if set == TypeSet::NIL || set == TypeSet::FALSY {
        vec![NIL]
    } else if set == TypeSet::BOOL || set == TypeSet::FALSE {
        vec![FALSE]
    } else if set == TypeSet::TRUE {
        vec![TRUE]
    } else if set == TypeSet::SMALL || single_tag(set).is_some() {
        Vec::new()
    } else if let [(a, z)] = kind_runs(set)[..] {
        let (lo, end) = (KIND_ORDER[a].1, run_end(z));
        match (a, z) {
            (0, SMALL_K) => Vec::new(),
            (_, SMALL_K) => vec![lo],
            (0, _) => vec![end],
            (_, z) if z <= THR_K => Vec::new(),
            (a, _) if a > THR_K => vec![lo],
            _ => vec![lo, end],
        }
    } else {
        Vec::new()
    }
}

/// The words a test of `set` compares with, part by part: what the IR
/// gives a type test as constant operands (`opt::words`). None where the
/// test needs more than two, which it then makes itself.
pub(crate) fn test_words(set: TypeSet) -> Vec<u64> {
    let w: Vec<u64> = type_parts(set).into_iter().flat_map(part_words).collect();
    if w.len() > 2 { Vec::new() } else { w }
}

/// Flags for a single-test type set; the condition meaning "in the set".
/// `w` holds its words (`part_words`), or nothing for the test to make them.
fn type_flags(asm: &mut Asm, v: Gpr, set: TypeSet, w: &[Gpr]) -> Cond {
    if set == TypeSet::SMALL {
        asm.lsr_imm(Sz::X, X16, v, 32);
        asm.cmn_imm(Sz::W, X16, 1);
        Cond::Eq
    } else if set == TypeSet::FLOAT {
        let k = word(asm, w, 0, X16, BOX);
        asm.cmp(Sz::X, v, k);
        Cond::Lo
    } else if set == TypeSet::SMALL | TypeSet::FLOAT {
        // Below `BOX` (a float), else the small-int high word.
        let k = word(asm, w, 0, X16, BOX);
        asm.cmp(Sz::X, v, k);
        asm.lsr_imm(Sz::X, X17, v, 32);
        asm.ccmn_imm(Sz::W, X17, 1, 0b0100, Cond::Hs);
        Cond::Eq
    } else if set == TypeSet::NIL {
        let k = word(asm, w, 0, X16, NIL);
        asm.cmp(Sz::X, v, k);
        Cond::Eq
    } else if set == TypeSet::FALSY {
        let k = word(asm, w, 0, X16, NIL);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, 2);
        Cond::Lo
    } else if set == TypeSet::BOOL {
        let k = word(asm, w, 0, X16, FALSE);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, 2);
        Cond::Lo
    } else if set == TypeSet::FALSE || set == TypeSet::TRUE {
        let value = if set == TypeSet::FALSE { FALSE } else { TRUE };
        let k = word(asm, w, 0, X16, value);
        asm.cmp(Sz::X, v, k);
        Cond::Eq
    } else if let Some(tag) = single_tag(set) {
        // The high 16 bits, sign extended, are `tag - 8`.
        asm.asr_imm(Sz::X, X16, v, 48);
        asm.cmn_imm(Sz::X, X16, 8 - tag as u64);
        Cond::Eq
    } else if let [(a, z)] = kind_runs(set)[..] {
        run_flags(asm, v, a, z, w)
    } else {
        panic!("type test of {set:?} needs more than one compare")
    }
}

/// The first word past a run ending at kind `z`, the unused words after a
/// thread or `true` taken in.
fn run_end(z: usize) -> u64 {
    match z {
        THR_K => NIL,
        TRUE_K => SMALL_BITS,
        SMALL_K => 0,
        _ => KIND_ORDER[z + 1].1,
    }
}

/// Flags for kinds `a..=z` of `KIND_ORDER`, one range of words.
fn run_flags(asm: &mut Asm, v: Gpr, a: usize, z: usize, w: &[Gpr]) -> Cond {
    let lo = KIND_ORDER[a].1;
    if z == SMALL_K {
        if a == 0 {
            asm.cmp(Sz::X, v, v);
            return Cond::Eq;
        }
        let k = word(asm, w, 0, X16, lo);
        asm.cmp(Sz::X, v, k);
        return Cond::Hs;
    }
    let end = run_end(z);
    if a == 0 {
        let k = word(asm, w, 0, X16, end);
        asm.cmp(Sz::X, v, k);
        Cond::Lo
    } else if z <= THR_K {
        // The high 16 bits are `tag - 8`, and the index the tag.
        asm.asr_imm(Sz::X, X16, v, 48);
        asm.add_imm(Sz::X, X16, X16, 8 - a as u64);
        asm.cmp_imm(Sz::X, X16, (z - a) as u64);
        Cond::Ls
    } else if a > THR_K {
        let k = word(asm, w, 0, X16, lo);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, (z - a) as u64);
        Cond::Ls
    } else {
        let k = word(asm, w, 0, X16, lo);
        asm.cmp(Sz::X, v, k);
        let e = word(asm, w, 1, X17, end);
        asm.ccmp(Sz::X, v, e, 0b0010, Cond::Hs);
        Cond::Lo
    }
}

/// `set`'s parts with the slices of `w` (`test_words`, or nothing) they read.
fn parts_with_words(set: TypeSet, w: &[Gpr]) -> Vec<(TypeSet, &[Gpr])> {
    let mut at = 0;
    type_parts(set)
        .into_iter()
        .map(|p| {
            let n = if w.is_empty() { 0 } else { part_words(p).len() };
            let s = &w[at..at + n];
            at += n;
            (p, s)
        })
        .collect()
}

/// Branch to `ok` when `v` is in `set`, falling through otherwise.
fn type_branch(asm: &mut Asm, v: Gpr, set: TypeSet, ok: Label, w: &[Gpr]) {
    for (p, pw) in parts_with_words(set, w) {
        let c = type_flags(asm, v, p, pw);
        asm.b_cond(c, ok);
    }
}

/// Go to `fail` unless `v` is in `set`, falling through otherwise.
fn type_fail(asm: &mut Asm, v: Gpr, set: TypeSet, fail: Label, w: &[Gpr]) {
    let parts = parts_with_words(set, w);
    if parts.is_empty() {
        asm.b(fail);
        return;
    }
    let ok = asm.new_label();
    for (k, &(p, pw)) in parts.iter().enumerate() {
        let c = type_flags(asm, v, p, pw);
        if k + 1 < parts.len() {
            asm.b_cond(c, ok);
        } else {
            asm.b_cond(c.invert(), fail);
        }
    }
    asm.bind(ok);
}

/// `set`'s maximal runs of `KIND_ORDER`, first and last index.
fn kind_runs(set: TypeSet) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut k = 0;
    while k < KIND_ORDER.len() {
        if set.contains(KIND_ORDER[k].0) {
            let a = k;
            while k + 1 < KIND_ORDER.len() && set.contains(KIND_ORDER[k + 1].0) {
                k += 1;
            }
            runs.push((a, k));
        }
        k += 1;
    }
    runs
}

/// The single tests `set` takes, in order: its runs, small integers first.
fn type_parts(set: TypeSet) -> Vec<TypeSet> {
    if is_single_test(set) {
        return vec![set];
    }
    let mut runs = kind_runs(set);
    runs.sort_by_key(|&(_, z)| z != SMALL_K);
    runs.into_iter()
        .map(|(a, z)| (KIND_ORDER[a..=z].iter()).fold(TypeSet::empty(), |s, &(k, _)| s | k))
        .collect()
}

#[allow(unused)]
fn unused(_: Value<'_>) {}

#[allow(unused)]
const _: usize = IMAGE_SPILLS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::alloc::CodeAllocator;

    /// Words of each type of `TYPE_WORDS`, with its index: both ends, their
    /// neighbours inside, the middle, and floats' special values.
    fn samples() -> Vec<(u64, usize)> {
        let mut out = Vec::new();
        for (k, &(lo, hi)) in TYPE_WORDS.iter().enumerate() {
            let mid = lo + (hi - lo) / 2;
            for w in [lo, lo.saturating_add(1), mid, hi.saturating_sub(1), hi] {
                if (lo..=hi).contains(&w) && !out.contains(&(w, k)) {
                    out.push((w, k));
                }
            }
        }
        for x in [
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            1.5,
        ] {
            out.push((f64::to_bits(x), 0));
        }
        out
    }

    /// Every set of kinds, tested by `type_branch` and by `type_fail`, its
    /// words made by the test or given in registers, on every sample,
    /// against membership.
    #[test]
    fn type_tests_match_their_sets() {
        const SIZE: usize = 1 << 15;
        let alloc = CodeAllocator::new();
        let block = alloc.reserve(SIZE).expect("code memory");
        let samples = samples();
        for bits in 0u32..1 << TypeSet::BY_WORDS.len() {
            let set = (0..TypeSet::BY_WORDS.len())
                .filter(|&k| bits >> k & 1 != 0)
                .fold(TypeSet::empty(), |s, k| s | TypeSet::BY_WORDS[k]);
            for (fail, in_regs) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut a = Asm::new();
                let other = a.new_label();
                // The words `test_words` names, in x1 and x2, or made by the test.
                let mut w = Vec::new();
                if in_regs {
                    for (j, &word) in test_words(set).iter().enumerate() {
                        a.mov_imm(Gpr(1 + j as u8), word);
                        w.push(Gpr(1 + j as u8));
                    }
                }
                if fail {
                    type_fail(&mut a, Gpr(0), set, other, &w);
                } else {
                    type_branch(&mut a, Gpr(0), set, other, &w);
                }
                a.mov_imm(Gpr(0), fail as u64);
                a.ret();
                a.bind(other);
                a.mov_imm(Gpr(0), !fail as u64);
                a.ret();
                let words = a.finish(block.rx() as usize).expect("assemble");
                block.write(&words);
                let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(block.rx()) };
                for &(w, k) in &samples {
                    let want = set.contains(TypeSet::BY_WORDS[k]);
                    assert_eq!(
                        f(w) != 0,
                        want,
                        "{set:?} ({}, words in registers: {in_regs}) on {w:#x}",
                        if fail { "type_fail" } else { "type_branch" }
                    );
                }
            }
        }
    }
}
