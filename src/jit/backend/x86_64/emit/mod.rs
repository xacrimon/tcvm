//! Allocated `MInst`s to machine code: the native frame, the allocator's moves,
//! every instruction's sequence, the exit stubs and the region's exit
//! trampoline in the cold section.

use crate::jit::backend::emit::{
    Emitted, FastEntry, Links, Origin, SiteOut, exit_out, rotate_loops, thread_jumps,
};
use crate::jit::backend::lower::{Lowered, Tramp};
use crate::jit::backend::ra::{Edit, InstOrEdit, Output};
use crate::jit::backend::reg::{Allocation, Block, Inst, PReg, RegClass};
use crate::jit::backend::vcode::{Remat, VCode};
use crate::jit::backend::x86_64::abi::{ARGS, BASE, MAX_SPILLS, THREAD};
use crate::jit::backend::x86_64::asm::{
    Arith, Asm, Cond, Gpr, Label, Mem, R8, R9, RAX, RBP, RCX, RDI, RDX, RSI, RSP, Sz, Xmm,
};
use crate::jit::backend::x86_64::inst::{ENTRY_FAIL, MInst};
use crate::jit::layout;

/// The address of a routine the code branches to.
macro_rules! addr {
    ($f:path) => {
        $f as *const () as usize
    };
}

mod call;
mod entry;
mod inst;

type OolPath<'a> = Box<dyn FnOnce(&mut Em<'a>) + 'a>;

/// A resume block's labels.
#[derive(Clone, Copy)]
struct ResumeLabels {
    /// Where a fast callee's `ret` lands.
    fast: Label,
    /// Its call's miss path (`miss_path`), which its site names
    /// (`abi::call_miss`).
    miss: Label,
    /// Past the fast resume's `BASE`, which a native call keeps and a
    /// level's return restores.
    based: Label,
    /// The landing of a level returning to its call, where its
    /// `GuardLanding` runs out of line.
    boxed: Option<Label>,
}

/// The labels the entries share with their stubs.
struct Entries {
    /// The fast entry, where the region has one.
    fast: Option<Label>,
    /// Past its arity check, where `fixup` comes back with the missing
    /// arguments filled in.
    known: Label,
    fixup: Label,
    /// Its failed checks: on to the caller's slow path, from `param_miss`
    /// with the parameters back in their slots.
    fast_miss: Label,
    param_miss: Label,
    /// The entry guards, where dispatch enters.
    guards: Label,
}

struct Em<'a> {
    a: Asm,
    l: &'a Lowered,
    v: &'a VCode<MInst>,
    out: &'a Output,
    frame: u32,
    labels: Vec<Label>,
    /// Where a branch to each block goes: a block that is only a deopt is
    /// its exit (`thread_jumps`).
    dest: Vec<Label>,
    exit_labels: Vec<Label>,
    exit_region: Label,
    entry_fail: Label,
    links: Links,
    /// The `TCVM_JIT_DEOPT_ALL` flags of the exits, by address.
    fired: Option<usize>,
    /// The exits that mode may take where first reached (`LExit::forced`).
    forced: Vec<bool>,
    /// By block, for the resume blocks.
    resume: Vec<ResumeLabels>,
    /// The code's first word, the region pointer (`Region::of_return`).
    region_word: Label,
    sites: Vec<SiteOut<Label>>,
    /// This region's prototype and its known entries, when it has them:
    /// arguments in slots, and in registers.
    own_entry: Option<(u64, Label, Label)>,
    /// The direct branches to known entries, by the callee's prototype.
    direct: Vec<(Label, u64, Option<Label>)>,
    lazy: Vec<(Label, Label)>,
    /// Where self-calls whose arguments pass the entry guards enter (the
    /// entry block, past the fast entry's tests), checking the window
    /// themselves.
    typed_entry: Option<Label>,
    /// The window a fast entry checks: the frame's, or the region's past it
    /// (`Lowered::window`).
    own_window: u16,
    /// A resume block whose `GuardLanding` follows its `Resume` (with the
    /// call's header slot): its slow resume is bound out of line, unless a
    /// move comes between.
    pending_based: Option<(usize, u16)>,
    landings: Vec<(Label, Label)>,
    /// `tail_level`'s level path, shared (it rewrites typed return
    /// addresses, all known only at the end).
    level_tail: Label,
    /// Code out of the way of the fast paths, emitted after the blocks: a
    /// call's paths besides its fast one, so its callee returns into the
    /// resume block.
    ool: Vec<OolPath<'a>>,
    /// With `trace`: each run of code's origin (`Emitted::origins`).
    origins: Option<Vec<(u32, u32)>>,
}

/// The loads of the first `n` parameters from their slots into `ARGS`.
fn param_loads(a: &mut Asm, n: u8) {
    for k in 0..n.min(ARGS.len() as u8) {
        a.load(Sz::Q, ARGS[k as usize], Mem::B(BASE, 8 * k as i32));
    }
}

/// A fast entry's arity check: on to `fixup` with arguments missing.
fn arity_check(a: &mut Asm, nparams: u8, fixup: Label) {
    a.cmp_imm(
        Sz::D,
        crate::jit::backend::x86_64::abi::INSN,
        nparams as i32,
    );
    a.jcc(Cond::B, fixup);
}

/// A fast entry's known entry, from it, for a caller that proved the arity:
/// past the arity check, and for a call passing the parameters in registers
/// past their loads from the slots.
pub(crate) fn known_offset(regs: bool, nparams: u8) -> usize {
    let mut a = Asm::new();
    let l = a.new_label();
    arity_check(&mut a, nparams, l);
    if regs {
        param_loads(&mut a, nparams);
    }
    a.offset()
}

/// The registers a C call may change that compiled code keeps values in.
const SAVED: [Gpr; 7] = [RAX, RCX, RDX, RSI, RDI, R8, R9];

/// A trampoline: a C call of `target` that keeps every allocatable register but
/// the result's and the caller's argument registers, behind a frame record that
/// keeps the frame chain walkable; and its frame's size.
pub(crate) fn trampoline(target: usize, kind: Tramp) -> (Asm, u32) {
    let ints: Vec<Gpr> = (SAVED.iter().copied())
        .filter(|&r| match kind {
            Tramp::Int => r != RAX,
            Tramp::Float => true,
            Tramp::Args(n) => !ARGS[..n as usize].contains(&r),
        })
        .collect();
    let floats: Vec<Xmm> = (0..15)
        .filter(|&r| kind != Tramp::Float || r != 0)
        .map(Xmm)
        .collect();
    let frame = (8 * (ints.len() + floats.len()) as u32).next_multiple_of(16) as i32;
    let mut a = Asm::new();
    a.push(RBP);
    a.mov(RBP, RSP);
    a.alu_imm(Arith::Sub, Sz::Q, RSP, frame);
    for (k, &r) in ints.iter().enumerate() {
        a.store(Sz::Q, Mem::B(RSP, 8 * k as i32), r);
    }
    for (k, &r) in floats.iter().enumerate() {
        a.fstore(Mem::B(RSP, 8 * (ints.len() + k) as i32), r);
    }
    a.call_far(target);
    for (k, &r) in ints.iter().enumerate() {
        a.load(Sz::Q, r, Mem::B(RSP, 8 * k as i32));
    }
    for (k, &r) in floats.iter().enumerate() {
        a.fload(r, Mem::B(RSP, 8 * (ints.len() + k) as i32));
    }
    a.mov(RSP, RBP);
    a.pop(RBP);
    a.ret();
    (a, frame as u32 + 16)
}

fn gpr(p: PReg) -> Gpr {
    debug_assert_eq!(p.class(), RegClass::Int);
    Gpr(p.hw_enc() as u8)
}

fn xmm(p: PReg) -> Xmm {
    debug_assert_eq!(p.class(), RegClass::Float);
    Xmm(p.hw_enc() as u8)
}

/// A spill slot, below the frame pointer (`abi::spill_addr`).
fn spill(slot: usize) -> Mem {
    Mem::B(RBP, -8 * (slot as i32 + 1))
}

pub(crate) fn emit(
    l: &Lowered,
    out: &Output,
    links: Links,
    fired: Option<usize>,
    fast: Option<FastEntry>,
    trace: bool,
) -> Result<Emitted, String> {
    if out.num_spillslots > MAX_SPILLS {
        return Err(format!("{} spill slots", out.num_spillslots));
    }
    let (mut em, skip) = Em::new(l, out, links, fired, trace);
    em.origin(Origin::Entry as u32);
    em.a.bind(em.region_word);
    em.a.quad(0);
    let entries = em.entries(fast);
    let (hot, cold) = em.blocks(&skip)?;
    // Each stub anew: an edge's moves inside one set the origin to theirs.
    for f in std::mem::take(&mut em.ool) {
        em.origin(Origin::OutOfLine as u32);
        f(&mut em);
    }
    em.origin(Origin::Stubs as u32);
    let dispatch = em.dispatch_entry(entries.guards);
    em.exit_stubs();
    em.entry_stubs(&entries, fast);
    em.level_tail_stub();
    let at = |l: Label| em.a.label_offset(l) as u32;
    let pairs = |v: &[(Label, Label)]| v.iter().map(|&(x, y)| (at(x), at(y))).collect();
    Ok(Emitted {
        entry: at(dispatch),
        fast: entries.fast.map(at),
        num_spills: out.num_spillslots as u32,
        frame_size: em.frame,
        region_word: at(em.region_word),
        exits: l.exits.iter().map(|e| exit_out(l, out, e)).collect(),
        sites: em.sites.into_iter().map(|s| s.placed(at)).collect(),
        direct: (em.direct.iter().map(|&(l, p, s)| (at(l), p, s.map(at)))).collect(),
        lazy: pairs(&em.lazy),
        landings: pairs(&em.landings),
        hot: hot.iter().map(|&(b, l)| (b, at(l))).collect(),
        cold: at(cold),
        origins: em.origins.take().unwrap_or_default(),
        asm: em.a,
    })
}

impl<'a> Em<'a> {
    /// The emitter with its labels made, and the blocks jump threading
    /// leaves out.
    fn new(
        l: &'a Lowered,
        out: &'a Output,
        links: Links,
        fired: Option<usize>,
        trace: bool,
    ) -> (Em<'a>, Vec<bool>) {
        let mut a = Asm::new();
        let nb = l.vcode.blocks.len();
        let labels: Vec<Label> = (0..nb).map(|_| a.new_label()).collect();
        let exit_labels: Vec<Label> = (0..l.exits.len()).map(|_| a.new_label()).collect();
        let entry_fail = a.new_label();
        let exit_of = |exit: u32| {
            if exit == ENTRY_FAIL {
                entry_fail
            } else {
                exit_labels[exit as usize]
            }
        };
        let (dest, skip, resume_to) = thread_jumps(l, out, &labels, exit_of);
        let resume = (resume_to.iter())
            .map(|to| ResumeLabels {
                fast: a.new_label(),
                miss: a.new_label(),
                // Bound in its block, but where its returns go elsewhere
                // (`thread_jumps`).
                based: to.unwrap_or_else(|| a.new_label()),
                boxed: None,
            })
            .collect();
        let em = Em {
            l,
            v: &l.vcode,
            out,
            frame: (8 * out.num_spillslots as u32).next_multiple_of(16),
            labels,
            dest,
            exit_labels,
            exit_region: a.new_label(),
            entry_fail,
            links,
            fired,
            forced: l.exits.iter().map(|e| e.forced).collect(),
            resume,
            region_word: a.new_label(),
            sites: Vec::new(),
            own_entry: None,
            direct: Vec::new(),
            lazy: Vec::new(),
            typed_entry: None,
            own_window: 0,
            pending_based: None,
            landings: Vec::new(),
            level_tail: a.new_label(),
            ool: Vec::new(),
            origins: trace.then(Vec::new),
            a,
        };
        (em, skip)
    }

    /// The blocks but `skip`'s in layout order, hot loops rotated, each with
    /// the allocator's moves. Returns where each hot block starts (by its
    /// IR block) and where the hot ones end.
    fn blocks(&mut self, skip: &[bool]) -> Result<(Vec<(u32, Label)>, Label), String> {
        let l = self.l;
        let mut laid: Vec<usize> = (l.emit_order.iter())
            .map(|&b| b as usize)
            .filter(|&b| !skip[b])
            .collect();
        let mut hot = vec![false; l.vcode.blocks.len()];
        for &b in &l.emit_order[..l.hot_blocks] {
            hot[b as usize] = true;
        }
        let hot_laid = laid.iter().take_while(|&&b| hot[b]).count();
        let tops = rotate_loops(l, &self.labels, &self.dest, &mut laid, hot_laid);
        let cold = self.a.new_label();
        let mut cold_bound = false;
        let mut hot_starts = Vec::new();
        for (at, &b) in laid.iter().enumerate() {
            if !cold_bound && !hot[b] {
                self.a.bind(cold);
                cold_bound = true;
            }
            if !cold_bound {
                hot_starts.push((l.ir_blocks[b], self.labels[b]));
            }
            let blk = &l.vcode.blocks[b];
            // A rotated loop's top, past the jump into its header: padding
            // there never runs.
            if tops.contains(&b) && self.a.after_jump() {
                self.a.align(16);
            }
            self.a.bind(self.labels[b]);
            let insts = self.out.block_insts_and_edits(&l.vcode, Block::new(b));
            for (k, ie) in insts.enumerate() {
                match ie {
                    // A fast callee's `ret` lands at the `Resume`, past anything
                    // before it.
                    InstOrEdit::Edit(_) if blk.resume && k == 0 => {
                        return Err("a move before a call's resume".into());
                    }
                    InstOrEdit::Edit(e) => {
                        if let Some((r, _)) = self.pending_based.take() {
                            self.a.bind(self.resume[r].based);
                        }
                        self.edit(e);
                    }
                    InstOrEdit::Inst(i) => {
                        let next = laid.get(at + 1).map(|&x| self.labels[x]);
                        self.origin(i.index() as u32);
                        self.inst(i.index(), b, next)?;
                    }
                }
            }
        }
        if !cold_bound {
            self.a.bind(cold);
        }
        Ok((hot_starts, cold))
    }
}

impl<'a> Em<'a> {
    fn allocs(&self, i: usize) -> &[Allocation] {
        self.out
            .inst_allocs(self.v, crate::jit::backend::reg::Inst::new(i))
    }

    fn r(&self, i: usize, k: usize) -> Gpr {
        gpr(self.allocs(i)[k].as_reg().expect("a register operand"))
    }

    fn fr(&self, i: usize, k: usize) -> Xmm {
        xmm(self.allocs(i)[k].as_reg().expect("a register operand"))
    }

    fn reg(&self, i: usize, k: usize) -> PReg {
        self.allocs(i)[k].as_reg().expect("a register operand")
    }

    fn origin(&mut self, o: u32) {
        if let Some(v) = &mut self.origins {
            v.push((self.a.offset() as u32, o));
        }
    }

    fn edit(&mut self, e: &Edit) {
        match *e {
            Edit::Move { from, to } => {
                self.origin(Origin::Move as u32);
                self.mov(from, to)
            }
            Edit::Remat { to, vreg, src } => {
                self.origin(Origin::Remat as u32);
                self.remat(to, vreg.vreg(), src)
            }
        }
    }

    /// A branch's targets, with the allocator's moves on its edges: the
    /// edge falling through has them in line after the branch (returned),
    /// any other in a stub out of line that jumps on.
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
            let (stub, to) = (self.a.new_label(), *l);
            *l = stub;
            self.ool.push(Box::new(move |em: &mut Em<'a>| {
                em.a.bind(stub);
                for (_, _, e) in edits {
                    em.edit(e);
                }
                em.a.jmp(to);
            }));
        }
        (labels, inline)
    }

    fn mov(&mut self, from: Allocation, to: Allocation) {
        match (from.as_reg(), to.as_reg(), from.as_stack(), to.as_stack()) {
            (Some(f), Some(t), _, _) => match f.class() {
                RegClass::Int => self.a.mov(gpr(t), gpr(f)),
                _ => self.a.fmov(xmm(t), xmm(f)),
            },
            (Some(f), None, _, Some(s)) => self.word_at(true, f, spill(s.index())),
            (None, Some(t), Some(s), _) => self.word_at(false, t, spill(s.index())),
            _ => panic!("stack-to-stack move"),
        }
    }

    fn fimm(&mut self, d: Xmm, bits: u64) {
        if bits == 0 {
            self.a.fzero(d);
        } else {
            self.a.fload(d, Mem::K(bits));
        }
    }

    /// A value computed again into `to` (`VCode::remat`).
    fn remat(&mut self, to: Allocation, vreg: usize, src: Allocation) {
        let r = to.as_reg().expect("a register to compute into");
        match self.v.remat[vreg] {
            Remat::Int(v) => self.a.mov_imm(gpr(r), v),
            // The transport register on the way to a stack slot
            // (`ra::resolve`): the word.
            Remat::Float(bits) if r.class() == RegClass::Int => self.a.mov_imm(gpr(r), bits),
            Remat::Float(bits) => self.fimm(xmm(r), bits),
            Remat::Slot(slot) => self.word_at(false, r, Mem::B(BASE, slot as i32 * 8)),
            Remat::AndImm { src: sv, imm } => {
                let d = gpr(r);
                match (src.as_reg(), src.as_stack(), self.v.remat[sv as usize]) {
                    (Some(s), _, _) => self.a.mov(d, gpr(s)),
                    (None, Some(slot), _) => self.a.load(Sz::Q, d, spill(slot.index())),
                    (None, None, Remat::Slot(slot)) => {
                        self.a.load(Sz::Q, d, Mem::B(BASE, slot as i32 * 8))
                    }
                    _ => panic!("v{vreg} computed again from nowhere"),
                }
                self.a.alu(Arith::And, Sz::Q, d, Mem::K(imm));
            }
            Remat::None => panic!("v{vreg} is not rematerializable"),
        }
    }

    /// At least `code::JUMP` bytes, which `Region::invalidate` overwrites
    /// at the entry.
    fn open_frame(&mut self) {
        self.a.push(RBP);
        self.a.mov(RBP, RSP);
        if self.frame > 0 {
            self.a.alu_imm(Arith::Sub, Sz::Q, RSP, self.frame as i32);
        } else {
            self.a.nop();
        }
    }

    fn close_frame(&mut self) {
        if self.frame > 0 {
            self.a.mov(RSP, RBP);
        }
        self.a.pop(RBP);
    }

    fn exit(&mut self, exit: u32) -> Label {
        if exit == ENTRY_FAIL {
            return self.entry_fail;
        }
        self.exit_labels[exit as usize]
    }

    /// A value word loaded or stored at `m`: from a general register, or
    /// from an xmm register holding a float's (`Lower::in_d`).
    fn word_at(&mut self, store: bool, t: PReg, m: Mem) {
        match (t.class(), store) {
            (RegClass::Int, false) => self.a.load(Sz::Q, gpr(t), m),
            (RegClass::Int, true) => self.a.store(Sz::Q, m, gpr(t)),
            (_, false) => self.a.fload(xmm(t), m),
            (_, true) => self.a.fstore(m, xmm(t)),
        }
    }

    /// `S0 = BASE + 8 * slots` against the thread's stack end: `A` when
    /// the window does not fit.
    fn window_check(&mut self, slots: u16) {
        let s0 = crate::jit::backend::x86_64::abi::S0;
        self.a.lea(Sz::Q, s0, Mem::B(BASE, slots as i32 * 8));
        self.a
            .cmp(Sz::Q, s0, Mem::B(THREAD, layout::THREAD_STACK_END as i32));
    }
}
