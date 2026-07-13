//! Register allocation, in a box.
//!
//! This module knows nothing about the machine IR and nothing about aarch64. It
//! sees a CFG over virtual registers through [`RegallocFunc`], a set of physical
//! registers through [`MachineEnv`], and hands back an [`Allocation`]. The target
//! supplies the register set; the client supplies the program. Neither is named
//! here, which is what lets a second target be an addition rather than an edit,
//! and what lets the allocator be tested without building any of the rest of the
//! backend (see the tests at the bottom: they construct a CFG directly).
//!
//! The interface is deliberately a subset of [regalloc2]'s and [regalloc3]'s, so
//! that swapping one of those in later is an adapter rather than a redesign:
//!
//!   - the program is a *trait*, as their `Function` is;
//!   - the register set is a *value the target passes in*, as their `MachineEnv` /
//!     `RegInfo` is;
//!   - operands carry [`Constraint`]s, as theirs do;
//!   - the result is indexed **per operand**, not per vreg, and carries a list of
//!     [`Move`] edits the client must insert.
//!
//! That last point is the one that matters and the one that is easy to get wrong.
//! A per-vreg map — "value 7 lives in x3" — is an interface no *splitting*
//! allocator can implement, because in a split allocation value 7 lives in x3 at
//! one instruction and in a stack slot at the next. Neither allocator here splits,
//! so both answer the same location for every mention of a value and emit no
//! edits; the query shape is what survives contact with regalloc3.
//!
//! Two allocators:
//!
//!   - [`spill_everything`]: every value gets a stack slot; the client reloads
//!     operands into scratch registers around each instruction. Terrible code —
//!     and worth keeping, because it is the differential oracle for the one below.
//!     Any program the two disagree on is a register allocation bug.
//!   - [`linear_scan`]: intervals over a linearized CFG, allocated greedily,
//!     spilling the furthest-ending value under pressure.
//!
//! [regalloc2]: https://github.com/bytecodealliance/regalloc2
//! [regalloc3]: https://github.com/Amanieu/regalloc3

use std::collections::HashMap;
use std::fmt;

// --- the vocabulary ---------------------------------------------------------
//
// Owned here rather than in the machine IR, so that this module is the leaf: it
// depends on nothing, and the IR and the encoder depend on it. `mach` re-exports
// `VReg` and `RegClass`, so the rest of the backend does not notice.

/// A virtual register. Holds exactly one machine word.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct VReg(pub u32);

/// Which register file a value lives in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RegClass {
    /// General purpose: integers, pointers, tags, and float *bits* in transit.
    Int,
    /// Floating point.
    Float,
}

impl RegClass {
    pub const ALL: [RegClass; 2] = [RegClass::Int, RegClass::Float];
    pub const COUNT: usize = 2;
}

/// A block in the CFG.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Block(pub u32);

/// An instruction, as an index into the client's flat instruction list.
pub type Inst = usize;

/// A physical register, numbered within its class.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PReg {
    class: RegClass,
    num: u8,
}

impl PReg {
    pub const fn new(class: RegClass, num: u8) -> Self {
        PReg { class, num }
    }

    pub fn class(self) -> RegClass {
        self.class
    }

    /// The register's number *within its class* — what the encoder puts in the
    /// instruction. `PReg(Int, 3)` and `PReg(Float, 3)` are different registers.
    pub fn num(self) -> u8 {
        self.num
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.class {
            RegClass::Int => write!(f, "x{}", self.num),
            RegClass::Float => write!(f, "d{}", self.num),
        }
    }
}

/// Where an operand is allowed to live.
///
/// `Any` and `Reg` are all aarch64 asks for, and all [`linear_scan`] implements;
/// it rejects the other two rather than silently ignoring them. The other two
/// exist because x86 cannot be expressed without them — `shl` wants its count in
/// `cl` ([`Constraint::Fixed`]), and every two-address ALU op wants its
/// destination to be one of its sources ([`Constraint::Reuse`]) — and because
/// getting them into the vocabulary now is what makes the x86 backend an addition
/// rather than a rewrite of this file and its callers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Constraint {
    /// A register or a stack slot, whichever is cheaper.
    Any,
    /// A register. The client cannot reload this one into a scratch itself.
    Reg,
    /// This exact physical register.
    Fixed(PReg),
    /// The same location as use `k` of the same instruction. A def-only
    /// constraint: this is how a two-address instruction is spelled.
    Reuse(usize),
}

/// One mention of a virtual register by an instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Operand {
    pub vreg: VReg,
    pub constraint: Constraint,
}

impl Operand {
    /// The unconstrained mention, which is nearly all of them.
    pub fn any(vreg: VReg) -> Self {
        Operand {
            vreg,
            constraint: Constraint::Any,
        }
    }

    pub fn reg(vreg: VReg) -> Self {
        Operand {
            vreg,
            constraint: Constraint::Reg,
        }
    }

    pub fn fixed(vreg: VReg, preg: PReg) -> Self {
        Operand {
            vreg,
            constraint: Constraint::Fixed(preg),
        }
    }

    pub fn reuse(vreg: VReg, use_idx: usize) -> Self {
        Operand {
            vreg,
            constraint: Constraint::Reuse(use_idx),
        }
    }
}

/// Where a value ended up.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Alloc {
    Reg(PReg),
    /// A slot in the client's frame, indexed in slots rather than bytes — the
    /// client decides how wide a slot is.
    ///
    /// These are *not* Lua stack slots. The Lua stack is the interpreter's, and
    /// holds 16-byte tagged values; these are one machine word and the collector
    /// never scans them — which is sound only because everything they hold is also
    /// reachable from the Lua stack at every point a collection can happen (an
    /// exit stub writes the whole frame back before anything else can run).
    Spill(u32),
}

impl fmt::Display for Alloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Alloc::Reg(r) => write!(f, "{r}"),
            Alloc::Spill(s) => write!(f, "slot{s}"),
        }
    }
}

/// Before or after an instruction — where an edit goes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Pos {
    Before,
    After,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ProgPoint {
    pub inst: Inst,
    pub pos: Pos,
}

impl ProgPoint {
    pub fn before(inst: Inst) -> Self {
        ProgPoint {
            inst,
            pos: Pos::Before,
        }
    }

    pub fn after(inst: Inst) -> Self {
        ProgPoint {
            inst,
            pos: Pos::After,
        }
    }
}

/// A data movement the client must emit at a [`ProgPoint`].
///
/// Neither allocator here produces one — they assign whole values, so a value
/// never has to move. A splitting allocator produces them by the hundred, and the
/// client that cannot insert them cannot host one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Alloc,
    pub to: Alloc,
    pub class: RegClass,
}

// --- what the allocator is shown --------------------------------------------

/// The program, as the allocator sees it: a CFG, and the values each instruction
/// reads and writes.
///
/// A use is read before the instruction executes and a def is written after, but
/// the two are *not* allowed to share a register: a def collides with everything
/// live across the instruction. (That is what [`Constraint::Reuse`] would relax,
/// for a target that needs it.)
pub trait RegallocFunc {
    fn num_blocks(&self) -> usize;
    fn entry(&self) -> Block;
    /// The instructions in this block, in order, as indices.
    fn block_insts(&self, b: Block) -> &[Inst];
    /// Blocks control may reach from the end of `b`.
    fn succs(&self, b: Block) -> Vec<Block>;

    fn num_insts(&self) -> usize;
    fn defs(&self, i: Inst) -> &[Operand];
    fn uses(&self, i: Inst) -> &[Operand];
    /// Registers this instruction destroys without defining. A call site clobbers
    /// the caller-saved set; nothing in this backend clobbers anything yet.
    fn clobbers(&self, _i: Inst) -> &[PReg] {
        &[]
    }

    fn num_vregs(&self) -> usize;
    fn class(&self, v: VReg) -> RegClass;

    /// Reverse postorder from the entry.
    ///
    /// Provided here, and not merely as a convenience: a live interval is a span
    /// of positions in a linearization, so the allocator and whoever lays the
    /// blocks out **must** agree on what that linearization is. Two independent
    /// implementations that happen to agree today are a bug waiting for a CFG
    /// shape nobody has written yet.
    fn block_order(&self) -> Vec<Block> {
        let mut seen = vec![false; self.num_blocks()];
        let mut post = Vec::with_capacity(self.num_blocks());

        // Iterative, because a deeply nested region would blow a recursive stack.
        let entry = self.entry();
        let mut stack = vec![(entry, 0usize)];
        seen[entry.0 as usize] = true;
        while let Some((b, next)) = stack.pop() {
            let succs = self.succs(b);
            if next < succs.len() {
                stack.push((b, next + 1));
                let s = succs[next];
                if !seen[s.0 as usize] {
                    seen[s.0 as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
            }
        }

        post.reverse();
        post
    }
}

/// The target's registers, as the target describes them.
///
/// The allocation order is the target's preference, not the allocator's: a
/// register that is free to use goes first, one that has to be saved goes last.
/// What is *absent* is as load-bearing as what is present — the encoder's scratch
/// registers are absent, and this is the only place that fact is recorded, which
/// is why this value is built next to them rather than here.
pub struct MachineEnv {
    pub allocation_order: [Vec<PReg>; RegClass::COUNT],
}

impl MachineEnv {
    pub fn order(&self, class: RegClass) -> &[PReg] {
        &self.allocation_order[class as usize]
    }
}

/// What [`linear_scan`] declines to do.
///
/// Not "unimplemented" as an apology: an allocator that met a constraint it did
/// not understand and allocated anyway would produce code that runs and is wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegallocError {
    /// A `Fixed` or `Reuse` operand. x86 will need both; whoever writes that
    /// backend implements them here, or hands the whole job to regalloc3.
    UnsupportedConstraint(Constraint),
    /// A clobber. Regions call nothing today, so nothing clobbers.
    UnsupportedClobber(PReg),
}

// --- the result -------------------------------------------------------------

/// Where every operand of every instruction lives.
///
/// Indexed per operand, not per value. See the module comment: this is the shape
/// that a splitting allocator can also fill in.
pub struct Allocation {
    /// Flattened, instruction by instruction: `ndefs[i]` defs followed by the uses.
    allocs: Vec<Alloc>,
    /// Where instruction `i`'s run starts in `allocs`.
    offsets: Vec<u32>,
    ndefs: Vec<u32>,
    /// Sorted by program point.
    edits: Vec<(ProgPoint, Move)>,
    pub num_spills: u32,
}

impl Allocation {
    /// Where def `k` of instruction `i` is written.
    pub fn def(&self, i: Inst, k: usize) -> Alloc {
        debug_assert!(k < self.ndefs[i] as usize, "inst {i} has no def {k}");
        self.allocs[self.offsets[i] as usize + k]
    }

    /// Where use `k` of instruction `i` is read from.
    pub fn use_(&self, i: Inst, k: usize) -> Alloc {
        let base = self.offsets[i] as usize + self.ndefs[i] as usize;
        debug_assert!(
            base + k < self.offsets[i + 1] as usize,
            "inst {i} has no use {k}"
        );
        self.allocs[base + k]
    }

    /// The moves to emit at `p`, in order. Empty for every allocation this module
    /// currently produces.
    pub fn edits_at(&self, p: ProgPoint) -> impl Iterator<Item = &Move> {
        let lo = self.edits.partition_point(|&(q, _)| q < p);
        self.edits[lo..]
            .iter()
            .take_while(move |&&(q, _)| q == p)
            .map(|(_, m)| m)
    }
}

/// Builds an [`Allocation`], per operand.
///
/// Public because the two allocators are not the only ones that will ever want to
/// produce one: an adapter around an external allocator fills the same slots, and
/// so does a test that wants to hand-build a deliberately broken allocation.
pub struct AllocationBuilder {
    allocs: Vec<Alloc>,
    offsets: Vec<u32>,
    ndefs: Vec<u32>,
    /// The value each slot of `allocs` is a mention of, so [`Self::assign`] can
    /// find every mention of a value without walking the function again.
    slot_vreg: Vec<VReg>,
    edits: Vec<(ProgPoint, Move)>,
}

impl AllocationBuilder {
    pub fn new(f: &impl RegallocFunc) -> Self {
        let mut offsets = Vec::with_capacity(f.num_insts() + 1);
        let mut ndefs = Vec::with_capacity(f.num_insts());
        let mut slot_vreg = Vec::new();

        for i in 0..f.num_insts() {
            offsets.push(slot_vreg.len() as u32);
            ndefs.push(f.defs(i).len() as u32);
            slot_vreg.extend(f.defs(i).iter().chain(f.uses(i)).map(|o| o.vreg));
        }
        offsets.push(slot_vreg.len() as u32);

        AllocationBuilder {
            // Overwritten for every operand a caller assigns; an operand left
            // untouched would be a bug in the allocator, which `verify` catches.
            allocs: vec![Alloc::Spill(u32::MAX); slot_vreg.len()],
            offsets,
            ndefs,
            slot_vreg,
            edits: Vec::new(),
        }
    }

    /// Put every mention of `v` in the same place. What a non-splitting allocator
    /// wants, and all either of the two below asks for.
    pub fn assign(&mut self, v: VReg, a: Alloc) {
        for (slot, &sv) in self.slot_vreg.iter().enumerate() {
            if sv == v {
                self.allocs[slot] = a;
            }
        }
    }

    pub fn set_def(&mut self, i: Inst, k: usize, a: Alloc) {
        let slot = self.offsets[i] as usize + k;
        self.allocs[slot] = a;
    }

    pub fn set_use(&mut self, i: Inst, k: usize, a: Alloc) {
        let slot = self.offsets[i] as usize + self.ndefs[i] as usize + k;
        self.allocs[slot] = a;
    }

    pub fn edit(&mut self, p: ProgPoint, m: Move) {
        self.edits.push((p, m));
    }

    pub fn finish(mut self, num_spills: u32) -> Allocation {
        self.edits.sort_by_key(|&(p, _)| p);
        Allocation {
            allocs: self.allocs,
            offsets: self.offsets,
            ndefs: self.ndefs,
            edits: self.edits,
            num_spills,
        }
    }
}

// --- the allocators ---------------------------------------------------------

/// Give every value its own stack slot.
pub fn spill_everything(
    f: &impl RegallocFunc,
    _env: &MachineEnv,
) -> Result<Allocation, RegallocError> {
    reject_unsupported(f)?;

    let mut b = AllocationBuilder::new(f);
    for v in 0..f.num_vregs() as u32 {
        b.assign(VReg(v), Alloc::Spill(v));
    }
    Ok(b.finish(f.num_vregs() as u32))
}

/// A value's live range, as one span of the linearized instruction order.
///
/// One span, not a set of them: a value live in two disjoint regions holds its
/// register through the gap between them. That over-reserves, and the cost is
/// paid in spills that a hole-aware allocator would avoid. It is not a
/// correctness question — a superset of the live points is always safe — and the
/// simplification keeps the scan below to one pass.
#[derive(Clone, Copy, Debug)]
struct Interval {
    v: VReg,
    start: u32,
    /// Exclusive.
    end: u32,
}

pub fn linear_scan(f: &impl RegallocFunc, env: &MachineEnv) -> Result<Allocation, RegallocError> {
    reject_unsupported(f)?;

    let order = f.block_order();

    // Number every instruction. A block's span is the half-open range of the
    // positions its instructions occupy.
    let mut pos = vec![0u32; f.num_insts()];
    let mut span = vec![(0u32, 0u32); f.num_blocks()];
    let mut p = 0u32;
    for &b in &order {
        let start = p;
        for &i in f.block_insts(b) {
            pos[i] = p;
            p += 1;
        }
        span[b.0 as usize] = (start, p);
    }

    let live_in = liveness(f, &order);

    // A value is live across a block if it is live on entry or on exit, and the
    // one-span simplification then covers the whole block. Otherwise it is live
    // only between the instructions that mention it, which the per-instruction
    // positions pin directly.
    let mut lo = vec![u32::MAX; f.num_vregs()];
    let mut hi = vec![0u32; f.num_vregs()];
    let mut extend = |v: VReg, s: u32, e: u32| {
        let i = v.0 as usize;
        lo[i] = lo[i].min(s);
        hi[i] = hi[i].max(e);
    };

    for &b in &order {
        let (bs, be) = span[b.0 as usize];
        for &s in &f.succs(b) {
            for &v in &live_in[s.0 as usize] {
                extend(v, bs, be);
            }
        }
        for &v in &live_in[b.0 as usize] {
            extend(v, bs, be);
        }
        for &i in f.block_insts(b) {
            for o in f.defs(i).iter().chain(f.uses(i)) {
                extend(o.vreg, pos[i], pos[i] + 1);
            }
        }
    }

    let mut ivs: Vec<Interval> = (0..f.num_vregs() as u32)
        .map(VReg)
        .filter(|v| lo[v.0 as usize] != u32::MAX)
        .map(|v| Interval {
            v,
            start: lo[v.0 as usize],
            end: hi[v.0 as usize],
        })
        .collect();
    ivs.sort_by_key(|i| i.start);

    let mut b = AllocationBuilder::new(f);
    let mut spills = 0u32;
    for class in RegClass::ALL {
        scan(
            ivs.iter().filter(|i| f.class(i.v) == class).copied(),
            env.order(class),
            &mut b,
            &mut spills,
        );
    }

    Ok(b.finish(spills))
}

/// Greedy scan over one register class.
///
/// Poletto–Sarkar: walk intervals by start, retire those whose end has passed,
/// and when nothing is free evict whichever live value is needed longest — that
/// value, or the incoming one if it outlives them all.
fn scan(
    ivs: impl Iterator<Item = Interval>,
    pool: &[PReg],
    out: &mut AllocationBuilder,
    spills: &mut u32,
) {
    let mut free: Vec<PReg> = pool.iter().rev().copied().collect();
    // (interval, register), kept sorted by end so the eviction candidate is last.
    let mut active: Vec<(Interval, PReg)> = Vec::new();

    for iv in ivs {
        active.retain(|&(a, r)| {
            if a.end <= iv.start {
                free.push(r);
                false
            } else {
                true
            }
        });

        if let Some(r) = free.pop() {
            out.assign(iv.v, Alloc::Reg(r));
            active.push((iv, r));
        } else {
            // Nothing free. The value with the furthest end is the cheapest to
            // lose: it is the one that would otherwise hold a register through the
            // most other intervals.
            let (worst, wr) = *active.last().expect("no free registers and none live");
            if worst.end > iv.end {
                out.assign(worst.v, Alloc::Spill(*spills));
                *spills += 1;
                out.assign(iv.v, Alloc::Reg(wr));
                active.pop();
                active.push((iv, wr));
            } else {
                out.assign(iv.v, Alloc::Spill(*spills));
                *spills += 1;
            }
        }
        active.sort_by_key(|&(a, _)| a.end);
    }
}

/// Neither allocator here honours a `Fixed`/`Reuse` operand or a clobber, so
/// neither pretends to. Checked up front so the answer is an error rather than
/// silently wrong code.
fn reject_unsupported(f: &impl RegallocFunc) -> Result<(), RegallocError> {
    for i in 0..f.num_insts() {
        for o in f.defs(i).iter().chain(f.uses(i)) {
            match o.constraint {
                Constraint::Any | Constraint::Reg => {}
                c => return Err(RegallocError::UnsupportedConstraint(c)),
            }
        }
        if let Some(&r) = f.clobbers(i).first() {
            return Err(RegallocError::UnsupportedClobber(r));
        }
    }
    Ok(())
}

/// Live-in sets, to a fixpoint.
///
/// Backwards over the block order, which converges in one pass for a loop-free
/// region and in a couple more with a back edge.
fn liveness(f: &impl RegallocFunc, order: &[Block]) -> Vec<Vec<VReg>> {
    let mut live_in: Vec<Vec<VReg>> = vec![Vec::new(); f.num_blocks()];

    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            let mut live: Vec<VReg> = f
                .succs(b)
                .iter()
                .flat_map(|s| live_in[s.0 as usize].iter().copied())
                .collect();
            live.sort();
            live.dedup();

            for &i in f.block_insts(b).iter().rev() {
                for o in f.defs(i) {
                    live.retain(|&v| v != o.vreg);
                }
                for o in f.uses(i) {
                    if !live.contains(&o.vreg) {
                        live.push(o.vreg);
                    }
                }
            }
            live.sort();

            if live != live_in[b.0 as usize] {
                live_in[b.0 as usize] = live;
                changed = true;
            }
        }
        if !changed {
            return live_in;
        }
    }
}

// --- the checker ------------------------------------------------------------

/// Does the allocated program still compute what the unallocated one did?
///
/// Symbolic execution, in the manner of regalloc2's checker: track, for each
/// physical register and stack slot, *which value it holds*, and assert at every
/// use that the location the allocator picked actually holds the value being read.
/// Meet at a join is intersection — a location the predecessors disagree about
/// holds nothing.
///
/// This subsumes an interference check without being one. Two values sharing a
/// register is not itself an error; it is an error exactly when the second def
/// destroys a value the program goes on to read, which is what this catches — and
/// it keeps catching it once an allocator starts *splitting* live ranges, at which
/// point "these two intervals overlap" stops being the right question and the
/// [`Move`] edits become part of the answer.
///
/// Run on every allocation in debug builds, from the encoder. The failure mode it
/// exists for is a value silently overwritten on one path through a loop — the
/// kind of bug that surfaces as a wrong answer six months later, on a program
/// nobody has written yet.
pub fn verify(f: &impl RegallocFunc, ra: &Allocation) -> Result<(), String> {
    check_constraints(f, ra)?;

    let order = f.block_order();
    let mut preds: Vec<Vec<Block>> = vec![Vec::new(); f.num_blocks()];
    for &b in &order {
        for s in f.succs(b) {
            preds[s.0 as usize].push(b);
        }
    }

    // `None` is "not reached yet", which is the top of the lattice: it constrains
    // nothing, so a back edge from a block we have not walked does not falsely
    // empty its target's state on the first pass. Intersection only ever shrinks a
    // state, so iterating to a fixpoint converges — and errors are only reported
    // after it does, because an optimistic intermediate state can name a location
    // as holding a value it does not yet hold.
    type State = HashMap<Alloc, VReg>;
    let mut entry_state: Vec<Option<State>> = vec![None; f.num_blocks()];
    entry_state[f.entry().0 as usize] = Some(State::new());

    loop {
        let mut changed = false;
        for &b in &order {
            let Some(before) = entry_state[b.0 as usize].clone() else {
                continue;
            };
            let after = transfer(f, ra, b, before, &mut |_| Ok(())).expect("no errors reported");

            for s in f.succs(b) {
                let merged = match &entry_state[s.0 as usize] {
                    None => after.clone(),
                    Some(old) => old
                        .iter()
                        .filter(|(loc, v)| after.get(loc) == Some(v))
                        .map(|(&loc, &v)| (loc, v))
                        .collect(),
                };
                if entry_state[s.0 as usize].as_ref() != Some(&merged) {
                    entry_state[s.0 as usize] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    for &b in &order {
        let Some(before) = entry_state[b.0 as usize].clone() else {
            continue;
        };
        transfer(f, ra, b, before, &mut |e| Err(e))?;
    }
    Ok(())
}

/// Run one block's instructions over the symbolic state, reporting each use that
/// reads a location not holding the value it wants.
fn transfer(
    f: &impl RegallocFunc,
    ra: &Allocation,
    b: Block,
    mut state: HashMap<Alloc, VReg>,
    report: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<HashMap<Alloc, VReg>, String> {
    for &i in f.block_insts(b) {
        for m in ra.edits_at(ProgPoint::before(i)) {
            apply_move(&mut state, m);
        }

        for (k, o) in f.uses(i).iter().enumerate() {
            let a = ra.use_(i, k);
            if state.get(&a) != Some(&o.vreg) {
                report(format!(
                    "mb{} inst {i}: use {k} reads {} for r{}, but it holds {}",
                    b.0,
                    a,
                    o.vreg.0,
                    match state.get(&a) {
                        Some(v) => format!("r{}", v.0),
                        None => "nothing".into(),
                    }
                ))?;
            }
        }

        for (k, o) in f.defs(i).iter().enumerate() {
            let a = ra.def(i, k);
            // A value defined here is no longer wherever it used to be: the machine
            // IR is not SSA (a block parameter is redefined on every edge), so a
            // stale copy left behind would let a later use read the previous value.
            state.retain(|_, v| *v != o.vreg);
            state.insert(a, o.vreg);
        }

        for &r in f.clobbers(i) {
            state.remove(&Alloc::Reg(r));
        }

        for m in ra.edits_at(ProgPoint::after(i)) {
            apply_move(&mut state, m);
        }
    }
    Ok(state)
}

/// A move copies: both ends hold the value afterwards. Whatever `to` held is gone,
/// which the checker notices at the next use of it — if there is one.
fn apply_move(state: &mut HashMap<Alloc, VReg>, m: &Move) {
    match state.get(&m.from).copied() {
        Some(v) => {
            state.insert(m.to, v);
        }
        None => {
            state.remove(&m.to);
        }
    }
}

/// The constraints the client asked for, honoured or not.
fn check_constraints(f: &impl RegallocFunc, ra: &Allocation) -> Result<(), String> {
    for i in 0..f.num_insts() {
        let ndefs = f.defs(i).len();
        for (k, o) in f.defs(i).iter().chain(f.uses(i)).enumerate() {
            let (a, what) = if k < ndefs {
                (ra.def(i, k), format!("def {k}"))
            } else {
                (ra.use_(i, k - ndefs), format!("use {}", k - ndefs))
            };
            match o.constraint {
                Constraint::Any => {}
                Constraint::Reg => {
                    if !matches!(a, Alloc::Reg(_)) {
                        return Err(format!("inst {i}: {what} wants a register, got {a}"));
                    }
                }
                Constraint::Fixed(want) => {
                    if a != Alloc::Reg(want) {
                        return Err(format!("inst {i}: {what} is pinned to {want}, got {a}"));
                    }
                }
                Constraint::Reuse(u) => {
                    let target = ra.use_(i, u);
                    if a != target {
                        return Err(format!(
                            "inst {i}: {what} must reuse use {u} ({target}), got {a}"
                        ));
                    }
                }
            }
            if a == Alloc::Spill(u32::MAX) {
                return Err(format!("inst {i}: {what} was never allocated"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CFG built directly — no machine IR, no instruction selection, no target.
    ///
    /// This is the point of the module boundary: a register allocation test says
    /// what is live where and nothing else, and a failure is a register allocation
    /// bug rather than a lowering bug that happens to show up here.
    #[derive(Default)]
    struct TestFunc {
        blocks: Vec<Vec<Inst>>,
        succs: Vec<Vec<Block>>,
        defs: Vec<Vec<Operand>>,
        uses: Vec<Vec<Operand>>,
        clobbers: Vec<Vec<PReg>>,
        classes: Vec<RegClass>,
    }

    impl TestFunc {
        fn block(&mut self) -> Block {
            self.blocks.push(Vec::new());
            self.succs.push(Vec::new());
            Block(self.blocks.len() as u32 - 1)
        }

        fn vreg(&mut self, class: RegClass) -> VReg {
            self.classes.push(class);
            VReg(self.classes.len() as u32 - 1)
        }

        fn int(&mut self) -> VReg {
            self.vreg(RegClass::Int)
        }

        fn inst(&mut self, b: Block, defs: Vec<Operand>, uses: Vec<Operand>) -> Inst {
            let i = self.defs.len();
            self.defs.push(defs);
            self.uses.push(uses);
            self.clobbers.push(Vec::new());
            self.blocks[b.0 as usize].push(i);
            i
        }

        fn goto(&mut self, b: Block, targets: &[Block]) {
            self.succs[b.0 as usize] = targets.to_vec();
        }
    }

    impl RegallocFunc for TestFunc {
        fn num_blocks(&self) -> usize {
            self.blocks.len()
        }
        fn entry(&self) -> Block {
            Block(0)
        }
        fn block_insts(&self, b: Block) -> &[Inst] {
            &self.blocks[b.0 as usize]
        }
        fn succs(&self, b: Block) -> Vec<Block> {
            self.succs[b.0 as usize].clone()
        }
        fn num_insts(&self) -> usize {
            self.defs.len()
        }
        fn defs(&self, i: Inst) -> &[Operand] {
            &self.defs[i]
        }
        fn uses(&self, i: Inst) -> &[Operand] {
            &self.uses[i]
        }
        fn clobbers(&self, i: Inst) -> &[PReg] {
            &self.clobbers[i]
        }
        fn num_vregs(&self) -> usize {
            self.classes.len()
        }
        fn class(&self, v: VReg) -> RegClass {
            self.classes[v.0 as usize]
        }
    }

    /// `n` integer registers and `n` float ones, numbered from zero.
    fn env(n: u8) -> MachineEnv {
        MachineEnv {
            allocation_order: [
                (0..n).map(|r| PReg::new(RegClass::Int, r)).collect(),
                (0..n).map(|r| PReg::new(RegClass::Float, r)).collect(),
            ],
        }
    }

    /// The smallest function with a register conflict in it: two values that are
    /// both live at the instruction consuming them.
    fn add_func() -> TestFunc {
        let mut f = TestFunc::default();
        let b = f.block();
        let (x, y, sum) = (f.int(), f.int(), f.int());
        f.inst(b, vec![Operand::any(x)], vec![]);
        f.inst(b, vec![Operand::any(y)], vec![]);
        f.inst(
            b,
            vec![Operand::any(sum)],
            vec![Operand::any(x), Operand::any(y)],
        );
        f.inst(b, vec![], vec![Operand::any(sum)]);
        f
    }

    #[test]
    fn overlapping_values_get_different_registers() {
        let f = add_func();
        let ra = linear_scan(&f, &env(4)).expect("no exotic constraints");
        verify(&f, &ra).expect("linear scan must satisfy its own invariant");
        assert_ne!(
            ra.use_(2, 0),
            ra.use_(2, 1),
            "both operands of the add are live at it"
        );
    }

    /// The checker has to be able to fail, or the assertion in the encoder proves
    /// nothing. Put every value in the same register and watch it complain.
    #[test]
    fn verify_catches_a_clobbered_value() {
        let f = add_func();
        let mut b = AllocationBuilder::new(&f);
        for v in 0..f.num_vregs() as u32 {
            b.assign(VReg(v), Alloc::Reg(PReg::new(RegClass::Int, 2)));
        }
        let err = verify(&f, &b.finish(0)).expect_err("every value in x2 destroys the last");
        assert!(err.contains("x2"), "{err}");
    }

    /// Under pressure something has to spill, and the result still has to be
    /// consistent — that is what the checker is for.
    #[test]
    fn spills_under_pressure_and_stays_consistent() {
        let mut f = TestFunc::default();
        let b = f.block();

        // Six values, all defined before any is used, then all read at once. Only
        // three registers exist, so at least three of them must go to the stack.
        let vs: Vec<VReg> = (0..6).map(|_| f.int()).collect();
        for &v in &vs {
            f.inst(b, vec![Operand::any(v)], vec![]);
        }
        let last = f.inst(b, vec![], vs.iter().map(|&v| Operand::any(v)).collect());

        let ra = linear_scan(&f, &env(3)).expect("no exotic constraints");
        verify(&f, &ra).expect("a spilled allocation is still a correct one");

        let spilled = (0..6)
            .filter(|&k| matches!(ra.use_(last, k), Alloc::Spill(_)))
            .count();
        assert!(spilled >= 3, "six values into three registers must spill");
    }

    /// A value live across a back edge is live for the whole loop, including the
    /// part of the body that runs before its use. The one-span interval covers
    /// that; this is the test that says so.
    #[test]
    fn value_live_across_a_back_edge_keeps_its_register() {
        let mut f = TestFunc::default();
        let (entry, body, exit) = (f.block(), f.block(), f.block());

        let carried = f.int();
        let scratch = f.int();
        f.inst(entry, vec![Operand::any(carried)], vec![]);
        f.goto(entry, &[body]);

        // The body defines a value *before* it reads the carried one, so an
        // allocator that let `carried` die at the top of the block would hand its
        // register to `scratch` and corrupt the next iteration.
        f.inst(body, vec![Operand::any(scratch)], vec![]);
        f.inst(
            body,
            vec![],
            vec![Operand::any(scratch), Operand::any(carried)],
        );
        f.goto(body, &[body, exit]);

        f.inst(exit, vec![], vec![Operand::any(carried)]);

        let ra = linear_scan(&f, &env(4)).expect("no exotic constraints");
        verify(&f, &ra).expect("the loop-carried value must survive the body");
        assert_ne!(ra.use_(2, 0), ra.use_(2, 1));
    }

    /// The differential oracle, as a property: whatever `linear_scan` decides, the
    /// checker accepts, and so does the naive allocation of the same program.
    #[test]
    fn both_allocators_pass_the_checker() {
        let f = add_func();
        for (name, ra) in [
            ("linear_scan", linear_scan(&f, &env(4))),
            ("spill_everything", spill_everything(&f, &env(4))),
        ] {
            let ra = ra.unwrap_or_else(|e| panic!("{name}: {e:?}"));
            verify(&f, &ra).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    /// A *split* allocation: the value is defined in one register and read from
    /// another, with a move in between. Neither allocator here produces one — this
    /// is hand-built, and it is what the checker will have to understand the day a
    /// splitting allocator (regalloc3) goes in.
    #[test]
    fn a_split_needs_its_move() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (v, w) = (f.int(), f.int());
        let def_v = f.inst(b, vec![Operand::any(v)], vec![]);
        f.inst(b, vec![Operand::any(w)], vec![]);
        let read = f.inst(b, vec![], vec![Operand::any(v), Operand::any(w)]);

        let (x2, x3) = (PReg::new(RegClass::Int, 2), PReg::new(RegClass::Int, 3));

        // `v` is defined into x2, then x2 is handed to `w`; `v` is read from x3.
        let split = |with_move: bool| {
            let mut b = AllocationBuilder::new(&f);
            b.set_def(def_v, 0, Alloc::Reg(x2));
            b.set_def(1, 0, Alloc::Reg(x2));
            b.set_use(read, 0, Alloc::Reg(x3));
            b.set_use(read, 1, Alloc::Reg(x2));
            if with_move {
                b.edit(
                    ProgPoint::after(def_v),
                    Move {
                        from: Alloc::Reg(x2),
                        to: Alloc::Reg(x3),
                        class: RegClass::Int,
                    },
                );
            }
            b.finish(0)
        };

        verify(&f, &split(true)).expect("the move carries v out of x2 before w lands in it");
        let err = verify(&f, &split(false)).expect_err("without the move x3 holds nothing");
        assert!(err.contains("nothing"), "{err}");
    }

    /// Both allocators decline what they cannot honour, rather than allocating
    /// around it and producing code that runs and is wrong.
    #[test]
    fn exotic_constraints_are_declined() {
        let x0 = PReg::new(RegClass::Int, 0);
        for constraint in [Constraint::Fixed(x0), Constraint::Reuse(0)] {
            let mut f = TestFunc::default();
            let b = f.block();
            let (v, w) = (f.int(), f.int());
            f.inst(b, vec![Operand::any(w)], vec![]);
            f.inst(
                b,
                vec![Operand {
                    vreg: v,
                    constraint,
                }],
                vec![Operand::any(w)],
            );

            let declined = linear_scan(&f, &env(4)).err().expect("must decline");
            assert_eq!(declined, RegallocError::UnsupportedConstraint(constraint));
        }
    }

    /// The constraints have no allocator behind them yet, but they are not inert:
    /// the checker enforces them, so an allocator that claims to honour one and
    /// does not is caught here rather than on the target that needed it.
    #[test]
    fn the_checker_enforces_a_fixed_register() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        let x0 = PReg::new(RegClass::Int, 0);
        f.inst(b, vec![Operand::fixed(v, x0)], vec![]);
        f.inst(b, vec![], vec![Operand::any(v)]);

        let mut wrong = AllocationBuilder::new(&f);
        wrong.assign(v, Alloc::Reg(PReg::new(RegClass::Int, 1)));
        let err = verify(&f, &wrong.finish(0)).expect_err("v is pinned to x0");
        assert!(err.contains("pinned to x0"), "{err}");

        let mut right = AllocationBuilder::new(&f);
        right.assign(v, Alloc::Reg(x0));
        verify(&f, &right.finish(0)).expect("x0 is where it was asked to go");
    }

    /// `Reg` is the constraint for an operand the client cannot reload into a
    /// scratch itself. Spilling one is not a slow allocation, it is a wrong one.
    #[test]
    fn the_checker_enforces_a_register_operand() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        f.inst(b, vec![Operand::reg(v)], vec![]);
        f.inst(b, vec![], vec![Operand::any(v)]);

        let mut spilled = AllocationBuilder::new(&f);
        spilled.assign(v, Alloc::Spill(0));
        let err = verify(&f, &spilled.finish(1)).expect_err("v may not go to the stack");
        assert!(err.contains("wants a register"), "{err}");
    }
}
