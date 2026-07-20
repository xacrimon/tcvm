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
//! one instruction and in a stack slot at the next. The allocator here does not
//! split, so it answers the same location for every mention of a value and emits
//! no edits; the query shape is what survives contact with regalloc3.
//!
//! One allocator, [`allocate`]: a linear scan over hole-aware live intervals of a
//! linearized CFG, with move coalescing. Its output is checked on every run by
//! [`verify`], a symbolic checker that is independent of how the allocation was
//! produced — so a bug in the allocator surfaces as a rejected allocation rather
//! than as wrong code six months later.
//!
//! [regalloc2]: https://github.com/bytecodealliance/regalloc2
//! [regalloc3]: https://github.com/Amanieu/regalloc3

use std::collections::HashMap;
use std::fmt;

use foldhash::fast::RandomState;

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
/// `Any` and `Reg` are all aarch64 asks for, and all [`allocate`] implements;
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
/// The allocator here produces none — it assigns whole values, so a value never
/// has to move. A splitting allocator produces them by the hundred, and the
/// client that cannot insert them cannot host one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Alloc,
    pub to: Alloc,
    pub class: RegClass,
}

/// A fix-up the client must emit at a [`ProgPoint`].
///
/// A [`Move`] is the ordinary one — a reload, a spill store, or a register bounced
/// to a scratch slot and back around an instruction that needed its register. A
/// [`Edit::Remat`] is how a spilled *constant* comes back: rather than load a slot
/// it never got, the client replays the instruction that defines it, straight into
/// a register. The value it reconstitutes is named so the checker can follow it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Edit {
    Move(Move),
    Remat { val: VReg, src: Inst, to: PReg },
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

    /// Values this block receives on every incoming edge, defined at its entry.
    ///
    /// Non-empty only when the client kept SSA form: a block parameter is a
    /// definition with no defining instruction, so the allocator must build its
    /// intervals and resolve its edges accordingly. A client that has no block
    /// parameters at all — a single-block function, say — answers empty and the
    /// allocator simply finds no edges to resolve.
    fn block_params(&self, _b: Block) -> &[VReg] {
        &[]
    }

    /// The values `b` passes to its sole successor, positionally matching that
    /// successor's [`RegallocFunc::block_params`].
    fn jump_args(&self, _b: Block) -> &[VReg] {
        &[]
    }

    /// Whether any block has parameters — i.e. whether this function is in SSA
    /// form. The allocator needs it to pick a code path.
    fn has_block_params(&self) -> bool {
        (0..self.num_blocks()).any(|b| !self.block_params(Block(b as u32)).is_empty())
    }

    /// A register this value would *prefer*, if one is free where it lands — an
    /// entry argument that arrives in a particular register, a call result the
    /// callee left in one. Soft: the allocator honours it when it fits and ignores
    /// it otherwise, so it can never force a spill. A hard requirement is a
    /// `Constraint::Fixed` operand instead.
    fn phys_hint(&self, _v: VReg) -> Option<PReg> {
        None
    }

    /// Scratch registers this instruction needs *for itself* — a temporary to
    /// materialize a wide immediate, the working registers a multi-instruction
    /// expansion writes between its parts. Each is a fresh vreg, live across the
    /// whole instruction so it collides with every operand and every other temp,
    /// and it always gets a register: a temp cannot be spilled, so under real
    /// pressure the allocator declines rather than inventing scratch it does not
    /// have. The encoder reads them back through [`Allocation::temp`].
    fn temps(&self, _i: Inst) -> &[Operand] {
        &[]
    }

    /// If this value is a pure constant a use can recompute more cheaply than a
    /// stack reload — an immediate, a constant-pool payload, an entry argument —
    /// name the instruction that defines it. When such a value is spilled the
    /// allocator reloads it by re-emitting that instruction (see the reload phase)
    /// instead of touching a slot, and its eviction cost is discounted to match.
    /// `None` means "spill it the ordinary way". The named instruction must be a
    /// pure single-def with no register inputs, so it can be replayed anywhere.
    fn remat(&self, _v: VReg) -> Option<Inst> {
        None
    }

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

/// What [`allocate`] declines to do.
///
/// Not "unimplemented" as an apology: an allocator that met a constraint it did
/// not understand and allocated anyway would produce code that runs and is wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegallocError {
    /// A `Fixed` or `Reuse` operand. x86 will need both; whoever writes that
    /// backend implements them here, or hands the whole job to regalloc3.
    UnsupportedConstraint(Constraint),
    /// A single instruction needs more registers live at once — its own operands
    /// plus its temps — than the machine has. Not a pressure artifact: the scan
    /// evicts and the reload phase bounces, so a value can always be found room
    /// *around* an instruction; this fires only when the instruction *itself* is
    /// unsatisfiable. The region stays interpreted rather than compiling wrong code.
    OutOfRegisters,
}

// --- the result -------------------------------------------------------------

/// Where every operand of every instruction lives.
///
/// Indexed per operand, not per value. See the module comment: this is the shape
/// that a splitting allocator can also fill in.
pub struct Allocation {
    /// Flattened, instruction by instruction: `ndefs[i]` defs, then the uses, then
    /// `ntemps[i]` temps.
    allocs: Vec<Alloc>,
    /// Where instruction `i`'s run starts in `allocs`.
    offsets: Vec<u32>,
    ndefs: Vec<u32>,
    ntemps: Vec<u32>,
    /// Sorted by program point.
    edits: Vec<(ProgPoint, Edit)>,
    /// Where each block's parameters live, per block. Empty for a function whose
    /// function with no block parameters at all.
    ///
    /// Parameters are not operands of any instruction, so nothing in `allocs`
    /// records where one is *defined*. The encoder does not need to know, but the
    /// checker does: without this it cannot tell whether an edge's moves actually
    /// delivered each parameter to the place its block reads it from.
    block_params: Vec<Vec<Alloc>>,
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
            base + k < self.offsets[i + 1] as usize - self.ntemps[i] as usize,
            "inst {i} has no use {k}"
        );
        self.allocs[base + k]
    }

    /// The register the encoder may use as temp `k` of instruction `i`.
    pub fn temp(&self, i: Inst, k: usize) -> Alloc {
        debug_assert!(k < self.ntemps[i] as usize, "inst {i} has no temp {k}");
        // Temps sit after the defs and uses: at the end of the instruction's run.
        let end = self.offsets[i + 1] as usize;
        self.allocs[end - self.ntemps[i] as usize + k]
    }

    /// Where parameter `k` of block `b` lives.
    pub fn block_param(&self, b: Block, k: usize) -> Alloc {
        self.block_params[b.0 as usize][k]
    }

    /// Every fix-up, with where it goes, ordered by program point.
    ///
    /// For measurement rather than encoding — the encoder wants [`Self::edits_at`]
    /// as it walks. Counting spill traffic needs the whole list *and* each edit's
    /// position, because what a reload costs depends on the loop depth of the block
    /// it lands in, not on how many there are.
    pub fn edits(&self) -> &[(ProgPoint, Edit)] {
        &self.edits
    }

    /// The fix-ups to emit at `p`, in insertion order among equal points.
    pub fn edits_at(&self, p: ProgPoint) -> impl Iterator<Item = &Edit> {
        let lo = self.edits.partition_point(|&(q, _)| q < p);
        self.edits[lo..]
            .iter()
            .take_while(move |&&(q, _)| q == p)
            .map(|(_, m)| m)
    }
}

/// Builds an [`Allocation`], per operand.
///
/// Public because the allocator is not the only thing that will ever want to
/// produce one: an adapter around an external allocator fills the same slots, and
/// so does a test that wants to hand-build a deliberately broken allocation.
pub struct AllocationBuilder {
    allocs: Vec<Alloc>,
    offsets: Vec<u32>,
    ndefs: Vec<u32>,
    ntemps: Vec<u32>,
    /// The value each slot of `allocs` is a mention of, so [`Self::assign`] can
    /// find every mention of a value without walking the function again.
    slot_vreg: Vec<VReg>,
    edits: Vec<(ProgPoint, Edit)>,
    block_params: Vec<Vec<Alloc>>,
}

impl AllocationBuilder {
    pub fn new(f: &impl RegallocFunc) -> Self {
        let mut offsets = Vec::with_capacity(f.num_insts() + 1);
        let mut ndefs = Vec::with_capacity(f.num_insts());
        let mut ntemps = Vec::with_capacity(f.num_insts());
        let mut slot_vreg = Vec::new();

        for i in 0..f.num_insts() {
            offsets.push(slot_vreg.len() as u32);
            ndefs.push(f.defs(i).len() as u32);
            ntemps.push(f.temps(i).len() as u32);
            slot_vreg.extend(
                f.defs(i)
                    .iter()
                    .chain(f.uses(i))
                    .chain(f.temps(i))
                    .map(|o| o.vreg),
            );
        }
        offsets.push(slot_vreg.len() as u32);

        AllocationBuilder {
            // Overwritten for every operand a caller assigns; an operand left
            // untouched would be a bug in the allocator, which `verify` catches.
            allocs: vec![Alloc::Spill(u32::MAX); slot_vreg.len()],
            offsets,
            ndefs,
            ntemps,
            slot_vreg,
            edits: Vec::new(),
            block_params: (0..f.num_blocks())
                .map(|b| vec![Alloc::Spill(u32::MAX); f.block_params(Block(b as u32)).len()])
                .collect(),
        }
    }

    /// Record where parameter `k` of block `b` lives.
    pub fn set_block_param(&mut self, b: Block, k: usize, a: Alloc) {
        self.block_params[b.0 as usize][k] = a;
    }

    /// Put every mention of `v` in the same place.
    ///
    /// What a non-splitting allocator wants. [`allocate`] no longer uses it — it
    /// fills operands one at a time from a position-indexed table, so that a split
    /// value can answer differently at different mentions — but a client
    /// hand-building an allocation (the tests below, the encoders' fixtures) still
    /// does.
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

    pub fn set_temp(&mut self, i: Inst, k: usize, a: Alloc) {
        let slot = self.offsets[i + 1] as usize - self.ntemps[i] as usize + k;
        self.allocs[slot] = a;
    }

    pub fn edit(&mut self, p: ProgPoint, e: Edit) {
        self.edits.push((p, e));
    }

    pub fn finish(mut self, num_spills: u32) -> Allocation {
        // Stable, so several fix-ups at one point keep the order they were pushed —
        // a bounce's save must precede the reload that borrows its register.
        self.edits.sort_by_key(|&(p, _)| p);
        Allocation {
            allocs: self.allocs,
            offsets: self.offsets,
            ndefs: self.ndefs,
            ntemps: self.ntemps,
            edits: self.edits,
            block_params: self.block_params,
            num_spills,
        }
    }
}

// --- the allocator ----------------------------------------------------------

/// A live interval as an ordered set of *segments*.
///
/// A hole matters: a loop-carried block parameter is dead through the tail of the
/// loop body — where the next iteration's value is computed — so the two do not
/// interfere and can share a register, which is exactly what turns a back-edge
/// copy into a no-op. The one-span model cannot see that hole and so can never
/// coalesce across a loop.
#[derive(Clone)]
struct Segs {
    v: VReg,
    /// Sorted, merged, half-open `[from, to)`, over a *doubled* position axis (see
    /// [`allocate`]): a use of value A and a def of value B at the same
    /// instruction get adjacent-but-disjoint segments, so B may reuse A's register
    /// — the mechanism behind copy coalescing and behind a two-address-friendly
    /// def reusing a dying source.
    ranges: Vec<(u32, u32)>,
}

impl Segs {
    fn start(&self) -> u32 {
        self.ranges[0].0
    }

    fn end(&self) -> u32 {
        self.ranges.last().unwrap().1
    }

    fn covers(&self, p: u32) -> bool {
        self.ranges.iter().any(|&(s, e)| s <= p && p < e)
    }

    /// The earliest position at which both are live, if they ever overlap.
    fn intersect(&self, other: &Segs) -> Option<u32> {
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (a0, a1) = self.ranges[i];
            let (b0, b1) = other.ranges[j];
            let lo = a0.max(b0);
            let hi = a1.min(b1);
            if lo < hi {
                return Some(lo);
            }
            if a1 < b1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        None
    }
}

/// Where each value lives, as a function of program position.
///
/// This is the one interface change splitting needs. A value that keeps a single
/// location for its whole life has a single entry, and every query answers the
/// same — which is what the scan produces today. A *split* value has several:
/// `(from, alloc)` says the value is at `alloc` from position `from` until the next
/// entry starts. Because asking always requires a position, a caller that has not
/// been taught which position it means will not compile, rather than silently
/// reading the wrong end of a split value.
///
/// Callers ask at the doubled positions [`allocate`] numbers by: a use of
/// instruction `i` at `2·pos[i]`, a def or temp at `2·pos[i] + 1`, a block
/// parameter at the start of its block.
struct Locations {
    /// Per value, `(from, alloc)` in increasing `from`.
    at: Vec<Vec<(u32, Alloc)>>,
}

impl Locations {
    fn new(num_vregs: usize) -> Self {
        Locations {
            at: vec![Vec::new(); num_vregs],
        }
    }

    /// Record that `v` lives at `a` from `from` until whatever comes next.
    ///
    /// Entries must be pushed in increasing `from` per value.
    fn put(&mut self, v: VReg, from: u32, a: Alloc) {
        debug_assert!(
            self.at[v.0 as usize].last().is_none_or(|&(f, _)| f < from),
            "v{} split points must be pushed in order",
            v.0
        );
        self.at[v.0 as usize].push((from, a));
    }

    /// Where `v` lives at position `p` — the latest split at or before `p`.
    ///
    /// `None` for a value with no location at all: one never mentioned, or a
    /// rematerializable value that was dropped rather than spilled and is replayed
    /// at each mention instead.
    fn get(&self, v: VReg, p: u32) -> Option<Alloc> {
        let e = &self.at[v.0 as usize];
        let k = e.partition_point(|&(from, _)| from <= p);
        (k > 0).then(|| e[k - 1].1)
    }
}

fn merge_ranges(mut r: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    r.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (s, e) in r {
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// The lexicographic key that ranks a spill victim: larger = more willing to
/// evict. Reversed loop weight first (protect loop-resident values), then
/// rematerializability (a constant is near-free to bring back), then Belady's
/// furthest end.
type EvictKey = (std::cmp::Reverse<u32>, bool, u32);

/// Linear scan over hole-aware intervals, with move coalescing.
///
/// A greedy scan — walk values by start, keep the live ones in `active`, retire the
/// rest — with two properties that together kill the shuffle:
///
///   - **Holes.** Intervals are segment sets, so a value dead in a gap frees its
///     register for that gap (the `inactive` list). A value whose interval merely
///     *straddles* the current point without covering it reserves its register
///     only up to where it becomes live again, not unconditionally.
///   - **Hints.** An edge links each argument to the parameter it feeds, and a
///     `Reuse` def links a two-address op's result to the input it overwrites;
///     whichever end is placed first biases the other toward the same register.
///     Because the argument's live range ends exactly where the parameter's
///     begins, the argument's register is free at that point, so the bias lands
///     and the edge needs no move at all. Loop-carried parameters coalesce the same
///     way: the parameter is placed when its header is seen, and the back-edge
///     argument — computed in the loop tail, inside the parameter's hole — is
///     hinted onto it.
///
/// The scan assigns whole values, so a value that keeps a register needs no edits
/// and joins need no shuffle. Under pressure a value that will not fit whole is
/// sent to the stack — either the one being placed or, by a blended cost model
/// ([`EvictKey`]: loop weight, then rematerializability, then Belady), a value
/// already placed — and the reload phase splits it back into registers at its
/// mentions with edits. The result is total on any target whose instructions each
/// fit the register file: it declines only when one instruction cannot (see
/// [`RegallocError::OutOfRegisters`]).
pub fn allocate(f: &impl RegallocFunc, env: &MachineEnv) -> Result<Allocation, RegallocError> {
    allocate_with(f, env, None)
}

/// Which values a spiller decided are in registers at each instruction.
///
/// Position-independent on purpose. The obvious alternative — handing the
/// allocator ready-made intervals on the doubled axis — requires the spiller to
/// number positions exactly as [`allocate_with`] does, and nothing would catch the
/// two drifting apart. Per-instruction sets cannot drift: the allocator derives the
/// runs itself, from its own numbering.
#[derive(Clone, Copy)]
pub struct RegisterSets<'a> {
    /// Values in registers as each instruction reads it.
    pub w_use: &'a [Vec<VReg>],
    /// Values in registers as each instruction leaves it.
    pub w_after: &'a [Vec<VReg>],
}

/// [`allocate`], optionally taking a spiller's decisions instead of making its own.
///
/// With `sets`, register pressure has already been lowered to the size of the
/// register file everywhere, so the scan colours pre-split runs and never has to
/// spill — see [`allocate_presplit`]. Without, it runs the whole-value scan below.
pub fn allocate_with(
    f: &impl RegallocFunc,
    env: &MachineEnv,
    sets: Option<RegisterSets<'_>>,
) -> Result<Allocation, RegallocError> {
    reject_unsupported(f)?;

    let order = f.block_order();

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

    if let Some(sets) = sets {
        return allocate_presplit(f, env, &order, &pos, &span, sets);
    }

    let raw = build_intervals(f, &order, &pos, &span);

    // Merge once; the scan reads these per class, and the reload phase reads them
    // again to know which register is free where.
    let mut ranges: Vec<Vec<(u32, u32)>> = raw.into_iter().map(merge_ranges).collect();

    let ord_idx = {
        let mut m = vec![0u32; f.num_blocks()];
        for (k, &b) in order.iter().enumerate() {
            m[b.0 as usize] = k as u32;
        }
        m
    };
    let mut loops: Vec<(u32, u32)> = Vec::new();
    for &b in &order {
        for s in f.succs(b) {
            if ord_idx[s.0 as usize] <= ord_idx[b.0 as usize] {
                loops.push((span[s.0 as usize].0 * 2, span[b.0 as usize].1 * 2));
            }
        }
    }

    // A value may be merged with another only if a single location can serve every
    // mention of both. Three cannot: a temp is scratch with no home of its own, a
    // rematerializable value has no home at all (it is replayed), and a value with
    // a fixed-register mention is already pinned somewhere a merge could contradict.
    let eligible: Vec<bool> = {
        let mut ok = vec![true; f.num_vregs()];
        for i in 0..f.num_insts() {
            for o in f.temps(i) {
                ok[o.vreg.0 as usize] = false;
            }
            for o in f.defs(i).iter().chain(f.uses(i)) {
                if matches!(o.constraint, Constraint::Fixed(_)) {
                    ok[o.vreg.0 as usize] = false;
                }
            }
        }
        for v in 0..f.num_vregs() {
            if f.remat(VReg(v as u32)).is_some() {
                ok[v] = false;
            }
        }
        ok
    };

    let mut sets = coalesce(f, &order, &ranges, &span, &loops, &eligible);
    // Every member reports its *set's* live range from here on: the set holds one
    // location for that whole extent, so anything asking "is this register busy at
    // p" has to see it. Only the leader is handed to the scan, below.
    for v in 0..f.num_vregs() {
        let leader = sets.find(v as u32);
        ranges[v] = sets.ranges[leader as usize].clone();
    }
    let ranges = ranges;

    // Copy affinities, both directions: whichever end is placed first pulls the
    // other toward its register. A `Reuse` def is coalesced the same way — put it
    // in the register of the source it reuses so the two-address op needs no copy —
    // but where a copy affinity is a hint, `Reuse` is enforced in the reload phase.
    let mut affin: HashMap<VReg, Vec<VReg>, RandomState> = HashMap::default();
    for i in 0..f.num_insts() {
        for o in f.defs(i) {
            if let Constraint::Reuse(uk) = o.constraint {
                let s = f.uses(i)[uk].vreg;
                affin.entry(o.vreg).or_default().push(s);
                affin.entry(s).or_default().push(o.vreg);
            }
        }
    }
    // Clobbers, as instruction-wide register reservations. A clobbered register is
    // busy across the whole slot `[2·pos, 2·pos + 2)` — a call destroys it, a
    // macro-op's exit stub writes it — so a value whose interval covers that slot
    // cannot live there. Stored by slot start; the two sub-slots are `lo` and
    // `lo + 1`.
    //
    // A `Fixed(preg)` operand reserves its register the same way, but for the scan
    // only: no *other* value may hold `preg` across the instruction (the reload
    // phase then moves the fixed value in and out of `preg`). Unlike a real clobber
    // this is not shown to the client's checker — the fixed mention legitimately
    // writes `preg` — so it is added here rather than to `clobbers(i)`.
    let mut clobbers: Vec<(u32, PReg)> = Vec::new();
    for (i, &p) in pos.iter().enumerate() {
        let lo = p * 2;
        for &r in f.clobbers(i) {
            clobbers.push((lo, r));
        }
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if let Constraint::Fixed(r) = o.constraint {
                clobbers.push((lo, r));
            }
        }
    }

    // A temp is scratch, not a value: it has no stack home, so it cannot be
    // spilled. If one will not fit, that is a decline, not a spill.
    let mut is_temp = vec![false; f.num_vregs()];
    for i in 0..f.num_insts() {
        for o in f.temps(i) {
            is_temp[o.vreg.0 as usize] = true;
        }
    }

    // --- static spill weights, for the eviction heuristic --------------------
    // Loops as doubled-axis position ranges: a back edge `s <- b` (a successor `s`
    // already placed when its predecessor `b` is) makes `[start(s), end(b))` a loop
    // body. A value whose live range meets a loop range is expensive to spill there
    // — the reload lands in the loop — and nesting counts once per enclosing loop.
    let vreg_loop: Vec<u32> = (0..f.num_vregs())
        .map(|v| {
            if ranges[v].is_empty() {
                return 0;
            }
            let (lo, hi) = (ranges[v][0].0, ranges[v].last().unwrap().1);
            loops.iter().filter(|&&(a, b)| lo < b && a < hi).count() as u32
        })
        .collect();

    // A value is rematerializable if the client can recompute it and every mention
    // is register-constrained — a stack mention would read a slot the remat reload
    // never writes. Used to discount its eviction cost (and, in the reload phase, to
    // replay the constant instead of loading a slot).
    let mut all_reg = vec![true; f.num_vregs()];
    for i in 0..f.num_insts() {
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if o.constraint != Constraint::Reg {
                all_reg[o.vreg.0 as usize] = false;
            }
        }
    }
    let remat_src: Vec<Option<Inst>> = (0..f.num_vregs() as u32)
        .map(|v| f.remat(VReg(v)).filter(|_| all_reg[v as usize]))
        .collect();

    let mut loc: Vec<Option<Alloc>> = vec![None; f.num_vregs()];
    // A value spilled because it is cheaper to recompute: no slot, no store; each
    // `Reg` use replays `remat_src`. Kept out of `loc` as a distinct state so the
    // reload phase can tell it apart from an ordinary stack spill.
    let mut remat_spilled = vec![false; f.num_vregs()];
    let mut spills = 0u32;

    for class in RegClass::ALL {
        let pool = env.order(class);

        let mut ivs: Vec<Segs> = (0..f.num_vregs() as u32)
            .map(VReg)
            .filter(|v| {
                // One interval per *set*, not per value: a merged set has one
                // location, so its members must not compete with each other.
                sets.find(v.0) == v.0 && f.class(*v) == class && !ranges[v.0 as usize].is_empty()
            })
            .map(|v| Segs {
                v,
                ranges: ranges[v.0 as usize].clone(),
            })
            .collect();
        ivs.sort_by_key(|iv| iv.start());

        let ridx = |r: PReg| pool.iter().position(|&x| x == r);

        // Indices into `ivs`.
        let mut active: Vec<usize> = Vec::new();
        let mut inactive: Vec<usize> = Vec::new();

        for cur in 0..ivs.len() {
            let at = ivs[cur].start();

            // Retire what has ended; park what is merely in a hole here.
            active.retain(|&a| {
                if ivs[a].end() <= at {
                    false
                } else if !ivs[a].covers(at) {
                    inactive.push(a);
                    false
                } else {
                    true
                }
            });
            inactive.retain(|&a| {
                if ivs[a].end() <= at {
                    false
                } else if ivs[a].covers(at) {
                    active.push(a);
                    false
                } else {
                    true
                }
            });

            // How far each pool register stays free from here on: 0 if an active
            // value holds it, else up to where an inactive value that overlaps `cur`
            // reclaims it. `holder`/`inactive_limit`/`clobbered` also record *why*, so
            // eviction can tell an evictable active holder from a register `cur` may
            // never take (clobbered, or reclaimed by an inactive value before it ends).
            let mut free_until: Vec<u32> = vec![u32::MAX; pool.len()];
            let mut holder: Vec<Option<usize>> = vec![None; pool.len()];
            let mut inactive_limit: Vec<u32> = vec![u32::MAX; pool.len()];
            let mut clobbered: Vec<bool> = vec![false; pool.len()];
            for &a in &active {
                if let Some(Alloc::Reg(r)) = loc[ivs[a].v.0 as usize]
                    && let Some(k) = ridx(r)
                {
                    free_until[k] = 0;
                    holder[k] = Some(a);
                }
            }
            for &a in &inactive {
                if let Some(Alloc::Reg(r)) = loc[ivs[a].v.0 as usize]
                    && let Some(x) = ivs[a].intersect(&ivs[cur])
                    && let Some(k) = ridx(r)
                {
                    free_until[k] = free_until[k].min(x);
                    inactive_limit[k] = inactive_limit[k].min(x);
                }
            }

            // A register clobbered anywhere `cur` is live is unavailable to it, and
            // eviction cannot buy it back — the clobber destroys `cur` regardless.
            for &(lo, r) in &clobbers {
                if r.class() == class
                    && (ivs[cur].covers(lo) || ivs[cur].covers(lo + 1))
                    && let Some(k) = ridx(r)
                {
                    free_until[k] = 0;
                    clobbered[k] = true;
                }
            }

            let end = ivs[cur].end();

            // Prefer a copy partner's register when it fits the whole interval.
            let mut chosen = affin.get(&ivs[cur].v).and_then(|parts| {
                parts.iter().find_map(|pv| match loc[pv.0 as usize] {
                    Some(Alloc::Reg(r)) => ridx(r).filter(|&k| free_until[k] >= end),
                    _ => None,
                })
            });

            // Then the target's soft preference, if it fits.
            if chosen.is_none()
                && let Some(r) = f.phys_hint(ivs[cur].v)
            {
                chosen = ridx(r).filter(|&k| free_until[k] >= end);
            }

            // Otherwise first-fit in the target's preference order.
            if chosen.is_none() {
                chosen = (0..pool.len()).find(|&k| free_until[k] >= end);
            }

            if let Some(k) = chosen {
                loc[ivs[cur].v.0 as usize] = Some(Alloc::Reg(pool[k]));
                active.push(cur);
                continue;
            }

            // No register is free for `cur`'s whole interval, so someone goes to the
            // stack. Weigh spilling `cur` against evicting a value already placed:
            // larger tuple = more willing to spill. Protect loop-resident values
            // first (they would reload inside the loop), then favour rematerializable
            // constants (near-free to bring back), then Belady — the value whose live
            // range reaches furthest is needed latest.
            let cur_v = ivs[cur].v;
            let evtuple = |v: VReg, e: u32| -> EvictKey {
                (
                    std::cmp::Reverse(vreg_loop[v.0 as usize]),
                    remat_src[v.0 as usize].is_some(),
                    e,
                )
            };
            // `None` victim = spill `cur` itself; `Some(k)` = evict `k`'s holder.
            let mut best: Option<(EvictKey, Option<usize>)> = None;
            if !is_temp[cur_v.0 as usize] {
                best = Some((evtuple(cur_v, end), None));
            }
            for k in 0..pool.len() {
                // A holder is evictable only where `cur` could actually live: not
                // clobbered, and no inactive value reclaims the register before `cur`
                // ends. A temp cannot be the victim — it has no stack home.
                if clobbered[k] || inactive_limit[k] < end {
                    continue;
                }
                let Some(a) = holder[k] else { continue };
                let vv = ivs[a].v;
                if is_temp[vv.0 as usize] {
                    continue;
                }
                let key = (evtuple(vv, ivs[a].end()), Some(k));
                if best.as_ref().is_none_or(|b| key.0 > b.0) {
                    best = Some(key);
                }
            }

            // Whoever loses gets sent to the stack. Evicting also hands the freed
            // register to `cur` and drops the victim from `active`.
            let victim = match best {
                // `cur` is a temp and every register is clobbered, reclaimed, or
                // holds another temp: more registers are needed here than exist.
                None => return Err(RegallocError::OutOfRegisters),
                Some((_, None)) => cur_v,
                Some((_, Some(k))) => {
                    let a = holder[k].unwrap();
                    let v = ivs[a].v;
                    active.retain(|&x| x != a);
                    loc[cur_v.0 as usize] = Some(Alloc::Reg(pool[k]));
                    active.push(cur);
                    v
                }
            };

            // Nothing to store and no slot if the victim can be rematerialized; an
            // ordinary spill slot otherwise. The reload phase brings it back at each
            // `Reg` mention.
            if remat_src[victim.0 as usize].is_some() {
                remat_spilled[victim.0 as usize] = true;
            } else {
                loc[victim.0 as usize] = Some(Alloc::Spill(spills));
                spills += 1;
            }
        }
    }

    // Members follow their leader: that is what makes the merge real rather than a
    // preference, and what makes an edge's `from == to` and cost nothing.
    for v in 0..f.num_vregs() {
        let leader = sets.find(v as u32) as usize;
        if leader != v {
            loc[v] = loc[leader];
            remat_spilled[v] = remat_spilled[leader];
        }
    }

    // The scan assigns one location per value, so every entry starts at position 0
    // and every query answers the same. A splitting spiller pushes several entries
    // per value here instead; nothing downstream has to change for that, which is
    // the point of routing every location read through this table.
    let mut locs = Locations::new(f.num_vregs());
    for v in 0..f.num_vregs() as u32 {
        if let Some(a) = loc[v as usize] {
            locs.put(VReg(v), 0, a);
        }
    }

    let mut b = AllocationBuilder::new(f);
    for (i, &p) in pos.iter().enumerate() {
        for (k, o) in f.defs(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2 + 1) {
                b.set_def(i, k, a);
            }
        }
        for (k, o) in f.uses(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2) {
                b.set_use(i, k, a);
            }
        }
        for (k, o) in f.temps(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2 + 1) {
                b.set_temp(i, k, a);
            }
        }
    }

    // Reload phase. A value that got a whole register is already right everywhere.
    // One that spilled needs a register at each `Reg`-constrained mention: a use is
    // loaded (or, for a rematerializable constant, replayed) into one before the
    // instruction, a def is stored out after it. The register is whatever is free
    // there — a spilled value has vacated its own, so there is usually room. When
    // every register is taken by a value live *across* the instruction, one of them
    // is bounced to a scratch slot for the instruction's length and restored right
    // after (see `reload_reg`); the only true failure is an instruction that needs
    // more registers at once than the machine has.
    //
    // An `Any` mention of a spilled value is left on the stack — the client either
    // takes a memory operand (x86) or does not exist (aarch64 asks for `Reg`).
    for (i, &p) in pos.iter().enumerate() {
        // Registers already spoken for at this instruction: its operands, its temps,
        // and whatever it clobbers. A reload avoids them, and a bounce victim is
        // chosen from *outside* this set, so it never disturbs the instruction.
        let mut used: Vec<PReg> = Vec::new();
        for o in f.uses(i) {
            if let Some(Alloc::Reg(r)) = locs.get(o.vreg, p * 2) {
                used.push(r);
            }
        }
        for o in f.defs(i).iter().chain(f.temps(i)) {
            if let Some(Alloc::Reg(r)) = locs.get(o.vreg, p * 2 + 1) {
                used.push(r);
            }
        }
        used.extend_from_slice(f.clobbers(i));

        // Whether `v` needs bringing back into a register at `q`: either it has no
        // home at all (replayed) or its home there is a slot.
        let spilled = |v: VReg, q: u32| {
            remat_spilled[v.0 as usize] || matches!(locs.get(v, q), Some(Alloc::Spill(_)))
        };

        // Reserve every fixed register up front, so an ordinary reload for another
        // operand never lands on one before the fixed operand claims it.
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if let Constraint::Fixed(r) = o.constraint
                && !used.contains(&r)
            {
                used.push(r);
            }
        }

        // Where each use ended up, so a `Reuse` def can take the same register.
        let mut use_final: Vec<Alloc> = vec![Alloc::Spill(u32::MAX); f.uses(i).len()];

        // The input a two-address def reuses is handled entirely by that def (it is
        // moved into the def's register there), so the use loop leaves it alone.
        let reused_uk = f.defs(i).iter().find_map(|o| match o.constraint {
            Constraint::Reuse(uk) => Some(uk),
            _ => None,
        });

        for (k, o) in f.uses(i).iter().enumerate() {
            if Some(k) == reused_uk {
                continue;
            }
            let class = f.class(o.vreg);
            match o.constraint {
                // The instruction demands this value in a specific register: move it
                // there (or replay it there), leaving its home untouched.
                Constraint::Fixed(preg) => {
                    if remat_spilled[o.vreg.0 as usize] {
                        let src = remat_src[o.vreg.0 as usize].unwrap();
                        b.edit(
                            ProgPoint::before(i),
                            Edit::Remat {
                                val: o.vreg,
                                src,
                                to: preg,
                            },
                        );
                    } else {
                        let from = locs
                            .get(o.vreg, p * 2)
                            .expect("a mentioned value has a home");
                        b.edit(
                            ProgPoint::before(i),
                            Edit::Move(Move {
                                from,
                                to: Alloc::Reg(preg),
                                class,
                            }),
                        );
                    }
                    b.set_use(i, k, Alloc::Reg(preg));
                    use_final[k] = Alloc::Reg(preg);
                }
                Constraint::Reg if spilled(o.vreg, p * 2) => {
                    let (r, bounce) = reload_reg(
                        &mut b,
                        &mut spills,
                        ProgPoint::before(i),
                        p * 2,
                        class,
                        &used,
                        env,
                        &loc,
                        &ranges,
                    )?;
                    used.push(r);
                    // Save (from `reload_reg`) is already in; the load/replay follows
                    // it, and the restore follows the load — all at their right points.
                    if remat_spilled[o.vreg.0 as usize] {
                        let src = remat_src[o.vreg.0 as usize].unwrap();
                        b.edit(
                            ProgPoint::before(i),
                            Edit::Remat {
                                val: o.vreg,
                                src,
                                to: r,
                            },
                        );
                    } else if let Some(Alloc::Spill(s)) = locs.get(o.vreg, p * 2) {
                        b.edit(
                            ProgPoint::before(i),
                            Edit::Move(Move {
                                from: Alloc::Spill(s),
                                to: Alloc::Reg(r),
                                class,
                            }),
                        );
                    }
                    b.set_use(i, k, Alloc::Reg(r));
                    use_final[k] = Alloc::Reg(r);
                    if let Some(bs) = bounce {
                        b.edit(
                            ProgPoint::after(i),
                            Edit::Move(Move {
                                from: Alloc::Spill(bs),
                                to: Alloc::Reg(r),
                                class,
                            }),
                        );
                    }
                }
                // A `Reg` use already in a register, or an `Any` use: the scan put it
                // where it belongs. Record it for a possible `Reuse`.
                _ => {
                    use_final[k] = locs
                        .get(o.vreg, p * 2)
                        .expect("a mentioned value has a home");
                }
            }
        }

        for (k, o) in f.defs(i).iter().enumerate() {
            let class = f.class(o.vreg);
            match o.constraint {
                // Written into a specific register; ferry it to its home afterward.
                Constraint::Fixed(preg) => {
                    b.set_def(i, k, Alloc::Reg(preg));
                    if !remat_spilled[o.vreg.0 as usize] {
                        let home = locs
                            .get(o.vreg, p * 2 + 1)
                            .expect("a defined value has a home");
                        if home != Alloc::Reg(preg) {
                            b.edit(
                                ProgPoint::after(i),
                                Edit::Move(Move {
                                    from: Alloc::Reg(preg),
                                    to: home,
                                    class,
                                }),
                            );
                        }
                    }
                }
                // A two-address def: the op reads and writes one register `d`. Bring
                // the reused source into `d` first — a no-op once coalescing has put
                // the def in a dead source's register — so the op overwrites `d` (the
                // result) while the source survives untouched in its own home. Because
                // the source is copied rather than consumed, it may be live afterward;
                // the non-reused inputs were made to interfere with the def, so `d` is
                // clear of them.
                Constraint::Reuse(uk) => {
                    let src = f.uses(i)[uk].vreg;
                    let (d, bounce) = match locs.get(o.vreg, p * 2 + 1) {
                        Some(Alloc::Reg(r)) => (r, None),
                        // A spilled two-address result still computes in a register.
                        _ => reload_reg(
                            &mut b,
                            &mut spills,
                            ProgPoint::before(i),
                            p * 2 + 1,
                            class,
                            &used,
                            env,
                            &loc,
                            &ranges,
                        )?,
                    };
                    used.push(d);
                    if remat_spilled[src.0 as usize] {
                        let s = remat_src[src.0 as usize].unwrap();
                        b.edit(
                            ProgPoint::before(i),
                            Edit::Remat {
                                val: src,
                                src: s,
                                to: d,
                            },
                        );
                    } else {
                        let from = locs.get(src, p * 2).expect("a reused source has a home");
                        if from != Alloc::Reg(d) {
                            b.edit(
                                ProgPoint::before(i),
                                Edit::Move(Move {
                                    from,
                                    to: Alloc::Reg(d),
                                    class,
                                }),
                            );
                        }
                    }
                    b.set_use(i, uk, Alloc::Reg(d));
                    b.set_def(i, k, Alloc::Reg(d));
                    if !remat_spilled[o.vreg.0 as usize]
                        && let Some(Alloc::Spill(s)) = locs.get(o.vreg, p * 2 + 1)
                    {
                        b.edit(
                            ProgPoint::after(i),
                            Edit::Move(Move {
                                from: Alloc::Reg(d),
                                to: Alloc::Spill(s),
                                class,
                            }),
                        );
                    }
                    if let Some(bs) = bounce {
                        b.edit(
                            ProgPoint::after(i),
                            Edit::Move(Move {
                                from: Alloc::Spill(bs),
                                to: Alloc::Reg(d),
                                class,
                            }),
                        );
                    }
                }
                Constraint::Reg if spilled(o.vreg, p * 2 + 1) => {
                    let (r, bounce) = reload_reg(
                        &mut b,
                        &mut spills,
                        ProgPoint::before(i),
                        p * 2 + 1,
                        class,
                        &used,
                        env,
                        &loc,
                        &ranges,
                    )?;
                    used.push(r);
                    b.set_def(i, k, Alloc::Reg(r));
                    // A rematerializable def computes its constant into `r` and drops
                    // it — every reader replays it instead — so there is nothing to store.
                    if !remat_spilled[o.vreg.0 as usize]
                        && let Some(Alloc::Spill(s)) = locs.get(o.vreg, p * 2 + 1)
                    {
                        b.edit(
                            ProgPoint::after(i),
                            Edit::Move(Move {
                                from: Alloc::Reg(r),
                                to: Alloc::Spill(s),
                                class,
                            }),
                        );
                    }
                    if let Some(bs) = bounce {
                        b.edit(
                            ProgPoint::after(i),
                            Edit::Move(Move {
                                from: Alloc::Spill(bs),
                                to: Alloc::Reg(r),
                                class,
                            }),
                        );
                    }
                }
                _ => {}
            }
        }
    }

    if f.has_block_params() {
        for &blk in &order {
            // A parameter is defined at the start of its block, so that is where to
            // ask: a split value may live somewhere else by the end of it.
            let entry = span[blk.0 as usize].0 * 2;
            for (k, &prm) in f.block_params(blk).iter().enumerate() {
                if let Some(a) = locs.get(prm, entry) {
                    b.set_block_param(blk, k, a);
                }
            }
        }
        resolve_edges(
            f,
            env,
            &order,
            &pos,
            &ranges,
            &span,
            &locs,
            &remat_spilled,
            &remat_src,
            &mut spills,
            &mut b,
        )?;
    }

    Ok(b.finish(spills))
}

/// A run of the doubled position axis and the register it was given.
type PlacedRun = ((u32, u32), PReg);

/// Colour pre-split runs, for a function whose register pressure a spiller has
/// already brought within the register file.
///
/// The scan here is the whole-value one with its hardest part removed. Because at
/// most `k` runs cover any position, a free register always exists, so there is no
/// eviction heuristic, no spill-versus-evict decision, and no retroactive
/// re-spilling of an interval already placed. What is left is: walk runs by start,
/// retire the ones that have ended, take a register.
///
/// Runs are contiguous by construction, so there is no `inactive` list either — a
/// run either covers the current position or is finished. That is the difference
/// between splitting *during* the scan and splitting before it: holes become
/// separate intervals rather than gaps to reason about.
fn allocate_presplit(
    f: &impl RegallocFunc,
    env: &MachineEnv,
    order: &[Block],
    pos: &[u32],
    span: &[(u32, u32)],
    sets: RegisterSets<'_>,
) -> Result<Allocation, RegallocError> {
    let nv = f.num_vregs();

    // The runs to colour: maximal stretches of the doubled axis over which the
    // spiller keeps a value in a register. Positions are visited in increasing
    // order, so a run continues exactly when the previous slot ended where this one
    // begins.
    let mut runs: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nv];
    // A run never crosses a block boundary, even when the two blocks are adjacent in
    // the layout. Positions are linear but control flow is not: a run spanning the
    // join would tell edge resolution the value is in the same register on both
    // sides of an edge that control may never take, so no move is emitted and on the
    // path actually taken nothing ever put it there. Clipped here, continuity across
    // a *real* edge is re-established by `resolve_edges` comparing the two ends —
    // which costs nothing when they agree.
    let mark = |runs: &mut Vec<Vec<(u32, u32)>>, v: VReg, at: u32, floor: u32| {
        let r = &mut runs[v.0 as usize];
        match r.last_mut() {
            Some(last) if last.1 == at && last.0 >= floor => last.1 = at + 1,
            _ => r.push((at, at + 1)),
        }
    };
    for &b in order {
        let floor = span[b.0 as usize].0 * 2;
        for &i in f.block_insts(b) {
            let p = pos[i];
            for &v in &sets.w_use[i] {
                mark(&mut runs, v, p * 2, floor);
            }
            // A temp is scratch belonging to this instruction, not a value that
            // flows between them, so no spiller tracks it — but it still needs a
            // register here. It spans the whole instruction slot, as in
            // `build_intervals`, so it collides with every operand and with the
            // other temps and lands somewhere of its own. The spiller reserved the
            // room: temps are counted in the `k - |defs|` limit.
            for o in f.temps(i) {
                mark(&mut runs, o.vreg, p * 2, floor);
                mark(&mut runs, o.vreg, p * 2 + 1, floor);
            }
            for &v in &sets.w_after[i] {
                mark(&mut runs, v, p * 2 + 1, floor);
            }
        }
    }

    // Full live ranges as well: edge resolution needs to know which registers hold
    // a live value at a branch, which is a question about liveness, not about where
    // the spiller chose to keep things.
    let live: Vec<Vec<(u32, u32)>> = build_intervals(f, order, pos, span)
        .into_iter()
        .map(merge_ranges)
        .collect();

    // A register a `Fixed` operand claims, or an instruction destroys, is not
    // available to anything live across that instruction. Same treatment as the
    // whole-value scan: reserved for the length of the instruction's slot.
    let mut clobbers: Vec<(u32, PReg)> = Vec::new();
    for (i, &p) in pos.iter().enumerate() {
        let lo = p * 2;
        for &r in f.clobbers(i) {
            clobbers.push((lo, r));
        }
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if let Constraint::Fixed(r) = o.constraint {
                clobbers.push((lo, r));
            }
        }
    }

    // Copy affinities, as in the whole-value scan: a two-address result wants the
    // register of the input it overwrites.
    let mut affin: HashMap<VReg, Vec<VReg>, RandomState> = HashMap::default();
    for i in 0..f.num_insts() {
        for o in f.defs(i) {
            if let Constraint::Reuse(uk) = o.constraint {
                let src = f.uses(i)[uk].vreg;
                affin.entry(o.vreg).or_default().push(src);
                affin.entry(src).or_default().push(o.vreg);
            }
        }
    }

    let mut all_reg = vec![true; nv];
    for i in 0..f.num_insts() {
        for o in f.defs(i).iter().chain(f.uses(i)) {
            if o.constraint != Constraint::Reg {
                all_reg[o.vreg.0 as usize] = false;
            }
        }
    }
    let remat_src: Vec<Option<Inst>> = (0..nv as u32)
        .map(|v| f.remat(VReg(v)).filter(|_| all_reg[v as usize]))
        .collect();

    // --- colour the runs ----------------------------------------------------

    // `(value, run)` pairs, and the register each is given.
    let mut iv: Vec<(VReg, (u32, u32))> = Vec::new();
    for (v, rs) in runs.iter().enumerate() {
        for &r in rs {
            iv.push((VReg(v as u32), r));
        }
    }
    iv.sort_by_key(|&(v, (lo, hi))| (lo, hi, v.0));
    let mut iv_reg: Vec<Option<PReg>> = vec![None; iv.len()];

    for class in RegClass::ALL {
        let pool = env.order(class);
        if pool.is_empty() {
            continue;
        }
        let ridx = |r: PReg| pool.iter().position(|&x| x == r);

        let mine: Vec<usize> = (0..iv.len())
            .filter(|&x| f.class(iv[x].0) == class)
            .collect();

        let mut active: Vec<usize> = Vec::new();
        for &cur in &mine {
            let (v, (lo, hi)) = iv[cur];
            active.retain(|&a| iv[a].1.1 > lo);

            let mut busy = vec![false; pool.len()];
            for &a in &active {
                if let Some(r) = iv_reg[a]
                    && let Some(k) = ridx(r)
                {
                    busy[k] = true;
                }
            }
            for &(cl, r) in &clobbers {
                if r.class() == class
                    && ((lo <= cl && cl < hi) || (lo <= cl + 1 && cl + 1 < hi))
                    && let Some(k) = ridx(r)
                {
                    busy[k] = true;
                }
            }

            // A copy partner's register first, then the target's preference, then
            // first fit. Unlike the whole-value scan there is no fallback to
            // spilling: the spiller has already guaranteed one of these lands.
            let mut chosen = affin.get(&v).and_then(|parts| {
                parts.iter().find_map(|pv| {
                    iv.iter()
                        .zip(&iv_reg)
                        .find(|((w, (a, b)), _)| *w == *pv && *a <= lo && lo < *b)
                        .and_then(|(_, r)| *r)
                        .and_then(ridx)
                        .filter(|&k| !busy[k])
                })
            });
            if chosen.is_none()
                && let Some(r) = f.phys_hint(v)
            {
                chosen = ridx(r).filter(|&k| !busy[k]);
            }
            if chosen.is_none() {
                chosen = (0..pool.len()).find(|&k| !busy[k]);
            }

            let Some(k) = chosen else {
                // Only reachable if clobbers took the register file below what the
                // spiller was told it had; the spiller does not model them.
                return Err(RegallocError::OutOfRegisters);
            };
            iv_reg[cur] = Some(pool[k]);
            active.push(cur);
        }
    }

    // --- slots, locations, and the edits between them ------------------------

    // Runs per value, in position order, with the register each got.
    let mut placed: Vec<Vec<PlacedRun>> = vec![Vec::new(); nv];
    for (x, &(v, r)) in iv.iter().enumerate() {
        if let Some(reg) = iv_reg[x] {
            placed[v.0 as usize].push((r, reg));
        }
    }
    for p in placed.iter_mut() {
        p.sort_by_key(|&((lo, _), _)| lo);
    }

    // A value needs a slot when it is ever live without being in a register — that
    // memory has to exist for it. A rematerializable value never needs one: it is
    // replayed rather than loaded.
    //
    // The test is per position, not a comparison of totals. Totals lie: a value
    // stays in `W` after its last use until something evicts it, so its runs can
    // reach past the end of its live range, and a run overhanging one end can
    // exactly offset a genuine gap in the middle. That reads as "fully covered",
    // no slot is allocated, no reload is emitted, and the register is read having
    // never been written.
    let mut spills = 0u32;
    let mut slot: Vec<Option<u32>> = vec![None; nv];
    for v in 0..nv {
        if live[v].is_empty() || remat_src[v].is_some() {
            continue;
        }
        let gap = live[v].iter().any(|&(lo, hi)| {
            (lo..hi).any(|q| !placed[v].iter().any(|&((a, b), _)| a <= q && q < b))
        });
        // More than one run means the value left registers between them, whether or
        // not the gap shows up as un-covered liveness.
        if gap || placed[v].len() > 1 {
            slot[v] = Some(spills);
            spills += 1;
        }
    }

    let mut locs = Locations::new(nv);
    for v in 0..nv {
        let vr = VReg(v as u32);
        let home = slot[v].map(Alloc::Spill);
        match placed[v].first() {
            // Never in a register: it lives in its slot, or is replayed.
            None => {
                if let Some(h) = home {
                    locs.put(vr, 0, h);
                }
            }
            Some(&(_, first_reg)) => {
                // From position 0 rather than from the run's start, so a query before
                // the value is live still answers — the whole-value scan behaved the
                // same way and edge resolution relies on it.
                locs.put(vr, 0, Alloc::Reg(first_reg));
                let mut prev_end = placed[v][0].0.1;
                let mut prev_reg = placed[v][0].1;
                for &((lo, hi), reg) in &placed[v][1..] {
                    if lo > prev_end {
                        // A real gap: the value spent it in memory.
                        if let Some(h) = home {
                            locs.put(vr, prev_end, h);
                        }
                        locs.put(vr, lo, Alloc::Reg(reg));
                    } else if reg != prev_reg {
                        // Touching runs are one stretch in registers split by a block
                        // boundary — the clipping above makes that the common case —
                        // so the value never left. It only changed register.
                        locs.put(vr, lo, Alloc::Reg(reg));
                    }
                    prev_end = hi;
                    prev_reg = reg;
                }
                if let Some(h) = home {
                    locs.put(vr, prev_end, h);
                }
            }
        }
    }

    let mut b = AllocationBuilder::new(f);
    for (i, &p) in pos.iter().enumerate() {
        for (k, o) in f.defs(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2 + 1) {
                b.set_def(i, k, a);
            }
        }
        for (k, o) in f.uses(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2) {
                b.set_use(i, k, a);
            }
        }
        for (k, o) in f.temps(i).iter().enumerate() {
            if let Some(a) = locs.get(o.vreg, p * 2 + 1) {
                b.set_temp(i, k, a);
            }
        }
    }

    // --- the edits that make a split real -----------------------------------
    //
    // Where the value is defined, and where it enters its first register. Every
    // *other* run begins with the value coming back from memory, which is a reload;
    // the run holding the definition begins with the definition itself.
    let mut def_at: Vec<Option<u32>> = vec![None; nv];
    for (i, &p) in pos.iter().enumerate() {
        for o in f.defs(i) {
            def_at[o.vreg.0 as usize] = Some(p * 2 + 1);
        }
    }
    for &blk in order {
        for &prm in f.block_params(blk) {
            def_at[prm.0 as usize] = Some(span[blk.0 as usize].0 * 2);
        }
    }

    // Instruction at each position, to turn a run boundary back into a program
    // point.
    let mut at_pos = vec![0usize; pos.len()];
    for (i, &p) in pos.iter().enumerate() {
        at_pos[p as usize] = i;
    }

    for v in 0..nv {
        let vr = VReg(v as u32);
        let class = f.class(vr);
        let born = def_at[v];
        // A reload belongs where the value comes *back from memory*, which is where a
        // run is preceded by a gap — not merely where a run starts. Since runs are
        // clipped at every block boundary, most run starts are continuations that
        // already have the value in hand; reloading at each of those would put a load
        // at the top of every block the value passes through.
        let mut prev_end: Option<u32> = None;
        for &((lo, hi), reg) in &placed[v] {
            let from_memory = match prev_end {
                Some(pe) => lo > pe,
                // The value's first run: it came from memory unless it is born here,
                // or arrives already in place with no definition of its own.
                None => born.is_some_and(|d| !(lo <= d && d < hi)),
            };
            prev_end = Some(hi);
            if !from_memory {
                continue;
            }
            let inst = at_pos[(lo / 2) as usize];
            if let Some(src) = remat_src[v] {
                b.edit(
                    ProgPoint::before(inst),
                    Edit::Remat {
                        val: vr,
                        src,
                        to: reg,
                    },
                );
            } else if let Some(sl) = slot[v] {
                b.edit(
                    ProgPoint::before(inst),
                    Edit::Move(Move {
                        from: Alloc::Spill(sl),
                        to: Alloc::Reg(reg),
                        class,
                    }),
                );
            }
        }

        // One store, immediately after the definition — Wimmer05 §4c's spill-store
        // elimination, which is exact rather than heuristic here: SSA gives the
        // value a single definition, so the slot's contents never go stale and every
        // later store would be writing what is already there.
        if let (Some(sl), Some(d)) = (slot[v], born)
            && let Some(&(_, reg)) = placed[v].iter().find(|&&((lo, hi), _)| lo <= d && d < hi)
        {
            let store = Edit::Move(Move {
                from: Alloc::Reg(reg),
                to: Alloc::Spill(sl),
                class,
            });
            match f
                .defs(at_pos[(d / 2) as usize])
                .iter()
                .any(|o| o.vreg == vr)
            {
                true => b.edit(ProgPoint::after(at_pos[(d / 2) as usize]), store),
                // A block parameter has no defining instruction; it is delivered by
                // the edge, so the store goes at the top of its own block.
                false => b.edit(ProgPoint::before(at_pos[(d / 2) as usize]), store),
            }
        }
    }

    if f.has_block_params() {
        for &blk in order {
            let entry = span[blk.0 as usize].0 * 2;
            for (k, &prm) in f.block_params(blk).iter().enumerate() {
                if let Some(a) = locs.get(prm, entry) {
                    b.set_block_param(blk, k, a);
                }
            }
        }
        let remat_spilled = vec![false; nv];
        resolve_edges(
            f,
            env,
            order,
            pos,
            &live,
            span,
            &locs,
            &remat_spilled,
            &remat_src,
            &mut spills,
            &mut b,
        )?;
    }

    Ok(b.finish(spills))
}

/// A register to reload into at position `q`, plus a slot to restore afterward if
/// one had to be freed.
///
/// A free register if any exists. Otherwise every register is held by a value live
/// across this instruction, so one is *bounced*: a register outside `used` — hence
/// not an operand, temp, or clobber here — is saved to a fresh scratch slot at
/// `save_pp` for the length of the instruction. The caller emits its own reload or
/// store and then restores the bounced value from the returned slot. The one
/// unrecoverable case is an instruction whose own operands and temps already need
/// every register: nothing is left to bounce, and the region stays interpreted.
#[allow(clippy::too_many_arguments)]
fn reload_reg(
    b: &mut AllocationBuilder,
    spills: &mut u32,
    save_pp: ProgPoint,
    q: u32,
    class: RegClass,
    used: &[PReg],
    env: &MachineEnv,
    loc: &[Option<Alloc>],
    ranges: &[Vec<(u32, u32)>],
) -> Result<(PReg, Option<u32>), RegallocError> {
    let held = |r: PReg| {
        loc.iter().enumerate().any(|(w, a)| {
            matches!(a, Some(Alloc::Reg(rr)) if *rr == r)
                && ranges[w].iter().any(|&(s, e)| s <= q && q < e)
        })
    };
    if let Some(r) = env
        .order(class)
        .iter()
        .copied()
        .find(|&r| !used.contains(&r) && !held(r))
    {
        return Ok((r, None));
    }
    let Some(r) = env
        .order(class)
        .iter()
        .copied()
        .find(|&r| !used.contains(&r))
    else {
        return Err(RegallocError::OutOfRegisters);
    };
    let bs = *spills;
    *spills += 1;
    b.edit(
        save_pp,
        Edit::Move(Move {
            from: Alloc::Reg(r),
            to: Alloc::Spill(bs),
            class,
        }),
    );
    Ok((r, Some(bs)))
}

/// The allocator honours `Any`, `Reg`, `Fixed`, and `Reuse`. The one shape it
/// rejects is a `Reuse` on a *use*: reuse is a def-only relationship — a
/// two-address def taking the register of one of its sources — so a `Reuse` use
/// is a client bug that would otherwise be silently ignored.
fn reject_unsupported(f: &impl RegallocFunc) -> Result<(), RegallocError> {
    for i in 0..f.num_insts() {
        for o in f.uses(i) {
            if let Constraint::Reuse(_) = o.constraint {
                return Err(RegallocError::UnsupportedConstraint(o.constraint));
            }
        }
    }
    Ok(())
}

/// Values merged into a single allocation unit, by union-find.
///
/// The point of coalescing on SSA form: an edge is a copy with no instruction to
/// hold it, so a value and the parameter it feeds are two names for one thing. If
/// they get one location the edge costs nothing; if they get two, it costs a move
/// — and, once the register file is full, a move with nowhere to route through.
///
/// The merge is *not* a hint. A hint is consulted while placing a value and
/// dropped when the register it wanted is taken, which is exactly what happens
/// under the pressure that makes coalescing matter. A merged set is one interval
/// with one location, so it cannot come apart. It also lowers pressure directly:
/// twenty-eight accumulators become twenty-eight units rather than the fifty-odd
/// values naming them.
///
/// The one thing a merge may never do is put two values that are live at the same
/// time in one register, so a pair is merged only when their live ranges are
/// disjoint. Following regalloc3, the merge order is by priority rather than
/// arbitrary: merging A with B can make A–C impossible, so the edges that would
/// execute most often get first claim.
struct Coalesced {
    parent: Vec<u32>,
    /// Merged live ranges, valid for a set's leader.
    ranges: Vec<Vec<(u32, u32)>>,
}

impl Coalesced {
    fn find(&mut self, v: u32) -> u32 {
        let mut r = v;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        // Path compression, so a long chain is walked once.
        let mut c = v;
        while self.parent[c as usize] != r {
            let next = self.parent[c as usize];
            self.parent[c as usize] = r;
            c = next;
        }
        r
    }

    /// Merge the sets of `a` and `b` if their live ranges are disjoint.
    fn try_union(&mut self, a: VReg, b: VReg) -> bool {
        let (ra, rb) = (self.find(a.0), self.find(b.0));
        if ra == rb {
            return true;
        }

        if overlap(&self.ranges[ra as usize], &self.ranges[rb as usize]) {
            return false;
        }
        let merged = union_ranges(&self.ranges[ra as usize], &self.ranges[rb as usize]);
        self.parent[rb as usize] = ra;
        self.ranges[ra as usize] = merged;
        self.ranges[rb as usize] = Vec::new();
        true
    }
}

/// Whether two sorted, disjoint range lists share any position. One lockstep walk.
fn overlap(a: &[(u32, u32)], b: &[(u32, u32)]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].0 >= b[j].1 {
            j += 1;
        } else if a[i].1 <= b[j].0 {
            i += 1;
        } else {
            return true;
        }
    }
    false
}

fn union_ranges(a: &[(u32, u32)], b: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let take_a = j >= b.len() || (i < a.len() && a[i].0 <= b[j].0);
        out.push(if take_a {
            i += 1;
            a[i - 1]
        } else {
            j += 1;
            b[j - 1]
        });
    }
    out
}

/// Merge each edge's arguments with the parameters they feed.
///
/// `loop_depth` orders the work: an edge inside a loop executes more often than
/// one outside it, and since an early merge can block a later one, the expensive
/// edges get first refusal.
fn coalesce(
    f: &impl RegallocFunc,
    order: &[Block],
    ranges: &[Vec<(u32, u32)>],
    span: &[(u32, u32)],
    loops: &[(u32, u32)],
    eligible: &[bool],
) -> Coalesced {
    let mut c = Coalesced {
        parent: (0..f.num_vregs() as u32).collect(),
        ranges: ranges.to_vec(),
    };

    let depth = |b: Block| {
        let (lo, hi) = span[b.0 as usize];
        loops
            .iter()
            .filter(|&&(a, z)| lo * 2 < z && a < hi * 2)
            .count()
    };

    let mut edges: Vec<Block> = order
        .iter()
        .copied()
        .filter(|&b| !f.jump_args(b).is_empty())
        .collect();
    // Hottest first; among equals, keep the layout order so this is deterministic.
    edges.sort_by_key(|&b| std::cmp::Reverse(depth(b)));

    for b in edges {
        let succ = f.succs(b)[0];
        for (&a, &p) in f.jump_args(b).iter().zip(f.block_params(succ)) {
            if eligible[a.0 as usize] && eligible[p.0 as usize] && f.class(a) == f.class(p) {
                c.try_union(a, p);
            }
        }
    }
    c
}

/// Lifetime intervals, in one reverse pass and without a dataflow analysis.
///
/// Wimmer & Franz, CGO'10 Fig. 4. Segments live on a doubled axis: instruction
/// `i` uses at `2·pos` and defs at `2·pos + 1`, so a use segment `[.., 2·pos + 1)`
/// and a def segment `[2·pos + 1, ..)` touch without overlapping. That adjacency
/// is what lets a copy's ends — and a two-address op's dying source and its dest —
/// share a register, while a source that outlives the op extends past the def slot
/// and correctly interferes.
///
/// # Why this needs no fixpoint
///
/// A value live across a back edge cannot be seen in one backward sweep: when the
/// loop's last block is processed the header has not been, so the header's live-in
/// is still empty. The repair is the loop-header case at the bottom — everything
/// live at a loop header is live through the *whole* loop, and because the layout
/// keeps a loop's blocks contiguous (see [`super::order`]) that entire extent is
/// **one range**. One range add per live value, instead of iterating a live-set
/// dataflow to convergence.
///
/// The `live_in` sets this leaves behind are therefore deliberately *incomplete* —
/// the loop case adds ranges without updating them. Nothing downstream reads them,
/// which is the only reason that is allowed.
///
/// # Precondition: one definition per value
///
/// The loop case is *sound* on a multi-def function but it is a pessimization
/// there, so this must not be handed one. Its justification is SSA dominance: a
/// value live at a loop header is necessarily defined before the loop, hence
/// genuinely live throughout it. Give a value two definitions and that stops being
/// true — a loop-carried value defined *inside* the loop is live-in at the header
/// yet dead through the tail of the body, where the next iteration's value is
/// computed. Extending it across the whole loop fills in exactly the hole that
/// lets the two ends share a register, and a back-edge copy that would have
/// coalesced away comes back as a real move.
///
/// This was measured, not reasoned about: pointing this at the old destructed form
/// cost `is_prime` one move it had not needed.
///
fn build_intervals(
    f: &impl RegallocFunc,
    order: &[Block],
    pos: &[u32],
    span: &[(u32, u32)],
) -> Vec<Vec<(u32, u32)>> {
    let mut raw: Vec<Vec<(u32, u32)>> = vec![Vec::new(); f.num_vregs()];
    let mut live_in: Vec<Vec<VReg>> = vec![Vec::new(); f.num_blocks()];

    let mut ord_idx = vec![u32::MAX; f.num_blocks()];
    for (k, &b) in order.iter().enumerate() {
        ord_idx[b.0 as usize] = k as u32;
    }

    // How far each loop header's loop extends. A successor already placed is a back
    // edge and its source is inside the loop; the furthest such source's end is the
    // loop's end. Exact, not approximate, because the layout keeps a loop contiguous.
    let mut loop_end = vec![0u32; f.num_blocks()];
    for &b in order {
        for s in f.succs(b) {
            if ord_idx[s.0 as usize] <= ord_idx[b.0 as usize] {
                let e = span[b.0 as usize].1 * 2;
                let slot = &mut loop_end[s.0 as usize];
                *slot = (*slot).max(e);
            }
        }
    }

    for &b in order.iter().rev() {
        let (bs, be) = span[b.0 as usize];
        let (bfrom, bto) = (bs * 2, be * 2);

        let mut open: HashMap<VReg, u32, RandomState> = HashMap::default();
        for &s in &f.succs(b) {
            for &v in &live_in[s.0 as usize] {
                open.entry(v).or_insert(bto);
            }
        }
        // A successor's parameters are supplied by this block's arguments. The
        // argument is live to the end of this block and the parameter starts at the
        // successor's entry, so the two never overlap — which is exactly what lets
        // them share a register and the edge cost nothing.
        for &a in f.jump_args(b) {
            open.entry(a).or_insert(bto);
        }

        for &i in f.block_insts(b).iter().rev() {
            let slot = pos[i] * 2;
            // A temp spans the whole instruction slot, so it collides with every
            // operand and with the other temps and lands in its own register.
            for o in f.temps(i) {
                raw[o.vreg.0 as usize].push((slot, slot + 2));
            }
            for o in f.defs(i) {
                // A def ends the value's open segment; a dead def gets a minimal one.
                let end = open.remove(&o.vreg).unwrap_or(slot + 2);
                raw[o.vreg.0 as usize].push((slot + 1, end));
            }
            // A two-address def reuses exactly one input's register; every *other*
            // input must survive across the def, or the op would clobber it.
            let reused_uk = f.defs(i).iter().find_map(|o| match o.constraint {
                Constraint::Reuse(uk) => Some(uk),
                _ => None,
            });
            for (k, o) in f.uses(i).iter().enumerate() {
                let end = if reused_uk.is_some() && Some(k) != reused_uk {
                    slot + 2
                } else {
                    slot + 1
                };
                open.entry(o.vreg).or_insert(end);
            }
        }

        // A parameter is defined at this block's entry. It closes here and must not
        // propagate to the predecessors — they supply it as an argument instead.
        for &p in f.block_params(b) {
            let end = open.remove(&p).unwrap_or(bfrom + 1);
            raw[p.0 as usize].push((bfrom, end));
        }

        // Whatever is still open is live from the block's start.
        for (&v, &end) in open.iter() {
            raw[v.0 as usize].push((bfrom, end));
        }

        // The loop case: everything live at a loop header is live for the whole loop.
        let lend = loop_end[b.0 as usize];
        if lend > bto {
            for &v in open.keys() {
                raw[v.0 as usize].push((bfrom, lend));
            }
        }

        let mut live: Vec<VReg> = open.into_keys().collect();
        live.sort();
        live_in[b.0 as usize] = live;
    }

    raw
}

/// Where one edge assignment gets its value.
///
/// Not simply an [`Alloc`], because a rematerializable argument that was spilled
/// has no home to read: it carries no slot and is replayed at each mention. An
/// edge has to replay it too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeSrc {
    Loc(Alloc),
    Remat(VReg, Inst),
}

/// SSA deconstruction, fused into edge resolution (Wimmer & Franz, CGO'10 Fig. 7).
///
/// A block parameter has to arrive where its block expects it, from wherever the
/// predecessor's matching argument ended up. `isel` materializes an edge block for
/// every conditional branch, so a block that passes arguments has exactly one
/// successor: the moves go before its terminator, and there is no critical edge
/// left to split here.
///
/// The moves on one edge are a **parallel** copy — they happen simultaneously, so
/// a loop that rotates values through each other forms a cycle that naive in-order
/// emission would clobber. This is the sequencing that used to live in `isel`, now
/// over physical locations rather than virtual registers, which is the point of
/// moving it: at this level the allocator knows which registers are actually free
/// and can borrow one, instead of inventing a vreg for the stage above to place.
#[allow(clippy::too_many_arguments)]
fn resolve_edges(
    f: &impl RegallocFunc,
    env: &MachineEnv,
    order: &[Block],
    pos: &[u32],
    ranges: &[Vec<(u32, u32)>],
    span: &[(u32, u32)],
    locs: &Locations,
    remat_spilled: &[bool],
    remat_src: &[Option<Inst>],
    spills: &mut u32,
    b: &mut AllocationBuilder,
) -> Result<(), RegallocError> {
    // How many ways into each block, to decide where an edge's moves may go.
    let mut npreds = vec![0usize; f.num_blocks()];
    for &b in order {
        for sb in f.succs(b) {
            npreds[sb.0 as usize] += 1;
        }
    }

    for &pred in order {
        let succs = f.succs(pred);
        for &succ in &succs {
            let term = *f.block_insts(pred).last().expect("block has a terminator");
            let at_pos = pos[term] * 2;
            let entry = span[succ.0 as usize].0 * 2;

            // Where the moves for this edge can be written. One of the two ends must
            // belong to the edge alone, which is what splitting critical edges buys:
            // if the predecessor branches only here, the end of it is the edge; if
            // the successor is entered only from here, the top of it is.
            let at = if succs.len() == 1 {
                // Sound only because a block with a single successor ends in a bare
                // control transfer: an operand on it would be read *after* these
                // moves had overwritten the registers holding it.
                debug_assert!(
                    f.uses(term).is_empty() && f.defs(term).is_empty(),
                    "mb{}'s terminator has operands, so edge moves cannot precede it",
                    pred.0,
                );
                ProgPoint::before(term)
            } else if npreds[succ.0 as usize] == 1 {
                ProgPoint::before(*f.block_insts(succ).first().expect("block is non-empty"))
            } else {
                // Neither end is private to the edge. `isel` splits these, so
                // reaching here means it did not.
                return Err(RegallocError::OutOfRegisters);
            };

            let params = f.block_params(succ);
            let args = f.jump_args(pred);
            debug_assert!(
                params.is_empty() || args.len() == params.len(),
                "edge arity"
            );

            // Every value live where the successor starts, not only its parameters.
            //
            // Without splitting the two are equivalent: any other value holds one
            // location for the whole of its life, so both ends of the edge agree by
            // construction and the comparison below always yields nothing. With
            // splitting they are not — a value split differently on two paths
            // disagrees at the join exactly as a parameter does, and Wimmer10 Fig. 7
            // resolves the two the same way. That is also what makes Braun09's
            // coupling code fall out for free: a value in a register on one side and
            // a slot on the other *is* a reload, and needs no separate list.
            let mut pending: Vec<(EdgeSrc, Alloc, RegClass)> = Vec::new();
            for v in 0..f.num_vregs() {
                let vr = VReg(v as u32);
                if !ranges[v].iter().any(|&(lo, hi)| lo <= entry && entry < hi) {
                    continue;
                }
                let Some(to) = locs.get(vr, entry) else {
                    continue;
                };

                match params.iter().position(|&p| p == vr) {
                    // A parameter is delivered from the matching argument.
                    Some(k) => {
                        let a = args[k];
                        // A rematerializable argument that was spilled has *no* home
                        // to move from — it is replayed at each mention instead.
                        // Reading a location for it would find nothing, or worse a
                        // stale register from before it was evicted.
                        if remat_spilled[a.0 as usize] {
                            let src = remat_src[a.0 as usize].expect("a remat value names its def");
                            pending.push((EdgeSrc::Remat(a, src), to, f.class(vr)));
                            continue;
                        }
                        let from = locs
                            .get(a, at_pos)
                            .expect("a live edge argument has a home at the branch");
                        // The coalesced case: argument and parameter already share a
                        // location, so the edge costs nothing.
                        if from != to {
                            pending.push((EdgeSrc::Loc(from), to, f.class(vr)));
                        }
                    }
                    // Anything else simply carries on across the edge. A replayed
                    // value is in the same non-place on both sides, so it needs
                    // nothing; everything else needs a move only where it moved.
                    None => {
                        if remat_spilled[v] {
                            continue;
                        }
                        if let Some(from) = locs.get(vr, at_pos)
                            && from != to
                        {
                            pending.push((EdgeSrc::Loc(from), to, f.class(vr)));
                        }
                    }
                }
            }
            if pending.is_empty() {
                continue;
            }

            // Registers holding a live value at the end of this block, plus every
            // location these moves themselves name. A scratch must avoid all of them.
            let mut busy: Vec<PReg> = Vec::new();
            for v in 0..f.num_vregs() {
                // A location is not the truth for a rematerializable value that was
                // spilled: it keeps whatever register the value held *before* it was
                // evicted, and that register is long since somebody else's. Counting it
                // here would reserve a register nothing occupies — which on a
                // 13-register machine is the difference between finding a scratch and
                // declining.
                if remat_spilled[v] {
                    continue;
                }
                if let Some(Alloc::Reg(r)) = locs.get(VReg(v as u32), at_pos)
                    && ranges[v]
                        .iter()
                        .any(|&(lo, hi)| lo <= at_pos && at_pos < hi)
                {
                    busy.push(r);
                }
            }
            for &(from, to, _) in &pending {
                if let EdgeSrc::Loc(Alloc::Reg(r)) = from {
                    busy.push(r);
                }
                if let Alloc::Reg(r) = to {
                    busy.push(r);
                }
            }
            // A register nothing needs across this edge, if the machine has one left.
            let free_reg = |class: RegClass| -> Option<PReg> {
                env.order(class).iter().copied().find(|r| !busy.contains(r))
            };

            // Emit every move whose destination nothing else still has to read; when
            // none qualifies, what remains is a permutation cycle, which one scratch
            // location breaks.
            let mut seq: Vec<(EdgeSrc, Alloc, RegClass)> = Vec::new();
            while !pending.is_empty() {
                let srcs: Vec<Alloc> = pending
                    .iter()
                    .filter_map(|&(from, _, _)| match from {
                        EdgeSrc::Loc(a) => Some(a),
                        // A replay reads nothing, so it constrains no ordering.
                        EdgeSrc::Remat(..) => None,
                    })
                    .collect();
                let (ready, blocked): (Vec<_>, Vec<_>) = pending
                    .into_iter()
                    .partition(|&(_, to, _)| !srcs.contains(&to));

                if ready.is_empty() {
                    // Only real moves can form a cycle — a replay has no source to be
                    // waited on — so everything blocked here is a `Loc`.
                    let (from, _, class) = blocked[0];
                    let EdgeSrc::Loc(from) = from else {
                        unreachable!("a replay cannot be part of a permutation cycle")
                    };
                    // A cycle needs somewhere to park one value, not specifically a
                    // register: `reg -> slot` and `slot -> reg` are both legal, so a
                    // fresh slot breaks it without disturbing anything else.
                    let tmp = match free_reg(class) {
                        Some(r) => Alloc::Reg(r),
                        None => {
                            let s = Alloc::Spill(*spills);
                            *spills += 1;
                            s
                        }
                    };
                    seq.push((EdgeSrc::Loc(from), tmp, class));
                    pending = blocked
                        .into_iter()
                        .map(|(f_, t_, c)| {
                            if f_ == EdgeSrc::Loc(from) {
                                (EdgeSrc::Loc(tmp), t_, c)
                            } else {
                                (f_, t_, c)
                            }
                        })
                        .collect();
                    continue;
                }
                seq.extend(ready);
                pending = blocked;
            }

            // Split out the moves that genuinely need a *register* to pass through: a
            // slot-to-slot move (no machine here moves memory to memory) and a replay
            // into a slot (the instruction being replayed writes a register). The rest
            // are emitted as they stand.
            let mut needs_reg: Vec<(EdgeSrc, Alloc, RegClass)> = Vec::new();
            for (from, to, class) in seq {
                match (from, to) {
                    (EdgeSrc::Remat(val, src), Alloc::Reg(r)) => {
                        b.edit(at, Edit::Remat { val, src, to: r });
                    }
                    (EdgeSrc::Loc(Alloc::Spill(_)), Alloc::Spill(_))
                    | (EdgeSrc::Remat(..), Alloc::Spill(_)) => needs_reg.push((from, to, class)),
                    (EdgeSrc::Loc(from), _) => {
                        b.edit(at, Edit::Move(Move { from, to, class }));
                    }
                }
            }

            // Now the ones that need a register, after every ordinary move is done. If
            // none is free the register file is saturated — which is exactly when this
            // path is reached — so one is *bounced*: saved to a fresh slot, borrowed,
            // restored. Whatever it held, a live-through value or a parameter just
            // delivered, comes back untouched.
            //
            // Any register will do, precisely because it is saved and restored, and an
            // edge has no operands of its own to work around. That is what makes edge
            // resolution total: unlike an over-subscribed instruction, an edge can
            // always be resolved, so it never declines the region.
            //
            // Edits at one program point keep the order they were pushed, which is what
            // makes the save and the restore actually bracket the borrow.
            for class in RegClass::ALL {
                let group: Vec<_> = needs_reg
                    .iter()
                    .copied()
                    .filter(|&(_, _, c)| c == class)
                    .collect();
                if group.is_empty() {
                    continue;
                }
                let (reg, bounced_to) = match free_reg(class) {
                    Some(r) => (r, None),
                    None => {
                        let r = env.order(class)[0];
                        let slot = Alloc::Spill(*spills);
                        *spills += 1;
                        b.edit(
                            at,
                            Edit::Move(Move {
                                from: Alloc::Reg(r),
                                to: slot,
                                class,
                            }),
                        );
                        (r, Some(slot))
                    }
                };
                for (from, to, c) in group {
                    match from {
                        EdgeSrc::Remat(val, src) => b.edit(at, Edit::Remat { val, src, to: reg }),
                        EdgeSrc::Loc(from) => b.edit(
                            at,
                            Edit::Move(Move {
                                from,
                                to: Alloc::Reg(reg),
                                class: c,
                            }),
                        ),
                    }
                    b.edit(
                        at,
                        Edit::Move(Move {
                            from: Alloc::Reg(reg),
                            to,
                            class: c,
                        }),
                    );
                }
                if let Some(slot) = bounced_to {
                    b.edit(
                        at,
                        Edit::Move(Move {
                            from: slot,
                            to: Alloc::Reg(reg),
                            class,
                        }),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Live-in sets, to a fixpoint.
///
/// **Not a code path.** This is the pre-SSA liveness analysis, kept only as the
/// independent oracle [`build_intervals`] is checked against in tests: it arrives
/// at the same information by a completely different route, so agreement between
/// them is evidence and not tautology. Nothing outside `#[cfg(test)]` calls it.
#[cfg(test)]
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
    type State = HashMap<Alloc, VReg, RandomState>;
    let mut entry_state: Vec<Option<State>> = vec![None; f.num_blocks()];
    entry_state[f.entry().0 as usize] = Some(State::default());

    loop {
        let mut changed = false;
        for &b in &order {
            let Some(before) = entry_state[b.0 as usize].clone() else {
                continue;
            };
            let after = transfer(f, ra, b, before, &mut |_| Ok(())).expect("no errors reported");

            for s in f.succs(b) {
                let crossed = cross_edge(f, ra, b, s, after.clone());
                let merged = match &entry_state[s.0 as usize] {
                    None => crossed.clone(),
                    Some(old) => old
                        .iter()
                        .filter(|(loc, v)| crossed.get(loc) == Some(v))
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

/// A predecessor's exit state, as the successor sees it on entry.
///
/// The edge's resolving moves have already run by this point (they sit before the
/// terminator), so each parameter's location holds the *argument's* value. Crossing
/// the edge renames it: the parameter is what the successor's instructions read.
///
/// Renaming rather than seeding is what makes this a real check. If resolution
/// failed to deliver an argument, the location holds something else and the rename
/// does not fire, so the successor's first use of that parameter reports a mismatch
/// instead of being quietly papered over.
fn cross_edge(
    f: &impl RegallocFunc,
    ra: &Allocation,
    pred: Block,
    succ: Block,
    mut state: HashMap<Alloc, VReg, RandomState>,
) -> HashMap<Alloc, VReg, RandomState> {
    let params = f.block_params(succ);
    if params.is_empty() {
        return state;
    }
    let args = f.jump_args(pred);
    let renamed: Vec<(Alloc, VReg)> = params
        .iter()
        .enumerate()
        .filter_map(|(k, &p)| {
            let at = ra.block_param(succ, k);
            let arg = *args.get(k)?;
            // The location must hold the argument feeding this parameter — moved
            // there by resolution, or already there because the two coalesced.
            (state.get(&at) == Some(&arg)).then_some((at, p))
        })
        .collect();
    // Drop every parameter location first: one that resolution did not satisfy must
    // not keep whatever stale value it happens to hold.
    for k in 0..params.len() {
        state.remove(&ra.block_param(succ, k));
    }
    state.extend(renamed);
    state
}

/// Run one block's instructions over the symbolic state, reporting each use that
/// reads a location not holding the value it wants.
fn transfer(
    f: &impl RegallocFunc,
    ra: &Allocation,
    b: Block,
    mut state: HashMap<Alloc, VReg, RandomState>,
    report: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<HashMap<Alloc, VReg, RandomState>, String> {
    for &i in f.block_insts(b) {
        for e in ra.edits_at(ProgPoint::before(i)) {
            apply_edit(&mut state, e);
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

        // A temp register is written during the instruction, so whatever a value
        // left there is gone — the same as a clobber. If a temp collided with a
        // live value, the next reader of that value finds nothing here.
        for k in 0..f.temps(i).len() {
            state.remove(&ra.temp(i, k));
        }

        for e in ra.edits_at(ProgPoint::after(i)) {
            apply_edit(&mut state, e);
        }
    }
    Ok(state)
}

/// Replay a fix-up on the symbolic state. A move copies: both ends hold the value
/// afterward, and whatever `to` held is gone — which the checker notices at the
/// next use of it, if there is one. A remat reconstitutes its named value in `to`
/// from nothing, so it holds regardless of what was reachable before.
fn apply_edit(state: &mut HashMap<Alloc, VReg, RandomState>, e: &Edit) {
    match *e {
        Edit::Move(m) => match state.get(&m.from).copied() {
            Some(v) => {
                state.insert(m.to, v);
            }
            None => {
                state.remove(&m.to);
            }
        },
        Edit::Remat { val, to, .. } => {
            state.insert(Alloc::Reg(to), val);
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

        // A temp is scratch: it always gets a register, never the stack.
        for k in 0..f.temps(i).len() {
            match ra.temp(i, k) {
                Alloc::Reg(_) => {}
                a => return Err(format!("inst {i}: temp {k} must be a register, got {a}")),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testfunc::TestFunc;
    use super::*;

    /// An edge is a copy that the machine IR holds no instruction for — the only
    /// record of it is the argument list. Without an affinity
    /// across it the parameter takes whatever register is free first, which is not
    /// the argument's, and the edge pays for a move that never needed to exist.
    ///
    /// Here `a` and `b` interfere, so they get different registers; `a` is then dead
    /// at the target, so the lowest free register there is `a`'s, not `b`'s. The
    /// parameter must follow `b` anyway.
    #[test]
    fn an_edge_coalesces_its_argument_with_its_parameter() {
        let mut f = TestFunc::default();
        let (entry, target) = (f.block(), f.block());
        let (a, b, param) = (f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(a)], vec![]);
        f.inst(entry, vec![Operand::any(b)], vec![]);
        let both = f.inst(entry, vec![], vec![Operand::any(a), Operand::any(b)]);
        // A terminator, as every machine-IR block has: the edge's moves go before
        // it, so it must not be an instruction that still reads anything.
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[target]);
        f.pass(entry, &[b]);

        f.params(target, &[param]);
        f.inst(target, vec![], vec![Operand::any(param)]);
        f.inst(target, vec![], vec![]);

        let ra = allocate(&f, &env(4)).expect("allocates");
        verify(&f, &ra).expect("verifies");

        assert_ne!(
            ra.use_(both, 0),
            ra.use_(both, 1),
            "a and b interfere, so the test is only meaningful if they differ"
        );
        assert_eq!(
            ra.block_param(target, 0),
            ra.use_(both, 1),
            "the parameter must land in its argument's register, not the first free one",
        );
    }

    /// The safety property. A parameter still read after its back-edge argument is
    /// computed genuinely interferes with it, so the merge must be refused and the
    /// edge must pay for a real move. Coalescing that ignored interference would
    /// put two simultaneously-live values in one register and silently lose one.
    #[test]
    fn coalescing_refuses_an_interfering_pair() {
        let mut f = TestFunc::default();
        // `latch` is the edge block a conditional branch's successors always are:
        // only a single-successor block may carry arguments.
        let (entry, header, latch, exit) = (f.block(), f.block(), f.block(), f.block());
        let (init, param, next) = (f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(init)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);
        f.pass(entry, &[init]);

        f.params(header, &[param]);
        f.inst(header, vec![Operand::any(next)], vec![Operand::any(param)]);
        // The parameter is read again, after `next` exists: the two are live at the
        // same time and cannot share a register.
        f.inst(header, vec![], vec![Operand::any(param)]);
        f.inst(header, vec![], vec![]);
        f.goto(header, &[latch, exit]);

        f.inst(latch, vec![], vec![]);
        f.goto(latch, &[header]);
        f.pass(latch, &[next]);

        f.inst(exit, vec![], vec![]);

        let ra = allocate(&f, &env(4)).expect("allocates");
        verify(&f, &ra).expect("verifies");

        let term = *f.block_insts(latch).last().unwrap();
        assert!(
            ra.edits_at(ProgPoint::before(term)).count() > 0,
            "the back edge must move: parameter and argument interfere",
        );
    }

    /// Edge resolution must be *total*: when the register file is saturated and a
    /// move still has to pass through a register, one is bounced — saved to a fresh
    /// slot, borrowed, restored — rather than the region being declined.
    ///
    /// The setup: `live1`/`live2` are live across everything and take both
    /// registers, while `param` and its back-edge argument `next` interfere (the
    /// parameter is read after `next` is computed) so coalescing must refuse them
    /// and both land in slots. The back edge then needs a slot-to-slot move with no
    /// register free for it.
    ///
    /// x86-64 declined `mix2` on exactly this shape before the bounce existed.
    #[test]
    fn a_saturated_edge_bounces_a_register_rather_than_declining() {
        let mut f = TestFunc::default();
        let (entry, header, latch, exit) = (f.block(), f.block(), f.block(), f.block());
        let (i1, i2) = (f.int(), f.int());
        let (p1, p2, n1, n2) = (f.int(), f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(i1)], vec![]);
        f.inst(entry, vec![Operand::any(i2)], vec![]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[header]);
        f.pass(entry, &[i1, i2]);

        f.params(header, &[p1, p2]);
        f.inst(header, vec![Operand::any(n1)], vec![Operand::any(p1)]);
        f.inst(header, vec![Operand::any(n2)], vec![Operand::any(p2)]);
        // Each parameter is read again after its back-edge argument exists, so the
        // two interfere and coalescing must refuse them.
        f.inst(header, vec![], vec![Operand::any(p1), Operand::any(p2)]);
        f.inst(header, vec![], vec![]);
        f.goto(header, &[latch, exit]);

        f.inst(latch, vec![], vec![]);
        f.goto(latch, &[header]);
        f.pass(latch, &[n1, n2]);

        f.inst(exit, vec![], vec![]);

        // One register for four simultaneously-live values: the back edge has to
        // move slot to slot with nothing free to route through.
        let ra = allocate(&f, &env(1)).expect("a saturated edge must not decline");
        verify(&f, &ra).expect("the bounced register must be restored intact");

        // The precondition, so this cannot quietly stop testing the bounce.
        let term = *f.block_insts(latch).last().unwrap();
        let edits: Vec<&Edit> = ra.edits_at(ProgPoint::before(term)).collect();
        let saved = edits.iter().find_map(|e| match e {
            Edit::Move(m) => match (m.from, m.to) {
                (Alloc::Reg(r), Alloc::Spill(_)) => Some(r),
                _ => None,
            },
            _ => None,
        });
        let restored = edits.iter().rev().find_map(|e| match e {
            Edit::Move(m) => match (m.from, m.to) {
                (Alloc::Spill(_), Alloc::Reg(r)) => Some(r),
                _ => None,
            },
            _ => None,
        });
        assert!(saved.is_some(), "no bounce happened: {edits:?}");
        assert_eq!(
            saved, restored,
            "the edge must restore the register it borrowed: {edits:?}",
        );
    }

    /// An edge argument that is a *rematerializable* constant has no home to move
    /// from: it carries no spill slot, and `loc` still names whatever register it
    /// held before it was evicted — long since somebody else's. The edge has to
    /// replay it, exactly as a use does.
    ///
    /// Found the hard way. Reading `loc` for such an argument silently delivered
    /// nothing, so the parameter's slot was never written and its first use read an
    /// uninitialised stack slot. It surfaced only on x86-64's 13-register pool,
    /// where `mix` spills; aarch64 has too many registers to reach it.
    #[test]
    fn an_edge_replays_a_rematerializable_argument() {
        let mut f = TestFunc::default();
        let (entry, target) = (f.block(), f.block());
        let (k, x, y, param) = (f.int(), f.int(), f.int(), f.int());

        // `k` is a constant the allocator may recompute rather than spill. Every
        // mention must be `Reg`-constrained or remat is disabled for it — an `Any`
        // mention could read a slot the replay never writes.
        let def_k = f.inst(entry, vec![Operand::reg(k)], vec![]);
        f.set_remat(k, def_k);
        f.inst(entry, vec![Operand::any(x)], vec![]);
        f.inst(entry, vec![Operand::any(y)], vec![]);
        // Both registers are taken here, so `k` — live across it to the edge — has
        // to be evicted, and being rematerializable it gets no slot.
        f.inst(entry, vec![], vec![Operand::any(x), Operand::any(y)]);
        f.inst(entry, vec![], vec![]);
        f.goto(entry, &[target]);
        f.pass(entry, &[k]);

        f.params(target, &[param]);
        f.inst(target, vec![], vec![Operand::reg(param)]);
        f.inst(target, vec![], vec![]);

        let ra = allocate(&f, &env(2)).expect("allocates");

        // The precondition, asserted so this cannot quietly stop testing anything:
        // the edge really does have to replay the argument rather than move it.
        let term = *f.block_insts(entry).last().unwrap();
        assert!(
            ra.edits_at(ProgPoint::before(term))
                .any(|e| matches!(e, Edit::Remat { val, .. } if *val == k)),
            "the setup should force `k` to be rematerialized on the edge",
        );

        // `verify` is the real assertion: it follows the value symbolically, so a
        // parameter that was never delivered reads a location holding nothing.
        verify(&f, &ra).expect("the edge must deliver the parameter");
    }

    /// Whether two merged interval lists share any position.
    fn overlaps(a: &[(u32, u32)], b: &[(u32, u32)]) -> bool {
        a.iter()
            .any(|&(al, ah)| b.iter().any(|&(bl, bh)| al < bh && bl < ah))
    }

    /// Positions and block spans, exactly as `allocate` computes them.
    fn axis(f: &impl RegallocFunc) -> (Vec<Block>, Vec<u32>, Vec<(u32, u32)>) {
        let order = f.block_order();
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
        (order, pos, span)
    }

    fn ssa_ranges(f: &impl RegallocFunc) -> (Vec<Vec<(u32, u32)>>, Vec<(u32, u32)>) {
        let (order, pos, span) = axis(f);
        let ranges = build_intervals(f, &order, &pos, &span)
            .into_iter()
            .map(merge_ranges)
            .collect();
        (ranges, span)
    }

    /// The one-pass construction must not lose liveness the dataflow fixpoint
    /// finds. Containment, not equality: the loop case deliberately
    /// over-approximates — everything live at a header is given the whole loop — so
    /// the intervals are a superset. It is the missing direction that would
    /// miscompile, so that is the direction asserted.
    fn assert_intervals_cover_liveness(f: &impl RegallocFunc) {
        let (order, _, span) = axis(f);
        let (ranges, _) = ssa_ranges(f);
        let live_in = liveness(f, &order);

        for &b in &order {
            let at = span[b.0 as usize].0 * 2;
            for &v in &live_in[b.0 as usize] {
                assert!(
                    ranges[v.0 as usize]
                        .iter()
                        .any(|&(lo, hi)| lo <= at && at < hi),
                    "v{} is live-in at mb{} (position {at}) but its interval {:?} does not cover it",
                    v.0,
                    b.0,
                    ranges[v.0 as usize],
                );
            }
        }
    }

    /// The straight-line case: no loop, so the one pass is trivially complete.
    #[test]
    fn intervals_match_liveness_without_loops() {
        assert_intervals_cover_liveness(&add_func());
    }

    /// The case the fixpoint existed for. Processing the body in reverse sees the
    /// header's live-in as still empty, so `carried` is only rescued by the
    /// loop-header range add.
    #[test]
    fn intervals_cover_a_value_live_across_a_back_edge() {
        let mut f = TestFunc::default();
        let (entry, body, exit) = (f.block(), f.block(), f.block());
        let (carried, scratch) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(carried)], vec![]);
        f.goto(entry, &[body]);
        f.inst(body, vec![Operand::any(scratch)], vec![]);
        f.inst(
            body,
            vec![],
            vec![Operand::any(scratch), Operand::any(carried)],
        );
        f.goto(body, &[body, exit]);
        f.inst(exit, vec![], vec![Operand::any(carried)]);

        assert_intervals_cover_liveness(&f);

        let (ranges, span) = ssa_ranges(&f);
        let (body_from, body_to) = span[body.0 as usize];
        assert!(
            ranges[carried.0 as usize]
                .iter()
                .any(|&(lo, hi)| lo <= body_from * 2 && hi >= body_to * 2),
            "carried must be live across the entire loop body, got {:?}",
            ranges[carried.0 as usize],
        );
    }

    /// A block parameter is defined at its block's entry and its argument dies at
    /// the end of the predecessor. The two must *not* overlap — that
    /// non-interference is what lets an edge cost nothing.
    #[test]
    fn a_block_parameter_does_not_overlap_its_argument() {
        let mut f = TestFunc::default();
        let (entry, target) = (f.block(), f.block());
        let (arg, param) = (f.int(), f.int());

        f.inst(entry, vec![Operand::any(arg)], vec![]);
        f.goto(entry, &[target]);
        f.pass(entry, &[arg]);
        f.params(target, &[param]);
        f.inst(target, vec![], vec![Operand::any(param)]);

        assert!(f.has_block_params());

        let (ranges, span) = ssa_ranges(&f);
        assert!(
            !overlaps(&ranges[arg.0 as usize], &ranges[param.0 as usize]),
            "argument {:?} and parameter {:?} must not interfere",
            ranges[arg.0 as usize],
            ranges[param.0 as usize],
        );
        assert_eq!(ranges[param.0 as usize][0].0, span[target.0 as usize].0 * 2);
    }

    /// A loop-carried parameter whose last use precedes the definition of its
    /// back-edge argument does *not* interfere with it — the parameter is dead
    /// through the tail of the body, where the next iteration's value is computed.
    ///
    /// This hole is the whole point: it is what lets both ends share one register
    /// and turns the back-edge copy into nothing at all.
    #[test]
    fn a_loop_carried_parameter_can_share_its_arguments_register() {
        let mut f = TestFunc::default();
        let (entry, header, exit) = (f.block(), f.block(), f.block());
        let (init, param, next) = (f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(init)], vec![]);
        f.goto(entry, &[header]);
        f.pass(entry, &[init]);

        f.params(header, &[param]);
        // `next = op(param)` — the parameter's last use is *before* next's def.
        f.inst(header, vec![Operand::any(next)], vec![Operand::any(param)]);
        f.goto(header, &[header, exit]);
        f.pass(header, &[next]);
        f.inst(exit, vec![], vec![]);

        let (ranges, span) = ssa_ranges(&f);
        assert!(
            !overlaps(&ranges[param.0 as usize], &ranges[next.0 as usize]),
            "parameter {:?} and its back-edge argument {:?} should not interfere",
            ranges[param.0 as usize],
            ranges[next.0 as usize],
        );
        assert_eq!(
            ranges[param.0 as usize][0].0,
            span[header.0 as usize].0 * 2,
            "the parameter is still defined at the header's entry",
        );
    }

    /// The converse, so the hole above is not mistaken for a blanket rule: when the
    /// parameter is still read *after* its back-edge argument is computed, the two
    /// genuinely interfere and the edge really does need a move. Splitting cannot
    /// remove this one.
    #[test]
    fn a_parameter_read_after_its_argument_is_defined_interferes() {
        let mut f = TestFunc::default();
        let (entry, header, exit) = (f.block(), f.block(), f.block());
        let (init, param, next) = (f.int(), f.int(), f.int());

        f.inst(entry, vec![Operand::any(init)], vec![]);
        f.goto(entry, &[header]);
        f.pass(entry, &[init]);

        f.params(header, &[param]);
        f.inst(header, vec![Operand::any(next)], vec![Operand::any(param)]);
        f.inst(header, vec![], vec![Operand::any(param)]);
        f.goto(header, &[header, exit]);
        f.pass(header, &[next]);
        f.inst(exit, vec![], vec![]);

        let (ranges, _) = ssa_ranges(&f);
        assert!(
            overlaps(&ranges[param.0 as usize], &ranges[next.0 as usize]),
            "parameter {:?} outlives its argument's definition {:?} and must interfere",
            ranges[param.0 as usize],
            ranges[next.0 as usize],
        );
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
        let ra = allocate(&f, &env(4)).expect("no exotic constraints");
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

        let ra = allocate(&f, &env(3)).expect("no exotic constraints");
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

        let ra = allocate(&f, &env(4)).expect("no exotic constraints");
        verify(&f, &ra).expect("the loop-carried value must survive the body");
        assert_ne!(ra.use_(2, 0), ra.use_(2, 1));
    }

    /// A temp gets its own register, distinct from every operand of the
    /// instruction — so the encoder can scribble in it without destroying an input
    /// or the result.
    #[test]
    fn a_temp_gets_a_register_clear_of_the_operands() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (x, y, d) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(x)], vec![]);
        f.inst(b, vec![Operand::any(y)], vec![]);
        // `d = x op y` needing one scratch register.
        let op = f.inst(
            b,
            vec![Operand::reg(d)],
            vec![Operand::reg(x), Operand::reg(y)],
        );
        f.temp(op, 1);
        f.inst(b, vec![], vec![Operand::any(d)]);

        let ra = allocate(&f, &env(4)).expect("room for two inputs, a result and a temp");
        verify(&f, &ra).expect("the temp must not alias an operand");

        let t = ra.temp(op, 0);
        assert!(matches!(t, Alloc::Reg(_)), "a temp is always a register");
        for slot in [ra.def(op, 0), ra.use_(op, 0), ra.use_(op, 1)] {
            assert_ne!(t, slot, "the temp must be clear of the operands");
        }
    }

    /// When the temp cannot be placed — every register taken by values live across
    /// the instruction — the allocator declines rather than inventing scratch.
    #[test]
    fn an_unplaceable_temp_declines() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (x, y) = (f.int(), f.int());

        f.inst(b, vec![Operand::any(x)], vec![]);
        f.inst(b, vec![Operand::any(y)], vec![]);
        // Two live inputs into two registers, and a temp that needs a third.
        let op = f.inst(b, vec![], vec![Operand::reg(x), Operand::reg(y)]);
        f.temp(op, 1);

        assert_eq!(
            allocate(&f, &env(2)).err(),
            Some(RegallocError::OutOfRegisters),
            "no register for the temp is a decline, not a silent alias"
        );
    }

    /// A value forced to the stack under pressure still has to be in a register at
    /// a `Reg` use — the allocator loads it back into whatever is free there,
    /// without any register having been reserved as scratch for the purpose.
    #[test]
    fn a_spilled_value_reloads_at_its_register_use() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (v0, v1, v2) = (f.int(), f.int(), f.int());

        // Three values live at once into two registers: one spills, and by the
        // scan's order that is `v2`.
        f.inst(b, vec![Operand::any(v0)], vec![]);
        f.inst(b, vec![Operand::any(v1)], vec![]);
        f.inst(b, vec![Operand::any(v2)], vec![]);
        f.inst(b, vec![], vec![Operand::reg(v0)]);
        f.inst(b, vec![], vec![Operand::reg(v1)]);
        let u2 = f.inst(b, vec![], vec![Operand::reg(v2)]);

        let ra = allocate(&f, &env(2)).expect("two registers is enough with a reload");
        verify(&f, &ra).expect("the reload keeps the value where the use reads it");

        assert!(
            matches!(ra.use_(u2, 0), Alloc::Reg(_)),
            "the Reg use must be in a register, not on the stack"
        );
        assert!(
            ra.edits_at(ProgPoint::before(u2)).next().is_some(),
            "a reload edit must precede the use"
        );
    }

    /// A soft physical hint lands when the register is free, without being pinned:
    /// the value would take `x0` by first-fit, but the hint pulls it to `x2`.
    #[test]
    fn a_soft_phys_hint_is_honoured_when_it_fits() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        let x2 = PReg::new(RegClass::Int, 2);

        f.inst(b, vec![Operand::any(v)], vec![]);
        f.inst(b, vec![], vec![Operand::any(v)]);
        f.hint(v, x2);

        let ra = allocate(&f, &env(4)).expect("no exotic constraints");
        verify(&f, &ra).expect("the hint changes where, not whether");
        assert_eq!(ra.use_(1, 0), Alloc::Reg(x2), "the free hint should win");
    }

    /// A value live across a clobbering instruction — a call, an exit stub — must
    /// not be in a register that instruction destroys.
    #[test]
    fn a_value_avoids_a_register_clobbered_across_it() {
        let mut f = TestFunc::default();
        let b = f.block();
        let carried = f.int();
        let x0 = PReg::new(RegClass::Int, 0);

        f.inst(b, vec![Operand::any(carried)], vec![]);
        let c = f.inst(b, vec![], vec![]);
        f.clobber(c, x0);
        f.inst(b, vec![], vec![Operand::any(carried)]);

        let ra = allocate(&f, &env(4)).expect("clobbers are supported");
        verify(&f, &ra).expect("a value in a clobbered register would be destroyed");
        assert_ne!(
            ra.use_(2, 0),
            Alloc::Reg(x0),
            "carried is live across the clobber, so it must dodge x0"
        );
    }

    /// As a property: whatever the allocator decides, the checker accepts.
    #[test]
    fn allocation_passes_the_checker() {
        let f = add_func();
        let ra = allocate(&f, &env(4)).expect("no exotic constraints");
        verify(&f, &ra).expect("the allocator must satisfy the checker's invariant");
    }

    /// A *split* allocation: the value is defined in one register and read from
    /// another, with a move in between. The allocator here produces none — this
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
                    Edit::Move(Move {
                        from: Alloc::Reg(x2),
                        to: Alloc::Reg(x3),
                        class: RegClass::Int,
                    }),
                );
            }
            b.finish(0)
        };

        verify(&f, &split(true)).expect("the move carries v out of x2 before w lands in it");
        let err = verify(&f, &split(false)).expect_err("without the move x3 holds nothing");
        assert!(err.contains("nothing"), "{err}");
    }

    /// `Reuse` describes a def taking a source's register; on a *use* it is
    /// meaningless, and the allocator says so rather than ignoring it.
    #[test]
    fn a_reuse_on_a_use_is_declined() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        f.inst(b, vec![Operand::any(v)], vec![]);
        f.inst(b, vec![], vec![Operand::reuse(v, 0)]);

        assert_eq!(
            allocate(&f, &env(4)).err(),
            Some(RegallocError::UnsupportedConstraint(Constraint::Reuse(0))),
            "reuse is a def-only relationship"
        );
    }

    /// A `Fixed` use puts the value in the demanded register for the instruction,
    /// with a move off its home to get it there, and leaves the home alone.
    #[test]
    fn a_fixed_use_moves_the_value_into_its_register() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        let x0 = PReg::new(RegClass::Int, 0);

        f.inst(b, vec![Operand::any(v)], vec![]);
        let u = f.inst(b, vec![], vec![Operand::fixed(v, x0)]);

        let ra = allocate(&f, &env(4)).expect("a fixed operand is honoured");
        verify(&f, &ra).expect("the value really is in x0 at the use");
        assert_eq!(
            ra.use_(u, 0),
            Alloc::Reg(x0),
            "the use reads x0 as demanded"
        );
        assert!(
            ra.edits_at(ProgPoint::before(u)).next().is_some(),
            "a move brings the value into x0 before the use"
        );
    }

    /// A `Fixed` def is written in the demanded register and then ferried to the
    /// value's home, which — being clear of that register — a later use reads.
    #[test]
    fn a_fixed_def_ferries_the_value_out_of_its_register() {
        let mut f = TestFunc::default();
        let b = f.block();
        let v = f.int();
        let x0 = PReg::new(RegClass::Int, 0);

        let d = f.inst(b, vec![Operand::fixed(v, x0)], vec![]);
        let u = f.inst(b, vec![], vec![Operand::reg(v)]);

        let ra = allocate(&f, &env(4)).expect("a fixed def is honoured");
        verify(&f, &ra).expect("the result is carried out of x0 intact");
        assert_eq!(ra.def(d, 0), Alloc::Reg(x0), "the def writes x0");
        assert_ne!(ra.use_(u, 0), Alloc::Reg(x0), "the home is clear of x0");
        assert!(
            ra.edits_at(ProgPoint::after(d)).next().is_some(),
            "a move carries the result out of x0"
        );
    }

    /// A value live across a fixed-register instruction must dodge that register —
    /// the fixed mention would otherwise overwrite it.
    #[test]
    fn a_fixed_register_is_kept_clear_of_a_live_value() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (carried, v) = (f.int(), f.int());
        let x0 = PReg::new(RegClass::Int, 0);

        f.inst(b, vec![Operand::any(carried)], vec![]);
        f.inst(b, vec![Operand::any(v)], vec![]);
        f.inst(b, vec![], vec![Operand::fixed(v, x0)]); // v pinned to x0 here
        f.inst(b, vec![], vec![Operand::any(carried)]); // carried still needed

        let ra = allocate(&f, &env(4)).expect("room to keep carried clear of x0");
        verify(&f, &ra).expect("carried survives the fixed use");
        assert_ne!(
            ra.use_(3, 0),
            Alloc::Reg(x0),
            "carried is live across the fixed use, so it dodges x0"
        );
    }

    /// A two-address `Reuse` def takes its dead source's register: the def and the
    /// reused use share a register, and coalescing removes any separate move.
    #[test]
    fn a_reuse_def_takes_its_dead_sources_register() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (s1, s2, dst) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(s1)], vec![]);
        f.inst(b, vec![Operand::any(s2)], vec![]);
        // dst = s1 OP s2 as one two-address op; s1 dies here.
        let op = f.inst(
            b,
            vec![Operand::reuse(dst, 0)],
            vec![Operand::reg(s1), Operand::reg(s2)],
        );
        f.inst(b, vec![], vec![Operand::any(dst)]);

        let ra = allocate(&f, &env(4)).expect("a reuse def is honoured");
        verify(&f, &ra).expect("the two-address result is where the reuse says");
        assert_eq!(
            ra.def(op, 0),
            ra.use_(op, 0),
            "the def and its reused source share a register"
        );
        assert!(
            ra.edits_at(ProgPoint::after(op)).next().is_none(),
            "coalescing put the def in its reused register: no reconciling move"
        );
    }

    /// A reused source that is still live is *copied* into the def's register, not
    /// consumed: the two-address op overwrites the copy while the original survives
    /// for its later use. The def still shares a register with the reused operand.
    #[test]
    fn a_reuse_copies_a_live_source_instead_of_consuming_it() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (s1, s2, dst) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(s1)], vec![]);
        f.inst(b, vec![Operand::any(s2)], vec![]);
        let op = f.inst(
            b,
            vec![Operand::reuse(dst, 0)],
            vec![Operand::reg(s1), Operand::reg(s2)],
        );
        f.inst(b, vec![], vec![Operand::any(dst)]);
        let us1 = f.inst(b, vec![], vec![Operand::reg(s1)]); // s1 still live afterward

        let ra = allocate(&f, &env(4)).expect("a live reused source is handled");
        verify(&f, &ra).expect("s1 survives the op that reused it");
        assert_eq!(
            ra.def(op, 0),
            ra.use_(op, 0),
            "the def still shares the reused operand's register"
        );
        assert_ne!(
            ra.def(op, 0),
            ra.use_(us1, 0),
            "but that register is a copy — s1's own register is untouched"
        );
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

    /// Belady: when one of several live values must go to the stack, spill the one
    /// whose next use is furthest off. Here `far` is read last, so it is the victim,
    /// and `near` keeps its register untouched.
    #[test]
    fn eviction_spills_the_furthest_use() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (near, c, far) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(near)], vec![]);
        f.inst(b, vec![Operand::any(far)], vec![]);
        f.inst(b, vec![Operand::any(c)], vec![]); // near, far, c all live now
        let un = f.inst(b, vec![], vec![Operand::reg(near)]);
        f.inst(b, vec![], vec![Operand::reg(c)]);
        let uf = f.inst(b, vec![], vec![Operand::reg(far)]);

        let ra = allocate(&f, &env(2)).expect("two registers with one eviction");
        verify(&f, &ra).expect("the evicted value still reads back correctly");

        assert!(
            ra.edits_at(ProgPoint::before(un)).next().is_none(),
            "the nearest use keeps its register, no reload"
        );
        assert!(
            ra.edits_at(ProgPoint::before(uf)).next().is_some(),
            "the furthest use is the one that had to spill and reload"
        );
    }

    /// A temp with nowhere to go used to be an outright decline. Now it may evict a
    /// value that *can* live on the stack — here a live-through value that is not an
    /// operand of the instruction — and only declines when nothing is evictable.
    #[test]
    fn a_temp_evicts_a_spillable_neighbour_instead_of_declining() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (w, x) = (f.int(), f.int());

        f.inst(b, vec![Operand::any(w)], vec![]);
        f.inst(b, vec![Operand::any(x)], vec![]);
        // `x` is read here in a register and the instruction needs a scratch temp;
        // `w` is live across it (read next), holding the other of two registers.
        let op = f.inst(b, vec![], vec![Operand::reg(x)]);
        f.temp(op, 1);
        f.inst(b, vec![], vec![Operand::any(w)]);

        let ra = allocate(&f, &env(2)).expect("the temp evicts w rather than declining");
        verify(&f, &ra).expect("w reads back from the stack after being evicted");

        assert!(
            matches!(ra.temp(op, 0), Alloc::Reg(_)),
            "the temp got a real register"
        );
        assert!(
            matches!(ra.use_(3, 0), Alloc::Spill(_)),
            "w was the one sent to the stack"
        );
    }

    /// The reload phase must never fail when the machine has the registers. When a
    /// spilled value's use sits where every register holds a value live across it,
    /// one is bounced to a scratch slot for the instruction and restored right
    /// after — no register is ever reserved for the purpose.
    #[test]
    fn a_reload_bounces_a_live_value_when_no_register_is_free() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (a, bb, c) = (f.int(), f.int(), f.int());

        f.inst(b, vec![Operand::any(a)], vec![]);
        f.inst(b, vec![Operand::any(bb)], vec![]);
        f.inst(b, vec![Operand::any(c)], vec![]); // a, bb, c all live; c spills (furthest)
        let uc = f.inst(b, vec![], vec![Operand::reg(c)]); // a, bb live across this
        f.inst(b, vec![], vec![Operand::reg(a)]);
        f.inst(b, vec![], vec![Operand::reg(bb)]);
        f.inst(b, vec![], vec![Operand::reg(c)]); // c read again last: furthest end

        let ra = allocate(&f, &env(2)).expect("a bounce makes the reload fit");
        verify(&f, &ra).expect("the bounced value is restored intact");

        assert!(
            matches!(ra.use_(uc, 0), Alloc::Reg(_)),
            "the reg use of c is in a register"
        );
        assert!(
            ra.edits_at(ProgPoint::before(uc)).count() >= 2,
            "a save and a reload precede the flanked use"
        );
        assert!(
            ra.edits_at(ProgPoint::after(uc)).next().is_some(),
            "the bounced value is restored after the instruction"
        );
    }

    /// A rematerializable constant that spills needs no slot and no store: each use
    /// replays the constant. Being cheap to bring back, it is also the preferred
    /// eviction victim.
    #[test]
    fn a_rematerializable_constant_reloads_without_a_slot() {
        let mut f = TestFunc::default();
        let b = f.block();
        let (k, a, bb) = (f.int(), f.int(), f.int());

        let def_k = f.inst(b, vec![Operand::reg(k)], vec![]); // a constant
        f.set_remat(k, def_k);
        f.inst(b, vec![Operand::any(a)], vec![]);
        f.inst(b, vec![Operand::any(bb)], vec![]); // k, a, bb live; k is evicted (remat)
        f.inst(b, vec![], vec![Operand::reg(a)]);
        f.inst(b, vec![], vec![Operand::reg(bb)]);
        let uk = f.inst(b, vec![], vec![Operand::reg(k)]);

        let ra = allocate(&f, &env(2)).expect("k rematerializes, a and bb keep registers");
        verify(&f, &ra).expect("the replayed constant is the value the use reads");

        assert_eq!(
            ra.num_spills, 0,
            "a rematerialized value claims no spill slot"
        );
        assert!(
            matches!(
                ra.edits_at(ProgPoint::before(uk)).next(),
                Some(Edit::Remat { val, .. }) if *val == k
            ),
            "k comes back by replaying its constant, not by loading a slot"
        );
    }

    /// The eviction cost model protects loop-resident values: `hot` spans the loop
    /// and is read after it, so Belady's furthest-use rule alone would spill it —
    /// but its loop weight keeps it, and the non-loop `cold` is spilled instead.
    #[test]
    fn a_loop_resident_value_is_protected_from_eviction() {
        let mut f = TestFunc::default();
        let (entry, loop_, exit) = (f.block(), f.block(), f.block());

        let hot = f.int();
        let cold = f.int();
        f.inst(entry, vec![Operand::any(hot)], vec![]);
        f.goto(entry, &[loop_]);

        let uhot_loop = f.inst(loop_, vec![], vec![Operand::reg(hot)]);
        f.goto(loop_, &[loop_, exit]);

        f.inst(exit, vec![Operand::any(cold)], vec![]); // cold born after the loop
        let ucold = f.inst(exit, vec![], vec![Operand::reg(cold)]);
        let uhot = f.inst(exit, vec![], vec![Operand::reg(hot)]); // hot read last of all

        let ra = allocate(&f, &env(1)).expect("one register, resolved by a spill");
        verify(&f, &ra).expect("whichever value spilled still reads back");

        // `hot` keeps one register throughout; `cold`, not the loop value, is the
        // one that had to move.
        assert_eq!(
            ra.use_(uhot_loop, 0),
            ra.use_(uhot, 0),
            "hot stays in a single register across the whole function"
        );
        assert!(
            ra.edits_at(ProgPoint::before(ucold)).next().is_some(),
            "cold is the value spilled and reloaded, not hot"
        );
    }
}
