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

    /// If this instruction is a pure register-to-register copy — its def `dk`
    /// receives exactly its use `uk`, same class, no other effect — say so.
    ///
    /// The allocator uses it to *coalesce*: place both ends in one register so the
    /// encoder's identity-move elision drops the copy entirely. Target-agnostic on
    /// purpose — the allocator does not know what a `Mov` is, the client does. A
    /// client that answers `None` for everything simply gets no coalescing, which
    /// is what the old scan did.
    fn is_copy(&self, _i: Inst) -> Option<(usize, usize)> {
        None
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
        }
    }

    /// Put every mention of `v` in the same place. What a non-splitting allocator
    /// wants, and all the allocator below asks for.
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
///   - **Hints.** A [`RegallocFunc::is_copy`] links a copy's two ends; whichever is
///     placed first biases the other toward the same register. Because a use of the
///     source ends exactly where the def of the destination begins, the source's
///     register is free at that point, so the bias lands and the encoder drops the
///     move. Loop-carried parameters coalesce the same way: the parameter is placed
///     when its header is seen, and the back-edge argument — computed in the loop
///     tail, inside the parameter's hole — is hinted onto it.
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

    let live_in = liveness(f, &order);

    // Build segments on a doubled axis: instruction `i` uses at `2·pos` and defs at
    // `2·pos + 1`, so a use segment `[.., 2·pos + 1)` and a def segment
    // `[2·pos + 1, ..)` touch without overlapping. That adjacency is what lets a
    // copy's ends — and a two-address op's dying source and its dest — share a
    // register, while a source that outlives the op extends past the def slot and
    // correctly interferes.
    let mut raw: Vec<Vec<(u32, u32)>> = vec![Vec::new(); f.num_vregs()];
    for &b in &order {
        let (bs, be) = span[b.0 as usize];
        let bfrom = bs * 2;
        let bto = be * 2;

        // Live-out is what any successor needs live on entry.
        let mut open: HashMap<VReg, u32, RandomState> = HashMap::default();
        for &s in &f.succs(b) {
            for &v in &live_in[s.0 as usize] {
                open.entry(v).or_insert(bto);
            }
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
            for o in f.uses(i) {
                // First seen going backward is the latest use, hence the furthest
                // end; keep it.
                open.entry(o.vreg).or_insert(slot + 1);
            }
        }

        // Whatever is still open is live from the block's start.
        for (v, end) in open {
            raw[v.0 as usize].push((bfrom, end));
        }
    }

    // Merge once; the scan reads these per class, and the reload phase reads them
    // again to know which register is free where.
    let ranges: Vec<Vec<(u32, u32)>> = raw.into_iter().map(merge_ranges).collect();

    // Copy affinities, both directions: whichever end is placed first pulls the
    // other toward its register.
    let mut affin: HashMap<VReg, Vec<VReg>, RandomState> = HashMap::default();
    for i in 0..f.num_insts() {
        if let Some((dk, uk)) = f.is_copy(i) {
            let d = f.defs(i)[dk].vreg;
            let s = f.uses(i)[uk].vreg;
            affin.entry(d).or_default().push(s);
            affin.entry(s).or_default().push(d);
        }
    }

    // Clobbers, as instruction-wide register reservations. A clobbered register is
    // busy across the whole slot `[2·pos, 2·pos + 2)` — a call destroys it, a
    // macro-op's exit stub writes it — so a value whose interval covers that slot
    // cannot live there. Stored by slot start; the two sub-slots are `lo` and
    // `lo + 1`.
    let mut clobbers: Vec<(u32, PReg)> = Vec::new();
    for (i, &p) in pos.iter().enumerate() {
        let lo = p * 2;
        for &r in f.clobbers(i) {
            clobbers.push((lo, r));
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
            .filter(|v| f.class(*v) == class && !ranges[v.0 as usize].is_empty())
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

    let mut b = AllocationBuilder::new(f);
    for v in 0..f.num_vregs() as u32 {
        if let Some(a) = loc[v as usize] {
            b.assign(VReg(v), a);
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
        for o in f.defs(i).iter().chain(f.uses(i)).chain(f.temps(i)) {
            if let Some(Alloc::Reg(r)) = loc[o.vreg.0 as usize] {
                used.push(r);
            }
        }
        used.extend_from_slice(f.clobbers(i));

        let spilled = |v: VReg| {
            remat_spilled[v.0 as usize] || matches!(loc[v.0 as usize], Some(Alloc::Spill(_)))
        };

        for (k, o) in f.uses(i).iter().enumerate() {
            if !spilled(o.vreg) || o.constraint != Constraint::Reg {
                continue;
            }
            let class = f.class(o.vreg);
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
            // Save (from `reload_reg`) is already in; the load/replay follows it, and
            // the restore follows the load — all at their right points.
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
            } else if let Some(Alloc::Spill(s)) = loc[o.vreg.0 as usize] {
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

        for (k, o) in f.defs(i).iter().enumerate() {
            if !spilled(o.vreg) || o.constraint != Constraint::Reg {
                continue;
            }
            let class = f.class(o.vreg);
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
            // A rematerializable def computes its constant into `r` and drops it —
            // every reader replays it instead — so there is nothing to store.
            if !remat_spilled[o.vreg.0 as usize]
                && let Some(Alloc::Spill(s)) = loc[o.vreg.0 as usize]
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

/// The allocator honours `Any`, `Reg`, and clobbers. It does not yet honour a
/// `Fixed`/`Reuse` operand — x86 will need both — so it declines rather than
/// allocating around one and producing code that runs and is wrong.
fn reject_unsupported(f: &impl RegallocFunc) -> Result<(), RegallocError> {
    for i in 0..f.num_insts() {
        for o in f.defs(i).iter().chain(f.uses(i)) {
            match o.constraint {
                Constraint::Any | Constraint::Reg => {}
                c => return Err(RegallocError::UnsupportedConstraint(c)),
            }
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
        temps: Vec<Vec<Operand>>,
        clobbers: Vec<Vec<PReg>>,
        classes: Vec<RegClass>,
        phys: HashMap<VReg, PReg, RandomState>,
        remat: HashMap<VReg, Inst, RandomState>,
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
            self.temps.push(Vec::new());
            self.clobbers.push(Vec::new());
            self.blocks[b.0 as usize].push(i);
            i
        }

        /// Give instruction `i` `n` fresh integer temp registers.
        fn temp(&mut self, i: Inst, n: usize) {
            for _ in 0..n {
                let t = self.int();
                self.temps[i].push(Operand::reg(t));
            }
        }

        fn goto(&mut self, b: Block, targets: &[Block]) {
            self.succs[b.0 as usize] = targets.to_vec();
        }

        fn clobber(&mut self, i: Inst, r: PReg) {
            self.clobbers[i].push(r);
        }

        fn hint(&mut self, v: VReg, r: PReg) {
            self.phys.insert(v, r);
        }

        /// Mark `v` as rematerializable, defined by instruction `i`.
        fn set_remat(&mut self, v: VReg, i: Inst) {
            self.remat.insert(v, i);
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
        fn phys_hint(&self, v: VReg) -> Option<PReg> {
            self.phys.get(&v).copied()
        }
        fn temps(&self, i: Inst) -> &[Operand] {
            &self.temps[i]
        }
        fn remat(&self, v: VReg) -> Option<Inst> {
            self.remat.get(&v).copied()
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

    /// The allocator declines what it cannot honour, rather than allocating
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

            let declined = allocate(&f, &env(4)).err().expect("must decline");
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
