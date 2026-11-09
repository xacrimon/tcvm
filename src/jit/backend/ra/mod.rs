//! The register allocator (M1, `m1-regalloc.md`): an SSA linear scan with
//! V8's split and spill positions (TurboFan `register-allocator.cc`) over a
//! `VCode`, writing an `Output`.
//!
//! Positions: instruction `n` of the linear order has an early point `2n`
//! (uses read, early defs and temporaries written) and a late point
//! `2n + 1` (late uses read, defs written, clobbers); moves go in the gap
//! before `2n`. A value holds its location over half-open intervals of
//! points. A spilled value is stored once, after its definition, so a
//! register piece may end at any point without a store; only a reload
//! needs a gap.
//!
//! Storage: a group's intervals and uses are runs of two flat arrays, and
//! a piece is a window of its group's runs, so splitting copies nothing.
//! A `Ctx` keeps every buffer across compiles: emptied, not freed.

/// Empty the listed buffers, keeping their storage.
macro_rules! clear {
    ($s:expr; $($f:ident),* $(,)?) => { $($s.$f.clear();)* };
}

mod build;
#[cfg(feature = "tcvm-dev")]
pub(crate) mod checker;
#[cfg(feature = "tcvm-dev")]
pub(crate) mod corpus;
mod resolve;
mod scan;
#[cfg(test)]
mod tests;

use crate::jit::backend::reg::{Allocation, Block, Inst, PReg, ProgPoint, RegClass, VReg};
use crate::jit::backend::vcode::{TargetInst, VCode};

/// What goes between two instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edit {
    Move {
        from: Allocation,
        to: Allocation,
    },
    /// Compute `vreg` again into `to`, as `VCode::remat` says, from `src`
    /// where it is computed from a value (none for a constant).
    Remat {
        to: Allocation,
        vreg: VReg,
        src: Allocation,
    },
}

/// The allocation: each operand's location, the edits between
/// instructions, sorted by program point, and those on the edges of
/// branches into joins, by (branch, edge).
#[derive(Debug)]
pub(crate) struct Output {
    pub(crate) num_spillslots: usize,
    pub(crate) edits: Vec<(ProgPoint, Edit)>,
    pub(crate) edge_edits: Vec<(Inst, u8, Edit)>,
    pub(crate) allocs: Vec<Allocation>,
}

pub(crate) enum InstOrEdit<'a> {
    Inst(Inst),
    Edit(&'a Edit),
}

impl Output {
    pub(crate) fn inst_allocs<I: TargetInst>(&self, f: &VCode<I>, inst: Inst) -> &[Allocation] {
        let (a, b) = f.ops[inst.index()];
        &self.allocs[a as usize..b as usize]
    }

    /// The edits on edge `k` of the branch `inst`.
    pub(crate) fn edge_edits(&self, inst: Inst, k: usize) -> &[(Inst, u8, Edit)] {
        let key = (inst, k as u8);
        let a = self.edge_edits.partition_point(|e| (e.0, e.1) < key);
        let n = self.edge_edits[a..].partition_point(|e| (e.0, e.1) == key);
        &self.edge_edits[a..a + n]
    }

    /// A block's instructions with the edits before each.
    pub(crate) fn block_insts_and_edits<'a, I: TargetInst>(
        &'a self,
        f: &VCode<I>,
        block: Block,
    ) -> impl Iterator<Item = InstOrEdit<'a>> + 'a {
        let range = f.block_insns(block);
        let first = ProgPoint::before(range.first());
        let mut k = self.edits.partition_point(|e| e.0 < first);
        let mut insts = range.iter().peekable();
        std::iter::from_fn(move || {
            let &next = insts.peek()?;
            if let Some((p, e)) = self.edits.get(k)
                && *p <= ProgPoint::before(next)
            {
                k += 1;
                return Some(InstOrEdit::Edit(e));
            }
            insts.next().map(InstOrEdit::Inst)
        })
    }
}

type Pos = u32;
const NONE: u32 = u32::MAX;

/// Registers per class the allocator can track (as bits of a `u32`).
const MAX_REGS: usize = 32;

/// The registers the allocator hands out and the ones its moves borrow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Env {
    /// Allocatable registers per class (int, float), preferred first.
    pub(crate) regs: [Vec<PReg>; 2],
    /// Per class, never allocated: holds one value while a cycle of moves
    /// turns.
    pub(crate) scratch: [PReg; 2],
    /// An integer register never allocated, carrying stack-to-stack moves
    /// (64 bits, any class).
    pub(crate) transport: PReg,
}

fn class_index(c: RegClass) -> usize {
    match c {
        RegClass::Int => 0,
        RegClass::Float => 1,
        RegClass::Vector => unreachable!("no vector registers"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Loc {
    #[default]
    None,
    Reg(PReg),
    Stack,
    /// Computed again where needed (a constant): no slot, no store.
    Remat,
}

impl Loc {
    /// Out of a register: on the stack or to be computed again.
    fn spilled(self) -> bool {
        matches!(self, Loc::Stack | Loc::Remat)
    }
}

/// What a use position asks of the location. Operands taking the value from
/// wherever it is or from its slot (snapshots, a call's kept values) are no
/// uses here: they only extend the range (`Ra::snaps`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    /// A register: a register use or a definition.
    Reg,
    /// The definition of a value computed again where needed.
    Any,
    /// Copied into the register before the instruction; being there
    /// already saves the copy.
    Fixed(PReg),
}

/// A use: its point, and in one word the operand's index in
/// `Output::allocs` (22 bits), its `Need`'s tag and fixed register.
#[derive(Clone, Copy)]
struct Use {
    pos: Pos,
    bits: u32,
}

impl Use {
    fn new(pos: Pos, alloc: u32, need: Need) -> Use {
        let mut u = Use { pos, bits: alloc };
        u.set_need(need);
        u
    }

    fn alloc(self) -> u32 {
        self.bits & 0x3f_ffff
    }

    fn need(self) -> Need {
        match self.bits >> 22 & 3 {
            0 => Need::Reg,
            1 => Need::Any,
            _ => Need::Fixed(PReg::from_index((self.bits >> 24) as usize)),
        }
    }

    fn set_need(&mut self, need: Need) {
        let (tag, r) = match need {
            Need::Reg => (0, 0),
            Need::Any => (1, 0),
            Need::Fixed(r) => (2, r.index() as u32),
        };
        self.bits = self.bits & 0x3f_ffff | tag << 22 | r << 24;
    }
}

/// One piece of a group's live range, with one location: the group's
/// intervals `lo..hi` cut to `start..end` (only the first and the last
/// can be), and its uses `ulo..uhi`.
#[derive(Clone, Copy)]
struct Piece {
    g: u32,
    class: u8,
    /// `FIXED`: a fixed definition's point in its register, never moved or
    /// spilled. `STACK`: taken from its slot by some operand (`resolve`).
    flags: u8,
    start: Pos,
    end: Pos,
    lo: u32,
    hi: u32,
    ulo: u32,
    uhi: u32,
    /// The group's pieces by start.
    prev: u32,
    next: u32,
    loc: Loc,
}

const FIXED: u8 = 1;
const STACK: u8 = 2;

impl Piece {
    /// Merged back into the piece before it.
    fn dead(&self) -> bool {
        self.lo == self.hi
    }

    fn ivs<'a>(&self, ivs: &'a [(Pos, Pos)]) -> Ivs<'a> {
        Ivs {
            raw: &ivs[self.lo as usize..self.hi as usize],
            start: self.start,
            end: self.end,
        }
    }

    fn uses<'a>(&self, uses: &'a [Use]) -> &'a [Use] {
        &uses[self.ulo as usize..self.uhi as usize]
    }
}

/// Sorted, disjoint, not adjacent intervals, cut to `start..end`.
#[derive(Clone, Copy)]
struct Ivs<'a> {
    raw: &'a [(Pos, Pos)],
    start: Pos,
    end: Pos,
}

impl<'a> Ivs<'a> {
    fn plain(raw: &'a [(Pos, Pos)]) -> Self {
        Ivs {
            raw,
            start: 0,
            end: Pos::MAX,
        }
    }

    fn len(&self) -> usize {
        self.raw.len()
    }

    fn get(&self, k: usize) -> (Pos, Pos) {
        let (a, b) = self.raw[k];
        (a.max(self.start), b.min(self.end))
    }

    /// The first interval ending after `p`.
    fn after(&self, p: Pos) -> usize {
        if p >= self.end {
            return self.raw.len();
        }
        self.raw.partition_point(|iv| iv.1 <= p)
    }

    fn covers(&self, p: Pos) -> bool {
        self.start <= p && p < self.end && self.raw[self.after(p)].0 <= p
    }

    /// The first point of `lo..hi` covered.
    fn first_in(&self, lo: Pos, hi: Pos) -> Option<Pos> {
        let k = self.after(lo);
        if lo >= hi || k == self.len() {
            return None;
        }
        let (a, b) = self.get(k);
        let p = a.max(lo);
        (p < b.min(hi)).then_some(p)
    }
}

/// The uses `next_use` looks for: needing a register, served by one (any
/// but a constant's definition), wanting a fixed one.
#[derive(Clone, Copy)]
enum Want {
    Reg,
    Beneficial,
    Fixed,
}

/// The first point at or after `from` both interval lists cover.
fn intersect(a: Ivs, b: Ivs, from: Pos) -> Option<Pos> {
    let (mut i, mut j) = (a.after(from), b.after(from));
    while i < a.len() && j < b.len() {
        let (x, y) = (a.get(i), b.get(j));
        let lo = x.0.max(y.0).max(from);
        if lo < x.1.min(y.1) {
            return Some(lo);
        }
        if x.1 <= y.1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    None
}

/// A piece in `active` or `inactive` with its register (`env.regs[cls][k]`)
/// and where it next changes sides: the end of the interval it is in, or
/// the start of its next. A listed piece is not split or merged.
#[derive(Clone, Copy)]
struct Held {
    x: u32,
    until: Pos,
    cls: u8,
    k: u8,
}

/// A group of values sharing one allocation (`build`), by its
/// representative; the fields after `class` are the group's.
struct VInfo {
    class: u8,
    /// A constant, computed again instead of spilled.
    remat: bool,
    /// A parameter every argument of which is in its group: their stores
    /// after their definitions already filled the slot.
    args_in_group: bool,
    /// The point it is defined at, `NONE` for a block parameter.
    def: Pos,
    /// The block whose parameter it is, else `NONE`.
    param_of: u32,
    /// Its first piece, `NONE` for none.
    head: u32,
    /// Its spill slot, once spilled anywhere or used from the stack.
    slot: u32,
}

/// The allocator, kept by the JIT runtime: the state of the function in
/// hand and each phase's working buffers.
pub(crate) struct Ctx(std::mem::ManuallyDrop<Parts>);

struct Parts {
    ra: Ra,
    live: Live,
    build: build::Build,
    resolve: resolve::Resolve,
}

impl Drop for Ctx {
    /// Freeing the many buffers is a lot of code: once, here, not in every
    /// crate that drops a state.
    #[inline(never)]
    fn drop(&mut self) {
        // SAFETY: dropped once, here.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.0) }
    }
}

impl Ctx {
    pub(crate) fn new(env: Env) -> Self {
        Ctx(std::mem::ManuallyDrop::new(Parts {
            ra: Ra::new(env),
            live: Live::default(),
            build: build::Build::default(),
            resolve: resolve::Resolve::default(),
        }))
    }

    pub(crate) fn env(&self) -> &Env {
        &self.0.ra.env
    }

    /// Allocate `f`'s registers, its blocks taken in `order`: the first
    /// `hot` of them hot, the rest (exits, cold arms) after them all.
    pub(crate) fn run<I: TargetInst>(
        &mut self,
        f: &VCode<I>,
        order: &[u32],
        hot: usize,
    ) -> Result<Output, String> {
        let Parts {
            ra,
            live,
            build,
            resolve,
        } = &mut *self.0;
        ra.clear();
        live.clear();
        build.clear();
        resolve.clear();
        ra.layout(f, order, hot)?;
        ra.liveness(f, live)?;
        ra.build(f, build)?;
        ra.scan(f)?;
        ra.resolve(f, resolve)
    }
}

/// `liveness`'s buffers: each block's gen set, its kill values (a run
/// `koff[b]..koff[b + 1]` of `kill`), a block's live-out, the worklist and
/// the blocks on it.
#[derive(Default)]
struct Live {
    gen_: Vec<u64>,
    kill: Vec<u32>,
    koff: Vec<u32>,
    out: Vec<u64>,
    queued: Vec<bool>,
    work: std::collections::VecDeque<u32>,
}

impl Live {
    fn clear(&mut self) {
        clear!(self; gen_, kill, koff, out, queued, work);
    }
}

/// The function in hand; `allocs` leaves in its `Output`.
struct Ra {
    env: Env,
    /// Blocks in allocation order.
    order: Vec<Block>,
    /// Position of each block in `order`.
    rank: Vec<u32>,
    /// First and last linear index of each block.
    first: Vec<u32>,
    last: Vec<u32>,
    /// The block of each linear index, and its instruction.
    lin_block: Vec<u32>,
    at_lin: Vec<Inst>,
    /// Each block's static frequency, and where cold code starts.
    freq: Vec<f32>,
    cold_start: Pos,
    /// Header of the innermost loop holding each block (itself for a
    /// header), else `NONE`; the enclosing loop's header of each header
    /// (`VCode::loop_of`, `loop_parent`).
    inner: Vec<u32>,
    outer: Vec<u32>,
    /// Per loop header: the end of the last hot block of its loop, 0 for
    /// none.
    hot_end: Vec<Pos>,
    words: usize,
    live_in: Vec<u64>,
    /// The values some use takes from wherever they are (`Any`, `Stack`).
    anyuse: Vec<u64>,
    /// Intervals, a run per group, and uses, a run per value with a merged
    /// group's after them (`Piece`).
    ivs: Vec<(Pos, Pos)>,
    uses: Vec<Use>,
    /// Per value (a run `snap_off[v]..snap_off[v + 1]`, by point) the
    /// operands taking it from wherever it is, as `operand << 1 | from its
    /// slot`: placed after the scan from the piece at their point.
    snaps: Vec<(Pos, u32)>,
    snap_off: Vec<u32>,
    /// Per use: the index of the first use from it on (in its group) that
    /// needs a register and that is fixed; past the group's last otherwise.
    next_of: Vec<[u32; 2]>,
    /// Per group (runs by `_off`): its members' definition points, sorted
    /// (a piece may start at one without a move); its members, ascending;
    /// its pieces by start, after the scan, and their starts.
    defs: Vec<Pos>,
    defs_off: Vec<u32>,
    members: Vec<u32>,
    members_off: Vec<u32>,
    pieces: Vec<Piece>,
    flat: Vec<u32>,
    flat_off: Vec<u32>,
    flat_start: Vec<Pos>,
    /// Per value; a group's pieces and slot are its representative's.
    vregs: Vec<VInfo>,
    /// Each value's group (`build`): itself, or the parameter it joined.
    rep: Vec<u32>,
    /// Per group: stored where it goes from a register to the stack, not
    /// after its definitions.
    deferred: Vec<bool>,
    /// Points fixed operands reserve allocatable registers over, per class
    /// by `from`: `(from, to, owner, k)` for `env.regs[c][k]`; the owner
    /// vreg may sit there (a fixed operand of its own).
    fixres: [Vec<(Pos, Pos, u32, u8)>; 2],
    /// Per class, the registers (as `env.regs` bits) with any, and the ones
    /// any clobber names.
    fixregs: [u32; 2],
    clobregs: [u32; 2],
    /// Where `fixed_conflicts` left off in each class's reservations and in
    /// `clobs`: it asks at the scan's position, which only grows.
    fixcur: [std::cell::Cell<u32>; 2],
    clobcur: std::cell::Cell<u32>,
    /// Late points of clobbering instructions, by point, with the
    /// registers they take per class as bits of `env.regs` indices.
    clobs: Vec<(Pos, [u32; 2])>,
    /// Per `PReg::index()`: its place in `env.regs` of its class, or `NONE`.
    reg_idx: Vec<u32>,
    allocs: Vec<Allocation>,
    num_slots: u32,
    /// Pieces to allocate by start: those `build` made, sorted, and a heap
    /// of the ones queued since.
    initial: Vec<u64>,
    next_initial: usize,
    unhandled: std::collections::BinaryHeap<std::cmp::Reverse<u64>>,
    /// Pieces holding their register at the scan's position, and those in
    /// a hole there.
    active: Vec<Held>,
    inactive: Vec<Held>,
    /// No piece of `active` or `inactive` changes sides or retires before
    /// this point (V8's `next_active_ranges_change_` and inactive's).
    next_change: Pos,
    scratch: Vec<Held>,
}

/// Operands of one instruction that may not share a register: an early
/// definition with any use, a late definition with a late use, two
/// definitions.
#[cfg(feature = "tcvm-dev")]
pub(crate) fn check_operands<I: TargetInst>(v: &VCode<I>, out: &Output) -> Result<(), String> {
    use crate::jit::backend::reg::OperandKind::{Def, Use};
    use crate::jit::backend::reg::OperandPos::{Early, Late};
    for i in 0..v.insts.len() {
        let ops = v.inst_operands_of(i);
        let al = out.inst_allocs(v, Inst::new(i));
        for a in 0..ops.len() {
            for b in a + 1..ops.len() {
                let (oa, ob) = (ops[a], ops[b]);
                if !al[a].is_reg() || al[a] != al[b] || oa.vreg() == ob.vreg() {
                    continue;
                }
                let clash = matches!(
                    (oa.kind(), oa.pos(), ob.kind(), ob.pos()),
                    (Def, Early, Use, _)
                        | (Use, _, Def, Early)
                        | (Def, Late, Use, Late)
                        | (Use, Late, Def, Late)
                        | (Def, _, Def, _)
                );
                if clash {
                    return Err(format!("inst {i}: {oa:?} and {ob:?} share {:?}", al[a]));
                }
            }
        }
    }
    Ok(())
}

/// For the checker: per vreg, whether it may be computed again, and the
/// value it is computed from.
#[cfg(feature = "tcvm-dev")]
pub(crate) fn remat_table<I: TargetInst>(f: &VCode<I>) -> (Vec<bool>, Vec<Option<VReg>>) {
    use crate::jit::backend::vcode::Remat;
    let remat = f.remat.iter().map(|&r| r != Remat::None).collect();
    let src = (f.remat.iter())
        .map(|&r| match r {
            Remat::AndImm { src, .. } => {
                Some(VReg::new(src as usize, f.vreg_classes[src as usize]))
            }
            _ => None,
        })
        .collect();
    (remat, src)
}

impl Ra {
    fn new(env: Env) -> Self {
        let mut reg_idx = vec![NONE; PReg::NUM_INDEX];
        for regs in &env.regs {
            assert!(regs.len() <= MAX_REGS, "more registers than a mask holds");
            for (k, r) in regs.iter().enumerate() {
                reg_idx[r.index()] = k as u32;
            }
        }
        Ra {
            env,
            order: Vec::new(),
            rank: Vec::new(),
            first: Vec::new(),
            last: Vec::new(),
            lin_block: Vec::new(),
            at_lin: Vec::new(),
            freq: Vec::new(),
            cold_start: Pos::MAX,
            inner: Vec::new(),
            outer: Vec::new(),
            hot_end: Vec::new(),
            words: 0,
            live_in: Vec::new(),
            anyuse: Vec::new(),
            ivs: Vec::new(),
            uses: Vec::new(),
            snaps: Vec::new(),
            snap_off: Vec::new(),
            next_of: Vec::new(),
            defs: Vec::new(),
            defs_off: Vec::new(),
            members: Vec::new(),
            members_off: Vec::new(),
            pieces: Vec::new(),
            flat: Vec::new(),
            flat_off: Vec::new(),
            flat_start: Vec::new(),
            vregs: Vec::new(),
            rep: Vec::new(),
            deferred: Vec::new(),
            fixres: [Vec::new(), Vec::new()],
            fixregs: [0; 2],
            clobregs: [0; 2],
            fixcur: [std::cell::Cell::new(0), std::cell::Cell::new(0)],
            clobcur: std::cell::Cell::new(0),
            clobs: Vec::new(),
            reg_idx,
            allocs: Vec::new(),
            num_slots: 0,
            initial: Vec::new(),
            next_initial: 0,
            unhandled: std::collections::BinaryHeap::new(),
            active: Vec::new(),
            inactive: Vec::new(),
            next_change: 0,
            scratch: Vec::new(),
        }
    }

    /// Ready for another function.
    fn clear(&mut self) {
        clear!(self; order, rank, first, last, lin_block, at_lin, freq, inner, outer, hot_end,
            live_in, anyuse, ivs, uses, snaps, snap_off, next_of, defs, defs_off, members,
            members_off, pieces, flat, flat_off, flat_start, vregs, rep, deferred, clobs, initial,
            unhandled, active, inactive, scratch);
        for l in &mut self.fixres {
            l.clear();
        }
        self.cold_start = Pos::MAX;
        self.words = 0;
        self.fixregs = [0; 2];
        self.clobregs = [0; 2];
        self.fixcur = [std::cell::Cell::new(0), std::cell::Cell::new(0)];
        self.clobcur.set(0);
        self.num_slots = 0;
        self.next_initial = 0;
        self.next_change = 0;
    }

    fn block_of_pos(&self, p: Pos) -> u32 {
        self.lin_block[(p / 2) as usize]
    }

    fn block_start(&self, b: u32) -> Pos {
        2 * self.first[b as usize]
    }

    fn block_end(&self, b: u32) -> Pos {
        2 * self.last[b as usize] + 2
    }

    /// The linear order (blocks as numbered: `order.rs` lays out the
    /// allocator's order), its loops (`VCode::loop_of`), and the static
    /// frequencies.
    fn layout<I: TargetInst>(
        &mut self,
        f: &VCode<I>,
        order: &[u32],
        hot: usize,
    ) -> Result<(), String> {
        let nb = f.num_blocks();
        if order.len() != nb {
            return Err("an order not of every block".into());
        }
        if f.loop_of.len() != nb || f.loop_parent.len() != nb {
            return Err("no loop nesting".into());
        }
        self.inner.extend_from_slice(&f.loop_of);
        self.outer.extend_from_slice(&f.loop_parent);
        self.order
            .extend(order.iter().map(|&b| Block::new(b as usize)));
        self.rank.resize(nb, NONE);
        for (k, &b) in order.iter().enumerate() {
            self.rank[b as usize] = k as u32;
        }
        self.first.resize(nb, 0);
        self.last.resize(nb, 0);
        self.at_lin.reserve(f.num_insts());
        self.lin_block.reserve(f.num_insts());
        for &b in &self.order {
            let r = f.block_insns(b);
            if r.len() == 0 {
                return Err(format!("empty block {}", b.index()));
            }
            self.first[b.index()] = self.at_lin.len() as u32;
            for i in r.iter() {
                self.at_lin.push(i);
                self.lin_block.push(b.index() as u32);
            }
            self.last[b.index()] = self.at_lin.len() as u32 - 1;
        }
        // Static frequencies: eight times per loop level, cold code next to
        // nothing.
        self.cold_start = order.get(hot).map_or(Pos::MAX, |&b| self.block_start(b));
        let (rank, inner, outer) = (&self.rank, &self.inner, &self.outer);
        self.freq.extend((0..nb).map(|b| {
            if rank[b] as usize >= hot {
                return 1.0 / 1024.0;
            }
            let mut depth = 0;
            let mut h = inner[b];
            while h != NONE {
                depth += 1;
                h = outer[h as usize];
            }
            8f32.powi(depth.min(6))
        }));
        self.hot_end.resize(nb, 0);
        for &b in &order[..hot.min(nb)] {
            let end = self.block_end(b);
            let mut h = self.inner[b as usize];
            while h != NONE {
                self.hot_end[h as usize] = self.hot_end[h as usize].max(end);
                h = self.outer[h as usize];
            }
        }
        Ok(())
    }

    /// The frequency of the block holding `p`.
    fn freq_at(&self, p: Pos) -> f32 {
        self.freq[self.block_of_pos(p) as usize]
    }

    fn is_header(&self, b: u32) -> bool {
        self.inner[b as usize] == b
    }

    /// V8's `GetContainingLoop`: the header of the loop holding `b`, the
    /// enclosing one for a header.
    fn containing_loop(&self, b: u32) -> u32 {
        if self.is_header(b) {
            self.outer[b as usize]
        } else {
            self.inner[b as usize]
        }
    }

    /// Live-in sets by a backward dataflow fixpoint.
    fn liveness<I: TargetInst>(&mut self, f: &VCode<I>, t: &mut Live) -> Result<(), String> {
        use crate::jit::backend::reg::{OperandConstraint, OperandKind};
        let nb = f.num_blocks();
        let w = f.num_vregs().div_ceil(64);
        self.words = w;
        let Live {
            gen_,
            kill,
            koff,
            out,
            queued,
            work,
        } = t;
        gen_.resize(nb * w, 0);
        let anyuse = &mut self.anyuse;
        anyuse.resize(w, 0);
        out.resize(w, 0);
        koff.reserve_exact(nb + 1);
        koff.push(0);
        let set = |s: &mut [u64], v: usize| s[v / 64] |= 1 << (v % 64);
        let has = |s: &[u64], v: usize| s[v / 64] >> (v % 64) & 1 != 0;
        // Kill is the block's parameters and definitions, each value's one,
        // so a list; gen its uses of values not defined in it before
        // (`out` marks those), a set.
        for b in 0..nb {
            out.fill(0);
            let g = &mut gen_[b * w..(b + 1) * w];
            for p in f.block_params(Block::new(b)) {
                set(out, p.vreg());
                kill.push(p.vreg() as u32);
            }
            for i in f.block_insns(Block::new(b)).iter() {
                let ops = f.inst_operands(i);
                for op in ops {
                    if op.kind() == OperandKind::Use {
                        let v = op.vreg().vreg();
                        if !has(out, v) {
                            set(g, v);
                        }
                        if matches!(
                            op.constraint(),
                            OperandConstraint::Any | OperandConstraint::Stack
                        ) {
                            set(anyuse, v);
                        }
                    }
                }
                for op in ops {
                    if op.kind() == OperandKind::Def {
                        set(out, op.vreg().vreg());
                        kill.push(op.vreg().vreg() as u32);
                    }
                }
            }
            for s in 0..f.block_succs(Block::new(b)).len() {
                for v in f.branch_blockparams(Block::new(b), s) {
                    if !has(out, v.vreg()) {
                        set(g, v.vreg());
                    }
                }
            }
            koff.push(kill.len() as u32);
        }
        // A worklist from the last block: a block is done again only when
        // a successor's set grew.
        let live_in = &mut self.live_in;
        live_in.resize(nb * w, 0);
        queued.resize(nb, true);
        work.extend(self.order.iter().rev().map(|b| b.index() as u32));
        while let Some(b) = work.pop_front() {
            let b = b as usize;
            queued[b] = false;
            let succs = f.block_succs(Block::new(b));
            match succs.split_first() {
                None => out.fill(0),
                Some((s, rest)) => {
                    out.copy_from_slice(&live_in[s.index() * w..(s.index() + 1) * w]);
                    for s in rest {
                        let si = s.index();
                        for (o, x) in out.iter_mut().zip(&live_in[si * w..(si + 1) * w]) {
                            *o |= x;
                        }
                    }
                }
            }
            for &v in &kill[koff[b] as usize..koff[b + 1] as usize] {
                out[v as usize / 64] &= !(1 << (v % 64));
            }
            let mut changed = false;
            for k in 0..w {
                let n = gen_[b * w + k] | out[k];
                changed |= n != live_in[b * w + k];
                live_in[b * w + k] = n;
            }
            if changed {
                for p in f.block_preds(Block::new(b)) {
                    if !std::mem::replace(&mut queued[p.index()], true) {
                        work.push_back(p.index() as u32);
                    }
                }
            }
        }
        let e = f.entry_block().index();
        if live_in[e * w..(e + 1) * w].iter().any(|&x| x != 0) {
            return Err("a value is used before any definition".into());
        }
        Ok(())
    }

    fn live_in(&self, b: usize) -> &[u64] {
        &self.live_in[b * self.words..(b + 1) * self.words]
    }

    fn slot_of(&mut self, v: u32) -> u32 {
        let vi = &mut self.vregs[v as usize];
        if vi.slot == NONE {
            vi.slot = self.num_slots;
            self.num_slots += 1;
        }
        vi.slot
    }

    fn ivs_of(&self, x: u32) -> Ivs<'_> {
        self.pieces[x as usize].ivs(&self.ivs)
    }

    fn uses_of(&self, x: u32) -> &[Use] {
        self.pieces[x as usize].uses(&self.uses)
    }

    fn members_of(&self, g: u32) -> &[u32] {
        let (a, b) = (
            self.members_off[g as usize],
            self.members_off[g as usize + 1],
        );
        &self.members[a as usize..b as usize]
    }

    fn defs_of(&self, g: u32) -> &[Pos] {
        let (a, b) = (self.defs_off[g as usize], self.defs_off[g as usize + 1]);
        &self.defs[a as usize..b as usize]
    }

    fn flat_run(&self, g: u32) -> (u32, u32) {
        (self.flat_off[g as usize], self.flat_off[g as usize + 1])
    }

    fn pieces_of(&self, g: u32) -> &[u32] {
        let (a, b) = self.flat_run(g);
        &self.flat[a as usize..b as usize]
    }

    fn start(&self, x: u32) -> Pos {
        self.pieces[x as usize].start
    }

    fn end(&self, x: u32) -> Pos {
        self.pieces[x as usize].end
    }

    /// The first use of `x` at or after `from` that `want` names.
    fn next_use(&self, x: u32, from: Pos, want: Want) -> Option<Use> {
        let p = &self.pieces[x as usize];
        let k = p.ulo + p.uses(&self.uses).partition_point(|u| u.pos < from) as u32;
        self.want_from(k, p.uhi, want)
            .map(|j| self.uses[j as usize])
    }

    /// The first use index from `k` below `hi` that `want` names.
    fn want_from(&self, k: u32, hi: u32, want: Want) -> Option<u32> {
        if k >= hi {
            return None;
        }
        let [r, f] = self.next_of[k as usize];
        let j = match want {
            Want::Reg => r,
            Want::Fixed => f,
            Want::Beneficial => r.min(f),
        };
        (j < hi).then_some(j)
    }

    /// The piece of group `g` covering `p`: from the index `resolve` makes,
    /// or along the list before it.
    fn piece_at(&self, g: u32, p: Pos) -> Option<u32> {
        let x = if !self.flat_off.is_empty() {
            let ps = self.pieces_of(g);
            let k = ps.partition_point(|&x| self.start(x) <= p);
            if k == 0 {
                return None;
            }
            ps[k - 1]
        } else {
            let (mut x, mut at) = (self.vregs[g as usize].head, NONE);
            while x != NONE && self.start(x) <= p {
                at = x;
                x = self.pieces[x as usize].next;
            }
            if at == NONE {
                return None;
            }
            at
        };
        self.ivs_of(x).covers(p).then_some(x)
    }

    /// Whether a member of group `g` is defined at `p`.
    fn is_def(&self, g: u32, p: Pos) -> bool {
        self.defs_of(g).binary_search(&p).is_ok()
    }

    /// Whether a piece of `g` spilled from `s` can be back in a register
    /// for `p`: a gap lies between, or a member is defined there.
    fn reload_room(&self, g: u32, s: Pos, p: Pos) -> bool {
        (p & !1) > s || p > s && self.is_def(g, p)
    }

    /// `v`'s location at `p` as an allocation.
    fn alloc_at(&self, v: u32, p: Pos) -> Option<Allocation> {
        let x = self.piece_at(self.rep[v as usize], p)?;
        Some(self.alloc_of(x))
    }

    #[inline]
    fn alloc_of(&self, x: u32) -> Allocation {
        let p = &self.pieces[x as usize];
        match p.loc {
            Loc::Reg(r) => Allocation::reg(r),
            Loc::Remat => Allocation::none(),
            Loc::Stack => Allocation::stack(crate::jit::backend::reg::SpillSlot::new(
                self.vregs[p.g as usize].slot as usize,
            )),
            Loc::None => panic!("piece {x} of v{} unallocated", p.g),
        }
    }
}
