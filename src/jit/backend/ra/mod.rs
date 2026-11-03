//! The register allocator (M1, `m1-regalloc.md`): an SSA linear scan with
//! V8's split and spill positions (TurboFan `register-allocator.cc`) over a
//! `VCode`, writing regalloc2's `Output`.
//!
//! Positions: instruction `n` of the linear order has an early point `2n`
//! (uses read, early defs and temporaries written) and a late point
//! `2n + 1` (late uses read, defs written, clobbers); moves go in the gap
//! before `2n`. A value holds its location over half-open intervals of
//! points. A spilled value is stored once, after its definition, so a
//! register piece may end at any point without a store; only a reload
//! needs a gap.

mod build;
mod resolve;
mod scan;
#[cfg(test)]
mod tests;

use regalloc2::{Allocation, Block, Function, Inst, Output, PReg, RegClass};

use crate::jit::backend::vcode::{TargetInst, VCode};

type Pos = u32;
const NONE: u32 = u32::MAX;

/// The registers the allocator hands out and the ones its moves borrow.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Loc {
    None,
    Reg(PReg),
    Stack,
}

/// What a use position asks of the location.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Need {
    /// A register: a register use or a definition.
    Reg,
    Any,
    /// The spill slot: the value is stored after its definition.
    Stack,
    /// Copied into the register before the instruction; being there
    /// already saves the copy.
    Fixed(PReg),
}

#[derive(Clone, Copy, Debug)]
struct Use {
    pos: Pos,
    /// The operand's index in `Output::allocs`.
    alloc: u32,
    need: Need,
}

/// One piece of a value's live range, with one location.
#[derive(Debug)]
struct Piece {
    vreg: u32,
    /// Sorted, disjoint, not adjacent.
    ivs: Vec<(Pos, Pos)>,
    uses: Vec<Use>,
    loc: Loc,
}

impl Piece {
    fn start(&self) -> Pos {
        self.ivs[0].0
    }

    fn end(&self) -> Pos {
        self.ivs[self.ivs.len() - 1].1
    }

    fn covers(&self, p: Pos) -> bool {
        let k = self.ivs.partition_point(|iv| iv.1 <= p);
        k < self.ivs.len() && self.ivs[k].0 <= p
    }

    /// The first use at or after `from` that `want` accepts.
    fn next_use(&self, from: Pos, want: impl Fn(Need) -> bool) -> Option<Use> {
        let k = self.uses.partition_point(|u| u.pos < from);
        self.uses[k..].iter().find(|u| want(u.need)).copied()
    }
}

fn needs_reg(n: Need) -> bool {
    n == Need::Reg
}

fn reg_beneficial(n: Need) -> bool {
    matches!(n, Need::Reg | Need::Fixed(_))
}

/// The first point at or after `from` both interval lists cover.
fn intersect(a: &[(Pos, Pos)], b: &[(Pos, Pos)], from: Pos) -> Option<Pos> {
    let mut i = a.partition_point(|iv| iv.1 <= from);
    let mut j = b.partition_point(|iv| iv.1 <= from);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0).max(from);
        if lo < a[i].1.min(b[j].1) {
            return Some(lo);
        }
        if a[i].1 <= b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    None
}

#[derive(Debug)]
struct VInfo {
    class: u8,
    /// Its pieces, by start.
    pieces: Vec<u32>,
    /// Its spill slot, once spilled anywhere or used from the stack.
    slot: u32,
    /// The point it is defined at, `NONE` for a block parameter.
    def: Pos,
    /// The block whose parameter it is, else `NONE`.
    param_of: u32,
    /// A parameter every argument of which is in its group: their stores
    /// after their definitions already filled the slot.
    args_in_group: bool,
}

struct Ra<'a, I: TargetInst> {
    f: &'a VCode<I>,
    env: &'a Env,
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
    /// Header of the innermost loop holding each block (itself for a
    /// header), else `NONE`; the enclosing loop's header of each header.
    inner: Vec<u32>,
    outer: Vec<u32>,
    words: usize,
    live_in: Vec<u64>,
    pieces: Vec<Piece>,
    /// Per value; a group's pieces and slot are its representative's.
    vregs: Vec<VInfo>,
    /// Each value's group (`build`): itself, or the parameter it joined.
    rep: Vec<u32>,
    /// Per `PReg::index()`: points it is reserved over, `(from, to, owner)`
    /// sorted by `from`; the owner vreg may sit there (a fixed operand of
    /// its own), `NONE` owns a clobber.
    fixed: Vec<Vec<(Pos, Pos, u32)>>,
    /// Per `PReg::index()`: its place in `env.regs` of its class, or `NONE`.
    reg_idx: Vec<u32>,
    allocs: Vec<Allocation>,
    alloc_base: Vec<u32>,
    num_slots: u32,
    unhandled: std::collections::BinaryHeap<std::cmp::Reverse<(Pos, u32)>>,
    /// Pieces holding their register at the scan's position, and those in
    /// a hole there.
    active: Vec<u32>,
    inactive: Vec<u32>,
}

/// Allocate `f`'s registers.
pub(crate) fn run<I: TargetInst + std::fmt::Debug>(
    f: &VCode<I>,
    env: &Env,
) -> Result<Output, String> {
    let mut ra = Ra::new(f, env);
    let r = (|| {
        ra.layout()?;
        ra.liveness()?;
        ra.build()?;
        ra.scan()?;
        ra.resolve()
    })();
    if r.is_err() && std::env::var_os("TCVM_RA_DEBUG").is_some() {
        eprintln!("{}", dump(f));
    }
    r
}

/// The function before allocation: blocks, operands, clobbers.
fn dump<I: TargetInst + std::fmt::Debug>(f: &VCode<I>) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for b in 0..f.num_blocks() {
        let rb = Block::new(b);
        let _ = writeln!(
            s,
            "b{b}{:?} <- {:?} -> {:?}",
            f.block_params(rb),
            f.block_preds(rb),
            f.block_succs(rb)
        );
        for i in f.block_insns(rb).iter() {
            let _ = writeln!(
                s,
                "  {:>4} {:?} {:?} clob {:?}",
                i.index(),
                f.insts[i.index()],
                f.inst_operands(i),
                f.inst_clobbers(i)
            );
        }
    }
    s
}

impl<'a, I: TargetInst> Ra<'a, I> {
    fn new(f: &'a VCode<I>, env: &'a Env) -> Self {
        let mut reg_idx = vec![NONE; PReg::NUM_INDEX];
        for regs in &env.regs {
            for (k, r) in regs.iter().enumerate() {
                reg_idx[r.index()] = k as u32;
            }
        }
        Ra {
            f,
            env,
            order: Vec::new(),
            rank: Vec::new(),
            first: Vec::new(),
            last: Vec::new(),
            lin_block: Vec::new(),
            at_lin: Vec::new(),
            inner: Vec::new(),
            outer: Vec::new(),
            words: 0,
            live_in: Vec::new(),
            pieces: Vec::new(),
            vregs: Vec::new(),
            rep: Vec::new(),
            fixed: vec![Vec::new(); PReg::NUM_INDEX],
            reg_idx,
            allocs: Vec::new(),
            alloc_base: Vec::new(),
            num_slots: 0,
            unhandled: std::collections::BinaryHeap::new(),
            active: Vec::new(),
            inactive: Vec::new(),
        }
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
    /// allocator's order) and its loops, from the dominators.
    fn layout(&mut self) -> Result<(), String> {
        let f = self.f;
        let nb = f.num_blocks();
        self.order = (0..nb).map(Block::new).collect();
        self.rank = (0..nb as u32).collect();
        self.first = vec![0; nb];
        self.last = vec![0; nb];
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
        // Dominators (Cooper, Harvey, Kennedy) over reverse postorder.
        let mut post = Vec::with_capacity(nb);
        let mut seen = vec![false; nb];
        let mut stack = vec![(f.entry_block().index(), 0usize)];
        seen[f.entry_block().index()] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let succs = f.block_succs(Block::new(b));
            if *k < succs.len() {
                let s = succs[*k].index();
                *k += 1;
                if !std::mem::replace(&mut seen[s], true) {
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        let mut po = vec![NONE; nb];
        for (k, &b) in post.iter().enumerate() {
            po[b] = k as u32;
        }
        let mut idom = vec![NONE; nb];
        let entry = f.entry_block().index();
        idom[entry] = entry as u32;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in post.iter().rev() {
                if b == entry {
                    continue;
                }
                let mut new = NONE;
                for p in f.block_preds(Block::new(b)) {
                    let p = p.index();
                    if idom[p] == NONE {
                        continue;
                    }
                    new = if new == NONE {
                        p as u32
                    } else {
                        let (mut x, mut y) = (new as usize, p);
                        while x != y {
                            while po[x] < po[y] {
                                x = idom[x] as usize;
                            }
                            while po[y] < po[x] {
                                y = idom[y] as usize;
                            }
                        }
                        x as u32
                    };
                }
                if new != idom[b] {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                return true;
            }
            if b == entry || idom[b] == NONE {
                return false;
            }
            b = idom[b] as usize;
        };
        // Natural loops, one per header; smallest first gives each block its
        // innermost loop on the first write.
        let mut loops: Vec<(usize, Vec<bool>)> = Vec::new();
        for (b, &d) in idom.iter().enumerate() {
            if d == NONE {
                continue;
            }
            for s in f.block_succs(Block::new(b)) {
                let h = s.index();
                if !dominates(h, b) {
                    continue;
                }
                let k = match loops.iter().position(|l| l.0 == h) {
                    Some(k) => k,
                    None => {
                        let mut body = vec![false; nb];
                        body[h] = true;
                        loops.push((h, body));
                        loops.len() - 1
                    }
                };
                let body = &mut loops[k].1;
                let mut work = vec![b];
                while let Some(x) = work.pop() {
                    if !std::mem::replace(&mut body[x], true) {
                        work.extend(f.block_preds(Block::new(x)).iter().map(|p| p.index()));
                    }
                }
            }
        }
        loops.sort_by_key(|l| l.1.iter().filter(|&&x| x).count());
        self.inner = vec![NONE; nb];
        self.outer = vec![NONE; nb];
        for (h, body) in &loops {
            for (x, &inside) in body.iter().enumerate() {
                if inside && self.inner[x] == NONE {
                    self.inner[x] = *h as u32;
                }
            }
        }
        for (k, (h, _)) in loops.iter().enumerate() {
            self.outer[*h] = loops[k + 1..]
                .iter()
                .find(|l| l.1[*h])
                .map_or(NONE, |l| l.0 as u32);
        }
        Ok(())
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
    fn liveness(&mut self) -> Result<(), String> {
        let f = self.f;
        let nb = f.num_blocks();
        let w = f.num_vregs().div_ceil(64);
        self.words = w;
        let mut gen_ = vec![0u64; nb * w];
        let mut kill = vec![0u64; nb * w];
        let set = |s: &mut [u64], v: usize| s[v / 64] |= 1 << (v % 64);
        let has = |s: &[u64], v: usize| s[v / 64] >> (v % 64) & 1 != 0;
        for b in 0..nb {
            let (g, k) = (&mut gen_[b * w..(b + 1) * w], &mut kill[b * w..(b + 1) * w]);
            for p in f.block_params(Block::new(b)) {
                set(k, p.vreg());
            }
            let insns = f.block_insns(Block::new(b));
            for i in insns.iter() {
                for op in f.inst_operands(i) {
                    if op.kind() == regalloc2::OperandKind::Use && !has(k, op.vreg().vreg()) {
                        set(g, op.vreg().vreg());
                    }
                }
                for op in f.inst_operands(i) {
                    if op.kind() == regalloc2::OperandKind::Def {
                        set(k, op.vreg().vreg());
                    }
                }
            }
            let term = insns.last();
            for s in 0..f.block_succs(Block::new(b)).len() {
                for v in f.branch_blockparams(Block::new(b), term, s) {
                    if !has(k, v.vreg()) {
                        set(g, v.vreg());
                    }
                }
            }
        }
        let mut live_in = vec![0u64; nb * w];
        let mut out = vec![0u64; w];
        let mut changed = true;
        while changed {
            changed = false;
            for &b in self.order.iter().rev() {
                let b = b.index();
                out.fill(0);
                for s in f.block_succs(Block::new(b)) {
                    for (o, x) in out
                        .iter_mut()
                        .zip(&live_in[s.index() * w..(s.index() + 1) * w])
                    {
                        *o |= x;
                    }
                }
                for k in 0..w {
                    let n = gen_[b * w + k] | (out[k] & !kill[b * w + k]);
                    if n != live_in[b * w + k] {
                        live_in[b * w + k] = n;
                        changed = true;
                    }
                }
            }
        }
        let e = f.entry_block().index();
        if live_in[e * w..(e + 1) * w].iter().any(|&x| x != 0) {
            return Err("a value is used before any definition".into());
        }
        self.live_in = live_in;
        Ok(())
    }

    fn live_in(&self, b: usize) -> &[u64] {
        &self.live_in[b * self.words..(b + 1) * self.words]
    }

    /// The live-out set of `b`.
    fn live_out(&self, b: usize) -> Vec<u64> {
        let mut out = vec![0u64; self.words];
        for s in self.f.block_succs(Block::new(b)) {
            for (o, x) in out.iter_mut().zip(self.live_in(s.index())) {
                *o |= x;
            }
        }
        out
    }

    fn slot_of(&mut self, v: u32) -> u32 {
        let vi = &mut self.vregs[v as usize];
        if vi.slot == NONE {
            vi.slot = self.num_slots;
            self.num_slots += 1;
        }
        vi.slot
    }

    /// The piece of `v` covering `p`.
    fn piece_at(&self, v: u32, p: Pos) -> Option<u32> {
        let ps = &self.vregs[v as usize].pieces;
        let k = ps.partition_point(|&x| self.pieces[x as usize].start() <= p);
        if k == 0 {
            return None;
        }
        let x = ps[k - 1];
        self.pieces[x as usize].covers(p).then_some(x)
    }

    /// `v`'s location at `p` as an allocation.
    fn alloc_at(&self, v: u32, p: Pos) -> Option<Allocation> {
        let x = self.piece_at(self.rep[v as usize], p)?;
        Some(self.alloc_of(x))
    }

    fn alloc_of(&self, x: u32) -> Allocation {
        let p = &self.pieces[x as usize];
        match p.loc {
            Loc::Reg(r) => Allocation::reg(r),
            Loc::Stack => Allocation::stack(regalloc2::SpillSlot::new(
                self.vregs[p.vreg as usize].slot as usize,
            )),
            Loc::None => panic!("piece {x} of v{} unallocated", p.vreg),
        }
    }
}
