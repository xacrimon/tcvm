//! Block layout. `order` puts every dominator of a block before it and keeps
//! each loop's blocks contiguous, which the graph's own order does not
//! promise (nothing stops a block outside a loop from standing between two
//! of the loop's): the blocks are taken in the graph's order, each into its
//! innermost loop's list, a loop into its parent's where its header stands,
//! and the lists are flattened. Machine blocks are numbered by `order`.
//!
//! `emit` keeps `order` for the hot blocks and moves the blocks that end in a
//! deopt (run at most once per entry), then the cold ones, after them, so
//! loop bodies fall through. The allocator and the encoder both take `emit`:
//! live ranges are intervals of it, and each loop's hot blocks are contiguous
//! in it. The encoder also rotates hot loops (`emit::rotate_loops`).

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, BlockFlags, Graph, Loop};

#[derive(Default)]
pub(crate) struct Layout {
    pub(crate) order: Vec<Block>,
    /// A block's index in `order`, `u32::MAX` for one the graph never opened.
    pub(crate) pos: Vec<u32>,
    /// The allocator's and the encoder's order, as indices into `order`.
    pub(crate) emit: Vec<u32>,
    /// How many of `emit` come before the first block ending in a deopt or
    /// cold.
    pub(crate) hot: usize,
    /// Each loop as the first and last index of its blocks in `order`, its
    /// header first.
    pub(crate) loops: Vec<(u32, u32)>,
    /// Per block of `order`: the header of the innermost loop holding it
    /// (itself for a header), and for a header the enclosing loop's; as
    /// indices into `order`, `u32::MAX` for none.
    pub(crate) loop_of: Vec<u32>,
    pub(crate) loop_parent: Vec<u32>,
    /// Each list's items, linked: a block, or a loop (`LOOP` set).
    next: Vec<u32>,
    items: Vec<u32>,
    /// Per loop, then the top level: a list's first and last link.
    lists: Vec<(u32, u32)>,
    stack: Vec<u32>,
}

const NONE: u32 = u32::MAX;
const LOOP: u32 = 1 << 31;

impl Layout {
    pub(crate) fn of(&mut self, g: &Graph) {
        let (nl, n) = (g.loops.len(), g.order.len());
        self.items.clear();
        self.next.clear();
        self.lists.clear();
        self.lists.resize(nl + 1, (NONE, NONE));
        let push = |s: &mut Self, list: usize, item: u32| {
            s.items.push(item);
            s.next.push(NONE);
            let at = s.items.len() as u32 - 1;
            match s.lists[list] {
                (NONE, _) => s.lists[list] = (at, at),
                (first, last) => {
                    s.next[last as usize] = at;
                    s.lists[list] = (first, at);
                }
            }
        };
        let list_of = |l: Loop| if l.is_none() { nl } else { l.idx() };
        for &b in &g.order {
            let d = g.block(b);
            if d.flags.contains(BlockFlags::LOOP_HEADER) {
                push(self, list_of(g.loops[d.lp.idx()].parent), LOOP | d.lp.0);
            }
            push(self, list_of(d.lp), b.0);
        }
        self.order.clear();
        self.pos.clear();
        self.pos.resize(g.blocks.len(), NONE);
        self.loops.clear();
        self.loops.resize(nl, (NONE, NONE));
        // Depth first through the lists: a loop's own before what follows it.
        self.stack.clear();
        self.stack.push(self.lists[nl].0);
        while let Some(at) = self.stack.pop() {
            if at == NONE {
                continue;
            }
            self.stack.push(self.next[at as usize]);
            let item = self.items[at as usize];
            if item & LOOP != 0 {
                self.stack.push(self.lists[(item & !LOOP) as usize].0);
            } else {
                self.pos[item as usize] = self.order.len() as u32;
                self.order.push(Block(item));
            }
        }
        self.loop_of.clear();
        self.loop_of.resize(n, NONE);
        self.loop_parent.clear();
        self.loop_parent.resize(n, NONE);
        let header = |s: &Self, l: Loop| {
            if l.is_none() {
                NONE
            } else {
                s.pos[g.loops[l.idx()].header.idx()]
            }
        };
        for k in 0..n {
            let d = g.block(self.order[k]);
            self.loop_of[k] = header(self, d.lp);
            let mut l = d.lp;
            while !l.is_none() {
                let r = &mut self.loops[l.idx()];
                *r = (r.0.min(k as u32), k as u32);
                l = g.loops[l.idx()].parent;
            }
            if d.flags.contains(BlockFlags::LOOP_HEADER) {
                self.loop_parent[k] = header(self, g.loops[d.lp.idx()].parent);
            }
        }
        self.emit.clear();
        let kind = |b: Block| {
            if g.block(b).flags.contains(BlockFlags::COLD) {
                2
            } else {
                (g.term(b).op == Op::Deopt) as u32
            }
        };
        for pass in 0..3 {
            if pass == 1 {
                self.hot = self.emit.len();
            }
            (self.emit).extend((0..n as u32).filter(|&k| kind(self.order[k as usize]) == pass));
        }
    }
}
