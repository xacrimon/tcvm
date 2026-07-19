//! Block layout order: the single linearization the allocator and the encoder
//! both work in.
//!
//! Two properties are required, both by SSA-form lifetime interval construction
//! (Wimmer & Franz, CGO'10 §4.1):
//!
//!   1. every dominator of a block precedes it, and
//!   2. the blocks of one loop are contiguous.
//!
//! (2) is the load-bearing one. It is what lets interval construction handle a
//! loop with a *single range add* per live value instead of a dataflow fixpoint:
//! everything live at a loop header is live through the entire loop, and with the
//! loop contiguous that whole extent is one span. Plain reverse postorder gives
//! (1) on a reducible CFG but not (2) — nothing stops it from emitting a
//! non-loop block between two loop blocks.
//!
//! The order is built by collapsing each loop to its header, laying the collapsed
//! graph out in reverse postorder, and splicing each loop's own layout in where
//! its header landed. Contiguity then holds by construction rather than by a
//! heuristic that usually gets it right.
//!
//! # Irreducible control flow
//!
//! Interval construction is silently *wrong* on loops with more than one entry —
//! a value live into only one entry is not seen as live in the loop at all
//! (Wimmer10 §4.3 walks through the failure). So this pass detects irreducibility
//! and reports it rather than producing an order that would miscompile. Lua's
//! loops are structured and the frontend should never emit one, which makes this
//! an assertion about the compiler rather than a case to support.

use crate::jit::backend::regalloc::{Block, RegallocFunc};

/// A retreating edge whose target does not dominate its source: the loop it
/// closes has more than one entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Irreducible {
    pub header: Block,
    pub tail: Block,
}

/// The layout, plus the loop structure computed on the way to it.
///
/// The loop information is kept rather than recomputed because everything
/// downstream wants it: spill weighting today, and Braun–Hack's loop-exit edge
/// lengths and per-loop maximum pressure later.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Blocks in layout order. Every reachable block appears exactly once.
    pub order: Vec<Block>,
    /// The header of the innermost loop containing each block, if any. A loop
    /// header is contained in its own loop.
    pub loop_header: Vec<Option<Block>>,
    /// The header of the innermost loop *strictly* containing each loop header —
    /// the loop nesting forest, as a parent pointer. `None` for a top-level loop
    /// and for every block that is not a loop header.
    pub loop_parent: Vec<Option<Block>>,
    /// How many loops contain each block.
    pub depth: Vec<u32>,
}

impl Layout {
    /// Whether `b` is a loop header.
    pub fn is_header(&self, b: Block) -> bool {
        self.loop_header[b.0 as usize] == Some(b)
    }
}

/// Compute the layout order and loop forest for `f`.
///
/// Unreachable blocks are dropped: they have no positions, so nothing downstream
/// can refer to them.
pub fn compute(f: &impl RegallocFunc) -> Result<Layout, Irreducible> {
    let n = f.num_blocks();
    let succs: Vec<Vec<Block>> = (0..n).map(|b| f.succs(Block(b as u32))).collect();

    let rpo = reverse_postorder(f, &succs);
    let mut rpo_num = vec![u32::MAX; n];
    for (k, &b) in rpo.iter().enumerate() {
        rpo_num[b.0 as usize] = k as u32;
    }

    let idom = dominators(f.entry(), &rpo, &rpo_num, &preds(n, &succs));

    // A retreating edge (one whose target was already numbered) closes a loop only
    // if its target dominates it. If it does not, the loop has a second entry and
    // interval construction cannot see it — refuse rather than miscompile.
    let mut back: Vec<(Block, Block)> = Vec::new();
    for &b in &rpo {
        for &s in &succs[b.0 as usize] {
            if rpo_num[s.0 as usize] <= rpo_num[b.0 as usize] {
                if !dominates(s, b, &idom, f.entry()) {
                    return Err(Irreducible { header: s, tail: b });
                }
                back.push((s, b));
            }
        }
    }

    let loops = natural_loops(n, &back, &preds(n, &succs));
    let (loop_header, loop_parent, depth) = nest(n, &loops);

    let mut order = Vec::with_capacity(rpo.len());
    layout_scope(
        None,
        f.entry(),
        &rpo,
        &rpo_num,
        &succs,
        &loop_header,
        &loop_parent,
        &mut order,
    );

    debug_assert_eq!(
        order.len(),
        rpo.len(),
        "layout must emit every reachable block exactly once"
    );
    Ok(Layout {
        order,
        loop_header,
        loop_parent,
        depth,
    })
}

fn preds(n: usize, succs: &[Vec<Block>]) -> Vec<Vec<Block>> {
    let mut p = vec![Vec::new(); n];
    for b in 0..n {
        for &s in &succs[b] {
            p[s.0 as usize].push(Block(b as u32));
        }
    }
    p
}

/// Plain reverse postorder, over reachable blocks only.
fn reverse_postorder(f: &impl RegallocFunc, succs: &[Vec<Block>]) -> Vec<Block> {
    let n = f.num_blocks();
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);

    // Iterative: a deeply nested region would blow a recursive stack.
    let entry = f.entry();
    let mut stack = vec![(entry, 0usize)];
    seen[entry.0 as usize] = true;
    while let Some((b, next)) = stack.pop() {
        let ss = &succs[b.0 as usize];
        if next < ss.len() {
            stack.push((b, next + 1));
            let s = ss[next];
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

/// Immediate dominators, by Cooper, Harvey & Kennedy's iterative algorithm.
///
/// Chosen over Lengauer–Tarjan for the usual reason: on the block counts a trace
/// compiler sees it is faster in practice and a fraction of the code. The entry
/// is its own immediate dominator; unreachable blocks keep `None`.
fn dominators(
    entry: Block,
    rpo: &[Block],
    rpo_num: &[u32],
    preds: &[Vec<Block>],
) -> Vec<Option<Block>> {
    let mut idom: Vec<Option<Block>> = vec![None; rpo_num.len()];
    idom[entry.0 as usize] = Some(entry);

    let intersect = |mut a: Block, mut b: Block, idom: &[Option<Block>]| {
        while a != b {
            while rpo_num[a.0 as usize] > rpo_num[b.0 as usize] {
                a = idom[a.0 as usize].expect("walked above the entry");
            }
            while rpo_num[b.0 as usize] > rpo_num[a.0 as usize] {
                b = idom[b.0 as usize].expect("walked above the entry");
            }
        }
        a
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo {
            if b == entry {
                continue;
            }
            let mut new: Option<Block> = None;
            for &p in &preds[b.0 as usize] {
                if idom[p.0 as usize].is_none() {
                    continue; // not processed yet, or unreachable
                }
                new = Some(match new {
                    None => p,
                    Some(cur) => intersect(p, cur, &idom),
                });
            }
            if new.is_some() && idom[b.0 as usize] != new {
                idom[b.0 as usize] = new;
                changed = true;
            }
        }
    }
    idom
}

fn dominates(a: Block, b: Block, idom: &[Option<Block>], entry: Block) -> bool {
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        if cur == entry {
            return false;
        }
        match idom[cur.0 as usize] {
            Some(d) if d != cur => cur = d,
            _ => return false,
        }
    }
}

/// The natural loop of each header: the header, plus everything that reaches a
/// back-edge tail without passing back through the header.
///
/// Back edges sharing a header describe one loop, so their bodies are unioned.
fn natural_loops(
    n: usize,
    back: &[(Block, Block)],
    preds: &[Vec<Block>],
) -> Vec<(Block, Vec<Block>)> {
    let mut by_header: Vec<Option<Vec<bool>>> = vec![None; n];

    for &(header, tail) in back {
        let body = by_header[header.0 as usize].get_or_insert_with(|| {
            let mut v = vec![false; n];
            v[header.0 as usize] = true;
            v
        });
        // Backward closure from the tail. The header is already marked, so the
        // walk stops there naturally and the body stays inside the loop.
        let mut stack = vec![tail];
        while let Some(b) = stack.pop() {
            if body[b.0 as usize] {
                continue;
            }
            body[b.0 as usize] = true;
            stack.extend_from_slice(&preds[b.0 as usize]);
        }
    }

    by_header
        .into_iter()
        .enumerate()
        .filter_map(|(h, body)| {
            body.map(|body| {
                let blocks = body
                    .iter()
                    .enumerate()
                    .filter(|&(_, &in_loop)| in_loop)
                    .map(|(b, _)| Block(b as u32))
                    .collect();
                (Block(h as u32), blocks)
            })
        })
        .collect()
}

/// Turn the loop bodies into the nesting forest.
///
/// On a reducible CFG two loops are either disjoint or properly nested, so
/// visiting them smallest-body-first assigns each block its innermost loop on the
/// first write.
fn nest(
    n: usize,
    loops: &[(Block, Vec<Block>)],
) -> (Vec<Option<Block>>, Vec<Option<Block>>, Vec<u32>) {
    let mut by_size: Vec<&(Block, Vec<Block>)> = loops.iter().collect();
    by_size.sort_by_key(|(_, body)| body.len());

    let mut loop_header = vec![None; n];
    let mut depth = vec![0u32; n];
    for (header, body) in &by_size {
        for &b in body.iter() {
            if loop_header[b.0 as usize].is_none() {
                loop_header[b.0 as usize] = Some(*header);
            }
            depth[b.0 as usize] += 1;
        }
    }

    // A loop's parent is the innermost *other* loop containing its header, which
    // is what that header's own innermost-loop entry would say if this loop were
    // not in the running — i.e. the next one out.
    let mut loop_parent = vec![None; n];
    for (header, _) in &by_size {
        let mut best: Option<(usize, Block)> = None;
        for (other, body) in &by_size {
            if other == header {
                continue;
            }
            if body.contains(header) {
                let smaller = best.map_or(true, |(len, _)| body.len() < len);
                if smaller {
                    best = Some((body.len(), *other));
                }
            }
        }
        loop_parent[header.0 as usize] = best.map(|(_, b)| b);
    }

    (loop_header, loop_parent, depth)
}

/// The unit `b` collapses to when laying out `scope`: the header of the outermost
/// loop that is strictly inside `scope` and contains `b`, or `b` itself.
fn unit_in(
    b: Block,
    scope: Option<Block>,
    loop_header: &[Option<Block>],
    loop_parent: &[Option<Block>],
) -> Block {
    let mut cur = loop_header[b.0 as usize];
    let mut unit = b;
    while let Some(h) = cur {
        if Some(h) == scope {
            break;
        }
        unit = h;
        cur = loop_parent[h.0 as usize];
    }
    unit
}

/// Lay out one scope — a loop body, or the whole function when `scope` is `None`
/// — appending to `out`.
///
/// Recursion depth is loop nesting depth, which is small by construction; a trace
/// with enough nested loops to overflow a stack would have failed to compile long
/// before reaching here.
#[allow(clippy::too_many_arguments)]
fn layout_scope(
    scope: Option<Block>,
    scope_entry: Block,
    rpo: &[Block],
    rpo_num: &[u32],
    succs: &[Vec<Block>],
    loop_header: &[Option<Block>],
    loop_parent: &[Option<Block>],
    out: &mut Vec<Block>,
) {
    // Which blocks this scope owns, and what each collapses to.
    let in_scope = |b: Block| match scope {
        None => rpo_num[b.0 as usize] != u32::MAX,
        Some(s) => {
            // Inside loop `s` iff `s` is on b's chain of enclosing loops.
            let mut cur = loop_header[b.0 as usize];
            while let Some(h) = cur {
                if h == s {
                    return true;
                }
                cur = loop_parent[h.0 as usize];
            }
            false
        }
    };

    // Successor edges between units, skipping anything leaving the scope and the
    // back edges into the scope's own header (which is where the loop closes).
    let mut units: Vec<Block> = Vec::new();
    let mut unit_succs: Vec<Vec<Block>> = Vec::new();
    let mut unit_idx = vec![usize::MAX; loop_header.len()];
    for &b in rpo {
        if !in_scope(b) {
            continue;
        }
        let u = unit_in(b, scope, loop_header, loop_parent);
        if unit_idx[u.0 as usize] == usize::MAX {
            unit_idx[u.0 as usize] = units.len();
            units.push(u);
            unit_succs.push(Vec::new());
        }
    }
    for &b in rpo {
        if !in_scope(b) {
            continue;
        }
        let u = unit_in(b, scope, loop_header, loop_parent);
        for &s in &succs[b.0 as usize] {
            if !in_scope(s) {
                continue;
            }
            let v = unit_in(s, scope, loop_header, loop_parent);
            if v == u || v == scope_entry {
                continue;
            }
            let ui = unit_idx[u.0 as usize];
            if !unit_succs[ui].contains(&v) {
                unit_succs[ui].push(v);
            }
        }
    }

    // Reverse postorder over the collapsed graph. Successors are visited in `rpo`
    // order so the result tracks the original layout where the loop structure does
    // not force otherwise.
    for ss in unit_succs.iter_mut() {
        ss.sort_by_key(|b| rpo_num[b.0 as usize]);
    }
    let mut seen = vec![false; units.len()];
    let mut post: Vec<Block> = Vec::with_capacity(units.len());
    let root = unit_in(scope_entry, scope, loop_header, loop_parent);
    let mut stack = vec![(root, 0usize)];
    seen[unit_idx[root.0 as usize]] = true;
    while let Some((u, next)) = stack.pop() {
        let ss = &unit_succs[unit_idx[u.0 as usize]];
        if next < ss.len() {
            stack.push((u, next + 1));
            let s = ss[next];
            let si = unit_idx[s.0 as usize];
            if !seen[si] {
                seen[si] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(u);
        }
    }
    post.reverse();

    // Splice: a unit that is a nested loop's header expands to that whole loop.
    for u in post {
        let is_nested_header = loop_header[u.0 as usize] == Some(u) && Some(u) != scope;
        if is_nested_header {
            layout_scope(
                Some(u),
                u,
                rpo,
                rpo_num,
                succs,
                loop_header,
                loop_parent,
                out,
            );
        } else {
            out.push(u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::regalloc::{Operand, RegClass};

    /// A CFG and nothing else — this pass reads no instructions.
    #[derive(Default)]
    struct Cfg {
        succs: Vec<Vec<Block>>,
    }

    impl Cfg {
        fn blocks(n: usize) -> Self {
            Cfg {
                succs: vec![Vec::new(); n],
            }
        }
        fn edge(mut self, from: usize, to: &[usize]) -> Self {
            self.succs[from] = to.iter().map(|&t| Block(t as u32)).collect();
            self
        }
    }

    impl RegallocFunc for Cfg {
        fn num_blocks(&self) -> usize {
            self.succs.len()
        }
        fn entry(&self) -> Block {
            Block(0)
        }
        fn block_insts(&self, _b: Block) -> &[usize] {
            &[]
        }
        fn succs(&self, b: Block) -> Vec<Block> {
            self.succs[b.0 as usize].clone()
        }
        fn num_insts(&self) -> usize {
            0
        }
        fn defs(&self, _i: usize) -> &[Operand] {
            &[]
        }
        fn uses(&self, _i: usize) -> &[Operand] {
            &[]
        }
        fn num_vregs(&self) -> usize {
            0
        }
        fn class(&self, _v: crate::jit::backend::regalloc::VReg) -> RegClass {
            RegClass::Int
        }
    }

    fn order_of(l: &Layout) -> Vec<u32> {
        l.order.iter().map(|b| b.0).collect()
    }

    /// Every loop's blocks occupy one unbroken run of the order.
    fn assert_loops_contiguous(l: &Layout) {
        let pos: Vec<usize> = {
            let mut p = vec![usize::MAX; l.loop_header.len()];
            for (k, &b) in l.order.iter().enumerate() {
                p[b.0 as usize] = k;
            }
            p
        };
        for h in 0..l.loop_header.len() {
            let header = Block(h as u32);
            if !l.is_header(header) {
                continue;
            }
            let members: Vec<usize> = l
                .order
                .iter()
                .enumerate()
                .filter(|&(_, &b)| {
                    let mut cur = l.loop_header[b.0 as usize];
                    while let Some(x) = cur {
                        if x == header {
                            return true;
                        }
                        cur = l.loop_parent[x.0 as usize];
                    }
                    false
                })
                .map(|(k, _)| k)
                .collect();
            let lo = *members.iter().min().unwrap();
            let hi = *members.iter().max().unwrap();
            assert_eq!(
                hi - lo + 1,
                members.len(),
                "loop at mb{h} is not contiguous: positions {members:?} in {:?}",
                order_of(l)
            );
            assert_eq!(pos[h], lo, "loop at mb{h} must start at its header");
        }
    }

    #[test]
    fn straight_line_is_rpo() {
        let cfg = Cfg::blocks(3).edge(0, &[1]).edge(1, &[2]);
        let l = compute(&cfg).unwrap();
        assert_eq!(order_of(&l), vec![0, 1, 2]);
        assert!(l.loop_header.iter().all(|h| h.is_none()));
    }

    #[test]
    fn diamond_puts_dominator_first() {
        let cfg = Cfg::blocks(4).edge(0, &[1, 2]).edge(1, &[3]).edge(2, &[3]);
        let l = compute(&cfg).unwrap();
        let o = order_of(&l);
        assert_eq!(o[0], 0);
        assert_eq!(o[3], 3);
    }

    #[test]
    fn simple_loop_is_contiguous() {
        // 0 -> 1 -> 2 -> 1, 1 -> 3
        let cfg = Cfg::blocks(4).edge(0, &[1]).edge(1, &[2, 3]).edge(2, &[1]);
        let l = compute(&cfg).unwrap();
        assert_eq!(l.loop_header[1], Some(Block(1)));
        assert_eq!(l.loop_header[2], Some(Block(1)));
        assert_eq!(l.loop_header[3], None);
        assert_eq!(order_of(&l), vec![0, 1, 2, 3]);
        assert_loops_contiguous(&l);
    }

    /// The case plain RPO gets wrong: a block outside the loop is reachable from
    /// the header and would otherwise be emitted between the loop's blocks.
    #[test]
    fn exit_block_does_not_split_a_loop() {
        // 0 -> 1(header) -> {2, 4}; 2 -> 3 -> 1 (back edge); 4 is the exit.
        let cfg = Cfg::blocks(5)
            .edge(0, &[1])
            .edge(1, &[2, 4])
            .edge(2, &[3])
            .edge(3, &[1]);
        // Plain RPO reaches the exit from the header before finishing the body,
        // so it interleaves: mb4 lands between the header and the loop's blocks.
        let plain: Vec<u32> = cfg.block_order().iter().map(|b| b.0).collect();
        assert_eq!(plain, vec![0, 1, 4, 2, 3], "the case this pass exists for");

        let l = compute(&cfg).unwrap();
        assert_loops_contiguous(&l);
        let o = order_of(&l);
        assert_eq!(o, vec![0, 1, 2, 3, 4], "loop body must precede the exit");
    }

    #[test]
    fn nested_loops_are_each_contiguous() {
        // outer header 1, inner header 2, inner tail 3, outer tail 4, exit 5.
        let cfg = Cfg::blocks(6)
            .edge(0, &[1])
            .edge(1, &[2])
            .edge(2, &[3, 4])
            .edge(3, &[2])
            .edge(4, &[1, 5]);
        let l = compute(&cfg).unwrap();
        assert_eq!(l.loop_header[2], Some(Block(2)), "inner header");
        assert_eq!(l.loop_header[3], Some(Block(2)), "inner body");
        assert_eq!(l.loop_parent[2], Some(Block(1)), "inner nests in outer");
        assert_eq!(l.depth[3], 2);
        assert_loops_contiguous(&l);
        assert_eq!(order_of(&l), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn two_back_edges_to_one_header_are_one_loop() {
        // header 1 with tails 2 and 3.
        let cfg = Cfg::blocks(5)
            .edge(0, &[1])
            .edge(1, &[2, 3])
            .edge(2, &[1])
            .edge(3, &[1, 4]);
        let l = compute(&cfg).unwrap();
        assert_eq!(l.loop_header[2], Some(Block(1)));
        assert_eq!(l.loop_header[3], Some(Block(1)));
        assert_eq!(l.depth[2], 1, "one loop, not two");
        assert_loops_contiguous(&l);
    }

    #[test]
    fn irreducible_loop_is_rejected() {
        // 0 branches into the middle of a 1 <-> 2 cycle: two entries, no header
        // dominating both.
        let cfg = Cfg::blocks(3).edge(0, &[1, 2]).edge(1, &[2]).edge(2, &[1]);
        assert!(matches!(compute(&cfg), Err(Irreducible { .. })));
    }

    #[test]
    fn unreachable_blocks_are_dropped() {
        let cfg = Cfg::blocks(3).edge(0, &[1]);
        let l = compute(&cfg).unwrap();
        assert_eq!(order_of(&l), vec![0, 1]);
    }
}
