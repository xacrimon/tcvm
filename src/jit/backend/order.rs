//! Block layout (9.9): the order the allocator and the encoder work in.
//! Every dominator of a block precedes it and each loop's blocks are
//! contiguous, which plain reverse postorder does not give (nothing stops it
//! from placing a block outside a loop between two of the loop's blocks).
//! Each loop collapses to its header, the collapsed graph is laid out in
//! reverse postorder, and each loop's own layout is spliced in where its
//! header landed. Blocks that end in a deopt run at most once per entry and
//! go last, so loop bodies fall through. Irreducible graphs never get here
//! (`cfg.rs` refuses them).

use crate::jit::ir::ops::Op;
use crate::jit::ir::{Block, CfgInfo, Func};

pub(crate) fn layout(f: &Func<'_>) -> Vec<Block> {
    let cfg = f.cfg();
    let n = f.blocks.len();
    let mut rpo_num = vec![u32::MAX; n];
    for (k, &b) in cfg.rpo.iter().enumerate() {
        rpo_num[b.idx()] = k as u32;
    }
    // Natural loops, one per header, from the back edges into it.
    let mut bodies: Vec<(Block, Vec<bool>)> = Vec::new();
    for &b in &cfg.rpo {
        for s in f.succs(b) {
            if rpo_num[s.idx()] > rpo_num[b.idx()] || !cfg.dominates(s, b) {
                continue;
            }
            let k = match bodies.iter().position(|(h, _)| *h == s) {
                Some(k) => k,
                None => {
                    let mut body = vec![false; n];
                    body[s.idx()] = true;
                    bodies.push((s, body));
                    bodies.len() - 1
                }
            };
            let body = &mut bodies[k].1;
            let mut work = vec![b];
            while let Some(x) = work.pop() {
                if !body[x.idx()] {
                    body[x.idx()] = true;
                    work.extend_from_slice(cfg.preds(x));
                }
            }
        }
    }
    // The nesting forest: smallest bodies first give each block its
    // innermost loop on the first write.
    bodies.sort_by_key(|(_, body)| body.iter().filter(|&&x| x).count());
    let mut loop_header: Vec<Option<Block>> = vec![None; n];
    for (h, body) in &bodies {
        for (x, &inside) in body.iter().enumerate() {
            if inside && loop_header[x].is_none() {
                loop_header[x] = Some(*h);
            }
        }
    }
    let mut loop_parent: Vec<Option<Block>> = vec![None; n];
    for (i, (h, _)) in bodies.iter().enumerate() {
        loop_parent[h.idx()] = bodies[i + 1..]
            .iter()
            .find(|(_, body)| body[h.idx()])
            .map(|(p, _)| *p);
    }
    let lp = Loops {
        cfg: &cfg,
        rpo_num: &rpo_num,
        loop_header: &loop_header,
        loop_parent: &loop_parent,
    };
    let mut order = Vec::with_capacity(cfg.rpo.len());
    lp.scope(f, None, f.entry, &mut order);
    debug_assert_eq!(order.len(), cfg.rpo.len());
    let deopts = |b: &Block| f.terminator(*b).is_some_and(|t| f.op(t) == Op::Deopt);
    let (cold, mut hot): (Vec<Block>, Vec<Block>) = order.into_iter().partition(deopts);
    hot.extend(cold);
    hot
}

struct Loops<'a> {
    cfg: &'a CfgInfo,
    rpo_num: &'a [u32],
    loop_header: &'a [Option<Block>],
    loop_parent: &'a [Option<Block>],
}

impl Loops<'_> {
    /// Whether `b` lies in loop `s` (or anywhere, for `None`).
    fn inside(&self, b: Block, s: Option<Block>) -> bool {
        let Some(s) = s else {
            return self.rpo_num[b.idx()] != u32::MAX;
        };
        let mut cur = self.loop_header[b.idx()];
        while let Some(h) = cur {
            if h == s {
                return true;
            }
            cur = self.loop_parent[h.idx()];
        }
        false
    }

    /// What `b` collapses to in `scope`: the header of the outermost loop
    /// strictly inside `scope` holding it, or `b`.
    fn unit(&self, b: Block, scope: Option<Block>) -> Block {
        let mut cur = self.loop_header[b.idx()];
        let mut unit = b;
        while let Some(h) = cur {
            if Some(h) == scope {
                break;
            }
            unit = h;
            cur = self.loop_parent[h.idx()];
        }
        unit
    }

    /// Lay out a loop's body (or the whole region for `None`) entered at
    /// `entry`, appending to `out`; recursion depth is loop depth.
    fn scope(&self, f: &Func<'_>, scope: Option<Block>, entry: Block, out: &mut Vec<Block>) {
        let mut units: Vec<Block> = Vec::new();
        let mut succs: Vec<Vec<Block>> = Vec::new();
        let mut index = vec![usize::MAX; self.loop_header.len()];
        for &b in &self.cfg.rpo {
            if !self.inside(b, scope) {
                continue;
            }
            let u = self.unit(b, scope);
            if index[u.idx()] == usize::MAX {
                index[u.idx()] = units.len();
                units.push(u);
                succs.push(Vec::new());
            }
        }
        // Edges between units, leaving out those leaving the scope and the
        // back edges into its header.
        for &b in &self.cfg.rpo {
            if !self.inside(b, scope) {
                continue;
            }
            let u = self.unit(b, scope);
            for s in f.succs(b) {
                if !self.inside(s, scope) {
                    continue;
                }
                let v = self.unit(s, scope);
                let ss = &mut succs[index[u.idx()]];
                if v != u && v != entry && !ss.contains(&v) {
                    ss.push(v);
                }
            }
        }
        // Successors in RPO order, so the layout keeps RPO where the loops
        // allow.
        for ss in &mut succs {
            ss.sort_by_key(|b| self.rpo_num[b.idx()]);
        }
        let root = self.unit(entry, scope);
        let mut seen = vec![false; units.len()];
        let mut post = Vec::with_capacity(units.len());
        let mut stack = vec![(root, 0usize)];
        seen[index[root.idx()]] = true;
        while let Some((u, next)) = stack.pop() {
            let ss = &succs[index[u.idx()]];
            if next < ss.len() {
                stack.push((u, next + 1));
                let s = ss[next];
                if !std::mem::replace(&mut seen[index[s.idx()]], true) {
                    stack.push((s, 0));
                }
            } else {
                post.push(u);
            }
        }
        for &u in post.iter().rev() {
            if self.loop_header[u.idx()] == Some(u) && Some(u) != scope {
                self.scope(f, Some(u), u, out);
            } else {
                out.push(u);
            }
        }
    }
}
