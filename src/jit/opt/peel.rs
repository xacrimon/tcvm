//! Loop peeling (9.3): each small innermost loop gets a copy of its body in
//! front of it, so the loop's header joins only steady-state values and the
//! copy's guards dominate the loop for GVN.

use crate::jit::FastMap as HashMap;
use crate::jit::ir::{Block, BlockCall, CfgInfo, Func, NO_SNAP, Snap, Val, ValDef};
use crate::jit::opt::natural_loops;

/// Loops larger than this many instructions are not peeled.
const PEEL_LIMIT: usize = 48;

pub(crate) fn peel(f: &mut Func<'_>) -> bool {
    let mut bodies: Vec<(Block, Vec<Block>)> = Vec::new();
    for (h, body) in natural_loops(f) {
        match bodies.iter_mut().find(|(x, _)| *x == h) {
            Some((_, b)) => {
                for x in body {
                    if !b.contains(&x) {
                        b.push(x);
                    }
                }
            }
            None => bodies.push((h, body)),
        }
    }
    let headers: Vec<Block> = bodies.iter().map(|(h, _)| *h).collect();
    let mut changed = false;
    for (h, body) in &bodies {
        let innermost = !headers.iter().any(|x| x != h && body.contains(x));
        let size: usize = body.iter().map(|b| f.blocks[b.idx()].insts.len()).sum();
        // A resume block continues one call: a call resuming outside the
        // loop would get a second one from the copy.
        let resumes_outside = body.iter().any(|&b| {
            f.terminator(b).is_some_and(|t| {
                matches!(f.op(t), crate::jit::ir::ops::Op::Call { .. })
                    && !body.contains(&f.edges(t)[0].target)
            })
        });
        if innermost && size <= PEEL_LIMIT && !resumes_outside && !f.blocks[h.idx()].peeled {
            peel_loop(f, *h, body);
            changed = true;
        }
    }
    changed
}

fn peel_loop(f: &mut Func<'_>, h: Block, body: &[Block]) {
    let order: Vec<Block> = f
        .cfg()
        .rpo
        .iter()
        .copied()
        .filter(|b| body.contains(b))
        .collect();
    let mut bmap: HashMap<Block, Block> = HashMap::default();
    for &b in &order {
        let nb = f.new_block();
        f.blocks[nb.idx()].resume = f.blocks[b.idx()].resume;
        bmap.insert(b, nb);
    }
    let mut vmap: HashMap<Val, Val> = HashMap::default();
    for &b in &order {
        for k in 0..f.blocks[b.idx()].params.len() {
            let p = f.params(b)[k];
            let np = f.add_param(bmap[&b], f.ty(p));
            vmap.insert(p, np);
        }
    }
    let map = |vmap: &HashMap<Val, Val>, v: Val| vmap.get(&v).copied().unwrap_or(v);
    let mut args: Vec<Val> = Vec::new();
    let mut edges: Vec<BlockCall> = Vec::new();
    for &b in &order {
        for k in 0..f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            let d = f.insts[i.idx()];
            args.clear();
            args.extend(f.args(i).iter().map(|&a| map(&vmap, a)));
            let snap = (d.snap != NO_SNAP).then(|| f.map_snap(Snap(d.snap), |v| map(&vmap, v)));
            let ni = f.make_inst(d.op, &args, snap, d.tag);
            for (r, nr) in f.results(i).zip(f.results(ni)) {
                f.vals[nr.idx()].ty = f.vals[r.idx()].ty;
                vmap.insert(r, nr);
            }
            // The copy's back edges enter the loop; its other edges stay in
            // the copy or leave as the loop's do.
            edges.clear();
            for e in 0..f.edges(i).len() {
                let t = f.edges(i)[e].target;
                args.clear();
                args.extend(f.edge_args(i, e).iter().map(|&a| map(&vmap, a)));
                let a = f.new_vlist(&args);
                edges.push(BlockCall {
                    target: if t == h {
                        h
                    } else {
                        bmap.get(&t).copied().unwrap_or(t)
                    },
                    args: a,
                });
            }
            if !edges.is_empty() {
                f.set_edges(ni, &edges);
            }
            f.append(bmap[&b], ni);
        }
    }
    // Entries into the loop now enter the copy.
    let nh = bmap[&h];
    for bd in 0..f.blocks.len() {
        let b = Block(bd as u32);
        if f.blocks[bd].dead || body.contains(&b) || bmap.values().any(|&x| x == b) {
            continue;
        }
        if let Some(t) = f.terminator(b) {
            for k in 0..f.edges(t).len() {
                if f.edges(t)[k].target == h {
                    f.set_target(t, k, nh);
                }
            }
        }
    }
    repair_ssa(f, body, &bmap, &vmap);
}

/// Values of the loop used after it now have two definitions, the loop's and
/// the copy's: give their uses outside both the reaching one, with block
/// parameters where the two meet.
fn repair_ssa(
    f: &mut Func<'_>,
    body: &[Block],
    bmap: &HashMap<Block, Block>,
    vmap: &HashMap<Val, Val>,
) {
    let in_body = |b: Block| body.contains(&b);
    let in_copy = |b: Block| bmap.values().any(|&x| x == b);
    let defined = |f: &Func<'_>, v: Val| match f.vals[v.idx()].def {
        ValDef::Param(b, _) => in_body(b),
        ValDef::Inst(i, _) => in_body(f.insts[i.idx()].block),
    };
    let outside: Vec<Block> = (0..f.blocks.len() as u32)
        .map(Block)
        .filter(|&b| !f.blocks[b.idx()].dead && !in_body(b) && !in_copy(b))
        .collect();
    let mut escaping: Vec<Val> = Vec::new();
    for &b in &outside {
        for &i in f.insts_of(b) {
            f.for_each_use(i, |v| {
                if defined(f, v) && !escaping.contains(&v) {
                    escaping.push(v);
                }
            });
        }
    }
    // The copy's edges into the loop and out of it are new.
    let cfg = f.cfg();
    for v in escaping {
        let copy = vmap[&v];
        let mut memo: HashMap<Block, Val> = HashMap::default();
        let mut reach = Reach {
            v,
            copy,
            body,
            bmap,
            cfg: &cfg,
            memo: &mut memo,
        };
        for &b in &outside {
            let mut uses = false;
            for &i in f.insts_of(b) {
                f.for_each_use(i, |x| uses |= x == v);
            }
            if !uses {
                continue;
            }
            let r = reach.at(f, b);
            for k in 0..f.blocks[b.idx()].insts.len() {
                let i = f.insts_of(b)[k];
                for a in f.args_mut(i) {
                    if *a == v {
                        *a = r;
                    }
                }
                for e in 0..f.edges(i).len() {
                    for a in f.edge_args_mut(i, e) {
                        if *a == v {
                            *a = r;
                        }
                    }
                }
                let s = f.insts[i.idx()].snap;
                // A shared snapshot is copied before it is changed.
                if s != NO_SNAP && f.entries(s).iter().any(|e| e.1 == v) {
                    let ns = f.map_snap(Snap(s), |x| if x == v { r } else { x });
                    f.insts[i.idx()].snap = ns.0;
                }
            }
        }
    }
}

/// The reaching definition of one escaping value (Braun et al.'s lookup,
/// with every block sealed).
struct Reach<'a> {
    v: Val,
    copy: Val,
    body: &'a [Block],
    bmap: &'a HashMap<Block, Block>,
    cfg: &'a CfgInfo,
    memo: &'a mut HashMap<Block, Val>,
}

impl Reach<'_> {
    /// The value at the start of `b`, or at its end for a block of the loop
    /// or the copy.
    fn at(&mut self, f: &mut Func<'_>, b: Block) -> Val {
        if self.body.contains(&b) {
            return self.v;
        }
        if self.bmap.values().any(|&x| x == b) {
            return self.copy;
        }
        if let Some(&r) = self.memo.get(&b) {
            return r;
        }
        let inc = self.cfg.incoming(b);
        if inc.is_empty() {
            return self.v;
        }
        if inc.len() == 1 {
            let p = f.insts[inc[0].0.idx()].block;
            let r = self.at(f, p);
            self.memo.insert(b, r);
            return r;
        }
        let p = f.add_param(b, f.ty(self.v));
        self.memo.insert(b, p);
        for &(t, k) in inc {
            let pred = f.insts[t.idx()].block;
            let a = self.at(f, pred);
            f.push_edge_arg(t, k as usize, a);
        }
        p
    }
}
