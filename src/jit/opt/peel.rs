//! Loop peeling (9.3): each small innermost loop gets a copy of its body in
//! front of it, so the loop's header joins only steady-state values and the
//! copy's guards dominate the loop for GVN.

use std::collections::HashMap;

use crate::jit::build::incoming;
use crate::jit::ir::{Block, BlockCall, Func, Inst, NO_SNAP, SnapData, Val, ValDef};
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
    let rpo = f.rpo();
    let order: Vec<Block> = rpo.iter().copied().filter(|b| body.contains(b)).collect();
    let mut bmap: HashMap<Block, Block> = HashMap::new();
    for &b in &order {
        let nb = f.new_block();
        f.blocks[nb.idx()].resume = f.blocks[b.idx()].resume;
        bmap.insert(b, nb);
    }
    let mut vmap: HashMap<Val, Val> = HashMap::new();
    for &b in &order {
        for p in f.blocks[b.idx()].params.clone() {
            let np = f.add_param(bmap[&b], f.ty(p));
            vmap.insert(p, np);
        }
    }
    let map = |vmap: &HashMap<Val, Val>, v: Val| vmap.get(&v).copied().unwrap_or(v);
    for &b in &order {
        for i in f.blocks[b.idx()].insts.clone() {
            let d = f.insts[i.idx()];
            let args: Vec<Val> = f.args(i).iter().map(|&a| map(&vmap, a)).collect();
            let snap = (d.snap != NO_SNAP).then(|| {
                let s = &f.snaps[d.snap as usize];
                let ns = SnapData {
                    pc: s.pc,
                    kind: s.kind,
                    entries: s.entries.iter().map(|&(r, v)| (r, map(&vmap, v))).collect(),
                };
                f.add_snap(ns)
            });
            let ni = f.make_inst(d.op, &args, snap, d.tag);
            for (r, nr) in f.results(i).zip(f.results(ni)) {
                f.vals[nr.idx()].ty = f.vals[r.idx()].ty;
                vmap.insert(r, nr);
            }
            // The copy's back edges enter the loop; its other edges stay in
            // the copy or leave as the loop's do.
            let edges: Vec<BlockCall> = f
                .edges(i)
                .iter()
                .map(|e| BlockCall {
                    target: if e.target == h {
                        h
                    } else {
                        bmap.get(&e.target).copied().unwrap_or(e.target)
                    },
                    args: e.args.iter().map(|&a| map(&vmap, a)).collect(),
                })
                .collect();
            if !edges.is_empty() {
                f.set_edges(ni, edges);
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
            for e in f.edges_mut(t) {
                if e.target == h {
                    e.target = nh;
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
        for &i in &f.blocks[b.idx()].insts {
            let s = f.insts[i.idx()].snap;
            let snap_vals = if s != NO_SNAP {
                {
                    f.snaps[s as usize]
                        .entries
                        .iter()
                        .map(|e| e.1)
                        .collect::<Vec<_>>()
                }
            } else {
                Default::default()
            };
            for v in f
                .args(i)
                .iter()
                .copied()
                .chain(f.edges(i).iter().flat_map(|e| e.args.iter().copied()))
                .chain(snap_vals)
            {
                if defined(f, v) && !escaping.contains(&v) {
                    escaping.push(v);
                }
            }
        }
    }
    let inc = incoming(f);
    for v in escaping {
        let copy = vmap[&v];
        let mut memo: HashMap<Block, Val> = HashMap::new();
        let mut reach = Reach {
            v,
            copy,
            body,
            bmap,
            inc: &inc,
            memo: &mut memo,
        };
        for &b in &outside {
            let insts = f.blocks[b.idx()].insts.clone();
            let uses = insts.iter().any(|&i| {
                let s = f.insts[i.idx()].snap;
                f.args(i).contains(&v)
                    || f.edges(i).iter().any(|e| e.args.contains(&v))
                    || (s != NO_SNAP && f.snaps[s as usize].entries.iter().any(|e| e.1 == v))
            });
            if !uses {
                continue;
            }
            let r = reach.at(f, b);
            for &i in &insts {
                for a in f.args_mut(i) {
                    if *a == v {
                        *a = r;
                    }
                }
                for e in f.edges_mut(i) {
                    for a in &mut e.args {
                        if *a == v {
                            *a = r;
                        }
                    }
                }
                let s = f.insts[i.idx()].snap;
                if s != NO_SNAP {
                    // A shared snapshot is copied before it is changed.
                    if f.snaps[s as usize].entries.iter().any(|e| e.1 == v) {
                        let mut ns = f.snaps[s as usize].clone();
                        for e in &mut ns.entries {
                            if e.1 == v {
                                e.1 = r;
                            }
                        }
                        f.snaps.push(ns);
                        f.insts[i.idx()].snap = f.snaps.len() as u32 - 1;
                    }
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
    inc: &'a [Vec<(Inst, usize)>],
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
        let inc = &self.inc[b.idx()];
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
            f.edges_mut(t)[k].args.push(a);
        }
        p
    }
}
