//! The call-boundary pass (9.8): a call clobbers every register and pops the
//! native frame, so a value live into a resume block (R6 allows only
//! constants) is re-emitted after the resume, with its later uses given
//! whichever definition reaches them.

use crate::jit::FastMap;
use crate::jit::ir::verify::live_in;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Snap, Val};

pub(crate) fn call_boundaries(f: &mut Func<'_>) -> Result<(), String> {
    f.boundaries = true;
    let cfg = f.cfg();
    if !cfg.rpo.iter().any(|b| f.blocks[b.idx()].resume) {
        return Ok(());
    }
    let live = live_in(f, &cfg.rpo);
    // Each value live across a call, with the resume blocks it is live into.
    let mut across: Vec<(Val, Vec<Block>)> = Vec::new();
    for &b in &cfg.rpo {
        if !f.blocks[b.idx()].resume {
            continue;
        }
        for v in live[b.idx()].iter() {
            let v = Val(v as u32);
            if !f.def_op(v).is_some_and(|op| op.is_const()) {
                return Err(format!(
                    "v{} is live across the call resuming at b{}",
                    v.0, b.0
                ));
            }
            match across.iter_mut().find(|(x, _)| *x == v) {
                Some((_, rs)) => rs.push(b),
                None => across.push((v, vec![b])),
            }
        }
    }
    for (v, resumes) in across {
        let d = f.def_inst(v).unwrap();
        // The copy goes after the resume block's `Resume`.
        let mut copies: FastMap<Block, Val> = FastMap::default();
        for &r in &resumes {
            let c = f.make_inst(f.op(d), &[], None, f.insts[d.idx()].tag);
            let cv = f.result(c);
            f.vals[cv.idx()].ty = f.ty(v);
            f.insert(r, 1, c);
            copies.insert(r, cv);
        }
        let mut s = Repair {
            v,
            def_block: f.insts[d.idx()].block,
            copies: &copies,
            cfg: &cfg,
            memo: FastMap::default(),
        };
        for &b in &cfg.rpo {
            let n = f.blocks[b.idx()].insts.len();
            let mut before: Option<Val> = None;
            for k in 0..n {
                let i = f.insts_of(b)[k];
                if i == d {
                    before = Some(v);
                    continue;
                }
                if copies.get(&b).is_some_and(|&c| f.def_inst(c) == Some(i)) {
                    before = Some(copies[&b]);
                    continue;
                }
                if !uses(f, i, v) {
                    continue;
                }
                // The definition reaching instruction `k` of `b`.
                let r = match before {
                    Some(x) => x,
                    None => {
                        let x = s.start(f, b);
                        before = Some(x);
                        x
                    }
                };
                if r != v {
                    rewrite(f, i, v, r);
                }
            }
        }
    }
    Ok(())
}

fn uses(f: &Func<'_>, i: Inst, v: Val) -> bool {
    let mut u = false;
    f.for_each_use(i, |x| u |= x == v);
    u
}

fn rewrite(f: &mut Func<'_>, i: Inst, v: Val, r: Val) {
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
    // Snapshots may be shared: rewrite a copy.
    if f.snap_entries(i).iter().any(|e| e.1 == v) {
        let ns = f.map_snap(Snap(s), |x| if x == v { r } else { x });
        f.insts[i.idx()].snap = ns.0;
    }
}

/// The definition of `v` reaching each block (Braun et al.'s lookup, every
/// block sealed), with parameters where definitions meet.
struct Repair<'a> {
    v: Val,
    def_block: Block,
    copies: &'a FastMap<Block, Val>,
    cfg: &'a CfgInfo,
    memo: FastMap<Block, Val>,
}

impl Repair<'_> {
    /// At the end of `b`.
    fn end(&mut self, f: &mut Func<'_>, b: Block) -> Val {
        if let Some(&c) = self.copies.get(&b) {
            return c;
        }
        if b == self.def_block {
            return self.v;
        }
        self.start(f, b)
    }

    /// At the start of `b`.
    fn start(&mut self, f: &mut Func<'_>, b: Block) -> Val {
        if let Some(&r) = self.memo.get(&b) {
            return r;
        }
        let inc = self.cfg.incoming(b);
        let r = match inc.len() {
            // Every use is dominated by the definition, so only its own
            // block can start without one.
            0 => self.v,
            1 => {
                let p = f.insts[inc[0].0.idx()].block;
                self.end(f, p)
            }
            _ => {
                let p = f.add_param(b, f.ty(self.v));
                self.memo.insert(b, p);
                for &(t, k) in inc {
                    let pred = f.insts[t.idx()].block;
                    let a = self.end(f, pred);
                    f.push_edge_arg(t, k as usize, a);
                }
                p
            }
        };
        self.memo.insert(b, r);
        r
    }
}
