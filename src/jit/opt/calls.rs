//! The call-boundary pass (9.8): a call clobbers every register and pops the
//! native frame, so a value live into a resume block (R6: a constant, or a
//! pure computation of constants and loop-invariant values LICM moved) is
//! computed again after the resume, with its later uses given whichever
//! definition reaches them.

use crate::jit::FastMap;
use crate::jit::ir::ops::Op;
use crate::jit::ir::verify::live_in;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Val};

pub(crate) fn call_boundaries(f: &mut Func<'_>) -> Result<(), String> {
    f.boundaries = true;
    let cfg = f.cfg();
    if !cfg.rpo.iter().any(|b| f.blocks[b.idx()].resume) {
        return Ok(());
    }
    let live = live_in(f, &cfg.rpo);
    // Each value live across a call, with its copy in each resume block it
    // is live into.
    let mut across: Vec<(Val, FastMap<Block, Val>)> = Vec::new();
    for &b in &cfg.rpo {
        if !f.blocks[b.idx()].resume {
            continue;
        }
        let mut memo: FastMap<Val, Val> = FastMap::default();
        // The copies go after the resume block's `Resume`.
        let mut at = 1;
        for v in live[b.idx()].iter().collect::<Vec<_>>() {
            let v = Val(v as u32);
            let Some(c) = remat(f, v, b, &mut memo, &mut at) else {
                return Err(format!(
                    "v{} is live across the call resuming at b{}",
                    v.0, b.0
                ));
            };
            match across.iter_mut().find(|(x, _)| *x == v) {
                Some((_, cs)) => {
                    cs.insert(b, c);
                }
                None => {
                    let mut cs = FastMap::default();
                    cs.insert(b, c);
                    across.push((v, cs));
                }
            }
        }
    }
    let cfg = f.cfg();
    for (v, copies) in across {
        let d = f.def_inst(v).unwrap();
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
                    f.replace_uses(i, v, r);
                }
            }
        }
    }
    Ok(())
}

/// Whether `v` can be computed again after a call: a constant, a pure
/// instruction of such operands, or a type guard's refinement of one,
/// which holds again since the value is the same.
pub(crate) fn rematerializable(f: &Func<'_>, v: Val) -> bool {
    let Some(d) = f.def_inst(v) else {
        return false;
    };
    let op = f.op(d);
    // A pure helper is a call again.
    (op.is_pure() && !matches!(op, Op::Helper(_)) || matches!(op, Op::Guard(_) | Op::Refine(_)))
        && f.args(d).iter().all(|&a| rematerializable(f, a))
}

/// `v` computed again at position `at` of `b` and on, its operands first.
fn remat(
    f: &mut Func<'_>,
    v: Val,
    b: Block,
    memo: &mut FastMap<Val, Val>,
    at: &mut usize,
) -> Option<Val> {
    if let Some(&c) = memo.get(&v) {
        return Some(c);
    }
    let d = f.def_inst(v)?;
    let op = match f.op(d) {
        Op::Guard(set) => Op::Refine(set),
        op if op.is_pure() && !matches!(op, Op::Helper(_)) || matches!(op, Op::Refine(_)) => op,
        _ => return None,
    };
    let args: Vec<Val> = f.args(d).to_vec();
    let mut copied = Vec::with_capacity(args.len());
    for a in args {
        copied.push(remat(f, a, b, memo, at)?);
    }
    let c = f.make_inst(op, &copied, None, f.insts[d.idx()].tag);
    let cv = f.result(c);
    f.vals[cv.idx()].ty = f.ty(v);
    f.insert(b, *at, c);
    *at += 1;
    memo.insert(v, cv);
    Some(cv)
}

fn uses(f: &Func<'_>, i: Inst, v: Val) -> bool {
    let mut u = false;
    f.for_each_use(i, |x| u |= x == v);
    u
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
