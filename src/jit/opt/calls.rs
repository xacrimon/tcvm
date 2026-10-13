//! The call-boundary pass (9.8, C1, H11): a call clobbers every register,
//! so a value live across it stays in the region's frame, and must then be
//! one the call-site snapshot names: the collector traces those spill slots
//! while the call runs. Constants, and pure computations of constants and
//! such values (raw pointers among them, which the collector does not see),
//! are computed again after the resume instead, with their later uses given
//! whichever definition reaches them.

use crate::jit::FastMap;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::verify::live_in;
use crate::jit::ir::{Block, CfgInfo, Func, Inst, Val};

pub(crate) fn call_boundaries(f: &mut Func<'_>) -> Result<(), String> {
    f.boundaries = true;
    let cfg = f.cfg();
    if !cfg.rpo.iter().any(|b| f.blocks[b.idx()].resume) {
        return Ok(());
    }
    // A copy may make a value another call kept live across it instead.
    for _ in 0..16 {
        if !boundaries_round(f)? {
            fold_constant_params(f);
            return Ok(());
        }
    }
    Err("the call boundaries did not settle".into())
}

/// Parameters that only join copies of one argument-free value (a constant
/// made again after each call) are that value, made where the block starts:
/// a loop's callee stays a constant (`find_known_callees`) and costs no
/// moves.
fn fold_constant_params(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut changed = false;
    for &b in &cfg.rpo {
        let incoming = cfg.incoming(b);
        if f.blocks[b.idx()].params.len() == 0 || incoming.is_empty() {
            continue;
        }
        let params = f.params(b).to_vec();
        let mut kept = Vec::with_capacity(params.len());
        for (k, &p) in params.iter().enumerate() {
            let mut op = None;
            let same = incoming.iter().all(|&(t, e)| {
                let a = map[f.edge_args(t, e as usize)[k].idx()];
                if a == p {
                    return true;
                }
                let o = f
                    .def_inst(a)
                    .filter(|&d| f.insts[d.idx()].an == 0 && f.op(d).is_pure())
                    .map(|d| f.op(d));
                o.is_some() && *op.get_or_insert(o) == o
            });
            let Some(Some(op)) = op.filter(|_| same) else {
                kept.push(k);
                continue;
            };
            let at = usize::from(
                f.insts_of(b)
                    .first()
                    .is_some_and(|&i| matches!(f.op(i), Op::Resume { .. })),
            );
            let i = f.make_inst(op, &[], None, ExitTag::Type);
            f.insert(b, at, i);
            let r = f.result(i);
            f.vals[r.idx()].ty = f.vals[p.idx()].ty;
            map.push(r);
            map[p.idx()] = r;
            changed = true;
        }
        if kept.len() == params.len() {
            continue;
        }
        for &(t, e) in incoming {
            let args: Vec<Val> = kept
                .iter()
                .map(|&k| f.edge_args(t, e as usize)[k])
                .collect();
            f.set_edge_args(t, e as usize, &args);
        }
        let ps: Vec<Val> = kept.iter().map(|&k| params[k]).collect();
        f.set_params(b, &ps);
    }
    if changed {
        f.apply_replacements(&mut map);
    }
}

/// Keep or copy what is live across each call, and give later uses the
/// copies; whether anything changed.
fn boundaries_round(f: &mut Func<'_>) -> Result<bool, String> {
    let cfg = f.cfg();
    let live = live_in(f, &cfg.rpo);
    let mut changed = false;
    // Each value live across a call, with its copy in each resume block it
    // is live into.
    let mut across: Vec<(Val, FastMap<Block, Val>)> = Vec::new();
    for &b in &cfg.rpo {
        if !f.blocks[b.idx()].resume {
            continue;
        }
        let mut kept = kept_across(f, &cfg, b);
        let mut memo: FastMap<Val, Val> = FastMap::default();
        // The copies go after the resume block's `Resume`.
        let mut at = 1;
        for v in live[b.idx()].iter().collect::<Vec<_>>() {
            let v = Val(v as u32);
            if kept.contains(&same_reg(f, v)) {
                continue;
            }
            let c = match remat(f, v, b, &mut memo, &mut at, &kept) {
                Some(c) => c,
                None => {
                    keep_in_free_reg(f, &cfg, b, v)?;
                    kept.push(same_reg(f, v));
                    changed = true;
                    continue;
                }
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
    changed |= !across.is_empty();
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
    Ok(changed)
}

/// Name `v`, which no register holds after the call into `b`, in the call's
/// snapshot under a register dead there, so the collector traces its spill
/// slot.
fn keep_in_free_reg(f: &mut Func<'_>, cfg: &CfgInfo, b: Block, v: Val) -> Result<(), String> {
    let none = || format!("v{} is live across the call resuming at b{}", v.0, b.0);
    if f.ty(v).rep == crate::jit::ir::types::Rep::Ptr {
        return Err(none());
    }
    let p = *cfg.preds(b).first().ok_or_else(none)?;
    let t = f.terminator(p).ok_or_else(none)?;
    let pc = match f.op(t) {
        Op::Call { pc, .. } | Op::CallMm { pc, .. } => pc,
        _ => return Err(none()),
    };
    let used: Vec<u8> = f.snap_entries(t).iter().map(|e| e.0).collect();
    let r = f
        .call_free
        .get(&pc)
        .and_then(|free| free.iter().find(|&r| !used.contains(&(r as u8))))
        .ok_or_else(none)?;
    let d = f.snaps[f.insts[t.idx()].snap as usize];
    let from = f.snap_pool.len();
    f.snap_pool
        .extend_from_within(d.entries.start as usize..(d.entries.start + d.entries.len) as usize);
    f.snap_pool.push((r as u8, v));
    let s = f.finish_snap(d.frame, d.pc, d.kind, from);
    if let Some(&homes) = f.call_homes.get(&f.insts[t.idx()].snap) {
        f.call_homes.insert(s.0, homes);
    }
    f.insts[t.idx()].snap = s.0;
    Ok(())
}

/// The value whose register `v` shares, as the lowering aliases them:
/// through refining guards and i32 unboxes, whose bits are the operand's
/// (not `LToI`, whose operand holds an i64).
pub(crate) fn same_reg(f: &Func<'_>, mut v: Val) -> Val {
    while let Some(d) = f.def_inst(v) {
        match f.op(d) {
            Op::Guard(_) | Op::Refine(_) | Op::Unbox(crate::jit::ir::types::Rep::I32) => {
                v = f.args(d)[0]
            }
            _ => break,
        }
    }
    v
}

/// Whether `v` is a constant, which needs no spill slot across a call.
pub(crate) fn is_const(f: &Func<'_>, v: Val) -> bool {
    matches!(
        f.def_op(v),
        Some(Op::KVal(_) | Op::KObj(_) | Op::KI32(_) | Op::KI64(_) | Op::KB1(_) | Op::KF64(_))
    )
}

/// The values the call into resume block `b` keeps in the frame: its
/// snapshot's, but raw pointers and constants, which are computed again
/// (the collector traces only values), each as `same_reg` gives it.
pub(crate) fn kept_across(f: &Func<'_>, cfg: &CfgInfo, b: Block) -> Vec<Val> {
    let Some(&p) = cfg.preds(b).first() else {
        return Vec::new();
    };
    let Some(t) = f.terminator(p) else {
        return Vec::new();
    };
    f.snap_entries(t)
        .iter()
        .map(|e| e.1)
        .filter(|&v| f.ty(v).rep != crate::jit::ir::types::Rep::Ptr)
        .map(|v| same_reg(f, v))
        .filter(|&v| !is_const(f, v))
        .collect()
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

/// `v` computed again at position `at` of `b` and on, its operands first;
/// `kept` values are there after the call as they are.
fn remat(
    f: &mut Func<'_>,
    v: Val,
    b: Block,
    memo: &mut FastMap<Val, Val>,
    at: &mut usize,
    kept: &[Val],
) -> Option<Val> {
    if let Some(&c) = memo.get(&v) {
        return Some(c);
    }
    if kept.contains(&same_reg(f, v)) {
        return Some(v);
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
        copied.push(remat(f, a, b, memo, at, kept)?);
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
