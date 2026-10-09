//! Structural checks of a `Func`: SSA dominance of every use (operands, edge
//! arguments, snapshot entries), terminators, edge arity, snapshots on every
//! exiting instruction, and R6 (nothing but constants lives across a call).

use crate::jit::ir::{Block, Func, Inst, NO_SNAP, Val, ValDef};

/// Dominators by the Cooper-Harvey-Kennedy iteration over an RPO.
pub(crate) fn dominators(f: &Func<'_>, rpo: &[Block], preds: &[Vec<Block>]) -> Vec<Option<Block>> {
    let n = f.blocks.len();
    let mut order = vec![usize::MAX; n];
    for (i, b) in rpo.iter().enumerate() {
        order[b.idx()] = i;
    }
    let mut idom: Vec<Option<Block>> = vec![None; n];
    idom[f.entry.idx()] = Some(f.entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new: Option<Block> = None;
            for &p in &preds[b.idx()] {
                if idom[p.idx()].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(mut a) => {
                        let mut c = p;
                        while a != c {
                            while order[a.idx()] > order[c.idx()] {
                                a = idom[a.idx()].unwrap();
                            }
                            while order[c.idx()] > order[a.idx()] {
                                c = idom[c.idx()].unwrap();
                            }
                        }
                        a
                    }
                });
            }
            if new.is_some() && idom[b.idx()] != new {
                idom[b.idx()] = new;
                changed = true;
            }
        }
    }
    idom
}

pub(crate) fn dominates(idom: &[Option<Block>], a: Block, mut b: Block) -> bool {
    loop {
        if a == b {
            return true;
        }
        match idom[b.idx()] {
            Some(p) if p != b => b = p,
            _ => return false,
        }
    }
}

pub(crate) fn verify(f: &Func<'_>) -> Result<(), String> {
    let rpo = f.rpo();
    let preds = f.preds();
    let idom = dominators(f, &rpo, &preds);
    let mut pos = vec![(Block(u32::MAX), usize::MAX); f.insts.len()];
    for &b in &rpo {
        for (k, &i) in f.blocks[b.idx()].insts.iter().enumerate() {
            pos[i.idx()] = (b, k);
        }
    }
    let def_pos = |v: Val| -> Option<(Block, usize)> {
        match f.vals[v.idx()].def {
            ValDef::Inst(i, _) => {
                let p = pos[i.idx()];
                (p.1 != usize::MAX).then_some(p)
            }
            ValDef::Param(b, _) => Some((b, 0)),
        }
    };
    let is_param = |v: Val| matches!(f.vals[v.idx()].def, ValDef::Param(..));
    let check_use = |v: Val, b: Block, k: usize, what: &str, i: Inst| -> Result<(), String> {
        let Some((db, dk)) = def_pos(v) else {
            return Err(format!(
                "v{} used by {what} of i{} in b{} has no placed definition",
                v.0, i.0, b.0
            ));
        };
        let ok = if db == b {
            is_param(v) || dk < k
        } else {
            dominates(&idom, db, b)
        };
        if !ok {
            return Err(format!(
                "v{} used by {what} of i{} in b{} does not dominate it",
                v.0, i.0, b.0
            ));
        }
        Ok(())
    };
    for &b in &rpo {
        let bd = &f.blocks[b.idx()];
        let Some(&last) = bd.insts.last() else {
            return Err(format!("b{} is empty", b.0));
        };
        if !f.op(last).is_terminator() {
            return Err(format!("b{} does not end in a terminator", b.0));
        }
        for (k, &i) in bd.insts.iter().enumerate() {
            let d = &f.insts[i.idx()];
            if d.op.is_terminator() && k + 1 != bd.insts.len() {
                return Err(format!("terminator i{} in the middle of b{}", i.0, b.0));
            }
            if d.block != b {
                return Err(format!(
                    "i{} records b{} but sits in b{}",
                    i.0, d.block.0, b.0
                ));
            }
            for &a in f.args(i) {
                check_use(a, b, k, "operand", i)?;
            }
            check_reps(f, i)?;
            if d.op.may_deopt() && d.snap == NO_SNAP {
                return Err(format!(
                    "exiting i{} ({}) has no snapshot",
                    i.0,
                    d.op.name()
                ));
            }
            if d.snap != NO_SNAP {
                for &(_, v) in &f.snaps[d.snap as usize].entries {
                    check_use(v, b, k, "snapshot", i)?;
                }
            }
            for e in f.edges(i) {
                let tb = &f.blocks[e.target.idx()];
                if tb.dead {
                    return Err(format!("edge from b{} to dead b{}", b.0, e.target.0));
                }
                if e.args.len() != tb.params.len() {
                    return Err(format!(
                        "edge b{} -> b{} passes {} args for {} params",
                        b.0,
                        e.target.0,
                        e.args.len(),
                        tb.params.len()
                    ));
                }
                for &a in &e.args {
                    check_use(a, b, k + 1, "edge argument", i)?;
                }
            }
        }
    }
    check_r6(f, &rpo, &pos)?;
    Ok(())
}

/// No value other than a constant is live into a resume block: everything
/// else lives in home slots across the call (R6).
fn check_r6(f: &Func<'_>, rpo: &[Block], _pos: &[(Block, usize)]) -> Result<(), String> {
    let live = live_in(f, rpo);
    for &b in rpo {
        if !f.blocks[b.idx()].resume {
            continue;
        }
        for v in live[b.idx()].iter() {
            if !f.def_op(Val(v as u32)).is_some_and(|op| op.is_const()) {
                return Err(format!("v{v} is live across the call resuming at b{}", b.0));
            }
        }
    }
    Ok(())
}

/// A set of values as a bit vector.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ValSet(Vec<u64>);

impl ValSet {
    pub(crate) fn new(n: usize) -> Self {
        ValSet(vec![0; n.div_ceil(64)])
    }

    pub(crate) fn insert(&mut self, v: usize) -> bool {
        let (w, b) = (v / 64, v % 64);
        let had = self.0[w] >> b & 1 != 0;
        self.0[w] |= 1 << b;
        !had
    }

    pub(crate) fn remove(&mut self, v: usize) {
        self.0[v / 64] &= !(1 << (v % 64));
    }

    pub(crate) fn contains(&self, v: usize) -> bool {
        self.0[v / 64] >> (v % 64) & 1 != 0
    }

    /// `self |= other`; whether anything was added.
    pub(crate) fn union(&mut self, other: &ValSet) -> bool {
        let mut changed = false;
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            let n = *a | *b;
            changed |= n != *a;
            *a = n;
        }
        changed
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &bits)| {
            let mut bits = bits;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let t = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(w * 64 + t)
            })
        })
    }
}

/// Values live into each block: used in it or later before being defined,
/// snapshot entries counting as uses. Block parameters are defined at entry.
pub(crate) fn live_in(f: &Func<'_>, rpo: &[Block]) -> Vec<ValSet> {
    let n = f.vals.len();
    let mut live = vec![ValSet::new(n); f.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().rev() {
            let mut cur = ValSet::new(n);
            for s in f.succs(b) {
                cur.union(&live[s.idx()]);
            }
            for &i in f.blocks[b.idx()].insts.iter().rev() {
                for e in f.edges(i) {
                    for &p in &f.blocks[e.target.idx()].params {
                        cur.remove(p.idx());
                    }
                    for &a in &e.args {
                        cur.insert(a.idx());
                    }
                }
                for r in f.results(i) {
                    cur.remove(r.idx());
                }
                for &a in f.args(i) {
                    cur.insert(a.idx());
                }
                let s = f.insts[i.idx()].snap;
                if s != NO_SNAP {
                    for &(_, v) in &f.snaps[s as usize].entries {
                        cur.insert(v.idx());
                    }
                }
            }
            for &p in &f.blocks[b.idx()].params {
                cur.remove(p.idx());
            }
            if cur != live[b.idx()] {
                live[b.idx()] = cur;
                changed = true;
            }
        }
    }
    live
}

/// Each operation's operands are the representations it takes.
fn check_reps(f: &Func<'_>, i: Inst) -> Result<(), String> {
    use crate::jit::ir::ops::{HelperId, Op::*};
    use crate::jit::ir::types::Rep::{self, *};
    let op = f.op(i);
    let args = f.args(i);
    let want: &[Rep] = match op {
        Store(_) | Unbox(_) | IsType(_) | IsFalsy | Guard(_) => &[Val],
        SameBits | GuardSame => &[Val, Val],
        GuardTrue | GuardFalse | Br => &[B1],
        IAdd | ISub | IMul | IAddNo | ISubNo | IMulNo | IAnd | IOr | IXor | IShl | IShr
        | IDivFloor | IModFloor | ICmp(_) => &[I32, I32],
        INeg | INot | IToF | IToL => &[I32],
        LAdd | LSub | LMul | LAnd | LOr | LXor | LShl | LShr | LDivFloor | LModFloor | LCmp(_) => {
            &[I64, I64]
        }
        LNeg | LNot | LToF | LToI => &[I64],
        FAdd | FSub | FMul | FDiv | FIDiv | FCmp(_) | Helper(HelperId::FMod | HelperId::FPow) => {
            &[F64, F64]
        }
        FNeg | FAbs | FSqrt | FFloor | FCeil | FToIExact => &[F64],
        Box => {
            if f.ty(args[0]).rep == Val {
                return Err(format!("i{} boxes a val", i.0));
            }
            &[]
        }
        Select => {
            if f.ty(args[0]).rep != B1 || f.ty(args[1]).rep != f.ty(args[2]).rep {
                return Err(format!("i{} selects mismatched representations", i.0));
            }
            &[]
        }
        _ => &[],
    };
    for (k, (&a, &r)) in args.iter().zip(want).enumerate() {
        if f.ty(a).rep != r {
            return Err(format!(
                "i{} ({}) operand {k} v{} is {:?}, not {r:?}",
                i.0,
                op.name(),
                a.0,
                f.ty(a).rep
            ));
        }
    }
    Ok(())
}
