//! The development verifier: what a finished graph must satisfy, checked
//! from the graph alone. The dominators here are computed afresh, not read
//! from what the emitter kept as it went.

use super::ops::{Arg, Dialect, Flags, Op};
use super::types::Rep;
use super::{Block, BlockFlags, Graph, Val};

/// Immediate dominators over every edge (Cooper, Harvey, Kennedy), by a
/// block's position in `g.order`.
fn dominators(g: &Graph, pos: &[u32]) -> Vec<u32> {
    let n = g.order.len();
    let mut idom = vec![u32::MAX; n];
    idom[0] = 0;
    let mut changed = true;
    while changed {
        changed = false;
        for k in 1..n {
            let mut new = u32::MAX;
            for p in g.preds(g.order[k]) {
                let mut p = pos[p.idx()];
                if idom[p as usize] == u32::MAX {
                    continue;
                }
                if new != u32::MAX {
                    // A late predecessor comes later in the order, so depth
                    // is not position: walk by known dominators' positions.
                    let mut q = new;
                    while p != q {
                        while p > q {
                            p = idom[p as usize];
                        }
                        while q > p {
                            q = idom[q as usize];
                        }
                    }
                }
                new = p;
            }
            if new != idom[k] {
                idom[k] = new;
                changed = true;
            }
        }
    }
    idom
}

pub(crate) fn verify(g: &Graph, lowered: bool) -> Result<(), String> {
    let Some(&entry) = g.order.first() else {
        return Err("no blocks".into());
    };
    if g.facts.len() != g.insts.len() {
        return Err("facts and instructions differ in number".into());
    }
    // Blocks: contiguous in order, parameters first, one terminator, last.
    let mut pos = vec![u32::MAX; g.blocks.len()];
    let mut block_of = vec![Block::NONE; g.insts.len()];
    let mut at = 0;
    for (k, &b) in g.order.iter().enumerate() {
        let d = g.block(b);
        if !d.flags.contains(BlockFlags::SEALED) || d.start != at || d.end <= d.start {
            return Err(format!("b{} is open, empty or out of place", b.0));
        }
        pos[b.idx()] = k as u32;
        let mut body = false;
        for v in d.start..d.end {
            block_of[v as usize] = b;
            let op = g.insts[v as usize].op;
            if op.has(Flags::TERMINATOR) != (v + 1 == d.end) {
                return Err(format!(
                    "v{v}: a terminator inside b{}, or none at its end",
                    b.0
                ));
            }
            if op == Op::Param && body {
                return Err(format!("v{v}: a parameter after other instructions"));
            }
            body |= op != Op::Param;
        }
        at = d.end;
    }
    if at as usize != g.insts.len() {
        return Err("instructions past the last block".into());
    }
    // Edges: a block's predecessors are exactly the blocks that branch to
    // it, forward ones first.
    let mut edges = 0i64;
    for &b in &g.order {
        let d = g.block(b);
        if b != entry && d.npreds == 0 {
            return Err(format!("b{} has no predecessor", b.0));
        }
        for (k, &p) in g.preds(b).iter().enumerate() {
            if p.is_none() || pos[p.idx()] == u32::MAX {
                return Err(format!("b{}: predecessor {k} never came", b.0));
            }
            let forward = pos[p.idx()] < pos[b.idx()];
            if forward != (k < d.nfwd as usize) {
                return Err(format!("b{}: predecessor b{} on the wrong side", b.0, p.0));
            }
            let (succs, n) = g.succs(g.term(p));
            let to = succs[..n].iter().filter(|&&s| s == b).count();
            let from = g.preds(b).iter().filter(|&&q| q == p).count();
            if to != from {
                return Err(format!("b{} and b{} disagree on their edges", p.0, b.0));
            }
        }
        edges += d.npreds as i64 - g.succs(g.term(b)).1 as i64;
        let header = d.flags.contains(BlockFlags::LOOP_HEADER);
        if header && (d.lp.is_none() || g.loops[d.lp.idx()].header != b) {
            return Err(format!("b{} heads no loop", b.0));
        }
    }
    if edges != 0 {
        return Err("an edge into a block that was never opened".into());
    }
    let idom = dominators(g, &pos);
    for (k, &b) in g.order.iter().enumerate() {
        let want = g.order[idom[k] as usize];
        if g.block(b).idom != want {
            return Err(format!(
                "b{}: dominator b{} kept, b{} found",
                b.0,
                g.block(b).idom.0,
                want.0
            ));
        }
    }
    let dominates = |a: Block, b: Block| {
        let (a, mut b) = (pos[a.idx()], pos[b.idx()]);
        while b > a {
            b = idom[b as usize];
        }
        a == b
    };
    // Instructions: their form, and each use dominated by its definition.
    for v in 0..g.insts.len() as u32 {
        let i = &g.insts[v as usize];
        let b = block_of[v as usize];
        let info = i.op.info();
        if lowered && info.dialect == Dialect::H {
            return Err(format!("v{v}: high-level {:?} in a low-level graph", i.op));
        }
        if let Err(what) = g.form(i) {
            return Err(format!("v{v}: {:?} breaks its form: {what}", i.op));
        }
        match i.op {
            Op::Param => {
                let preds = g.preds(b);
                if i.b as usize != preds.len() {
                    return Err(format!("v{v}: an input per predecessor"));
                }
                for (k, &x) in g.inputs(Val(v)).iter().enumerate() {
                    if x == u32::MAX || g.insts[x as usize].rep != i.rep {
                        return Err(format!(
                            "v{v}: input {k} missing or of another representation"
                        ));
                    }
                    if !dominates(block_of[x as usize], preds[k]) {
                        return Err(format!("v{v}: input v{x} does not reach b{}", preds[k].0));
                    }
                }
            }
            Op::FrameState => {
                let (_, entries) = g.entries(Val(v));
                if !entries.is_sorted_by(|x, y| x[0] & !super::HOME < y[0] & !super::HOME) {
                    return Err(format!("v{v}: entries out of register order"));
                }
            }
            // Its operand is typed by definition: nothing else names what
            // makes the pointer one.
            Op::UnboxPtr if g.facts[i.a as usize].set.unboxed() != Rep::Ptr => {
                return Err(format!("v{v}: unchecked unbox of an untyped value"));
            }
            Op::UnboxPtr => {}
            // It stands under its branch: up the dominators there is a block
            // entered only by the edge of a branch on its condition that
            // its arm names.
            Op::Proved => {
                let arm = i.aux & super::ops::SET_ARM == 0;
                let mut at = b;
                let under = loop {
                    let d = g.block(at);
                    if d.npreds == 1 {
                        let t = g.term(g.preds(at)[0]);
                        if t.op == Op::Br && t.a == i.a && t.b != t.c {
                            break (t.b == at.0) == arm;
                        }
                    }
                    if d.idom == at {
                        break false;
                    }
                    at = d.idom;
                };
                if !under {
                    return Err(format!("v{v}: not under the branch it names"));
                }
            }
            // Nothing more than its form.
            Op::K32
            | Op::K64
            | Op::KF64
            | Op::KObj
            | Op::Load
            | Op::Store
            | Op::LoadIdx
            | Op::StoreIdx
            | Op::Base
            | Op::Thread
            | Op::SlotLoad
            | Op::SlotStore
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Not
            | Op::Neg
            | Op::Shl
            | Op::Shr
            | Op::Sar
            | Op::SDiv
            | Op::UDiv
            | Op::ZExt32
            | Op::SExt32
            | Op::Trunc32
            | Op::Cmp
            | Op::Select
            | Op::AddOvf
            | Op::SubOvf
            | Op::MulOvf
            | Op::NegOvf
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FNeg
            | Op::FAbs
            | Op::FSqrt
            | Op::FFloor
            | Op::FCeil
            | Op::FCmp
            | Op::IToF
            | Op::LToF
            | Op::FToI
            | Op::FToL
            | Op::BoxI32
            | Op::BoxF64
            | Op::BoxPtr
            | Op::UnboxF64
            | Op::Guard
            | Op::GuardNot
            | Op::Deopt
            | Op::LazyExit
            | Op::Jump
            | Op::Br
            | Op::Ret
            | Op::CallHelper
            | Op::CheckType
            | Op::IsType
            | Op::BoxI64
            | Op::UnboxI64
            | Op::ToF64
            | Op::DivFloor
            | Op::ModFloor
            | Op::LuaShl
            | Op::LuaShr
            | Op::FToIExact
            | Op::FToLExact
            | Op::LToI
            | Op::AbsOvf
            | Op::FIDiv
            | Op::FMod
            | Op::Barrier
            | Op::NewTable
            | Op::GcCheck
            | Op::Arith
            | Op::Unm
            | Op::Compare
            | Op::Truthy
            | Op::GetIndex
            | Op::SetIndex
            | Op::GetField
            | Op::SetField => {}
        }
        let mut bad = None;
        if i.op != Op::Param {
            g.uses(i, |x| {
                let db = block_of[x.idx()];
                if x.0 >= v || !(db == b || dominates(db, b)) {
                    bad = Some(x);
                }
            });
        }
        if let Some(x) = bad {
            return Err(format!("v{v}: v{} does not dominate its use", x.0));
        }
        if info.args.contains(&Arg::Blk) && !i.op.has(Flags::TERMINATOR) {
            return Err(format!("v{v}: a block operand outside a terminator"));
        }
    }
    Ok(())
}
