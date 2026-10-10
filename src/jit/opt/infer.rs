//! Type inference (9.2) and block-parameter narrowing (9.6): optimistic
//! forward dataflow over the SSA graph, then parameters whose incoming values
//! are all one numeric kind take that kind's unboxed representation.

use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, Ty, TypeSet};
use crate::jit::ir::{Block, Func, Val};

/// Whether the builder fixed the result type of `op` from knowledge the
/// operands do not carry.
fn keeps_type(op: Op) -> bool {
    matches!(
        op,
        Op::Load(_) | Op::KObj(_) | Op::Resume { .. } | Op::UpvalValue(_) | Op::Helper(_)
    )
}

pub(crate) fn infer(f: &mut Func<'_>) {
    let cfg = f.cfg();
    // Parameters start at the bottom and only grow.
    for &b in &cfg.rpo {
        for k in 0..f.blocks[b.idx()].params.len() {
            let p = f.params(b)[k];
            let t = &mut f.vals[p.idx()].ty;
            *t = Ty {
                set: TypeSet::empty(),
                ..*t
            };
        }
    }
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 50 {
        changed = false;
        rounds += 1;
        for &b in &cfg.rpo {
            for k in 0..f.blocks[b.idx()].params.len() {
                let p = f.params(b)[k];
                let rep = f.vals[p.idx()].ty.rep;
                let mut t = Ty {
                    rep,
                    set: TypeSet::empty(),
                    refine: crate::jit::ir::types::Refine::None,
                };
                let mut first = true;
                for &(term, e) in cfg.incoming(b) {
                    let a = f.edge_args(term, e as usize)[k];
                    let at = f.vals[a.idx()].ty;
                    if first {
                        t = Ty { rep, ..at };
                        first = false;
                    } else {
                        t = Ty {
                            rep,
                            set: t.set | at.set,
                            refine: if t.refine == at.refine {
                                t.refine
                            } else {
                                crate::jit::ir::types::Refine::None
                            },
                        };
                    }
                }
                if t != f.vals[p.idx()].ty {
                    f.vals[p.idx()].ty = t;
                    changed = true;
                }
            }
            for k in 0..f.blocks[b.idx()].insts.len() {
                let i = f.insts_of(b)[k];
                let op = f.op(i);
                if keeps_type(op) || f.insts[i.idx()].rn != 1 {
                    continue;
                }
                let t = f.result_ty(op, f.args(i));
                let r = f.result(i);
                if t != f.vals[r.idx()].ty {
                    f.vals[r.idx()].ty = t;
                    changed = true;
                }
            }
        }
    }
}

/// The unboxed representation a parameter of type `t` can take, `i64`
/// when one of its arguments is one.
fn narrow_rep(t: Ty, i64_arg: bool) -> Option<Rep> {
    if t.rep != Rep::Val || t.set.is_empty() {
        return None;
    }
    if t.within(TypeSet::SMALL) {
        Some(Rep::I32)
    } else if t.within(TypeSet::FLOAT) {
        Some(Rep::F64)
    } else if t.within(TypeSet::INT) && i64_arg {
        Some(Rep::I64)
    } else {
        None
    }
}

/// Narrow parameters; returns whether any changed.
pub(crate) fn narrow(f: &mut Func<'_>) -> bool {
    let cfg = f.cfg();
    let mut changed = false;
    for &block in &cfg.rpo {
        for k in 0..f.blocks[block.idx()].params.len() {
            let p = f.params(block)[k];
            // An argument boxed for the edge counts as what it boxes.
            let i64_arg = cfg.incoming(block).iter().any(|&(t, e)| {
                let a = f.edge_args(t, e as usize)[k];
                let a = match f.def_op(a) {
                    Some(Op::Box) => f.args(f.def_inst(a).unwrap())[0],
                    _ => a,
                };
                f.vals[a.idx()].ty.rep == Rep::I64
            });
            let Some(rep) = narrow_rep(f.vals[p.idx()].ty, i64_arg) else {
                continue;
            };
            changed = true;
            // The parameter becomes `rep`; its old uses see it boxed.
            let set = f.vals[p.idx()].ty.set;
            f.vals[p.idx()].ty = Ty {
                rep,
                set,
                refine: crate::jit::ir::types::Refine::None,
            };
            box_uses(f, p);
            for &(term, e) in cfg.incoming(block) {
                let a = f.edge_args(term, e as usize)[k];
                let na = convert(f, a, rep, f.insts[term.idx()].block);
                f.edge_args_mut(term, e as usize)[k] = na;
            }
        }
    }
    changed
}

/// Give each operand and edge use of the now unboxed `p` a box of its own
/// just before it; GVN merges the dominated ones. Snapshots take `p` as is.
fn box_uses(f: &mut Func<'_>, p: Val) {
    let set = f.vals[p.idx()].ty.set;
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead {
            continue;
        }
        let b = Block(b as u32);
        let mut k = 0;
        while k < f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            let in_args = f.args(i).contains(&p);
            let in_edges = f.edges(i).iter().any(|e| f.vl(e.args).contains(&p));
            if !in_args && !in_edges {
                k += 1;
                continue;
            }
            let bx = f.make_inst(Op::Box, &[p], None, ExitTag::Type);
            f.insert(b, k, bx);
            let bv = f.result(bx);
            f.vals[bv.idx()].ty = Ty::val(set);
            for a in f.args_mut(i) {
                if *a == p {
                    *a = bv;
                }
            }
            for e in 0..f.edges(i).len() {
                for a in f.edge_args_mut(i, e) {
                    if *a == p {
                        *a = bv;
                    }
                }
            }
            k += 2;
        }
    }
}

/// `a` as `rep`, with any conversion inserted at the end of `at`.
pub(crate) fn convert(f: &mut Func<'_>, a: Val, rep: Rep, at: Block) -> Val {
    let ty = f.vals[a.idx()].ty;
    if ty.rep == rep {
        return a;
    }
    if let Some(Op::Box) = f.def_op(a) {
        let x = f.args(f.def_inst(a).unwrap())[0];
        if f.vals[x.idx()].ty.rep == rep {
            return x;
        }
    }
    let op = match (ty.rep, rep) {
        (Rep::I32, Rep::I64) => Op::IToL,
        (Rep::I32, Rep::F64) => Op::IToF,
        (Rep::Val, r) => {
            if let Some(Op::KVal(bits)) = f.def_op(a) {
                let v: crate::env::value::Value<'_> =
                    unsafe { std::mem::transmute::<u64, crate::env::value::Value<'static>>(bits) };
                let k = match r {
                    Rep::I32 => v.get_small().map(Op::KI32),
                    Rep::I64 => v.get_integer().map(Op::KI64),
                    Rep::F64 => v.get_float().map(|x| Op::KF64(x.to_bits())),
                    _ => None,
                };
                if let Some(k) = k {
                    let i = f.make_inst(k, &[], None, ExitTag::Type);
                    f.insert_before_term(at, i);
                    return f.result(i);
                }
            }
            Op::Unbox(r)
        }
        (Rep::I64, Rep::Val)
        | (Rep::I32, Rep::Val)
        | (Rep::F64, Rep::Val)
        | (Rep::B1, Rep::Val) => Op::Box,
        _ => Op::Unbox(rep),
    };
    let i = f.make_inst(op, &[a], None, ExitTag::Type);
    f.insert_before_term(at, i);
    let r = f.result(i);
    if matches!(op, Op::Unbox(_)) {
        f.vals[r.idx()].ty.set = ty.set & Ty::of_rep(rep).set;
    }
    r
}

/// Make every edge argument the representation of its parameter, boxing or
/// unboxing in the predecessor.
pub(crate) fn legalize(f: &mut Func<'_>) {
    let cfg = f.cfg();
    for &b in &cfg.rpo {
        for k in 0..f.blocks[b.idx()].params.len() {
            let rep = f.vals[f.params(b)[k].idx()].ty.rep;
            for &(term, e) in cfg.incoming(b) {
                let a = f.edge_args(term, e as usize)[k];
                if f.vals[a.idx()].ty.rep != rep {
                    let na = convert(f, a, rep, f.insts[term.idx()].block);
                    f.edge_args_mut(term, e as usize)[k] = na;
                }
            }
        }
    }
}
