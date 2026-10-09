//! Type inference (9.2) and block-parameter narrowing (9.6): optimistic
//! forward dataflow over the SSA graph, then parameters whose incoming values
//! are all one numeric kind take that kind's unboxed representation.

use crate::jit::build::incoming;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, Ty, TypeSet};
use crate::jit::ir::{Block, Func, Inst, Val, ValDef};

/// Whether the builder fixed the result type of `op` from knowledge the
/// operands do not carry.
fn keeps_type(op: Op) -> bool {
    matches!(
        op,
        Op::Load(_) | Op::KObj(_) | Op::Resume { .. } | Op::UpvalValue(_) | Op::Helper(_)
    )
}

pub(crate) fn infer(f: &mut Func<'_>) {
    let rpo = f.rpo();
    let inc = incoming(f);
    // Parameters start at the bottom and only grow.
    for &b in &rpo {
        for &p in &f.blocks[b.idx()].params {
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
        for &b in &rpo {
            let params = f.blocks[b.idx()].params.clone();
            for (k, &p) in params.iter().enumerate() {
                let rep = f.vals[p.idx()].ty.rep;
                let mut t = Ty {
                    rep,
                    set: TypeSet::empty(),
                    refine: crate::jit::ir::types::Refine::None,
                };
                let mut first = true;
                for &(term, e) in &inc[b.idx()] {
                    if f.blocks[f.insts[term.idx()].block.idx()].dead {
                        continue;
                    }
                    let a = f.edges(term)[e].args[k];
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
            let insts = f.blocks[b.idx()].insts.clone();
            for i in insts {
                let op = f.op(i);
                if keeps_type(op) || f.insts[i.idx()].rn != 1 {
                    continue;
                }
                let tys: Vec<Ty> = f.args(i).iter().map(|&a| f.vals[a.idx()].ty).collect();
                let t = op.result_ty(&tys);
                let r = f.result(i);
                if t != f.vals[r.idx()].ty {
                    f.vals[r.idx()].ty = t;
                    changed = true;
                }
            }
        }
    }
}

/// The unboxed representation a parameter of type `t` can take.
fn narrow_rep(t: Ty, args: &[Ty]) -> Option<Rep> {
    if t.rep != Rep::Val || t.set.is_empty() {
        return None;
    }
    if t.within(TypeSet::SMALL) {
        Some(Rep::I32)
    } else if t.within(TypeSet::FLOAT) {
        Some(Rep::F64)
    } else if t.within(TypeSet::INT) && args.iter().any(|a| a.rep == Rep::I64) {
        Some(Rep::I64)
    } else {
        None
    }
}

/// Narrow parameters; returns whether any changed.
pub(crate) fn narrow(f: &mut Func<'_>) -> bool {
    let inc = incoming(f);
    let mut changed = false;
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead || f.blocks[b].params.is_empty() {
            continue;
        }
        let block = Block(b as u32);
        let params = f.blocks[b].params.clone();
        for (k, &p) in params.iter().enumerate() {
            let arg_tys: Vec<Ty> = inc[b]
                .iter()
                .map(|&(t, e)| f.vals[f.edges(t)[e].args[k].idx()].ty)
                .collect();
            let Some(rep) = narrow_rep(f.vals[p.idx()].ty, &arg_tys) else {
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
            let boxed = f.make_inst(Op::Box, &[p], None, ExitTag::Type);
            f.insts[boxed.idx()].block = block;
            f.blocks[b].insts.insert(0, boxed);
            let bv = f.result(boxed);
            f.vals[bv.idx()].ty = Ty::val(set);
            replace_uses_except(f, p, bv, boxed);
            for &(term, e) in &inc[b] {
                let a = f.edges(term)[e].args[k];
                let na = convert(f, a, rep, f.insts[term.idx()].block);
                f.edges_mut(term)[e].args[k] = na;
            }
        }
    }
    changed
}

fn replace_uses_except(f: &mut Func<'_>, from: Val, to: Val, except: Inst) {
    let d = f.insts[except.idx()];
    let skip = d.a0 as usize..d.a0 as usize + d.an as usize;
    for (i, a) in f.args.iter_mut().enumerate() {
        if *a == from && !skip.contains(&i) {
            *a = to;
        }
    }
    for e in &mut f.edges {
        for a in &mut e.args {
            if *a == from {
                *a = to;
            }
        }
    }
    for s in &mut f.snaps {
        for (_, v) in &mut s.entries {
            if *v == from {
                *v = to;
            }
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

/// Whether `v` is a block parameter.
pub(crate) fn is_param(f: &Func<'_>, v: Val) -> bool {
    matches!(f.vals[v.idx()].def, ValDef::Param(..))
}

/// Make every edge argument the representation of its parameter, boxing or
/// unboxing in the predecessor.
pub(crate) fn legalize(f: &mut Func<'_>) {
    let inc = incoming(f);
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead {
            continue;
        }
        let params = f.blocks[b].params.clone();
        for (k, &p) in params.iter().enumerate() {
            let rep = f.vals[p.idx()].ty.rep;
            for &(term, e) in &inc[b] {
                let a = f.edges(term)[e].args[k];
                if f.vals[a.idx()].ty.rep != rep {
                    let na = convert(f, a, rep, f.insts[term.idx()].block);
                    f.edges_mut(term)[e].args[k] = na;
                }
            }
        }
    }
}
