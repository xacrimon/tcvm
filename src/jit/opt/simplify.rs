//! Local folds: redundant guards and conversions, constant compares and
//! branches.

use crate::env::value::Value;
use crate::jit::ir::ops::{Cc, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Func, Inst, Val};

fn konst_val(f: &Func<'_>, v: Val) -> Option<Value<'static>> {
    match f.def_op(v) {
        Some(Op::KVal(bits)) => Some(unsafe { std::mem::transmute::<u64, Value<'static>>(bits) }),
        _ => None,
    }
}

fn konst_int(f: &Func<'_>, v: Val) -> Option<i64> {
    match f.def_op(v) {
        Some(Op::KI32(n)) => Some(n as i64),
        Some(Op::KI64(n)) => Some(n),
        _ => None,
    }
}

fn konst_f64(f: &Func<'_>, v: Val) -> Option<f64> {
    match f.def_op(v) {
        Some(Op::KF64(b)) => Some(f64::from_bits(b)),
        _ => None,
    }
}

fn konst_b1(f: &Func<'_>, v: Val) -> Option<bool> {
    match f.def_op(v) {
        Some(Op::KB1(c)) => Some(c),
        _ => None,
    }
}

/// One round of folds; returns whether anything changed.
pub(crate) fn simplify(f: &mut Func<'_>) -> bool {
    let mut map: Vec<Val> = (0..f.vals.len() as u32).map(Val).collect();
    let mut changed = false;
    let mut remove: Vec<Inst> = Vec::new();
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead {
            continue;
        }
        let insts = f.blocks[b].insts.clone();
        for i in insts {
            let op = f.op(i);
            let args: Vec<Val> = f.args(i).to_vec();
            let res = |f: &Func<'_>| f.result(i);
            match op {
                Op::Guard(set) => {
                    let t = f.ty(args[0]);
                    if t.within(set) && !t.set.is_empty() {
                        map[res(f).idx()] = args[0];
                        remove.push(i);
                        changed = true;
                    }
                }
                Op::Unbox(rep) => {
                    let x = args[0];
                    if f.ty(x).rep == rep {
                        map[res(f).idx()] = x;
                        changed = true;
                        continue;
                    }
                    if let Some(Op::Box) = f.def_op(x) {
                        let y = f.args(f.def_inst(x).unwrap())[0];
                        if f.ty(y).rep == rep {
                            map[res(f).idx()] = y;
                            changed = true;
                            continue;
                        }
                        if rep == Rep::I64 && f.ty(y).rep == Rep::I32 {
                            f.insts[i.idx()].op = Op::IToL;
                            f.args_mut(i)[0] = y;
                            changed = true;
                            continue;
                        }
                        if rep == Rep::F64
                            && f.ty(y).rep == Rep::I32
                            && f.ty(x).within(TypeSet::FLOAT)
                        {
                            continue;
                        }
                    }
                    if let Some(k) = konst_val(f, x) {
                        let nk = match rep {
                            Rep::I32 => k.get_small().map(Op::KI32),
                            Rep::I64 => k.get_integer().map(Op::KI64),
                            Rep::F64 => k.get_float().map(|x| Op::KF64(x.to_bits())),
                            _ => None,
                        };
                        if let Some(nk) = nk {
                            f.insts[i.idx()].op = nk;
                            f.insts[i.idx()].an = 0;
                            changed = true;
                        }
                    }
                }
                Op::Box => {
                    let x = args[0];
                    if let Some(Op::Unbox(_)) = f.def_op(x) {
                        let y = f.args(f.def_inst(x).unwrap())[0];
                        map[res(f).idx()] = y;
                        changed = true;
                        continue;
                    }
                    let k = match f.def_op(x) {
                        Some(Op::KI32(n)) if f.ty(x).rep == Rep::I32 => {
                            Some(Value::small(n).to_raw())
                        }
                        Some(Op::KF64(bits)) => Some(Value::float(f64::from_bits(bits)).to_raw()),
                        Some(Op::KI64(n)) if i32::try_from(n).is_ok() => {
                            Some(Value::small(n as i32).to_raw())
                        }
                        _ => None,
                    };
                    if let Some(k) = k {
                        f.insts[i.idx()].op = Op::KVal(k);
                        f.insts[i.idx()].an = 0;
                        changed = true;
                    }
                }
                op if fold_arith(f, i, op, &args) => changed = true,
                Op::ICmp(cc) | Op::LCmp(cc) => {
                    if let (Some(x), Some(y)) = (konst_int(f, args[0]), konst_int(f, args[1])) {
                        set_b1(f, i, cc.eval_i(x, y));
                        changed = true;
                    }
                }
                Op::FCmp(cc) => {
                    if let (Some(x), Some(y)) = (konst_f64(f, args[0]), konst_f64(f, args[1])) {
                        let r = match cc {
                            Cc::Eq => x == y,
                            Cc::Ne => x != y,
                            Cc::Lt => x < y,
                            Cc::Le => x <= y,
                            Cc::Gt => x > y,
                            Cc::Ge => x >= y,
                        };
                        set_b1(f, i, r);
                        changed = true;
                    }
                }
                Op::IsType(set) => {
                    let t = f.ty(args[0]);
                    if !t.set.is_empty() && t.within(set) {
                        set_b1(f, i, true);
                        changed = true;
                    } else if !t.set.intersects(set) && !t.set.is_empty() {
                        set_b1(f, i, false);
                        changed = true;
                    }
                }
                Op::IsFalsy => {
                    let t = f.ty(args[0]);
                    if !t.set.is_empty() && !t.set.intersects(TypeSet::FALSY) {
                        set_b1(f, i, false);
                        changed = true;
                    } else if !t.set.is_empty() && t.within(TypeSet::FALSY) {
                        set_b1(f, i, true);
                        changed = true;
                    }
                }
                Op::GuardTrue | Op::GuardFalse => {
                    if let Some(c) = konst_b1(f, args[0])
                        && c == (op == Op::GuardTrue)
                    {
                        remove.push(i);
                        changed = true;
                    }
                }
                Op::Select => {
                    if let Some(c) = konst_b1(f, args[0]) {
                        map[res(f).idx()] = if c { args[1] } else { args[2] };
                        changed = true;
                    }
                }
                Op::Br => {
                    if let Some(c) = konst_b1(f, args[0]) {
                        let keep = if c { 0 } else { 1 };
                        let e = f.edges(i)[keep].clone();
                        f.insts[i.idx()].op = Op::Jump;
                        f.insts[i.idx()].an = 0;
                        f.set_edges(i, vec![e]);
                        changed = true;
                    }
                }
                _ => {}
            }
        }
    }
    if !remove.is_empty() {
        for b in 0..f.blocks.len() {
            f.blocks[b].insts.retain(|i| !remove.contains(i));
        }
    }
    f.apply_replacements(&mut map);
    // An exit boxes what it writes: snapshots take values unboxed.
    for si in 0..f.snaps.len() {
        for k in 0..f.snaps[si].entries.len() {
            let v = f.snaps[si].entries[k].1;
            if let Some(Op::Box) = f.def_op(v) {
                let x = f.args(f.def_inst(v).unwrap())[0];
                f.snaps[si].entries[k].1 = x;
            }
        }
    }
    if changed {
        mark_unreachable(f);
    }
    changed
}

/// Fold integer and float arithmetic on constants into a constant, where
/// the operation would not exit.
fn fold_arith(f: &mut Func<'_>, i: Inst, op: Op, args: &[Val]) -> bool {
    use Op::*;
    let int = |k: usize| args.get(k).and_then(|&a| konst_int(f, a));
    let flt = |k: usize| args.get(k).and_then(|&a| konst_f64(f, a));
    let small = |n: i64| i32::try_from(n).ok().map(KI32);
    let folded = match op {
        IAdd | IAddNo => int(0).zip(int(1)).and_then(|(x, y)| small(x + y)),
        ISub | ISubNo => int(0).zip(int(1)).and_then(|(x, y)| small(x - y)),
        IMul | IMulNo => int(0).zip(int(1)).and_then(|(x, y)| small(x * y)),
        IAnd => int(0).zip(int(1)).and_then(|(x, y)| small(x & y)),
        IOr => int(0).zip(int(1)).and_then(|(x, y)| small(x | y)),
        IXor => int(0).zip(int(1)).and_then(|(x, y)| small(x ^ y)),
        INeg => int(0).and_then(|x| small(-x)),
        IDivFloor => int(0)
            .zip(int(1))
            .filter(|&(_, y)| y != 0)
            .and_then(|(x, y)| {
                let q = x / y;
                small(if x % y != 0 && (x < 0) != (y < 0) {
                    q - 1
                } else {
                    q
                })
            }),
        IModFloor => int(0)
            .zip(int(1))
            .filter(|&(_, y)| y != 0)
            .and_then(|(x, y)| {
                let r = x % y;
                small(if r != 0 && (r < 0) != (y < 0) {
                    r + y
                } else {
                    r
                })
            }),
        LAdd => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_add(y))),
        LSub => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_sub(y))),
        LMul => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_mul(y))),
        LAnd => int(0).zip(int(1)).map(|(x, y)| KI64(x & y)),
        LOr => int(0).zip(int(1)).map(|(x, y)| KI64(x | y)),
        LXor => int(0).zip(int(1)).map(|(x, y)| KI64(x ^ y)),
        IToL => int(0).map(KI64),
        LToI => int(0).and_then(small),
        IToF | LToF => int(0).map(|x| KF64((x as f64).to_bits())),
        FAdd => flt(0).zip(flt(1)).map(|(x, y)| KF64((x + y).to_bits())),
        FSub => flt(0).zip(flt(1)).map(|(x, y)| KF64((x - y).to_bits())),
        FMul => flt(0).zip(flt(1)).map(|(x, y)| KF64((x * y).to_bits())),
        _ => None,
    };
    let Some(k) = folded else {
        return false;
    };
    let d = &mut f.insts[i.idx()];
    d.op = k;
    d.an = 0;
    d.snap = crate::jit::ir::NO_SNAP;
    let r = f.result(i);
    f.vals[r.idx()].ty = k.result_ty(&[]);
    true
}

fn set_b1(f: &mut Func<'_>, i: Inst, v: bool) {
    f.insts[i.idx()].op = Op::KB1(v);
    f.insts[i.idx()].an = 0;
}

/// Mark blocks no edge from the entry reaches.
pub(crate) fn mark_unreachable(f: &mut Func<'_>) {
    let reach = f.rpo();
    let mut live = vec![false; f.blocks.len()];
    for b in reach {
        live[b.idx()] = true;
    }
    for (b, l) in live.iter().enumerate() {
        if !l {
            f.blocks[b].dead = true;
        }
    }
}
