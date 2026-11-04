//! Local folds: redundant guards and conversions, constant compares and
//! branches, and the strength reductions of arithmetic.

use crate::env::value::Value;
use crate::jit::ir::ops::{Cc, ExitTag, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, BlockCall, Func, Inst, Val};
use crate::vm::num::shift_left;

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
    let mut removed = vec![false; f.insts.len()];
    let mut any_removed = false;
    let made = f.insts.len();
    let mut remove = |i: Inst| {
        removed[i.idx()] = true;
        any_removed = true;
    };
    for b in 0..f.blocks.len() {
        if f.blocks[b].dead {
            continue;
        }
        let b = Block(b as u32);
        for k in 0..f.blocks[b.idx()].insts.len() {
            let i = f.insts_of(b)[k];
            let op = f.op(i);
            let mut args = [Val(0); 3];
            let n = f.args(i).len().min(3);
            args[..n].copy_from_slice(&f.args(i)[..n]);
            let args = &args[..n];
            let res = |f: &Func<'_>| f.result(i);
            if fold_arith(f, i, op, args) {
                changed = true;
                continue;
            }
            // What a reduction makes before `i` waits for the next round.
            if i.idx() < made {
                match reduce(f, b, k, i, op, args) {
                    Reduced::No => {}
                    Reduced::To(v) => {
                        map[res(f).idx()] = v;
                        remove(i);
                        changed = true;
                        continue;
                    }
                    Reduced::Changed => {
                        changed = true;
                        continue;
                    }
                }
            }
            match op {
                Op::Guard(set) => {
                    let t = f.ty(args[0]);
                    if t.within(set) && !t.set.is_empty() {
                        map[res(f).idx()] = args[0];
                        remove(i);
                        changed = true;
                    }
                }
                // Only kinds in `set` convert: the others must still exit.
                Op::ToF64(set) => {
                    let x = args[0];
                    if let Some(Op::Box) = f.def_op(x) {
                        let y = f.args(f.def_inst(x).unwrap())[0];
                        match f.ty(y).rep {
                            Rep::F64 if set.contains(TypeSet::FLOAT) => {
                                map[res(f).idx()] = y;
                                remove(i);
                                changed = true;
                                continue;
                            }
                            Rep::I32 if set.contains(TypeSet::SMALL) => {
                                f.insts[i.idx()].op = Op::IToF;
                                f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                                f.args_mut(i)[0] = y;
                                changed = true;
                                continue;
                            }
                            Rep::I64 if set.contains(TypeSet::INT) => {
                                f.insts[i.idx()].op = Op::LToF;
                                f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                                f.args_mut(i)[0] = y;
                                changed = true;
                                continue;
                            }
                            _ => {}
                        }
                    }
                    if let Some(k) = konst_val(f, x)
                        && let Some(n) = if let Some(n) = k.get_small() {
                            set.contains(TypeSet::SMALL).then_some(f64::from(n))
                        } else if let Some(x) = k.get_float() {
                            set.contains(TypeSet::FLOAT).then_some(x)
                        } else {
                            let n = k.get_integer().filter(|_| set.contains(TypeSet::BIGINT));
                            n.map(|n| n as f64)
                        }
                    {
                        f.insts[i.idx()].op = Op::KF64(n.to_bits());
                        f.insts[i.idx()].an = 0;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        changed = true;
                        continue;
                    }
                    let t = f.ty(x);
                    if t.within(TypeSet::FLOAT) && !t.set.is_empty() && set.contains(TypeSet::FLOAT)
                    {
                        f.insts[i.idx()].op = Op::Unbox(Rep::F64);
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        changed = true;
                    } else if t.rep == Rep::Val
                        && t.within(TypeSet::SMALL)
                        && !t.set.is_empty()
                        && set.contains(TypeSet::SMALL)
                    {
                        // The block's last instruction waits for the next round.
                        let u = f.make_inst(Op::Unbox(Rep::I32), &[x], None, ExitTag::Type);
                        f.insert(b, k, u);
                        let u = f.result(u);
                        f.insts[i.idx()].op = Op::IToF;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        f.args_mut(i)[0] = u;
                        changed = true;
                    }
                }
                // Only kinds in `set` convert: the others must still exit.
                Op::ToPos(set) => {
                    let x = args[0];
                    if let Some(Op::Box) = f.def_op(x)
                        && set.contains(TypeSet::SMALL)
                        && f.ty(f.args(f.def_inst(x).unwrap())[0]).rep == Rep::I32
                    {
                        map[res(f).idx()] = f.args(f.def_inst(x).unwrap())[0];
                        remove(i);
                        changed = true;
                        continue;
                    }
                    if let Some(k) = konst_val(f, x)
                        && let Some(n) = k.get_integer()
                        && set.contains(if k.get_small().is_some() {
                            TypeSet::SMALL
                        } else {
                            TypeSet::BIGINT
                        })
                    {
                        let n = n.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                        f.insts[i.idx()].op = Op::KI32(n);
                        f.insts[i.idx()].an = 0;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        changed = true;
                        continue;
                    }
                    let t = f.ty(x);
                    if t.rep == Rep::Val
                        && !t.set.is_empty()
                        && t.within(TypeSet::SMALL)
                        && set.contains(TypeSet::SMALL)
                    {
                        f.insts[i.idx()].op = Op::Unbox(Rep::I32);
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        changed = true;
                    }
                }
                Op::FToIExact => {
                    let x = args[0];
                    if let Some(Op::IToF) = f.def_op(x) {
                        map[res(f).idx()] = f.args(f.def_inst(x).unwrap())[0];
                        remove(i);
                        changed = true;
                    } else if let Some(c) = konst_f64(f, x)
                        && c as i32 as f64 == c
                    {
                        f.insts[i.idx()].op = Op::KI32(c as i32);
                        f.insts[i.idx()].an = 0;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        changed = true;
                    }
                }
                Op::FToLExact => {
                    let x = args[0];
                    if let Some(Op::IToF) = f.def_op(x) {
                        f.insts[i.idx()].op = Op::IToL;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
                        f.args_mut(i)[0] = f.args(f.def_inst(x).unwrap())[0];
                        changed = true;
                    } else if let Some(c) = konst_f64(f, x)
                        && let Some(n) = crate::vm::num::exact_float_to_int(c)
                    {
                        f.insts[i.idx()].op = Op::KI64(n);
                        f.insts[i.idx()].an = 0;
                        f.insts[i.idx()].snap = crate::jit::ir::NO_SNAP;
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
                Op::ICmp(cc) | Op::LCmp(cc) => {
                    if let (Some(x), Some(y)) = (konst_int(f, args[0]), konst_int(f, args[1])) {
                        set_b1(f, i, cc.eval_i(x, y));
                        changed = true;
                    }
                }
                Op::FCmp(cc) => {
                    // Integers compare as they do converted, and an integral
                    // constant in range converts back exactly.
                    let int_of = |f: &Func<'_>, v: Val| match f.def_op(v) {
                        Some(Op::IToF) => Some(Ok(f.args(f.def_inst(v).unwrap())[0])),
                        Some(Op::KF64(bits)) => {
                            let c = f64::from_bits(bits);
                            (c == c as i32 as f64).then_some(Err(c as i32))
                        }
                        _ => None,
                    };
                    if let (Some(x), Some(y)) = (int_of(f, args[0]), int_of(f, args[1]))
                        && (x.is_ok() || y.is_ok())
                    {
                        let mut ops = [Val(0); 2];
                        let mut at = k;
                        for (n, s) in [x, y].into_iter().enumerate() {
                            ops[n] = match s {
                                Ok(v) => v,
                                Err(c) => {
                                    let ki = f.make_inst(Op::KI32(c), &[], None, ExitTag::Type);
                                    f.insert(b, at, ki);
                                    at += 1;
                                    f.result(ki)
                                }
                            };
                        }
                        f.insts[i.idx()].op = Op::ICmp(cc);
                        f.args_mut(i).copy_from_slice(&ops);
                        changed = true;
                        continue;
                    }
                    if let (Some(x), Some(y)) = (konst_f64(f, args[0]), konst_f64(f, args[1])) {
                        let r = match cc {
                            Cc::Eq => x == y,
                            Cc::Ne => x != y,
                            Cc::Lt => x < y,
                            Cc::Le => x <= y,
                            Cc::Gt => x > y,
                            Cc::Ge => x >= y,
                            Cc::Lo | Cc::Hi => unreachable!("an unsigned float compare"),
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
                Op::GuardSame if args[0] == args[1] => {
                    remove(i);
                    changed = true;
                }
                Op::SameBits if args[0] == args[1] => {
                    set_b1(f, i, true);
                    changed = true;
                }
                Op::GuardTrue | Op::GuardFalse => {
                    if let Some(c) = konst_b1(f, args[0])
                        && c == (op == Op::GuardTrue)
                    {
                        remove(i);
                        changed = true;
                    }
                }
                Op::Barrier
                    if args.len() == 2
                        && !f.ty(args[1]).set.is_empty()
                        && !f.ty(args[1]).set.intersects(TypeSet::HEAP) =>
                {
                    remove(i);
                    changed = true;
                }
                Op::GuardMiss
                    if konst_b1(f, args[1]) == Some(false)
                        || !f.ty(args[0]).set.intersects(TypeSet::NIL) =>
                {
                    remove(i);
                    changed = true;
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
                        let e = f.edges(i)[keep];
                        f.insts[i.idx()].op = Op::Jump;
                        f.insts[i.idx()].an = 0;
                        f.set_edges(i, &[e]);
                        changed = true;
                    }
                }
                _ => {}
            }
        }
    }
    if any_removed {
        for b in 0..f.blocks.len() {
            f.retain(Block(b as u32), |_, i| {
                !removed.get(i.idx()).is_some_and(|&r| r)
            });
        }
    }
    // Instructions made here have values past the map.
    while map.len() < f.vals.len() {
        map.push(Val(map.len() as u32));
    }
    f.apply_replacements(&mut map);
    // An exit boxes what it writes: snapshots take values unboxed.
    for k in 0..f.snap_pool.len() {
        let v = f.snap_pool[k].1;
        if let Some(Op::Box) = f.def_op(v) {
            f.snap_pool[k].1 = f.args(f.def_inst(v).unwrap())[0];
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
        IRem => int(0)
            .zip(int(1))
            .filter(|&(_, y)| y != 0)
            .and_then(|(x, y)| small((x as i32).wrapping_rem(y as i32) as i64)),
        LRem => int(0)
            .zip(int(1))
            .filter(|&(_, y)| y != 0)
            .map(|(x, y)| KI64(x.wrapping_rem(y))),
        LAdd => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_add(y))),
        LSub => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_sub(y))),
        LMul => int(0).zip(int(1)).map(|(x, y)| KI64(x.wrapping_mul(y))),
        LAnd => int(0).zip(int(1)).map(|(x, y)| KI64(x & y)),
        LOr => int(0).zip(int(1)).map(|(x, y)| KI64(x | y)),
        LXor => int(0).zip(int(1)).map(|(x, y)| KI64(x ^ y)),
        IShl => int(0)
            .zip(int(1))
            .and_then(|(x, y)| small(shift_left(x, y))),
        IShr => int(0)
            .zip(int(1))
            .and_then(|(x, y)| small(shift_left(x, y.wrapping_neg()))),
        IAsr => int(0)
            .zip(int(1))
            .map(|(x, y)| KI32((x as i32) >> (y & 31))),
        IUMulShr(s) => int(0)
            .zip(int(1))
            .map(|(x, y)| KI32(((x as u32 as u64 * y as u32 as u64) >> s) as u32 as i32)),
        INot => int(0).map(|x| KI32(!(x as i32))),
        LNeg => int(0).map(|x| KI64(x.wrapping_neg())),
        LNot => int(0).map(|x| KI64(!x)),
        LShl => int(0).zip(int(1)).map(|(x, y)| KI64(shift_left(x, y))),
        LShr => int(0)
            .zip(int(1))
            .map(|(x, y)| KI64(shift_left(x, y.wrapping_neg()))),
        LAsr => int(0).zip(int(1)).map(|(x, y)| KI64(x >> (y & 63))),
        LUDiv => int(0)
            .zip(int(1))
            .filter(|&(_, y)| y != 0)
            .map(|(x, y)| KI64((x as u64 / y as u64) as i64)),
        LDivFloor => int(0).zip(int(1)).filter(|&(_, y)| y != 0).map(|(x, y)| {
            let q = x.wrapping_div(y);
            KI64(if x.wrapping_rem(y) != 0 && (x ^ y) < 0 {
                q - 1
            } else {
                q
            })
        }),
        LModFloor => int(0).zip(int(1)).filter(|&(_, y)| y != 0).map(|(x, y)| {
            let r = x.wrapping_rem(y);
            KI64(if r != 0 && (r ^ y) < 0 { r + y } else { r })
        }),
        IToL => int(0).map(KI64),
        LToI => int(0).and_then(small),
        IToF | LToF => int(0).map(|x| KF64((x as f64).to_bits())),
        FAdd => flt(0).zip(flt(1)).map(|(x, y)| KF64((x + y).to_bits())),
        FSub => flt(0).zip(flt(1)).map(|(x, y)| KF64((x - y).to_bits())),
        FMul => flt(0).zip(flt(1)).map(|(x, y)| KF64((x * y).to_bits())),
        FDiv => flt(0).zip(flt(1)).map(|(x, y)| KF64((x / y).to_bits())),
        FIDiv => flt(0)
            .zip(flt(1))
            .map(|(x, y)| KF64((x / y).floor().to_bits())),
        FNeg => flt(0).map(|x| KF64((-x).to_bits())),
        FAbs => flt(0).map(|x| KF64(x.abs().to_bits())),
        FSqrt => flt(0).map(|x| KF64(x.sqrt().to_bits())),
        FFloor => flt(0).map(|x| KF64(x.floor().to_bits())),
        FCeil => flt(0).map(|x| KF64(x.ceil().to_bits())),
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

/// What a reduction did with an instruction.
enum Reduced {
    No,
    /// Its result is this value.
    To(Val),
    /// It became another instruction, maybe after new ones before it.
    Changed,
}

/// Instructions made before one being reduced, in order.
struct Before {
    b: Block,
    at: usize,
}

impl Before {
    fn ins(&mut self, f: &mut Func<'_>, op: Op, args: &[Val]) -> Val {
        let u = f.make_inst(op, args, None, ExitTag::Type);
        f.insert(self.b, self.at, u);
        self.at += 1;
        f.result(u)
    }
}

/// Make `i` the instruction `op` of `args`, no more than it had; it keeps
/// its exit where `op` may take one.
fn become_op(f: &mut Func<'_>, i: Inst, op: Op, args: &[Val]) {
    debug_assert!(args.len() <= f.insts[i.idx()].an as usize);
    let d = &mut f.insts[i.idx()];
    d.op = op;
    d.an = args.len() as u16;
    if !op.may_deopt() {
        d.snap = crate::jit::ir::NO_SNAP;
    }
    f.args_mut(i).copy_from_slice(args);
    let r = f.result(i);
    f.vals[r.idx()].ty = f.result_ty(op, args);
}

/// `n` as a constant of the integer operation `op`'s width.
fn int_k(op: Op, n: i64) -> Op {
    if op.result_ty(&[]).rep == Rep::I64 {
        Op::KI64(n)
    } else {
        Op::KI32(n as i32)
    }
}

/// The multiplier and shift dividing an `n` in `0..2^31` by `c` (3 or
/// more, not a power of two) as `(n * m) >> s` (Granlund and Montgomery,
/// 1994, theorem 4.2, with N = 31).
fn magic(c: i64) -> (u32, u8) {
    let l = 64 - (c - 1).leading_zeros();
    let m = (1u64 << (31 + l)) / c as u64 + 1;
    debug_assert!(m < 1 << 32);
    (m as u32, (31 + l) as u8)
}

/// Strength reductions and identities of arithmetic, as V8's machine
/// optimization reducer and JSC B3's strength reduction have them: each
/// gives the same bits for every operand, NaNs included, and exits where
/// the original would.
fn reduce(f: &mut Func<'_>, b: Block, k: usize, i: Inst, op: Op, args: &[Val]) -> Reduced {
    use Op::*;
    use Reduced::*;
    let n = args.len();
    match op {
        // Constants go right, where the rules and lowering look for them.
        IAdd | IAddNo | IMul | IMulNo | IAnd | IOr | IXor | LAdd | LMul | LAnd | LOr | LXor
            if konst_int(f, args[0]).is_some() && konst_int(f, args[1]).is_none() =>
        {
            f.args_mut(i).swap(0, 1);
            return Changed;
        }
        ICmp(cc) | LCmp(cc) => {
            if args[0] == args[1] {
                become_op(f, i, KB1(cc.eval_i(0, 0)), &[]);
                return Changed;
            }
            if konst_int(f, args[0]).is_some() && konst_int(f, args[1]).is_none() {
                let swapped = if op == ICmp(cc) {
                    ICmp(cc.swap())
                } else {
                    LCmp(cc.swap())
                };
                become_op(f, i, swapped, &[args[1], args[0]]);
                return Changed;
            }
            return No;
        }
        Select if args[1] == args[2] => return To(args[1]),
        Select => {
            // `c ? k : w` where `c` is `v == k` picks `v`, which needs no `k`
            // made; `c ? w : k` where `c` is `v != k` likewise.
            if let Some(d) = f.def_inst(args[0]) {
                let (top, ta) = (f.op(d), f.args(d));
                let word = |f: &Func<'_>, v: Val| match f.def_op(v) {
                    Some(KVal(bits)) => Some(bits),
                    _ => None,
                };
                let single = |set: TypeSet| {
                    [
                        (TypeSet::NIL, crate::env::value::NIL),
                        (TypeSet::FALSE, crate::env::value::FALSE),
                        (TypeSet::TRUE, crate::env::value::TRUE),
                    ]
                    .into_iter()
                    .find(|&(s, _)| s == set)
                    .map(|(_, w)| w)
                };
                let same = |f: &Func<'_>, a: Val, b: Val| {
                    a == b || word(f, a).is_some_and(|w| word(f, b) == Some(w))
                };
                let tested = match top {
                    IsType(set) => single(set)
                        .filter(|&w| word(f, args[1]) == Some(w))
                        .map(|_| (ta[0], 1)),
                    SameBits | ICmp(Cc::Eq) | LCmp(Cc::Eq) => (0..2)
                        .find(|&k| same(f, ta[k], args[1]))
                        .map(|k| (ta[1 - k], 1)),
                    ICmp(Cc::Ne) | LCmp(Cc::Ne) => (0..2)
                        .find(|&k| same(f, ta[k], args[2]))
                        .map(|k| (ta[1 - k], 2)),
                    _ => None,
                };
                if let Some((v, at)) = tested
                    && f.ty(v).rep == f.ty(args[at]).rep
                {
                    f.args_mut(i)[at] = v;
                    return Changed;
                }
            }
            return No;
        }
        _ => {}
    }
    let x = args.first().copied().unwrap_or(Val(0));
    let inner = |f: &Func<'_>| {
        let d = f.def_inst(x)?;
        Some((f.op(d), *f.args(d).first()?))
    };
    // One operand.
    if n == 1 {
        match (op, inner(f)) {
            (INot | LNot | LNeg | FNeg, Some((o, z))) if o == op => return To(z),
            (FAbs, Some((FAbs, _))) => return To(x),
            (FAbs, Some((FNeg, z))) => {
                become_op(f, i, FAbs, &[z]);
                return Changed;
            }
            (FFloor | FCeil, Some((IToF | LToF | FFloor | FCeil, _))) => return To(x),
            (LToI, Some((IToL, z))) => return To(z),
            (LToF, Some((IToL, z))) => {
                become_op(f, i, IToF, &[z]);
                return Changed;
            }
            _ => {}
        }
        return No;
    }
    if n != 2 {
        return No;
    }
    let y = args[1];
    if x == y {
        match op {
            IAnd | LAnd | IOr | LOr => return To(x),
            ISub | ISubNo | LSub | IXor | LXor => {
                become_op(f, i, int_k(op, 0), &[]);
                return Changed;
            }
            _ => {}
        }
    }
    if matches!(op, FDiv | FIDiv)
        && let Some(c) = konst_f64(f, y)
        && c.is_normal()
        && c.to_bits() & ((1 << 52) - 1) == 0
        && (1.0 / c).is_normal()
    {
        // A power of two's reciprocal is exact, and so the product.
        let mut at = Before { b, at: k };
        let r = at.ins(f, KF64((1.0 / c).to_bits()), &[]);
        if op == FDiv {
            become_op(f, i, FMul, &[x, r]);
        } else {
            let q = at.ins(f, FMul, &[x, r]);
            become_op(f, i, FFloor, &[q]);
        }
        return Changed;
    }
    if op == FMul && konst_f64(f, y) == Some(2.0) {
        become_op(f, i, FAdd, &[x, x]);
        return Changed;
    }
    // Wrapping negations fold into the sum or difference.
    let neg = |f: &Func<'_>, v: Val| match f.def_inst(v) {
        Some(d) if f.op(d) == LNeg => Some(f.args(d)[0]),
        _ => None,
    };
    match (op, neg(f, x), neg(f, y)) {
        (LAdd, _, Some(z)) => {
            become_op(f, i, LSub, &[x, z]);
            return Changed;
        }
        (LAdd, Some(z), None) => {
            become_op(f, i, LSub, &[y, z]);
            return Changed;
        }
        (LSub, _, Some(z)) => {
            become_op(f, i, LAdd, &[x, z]);
            return Changed;
        }
        _ => {}
    }
    let Some(c) = konst_int(f, y) else {
        return No;
    };
    let mut at = Before { b, at: k };
    let pow2 = c > 1 && c & (c - 1) == 0;
    let log2 = c.trailing_zeros() as i64;
    let (to, with): (Op, &[Option<i64>]) = match (op, c) {
        (
            IAdd | IAddNo | LAdd | ISub | ISubNo | LSub | IOr | LOr | IXor | LXor | IShl | IShr
            | LShl | LShr,
            0,
        )
        | (IMul | IMulNo | LMul | IDivFloor | LDivFloor | LUDiv, 1)
        | (IAnd | LAnd, -1) => return To(x),
        (IAsr, c) if c & 31 == 0 => return To(x),
        (LAsr, c) if c & 63 == 0 => return To(x),
        (IMul | IMulNo | LMul | IAnd | LAnd, 0) | (IModFloor | LModFloor | IRem | LRem, 1 | -1) => {
            (int_k(op, 0), &[])
        }
        (LShl | LShr, c) if !(-63..=63).contains(&c) => (int_k(op, 0), &[]),
        (IOr | LOr, -1) => (int_k(op, -1), &[]),
        (IXor, -1) => (INot, &[None]),
        (LXor, -1) => (LNot, &[None]),
        // These exit where the operation would: `MIN * -1`, `MIN // -1`.
        (IMul | IDivFloor, -1) => (INeg, &[None]),
        (LMul | LDivFloor, -1) => (LNeg, &[None]),
        (IMul, 2) => (IAdd, &[None, None]),
        (IMulNo, 2) => (IAddNo, &[None, None]),
        (LMul, 2) => (LAdd, &[None, None]),
        (LMul, _) if pow2 => (LShl, &[None, Some(log2)]),
        (LSub, c) => (LAdd, &[None, Some(c.wrapping_neg())]),
        (LShl, c) if c < 0 => (LShr, &[None, Some(-c)]),
        (LShr, c) if c < 0 => (LShl, &[None, Some(-c)]),
        // Floor division by 2^k shifts, and its remainder is the low bits.
        (IDivFloor, _) if pow2 => (IAsr, &[None, Some(log2)]),
        (LDivFloor, _) if pow2 => (LAsr, &[None, Some(log2)]),
        (IModFloor | LModFloor, _) if pow2 => (
            if op == IModFloor { IAnd } else { LAnd },
            &[None, Some(c - 1)],
        ),
        (LUDiv, c) if (c as u64).is_power_of_two() => (LShr, &[None, Some(log2)]),
        // x // c for c > 0 is ~(~x // c) for x < 0, so with s the sign mask
        // ((x ^ s) // c) ^ s, the division of a non-negative i32 by magic.
        (IDivFloor | IModFloor, 3..) => {
            let (m, sh) = magic(c);
            let k31 = at.ins(f, KI32(31), &[]);
            let s = at.ins(f, IAsr, &[x, k31]);
            let u = at.ins(f, IXor, &[x, s]);
            let km = at.ins(f, KI32(m as i32), &[]);
            let q = at.ins(f, IUMulShr(sh), &[u, km]);
            if op == IDivFloor {
                become_op(f, i, IXor, &[q, s]);
            } else {
                // r = u - q * c is x mod c for x >= 0; below, the floor's
                // remainder is c - 1 - r, which (r ^ s) + (s & c) is.
                let kc = at.ins(f, KI32(c as i32), &[]);
                let p = at.ins(f, IMulNo, &[q, kc]);
                let r = at.ins(f, ISubNo, &[u, p]);
                let t = at.ins(f, IXor, &[r, s]);
                let a = at.ins(f, IAnd, &[s, kc]);
                become_op(f, i, IAddNo, &[t, a]);
            }
            return Changed;
        }
        (IAsr | LAsr | IAnd | LAnd | IOr | LOr | IXor | LXor | LAdd, c2) => {
            // Constant chains combine.
            let Some(d) = f.def_inst(x).filter(|&d| f.op(d) == op) else {
                return No;
            };
            let (z, c1) = (f.args(d)[0], f.args(d)[1]);
            let Some(c1) = konst_int(f, c1) else {
                return No;
            };
            let k = match op {
                IAsr => ((c1 & 31) + (c2 & 31)).min(31),
                LAsr => ((c1 & 63) + (c2 & 63)).min(63),
                IAnd | LAnd => c1 & c2,
                IOr | LOr => c1 | c2,
                IXor | LXor => c1 ^ c2,
                _ => c1.wrapping_add(c2),
            };
            let kv = at.ins(f, int_k(op, k), &[]);
            become_op(f, i, op, &[z, kv]);
            return Changed;
        }
        _ => return No,
    };
    let mut ops = [Val(0); 2];
    for (slot, w) in ops.iter_mut().zip(with) {
        *slot = match *w {
            None => x,
            Some(n) => at.ins(f, int_k(op, n), &[]),
        };
    }
    become_op(f, i, to, &ops[..with.len()]);
    Changed
}

/// Make jumps of the branches whose edges meet before anything runs, until
/// none is left: a branch made a jump may leave its block empty once `dce`
/// took its test. After the other passes, whose folds empty the arms.
pub(crate) fn fold_one_destination(f: &mut Func<'_>) {
    loop {
        let mut changed = false;
        for &b in &f.cfg().rpo {
            if let Some(t) = f.terminator(b)
                && f.op(t) == Op::Br
                && let Some(e) = one_destination(f, t)
            {
                f.insts[t.idx()].op = Op::Jump;
                f.insts[t.idx()].an = 0;
                f.set_edges(t, &[e]);
                changed = true;
            }
        }
        if !changed {
            break;
        }
        mark_unreachable(f);
        crate::jit::opt::dce(f);
    }
}

/// The edge both of `br`'s lead to through blocks that only jump on, when
/// they meet with the same arguments: the branch decides nothing.
fn one_destination(f: &Func<'_>, br: Inst) -> Option<BlockCall> {
    let end = |mut e: BlockCall| {
        // Bounded: empty blocks may jump in a cycle.
        for _ in 0..8 {
            match f.insts_of(e.target) {
                &[j] if e.args.is_empty()
                    && f.params(e.target).is_empty()
                    && f.op(j) == Op::Jump =>
                {
                    e = f.edges(j)[0]
                }
                _ => break,
            }
        }
        e
    };
    let (a, b) = (end(f.edges(br)[0]), end(f.edges(br)[1]));
    (a.target == b.target && f.vl(a.args) == f.vl(b.args)).then_some(a)
}

fn set_b1(f: &mut Func<'_>, i: Inst, v: bool) {
    f.insts[i.idx()].op = Op::KB1(v);
    f.insts[i.idx()].an = 0;
}

/// Mark blocks no edge from the entry reaches.
pub(crate) fn mark_unreachable(f: &mut Func<'_>) {
    let mut live = vec![false; f.blocks.len()];
    for &b in &f.cfg().rpo {
        live[b.idx()] = true;
    }
    for (b, l) in live.iter().enumerate() {
        if !l {
            f.blocks[b].dead = true;
        }
    }
}
