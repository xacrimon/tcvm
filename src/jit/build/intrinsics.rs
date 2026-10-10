//! Builtins compiled inline (`jit-design.md` 12): a CALL whose callee is,
//! at compile time, a builtin with an intrinsic guards the callee by
//! identity and computes the result in place; anything the inline code does
//! not cover exits before the CALL, and the interpreter runs the builtin.

use crate::env::function::Function;
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::jit::build::Builder;
use crate::jit::build::emit::{arm_end, branch_arm, enter_join};
use crate::jit::feedback as fb;
use crate::jit::ir::Val;
use crate::jit::ir::ops::{Cc, ExitTag, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::vm::abi::Handler;
use crate::vm::ff;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Intrinsic {
    Sqrt,
    Sin,
    Cos,
    Floor,
    Ceil,
    Abs,
    Assert,
    Pairs,
    Ipairs,
}

/// The intrinsic of a builtin, known by its fast entry (each builtin with
/// one has its own).
fn intrinsic_of(f: Function<'_>) -> Option<Intrinsic> {
    let entry = f.as_native()?.entry as usize;
    let table: [(Handler, Intrinsic); 9] = [
        (ff::ff_sqrt, Intrinsic::Sqrt),
        (ff::ff_sin, Intrinsic::Sin),
        (ff::ff_cos, Intrinsic::Cos),
        (ff::ff_floor, Intrinsic::Floor),
        (ff::ff_ceil, Intrinsic::Ceil),
        (ff::ff_abs, Intrinsic::Abs),
        (ff::ff_assert, Intrinsic::Assert),
        (crate::vm::native::ff_pairs, Intrinsic::Pairs),
        (crate::vm::native::ff_ipairs, Intrinsic::Ipairs),
    ];
    table
        .iter()
        .find(|(h, _)| *h as usize == entry)
        .map(|&(_, i)| i)
}

/// The value `v` has now, when it is a load the builder can redo: a
/// constant, an upvalue of the closure being compiled, or a field read of
/// such a table. The region guards what it uses of it.
fn value_now<'gc>(b: &Builder<'_, 'gc>, v: Val) -> Option<Value<'gc>> {
    let i = b.f.def_inst(v)?;
    match b.f.op(i) {
        Op::KObj(p) => Some(b.f.pool[p as usize]),
        Op::Guard(_) => value_now(b, b.f.args(i)[0]),
        Op::UpvalValue(k) => Some(unsafe { (*b.closure.upvalue_ptr().add(k as usize)).value }),
        Op::CellLoad => {
            let c = b.f.def_inst(b.f.args(i)[0])?;
            let Op::UpvalCell(k) = b.f.op(c) else {
                return None;
            };
            Some(unsafe { (*b.closure.upvalue_ptr().add(k as usize)).cell }.get())
        }
        _ => {
            let &(recv, key) = b.field_loads.get(&v)?;
            Some(value_now(b, recv)?.get_table()?.raw_get(key))
        }
    }
}

/// Compile the CALL at `pc` as its callee's intrinsic, if it has one that
/// covers the call's shape; false when the CALL compiles as a call.
pub(super) fn emit(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, callee: Val) -> bool {
    // A tail call returns the one result.
    let (a, nb, c) = if insn.op() == crate::instruction::Op::TAILCALL {
        (insn.a(), insn.b(), 2)
    } else {
        (insn.a(), insn.b(), insn.c())
    };
    if nb == 0 {
        return false;
    }
    let Some(f) = value_now(b, callee).and_then(|v| v.get_function()) else {
        return false;
    };
    let Some(intr) = intrinsic_of(f) else {
        return false;
    };
    let nargs = nb - 1;
    let iterators = matches!(intr, Intrinsic::Pairs | Intrinsic::Ipairs);
    // An exit saw an argument the inline code does not take.
    let seen = b.feedback(pc) & fb::KINDS;
    let numbers = !matches!(intr, Intrinsic::Assert) && !iterators;
    if numbers && seen & !(fb::SMALL | fb::FLOAT) != 0 {
        return false;
    }
    // A tail call or MULTRET wants all results; the iterators make up to
    // four, the others one.
    let tail = insn.op() == crate::instruction::Op::TAILCALL;
    if c == 0
        || (c > 2 && !iterators)
        || (iterators && (tail || c > 5))
        || (intr != Intrinsic::Assert && nargs != 1)
        || nargs == 0
    {
        return false;
    }
    let k = b.konst(Value::function(f));
    let cv = b.boxed(callee);
    b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Type);
    let x = b.reg(a + 4);
    if iterators {
        let x = b.boxed(x);
        let (iter, control) = if intr == Intrinsic::Pairs {
            // A table without `__pairs` (`ff_pairs`).
            let t = b.guard(x, TypeSet::TAB);
            let mm = b.ins(
                Op::Helper(HelperId::Mm(crate::env::MetamethodBits::PAIRS.bits())),
                &[t, t],
            );
            let none = b.ins(Op::IsType(TypeSet::NIL), &[mm]);
            let s = b.snap_before();
            b.ins_snap(Op::GuardTrue, &[none], s, ExitTag::Slow);
            (b.ctx.next_fn(), Value::nil())
        } else {
            (b.ctx.ipairs_iter(), Value::small(0))
        };
        let results = [
            b.konst(Value::function(iter)),
            x,
            b.konst(control),
            b.konst(Value::nil()),
        ];
        for (k, &r) in results.iter().take(c as usize - 1).enumerate() {
            b.set(a + k as u8, r);
        }
        return true;
    }
    let r = match intr {
        Intrinsic::Sqrt | Intrinsic::Sin | Intrinsic::Cos => {
            let x = number_f64(b, x);
            match intr {
                Intrinsic::Sqrt => b.ins(Op::FSqrt, &[x]),
                Intrinsic::Sin => b.ins(Op::Helper(HelperId::Sin), &[x]),
                _ => b.ins(Op::Helper(HelperId::Cos), &[x]),
            }
        }
        Intrinsic::Floor | Intrinsic::Ceil => {
            let ty = b.ty(x);
            let int = ty.rep == Rep::I32
                || ty.rep == Rep::I64
                || (!ty.set.is_empty() && ty.within(TypeSet::INT));
            if int {
                // An integer is its own floor.
                x
            } else {
                let x = number_f64(b, x);
                let op = if intr == Intrinsic::Floor {
                    Op::FFloor
                } else {
                    Op::FCeil
                };
                let y = b.ins(op, &[x]);
                // Outside i32 the builtin makes a big integer or a float.
                let i = b.ins_exit(Op::FToIExact, &[y], ExitTag::Slow);
                b.f.result(i)
            }
        }
        Intrinsic::Abs => {
            let ty = b.ty(x);
            let known = |set| !ty.set.is_empty() && ty.within(set);
            if ty.rep == Rep::F64 || known(TypeSet::FLOAT) {
                let x = b.as_f64(x);
                b.ins(Op::FAbs, &[x])
            } else if ty.rep == Rep::I32 || known(TypeSet::SMALL) {
                let x = b.as_i32(x);
                abs_i32(b, x)
            } else {
                // A float, a small integer, else the builtin.
                let xv = b.boxed(x);
                let join = b.new_block();
                let is_float = b.ins(Op::IsType(TypeSet::FLOAT), &[xv]);
                let rest = branch_arm(b, is_float);
                let f = b.ins(Op::Unbox(Rep::F64), &[xv]);
                let r = b.ins(Op::FAbs, &[f]);
                arm_end(b, join, a, r);
                b.switch_to(rest);
                let is_small = b.ins(Op::IsType(TypeSet::SMALL), &[xv]);
                let other = branch_arm(b, is_small);
                let i = b.ins(Op::Unbox(Rep::I32), &[xv]);
                let r = abs_i32(b, i);
                arm_end(b, join, a, r);
                b.switch_to(other);
                b.deopt(ExitTag::Slow);
                enter_join(b, join);
                return true;
            }
        }
        Intrinsic::Pairs | Intrinsic::Ipairs => unreachable!("returned above"),
        Intrinsic::Assert => {
            // A falsy one raises in the builtin.
            let xv = b.boxed(x);
            let falsy = b.ins(Op::IsFalsy, &[xv]);
            let s = b.snap_before();
            b.ins_snap(Op::GuardFalse, &[falsy], s, ExitTag::Slow);
            xv
        }
    };
    if c == 2 {
        b.set(a, r);
    }
    true
}

/// `|x|` of an i32; `abs(i32::MIN)` leaves i32 and exits to the builtin.
fn abs_i32(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let i = b.ins_exit(Op::INeg, &[x], ExitTag::Slow);
    let neg = b.f.result(i);
    let zero = b.ki32(0);
    let below = b.ins(Op::ICmp(Cc::Lt), &[x, zero]);
    b.ins(Op::Select, &[below, neg, x])
}

/// The f64 of a number argument: typed ones converted, others a small
/// integer or a float, else an exit to the builtin (a big integer, an
/// error).
fn number_f64(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let ty = b.ty(x);
    if matches!(ty.rep, Rep::F64 | Rep::I32 | Rep::I64)
        || (!ty.set.is_empty() && (ty.within(TypeSet::FLOAT) || ty.within(TypeSet::SMALL)))
    {
        return b.num_f64(x);
    }
    let xv = b.boxed(x);
    let i = b.ins_exit(Op::ToF64, &[xv], ExitTag::Type);
    b.f.result(i)
}
