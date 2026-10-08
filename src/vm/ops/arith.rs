//! Arithmetic and bitwise operators: the guarded forms, the metamethod
//! forms, and one generic handler that executes every case and specializes
//! the site by the operand kinds it saw.

use crate::env::shape::{MetamethodBits, MmIndex};
use crate::env::table::Table;
use crate::env::value::Value;
use crate::instruction::{ArithKind, Family, Instruction, MISSES_TO_LOCK, Op};
use crate::vm::abi::{Slot, handler, handler_bits};
use crate::vm::frame::{self, HDR};
use crate::vm::num::{self, ArithOp, BitOp};
use crate::vm::ops::meta::{ret_store_a, stage_mm};
use crate::vm::unwind::OpError;

/// What a form stores, or that the operation left its fast case.
enum Out<'gc> {
    Value(Value<'gc>),
    /// A hardware result of canonical operands: stored unchecked.
    Float(f64),
    /// A libm result: checked for a NaN in box space.
    FloatChecked(f64),
    Miss,
}

#[inline(always)]
fn val(v: Option<Value<'_>>) -> Out<'_> {
    match v {
        Some(v) => Out::Value(v),
        None => Out::Miss,
    }
}

#[inline(always)]
fn float_small(l: &Value<'_>, r: &Value<'_>) -> Option<(f64, i32)> {
    Value::small_float(r, l).map(|(i, f)| (f, i))
}

/// A register form: `$guard` binds `R[b]` and `R[c]` as the kinds the form
/// is for in one branch; `$body` computes the [`Out`]. The guard's failure
/// and a `Miss` go to the generic handler.
macro_rules! reg_form {
    ($name:ident, $guard:expr, |$l:ident, $r:ident| $body:expr) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = (insn.a(), insn.b(), insn.c());
                if let Some(($l, $r)) = $guard(&reg![lhs], &reg![rhs]) {
                    match $body {
                        Out::Value(v) => {
                            reg![dst] = v;
                            next!()
                        }
                        Out::Float(f) => {
                            reg![dst].write_float_unchecked(f);
                            next!()
                        }
                        Out::FloatChecked(f) => {
                            reg![dst].write_float(f);
                            next!()
                        }
                        Out::Miss => {}
                    }
                }
                tail!(arith_generic)
            }
        }
    };
}

/// An immediate form: `$guard` binds `R[b]` and the immediate, in source
/// order (`$reversed` puts the immediate first).
macro_rules! imm_form {
    ($name:ident, $guard:ident, $reversed:expr, |$l:ident, $r:ident| $body:expr) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, src) = (insn.a(), insn.b());
                if let Some(($l, $r)) = $guard(&reg![src], insn, $reversed) {
                    match $body {
                        Out::Value(v) => {
                            reg![dst] = v;
                            next!()
                        }
                        Out::Float(f) => {
                            reg![dst].write_float_unchecked(f);
                            next!()
                        }
                        Out::FloatChecked(f) => {
                            reg![dst].write_float(f);
                            next!()
                        }
                        Out::Miss => {}
                    }
                }
                tail!(arith_generic)
            }
        }
    };
}

/// The immediate as a float, whichever kind it is.
#[inline(always)]
fn imm_as_float(insn: Instruction) -> f64 {
    if insn.imm_is_int() {
        insn.imm_int() as f64
    } else {
        insn.imm_float()
    }
}

/// Small register, integer immediate (31-bit, so an i32). The `_I` form is
/// only written for an integer immediate, and a site's immediate never
/// changes.
#[inline(always)]
fn imm_i(v: &Value<'_>, insn: Instruction, reversed: bool) -> Option<(i32, i32)> {
    debug_assert!(insn.imm_is_int());
    let i = v.get_small()?;
    let k = insn.imm_int() as i32;
    Some(if reversed { (k, i) } else { (i, k) })
}

/// Float register, immediate of either kind.
#[inline(always)]
fn imm_f(v: &Value<'_>, insn: Instruction, reversed: bool) -> Option<(f64, f64)> {
    if !v.is_float() {
        return None;
    }
    let (f, k) = (v.read_float(), imm_as_float(insn));
    Some(if reversed { (k, f) } else { (f, k) })
}

/// Small register, immediate of either kind, float result (the ops whose
/// `_I` form takes an integer immediate only see a float one here).
#[inline(always)]
fn imm_if(v: &Value<'_>, insn: Instruction, reversed: bool) -> Option<(f64, f64)> {
    let i = v.get_small()? as f64;
    let k = imm_as_float(insn);
    Some(if reversed { (k, i) } else { (i, k) })
}

/// Small register, float immediate.
#[inline(always)]
fn imm_if_float(v: &Value<'_>, insn: Instruction, reversed: bool) -> Option<(f64, f64)> {
    debug_assert!(!insn.imm_is_int());
    let i = v.get_small()? as f64;
    let k = insn.imm_float();
    Some(if reversed { (k, i) } else { (i, k) })
}

/// Small register; a bitwise immediate is always an integer.
#[inline(always)]
fn immbit_i(v: &Value<'_>, insn: Instruction, reversed: bool) -> Option<(i32, i32)> {
    debug_assert!(insn.imm_is_int());
    let i = v.get_small()?;
    let k = insn.imm_int() as i32;
    Some(if reversed { (k, i) } else { (i, k) })
}

reg_form!(op_add_ii, Value::both_small, |l, r| val(num::Add::small(
    l, r
)));
reg_form!(op_sub_ii, Value::both_small, |l, r| val(num::Sub::small(
    l, r
)));
reg_form!(op_mul_ii, Value::both_small, |l, r| val(num::Mul::small(
    l, r
)));
reg_form!(op_mod_ii, Value::both_small, |l, r| val(num::Mod::small(
    l, r
)));
reg_form!(op_idiv_ii, Value::both_small, |l, r| val(num::IDiv::small(
    l, r
)));
reg_form!(op_div_ii, Value::both_small, |l, r| Out::Float(
    num::Div::float_raw(l as f64, r as f64)
));
reg_form!(op_pow_ii, Value::both_small, |l, r| Out::FloatChecked(
    num::Pow::float_raw(l as f64, r as f64)
));
reg_form!(op_add_ff, Value::both_float, |l, r| Out::Float(
    num::Add::float_raw(l, r)
));
reg_form!(op_sub_ff, Value::both_float, |l, r| Out::Float(
    num::Sub::float_raw(l, r)
));
reg_form!(op_mul_ff, Value::both_float, |l, r| Out::Float(
    num::Mul::float_raw(l, r)
));
reg_form!(op_mod_ff, Value::both_float, |l, r| Out::FloatChecked(
    num::Mod::float_raw(l, r)
));
reg_form!(op_pow_ff, Value::both_float, |l, r| Out::FloatChecked(
    num::Pow::float_raw(l, r)
));
reg_form!(op_div_ff, Value::both_float, |l, r| Out::Float(
    num::Div::float_raw(l, r)
));
reg_form!(op_idiv_ff, Value::both_float, |l, r| Out::Float(
    num::IDiv::float_raw(l, r)
));
reg_form!(op_add_if, Value::small_float, |l, r| Out::Float(
    num::Add::float_raw(l as f64, r)
));
reg_form!(op_sub_if, Value::small_float, |l, r| Out::Float(
    num::Sub::float_raw(l as f64, r)
));
reg_form!(op_mul_if, Value::small_float, |l, r| Out::Float(
    num::Mul::float_raw(l as f64, r)
));
reg_form!(op_div_if, Value::small_float, |l, r| Out::Float(
    num::Div::float_raw(l as f64, r)
));
reg_form!(op_add_fi, float_small, |l, r| Out::Float(
    num::Add::float_raw(l, r as f64)
));
reg_form!(op_sub_fi, float_small, |l, r| Out::Float(
    num::Sub::float_raw(l, r as f64)
));
reg_form!(op_mul_fi, float_small, |l, r| Out::Float(
    num::Mul::float_raw(l, r as f64)
));
reg_form!(op_div_fi, float_small, |l, r| Out::Float(
    num::Div::float_raw(l, r as f64)
));
/// Any inline numbers (`_NN`): the form of a site whose kinds keep changing.
macro_rules! nn_form {
    ($name:ident, $k:ty, $store:ident) => {
        reg_form!($name, nn_guard, |l, r| {
            if let Some((li, ri)) = Value::both_small(l, r) {
                val(<$k as ArithOp>::small(li, ri))
            } else if let Some(lf) = small_or_float(l)
                && let Some(rf) = small_or_float(r)
            {
                Out::$store(<$k as ArithOp>::float_raw(lf, rf))
            } else {
                Out::Miss
            }
        });
    };
}

#[inline(always)]
fn nn_guard<'a, 'gc>(
    l: &'a Value<'gc>,
    r: &'a Value<'gc>,
) -> Option<(&'a Value<'gc>, &'a Value<'gc>)> {
    Some((l, r))
}

nn_form!(op_add_nn, num::Add, Float);
nn_form!(op_sub_nn, num::Sub, Float);
nn_form!(op_mul_nn, num::Mul, Float);
nn_form!(op_mod_nn, num::Mod, FloatChecked);
nn_form!(op_pow_nn, num::Pow, FloatChecked);
nn_form!(op_div_nn, num::Div, Float);
nn_form!(op_idiv_nn, num::IDiv, Float);

reg_form!(op_band_ii, Value::both_small, |l, r| val(num::BAnd::small(
    l, r
)));
reg_form!(op_bor_ii, Value::both_small, |l, r| val(num::BOr::small(
    l, r
)));
reg_form!(op_bxor_ii, Value::both_small, |l, r| val(num::BXor::small(
    l, r
)));
reg_form!(op_shl_ii, Value::both_small, |l, r| val(num::Shl::small(
    l, r
)));
reg_form!(op_shr_ii, Value::both_small, |l, r| val(num::Shr::small(
    l, r
)));

imm_form!(op_addi_i, imm_i, false, |l, r| val(num::Add::small(l, r)));
imm_form!(op_subi_i, imm_i, false, |l, r| val(num::Sub::small(l, r)));
imm_form!(op_muli_i, imm_i, false, |l, r| val(num::Mul::small(l, r)));
imm_form!(op_modi_i, imm_i, false, |l, r| val(num::Mod::small(l, r)));
imm_form!(op_idivi_i, imm_i, false, |l, r| val(num::IDiv::small(l, r)));
imm_form!(op_rsubi_i, imm_i, true, |l, r| val(num::Sub::small(l, r)));
imm_form!(op_addi_f, imm_f, false, |l, r| Out::Float(
    num::Add::float_raw(l, r)
));
imm_form!(op_subi_f, imm_f, false, |l, r| Out::Float(
    num::Sub::float_raw(l, r)
));
imm_form!(op_muli_f, imm_f, false, |l, r| Out::Float(
    num::Mul::float_raw(l, r)
));
imm_form!(op_modi_f, imm_f, false, |l, r| Out::FloatChecked(
    num::Mod::float_raw(l, r)
));
imm_form!(op_idivi_f, imm_f, false, |l, r| Out::Float(
    num::IDiv::float_raw(l, r)
));
imm_form!(op_rsubi_f, imm_f, true, |l, r| Out::Float(
    num::Sub::float_raw(l, r)
));
imm_form!(op_powi_f, imm_f, false, |l, r| Out::FloatChecked(
    num::Pow::float_raw(l, r)
));
imm_form!(op_divi_f, imm_f, false, |l, r| Out::Float(
    num::Div::float_raw(l, r)
));
imm_form!(op_rdivi_f, imm_f, true, |l, r| Out::Float(
    num::Div::float_raw(l, r)
));
imm_form!(op_addi_if, imm_if_float, false, |l, r| Out::Float(
    num::Add::float_raw(l, r)
));
imm_form!(op_subi_if, imm_if_float, false, |l, r| Out::Float(
    num::Sub::float_raw(l, r)
));
imm_form!(op_muli_if, imm_if_float, false, |l, r| Out::Float(
    num::Mul::float_raw(l, r)
));
imm_form!(op_divi_if, imm_if, false, |l, r| Out::Float(
    num::Div::float_raw(l, r)
));
imm_form!(op_rdivi_if, imm_if, true, |l, r| Out::Float(
    num::Div::float_raw(l, r)
));
imm_form!(op_powi_if, imm_if, false, |l, r| Out::FloatChecked(
    num::Pow::float_raw(l, r)
));
imm_form!(op_rpowi_if, imm_if, true, |l, r| Out::FloatChecked(
    num::Pow::float_raw(l, r)
));
imm_form!(op_bandi_i, immbit_i, false, |l, r| val(num::BAnd::small(
    l, r
)));
imm_form!(op_bori_i, immbit_i, false, |l, r| val(num::BOr::small(
    l, r
)));
imm_form!(op_bxori_i, immbit_i, false, |l, r| val(num::BXor::small(
    l, r
)));
imm_form!(op_shli_i, immbit_i, false, |l, r| val(num::Shl::small(
    l, r
)));
imm_form!(op_shri_i, immbit_i, false, |l, r| val(num::Shr::small(
    l, r
)));

/// The metamethod of an arithmetic kind.
const fn kind_mm(kind: ArithKind) -> MetamethodBits {
    match kind {
        ArithKind::Add => MetamethodBits::ADD,
        ArithKind::Sub => MetamethodBits::SUB,
        ArithKind::Mul => MetamethodBits::MUL,
        ArithKind::Mod => MetamethodBits::MOD,
        ArithKind::Pow => MetamethodBits::POW,
        ArithKind::Div => MetamethodBits::DIV,
        ArithKind::IDiv => MetamethodBits::IDIV,
        ArithKind::BAnd => MetamethodBits::BAND,
        ArithKind::BOr => MetamethodBits::BOR,
        ArithKind::BXor => MetamethodBits::BXOR,
        ArithKind::Shl => MetamethodBits::SHL,
        ArithKind::Shr => MetamethodBits::SHR,
        ArithKind::None => MetamethodBits::ADD,
    }
}

/// The metamethod index by generic opcode byte, so the metamethod forms read
/// theirs with one load instead of a `match`.
static BINOP_MM: [MmIndex; 256] = {
    let mut t = [MmIndex::of(MetamethodBits::ADD); 256];
    let mut i = 0;
    while i < Op::COUNT {
        t[i] = MmIndex::of(kind_mm(crate::instruction::OP_INFO[i].kind));
        i += 1;
    }
    t
};

/// The metamethod `t`'s metatable has for the binary opcode `orig`; nil when
/// none.
#[inline(always)]
fn table_binop_mm<'gc>(t: Table<'gc>, orig: Op) -> Value<'gc> {
    match t.shape().mt_cache() {
        Some(cache) => cache.mm_at(BINOP_MM[orig as usize]),
        None => Value::nil(),
    }
}

/// Stage `mm(a, b)` above the frame's window without growing the stack
/// (`false` when that's needed), so the metamethod forms need no stack frame.
macro_rules! stage_binop_mm {
    ($pc:ident, $base:ident, $closure:ident, $mm:expr, $a:expr, $b:expr) => {{
        let hdr = unsafe { $base.add($closure.max_stack_size as usize) };
        if std::hint::likely(unsafe { hdr.add(HDR + 2) }.cast_const() <= thread!().stack_end) {
            unsafe {
                frame::write_hdr(hdr, $mm.to_raw(), handler_bits(ret_store_a), $base, $pc);
                hdr.add(HDR).write($a);
                hdr.add(HDR + 1).write($b);
            }
            tail!(
                crate::vm::ops::call::enter,
                pc = hdr as *const Instruction,
                insn = Slot::nret(2)
            )
        }
    }};
}

/// The operand kind a site is specialized on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Small,
    Float,
    Other,
}

#[inline(always)]
fn kind_of(v: &Value<'_>) -> Kind {
    if v.is_float() {
        Kind::Float
    } else if v.get_small().is_some() {
        Kind::Small
    } else {
        Kind::Other
    }
}

/// The form of `generic` for the numeric operand kinds seen, if one
/// exists. `rhs` is the immediate for an immediate family.
fn numeric_form(generic: Op, lhs: &Value<'_>, rhs: &Value<'_>, imm_int: bool) -> Option<Op> {
    let info = generic.info();
    let forms = &info.forms;
    match info.family {
        Family::RegArith | Family::RegBit => match (kind_of(lhs), kind_of(rhs)) {
            (Kind::Small, Kind::Small) => forms[0],
            (Kind::Float, Kind::Float) => forms[1],
            (Kind::Small, Kind::Float) => forms[2],
            (Kind::Float, Kind::Small) => forms[3],
            _ => None,
        },
        Family::ImmArith | Family::ImmBit => {
            // The register operand: the immediate is on the left for the
            // reversed forms (and the slow path passed it as `lhs`).
            let reg = if info.reversed { rhs } else { lhs };
            match (kind_of(reg), imm_int) {
                (Kind::Small, true) => forms[0].or(forms[2]),
                (Kind::Small, false) => forms[2],
                (Kind::Float, _) => forms[1],
                _ => None,
            }
        }
        _ => None,
    }
}

/// Rewrite the site for `form`, the form of the operand
/// kinds just seen. Only a change of specializable kinds counts as a miss
/// (three lock the site to its generic opcode): a form whose operation
/// failed on its own kinds (overflow, a zero divisor) and a boxed integer
/// operand leave the site as it is, so a site that mostly sees small
/// integers keeps its form. With `lock`, a `None` form counts too (the
/// kinds are ones the generic handles inline but no form covers, or a
/// metamethod site without a form).
#[inline(always)]
pub(crate) unsafe fn specialize(
    site: *mut Instruction,
    insn: Instruction,
    form: Option<Op>,
    lock: bool,
) {
    let current = insn.op();
    let generic = insn.unquickened();
    let (misses, _) = insn.adaptive();
    let is_form = current != generic.op();
    let word = match form {
        Some(f) if f == current => return,
        None if !lock => return,
        _ => {
            let misses = misses + is_form as u8;
            if misses >= MISSES_TO_LOCK {
                // A numeric site that keeps changing kinds takes the
                // any-numbers form, locked, where there is one.
                match generic.op().info().forms[4] {
                    Some(nn) if !lock => generic.with_op(nn).with_adaptive(misses.min(3), true),
                    _ => generic.with_adaptive(misses.min(3), true),
                }
            } else {
                match form {
                    Some(f @ (Op::ARITH_MM | Op::ARITH_MM_R | Op::ARITH_MMI)) => generic
                        .with_mm_form(f, generic.op())
                        .with_adaptive(misses, false),
                    Some(f) => generic.with_op(f).with_adaptive(misses, false),
                    // Counted; the generic opcode specializes again on the
                    // next kinds that have a form, until the misses lock it.
                    None => generic.with_adaptive(misses, false),
                }
            }
        }
    };
    unsafe { site.write(word) }
}

#[inline(always)]
fn small_or_float(v: &Value<'_>) -> Option<f64> {
    if v.is_float() {
        Some(v.read_float())
    } else {
        v.get_small().map(f64::from)
    }
}

/// The inline-number cases of an arithmetic kind, what a locked site runs
/// on every execution: small integers, floats and their mixes.
#[inline(always)]
fn inline_arith<'gc>(kind: ArithKind, lhs: &Value<'gc>, rhs: &Value<'gc>) -> Option<Value<'gc>> {
    use ArithKind::*;
    macro_rules! arith {
        ($k:ty) => {{
            if let Some((li, ri)) = Value::both_small(lhs, rhs) {
                <$k as ArithOp>::small(li, ri)
            } else if let Some(lf) = small_or_float(lhs)
                && let Some(rf) = small_or_float(rhs)
            {
                Some(Value::float(<$k as ArithOp>::float_raw(lf, rf)))
            } else {
                Option::None
            }
        }};
    }
    macro_rules! bit {
        ($k:ty) => {{
            let (li, ri) = Value::both_small(lhs, rhs)?;
            <$k as BitOp>::small(li, ri)
        }};
    }
    match kind {
        Add => arith!(num::Add),
        Sub => arith!(num::Sub),
        Mul => arith!(num::Mul),
        Mod => arith!(num::Mod),
        Pow => arith!(num::Pow),
        Div => arith!(num::Div),
        IDiv => arith!(num::IDiv),
        BAnd => bit!(num::BAnd),
        BOr => bit!(num::BOr),
        BXor => bit!(num::BXor),
        Shl => bit!(num::Shl),
        Shr => bit!(num::Shr),
        None => Option::None,
    }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `ARITH_MM`: `R[a] = mm(R[b], R[c])`, `mm` from `R[b]`, a table.
    op fn op_arith_mm {
        let (lhs, rhs) = (insn.b(), insn.c());
        let (l, r) = (reg![lhs], reg![rhs]);
        if let Some(t) = l.get_table() {
            let mm = table_binop_mm(t, insn.mm_orig_reg());
            if !mm.is_nil() {
                stage_binop_mm!(pc, base, closure, mm, l, r);
            }
        }
        tail!(arith_generic)
    }

    /// `ARITH_MM_R`: as `ARITH_MM`, `mm` from `R[c]`, a table, `R[b]` a
    /// number.
    op fn op_arith_mm_r {
        let (lhs, rhs) = (insn.b(), insn.c());
        let (l, r) = (reg![lhs], reg![rhs]);
        if let Some(t) = r.get_table()
            && l.is_number()
            && rt.number_metatable().is_none()
        {
            let mm = table_binop_mm(t, insn.mm_orig_reg());
            if !mm.is_nil() {
                stage_binop_mm!(pc, base, closure, mm, l, r);
            }
        }
        tail!(arith_generic)
    }

    /// `ARITH_MMI`: an immediate form whose register operand is a table with
    /// the metamethod. With the immediate on the left, numbers must have none.
    op fn op_arith_mmi {
        let src = insn.b();
        let flipped = insn.c() & 1 != 0;
        let v = reg![src];
        if let Some(t) = v.get_table() {
            let orig = insn.mm_orig_imm();
            let mm = table_binop_mm(t, orig);
            let imm_left = flipped || orig.is_reversed();
            if !mm.is_nil() && (!imm_left || rt.number_metatable().is_none()) {
                // Immediates are 31-bit, so an integer one is inline.
                let k = if insn.imm_is_int() {
                    Value::small(insn.imm_int() as i32)
                } else {
                    Value::float(insn.imm_float())
                };
                let (a, b) = if imm_left { (k, v) } else { (v, k) };
                stage_binop_mm!(pc, base, closure, mm, a, b);
            }
        }
        tail!(arith_generic)
    }

    /// `R[a] = -R[b]`
    op fn op_unm {
        let (dst, src) = insn.ab();
        let v = &reg![src];
        if v.is_float() {
            let f = v.read_float();
            reg![dst].write_float_unchecked(-f);
            next!()
        }
        if let Some(i) = v.get_small()
            && let Some(n) = i.checked_neg()
        {
            reg![dst] = Value::small(n);
            next!()
        }
        tail!(unm_slow)
    }

    slow fn unm_slow {
        let insn = insn_at!();
        let (dst, src) = insn.ab();
        let v = reg![src];
        if let Some(i) = v.get_integer() {
            reg![dst] = Value::integer(rt.mutation(), i.wrapping_neg());
            next!()
        }
        let mm = rt.mm_of(v, MetamethodBits::UNM);
        if mm.is_nil() {
            raise!(OpError::Arith(v, v))
        }
        // Lua passes the operand twice for unary metamethods.
        stage_mm!(pc, base, rt, ret_store_a, mm, [v, v])
    }

    /// `R[a] = ~R[b]`
    op fn op_bnot {
        let (dst, src) = insn.ab();
        if let Some(i) = reg![src].get_small() {
            reg![dst] = Value::small(!i);
            next!()
        }
        tail!(bnot_slow)
    }

    slow fn bnot_slow {
        let insn = insn_at!();
        let (dst, src) = insn.ab();
        let v = reg![src];
        if let Some(i) = v.get_integer() {
            reg![dst] = Value::integer(rt.mutation(), !i);
            next!()
        }
        let mm = rt.mm_of(v, MetamethodBits::BNOT);
        if mm.is_nil() {
            raise!(OpError::Bitwise(v, v))
        }
        stage_mm!(pc, base, rt, ret_store_a, mm, [v, v])
    }

    /// `R[a] = not R[b]`
    op fn op_not {
        let (dst, src) = insn.ab();
        let v = reg![src];
        reg![dst] = Value::boolean(v.is_falsy());
        next!()
    }

    /// The generic handler of every binary arithmetic and bitwise site:
    /// dispatched for a generic opcode (a first execution or a locked
    /// site) and tailed to by a form's guard failure. It does the inline
    /// numbers here, specializing the site, and leaves the rest (boxed
    /// integers, metamethods, errors) to `arith_slow`, so a locked site
    /// still runs without a stack frame.
    op fn arith_generic {
        // SAFETY: `Code` keeps instructions in cells, and `pc` is past this one.
        let site = unsafe { pc.sub(1).cast_mut() };
        let generic = insn.generic_op();
        let info = generic.info();
        let (_, locked) = insn.adaptive();
        let dst = insn.a();
        let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
        let (lhs, rhs) = if imm {
            // Immediates are 31-bit, so an integer one is inline.
            let k = if insn.imm_is_int() {
                Value::small(insn.imm_int() as i32)
            } else {
                Value::float(insn.imm_float())
            };
            let v = reg![insn.b()];
            let flipped = insn.c() & 1 != 0;
            if flipped || info.reversed { (k, v) } else { (v, k) }
        } else {
            (reg![insn.b()], reg![insn.c()])
        };
        if let Some(v) = inline_arith(info.kind, &lhs, &rhs) {
            if !locked {
                let form = numeric_form(generic, &lhs, &rhs, insn.imm_is_int());
                unsafe { specialize(site, insn, form, false) };
            }
            reg![dst] = v;
            next!()
        }
        tail!(arith_slow)
    }

    /// The rest of `arith_generic`: boxed integers, string coercion,
    /// metamethods and the errors, with the metamethod forms' specialization.
    slow fn arith_slow {
        use ArithKind::*;
        let insn = insn_at!();
        let site = unsafe { pc.sub(1).cast_mut() };
        let generic = insn.generic_op();
        let info = generic.info();
        let (_, locked) = insn.adaptive();
        let dst = insn.a();
        let mc = rt.mutation();
        let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
        let (lhs, rhs) = if imm {
            let (v, k) = (reg![insn.b()], insn.imm_value(mc));
            let flipped = insn.c() & 1 != 0;
            if flipped || info.reversed { (k, v) } else { (v, k) }
        } else {
            (reg![insn.b()], reg![insn.c()])
        };
        let r = match info.kind {
            Add => num::op_arith_slow::<num::Add>(mc, lhs, rhs),
            Sub => num::op_arith_slow::<num::Sub>(mc, lhs, rhs),
            Mul => num::op_arith_slow::<num::Mul>(mc, lhs, rhs),
            Mod => num::op_arith_slow::<num::Mod>(mc, lhs, rhs),
            Pow => num::op_arith_slow::<num::Pow>(mc, lhs, rhs),
            Div => num::op_arith_slow::<num::Div>(mc, lhs, rhs),
            IDiv => num::op_arith_slow::<num::IDiv>(mc, lhs, rhs),
            BAnd => num::op_bit_slow::<num::BAnd>(mc, lhs, rhs),
            BOr => num::op_bit_slow::<num::BOr>(mc, lhs, rhs),
            BXor => num::op_bit_slow::<num::BXor>(mc, lhs, rhs),
            Shl => num::op_bit_slow::<num::Shl>(mc, lhs, rhs),
            Shr => num::op_bit_slow::<num::Shr>(mc, lhs, rhs),
            None => unreachable!("arith_slow on {generic:?}"),
        };
        let bit = kind_mm(info.kind);
        match r {
            num::SlowNum::Value(v) => {
                // Boxed integers, or an overflow of the inline case: no form
                // change, and the site stays as it is.
                reg![dst] = v;
                gc_check!();
                next!()
            }
            num::SlowNum::ModByZero => raise!(OpError::ModByZero),
            num::SlowNum::DivByZero => raise!(OpError::DivByZero),
            num::SlowNum::NotNumbers => {}
        }
        let lhs_mm = rt.mm_of(lhs, bit);
        let mm = if lhs_mm.is_nil() { rt.mm_of(rhs, bit) } else { lhs_mm };
        if mm.is_nil() {
            if !locked {
                unsafe { specialize(site, insn, Option::None, true) };
            }
            let bitwise = MetamethodBits::BAND | MetamethodBits::BOR | MetamethodBits::BXOR;
            raise!(if (bitwise | MetamethodBits::SHL | MetamethodBits::SHR).contains(bit) {
                OpError::Bitwise(lhs, rhs)
            } else {
                OpError::Arith(lhs, rhs)
            })
        }
        if !locked {
            // The forms read the metamethod from a table's shape; taking it
            // from the right needs the left to be a number without one.
            let form = if !lhs_mm.is_nil() {
                lhs.get_table().map(|_| Op::ARITH_MM)
            } else if rhs.get_table().is_some() && lhs.is_number() && rt.number_metatable().is_none() {
                Some(Op::ARITH_MM_R)
            } else {
                Option::None
            };
            let form = match info.family {
                Family::ImmArith => form.map(|_| Op::ARITH_MMI),
                Family::ImmBit => Option::None,
                _ => form,
            };
            unsafe { specialize(site, insn, form, true) };
        }
        stage_mm!(pc, base, rt, ret_store_a, mm, [lhs, rhs])
    }
}
