//! Arithmetic and bitwise operators, register and immediate forms, the
//! quickened `_NUM` and metamethod forms, and their shared slow path.

use crate::env::shape::{MetamethodBits, MmIndex};
use crate::env::table::Table;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::vm::abi::{Slot, handler, handler_bits};
use crate::vm::frame::{self, HDR};
use crate::vm::num;
use crate::vm::ops::meta::{ret_store_a, stage_mm};
use crate::vm::unwind::OpError;

/// `R[a] = R[b] <op> R[c]` for the arithmetic opcodes.
///
/// Inline ints and floats only; boxed ints, overflow, zero divisors and mixes
/// all go to the slow handler so no call (and no stack frame) lands here.
/// Floats are the fall-through arm.
macro_rules! arith_handler {
    ($name:ident, $num_kind:ty) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = insn.abc();
                let (l, r) = (&reg![lhs], &reg![rhs]);
                if std::hint::likely(l.is_float() && r.is_float()) {
                    let (lf, rf) = (l.read_float(), r.read_float());
                    reg![dst].write_float(<$num_kind as num::ArithOp>::float_raw(lf, rf));
                    next!()
                } else if let Some((li, ri)) = Value::both_small(l, r) {
                    if let Some(v) = <$num_kind as num::ArithOp>::small(li, ri) {
                        reg![dst] = v;
                        next!()
                    }
                }
                tail!(binop_slow)
            }
        }
    };
}

/// `R[a] = R[b] <op> R[c]` for the bitwise opcodes.
macro_rules! bit_handler {
    ($name:ident, $num_kind:ty) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = insn.abc();
                let (l, r) = (&reg![lhs], &reg![rhs]);
                if let Some(li) = l.get_small()
                    && let Some(ri) = r.get_small()
                    && let Some(v) = <$num_kind as num::BitOp>::small(li, ri)
                {
                    reg![dst] = v;
                    next!()
                }
                tail!(binop_slow)
            }
        }
    };
}

arith_handler!(op_add, num::Add);
arith_handler!(op_sub, num::Sub);
arith_handler!(op_mul, num::Mul);
arith_handler!(op_mod, num::Mod);
arith_handler!(op_pow, num::Pow);
arith_handler!(op_div, num::Div);
arith_handler!(op_idiv, num::IDiv);
bit_handler!(op_band, num::BAnd);
bit_handler!(op_bor, num::BOr);
bit_handler!(op_bxor, num::BXor);
bit_handler!(op_shl, num::Shl);
bit_handler!(op_shr, num::Shr);

#[inline(always)]
fn small_or_float(v: &Value<'_>) -> Option<f64> {
    if v.is_float() {
        Some(v.read_float())
    } else {
        v.get_small().map(f64::from)
    }
}

/// A register-form arithmetic site quickened to its `_NUM` form: the generic
/// handler's cases plus a float and an inline int in either order.
macro_rules! arith_num_handler {
    ($name:ident, $num_kind:ty) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = insn.abc();
                let (l, r) = (&reg![lhs], &reg![rhs]);
                if let Some((li, ri)) = Value::both_small(l, r) {
                    if let Some(v) = <$num_kind as num::ArithOp>::small(li, ri) {
                        reg![dst] = v;
                        next!()
                    }
                } else if let Some(lf) = small_or_float(l)
                    && let Some(rf) = small_or_float(r)
                {
                    reg![dst].write_float(<$num_kind as num::ArithOp>::float_raw(lf, rf));
                    next!()
                }
                tail!(binop_slow)
            }
        }
    };
}

arith_num_handler!(op_add_num, num::Add);
arith_num_handler!(op_sub_num, num::Sub);
arith_num_handler!(op_mul_num, num::Mul);
arith_num_handler!(op_mod_num, num::Mod);
arith_num_handler!(op_pow_num, num::Pow);
arith_num_handler!(op_div_num, num::Div);
arith_num_handler!(op_idiv_num, num::IDiv);

/// `R[a] = R[b] <op> imm`, or `imm <op> R[b]` when `$swap`. Unlike the
/// register form the int/float mixes are inline: the constant side converts
/// for free.
macro_rules! arith_imm_handler {
    ($name:ident, $num_kind:ty, $swap:expr) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, src, _) = insn.abc_imm();
                let v = &reg![src];
                if std::hint::likely(insn.imm_is_int()) {
                    let k = insn.imm_int();
                    if let Some(i) = v.get_small() {
                        // Immediates are 31-bit, so both operands are i32.
                        let (l, r) = if $swap { (k as i32, i) } else { (i, k as i32) };
                        if let Some(out) = <$num_kind as num::ArithOp>::small(l, r) {
                            reg![dst] = out;
                            next!()
                        }
                    } else if v.is_float() {
                        let (f, k) = (v.read_float(), k as f64);
                        let (l, r) = if $swap { (k, f) } else { (f, k) };
                        reg![dst].write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                        next!()
                    }
                } else {
                    let k = insn.imm_float();
                    if v.is_float() {
                        let f = v.read_float();
                        let (l, r) = if $swap { (k, f) } else { (f, k) };
                        reg![dst].write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                        next!()
                    } else if let Some(i) = v.get_small() {
                        let f = i as f64;
                        let (l, r) = if $swap { (k, f) } else { (f, k) };
                        reg![dst].write_float(<$num_kind as num::ArithOp>::float_raw(l, r));
                        next!()
                    }
                }
                tail!(binop_slow)
            }
        }
    };
}

/// `R[a] = R[b] <op> imm` for the bitwise opcodes. The immediate is always an
/// integer; a float register goes through the slow path's exact conversion.
macro_rules! bit_imm_handler {
    ($name:ident, $num_kind:ty, $swap:expr) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, src, _) = insn.abc_imm();
                debug_assert!(insn.imm_is_int());
                let k = insn.imm_int();
                if let Some(i) = reg![src].get_small() {
                    let (l, r) = if $swap { (k as i32, i) } else { (i, k as i32) };
                    if let Some(out) = <$num_kind as num::BitOp>::small(l, r) {
                        reg![dst] = out;
                        next!()
                    }
                }
                tail!(binop_slow)
            }
        }
    };
}

arith_imm_handler!(op_addi, num::Add, false);
arith_imm_handler!(op_subi, num::Sub, false);
arith_imm_handler!(op_muli, num::Mul, false);
arith_imm_handler!(op_modi, num::Mod, false);
arith_imm_handler!(op_powi, num::Pow, false);
arith_imm_handler!(op_divi, num::Div, false);
arith_imm_handler!(op_idivi, num::IDiv, false);
arith_imm_handler!(op_rsubi, num::Sub, true);
arith_imm_handler!(op_rmodi, num::Mod, true);
arith_imm_handler!(op_rpowi, num::Pow, true);
arith_imm_handler!(op_rdivi, num::Div, true);
arith_imm_handler!(op_ridivi, num::IDiv, true);
bit_imm_handler!(op_bandi, num::BAnd, false);
bit_imm_handler!(op_bori, num::BOr, false);
bit_imm_handler!(op_bxori, num::BXor, false);
bit_imm_handler!(op_shli, num::Shl, false);
bit_imm_handler!(op_shri, num::Shr, false);
bit_imm_handler!(op_rshli, num::Shl, true);
bit_imm_handler!(op_rshri, num::Shr, true);

/// The metamethod a binary arithmetic or bitwise opcode calls.
const fn binop_mm_bit(op: Op) -> Option<MetamethodBits> {
    use Op::*;
    Some(match op {
        ADD | ADDI => MetamethodBits::ADD,
        SUB | SUBI | RSUBI => MetamethodBits::SUB,
        MUL | MULI => MetamethodBits::MUL,
        MOD | MODI | RMODI => MetamethodBits::MOD,
        POW | POWI | RPOWI => MetamethodBits::POW,
        DIV | DIVI | RDIVI => MetamethodBits::DIV,
        IDIV | IDIVI | RIDIVI => MetamethodBits::IDIV,
        BAND | BANDI => MetamethodBits::BAND,
        BOR | BORI => MetamethodBits::BOR,
        BXOR | BXORI => MetamethodBits::BXOR,
        SHL | SHLI | RSHLI => MetamethodBits::SHL,
        SHR | SHRI | RSHRI => MetamethodBits::SHR,
        _ => return None,
    })
}

/// [`binop_mm_bit`] by opcode byte, so the metamethod forms read their
/// metamethod with one load instead of a `match`.
static BINOP_MM: [MmIndex; 256] = {
    let mut t = [MmIndex::of(MetamethodBits::ADD); 256];
    let mut i = 0;
    while i < Op::COUNT {
        if let Some(bit) = binop_mm_bit(Op::ALL[i]) {
            t[i] = MmIndex::of(bit);
        }
        i += 1;
    }
    t
};

/// The metamethod `t`'s metatable has for the binary opcode `orig`; nil when
/// none.
#[inline(always)]
fn table_binop_mm<'gc>(t: Table<'gc>, orig: u8) -> Value<'gc> {
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

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `ARITH_MM`: `R[a] = mm(R[b], R[c])`, `mm` from `R[b]`, a table.
    op fn op_arith_mm {
        let (_, lhs, rhs) = insn.abc();
        let (l, r) = (reg![lhs], reg![rhs]);
        if let Some(t) = l.get_table() {
            let mm = table_binop_mm(t, insn.d() as u8);
            if !mm.is_nil() {
                stage_binop_mm!(pc, base, closure, mm, l, r);
            }
        }
        tail!(binop_slow)
    }

    /// `ARITH_MM_R`: as `ARITH_MM`, `mm` from `R[c]`, a table, `R[b]` a
    /// number.
    op fn op_arith_mm_r {
        let (_, lhs, rhs) = insn.abc();
        let (l, r) = (reg![lhs], reg![rhs]);
        if let Some(t) = r.get_table()
            && l.is_number()
            && rt.number_metatable().is_none()
        {
            let mm = table_binop_mm(t, insn.d() as u8);
            if !mm.is_nil() {
                stage_binop_mm!(pc, base, closure, mm, l, r);
            }
        }
        tail!(binop_slow)
    }

    /// `ARITH_MMI`: an immediate form whose register operand is a table with
    /// the metamethod. With the immediate on the left, numbers must have none.
    op fn op_arith_mmi {
        let (_, src, flipped) = insn.abc_imm();
        let v = reg![src];
        if let Some(t) = v.get_table() {
            let orig = insn.c() >> 1;
            let mm = table_binop_mm(t, orig);
            let imm_left = flipped || (Op::RSUBI as u8..=Op::RSHRI as u8).contains(&orig);
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
        tail!(binop_slow)
    }

    /// `R[a] = -R[b]`
    op fn op_unm {
        let (dst, src) = insn.ab();
        let v = &reg![src];
        if v.is_float() {
            let f = v.read_float();
            reg![dst].write_float(-f);
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

    /// The slow path of every binary arithmetic and bitwise opcode, register
    /// and immediate forms alike: mixed and boxed numbers, division by zero,
    /// and the metamethods. Quickens the site by what reaches it here.
    slow fn binop_slow {
        use Op::*;
        let instruction = insn_at!();
        // SAFETY: `Code` keeps instructions in cells, and `pc` is past this one.
        let site = unsafe { pc.sub(1).cast_mut() };
        let insn = instruction.unquickened();
        let op = insn.op();
        let quicken = instruction.op() == op && insn.quickenable();
        // A metamethod form also lands here when the stack needs growing,
        // which must not undo it; other misses undo it for good.
        let mm_form = matches!(instruction.op(), ARITH_MM | ARITH_MM_R | ARITH_MMI);
        let dst = insn.a();
        let mc = rt.mutation();
        let (lhs, rhs) = match op {
            ADD | SUB | MUL | MOD | POW | DIV | IDIV | BAND | BOR | BXOR | SHL | SHR => {
                (reg![insn.b()], reg![insn.c()])
            }
            _ => {
                let (_, src, flipped) = insn.abc_imm();
                let (v, k) = (reg![src], insn.imm_value(mc));
                if flipped || op.is_reversed() { (k, v) } else { (v, k) }
            }
        };
        let (r, bit) = match op {
            ADD | ADDI => (num::op_arith_slow::<num::Add>(mc, lhs, rhs), MetamethodBits::ADD),
            SUB | SUBI | RSUBI => (num::op_arith_slow::<num::Sub>(mc, lhs, rhs), MetamethodBits::SUB),
            MUL | MULI => (num::op_arith_slow::<num::Mul>(mc, lhs, rhs), MetamethodBits::MUL),
            MOD | MODI | RMODI => (num::op_arith_slow::<num::Mod>(mc, lhs, rhs), MetamethodBits::MOD),
            POW | POWI | RPOWI => (num::op_arith_slow::<num::Pow>(mc, lhs, rhs), MetamethodBits::POW),
            DIV | DIVI | RDIVI => (num::op_arith_slow::<num::Div>(mc, lhs, rhs), MetamethodBits::DIV),
            IDIV | IDIVI | RIDIVI => (num::op_arith_slow::<num::IDiv>(mc, lhs, rhs), MetamethodBits::IDIV),
            BAND | BANDI => (num::op_bit_slow::<num::BAnd>(mc, lhs, rhs), MetamethodBits::BAND),
            BOR | BORI => (num::op_bit_slow::<num::BOr>(mc, lhs, rhs), MetamethodBits::BOR),
            BXOR | BXORI => (num::op_bit_slow::<num::BXor>(mc, lhs, rhs), MetamethodBits::BXOR),
            SHL | SHLI | RSHLI => (num::op_bit_slow::<num::Shl>(mc, lhs, rhs), MetamethodBits::SHL),
            SHR | SHRI | RSHRI => (num::op_bit_slow::<num::Shr>(mc, lhs, rhs), MetamethodBits::SHR),
            _ => unreachable!("binop_slow on {op:?}"),
        };
        if mm_form && !matches!(r, num::SlowNum::NotNumbers) {
            unsafe { site.write(insn.with_no_quicken()) };
        }
        match r {
            num::SlowNum::Value(v) => {
                if quicken
                    && lhs.is_float() != rhs.is_float()
                    && let Some(n) = op.num_form()
                {
                    unsafe { site.write(insn.with_op(n)) };
                }
                reg![dst] = v;
                next!()
            }
            num::SlowNum::ModByZero => raise!(OpError::ModByZero),
            num::SlowNum::DivByZero => raise!(OpError::DivByZero),
            num::SlowNum::NotNumbers => {}
        }
        let lhs_mm = rt.mm_of(lhs, bit);
        let mm = if lhs_mm.is_nil() { rt.mm_of(rhs, bit) } else { lhs_mm };
        if quicken || mm_form {
            // The forms read the metamethod from a table's shape; taking it
            // from the right needs the left to be a number without one.
            let form = if mm.is_nil() {
                None
            } else if !lhs_mm.is_nil() {
                lhs.get_table().map(|_| ARITH_MM)
            } else if rhs.get_table().is_some() && lhs.is_number() && rt.number_metatable().is_none() {
                Some(ARITH_MM_R)
            } else {
                None
            };
            let form = form.map(|f| {
                if op.shape() == crate::instruction::Shape::AbcImm { ARITH_MMI } else { f }
            });
            if quicken && let Some(form) = form {
                unsafe { site.write(insn.with_mm_form(form)) };
            } else if mm_form && form != Some(instruction.op()) {
                unsafe { site.write(insn.with_no_quicken()) };
            }
        }
        if mm.is_nil() {
            let bitwise = MetamethodBits::BAND | MetamethodBits::BOR | MetamethodBits::BXOR;
            raise!(if (bitwise | MetamethodBits::SHL | MetamethodBits::SHR).contains(bit) {
                OpError::Bitwise(lhs, rhs)
            } else {
                OpError::Arith(lhs, rhs)
            })
        }
        stage_mm!(pc, base, rt, ret_store_a, mm, [lhs, rhs])
    }
}
