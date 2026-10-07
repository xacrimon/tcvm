//! Register-key table access, table construction, length and concatenation.

use crate::env::MetamethodBits;
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::value::Value;
use crate::vm::abi::handler;
use crate::vm::num;
use crate::vm::ops::meta::{binop_metamethod, ret_store_a};
use crate::vm::unwind::OpError;

/// Keys a raw table write must refuse (`luaH_set`); an absent key only errors
/// once it is actually stored, so `__newindex` still sees nil/NaN keys first.
macro_rules! check_index_key {
    ($k:expr) => {{
        let k: Value<'gc> = $k;
        if std::hint::unlikely(k.is_nil()) {
            raise!(OpError::NilIndex)
        }
        if std::hint::unlikely(k.get_float().is_some_and(f64::is_nan)) {
            raise!(OpError::NanIndex)
        }
    }};
}
pub(crate) use check_index_key;

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `R[a] = R[b][R[c]]`
    op fn op_gettable {
        let (dst, table, key) = insn.abc();
        let Some(t) = reg![table].get_table() else {
            tail!(crate::vm::ops::field::get_slow)
        };
        // An integer key inside the array part, handled without calls so this
        // handler needs no stack frame; any other key goes to the general path.
        if let Some(i) = reg![key].get_small() {
            let state = t.inner().borrow();
            if let Some(v) = state.array_get(i as usize)
                && !(v.is_nil() && state.shape().has_mm(MetamethodBits::INDEX))
            {
                reg![dst] = v;
                next!()
            }
        }
        tail!(gettable_general)
    }

    /// GETTABLE past the array fast path: any raw key, `__index` to `get_slow`.
    slow fn gettable_general {
        let insn = insn_at!();
        let (dst, table, key) = insn.abc();
        let Some(t) = reg![table].get_table() else {
            tail!(crate::vm::ops::field::get_slow)
        };
        let k = reg![key];
        let (v, need_index) = {
            let state = t.inner().borrow();
            let v = state.raw_get(k);
            let need = v.is_nil() && state.shape().has_mm(MetamethodBits::INDEX);
            (v, need)
        };
        if need_index {
            tail!(crate::vm::ops::field::get_slow)
        }
        reg![dst] = v;
        next!()
    }

    /// `R[b][R[c]] = R[a]`
    op fn op_settable {
        let (src, table, key) = insn.abc();
        let Some(t) = reg![table].get_table() else {
            tail!(crate::vm::ops::field::set_slow)
        };
        // As in GETTABLE; barrier work also goes to the general path.
        if let Some(i) = reg![key].get_small() {
            let state = t.inner().borrow();
            if let Some(old) = state.array_get(i as usize)
                && !(old.is_nil() && state.shape().has_mm(MetamethodBits::NEWINDEX))
            {
                drop(state);
                barrier!(t.inner());
                let v = reg![src];
                // In range: checked above, and nothing ran in between.
                unsafe { t.borrow_mut_barriered().set_array_at(i as usize, v) };
                next!()
            }
        }
        tail!(settable_general)
    }

    /// SETTABLE past the array fast path.
    slow fn settable_general {
        let insn = insn_at!();
        let (src, table, key) = insn.abc();
        let Some(t) = reg![table].get_table() else {
            tail!(crate::vm::ops::field::set_slow)
        };
        let k = reg![key];
        let v = reg![src];
        let needs_newindex = {
            let state = t.inner().borrow();
            state.shape().has_mm(MetamethodBits::NEWINDEX) && state.raw_get(k).is_nil()
        };
        if needs_newindex {
            tail!(crate::vm::ops::field::set_slow)
        }
        check_index_key!(k);
        let mut state = t.inner().borrow_mut(rt.mutation());
        state.raw_set_keyed(rt, k, v);
        next!()
    }

    /// `R[a] = {}` from `templates[d]`.
    op fn op_newtable {
        let (dst, template) = insn.ad();
        // SAFETY: the compiler gives each NEWTABLE a template.
        let t = unsafe { closure.proto.templates.get_unchecked(template as usize) };
        reg![dst] = Value::table(Table::from_template(rt.mutation(), t));
        gc_check!();
        next!()
    }

    /// `R[a][d + i] = R[a + i]` for `i in 1..=b`; `b == 0` takes the count
    /// from `top`.
    op fn op_setlist {
        let (table, count, offset) = insn.abd();
        let Some(t) = reg![table].get_table() else {
            raise!(OpError::Internal("SETLIST on a non-table"))
        };
        let ts = thread!();
        let bi = ts.slot_index(base);
        let start = bi + table as usize + 1;
        let n = if count == 0 { ts.top - start } else { count as usize };
        barrier!(t.inner());
        let items = &ts.stack[start..start + n];
        unsafe { t.borrow_mut_barriered() }.set_list(rt.mutation(), offset as usize, items);
        next!()
    }

    /// `R[a] = #R[b]`
    op fn op_len {
        let (dst, src) = insn.ab();
        let v = reg![src];
        if let Some(t) = v.get_table()
            && !t.shape().has_mm(MetamethodBits::LEN)
        {
            reg![dst] = Value::integer(rt.mutation(), t.raw_len() as i64);
            next!()
        }
        tail!(len_slow)
    }

    /// LEN of a string, a table with `__len`, or anything else.
    slow fn len_slow {
        let insn = insn_at!();
        let (dst, src) = insn.ab();
        let v = reg![src];
        // Strings never consult `__len`.
        if let Some(s) = v.get_string() {
            reg![dst] = Value::integer(rt.mutation(), s.len() as i64);
            next!()
        }
        let mm = rt.mm_of(v, MetamethodBits::LEN);
        if mm.is_nil() {
            let Some(t) = v.get_table() else {
                raise!(OpError::Len(v))
            };
            reg![dst] = Value::integer(rt.mutation(), t.raw_len() as i64);
            next!()
        }
        // Like the other unary metamethods, `__len` gets its operand twice.
        crate::vm::ops::meta::stage_mm!(pc, base, rt, ret_store_a, mm, [v, v])
    }

    /// `R[a] = R[b] .. R[c]`
    op fn op_concat {
        let (dst, lhs, rhs) = insn.abc();
        let a = reg![lhs];
        let b = reg![rhs];
        let s = rt.with_buf(|buf| {
            buf.reserve(num::concat_len(a) + num::concat_len(b));
            (num::coerce_to_str(buf, a) && num::coerce_to_str(buf, b)).then(|| LuaString::new(rt, buf))
        });
        if let Some(s) = s {
            reg![dst] = Value::string(s);
            gc_check!();
            next!()
        }
        let mm = binop_metamethod(rt, a, b, MetamethodBits::CONCAT);
        if mm.is_nil() {
            raise!(OpError::Concat(a, b))
        }
        call_mm!(ret_store_a, mm, [a, b])
    }
}
