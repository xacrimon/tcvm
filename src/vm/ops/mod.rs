//! The opcode handlers, by family. Every dispatch target is declared with
//! `handler!` (`vm::abi`); shared logic is an `#[inline(always)]` function.

pub(crate) mod arith;
pub(crate) mod call;
pub(crate) mod compare;
pub(crate) mod control;
pub(crate) mod data;
pub(crate) mod field;
pub(crate) mod meta;
pub(crate) mod table;

/// Specialized handlers from rows. A row names the opcode, its
/// handler, the guard that binds the operands as the form's kinds, and the
/// operation; each arm is the shell every form of one template shares, and
/// a family's rows also list its handlers for the dispatch table as `$rows`.
/// The templates name the files' items by path; the rows live beside the
/// guards and operations they use.
macro_rules! family {
    // `R[a] = R[b] <op> R[c]`: `$guard(&R[b], &R[c])` binds the operands and
    // `$body` is an `arith::Out`; its miss and the guard's failure go to the
    // generic handler.
    (arith $rows:ident, generic = $g:ident;
     $($OP:ident = $name:ident ($guard:expr) |$l:ident, $r:ident| $body:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = (insn.a(), insn.b(), insn.c());
                if let Some(($l, $r)) = $guard(&reg![lhs], &reg![rhs]) {
                    $crate::vm::ops::family!(@store dst, $body)
                }
                tail!($g)
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // `R[a] = R[b] <op> imm`: `$guard(&R[b], insn, $reversed)` binds the
    // operands in source order (`$reversed` puts the immediate first).
    (imm $rows:ident, generic = $g:ident;
     $($OP:ident = $name:ident ($guard:ident, $reversed:expr) |$l:ident, $r:ident| $body:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, src) = (insn.a(), insn.b());
                if let Some(($l, $r)) = $guard(&reg![src], insn, $reversed) {
                    $crate::vm::ops::family!(@store dst, $body)
                }
                tail!($g)
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // Register compares: `$guard(&R[a], &R[b])` binds the operands, the
    // branch is taken when `$cond == $k`; otherwise the generic handler.
    (branch $rows:ident;
     $($OP:ident = $name:ident ($guard:expr, $g:ident, $k:literal) |$x:ident, $y:ident| $cond:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let Some(($x, $y)) = $guard(&reg![insn.a()], &reg![insn.b()]) else {
                    tail!($g)
                };
                branch!(($cond) == $k, insn.branch_offset())
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // Immediate compares of a float register: the 15-bit immediate converts
    // exactly; `$swap` puts it on the left. Anything but a float goes to
    // the generic handler.
    (branch_imm $rows:ident;
     $($OP:ident = $name:ident ($swap:literal, $g:ident, $k:literal) |$x:ident, $y:ident| $cond:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = &reg![insn.a()];
                if !v.is_float() {
                    tail!($g)
                }
                let f = v.read_float();
                let k = insn.cmp_imm_int() as f64;
                let ($x, $y) = if $swap { (k, f) } else { (f, k) };
                branch!(($cond) == $k, insn.branch_offset())
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // Constant-key reads through the IC: `$recv` is the receiver operand
    // (`reg`, `upval`), `$kind` the cached entry's kind, `$self_` writes the
    // receiver to `R[a + 4]` (SELF).
    (get $rows:ident, slow = $slow:ident;
     $($OP:ident = $name:ident ($recv:ident, $kind:ident, $self_:literal)),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, b, ic_idx, _) = insn.abde();
                let recv_val = receiver!($recv, b);
                let Some(t) = recv_val.get_table() else {
                    tail!($slow)
                };
                let cache = $crate::vm::ops::field::read_ic(closure, ic_idx);
                let v = $crate::vm::ops::family!(@load $kind, $slow, cache, t, recv_val, insn, dst, $self_);
                if $self_ {
                    reg![dst + 4] = recv_val;
                }
                reg![dst] = v;
                next!()
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // Constant-key writes through the IC, as `get`; the barrier is the retry
    // form, so the handlers need no stack frame.
    (set $rows:ident, slow = $slow:ident;
     $($OP:ident = $name:ident ($recv:ident, $kind:ident)),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (src, b, ic_idx, _) = insn.abde();
                let recv_val = receiver!($recv, b);
                let Some(t) = recv_val.get_table() else {
                    tail!($slow)
                };
                let cache = $crate::vm::ops::field::read_ic(closure, ic_idx);
                $crate::vm::ops::family!(@store $kind, $slow, insn, cache, t, recv_val, src);
                next!()
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // Own and Absent entries only change through `fill_ic`, which rewrites
    // the site, so the form implies the entry's kind; the collector may
    // empty a ProtoLoad one, so that kind is checked.
    (@load Inl, $slow:ident, $cache:expr, $t:ident, $recv:ident, $insn:ident, $dst:ident, $self_:literal) => {{
        let $crate::env::function::InlineCache::Own { shape, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        if !$crate::env::shape::Shape::ptr_eq($t.shape(), shape) {
            tail!($slow)
        }
        let v = unsafe { $t.load_inline(loc) };
        // A nil own slot answers unless the metatable gained `__index`.
        if v.is_nil() && shape.has_mm($crate::env::MetamethodBits::INDEX) {
            tail!($slow)
        }
        v
    }};
    (@load Aux, $slow:ident, $cache:expr, $t:ident, $recv:ident, $insn:ident, $dst:ident, $self_:literal) => {{
        let $crate::env::function::InlineCache::Own { shape, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        let state = $t.inner().borrow();
        if !$crate::env::shape::Shape::ptr_eq(state.shape(), shape) {
            drop(state);
            tail!($slow)
        }
        let v = unsafe { $t.load_aux(&state, loc) };
        drop(state);
        if v.is_nil() && shape.has_mm($crate::env::MetamethodBits::INDEX) {
            tail!($slow)
        }
        v
    }};
    (@load Absent, $slow:ident, $cache:expr, $t:ident, $recv:ident, $insn:ident, $dst:ident, $self_:literal) => {{
        let $crate::env::function::InlineCache::Absent { shape } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        if !$crate::env::shape::Shape::ptr_eq($t.shape(), shape) {
            tail!($slow)
        }
        if shape.has_mm($crate::env::MetamethodBits::INDEX) {
            // An `__index` function is called from here, as `$slow` would.
            if let Some(mt) = shape.mt_cache()
                && let index = mt.mm($crate::env::MetamethodBits::INDEX)
                && index.get_function().is_some()
            {
                if $self_ {
                    reg![$dst + 4] = $recv;
                }
                call_mm!($crate::vm::ops::meta::ret_store_a, index, [$recv, k![$insn.e()]])
            }
            tail!($slow)
        }
        $crate::env::value::Value::nil()
    }};
    (@load Proto, $slow:ident, $cache:expr, $t:ident, $recv:ident, $insn:ident, $dst:ident, $self_:literal) => {{
        let $crate::env::function::InlineCache::ProtoLoad {
            recv,
            holder,
            holder_shape,
            loc,
        } = $cache
        else {
            tail!($slow)
        };
        let live = $t.shape();
        if !$crate::env::shape::Shape::ptr_eq(live, recv) {
            tail!($slow)
        }
        // `recv` was filled with a metatable whose `__index` was `holder`.
        let mt = unsafe { live.mt_cache().unwrap_unchecked() };
        if mt.index_table() != holder.as_ptr() as usize {
            tail!($slow)
        }
        // SAFETY: `__index` is still `holder`, so the receiver keeps it alive.
        let holder = $crate::env::table::Table::from_inner(unsafe { $crate::dmm::Gc::from_ptr(holder.as_ptr()) });
        let h = holder.inner().borrow();
        if !$crate::env::shape::Shape::ptr_eq(h.shape(), holder_shape) {
            drop(h);
            tail!($slow)
        }
        let v = unsafe { holder.load(&h, loc) };
        drop(h);
        // A nil slot means the walk goes on past `holder`.
        if v.is_nil() {
            tail!($slow)
        }
        v
    }};
    (@store Inl, $slow:ident, $insn:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        let $crate::env::function::InlineCache::Own { shape, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        if !$crate::env::shape::Shape::ptr_eq($t.shape(), shape) {
            tail!($slow)
        }
        // `__newindex` fires only on currently-nil keys.
        let existing = unsafe { $t.load_inline(loc) };
        if existing.is_nil() && shape.has_mm($crate::env::MetamethodBits::NEWINDEX) {
            tail!($slow)
        }
        barrier!($t.inner());
        unsafe { $t.store_inline(loc, reg![$src]) };
    };
    (@store Aux, $slow:ident, $insn:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        let $crate::env::function::InlineCache::Own { shape, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        let state = $t.inner().borrow();
        if !$crate::env::shape::Shape::ptr_eq(state.shape(), shape) {
            drop(state);
            tail!($slow)
        }
        let slot = unsafe { state.aux_slot(loc) };
        drop(state);
        let existing = unsafe { *slot };
        if existing.is_nil() && shape.has_mm($crate::env::MetamethodBits::NEWINDEX) {
            tail!($slow)
        }
        barrier!($t.inner());
        // SAFETY: the shape still matched, so the spill cell holds `loc`.
        unsafe { *slot = reg![$src] };
    };
    (@store Trans, $slow:ident, $insn:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        let $crate::env::function::InlineCache::Transition { from, to, loc } = $cache else {
            unsafe { std::hint::unreachable_unchecked() }
        };
        let state = $t.inner().borrow();
        if !$crate::env::shape::Shape::ptr_eq(state.shape(), from) || from.has_mm($crate::env::MetamethodBits::NEWINDEX) {
            drop(state);
            tail!($slow)
        }
        let v = reg![$src];
        // Storing nil to an absent key adds nothing.
        if v.is_nil() {
            drop(state);
            next!()
        }
        if !state.has_room(loc) {
            drop(state);
            tail!($slow)
        }
        drop(state);
        barrier!($t.inner());
        let mut state = unsafe { $t.borrow_mut_barriered() };
        // SAFETY: the live shape is `from`, and there is room.
        unsafe { $t.push(&mut state, to, loc, v) };
        drop(state);
    };
    (@store Absent, $slow:ident, $insn:ident, $cache:ident, $t:ident, $recv:ident, $src:ident) => {
        // Filled for a shape with `__newindex`, which a function may answer
        // from here.
        if let $crate::env::function::InlineCache::Absent { shape } = $cache
            && $crate::env::shape::Shape::ptr_eq($t.shape(), shape)
            && let Some(mt) = shape.mt_cache()
            && let newindex = mt.mm($crate::env::MetamethodBits::NEWINDEX)
            && newindex.get_function().is_some()
        {
            call_mm!($crate::vm::ops::meta::ret_discard, newindex, [$recv, k![$insn.e()], reg![$src]])
        }
        tail!($slow)
    };
    (@store $dst:ident, $body:expr) => {
        match $body {
            $crate::vm::ops::arith::Out::Value(v) => {
                reg![$dst] = v;
                next!()
            }
            $crate::vm::ops::arith::Out::Float(f) => {
                reg![$dst].write_float_unchecked(f);
                next!()
            }
            $crate::vm::ops::arith::Out::FloatChecked(f) => {
                reg![$dst].write_float(f);
                next!()
            }
            $crate::vm::ops::arith::Out::Miss => {}
        }
    };
}
pub(crate) use family;
