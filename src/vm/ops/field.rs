//! Constant-key table access (GETFIELD, SETFIELD, GETTABUP, SETTABUP, SELF)
//! with its inline caches and quickened forms, and the shared slow paths of
//! every table read and write.

use crate::dmm::Gc;
use crate::env::MetamethodBits;
use crate::env::function::{InlineCache, LuaFn};
use crate::env::shape::{MAX_PROPERTIES_FAST, Shape, mirrored};
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, Table, TableState};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::lua::Context;
use crate::vm::abi::{Slot, handler};
use crate::vm::ops::family;
use crate::vm::ops::meta::{
    IndexChain, NewIndexChain, ret_discard, ret_store_a, stage_mm, walk_index_chain,
    walk_newindex_chain,
};
use crate::vm::ops::table::check_index_key;
use crate::vm::unwind::OpError;

// ---------------------------------------------------------------------------
// Inline cache helpers
// ---------------------------------------------------------------------------

/// The IC entry of the current site.
#[inline(always)]
pub(crate) fn read_ic<'gc>(closure: LuaFn<'gc>, ic_idx: u16) -> InlineCache<'gc> {
    debug_assert!((ic_idx as usize) < closure.proto.ic_table.len());
    unsafe { (*closure.ic_table.add(ic_idx as usize)).get() }
}

/// Refill the IC entry after a full lookup, and rewrite the site.
#[inline(always)]
fn fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    site: *const Instruction,
    entry: InlineCache<'gc>,
) {
    // Every dict table with the same metatable shares one sentinel shape,
    // so an entry on it would answer for keys it never saw.
    debug_assert!(match entry {
        InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => !shape.is_dict(),
        InlineCache::Transition { from, to, .. } => !from.is_dict() && !to.is_dict(),
        InlineCache::ProtoLoad {
            recv, holder_shape, ..
        } => !recv.is_dict() && !holder_shape.is_dict(),
        InlineCache::Empty => true,
    });
    let proto_gc = closure.proto;
    if let Some(slot_lock) = proto_gc.ic_table.get(ic_idx as usize) {
        // A fresh `Shape` pointer is adopted through this slot, so the
        // prototype gets its barrier before the write through `as_cell`.
        ctx.mutation().backward_barrier(Gc::erase(proto_gc), None);
        let first = matches!(slot_lock.get(), InlineCache::Empty);
        unsafe { slot_lock.as_cell() }.set(entry);
        quicken(site, &entry, first);
    }
}

/// Refills a site takes before it counts as megamorphic and stops refilling.
const MEGAMORPHIC: u8 = 16;

/// Whether a miss at `insn` should refill its cache: not once it has been
/// refilled [`MEGAMORPHIC`] times, which its unused `c` slot counts.
#[inline(always)]
fn site_fills(insn: Instruction) -> bool {
    insn.c() < MEGAMORPHIC
}

/// Rewrite the site to the form for its cache's new contents, and count a
/// refill. Every fill rewrites: the form must match the entry's kind, and the
/// generic handler is the full lookup, so a site that refills (an `Absent`
/// entry becoming a `ProtoLoad` on the `__index` walk, a shape change) takes
/// its new form until the refill count stops the fills.
#[inline]
fn quicken(site: *const Instruction, entry: &InlineCache<'_>, first: bool) {
    let mut insn = unsafe { *site };
    if !first {
        insn.set_c(insn.c().saturating_add(1));
    }
    use InlineCache::*;
    let own = |inl: Op, aux: Op, loc: &SlotLoc| if loc.spilled() { aux } else { inl };
    let op = match (insn.op().unquickened(), entry) {
        (Op::GETFIELD, Own { loc, .. }) => own(Op::GETFIELD_INL, Op::GETFIELD_AUX, loc),
        (Op::GETFIELD, Absent { .. }) => Op::GETFIELD_ABSENT,
        (Op::GETFIELD, ProtoLoad { .. }) => Op::GETFIELD_PROTO,
        (Op::GETTABUP, Own { loc, .. }) => own(Op::GETTABUP_INL, Op::GETTABUP_AUX, loc),
        (Op::GETTABUP, Absent { .. }) => Op::GETTABUP_ABSENT,
        (Op::GETTABUP, ProtoLoad { .. }) => Op::GETTABUP_PROTO,
        (Op::SELF, Own { loc, .. }) => own(Op::SELF_INL, Op::SELF_AUX, loc),
        (Op::SELF, Absent { .. }) => Op::SELF_ABSENT,
        (Op::SELF, ProtoLoad { .. }) => Op::SELF_PROTO,
        (Op::SETFIELD, Own { loc, .. }) => own(Op::SETFIELD_INL, Op::SETFIELD_AUX, loc),
        (Op::SETFIELD, Transition { .. }) => Op::SETFIELD_TRANS,
        (Op::SETFIELD, Absent { .. }) => Op::SETFIELD_ABSENT,
        (Op::SETTABUP, Own { loc, .. }) => own(Op::SETTABUP_INL, Op::SETTABUP_AUX, loc),
        (Op::SETTABUP, Transition { .. }) => Op::SETTABUP_TRANS,
        (Op::SETTABUP, Absent { .. }) => Op::SETTABUP_ABSENT,
        (op, _) => op,
    };
    // SAFETY: `Code` keeps instructions in cells, and `site` came from one.
    unsafe { site.cast_mut().write(insn.with_op(op)) };
}

/// The entry for a lookup of a key in `shape` that found `slot`.
#[inline(always)]
fn shape_entry<'gc>(shape: Shape<'gc>, slot: Option<u32>) -> InlineCache<'gc> {
    match slot {
        Some(slot) => InlineCache::Own {
            shape,
            loc: SlotLoc::new(shape, slot),
        },
        None => InlineCache::Absent { shape },
    }
}

/// The cached result of a constant-key load from `t`, or `None` when the
/// slow path must run.
#[inline(always)]
fn ic_get<'gc>(
    cache: InlineCache<'gc>,
    t: Table<'gc>,
    state: &TableState<'gc>,
) -> Option<Value<'gc>> {
    let live = state.shape();
    let v = if std::hint::likely(matches!(cache, InlineCache::Own { .. }))
        && let InlineCache::Own { shape, loc } = cache
        && Shape::ptr_eq(live, shape)
    {
        unsafe { t.load(state, loc) }
    } else if let InlineCache::Absent { shape } = cache
        && Shape::ptr_eq(live, shape)
    {
        Value::nil()
    } else if let InlineCache::ProtoLoad {
        recv,
        holder,
        holder_shape,
        loc,
    } = cache
        && Shape::ptr_eq(live, recv)
    {
        // `recv` was filled with a metatable whose `__index` was `holder`.
        let mt = unsafe { live.mt_cache().unwrap_unchecked() };
        if mt.index_table() != holder.as_ptr() as usize {
            return None;
        }
        // SAFETY: `__index` is still `holder`, so the receiver keeps it alive.
        let holder = Table::from_inner(unsafe { Gc::from_ptr(holder.as_ptr()) });
        let h = holder.inner().borrow();
        if !Shape::ptr_eq(h.shape(), holder_shape) {
            return None;
        }
        // A nil slot means the walk goes on past `holder`.
        let v = unsafe { holder.load(&h, loc) };
        return (!v.is_nil()).then_some(v);
    } else {
        return None;
    };
    (!(v.is_nil() && live.has_mm(MetamethodBits::INDEX))).then_some(v)
}

/// Store `v` through the entry for a constant-key store to `t`. Returns
/// false, having stored nothing, when the slow path must run.
#[inline(always)]
fn ic_set<'gc>(ctx: Context<'gc>, cache: InlineCache<'gc>, t: Table<'gc>, v: Value<'gc>) -> bool {
    let state = t.inner().borrow();
    let live = state.shape();
    if std::hint::likely(matches!(cache, InlineCache::Own { .. }))
        && let InlineCache::Own { shape, loc } = cache
        && Shape::ptr_eq(live, shape)
    {
        // `__newindex` fires only on currently-nil keys.
        let existing = unsafe { t.load(&state, loc) };
        if existing.is_nil() && live.has_mm(MetamethodBits::NEWINDEX) {
            return false;
        }
        // Barrier work goes to the slow path, keeping calls out of this handler.
        drop(state);
        let Some(w) = Gc::write_if_clean(ctx.mutation(), t.inner()) else {
            return false;
        };
        let state = w.unlock().borrow();
        unsafe { t.store(&state, loc, v) };
        return true;
    }
    if let InlineCache::Transition { from, to, loc } = cache
        && Shape::ptr_eq(live, from)
    {
        if live.has_mm(MetamethodBits::NEWINDEX) {
            return false;
        }
        // Storing nil to an absent key adds nothing.
        if v.is_nil() {
            return true;
        }
        if !state.has_room(loc) {
            return false;
        }
        drop(state);
        let Some(w) = Gc::write_if_clean(ctx.mutation(), t.inner()) else {
            return false;
        };
        let mut state = w.unlock().borrow_mut();
        // SAFETY: the live shape is `from`, and there is room.
        unsafe { t.push(&mut state, to, loc, v) };
        return true;
    }
    false
}

/// IC sites only carry constant string keys.
#[inline(always)]
fn constant_key<'gc>(k: Value<'gc>) -> LuaString<'gc> {
    debug_assert!(k.get_string().is_some(), "IC site with a non-string key");
    unsafe { k.get_string().unwrap_unchecked() }
}

/// `t[k]` for a constant-key IC miss, resolved as far as an entry can cache
/// it. Returns the value, or the receiver to continue the `__index` walk
/// from, its own raw lookup having missed.
#[inline(always)]
fn get_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    site: *const Instruction,
    t: Table<'gc>,
    k: Value<'gc>,
) -> Result<Value<'gc>, Value<'gc>> {
    let state = t.inner().borrow();
    let shape = state.shape();
    if shape.is_dict() {
        let v = state.raw_get(k);
        return if v.is_nil() && shape.has_mm(MetamethodBits::INDEX) {
            Err(Value::table(t))
        } else {
            Ok(v)
        };
    }
    // The entry already says `t` lacks the key: only `__index` is left.
    if let InlineCache::Absent { shape: cached } = read_ic(closure, ic_idx)
        && Shape::ptr_eq(cached, shape)
        && let Some(mt) = shape.mt_cache()
    {
        let index = mt.mm(MetamethodBits::INDEX);
        drop(state);
        if index.get_function().is_some() {
            return Err(Value::table(t));
        }
        return get_index_fill_ic(ctx, closure, ic_idx, site, t, index, Some(shape), k);
    }
    let slot = shape.find_slot(constant_key(k));
    if site_fills(unsafe { *site }) {
        fill_ic(ctx, closure, ic_idx, site, shape_entry(shape, slot));
    }
    let v = slot.map_or(Value::nil(), |s| state.named_get(s));
    if !v.is_nil() || !shape.has_mm(MetamethodBits::INDEX) {
        return Ok(v);
    }
    let index = unsafe { shape.mt_cache().unwrap_unchecked() }.mm(MetamethodBits::INDEX);
    drop(state);
    let recv = slot.is_none().then_some(shape);
    get_index_fill_ic(ctx, closure, ic_idx, site, t, index, recv, k)
}

/// The `__index` half of [`get_fill_ic`]: `t`'s raw lookup missed and its
/// metatable's `__index` is `index`. `recv` is `t`'s shape when that shape
/// lacks the key, which a hit in an `__index` table can be cached against.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn get_index_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    site: *const Instruction,
    t: Table<'gc>,
    index: Value<'gc>,
    recv: Option<Shape<'gc>>,
    k: Value<'gc>,
) -> Result<Value<'gc>, Value<'gc>> {
    let Some(holder) = index.get_table() else {
        // A function is called with `t`; anything else is indexed in turn.
        return Err(if index.get_function().is_some() {
            Value::table(t)
        } else {
            index
        });
    };
    let h = holder.inner().borrow();
    let holder_shape = h.shape();
    if holder_shape.is_dict() {
        let v = h.raw_get(k);
        return if v.is_nil() { Err(index) } else { Ok(v) };
    }
    let Some(holder_slot) = holder_shape.find_slot(constant_key(k)) else {
        return Err(index);
    };
    let v = h.named_get(holder_slot);
    if v.is_nil() {
        return Err(index);
    }
    if let Some(recv) = recv {
        debug_assert_eq!(
            unsafe { recv.mt_cache().unwrap_unchecked() }.index_table(),
            Gc::as_ptr(holder.inner()) as usize
        );
        let entry = InlineCache::ProtoLoad {
            recv,
            holder: Gc::downgrade(holder.inner()),
            holder_shape,
            loc: SlotLoc::new(holder_shape, holder_slot),
        };
        if site_fills(unsafe { *site }) {
            fill_ic(ctx, closure, ic_idx, site, entry);
        }
    }
    Ok(v)
}

/// `t[k] = v` for a constant-key IC miss, with one slot lookup that both
/// stores and refills the entry. Returns false, having stored nothing, when
/// `__newindex` may fire.
#[inline(always)]
fn set_own_fill_ic<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    ic_idx: u16,
    site: *const Instruction,
    t: Table<'gc>,
    k: Value<'gc>,
    v: Value<'gc>,
) -> bool {
    let state = t.inner().borrow();
    let shape = state.shape();
    let newindex = shape.has_mm(MetamethodBits::NEWINDEX);
    if shape.is_dict() {
        if newindex {
            return false;
        }
        drop(state);
        t.raw_set(ctx, k, v);
        return true;
    }
    // As in `get_fill_ic`: the entry already says only `__newindex` is left.
    if newindex
        && let InlineCache::Absent { shape: cached } = read_ic(closure, ic_idx)
        && Shape::ptr_eq(cached, shape)
    {
        return false;
    }
    let key = constant_key(k);
    // A hit stores without telling a metatable's cache, so the keys it
    // mirrors are left uncached.
    let cache = !mirrored(key);
    let slot = shape.find_slot(key);
    let existing = slot.map_or(Value::nil(), |s| state.named_get(s));
    if existing.is_nil() && newindex {
        if cache && site_fills(unsafe { *site }) {
            fill_ic(ctx, closure, ic_idx, site, shape_entry(shape, slot));
        }
        return false;
    }
    drop(state);
    let mut state = t.inner().borrow_mut(ctx.mutation());
    let entry = match slot {
        Some(slot) => {
            state.named_set(slot, v);
            state.maybe_update_mt_bit(ctx.mutation(), k, v);
            shape_entry(shape, Some(slot))
        }
        None => {
            state.add_string_key(ctx, key, v, MAX_PROPERTIES_FAST);
            let to = state.shape();
            // A nil store adds nothing; past the cap the table went dict.
            if Shape::ptr_eq(to, shape) || to.is_dict() {
                InlineCache::Absent { shape }
            } else {
                let loc = SlotLoc::new(to, shape.slot_count());
                InlineCache::Transition {
                    from: shape,
                    to,
                    loc,
                }
            }
        }
    };
    drop(state);
    if cache && site_fills(unsafe { *site }) {
        fill_ic(ctx, closure, ic_idx, site, entry);
    }
    true
}

/// The `__index` walk for `R[dst] = recv[k]`, the receiver's own raw lookup
/// (if a table) having missed.
macro_rules! index_chain {
    ($pc:ident, $base:ident, $rt:ident, $recv:expr, $k:expr, $dst:expr) => {{
        let recv: Value<'gc> = $recv;
        let k: Value<'gc> = $k;
        let dst: u8 = $dst;
        match walk_index_chain($rt, recv, k) {
            IndexChain::Resolved(v) => {
                reg![dst] = v;
                next!()
            }
            IndexChain::Invoke { func, receiver } => {
                stage_mm!(
                    $pc,
                    $base,
                    $rt,
                    ret_store_a,
                    Value::function(func),
                    [receiver, k]
                )
            }
            IndexChain::NotIndexable(v) => raise!(OpError::Index(v)),
            IndexChain::Exhausted => raise!(OpError::IndexChainLoop),
        }
    }};
}

/// `recv[k] = v` on any receiver, through `__newindex`.
macro_rules! newindex_chain {
    ($pc:ident, $base:ident, $rt:ident, $recv:expr, $k:expr, $v:expr, $raw_set:ident) => {{
        let k: Value<'gc> = $k;
        let v: Value<'gc> = $v;
        match walk_newindex_chain($rt, $recv, k) {
            NewIndexChain::RawSet(target) => {
                check_index_key!(k);
                target.$raw_set($rt, k, v);
                next!()
            }
            NewIndexChain::Invoke { func, receiver } => {
                stage_mm!(
                    $pc,
                    $base,
                    $rt,
                    ret_discard,
                    Value::function(func),
                    [receiver, k, v]
                )
            }
            NewIndexChain::NotIndexable(v) => raise!(OpError::Index(v)),
            NewIndexChain::Exhausted => raise!(OpError::NewIndexChainLoop),
        }
    }};
}

/// The receiver operand `b` of a constant-key access, by opcode: a register,
/// a by-value upvalue, or a shared-cell upvalue.
macro_rules! receiver {
    (reg, $b:expr) => {
        reg![$b]
    };
    (upval, $b:expr) => {
        upval!(value $b)
    };
    (cell, $b:expr) => {
        upval!(cell $b).get()
    };
}

family! {
    get GET_ROWS, slow = get_slow;
    GETFIELD_INL = getfield_inl (reg, Inl, false),
    GETFIELD_AUX = getfield_aux (reg, Aux, false),
    GETFIELD_ABSENT = getfield_absent (reg, Absent, false),
    GETFIELD_PROTO = getfield_proto (reg, Proto, false),
    GETTABUP_INL = gettabup_inl (upval, Inl, false),
    GETTABUP_AUX = gettabup_aux (upval, Aux, false),
    GETTABUP_ABSENT = gettabup_absent (upval, Absent, false),
    GETTABUP_PROTO = gettabup_proto (upval, Proto, false),
    SELF_INL = self_inl (reg, Inl, true),
    SELF_AUX = self_aux (reg, Aux, true),
    SELF_ABSENT = self_absent (reg, Absent, true),
    SELF_PROTO = self_proto (reg, Proto, true),
}

family! {
    set SET_ROWS, slow = set_slow;
    SETFIELD_INL = setfield_inl (reg, Inl),
    SETFIELD_AUX = setfield_aux (reg, Aux),
    SETFIELD_TRANS = setfield_trans (reg, Trans),
    SETFIELD_ABSENT = setfield_absent (reg, Absent),
    SETTABUP_INL = settabup_inl (upval, Inl),
    SETTABUP_AUX = settabup_aux (upval, Aux),
    SETTABUP_TRANS = settabup_trans (upval, Trans),
    SETTABUP_ABSENT = settabup_absent (upval, Absent),
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// The shared-cell upvalue read (`GETTABUP_REF`, never specialized):
    /// `R[a] = recv[K[e]]` through the IC, whatever its entry.
    op fn op_gettabup_ref {
        let (dst, b, ic_idx, _) = insn.abde();
        let recv_val = receiver!(cell, b);
        let Some(t) = recv_val.get_table() else {
            tail!(get_slow)
        };
        let state = t.inner().borrow();
        if let Some(v) = ic_get(read_ic(closure, ic_idx), t, &state) {
            drop(state);
            reg![dst] = v;
            next!()
        }
        drop(state);
        tail!(get_slow)
    }

    /// The shared-cell upvalue write (`SETTABUP_REF`, never specialized).
    op fn op_settabup_ref {
        let (src, b, ic_idx, _) = insn.abde();
        let Some(t) = receiver!(cell, b).get_table() else {
            tail!(set_slow)
        };
        if ic_set(rt, read_ic(closure, ic_idx), t, reg![src]) {
            next!()
        }
        tail!(set_slow)
    }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// The slow path of the table reads (GETTABUP, GETTABLE, GETFIELD, SELF):
    /// a receiver that isn't a table, a cache miss, a key the table lacks, and
    /// `__index`. The opcode says where the receiver and key are.
    slow fn get_slow {
        let insn = insn_at!();
        let closure: LuaFn<'gc> = unsafe { crate::vm::frame::closure(base) };
        let op = insn.op().unquickened();
        let dst = insn.a();
        let recv = match op {
            Op::GETTABUP => unsafe { (*closure.upvalue_ptr().add(insn.b() as usize)).value },
            Op::GETTABUP_REF => unsafe { (*closure.upvalue_ptr().add(insn.b() as usize)).cell }.get(),
            _ => reg![insn.b()],
        };
        if op == Op::SELF {
            reg![dst + 4] = recv;
        }
        if op == Op::GETTABLE {
            let k = reg![insn.c()];
            if let Some(t) = recv.get_table() {
                let v = t.raw_get(k);
                if !v.is_nil() {
                    reg![dst] = v;
                    next!()
                }
            }
            index_chain!(pc, base, rt, recv, k, dst)
        }
        let k = unsafe { *closure.constants.add(insn.e() as usize) };
        let Some(t) = recv.get_table() else {
            index_chain!(pc, base, rt, recv, k, dst)
        };
        match get_fill_ic(rt, closure, insn.d(), unsafe { pc.sub(1) }, t, k) {
            Ok(v) => {
                reg![dst] = v;
                next!()
            }
            Err(from) => index_chain!(pc, base, rt, from, k, dst),
        }
    }

    /// The slow path of the table writes (SETTABUP, SETTABLE, SETFIELD).
    slow fn set_slow {
        let insn = insn_at!();
        let closure: LuaFn<'gc> = unsafe { crate::vm::frame::closure(base) };
        let op = insn.op().unquickened();
        let v = reg![insn.a()];
        let recv = match op {
            Op::SETTABUP => unsafe { (*closure.upvalue_ptr().add(insn.b() as usize)).value },
            Op::SETTABUP_REF => unsafe { (*closure.upvalue_ptr().add(insn.b() as usize)).cell }.get(),
            _ => reg![insn.b()],
        };
        if op == Op::SETTABLE {
            let k = reg![insn.c()];
            newindex_chain!(pc, base, rt, recv, k, v, raw_set_keyed)
        }
        let k = unsafe { *closure.constants.add(insn.e() as usize) };
        if let Some(t) = recv.get_table()
            && set_own_fill_ic(rt, closure, insn.d(), unsafe { pc.sub(1) }, t, k, v)
        {
            next!()
        }
        newindex_chain!(pc, base, rt, recv, k, v, raw_set)
    }

    /// A store handler's object was not gray: run its backward
    /// barrier here, out of line, and run the store again.
    slow fn barrier_retry {
        unsafe { rt.mutation().backward_barrier_erased(closure.raw() as *const ()) };
        closure = Slot::closure(unsafe { crate::vm::frame::closure(base) });
        jump_by!(-1);
        next!()
    }
}
