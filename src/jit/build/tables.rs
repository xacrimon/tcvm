//! Table access (`jit-design.md` 7.4, 7.6): constant-key sites from their
//! inline caches, integer keys through the array part, everything else
//! through `jit_index` and `jit_newindex`, whose failure (a metamethod to
//! call, an error) deopts.

use crate::dmm::Gc;
use crate::env::function::InlineCache;
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, Table};
use crate::env::value::Value;
use crate::env::{Function, MetamethodBits, MtCache, Shape};
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::emit::{
    arm_end, branch_arm, call_arm_end, calls_mm, cold, enter_join, mm_call,
};
use crate::jit::build::{Builder, Shapes};
use crate::jit::deps;
use crate::jit::feedback as fb;
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL_VAL, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, Val};
use crate::vm::ops::field::read_ic;

/// The receiver of a constant-key site: a register, a by-value upvalue, or
/// a shared cell's value.
fn receiver(b: &mut Builder<'_, '_>, insn: Instruction) -> Val {
    match insn.op().unquickened() {
        BcOp::GETTABUP | BcOp::SETTABUP if b.inline.is_some() => {
            match b.inline.as_ref().and_then(|i| i.closure) {
                Some(c) => b.ins(Op::UpvalValueOf(insn.b()), &[c]),
                None => {
                    let now = unsafe { (*b.closure.upvalue_ptr().add(insn.b() as usize)).value };
                    b.konst(now)
                }
            }
        }
        BcOp::GETTABUP | BcOp::SETTABUP => b.ins(Op::UpvalValue(insn.b()), &[]),
        BcOp::GETTABUP_REF | BcOp::SETTABUP_REF => {
            let c = b.ins(Op::UpvalCell(insn.b()), &[]);
            b.ins(Op::CellLoad, &[c])
        }
        _ => b.reg(insn.b()),
    }
}

/// The lock of the table `v` is, guarded to be one.
fn table_of(b: &mut Builder<'_, '_>, v: Val) -> Val {
    let g = b.guard(v, TypeSet::TAB);
    b.ins(Op::Unbox(Rep::Ptr), &[g])
}

/// `t` guarded to have `shape`, unless it is known to have it already.
fn guard_shape<'gc>(b: &mut Builder<'_, 'gc>, t: Val, shape: Shape<'gc>) {
    let k = b.intern_shape(shape);
    if b.known_shapes(t) == Some(Shapes::one(k)) {
        return;
    }
    b.ins_exit(Op::GuardShape(k), &[t], ExitTag::Shape);
    b.set_shapes(t, Shapes::one(k));
}

/// What an access does for tables of a shape: shapes with equal ways are
/// served by one arm under one shape-set test (V8's merged property
/// accesses).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Way {
    /// A load of the own slot at this location, and what a nil there walks
    /// to.
    Own(u32, FoundKey),
    /// A store to the own slot at this location, and the class when its
    /// `__newindex` may fire (0 when it may not).
    Store(u32, u64),
    /// Absent: what the walk finds, unless it is `Live`.
    Walk(FoundKey),
    /// The slot at `loc` of an `__index` table of a shape, for this class
    /// read live.
    Proto(u64, u64, u32, u64),
    /// Absent: nil unless this class, read live, has `__index`.
    Absent(u64),
    /// The generic helper.
    Helper,
    /// Its own arm.
    Unique(usize),
}

/// Classes and holders an access walks at compile time.
const MAX_WALK: usize = 8;

/// What the `__index` walk gives for a key that a table of some class lacks
/// or holds nil, as far as the region may assume it; each assumption is a
/// dependency (ws2 §5.5).
#[derive(Clone, Copy)]
enum Found<'gc> {
    /// Nil: no `__index` past the tables without the key.
    Nil,
    /// A slot of a watched holder that no store changed.
    Const(Value<'gc>),
    /// The slot at `loc` of a holder: a watched one's whose value may change,
    /// or an unwatched one's while it has this shape. Nil walks on.
    Slot(Table<'gc>, Option<Shape<'gc>>, SlotLoc),
    /// The `__index` function, called with the receiver, or with the holder
    /// it is reached from.
    Call(Function<'gc>, Option<Table<'gc>>),
    /// Past what the region may assume: the generic walk.
    Helper,
    /// The receiver's class may change (volatile or stale): read live.
    Live(MtCache<'gc>),
}

/// What makes walks the same, for merged arms (`Found::key`).
type FoundKey = (u8, u64, u64, u64);

impl Found<'_> {
    fn key(self) -> FoundKey {
        let t = |t: Table<'_>| Gc::as_ptr(t.inner()) as u64;
        match self {
            Found::Nil => (0, 0, 0, 0),
            Found::Const(v) => (1, v.to_raw(), 0, 0),
            Found::Slot(h, s, loc) => (2, t(h), s.map_or(0, |s| s.as_word()), loc.raw() as u64),
            Found::Call(f, on) => (3, Value::function(f).to_raw(), on.map_or(0, t), 0),
            Found::Helper => (4, 0, 0, 0),
            Found::Live(c) => (5, c.as_word(), 0, 0),
        }
    }
}

/// The `__index` walk for `key` from a table of `class` that lacks it or
/// holds nil: through `__index` tables that are watched holders, their
/// shapes and unchanged slots assumed (ws2 §5.5).
fn walk<'gc>(
    b: &mut Builder<'_, 'gc>,
    class: Option<MtCache<'gc>>,
    key: LuaString<'gc>,
) -> Found<'gc> {
    let Some(mut class) = class else {
        return Found::Nil;
    };
    if class.is_stale() || class.is_volatile() {
        return Found::Live(class);
    }
    let mut on = None;
    for _ in 0..MAX_WALK {
        b.depend_class(class, MetamethodBits::INDEX);
        let index = class.mm(MetamethodBits::INDEX);
        if index.is_nil() {
            return Found::Nil;
        }
        if let Some(f) = index.get_function() {
            return Found::Call(f, on);
        }
        let Some(h) = index.get_table() else {
            return Found::Helper;
        };
        let hs = h.shape();
        if hs.is_dict() {
            return Found::Helper;
        }
        let slot = hs.find_slot(key);
        if !hs.is_watched() || hs.is_volatile_holder() {
            return match slot {
                Some(s) => Found::Slot(h, Some(hs), SlotLoc::new(hs, s)),
                None => Found::Helper,
            };
        }
        b.depend_holder(hs, deps::SHAPE_GROUP);
        if let Some(s) = slot {
            if hs.slot_mutable(s) {
                return Found::Slot(h, None, SlotLoc::new(hs, s));
            }
            b.depend_holder(hs, deps::slot_group(s));
            let v = h.inner().borrow().named_get(s);
            if !v.is_nil() {
                return Found::Const(v);
            }
        }
        match hs.mt_cache() {
            None => return Found::Nil,
            Some(c) if c.is_stale() || c.is_volatile() => return Found::Helper,
            Some(c) => (class, on) = (c, Some(h)),
        }
    }
    Found::Helper
}

/// `dst` = what `found` (not `Live`) gives the read at `site`.
fn emit_found<'gc>(b: &mut Builder<'_, 'gc>, dst: u8, found: Found<'gc>, site: &Site<'gc>) -> Val {
    match found {
        Found::Nil => b.konst(Value::nil()),
        Found::Const(v) => b.konst(v),
        Found::Slot(h, guard, loc) => {
            let hv = b.konst(Value::table(h));
            let hp = b.ins(Op::Unbox(Rep::Ptr), &[hv]);
            if let Some(hs) = guard {
                guard_shape(b, hp, hs);
            }
            let v = b.ins(
                Op::TabLoad {
                    loc: loc.raw(),
                    key: site.kid,
                },
                &[hp],
            );
            b.field_loads.insert(v, (hv, site.kv));
            let join = b.new_block();
            let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
            let present = branch_arm(b, nil);
            cold(b, |b| {
                let w = index(b, dst, site.recv, site.key);
                arm_end(b, join, dst, w);
            });
            b.switch_to(present);
            arm_end(b, join, dst, v);
            enter_join(b, join);
            b.reg(dst)
        }
        Found::Call(f, on) if calls_mm(b) => {
            let fv = b.konst(Value::function(f));
            let this = match on {
                Some(t) => b.konst(Value::table(t)),
                None => site.recv,
            };
            let join = b.new_block();
            let w = mm_call(b, fv, &[this, site.key]);
            call_arm_end(b, join, Some((dst, w)));
            enter_join(b, join);
            b.reg(dst)
        }
        Found::Call(..) | Found::Helper | Found::Live(_) => index(b, dst, site.recv, site.key),
    }
}

/// The class `shape` pins, as a word, 0 without one.
fn class_word(shape: Shape<'_>) -> u64 {
    shape.mt_cache().map_or(0, |c| c.as_word())
}

/// `t` guarded to one of `ks`, unless it is known to have one of them.
fn guard_shapes(b: &mut Builder<'_, '_>, t: Val, ks: &[u32]) {
    if b.known_shapes(t)
        .is_some_and(|k| k.as_slice().iter().all(|x| ks.contains(x)))
    {
        return;
    }
    let op = match ks {
        [k] => Op::GuardShape(*k),
        _ => Op::GuardShapeIn(b.f.intern_shape_set(ks)),
    };
    b.ins_exit(op, &[t], ExitTag::Shape);
    let known = b.known_shapes(t);
    let set = match known {
        Some(kn) => kn
            .as_slice()
            .iter()
            .copied()
            .filter(|x| ks.contains(x))
            .collect(),
        None => ks.to_vec(),
    };
    if let Some(s) = Shapes::of(&set) {
        b.set_shapes(t, s);
    }
}

/// An access of table `t` dispatched on its shape, `shapes` grouped by
/// `way`: an arm per group, which `arm` emits for its first shape, each
/// but the last behind a test of the group's set; the last guarded, or
/// with no check when the facts leave no other shape.
fn dispatch<'gc>(
    b: &mut Builder<'_, 'gc>,
    t: Val,
    shapes: &[Shape<'gc>],
    way: impl Fn(&mut Builder<'_, 'gc>, usize, Shape<'gc>) -> Way,
    mut arm: impl FnMut(&mut Builder<'_, 'gc>, Shape<'gc>),
) {
    let known = b.known_shapes(t);
    let mut groups: Vec<(Way, Vec<u32>, Shape<'gc>)> = Vec::new();
    for (n, &s) in shapes.iter().enumerate() {
        let k = b.intern_shape(s);
        if known.is_some_and(|kn| !kn.contains(k)) {
            continue;
        }
        let w = way(b, n, s);
        match groups.iter_mut().find(|g| g.0 == w) {
            Some(g) => g.1.push(k),
            None => groups.push((w, vec![k], s)),
        }
    }
    let join = b.new_block();
    let last = groups.len().saturating_sub(1);
    for (n, (_, ks, rep)) in groups.into_iter().enumerate() {
        let covers = b
            .known_shapes(t)
            .is_some_and(|kn| kn.as_slice().iter().all(|x| ks.contains(x)));
        let next = if covers {
            None
        } else if n == last {
            guard_shapes(b, t, &ks);
            None
        } else {
            let op = match ks[..] {
                [k] => Op::ShapeIs(k),
                _ => Op::ShapeIn(b.f.intern_shape_set(&ks)),
            };
            let is = b.ins(op, &[t]);
            Some(branch_arm(b, is))
        };
        arm(b, rep);
        b.flush();
        b.jump(join);
        match next {
            Some(next) => b.switch_to(next),
            None => return enter_join(b, join),
        }
    }
    b.deopt(ExitTag::Shape);
    enter_join(b, join);
}

/// Whether the class `shape` pins has `bit` (a shape pins its class, not
/// the class's bits): a constant the region depends on (ws2 §5.5), or read
/// live from a class whose dependents were invalidated before; `None` for a
/// shape without a class, whose tables have no metatable.
fn class_has<'gc>(b: &mut Builder<'_, 'gc>, shape: Shape<'gc>, bit: MetamethodBits) -> Option<Val> {
    shape.mt_cache()?;
    if let Some(has) = known_bit(b, shape, bit) {
        return Some(b.ins(Op::KB1(has), &[]));
    }
    let k = b.intern_shape(shape);
    let bits = b.ins(Op::ClassBits(k), &[]);
    let mask = b.ki32(bit.bits() as i32);
    let has = b.ins(Op::IAnd, &[bits, mask]);
    let zero = b.ki32(0);
    Some(b.ins(Op::ICmp(Cc::Ne), &[has, zero]))
}

/// Whether the class `shape` pins has `bit`, when the region may assume it:
/// none has none; a stale class has every bit and changes no more; else a
/// dependency, unless the class is volatile.
fn known_bit<'gc>(
    b: &mut Builder<'_, 'gc>,
    shape: Shape<'gc>,
    bit: MetamethodBits,
) -> Option<bool> {
    let has = bit_now(shape, bit)?;
    if let Some(class) = shape.mt_cache()
        && !class.is_stale()
    {
        b.depend_class(class, bit);
    }
    Some(has)
}

/// [`known_bit`] without the dependency.
fn bit_now(shape: Shape<'_>, bit: MetamethodBits) -> Option<bool> {
    match shape.mt_cache() {
        None => Some(false),
        Some(c) if c.is_stale() => Some(true),
        Some(c) if c.is_volatile() => None,
        Some(c) => Some(c.get().contains(bit)),
    }
}

/// Whether `has` is a constant false: the class lacks the bit for good.
fn never(b: &Builder<'_, '_>, has: Val) -> bool {
    b.f.def_op(has) == Some(Op::KB1(false))
}

/// Whether the class of table `t` (a `Ptr`) has `bit`: a constant when the
/// shapes `t` may have are known and their classes agree, else read live.
pub(super) fn class_bit(b: &mut Builder<'_, '_>, t: Val, bit: MetamethodBits) -> Val {
    if let Some(known) = b.known_shapes(t) {
        let shapes: Vec<Shape<'_>> = known
            .as_slice()
            .iter()
            .map(|&k| b.f.shapes[k as usize])
            .collect();
        if let Some(has) = shapes.first().and_then(|&s| bit_now(s, bit))
            && shapes.iter().all(|&s| bit_now(s, bit) == Some(has))
        {
            for &s in &shapes {
                known_bit(b, s, bit);
            }
            return b.ins(Op::KB1(has), &[]);
        }
    }
    let bits = b.ins(Op::TabMmBits, &[t]);
    let mask = b.ki32(bit.bits() as i32);
    let has = b.ins(Op::IAnd, &[bits, mask]);
    let zero = b.ki32(0);
    b.ins(Op::ICmp(Cc::Ne), &[has, zero])
}

/// The metamethod of `bit` the class of table `t` (a `Ptr`) has: a constant
/// when the shapes `t` may have are known and share a class that has it,
/// else read live (nil when it has none).
pub(super) fn class_mm(b: &mut Builder<'_, '_>, t: Val, bit: MetamethodBits) -> Val {
    if let Some(known) = b.known_shapes(t) {
        let classes: Vec<Option<MtCache<'_>>> = known
            .as_slice()
            .iter()
            .map(|&k| b.f.shapes[k as usize].mt_cache())
            .collect();
        if let Some(Some(c)) = classes.first().copied()
            && classes
                .iter()
                .all(|x| x.is_some_and(|x| MtCache::ptr_eq(x, c)))
            && !c.is_stale()
            && !c.is_volatile()
            && !c.mm(bit).is_nil()
        {
            b.depend_class(c, bit);
            return b.konst(c.mm(bit));
        }
    }
    b.ins(Op::ClassMm(bit.bits().trailing_zeros() as u8), &[t])
}

/// The store's barrier, unless the value is never a heap object.
fn barrier_for(b: &mut Builder<'_, '_>, t: Val, v: Val) {
    let set = b.ty(v).set;
    if set.is_empty() || set.intersects(TypeSet::HEAP) {
        b.push(Op::Barrier, &[t]);
    }
}

/// `recv[key]` through `__index` tables into `dst`, an `__index` function
/// on `recv`'s own metatable called (5.4); any other walk deopts.
fn index(b: &mut Builder<'_, '_>, dst: u8, recv: Val, key: Val) -> Val {
    let recv = b.boxed(recv);
    let key = b.boxed(key);
    let r = b.ins(Op::Helper(HelperId::Index), &[recv, key]);
    let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
    let failed = b.ins(Op::SameBits, &[r, fail]);
    if !calls_mm(b) {
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Mm);
        return r;
    }
    let join = b.new_block();
    let ok = branch_arm(b, failed);
    cold(b, |b| {
        let f = function_or_deopt(b, HelperId::IndexFn, recv, key);
        let w = mm_call(b, f, &[recv, key]);
        call_arm_end(b, join, Some((dst, w)));
    });
    b.switch_to(ok);
    arm_end(b, join, dst, r);
    enter_join(b, join);
    b.reg(dst)
}

/// `recv[key] = v` through `__newindex` tables, an `__newindex` function on
/// `recv`'s own metatable called; any other walk deopts.
fn newindex(b: &mut Builder<'_, '_>, recv: Val, key: Val, v: Val, keyed: bool) {
    let recv = b.boxed(recv);
    let key = b.boxed(key);
    let r = b.ins(Op::Helper(HelperId::NewIndex { keyed }), &[recv, key, v]);
    let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
    let failed = b.ins(Op::SameBits, &[r, fail]);
    // The store may change a class (`MtCache::member_write`).
    if !calls_mm(b) {
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Mm);
        b.flush();
        b.lazy_exit_after();
        return;
    }
    let join = b.new_block();
    let ok = branch_arm(b, failed);
    cold(b, |b| {
        let f = function_or_deopt(b, HelperId::NewIndexFn, recv, key);
        mm_call(b, f, &[recv, key, v]);
        call_arm_end(b, join, None);
    });
    b.switch_to(ok);
    call_arm_end(b, join, None);
    enter_join(b, join);
}

/// The metamethod function a lookup helper finds, deopting when none.
fn function_or_deopt(b: &mut Builder<'_, '_>, h: HelperId, recv: Val, key: Val) -> Val {
    let f = b.ins(Op::Helper(h), &[recv, key]);
    let none = b.ins(Op::IsType(TypeSet::NIL), &[f]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[none], s, ExitTag::Slow);
    f
}

/// `dst = v`, or where `v` is nil what `found`, the walk from a table of
/// `shape`, gives.
fn nil_walks<'gc>(
    b: &mut Builder<'_, 'gc>,
    dst: u8,
    shape: Shape<'gc>,
    v: Val,
    site: &Site<'gc>,
    found: Found<'gc>,
) {
    if let Found::Nil = found {
        b.set(dst, v);
        b.note_def(&[v], None);
        return;
    }
    let join = b.new_block();
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let not_nil = branch_arm(b, nil);
    if let Found::Live(_) = found {
        let has = class_has(b, shape, MetamethodBits::INDEX).unwrap();
        let plain = branch_arm(b, has);
        cold(b, |b| {
            let w = index(b, dst, site.recv, site.key);
            arm_end(b, join, dst, w);
        });
        b.switch_to(plain);
        arm_end(b, join, dst, v);
    } else {
        cold(b, |b| {
            let w = emit_found(b, dst, found, site);
            arm_end(b, join, dst, w);
        });
    }
    b.switch_to(not_nil);
    arm_end(b, join, dst, v);
    enter_join(b, join);
}

/// `dst = t.key` for a table of `shape` that lacks the key.
fn absent<'gc>(b: &mut Builder<'_, 'gc>, dst: u8, site: &Site<'gc>, shape: Shape<'gc>) {
    match walk(b, shape.mt_cache(), site.str()) {
        found @ Found::Live(_) => {
            let nil = b.konst(Value::nil());
            nil_walks(b, dst, shape, nil, site, found);
        }
        found => {
            let v = emit_found(b, dst, found, site);
            b.set(dst, v);
        }
    }
}

/// `GETFIELD`, `GETTABUP` and `SELF` in all their forms: what the site's
/// cache entry holds, which is authoritative over its form (7.4).
pub(super) fn emit_get<'gc>(b: &mut Builder<'_, 'gc>, insn: Instruction) {
    let (dst, _, ic, e) = insn.abde();
    let recv = receiver(b, insn);
    if insn.op().unquickened() == BcOp::SELF {
        // Now, not at the end: a metamethod call below stores it first.
        // An exit before it runs SELF again, which writes it again.
        b.set(dst + 4, recv);
        b.flush();
    }
    let kv = b.closure.proto.constants[e as usize];
    let key = b.konst(kv);
    let kid = b.f.intern(kv);
    let site = Site { recv, key, kv, kid };
    if string_method(b, dst, &site) {
        return;
    }
    let ic = current_entry(b, read_ic(b.closure, ic), kv);
    let first = match ic {
        InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => Some(shape),
        InlineCache::ProtoLoad { recv, .. } => Some(recv),
        InlineCache::Transition { from, .. } => Some(from),
        InlineCache::Empty => None,
    };
    match site_shapes(b, first, recv) {
        None => {
            let v = index(b, dst, recv, key);
            b.set(dst, v);
            return;
        }
        Some(shapes) if shapes.len() > 1 || first.is_none() && !shapes.is_empty() => {
            get_chain(b, dst, &site, &shapes);
            return;
        }
        Some(_) => {}
    }
    match ic {
        InlineCache::Own { shape, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            own_load(b, dst, &site, shape, t, loc);
        }
        InlineCache::Absent { shape } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            absent(b, dst, &site, shape);
        }
        InlineCache::ProtoLoad {
            recv: shape,
            holder,
            holder_shape,
            loc,
        } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            match walk(b, shape.mt_cache(), site.str()) {
                Found::Live(_) => {
                    // The entry's references are live: the collector empties
                    // it when one dies.
                    let holder = Table::from_inner(unsafe { Gc::from_ptr(holder.as_ptr()) });
                    proto_load(b, dst, &site, shape, holder, holder_shape, loc);
                }
                found => {
                    let v = emit_found(b, dst, found, &site);
                    b.set(dst, v);
                }
            }
        }
        InlineCache::Transition { .. } | InlineCache::Empty => {
            let v = index(b, dst, recv, key);
            b.set(dst, v);
        }
    }
}

/// A read from strings, which the site only saw (11.5): the strings'
/// metatable still the one now, its `__index` the table it is now, that
/// table's slot for the key, each guarded. False when that chain is not
/// there to compile (a dict, an `__index` function), for the helper.
fn string_method<'gc>(b: &mut Builder<'_, 'gc>, dst: u8, site: &Site<'gc>) -> bool {
    let ty = b.ty(site.recv);
    let kinds = b.feedback(b.pc) & fb::KINDS;
    if !(kinds == fb::STR || !ty.set.is_empty() && ty.within(TypeSet::STR)) {
        return false;
    }
    let ctx = b.ctx;
    let index = ctx
        .symbols()
        .metamethod(MetamethodBits::INDEX.bits().trailing_zeros());
    let Some(mt) = ctx.metatable_of(site.kv) else {
        return false;
    };
    let Some(lib) = mt.raw_get(Value::string(index)).get_table() else {
        return false;
    };
    let (ms, ls) = (mt.shape(), lib.shape());
    let key = site.kv.get_string().unwrap();
    let (Some(mslot), Some(lslot)) = (
        (!ms.is_dict()).then(|| ms.find_slot(index)).flatten(),
        (!ls.is_dict()).then(|| ls.find_slot(key)).flatten(),
    ) else {
        return false;
    };
    b.guard(site.recv, TypeSet::STR);
    let word = ctx.string_metatable_word();
    let same = b.ins(
        Op::TypeMtIs {
            slot: crate::jit::layout::STRING_MT_SLOT,
            word,
        },
        &[],
    );
    let s = b.snap_before();
    b.ins_snap(Op::GuardTrue, &[same], s, ExitTag::Shape);
    if let Some(class) = ms.as_mt()
        && let Found::Const(m) = walk(b, Some(class), key)
    {
        let m = b.konst(m);
        b.set(dst, m);
        return true;
    }
    // The pool keeps `mt` alive, so its word is not another table's.
    let mv = b.konst(Value::table(mt));
    let mp = b.ins(Op::Unbox(Rep::Ptr), &[mv]);
    guard_shape(b, mp, ms);
    let ik = b.f.intern(Value::string(index));
    let iv = b.ins(
        Op::TabLoad {
            loc: SlotLoc::new(ms, mslot).raw(),
            key: ik,
        },
        &[mp],
    );
    let lv = b.konst(Value::table(lib));
    b.ins_exit(Op::GuardSame, &[iv, lv], ExitTag::Shape);
    let lp = b.ins(Op::Unbox(Rep::Ptr), &[lv]);
    guard_shape(b, lp, ls);
    let m = b.ins(
        Op::TabLoad {
            loc: SlotLoc::new(ls, lslot).raw(),
            key: site.kid,
        },
        &[lp],
    );
    b.field_loads.insert(m, (lv, site.kv));
    b.set(dst, m);
    true
}

/// The interpreter's entry as for tables that left a deprecated shape:
/// they move to its poly twin on their next slow path, the twin's slots
/// being the deprecated shape's and the metatable's.
fn current_entry<'gc>(
    b: &mut Builder<'_, 'gc>,
    ic: InlineCache<'gc>,
    kv: Value<'gc>,
) -> InlineCache<'gc> {
    let ctx = b.ctx;
    let twin = |s: Shape<'gc>| {
        s.poly_twin(
            ctx.mutation(),
            ctx.symbols().mt_key,
            ctx.empty_dict_sentinel(),
        )
    };
    let Some(key) = kv.get_string() else {
        return ic;
    };
    match ic {
        InlineCache::Own { shape, .. } if shape.is_deprecated() => {
            let p = twin(shape);
            match (!p.is_dict()).then(|| p.find_slot(key)).flatten() {
                Some(slot) => InlineCache::Own {
                    shape: p,
                    loc: SlotLoc::new(p, slot),
                },
                None => InlineCache::Empty,
            }
        }
        InlineCache::Absent { shape } if shape.is_deprecated() => {
            InlineCache::Absent { shape: twin(shape) }
        }
        InlineCache::ProtoLoad {
            recv,
            holder,
            holder_shape,
            loc,
        } if recv.is_deprecated() => InlineCache::ProtoLoad {
            recv: twin(recv),
            holder,
            holder_shape,
            loc,
        },
        InlineCache::Transition { from, .. } if from.is_deprecated() => {
            let p = twin(from);
            match p.prop_child(ctx.mutation(), key) {
                Some(to) if !p.is_dict() && !to.is_dict() => InlineCache::Transition {
                    from: p,
                    to,
                    loc: SlotLoc::new(to, p.slot_count()),
                },
                _ => InlineCache::Empty,
            }
        }
        ic => ic,
    }
}

/// A constant-key site's receiver and key.
struct Site<'gc> {
    recv: Val,
    key: Val,
    kv: Value<'gc>,
    kid: u32,
}

impl<'gc> Site<'gc> {
    fn str(&self) -> LuaString<'gc> {
        self.kv.get_string().unwrap()
    }
}

/// The shapes a site tests for: its cache's first, those its shape exits
/// recorded, then, while there is room, those earlier sites tested on the
/// same receiver value, which it most likely has here too. `None` past
/// `MAX_SHAPES` or for a dict shape, a site left to the helper (6.4).
fn site_shapes<'gc>(
    b: &mut Builder<'_, 'gc>,
    first: Option<Shape<'gc>>,
    recv: Val,
) -> Option<Vec<Shape<'gc>>> {
    const MAX: usize = crate::jit::MAX_SHAPES;
    let has = |v: &[Shape<'gc>], s| v.iter().any(|&x| Shape::ptr_eq(x, s));
    // A deprecated shape's tables move to the poly one at their next exit.
    let first = first.filter(|s| !s.is_deprecated());
    let mut v: Vec<Shape<'gc>> = first.into_iter().collect();
    let vector = &b.proto.feedback;
    let pc = b.pc as usize;
    let mut seen = 0;
    for s in vector.shapes(pc) {
        seen += 1;
        if !has(&v, s) && !s.is_deprecated() {
            v.push(s);
        }
    }
    // The interpreter refills the cache after an exit, so its shape now is
    // kept: dropped, the next compile would exit on it again.
    if let Some(f) = first
        && v.len() > seen
        && seen <= MAX
    {
        vector.add_shape(pc, f);
        b.ctx.mutation().backward_barrier(Gc::erase(b.proto), None);
    }
    if v.len() > MAX || v.iter().any(|s| s.is_dict()) {
        return None;
    }
    let known = b.recv_shapes.entry(b.root_of(recv)).or_default();
    for &s in known.iter() {
        if v.len() < MAX && !has(&v, s) && !s.is_deprecated() {
            v.push(s);
        }
    }
    for &s in &v {
        if !has(known, s) {
            known.push(s);
        }
    }
    Some(v)
}

/// `dst = t.key` from the own slot at `loc`.
fn own_load<'gc>(
    b: &mut Builder<'_, 'gc>,
    dst: u8,
    site: &Site<'gc>,
    shape: Shape<'gc>,
    t: Val,
    loc: SlotLoc,
) {
    let v = b.ins(
        Op::TabLoad {
            loc: loc.raw(),
            key: site.kid,
        },
        &[t],
    );
    b.field_loads.insert(v, (site.recv, site.kv));
    let found = walk(b, shape.mt_cache(), site.str());
    nil_walks(b, dst, shape, v, site, found);
}

/// `dst = t.key` from the slot at `loc` of `holder`, the `__index` table of
/// the class `shape` pins, while it is that table and has its shape.
fn proto_load<'gc>(
    b: &mut Builder<'_, 'gc>,
    dst: u8,
    site: &Site<'gc>,
    shape: Shape<'gc>,
    holder: Table<'gc>,
    holder_shape: Shape<'gc>,
    loc: SlotLoc,
) {
    let hv = b.konst(Value::table(holder));
    let k = b.intern_shape(shape);
    let live = b.ins(Op::ClassIndex(k), &[]);
    b.ins_exit(Op::GuardSame, &[live, hv], ExitTag::Shape);
    let h = b.ins(Op::Unbox(Rep::Ptr), &[hv]);
    guard_shape(b, h, holder_shape);
    let v = b.ins(
        Op::TabLoad {
            loc: loc.raw(),
            key: site.kid,
        },
        &[h],
    );
    // A nil slot means the walk goes on past `holder`.
    let join = b.new_block();
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let found = branch_arm(b, nil);
    cold(b, |b| {
        let w = index(b, dst, site.recv, site.key);
        arm_end(b, join, dst, w);
    });
    b.switch_to(found);
    arm_end(b, join, dst, v);
    enter_join(b, join);
}

/// The `__index` table of `shape`'s class holding `key` in a slot of its
/// shape, as a `ProtoLoad` entry caches it.
fn holder_of<'gc>(
    shape: Shape<'gc>,
    key: LuaString<'gc>,
) -> Option<(Table<'gc>, Shape<'gc>, SlotLoc)> {
    let holder = shape.mt_cache()?.mm(MetamethodBits::INDEX).get_table()?;
    let hs = holder.shape();
    if hs.is_dict() {
        return None;
    }
    let slot = hs.find_slot(key)?;
    Some((holder, hs, SlotLoc::new(hs, slot)))
}

/// A read at a site that saw several shapes: a test per shape, each with
/// what tables of that shape hold for the key, resolved now; another shape
/// exits and is recorded.
fn get_chain<'gc>(b: &mut Builder<'_, 'gc>, dst: u8, site: &Site<'gc>, shapes: &[Shape<'gc>]) {
    let t = table_of(b, site.recv);
    let key = site.str();
    // Each shape's walk, its dependencies taken: an arm serves shapes of
    // any class whose walks agree.
    let way = |b: &mut Builder<'_, 'gc>, _, s: Shape<'gc>| {
        let found = walk(b, s.mt_cache(), key);
        if let Some(slot) = s.find_slot(key) {
            Way::Own(SlotLoc::new(s, slot).raw(), found.key())
        } else if !matches!(found, Found::Live(_)) {
            Way::Walk(found.key())
        } else if let Some((holder, hs, loc)) = holder_of(s, key) {
            let h = Gc::as_ptr(holder.inner()) as u64;
            Way::Proto(h, hs.as_word(), loc.raw(), class_word(s))
        } else {
            Way::Absent(class_word(s))
        }
    };
    dispatch(b, t, shapes, way, |b, s| {
        let found = walk(b, s.mt_cache(), key);
        if let Some(slot) = s.find_slot(key) {
            own_load(b, dst, site, s, t, SlotLoc::new(s, slot));
        } else if let (Found::Live(_), Some((holder, hs, loc))) = (found, holder_of(s, key)) {
            proto_load(b, dst, site, s, holder, hs, loc);
        } else {
            absent(b, dst, site, s);
        }
    });
}

/// `SETFIELD` and `SETTABUP` in all their forms.
pub(super) fn emit_set<'gc>(b: &mut Builder<'_, 'gc>, insn: Instruction) {
    let (src, _, ic, e) = insn.abde();
    let recv = receiver(b, insn);
    let v = b.reg(src);
    let v = b.boxed(v);
    let kv = b.closure.proto.constants[e as usize];
    let key = b.konst(kv);
    let kid = b.f.intern(kv);
    let ic = current_entry(b, read_ic(b.closure, ic), kv);
    let first = match ic {
        InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => Some(shape),
        InlineCache::ProtoLoad { recv, .. } => Some(recv),
        InlineCache::Transition { from, .. } => Some(from),
        InlineCache::Empty => None,
    };
    match site_shapes(b, first, recv) {
        None => {
            newindex(b, recv, key, v, false);
            return;
        }
        Some(shapes) if shapes.len() > 1 || first.is_none() && !shapes.is_empty() => {
            // What the interpreter's cache would hold for each shape; own
            // stores to one location of one class share an arm.
            let t = table_of(b, recv);
            let ks = kv.get_string().unwrap();
            let mc = b.ctx.mutation();
            // A store to a metatable's mirrored key tells its class, one to a
            // watched holder its dependents.
            let tells = |s: Shape<'gc>| s.as_mt().is_some() && crate::env::shape::mirrored(ks);
            let way = |b: &mut Builder<'_, 'gc>, n, s: Shape<'gc>| match (
                s.find_slot(ks),
                s.prop_child(mc, ks),
            ) {
                _ if tells(s) || !stores_inline(s, ks) => Way::Helper,
                (Some(slot), _) => {
                    let k = match known_bit(b, s, MetamethodBits::NEWINDEX) {
                        Some(false) => 0,
                        _ => class_word(s),
                    };
                    Way::Store(SlotLoc::new(s, slot).raw(), k)
                }
                (None, Some(to)) if !to.is_dict() => Way::Unique(n),
                (None, _) => Way::Helper,
            };
            dispatch(b, t, &shapes, way, |b, s| {
                match (s.find_slot(ks), s.prop_child(mc, ks)) {
                    _ if tells(s) || !stores_inline(s, ks) => newindex(b, recv, key, v, false),
                    (Some(slot), _) => own_store(b, s, t, SlotLoc::new(s, slot), kid, v),
                    (None, Some(to)) if !to.is_dict() => {
                        let loc = SlotLoc::new(to, s.slot_count());
                        trans_store(b, recv, key, s, to, t, loc, kid, v);
                    }
                    (None, _) => newindex(b, recv, key, v, false),
                }
            });
            return;
        }
        Some(_) => {}
    }
    let ks = kv.get_string().unwrap();
    match ic {
        InlineCache::Own { shape, .. } | InlineCache::Transition { from: shape, .. }
            if !stores_inline(shape, ks) =>
        {
            newindex(b, recv, key, v, false);
        }
        InlineCache::Own { shape, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            own_store(b, shape, t, loc, kid, v);
        }
        InlineCache::Transition { from, to, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, from);
            trans_store(b, recv, key, from, to, t, loc, kid, v);
        }
        InlineCache::Absent { .. } | InlineCache::ProtoLoad { .. } | InlineCache::Empty => {
            newindex(b, recv, key, v, false);
        }
    }
}

/// Whether compiled code may store `key` to tables of `shape` itself: a
/// watched holder only to a slot an earlier store changed, so nothing
/// assumes its value, and never adding a key (ws2 §5.2).
fn stores_inline<'gc>(shape: Shape<'gc>, key: LuaString<'gc>) -> bool {
    !shape.is_watched() || shape.find_slot(key).is_some_and(|s| shape.slot_mutable(s))
}

/// `t.key = v` for a table of shape `from` that lacks the key: `to` with
/// the key at `loc`, as the interpreter's transition takes it.
#[allow(clippy::too_many_arguments)]
fn trans_store<'gc>(
    b: &mut Builder<'_, 'gc>,
    recv: Val,
    key: Val,
    from: Shape<'gc>,
    to: Shape<'gc>,
    t: Val,
    loc: SlotLoc,
    kid: u32,
    v: Val,
) {
    if let Some(has) = class_has(b, from, MetamethodBits::NEWINDEX)
        && !never(b, has)
    {
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Slow);
    }
    let join = b.new_block();
    // Storing nil to an absent key adds nothing.
    let set = b.ty(v).set;
    if set.is_empty() || set.contains(TypeSet::NIL) {
        let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
        let store = branch_arm(b, nil);
        b.jump(join);
        b.switch_to(store);
    }
    // A full spill cell grows in the generic store.
    let full = loc.spilled().then(|| {
        let room = b.ins(Op::TabHasRoom(loc.raw()), &[t]);
        branch_arm(b, room)
    });
    // The table now refers to `to` as well as to `v`.
    b.push(Op::Barrier, &[t]);
    b.push(
        Op::TabStore {
            loc: loc.raw(),
            key: kid,
        },
        &[t, v],
    );
    let k = b.intern_shape(to);
    b.push(Op::TabSetShape(k), &[t]);
    b.set_shapes(t, Shapes::one(k));
    b.jump(join);
    if let Some(full) = full {
        b.switch_to(full);
        cold(b, |b| {
            newindex(b, recv, key, v, false);
            b.jump(join);
        });
    }
    enter_join(b, join);
}

/// `t.key = v` into the own slot at `loc`, `__newindex` checked live where
/// the class may have it.
fn own_store<'gc>(
    b: &mut Builder<'_, 'gc>,
    shape: Shape<'gc>,
    t: Val,
    loc: SlotLoc,
    kid: u32,
    v: Val,
) {
    if let Some(has) = class_has(b, shape, MetamethodBits::NEWINDEX)
        && !never(b, has)
    {
        // `__newindex` fires only on a nil slot.
        let old = b.ins(
            Op::TabLoad {
                loc: loc.raw(),
                key: kid,
            },
            &[t],
        );
        let join = b.new_block();
        let nil = b.ins(Op::IsType(TypeSet::NIL), &[old]);
        let rest = branch_arm(b, nil);
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Slow);
        b.jump(join);
        b.switch_to(rest);
        b.jump(join);
        enter_join(b, join);
    }
    barrier_for(b, t, v);
    b.push(
        Op::TabStore {
            loc: loc.raw(),
            key: kid,
        },
        &[t, v],
    );
}

/// The receiver's lock when the site's feedback saw only tables (or
/// nothing yet); `None` when it saw other receivers (`OTHER`, which an odd
/// key also sets), which take the helper.
fn table_site(b: &mut Builder<'_, '_>, pc: u32, recv: Val) -> Option<Val> {
    let ty = b.ty(recv);
    let tables = !ty.set.is_empty() && ty.within(TypeSet::TAB);
    (tables || b.feedback(pc) & fb::OTHER == 0).then(|| table_of(b, recv))
}

/// The small-integer key of an array access: `Ok` its `I32`, guarded small
/// unless the site's feedback saw other keys; then `Err(other)`, the block a
/// test sends them to. `None` for a key never small: no array access.
fn int_key(b: &mut Builder<'_, '_>, pc: u32, k: Val) -> Option<Result<Val, (Val, Block)>> {
    let ty = b.ty(k);
    if ty.rep == Rep::I32 || (!ty.set.is_empty() && ty.within(TypeSet::SMALL)) {
        return Some(Ok(b.as_i32(k)));
    }
    if !ty.set.is_empty() && !ty.set.intersects(TypeSet::SMALL) {
        return None;
    }
    let others = fb::FLOAT | fb::STR | fb::TAB | fb::OTHER;
    if b.feedback(pc) & others == 0 {
        return Some(Ok(b.as_i32(k)));
    }
    let kb = b.boxed(k);
    let is = b.ins(Op::IsType(TypeSet::SMALL), &[kb]);
    let other = branch_arm(b, is);
    Some(Err((b.ins(Op::Unbox(Rep::I32), &[kb]), other)))
}

/// The array part's cell and `i` inside it, where the site's feedback never
/// saw a miss: the class without `bit` and `i` below the size, guarded
/// (`Bounds`). `None` after a miss: the caller tests both.
fn array_guards(
    b: &mut Builder<'_, '_>,
    pc: u32,
    t: Val,
    i: Val,
    bit: MetamethodBits,
) -> Option<Val> {
    if b.feedback(pc) & fb::ARR_MISS != 0 {
        return None;
    }
    // Before the access: its class is the table's for the whole loop.
    let has = class_bit(b, t, bit);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Bounds);
    let n = b.ins(Op::TabASize, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    b.ins_snap(Op::GuardTrue, &[inside], s, ExitTag::Bounds);
    Some(b.ins(Op::TabArray, &[t]))
}

/// `R[a] = R[b][R[c]]`: a small integer key inside the array part loads
/// inline, guarded unless the site saw misses; then the misses, other keys
/// and other receivers take the helper.
pub(super) fn emit_gettable(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    let (dst, rt, rk) = (insn.a(), insn.b(), insn.c());
    let (recv, key) = (b.reg(rt), b.reg(rk));
    let Some(t) = table_site(b, pc, recv) else {
        let v = index(b, dst, recv, key);
        b.set(dst, v);
        return;
    };
    let (i, other) = match int_key(b, pc, key) {
        None => {
            let v = index(b, dst, recv, key);
            b.set(dst, v);
            return;
        }
        Some(Ok(i)) => (i, None),
        Some(Err((i, other))) => (i, Some(other)),
    };
    if let Some(arr) = array_guards(b, pc, t, i, MetamethodBits::INDEX) {
        let v = b.ins(Op::ArrLoad, &[arr, i]);
        match other {
            None => b.set(dst, v),
            Some(other) => {
                let join = b.new_block();
                arm_end(b, join, dst, v);
                b.switch_to(other);
                cold(b, |b| {
                    let w = index(b, dst, recv, key);
                    arm_end(b, join, dst, w);
                });
                enter_join(b, join);
            }
        }
        return;
    }
    let join = b.new_block();
    let n = b.ins(Op::TabASize, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    let outside = branch_arm(b, inside);
    let arr = b.ins(Op::TabArray, &[t]);
    let v = b.ins(Op::ArrLoad, &[arr, i]);
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let found = branch_arm(b, nil);
    // A nil is the result unless the class has `__index` now.
    let has = class_bit(b, t, MetamethodBits::INDEX);
    let plain = branch_arm(b, has);
    let walk = |b: &mut Builder<'_, '_>| {
        cold(b, |b| {
            let w = index(b, dst, recv, key);
            arm_end(b, join, dst, w);
        })
    };
    walk(b);
    b.switch_to(plain);
    arm_end(b, join, dst, v);
    b.switch_to(found);
    arm_end(b, join, dst, v);
    b.switch_to(outside);
    walk(b);
    if let Some(other) = other {
        b.switch_to(other);
        walk(b);
    }
    enter_join(b, join);
}

/// `R[b][R[c]] = R[a]`: a small integer key inside the array part stores
/// inline, guarded as for GETTABLE; past a miss, over a nil only where the
/// class has no `__newindex` now; the rest takes the helper.
pub(super) fn emit_settable(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    let (src, rt, rk) = (insn.a(), insn.b(), insn.c());
    let (recv, key, v) = (b.reg(rt), b.reg(rk), b.reg(src));
    let v = b.boxed(v);
    let Some(t) = table_site(b, pc, recv) else {
        newindex(b, recv, key, v, true);
        return;
    };
    let (i, other) = match int_key(b, pc, key) {
        None => {
            newindex(b, recv, key, v, true);
            return;
        }
        Some(Ok(i)) => (i, None),
        Some(Err((i, other))) => (i, Some(other)),
    };
    let join = b.new_block();
    if let Some(arr) = array_guards(b, pc, t, i, MetamethodBits::NEWINDEX) {
        barrier_for(b, t, v);
        b.push(Op::ArrStore, &[arr, i, v]);
        b.jump(join);
    } else {
        let n = b.ins(Op::TabASize, &[t]);
        let inside = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
        let outside = branch_arm(b, inside);
        let arr = b.ins(Op::TabArray, &[t]);
        let old = b.ins(Op::ArrLoad, &[arr, i]);
        // `__newindex` may fire over a nil, if the class has it now.
        let nil = b.ins(Op::IsType(TypeSet::NIL), &[old]);
        let present = branch_arm(b, nil);
        let has = class_bit(b, t, MetamethodBits::NEWINDEX);
        let plain = branch_arm(b, has);
        cold(b, |b| {
            newindex(b, recv, key, v, true);
            b.jump(join);
        });
        for blk in [plain, present] {
            b.switch_to(blk);
            barrier_for(b, t, v);
            b.push(Op::ArrStore, &[arr, i, v]);
            b.jump(join);
        }
        b.switch_to(outside);
        cold(b, |b| {
            newindex(b, recv, key, v, true);
            b.jump(join);
        });
    }
    if let Some(other) = other {
        b.switch_to(other);
        cold(b, |b| {
            newindex(b, recv, key, v, true);
            b.jump(join);
        });
    }
    enter_join(b, join);
}
