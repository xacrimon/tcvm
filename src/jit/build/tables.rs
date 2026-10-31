//! Table access (`jit-design.md` 7.4, 7.6): constant-key sites from their
//! inline caches, integer keys through the array part, everything else
//! through `jit_index` and `jit_newindex`, whose failure (a metamethod to
//! call, an error) deopts.

use crate::dmm::Gc;
use crate::env::function::InlineCache;
use crate::env::string::LuaString;
use crate::env::table::{SlotLoc, Table};
use crate::env::value::Value;
use crate::env::{MetamethodBits, Shape};
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::emit::{arm_end, branch_arm, calls_mm, cold, enter_join, mm_call};
use crate::jit::build::{Builder, Shapes};
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

/// Which of a chain's shapes `t` may have here, each with its index and
/// whether it is the only one left (no test needed); `None` for a shape it
/// is known not to have.
fn chain_step<'gc>(b: &mut Builder<'_, 'gc>, t: Val, s: Shape<'gc>) -> Option<(u32, bool)> {
    let k = b.intern_shape(s);
    match b.known_shapes(t) {
        Some(set) if !set.contains(k) => None,
        Some(set) => Some((k, set.as_slice().len() == 1)),
        None => Some((k, false)),
    }
}

/// Whether the class `shape` pins has `bit` now, read live (a shape pins
/// its class, not the class's bits); `None` for a shape without a class,
/// whose tables have no metatable.
fn class_has<'gc>(b: &mut Builder<'_, 'gc>, shape: Shape<'gc>, bit: MetamethodBits) -> Option<Val> {
    shape.mt_cache()?;
    let k = b.intern_shape(shape);
    let bits = b.ins(Op::ClassBits(k), &[]);
    let mask = b.ki32(bit.bits() as i32);
    let has = b.ins(Op::IAnd, &[bits, mask]);
    let zero = b.ki32(0);
    Some(b.ins(Op::ICmp(Cc::Ne), &[has, zero]))
}

/// Whether the class of table `t` (a `Ptr`) has `bit`, read live: a table's
/// shape may be any here.
pub(super) fn class_bit(b: &mut Builder<'_, '_>, t: Val, bit: MetamethodBits) -> Val {
    let bits = b.ins(Op::TabMmBits, &[t]);
    let mask = b.ki32(bit.bits() as i32);
    let has = b.ins(Op::IAnd, &[bits, mask]);
    let zero = b.ki32(0);
    b.ins(Op::ICmp(Cc::Ne), &[has, zero])
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
        arm_end(b, join, dst, w);
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
    if !calls_mm(b) {
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Mm);
        return;
    }
    let join = b.new_block();
    let ok = branch_arm(b, failed);
    cold(b, |b| {
        let f = function_or_deopt(b, HelperId::NewIndexFn, recv, key);
        mm_call(b, f, &[recv, key, v]);
        b.jump(join);
    });
    b.switch_to(ok);
    b.jump(join);
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

/// `dst = v`, or the `__index` walk's answer when `v` is nil and the class
/// of `shape` has `__index` by then.
fn nil_walks<'gc>(
    b: &mut Builder<'_, 'gc>,
    dst: u8,
    shape: Shape<'gc>,
    v: Val,
    recv: Val,
    key: Val,
) {
    if shape.mt_cache().is_none() {
        b.set(dst, v);
        b.note_def(&[v], None);
        return;
    }
    let join = b.new_block();
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let not_nil = branch_arm(b, nil);
    let has = class_has(b, shape, MetamethodBits::INDEX).unwrap();
    let plain = branch_arm(b, has);
    cold(b, |b| {
        let w = index(b, dst, recv, key);
        arm_end(b, join, dst, w);
    });
    b.switch_to(plain);
    arm_end(b, join, dst, v);
    b.switch_to(not_nil);
    arm_end(b, join, dst, v);
    enter_join(b, join);
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
    let ic = read_ic(b.closure, ic);
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
            let nil = b.konst(Value::nil());
            nil_walks(b, dst, shape, nil, recv, key);
        }
        InlineCache::ProtoLoad {
            recv: shape,
            holder,
            holder_shape,
            loc,
        } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            // The entry's references are live: the collector empties it
            // when one dies.
            let holder = Table::from_inner(unsafe { Gc::from_ptr(holder.as_ptr()) });
            proto_load(b, dst, &site, shape, holder, holder_shape, loc);
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

/// A constant-key site's receiver and key.
struct Site<'gc> {
    recv: Val,
    key: Val,
    kv: Value<'gc>,
    kid: u32,
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
    let mut v: Vec<Shape<'gc>> = first.into_iter().collect();
    let vector = &b.proto.feedback;
    let pc = b.pc as usize;
    let mut seen = 0;
    for s in vector.shapes(pc) {
        seen += 1;
        if !has(&v, s) {
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
        if v.len() < MAX && !has(&v, s) {
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
    nil_walks(b, dst, shape, v, site.recv, site.key);
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
    let key = site.kv.get_string().unwrap();
    let join = b.new_block();
    for &s in shapes {
        let Some((k, only)) = chain_step(b, t, s) else {
            continue;
        };
        let next = (!only).then(|| {
            let is = b.ins(Op::ShapeIs(k), &[t]);
            branch_arm(b, is)
        });
        if let Some(slot) = s.find_slot(key) {
            own_load(b, dst, site, s, t, SlotLoc::new(s, slot));
        } else if let Some((holder, hs, loc)) = holder_of(s, key) {
            proto_load(b, dst, site, s, holder, hs, loc);
        } else {
            let nil = b.konst(Value::nil());
            nil_walks(b, dst, s, nil, site.recv, site.key);
        }
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

/// `SETFIELD` and `SETTABUP` in all their forms.
pub(super) fn emit_set<'gc>(b: &mut Builder<'_, 'gc>, insn: Instruction) {
    let (src, _, ic, e) = insn.abde();
    let recv = receiver(b, insn);
    let v = b.reg(src);
    let v = b.boxed(v);
    let kv = b.closure.proto.constants[e as usize];
    let key = b.konst(kv);
    let kid = b.f.intern(kv);
    let ic = read_ic(b.closure, ic);
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
            // A test per shape: what the interpreter's cache would hold for it.
            let t = table_of(b, recv);
            let ks = kv.get_string().unwrap();
            let join = b.new_block();
            for &s in &shapes {
                let Some((k, only)) = chain_step(b, t, s) else {
                    continue;
                };
                let next = (!only).then(|| {
                    let is = b.ins(Op::ShapeIs(k), &[t]);
                    branch_arm(b, is)
                });
                // A store to a metatable's mirrored key tells its class.
                let tells = s.as_mt().is_some() && crate::env::shape::mirrored(ks);
                match (s.find_slot(ks), s.prop_child(b.ctx.mutation(), ks)) {
                    _ if tells => newindex(b, recv, key, v, false),
                    (Some(slot), _) => own_store(b, s, t, SlotLoc::new(s, slot), kid, v),
                    (None, Some(to)) if !to.is_dict() => {
                        let loc = SlotLoc::new(to, s.slot_count());
                        trans_store(b, recv, key, s, to, t, loc, kid, v);
                    }
                    (None, _) => newindex(b, recv, key, v, false),
                }
                b.flush();
                b.jump(join);
                match next {
                    Some(next) => b.switch_to(next),
                    None => return enter_join(b, join),
                }
            }
            b.deopt(ExitTag::Shape);
            enter_join(b, join);
            return;
        }
        Some(_) => {}
    }
    match ic {
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
    if let Some(has) = class_has(b, from, MetamethodBits::NEWINDEX) {
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
    if shape.mt_cache().is_some() {
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
        let has = class_has(b, shape, MetamethodBits::NEWINDEX).unwrap();
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
