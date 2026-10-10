//! Table access (`jit-design.md` 7.4, 7.6): constant-key sites from their
//! inline caches, integer keys through the array part, everything else
//! through `jit_index` and `jit_newindex`, whose failure (a metamethod to
//! call, an error) deopts.

use crate::dmm::Gc;
use crate::env::function::InlineCache;
use crate::env::table::Table;
use crate::env::value::Value;
use crate::env::{MetamethodBits, Shape};
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::Builder;
use crate::jit::build::emit::{arm_end, branch_arm, enter_join, helper_val};
use crate::jit::feedback as fb;
use crate::jit::ir::Val;
use crate::jit::ir::ops::{Cc, ExitTag, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::vm::ops::field::read_ic;

/// The receiver of a constant-key site: a register, a by-value upvalue, or
/// a shared cell's value.
fn receiver(b: &mut Builder<'_, '_>, insn: Instruction) -> Val {
    match insn.op().unquickened() {
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

fn guard_shape<'gc>(b: &mut Builder<'_, 'gc>, t: Val, shape: Shape<'gc>) {
    let k = b.intern_shape(shape);
    b.ins_exit(Op::GuardShape(k), &[t], ExitTag::Shape);
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

/// The store's barrier, unless the value is never a heap object.
fn barrier_for(b: &mut Builder<'_, '_>, t: Val, v: Val) {
    let set = b.ty(v).set;
    if set.is_empty() || set.intersects(TypeSet::HEAP) {
        b.push(Op::Barrier, &[t]);
    }
}

fn index(b: &mut Builder<'_, '_>, recv: Val, key: Val) -> Val {
    let recv = b.boxed(recv);
    let key = b.boxed(key);
    helper_val(b, HelperId::Index, &[recv, key])
}

fn newindex(b: &mut Builder<'_, '_>, recv: Val, key: Val, v: Val, keyed: bool) {
    let recv = b.boxed(recv);
    let key = b.boxed(key);
    helper_val(b, HelperId::NewIndex { keyed }, &[recv, key, v]);
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
        return;
    }
    let join = b.new_block();
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let not_nil = branch_arm(b, nil);
    let has = class_has(b, shape, MetamethodBits::INDEX).unwrap();
    let plain = branch_arm(b, has);
    let w = index(b, recv, key);
    arm_end(b, join, dst, w);
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
        b.set(dst + 4, recv);
    }
    let kv = b.closure.proto.constants[e as usize];
    let key = b.konst(kv);
    let kid = b.f.intern(kv);
    match read_ic(b.closure, ic) {
        InlineCache::Own { shape, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            let v = b.ins(
                Op::TabLoad {
                    loc: loc.raw(),
                    key: kid,
                },
                &[t],
            );
            nil_walks(b, dst, shape, v, recv, key);
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
            // The collector empties the entry when `holder` dies, and none
            // ran since the interpreter last read it.
            let holder = Table::from_inner(unsafe { Gc::from_ptr(holder.as_ptr()) });
            let hv = b.konst(Value::table(holder));
            let k = b.intern_shape(shape);
            let live = b.ins(Op::ClassIndex(k), &[]);
            b.ins_exit(Op::GuardSame, &[live, hv], ExitTag::Shape);
            let h = b.ins(Op::Unbox(Rep::Ptr), &[hv]);
            guard_shape(b, h, holder_shape);
            let v = b.ins(
                Op::TabLoad {
                    loc: loc.raw(),
                    key: kid,
                },
                &[h],
            );
            // A nil slot means the walk goes on past `holder`.
            let join = b.new_block();
            let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
            let found = branch_arm(b, nil);
            let w = index(b, recv, key);
            arm_end(b, join, dst, w);
            b.switch_to(found);
            arm_end(b, join, dst, v);
            enter_join(b, join);
        }
        InlineCache::Transition { .. } | InlineCache::Empty => {
            let v = index(b, recv, key);
            b.set(dst, v);
        }
    }
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
    match read_ic(b.closure, ic) {
        InlineCache::Own { shape, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, shape);
            let slot = Op::TabLoad {
                loc: loc.raw(),
                key: kid,
            };
            if shape.mt_cache().is_some() {
                // `__newindex` fires only on a nil slot.
                let old = b.ins(slot, &[t]);
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
        InlineCache::Transition { from, to, loc } => {
            let t = table_of(b, recv);
            guard_shape(b, t, from);
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
            if loc.spilled() {
                // A full spill cell grows in the generic store.
                let room = b.ins(Op::TabHasRoom(loc.raw()), &[t]);
                let full = branch_arm(b, room);
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
                b.jump(join);
                b.switch_to(full);
                newindex(b, recv, key, v, false);
                b.jump(join);
            } else {
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
                b.jump(join);
            }
            enter_join(b, join);
        }
        InlineCache::Absent { .. } | InlineCache::ProtoLoad { .. } | InlineCache::Empty => {
            newindex(b, recv, key, v, false);
        }
    }
}

/// The receiver's lock when the site's feedback saw only tables (or
/// nothing yet); `None` when it saw other receivers, which take the helper.
fn table_site(b: &mut Builder<'_, '_>, pc: u32, recv: Val) -> Option<Val> {
    let kinds = b.feedback(pc) & fb::KINDS;
    let ty = b.ty(recv);
    let tables = !ty.set.is_empty() && ty.within(TypeSet::TAB);
    (tables || kinds & !fb::TAB == 0).then(|| table_of(b, recv))
}

/// An `I32` key, or `None` when the key may not be a small integer.
fn int_key(b: &Builder<'_, '_>, k: Val) -> Option<bool> {
    let ty = b.ty(k);
    if ty.rep == Rep::I32 || (!ty.set.is_empty() && ty.within(TypeSet::SMALL)) {
        Some(true)
    } else if !ty.set.is_empty() && !ty.set.intersects(TypeSet::SMALL) {
        Some(false)
    } else {
        None
    }
}

/// `R[a] = R[b][R[c]]`: a small integer key inside the array part loads
/// inline; a nil there, any other key and other receivers take the helper.
pub(super) fn emit_gettable(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    let (dst, rt, rk) = (insn.a(), insn.b(), insn.c());
    let (recv, key) = (b.reg(rt), b.reg(rk));
    let Some(t) = table_site(b, pc, recv) else {
        let v = index(b, recv, key);
        b.set(dst, v);
        return;
    };
    let join = b.new_block();
    let small = match int_key(b, key) {
        Some(false) => {
            let v = index(b, recv, key);
            b.set(dst, v);
            return;
        }
        Some(true) => None,
        None => {
            let kb = b.boxed(key);
            let is = b.ins(Op::IsType(TypeSet::SMALL), &[kb]);
            Some((branch_arm(b, is), kb))
        }
    };
    let i = match small {
        None => b.as_i32(key),
        Some((_, kb)) => b.ins(Op::Unbox(Rep::I32), &[kb]),
    };
    let n = b.ins(Op::TabASize, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    let outside = branch_arm(b, inside);
    let arr = b.ins(Op::TabArray, &[t]);
    let v = b.ins(Op::ArrLoad, &[arr, i]);
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let found = branch_arm(b, nil);
    let w = index(b, recv, key);
    arm_end(b, join, dst, w);
    b.switch_to(found);
    arm_end(b, join, dst, v);
    b.switch_to(outside);
    let w = index(b, recv, key);
    arm_end(b, join, dst, w);
    if let Some((other, _)) = small {
        b.switch_to(other);
        let w = index(b, recv, key);
        arm_end(b, join, dst, w);
    }
    enter_join(b, join);
}

/// `R[b][R[c]] = R[a]`: a small integer key inside the array part over a
/// non-nil value stores inline; the rest takes the helper.
pub(super) fn emit_settable(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    let (src, rt, rk) = (insn.a(), insn.b(), insn.c());
    let (recv, key, v) = (b.reg(rt), b.reg(rk), b.reg(src));
    let v = b.boxed(v);
    let Some(t) = table_site(b, pc, recv) else {
        newindex(b, recv, key, v, true);
        return;
    };
    let join = b.new_block();
    let small = match int_key(b, key) {
        Some(false) => {
            newindex(b, recv, key, v, true);
            return;
        }
        Some(true) => None,
        None => {
            let kb = b.boxed(key);
            let is = b.ins(Op::IsType(TypeSet::SMALL), &[kb]);
            Some((branch_arm(b, is), kb))
        }
    };
    let i = match small {
        None => b.as_i32(key),
        Some((_, kb)) => b.ins(Op::Unbox(Rep::I32), &[kb]),
    };
    let n = b.ins(Op::TabASize, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    let outside = branch_arm(b, inside);
    let arr = b.ins(Op::TabArray, &[t]);
    let old = b.ins(Op::ArrLoad, &[arr, i]);
    // `__newindex` may fire over a nil.
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[old]);
    let present = branch_arm(b, nil);
    newindex(b, recv, key, v, true);
    b.jump(join);
    b.switch_to(present);
    barrier_for(b, t, v);
    b.push(Op::ArrStore, &[arr, i, v]);
    b.jump(join);
    b.switch_to(outside);
    newindex(b, recv, key, v, true);
    b.jump(join);
    if let Some((other, _)) = small {
        b.switch_to(other);
        newindex(b, recv, key, v, true);
        b.jump(join);
    }
    enter_join(b, join);
}
