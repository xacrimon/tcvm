//! The emitter on hand-built graphs.

use std::cell::{RefCell, RefMut};

use super::emit::Emitter;
use super::ops::{Cc, ExitKind, ExitTag, Heaps, Op};
use super::table::{Key, NONE, Snap, Table};
use super::types::{Rep, TypeSet};
use super::verify::verify;
use super::{Block, BlockFlags, HOME, Inst, Loop, Val};
use crate::jit::compile::Bail;

/// An emitter with its entry block open, behind a cell so that calls nest.
struct T {
    e: RefCell<Emitter>,
    /// An empty frame state at pc 0.
    fs: Val,
}

fn start(lowered: bool) -> T {
    let mut e = Emitter::default();
    e.reset(lowered);
    let b = e.new_block();
    assert!(e.begin(b, Loop::NONE, 0, BlockFlags::empty()));
    let fs = e.frame_state(0, ExitKind::Before, Val::NONE, &[]);
    T {
        e: RefCell::new(e),
        fs,
    }
}

impl T {
    fn e(&self) -> RefMut<'_, Emitter> {
        self.e.borrow_mut()
    }

    fn emit(&self, op: Op, rep: Rep, aux: u16, a: Val, b: Val, c: Val) -> Val {
        self.e().emit(op, rep, aux, a.0, b.0, c.0)
    }

    fn op1(&self, op: Op, rep: Rep, a: Val) -> Val {
        self.e().op1(op, rep, a)
    }

    fn i32(&self, op: Op, a: Val, b: Val) -> Val {
        self.e().op2(op, Rep::I32, a, b)
    }

    fn f64(&self, op: Op, a: Val, b: Val) -> Val {
        self.e().op2(op, Rep::F64, a, b)
    }

    fn cmp(&self, op: Op, cc: Cc, a: Val, b: Val) -> Val {
        self.e().cmp(op, cc, a, b)
    }

    fn k(&self, n: i32) -> Val {
        self.e().k32(Rep::I32, n as u32)
    }

    fn kf(&self, x: f64) -> Val {
        self.e().kf64(x)
    }

    fn slot(&self, r: u16) -> Val {
        self.emit(Op::SlotLoad, Rep::Val, r, Val(0), Val(0), Val(0))
    }

    fn check(&self, v: Val, set: TypeSet, tag: ExitTag) -> Val {
        self.e().check_type(v, set, tag, self.fs)
    }

    /// Register `r` as a checked small integer.
    fn small(&self, r: u16) -> Val {
        self.check(self.slot(r), TypeSet::SMALL, ExitTag::Type)
    }

    /// An op that leaves to the frame state, which is its last operand.
    fn exiting(&self, op: Op, rep: Rep, tag: ExitTag, a: Val, b: Val) -> Val {
        self.emit(op, rep, tag as u16, a, b, self.fs)
    }

    fn is_type(&self, v: Val, set: TypeSet) -> Val {
        self.emit(Op::IsType, Rep::B1, set.bits(), v, Val(0), Val(0))
    }

    fn states(&self, pc: u32, kind: ExitKind, entries: &[(u32, Val)]) -> Val {
        self.e().frame_state(pc, kind, Val::NONE, entries)
    }

    fn inst(&self, v: Val) -> Inst {
        *self.e().g.inst(v)
    }

    fn op(&self, v: Val) -> Op {
        self.inst(v).op
    }

    fn ki(&self, v: Val) -> Option<i64> {
        self.e().ki(v)
    }

    fn set(&self, v: Val) -> TypeSet {
        self.e().fact(v).set
    }

    fn known(&self, v: Val) -> Option<bool> {
        self.e().known(v)
    }

    fn len(&self) -> usize {
        self.e().g.insts.len()
    }

    fn block(&self, b: Block) {
        assert!(self.e().begin(b, Loop::NONE, 0, BlockFlags::empty()));
    }

    /// Branch on `c` to two new blocks, which are to meet in a third.
    fn diamond(&self, c: Val) -> [Block; 3] {
        let mut e = self.e();
        let b = [e.new_block(), e.new_block(), e.new_block()];
        e.br(c, b[0], b[1]);
        b
    }

    fn jump(&self, to: Block) {
        self.e().jump(to)
    }

    /// End the graph and verify it.
    fn done(&self) -> String {
        self.emit(Op::Ret, Rep::None, 0, Val(0), Val(0), Val(0));
        let e = self.e();
        if let Err(why) = verify(&e.g, e.lowered) {
            panic!("{why}\n{}", e.g.print());
        }
        e.g.print()
    }
}

#[test]
fn integer_rules() {
    use Op::*;
    let t = start(false);
    let (x, y) = (t.small(0), t.small(1));
    assert_eq!(t.i32(Add, t.k(2), t.k(3)), t.k(5));
    assert_eq!(t.i32(Add, x, t.k(0)), x);
    assert_eq!(t.i32(Add, t.k(0), x), x);
    assert_eq!(t.i32(Sub, x, x), t.k(0));
    assert_eq!(t.i32(Mul, x, t.k(1)), x);
    assert_eq!(t.i32(Mul, x, t.k(0)), t.k(0));
    assert_eq!(t.i32(And, x, t.k(-1)), x);
    assert_eq!(t.i32(Or, x, t.k(-1)), t.k(-1));
    assert_eq!(t.i32(And, x, x), x);
    assert_eq!(t.i32(Shl, x, t.k(32)), x);
    // Commutative operands are ordered, so either order is one value.
    let sum = t.i32(Add, x, y);
    assert_eq!(t.op(sum), Add);
    assert_eq!(t.i32(Add, y, x), sum);
    assert_eq!(t.i32(Mul, x, t.k(2)), t.i32(Add, x, x));
    let not = t.i32(Xor, x, t.k(-1));
    assert_eq!(t.op(not), Not);
    assert_eq!(t.op1(Not, Rep::I32, not), x);
    let neg = t.i32(Mul, x, t.k(-1));
    assert_eq!(t.op(neg), Neg);
    assert_eq!(t.i32(Add, y, neg), t.i32(Sub, y, x));
    // x - 3 is x + -3, and (x + -3) + 5 is x + 2.
    let d = t.i32(Sub, x, t.k(3));
    assert_eq!(d, t.i32(Add, x, t.k(-3)));
    assert_eq!(t.i32(Add, d, t.k(5)), t.i32(Add, x, t.k(2)));
    // Shifts fold with the machine's count; a division that would trap stays.
    assert_eq!(t.i32(Shl, t.k(1), t.k(33)), t.k(2));
    assert_eq!(t.i32(Shr, t.k(-1), t.k(31)), t.k(1));
    assert_eq!(t.i32(Sar, t.k(-1), t.k(31)), t.k(-1));
    assert_eq!(t.op(t.i32(SDiv, t.k(i32::MIN), t.k(-1))), SDiv);
    assert_eq!(t.op(t.i32(SDiv, x, t.k(0))), SDiv);
    assert_eq!(t.i32(SDiv, t.k(-7), t.k(2)), t.k(-3));
    assert_eq!(t.i32(UDiv, t.k(-1), t.k(2)), t.k(i32::MAX));
    // 64-bit: a multiply by a power of two is a shift.
    let w = t.op1(SExt32, Rep::I64, x);
    let k8 = t.e().k64(Rep::I64, 8);
    let sh = t.e().op2(Mul, Rep::I64, w, k8);
    assert_eq!((t.op(sh), t.ki(Val(t.inst(sh).b))), (Shl, Some(3)));
    assert_eq!(t.op1(Trunc32, Rep::I32, w), x);
    // Compares: constants go right, equal operands decide.
    let c = t.cmp(Cmp, Cc::Lt, t.k(4), x);
    let i = t.inst(c);
    assert_eq!(
        (Cc::of(i.aux), Val(i.a), t.ki(Val(i.b))),
        (Cc::Gt, x, Some(4))
    );
    assert_eq!(t.known(t.cmp(Cmp, Cc::Le, x, x)), Some(true));
    assert_eq!(t.known(t.cmp(Cmp, Cc::Lo, t.k(1), t.k(-1))), Some(true));
    assert_eq!(t.known(t.cmp(Cmp, Cc::Lt, t.k(1), t.k(-1))), Some(false));
    t.done();
}

#[test]
fn checked_and_floor_rules() {
    use Op::*;
    let t = start(false);
    let x = t.small(0);
    let ovf = |op, a, b| t.exiting(op, Rep::I32, ExitTag::Overflow, a, b);
    assert_eq!(t.op(ovf(AddOvf, t.k(i32::MAX), t.k(1))), AddOvf);
    assert_eq!(ovf(AddOvf, t.k(1), t.k(2)), t.k(3));
    assert_eq!(ovf(MulOvf, x, t.k(1)), x);
    // A dominated identical checked op is the first one, whatever its exit.
    let s = ovf(AddOvf, x, t.k(1));
    let other = t.states(1, ExitKind::Before, &[]);
    assert_eq!(
        t.emit(AddOvf, Rep::I32, ExitTag::Overflow as u16, x, t.k(1), other),
        s
    );
    let floor = |op, a, b| t.ki(t.exiting(op, Rep::I32, ExitTag::Slow, t.k(a), t.k(b)));
    assert_eq!(floor(DivFloor, -7, 2), Some(-4));
    assert_eq!(floor(DivFloor, 7, -2), Some(-4));
    assert_eq!(floor(DivFloor, -7, -2), Some(3));
    assert_eq!(floor(DivFloor, 6, -2), Some(-3));
    assert_eq!(floor(ModFloor, -7, 2), Some(1));
    assert_eq!(floor(ModFloor, 7, -2), Some(-1));
    assert_eq!(floor(ModFloor, -7, -2), Some(-1));
    assert_eq!(floor(ModFloor, 6, -2), Some(0));
    assert_eq!(floor(DivFloor, 1, 0), None);
    assert_eq!(floor(DivFloor, i32::MIN, -1), None);
    let q = t.exiting(DivFloor, Rep::I32, ExitTag::Slow, x, t.k(8));
    let r = t.exiting(ModFloor, Rep::I32, ExitTag::Slow, x, t.k(8));
    assert_eq!((t.op(q), t.op(r)), (Sar, And));
    assert_eq!(
        t.op(t.exiting(DivFloor, Rep::I32, ExitTag::Slow, x, t.k(-1))),
        NegOvf
    );
    t.done();
}

#[test]
fn float_and_boxing_rules() {
    use Op::*;
    let t = start(false);
    let slot0 = t.slot(0);
    let x = t.check(slot0, TypeSet::SMALL, ExitTag::Type);
    let v = t.slot(1);
    let f = t.check(v, TypeSet::FLOAT, ExitTag::Type);
    assert_eq!(t.inst(f).rep, Rep::F64);
    assert_eq!(t.f64(FMul, f, t.kf(2.0)), t.f64(FAdd, f, f));
    assert_eq!(t.f64(FDiv, f, t.kf(2.0)), t.f64(FMul, f, t.kf(0.5)));
    assert_eq!(t.op(t.f64(FDiv, f, t.kf(3.0))), FDiv);
    // No NaN is made at compile time.
    assert_eq!(t.op(t.f64(FDiv, t.kf(0.0), t.kf(0.0))), FDiv);
    assert_eq!(t.f64(FAdd, t.kf(2.0), t.kf(0.5)), t.kf(2.5));
    let neg = t.op1(FNeg, Rep::F64, f);
    assert_eq!(t.op1(FNeg, Rep::F64, neg), f);
    assert_eq!(t.op1(FAbs, Rep::F64, neg), t.op1(FAbs, Rep::F64, f));
    let xf = t.op1(IToF, Rep::F64, x);
    assert_eq!(t.op1(FFloor, Rep::F64, xf), xf);
    // A small integer against an integral float compares as integers.
    let i = t.inst(t.cmp(FCmp, Cc::Lt, t.kf(2.0), xf));
    assert_eq!(
        (i.op, Cc::of(i.aux), Val(i.a), t.ki(Val(i.b))),
        (Cmp, Cc::Gt, x, Some(2))
    );
    assert_eq!(t.op(t.cmp(FCmp, Cc::Lt, xf, t.kf(0.5))), FCmp);
    // Boxes cancel; a box of what was unboxed is the value.
    assert_eq!(t.op1(BoxF64, Rep::Val, f), v);
    assert_eq!(t.op1(BoxI32, Rep::Val, x), slot0);
    let twice = t.i32(Add, x, x);
    let boxed = t.op1(BoxI32, Rep::Val, twice);
    assert_eq!((t.op(boxed), t.set(boxed)), (BoxI32, TypeSet::SMALL));
    assert_eq!(t.check(boxed, TypeSet::SMALL, ExitTag::Type), twice);
    let seven = t.op1(BoxI32, Rep::Val, t.k(7));
    assert_eq!(t.inst(seven).k64(), crate::env::Value::small(7).to_raw());
    assert_eq!(t.set(seven), TypeSet::SMALL);
    assert_eq!(t.op1(UnboxI64, Rep::I64, seven), t.e().k64(Rep::I64, 7));
    assert_eq!(t.set(t.op1(BoxF64, Rep::Val, t.kf(1.5))), TypeSet::FLOAT);
    let aux = TypeSet::NUM.bits() | (ExitTag::Type as u16) << TypeSet::BITS;
    assert_eq!(
        t.emit(ToF64, Rep::F64, aux, boxed, t.fs, Val(0)),
        t.op1(IToF, Rep::F64, twice)
    );
    assert_eq!(t.emit(ToF64, Rep::F64, aux, v, t.fs, Val(0)), f);
    t.done();
}

#[test]
fn numbering_follows_dominance() {
    use Op::*;
    let t = start(false);
    let (x, y) = (t.small(0), t.small(1));
    let above = t.i32(Sub, x, y);
    let [left, right, join] = t.diamond(t.cmp(Cmp, Cc::Lt, x, y));
    t.block(left);
    let in_left = t.i32(Add, x, y);
    assert_eq!(t.i32(Add, x, y), in_left);
    assert_eq!(t.i32(Sub, x, y), above);
    t.jump(join);
    t.block(right);
    let in_right = t.i32(Add, x, y);
    assert_ne!(in_right, in_left);
    t.jump(join);
    t.block(join);
    let p = t.e().param(Rep::I32, None, &[in_left, in_right]);
    let in_join = t.i32(Add, x, y);
    assert!(in_join != in_left && in_join != in_right);
    assert_eq!(t.i32(Sub, x, y), above);
    assert_eq!(t.op(t.i32(Add, p, x)), Add);
    t.done();
}

#[test]
#[should_panic(expected = "out of place")]
fn parameters_come_first() {
    let t = start(false);
    let x = t.small(0);
    let [left, right, join] = t.diamond(t.cmp(Op::Cmp, Cc::Lt, x, t.k(0)));
    t.block(left);
    t.jump(join);
    t.block(right);
    t.jump(join);
    t.block(join);
    t.i32(Op::Add, x, t.k(1));
    t.e().param(Rep::I32, None, &[x, x]);
}

#[test]
fn type_facts() {
    let t = start(false);
    let v = t.slot(0);
    let is_tab = t.is_type(v, TypeSet::TAB);
    assert_eq!(t.op(is_tab), Op::IsType);
    let [yes, no, join] = t.diamond(is_tab);
    t.block(yes);
    // Typed by the branch: the pointer is good from here on only.
    assert_eq!((t.set(v), t.known(is_tab)), (TypeSet::TAB, Some(true)));
    let p = t.check(v, TypeSet::TAB, ExitTag::Type);
    assert_eq!(t.op(p), Op::UnboxPtrPin);
    assert_eq!(
        t.emit(Op::BoxPtr, Rep::Val, TypeSet::TAB.bits(), p, Val(0), Val(0)),
        v
    );
    t.jump(join);
    t.block(no);
    assert_eq!(
        (t.set(v), t.known(is_tab)),
        (TypeSet::ANY - TypeSet::TAB, Some(false))
    );
    // A check, then a second one, which is the first; the fact holds after.
    let n = t.check(v, TypeSet::SMALL, ExitTag::Type);
    assert_eq!(t.op(n), Op::CheckType);
    assert_eq!(t.check(v, TypeSet::SMALL, ExitTag::Arg), n);
    assert_eq!(t.set(v), TypeSet::SMALL);
    assert_eq!(t.known(t.is_type(v, TypeSet::NUM)), Some(true));
    assert_eq!(t.known(t.is_type(v, TypeSet::STR)), Some(false));
    let before = t.len();
    assert_eq!(t.check(v, TypeSet::NUM, ExitTag::Type), v);
    assert_eq!(t.len(), before);
    t.jump(join);
    t.block(join);
    // What both arms know, and no more.
    assert_eq!(
        (t.set(v), t.known(is_tab)),
        (TypeSet::TAB | TypeSet::SMALL, None)
    );
    let again = t.check(v, TypeSet::SMALL, ExitTag::Type);
    assert!(again != n && t.op(again) == Op::CheckType);
    t.done();
}

#[test]
fn guards_and_known_conditions() {
    let t = start(false);
    let (x, y) = (t.small(0), t.small(1));
    let c = t.cmp(Op::Cmp, Cc::Lo, x, y);
    let guard = |op, pass: Val| {
        let rep = if pass.is_none() { Rep::None } else { Rep::I32 };
        t.emit(op, rep, ExitTag::Bounds as u16, c, t.fs, pass)
    };
    // The checked index is the guard's result; an equal guard is that one.
    let checked = guard(Op::Guard, x);
    assert!(checked != x && t.op(checked) == Op::Guard);
    assert_eq!(guard(Op::Guard, x), checked);
    // Without a result, a guard on a known condition is nothing.
    let before = t.len();
    guard(Op::Guard, Val::NONE);
    assert_eq!(t.len(), before);
    // One that always leaves stays.
    guard(Op::GuardNot, Val::NONE);
    assert_eq!(t.len(), before + 1);
    assert_eq!(t.emit(Op::Select, Rep::I32, 0, c, x, y), x);
    // A branch on it is a jump: the other arm gets no edge and never opens.
    let [taken, dead, _] = t.diamond(c);
    assert_eq!(t.op(Val(t.len() as u32 - 1)), Op::Jump);
    assert!(!t.e().begin(dead, Loop::NONE, 0, BlockFlags::empty()));
    t.block(taken);
    t.done();
}

#[test]
fn loop_parameters_and_frame_states() {
    let t = start(false);
    let n = t.small(0);
    let zero = t.k(0);
    let (head, body, exit, lp) = {
        let mut e = t.e();
        let (head, body, exit) = (e.new_block(), e.new_block(), e.new_block());
        e.jump(head);
        let lp = e.new_loop(head, Loop::NONE, Heaps::empty(), t.fs);
        assert!(e.begin(head, lp, 1, BlockFlags::LOOP_HEADER));
        (head, body, exit, lp)
    };
    let i = t.e().param(Rep::I32, None, &[zero]);
    let more = t.cmp(Op::Cmp, Cc::Lt, i, n);
    t.e().br(more, body, exit);
    assert!(t.e().begin(body, lp, 0, BlockFlags::empty()));
    let boxed = t.op1(Op::BoxI32, Rep::Val, i);
    let entries = [(HOME, t.slot(0)), (2, boxed)];
    let state = t.states(4, ExitKind::Before, &entries);
    // An equal state is the same one and leaves nothing behind.
    let extra = t.e().g.extra.len();
    assert_eq!(t.states(4, ExitKind::Before, &entries), state);
    assert_eq!(t.e().g.extra.len(), extra);
    assert_ne!(t.states(4, ExitKind::After, &entries), state);
    let next = t.emit(
        Op::AddOvf,
        Rep::I32,
        ExitTag::Overflow as u16,
        i,
        t.k(1),
        state,
    );
    let k = t.e().next_pred(head);
    t.e().set_input(i, k, next);
    t.jump(head);
    t.block(exit);
    // The loop's value leaves through a parameter of the exit block.
    let out = t.e().param(Rep::I32, None, &[i]);
    t.op1(Op::BoxI32, Rep::Val, out);
    assert_eq!(
        t.done(),
        "\
b0:
  v0 = framestate.before pc0
  v1: val = slotload r0
  v2: i32 = checktype.small!type v1 v0
  v3: i32 = k32 0
  v4 = jump b1
b1: <- b0 b2 l0 header
  v5: i32 = param v3 v12
  v6: b1 = cmp.lt v5 v2
  v7 = br v6 b2 b3
b2: <- b1 l0
  v8: val = boxi32 v5
  v9 = framestate.before pc4 r0=*v1 r2=v5
  v10 = framestate.after pc4 r0=*v1 r2=v5
  v11: i32 = k32 1
  v12: i32 = addovf!overflow v5 v11 v9
  v13 = jump b1
b3: <- b1
  v14: i32 = param v5
  v15: val = boxi32 v14
  v16 = ret 0 0
"
    );
}

#[test]
fn memory() {
    use super::fields::Field;
    let t = start(false);
    let table = |r| t.check(t.slot(r), TypeSet::TAB, ExitTag::Type);
    let (a, b) = (table(0), table(1));
    let load = |f: Field, base: Val, k: u32| {
        let rep = Field::info(f as u16).rep;
        t.emit(Op::Load, rep, f as u16, base, Val::NONE, Val(k))
    };
    let store = |f: Field, base: Val, v: Val, k: u32| {
        t.emit(Op::Store, Rep::None, f as u16, base, v, Val(k))
    };
    // A load again is the load; a slot is read once.
    let x = load(Field::Inline, a, 0);
    assert_eq!(load(Field::Inline, a, 0), x);
    assert_eq!(t.slot(0), t.slot(0));
    // A store to another slot leaves it, one to the same slot of any table
    // does not, and the stored value is what a load there gives.
    let seven = t.op1(Op::BoxI32, Rep::Val, t.k(7));
    store(Field::Inline, b, seven, 1);
    assert_eq!(load(Field::Inline, a, 0), x);
    assert_eq!(load(Field::Inline, b, 1), seven);
    store(Field::Inline, b, seven, 0);
    assert_ne!(load(Field::Inline, a, 0), x);
    // A store of what is there is nothing.
    let n = t.len();
    store(Field::Inline, b, seven, 0);
    assert_eq!(t.len(), n);
    // Heaps are apart: a named store leaves the array's size.
    let size = load(Field::ASize, a, 0);
    store(Field::Inline, a, seven, 2);
    assert_eq!(load(Field::ASize, a, 0), size);
    // What a slot holds is what was stored there, until a branch's arms
    // disagree.
    t.emit(Op::SlotStore, Rep::None, 3, seven, Val(0), Val(0));
    assert_eq!(t.slot(3), seven);
    assert!(t.e().slot_holds(3, seven));
    let [left, right, join] = t.diamond(t.cmp(Op::Cmp, Cc::Lt, t.small(4), t.k(0)));
    t.block(left);
    t.emit(Op::SlotStore, Rep::None, 3, x, Val(0), Val(0));
    t.jump(join);
    t.block(right);
    t.jump(join);
    t.block(join);
    assert!(!t.e().slot_holds(3, seven));
    assert_eq!(load(Field::Inline, b, 1), seven);
    t.done();
}

#[test]
fn snapshot_table() {
    let mut t = Table::default();
    let (a, b, c) = (Key::reg(0), Key::reg(1), Key::fact(7));
    t.start(&[]);
    t.set(a, 1);
    t.set(c, 5);
    let root = t.seal();
    t.start(&[root]);
    t.set(a, 2);
    t.set(b, 9);
    let left = t.seal();
    t.start(&[root]);
    assert_eq!((t.get(a), t.get(b), t.get(c)), (1, NONE, 5));
    t.set(b, 9);
    t.set(c, 6);
    let right = t.seal();
    // At the join the ancestor's values stand, and each changed key lists
    // what the predecessors hold.
    t.start(&[left, right]);
    assert_eq!((t.get(a), t.get(b), t.get(c)), (1, NONE, 5));
    let mut seen: Vec<(Key, Vec<u32>)> = (0..t.merges())
        .map(|i| (t.merge(i).0, t.merge(i).1.to_vec()))
        .collect();
    seen.sort_by_key(|(k, _)| (k.is_fact(), k.index()));
    assert_eq!(seen, [(a, vec![2, 1]), (b, vec![9, 9]), (c, vec![5, 6])]);
    assert_eq!((t.pred_value(a, 0), t.pred_value(a, 1)), (2, 1));
    t.set(b, 9);
    let join = t.seal();
    t.start(&[left]);
    assert_eq!((t.get(a), t.get(b), t.get(c)), (2, 9, 5));
    t.start(&[join]);
    assert_eq!((t.get(a), t.get(b), t.get(c)), (1, 9, 5));
    t.start(&[Snap::ROOT]);
    assert_eq!((t.get(a), t.get(b), t.get(c)), (NONE, NONE, NONE));
}

#[test]
fn verifier_rejects() {
    let t = start(false);
    let x = t.small(0);
    let s = t.i32(Op::Add, x, x);
    t.done();
    let mut e = t.e();
    // A use that is not before its user.
    e.g.insts[s.idx()].b = s.0;
    assert!(
        verify(&e.g, false)
            .unwrap_err()
            .contains("does not dominate")
    );
    e.g.insts[s.idx()].b = x.0;
    // An operand of the wrong representation.
    e.g.insts[s.idx()].b = t.fs.0;
    assert!(verify(&e.g, false).unwrap_err().contains("breaks its form"));
    e.g.insts[s.idx()].b = x.0;
    // A high-level op where only low-level ones may be.
    assert!(verify(&e.g, true).unwrap_err().contains("high-level"));
    drop(e);
    // An unbox that nothing justifies.
    let t = start(false);
    let p = t.op1(Op::UnboxPtrPin, Rep::Ptr, t.slot(0));
    t.done();
    let mut e = t.e();
    e.g.insts[p.idx()].op = Op::UnboxPtr;
    assert!(verify(&e.g, false).unwrap_err().contains("unchecked unbox"));
}

#[test]
#[should_panic(expected = "breaks its form")]
fn operand_representations_are_checked() {
    let t = start(false);
    t.i32(Op::Add, t.small(0), t.kf(1.0));
}

#[test]
#[should_panic(expected = "high-level")]
fn low_level_graphs_take_no_high_level_ops() {
    let t = start(true);
    t.op1(Op::UnboxI64, Rep::I64, t.slot(0));
}

#[test]
fn a_bail_is_sticky() {
    let t = start(false);
    let x = t.small(0);
    t.e().bail(Bail::TooLarge);
    t.e().bail(Bail::Useless);
    let n = t.len();
    t.i32(Op::Add, x, x);
    assert_eq!((t.len(), t.e().bail), (n, Some(Bail::TooLarge)));
}
