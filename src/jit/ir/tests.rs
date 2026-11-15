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
    // The pointer is of what the branch proved, which is made once.
    let p = t.check(v, TypeSet::TAB, ExitTag::Type);
    let proved = Val(t.inst(p).a);
    assert_eq!((t.op(p), t.op(proved)), (Op::UnboxPtr, Op::Proved));
    assert_eq!(
        (t.inst(proved).a, t.inst(proved).c, t.inst(proved).aux),
        (is_tab.0, v.0, TypeSet::TAB.bits())
    );
    assert_eq!(t.set(proved), TypeSet::TAB);
    let n = t.len();
    assert_eq!(t.check(v, TypeSet::TAB, ExitTag::Type), p);
    assert_eq!(t.len(), n);
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
    t.op1(Op::UnboxPtr, Rep::Ptr, t.slot(0));
    t.emit(Op::Ret, Rep::None, 0, Val(0), Val(0), Val(0));
    let e = t.e();
    assert!(verify(&e.g, false).unwrap_err().contains("unchecked unbox"));
    drop(e);
    // What a branch proved, outside the branch, and in its other arm.
    for wrong in [0, 1] {
        let t = start(false);
        let v = t.slot(0);
        let is_tab = t.is_type(v, TypeSet::TAB);
        if wrong == 0 {
            t.e().proved(is_tab, true, v, TypeSet::TAB);
        }
        let [yes, no, join] = t.diamond(is_tab);
        t.block(yes);
        t.jump(join);
        t.block(no);
        if wrong == 1 {
            t.e().proved(is_tab, true, v, TypeSet::TAB);
        }
        t.jump(join);
        t.block(join);
        t.emit(Op::Ret, Rep::None, 0, Val(0), Val(0), Val(0));
        let e = t.e();
        assert!(
            verify(&e.g, false)
                .unwrap_err()
                .contains("not under the branch")
        );
    }
}

#[test]
fn proved_values() {
    let not_tab = TypeSet::ANY - TypeSet::TAB;
    let t = start(false);
    let v = t.slot(0);
    // The false arm of a test for everything but a table: a table.
    let other = t.is_type(v, not_tab);
    let [yes, no, join] = t.diamond(other);
    t.block(yes);
    t.jump(join);
    t.block(no);
    let p = t.check(v, TypeSet::TAB, ExitTag::Type);
    let proved = t.inst(Val(t.inst(p).a));
    assert_eq!((t.op(p), proved.op), (Op::UnboxPtr, Op::Proved));
    assert_eq!(proved.aux, TypeSet::TAB.bits() | super::ops::SET_ARM);
    t.jump(join);
    t.block(join);
    // Each arm of a second test proves it; past their join nothing names
    // one branch, and the check is made again.
    let is_tab = t.is_type(v, TypeSet::TAB);
    let [yes, no, join] = t.diamond(is_tab);
    t.block(yes);
    t.jump(join);
    t.block(no);
    t.check(v, TypeSet::TAB, ExitTag::Type);
    t.jump(join);
    t.block(join);
    assert_eq!(t.set(v), TypeSet::TAB);
    let again = t.check(v, TypeSet::TAB, ExitTag::Type);
    assert_eq!(t.op(again), Op::CheckType);
    t.done();
    // A constant condition is a folded branch's: the value as something
    // else types it, here its own definition.
    let t = start(false);
    let x = t.op1(Op::BoxI32, Rep::Val, t.i32(Op::Add, t.small(0), t.k(1)));
    let yes = t.e().k32(Rep::B1, 1);
    assert_eq!(t.e().proved(yes, true, x, TypeSet::SMALL), x);
    t.done();
}

#[test]
fn arms_and_joins() {
    use super::fields::Field;
    let t = start(false);
    let (x, y) = (t.small(0), t.small(1));
    let table = t.check(t.slot(2), TypeSet::TAB, ExitTag::Type);
    let load = |k: u32| {
        t.emit(
            Op::Load,
            Rep::Val,
            Field::Inline as u16,
            table,
            Val::NONE,
            Val(k),
        )
    };
    let (kept, lost) = (load(0), load(1));
    let c = t.cmp(Op::Cmp, Cc::Lt, x, y);
    // Two arms: one value they agree on, one they do not; a store in one.
    let mut e = t.e();
    let mut join = e.new_join::<2>();
    let (yes, no) = e.fork(c, false, true);
    assert!(e.arm(yes));
    let a = e.op2(Op::Add, Rep::I32, x, y);
    e.leave(&mut join, [x, a]);
    assert!(e.arm(no));
    let seven = e.k32(Rep::I32, 7);
    let boxed = e.op1(Op::BoxI32, Rep::Val, seven);
    e.emit(
        Op::Store,
        Rep::None,
        Field::Inline as u16,
        table.0,
        boxed.0,
        1,
    );
    e.leave(&mut join, [x, y]);
    let [same, differ] = e.join(join).expect("both arms came");
    drop(e);
    assert_eq!(same, x);
    assert_eq!(t.op(differ), Op::Param);
    assert_eq!(t.e().g.inputs(differ), [a.0, y.0]);
    // What both arms left of memory stays; what one wrote does not.
    assert_eq!(load(0), kept);
    assert_ne!(load(1), lost);
    // An arm that leaves the region brings nothing, and a known condition
    // leaves an arm nothing reaches.
    let mut e = t.e();
    let mut join = e.new_join::<1>();
    let (yes, no) = e.fork(c, true, false);
    assert!(e.arm(yes));
    e.emit(Op::Deopt, Rep::None, ExitTag::Type as u16, t.fs.0, 0, 0);
    assert!(e.arm(no));
    e.leave(&mut join, [y]);
    assert_eq!(e.join(join), Some([y]));
    let never = e.k32(Rep::B1, 0);
    let mut join = e.new_join::<1>();
    let cold = e.fork_or_leave(never, true, &mut join, [x]);
    assert!(!e.arm(cold));
    assert_eq!(e.join(join), Some([x]));
    drop(e);
    // `unless` over them: the cold arm, then on.
    let n = t.e().g.blocks.len();
    let c = t.cmp(Op::Cmp, Cc::Gt, x, y);
    t.e().unless(c, |e| {
        e.emit(Op::Deopt, Rep::None, ExitTag::Type as u16, t.fs.0, 0, 0);
    });
    assert_eq!(t.e().g.blocks.len(), n + 2);
    t.done();
}

#[test]
fn loops_of_their_own() {
    use super::fields::Field;
    let t = start(false);
    let n = t.small(0);
    let table = t.check(t.slot(1), TypeSet::TAB, ExitTag::Type);
    let load = |f: Field| {
        let rep = Field::info(f as u16).rep;
        t.emit(Op::Load, rep, f as u16, table, Val::NONE, Val(0))
    };
    let (size, named) = (load(Field::ASize), load(Field::Inline));
    let zero = t.k(0);
    let one = t.k(1);
    // Count to `n`, leaving two ways for one join: at the end, and early.
    let mut e = t.e();
    let mut out = e.new_join::<1>();
    let (lp, [i]) = e.loop_open(Heaps::FIELD, t.fs, [zero]);
    assert_eq!(e.g.op(i), Op::Param);
    let more = e.cmp(Op::Cmp, Cc::Lt, i, n);
    let body = e.fork_or_leave(more, false, &mut out, [i]);
    assert!(e.arm(body));
    let early = e.cmp(Op::Cmp, Cc::Eq, i, one);
    let (stop, on) = e.fork(early, true, false);
    assert!(e.arm(stop));
    e.leave(&mut out, [i]);
    assert!(e.arm(on));
    let next = e.op2(Op::Add, Rep::I32, i, one);
    e.loop_back(lp, [next]);
    // The loop's value leaves through a parameter, kept though its inputs
    // are one value: the loop-closed form.
    let [last] = e.join(out).expect("the exits came");
    drop(e);
    assert!(last != i && t.op(last) == Op::Param);
    assert_eq!(t.e().g.inputs(last), [i.0, i.0]);
    assert_eq!(t.e().g.inputs(i), [zero.0, next.0]);
    // What the loop was said to write is unknown past its header; the
    // rest stands.
    assert_eq!(load(Field::ASize), size);
    assert_ne!(load(Field::Inline), named);
    t.op1(Op::BoxI32, Rep::Val, last);
    t.done();
}

#[test]
fn table_columns() {
    use super::ops::{FactOf, Mem, OPS};
    let facts: Vec<(&str, FactOf)> = (OPS.iter())
        .filter(|o| o.fact != FactOf::Rep)
        .map(|o| (o.name, o.fact))
        .collect();
    assert_eq!(
        facts,
        [
            ("K64", FactOf::Konst),
            ("KObj", FactOf::AuxSet),
            ("Load", FactOf::Field),
            ("LoadIdx", FactOf::Field),
            ("Select", FactOf::JoinBC),
            ("BoxI32", FactOf::Set(TypeSet::SMALL)),
            ("BoxF64", FactOf::Set(TypeSet::FLOAT)),
            ("BoxPtr", FactOf::AuxSet),
            ("UnboxPtr", FactOf::HeapA),
            ("Guard", FactOf::PassC),
            ("GuardNot", FactOf::PassC),
            ("Proved", FactOf::MeetC),
            ("CallHelper", FactOf::Helper),
            ("CheckType", FactOf::AuxSet),
            ("BoxI64", FactOf::Set(TypeSet::INT)),
            ("NewTable", FactOf::Set(TypeSet::TAB)),
        ]
    );
    let mem: Vec<(&str, Mem)> = (OPS.iter())
        .filter(|o| o.mem != Mem::No)
        .map(|o| (o.name, o.mem))
        .collect();
    assert_eq!(
        mem,
        [
            ("Load", Mem::Load),
            ("Store", Mem::Store),
            ("LoadIdx", Mem::LoadIdx),
            ("StoreIdx", Mem::StoreIdx),
            ("SlotLoad", Mem::SlotLoad),
            ("SlotStore", Mem::SlotStore),
        ]
    );
    // The ops a copy has an arm of its own for.
    let own: Vec<&str> = (OPS.iter())
        .filter(|o| !o.flags.contains(super::ops::Flags::PLAIN))
        .map(|o| o.name)
        .collect();
    assert_eq!(
        own,
        [
            "FrameState",
            "Param",
            "Jump",
            "Br",
            "CallHelper",
            "SetIndex"
        ]
    );
    let tab: Vec<&str> = (super::ops::HELPERS.iter())
        .filter(|h| h.fact != TypeSet::ANY)
        .map(|h| h.name)
        .collect();
    assert_eq!(tab, ["BoxInt", "NewTable"]);
}

/// How many instructions of `op` the graph has.
fn count(t: &T, op: Op) -> usize {
    t.e().g.insts.iter().filter(|i| i.op == op).count()
}

/// The unit ops with no blocks, each as the sequence its rule gives in a
/// low-level graph.
#[test]
fn unit_ops_lower_in_line() {
    use Op::*;
    let t = start(true);
    let tag = ExitTag::Type as u16;
    let none = Val(0);
    let x = t.check(t.slot(0), TypeSet::FLOAT, ExitTag::Type);
    let y = t.check(t.slot(1), TypeSet::FLOAT, ExitTag::Type);
    // An exact conversion: the conversion, checked by converting back.
    let i = t.emit(FToIExact, Rep::I32, tag, x, t.fs, none);
    let (g, n) = (t.inst(i), t.inst(Val(t.inst(i).c)));
    assert_eq!((g.op, g.rep, n.op, n.a), (Guard, Rep::I32, FToI, x.0));
    let same = t.inst(Val(g.a));
    assert_eq!((same.op, t.op(Val(same.a)), same.b), (FCmp, IToF, x.0));
    // To 64 bits, also below 2^63.
    let l = t.emit(FToLExact, Rep::I64, tag, y, t.fs, none);
    let both = t.inst(Val(t.inst(l).a));
    assert_eq!(
        (t.op(l), both.op, t.op(Val(t.inst(l).c))),
        (Guard, And, FToL)
    );
    let below = t.inst(Val(both.b));
    assert_eq!(
        (below.op, t.e().kf(Val(below.b))),
        (FCmp, Some(2f64.powi(63)))
    );
    // A 64-bit integer that is its low half extended.
    let l = Val(t.inst(l).c);
    let narrow = t.emit(LToI, Rep::I32, tag, l, t.fs, none);
    let (g, low) = (t.inst(narrow), t.inst(Val(t.inst(narrow).c)));
    assert_eq!((g.op, low.op, low.a), (Guard, Trunc32, l.0));
    let fits = t.inst(Val(g.a));
    assert_eq!((fits.op, t.op(Val(fits.a)), fits.b), (Cmp, SExt32, l.0));
    // The absolute value: the minimum leaves, the rest is a select.
    let s = t.small(2);
    let abs = t.emit(AbsOvf, Rep::I32, ExitTag::Overflow as u16, s, t.fs, none);
    let sel = t.inst(abs);
    assert_eq!(
        (sel.op, t.op(Val(sel.b)), t.op(Val(sel.c))),
        (Select, Neg, Guard)
    );
    // Float floor division and modulo.
    let q = t.f64(FIDiv, x, y);
    assert_eq!((t.op(q), t.op(Val(t.inst(q).a))), (FFloor, FDiv));
    let m = t.f64(FMod, x, y);
    let call = t.inst(m);
    assert_eq!(
        (call.op, call.aux),
        (CallHelper, super::ops::Helper::FMod as u16)
    );
    // Lua's shifts: each way where its count is in range, else zero; the
    // `I32` form over the 64-bit one, leaving where the result is wider.
    let by = t.small(3);
    let (shl, shr) = (count(&t, Shl), count(&t, Shr));
    let wide = t.emit(LuaShl, Rep::I64, tag, l, l, Val::NONE);
    let (outer, inner) = (t.inst(wide), t.inst(Val(t.inst(wide).c)));
    assert_eq!(
        (outer.op, t.op(Val(outer.b)), inner.op),
        (Select, Shl, Select)
    );
    assert_eq!((t.op(Val(inner.b)), t.ki(Val(inner.c))), (Shr, Some(0)));
    let right = t.emit(LuaShr, Rep::I64, tag, l, l, Val::NONE);
    assert_eq!(t.op(Val(t.inst(right).b)), Shr);
    let small = t.emit(LuaShl, Rep::I32, tag, s, by, t.fs);
    assert_eq!((t.op(small), t.inst(small).rep), (Guard, Rep::I32));
    assert_eq!((count(&t, Shl) - shl, count(&t, Shr) - shr), (3, 3));
    // A table from its template, and the collector's check.
    let template = t.e().k64(Rep::I64, 0x1000);
    let table = t.op1(NewTable, Rep::Val, template);
    assert_eq!((t.op(table), t.set(table)), (CallHelper, TypeSet::TAB));
    let check = t.emit(GcCheck, Rep::None, ExitTag::Gc as u16, t.fs, none, none);
    let below = t.inst(Val(t.inst(check).a));
    assert_eq!(
        (t.op(check), below.op, t.op(Val(below.a))),
        (Guard, Cmp, Load)
    );
    // A boxed integer's payload.
    let big = t.check(t.slot(4), TypeSet::BIGINT, ExitTag::Type);
    assert_eq!(t.op(big), UnboxPtr);
    let boxed = t.emit(BoxPtr, Rep::Val, TypeSet::BIGINT.bits(), big, none, none);
    let payload = t.op1(UnboxI64, Rep::I64, boxed);
    let load = t.inst(payload);
    assert_eq!(
        (load.op, load.aux),
        (Load, super::fields::Field::BoxedInt as u16)
    );
    t.done();
}

/// The unit ops that take arms, through `lower`.
#[test]
fn unit_ops_rewrite() {
    use Op::*;
    let lower =
        |t: &T, op: Op, rep: Rep, aux: u16, a: Val, b: Val| t.e().lower(op, rep, aux, a.0, b.0, 0);
    let none = Val(0);
    // The barrier: the header's test, and the helper in a cold arm.
    let t = start(true);
    let table = t.check(t.slot(0), TypeSet::TAB, ExitTag::Type);
    lower(&t, Barrier, Rep::None, 0, table, none);
    assert_eq!((count(&t, Br), count(&t, CallHelper)), (1, 1));
    t.done();
    // A 64-bit integer boxed: small where it fits, else the helper.
    let t = start(true);
    let x = t.check(t.slot(0), TypeSet::FLOAT, ExitTag::Type);
    let l = t.op1(FToL, Rep::I64, x);
    let boxed = lower(&t, BoxI64, Rep::Val, 0, l, none);
    let ins = t.e().g.inputs(boxed).to_vec();
    let ins: Vec<Op> = ins.iter().map(|&v| t.op(Val(v))).collect();
    assert_eq!((t.op(boxed), ins), (Param, vec![BoxI32, CallHelper]));
    assert_eq!(t.set(boxed), TypeSet::INT);
    // An integer unboxed: each kind in its arm, read as what the branch
    // proved.
    let (brs, proved) = (count(&t, Br), count(&t, Proved));
    let wide = lower(&t, UnboxI64, Rep::I64, 0, boxed, none);
    let ins = t.e().g.inputs(wide).to_vec();
    let ins: Vec<Op> = ins.iter().map(|&v| t.op(Val(v))).collect();
    assert_eq!((t.op(wide), ins), (Param, vec![SExt32, Load]));
    assert_eq!((count(&t, Br) - brs, count(&t, Proved) - proved), (1, 2));
    // A number as a float: an arm per kind, the last one untested.
    let float = t.op1(BoxF64, Rep::Val, x);
    let which = t.cmp(FCmp, Cc::Lt, x, x);
    let v = t.emit(Select, Rep::Val, 0, which, boxed, float);
    assert_eq!(t.set(v), TypeSet::NUM);
    let (brs, proved) = (count(&t, Br), count(&t, Proved));
    let aux = TypeSet::NUM.bits() | (ExitTag::Type as u16) << TypeSet::BITS;
    let f = lower(&t, ToF64, Rep::F64, aux, v, t.fs);
    let ins = t.e().g.inputs(f).to_vec();
    let ins: Vec<Op> = ins.iter().map(|&v| t.op(Val(v))).collect();
    assert_eq!((t.op(f), ins), (Param, vec![UnboxF64, IToF, LToF]));
    assert_eq!((count(&t, Br) - brs, count(&t, Proved) - proved), (2, 3));
    assert_eq!(count(&t, Deopt), 0);
    t.done();
    // Only the kinds the set takes, and a last arm that leaves for a value
    // the fact still allows outside it.
    let t = start(true);
    let v = t.slot(0);
    let ints = TypeSet::INT.bits() | (ExitTag::Type as u16) << TypeSet::BITS;
    let f = lower(&t, ToF64, Rep::F64, ints, v, t.fs);
    assert_eq!(t.e().g.inputs(f).len(), 2);
    assert_eq!((count(&t, Br), count(&t, Deopt)), (2, 1));
    t.done();
}

/// A repeat of a barrier or of the collector's check, with no allocation
/// between, is the first one.
#[test]
fn barriers_and_checks_repeat() {
    use Op::*;
    let t = start(false);
    let table = t.check(t.slot(0), TypeSet::TAB, ExitTag::Type);
    let (heap, number) = (t.slot(1), t.op1(BoxI32, Rep::Val, t.small(2)));
    t.e().barrier(table, number);
    assert_eq!(count(&t, Barrier), 0);
    t.e().barrier(table, heap);
    t.e().barrier(table, heap);
    assert_eq!(count(&t, Barrier), 1);
    let check = |t: &T| t.emit(GcCheck, Rep::None, ExitTag::Gc as u16, t.fs, Val(0), Val(0));
    check(&t);
    check(&t);
    assert_eq!(count(&t, GcCheck), 1);
    let template = t.e().k64(Rep::I64, 0x1000);
    t.op1(NewTable, Rep::Val, template);
    t.e().barrier(table, heap);
    check(&t);
    assert_eq!((count(&t, Barrier), count(&t, GcCheck)), (2, 2));
    t.done();
}

#[test]
#[should_panic(expected = "high-level")]
fn a_rewrite_is_not_a_rule() {
    let t = start(true);
    let table = t.check(t.slot(0), TypeSet::TAB, ExitTag::Type);
    t.emit(Op::Barrier, Rep::None, 0, table, Val(0), Val(0));
}

/// An array access through its rewrite: the table and key checks, the
/// bounds guard, the load, and the cold arm a nil takes, whose class is
/// read as what its branch proved.
#[test]
fn index_rewrite() {
    let t = start(true);
    let (recv, key) = (t.slot(0), t.slot(1));
    let v = (t.e()).lower(
        Op::GetIndex,
        Rep::Val,
        super::access::ARRAY,
        recv.0,
        key.0,
        t.fs.0,
    );
    assert_eq!(t.op(v), Op::LoadIdx);
    let e = t.e();
    let has = |op: Op| e.g.insts.iter().filter(|i| i.op == op).count();
    assert_eq!((has(Op::Br), has(Op::Proved), has(Op::GetIndex)), (2, 1, 0));
    assert!(e.g.blocks.len() > 1 && e.cur != Block(0));
    drop(e);
    t.done();
}

#[test]
#[should_panic(expected = "high-level")]
fn emit_runs_no_rewrite() {
    let t = start(true);
    let (recv, key) = (t.slot(0), t.slot(1));
    t.emit(
        Op::GetIndex,
        Rep::Val,
        super::access::ARRAY,
        recv,
        key,
        t.fs,
    );
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
