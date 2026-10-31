//! A prototype's feedback vector: per site, the interpreter's inline cache or
//! callee, and what slow paths and compiled code's exits saw there, in slots
//! of 64-bit words laid out by the compiler.
//!
//! Every reference held here is weak: the vector's trace skips them and defers
//! its prototype, and `clear_dead` empties each dead one before the sweep, so
//! a reference read here always names a live object.

use std::cell::Cell;

use crate::dmm::{Collect, Finalization, Gc, Trace};
use crate::env::function::{Code, InlineCache, LuaFn, Prototype};
use crate::env::shape::Shape;
use crate::env::value::Value;
use crate::instruction::{Family, Instruction, Op};

/// What a site's slot holds. Per word in [`FeedbackVector::kinds`]: the kind
/// at a slot's first word, `Cont` after it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum SlotKind {
    Cont,
    /// The slot overflowing sites share: an entry that stays empty and a
    /// megamorphic callee word.
    Null,
    /// Constant-key access: the site word, the inline cache (4 words).
    Field,
    /// The site word, the callee word, the call count.
    Call,
    /// GETTABLE, SETTABLE.
    Table,
    /// Arithmetic, bitwise, unary, LEN, CONCAT.
    Arith,
    /// Compares and conditional branches.
    Compare,
    /// FORPREP, FORLOOP.
    Loop,
}

impl SlotKind {
    pub(crate) const fn words(self) -> usize {
        match self {
            SlotKind::Null => NULL_WORDS,
            SlotKind::Field => 5,
            SlotKind::Call => 3,
            _ => 1,
        }
    }

    /// The slot an instruction of `op` gets, if any.
    pub(crate) fn of(op: Op) -> Option<SlotKind> {
        let info = op.info();
        Some(match op {
            Op::GETTABUP
            | Op::SETTABUP
            | Op::GETFIELD
            | Op::SETFIELD
            | Op::SELF
            | Op::GETTABUP_REF
            | Op::SETTABUP_REF => SlotKind::Field,
            _ if info.family == Family::Field => SlotKind::Field,
            Op::CALL
            | Op::CALL_R0
            | Op::CALL_R1
            | Op::CALLS
            | Op::CALLS_R0
            | Op::CALLS_R1
            | Op::TAILCALL => SlotKind::Call,
            Op::GETTABLE | Op::SETTABLE => SlotKind::Table,
            Op::UNM | Op::BNOT | Op::LEN | Op::CONCAT => SlotKind::Arith,
            Op::ARITH_MM | Op::ARITH_MM_R | Op::ARITH_MMI => SlotKind::Arith,
            Op::FORPREP | Op::FORLOOP | Op::FORLOOP_I | Op::FORLOOP_F => SlotKind::Loop,
            _ => match info.family {
                Family::RegArith | Family::RegBit | Family::ImmArith | Family::ImmBit => {
                    SlotKind::Arith
                }
                Family::CmpReg | Family::CmpImm => SlotKind::Compare,
                _ if op.branch_sense().is_some() => SlotKind::Compare,
                _ => return None,
            },
        })
    }
}

/// [`FeedbackVector::slot_of_pc`] of an instruction without a slot.
pub(crate) const NO_SLOT: u16 = u16::MAX;
/// Words a vector may hold: a slot index is a `u16` short of `NO_SLOT`.
const MAX_WORDS: usize = NO_SLOT as usize;
/// The null slot: a field slot, then a call slot.
const NULL_WORDS: usize = 8;
const NULL_CALL: u16 = 5;
/// An upvalue's mask bits: a closure gave it a value, and two gave it
/// different ones.
const SEEN: u64 = 1;
const VARIES: u64 = 2;
/// Closures of a prototype `note_closure` learns from; later ones that
/// differ are found by a failed constant guard (`set_upvalue_varies`).
const LEARN_CLOSURES: u8 = 8;

/// A site word: bit 0 set, inline bits above it; clear, a [`ShapeRecord`].
const INLINE: u64 = 1;

/// A site's feedback bits (`jit::feedback`).
pub(crate) type Bits = u32;

/// A callee word that saw two functions: no `Value` has these bits. Bit 0
/// marks it and a `proto_slot` word, which the interpreter does not record
/// over (a function's payload is aligned).
pub(crate) const MEGAMORPHIC: u64 = 0xFFF8_0000_0000_0001;

/// A callee word's value for closures of `proto`: its address with bit 0 set.
pub(crate) fn proto_slot(proto: Gc<'_, Prototype<'_>>) -> u64 {
    Gc::as_ptr(proto) as u64 | 1
}

/// The prototype a callee word names (`proto_slot`).
pub(crate) fn slot_proto<'gc>(w: u64) -> Option<Gc<'gc, Prototype<'gc>>> {
    (w & 1 != 0 && w != MEGAMORPHIC).then(|| unsafe { Gc::from_ptr((w & !1) as *const _) })
}

/// The shapes compiled code's shape exits saw at a field site, the first
/// `len` of `shapes`; one past `MAX_SHAPES` makes the site megamorphic.
pub(crate) struct ShapeRecord<'gc> {
    bits: Cell<Bits>,
    len: Cell<u8>,
    shapes: [Cell<Option<Shape<'gc>>>; RECORD_SHAPES],
}

const RECORD_SHAPES: usize = crate::jit::MAX_SHAPES + 1;

/// See the module doc. The words and their layout are fixed when the
/// prototype is made; `kinds` and `slot_of_pc` never change.
pub struct FeedbackVector<'gc> {
    /// The header, then the slots. The header: per upvalue the first
    /// closure's value bits, per 32 upvalues a mask word (`SEEN`, `VARIES`
    /// per upvalue), then the invocation count, the word before slot 0.
    words: Box<[Cell<u64>]>,
    /// Header words.
    head: u32,
    upvalues: u16,
    /// Closures left to learn from: 0 once every by-value upvalue varies.
    learn: Cell<u8>,
    kinds: Box<[SlotKind]>,
    /// Per instruction, its slot's first word, or `NO_SLOT`. Field and call
    /// instructions also hold it in `d`, for the interpreter's fast paths.
    slot_of_pc: Box<[u16]>,
    /// Whether any slot may hold a reference.
    refs: bool,
    _gc: std::marker::PhantomData<Shape<'gc>>,
}

const _: () =
    assert!(size_of::<InlineCache<'static>>() == 32 && align_of::<InlineCache<'static>>() <= 8);

impl<'gc> FeedbackVector<'gc> {
    /// The vector of `code`, of a prototype with `upvalues` upvalues, whose
    /// field and call instructions get their slot in `d`. Sites past
    /// `MAX_WORDS` share the null slot and record nothing.
    pub(crate) fn new(code: &mut [Instruction], upvalues: usize) -> Self {
        let slot_kinds: Vec<Option<SlotKind>> = code.iter().map(|i| SlotKind::of(i.op())).collect();
        let total: usize = slot_kinds.iter().flatten().map(|k| k.words()).sum();
        let mut kinds: Vec<SlotKind> = Vec::with_capacity(total.min(MAX_WORDS));
        let push = |kinds: &mut Vec<SlotKind>, k: SlotKind| {
            kinds.push(k);
            kinds.extend(std::iter::repeat_n(SlotKind::Cont, k.words() - 1));
        };
        if total > MAX_WORDS {
            push(&mut kinds, SlotKind::Null);
        }
        let mut slot_of_pc = Vec::with_capacity(code.len());
        for (i, k) in code.iter_mut().zip(slot_kinds) {
            let Some(k) = k else {
                slot_of_pc.push(NO_SLOT);
                continue;
            };
            let w = kinds.len();
            let fast = matches!(k, SlotKind::Field | SlotKind::Call);
            if w + k.words() > MAX_WORDS {
                slot_of_pc.push(NO_SLOT);
                if fast {
                    i.set_d(if k == SlotKind::Call { NULL_CALL } else { 0 });
                }
                continue;
            }
            push(&mut kinds, k);
            slot_of_pc.push(w as u16);
            if fast {
                i.set_d(w as u16);
            }
        }
        let head = upvalues + upvalues.div_ceil(32) + 1;
        let v = FeedbackVector {
            words: (0..head + kinds.len()).map(|_| Cell::new(0)).collect(),
            head: head as u32,
            upvalues: upvalues as u16,
            learn: Cell::new(LEARN_CLOSURES),
            refs: kinds
                .iter()
                .any(|k| matches!(k, SlotKind::Field | SlotKind::Call)),
            kinds: kinds.into_boxed_slice(),
            slot_of_pc: slot_of_pc.into_boxed_slice(),
            _gc: std::marker::PhantomData,
        };
        let mut w = 0;
        while w < v.kinds.len() {
            let k = v.kinds[w];
            match k {
                SlotKind::Null => {
                    v.cache(w).set(InlineCache::Empty);
                    v.callee_at(w + NULL_CALL as usize).set(MEGAMORPHIC);
                }
                SlotKind::Field => v.cache(w).set(InlineCache::Empty),
                _ => {}
            }
            v.word(w).set(INLINE);
            w += k.words();
        }
        v
    }

    /// Slot 0's word, which closures cache; the header is below it. The
    /// interpreter counts calls in the word after a callee word, and
    /// invocations (`FUNC`) in the word before slot 0.
    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const Cell<u64> {
        // SAFETY: the header is `head` words.
        unsafe { self.words.as_ptr().add(self.head as usize) }
    }

    /// Slot words.
    pub(crate) fn len(&self) -> usize {
        self.words.len() - self.head as usize
    }

    #[inline(always)]
    fn word(&self, w: usize) -> &Cell<u64> {
        &self.words[self.head as usize + w]
    }

    /// The slot of the instruction at `pc`, with its kind.
    #[inline]
    pub(crate) fn slot(&self, pc: usize) -> Option<(usize, SlotKind)> {
        let w = *self.slot_of_pc.get(pc)?;
        (w != NO_SLOT).then(|| (w as usize, self.kinds[w as usize]))
    }

    /// The inline cache of the field slot at word `w`, the null slot's included.
    #[inline(always)]
    pub(crate) fn cache(&self, w: usize) -> &Cell<InlineCache<'gc>> {
        debug_assert!(matches!(self.kinds[w], SlotKind::Field | SlotKind::Null));
        // SAFETY: a field slot's words after its site word are an entry, which
        // `new` initialized as one; the assert above keeps its alignment.
        unsafe { &*self.as_ptr().add(w + 1).cast::<Cell<InlineCache<'gc>>>() }
    }

    /// Whether word `w` starts a field slot the interpreter may fill (not the
    /// null slot).
    #[inline]
    pub(crate) fn is_field(&self, w: usize) -> bool {
        self.kinds.get(w) == Some(&SlotKind::Field)
    }

    /// The callee word of the call at `pc`.
    #[inline]
    pub(crate) fn callee(&self, pc: usize) -> Option<&Cell<u64>> {
        match self.slot(pc)? {
            (w, SlotKind::Call) => Some(self.callee_at(w)),
            _ => None,
        }
    }

    /// The callee word of the call slot at word `w` (a call's `d`, the null
    /// slot's included); its count is the word after.
    #[inline(always)]
    pub(crate) fn callee_at(&self, w: usize) -> &Cell<u64> {
        self.word(w + 1)
    }

    /// Upvalue `k`'s mask word and the shift of its bits there.
    fn upvalue_mask(&self, k: usize) -> (&Cell<u64>, u32) {
        (
            &self.words[self.upvalues as usize + k / 32],
            2 * (k % 32) as u32,
        )
    }

    /// Learn from a new closure `f` of this vector's prototype, one of its
    /// first `LEARN_CLOSURES`, which of its by-value upvalues differ between
    /// closures: the first closure's value bits are kept, a later one that
    /// differs marks the upvalue `VARIES`. Bits only, untraced: a dead value
    /// whose address is reused reads as the same, which costs one failed
    /// constant guard.
    #[inline(always)]
    pub(crate) fn note_closure(&self, f: LuaFn<'gc>) {
        if self.learn.get() != 0 {
            self.learn_closure(f);
        }
    }

    #[inline(never)]
    fn learn_closure(&self, f: LuaFn<'gc>) {
        self.learn.set(self.learn.get() - 1);
        let mut undecided = false;
        for (k, (desc, slot)) in f.proto.upvalue_desc.iter().zip(f.upvalues()).enumerate() {
            if !desc.by_value {
                continue;
            }
            let (m, sh) = self.upvalue_mask(k);
            let bits = m.get() >> sh;
            if bits & VARIES != 0 {
                continue;
            }
            // SAFETY: `by_value` says the slot holds a value.
            let v = unsafe { slot.value }.to_raw();
            if bits & SEEN == 0 {
                self.words[k].set(v);
                m.set(m.get() | SEEN << sh);
                undecided = true;
            } else if self.words[k].get() != v {
                m.set(m.get() | VARIES << sh);
            } else {
                undecided = true;
            }
        }
        if !undecided {
            self.learn.set(0);
        }
    }

    /// Whether closures gave upvalue `k` different values (`note_closure`),
    /// or a region's constant guard on it failed (`set_upvalue_varies`).
    pub(crate) fn upvalue_varies(&self, k: usize) -> bool {
        let (m, sh) = self.upvalue_mask(k);
        m.get() >> sh & VARIES != 0
    }

    /// Mark upvalue `k` as varying; whether it was not already.
    pub(crate) fn set_upvalue_varies(&self, k: usize) -> bool {
        let (m, sh) = self.upvalue_mask(k);
        let old = m.get();
        m.set(old | VARIES << sh);
        old >> sh & VARIES == 0
    }

    /// The site word of the instruction at `pc`: its slot's first word.
    #[inline(always)]
    fn site(&self, pc: usize) -> Option<&Cell<u64>> {
        let w = *self.slot_of_pc.get(pc)?;
        (w != NO_SLOT).then(|| self.word(w as usize))
    }

    fn record(&self, w: u64) -> Option<&ShapeRecord<'gc>> {
        // SAFETY: a site word with bit 0 clear is a record this vector owns.
        (w & INLINE == 0).then(|| unsafe { &*(w as *const ShapeRecord<'gc>) })
    }

    /// The bits of the site at `pc`; none for one without a slot.
    #[inline]
    pub(crate) fn bits(&self, pc: usize) -> Bits {
        let Some(s) = self.site(pc) else { return 0 };
        let w = s.get();
        match self.record(w) {
            Some(r) => r.bits.get(),
            None => (w >> 1) as Bits,
        }
    }

    /// `add_bits` for the interpreter's slow paths, reaching the site word
    /// through `base`, a closure's cached `as_ptr`.
    ///
    /// # Safety
    /// `pc` is an instruction of this vector's prototype, `base` its `as_ptr`.
    #[inline(always)]
    pub(crate) unsafe fn add_bits_at(&self, base: *const Cell<u64>, pc: usize, bits: Bits) {
        debug_assert!(pc < self.slot_of_pc.len() && std::ptr::eq(base, self.as_ptr()));
        let w = unsafe { *self.slot_of_pc.get_unchecked(pc) };
        if w == NO_SLOT {
            return;
        }
        let s = unsafe { &*base.add(w as usize) };
        let v = s.get();
        match self.record(v) {
            Some(r) => r.bits.set(r.bits.get() | bits),
            None => {
                let n = v | (bits as u64) << 1;
                if n != v {
                    s.set(n);
                }
            }
        }
    }

    /// OR `bits` into the site at `pc`'s, stored only on a change: slow paths
    /// record on every run.
    #[inline]
    pub(crate) fn add_bits(&self, pc: usize, bits: Bits) {
        let Some(s) = self.site(pc) else { return };
        let w = s.get();
        match self.record(w) {
            Some(r) => r.bits.set(r.bits.get() | bits),
            None => {
                let n = w | (bits as u64) << 1;
                if n != w {
                    s.set(n);
                }
            }
        }
    }

    /// The shapes recorded at the field site at `pc`.
    pub(crate) fn shapes(&self, pc: usize) -> impl Iterator<Item = Shape<'gc>> + '_ {
        let r = self.site(pc).and_then(|s| self.record(s.get()));
        r.into_iter().flat_map(|r| {
            r.shapes[..r.len.get() as usize]
                .iter()
                .map(|s| s.get().unwrap())
        })
    }

    /// Record `shape` at the field site at `pc`, up to `MAX_SHAPES + 1`. The
    /// caller runs the prototype's barrier.
    pub(crate) fn add_shape(&self, pc: usize, shape: Shape<'gc>) {
        let Some((w, SlotKind::Field)) = self.slot(pc) else {
            return;
        };
        let s = self.word(w);
        let r = match self.record(s.get()) {
            Some(r) => r,
            None => {
                let r = Box::new(ShapeRecord {
                    bits: Cell::new((s.get() >> 1) as Bits),
                    len: Cell::new(0),
                    shapes: Default::default(),
                });
                let r = Box::into_raw(r);
                s.set(r as u64);
                // SAFETY: just made, owned by the vector from here on.
                unsafe { &*r }
            }
        };
        r.push(shape);
    }

    /// What exits compare to tell whether a recompile learns something: the
    /// site's bits and its shape count.
    pub(crate) fn seen(&self, pc: usize) -> u64 {
        let n = self
            .site(pc)
            .and_then(|s| self.record(s.get()))
            .map_or(0, |r| r.len.get());
        self.bits(pc) as u64 | (n as u64) << 32
    }

    /// Empty every reference to an object this collection found dead: a cache
    /// entry (and its site's quickened form, which implies the entry's kind),
    /// a callee word, a recorded shape.
    fn clear_dead(&self, fc: &Finalization<'gc>, code: &Code) {
        let dead = |s: Shape<'gc>| Gc::is_dead(fc, s.inner());
        for (pc, &w) in self.slot_of_pc.iter().enumerate() {
            if w == NO_SLOT {
                continue;
            }
            let w = w as usize;
            match self.kinds[w] {
                SlotKind::Field => {
                    let entry = self.cache(w).get();
                    let gone = match entry {
                        InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => {
                            dead(shape)
                        }
                        InlineCache::Transition { from, to, .. } => dead(from) || dead(to),
                        InlineCache::ProtoLoad {
                            recv,
                            holder,
                            holder_shape,
                            ..
                        } => dead(recv) || holder.is_dead(fc) || dead(holder_shape),
                        InlineCache::Empty => false,
                    };
                    if gone {
                        self.cache(w).set(InlineCache::Empty);
                        code.unquicken(pc);
                    }
                    let site = self.word(w).get();
                    if let Some(r) = self.record(site) {
                        r.retain(|s| !dead(s));
                    }
                }
                SlotKind::Call => {
                    let c = self.callee_at(w);
                    let v = c.get();
                    let gone = if v == 0 || v == MEGAMORPHIC {
                        false
                    } else if let Some(p) = slot_proto(v) {
                        Gc::is_dead(fc, p)
                    } else {
                        // SAFETY: a callee word that names no prototype holds a function's bits.
                        let f = unsafe { Value::from_raw(v) }.get_function();
                        Gc::is_dead(fc, f.expect("a callee word's function").inner())
                    };
                    if gone {
                        c.set(0);
                    }
                }
                _ => {}
            }
        }
    }
}

impl<'gc> ShapeRecord<'gc> {
    fn push(&self, shape: Shape<'gc>) {
        let n = self.len.get() as usize;
        let have = self.shapes[..n]
            .iter()
            .any(|s| Shape::ptr_eq(s.get().unwrap(), shape));
        if !have && n < RECORD_SHAPES {
            self.shapes[n].set(Some(shape));
            self.len.set(n as u8 + 1);
        }
    }

    fn retain(&self, keep: impl Fn(Shape<'gc>) -> bool) {
        let mut n = 0;
        for i in 0..self.len.get() as usize {
            let s = self.shapes[i].get().unwrap();
            if keep(s) {
                self.shapes[n].set(Some(s));
                n += 1;
            }
        }
        for s in &self.shapes[n..] {
            s.set(None);
        }
        self.len.set(n as u8);
    }
}

impl Drop for FeedbackVector<'_> {
    fn drop(&mut self) {
        for (w, k) in self.kinds.iter().enumerate() {
            if *k == SlotKind::Field
                && let s = self.word(w).get()
                && s & INLINE == 0
            {
                // SAFETY: made by `add_shape` with `Box::into_raw`, owned here.
                drop(unsafe { Box::from_raw(s as *mut ShapeRecord<'_>) });
            }
        }
    }
}

// SAFETY: holds no strong reference; the weak ones are cleared through the
// deferral before the sweep (`Prototype::clear_dead_feedback`). A write of a
// reference takes the prototype's barrier, so a vector holding a young object
// is traced, and deferred, in the minor collection that may free it.
unsafe impl<'gc> Collect<'gc> for FeedbackVector<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if self.refs {
            cc.defer(crate::env::DEFER_FEEDBACK);
        }
    }
}

impl<'gc> Prototype<'gc> {
    /// Clear the dead references of the deferred prototypes' feedback
    /// (`FeedbackVector::clear_dead`). Clearing adopts no pointer, so takes no
    /// barrier, which would re-gray the prototype past `clear_deferred`.
    pub(crate) fn clear_dead_feedback(fc: &Finalization<'gc>) {
        for p in fc.deferred(crate::env::DEFER_FEEDBACK) {
            // SAFETY: only `FeedbackVector::trace`, a field of a prototype,
            // defers to `DEFER_FEEDBACK`.
            let p: Gc<'gc, Prototype<'gc>> = unsafe { Gc::cast(p) };
            p.feedback.clear_dead(fc, &p.code);
        }
    }
}
