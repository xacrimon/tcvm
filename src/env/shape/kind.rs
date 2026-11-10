//! Slot kinds (ws2 §6.1): what every table of the shapes sharing a named
//! slot has held in it, so compiled loads need no type or nil checks and
//! the interpreter checks a store with one compare.

use core::cell::Cell;

use crate::dmm::{Collect, Gc, Lock, Mutation};
use crate::env::shape::Shape;
use crate::env::value::{TYPE_WORDS, Value};
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
use crate::jit::deps::{self, DepSlot};

/// The upper 32 bits of a value's encoding give each type a contiguous
/// range, in this order: floats, userdata, boxed integers, strings, tables,
/// functions, threads, then nil and the booleans (one value), then small
/// integers. A kind is an arc of that circle, `lo` and the `span` past it
/// wrapping, so small integers and floats are one arc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Kind {
    lo: u32,
    span: u32,
}

/// A type's range of upper words, by `Class` order: `TYPE_WORDS`' with nil
/// and the booleans one.
const CLASSES: [(u32, u32); 9] = {
    const fn up(w: u64) -> u32 {
        (w >> 32) as u32
    }
    let t = &TYPE_WORDS;
    let mut c = [(0, 0); 9];
    let mut k = 0;
    while k < 7 {
        c[k] = (up(t[k].0), up(t[k].1));
        k += 1;
    }
    c[7] = (up(t[7].0), up(t[9].1));
    c[8] = (up(t[10].0), up(t[10].1));
    c
};

/// The types a kind tells apart, bits of [`Kind::classes`].
pub mod class {
    pub const FLOAT: u16 = 1;
    pub const UDATA: u16 = 2;
    pub const BIGINT: u16 = 4;
    pub const STR: u16 = 8;
    pub const TAB: u16 = 16;
    pub const FUN: u16 = 32;
    pub const THR: u16 = 64;
    /// Nil and the booleans, which share their upper word.
    pub const NIL_BOOL: u16 = 128;
    pub const SMALL: u16 = 256;
}

/// Whether `v` is a table of shape `s`.
#[inline(always)]
pub fn is_table_of<'gc>(s: Option<Shape<'gc>>, v: Value<'gc>) -> bool {
    match (v.get_table(), s) {
        (Some(t), Some(s)) => Shape::ptr_eq(t.shape_now(), s),
        _ => false,
    }
}

/// Widenings after which a slot takes anything, bounding the recompiles a
/// slot's changing kind causes (V8's lattice is three high).
const MAX_WIDENINGS: u8 = 3;

impl Kind {
    /// Before any store: in the gap between threads and nil, which no value
    /// reaches.
    pub const NONE: Kind = Kind {
        lo: (TYPE_WORDS[6].1 >> 32) as u32 + 1,
        span: 0,
    };
    pub const ANY: Kind = Kind {
        lo: 0,
        span: u32::MAX,
    };

    #[inline(always)]
    pub fn fits(self, v: Value<'_>) -> bool {
        ((v.to_raw() >> 32) as u32).wrapping_sub(self.lo) <= self.span
    }

    /// The smallest arc holding this one and `v`'s type.
    pub fn widened(self, v: Value<'_>) -> Kind {
        if self.fits(v) {
            return self;
        }
        let hi = (v.to_raw() >> 32) as u32;
        let &(clo, chi) = CLASSES
            .iter()
            .find(|&&(lo, hi_)| (lo..=hi_).contains(&hi))
            .expect("a value's upper word is in a class");
        if self == Kind::NONE {
            return Kind {
                lo: clo,
                span: chi - clo,
            };
        }
        // Arcs hold whole classes, so `v`'s lies past this one's end and
        // before its start: grow the end forward or the start back.
        let end = self.lo.wrapping_add(self.span);
        let forward = chi.wrapping_sub(self.lo);
        let back = end.wrapping_sub(clo);
        if forward <= back {
            Kind {
                lo: self.lo,
                span: forward,
            }
        } else {
            Kind {
                lo: clo,
                span: back,
            }
        }
    }

    /// Whether tables fit: arcs hold whole classes, so whether the first
    /// table word does.
    #[inline(always)]
    pub fn has_tables(self) -> bool {
        CLASSES[4].0.wrapping_sub(self.lo) <= self.span
    }

    /// This arc without tables: where they lie inside it, the part after
    /// them, which holds nil.
    #[inline(always)]
    pub fn without_tables(self) -> Kind {
        let (tlo, thi) = CLASSES[4];
        if !self.has_tables() {
            return self;
        }
        // Arcs hold whole classes, so tables start it, end it or lie inside.
        let end = self.lo.wrapping_add(self.span);
        match (self.lo == tlo, end == thi) {
            (true, true) => Kind::NONE,
            (false, true) => Kind {
                lo: self.lo,
                span: (tlo - 1).wrapping_sub(self.lo),
            },
            _ => Kind {
                lo: thi + 1,
                span: end.wrapping_sub(thi + 1),
            },
        }
    }

    /// The types in this arc, as [`class`] bits.
    pub fn classes(self) -> u16 {
        let mut out = 0;
        for (i, &(lo, hi)) in CLASSES.iter().enumerate() {
            let in_arc = |w: u32| w.wrapping_sub(self.lo) <= self.span;
            // An arc meets a class iff it holds one of its ends or starts in it.
            if in_arc(lo) || in_arc(hi) || (lo..=hi).contains(&self.lo) {
                out |= 1 << i;
            }
        }
        out
    }

    #[inline(always)]
    pub(crate) fn to_word(self) -> u64 {
        (self.lo as u64) << 32 | self.span as u64
    }

    #[inline(always)]
    fn from_word(w: u64) -> Self {
        Kind {
            lo: (w >> 32) as u32,
            span: w as u32,
        }
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct FieldData<'gc> {
    #[collect(require_static)]
    kind: Cell<u64>,
    #[collect(require_static)]
    widenings: Cell<u8>,
    /// The shape every table stored here had when stored, while that was
    /// one shape and stable then (ws2 §6.1's `table(S)`); `None` once the
    /// arc holds tables of several (or before the first table).
    shape: Lock<Option<Shape<'gc>>>,
    /// The regions assuming the kind stays.
    #[cfg(all(feature = "jit", target_arch = "aarch64"))]
    deps: DepSlot<'gc>,
}

/// A named slot of the shapes sharing it (V8's field owner): the slot of
/// every shape whose keys hold it from the same transition, its kind widened
/// by every store, so it is true of all their tables.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Field<'gc>(Gc<'gc, FieldData<'gc>>);

impl<'gc> Field<'gc> {
    pub(crate) fn new(mc: &Mutation<'gc>) -> Self {
        Field(Gc::new(
            mc,
            FieldData {
                kind: Cell::new(Kind::NONE.to_word()),
                widenings: Cell::new(0),
                shape: Lock::new(None),
                #[cfg(all(feature = "jit", target_arch = "aarch64"))]
                deps: Lock::new(None),
            },
        ))
    }

    #[inline(always)]
    pub fn kind(self) -> Kind {
        Kind::from_word(self.0.kind.get())
    }

    /// The shape of every table stored here, while there was one.
    #[inline(always)]
    pub fn shape(self) -> Option<Shape<'gc>> {
        self.0.shape.get()
    }

    /// Whether storing `v` keeps what the slot says.
    #[inline(always)]
    pub fn fits(self, v: Value<'gc>) -> bool {
        self.kind().fits(v) && self.shape_fits(v)
    }

    /// Whether `v` is no table or one of the tracked shape, if any.
    #[inline(always)]
    pub fn shape_fits(self, v: Value<'gc>) -> bool {
        match (self.shape(), v.get_table()) {
            (Some(s), Some(t)) => Shape::ptr_eq(t.shape_now(), s),
            _ => true,
        }
    }

    /// Whether `v` is a table of the shape tracked, which a cache entry's
    /// kind leaves out (`store_kind`): the check a store of a table takes
    /// out of line.
    #[inline(always)]
    pub fn tracks(self, v: Value<'gc>) -> bool {
        is_table_of(self.shape(), v)
    }

    /// The kind a cache entry checks stores against: without tables while
    /// a shape is tracked, so a table takes the check of its shape.
    #[inline(always)]
    pub fn store_kind(self) -> Kind {
        match self.shape() {
            Some(_) => self.kind().without_tables(),
            None => self.kind(),
        }
    }

    /// Note that `v` is stored to this slot.
    #[inline(always)]
    pub fn admit(self, mc: &Mutation<'gc>, v: Value<'gc>) {
        if !self.fits(v) {
            self.widen(mc, v);
        }
    }

    /// Takes `mc` for the dependents only, so a collector's finalization may
    /// call it.
    #[cold]
    #[inline(never)]
    fn widen(self, mc: &Mutation<'gc>, v: Value<'gc>) {
        let old = self.kind();
        let mut new = old.widened(v);
        let mut shape = self.shape();
        if let Some(t) = v.get_table() {
            // The first table names the shape, if it is one code may rely on.
            shape = match old.has_tables() {
                false => Some(t.shape_now()).filter(|s| s.is_stable() && !s.is_dict()),
                true => None,
            };
        }
        if old != Kind::NONE {
            let n = self.0.widenings.get() + 1;
            self.0.widenings.set(n);
            if n >= MAX_WIDENINGS {
                new = Kind::ANY;
                shape = None;
            }
        }
        self.0.kind.set(new.to_word());
        let same = match (shape, self.shape()) {
            (Some(a), Some(b)) => Shape::ptr_eq(a, b),
            (a, b) => a.is_none() && b.is_none(),
        };
        if !same {
            // A nil (a weak clear, from finalization) never gets here.
            mc.backward_barrier(Gc::erase(self.0), None);
            unsafe { self.0.shape.as_cell() }.set(shape);
        }
        #[cfg(all(feature = "jit", target_arch = "aarch64"))]
        deps::invalidate(mc, &self.0.deps, u64::MAX);
    }

    /// Note that `region` assumes the kind stays.
    #[cfg(all(feature = "jit", target_arch = "aarch64"))]
    pub(crate) fn depend(
        self,
        mc: &Mutation<'gc>,
        region: Gc<'gc, crate::jit::region::Region<'gc>>,
    ) {
        deps::add(mc, self.0, &self.0.deps, region, 1);
    }

    #[cfg(all(feature = "jit", target_arch = "aarch64"))]
    pub(crate) fn as_word(self) -> u64 {
        Gc::as_ptr(self.0) as u64
    }

    pub fn ptr_eq(a: Self, b: Self) -> bool {
        Gc::ptr_eq(a.0, b.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fits(k: Kind, v: Value<'_>) -> bool {
        k.fits(v)
    }

    #[test]
    fn arcs_hold_what_was_stored() {
        let small = Kind::NONE.widened(Value::small(3));
        assert_eq!(small.classes(), class::SMALL);
        assert!(fits(small, Value::small(-7)) && !fits(small, Value::float(1.5)));
        let num = small.widened(Value::float(1.5));
        assert_eq!(num.classes(), class::SMALL | class::FLOAT);
        assert!(fits(num, Value::float(f64::NEG_INFINITY)) && fits(num, Value::float(f64::NAN)));
        assert!(!fits(num, Value::nil()));
        let nilable = num.widened(Value::nil());
        assert_eq!(
            nilable.classes(),
            class::SMALL | class::FLOAT | class::NIL_BOOL
        );
        assert!(fits(nilable, Value::boolean(true)));
        assert!(!Kind::NONE.fits(Value::nil()) && !Kind::NONE.fits(Value::small(0)));
        assert!(Kind::ANY.fits(Value::nil()) && Kind::ANY.classes() == 0x1ff);
    }

    #[test]
    fn store_kinds_leave_tables_out() {
        let nil = Kind::NONE.widened(Value::nil());
        let tab_nil = Kind {
            lo: CLASSES[4].0,
            span: CLASSES[7].1 - CLASSES[4].0,
        };
        assert_eq!(
            tab_nil.classes(),
            class::TAB | class::FUN | class::THR | class::NIL_BOOL
        );
        let rest = tab_nil.without_tables();
        assert_eq!(rest.classes(), class::FUN | class::THR | class::NIL_BOOL);
        assert!(rest.fits(Value::nil()) && !rest.has_tables());
        let tab = Kind {
            lo: CLASSES[4].0,
            span: CLASSES[4].1 - CLASSES[4].0,
        };
        assert_eq!(tab.without_tables(), Kind::NONE);
        assert_eq!(nil.without_tables(), nil);
        assert!(!Kind::NONE.has_tables() && Kind::ANY.has_tables());
        let str_fun = Kind {
            lo: CLASSES[3].0,
            span: CLASSES[5].1 - CLASSES[3].0,
        };
        assert_eq!(str_fun.without_tables().classes(), class::FUN);
    }
}
