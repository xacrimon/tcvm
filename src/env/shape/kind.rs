//! Slot kinds (ws2 §6.1): what every table of the shapes sharing a named
//! slot has held in it, so compiled loads need no type or nil checks and
//! the interpreter checks a store with one compare.

use core::cell::Cell;

use crate::dmm::{Collect, Gc, Mutation};
use crate::env::value::Value;
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

/// A type's range of upper words, by `Class` order.
const CLASSES: [(u32, u32); 9] = [
    (0, 0xFFF8_FFFF),
    (0xFFF9_0000, 0xFFF9_FFFF),
    (0xFFFA_0000, 0xFFFA_FFFF),
    (0xFFFB_0000, 0xFFFB_FFFF),
    (0xFFFC_0000, 0xFFFC_FFFF),
    (0xFFFD_0000, 0xFFFD_FFFF),
    (0xFFFE_0000, 0xFFFE_FFFF),
    (0xFFFF_FFFE, 0xFFFF_FFFE),
    (0xFFFF_FFFF, 0xFFFF_FFFF),
];

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

/// Widenings after which a slot takes anything, bounding the recompiles a
/// slot's changing kind causes (V8's lattice is three high).
const MAX_WIDENINGS: u8 = 3;

impl Kind {
    /// Before any store: in the gap between threads and nil, which no value
    /// reaches.
    pub const NONE: Kind = Kind {
        lo: 0xFFFF_0000,
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
    /// The regions assuming the kind stays.
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
                deps: crate::dmm::Lock::new(None),
            },
        ))
    }

    #[inline(always)]
    pub fn kind(self) -> Kind {
        Kind::from_word(self.0.kind.get())
    }

    /// Note that `v` is stored to this slot.
    #[inline(always)]
    pub fn admit(self, mc: &Mutation<'gc>, v: Value<'gc>) {
        if !self.kind().fits(v) {
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
        if old != Kind::NONE {
            let n = self.0.widenings.get() + 1;
            self.0.widenings.set(n);
            if n >= MAX_WIDENINGS {
                new = Kind::ANY;
            }
        }
        self.0.kind.set(new.to_word());
        deps::invalidate(mc, &self.0.deps, u64::MAX);
    }

    /// Note that `region` assumes the kind stays.
    pub(crate) fn depend(
        self,
        mc: &Mutation<'gc>,
        region: Gc<'gc, crate::jit::region::Region<'gc>>,
    ) {
        deps::add(mc, self.0, &self.0.deps, region, 1);
    }

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
}
