//! The type lattice.
//!
//! A value's type is three orthogonal facts: how it is physically held
//! (`Rep`), which Lua types it might be (`TypeSet`), and any single sharper
//! fact we've proven about it (`Refine`).
//!
//! Guards *produce* refined values rather than mutating a value's type in
//! place, so type facts ride the use-def graph and a block's entry context is
//! just the types of its parameters — no flow-sensitive side table to keep in
//! sync.

use bitflags::bitflags;

use crate::env::value::{Value, ValueKind};
use crate::jit::ir::pool::{ConstRef, ProtoRef, ShapeRef};

bitflags! {
    /// Which Lua types a value may be.
    ///
    /// Booleans are split into `FALSE`/`TRUE` rather than a single `BOOL` so
    /// that truthiness is decidable from the set alone: `TEST`/`TESTSET` fold
    /// whenever the set is disjoint from `FALSY` or from `TRUTHY`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct TypeSet: u16 {
        const NIL   = 1 << 0;
        const FALSE = 1 << 1;
        const TRUE  = 1 << 2;
        const INT   = 1 << 3;
        const FLOAT = 1 << 4;
        const STR   = 1 << 5;
        const TAB   = 1 << 6;
        const FUN   = 1 << 7;
        const THR   = 1 << 8;
        const UDATA = 1 << 9;
    }
}

impl TypeSet {
    pub const BOOL: Self = Self::FALSE.union(Self::TRUE);
    pub const NUM: Self = Self::INT.union(Self::FLOAT);
    pub const FALSY: Self = Self::NIL.union(Self::FALSE);
    pub const ANY: Self = Self::all();
    /// Types whose payload is a `Gc` pointer — the ones the collector cares
    /// about, and the ones `unbox.ptr` accepts.
    pub const HEAP: Self = Self::STR
        .union(Self::TAB)
        .union(Self::FUN)
        .union(Self::THR)
        .union(Self::UDATA);

    pub const TRUTHY: Self = Self::ANY.difference(Self::FALSY);

    /// The exact set for a concrete value. Note this splits booleans, which
    /// `ValueKind` alone cannot.
    pub fn of_value(v: Value<'_>) -> Self {
        match v.kind() {
            ValueKind::Nil => Self::NIL,
            ValueKind::Boolean => {
                if v.get_boolean() == Some(true) {
                    Self::TRUE
                } else {
                    Self::FALSE
                }
            }
            ValueKind::Integer => Self::INT,
            ValueKind::Float => Self::FLOAT,
            ValueKind::String => Self::STR,
            ValueKind::Table => Self::TAB,
            ValueKind::Function => Self::FUN,
            ValueKind::Thread => Self::THR,
            ValueKind::Userdata => Self::UDATA,
        }
    }

    /// True if every value in this set is falsy (`nil` or `false`).
    pub fn is_falsy(self) -> bool {
        !self.is_empty() && self.difference(Self::FALSY).is_empty()
    }

    /// True if no value in this set is falsy.
    pub fn is_truthy(self) -> bool {
        !self.is_empty() && self.intersection(Self::FALSY).is_empty()
    }

    /// True if this set names exactly one Lua type, i.e. no type check is
    /// needed to act on it.
    pub fn is_monomorphic(self) -> bool {
        self.bits().count_ones() == 1
    }
}

/// How a value is physically held in compiled code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Rep {
    /// Full 16-byte boxed `Value` (tag + payload).
    Val,
    I64,
    F64,
    /// A condition, not a Lua value. Only branches and guards consume these.
    B1,
    /// Raw untagged pointer — a `Gc` target, or an interior pointer derived
    /// from one. Sound to hold without rooting *only* because the collector
    /// does not move objects; see the rooting rule in `op::Effects`.
    Ptr,
}

/// At most one sharper fact about a value. Each implies a `TypeSet`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Refine {
    #[default]
    None,
    Const(ConstRef),
    /// A concrete, non-dict shape. Dict sentinels are shared across tables and
    /// so prove nothing about layout — the frontend must never build one.
    Shape(ShapeRef),
    Proto(ProtoRef),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ty {
    pub rep: Rep,
    pub set: TypeSet,
    pub refine: Refine,
}

impl Ty {
    pub const fn new(rep: Rep, set: TypeSet) -> Self {
        Ty {
            rep,
            set,
            refine: Refine::None,
        }
    }

    pub const ANY: Self = Ty::new(Rep::Val, TypeSet::ANY);
    pub const I64: Self = Ty::new(Rep::I64, TypeSet::INT);
    pub const F64: Self = Ty::new(Rep::F64, TypeSet::FLOAT);
    pub const B1: Self = Ty::new(Rep::B1, TypeSet::BOOL);
    pub const PTR: Self = Ty::new(Rep::Ptr, TypeSet::HEAP);

    pub fn boxed(set: TypeSet) -> Self {
        Ty::new(Rep::Val, set)
    }

    pub fn with_shape(shape: ShapeRef) -> Self {
        Ty {
            rep: Rep::Val,
            set: TypeSet::TAB,
            refine: Refine::Shape(shape),
        }
    }

    pub fn shape(self) -> Option<ShapeRef> {
        match self.refine {
            Refine::Shape(s) => Some(s),
            _ => None,
        }
    }

    /// Narrow to `set`, dropping any refinement the narrowing invalidates.
    /// This is what a guard produces.
    pub fn refined_to(self, set: TypeSet) -> Self {
        let set = self.set & set;
        let refine = if self.refine_implied_set().intersects(set) {
            self.refine
        } else {
            Refine::None
        };
        Ty {
            rep: self.rep,
            set,
            refine,
        }
    }

    fn refine_implied_set(self) -> TypeSet {
        match self.refine {
            Refine::None => TypeSet::ANY,
            Refine::Const(_) => TypeSet::ANY,
            Refine::Shape(_) => TypeSet::TAB,
            Refine::Proto(_) => TypeSet::FUN,
        }
    }

    /// Least upper bound, for merging two edges into a block parameter.
    ///
    /// Representations must agree; when they don't, the merge is only
    /// expressible boxed, and the caller is responsible for inserting the
    /// `box.*` on the disagreeing edge. Returning `Rep::Val` here rather than
    /// failing keeps the frontend's merge logic in one place.
    pub fn join(self, other: Self) -> Self {
        let rep = if self.rep == other.rep {
            self.rep
        } else {
            Rep::Val
        };
        let refine = if self.refine == other.refine {
            self.refine
        } else {
            Refine::None
        };
        Ty {
            rep,
            set: self.set | other.set,
            refine,
        }
    }

    /// The type as it appears in a block's *entry context* — i.e. the
    /// versioning key.
    ///
    /// Constant refinements are dropped. Keying versions on a literal splits a
    /// block for a fact that ordinary constant folding recovers anyway (a merge
    /// of two distinct literals is not a constant, so the only thing gained is
    /// folding a first-iteration compare), while constants are far more numerous
    /// than types — so the splits burn the version cap, and hitting the cap
    /// drops the block to the fully generic version, costing the *type*
    /// specialization too. A bad trade.
    ///
    /// Shape and prototype refinements are kept. Those were *paid for by a
    /// guard*, and carrying them across the back-edge is exactly how the loop
    /// body avoids re-guarding.
    pub fn for_context(self) -> Self {
        match self.refine {
            Refine::Const(_) => Ty {
                refine: Refine::None,
                ..self
            },
            _ => self,
        }
    }

    /// Drop everything but the representation. Used when a block accumulates
    /// too many versions and we give up on specializing it.
    pub fn widen(self) -> Self {
        match self.rep {
            Rep::Val => Ty::ANY,
            _ => Ty::new(self.rep, self.set),
        }
    }
}

/// A block's entry type context: the types of its parameters, in order.
///
/// This *is* the versioning key — two edges reaching the same bytecode pc with
/// an equal context land on the same compiled block.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct TypeContext(pub Vec<Ty>);

impl TypeContext {
    pub fn widen(&self) -> TypeContext {
        TypeContext(self.0.iter().map(|t| t.widen()).collect())
    }
}
