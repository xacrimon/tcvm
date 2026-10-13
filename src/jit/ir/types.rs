//! Value types: a representation, the set of Lua types the value may have,
//! and an optional refinement to an exact constant or prototype.

use std::fmt;

/// How a value is held in compiled code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Rep {
    /// The boxed 64-bit `Value` word.
    Val,
    /// A small integer, sign-extended in an x register.
    I32,
    /// A full integer in an x register.
    I64,
    F64,
    /// A boolean in a w register (0 or 1).
    B1,
    /// A raw pointer (a table, a shape, a class).
    Ptr,
}

impl Rep {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Rep::Val => "val",
            Rep::I32 => "i32",
            Rep::I64 => "i64",
            Rep::F64 => "f64",
            Rep::B1 => "b1",
            Rep::Ptr => "ptr",
        }
    }
}

bitflags::bitflags! {
    /// The Lua types a value may have.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct TypeSet: u16 {
        const NIL = 1;
        const FALSE = 2;
        const TRUE = 4;
        const SMALL = 8;
        const BIGINT = 16;
        const FLOAT = 32;
        const STR = 64;
        const TAB = 128;
        const FUN = 256;
        const THR = 512;
        const UDATA = 1024;

        const BOOL = Self::FALSE.bits() | Self::TRUE.bits();
        const INT = Self::SMALL.bits() | Self::BIGINT.bits();
        const NUM = Self::INT.bits() | Self::FLOAT.bits();
        const FALSY = Self::NIL.bits() | Self::FALSE.bits();
        const HEAP = Self::STR.bits() | Self::TAB.bits() | Self::FUN.bits()
            | Self::THR.bits() | Self::UDATA.bits() | Self::BIGINT.bits();
        const ANY = 0x7ff;
    }
}

/// A fact about a value beyond its type set, valid for as long as the value
/// exists. Shapes are not facts of a value (a store changes them), so none is
/// here.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Refine {
    None,
    /// The exact value, the pool index of its boxed word.
    Const(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Ty {
    pub(crate) rep: Rep,
    pub(crate) set: TypeSet,
    pub(crate) refine: Refine,
}

impl Ty {
    pub(crate) const ANY: Ty = Ty::val(TypeSet::ANY);
    pub(crate) const I32: Ty = Ty {
        rep: Rep::I32,
        set: TypeSet::SMALL,
        refine: Refine::None,
    };
    pub(crate) const I64: Ty = Ty {
        rep: Rep::I64,
        set: TypeSet::INT,
        refine: Refine::None,
    };
    pub(crate) const F64: Ty = Ty {
        rep: Rep::F64,
        set: TypeSet::FLOAT,
        refine: Refine::None,
    };
    pub(crate) const B1: Ty = Ty {
        rep: Rep::B1,
        set: TypeSet::BOOL,
        refine: Refine::None,
    };
    pub(crate) const PTR: Ty = Ty {
        rep: Rep::Ptr,
        set: TypeSet::empty(),
        refine: Refine::None,
    };

    pub(crate) const fn val(set: TypeSet) -> Ty {
        Ty {
            rep: Rep::Val,
            set,
            refine: Refine::None,
        }
    }

    pub(crate) const fn of_rep(rep: Rep) -> Ty {
        match rep {
            Rep::Val => Ty::ANY,
            Rep::I32 => Ty::I32,
            Rep::I64 => Ty::I64,
            Rep::F64 => Ty::F64,
            Rep::B1 => Ty::B1,
            Rep::Ptr => Ty::PTR,
        }
    }

    /// The type a value of this type has once boxed.
    pub(crate) fn boxed(self) -> Ty {
        Ty {
            rep: Rep::Val,
            set: self.set,
            refine: self.refine,
        }
    }

    pub(crate) fn join(self, other: Ty) -> Ty {
        Ty {
            rep: if self.rep == other.rep {
                self.rep
            } else {
                Rep::Val
            },
            set: self.set | other.set,
            refine: if self.refine == other.refine {
                self.refine
            } else {
                Refine::None
            },
        }
    }

    /// Whether every value of this type is in `set`.
    pub(crate) fn within(self, set: TypeSet) -> bool {
        set.contains(self.set)
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rep.name())?;
        if self.rep == Rep::Val || self.rep == Rep::I64 {
            if self.set == TypeSet::ANY {
                write!(f, ":any")?;
            } else if self.set.is_empty() {
                write!(f, ":none")?;
            } else {
                let names = [
                    (TypeSet::NIL, "nil"),
                    (TypeSet::FALSE, "false"),
                    (TypeSet::TRUE, "true"),
                    (TypeSet::SMALL, "small"),
                    (TypeSet::BIGINT, "big"),
                    (TypeSet::FLOAT, "float"),
                    (TypeSet::STR, "str"),
                    (TypeSet::TAB, "tab"),
                    (TypeSet::FUN, "fun"),
                    (TypeSet::THR, "thr"),
                    (TypeSet::UDATA, "udata"),
                ];
                let mut first = true;
                for (bit, name) in names {
                    if self.set.contains(bit) {
                        write!(f, "{}{name}", if first { ":" } else { "|" })?;
                        first = false;
                    }
                }
            }
        }
        if let Refine::Const(p) = self.refine {
            write!(f, "=k{p}")?;
        }
        Ok(())
    }
}
