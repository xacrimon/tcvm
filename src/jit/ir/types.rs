//! Value types: a representation, the set of Lua types the value may have,
//! and an optional refinement to an exact constant or prototype.

use std::fmt;

/// How a value is held in compiled code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Rep {
    /// The boxed 64-bit `Value` word.
    Val,
    /// A small integer in a register's low 32 bits; the upper half is
    /// undefined (an unboxed value shares the boxed value's register).
    I32,
    /// A full integer in a 64-bit register.
    I64,
    F64,
    /// A boolean in a register's low 32 bits (0 or 1).
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

impl TypeSet {
    /// The types of `env::value::TYPE_WORDS`, by index.
    pub(crate) const BY_WORDS: [TypeSet; 11] = [
        TypeSet::FLOAT,
        TypeSet::UDATA,
        TypeSet::BIGINT,
        TypeSet::STR,
        TypeSet::TAB,
        TypeSet::FUN,
        TypeSet::THR,
        TypeSet::NIL,
        TypeSet::FALSE,
        TypeSet::TRUE,
        TypeSet::SMALL,
    ];

    /// The types of a slot kind's classes (`Kind::classes`): those of
    /// `BY_WORDS` with nil and the booleans, which share an upper word, one.
    pub(crate) fn of_classes(classes: u16) -> TypeSet {
        let mut set = TypeSet::empty();
        for (k, &t) in Self::BY_WORDS.iter().enumerate() {
            let class = match k {
                0..=6 => k,
                7..=9 => 7,
                _ => 8,
            };
            if classes >> class & 1 != 0 {
                set |= t;
            }
        }
        set
    }
}

/// A fact about a value beyond its type set, valid for as long as the value
/// exists. Shapes are not facts of a value (a store changes them), except a
/// stable one's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Refine {
    None,
    /// The exact value, the pool index of its boxed word.
    Const(u32),
    /// If a table, one in the stable shape of this index (`Func::shapes`),
    /// wherever no table may have moved since a lazy exit
    /// (`opt::stable`).
    Shape(u32),
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
        match self.refine {
            Refine::Const(p) => write!(f, "=k{p}")?,
            Refine::Shape(s) => write!(f, "@s{s}")?,
            Refine::None => {}
        }
        Ok(())
    }
}
