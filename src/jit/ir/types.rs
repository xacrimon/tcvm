//! Representations, type sets and facts.

/// How a value is held in compiled code.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Rep {
    /// No value.
    None,
    /// The boxed 64-bit `Value` word.
    Val,
    /// A small integer in a register's low 32 bits; the upper half is
    /// undefined (an unboxed value shares the boxed value's register).
    I32,
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
            Rep::None => "none",
            Rep::Val => "val",
            Rep::I32 => "i32",
            Rep::I64 => "i64",
            Rep::F64 => "f64",
            Rep::B1 => "b1",
            Rep::Ptr => "ptr",
        }
    }

    /// A 64-bit integer register's worth: integer ops take any of these as a
    /// word.
    pub(crate) fn word64(self) -> bool {
        matches!(self, Rep::I64 | Rep::Val | Rep::Ptr)
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
    /// Bits of an `aux` a type set takes; an exit tag sits above them.
    pub(crate) const BITS: u32 = 11;

    /// The set of a constant value.
    pub(crate) fn of(v: crate::env::value::Value<'_>) -> TypeSet {
        if v.is_nil() {
            TypeSet::NIL
        } else if let Some(b) = v.get_boolean() {
            if b { TypeSet::TRUE } else { TypeSet::FALSE }
        } else if v.get_small().is_some() {
            TypeSet::SMALL
        } else if v.get_integer().is_some() {
            TypeSet::BIGINT
        } else if v.is_float() {
            TypeSet::FLOAT
        } else if v.get_string().is_some() {
            TypeSet::STR
        } else if v.get_table().is_some() {
            TypeSet::TAB
        } else if v.get_function().is_some() {
            TypeSet::FUN
        } else if v.get_thread().is_some() {
            TypeSet::THR
        } else {
            TypeSet::UDATA
        }
    }

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

    /// The representation a value known to be in the set unboxes to: `None`
    /// where the set has no single one.
    pub(crate) fn unboxed(self) -> Rep {
        if self.is_empty() {
            Rep::None
        } else if self == TypeSet::SMALL {
            Rep::I32
        } else if self == TypeSet::FLOAT {
            Rep::F64
        } else if TypeSet::HEAP.contains(self) && self.bits().is_power_of_two() {
            Rep::Ptr
        } else {
            Rep::None
        }
    }

    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn print(self, out: &mut String) {
        const NAMES: [&str; 11] = [
            "nil", "false", "true", "small", "big", "float", "str", "tab", "fun", "thr", "udata",
        ];
        if self == TypeSet::ANY {
            return out.push_str("any");
        }
        if self.is_empty() {
            return out.push_str("none");
        }
        let mut first = true;
        for (k, name) in NAMES.iter().enumerate() {
            if self.bits() >> k & 1 != 0 {
                if !first {
                    out.push('|');
                }
                out.push_str(name);
                first = false;
            }
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub(crate) struct FactFlags: u8 {
        /// A loop parameter's type and representation are not resolved yet,
        /// or a generic op's result hangs on one that is not.
        const PENDING = 1;
        /// A `B1` known true, or known false, at this point.
        const TRUE = 2;
        const FALSE = 4;
    }
}

/// What is known of a value at a point: its definition's fact refined by
/// what dominates the point. Facts decide what code to emit, never whether
/// an op may move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Fact {
    /// The Lua types the value may have; for a `Ptr`, those of its object.
    pub(crate) set: TypeSet,
    pub(crate) flags: FactFlags,
}

impl Fact {
    pub(crate) const ANY: Fact = Fact::of(TypeSet::ANY);
    pub(crate) const NONE: Fact = Fact::of(TypeSet::empty());

    pub(crate) const fn of(set: TypeSet) -> Fact {
        Fact {
            set,
            flags: FactFlags::empty(),
        }
    }

    /// The fact every value of a representation has.
    pub(crate) const fn of_rep(rep: Rep) -> Fact {
        Fact::of(match rep {
            Rep::Val => TypeSet::ANY,
            Rep::I32 => TypeSet::SMALL,
            Rep::I64 => TypeSet::INT,
            Rep::F64 => TypeSet::FLOAT,
            Rep::B1 => TypeSet::BOOL,
            Rep::None | Rep::Ptr => TypeSet::empty(),
        })
    }

    pub(crate) fn pending(self) -> bool {
        self.flags.contains(FactFlags::PENDING)
    }

    /// What holds when either fact does.
    pub(crate) fn join(self, o: Fact) -> Fact {
        Fact {
            set: self.set | o.set,
            flags: (self.flags & o.flags) | ((self.flags | o.flags) & FactFlags::PENDING),
        }
    }

    /// What holds when both facts do.
    pub(crate) fn meet(self, o: Fact) -> Fact {
        Fact {
            set: self.set & o.set,
            flags: self.flags | o.flags,
        }
    }

    pub(crate) fn bits(self) -> u32 {
        self.set.bits() as u32 | (self.flags.bits() as u32) << 16
    }

    pub(crate) fn from_bits(w: u32) -> Fact {
        Fact {
            set: TypeSet::from_bits_retain(w as u16),
            flags: FactFlags::from_bits_retain((w >> 16) as u8),
        }
    }
}
