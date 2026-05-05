//! Globally-interned `LuaString` symbols. One `Symbols<'gc>` lives on
//! `State` for the lifetime of the runtime; access via
//! `Context::symbols()`. Holds the interned identity of every Lua
//! metamethod name plus any other ambient symbol the runtime needs.

use crate::dmm::{Collect, Mutation};
use crate::env::for_each_metamethod;
use crate::env::shape::MetamethodBits;
use crate::env::string::{Interner, LuaString};

/// Number of distinct metamethod names. Derived from the master list
/// in `crate::env::for_each_metamethod`.
macro_rules! emit_count {
    ($(($upper:ident, $bit:literal, $bytes:literal, $field:ident);)*) => {
        pub const METAMETHOD_COUNT: usize = ${count($field)};
    };
}
for_each_metamethod!(emit_count);

/// Pre-interned identities for ambient symbols used across the
/// runtime. Built once at `Lua::new` and shared (by Gc identity) with
/// every other use of the same byte sequence.
macro_rules! emit_struct {
    ($(($upper:ident, $bit:literal, $bytes:literal, $field:ident);)*) => {
        #[derive(Collect)]
        #[collect(internal, no_drop)]
        pub struct Symbols<'gc> {
            $(pub $field: LuaString<'gc>,)*
        }
    };
}
for_each_metamethod!(emit_struct);

macro_rules! emit_intern_all {
    ($(($upper:ident, $bit:literal, $bytes:literal, $field:ident);)*) => {
        impl<'gc> Symbols<'gc> {
            pub(crate) fn intern_all(mc: &Mutation<'gc>, interner: &Interner<'gc>) -> Self {
                Symbols {
                    $($field: interner.intern(mc, $bytes),)*
                }
            }

            /// All metamethod (name, bit) pairs. Used by
            /// `Table::ensure_mt_cache` to seed an `MtCache` by walking a
            /// metatable's slots once at first-adoption time.
            #[inline]
            pub fn metamethods(&self) -> [(LuaString<'gc>, MetamethodBits); METAMETHOD_COUNT] {
                [
                    $((self.$field, MetamethodBits::$upper),)*
                ]
            }
        }
    };
}
for_each_metamethod!(emit_intern_all);
