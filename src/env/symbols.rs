//! Globally-interned `LuaString` symbols. One `Symbols<'gc>` lives on
//! `State` for the lifetime of the runtime; access via
//! `Context::symbols()`. Holds the interned identity of every Lua
//! metamethod name plus any other ambient symbol the runtime needs.

use crate::dmm::{Collect, Mutation};
use crate::env::shape::MetamethodBits;
use crate::env::string::{Interner, LuaString};

/// Pre-interned identities for ambient symbols used across the
/// runtime. Built once at `Lua::new` and shared (by Gc identity) with
/// every other use of the same byte sequence.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Symbols<'gc> {
    pub mm_index: LuaString<'gc>,
    pub mm_newindex: LuaString<'gc>,
    pub mm_add: LuaString<'gc>,
    pub mm_sub: LuaString<'gc>,
    pub mm_mul: LuaString<'gc>,
    pub mm_div: LuaString<'gc>,
    pub mm_mod: LuaString<'gc>,
    pub mm_pow: LuaString<'gc>,
    pub mm_idiv: LuaString<'gc>,
    pub mm_band: LuaString<'gc>,
    pub mm_bor: LuaString<'gc>,
    pub mm_bxor: LuaString<'gc>,
    pub mm_bnot: LuaString<'gc>,
    pub mm_shl: LuaString<'gc>,
    pub mm_shr: LuaString<'gc>,
    pub mm_unm: LuaString<'gc>,
    pub mm_eq: LuaString<'gc>,
    pub mm_lt: LuaString<'gc>,
    pub mm_le: LuaString<'gc>,
    pub mm_concat: LuaString<'gc>,
    pub mm_len: LuaString<'gc>,
    pub mm_call: LuaString<'gc>,
    pub mm_tostring: LuaString<'gc>,
}

impl<'gc> Symbols<'gc> {
    pub(crate) fn intern_all(mc: &Mutation<'gc>, interner: &Interner<'gc>) -> Self {
        Symbols {
            mm_index: interner.intern(mc, b"__index"),
            mm_newindex: interner.intern(mc, b"__newindex"),
            mm_add: interner.intern(mc, b"__add"),
            mm_sub: interner.intern(mc, b"__sub"),
            mm_mul: interner.intern(mc, b"__mul"),
            mm_div: interner.intern(mc, b"__div"),
            mm_mod: interner.intern(mc, b"__mod"),
            mm_pow: interner.intern(mc, b"__pow"),
            mm_idiv: interner.intern(mc, b"__idiv"),
            mm_band: interner.intern(mc, b"__band"),
            mm_bor: interner.intern(mc, b"__bor"),
            mm_bxor: interner.intern(mc, b"__bxor"),
            mm_bnot: interner.intern(mc, b"__bnot"),
            mm_shl: interner.intern(mc, b"__shl"),
            mm_shr: interner.intern(mc, b"__shr"),
            mm_unm: interner.intern(mc, b"__unm"),
            mm_eq: interner.intern(mc, b"__eq"),
            mm_lt: interner.intern(mc, b"__lt"),
            mm_le: interner.intern(mc, b"__le"),
            mm_concat: interner.intern(mc, b"__concat"),
            mm_len: interner.intern(mc, b"__len"),
            mm_call: interner.intern(mc, b"__call"),
            mm_tostring: interner.intern(mc, b"__tostring"),
        }
    }

    /// All metamethod (name, bit) pairs. Used by
    /// `Table::ensure_mt_cache` to seed an `MtCache` by walking a
    /// metatable's slots once at first-adoption time.
    #[inline]
    pub fn metamethods(&self) -> [(LuaString<'gc>, MetamethodBits); 23] {
        [
            (self.mm_index, MetamethodBits::INDEX),
            (self.mm_newindex, MetamethodBits::NEWINDEX),
            (self.mm_add, MetamethodBits::ADD),
            (self.mm_sub, MetamethodBits::SUB),
            (self.mm_mul, MetamethodBits::MUL),
            (self.mm_div, MetamethodBits::DIV),
            (self.mm_mod, MetamethodBits::MOD),
            (self.mm_pow, MetamethodBits::POW),
            (self.mm_idiv, MetamethodBits::IDIV),
            (self.mm_band, MetamethodBits::BAND),
            (self.mm_bor, MetamethodBits::BOR),
            (self.mm_bxor, MetamethodBits::BXOR),
            (self.mm_bnot, MetamethodBits::BNOT),
            (self.mm_shl, MetamethodBits::SHL),
            (self.mm_shr, MetamethodBits::SHR),
            (self.mm_unm, MetamethodBits::UNM),
            (self.mm_eq, MetamethodBits::EQ),
            (self.mm_lt, MetamethodBits::LT),
            (self.mm_le, MetamethodBits::LE),
            (self.mm_concat, MetamethodBits::CONCAT),
            (self.mm_len, MetamethodBits::LEN),
            (self.mm_call, MetamethodBits::CALL),
            (self.mm_tostring, MetamethodBits::TOSTRING),
        ]
    }
}
