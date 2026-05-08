pub mod error;
pub mod function;
pub mod shape;
pub mod string;
pub mod symbols;
pub mod table;
pub mod thread;
pub mod userdata;
pub mod value;

/// Master metamethod table — single source of truth for the set of
/// names that drive bit positions in `MetamethodBits`, byte-name
/// lookups in `shape::metamethod_bit_of_bytes`, and pre-interned
/// `LuaString` identities on `Symbols`. Adding a metamethod here
/// threads it through every consumer.
///
/// Each row: `(BIT_NAME, bit_position, b"__lua_name", symbols_field)`.
/// Bit positions are explicit so reordering rows can't silently shift
/// them.
///
/// Invoke as `for_each_metamethod!(callback)`; the callback receives
/// the full list as a `;`-separated repetition and is responsible for
/// generating the consumer-specific definitions.
macro_rules! for_each_metamethod {
    ($cb:ident) => {
        $cb! {
            (INDEX,    0,  b"__index",    mm_index);
            (NEWINDEX, 1,  b"__newindex", mm_newindex);
            (ADD,      2,  b"__add",      mm_add);
            (SUB,      3,  b"__sub",      mm_sub);
            (MUL,      4,  b"__mul",      mm_mul);
            (DIV,      5,  b"__div",      mm_div);
            (MOD,      6,  b"__mod",      mm_mod);
            (POW,      7,  b"__pow",      mm_pow);
            (IDIV,     8,  b"__idiv",     mm_idiv);
            (BAND,     9,  b"__band",     mm_band);
            (BOR,      10, b"__bor",      mm_bor);
            (BXOR,     11, b"__bxor",     mm_bxor);
            (BNOT,     12, b"__bnot",     mm_bnot);
            (SHL,      13, b"__shl",      mm_shl);
            (SHR,      14, b"__shr",      mm_shr);
            (UNM,      15, b"__unm",      mm_unm);
            (EQ,       16, b"__eq",       mm_eq);
            (LT,       17, b"__lt",       mm_lt);
            (LE,       18, b"__le",       mm_le);
            (CONCAT,   19, b"__concat",   mm_concat);
            (LEN,      20, b"__len",      mm_len);
            (CALL,     21, b"__call",     mm_call);
            (TOSTRING, 22, b"__tostring", mm_tostring);
        }
    };
}
pub(crate) use for_each_metamethod;

pub use error::Error;
pub use function::{Function, NativeContext, NativeFn, Prototype, Stack};
pub use shape::{MetamethodBits, MtCache, Shape};
pub use string::LuaString;
pub use symbols::Symbols;
pub use table::Table;
pub use thread::Thread;
pub use userdata::Userdata;
pub use value::{Value, ValueKind};
