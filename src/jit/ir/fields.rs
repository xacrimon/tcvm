//! The fields compiled code reads and writes at a fixed offset from a
//! pointer: one row gives a field's offset, representation and abstract heap.
//! A `Load` or `Store` names its field in `aux`; an indexed field takes the
//! index of its 8-byte slot in `c`, and each index is a heap key of its own.

use super::ops::Heaps;
use super::types::{Rep, TypeSet};
use crate::jit::layout;

pub(crate) struct FieldInfo {
    pub(crate) name: &'static str,
    pub(crate) off: i32,
    pub(crate) rep: Rep,
    /// What a `Val` field holds, by its definition.
    pub(crate) set: TypeSet,
    pub(crate) heap: Heaps,
    pub(crate) indexed: bool,
}

macro_rules! fields {
    ($($name:ident $off:expr, $rep:ident $set:ident $heap:ident $indexed:literal;)*) => {
        #[repr(u16)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum Field { $($name),* }

        pub(crate) static FIELDS: [FieldInfo; [$(Field::$name),*].len()] = [$(FieldInfo {
            name: stringify!($name),
            off: $off as i32,
            rep: Rep::$rep,
            set: TypeSet::$set,
            heap: Heaps::$heap,
            indexed: $indexed,
        }),*];
    };
}

// name, offset, representation, types, heap, indexed. A table's fields are
// from its lock, a table value's payload; `IMMUTABLE` is memory nothing writes.
fields! {
    Shape    layout::T_SHAPE,  Ptr ANY SHAPE     false;
    // The spill array moves only as a transition grows it.
    Spill    layout::T_SPILL,  Ptr ANY SHAPE     false;
    Array    layout::T_ARRAY,  Ptr ANY ARR_SIZE  false;
    ASize    layout::T_ASIZE,  I32 ANY ARR_SIZE  false;
    // A named slot in the table, and one in its spill array (from that).
    Inline   layout::T_INLINE, Val ANY FIELD     true;
    Spilled  0,                Val ANY FIELD     true;
    // An array part's element, from the array (`LoadIdx`).
    Item     0,                Val ANY ARR_ITEMS false;
    // A table's collector header: bit 0 set while no barrier is due.
    GcHeader layout::T_HEADER, I64 ANY GC        false;
    // A shape's class (null for none), and the class's metamethod bits.
    ShapeClass crate::env::Shape::MT_CACHE_OFFSET, Ptr ANY IMMUTABLE false;
    ClassBits crate::env::MtCache::BITS_OFFSET, I32 ANY META false;
    // A class's metamethods besides `__index` and `__newindex` (null while
    // it has none), and one of them by its bit's index.
    ClassRest crate::env::MtCache::REST_OFFSET, Ptr ANY META false;
    RestMm   crate::env::MtCache::REST_VALUES_OFFSET, Val ANY META true;
    // A basic type's metatable (null for none), from its cell.
    TypeMt   layout::NUMBER_MT, Ptr ANY META  false;
    // The frame's function, from the base: its closure under header flags.
    FrameFn  -32,              Val FUN FRAME     false;
    // A function's kind, a byte (0: a Lua closure without varargs), and
    // such a closure's prototype.
    FnKind   layout::FN_IS_VARARG, I32 ANY IMMUTABLE false;
    FnProto  layout::FN_PROTO, Ptr ANY IMMUTABLE false;
    // A string's length in bytes.
    StrLen   crate::env::string::LuaString::LEN_OFFSET, I64 ANY IMMUTABLE false;
    // The frame's flags (`vm::frame::flag`), from the base.
    FrameFlags -24,            I64 ANY FRAME     false;
    // A Lua closure's upvalues, from the closure (the offset is the
    // runtime's, `compile::upvalues_offset`): a value, or a shared cell.
    Upval    0,                Val ANY IMMUTABLE true;
    UpvalCell 0,               Ptr ANY IMMUTABLE true;
    // Where a cell's value is (a stack slot while open), and the value there.
    CellAt   crate::env::function::UpvalueCell::V_OFFSET, Ptr ANY UPVAL false;
    CellValue 0,               Val ANY UPVAL     false;
    // The collector's allocation counter and its threshold, from the
    // arena's metrics.
    GcUsed   layout::GC_CHECK, I64 ANY GC        false;
    GcLimit  layout::GC_CHECK + 8, I64 ANY GC    false;
}

impl Field {
    pub(crate) fn info(aux: u16) -> &'static FieldInfo {
        &FIELDS[aux as usize]
    }

    /// The byte offset of an access to field `aux` at index `c`.
    pub(crate) fn offset(aux: u16, c: u32) -> i32 {
        let f = Field::info(aux);
        let upvalue = aux == Field::Upval as u16 || aux == Field::UpvalCell as u16;
        let base = if upvalue {
            crate::jit::compile::upvalues_offset() as i32
        } else {
            f.off
        };
        base + if f.indexed { 8 * c as i32 } else { 0 }
    }
}
