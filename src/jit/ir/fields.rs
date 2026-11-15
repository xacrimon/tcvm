//! The fields compiled code reads and writes at a fixed offset from a
//! pointer: one row gives a field's offset, representation and abstract heap.
//! A `Load` or `Store` names its field in `aux`; an indexed field takes the
//! index of its 8-byte slot in `c`, and each index is a heap key of its own.

use super::ops::Heaps;
use super::types::Rep;
use crate::jit::layout;

pub(crate) struct FieldInfo {
    pub(crate) name: &'static str,
    pub(crate) off: i32,
    pub(crate) rep: Rep,
    pub(crate) heap: Heaps,
    pub(crate) indexed: bool,
}

macro_rules! fields {
    ($($name:ident $off:expr, $rep:ident $heap:ident $indexed:literal;)*) => {
        #[repr(u16)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum Field { $($name),* }

        pub(crate) static FIELDS: [FieldInfo; [$(Field::$name),*].len()] = [$(FieldInfo {
            name: stringify!($name),
            off: $off as i32,
            rep: Rep::$rep,
            heap: Heaps::$heap,
            indexed: $indexed,
        }),*];
    };
}

// name, offset, representation, heap, indexed. A table's fields are from
// its lock, a table value's payload.
fields! {
    Shape    layout::T_SHAPE,  Ptr SHAPE     false;
    // The spill array moves only as a transition grows it.
    Spill    layout::T_SPILL,  Ptr SHAPE     false;
    Array    layout::T_ARRAY,  Ptr ARR_SIZE  false;
    ASize    layout::T_ASIZE,  I32 ARR_SIZE  false;
    // A named slot in the table, and one in its spill array (from that).
    Inline   layout::T_INLINE, Val FIELD     true;
    Spilled  0,                Val FIELD     true;
    // An array part's element, from the array (`LoadIdx`).
    Item     0,                Val ARR_ITEMS false;
}

impl Field {
    pub(crate) fn info(aux: u16) -> &'static FieldInfo {
        &FIELDS[aux as usize]
    }
}
