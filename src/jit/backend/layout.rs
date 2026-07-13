//! Runtime memory layout, as constants the encoder can bake into instructions.
//!
//! Compiled code loads and stores runtime fields at constant displacements off
//! a raw pointer, which means it hardcodes this crate's struct layouts. None of
//! the types involved is `#[repr(C)]`, so every offset here is *derived* with
//! `offset_of!` rather than written down. That is not fastidiousness:
//!
//!   - `Value`'s field order is the compiler's choice, and it will happily put
//!     the `u64` payload before the one-byte tag.
//!   - `RefLock`'s payload offset **differs between debug and release**, because
//!     the borrow flag inside `CheckedCell` is `cfg`-gated on `debug_assertions`.
//!     A hand-written constant would pass every test and miscompile what ships.
//!
//! Derived constants make a field reorder move the generated code with it. The
//! asserts below then pin the facts that codegen *reasons* with — a 16-byte
//! `Value` stride is what lets a slot index become a shift — so that changing
//! one breaks the build here instead of the JIT.

use crate::dmm::RefLock;
use crate::env::table::TableState;
use crate::env::value;

/// A `Value`: `{ kind: ValueKind, data: u64 }`, 16 bytes.
pub mod val {
    pub const KIND: usize = crate::env::value::layout::KIND;
    pub const DATA: usize = crate::env::value::layout::DATA;
    pub const SIZE: usize = crate::env::value::layout::SIZE;
}

/// A `Table`, whose `Gc` pointer addresses a `RefLock<TableState>`.
///
/// Offsets are from the start of the `RefLock`, i.e. straight off the pointer in
/// a table `Value`'s payload. Compiled code reads through the lock without
/// touching the borrow flag: it never calls back into anything that could take a
/// conflicting borrow, and the flag is debug-only anyway.
pub mod table {
    use super::*;

    /// Payload start: past the (debug-only) borrow flag.
    const BASE: usize = RefLock::<TableState<'static>>::PAYLOAD_OFFSET;

    /// The `Shape` pointer a `guard.shape` compares against.
    pub const SHAPE: usize = BASE + core::mem::offset_of!(TableState<'static>, shape);
    /// Mirror of the property array's data pointer. See `TableState::props_ptr`.
    pub const PROPS_PTR: usize = BASE + core::mem::offset_of!(TableState<'static>, props_ptr);
}

/// A `ValueKind` discriminant, as the byte compiled code compares a tag against.
pub const fn kind(k: value::ValueKind) -> u8 {
    k as u8
}

// Codegen turns a property slot index into a byte displacement by shifting, and
// materializes a tag test as a single byte compare. Both assume these.
const _: () = assert!(val::SIZE == 16, "Value stride must be 16 for slot.get");
const _: () = assert!(val::KIND != val::DATA);
const _: () = assert!(size_of::<value::ValueKind>() == 1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::value::ValueKind;

    /// The layout constants are only worth anything if they match what Rust
    /// actually laid out. Build a real `Value`, look at its bytes, and confirm
    /// the tag and payload are where the encoder will go looking.
    #[test]
    fn value_offsets_match_reality() {
        let v = crate::env::value::Value::integer(-2);
        let bytes: [u8; val::SIZE] = unsafe { std::mem::transmute(v) };

        assert_eq!(bytes[val::KIND], kind(ValueKind::Integer));
        let data = u64::from_le_bytes(bytes[val::DATA..val::DATA + 8].try_into().unwrap());
        assert_eq!(data as i64, -2);
    }
}
