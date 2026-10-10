//! The per-instruction feedback byte (`Prototype::feedback`): what the
//! interpreter's slow paths and the JIT's exits saw at a site. Only ever
//! OR-ed into, so it grows monotonically.

use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::instruction::Instruction;

pub(crate) const SMALL: u8 = 1;
pub(crate) const BIGINT: u8 = 2;
pub(crate) const FLOAT: u8 = 4;
pub(crate) const STR: u8 = 8;
pub(crate) const TAB: u8 = 16;
pub(crate) const OTHER: u8 = 32;
/// Small operands gave a result outside i32.
pub(crate) const OVERFLOW: u8 = 64;
/// A metamethod ran.
pub(crate) const MM: u8 = 128;
/// At a `FORPREP`: an integer loop's limit was a float.
pub(crate) const FLOAT_LIMIT: u8 = OTHER;

pub(crate) const INT: u8 = SMALL | BIGINT;
pub(crate) const KINDS: u8 = SMALL | BIGINT | FLOAT | STR | TAB | OTHER;

/// The kind bit of `v`.
#[inline(always)]
pub(crate) fn kind(v: Value<'_>) -> u8 {
    if v.is_float() {
        FLOAT
    } else if v.get_small().is_some() {
        SMALL
    } else if v.get_integer().is_some() {
        BIGINT
    } else if v.get_string().is_some() {
        STR
    } else if v.get_table().is_some() {
        TAB
    } else {
        OTHER
    }
}

/// OR `bits` into the feedback of the instruction at `site` of `closure`.
/// Stored only on a change: slow paths record on every run.
#[inline(always)]
pub(crate) fn record(closure: LuaFn<'_>, site: *const Instruction, bits: u8) {
    // SAFETY: `site` is an instruction of `closure`'s code, which the
    // feedback parallels.
    unsafe {
        let i = site.offset_from_unsigned(closure.code);
        let cell = closure.proto.feedback.get_unchecked(i);
        let old = cell.get();
        if old | bits != old {
            cell.set(old | bits);
        }
    }
}
