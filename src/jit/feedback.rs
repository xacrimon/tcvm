//! A site's feedback bits (`FeedbackVector`): what the interpreter's slow
//! paths and the JIT's exits saw there. Only ever OR-ed into, so they grow
//! monotonically.

pub(crate) use crate::env::feedback::Bits;
use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::instruction::Instruction;

pub(crate) const SMALL: Bits = 1;
pub(crate) const BIGINT: Bits = 2;
pub(crate) const FLOAT: Bits = 4;
pub(crate) const STR: Bits = 8;
pub(crate) const TAB: Bits = 16;
pub(crate) const OTHER: Bits = 32;
/// Small operands gave a result outside i32.
pub(crate) const OVERFLOW: Bits = 64;
/// A metamethod ran.
pub(crate) const MM: Bits = 128;
/// At a `FORPREP`: an integer loop's limit was a float.
pub(crate) const FLOAT_LIMIT: Bits = OTHER;
/// At a `GETTABLE` or `SETTABLE`, whose kinds are the key's (`OTHER` also
/// for a receiver that is no table): a small key was outside the array part,
/// or over a nil there with the class having `__index`/`__newindex`.
pub(crate) const ARR_MISS: Bits = BIGINT;
/// At a call (whose kinds are its first argument's) or a `GETUPVAL`: a value
/// a region took as constant differed (`ExitTag::Const`).
pub(crate) const CONST_MISS: Bits = MM;
/// Compiled code exited on operand kinds the site's form does not cover.
pub(crate) const KIND_EXIT: Bits = 256;

pub(crate) const INT: Bits = SMALL | BIGINT;
pub(crate) const KINDS: Bits = SMALL | BIGINT | FLOAT | STR | TAB | OTHER;

/// The kind bit of `v`.
#[inline(always)]
pub(crate) fn kind(v: Value<'_>) -> Bits {
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

/// The bits of a `GETTABLE`/`SETTABLE` that left the array fast path.
#[inline(always)]
pub(crate) fn table_key(recv: Value<'_>, key: Value<'_>) -> Bits {
    let recv = if recv.get_table().is_some() { 0 } else { OTHER };
    let miss = if key.get_small().is_some() {
        ARR_MISS
    } else {
        0
    };
    recv | kind(key) | miss
}

/// OR `bits` into the feedback of the instruction at `site` of `closure`.
#[inline(always)]
pub(crate) fn record(closure: LuaFn<'_>, site: *const Instruction, bits: Bits) {
    // SAFETY: `site` is an instruction of `closure`'s code.
    let pc = unsafe { site.offset_from_unsigned(closure.code) };
    closure.proto.feedback.add_bits(pc, bits);
}
