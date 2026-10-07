//! The opcode handlers, by family. Every dispatch target is declared with
//! `handler!` (`vm::abi`); shared logic is an `#[inline(always)]` function.

pub(crate) mod arith;
pub(crate) mod call;
pub(crate) mod compare;
pub(crate) mod control;
pub(crate) mod data;
pub(crate) mod field;
pub(crate) mod meta;
pub(crate) mod table;
