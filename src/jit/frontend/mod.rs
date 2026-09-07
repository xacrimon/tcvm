//! Bytecode -> SSA.
//!
//! The frontend is *simultaneously* the SSA builder and the block-versioning
//! driver: it does not build a generic SSA function and then specialize it, it
//! abstract-interprets the bytecode with a symbolic register state and emits
//! typed IR directly. Where the type state is known, it emits specialized ops
//! with no guard; where it is unknown but a feedback source has an opinion, it
//! emits a guard and specializes on the refined value; where nothing is known,
//! it falls back to the generic `lua.*` ops.

pub mod cfg;
pub mod lower;
pub mod print;
pub mod sink;
pub mod ssa;

#[cfg(test)]
mod snapshot_tests;
