//! Native code generation.
//!
//! The optimizing IR is lowered to a machine IR, register-allocated, and
//! encoded. aarch64 is the only target for now; the machine IR is kept free of
//! aarch64 specifics so a second encoder is an addition rather than a rewrite.

pub mod aarch64;
pub mod asm;
pub mod code;
pub mod isel;
pub mod layout;
pub mod mach;
pub mod regalloc;

#[cfg(test)]
mod exec_tests;
#[cfg(test)]
mod snapshot_tests;
