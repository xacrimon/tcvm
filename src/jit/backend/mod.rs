//! Native code generation.
//!
//! The optimizing IR is lowered to a machine IR, register-allocated, and
//! encoded. aarch64 is the only target for now; the machine IR is kept free of
//! aarch64 specifics so a second encoder is an addition rather than a rewrite.

pub mod code;
pub mod isel;
pub mod layout;
pub mod mach;

#[cfg(test)]
mod snapshot_tests;
