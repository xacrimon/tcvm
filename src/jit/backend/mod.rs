//! Native code generation.
//!
//! The optimizing IR is lowered to a machine IR, register-allocated, and
//! encoded. aarch64 is the only target for now; the machine IR is kept free of
//! aarch64 specifics so a second encoder is an addition rather than a rewrite.
//!
//! The layering runs one way. [`regalloc`] is the leaf: it names the vocabulary a
//! program is described in (`VReg`, `RegClass`, `Operand`) and knows nothing about
//! either level above it. [`mach`] describes the machine IR *in* that vocabulary
//! and implements `regalloc::RegallocFunc` over it. [`aarch64`] owns everything
//! that is actually about the machine — which registers exist, which are scratch,
//! how wide a spill slot is — and hands the allocator a `MachineEnv` describing
//! them. A second target adds a sibling of `aarch64` and touches nothing else.

pub mod aarch64;
pub mod alloc;
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
