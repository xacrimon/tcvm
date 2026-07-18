//! Native code generation.
//!
//! The optimizing IR is lowered to a machine IR, register-allocated, and
//! encoded. Two targets exist — aarch64 and x86-64 — and the machine IR is kept
//! free of either's specifics, so an encoder is an addition rather than a rewrite.
//!
//! The layering runs one way. [`regalloc`] is the leaf: it names the vocabulary a
//! program is described in (`VReg`, `RegClass`, `Operand`) and knows nothing about
//! either level above it. [`mach`] describes the machine IR *in* that vocabulary
//! and implements `regalloc::RegallocFunc` over it. The per-target encoder module
//! ([`aarch64`], [`x64`]) owns everything that is actually about the machine —
//! which registers exist, which are scratch, how wide a spill slot is — and hands
//! the allocator a `MachineEnv` describing them. A second target adds a sibling
//! and touches nothing else.
//!
//! The host's target is selected once, here, as [`target`]: it re-exports the
//! encoder module for the architecture being built, so the rest of the crate
//! (`region.rs`, the tests) names `target::encode` / `target::machine_env` /
//! `target::Status` and stays architecture-agnostic. Only the module for the host
//! ISA is compiled — the other carries encodings the host cannot run.

pub mod alloc;
pub mod code;
pub mod isel;
pub mod layout;
pub mod mach;
pub mod regalloc;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "aarch64")]
pub mod aarch64_asm;

#[cfg(target_arch = "x86_64")]
pub mod x64;
#[cfg(target_arch = "x86_64")]
pub mod x64_asm;

/// The encoder for the architecture being built. See the module docs.
#[cfg(target_arch = "aarch64")]
pub use aarch64 as target;
#[cfg(target_arch = "x86_64")]
pub use x64 as target;

#[cfg(all(test, target_arch = "aarch64"))]
mod asm_dump;
#[cfg(test)]
mod exec_tests;
#[cfg(test)]
mod snapshot_tests;
