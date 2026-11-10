//! The method JIT (`jit-design.md`): regions compiled from a prototype's
//! function entry or a hot loop header, entered and left as dispatch targets.
//! Compiled with the `jit` feature on aarch64; otherwise only what the
//! interpreter keeps for it remains: the feedback kinds it records and the
//! groups of a watched holder's slots.

#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod backend;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod build;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod compile;
pub(crate) mod deps;
pub(crate) mod feedback;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod helpers;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod ir;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod layout;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod opt;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod region;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) mod state;
#[cfg(all(test, feature = "jit", target_arch = "aarch64"))]
mod tests;
#[cfg(all(feature = "jit", target_arch = "aarch64"))]
mod tier;

#[cfg(all(feature = "jit", target_arch = "aarch64"))]
pub(crate) use tier::*;
