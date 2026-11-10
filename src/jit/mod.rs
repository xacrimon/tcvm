//! The method JIT (`jit-design.md`): regions compiled from a prototype's
//! function entry or a hot loop header, entered and left as dispatch targets.
//! Compiled under `cfg(jit)` (`build.rs`); otherwise only what the
//! interpreter keeps for it remains: the feedback kinds it records and the
//! groups of a watched holder's slots.

#[cfg(jit)]
pub(crate) mod backend;
#[cfg(jit)]
pub(crate) mod build;
#[cfg(jit)]
pub(crate) mod compile;
pub(crate) mod deps;
pub(crate) mod feedback;
#[cfg(jit)]
pub(crate) mod helpers;
#[cfg(jit)]
pub(crate) mod ir;
#[cfg(jit)]
pub(crate) mod layout;
#[cfg(jit)]
pub(crate) mod opt;
#[cfg(jit)]
pub(crate) mod region;
#[cfg(jit)]
pub(crate) mod state;
#[cfg(all(test, jit))]
mod tests;
#[cfg(jit)]
mod tier;

#[cfg(jit)]
pub(crate) use tier::*;
