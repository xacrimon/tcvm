//! The method JIT: regions compiled from a prototype's function entry or a hot
//! loop header, entered and left as dispatch targets. Compiled under `cfg(jit)`
//! (`build.rs`); otherwise only what the interpreter keeps for it remains: the
//! feedback kinds it records and the groups of a watched holder's slots.

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
#[cfg(all(test, jit))]
mod rails;
#[cfg(jit)]
pub(crate) mod region;
#[cfg(jit)]
pub(crate) mod state;
// The old compiler's tests, off until the stages that bring their subjects
// back.
#[cfg(any())]
mod tests;
#[cfg(jit)]
mod tier;

#[cfg(jit)]
pub(crate) use tier::*;

/// A violated compiler invariant: a panic in every build, through one cold
/// path that carries its message only with `tcvm-dev`.
#[cfg(jit)]
macro_rules! bug {
    ($($t:tt)*) => {{
        // Never run without `tcvm-dev`: it only uses what the message names.
        if cfg!(feature = "tcvm-dev") {
            panic!($($t)*)
        }
        $crate::jit::internal_error()
    }};
}
#[cfg(jit)]
pub(crate) use bug;

#[cfg(jit)]
#[cold]
#[inline(never)]
pub(crate) fn internal_error() -> ! {
    panic!("jit: internal error")
}
