//! Branch hints: core's with `nightly`, built from the stable `cold_path` otherwise.

#[cfg(nightly)]
pub(crate) use core::hint::{likely, unlikely};

#[cfg(not(nightly))]
#[inline(always)]
pub(crate) fn likely(b: bool) -> bool {
    if !b {
        core::hint::cold_path();
    }
    b
}

#[cfg(not(nightly))]
#[inline(always)]
pub(crate) fn unlikely(b: bool) -> bool {
    if b {
        core::hint::cold_path();
    }
    b
}
