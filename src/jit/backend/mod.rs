//! Machine code: code memory, the target-neutral VCode with its regalloc2
//! adapter, and the per-target lowering and emission.

pub(crate) mod alloc;
pub(crate) mod code;
pub(crate) mod vcode;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
