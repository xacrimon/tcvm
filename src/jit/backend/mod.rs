//! Machine code: code memory, the block layout, the target-neutral VCode,
//! the register allocator, and the per-target lowering and emission.

pub(crate) mod alloc;
pub(crate) mod code;
pub(crate) mod order;
pub(crate) mod ra;
pub(crate) mod reg;
pub(crate) mod vcode;

// The one place a target is chosen: everything else names `target`. No arm
// for other targets: `cfg(jit)` (`build.rs`) holds only where one exists.
cfg_select! {
    target_arch = "aarch64" => {
        pub(crate) mod aarch64;
        pub(crate) use aarch64 as target;
    }
}

/// The words the target's test of `set` compares with, which the IR gives
/// it as constant operands (`opt::words`).
pub(crate) fn test_words(set: crate::jit::ir::types::TypeSet) -> Vec<u64> {
    target::emit::test_words(set)
}
