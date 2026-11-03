//! Machine code: code memory, the block layout, the target-neutral VCode,
//! the register allocator, and the per-target lowering and emission.

pub(crate) mod alloc;
pub(crate) mod code;
pub(crate) mod order;
pub(crate) mod ra;
pub(crate) mod vcode;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

/// The words the target's test of `set` compares with, which the IR gives
/// it as constant operands (`opt::words`).
pub(crate) fn test_words(set: crate::jit::ir::types::TypeSet) -> Vec<u64> {
    #[cfg(target_arch = "aarch64")]
    return aarch64::emit::test_words(set);
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = set;
        Vec::new()
    }
}
