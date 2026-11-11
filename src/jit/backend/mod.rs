//! Machine code: code memory, the block layout, the target-neutral VCode,
//! the register allocator, and the per-target instruction selection and
//! emission.
//!
//! Everything outside a target's own module names it as `target`, and uses
//! of it only what is listed here; a new target exports the same items.
//!
//! - `target::inst`: `MInst`, its instructions (`vcode::TargetInst`).
//! - `target::lower`: `lower::Select` for `lower::Lower`, and `Isel`, its
//!   state.
//! - `target::emit`: `emit` (a lowered, allocated region to `emit::Emitted`),
//!   `trampoline`, `known_offset`.
//! - `target::types`: `test_words`, the words its type tests compare with.
//! - `target::asm`: `Asm`, with `max_len` and `finish`.
//! - `target::abi`: the allocator's registers (`ra_env`, `allocatable`), the
//!   registers calls pass arguments in (`REG_ARGS`), the exit's register
//!   image (`EXIT_REGS`, `IMAGE_SPILLS`, `image_index`), the frame's spill
//!   slots (`spill_addr`), a return address's tag (`RA_TAG`),
//!   `stack_pointer`, register names for the dumps (`reg_name`), and the stubs
//!   between compiled code, the interpreter's handlers and levels
//!   (`dispatch_entry`, `level_jump`, `level_land`, `level_return_addr`,
//!   `ret_to_native`, `ret_shim`, `direct_miss`).
//! - `target::code`: branch reach and the encodings code memory patches
//!   (`REACH`, `RANGE_SIZES`, `reaches`, `BRANCH`, `retarget`, `call`, `JUMP`,
//!   `jump`, `ISLAND`, `island`), `sync_icache`, and `objdump` for the
//!   dumps.
//! - `target::unwind`: the DWARF registers and rules of `unwind::Fde`
//!   (`CODE_ALIGN`, `RETURN_COLUMN`, `entry_rules`, `frame_rules`).

pub(crate) mod alloc;
pub(crate) mod code;
pub(crate) mod emit;
pub(crate) mod lower;
pub(crate) mod order;
pub(crate) mod ra;
pub(crate) mod reg;
pub(crate) mod unwind;
pub(crate) mod vcode;

// The one place a target is chosen. No arm for other targets: `cfg(jit)`
// (`build.rs`) holds only where one exists.
cfg_select! {
    target_arch = "aarch64" => {
        pub(crate) mod aarch64;
        pub(crate) use aarch64 as target;
    }
    target_arch = "x86_64" => {
        pub(crate) mod x86_64;
        pub(crate) use x86_64 as target;
    }
}

pub(crate) use target::types::test_words;
