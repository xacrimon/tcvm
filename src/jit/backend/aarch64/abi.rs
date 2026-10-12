//! Registers of a region (Appendix A), the allocator's machine environment,
//! the register image order, and `exit_common`.

use regalloc2::{MachineEnv, PReg, PRegSet, RegClass};

use crate::jit::backend::aarch64::asm::{Fpr, Gpr};
use crate::jit::state::EXIT_REGS;
pub(crate) use crate::jit::state::IMAGE_SPILLS;

/// Pinned for a region's whole body.
pub(crate) const BASE: Gpr = Gpr(22);
pub(crate) const RT: Gpr = Gpr(23);
pub(crate) const CLOSURE: Gpr = Gpr(24);
pub(crate) const THREAD: Gpr = Gpr(25);
/// The handler slots loaded at every tail-out.
pub(crate) const INSN: Gpr = Gpr(20);
pub(crate) const PC: Gpr = Gpr(21);
/// Scratch inside one instruction; never allocated.
pub(crate) const X16: Gpr = Gpr(16);
pub(crate) const X17: Gpr = Gpr(17);
pub(crate) const D31: Fpr = Fpr(31);

/// Allocatable integer registers, caller-saved first.
pub(crate) const INT_PREFERRED: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
pub(crate) const INT_OTHER: [u8; 6] = [19, 20, 21, 26, 27, 28];
pub(crate) const FLOAT_PREFERRED: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
];
pub(crate) const FLOAT_OTHER: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

pub(crate) fn machine_env() -> MachineEnv {
    let mut ip = PRegSet::empty();
    for r in INT_PREFERRED {
        ip.add(PReg::new(r as usize, RegClass::Int));
    }
    let mut io = PRegSet::empty();
    for r in INT_OTHER {
        io.add(PReg::new(r as usize, RegClass::Int));
    }
    let mut fp = PRegSet::empty();
    for r in FLOAT_PREFERRED {
        fp.add(PReg::new(r as usize, RegClass::Float));
    }
    let mut fo = PRegSet::empty();
    for r in FLOAT_OTHER {
        fo.add(PReg::new(r as usize, RegClass::Float));
    }
    MachineEnv {
        preferred_regs_by_class: [ip, fp, PRegSet::empty()],
        non_preferred_regs_by_class: [io, fo, PRegSet::empty()],
        scratch_by_class: [
            Some(PReg::new(17, RegClass::Int)),
            Some(PReg::new(31, RegClass::Float)),
            None,
        ],
        fixed_stack_slots: Vec::new(),
    }
}

/// Every register the allocator hands out: what a call into another region
/// may change.
pub(crate) fn allocatable() -> PRegSet {
    let mut s = PRegSet::empty();
    for r in INT_PREFERRED.iter().chain(&INT_OTHER) {
        s.add(PReg::new(*r as usize, RegClass::Int));
    }
    for r in FLOAT_PREFERRED.iter().chain(&FLOAT_OTHER) {
        s.add(PReg::new(*r as usize, RegClass::Float));
    }
    s
}

/// What a C-ABI call clobbers: x0-x17 and d0-d7, d16-d31.
pub(crate) fn caller_saved() -> PRegSet {
    let mut s = PRegSet::empty();
    for r in 0..=17 {
        s.add(PReg::new(r, RegClass::Int));
    }
    for r in (0..=7).chain(16..=31) {
        s.add(PReg::new(r, RegClass::Float));
    }
    s
}

pub(crate) fn preg_int(r: u8) -> PReg {
    PReg::new(r as usize, RegClass::Int)
}

pub(crate) fn preg_float(r: u8) -> PReg {
    PReg::new(r as usize, RegClass::Float)
}

/// The register image slot `exit_common` stores a register in.
pub(crate) fn image_index(p: PReg) -> u8 {
    let n = p.hw_enc() as u8;
    match p.class() {
        RegClass::Int => match n {
            0..=15 => n,
            19..=21 => 16 + (n - 19),
            26..=28 => 19 + (n - 26),
            _ => panic!("x{n} is not allocatable"),
        },
        RegClass::Float => 22 + n,
        RegClass::Vector => unreachable!(),
    }
}

/// The spill slots a region may have, all copied to the image.
pub(crate) const MAX_SPILLS: usize = 256;
const _: () = assert!(IMAGE_SPILLS + MAX_SPILLS <= EXIT_REGS);

/// Entered from a region's exit trampoline with x16 = region | exit << 48
/// and the region's frame still open: stores the register image in
/// `image_index` order, copies the frame's spill slots after it, and pops
/// the frame. A frame dispatch entered tails `jit_exit` in its level; a
/// fast-called one starts a level that returns to its caller, and the
/// interpreter runs the rest of it there (H10).
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn exit_common() {
    core::arch::naked_asm!(
        "ldr x17, [x23, #{regs}]",
        "stp x0, x1, [x17, #0]",
        "stp x2, x3, [x17, #16]",
        "stp x4, x5, [x17, #32]",
        "stp x6, x7, [x17, #48]",
        "stp x8, x9, [x17, #64]",
        "stp x10, x11, [x17, #80]",
        "stp x12, x13, [x17, #96]",
        "stp x14, x15, [x17, #112]",
        "stp x19, x20, [x17, #128]",
        "str x21, [x17, #144]",
        "stp x26, x27, [x17, #152]",
        "str x28, [x17, #168]",
        "stp d0, d1, [x17, #176]",
        "stp d2, d3, [x17, #192]",
        "stp d4, d5, [x17, #208]",
        "stp d6, d7, [x17, #224]",
        "stp d8, d9, [x17, #240]",
        "stp d10, d11, [x17, #256]",
        "stp d12, d13, [x17, #272]",
        "stp d14, d15, [x17, #288]",
        "stp d16, d17, [x17, #304]",
        "stp d18, d19, [x17, #320]",
        "stp d20, d21, [x17, #336]",
        "stp d22, d23, [x17, #352]",
        "stp d24, d25, [x17, #368]",
        "stp d26, d27, [x17, #384]",
        "stp d28, d29, [x17, #400]",
        "stp d30, d31, [x17, #416]",
        // The spill slots, [x29 + 24 + 8i], to the image's spill words.
        "and x9, x16, #0xffffffffffff",
        "ldr w10, [x9, #{num_spills}]",
        "add x11, x29, #{frame_spills}",
        "add x12, x17, #{spills}",
        "cbz w10, 2f",
        "1:",
        "ldr x13, [x11], #8",
        "str x13, [x12], #8",
        "subs w10, w10, #1",
        "b.ne 1b",
        // Pop the frame.
        "2:",
        "ldr w10, [x9, #{frame_size}]",
        "add x10, x29, x10",
        "ldp x29, x30, [x29]",
        "mov sp, x10",
        "mov x20, x16",
        "ldr x11, [x23, #{level_ret}]",
        "cmp x30, x11",
        "b.eq 3f",
        "adr x16, 3f",
        "mov x17, x22",
        "b {jit_level}",
        "3:",
        "b {jit_exit}",
        regs = const crate::jit::layout::EXIT_REGS,
        num_spills = const crate::jit::region::layout::NUM_SPILLS,
        frame_size = const crate::jit::region::layout::FRAME_SIZE,
        spills = const IMAGE_SPILLS * 8,
        level_ret = const crate::jit::layout::LEVEL_RET,
        frame_spills = const crate::jit::backend::aarch64::emit::FRAME_SPILLS,
        jit_level = sym jit_level,
        jit_exit = sym crate::jit::jit_exit,
    )
}

/// Native stack below a level's start past which fast calls branch with a
/// tagged return address instead of `blr`, and return with `br`: the return
/// address stack holds about 48 entries here, and a `ret` past it
/// mispredicts while the indirect predictor keeps up (5.4).
pub(crate) const RAS_STACK: usize = 512;

/// Start a level (H6): entered by `bl` or `b` with the return address in
/// x30, the handler to run in x16, the base of the level's first frame in
/// x17 (0 for none) and the handler's arguments in x20-x25. Pushes an entry
/// record, makes it the thread's, and branches to the handler with x30 =
/// `level_return`. The level ends in `ret_to_native` or `level_return`.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn jit_level() {
    core::arch::naked_asm!(
        "sub sp, sp, #{size}",
        "stp x29, x30, [sp, #0]",
        "ldr x9, [x25, #{record}]",
        "ldr x10, [x25, #{deep}]",
        "stp x9, x17, [sp, #{prev}]",
        "str x10, [sp, #{deep_saved}]",
        "stp x19, x20, [sp, #{x}]",
        "stp x21, x22, [sp, #{x} + 16]",
        "stp x23, x24, [sp, #{x} + 32]",
        "stp x25, x26, [sp, #{x} + 48]",
        "stp x27, x28, [sp, #{x} + 64]",
        "stp d8, d9, [sp, #{d}]",
        "stp d10, d11, [sp, #{d} + 16]",
        "stp d12, d13, [sp, #{d} + 32]",
        "stp d14, d15, [sp, #{d} + 48]",
        "mov x29, sp",
        "str x29, [x25, #{record}]",
        "sub x9, x29, #{ras}",
        "str x9, [x25, #{deep}]",
        "ldr x30, [x23, #{level_ret}]",
        "br x16",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        deep_saved = const crate::vm::stack::rec::DEEP,
        x = const crate::vm::stack::rec::X,
        d = const crate::vm::stack::rec::D,
        record = const crate::jit::layout::RECORD,
        deep = const crate::jit::layout::NATIVE_DEEP,
        level_ret = const crate::jit::layout::LEVEL_RET,
        ras = const RAS_STACK,
    )
}

/// Where a handler returns its `Exit` (in x0), which only the bottom level
/// does: pops its record and returns to `dispatch_entry`'s caller. Handlers
/// return 4 bytes in (`level_return_addr`), so an unwinder looking up the
/// return address less one finds this routine, whose unwind rule reads the
/// caller's frame from the record: with compiled code's (`unwind::Fde`), a
/// panic in any level reaches the thread's start.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn level_return() {
    core::arch::naked_asm!(
        ".cfi_startproc",
        ".cfi_def_cfa sp, {size}",
        ".cfi_offset x29, -{size}",
        ".cfi_offset x30, -{size} + 8",
        ".cfi_offset x19, -{size} + {x}",
        ".cfi_offset x20, -{size} + {x} + 8",
        ".cfi_offset x21, -{size} + {x} + 16",
        ".cfi_offset x22, -{size} + {x} + 24",
        ".cfi_offset x23, -{size} + {x} + 32",
        ".cfi_offset x24, -{size} + {x} + 40",
        ".cfi_offset x25, -{size} + {x} + 48",
        ".cfi_offset x26, -{size} + {x} + 56",
        ".cfi_offset x27, -{size} + {x} + 64",
        ".cfi_offset x28, -{size} + {x} + 72",
        "nop",
        "mov x9, sp",
        "ldr x25, [x9, #{x} + 48]",
        "ldr x10, [x9, #{prev}]",
        "ldr x11, [x9, #{deep_saved}]",
        "str x10, [x25, #{record}]",
        "str x11, [x25, #{deep}]",
        "ldp x19, x20, [x9, #{x}]",
        "ldp x21, x22, [x9, #{x} + 16]",
        "ldp x23, x24, [x9, #{x} + 32]",
        "ldp x25, x26, [x9, #{x} + 48]",
        "ldp x27, x28, [x9, #{x} + 64]",
        "ldp d8, d9, [x9, #{d}]",
        "ldp d10, d11, [x9, #{d} + 16]",
        "ldp d12, d13, [x9, #{d} + 32]",
        "ldp d14, d15, [x9, #{d} + 48]",
        "ldp x29, x30, [x9, #0]",
        "add sp, x9, #{size}",
        "ret",
        ".cfi_endproc",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        deep_saved = const crate::vm::stack::rec::DEEP,
        x = const crate::vm::stack::rec::X,
        d = const crate::vm::stack::rec::D,
        record = const crate::jit::layout::RECORD,
        deep = const crate::jit::layout::NATIVE_DEEP,
    )
}

/// The address handlers in a level return to: past `level_return`'s `nop`.
pub(crate) fn level_return_addr() -> usize {
    level_return as *const () as usize + 4
}

/// The continuation of a level's first frame: its results go back to the
/// compiled code that started the level, as a fast callee leaves them (the
/// first or nil in x0, the count in `insn`, the values in `pc`), with that
/// code's registers and the level below back.
#[unsafe(naked)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_to_native<'gc>(
    _nret: crate::vm::abi::Slot,
    _values: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "ldr x9, [x25, #{record}]",
        "mov sp, x9",
        "movz x0, #0xfffe, lsl #32",
        "movk x0, #0xffff, lsl #48",
        "cbz x20, 1f",
        "ldr x0, [x21]",
        "1:",
        "ldr x10, [x9, #{prev}]",
        "ldr x11, [x9, #{deep_saved}]",
        "str x10, [x25, #{record}]",
        "str x11, [x25, #{deep}]",
        "ldr x19, [x9, #{x}]",
        "ldp x22, x23, [x9, #{x} + 24]",
        "ldp x24, x25, [x9, #{x} + 40]",
        "ldp x26, x27, [x9, #{x} + 56]",
        "ldr x28, [x9, #{x} + 72]",
        "ldp d8, d9, [x9, #{d}]",
        "ldp d10, d11, [x9, #{d} + 16]",
        "ldp d12, d13, [x9, #{d} + 32]",
        "ldp d14, d15, [x9, #{d} + 48]",
        "ldp x29, x30, [x9, #0]",
        "add sp, x9, #{size}",
        "tbnz x30, #0, 2f",
        "ret",
        "2:",
        "and x30, x30, #~1",
        "br x30",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        deep_saved = const crate::vm::stack::rec::DEEP,
        x = const crate::vm::stack::rec::X,
        d = const crate::vm::stack::rec::D,
        record = const crate::jit::layout::RECORD,
        deep = const crate::jit::layout::NATIVE_DEEP,
    )
}

/// The continuation of a native's frame while `jit_call_native` runs it for
/// compiled code: it only marks the frame for walks, which find its caller
/// in `ThreadState::shim`. A native that does not return plainly gets
/// `ret_to_native` in its place before it goes on in a level.
#[unsafe(naked)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_shim<'gc>(
    _nret: crate::vm::abi::Slot,
    _values: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!("brk #0xf1")
}

/// Compiled code's call of a native (I1), with `jit_call_native`'s
/// arguments: records the caller's frame pointer and return address in the
/// thread (x25) for walks, and goes on in `jit_call_native`, which returns
/// to the caller.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn native_entry() {
    core::arch::naked_asm!(
        "stp x29, x30, [x25, #{shim}]",
        "b {body}",
        shim = const crate::jit::layout::SHIM,
        body = sym crate::jit::helpers::jit_call_native,
    )
}

/// Continue in the handler in `JitRuntime::jump_target` at the start of the
/// thread's current level: where the unwinder resets a thread to the level
/// of the frame that catches (H8).
#[unsafe(naked)]
pub(crate) extern "rust-preserve-none" fn level_jump<'gc>(
    _insn: crate::vm::abi::Slot,
    _pc: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "ldr x9, [x25, #{record}]",
        "mov sp, x9",
        "mov x29, x9",
        "ldr x30, [x23, #{level_ret}]",
        "ldr x16, [x23, #{target}]",
        "br x16",
        record = const crate::jit::layout::RECORD,
        level_ret = const crate::jit::layout::LEVEL_RET,
        target = const crate::jit::layout::JUMP_TARGET,
    )
}

/// The bottom level's start, from `dispatch::call_handler`: `jit_level`
/// for the handler `target`.
#[unsafe(naked)]
pub(crate) unsafe extern "rust-preserve-none" fn dispatch_entry<'gc>(
    _insn: crate::vm::abi::Slot,
    _pc: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
    _target: crate::vm::abi::Handler,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "mov x16, x26",
        "mov x17, #0",
        "b {jit_level}",
        jit_level = sym jit_level,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_order() {
        // `exit_common`'s stores, by image word.
        assert_eq!(image_index(preg_int(15)), 15);
        assert_eq!(image_index(preg_int(19)), 16);
        assert_eq!(image_index(preg_int(21)), 18);
        assert_eq!(image_index(preg_int(26)), 19);
        assert_eq!(image_index(preg_int(28)), 21);
        assert_eq!(image_index(preg_float(0)), 22);
        assert_eq!(image_index(preg_float(31)), 53);
    }
}
