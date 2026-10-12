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
/// `image_index` order, copies the frame's spill slots after it, writes the
/// frames of compiled callers under it to their slots (`jit_leave`), returns
/// to dispatch's entry state (`NativeBase`) and tails `jit_exit`.
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
        // The callers' frames, from this one's record and base.
        "2:",
        "mov x20, x16",
        "mov x0, x23",
        "ldp x1, x2, [x29]",
        "mov x3, x22",
        "mov x4, #1",
        "bl {jit_leave}",
        // Back to dispatch's entry state, dropping this frame and any fast
        // callers' under it.
        "ldr x11, [x25, #{nsp}]",
        "mov sp, x11",
        "ldr x29, [x25, #{nfp}]",
        "ldr x30, [x25, #{nlr}]",
        "b {jit_exit}",
        regs = const crate::jit::layout::EXIT_REGS,
        num_spills = const crate::jit::region::layout::NUM_SPILLS,
        spills = const IMAGE_SPILLS * 8,
        nsp = const crate::jit::layout::NATIVE_SP,
        nfp = const crate::jit::layout::NATIVE_FP,
        nlr = const crate::jit::layout::NATIVE_LR,
        frame_spills = const crate::jit::backend::aarch64::emit::FRAME_SPILLS,
        jit_leave = sym crate::jit::jit_leave,
        jit_exit = sym crate::jit::jit_exit,
    )
}

/// Room below dispatch's entry for chains of fast calls; a fast entry past
/// it takes the slow path, which uses no native stack.
pub(crate) const FAST_STACK: usize = 256 * 1024;
/// Native stack below dispatch's entry past which fast calls branch with a
/// tagged return address instead of `blr`, and return with `br`: the return
/// address stack holds about 48 entries here, and a `ret` past it
/// mispredicts while the indirect predictor keeps up (5.4).
pub(crate) const RAS_STACK: usize = 512;

/// Dispatch's entry on a thread's stack, in place of the first handler:
/// records in the thread the stack pointer, frame pointer and return address
/// every handler runs with and the fast entries' limit (`NativeBase`), then
/// branches to the handler in `NativeBase::target` with the arguments as
/// they came.
#[unsafe(naked)]
pub(crate) unsafe extern "rust-preserve-none" fn dispatch_entry<'gc>(
    _insn: crate::vm::abi::Slot,
    _pc: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "mov x16, sp",
        "str x16, [x25, #{sp}]",
        "str x29, [x25, #{fp}]",
        "str x30, [x25, #{lr}]",
        "sub x17, x16, #{ras}",
        "str x17, [x25, #{deep}]",
        "sub x16, x16, #{budget}",
        "str x16, [x25, #{limit}]",
        "ldr x16, [x25, #{target}]",
        "br x16",
        sp = const crate::jit::layout::NATIVE_SP,
        fp = const crate::jit::layout::NATIVE_FP,
        lr = const crate::jit::layout::NATIVE_LR,
        limit = const crate::jit::layout::NATIVE_LIMIT,
        target = const crate::jit::layout::NATIVE_TARGET,
        budget = const FAST_STACK,
        ras = const RAS_STACK,
        deep = const crate::jit::layout::NATIVE_DEEP,
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
