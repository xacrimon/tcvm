//! Registers of a region (Appendix A), the allocator's machine environment,
//! the register image order, and the per-segment `exit_common`.

use regalloc2::{MachineEnv, PReg, PRegSet, RegClass};

use crate::jit::backend::aarch64::asm::{Asm, FP, Fpr, Gpr, LR, SP, Sz};
use crate::jit::state::EXIT_REGS;

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

/// Image word of the first spill slot.
pub(crate) const IMAGE_SPILLS: usize = 64;
const _: () = assert!(IMAGE_SPILLS + 64 <= EXIT_REGS);

/// `exit_common` at `rx`: entered from a region's exit trampoline with x16 =
/// region | exit << 48 and the region's frame still open; stores the image,
/// copies the spill slots, pops the frame and tails `jit_exit`.
pub(crate) fn exit_common(rx: usize) -> Vec<u32> {
    let mut a = Asm::new();
    let img = X17;
    a.ldr(img, RT, crate::jit::layout::EXIT_REGS as i32);
    for r in (0..16).step_by(2) {
        a.stp(Gpr(r), Gpr(r + 1), img, r as i32 * 8);
    }
    a.stp(Gpr(19), Gpr(20), img, 16 * 8);
    a.str(Gpr(21), img, 18 * 8);
    a.stp(Gpr(26), Gpr(27), img, 19 * 8);
    a.str(Gpr(28), img, 21 * 8);
    for r in (0..32).step_by(2) {
        a.stp_d(Fpr(r), Fpr(r + 1), img, (22 + r as i32) * 8);
    }
    // The spill slots, [x29 + 16 + 8i], to image words 64..
    let region = Gpr(9);
    a.and_imm(Sz::X, region, X16, (1 << 48) - 1);
    a.ldr_w(
        Gpr(10),
        region,
        crate::jit::region::layout::NUM_SPILLS as i32,
    );
    a.add_imm(Sz::X, Gpr(11), FP, 16);
    a.add_imm(Sz::X, Gpr(12), img, (IMAGE_SPILLS * 8) as u64);
    let done = a.new_label();
    let lp = a.new_label();
    a.cbz(Sz::W, Gpr(10), done);
    a.bind(lp);
    a.ldr(Gpr(13), Gpr(11), 0);
    a.add_imm(Sz::X, Gpr(11), Gpr(11), 8);
    a.str(Gpr(13), Gpr(12), 0);
    a.add_imm(Sz::X, Gpr(12), Gpr(12), 8);
    a.subs_imm(Sz::W, Gpr(10), Gpr(10), 1);
    a.b_cond(crate::jit::backend::aarch64::asm::Cond::Ne, lp);
    a.bind(done);
    // Pop the frame, whatever its size.
    a.ldr_w(
        Gpr(10),
        region,
        crate::jit::region::layout::FRAME_SIZE as i32,
    );
    a.ldp(Gpr(13), LR, FP, 0);
    a.add(Sz::X, Gpr(11), FP, Gpr(10));
    a.mov_sp(SP, Gpr(11));
    a.mov(FP, Gpr(13));
    a.mov(INSN, X16);
    a.b_far(crate::jit::jit_exit as *const () as usize);
    a.finish(rx).expect("exit_common reaches jit_exit")
}
