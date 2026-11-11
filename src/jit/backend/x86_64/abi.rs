//! Registers of a region, the allocator's machine environment, the
//! register image order, and the stubs between compiled code, the
//! interpreter's handlers and levels.
//!
//! Handlers run with the level's record above `rsp` and `level_return`'s
//! address pushed below it, which the bottom level's last handler returns
//! to. Compiled code never tags a return address (`RA_TAG`): every call is a
//! `call` and every return a `ret`.

use crate::jit::backend::reg::{PReg, PRegSet, RegClass};
use crate::jit::backend::x86_64::asm::{
    Gpr, R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBX, RCX, RDI, RDX, RSI, Xmm,
};

/// The bit of a return address that marks a return by branch: none here.
pub(crate) const RA_TAG: usize = 0;

/// Words of the register image `exit_common` writes: the allocatable
/// integer registers in `INT_IMAGE` order, xmm0-xmm15, then the region's
/// spill slots from word `IMAGE_SPILLS`.
pub(crate) const EXIT_REGS: usize = IMAGE_SPILLS + MAX_SPILLS;
/// Image word of the first spill slot.
pub(crate) const IMAGE_SPILLS: usize = 32;

/// Pinned for a region's whole body.
pub(crate) const BASE: Gpr = R14;
pub(crate) const THREAD: Gpr = RBX;
/// The handler ABI's slots besides `base`: allocatable in a region, set
/// where code goes on into a handler or a stub that reads them. The runtime
/// is a constant of the region; a level's record keeps it (`jit_level`),
/// where stubs without it find it.
pub(crate) const INSN: Gpr = R12;
pub(crate) const PC: Gpr = R13;
pub(crate) const RT: Gpr = R15;
pub(crate) const CLOSURE: Gpr = RDI;
/// Scratch inside one instruction; never allocated.
pub(crate) const S0: Gpr = R11;
pub(crate) const S1: Gpr = R10;
pub(crate) const FS: Xmm = Xmm(15);

/// The registers a call passes its first arguments in, and a function
/// region takes its parameters in: the C ABI's.
pub(crate) const ARGS: [Gpr; 6] = [RDI, RSI, RDX, RCX, R8, R9];
/// How many arguments a call passes in registers.
pub(crate) const REG_ARGS: u8 = ARGS.len() as u8;
/// The first and second result of a call.
pub(crate) const RESULTS: [Gpr; 2] = [RAX, RDX];

/// Allocatable integer registers, the ones calls and helpers fix last
/// among the caller-saved: a value with no hint taking one would stand in
/// the way of a fixed operand. Their order is the register image's.
const INT_PREFERRED: [Gpr; 7] = [R9, R8, RAX, RCX, RDX, RSI, RDI];
const INT_OTHER: [Gpr; 3] = [R12, R13, R15];
const INT_IMAGE: [Gpr; 10] = [RAX, RCX, RDX, RSI, RDI, R8, R9, R12, R13, R15];
const FLOATS: u8 = 15;

/// The allocator's registers: its classes' allocatable ones, preferred
/// first; r10 and xmm15 park a value in a cycle of moves, r11 carries
/// stack-to-stack moves.
pub(crate) fn ra_env() -> crate::jit::backend::ra::Env {
    crate::jit::backend::ra::Env {
        regs: [
            (INT_PREFERRED.iter().chain(&INT_OTHER))
                .map(|&r| preg_int(r))
                .collect(),
            (8..FLOATS)
                .chain(0..8)
                .map(|r| preg_float(Xmm(r)))
                .collect(),
        ],
        scratch: [preg_int(S1), preg_float(FS)],
        transport: preg_int(S0),
    }
}

/// Every register the allocator hands out: what a call into another region
/// may change.
pub(crate) fn allocatable() -> PRegSet {
    let mut s = PRegSet::empty();
    for &r in &INT_IMAGE {
        s.add(preg_int(r));
    }
    for r in 0..FLOATS {
        s.add(preg_float(Xmm(r)));
    }
    s
}

/// What a C-ABI call clobbers of the allocatable registers: every float
/// one, and the integer ones but r12, r13 and r15.
pub(crate) fn caller_saved() -> PRegSet {
    let mut s = allocatable();
    for r in INT_OTHER {
        s.remove(preg_int(r));
    }
    s
}

pub(crate) fn preg_int(r: Gpr) -> PReg {
    PReg::new(r.0 as usize, RegClass::Int)
}

pub(crate) fn preg_float(r: Xmm) -> PReg {
    PReg::new(r.0 as usize, RegClass::Float)
}

/// The register image slot `exit_common` stores a register in.
pub(crate) fn image_index(p: PReg) -> u8 {
    let n = p.hw_enc() as u8;
    match p.class() {
        RegClass::Int => (INT_IMAGE.iter())
            .position(|r| r.0 == n)
            .unwrap_or_else(|| panic!("r{n} is not allocatable")) as u8,
        RegClass::Float => INT_IMAGE.len() as u8 + n,
        RegClass::Vector => unreachable!(),
    }
}

/// The spill slots a region may have, all copied to the image.
pub(crate) const MAX_SPILLS: usize = 256;

/// The address of spill slot `i` of the compiled frame whose frame pointer
/// is `fp`: below the frame record.
#[inline]
pub(crate) fn spill_addr(fp: usize, i: usize) -> usize {
    fp - 8 * (i + 1)
}

const PTR_SHIFT: usize = 16;

/// Entered from a region's exit trampoline with r11 = region | exit << 48
/// and the region's frame still open: stores the register image in
/// `image_index` order, copies the frame's spill slots after it, and pops
/// the frame. A frame dispatch entered tails `jit_exit` in its level; a
/// fast-called one starts a level that returns to its caller, and the
/// interpreter runs the rest of it there. Every allocatable register may
/// hold a value: the runtime comes from the level's record.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn exit_common() {
    core::arch::naked_asm!(
        "mov r10, [rbx + {record}]",
        "mov r10, [r10 + {rec_rt}]",
        "mov r10, [r10 + {regs}]",
        "mov [r10], rax",
        "mov [r10 + 8], rcx",
        "mov [r10 + 16], rdx",
        "mov [r10 + 24], rsi",
        "mov [r10 + 32], rdi",
        "mov [r10 + 40], r8",
        "mov [r10 + 48], r9",
        "mov [r10 + 56], r12",
        "mov [r10 + 64], r13",
        "mov [r10 + 72], r15",
        "vmovsd [r10 + 80], xmm0",
        "vmovsd [r10 + 88], xmm1",
        "vmovsd [r10 + 96], xmm2",
        "vmovsd [r10 + 104], xmm3",
        "vmovsd [r10 + 112], xmm4",
        "vmovsd [r10 + 120], xmm5",
        "vmovsd [r10 + 128], xmm6",
        "vmovsd [r10 + 136], xmm7",
        "vmovsd [r10 + 144], xmm8",
        "vmovsd [r10 + 152], xmm9",
        "vmovsd [r10 + 160], xmm10",
        "vmovsd [r10 + 168], xmm11",
        "vmovsd [r10 + 176], xmm12",
        "vmovsd [r10 + 184], xmm13",
        "vmovsd [r10 + 192], xmm14",
        "vmovsd [r10 + 200], xmm15",
        // The spill slots, [rbp - 8 - 8i], to the image's spill words.
        "mov rax, r11",
        "shl rax, {ptr_shift}",
        "shr rax, {ptr_shift}",
        "mov ecx, [rax + {num_spills}]",
        "lea rdx, [rbp - 8]",
        "lea rsi, [r10 + {spills}]",
        "test ecx, ecx",
        "jz 3f",
        "2:",
        "mov rdi, [rdx]",
        "mov [rsi], rdi",
        "sub rdx, 8",
        "add rsi, 8",
        "dec ecx",
        "jnz 2b",
        // Pop the frame: its return address is on top again.
        "3:",
        "mov rsp, rbp",
        "pop rbp",
        "mov r12, r11",
        "mov r15, [rbx + {record}]",
        "mov r15, [r15 + {rec_rt}]",
        "mov rdi, [r14 - 32]",
        "shl rdi, {ptr_shift}",
        "shr rdi, {ptr_shift}",
        "mov rax, [r15 + {level_land}]",
        "cmp [rsp], rax",
        "je 4f",
        "lea r11, [rip + {jit_exit}]",
        "mov r10, r14",
        "jmp {jit_level}",
        "4:",
        "mov rax, [r15 + {level_ret}]",
        "mov [rsp], rax",
        "jmp {jit_exit}",
        regs = const crate::jit::layout::EXIT_REGS,
        record = const crate::jit::layout::RECORD,
        rec_rt = const crate::vm::stack::rec::RT,
        ptr_shift = const PTR_SHIFT,
        num_spills = const crate::jit::region::layout::NUM_SPILLS,
        spills = const IMAGE_SPILLS * 8,
        level_ret = const crate::jit::layout::LEVEL_RET,
        level_land = const crate::jit::layout::LEVEL_LAND,
        jit_level = sym jit_level,
        jit_exit = sym crate::jit::jit_exit,
    )
}

/// Start a level for compiled code: entered by `call` or `jmp` with the
/// return address on the stack, the handler to run in r11, the base of the
/// level's first frame in r10 and the handler's arguments in their
/// registers. Pushes an entry record keeping `BASE` and the runtime, as the
/// call clobbers every allocatable register, makes it the thread's, and
/// branches to the handler with `level_return` to return to. The level ends
/// in `ret_to_native`.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn jit_level() {
    core::arch::naked_asm!(
        "push rbp",
        "sub rsp, {size} - 16",
        "mov [rsp + {first}], r10",
        "mov r10, [rbx + {record}]",
        "mov [rsp + {prev}], r10",
        "mov [rsp + {base}], r14",
        "mov [rsp + {rt}], r15",
        "lea rbp, [rsp + {size} - 16]",
        "mov [rbx + {record}], rsp",
        "push qword ptr [r15 + {level_ret}]",
        "jmp r11",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        first = const crate::vm::stack::rec::FIRST,
        base = const crate::vm::stack::rec::BASE,
        rt = const crate::vm::stack::rec::RT,
        record = const crate::jit::layout::RECORD,
        level_ret = const crate::jit::layout::LEVEL_RET,
    )
}

/// The unwind rules of code running with `rsp` at a level's record
/// (`vm::stack::rec`, `{size}` bytes): the caller's frame is read from it.
/// The callee-saved registers are left as they are: compiled frames hold
/// nothing in them, and the Rust caller of the bottom level, which calls
/// with every register clobbered, restores its own.
macro_rules! level_record_cfi {
    () => {
        concat!(
            ".cfi_startproc\n",
            ".cfi_def_cfa rsp, {size}\n",
            ".cfi_offset rbp, -16\n",
        )
    };
}

/// Where the bottom level's last handler returns: pops its record and
/// returns to `dispatch_entry`'s caller. Handlers return a byte in
/// (`level_return_addr`), so an unwinder looking up the return address less
/// one finds this routine, whose unwind rule reads the caller's frame from
/// the record: with compiled code's (`unwind::Fde`), a panic in any level
/// reaches the thread's start.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn level_return() {
    core::arch::naked_asm!(
        level_record_cfi!(),
        "nop",
        "mov rax, [rsp + {rt}]",
        "mov rax, [rax + {thread}]",
        "mov rcx, [rsp + {prev}]",
        "mov [rax + {record}], rcx",
        "add rsp, {size} - 16",
        "pop rbp",
        "ret",
        ".cfi_endproc",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        rt = const crate::vm::stack::rec::RT,
        thread = const crate::jit::layout::RT_THREAD,
        record = const crate::jit::layout::RECORD,
    )
}

/// Where a fast entry's failed checks go, and a direct tail call while its
/// callee's prototype has no fast entry (`jit::tail_miss`). A tail call
/// from a frame dispatch entered has its landing for a return address:
/// `enter` runs the call from the header the tail call wrote (its closure
/// there), in the level. Otherwise on to the caller's slow path.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn direct_miss() {
    core::arch::naked_asm!(
        "mov r10, [rbx + {record}]",
        "mov r10, [r10 + {rec_rt}]",
        "mov r11, [r10 + {level_land}]",
        "cmp [rsp], r11",
        "je 2f",
        "jmp {call_miss}",
        "2:",
        "mov r11, [r14 - 32]",
        "shl r11, {ptr_shift}",
        "shr r11, {ptr_shift}",
        "mov r15, {tag}",
        "shl r15, 48",
        "or r11, r15",
        "mov [r14 - 32], r11",
        "lea r13, [r14 - 32]",
        "xor r14d, r14d",
        "mov r15, r10",
        "mov r11, [r15 + {level_ret}]",
        "mov [rsp], r11",
        "jmp {enter}",
        record = const crate::jit::layout::RECORD,
        rec_rt = const crate::vm::stack::rec::RT,
        ptr_shift = const PTR_SHIFT,
        level_land = const crate::jit::layout::LEVEL_LAND,
        level_ret = const crate::jit::layout::LEVEL_RET,
        tag = const crate::jit::layout::FUNCTION_TAG,
        call_miss = sym call_miss,
        enter = sym crate::vm::ops::call::enter,
    )
}

/// A fast callee's entry checks failed: on to its caller's slow path, which
/// the caller's site table names by the return address on the stack, the
/// arguments a register call passed kept for it to store. The slow path is
/// reached by returning to it, so the return address stack stays in step.
/// The pinned registers and `insn` are callee-saved.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn call_miss() {
    core::arch::naked_asm!(
        "push rdi",
        "push rsi",
        "push rdx",
        "push rcx",
        "push r8",
        "push r9",
        "push rax",
        "mov rdi, [rsp + 56]",
        "call {target}",
        "mov [rsp + 56], rax",
        "pop rax",
        "pop r9",
        "pop r8",
        "pop rcx",
        "pop rdx",
        "pop rsi",
        "pop rdi",
        "ret",
        target = sym crate::jit::jit_call_miss,
    )
}

/// The address handlers in a level return to: past `level_return`'s `nop`.
pub(crate) fn level_return_addr() -> usize {
    level_return as *const () as usize + 1
}

/// Enter the region whose entry is in r11 from dispatch, `rsp` at the
/// level's record (the region's dispatch entry dropped `level_return`'s
/// address), its return address `level_land`, so the region returns with a
/// plain `ret`. The unwind rule is `level_return`'s: an unwinder at a
/// return address of `level_land` reads the caller's frame from the record.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn level_enter() {
    core::arch::naked_asm!(
        level_record_cfi!(),
        "call r11",
        // A compiled callee returns its one result in rax alone.
        "test r12, r12",
        "jz 2f",
        "mov [r13], rax",
        "2:",
        "mov r15, [rsp + {rt}]",
        "mov rdi, [r14 - 32]",
        "shl rdi, {ptr_shift}",
        "shr rdi, {ptr_shift}",
        "mov rax, [r14 - 24]",
        "and rax, -32",
        "push qword ptr [r15 + {level_ret}]",
        "jmp rax",
        ".cfi_endproc",
        size = const crate::vm::stack::rec::SIZE,
        rt = const crate::vm::stack::rec::RT,
        ptr_shift = const PTR_SHIFT,
        level_ret = const crate::jit::layout::LEVEL_RET,
    )
}

/// Where a region dispatch entered returns: on in its level through the
/// frame's continuation, as an interpreted callee returns (the results'
/// count in `insn`, the values at `pc`), the handler ABI's runtime and
/// closure back. Past `level_enter`'s three-byte `call r11`.
pub(crate) fn level_land() -> usize {
    level_enter as *const () as usize + 3
}

/// The continuation of a level's first frame: its results go back to the
/// compiled code that started the level, as a fast callee leaves them (the
/// first or nil in rax, the count in `insn`, the values in `pc`), with that
/// code's pinned registers and the level below back.
#[unsafe(naked)]
#[rustc_align(32)]
pub(crate) extern "rust-preserve-none" fn ret_to_native<'gc>(
    _nret: crate::vm::abi::Slot,
    _values: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "mov rbx, [r15 + {thread}]",
        "mov rsp, [rbx + {record}]",
        "mov rax, {nil}",
        "test r12, r12",
        "jz 2f",
        "mov rax, [r13]",
        "2:",
        "mov r10, [rsp + {prev}]",
        "mov [rbx + {record}], r10",
        "mov r14, [rsp + {base}]",
        "add rsp, {size} - 16",
        "pop rbp",
        "ret",
        size = const crate::vm::stack::rec::SIZE,
        prev = const crate::vm::stack::rec::PREV,
        base = const crate::vm::stack::rec::BASE,
        thread = const crate::jit::layout::RT_THREAD,
        record = const crate::jit::layout::RECORD,
        nil = const crate::env::value::NIL,
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
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!("ud2")
}

/// Compiled code's call of a native, with `jit_call_native`'s arguments:
/// records the caller's frame pointer and return address in the thread for
/// walks, and goes on in `jit_call_native`, which returns to the caller.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn native_entry() {
    core::arch::naked_asm!(
        "mov [rbx + {shim}], rbp",
        "mov rax, [rsp]",
        "mov [rbx + {shim} + 8], rax",
        "jmp {body}",
        shim = const crate::jit::layout::SHIM,
        body = sym crate::jit::helpers::jit_call_native,
    )
}

/// Compiled code's call of a coroutine builtin (`coro::Switch`), with
/// `jit_call_native`'s arguments, recording the caller as `native_entry`
/// does. The switch function either ran the native, whose result goes back,
/// or names the thread to switch to: this one then waits at its switch
/// point, `BASE` and the runtime in the frame, and when switched back to
/// returns what the landing function makes of it. Switches are branches,
/// and the stub is entered and returns with branches too (its return
/// address pushed by `native_path`), so the return address stack sees only
/// the Rust calls' pairs: a return after a switch would find the other
/// thread's call on top.
macro_rules! coro_stub {
    ($(#[$doc:meta])* $name:ident, $switch:path, $landed:path) => {
        $(#[$doc])*
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $name() {
            core::arch::naked_asm!(
                ".cfi_startproc",
                "mov [rbx + {shim}], rbp",
                "mov rax, [rsp]",
                "mov [rbx + {shim} + 8], rax",
                "push rbp",
                "push r14",
                "push rdi",
                ".cfi_def_cfa_offset 32",
                ".cfi_offset rbp, -16",
                ".cfi_offset r14, -24",
                "call {switch}",
                "cmp rdx, -2",
                "je 2f",
                "add rsp, 8",
                "pop r14",
                "pop rbp",
                "pop r10",
                "jmp r10",
                // Wait to continue at 3, and switch to the thread in rax,
                // whose fresh stack starts with the runtime in r15
                // (`thread_entry`).
                "2:",
                "lea r10, [rip + 3f]",
                "mov [rbx + {tsw} + {sp}], rsp",
                "mov [rbx + {tsw} + {cont}], r10",
                "mov [rbx + {tsw} + {fp}], rbp",
                "mov r15, [rsp]",
                "mov r11, rax",
                "mov rsp, [r11 + {tsw} + {sp}]",
                "mov rbp, [r11 + {tsw} + {fp}]",
                "mov r10, [r11 + {tsw} + {cont}]",
                "xor eax, eax",
                "jmp r10",
                "3:",
                "mov rdi, [rsp]",
                "mov rbx, [rdi + {thread}]",
                "call {landed}",
                "add rsp, 8",
                "pop r14",
                "pop rbp",
                "pop r10",
                "jmp r10",
                ".cfi_endproc",
                shim = const crate::jit::layout::SHIM,
                thread = const crate::jit::layout::RT_THREAD,
                tsw = const crate::vm::stack::sw::THREAD,
                sp = const crate::vm::stack::sw::SP,
                cont = const crate::vm::stack::sw::CONT,
                fp = const crate::vm::stack::sw::FP,
                switch = sym $switch,
                landed = sym $landed,
            )
        }
    };
}

coro_stub!(
    /// `coroutine.resume`.
    coro_resume,
    crate::jit::helpers::jit_resume_switch,
    crate::jit::helpers::jit_resume_landed
);
coro_stub!(
    /// A `coroutine.wrap` function.
    coro_wrap,
    crate::jit::helpers::jit_wrap_switch,
    crate::jit::helpers::jit_coro_landed
);
coro_stub!(
    /// `coroutine.yield`.
    coro_yield,
    crate::jit::helpers::jit_yield_switch,
    crate::jit::helpers::jit_coro_landed
);

/// Continue in the handler in `JitRuntime::jump_target` at the start of the
/// thread's current level: where the unwinder resets a thread to the level
/// of the frame that catches.
#[unsafe(naked)]
pub(crate) extern "rust-preserve-none" fn level_jump<'gc>(
    _insn: crate::vm::abi::Slot,
    _pc: *const crate::instruction::Instruction,
    _base: *mut crate::env::value::Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _closure: crate::vm::abi::Slot,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "mov rax, [r15 + {thread}]",
        "mov rsp, [rax + {record}]",
        "lea rbp, [rsp + {size} - 16]",
        "push qword ptr [r15 + {level_ret}]",
        "jmp qword ptr [r15 + {target}]",
        size = const crate::vm::stack::rec::SIZE,
        thread = const crate::jit::layout::RT_THREAD,
        record = const crate::jit::layout::RECORD,
        level_ret = const crate::jit::layout::LEVEL_RET,
        target = const crate::jit::layout::JUMP_TARGET,
    )
}

/// The bottom level's start, from `dispatch::call_handler`: `jit_level`
/// for the handler `target`, with the thread where compiled code keeps it.
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
        "mov rbx, rsi",
        "mov r11, rdx",
        "xor r10d, r10d",
        "jmp {jit_level}",
        jit_level = sym jit_level,
    )
}

/// The native stack pointer.
#[inline(always)]
pub(crate) fn stack_pointer() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack, preserves_flags))
    };
    sp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_order() {
        // `exit_common`'s stores, by image word.
        assert_eq!(image_index(preg_int(RAX)), 0);
        assert_eq!(image_index(preg_int(RSI)), 3);
        assert_eq!(image_index(preg_int(R9)), 6);
        assert_eq!(image_index(preg_int(R12)), 7);
        assert_eq!(image_index(preg_int(R15)), 9);
        assert_eq!(image_index(preg_float(Xmm(0))), 10);
        assert_eq!(image_index(preg_float(Xmm(15))), 25);
    }

    #[test]
    fn level_land_follows_the_call() {
        // `call r11`.
        let code = unsafe { std::slice::from_raw_parts(level_enter as *const u8, 3) };
        assert_eq!(code, [0x41, 0xff, 0xd3]);
    }
}
