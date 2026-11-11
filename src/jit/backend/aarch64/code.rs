//! The target's side of code memory (`backend::alloc`): branch reach and
//! encodings for patching and islands, and making written code fetchable,
//! as aarch64 does not keep the instruction cache coherent with stores.

use crate::jit::backend::aarch64::abi::{exit_common, jit_level, level_enter};

/// Reach of direct branches: code in `[a, a + len)` reaches every target in
/// `[lo, hi)` when `a + len - lo <= REACH` and `hi - a <= REACH` (`b`/`bl`
/// reach `[-2^27, 2^27)` from the instruction).
pub(crate) const REACH: usize = 128 << 20;

/// Sizes tried for the code range within reach of the text, largest first;
/// a far range is the first of them.
pub(crate) const RANGE_SIZES: [usize; 3] = [64 << 20, 32 << 20, 16 << 20];

/// Whether a `b`/`bl` at `from` reaches `to`.
pub(crate) fn reaches(from: usize, to: usize) -> bool {
    let d = to as i64 - from as i64;
    d % 4 == 0 && (-(1 << 27)..(1 << 27)).contains(&d)
}

/// Bytes of the branch `retarget` rewrites.
pub(crate) const BRANCH: usize = 4;

/// Repoint the `b` or `bl` `inst`, at address `at`, to `to`, in reach.
pub(crate) fn retarget(inst: &mut [u8; BRANCH], at: usize, to: usize) {
    debug_assert!(at.is_multiple_of(4));
    let d = to as i64 - at as i64;
    let w = u32::from_le_bytes(*inst) & 0xfc00_0000 | (d >> 2) as u32 & 0x03ff_ffff;
    *inst = w.to_le_bytes();
}

/// A `bl` at `at` to `to`, in reach.
pub(crate) fn call(at: usize, to: usize) -> [u8; BRANCH] {
    let d = (to as i64 - at as i64) >> 2;
    (0x9400_0000 | d as u32 & 0x03ff_ffff).to_le_bytes()
}

/// Bytes a patch point holds (`Asm::patch_point`).
pub(crate) const JUMP: usize = 4;

/// A `b` at `at` to `to`, in reach.
pub(crate) fn jump(at: usize, to: usize) -> [u8; JUMP] {
    let d = (to as i64 - at as i64) >> 2;
    (0x1400_0000 | d as u32 & 0x03ff_ffff).to_le_bytes()
}

/// Bytes of an island: `ldr` of its target, `br`, the target.
pub(crate) const ISLAND: usize = 16;

/// An island (`alloc::branch_target`): a branch on to `target` out of the
/// code range's reach, through a register free at the target's entry,
/// which the caller's `b`/`bl` left everything else for.
pub(crate) fn island(target: usize) -> [u8; ISLAND] {
    // x16 is free at entry to Rust code (IP0); the stubs taking arguments
    // in it, and `jit_level` in x17 too, have their own.
    let s: u32 = if target == exit_common as *const () as usize
        || target == level_enter as *const () as usize
    {
        17
    } else if target == jit_level as *const () as usize {
        9
    } else {
        16
    };
    let mut b = [0; ISLAND];
    b[0..4].copy_from_slice(&(0x5800_0040 | s).to_le_bytes()); // ldr xs, #8
    b[4..8].copy_from_slice(&(0xD61F_0000 | s << 5).to_le_bytes()); // br xs
    b[8..].copy_from_slice(&(target as u64).to_le_bytes());
    b
}

/// Code returning `imm` (tests).
#[cfg(test)]
pub(crate) fn ret_imm(imm: u16) -> [u8; 8] {
    let mut b = [0; 8];
    b[0..4].copy_from_slice(&(0xD280_0000 | (imm as u32) << 5).to_le_bytes()); // mov x0, #imm
    b[4..].copy_from_slice(&0xD65F_03C0u32.to_le_bytes()); // ret
    b
}

/// Make bytes written through `rw` fetchable through `rx`. Apple Silicon has
/// `CTR_EL0.IDC = 1` (no data-cache clean needed) and a fixed 64-byte line,
/// and its `CTR_EL0` is not readable from EL0.
#[cfg(target_os = "macos")]
pub(crate) unsafe fn sync_icache(_rw: *mut u8, rx: *mut u8, len: usize) {
    use std::arch::asm;
    const LINE: usize = 64;
    let mut p = rx as usize & !(LINE - 1);
    let end = rx as usize + len;
    while p < end {
        unsafe { asm!("ic ivau, {}", in(reg) p, options(nostack, preserves_flags)) };
        p += LINE;
    }
    unsafe {
        asm!("dsb ish", options(nostack, preserves_flags));
        asm!("isb", options(nostack, preserves_flags));
    }
}

#[cfg(target_os = "linux")]
pub(crate) unsafe fn sync_icache(rw: *mut u8, rx: *mut u8, len: usize) {
    use std::arch::asm;
    let ctr: u64;
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) };
    let dline = 4usize << ((ctr >> 16) & 0xf);
    let iline = 4usize << (ctr & 0xf);
    let idc = (ctr >> 28) & 1 != 0;
    let dic = (ctr >> 29) & 1 != 0;
    if !idc {
        let mut p = rw as usize & !(dline - 1);
        let end = rw as usize + len;
        while p < end {
            unsafe { asm!("dc cvau, {}", in(reg) p, options(nostack, preserves_flags)) };
            p += dline;
        }
        unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };
    }
    if !dic {
        let mut p = rx as usize & !(iline - 1);
        let end = rx as usize + len;
        while p < end {
            unsafe { asm!("ic ivau, {}", in(reg) p, options(nostack, preserves_flags)) };
            p += iline;
        }
        unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };
    }
    unsafe { asm!("isb", options(nostack, preserves_flags)) };
}

/// `objdump`'s listing of `code`, through the system assembler (the words
/// wrapped in an object, in `dir`), and its header lines.
#[cfg(feature = "tcvm-dev")]
pub(crate) fn objdump(dir: &std::path::Path, code: &[u8]) -> Option<(std::process::Output, usize)> {
    let (src, obj) = (dir.join("code.s"), dir.join("code.o"));
    let mut text = String::from(".text\n");
    for c in code.as_chunks::<4>().0 {
        text.push_str(&format!(".inst {:#010x}\n", u32::from_le_bytes(*c)));
    }
    std::fs::write(&src, text).ok()?;
    let built = std::process::Command::new("clang")
        .args(["-c", "-arch", "arm64", "-o"])
        .arg(&obj)
        .arg(&src)
        .status()
        .ok()?;
    if !built.success() {
        return None;
    }
    let out = std::process::Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(&obj)
        .output()
        .ok()?;
    Some((out, 6))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::code::{in_reach, map_in, reserve, unmap};

    #[test]
    fn maps_in_a_reservation_and_runs() {
        const SIZE: usize = 1 << 16;
        let rx = match reserve(SIZE, SIZE, true, false) {
            Ok(p) => {
                assert!(in_reach(p.as_ptr() as usize, SIZE));
                p
            }
            // The window can be full (other tests' heaps), as the code range
            // expects too.
            Err(_) => reserve(SIZE, SIZE, false, false).expect("reserve"),
        };
        let rw = map_in(rx, SIZE).expect("map");
        assert_eq!(rw.as_ptr() as usize % SIZE, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(ret_imm(42).as_ptr(), rw.as_ptr(), 8);
            sync_icache(rw.as_ptr(), rx.as_ptr(), 8);
            let f: extern "C" fn() -> u64 = std::mem::transmute(rx.as_ptr());
            assert_eq!(f(), 42);
        }
        unmap(rw, rx, SIZE);
        unsafe { libc::munmap(rx.as_ptr().cast(), SIZE) };
    }
}
