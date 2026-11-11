//! The target's side of code memory (`backend::alloc`): branch reach and
//! encodings for patching and islands.

/// Reach of direct branches: code in `[a, a + len)` reaches every target in
/// `[lo, hi)` when `a + len - lo <= REACH` and `hi - a <= REACH` (`rel32`
/// counts from the end of its instruction).
pub(crate) const REACH: usize = (1 << 31) - 4096;

/// Sizes tried for the code range within reach of the text, largest first;
/// a far range is the first of them. One range serves the process: the
/// reach is wide enough for a reservation that never fills.
pub(crate) const RANGE_SIZES: [usize; 3] = [1 << 30, 256 << 20, 64 << 20];

/// Whether a `jmp`/`call rel32` at `from` reaches `to`.
pub(crate) fn reaches(from: usize, to: usize) -> bool {
    i32::try_from(to as i64 - (from as i64 + 5)).is_ok()
}

/// Bytes of the branch `retarget` rewrites: its opcode and `rel32`.
pub(crate) const BRANCH: usize = 5;

/// Repoint the `jmp` or `call` `inst`, at address `at`, to `to`, in reach.
pub(crate) fn retarget(inst: &mut [u8; BRANCH], at: usize, to: usize) {
    let rel = (to as i64 - (at as i64 + 5)) as i32;
    inst[1..].copy_from_slice(&rel.to_le_bytes());
}

/// A `call` at `at` to `to`, in reach.
pub(crate) fn call(at: usize, to: usize) -> [u8; BRANCH] {
    let mut b = [0xe8, 0, 0, 0, 0];
    retarget(&mut b, at, to);
    b
}

/// Bytes a patch point holds (`Asm::patch_point`).
pub(crate) const JUMP: usize = 5;

/// A `jmp` at `at` to `to`, in reach.
pub(crate) fn jump(at: usize, to: usize) -> [u8; JUMP] {
    let mut b = [0xe9, 0, 0, 0, 0];
    retarget(&mut b, at, to);
    b
}

/// Bytes of an island: `jmp [rip]`, the target, padding.
pub(crate) const ISLAND: usize = 16;

/// An island (`alloc::branch_target`): a branch on to `target` out of the
/// code range's reach, which takes no register.
pub(crate) fn island(target: usize) -> [u8; ISLAND] {
    let mut b = [0xcc; ISLAND];
    b[..6].copy_from_slice(&[0xff, 0x25, 0, 0, 0, 0]);
    b[6..14].copy_from_slice(&(target as u64).to_le_bytes());
    b
}

/// Code returning `imm` (tests).
#[cfg(test)]
pub(crate) fn ret_imm(imm: u16) -> [u8; 8] {
    let mut b = [0xcc; 8];
    b[0] = 0xb8; // mov eax, imm
    b[1..5].copy_from_slice(&(imm as u32).to_le_bytes());
    b[5] = 0xc3; // ret
    b
}

/// Nothing to do: code is written and fetched at one address, and the
/// processor keeps its instruction cache coherent with stores.
pub(crate) unsafe fn sync_icache(_rw: *mut u8, _rx: *mut u8, _len: usize) {}

/// GNU `objdump`'s listing of `code` (a raw image in `dir`), and its header
/// lines.
#[cfg(feature = "tcvm-dev")]
pub(crate) fn objdump(dir: &std::path::Path, code: &[u8]) -> Option<(std::process::Output, usize)> {
    let bin = dir.join("code.bin");
    std::fs::write(&bin, code).ok()?;
    let out = std::process::Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel"])
        .arg("--no-show-raw-insn")
        .arg(&bin)
        .output()
        .ok()?;
    Some((out, 7))
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
        assert_eq!(rw, rx);
        for imm in [42, 43] {
            unsafe {
                std::ptr::copy_nonoverlapping(ret_imm(imm).as_ptr(), rw.as_ptr(), 8);
                let f: extern "C" fn() -> u64 = std::mem::transmute(rx.as_ptr());
                assert_eq!(f(), imm as u64);
            }
        }
        unmap(rw, rx, SIZE);
        unsafe { libc::munmap(rx.as_ptr().cast(), SIZE) };
    }

    #[test]
    fn patches() {
        let mut call = [0xe8, 0, 0, 0, 0];
        retarget(&mut call, 0x1000, 0x2000);
        assert_eq!(call, [0xe8, 0xfb, 0x0f, 0, 0]);
        assert_eq!(jump(0x2000, 0x1000), [0xe9, 0xfb, 0xef, 0xff, 0xff]);
        assert!(reaches(0, (1 << 31) + 4));
        assert!(!reaches(0, (1 << 31) + 5));
    }
}
