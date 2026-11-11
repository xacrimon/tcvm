//! The target's side of code memory (`backend::alloc`): branch reach and
//! encodings for patching and islands. The processor keeps its instruction
//! cache coherent with stores, through either alias of a page.

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

/// Nothing to do: the processor tracks code writes by physical address. A
/// user-mode emulator (Rosetta, QEMU) translates the executable alias and
/// does not see a write through the other one: under `TCVM_JIT_EMULATED=1`
/// (`tcvm-dev`) it is told by a change of the pages' protection and back,
/// which never takes the execute permission away (other threads run the
/// islands' page).
pub(crate) unsafe fn sync_icache(_rw: *mut u8, rx: *mut u8, len: usize) {
    #[cfg(feature = "tcvm-dev")]
    {
        static EMULATED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = EMULATED
            .get_or_init(|| std::env::var_os("TCVM_JIT_EMULATED").is_some_and(|v| v != "0"));
        if *on {
            const PAGE: usize = 4096;
            const RX: i32 = libc::PROT_READ | libc::PROT_EXEC;
            let start = rx as usize & !(PAGE - 1);
            let len = (rx as usize + len).next_multiple_of(PAGE) - start;
            // SAFETY: pages of the executable alias.
            unsafe {
                libc::mprotect(start as *mut _, len, RX | libc::PROT_WRITE);
                libc::mprotect(start as *mut _, len, RX);
            }
        }
    }
    #[cfg(not(feature = "tcvm-dev"))]
    let _ = (rx, len);
}

/// Print `code`, placed at `base`, disassembled (`TCVM_JIT_DUMP=asm`), each
/// line tagged `@origin` after a `;` (`tags`: byte offset, origin, sorted).
#[cfg(feature = "tcvm-dev")]
pub(crate) fn disassemble(base: usize, code: &[u8], tags: &[(u32, String)]) {
    let tag = |line: &str| -> Option<&str> {
        let off = line.trim_start().split(':').next()?;
        let at = usize::from_str_radix(off, 16).ok()? as u32;
        let k = tags.partition_point(|t| t.0 <= at).checked_sub(1)?;
        Some(&tags[k].1)
    };
    // Through binutils: the bytes as a raw image.
    let dir = std::env::temp_dir().join(format!("tcvm-jit-dump-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let bin = dir.join("code.bin");
    let out = std::fs::write(&bin, code).ok().and_then(|()| {
        std::process::Command::new("objdump")
            .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel"])
            .arg("--no-show-raw-insn")
            .arg(&bin)
            .output()
            .ok()
    });
    match out {
        Some(o) if o.status.success() => {
            eprintln!("(offsets from {base:#x})");
            for line in String::from_utf8_lossy(&o.stdout).lines().skip(7) {
                match tag(line) {
                    Some(t) => eprintln!("{line}\t; @{t}"),
                    None => eprintln!("{line}"),
                }
            }
        }
        _ => {
            for (k, c) in code.chunks(16).enumerate() {
                let hex: Vec<String> = c.iter().map(|b| format!("{b:02x}")).collect();
                eprintln!("{:#x}: {}", base + k * 16, hex.join(" "));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::code::{in_reach, map_in, reserve, unmap_out};

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
            libc::munmap(rw.as_ptr().cast(), SIZE);
        }
        unmap_out(rx, SIZE);
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
