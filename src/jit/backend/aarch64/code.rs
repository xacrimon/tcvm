//! Making written code fetchable: aarch64 does not keep the instruction
//! cache coherent with stores.

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
            let w = rw.as_ptr().cast::<u32>();
            w.write(0xD280_0540); // mov x0, #42
            w.add(1).write(0xD65F_03C0); // ret
            sync_icache(rw.as_ptr(), rx.as_ptr(), 8);
            let f: extern "C" fn() -> u64 = std::mem::transmute(rx.as_ptr());
            assert_eq!(f(), 42);
            libc::munmap(rw.as_ptr().cast(), SIZE);
        }
        unmap_out(rx, SIZE);
        unsafe { libc::munmap(rx.as_ptr().cast(), SIZE) };
    }
}
