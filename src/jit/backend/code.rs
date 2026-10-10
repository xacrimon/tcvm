//! Executable memory: every page mapped twice, a read/write alias the JIT
//! writes through and a read/execute alias the CPU fetches from, so no page
//! is ever both. The executable alias is placed within direct-branch reach
//! of tcvm's text (as LuaJIT's `mcode_alloc`), so regions branch to handlers
//! and helpers with `b`/`bl`.
//!
//! After writing, the instruction cache must be invalidated over the
//! executable range (`sync_icache`): aarch64 does not keep it coherent.

use std::io;
use std::ptr::NonNull;
use std::sync::OnceLock;

/// Reach of `b`/`bl`, less a margin for code inside the segment.
const BRANCH_REACH: usize = (128 << 20) - (1 << 20);

/// Bounds of the executable text holding tcvm, which every branch target of a
/// region lies in.
pub(crate) fn text_range() -> (usize, usize) {
    static RANGE: OnceLock<(usize, usize)> = OnceLock::new();
    *RANGE.get_or_init(find_text)
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn find_text() -> (usize, usize) {
    let anchor = text_range as *const () as *const libc::c_void;
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let ok = unsafe { libc::dladdr(anchor, &mut info) };
    assert!(ok != 0, "dladdr failed on tcvm's own text");
    let base = info.dli_fbase as usize;
    let hdr = unsafe { &*(base as *const libc::mach_header_64) };
    let mut cmd = base + size_of::<libc::mach_header_64>();
    for _ in 0..hdr.ncmds {
        let lc = unsafe { &*(cmd as *const libc::load_command) };
        if lc.cmd == libc::LC_SEGMENT_64 {
            let seg = unsafe { &*(cmd as *const libc::segment_command_64) };
            let name: Vec<u8> = seg
                .segname
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            if name == b"__TEXT" {
                return (base, base + seg.vmsize as usize);
            }
        }
        cmd += lc.cmdsize as usize;
    }
    panic!("no __TEXT segment in tcvm's image");
}

#[cfg(target_os = "linux")]
fn find_text() -> (usize, usize) {
    struct Find {
        anchor: usize,
        found: Option<(usize, usize)>,
    }
    unsafe extern "C" fn cb(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        data: *mut libc::c_void,
    ) -> libc::c_int {
        let f = unsafe { &mut *(data as *mut Find) };
        let info = unsafe { &*info };
        let phdrs = unsafe { std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize) };
        for ph in phdrs {
            if ph.p_type == libc::PT_LOAD && ph.p_flags & libc::PF_X != 0 {
                let lo = info.dlpi_addr as usize + ph.p_vaddr as usize;
                let hi = lo + ph.p_memsz as usize;
                if (lo..hi).contains(&f.anchor) {
                    f.found = Some((lo, hi));
                    return 1;
                }
            }
        }
        0
    }
    let mut f = Find {
        anchor: text_range as *const () as usize,
        found: None,
    };
    unsafe { libc::dl_iterate_phdr(Some(cb), &mut f as *mut Find as *mut libc::c_void) };
    f.found.expect("tcvm's text segment")
}

/// Whether code at `[addr, addr + len)` reaches all of the text directly.
pub(crate) fn in_reach(addr: usize, len: usize) -> bool {
    let (lo, hi) = text_range();
    addr + len <= lo + BRANCH_REACH && addr + BRANCH_REACH >= hi
}

/// Candidate executable addresses for a mapping of `size`, nearest the text
/// first, alternating above and below, in `step` increments.
fn candidates(size: usize, step: usize) -> impl Iterator<Item = usize> {
    let (lo, hi) = text_range();
    let up = hi.next_multiple_of(step);
    let down = (lo.saturating_sub(size)) / step * step;
    let n = BRANCH_REACH / step;
    (0..n)
        .flat_map(move |i| {
            let a = up + i * step;
            let b = down.checked_sub(i * step);
            [Some(a), b].into_iter().flatten()
        })
        .filter(move |&a| a != 0 && in_reach(a, size))
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_vm_remap(
        target_task: libc::vm_map_t,
        target_address: *mut libc::mach_vm_address_t,
        size: libc::mach_vm_size_t,
        mask: libc::mach_vm_offset_t,
        flags: libc::c_int,
        src_task: libc::vm_map_t,
        src_address: libc::mach_vm_address_t,
        copy: libc::boolean_t,
        cur_protection: *mut libc::vm_prot_t,
        max_protection: *mut libc::vm_prot_t,
        inheritance: libc::vm_inherit_t,
    ) -> libc::kern_return_t;

    fn mach_vm_protect(
        target_task: libc::vm_map_t,
        address: libc::mach_vm_address_t,
        size: libc::mach_vm_size_t,
        set_maximum: libc::boolean_t,
        new_protection: libc::vm_prot_t,
    ) -> libc::kern_return_t;
}

#[cfg(target_os = "macos")]
const VM_INHERIT_NONE: libc::vm_inherit_t = 2;

/// Map `size` bytes twice: a read/write alias aligned to `align_mask + 1`, and
/// a read/execute alias within branch reach of the text. Returns `(rw, rx)`.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
pub(crate) fn map_dual(size: usize, align_mask: u64) -> io::Result<(NonNull<u8>, NonNull<u8>)> {
    let task = unsafe { libc::mach_task_self() };
    let mut rw_addr: libc::mach_vm_address_t = 0;
    let kr = unsafe {
        libc::mach_vm_map(
            task,
            &mut rw_addr,
            size as libc::mach_vm_size_t,
            align_mask as libc::mach_vm_offset_t,
            libc::VM_FLAGS_ANYWHERE,
            0,
            0,
            0,
            libc::VM_PROT_READ | libc::VM_PROT_WRITE,
            libc::VM_PROT_READ | libc::VM_PROT_WRITE | libc::VM_PROT_EXECUTE,
            VM_INHERIT_NONE,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return Err(io::Error::other(format!("mach_vm_map failed: {kr}")));
    }
    let unmap_rw = || unsafe {
        libc::munmap(rw_addr as *mut libc::c_void, size);
    };
    // The executable alias, at the first free address in reach.
    let mut rx_addr = None;
    for cand in candidates(size, 64 << 10) {
        let mut addr = cand as libc::mach_vm_address_t;
        let mut cur: libc::vm_prot_t = 0;
        let mut max: libc::vm_prot_t = 0;
        let kr = unsafe {
            mach_vm_remap(
                task,
                &mut addr,
                size as libc::mach_vm_size_t,
                0,
                libc::VM_FLAGS_FIXED,
                task,
                rw_addr,
                0,
                &mut cur,
                &mut max,
                VM_INHERIT_NONE,
            )
        };
        if kr == libc::KERN_SUCCESS {
            rx_addr = Some(addr);
            break;
        }
    }
    let Some(rx_addr) = rx_addr else {
        unmap_rw();
        return Err(io::Error::other(
            "no code address within branch reach of the text",
        ));
    };
    let kr = unsafe {
        mach_vm_protect(
            task,
            rx_addr,
            size as libc::mach_vm_size_t,
            0,
            libc::VM_PROT_READ | libc::VM_PROT_EXECUTE,
        )
    };
    if kr != libc::KERN_SUCCESS {
        unmap_rw();
        unsafe { libc::munmap(rx_addr as *mut libc::c_void, size) };
        return Err(io::Error::other(format!("mach_vm_protect failed: {kr}")));
    }
    Ok((
        NonNull::new(rw_addr as *mut u8).expect("mach_vm_map returned null"),
        NonNull::new(rx_addr as *mut u8).expect("mach_vm_remap returned null"),
    ))
}

/// Linux: two `MAP_SHARED` views of one `memfd`, the executable one placed
/// with `MAP_FIXED_NOREPLACE` in reach of the text.
#[cfg(target_os = "linux")]
pub(crate) fn map_dual(size: usize, align_mask: u64) -> io::Result<(NonNull<u8>, NonNull<u8>)> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let fd = unsafe { libc::memfd_create(c"tcvm-jit".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let raw = fd.as_raw_fd();
    if unsafe { libc::ftruncate(raw, size as libc::off_t) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let align = align_mask as usize + 1;
    let reserve = size + align;
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            reserve,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let base = base as usize;
    let aligned = (base + align_mask as usize) & !(align_mask as usize);
    let rw = unsafe {
        libc::mmap(
            aligned as *mut libc::c_void,
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_FIXED,
            raw,
            0,
        )
    };
    if rw == libc::MAP_FAILED {
        let err = io::Error::last_os_error();
        unsafe { libc::munmap(base as *mut libc::c_void, reserve) };
        return Err(err);
    }
    if aligned > base {
        unsafe { libc::munmap(base as *mut libc::c_void, aligned - base) };
    }
    let tail = aligned + size;
    unsafe { libc::munmap(tail as *mut libc::c_void, base + reserve - tail) };
    for cand in candidates(size, 64 << 10) {
        let rx = unsafe {
            libc::mmap(
                cand as *mut libc::c_void,
                size,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_SHARED | libc::MAP_FIXED_NOREPLACE,
                raw,
                0,
            )
        };
        if rx == libc::MAP_FAILED {
            continue;
        }
        if rx as usize != cand {
            unsafe { libc::munmap(rx, size) };
            continue;
        }
        return Ok((
            NonNull::new(rw.cast()).expect("mmap returned null"),
            NonNull::new(rx.cast()).expect("mmap returned null"),
        ));
    }
    unsafe { libc::munmap(rw, size) };
    Err(io::Error::other(
        "no code address within branch reach of the text",
    ))
}

/// Make bytes written through `rw` fetchable through `rx`. Apple Silicon has
/// `CTR_EL0.IDC = 1` (no data-cache clean needed) and a fixed 64-byte line,
/// and its `CTR_EL0` is not readable from EL0.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
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

#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn sync_icache(_rw: *mut u8, _rx: *mut u8, _len: usize) {
    unsafe { std::arch::asm!("mfence", options(nostack, preserves_flags)) };
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn maps_within_reach_and_runs() {
        let (rw, rx) = map_dual(1 << 16, (1 << 16) - 1).expect("map");
        assert!(in_reach(rx.as_ptr() as usize, 1 << 16));
        assert_eq!(rw.as_ptr() as usize % (1 << 16), 0);
        unsafe {
            let w = rw.as_ptr().cast::<u32>();
            w.write(0xD280_0540); // mov x0, #42
            w.add(1).write(0xD65F_03C0); // ret
            sync_icache(rw.as_ptr(), rx.as_ptr(), 8);
            let f: extern "C" fn() -> u64 = std::mem::transmute(rx.as_ptr());
            assert_eq!(f(), 42);
            libc::munmap(rw.as_ptr().cast(), 1 << 16);
            libc::munmap(rx.as_ptr().cast(), 1 << 16);
        }
    }
}
