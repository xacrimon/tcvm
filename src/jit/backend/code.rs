//! Executable code memory.
//!
//! Apple Silicon enforces W^X, so a single page cannot be writable and
//! executable at once. Rather than flip a page's protection between the two
//! states — a per-thread `pthread_jit_write_protect_np` toggle, or worse an
//! `mprotect` syscall — this maps the same physical pages *twice*: one alias is
//! read/write, the other read/execute. Writes go through the RW alias, the CPU
//! fetches through the RX alias, and neither page is ever both. The split is set
//! up once at buffer creation (`mach_vm_remap` aliases an anonymous RW mapping,
//! then `mach_vm_protect` raises the alias to R+X), so writing costs nothing at
//! the page-table level and the executable alias never loses execute permission.
//!
//! Aliasing is not enough on its own. After writing through the RW alias the
//! instruction cache does not know the bytes changed; on aarch64 it is not
//! coherent with the data side. `finalize` invalidates the instruction cache
//! (`ic ivau`) over the affected lines. Skipping this buys a crash that
//! reproduces once every few hundred runs.
//!
//! Two notes on the line size and the (absent) clean step, both settled by
//! disassembling libplatform's `sys_icache_invalidate` (the routine this
//! replaces):
//!   * The maintenance line size is a fixed 64 bytes. `CTR_EL0`, which reports
//!     it, is not readable from EL0 on Darwin — `mrs x, ctr_el0` faults with
//!     SIGILL — so libplatform hardcodes 64 (the Apple Silicon minimum line) and
//!     so do we.
//!   * There is no `dc cvau` pass. Apple Silicon reports `CTR_EL0.IDC = 1`, so
//!     stores are already at the point of unification and a data-cache clean is
//!     unnecessary; libplatform's routine omits it too. This is Apple-only code
//!     (the `cfg` below is macOS + aarch64), so the assumption always holds.

use std::io;
use std::ptr::NonNull;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    // libSystem Mach VM calls, not surfaced by the `libc` crate.
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

/// A page-aligned region of executable memory.
///
/// Created writable-but-not-executable, filled in via [`CodeBuf::write`], then
/// sealed by [`CodeBuf::finalize`]. Calling into it before `finalize` is a bug
/// the type prevents: [`CodeBuf::entry`] does not exist until then.
pub struct CodeBuf {
    /// Writable alias. All writes go here; never executable.
    rw: NonNull<u8>,
    /// Executable alias of the same physical pages; never writable.
    rx: NonNull<u8>,
    /// Rounded up to a page. `munmap` needs the mapped length, not the used one.
    mapped: usize,
    len: usize,
}

// The mapping is owned outright and the pointers are not shared. Sending one to
// another thread is fine; concurrent access is ruled out by `write` taking
// `&mut self`, so there is no `Sync`.
unsafe impl Send for CodeBuf {}

impl CodeBuf {
    /// Map `len` bytes of code memory. The contents are undefined until written.
    pub fn new(len: usize) -> io::Result<Self> {
        assert!(len > 0, "empty code buffer");
        let mapped = len.next_multiple_of(page_size());
        let (rw, rx) = map_dual(mapped, 0)?;
        Ok(CodeBuf {
            rw,
            rx,
            mapped,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fill the buffer through its writable alias.
    ///
    /// The executable alias stays executable throughout, so unlike a
    /// protection-toggle scheme this places no constraint on what else runs
    /// meanwhile. The closure form is kept only to bound the borrow of the
    /// backing bytes.
    pub fn write(&mut self, f: impl FnOnce(&mut [u8])) {
        let slice = unsafe { std::slice::from_raw_parts_mut(self.rw.as_ptr(), self.len) };
        f(slice);
    }

    /// Seal the buffer and publish the code to the instruction cache.
    pub fn finalize(self) -> Code {
        unsafe { sync_icache(self.rw.as_ptr(), self.rx.as_ptr(), self.len) };
        Code(self)
    }
}

impl Drop for CodeBuf {
    fn drop(&mut self) {
        // Two independent mappings of the same physical pages; free both.
        unsafe {
            libc::munmap(self.rw.as_ptr().cast(), self.mapped);
            libc::munmap(self.rx.as_ptr().cast(), self.mapped);
        }
    }
}

/// A finalized, executable code region.
pub struct Code(CodeBuf);

impl Code {
    /// Map, fill, and seal a one-off region from a little-endian instruction
    /// stream. Used by tests; the runtime path goes through the segment
    /// allocator, which does not use `CodeBuf`.
    pub fn from_words(words: &[u32]) -> io::Result<Code> {
        let mut buf = CodeBuf::new(words.len() * 4)?;
        buf.write(|code| {
            for (i, w) in words.iter().enumerate() {
                code[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        });
        Ok(buf.finalize())
    }

    /// Address of the first byte of the executable region.
    ///
    /// # Safety
    /// The caller must transmute this to a function type matching what was
    /// actually encoded. Nothing here checks that.
    pub fn entry(&self) -> *const u8 {
        self.0.rx.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The encoded bytes, for tests and disassembly. Read through the RW alias,
    /// which is always readable.
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0.rw.as_ptr(), self.0.len) }
    }
}

fn page_size() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(n > 0, "sysconf(_SC_PAGESIZE) failed");
    n as usize
}

/// Map `size` bytes twice: a read/write alias and a read/execute alias of the
/// same physical pages. Returns `(rw, rx)`.
///
/// `align_mask` constrains the *RW* alias's base: bits set in the mask are forced
/// to zero in the returned address (`0` for natural page alignment, `SEG-1` for a
/// 64 KiB-aligned segment). The RX alias needs no such alignment — the allocator
/// only ever masks the RW pointer to recover a segment header — so its remap is
/// left to land wherever the kernel picks.
// The Mach aliasing dance is architecture-neutral — only [`sync_icache`] differs
// between aarch64 and x86-64 — so it serves x86-64 macOS (e.g. under Rosetta) too.
#[cfg(target_os = "macos")]
// `mach_task_self` / `mach_vm_map` are deprecated in `libc` in favour of the
// `mach2` crate; the underlying calls are stable and a whole dependency for two
// of them is not worth it.
#[allow(deprecated)]
pub(crate) fn map_dual(size: usize, align_mask: u64) -> io::Result<(NonNull<u8>, NonNull<u8>)> {
    let task = unsafe { libc::mach_task_self() };

    // Writable alias via `mach_vm_map` rather than `mmap`, because only mach lets
    // us demand an alignment (`mask`). `object = 0` makes it anonymous
    // zero-filled; the max protection includes execute so the RX alias can be
    // raised to R+X below — the RW alias's *current* protection never is.
    let mut rw_addr: libc::mach_vm_address_t = 0;
    let kr = unsafe {
        libc::mach_vm_map(
            task,
            &mut rw_addr,
            size as libc::mach_vm_size_t,
            align_mask as libc::mach_vm_offset_t,
            libc::VM_FLAGS_ANYWHERE,
            0, // object: anonymous
            0, // offset
            0, // copy = FALSE
            libc::VM_PROT_READ | libc::VM_PROT_WRITE,
            libc::VM_PROT_READ | libc::VM_PROT_WRITE | libc::VM_PROT_EXECUTE,
            VM_INHERIT_NONE,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return Err(io::Error::other(format!("mach_vm_map failed: {kr}")));
    }

    // Alias the same pages at a fresh address. `copy = FALSE` shares the backing
    // store rather than copy-on-writing it, so writes through `rw` are visible
    // through the new alias.
    let mut rx_addr: libc::mach_vm_address_t = 0;
    let mut cur: libc::vm_prot_t = 0;
    let mut max: libc::vm_prot_t = 0;
    let kr = unsafe {
        mach_vm_remap(
            task,
            &mut rx_addr,
            size as libc::mach_vm_size_t,
            0,
            libc::VM_FLAGS_ANYWHERE,
            task,
            rw_addr,
            0, // FALSE
            &mut cur,
            &mut max,
            VM_INHERIT_NONE,
        )
    };
    if kr != libc::KERN_SUCCESS {
        unsafe { libc::munmap(rw_addr as *mut libc::c_void, size) };
        return Err(io::Error::other(format!("mach_vm_remap failed: {kr}")));
    }

    let kr = unsafe {
        mach_vm_protect(
            task,
            rx_addr,
            size as libc::mach_vm_size_t,
            0, // set_maximum = FALSE
            libc::VM_PROT_READ | libc::VM_PROT_EXECUTE,
        )
    };
    if kr != libc::KERN_SUCCESS {
        unsafe {
            libc::munmap(rw_addr as *mut libc::c_void, size);
            libc::munmap(rx_addr as *mut libc::c_void, size);
        }
        return Err(io::Error::other(format!("mach_vm_protect failed: {kr}")));
    }

    let rw = NonNull::new(rw_addr as *mut u8).expect("mach_vm_map returned null");
    let rx = NonNull::new(rx_addr as *mut u8).expect("mach_vm_remap returned null");
    Ok((rw, rx))
}

/// `VM_INHERIT_NONE` — not surfaced by `libc`. JIT pages are not inherited
/// across `fork`.
#[cfg(target_os = "macos")]
const VM_INHERIT_NONE: libc::vm_inherit_t = 2;

/// Linux dual mapping. Two `MAP_SHARED` views of one anonymous `memfd` give the
/// same physical pages an RW alias and an RX alias, the W^X-safe equivalent of
/// the Mach aliasing above; the fd is closed once mapped, the views keep it live.
///
/// `align_mask` constrains the RW base (see the macOS variant). Linux `mmap` has
/// no alignment request, so a non-zero mask is honoured by reserving an oversized
/// anonymous window, mapping the fd `MAP_FIXED` over its aligned interior, and
/// trimming the slack. The RX alias needs no alignment.
///
/// Architecture-independent — the memfd aliasing dance is the same on aarch64 and
/// x86-64; only [`sync_icache`] differs between them.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
pub(crate) fn map_dual(size: usize, align_mask: u64) -> io::Result<(NonNull<u8>, NonNull<u8>)> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let fd = unsafe { libc::memfd_create(c"tcvm-jit".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // Owned so every early return closes it; the established mappings outlive it.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let raw = fd.as_raw_fd();
    if unsafe { libc::ftruncate(raw, size as libc::off_t) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let rw = if align_mask == 0 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        p
    } else {
        let align = align_mask as usize + 1;
        // Reserve address space, then drop the fd mapping into its aligned interior.
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
        // Trim the reservation slack the `MAP_FIXED` window did not replace.
        if aligned > base {
            unsafe { libc::munmap(base as *mut libc::c_void, aligned - base) };
        }
        let tail = aligned + size;
        unsafe { libc::munmap(tail as *mut libc::c_void, base + reserve - tail) };
        rw
    };

    let rx = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_EXEC,
            libc::MAP_SHARED,
            raw,
            0,
        )
    };
    if rx == libc::MAP_FAILED {
        let err = io::Error::last_os_error();
        unsafe { libc::munmap(rw, size) };
        return Err(err);
    }

    let rw = NonNull::new(rw.cast()).expect("mmap returned null without MAP_FAILED");
    let rx = NonNull::new(rx.cast()).expect("mmap returned null without MAP_FAILED");
    Ok((rw, rx))
}

/// Make bytes written through `rw` fetchable through `rx`: clean the data cache
/// to the point of unification, then invalidate the instruction cache. The two
/// aliases share physical pages, and both `dc cvau` / `ic ivau` operate by
/// physical address, so cleaning the RW range and invalidating the RX range
/// reach the same lines.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn sync_icache(_rw: *mut u8, rx: *mut u8, len: usize) {
    use std::arch::asm;

    // Apple Silicon minimum cache line, per libplatform (see module docs). No
    // `dc cvau` pass: Apple Silicon reports CTR_EL0.IDC=1, so stores are already
    // at the point of unification. Only the instruction cache needs invalidating
    // — the same sequence libplatform's `sys_icache_invalidate` runs.
    const LINE: usize = 64;

    // Invalidate the instruction cache over the executable alias, where the CPU
    // fetches. `ic ivau` works by physical address, so it reaches the lines the
    // RW alias wrote even though the VA differs.
    let mut p = rx as usize & !(LINE - 1);
    let end = rx as usize + len;
    while p < end {
        unsafe { asm!("ic ivau, {}", in(reg) p, options(nostack, preserves_flags)) };
        p += LINE;
    }
    // Complete the invalidations, then flush the fetched instruction stream.
    unsafe {
        asm!("dsb ish", options(nostack, preserves_flags));
        asm!("isb", options(nostack, preserves_flags));
    }
}

/// Make bytes written through `rw` fetchable through `rx` on Linux aarch64.
///
/// `CTR_EL0` is readable from EL0 here (it faults on Darwin, which is why the
/// macOS variant hardcodes its constants), so the cache-line sizes and the
/// coherency bits come straight from it. `IDC = 1` means stores already reach the
/// point of unification and the `dc cvau` clean is unnecessary; `DIC = 1` means
/// the `ic ivau` invalidate is unnecessary. Generic aarch64 cores may report
/// either as `0`, so neither step can be assumed away the way Apple Silicon lets
/// the macOS path drop the clean.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub(crate) unsafe fn sync_icache(rw: *mut u8, rx: *mut u8, len: usize) {
    use std::arch::asm;

    let ctr: u64;
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) };
    // IminLine/DminLine hold log2 of the line size in 4-byte words.
    let dline = 4usize << ((ctr >> 16) & 0xf);
    let iline = 4usize << (ctr & 0xf);
    let idc = (ctr >> 28) & 1 != 0;
    let dic = (ctr >> 29) & 1 != 0;

    // Clean the stores through the RW alias to the PoU; same physical lines the
    // RX alias fetches, reached by physical address despite the differing VA.
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

/// Publish freshly written code on x86-64 Linux.
///
/// x86-64 keeps its instruction cache coherent with the data side in hardware:
/// a store is snooped out of any stale instruction-cache line, so there is no
/// `ic`/`dc` maintenance to perform (the whole reason the aarch64 paths exist).
/// The one thing that *is* required is that the stores through the RW alias be
/// globally visible before the CPU fetches through the RX alias — otherwise a
/// core could fetch stale bytes that were still sitting in the store buffer. A
/// full fence drains it; nothing else is needed, and the aliases share physical
/// pages so no address translation of the written range matters here.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn sync_icache(_rw: *mut u8, _rx: *mut u8, _len: usize) {
    use std::arch::asm;
    unsafe { asm!("mfence", options(nostack, preserves_flags)) };
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// The whole platform story in one test: map, write, seal, call. If this
    /// fails, no amount of correct codegen matters.
    #[test]
    fn map_write_execute() {
        let mut buf = CodeBuf::new(8).expect("mmap");
        buf.write(|code| {
            // mov x0, #42
            code[0..4].copy_from_slice(&0xD280_0540u32.to_le_bytes());
            // ret
            code[4..8].copy_from_slice(&0xD65F_03C0u32.to_le_bytes());
        });
        let code = buf.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(), 42);
    }

    /// Arguments in and out, so the entry ABI is exercised rather than assumed.
    #[test]
    fn passes_arguments() {
        let mut buf = CodeBuf::new(8).expect("mmap");
        buf.write(|code| {
            // add x0, x0, x1
            code[0..4].copy_from_slice(&0x8B01_0000u32.to_le_bytes());
            // ret
            code[4..8].copy_from_slice(&0xD65F_03C0u32.to_le_bytes());
        });
        let code = buf.finalize();

        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(3, 39), 42);
    }

    /// Writing after finalize would be a bug, but the reverse — that the RX
    /// alias really does observe what the RW alias wrote — is the whole premise
    /// of the dual mapping. Exercise a rewrite-then-refinalize path to make sure
    /// the aliasing (not a lucky one-shot) is what carries the bytes across.
    #[test]
    fn alias_observes_writes() {
        let mut buf = CodeBuf::new(8).expect("mmap");
        buf.write(|code| {
            // mov x0, #1
            code[0..4].copy_from_slice(&0xD280_0020u32.to_le_bytes());
            code[4..8].copy_from_slice(&0xD65F_03C0u32.to_le_bytes());
        });
        // Overwrite before sealing; the RX alias must see the final bytes.
        buf.write(|code| {
            // mov x0, #7
            code[0..4].copy_from_slice(&0xD280_00E0u32.to_le_bytes());
        });
        let code = buf.finalize();
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(), 7);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// The whole platform story in one test: map, write, seal, call.
    #[test]
    fn map_write_execute() {
        let mut buf = CodeBuf::new(16).expect("mmap");
        buf.write(|code| {
            // mov eax, 42 ; ret
            code[0..5].copy_from_slice(&[0xB8, 42, 0, 0, 0]);
            code[5] = 0xC3;
        });
        let code = buf.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(), 42);
    }

    /// Arguments in and out, so the entry ABI is exercised rather than assumed.
    /// System V passes the first two integer args in rdi, rsi.
    #[test]
    fn passes_arguments() {
        let mut buf = CodeBuf::new(16).expect("mmap");
        buf.write(|code| {
            // mov rax, rdi ; add rax, rsi ; ret
            code[0..3].copy_from_slice(&[0x48, 0x89, 0xF8]); // mov rax, rdi
            code[3..6].copy_from_slice(&[0x48, 0x01, 0xF0]); // add rax, rsi
            code[6] = 0xC3;
        });
        let code = buf.finalize();

        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(3, 39), 42);
    }

    /// The RX alias must observe writes made through the RW alias, even a rewrite
    /// before sealing.
    #[test]
    fn alias_observes_writes() {
        let mut buf = CodeBuf::new(16).expect("mmap");
        buf.write(|code| {
            // mov eax, 1 ; ret
            code[0..5].copy_from_slice(&[0xB8, 1, 0, 0, 0]);
            code[5] = 0xC3;
        });
        buf.write(|code| {
            // mov eax, 7
            code[0..5].copy_from_slice(&[0xB8, 7, 0, 0, 0]);
        });
        let code = buf.finalize();
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(code.entry()) };
        assert_eq!(f(), 7);
    }
}
