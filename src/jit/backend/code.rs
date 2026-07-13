//! Executable code memory.
//!
//! Apple Silicon enforces W^X, so a page cannot be writable and executable at
//! once. The supported escape is `MAP_JIT`: the mapping is created RWX, but the
//! *thread* decides which half is live, via `pthread_jit_write_protect_np`. That
//! makes the protection flip a register write rather than an `mprotect` syscall,
//! and it is per-thread — which is also the catch. The toggle is a property of
//! the calling thread, so a `CodeBuf` must be written and finalized on one
//! thread, and nothing else on that thread may assume the JIT pages are
//! executable while a write window is open. `write()` owns the whole window for
//! exactly this reason.
//!
//! Toggling protection is not enough on its own. The data cache holds the bytes
//! we just wrote and the instruction cache does not know they changed; on
//! aarch64 those are not coherent, and skipping `sys_icache_invalidate` buys a
//! crash that reproduces once every few hundred runs.

use std::io;
use std::ptr::NonNull;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    /// libSystem, `<libkern/OSCacheControl.h>`. Not in `libc`.
    fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
}

/// A page-aligned region of executable memory.
///
/// Created writable-but-not-executable, filled in via [`CodeBuf::write`], then
/// sealed by [`CodeBuf::finalize`]. Calling into it before `finalize` is a bug
/// the type prevents: [`CodeBuf::entry`] does not exist until then.
pub struct CodeBuf {
    ptr: NonNull<u8>,
    /// Rounded up to a page. `munmap` needs the mapped length, not the used one.
    mapped: usize,
    len: usize,
}

// The mapping is owned outright and the pointer is not shared. Sending one to
// another thread is fine; *writing* it there is not, and `write` is `&mut self`
// on a thread that must own the JIT write window, which is why there is no
// `Sync`.
unsafe impl Send for CodeBuf {}

impl CodeBuf {
    /// Map `len` bytes of code memory. The contents are undefined until written.
    pub fn new(len: usize) -> io::Result<Self> {
        assert!(len > 0, "empty code buffer");
        let page = page_size();
        let mapped = len.next_multiple_of(page);

        // MAP_JIT hands back a mapping that is RWX in the page tables; the
        // per-thread toggle decides which access is actually permitted.
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | map_jit();
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                flags,
                -1,
                0,
            )
        };

        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(CodeBuf {
            ptr: NonNull::new(p.cast()).expect("mmap returned null without MAP_FAILED"),
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

    /// Open the write window, hand the caller the buffer, and close it again.
    ///
    /// The window is thread-wide: while `f` runs, *no* JIT page is executable on
    /// this thread. So `f` must not call into previously compiled code — hence a
    /// closure rather than a `&mut [u8]` the caller may hold across a call.
    pub fn write(&mut self, f: impl FnOnce(&mut [u8])) {
        unsafe { jit_write_protect(false) };
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) };
        f(slice);
        unsafe { jit_write_protect(true) };
    }

    /// Seal the buffer and publish the code to the instruction cache.
    pub fn finalize(self) -> Code {
        unsafe { icache_invalidate(self.ptr.as_ptr(), self.len) };
        Code(self)
    }
}

impl Drop for CodeBuf {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.mapped) };
    }
}

/// A finalized, executable code region.
pub struct Code(CodeBuf);

impl Code {
    /// Address of the first byte of the region.
    ///
    /// # Safety
    /// The caller must transmute this to a function type matching what was
    /// actually encoded. Nothing here checks that.
    pub fn entry(&self) -> *const u8 {
        self.0.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The encoded bytes, for tests and disassembly.
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0.ptr.as_ptr(), self.0.len) }
    }
}

fn page_size() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(n > 0, "sysconf(_SC_PAGESIZE) failed");
    n as usize
}

#[cfg(target_vendor = "apple")]
fn map_jit() -> libc::c_int {
    libc::MAP_JIT
}

#[cfg(not(target_vendor = "apple"))]
fn map_jit() -> libc::c_int {
    0
}

/// Flip this thread's JIT write window. A no-op anywhere W^X is not enforced
/// per-thread, where the RWX mapping is already both.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn jit_write_protect(protect: bool) {
    unsafe { libc::pthread_jit_write_protect_np(protect as libc::c_int) };
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
unsafe fn jit_write_protect(_protect: bool) {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn icache_invalidate(ptr: *mut u8, len: usize) {
    unsafe { sys_icache_invalidate(ptr.cast(), len) };
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
unsafe fn icache_invalidate(_ptr: *mut u8, _len: usize) {}

#[cfg(test)]
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
}
