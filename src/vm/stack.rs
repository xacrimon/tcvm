//! Thread stacks: one mapping per thread, holding its native stack (growing
//! down) and its value stack (growing up), which therefore never moves.
//!
//! ```text
//!  guard | native stack ............ | value stack ........ | guard
//!                            the two meet 8 KB into a page
//! ```
//!
//! A thread that uses under 8 KB of each touches a single page. Mappings
//! come from a per-`Lua` pool, which keeps a few free ones with their pages
//! past the meeting page released.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::rc::Rc;

use crate::env::thread::{MAX_STACK, NATIVE_SLACK};
use crate::env::value::Value;

/// A multiple of every page size the runtime meets (16 KB on Apple
/// silicon, 4 KB elsewhere).
pub(crate) const PAGE: usize = 16 * 1024;
/// The value stack's bytes: its half of the meeting page, then 512 KB.
pub(crate) const VALUE_BYTES: usize = 512 * 1024 + PAGE / 2;
/// Slots the value stack holds.
pub(crate) const VALUE_SLOTS: usize = VALUE_BYTES / size_of::<Value<'static>>();
/// The default native stack size.
pub(crate) const NATIVE_BYTES: usize = 1024 * 1024;
/// Native stack above its end that compiled code leaves to natives and the
/// interpreter: past it, compiled entries refuse (H9).
pub(crate) const NATIVE_HEADROOM: usize = 256 * 1024;
/// Free mappings the pool keeps; more are unmapped.
const POOL_MAX: usize = 64;

// The limit, the natives' slack and the unwinder's runner frames above them.
const _: () = assert!(MAX_STACK + NATIVE_SLACK + 64 <= VALUE_SLOTS);

/// The free mappings of one `Lua`, and the native stack size of new ones.
pub(crate) struct StackPool {
    native: Cell<usize>,
    free: RefCell<Vec<(NonNull<u8>, usize)>>,
}

impl StackPool {
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(StackPool {
            native: Cell::new(NATIVE_BYTES),
            free: RefCell::new(Vec::new()),
        })
    }

    /// Native stack bytes of the mappings handed out from now on.
    pub(crate) fn set_native_size(&self, bytes: usize) {
        self.native.set(bytes.next_multiple_of(PAGE).max(4 * PAGE));
    }

    /// A mapping, reused or new.
    pub(crate) fn acquire(self: &Rc<Self>) -> ThreadStack {
        let native = self.native.get();
        let reused = {
            let mut free = self.free.borrow_mut();
            free.iter()
                .rposition(|&(_, n)| n == native)
                .map(|k| free.swap_remove(k).0)
        };
        let map = reused.unwrap_or_else(|| map_stack(native));
        ThreadStack {
            map,
            native,
            value_used: VALUE_BYTES,
            pool: self.clone(),
        }
    }

    fn release(&self, map: NonNull<u8>, native: usize, value_used: usize) {
        let base = map.as_ptr() as usize;
        let meet = base + PAGE + native + PAGE / 2;
        unsafe {
            libc::madvise((base + PAGE) as *mut _, native, libc::MADV_FREE);
            if value_used > PAGE / 2 {
                libc::madvise(
                    (meet + PAGE / 2) as *mut _,
                    VALUE_BYTES - PAGE / 2,
                    libc::MADV_FREE,
                );
            }
        }
        let mut free = self.free.borrow_mut();
        if free.len() < POOL_MAX && native == self.native.get() {
            free.push((map, native));
        } else {
            unmap_stack(map, native);
        }
    }
}

impl Drop for StackPool {
    fn drop(&mut self) {
        for (map, native) in self.free.get_mut().drain(..) {
            unmap_stack(map, native);
        }
    }
}

fn total_bytes(native: usize) -> usize {
    PAGE + native + PAGE / 2 + VALUE_BYTES + PAGE
}

fn map_stack(native: usize) -> NonNull<u8> {
    let total = total_bytes(native);
    unsafe {
        let p = libc::mmap(
            std::ptr::null_mut(),
            total,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert!(p != libc::MAP_FAILED, "mapping a thread stack failed");
        let base = p as usize;
        let lo = libc::mprotect(p, PAGE, libc::PROT_NONE);
        let hi = libc::mprotect((base + total - PAGE) as *mut _, PAGE, libc::PROT_NONE);
        assert!(
            lo == 0 && hi == 0,
            "protecting a thread stack's guard pages failed"
        );
        NonNull::new_unchecked(p as *mut u8)
    }
}

fn unmap_stack(map: NonNull<u8>, native: usize) {
    unsafe {
        libc::munmap(map.as_ptr() as *mut _, total_bytes(native));
    }
}

/// A thread's mapping; dropping it hands it back to its pool.
pub(crate) struct ThreadStack {
    map: NonNull<u8>,
    native: usize,
    /// Bytes of the value stack that may have been touched.
    pub(crate) value_used: usize,
    pool: Rc<StackPool>,
}

impl ThreadStack {
    /// Where the native stack starts (its highest address) and the value
    /// stack begins.
    pub(crate) fn meet(&self) -> usize {
        self.map.as_ptr() as usize + PAGE + self.native + PAGE / 2
    }

    /// The lowest address the native stack may use.
    pub(crate) fn native_end(&self) -> usize {
        self.map.as_ptr() as usize + PAGE
    }

    pub(crate) fn values<'gc>(&self) -> *mut Value<'gc> {
        self.meet() as *mut Value<'gc>
    }
}

impl Drop for ThreadStack {
    fn drop(&mut self) {
        self.pool.release(self.map, self.native, self.value_used);
    }
}

/// A level's entry record (`jit-perf-fixes2.md` H6): a run of dispatch
/// started on top of compiled frames on the same native stack, which
/// returns to them through `ret_to_native`. Pushed on the native stack by
/// `abi::jit_level`, which handlers then run below; the bottom level's is
/// `dispatch_entry`'s.
#[repr(C)]
pub(crate) struct EntryRecord {
    /// The frame record of the code that started the level: its frame
    /// pointer and the return address (tagged in bit 0 for a `br` return).
    pub(crate) fp: usize,
    pub(crate) lr: usize,
    /// The record of the level below, 0 for the bottom level's.
    pub(crate) prev: usize,
    /// The base of the level's first frame: frames below it are the
    /// compiled caller's. 0 for the bottom level.
    pub(crate) first: usize,
    /// `ThreadState::native_deep` of the level below.
    pub(crate) deep: usize,
    _pad: usize,
    /// x19 to x28 and d8 to d15 of the Rust code that started the bottom
    /// level; a level compiled code starts keeps only x22 to x25, x22 its
    /// `BASE` (`abi::jit_level`).
    pub(crate) x: [usize; 10],
    pub(crate) d: [u64; 8],
}

pub(crate) mod rec {
    use std::mem::offset_of;

    use super::EntryRecord;
    pub(crate) const SIZE: usize = size_of::<EntryRecord>();
    pub(crate) const PREV: usize = offset_of!(EntryRecord, prev);
    pub(crate) const DEEP: usize = offset_of!(EntryRecord, deep);
    pub(crate) const X: usize = offset_of!(EntryRecord, x);
    pub(crate) const D: usize = offset_of!(EntryRecord, d);
    const _: () = assert!(SIZE == 192 && PREV == 16 && X == 48 && D == 128);
}

/// Where handlers in a level return to (`abi::level_return`), 0 where
/// there are no levels.
#[inline]
pub(crate) fn level_ret() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        crate::jit::backend::aarch64::abi::level_return_addr()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}

/// `abi::ret_to_native` as header words hold it, a value no handler has
/// where there are no levels.
#[inline]
pub(crate) fn ret_to_native_bits() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::vm::abi::handler_bits(crate::jit::backend::aarch64::abi::ret_to_native)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        u64::MAX
    }
}

/// `abi::ret_shim` as header words hold it, a value no handler has where
/// there are no compiled natives.
#[inline]
pub(crate) fn ret_shim_bits() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::vm::abi::handler_bits(crate::jit::backend::aarch64::abi::ret_shim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        u64::MAX - 1
    }
}

/// Why a thread handed control back to the host.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u64)]
pub(crate) enum Park {
    /// The collector is owed work: the host collects, then switches back.
    Gc = 1,
    /// An async native waits on the host.
    Pending = 2,
    /// A main thread yielded to the host.
    Yield = 3,
    /// The thread ended, or an error nothing caught ended it.
    End = 4,
    /// Run `Runtime::host_call` on the host stack (`on_host`), then switch
    /// back.
    Host = 5,
    /// A Rust panic reached the thread's start: the host resumes it from
    /// `Runtime::panic`.
    Panic = 6,
}

impl Park {
    fn from_raw(v: u64) -> Self {
        match v {
            1 => Park::Gc,
            2 => Park::Pending,
            3 => Park::Yield,
            4 => Park::End,
            5 => Park::Host,
            6 => Park::Panic,
            _ => unreachable!("park reason {v}"),
        }
    }
}

/// Where a waiting thread (or the host) continues: its stack and frame
/// pointers, the code to branch to, and on aarch64 the return address its
/// handlers run with. A switch branches to `cont` with these loaded, `rt`
/// and `thread` in the handler registers and the switch's value in the
/// first argument register; it is never a return, so the return address
/// stack keeps predicting the code on either side.
#[derive(Default)]
#[repr(C)]
pub(crate) struct SwitchPoint {
    pub(crate) sp: usize,
    pub(crate) cont: usize,
    pub(crate) lr: usize,
    pub(crate) fp: usize,
}

pub(crate) mod sw {
    pub(crate) const SP: usize = std::mem::offset_of!(super::SwitchPoint, sp);
    #[cfg(target_arch = "x86_64")]
    pub(crate) const CONT: usize = std::mem::offset_of!(super::SwitchPoint, cont);
    pub(crate) const LR: usize = std::mem::offset_of!(super::SwitchPoint, lr);
    #[cfg(target_arch = "x86_64")]
    pub(crate) const FP: usize = std::mem::offset_of!(super::SwitchPoint, fp);
    /// `ThreadState::sw` from a thread pointer.
    pub(crate) const THREAD: usize =
        std::mem::offset_of!(crate::env::thread::ThreadState<'static>, sw);
}

/// Wait at `save` with the callee-saved registers on the current stack and
/// switch to `load`, passing `v`, `rt` and `thread` (the thread that runs
/// there); returns the value of the switch back to `save`.
///
/// # Safety
/// `load` was saved by a switch away from a stack that is still mapped, or
/// comes from [`fresh`]. Code waiting on either stack holds no reference to
/// state the other side writes (the threads' states, their stacks), and
/// keeps everything it needs alive reachable from its thread: the collector
/// scans no Rust frame. The `'gc` of a waiting frame stays valid across
/// `Lua::enter`s because the heap does not move and `State` and `Mutation`
/// keep their addresses (both are boxed).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn stack_switch(
    save: *mut SwitchPoint,
    load: *const SwitchPoint,
    v: u64,
    rt: *const crate::lua::State<'_>,
    thread: *mut crate::env::thread::ThreadState<'_>,
) -> u64 {
    // The save area is addressed from x9, not `sp`: `sp`-relative stores
    // and loads at the same offsets on two stacks cost the core a flush.
    core::arch::naked_asm!(
        "sub x9, sp, #160",
        "stp x19, x20, [x9, #0]",
        "stp x21, x22, [x9, #16]",
        "stp x23, x24, [x9, #32]",
        "stp x25, x26, [x9, #48]",
        "stp x27, x28, [x9, #64]",
        "stp x29, x30, [x9, #80]",
        "stp d8, d9, [x9, #96]",
        "stp d10, d11, [x9, #112]",
        "stp d12, d13, [x9, #128]",
        "stp d14, d15, [x9, #144]",
        "adr x10, 2f",
        "stp x9, x10, [x0, #{sp}]",
        "ldp x11, x12, [x1, #{sp}]",
        "ldp x30, x29, [x1, #{lr}]",
        "mov sp, x11",
        "mov x0, x2",
        "mov x23, x3",
        "mov x25, x4",
        "br x12",
        // Back at `save`.
        "2:",
        "mov x9, sp",
        "ldp x19, x20, [x9, #0]",
        "ldp x21, x22, [x9, #16]",
        "ldp x23, x24, [x9, #32]",
        "ldp x25, x26, [x9, #48]",
        "ldp x27, x28, [x9, #64]",
        "ldp x29, x30, [x9, #80]",
        "ldp d8, d9, [x9, #96]",
        "ldp d10, d11, [x9, #112]",
        "ldp d12, d13, [x9, #128]",
        "ldp d14, d15, [x9, #144]",
        "add sp, x9, #160",
        "ret",
        sp = const sw::SP,
        lr = const sw::LR,
    )
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn stack_switch(
    save: *mut SwitchPoint,
    load: *const SwitchPoint,
    v: u64,
    rt: *const crate::lua::State<'_>,
    thread: *mut crate::env::thread::ThreadState<'_>,
) -> u64 {
    // The thread is the runtime's (`Context::set_thread`); `rt` goes where
    // the handlers take it.
    core::arch::naked_asm!(
        "push rbp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "lea rax, [rip + 2f]",
        "mov [rdi + {sp}], rsp",
        "mov [rdi + {cont}], rax",
        "mov rsp, [rsi + {sp}]",
        "mov rbp, [rsi + {fp}]",
        "mov r9, [rsi + {cont}]",
        "mov rax, rdx",
        "mov r15, rcx",
        "jmp r9",
        "2:",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret",
        sp = const sw::SP,
        cont = const sw::CONT,
        fp = const sw::FP,
    )
}

/// The switch out of a handler (`switch_out!`): the running thread `from`
/// (in the `insn` slot) waits at its switch point to continue in the
/// handler in the `pc` slot, and `to` (in the `closure` slot) runs.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub(crate) extern "rust-preserve-none" fn switch_handler<'gc>(
    _from: crate::vm::abi::Slot,
    _cont: *const crate::instruction::Instruction,
    _base: *mut Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _to: crate::vm::abi::Slot,
    _thread: *mut crate::env::thread::ThreadState<'gc>,
) -> crate::vm::abi::Exit {
    core::arch::naked_asm!(
        "mov x9, sp",
        "add x10, x20, #{tsw}",
        "stp x9, x21, [x10, #{sp}]",
        "stp x30, x29, [x10, #{lr}]",
        "add x11, x24, #{tsw}",
        "ldp x9, x12, [x11, #{sp}]",
        "ldp x30, x29, [x11, #{lr}]",
        "mov sp, x9",
        "mov x25, x24",
        "br x12",
        tsw = const sw::THREAD,
        sp = const sw::SP,
        lr = const sw::LR,
    )
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub(crate) extern "rust-preserve-none" fn switch_handler<'gc>(
    _from: crate::vm::abi::Slot,
    _cont: *const crate::instruction::Instruction,
    _base: *mut Value<'gc>,
    _rt: crate::lua::Context<'gc>,
    _to: crate::vm::abi::Slot,
) -> crate::vm::abi::Exit {
    // preserve_none's argument order: r12, r13, r14, r15, rdi.
    core::arch::naked_asm!(
        "mov [r12 + {tsw} + {sp}], rsp",
        "mov [r12 + {tsw} + {cont}], r13",
        "mov [r12 + {tsw} + {fp}], rbp",
        "mov rsp, [rdi + {tsw} + {sp}]",
        "mov rbp, [rdi + {tsw} + {fp}]",
        "mov rax, [rdi + {tsw} + {cont}]",
        "jmp rax",
        tsw = const sw::THREAD,
        sp = const sw::SP,
        cont = const sw::CONT,
        fp = const sw::FP,
    )
}

/// A fresh stack's first code: calls `thread_start(rt, thread)`.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn thread_entry() {
    core::arch::naked_asm!(
        "mov x29, #0",
        "mov x0, x23",
        "mov x1, x25",
        "bl {start}",
        "brk #1",
        start = sym thread_start,
    )
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn thread_entry() {
    // `rsp` is the stack's 16-aligned top, as the `call` needs.
    core::arch::naked_asm!(
        "mov rdi, r15",
        "mov rsi, [r15 + {thread}]",
        "call {start}",
        "ud2",
        thread = const crate::jit::layout::RT_THREAD,
        start = sym thread_start,
    )
}

/// The switch point of `mem`'s native stack, laid out so that the first
/// switch to it runs `thread_start`.
pub(crate) fn fresh(mem: &ThreadStack) -> SwitchPoint {
    SwitchPoint {
        sp: mem.meet(),
        cont: thread_entry as *const () as usize,
        lr: 0,
        fp: 0,
    }
}

/// A thread's first code on its own stack: its seeded call, then the park
/// that tells the host it ended. Nothing switches back into an ended thread.
unsafe extern "C" fn thread_start(
    state: *const crate::lua::State<'static>,
    ts: *mut crate::env::thread::ThreadState<'static>,
) -> ! {
    let ctx = unsafe { crate::lua::Context::from_state(state) };
    // A panic does not unwind past here: the host resumes it on its stack.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        crate::vm::dispatch::start(ctx, ts)
    }));
    if let Err(p) = r {
        unsafe { (*ts).status = crate::env::thread::ThreadStatus::Stopped };
        ctx.runtime().panic.set(Some(p));
        park(ctx, Park::Panic);
    }
    park(ctx, Park::End);
    std::process::abort()
}

/// Make `to` the running thread and switch to it, waiting at `from`'s
/// switch point; returns the value of the switch back. Gives `to` its fresh
/// switch point if it never ran.
///
/// # Safety
/// As [`stack_switch`]: no reference to either state is live.
pub(crate) unsafe fn switch_to<'gc>(
    ctx: crate::lua::Context<'gc>,
    from: *mut SwitchPoint,
    to: *mut crate::env::thread::ThreadState<'gc>,
    v: u64,
) -> u64 {
    ctx.set_thread(to);
    unsafe {
        if (*to).sw.cont == 0 {
            (*to).sw = fresh(
                (*to)
                    .mem
                    .as_ref()
                    .expect("switching to a thread without a stack"),
            );
        }
        let r = stack_switch(from, &raw const (*to).sw, v, ctx.state_ptr(), to);
        release_dead(ctx);
        r
    }
}

/// Hand back the stack a coroutine that ended handed over (`Runtime::dead`)
/// once off it.
#[inline]
pub(crate) fn release_dead(ctx: crate::lua::Context<'_>) {
    if let Some(mem) = ctx.runtime().dead.take() {
        drop(mem);
    }
}

/// Hand control to the host for `why`; returns when the host switches back.
pub(crate) fn park(ctx: crate::lua::Context<'_>, why: Park) {
    let rt = ctx.runtime();
    let ts = ctx.thread_ptr();
    unsafe {
        stack_switch(
            &raw mut (*ts).sw,
            rt.host.as_ptr(),
            why as u64,
            ctx.state_ptr(),
            std::ptr::null_mut(),
        )
    };
}

/// Switch to `ts` and run it until a thread parks: that thread (the one
/// running then, perhaps a coroutine `ts` resumed) and why. Host calls
/// (`on_host`) run here, and the thread goes on.
///
/// # Safety
/// `ts` has a stack, is not running, and its parked code (if any) parked
/// for a reason the caller has dealt with.
pub(crate) unsafe fn run<'gc>(
    ctx: crate::lua::Context<'gc>,
    mut ts: *mut crate::env::thread::ThreadState<'gc>,
) -> (Park, *mut crate::env::thread::ThreadState<'gc>) {
    let rt = ctx.runtime();
    // A host call from a native may step another executor: this host is
    // then a thread's stack, whose own switch point is restored after.
    let outer = rt.host.take();
    let outer_on = rt.on_thread.get();
    let why = loop {
        rt.on_thread.set(true);
        let why = Park::from_raw(unsafe { switch_to(ctx, rt.host.as_ptr(), ts, 0) });
        rt.on_thread.set(outer_on);
        ts = ctx.thread_ptr();
        if why == Park::Host {
            let f = rt.host_call.take().expect("a host park without a call");
            unsafe { (*f.as_ptr())() };
            continue;
        }
        break why;
    };
    rt.host.set(outer);
    if why == Park::Panic {
        std::panic::resume_unwind(rt.panic.take().expect("a panic park without a panic"));
    }
    (why, ts)
}

/// Run `f` on the host's stack, as heavy Rust work with no depth bound must
/// (the parser and compiler, JIT compilation, error rendering): from a
/// thread's stack this parks with `Park::Host`.
pub(crate) fn on_host<R>(ctx: crate::lua::Context<'_>, f: impl FnOnce() -> R) -> R {
    let rt = ctx.runtime();
    if !rt.on_thread.get() {
        return f();
    }
    let mut f = Some(f);
    let mut out: Option<R> = None;
    {
        let mut call = || out = Some((f.take().expect("a host call run twice"))());
        let call: &mut dyn FnMut() = &mut call;
        // SAFETY: the host runs it before switching back, while this frame,
        // which owns it, waits parked.
        let call: NonNull<dyn FnMut()> = unsafe { std::mem::transmute(NonNull::from(call)) };
        rt.host_call.set(Some(call));
        park(ctx, Park::Host);
    }
    out.expect("a host call that did not run")
}
