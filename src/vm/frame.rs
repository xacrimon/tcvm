//! Frame headers in the value stack.
//!
//! Every frame, Lua or native, has four words below its base:
//!
//! ```text
//!  base-4  func    closure pointer | nv << 48      native: nc | at << 48 | cont << 56 | ok << 62
//!  base-3  ret     continuation handler | flags
//!  base-2  caller  the caller's base, null for a thread's bottom frame
//!  base-1  pc      the caller's resume pc
//! ```
//!
//! Header words are not `Value`s: nothing reads them as one, and the tracer
//! skips words 1 to 3.

use crate::dmm::Gc;
use crate::env::function::{FunctionKind, LuaFn, NativeClosure};
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::vm::abi::{Handler, handler_from_bits};

/// Header words below a frame's base.
pub(crate) const HDR: usize = 4;

/// Low 48 bits of a header word: a pointer.
pub(crate) const PTR_MASK: u64 = (1 << 48) - 1;

/// `ret` word flag bits. Handler addresses are 32-byte aligned, so five bits
/// are free.
pub(crate) mod flag {
    /// Word 0 is a native closure; word 3 is still the caller's pc.
    pub(crate) const NATIVE: u64 = 1;
    /// A CLOSURE in this frame captured a local by reference.
    pub(crate) const HAS_OPEN: u64 = 2;
    /// A TBC in this frame registered a to-be-closed slot.
    pub(crate) const HAS_TBC: u64 = 4;
    /// Native frame: errors reaching it go to its continuation.
    pub(crate) const PROTECTED: u64 = 8;
    /// Native frame: window slot 0 holds the `xpcall` message handler.
    pub(crate) const HANDLER: u64 = 16;
    pub(crate) const MASK: u64 = 31;
}

/// Native frame word 0 fields.
pub(crate) mod native_word {
    pub(crate) const AT_SHIFT: u32 = 48;
    pub(crate) const CONT_SHIFT: u32 = 56;
    pub(crate) const OK_SHIFT: u32 = 62;
}

#[inline(always)]
pub(crate) unsafe fn hdr<'gc>(base: *mut Value<'gc>) -> *mut u64 {
    unsafe { base.sub(HDR).cast() }
}

#[inline(always)]
pub(crate) unsafe fn func_word<'gc>(base: *mut Value<'gc>) -> u64 {
    unsafe { hdr(base).read() }
}

#[inline(always)]
pub(crate) unsafe fn set_func_word<'gc>(base: *mut Value<'gc>, w: u64) {
    unsafe { hdr(base).write(w) }
}

#[inline(always)]
pub(crate) unsafe fn ret_word<'gc>(base: *mut Value<'gc>) -> u64 {
    unsafe { hdr(base).add(1).read() }
}

#[inline(always)]
pub(crate) unsafe fn set_ret_word<'gc>(base: *mut Value<'gc>, w: u64) {
    unsafe { hdr(base).add(1).write(w) }
}

#[inline(always)]
pub(crate) unsafe fn flags<'gc>(base: *mut Value<'gc>) -> u64 {
    unsafe { ret_word(base) & flag::MASK }
}

#[inline(always)]
pub(crate) unsafe fn set_flags<'gc>(base: *mut Value<'gc>, bits: u64) {
    unsafe { set_ret_word(base, ret_word(base) | bits) }
}

#[inline(always)]
pub(crate) unsafe fn clear_flags<'gc>(base: *mut Value<'gc>, bits: u64) {
    unsafe { set_ret_word(base, ret_word(base) & !bits) }
}

#[inline(always)]
pub(crate) unsafe fn is_native<'gc>(base: *mut Value<'gc>) -> bool {
    unsafe { ret_word(base) & flag::NATIVE != 0 }
}

/// The frame's continuation, flags stripped.
#[inline(always)]
pub(crate) unsafe fn ret<'gc>(base: *mut Value<'gc>) -> Handler {
    unsafe { handler_from_bits(ret_word(base)) }
}

/// The caller's base and resume pc, one pair load of words 2 and 3.
#[inline(always)]
pub(crate) unsafe fn caller<'gc>(base: *mut Value<'gc>) -> (*mut Value<'gc>, *const Instruction) {
    unsafe {
        let p = hdr(base);
        (
            p.add(2).read() as usize as *mut Value<'gc>,
            p.add(3).read() as usize as *const Instruction,
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn caller_base<'gc>(base: *mut Value<'gc>) -> *mut Value<'gc> {
    unsafe { hdr(base).add(2).read() as usize as *mut Value<'gc> }
}

#[inline(always)]
pub(crate) unsafe fn set_caller_base<'gc>(base: *mut Value<'gc>, caller: *mut Value<'gc>) {
    unsafe { hdr(base).add(2).write(caller as usize as u64) }
}

#[inline(always)]
pub(crate) unsafe fn caller_pc<'gc>(base: *mut Value<'gc>) -> *const Instruction {
    unsafe { hdr(base).add(3).read() as usize as *const Instruction }
}

/// The Lua frame's closure (word 0 masked to its pointer).
#[inline(always)]
pub(crate) unsafe fn closure<'gc>(base: *mut Value<'gc>) -> LuaFn<'gc> {
    unsafe { LuaFn::from_ptr((func_word(base) & PTR_MASK) as usize as *const FunctionKind<'gc>) }
}

/// Word 0 as a function of either kind.
#[inline(always)]
pub(crate) unsafe fn function<'gc>(base: *mut Value<'gc>) -> Gc<'gc, FunctionKind<'gc>> {
    unsafe { Gc::from_ptr((func_word(base) & PTR_MASK) as usize as *const FunctionKind<'gc>) }
}

/// The native frame's closure.
#[inline(always)]
pub(crate) unsafe fn native_closure<'gc>(base: *mut Value<'gc>) -> &'gc NativeClosure<'gc> {
    unsafe {
        match function(base).as_ref() {
            FunctionKind::Native(nc) => &*(nc as *const NativeClosure<'gc>),
            FunctionKind::Lua(_) => std::hint::unreachable_unchecked(),
        }
    }
}

/// A Lua frame's vararg count.
#[inline(always)]
pub(crate) unsafe fn nv<'gc>(base: *mut Value<'gc>) -> usize {
    unsafe { (func_word(base) >> 48) as usize }
}

/// How far below `base - 4` the frame's header originally started: a Lua
/// frame's vararg count, 0 for a native frame (whose word 0 has no `nv`).
#[inline(always)]
pub(crate) unsafe fn extras<'gc>(base: *mut Value<'gc>) -> usize {
    unsafe { if is_native(base) { 0 } else { nv(base) } }
}

/// A Lua frame's word 0.
#[inline(always)]
pub(crate) fn lua_func_word(f: LuaFn<'_>, nv: usize) -> u64 {
    f.as_ptr() as usize as u64 | (nv as u64) << 48
}

/// A native frame's word 0 fields.
#[derive(Clone, Copy)]
pub(crate) struct NativeHdr {
    pub(crate) at: usize,
    pub(crate) cont: u8,
    pub(crate) ok: u8,
}

impl NativeHdr {
    #[inline(always)]
    pub(crate) fn unpack(w: u64) -> Self {
        NativeHdr {
            at: ((w >> native_word::AT_SHIFT) & 0xff) as usize,
            cont: ((w >> native_word::CONT_SHIFT) & 0x3f) as u8,
            ok: (w >> native_word::OK_SHIFT) as u8,
        }
    }

    #[inline(always)]
    pub(crate) fn pack(self, nc: &NativeClosure<'_>) -> u64 {
        debug_assert!(self.at < 256 && self.cont < 64 && self.ok < 4);
        (nc as *const NativeClosure<'_> as usize as u64)
            | (self.at as u64) << native_word::AT_SHIFT
            | (self.cont as u64) << native_word::CONT_SHIFT
            | (self.ok as u64) << native_word::OK_SHIFT
    }
}

/// Write a complete header at `hdr`: two pair stores.
///
/// # Safety
/// `hdr .. hdr + 4` is inside the stack.
#[inline(always)]
pub(crate) unsafe fn write_hdr<'gc>(
    hdr: *mut Value<'gc>,
    func: u64,
    ret: u64,
    caller: *mut Value<'gc>,
    pc: *const Instruction,
) {
    unsafe {
        let p = hdr.cast::<u64>();
        p.write(func);
        p.add(1).write(ret);
        p.add(2).write(caller as usize as u64);
        p.add(3).write(pc as usize as u64);
    }
}

/// Write nil to `n` slots at `p`. The opaque pointer keeps this a loop: as a
/// `memset` call it would give every handler using it a stack frame.
///
/// # Safety
/// `p .. p + n` is in bounds.
#[inline(always)]
pub(crate) unsafe fn fill_nil<'gc>(mut p: *mut Value<'gc>, n: usize) {
    for _ in 0..n {
        unsafe {
            p.write(Value::nil());
            p = p.add(1);
        }
        #[allow(clippy::pointers_in_nomem_asm_block)]
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) p, options(nomem, nostack, preserves_flags));
        }
    }
}

/// Copy `n` values from `src` to `dst` in ascending order, so `dst` may
/// overlap `src` from below. A loop like [`fill_nil`].
///
/// # Safety
/// Both ranges are in bounds, and `dst <= src` if they overlap.
#[inline(always)]
pub(crate) unsafe fn copy_values<'gc>(
    mut dst: *mut Value<'gc>,
    mut src: *const Value<'gc>,
    n: usize,
) {
    for _ in 0..n {
        unsafe {
            dst.write(src.read());
            dst = dst.add(1);
            src = src.add(1);
        }
        #[allow(clippy::pointers_in_nomem_asm_block)]
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) dst, options(nomem, nostack, preserves_flags));
        }
    }
}

/// Land `nret` values from `src` as `wanted` values at `dst`: the first
/// `min(nret, wanted)` copied, the rest nil.
///
/// # Safety
/// As [`copy_values`] for the copied part, `dst .. dst + wanted` in bounds.
#[inline(always)]
pub(crate) unsafe fn land_results<'gc>(
    dst: *mut Value<'gc>,
    src: *const Value<'gc>,
    nret: usize,
    wanted: usize,
) {
    let n = nret.min(wanted);
    unsafe {
        copy_values(dst, src, n);
        fill_nil(dst.add(n), wanted - n);
    }
}

/// A frame seen while walking a thread's frames, innermost first.
#[derive(Clone, Copy)]
pub(crate) struct Frame<'gc> {
    pub(crate) base: *mut Value<'gc>,
    pub(crate) func: u64,
    pub(crate) ret: u64,
    /// This frame's current pc: the published `top_pc` for the top frame,
    /// the frame above's header word 3 otherwise. Meaningless for a native.
    pub(crate) pc: *const Instruction,
}

impl<'gc> Frame<'gc> {
    #[inline]
    pub(crate) fn is_native(&self) -> bool {
        self.ret & flag::NATIVE != 0
    }

    /// Whether the continuation is `h`.
    #[inline]
    pub(crate) fn returns_through(&self, h: Handler) -> bool {
        self.ret & !flag::MASK == h as usize as u64
    }

    /// The Lua frame's closure.
    pub(crate) fn closure(&self) -> LuaFn<'gc> {
        debug_assert!(!self.is_native());
        unsafe { LuaFn::from_ptr((self.func & PTR_MASK) as usize as *const FunctionKind<'gc>) }
    }

    /// `pc` as an index into the closure's code.
    pub(crate) fn pc_index(&self) -> usize {
        debug_assert!(!self.is_native());
        unsafe { self.pc.offset_from_unsigned(self.closure().code) }
    }

    /// The source line the Lua frame executes (`pc` points past it).
    pub(crate) fn line(&self) -> Option<u32> {
        if self.is_native() || self.pc.is_null() {
            return None;
        }
        let idx = self.pc_index().checked_sub(1)?;
        self.closure().proto.line_for_pc(idx)
    }
}

/// The thread's frames from the published top down. A `pcall`/`xpcall`
/// that called a frame without a frame of its own still counts as a
/// level: a synthetic native frame follows the frame it called, so error
/// levels match the reference.
pub(crate) fn frames<'a, 'gc>(ts: &'a ThreadState<'gc>) -> impl Iterator<Item = Frame<'gc>> + 'a {
    let mut base = ts.top_base;
    let mut pc = ts.top_pc;
    let mut elided: Option<Frame<'gc>> = None;
    std::iter::from_fn(move || {
        if let Some(f) = elided.take() {
            return Some(f);
        }
        if base.is_null() {
            return None;
        }
        let f = unsafe {
            Frame {
                base,
                func: func_word(base),
                ret: ret_word(base),
                pc,
            }
        };
        if f.returns_through(crate::vm::native::ret_pcall)
            || f.returns_through(crate::vm::native::ret_xpcall)
        {
            elided = Some(Frame {
                base,
                func: 0,
                ret: flag::NATIVE,
                pc: std::ptr::null(),
            });
        }
        unsafe {
            pc = caller_pc(base);
            base = caller_base(base);
        }
        Some(f)
    })
}

/// Point every header's caller word, and the published base, into a stack
/// that moved from address `old` to `new`.
pub(crate) fn rebase<'gc>(ts: &mut ThreadState<'gc>, old: usize, new: *mut Value<'gc>) {
    let shift = |p: *mut Value<'gc>| -> *mut Value<'gc> {
        if p.is_null() {
            p
        } else {
            unsafe { new.add((p.addr() - old) / size_of::<Value<'gc>>()) }
        }
    };
    ts.top_base = shift(ts.top_base);
    let mut base = ts.top_base;
    while !base.is_null() {
        unsafe {
            let c = shift(caller_base(base));
            set_caller_base(base, c);
            base = c;
        }
    }
}
