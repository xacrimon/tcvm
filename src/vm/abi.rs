//! The handler ABI: the slots every dispatch target receives, the declaration
//! macro and the body macros handlers are written with.
//!
//! Every target has the same signature, which is what `become` requires. The
//! five portable slots are `insn`, `pc`, `base`, `rt`, `closure`; `thread` is a
//! sixth on aarch64 only (x86-64 loads it from `rt`). What a slot holds depends
//! on the target's kind:
//!
//! | slot      | `op`              | `cont`                     | `slow` / `entry`            |
//! |-----------|-------------------|----------------------------|-----------------------------|
//! | `insn`    | instruction word  | result count               | payload (insn is at `pc-1`) |
//! | `pc`      | next instruction  | first result               | next instruction, or header |
//! | `base`    | register window   | the finished callee's base | register window             |
//! | `closure` | running closure   | unspecified                | payload, often the native   |

use crate::env::function::{FunctionKind, LuaFn, NativeClosure};
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::lua::Context;

/// A handler argument whose meaning the target's kind defines. Plain integer
/// in the ABI, so a non-pointer payload is sound to pass through it.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct Slot(u64);

impl Slot {
    #[inline(always)]
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub(crate) const fn from_raw(v: u64) -> Self {
        Slot(v)
    }

    #[inline(always)]
    pub(crate) fn insn(i: Instruction) -> Self {
        Slot(i.raw())
    }

    #[inline(always)]
    pub(crate) fn as_insn(self) -> Instruction {
        Instruction::from_raw(self.0)
    }

    #[inline(always)]
    pub(crate) fn nret(n: usize) -> Self {
        Slot(n as u64)
    }

    #[inline(always)]
    pub(crate) fn as_nret(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub(crate) fn closure(f: LuaFn<'_>) -> Self {
        Slot(f.as_ptr() as u64)
    }

    /// # Safety
    /// The slot holds a `Slot::closure`.
    #[inline(always)]
    pub(crate) unsafe fn as_closure<'gc>(self) -> LuaFn<'gc> {
        unsafe { LuaFn::from_ptr(self.0 as *const FunctionKind<'gc>) }
    }

    #[inline(always)]
    pub(crate) fn native(nc: &NativeClosure<'_>) -> Self {
        Slot(nc as *const _ as u64)
    }

    /// # Safety
    /// The slot holds a `Slot::native` whose closure is live.
    #[inline(always)]
    pub(crate) unsafe fn as_native<'gc>(self) -> &'gc NativeClosure<'gc> {
        unsafe { &*(self.0 as *const NativeClosure<'gc>) }
    }
}

/// Why the dispatch chain returned to the trampoline.
pub(crate) enum Exit {
    /// The executor takes over from the thread state (the entry frame
    /// returned, a suspension, an uncaught error).
    End,
    /// Allocation crossed the GC check threshold; the host collects before
    /// stepping again.
    Gc,
    /// An async native waits on the host.
    Pending,
}

/// A dispatch target. Every one carries `#[rustc_align(32)]`: an unaligned
/// entry starves the fetch unit for the whole handler.
#[cfg(target_arch = "aarch64")]
pub(crate) type Handler = for<'gc> extern "rust-preserve-none" fn(
    insn: Slot,
    pc: *const Instruction,
    base: *mut Value<'gc>,
    rt: Context<'gc>,
    closure: Slot,
    thread: *mut ThreadState<'gc>,
) -> Exit;

#[cfg(not(target_arch = "aarch64"))]
pub(crate) type Handler = for<'gc> extern "rust-preserve-none" fn(
    insn: Slot,
    pc: *const Instruction,
    base: *mut Value<'gc>,
    rt: Context<'gc>,
    closure: Slot,
) -> Exit;

/// Handler addresses as header words: 32-byte aligned, so the low five bits
/// hold frame flags (`frame::flag`).
#[inline(always)]
pub(crate) fn handler_bits(h: Handler) -> u64 {
    h as usize as u64
}

/// # Safety
/// `bits` came from [`handler_bits`], flag bits allowed.
#[inline(always)]
pub(crate) unsafe fn handler_from_bits(bits: u64) -> Handler {
    unsafe { std::mem::transmute::<usize, Handler>((bits & !0x1f) as usize) }
}

/// Where a cold Rust routine (unwinder, native driver, coroutine switch)
/// tells its handler to go next; the handler `become`s it with `jump!`.
pub(crate) enum Jump<'gc> {
    /// Dispatch the instruction at `pc` in the Lua frame at `base`.
    Dispatch {
        pc: *const Instruction,
        base: *mut Value<'gc>,
    },
    /// Run `ret` with `nret` results at `values`; `base` is the finished
    /// frame's base, so `base - 4` is its header.
    Ret {
        ret: Handler,
        nret: usize,
        values: *mut Value<'gc>,
        base: *mut Value<'gc>,
    },
    /// `enter` the call whose header is written at `hdr`, `nargs` arguments
    /// above it; `base` is the calling Lua frame's window, or null when the
    /// caller is a native frame (already published).
    Enter {
        hdr: *mut Value<'gc>,
        nargs: usize,
        base: *mut Value<'gc>,
    },
    Exit(Exit),
}

/// Declares dispatch targets. `bind(...)` names the slot bindings the bodies
/// use (a module writes it once); each item is `kind fn name { body }` with
/// `kind` one of `op`, `cont`, `slow`, `entry`.
///
/// - `op`: `insn: Instruction`, `closure: LuaFn`, and `reg!`, `k!`, `upval!`
///   are available.
/// - `cont`: `nret: usize`, `values: *mut Value`; `base` is the finished
///   callee's; `closure` is an opaque `Slot` until `resume!`.
/// - `slow`: `insn` and `closure` are opaque `Slot` payloads; the instruction
///   is `insn_at!()` and the closure `frame::closure(base)`.
/// - `entry`: `closure` is the native closure; `insn` is the CALL or TAILCALL
///   instruction (a slow path may hand over a modified one), or the argument
///   count for `native_enter`.
///
/// Bodies end in `next!()`, a `tail!`, a `become`, a `jump!` or a `return`.
#[cfg(target_arch = "aarch64")]
macro_rules! handler {
    (bind($insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident, $nret:ident, $values:ident); $($items:tt)*) => {
        $crate::vm::abi::handler_impl!(@items [$insn, $pc, $base, $rt, $closure, $thread, $nret, $values] [$thread] $($items)*);
    };
}

#[cfg(not(target_arch = "aarch64"))]
macro_rules! handler {
    (bind($insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident, $nret:ident, $values:ident); $($items:tt)*) => {
        $crate::vm::abi::handler_impl!(@items [$insn, $pc, $base, $rt, $closure, $thread, $nret, $values] [] $($items)*);
    };
}

/// The architecture-independent part of [`handler!`]: `[$($th)?]` is the
/// thread parameter on aarch64 and empty on x86-64.
macro_rules! handler_impl {
    (@items [$($names:ident),*] [$($th:ident)?]) => {};
    (@items [$($names:ident),*] [$($th:ident)?]
     $(#[$m:meta])* $kind:ident fn $name:ident $body:block $($rest:tt)*) => {
        $crate::vm::abi::handler_impl!(@one $kind [$($names),*] [$($th)?] $(#[$m])* $name $body);
        $crate::vm::abi::handler_impl!(@items [$($names),*] [$($th)?] $($rest)*);
    };

    (@one $kind:ident [$insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident, $nret:ident, $values:ident]
     [$($th:ident)?] $(#[$m:meta])* $name:ident $body:block) => {
        $(#[$m])*
        #[inline(never)]
        #[rustc_align(32)]
        #[allow(unused_variables, unused_mut, unused_assignments, unused_macros, unreachable_code, unused_unsafe, clippy::macro_metavars_in_unsafe)]
        pub(crate) extern "rust-preserve-none" fn $name<'gc>(
            __insn: $crate::vm::abi::Slot,
            __pc: *const $crate::instruction::Instruction,
            __base: *mut $crate::env::value::Value<'gc>,
            __rt: $crate::lua::Context<'gc>,
            __closure: $crate::vm::abi::Slot,
            $($th: *mut $crate::env::thread::ThreadState<'gc>,)?
        ) -> $crate::vm::abi::Exit {
            let mut $pc = __pc;
            let mut $base = __base;
            let $rt = __rt;
            $(let mut $th = $th;)?
            $crate::vm::abi::handler_impl!(@bind $kind, __insn, __closure, $insn, $closure, $pc, $nret, $values);
            $crate::vm::abi::handler_impl!(@thread [$($th)?], $rt, $thread);
            $crate::vm::abi::handler_impl!(@slots $kind, $insn, $closure);
            $crate::vm::abi::handler_impl!(@body $kind, $insn, $pc, $base, $rt, $closure, $thread, $nret, $values, [$($th)?]);
            $body
        }
    };

    // --- per-kind bindings --------------------------------------------------

    (@bind op, $ri:ident, $rc:ident, $insn:ident, $closure:ident, $pc:ident, $nret:ident, $values:ident) => {
        let $insn: $crate::instruction::Instruction = $ri.as_insn();
        // SAFETY: an `op` target is only reached by dispatch, with the
        // running closure in the slot.
        let mut $closure: $crate::env::function::LuaFn<'gc> = unsafe { $rc.as_closure() };
    };
    (@bind cont, $ri:ident, $rc:ident, $insn:ident, $closure:ident, $pc:ident, $nret:ident, $values:ident) => {
        let $nret: usize = $ri.as_nret();
        let $values: *mut $crate::env::value::Value<'gc> = $pc as *mut _;
        let $insn: $crate::vm::abi::Slot = $ri;
        let mut $closure: $crate::vm::abi::Slot = $rc;
    };
    (@bind slow, $ri:ident, $rc:ident, $insn:ident, $closure:ident, $pc:ident, $nret:ident, $values:ident) => {
        let $insn: $crate::vm::abi::Slot = $ri;
        let mut $closure: $crate::vm::abi::Slot = $rc;
    };
    (@bind entry, $ri:ident, $rc:ident, $insn:ident, $closure:ident, $pc:ident, $nret:ident, $values:ident) => {
        let $insn: $crate::vm::abi::Slot = $ri;
        let mut $closure: $crate::vm::abi::Slot = $rc;
    };

    // The thread: the slot on aarch64, a load from the runtime on x86-64.
    (@thread [$th:ident], $rt:ident, $thread:ident) => {
        macro_rules! thread {
            () => {
                (unsafe { &mut *$th })
            };
        }
        /// Make `$$ts` the running thread, in the slot and the runtime.
        macro_rules! switch {
            ($$ts:expr) => {{
                let __ts: *mut $crate::env::thread::ThreadState<'gc> = $$ts;
                $th = __ts;
                $rt.set_thread(__ts);
            }};
        }
        /// Reload the thread slot after a Rust routine may have switched it.
        macro_rules! reload_thread {
            () => {{
                $th = $rt.thread_ptr();
            }};
        }
    };
    (@thread [], $rt:ident, $thread:ident) => {
        macro_rules! thread {
            () => {
                (unsafe { &mut *$rt.thread_ptr() })
            };
        }
        macro_rules! switch {
            ($$ts:expr) => {{
                let __ts: *mut $crate::env::thread::ThreadState<'gc> = $$ts;
                $rt.set_thread(__ts);
            }};
        }
        macro_rules! reload_thread {
            () => {{}};
        }
    };

    // How the running closure travels on to the next target.
    (@slots op, $insn:ident, $closure:ident) => {
        macro_rules! closure_slot {
            () => {
                $crate::vm::abi::Slot::closure($closure)
            };
        }
        macro_rules! set_closure {
            ($$f:expr) => {
                $closure = $$f
            };
        }
        macro_rules! insn_slot {
            () => {
                $crate::vm::abi::Slot::insn($insn)
            };
        }
    };
    (@slots $kind:ident, $insn:ident, $closure:ident) => {
        macro_rules! closure_slot {
            () => {
                $closure
            };
        }
        macro_rules! set_closure {
            ($$f:expr) => {
                $closure = $crate::vm::abi::Slot::closure($$f)
            };
        }
        macro_rules! insn_slot {
            () => {
                $insn
            };
        }
    };

    // --- body macros --------------------------------------------------------

    (@body $kind:ident, $insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident, $nret:ident, $values:ident, [$($th:ident)?]) => {
        /// Dispatch the instruction at `pc`.
        macro_rules! next {
            () => {{
                let __w: $crate::instruction::Instruction = unsafe { *$pc };
                $pc = unsafe { $pc.add(1) };
                let __h = $rt.handler(__w.opcode());
                become __h($crate::vm::abi::Slot::insn(__w), $pc, $base, $rt, closure_slot!() $(, $th)?)
            }};
        }

        /// Tail-call another target with these slots, some replaced.
        macro_rules! tail {
            ($$f:expr) => {
                become $$f(insn_slot!(), $pc, $base, $rt, closure_slot!() $(, $th)?)
            };
            ($$f:expr, insn = $$i:expr) => {
                become $$f($$i, $pc, $base, $rt, closure_slot!() $(, $th)?)
            };
            ($$f:expr, closure = $$c:expr) => {
                become $$f(insn_slot!(), $pc, $base, $rt, $$c $(, $th)?)
            };
            ($$f:expr, insn = $$i:expr, closure = $$c:expr) => {
                become $$f($$i, $pc, $base, $rt, $$c $(, $th)?)
            };
            ($$f:expr, pc = $$p:expr, insn = $$i:expr) => {
                become $$f($$i, $$p, $base, $rt, closure_slot!() $(, $th)?)
            };
            ($$f:expr, pc = $$p:expr, insn = $$i:expr, closure = $$c:expr) => {
                become $$f($$i, $$p, $base, $rt, $$c $(, $th)?)
            };
            ($$f:expr, pc = $$p:expr, insn = $$i:expr, base = $$b:expr) => {
                become $$f($$i, $$p, $$b, $rt, closure_slot!() $(, $th)?)
            };
        }

        /// Register `$$i` of the running frame, as a place.
        macro_rules! reg {
            ($$i:expr) => {
                (*unsafe { &mut *$base.add(($$i) as usize) })
            };
        }

        /// The instruction that reached this target, for `slow` kinds.
        macro_rules! insn_at {
            () => {
                unsafe { *$pc.sub(1) }
            };
        }

        /// Raise an `OpError` from this frame: the fault goes to the runtime,
        /// `impl_error` renders and unwinds it.
        macro_rules! raise {
            ($$kind:expr) => {{
                ::std::hint::cold_path();
                $rt.set_fault($$kind);
                tail!($crate::vm::unwind::impl_error)
            }};
        }

        /// Raise an `Error` from this frame.
        macro_rules! throw {
            ($$err:expr) => {
                raise!($crate::vm::unwind::OpError::Thrown($$err))
            };
        }

        /// Publish the top frame's state.
        macro_rules! sync {
            () => {{
                let __t = thread!();
                __t.top_base = $base;
                __t.top_pc = $pc;
            }};
        }

        /// Leave dispatch with `$$e`, the frame published.
        macro_rules! exit {
            ($$e:expr) => {{
                sync!();
                return $$e;
            }};
        }

        /// Before a store into `$$gc`: a gray object needs no
        /// barrier; any other goes to `barrier_retry`, which runs the barrier
        /// out of line and re-dispatches this instruction.
        macro_rules! barrier {
            ($$gc:expr) => {{
                let __g = $$gc;
                if ::std::hint::unlikely(!$crate::dmm::Gc::is_gray(__g)) {
                    tail!(
                        $crate::vm::ops::field::barrier_retry,
                        closure = $crate::vm::abi::Slot::from_raw($crate::dmm::Gc::erased_ptr(__g) as u64)
                    )
                }
            }};
        }

        /// After an allocation: leave for the collector if it is owed work.
        macro_rules! gc_check {
            () => {{
                if ::std::hint::unlikely($rt.gc_due()) {
                    exit!($crate::vm::abi::Exit::Gc);
                }
            }};
        }

        /// A conditional branch's exit: jump `$$off` past the next
        /// instruction if `$$cond`. The opaque asm in both arms keeps LLVM
        /// from if-converting the `pc` update into a select and from merging
        /// the two dispatch tails.
        macro_rules! branch {
            ($$cond:expr, $$off:expr) => {{
                if $$cond {
                    $pc = unsafe { $pc.offset(($$off) as isize) };
                    #[allow(clippy::pointers_in_nomem_asm_block)]
                    unsafe {
                        ::core::arch::asm!("/* {0} */", inout(reg) $pc, options(nomem, nostack, preserves_flags));
                    }
                    next!()
                } else {
                    #[allow(clippy::pointers_in_nomem_asm_block)]
                    unsafe {
                        ::core::arch::asm!("/* {0} */", inout(reg) $pc, options(nomem, nostack, preserves_flags));
                    }
                    next!()
                }
            }};
        }

        /// Unconditional jump by `$$off` instructions.
        macro_rules! jump_by {
            ($$off:expr) => {
                $pc = unsafe { $pc.offset(($$off) as isize) }
            };
        }

        /// The caller of the frame at `base`: its base and resume pc (header
        /// words 2 and 3, one pair load).
        macro_rules! caller {
            () => {
                unsafe { $crate::vm::frame::caller($base) }
            };
        }

        /// Continue in the Lua frame at `$$b` at `$$p` (a continuation
        /// resuming its caller).
        macro_rules! resume {
            ($$b:expr, $$p:expr) => {{
                $base = $$b;
                $pc = $$p;
                set_closure!(unsafe { $crate::vm::frame::closure($base) });
                next!()
            }};
        }

        /// Go where a cold routine says.
        macro_rules! jump {
            ($$j:expr) => {{
                let __j: $crate::vm::abi::Jump<'gc> = $$j;
                reload_thread!();
                match __j {
                    $crate::vm::abi::Jump::Dispatch { pc: __p, base: __b } => {
                        resume!(__b, __p)
                    }
                    $crate::vm::abi::Jump::Ret { ret: __r, nret: __n, values: __v, base: __b } => {
                        become __r(
                            $crate::vm::abi::Slot::nret(__n),
                            __v as *const $crate::instruction::Instruction,
                            __b,
                            $rt,
                            closure_slot!()
                            $(, $th)?
                        )
                    }
                    $crate::vm::abi::Jump::Enter { hdr: __h, nargs: __n, base: __b } => {
                        become $crate::vm::ops::call::enter(
                            $crate::vm::abi::Slot::nret(__n),
                            __h as *const $crate::instruction::Instruction,
                            __b,
                            $rt,
                            closure_slot!()
                            $(, $th)?
                        )
                    }
                    $crate::vm::abi::Jump::Exit(__e) => return __e,
                }
            }};
        }

        $crate::vm::abi::handler_impl!(@kind_body $kind, $insn, $pc, $base, $rt, $closure, $thread);
    };

    (@kind_body op, $insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident) => {
        /// Constant `$$i` of the running prototype.
        macro_rules! k {
            ($$i:expr) => {{
                debug_assert!((($$i) as usize) < $closure.proto.constants.len());
                unsafe { *$closure.constants.add(($$i) as usize) }
            }};
        }

        /// Upvalue slot `$$i`, or the field its descriptor says it holds.
        macro_rules! upval {
            ($$i:expr) => {{
                debug_assert!((($$i) as usize) < $closure.upvalues().len());
                unsafe { *$closure.upvalue_ptr().add(($$i) as usize) }
            }};
            (value $$i:expr) => {{
                let __s = upval!($$i);
                debug_assert!($closure.proto.upvalue_desc[($$i) as usize].by_value);
                unsafe { __s.value }
            }};
            (cell $$i:expr) => {{
                let __s = upval!($$i);
                debug_assert!(!$closure.proto.upvalue_desc[($$i) as usize].by_value);
                unsafe { __s.cell }
            }};
        }

        /// Stage the metamethod call `$$f($$args..)` above the window and
        /// enter it, `$$ret` taking its results.
        macro_rules! call_mm {
            ($$ret:expr, $$f:expr, [$$($$arg:expr),* $$(,)?]) => {{
                let __f: $crate::env::value::Value<'gc> = $$f;
                let __args: [$crate::env::value::Value<'gc>; _] = [$$($$arg),*];
                let __hdr = unsafe { $base.add($closure.max_stack_size as usize) };
                if ::std::hint::unlikely(unsafe { __hdr.add(4 + __args.len()) }.cast_const() > thread!().stack_end) {
                    tail!($crate::vm::ops::meta::stage_grow);
                }
                unsafe {
                    $crate::vm::frame::write_hdr(
                        __hdr,
                        __f.to_raw(),
                        $crate::vm::abi::handler_bits($$ret),
                        $base,
                        $pc,
                    );
                    let mut __p = __hdr.add(4);
                    for __a in __args {
                        __p.write(__a);
                        __p = __p.add(1);
                    }
                }
                tail!(
                    $crate::vm::ops::call::enter,
                    pc = __hdr as *const $crate::instruction::Instruction,
                    insn = $crate::vm::abi::Slot::nret(__args.len())
                )
            }};
        }
    };
    (@kind_body $kind:ident, $insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident) => {};
}

pub(crate) use {handler, handler_impl};
