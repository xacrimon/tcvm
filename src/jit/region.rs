//! A compiled region: the runtime's handle on native code, and the rules for
//! entering it.
//!
//! # Why entering is allowed to be this cheap
//!
//! Compiled code runs with Lua values live in machine registers, which the
//! collector cannot see (`ThreadState::trace` walks the *stack*). It also runs
//! under assumptions — `assume.no_mm` — that a metatable write can falsify at
//! any moment. Both are real hazards, and both are neutralized here by a single
//! restriction: [`compile`] refuses any region containing an op that `MAY_GC`,
//! which by [`Op::effects`] is exactly the set that can call, allocate, or
//! transfer control anywhere.
//!
//! So a region, once entered, runs to completion without the collector or the
//! mutator observing anything. That collapses two hard problems into one cheap
//! one:
//!
//!   - **Rooting.** No GC can happen inside, so registers need no anchoring.
//!   - **Invalidation.** No metatable can change inside, so the assumptions only
//!     have to hold *at entry* — [`Region::assumptions_hold`] rechecks them, and
//!     that check is sound for the whole execution. Watchpoints become an
//!     optimization (moving this off the entry path), not a correctness fix.
//!
//! Both properties expire the moment `Op::Call` is selectable, and both checks
//! are written so that they fail closed when it is.

use std::cell::Cell;

use crate::Context;
use crate::dmm::{Collect, Gc};
use crate::env::function::Prototype;
use crate::env::shape::MetamethodBits;
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::jit::backend::aarch64::{self, Status};
/// The compiled-code handle a `Region` owns. On aarch64 (macOS or Linux) it is a
/// [`CodeBlock`] sub-allocated from a shared segment; elsewhere it is a one-off
/// [`Code`] mapping. Both expose `entry()`.
#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
use crate::jit::backend::alloc::CodeBlock as Compiled;
#[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))))]
use crate::jit::backend::code::Code as Compiled;
use crate::jit::backend::isel;
use crate::jit::backend::regalloc;
use crate::jit::frontend::lower;
use crate::jit::ir::Func;
use crate::jit::ir::op::{Flags, Op};
use crate::jit::ir::pool::{ConstPool, ShapeRef};
use crate::jit::ir::ty::{Rep, Ty, TypeSet};

/// Native calls into a region: `(thread, frame base) -> packed status`.
///
/// The thread pointer is passed but unused — nothing a region currently compiles
/// to needs it. It is in the signature because the first op that does (a call)
/// will need it in a register, and adding an argument later would mean changing
/// every prologue.
type Entry = unsafe extern "C" fn(*mut (), *mut Value<'static>) -> u64;

/// Calls a prototype must see before we try to compile it.
///
/// Call counting, not back-edge counting: it needs no bytecode changes and it
/// finds exactly the functions a benchmark spends its time in. It cannot find a
/// hot loop inside a cold function — that needs OSR, and OSR needs entry at a pc
/// other than 0, which the frontend already supports and the entry protocol here
/// does not.
pub const HOT_CALL: u32 = 64;

/// Counter sentinel: compilation was tried and refused. Never retry.
const DECLINED: u32 = u32::MAX;

/// Why a prototype was not compiled. Recorded so we stop retrying, and so the
/// reason is inspectable rather than a silent absence.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Collect)]
#[collect(internal, require_static)]
pub enum Declined {
    /// The frontend refused the bytecode (varargs, MULTRET, TBC).
    Lower,
    /// Instruction selection has no pattern for some op in the region.
    Isel,
    /// The allocator met a constraint it does not implement. Unreachable from
    /// aarch64, which constrains nothing — but a decline is always a legal answer,
    /// and a target that does constrain something should not have to remember to
    /// add this.
    Regalloc,
    Encode,
    /// The region contains an op that can let the collector run, and nothing
    /// roots the values in its registers yet.
    MayGc,
}

/// One deopt exit, as the runtime needs it: where to resume, and how often we
/// have landed here.
struct ExitInfo {
    pc: u32,
    /// Hot exits are the signal that this region's entry context was wrong.
    /// Nothing consumes it yet; recording it is free and the alternative is
    /// discovering later that we never had the data.
    count: Cell<u32>,
}

/// Compiled native code for one `Prototype`, entered at pc 0.
///
/// The `pool` is not dead weight: [`aarch64::encode`] bakes shape *box
/// addresses* into the guard sequences, so the region is only valid while those
/// shapes are alive. Holding the pool is what keeps them alive.
#[derive(Collect)]
#[collect(internal, unsafe_drop)]
pub struct Region<'gc> {
    #[collect(require_static)]
    code: Compiled,
    pool: ConstPool<'gc>,
    #[collect(require_static)]
    exits: Box<[ExitInfo]>,

    /// Lua registers the entry block reads, and the type each was compiled to
    /// assume. The prologue does *not* check these — see `isel::prologue` — so
    /// this is the caller's obligation, discharged by [`Region::accepts`].
    #[collect(require_static)]
    entry: Box<[(u8, TypeSet)]>,

    /// The `assume.no_mm` set: shapes whose metatables must not have gained
    /// these metamethods. Checked at entry; see the module docs for why that is
    /// enough.
    #[collect(require_static)]
    assumptions: Box<[(ShapeRef, MetamethodBits)]>,

    /// Entries and guard failures.
    ///
    /// Not diagnostics for their own sake: without them a region that is never
    /// entered — because the entry check rejects every call — is indistinguishable
    /// from one that runs perfectly, since both produce the interpreter's answer.
    #[collect(require_static)]
    pub entries: Cell<u64>,
    #[collect(require_static)]
    pub deopts: Cell<u64>,
}

impl<'gc> Region<'gc> {
    /// Do the values in this frame match the types the region was compiled for?
    ///
    /// `regs` is the frame's register window, so `regs[r]` is Lua register `r`.
    pub fn accepts(&self, regs: &[Value<'gc>]) -> bool {
        self.entry
            .iter()
            .all(|&(r, set)| set.contains(TypeSet::of_value(regs[r as usize])))
    }

    /// Have any of the metatables we specialized against sprouted a metamethod?
    pub fn assumptions_hold(&self) -> bool {
        self.assumptions
            .iter()
            .all(|&(s, bits)| !self.pool.shape(s).has_mm(bits))
    }

    pub fn resume_pc(&self, exit: u32) -> usize {
        let e = &self.exits[exit as usize];
        e.count.set(e.count.get().saturating_add(1));
        e.pc as usize
    }

    /// Run the region over the frame based at `base`.
    ///
    /// # Safety
    ///
    /// `base` must point at a register window of at least the prototype's
    /// `max_stack_size` values, and [`Region::accepts`] and
    /// [`Region::assumptions_hold`] must both have just returned true for it.
    /// The code reads and writes those slots directly.
    pub unsafe fn enter(&self, base: *mut Value<'gc>) -> Status {
        let f: Entry = unsafe { std::mem::transmute(self.code.entry()) };
        // The region's `'gc` and the stack's are the same lifetime; the cast is
        // only to launder it through a `#[repr(C)]` signature.
        Status::unpack(unsafe { f(std::ptr::null_mut(), base.cast()) })
    }
}

/// `TCVM_JIT_LOG=1` reports every compile decision. A JIT that quietly declines
/// everything and a JIT that works are the same program from the outside; this is
/// how you tell them apart without a debugger.
fn log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TCVM_JIT_LOG").is_some())
}

/// `TCVM_JIT_OFF=1` keeps everything interpreted. The point is to A/B one binary:
/// any difference in output between a run with this set and a run without it is a
/// JIT bug, and any difference in time is the JIT's actual worth.
fn jit_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("TCVM_JIT_OFF").is_some())
}

/// What the interpreter should do with a Lua frame it has just pushed.
pub enum Outcome {
    /// There is no native code for this frame. Interpret it.
    Interpret,
    /// The region ran to a Lua `return`, leaving `nret` values at `base + 0`.
    Returned(usize),
    /// A guard failed. The exit stub has already written every live Lua register
    /// back to the frame, so the frame is exactly what the interpreter would
    /// have built by running the bytecode up to this pc. Resume there.
    Deopt(usize),
}

/// Called by `op_call` on every Lua-to-Lua call, with the callee's frame already
/// pushed and its registers in place.
///
/// This is the only door into compiled code.
pub fn on_call<'gc>(
    ctx: Context<'gc>,
    thread: &mut ThreadState<'gc>,
    proto: Gc<'gc, Prototype<'gc>>,
    base: usize,
) -> Outcome {
    let n = proto.jit_calls.get();
    if n == DECLINED || jit_disabled() {
        return Outcome::Interpret;
    }
    if n < HOT_CALL {
        proto.jit_calls.set(n + 1);
        if n + 1 < HOT_CALL {
            return Outcome::Interpret;
        }
        // Just crossed. Compile against *this* call's argument types, then fall
        // through and run what we built — the arguments we specialized on are
        // still sitting in the frame, so the first execution is free of charge.
        let args = &thread.stack[base..base + proto.num_params as usize];
        let result = compile(ctx, proto, args);
        if log_enabled() {
            // A `Prototype` carries no name and no line, so it is identified by
            // the only things it has: how many parameters and how much bytecode.
            let id = format!("fn/{}p/{}i", proto.num_params, proto.code.len());
            match &result {
                Ok(_) => eprintln!("[jit] compiled {id}"),
                Err(e) => eprintln!("[jit] declined {id}: {e:?}"),
            }
        }
        match result {
            Ok(r) => *proto.jit.borrow_mut(ctx.mutation()) = Some(r),
            Err(_) => {
                proto.jit_calls.set(DECLINED);
                return Outcome::Interpret;
            }
        }
    }

    let Some(region) = *proto.jit.borrow() else {
        return Outcome::Interpret;
    };

    // Two obligations the compiled code does not discharge for itself: its entry
    // block assumes the parameter types it was specialized for, and its guards
    // assume no metamethod has appeared behind the shapes it baked in. Failing
    // either is not an error — it just means this call gets interpreted.
    if !region.accepts(&thread.stack[base..]) || !region.assumptions_hold() {
        return Outcome::Interpret;
    }

    let frame = unsafe { thread.stack.as_mut_ptr().add(base) };
    region.entries.set(region.entries.get() + 1);
    match unsafe { region.enter(frame) } {
        Status::Return(nret) => Outcome::Returned(nret as usize),
        Status::Deopt(exit) => {
            region.deopts.set(region.deopts.get() + 1);
            Outcome::Deopt(region.resume_pc(exit))
        }
    }
}

/// Compile `proto`, entered at pc 0 with `args` as its parameters.
///
/// `args` is the live argument window, used purely as type feedback: the region
/// is specialized to the types actually passed, and [`Region::accepts`] refuses
/// any later call that does not match.
pub fn compile<'gc>(
    ctx: Context<'gc>,
    proto: Gc<'gc, Prototype<'gc>>,
    args: &[Value<'gc>],
) -> Result<Gc<'gc, Region<'gc>>, Declined> {
    let entry_tys: Vec<Ty> = (0..proto.num_params as usize)
        .map(|i| {
            let v = args.get(i).copied().unwrap_or_else(Value::nil);
            Ty::new(Rep::Val, TypeSet::of_value(v))
        })
        .collect();

    let func = lower::lower(proto, 0, entry_tys.clone()).map_err(|_| Declined::Lower)?;

    if let Some(_op) = may_gc(&func) {
        return Err(Declined::MayGc);
    }

    let m = isel::select(&func).map_err(|_| Declined::Isel)?;
    let ra = regalloc::linear_scan(&m, &aarch64::machine_env()).map_err(|_| Declined::Regalloc)?;
    let words = aarch64::encode(&m, &func.pool, &ra).map_err(|_| Declined::Encode)?;
    // Placing the encoded words is where the two targets diverge: the segment
    // allocator on aarch64 (macOS or Linux), a per-function mapping elsewhere.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let code = ctx
        .code_alloc()
        .alloc(&words)
        .map_err(|_| Declined::Encode)?;
    #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))))]
    let code = Compiled::from_words(&words).map_err(|_| Declined::Encode)?;

    let entry: Box<[(u8, TypeSet)]> = func
        .entry_regs
        .iter()
        .map(|&r| (r, entry_tys.get(r as usize).map_or(TypeSet::ANY, |t| t.set)))
        .collect();

    let exits = (0..func.num_exits() as u32)
        .map(|e| ExitInfo {
            pc: func
                .frame_state(func.exit(crate::jit::ir::ExitRef(e)).fs)
                .pc,
            count: Cell::new(0),
        })
        .collect();

    let assumptions = assumptions_of(&func);

    Ok(Gc::new(
        ctx.mutation(),
        Region {
            code,
            pool: func.pool,
            exits,
            entry,
            assumptions,
            entries: Cell::new(0),
            deopts: Cell::new(0),
        },
    ))
}

/// The first op in `func` that can let the collector run, if any.
///
/// This is the check the whole entry protocol rests on, so it asks the effect
/// system rather than enumerating opcodes: a new op that can call or allocate is
/// rejected the day it is added, not the day someone remembers this file.
fn may_gc(func: &Func<'_>) -> Option<Op> {
    func.blocks()
        .flat_map(|b| func.block(b).insts.clone())
        .map(|i| func.inst(i).op)
        .find(|op| op.effects().flags.contains(Flags::MAY_GC))
}

fn assumptions_of(func: &Func<'_>) -> Box<[(ShapeRef, MetamethodBits)]> {
    let mut out: Vec<(ShapeRef, MetamethodBits)> = Vec::new();
    for b in func.blocks() {
        for &i in &func.block(b).insts {
            if let Op::AssumeNoMm(s, bits) = func.inst(i).op {
                match out.iter_mut().find(|(t, _)| *t == s) {
                    Some((_, acc)) => *acc |= bits,
                    None => out.push((s, bits)),
                }
            }
        }
    }
    out.into_boxed_slice()
}
