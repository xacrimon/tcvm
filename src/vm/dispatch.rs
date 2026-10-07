//! The dispatch tables, the runtime cells behind a `Context`, and the
//! trampoline from the executor into dispatch.

use std::cell::Cell;

use crate::dmm::metrics::Metrics;
use crate::dmm::{Collect, Mutation, Trace};
use crate::env::Thread;
use crate::env::thread::{ThreadState, ThreadStatus};
use crate::instruction::{Instruction, Op};
use crate::lua::Context;
use crate::vm::abi::{Exit, Handler, Jump, Slot};
use crate::vm::ops::{arith, call, compare, control, data, field, table};
use crate::vm::unwind::OpError;
use crate::vm::{coro, frame, native};

/// The handler of every opcode, the unused bytes trapping.
pub(crate) const TABLE: [Handler; 256] = {
    let ops: [Handler; Op::COUNT] = Op::table([
        (Op::MOVE, data::op_move),
        (Op::LOAD, data::op_load),
        (Op::LOADI, data::op_loadi),
        (Op::LOADNIL, data::op_loadnil),
        (Op::LFALSESKIP, data::op_lfalseskip),
        (Op::GETUPVAL, data::op_getupval),
        (Op::GETUPVAL_REF, data::op_getupval_ref),
        (Op::SETUPVAL, data::op_setupval),
        (Op::GETTABUP, field::op_gettabup),
        (Op::GETTABUP_REF, field::op_gettabup_ref),
        (Op::SETTABUP, field::op_settabup),
        (Op::SETTABUP_REF, field::op_settabup_ref),
        (Op::GETTABLE, table::op_gettable),
        (Op::SETTABLE, table::op_settable),
        (Op::GETFIELD, field::op_getfield),
        (Op::SETFIELD, field::op_setfield),
        (Op::SELF, field::op_self),
        (Op::NEWTABLE, table::op_newtable),
        (Op::ADD, arith::op_add),
        (Op::SUB, arith::op_sub),
        (Op::MUL, arith::op_mul),
        (Op::MOD, arith::op_mod),
        (Op::POW, arith::op_pow),
        (Op::DIV, arith::op_div),
        (Op::IDIV, arith::op_idiv),
        (Op::BAND, arith::op_band),
        (Op::BOR, arith::op_bor),
        (Op::BXOR, arith::op_bxor),
        (Op::SHL, arith::op_shl),
        (Op::SHR, arith::op_shr),
        (Op::UNM, arith::op_unm),
        (Op::BNOT, arith::op_bnot),
        (Op::NOT, arith::op_not),
        (Op::LEN, table::op_len),
        (Op::CONCAT, table::op_concat),
        (Op::CLOSE, control::op_close),
        (Op::TBC, control::op_tbc),
        (Op::JMP, control::op_jmp),
        (Op::JEQ, compare::op_jeq),
        (Op::JNEQ, compare::op_jneq),
        (Op::JLT, compare::op_jlt),
        (Op::JNLT, compare::op_jnlt),
        (Op::JLE, compare::op_jle),
        (Op::JNLE, compare::op_jnle),
        (Op::JEQI, compare::op_jeqi),
        (Op::JNEQI, compare::op_jneqi),
        (Op::JLTI, compare::op_jlti),
        (Op::JNLTI, compare::op_jnlti),
        (Op::JLEI, compare::op_jlei),
        (Op::JNLEI, compare::op_jnlei),
        (Op::JGTI, compare::op_jgti),
        (Op::JNGTI, compare::op_jngti),
        (Op::JGEI, compare::op_jgei),
        (Op::JNGEI, compare::op_jngei),
        (Op::JEQS, compare::op_jeqs),
        (Op::JNEQS, compare::op_jneqs),
        (Op::JT, compare::op_jt),
        (Op::JF, compare::op_jf),
        (Op::JTSET, compare::op_jtset),
        (Op::JFSET, compare::op_jfset),
        (Op::CALL, call::op_call),
        (Op::CALL_R0, call::op_call_r0),
        (Op::CALL_R1, call::op_call_r1),
        (Op::TAILCALL, call::op_tailcall),
        (Op::RETURN, call::op_return),
        (Op::RETURN0, call::op_return0),
        (Op::RETURN1, call::op_return1),
        (Op::FORLOOP, control::op_forloop),
        (Op::FORPREP, control::op_forprep),
        (Op::TFORPREP, control::op_tforprep),
        (Op::TFORCALL, control::op_tforcall),
        (Op::TFORLOOP, control::op_tforloop),
        (Op::SETLIST, table::op_setlist),
        (Op::CLOSURE, control::op_closure),
        (Op::VARARG, control::op_vararg),
        (Op::VARARGGET, control::op_varargget),
        (Op::VARARGPREP, control::op_varargprep),
        (Op::ERRNNIL, control::op_errnnil),
        (Op::NOP, control::op_nop),
        (Op::STOP, control::op_stop),
        (Op::ADDI, arith::op_addi),
        (Op::SUBI, arith::op_subi),
        (Op::MULI, arith::op_muli),
        (Op::MODI, arith::op_modi),
        (Op::POWI, arith::op_powi),
        (Op::DIVI, arith::op_divi),
        (Op::IDIVI, arith::op_idivi),
        (Op::BANDI, arith::op_bandi),
        (Op::BORI, arith::op_bori),
        (Op::BXORI, arith::op_bxori),
        (Op::SHLI, arith::op_shli),
        (Op::SHRI, arith::op_shri),
        (Op::RSUBI, arith::op_rsubi),
        (Op::RMODI, arith::op_rmodi),
        (Op::RPOWI, arith::op_rpowi),
        (Op::RDIVI, arith::op_rdivi),
        (Op::RIDIVI, arith::op_ridivi),
        (Op::RSHLI, arith::op_rshli),
        (Op::RSHRI, arith::op_rshri),
        (Op::GETFIELD_OWN, field::getfield_own),
        (Op::GETFIELD_ABSENT, field::getfield_absent),
        (Op::GETFIELD_PROTO, field::getfield_proto),
        (Op::GETTABUP_OWN, field::gettabup_own),
        (Op::GETTABUP_ABSENT, field::gettabup_absent),
        (Op::GETTABUP_PROTO, field::gettabup_proto),
        (Op::SELF_OWN, field::self_own),
        (Op::SELF_ABSENT, field::self_absent),
        (Op::SELF_PROTO, field::self_proto),
        (Op::SETFIELD_OWN, field::setfield_own),
        (Op::SETFIELD_TRANS, field::setfield_trans),
        (Op::SETTABUP_OWN, field::settabup_own),
        (Op::SETTABUP_TRANS, field::settabup_trans),
        (Op::SETFIELD_ABSENT, field::setfield_absent),
        (Op::SETTABUP_ABSENT, field::settabup_absent),
        (Op::ADD_NUM, arith::op_add_num),
        (Op::SUB_NUM, arith::op_sub_num),
        (Op::MUL_NUM, arith::op_mul_num),
        (Op::MOD_NUM, arith::op_mod_num),
        (Op::POW_NUM, arith::op_pow_num),
        (Op::DIV_NUM, arith::op_div_num),
        (Op::IDIV_NUM, arith::op_idiv_num),
        (Op::ARITH_MM, arith::op_arith_mm),
        (Op::ARITH_MM_R, arith::op_arith_mm_r),
        (Op::ARITH_MMI, arith::op_arith_mmi),
    ]);
    let mut t: [Handler; 256] = [control::op_invalid; 256];
    let mut i = 0;
    while i < Op::COUNT {
        t[i] = ops[i];
        i += 1;
    }
    t
};

/// CALL's continuation by its `c` operand (0 = MULTRET).
pub(crate) const RETS: [Handler; 256] = {
    let mut t: [Handler; 256] = [call::ret_call; 256];
    t[1] = call::ret_call0;
    t[2] = call::ret_call1;
    t[3] = call::ret_call2;
    t
};

/// The cells dispatch keeps in the runtime, behind the two tables.
pub(crate) struct Runtime<'gc> {
    /// The running thread's state; coroutine switches update it.
    pub(crate) thread: Cell<*mut ThreadState<'gc>>,
    /// The arena's mutation context, refreshed by `Lua::enter`.
    pub(crate) mutation: Cell<*const Mutation<'gc>>,
    /// The allocation counters `gc_check!` compares; stable for the arena's life.
    pub(crate) metrics: *const Metrics,
    /// The fault `raise!` leaves for `impl_error`.
    pub(crate) fault: Cell<Option<OpError<'gc>>>,
}

// SAFETY: holds no `Gc` pointer that outlives a `Lua::enter`: the fault is
// consumed by the raise that set it, the rest are plain pointers.
unsafe impl<'gc> Collect<'gc> for Runtime<'gc> {
    const NEEDS_TRACE: bool = false;
    fn trace<T: Trace<'gc>>(&self, _cc: &mut T) {}
}

impl<'gc> Runtime<'gc> {
    pub(crate) fn new(metrics: &Metrics) -> Self {
        // The unwinder tells `pcall` and `xpcall` catch points apart by their
        // continuations' addresses, which identical bodies would fold.
        assert!(
            crate::vm::native::ret_pcall as *const () != crate::vm::native::ret_xpcall as *const (),
            "ret_pcall and ret_xpcall were merged"
        );
        Runtime {
            thread: Cell::new(std::ptr::null_mut()),
            mutation: Cell::new(std::ptr::null()),
            metrics,
            fault: Cell::new(None),
        }
    }
}

/// Call `h` as the trampoline does: a plain call, which dispatch leaves by
/// returning its `Exit`.
#[inline(always)]
fn call_handler<'gc>(
    h: Handler,
    insn: Slot,
    pc: *const Instruction,
    base: *mut crate::env::value::Value<'gc>,
    rt: Context<'gc>,
    closure: Slot,
    thread: *mut ThreadState<'gc>,
) -> Exit {
    #[cfg(target_arch = "aarch64")]
    {
        h(insn, pc, base, rt, closure, thread)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = thread;
        h(insn, pc, base, rt, closure)
    }
}

/// Go where a cold routine said, from outside dispatch.
fn run<'gc>(ctx: Context<'gc>, j: Jump<'gc>) -> Exit {
    let ts = ctx.thread_ptr();
    match j {
        Jump::Dispatch { pc, base } => {
            let closure = unsafe { frame::closure(base) };
            call_handler(
                control::op_nop,
                Slot::insn(Instruction::nop()),
                pc,
                base,
                ctx,
                Slot::closure(closure),
                ts,
            )
        }
        Jump::Ret {
            ret,
            nret,
            values,
            base,
        } => call_handler(
            ret,
            Slot::nret(nret),
            values as *const Instruction,
            base,
            ctx,
            Slot::from_raw(0),
            ts,
        ),
        Jump::Enter { hdr, nargs, base } => call_handler(
            call::enter,
            Slot::nret(nargs),
            hdr as *const Instruction,
            base,
            ctx,
            Slot::from_raw(0),
            ts,
        ),
        Jump::Exit(e) => e,
    }
}

/// Run `thread` from where it stands: its seeded call, the values a
/// host resume delivered, an async native waiting on the host, or the
/// published top Lua frame.
#[inline(never)]
pub(crate) fn enter<'gc>(ctx: Context<'gc>, thread: Thread<'gc>) -> Exit {
    // SAFETY: the executor runs one thread at a time and holds no borrow of
    // it; dispatch may switch to coroutines it resumes the same way.
    let mut ts: &mut ThreadState<'gc> = unsafe { thread.state_mut(ctx.mutation()) };
    ctx.set_thread(ts);
    if !ts.started {
        let nargs = call::seed_header(ts, call::ret_exit);
        ts.status = ThreadStatus::Normal;
        let hdr = ts.slot_ptr(0);
        return run(
            ctx,
            Jump::Enter {
                hdr,
                nargs,
                base: std::ptr::null_mut(),
            },
        );
    }
    if let Some(yb) = ts.yield_bottom.take() {
        ts.status = ThreadStatus::Normal;
        let j = coro::deliver(ctx, &mut ts, yb);
        return run(ctx, j);
    }
    debug_assert!(
        !ts.top_base.is_null(),
        "dispatch of a thread with no frames"
    );
    if ts.top_is_native() {
        let win = ts.top_base_index();
        let j = native::repoll(ctx, &mut ts, win);
        return run(ctx, j);
    }
    let (pc, base) = (ts.top_pc, ts.top_base);
    run(ctx, Jump::Dispatch { pc, base })
}
