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
        (Op::GETTABUP, field::get_slow),
        (Op::GETTABUP_REF, field::op_gettabup_ref),
        (Op::SETTABUP, field::set_slow),
        (Op::SETTABUP_REF, field::op_settabup_ref),
        (Op::GETTABLE, table::op_gettable),
        (Op::SETTABLE, table::op_settable),
        (Op::GETFIELD, field::get_slow),
        (Op::SETFIELD, field::set_slow),
        (Op::SELF, field::get_slow),
        (Op::NEWTABLE, table::op_newtable),
        (Op::ADD, arith::arith_generic),
        (Op::SUB, arith::arith_generic),
        (Op::MUL, arith::arith_generic),
        (Op::MOD, arith::arith_generic),
        (Op::POW, arith::arith_generic),
        (Op::DIV, arith::arith_generic),
        (Op::IDIV, arith::arith_generic),
        (Op::BAND, arith::arith_generic),
        (Op::BOR, arith::arith_generic),
        (Op::BXOR, arith::arith_generic),
        (Op::SHL, arith::arith_generic),
        (Op::SHR, arith::arith_generic),
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
        (Op::ADDI, arith::arith_generic),
        (Op::SUBI, arith::arith_generic),
        (Op::MULI, arith::arith_generic),
        (Op::MODI, arith::arith_generic),
        (Op::POWI, arith::arith_generic),
        (Op::DIVI, arith::arith_generic),
        (Op::IDIVI, arith::arith_generic),
        (Op::BANDI, arith::arith_generic),
        (Op::BORI, arith::arith_generic),
        (Op::BXORI, arith::arith_generic),
        (Op::SHLI, arith::arith_generic),
        (Op::SHRI, arith::arith_generic),
        (Op::RSUBI, arith::arith_generic),
        (Op::RMODI, arith::arith_generic),
        (Op::RPOWI, arith::arith_generic),
        (Op::RDIVI, arith::arith_generic),
        (Op::RIDIVI, arith::arith_generic),
        (Op::RSHLI, arith::arith_generic),
        (Op::RSHRI, arith::arith_generic),
        (Op::GETFIELD_INL, field::getfield_inl),
        (Op::GETFIELD_AUX, field::getfield_aux),
        (Op::GETFIELD_ABSENT, field::getfield_absent),
        (Op::GETFIELD_PROTO, field::getfield_proto),
        (Op::GETTABUP_INL, field::gettabup_inl),
        (Op::GETTABUP_AUX, field::gettabup_aux),
        (Op::GETTABUP_ABSENT, field::gettabup_absent),
        (Op::GETTABUP_PROTO, field::gettabup_proto),
        (Op::SELF_INL, field::self_inl),
        (Op::SELF_AUX, field::self_aux),
        (Op::SELF_ABSENT, field::self_absent),
        (Op::SELF_PROTO, field::self_proto),
        (Op::SETFIELD_INL, field::setfield_inl),
        (Op::SETFIELD_AUX, field::setfield_aux),
        (Op::SETFIELD_TRANS, field::setfield_trans),
        (Op::SETFIELD_ABSENT, field::setfield_absent),
        (Op::SETTABUP_INL, field::settabup_inl),
        (Op::SETTABUP_AUX, field::settabup_aux),
        (Op::SETTABUP_TRANS, field::settabup_trans),
        (Op::SETTABUP_ABSENT, field::settabup_absent),
        (Op::ARITH_MM, arith::op_arith_mm),
        (Op::ARITH_MM_R, arith::op_arith_mm_r),
        (Op::ARITH_MMI, arith::op_arith_mmi),
        (Op::ADD_II, arith::op_add_ii),
        (Op::SUB_II, arith::op_sub_ii),
        (Op::MUL_II, arith::op_mul_ii),
        (Op::MOD_II, arith::op_mod_ii),
        (Op::IDIV_II, arith::op_idiv_ii),
        (Op::DIV_II, arith::op_div_ii),
        (Op::ADD_FF, arith::op_add_ff),
        (Op::SUB_FF, arith::op_sub_ff),
        (Op::MUL_FF, arith::op_mul_ff),
        (Op::MOD_FF, arith::op_mod_ff),
        (Op::POW_FF, arith::op_pow_ff),
        (Op::DIV_FF, arith::op_div_ff),
        (Op::IDIV_FF, arith::op_idiv_ff),
        (Op::ADD_IF, arith::op_add_if),
        (Op::SUB_IF, arith::op_sub_if),
        (Op::MUL_IF, arith::op_mul_if),
        (Op::DIV_IF, arith::op_div_if),
        (Op::ADD_FI, arith::op_add_fi),
        (Op::SUB_FI, arith::op_sub_fi),
        (Op::MUL_FI, arith::op_mul_fi),
        (Op::DIV_FI, arith::op_div_fi),
        (Op::BAND_II, arith::op_band_ii),
        (Op::BOR_II, arith::op_bor_ii),
        (Op::BXOR_II, arith::op_bxor_ii),
        (Op::SHL_II, arith::op_shl_ii),
        (Op::SHR_II, arith::op_shr_ii),
        (Op::ADDI_I, arith::op_addi_i),
        (Op::SUBI_I, arith::op_subi_i),
        (Op::MULI_I, arith::op_muli_i),
        (Op::MODI_I, arith::op_modi_i),
        (Op::IDIVI_I, arith::op_idivi_i),
        (Op::RSUBI_I, arith::op_rsubi_i),
        (Op::ADDI_F, arith::op_addi_f),
        (Op::SUBI_F, arith::op_subi_f),
        (Op::MULI_F, arith::op_muli_f),
        (Op::MODI_F, arith::op_modi_f),
        (Op::IDIVI_F, arith::op_idivi_f),
        (Op::RSUBI_F, arith::op_rsubi_f),
        (Op::POWI_F, arith::op_powi_f),
        (Op::DIVI_F, arith::op_divi_f),
        (Op::RDIVI_F, arith::op_rdivi_f),
        (Op::ADDI_IF, arith::op_addi_if),
        (Op::SUBI_IF, arith::op_subi_if),
        (Op::MULI_IF, arith::op_muli_if),
        (Op::DIVI_IF, arith::op_divi_if),
        (Op::RDIVI_IF, arith::op_rdivi_if),
        (Op::BANDI_I, arith::op_bandi_i),
        (Op::BORI_I, arith::op_bori_i),
        (Op::BXORI_I, arith::op_bxori_i),
        (Op::SHLI_I, arith::op_shli_i),
        (Op::SHRI_I, arith::op_shri_i),
        (Op::POW_II, arith::op_pow_ii),
        (Op::POWI_IF, arith::op_powi_if),
        (Op::RPOWI_IF, arith::op_rpowi_if),
        (Op::ADD_NN, arith::op_add_nn),
        (Op::SUB_NN, arith::op_sub_nn),
        (Op::MUL_NN, arith::op_mul_nn),
        (Op::MOD_NN, arith::op_mod_nn),
        (Op::POW_NN, arith::op_pow_nn),
        (Op::DIV_NN, arith::op_div_nn),
        (Op::IDIV_NN, arith::op_idiv_nn),
        (Op::JLT_II, compare::op_jlt_ii),
        (Op::JNLT_II, compare::op_jnlt_ii),
        (Op::JLE_II, compare::op_jle_ii),
        (Op::JNLE_II, compare::op_jnle_ii),
        (Op::JEQ_II, compare::op_jeq_ii),
        (Op::JNEQ_II, compare::op_jneq_ii),
        (Op::JLTI_F, compare::op_jlti_f),
        (Op::JNLTI_F, compare::op_jnlti_f),
        (Op::JLEI_F, compare::op_jlei_f),
        (Op::JNLEI_F, compare::op_jnlei_f),
        (Op::JGTI_F, compare::op_jgti_f),
        (Op::JNGTI_F, compare::op_jngti_f),
        (Op::JGEI_F, compare::op_jgei_f),
        (Op::JNGEI_F, compare::op_jngei_f),
        (Op::FORLOOP_I, control::op_forloop_i),
        (Op::FORLOOP_F, control::op_forloop_f),
        (Op::TFORCALL_NEXT, control::op_tforcall_next),
        (Op::TFORCALL_IPAIRS, control::op_tforcall_ipairs),
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
