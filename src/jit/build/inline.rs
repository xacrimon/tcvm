//! Inlining (E1): a call of a known small Lua function becomes its body,
//! built by a builder of its own into the same function. The body's
//! snapshots name its frame and the callers' registers at the call, so an
//! exit rebuilds both; its returns jump to a block taking the results.
//! A body that needs its frame on the stack (a call, its slots, a cell
//! upvalue) is not inlined. The callee is what the call would guard
//! (`intrinsics::callee_of`): a closure, whose upvalues are constants, or
//! one of a prototype, whose upvalues are loaded from it.

use super::intrinsics::{Callee, callee_of};
use super::{Builder, Inline, SlotState};
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::cfg::Cfg;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, Ty};
use crate::jit::ir::{Func, InlineFrame, Val};

/// Callees longer than this many bytecodes are called. Fitted to two
/// programs: ray's `sphere_distance` (41) inlined runs 7% fewer cycles,
/// bounce's `Ball:bounce` (49), copied again by loop peeling, 10% more (on
/// 3% fewer instructions: instruction delivery stalls).
const INLINE_LIMIT: usize = 48;
/// Inlined frames nest at most this deep.
const MAX_DEPTH: u8 = 3;
/// No inlining grows a region past this many instructions: compile time
/// grows faster than the region (json at 3000: 81 ms of compiles, 22 without
/// inlining).
const MAX_INSTS: usize = 1500;

/// Inline the CALL at `pc` of `callee`; whether it did, the current block
/// then the one after the call.
pub(super) fn try_inline(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, callee: Val) -> bool {
    if !b.ctx.jit().config.inline {
        return false;
    }
    let (a, nb, c) = (insn.a(), insn.b(), insn.c());
    if nb == 0 || c == 0 || insn.op() == BcOp::TAILCALL {
        return false;
    }
    let depth = b.inline.as_ref().map_or(0, |i| i.depth);
    if depth >= MAX_DEPTH || b.f.insts.len() > MAX_INSTS {
        return false;
    }
    // A closure of a prototype that made several closures stands for any
    // of them: only its prototype's facts are used.
    let (lf, dynamic) = match callee_of(b, pc, callee) {
        Callee::Const(f) | Callee::Same(f) => match f.as_lua() {
            Some(lf) => (lf, false),
            None => return false,
        },
        Callee::Proto(_, Some(lf)) => (lf, true),
        Callee::Proto(_, None) | Callee::Unknown => return false,
    };
    let proto = lf.proto;
    let base = b.reg_base() + a as usize + 4;
    if proto.is_vararg
        || proto.code.len() > INLINE_LIMIT
        || crate::dmm::Gc::ptr_eq(proto, b.proto)
        || base + proto.max_stack_size as usize > 255
    {
        return false;
    }
    let mut code: Vec<Instruction> = proto
        .code
        .iter()
        .map(|i| crate::jit::compile::resolve_original(lf, i))
        .collect();
    if code.iter().any(|i| needs_frame(i.op())) {
        return false;
    }
    crate::jit::compile::canonical_back_edges(lf, &mut code);
    let Ok(cfg) = Cfg::build(&proto, code, 0) else {
        return false;
    };
    if cfg.never_ran[cfg.block_of[0] as usize] {
        return false;
    }
    // What the callers hold at the call, and the callee's arguments.
    let live = b
        .cfg
        .live_in
        .get(pc as usize + 1)
        .copied()
        .unwrap_or_default();
    let mut outer = b.call_entries(live, a);
    let regs: Vec<usize> = b
        .exit_regs(live)
        .iter()
        .filter(|&r| r < a as usize && r < b.nregs)
        .collect();
    let kept: Vec<(u8, Val)> = regs
        .into_iter()
        .map(|r| (r as u8, b.reg(r as u8)))
        .collect();
    let args: Vec<Val> = (0..nb - 1).map(|k| b.reg(a + 4 + k)).collect();
    let cv = b.boxed(callee);
    // The guard's operand: the closure, or the prototype (`Func::protos`).
    let (guard, func, closure) = if dynamic {
        let k = b.f.intern_proto(proto);
        // Exits write the closure to the frame's header.
        outer.push(((base - 4) as u8, cv));
        let p = b.ins(Op::Unbox(Rep::Ptr), &[cv]);
        (Err(k), crate::jit::region::DYNAMIC | k, Some(p))
    } else {
        let k = b.konst(crate::env::value::Value::function(lf.function()));
        let Some(Op::KObj(func)) = b.f.def_op(k) else {
            return false;
        };
        (Ok(k), func, None)
    };
    let frame = b.f.inline_frames.len() as u32;
    let parent = b.frame_id();
    b.f.inline_frames.push(InlineFrame {
        parent,
        func,
        a,
        c,
        caller_pc: pc,
    });
    let wanted = c - 1;
    let cont = b.new_block();
    let results: Vec<Val> = (0..wanted).map(|_| b.f.add_param(cont, Ty::ANY)).collect();
    let def_snaps = b.f.def_snaps.len();
    let meta = b.f.meta;
    let f = std::mem::replace(&mut b.f, Func::new(meta));
    let inline = Inline {
        frame,
        base,
        outer,
        cont,
        wanted,
        depth: depth + 1,
        failed: false,
        returned: false,
        closure,
    };
    let mut ib = Builder::new_inline(b.ctx, lf, &cfg, f, inline, b.cold, b.deopt_all);
    let entry = ib.build_inline(&args, MAX_INSTS);
    b.f = std::mem::replace(&mut ib.f, Func::new(meta));
    drop(ib);
    b.pad_blocks();
    let Some(entry) = entry else {
        b.f.blocks[cont.idx()].dead = true;
        b.f.def_snaps.truncate(def_snaps);
        return false;
    };
    // The callee pinned, its frame's window, then its body.
    match guard {
        Ok(k) => b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const),
        Err(k) => b.ins_exit(Op::GuardProto(k), &[cv], ExitTag::Const),
    };
    let end = (base + proto.max_stack_size as usize) as u16;
    b.ins_exit(Op::GuardWindow(end), &[], ExitTag::Slow);
    b.flush();
    let slots: Vec<SlotState> = (0..a as usize).map(|r| b.slot(b.cur, r)).collect();
    b.jump(entry);
    // After it: the callers' registers as at the call, the results.
    b.bs[cont.idx()].sealed = true;
    for &(r, v) in &kept {
        b.set_def(cont, r, v);
    }
    for (k, &v) in results.iter().enumerate() {
        b.set_def(cont, a + k as u8, v);
    }
    for (r, s) in slots.into_iter().enumerate() {
        b.set_slot(cont, r, s);
    }
    b.switch_to(cont);
    true
}

/// Whether a callee's instruction needs its frame on the stack: its slots,
/// closures over it, cells, varargs, metamethod calls, a constructor (whose
/// helper finds the template through the frame). A call may be an
/// intrinsic's inline code; the build finds out.
fn needs_frame(op: BcOp) -> bool {
    use BcOp::*;
    matches!(
        op.unquickened(),
        TAILCALL
            | CONCAT
            | CLOSE
            | TBC
            | TFORPREP
            | TFORCALL
            | TFORLOOP
            | NEWTABLE
            | SETLIST
            | CLOSURE
            | VARARG
            | VARARGGET
            | VARARGPREP
            | SETUPVAL
            | GETUPVAL_REF
            | GETTABUP_REF
            | SETTABUP_REF
            | ARITH_MM
            | ARITH_MM_R
            | ARITH_MMI
    )
}

/// The inlined callee's `RETURN` of `n` results from `R[a..]`: a jump to
/// the block after the call with the results it wants.
pub(super) fn inline_return(b: &mut Builder<'_, '_>, a: u8, n: u8) {
    let i = b.inline.as_mut().expect("an inlined body");
    i.returned = true;
    let (cont, wanted) = (i.cont, i.wanted);
    let mut res = Vec::with_capacity(wanted as usize);
    for k in 0..wanted {
        let v = if k < n {
            let v = b.reg(a + k);
            b.boxed(v)
        } else {
            b.konst(crate::env::value::Value::nil())
        };
        res.push(v);
    }
    let t = b.terminate(Op::Jump, &[], &[cont], None);
    b.f.set_edge_args(t, 0, &res);
}
