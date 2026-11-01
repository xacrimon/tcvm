//! Inlining (E1): a call of a known small Lua function becomes its body,
//! built by a builder of its own into the same function. The body's
//! snapshots name its frame and the callers' registers at the call, so an
//! exit rebuilds both; its returns jump to a block taking the results.
//! A call in the body is the region's, its frame past the body's registers;
//! the frame walk finds the inlined frames from its site (W3 §4). A body
//! that needs its frame on the stack (its closure's cells, a frame-relative
//! helper) is not inlined. The callee is what the call would guard
//! (`intrinsics::callee_of`): a closure, whose upvalues are constants, or
//! one of a prototype, whose upvalues are loaded from it.

use std::rc::Rc;

use super::intrinsics::{Callee, callee_of};
use super::{Builder, Inline, SlotState};
use crate::env::function::LuaFn;
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::cfg::{Cfg, RegSet, use_def};
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::{Rep, Ty};
use crate::jit::ir::{InlineFrame, Val};

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

/// A callee's code as its bodies are built, kept per prototype for the
/// compile (`Known::callees`).
pub(crate) struct InlineCode {
    cfg: Cfg,
    /// The registers some instruction writes.
    written: RegSet,
    /// Some instruction is a call, which a body may make.
    calls: bool,
}

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
    let key = crate::dmm::Gc::as_ptr(proto) as usize;
    let code = match b.known.callees.get(&key) {
        Some(code) => code.clone(),
        None => {
            let code = prepare(lf).map(Rc::new);
            b.known.callees.insert(key, code.clone());
            code
        }
    };
    let Some(code) = code else {
        return false;
    };
    // A call in the body may write the captured registers through their
    // cells: they live in their slots across it, read from them after.
    let captured = if code.calls {
        b.cfg.captured
    } else {
        RegSet::EMPTY
    };
    b.store_regs(captured.iter());
    // What the callers hold at the call, and the callee's arguments.
    let live = b
        .cfg
        .live_in
        .get(pc as usize + 1)
        .copied()
        .unwrap_or_default();
    let mut outer = b.call_entries(live, a);
    // A body's watched parameters hold their arguments to its returns, live
    // or not (any call is above them, as it would write them).
    let mut keep = b.exit_regs(live);
    for &k in b.inline.iter().flat_map(|i| &i.watch) {
        keep.insert(k as usize);
    }
    let regs: Vec<usize> = keep
        .iter()
        .filter(|&r| r < a as usize && r < b.nregs)
        .collect();
    // The callers' slots a call in the body keeps, by region slot.
    let mut outer_homes = b.inline.as_ref().map_or(RegSet::EMPTY, |i| i.outer_homes);
    for r in captured.iter() {
        outer_homes.insert(b.reg_base() + r);
    }
    for &r in &regs {
        let v = b.reg(r as u8);
        if b.untouched(r) || b.slot_holds(r as u8, v) {
            outer_homes.insert(b.reg_base() + r);
        }
    }
    let kept: Vec<(u8, Val)> = regs
        .into_iter()
        .map(|r| (r as u8, b.reg(r as u8)))
        .collect();
    let args: Vec<Val> = (0..nb - 1).map(|k| b.reg(a + 4 + k)).collect();
    let mut watch: Vec<u8> = Vec::new();
    for (k, &v) in args.iter().enumerate().take(proto.num_params as usize) {
        if !code.written.contains(k)
            && kept.iter().any(|&(_, x)| x == v)
            && !watch.iter().any(|&j| args[j as usize] == v)
        {
            watch.push(k as u8);
        }
    }
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
        code: proto.code.as_ptr() as usize,
    });
    let wanted = c - 1;
    let cont = b.new_block();
    let results: Vec<Val> = (0..wanted).map(|_| b.f.add_param(cont, Ty::ANY)).collect();
    // What a body that fails leaves behind is undone.
    let def_snaps = b.f.def_snaps.len();
    let stable_used = b.f.stable_used.borrow().len();
    let recv_shapes = b.known.recv_shapes.clone();
    let inline = Inline {
        frame,
        base,
        outer,
        outer_homes,
        cont,
        wanted,
        depth: depth + 1,
        failed: false,
        watch,
        returns: Vec::new(),
        closure,
    };
    let from = b.cur;
    let mut ib = Builder::new_inline(b, lf, &code.cfg, inline);
    let entry = ib.build_inline(&args, from, MAX_INSTS);
    let inline = ib.finish(b);
    let Some(entry) = entry else {
        b.f.blocks[cont.idx()].dead = true;
        b.f.def_snaps.truncate(def_snaps);
        b.f.stable_used.borrow_mut().truncate(stable_used);
        b.known.recv_shapes = recv_shapes;
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
    // The results have the types the returns give them.
    let rets = &inline.returns;
    for (k, &p) in results.iter().enumerate() {
        let t = rets
            .iter()
            .map(|&(j, _)| b.ty(b.f.edge_args(j, 0)[k]))
            .reduce(Ty::join)
            .expect("a return");
        b.set_ty(p, t);
    }
    // What the returns refined of arguments callers' registers hold: one
    // value, or a join where that is narrower than the argument.
    let mut refined = Vec::new();
    for (w, &k) in inline.watch.iter().enumerate() {
        let v = args[k as usize];
        let x0 = rets[0].1[w];
        let to = if rets.iter().all(|r| r.1[w] == x0) {
            x0
        } else {
            let t = rets.iter().map(|r| b.ty(r.1[w])).reduce(Ty::join).unwrap();
            let tv = b.ty(v);
            if t.set == tv.set && t.refine == tv.refine {
                continue;
            }
            let q = b.f.add_param(cont, t);
            for r in rets {
                b.f.push_edge_arg(r.0, 0, r.1[w]);
            }
            q
        };
        if to != v {
            refined.push((v, to));
        }
    }
    // After it: the callers' registers as at the call, the results. Every
    // register is defined, so no read walks into the body's blocks, but for
    // the captured ones a call in it may have written, read from their
    // slots.
    b.bs[cont.idx()].sealed = true;
    let nil = b.konst_in(cont, crate::env::value::Value::nil());
    for r in 0..b.nregs as u8 {
        if !captured.contains(r as usize) {
            b.set_def(cont, r, nil);
        }
    }
    for &(r, v) in &kept {
        if !captured.contains(r as usize) {
            b.set_def(cont, r, v);
        }
    }
    for (k, &v) in results.iter().enumerate() {
        b.set_def(cont, a + k as u8, v);
    }
    for (r, s) in slots.into_iter().enumerate() {
        b.set_slot(cont, r, s);
    }
    b.bs[cont.idx()].reload = captured;
    for r in captured.iter() {
        b.set_slot(cont, r, SlotState::Lazy(cont));
    }
    b.init_facts(cont);
    b.switch_to(cont);
    for (v, to) in refined {
        b.refine(v, to);
    }
    true
}

/// A callee's code for inlining, or none when a body of it would need its
/// frame or its entry never ran.
fn prepare(lf: LuaFn<'_>) -> Option<InlineCode> {
    let proto = lf.proto;
    let mut code: Vec<Instruction> = proto
        .code
        .iter()
        .map(|i| crate::jit::compile::resolve_original(lf, i))
        .collect();
    if code.iter().any(|i| needs_frame(i.op())) {
        return None;
    }
    crate::jit::compile::canonical_back_edges(lf, &mut code);
    let cfg = Cfg::build(&proto, code, 0).ok()?;
    if cfg.never_ran[cfg.block_of[0] as usize] {
        return None;
    }
    let calls = cfg.code.iter().any(|i| {
        use BcOp::*;
        matches!(
            i.op().unquickened(),
            CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1
        )
    });
    let mut written = RegSet::EMPTY;
    for &i in &cfg.code {
        written.union(&use_def(&proto, i, cfg.nregs, &cfg.captured).defs);
        // Written on their jump edge only.
        if matches!(i.op(), BcOp::JTSET | BcOp::JFSET) && (i.a() as usize) < cfg.nregs {
            written.insert(i.a() as usize);
        }
    }
    Some(InlineCode {
        cfg,
        written,
        calls,
    })
}

/// Whether a callee's instruction needs its frame on the stack: closures
/// over it, cells, varargs, metamethod calls.
fn needs_frame(op: BcOp) -> bool {
    use BcOp::*;
    matches!(
        op.unquickened(),
        TAILCALL
            | CLOSE
            | TBC
            | TFORPREP
            | TFORCALL
            | TFORLOOP
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

/// The inlined callee's `RETURN` of `n` results from `R[a]`: a jump to the
/// block after the call with the results it wants, as they are.
pub(super) fn inline_return(b: &mut Builder<'_, '_>, a: u8, n: u8) {
    let i = b.inline.as_ref().expect("an inlined body");
    let (cont, wanted, watch) = (i.cont, i.wanted, i.watch.clone());
    let mut res = Vec::with_capacity(wanted as usize);
    for k in 0..wanted {
        let v = if k < n {
            b.reg(a + k)
        } else {
            b.konst(crate::env::value::Value::nil())
        };
        res.push(v);
    }
    let seen: Vec<Val> = watch.iter().map(|&r| b.reg(r)).collect();
    let t = b.terminate(Op::Jump, &[], &[cont], None);
    b.f.set_edge_args(t, 0, &res);
    b.inline.as_mut().unwrap().returns.push((t, seen));
}
