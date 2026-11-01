//! Inlining (E1): a call of a known small Lua function becomes its body,
//! built by a builder of its own into the same function. The body's
//! snapshots name its frame and the callers' registers at the call, so an
//! exit rebuilds both; its returns jump to a block taking the results.
//! A call in the body is the region's, its frame past the body's registers;
//! the frame walk finds the inlined frames from its site (W3 §4). Its
//! cells and closures go through the callee closure, and what its closures
//! open on its slots closes at its returns (§6). A body that needs its
//! frame on the stack (a to-be-closed variable, varargs) is not inlined.
//! The callee is what the call would guard
//! (`intrinsics::callee_of`): a closure, whose upvalues are constants, or
//! one of a prototype, whose upvalues are loaded from it.

use std::rc::Rc;

use super::intrinsics::{Callee, callee_of};
use super::{Builder, Inline, SlotState};
use crate::env::function::LuaFn;
use crate::instruction::{Instruction, Op as BcOp};
use crate::jit::build::cfg::{Cfg, RegSet, use_def};
use crate::jit::ir::ops::{Cc, ExitTag, HelperId, Op};
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
    /// Some instruction may read or write its callers' captured registers
    /// through their cells: a call, or a load or store of a cell upvalue.
    cells: bool,
}

/// Inline the CALL at `pc` of `callee`; whether it did, the current block
/// then the one after the call. A `TAILCALL` inlines its callee in its
/// frame's place (W3 §5): the frame it replaces is gone, so the callee has
/// its base, caller and continuation, and its returns are that frame's. In
/// the region's own frame (none of whose registers a closure captured) a
/// constant callee takes it over: exits give frame 0's header the callee,
/// and its returns are the region's.
pub(super) fn try_inline(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, callee: Val) -> bool {
    if !b.ctx.jit().config.inline {
        return false;
    }
    let tail = insn.op() == BcOp::TAILCALL;
    let replace = tail && b.inline.is_none();
    if replace && b.cfg.captured.iter().next().is_some() {
        return false;
    }
    let (a, nb) = (insn.a(), insn.b());
    let c = match &b.inline {
        Some(i) if tail => b.f.inline_frames[i.frame as usize].c,
        None if tail => 1,
        _ => insn.c(),
    };
    if c == 0 {
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
        Callee::Proto(_, Some(lf)) if !replace => (lf, true),
        Callee::Proto(..) | Callee::Unknown => return false,
    };
    let proto = lf.proto;
    let base = if tail {
        b.reg_base()
    } else {
        b.reg_base() + a as usize + 4
    };
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
    // The body may read or write the captured registers through their
    // cells: they live in their slots across it, read from them after. Its
    // code may say so (a call, a cell); else a body that called anyway (a
    // metamethod, a builtin's fallback) is built again with them there.
    // Arguments up to `top`, which a real call set (W3 §7).
    let top_args = if nb == 0 {
        let Some(n) = top_count(b, pc, b.slot_of(a)) else {
            return false;
        };
        top_values(b, pc, a + 4, proto.num_params, n)
    } else {
        Vec::new()
    };
    // A tail call's frame is gone: what the body it replaces watched, the
    // values its returns report.
    let mut sync = code.cells && !tail;
    let seen = if tail && !replace {
        watched(b)
    } else {
        Vec::new()
    };
    let (entry, inline, cont, results, args, kept, captured, guard, cv) = loop {
        let captured = if sync { b.cfg.captured } else { RegSet::EMPTY };
        b.store_regs(captured.iter());
        // What the callers hold at the call, and the callee's arguments.
        let live = b
            .cfg
            .live_in
            .get(pc as usize + 1)
            .copied()
            .unwrap_or_default();
        let mut outer = match &b.inline {
            Some(i) if tail => i.outer_entries(),
            None if tail => Vec::new(),
            _ => b.call_entries(live, a),
        };
        // A body's watched parameters hold their arguments to its returns, live
        // or not (any call is above them, as it would write them).
        let mut keep = b.exit_regs(live);
        for &k in b.inline.iter().flat_map(|i| &i.watch) {
            keep.insert(k as usize);
        }
        let regs: Vec<usize> = keep
            .iter()
            .filter(|&r| !tail && r < a as usize && r < b.nregs)
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
        let args: Vec<Val> = if nb == 0 {
            top_args.clone()
        } else {
            (0..nb - 1).map(|k| b.reg(a + 4 + k)).collect()
        };
        let mut watch: Vec<u8> = Vec::new();
        for (k, &v) in args.iter().enumerate().take(proto.num_params as usize) {
            if !tail
                && !code.written.contains(k)
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
        let (parent, fa, caller_pc, replaces) = match &b.inline {
            Some(i) if tail => {
                let x = b.f.inline_frames[i.frame as usize];
                (x.parent, x.a, x.caller_pc, x.replaces)
            }
            _ => (b.frame_id(), a, pc, replace),
        };
        b.f.inline_frames.push(InlineFrame {
            parent,
            func,
            a: fa,
            c,
            caller_pc,
            code: proto.code.as_ptr() as usize,
            open: code.cfg.captured.iter().next().is_some(),
            replaces,
        });
        // A call keeps none of them as a home (a body that fails leaves them
        // in, which only costs reloads).
        for r in code.cfg.captured.iter() {
            b.f.captured.insert(base + r);
        }
        let wanted = c - 1;
        let region_return = b.inline.as_ref().is_some_and(|i| tail && i.region_return) || replace;
        let (cont, results) = match &b.inline {
            Some(i) if tail => (i.cont, Vec::new()),
            None if tail => (b.f.entry, Vec::new()),
            _ => {
                let cont = b.new_block();
                let results: Vec<Val> = (0..wanted).map(|_| b.f.add_param(cont, Ty::ANY)).collect();
                (cont, results)
            }
        };
        // What a body that fails leaves behind is undone.
        let def_snaps = b.f.def_snaps.len();
        let stable_used = b.f.stable_used.borrow().len();
        let recv_shapes = b.known.recv_shapes.clone();
        let inline = Inline {
            frame,
            base,
            outer,
            outer_homes,
            called: false,
            region_return,
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
            if !tail {
                b.f.blocks[cont.idx()].dead = true;
            }
            b.f.def_snaps.truncate(def_snaps);
            b.f.stable_used.borrow_mut().truncate(stable_used);
            b.known.recv_shapes = recv_shapes;
            return false;
        };
        if inline.called && !sync && b.cfg.captured.iter().next().is_some() {
            b.f.blocks[cont.idx()].dead = true;
            b.f.def_snaps.truncate(def_snaps);
            b.f.stable_used.borrow_mut().truncate(stable_used);
            b.known.recv_shapes = recv_shapes;
            sync = true;
            continue;
        }
        break (
            entry, inline, cont, results, args, kept, captured, guard, cv,
        );
    };
    // A call in it is a call of its caller's.
    if inline.called
        && let Some(i) = b.inline.as_mut()
    {
        i.called = true;
    }
    // The frame it replaces closes; the callee pinned, its frame's window,
    // then its body.
    if tail {
        close_frame(b);
    }
    match guard {
        Ok(k) => b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const),
        Err(k) => b.ins_exit(Op::GuardProto(k), &[cv], ExitTag::Const),
    };
    let end = (base + proto.max_stack_size as usize) as u16;
    b.ins_exit(Op::GuardWindow(end), &[], ExitTag::Slow);
    b.flush();
    let slots: Vec<SlotState> = (0..a as usize).map(|r| b.slot(b.cur, r)).collect();
    b.jump(entry);
    // Its returns are the body's, reporting what the body's watched, or
    // the region's.
    if tail {
        if let Some(i) = b.inline.as_mut() {
            for (t, _) in inline.returns {
                i.returns.push((t, seen.clone()));
            }
        }
        return true;
    }
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
    let cfg = Cfg::build(&proto, code, 0, Vec::new()).ok()?;
    if cfg.never_ran[cfg.block_of[0] as usize] {
        return None;
    }
    let cells = cfg.code.iter().any(|i| {
        use BcOp::*;
        matches!(
            i.op().unquickened(),
            CALL | CALL_R0
                | CALL_R1
                | TAILCALL
                | CALLS
                | CALLS_R0
                | CALLS_R1
                | GETUPVAL_REF
                | SETUPVAL
                | GETTABUP_REF
                | SETTABUP_REF
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
        cells,
    })
}

/// Whether a callee's instruction needs its frame on the stack: a
/// to-be-closed variable, varargs.
fn needs_frame(op: BcOp) -> bool {
    use BcOp::*;
    matches!(op.unquickened(), TBC | VARARG | VARARGGET | VARARGPREP)
}

/// The inlined callee's `RETURN` of `n` results from `R[a]`: a jump to the
/// block after the call with the results it wants, as they are.
pub(super) fn inline_return(b: &mut Builder<'_, '_>, a: u8, n: u8) {
    let i = b.inline.as_ref().expect("an inlined body");
    if i.region_return {
        close_frame(b);
        super::emit::store_results(b, a, n);
        let t = b.terminate(Op::Return { a, n }, &[], &[], None);
        b.inline.as_mut().unwrap().returns.push((t, Vec::new()));
        return;
    }
    let wanted = i.wanted;
    let mut res = Vec::with_capacity(wanted as usize);
    for k in 0..wanted {
        let v = if k < n {
            b.reg(a + k)
        } else {
            b.konst(crate::env::value::Value::nil())
        };
        res.push(v);
    }
    let seen = watched(b);
    close_frame(b);
    return_values(b, &res, seen);
}

/// The body's `RETURN` of `R[a..top)` at `pc`, which a real call set (W3
/// §7); false if no call before it did.
pub(super) fn inline_return_top(b: &mut Builder<'_, '_>, pc: u32, a: u8) -> bool {
    let wanted = b.inline.as_ref().expect("an inlined body").wanted;
    // A body's slots start past a header: `TopArgs(s)` counts from `s + 4`.
    let Some(n) = top_count(b, pc, b.slot_of(a) - 4) else {
        return false;
    };
    let res = top_values(b, pc, a, wanted, n);
    let seen = watched(b);
    close_frame(b);
    return_values(b, &res, seen);
    true
}

/// How many values there are from region slot `s + 4` to `top`, which a
/// real call at the instruction before `pc` set (`TopArgs`): none if no
/// call did.
fn top_count(b: &mut Builder<'_, '_>, pc: u32, s: u8) -> Option<Val> {
    producer(b, pc)?;
    Some(b.ins(Op::Helper(HelperId::TopArgs(s)), &[]))
}

/// The first register of the call at the instruction before `pc` that
/// returned up to `top`.
fn producer(b: &Builder<'_, '_>, pc: u32) -> Option<u8> {
    use BcOp::*;
    let p = b.code(pc.checked_sub(1)?);
    let call = matches!(
        p.op().unquickened(),
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1
    );
    (call && p.c() == 0).then_some(p.a())
}

/// `n` values of `R[r..top)` (`count` of them, an i32) for the instruction
/// at `pc`: the registers before the call that set `top` as they are, its
/// results from their slots where below `top` and nil past it. The window
/// guard keeps the loads in the stack.
fn top_values(b: &mut Builder<'_, '_>, pc: u32, r: u8, n: u8, count: Val) -> Vec<Val> {
    let first = producer(b, pc).expect("a call returning to top");
    let end = b.slot_of(r) as u16 + n as u16;
    b.ins_exit(Op::GuardWindow(end), &[], ExitTag::Slow);
    let nil = b.konst(crate::env::value::Value::nil());
    (0..n)
        .map(|k| {
            if r + k < first {
                return b.reg(r + k);
            }
            let kk = b.ki32(k as i32);
            let has = b.ins(Op::ICmp(Cc::Lt), &[kk, count]);
            let v = b.ins(Op::Load(b.slot_of(r + k)), &[]);
            b.ins(Op::Select, &[has, v, nil])
        })
        .collect()
}

/// What the body's watched parameters hold now.
fn watched(b: &mut Builder<'_, '_>) -> Vec<Val> {
    let watch = b.inline.as_ref().expect("an inlined body").watch.clone();
    watch.iter().map(|&r| b.reg(r)).collect()
}

/// Close what the body's closures opened on its slots, from their values
/// now: its frame goes.
pub(super) fn close_frame(b: &mut Builder<'_, '_>) {
    let captured = b.cfg.captured;
    if captured.iter().next().is_some() {
        b.store_regs(captured.iter());
        let at = b.slot_of(0);
        b.push(Op::Helper(crate::jit::ir::ops::HelperId::Close(at)), &[]);
    }
}

/// Jump to the block after the call with the results, `seen` what the
/// watched parameters held.
fn return_values(b: &mut Builder<'_, '_>, res: &[Val], seen: Vec<Val>) {
    let cont = b.inline.as_ref().expect("an inlined body").cont;
    let t = b.terminate(Op::Jump, &[], &[cont], None);
    b.f.set_edge_args(t, 0, res);
    b.inline.as_mut().unwrap().returns.push((t, seen));
}

/// The body's `TAILCALL` of `callee` with `nargs` arguments from `R[a+4]`
/// as a real call replacing its frame (W3 §5): from the body's header
/// slot, wanting what its caller wanted, its results the body's returns.
/// A lazy exit past it leaves as the caller, after the call it made.
pub(super) fn tail_call(b: &mut Builder<'_, '_>, pc: u32, a: u8, nargs: u8, callee: Val) {
    let i = b.inline.as_ref().expect("an inlined body");
    let (base, wanted) = (i.base, i.wanted);
    let x = b.f.inline_frames[i.frame as usize];
    let seen = watched(b);
    let (callee, proto) = super::emit::known_callee(b, pc, callee);
    let vals: Vec<Val> = (0..nargs)
        .map(|k| {
            let v = b.reg(a + 4 + k);
            b.boxed(v)
        })
        .collect();
    close_frame(b);
    // The first eight are the call's operands (W4 §2), the rest go to the
    // callee's slots, the body's from its base.
    let in_regs = (nargs as usize).min(8);
    for (k, &v) in vals.iter().enumerate().skip(in_regs) {
        b.push(Op::Store((base + k) as u8), &[v]);
    }
    let cv = b.boxed(callee);
    let hdr = (base - 4) as u8;
    b.push(Op::Store(hdr), &[cv]);
    let op = Op::Call {
        a: hdr,
        nargs,
        c: x.c,
        pc: x.caller_pc,
        proto,
    };
    let resume = b.tail_call(op, &vals[..in_regs], x.c);
    let res: Vec<Val> = if x.c == 2 || x.c == 3 {
        let first = b.f.insts_of(resume)[0];
        b.f.results(first).collect()
    } else {
        (0..wanted).map(|k| b.ins(Op::Load(hdr + k), &[])).collect()
    };
    let from = b.f.snap_pool.len();
    let outer = b.inline.as_ref().unwrap().outer_entries();
    b.f.snap_pool.extend_from_slice(&outer);
    for (k, &v) in res.iter().enumerate() {
        b.f.snap_pool.push((hdr + k as u8, v));
    }
    let s =
        b.f.finish_snap(x.parent, x.caller_pc, crate::jit::ir::ExitKind::After, from);
    b.lazy_exit(s);
    return_values(b, &res, seen);
}
