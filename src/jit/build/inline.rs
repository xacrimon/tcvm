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
use super::{Builder, Inline, SlotState, Top};
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
    /// The most values a return gives, none if one returns up to `top`.
    returns: Option<u8>,
    /// Some instruction is a tail call.
    tails: bool,
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
    // In the region frame's place (its tail call or a replacement's) only a
    // constant callee: below the frame's window no slot holds a closure.
    let in_region_place = tail && (b.inline.as_ref()).is_none_or(|i| i.region_return);
    let (a, nb) = (insn.a(), insn.b());
    let c = match &b.inline {
        Some(i) if tail => b.f.inline_frames[i.frame as usize].c,
        None if tail => 1,
        _ => insn.c(),
    };
    // Every result (W3 §7): for an instruction after it that takes them up
    // to `top`, from a callee each of whose returns gives a fixed count.
    let multi = c == 0 && !tail;
    if multi && !takes_top(b.code(pc + 1)) {
        return false;
    }
    let depth = b.inline.as_ref().map_or(0, |i| i.depth);
    if depth >= MAX_DEPTH || b.f.insts.len() > MAX_INSTS {
        return false;
    }
    // A callee known by its prototype (several closures of it, or one this
    // compile made) is built from its prototype, its upvalues loaded from
    // the closure value (W3 §10); one the compile made needs no guard.
    let (lf, proto, dynamic, made) = match callee_of(b, pc, callee) {
        Callee::Const(f) | Callee::Same(f) => match f.as_lua() {
            Some(lf) => (Some(lf), lf.proto, false, false),
            None => return false,
        },
        Callee::Proto(p, lf) if !in_region_place => (lf, p, true, false),
        Callee::Made(p) if !in_region_place => (None, p, true, true),
        Callee::Proto(..) | Callee::Made(_) | Callee::Unknown => return false,
    };
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
            let code = prepare(proto).map(Rc::new);
            b.known.callees.insert(key, code.clone());
            code
        }
    };
    let Some(code) = code else {
        return false;
    };
    let returns = code.returns.unwrap_or(u8::MAX);
    if multi && (code.tails || a as usize + returns as usize > b.nregs) {
        return false;
    }
    // Arguments up to `top`, which an inlined call or a real one set.
    let top_args = if nb == 0 {
        match b.cur_top {
            Some((from, Top::Count(n, most))) => {
                reg_top_values(b, a + 4, proto.num_params, from, n, most)
            }
            _ => {
                let Some(n) = top_count(b, pc, b.slot_of(a)) else {
                    return false;
                };
                top_values(b, pc, a + 4, proto.num_params, n)
            }
        }
    } else {
        Vec::new()
    };
    // The body may read or write the captured registers through their
    // cells: they live in their slots across it, read from them after. Its
    // code may say so (a call, a cell); else a body that called anyway (a
    // metamethod, a builtin's fallback) is built again with them there. A
    // tail call's frame is gone: what the body it replaces watched, the
    // values its returns report.
    let mut sync = code.cells && !tail;
    let seen = if tail && !replace {
        watched(b)
    } else {
        Vec::new()
    };
    let (entry, inline, cont, results, args, kept, captured, guard, cv, top) = loop {
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
            let lf = lf.expect("a constant callee");
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
        let wanted = if multi { returns } else { c - 1 };
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
        let count = multi.then(|| b.f.add_param(cont, Ty::I32));
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
            counted: multi,
            cont,
            wanted,
            depth: depth + 1,
            failed: false,
            watch,
            returns: Vec::new(),
            closure,
        };
        let from = b.cur;
        let mut ib = Builder::new_inline(b, lf, proto, &code.cfg, inline);
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
        // How many it left: one count, or a value only a call or a body's
        // return after it takes.
        let top = count.map(|n| {
            let counts: Vec<Option<i32>> = (inline.returns.iter())
                .map(
                    |&(t, _)| match b.f.def_op(b.f.edge_args(t, 0)[wanted as usize]) {
                        Some(Op::KI32(k)) => Some(k),
                        _ => None,
                    },
                )
                .collect();
            match counts[0] {
                Some(k) if counts.iter().all(|&x| x == Some(k)) => Some(Top::Fixed(k as u8)),
                _ => takes_count(b, b.code(pc + 1)).then_some(Top::Count(n, wanted)),
            }
        });
        if matches!(top, Some(None)) {
            b.f.blocks[cont.idx()].dead = true;
            b.f.def_snaps.truncate(def_snaps);
            b.f.stable_used.borrow_mut().truncate(stable_used);
            b.known.recv_shapes = recv_shapes;
            return false;
        }
        break (
            entry, inline, cont, results, args, kept, captured, guard, cv, top,
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
        Ok(k) => {
            b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const);
        }
        Err(k) if !made => {
            b.ins_exit(Op::GuardProto(k), &[cv], ExitTag::Const);
        }
        Err(_) => {}
    }
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
    // As a call returning them leaves them, for an exit before the next
    // instruction takes them, which runs it in the interpreter.
    if let Some(Some(t)) = top {
        b.store_regs(a as usize..a as usize + inline.wanted as usize);
        let n = match t {
            Top::Fixed(k) => b.ki32(k as i32),
            Top::Count(n, _) => n,
        };
        b.push(Op::SetTop(b.slot_of(a)), &[n]);
        b.top = Some((a, t));
    }
    true
}

/// Whether `next` takes values up to `top`: a call, tail call or return.
fn takes_top(next: Instruction) -> bool {
    use BcOp::*;
    let op = next.op().unquickened();
    next.b() == 0
        && matches!(
            op,
            CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 | TAILCALL | RETURN
        )
}

/// Whether `next` takes a count of values up to `top` as a value: a call,
/// or an inlined body's return.
fn takes_count(b: &Builder<'_, '_>, next: Instruction) -> bool {
    use BcOp::*;
    match next.op().unquickened() {
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => true,
        RETURN => b.inline.as_ref().is_some_and(|i| !i.region_return),
        _ => false,
    }
}

/// A callee's code for inlining, or none when a body of it would need its
/// frame or its entry never ran.
fn prepare(proto: crate::dmm::Gc<'_, crate::env::function::Prototype<'_>>) -> Option<InlineCode> {
    let mut code: Vec<Instruction> = proto
        .code
        .iter()
        .map(|i| crate::jit::compile::resolve_original(&proto, i))
        .collect();
    if code.iter().any(|i| needs_frame(i.op())) {
        return None;
    }
    crate::jit::compile::canonical_back_edges(&proto, &mut code);
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
    let (mut returns, mut tails) = (Some(0u8), false);
    for &i in &cfg.code {
        let n = match i.op().unquickened() {
            BcOp::RETURN0 => 0,
            BcOp::RETURN1 => 1,
            BcOp::RETURN if i.b() == 0 => {
                returns = None;
                continue;
            }
            BcOp::RETURN => i.b() - 1,
            BcOp::TAILCALL => {
                tails = true;
                continue;
            }
            _ => continue,
        };
        returns = returns.map(|r| r.max(n));
    }
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
        returns,
        tails,
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
    let count = b.inline.as_ref().unwrap().counted.then_some(n);
    return_values(b, &res, seen, count);
}

/// The body's `RETURN` of `R[a..top)` at `pc`, which a real call set (W3
/// §7); false if no call before it did.
pub(super) fn inline_return_top(b: &mut Builder<'_, '_>, pc: u32, a: u8) -> bool {
    let wanted = b.inline.as_ref().expect("an inlined body").wanted;
    // A body's slots start past a header: `TopArgs(s)` counts from `s + 4`.
    let res = match b.cur_top {
        Some((from, Top::Count(n, most))) => reg_top_values(b, a, wanted, from, n, most),
        _ => {
            let Some(n) = top_count(b, pc, b.slot_of(a) - 4) else {
                return false;
            };
            top_values(b, pc, a, wanted, n)
        }
    };
    let seen = watched(b);
    close_frame(b);
    return_values(b, &res, seen, None);
    true
}

/// `n` values of `R[r..top)`, which an inlined call left in registers from
/// `from`, `count` (an i32, at most `most`) of them, nil past it.
fn reg_top_values(
    b: &mut Builder<'_, '_>,
    r: u8,
    n: u8,
    from: u8,
    count: Val,
    most: u8,
) -> Vec<Val> {
    let nil = b.konst(crate::env::value::Value::nil());
    (0..n)
        .map(|k| {
            let x = r + k;
            if x < from {
                return b.reg(x);
            }
            if x >= from + most {
                return nil;
            }
            let kk = b.ki32((x - from) as i32);
            let has = b.ins(Op::ICmp(Cc::Lt), &[kk, count]);
            let v = b.reg(x);
            b.ins(Op::Select, &[has, v, nil])
        })
        .collect()
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
fn return_values(b: &mut Builder<'_, '_>, res: &[Val], seen: Vec<Val>, count: Option<u8>) {
    let cont = b.inline.as_ref().expect("an inlined body").cont;
    let mut args = res.to_vec();
    if let Some(n) = count {
        args.push(b.ki32(n as i32));
    }
    let t = b.terminate(Op::Jump, &[], &[cont], None);
    b.f.set_edge_args(t, 0, &args);
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
    return_values(b, &res, seen, None);
}
