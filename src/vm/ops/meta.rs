//! Metamethod staging, the metamethod continuations, and the
//! `__index`, `__newindex` and `__call` chain walks.

use crate::env::MetamethodBits;
use crate::env::function::Function;
use crate::env::table::Table;
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::instruction::{Instruction, TFOR_VARS};
use crate::lua::Context;
use crate::vm::abi::handler;
use crate::vm::frame::{self, land_results};
use crate::vm::unwind::OpError;

/// Stage the call `$f($args..)` above the running frame's window and enter
/// it, `$ret` taking its results. For `slow` handlers, which have no
/// `call_mm!`; the frame's closure is read from its header.
macro_rules! stage_mm {
    ($pc:ident, $base:ident, $rt:ident, $ret:expr, $f:expr, [$($arg:expr),* $(,)?]) => {{
        let f: $crate::env::value::Value<'gc> = $f;
        let args: [$crate::env::value::Value<'gc>; _] = [$($arg),*];
        let max_stack = unsafe { $crate::vm::frame::closure($base) }.max_stack_size as usize;
        let hdr = unsafe { $base.add(max_stack) };
        if unsafe { hdr.add($crate::vm::frame::HDR + args.len()) }.cast_const() > thread!().stack_end {
            tail!($crate::vm::ops::meta::stage_grow)
        }
        unsafe {
            $crate::vm::frame::write_hdr(hdr, f.to_raw(), $crate::vm::abi::handler_bits($ret), $base, $pc);
            let mut p = hdr.add($crate::vm::frame::HDR);
            for a in args {
                p.write(a);
                p = p.add(1);
            }
        }
        tail!(
            $crate::vm::ops::call::enter,
            pc = hdr as *const $crate::instruction::Instruction,
            insn = $crate::vm::abi::Slot::nret(args.len())
        )
    }};
}
pub(crate) use stage_mm;

/// The first of a call's results, nil if it has none.
#[inline(always)]
fn first_result<'gc>(nret: usize, values: *const Value<'gc>) -> Value<'gc> {
    if nret > 0 {
        unsafe { values.read() }
    } else {
        Value::nil()
    }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// A metamethod call could not be staged above the window: grow the
    /// stack, then run the instruction again.
    slow fn stage_grow {
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let max_stack = unsafe { frame::closure(base) }.max_stack_size as usize;
        if !ts.ensure_frame_slots(bi + max_stack + frame::HDR + 4) {
            raise!(OpError::StackOverflow)
        }
        base = ts.slot_ptr(bi);
        pc = unsafe { pc.sub(1) };
        next!()
    }

    /// Continuation of a metamethod whose result goes to `R[A]`: arithmetic,
    /// unary, concatenation and `__index`.
    cont fn ret_store_a {
        let v = first_result(nret, values);
        let (caller, cpc) = caller!();
        let op: Instruction = unsafe { *cpc.sub(1) };
        unsafe { caller.add(op.a() as usize).write(v) };
        resume!(caller, cpc)
    }

    /// Continuation of a `__newindex` call: the results are dropped.
    cont fn ret_discard {
        let (caller, cpc) = caller!();
        resume!(caller, cpc)
    }

    /// Continuation of a comparison metamethod staged by a branch-if-true
    /// compare: the caller's branch jumps when the result is truthy. The
    /// sense is the continuation's identity, so nothing decodes the opcode.
    cont fn ret_cond_t {
        let truthy = !first_result(nret, values).is_falsy();
        let (caller, cpc) = caller!();
        let op: Instruction = unsafe { *cpc.sub(1) };
        base = caller;
        pc = cpc;
        set_closure!(unsafe { frame::closure(base) });
        branch!(truthy, op.branch_offset())
    }

    /// As [`ret_cond_t`] for a branch-if-false compare.
    cont fn ret_cond_f {
        let truthy = !first_result(nret, values).is_falsy();
        let (caller, cpc) = caller!();
        let op: Instruction = unsafe { *cpc.sub(1) };
        base = caller;
        pc = cpc;
        set_closure!(unsafe { frame::closure(base) });
        branch!(!truthy, op.branch_offset())
    }

    /// Continuation of a generic for's iterator: its results are the loop's
    /// variables, nil-padded.
    cont fn ret_tfor {
        let (caller, cpc) = caller!();
        let op: Instruction = unsafe { *cpc.sub(1) };
        let (a, count) = op.ab();
        unsafe { land_results(caller.add((a + TFOR_VARS) as usize), values, nret, count as usize) };
        resume!(caller, cpc)
    }

    /// Continuation of a CLOSE's `__close`: run the CLOSE again, for the next
    /// variable or to move on.
    cont fn ret_close {
        let (caller, cpc) = caller!();
        resume!(caller, unsafe { cpc.sub(1) })
    }

    /// Continuation of a RETURN's `__close`: run the RETURN again, with `top`
    /// back at the end of its results, which the slot below the call recorded.
    cont fn ret_return {
        // The callee may be a native (`__close = coroutine.yield`).
        let nv = unsafe { frame::extras(base) };
        let values_end = unsafe { base.sub(frame::HDR + nv + 1).read() }.get_small();
        thread!().set_top_unchecked(unsafe { values_end.unwrap_unchecked() } as usize);
        let (caller, cpc) = caller!();
        resume!(caller, unsafe { cpc.sub(1) })
    }
}

/// Maximum depth of `__index` / `__newindex` chains before we give up and
/// raise (Lua's `MAXTAGLOOP`).
pub(crate) const MAX_TAG_LOOP: usize = 2000;

/// Result of walking an `__index` chain.
pub(crate) enum IndexChain<'gc> {
    /// The chain resolved to a value (possibly nil).
    Resolved(Value<'gc>),
    /// The chain ended in a function to call with `(receiver, key)`,
    /// `receiver` being the value whose metatable held it.
    Invoke {
        func: Function<'gc>,
        receiver: Value<'gc>,
    },
    /// A non-table in the chain has no `__index`.
    NotIndexable(Value<'gc>),
    /// Chain depth exceeded `MAX_TAG_LOOP`.
    Exhausted,
}

/// Resolve `receiver[key]` through `__index` (`luaV_finishget`), given that
/// `receiver`'s own raw lookup, if it is a table, already missed.
pub(crate) fn walk_index_chain<'gc>(
    ctx: Context<'gc>,
    mut receiver: Value<'gc>,
    key: Value<'gc>,
) -> IndexChain<'gc> {
    for _ in 0..MAX_TAG_LOOP {
        let mm = match receiver.get_table() {
            Some(t) => {
                let mm = t
                    .shape()
                    .mt_cache()
                    .map_or(Value::nil(), |c| c.mm(MetamethodBits::INDEX));
                if mm.is_nil() {
                    return IndexChain::Resolved(Value::nil());
                }
                mm
            }
            None => {
                let mm = ctx.mm_of(receiver, MetamethodBits::INDEX);
                if mm.is_nil() {
                    return IndexChain::NotIndexable(receiver);
                }
                mm
            }
        };
        if let Some(func) = mm.get_function() {
            return IndexChain::Invoke { func, receiver };
        }
        if let Some(t) = mm.get_table() {
            let v = t.raw_get(key);
            if !v.is_nil() {
                return IndexChain::Resolved(v);
            }
        }
        receiver = mm;
    }
    IndexChain::Exhausted
}

/// Result of walking a `__newindex` chain.
pub(crate) enum NewIndexChain<'gc> {
    /// Raw-assign into this table.
    RawSet(Table<'gc>),
    /// The chain ended in a function; call it with `(receiver, key, value)`.
    Invoke {
        func: Function<'gc>,
        receiver: Value<'gc>,
    },
    /// A non-table in the chain has no `__newindex`.
    NotIndexable(Value<'gc>),
    /// Chain depth exceeded `MAX_TAG_LOOP`.
    Exhausted,
}

/// Find where `t[key] = v` lands (`luaV_finishset`): the first table that
/// already has `key` or lacks `__newindex`, or a function `__newindex`.
#[inline]
pub(crate) fn walk_newindex_chain<'gc>(
    ctx: Context<'gc>,
    mut t: Value<'gc>,
    key: Value<'gc>,
) -> NewIndexChain<'gc> {
    for _ in 0..MAX_TAG_LOOP {
        let mm = match t.get_table() {
            Some(tbl) => {
                let mm = tbl
                    .shape()
                    .mt_cache()
                    .map_or(Value::nil(), |c| c.mm(MetamethodBits::NEWINDEX));
                if mm.is_nil() || !tbl.raw_get(key).is_nil() {
                    return NewIndexChain::RawSet(tbl);
                }
                mm
            }
            None => {
                let mm = ctx.mm_of(t, MetamethodBits::NEWINDEX);
                if mm.is_nil() {
                    return NewIndexChain::NotIndexable(t);
                }
                mm
            }
        };
        if let Some(func) = mm.get_function() {
            return NewIndexChain::Invoke { func, receiver: t };
        }
        t = mm;
    }
    NewIndexChain::Exhausted
}

/// `__call` hops a call may take before "'__call' chain too long", like the
/// reference's 4-bit `CIST_CCMT` counter.
const MAX_CALL_CHAIN: usize = 15;

/// Walk the `__call` chain of the value at header `hdr` until a function is
/// there, each hop shifting the `nargs` arguments (from `hdr + 4`) up one
/// slot to prepend the callable (Lua 5.5 `tryfuncTM`). Returns the new
/// argument count, `top` set past the arguments.
#[cold]
#[inline(never)]
pub(crate) fn resolve_call_chain<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    hdr: usize,
    mut nargs: usize,
) -> Result<usize, OpError<'gc>> {
    let args = hdr + frame::HDR;
    let mut hops = 0;
    loop {
        let fv = ts.stack[hdr];
        if fv.get_function().is_some() {
            ts.set_top(args + nargs);
            return Ok(nargs);
        }
        let mm = ctx.mm_of(fv, MetamethodBits::CALL);
        if mm.is_nil() {
            return Err(OpError::Call(fv));
        }
        if hops == MAX_CALL_CHAIN {
            return Err(OpError::CallChainTooLong);
        }
        hops += 1;
        ts.ensure_slots(args + nargs + 1);
        ts.stack.copy_within(args..args + nargs, args + 1);
        ts.stack[args] = fv;
        ts.stack[hdr] = mm;
        nargs += 1;
    }
}

/// The error [`resolve_call_chain`] would raise for calling `v`, found
/// without touching the stack.
pub(crate) fn call_chain_error<'gc>(ctx: Context<'gc>, mut v: Value<'gc>) -> Option<OpError<'gc>> {
    for _ in 0..=MAX_CALL_CHAIN {
        if v.get_function().is_some() {
            return None;
        }
        let mm = ctx.mm_of(v, MetamethodBits::CALL);
        if mm.is_nil() {
            return Some(OpError::Call(v));
        }
        v = mm;
    }
    Some(OpError::CallChainTooLong)
}

/// Binary metamethod `bit`, taken from `lhs` first, then `rhs`.
#[inline]
pub(crate) fn binop_metamethod<'gc>(
    ctx: Context<'gc>,
    lhs: Value<'gc>,
    rhs: Value<'gc>,
    bit: MetamethodBits,
) -> Value<'gc> {
    let m = ctx.mm_of(lhs, bit);
    if !m.is_nil() {
        return m;
    }
    ctx.mm_of(rhs, bit)
}
