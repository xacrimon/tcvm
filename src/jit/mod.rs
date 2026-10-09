//! The method JIT (`jit-design.md`): regions compiled from a prototype's
//! function entry or a hot loop header, entered and left as dispatch targets.

pub(crate) mod backend;
pub(crate) mod build;
pub(crate) mod compile;
pub(crate) mod feedback;
pub(crate) mod helpers;
pub(crate) mod ir;
pub(crate) mod layout;
pub(crate) mod opt;
pub(crate) mod region;
pub(crate) mod state;
#[cfg(test)]
mod tests;

use crate::dmm::Gc;
use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::jit::ir::ExitKind;
use crate::jit::region::{ExitInfo, Loc, Region, SRep};
use crate::jit::state::{
    EXIT_HOT, Entry, EntryState, JitStateRef, MAX_COMPILES, MAX_ENTRIES, MAX_RECOMPILES,
    MAX_STRIKES, jit_state,
};
use crate::lua::Context;
use crate::vm::abi::{Handler, Slot, handler};
use crate::vm::frame;

const PTR_MASK: u64 = (1 << 48) - 1;

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// A compiled function entry: tail into its region.
    op fn op_jit_entry {
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A compiled loop entry.
    op fn op_jit_loop {
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A counting instruction's counter ran out: compile its entry and enter
    /// it, or run the instruction. `insn` is the counting word, `pc` past it,
    /// and the frame stands exactly at it.
    slow fn jit_hot {
        let word = insn.as_insn();
        let jit = rt.jit();
        let counter = &jit.hot[word.hot_counter()];
        if !jit.config.enabled || jit.exhausted.get() {
            counter.set(u16::MAX);
            let h = rt.handler(word.opcode());
            tail!(h, insn = Slot::insn(word))
        }
        counter.set(jit.hot_start(word));
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let site = unsafe { pc.sub(1) };
        // Not when the word in `Code` is an entry an exit dispatched around.
        if unsafe { *site } == word {
            let at = unsafe { site.offset_from_unsigned(closure.code) } as u32;
            if let Some(entry) = try_compile(rt, closure, at, word, base) {
                let jw = unsafe { *site };
                let h = unsafe { as_handler(entry) };
                tail!(h, insn = Slot::insn(jw), closure = Slot::closure(closure))
            }
        }
        // The instruction counts again when it runs: not to trip at once.
        counter.set(jit.hot_start(word).saturating_add(1));
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// Leave a region (from `exit_common`): `insn` is the region and the
    /// exit, the register image is in the runtime. Rebuild the frame from the
    /// exit's snapshot and continue in the interpreter.
    slow fn jit_exit {
        let raw = insn.raw();
        let region: &Region<'gc> = unsafe { &*((raw & PTR_MASK) as *const Region<'gc>) };
        let k = (raw >> 48) as usize;
        let e = &region.exits[k];
        let mc = rt.mutation();
        let img = rt.jit().exit_regs();
        for s in region.snap(e) {
            let w = match s.loc {
                Loc::Reg(i) => img[i as usize].get(),
                Loc::Spill(i) => img[crate::jit::backend::aarch64::abi::IMAGE_SPILLS + i as usize].get(),
                Loc::Const(c) => region.consts[c as usize],
            };
            let v = match s.rep {
                SRep::Val => unsafe { Value::from_raw(w) },
                SRep::I32 => Value::small(w as i32),
                SRep::I64 => Value::integer(mc, w as i64),
                SRep::F64 => Value::float(f64::from_bits(w)),
                SRep::B1 => Value::boolean(w & 1 != 0),
            };
            unsafe { base.add(s.reg as usize).write(v) };
        }
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        on_exit(rt, closure, base, region, e);
        let at = if e.kind == ExitKind::Before { e.pc } else { e.pc + 1 };
        let resume = unsafe { closure.code.add(at as usize) };
        if e.kind == ExitKind::Gc || e.tag == crate::jit::ir::ops::ExitTag::Gc {
            pc = resume;
            exit!(crate::vm::abi::Exit::Gc)
        }
        let mut word = unsafe { *resume };
        if word.is_jit_word() {
            word = original_of(closure, word);
        }
        pc = unsafe { resume.add(1) };
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// An entry whose slot a hot exit pointed here: compile it again with the
    /// widened feedback and enter it, or blacklist it.
    slow fn jit_recompile {
        let word = insn.as_insn();
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let site = unsafe { pc.sub(1) };
        let at = unsafe { site.offset_from_unsigned(closure.code) } as u32;
        match recompile(rt, closure, at, word.d(), base) {
            Ok(entry) => {
                let h = unsafe { as_handler(entry) };
                tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
            }
            Err(original) => {
                let h = rt.handler(original.opcode());
                tail!(h, insn = Slot::insn(original), closure = Slot::closure(closure))
            }
        }
    }
}

/// `code` as the dispatch target it is.
///
/// # Safety
/// `code` is the entry of a region or a handler.
#[inline(always)]
pub(crate) unsafe fn as_handler(code: *const u8) -> Handler {
    unsafe { std::mem::transmute::<*const u8, Handler>(code) }
}

/// The word a JIT word replaced.
fn original_of(closure: LuaFn<'_>, word: Instruction) -> Instruction {
    let state = closure
        .proto
        .jit
        .get()
        .expect("a JIT word without a JIT state");
    let st = state.borrow();
    st.entries
        .iter()
        .find(|e| e.slot == word.d())
        .map(|e| e.original)
        .expect("a JIT word without its entry")
}

fn log(ctx: Context<'_>, msg: impl FnOnce() -> String) {
    if ctx.jit().config.log {
        eprintln!("jit: {}", msg());
    }
}

/// Compile the entry at `at` and install it; its entry address.
fn try_compile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    at: u32,
    word: Instruction,
    frame: *const Value<'gc>,
) -> Option<*const u8> {
    let config = &ctx.jit().config;
    if let Some(only) = &config.only
        && !crate::jit::compile::chunk_name(closure).contains(only.as_str())
    {
        return None;
    }
    let mc = ctx.mutation();
    let state = jit_state(mc, closure.proto);
    {
        let st = state.borrow();
        if st.refused.contains(&at) || st.compiles >= MAX_COMPILES || st.strikes >= MAX_STRIKES {
            return None;
        }
    }
    match compile_caught(ctx, closure, at, frame, &[]) {
        Ok(region) => {
            let entry = install(ctx, closure, state, at, word, region)?;
            log(ctx, || {
                format!(
                    "compiled {} pc{at} ({}), {} bytes, {} exits",
                    crate::jit::compile::chunk_name(closure),
                    word.op().name(),
                    region.code.len(),
                    region.exits.len()
                )
            });
            Some(entry)
        }
        Err(e) => {
            log(ctx, || {
                format!(
                    "compile of {} pc{at} failed: {e:?}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
            let mut st = state.borrow_mut(mc);
            if !matches!(e, crate::jit::compile::CompileError::Useless) {
                st.strikes += 1;
            }
            st.refused.push(at);
            None
        }
    }
}

fn compile_caught<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    at: u32,
    frame: *const Value<'gc>,
    seen: &[(u8, crate::jit::ir::types::TypeSet)],
) -> Result<Gc<'gc, Region<'gc>>, crate::jit::compile::CompileError> {
    #[cfg(target_arch = "aarch64")]
    {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::jit::compile::compile(ctx, closure, at, frame, seen)
        }));
        match r {
            Ok(r) => r,
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Err(crate::jit::compile::CompileError::Backend(format!(
                    "panic: {msg}"
                )))
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (ctx, closure, at, frame, seen);
        Err(crate::jit::compile::CompileError::Backend(
            "no backend".into(),
        ))
    }
}

/// Install a compiled region over the counting word at `at`.
fn install<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    state: JitStateRef<'gc>,
    at: u32,
    word: Instruction,
    region: Gc<'gc, Region<'gc>>,
) -> Option<*const u8> {
    let jit = ctx.jit();
    let slot = jit.alloc_slot(region.entry)?;
    let hot = word.hot_counter() as u8;
    let jw = if word.op() == Op::FUNC {
        Instruction::jit_entry(hot, slot)
    } else {
        Instruction::jit_loop(hot, slot)
    };
    let mc = ctx.mutation();
    let mut st = state.borrow_mut(mc);
    st.age += 1;
    let age = st.age;
    st.entries.push(Entry {
        pc: at,
        original: word,
        slot,
        region: Some(region),
        recompiles: 0,
        free_recompile: false,
        state: EntryState::Compiled,
        depth: (word.op() != Op::FUNC) as u16,
        age,
        seen: Vec::new(),
    });
    st.regions.push(region);
    st.compiles += 1;
    write_code(closure, at, jw);
    if st.entries.len() > MAX_ENTRIES {
        // The oldest loop entry other than this one.
        let victim = st
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.pc != at && e.depth > 0)
            .min_by_key(|(_, e)| e.age)
            .map(|(i, _)| i);
        if let Some(v) = victim {
            let e = st.entries.remove(v);
            retire_entry(ctx, closure, &e);
        }
    }
    Some(region.entry)
}

fn write_code(closure: LuaFn<'_>, at: u32, word: Instruction) {
    // SAFETY: `Code` keeps instructions in cells.
    unsafe { closure.code.add(at as usize).cast_mut().write(word) };
}

/// Put an entry's original word back and let go of its slot and region.
fn retire_entry(ctx: Context<'_>, closure: LuaFn<'_>, e: &Entry<'_>) {
    write_code(closure, e.pc, e.original);
    ctx.jit().free_slot(e.slot);
    if let Some(r) = e.region {
        r.retired.set(true);
    }
}

/// What an exit leaves behind: the operand kinds the guard saw, the count,
/// and a recompile request when the exit is hot.
fn on_exit<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    base: *mut Value<'gc>,
    region: &Region<'gc>,
    e: &ExitInfo,
) {
    use crate::jit::ir::ops::ExitTag;
    if let ExitTag::Entry(r) = e.tag {
        record_entry(ctx, closure, base, region, r);
    }
    if e.kind == ExitKind::Before && e.tag.widenable() {
        record_site(closure, base, e.pc, e.tag == ExitTag::Overflow);
    }
    let n = e.count.get() + 1;
    e.count.set(n);
    if n == 1 {
        log(ctx, || {
            format!(
                "first exit {} pc{} ({:?}) of the region at pc{}",
                crate::jit::compile::chunk_name(closure),
                e.pc,
                e.tag,
                region.entry_pc
            )
        });
    }
    if region.retired.get() {
        return;
    }
    if e.tag == ExitTag::NeverRan && n == 1 {
        request_recompile(ctx, closure, region, true);
    } else if e.tag.widenable() && n == EXIT_HOT {
        log(ctx, || {
            format!(
                "hot exit {} pc{} ({:?}) of the region at pc{}",
                crate::jit::compile::chunk_name(closure),
                e.pc,
                e.tag,
                region.entry_pc
            )
        });
        request_recompile(ctx, closure, region, false);
    }
}

/// Record the kinds of the operands of the site at `pc`, from the frame just
/// rebuilt, into its feedback byte.
fn record_site<'gc>(closure: LuaFn<'gc>, base: *mut Value<'gc>, pc: u32, overflow: bool) {
    use crate::instruction::Family;
    use crate::jit::feedback as fb;
    let site = unsafe { closure.code.add(pc as usize) };
    let mut i = unsafe { *site };
    if i.is_jit_word() {
        i = original_of(closure, i);
    }
    // SAFETY: the operands are registers of the frame's window.
    let reg = |r: u8| unsafe { *base.add(r as usize) };
    let kind = |r: u8| fb::kind(reg(r));
    let imm = |i: Instruction| if i.imm_is_int() { fb::SMALL } else { fb::FLOAT };
    let num = |v: Value<'_>| v.is_float() || v.get_integer().is_some();
    let mut bits = match i.op() {
        // The loop's kind: integer (and whether it is big) or float.
        Op::FORPREP => {
            let (init, limit, step) = (reg(i.a()), reg(i.a() + 1), reg(i.a() + 2));
            if init.get_integer().is_some() && step.get_integer().is_some() {
                let float_limit = if limit.is_float() { fb::FLOAT_LIMIT } else { 0 };
                fb::kind(init) | fb::kind(step) | (fb::kind(limit) & fb::BIGINT) | float_limit
            } else if num(init) && num(step) {
                fb::FLOAT
            } else {
                0
            }
        }
        Op::FORLOOP | Op::FORLOOP_I | Op::FORLOOP_F => kind(i.a() + 1) | kind(i.a() + 2),
        _ => 0,
    };
    bits |= match i.op().info().family {
        Family::RegArith | Family::RegBit | Family::CmpReg
            if i.op().info().family != Family::CmpReg =>
        {
            kind(i.b()) | kind(i.c())
        }
        Family::CmpReg => kind(i.a()) | kind(i.b()),
        Family::ImmArith | Family::ImmBit => kind(i.b()) | imm(i),
        Family::CmpImm => kind(i.a()),
        _ if i.op().branch_sense().is_some() => kind(i.a()),
        _ => 0,
    };
    if overflow {
        bits |= fb::OVERFLOW;
    }
    if bits != 0 {
        fb::record(closure, site, bits);
    }
}

/// Record the kind of `R[r]` a failed entry guard saw on its entry.
fn record_entry<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    base: *mut Value<'gc>,
    region: &Region<'gc>,
    r: u8,
) {
    let Some(state) = closure.proto.jit.get() else {
        return;
    };
    // SAFETY: an entry guard tests a live register of the frame.
    let kind = crate::jit::build::value_set(unsafe { *base.add(r as usize) });
    let mut st = state.borrow_mut(ctx.mutation());
    let me = region as *const Region<'gc>;
    if let Some(e) = st
        .entries
        .iter_mut()
        .find(|e| e.region.is_some_and(|x| Gc::as_ptr(x) == me))
    {
        match e.seen.iter_mut().find(|(x, _)| *x == r) {
            Some((_, s)) => *s |= kind,
            None => e.seen.push((r, kind)),
        }
    }
}

fn request_recompile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    region: &Region<'gc>,
    free: bool,
) {
    let Some(state) = closure.proto.jit.get() else {
        return;
    };
    let mc = ctx.mutation();
    let mut st = state.borrow_mut(mc);
    let me = region as *const Region<'gc>;
    if let Some(e) = st
        .entries
        .iter_mut()
        .find(|e| e.region.is_some_and(|r| Gc::as_ptr(r) == me))
        && e.state == EntryState::Compiled
    {
        e.state = EntryState::Recompile;
        e.free_recompile |= free;
        ctx.jit().set_slot(e.slot, jit_recompile as *const u8);
    }
}

/// Recompile the entry in `slot` at `at`: its new entry address, or the
/// original word to run when it was blacklisted instead.
fn recompile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    at: u32,
    slot: u16,
    frame: *const Value<'gc>,
) -> Result<*const u8, Instruction> {
    let mc = ctx.mutation();
    let state = closure
        .proto
        .jit
        .get()
        .expect("a recompile without a JIT state");
    let (original, budget_left, seen) = {
        let st = state.borrow();
        let e = st
            .entries
            .iter()
            .find(|e| e.slot == slot)
            .expect("a recompile without its entry");
        (
            e.original,
            e.free_recompile || e.recompiles < MAX_RECOMPILES,
            e.seen.clone(),
        )
    };
    let result = if budget_left && state.borrow().compiles < MAX_COMPILES {
        compile_caught(ctx, closure, at, frame, &seen).map_err(|e| {
            log(ctx, || {
                format!(
                    "recompile of {} pc{at} failed: {e:?}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
        })
    } else {
        Err(())
    };
    let mut st = state.borrow_mut(mc);
    let ei = st.entries.iter().position(|e| e.slot == slot).unwrap();
    match result {
        Ok(region) => {
            let e = &mut st.entries[ei];
            if let Some(old) = e.region {
                old.retired.set(true);
            }
            if !e.free_recompile {
                e.recompiles += 1;
            }
            e.free_recompile = false;
            e.region = Some(region);
            e.state = EntryState::Compiled;
            st.regions.push(region);
            st.compiles += 1;
            ctx.jit().set_slot(slot, region.entry);
            log(ctx, || {
                format!(
                    "recompiled {} pc{at}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
            Ok(region.entry)
        }
        Err(()) => {
            let e = st.entries.remove(ei);
            retire_entry(ctx, closure, &e);
            st.refused.push(at);
            log(ctx, || {
                format!(
                    "blacklisted {} pc{at}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
            Err(original)
        }
    }
}
