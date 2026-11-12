//! Tiering: when entries compile, entering and leaving their regions as
//! dispatch targets, and installing, retiring and recompiling them.

use crate::dmm::Gc;
use crate::env::function::{LuaFn, Prototype};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::jit::ir::ExitKind;
use crate::jit::region::{ExitInfo, Loc, Region, SRep};
use crate::jit::state::{
    EXIT_HOT, Entry, EntryState, JitState, JitStateRef, MAX_COMPILES, MAX_ENTRIES, MAX_RETRIES,
    MAX_STRIKES, jit_state,
};
use crate::lua::Context;
use crate::vm::abi::{Handler, Slot, handler};
use crate::vm::frame;

const PTR_MASK: u64 = (1 << 48) - 1;

pub(crate) use crate::jit::backend::target::abi::{RA_TAG, REG_ARGS, spill_addr};

/// A hash map with a fast, unkeyed hasher, for compile-time tables.
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, foldhash::fast::FixedState>;

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// A compiled function entry: tail into its region, unless the native stack
    /// is past its budget: the function runs interpreted, which takes no native
    /// stack.
    op fn op_jit_entry {
        if std::hint::unlikely(below_budget(thread!())) {
            next!()
        }
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A compiled loop entry, or the instruction it replaced past the
    /// budget.
    op fn op_jit_loop {
        if std::hint::unlikely(below_budget(thread!())) {
            let word = original_of(closure.proto, insn);
            let h = rt.handler(word.opcode());
            tail!(h, insn = Slot::insn(word))
        }
        let code = rt.jit_entry(insn.d());
        tail!(code)
    }

    /// A counting instruction's counter ran out: compile its entry and enter
    /// it, or run the instruction. `insn` is the counting word, `pc` past it,
    /// and the frame stands exactly at it.
    slow fn jit_hot {
        let word = insn.as_insn();
        let jit = rt.jit();
        // A function's own invocation count trips it (`call_mask`).
        let func = word.op() == Op::FUNC;
        let counter = &jit.hot[word.hot_counter()];
        if !jit.config.enabled || jit.exhausted.get() {
            if func {
                jit.call_mask.set(u64::MAX);
                next!()
            }
            counter.set(u16::MAX);
            let h = rt.handler(word.opcode());
            tail!(h, insn = Slot::insn(word))
        }
        if !func {
            counter.set(jit.hot_start(word));
        }
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let site = unsafe { pc.sub(1).offset_from_unsigned(closure.code) } as u32;
        if let Some(at) = entry_site(closure, site, word) {
            let w = unsafe { *closure.code.add(at as usize) };
            if let Some(entry) = try_compile(rt, closure, at, w, base) {
                let jw = unsafe { *closure.code.add(at as usize) };
                let h = unsafe { as_handler(entry) };
                let pc = unsafe { closure.code.add(at as usize + 1) };
                tail!(h, pc = pc, insn = Slot::insn(jw), closure = Slot::closure(closure))
            }
        }
        // A FUNC has counted this invocation: go on past it.
        if func {
            next!()
        }
        // The instruction counts again when it runs: not to trip at once.
        counter.set(jit.hot_start(word).saturating_add(1));
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// Leave a region (from `exit_common`, its frame popped): `insn` is the
    /// region and the exit, the register image is in the runtime. Rebuild
    /// the frame from the exit's snapshot and continue in the interpreter,
    /// in a level of its own when compiled code had called the frame.
    slow fn jit_exit {
        let raw = insn.raw();
        let region: &Region<'gc> = unsafe { &*((raw & PTR_MASK) as *const Region<'gc>) };
        let k = (raw >> 48) as usize;
        let e = &region.exits[k];
        #[cfg(feature = "tcvm-dev")]
        if let Some(f) = region.fired.get(k) {
            f.set(1);
        }
        let mc = rt.mutation();
        let img = rt.jit().exit_regs();
        let snap = region.snap(e);
        // A level returning to a self-call whose result guard runs out of
        // line returns to its landing: the interpreter's result is any value.
        if !region.landings.is_empty() {
            let rec = thread!().record as *mut crate::vm::stack::EntryRecord;
            let start = region.code.rx() as usize;
            // SAFETY: the level `exit_common` runs in has its record there.
            unsafe {
                let lr = (*rec).lr;
                let off = (lr & !RA_TAG).wrapping_sub(start);
                if let Some(&(_, to)) = region.landings.iter().find(|l| l.0 as usize == off) {
                    (*rec).lr = (start + to as usize) | (lr & RA_TAG);
                }
            }
        }
        // A moved exit's blame, from the frame as the exit finds it.
        let blame: Vec<(u8, Value<'gc>)> = region.blame(e).map_or_else(Vec::new, |b| {
            (b.entries())
                .map(|s| {
                    let w = match s.loc {
                        Loc::Reg(i) => img[i as usize].get(),
                        Loc::Spill(i) => img[crate::jit::backend::target::abi::IMAGE_SPILLS + i as usize].get(),
                        // SAFETY: a slot of the region's window.
                        Loc::Slot(r) => unsafe { *(base.add(r as usize) as *const u64) },
                        loc => const_word(region, loc),
                    };
                    (s.reg, entry_value(mc, s.rep, w))
                })
                .collect()
        });
        // A slot one entry reads another may write: those are read first.
        let mut held = [const { std::mem::MaybeUninit::<u64>::uninit() }; 256];
        for s in snap.entries() {
            if let Loc::Slot(r) = s.loc {
                held[s.reg as usize].write(unsafe { *(base.add(r as usize) as *const u64) });
            }
        }
        for s in snap.entries() {
            let w = match s.loc {
                Loc::Reg(i) => img[i as usize].get(),
                Loc::Spill(i) => img[crate::jit::backend::target::abi::IMAGE_SPILLS + i as usize].get(),
                // SAFETY: written by the loop above.
                Loc::Slot(_) => unsafe { held[s.reg as usize].assume_init() },
                loc => const_word(region, loc),
            };
            unsafe { base.add(s.reg as usize).write(entry_value(mc, s.rep, w)) };
        }
        // Inlined calls' frames, outermost first, each a callee of the one
        // around it; one known by its prototype is the value the entries
        // wrote to its header.
        let mut fbase = base;
        let mut fclosure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let region_closure = fclosure;
        for fr in snap.frames() {
            debug_assert!(fr.delta != 0 || fr.func & crate::jit::region::DYNAMIC == 0);
            let f = if fr.func & crate::jit::region::DYNAMIC != 0 {
                unsafe { *fbase.add(fr.a as usize) }
            } else {
                region.pool[fr.func as usize]
            };
            let callee = f
                .get_function()
                .and_then(|f| f.as_lua())
                .expect("an inlined Lua closure");
            // A tail callee in the region frame's place: its header, as the
            // interpreter's `TAILCALL` leaves it.
            if fr.delta == 0 {
                unsafe {
                    frame::set_func_word(fbase, frame::lua_func_word(callee, 0));
                    if fr.open {
                        frame::set_flags(fbase, frame::flag::HAS_OPEN);
                    }
                }
                fclosure = callee;
                continue;
            }
            debug_assert_eq!(fr.delta as usize, fr.a as usize + frame::HDR);
            unsafe {
                let nb = fbase.add(fr.delta as usize);
                let open = if fr.open { frame::flag::HAS_OPEN } else { 0 };
                frame::write_hdr(
                    nb.sub(frame::HDR),
                    frame::lua_func_word(callee, 0),
                    crate::vm::abi::handler_bits(rt.ret(fr.c)) | open,
                    fbase,
                    fclosure.code.add(fr.caller_pc as usize + 1),
                );
                fbase = nb;
            }
            fclosure = callee;
        }
        // A fast-called frame's header has no continuation: it returns through
        // the level `exit_common` started.
        unsafe {
            let w = frame::ret_word(base);
            if w & !frame::flag::MASK == 0 {
                let r = crate::jit::backend::target::abi::ret_to_native;
                frame::set_ret_word(base, w | crate::vm::abi::handler_bits(r));
            }
        }
        base = fbase;
        let closure = fclosure;
        if region.invalidated.get() {
            retire_invalidated(rt, region_closure, region);
        }
        on_exit(rt, region_closure, closure, base, region, e, snap.kind, snap.pc, &blame);
        let at = if snap.kind == ExitKind::Before { snap.pc } else { snap.pc + 1 };
        let resume = unsafe { closure.code.add(at as usize) };
        if e.tag == crate::jit::ir::ops::ExitTag::Gc {
            tail!(crate::vm::dispatch::gc_park, pc = resume, insn = Slot::from_raw(0), closure = Slot::closure(closure))
        }
        let mut word = unsafe { *resume };
        if word.is_jit_word() {
            word = original_of(closure.proto, word);
        }
        pc = unsafe { resume.add(1) };
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// A failed entry guard (from the region's entry-fail stub, `insn` the
    /// region, its frame popped and nothing else changed): note the entry's
    /// kinds and run the original instruction.
    slow fn jit_entry_fail {
        let region: &Region<'gc> = unsafe { &*(insn.raw() as *const Region<'gc>) };
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        on_entry_fail(rt, closure, base, region);
        let (site, word) = entry_word(closure, region, false);
        pc = unsafe { site.add(1) };
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// An interpreter entry whose inlined frames' window does not fit (from
    /// its stub, `insn` the region, its frame not open): grow the stack and
    /// enter again, else run the original instruction, which overflows only
    /// where the frames do.
    slow fn jit_window {
        let region: &Region<'gc> = unsafe { &*(insn.raw() as *const Region<'gc>) };
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let ts = thread!();
        let end = ts.slot_index(base) + region.window as usize;
        let (site, word) = entry_word(closure, region, ts.ensure_frame_slots(end));
        pc = unsafe { site.add(1) };
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// The entry of a region a broken dependency invalidated
    /// (`Region::invalidate`), from its stub with `insn` the region and its
    /// frame not open: retire the entry and run the original word.
    slow fn jit_dead_entry {
        let region: &Region<'gc> = unsafe { &*(insn.raw() as *const Region<'gc>) };
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        retire_invalidated(rt, closure, region);
        let (site, word) = entry_word(closure, region, false);
        pc = unsafe { site.add(1) };
        let h = rt.handler(word.opcode());
        tail!(h, insn = Slot::insn(word), closure = Slot::closure(closure))
    }

    /// An entry whose slot a hot exit pointed here: run the original word
    /// while its feedback settles, then compile it again and enter it, or
    /// blacklist it.
    slow fn jit_settle {
        let word = insn.as_insn();
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let site = unsafe { pc.sub(1) };
        let at = unsafe { site.offset_from_unsigned(closure.code) } as u32;
        match settle(rt, closure, at, word.d(), base) {
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

/// The word of a constant snapshot location.
pub(crate) fn const_word(region: &Region<'_>, loc: Loc) -> u64 {
    match loc {
        Loc::Const(c) => region.consts[c as usize],
        Loc::Nil => Value::nil().to_raw(),
        Loc::False => Value::boolean(false).to_raw(),
        Loc::True => Value::boolean(true).to_raw(),
        Loc::Reg(_) | Loc::Spill(_) | Loc::Slot(_) => {
            unreachable!("not a constant location")
        }
    }
}

/// A snapshot entry's word as the value it stands for.
fn entry_value<'gc>(mc: &crate::dmm::Mutation<'gc>, rep: SRep, w: u64) -> Value<'gc> {
    match rep {
        SRep::Val => unsafe { Value::from_raw(w) },
        SRep::I32 => Value::small(w as i32),
        SRep::I64 => Value::integer(mc, w as i64),
        SRep::F64 => Value::float(f64::from_bits(w)),
        SRep::B1 => Value::boolean(w & 1 != 0),
    }
}

/// Whether the native stack pointer is past the thread's budget, where
/// compiled code refuses entry.
#[inline(always)]
fn below_budget(ts: &crate::env::thread::ThreadState<'_>) -> bool {
    crate::jit::backend::target::abi::stack_pointer() < ts.native_limit
}

/// `code` as the dispatch target it is.
///
/// # Safety
/// `code` is the entry of a region or a handler.
#[inline(always)]
pub(crate) unsafe fn as_handler(code: *const u8) -> Handler {
    unsafe { std::mem::transmute::<*const u8, Handler>(code) }
}

/// The entry the counting `word` at `site` compiles, the frame being the
/// same there: the `LOOP` a back edge jumps past, else the word itself (a
/// `goto`'s `JMP_BACK` too). None when a JIT word is there already (an exit
/// dispatched `word` around it), or the back edge is an entry's.
fn entry_site(closure: LuaFn<'_>, site: u32, word: Instruction) -> Option<u32> {
    let code = |pc: u32| unsafe { *closure.code.add(pc as usize) };
    let at = match word.op() {
        Op::JMP_BACK | Op::TFORLOOP => {
            if let Some(state) = closure.proto.jit.get()
                && state
                    .borrow()
                    .entries
                    .iter()
                    .any(|e| e.back_edges.contains(&site))
            {
                return None;
            }
            let t = (site as i64 + 1 + word.branch_offset() as i64) as u32;
            match code(t - 1) {
                h if h.op() == Op::LOOP => t - 1,
                h if h.is_jit_word() && original_of(closure.proto, h).op() == Op::LOOP => {
                    return None;
                }
                _ => site,
            }
        }
        _ => site,
    };
    (!code(at).is_jit_word()).then_some(at)
}

/// The site of `region`'s entry in `closure` and the word to run there: the
/// entry's own to `enter` it again, else the instruction it replaced.
fn entry_word<'gc>(
    closure: LuaFn<'gc>,
    region: &Region<'gc>,
    enter: bool,
) -> (*const Instruction, Instruction) {
    // SAFETY: the entry is an instruction of the region's function.
    let site = unsafe { closure.code.add(region.entry_pc as usize) };
    let word = unsafe { *site };
    if word.is_jit_word() && !enter {
        (site, original_of(closure.proto, word))
    } else {
        (site, word)
    }
}

/// The word a JIT word replaced.
pub(super) fn original_of(proto: Gc<'_, Prototype<'_>>, word: Instruction) -> Instruction {
    let state = proto.jit.get().expect("a JIT word without a JIT state");
    let st = state.borrow();
    st.entries
        .iter()
        .find(|e| e.slot == word.d())
        .map(|e| e.original)
        .expect("a JIT word without its entry")
}

fn log(ctx: Context<'_>, msg: impl FnOnce() -> String) {
    #[cfg(feature = "tcvm-dev")]
    if ctx.jit().config.dev.log {
        eprintln!("jit: {}", msg());
    }
    #[cfg(not(feature = "tcvm-dev"))]
    let _ = (ctx, msg);
}

/// Compile the entry at `at` and install it; its entry address.
fn try_compile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    at: u32,
    word: Instruction,
    frame: *const Value<'gc>,
) -> Option<*const u8> {
    #[cfg(feature = "tcvm-dev")]
    if let Some(only) = &ctx.jit().config.dev.only
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
                    "compile of {} pc{at} failed: {e}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
            #[cfg(feature = "tcvm-dev")]
            if e.internal() {
                let n = &ctx.jit().internal_errors;
                n.set(n.get() + 1);
            }
            if matches!(e, crate::jit::compile::CompileError::OutOfCode) {
                ctx.jit().exhausted.set(true);
                return None;
            }
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
    let r = crate::vm::stack::on_host(ctx, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::jit::compile::compile(ctx, closure, at, frame, seen)
        }))
    });
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
    // A loop's back edges, which jump past its `LOOP`, re-enter it now.
    let mut back_edges = Vec::new();
    if word.op() == Op::LOOP {
        for pc in 0..closure.proto.code.len() as u32 {
            let mut i = unsafe { *closure.code.add(pc as usize) };
            if matches!(i.op(), Op::JMP_BACK | Op::TFORLOOP)
                && pc as i64 + 1 + i.branch_offset() as i64 == at as i64 + 1
                && !st.entries.iter().any(|e| e.back_edges.contains(&pc))
            {
                assert!(i.set_branch_offset(i.branch_offset() - 1));
                write_code(closure, pc, i);
                back_edges.push(pc);
            }
        }
    }
    st.entries.push(Entry {
        pc: at,
        original: word,
        slot,
        region: Some(region),
        retries: 0,
        free_recompile: false,
        settle: 0,
        state: EntryState::Compiled,
        nest: region.nest.clone(),
        age,
        back_edges: back_edges.into_boxed_slice(),
        seen: Vec::new(),
    });
    st.regions.push(region);
    st.compiles += 1;
    write_code(closure, at, jw);
    if !region.fast_entry.is_null() {
        set_fast(ctx, &closure.proto, &mut st, region.fast_entry as usize);
    }
    if st.entries.len() > MAX_ENTRIES {
        // A loop entry inside another live entry's loop, whose region covers it
        // but for the rest of one outer iteration after a deopt: the deepest
        // such, else the oldest loop entry, never this one.
        let inside = |e: &Entry<'_>| {
            st.entries
                .iter()
                .any(|o| o.pc != e.pc && !o.nest.is_empty() && e.nest[1..].contains(&o.nest[0]))
        };
        let victim = st
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.pc != at && !e.nest.is_empty())
            .max_by_key(|(_, e)| {
                let n = inside(e);
                (
                    n,
                    if n { e.nest.len() } else { 0 },
                    std::cmp::Reverse(e.age),
                )
            })
            .map(|(i, _)| i);
        if let Some(v) = victim {
            let e = st.entries.remove(v);
            retire_entry(ctx, closure, &mut st, &e);
        }
    }
    Some(region.entry)
}

/// Set the prototype's fast entry (0 for none) and point the direct calls
/// of it there (`direct_target`); `st` is its state.
fn set_fast<'gc>(ctx: Context<'gc>, proto: &Prototype<'gc>, st: &mut JitState<'gc>, entry: usize) {
    proto.jit_fast.set(entry);
    st.callers
        .retain(|&(r, off, stub)| match r.upgrade(ctx.mutation()) {
            Some(r) => {
                let target = direct_target(entry, stub.is_some(), proto.num_params);
                match (stub, target) {
                    (Some(_), Some(t)) => r.code.patch_call(off as usize, t),
                    // Not a call that misses: nothing is pushed for the
                    // return predictor, and no site lookup runs.
                    (Some(s), None) => r.code.patch_jump(off as usize, s as usize),
                    (None, t) => r
                        .code
                        .repoint_branch(off as usize, t.unwrap_or(tail_miss())),
                }
                true
            }
            None => false,
        });
}

/// The known entry (the target's `known_offset`) of a prototype of
/// `nparams` parameters whose fast entry is `fast`, none without one.
pub(crate) fn direct_target(fast: usize, regs: bool, nparams: u8) -> Option<usize> {
    (fast != 0).then(|| fast + crate::jit::backend::target::emit::known_offset(regs, nparams))
}

/// Where a direct tail call branches while its callee has no fast entry.
pub(crate) fn tail_miss() -> usize {
    crate::jit::backend::target::abi::direct_miss as *const () as usize
}

/// Where the fast call whose return address is `ra` goes when its callee's
/// entry checks fail (`abi::call_miss`): the call's slow path.
pub(crate) extern "C" fn jit_call_miss(ra: usize) -> usize {
    // SAFETY: a fast call's return address lies in its region's code.
    let region = unsafe { &*crate::jit::region::Region::of_return(ra) };
    let (site, _) = region.site(ra).expect("a fast call's site");
    region.code.rx() as usize + site.miss as usize
}

fn write_code(closure: LuaFn<'_>, at: u32, word: Instruction) {
    // SAFETY: `Code` keeps instructions in cells.
    unsafe { closure.code.add(at as usize).cast_mut().write(word) };
}

/// Put an entry's original word back, and its back edges past it, and let
/// go of its slot and region; `st` is the prototype's state.
pub(super) fn retire_entry<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    st: &mut JitState<'gc>,
    e: &Entry<'gc>,
) {
    write_code(closure, e.pc, e.original);
    if e.pc == 0 {
        set_fast(ctx, &closure.proto, st, 0);
    }
    for &pc in &e.back_edges {
        let mut i = unsafe { *closure.code.add(pc as usize) };
        assert!(i.set_branch_offset(i.branch_offset() + 1));
        write_code(closure, pc, i);
    }
    ctx.jit().free_slot(e.slot);
    if let Some(r) = e.region {
        r.retired.set(true);
    }
}

/// Retire the entry of invalidated `region` of `closure`'s prototype, unless
/// a recompile replaced it already.
fn retire_invalidated<'gc>(ctx: Context<'gc>, closure: LuaFn<'gc>, region: &Region<'gc>) {
    if region.retired.replace(true) {
        return;
    }
    log(ctx, || {
        format!(
            "invalidated {} pc{}",
            crate::jit::compile::chunk_name(closure),
            region.entry_pc
        )
    });
    let Some(state) = closure.proto.jit.get() else {
        return;
    };
    let mut st = state.borrow_mut(ctx.mutation());
    let me = region as *const Region<'gc>;
    if let Some(k) = st
        .entries
        .iter()
        .position(|e| e.region.is_some_and(|r| Gc::as_ptr(r) == me))
    {
        let e = st.entries.remove(k);
        retire_entry(ctx, closure, &mut st, &e);
    }
}

/// What an exit leaves behind: the operand kinds the guard saw, the count, and
/// a recompile request when the exit is hot. `closure` and `base` are the
/// innermost frame's, an inlined callee's when the exit is in its body;
/// `region_closure` the region's. The feedback is for the exit's site, whose
/// operands a moved exit's `blame` holds.
#[allow(clippy::too_many_arguments)]
fn on_exit<'gc>(
    ctx: Context<'gc>,
    region_closure: LuaFn<'gc>,
    closure: LuaFn<'gc>,
    base: *mut Value<'gc>,
    region: &Region<'gc>,
    e: &ExitInfo,
    kind: ExitKind,
    pc: u32,
    blame: &[(u8, Value<'gc>)],
) {
    use crate::jit::ir::ops::ExitTag;
    let (proto, at) = (site_proto(region, e, closure), e.site().unwrap_or(pc));
    let ops = match e.site() {
        None => Operands::Frame(closure, base),
        Some(_) => Operands::Blame(blame),
    };
    // The interpreter raises on such arguments: the code is as it should be.
    if matches!(e.tag, ExitTag::Arg | ExitTag::ArgCmp)
        && kind == ExitKind::Before
        && call_arg_raises(proto, at, ops, e.tag == ExitTag::ArgCmp)
    {
        return;
    }
    // A type guard standing for an array load's miss guard
    // (`opt::miss::covered`), its element nil under a class with `__index`:
    // the load's miss, which tells the use's kinds nothing. The interpreter
    // records it (`ARR_MISS`, `gettable_general`) as it runs the load again.
    let missed = (ops.named(crate::jit::ir::BLAME_TESTED)).is_some_and(|v| v.is_nil())
        && (ops.named(crate::jit::ir::BLAME_TABLE))
            .is_some_and(|t| !ctx.mm_of(t, crate::env::MetamethodBits::INDEX).is_nil());
    // A `Before` exit's operands are in their home slots or its blame; the
    // other kinds follow a completed instruction and have no failed guard,
    // as a lazy exit has none.
    if kind == ExitKind::Before && e.tag != ExitTag::Lazy && !missed {
        let extra = match e.tag {
            ExitTag::Overflow => crate::jit::feedback::OVERFLOW,
            ExitTag::Mm => crate::jit::feedback::MM,
            ExitTag::Bounds => crate::jit::feedback::ARR_MISS,
            ExitTag::OtherResult => crate::jit::feedback::OTHER_RESULT,
            // A closure of the prototype a guard took one closure of: the
            // next compile guards the prototype.
            ExitTag::Const if callee_of_proto(ctx, proto, at, ops) => 0,
            ExitTag::Const => match upvalue_of(proto, at) {
                // Every site of the upvalue loads it from now on.
                Some(k) => {
                    proto.feedback.set_upvalue_varies(k);
                    0
                }
                None => crate::jit::feedback::CONST_MISS,
            },
            _ => 0,
        };
        let extra = extra
            | if matches!(e.tag, ExitTag::Type | ExitTag::Arg | ExitTag::ArgCmp) {
                crate::jit::feedback::KIND_EXIT
            } else {
                0
            };
        record_site(proto, at, ops, extra);
        if e.tag == ExitTag::Shape {
            record_shape(ctx, proto, at, ops);
        }
    }
    let n = e.count.get() + 1;
    e.count.set(n);
    if n == 1 {
        log(ctx, || {
            format!(
                "first exit {}:{} pc{} ({:?}) of the region at pc{}",
                String::from_utf8_lossy(proto.source.as_bytes()),
                proto.line_defined,
                at,
                e.tag,
                region.entry_pc
            )
        });
    }
    if region.retired.get() {
        return;
    }
    if e.tag == ExitTag::NeverRan && n == 1 {
        request_recompile(ctx, region_closure, region, true);
    } else if e.tag.widenable() && n == region.exit_hot.get() {
        log(ctx, || {
            format!(
                "hot exit {}:{} pc{} ({:?}) of the region at pc{}",
                String::from_utf8_lossy(proto.source.as_bytes()),
                proto.line_defined,
                at,
                e.tag,
                region.entry_pc
            )
        });
        // Feedback only grows (kinds, the overflow, misses and metamethod
        // bits, shapes up to `MAX_SHAPES + 1`), so a recompile that learns
        // some is outside the budget, as the never-executed rule's are.
        let learned =
            missed || exit_seen(site_proto(region, e, closure), e.site().unwrap_or(pc)) != e.seen;
        request_recompile(ctx, region_closure, region, learned);
    }
}

/// The function whose pc `e.site` is: `closure`'s unless the exit moved out
/// of an inlined body.
fn site_proto<'gc>(
    region: &Region<'gc>,
    e: &ExitInfo,
    closure: LuaFn<'gc>,
) -> Gc<'gc, Prototype<'gc>> {
    match e.site_proto {
        u16::MAX => closure.proto,
        k => region.site_protos[k as usize],
    }
}

/// The by-value upvalue `GETUPVAL` at `pc` reads.
fn upvalue_of(proto: Gc<'_, Prototype<'_>>, pc: u32) -> Option<usize> {
    // SAFETY: `pc` is an instruction of `proto`.
    let i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    (i.op() == Op::GETUPVAL).then(|| i.b() as usize)
}

/// What a hot exit at `pc` compares with its copy from the compile
/// (`ExitInfo::seen`) to tell whether a recompile learns something: the
/// site's feedback, or for a `GETUPVAL` whether its upvalue varies.
pub(crate) fn exit_seen(proto: Gc<'_, Prototype<'_>>, pc: u32) -> u64 {
    // SAFETY: `pc` is an instruction of `proto`.
    let i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    if i.op() == Op::GETUPVAL {
        return proto.feedback.upvalue_varies(i.b() as usize) as u64;
    }
    proto.feedback.seen(pc as usize)
}

/// Whether the callee of the call at `pc`, whose guard failed, is a closure
/// of the prototype the site's callee word names (or of one that made
/// several, the word emptied by its function's death): then the word names
/// the prototype (`call_note`), which the next compile guards instead.
fn callee_of_proto<'gc>(
    ctx: Context<'gc>,
    proto: Gc<'gc, Prototype<'gc>>,
    pc: u32,
    ops: Operands<'_, 'gc>,
) -> bool {
    use crate::env::feedback::{MEGAMORPHIC, proto_slot, slot_proto};
    let mut i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    if i.is_jit_word() {
        i = original_of(proto, i);
    }
    let r = match i.op() {
        Op::CALL | Op::CALL_R0 | Op::CALL_R1 | Op::TAILCALL => i.a(),
        Op::CALLS | Op::CALLS_R0 | Op::CALLS_R1 => i.e() as u8,
        _ => return false,
    };
    let Some(fv) = ops.reg(r) else {
        return false;
    };
    let Some(lf) = fv.get_function().and_then(|f| f.as_lua()) else {
        return false;
    };
    let Some(slot) = proto.feedback.callee(pc as usize) else {
        return false;
    };
    let seen = slot.get();
    let same = match slot_proto(seen) {
        Some(p) => Gc::ptr_eq(p, lf.proto),
        None if seen == 0 => lf.proto.closures.get() > 1,
        None if seen == MEGAMORPHIC => false,
        // SAFETY: any other word names a live function.
        None => unsafe { Value::from_raw(seen) }
            .get_function()
            .and_then(|f| f.as_lua())
            .is_some_and(|c| Gc::ptr_eq(c.proto, lf.proto)),
    };
    if same {
        slot.set(proto_slot(lf.proto));
        ctx.mutation().backward_barrier(Gc::erase(proto), None);
    }
    same
}

/// A shape exit at a field site: its receiver's shape, for a chain in the
/// next compile.
fn record_shape<'gc>(
    ctx: Context<'gc>,
    proto: Gc<'gc, Prototype<'gc>>,
    pc: u32,
    ops: Operands<'_, 'gc>,
) {
    let mut i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    if i.is_jit_word() {
        i = original_of(proto, i);
    }
    let recv = match i.op().unquickened() {
        Op::GETTABUP | Op::SETTABUP => ops.upval_table(i.b(), false),
        Op::GETTABUP_REF | Op::SETTABUP_REF => ops.upval_table(i.b(), true),
        Op::GETFIELD | Op::SELF | Op::SETFIELD => ops.reg(i.b()),
        _ => return,
    };
    let Some(t) = recv.and_then(|v| v.get_table()) else {
        return;
    };
    // A table on a deprecated shape moves now: the interpreter's entry may
    // still hold that shape, and the next compile guards the poly one.
    if t.shape().is_deprecated() {
        t.refresh_meta(ctx);
    }
    proto.feedback.add_shape(pc as usize, t.shape());
    ctx.mutation().backward_barrier(Gc::erase(proto), None);
}

/// Whether the instruction at `pc` is a call whose arguments a builtin's
/// `ExitTag::Arg` guard raises on: one that is no number and no string, or
/// (`cmp`: `min`, `max`, which compare them, `__lt` ordering any other kind)
/// strings mixed with numbers.
fn call_arg_raises<'gc>(
    proto: Gc<'gc, Prototype<'gc>>,
    pc: u32,
    ops: Operands<'_, 'gc>,
    cmp: bool,
) -> bool {
    let mut i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    if i.is_jit_word() {
        i = original_of(proto, i);
    }
    let call = matches!(
        i.op(),
        Op::CALL
            | Op::CALL_R0
            | Op::CALL_R1
            | Op::CALLS
            | Op::CALLS_R0
            | Op::CALLS_R1
            | Op::TAILCALL
    );
    if !call {
        return false;
    }
    let (mut num, mut str) = (false, false);
    for r in 4..i.b().saturating_add(3) {
        let Some(v) = ops.reg(i.a() + r) else {
            return false;
        };
        if v.is_float() || v.get_integer().is_some() {
            num = true;
        } else if v.get_string().is_some() {
            str = true;
        } else {
            return !cmp;
        }
    }
    cmp && num && str
}

/// Record the kinds of the operands of the site at `pc` into its feedback;
/// an operand `ops` lacks records none.
fn record_site<'gc>(
    proto: Gc<'gc, Prototype<'gc>>,
    pc: u32,
    ops: Operands<'_, 'gc>,
    extra: crate::jit::feedback::Bits,
) {
    use crate::instruction::Family;
    use crate::jit::feedback as fb;
    let mut i = unsafe { *proto.code.as_ptr().add(pc as usize) };
    if i.is_jit_word() {
        i = original_of(proto, i);
    }
    let reg = |r: u8| ops.reg(r);
    let kind = |r: u8| reg(r).map_or(0, fb::kind);
    let imm = |i: Instruction| if i.imm_is_int() { fb::SMALL } else { fb::FLOAT };
    let num = |v: Value<'_>| v.is_float() || v.get_integer().is_some();
    let mut bits = match i.op() {
        // The loop's kind: integer (and whether it is big) or float.
        Op::FORPREP => match (reg(i.a()), reg(i.a() + 1), reg(i.a() + 2)) {
            (Some(init), Some(limit), Some(step))
                if init.get_integer().is_some() && step.get_integer().is_some() =>
            {
                let float_limit = if limit.is_float() { fb::FLOAT_LIMIT } else { 0 };
                fb::kind(init) | fb::kind(step) | (fb::kind(limit) & fb::BIGINT) | float_limit
            }
            (Some(init), Some(_), Some(step)) if num(init) && num(step) => fb::FLOAT,
            _ => 0,
        },
        Op::FORLOOP | Op::FORLOOP_I | Op::FORLOOP_F => kind(i.a() + 1) | kind(i.a() + 2),
        // An intrinsic's arguments: kinds it did not cover make the next
        // compile take them. A variable count records only the first.
        Op::CALL
        | Op::CALL_R0
        | Op::CALL_R1
        | Op::CALLS
        | Op::CALLS_R0
        | Op::CALLS_R1
        | Op::TAILCALL => {
            let rest = (5..i.b().saturating_add(3)).fold(0, |k, r| k | kind(i.a() + r));
            kind(i.a() + 4) | rest << fb::ARGS_SHIFT
        }
        Op::GETTABLE | Op::SETTABLE => {
            let recv = match reg(i.b()) {
                Some(v) if v.get_table().is_none() => fb::OTHER,
                _ => 0,
            };
            recv | kind(i.c())
        }
        Op::UNM | Op::BNOT | Op::LEN => kind(i.b()),
        Op::CONCAT => kind(i.b()) | kind(i.c()),
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
    bits |= extra;
    if bits != 0 {
        proto.feedback.add_bits(pc as usize, bits);
    }
}

/// The operands of the instruction an exit failed at: the rebuilt frame's
/// registers and its closure's upvalues for an exit at its own pc, a moved
/// exit's blame (`ir::InstData::blame`), which has none past its entries.
#[derive(Clone, Copy)]
enum Operands<'a, 'gc> {
    Frame(LuaFn<'gc>, *mut Value<'gc>),
    Blame(&'a [(u8, Value<'gc>)]),
}

impl<'gc> Operands<'_, 'gc> {
    fn reg(self, r: u8) -> Option<Value<'gc>> {
        match self {
            // SAFETY: the instruction's registers are in the frame's window.
            Operands::Frame(_, base) => Some(unsafe { *base.add(r as usize) }),
            Operands::Blame(b) => b.iter().find(|e| e.0 == r).map(|e| e.1),
        }
    }

    /// The table a `GETTABUP` or `SETTABUP` of upvalue `k` indexes.
    fn upval_table(self, k: u8, by_ref: bool) -> Option<Value<'gc>> {
        match self {
            Operands::Frame(c, _) => {
                let u = unsafe { *c.upvalue_ptr().add(k as usize) };
                Some(if by_ref {
                    unsafe { u.cell }.get()
                } else {
                    unsafe { u.value }
                })
            }
            Operands::Blame(_) => self.named(crate::jit::ir::BLAME_TESTED),
        }
    }

    /// A blame's entry of what no register names (`BLAME_TESTED`,
    /// `BLAME_TABLE`).
    fn named(self, r: u8) -> Option<Value<'gc>> {
        match self {
            Operands::Frame(..) => None,
            Operands::Blame(b) => b.iter().find(|e| e.0 == r).map(|e| e.1),
        }
    }
}

/// A failed entry guard: the kinds of every register the entry guards test
/// into the entry's feedback, the entry instruction's operand kinds, and the
/// count toward a recompile.
fn on_entry_fail<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    base: *mut Value<'gc>,
    region: &Region<'gc>,
) {
    record_site(
        closure.proto,
        region.entry_pc,
        Operands::Frame(closure, base),
        0,
    );
    if let Some(state) = closure.proto.jit.get() {
        let mut st = state.borrow_mut(ctx.mutation());
        let me = region as *const Region<'gc>;
        if let Some(e) = st
            .entries
            .iter_mut()
            .find(|e| e.region.is_some_and(|x| Gc::as_ptr(x) == me))
        {
            for &r in &region.entry_regs {
                // SAFETY: an entry guard tests a live register of the frame.
                let kind = crate::jit::build::value_set(unsafe { *base.add(r as usize) });
                match e.seen.iter_mut().find(|(x, _)| *x == r) {
                    Some((_, s)) => *s |= kind,
                    None => e.seen.push((r, kind)),
                }
            }
        }
    }
    let n = region.entry_fails.get() + 1;
    region.entry_fails.set(n);
    let name = || crate::jit::compile::chunk_name(closure);
    if n == 1 {
        log(ctx, || {
            format!("first entry fail {} pc{}", name(), region.entry_pc)
        });
    }
    if n == region.exit_hot.get() && !region.retired.get() {
        log(ctx, || {
            format!("hot entry fail {} pc{}", name(), region.entry_pc)
        });
        request_recompile(ctx, closure, region, false);
    }
}

/// Point the region's entry at `jit_settle`: a quarter of the hot count
/// after an exit that learned something, so kinds that arrive close
/// together make one compile; the hot count doubled per retry otherwise.
fn request_recompile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    region: &Region<'gc>,
    learned: bool,
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
        let hot = ctx.jit().hot_start(e.original) as u32;
        e.state = EntryState::Settling;
        e.free_recompile |= learned;
        e.settle = if e.free_recompile {
            hot / 4
        } else {
            hot << e.retries
        };
        ctx.jit().set_slot(e.slot, jit_settle as *const u8);
        // Compiled callers come through `enter` to the recompile too.
        if e.pc == 0 {
            set_fast(ctx, &closure.proto, &mut st, 0);
        }
    }
}

/// One execution of a settling entry: the original word to run while it
/// settles, else the recompile's.
fn settle<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    at: u32,
    slot: u16,
    frame: *const Value<'gc>,
) -> Result<*const u8, Instruction> {
    let state = closure
        .proto
        .jit
        .get()
        .expect("a settling entry without a JIT state");
    {
        let mut st = state.borrow_mut(ctx.mutation());
        let e = st
            .entries
            .iter_mut()
            .find(|e| e.slot == slot)
            .expect("a settling entry without its entry");
        if e.settle > 0 {
            e.settle -= 1;
            return Err(e.original);
        }
    }
    recompile(ctx, closure, at, slot, frame)
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
            e.free_recompile || e.retries < MAX_RETRIES,
            e.seen.clone(),
        )
    };
    let jit = ctx.jit();
    let result = if budget_left && state.borrow().compiles < MAX_COMPILES && !jit.exhausted.get() {
        compile_caught(ctx, closure, at, frame, &seen).map_err(|e| {
            log(ctx, || {
                format!(
                    "recompile of {} pc{at} failed: {e}",
                    crate::jit::compile::chunk_name(closure)
                )
            });
            #[cfg(feature = "tcvm-dev")]
            if e.internal() {
                let n = &jit.internal_errors;
                n.set(n.get() + 1);
            }
            if matches!(e, crate::jit::compile::CompileError::OutOfCode) {
                jit.exhausted.set(true);
            }
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
                e.retries += 1;
            }
            e.free_recompile = false;
            region.exit_hot.set(EXIT_HOT << e.retries);
            e.region = Some(region);
            e.state = EntryState::Compiled;
            st.regions.push(region);
            st.compiles += 1;
            ctx.jit().set_slot(slot, region.entry);
            if !region.fast_entry.is_null() {
                set_fast(ctx, &closure.proto, &mut st, region.fast_entry as usize);
            }
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
            retire_entry(ctx, closure, &mut st, &e);
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
