//! The compile pipeline: bytecode to IR, the passes, lowering, register
//! allocation and emission.

use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::jit::build::cfg::{Cfg, CfgError};
use crate::jit::build::{BuildError, Builder, remove_trivial_params, value_set};
use crate::jit::ir::types::TypeSet;
#[cfg(feature = "tcvm-dev")]
use crate::jit::ir::verify::verify;
use crate::jit::ir::{DepOn, Func};
use crate::jit::opt;
use crate::jit::opt::speculate::EntryKinds;
use crate::lua::Context;

/// Why a compile failed: always an internal limit, never a semantic decline; it
/// counts as a strike.
#[derive(Debug)]
pub(crate) enum CompileError {
    Irreducible,
    TooLarge,
    Verify(String),
    Backend(String),
    /// No path does work before a deopt nothing widens (`opt::completes`).
    Useless,
    /// Code memory ran out: the JIT turns off.
    OutOfCode,
}

impl CompileError {
    /// A failure of the compiler rather than a refusal of the code.
    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn internal(&self) -> bool {
        matches!(self, CompileError::Verify(_) | CompileError::Backend(_))
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Irreducible => write!(f, "irreducible control flow"),
            CompileError::TooLarge => write!(f, "too large"),
            CompileError::Verify(s) => write!(f, "verifier: {s}"),
            CompileError::Backend(s) => write!(f, "backend: {s}"),
            CompileError::Useless => write!(f, "no path does work before a deopt"),
            CompileError::OutOfCode => write!(f, "out of code memory"),
        }
    }
}

impl From<CfgError> for CompileError {
    fn from(e: CfgError) -> Self {
        match e {
            CfgError::Irreducible => CompileError::Irreducible,
        }
    }
}

impl From<crate::jit::backend::alloc::ReserveError> for CompileError {
    fn from(e: crate::jit::backend::alloc::ReserveError) -> Self {
        match e {
            crate::jit::backend::alloc::ReserveError::TooLarge => CompileError::TooLarge,
            crate::jit::backend::alloc::ReserveError::Exhausted(_) => CompileError::OutOfCode,
        }
    }
}

impl From<BuildError> for CompileError {
    fn from(e: BuildError) -> Self {
        match e {
            BuildError::TooLarge => CompileError::TooLarge,
        }
    }
}

pub(crate) struct Options {
    /// Verify the IR after each pass group and check the allocation.
    #[cfg(feature = "tcvm-dev")]
    pub(crate) check: bool,
    /// Every exit leaves (the test mode).
    pub(crate) deopt_all: bool,
}

impl Options {
    fn of(config: &crate::jit::state::JitConfig) -> Options {
        #[cfg(feature = "tcvm-dev")]
        return Options {
            check: config.dev.check,
            deopt_all: config.dev.deopt_all,
        };
        #[cfg(not(feature = "tcvm-dev"))]
        {
            let _ = config;
            Options { deopt_all: false }
        }
    }
}

/// A compile's stages, in order.
#[cfg(feature = "tcvm-dev")]
#[derive(Clone, Copy)]
enum Stage {
    Build,
    Passes,
    Lower,
    Regalloc,
    Emit,
}

/// Microseconds spent per stage and pass of one compile (`TCVM_JIT_TIME`);
/// nothing without `tcvm-dev`.
#[cfg(feature = "tcvm-dev")]
pub(crate) struct Times {
    last: std::time::Instant,
    stages: [u128; 5],
    /// Each pass's share of `Passes` in nanoseconds, summed over its runs.
    pass: Vec<(&'static str, u128)>,
}

#[cfg(not(feature = "tcvm-dev"))]
pub(crate) struct Times;

#[cfg(feature = "tcvm-dev")]
impl Times {
    pub(crate) fn new() -> Times {
        Times {
            last: std::time::Instant::now(),
            stages: [0; 5],
            pass: Vec::new(),
        }
    }

    /// Stage `s` ends here.
    fn mark(&mut self, s: Stage) {
        let now = std::time::Instant::now();
        self.stages[s as usize] = (now - self.last).as_micros();
        self.last = now;
    }

    /// Run `f` as the pass `name`, adding its time.
    fn timed<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let r = f();
        let ns = t.elapsed().as_nanos();
        match self.pass.iter_mut().find(|(n, _)| *n == name) {
            Some((_, x)) => *x += ns,
            None => self.pass.push((name, ns)),
        }
        r
    }
}

#[cfg(not(feature = "tcvm-dev"))]
impl Times {
    pub(crate) fn new() -> Times {
        Times
    }

    #[inline(always)]
    fn timed<T>(&mut self, _: &'static str, f: impl FnOnce() -> T) -> T {
        f()
    }
}

/// The IR for an error message, with `tcvm-dev`.
fn ir_text(f: &Func<'_>) -> String {
    #[cfg(feature = "tcvm-dev")]
    return f.print();
    #[cfg(not(feature = "tcvm-dev"))]
    {
        let _ = f;
        String::new()
    }
}

/// The optimized IR of the region of `closure` entered at `pc`. `frame` is
/// the frame standing at the entry, or null.
pub(crate) fn build_ir<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    pc: u32,
    opts: &Options,
    frame: *const Value<'gc>,
    seen: &[(u8, TypeSet)],
    times: &mut Times,
) -> Result<(Func<'gc>, Cfg), CompileError> {
    let mut code: Vec<_> = closure
        .proto
        .code
        .iter()
        .map(|i| resolve_original(&closure.proto, i))
        .collect();
    canonical_back_edges(&closure.proto, &mut code);
    let loop_entry = code[pc as usize].op() != crate::instruction::Op::FUNC;
    let tails = if loop_entry {
        Vec::new()
    } else {
        self_tail_calls(closure, &code)
    };
    let cfg = Cfg::build(&closure.proto, code, pc, tails)?;
    let mut b = Builder::new(ctx, closure, &cfg, pc, loop_entry);
    b.deopt_all = opts.deopt_all;
    if loop_entry && !frame.is_null() {
        // SAFETY: the frame's window holds the prototype's registers.
        b.entry_frame = (0..cfg.nregs).map(|r| unsafe { *frame.add(r) }).collect();
    }
    let mut f = b.build()?;
    // A function region's IR is what inlining the function costs.
    if !loop_entry {
        closure.proto.ir_size.set(f.built() as u32);
    }
    let check = |f: &Func<'gc>, stage: &str| -> Result<(), CompileError> {
        #[cfg(feature = "tcvm-dev")]
        if opts.check {
            verify(f)
                .map_err(|e| CompileError::Verify(format!("after {stage}: {e}\n{}", f.print())))?;
        }
        #[cfg(not(feature = "tcvm-dev"))]
        let _ = (f, stage);
        Ok(())
    };
    check(&f, "build")?;
    #[cfg(feature = "tcvm-dev")]
    if opts.check {
        crate::jit::ir::verify::checkpoints(&f)
            .map_err(|e| CompileError::Verify(format!("after build: {e}\n{}", f.print())))?;
    }
    #[cfg(feature = "tcvm-dev")]
    times.mark(Stage::Build);
    let mut kinds = EntryKinds {
        frame: Vec::new(),
        seen: seen.to_vec(),
    };
    if loop_entry && !frame.is_null() {
        kinds.frame = vec![TypeSet::empty(); cfg.nregs];
        for r in cfg.live_in[pc as usize].iter() {
            if r < cfg.nregs {
                // SAFETY: a live register of the frame holds a value.
                kinds.frame[r] = value_set(unsafe { *frame.add(r) });
            }
        }
    }
    times.timed("infer", || opt::infer::infer(&mut f));
    if times.timed("speculate", || opt::speculate::speculate(&mut f, &kinds)) {
        times.timed("verify", || check(&f, "speculate"))?;
    }
    if times.timed("peel", || opt::peel::peel(&mut f)) {
        times.timed("params", || remove_trivial_params(&mut f));
        times.timed("verify", || check(&f, "peel"))?;
    }
    #[cfg(feature = "tcvm-dev")]
    let settled = opts.check;
    #[cfg(not(feature = "tcvm-dev"))]
    let settled = false;
    optimize(&mut f, times, settled).map_err(CompileError::Verify)?;
    if times.timed("jumps", || opt::jumps::thread(&mut f)) {
        times.timed("verify", || check(&f, "jumps"))?;
    }
    if times.timed("sink", || opt::sink::sink(&mut f)) {
        times.timed("verify", || check(&f, "sink"))?;
    }
    times.timed("gc checks", || opt::prune_gc_checks(&mut f));
    times.timed("dce", || opt::dce(&mut f));
    times.timed("verify", || check(&f, "optimize"))?;
    if times.timed("self calls", || {
        opt::selfcall::trust_results(&mut f, closure.proto)
    }) {
        times.timed("verify", || check(&f, "self calls"))?;
    }
    times.timed("test words", || opt::words::place(&mut f));
    if let Err(e) = times.timed("calls", || opt::calls::call_boundaries(&mut f)) {
        return Err(CompileError::Verify(format!("{e}\n{}", ir_text(&f))));
    }
    times.timed("verify", || check(&f, "call boundaries"))?;
    times.timed("snapshots", || opt::snaps::prune_snapshots(&mut f));
    times.timed("dce", || opt::dce(&mut f));
    times.timed("verify", || check(&f, "snapshot pruning"))?;
    #[cfg(feature = "tcvm-dev")]
    times.mark(Stage::Passes);
    if !opt::completes(&f) {
        return Err(CompileError::Useless);
    }
    // `cfg` is borrowed by nothing past here.
    Ok((f, cfg))
}

/// `check`: make sure of what skipping an inference assumes.
pub(crate) fn optimize(f: &mut Func<'_>, t: &mut Times, check: bool) -> Result<(), String> {
    // Inference again on unchanged IR finds what it found: it starts from
    // the types it left, so nothing but this check says so.
    let settled = |f: &mut Func<'_>| {
        if !check {
            return Ok(());
        }
        let before: Vec<_> = f.vals.iter().map(|v| v.ty).collect();
        opt::infer::infer(f);
        match (0..before.len()).find(|&k| f.vals[k].ty != before[k]) {
            Some(k) => Err(format!("inference changed v{k} on unchanged IR")),
            None => Ok(()),
        }
    };
    for _ in 0..4 {
        t.timed("infer", || opt::infer::infer(f));
        let mut changed = t.timed("narrow", || opt::infer::narrow(f));
        if changed {
            t.timed("infer", || opt::infer::infer(f));
        } else {
            settled(f)?;
        }
        changed |= t.timed("stable", || opt::stable::fold(f));
        changed |= t.timed("simplify", || opt::simplify::simplify(f));
        t.timed("params", || remove_trivial_params(f));
        changed |= t.timed("gvn", || opt::gvn::gvn(f));
        changed |= t.timed("branches", || opt::branch::eliminate(f));
        changed |= t.timed("licm", || opt::licm::licm(f));
        changed |= t.timed("range", || opt::range::loop_bounds(f));
        t.timed("dce", || opt::dce(f));
        if !changed {
            break;
        }
    }
    if let Some(arms) = t.timed("misses", || opt::miss::expand(f)) {
        t.timed("licm", || opt::licm::licm(f));
        t.timed("gvn", || opt::gvn::gvn(f));
        t.timed("misses", || opt::miss::recompute(f, &arms));
    }
    t.timed("one destination", || opt::simplify::fold_one_destination(f));
    t.timed("infer", || opt::infer::infer(f));
    if t.timed("legalize", || opt::infer::legalize(f)) {
        t.timed("infer", || opt::infer::infer(f));
    } else {
        settled(f)?;
    }
    t.timed("dce", || opt::dce(f));
    Ok(())
}

/// The word a JIT entry replaced, for an instruction that is one.
pub(crate) fn resolve_original(
    proto: &crate::env::function::Prototype<'_>,
    i: crate::instruction::Instruction,
) -> crate::instruction::Instruction {
    if !i.is_jit_word() {
        return i;
    }
    let state = proto.jit.get().expect("a JIT word without a JIT state");
    let st = state.borrow();
    st.entries
        .iter()
        .find(|e| e.slot == i.d())
        .map(|e| e.original)
        .expect("a JIT word without its entry")
}

/// The `TAILCALL`s of a function region whose call sites saw only its own
/// prototype's closures, which become jumps to its start, the callee guarded to
/// be the frame's: not for a vararg function, nor one whose closures capture
/// its registers (the frame's open upvalues would have to close first).
pub(crate) fn self_tail_calls(
    closure: LuaFn<'_>,
    code: &[crate::instruction::Instruction],
) -> Vec<u32> {
    use crate::env::feedback::slot_proto;
    let proto = closure.proto;
    if proto.is_vararg
        || crate::jit::build::cfg::captured(&proto)
            .iter()
            .next()
            .is_some()
    {
        return Vec::new();
    }
    let me = crate::env::value::Value::function(closure.function()).to_raw();
    (0..code.len() as u32)
        .filter(|&t| {
            let i = code[t as usize];
            // Up to `top` only after a call setting it.
            let after_call = t > 0 && {
                let p = code[t as usize - 1];
                p.c() == 0
                    && matches!(
                        p.op().unquickened(),
                        crate::instruction::Op::CALL
                            | crate::instruction::Op::CALL_R0
                            | crate::instruction::Op::CALL_R1
                            | crate::instruction::Op::CALLS
                            | crate::instruction::Op::CALLS_R0
                            | crate::instruction::Op::CALLS_R1
                    )
            };
            if i.op() != crate::instruction::Op::TAILCALL
                || (i.b() == 0 && !after_call)
                || proto.feedback.bits(t as usize) & crate::jit::feedback::CONST_MISS != 0
            {
                return false;
            }
            let w = proto.feedback.callee(t as usize).map_or(0, |c| c.get());
            w == me || slot_proto(w).is_some_and(|p| crate::dmm::Gc::ptr_eq(p, proto))
        })
        .collect()
}

/// Point each counted back edge that jumps past a `LOOP` at it, as the CFG
/// takes the `LOOP` for the loop's header; an entry there pointed its own
/// already (`Entry::back_edges`).
pub(crate) fn canonical_back_edges(
    proto: &crate::env::function::Prototype<'_>,
    code: &mut [crate::instruction::Instruction],
) {
    use crate::instruction::Op;
    let pointed: Vec<u32> = proto.jit.get().map_or(Vec::new(), |s| {
        s.borrow()
            .entries
            .iter()
            .flat_map(|e| e.back_edges.iter().copied())
            .collect()
    });
    for pc in 0..code.len() {
        let mut i = code[pc];
        if !matches!(i.op(), Op::JMP_BACK | Op::TFORLOOP) || pointed.contains(&(pc as u32)) {
            continue;
        }
        let t = (pc as i64 + 1 + i.branch_offset() as i64) as usize;
        if code[t - 1].op() == Op::LOOP {
            assert!(i.set_branch_offset(i.branch_offset() - 1));
            code[pc] = i;
        }
    }
}

static UPVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Bytes from a Lua closure's cell to its first upvalue slot: the same for
/// every closure, noted at each compile.
pub(crate) fn upvalues_offset() -> usize {
    UPVALS.load(std::sync::atomic::Ordering::Relaxed)
}

fn note_upvalues_offset(closure: LuaFn<'_>) {
    let o = closure.upvalue_ptr() as usize - closure.as_ptr() as usize;
    UPVALS.store(o, std::sync::atomic::Ordering::Relaxed);
}

/// Compile the region of `closure` entered at `pc`, place its code, and
/// return it ready to install.
pub(crate) fn compile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    pc: u32,
    frame: *const Value<'gc>,
    seen: &[(u8, TypeSet)],
) -> Result<crate::dmm::Gc<'gc, crate::jit::region::Region<'gc>>, CompileError> {
    use crate::jit::backend::emit::{FastEntry, Links};
    use crate::jit::backend::target;
    use crate::jit::region::{ExitInfo, Region, encode_snap};
    let config = &ctx.jit().config;
    let opts = Options::of(config);
    note_upvalues_offset(closure);
    let mut times = Times::new();
    let (f, cfg) = build_ir(ctx, closure, pc, &opts, frame, seen, &mut times)?;
    let mut nest = Vec::new();
    if f.meta.loop_entry {
        let mut l = cfg.loop_of[cfg.block_of[pc as usize] as usize];
        while let Some(i) = l {
            nest.push(cfg.blocks[cfg.loops[i].header as usize].start);
            l = cfg.loops[i].parent;
        }
    }
    #[cfg(feature = "tcvm-dev")]
    if config.dev.dump.ir || config.dev.dump.opt {
        eprintln!("-- jit ir {} pc{pc}\n{}", chunk_name(closure), f.print());
    }
    // A function region of a prototype without varargs is what a compiled
    // caller's `blr` can enter.
    let enterable = pc == 0 && !f.meta.loop_entry && !closure.proto.is_vararg;
    let own = enterable.then(|| closure.proto.ptr.as_ptr() as u64);
    let lowered = crate::jit::backend::lower::lower(
        &f,
        closure.code as usize,
        own,
        closure.proto.max_stack_size as u16,
    )
    .map_err(CompileError::Backend)?;
    #[cfg(feature = "tcvm-dev")]
    times.mark(Stage::Lower);
    let out = &(ctx.jit().ra.borrow_mut())
        .run(&lowered.vcode, &lowered.emit_order, lowered.hot_blocks)
        .map_err(|e| CompileError::Backend(format!("regalloc: {e}")))?;
    #[cfg(feature = "tcvm-dev")]
    if let Some(dir) = &config.dev.ra_capture {
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = format!("{dir}/{}-{n:05}.ra", std::process::id());
        let rec = crate::jit::backend::ra::corpus::write(
            &lowered.vcode,
            ctx.jit().ra.borrow().env(),
            &lowered.emit_order,
            lowered.hot_blocks,
            out,
        );
        std::fs::write(&path, rec)
            .map_err(|e| CompileError::Backend(format!("writing {path}: {e}")))?;
    }
    #[cfg(feature = "tcvm-dev")]
    if config.dev.dump.ra {
        eprintln!(
            "-- jit ra {} pc{pc}\n{}",
            chunk_name(closure),
            ctx.jit().ra.borrow().print(&lowered.vcode)
        );
    }
    #[cfg(feature = "tcvm-dev")]
    if config.dev.dump.vcode {
        eprintln!(
            "-- jit vcode {} pc{pc}, {} spill slots\n{}",
            chunk_name(closure),
            out.num_spillslots,
            lowered.vcode.print(out, &lowered.emit_order)
        );
    }
    #[cfg(feature = "tcvm-dev")]
    if opts.check {
        let mut checker = crate::jit::backend::ra::checker::Checker::new(&lowered.vcode);
        let (remat, src) = crate::jit::backend::ra::remat_table(&lowered.vcode);
        checker.allow_unmaterialized(remat, src);
        checker.prepare(out);
        checker
            .run()
            .map_err(|e| CompileError::Backend(format!("regalloc checker: {e:?}")))?;
        crate::jit::backend::ra::check_operands(&lowered.vcode, out)
            .map_err(|e| CompileError::Backend(format!("regalloc checker: {e}")))?;
    }
    #[cfg(feature = "tcvm-dev")]
    times.mark(Stage::Regalloc);
    let jit = ctx.jit();
    let mut trampolines = Vec::with_capacity(lowered.trampolines.len());
    for &(helper, kind) in &lowered.trampolines {
        trampolines.push((helper, jit.trampoline(helper, kind)?));
    }
    let links = Links {
        rt: crate::jit::layout::rt_word(ctx) as u64,
        metrics: crate::jit::layout::rt_read(ctx, crate::jit::layout::METRICS) as u64,
        type_mt_cells: std::array::from_fn(|s| {
            crate::jit::layout::rt_read(ctx, crate::jit::layout::type_mt_cell(s as u8)) as u64
        }),
        trampolines,
        direct: f
            .protos
            .iter()
            .map(|&p| (p.ptr.as_ptr() as u64, p.num_params, p.jit_fast.get()))
            .collect(),
    };
    // Allocated now, moved into the region after: the code embeds its address.
    let fired: Box<[std::cell::Cell<u32>]> = if opts.deopt_all {
        (0..lowered.exits.len())
            .map(|_| std::cell::Cell::new(0))
            .collect()
    } else {
        Box::new([])
    };
    let fired_at = opts.deopt_all.then_some(fired.as_ptr() as usize);
    let fast = enterable.then_some(FastEntry {
        max_stack: closure.proto.max_stack_size,
        num_params: closure.proto.num_params,
        proto: closure.proto.ptr.as_ptr() as u64,
    });
    #[cfg(feature = "tcvm-dev")]
    let trace = config.dev.dump.asm;
    #[cfg(not(feature = "tcvm-dev"))]
    let trace = false;
    let mut em = target::emit::emit(&lowered, out, links, fired_at, fast, trace)
        .map_err(CompileError::Backend)?;
    let alloc = ctx.jit().code_alloc();
    let mut block = alloc.reserve(em.asm.max_len())?;
    let start = block.rx();
    let entry = start.wrapping_add(em.entry as usize);
    let fast_entry = em
        .fast
        .map_or(std::ptr::null(), |at| start.wrapping_add(at as usize));
    let code = em
        .asm
        .finish(start as usize)
        .map_err(|e| CompileError::Backend(format!("assembler: {e:?}")))?;
    block.write(&code);
    block.describe_frame(em.frame_size);
    #[cfg(feature = "tcvm-dev")]
    if config.dev.dump.asm {
        eprintln!(
            "-- jit asm {} pc{pc} at {start:p}, {} bytes, entry +{:#x}, cold +{:#x}",
            chunk_name(closure),
            code.len(),
            em.entry,
            em.cold
        );
        let hot: Vec<String> = (em.hot.iter())
            .map(|(b, o)| format!("b{b}+{o:#x}"))
            .collect();
        eprintln!("(hot blocks at {})", hot.join(" "));
        // Each run's origin: its VCode instruction's kind.
        let tags: Vec<(u32, String)> = (em.origins.iter())
            .map(|&(at, o)| {
                let tag = match crate::jit::backend::emit::Origin::name(o) {
                    Some(name) => name.into(),
                    None => {
                        let d = format!("{:?}", lowered.vcode.insts[o as usize]);
                        d.chars().take_while(|c| c.is_alphanumeric()).collect()
                    }
                };
                (at, tag)
            })
            .collect();
        target::code::disassemble(start as usize, &code, &tags);
    }
    #[cfg(feature = "tcvm-dev")]
    times.mark(Stage::Emit);
    #[cfg(feature = "tcvm-dev")]
    if config.dev.time {
        let [build, passes, lower, regalloc, emit] = times.stages;
        eprintln!(
            "jit time {} pc{pc}: {} IR insts, build {build}us, passes {passes}us, lower {lower}us, regalloc {regalloc}us, emit {emit}us; passes: {}",
            chunk_name(closure),
            f.blocks
                .iter()
                .filter(|b| !b.dead)
                .map(|b| b.insts.len())
                .sum::<usize>(),
            times
                .pass
                .iter()
                .map(|(n, ns)| format!("{n} {:.1}", *ns as f64 / 1000.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut exits = Vec::with_capacity(em.exits.len());
    let mut site_protos: Vec<crate::dmm::Gc<'_, crate::env::function::Prototype<'_>>> = Vec::new();
    let mut snaps: Vec<u32> = Vec::new();
    let mut consts: Vec<u64> = Vec::new();
    let mut last: Option<(usize, usize)> = None;
    for e in &mut em.exits {
        for s in e.entries.iter_mut().chain(&mut e.blame) {
            if let crate::jit::region::Loc::Const(c) = s.loc {
                let w = e.consts[c as usize];
                let k = consts.iter().position(|&x| x == w).unwrap_or_else(|| {
                    consts.push(w);
                    consts.len() - 1
                });
                s.loc = crate::jit::region::Loc::Const(k as u32);
            }
        }
        let at = snaps.len();
        encode_snap(&mut snaps, e.kind, e.pc, &e.frames, &e.entries);
        // Consecutive exits often share their snapshot.
        let at = match last {
            Some((s, n)) if snaps[s..s + n] == snaps[at..] => {
                snaps.truncate(at);
                s
            }
            _ => {
                last = Some((at, snaps.len() - at));
                at
            }
        };
        // An exit in an inlined body is a site of its callee's.
        let func_proto = |func: u32| {
            if func & crate::jit::region::DYNAMIC != 0 {
                f.protos[(func & !crate::jit::region::DYNAMIC) as usize]
            } else {
                f.pool[func as usize]
                    .get_function()
                    .and_then(|x| x.as_lua())
                    .expect("an inlined Lua closure")
                    .proto
            }
        };
        let proto = e
            .frames
            .last()
            .map_or(closure.proto, |fr| func_proto(fr.func));
        // The function a site of inlined frame `fr` is in, by its index in
        // `site_protos` unless it is the snapshot's.
        let mut proto_of = |fr: u32| {
            if fr == e.frame {
                return (u16::MAX, proto);
            }
            let p = match fr {
                crate::jit::ir::NO_FRAME => closure.proto,
                fr => func_proto(f.inline_frames[fr as usize].func),
            };
            let k = site_protos
                .iter()
                .position(|&q| crate::dmm::Gc::ptr_eq(q, p))
                .unwrap_or_else(|| {
                    site_protos.push(p);
                    site_protos.len() - 1
                });
            (k as u16, p)
        };
        let (site_proto, seen_proto) = e.site.map_or((u16::MAX, proto), |s| proto_of(s.frame));
        let site = e.site.map(|s| s.pc);
        let blame = match (site, e.blame.is_empty()) {
            (Some(pc), false) => {
                let at = snaps.len() as u32;
                encode_snap(
                    &mut snaps,
                    crate::jit::ir::ExitKind::Before,
                    pc,
                    &[],
                    &e.blame,
                );
                at
            }
            _ => crate::jit::region::NO_BLAME,
        };
        exits.push(ExitInfo::new(
            at as u32,
            e.tag,
            crate::jit::exit_seen(seen_proto, site.unwrap_or(e.pc)),
            site,
            site_proto,
            blame,
        ));
    }
    // The sites, their snapshots after the exits'.
    let mut sites: Vec<crate::jit::region::Site> = Vec::new();
    for site in &mut em.sites {
        for s in &mut site.entries {
            if let crate::jit::region::Loc::Const(c) = s.loc {
                let w = site.consts[c as usize];
                let k = consts.iter().position(|&x| x == w).unwrap_or_else(|| {
                    consts.push(w);
                    consts.len() - 1
                });
                s.loc = crate::jit::region::Loc::Const(k as u32);
            }
        }
        let at = snaps.len() as u32;
        encode_snap(
            &mut snaps,
            crate::jit::ir::ExitKind::After,
            0,
            &site.frames,
            &site.entries,
        );
        sites.extend(
            site.keys
                .iter()
                .map(|&(key, level)| crate::jit::region::Site {
                    key,
                    snap: at,
                    hdr: site.hdr,
                    pc_after: site.pc_after,
                    level,
                    miss: site.miss,
                }),
        );
    }
    sites.sort_unstable_by_key(|s| s.key);
    let region = crate::dmm::Gc::new(
        ctx.mutation(),
        Region {
            code: block,
            _alloc: alloc,
            entry,
            fast_entry,
            pool: f.pool.clone().into_boxed_slice(),
            shapes: f.shapes.clone().into_boxed_slice(),
            exits: exits.into_boxed_slice(),
            fired,
            snaps: snaps.into_boxed_slice(),
            consts: consts.into_boxed_slice(),
            entry_regs: lowered.entry_regs.clone().into_boxed_slice(),
            entry_fails: std::cell::Cell::new(0),
            proto: closure.proto,
            protos: f.protos.clone().into_boxed_slice(),
            site_protos: site_protos.into_boxed_slice(),
            entry_pc: pc,
            window: lowered.window.unwrap_or(0),
            nest: nest.into_boxed_slice(),
            num_spills: em.num_spills,
            frame_size: em.frame_size,
            retired: std::cell::Cell::new(false),
            exit_hot: std::cell::Cell::new(crate::jit::state::EXIT_HOT),
            sites: sites.into_boxed_slice(),
            lazy: em.lazy.into_boxed_slice(),
            landings: em.landings.into_boxed_slice(),
            invalidated: std::cell::Cell::new(false),
        },
    );
    region.code.patch_u64(
        em.region_word as usize,
        crate::dmm::Gc::as_ptr(region) as u64,
    );
    // The callees repoint the direct calls as their fast entries change.
    for (off, p, stub) in em.direct {
        let proto = *f
            .protos
            .iter()
            .find(|q| q.ptr.as_ptr() as u64 == p)
            .unwrap();
        let state = crate::jit::state::jit_state(ctx.mutation(), proto);
        let weak = crate::dmm::Gc::downgrade(region);
        state
            .borrow_mut(ctx.mutation())
            .callers
            .push((weak, off, stub));
    }
    for &(on, groups) in &f.deps {
        match on {
            DepOn::Class(c) => {
                let bits = crate::env::MetamethodBits::from_bits_retain(groups as u32);
                c.depend(ctx.mutation(), region, bits);
            }
            DepOn::Shape(s) => s.depend(ctx.mutation(), region, groups),
            DepOn::Field(f) => f.depend(ctx.mutation(), region),
        }
    }
    Ok(region)
}

pub(crate) fn chunk_name(closure: LuaFn<'_>) -> String {
    format!(
        "{}:{}",
        String::from_utf8_lossy(closure.proto.source.as_bytes()),
        closure.proto.line_defined
    )
}
