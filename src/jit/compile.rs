//! The compile pipeline: bytecode to the high-level graph, the copies,
//! selection, register allocation and encoding.

use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::jit::build::cfg::{Cfg, CfgError};
use crate::jit::ir::emit::Emitter;
use crate::jit::ir::types::TypeSet;
use crate::lua::Context;

/// A compile that legitimately gives up. A compiler bug is a panic
/// (`jit::bug!`), never one of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bail {
    /// The region uses what the compiler does not handle.
    Unsupported(&'static str),
    TooLarge,
    /// Code memory ran out.
    OutOfCode,
    /// No path does work before an exit nothing widens.
    Useless,
}

/// What tiering does about a bail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BailAction {
    /// Refuse the entry and count a strike against its prototype.
    Strike,
    /// Refuse the entry alone.
    Refuse,
    /// Turn the JIT off.
    Exhaust,
}

impl Bail {
    pub(crate) fn action(self) -> BailAction {
        match self {
            Bail::Unsupported(_) | Bail::TooLarge => BailAction::Strike,
            Bail::Useless => BailAction::Refuse,
            Bail::OutOfCode => BailAction::Exhaust,
        }
    }

    /// The bail's kind, without its reason.
    pub(crate) fn class(self) -> &'static str {
        match self {
            Bail::Unsupported(_) => "unsupported",
            Bail::TooLarge => "too-large",
            Bail::OutOfCode => "out-of-code",
            Bail::Useless => "useless",
        }
    }
}

impl std::fmt::Display for Bail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bail::Unsupported(why) => write!(f, "unsupported: {why}"),
            Bail::TooLarge => write!(f, "too large"),
            Bail::OutOfCode => write!(f, "out of code memory"),
            Bail::Useless => write!(f, "no path does work before a deopt"),
        }
    }
}

impl From<CfgError> for Bail {
    fn from(e: CfgError) -> Self {
        match e {
            CfgError::Irreducible => Bail::Unsupported("irreducible control flow"),
        }
    }
}

impl From<crate::jit::backend::alloc::ReserveError> for Bail {
    fn from(e: crate::jit::backend::alloc::ReserveError) -> Self {
        match e {
            crate::jit::backend::alloc::ReserveError::TooLarge => Bail::TooLarge,
            crate::jit::backend::alloc::ReserveError::Exhausted(_) => Bail::OutOfCode,
        }
    }
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

/// What a compile works in, kept by the runtime across compiles: emptied,
/// not freed.
#[derive(Default)]
pub(crate) struct Scratch {
    /// The builder's graph, and copy 1's.
    pub(crate) high: Emitter,
    pub(crate) low: Emitter,
    pub(crate) analysis: crate::jit::ir::analyse::Analysis,
    pub(crate) copy: crate::jit::ir::copy::Copy,
    pub(crate) select: crate::jit::backend::target::select::Select,
    pub(crate) selected: crate::jit::backend::emit::Selected,
    pub(crate) pool: crate::jit::ir::Pool,
}

/// A graph as a pass left it: dumped, and verified where asked.
fn stage(
    config: &crate::jit::state::JitConfig,
    closure: LuaFn<'_>,
    pc: u32,
    name: &str,
    e: &Emitter,
) {
    #[cfg(feature = "tcvm-dev")]
    {
        if e.bail.is_some() {
            return;
        }
        if config.dev.dump.ir {
            eprintln!(
                "-- jit {name} {} pc{pc}\n{}",
                chunk_name(closure),
                e.g.print()
            );
        }
        if config.dev.check
            && let Err(why) = crate::jit::ir::verify::verify(&e.g, e.lowered)
        {
            crate::jit::bug!("after {name}: {why}\n{}", e.g.print());
        }
    }
    #[cfg(not(feature = "tcvm-dev"))]
    let _ = (config, closure, pc, name, e);
}

/// Compile the region of `closure` entered at `pc`, place its code, and
/// return it ready to install.
pub(crate) fn compile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    pc: u32,
    _frame: *const Value<'gc>,
    _seen: &[(u8, TypeSet)],
) -> Result<crate::dmm::Gc<'gc, crate::jit::region::Region<'gc>>, Bail> {
    use crate::jit::backend::target;
    use crate::jit::region::{ExitInfo, Loc, NO_BLAME, Region, encode_snap};
    let config = &ctx.jit().config;
    note_upvalues_offset(closure);
    #[cfg(feature = "tcvm-dev")]
    let mut clock = (std::time::Instant::now(), [0u128; 6]);
    // The time since the last mark, as stage `k`'s.
    macro_rules! mark {
        ($k:expr) => {
            #[cfg(feature = "tcvm-dev")]
            {
                let now = std::time::Instant::now();
                clock.1[$k] = (now - clock.0).as_nanos();
                clock.0 = now;
            }
        };
    }
    let mut code: Vec<_> = (closure.proto.code.iter())
        .map(|i| resolve_original(&closure.proto, i))
        .collect();
    canonical_back_edges(&closure.proto, &mut code);
    let cfg = Cfg::build(&closure.proto, code, pc, Vec::new())?;
    mark!(5);
    let scratch = &mut *ctx.jit().scratch.borrow_mut();
    let high = &mut scratch.high;
    high.reset(false);
    let pool = &mut scratch.pool;
    pool.words.clear();
    let (shapes, missing) = crate::jit::build::build(high, pool, &cfg, closure.proto, pc);
    #[cfg(feature = "tcvm-dev")]
    for op in missing {
        (ctx.runtime().stats).count(1, || format!("jit unsupported {}", op.name()));
    }
    #[cfg(not(feature = "tcvm-dev"))]
    let _ = missing;
    high.finish();
    stage(config, closure, pc, "build", high);
    if let Some(why) = high.bail {
        return Err(why);
    }
    mark!(0);
    let (analysis, low) = (&mut scratch.analysis, &mut scratch.low);
    analysis.uses(&high.g);
    analysis.types(&high.g);
    low.reset(true);
    scratch.copy.run(&high.g, analysis, low);
    stage(config, closure, pc, "lower", low);
    if let Some(why) = low.bail {
        return Err(why);
    }
    // A region that only leaves does no work: none of its loops comes
    // around and nothing returns.
    let around = |b: &crate::jit::ir::Block| low.g.block(*b).npreds > low.g.block(*b).nfwd;
    let returns = (low.g.insts.iter()).any(|i| i.op == crate::jit::ir::ops::Op::Ret);
    if !low.g.order.iter().any(around) && !returns {
        return Err(Bail::Useless);
    }
    mark!(1);
    let sel = &mut scratch.selected;
    scratch.select.run(&low.g, pool, sel)?;
    mark!(2);
    let out = &(ctx.jit().ra.borrow_mut())
        .run(&sel.vcode, &sel.layout.emit, sel.layout.hot)
        .unwrap_or_else(|e| crate::jit::bug!("regalloc: {e}"));
    #[cfg(feature = "tcvm-dev")]
    {
        if config.dev.dump.ra {
            let ra = ctx.jit().ra.borrow();
            eprintln!(
                "-- jit ra {} pc{pc}\n{}",
                chunk_name(closure),
                ra.print(&sel.vcode)
            );
        }
        if config.dev.dump.vcode {
            eprintln!(
                "-- jit vcode {} pc{pc}, {} spill slots\n{}",
                chunk_name(closure),
                out.num_spillslots,
                sel.vcode.print(out, &sel.layout.emit)
            );
        }
        if config.dev.check {
            let mut checker = crate::jit::backend::ra::checker::Checker::new(&sel.vcode);
            let (remat, src) = crate::jit::backend::ra::remat_table(&sel.vcode);
            checker.allow_unmaterialized(remat, src);
            checker.prepare(out);
            if let Err(e) = checker.run() {
                crate::jit::bug!("regalloc checker: {e:?}");
            }
            if let Err(e) = crate::jit::backend::ra::check_operands(&sel.vcode, out) {
                crate::jit::bug!("regalloc checker: {e}");
            }
        }
    }
    mark!(3);
    let mut em = target::emit::emit(sel, out, crate::jit::layout::rt_word(ctx) as u64)?;
    let alloc = ctx.jit().code_alloc();
    let mut block = alloc.reserve(em.asm.max_len())?;
    let start = block.rx();
    let entry = start.wrapping_add(em.entry as usize);
    let code = (em.asm)
        .finish(start as usize)
        .unwrap_or_else(|e| crate::jit::bug!("assembler: {e:?}"));
    block.write(&code);
    block.describe_frame(em.frame_size);
    mark!(4);
    #[cfg(feature = "tcvm-dev")]
    {
        if config.dev.dump.asm {
            eprintln!(
                "-- jit asm {} pc{pc} at {start:p}, {} bytes, entry +{:#x}",
                chunk_name(closure),
                code.len(),
                em.entry
            );
            crate::jit::backend::code::disassemble(start as usize, &code, &[]);
        }
        if config.dev.time {
            let [build, lower, select, regalloc, emit, cfg] = clock.1;
            eprintln!(
                "jit time {} pc{pc}: {} high insts, {} low insts, {} machine insts, {} bytes; \
                 cfg {cfg}ns, build {build}ns, lower {lower}ns, select {select}ns, regalloc {regalloc}ns, emit {emit}ns",
                chunk_name(closure),
                high.g.insts.len(),
                low.g.insts.len(),
                sel.vcode.insts.len(),
                code.len(),
            );
        }
    }
    // The exits' snapshots, consecutive equal ones shared.
    let mut exits = Vec::with_capacity(em.exits.len());
    let mut snaps: Vec<u32> = Vec::new();
    let mut consts: Vec<u64> = Vec::new();
    let mut last: Option<(usize, usize)> = None;
    for e in &mut em.exits {
        for s in &mut e.entries {
            if let Loc::Const(c) = s.loc {
                let w = e.consts[c as usize];
                let k = consts.iter().position(|&x| x == w).unwrap_or_else(|| {
                    consts.push(w);
                    consts.len() - 1
                });
                s.loc = Loc::Const(k as u32);
            }
        }
        let at = snaps.len();
        encode_snap(&mut snaps, e.kind, e.pc, &[], &e.entries);
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
        let seen = crate::jit::exit_seen(closure.proto, e.pc);
        exits.push(ExitInfo::new(
            at as u32,
            e.tag,
            seen,
            None,
            u16::MAX,
            NO_BLAME,
        ));
    }
    let mut nest = Vec::new();
    if cfg.code[pc as usize].op() != crate::instruction::Op::FUNC {
        let mut l = cfg.loop_of[cfg.block_of[pc as usize] as usize];
        while let Some(i) = l {
            nest.push(cfg.blocks[cfg.loops[i].header as usize].start);
            l = cfg.loops[i].parent;
        }
    }
    let region = crate::dmm::Gc::new(
        ctx.mutation(),
        Region {
            code: block,
            _alloc: alloc,
            entry,
            fast_entry: std::ptr::null(),
            // SAFETY: the words of values the builder read from live objects.
            pool: (pool.words.iter())
                .map(|&w| unsafe { Value::from_raw(w) })
                .collect(),
            shapes: shapes.into_boxed_slice(),
            exits: exits.into_boxed_slice(),
            #[cfg(feature = "tcvm-dev")]
            fired: Box::new([]),
            snaps: snaps.into_boxed_slice(),
            consts: consts.into_boxed_slice(),
            entry_regs: Box::new([]),
            entry_fails: std::cell::Cell::new(0),
            proto: closure.proto,
            protos: Box::new([]),
            site_protos: Box::new([]),
            entry_pc: pc,
            window: 0,
            nest: nest.into_boxed_slice(),
            num_spills: em.num_spills,
            #[cfg(target_arch = "aarch64")]
            frame_size: em.frame_size,
            retired: std::cell::Cell::new(false),
            exit_hot: std::cell::Cell::new(crate::jit::state::EXIT_HOT),
            sites: Box::new([]),
            lazy: Box::new([]),
            landings: Box::new([]),
            invalidated: std::cell::Cell::new(false),
        },
    );
    region.code.patch_u64(
        em.region_word as usize,
        crate::dmm::Gc::as_ptr(region) as u64,
    );
    Ok(region)
}

pub(crate) fn chunk_name(closure: LuaFn<'_>) -> String {
    format!(
        "{}:{}",
        String::from_utf8_lossy(closure.proto.source.as_bytes()),
        closure.proto.line_defined
    )
}
