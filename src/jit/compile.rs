//! The compile pipeline: bytecode to IR, the passes, lowering, register
//! allocation and emission.

use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::jit::build::cfg::{Cfg, CfgError};
use crate::jit::build::{BuildError, Builder, remove_trivial_params, value_set};
use crate::jit::ir::Func;
use crate::jit::ir::types::TypeSet;
use crate::jit::ir::verify::verify;
use crate::jit::opt;
use crate::jit::opt::speculate::EntryKinds;
use crate::lua::Context;

/// Why a compile failed: always an internal limit, never a semantic decline
/// (6.2); it counts as a strike.
#[derive(Debug)]
pub(crate) enum CompileError {
    Irreducible,
    TooLarge,
    Verify(String),
    Backend(String),
    /// No path does work before a deopt nothing widens (`opt::completes`).
    Useless,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Irreducible => write!(f, "irreducible control flow"),
            CompileError::TooLarge => write!(f, "too large"),
            CompileError::Verify(s) => write!(f, "verifier: {s}"),
            CompileError::Backend(s) => write!(f, "backend: {s}"),
            CompileError::Useless => write!(f, "no path does work before a deopt"),
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

impl From<BuildError> for CompileError {
    fn from(e: BuildError) -> Self {
        match e {
            BuildError::TooLarge => CompileError::TooLarge,
        }
    }
}

pub(crate) struct Options {
    pub(crate) check: bool,
    pub(crate) deopt_all: bool,
}

/// Microseconds spent per stage of one compile.
#[derive(Default, Debug)]
pub(crate) struct Times {
    pub(crate) build: u128,
    pub(crate) passes: u128,
    pub(crate) lower: u128,
    pub(crate) regalloc: u128,
    pub(crate) emit: u128,
}

/// The optimized IR of the region of `closure` entered at `pc`. `frame` is
/// the frame standing at the entry, or null.
pub(crate) fn build_ir<'gc>(
    closure: LuaFn<'gc>,
    pc: u32,
    opts: &Options,
    frame: *const Value<'gc>,
    seen: &[(u8, TypeSet)],
    times: &mut Times,
) -> Result<(Func<'gc>, Cfg), CompileError> {
    let start = std::time::Instant::now();
    let code: Vec<_> = closure
        .proto
        .code
        .iter()
        .map(|i| resolve_original(closure, i))
        .collect();
    let loop_entry = code[pc as usize].op() != crate::instruction::Op::FUNC;
    let cfg = Cfg::build(&closure.proto, code, pc)?;
    let mut b = Builder::new(closure, &cfg, pc, loop_entry);
    b.deopt_all = opts.deopt_all;
    let mut f = b.build()?;
    let check = |f: &Func<'gc>, stage: &str| -> Result<(), CompileError> {
        if opts.check {
            verify(f)
                .map_err(|e| CompileError::Verify(format!("after {stage}: {e}\n{}", f.print())))?;
        }
        Ok(())
    };
    check(&f, "build")?;
    let built = std::time::Instant::now();
    times.build = (built - start).as_micros();
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
    opt::infer::infer(&mut f);
    if opt::speculate::speculate(&mut f, &kinds) {
        check(&f, "speculate")?;
    }
    if opt::peel::peel(&mut f) {
        remove_trivial_params(&mut f);
        check(&f, "peel")?;
    }
    optimize(&mut f);
    opt::prune_gc_checks(&mut f);
    opt::dce(&mut f);
    check(&f, "optimize")?;
    opt::calls::call_boundaries(&mut f).map_err(CompileError::Verify)?;
    check(&f, "call boundaries")?;
    times.passes = built.elapsed().as_micros();
    if !opt::completes(&f) {
        return Err(CompileError::Useless);
    }
    // `cfg` is borrowed by nothing past here.
    Ok((f, cfg))
}

pub(crate) fn optimize(f: &mut Func<'_>) {
    for _ in 0..4 {
        opt::infer::infer(f);
        let mut changed = opt::infer::narrow(f);
        opt::infer::infer(f);
        changed |= opt::simplify::simplify(f);
        remove_trivial_params(f);
        changed |= opt::gvn::gvn(f);
        opt::dce(f);
        if !changed {
            break;
        }
    }
    opt::infer::infer(f);
    opt::infer::legalize(f);
    opt::infer::infer(f);
    opt::dce(f);
}

/// The word a JIT entry replaced, for an instruction that is one.
fn resolve_original(
    closure: LuaFn<'_>,
    i: crate::instruction::Instruction,
) -> crate::instruction::Instruction {
    if !i.is_jit_word() {
        return i;
    }
    let state = closure
        .proto
        .jit
        .get()
        .expect("a JIT word without a JIT state");
    let st = state.borrow();
    st.entries
        .iter()
        .find(|e| e.slot == i.d())
        .map(|e| e.original)
        .expect("a JIT word without its entry")
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
#[cfg(target_arch = "aarch64")]
pub(crate) fn compile<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    pc: u32,
    frame: *const Value<'gc>,
    seen: &[(u8, TypeSet)],
) -> Result<crate::dmm::Gc<'gc, crate::jit::region::Region<'gc>>, CompileError> {
    use crate::jit::backend::aarch64::{abi, emit, lower};
    use crate::jit::region::{ExitInfo, Region, SnapEntry};
    let config = &ctx.jit().config;
    let opts = Options {
        check: config.check,
        deopt_all: config.deopt_all,
    };
    note_upvalues_offset(closure);
    let mut times = Times::default();
    let (mut f, _cfg) = build_ir(closure, pc, &opts, frame, seen, &mut times)?;
    opt::split_critical_edges(&mut f);
    if opts.check {
        verify(&f)
            .map_err(|e| CompileError::Verify(format!("after splitting: {e}\n{}", f.print())))?;
    }
    if config.dump.ir || config.dump.opt {
        eprintln!("-- jit ir {} pc{pc}\n{}", chunk_name(closure), f.print());
    }
    let t = std::time::Instant::now();
    let lowered = lower::lower(&f, closure.code as usize).map_err(CompileError::Backend)?;
    times.lower = t.elapsed().as_micros();
    let t = std::time::Instant::now();
    let env = abi::machine_env();
    let ra = regalloc2::RegallocOptions {
        verbose_log: false,
        validate_ssa: opts.check,
        algorithm: if config.fastalloc {
            regalloc2::Algorithm::Fastalloc
        } else {
            regalloc2::Algorithm::Ion
        },
    };
    let mut rctx = ctx.jit().ra.borrow_mut();
    let out = regalloc2::run_with_ctx(&lowered.vcode, &env, &ra, &mut rctx)
        .map_err(|e| CompileError::Backend(format!("regalloc: {e:?}")))?;
    if opts.check {
        let mut checker = regalloc2::checker::Checker::new(&lowered.vcode, &env);
        checker.prepare(&out);
        checker
            .run()
            .map_err(|e| CompileError::Backend(format!("regalloc checker: {e:?}")))?;
    }
    times.regalloc = t.elapsed().as_micros();
    let t = std::time::Instant::now();
    let helpers = emit::Helpers {
        enter: crate::vm::ops::call::enter as *const () as usize,
        fmod: crate::jit::helpers::jit_fmod as *const () as usize,
        pow: crate::jit::helpers::jit_pow as *const () as usize,
        box_i64: crate::jit::helpers::jit_box_i64 as *const () as usize,
        land: crate::jit::helpers::jit_land as *const () as usize,
    };
    let mut em = emit::emit(&lowered, &out, helpers).map_err(CompileError::Backend)?;
    let size = em.asm.offset() + em.asm.pool_bytes() + 16;
    let alloc = ctx.jit().code_alloc();
    let block = alloc
        .reserve(size)
        .map_err(|e| CompileError::Backend(format!("code memory: {e}")))?;
    em.asm
        .retarget_extern(em.exit_common, block.exit_common() as usize);
    let region_word = em.asm.label_offset(em.region_word);
    let entry = block.rx();
    let words = em
        .asm
        .finish(entry as usize)
        .map_err(|e| CompileError::Backend(format!("assembler: {e:?}")))?;
    block.write(&words);
    if config.dump.asm {
        eprintln!(
            "-- jit asm {} pc{pc} at {entry:p}, {} bytes",
            chunk_name(closure),
            words.len() * 4
        );
        dump_words(entry as usize, &words);
    }
    times.emit = t.elapsed().as_micros();
    if config.time {
        eprintln!(
            "jit time {} pc{pc}: {} IR insts, build {}us, passes {}us, lower {}us, regalloc {}us, emit {}us",
            chunk_name(closure),
            f.blocks
                .iter()
                .filter(|b| !b.dead)
                .map(|b| b.insts.len())
                .sum::<usize>(),
            times.build,
            times.passes,
            times.lower,
            times.regalloc,
            times.emit
        );
    }
    let mut exits = Vec::with_capacity(em.exits.len());
    let mut snaps: Vec<SnapEntry> = Vec::new();
    let mut consts: Vec<u64> = Vec::new();
    for e in &em.exits {
        let start = snaps.len() as u32;
        for s in &e.entries {
            let mut s = *s;
            if let crate::jit::region::Loc::Const(c) = s.loc {
                consts.push(e.consts[c as usize]);
                s.loc = crate::jit::region::Loc::Const(consts.len() as u32 - 1);
            }
            snaps.push(s);
        }
        exits.push(ExitInfo {
            pc: e.pc,
            kind: e.kind,
            tag: e.tag,
            snap_start: start,
            snap_len: e.entries.len() as u32,
            count: std::cell::Cell::new(0),
        });
    }
    let region = crate::dmm::Gc::new(
        ctx.mutation(),
        Region {
            code: block,
            _alloc: alloc,
            entry,
            pool: f.pool.clone().into_boxed_slice(),
            exits: exits.into_boxed_slice(),
            snaps: snaps.into_boxed_slice(),
            consts: consts.into_boxed_slice(),
            proto: closure.proto,
            entry_pc: pc,
            frame_size: em.frame_size,
            num_spills: em.num_spills,
            retired: std::cell::Cell::new(false),
        },
    );
    region
        .code
        .patch_u64(region_word, crate::dmm::Gc::as_ptr(region) as u64);
    Ok(region)
}

pub(crate) fn chunk_name(closure: LuaFn<'_>) -> String {
    format!(
        "{}:{}",
        String::from_utf8_lossy(closure.proto.source.as_bytes()),
        closure.proto.line_defined
    )
}

fn dump_words(base: usize, words: &[u32]) {
    // Through the system assembler: wrap the words in an object and disassemble it.
    let dir = std::env::temp_dir().join(format!("tcvm-jit-dump-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let (src, obj) = (dir.join("code.s"), dir.join("code.o"));
    let mut text = String::from(".text\n");
    for w in words {
        text.push_str(&format!(".inst {w:#010x}\n"));
    }
    let ok = std::fs::write(&src, text).is_ok()
        && std::process::Command::new("clang")
            .args(["-c", "-arch", "arm64", "-o"])
            .arg(&obj)
            .arg(&src)
            .status()
            .is_ok_and(|s| s.success());
    let out = ok
        .then(|| {
            std::process::Command::new("objdump")
                .args(["-d", "--no-show-raw-insn"])
                .arg(&obj)
                .output()
                .ok()
        })
        .flatten();
    match out {
        Some(o) if o.status.success() => {
            eprintln!("(offsets from {base:#x})");
            for line in String::from_utf8_lossy(&o.stdout).lines().skip(6) {
                eprintln!("{line}");
            }
        }
        _ => {
            for (k, w) in words.iter().enumerate() {
                eprintln!("{:#x}: {w:08x}", base + k * 4);
            }
        }
    }
}
