//! The compile pipeline: bytecode to IR, the passes, lowering, register
//! allocation and emission.

use crate::env::function::LuaFn;
use crate::jit::build::cfg::{Cfg, CfgError};
use crate::jit::build::{BuildError, Builder, remove_trivial_params};
use crate::jit::ir::Func;
use crate::jit::ir::verify::verify;
use crate::jit::opt;
use crate::lua::Context;

/// Why a compile failed: always an internal limit, never a semantic decline
/// (6.2); it counts as a strike.
#[derive(Debug)]
pub(crate) enum CompileError {
    Irreducible,
    TooLarge,
    Verify(String),
    Backend(String),
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
            BuildError::Irreducible => CompileError::Irreducible,
            BuildError::TooLarge => CompileError::TooLarge,
        }
    }
}

pub(crate) struct Options {
    pub(crate) check: bool,
    pub(crate) deopt_all: bool,
}

/// The optimized IR of the region of `closure` entered at `pc`.
pub(crate) fn build_ir<'gc>(
    ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    pc: u32,
    opts: &Options,
) -> Result<(Func<'gc>, Cfg), CompileError> {
    let code: Vec<_> = closure
        .proto
        .code
        .iter()
        .map(|i| resolve_original(closure, i))
        .collect();
    let loop_entry = code[pc as usize].op() != crate::instruction::Op::FUNC;
    let cfg = Cfg::build(&closure.proto, code, pc)?;
    let mut b = Builder::new(ctx, closure, &cfg, pc, loop_entry);
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
    optimize(&mut f);
    check(&f, "optimize")?;
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
) -> Result<crate::dmm::Gc<'gc, crate::jit::region::Region<'gc>>, CompileError> {
    use crate::jit::backend::aarch64::{abi, emit, lower};
    use crate::jit::region::{ExitInfo, Region, SnapEntry};
    let config = &ctx.jit().config;
    let opts = Options {
        check: config.check,
        deopt_all: config.deopt_all,
    };
    note_upvalues_offset(closure);
    let (mut f, _cfg) = build_ir(ctx, closure, pc, &opts)?;
    opt::split_critical_edges(&mut f);
    if opts.check {
        verify(&f)
            .map_err(|e| CompileError::Verify(format!("after splitting: {e}\n{}", f.print())))?;
    }
    if config.dump.ir || config.dump.opt {
        eprintln!("-- jit ir {} pc{pc}\n{}", chunk_name(closure), f.print());
    }
    let lowered = lower::lower(&f, closure.code as usize).map_err(CompileError::Backend)?;
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
    let out = regalloc2::run(&lowered.vcode, &env, &ra)
        .map_err(|e| CompileError::Backend(format!("regalloc: {e:?}")))?;
    if opts.check {
        let mut checker = regalloc2::checker::Checker::new(&lowered.vcode, &env);
        checker.prepare(&out);
        checker
            .run()
            .map_err(|e| CompileError::Backend(format!("regalloc checker: {e:?}")))?;
    }
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
            alloc,
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
            entry_fails: std::cell::Cell::new(0),
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
