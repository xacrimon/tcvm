//! A measurement harness, not a correctness test.
//!
//! It compiles one hot function end to end and reports how many of the emitted
//! aarch64 instructions are register-to-register moves — the shuffle the register
//! allocator is supposed to avoid. It exists to put a real number under
//! "the register allocator emits too many moves" and to A/B two allocators on the
//! same input, so a change to `regalloc.rs` can be judged rather than guessed at.
//!
//! Run it with output shown:
//!
//! ```text
//! cargo test -p tcvm --lib jit::backend::asm_dump -- --nocapture
//! ```
//!
//! For reading the code rather than counting it, the `disasm_*` tests print real
//! disassembly. They are `#[ignore]`d because they shell out to the toolchain:
//!
//! ```text
//! cargo test -p tcvm --lib disasm_mix2 -- --ignored --nocapture
//! ```
//!
//! The metric is exact, not a heuristic: the assembler already elides `mov xd,xd`
//! / `fmov dd,dd` (see `aarch64_asm`), so every move word that survives to the
//! output is a genuine shuffle the allocator failed to coalesce. Move-immediate
//! (`movz`/`movk`) is a real materialization and is deliberately *not* counted.

use super::isel::select;
use super::regalloc::allocate;
use super::spillcost;
use super::target::{encode, machine_env};
use crate::Lua;
use crate::jit::frontend::lower::lower;
use crate::jit::ir::ty::{Rep, Ty, TypeSet};

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

/// `mov xd, xn` is `orr xd, xzr, xn`: ORR (shifted register), `Rn == 31`, no
/// shift. Mask out `Rm` (bits 16-20) and `Rd` (bits 0-4) and match the rest.
fn is_int_reg_move(w: u32) -> bool {
    (w & 0xFFE0_FFE0) == 0xAA00_03E0
}

/// `fmov dd, dn` (register). Mask out `Rn` (bits 5-9) and `Rd` (bits 0-4).
fn is_fp_reg_move(w: u32) -> bool {
    (w & 0xFFFF_FC00) == 0x1E60_4000
}

fn is_reg_move(w: u32) -> bool {
    is_int_reg_move(w) || is_fp_reg_move(w)
}

/// The first nested prototype of `test-files/primes.lua` is `is_prime(x)`. Load
/// the file but do not run it — the chunk's proto carries its children — and pull
/// the function out by position.
#[test]
fn dump_is_prime_asm() {
    let source = std::fs::read_to_string("test-files/primes.lua").unwrap();
    let mut lua = Lua::new();
    lua.load_all();

    let words = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some("primes")).expect("compile");
        let closure = chunk.as_lua().expect("chunk is a Lua closure");
        let is_prime = closure.proto.prototypes[0];

        let func = lower(is_prime, 0, vec![INT]).expect("lower is_prime");
        let mut m = select(&func).expect("isel");
        super::target::annotate(&mut m);
        let ra = allocate(&m, &machine_env()).expect("allocate");
        encode(&m, &func.pool, &ra).expect("encode")
    });

    let total = words.len();
    let moves = words.iter().filter(|&&w| is_reg_move(w)).count();
    let pct = if total == 0 {
        0.0
    } else {
        100.0 * moves as f64 / total as f64
    };

    eprintln!("\n=== is_prime ===");
    eprintln!("  {total} instructions, {moves} reg-reg moves ({pct:.1}%)");
    for (i, &w) in words.iter().enumerate() {
        let mark = if is_reg_move(w) { "  <- move" } else { "" };
        eprintln!("  {i:3}  {w:08x}{mark}");
    }
}

// --- disassembly, for reading the output by hand ---------------------------

/// Compile one function and hand back its encoded words plus the spill count.
fn compile_one(file: &str, chunk_name: &str) -> (Vec<u32>, u32) {
    let source = std::fs::read_to_string(format!("test-files/{file}.lua")).unwrap();
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some(chunk_name)).expect("compile");
        let proto = chunk
            .as_lua()
            .expect("chunk is a Lua closure")
            .proto
            .prototypes[0];
        let func = lower(proto, 0, vec![INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        super::target::annotate(&mut m);
        let ra = allocate(&m, &machine_env()).expect("allocate");
        let words = encode(&m, &func.pool, &ra).expect("encode");
        (words, ra.num_spills)
    })
}

/// Compile one function and measure its spill traffic against the loop nesting.
///
/// The static count and the weighted one are both reported because they answer
/// different questions, and the whole point of Braun–Hack is that they can move in
/// opposite directions: hoisting a reload out of a loop leaves the static count
/// alone and divides the weighted one by the trip count. See [`spillcost`].
fn spill_cost_of(file: &str, chunk_name: &str) -> spillcost::SpillCost {
    let source = std::fs::read_to_string(format!("test-files/{file}.lua")).unwrap();
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some(chunk_name)).expect("compile");
        let proto = chunk
            .as_lua()
            .expect("chunk is a Lua closure")
            .proto
            .prototypes[0];
        let func = lower(proto, 0, vec![INT]).expect("lower");
        let mut m = select(&func).expect("isel");
        super::target::annotate(&mut m);
        let ra = allocate(&m, &machine_env()).expect("allocate");
        spillcost::measure(&m, &ra).expect("reducible control flow")
    })
}

/// The spilling yardstick. Run it before and after any change to the allocator's
/// spill policy:
///
/// ```text
/// cargo test -p tcvm --lib jit::backend::asm_dump::spill_traffic -- --nocapture
/// ```
#[test]
fn spill_traffic_report() {
    eprintln!("\n=== spill traffic ===");
    for (file, chunk) in [("primes", "primes"), ("mix", "mix"), ("mix2", "mix2")] {
        let name = if file == "primes" { "is_prime" } else { file };
        eprintln!("  {name:9} {}", spill_cost_of(file, chunk));
    }
}

/// Disassemble `words` by way of the system toolchain.
///
/// Apple's `objdump` will not read a flat binary, so the words go through an
/// assembly file of `.word` directives and a real object file. If the toolchain
/// is not there, say where the assembly was left rather than failing — this is a
/// thing to read, not a thing to pass.
fn disasm(name: &str, words: &[u32], spills: u32) {
    let moves = words.iter().filter(|&&w| is_reg_move(w)).count();
    eprintln!(
        "\n=== {name}: {} instructions, {moves} reg-reg moves, {spills} spill slots ===",
        words.len()
    );

    let dir = std::env::temp_dir();
    let (asm, obj) = (dir.join(format!("{name}.s")), dir.join(format!("{name}.o")));
    let body: String = words.iter().map(|w| format!(".word 0x{w:08x}\n")).collect();
    if std::fs::write(&asm, format!(".text\n_{name}:\n{body}")).is_err() {
        eprintln!("  (could not write {})", asm.display());
        return;
    }

    let assembled = std::process::Command::new("clang")
        .args(["-c", "-arch", "arm64"])
        .arg(&asm)
        .arg("-o")
        .arg(&obj)
        .status();
    let dumped = match assembled {
        Ok(st) if st.success() => std::process::Command::new("objdump")
            .arg("-d")
            .arg(&obj)
            .output(),
        _ => {
            eprintln!("  (no assembler; disassemble {} yourself)", asm.display());
            return;
        }
    };
    let Ok(out) = dumped else {
        eprintln!("  (no objdump; object left at {})", obj.display());
        return;
    };

    // Spill traffic is what these dumps are usually being read for, so mark it:
    // the frame base lives in its own register, which leaves `sp`-relative
    // loads and stores as the spill area.
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let spill = line.contains("[sp") && (line.contains("ldr") || line.contains("str"));
        eprintln!("{line}{}", if spill { "   <- spill" } else { "" });
    }
}

#[test]
#[ignore = "shells out to clang/objdump; for reading by hand"]
fn disasm_is_prime() {
    let (w, s) = compile_one("primes", "primes");
    disasm("is_prime", &w, s);
}

#[test]
#[ignore = "shells out to clang/objdump; for reading by hand"]
fn disasm_mix() {
    let (w, s) = compile_one("mix", "mix");
    disasm("mix", &w, s);
}

#[test]
#[ignore = "shells out to clang/objdump; for reading by hand"]
fn disasm_mix2() {
    let (w, s) = compile_one("mix2", "mix2");
    disasm("mix2", &w, s);
}

/// The next-use analysis, against a function whose spilling is independently
/// known. `mix2` spills on aarch64 and `mix` does not, so the computed peak
/// pressure has to straddle the register pool — if it does not, the analysis is
/// not measuring what it claims to.
#[test]
fn pressure_explains_which_benchmarks_spill() {
    use super::nextuse;
    use super::order;
    use super::regalloc::RegClass;

    let pool = machine_env().order(RegClass::Int).len() as u32;

    let peak = |file: &str, chunk: &str| -> u32 {
        let source = std::fs::read_to_string(format!("test-files/{file}.lua")).unwrap();
        let mut lua = Lua::new();
        lua.load_all();
        lua.enter(|ctx| {
            let c = ctx.load(&source, Some(chunk)).expect("compile");
            let proto = c.as_lua().expect("lua closure").proto.prototypes[0];
            let func = lower(proto, 0, vec![INT]).expect("lower");
            let mut m = select(&func).expect("isel");
            super::target::annotate(&mut m);
            let layout = order::compute(&m).expect("reducible");
            let nu = nextuse::analyze(&m, &layout);
            nu.pressure
                .iter()
                .map(|p| p[RegClass::Int as usize])
                .max()
                .unwrap_or(0)
        })
    };

    let (mix, mix2) = (peak("mix", "mix"), peak("mix2", "mix2"));
    eprintln!("\n=== peak int pressure (pool {pool}) ===\n  mix {mix}\n  mix2 {mix2}");
    assert!(
        mix <= pool,
        "mix allocates with no spills, so its peak pressure ({mix}) must fit the \
         pool ({pool})"
    );
    assert!(
        mix2 > pool,
        "mix2 spills, so its peak pressure ({mix2}) must exceed the pool ({pool})"
    );
}

/// The spiller, on real functions.
///
/// Three properties, each of which caught a real bug while this was being written:
///
/// - **Pressure is within the register file everywhere.** Every claim downstream —
///   above all that assignment can then proceed without spilling at all — rests on
///   this.
/// - **Nothing is reloaded that was never stored.** A reload reads a slot; if no
///   store ever wrote that slot it reads garbage. This is the invariant that
///   exposed dead values occupying `W`, jump arguments being dropped at block
///   exits, and block parameters being reloaded rather than delivered by the edge.
/// - **A function that fits needs no spill code.** `is_prime` and `mix` both peak
///   below the register file, so a correct spiller must leave them completely
///   alone — and does, which is a claim about the whole pipeline agreeing with an
///   allocator that reached the same answer by an entirely different route.
#[test]
fn the_spill_plan_is_within_k_and_coherent() {
    use super::regalloc::{RegClass, RegallocFunc, VReg};
    use super::{nextuse, order, spill, spillcost};

    eprintln!("\n=== spill plan ===");
    for (file, chunk) in [("primes", "primes"), ("mix", "mix"), ("mix2", "mix2")] {
        let source = std::fs::read_to_string(format!("test-files/{file}.lua")).unwrap();
        let mut lua = Lua::new();
        lua.load_all();
        lua.enter(|ctx| {
            let c = ctx.load(&source, Some(chunk)).expect("compile");
            let proto = c.as_lua().expect("lua closure").proto.prototypes[0];
            let func = lower(proto, 0, vec![INT]).expect("lower");
            let mut m = select(&func).expect("isel");
            super::target::annotate(&mut m);

            let env = machine_env();
            let layout = order::compute(&m).expect("reducible");
            let nu = nextuse::analyze(&m, &layout);
            let plan = spill::plan(&m, &layout, &nu, &env);

            for class in RegClass::ALL {
                let k = env.order(class).len();
                for b in 0..m.num_blocks() {
                    for (what, set) in [("entry", &plan.w_entry[b]), ("exit", &plan.w_exit[b])] {
                        let n = set.iter().filter(|&&v| m.class(v) == class).count();
                        assert!(
                            n <= k,
                            "{file} mb{b} {what}: {n} {class:?} values in registers, k = {k}"
                        );
                    }
                }
            }

            let stored: Vec<VReg> = plan
                .spill_before
                .iter()
                .flatten()
                .chain(plan.edge_spill.values().flatten())
                .copied()
                .collect();
            for v in plan
                .reload_before
                .iter()
                .flatten()
                .chain(plan.edge_reload.values().flatten())
            {
                assert!(
                    stored.contains(v) || m.remat(*v).is_some(),
                    "{file}: v{} is reloaded but never stored — the slot it reads \
                     was never written",
                    v.0
                );
            }

            // The plan's own cost, weighted the same way `spillcost` weights the
            // allocator's output, so the two are comparable. Two conventions have to
            // match for that: an edge's code lands in the predecessor, so it is
            // charged at the predecessor's depth; and a rematerialized value is
            // replayed rather than loaded, so like `spillcost` it is counted apart
            // from memory traffic rather than as some of it.
            let mem = |v: &&VReg| m.remat(**v).is_none();
            let ops = plan
                .reload_before
                .iter()
                .chain(&plan.spill_before)
                .chain(plan.edge_reload.values())
                .chain(plan.edge_spill.values())
                .flatten()
                .filter(mem)
                .count();
            let mut weighted = 0u64;
            let mut depth_of_inst = vec![0u32; m.num_insts()];
            for &b in &layout.order {
                for &i in m.block_insts(b) {
                    depth_of_inst[i] = layout.depth[b.0 as usize];
                }
            }
            for (i, (r, sp)) in plan
                .reload_before
                .iter()
                .zip(&plan.spill_before)
                .enumerate()
            {
                let n = r.iter().chain(sp).filter(mem).count();
                weighted += n as u64 * 10u64.pow(depth_of_inst[i]);
            }
            for (&(p, _), vs) in plan.edge_reload.iter().chain(plan.edge_spill.iter()) {
                let n = vs.iter().filter(mem).count();
                weighted += n as u64 * 10u64.pow(layout.depth[p.0 as usize]);
            }

            let now =
                spillcost::measure(&m, &allocate(&m, &env).expect("allocate")).expect("reducible");
            eprintln!(
                "  {file:6} plan {} ops (weighted {weighted})   vs today {} ops \
                 (weighted {})",
                ops,
                now.total(),
                now.weighted,
            );

            if file != "mix2" {
                assert_eq!(
                    plan.total_reloads() + plan.total_spills(),
                    0,
                    "{file} peaks below the register file, so it must need no spill code"
                );
            }
        });
    }
}
