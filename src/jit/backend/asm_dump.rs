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
//! The metric is exact, not a heuristic: the assembler already elides `mov xd,xd`
//! / `fmov dd,dd` (see `aarch64_asm`), so every move word that survives to the
//! output is a genuine shuffle the allocator failed to coalesce. Move-immediate
//! (`movz`/`movk`) is a real materialization and is deliberately *not* counted.

use super::isel::select;
use super::regalloc::{Allocation, MachineEnv, RegallocError, linear_scan, spill_everything};
use super::target::{encode, machine_env};
use crate::jit::frontend::lower::lower;
use crate::jit::ir::ty::{Rep, Ty, TypeSet};
use crate::Lua;

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

type Allocator = fn(
    &super::mach::MFunc,
    &MachineEnv,
) -> Result<Allocation, RegallocError>;

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
fn dump(name: &str, alloc: Allocator) {
    let source = std::fs::read_to_string("test-files/primes.lua").unwrap();
    let mut lua = Lua::new();
    lua.load_all();

    let words = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some("primes")).expect("compile");
        let closure = chunk.as_lua().expect("chunk is a Lua closure");
        let is_prime = closure.proto.prototypes[0];

        let func = lower(is_prime, 0, vec![INT]).expect("lower is_prime");
        let m = select(&func).expect("isel");
        let ra = alloc(&m, &machine_env()).expect("allocate");
        encode(&m, &func.pool, &ra).expect("encode")
    });

    let total = words.len();
    let moves = words.iter().filter(|&&w| is_reg_move(w)).count();
    let pct = if total == 0 {
        0.0
    } else {
        100.0 * moves as f64 / total as f64
    };

    eprintln!("\n=== is_prime under {name} ===");
    eprintln!("  {total} instructions, {moves} reg-reg moves ({pct:.1}%)");
    for (i, &w) in words.iter().enumerate() {
        let mark = if is_reg_move(w) { "  <- move" } else { "" };
        eprintln!("  {i:3}  {w:08x}{mark}");
    }
}

#[test]
fn dump_is_prime_asm() {
    for (name, alloc) in [
        ("linear_scan", linear_scan as Allocator),
        ("spill_everything", spill_everything as Allocator),
    ] {
        dump(name, alloc);
    }
}
