//! Tests of a `Value` word's type: a set of kinds as runs of the word
//! order, each one compare. The words they compare with are pooled
//! literals, read as memory operands: none is asked of the IR
//! (`test_words`).

use crate::env::value::{NIL, SMALL_INT as SMALL_BITS, TYPE_WORDS};
use crate::jit::backend::x86_64::abi::S0;
use crate::jit::backend::x86_64::asm::{Arith, Asm, Cond, Gpr, Label, Mem, Shift, Sz};
use crate::jit::ir::types::TypeSet;

/// The kinds in the order of their words (`TYPE_WORDS`), each with its
/// first word. Nothing has the words between threads and nil or between
/// true and the small integers, so a run of kinds is one range.
const KIND_ORDER: [(TypeSet, u64); 11] = {
    let mut o = [(TypeSet::empty(), 0); 11];
    let mut k = 0;
    while k < 11 {
        o[k] = (TypeSet::BY_WORDS[k], TYPE_WORDS[k].0);
        k += 1;
    }
    o
};
const THR_K: usize = 6;
const TRUE_K: usize = 9;
const SMALL_K: usize = 10;
const _: () = assert!(
    KIND_ORDER[THR_K].0.bits() == TypeSet::THR.bits()
        && KIND_ORDER[TRUE_K].0.bits() == TypeSet::TRUE.bits()
        && KIND_ORDER[SMALL_K].0.bits() == TypeSet::SMALL.bits()
);

/// Whether one compare's flags tell `set`: its kinds are one run.
pub(crate) fn is_single_test(set: TypeSet) -> bool {
    kind_runs(set).len() == 1
}

/// The words a test of `set` wants in registers: none.
pub(crate) fn test_words(_set: TypeSet) -> Vec<u64> {
    Vec::new()
}

/// The first word past a run ending at kind `z`, the unused words after a
/// thread or `true` taken in.
fn run_end(z: usize) -> u64 {
    match z {
        THR_K => NIL,
        TRUE_K => SMALL_BITS,
        SMALL_K => 0,
        _ => KIND_ORDER[z + 1].1,
    }
}

/// Flags for a single-test type set; the condition meaning "in the set".
pub(crate) fn type_flags(asm: &mut Asm, v: Gpr, set: TypeSet) -> Cond {
    let [(a, z)] = kind_runs(set)[..] else {
        panic!("type test of {set:?} needs more than one compare")
    };
    let (lo, end) = (KIND_ORDER[a].1, run_end(z));
    match (a, z) {
        (0, SMALL_K) => {
            asm.cmp(Sz::Q, v, v);
            Cond::E
        }
        (_, SMALL_K) => {
            asm.cmp(Sz::Q, v, Mem::K(lo));
            Cond::Ae
        }
        (0, _) => {
            asm.cmp(Sz::Q, v, Mem::K(end));
            Cond::B
        }
        // One word: nil, false or true.
        _ if a == z && a > THR_K => {
            asm.cmp(Sz::Q, v, Mem::K(lo));
            Cond::E
        }
        // The high 16 bits, sign extended, are `tag - 8`, and the index
        // the tag.
        _ if z <= THR_K => {
            asm.mov(S0, v);
            asm.shift_imm(Shift::Sar, Sz::Q, S0, 48);
            if a == z {
                asm.cmp_imm(Sz::D, S0, a as i32 - 8);
                Cond::E
            } else {
                asm.alu_imm(Arith::Add, Sz::D, S0, 8 - a as i32);
                asm.cmp_imm(Sz::D, S0, (z - a) as i32);
                Cond::Be
            }
        }
        _ => {
            asm.mov(S0, v);
            asm.alu(Arith::Sub, Sz::Q, S0, Mem::K(lo));
            match i32::try_from(end - lo) {
                Ok(n) => asm.cmp_imm(Sz::Q, S0, n),
                Err(_) => asm.cmp(Sz::Q, S0, Mem::K(end - lo)),
            }
            Cond::B
        }
    }
}

/// Branch to `ok` when `v` is in `set`, falling through otherwise.
pub(crate) fn type_branch(asm: &mut Asm, v: Gpr, set: TypeSet, ok: Label) {
    for p in type_parts(set) {
        let c = type_flags(asm, v, p);
        asm.jcc(c, ok);
    }
}

/// Go to `fail` unless `v` is in `set`, falling through otherwise.
pub(crate) fn type_fail(asm: &mut Asm, v: Gpr, set: TypeSet, fail: Label) {
    let parts = type_parts(set);
    if parts.is_empty() {
        asm.jmp(fail);
        return;
    }
    let ok = asm.new_label();
    for (k, &p) in parts.iter().enumerate() {
        let c = type_flags(asm, v, p);
        if k + 1 < parts.len() {
            asm.jcc(c, ok);
        } else {
            asm.jcc(c.invert(), fail);
        }
    }
    asm.bind(ok);
}

/// `set`'s maximal runs of `KIND_ORDER`, first and last index.
fn kind_runs(set: TypeSet) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut k = 0;
    while k < KIND_ORDER.len() {
        if set.contains(KIND_ORDER[k].0) {
            let a = k;
            while k + 1 < KIND_ORDER.len() && set.contains(KIND_ORDER[k + 1].0) {
                k += 1;
            }
            runs.push((a, k));
        }
        k += 1;
    }
    runs
}

/// The single tests `set` takes, in order: its runs, small integers first.
fn type_parts(set: TypeSet) -> Vec<TypeSet> {
    let mut runs = kind_runs(set);
    runs.sort_by_key(|&(_, z)| z != SMALL_K);
    runs.into_iter()
        .map(|(a, z)| (KIND_ORDER[a..=z].iter()).fold(TypeSet::empty(), |s, &(k, _)| s | k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::alloc::CodeAllocator;
    use crate::jit::backend::x86_64::asm::{RAX, RDI};

    /// Words of each type of `TYPE_WORDS`, with its index: both ends, their
    /// neighbours inside, the middle, and floats' special values.
    fn samples() -> Vec<(u64, usize)> {
        let mut out = Vec::new();
        for (k, &(lo, hi)) in TYPE_WORDS.iter().enumerate() {
            let mid = lo + (hi - lo) / 2;
            for w in [lo, lo.saturating_add(1), mid, hi.saturating_sub(1), hi] {
                if (lo..=hi).contains(&w) && !out.contains(&(w, k)) {
                    out.push((w, k));
                }
            }
        }
        for x in [
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            1.5,
        ] {
            out.push((f64::to_bits(x), 0));
        }
        out
    }

    /// Every set of kinds, tested by `type_branch` and by `type_fail`, on
    /// every sample, against membership.
    #[test]
    fn type_tests_match_their_sets() {
        const SIZE: usize = 1 << 15;
        let alloc = CodeAllocator::new();
        let block = alloc.reserve(SIZE).expect("code memory");
        let samples = samples();
        for bits in 0u32..1 << TypeSet::BY_WORDS.len() {
            let set = (0..TypeSet::BY_WORDS.len())
                .filter(|&k| bits >> k & 1 != 0)
                .fold(TypeSet::empty(), |s, k| s | TypeSet::BY_WORDS[k]);
            for fail in [false, true] {
                let mut a = Asm::new();
                let other = a.new_label();
                if fail {
                    type_fail(&mut a, RDI, set, other);
                } else {
                    type_branch(&mut a, RDI, set, other);
                }
                a.mov_imm(RAX, fail as u64);
                a.ret();
                a.bind(other);
                a.mov_imm(RAX, !fail as u64);
                a.ret();
                let code = a.finish(block.rx() as usize).expect("assemble");
                block.write(&code);
                let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(block.rx()) };
                for &(w, k) in &samples {
                    let want = set.contains(TypeSet::BY_WORDS[k]);
                    assert_eq!(
                        f(w) != 0,
                        want,
                        "{set:?} ({}) on {w:#x}",
                        if fail { "type_fail" } else { "type_branch" }
                    );
                }
            }
        }
    }
}
