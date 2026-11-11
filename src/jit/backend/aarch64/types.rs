//! Tests of a `Value` word's type: a set of kinds as runs of the word
//! order, each one compare.

use crate::env::value::{BOX, FALSE, NIL, SMALL_INT as SMALL_BITS, TRUE, TYPE_WORDS};
use crate::jit::backend::aarch64::abi::{X16, X17};
use crate::jit::backend::aarch64::asm::{Asm, Cond, Gpr, Label, Sz};
use crate::jit::ir::types::TypeSet;

/// The tag of a boxed type alone, its index in `TYPE_WORDS`.
fn single_tag(set: TypeSet) -> Option<u16> {
    (1..=THR_K)
        .find(|&k| TypeSet::BY_WORDS[k] == set)
        .map(|k| k as u16)
}

pub(crate) fn is_single_test(set: TypeSet) -> bool {
    set == TypeSet::SMALL
        || set == TypeSet::FLOAT
        || set == TypeSet::SMALL | TypeSet::FLOAT
        || set == TypeSet::NIL
        || set == TypeSet::FALSY
        || set == TypeSet::BOOL
        || set == TypeSet::FALSE
        || set == TypeSet::TRUE
        || single_tag(set).is_some()
        || kind_runs(set).len() == 1
}

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

/// The register holding word `k` of a test's words (`part_words`), else
/// `value` made in `scratch`.
fn word(asm: &mut Asm, w: &[Gpr], k: usize, scratch: Gpr, value: u64) -> Gpr {
    match w.get(k) {
        Some(&r) => r,
        None => {
            asm.mov_imm(scratch, value);
            scratch
        }
    }
}

/// The words the test of single-test `set` compares with, in the order
/// `type_flags` reads them.
fn part_words(set: TypeSet) -> Vec<u64> {
    if set == TypeSet::FLOAT || set == TypeSet::SMALL | TypeSet::FLOAT {
        vec![BOX]
    } else if set == TypeSet::NIL || set == TypeSet::FALSY {
        vec![NIL]
    } else if set == TypeSet::BOOL || set == TypeSet::FALSE {
        vec![FALSE]
    } else if set == TypeSet::TRUE {
        vec![TRUE]
    } else if set == TypeSet::SMALL || single_tag(set).is_some() {
        Vec::new()
    } else if let [(a, z)] = kind_runs(set)[..] {
        let (lo, end) = (KIND_ORDER[a].1, run_end(z));
        match (a, z) {
            (0, SMALL_K) => Vec::new(),
            (_, SMALL_K) => vec![lo],
            (0, _) => vec![end],
            (_, z) if z <= THR_K => Vec::new(),
            (a, _) if a > THR_K => vec![lo],
            _ => vec![lo, end],
        }
    } else {
        Vec::new()
    }
}

/// The words a test of `set` compares with, part by part: what the IR
/// gives a type test as constant operands (`opt::words`). None where the
/// test needs more than two, which it then makes itself.
pub(crate) fn test_words(set: TypeSet) -> Vec<u64> {
    let w: Vec<u64> = type_parts(set).into_iter().flat_map(part_words).collect();
    if w.len() > 2 { Vec::new() } else { w }
}

/// Flags for a single-test type set; the condition meaning "in the set".
/// `w` holds its words (`part_words`), or nothing for the test to make them.
pub(crate) fn type_flags(asm: &mut Asm, v: Gpr, set: TypeSet, w: &[Gpr]) -> Cond {
    if set == TypeSet::SMALL {
        asm.lsr_imm(Sz::X, X16, v, 32);
        asm.cmn_imm(Sz::W, X16, 1);
        Cond::Eq
    } else if set == TypeSet::FLOAT {
        let k = word(asm, w, 0, X16, BOX);
        asm.cmp(Sz::X, v, k);
        Cond::Lo
    } else if set == TypeSet::SMALL | TypeSet::FLOAT {
        // Below `BOX` (a float), else the small-int high word.
        let k = word(asm, w, 0, X16, BOX);
        asm.cmp(Sz::X, v, k);
        asm.lsr_imm(Sz::X, X17, v, 32);
        asm.ccmn_imm(Sz::W, X17, 1, 0b0100, Cond::Hs);
        Cond::Eq
    } else if set == TypeSet::NIL {
        let k = word(asm, w, 0, X16, NIL);
        asm.cmp(Sz::X, v, k);
        Cond::Eq
    } else if set == TypeSet::FALSY {
        let k = word(asm, w, 0, X16, NIL);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, 2);
        Cond::Lo
    } else if set == TypeSet::BOOL {
        let k = word(asm, w, 0, X16, FALSE);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, 2);
        Cond::Lo
    } else if set == TypeSet::FALSE || set == TypeSet::TRUE {
        let value = if set == TypeSet::FALSE { FALSE } else { TRUE };
        let k = word(asm, w, 0, X16, value);
        asm.cmp(Sz::X, v, k);
        Cond::Eq
    } else if let Some(tag) = single_tag(set) {
        // The high 16 bits, sign extended, are `tag - 8`.
        asm.asr_imm(Sz::X, X16, v, 48);
        asm.cmn_imm(Sz::X, X16, 8 - tag as u64);
        Cond::Eq
    } else if let [(a, z)] = kind_runs(set)[..] {
        run_flags(asm, v, a, z, w)
    } else {
        panic!("type test of {set:?} needs more than one compare")
    }
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

/// Flags for kinds `a..=z` of `KIND_ORDER`, one range of words.
fn run_flags(asm: &mut Asm, v: Gpr, a: usize, z: usize, w: &[Gpr]) -> Cond {
    let lo = KIND_ORDER[a].1;
    if z == SMALL_K {
        if a == 0 {
            asm.cmp(Sz::X, v, v);
            return Cond::Eq;
        }
        let k = word(asm, w, 0, X16, lo);
        asm.cmp(Sz::X, v, k);
        return Cond::Hs;
    }
    let end = run_end(z);
    if a == 0 {
        let k = word(asm, w, 0, X16, end);
        asm.cmp(Sz::X, v, k);
        Cond::Lo
    } else if z <= THR_K {
        // The high 16 bits are `tag - 8`, and the index the tag.
        asm.asr_imm(Sz::X, X16, v, 48);
        asm.add_imm(Sz::X, X16, X16, 8 - a as u64);
        asm.cmp_imm(Sz::X, X16, (z - a) as u64);
        Cond::Ls
    } else if a > THR_K {
        let k = word(asm, w, 0, X16, lo);
        asm.sub(Sz::X, X16, v, k);
        asm.cmp_imm(Sz::X, X16, (z - a) as u64);
        Cond::Ls
    } else {
        let k = word(asm, w, 0, X16, lo);
        asm.cmp(Sz::X, v, k);
        let e = word(asm, w, 1, X17, end);
        asm.ccmp(Sz::X, v, e, 0b0010, Cond::Hs);
        Cond::Lo
    }
}

/// `set`'s parts with the slices of `w` (`test_words`, or nothing) they read.
fn parts_with_words(set: TypeSet, w: &[Gpr]) -> Vec<(TypeSet, &[Gpr])> {
    let mut at = 0;
    type_parts(set)
        .into_iter()
        .map(|p| {
            let n = if w.is_empty() { 0 } else { part_words(p).len() };
            let s = &w[at..at + n];
            at += n;
            (p, s)
        })
        .collect()
}

/// Branch to `ok` when `v` is in `set`, falling through otherwise.
pub(crate) fn type_branch(asm: &mut Asm, v: Gpr, set: TypeSet, ok: Label, w: &[Gpr]) {
    for (p, pw) in parts_with_words(set, w) {
        let c = type_flags(asm, v, p, pw);
        asm.b_cond(c, ok);
    }
}

/// Go to `fail` unless `v` is in `set`, falling through otherwise.
pub(crate) fn type_fail(asm: &mut Asm, v: Gpr, set: TypeSet, fail: Label, w: &[Gpr]) {
    let parts = parts_with_words(set, w);
    if parts.is_empty() {
        asm.b(fail);
        return;
    }
    let ok = asm.new_label();
    for (k, &(p, pw)) in parts.iter().enumerate() {
        let c = type_flags(asm, v, p, pw);
        if k + 1 < parts.len() {
            asm.b_cond(c, ok);
        } else {
            asm.b_cond(c.invert(), fail);
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
    if is_single_test(set) {
        return vec![set];
    }
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

    /// Every set of kinds, tested by `type_branch` and by `type_fail`, its
    /// words made by the test or given in registers, on every sample,
    /// against membership.
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
            for (fail, in_regs) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut a = Asm::new();
                let other = a.new_label();
                // The words `test_words` names, in x1 and x2, or made by the test.
                let mut w = Vec::new();
                if in_regs {
                    for (j, &word) in test_words(set).iter().enumerate() {
                        a.mov_imm(Gpr(1 + j as u8), word);
                        w.push(Gpr(1 + j as u8));
                    }
                }
                if fail {
                    type_fail(&mut a, Gpr(0), set, other, &w);
                } else {
                    type_branch(&mut a, Gpr(0), set, other, &w);
                }
                a.mov_imm(Gpr(0), fail as u64);
                a.ret();
                a.bind(other);
                a.mov_imm(Gpr(0), !fail as u64);
                a.ret();
                let code = a.finish(block.rx() as usize).expect("assemble");
                block.write(&code);
                let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(block.rx()) };
                for &(w, k) in &samples {
                    let want = set.contains(TypeSet::BY_WORDS[k]);
                    assert_eq!(
                        f(w) != 0,
                        want,
                        "{set:?} ({}, words in registers: {in_regs}) on {w:#x}",
                        if fail { "type_fail" } else { "type_branch" }
                    );
                }
            }
        }
    }
}
