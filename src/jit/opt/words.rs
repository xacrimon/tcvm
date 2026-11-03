//! The words type tests compare with (`BOX`, nil, a range's ends) as
//! constant operands: made once in the preheader of the innermost loop
//! around a test (the allocator makes a word again past a clobber or where
//! registers run short), else once for the tests sharing one in code no
//! clobber interrupts. A test alone in reading a word outside loops makes
//! it in a scratch register, as do tests in cold blocks, entry guards and
//! landings.

use crate::jit::FastMap;
use crate::jit::ir::ops::{ExitTag, Op};
use crate::jit::ir::types::TypeSet;
use crate::jit::ir::{Block, Func, Inst, Val};
use crate::jit::opt::natural_loops;

pub(crate) fn place(f: &mut Func<'_>) {
    let cfg = f.cfg();
    // What clobbers the caller-saved registers: a cold arm's call too, not
    // its helpers, whose trampoline keeps the others.
    let clobbers = |f: &Func<'_>, i: Inst, cold: bool| match f.op(i) {
        Op::Call { .. } | Op::CallMm { .. } | Op::Barrier => true,
        Op::Helper(_) => !cold,
        _ => false,
    };
    // Each loop's blocks, its header's back edges merged, innermost last.
    let mut loops: Vec<(Block, Vec<Block>)> = Vec::new();
    for (h, body) in natural_loops(f) {
        match loops.iter_mut().find(|(x, _)| *x == h) {
            Some((_, b)) => {
                for x in body {
                    if !b.contains(&x) {
                        b.push(x);
                    }
                }
            }
            None => loops.push((h, body)),
        }
    }
    loops.sort_by_key(|(_, b)| std::cmp::Reverse(b.len()));
    // The preheader of the innermost loop around each block, where it has
    // one.
    let mut home: Vec<Option<Block>> = vec![None; f.blocks.len()];
    for (h, body) in &loops {
        let outside: Vec<Block> = (cfg.preds(*h).iter().copied())
            .filter(|p| !body.contains(p))
            .collect();
        let &[p] = &outside[..] else {
            continue;
        };
        if !f
            .terminator(p)
            .is_some_and(|t| matches!(f.op(t), Op::Jump | Op::Br))
        {
            continue;
        }
        for &b in body {
            home[b.idx()] = Some(p);
        }
    }
    // Elsewhere a word is shared within a segment, code no clobber
    // interrupts: a block continues its one hot predecessor's last segment.
    // A word two of a segment's tests read goes before the first test in its
    // first block if one is there, else where the segment starts: either
    // dominates them all.
    // A segment's first block, and the clobber it starts after.
    let mut segs: Vec<(Block, Option<Inst>)> = Vec::new();
    let mut end_seg: Vec<Option<usize>> = vec![None; f.blocks.len()];
    let mut tests: Vec<(Inst, usize, Vec<u64>)> = Vec::new();
    let mut looped: Vec<(Inst, Block, Vec<u64>)> = Vec::new();
    for &b in &cfg.rpo {
        if f.blocks[b.idx()].cold {
            continue;
        }
        let mut seg = match cfg.preds(b) {
            &[p] if !f.blocks[p.idx()].cold && end_seg[p.idx()].is_some() => {
                end_seg[p.idx()].unwrap()
            }
            _ => {
                segs.push((b, None));
                segs.len() - 1
            }
        };
        let mut open = true;
        for &i in f.insts_of(b) {
            if clobbers(f, i, false) {
                segs.push((b, Some(i)));
                seg = segs.len() - 1;
                // Nothing goes after a terminator: what follows starts anew.
                open = !f.op(i).is_terminator();
                continue;
            }
            let set = match f.op(i) {
                Op::IsType(set) => set,
                Op::IsFalsy => TypeSet::FALSY,
                Op::ToF64(_) => TypeSet::FLOAT,
                Op::Guard(set)
                    if f.insts[i.idx()].tag != ExitTag::Entry && !f.landings.contains(&i) =>
                {
                    set
                }
                _ => continue,
            };
            let words = crate::jit::backend::test_words(set);
            if f.args(i).len() != 1 || words.is_empty() {
                continue;
            }
            match home[b.idx()] {
                Some(p) => looped.push((i, p, words)),
                None => tests.push((i, seg, words)),
            }
        }
        end_seg[b.idx()] = open.then_some(seg);
    }
    let mut count: FastMap<(usize, u64), u32> = FastMap::default();
    for (_, seg, words) in &tests {
        for &w in words {
            *count.entry((*seg, w)).or_default() += 1;
        }
    }
    let mut made: FastMap<(Block, usize, u64), Val> = FastMap::default();
    let konst = |f: &mut Func<'_>, w: u64| {
        let k = f.make_inst(Op::KI64(w as i64), &[], None, ExitTag::Type);
        (k, f.result(k))
    };
    for (i, p, words) in looped {
        let mut args = vec![f.args(i)[0]];
        for w in words {
            let v = *made.entry((p, usize::MAX, w)).or_insert_with(|| {
                let (k, v) = konst(f, w);
                f.insert_before_term(p, k);
                v
            });
            args.push(v);
        }
        f.set_args(i, &args);
    }
    for (i, seg, words) in tests {
        if !words.iter().all(|&w| count[&(seg, w)] > 1) {
            continue;
        }
        let (root, after) = segs[seg];
        let mut args = vec![f.args(i)[0]];
        for w in words {
            let v = *made.entry((root, seg, w)).or_insert_with(|| {
                let (k, v) = konst(f, w);
                let insts = f.insts_of(root);
                let pos = if f.insts[i.idx()].block == root {
                    insts.iter().position(|&j| j == i).unwrap()
                } else if let Some(c) = after {
                    insts.iter().position(|&j| j == c).unwrap() + 1
                } else {
                    // Past what must lead a block: parameters, entry guards
                    // and a call's resume.
                    insts
                        .iter()
                        .position(|&j| {
                            !matches!(f.op(j), Op::Param(_) | Op::Resume { .. })
                                && !(matches!(f.op(j), Op::Guard(_))
                                    && f.insts[j.idx()].tag == ExitTag::Entry)
                        })
                        .unwrap_or(insts.len() - 1)
                };
                f.insert(root, pos, k);
                v
            });
            args.push(v);
        }
        f.set_args(i, &args);
    }
}
