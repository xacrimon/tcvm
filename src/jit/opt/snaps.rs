//! Snapshot re-pruning (`jit-design.md` 9.8): an exit need not restore a
//! register whose slot still holds the value its snapshot names, a load of
//! that slot no path from the load to the exit writes. A value only exits
//! used then dies at its last other use instead of living to the last exit.

use crate::jit::FastMap;
use crate::jit::ir::ops::{Effects, Op};
use crate::jit::ir::{Block, CfgInfo, Func, Inst, NO_SNAP, Val};

/// Blocks as a bit set.
type Blocks = Vec<u64>;

fn has(s: &Blocks, b: Block) -> bool {
    s[b.idx() / 64] >> (b.idx() % 64) & 1 != 0
}

/// The slots one instruction writes: `Some(r)` for one, `None` for any.
fn writes(op: Op) -> Option<Option<u8>> {
    match op {
        Op::Store(r) => Some(Some(r)),
        op if op.effects().has(Effects::W_SLOT) => Some(None),
        _ => None,
    }
}

pub(crate) fn prune_snapshots(f: &mut Func<'_>) {
    let cfg = f.cfg();
    let nb = f.blocks.len();
    let words = nb.div_ceil(64);
    // The slots each block writes, and whether it may write any.
    let mut wrote = vec![[0u64; 4]; nb];
    let mut any = vec![false; nb];
    for &b in &cfg.rpo {
        for &i in f.insts_of(b) {
            match writes(f.op(i)) {
                Some(Some(r)) => wrote[b.idx()][r as usize / 64] |= 1 << (r % 64),
                Some(None) => any[b.idx()] = true,
                None => {}
            }
        }
    }
    let block_writes =
        |b: Block, r: u8| any[b.idx()] || wrote[b.idx()][r as usize / 64] >> (r % 64) & 1 != 0;
    // From a load's block, the blocks a path reaches without entering it
    // again (which would load again), and those reaching an exit's block.
    let mut fwd: FastMap<Block, Blocks> = FastMap::default();
    let mut bwd: FastMap<(Block, Block), Blocks> = FastMap::default();
    let mut done: FastMap<(u32, Vec<bool>), u32> = FastMap::default();
    for bi in 0..nb {
        let b = Block(bi as u32);
        if f.blocks[bi].dead {
            continue;
        }
        for k in 0..f.insts_of(b).len() {
            let e = f.insts_of(b)[k];
            let s = f.insts[e.idx()].snap;
            if s == NO_SNAP {
                continue;
            }
            let entries = f.entries(s).to_vec();
            let keep: Vec<bool> = entries
                .iter()
                .map(|&(r, v)| {
                    let Some((l, lr)) = root_load(f, v) else {
                        return true;
                    };
                    if lr != r {
                        return true;
                    }
                    let lb = f.insts[l.idx()].block;
                    let fw = fwd
                        .entry(lb)
                        .or_insert_with(|| reach(f, &cfg, lb, words, true, lb));
                    let bw = bwd
                        .entry((b, lb))
                        .or_insert_with(|| reach(f, &cfg, b, words, false, lb));
                    written(f, l, e, r, &block_writes, fw, bw)
                })
                .collect();
            if keep.iter().all(|&x| x) {
                continue;
            }
            let ns = *done.entry((s, keep.clone())).or_insert_with(|| {
                let d = f.snaps[s as usize];
                let from = f.snap_pool.len();
                for (&x, &(r, v)) in keep.iter().zip(&entries) {
                    if x {
                        f.snap_pool.push((r, v));
                    }
                }
                f.finish_snap(d.pc, d.kind, from).0
            });
            f.insts[e.idx()].snap = ns;
        }
    }
}

/// The load a snapshot value is, through boxing and refinement.
fn root_load(f: &Func<'_>, mut v: Val) -> Option<(Inst, u8)> {
    loop {
        let i = f.def_inst(v)?;
        match f.op(i) {
            Op::Box | Op::Unbox(_) | Op::Guard(_) | Op::Refine(_) => v = f.args(i)[0],
            Op::Load(r) => return Some((i, r)),
            _ => return None,
        }
    }
}

/// The blocks reachable from `b`'s successors (`forward`) or reaching its
/// predecessors, without passing through `avoid`.
fn reach(
    f: &Func<'_>,
    cfg: &CfgInfo,
    b: Block,
    words: usize,
    forward: bool,
    avoid: Block,
) -> Blocks {
    let mut set = vec![0u64; words];
    let mut work: Vec<Block> = if forward {
        f.succs(b).collect()
    } else {
        cfg.preds(b).to_vec()
    };
    while let Some(x) = work.pop() {
        if x == avoid || has(&set, x) {
            continue;
        }
        set[x.idx() / 64] |= 1 << (x.idx() % 64);
        if forward {
            work.extend(f.succs(x));
        } else {
            work.extend_from_slice(cfg.preds(x));
        }
    }
    set
}

/// Whether some path from load `l` to exit `e` may write slot `r`.
fn written(
    f: &Func<'_>,
    l: Inst,
    e: Inst,
    r: u8,
    block_writes: &impl Fn(Block, u8) -> bool,
    fwd: &Blocks,
    bwd: &Blocks,
) -> bool {
    let (lb, eb) = (f.insts[l.idx()].block, f.insts[e.idx()].block);
    let hits = |insts: &[Inst]| {
        insts
            .iter()
            .any(|&i| writes(f.op(i)).is_some_and(|w| w.is_none_or(|x| x == r)))
    };
    let li = f.insts_of(lb).iter().position(|&i| i == l).unwrap();
    let ei = f.insts_of(eb).iter().position(|&i| i == e).unwrap();
    if lb == eb {
        // `l` dominates `e`; a path leaving the block loads again.
        return hits(&f.insts_of(lb)[li + 1..ei]);
    }
    if hits(&f.insts_of(lb)[li + 1..]) || hits(&f.insts_of(eb)[..ei]) {
        return true;
    }
    (0..f.blocks.len()).any(|x| {
        let x = Block(x as u32);
        has(fwd, x) && has(bwd, x) && x != eb && block_writes(x, r)
    }) || (has(fwd, eb) && has(bwd, eb) && block_writes(eb, r))
}
