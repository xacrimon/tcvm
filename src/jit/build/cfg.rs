//! The bytecode side of the builder: blocks, register liveness, dominators,
//! the loop forest, and which blocks have run (`jit-design.md` 6.3, 8.7).

use crate::env::function::{InlineCache, Prototype};
use crate::instruction::{Family, Instruction, Op, TFOR_VARS, UpvalSource};

/// Registers as a bit set (a window has at most 256).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct RegSet(pub(crate) [u64; 4]);

impl RegSet {
    pub(crate) const EMPTY: RegSet = RegSet([0; 4]);

    #[inline]
    pub(crate) fn insert(&mut self, r: usize) {
        self.0[r / 64] |= 1 << (r % 64);
    }

    #[inline]
    pub(crate) fn remove(&mut self, r: usize) {
        self.0[r / 64] &= !(1 << (r % 64));
    }

    pub(crate) fn union(&mut self, o: &RegSet) {
        for i in 0..4 {
            self.0[i] |= o.0[i];
        }
    }

    pub(crate) fn minus(&mut self, o: &RegSet) {
        for i in 0..4 {
            self.0[i] &= !o.0[i];
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..4).flat_map(move |w| {
            let mut bits = self.0[w];
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let t = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(w * 64 + t)
            })
        })
    }
}

/// Registers an instruction reads and writes. `defs` are the registers it
/// writes on every path out of it (kills); `edge_defs` are written on one
/// edge only and kill nothing.
#[derive(Default)]
pub(crate) struct UseDef {
    pub(crate) uses: RegSet,
    pub(crate) defs: RegSet,
}

/// The registers child prototypes capture by reference.
pub(crate) fn captured(proto: &Prototype<'_>) -> RegSet {
    let mut s = RegSet::EMPTY;
    for child in proto.prototypes.iter() {
        for d in child.upvalue_desc.iter() {
            if let UpvalSource::ParentLocal(r) = d.source
                && !d.by_value
            {
                s.insert(r as usize);
            }
        }
    }
    s
}

pub(crate) fn use_def(
    proto: &Prototype<'_>,
    i: Instruction,
    nregs: usize,
    captured: &RegSet,
) -> UseDef {
    let mut ud = UseDef::default();
    let (a, b, c) = (i.a() as usize, i.b() as usize, i.c() as usize);
    let mut u = |r: usize| {
        if r < nregs {
            ud.uses.insert(r);
        }
    };
    let mut d = RegSet::EMPTY;
    let mut def = |r: usize| {
        if r < nregs {
            d.insert(r);
        }
    };
    use Op::*;
    let op = i.op();
    match op {
        MOVE => {
            u(b);
            def(a);
        }
        LOAD | LOADI | GETUPVAL | GETUPVAL_REF | NEWTABLE | LFALSESKIP => def(a),
        LOADNIL => {
            for r in a..a + b {
                def(r);
            }
        }
        SETUPVAL | TBC | ERRNNIL | JT | JF | JEQI | JNEQI | JLTI | JNLTI | JLEI | JNLEI | JGTI
        | JNGTI | JGEI | JNGEI | JEQS | JNEQS | JLTI_F | JNLTI_F | JLEI_F | JNLEI_F | JGTI_F
        | JNGTI_F | JGEI_F | JNGEI_F | RETURN1 => u(a),
        JEQ | JNEQ | JLT | JNLT | JLE | JNLE | JLT_II | JNLT_II | JLE_II | JNLE_II | JEQ_II
        | JNEQ_II => {
            u(a);
            u(b);
        }
        // The destination is written on the jump edge only.
        JTSET | JFSET => u(b),
        GETTABLE => {
            u(b);
            u(c);
            def(a);
        }
        SETTABLE => {
            u(a);
            u(b);
            u(c);
        }
        VARARGGET => {
            u(c);
            def(a);
        }
        UNM | BNOT | NOT | LEN => {
            u(b);
            def(a);
        }
        CONCAT => {
            u(b);
            u(c);
            def(a);
        }
        CLOSE => {
            for r in captured.iter().filter(|&r| r >= a) {
                u(r);
            }
        }
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => {
            if matches!(op, CALLS | CALLS_R0 | CALLS_R1) {
                u(i.imm() as u8 as usize);
            } else {
                u(a);
            }
            if b == 0 {
                for r in a + 4..nregs {
                    u(r);
                }
            } else {
                for r in a + 4..a + 4 + (b - 1) {
                    u(r);
                }
            }
            // The header clobbers `a..a+4`; the results land from `a`.
            if c == 0 {
                for r in a..nregs {
                    def(r);
                }
            } else {
                for r in a..(a + 4).max(a + c - 1) {
                    def(r);
                }
            }
        }
        TAILCALL => {
            u(a);
            let hi = if b == 0 { nregs } else { a + 4 + b - 1 };
            for r in a + 4..hi {
                u(r);
            }
        }
        RETURN => {
            let hi = if b == 0 { nregs } else { a + b - 1 };
            for r in a..hi {
                u(r);
            }
        }
        FORPREP => {
            for r in a..a + 3 {
                u(r);
            }
            for r in a..a + 4 {
                def(r);
            }
        }
        FORLOOP | FORLOOP_I | FORLOOP_F => {
            for r in a..a + 3 {
                u(r);
            }
        }
        TFORPREP => {
            for r in a..a + 4 {
                u(r);
            }
            for r in a + 2..a + TFOR_VARS as usize + 1 {
                def(r);
            }
        }
        TFORCALL | TFORCALL_NEXT | TFORCALL_IPAIRS => {
            for r in [a, a + 1, a + 2, a + 3, a + TFOR_VARS as usize] {
                u(r);
            }
            for r in a + TFOR_VARS as usize..a + TFOR_VARS as usize + b {
                def(r);
            }
        }
        TFORLOOP => u(a + TFOR_VARS as usize),
        SETLIST => {
            u(a);
            let hi = if b == 0 { nregs } else { a + 1 + b };
            for r in a + 1..hi {
                u(r);
            }
        }
        CLOSURE => {
            let child = &proto.prototypes[i.d() as usize];
            for desc in child.upvalue_desc.iter() {
                if let UpvalSource::ParentLocal(r) = desc.source {
                    u(r as usize);
                }
            }
            def(a);
        }
        VARARG => {
            if b == 0 {
                for r in a..nregs {
                    def(r);
                }
            } else {
                for r in a..a + b - 1 {
                    def(r);
                }
            }
        }
        VARARGPREP => {
            if proto.needs_vararg_table {
                def(proto.num_params as usize);
            }
        }
        JMP | JMP_BACK | NOP | STOP | RETURN0 | FUNC | LOOP | JIT_ENTRY | JIT_LOOP => {}
        GETTABUP | GETTABUP_INL | GETTABUP_AUX | GETTABUP_ABSENT | GETTABUP_PROTO
        | GETTABUP_REF => def(a),
        SETTABUP | SETTABUP_INL | SETTABUP_AUX | SETTABUP_TRANS | SETTABUP_ABSENT
        | SETTABUP_REF => u(a),
        GETFIELD | GETFIELD_INL | GETFIELD_AUX | GETFIELD_ABSENT | GETFIELD_PROTO => {
            u(b);
            def(a);
        }
        SELF | SELF_INL | SELF_AUX | SELF_ABSENT | SELF_PROTO => {
            u(b);
            def(a);
            def(a + 4);
        }
        SETFIELD | SETFIELD_INL | SETFIELD_AUX | SETFIELD_TRANS | SETFIELD_ABSENT => {
            u(a);
            u(b);
        }
        _ => match op.info().family {
            Family::RegArith | Family::RegBit => {
                u(b);
                u(c);
                def(a);
            }
            Family::ImmArith | Family::ImmBit => {
                u(b);
                def(a);
            }
            _ => unreachable!("use_def of {op:?}"),
        },
    }
    ud.defs = d;
    ud
}

/// Where control goes after the instruction at `pc`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Flow {
    Next,
    /// Ends the region path (returns, tail calls).
    End,
    Jump(u32),
    /// Falls through to `pc + 1` or jumps to the target.
    Branch(u32),
    /// `LFALSESKIP`: continues at `pc + 2`.
    Skip,
}

pub(crate) fn flow(code: &[Instruction], pc: u32) -> Flow {
    let i = code[pc as usize];
    let target = |off: i32| (pc as i64 + 1 + off as i64) as u32;
    use Op::*;
    match i.op() {
        JMP | JMP_BACK => Flow::Jump(target(i.branch_offset())),
        RETURN | RETURN0 | RETURN1 | TAILCALL | STOP => Flow::End,
        LFALSESKIP => Flow::Skip,
        TFORPREP => Flow::Jump(target(i.branch_offset())),
        FORPREP | FORLOOP | FORLOOP_I | FORLOOP_F | TFORLOOP => {
            Flow::Branch(target(i.branch_offset()))
        }
        op if op.branch_sense().is_some()
            || matches!(op.info().family, Family::CmpReg | Family::CmpImm) =>
        {
            Flow::Branch(target(i.branch_offset()))
        }
        _ => Flow::Next,
    }
}

pub(crate) struct BcBlock {
    pub(crate) start: u32,
    /// One past the last instruction.
    pub(crate) end: u32,
    pub(crate) succs: Vec<u32>,
    pub(crate) preds: Vec<u32>,
}

/// A natural loop of the bytecode CFG.
pub(crate) struct Loop {
    pub(crate) header: u32,
    /// Blocks of the loop, the header included.
    pub(crate) body: Vec<bool>,
    pub(crate) parent: Option<usize>,
    pub(crate) depth: u32,
}

pub(crate) struct Cfg {
    /// The code with JIT words replaced by their originals.
    pub(crate) code: Vec<Instruction>,
    pub(crate) nregs: usize,
    pub(crate) blocks: Vec<BcBlock>,
    pub(crate) block_of: Vec<u32>,
    /// Registers live into each instruction.
    pub(crate) live_in: Vec<RegSet>,
    pub(crate) captured: RegSet,
    pub(crate) loops: Vec<Loop>,
    /// The innermost loop of each block.
    pub(crate) loop_of: Vec<Option<usize>>,
    /// Blocks whose recording sites all lack a record.
    pub(crate) never_ran: Vec<bool>,
}

#[derive(Debug)]
pub(crate) enum CfgError {
    Irreducible,
}

/// Whether the site at `pc` left a record of running (6.3), or `None` when
/// it is not a recording site.
fn recorded(proto: &Prototype<'_>, i: Instruction, pc: usize) -> Option<bool> {
    let op = i.op();
    let info = op.info();
    let byte = proto.feedback[pc].get();
    match info.family {
        Family::RegArith | Family::RegBit | Family::ImmArith | Family::ImmBit => {
            let generic = i.generic_op();
            let (misses, locked) = i.adaptive();
            Some(op != generic || misses > 0 || locked || byte != 0)
        }
        Family::Field => Some(true),
        _ => match op {
            Op::GETFIELD | Op::SETFIELD | Op::GETTABUP | Op::SETTABUP | Op::SELF => {
                let ic = proto.ic_table[i.d() as usize].get();
                Some(!matches!(ic, InlineCache::Empty) || byte != 0)
            }
            _ => None,
        },
    }
}

impl Cfg {
    pub(crate) fn build(
        proto: &Prototype<'_>,
        code: Vec<Instruction>,
        entry_pc: u32,
    ) -> Result<Cfg, CfgError> {
        let n = code.len();
        let nregs = proto.max_stack_size as usize;
        let captured = captured(proto);
        // Leaders.
        let mut leader = vec![false; n + 1];
        leader[0] = true;
        leader[entry_pc as usize] = true;
        for pc in 0..n as u32 {
            match flow(&code, pc) {
                Flow::Next => {}
                Flow::End => leader[pc as usize + 1] = true,
                Flow::Jump(t) => {
                    leader[t as usize] = true;
                    leader[pc as usize + 1] = true;
                }
                Flow::Branch(t) => {
                    leader[t as usize] = true;
                    leader[pc as usize + 1] = true;
                }
                Flow::Skip => {
                    leader[(pc as usize + 2).min(n)] = true;
                    leader[pc as usize + 1] = true;
                }
            }
        }
        let mut blocks = Vec::new();
        let mut block_of = vec![0u32; n];
        let mut start = 0;
        for pc in 1..=n {
            if leader[pc] || pc == n {
                let id = blocks.len() as u32;
                for p in start..pc {
                    block_of[p] = id;
                }
                blocks.push(BcBlock {
                    start: start as u32,
                    end: pc as u32,
                    succs: Vec::new(),
                    preds: Vec::new(),
                });
                start = pc;
            }
        }
        for b in 0..blocks.len() {
            let last = blocks[b].end - 1;
            let succs = match flow(&code, last) {
                Flow::Next => vec![last + 1],
                Flow::End => vec![],
                Flow::Jump(t) => vec![t],
                Flow::Branch(t) => vec![last + 1, t],
                Flow::Skip => vec![last + 2],
            };
            let succs: Vec<u32> = succs
                .into_iter()
                .filter(|&pc| (pc as usize) < n)
                .map(|pc| block_of[pc as usize])
                .collect();
            for &s in &succs {
                blocks[s as usize].preds.push(b as u32);
            }
            blocks[b].succs = succs;
        }

        // Liveness over the whole function.
        let nb = blocks.len();
        let uds: Vec<UseDef> = code
            .iter()
            .map(|&i| use_def(proto, i, nregs, &captured))
            .collect();
        let mut live_in_b = vec![RegSet::EMPTY; nb];
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..nb).rev() {
                let mut cur = RegSet::EMPTY;
                for &s in &blocks[b].succs {
                    cur.union(&live_in_b[s as usize]);
                }
                for pc in (blocks[b].start..blocks[b].end).rev() {
                    let ud = &uds[pc as usize];
                    cur.minus(&ud.defs);
                    cur.union(&ud.uses);
                }
                if cur != live_in_b[b] {
                    live_in_b[b] = cur;
                    changed = true;
                }
            }
        }
        let mut live_in = vec![RegSet::EMPTY; n];
        for b in 0..nb {
            let mut cur = RegSet::EMPTY;
            for &s in &blocks[b].succs {
                cur.union(&live_in_b[s as usize]);
            }
            for pc in (blocks[b].start..blocks[b].end).rev() {
                let ud = &uds[pc as usize];
                cur.minus(&ud.defs);
                cur.union(&ud.uses);
                live_in[pc as usize] = cur;
            }
        }

        // Dominators from pc 0.
        let (rpo, order) = rpo_of(&blocks, 0);
        let mut idom = vec![u32::MAX; nb];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new = u32::MAX;
                for &p in &blocks[b as usize].preds {
                    if idom[p as usize] == u32::MAX {
                        continue;
                    }
                    new = if new == u32::MAX {
                        p
                    } else {
                        let (mut x, mut y) = (new, p);
                        while x != y {
                            while order[x as usize] > order[y as usize] {
                                x = idom[x as usize];
                            }
                            while order[y as usize] > order[x as usize] {
                                y = idom[y as usize];
                            }
                        }
                        x
                    };
                }
                if new != u32::MAX && idom[b as usize] != new {
                    idom[b as usize] = new;
                    changed = true;
                }
            }
        }
        let dominates = |a: u32, mut b: u32| -> bool {
            loop {
                if a == b {
                    return true;
                }
                if idom[b as usize] == b || idom[b as usize] == u32::MAX {
                    return false;
                }
                b = idom[b as usize];
            }
        };

        // Natural loops; a retreating edge to a non-dominator is irreducible.
        let mut loops: Vec<Loop> = Vec::new();
        for &b in &rpo {
            for &s in &blocks[b as usize].succs {
                if order[s as usize] <= order[b as usize] {
                    if !dominates(s, b) {
                        return Err(CfgError::Irreducible);
                    }
                    let li = match loops.iter().position(|l| l.header == s) {
                        Some(li) => li,
                        None => {
                            let mut body = vec![false; nb];
                            body[s as usize] = true;
                            loops.push(Loop {
                                header: s,
                                body,
                                parent: None,
                                depth: 0,
                            });
                            loops.len() - 1
                        }
                    };
                    let mut stack = vec![b];
                    while let Some(x) = stack.pop() {
                        if loops[li].body[x as usize] {
                            continue;
                        }
                        loops[li].body[x as usize] = true;
                        for &p in &blocks[x as usize].preds {
                            if order[p as usize] != usize::MAX {
                                stack.push(p);
                            }
                        }
                    }
                }
            }
        }
        // Nesting: the parent is the smallest strictly larger loop holding the header.
        let sizes: Vec<usize> = loops
            .iter()
            .map(|l| l.body.iter().filter(|&&x| x).count())
            .collect();
        for i in 0..loops.len() {
            let mut best: Option<usize> = None;
            for j in 0..loops.len() {
                if i != j
                    && loops[j].body[loops[i].header as usize]
                    && sizes[j] > sizes[i]
                    && best.is_none_or(|k| sizes[j] < sizes[k])
                {
                    best = Some(j);
                }
            }
            loops[i].parent = best;
        }
        for i in 0..loops.len() {
            let mut d = 0;
            let mut p = loops[i].parent;
            while let Some(j) = p {
                d += 1;
                p = loops[j].parent;
            }
            loops[i].depth = d;
        }
        let mut loop_of: Vec<Option<usize>> = vec![None; nb];
        for (b, slot) in loop_of.iter_mut().enumerate() {
            let mut best: Option<usize> = None;
            for (i, l) in loops.iter().enumerate() {
                if l.body[b] && best.is_none_or(|k| sizes[i] < sizes[k]) {
                    best = Some(i);
                }
            }
            *slot = best;
        }

        // What ran (6.3).
        let mut ran = vec![false; nb];
        let mut has_site = vec![false; nb];
        for (b, blk) in blocks.iter().enumerate() {
            for pc in blk.start..blk.end {
                if let Some(r) = recorded(proto, code[pc as usize], pc as usize) {
                    has_site[b] = true;
                    ran[b] |= r;
                }
            }
        }
        ran[block_of[entry_pc as usize] as usize] = true;
        let seeds: Vec<u32> = (0..nb as u32).filter(|&b| ran[b as usize]).collect();
        for b in seeds {
            let mut x = b;
            while idom[x as usize] != u32::MAX {
                ran[x as usize] = true;
                if idom[x as usize] == x {
                    break;
                }
                x = idom[x as usize];
            }
        }
        let never_ran: Vec<bool> = (0..nb).map(|b| !ran[b] && has_site[b]).collect();

        Ok(Cfg {
            code,
            nregs,
            blocks,
            block_of,
            live_in,
            captured,
            loops,
            loop_of,
            never_ran,
        })
    }
}

/// RPO of the blocks reachable from `entry`, and each block's position in it
/// (`usize::MAX` when unreachable).
fn rpo_of(blocks: &[BcBlock], entry: u32) -> (Vec<u32>, Vec<usize>) {
    let n = blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    let mut stack: Vec<(u32, usize)> = vec![(entry, 0)];
    seen[entry as usize] = true;
    while let Some(&mut (b, ref mut i)) = stack.last_mut() {
        let succs = &blocks[b as usize].succs;
        if *i < succs.len() {
            let s = succs[*i];
            *i += 1;
            if !seen[s as usize] {
                seen[s as usize] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    post.reverse();
    let mut order = vec![usize::MAX; n];
    for (i, &b) in post.iter().enumerate() {
        order[b as usize] = i;
    }
    (post, order)
}
