//! Random functions shaped like selection's (loops and joins with block
//! parameters, helpers with fixed registers, frame-state uses), allocated
//! with few registers and checked by the vendored checker and for operands
//! sharing a register within an instruction, which that checker does not
//! see. An instruction is a record read through a row of a table each
//! function makes for itself: what no row can say, no test asks.

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use super::{Ctx, Env, Output};
use crate::jit::backend::reg::{Block as RBlock, Inst, PReg, PRegSet, ProgPoint, RegClass, VReg};
use crate::jit::backend::vcode::{
    CAbi, Clobbers, Constraint, Flow, List, MInst, Remat, Row, Slot, VCode,
};

const INTS: [usize; 8] = [0, 1, 2, 3, 4, 5, 19, 20];
const FLOATS: [usize; 5] = [0, 1, 2, 3, 8];

fn int(r: usize) -> PReg {
    PReg::new(r, RegClass::Int)
}

fn float(r: usize) -> PReg {
    PReg::new(r, RegClass::Float)
}

/// Arguments in x0.. and d0.., a result in x0 or d0.
fn env() -> Env {
    Env {
        regs: [
            INTS.iter().map(|&r| int(r)).collect(),
            FLOATS.iter().map(|&r| float(r)).collect(),
        ],
        scratch: [int(17), float(31)],
        transport: int(16),
        c: CAbi {
            args: [INTS.map(int), [0, 1, 2, 3, 8, 9, 10, 11].map(float)],
            ret: [int(0), float(0)],
            caller_saved: caller_saved(),
        },
    }
}

fn all_regs() -> PRegSet {
    let mut s = PRegSet::empty();
    INTS.iter().for_each(|&r| s.add(int(r)));
    FLOATS.iter().for_each(|&r| s.add(float(r)));
    s
}

fn caller_saved() -> PRegSet {
    let mut s = PRegSet::empty();
    INTS[..6].iter().for_each(|&r| s.add(int(r)));
    FLOATS[..4].iter().for_each(|&r| s.add(float(r)));
    s
}

/// A register slot's part: defined or used, at the late point or the
/// early one, and where.
type Part = (bool, bool, Constraint);
const USE: Part = (false, false, Constraint::Any);
const DEF: Part = (true, true, Constraint::Any);

/// An instruction before its row has a number: its register slots, its
/// list (`n` definitions first), and the row's other columns.
#[derive(Clone)]
struct I {
    regs: Vec<(VReg, Part)>,
    list: List,
    words: Vec<VReg>,
    n: u8,
    clobbers: Clobbers,
    flow: Flow,
}

impl I {
    fn new(regs: &[(VReg, Part)]) -> I {
        I {
            regs: regs.to_vec(),
            list: List::None,
            words: Vec::new(),
            n: 0,
            clobbers: Clobbers::None,
            flow: Flow::Fall,
        }
    }

    fn with_list(mut self, list: List, n: u8, words: Vec<VReg>) -> I {
        (self.list, self.n, self.words) = (list, n, words);
        self
    }

    fn row(&self) -> Row {
        let mut slots = [Slot::None; 3];
        for (s, &(v, (def, late, constraint))) in slots.iter_mut().zip(&self.regs) {
            *s = Slot::Reg {
                class: v.class(),
                def,
                late,
                constraint,
            };
        }
        if self.list != List::None {
            let k = self.regs.len();
            (slots[k], slots[k + 1]) = (Slot::Start, Slot::Len);
        }
        let row = Row {
            name: "t",
            slots,
            list: self.list,
            clobbers: self.clobbers,
            flow: self.flow,
            remat: false,
            pseudo: false,
        };
        row.check().expect("a row the decoder reads");
        row
    }
}

#[derive(Default)]
struct B {
    insts: Vec<I>,
    params: Vec<VReg>,
    succs: Vec<u32>,
    args: Vec<Vec<VReg>>,
    preds: Vec<u32>,
}

struct Gen {
    rng: Pcg64,
    v: VCode,
    blocks: Vec<B>,
    /// Blocks made inside a cold arm, which the order puts last.
    cold: Vec<bool>,
    in_cold: bool,
}

impl Gen {
    fn new(seed: u64, v: VCode) -> Gen {
        Gen {
            rng: Pcg64::seed_from_u64(seed),
            v,
            blocks: Vec::new(),
            cold: Vec::new(),
            in_cold: false,
        }
    }

    fn block(&mut self) -> u32 {
        self.cold.push(self.in_cold);
        self.blocks.push(B::default());
        self.blocks.len() as u32 - 1
    }

    fn class(&mut self) -> RegClass {
        if self.rng.random_ratio(30, 100) {
            RegClass::Float
        } else {
            RegClass::Int
        }
    }

    fn pick(&mut self, avail: &[VReg]) -> Option<VReg> {
        (!avail.is_empty()).then(|| avail[self.rng.random_range(0..avail.len())])
    }

    /// A value of class `c`, made in `b` when none is available.
    fn of_class(&mut self, b: u32, c: RegClass, avail: &mut Vec<VReg>) -> VReg {
        let have: Vec<VReg> = avail.iter().copied().filter(|v| v.class() == c).collect();
        if let Some(v) = self.pick(&have) {
            return v;
        }
        let v = self.v.new_vreg(c);
        self.blocks[b as usize].insts.push(I::new(&[(v, DEF)]));
        avail.push(v);
        v
    }

    fn edge(&mut self, from: u32, to: u32, args: Vec<VReg>) {
        self.blocks[from as usize].succs.push(to);
        self.blocks[from as usize].args.push(args);
        self.blocks[to as usize].preds.push(from);
    }

    /// Arguments for `to`'s parameters, from `avail` at the end of `from`.
    fn args_for(&mut self, from: u32, to: u32, avail: &mut Vec<VReg>) -> Vec<VReg> {
        let classes: Vec<RegClass> = self.blocks[to as usize]
            .params
            .iter()
            .map(|p| p.class())
            .collect();
        classes
            .into_iter()
            .map(|c| self.of_class(from, c, avail))
            .collect()
    }

    /// A frame state's values.
    fn snapshot(&mut self, avail: &[VReg]) -> Vec<VReg> {
        (0..self.rng.random_range(0..6))
            .filter_map(|_| self.pick(avail))
            .collect()
    }

    fn op(&mut self, b: u32, avail: &mut Vec<VReg>) {
        // A constant, which the allocator may compute again.
        if self.rng.random_ratio(15, 100) {
            let c = self.class();
            let d = self.v.new_vreg(c);
            self.v.remat[d.vreg()] = if c != RegClass::Int {
                Remat::Float(self.rng.random())
            } else if self.rng.random_ratio(25, 100) {
                Remat::Slot(self.rng.random_range(0..16))
            } else {
                Remat::Int(self.rng.random())
            };
            self.blocks[b as usize].insts.push(I::new(&[(d, DEF)]));
            avail.push(d);
            return;
        }
        // A value computed from another, which the allocator may compute
        // again from it.
        if self.rng.random_ratio(10, 100) {
            let ints: Vec<VReg> = avail
                .iter()
                .copied()
                .filter(|v| v.class() == RegClass::Int)
                .collect();
            if let Some(src) = self.pick(&ints) {
                let d = self.v.new_vreg(RegClass::Int);
                self.v.remat[d.vreg()] = Remat::AndImm {
                    src: src.vreg() as u32,
                    imm: self.rng.random(),
                };
                (self.blocks[b as usize].insts).push(I::new(&[(d, DEF), (src, USE)]));
                avail.push(d);
                return;
            }
        }
        let mut defs = Vec::new();
        let i = match self.rng.random_range(0..10) {
            0..=5 => {
                // A list takes two of the three slots.
                let snap = self.rng.random_ratio(30, 100);
                let room = if snap { 1 } else { 3 };
                // Registers the instruction needs at its early and late
                // points, per class, kept within what the classes have.
                let mut need = [[0u32; 2]; 2];
                let fits = |need: &mut [[u32; 2]; 2], c: RegClass, early: bool, late: bool| {
                    let n = &mut need[(c == RegClass::Float) as usize];
                    if n[0] + early as u32 > 4 || n[1] + late as u32 > 4 {
                        return false;
                    }
                    n[0] += early as u32;
                    n[1] += late as u32;
                    true
                };
                let mut regs: Vec<(VReg, Part)> = Vec::new();
                for _ in 0..self.rng.random_range(0..4) {
                    let Some(v) = self.pick(avail) else { break };
                    let late = !self.rng.random_ratio(70, 100);
                    if regs.len() < room && fits(&mut need, v.class(), true, late) {
                        regs.push((v, (false, late, Constraint::Any)));
                    }
                }
                for _ in 0..self.rng.random_range(0..3) {
                    let c = self.class();
                    let late = self.rng.random_ratio(80, 100);
                    if regs.len() == room || !fits(&mut need, c, !late, true) {
                        continue;
                    }
                    let d = self.v.new_vreg(c);
                    regs.push((d, (true, late, Constraint::Any)));
                    defs.push(d);
                }
                // A definition reusing an input (a two-address instruction):
                // the other inputs are then held to the late point too.
                let tied = regs.iter().position(|r| r.1 == DEF);
                let input = regs.iter().position(|r| {
                    r.1 == USE && tied.is_some_and(|d| regs[d].0.class() == r.0.class())
                });
                if let (Some(d), Some(t)) = (tied, input)
                    && need.iter().all(|n| n[0] + n[1] <= 4)
                    && self.rng.random_ratio(40, 100)
                {
                    regs[d].1 = (true, true, Constraint::Tied(t as u8));
                }
                let i = I::new(&regs);
                if snap {
                    i.with_list(List::AnyLoc, 0, self.snapshot(avail))
                } else {
                    i
                }
            }
            6..=7 => {
                // A helper: the result, if any, then the arguments.
                let mut words = Vec::new();
                if self.rng.random_ratio(60, 100) {
                    let c = self.class();
                    let d = self.v.new_vreg(c);
                    words.push(d);
                    defs.push(d);
                }
                let n = words.len() as u8;
                for _ in 0..self.rng.random_range(0..4) {
                    let Some(v) = self.pick(avail) else { break };
                    words.push(v);
                }
                let mut i = I::new(&[]).with_list(List::CArgs, n, words);
                i.clobbers = Clobbers::CallerSaved;
                i
            }
            _ => {
                // An exit's frame state.
                let mut regs = Vec::new();
                if let Some(v) = self.pick(avail)
                    && self.rng.random_ratio(50, 100)
                {
                    regs.push((v, USE));
                }
                I::new(&regs).with_list(List::AnyLoc, 0, self.snapshot(avail))
            }
        };
        self.blocks[b as usize].insts.push(i);
        avail.extend(defs);
    }

    fn region(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        let mut cur = cur;
        for _ in 0..self.rng.random_range(1..=4) {
            match self.rng.random_range(0..10) {
                4..=5 if depth < 3 => cur = self.diamond(cur, depth, avail),
                6..=7 if depth < 3 => cur = self.looop(cur, depth, avail),
                _ => {
                    for _ in 0..self.rng.random_range(1..=4) {
                        self.op(cur, avail);
                    }
                }
            }
        }
        cur
    }

    fn params(&mut self, b: u32) {
        for _ in 0..self.rng.random_range(0..4) {
            let c = self.class();
            let p = self.v.new_vreg(c);
            self.blocks[b as usize].params.push(p);
        }
    }

    fn br(&mut self, b: u32, avail: &[VReg]) {
        let mut i = I::new(&[]);
        if let Some(v) = self.pick(avail) {
            i = I::new(&[(v, USE)]);
        }
        i.flow = Flow::Branch;
        self.blocks[b as usize].insts.push(i);
    }

    fn jump(&mut self, from: u32, to: u32, avail: &mut Vec<VReg>) {
        let args = self.args_for(from, to, avail);
        let mut i = I::new(&[]);
        i.flow = Flow::Jump;
        self.blocks[from as usize].insts.push(i);
        self.edge(from, to, args);
    }

    /// A branch's edge straight into `j`, a join (critical), when `direct`
    /// allows, or through an arm of `depth`; the arguments are made before
    /// the branch.
    fn arm(
        &mut self,
        cur: u32,
        j: u32,
        depth: u32,
        direct: bool,
        avail: &mut Vec<VReg>,
    ) -> Option<Vec<VReg>> {
        if direct && self.rng.random_ratio(30, 100) {
            return Some(self.args_for(cur, j, avail));
        }
        let a = self.block();
        let al = self.region(a, depth + 1, avail);
        self.jump(al, j, avail);
        self.blocks[cur as usize].succs.push(a);
        self.blocks[cur as usize].args.push(Vec::new());
        self.blocks[a as usize].preds.push(cur);
        None
    }

    fn diamond(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        let j = self.block();
        self.params(j);
        let mut at = avail.clone();
        let mut ae = avail.clone();
        let direct_t = self.arm(cur, j, depth, true, &mut at).map(|a| (0, a));
        let was = self.in_cold;
        self.in_cold |= self.rng.random_ratio(25, 100);
        // Both edges to the join would be no branch.
        let direct_e = (self.arm(cur, j, depth, direct_t.is_none(), &mut ae)).map(|a| (1, a));
        self.in_cold = was;
        self.br(cur, avail);
        // The branch's edges in order: a direct one sits where its arm would.
        for (k, args) in direct_t.into_iter().chain(direct_e) {
            self.blocks[cur as usize].succs.insert(k, j);
            self.blocks[cur as usize].args.insert(k, args);
            self.blocks[j as usize].preds.push(cur);
        }
        avail.extend(self.blocks[j as usize].params.clone());
        j
    }

    fn looop(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        let h = self.block();
        self.params(h);
        self.jump(cur, h, avail);
        avail.extend(self.blocks[h as usize].params.clone());
        let latch = self.region(h, depth + 1, avail);
        // The back edge straight from the branch (critical) or via a block.
        let direct = self
            .rng
            .random_ratio(30, 100)
            .then(|| self.args_for(latch, h, avail));
        self.br(latch, avail);
        match direct {
            Some(args) => self.edge(latch, h, args),
            None => {
                let back = self.block();
                self.edge(latch, back, Vec::new());
                self.jump(back, h, avail);
            }
        }
        let x = self.block();
        self.edge(latch, x, Vec::new());
        x
    }

    /// The function, read through the table of the rows its instructions
    /// have.
    fn finish(mut self) -> VCode {
        let mut rows: Vec<Row> = Vec::new();
        let mut ops = Vec::new();
        for i in self.blocks.iter().flat_map(|b| &b.insts) {
            let row = i.row();
            let k = rows.iter().position(|r| *r == row).unwrap_or_else(|| {
                rows.push(row);
                rows.len() - 1
            });
            ops.push(u8::try_from(k).expect("at most 256 rows"));
        }
        self.v.rows = rows.leak();
        let mut ops = ops.into_iter();
        for (bi, b) in self.blocks.iter().enumerate() {
            let start = self.v.insts.len() as u32;
            for i in &b.insts {
                let mut m = MInst {
                    op: ops.next().unwrap(),
                    sz: 0,
                    cc: 0,
                    n: i.n,
                    ops: [0; 3],
                };
                for (o, r) in m.ops.iter_mut().zip(&i.regs) {
                    *o = r.0.vreg() as u32;
                }
                if i.list != List::None {
                    let k = i.regs.len();
                    m.ops[k] = self.v.pool.len() as u32;
                    m.ops[k + 1] = i.words.len() as u32;
                    self.v.pool.extend(i.words.iter().map(|v| v.vreg() as u32));
                }
                self.v.push(m);
            }
            let end = self.v.insts.len() as u32;
            let succs: Vec<RBlock> = b.succs.iter().map(|&s| RBlock::new(s as usize)).collect();
            let args: Vec<VReg> = b.args.iter().flatten().copied().collect();
            let nargs: Vec<u32> = b.args.iter().map(|a| a.len() as u32).collect();
            assert!(end > start, "block {bi} is empty");
            self.v.push_block(
                start,
                end,
                &succs,
                b.preds.iter().map(|&p| RBlock::new(p as usize)),
                b.params.iter().copied(),
                &args,
                &nargs,
                false,
            );
        }
        self.v.entry = RBlock::new(0);
        self.v
    }
}

/// A function, and its blocks in allocation order with how many are hot.
fn generate(seed: u64) -> (VCode, Vec<u32>, usize) {
    let mut g = Gen::new(seed, VCode::new(&[]));
    let entry = g.block();
    let mut avail = Vec::new();
    // The entry's parameters, fixed in x0.. .
    let mut regs = Vec::new();
    for &r in &INTS[..g.rng.random_range(1..=3)] {
        let p = g.v.new_vreg(RegClass::Int);
        regs.push((p, (true, true, Constraint::Fixed(int(r)))));
        avail.push(p);
    }
    g.blocks[0].insts.push(I::new(&regs));
    let end = g.region(entry, 0, &mut avail);
    let mut regs = Vec::new();
    for _ in 0..g.rng.random_range(0..3) {
        if let Some(v) = g.pick(&avail) {
            regs.push((v, USE));
        }
    }
    g.blocks[end as usize].insts.push(I::new(&regs));
    // Odd blocks cold too: loop headers and blocks dominating hot ones
    // laid out after them.
    for b in 1..g.blocks.len() {
        if g.rng.random_ratio(10, 100) {
            g.cold[b] = true;
        }
    }
    let n = g.blocks.len() as u32;
    let mut order: Vec<u32> = (0..n).filter(|&b| !g.cold[b as usize]).collect();
    let hot = order.len();
    order.extend((0..n).filter(|&b| g.cold[b as usize]));
    let mut v = g.finish();
    (v.loop_of, v.loop_parent) = loops(&v);
    (v, order, hot)
}

/// What `order.rs` gives lowering: each block's innermost natural loop
/// (its header) and each header's enclosing loop, from the dominators.
fn loops(f: &VCode) -> (Vec<u32>, Vec<u32>) {
    const NONE: u32 = u32::MAX;
    let nb = f.num_blocks();
    let entry = f.entry_block().index();
    let mut post = Vec::with_capacity(nb);
    let mut seen = vec![false; nb];
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        let succs = f.block_succs(RBlock::new(b));
        if *k < succs.len() {
            let s = succs[*k].index();
            *k += 1;
            if !std::mem::replace(&mut seen[s], true) {
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    let mut po = vec![NONE; nb];
    for (k, &b) in post.iter().enumerate() {
        po[b] = k as u32;
    }
    let mut idom = vec![NONE; nb];
    idom[entry] = entry as u32;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in post.iter().rev() {
            if b == entry {
                continue;
            }
            let mut new = NONE;
            for p in f.block_preds(RBlock::new(b)) {
                let p = p.index();
                if idom[p] == NONE {
                    continue;
                }
                new = if new == NONE {
                    p as u32
                } else {
                    let (mut x, mut y) = (new as usize, p);
                    while x != y {
                        while po[x] < po[y] {
                            x = idom[x] as usize;
                        }
                        while po[y] < po[x] {
                            y = idom[y] as usize;
                        }
                    }
                    x as u32
                };
            }
            if new != idom[b] {
                idom[b] = new;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == entry || idom[b] == NONE {
            return false;
        }
        b = idom[b] as usize;
    };
    let mut loops: Vec<(usize, Vec<bool>)> = Vec::new();
    for (b, &d) in idom.iter().enumerate() {
        if d == NONE {
            continue;
        }
        for s in f.block_succs(RBlock::new(b)) {
            let h = s.index();
            if !dominates(h, b) {
                continue;
            }
            let k = match loops.iter().position(|l| l.0 == h) {
                Some(k) => k,
                None => {
                    let mut body = vec![false; nb];
                    body[h] = true;
                    loops.push((h, body));
                    loops.len() - 1
                }
            };
            let body = &mut loops[k].1;
            let mut work = vec![b];
            while let Some(x) = work.pop() {
                if !std::mem::replace(&mut body[x], true) {
                    work.extend(f.block_preds(RBlock::new(x)).iter().map(|p| p.index()));
                }
            }
        }
    }
    loops.sort_by_key(|l| l.1.iter().filter(|&&x| x).count());
    let mut inner = vec![NONE; nb];
    let mut outer = vec![NONE; nb];
    for (h, body) in &loops {
        for (x, &inside) in body.iter().enumerate() {
            if inside && inner[x] == NONE {
                inner[x] = *h as u32;
            }
        }
    }
    for (k, (h, _)) in loops.iter().enumerate() {
        outer[*h] = (loops[k + 1..].iter())
            .find(|l| l.1[*h])
            .map_or(NONE, |l| l.0 as u32);
    }
    (inner, outer)
}

fn check(v: &VCode, abi: CAbi, out: &Output) -> Result<(), String> {
    let mut checker = super::checker::Checker::new(v, abi);
    let (remat, src) = super::remat_table(v);
    checker.allow_unmaterialized(remat, src);
    checker.prepare(out);
    checker.run().map_err(|e| format!("{e:?}"))?;
    super::check_operands(v, &abi, out)
}

/// 300 functions by default; `TCVM_RA_CASES` runs more (a change to the
/// allocator wants 100k or more, in a release build), from `TCVM_RA_SEED`;
/// `TCVM_RA_RECORD=dir` records each (`corpus`).
#[test]
fn random_functions_allocate_correctly() {
    let env = env();
    let mut ctx = Ctx::new(env.clone());
    let n: u64 = std::env::var("TCVM_RA_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let first: u64 = std::env::var("TCVM_RA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let record = std::env::var("TCVM_RA_RECORD").ok();
    for seed in first..first + n {
        let (v, order, hot) = generate(seed);
        let out = match ctx.run(&v, &order, hot) {
            Ok(out) => out,
            Err(e) => panic!("seed {seed}: {e}"),
        };
        if let Some(dir) = &record {
            let rec = super::corpus::write(&v, &env, &order, hot, &out);
            std::fs::write(format!("{dir}/seed-{seed:07}.ra"), rec).expect("writing a record");
        }
        if let Err(e) = check(&v, env.c, &out) {
            panic!("seed {seed}: {e}\n{}", v.print(&env.c, &out, &order));
        }
    }
}

/// Every allocation recorded in `TCVM_RA_CORPUS` (`corpus`) comes out the
/// same and passes the checker; then all of them are allocated
/// `TCVM_RA_ROUNDS` times (default 0), timed. In a release build:
/// `TCVM_RA_CORPUS=dir cargo test --release --lib recorded -- --ignored --nocapture`.
#[test]
#[ignore]
fn recorded_allocations_stay_the_same() {
    let dir = std::env::var("TCVM_RA_CORPUS").expect("TCVM_RA_CORPUS");
    let rounds: u32 = std::env::var("TCVM_RA_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut paths: Vec<_> = (std::fs::read_dir(&dir).expect("the corpus directory"))
        .map(|e| e.expect("an entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ra"))
        .collect();
    paths.sort();
    let cases: Vec<_> = (paths.iter())
        .map(|p| {
            let d = std::fs::read(p).expect("a record");
            super::corpus::read(&d).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        })
        .collect();
    // One allocator for all of them, as the JIT runs: its buffers carry over.
    let mut ctx = Ctx::new(cases.first().expect("a non-empty corpus").env.clone());
    let mut differ = 0;
    for (p, c) in paths.iter().zip(&cases) {
        assert!(c.env == *ctx.env(), "{}: other registers", p.display());
        let out =
            (ctx.run(&c.f, &c.order, c.hot)).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let same = out.num_spillslots == c.out.num_spillslots
            && out.edits == c.out.edits
            && out.edge_edits == c.out.edge_edits
            && out.allocs == c.out.allocs;
        if !same {
            differ += 1;
            if differ <= 5 {
                eprintln!("{}: allocation differs", p.display());
            }
            if differ == 1 {
                let abi = &c.env.c;
                let (got, want) = (
                    c.f.print(abi, &out, &c.order),
                    c.f.print(abi, &c.out, &c.order),
                );
                eprintln!("-- got\n{got}\n-- recorded\n{want}");
            }
        }
        if let Err(e) = check(&c.f, c.env.c, &out) {
            panic!("{}: {e}", p.display());
        }
    }
    let insts: usize = cases.iter().map(|c| c.f.num_insts()).sum();
    if rounds > 0 {
        let t = std::time::Instant::now();
        for _ in 0..rounds {
            for c in &cases {
                std::hint::black_box(ctx.run(&c.f, &c.order, c.hot).expect("allocated once"));
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / rounds as f64;
        eprintln!(
            "{} allocations, {insts} instructions: {:.3} ms a round, {:.1} ns an instruction",
            cases.len(),
            ns / 1e6,
            ns / insts as f64
        );
    }
    assert_eq!(differ, 0, "{differ} of {} allocations differ", cases.len());
}

/// One block of `insts` and a return of `ret`, allocated with `env` and
/// checked.
fn straight(v: VCode, env: Env, insts: Vec<I>, ret: &[VReg]) -> (VCode, Output) {
    let mut g = Gen::new(1, v);
    let b = g.block() as usize;
    g.blocks[b].insts = insts;
    let ret: Vec<(VReg, Part)> = ret.iter().map(|&r| (r, USE)).collect();
    g.blocks[b].insts.push(I::new(&ret));
    let mut v = g.finish();
    (v.loop_of, v.loop_parent) = loops(&v);
    let out = Ctx::new(env.clone()).run(&v, &[0], 1).expect("allocated");
    if let Err(e) = check(&v, env.c, &out) {
        panic!("{e}\n{}", v.print(&env.c, &out, &[0]));
    }
    (v, out)
}

/// `a`, `b`, then `d = op a, b` reusing `a` as instruction 2, or 3 after
/// one that clobbers the caller-saved registers (`mid`).
fn two_address(mid: bool, konst: bool) -> (VCode, [VReg; 3], Vec<I>) {
    let mut v = VCode::new(&[]);
    let [a, b, d] = [(); 3].map(|_| v.new_vreg(RegClass::Int));
    if konst {
        v.remat[a.vreg()] = Remat::Int(7);
    }
    let mut insts = vec![I::new(&[(a, DEF)]), I::new(&[(b, DEF)])];
    if mid {
        let mut i = I::new(&[]);
        i.clobbers = Clobbers::CallerSaved;
        insts.push(i);
    }
    let tied = (true, true, Constraint::Tied(1));
    insts.push(I::new(&[(d, tied), (a, USE), (b, USE)]));
    (v, [a, b, d], insts)
}

fn edits_before(out: &Output, i: usize) -> Vec<super::Edit> {
    (out.edits.iter())
        .filter(|e| e.0 == ProgPoint::before(Inst::new(i)))
        .map(|e| e.1)
        .collect()
}

#[test]
fn reused_input_ending_there_gives_its_register() {
    let (v, [_, _, d], insts) = two_address(false, false);
    let (v, out) = straight(v, env(), insts, &[d]);
    let al = out.inst_allocs(&v, Inst::new(2));
    assert!(out.edits.is_empty(), "{:?}", out.edits);
    assert!(al[0].is_reg() && al[0] == al[1] && al[2] != al[0]);
    assert_eq!(al[1], out.inst_allocs(&v, Inst::new(0))[0]);
}

#[test]
fn reused_input_live_after_is_copied() {
    let (v, [a, _, d], insts) = two_address(false, false);
    let (v, out) = straight(v, env(), insts, &[d, a]);
    let al = out.inst_allocs(&v, Inst::new(2));
    let from = out.inst_allocs(&v, Inst::new(0))[0];
    assert!(al[0] == al[1] && al[2] != al[0] && from != al[0]);
    assert_eq!(
        edits_before(&out, 2),
        [super::Edit::Move { from, to: al[0] }]
    );
    assert_eq!(out.inst_allocs(&v, Inst::new(3))[1], from);
}

#[test]
fn reused_input_on_the_stack_is_loaded_into_the_definition() {
    // Every register is caller-saved here, so both inputs wait on the stack.
    let mut env = env();
    env.c.caller_saved = all_regs();
    let abi = env.c;
    let (v, [_, b, d], insts) = two_address(true, false);
    let (v, out) = straight(v, env, insts, &[d, b]);
    let al = out.inst_allocs(&v, Inst::new(3));
    assert!(al[0].is_reg() && al[0] == al[1] && al[2].is_reg() && al[2] != al[0]);
    let loads: Vec<_> = (edits_before(&out, 3).into_iter())
        .filter(|e| matches!(e, super::Edit::Move { to, .. } if *to == al[0]))
        .collect();
    assert!(
        matches!(loads[..], [super::Edit::Move { from, .. }] if from.is_stack()),
        "{}",
        v.print(&abi, &out, &[0])
    );
}

#[test]
fn reused_constant_is_made_in_the_definition() {
    let (v, [a, _, d], insts) = two_address(false, true);
    let (v, out) = straight(v, env(), insts, &[d]);
    let al = out.inst_allocs(&v, Inst::new(2));
    assert!(al[0].is_reg() && al[0] == al[1] && al[2] != al[0]);
    assert!(
        matches!(out.edits[..], [(p, super::Edit::Remat { to, vreg, .. })]
            if p == ProgPoint::before(Inst::new(2)) && to == al[0] && vreg == a),
        "{}",
        v.print(&env().c, &out, &[0])
    );
}

#[test]
fn reused_input_twice_is_read_where_it_is() {
    let mut v = VCode::new(&[]);
    let [a, d] = [(); 2].map(|_| v.new_vreg(RegClass::Int));
    let tied = (true, true, Constraint::Tied(1));
    let insts = vec![
        I::new(&[(a, DEF)]),
        I::new(&[(d, tied), (a, USE), (a, USE)]),
    ];
    let (v, out) = straight(v, env(), insts, &[d]);
    let al = out.inst_allocs(&v, Inst::new(1));
    assert!(out.edits.is_empty() && al[0] == al[1] && al[1] == al[2]);
}

#[test]
fn reused_input_is_made_where_the_definition_is_wanted() {
    let (v, [_, _, d], mut insts) = two_address(false, false);
    let fixed = (false, false, Constraint::Fixed(int(INTS[3])));
    insts.push(I::new(&[(d, fixed)]));
    let (v, out) = straight(v, env(), insts, &[]);
    assert!(out.edits.is_empty(), "{}", v.print(&env().c, &out, &[0]));
    let made = out.inst_allocs(&v, Inst::new(0))[0];
    assert_eq!(made.as_reg(), Some(int(INTS[3])));
}

/// A recorded allocation reads back as the function it was made from: it
/// allocates the same again.
#[test]
fn a_record_reads_back() {
    let env = env();
    let mut ctx = Ctx::new(env.clone());
    for seed in 0..20 {
        let (v, order, hot) = generate(seed);
        let out = ctx.run(&v, &order, hot).expect("allocated");
        let rec = super::corpus::write(&v, &env, &order, hot, &out);
        let c = super::corpus::read(&rec).expect("a record");
        assert!(c.env == env && c.order == order && c.hot == hot);
        assert!(c.f.insts == v.insts && c.f.pool == v.pool && c.f.offs == v.offs);
        let again = ctx.run(&c.f, &c.order, c.hot).expect("allocated");
        assert!(
            again.allocs == out.allocs && again.edits == out.edits && c.out.allocs == out.allocs
        );
    }
}
