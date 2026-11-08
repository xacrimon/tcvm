//! Random functions shaped like lowering's (loops and joins with block
//! parameters, calls into resume blocks clobbering everything, helpers with
//! fixed registers, snapshot uses), allocated with few registers and
//! checked by the vendored checker and for operands sharing a register
//! within an instruction, which that checker does not see.

use super::{Ctx, Env, Output};
use crate::jit::backend::reg::{
    Block as RBlock, Inst, Operand, OperandConstraint, OperandKind, OperandPos, PReg, PRegSet,
    ProgPoint, RegClass, VReg,
};
use crate::jit::backend::vcode::{Remat, TargetInst, VCode};

#[derive(Clone, Debug)]
enum T {
    Op,
    Jump,
    Br,
    Call,
    Ret,
}

impl TargetInst for T {
    fn is_branch(&self) -> bool {
        matches!(self, T::Jump | T::Br | T::Call)
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

const INTS: [usize; 8] = [0, 1, 2, 3, 4, 5, 19, 20];
const FLOATS: [usize; 5] = [0, 1, 2, 3, 8];

fn int(r: usize) -> PReg {
    PReg::new(r, RegClass::Int)
}

fn float(r: usize) -> PReg {
    PReg::new(r, RegClass::Float)
}

fn env() -> Env {
    Env {
        regs: [
            INTS.iter().map(|&r| int(r)).collect(),
            FLOATS.iter().map(|&r| float(r)).collect(),
        ],
        scratch: [int(17), float(31)],
        transport: int(16),
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

#[derive(Default)]
struct B {
    insts: Vec<(T, Vec<Operand>, PRegSet)>,
    params: Vec<VReg>,
    succs: Vec<u32>,
    args: Vec<Vec<VReg>>,
    preds: Vec<u32>,
    resume: bool,
}

struct Gen {
    rng: Rng,
    v: VCode<T>,
    blocks: Vec<B>,
    /// Blocks made inside a cold arm, which the order puts last.
    cold: Vec<bool>,
    in_cold: bool,
}

impl Gen {
    fn block(&mut self) -> u32 {
        self.cold.push(self.in_cold);
        self.blocks.push(B::default());
        self.blocks.len() as u32 - 1
    }

    fn class(&mut self) -> RegClass {
        if self.rng.chance(30) {
            RegClass::Float
        } else {
            RegClass::Int
        }
    }

    fn pick(&mut self, avail: &[VReg]) -> Option<VReg> {
        (!avail.is_empty()).then(|| avail[self.rng.below(avail.len() as u64) as usize])
    }

    /// A value of class `c`, made in `b` when none is available.
    fn of_class(&mut self, b: u32, c: RegClass, avail: &mut Vec<VReg>) -> VReg {
        let have: Vec<VReg> = avail.iter().copied().filter(|v| v.class() == c).collect();
        if let Some(v) = self.pick(&have) {
            return v;
        }
        let v = self.v.new_vreg(c);
        self.blocks[b as usize]
            .insts
            .push((T::Op, vec![Operand::reg_def(v)], PRegSet::empty()));
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

    fn snapshot(&mut self, ops: &mut Vec<Operand>, avail: &[VReg]) {
        for _ in 0..self.rng.below(6) {
            if let Some(v) = self.pick(avail) {
                ops.push(Operand::new(
                    v,
                    OperandConstraint::Any,
                    OperandKind::Use,
                    OperandPos::Late,
                ));
            }
        }
    }

    fn op(&mut self, b: u32, avail: &mut Vec<VReg>) {
        // A constant, which the allocator may compute again.
        if self.rng.chance(15) {
            let c = self.class();
            let d = self.v.new_vreg(c);
            self.v.remat[d.vreg()] = if c == RegClass::Int {
                Remat::Int(self.rng.next())
            } else {
                Remat::Float(self.rng.next())
            };
            self.blocks[b as usize].insts.push((
                T::Op,
                vec![Operand::reg_def(d)],
                PRegSet::empty(),
            ));
            avail.push(d);
            return;
        }
        // A value computed from another, which the allocator may compute
        // again from it.
        if self.rng.chance(10) {
            let ints: Vec<VReg> = avail
                .iter()
                .copied()
                .filter(|v| v.class() == RegClass::Int)
                .collect();
            if let Some(src) = self.pick(&ints) {
                let d = self.v.new_vreg(RegClass::Int);
                self.v.remat[d.vreg()] = Remat::AndImm {
                    src: src.vreg() as u32,
                    imm: self.rng.next(),
                };
                self.blocks[b as usize].insts.push((
                    T::Op,
                    vec![Operand::reg_def(d), Operand::reg_use(src)],
                    PRegSet::empty(),
                ));
                avail.push(d);
                return;
            }
        }
        let mut ops = Vec::new();
        let mut clob = PRegSet::empty();
        let mut defs = Vec::new();
        match self.rng.below(10) {
            0..=5 => {
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
                for _ in 0..self.rng.below(4) {
                    let Some(v) = self.pick(avail) else { break };
                    let pos = if self.rng.chance(70) {
                        OperandPos::Early
                    } else {
                        OperandPos::Late
                    };
                    let late = pos == OperandPos::Late;
                    let c = if self.rng.chance(75) && fits(&mut need, v.class(), true, late) {
                        OperandConstraint::Reg
                    } else {
                        OperandConstraint::Any
                    };
                    ops.push(Operand::new(v, c, OperandKind::Use, pos));
                }
                if self.rng.chance(30) {
                    self.snapshot(&mut ops, avail);
                }
                for _ in 0..self.rng.below(3) {
                    let c = self.class();
                    let pos = if self.rng.chance(80) {
                        OperandPos::Late
                    } else {
                        OperandPos::Early
                    };
                    if !fits(&mut need, c, pos == OperandPos::Early, true) {
                        continue;
                    }
                    let d = self.v.new_vreg(c);
                    ops.push(Operand::new(
                        d,
                        OperandConstraint::Reg,
                        OperandKind::Def,
                        pos,
                    ));
                    defs.push(d);
                }
                if self.rng.chance(15) {
                    let c = self.class();
                    if fits(&mut need, c, true, true) {
                        ops.push(Operand::reg_temp(self.v.new_vreg(c)));
                    }
                }
            }
            6..=7 => {
                // A helper: arguments in x0.. and d0.., the result in x0 or d0.
                let (mut xn, mut dn) = (0, 0);
                for _ in 0..self.rng.below(4) {
                    let Some(v) = self.pick(avail) else { break };
                    let r = if v.class() == RegClass::Int {
                        xn += 1;
                        int(INTS[xn - 1])
                    } else {
                        dn += 1;
                        float(FLOATS[dn - 1])
                    };
                    ops.push(Operand::reg_fixed_use(v, r));
                }
                clob = caller_saved();
                if self.rng.chance(60) {
                    let c = self.class();
                    let d = self.v.new_vreg(c);
                    let r = if c == RegClass::Int { int(0) } else { float(0) };
                    ops.push(Operand::reg_fixed_def(d, r));
                    clob.remove(r);
                    defs.push(d);
                }
                if self.rng.chance(30) {
                    self.snapshot(&mut ops, avail);
                }
            }
            _ => {
                // An exit's snapshot.
                if let Some(v) = self.pick(avail)
                    && self.rng.chance(50)
                {
                    ops.push(Operand::reg_use(v));
                }
                self.snapshot(&mut ops, avail);
            }
        }
        self.blocks[b as usize].insts.push((T::Op, ops, clob));
        avail.extend(defs);
    }

    fn region(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        let mut cur = cur;
        for _ in 0..1 + self.rng.below(4) {
            match self.rng.below(10) {
                4..=5 if depth < 3 => cur = self.diamond(cur, depth, avail),
                6..=7 if depth < 3 => cur = self.looop(cur, depth, avail),
                8 => cur = self.call(cur, avail),
                _ => {
                    for _ in 0..1 + self.rng.below(4) {
                        self.op(cur, avail);
                    }
                }
            }
        }
        cur
    }

    fn params(&mut self, b: u32) {
        for _ in 0..self.rng.below(4) {
            let c = self.class();
            let p = self.v.new_vreg(c);
            self.blocks[b as usize].params.push(p);
        }
    }

    fn br(&mut self, b: u32, avail: &[VReg]) {
        let mut ops = Vec::new();
        if let Some(v) = self.pick(avail) {
            ops.push(Operand::reg_use(v));
        }
        self.blocks[b as usize]
            .insts
            .push((T::Br, ops, PRegSet::empty()));
    }

    fn jump(&mut self, from: u32, to: u32, avail: &mut Vec<VReg>) {
        let args = self.args_for(from, to, avail);
        self.blocks[from as usize]
            .insts
            .push((T::Jump, Vec::new(), PRegSet::empty()));
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
        if direct && self.rng.chance(30) {
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
        self.in_cold |= self.rng.chance(25);
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
        let direct = self.rng.chance(30).then(|| self.args_for(latch, h, avail));
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

    fn call(&mut self, cur: u32, avail: &mut Vec<VReg>) -> u32 {
        let mut ops = Vec::new();
        let ints: Vec<VReg> = avail
            .iter()
            .copied()
            .filter(|v| v.class() == RegClass::Int)
            .collect();
        for &r in &INTS[..self.rng.below(3) as usize] {
            if let Some(v) = self.pick(&ints) {
                ops.push(Operand::reg_fixed_use(v, int(r)));
            }
        }
        for _ in 0..self.rng.below(5) {
            if let Some(v) = self.pick(avail) {
                ops.push(Operand::new(
                    v,
                    OperandConstraint::Stack,
                    OperandKind::Use,
                    OperandPos::Late,
                ));
            }
        }
        self.blocks[cur as usize]
            .insts
            .push((T::Call, ops, all_regs()));
        let r = self.block();
        self.blocks[r as usize].resume = true;
        self.edge(cur, r, Vec::new());
        let mut clob = all_regs();
        let mut ops = Vec::new();
        for &r in &INTS[..self.rng.below(3) as usize] {
            let d = self.v.new_vreg(RegClass::Int);
            ops.push(Operand::reg_fixed_def(d, int(r)));
            clob.remove(int(r));
            avail.push(d);
        }
        self.blocks[r as usize].insts.push((T::Op, ops, clob));
        r
    }

    fn finish(mut self) -> VCode<T> {
        for (bi, b) in self.blocks.iter().enumerate() {
            let start = self.v.insts.len() as u32;
            for (t, ops, clob) in &b.insts {
                self.v.push(t.clone(), ops, *clob);
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
                b.resume,
            );
        }
        self.v.entry = RBlock::new(0);
        self.v
    }
}

/// A function, and its blocks in allocation order with how many are hot.
fn generate(seed: u64) -> (VCode<T>, Vec<u32>, usize) {
    let mut g = Gen {
        rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
        v: VCode::new(),
        blocks: Vec::new(),
        cold: Vec::new(),
        in_cold: false,
    };
    let entry = g.block();
    let mut avail = Vec::new();
    // The entry's parameters, fixed in x0.. as `MInst::Params`.
    let mut ops = Vec::new();
    for &r in &INTS[..1 + g.rng.below(3) as usize] {
        let p = g.v.new_vreg(RegClass::Int);
        ops.push(Operand::reg_fixed_def(p, int(r)));
        avail.push(p);
    }
    g.blocks[0].insts.push((T::Op, ops, PRegSet::empty()));
    let end = g.region(entry, 0, &mut avail);
    let mut ops = Vec::new();
    for _ in 0..g.rng.below(3) {
        if let Some(v) = g.pick(&avail) {
            ops.push(Operand::reg_use(v));
        }
    }
    g.blocks[end as usize]
        .insts
        .push((T::Ret, ops, PRegSet::empty()));
    // Odd blocks cold too: loop headers and blocks dominating hot ones
    // laid out after them.
    for b in 1..g.blocks.len() {
        if g.rng.chance(10) {
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
fn loops(f: &VCode<T>) -> (Vec<u32>, Vec<u32>) {
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

fn check_operands(v: &VCode<T>, out: &Output) -> Result<(), String> {
    super::check_operands(v, out)?;
    for (b, blk) in v.blocks.iter().enumerate() {
        if blk.resume
            && out
                .edits
                .iter()
                .any(|(p, _)| *p == ProgPoint::before(Inst::new(blk.start as usize)))
        {
            return Err(format!(
                "a move before resume block b{b}'s first instruction"
            ));
        }
    }
    Ok(())
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
        let mut checker = super::checker::Checker::new(&v);
        let (remat, src) = super::remat_table(&v);
        checker.allow_unmaterialized(remat, src);
        checker.prepare(&out);
        if let Err(e) = checker.run() {
            panic!("seed {seed}: {e:?}\n{}", v.print(&out, &order));
        }
        if let Err(e) = check_operands(&v, &out) {
            panic!("seed {seed}: {e}\n{}", v.print(&out, &order));
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
            let mk = |b: bool| if b { T::Br } else { T::Op };
            super::corpus::read(&d, mk).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
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
                let (got, want) = (c.f.print(&out, &c.order), c.f.print(&c.out, &c.order));
                eprintln!("-- got\n{got}\n-- recorded\n{want}");
            }
        }
        let mut checker = super::checker::Checker::new(&c.f);
        let (remat, src) = super::remat_table(&c.f);
        checker.allow_unmaterialized(remat, src);
        checker.prepare(&out);
        if let Err(e) = checker.run() {
            panic!("{}: {e:?}", p.display());
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
