//! Random functions shaped like lowering's (loops and joins with block
//! parameters, calls into resume blocks clobbering everything, helpers with
//! fixed registers, snapshot uses), allocated with few registers and
//! checked by regalloc2's checker and for operands sharing a register
//! within an instruction, which that checker does not see.

use regalloc2::{
    Block as RBlock, Inst, MachineEnv, Operand, OperandConstraint, OperandKind, OperandPos, Output,
    PReg, PRegSet, ProgPoint, RegClass, VReg,
};

use super::{Env, run};
use crate::jit::backend::vcode::{TargetInst, VCode};

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

    fn is_ret(&self) -> bool {
        matches!(self, T::Ret)
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

fn envs() -> (Env, MachineEnv) {
    let ints: Vec<PReg> = INTS.iter().map(|&r| int(r)).collect();
    let floats: Vec<PReg> = FLOATS.iter().map(|&r| float(r)).collect();
    let set = |rs: &[PReg]| {
        let mut s = PRegSet::empty();
        for &r in rs {
            s.add(r);
        }
        s
    };
    let me = MachineEnv {
        preferred_regs_by_class: [set(&ints[..6]), set(&floats[..4]), PRegSet::empty()],
        non_preferred_regs_by_class: [set(&ints[6..]), set(&floats[4..]), PRegSet::empty()],
        scratch_by_class: [Some(int(17)), Some(float(31)), None],
        fixed_stack_slots: Vec::new(),
    };
    let env = Env {
        regs: [ints, floats],
        scratch: [int(17), float(31)],
        transport: int(16),
    };
    (env, me)
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

    fn diamond(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        self.br(cur, avail);
        let t = self.block();
        self.edge(cur, t, Vec::new());
        let mut at = avail.clone();
        let tl = self.region(t, depth + 1, &mut at);
        let was = self.in_cold;
        self.in_cold |= self.rng.chance(25);
        let e = self.block();
        self.edge(cur, e, Vec::new());
        let mut ae = avail.clone();
        let el = self.region(e, depth + 1, &mut ae);
        self.in_cold = was;
        let j = self.block();
        self.params(j);
        self.jump(tl, j, &mut at);
        self.jump(el, j, &mut ae);
        avail.extend(self.blocks[j as usize].params.clone());
        j
    }

    fn looop(&mut self, cur: u32, depth: u32, avail: &mut Vec<VReg>) -> u32 {
        let h = self.block();
        self.params(h);
        self.jump(cur, h, avail);
        avail.extend(self.blocks[h as usize].params.clone());
        let latch = self.region(h, depth + 1, avail);
        self.br(latch, avail);
        let back = self.block();
        self.edge(latch, back, Vec::new());
        self.jump(back, h, avail);
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
    (g.finish(), order, hot)
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

#[test]
fn random_functions_allocate_correctly() {
    let (env, me) = envs();
    let n: u64 = std::env::var("TCVM_RA_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let first: u64 = std::env::var("TCVM_RA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for seed in first..first + n {
        let (v, order, hot) = generate(seed);
        let out = match run(&v, &env, &order, hot) {
            Ok(out) => out,
            Err(e) => panic!("seed {seed}: {e}"),
        };
        let mut checker = regalloc2::checker::Checker::new(&v, &me);
        checker.prepare(&out);
        if let Err(e) = checker.run() {
            panic!("seed {seed}: {e:?}\n{}", v.print(&out, &order));
        }
        if let Err(e) = check_operands(&v, &out) {
            panic!("seed {seed}: {e}\n{}", v.print(&out, &order));
        }
    }
}
