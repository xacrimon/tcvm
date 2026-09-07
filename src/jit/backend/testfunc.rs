//! A CFG built by hand, for tests that are about the allocator and nothing else.
//!
//! Lives outside `regalloc` because it is not only `regalloc` that needs it: the
//! next-use analysis and the spiller take the same `RegallocFunc` and want to be
//! tested against a CFG whose liveness the test states outright, rather than
//! against whatever machine IR a Lua function happens to lower to.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;

use super::regalloc::{Block, Inst, Operand, PReg, RegClass, RegallocFunc, VReg};

/// A CFG built directly — no machine IR, no instruction selection, no target.
///
/// This is the point of the module boundary: a register allocation test says
/// what is live where and nothing else, and a failure is a register allocation
/// bug rather than a lowering bug that happens to show up here.
#[derive(Default)]
pub struct TestFunc {
    blocks: Vec<Vec<Inst>>,
    succs: Vec<Vec<Block>>,
    defs: Vec<Vec<Operand>>,
    uses: Vec<Vec<Operand>>,
    temps: Vec<Vec<Operand>>,
    clobbers: Vec<Vec<PReg>>,
    classes: Vec<RegClass>,
    params: Vec<Vec<VReg>>,
    jump_args: Vec<Vec<VReg>>,
    phys: HashMap<VReg, PReg, RandomState>,
    remat: HashMap<VReg, Inst, RandomState>,
}

impl TestFunc {
    pub fn block(&mut self) -> Block {
        self.blocks.push(Vec::new());
        self.succs.push(Vec::new());
        self.params.push(Vec::new());
        self.jump_args.push(Vec::new());
        Block(self.blocks.len() as u32 - 1)
    }

    /// Give `b` these parameters — i.e. keep this function in SSA form.
    pub fn params(&mut self, b: Block, ps: &[VReg]) {
        self.params[b.0 as usize] = ps.to_vec();
    }

    /// The arguments `b` passes along its outgoing edge.
    pub fn pass(&mut self, b: Block, args: &[VReg]) {
        self.jump_args[b.0 as usize] = args.to_vec();
    }

    pub fn vreg(&mut self, class: RegClass) -> VReg {
        self.classes.push(class);
        VReg(self.classes.len() as u32 - 1)
    }

    pub fn int(&mut self) -> VReg {
        self.vreg(RegClass::Int)
    }

    pub fn inst(&mut self, b: Block, defs: Vec<Operand>, uses: Vec<Operand>) -> Inst {
        let i = self.defs.len();
        self.defs.push(defs);
        self.uses.push(uses);
        self.temps.push(Vec::new());
        self.clobbers.push(Vec::new());
        self.blocks[b.0 as usize].push(i);
        i
    }

    /// Give instruction `i` `n` fresh integer temp registers.
    pub fn temp(&mut self, i: Inst, n: usize) {
        for _ in 0..n {
            let t = self.int();
            self.temps[i].push(Operand::reg(t));
        }
    }

    pub fn goto(&mut self, b: Block, targets: &[Block]) {
        self.succs[b.0 as usize] = targets.to_vec();
    }

    pub fn clobber(&mut self, i: Inst, r: PReg) {
        self.clobbers[i].push(r);
    }

    pub fn hint(&mut self, v: VReg, r: PReg) {
        self.phys.insert(v, r);
    }

    /// Mark `v` as rematerializable, defined by instruction `i`.
    pub fn set_remat(&mut self, v: VReg, i: Inst) {
        self.remat.insert(v, i);
    }
}

impl RegallocFunc for TestFunc {
    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    fn entry(&self) -> Block {
        Block(0)
    }
    fn block_insts(&self, b: Block) -> &[Inst] {
        &self.blocks[b.0 as usize]
    }
    fn succs(&self, b: Block) -> Vec<Block> {
        self.succs[b.0 as usize].clone()
    }
    fn num_insts(&self) -> usize {
        self.defs.len()
    }
    fn defs(&self, i: Inst) -> &[Operand] {
        &self.defs[i]
    }
    fn uses(&self, i: Inst) -> &[Operand] {
        &self.uses[i]
    }
    fn clobbers(&self, i: Inst) -> &[PReg] {
        &self.clobbers[i]
    }
    fn num_vregs(&self) -> usize {
        self.classes.len()
    }
    fn class(&self, v: VReg) -> RegClass {
        self.classes[v.0 as usize]
    }
    fn phys_hint(&self, v: VReg) -> Option<PReg> {
        self.phys.get(&v).copied()
    }
    fn temps(&self, i: Inst) -> &[Operand] {
        &self.temps[i]
    }
    fn remat(&self, v: VReg) -> Option<Inst> {
        self.remat.get(&v).copied()
    }
    fn block_params(&self, b: Block) -> &[VReg] {
        &self.params[b.0 as usize]
    }
    fn jump_args(&self, b: Block) -> &[VReg] {
        &self.jump_args[b.0 as usize]
    }
}
