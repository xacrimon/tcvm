//! Target-neutral machine code before register allocation: instructions of a
//! target's `Inst` type in blocks with parameters, each with its operands in
//! a flat pool, as `regalloc2::Function`.

use regalloc2::{
    Block as RBlock, Function, Inst as RInst, InstRange, Operand, PRegSet, RegClass, VReg,
};

/// What a target's instruction type tells the allocator.
pub(crate) trait TargetInst: Clone {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;
}

pub(crate) struct VBlock {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) succs: Vec<RBlock>,
    pub(crate) preds: Vec<RBlock>,
    pub(crate) params: Vec<VReg>,
    /// Arguments per successor, for the branch ending the block.
    pub(crate) branch_args: Vec<Vec<VReg>>,
    /// A call's resume point: aligned and entered by a return.
    pub(crate) resume: bool,
}

pub(crate) struct VCode<I> {
    pub(crate) insts: Vec<I>,
    pub(crate) operands: Vec<Operand>,
    /// `operands[ops[i].0 .. ops[i].1]` are instruction `i`'s.
    pub(crate) ops: Vec<(u32, u32)>,
    pub(crate) clobbers: Vec<PRegSet>,
    pub(crate) blocks: Vec<VBlock>,
    pub(crate) vreg_classes: Vec<RegClass>,
    pub(crate) entry: RBlock,
}

impl<I: TargetInst> VCode<I> {
    pub(crate) fn new() -> Self {
        VCode {
            insts: Vec::new(),
            operands: Vec::new(),
            ops: Vec::new(),
            clobbers: Vec::new(),
            blocks: Vec::new(),
            vreg_classes: Vec::new(),
            entry: RBlock::new(0),
        }
    }

    pub(crate) fn new_vreg(&mut self, class: RegClass) -> VReg {
        let v = VReg::new(self.vreg_classes.len(), class);
        self.vreg_classes.push(class);
        v
    }

    pub(crate) fn push(&mut self, inst: I, operands: &[Operand], clobbers: PRegSet) -> RInst {
        let i = RInst::new(self.insts.len());
        let a = self.operands.len() as u32;
        self.operands.extend_from_slice(operands);
        self.ops.push((a, self.operands.len() as u32));
        self.insts.push(inst);
        self.clobbers.push(clobbers);
        i
    }

    pub(crate) fn inst_operands_of(&self, i: usize) -> &[Operand] {
        let (a, b) = self.ops[i];
        &self.operands[a as usize..b as usize]
    }
}

impl<I: TargetInst> Function for VCode<I> {
    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> RBlock {
        self.entry
    }

    fn block_insns(&self, block: RBlock) -> InstRange {
        let b = &self.blocks[block.index()];
        InstRange::new(RInst::new(b.start as usize), RInst::new(b.end as usize))
    }

    fn block_succs(&self, block: RBlock) -> &[RBlock] {
        &self.blocks[block.index()].succs
    }

    fn block_preds(&self, block: RBlock) -> &[RBlock] {
        &self.blocks[block.index()].preds
    }

    fn block_params(&self, block: RBlock) -> &[VReg] {
        &self.blocks[block.index()].params
    }

    fn is_ret(&self, insn: RInst) -> bool {
        self.insts[insn.index()].is_ret()
    }

    fn is_branch(&self, insn: RInst) -> bool {
        self.insts[insn.index()].is_branch()
    }

    fn branch_blockparams(&self, block: RBlock, _insn: RInst, succ_idx: usize) -> &[VReg] {
        &self.blocks[block.index()].branch_args[succ_idx]
    }

    fn inst_operands(&self, insn: RInst) -> &[Operand] {
        self.inst_operands_of(insn.index())
    }

    fn inst_clobbers(&self, insn: RInst) -> PRegSet {
        self.clobbers[insn.index()]
    }

    fn num_vregs(&self) -> usize {
        self.vreg_classes.len()
    }

    fn spillslot_size(&self, _regclass: RegClass) -> usize {
        1
    }
}
