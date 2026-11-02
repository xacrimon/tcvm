//! Target-neutral machine code before register allocation: instructions of a
//! target's `Inst` type in blocks with parameters, each with its operands in
//! a flat pool, as `regalloc2::Function`.

use regalloc2::{
    Block as RBlock, Function, Inst as RInst, InstRange, Operand, OperandKind, PRegSet, RegClass,
    VReg,
};

/// What a target's instruction type tells the allocator.
pub(crate) trait TargetInst: Clone {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;
}

pub(crate) struct VBlock {
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// Runs of `VCode::bpool`.
    succs: (u32, u32),
    preds: (u32, u32),
    /// A run of `VCode::rpool`.
    params: (u32, u32),
    /// `args[k]` is the run of `rpool` successor `k` is passed.
    args: u32,
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
    bpool: Vec<RBlock>,
    rpool: Vec<VReg>,
    args: Vec<(u32, u32)>,
    pub(crate) vreg_classes: Vec<RegClass>,
    pub(crate) entry: RBlock,
}

fn run<T: Copy>(pool: &mut Vec<T>, xs: impl IntoIterator<Item = T>) -> (u32, u32) {
    let s = pool.len() as u32;
    pool.extend(xs);
    (s, pool.len() as u32)
}

impl<I: TargetInst> VCode<I> {
    pub(crate) fn new() -> Self {
        VCode {
            insts: Vec::new(),
            operands: Vec::new(),
            ops: Vec::new(),
            clobbers: Vec::new(),
            blocks: Vec::new(),
            bpool: Vec::new(),
            rpool: Vec::new(),
            args: Vec::new(),
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

    /// Add a block over instructions `start..end`; `args` holds each
    /// successor's arguments in turn, `nargs[k]` of them for successor `k`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_block(
        &mut self,
        start: u32,
        end: u32,
        succs: &[RBlock],
        preds: impl IntoIterator<Item = RBlock>,
        params: impl IntoIterator<Item = VReg>,
        args: &[VReg],
        nargs: &[u32],
        resume: bool,
    ) {
        let succs = run(&mut self.bpool, succs.iter().copied());
        let preds = run(&mut self.bpool, preds);
        let params = run(&mut self.rpool, params);
        let at = self.args.len() as u32;
        let mut k = 0;
        for &n in nargs {
            let r = run(&mut self.rpool, args[k..k + n as usize].iter().copied());
            self.args.push(r);
            k += n as usize;
        }
        self.blocks.push(VBlock {
            start,
            end,
            succs,
            preds,
            params,
            args: at,
            resume,
        });
    }

    pub(crate) fn succs(&self, b: usize) -> &[RBlock] {
        let (s, e) = self.blocks[b].succs;
        &self.bpool[s as usize..e as usize]
    }

    pub(crate) fn inst_operands_of(&self, i: usize) -> &[Operand] {
        let (a, b) = self.ops[i];
        &self.operands[a as usize..b as usize]
    }

    /// Which vregs an operand or a branch argument reads.
    pub(crate) fn read_vregs(&self) -> Vec<bool> {
        let mut read = vec![false; self.vreg_classes.len()];
        for k in 0..self.insts.len() {
            for op in self.inst_operands_of(k) {
                if op.kind() == OperandKind::Use {
                    read[op.vreg().vreg()] = true;
                }
            }
        }
        for &(s, e) in &self.args {
            for v in &self.rpool[s as usize..e as usize] {
                read[v.vreg()] = true;
            }
        }
        read
    }

    /// Make instruction `i` `by`, which has no operands.
    pub(crate) fn replace_bare(&mut self, i: usize, by: I) {
        self.insts[i] = by;
        self.ops[i].1 = self.ops[i].0;
    }
}

impl<I: TargetInst + std::fmt::Debug> VCode<I> {
    /// The allocated code, blocks in `order` (`TCVM_JIT_DUMP=vcode`): each
    /// instruction's defs and uses with their locations, and the allocator's
    /// moves between them.
    pub(crate) fn print(&self, out: &regalloc2::Output, order: &[u32]) -> String {
        use std::fmt::Write;

        use regalloc2::{Allocation, AllocationKind, Edit, InstOrEdit};
        let loc = |a: Allocation| match a.kind() {
            AllocationKind::Reg => {
                let r = a.as_reg().unwrap();
                let c = match r.class() {
                    RegClass::Int => 'x',
                    RegClass::Float => 'd',
                    RegClass::Vector => 'q',
                };
                format!("{c}{}", r.hw_enc())
            }
            AllocationKind::Stack => format!("[s{}]", a.as_stack().unwrap().index()),
            AllocationKind::None => "-".into(),
        };
        let vregs = |vs: &[VReg]| {
            vs.iter()
                .map(|v| format!("v{}", v.vreg()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut s = String::new();
        for &b in order {
            let rb = RBlock::new(b as usize);
            let preds: Vec<String> = (self.block_preds(rb).iter())
                .map(|p| format!("b{}", p.index()))
                .collect();
            let resume = if self.blocks[b as usize].resume {
                " resume"
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "b{b}({}){resume} <- {}",
                vregs(self.block_params(rb)),
                preds.join(" ")
            );
            for ie in out.block_insts_and_edits(self, rb) {
                match ie {
                    InstOrEdit::Inst(i) => {
                        let ops = self.inst_operands(i).iter().zip(out.inst_allocs(i));
                        let side = |kind| {
                            (ops.clone())
                                .filter(|(o, _)| o.kind() == kind)
                                .map(|(o, &a)| format!("v{}:{}", o.vreg().vreg(), loc(a)))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        let _ = writeln!(
                            s,
                            "  {:>4}  {:?}  {} <- {}",
                            i.index(),
                            self.insts[i.index()],
                            side(OperandKind::Def),
                            side(OperandKind::Use)
                        );
                    }
                    InstOrEdit::Edit(Edit::Move { from, to }) => {
                        let _ = writeln!(s, "        move {} <- {}", loc(*to), loc(*from));
                    }
                }
            }
            for (k, succ) in self.block_succs(rb).iter().enumerate() {
                let end = RInst::new(self.blocks[b as usize].end as usize - 1);
                let args = self.branch_blockparams(rb, end, k);
                let _ = writeln!(s, "  -> b{}({})", succ.index(), vregs(args));
            }
        }
        s
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
        self.succs(block.index())
    }

    fn block_preds(&self, block: RBlock) -> &[RBlock] {
        let (s, e) = self.blocks[block.index()].preds;
        &self.bpool[s as usize..e as usize]
    }

    fn block_params(&self, block: RBlock) -> &[VReg] {
        let (s, e) = self.blocks[block.index()].params;
        &self.rpool[s as usize..e as usize]
    }

    fn is_ret(&self, insn: RInst) -> bool {
        self.insts[insn.index()].is_ret()
    }

    fn is_branch(&self, insn: RInst) -> bool {
        self.insts[insn.index()].is_branch()
    }

    fn branch_blockparams(&self, block: RBlock, _insn: RInst, succ_idx: usize) -> &[VReg] {
        let (s, e) = self.args[self.blocks[block.index()].args as usize + succ_idx];
        &self.rpool[s as usize..e as usize]
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
