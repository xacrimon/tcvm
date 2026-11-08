//! Target-neutral machine code before register allocation: instructions of a
//! target's `Inst` type in blocks with parameters, each with its operands in
//! a flat pool.

use crate::jit::backend::reg::{
    Block as RBlock, Inst as RInst, InstRange, Operand, OperandKind, PRegSet, RegClass, VReg,
};

/// What a target's instruction type tells the allocator.
pub(crate) trait TargetInst: Clone {
    fn is_branch(&self) -> bool;
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
    /// Per vreg: how to compute it again instead of reloading it.
    pub(crate) remat: Vec<Remat>,
    pub(crate) entry: RBlock,
    /// Per block: the header of the innermost loop holding it (itself for a
    /// header), and for a header the enclosing loop's; `u32::MAX` for none.
    pub(crate) loop_of: Vec<u32>,
    pub(crate) loop_parent: Vec<u32>,
}

/// A value the allocator may compute again where it is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Remat {
    None,
    /// An integer constant.
    Int(u64),
    /// A float constant, by its bits.
    Float(u64),
    /// `src & imm` (a table's pointer from its value), from wherever `src`
    /// is; only where `src` is live.
    AndImm {
        src: u32,
        imm: u64,
    },
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
            remat: Vec::new(),
            entry: RBlock::new(0),
            loop_of: Vec::new(),
            loop_parent: Vec::new(),
        }
    }

    pub(crate) fn new_vreg(&mut self, class: RegClass) -> VReg {
        let v = VReg::new(self.vreg_classes.len(), class);
        self.vreg_classes.push(class);
        self.remat.push(Remat::None);
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
}

impl<I: TargetInst + std::fmt::Debug> VCode<I> {
    /// The allocated code, blocks in `order` (`TCVM_JIT_DUMP=vcode`): each
    /// instruction's defs and uses with their locations, and the allocator's
    /// moves between them.
    pub(crate) fn print(&self, out: &crate::jit::backend::ra::Output, order: &[u32]) -> String {
        use std::fmt::Write;

        use crate::jit::backend::ra::{Edit, InstOrEdit};
        use crate::jit::backend::reg::{Allocation, AllocationKind};
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
                    InstOrEdit::Edit(Edit::Remat { to, vreg, src }) => {
                        let _ = writeln!(
                            s,
                            "        remat {} <- v{} {}",
                            loc(*to),
                            vreg.vreg(),
                            loc(*src)
                        );
                    }
                }
            }
            for (k, succ) in self.block_succs(rb).iter().enumerate() {
                let args = self.branch_blockparams(rb, k);
                let _ = writeln!(s, "  -> b{}({})", succ.index(), vregs(args));
            }
        }
        s
    }
}

impl<I: TargetInst> VCode<I> {
    pub(crate) fn num_insts(&self) -> usize {
        self.insts.len()
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn entry_block(&self) -> RBlock {
        self.entry
    }

    pub(crate) fn block_insns(&self, block: RBlock) -> InstRange {
        let b = &self.blocks[block.index()];
        InstRange::new(RInst::new(b.start as usize), RInst::new(b.end as usize))
    }

    pub(crate) fn block_succs(&self, block: RBlock) -> &[RBlock] {
        self.succs(block.index())
    }

    pub(crate) fn block_preds(&self, block: RBlock) -> &[RBlock] {
        let (s, e) = self.blocks[block.index()].preds;
        &self.bpool[s as usize..e as usize]
    }

    pub(crate) fn block_params(&self, block: RBlock) -> &[VReg] {
        let (s, e) = self.blocks[block.index()].params;
        &self.rpool[s as usize..e as usize]
    }

    pub(crate) fn is_branch(&self, insn: RInst) -> bool {
        self.insts[insn.index()].is_branch()
    }

    pub(crate) fn branch_blockparams(&self, block: RBlock, succ_idx: usize) -> &[VReg] {
        let (s, e) = self.args[self.blocks[block.index()].args as usize + succ_idx];
        &self.rpool[s as usize..e as usize]
    }

    pub(crate) fn inst_operands(&self, insn: RInst) -> &[Operand] {
        self.inst_operands_of(insn.index())
    }

    pub(crate) fn inst_clobbers(&self, insn: RInst) -> PRegSet {
        self.clobbers[insn.index()]
    }

    pub(crate) fn num_vregs(&self) -> usize {
        self.vreg_classes.len()
    }
}
