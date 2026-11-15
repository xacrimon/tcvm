//! Target-neutral machine code before register allocation: 16-byte
//! instruction records in blocks with parameters, read through the target's
//! opcode table (`Row`), with a side pool for what a record's three slots
//! do not hold.

#[cfg(feature = "tcvm-dev")]
use crate::jit::backend::reg::OperandKind;
use crate::jit::backend::reg::{
    Block as RBlock, Inst as RInst, InstRange, Operand, OperandConstraint, OperandKind as Kind,
    OperandPos, PReg, PRegSet, RegClass, VReg,
};
use crate::jit::bug;

/// One machine instruction, or a pseudo-instruction of its target's table.
/// What each slot holds is its opcode's row's to say.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct MInst {
    /// The target's opcode, an index of `VCode::rows`.
    pub(crate) op: u8,
    /// The target's operand size, or a load's or store's width.
    pub(crate) sz: u8,
    /// The target's condition code.
    pub(crate) cc: u8,
    /// How many of the list's leading words are definitions.
    pub(crate) n: u8,
    pub(crate) ops: [u32; 3],
}

const _: () = assert!(size_of::<MInst>() == 16);

/// Where a register operand may be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Constraint {
    /// Any register of its class.
    Any,
    /// The register of the use in slot `k`.
    Tied(u8),
    Fixed(PReg),
}

/// What one of a record's slots holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot {
    None,
    /// A virtual register's number. A use is read at the early point and a
    /// definition written at the late one unless `late` says otherwise.
    Reg {
        class: RegClass,
        def: bool,
        late: bool,
        constraint: Constraint,
    },
    /// 32 bits, zero-extended.
    Imm,
    /// The pool index of a 64-bit constant, low half first.
    Wide,
    /// The pool index of the list, and its length in words.
    Start,
    Len,
}

/// What the words of an instruction's list in the pool are: virtual
/// registers' numbers, the first `MInst::n` of them definitions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum List {
    None,
    /// A C call's: the definitions in the return register of their class,
    /// the uses in the argument registers of their class, in order.
    CArgs,
    /// Uses read at the late point from wherever each value is.
    AnyLoc,
}

/// The registers an instruction overwrites beside its definitions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Clobbers {
    None,
    /// The C caller-saved registers (`CAbi::caller_saved`).
    CallerSaved,
}

/// An instruction as the block layout sees it (`emit::thread_jumps`,
/// `emit::rotate_loops`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Flow {
    Fall,
    /// To successor 0.
    Jump,
    /// To successor 0 or 1.
    Branch,
    /// Leaves through the exit in its first immediate slot where its
    /// condition holds; `Exit` always.
    CondExit,
    Exit,
}

/// An opcode's whole convention: selection, the allocator and the encoder
/// read it here and nowhere else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Row {
    pub(crate) name: &'static str,
    /// Register slots first: the register in slot `k` is operand `k`.
    pub(crate) slots: [Slot; 3],
    pub(crate) list: List,
    pub(crate) clobbers: Clobbers,
    pub(crate) flow: Flow,
    /// It only defines a value, its operand 0, that the allocator may make
    /// again where used (`VCode::remat`): left without an allocation it has
    /// no code.
    pub(crate) remat: bool,
    /// More than one machine instruction, or none.
    pub(crate) pseudo: bool,
}

impl Row {
    pub(crate) fn regs(&self) -> usize {
        (self.slots.iter())
            .take_while(|s| matches!(s, Slot::Reg { .. }))
            .count()
    }

    fn slot_of(&self, kind: Slot) -> Option<usize> {
        self.slots.iter().position(|&s| s == kind)
    }

    /// What a table must hold for the decoder to read it.
    pub(crate) fn check(&self) -> Result<(), &'static str> {
        let n = self.regs();
        if self.slots[n..]
            .iter()
            .any(|s| matches!(s, Slot::Reg { .. }))
        {
            return Err("a register slot after another kind");
        }
        let (start, len) = (self.slot_of(Slot::Start), self.slot_of(Slot::Len));
        if start.is_some() != len.is_some() || start.is_some() != (self.list != List::None) {
            return Err("a list without its start and length, or those without a list");
        }
        for s in &self.slots[..n] {
            let Slot::Reg {
                def,
                constraint: Constraint::Tied(k),
                ..
            } = *s
            else {
                continue;
            };
            let to = self.slots.get(k as usize);
            if !def || !matches!(to, Some(Slot::Reg { def: false, .. })) {
                return Err("a tie that is not a definition's to a use");
            }
        }
        if self.remat && !(n == 1 && matches!(self.slots[0], Slot::Reg { def: true, .. })) {
            return Err("a rematerialisable row that is not one definition");
        }
        Ok(())
    }
}

/// The C calling convention's registers, by class (integer, float).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CAbi {
    pub(crate) args: [[PReg; 8]; 2],
    pub(crate) ret: [PReg; 2],
    pub(crate) caller_saved: PRegSet,
}

fn class_index(c: RegClass) -> usize {
    match c {
        RegClass::Int => 0,
        _ => 1,
    }
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

pub(crate) struct VCode {
    pub(crate) insts: Vec<MInst>,
    pub(crate) rows: &'static [Row],
    /// Lists of virtual registers' numbers, and 64-bit constants as two
    /// words, low half first.
    pub(crate) pool: Vec<u32>,
    /// Per instruction, where its register operands start among all of
    /// them, which is where its allocations are (`ra::Output`); one more
    /// closes the last.
    pub(crate) offs: Vec<u32>,
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

/// A pinned register, as a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pin {
    Base,
    Thread,
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
    /// The value of frame slot `r`, which holds it wherever the value is
    /// read (`Op::Home`).
    Slot(u8),
    /// A pinned register's value: one move from it.
    Pinned(Pin),
}

impl Remat {
    /// Made from nothing: where a register first serves it, once at a
    /// join, on an edge rather than in a parameter's group.
    pub(crate) fn is_free(self) -> bool {
        matches!(
            self,
            Remat::Int(_) | Remat::Float(_) | Remat::Slot(_) | Remat::Pinned(_)
        )
    }
}

fn run<T: Copy>(pool: &mut Vec<T>, xs: impl IntoIterator<Item = T>) -> (u32, u32) {
    let s = pool.len() as u32;
    pool.extend(xs);
    (s, pool.len() as u32)
}

impl VCode {
    pub(crate) fn new(rows: &'static [Row]) -> Self {
        VCode {
            insts: Vec::new(),
            rows,
            pool: Vec::new(),
            offs: vec![0],
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

    /// Empty it, keeping its storage.
    pub(crate) fn clear(&mut self) {
        self.insts.clear();
        self.pool.clear();
        self.offs.clear();
        self.offs.push(0);
        self.blocks.clear();
        self.bpool.clear();
        self.rpool.clear();
        self.args.clear();
        self.vreg_classes.clear();
        self.remat.clear();
        self.loop_of.clear();
        self.loop_parent.clear();
    }

    pub(crate) fn new_vreg(&mut self, class: RegClass) -> VReg {
        let v = VReg::new(self.vreg_classes.len(), class);
        self.vreg_classes.push(class);
        self.remat.push(Remat::None);
        v
    }

    /// Append `inst`, whose list is in the pool already.
    pub(crate) fn push(&mut self, inst: MInst) -> RInst {
        let i = RInst::new(self.insts.len());
        let row = &self.rows[inst.op as usize];
        let n = row.regs() as u32 + list_of(row, &inst).1;
        self.offs.push(self.offs[i.index()] + n);
        self.insts.push(inst);
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

    #[inline]
    pub(crate) fn row(&self, i: usize) -> &'static Row {
        &self.rows[self.insts[i].op as usize]
    }

    /// The 64-bit constant at pool index `at`.
    pub(crate) fn wide(&self, at: u32) -> u64 {
        self.pool[at as usize] as u64 | (self.pool[at as usize + 1] as u64) << 32
    }

    /// How many register operands all the instructions have.
    pub(crate) fn num_operands(&self) -> usize {
        self.offs[self.insts.len()] as usize
    }

    /// Where instruction `i`'s register operands start among all of them,
    /// and how many it has.
    #[inline]
    pub(crate) fn operands_of(&self, i: usize) -> (u32, usize) {
        (self.offs[i], (self.offs[i + 1] - self.offs[i]) as usize)
    }

    /// Operand `k` of instruction `i`, as the allocator takes it: a register
    /// slot, or past those a word of the list, by the row.
    pub(crate) fn operand(&self, abi: &CAbi, i: usize, k: usize) -> Operand {
        let m = &self.insts[i];
        let row = &self.rows[m.op as usize];
        if let Some(&Slot::Reg {
            class,
            def,
            late,
            constraint,
        }) = row.slots.get(k)
        {
            let constraint = match constraint {
                Constraint::Any => OperandConstraint::Reg,
                Constraint::Tied(t) => OperandConstraint::Reuse(t as usize),
                Constraint::Fixed(p) => OperandConstraint::FixedReg(p),
            };
            let kind = if def { Kind::Def } else { Kind::Use };
            let pos = if late {
                OperandPos::Late
            } else {
                OperandPos::Early
            };
            return Operand::new(VReg::new(m.ops[k] as usize, class), constraint, kind, pos);
        }
        let (start, _) = list_of(row, m);
        let j = k - row.regs();
        let reg = |j: usize| {
            let v = self.pool[start as usize + j] as usize;
            VReg::new(v, self.vreg_classes[v])
        };
        let v = reg(j);
        let c = class_index(v.class());
        match row.list {
            List::AnyLoc => Operand::new(v, OperandConstraint::Any, Kind::Use, OperandPos::Late),
            List::CArgs if j < m.n as usize => Operand::reg_fixed_def(v, abi.ret[c]),
            List::CArgs => {
                let before = (m.n as usize..j)
                    .filter(|&x| reg(x).class() == v.class())
                    .count();
                if before >= 8 {
                    bug!("a C call with more arguments than registers");
                }
                Operand::reg_fixed_use(v, abi.args[c][before])
            }
            List::None => bug!("operand {k} of {}", row.name),
        }
    }

    /// The operand at `at` among all the instructions' (`offs`).
    pub(crate) fn operand_at(&self, abi: &CAbi, at: u32) -> Operand {
        let i = self.offs.partition_point(|&o| o <= at) - 1;
        self.operand(abi, i, (at - self.offs[i]) as usize)
    }

    /// The registers instruction `i` overwrites beside its definitions.
    pub(crate) fn inst_clobbers(&self, abi: &CAbi, insn: RInst) -> PRegSet {
        let i = insn.index();
        let row = self.row(i);
        match row.clobbers {
            Clobbers::None => PRegSet::empty(),
            Clobbers::CallerSaved => {
                let mut set = abi.caller_saved;
                for k in 0..self.operands_of(i).1 {
                    let op = self.operand(abi, i, k);
                    if let (Kind::Def, OperandConstraint::FixedReg(p)) =
                        (op.kind(), op.constraint())
                    {
                        set.remove(p);
                    }
                }
                set
            }
        }
    }

    /// The exit an instruction of flow `CondExit` or `Exit` leaves through.
    pub(crate) fn exit_of(&self, i: usize) -> u32 {
        let row = self.row(i);
        match row.slot_of(Slot::Imm) {
            Some(k) => self.insts[i].ops[k],
            None => bug!("{} names no exit", row.name),
        }
    }
}

/// Where an instruction's list is in the pool, and its length.
fn list_of(row: &Row, m: &MInst) -> (u32, u32) {
    match (row.slot_of(Slot::Start), row.slot_of(Slot::Len)) {
        (Some(s), Some(l)) => (m.ops[s], m.ops[l]),
        _ => (0, 0),
    }
}

impl VCode {
    /// The allocated code, blocks in `order` (`TCVM_JIT_DUMP=vcode`): each
    /// instruction's defs and uses with their locations, and the allocator's
    /// moves between them.
    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn print(
        &self,
        abi: &CAbi,
        out: &crate::jit::backend::ra::Output,
        order: &[u32],
    ) -> String {
        use std::fmt::Write;

        use crate::jit::backend::ra::{Edit, InstOrEdit};
        use crate::jit::backend::reg::{Allocation, AllocationKind};
        let loc = |a: Allocation| match a.kind() {
            AllocationKind::Reg => crate::jit::backend::target::abi::reg_name(a.as_reg().unwrap()),
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
                        let n = self.operands_of(i.index()).1;
                        let ops = (0..n)
                            .map(|k| self.operand(abi, i.index(), k))
                            .zip(out.inst_allocs(self, i));
                        let side = |kind| {
                            (ops.clone())
                                .filter(|(o, _)| o.kind() == kind)
                                .map(|(o, &a)| format!("v{}:{}", o.vreg().vreg(), loc(a)))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        let m = &self.insts[i.index()];
                        let _ = writeln!(
                            s,
                            "  {:>4}  {} sz{} cc{} {:?}  {} <- {}",
                            i.index(),
                            self.row(i.index()).name,
                            m.sz,
                            m.cc,
                            m.ops,
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
            let term = RInst::new(self.blocks[b as usize].end as usize - 1);
            for (k, succ) in self.block_succs(rb).iter().enumerate() {
                let args = self.branch_blockparams(rb, k);
                let _ = writeln!(s, "  -> b{}({})", succ.index(), vregs(args));
                for (_, _, e) in out.edge_edits(term, k) {
                    match e {
                        Edit::Move { from, to } => {
                            let _ = writeln!(s, "        move {} <- {}", loc(*to), loc(*from));
                        }
                        Edit::Remat { to, vreg, src } => {
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
            }
        }
        s
    }
}

impl VCode {
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

    #[cfg(feature = "tcvm-dev")]
    pub(crate) fn is_branch(&self, insn: RInst) -> bool {
        matches!(self.row(insn.index()).flow, Flow::Jump | Flow::Branch)
    }

    pub(crate) fn branch_blockparams(&self, block: RBlock, succ_idx: usize) -> &[VReg] {
        let (s, e) = self.args[self.blocks[block.index()].args as usize + succ_idx];
        &self.rpool[s as usize..e as usize]
    }

    pub(crate) fn num_vregs(&self) -> usize {
        self.vreg_classes.len()
    }
}
