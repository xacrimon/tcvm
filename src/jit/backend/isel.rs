//! Instruction selection: optimizing IR to machine IR.
//!
//! Three things happen here, and only these three. Nothing is optimized.
//!
//! 1. **Values get machine locations.** An IR value of `Rep::Val` becomes a
//!    payload register plus a [`Tag`], which is an *immediate* whenever the type
//!    set is monomorphic. This is why the bulk of the pack/unpack ops emit no
//!    code at all: `pack.int` of an `i64` is the same 64 bits with a tag the
//!    compiler already knows, so it is a rename, not an instruction.
//!
//! 2. **Block parameters are destroyed into copies.** Arguments are moved into
//!    the target's parameter registers on the edge. The moves are *parallel* —
//!    a loop that swaps two variables produces a cycle, and sequentializing it
//!    naively would clobber. On a two-way branch the copies go in a dedicated
//!    edge block, because a critical edge has nowhere else to put them.
//!
//! 3. **Guards get exit stubs.** Each guard records what the interpreter needs
//!    written back, and lists every one of those values among its *uses* — which
//!    is what keeps them alive through register allocation. A value that the
//!    fast path never reads again but that a deopt still needs would otherwise
//!    have its register reassigned out from under the stub.
//!
//! Anything outside the supported op set declines. Declining is always a legal
//! answer: the region simply stays interpreted.

use crate::env::value::ValueKind;
use crate::jit::backend::layout;
use crate::jit::backend::mach::{
    AluOp, ExitId, ExitSrc, ExitStub, FAluOp, MBlock, MFunc, MInst, MOp, RegClass, Tag, VReg, Width,
};
use crate::jit::backend::regalloc::{Inst, RegallocFunc};
use crate::jit::ir::op::{Cc, FloatOp, IntOp, Op};
use crate::jit::ir::ty::{Rep, Ty, TypeSet};
use crate::jit::ir::{Block, Func, Val};

#[derive(Debug)]
pub enum IselError {
    /// An op the backend does not implement yet.
    Unsupported(&'static str),
    /// A guard whose target set is not a single Lua type, so it cannot become one
    /// tag compare.
    PolymorphicGuard(TypeSet),
}

/// The machine realization of one IR value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    /// `Rep::I64`, `Rep::Ptr`, or `Rep::B1` — one general register.
    Int(VReg),
    /// `Rep::F64` — one float register.
    Float(VReg),
    /// `Rep::Val` — payload register, plus a tag that is usually an immediate.
    Boxed { payload: VReg, tag: Tag },
}

/// The source of one parallel-copy move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CopySrc {
    Reg(VReg),
    Imm(i64),
}

pub fn select(f: &Func<'_>) -> Result<MFunc, IselError> {
    let mut isel = Isel {
        f,
        m: MFunc::new(),
        slots: vec![None; f.num_values()],
        bmap: Vec::new(),
        base: VReg(0),
    };
    isel.run()?;
    Ok(isel.m)
}

struct Isel<'a, 'gc> {
    f: &'a Func<'gc>,
    m: MFunc,
    slots: Vec<Option<Slot>>,
    bmap: Vec<MBlock>,
    /// The Lua frame base pointer, `&stack[base]`. Every stack access is relative
    /// to this.
    base: VReg,
}

impl<'a, 'gc> Isel<'a, 'gc> {
    fn run(&mut self) -> Result<(), IselError> {
        // Machine blocks mirror IR blocks one-to-one; edge blocks are appended
        // afterwards, so this index mapping stays valid.
        for _ in 0..self.f.num_blocks() {
            let b = self.m.new_block();
            self.bmap.push(b);
        }
        self.m.entry = self.bmap[self.f.entry.index()];

        // Placeholder stubs, filled in as the guards that own them are selected.
        // `ExitId` and `ExitRef` are the same index by construction.
        self.m.exits = (0..self.f.num_exits())
            .map(|_| ExitStub {
                pc: 0,
                slots: Vec::new(),
                inst: usize::MAX,
            })
            .collect();
        self.m.pinned_regs = self.f.pinned_regs.clone();
        self.m.max_lua_reg = (0..self.f.num_frame_states())
            .map(|i| {
                self.f
                    .frame_state(crate::jit::ir::FsRef(i as u32))
                    .regs
                    .len()
            })
            .max()
            .unwrap_or(0)
            .saturating_sub(1) as u8;

        // The frame base, defined once at the top of the entry block.
        self.base = self.m.new_vreg(RegClass::Int);
        let entry = self.m.entry;
        let base = self.base;
        self.m.frame_base = base;
        self.m
            .push(entry, MInst::new(MOp::EntryArg(1), vec![base], vec![]));

        // Parameters get their registers before any block is walked: an edge into
        // a block may be selected before the block itself.
        for b in self.f.blocks() {
            for &p in &self.f.block(b).params {
                let slot = self.fresh_slot(self.f.ty(p));
                self.slots[p.index()] = Some(slot);
            }
        }

        // Every block but the entry receives its parameters on an edge. The entry
        // has no predecessor: its parameters are the region's live-in Lua
        // registers, so the prologue reads them off the stack. Their tags are
        // *not* loaded — the entry context already asserts each one's type, and
        // that assertion is the caller's to uphold, not something to re-check
        // here.
        let params = self.f.block(self.f.entry).params.clone();
        let regs = self.f.entry_regs.clone();
        assert_eq!(
            params.len(),
            regs.len(),
            "entry_regs must name every entry parameter"
        );
        for (&p, &r) in params.iter().zip(&regs) {
            self.load_lua_reg(entry, r, p);
        }

        // Reverse postorder, so every definition is selected before its uses and
        // `self.slots` is populated when a guard reaches for a frame-state value.
        for b in self.rpo() {
            self.block(b)?;
        }
        Ok(())
    }

    /// Reverse postorder over the IR CFG.
    fn rpo(&self) -> Vec<Block> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.f.num_blocks()];
        // (block, whether its successors have been pushed)
        let mut stack = vec![(self.f.entry, false)];
        seen[self.f.entry.index()] = true;
        while let Some((b, expanded)) = stack.pop() {
            if expanded {
                order.push(b);
                continue;
            }
            stack.push((b, true));
            let term = *self
                .f
                .block(b)
                .insts
                .last()
                .expect("block has a terminator");
            for t in self.f.inst(term).targets.iter().rev() {
                if !seen[t.block.index()] {
                    seen[t.block.index()] = true;
                    stack.push((t.block, false));
                }
            }
        }
        order.reverse();
        order
    }

    // --- value slots --------------------------------------------------------

    fn fresh_slot(&mut self, ty: Ty) -> Slot {
        match ty.rep {
            Rep::I64 | Rep::Ptr | Rep::B1 => Slot::Int(self.m.new_vreg(RegClass::Int)),
            Rep::F64 => Slot::Float(self.m.new_vreg(RegClass::Float)),
            Rep::Val => {
                let payload = self.m.new_vreg(RegClass::Int);
                let tag = match kind_of(ty.set) {
                    Some(k) => Tag::Const(k),
                    None => Tag::Dyn(self.m.new_vreg(RegClass::Int)),
                };
                Slot::Boxed { payload, tag }
            }
        }
    }

    fn slot(&self, v: Val) -> Slot {
        self.slots[v.index()].expect("value used before it was selected")
    }

    fn set(&mut self, v: Val, s: Slot) {
        self.slots[v.index()] = Some(s);
    }

    /// Every machine register a slot occupies. Used to build a guard's use list.
    fn regs_of(s: Slot) -> Vec<VReg> {
        match s {
            Slot::Int(v) | Slot::Float(v) => vec![v],
            Slot::Boxed {
                payload,
                tag: Tag::Const(_),
            } => vec![payload],
            Slot::Boxed {
                payload,
                tag: Tag::Dyn(t),
            } => vec![payload, t],
        }
    }

    fn int(&self, v: Val) -> VReg {
        match self.slot(v) {
            Slot::Int(r) => r,
            other => panic!("expected an integer slot, got {other:?}"),
        }
    }

    fn float(&self, v: Val) -> VReg {
        match self.slot(v) {
            Slot::Float(r) => r,
            other => panic!("expected a float slot, got {other:?}"),
        }
    }

    fn boxed(&self, v: Val) -> (VReg, Tag) {
        match self.slot(v) {
            Slot::Boxed { payload, tag } => (payload, tag),
            other => panic!("expected a boxed slot, got {other:?}"),
        }
    }

    // --- emission -----------------------------------------------------------

    fn emit(&mut self, b: MBlock, op: MOp, defs: Vec<VReg>, uses: Vec<VReg>) -> Inst {
        self.m.push(b, MInst::new(op, defs, uses))
    }

    /// Emit the instruction that branches to `e`'s stub, and tell the stub which
    /// one it was. The stub reads its values out of this instruction's uses — it
    /// has no operands of its own — so it cannot be encoded until it knows.
    ///
    /// `uses` are the guard's real operands (registers); `keepalives` are the
    /// stub's values, held alive but left wherever they landed.
    fn emit_exiting(
        &mut self,
        b: MBlock,
        op: MOp,
        uses: Vec<VReg>,
        keepalives: Vec<VReg>,
        e: ExitId,
    ) {
        let i = self.m.push(b, MInst::guard(op, uses, keepalives));
        self.m.exits[e.0 as usize].inst = i;
    }

    fn emit_imm(&mut self, b: MBlock, imm: i64) -> VReg {
        let r = self.m.new_vreg(RegClass::Int);
        self.emit(b, MOp::Imm(imm), vec![r], vec![]);
        r
    }

    // --- blocks -------------------------------------------------------------

    fn block(&mut self, b: Block) -> Result<(), IselError> {
        let mb = self.bmap[b.index()];
        let insts = self.f.block(b).insts.clone();
        for i in insts {
            let d = self.f.inst(i);
            if d.op.is_terminator() {
                self.terminator(mb, b, i)?;
            } else {
                self.inst(mb, i)?;
            }
        }
        Ok(())
    }

    fn inst(&mut self, mb: MBlock, i: crate::jit::ir::Inst) -> Result<(), IselError> {
        let d = self.f.inst(i);
        let args = d.args.clone();
        let results = d.results.clone();
        let op = d.op;
        let exit = d.exit;

        match op {
            // --- constants ---------------------------------------------------
            Op::IConst(n) => {
                let r = self.emit_imm(mb, n);
                self.set(results[0], Slot::Int(r));
            }
            Op::FConst(bits) => {
                // No aarch64 immediate reaches an arbitrary double, so the bits go
                // through a general register.
                let g = self.emit_imm(mb, bits as i64);
                let f = self.m.new_vreg(RegClass::Float);
                self.emit(mb, MOp::BitsToFloat, vec![f], vec![g]);
                self.set(results[0], Slot::Float(f));
            }
            Op::BConst(v) => {
                let r = self.emit_imm(mb, v as i64);
                self.set(results[0], Slot::Int(r));
            }
            Op::KConst(c) => {
                let kind = self.f.pool.value(c).kind();
                let payload = self.m.new_vreg(RegClass::Int);
                self.emit(mb, MOp::ConstPayload(c), vec![payload], vec![]);
                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(kind),
                    },
                );
            }

            // --- representation changes: renames, not instructions ------------
            //
            // A `Value` is a tagged pair, so packing an `i64` writes a tag the
            // compiler already knows and keeps the same 64 payload bits. There is
            // nothing to compute. Only the float forms move anything, and only
            // because the bits have to change register file.
            Op::PackInt => {
                let payload = self.int(args[0]);
                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(ValueKind::Integer),
                    },
                );
            }
            Op::PackBool => {
                let payload = self.int(args[0]);
                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(ValueKind::Boolean),
                    },
                );
            }
            Op::PackFloat => {
                let f = self.float(args[0]);
                let payload = self.m.new_vreg(RegClass::Int);
                self.emit(mb, MOp::FloatToBits, vec![payload], vec![f]);
                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(ValueKind::Float),
                    },
                );
            }
            Op::UnpackInt | Op::UnpackPtr => {
                let (payload, _) = self.boxed(args[0]);
                self.set(results[0], Slot::Int(payload));
            }
            Op::UnpackFloat => {
                let (payload, _) = self.boxed(args[0]);
                let f = self.m.new_vreg(RegClass::Float);
                self.emit(mb, MOp::BitsToFloat, vec![f], vec![payload]);
                self.set(results[0], Slot::Float(f));
            }
            Op::SiToFp => {
                let g = self.int(args[0]);
                let f = self.m.new_vreg(RegClass::Float);
                self.emit(mb, MOp::SiToFp, vec![f], vec![g]);
                self.set(results[0], Slot::Float(f));
            }
            Op::TagOf => {
                let (_, tag) = self.boxed(args[0]);
                let r = match tag {
                    Tag::Const(k) => self.emit_imm(mb, layout::kind(k) as i64),
                    Tag::Dyn(t) => t,
                };
                self.set(results[0], Slot::Int(r));
            }

            // --- guards -------------------------------------------------------
            Op::GuardType(set) => {
                let e = ExitId(exit.expect("a guard carries an exit").0);
                let (payload, tag) = self.boxed(args[0]);
                let want = kind_of(set).ok_or(IselError::PolymorphicGuard(set))?;

                match tag {
                    // The tag is already proven — a dominating guard or the entry
                    // context established it. Nothing to check.
                    Tag::Const(k) => {
                        assert_eq!(k, want, "guard on a value already known to be {k:?}");
                    }
                    Tag::Dyn(t) => {
                        self.stub(e, i);
                        self.emit_exiting(
                            mb,
                            MOp::GuardCmpImm {
                                cc: Cc::Eq,
                                imm: layout::kind(want) as i64,
                                exit: e,
                            },
                            vec![t],
                            self.stub_regs(i),
                            e,
                        );
                    }
                }
                // Refined, but the same bits in the same register — and the tag is
                // now a constant, so its register (if any) simply dies here.
                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(want),
                    },
                );
            }
            Op::GuardShape(s) => {
                let e = ExitId(exit.expect("a guard carries an exit").0);
                let (payload, _) = self.boxed(args[0]);
                self.stub(e, i);

                let got = self.m.new_vreg(RegClass::Int);
                self.emit(
                    mb,
                    MOp::Load {
                        off: layout::table::SHAPE as i32,
                        width: Width::U64,
                    },
                    vec![got],
                    vec![payload],
                );
                let want_reg = self.m.new_vreg(RegClass::Int);
                self.emit(mb, MOp::ShapeAddr(s), vec![want_reg], vec![]);

                self.emit_exiting(
                    mb,
                    MOp::GuardCmp {
                        cc: Cc::Eq,
                        exit: e,
                    },
                    vec![got, want_reg],
                    self.stub_regs(i),
                    e,
                );

                self.set(
                    results[0],
                    Slot::Boxed {
                        payload,
                        tag: Tag::Const(ValueKind::Table),
                    },
                );
            }
            Op::GuardCond => {
                let e = ExitId(exit.expect("a guard carries an exit").0);
                self.stub(e, i);
                let c = self.int(args[0]);
                self.emit_exiting(mb, MOp::GuardNz { exit: e }, vec![c], self.stub_regs(i), e);
            }
            // A watchpoint, not a check. It emits nothing; the compiled artifact
            // records the dependency and a metatable write invalidates it.
            Op::AssumeNoMm(s, _) => {
                if !self.m.watchpoints.contains(&s) {
                    self.m.watchpoints.push(s);
                }
            }

            // --- tables -------------------------------------------------------
            Op::TabProps => {
                let (payload, _) = self.boxed(args[0]);
                let p = self.m.new_vreg(RegClass::Int);
                self.emit(
                    mb,
                    MOp::Load {
                        off: layout::table::PROPS_PTR as i32,
                        width: Width::U64,
                    },
                    vec![p],
                    vec![payload],
                );
                self.set(results[0], Slot::Int(p));
            }
            Op::SlotGet(n) => {
                let base = self.int(args[0]);
                let off = n as i32 * layout::val::SIZE as i32;
                let ty = self.f.ty(results[0]);

                let payload = self.m.new_vreg(RegClass::Int);
                self.emit(
                    mb,
                    MOp::Load {
                        off: off + layout::val::DATA as i32,
                        width: Width::U64,
                    },
                    vec![payload],
                    vec![base],
                );
                // Load the tag only if the type feedback did not already pin it.
                let tag = match kind_of(ty.set) {
                    Some(k) => Tag::Const(k),
                    None => {
                        let t = self.m.new_vreg(RegClass::Int);
                        self.emit(
                            mb,
                            MOp::Load {
                                off: off + layout::val::KIND as i32,
                                width: Width::U8,
                            },
                            vec![t],
                            vec![base],
                        );
                        Tag::Dyn(t)
                    }
                };
                self.set(results[0], Slot::Boxed { payload, tag });
            }

            // --- stack-pinned registers ---------------------------------------
            Op::StackGet(r) => self.load_lua_reg(mb, r, results[0]),
            Op::StackSet(r) => {
                let slot = self.slot(args[0]);
                self.store_lua_reg(mb, r, slot);
            }

            // --- arithmetic ----------------------------------------------------
            Op::IntArith(o) => {
                let alu = int_alu(o)?;
                let d = self.m.new_vreg(RegClass::Int);
                let uses = if matches!(o, IntOp::Neg | IntOp::BNot) {
                    vec![self.int(args[0])]
                } else {
                    vec![self.int(args[0]), self.int(args[1])]
                };
                self.emit(mb, MOp::Alu(alu), vec![d], uses);
                self.set(results[0], Slot::Int(d));
            }
            Op::FloatArith(o) => {
                let alu = float_alu(o)?;
                let d = self.m.new_vreg(RegClass::Float);
                let uses = if matches!(o, FloatOp::Neg) {
                    vec![self.float(args[0])]
                } else {
                    vec![self.float(args[0]), self.float(args[1])]
                };
                self.emit(mb, MOp::FAlu(alu), vec![d], uses);
                self.set(results[0], Slot::Float(d));
            }
            Op::ICmp(cc) => {
                let d = self.m.new_vreg(RegClass::Int);
                let uses = vec![self.int(args[0]), self.int(args[1])];
                self.emit(mb, MOp::ICmpSet(cc), vec![d], uses);
                self.set(results[0], Slot::Int(d));
            }
            Op::FCmp(cc) => {
                let d = self.m.new_vreg(RegClass::Int);
                let uses = vec![self.float(args[0]), self.float(args[1])];
                self.emit(mb, MOp::FCmpSet(cc), vec![d], uses);
                self.set(results[0], Slot::Int(d));
            }

            other => return Err(IselError::Unsupported(op_name(other))),
        }
        Ok(())
    }

    fn terminator(
        &mut self,
        mb: MBlock,
        b: Block,
        i: crate::jit::ir::Inst,
    ) -> Result<(), IselError> {
        let d = self.f.inst(i);
        let op = d.op;
        let args = d.args.clone();
        let targets = d.targets.clone();
        let exit = d.exit;

        match op {
            Op::Jump => {
                // One successor, so the predecessor has nowhere else to send
                // control: the copies can sit in this block.
                let t = &targets[0];
                let copies = self.edge_copies(t.block, &t.args);
                self.parallel_copy(mb, copies);
                let dst = self.bmap[t.block.index()];
                self.emit(mb, MOp::Jump(dst), vec![], vec![]);
            }
            Op::Br => {
                let cond = self.int(args[0]);
                // Two successors: this is a critical edge whenever the target has
                // more than one predecessor, and the copies have nowhere to live
                // but on the edge itself. Always giving each side its own block is
                // simpler than detecting criticality, and costs at most one branch
                // that block layout can later fold away.
                let then_ = self.edge_block(&targets[0]);
                let else_ = self.edge_block(&targets[1]);
                self.emit(mb, MOp::BrNz { then_, else_ }, vec![], vec![cond]);
            }
            Op::Ret => {
                // Results land at `base + 0..`, which the executor passes to
                // `frame_return` as `values_base`. Deliberately *not* the register
                // the bytecode `RETURN` names: compiled code never reaches
                // `op_return`, so that operand is the interpreter's business and
                // agreeing with the executor is the only constraint.
                for (n, &a) in args.iter().enumerate() {
                    let slot = self.slot(a);
                    self.store_lua_reg(mb, n as u8, slot);
                }
                // The results are already on the stack; these uses only keep them
                // alive to the return, so they are keepalives, not register
                // operands — a spilled one need not be reloaded for a `ret`.
                let keepalives: Vec<VReg> = args
                    .iter()
                    .flat_map(|&a| Self::regs_of(self.slot(a)))
                    .collect();
                self.m.push(
                    mb,
                    MInst::guard(
                        MOp::Ret {
                            nret: args.len() as u8,
                        },
                        vec![],
                        keepalives,
                    ),
                );
            }
            Op::Deopt => {
                let e = ExitId(exit.expect("Deopt carries an exit").0);
                self.stub(e, i);
                self.emit_exiting(mb, MOp::ExitTo(e), vec![], self.stub_regs(i), e);
            }
            other => return Err(IselError::Unsupported(op_name(other))),
        }
        let _ = b;
        Ok(())
    }

    /// A block holding just this edge's parameter copies, then a jump.
    fn edge_block(&mut self, call: &crate::jit::ir::BlockCall) -> MBlock {
        let eb = self.m.new_block();
        let copies = self.edge_copies(call.block, &call.args);
        self.parallel_copy(eb, copies);
        let dst = self.bmap[call.block.index()];
        self.emit(eb, MOp::Jump(dst), vec![], vec![]);
        eb
    }

    /// The moves that realize one edge: each argument into its parameter's
    /// register(s).
    fn edge_copies(&mut self, target: Block, args: &[Val]) -> Vec<(VReg, CopySrc)> {
        let params = self.f.block(target).params.clone();
        assert_eq!(params.len(), args.len(), "edge arity");

        let mut copies = Vec::new();
        for (&p, &a) in params.iter().zip(args) {
            match (self.slot(p), self.slot(a)) {
                (Slot::Int(dp), Slot::Int(sa)) | (Slot::Float(dp), Slot::Float(sa)) => {
                    copies.push((dp, CopySrc::Reg(sa)));
                }
                (
                    Slot::Boxed {
                        payload: dp,
                        tag: dt,
                    },
                    Slot::Boxed {
                        payload: sp,
                        tag: st,
                    },
                ) => {
                    copies.push((dp, CopySrc::Reg(sp)));
                    // A parameter whose set is polymorphic can receive an argument
                    // whose set is not — `Ty::accepts` allows exactly that — and
                    // then the tag the source knew statically has to become real.
                    match (dt, st) {
                        (Tag::Dyn(d), Tag::Dyn(s)) => copies.push((d, CopySrc::Reg(s))),
                        (Tag::Dyn(d), Tag::Const(k)) => {
                            copies.push((d, CopySrc::Imm(layout::kind(k) as i64)))
                        }
                        (Tag::Const(_), _) => {}
                    }
                }
                (dst, src) => panic!("edge type mismatch: {dst:?} <- {src:?}"),
            }
        }
        copies
    }

    /// Sequentialize a parallel move.
    ///
    /// The moves happen *simultaneously*, so a naive in-order emission clobbers:
    /// a loop that swaps two variables yields `p0 <- p1; p1 <- p0`, and doing the
    /// first move destroys the second's source. Emit every move whose destination
    /// nothing else still needs to read; when none qualifies, what remains is a
    /// permutation cycle, which one scratch register breaks.
    fn parallel_copy(&mut self, mb: MBlock, copies: Vec<(VReg, CopySrc)>) {
        let mut pending: Vec<(VReg, CopySrc)> = copies
            .into_iter()
            .filter(|&(d, s)| s != CopySrc::Reg(d))
            .collect();

        while !pending.is_empty() {
            let sources: Vec<VReg> = pending
                .iter()
                .filter_map(|&(_, s)| match s {
                    CopySrc::Reg(r) => Some(r),
                    CopySrc::Imm(_) => None,
                })
                .collect();

            let (ready, blocked): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|&(d, _)| !sources.contains(&d));

            if ready.is_empty() {
                // Every remaining destination is still someone's source: a cycle.
                // Park one value in a fresh register and the cycle opens up.
                let (d, _) = blocked[0];
                let tmp = self.m.new_vreg(self.m.class(d));
                self.emit(mb, MOp::Mov, vec![tmp], vec![d]);
                pending = blocked
                    .into_iter()
                    .map(|(dd, ss)| match ss {
                        CopySrc::Reg(r) if r == d => (dd, CopySrc::Reg(tmp)),
                        other => (dd, other),
                    })
                    .collect();
                continue;
            }

            for (d, s) in ready {
                match s {
                    CopySrc::Reg(r) => self.emit(mb, MOp::Mov, vec![d], vec![r]),
                    CopySrc::Imm(v) => self.emit(mb, MOp::Imm(v), vec![d], vec![]),
                };
            }
            pending = blocked;
        }
    }

    // --- exits --------------------------------------------------------------

    /// Fill in the exit stub for a guard: what the interpreter needs written back,
    /// and where it currently lives.
    fn stub(&mut self, e: ExitId, i: crate::jit::ir::Inst) {
        let fs = self
            .f
            .frame_state(self.f.inst(i).fs.expect("a guard carries a FrameState"));
        let pc = fs.pc;
        let mut slots = Vec::new();
        for (r, entry) in fs.regs.iter().enumerate() {
            let Some(v) = *entry else { continue };
            let src = match self.slot(v) {
                Slot::Boxed { payload, tag } => ExitSrc::Boxed { payload, tag },
                Slot::Int(vr) => ExitSrc::Int(vr),
                Slot::Float(vr) => ExitSrc::Float(vr),
            };
            slots.push((r as u8, src));
        }
        // `inst` is filled in by `emit_exiting`, once the guard that branches here
        // exists to be named.
        self.m.exits[e.0 as usize] = ExitStub {
            pc,
            slots,
            inst: usize::MAX,
        };
    }

    /// Every register the stub for this instruction's exit will read. These become
    /// *uses* of the guard, which is what stops the allocator reassigning them —
    /// and, afterwards, what tells the stub where they ended up.
    fn stub_regs(&self, i: crate::jit::ir::Inst) -> Vec<VReg> {
        let fs = self
            .f
            .frame_state(self.f.inst(i).fs.expect("a guard carries a FrameState"));

        // The frame base among them: every store the stub makes is relative to it,
        // so it is as much a value the stub reads as any of the frame's registers.
        // Listing it here rather than telling the allocator it is live everywhere
        // is the difference between a register the allocator may reuse after the
        // last exit and one it may never touch.
        let mut regs = vec![self.base];
        for entry in fs.regs.iter().flatten() {
            regs.extend(Self::regs_of(self.slot(*entry)));
        }
        regs.sort();
        regs.dedup();
        regs
    }

    // --- helpers ------------------------------------------------------------

    /// Load Lua register `r` of this frame into `v`'s slot.
    ///
    /// If `v` already has a slot — an entry parameter, whose register was fixed
    /// before any block was walked — the load targets it rather than minting a
    /// new one.
    fn load_lua_reg(&mut self, mb: MBlock, r: u8, v: Val) {
        let slot = match self.slots[v.index()] {
            Some(s) => s,
            None => {
                let s = self.fresh_slot(self.f.ty(v));
                self.set(v, s);
                s
            }
        };
        let Slot::Boxed { payload, tag } = slot else {
            panic!("a Lua stack slot holds a tagged Value, not {slot:?}");
        };

        let off = r as i32 * layout::val::SIZE as i32;
        let base = self.base;
        self.emit(
            mb,
            MOp::Load {
                off: off + layout::val::DATA as i32,
                width: Width::U64,
            },
            vec![payload],
            vec![base],
        );
        // A constant tag needs no load: the type is already known, either from
        // the entry context or from the IC feedback that typed this slot.
        if let Tag::Dyn(t) = tag {
            self.emit(
                mb,
                MOp::Load {
                    off: off + layout::val::KIND as i32,
                    width: Width::U8,
                },
                vec![t],
                vec![base],
            );
        }
    }

    /// Store a value into Lua register `r` of this frame, as a tagged `Value`.
    fn store_lua_reg(&mut self, mb: MBlock, r: u8, slot: Slot) {
        let off = r as i32 * layout::val::SIZE as i32;
        let base = self.base;

        let (payload, tag) = match slot {
            Slot::Boxed { payload, tag } => (payload, tag),
            Slot::Int(v) => (v, Tag::Const(ValueKind::Integer)),
            Slot::Float(v) => {
                let g = self.m.new_vreg(RegClass::Int);
                self.emit(mb, MOp::FloatToBits, vec![g], vec![v]);
                (g, Tag::Const(ValueKind::Float))
            }
        };

        self.emit(
            mb,
            MOp::Store {
                off: off + layout::val::DATA as i32,
                width: Width::U64,
            },
            vec![],
            vec![base, payload],
        );
        let tag_reg = match tag {
            Tag::Dyn(t) => t,
            Tag::Const(k) => self.emit_imm(mb, layout::kind(k) as i64),
        };
        self.emit(
            mb,
            MOp::Store {
                off: off + layout::val::KIND as i32,
                width: Width::U8,
            },
            vec![],
            vec![base, tag_reg],
        );
    }
}

/// The single `ValueKind` a type set implies, if it implies one.
///
/// `FALSE` and `TRUE` both mean `Boolean`, which is exactly why the set is finer
/// than the tag: the tag alone cannot distinguish them, so a guard narrowing to
/// one of them would need to test the payload as well. Neither arises yet.
fn kind_of(set: TypeSet) -> Option<ValueKind> {
    if !set.is_monomorphic() {
        return None;
    }
    Some(match set {
        TypeSet::NIL => ValueKind::Nil,
        TypeSet::FALSE | TypeSet::TRUE => ValueKind::Boolean,
        TypeSet::INT => ValueKind::Integer,
        TypeSet::FLOAT => ValueKind::Float,
        TypeSet::STR => ValueKind::String,
        TypeSet::TAB => ValueKind::Table,
        TypeSet::FUN => ValueKind::Function,
        TypeSet::THR => ValueKind::Thread,
        TypeSet::UDATA => ValueKind::Userdata,
        _ => unreachable!("monomorphic set with no kind: {set:?}"),
    })
}

fn int_alu(o: IntOp) -> Result<AluOp, IselError> {
    Ok(match o {
        IntOp::Add => AluOp::Add,
        IntOp::Sub => AluOp::Sub,
        IntOp::Mul => AluOp::Mul,
        IntOp::BAnd => AluOp::And,
        IntOp::BOr => AluOp::Or,
        IntOp::BXor => AluOp::Xor,
        IntOp::Neg => AluOp::Neg,
        IntOp::BNot => AluOp::Not,
        // Lua's shifts are not the machine's: the count is unmasked and a shift of
        // 64 or more yields zero, where aarch64 masks to 6 bits and yields the
        // operand. Needs a guard or a select; not yet.
        IntOp::Shl | IntOp::Shr => return Err(IselError::Unsupported("shift")),
        // The zero divisor is not our problem: lowering has already put a
        // `guard.cond` in front of these, because Lua raises on `x % 0`.
        IntOp::Mod => AluOp::Mod,
        IntOp::IDiv => AluOp::IDiv,
    })
}

fn float_alu(o: FloatOp) -> Result<FAluOp, IselError> {
    Ok(match o {
        FloatOp::Add => FAluOp::Add,
        FloatOp::Sub => FAluOp::Sub,
        FloatOp::Mul => FAluOp::Mul,
        FloatOp::Div => FAluOp::Div,
        FloatOp::Neg => FAluOp::Neg,
        FloatOp::IDiv | FloatOp::Mod | FloatOp::Pow => {
            return Err(IselError::Unsupported("float idiv/mod/pow"));
        }
    })
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::LuaArith(_) => "lua.arith",
        Op::LuaCmp(_) => "lua.cmp",
        Op::LuaEq => "lua.eq",
        Op::LuaConcat => "lua.concat",
        Op::LuaLen => "lua.len",
        Op::LuaGetIndex => "lua.getindex",
        Op::LuaSetIndex => "lua.setindex",
        Op::TabNew { .. } => "tab.new",
        Op::SlotSet(_) => "slot.set",
        Op::TabArr => "tab.arr",
        Op::TabArrLen => "tab.arrlen",
        Op::ArrGet => "arr.get",
        Op::ArrSet => "arr.set",
        Op::TabHashGet => "tab.hashget",
        Op::GcBarrierBack | Op::GcBarrierFwd => "gc.barrier",
        Op::UpvalCell(_) | Op::UpvalGet | Op::UpvalSet | Op::UpvalClose(_) => "upval",
        Op::Call { .. } => "call",
        Op::ClosureNew(_) => "closure.new",
        Op::GetGlobal(_) | Op::SetGlobal(_) => "global",
        Op::Safepoint => "safepoint",
        Op::FpToIntExact => "fp->int",
        Op::IsType(_) => "is.type",
        Op::IsFalsy => "is.falsy",
        _ => "op",
    }
}

/// Terminators, as the IR does not expose the predicate.
trait IsTerm {
    fn is_terminator(self) -> bool;
}

impl IsTerm for Op {
    fn is_terminator(self) -> bool {
        matches!(self, Op::Jump | Op::Br | Op::Ret | Op::Deopt)
    }
}
