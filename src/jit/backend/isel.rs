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
use crate::jit::ir::{Block, Def, Func, Val};

#[derive(Debug)]
pub enum IselError {
    /// An op the backend does not implement yet.
    Unsupported(&'static str),
    /// A guard whose target set is not a single Lua type, so it cannot become one
    /// tag compare.
    PolymorphicGuard(TypeSet),
    /// A loop with more than one entry. Lifetime interval construction cannot see
    /// liveness through one (see `order`), and Lua's structured loops should never
    /// produce one — so this is a decline rather than a case to support.
    IrreducibleLoop,
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

/// A compare fused into its branch, resolved to machine registers. `Imm` is the
/// folded form: one operand was a constant used nowhere else, so it rides in the
/// instruction instead of a register.
#[derive(Clone, Copy, Debug)]
enum FusedCmp {
    Reg(Cc, VReg, VReg),
    Imm(Cc, VReg, i64),
}

pub fn select(f: &Func<'_>) -> Result<MFunc, IselError> {
    select_with(f, false)
}

/// Select without destructing SSA: block parameters survive into the machine IR
/// as `MBlockData::params` and the edges carry `jump_args`, leaving the allocator
/// to resolve them.
///
/// The destructing path ([`select`]) is still the default while the allocator
/// grows the interval construction and edge resolution that this form needs. Two
/// paths rather than a fork: the difference is one flag, and keeping both lets the
/// same function be selected each way and the results compared.
pub fn select_ssa(f: &Func<'_>) -> Result<MFunc, IselError> {
    select_with(f, true)
}

fn select_with(f: &Func<'_>, ssa: bool) -> Result<MFunc, IselError> {
    let mut isel = Isel {
        f,
        m: MFunc::new(),
        slots: vec![None; f.num_values()],
        fused: Vec::new(),
        bmap: Vec::new(),
        base: VReg(0),
        ssa,
    };
    isel.run()?;
    // The CFG is complete only now: edge blocks are created as their branches are
    // selected. Everything downstream works in this linearization.
    isel.m
        .set_layout()
        .map_err(|_| IselError::IrreducibleLoop)?;
    Ok(isel.m)
}

struct Isel<'a, 'gc> {
    f: &'a Func<'gc>,
    m: MFunc,
    slots: Vec<Option<Slot>>,
    /// One entry per IR instruction: true for an `ICmp` whose sole consumer is a
    /// `Br`/`GuardCond` in the same block. Such a compare is not emitted on its
    /// own — its consumer fuses it into a flags-driven branch — so its result
    /// never gets a machine slot.
    fused: Vec<bool>,
    bmap: Vec<MBlock>,
    /// The Lua frame base pointer, `&stack[base]`. Every stack access is relative
    /// to this.
    base: VReg,
    /// Keep block parameters instead of lowering them into edge copies.
    ssa: bool,
}

impl<'a, 'gc> Isel<'a, 'gc> {
    fn run(&mut self) -> Result<(), IselError> {
        self.mark_fusions();

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

        // Under SSA the parameters stay parameters. The entry is deliberately
        // excluded: it has no incoming edge, and the loads just emitted *are* the
        // definitions of its parameters.
        if self.ssa {
            for b in self.f.blocks() {
                if b == self.f.entry {
                    continue;
                }
                let ps = self.machine_params(b);
                let mb = self.bmap[b.index()];
                self.m.blocks[mb.0 as usize].params = ps;
            }
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

    // --- compare/branch fusion ----------------------------------------------

    /// Mark every `ICmp` whose result feeds nothing but one `Br`/`GuardCond` in
    /// the same block. The comparison is then realized as flags at that branch
    /// rather than materialized into a register with `ICmpSet` — the compiler's
    /// job #1 (a value's machine location) answered as "the flags", which is
    /// exactly how a guard's own compare already lowers.
    ///
    /// Same-block is required: deferring the (pure) compare down to its consumer
    /// is only trivially sound when the consumer sits in the block the operands
    /// are already live in. Single-use covers the boolean not also being
    /// materialized, stored, or held by a frame state.
    /// Also fuses shift-count constants: a shift reads its count as an immediate
    /// (see [`Self::shift`]), so a count `iconst` referenced by nothing else is a
    /// dead materialization — the same single-use soundness argument as the
    /// compare fold, since the sole reference is the shift's count operand and no
    /// register or frame state still needs it.
    fn mark_fusions(&mut self) {
        self.fused = vec![false; self.f.num_insts()];

        let mut uses = vec![0u32; self.f.num_values()];
        for b in self.f.blocks() {
            for &i in &self.f.block(b).insts {
                let d = self.f.inst(i);
                for &a in &d.args {
                    uses[a.index()] += 1;
                }
                for t in &d.targets {
                    for &a in &t.args {
                        uses[a.index()] += 1;
                    }
                }
            }
        }
        for s in 0..self.f.num_frame_states() {
            let fs = self.f.frame_state(crate::jit::ir::FsRef(s as u32));
            for v in fs.regs.iter().flatten() {
                uses[v.index()] += 1;
            }
        }

        for b in self.f.blocks() {
            for &i in &self.f.block(b).insts {
                let d = self.f.inst(i);
                let cond = match d.op {
                    Op::Br | Op::GuardCond => d.args[0],
                    _ => continue,
                };
                if uses[cond.index()] != 1 {
                    continue;
                }
                let Def::Inst(di) = self.f.def(cond) else {
                    continue;
                };
                // The def must be in this same block. `di < i` and both in `b`
                // means the compare precedes the branch; the block list order
                // gives that, so a bare membership check suffices.
                if !matches!(self.f.inst(di).op, Op::ICmp(_))
                    || !self.f.block(b).insts.contains(&di)
                {
                    continue;
                }
                self.fused[di.index()] = true;

                // Fold a compare operand to an immediate only when it is a
                // constant used nowhere else: its defining `iconst` is then
                // skipped entirely, turning a `mov`+register compare into one
                // immediate compare. RHS preferred; a LHS fold swaps the
                // condition at the branch (see `fused_cmp_of`). At most one side.
                let cmp = self.f.inst(di);
                for &operand in &[cmp.args[1], cmp.args[0]] {
                    if uses[operand.index()] != 1 {
                        continue;
                    }
                    let Def::Inst(ki) = self.f.def(operand) else {
                        continue;
                    };
                    if matches!(self.f.inst(ki).op, Op::IConst(_)) {
                        self.fused[ki.index()] = true;
                        break;
                    }
                }
            }
        }

        for b in self.f.blocks() {
            for &i in &self.f.block(b).insts {
                let d = self.f.inst(i);
                if !matches!(d.op, Op::IntArith(IntOp::Shl | IntOp::Shr)) {
                    continue;
                }
                let count = d.args[1];
                if uses[count.index()] != 1 {
                    continue;
                }
                let Def::Inst(ki) = self.f.def(count) else {
                    continue;
                };
                if matches!(self.f.inst(ki).op, Op::IConst(_)) {
                    self.fused[ki.index()] = true;
                }
            }
        }
    }

    /// The constant an operand folds to, if its defining `iconst` was marked
    /// skipped by [`Self::mark_fusions`] — i.e. it is a compare operand
    /// eligible to ride in the instruction instead of a register.
    fn folded(&self, v: Val) -> Option<i64> {
        let Def::Inst(di) = self.f.def(v) else {
            return None;
        };
        if !self.fused[di.index()] {
            return None;
        }
        match self.f.inst(di).op {
            Op::IConst(n) => Some(n),
            _ => None,
        }
    }

    /// If `cond` is the result of a fused compare, its condition and operands,
    /// resolved to machine locations — a register pair, or a register against a
    /// folded immediate. The operands' slots are still valid: they were selected
    /// before the skipped `ICmp`, hence before this branch.
    fn fused_cmp_of(&self, cond: Val) -> Option<FusedCmp> {
        let Def::Inst(di) = self.f.def(cond) else {
            return None;
        };
        if !self.fused[di.index()] {
            return None;
        }
        let d = self.f.inst(di);
        let Op::ICmp(cc) = d.op else { return None };
        let (a, b) = (d.args[0], d.args[1]);
        if let Some(n) = self.folded(b) {
            Some(FusedCmp::Imm(cc, self.int(a), n))
        } else if let Some(n) = self.folded(a) {
            Some(FusedCmp::Imm(cc.swapped(), self.int(b), n))
        } else {
            Some(FusedCmp::Reg(cc, self.int(a), self.int(b)))
        }
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

    /// The compile-time integer an operand holds, if it is a plain `iconst`.
    fn int_const(&self, v: Val) -> Option<i64> {
        let Def::Inst(di) = self.f.def(v) else {
            return None;
        };
        match self.f.inst(di).op {
            Op::IConst(n) => Some(n),
            _ => None,
        }
    }

    /// Lower a Lua shift (`args[0] <dir> args[1]`) whose count is a compile-time
    /// constant, folding the language's edge rules into a single machine
    /// shift-by-immediate — or a constant zero. A dynamic count would need the
    /// full branchy `luaV_shiftl` expansion (compare against 0 and 64, pick a
    /// direction, select), which we don't emit yet, so it is declined.
    ///
    /// Because the count is known here, the machine shift never sees an
    /// out-of-range amount, so the target's count-masking (aarch64's low 6 bits,
    /// x86's `& 63`) never bites — we have already turned every such case into a 0.
    fn shift(
        &mut self,
        mb: MBlock,
        dir: IntOp,
        args: &[Val],
        result: Val,
    ) -> Result<(), IselError> {
        let count = self
            .int_const(args[1])
            .ok_or(IselError::Unsupported("dynamic shift count"))?;

        // |count| >= 64 shifts every bit out; Lua defines the result as 0.
        if !(-63..=63).contains(&count) {
            let d = self.emit_imm(mb, 0);
            self.set(result, Slot::Int(d));
            return Ok(());
        }

        // A negative count reverses the direction; `n` is then a real bit count
        // in `1..=63` (`-count` cannot overflow, count is bounded above).
        let want_left = matches!(dir, IntOp::Shl);
        let (left, n) = if count >= 0 {
            (want_left, count)
        } else {
            (!want_left, -count)
        };
        if n == 0 {
            // Shift by zero is the identity; forward the source unchanged.
            self.set(result, self.slot(args[0]));
            return Ok(());
        }

        // Lua's shifts are logical in both directions, so a right shift is `Lsr`,
        // never the sign-propagating `Sar`.
        let alu = if left { AluOp::Shl } else { AluOp::Lsr };
        let x = self.int(args[0]);
        let d = self.m.new_vreg(RegClass::Int);
        self.emit(mb, MOp::AluImm(alu, n), vec![d], vec![x]);
        self.set(result, Slot::Int(d));
        Ok(())
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
        // Folded into a consumer — a compare fused into its branch, or the
        // constant folded into that compare's immediate. Emits nothing and
        // defines no slot; the branch reads it through `fused_cmp_of`.
        if self.fused[i.index()] {
            return Ok(());
        }

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
                // `GuardCmp` exits unless `a cc b`; `GuardCond` exits unless its
                // condition holds — the same thing when the condition is `a cc b`.
                match self.fused_cmp_of(args[0]) {
                    Some(FusedCmp::Reg(cc, a, b)) => self.emit_exiting(
                        mb,
                        MOp::GuardCmp { cc, exit: e },
                        vec![a, b],
                        self.stub_regs(i),
                        e,
                    ),
                    Some(FusedCmp::Imm(cc, a, imm)) => self.emit_exiting(
                        mb,
                        MOp::GuardCmpImm { cc, imm, exit: e },
                        vec![a],
                        self.stub_regs(i),
                        e,
                    ),
                    None => {
                        let c = self.int(args[0]);
                        self.emit_exiting(
                            mb,
                            MOp::GuardNz { exit: e },
                            vec![c],
                            self.stub_regs(i),
                            e,
                        );
                    }
                }
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
            // Shifts are not an ordinary two-register ALU op: Lua's semantics
            // (logical fill, negative count reverses direction, |count| >= 64
            // yields zero) diverge from every machine shift, so they get their own
            // lowering that folds those rules against a constant count.
            Op::IntArith(o @ (IntOp::Shl | IntOp::Shr)) => {
                self.shift(mb, o, &args, results[0])?;
            }
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
                // A compare consumed only by a branch/guard was marked fused and
                // never reaches here — see `mark_fusions` and the early
                // return above. What remains materializes a boolean.
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
                let t = targets[0].clone();
                let dst = self.bmap[t.block.index()];
                if self.ssa {
                    let a = self.edge_args(mb, t.block, &t.args);
                    self.m.blocks[mb.0 as usize].jump_args = a;
                } else {
                    let copies = self.edge_copies(t.block, &t.args);
                    self.parallel_copy(mb, copies);
                }
                self.emit(mb, MOp::Jump(dst), vec![], vec![]);
            }
            Op::Br => {
                // Two successors: this is a critical edge whenever the target has
                // more than one predecessor, and the copies have nowhere to live
                // but on the edge itself. Always giving each side its own block is
                // simpler than detecting criticality, and costs at most one branch
                // that block layout can later fold away. The parameter copies go in
                // those edge blocks, so nothing lands between a fused compare and
                // its branch.
                let then_ = self.edge_block(&targets[0]);
                let else_ = self.edge_block(&targets[1]);
                match self.fused_cmp_of(args[0]) {
                    Some(FusedCmp::Reg(cc, a, b)) => {
                        self.emit(mb, MOp::BrCmp { cc, then_, else_ }, vec![], vec![a, b]);
                    }
                    Some(FusedCmp::Imm(cc, a, imm)) => {
                        self.emit(
                            mb,
                            MOp::BrCmpImm {
                                cc,
                                imm,
                                then_,
                                else_,
                            },
                            vec![],
                            vec![a],
                        );
                    }
                    None => {
                        let cond = self.int(args[0]);
                        self.emit(mb, MOp::BrNz { then_, else_ }, vec![], vec![cond]);
                    }
                }
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
    ///
    /// Under `ssa` it holds only the jump and the edge's arguments — and is then
    /// empty whenever the allocator manages to give an argument and its parameter
    /// the same register, at which point block placement can fold it away.
    fn edge_block(&mut self, call: &crate::jit::ir::BlockCall) -> MBlock {
        let eb = self.m.new_block();
        let dst = self.bmap[call.block.index()];
        if self.ssa {
            let a = self.edge_args(eb, call.block, &call.args);
            self.m.blocks[eb.0 as usize].jump_args = a;
        } else {
            let copies = self.edge_copies(call.block, &call.args);
            self.parallel_copy(eb, copies);
        }
        self.emit(eb, MOp::Jump(dst), vec![], vec![]);
        eb
    }

    /// A block's parameters as machine registers, in the order an edge must supply
    /// them. One IR parameter yields two registers when its tag is dynamic.
    fn machine_params(&self, target: Block) -> Vec<VReg> {
        let mut out = Vec::new();
        for &p in &self.f.block(target).params {
            match self.slot(p) {
                Slot::Int(v) | Slot::Float(v) => out.push(v),
                Slot::Boxed { payload, tag } => {
                    out.push(payload);
                    if let Tag::Dyn(t) = tag {
                        out.push(t);
                    }
                }
            }
        }
        out
    }

    /// The registers one edge supplies, positionally matching [`Self::machine_params`]
    /// of the target.
    fn edge_args(&mut self, mb: MBlock, target: Block, args: &[Val]) -> Vec<VReg> {
        let params = self.f.block(target).params.clone();
        assert_eq!(params.len(), args.len(), "edge arity");

        let mut out = Vec::new();
        for (&p, &a) in params.iter().zip(args) {
            match (self.slot(p), self.slot(a)) {
                (Slot::Int(_), Slot::Int(sa)) | (Slot::Float(_), Slot::Float(sa)) => out.push(sa),
                (
                    Slot::Boxed { tag: dt, .. },
                    Slot::Boxed {
                        payload: sp,
                        tag: st,
                    },
                ) => {
                    out.push(sp);
                    match (dt, st) {
                        (Tag::Dyn(_), Tag::Dyn(s)) => out.push(s),
                        (Tag::Dyn(_), Tag::Const(k)) => {
                            // The parameter's set is polymorphic but this argument
                            // knew its tag statically. An edge carries registers
                            // only, so materialize it here rather than teaching the
                            // allocator about constant arguments — `Imm` is
                            // rematerializable, so the reload cost comes back.
                            let t = self.m.new_vreg(RegClass::Int);
                            self.emit(mb, MOp::Imm(layout::kind(k) as i64), vec![t], vec![]);
                            out.push(t);
                        }
                        (Tag::Const(_), _) => {}
                    }
                }
                (dst, src) => panic!("edge type mismatch: {dst:?} <- {src:?}"),
            }
        }
        out
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
        // Shifts never reach here: `inst` intercepts `IntArith(Shl | Shr)` and
        // lowers them in `shift`, which folds Lua's count semantics against a
        // constant. A dynamic count is declined there, not here.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lua;
    use crate::jit::backend::regalloc::RegallocFunc;
    use crate::jit::frontend::lower::lower;

    const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

    /// `is_prime` from test-files/primes.lua, selected both ways — a loop with two
    /// diamonds, which is what makes it worth comparing.
    fn is_prime_both() -> (MFunc, MFunc) {
        let source = std::fs::read_to_string("test-files/primes.lua").unwrap();
        let mut lua = Lua::new();
        lua.load_all();
        lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("primes")).expect("compile");
            let closure = chunk.as_lua().expect("chunk is a Lua closure");
            let proto = closure.proto.prototypes[0];
            let func = lower(proto, 0, vec![INT]).expect("lower is_prime");
            (
                select(&func).expect("isel"),
                select_ssa(&func).expect("isel ssa"),
            )
        })
    }

    /// Every edge supplies exactly the registers its target declares, of matching
    /// class. This is the invariant edge resolution will rely on, so it is worth
    /// pinning at the source rather than discovering downstream.
    #[test]
    fn ssa_edges_match_their_target_parameters() {
        let (_, m) = is_prime_both();
        assert!(m.has_block_params(), "is_prime has a loop-carried value");

        for b in m.block_order() {
            let args = m.jump_args(b);
            if !args.is_empty() {
                let succs = m.succs(b);
                assert_eq!(succs.len(), 1, "only a Jump may carry arguments");
                let params = m.block_params(succs[0]);
                assert_eq!(args.len(), params.len(), "arity on mb{}", b.0);
                for (a, p) in args.iter().zip(params) {
                    assert_eq!(m.class(*a), m.class(*p), "class mismatch on edge");
                }
            }
            // ...and a block with parameters is supplied by *every* predecessor.
            for s in m.succs(b) {
                if !m.block_params(s).is_empty() {
                    assert_eq!(
                        m.jump_args(b).len(),
                        m.block_params(s).len(),
                        "mb{} -> mb{} supplies nothing",
                        b.0,
                        s.0
                    );
                }
            }
        }
    }

    /// The point of the exercise: the copies isel used to sequentialize are gone.
    #[test]
    fn ssa_selection_emits_no_edge_movs() {
        let (destructed, ssa) = is_prime_both();
        let movs = |m: &MFunc| {
            (0..m.num_insts())
                .filter(|&i| matches!(m.inst(i).op, MOp::Mov))
                .count()
        };
        assert!(movs(&destructed) > 0, "destructing path emits edge copies");
        assert_eq!(movs(&ssa), 0, "SSA form has parameters, not copies");
    }

    /// The entry block's parameters are the region's live-in Lua registers, loaded
    /// from the stack by the prologue. It has no incoming edge, so they must not be
    /// modelled as block parameters — nothing would ever supply them.
    #[test]
    fn ssa_entry_block_has_no_parameters() {
        let (_, m) = is_prime_both();
        assert!(m.block_params(m.entry()).is_empty());
    }

    /// The SSA form allocates and verifies end to end: interval construction sees
    /// the parameters, and edge resolution delivers each argument to where its
    /// parameter is read from. `verify` is the real assertion here — it is
    /// symbolic and checks every location against the value it should hold.
    #[test]
    fn ssa_form_allocates_and_verifies() {
        use crate::jit::backend::regalloc::{allocate, verify};
        use crate::jit::backend::target::{annotate, machine_env};

        let (_, mut ssa) = is_prime_both();
        annotate(&mut ssa);
        let ra = allocate(&ssa, &machine_env()).expect("SSA form allocates");
        verify(&ssa, &ra).expect("SSA allocation must check out");
    }

    /// Both selections of the same function must allocate to code that verifies.
    /// The point is that dropping SSA destruction changed no invariant the checker
    /// enforces — only who performs the deconstruction, and when.
    #[test]
    fn both_selections_verify() {
        use crate::jit::backend::regalloc::{allocate, verify};
        use crate::jit::backend::target::{annotate, machine_env};

        let (mut destructed, mut ssa) = is_prime_both();
        for m in [&mut destructed, &mut ssa] {
            annotate(m);
            let ra = allocate(m, &machine_env()).expect("allocates");
            verify(m, &ra).expect("verifies");
        }
    }
}
