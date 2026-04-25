use std::collections::HashMap;

use cstree::interning::TokenInterner;

use super::defs::{Chunk, ExprDesc, ExprKind, JumpList, RegisterIndex};
use super::{CompileError, CompileErrorKind, LineNumber};
use crate::dmm::{Gc, Mutation};
use crate::env::{LuaString, Prototype, Value};
use crate::instruction::{Instruction, UpValueDescriptor};
use crate::parser::syntax::{
    Assign, BinaryOp, BinaryOperator, Break, Decl, DeclModifier, Do, Expr, ForGen, ForNum, Func,
    FuncCall, FuncExpr, Goto, Ident, If, Index, Label, Literal, LiteralValue, MethodCall, PrefixOp,
    PrefixOperator, Repeat, Return, Root, Stmt, Table, TableEntry, While,
};
use std::mem;

/// Sentinel value used as the `dst` field of a `TESTSET` while the real
/// destination register is still unknown — i.e. the jump is pending in a
/// jump list and its ultimate consumer (a value-context discharge or a
/// branch-context patch) hasn't been reached yet.
///
/// `patch_list_aux` rewrites `NO_REG` to the concrete destination register
/// when discharging to a register; `downgrade_testsets` rewrites the whole
/// `TESTSET` to a plain `TEST` when the list is consumed in branch context
/// (no value preservation needed). Every `TESTSET` with this sentinel must
/// be patched or downgraded before the chunk is assembled — executing one
/// at runtime would write to an out-of-bounds register.
const NO_REG: u8 = u8::MAX;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn ice(msg: &'static str) -> CompileError {
    CompileError::internal(msg)
}

fn err(kind: CompileErrorKind, line: LineNumber) -> CompileError {
    CompileError {
        kind,
        line_number: line,
    }
}

// ---------------------------------------------------------------------------
// Variable tracking
// ---------------------------------------------------------------------------

struct VariableData {
    register: RegisterIndex,
    is_const: bool,
}

/// Snapshot of the register allocator's state at scope entry. Restored
/// verbatim on `pop_scope` so nested scopes cleanly release any temps and
/// named locals they allocated.
#[derive(Clone, Copy)]
struct ScopeMark {
    freereg: u8,
    nactvar: u8,
}

// ---------------------------------------------------------------------------
// Compilation context
// ---------------------------------------------------------------------------

/// Implemented by a function's compilation context so a nested function
/// can ask what `UpValueDescriptor` it should use to reference a free
/// variable from this scope. The parent captures from its own parent
/// on demand as the lookup cascades upward.
trait UpvalueResolver {
    fn resolve_for_child(&mut self, name: &str) -> Option<UpValueDescriptor>;
}

struct Ctx<'gc, 'a> {
    interner: &'a TokenInterner,
    mc: &'a Mutation<'gc>,
    chunk: Chunk<'gc>,

    /// Stack of break-target labels for nested loops.
    control_end_label: Vec<u16>,

    /// Lexical scope stack: each frame maps variable names to register data.
    scope: Vec<HashMap<String, VariableData>>,
    /// Saved `(freereg, nactvar)` per scope entry, restored on pop so that
    /// any temps or locals allocated within the scope are reclaimed together.
    scope_marks: Vec<ScopeMark>,
    /// Registers that need CLOSE when scope is popped (to-be-closed variables).
    scope_close: Vec<Vec<RegisterIndex>>,

    /// Named labels for goto/label statements (name → label index).
    goto_labels: HashMap<String, u16>,

    /// Parent function's resolver, or `None` for the main chunk. A nested
    /// function calls this to walk the lexical chain when it encounters a
    /// free variable.
    capture: Option<&'a mut dyn UpvalueResolver>,
    /// This function's upvalue list, with names retained so children can
    /// look entries up by name. Flattened to
    /// `Chunk::upvalue_desc: Box<[UpValueDescriptor]>` at assembly time.
    upvalues: Vec<(String, UpValueDescriptor)>,
}

impl<'gc, 'a> Ctx<'gc, 'a> {
    fn emit(&mut self, instruction: Instruction) {
        self.chunk.tape.push(instruction);
    }

    /// Reserve a single fresh temp register at `freereg` and return it.
    /// Replaces the older `alloc_register` name to match Lua's
    /// `luaK_reserveregs(fs, 1)`.
    fn reserve_reg(&mut self) -> Result<RegisterIndex, CompileError> {
        assert!(self.chunk.nactvar <= self.chunk.freereg);
        if self.chunk.freereg == 255 {
            // 255 is the last addressable slot; allocating a new one would
            // wrap freereg to 0 and silently corrupt downstream allocations.
            return Err(err(CompileErrorKind::Registers, LineNumber(0)));
        }
        let reg = self.chunk.freereg;
        self.chunk.freereg += 1;
        if self.chunk.freereg > self.chunk.max_stack {
            self.chunk.max_stack = self.chunk.freereg;
        }
        Ok(RegisterIndex(reg))
    }

    /// Reserve `n` consecutive temp registers starting at `freereg`.
    /// Returns the base register.
    fn reserve_regs(&mut self, n: u8) -> Result<RegisterIndex, CompileError> {
        assert!(self.chunk.nactvar <= self.chunk.freereg);
        let base = self.chunk.freereg;
        // `checked_add` returning Some guarantees the result fits in u8
        // (<= 255), so no further range check is needed.
        let end = base
            .checked_add(n)
            .ok_or_else(|| err(CompileErrorKind::Registers, LineNumber(0)))?;
        self.chunk.freereg = end;
        if end > self.chunk.max_stack {
            self.chunk.max_stack = end;
        }
        Ok(RegisterIndex(base))
    }

    /// Back-compat alias kept so existing call sites compile unchanged
    /// during the phased refactor. Phase 5 migrates remaining uses to
    /// `reserve_reg`/`reserve_regs`.
    fn alloc_register(&mut self) -> Result<RegisterIndex, CompileError> {
        self.reserve_reg()
    }

    /// Free register `reg` iff it's a temp (`reg >= nactvar`). Constants
    /// and locals are no-ops. Enforces LIFO stack discipline in debug:
    /// the register being freed must be the most recently reserved
    /// (`reg == freereg - 1`).
    #[allow(dead_code)]
    fn free_reg(&mut self, reg: RegisterIndex) {
        if reg.0 >= self.chunk.nactvar {
            assert!(
                reg.0 + 1 == self.chunk.freereg,
                "free_reg out of order: reg {} but freereg {}",
                reg.0,
                self.chunk.freereg,
            );
            self.chunk.freereg -= 1;
        }
    }

    /// Free the register backing `expr` if it's a Reg-kind expression whose
    /// register is a temp. No-op otherwise.
    #[allow(dead_code)]
    fn free_exp(&mut self, expr: &ExprDesc) {
        if let ExprKind::Reg(reg) = expr.kind {
            self.free_reg(reg);
        }
    }

    /// Free two registers in correct (higher-first) order — matches
    /// Lua 5.4's `freeregs(fs, r1, r2)`.
    #[allow(dead_code)]
    fn free_regs(&mut self, r1: RegisterIndex, r2: RegisterIndex) {
        if r1.0 > r2.0 {
            self.free_reg(r1);
            self.free_reg(r2);
        } else {
            self.free_reg(r2);
            self.free_reg(r1);
        }
    }

    /// Free two ExprDescs in correct order — skips non-Reg-kind inputs.
    #[allow(dead_code)]
    fn free_exps(&mut self, e1: &ExprDesc, e2: &ExprDesc) {
        match (e1.kind, e2.kind) {
            (ExprKind::Reg(r1), ExprKind::Reg(r2)) => self.free_regs(r1, r2),
            (ExprKind::Reg(r), _) | (_, ExprKind::Reg(r)) => self.free_reg(r),
            _ => {}
        }
    }

    /// Promote the top `n` temp slots to named-local status by advancing
    /// `nactvar`. Called after a `local`-decl's initialiser has written
    /// values into what were temps and the targets are about to be bound
    /// by name. Must match a prior reserve; the caller's design ensures
    /// `freereg >= nactvar + n` at call time.
    fn adjust_locals(&mut self, n: u8) {
        assert!(self.chunk.nactvar as usize + n as usize <= self.chunk.freereg as usize);
        self.chunk.nactvar += n;
    }

    /// Use a destination hint register if provided, otherwise reserve a
    /// fresh one. When the hint is beyond the current cursor, `freereg` is
    /// bumped so later allocations sit above it.
    fn dst_or_alloc(&mut self, dst: Option<RegisterIndex>) -> Result<RegisterIndex, CompileError> {
        match dst {
            Some(reg) => {
                if self.chunk.freereg <= reg.0 {
                    self.chunk.freereg = reg.0 + 1;
                    if self.chunk.freereg > self.chunk.max_stack {
                        self.chunk.max_stack = self.chunk.freereg;
                    }
                }
                Ok(reg)
            }
            None => self.reserve_reg(),
        }
    }

    fn next_offset(&self) -> usize {
        self.chunk.tape.len()
    }

    fn new_label(&mut self) -> u16 {
        let idx = self.chunk.labels.len();
        self.chunk.labels.push(0);
        idx as u16
    }

    fn set_label(&mut self, label: u16, offset: usize) {
        self.chunk.labels[label as usize] = offset;
    }

    fn emit_jump(&mut self, label: u16) {
        let idx = self.next_offset();
        self.chunk.jump_patches.push((idx, label));
        self.emit(Instruction::JMP { offset: 0 });
    }

    fn emit_jump_instr(&mut self, label: u16, instr: Instruction) {
        let idx = self.next_offset();
        self.chunk.jump_patches.push((idx, label));
        self.emit(instr);
    }

    // ---------------------------------------------------------------------
    // Jump-list primitives (Lua 5.4-style true/false list patching)
    // ---------------------------------------------------------------------

    /// Emit a `JMP` with a placeholder offset and return its tape index so
    /// the caller can thread it through a `JumpList`.
    fn emit_unfilled_jmp(&mut self) -> usize {
        let idx = self.next_offset();
        self.emit(Instruction::JMP { offset: 0 });
        idx
    }

    /// Debug-only check of the `JumpList` predecessor invariant: for any
    /// `jmp_idx` stored in a list, `tape[jmp_idx - 1]` must be a CMP/TEST/
    /// TESTSET control instruction (or `jmp_idx == 0`, which every consumer
    /// shortcuts). See the `JumpList` doc comment.
    fn assert_ctrl_predecessor(&self, jmp_idx: usize) {
        if jmp_idx == 0 {
            return;
        }
        let prev = &self.chunk.tape[jmp_idx - 1];
        assert!(
            matches!(
                prev,
                Instruction::EQ { .. }
                    | Instruction::LT { .. }
                    | Instruction::LE { .. }
                    | Instruction::TEST { .. }
                    | Instruction::TESTSET { .. }
            ),
            "JumpList invariant violated: tape[{}] = {:?} is not a \
             CMP/TEST/TESTSET control instruction for JMP at tape[{}]",
            jmp_idx - 1,
            prev,
            jmp_idx,
        );
    }

    /// Downgrade any `TESTSET` with a `NO_REG` dst preceding a jump in the
    /// list to a plain `TEST`. Called when a list is about to be consumed
    /// in a context that doesn't need value preservation — mid-expression
    /// fall-through patches and branch-context patches both qualify.
    ///
    /// Mid-expression patching is safe to downgrade because the jumps being
    /// patched correspond to paths whose values are discarded by the
    /// enclosing logical operator. For example, in `a and b`, LHS's
    /// `true_list` gets patched to RHS's start — LHS's truthy value is
    /// not the result (RHS's is), so preservation would be wasted.
    ///
    /// Reads `tape[jmp_idx - 1]`; see the `JumpList` invariant.
    fn downgrade_testsets(&mut self, list: &JumpList) {
        for &jmp_idx in &list.jumps {
            if jmp_idx == 0 {
                continue;
            }

            self.assert_ctrl_predecessor(jmp_idx);
            if let Instruction::TESTSET { dst, src, inverted } = self.chunk.tape[jmp_idx - 1]
                && dst == NO_REG
            {
                // TEST skips on `truthy != inverted`; TESTSET skips on
                // `truthy == inverted`. They're inverses, so preserving
                // the same skip behaviour across the rewrite requires
                // flipping the flag.
                self.chunk.tape[jmp_idx - 1] = Instruction::TEST {
                    src,
                    inverted: !inverted,
                };
            }
        }
    }

    /// Patch every jump in `list` to `target`. Downgrades any `TESTSET`
    /// controls that still hold `NO_REG` as their dst — this path is for
    /// branch/fall-through consumers that don't materialise a value.
    fn patch_to(&mut self, list: JumpList, target: usize) {
        self.downgrade_testsets(&list);
        for idx in list.jumps {
            let offset = target as i32 - (idx as i32 + 1);
            match &mut self.chunk.tape[idx] {
                Instruction::JMP { offset: o } => *o = offset,
                _ => panic!("jump-list entry is not a JMP"),
            }
        }
    }

    /// Patch every jump in `list` to the current tape position (the next
    /// instruction to be emitted).
    fn patch_to_here(&mut self, list: JumpList) {
        let target = self.next_offset();
        self.patch_to(list, target);
    }

    /// Does this list contain any jump whose control instruction can't
    /// self-materialise a boolean value? TESTSET-controlled jumps preserve
    /// the operand value on the taken edge; TEST/LT/LE/EQ jumps don't. The
    /// caller uses this to decide whether the `LFALSESKIP` / `LOAD true`
    /// fixup tail is needed at discharge.
    ///
    /// Reads `tape[idx - 1]`; see the `JumpList` invariant.
    fn need_value(&self, list: &JumpList) -> bool {
        list.jumps.iter().any(|&idx| {
            if idx == 0 {
                return true;
            }

            self.assert_ctrl_predecessor(idx);
            !matches!(
                self.chunk.tape[idx - 1],
                Instruction::TESTSET { dst: NO_REG, .. }
            )
        })
    }

    /// Value-context patching of a jump list. For each jump, if its control
    /// is a `TESTSET { dst: NO_REG, .. }`, patch the dst to `reg` and aim
    /// the JMP at `vtarget` (the final/merge point — the TESTSET preserves
    /// value on the taken edge so we don't need the fixup tail).
    /// Self-assigning TESTSETs (src == reg) are downgraded to TEST and
    /// aimed at `dtarget` along with every non-TESTSET-controlled jump.
    ///
    /// Reads `tape[jmp_idx - 1]`; see the `JumpList` invariant.
    fn patch_list_aux(
        &mut self,
        list: JumpList,
        vtarget: usize,
        reg: RegisterIndex,
        dtarget: usize,
    ) {
        for jmp_idx in list.jumps {
            let target = if jmp_idx == 0 {
                dtarget
            } else {
                self.assert_ctrl_predecessor(jmp_idx);
                let ctrl_idx = jmp_idx - 1;
                match self.chunk.tape[ctrl_idx] {
                    Instruction::TESTSET {
                        dst: NO_REG,
                        src,
                        inverted,
                    } => {
                        // Both arms preserve the operand value at `reg`:
                        // the assignment case writes `src` → `reg`, the
                        // self-assign case has the value already in `reg`.
                        // Either way the jump skips the materialisation
                        // tail and lands on `vtarget`.
                        if src == reg.0 {
                            self.chunk.tape[ctrl_idx] = Instruction::TEST {
                                src,
                                inverted: !inverted,
                            };
                        } else {
                            self.chunk.tape[ctrl_idx] = Instruction::TESTSET {
                                dst: reg.0,
                                src,
                                inverted,
                            };
                        }
                        vtarget
                    }
                    _ => dtarget,
                }
            };
            let offset = target as i32 - (jmp_idx as i32 + 1);
            if let Instruction::JMP { offset: o } = &mut self.chunk.tape[jmp_idx] {
                *o = offset;
            }
        }
    }

    /// Emit a TESTSET + unfilled JMP that fires when the value at `src`
    /// matches `jump_if_truthy`, assigning the TESTSET's dst to the source
    /// value on the same edge. The `dst` field is left as `NO_REG`: if this
    /// jump is ultimately consumed by a value-context discharge,
    /// `patch_list_aux` rewrites the dst to the final destination register;
    /// otherwise `downgrade_testsets` rewrites the TESTSET to a plain TEST.
    ///
    /// Why TESTSET instead of TEST: for `a or b` in value context we need
    /// the truthy short-circuit to preserve `a`'s value in the destination
    /// register. The TESTSET does that assign on the same path as the JMP,
    /// so no extra MOVE is needed.
    fn emit_test_jump(&mut self, src: RegisterIndex, jump_if_truthy: bool) -> usize {
        // TESTSET semantics: skip iff `truthy(src) == inverted`, else
        // `R[dst] := R[src]` and fall through. We want the fall-through
        // path (which leads to the JMP) to be the "wanted truthiness" path.
        //   jump_if_truthy=true  → fall-through on truthy → assign on truthy
        //                          → `truthy != inverted` → inverted = false
        //   jump_if_truthy=false → fall-through on falsy  → inverted = true
        self.emit(Instruction::TESTSET {
            dst: NO_REG,
            src: src.0,
            inverted: !jump_if_truthy,
        });
        self.emit_unfilled_jmp()
    }

    /// Flip the `inverted` flag of the CMP/TEST/TESTSET preceding `jmp_idx`.
    /// Shared helper for `negate_cond` and the pending-flip done by `not`.
    fn flip_control_polarity(&mut self, jmp_idx: usize) {
        // Defensive: a JMP at tape index 0 has no predecessor to invert.
        // Doesn't arise in practice (functions start with VARARGPREP) but
        // cheap to guard against underflow.
        if jmp_idx == 0 {
            return;
        }
        match &mut self.chunk.tape[jmp_idx - 1] {
            Instruction::LT { inverted, .. }
            | Instruction::LE { inverted, .. }
            | Instruction::EQ { inverted, .. }
            | Instruction::TEST { inverted, .. }
            | Instruction::TESTSET { inverted, .. } => {
                *inverted = !*inverted;
            }
            other => unreachable!(
                "flip_control_polarity: jump at {jmp_idx} has no \
                 CMP/TEST control instruction (found {other:?})"
            ),
        }
    }

    /// Flip the polarity of every pending conditional jump in `expr`: invert
    /// the `inverted` flag of each control instruction (CMP/TEST/TESTSET)
    /// and swap the true/false lists. After the call every pending JMP
    /// fires on the opposite runtime condition and the lists are re-labelled
    /// to match — the whole `ExprDesc` stays internally consistent.
    ///
    /// **Precondition:** the Jump-kind pending head (if the expression is
    /// Jump-kind) must have been absorbed beforehand, and every jump in
    /// `true_list` ∪ `false_list` must fire on the same runtime polarity
    /// (i.e. one of the two lists is empty). `goiftrue` / `goiffalse`
    /// enforce both conditions. In mixed-polarity states (lists populated
    /// from both short-circuit edges of an `and`/`or`) flipping every
    /// entry would miscompile composed expressions.
    fn negate_cond(&mut self, expr: &mut ExprDesc) {
        assert!(
            expr.pending().is_none(),
            "negate_cond requires the pending head to be absorbed first"
        );

        let indices = expr
            .true_list
            .jumps
            .iter()
            .chain(expr.false_list.jumps.iter());

        for &jmp_idx in indices {
            self.flip_control_polarity(jmp_idx);
        }

        mem::swap(&mut expr.true_list, &mut expr.false_list);
    }

    /// "Go if true": arrange for control to fall through when `expr` is
    /// truthy and to jump otherwise. Produces a jump that fires on falsy,
    /// stored in `false_list`. Mirrors Lua 5.4's `luaK_goiftrue`.
    fn goiftrue(&mut self, expr: &mut ExprDesc) {
        match expr.kind {
            ExprKind::Jump(_) => {
                if let Some(idx) = expr.take_pending() {
                    // Absorb the pending head into false_list. Pending
                    // fires on truthy, so flip its CMP first to make it
                    // fire on falsy.
                    self.flip_control_polarity(idx);
                    expr.false_list.jumps.push(idx);
                } else if expr.false_list.is_empty() && !expr.true_list.is_empty() {
                    // No pending head; if the remaining jumps all fire on
                    // truthy (populating `true_list` only), flip polarity
                    // so they land in `false_list` with falsy-firing
                    // semantics.
                    self.negate_cond(expr);
                }
            }
            ExprKind::Reg(reg) => {
                let jmp = self.emit_test_jump(reg, false);
                expr.false_list.jumps.push(jmp);
            }
        }
    }

    /// "Go if false": arrange for control to fall through when `expr` is
    /// falsy and to jump otherwise. Produces a jump that fires on truthy,
    /// stored in `true_list`. Mirrors Lua 5.4's `luaK_goiffalse`.
    fn goiffalse(&mut self, expr: &mut ExprDesc) {
        match expr.kind {
            ExprKind::Jump(_) => {
                if let Some(idx) = expr.take_pending() {
                    // Absorb the pending head into true_list (it already
                    // fires on truthy by construction).
                    expr.true_list.jumps.push(idx);
                } else if expr.true_list.is_empty() && !expr.false_list.is_empty() {
                    // No pending head; if every jump fires on falsy, flip
                    // polarity so they move into `true_list` with
                    // truthy-firing semantics.
                    self.negate_cond(expr);
                }
            }
            ExprKind::Reg(reg) => {
                let jmp = self.emit_test_jump(reg, true);
                expr.true_list.jumps.push(jmp);
            }
        }
    }

    /// Discharge an expression to a concrete register, resolving any pending
    /// jump lists. The emitted shape follows Lua 5.4's `exp2reg`:
    ///
    /// * If every pending jump is TESTSET-controlled (all preserve their
    ///   operand value via the assign-on-short-circuit edge) we skip the
    ///   `LFALSESKIP` / `LOAD K(true)` fixup tail entirely — `patch_list_aux`
    ///   redirects those jumps to the final merge point, no boolean
    ///   materialisation needed.
    /// * Otherwise we emit the two-instruction tail. TESTSET jumps still
    ///   skip to the merge point (preserving their value); plain TEST /
    ///   CMP jumps land on LFALSESKIP or LOADTRUE to produce a boolean in
    ///   `dst`.
    /// * For a Reg-kind expression with pending jumps, the fall-through at
    ///   the moment of discharge already holds the RHS value in `dst`; we
    ///   emit a JMP over the tail so the fall-through value survives.
    /// * For a Jump-kind expression we absorb the `pending` head into
    ///   `true_list` (mirroring Lua's `luaK_concat(&e->t, e->u.info)`).
    ///   Physical fall-through of the last control instruction is
    ///   guaranteed falsy by our invariants, so it lands naturally on
    ///   `LFALSESKIP` (the false_target) and produces the correct boolean
    ///   — no routing JMP needed.
    fn discharge_to_reg_mut(
        &mut self,
        expr: &mut ExprDesc,
        hint: Option<RegisterIndex>,
    ) -> Result<RegisterIndex, CompileError> {
        // Fast path: plain Reg with no pending jumps. `hint` is advisory —
        // honouring it here would emit a MOVE that many callers re-emit,
        // flipping instruction order and inflating frame size. Callers that
        // want the value in a specific register MOVE it themselves.
        if let ExprKind::Reg(reg) = expr.kind
            && !expr.has_jumps()
        {
            return Ok(reg);
        }

        let is_jump = matches!(expr.kind, ExprKind::Jump(_));
        let dst = match expr.kind {
            ExprKind::Reg(reg) => {
                // Reg with pending jumps. Honor the hint: jumps in the list
                // preserve the short-circuit operand by having their TESTSET
                // dst patched to our `dst`. If we used `reg` directly and
                // `reg` happens to be a local's register (e.g. `b` in
                // `local x = a and b`), the TESTSETs would overwrite that
                // local on the falsy short-circuit — a real miscompile. So
                // MOVE the fall-through value into the hint, then patching
                // can target hint safely.
                if let Some(h) = hint
                    && h != reg
                {
                    self.dst_or_alloc(Some(h))?;
                    self.emit(Instruction::MOVE {
                        dst: h.0,
                        src: reg.0,
                    });
                    h
                } else {
                    reg
                }
            }
            ExprKind::Jump(_) => self.dst_or_alloc(hint)?,
        };

        if is_jump {
            // Absorb pending head into true_list (fires on truthy by
            // construction). Fall-through past this JMP is guaranteed
            // falsy, so the LFALSESKIP tail handles it without a routing
            // JMP.
            if let Some(idx) = expr.take_pending() {
                expr.true_list.jumps.push(idx);
            }
        }

        if expr.has_jumps() {
            let need_tail = self.need_value(&expr.false_list) || self.need_value(&expr.true_list);

            // Reg-kind with a live fall-through value needs to skip past
            // the boolean fixup tail so the value stays in `dst`. Jump-kind
            // has no fall-through value (physical fall-through is falsy
            // and is meant to land on LFALSESKIP), so no skip jump.
            let skip_fixup = if !is_jump && need_tail {
                Some(self.emit_unfilled_jmp())
            } else {
                None
            };

            let (false_target, true_target) = if need_tail {
                let ft = self.next_offset();
                self.emit(Instruction::LFALSESKIP { src: dst.0 });
                let tt = self.next_offset();
                let true_idx = self.alloc_constant(Value::Boolean(true))?;
                self.emit(Instruction::LOAD {
                    dst: dst.0,
                    idx: true_idx,
                });
                (ft, tt)
            } else {
                // Unused: every jump in both lists is TESTSET-controlled and
                // will be redirected to `final_pos` by patch_list_aux.
                (0, 0)
            };

            let final_pos = self.next_offset();
            let fl = mem::take(&mut expr.false_list);
            let tl = mem::take(&mut expr.true_list);
            self.patch_list_aux(fl, final_pos, dst, false_target);
            self.patch_list_aux(tl, final_pos, dst, true_target);

            if let Some(idx) = skip_fixup {
                let end = self.next_offset();
                let offset = end as i32 - (idx as i32 + 1);
                if let Instruction::JMP { offset: o } = &mut self.chunk.tape[idx] {
                    *o = offset;
                }
            }
        }

        expr.kind = ExprKind::Reg(dst);
        Ok(dst)
    }

    fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
        self.scope_marks.push(ScopeMark {
            freereg: self.chunk.freereg,
            nactvar: self.chunk.nactvar,
        });
        self.scope_close.push(Vec::new());
    }

    fn pop_scope(&mut self) -> Result<Vec<RegisterIndex>, CompileError> {
        self.scope.pop().ok_or_else(|| ice("missing scope"))?;
        let mark = self
            .scope_marks
            .pop()
            .ok_or_else(|| ice("missing scope register base"))?;
        self.chunk.freereg = mark.freereg;
        self.chunk.nactvar = mark.nactvar;
        self.scope_close
            .pop()
            .ok_or_else(|| ice("missing scope close list"))
    }

    fn mark_close(&mut self, register: RegisterIndex) -> Result<(), CompileError> {
        self.scope_close
            .last_mut()
            .ok_or_else(|| ice("missing scope close list"))?
            .push(register);
        Ok(())
    }

    fn define(&mut self, name: String, data: VariableData) -> Result<(), CompileError> {
        let scope = self.scope.last_mut().ok_or_else(|| ice("missing scope"))?;
        scope.insert(name, data);
        Ok(())
    }

    fn resolve_local(&self, name: &str) -> Option<&VariableData> {
        for scope in self.scope.iter().rev() {
            if let Some(data) = scope.get(name) {
                return Some(data);
            }
        }
        None
    }

    fn alloc_constant(&mut self, value: Value<'gc>) -> Result<u16, CompileError> {
        // Check for an existing identical constant (exact type match, no int/float coercion).
        let existing = self.chunk.constants.iter().position(|c| match (c, &value) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        });

        if let Some(idx) = existing {
            return Ok(idx as u16);
        }

        if self.chunk.constants.len() >= u16::MAX as usize {
            return Err(err(CompileErrorKind::Constants, LineNumber(0)));
        }
        let idx = self.chunk.constants.len() as u16;
        self.chunk.constants.push(value);
        Ok(idx)
    }

    fn alloc_string_constant(&mut self, s: &[u8]) -> Result<u16, CompileError> {
        let lua_str = LuaString::new(self.mc, s);
        self.alloc_constant(Value::String(lua_str))
    }

    /// Resolve `name` into an index in THIS function's upvalue list,
    /// capturing on first reference (via the parent's
    /// `resolve_for_child`). Returns `None` if the name isn't reachable
    /// through any enclosing scope.
    fn resolve_or_capture(&mut self, name: &str) -> Option<u8> {
        if let Some(i) = self.upvalues.iter().position(|(n, _)| n == name) {
            return Some(i as u8);
        }
        let parent = self.capture.as_deref_mut()?;
        let desc = parent.resolve_for_child(name)?;
        let i = self.upvalues.len() as u8;
        self.upvalues.push((name.to_owned(), desc));
        Some(i)
    }

    /// Resolve a variable name as an upvalue from enclosing scopes.
    /// Returns the index in this function's upvalue list, capturing on
    /// first reference.
    fn resolve_upvalue(&mut self, name: &str) -> Option<u8> {
        self.resolve_or_capture(name)
    }
}

impl<'gc, 'a> UpvalueResolver for Ctx<'gc, 'a> {
    /// Given a free-variable reference from a direct child of this
    /// function, return the `UpValueDescriptor` the child should put in
    /// its own upvalue list. `ParentLocal(reg)` when `name` is a local
    /// here; `ParentUpvalue(i)` when it sits in this function's upvalue
    /// list (captured now if necessary).
    fn resolve_for_child(&mut self, name: &str) -> Option<UpValueDescriptor> {
        // 1. Own local? The child references our register directly — no
        //    entry is added to our own upvalues.
        if let Some(reg) = self.resolve_local(name).map(|d| d.register.0) {
            return Some(UpValueDescriptor::ParentLocal(reg));
        }
        // 2. Already captured in our upvalue list?
        if let Some(i) = self.upvalues.iter().position(|(n, _)| n == name) {
            return Some(UpValueDescriptor::ParentUpvalue(i as u8));
        }
        // 3. Cascade to our own parent; if it resolves, we capture the
        //    returned descriptor into our upvalue list and the child
        //    references that new slot.
        let parent = self.capture.as_deref_mut()?;
        let desc = parent.resolve_for_child(name)?;
        let i = self.upvalues.len() as u8;
        self.upvalues.push((name.to_owned(), desc));
        Some(UpValueDescriptor::ParentUpvalue(i))
    }
}

// ---------------------------------------------------------------------------
// Scope helpers
// ---------------------------------------------------------------------------

fn scope_lexical<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    ctx.push_scope();
    compile(ctx)?;

    let close_regs = ctx.pop_scope()?;
    if let Some(first) = close_regs.first() {
        ctx.emit(Instruction::CLOSE { start: first.0 });
    }

    Ok(())
}

fn scope_break<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    let label = ctx.new_label();
    ctx.control_end_label.push(label);
    compile(ctx)?;
    ctx.set_label(label, ctx.next_offset());
    ctx.control_end_label.pop();
    Ok(())
}

fn scope_lexical_break<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    scope_lexical(ctx, |ctx| scope_break(ctx, compile))
}

// ---------------------------------------------------------------------------
// Top-level compilation entry point
// ---------------------------------------------------------------------------

pub fn compile<'gc>(
    mc: &Mutation<'gc>,
    root: &Root,
    interner: &TokenInterner,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    let chunk = compile_function_to_chunk(
        mc,
        interner,
        None, // main chunk has no enclosing function
        root.block(),
        std::iter::empty(),
        true, // main chunk is vararg
        0,
        None,
        // Pre-seed `_ENV` at upvalue 0. The runtime wiring in
        // `src/lua/context.rs` (ctx.load) sets the top-level closure's
        // upvalues directly, so the descriptor here is purely a
        // placeholder — nested functions reference this slot by
        // cascading ParentUpvalue(0).
        vec![("_ENV".to_owned(), UpValueDescriptor::ParentLocal(0))],
    )?;
    Ok(chunk.assemble(mc))
}

/// Compile a function body into a Chunk (not yet assembled).
#[allow(clippy::too_many_arguments)]
fn compile_function_to_chunk<'gc, 'a>(
    mc: &'a Mutation<'gc>,
    interner: &'a TokenInterner,
    parent_capture: Option<&'a mut dyn UpvalueResolver>,
    stmts: impl Iterator<Item = Stmt>,
    params: impl Iterator<Item = Ident>,
    is_vararg: bool,
    arity: u8,
    source: Option<LuaString<'gc>>,
    initial_upvalues: Vec<(String, UpValueDescriptor)>,
) -> Result<Chunk<'gc>, CompileError> {
    let mut chunk = Chunk::new();
    chunk.is_vararg = is_vararg;
    chunk.arity = arity;
    chunk.source = source;

    let mut ctx = Ctx {
        interner,
        mc,
        chunk,
        control_end_label: Vec::new(),
        scope: Vec::new(),
        scope_marks: Vec::new(),
        scope_close: Vec::new(),
        goto_labels: HashMap::new(),
        capture: parent_capture,
        upvalues: initial_upvalues,
    };

    ctx.push_scope();

    // Emit VARARGPREP for vararg functions
    if is_vararg {
        ctx.emit(Instruction::VARARGPREP { num_fixed: arity });
    }

    // Allocate registers for parameters and bind them in scope
    let mut num_params: u8 = 0;
    for param in params {
        let name = param
            .name(interner)
            .ok_or_else(|| ice("parameter without name"))?;
        let reg = ctx.alloc_register()?;
        ctx.define(
            name.to_owned(),
            VariableData {
                register: reg,
                is_const: false,
            },
        )?;
        num_params += 1;
    }
    // Promote parameters to active locals so the body sees them as locals
    // (upvalue capture by the body relies on `nactvar`) and temp reclaims
    // never touch their slots.
    ctx.adjust_locals(num_params);

    // Compile the body
    for stmt in stmts {
        compile_stmt(&mut ctx, stmt)?;
    }

    // Emit implicit return at the end (skip if the last instruction already
    // terminates the frame — RETURN, or TAILCALL which is self-unwinding).
    let needs_return = !matches!(
        ctx.chunk.tape.last(),
        Some(Instruction::RETURN { .. } | Instruction::TAILCALL { .. }),
    );
    if needs_return {
        ctx.emit(Instruction::RETURN {
            values: 0,
            count: 1,
        });
    }

    let close_regs = ctx.pop_scope()?;
    if let Some(first) = close_regs.first() {
        // Insert CLOSE before the final RETURN/TAILCALL
        let return_instr = ctx.chunk.tape.pop().unwrap();
        ctx.chunk.tape.push(Instruction::CLOSE { start: first.0 });
        ctx.chunk.tape.push(return_instr);
    }

    // Flatten the named upvalue list into the chunk's descriptor array.
    ctx.chunk.upvalue_desc = ctx.upvalues.into_iter().map(|(_, d)| d).collect();
    Ok(ctx.chunk)
}

// ---------------------------------------------------------------------------
// Statement compilation
// ---------------------------------------------------------------------------

fn compile_stmt(ctx: &mut Ctx, item: Stmt) -> Result<(), CompileError> {
    match item {
        Stmt::Label(item) => compile_label(ctx, item),
        Stmt::Goto(item) => compile_goto(ctx, item),
        Stmt::Decl(item) => compile_decl(ctx, item),
        Stmt::Assign(item) => compile_assign(ctx, item),
        Stmt::Func(item) => compile_func(ctx, item),
        Stmt::Expr(item) => {
            compile_expr_to_reg(ctx, item, None)?;
            Ok(())
        }
        Stmt::Break(item) => compile_break(ctx, item),
        Stmt::Return(item) => compile_return(ctx, item),
        Stmt::Do(item) => compile_do(ctx, item),
        Stmt::While(item) => compile_while(ctx, item),
        Stmt::Repeat(item) => compile_repeat(ctx, item),
        Stmt::If(item) => compile_if(ctx, item),
        Stmt::ForNum(item) => compile_for_num(ctx, item),
        Stmt::ForGen(item) => compile_for_gen(ctx, item),
    }
}

// ---------------------------------------------------------------------------
// Label / Goto
// ---------------------------------------------------------------------------

fn compile_label(ctx: &mut Ctx, item: Label) -> Result<(), CompileError> {
    let name = item
        .name()
        .ok_or_else(|| ice("label without name"))?
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?
        .to_owned();

    if let Some(&label_idx) = ctx.goto_labels.get(&name) {
        // Forward reference already allocated — resolve it now
        ctx.set_label(label_idx, ctx.next_offset());
    } else {
        let label_idx = ctx.new_label();
        ctx.set_label(label_idx, ctx.next_offset());
        ctx.goto_labels.insert(name, label_idx);
    }

    Ok(())
}

fn compile_goto(ctx: &mut Ctx, item: Goto) -> Result<(), CompileError> {
    let name = item
        .label()
        .ok_or_else(|| ice("goto without label"))?
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?
        .to_owned();

    if let Some(&label_idx) = ctx.goto_labels.get(&name) {
        ctx.emit_jump(label_idx);
    } else {
        // Forward goto — allocate a label that will be resolved when ::name:: is encountered
        let label_idx = ctx.new_label();
        ctx.goto_labels.insert(name, label_idx);
        ctx.emit_jump(label_idx);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

fn compile_decl(ctx: &mut Ctx, item: Decl) -> Result<(), CompileError> {
    // Check for `local function ...` syntax
    if let Some(func) = item.function() {
        let name = func
            .name(ctx.interner)
            .ok_or_else(|| ice("local function without name"))?
            .to_string();

        // Allocate register first so the function can reference itself
        let reg = ctx.alloc_register()?;
        ctx.define(
            name,
            VariableData {
                register: reg,
                is_const: false,
            },
        )?;
        // Promote to an active local BEFORE compiling the body so the
        // child function's upvalue capture sees a stable parent register
        // and any temps allocated during body compilation don't reclaim
        // the local's slot.
        ctx.adjust_locals(1);

        let func_reg = compile_func_body(ctx, &func, Some(reg))?;
        if func_reg != reg {
            ctx.emit(Instruction::MOVE {
                dst: reg.0,
                src: func_reg.0,
            });
        }
        return Ok(());
    }

    let targets: Vec<_> = item
        .targets()
        .ok_or_else(|| ice("decl without targets"))?
        .collect();

    let values: Vec<_> = item
        .values()
        .map(|v| v.collect::<Vec<_>>())
        .unwrap_or_default();

    let num_targets = targets.len();
    let num_values = values.len();

    // Target slots will land contiguously at [base, base+num_targets).
    // Don't pre-reserve: compile each value expression into the next free
    // slot, letting its natural register placement (e.g. CALL's result
    // slot, LOAD dst) line up with the target. This avoids the
    // "pre-allocate then MOVE result down" pattern for calls, literals,
    // and arithmetic. MOVEs are only emitted when the expression returns
    // a register that's not at the expected slot (e.g. `local x = y`
    // where `y` is an existing local).
    let base = ctx.chunk.freereg;

    for (i, expr) in values.into_iter().enumerate() {
        let is_last = i == num_values - 1;
        let expected = base + i as u8;

        if is_last && num_targets > num_values {
            // Last RHS supplies multiple values via call or vararg.
            if let Expr::FuncCall(call) = expr {
                let want = num_targets - i;
                let regs = compile_expr_func_call(ctx, call, want)?;
                // Results occupy [regs[0], regs[0]+want); in practice this
                // lines up with `expected` because the call's func slot
                // is freereg at setup time, which is `expected`.
                debug_assert_eq!(regs[0].0, expected);
                continue;
            }
            if let Expr::VarArg = expr {
                let want = num_targets - i;
                let dst = ctx.reserve_regs(want as u8)?;
                debug_assert_eq!(dst.0, expected);
                ctx.emit(Instruction::VARARG {
                    dst: dst.0,
                    count: want as u8 + 1,
                });
                continue;
            }
        }

        // Regular single-value expression. Compile without a pre-reserved
        // slot so CALL / LOAD / arith destinations land naturally at
        // `expected`. If the expression returns a different register
        // (e.g. a local Ident that `compile_expr_ident` returned bare, or
        // a short-circuit expression whose fall-through landed mid-stack),
        // emit a MOVE. Either way, end this iteration with freereg =
        // expected + 1 — any temps the expression leaked above are dead.
        let reg = compile_expr_to_reg(ctx, expr, None)?;
        if reg.0 != expected {
            // Reclaim any leaked temps, then reserve the target slot
            // (reserve_reg updates max_stack).
            ctx.chunk.freereg = expected;
            let slot = ctx.reserve_reg()?;
            debug_assert_eq!(slot.0, expected);
            ctx.emit(Instruction::MOVE {
                dst: slot.0,
                src: reg.0,
            });
        } else if ctx.chunk.freereg > expected + 1 {
            // Expression landed at `expected` but leaked additional temps
            // above (e.g. short-circuit fall-through). Drop them. The
            // previous expression compilation already updated max_stack to
            // reflect this peak.
            ctx.chunk.freereg = expected + 1;
        }
    }

    // Pad with nil for any targets without supplied values. A multi-return
    // call or vararg may have already filled extra slots above the last
    // non-expander value, so check freereg rather than iterating by index.
    let target_top = base + num_targets as u8;
    while ctx.chunk.freereg < target_top {
        let slot = ctx.reserve_reg()?;
        let idx = ctx.alloc_constant(Value::Nil)?;
        ctx.emit(Instruction::LOAD { dst: slot.0, idx });
    }

    // Bind each target name to its slot and handle `<close>` / `<const>`.
    for (i, target) in targets.into_iter().enumerate() {
        let name = target
            .name()
            .ok_or_else(|| ice("decl target without name"))?
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?
            .to_owned();

        let modifier = target.modifier();
        let is_const = matches!(modifier, Some(DeclModifier::Const));
        let is_close = matches!(modifier, Some(DeclModifier::Close));

        let reg = RegisterIndex(base + i as u8);

        if is_close {
            ctx.emit(Instruction::TBC { val: reg.0 });
            ctx.mark_close(reg)?;
        }

        ctx.define(
            name,
            VariableData {
                register: reg,
                is_const,
            },
        )?;
    }

    // Promote the freshly bound targets to active locals so their slots
    // are stable for the rest of the scope (upvalue-capture depends on
    // the register staying put).
    ctx.adjust_locals(num_targets as u8);

    Ok(())
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Pre-resolved LHS of a multi-assignment. Index/Property targets carry
/// their table-and-key registers, computed eagerly so later RHS values
/// or earlier-slot stores can't clobber them before the SETTABLE fires.
enum Lvalue {
    Local {
        dst: RegisterIndex,
    },
    Upvalue {
        idx: u8,
    },
    Global {
        env_idx: u8,
        key: u16,
    },
    Indexed {
        table: RegisterIndex,
        key: RegisterIndex,
    },
}

/// Lua 5.4 §3.3.3 specifies "first evaluate all its expressions and only
/// then perform the assignments." We honour that without losing the
/// hint-into-target optimisation by:
///
///   1. resolving every LHS to an `Lvalue` up front (Index/Property
///      sub-expressions land in registers, copied to temps if they
///      reference a local that's also a target);
///   2. compiling each RHS, hinting into the slot's target local iff no
///      later RHS reads it — that lets value computation perform the
///      assignment as a side effect (no MOVE in pass 4);
///   3. saving any non-hinted source register that aliases an earlier
///      target local, so the earlier store doesn't clobber the value a
///      later store still needs (`b, a = a, b` cycles);
///   4. emitting the surviving stores.
fn compile_assign(ctx: &mut Ctx, item: Assign) -> Result<(), CompileError> {
    let targets: Vec<_> = item
        .targets()
        .ok_or_else(|| ice("assign without targets"))?
        .collect();
    let values: Vec<Expr> = item
        .values()
        .ok_or_else(|| ice("assign without values"))?
        .collect();

    let num_targets = targets.len();
    let num_values = values.len();
    let freereg_before = ctx.chunk.freereg;

    // Set of local registers that are themselves LHS targets — drives
    // the conflict checks in pass 1 and pass 3.
    let target_local_regs: Vec<u8> = targets
        .iter()
        .filter_map(|t| target_local_reg(ctx, t))
        .collect();

    // Pass 1: resolve LHS targets.
    let mut lvalues: Vec<Lvalue> = Vec::with_capacity(num_targets);
    for target in targets {
        lvalues.push(compile_lvalue(ctx, target, &target_local_regs)?);
    }

    // Decide per-slot whether it's safe to hint the RHS directly into
    // the target local: only when the target is a local AND no later
    // RHS reads that local (an in-place hint *is* an early assignment).
    let mut hints: Vec<Option<RegisterIndex>> = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let h = match local_dst(&lvalues[i]) {
            Some(dst) if !any_reads(ctx, &values[i + 1..], dst.0)? => Some(dst),
            _ => None,
        };
        hints.push(h);
    }

    // Pass 2: compile RHS values. `pending[i] = Some(reg)` means pass 4
    // must emit a store from `reg`; `None` means the value was hinted
    // straight into the target local already.
    let mut pending: Vec<Option<u8>> = Vec::with_capacity(num_targets);
    for (i, expr) in values.into_iter().enumerate() {
        let is_last = i == num_values - 1;
        if is_last
            && num_targets > num_values
            && let Expr::FuncCall(call) = expr
        {
            let want = num_targets - pending.len();
            for r in compile_expr_func_call(ctx, call, want)? {
                pending.push(Some(r.0));
            }
            continue;
        }
        let hint = hints[i];
        let reg = compile_expr_to_reg(ctx, expr, hint)?;
        pending.push(if hint == Some(reg) { None } else { Some(reg.0) });
    }

    // Pad with nil if fewer values than targets. A nil pad reads nothing,
    // so hinting into a Local target is always safe.
    while pending.len() < num_targets {
        let i = pending.len();
        let nil = ctx.alloc_constant(Value::Nil)?;
        let hint = local_dst(&lvalues[i]);
        let reg = ctx.dst_or_alloc(hint)?;
        ctx.emit(Instruction::LOAD {
            dst: reg.0,
            idx: nil,
        });
        pending.push(if hint == Some(reg) { None } else { Some(reg.0) });
    }

    // Pass 3: any source register that aliases an earlier slot's target
    // local will be clobbered by that earlier store — save it now.
    for j in 0..num_targets {
        let Some(r) = pending[j] else { continue };
        let collides = (0..j).any(|i| local_dst(&lvalues[i]).is_some_and(|d| d.0 == r));
        if collides {
            let temp = ctx.alloc_register()?;
            ctx.emit(Instruction::MOVE {
                dst: temp.0,
                src: r,
            });
            pending[j] = Some(temp.0);
        }
    }

    // Pass 4: emit stores for the non-hinted slots.
    for (lv, src) in lvalues.into_iter().zip(&pending) {
        let Some(val) = *src else { continue };
        emit_store(ctx, lv, val);
    }

    assert!(ctx.chunk.freereg >= freereg_before);
    ctx.chunk.freereg = freereg_before;
    Ok(())
}

/// Resolve an LHS target to an `Lvalue`. Index/Property sub-expressions
/// are materialised into registers; any sub-expression result that
/// happens to be a target local is copied into a temp so that a later
/// assignment to that local cannot disturb this Indexed's table or key.
fn compile_lvalue(
    ctx: &mut Ctx,
    target: Expr,
    target_local_regs: &[u8],
) -> Result<Lvalue, CompileError> {
    match target {
        Expr::Ident(ident) => {
            let name = ident
                .name(ctx.interner)
                .ok_or_else(|| ice("ident without name"))?;
            if let Some(data) = ctx.resolve_local(name) {
                if data.is_const {
                    return Err(err(
                        CompileErrorKind::Internal("assignment to const variable"),
                        LineNumber(0),
                    ));
                }
                Ok(Lvalue::Local { dst: data.register })
            } else if let Some(idx) = ctx.resolve_upvalue(name) {
                Ok(Lvalue::Upvalue { idx })
            } else {
                let key = ctx.alloc_string_constant(name.as_bytes())?;
                let env_idx = ctx
                    .resolve_or_capture("_ENV")
                    .ok_or_else(|| ice("_ENV must resolve; main chunk pre-seeds it"))?;
                Ok(Lvalue::Global { env_idx, key })
            }
        }
        Expr::Index(index) => {
            let t = index.target().ok_or_else(|| ice("index without target"))?;
            let k = index.index().ok_or_else(|| ice("index without key"))?;
            let table = compile_indexed_subexpr(ctx, t, target_local_regs)?;
            let key = compile_indexed_subexpr(ctx, k, target_local_regs)?;
            Ok(Lvalue::Indexed { table, key })
        }
        Expr::BinaryOp(binop) if binop.op() == Some(BinaryOperator::Property) => {
            let t = binop.lhs().ok_or_else(|| ice("binop without lhs"))?;
            let f = binop.rhs().ok_or_else(|| ice("binop without rhs"))?;
            let table = compile_indexed_subexpr(ctx, t, target_local_regs)?;
            // Property key is a freshly-LOADed string constant — its
            // register is a fresh temp, never a target local.
            let key = compile_property_key(ctx, f, None)?;
            Ok(Lvalue::Indexed { table, key })
        }
        Expr::BinaryOp(_) => Err(ice("non-property binop as assignment target")),
        _ => Err(ice("invalid assignment target")),
    }
}

/// Compile a sub-expression of an Indexed lvalue (table or key) into a
/// register, copying to a fresh temp if the result coincides with a
/// local that's also a target.
fn compile_indexed_subexpr(
    ctx: &mut Ctx,
    expr: Expr,
    target_local_regs: &[u8],
) -> Result<RegisterIndex, CompileError> {
    let reg = compile_expr_to_reg(ctx, expr, None)?;
    if !target_local_regs.contains(&reg.0) {
        return Ok(reg);
    }
    let temp = ctx.alloc_register()?;
    ctx.emit(Instruction::MOVE {
        dst: temp.0,
        src: reg.0,
    });
    Ok(temp)
}

/// The local register a non-const local-ident target compiles into,
/// or `None` for any other target shape.
fn target_local_reg(ctx: &Ctx, target: &Expr) -> Option<u8> {
    let Expr::Ident(ident) = target else {
        return None;
    };
    let name = ident.name(ctx.interner)?;
    let data = ctx.resolve_local(name)?;
    (!data.is_const).then_some(data.register.0)
}

fn local_dst(lv: &Lvalue) -> Option<RegisterIndex> {
    match lv {
        Lvalue::Local { dst } => Some(*dst),
        _ => None,
    }
}

/// True if any of `exprs`, evaluated immediately, performs a register
/// read from `reg`. Used to gate hint-into-target — if a later RHS
/// reads the local, hinting would corrupt that read.
///
/// `function ... end` literals don't count: the body's reads happen at
/// call time via upvalues. Field-name idents under `.`/`:` don't count
/// either: they're string keys, not variable reads.
fn any_reads(ctx: &Ctx, exprs: &[Expr], reg: u8) -> Result<bool, CompileError> {
    for e in exprs {
        if expr_reads(ctx, e, reg)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn expr_reads(ctx: &Ctx, expr: &Expr, reg: u8) -> Result<bool, CompileError> {
    let any = |opt: Option<Expr>| -> Result<bool, CompileError> {
        opt.as_ref().map_or(Ok(false), |e| expr_reads(ctx, e, reg))
    };
    Ok(match expr {
        Expr::Literal(_) | Expr::VarArg | Expr::Func(_) => false,
        Expr::Ident(ident) => ident
            .name(ctx.interner)
            .and_then(|n| ctx.resolve_local(n))
            .is_some_and(|d| d.register.0 == reg),
        Expr::PrefixOp(p) => any(p.rhs())?,
        Expr::BinaryOp(b) => {
            let skip_rhs = matches!(
                b.op(),
                Some(BinaryOperator::Property | BinaryOperator::Method)
            );
            any(b.lhs())? || (!skip_rhs && any(b.rhs())?)
        }
        Expr::Index(i) => any(i.target())? || any(i.index())?,
        Expr::FuncCall(c) => {
            if any(c.target())? {
                return Ok(true);
            }
            for arg in c.args().into_iter().flatten() {
                if expr_reads(ctx, &arg, reg)? {
                    return Ok(true);
                }
            }
            false
        }
        Expr::Method(m) => {
            if any(m.object())? {
                return Ok(true);
            }
            for arg in m.args().into_iter().flatten() {
                if expr_reads(ctx, &arg, reg)? {
                    return Ok(true);
                }
            }
            false
        }
        Expr::Table(t) => {
            for entry in t.entries() {
                let touched = match entry {
                    TableEntry::Array(a) => any(a.value())?,
                    TableEntry::Map(m) => any(m.value())?,
                    TableEntry::Generic(g) => any(g.index())? || any(g.value())?,
                };
                if touched {
                    return Ok(true);
                }
            }
            false
        }
    })
}

/// Emit the store instruction for a resolved Lvalue from a source register.
/// Local self-stores are elided.
fn emit_store(ctx: &mut Ctx, lv: Lvalue, src: u8) {
    match lv {
        Lvalue::Local { dst } if dst.0 != src => ctx.emit(Instruction::MOVE { dst: dst.0, src }),
        Lvalue::Local { .. } => {} // self-MOVE; skip
        Lvalue::Upvalue { idx } => ctx.emit(Instruction::SETUPVAL { src, idx }),
        Lvalue::Global { env_idx, key } => ctx.emit(Instruction::SETTABUP {
            src,
            idx: env_idx,
            key,
        }),
        Lvalue::Indexed { table, key } => ctx.emit(Instruction::SETTABLE {
            src,
            table: table.0,
            key: key.0,
        }),
    }
}

/// Single-target assignment: resolve the LHS and emit the store. Used by
/// `function name() ... end`, where there's exactly one target and no
/// aliasing to guard against.
fn compile_assign_lhs(ctx: &mut Ctx, target: Expr, value: u8) -> Result<(), CompileError> {
    let freereg_before = ctx.chunk.freereg;
    let lv = compile_lvalue(ctx, target, &[])?;
    emit_store(ctx, lv, value);
    ctx.chunk.freereg = freereg_before;
    Ok(())
}

// ---------------------------------------------------------------------------
// Function statement
// ---------------------------------------------------------------------------

fn compile_func(ctx: &mut Ctx, item: Func) -> Result<(), CompileError> {
    // `function target() ... end` is sugar for `target = function() ... end`
    let target = item
        .target()
        .ok_or_else(|| ice("func stmt without target"))?;

    let func_reg = compile_func_body(ctx, &item, None)?;

    compile_assign_lhs(ctx, target, func_reg.0)?;

    Ok(())
}

fn compile_func_body(
    ctx: &mut Ctx,
    item: &Func,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::CLOSURE {
        dst: dst.0,
        proto: proto_idx,
    });

    Ok(dst)
}

/// Compile a nested function, handling upvalue capture from the parent
/// context. Upvalues are resolved on demand: the child's
/// `resolve_or_capture` calls back into `ctx` (the parent) through the
/// `UpvalueResolver` trait, which cascades upward from any depth and
/// inserts `ParentLocal` / `ParentUpvalue` descriptors into each
/// function's list as it goes.
fn compile_nested<'gc>(
    ctx: &mut Ctx<'gc, '_>,
    stmts: Vec<Stmt>,
    params: Vec<Ident>,
    is_vararg: bool,
    arity: u8,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    let mc = ctx.mc;
    let interner = ctx.interner;
    let parent: &mut dyn UpvalueResolver = ctx;

    let chunk = compile_function_to_chunk(
        mc,
        interner,
        Some(parent),
        stmts.into_iter(),
        params.into_iter(),
        is_vararg,
        arity,
        None,
        Vec::new(),
    )?;

    Ok(chunk.assemble(mc))
}

// ---------------------------------------------------------------------------
// Expression compilation
// ---------------------------------------------------------------------------

/// Compile an expression, returning an `ExprDesc` that may carry pending
/// jump lists instead of a materialised value (for comparisons, `not`, and
/// short-circuit `and`/`or`). Callers who need a concrete register call
/// `compile_expr_to_reg` instead, which discharges through the standard
/// `LFALSESKIP` / `LOAD true` fixup tail when lists are non-empty.
fn compile_expr(
    ctx: &mut Ctx,
    item: Expr,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    match item {
        Expr::PrefixOp(item) => compile_expr_prefix_op(ctx, item, dst),
        Expr::BinaryOp(item) => compile_expr_binary_op(ctx, item, dst),
        Expr::Method(item) => {
            let regs = compile_expr_method_call(ctx, item, 1)?;
            Ok(ExprDesc::from_reg(regs[0]))
        }
        Expr::Ident(item) => compile_expr_ident(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Literal(item) => compile_expr_literal(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Func(item) => compile_expr_func(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Table(item) => compile_expr_table(ctx, item).map(ExprDesc::from_reg),
        Expr::FuncCall(item) => {
            let regs = compile_expr_func_call(ctx, item, 1)?;
            Ok(ExprDesc::from_reg(regs[0]))
        }
        Expr::Index(item) => compile_expr_index(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::VarArg => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::VARARG {
                dst: dst.0,
                count: 2,
            });
            Ok(ExprDesc::from_reg(dst))
        }
    }
}

fn compile_expr_to_reg(
    ctx: &mut Ctx,
    item: Expr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let mut expr = compile_expr(ctx, item, dst)?;
    ctx.discharge_to_reg_mut(&mut expr, dst)
}

fn compile_expr_ident(
    ctx: &mut Ctx,
    item: Ident,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let name = item
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?;

    // Local variable — returns its own register, ignores dst hint
    if let Some(data) = ctx.resolve_local(name) {
        return Ok(data.register);
    }

    // Upvalue
    if let Some(idx) = ctx.resolve_upvalue(name) {
        let dst = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::GETUPVAL { dst: dst.0, idx });
        return Ok(dst);
    }

    // Global: _ENV[name]
    let key = ctx.alloc_string_constant(name.as_bytes())?;
    let dst = ctx.dst_or_alloc(dst)?;
    let env_idx = ctx
        .resolve_or_capture("_ENV")
        .ok_or_else(|| ice("_ENV must resolve; main chunk pre-seeds it"))?;
    ctx.emit(Instruction::GETTABUP {
        dst: dst.0,
        idx: env_idx,
        key,
    });
    Ok(dst)
}

fn compile_expr_literal(
    ctx: &mut Ctx,
    item: Literal,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let value = item
        .value(ctx.interner)
        .ok_or_else(|| ice("literal without value"))?;

    let constant = match value {
        LiteralValue::Nil => Value::Nil,
        LiteralValue::Bool(b) => Value::Boolean(b),
        LiteralValue::Int(n) => Value::Integer(n),
        LiteralValue::Float(n) => Value::Float(n),
        LiteralValue::String(bytes) => Value::String(LuaString::new(ctx.mc, &bytes)),
    };

    let idx = ctx.alloc_constant(constant)?;
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::LOAD { dst: dst.0, idx });
    Ok(dst)
}

fn compile_expr_func(
    ctx: &mut Ctx,
    item: FuncExpr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::CLOSURE {
        dst: dst.0,
        proto: proto_idx,
    });

    Ok(dst)
}

fn compile_property_key(
    ctx: &mut Ctx,
    field: Expr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    // Property access RHS should be an identifier used as a string key
    if let Expr::Ident(ident) = field {
        let name = ident
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?;
        let idx = ctx.alloc_string_constant(name.as_bytes())?;
        let reg = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::LOAD { dst: reg.0, idx });
        Ok(reg)
    } else {
        // Fallback: compile as expression
        compile_expr_to_reg(ctx, field, dst)
    }
}

fn compile_expr_table(ctx: &mut Ctx, item: Table) -> Result<RegisterIndex, CompileError> {
    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::NEWTABLE { dst: dst.0 });

    let mut array_count: u16 = 0;
    let mut array_pending = 0u8;

    // Reclaim per-entry temps back to this cursor after each SETLIST flush
    // and after each Map/Generic SETTABLE.
    let pending_base = ctx.chunk.freereg;

    for entry in item.entries() {
        match entry {
            TableEntry::Array(arr) => {
                let value_expr = arr
                    .value()
                    .ok_or_else(|| ice("table array without value"))?;
                // Compile the value into the next array slot (dst+1+pending).
                let slot = RegisterIndex(pending_base + array_pending);
                let val = compile_expr_to_reg(ctx, value_expr, Some(slot))?;
                if val != slot {
                    // Ensure the slot exists, then MOVE the value in and
                    // free the source temp.
                    while ctx.chunk.freereg <= slot.0 {
                        ctx.alloc_register()?;
                    }
                    ctx.emit(Instruction::MOVE {
                        dst: slot.0,
                        src: val.0,
                    });
                    ctx.free_reg(val);
                }
                array_pending += 1;
                array_count += 1;

                // Flush when we hit the batch limit
                if array_pending >= 50 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                    ctx.chunk.freereg = pending_base;
                }
            }

            TableEntry::Map(map) => {
                // Flush pending array entries first
                if array_pending > 0 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                    ctx.chunk.freereg = pending_base;
                }

                let field = map.field().ok_or_else(|| ice("table map without field"))?;
                let key = compile_property_key(ctx, Expr::Ident(field), None)?;

                let value_expr = map.value().ok_or_else(|| ice("table map without value"))?;
                let val = compile_expr_to_reg(ctx, value_expr, None)?;

                ctx.emit(Instruction::SETTABLE {
                    src: val.0,
                    table: dst.0,
                    key: key.0,
                });
                // Free val (higher) then key — SETTABLE has captured both.
                ctx.free_regs(val, key);
            }

            TableEntry::Generic(r#gen) => {
                // Flush pending array entries first
                if array_pending > 0 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                    ctx.chunk.freereg = pending_base;
                }

                let key_expr = r#gen
                    .index()
                    .ok_or_else(|| ice("table generic without index"))?;
                let val_expr = r#gen
                    .value()
                    .ok_or_else(|| ice("table generic without value"))?;
                let key = compile_expr_to_reg(ctx, key_expr, None)?;
                let val = compile_expr_to_reg(ctx, val_expr, None)?;

                ctx.emit(Instruction::SETTABLE {
                    src: val.0,
                    table: dst.0,
                    key: key.0,
                });
                ctx.free_regs(val, key);
            }
        }
    }

    // Flush remaining array entries
    if array_pending > 0 {
        ctx.emit(Instruction::SETLIST {
            table: dst.0,
            count: array_pending,
            offset: array_count - array_pending as u16,
        });
        ctx.chunk.freereg = pending_base;
    }

    Ok(dst)
}

fn compile_expr_prefix_op(
    ctx: &mut Ctx,
    item: PrefixOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let op = item.op().ok_or_else(|| ice("prefix op without operator"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("prefix op without operand"))?;

    // `not <expr>`: when the operand is a jump-list expression (comparison,
    // `not`-of-comparison, etc.) we relabel its lists and flip the
    // `pending` head's CMP polarity in place (Lua's `codenot` on VJMP).
    // The polarity flip is what keeps "pending fires on current truthy"
    // and "fall-through is falsy" invariant after the label change: each
    // still-in-list jump retains its runtime firing condition (its former
    // list-label relabels to the opposite meaning via the swap), but the
    // tail CMP flip ensures physical fall-through now represents outer
    // falsy instead of outer truthy. For plain Reg operands fall back to
    // the NOT opcode.
    if matches!(op, PrefixOperator::Not) {
        let mut inner = compile_expr(ctx, rhs_expr, None)?;
        match inner.kind {
            ExprKind::Jump(Some(idx)) => {
                ctx.flip_control_polarity(idx);
                mem::swap(&mut inner.true_list, &mut inner.false_list);
                return Ok(inner);
            }
            // Jump-kind without a pending head is only reachable after a
            // `goif*` or `discharge_to_reg_mut` consumes it, and `compile_expr` never returns such a
            // state. If we ever got here, relabeling lists without
            // flipping the physical fall-through would silently invert
            // the materialised boolean — fail loudly instead.
            ExprKind::Jump(None) => {
                return Err(ice("not on Jump-kind ExprDesc with no pending head"));
            }
            ExprKind::Reg(src) => {
                ctx.free_reg(src);
                let dst = ctx.dst_or_alloc(dst)?;
                ctx.emit(Instruction::NOT {
                    dst: dst.0,
                    src: src.0,
                });
                return Ok(ExprDesc::from_reg(dst));
            }
        }
    }

    let src = compile_expr_to_reg(ctx, rhs_expr, None)?;

    let result_reg = match op {
        PrefixOperator::None => {
            // Unary + is a no-op — keep `src` live, no free.
            src
        }
        PrefixOperator::Neg => {
            ctx.free_reg(src);
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::UNM {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
        PrefixOperator::Not => unreachable!("handled above"),
        PrefixOperator::Len => {
            ctx.free_reg(src);
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::LEN {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
        PrefixOperator::BitNot => {
            ctx.free_reg(src);
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::BNOT {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
    };
    Ok(ExprDesc::from_reg(result_reg))
}

fn compile_expr_binary_op(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let op = item.op().ok_or_else(|| ice("binary op without operator"))?;

    // Comparisons and logical operators produce jump-list expressions so
    // callers that branch on the result can avoid the LOAD-true / LFALSESKIP
    // materialisation tail, and nested chains (`a < b and c < d`,
    // `x or y or z`) short-circuit through the shared list machinery.
    match op {
        BinaryOperator::Eq
        | BinaryOperator::NEq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LEq
        | BinaryOperator::GEq => return compile_comparison_desc(ctx, item, op),
        BinaryOperator::And => return compile_logical_and_desc(ctx, item, dst),
        BinaryOperator::Or => return compile_logical_or_desc(ctx, item, dst),
        _ => {}
    }

    compile_expr_binary_op_to_reg(ctx, item, dst).map(ExprDesc::from_reg)
}

fn compile_expr_binary_op_to_reg(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let op = item.op().ok_or_else(|| ice("binary op without operator"))?;

    // Property access: a.b
    if op == BinaryOperator::Property {
        let lhs = item.lhs().ok_or_else(|| ice("binop without lhs"))?;
        let rhs = item.rhs().ok_or_else(|| ice("binop without rhs"))?;
        let table = compile_expr_to_reg(ctx, lhs, None)?;
        let key = compile_property_key(ctx, rhs, None)?;
        ctx.free_regs(table, key);
        let dst = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::GETTABLE {
            dst: dst.0,
            table: table.0,
            key: key.0,
        });
        return Ok(dst);
    }

    // Method syntax in expressions (a:b) — should not appear here, handled by MethodCall
    if op == BinaryOperator::Method {
        return Err(ice("method operator in binary expression"));
    }

    let lhs_expr = item.lhs().ok_or_else(|| ice("binop without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("binop without rhs"))?;
    let lhs = compile_expr_to_reg(ctx, lhs_expr, None)?;
    let rhs = compile_expr_to_reg(ctx, rhs_expr, None)?;

    // Arithmetic and bitwise operations
    match op {
        BinaryOperator::Add => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::ADD {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Sub => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SUB {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Mul => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::MUL {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Div => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::DIV {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::IntDiv => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::IDIV {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Mod => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::MOD {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Exp => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::POW {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitAnd => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BAND {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitOr => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BOR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitXor => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BXOR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::LShift => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SHL {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::RShift => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SHR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Concat => {
            ctx.free_regs(lhs, rhs);
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::CONCAT {
                dst: dst.0,
                lhs: lhs.0,
                rhs: rhs.0,
            });
            Ok(dst)
        }

        BinaryOperator::Eq
        | BinaryOperator::NEq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LEq
        | BinaryOperator::GEq => {
            unreachable!(
                "comparisons intercepted by compile_expr_binary_op → compile_comparison_desc"
            )
        }
        BinaryOperator::And | BinaryOperator::Or => {
            unreachable!(
                "and/or intercepted by compile_expr_binary_op → compile_logical_{{and,or}}_desc"
            )
        }
        BinaryOperator::Property | BinaryOperator::Method => {
            unreachable!("property/method handled earlier in this function")
        }
    }
}

fn emit_arith(
    ctx: &mut Ctx,
    lhs: RegisterIndex,
    rhs: RegisterIndex,
    mut instr: Instruction,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    // Free operand temps first so the dst allocation can reuse their
    // slot(s). Locals are no-ops. Matches Lua's `codebinexpval`.
    ctx.free_regs(lhs, rhs);
    let dst = ctx.dst_or_alloc(dst)?;
    // Patch the dst field in the instruction
    match &mut instr {
        Instruction::ADD { dst: d, .. }
        | Instruction::SUB { dst: d, .. }
        | Instruction::MUL { dst: d, .. }
        | Instruction::DIV { dst: d, .. }
        | Instruction::IDIV { dst: d, .. }
        | Instruction::MOD { dst: d, .. }
        | Instruction::POW { dst: d, .. }
        | Instruction::BAND { dst: d, .. }
        | Instruction::BOR { dst: d, .. }
        | Instruction::BXOR { dst: d, .. }
        | Instruction::SHL { dst: d, .. }
        | Instruction::SHR { dst: d, .. } => *d = dst.0,
        _ => unreachable!(),
    }
    ctx.emit(instr);
    Ok(dst)
}

/// Compile a comparison (`==`, `~=`, `<`, `>`, `<=`, `>=`) as a jump-list
/// expression: emit just `CMP` + an unfilled `JMP` and return a `Jump`-kind
/// `ExprDesc` whose `pending` head holds the JMP (fires on truthy).
/// Consumers that branch on the result call `goiffalse` / `goiftrue` to
/// absorb the head into the right list; consumers that need a boolean in
/// a register call `discharge_to_reg_mut`, which emits the standard
/// `LFALSESKIP` / `LOAD K(true)` fixup tail.
fn compile_comparison_desc(
    ctx: &mut Ctx,
    item: BinaryOp,
    op: BinaryOperator,
) -> Result<ExprDesc, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("cmp without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("cmp without rhs"))?;
    let lhs = compile_expr_to_reg(ctx, lhs_expr, None)?;
    let rhs = compile_expr_to_reg(ctx, rhs_expr, None)?;

    // Lua 5.4 convention: emit the CMP so the paired JMP fires on the
    // TRUTHY outcome of the comparison. The VM's `op_lt` / `op_le` /
    // `op_eq` handlers skip the following instruction iff
    // `(cmp_result != inverted) == true` — with `inverted=true`, the skip
    // fires on falsy and the JMP fires on truthy, which is the polarity
    // we want for value contexts (`discharge_to_reg_mut` emits
    // `LFALSESKIP`-first, so fall-through naturally produces the falsy
    // boolean). NEq / opposite-comparisons carry `inverted=false` to
    // cancel the surface-level negation.
    let instr = match op {
        BinaryOperator::Eq => Instruction::EQ {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: true,
        },
        BinaryOperator::NEq => Instruction::EQ {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: false,
        },
        BinaryOperator::Lt => Instruction::LT {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: true,
        },
        BinaryOperator::Gt => Instruction::LT {
            lhs: rhs.0,
            rhs: lhs.0,
            inverted: true,
        },
        BinaryOperator::LEq => Instruction::LE {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: true,
        },
        BinaryOperator::GEq => Instruction::LE {
            lhs: rhs.0,
            rhs: lhs.0,
            inverted: true,
        },
        _ => return Err(ice("compile_comparison_desc called with non-comparison op")),
    };
    ctx.emit(instr);
    let jmp = ctx.emit_unfilled_jmp();
    // CMP + JMP have captured the operands; reclaim the temps so the
    // enclosing jump-list discharge can use their slots.
    ctx.free_regs(lhs, rhs);

    Ok(ExprDesc {
        // Pending head — fires on truthy of this expression. `goiftrue` /
        // `goiffalse` / discharge absorb it into a concrete list; `not`
        // flips its CMP polarity in place.
        kind: ExprKind::Jump(Some(jmp)),
        true_list: JumpList::new(),
        false_list: JumpList::new(),
    })
}

/// Jump-list compilation of `lhs and rhs`. When `lhs` is falsy the whole
/// expression's value is `lhs` — `goiftrue` arranges a jump on falsy that
/// exits the expression (and, for Reg operands, the TESTSET it emits
/// preserves `lhs`'s value on that edge). When `lhs` is truthy, control
/// falls through to the RHS evaluation and the result is `rhs`.
///
/// The merged `ExprDesc` has:
///   * `false_list = lhs.false_list ∪ rhs.false_list` — any falsy exit
///     from either operand short-circuits the whole expression as falsy.
///   * `true_list  = rhs.true_list` — only `rhs` being truthy makes the
///     whole thing truthy.
fn compile_logical_and_desc(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("and without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("and without rhs"))?;

    // Thread the destination hint to both operands so the short-circuit
    // TESTSET can end up as a `TESTSET dst dst` (self-assign, downgraded
    // to TEST by `patch_list_aux`) — saving the intermediate register
    // and MOVE that a fresh-register allocation would otherwise need.
    let mut lhs = compile_expr(ctx, lhs_expr, dst)?;
    ctx.goiftrue(&mut lhs);

    // LHS truthy path: evaluate RHS here. Any TESTSETs in lhs.true_list
    // represent paths whose values are discarded (RHS's value wins), so
    // `patch_to_here` downgrades them to plain TESTs.
    let lhs_true = mem::take(&mut lhs.true_list);
    ctx.patch_to_here(lhs_true);

    let mut rhs = compile_expr(ctx, rhs_expr, dst)?;

    rhs.false_list.concat(lhs.false_list);
    Ok(rhs)
}

/// Jump-list compilation of `lhs or rhs`. Symmetric to `and`: truthy `lhs`
/// short-circuits with `lhs`'s value (via `goiffalse`'s TESTSET), falsy
/// `lhs` falls through to RHS evaluation.
fn compile_logical_or_desc(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("or without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("or without rhs"))?;

    let mut lhs = compile_expr(ctx, lhs_expr, dst)?;
    ctx.goiffalse(&mut lhs);

    let lhs_false = mem::take(&mut lhs.false_list);
    ctx.patch_to_here(lhs_false);

    let mut rhs = compile_expr(ctx, rhs_expr, dst)?;

    rhs.true_list.concat(lhs.true_list);
    Ok(rhs)
}

/// Compile the target expression and arguments of a plain (non-method)
/// call into consecutive registers `func`, `func+1`, `func+2`, ... ready
/// for a `CALL` or `TAILCALL` instruction. Returns the function register
/// and argument count (excluding the function slot itself).
fn emit_func_call_setup(
    ctx: &mut Ctx,
    item: &FuncCall,
) -> Result<(RegisterIndex, usize), CompileError> {
    let target = item
        .target()
        .ok_or_else(|| ice("func call without target"))?;
    let func = compile_expr_to_reg(ctx, target, None)?;

    // Two situations force a copy of the function value to a fresh top
    // register before arg setup:
    //   1. Anything is already live at or above func+1 (typical when target
    //      resolves to a low local while higher temps/locals live above it).
    //      Without the copy, arg setup writes into func+1.. and clobbers
    //      that state.
    //   2. func is a register bound to a named local. CALL writes its result
    //      back to func, so leaving the local there destroys it; subsequent
    //      reads of the same name (including the next iteration of a chain
    //      like `f(f(x))`) would see the call result instead of the local.
    //
    // Fresh temps (GETUPVAL/GETTABUP/GETTABLE results, etc.) at the top of
    // stack are safe to overwrite since nothing else references them.
    let needs_copy = (func.0 + 1 != ctx.chunk.freereg) || func.0 < ctx.chunk.nactvar;
    let func = if needs_copy {
        let top = ctx.alloc_register()?;
        ctx.emit(Instruction::MOVE {
            dst: top.0,
            src: func.0,
        });
        top
    } else {
        func
    };

    let args: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let nargs = args.len();

    for (i, arg_expr) in args.into_iter().enumerate() {
        let expected_reg = RegisterIndex(func.0 + 1 + i as u8);
        let arg = compile_expr_to_reg(ctx, arg_expr, Some(expected_reg))?;
        if arg != expected_reg {
            while ctx.chunk.freereg <= expected_reg.0 {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: expected_reg.0,
                src: arg.0,
            });
        }
    }

    Ok((func, nargs))
}

fn compile_expr_func_call(
    ctx: &mut Ctx,
    item: FuncCall,
    want: usize,
) -> Result<Vec<RegisterIndex>, CompileError> {
    let (func, nargs) = emit_func_call_setup(ctx, &item)?;

    ctx.emit(Instruction::CALL {
        func: func.0,
        args: nargs as u8 + 1,
        returns: want as u8 + 1,
    });

    // Results are placed starting at func register; arg temps above
    // func+want are reclaimed (freereg snaps back to func+want). Preserves
    // `max_stack` since it already recorded the peak during arg setup.
    ctx.chunk.freereg = func.0 + want as u8;

    let mut results = Vec::with_capacity(want);
    for i in 0..want {
        results.push(RegisterIndex(func.0 + i as u8));
    }
    Ok(results)
}

/// Tail-call form of [`compile_expr_func_call`]: emits `TAILCALL` with
/// no trailing `RETURN` (the VM's TAILCALL is self-unwinding — the Lua
/// path replaces the current frame, the native path goes through
/// `frame_return`).
fn compile_tail_func_call(ctx: &mut Ctx, item: FuncCall) -> Result<(), CompileError> {
    let (func, nargs) = emit_func_call_setup(ctx, &item)?;
    ctx.emit(Instruction::TAILCALL {
        func: func.0,
        args: nargs as u8 + 1,
    });
    // Frame is about to be torn down; conservatively snap freereg to func
    // so any post-TAILCALL code in the compiler (there shouldn't be any
    // reachable) sees a clean cursor.
    ctx.chunk.freereg = func.0;
    Ok(())
}

/// Tail-call form of [`compile_expr_method_call`]. Same setup as the
/// CALL variant, but emits `TAILCALL` and does not produce a result
/// register list.
fn compile_tail_method_call(ctx: &mut Ctx, item: MethodCall) -> Result<(), CompileError> {
    let (func, nargs) = emit_method_call_setup(ctx, &item)?;
    ctx.emit(Instruction::TAILCALL {
        func: func.0,
        args: nargs as u8 + 1,
    });
    ctx.chunk.freereg = func.0;
    Ok(())
}

/// Set up a method call `obj:m(args)` in registers `func`, `func+1=self`,
/// `func+2..func+1+nargs` so a `CALL` or `TAILCALL` can be emitted with
/// that register as its function slot. Returns the function register and
/// total argument count (including the implicit `self`).
fn emit_method_call_setup(
    ctx: &mut Ctx,
    item: &MethodCall,
) -> Result<(RegisterIndex, usize), CompileError> {
    let object_expr = item
        .object()
        .ok_or_else(|| ice("method call without object"))?;
    let method_ident = item
        .method()
        .ok_or_else(|| ice("method call without method name"))?;
    let method_name = method_ident
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?;

    let object = compile_expr_to_reg(ctx, object_expr, None)?;

    let key_idx = ctx.alloc_string_constant(method_name.as_bytes())?;
    let key_reg = ctx.alloc_register()?;
    ctx.emit(Instruction::LOAD {
        dst: key_reg.0,
        idx: key_idx,
    });

    let func = ctx.alloc_register()?;
    ctx.emit(Instruction::GETTABLE {
        dst: func.0,
        table: object.0,
        key: key_reg.0,
    });

    let self_reg = RegisterIndex(func.0 + 1);
    while ctx.chunk.freereg <= self_reg.0 {
        ctx.alloc_register()?;
    }
    ctx.emit(Instruction::MOVE {
        dst: self_reg.0,
        src: object.0,
    });

    let args: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let nargs = args.len() + 1; // +1 for self

    for (i, arg_expr) in args.into_iter().enumerate() {
        let expected_reg = RegisterIndex(func.0 + 2 + i as u8);
        let arg = compile_expr_to_reg(ctx, arg_expr, Some(expected_reg))?;
        if arg != expected_reg {
            while ctx.chunk.freereg <= expected_reg.0 {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: expected_reg.0,
                src: arg.0,
            });
        }
    }

    Ok((func, nargs))
}

fn compile_expr_method_call(
    ctx: &mut Ctx,
    item: MethodCall,
    want: usize,
) -> Result<Vec<RegisterIndex>, CompileError> {
    let (func, nargs) = emit_method_call_setup(ctx, &item)?;

    ctx.emit(Instruction::CALL {
        func: func.0,
        args: nargs as u8 + 1,
        returns: want as u8 + 1,
    });

    // Results occupy [func, func+want); reclaim arg temps above.
    ctx.chunk.freereg = func.0 + want as u8;

    let mut results = Vec::with_capacity(want);
    for i in 0..want {
        results.push(RegisterIndex(func.0 + i as u8));
    }
    Ok(results)
}

fn compile_expr_index(
    ctx: &mut Ctx,
    item: Index,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let target_expr = item.target().ok_or_else(|| ice("index without target"))?;
    let key_expr = item.index().ok_or_else(|| ice("index without key"))?;
    let table = compile_expr_to_reg(ctx, target_expr, None)?;
    let key = compile_expr_to_reg(ctx, key_expr, None)?;
    // Both operands have been captured into the upcoming GETTABLE; free
    // their temps (higher first) so `dst` can reuse the slot.
    ctx.free_regs(table, key);
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::GETTABLE {
        dst: dst.0,
        table: table.0,
        key: key.0,
    });
    Ok(dst)
}

// ---------------------------------------------------------------------------
// Break / Return
// ---------------------------------------------------------------------------

fn compile_break(ctx: &mut Ctx, _item: Break) -> Result<(), CompileError> {
    let label = *ctx
        .control_end_label
        .last()
        .ok_or_else(|| ice("break outside of loop"))?;
    ctx.emit_jump(label);
    Ok(())
}

fn compile_return(ctx: &mut Ctx, item: Return) -> Result<(), CompileError> {
    let mut exprs: Vec<_> = item.exprs().map(|e| e.collect()).unwrap_or_default();

    // Tail-call optimisation: `return f(args)` / `return obj:m(args)` —
    // exactly one return expression, directly a call. Matches Lua 5.4's
    // tail-call rule.
    //
    // TODO: Lua 5.4 does NOT tail-call parenthesised `return (f())`
    // because the parens force adjust-to-one. The parser here doesn't
    // surface a distinct paren node, so we over-eagerly tail-call that
    // form for now.
    if exprs.len() == 1 {
        let only = exprs.pop().unwrap();
        match only {
            Expr::FuncCall(call) => return compile_tail_func_call(ctx, call),
            Expr::Method(call) => return compile_tail_method_call(ctx, call),
            other => exprs.push(other),
        }
    }

    compile_return_generic(ctx, exprs)
}

/// Compile a plain `return ...` — the fallback for any return that
/// doesn't qualify for `TAILCALL`.
fn compile_return_generic(ctx: &mut Ctx, mut exprs: Vec<Expr>) -> Result<(), CompileError> {
    if exprs.is_empty() {
        ctx.emit(Instruction::RETURN {
            values: 0,
            count: 1,
        });
        return Ok(());
    }

    // Single-value fast path: if the expression discharges to a plain Reg
    // with no pending jumps, RETURN directly from that register. Saves a
    // fresh reservation and a MOVE for the common `return local_var`
    // pattern. Jump-kind expressions (comparisons, short-circuit logic)
    // fall through to the contiguous-slot path since LFALSESKIP/LOAD-true
    // materialisation needs a concrete dst.
    if exprs.len() == 1 {
        let only = exprs.pop().unwrap();
        let mut desc = compile_expr(ctx, only, None)?;
        if !desc.has_jumps()
            && let ExprKind::Reg(reg) = desc.kind
        {
            ctx.emit(Instruction::RETURN {
                values: reg.0,
                count: 2,
            });
            return Ok(());
        }
        // Fall through to the generic path, discharging through a fresh
        // register slot so pending jumps get materialised correctly.
        let first_reg = ctx.alloc_register()?;
        let reg = ctx.discharge_to_reg_mut(&mut desc, Some(first_reg))?;
        if reg != first_reg {
            ctx.emit(Instruction::MOVE {
                dst: first_reg.0,
                src: reg.0,
            });
        }
        ctx.emit(Instruction::RETURN {
            values: first_reg.0,
            count: 2,
        });
        return Ok(());
    }

    let n = exprs.len();
    let first_reg = ctx.alloc_register()?;

    for (i, expr) in exprs.into_iter().enumerate() {
        let target = RegisterIndex(first_reg.0 + i as u8);
        let reg = compile_expr_to_reg(ctx, expr, Some(target))?;
        if reg != target {
            while ctx.chunk.freereg <= target.0 {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: target.0,
                src: reg.0,
            });
        }
    }

    ctx.emit(Instruction::RETURN {
        values: first_reg.0,
        count: n as u8 + 1,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

fn compile_do(ctx: &mut Ctx, item: Do) -> Result<(), CompileError> {
    scope_lexical(ctx, |ctx| {
        let stmts: Vec<_> = item.stmts().map(|s| s.collect()).unwrap_or_default();
        for stmt in stmts {
            compile_stmt(ctx, stmt)?;
        }
        Ok(())
    })
}

/// Compile `cond` as the condition of a branching construct. Returns the
/// list of jumps that should be patched to the branch-out target (i.e. the
/// "else" of an `if`, the "break" of a `while`, the loop-back of a
/// `repeat ... until`). The fall-through after this call is the path taken
/// when the condition evaluates to truthy.
fn compile_branch_cond_false(ctx: &mut Ctx, cond_expr: Expr) -> Result<JumpList, CompileError> {
    let mut desc = compile_expr(ctx, cond_expr, None)?;
    ctx.goiftrue(&mut desc);
    // After `goiftrue`, `false_list` holds every jump that fires on falsy
    // (the caller patches these to the branch-out target) and `true_list`
    // holds any jumps accumulated during LHS evaluation of an `and`/`or`
    // chain that need to land here at the truthy-fall-through position.
    let true_list = mem::take(&mut desc.true_list);
    ctx.patch_to_here(true_list);
    Ok(desc.false_list)
}

fn compile_while(ctx: &mut Ctx, item: While) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        let loop_start = ctx.new_label();
        ctx.set_label(loop_start, ctx.next_offset());

        let cond_expr = item.cond().ok_or_else(|| ice("while without condition"))?;
        let break_list = compile_branch_cond_false(ctx, cond_expr)?;

        // Compile body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        // Jump back to condition check
        ctx.emit_jump(loop_start);

        // Patch all "condition false" jumps to the post-loop break target.
        let break_label = *ctx
            .control_end_label
            .last()
            .ok_or_else(|| ice("missing break label"))?;
        // Jumps in `break_list` are branch-context: their targets get
        // resolved by the label-based `jump_patches` system at assemble
        // time, which only touches JMP offsets. Downgrade any TESTSETs
        // now (while we still have the list) so no `NO_REG` placeholder
        // makes it into the final chunk.
        ctx.downgrade_testsets(&break_list);
        for idx in break_list.jumps {
            ctx.chunk.jump_patches.push((idx, break_label));
        }

        Ok(())
    })
}

fn compile_repeat(ctx: &mut Ctx, item: Repeat) -> Result<(), CompileError> {
    scope_break(ctx, |ctx| {
        // Note: repeat-until has special scoping — the condition can see
        // locals defined in the body. So we wrap the body AND condition
        // in the same lexical scope.
        scope_lexical(ctx, |ctx| {
            let loop_start_off = ctx.next_offset();

            // Compile body
            let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }

            // Compile condition — jumps from false_list (condition falsy)
            // get patched back to the loop start. Truthy fall-through exits
            // the loop.
            let cond_expr = item.cond().ok_or_else(|| ice("repeat without condition"))?;
            let false_list = compile_branch_cond_false(ctx, cond_expr)?;
            ctx.patch_to(false_list, loop_start_off);

            Ok(())
        })
    })
}

fn compile_if(ctx: &mut Ctx, item: If) -> Result<(), CompileError> {
    let end_label = ctx.new_label();

    compile_if_chain(ctx, item, end_label)?;

    ctx.set_label(end_label, ctx.next_offset());
    Ok(())
}

fn compile_if_chain(ctx: &mut Ctx, item: If, end_label: u16) -> Result<(), CompileError> {
    let cond_expr = item.cond().ok_or_else(|| ice("if without condition"))?;
    let else_list = compile_branch_cond_false(ctx, cond_expr)?;

    // Then block
    scope_lexical(ctx, |ctx| {
        let stmts: Vec<_> = item.stmts().map(|s| s.collect()).unwrap_or_default();
        for stmt in stmts {
            compile_stmt(ctx, stmt)?;
        }
        Ok(())
    })?;

    // Only emit jump-to-end when there's an else/elseif (otherwise we just fall through)
    if item.else_chain().is_some() {
        ctx.emit_jump(end_label);
    }

    // Patch the "condition false" jumps to this point (start of else branch).
    ctx.patch_to_here(else_list);

    // Else chain
    if let Some(chain) = item.else_chain() {
        if let Some(elseif) = chain.elseif_block() {
            compile_if_chain(ctx, elseif, end_label)?;
        } else if let Some(else_stmts) = chain.else_block() {
            scope_lexical(ctx, |ctx| {
                let stmts: Vec<_> = else_stmts.collect();
                for stmt in stmts {
                    compile_stmt(ctx, stmt)?;
                }
                Ok(())
            })?;
        }
    }

    Ok(())
}

fn compile_for_num(ctx: &mut Ctx, item: ForNum) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        let (counter_ident, init_expr) = item
            .counter()
            .ok_or_else(|| ice("for_num without counter"))?;
        let counter_name = counter_ident
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?
            .to_owned();

        let limit_expr = item.end().ok_or_else(|| ice("for_num without limit"))?;

        // Pre-allocate consecutive registers: base, base+1, base+2
        let base = ctx.alloc_register()?;
        let limit_reg = ctx.alloc_register()?;
        let step_reg = ctx.alloc_register()?;

        // Compile init, limit, step with destination hints. Each sub-MOVE
        // path frees the source temp so freereg settles at step_reg+1
        // when this block finishes — the explicit reset is no longer
        // needed (see the assert below).
        let init = compile_expr_to_reg(ctx, init_expr, Some(base))?;
        if init != base {
            ctx.emit(Instruction::MOVE {
                dst: base.0,
                src: init.0,
            });
            ctx.free_reg(init);
        }

        let limit = compile_expr_to_reg(ctx, limit_expr, Some(limit_reg))?;
        if limit != limit_reg {
            ctx.emit(Instruction::MOVE {
                dst: limit_reg.0,
                src: limit.0,
            });
            ctx.free_reg(limit);
        }

        if let Some(step_expr) = item.step() {
            let step = compile_expr_to_reg(ctx, step_expr, Some(step_reg))?;
            if step != step_reg {
                ctx.emit(Instruction::MOVE {
                    dst: step_reg.0,
                    src: step.0,
                });
                ctx.free_reg(step);
            }
        } else {
            // Default step = 1
            let one_idx = ctx.alloc_constant(Value::Integer(1))?;
            ctx.emit(Instruction::LOAD {
                dst: step_reg.0,
                idx: one_idx,
            });
        }

        // FORPREP/FORLOOP hardcode the loop variable at base+3; with
        // freereg discipline wired into the subexpression pipeline, this
        // invariant holds without the former `= step_reg+1` reset.
        assert_eq!(ctx.chunk.freereg, step_reg.0 + 1);

        // Promote the 3 anonymous control slots to "protected locals" so
        // subexpressions inside the body can't reclaim them via free_reg.
        // They're unnamed so upvalue capture won't find them; nactvar's
        // sole role here is as the free_reg cutoff.
        ctx.adjust_locals(3);

        // base+3 is the visible loop variable
        let loop_var = ctx.alloc_register()?;
        assert_eq!(loop_var.0, base.0 + 3);
        ctx.define(
            counter_name,
            VariableData {
                register: loop_var,
                is_const: false,
            },
        )?;
        // Promote the loop variable to an active local so upvalue-capture
        // logic sees it and temp reclaims don't touch it.
        ctx.adjust_locals(1);

        let loop_body = ctx.new_label();
        let loop_end = ctx.new_label();

        // FORPREP: initialize and jump past body if loop shouldn't execute
        ctx.emit_jump_instr(
            loop_end,
            Instruction::FORPREP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_body, ctx.next_offset());

        // Body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        // FORLOOP: increment and jump back if still in range
        ctx.emit_jump_instr(
            loop_body,
            Instruction::FORLOOP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_end, ctx.next_offset());
        Ok(())
    })
}

fn compile_for_gen(ctx: &mut Ctx, item: ForGen) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        // Compile iterator expressions into base, base+1, base+2
        let values: Vec<_> = item
            .values()
            .ok_or_else(|| ice("for_gen without values"))?
            .collect();
        let targets: Vec<_> = item
            .targets()
            .ok_or_else(|| ice("for_gen without targets"))?
            .collect();
        let num_targets = targets.len();

        // We need 3 control registers + N target registers
        // R[base] = iterator function
        // R[base+1] = state
        // R[base+2] = initial control variable
        // R[base+3]..R[base+2+N] = loop variables
        let base = ctx.alloc_register()?;
        // Pre-allocate the remaining two control registers
        ctx.alloc_register()?; // base+1
        ctx.alloc_register()?; // base+2

        // Compile up to 3 iterator values with destination hints. Each
        // MOVE path frees its source temp so freereg settles at base+3
        // naturally.
        for (i, val_expr) in values.into_iter().enumerate().take(3) {
            let target_reg = RegisterIndex(base.0 + i as u8);
            let val = compile_expr_to_reg(ctx, val_expr, Some(target_reg))?;
            if val != target_reg {
                while ctx.chunk.freereg <= target_reg.0 {
                    ctx.alloc_register()?;
                }
                ctx.emit(Instruction::MOVE {
                    dst: target_reg.0,
                    src: val.0,
                });
                ctx.free_reg(val);
            }
        }

        // TFORCALL/TFORLOOP hardcode loop variables at base+3..base+2+count.
        // The per-expression MOVE frees above keep the cursor disciplined.
        assert_eq!(ctx.chunk.freereg, base.0 + 3);

        // Promote the 3 anonymous control slots (iterator, state, control)
        // so body subexpressions can't reclaim them via free_reg.
        ctx.adjust_locals(3);

        // Allocate registers for loop variables and bind them
        let num_loop_vars = targets.len();
        for (i, target_ident) in targets.into_iter().enumerate() {
            let name = target_ident
                .name(ctx.interner)
                .ok_or_else(|| ice("ident without name"))?
                .to_owned();
            let reg = ctx.alloc_register()?;
            assert_eq!(reg.0, base.0 + 3 + i as u8);
            ctx.define(
                name,
                VariableData {
                    register: reg,
                    is_const: false,
                },
            )?;
        }
        // Promote the loop variables to active locals.
        ctx.adjust_locals(num_loop_vars as u8);

        let loop_body = ctx.new_label();
        let loop_test = ctx.new_label();

        // TFORPREP: jump to the test
        ctx.emit_jump_instr(
            loop_test,
            Instruction::TFORPREP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_body, ctx.next_offset());

        // Body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        ctx.set_label(loop_test, ctx.next_offset());

        // TFORCALL: call iterator, results go to base+3..base+2+count
        ctx.emit(Instruction::TFORCALL {
            base: base.0,
            count: num_targets as u8,
        });

        // TFORLOOP: if control variable is not nil, jump back to body
        ctx.emit_jump_instr(
            loop_body,
            Instruction::TFORLOOP {
                base: base.0,
                offset: 0,
            },
        );

        Ok(())
    })
}
