use crate::dmm::{Gc, Mutation};
use crate::env::{LuaString, Prototype, Value};
use crate::instruction::{Instruction, UpValueDescriptor};

/// Newtype for register indices, providing type safety over raw u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterIndex(pub u8);

/// A list of unfilled `JMP` (or conditional-fall-through `JMP`) instructions
/// in the tape whose offset fields still need to be patched to a target. The
/// two lists attached to an `ExprDesc` represent the expression's "true" and
/// "false" exit paths — jumps in `true_list` are taken when the expression
/// evaluates to a truthy value, jumps in `false_list` fire when falsy.
///
/// **Invariant (control-instruction predecessor):** every `jmp_idx` stored in
/// `jumps` must satisfy either `jmp_idx == 0` (the sentinel tolerated by every
/// consumer) or `tape[jmp_idx - 1]` is one of `EQ`, `LT`, `LE`, `TEST`,
/// `TESTSET` — the control instruction whose taken-vs-fall-through decision
/// the `JMP` at `tape[jmp_idx]` realises. The consumers `need_value`,
/// `downgrade_testsets`, `patch_list_aux`, and `flip_control_polarity` all
/// read `tape[jmp_idx - 1]` under this assumption. Only `emit_test_jump`
/// (TESTSET + JMP) and `compile_comparison_desc` (CMP + JMP) mint jump indices
/// that may enter a list; pushing any other JMP here would silently miscompile.
#[derive(Debug, Clone, Default)]
pub(super) struct JumpList {
    pub(super) jumps: Vec<usize>,
}

impl JumpList {
    pub(super) fn new() -> Self {
        JumpList { jumps: Vec::new() }
    }

    pub(super) fn single(idx: usize) -> Self {
        JumpList { jumps: vec![idx] }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.jumps.is_empty()
    }

    /// Consume `other`, appending its jumps onto `self`.
    pub(super) fn concat(&mut self, mut other: JumpList) {
        self.jumps.append(&mut other.jumps);
    }
}

/// Where an expression's value currently lives during compilation, with any
/// pending short-circuit jumps that will be patched by the consumer.
///
/// **Invariants** (matching Lua 5.5's `expdesc` conventions):
/// * A `JMP` in `true_list` fires when the expression is truthy.
/// * A `JMP` in `false_list` fires when the expression is falsy.
/// * A `Jump(Some(idx))` head is the last-emitted control jump for this
///   expression; it fires on truthy and its fall-through is falsy.
/// * Comparisons emit `CMP` with `inverted=true` so the paired `JMP` fires
///   on truthy — fall-through is falsy. This mirrors Lua's convention and
///   lets the `LFALSESKIP` / `LOAD true` materialisation tail handle
///   fall-through correctly without a routing jump.
#[derive(Debug, Clone)]
pub(super) struct ExprDesc {
    pub(super) kind: ExprKind,
    pub(super) true_list: JumpList,
    pub(super) false_list: JumpList,
}

/// The shape of an expression's current representation. Comparisons, `not`
/// on comparisons, and short-circuit `and`/`or` whose tail operand is a
/// comparison all stay as `Jump` until the consumer decides how to use the
/// result — branches patch the lists directly, value contexts turn them
/// back into `Reg` via `discharge_to_reg_mut`.
#[derive(Debug, Clone, Copy)]
pub(super) enum ExprKind {
    /// Value already sits in this register. `ExprDesc` may still carry
    /// pending jumps (e.g. short-circuit paths from `and`/`or` whose
    /// fall-through put the value here); the consumer must resolve them.
    Reg(RegisterIndex),
    /// No register yet — the expression's truthiness is encoded entirely
    /// by the pending jumps (`true_list` / `false_list` plus the optional
    /// pending head embedded here). The inner `Option<usize>` is the
    /// "pending" head: when `Some(idx)`, the `JMP` at `tape[idx]` has not
    /// yet been absorbed into either list; `goiftrue` / `goiffalse` /
    /// `discharge_to_reg_mut` absorb it; `not` flips its CMP polarity in
    /// place so the invariant "pending fires on current truthy" continues
    /// to hold across the label flip. The `None` state is reached after
    /// a `goif*` consumes the head; it represents a mixed-polarity list
    /// composition (from `and`/`or`) with no standalone tail jump.
    Jump(Option<usize>),
}

impl ExprDesc {
    pub(super) fn from_reg(reg: RegisterIndex) -> Self {
        ExprDesc {
            kind: ExprKind::Reg(reg),
            true_list: JumpList::new(),
            false_list: JumpList::new(),
        }
    }

    pub(super) fn has_jumps(&self) -> bool {
        !self.true_list.is_empty()
            || !self.false_list.is_empty()
            || matches!(self.kind, ExprKind::Jump(Some(_)))
    }

    /// Peek at the Jump-kind pending head without consuming it. Returns
    /// `None` for non-Jump kinds.
    pub(super) fn pending(&self) -> Option<usize> {
        if let ExprKind::Jump(p) = self.kind {
            p
        } else {
            None
        }
    }

    /// Take the Jump-kind pending head, clearing the slot. Returns `None`
    /// (and leaves the expression untouched) for non-Jump kinds.
    pub(super) fn take_pending(&mut self) -> Option<usize> {
        if let ExprKind::Jump(ref mut p) = self.kind {
            p.take()
        } else {
            None
        }
    }
}

/// Mutable accumulator used during compilation of a single function.
pub struct Chunk<'gc> {
    pub(super) tape: Vec<Instruction>,
    pub(super) constants: Vec<Value<'gc>>,
    pub(super) prototypes: Vec<Gc<'gc, Prototype<'gc>>>,
    pub(super) upvalue_desc: Vec<UpValueDescriptor>,
    /// Next free register slot; cursor for temp allocation. Locals occupy
    /// `[0, nactvar)`, temps occupy `[nactvar, freereg)`. Matches Lua 5.5's
    /// `fs->freereg` semantics.
    pub(super) freereg: u8,
    /// Number of active named locals. Registers `[0, nactvar)` are reserved
    /// for bound locals and must not be reclaimed before scope exit —
    /// upvalue descriptors (see `UpValueDescriptor::ParentLocal`) embed
    /// these register numbers and rely on their stability.
    pub(super) nactvar: u8,
    /// Peak register count seen during compilation; becomes the prototype's
    /// `max_stack_size`.
    pub(super) max_stack: u8,
    pub(super) arity: u8,
    pub(super) is_vararg: bool,
    pub(super) labels: Vec<usize>,
    pub(super) jump_patches: Vec<(usize, u16)>,
    pub(super) source: Option<LuaString<'gc>>,
}

impl<'gc> Chunk<'gc> {
    pub fn new() -> Self {
        Chunk {
            tape: Vec::new(),
            constants: Vec::new(),
            prototypes: Vec::new(),
            upvalue_desc: Vec::new(),
            freereg: 0,
            nactvar: 0,
            max_stack: 0,
            arity: 0,
            is_vararg: false,
            labels: Vec::new(),
            jump_patches: Vec::new(),
            source: None,
        }
    }

    /// Resolve jump patches and convert into an immutable Prototype.
    pub fn assemble(mut self, mc: &Mutation<'gc>) -> Gc<'gc, Prototype<'gc>> {
        // Resolve all jump patches
        for &(instr_idx, label_idx) in &self.jump_patches {
            let target = self.labels[label_idx as usize];
            let offset = target as i32 - (instr_idx as i32 + 1);
            match &mut self.tape[instr_idx] {
                Instruction::JMP { offset: o } => *o = offset,
                Instruction::FORPREP { offset: o, .. } => *o = offset,
                Instruction::FORLOOP { offset: o, .. } => *o = offset,
                Instruction::TFORPREP { offset: o, .. } => *o = offset,
                Instruction::TFORLOOP { offset: o, .. } => *o = offset,
                _ => panic!("jump patch on non-jump instruction"),
            }
        }

        let num_upvalues = self.upvalue_desc.len() as u8;

        Gc::new(
            mc,
            Prototype {
                code: self.tape.into_boxed_slice(),
                constants: self.constants.into_boxed_slice(),
                prototypes: self.prototypes.into_boxed_slice(),
                upvalue_desc: self.upvalue_desc.into_boxed_slice(),
                num_params: self.arity,
                is_vararg: self.is_vararg,
                max_stack_size: self.max_stack,
                num_upvalues,
                source: self.source,
            },
        )
    }
}
