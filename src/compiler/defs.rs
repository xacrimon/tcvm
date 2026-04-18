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
#[derive(Debug, Clone)]
pub(super) struct ExprDesc {
    pub(super) kind: ExprKind,
    pub(super) true_list: JumpList,
    pub(super) false_list: JumpList,
    /// Truthiness of the fall-through path at the point where this
    /// expression's last instruction was emitted. Our comparison convention
    /// (`CMP inv=false` + JMP-on-falsy) makes fall-through the truthy
    /// outcome of a bare comparison, so this defaults to `true`. `not <e>`
    /// flips it — the same pending JMPs fire on the same runtime condition,
    /// but they now represent the opposite truthiness of the surrounding
    /// expression.
    ///
    /// `discharge_to_reg_mut` uses this to decide which list the routing
    /// JMP belongs in when materialising a Jump-kind expression.
    pub(super) fall_truthy: bool,
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
    /// by the pending `true_list` / `false_list` jumps plus `fall_truthy`.
    /// Produced by bare comparisons and propagated through `not` / `and` /
    /// `or` chains whose last operand is itself a comparison.
    Jump,
}

impl ExprDesc {
    pub(super) fn from_reg(reg: RegisterIndex) -> Self {
        ExprDesc {
            kind: ExprKind::Reg(reg),
            true_list: JumpList::new(),
            false_list: JumpList::new(),
            fall_truthy: true,
        }
    }

    pub(super) fn has_jumps(&self) -> bool {
        !self.true_list.is_empty() || !self.false_list.is_empty()
    }
}

/// Mutable accumulator used during compilation of a single function.
pub struct Chunk<'gc> {
    pub(super) tape: Vec<Instruction>,
    pub(super) constants: Vec<Value<'gc>>,
    pub(super) prototypes: Vec<Gc<'gc, Prototype<'gc>>>,
    pub(super) upvalue_desc: Vec<UpValueDescriptor>,
    pub(super) register_count: usize,
    pub(super) max_register_count: usize,
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
            register_count: 0,
            max_register_count: 0,
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
                max_stack_size: self.max_register_count as u8,
                num_upvalues,
                source: self.source,
            },
        )
    }
}
