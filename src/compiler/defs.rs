use crate::dmm::{Gc, Mutation};
use crate::env::{LuaString, Prototype, Value};
use crate::instruction::{Instruction, UpValueDescriptor};

/// Newtype for register indices, providing type safety over raw u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterIndex(pub u8);

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
