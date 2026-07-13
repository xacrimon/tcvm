//! The SSA IR: entities, instructions, blocks, and the deopt metadata that
//! hangs off them.
//!
//! Blocks take **parameters** rather than carrying phi nodes. That is not a
//! stylistic choice: a block's entry type context — the versioning key for
//! basic-block versioning — is then literally the types of its parameters, so
//! version lookup is a hash of the param types with no side table to maintain.

pub mod op;
pub mod pool;
pub mod print;
pub mod ty;
pub mod verify;

use crate::jit::ir::op::{Effects, Op};
use crate::jit::ir::pool::ConstPool;
use crate::jit::ir::ty::{Ty, TypeContext};

macro_rules! entity {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

entity!(
    /// An SSA value: either an instruction result or a block parameter.
    Val
);
entity!(Inst);
entity!(Block);
entity!(
    /// Index of a `FrameState`.
    FsRef
);
entity!(
    /// Index of an `Exit`.
    ExitRef
);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Def {
    Inst(Inst),
    /// The nth parameter of a block.
    Param(Block, u32),
}

#[derive(Clone, Copy, Debug)]
pub struct ValData {
    pub ty: Ty,
    pub def: Def,
}

/// A branch target plus the arguments passed to its parameters.
#[derive(Clone, Debug)]
pub struct BlockCall {
    pub block: Block,
    pub args: Vec<Val>,
}

#[derive(Clone, Debug)]
pub struct InstData {
    pub op: Op,
    pub args: Vec<Val>,
    /// Empty for non-terminators; one entry for `Jump`, two for `Br`.
    pub targets: Vec<BlockCall>,
    /// Most ops produce one value; `Call` produces `nret` results plus a
    /// status.
    pub results: Vec<Val>,
    /// Present iff `op.needs_frame_state()`.
    pub fs: Option<FsRef>,
    /// Present iff `op.is_guard()` or the op is a `Deopt`.
    pub exit: Option<ExitRef>,
}

impl InstData {
    pub fn effects(&self) -> Effects {
        self.op.effects()
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockData {
    pub params: Vec<Val>,
    pub insts: Vec<Inst>,
    /// The bytecode pc this block was compiled from, and the entry context it
    /// was specialized for. Together these are the block's version identity.
    pub origin: Option<Origin>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Origin {
    pub pc: u32,
    pub ctx: TypeContext,
}

/// The abstract interpreter state at a program point: everything needed to
/// rebuild a `LuaFrame` the interpreter will accept.
///
/// Varargs and the below-base region need no representation — the region never
/// writes them, so they sit on the stack untouched.
#[derive(Clone, Debug)]
pub struct FrameState {
    /// Bytecode pc to resume at.
    pub pc: u32,
    /// One entry per Lua register; `None` means dead here.
    pub regs: Vec<Option<Val>>,
    /// Always `None` in v1. Inlining fills this in, and the deopt writer
    /// already walks the chain, so that stays an additive change.
    pub parent: Option<FsRef>,
}

/// Why compiled code left, and what the recompiler should learn from it.
#[derive(Clone, Debug)]
pub struct Exit {
    pub fs: FsRef,
    /// The entry context in force here. A hot exit means this context was
    /// wrong; recompilation widens it rather than re-specializing identically.
    pub ctx: TypeContext,
    /// Bumped by the deopt handler. Crossing a threshold triggers recompile
    /// with a widened context — the guard against deopt thrash on a
    /// polymorphic site, which eager (non-lazy) versioning is otherwise prone
    /// to.
    pub count: u32,
}

pub struct Func<'gc> {
    blocks: Vec<BlockData>,
    insts: Vec<InstData>,
    values: Vec<ValData>,
    states: Vec<FrameState>,
    exits: Vec<Exit>,
    pub pool: ConstPool<'gc>,
    pub entry: Block,
    /// Registers captured as `ParentLocal` by a `CLOSURE` in this prototype.
    /// These are pinned to memory: an open upvalue can observe the slot at any
    /// time, so they are read and written via `StackGet`/`StackSet` rather
    /// than living in SSA.
    pub pinned_regs: Vec<u8>,
}

impl<'gc> Func<'gc> {
    pub fn new() -> Self {
        let mut f = Func {
            blocks: Vec::new(),
            insts: Vec::new(),
            values: Vec::new(),
            states: Vec::new(),
            exits: Vec::new(),
            pool: ConstPool::new(),
            entry: Block(0),
            pinned_regs: Vec::new(),
        };
        f.entry = f.new_block();
        f
    }

    pub fn new_block(&mut self) -> Block {
        let b = Block(self.blocks.len() as u32);
        self.blocks.push(BlockData::default());
        b
    }

    pub fn append_param(&mut self, block: Block, ty: Ty) -> Val {
        let n = self.blocks[block.index()].params.len() as u32;
        let v = self.new_value(ValData {
            ty,
            def: Def::Param(block, n),
        });
        self.blocks[block.index()].params.push(v);
        v
    }

    /// Append an instruction, minting one result value per entry in `result_tys`.
    pub fn append_inst(
        &mut self,
        block: Block,
        mut data: InstData,
        result_tys: &[Ty],
    ) -> (Inst, Vec<Val>) {
        debug_assert!(
            data.fs.is_some() || !data.op.needs_frame_state(),
            "{:?} requires a FrameState",
            data.op
        );

        let inst = Inst(self.insts.len() as u32);
        let results: Vec<Val> = result_tys
            .iter()
            .map(|&ty| {
                self.new_value(ValData {
                    ty,
                    def: Def::Inst(inst),
                })
            })
            .collect();
        data.results = results.clone();
        self.insts.push(data);
        self.blocks[block.index()].insts.push(inst);
        (inst, results)
    }

    fn new_value(&mut self, data: ValData) -> Val {
        let v = Val(self.values.len() as u32);
        self.values.push(data);
        v
    }

    pub fn add_frame_state(&mut self, fs: FrameState) -> FsRef {
        let r = FsRef(self.states.len() as u32);
        self.states.push(fs);
        r
    }

    pub fn add_exit(&mut self, exit: Exit) -> ExitRef {
        let r = ExitRef(self.exits.len() as u32);
        self.exits.push(exit);
        r
    }

    pub fn ty(&self, v: Val) -> Ty {
        self.values[v.index()].ty
    }

    pub fn def(&self, v: Val) -> Def {
        self.values[v.index()].def
    }

    /// Narrow a value's type in place. Only legal on a value a guard just
    /// produced — never to retype an existing definition, which would let a
    /// fact flow backwards to uses that predate the guard.
    pub fn set_ty(&mut self, v: Val, ty: Ty) {
        self.values[v.index()].ty = ty;
    }

    pub fn inst(&self, i: Inst) -> &InstData {
        &self.insts[i.index()]
    }

    pub fn inst_mut(&mut self, i: Inst) -> &mut InstData {
        &mut self.insts[i.index()]
    }

    pub fn block(&self, b: Block) -> &BlockData {
        &self.blocks[b.index()]
    }

    pub fn block_mut(&mut self, b: Block) -> &mut BlockData {
        &mut self.blocks[b.index()]
    }

    pub fn frame_state(&self, r: FsRef) -> &FrameState {
        &self.states[r.index()]
    }

    pub fn exit(&self, r: ExitRef) -> &Exit {
        &self.exits[r.index()]
    }

    pub fn exit_mut(&mut self, r: ExitRef) -> &mut Exit {
        &mut self.exits[r.index()]
    }

    pub fn blocks(&self) -> impl Iterator<Item = Block> {
        (0..self.blocks.len() as u32).map(Block)
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_insts(&self) -> usize {
        self.insts.len()
    }

    pub fn num_frame_states(&self) -> usize {
        self.states.len()
    }

    pub fn num_exits(&self) -> usize {
        self.exits.len()
    }
}

impl<'gc> Default for Func<'gc> {
    fn default() -> Self {
        Self::new()
    }
}
