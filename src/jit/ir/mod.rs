//! The SSA IR: instructions in CFG order, appended and never edited. A value
//! is its instruction's index. A pass is a copy of one graph into another
//! through the emitter (`emit`), which rewrites, numbers and tracks facts as
//! it appends; analyses read the finished input.
//!
//! One container holds two dialects (`ops::Dialect`): the builder emits
//! high-level ops, copy 1 lowers them, and everything after sees low-level
//! ops only.

pub(crate) mod access;
pub(crate) mod analyse;
pub(crate) mod arith;
pub(crate) mod compare;
pub(crate) mod copy;
pub(crate) mod emit;
pub(crate) mod fields;
pub(crate) mod forloop;
pub(crate) mod ops;
#[cfg(feature = "tcvm-dev")]
mod print;
mod rules;
pub(crate) mod sites;
pub(crate) mod table;
#[cfg(test)]
mod tests;
pub(crate) mod types;
mod units;
#[cfg(feature = "tcvm-dev")]
pub(crate) mod verify;

use ops::{Arg, Flags, Heaps, Op};
use types::{Fact, Rep};

macro_rules! idx {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        pub(crate) struct $name(pub(crate) u32);

        impl $name {
            pub(crate) const NONE: $name = $name(u32::MAX);

            #[inline]
            pub(crate) fn idx(self) -> usize {
                self.0 as usize
            }

            #[inline]
            pub(crate) fn is_none(self) -> bool {
                self.0 == u32::MAX
            }
        }
    };
}
idx!(Val);
idx!(Block);
idx!(Loop);

/// `a`, `b` and `c` are operands or immediates by the op's form
/// (`ops::OpInfo::args`); `aux` a small immediate by its `ops::Aux`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Inst {
    pub(crate) op: Op,
    /// The result's representation.
    pub(crate) rep: Rep,
    pub(crate) aux: u16,
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
}

const _: () = assert!(size_of::<Inst>() == 16);

impl Inst {
    #[inline]
    pub(crate) fn args(&self) -> [u32; 3] {
        [self.a, self.b, self.c]
    }

    /// A `K64` or `KF64`'s bits.
    #[inline]
    pub(crate) fn k64(&self) -> u64 {
        self.a as u64 | (self.b as u64) << 32
    }
}

/// A frame-state entry's register word: the frame slot holds the value
/// already.
pub(crate) const HOME: u32 = 1 << 31;

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) struct BlockFlags: u8 {
        /// Laid out after the hot chain.
        const COLD = 1;
        /// Entered by a callee's return, not a branch.
        const RESUME = 2;
        const LOOP_HEADER = 4;
        /// Ended by its terminator.
        const SEALED = 8;
    }
}

pub(crate) struct BlockData {
    /// Its instructions, `insts[start..end]`: parameters first, a terminator
    /// last.
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// Its predecessors, `preds[preds..preds + npreds]`: the forward ones,
    /// known when it opens, then the late ones (back edges, cold arms),
    /// `Block::NONE` until their terminators are emitted. A parameter's
    /// inputs are in this order.
    pub(crate) preds: u32,
    pub(crate) npreds: u32,
    /// How many of them are forward.
    pub(crate) nfwd: u32,
    pub(crate) idom: Block,
    /// A dominator further up, for `Graph::dom_at` in logarithmic time.
    jump: Block,
    /// Its depth in the dominator tree.
    pub(crate) depth: u32,
    /// The innermost loop it is in.
    pub(crate) lp: Loop,
    pub(crate) flags: BlockFlags,
}

pub(crate) struct LoopData {
    pub(crate) header: Block,
    pub(crate) parent: Loop,
    /// The heaps its body may write.
    pub(crate) writes: Heaps,
    /// The frame state at its entry, before its header's first instruction:
    /// where an exit moved in front of the loop leaves to.
    pub(crate) entry_state: Val,
}

/// The heap values a compile's graphs name by index (`KObj`), as raw words:
/// the region keeps them alive once installed, and nothing collects before.
#[derive(Default)]
pub(crate) struct Pool {
    pub(crate) words: Vec<u64>,
}

impl Pool {
    pub(crate) fn intern(&mut self, w: u64) -> u32 {
        let at = self.words.iter().position(|&x| x == w).unwrap_or_else(|| {
            self.words.push(w);
            self.words.len() - 1
        });
        at as u32
    }
}

/// Everything is appended; a graph is recycled by `reset`.
#[derive(Default)]
pub(crate) struct Graph {
    pub(crate) insts: Vec<Inst>,
    /// Each instruction's definition fact.
    pub(crate) facts: Vec<Fact>,
    /// Operand runs of the ops with more than three: parameter inputs and
    /// frame-state entries.
    pub(crate) extra: Vec<u32>,
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) preds: Vec<Block>,
    pub(crate) loops: Vec<LoopData>,
    /// The opened blocks in emission order: a block comes after its forward
    /// predecessors.
    pub(crate) order: Vec<Block>,
}

impl Graph {
    pub(crate) fn reset(&mut self) {
        self.insts.clear();
        self.facts.clear();
        self.extra.clear();
        self.blocks.clear();
        self.preds.clear();
        self.loops.clear();
        self.order.clear();
    }

    #[inline]
    pub(crate) fn inst(&self, v: Val) -> &Inst {
        &self.insts[v.idx()]
    }

    #[inline]
    pub(crate) fn op(&self, v: Val) -> Op {
        self.insts[v.idx()].op
    }

    #[inline]
    pub(crate) fn rep(&self, v: Val) -> Rep {
        self.insts[v.idx()].rep
    }

    #[inline]
    pub(crate) fn block(&self, b: Block) -> &BlockData {
        &self.blocks[b.idx()]
    }

    pub(crate) fn preds(&self, b: Block) -> &[Block] {
        let d = self.block(b);
        &self.preds[d.preds as usize..(d.preds + d.npreds) as usize]
    }

    pub(crate) fn run(&self, at: u32, len: u32) -> &[u32] {
        &self.extra[at as usize..(at + len) as usize]
    }

    /// The extra words of an instruction that has a run, else none.
    pub(crate) fn run_of(&self, i: &Inst) -> &[u32] {
        match i.op.info().args[0] {
            Arg::Vals => self.run(i.a, i.b),
            Arg::Entries => self.run(i.a, 1 + 2 * i.b),
            Arg::No
            | Arg::Imm
            | Arg::Val
            | Arg::I32
            | Arg::I64
            | Arg::F64
            | Arg::B1
            | Arg::Ptr
            | Arg::Int
            | Arg::IntB
            | Arg::W64
            | Arg::Same
            | Arg::Any
            | Arg::Fs
            | Arg::OptFs
            | Arg::Opt
            | Arg::Blk
            | Arg::Len => &[],
        }
    }

    /// A parameter's inputs, by predecessor: `Val::NONE` for a late
    /// predecessor not emitted yet.
    pub(crate) fn inputs(&self, param: Val) -> &[u32] {
        let i = self.inst(param);
        debug_assert!(i.op == Op::Param);
        self.run(i.a, i.b)
    }

    /// A frame state's pc and (register, value) entries.
    pub(crate) fn entries(&self, fs: Val) -> (u32, &[[u32; 2]]) {
        let i = self.inst(fs);
        debug_assert!(i.op == Op::FrameState);
        let (pc, rest) = self.run(i.a, 1 + 2 * i.b).split_first().unwrap();
        (*pc, rest.as_chunks().0)
    }

    /// The block a sealed block's terminator is in is the block itself: its
    /// last instruction.
    pub(crate) fn term(&self, b: Block) -> &Inst {
        let d = self.block(b);
        debug_assert!(d.flags.contains(BlockFlags::SEALED));
        &self.insts[d.end as usize - 1]
    }

    /// A terminator's successors.
    pub(crate) fn succs(&self, t: &Inst) -> ([Block; 2], usize) {
        let mut out = ([Block::NONE; 2], 0);
        for (k, kind) in t.op.info().args.iter().enumerate() {
            if *kind == Arg::Blk {
                out.0[out.1] = Block(t.args()[k]);
                out.1 += 1;
            }
        }
        out
    }

    /// Every value an instruction uses: operands, a parameter's inputs, a
    /// frame state's entries.
    #[inline]
    pub(crate) fn uses(&self, i: &Inst, mut f: impl FnMut(Val)) {
        for (k, kind) in i.op.info().args.iter().enumerate() {
            let w = i.args()[k];
            match kind {
                Arg::No | Arg::Imm | Arg::Blk | Arg::Len => {}
                Arg::Val
                | Arg::I32
                | Arg::I64
                | Arg::F64
                | Arg::B1
                | Arg::Ptr
                | Arg::Int
                | Arg::IntB
                | Arg::W64
                | Arg::Same
                | Arg::Any
                | Arg::Fs => f(Val(w)),
                Arg::Opt | Arg::OptFs => {
                    if w != u32::MAX {
                        f(Val(w))
                    }
                }
                Arg::Vals => {
                    for &v in self.run(i.a, i.b) {
                        if v != u32::MAX {
                            f(Val(v))
                        }
                    }
                }
                Arg::Entries => {
                    for e in self.run(i.a + 1, 2 * i.b).as_chunks::<2>().0 {
                        f(Val(e[1]))
                    }
                }
            }
        }
    }

    /// The frame state an exiting instruction takes.
    pub(crate) fn frame_state(&self, i: &Inst) -> Val {
        let k = (i.op.info().args.iter())
            .position(|a| matches!(a, Arg::Fs | Arg::OptFs))
            .unwrap_or_else(|| crate::jit::bug!("{:?} takes no frame state", i.op));
        Val(i.args()[k])
    }

    /// Whether an instruction's operands and result fit its op's row
    /// (`ops::OPS`), given the instructions before it.
    pub(crate) fn form(&self, i: &Inst) -> Result<(), &'static str> {
        let info = i.op.info();
        let n = self.insts.len() as u32;
        let rep_of = |w: u32| self.insts.get(w as usize).map_or(Rep::None, |d| d.rep);
        // 32, 64, or 1 for `B1`.
        let mut width = 0;
        for (k, kind) in info.args.iter().enumerate() {
            let w = i.args()[k];
            let ok = match kind {
                Arg::No | Arg::Imm | Arg::Len => true,
                Arg::Blk => (w as usize) < self.blocks.len(),
                Arg::Val => rep_of(w) == Rep::Val,
                Arg::I32 => rep_of(w) == Rep::I32,
                Arg::I64 => rep_of(w) == Rep::I64,
                Arg::F64 => rep_of(w) == Rep::F64,
                Arg::B1 => rep_of(w) == Rep::B1,
                Arg::Ptr => rep_of(w) == Rep::Ptr,
                Arg::W64 => rep_of(w).word64(),
                Arg::Same => rep_of(w) == i.rep,
                Arg::Any => rep_of(w) != Rep::None,
                Arg::Opt => w == u32::MAX || rep_of(w) != Rep::None,
                Arg::Fs | Arg::OptFs => {
                    (w == u32::MAX && *kind == Arg::OptFs)
                        || (w < n && self.insts[w as usize].op == Op::FrameState)
                }
                Arg::Int | Arg::IntB => {
                    let here = match rep_of(w) {
                        Rep::I32 => 32,
                        Rep::B1 if *kind == Arg::IntB => 1,
                        r if r.word64() => 64,
                        _ => 0,
                    };
                    let first = std::mem::replace(&mut width, here);
                    here != 0 && (first == 0 || first == here)
                }
                Arg::Vals | Arg::Entries => {
                    let len = if *kind == Arg::Vals { i.b } else { 1 + 2 * i.b };
                    (w as usize + len as usize) <= self.extra.len()
                }
            };
            if !ok {
                return Err(["operand a", "operand b", "operand c"][k]);
            }
        }
        let set = types::TypeSet::from_bits_retain(i.aux & types::TypeSet::ANY.bits());
        let ok = match info.res {
            ops::Res::Is(r) => i.rep == r,
            ops::Res::Int => match width {
                32 => i.rep == Rep::I32,
                64 => i.rep == Rep::I64,
                _ => i.rep == Rep::B1,
            },
            ops::Res::Pass => {
                i.rep
                    == if i.c == u32::MAX {
                        Rep::None
                    } else {
                        rep_of(i.c)
                    }
            }
            ops::Res::K32 => matches!(i.rep, Rep::I32 | Rep::B1),
            ops::Res::Word => i.rep.word64(),
            ops::Res::Obj => matches!(i.rep, Rep::Val | Rep::Ptr),
            ops::Res::Field => {
                (i.aux as usize) < fields::FIELDS.len() && i.rep == fields::Field::info(i.aux).rep
            }
            ops::Res::Some => i.rep != Rep::None,
            ops::Res::Helper => {
                (i.aux as usize) < ops::HELPERS.len() && {
                    let h = &ops::HELPERS[i.aux as usize];
                    let reps = self.run(i.a, i.b).iter().map(|&v| rep_of(v));
                    i.rep == h.ret && reps.eq(h.operands())
                }
            }
            ops::Res::Unboxed => i.rep == set.unboxed(),
        };
        if !ok {
            return Err("result");
        }
        let any = types::TypeSet::ANY.bits();
        let ok = match info.aux {
            // A generic op keeps its operands' speculated kinds above.
            ops::Aux::Cc => (i.aux as usize & 15) < 10,
            ops::Aux::FCc => (i.aux as usize) < 6,
            ops::Aux::Set => i.aux & !any == 0,
            ops::Aux::SetArm => i.aux & !(any | ops::SET_ARM) == 0,
            ops::Aux::Field => (i.aux as usize) < fields::FIELDS.len(),
            ops::Aux::Helper => (i.aux as usize) < ops::HELPERS.len(),
            ops::Aux::Exit | ops::Aux::Array | ops::Aux::Loop => i.aux < 2,
            // A tag's, a slot's and a kind's own types check theirs.
            ops::Aux::No | ops::Aux::SetTag | ops::Aux::Tag | ops::Aux::Slot | ops::Aux::Kind => {
                true
            }
            // Its table checks it, where the rewrite reads the record.
            ops::Aux::Site => true,
        };
        if !ok {
            return Err("aux");
        }
        // The `I32` forms of the Lua shifts can leave `I32`: they take a
        // frame state.
        if matches!(i.op, Op::LuaShl | Op::LuaShr) && i.rep != Rep::I64 && i.c == u32::MAX {
            return Err("operand c");
        }
        Ok(())
    }

    /// The dominator of `b` at depth `depth`, at most `b`'s.
    pub(crate) fn dom_at(&self, mut b: Block, depth: u32) -> Block {
        while self.block(b).depth > depth {
            let d = self.block(b);
            b = if self.block(d.jump).depth >= depth {
                d.jump
            } else {
                d.idom
            };
        }
        b
    }

    pub(crate) fn dominates(&self, a: Block, b: Block) -> bool {
        let da = self.block(a).depth;
        da <= self.block(b).depth && self.dom_at(b, da) == a
    }

    pub(crate) fn common_dom(&self, mut a: Block, mut b: Block) -> Block {
        let depth = self.block(a).depth.min(self.block(b).depth);
        (a, b) = (self.dom_at(a, depth), self.dom_at(b, depth));
        // At one depth their jumps are equally far up.
        while a != b {
            let (x, y) = (self.block(a), self.block(b));
            (a, b) = if x.jump != y.jump {
                (x.jump, y.jump)
            } else {
                (x.idom, y.idom)
            };
        }
        a
    }

    /// Whether loop `l` is `outer` or inside it.
    pub(crate) fn loop_within(&self, mut l: Loop, outer: Loop) -> bool {
        while !l.is_none() {
            if l == outer {
                return true;
            }
            l = self.loops[l.idx()].parent;
        }
        outer.is_none()
    }

    pub(crate) fn is_terminator(&self, v: Val) -> bool {
        self.op(v).has(Flags::TERMINATOR)
    }
}
