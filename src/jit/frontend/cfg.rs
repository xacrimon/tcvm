//! Bytecode CFG and register liveness.
//!
//! Two jobs. First, carve the bytecode into basic blocks — which is not quite
//! trivial, because Lua's compare and test opcodes *conditionally skip the next
//! instruction*, and that instruction is always a `JMP`. So a compare and the
//! `JMP` after it are jointly one two-way branch, not two instructions.
//!
//! Second, compute live-in registers per block. A compiled block's parameters
//! are exactly its live-in registers: without liveness we would pass every
//! register in `max_stack_size` as a parameter, most of them dead, and the type
//! context we version on would be full of noise.

use std::collections::HashMap;

use foldhash::fast::RandomState;

use crate::instruction::{Instruction, Op};

/// A construct we've chosen not to compile. Since guard failure deopts to the
/// interpreter anyway, refusing to compile is always a legal answer — the
/// region simply stays interpreted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// `CALL`/`TAILCALL`/`RETURN`/`VARARG`/`SETLIST` with the `0` sentinel.
    /// Needs a `vals` pseudo-type and stack-window ops.
    MultRet(u32),
    /// `VARARGPREP`/`VARARG`: a vararg function.
    Vararg(u32),
    /// To-be-closed slots interact with error unwinding in ways not yet modelled.
    Tbc(u32),
    /// A jump target outside the code array — should be impossible.
    BadJump(u32),
}

#[derive(Clone, Debug)]
pub struct BcBlock {
    /// First pc in the block.
    pub start: u32,
    /// One past the last pc.
    pub end: u32,
    /// Successor leader pcs. Empty for a returning block. For a two-way
    /// branch, `[0]` is the taken-skip edge and `[1]` the jump edge — see
    /// `terminator`.
    pub succs: Vec<u32>,
    /// Live-in registers, ascending. These become the block's IR parameters.
    pub live_in: Vec<u8>,
    /// Live-in registers that some predecessor's terminator assigns on one edge
    /// only, so they hold a different value per incoming edge. Ascending, and a
    /// subset of `live_in`. See [`edge_defs`].
    pub edge_params: Vec<u8>,
}

/// How a block ends. `Branch` covers a compare/test paired with its trailing
/// `JMP`: `skip` is where control goes when the compare *skips* the `JMP`,
/// `jump` where it goes when the `JMP` executes.
#[derive(Clone, Copy, Debug)]
pub enum Term {
    Fallthrough(u32),
    Jump(u32),
    Branch { cond: u32, skip: u32, jump: u32 },
    Return,
}

#[derive(Debug)]
pub struct Cfg {
    pub blocks: Vec<BcBlock>,
    /// Leader pc -> index into `blocks`.
    pub index_of: HashMap<u32, usize, RandomState>,
}

impl Cfg {
    pub fn block_at(&self, pc: u32) -> &BcBlock {
        &self.blocks[self.index_of[&pc]]
    }
}

/// A `JMP`'s destination. The interpreter advances `ip` past the instruction
/// before applying the offset, so the target is `pc + 1 + offset`.
fn jump_target(pc: u32, offset: i32) -> i64 {
    pc as i64 + 1 + offset as i64
}

/// True if this opcode conditionally skips the following instruction (which the
/// compiler always emits as a `JMP`).
fn is_skip_test(i: Instruction) -> bool {
    i.is_control()
}

pub fn terminator(code: &[Instruction], pc: u32) -> Result<Term, Unsupported> {
    let i = code[pc as usize];
    let next = pc + 1;

    let target = |off: i32| -> Result<u32, Unsupported> {
        let t = jump_target(pc, off);
        if t < 0 || t as usize >= code.len() {
            return Err(Unsupported::BadJump(pc));
        }
        Ok(t as u32)
    };

    Ok(match i.op() {
        Op::JMP => Term::Jump(target(i.imm())?),

        // The compare skips the `JMP` at pc+1 when it succeeds, so the two
        // edges are pc+2 and that `JMP`'s target.
        _ if is_skip_test(i) => {
            let Some(jmp) = code
                .get(next as usize)
                .copied()
                .filter(|j| j.op() == Op::JMP)
            else {
                // The compiler always pairs these; anything else is a bytecode
                // we don't understand, so decline the region.
                return Err(Unsupported::BadJump(pc));
            };
            Term::Branch {
                cond: pc,
                skip: pc + 2,
                jump: jump_target(next, jmp.imm()) as u32,
            }
        }
        // `LFALSESKIP` sets false and unconditionally skips the next
        // instruction — a jump, not a branch.
        Op::LFALSESKIP => Term::Jump(pc + 2),

        Op::RETURN => {
            if i.b() == 0 {
                return Err(Unsupported::MultRet(pc));
            }
            Term::Return
        }
        Op::TAILCALL => {
            if i.b() == 0 {
                return Err(Unsupported::MultRet(pc));
            }
            Term::Return
        }
        Op::STOP => Term::Return,

        // Jumps past the body when the loop shouldn't run at all; otherwise
        // falls into it.
        Op::FORPREP => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(i.imm())?,
        },
        // Jumps back to the body when the loop continues, else falls through.
        Op::FORLOOP => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(i.imm())?,
        },
        Op::TFORPREP => Term::Jump(target(i.imm())?),
        Op::TFORLOOP => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(i.imm())?,
        },

        Op::VARARGPREP => return Err(Unsupported::Vararg(pc)),
        Op::VARARG => return Err(Unsupported::Vararg(pc)),
        Op::TBC => return Err(Unsupported::Tbc(pc)),
        // `args` / `returns`, and `SETLIST`'s `count`, are MULTRET sentinels
        // when zero.
        Op::CALL if i.b() == 0 || i.c() == 0 => {
            return Err(Unsupported::MultRet(pc));
        }
        Op::SETLIST if i.b() == 0 => {
            return Err(Unsupported::MultRet(pc));
        }

        _ => Term::Fallthrough(next),
    })
}

pub fn build(code: &[Instruction]) -> Result<Cfg, Unsupported> {
    // Leaders: pc 0, plus every branch/jump target and every instruction that
    // can be reached by falling out of a terminator.
    let mut leaders = vec![0u32];
    for pc in 0..code.len() as u32 {
        match terminator(code, pc)? {
            Term::Fallthrough(_) => {}
            Term::Jump(t) => {
                leaders.push(t);
                if pc + 1 < code.len() as u32 {
                    leaders.push(pc + 1);
                }
            }
            Term::Branch { skip, jump, .. } => {
                leaders.push(skip);
                leaders.push(jump);
            }
            Term::Return => {
                if pc + 1 < code.len() as u32 {
                    leaders.push(pc + 1);
                }
            }
        }
    }
    leaders.retain(|&pc| (pc as usize) < code.len());
    leaders.sort_unstable();
    leaders.dedup();

    let mut blocks: Vec<BcBlock> = Vec::with_capacity(leaders.len());
    for (n, &start) in leaders.iter().enumerate() {
        let limit = leaders.get(n + 1).copied().unwrap_or(code.len() as u32);

        // Walk to the block's real end: the first terminator at or before the
        // next leader.
        let mut end = start;
        let mut succs = Vec::new();
        while end < limit {
            let term = terminator(code, end)?;
            match term {
                Term::Fallthrough(next) => {
                    end = next;
                    if end >= limit {
                        // Falling out of the last block would run off the code
                        // array; well-formed bytecode always ends in RETURN/STOP.
                        if limit < code.len() as u32 {
                            succs.push(limit);
                        }
                        break;
                    }
                }
                Term::Jump(t) => {
                    end += 1;
                    succs.push(t);
                    break;
                }
                Term::Branch { skip, jump, .. } => {
                    // The block ends after the paired JMP for a compare, or
                    // after the loop op itself.
                    end = skip.max(end + 1);
                    succs.push(skip);
                    succs.push(jump);
                    break;
                }
                Term::Return => {
                    end += 1;
                    break;
                }
            }
        }

        blocks.push(BcBlock {
            start,
            end,
            succs,
            live_in: Vec::new(),
            edge_params: Vec::new(),
        });
    }

    let index_of: HashMap<u32, usize, RandomState> = blocks
        .iter()
        .enumerate()
        .map(|(n, b)| (b.start, n))
        .collect();

    let mut cfg = Cfg { blocks, index_of };
    liveness(code, &mut cfg)?;
    edge_params(code, &mut cfg);
    Ok(cfg)
}

/// Registers a *branching* terminator assigns on one of its edges but not the
/// other, so that the block has no single exit value for them.
///
/// On-the-fly SSA construction assumes one exit value per variable per block:
/// a read that walks into a predecessor takes that predecessor's definition,
/// with no way to say "but only along this edge". These registers therefore do
/// not go through the builder at all — the target takes an explicit parameter
/// and lowering supplies the right value per edge, exactly as it does today.
///
/// `FORPREP`/`FORLOOP` fold increment-and-branch into one instruction, so the
/// counter and the visible loop variable advance on the body edge and not on
/// the exit edge. `TESTSET` assigns `dst` only on the edge that does not skip.
pub fn edge_defs(i: Instruction, out: &mut Vec<u8>) {
    match i.op() {
        Op::FORPREP | Op::FORLOOP => {
            let base = i.a();
            out.push(base);
            out.push(base + 3);
        }
        Op::TESTSET => out.push(i.a()),
        _ => {}
    }
}

/// Propagate [`edge_defs`] from each branching terminator to its targets, and
/// intersect with liveness: a register nothing reads needs no parameter.
fn edge_params(code: &[Instruction], cfg: &mut Cfg) {
    let mut per_block: Vec<Vec<u8>> = vec![Vec::new(); cfg.blocks.len()];
    let mut defs = Vec::new();

    for b in 0..cfg.blocks.len() {
        let (start, end) = (cfg.blocks[b].start, cfg.blocks[b].end);
        defs.clear();
        for pc in start..end {
            // `terminator` re-derives the block's ending instruction; it cannot
            // fail here, `build` already walked the same range.
            if let Ok(Term::Branch { .. }) = terminator(code, pc) {
                edge_defs(code[pc as usize], &mut defs);
                break;
            }
        }
        if defs.is_empty() {
            continue;
        }
        for &s in &cfg.blocks[b].succs {
            per_block[cfg.index_of[&s]].extend_from_slice(&defs);
        }
    }

    for (b, block) in cfg.blocks.iter_mut().enumerate() {
        let mut regs = std::mem::take(&mut per_block[b]);
        regs.retain(|r| block.live_in.contains(r));
        regs.sort_unstable();
        regs.dedup();
        block.edge_params = regs;
    }
}

/// Registers an instruction reads.
pub fn reg_uses(i: Instruction, out: &mut Vec<u8>) {
    let mut push = |r: u8| out.push(r);
    match i.op() {
        // Single source operand, in `b`.
        Op::MOVE | Op::UNM | Op::BNOT | Op::NOT | Op::LEN | Op::TESTSET => push(i.b()),
        // Single source operand, in `a`.
        Op::SETUPVAL | Op::TEST | Op::LFALSESKIP | Op::ERRNNIL | Op::SETTABUP | Op::TBC => {
            push(i.a())
        }

        Op::GETTABLE => {
            push(i.b());
            push(i.c());
        }
        Op::SETTABLE => {
            push(i.a());
            push(i.b());
            push(i.c());
        }
        Op::GETFIELD => push(i.b()),
        Op::SETFIELD => {
            push(i.a());
            push(i.b());
        }
        Op::SELF => push(i.b()),

        // Binary operands in `b`/`c`.
        Op::ADD
        | Op::SUB
        | Op::MUL
        | Op::MOD
        | Op::POW
        | Op::DIV
        | Op::IDIV
        | Op::BAND
        | Op::BOR
        | Op::BXOR
        | Op::SHL
        | Op::SHR
        | Op::CONCAT
        | Op::VARARGGET => {
            push(i.b());
            push(i.c());
        }
        // Comparisons put their operands one slot lower, in `a`/`b`.
        Op::EQ | Op::LT | Op::LE => {
            push(i.a());
            push(i.b());
        }

        // `args` counts the function plus its arguments.
        Op::CALL | Op::TAILCALL => {
            let (func, args) = (i.a(), i.b());
            for r in func..func.saturating_add(args) {
                push(r);
            }
        }
        Op::RETURN => {
            let (values, count) = i.ab();
            for r in values..values.saturating_add(count.saturating_sub(1)) {
                push(r);
            }
        }

        // Control registers: init, limit, step. (`TFORCALL`/`TFORPREP`:
        // iterator, state, control.)
        Op::FORPREP | Op::FORLOOP | Op::TFORCALL | Op::TFORPREP => {
            let base = i.a();
            push(base);
            push(base + 1);
            push(base + 2);
        }
        // Reads the first result to decide whether to loop.
        Op::TFORLOOP => push(i.a() + 3),

        Op::SETLIST => {
            let (table, count, _offset) = i.abd();
            push(table);
            for r in table + 1..=table.saturating_add(count) {
                push(r);
            }
        }

        Op::LOAD
        | Op::GETUPVAL
        | Op::GETTABUP
        | Op::NEWTABLE
        | Op::CLOSE
        | Op::JMP
        | Op::CLOSURE
        | Op::VARARG
        | Op::VARARGPREP
        | Op::NOP
        | Op::STOP => {}
    }
}

/// Registers an instruction writes. A conditional write (`TESTSET`) is *not*
/// reported here: it is modelled on the CFG edge instead, which is what makes
/// it expressible in SSA at all.
pub fn reg_defs(i: Instruction, out: &mut Vec<u8>) {
    let mut push = |r: u8| out.push(r);
    match i.op() {
        // Destination register in `a`. (`LFALSESKIP` writes the register it
        // names, which its shape calls `src`.)
        Op::MOVE
        | Op::LOAD
        | Op::GETUPVAL
        | Op::GETTABUP
        | Op::GETTABLE
        | Op::GETFIELD
        | Op::NEWTABLE
        | Op::ADD
        | Op::SUB
        | Op::MUL
        | Op::MOD
        | Op::POW
        | Op::DIV
        | Op::IDIV
        | Op::BAND
        | Op::BOR
        | Op::BXOR
        | Op::SHL
        | Op::SHR
        | Op::UNM
        | Op::BNOT
        | Op::NOT
        | Op::LEN
        | Op::CONCAT
        | Op::CLOSURE
        | Op::VARARGGET
        | Op::LFALSESKIP => push(i.a()),

        // Method dispatch writes both the method and the receiver.
        Op::SELF => {
            let dst = i.a();
            push(dst);
            push(dst + 1);
        }

        Op::CALL => {
            let (func, _args, returns) = i.abc();
            for r in func..func.saturating_add(returns.saturating_sub(1)) {
                push(r);
            }
        }

        // Writes the internal counter and the visible loop variable.
        Op::FORPREP | Op::FORLOOP => {
            let base = i.a();
            push(base);
            push(base + 3);
        }
        // Results land at base+3.
        Op::TFORCALL => {
            let (base, count) = i.ab();
            for r in base + 3..base.saturating_add(3).saturating_add(count) {
                push(r);
            }
        }
        // Copies the first result into the control register.
        Op::TFORLOOP => push(i.a() + 2),

        // Conditional write, modelled on the CFG edge instead.
        Op::TESTSET => {}

        Op::SETUPVAL
        | Op::SETTABUP
        | Op::SETTABLE
        | Op::SETFIELD
        | Op::CLOSE
        | Op::TBC
        | Op::JMP
        | Op::EQ
        | Op::LT
        | Op::LE
        | Op::TEST
        | Op::TAILCALL
        | Op::RETURN
        | Op::TFORPREP
        | Op::SETLIST
        | Op::VARARG
        | Op::VARARGPREP
        | Op::ERRNNIL
        | Op::NOP
        | Op::STOP => {}
    }
}

/// Backward dataflow to a fixpoint. Standard `live_in = use ∪ (live_out - def)`.
///
/// `TESTSET` is the one wrinkle: it defines its `dst` only on the branch that
/// doesn't skip. Treating it as an unconditional def would kill a live value
/// on the skip path, so it declares no def here and the lowering passes `dst`'s
/// incoming value along the skip edge instead.
/// A set of Lua registers. There are at most 256, so the whole set is four
/// words: union and comparison are a handful of instructions rather than a loop,
/// and the fixpoint allocates nothing.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct RegSet([u64; 4]);

impl RegSet {
    fn insert(&mut self, r: u8) {
        self.0[r as usize >> 6] |= 1 << (r & 63);
    }

    fn remove(&mut self, r: u8) {
        self.0[r as usize >> 6] &= !(1 << (r & 63));
    }

    fn union(&mut self, other: &RegSet) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= *b;
        }
    }
    fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0u32..256)
            .filter(|&r| self.0[r as usize >> 6] & (1 << (r & 63)) != 0)
            .map(|r| r as u8)
    }
}

fn liveness(code: &[Instruction], cfg: &mut Cfg) -> Result<(), Unsupported> {
    let n = cfg.blocks.len();
    let mut live_in: Vec<RegSet> = vec![RegSet::default(); n];

    let mut uses = Vec::new();
    let mut defs = Vec::new();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut live = RegSet::default();
            for &s in &cfg.blocks[b].succs {
                let si = cfg.index_of[&s];
                live.union(&live_in[si]);
            }

            let (start, end) = (cfg.blocks[b].start, cfg.blocks[b].end);
            for pc in (start..end).rev() {
                let i = code[pc as usize];
                defs.clear();
                uses.clear();
                reg_defs(i, &mut defs);
                reg_uses(i, &mut uses);
                for &r in &defs {
                    live.remove(r);
                }
                for &r in &uses {
                    live.insert(r);
                }
            }

            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
    }

    for (b, block) in cfg.blocks.iter_mut().enumerate() {
        block.live_in = live_in[b].iter().collect();
    }
    Ok(())
}
