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

use crate::instruction::Instruction;

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
    matches!(
        i,
        Instruction::EQ { .. }
            | Instruction::LT { .. }
            | Instruction::LE { .. }
            | Instruction::TEST { .. }
            | Instruction::TESTSET { .. }
    )
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

    Ok(match i {
        Instruction::JMP { offset } => Term::Jump(target(offset)?),

        // The compare skips the `JMP` at pc+1 when it succeeds, so the two
        // edges are pc+2 and that `JMP`'s target.
        _ if is_skip_test(i) => {
            let Some(Instruction::JMP { offset }) = code.get(next as usize).copied() else {
                // The compiler always pairs these; anything else is a bytecode
                // we don't understand, so decline the region.
                return Err(Unsupported::BadJump(pc));
            };
            Term::Branch {
                cond: pc,
                skip: pc + 2,
                jump: jump_target(next, offset) as u32,
            }
        }
        // `LFALSESKIP` sets false and unconditionally skips the next
        // instruction — a jump, not a branch.
        Instruction::LFALSESKIP { .. } => Term::Jump(pc + 2),

        Instruction::RETURN { count, .. } => {
            if count == 0 {
                return Err(Unsupported::MultRet(pc));
            }
            Term::Return
        }
        Instruction::TAILCALL { args, .. } => {
            if args == 0 {
                return Err(Unsupported::MultRet(pc));
            }
            Term::Return
        }
        Instruction::STOP => Term::Return,

        // Jumps past the body when the loop shouldn't run at all; otherwise
        // falls into it.
        Instruction::FORPREP { offset, .. } => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(offset)?,
        },
        // Jumps back to the body when the loop continues, else falls through.
        Instruction::FORLOOP { offset, .. } => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(offset)?,
        },
        Instruction::TFORPREP { offset, .. } => Term::Jump(target(offset)?),
        Instruction::TFORLOOP { offset, .. } => Term::Branch {
            cond: pc,
            skip: next,
            jump: target(offset)?,
        },

        Instruction::VARARGPREP { .. } => return Err(Unsupported::Vararg(pc)),
        Instruction::VARARG { .. } => return Err(Unsupported::Vararg(pc)),
        Instruction::TBC { .. } => return Err(Unsupported::Tbc(pc)),
        Instruction::CALL { args, returns, .. } if args == 0 || returns == 0 => {
            return Err(Unsupported::MultRet(pc));
        }
        Instruction::SETLIST { count: 0, .. } => {
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
    match i {
        Instruction::FORPREP { base, .. } | Instruction::FORLOOP { base, .. } => {
            out.push(base);
            out.push(base + 3);
        }
        Instruction::TESTSET { dst, .. } => out.push(dst),
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
    match i {
        Instruction::MOVE { src, .. }
        | Instruction::SETUPVAL { src, .. }
        | Instruction::UNM { src, .. }
        | Instruction::BNOT { src, .. }
        | Instruction::NOT { src, .. }
        | Instruction::LEN { src, .. }
        | Instruction::TEST { src, .. }
        | Instruction::TESTSET { src, .. }
        | Instruction::LFALSESKIP { src }
        | Instruction::ERRNNIL { src, .. } => push(src),

        Instruction::SETTABUP { src, .. } => push(src),

        Instruction::GETTABLE { table, key, .. } => {
            push(table);
            push(key);
        }
        Instruction::SETTABLE {
            src, table, key, ..
        } => {
            push(src);
            push(table);
            push(key);
        }
        Instruction::GETFIELD { table, .. } => push(table),
        Instruction::SETFIELD { src, table, .. } => {
            push(src);
            push(table);
        }
        Instruction::SELF { object, .. } => push(object),

        Instruction::ADD { lhs, rhs, .. }
        | Instruction::SUB { lhs, rhs, .. }
        | Instruction::MUL { lhs, rhs, .. }
        | Instruction::MOD { lhs, rhs, .. }
        | Instruction::POW { lhs, rhs, .. }
        | Instruction::DIV { lhs, rhs, .. }
        | Instruction::IDIV { lhs, rhs, .. }
        | Instruction::BAND { lhs, rhs, .. }
        | Instruction::BOR { lhs, rhs, .. }
        | Instruction::BXOR { lhs, rhs, .. }
        | Instruction::SHL { lhs, rhs, .. }
        | Instruction::SHR { lhs, rhs, .. }
        | Instruction::CONCAT { lhs, rhs, .. }
        | Instruction::EQ { lhs, rhs, .. }
        | Instruction::LT { lhs, rhs, .. }
        | Instruction::LE { lhs, rhs, .. } => {
            push(lhs);
            push(rhs);
        }

        Instruction::CALL { func, args, .. } => {
            // `args` counts the function plus its arguments.
            for r in func..func.saturating_add(args) {
                push(r);
            }
        }
        Instruction::TAILCALL { func, args } => {
            for r in func..func.saturating_add(args) {
                push(r);
            }
        }
        Instruction::RETURN { values, count } => {
            for r in values..values.saturating_add(count.saturating_sub(1)) {
                push(r);
            }
        }

        // Control registers: init, limit, step.
        Instruction::FORPREP { base, .. } | Instruction::FORLOOP { base, .. } => {
            push(base);
            push(base + 1);
            push(base + 2);
        }
        // Iterator, state, control.
        Instruction::TFORCALL { base, .. } => {
            push(base);
            push(base + 1);
            push(base + 2);
        }
        Instruction::TFORPREP { base, .. } => {
            push(base);
            push(base + 1);
            push(base + 2);
        }
        // Reads the first result to decide whether to loop.
        Instruction::TFORLOOP { base, .. } => push(base + 3),

        Instruction::SETLIST {
            table,
            count,
            offset: _,
        } => {
            push(table);
            for r in table + 1..=table.saturating_add(count) {
                push(r);
            }
        }
        Instruction::VARARGGET { base, key, .. } => {
            push(base);
            push(key);
        }
        Instruction::TBC { val } => push(val),

        Instruction::LOAD { .. }
        | Instruction::GETUPVAL { .. }
        | Instruction::GETTABUP { .. }
        | Instruction::NEWTABLE { .. }
        | Instruction::CLOSE { .. }
        | Instruction::JMP { .. }
        | Instruction::CLOSURE { .. }
        | Instruction::VARARG { .. }
        | Instruction::VARARGPREP { .. }
        | Instruction::NOP
        | Instruction::STOP => {}
    }
}

/// Registers an instruction writes. A conditional write (`TESTSET`) is *not*
/// reported here: it is modelled on the CFG edge instead, which is what makes
/// it expressible in SSA at all.
pub fn reg_defs(i: Instruction, out: &mut Vec<u8>) {
    let mut push = |r: u8| out.push(r);
    match i {
        Instruction::MOVE { dst, .. }
        | Instruction::LOAD { dst, .. }
        | Instruction::GETUPVAL { dst, .. }
        | Instruction::GETTABUP { dst, .. }
        | Instruction::GETTABLE { dst, .. }
        | Instruction::GETFIELD { dst, .. }
        | Instruction::NEWTABLE { dst }
        | Instruction::ADD { dst, .. }
        | Instruction::SUB { dst, .. }
        | Instruction::MUL { dst, .. }
        | Instruction::MOD { dst, .. }
        | Instruction::POW { dst, .. }
        | Instruction::DIV { dst, .. }
        | Instruction::IDIV { dst, .. }
        | Instruction::BAND { dst, .. }
        | Instruction::BOR { dst, .. }
        | Instruction::BXOR { dst, .. }
        | Instruction::SHL { dst, .. }
        | Instruction::SHR { dst, .. }
        | Instruction::UNM { dst, .. }
        | Instruction::BNOT { dst, .. }
        | Instruction::NOT { dst, .. }
        | Instruction::LEN { dst, .. }
        | Instruction::CONCAT { dst, .. }
        | Instruction::CLOSURE { dst, .. }
        | Instruction::VARARGGET { dst, .. } => push(dst),

        Instruction::LFALSESKIP { src } => push(src),

        // Method dispatch writes both the method and the receiver.
        Instruction::SELF { dst, .. } => {
            push(dst);
            push(dst + 1);
        }

        Instruction::CALL { func, returns, .. } => {
            for r in func..func.saturating_add(returns.saturating_sub(1)) {
                push(r);
            }
        }

        // Writes the internal counter and the visible loop variable.
        Instruction::FORPREP { base, .. } | Instruction::FORLOOP { base, .. } => {
            push(base);
            push(base + 3);
        }
        // Results land at base+3.
        Instruction::TFORCALL { base, count } => {
            for r in base + 3..base.saturating_add(3).saturating_add(count) {
                push(r);
            }
        }
        // Copies the first result into the control register.
        Instruction::TFORLOOP { base, .. } => push(base + 2),

        Instruction::TESTSET { .. } => {}

        Instruction::SETUPVAL { .. }
        | Instruction::SETTABUP { .. }
        | Instruction::SETTABLE { .. }
        | Instruction::SETFIELD { .. }
        | Instruction::CLOSE { .. }
        | Instruction::TBC { .. }
        | Instruction::JMP { .. }
        | Instruction::EQ { .. }
        | Instruction::LT { .. }
        | Instruction::LE { .. }
        | Instruction::TEST { .. }
        | Instruction::TAILCALL { .. }
        | Instruction::RETURN { .. }
        | Instruction::TFORPREP { .. }
        | Instruction::SETLIST { .. }
        | Instruction::VARARG { .. }
        | Instruction::VARARGPREP { .. }
        | Instruction::ERRNNIL { .. }
        | Instruction::NOP
        | Instruction::STOP => {}
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
