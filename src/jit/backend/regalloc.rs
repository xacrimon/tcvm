//! Where each virtual register lives.
//!
//! The encoder consumes an [`Allocation`] and nothing else, so the choice of
//! allocator is a swappable policy. Two exist:
//!
//!   - [`spill_everything`]: every value gets a stack slot; the encoder reloads
//!     operands into scratch registers around each instruction. Terrible code —
//!     and worth keeping, because it is the differential oracle for the one below.
//!     Any program the two disagree on is a register allocation bug.
//!   - [`linear_scan`]: intervals over a linearized CFG, allocated greedily,
//!     spilling the furthest-ending value under pressure.

use std::collections::HashSet;

use crate::jit::backend::mach::{MFunc, RegClass, VReg};

/// A physical register, numbered within its class: `x0`–`x30`, or `d0`–`d31`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PReg(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alloc {
    Reg(PReg),
    /// A slot in the native frame, indexed in words.
    ///
    /// Spill slots are *not* Lua stack slots. The Lua stack is the interpreter's,
    /// and holds 16-byte tagged values; these are one machine word and the
    /// collector never scans them — which is sound only because everything they
    /// hold is also reachable from the Lua stack at every point a collection can
    /// happen (an exit stub writes the whole frame back before anything else can
    /// run).
    Spill(u32),
}

pub struct Allocation {
    map: Vec<Alloc>,
    pub num_spills: u32,
}

impl Allocation {
    pub fn of(&self, v: VReg) -> Alloc {
        self.map[v.0 as usize]
    }

    /// How many values ended up in registers, for tests and for looking at.
    pub fn num_in_regs(&self) -> usize {
        self.map
            .iter()
            .filter(|a| matches!(a, Alloc::Reg(_)))
            .count()
    }
}

/// Give every virtual register its own stack slot.
pub fn spill_everything(f: &MFunc) -> Allocation {
    Allocation {
        map: (0..f.num_vregs() as u32).map(Alloc::Spill).collect(),
        num_spills: f.num_vregs() as u32,
    }
}

// The allocatable registers.
//
// Caller-saved only. Compiled regions are leaves — they call nothing — so a
// caller-saved register costs nothing to use, while a callee-saved one would have
// to be spilled and restored in the prologue for no gain at the register counts
// these regions actually reach. If pressure ever justifies it, x19–x28 and d8–d15
// are sitting there.
//
// Excluded and why:
//   x0, x1   incoming arguments, and x0 is the status word on the way out
//   x9-x11   the encoder's integer scratch
//   x16, x17 IP0/IP1 — the linker may insert a veneer that clobbers them
//   x18      reserved by the platform on Darwin
//   x29, x30 frame pointer and link register
//   d16-d18  the encoder's float scratch
const INT_REGS: &[u8] = &[2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15];
const FLOAT_REGS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

/// A value's live range, as one span of the linearized instruction order.
///
/// One span, not a set of them: a value live in two disjoint regions holds its
/// register through the gap between them. That over-reserves, and the cost is
/// paid in spills that a hole-aware allocator would avoid. It is not a
/// correctness question — a superset of the live points is always safe — and the
/// simplification keeps the scan below to one pass.
#[derive(Clone, Copy, Debug)]
struct Interval {
    v: VReg,
    start: u32,
    /// Exclusive.
    end: u32,
}

pub fn linear_scan(f: &MFunc) -> Allocation {
    let order = f.block_order();

    // Number every instruction. A block's span is the half-open range of the
    // positions its instructions occupy.
    let mut pos = vec![0u32; f.insts.len()];
    let mut span = vec![(0u32, 0u32); f.blocks.len()];
    let mut p = 0u32;
    for &b in &order {
        let start = p;
        for &i in &f.block(b).insts {
            pos[i] = p;
            p += 1;
        }
        span[b.0 as usize] = (start, p);
    }
    let end_of_code = p;

    let live_in = liveness(f, &order);

    // A value is live across a block if it is live on entry or on exit, and the
    // one-span simplification then covers the whole block. Otherwise it is live
    // only between the instructions that mention it, which the per-instruction
    // positions pin directly.
    let mut lo = vec![u32::MAX; f.num_vregs()];
    let mut hi = vec![0u32; f.num_vregs()];
    let mut extend = |v: VReg, s: u32, e: u32| {
        let i = v.0 as usize;
        lo[i] = lo[i].min(s);
        hi[i] = hi[i].max(e);
    };

    for &b in &order {
        let (bs, be) = span[b.0 as usize];
        for &s in &f.succs(b) {
            for &v in &live_in[s.0 as usize] {
                extend(v, bs, be);
            }
        }
        for &v in &live_in[b.0 as usize] {
            extend(v, bs, be);
        }
        for &i in &f.block(b).insts {
            let inst = f.inst(i);
            for &v in inst.defs.iter().chain(&inst.uses) {
                extend(v, pos[i], pos[i] + 1);
            }
        }
    }

    // The frame base is live everywhere, and not because the fast path says so:
    // every exit stub addresses the Lua stack through it, and the stubs are not
    // instructions, so nothing in the interval computation above sees those uses.
    // Left to the ordinary rules it would die at its last fast-path use and a stub
    // reached after that point would write the frame through whatever happened to
    // land in its register.
    extend(f.frame_base, 0, end_of_code);

    let mut ivs: Vec<Interval> = (0..f.num_vregs() as u32)
        .map(VReg)
        .filter(|v| lo[v.0 as usize] != u32::MAX)
        .map(|v| Interval {
            v,
            start: lo[v.0 as usize],
            end: hi[v.0 as usize],
        })
        .collect();
    ivs.sort_by_key(|i| i.start);

    let mut map = vec![Alloc::Spill(0); f.num_vregs()];
    let mut spills = 0u32;
    for class in [RegClass::Int, RegClass::Float] {
        let pool = match class {
            RegClass::Int => INT_REGS,
            RegClass::Float => FLOAT_REGS,
        };
        scan(
            ivs.iter().filter(|i| f.class(i.v) == class).copied(),
            pool,
            &mut map,
            &mut spills,
        );
    }

    Allocation {
        map,
        num_spills: spills,
    }
}

/// Greedy scan over one register class.
///
/// Poletto–Sarkar: walk intervals by start, retire those whose end has passed,
/// and when nothing is free evict whichever live value is needed longest — that
/// value, or the incoming one if it outlives them all.
fn scan(ivs: impl Iterator<Item = Interval>, pool: &[u8], map: &mut [Alloc], spills: &mut u32) {
    let mut free: Vec<u8> = pool.iter().rev().copied().collect();
    // (interval, register), kept sorted by end so the eviction candidate is last.
    let mut active: Vec<(Interval, u8)> = Vec::new();

    for iv in ivs {
        active.retain(|&(a, r)| {
            if a.end <= iv.start {
                free.push(r);
                false
            } else {
                true
            }
        });

        if let Some(r) = free.pop() {
            map[iv.v.0 as usize] = Alloc::Reg(PReg(r));
            active.push((iv, r));
        } else {
            // Nothing free. The value with the furthest end is the cheapest to
            // lose: it is the one that would otherwise hold a register through the
            // most other intervals.
            let (worst, wr) = *active.last().expect("no free registers and none live");
            if worst.end > iv.end {
                map[worst.v.0 as usize] = Alloc::Spill(*spills);
                *spills += 1;
                map[iv.v.0 as usize] = Alloc::Reg(PReg(wr));
                active.pop();
                active.push((iv, wr));
            } else {
                map[iv.v.0 as usize] = Alloc::Spill(*spills);
                *spills += 1;
            }
        }
        active.sort_by_key(|&(a, _)| a.end);
    }
}

/// The one invariant an allocation has to satisfy: two values that are live at
/// the same instruction never share a physical register.
///
/// Checked directly rather than inferred from "the tests pass", because the
/// failure mode is a value silently overwritten on one path through a loop — the
/// kind of bug that shows up as a wrong answer six months later on a program
/// nobody has yet written. `encode` runs this on every allocation in debug builds.
///
/// The set checked at each instruction is its live-in plus its definitions: a
/// definition writes its register *at* the instruction, so it must not collide
/// with anything live across it, even though it is not itself live on the way in.
pub fn verify(f: &MFunc, ra: &Allocation) -> Result<(), String> {
    let order = f.block_order();
    let live_in = liveness(f, &order);

    for &b in &order {
        let mut live: HashSet<VReg> = f
            .succs(b)
            .iter()
            .flat_map(|s| live_in[s.0 as usize].iter().copied())
            .collect();

        for &i in f.block(b).insts.iter().rev() {
            let inst = f.inst(i);
            for &d in &inst.defs {
                live.remove(&d);
            }
            for &u in &inst.uses {
                live.insert(u);
            }

            let mut seen: Vec<(PReg, VReg)> = Vec::new();
            for &v in live.iter().chain(&inst.defs) {
                if let Alloc::Reg(r) = ra.of(v) {
                    if let Some(&(_, other)) = seen.iter().find(|&&(s, o)| s == r && o != v) {
                        return Err(format!(
                            "mb{} inst {i} ({:?}): {v:?} and {other:?} are both live and both in x{}",
                            b.0, inst.op, r.0
                        ));
                    }
                    seen.push((r, v));
                }
            }
        }
    }
    Ok(())
}

/// Live-in sets, to a fixpoint.
///
/// Backwards over the block order, which converges in one pass for a loop-free
/// region and in a couple more with a back edge.
fn liveness(f: &MFunc, order: &[crate::jit::backend::mach::MBlock]) -> Vec<HashSet<VReg>> {
    let mut live_in: Vec<HashSet<VReg>> = vec![HashSet::new(); f.blocks.len()];

    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            let mut live: HashSet<VReg> = f
                .succs(b)
                .iter()
                .flat_map(|s| live_in[s.0 as usize].iter().copied())
                .collect();

            for &i in f.block(b).insts.iter().rev() {
                let inst = f.inst(i);
                for &d in &inst.defs {
                    live.remove(&d);
                }
                for &u in &inst.uses {
                    live.insert(u);
                }
            }

            if live != live_in[b.0 as usize] {
                live_in[b.0 as usize] = live;
                changed = true;
            }
        }
        if !changed {
            return live_in;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::mach::{AluOp, MInst, MOp};

    /// The smallest function with a register conflict in it: two constants that
    /// are both live at the add that consumes them.
    fn add_func() -> MFunc {
        let mut m = MFunc::new();
        let b = m.new_block();
        m.entry = b;

        let base = m.new_vreg(RegClass::Int);
        m.frame_base = base;
        let x = m.new_vreg(RegClass::Int);
        let y = m.new_vreg(RegClass::Int);
        let sum = m.new_vreg(RegClass::Int);

        m.push(b, MInst::new(MOp::EntryArg(1), vec![base], vec![]));
        m.push(b, MInst::new(MOp::Imm(1), vec![x], vec![]));
        m.push(b, MInst::new(MOp::Imm(2), vec![y], vec![]));
        m.push(b, MInst::new(MOp::Alu(AluOp::Add), vec![sum], vec![x, y]));
        m.push(b, MInst::new(MOp::Ret { nret: 1 }, vec![], vec![sum]));
        m
    }

    #[test]
    fn overlapping_values_get_different_registers() {
        let m = add_func();
        let ra = linear_scan(&m);
        verify(&m, &ra).expect("linear scan must satisfy its own invariant");
        assert_ne!(
            ra.of(VReg(1)),
            ra.of(VReg(2)),
            "both operands of the add are live at it"
        );
    }

    /// The verifier has to be able to fail, or the assertion in `encode` proves
    /// nothing. Put every value in the same register and watch it complain.
    #[test]
    fn verify_catches_a_collision() {
        let m = add_func();
        let bad = Allocation {
            map: vec![Alloc::Reg(PReg(2)); m.num_vregs()],
            num_spills: 0,
        };
        let err = verify(&m, &bad).expect_err("every value in x2 is a collision");
        assert!(err.contains("x2"), "{err}");
    }
}
