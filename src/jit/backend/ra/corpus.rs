//! Allocations recorded from real compiles (`TCVM_JIT_RA_CAPTURE=dir`, one
//! file per compile) and replayed by `ra::tests`: an allocator change meant
//! to keep every allocation is checked against them, and timed on them.

use super::{Edit, Env, Output};
use crate::jit::backend::reg::Allocation;
#[cfg(test)]
use crate::jit::backend::reg::{Inst, PReg, ProgPoint, RegClass, VReg};
use crate::jit::backend::vcode::{Remat, TargetInst, VCode};

const MAGIC: u32 = 0x5241_0003;
/// Format 2: no edge edits (every edge was split).
#[cfg(test)]
const MAGIC_2: u32 = 0x5241_0002;

/// One compile's allocation problem and the allocation it got.
pub(crate) fn write<I: TargetInst>(
    f: &VCode<I>,
    env: &Env,
    order: &[u32],
    hot: usize,
    out: &Output,
) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.u32(MAGIC);
    for regs in &env.regs {
        w.list(regs.iter().map(|r| r.index() as u32));
    }
    w.list(env.scratch.iter().map(|r| r.index() as u32));
    w.u32(env.transport.index() as u32);
    w.u32(f.num_insts() as u32);
    for i in 0..f.num_insts() {
        w.u32(f.insts[i].is_branch() as u32);
        let ops = f.inst_operands_of(i);
        w.u32(ops.len() as u32);
        for op in ops {
            w.u64(op.bits());
        }
        w.list(f.clobbers[i].into_iter().map(|r| r.index() as u32));
    }
    w.u32(f.num_blocks() as u32);
    for b in 0..f.num_blocks() {
        let rb = crate::jit::backend::reg::Block::new(b);
        w.u32(f.blocks[b].start);
        w.u32(f.blocks[b].end);
        w.u32(f.blocks[b].resume as u32);
        w.list(f.block_succs(rb).iter().map(|s| s.index() as u32));
        w.list(f.block_preds(rb).iter().map(|p| p.index() as u32));
        w.list(f.block_params(rb).iter().map(|v| v.bits() as u32));
        for k in 0..f.block_succs(rb).len() {
            w.list(f.branch_blockparams(rb, k).iter().map(|v| v.bits() as u32));
        }
    }
    w.u32(f.num_vregs() as u32);
    for v in 0..f.num_vregs() {
        w.u32(f.vreg_classes[v] as u32);
        let (tag, a, b) = match f.remat[v] {
            Remat::None => (0, 0, 0),
            Remat::Int(x) => (1, x, 0),
            Remat::Float(x) => (2, x, 0),
            Remat::AndImm { src, imm } => (3, src as u64, imm),
            Remat::Slot(r) => (4, r as u64, 0),
        };
        w.u32(tag);
        w.u64(a);
        w.u64(b);
    }
    w.u32(f.entry_block().index() as u32);
    w.list(f.loop_of.iter().copied());
    w.list(f.loop_parent.iter().copied());
    w.list(order.iter().copied());
    w.u32(hot as u32);
    w.u32(out.num_spillslots as u32);
    let edit = |w: &mut W, e: &Edit| match *e {
        Edit::Move { from, to } => {
            w.u32(0);
            w.u32(alloc_bits(from));
            w.u32(alloc_bits(to));
        }
        Edit::Remat { to, vreg, src } => {
            w.u32(1);
            w.u32(alloc_bits(to));
            w.u32(vreg.bits() as u32);
            w.u32(alloc_bits(src));
        }
    };
    w.u32(out.edits.len() as u32);
    for (p, e) in &out.edits {
        w.u32(p.to_index());
        edit(&mut w, e);
    }
    w.u32(out.edge_edits.len() as u32);
    for (i, k, e) in &out.edge_edits {
        w.u32(i.index() as u32);
        w.u32(*k as u32);
        edit(&mut w, e);
    }
    w.list(out.allocs.iter().map(|&a| alloc_bits(a)));
    w.0
}

/// A recorded compile: the function (each instruction made by `mk` from
/// whether it is a branch), the registers, the order, and its allocation.
#[cfg(test)]
pub(crate) struct Case<I> {
    pub(crate) f: VCode<I>,
    pub(crate) env: Env,
    pub(crate) order: Vec<u32>,
    pub(crate) hot: usize,
    pub(crate) out: Output,
}

#[cfg(test)]
pub(crate) fn read<I: TargetInst>(d: &[u8], mk: impl Fn(bool) -> I) -> Result<Case<I>, String> {
    use crate::jit::backend::reg::{Block, Operand, PRegSet};
    let mut r = R(d, 0);
    let magic = r.u32();
    if magic != MAGIC && magic != MAGIC_2 {
        return Err("not a recorded allocation".into());
    }
    let preg = |x: u32| PReg::from_index(x as usize);
    let regs = [r.list().map(preg).collect(), r.list().map(preg).collect()];
    let s: Vec<PReg> = r.list().map(preg).collect();
    let env = Env {
        regs,
        scratch: [s[0], s[1]],
        transport: preg(r.u32()),
        kept: [0; 2],
    };
    let mut f = VCode::new();
    for _ in 0..r.u32() {
        let br = r.u32() != 0;
        let n = r.u32();
        let ops: Vec<Operand> = (0..n).map(|_| Operand::from_bits(r.u64())).collect();
        let clob: PRegSet = r.list().map(preg).collect();
        f.push(mk(br), &ops, clob);
    }
    for _ in 0..r.u32() {
        let (start, end, resume) = (r.u32(), r.u32(), r.u32() != 0);
        let succs: Vec<Block> = r.list().map(|b| Block::new(b as usize)).collect();
        let preds: Vec<Block> = r.list().map(|b| Block::new(b as usize)).collect();
        let params: Vec<VReg> = r.list().map(VReg::from_bits).collect();
        let (mut args, mut nargs) = (Vec::new(), Vec::new());
        for _ in 0..succs.len() {
            let a: Vec<VReg> = r.list().map(VReg::from_bits).collect();
            nargs.push(a.len() as u32);
            args.extend(a);
        }
        f.push_block(start, end, &succs, preds, params, &args, &nargs, resume);
    }
    for _ in 0..r.u32() {
        let class = match r.u32() {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => RegClass::Vector,
        };
        let v = f.new_vreg(class);
        let (tag, a, b) = (r.u32(), r.u64(), r.u64());
        f.remat[v.vreg()] = match tag {
            0 => Remat::None,
            1 => Remat::Int(a),
            2 => Remat::Float(a),
            3 => Remat::AndImm {
                src: a as u32,
                imm: b,
            },
            _ => Remat::Slot(a as u8),
        };
    }
    f.entry = Block::new(r.u32() as usize);
    f.loop_of = r.list().collect();
    f.loop_parent = r.list().collect();
    let order = r.list().collect();
    let hot = r.u32() as usize;
    let num_spillslots = r.u32() as usize;
    let edit = |r: &mut R| match r.u32() {
        0 => Edit::Move {
            from: alloc(r.u32()),
            to: alloc(r.u32()),
        },
        _ => Edit::Remat {
            to: alloc(r.u32()),
            vreg: VReg::from_bits(r.u32()),
            src: alloc(r.u32()),
        },
    };
    let edits = (0..r.u32())
        .map(|_| {
            let p = ProgPoint::from_index(r.u32());
            (p, edit(&mut r))
        })
        .collect();
    let edge_edits = (0..if magic == MAGIC { r.u32() } else { 0 })
        .map(|_| {
            let i = Inst::new(r.u32() as usize);
            let k = r.u32() as u8;
            (i, k, edit(&mut r))
        })
        .collect();
    let allocs = r.list().map(alloc).collect();
    if r.1 != d.len() {
        return Err("trailing bytes".into());
    }
    Ok(Case {
        f,
        env,
        order,
        hot,
        out: Output {
            num_spillslots,
            edits,
            edge_edits,
            allocs,
            kept: Vec::new(),
        },
    })
}

/// An allocation as its kind and index (`Allocation::from_bits` asserts
/// a kind none of them has).
fn alloc_bits(a: Allocation) -> u32 {
    match (a.as_reg(), a.as_stack()) {
        (Some(r), _) => 1 << 29 | r.index() as u32,
        (_, Some(s)) => 2 << 29 | s.index() as u32,
        _ => 0,
    }
}

#[cfg(test)]
fn alloc(bits: u32) -> Allocation {
    let k = (bits & ((1 << 29) - 1)) as usize;
    match bits >> 29 {
        1 => Allocation::reg(PReg::from_index(k)),
        2 => Allocation::stack(crate::jit::backend::reg::SpillSlot::new(k)),
        _ => Allocation::none(),
    }
}

struct W(Vec<u8>);

impl W {
    fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }

    fn u64(&mut self, x: u64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }

    fn list(&mut self, xs: impl IntoIterator<Item = u32>) {
        let xs: Vec<u32> = xs.into_iter().collect();
        self.u32(xs.len() as u32);
        for x in xs {
            self.u32(x);
        }
    }
}

#[cfg(test)]
struct R<'a>(&'a [u8], usize);

#[cfg(test)]
impl R<'_> {
    fn u32(&mut self) -> u32 {
        let x = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().expect("4 bytes"));
        self.1 += 4;
        x
    }

    fn u64(&mut self) -> u64 {
        let x = u64::from_le_bytes(self.0[self.1..self.1 + 8].try_into().expect("8 bytes"));
        self.1 += 8;
        x
    }

    fn list(&mut self) -> impl Iterator<Item = u32> + '_ {
        let n = self.u32();
        (0..n).map(move |_| self.u32())
    }
}
