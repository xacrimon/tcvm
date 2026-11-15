//! Allocations recorded from real compiles (`TCVM_JIT_RA_CAPTURE=dir`, one
//! file per compile) and replayed by `ra::tests`: an allocator change meant
//! to keep every allocation is checked against them, and timed on them.

use super::{Edit, Env, Output};
use crate::jit::backend::reg::Allocation;
#[cfg(test)]
use crate::jit::backend::reg::{Inst, PReg, ProgPoint, RegClass, VReg};
#[cfg(test)]
use crate::jit::backend::vcode::{CAbi, Clobbers, Flow, List, MInst, Pin, Row};
use crate::jit::backend::vcode::{Constraint, Remat, Slot, VCode};

/// Format 4: instruction records, the pool and the opcode table they are
/// read through. Earlier formats held operand lists and are refused.
const MAGIC: u32 = 0x5241_0004;

fn slot_bits(s: Slot) -> u32 {
    match s {
        Slot::None => 0,
        Slot::Imm => 1,
        Slot::Wide => 2,
        Slot::Start => 3,
        Slot::Len => 4,
        Slot::Reg {
            class,
            def,
            late,
            constraint,
        } => {
            let (tag, x) = match constraint {
                Constraint::Any => (0, 0),
                Constraint::Tied(k) => (1, k as u32),
                Constraint::Fixed(p) => (2, p.index() as u32),
            };
            5 | (class as u32) << 4 | (def as u32) << 6 | (late as u32) << 7 | tag << 8 | x << 12
        }
    }
}

#[cfg(test)]
fn slot(bits: u32) -> Slot {
    match bits & 15 {
        0 => Slot::None,
        1 => Slot::Imm,
        2 => Slot::Wide,
        3 => Slot::Start,
        4 => Slot::Len,
        _ => Slot::Reg {
            class: class(bits >> 4 & 3),
            def: bits >> 6 & 1 != 0,
            late: bits >> 7 & 1 != 0,
            constraint: match bits >> 8 & 15 {
                0 => Constraint::Any,
                1 => Constraint::Tied((bits >> 12) as u8),
                _ => Constraint::Fixed(PReg::from_index((bits >> 12) as usize)),
            },
        },
    }
}

#[cfg(test)]
fn class(x: u32) -> RegClass {
    match x {
        0 => RegClass::Int,
        1 => RegClass::Float,
        _ => RegClass::Vector,
    }
}

/// One compile's allocation problem and the allocation it got.
pub(crate) fn write(f: &VCode, env: &Env, order: &[u32], hot: usize, out: &Output) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.u32(MAGIC);
    for regs in &env.regs {
        w.list(regs.iter().map(|r| r.index() as u32));
    }
    w.list(env.scratch.iter().map(|r| r.index() as u32));
    w.u32(env.transport.index() as u32);
    for args in &env.c.args {
        w.list(args.iter().map(|r| r.index() as u32));
    }
    w.list(env.c.ret.iter().map(|r| r.index() as u32));
    w.list(env.c.caller_saved.into_iter().map(|r| r.index() as u32));
    w.u32(f.rows.len() as u32);
    for row in f.rows {
        w.list(row.slots.iter().map(|&s| slot_bits(s)));
        w.u32(row.list as u32);
        w.u32(row.clobbers as u32);
        w.u32(row.flow as u32);
        w.u32(row.remat as u32 | (row.pseudo as u32) << 1);
    }
    w.u32(f.num_insts() as u32);
    for m in &f.insts {
        w.u32(m.op as u32 | (m.sz as u32) << 8 | (m.cc as u32) << 16 | (m.n as u32) << 24);
        w.list(m.ops);
    }
    w.list(f.pool.iter().copied());
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
            Remat::Pinned(p) => (5, p as u64, 0),
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

/// A recorded compile: the function with its opcode table, the registers,
/// the order, and its allocation.
#[cfg(test)]
pub(crate) struct Case {
    pub(crate) f: VCode,
    pub(crate) env: Env,
    pub(crate) order: Vec<u32>,
    pub(crate) hot: usize,
    pub(crate) out: Output,
}

#[cfg(test)]
pub(crate) fn read(d: &[u8]) -> Result<Case, String> {
    use crate::jit::backend::reg::{Block, PRegSet};
    let mut r = R(d, 0);
    if r.u32() != MAGIC {
        return Err("not a recorded allocation of this format".into());
    }
    let preg = |x: u32| PReg::from_index(x as usize);
    let regs = [r.list().map(preg).collect(), r.list().map(preg).collect()];
    let s: Vec<PReg> = r.list().map(preg).collect();
    let transport = preg(r.u32());
    let args: Vec<Vec<PReg>> = (0..2).map(|_| r.list().map(preg).collect()).collect();
    let ret: Vec<PReg> = r.list().map(preg).collect();
    let caller_saved: PRegSet = r.list().map(preg).collect();
    let env = Env {
        regs,
        scratch: [s[0], s[1]],
        transport,
        c: CAbi {
            args: [0, 1].map(|c| std::array::from_fn(|k| args[c][k])),
            ret: [ret[0], ret[1]],
            caller_saved,
        },
    };
    let rows: Vec<Row> = (0..r.u32())
        .map(|_| {
            let slots: Vec<Slot> = r.list().map(slot).collect();
            let (list, clobbers, flow, flags) = (r.u32(), r.u32(), r.u32(), r.u32());
            Row {
                name: "",
                slots: [slots[0], slots[1], slots[2]],
                list: [List::None, List::CArgs, List::AnyLoc][list as usize],
                clobbers: [Clobbers::None, Clobbers::CallerSaved][clobbers as usize],
                flow: [
                    Flow::Fall,
                    Flow::Jump,
                    Flow::Branch,
                    Flow::CondExit,
                    Flow::Exit,
                ][flow as usize],
                remat: flags & 1 != 0,
                pseudo: flags & 2 != 0,
            }
        })
        .collect();
    // The pool is read before the instructions count their lists.
    let insts: Vec<MInst> = (0..r.u32())
        .map(|_| {
            let w = r.u32();
            let ops: Vec<u32> = r.list().collect();
            MInst {
                op: w as u8,
                sz: (w >> 8) as u8,
                cc: (w >> 16) as u8,
                n: (w >> 24) as u8,
                ops: [ops[0], ops[1], ops[2]],
            }
        })
        .collect();
    let mut f = VCode::new(rows.leak());
    f.pool = r.list().collect();
    for m in insts {
        f.push(m);
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
        let v = f.new_vreg(class(r.u32()));
        let (tag, a, b) = (r.u32(), r.u64(), r.u64());
        f.remat[v.vreg()] = match tag {
            0 => Remat::None,
            1 => Remat::Int(a),
            2 => Remat::Float(a),
            3 => Remat::AndImm {
                src: a as u32,
                imm: b,
            },
            4 => Remat::Slot(a as u8),
            _ => Remat::Pinned([Pin::Base, Pin::Thread][a as usize]),
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
    let edge_edits = (0..r.u32())
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
