//! A graph as text, by the op table.

use std::fmt::Write;

use super::ops::{Arg, Aux, Cc, ExitTag, Op};
use super::types::{Rep, TypeSet};
use super::{BlockFlags, Graph, HOME, Val};

const KINDS: [&str; 13] = [
    "none", "add", "sub", "mul", "mod", "pow", "div", "idiv", "band", "bor", "bxor", "shl", "shr",
];

impl Graph {
    pub(crate) fn print(&self) -> String {
        let mut o = String::new();
        for &b in &self.order {
            let d = self.block(b);
            let _ = write!(o, "b{}:", b.0);
            for (k, p) in self.preds(b).iter().enumerate() {
                o.push_str(if k == 0 { " <-" } else { "" });
                match p.is_none() {
                    true => o.push_str(" _"),
                    false => _ = write!(o, " b{}", p.0),
                }
            }
            if !d.lp.is_none() {
                let _ = write!(o, " l{}", d.lp.0);
            }
            for (f, name) in [
                (BlockFlags::LOOP_HEADER, "header"),
                (BlockFlags::COLD, "cold"),
                (BlockFlags::RESUME, "resume"),
            ] {
                if d.flags.contains(f) {
                    let _ = write!(o, " {name}");
                }
            }
            o.push('\n');
            let end = if d.end == u32::MAX {
                self.insts.len() as u32
            } else {
                d.end
            };
            for v in d.start..end {
                self.print_inst(&mut o, Val(v));
            }
        }
        o
    }

    fn print_inst(&self, o: &mut String, v: Val) {
        let i = self.inst(v);
        let info = i.op.info();
        let _ = write!(o, "  v{}", v.0);
        if i.rep != Rep::None {
            let _ = write!(o, ": {}", i.rep.name());
        }
        let _ = write!(o, " = {}", info.name.to_lowercase());
        let tag = |o: &mut String, bits: u16| {
            let _ = write!(o, "!{}", format!("{:?}", ExitTag::of(bits)).to_lowercase());
        };
        match info.aux {
            Aux::No => {}
            Aux::Cc | Aux::FCc => _ = write!(o, ".{}", Cc::of(i.aux & 15).name()),
            Aux::Set | Aux::SetTag => {
                o.push('.');
                TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits()).print(o);
                if info.aux == Aux::SetTag {
                    tag(o, i.aux >> TypeSet::BITS);
                }
            }
            Aux::Tag => tag(o, i.aux),
            Aux::Slot => _ = write!(o, " r{}", i.aux),
            Aux::Field => _ = write!(o, ".f{}", i.aux),
            Aux::Kind => _ = write!(o, ".{}", KINDS[i.aux as usize & 15]),
            Aux::Exit => o.push_str(if i.aux == 0 { ".before" } else { ".after" }),
            Aux::Array if i.aux & super::access::ARRAY != 0 => o.push_str(".array"),
            Aux::Array => {}
            Aux::Site => _ = write!(o, ".s{}", i.aux),
            Aux::SetArm => {
                o.push('.');
                TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits()).print(o);
                if i.aux & super::ops::SET_ARM != 0 {
                    o.push_str(".else");
                }
            }
            Aux::Helper => {
                let _ = write!(
                    o,
                    ".{}",
                    super::ops::HELPERS[i.aux as usize].name.to_lowercase()
                );
            }
        }
        // A generic op's speculated operand kinds.
        if matches!(i.op, Op::Arith | Op::Compare) {
            for shift in crate::jit::build::SPEC_SHIFT {
                o.push_str(match crate::jit::build::Spec::of(i.aux >> shift) {
                    crate::jit::build::Spec::I32 => ".i",
                    crate::jit::build::Spec::F64 => ".f",
                });
            }
        }
        match i.op {
            Op::K32 if i.rep == Rep::I32 => _ = write!(o, " {}", i.a as i32),
            Op::K64 => _ = write!(o, " {:#x}", i.k64()),
            Op::KF64 => _ = write!(o, " {:?}", f64::from_bits(i.k64())),
            Op::KObj => _ = write!(o, " #{}", i.a),
            // Operands by the row's forms.
            Op::K32
            | Op::Load
            | Op::Store
            | Op::LoadIdx
            | Op::StoreIdx
            | Op::Base
            | Op::Thread
            | Op::SlotLoad
            | Op::SlotStore
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Not
            | Op::Neg
            | Op::Shl
            | Op::Shr
            | Op::Sar
            | Op::SDiv
            | Op::UDiv
            | Op::ZExt32
            | Op::SExt32
            | Op::Trunc32
            | Op::Cmp
            | Op::Select
            | Op::AddOvf
            | Op::SubOvf
            | Op::MulOvf
            | Op::NegOvf
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FNeg
            | Op::FAbs
            | Op::FSqrt
            | Op::FFloor
            | Op::FCeil
            | Op::FCmp
            | Op::IToF
            | Op::LToF
            | Op::FToI
            | Op::FToL
            | Op::BoxI32
            | Op::BoxF64
            | Op::BoxPtr
            | Op::UnboxF64
            | Op::UnboxPtr
            | Op::Guard
            | Op::GuardNot
            | Op::Proved
            | Op::Deopt
            | Op::LazyExit
            | Op::FrameState
            | Op::Param
            | Op::Jump
            | Op::Br
            | Op::Ret
            | Op::CallHelper
            | Op::CheckType
            | Op::IsType
            | Op::BoxI64
            | Op::UnboxI64
            | Op::ToF64
            | Op::DivFloor
            | Op::ModFloor
            | Op::LuaShl
            | Op::LuaShr
            | Op::FToIExact
            | Op::FToLExact
            | Op::LToI
            | Op::AbsOvf
            | Op::FIDiv
            | Op::FMod
            | Op::Barrier
            | Op::NewTable
            | Op::GcCheck
            | Op::Arith
            | Op::Unm
            | Op::Compare
            | Op::Truthy
            | Op::GetIndex
            | Op::SetIndex
            | Op::GetField
            | Op::SetField => {
                for (k, kind) in info.args.iter().enumerate() {
                    let w = i.args()[k];
                    match kind {
                        Arg::No | Arg::Len => {}
                        Arg::Imm => _ = write!(o, " {w}"),
                        Arg::Blk => _ = write!(o, " b{w}"),
                        Arg::Opt | Arg::OptFs if w == u32::MAX => {}
                        Arg::Vals => {
                            for &x in self.run(i.a, i.b) {
                                match x {
                                    u32::MAX => o.push_str(" _"),
                                    x => _ = write!(o, " v{x}"),
                                }
                            }
                        }
                        Arg::Entries => {
                            let (pc, entries) = self.entries(v);
                            let _ = write!(o, " pc{pc}");
                            for &[r, x] in entries {
                                let home = if r & HOME != 0 { "*" } else { "" };
                                let _ = write!(o, " r{}={home}v{x}", r & !HOME);
                            }
                        }
                        _ => _ = write!(o, " v{w}"),
                    }
                }
            }
        }
        o.push('\n');
    }
}
