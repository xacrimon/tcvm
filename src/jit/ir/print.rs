//! Textual form of the IR. This is the debugging surface for every later pass,
//! and what the frontend's tests assert against.

use std::fmt::Write;

use crate::compiler::format::format_value;
use crate::jit::ir::op::{ArithKind, Cc, FloatOp, IntOp, Op};
use crate::jit::ir::ty::{Refine, Rep, Ty, TypeSet};
use crate::jit::ir::{Block, Func, Inst};

pub fn print_func(f: &Func<'_>) -> String {
    let mut out = String::new();
    for b in f.blocks() {
        print_block(f, b, &mut out);
    }
    out
}

fn print_block(f: &Func<'_>, b: Block, out: &mut String) {
    let data = f.block(b);
    let _ = write!(out, "block{}", b.0);
    if !data.params.is_empty() {
        let params: Vec<String> = data
            .params
            .iter()
            .map(|&v| format!("v{}: {}", v.0, fmt_ty(f, f.ty(v))))
            .collect();
        let _ = write!(out, "({})", params.join(", "));
    }
    let _ = writeln!(out, ":");
    for &i in &data.insts {
        print_inst(f, i, out);
    }
}

fn print_inst(f: &Func<'_>, i: Inst, out: &mut String) {
    let d = f.inst(i);

    let lhs = if d.results.is_empty() {
        " ".repeat(8)
    } else {
        let names: Vec<String> = d.results.iter().map(|v| format!("v{}", v.0)).collect();
        format!("{:>5} = ", names.join(", "))
    };

    let args: Vec<String> = d.args.iter().map(|v| format!("v{}", v.0)).collect();
    let body = fmt_op(f, d.op, &args);

    let targets: Vec<String> = d
        .targets
        .iter()
        .map(|t| {
            let a: Vec<String> = t.args.iter().map(|v| format!("v{}", v.0)).collect();
            if a.is_empty() {
                format!("block{}", t.block.0)
            } else {
                format!("block{}({})", t.block.0, a.join(", "))
            }
        })
        .collect();

    let _ = write!(out, "    {lhs}{body}");
    if !targets.is_empty() {
        let _ = write!(out, " {}", targets.join(", "));
    }
    if let Some(e) = d.exit {
        let _ = write!(out, "  ; exit{}", e.0);
    }
    let _ = writeln!(out);
}

fn fmt_op(f: &Func<'_>, op: Op, args: &[String]) -> String {
    let a = |n: usize| args.get(n).cloned().unwrap_or_else(|| "?".into());
    let all = args.join(", ");

    match op {
        Op::KConst(c) => format!("kconst {}", format_value(&f.pool.value(c))),
        Op::IConst(v) => format!("iconst {v}"),
        Op::FConst(bits) => format!("fconst {}", f64::from_bits(bits)),
        Op::BConst(v) => format!("bconst {v}"),

        Op::PackInt => format!("pack.int {}", a(0)),
        Op::PackFloat => format!("pack.float {}", a(0)),
        Op::PackBool => format!("pack.bool {}", a(0)),
        Op::UnpackInt => format!("unpack.int {}", a(0)),
        Op::UnpackFloat => format!("unpack.float {}", a(0)),
        Op::UnpackPtr => format!("unpack.ptr {}", a(0)),
        Op::TagOf => format!("tag.of {}", a(0)),
        Op::IsType(s) => format!("is.type {}, {}", a(0), fmt_set(s)),
        Op::IsFalsy => format!("is.falsy {}", a(0)),

        Op::GuardType(s) => format!("guard.type {}, {}", a(0), fmt_set(s)),
        Op::GuardShape(s) => format!("guard.shape {}, S{}", a(0), s.0),
        Op::GuardCond => format!("guard.cond {}", a(0)),
        Op::AssumeNoMm(s, bits) => {
            let names: Vec<&str> = bits.iter_names().map(|(n, _)| n).collect();
            format!("assume.no_mm S{}, {}", s.0, names.join("|"))
        }

        Op::LuaArith(k) => format!("lua.{} {all}", arith_name(k)),
        Op::LuaCmp(cc) => format!("lua.cmp {} {all}", cc_name(cc)),
        Op::LuaEq => format!("lua.eq {all}"),
        Op::LuaConcat => format!("lua.concat {all}"),
        Op::LuaLen => format!("lua.len {all}"),
        Op::LuaGetIndex => format!("lua.getindex {all}"),
        Op::LuaSetIndex => format!("lua.setindex {all}"),

        Op::IntArith(k) => format!("{}.i64 {all}", int_name(k)),
        Op::FloatArith(k) => format!("{}.f64 {all}", float_name(k)),
        Op::ICmp(cc) => format!("icmp {} {all}", cc_name(cc)),
        Op::FCmp(cc) => format!("fcmp {} {all}", cc_name(cc)),
        Op::SiToFp => format!("sitofp {}", a(0)),
        Op::FpToIntExact => format!("fp_to_int_exact {}", a(0)),

        Op::TabNew { array_hint } => format!("tab.new hint={array_hint}"),
        Op::TabProps => format!("tab.props {}", a(0)),
        Op::SlotGet(slot) => format!("slot.get {}, {slot}", a(0)),
        Op::SlotSet(slot) => format!("slot.set {}, {slot}, {}", a(0), a(1)),
        Op::TabArr => format!("tab.arr {}", a(0)),
        Op::TabArrLen => format!("tab.arr_len {}", a(0)),
        Op::ArrGet => format!("arr.get {}, {}", a(0), a(1)),
        Op::ArrSet => format!("arr.set {}, {}, {}", a(0), a(1), a(2)),
        Op::TabHashGet => format!("tab.hash_get {}, {}", a(0), a(1)),

        Op::GcBarrierBack => format!("gc.barrier_back {}", a(0)),
        Op::GcBarrierFwd => format!("gc.barrier_fwd {}, {}", a(0), a(1)),

        Op::UpvalCell(idx) => format!("upval.cell {idx}"),
        Op::UpvalGet => format!("upval.get {}", a(0)),
        Op::UpvalSet => format!("upval.set {}, {}", a(0), a(1)),
        Op::UpvalClose(start) => format!("upval.close {start}"),
        Op::StackGet(reg) => format!("stack.get r{reg}"),
        Op::StackSet(reg) => format!("stack.set r{reg}, {}", a(0)),

        Op::Call { nret } => format!("call {all} -> {nret}"),
        Op::ClosureNew(p) => format!("closure.new P{} {all}", p.0),

        Op::GetGlobal(s) => format!(
            "getglobal {}",
            String::from_utf8_lossy(f.pool.string(s).as_bytes())
        ),
        Op::SetGlobal(s) => format!(
            "setglobal {}, {}",
            String::from_utf8_lossy(f.pool.string(s).as_bytes()),
            a(0)
        ),

        Op::Jump => "jump".into(),
        Op::Br => format!("br {}", a(0)),
        Op::Ret => format!("ret {all}"),
        Op::Deopt => "deopt".into(),
        Op::Safepoint => "safepoint".into(),
    }
}

fn fmt_ty(f: &Func<'_>, t: Ty) -> String {
    let base = match t.rep {
        Rep::I64 => "i64".to_string(),
        Rep::F64 => "f64".to_string(),
        Rep::B1 => "b1".to_string(),
        Rep::Ptr => "ptr".to_string(),
        Rep::Val => fmt_set(t.set),
    };
    match t.refine {
        Refine::None => base,
        Refine::Const(c) => format!("{base}<{}>", format_value(&f.pool.value(c))),
        Refine::Shape(s) => format!("{base}<S{}>", s.0),
        Refine::Proto(p) => format!("{base}<P{}>", p.0),
    }
}

fn fmt_set(s: TypeSet) -> String {
    if s == TypeSet::ANY {
        return "any".into();
    }
    if s == TypeSet::NUM {
        return "num".into();
    }
    if s == TypeSet::BOOL {
        return "bool".into();
    }
    if s.is_empty() {
        return "none".into();
    }
    const NAMES: &[(TypeSet, &str)] = &[
        (TypeSet::NIL, "nil"),
        (TypeSet::FALSE, "false"),
        (TypeSet::TRUE, "true"),
        (TypeSet::INT, "int"),
        (TypeSet::FLOAT, "float"),
        (TypeSet::STR, "str"),
        (TypeSet::TAB, "tab"),
        (TypeSet::FUN, "fun"),
        (TypeSet::THR, "thr"),
        (TypeSet::UDATA, "udata"),
    ];
    NAMES
        .iter()
        .filter(|(bit, _)| s.contains(*bit))
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join("|")
}

fn arith_name(k: ArithKind) -> &'static str {
    match k {
        ArithKind::Add => "add",
        ArithKind::Sub => "sub",
        ArithKind::Mul => "mul",
        ArithKind::Mod => "mod",
        ArithKind::Pow => "pow",
        ArithKind::Div => "div",
        ArithKind::IDiv => "idiv",
        ArithKind::BAnd => "band",
        ArithKind::BOr => "bor",
        ArithKind::BXor => "bxor",
        ArithKind::Shl => "shl",
        ArithKind::Shr => "shr",
        ArithKind::Unm => "unm",
        ArithKind::BNot => "bnot",
    }
}

fn int_name(k: IntOp) -> &'static str {
    match k {
        IntOp::Add => "add",
        IntOp::Sub => "sub",
        IntOp::Mul => "mul",
        IntOp::IDiv => "idiv",
        IntOp::Mod => "mod",
        IntOp::BAnd => "band",
        IntOp::BOr => "bor",
        IntOp::BXor => "bxor",
        IntOp::Shl => "shl",
        IntOp::Shr => "shr",
        IntOp::Neg => "neg",
        IntOp::BNot => "bnot",
    }
}

fn float_name(k: FloatOp) -> &'static str {
    match k {
        FloatOp::Add => "add",
        FloatOp::Sub => "sub",
        FloatOp::Mul => "mul",
        FloatOp::Div => "div",
        FloatOp::IDiv => "idiv",
        FloatOp::Mod => "mod",
        FloatOp::Pow => "pow",
        FloatOp::Neg => "neg",
    }
}

fn cc_name(cc: Cc) -> &'static str {
    match cc {
        Cc::Eq => "eq",
        Cc::Ne => "ne",
        Cc::Lt => "lt",
        Cc::Le => "le",
        Cc::Gt => "gt",
        Cc::Ge => "ge",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::shape::MetamethodBits;
    use crate::jit::ir::op::IntOp;
    use crate::jit::ir::pool::ShapeRef;
    use crate::jit::ir::ty::{Refine, TypeContext};
    use crate::jit::ir::{BlockCall, Exit, FrameState, Func, InstData, Val};

    fn inst(op: Op, args: Vec<Val>) -> InstData {
        InstData {
            op,
            args,
            targets: Vec::new(),
            results: Vec::new(),
            fs: None,
            exit: None,
        }
    }

    /// Hand-build `for i = 1, n do s = s + t.x end` as it should look after the
    /// frontend, given an OSR entry with `i: int, n: int, s: int, t: tab<S0>`.
    ///
    /// Doubles as a check that the IR is constructible with no arena — the
    /// whole point of keeping `'gc` behind the constant pool.
    #[test]
    fn numeric_loop_over_shaped_table() {
        let mut f = Func::new();
        let s0 = ShapeRef(0);
        let entry = f.entry;

        let i0 = f.append_param(entry, Ty::boxed(TypeSet::INT));
        let n0 = f.append_param(entry, Ty::boxed(TypeSet::INT));
        let s_0 = f.append_param(entry, Ty::boxed(TypeSet::INT));
        let t0 = f.append_param(entry, Ty::boxed(TypeSet::TAB));

        let fs = f.add_frame_state(FrameState {
            pc: 0,
            regs: vec![Some(i0), Some(n0), Some(s_0), Some(t0)],
            parent: None,
        });
        let exit = f.add_exit(Exit {
            fs,
            ctx: TypeContext::default(),
            count: 0,
        });

        let mut guard = inst(Op::GuardShape(s0), vec![t0]);
        guard.fs = Some(fs);
        guard.exit = Some(exit);
        let (_, r) = f.append_inst(entry, guard, &[Ty::with_shape(s0)]);
        let t1 = r[0];

        f.append_inst(
            entry,
            inst(Op::AssumeNoMm(s0, MetamethodBits::INDEX), vec![]),
            &[],
        );
        let (_, r) = f.append_inst(entry, inst(Op::TabProps, vec![t1]), &[Ty::PTR]);
        let props = r[0];

        let (_, r) = f.append_inst(entry, inst(Op::UnpackInt, vec![i0]), &[Ty::I64]);
        let i = r[0];
        let (_, r) = f.append_inst(entry, inst(Op::UnpackInt, vec![n0]), &[Ty::I64]);
        let n = r[0];
        let (_, r) = f.append_inst(entry, inst(Op::UnpackInt, vec![s_0]), &[Ty::I64]);
        let s = r[0];

        let header = f.new_block();
        let mut jump = inst(Op::Jump, vec![]);
        jump.targets = vec![BlockCall {
            block: header,
            args: vec![i, s],
        }];
        f.append_inst(entry, jump, &[]);

        // Loop header: the body carries no tag checks, no IC compare, no
        // metamethod bit test.
        let hi = f.append_param(header, Ty::I64);
        let hs = f.append_param(header, Ty::I64);

        let hfs = f.add_frame_state(FrameState {
            pc: 12,
            regs: vec![Some(hi), Some(n), Some(hs), Some(t1)],
            parent: None,
        });
        let mut sp = inst(Op::Safepoint, vec![]);
        sp.fs = Some(hfs);
        f.append_inst(header, sp, &[]);

        let (_, r) = f.append_inst(header, inst(Op::SlotGet(3), vec![props]), &[Ty::ANY]);
        let x = r[0];

        let hexit = f.add_exit(Exit {
            fs: hfs,
            ctx: TypeContext::default(),
            count: 0,
        });
        let mut g = inst(Op::GuardType(TypeSet::INT), vec![x]);
        g.fs = Some(hfs);
        g.exit = Some(hexit);
        let (_, r) = f.append_inst(header, g, &[Ty::boxed(TypeSet::INT)]);
        let x1 = r[0];

        let (_, r) = f.append_inst(header, inst(Op::UnpackInt, vec![x1]), &[Ty::I64]);
        let x2 = r[0];
        let (_, r) = f.append_inst(
            header,
            inst(Op::IntArith(IntOp::Add), vec![hs, x2]),
            &[Ty::I64],
        );
        let s_next = r[0];

        let (_, r) = f.append_inst(header, inst(Op::IConst(1), vec![]), &[Ty::I64]);
        let one = r[0];
        let (_, r) = f.append_inst(
            header,
            inst(Op::IntArith(IntOp::Add), vec![hi, one]),
            &[Ty::I64],
        );
        let i_next = r[0];
        let (_, r) = f.append_inst(header, inst(Op::ICmp(Cc::Le), vec![i_next, n]), &[Ty::B1]);
        let cond = r[0];

        let done = f.new_block();
        let mut br = inst(Op::Br, vec![cond]);
        br.targets = vec![
            BlockCall {
                block: header,
                args: vec![i_next, s_next],
            },
            BlockCall {
                block: done,
                args: vec![s_next],
            },
        ];
        f.append_inst(header, br, &[]);

        let ds = f.append_param(done, Ty::I64);
        let (_, r) = f.append_inst(
            done,
            inst(Op::PackInt, vec![ds]),
            &[Ty::boxed(TypeSet::INT)],
        );
        f.append_inst(done, inst(Op::Ret, vec![r[0]]), &[]);

        // The guard refined `t0: tab` into `t1: tab<S0>` — that refinement is
        // what lets `slot.get` stand alone in the loop body.
        assert_eq!(f.ty(t1).refine, Refine::Shape(s0));
        assert!(f.ty(hi).set.is_monomorphic());

        if let Err(e) = crate::jit::ir::verify::verify(&f) {
            panic!("hand-built IR failed verification:\n{e}");
        }

        let text = print_func(&f);
        let expected = "\
block0(v0: int, v1: int, v2: int, v3: tab):
       v4 = guard.shape v3, S0  ; exit0
            assume.no_mm S0, INDEX
       v5 = tab.props v4
       v6 = unpack.int v0
       v7 = unpack.int v1
       v8 = unpack.int v2
            jump block1(v6, v8)
block1(v9: i64, v10: i64):
            safepoint
      v11 = slot.get v5, 3
      v12 = guard.type v11, int  ; exit1
      v13 = unpack.int v12
      v14 = add.i64 v10, v13
      v15 = iconst 1
      v16 = add.i64 v9, v15
      v17 = icmp le v16, v7
            br v17 block1(v16, v14), block2(v14)
block2(v18: i64):
      v19 = pack.int v18
            ret v19
";
        assert_eq!(text, expected, "\n--- got ---\n{text}");
    }
}
