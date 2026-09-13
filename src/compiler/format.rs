use crate::env::{Prototype, Value};
use crate::instruction::{Instruction, Op, UpValueDescriptor};

pub fn format_prototype(proto: &Prototype<'_>) -> String {
    let mut out = String::new();
    format_prototype_into(&mut out, proto, 0);
    out
}

fn format_prototype_into(out: &mut String, proto: &Prototype<'_>, depth: usize) {
    let indent = "  ".repeat(depth);

    out.push_str(&format!(
        "{indent}; function (params={}, vararg={}, stack={}, upvalues={})\n",
        proto.num_params, proto.is_vararg, proto.max_stack_size, proto.num_upvalues,
    ));

    if !proto.constants.is_empty() {
        out.push_str(&format!("{indent}; constants:\n"));
        for (i, c) in proto.constants.iter().enumerate() {
            out.push_str(&format!("{indent};   K{i} = {}\n", format_value(c)));
        }
    }

    if !proto.upvalue_desc.is_empty() {
        out.push_str(&format!("{indent}; upvalues:\n"));
        for (i, desc) in proto.upvalue_desc.iter().enumerate() {
            let desc_str = match desc {
                UpValueDescriptor::ParentLocal(r) => format!("local R{r}"),
                UpValueDescriptor::ParentUpvalue(u) => format!("upvalue U{u}"),
            };
            out.push_str(&format!("{indent};   U{i} = {desc_str}\n"));
        }
    }

    out.push_str(&format!("{indent}; code:\n"));
    for (i, instr) in proto.code.iter().enumerate() {
        out.push_str(&format!(
            "{indent}{i:04}  {}\n",
            format_instruction(instr, &proto.constants)
        ));
    }

    for (i, child) in proto.prototypes.iter().enumerate() {
        out.push_str(&format!("\n{indent}; prototype {i}:\n"));
        format_prototype_into(out, child, depth + 1);
    }
}

pub(crate) fn format_value(v: &Value<'_>) -> String {
    use crate::env::ValueKind;
    match v.kind() {
        ValueKind::Nil => "nil".to_string(),
        ValueKind::Boolean => v.get_boolean().unwrap().to_string(),
        ValueKind::Integer => v.get_integer().unwrap().to_string(),
        ValueKind::Float => format!("{:?}", v.get_float().unwrap()),
        ValueKind::String => {
            let s = v.get_string().unwrap();
            match std::str::from_utf8(s.as_bytes()) {
                Ok(text) => format!("{text:?}"),
                Err(_) => format!("<bytes:{}>", s.len()),
            }
        }
        ValueKind::Table => "<table>".to_string(),
        ValueKind::Function => "<function>".to_string(),
        ValueKind::Thread => "<thread>".to_string(),
        ValueKind::Userdata => "<userdata>".to_string(),
    }
}

pub(crate) fn format_instruction(instr: &Instruction, constants: &[Value<'_>]) -> String {
    fn const_comment(constants: &[Value<'_>], idx: u16) -> String {
        if let Some(v) = constants.get(idx as usize) {
            format!("  ; {}", format_value(v))
        } else {
            String::new()
        }
    }

    match instr.op() {
        Op::MOVE => {
            let (dst, src) = instr.ab();
            format!("MOVE            R{dst} R{src}")
        }
        Op::LOAD => {
            let (dst, idx) = instr.ad();
            format!(
                "LOAD            R{dst} K{idx}{}",
                const_comment(constants, idx)
            )
        }
        Op::LFALSESKIP => {
            let src = instr.a();
            format!("LFALSESKIP      R{src}")
        }
        Op::GETUPVAL => {
            let (dst, idx) = instr.ab();
            format!("GETUPVAL        R{dst} U{idx}")
        }
        Op::SETUPVAL => {
            let (src, idx) = instr.ab();
            format!("SETUPVAL        R{src} U{idx}")
        }
        Op::GETTABUP => {
            let (dst, idx, _, key) = instr.abde();
            format!(
                "GETTABUP        R{dst} U{idx} K{key}{}",
                const_comment(constants, key)
            )
        }
        Op::SETTABUP => {
            let (src, idx, _, key) = instr.abde();
            format!(
                "SETTABUP        R{src} U{idx} K{key}{}",
                const_comment(constants, key)
            )
        }
        Op::GETTABLE => {
            let (dst, table, key) = instr.abc();
            format!("GETTABLE        R{dst} R{table} R{key}")
        }
        Op::SETTABLE => {
            let (src, table, key) = instr.abc();
            format!("SETTABLE        R{src} R{table} R{key}")
        }
        Op::GETFIELD => {
            let (dst, table, _, key_idx) = instr.abde();
            format!(
                "GETFIELD        R{dst} R{table} K{key_idx}{}",
                const_comment(constants, key_idx)
            )
        }
        Op::SETFIELD => {
            let (src, table, _, key_idx) = instr.abde();
            format!(
                "SETFIELD        R{src} R{table} K{key_idx}{}",
                const_comment(constants, key_idx)
            )
        }
        Op::SELF => {
            let (dst, object, key_idx) = instr.abd();
            format!(
                "SELF            R{dst} R{object} K{key_idx}{}",
                const_comment(constants, key_idx)
            )
        }
        Op::NEWTABLE => {
            let dst = instr.a();
            format!("NEWTABLE        R{dst}")
        }
        Op::ADD => {
            let (dst, lhs, rhs) = instr.abc();
            format!("ADD             R{dst} R{lhs} R{rhs}")
        }
        Op::SUB => {
            let (dst, lhs, rhs) = instr.abc();
            format!("SUB             R{dst} R{lhs} R{rhs}")
        }
        Op::MUL => {
            let (dst, lhs, rhs) = instr.abc();
            format!("MUL             R{dst} R{lhs} R{rhs}")
        }
        Op::MOD => {
            let (dst, lhs, rhs) = instr.abc();
            format!("MOD             R{dst} R{lhs} R{rhs}")
        }
        Op::POW => {
            let (dst, lhs, rhs) = instr.abc();
            format!("POW             R{dst} R{lhs} R{rhs}")
        }
        Op::DIV => {
            let (dst, lhs, rhs) = instr.abc();
            format!("DIV             R{dst} R{lhs} R{rhs}")
        }
        Op::IDIV => {
            let (dst, lhs, rhs) = instr.abc();
            format!("IDIV            R{dst} R{lhs} R{rhs}")
        }
        Op::BAND => {
            let (dst, lhs, rhs) = instr.abc();
            format!("BAND            R{dst} R{lhs} R{rhs}")
        }
        Op::BOR => {
            let (dst, lhs, rhs) = instr.abc();
            format!("BOR             R{dst} R{lhs} R{rhs}")
        }
        Op::BXOR => {
            let (dst, lhs, rhs) = instr.abc();
            format!("BXOR            R{dst} R{lhs} R{rhs}")
        }
        Op::SHL => {
            let (dst, lhs, rhs) = instr.abc();
            format!("SHL             R{dst} R{lhs} R{rhs}")
        }
        Op::SHR => {
            let (dst, lhs, rhs) = instr.abc();
            format!("SHR             R{dst} R{lhs} R{rhs}")
        }
        Op::UNM => {
            let (dst, src) = instr.ab();
            format!("UNM             R{dst} R{src}")
        }
        Op::BNOT => {
            let (dst, src) = instr.ab();
            format!("BNOT            R{dst} R{src}")
        }
        Op::NOT => {
            let (dst, src) = instr.ab();
            format!("NOT             R{dst} R{src}")
        }
        Op::LEN => {
            let (dst, src) = instr.ab();
            format!("LEN             R{dst} R{src}")
        }
        Op::CONCAT => {
            let (dst, lhs, rhs) = instr.abc();
            format!("CONCAT          R{dst} R{lhs} R{rhs}")
        }
        Op::CLOSE => {
            let start = instr.a();
            format!("CLOSE           R{start}")
        }
        Op::TBC => {
            let val = instr.a();
            format!("TBC             R{val}")
        }
        Op::JMP => {
            let offset = instr.imm();
            format!("JMP             {offset:+}")
        }
        Op::EQ => {
            let (lhs, rhs, inverted) = instr.abc_flag();
            format!("EQ              R{lhs} R{rhs} inv={inverted}")
        }
        Op::LT => {
            let (lhs, rhs, inverted) = instr.abc_flag();
            format!("LT              R{lhs} R{rhs} inv={inverted}")
        }
        Op::LE => {
            let (lhs, rhs, inverted) = instr.abc_flag();
            format!("LE              R{lhs} R{rhs} inv={inverted}")
        }
        Op::TEST => {
            let (src, inverted) = instr.ab_flag();
            format!("TEST            R{src} inv={inverted}")
        }
        Op::TESTSET => {
            let (dst, src, inverted) = instr.abc_flag();
            format!("TESTSET         R{dst} R{src} inv={inverted}")
        }
        Op::CALL => {
            let (func, args, returns) = instr.abc();
            format!("CALL            R{func} args={args} ret={returns}")
        }
        Op::TAILCALL => {
            let (func, args) = instr.ab();
            format!("TAILCALL        R{func} args={args}")
        }
        Op::RETURN => {
            let (values, count) = instr.ab();
            format!("RETURN          R{values} count={count}")
        }
        Op::FORLOOP => {
            let (base, offset) = instr.a_imm();
            format!("FORLOOP         R{base} {offset:+}")
        }
        Op::FORPREP => {
            let (base, offset) = instr.a_imm();
            format!("FORPREP         R{base} {offset:+}")
        }
        Op::TFORPREP => {
            let (base, offset) = instr.a_imm();
            format!("TFORPREP        R{base} {offset:+}")
        }
        Op::TFORCALL => {
            let (base, count) = instr.ab();
            format!("TFORCALL        R{base} count={count}")
        }
        Op::TFORLOOP => {
            let (base, offset) = instr.a_imm();
            format!("TFORLOOP        R{base} {offset:+}")
        }
        Op::SETLIST => {
            let (table, count, offset) = instr.abd();
            format!("SETLIST         R{table} count={count} offset={offset}")
        }
        Op::CLOSURE => {
            let (dst, proto) = instr.ad();
            format!("CLOSURE         R{dst} P{proto}")
        }
        Op::VARARG => {
            let (dst, count) = instr.ab();
            format!("VARARG          R{dst} count={count}")
        }
        Op::VARARGGET => {
            let (dst, base, key) = instr.abc();
            format!("VARARGGET       R{dst} R{base} R{key}")
        }
        Op::VARARGPREP => {
            let num_fixed = instr.a();
            format!("VARARGPREP      fixed={num_fixed}")
        }
        Op::ERRNNIL => {
            let (src, name_key) = instr.ad();
            format!("ERRNNIL         R{src} name=K{name_key}")
        }
        Op::NOP => "NOP".to_string(),
        Op::STOP => "STOP".to_string(),
    }
}
