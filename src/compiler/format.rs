use crate::env::{Prototype, Value};
use crate::instruction::{Instruction, UpValueDescriptor};

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

fn format_value(v: &Value<'_>) -> String {
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

fn format_instruction(instr: &Instruction, constants: &[Value<'_>]) -> String {
    fn const_comment(constants: &[Value<'_>], idx: u16) -> String {
        if let Some(v) = constants.get(idx as usize) {
            format!("  ; {}", format_value(v))
        } else {
            String::new()
        }
    }

    match *instr {
        Instruction::MOVE { dst, src } => format!("MOVE            R{dst} R{src}"),
        Instruction::LOAD { dst, idx } => {
            format!(
                "LOAD            R{dst} K{idx}{}",
                const_comment(constants, idx)
            )
        }
        Instruction::LFALSESKIP { src } => format!("LFALSESKIP      R{src}"),
        Instruction::GETUPVAL { dst, idx } => format!("GETUPVAL        R{dst} U{idx}"),
        Instruction::SETUPVAL { src, idx } => format!("SETUPVAL        R{src} U{idx}"),
        Instruction::GETTABUP { dst, idx, key } => {
            format!(
                "GETTABUP        R{dst} U{idx} K{key}{}",
                const_comment(constants, key)
            )
        }
        Instruction::SETTABUP { src, idx, key } => {
            format!(
                "SETTABUP        R{src} U{idx} K{key}{}",
                const_comment(constants, key)
            )
        }
        Instruction::GETTABLE { dst, table, key } => {
            format!("GETTABLE        R{dst} R{table} R{key}")
        }
        Instruction::SETTABLE { src, table, key } => {
            format!("SETTABLE        R{src} R{table} R{key}")
        }
        Instruction::GETFIELD {
            dst,
            table,
            key_idx,
        } => {
            format!(
                "GETFIELD        R{dst} R{table} K{key_idx}{}",
                const_comment(constants, key_idx)
            )
        }
        Instruction::SETFIELD {
            src,
            table,
            key_idx,
        } => {
            format!(
                "SETFIELD        R{src} R{table} K{key_idx}{}",
                const_comment(constants, key_idx)
            )
        }
        Instruction::NEWTABLE { dst } => format!("NEWTABLE        R{dst}"),
        Instruction::ADD { dst, lhs, rhs } => format!("ADD             R{dst} R{lhs} R{rhs}"),
        Instruction::SUB { dst, lhs, rhs } => format!("SUB             R{dst} R{lhs} R{rhs}"),
        Instruction::MUL { dst, lhs, rhs } => format!("MUL             R{dst} R{lhs} R{rhs}"),
        Instruction::MOD { dst, lhs, rhs } => format!("MOD             R{dst} R{lhs} R{rhs}"),
        Instruction::POW { dst, lhs, rhs } => format!("POW             R{dst} R{lhs} R{rhs}"),
        Instruction::DIV { dst, lhs, rhs } => format!("DIV             R{dst} R{lhs} R{rhs}"),
        Instruction::IDIV { dst, lhs, rhs } => format!("IDIV            R{dst} R{lhs} R{rhs}"),
        Instruction::BAND { dst, lhs, rhs } => format!("BAND            R{dst} R{lhs} R{rhs}"),
        Instruction::BOR { dst, lhs, rhs } => format!("BOR             R{dst} R{lhs} R{rhs}"),
        Instruction::BXOR { dst, lhs, rhs } => format!("BXOR            R{dst} R{lhs} R{rhs}"),
        Instruction::SHL { dst, lhs, rhs } => format!("SHL             R{dst} R{lhs} R{rhs}"),
        Instruction::SHR { dst, lhs, rhs } => format!("SHR             R{dst} R{lhs} R{rhs}"),
        Instruction::UNM { dst, src } => format!("UNM             R{dst} R{src}"),
        Instruction::BNOT { dst, src } => format!("BNOT            R{dst} R{src}"),
        Instruction::NOT { dst, src } => format!("NOT             R{dst} R{src}"),
        Instruction::LEN { dst, src } => format!("LEN             R{dst} R{src}"),
        Instruction::CONCAT { dst, lhs, rhs } => {
            format!("CONCAT          R{dst} R{lhs} R{rhs}")
        }
        Instruction::CLOSE { start } => format!("CLOSE           R{start}"),
        Instruction::TBC { val } => format!("TBC             R{val}"),
        Instruction::JMP { offset } => format!("JMP             {offset:+}"),
        Instruction::EQ { lhs, rhs, inverted } => {
            format!("EQ              R{lhs} R{rhs} inv={inverted}")
        }
        Instruction::LT { lhs, rhs, inverted } => {
            format!("LT              R{lhs} R{rhs} inv={inverted}")
        }
        Instruction::LE { lhs, rhs, inverted } => {
            format!("LE              R{lhs} R{rhs} inv={inverted}")
        }
        Instruction::TEST { src, inverted } => {
            format!("TEST            R{src} inv={inverted}")
        }
        Instruction::TESTSET { dst, src, inverted } => {
            format!("TESTSET         R{dst} R{src} inv={inverted}")
        }
        Instruction::CALL {
            func,
            args,
            returns,
        } => {
            format!("CALL            R{func} args={args} ret={returns}")
        }
        Instruction::TAILCALL { func, args } => {
            format!("TAILCALL        R{func} args={args}")
        }
        Instruction::RETURN { values, count } => {
            format!("RETURN          R{values} count={count}")
        }
        Instruction::FORLOOP { base, offset } => {
            format!("FORLOOP         R{base} {offset:+}")
        }
        Instruction::FORPREP { base, offset } => {
            format!("FORPREP         R{base} {offset:+}")
        }
        Instruction::TFORPREP { base, offset } => {
            format!("TFORPREP        R{base} {offset:+}")
        }
        Instruction::TFORCALL { base, count } => {
            format!("TFORCALL        R{base} count={count}")
        }
        Instruction::TFORLOOP { base, offset } => {
            format!("TFORLOOP        R{base} {offset:+}")
        }
        Instruction::SETLIST {
            table,
            count,
            offset,
        } => {
            format!("SETLIST         R{table} count={count} offset={offset}")
        }
        Instruction::CLOSURE { dst, proto } => {
            format!("CLOSURE         R{dst} P{proto}")
        }
        Instruction::VARARG { dst, count } => format!("VARARG          R{dst} count={count}"),
        Instruction::VARARGPREP { num_fixed } => {
            format!("VARARGPREP      fixed={num_fixed}")
        }
        Instruction::ERRNNIL { src, name_key } => {
            format!("ERRNNIL         R{src} name=K{name_key}")
        }
        Instruction::NOP => "NOP".to_string(),
        Instruction::STOP => "STOP".to_string(),
    }
}
