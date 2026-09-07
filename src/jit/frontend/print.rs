//! Textual form of the **bytecode** CFG — the *input* to lowering, not the SSA
//! IR. Blocks here hold raw `Instruction`s addressing Lua registers (`R0`);
//! the SSA IR that lowering produces from them is printed by
//! `crate::jit::ir::print`, in terms of values (`v0`) and block parameters.
//!
//! Walks a prototype and every nested prototype, printing each one's CFG — or,
//! when the region is declined, *why*. Making the decline reason part of the
//! snapshot is deliberate: what the JIT refuses to compile is as much a
//! property worth pinning down as what it accepts.

use std::fmt::Write;

use crate::compiler::format::format_instruction;
use crate::env::Prototype;
use crate::jit::frontend::cfg::{self, Cfg};

pub fn format_cfgs(proto: &Prototype<'_>) -> String {
    let mut out = String::new();
    format_proto(&mut out, proto, "main");
    out
}

fn format_proto(out: &mut String, proto: &Prototype<'_>, name: &str) {
    let _ = writeln!(
        out,
        "=== bytecode cfg: {name} (params={}, vararg={}, stack={})",
        proto.num_params, proto.is_vararg, proto.max_stack_size
    );

    match cfg::build(&proto.code) {
        Ok(cfg) => format_cfg(out, &cfg, proto),
        Err(reason) => {
            let _ = writeln!(out, "  declined: {reason:?}");
        }
    }
    let _ = writeln!(out);

    for (n, sub) in proto.prototypes.iter().enumerate() {
        format_proto(out, sub, &format!("{name}/P{n}"));
    }
}

fn format_cfg(out: &mut String, cfg: &Cfg, proto: &Prototype<'_>) {
    for b in &cfg.blocks {
        let succs: Vec<String> = b.succs.iter().map(|s| format!("b{s}")).collect();
        let live: Vec<String> = b.live_in.iter().map(|r| format!("R{r}")).collect();
        // Printed only where non-empty: a register a predecessor's terminator
        // assigns on one edge only is rare, and always worth noticing.
        let edge = if b.edge_params.is_empty() {
            String::new()
        } else {
            let regs: Vec<String> = b.edge_params.iter().map(|r| format!("R{r}")).collect();
            format!(" edge_params=[{}]", regs.join(", "))
        };
        let _ = writeln!(
            out,
            "  b{}: succs=[{}] live_in=[{}]{edge}",
            b.start,
            succs.join(", "),
            live.join(", ")
        );
        for pc in b.start..b.end {
            let _ = writeln!(
                out,
                "    {pc:04}  {}",
                format_instruction(&proto.code[pc as usize], &proto.constants)
            );
        }
    }
}
