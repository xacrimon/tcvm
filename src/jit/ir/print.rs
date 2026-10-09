//! A one-line-per-instruction listing of a `Func`.

use std::fmt::Write;

use crate::jit::ir::{ExitKind, Func, NO_SNAP};

impl Func<'_> {
    pub(crate) fn print(&self) -> String {
        let mut out = String::new();
        let order = self.rpo();
        for b in order {
            let bd = &self.blocks[b.idx()];
            let params: Vec<String> = bd
                .params
                .iter()
                .map(|&p| format!("v{}: {}", p.0, self.ty(p)))
                .collect();
            let _ = writeln!(
                out,
                "b{}({}){}{}:",
                b.0,
                params.join(", "),
                if bd.resume { " resume" } else { "" },
                if b == self.entry { " entry" } else { "" }
            );
            for &i in &bd.insts {
                let d = &self.insts[i.idx()];
                let _ = write!(out, "  ");
                let res: Vec<String> = self
                    .results(i)
                    .map(|v| format!("v{}: {}", v.0, self.ty(v)))
                    .collect();
                if !res.is_empty() {
                    let _ = write!(out, "{} = ", res.join(", "));
                }
                let _ = write!(out, "{}", d.op.name());
                let args: Vec<String> = self.args(i).iter().map(|a| format!("v{}", a.0)).collect();
                if !args.is_empty() {
                    let _ = write!(out, " {}", args.join(", "));
                }
                for e in self.edges(i) {
                    let a: Vec<String> = e.args.iter().map(|a| format!("v{}", a.0)).collect();
                    let _ = write!(out, " -> b{}({})", e.target.0, a.join(", "));
                }
                if d.snap != NO_SNAP {
                    let s = &self.snaps[d.snap as usize];
                    let kind = match s.kind {
                        ExitKind::Before => "before",
                        ExitKind::After => "after",
                        ExitKind::Gc => "gc",
                    };
                    let ents: Vec<String> = s
                        .entries
                        .iter()
                        .map(|&(r, v)| format!("r{r}=v{}", v.0))
                        .collect();
                    let _ = write!(out, "  [{:?} {kind} pc{} {}]", d.tag, s.pc, ents.join(" "));
                }
                let _ = writeln!(out);
            }
        }
        out
    }
}
