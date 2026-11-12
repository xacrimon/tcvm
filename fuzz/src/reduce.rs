//! Shrinking a program that shows a finding: first on the generator's tree (whole units, then
//! statements, then expressions down to their operands), where every candidate is still a
//! program, then on the text's lines, which reaches the prelude and the drivers.

use crate::ast::{Block, Expr, Item, Stmt};
use crate::generate::Program;

/// Statements in `b`, nested ones included.
fn count(b: &Block) -> usize {
    b.iter()
        .map(|s| 1 + children(s).iter().map(|c| count(c)).sum::<usize>())
        .sum()
}

fn children(s: &Stmt) -> Vec<&Block> {
    match s {
        Stmt::Do(b)
        | Stmt::While(_, b)
        | Stmt::Repeat(b, _)
        | Stmt::NumFor(.., b)
        | Stmt::GenFor(.., b) => vec![b],
        Stmt::If(arms, els) => arms.iter().map(|(_, b)| b).chain(els).collect(),
        Stmt::LocalFunc(_, f) | Stmt::FuncDef(.., f) => vec![&f.body],
        _ => vec![],
    }
}

fn children_mut(s: &mut Stmt) -> Vec<&mut Block> {
    match s {
        Stmt::Do(b)
        | Stmt::While(_, b)
        | Stmt::Repeat(b, _)
        | Stmt::NumFor(.., b)
        | Stmt::GenFor(.., b) => vec![b],
        Stmt::If(arms, els) => arms.iter_mut().map(|(_, b)| b).chain(els).collect(),
        Stmt::LocalFunc(_, f) | Stmt::FuncDef(.., f) => vec![&mut f.body],
        _ => vec![],
    }
}

/// Remove the `n`th statement of `b` in preorder, or with `unwrap` put its first block in its
/// place. Whether it did.
fn edit(b: &mut Block, n: &mut usize, unwrap: bool) -> bool {
    let mut i = 0;
    while i < b.len() {
        if *n == 0 {
            if !unwrap {
                b.remove(i);
                return true;
            }
            let Some(inner) = children(&b[i]).first().map(|c| (*c).clone()) else {
                return false;
            };
            // A function's body is no block of the code around it.
            if matches!(b[i], Stmt::LocalFunc(..) | Stmt::FuncDef(..)) {
                return false;
            }
            b.splice(i..=i, inner);
            return true;
        }
        *n -= 1;
        for c in children_mut(&mut b[i]) {
            if edit(c, n, unwrap) {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn exprs_of(s: &mut Stmt) -> Vec<&mut Expr> {
    match s {
        Stmt::Local(_, es) | Stmt::Return(es) => es.iter_mut().collect(),
        Stmt::Assign(ts, es) => ts.iter_mut().chain(es).collect(),
        Stmt::Call(e) | Stmt::While(e, _) | Stmt::Repeat(_, e) => vec![e],
        Stmt::If(arms, _) => arms.iter_mut().map(|(c, _)| c).collect(),
        Stmt::NumFor(_, a, b, c, _) => [a, b].into_iter().chain(c).collect(),
        Stmt::GenFor(_, es, _) => es.iter_mut().collect(),
        _ => vec![],
    }
}

fn operands(e: &mut Expr) -> Vec<&mut Expr> {
    match e {
        Expr::Index(a, b) | Expr::Bin(_, a, b) => vec![a, b],
        Expr::Field(a, _) | Expr::Un(_, a) | Expr::Paren(a) => vec![a],
        Expr::Call(f, args) => std::iter::once(&mut **f).chain(args).collect(),
        Expr::Method(o, _, args) => std::iter::once(&mut **o).chain(args).collect(),
        Expr::Table(items) => (items.iter_mut())
            .map(|i| match i {
                Item::Pos(e) | Item::Named(_, e) => e,
            })
            .collect(),
        _ => vec![],
    }
}

/// Replace the `n`th expression of `e` in preorder by its `k`th operand.
fn hoist(e: &mut Expr, n: &mut usize, k: usize) -> bool {
    if *n == 0 {
        let Some(inner) = operands(e).into_iter().nth(k).map(|o| o.clone()) else {
            return false;
        };
        *e = inner;
        return true;
    }
    *n -= 1;
    if let Expr::Func(f) = e {
        return hoist_block(&mut f.body, n, k);
    }
    operands(e).into_iter().any(|o| hoist(o, n, k))
}

fn hoist_block(b: &mut Block, n: &mut usize, k: usize) -> bool {
    for s in b {
        for e in exprs_of(s) {
            if hoist(e, n, k) {
                return true;
            }
        }
        for c in children_mut(s) {
            if hoist_block(c, n, k) {
                return true;
            }
        }
    }
    false
}

fn count_exprs(b: &mut Block) -> usize {
    fn of(e: &mut Expr) -> usize {
        if let Expr::Func(f) = e {
            return 1 + count_exprs(&mut f.body);
        }
        1 + operands(e).into_iter().map(of).sum::<usize>()
    }
    let mut n = 0;
    for s in b {
        n += exprs_of(s).into_iter().map(of).sum::<usize>();
        n += children_mut(s).into_iter().map(count_exprs).sum::<usize>();
    }
    n
}

pub fn program(mut p: Program, fails: &dyn Fn(&str) -> bool, log: &dyn Fn(&str)) -> Program {
    // Units: often one alone shows it.
    let mut i = p.units.len();
    while i > 0 {
        i -= 1;
        if p.units.len() > 1 {
            let u = p.units.remove(i);
            if !fails(&p.text()) {
                p.units.insert(i, u);
            }
        }
    }
    log(&format!("{} units", p.units.len()));
    loop {
        let mut shrunk = false;
        for u in 0..p.units.len() {
            for unwrap in [false, true] {
                let mut n = count(&p.units[u].code);
                while n > 0 {
                    n -= 1;
                    let saved = p.units[u].code.clone();
                    if edit(&mut p.units[u].code, &mut { n }, unwrap) && fails(&p.text()) {
                        shrunk = true;
                    } else {
                        p.units[u].code = saved;
                    }
                }
            }
            let mut n = count_exprs(&mut p.units[u].code);
            while n > 0 {
                n -= 1;
                for k in 0..3 {
                    let saved = p.units[u].code.clone();
                    if hoist_block(&mut p.units[u].code, &mut { n }, k) && fails(&p.text()) {
                        shrunk = true;
                        break;
                    }
                    p.units[u].code = saved;
                }
            }
            log(&format!(
                "{} statements in {}",
                count(&p.units[u].code),
                p.units[u].tag
            ));
        }
        if !shrunk {
            return p;
        }
    }
}

/// Remove runs of lines, halving the run length, while `fails` holds.
pub fn lines(text: &str, fails: &dyn Fn(&str) -> bool, log: &dyn Fn(&str)) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    let join = |l: &[&str]| l.join("\n") + "\n";
    let mut n = lines.len() / 2;
    while n >= 1 {
        let (mut i, mut shrunk) = (0, false);
        while i < lines.len() {
            let mut rest = lines[..i].to_vec();
            rest.extend_from_slice(&lines[(i + n).min(lines.len())..]);
            if fails(&join(&rest)) {
                lines = rest;
                shrunk = true;
            } else {
                i += n;
            }
        }
        if !shrunk {
            n /= 2;
        }
        log(&format!("{} lines, runs of {n}", lines.len()));
    }
    join(&lines)
}
