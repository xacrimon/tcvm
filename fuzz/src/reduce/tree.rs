//! The grammar passes: shrinkray's `treesitter.py` on tcvm's own syntax tree. Its nodes are
//! tree-sitter's named nodes, its tokens the anonymous ones. The parser drops whitespace and
//! comments, so a node's source span runs from its first token to its last, each tree token
//! mapped to the lexer's in order. The parser recovers from syntax errors, so the passes keep
//! working on the broken states other passes leave behind.

use std::collections::{BTreeMap, HashMap, HashSet};

use cstree::build::NodeCache;
use cstree::util::NodeOrToken;
use tcvm::parser::kind::SyntaxKind;
use tcvm::parser::syntax::SyntaxNode;

use super::patch::{Kind, Patch, apply_patches, cut};
use super::problem::Problem;
use crate::rng::Rng;

const SOFT: Kind = Kind::Cuts {
    keep_newlines: true,
};

pub struct Node {
    kind: SyntaxKind,
    start: usize,
    end: usize,
    children: Vec<Child>,
}

pub enum Child {
    Node(Node),
    Token { start: usize, end: usize },
}

impl Child {
    fn span(&self) -> (usize, usize) {
        match self {
            Child::Node(n) => (n.start, n.end),
            Child::Token { start, end } => (*start, *end),
        }
    }
}

/// The tree of `source`, `None` when the tree's tokens are not the lexer's.
pub fn parse(source: &str) -> Option<Node> {
    let tokens = tcvm::parser::lex(source);
    let parse = tcvm::parser::parse(&mut NodeCache::new(), source);
    let mut next = 0;
    let root = build(&parse.root, &tokens, &mut next, 0)?;
    (next == tokens.len()).then_some(root)
}

fn build(
    node: &SyntaxNode,
    tokens: &[(SyntaxKind, std::ops::Range<usize>)],
    next: &mut usize,
    at: usize,
) -> Option<Node> {
    let mut children = Vec::new();
    let mut pos = at;
    for c in node.children_with_tokens() {
        match c {
            NodeOrToken::Node(n) => {
                let n = build(n, tokens, next, pos)?;
                pos = n.end;
                children.push(Child::Node(n));
            }
            NodeOrToken::Token(t) if t.kind() == SyntaxKind::Eof => {}
            NodeOrToken::Token(t) => {
                let (kind, r) = tokens.get(*next)?;
                if *kind != t.kind() {
                    return None;
                }
                *next += 1;
                pos = r.end;
                children.push(Child::Token {
                    start: r.start,
                    end: r.end,
                });
            }
        }
    }
    let first = children.iter().map(Child::span).find(|(s, e)| s < e);
    let start = first.map_or(pos, |(s, _)| s);
    Some(Node {
        kind: node.kind(),
        start,
        end: pos.max(start),
        children,
    })
}

fn preorder(root: &Node) -> Vec<&Node> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        out.push(n);
        for c in n.children.iter().rev() {
            if let Child::Node(c) = c {
                stack.push(c);
            }
        }
    }
    out
}

/// Runs this long of adjacent children are deleted as one patch; longer ones come from merging.
const MAX_CHILD_RUN: usize = 3;

fn is_separator(source: &str, c: &Child) -> bool {
    matches!(c, Child::Token { start, end } if matches!(&source[*start..*end], "," | ";"))
}

/// Delete runs of child nodes, with the whitespace up to the next one or a separator beside
/// them.
fn child_deletion_cuts(root: &Node, source: &str) -> Vec<Patch> {
    let mut out = Vec::new();
    for node in preorder(root) {
        let named: Vec<usize> = (0..node.children.len())
            .filter(|&j| matches!(node.children[j], Child::Node(_)))
            .collect();
        for (i, &first) in named.iter().enumerate() {
            for run in 1..=MAX_CHILD_RUN {
                if i + run > named.len() {
                    break;
                }
                let last = named[i + run - 1];
                let (start, end) = (node.children[first].span().0, node.children[last].span().1);
                if start == end {
                    continue;
                }
                out.push(vec![cut(start, end)]);
                if let Some(&after) = named.get(i + run) {
                    out.push(vec![cut(start, node.children[after].span().0)]);
                }
                if let Some(after) = node.children.get(last + 1)
                    && is_separator(source, after)
                {
                    out.push(vec![cut(start, after.span().1)]);
                }
                if first > 0 && is_separator(source, &node.children[first - 1]) {
                    out.push(vec![cut(node.children[first - 1].span().0, end)]);
                }
            }
        }
    }
    out
}

/// Arbitrary descendants are promoted from this many levels down; same-type ones from any depth.
const MAX_PROMOTION_DEPTH: usize = 2;

/// Replace a node with one of its descendants: the nearest of its own kind along each path, or
/// any within two levels.
fn lift_cuts(root: &Node) -> Vec<Patch> {
    let mut out = Vec::new();
    for node in preorder(root) {
        let span = (node.start, node.end);
        let mut stack: Vec<(&Child, usize)> = node.children.iter().map(|c| (c, 1)).collect();
        while let Some((d, depth)) = stack.pop() {
            let Child::Node(d) = d else { continue };
            let d_span = (d.start, d.end);
            let same = d_span != span && d.kind == node.kind;
            if d_span != span && (same || depth <= MAX_PROMOTION_DEPTH) {
                let q: Patch = [cut(span.0, d_span.0), cut(d_span.1, span.1)]
                    .into_iter()
                    .filter(|c| c.0 < c.1)
                    .collect();
                if !q.is_empty() {
                    out.push(q);
                }
            }
            // Past a same-kind descendant only by applying the pass again.
            if !same {
                stack.extend(d.children.iter().map(|c| (c, depth + 1)));
            }
        }
    }
    out
}

/// Replace each node by the smallest text of a node of its kind elsewhere in the file.
fn minimal_substitutions(root: &Node, source: &str) -> Vec<Patch> {
    let nodes = preorder(root);
    let mut minimal: HashMap<SyntaxKind, &str> = HashMap::new();
    for n in &nodes {
        let text = &source[n.start..n.end];
        let best = minimal.entry(n.kind).or_insert(text);
        if (text.len(), text) < (best.len(), *best) {
            *best = text;
        }
    }
    (nodes.iter())
        .filter(|n| minimal[&n.kind].len() < n.end - n.start)
        .map(|n| vec![(n.start, n.end, minimal[&n.kind].to_string())])
        .collect()
}

fn word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The names a node mentions: identifier tokens, and the last path segment of string literals.
fn names_mentioned(node: &Node, source: &str) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n.kind {
            SyntaxKind::LiteralExpr if source[n.start..n.end].starts_with(['"', '\'']) => {
                let text = source[n.start..n.end].trim_matches(['"', '\'']);
                if let Some(base) = text.rsplit('/').next().filter(|b| !b.is_empty()) {
                    names.insert(base.to_string());
                    let w: String = base
                        .bytes()
                        .take_while(|&c| word(c))
                        .map(char::from)
                        .collect();
                    if !w.is_empty() {
                        names.insert(w);
                    }
                }
            }
            SyntaxKind::Ident if n.end > n.start => {
                names.insert(source[n.start..n.end].to_string());
            }
            _ => {
                for c in &n.children {
                    if let Child::Node(c) = c {
                        stack.push(c);
                    }
                }
            }
        }
    }
    names
}

/// The names a top-level statement binds: a function's name, a local's targets; otherwise every
/// name it mentions (conservative: fewer orphans).
fn names_defined(node: &Node, source: &str) -> HashSet<String> {
    let first_ident = |n: &Node| {
        n.children.iter().find_map(|c| match c {
            Child::Node(c) if c.kind == SyntaxKind::Ident && c.end > c.start => {
                Some(source[c.start..c.end].to_string())
            }
            _ => None,
        })
    };
    let func = |n: &Node| {
        (n.children.iter()).find_map(|c| match c {
            Child::Node(c) if c.kind == SyntaxKind::FuncStmt => first_ident(c),
            _ => None,
        })
    };
    let targets = |n: &Node| -> HashSet<String> {
        let mut out = HashSet::new();
        for c in &n.children {
            if let Child::Node(list) = c
                && list.kind == SyntaxKind::AssignList
            {
                for t in &list.children {
                    if let Child::Node(t) = t
                        && matches!(t.kind, SyntaxKind::DeclTarget | SyntaxKind::GlobalTarget)
                        && let Some(name) = first_ident(t)
                    {
                        out.insert(name);
                    }
                }
            }
        }
        out
    };
    let found: HashSet<String> = match node.kind {
        SyntaxKind::FuncStmt => first_ident(node).into_iter().collect(),
        SyntaxKind::DeclStmt | SyntaxKind::GlobalStmt => {
            func(node).map_or_else(|| targets(node), |f| HashSet::from([f]))
        }
        _ => HashSet::new(),
    };
    if found.is_empty() {
        names_mentioned(node, source)
    } else {
        found
    }
}

/// The spans of the nodes covering `[lo, hi)`, from the root down.
fn covering_chain(root: &Node, lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut node = root;
    loop {
        if node.start <= lo && hi <= node.end && spans.last() != Some(&(node.start, node.end)) {
            spans.push((node.start, node.end));
        }
        let next = node.children.iter().find_map(|c| match c {
            Child::Node(c) if c.start <= lo && hi <= c.end => Some(c),
            _ => None,
        });
        match next {
            Some(c) => node = c,
            None => return spans,
        }
    }
}

/// Delete a node with the top-level statements it was the last user of, cascading: a
/// statement goes once every occurrence of its names outside it lies in what is deleted
/// (shrinkray's `while changed` fixpoint, which its worklist reproduces).
fn orphaned_declaration_cuts(root: &Node, source: &str) -> Vec<Patch> {
    let gc: Vec<(&Node, HashSet<String>)> = (root.children.iter())
        .filter_map(|c| match c {
            Child::Node(n) if n.end > n.start => Some((n, names_defined(n, source))),
            _ => None,
        })
        .collect();
    let b = source.as_bytes();
    let mut occurrences: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut i = 0;
    while i < b.len() {
        if word(b[i]) {
            let j = (i..b.len()).find(|&j| !word(b[j])).unwrap_or(b.len());
            occurrences.entry(&source[i..j]).or_default().push(i);
            i = j;
        } else {
            i += 1;
        }
    }
    let spans: Vec<(usize, usize)> = gc.iter().map(|(n, _)| (n.start, n.end)).collect();
    // Each candidate's occurrences outside itself.
    let external: Vec<Vec<usize>> = (gc.iter().zip(&spans))
        .map(|((_, names), &(lo, hi))| {
            let mut v: Vec<usize> = (names.iter())
                .flat_map(|n| occurrences.get(n.as_str()).into_iter().flatten().copied())
                .filter(|&p| !(lo <= p && p < hi))
                .collect();
            v.sort();
            v.dedup();
            v
        })
        .collect();
    let mut triggers: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, ext) in external.iter().enumerate() {
        // Unreferenced to begin with: a plain cut deletes it.
        if let (Some(&lo), Some(&hi)) = (ext.first(), ext.last()) {
            for span in covering_chain(root, lo, hi + 1) {
                triggers.entry(span).or_default().push(i);
            }
        }
    }
    let mut out = Vec::new();
    for (x, indices) in triggers {
        let overlaps = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
        let seeds: Vec<usize> = indices
            .into_iter()
            .filter(|&i| !overlaps(spans[i], x))
            .collect();
        if seeds.is_empty() {
            continue;
        }
        let mut deleted: Vec<(usize, usize)> = vec![x];
        deleted.extend(seeds.iter().map(|&i| spans[i]));
        let mut collected: HashSet<usize> = seeds.into_iter().collect();
        loop {
            let mut changed = false;
            for i in 0..gc.len() {
                if collected.contains(&i) || external[i].is_empty() {
                    continue;
                }
                let covered = |p: usize| deleted.iter().any(|&(s, e)| s <= p && p < e);
                if external[i].iter().all(|&p| covered(p))
                    && !deleted.iter().any(|&d| overlaps(d, spans[i]))
                {
                    collected.insert(i);
                    deleted.push(spans[i]);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        deleted.sort();
        deleted.dedup();
        out.push(deleted.into_iter().map(|(s, e)| cut(s, e)).collect());
    }
    out
}

fn cut_pass(p: &mut Problem, rng: &mut Rng, cuts: fn(&Node, &str) -> Vec<Patch>) {
    let source = p.current().to_string();
    if let Some(root) = parse(&source) {
        apply_patches(p, rng, SOFT, cuts(&root, &source));
    }
}

pub fn delete_children(p: &mut Problem, rng: &mut Rng) {
    cut_pass(p, rng, child_deletion_cuts);
}

pub fn delete_orphaned_declarations(p: &mut Problem, rng: &mut Rng) {
    cut_pass(p, rng, orphaned_declaration_cuts);
}

pub fn lift_nodes(p: &mut Problem, rng: &mut Rng) {
    cut_pass(p, rng, |root, _| lift_cuts(root));
}

pub fn substitute_nodes(p: &mut Problem, rng: &mut Rng) {
    let source = p.current().to_string();
    if let Some(root) = parse(&source) {
        let patches = minimal_substitutions(&root, &source);
        apply_patches(p, rng, Kind::Replace, patches);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_are_source_spans() {
        let src = "local x = 1 -- c\nprint( x )\n";
        let root = parse(src).expect("a tree");
        let stmts: Vec<&str> = (root.children.iter())
            .filter_map(|c| match c {
                Child::Node(n) => Some(&src[n.start..n.end]),
                _ => None,
            })
            .collect();
        assert_eq!(stmts, ["local x = 1", "print( x )"]);
    }

    #[test]
    fn broken_programs_still_parse() {
        assert!(parse("local x = = 1\nend end\n").is_some());
    }

    #[test]
    fn orphans_go_with_their_last_user() {
        let src = "local function f() end\nlocal function g() f() end\ng()\n";
        let root = parse(src).expect("a tree");
        let cuts = orphaned_declaration_cuts(&root, src);
        // Deleting the call to g orphans g, and with it f.
        assert!(cuts.iter().any(|q| q.len() == 3), "{cuts:?}");
    }
}
