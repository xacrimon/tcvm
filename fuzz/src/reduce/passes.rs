//! The passes on text and tokens: shrinkray's `bytes.py`, `genericlanguages.py` and
//! `sequences.py`, with the Lua lexer's tokens where shrinkray uses its generic tokenizer or
//! single bytes, and Lua comments where it cuts C and Python ones. Cuts keep line breaks
//! (`SOFT`) except where whole lines go.

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::patch::{Integer, Kind, Once, Patch, apply_patches, cut, region_pass};
use super::problem::{Problem, less};
use crate::rng::Rng;

const SOFT: Kind = Kind::Cuts {
    keep_newlines: true,
};

fn cuts(spans: impl IntoIterator<Item = (usize, usize)>) -> Vec<Patch> {
    (spans.into_iter())
        .filter(|(s, e)| s < e)
        .map(|(s, e)| vec![cut(s, e)])
        .collect()
}

fn tokens(text: &str) -> Vec<std::ops::Range<usize>> {
    tcvm::parser::lex(text)
        .into_iter()
        .map(|(_, r)| r)
        .collect()
}

/// `--` comments, line and long, in the gaps between tokens.
pub fn cut_comments(p: &mut Problem, rng: &mut Rng) {
    let text = p.current().to_string();
    let b = text.as_bytes();
    let mut spans = Vec::new();
    let mut gap_start = 0;
    let mut bounds: Vec<(usize, usize)> = tokens(&text).iter().map(|r| (r.start, r.end)).collect();
    bounds.push((text.len(), text.len()));
    for (s, e) in bounds {
        let mut i = gap_start;
        while i + 1 < s {
            if b[i] == b'-' && b[i + 1] == b'-' {
                let rest = &text[i + 2..s];
                let level = rest
                    .strip_prefix('[')
                    .map(|r| r.len() - r.trim_start_matches('=').len())
                    .filter(|&l| rest[1 + l..].starts_with('['));
                let end = match level {
                    Some(l) => {
                        let close = format!("]{}]", "=".repeat(l));
                        rest.find(&close).map_or(s, |j| i + 2 + j + close.len())
                    }
                    None => rest.find(['\n', '\r']).map_or(s, |j| i + 2 + j),
                };
                spans.push((i, end.min(s)));
                i = end.max(i + 2);
            } else {
                i += 1;
            }
        }
        gap_start = e;
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

fn brace_intervals(b: &[u8], open: u8, close: u8) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for (i, &c) in b.iter().enumerate() {
        if c == open {
            stack.push(i);
        } else if c == close
            && let Some(s) = stack.pop()
            && i > s + 1
        {
            out.push((s + 1, i));
        }
    }
    out
}

/// Empty the insides of quotes, `[]` and `{}`, the smallest first.
pub fn hollow(p: &mut Problem, rng: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let mut spans = Vec::new();
    let mut quotes = Vec::new();
    for q in *b"\"'" {
        let xs: Vec<usize> = (0..b.len()).filter(|&i| b[i] == q).collect();
        quotes.extend(
            xs.windows(2)
                .filter(|w| w[0] + 1 < w[1])
                .map(|w| (w[0] + 1, w[1])),
        );
    }
    for mut group in [
        quotes,
        brace_intervals(&b, b'[', b']'),
        brace_intervals(&b, b'{', b'}'),
    ] {
        group.sort_by_key(|&(s, e)| (e - s, s));
        spans.extend(group);
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// Remove a bracket pair and keep what it held.
pub fn debracket(p: &mut Problem, rng: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let patches = [(b'{', b'}'), (b'(', b')'), (b'[', b']')]
        .into_iter()
        .flat_map(|(o, c)| brace_intervals(&b, o, c))
        .map(|(u, v)| vec![cut(u - 1, u), cut(v, v + 1)])
        .collect();
    apply_patches(p, rng, SOFT, patches);
}

/// `{A{B}C}` to `{B}`: delete a brace group's contents around one of its children.
pub fn lift_braces(p: &mut Problem, rng: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let mut starts = Vec::new();
    let mut children: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut patches = Vec::new();
    for (i, &c) in b.iter().enumerate() {
        if c == b'{' {
            starts.push(i);
            children.push(Vec::new());
        } else if c == b'}'
            && let Some(s) = starts.pop()
        {
            let (start, end) = (s + 1, i);
            let kids = children.pop().expect("a child list per open brace");
            if let Some(parent) = children.last_mut() {
                parent.push((start, end));
            }
            if end > start {
                for (cs, ce) in kids {
                    patches.push(vec![cut(start, cs), cut(ce, end)]);
                }
            }
        }
    }
    // A cut of zero width is no cut.
    for q in &mut patches {
        q.retain(|(s, e, _)| s < e);
    }
    patches.retain(|q| !q.is_empty());
    apply_patches(p, rng, SOFT, patches);
}

/// Delete runs of `min..=max` elements of the text split at `sep`, at every offset.
pub fn block_deletion(p: &mut Problem, rng: &mut Rng, sep: &str, min: usize, max: usize) {
    let n = p.current().split(sep).count();
    if n <= min {
        return;
    }
    let mut patches = Vec::new();
    for k in min..=max {
        for offset in 0..k {
            let mut i = offset;
            while i + k <= n {
                patches.push(vec![cut(i, i + k)]);
                i += k;
            }
        }
    }
    apply_patches(p, rng, Kind::Split(sep), patches);
}

/// Delete every copy of a repeated line at once.
pub fn delete_duplicate_lines(p: &mut Problem, rng: &mut Rng) {
    let mut index: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let text = p.current().to_string();
    for (i, line) in text.split('\n').enumerate() {
        index.entry(line).or_default().push(i);
    }
    let patches = (index.into_values())
        .filter(|ix| ix.len() > 1)
        .map(|ix| ix.into_iter().map(|i| cut(i, i + 1)).collect())
        .collect();
    apply_patches(p, rng, Kind::Split("\n"), patches);
}

/// Delete runs of `min..=max` tokens.
pub fn token_block_deletion(p: &mut Problem, rng: &mut Rng, min: usize, max: usize) {
    let toks = tokens(p.current());
    let n = toks.len();
    if n <= min {
        return;
    }
    let mut spans = Vec::new();
    for k in min..=max {
        for offset in 0..k {
            let mut i = offset;
            while i + k <= n {
                spans.push((toks[i].start, toks[i + k - 1].end));
                i += k;
            }
        }
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// For each token text that occurs more than once: delete up to the first occurrence, from
/// each occurrence to the next, and from the last to the end (shrinkray's `delete_byte_spans`
/// on tokens instead of bytes).
pub fn token_spans(p: &mut Problem, rng: &mut Rng) {
    let text = p.current().to_string();
    let toks = tokens(&text);
    let mut index: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, r) in toks.iter().enumerate() {
        index.entry(&text[r.clone()]).or_default().push(i);
    }
    let mut spans = Vec::new();
    for ix in index.values().filter(|ix| ix.len() > 1) {
        spans.push((0, toks[ix[0]].end));
        spans.extend(ix.windows(2).map(|w| (toks[w[0]].start, toks[w[1]].start)));
        spans.push((toks[ix[ix.len() - 1]].start, text.len()));
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// The spaces starting each line.
pub fn remove_indents(p: &mut Problem, rng: &mut Rng) {
    let b = p.current().as_bytes();
    let mut spans = Vec::new();
    for i in 0..b.len() {
        if b[i] == b'\n' {
            let j = (i + 1..b.len()).find(|&j| b[j] != b' ').unwrap_or(b.len());
            spans.push((i + 1, j));
        }
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// Each whitespace run, whole and less its first character.
pub fn remove_whitespace(p: &mut Problem, rng: &mut Rng) {
    let b = p.current().as_bytes();
    let mut spans = Vec::new();
    for i in 0..b.len() {
        if b[i].is_ascii_whitespace() {
            let j = (i + 1..b.len())
                .find(|&j| !b[j].is_ascii_whitespace())
                .unwrap_or(b.len());
            if j > i + 1 {
                spans.push((i, j));
            }
            if j > i + 2 {
                spans.push((i + 1, j));
            }
        }
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// Every deletion of 1 to 10 bytes.
pub fn short_deletions(p: &mut Problem, rng: &mut Rng) {
    let n = p.current().len();
    let spans = (0..n).flat_map(|i| (i + 1..(i + 11).min(n + 1)).map(move |j| (i, j)));
    apply_patches(p, rng, SOFT, cuts(spans));
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Maximal runs of word characters.
fn word_runs(b: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if is_word(b[i]) {
            let j = (i..b.len()).find(|&j| !is_word(b[j])).unwrap_or(b.len());
            out.push((i, j));
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Maximal runs of digits.
fn digit_runs(b: &[u8], from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = from;
    while i < to {
        if b[i].is_ascii_digit() {
            let j = (i..to).find(|&j| !b[j].is_ascii_digit()).unwrap_or(to);
            out.push((i, j));
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Each integer literal down to the smallest interesting value (`[0-9]+`).
pub fn reduce_integer_literals(p: &mut Problem, _: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    region_pass(p, digit_runs(&b, 0, b.len()), Integer::new);
}

/// `''`, `""`, `false`, `()` and `[]` (case-insensitive) to `0`.
pub fn replace_falsey_with_zero(p: &mut Problem, _: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let mut regions = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let n = if b[i..].starts_with(b"''")
            || b[i..].starts_with(b"\"\"")
            || b[i..].starts_with(b"()")
            || b[i..].starts_with(b"[]")
        {
            2
        } else if b.len() - i >= 5 && b[i..i + 5].eq_ignore_ascii_case(b"false") {
            5
        } else {
            0
        };
        if n > 0 {
            regions.push((i, i + n));
            i += n;
        } else {
            i += 1;
        }
    }
    region_pass(p, regions, |_| Some(Once::new("0")));
}

/// Each identifier, keywords included, to `0`.
pub fn replace_identifiers_with_zero(p: &mut Problem, _: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let regions = (word_runs(&b).into_iter())
        .filter(|&(s, _)| b[s].is_ascii_alphabetic() || b[s] == b'_')
        .collect();
    region_pass(p, regions, |_| Some(Once::new("0")));
}

/// `a op b` of integers to its value (`[0-9]+ [*+/-] [0-9]+`, exact division only).
pub fn combine_expressions(p: &mut Problem, _: &mut Rng) {
    let text = p.current().to_string();
    let b = text.as_bytes();
    let runs = digit_runs(b, 0, b.len());
    let mut regions = Vec::new();
    let mut last_end = 0;
    for (k, &(s, e)) in runs.iter().enumerate() {
        if s < last_end {
            continue;
        }
        if b.len() >= e + 3
            && b[e] == b' '
            && b"*+/-".contains(&b[e + 1])
            && b[e + 2] == b' '
            && let Some(&(s2, e2)) = runs.get(k + 1)
            && s2 == e + 3
        {
            regions.push((s, e2));
            last_end = e2;
        }
    }
    region_pass(p, regions, |r| {
        let mut it = r.split(' ');
        let (l, op, rhs) = (it.next()?, it.next()?, it.next()?);
        let (l, rhs): (i128, i128) = (l.parse().ok()?, rhs.parse().ok()?);
        let v = match op {
            "*" => l.checked_mul(rhs)?,
            "+" => l + rhs,
            "-" => l - rhs,
            _ if rhs == 0 || l % rhs != 0 => return None,
            _ => l / rhs,
        };
        Some(Once::new(&v.to_string()))
    });
}

/// A quote, whitespace and the same quote, to nothing.
pub fn merge_adjacent_strings(p: &mut Problem, _: &mut Rng) {
    let b = p.current().as_bytes().to_vec();
    let mut regions = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' || b[i] == b'\'' {
            let j = (i + 1..b.len())
                .find(|&j| !b[j].is_ascii_whitespace())
                .unwrap_or(b.len());
            if j < b.len() && b[j] == b[i] {
                regions.push((i, j + 1));
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    region_pass(p, regions, |_| Some(Once::new("")));
}

/// Repeated byte sequences and their start offsets: shrinkray's `find_ngram_endpoints`.
fn ngram_endpoints(b: &[u8]) -> Vec<(usize, Vec<usize>)> {
    if b.iter().all(|&c| c == b[0]) {
        return Vec::new();
    }
    let mut queue: VecDeque<(usize, Vec<usize>)> = VecDeque::from([(0, (0..b.len()).collect())]);
    let mut results = Vec::new();
    while let Some((mut k, mut indices)) = queue.pop_front() {
        if results.len() >= 10000 {
            break;
        }
        if k > 1 {
            let mut normalized: Vec<usize> = Vec::new();
            for i in indices {
                if normalized.last().is_none_or(|&l| i >= l + k) {
                    normalized.push(i);
                }
            }
            indices = normalized;
        }
        while indices[indices.len() - 1] + k < b.len()
            && indices.iter().all(|&i| b[i + k] == b[indices[0] + k])
        {
            k += 1;
        }
        if k > 0 && (indices[0] == 0 || indices.iter().any(|&i| b[i - 1] != b[indices[0] - 1])) {
            results.push((k, indices.clone()));
        }
        let mut split: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for &i in &indices {
            if let Some(&c) = b.get(i + k) {
                split.entry(c).or_default().push(i);
            }
        }
        queue.extend(
            split
                .into_values()
                .filter(|v| v.len() > 1)
                .map(|v| (k + 1, v)),
        );
    }
    results
}

/// Delete between consecutive occurrences of a repeated sequence, at least 8 bytes, longer
/// sequences first.
pub fn lexeme_based_deletions(p: &mut Problem, rng: &mut Rng) {
    let mut by_k: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (k, ends) in ngram_endpoints(p.current().as_bytes()) {
        by_k.entry(k)
            .or_default()
            .extend(ends.windows(2).map(|w| (w[0], w[1])));
    }
    let mut spans = Vec::new();
    for (_, mut iv) in by_k.into_iter().rev() {
        iv.sort();
        iv.dedup();
        iv.sort_by_key(|&(s, e)| std::cmp::Reverse((e - s, s)));
        spans.extend(iv.into_iter().filter(|(s, e)| e - s >= 8));
    }
    apply_patches(p, rng, SOFT, cuts(spans));
}

/// Shrinkray's ordering of single words: shorter, then digits before lowercase before
/// uppercase before the rest.
fn word_key(w: &str) -> (usize, Vec<u32>) {
    let class = |c: char| match c {
        '0'..='9' => c as u32 - '0' as u32,
        'a'..='z' => 10 + c as u32 - 'a' as u32,
        'A'..='Z' => 36 + c as u32 - 'A' as u32,
        _ => 62 + c as u32,
    };
    (w.len(), w.chars().map(class).collect())
}

/// The word runs of `b` that are names (a letter first) or numbers.
fn identifier_spans(b: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (s, e) in word_runs(b) {
        if b[s].is_ascii_alphabetic() {
            out.push((s, e));
        } else {
            out.extend(digit_runs(b, s, e));
        }
    }
    out
}

/// Rename each name to a smaller one: another name of the file or the first unused lowercase
/// and uppercase letter. Shrinkray tries one name at a time, the first interesting replacement
/// in order; here every rename is a patch, so they are tried in parallel and merged, and two
/// renames of one name conflict.
pub fn normalize_identifiers(p: &mut Problem, rng: &mut Rng) {
    let text = p.current().to_string();
    let b = text.as_bytes();
    let mut ids: Vec<&str> = (identifier_spans(b).into_iter())
        .map(|(s, e)| &text[s..e])
        .collect();
    ids.sort();
    ids.dedup();
    let mut replacements: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
    for range in ['a'..='z', 'A'..='Z'] {
        if let Some(c) = range.map(String::from).find(|c| !replacements.contains(c)) {
            replacements.push(c);
        }
    }
    let mut spans: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for (s, e) in word_runs(b) {
        spans.entry(&text[s..e]).or_default().push((s, e));
    }
    let mut patches = Vec::new();
    for t in ids {
        let Some(at) = spans.get(t) else { continue };
        for r in replacements.iter().filter(|r| word_key(r) < word_key(t)) {
            patches.push(at.iter().map(|&(s, e)| (s, e, r.clone())).collect());
        }
    }
    apply_patches(p, rng, Kind::Replace, patches);
}

/// Insertion sort of the lines, a swap kept when interesting and smaller.
pub fn line_sorter(p: &mut Problem, _: &mut Rng) {
    let mut lines: Vec<String> = p.current().split('\n').map(String::from).collect();
    for i in 1..lines.len() {
        let mut j = i;
        while j > 0 && !p.cancelled() {
            let mut attempt = lines.clone();
            attempt.swap(j - 1, j);
            let text = attempt.join("\n");
            if !less(&text, p.current()) || !p.is_interesting(text) {
                break;
            }
            j -= 1;
            lines = attempt;
        }
    }
}

/// `{}` to `[]` or `()`, and `[]` to `()`, everywhere at once.
pub fn simplify_brackets(p: &mut Problem, rng: &mut Rng) {
    let text = p.current().to_string();
    let mut patches = Vec::new();
    for (from, to) in [("{}", "[]"), ("{}", "()"), ("[]", "()")] {
        let map: HashMap<char, char> = from.chars().zip(to.chars()).collect();
        let replaced: Patch = (text.char_indices())
            .filter_map(|(i, c)| map.get(&c).map(|r| (i, i + 1, r.to_string())))
            .collect();
        if !replaced.is_empty() {
            patches.push(replaced);
        }
    }
    apply_patches(p, rng, Kind::Replace, patches);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngrams_find_repeats() {
        let r = ngram_endpoints(b"ab ab ab");
        assert!(r.iter().any(|(k, ix)| *k >= 2 && ix == &vec![0, 3, 6]));
    }

    #[test]
    fn identifiers_are_letter_first_word_runs_and_numbers() {
        let b = b"local x2 = _y1 + 3";
        let got: Vec<&str> = identifier_spans(b)
            .into_iter()
            .map(|(s, e)| std::str::from_utf8(&b[s..e]).unwrap())
            .collect();
        assert_eq!(got, ["local", "x2", "1", "3"]);
    }
}
