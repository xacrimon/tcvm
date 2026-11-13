//! Edits a pass proposes and how they are tried: shrinkray's `patching.py`. A pass hands
//! `apply_patches` its patches against the current test case; all of them at once are tried
//! first, then one by one, largest first, a window of them at a time in parallel. The interesting
//! ones of a window are merged in candidate order: the largest prefix that still combines into a
//! reduction is kept, the first one after it dropped, and the rest merged again.

use std::collections::VecDeque;

use super::problem::{Pending, Problem, less};
use crate::par;
use crate::rng::Rng;

/// How a pass's patches are applied. Every patch of a pass is of the same kind.
#[derive(Clone, Copy)]
pub enum Kind<'s> {
    /// Delete byte ranges; overlapping ones merge. With `keep_newlines` the line breaks inside a
    /// range stay, so a deletion never joins lines.
    Cuts { keep_newlines: bool },
    /// Replace byte ranges; overlapping ones conflict.
    Replace,
    /// Replace fixed regions, a later replacement of a region overriding an earlier one.
    Regions,
    /// Delete ranges of the elements `sep` splits the text into, separators with them.
    Split(&'s str),
}

/// Cuts that keep line breaks.
pub const SOFT: Kind = Kind::Cuts {
    keep_newlines: true,
};

/// Sorted `(start, end, replacement)`; for cuts the replacement is empty.
pub type Patch = Vec<(usize, usize, String)>;

pub fn cut(start: usize, end: usize) -> (usize, usize, String) {
    (start, end, String::new())
}

impl Kind<'_> {
    /// `None` on a conflict.
    fn combine(self, patches: &[&Patch]) -> Option<Patch> {
        if let Kind::Regions = self {
            let mut m = std::collections::BTreeMap::new();
            for (s, e, r) in patches.iter().flat_map(|p| p.iter()) {
                m.insert((*s, *e), r.clone());
            }
            return Some(m.into_iter().map(|((s, e), r)| (s, e, r)).collect());
        }
        let mut all: Patch = patches.iter().flat_map(|p| p.iter().cloned()).collect();
        all.sort();
        match self {
            Kind::Cuts { .. } | Kind::Split(_) => {
                let mut out: Patch = Vec::new();
                for (s, e, r) in all {
                    match out.last_mut() {
                        Some(last) if last.1 >= s => last.1 = last.1.max(e),
                        _ => out.push((s, e, r)),
                    }
                }
                Some(out)
            }
            // `Regions` returned above.
            Kind::Replace | Kind::Regions => {
                all.dedup();
                for w in all.windows(2) {
                    if w[1].0 < w[0].1 || w[0].0 == w[1].0 {
                        return None;
                    }
                }
                Some(all)
            }
        }
    }

    fn apply(self, patch: &Patch, target: &str) -> String {
        match self {
            Kind::Split(sep) => {
                let parts: Vec<&str> = target.split(sep).collect();
                let mut kept = Vec::with_capacity(parts.len());
                let mut cuts = patch.iter().peekable();
                for (i, part) in parts.iter().enumerate() {
                    while cuts.next_if(|c| c.1 <= i).is_some() {}
                    if !cuts.peek().is_some_and(|c| c.0 <= i) {
                        kept.push(*part);
                    }
                }
                kept.join(sep)
            }
            Kind::Cuts { keep_newlines } => {
                let mut out = String::with_capacity(target.len());
                let mut prev = 0;
                for (s, e, _) in patch {
                    out.push_str(&target[prev..*s]);
                    if keep_newlines {
                        out.extend(target[*s..*e].chars().filter(|&c| c == '\n'));
                    }
                    prev = *e;
                }
                out.push_str(&target[prev..]);
                out
            }
            Kind::Replace | Kind::Regions => {
                let mut out = String::with_capacity(target.len());
                let mut prev = 0;
                for (s, e, r) in patch {
                    out.push_str(&target[prev..*s]);
                    out.push_str(r);
                    prev = *e;
                }
                out.push_str(&target[prev..]);
                out
            }
        }
    }

    fn size(self, patch: &Patch) -> isize {
        (patch.iter())
            .map(|(s, e, r)| (e - s) as isize - r.len() as isize)
            .sum()
    }
}

/// Patches per window. Merges happen within a window, so its size is part of what a pass does
/// and does not depend on the number of workers.
const WINDOW: usize = 64;

/// The combination of a pass's patches accepted so far, against the test case the pass started
/// from: shrinkray's `PatchApplier`.
pub struct Applier<'s> {
    kind: Kind<'s>,
    initial: String,
    current: Patch,
}

impl<'s> Applier<'s> {
    pub fn new(kind: Kind<'s>, p: &Problem) -> Applier<'s> {
        Applier {
            kind,
            initial: p.current().to_string(),
            current: Vec::new(),
        }
    }

    /// What `patch` on top of what was accepted comes to, for each of `patches`, in parallel.
    fn candidates(&self, p: &Problem, patches: &[Patch]) -> Vec<Candidate> {
        let filter = p.filter();
        par::map(p.workers(), patches.len(), |i| {
            let Some(combined) = self.kind.combine(&[&self.current, &patches[i]]) else {
                return Candidate::Conflict;
            };
            if combined == self.current {
                return Candidate::Applied;
            }
            let text = self.kind.apply(&combined, &self.initial);
            match filter.check(&text) {
                Some(true) => Candidate::Applied,
                Some(false) => Candidate::Rejected,
                None => Candidate::Text(text),
            }
        })
    }

    /// Merge the patches whose candidates were interesting (`ok`); which of them were merged.
    fn merge(&mut self, p: &mut Problem, patches: &[Patch], ok: &[bool]) -> Vec<bool> {
        let mut queue: Vec<usize> = (0..patches.len()).filter(|&i| ok[i]).collect();
        let mut merged = vec![false; patches.len()];
        while !queue.is_empty() {
            // Shrinkray probes prefix lengths one test at a time; every prefix in one batch keeps
            // the workers busy when most of a window succeeds. The largest prefix that became the
            // current test case is kept, the patch after it dropped, and the rest merged again.
            let mut prefixes: Vec<(usize, Patch)> = Vec::new();
            let mut parts: Vec<&Patch> = vec![&self.current];
            for (k, &i) in queue.iter().enumerate() {
                parts.push(&patches[i]);
                if let Some(c) = self.kind.combine(&parts) {
                    prefixes.push((k + 1, c));
                }
            }
            let texts: Vec<String> = par::map(p.workers(), prefixes.len(), |j| {
                self.kind.apply(&prefixes[j].1, &self.initial)
            });
            let current = p.current();
            // Only texts no larger than the current test case can be reductions.
            let candidates: Vec<String> = (texts.iter())
                .filter(|t| *t == current || !less(current, t))
                .cloned()
                .collect();
            p.test(&candidates);
            let n = queue.len();
            let k = match (0..prefixes.len()).rev().find(|&j| texts[j] == p.current()) {
                Some(j) => {
                    self.current = prefixes[j].1.clone();
                    prefixes[j].0
                }
                None => 0,
            };
            for &i in &queue[..k] {
                merged[i] = true;
            }
            queue.drain(..(k + 1).min(n));
        }
        merged
    }

    /// The next window from `patches[at..]`: up to `size` patches with something to test, their
    /// candidates, the position after them, and how many were skipped as rejected. Candidates
    /// are built a chunk at a time, the chunk doubling while it yields too few.
    fn window(
        &self,
        p: &Problem,
        patches: &[Patch],
        mut at: usize,
        size: usize,
    ) -> (Vec<Patch>, Vec<String>, usize, usize) {
        let (mut window, mut texts, mut rejected) = (Vec::new(), Vec::new(), 0);
        let mut chunk = size;
        while at < patches.len() && window.len() < size {
            let end = (at + chunk).min(patches.len());
            for c in self.candidates(p, &patches[at..end]) {
                if window.len() == size {
                    break;
                }
                at += 1;
                match c {
                    Candidate::Text(t) => {
                        window.push(patches[at - 1].clone());
                        texts.push(t);
                    }
                    Candidate::Rejected => rejected += 1,
                    Candidate::Applied | Candidate::Conflict => {}
                }
            }
            chunk *= 2;
        }
        (window, texts, at, rejected)
    }

    /// Test `patches` in parallel and merge the interesting ones; which of them were applied.
    pub fn try_patches(&mut self, p: &mut Problem, patches: &[Patch]) -> Vec<bool> {
        let cands = self.candidates(p, patches);
        let mut window = Vec::new();
        let mut texts = Vec::new();
        let mut of = Vec::new();
        let mut rejected = 0;
        for (i, c) in cands.into_iter().enumerate() {
            match c {
                Candidate::Text(t) => {
                    window.push(patches[i].clone());
                    texts.push(t);
                    of.push(i);
                }
                Candidate::Rejected => rejected += 1,
                Candidate::Applied | Candidate::Conflict => {}
            }
        }
        p.note_rejected(rejected);
        let (ok, _) = p.test(&texts);
        let tested = self.merge(p, &window, &ok);
        let mut out = vec![false; patches.len()];
        for (j, &i) in of.iter().enumerate() {
            out[i] = tested[j];
        }
        // A patch the combination already holds, or whose result is the current test case,
        // counts as applied.
        for (i, patch) in patches.iter().enumerate() {
            if !of.contains(&i)
                && let Some(combined) = self.kind.combine(&[&self.current, patch])
            {
                out[i] = combined == self.current
                    || self.kind.apply(&combined, &self.initial) == p.current();
            }
        }
        out
    }
}

enum Candidate {
    Conflict,
    /// Changes nothing, or is known to be interesting.
    Applied,
    /// Known not to be interesting.
    Rejected,
    Text(String),
}

/// A window of a pass's patches, built against the state the pass was in, tested or being
/// tested.
struct Window {
    patches: Vec<Patch>,
    pending: Pending,
    /// The position after its last patch, and the candidates it skipped as rejected.
    end: usize,
    rejected: usize,
    /// What the applier had accepted when it was built.
    base: Patch,
    /// Its candidates' number and bytes.
    queued: (usize, usize),
}

/// Speculation stops at this many queued candidates, or this many bytes of them.
const MAX_AHEAD: usize = 50_000;
const MAX_AHEAD_BYTES: usize = 256 << 20;

/// Try `patches`, adopting any that reduce the test case. While one window is tested, the next
/// ones are built and queued on the assumption that it changes nothing; a window that does
/// change something cancels them. How far ahead it goes only decides which tests run sooner,
/// never what the pass does: it starts at one window and doubles with each window that changed
/// nothing.
pub fn apply_patches(p: &mut Problem, rng: &mut Rng, kind: Kind, patches: Vec<Patch>) {
    if patches.is_empty() {
        return;
    }
    let before = p.current().to_string();
    let mut patches = patches;
    if !matches!(kind, Kind::Split(_)) {
        // Byte-level passes may split a character.
        patches.retain(|q| {
            (q.iter()).all(|(s, e, _)| before.is_char_boundary(*s) && before.is_char_boundary(*e))
        });
    }
    let all: Vec<&Patch> = patches.iter().collect();
    if let Some(combined) = kind.combine(&all) {
        let text = kind.apply(&combined, &before);
        if text == before || (p.is_interesting(text) && p.current() != before) {
            return;
        }
    }
    rng.shuffle(&mut patches);
    patches.sort_by_key(|q| std::cmp::Reverse(kind.size(q)));
    let mut applier = Applier::new(kind, p);
    let size = WINDOW;
    let mut ahead = size;
    let mut windows: VecDeque<Window> = std::collections::VecDeque::new();
    let mut next = 0;
    let cancel_all = |p: &mut Problem, windows: &mut VecDeque<Window>| {
        for w in windows.drain(..) {
            p.cancel(w.pending);
        }
    };
    while !p.cancelled() {
        // Build and submit windows up to `ahead` candidates past the one tested next.
        let mut at = windows.back().map_or(next, |w| w.end);
        let (mut queued, mut bytes) =
            (windows.iter().skip(1)).fold((0, 0), |(n, b), w| (n + w.queued.0, b + w.queued.1));
        while at < patches.len()
            && (windows.is_empty() || (queued < ahead && bytes < MAX_AHEAD_BYTES))
        {
            let (window, texts, end, rejected) = applier.window(p, &patches, at, size);
            let these = (texts.len(), texts.iter().map(String::len).sum::<usize>());
            if !windows.is_empty() {
                queued += these.0;
                bytes += these.1;
            }
            let pending = p.submit(texts, !windows.is_empty());
            windows.push_back(Window {
                patches: window,
                pending,
                end,
                rejected,
                base: applier.current.clone(),
                queued: these,
            });
            at = end;
        }
        let Some(w) = windows.pop_front() else {
            break;
        };
        if w.base != applier.current {
            p.cancel(w.pending);
            cancel_all(p, &mut windows);
            continue;
        }
        p.note_rejected(w.rejected);
        let before = (p.version(), applier.current.clone());
        let Some((ok, _)) = p.finish(w.pending) else {
            cancel_all(p, &mut windows);
            continue;
        };
        next = w.end;
        applier.merge(p, &w.patches, &ok);
        if (p.version(), &applier.current) != (before.0, &before.1) {
            cancel_all(p, &mut windows);
            ahead = size;
        } else {
            ahead = (2 * ahead).min(MAX_AHEAD);
        }
    }
    cancel_all(p, &mut windows);
}

/// A search over one region's replacement: the next value to try given whether the last one
/// was applied, or `None` when done. Shrinkray runs one sub-reduction per region on a worker
/// pool; here every region's next value is tested in one batch per round.
pub trait Search {
    fn next(&mut self, last: Option<bool>) -> Option<String>;
}

/// Run `search(region text)` on every match of `find` in the current test case, all regions in
/// lockstep: shrinkray's `regex_pass`.
pub fn region_pass<S: Search>(
    p: &mut Problem,
    regions: Vec<(usize, usize)>,
    mut search: impl FnMut(&str) -> Option<S>,
) {
    if regions.is_empty() {
        return;
    }
    let mut applier = Applier::new(Kind::Regions, p);
    let initial = applier.initial.clone();
    let regions: Vec<(usize, usize)> = (regions.into_iter())
        .filter(|&(u, v)| initial.is_char_boundary(u) && initial.is_char_boundary(v))
        .collect();
    let mut active: Vec<(usize, S, Option<bool>)> = (regions.iter().enumerate())
        .filter_map(|(i, &(u, v))| search(&initial[u..v]).map(|s| (i, s, None)))
        .collect();
    while !active.is_empty() && !p.cancelled() {
        let mut round = Vec::new();
        active.retain_mut(|(i, s, last)| match s.next(last.take()) {
            Some(value) => {
                let (u, v) = regions[*i];
                round.push((*i, vec![(u, v, value)]));
                true
            }
            None => false,
        });
        if round.is_empty() {
            break;
        }
        let patches: Vec<Patch> = round.iter().map(|(_, q)| q.clone()).collect();
        let merged = applier.try_patches(p, &patches);
        for ((i, _), m) in round.iter().zip(merged) {
            if let Some(a) = active.iter_mut().find(|a| a.0 == *i) {
                a.2 = Some(m);
            }
        }
    }
}

/// One fixed replacement.
pub struct Once(Option<String>);

impl Once {
    pub fn new(value: &str) -> Once {
        Once(Some(value.to_string()))
    }
}

impl Search for Once {
    fn next(&mut self, _: Option<bool>) -> Option<String> {
        self.0.take()
    }
}

/// The smallest interesting integer: 0, then bisection, stepping both ends by one per round
/// (shrinkray's `reduce_integer`).
pub struct Integer {
    state: IntState,
    lo: u128,
    hi: u128,
}

enum IntState {
    Zero,
    Mid,
    HiMinus,
    LoPlus,
    Done,
}

impl Integer {
    pub fn new(text: &str) -> Option<Integer> {
        Some(Integer {
            state: IntState::Zero,
            lo: 0,
            hi: text.parse().ok()?,
        })
    }

    fn mid(&mut self) -> Option<String> {
        if self.lo + 1 < self.hi {
            self.state = IntState::Mid;
            Some(((self.lo + self.hi) / 2).to_string())
        } else {
            self.state = IntState::Done;
            None
        }
    }
}

impl Search for Integer {
    fn next(&mut self, last: Option<bool>) -> Option<String> {
        match self.state {
            IntState::Zero if last.is_none() => Some("0".into()),
            IntState::Zero => match last {
                Some(true) => None,
                _ => self.mid(),
            },
            IntState::Mid => {
                let mid = (self.lo + self.hi) / 2;
                if last == Some(true) {
                    self.hi = mid
                } else {
                    self.lo = mid
                }
                self.state = IntState::HiMinus;
                Some((self.hi - 1).to_string())
            }
            IntState::HiMinus => {
                if last == Some(true) {
                    self.hi -= 1;
                }
                self.state = IntState::LoPlus;
                Some((self.lo + 1).to_string())
            }
            IntState::LoPlus => {
                if last == Some(true) {
                    self.state = IntState::Done;
                    return None;
                }
                self.lo += 1;
                self.mid()
            }
            IntState::Done => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_cuts_drop_elements_and_their_separators() {
        let k = Kind::Split("\n");
        assert_eq!(k.apply(&vec![cut(1, 2)], "a\nb\nc"), "a\nc");
        assert_eq!(k.apply(&vec![cut(2, 3)], "a\nb\nc"), "a\nb");
        assert_eq!(k.apply(&vec![cut(0, 1), cut(2, 3)], "a\nb\nc"), "b");
    }

    #[test]
    fn soft_cuts_keep_line_breaks() {
        let k = Kind::Cuts {
            keep_newlines: true,
        };
        assert_eq!(k.apply(&vec![cut(1, 6)], "a b\nc d"), "a\nd");
    }

    #[test]
    fn integer_search_finds_the_smallest_value() {
        for target in [0u128, 1, 5, 17, 99, 100] {
            let mut s = Integer::new("100").unwrap();
            let mut last = None;
            let mut best = 100;
            while let Some(v) = s.next(last) {
                let v: u128 = v.parse().unwrap();
                last = Some(v >= target);
                if v >= target {
                    best = best.min(v);
                }
            }
            assert_eq!(best, target);
        }
    }
}
