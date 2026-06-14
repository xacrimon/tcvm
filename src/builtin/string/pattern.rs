//! Lua pattern-matching engine — a faithful port of the matcher in PUC-Lua
//! 5.5's `lstrlib.c` (`match`, `classend`, `singlematch`, `max_expand`, …).
//!
//! This module is deliberately free of any GC / `Context` dependency: it
//! operates purely on byte slices and reports matches as index ranges, so it
//! is exhaustively unit-testable on its own. The `string.{find,match,gmatch,
//! gsub}` callbacks in the parent module wrap it, turning captures into
//! `Value`s and (for `gsub`) driving native→Lua re-entry.
//!
//! Semantics mirror the reference exactly, including the C-locale `ctype`
//! classification (ASCII only — bytes ≥ 0x80 are unclassified) and the
//! `'^' $ * + ? . ( [ % -` set of magic characters. Pattern positions and
//! subject positions are byte indices; anchoring (`^`) is handled by the
//! caller (`find`/`match`/`gsub` strip it; `gmatch` does not, so there `^`
//! is a literal), matching `str_find_aux` / `gmatch`.

/// `LUA_MAXCAPTURES` — the capture array is fixed-size in the reference.
pub const MAX_CAPTURES: usize = 32;

/// `MAXCCALLS` — recursion-depth guard for `match` (raises "pattern too
/// complex" rather than overflowing the native stack).
const MAX_CCALLS: i32 = 200;

const L_ESC: u8 = b'%';
/// `SPECIALS` from `lstrlib.c`; a pattern with none of these is a plain
/// substring (`nospecials`).
const SPECIALS: &[u8] = b"^$*+?.([%-";

/// Length/kind of a capture, mirroring the `CAP_*` sentinels.
#[derive(Clone, Copy)]
enum CapLen {
    /// Capture opened (`(`) but not yet closed (`CAP_UNFINISHED`).
    Unfinished,
    /// Position capture `()` (`CAP_POSITION`).
    Position,
    /// Closed string capture of this byte length.
    Len(usize),
}

#[derive(Clone, Copy)]
struct Capture {
    /// Byte offset into the subject where this capture begins.
    init: usize,
    len: CapLen,
}

/// A resolved capture, ready to become a `Value`.
pub enum CapValue {
    /// A substring of the subject (`subject[start..end]`).
    Str { start: usize, end: usize },
    /// A position capture: a 1-based byte position.
    Pos(i64),
}

/// A malformed-pattern / capture error. Carries enough to reproduce the
/// reference's exact message (raised as a Lua error at the call boundary).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatError {
    EndsWithPercent,
    MissingBracket,
    BalanceArgs,
    MissingFrontierBracket,
    /// `invalid capture index %N` — `N` may be 0 (from `%0` as a back-reference).
    InvalidCaptureIndex(i32),
    InvalidPatternCapture,
    UnfinishedCapture,
    TooManyCaptures,
    PatternTooComplex,
}

impl PatError {
    pub fn message(self) -> String {
        match self {
            PatError::EndsWithPercent => "malformed pattern (ends with '%')".into(),
            PatError::MissingBracket => "malformed pattern (missing ']')".into(),
            PatError::BalanceArgs => "malformed pattern (missing arguments to '%b')".into(),
            PatError::MissingFrontierBracket => "missing '[' after '%f' in pattern".into(),
            PatError::InvalidCaptureIndex(n) => format!("invalid capture index %{n}"),
            PatError::InvalidPatternCapture => "invalid pattern capture".into(),
            PatError::UnfinishedCapture => "unfinished capture".into(),
            PatError::TooManyCaptures => "too many captures".into(),
            PatError::PatternTooComplex => "pattern too complex".into(),
        }
    }
}

type PatResult<T> = Result<T, PatError>;

/// One step of `match`'s manual tail-call loop (the `goto init` vs fall-through
/// distinction in the reference). `Init` re-enters the loop with new `(s, p)`;
/// `Done` returns the match result.
enum Step {
    Init(usize, usize),
    Done(Option<usize>),
}

// ---------------------------------------------------------------------------
// C-locale ctype helpers (ASCII only; bytes >= 0x80 classify as nothing).
// ---------------------------------------------------------------------------

/// `isspace`: space, `\t \n \v \f \r`. NB: includes `\v` (0x0B), which Rust's
/// `u8::is_ascii_whitespace` omits — so it must be spelled out here.
#[inline]
fn is_space(c: u8) -> bool {
    c == b' ' || (0x09..=0x0D).contains(&c)
}

/// `match_class`: does byte `c` match the single class letter `cl`?
/// A lowercase letter is the class; its uppercase form is the complement.
/// A non-letter `cl` matches itself literally (e.g. `%.` → '.').
fn match_class(c: u8, cl: u8) -> bool {
    let res = match cl.to_ascii_lowercase() {
        b'a' => c.is_ascii_alphabetic(),
        b'c' => c.is_ascii_control(),
        b'd' => c.is_ascii_digit(),
        b'g' => c.is_ascii_graphic(),
        b'l' => c.is_ascii_lowercase(),
        b'p' => c.is_ascii_punctuation(),
        b's' => is_space(c),
        b'u' => c.is_ascii_uppercase(),
        b'w' => c.is_ascii_alphanumeric(),
        b'x' => c.is_ascii_hexdigit(),
        b'z' => c == 0, // deprecated, but still honored by the reference
        _ => return cl == c,
    };
    if cl.is_ascii_lowercase() { res } else { !res }
}

/// `matchbracketclass`: does `c` match the set `pat[p_brk..=ec]` (`p_brk` is
/// the `[`, `ec` is the closing `]`)? Handles `^` negation, `%x` class escapes,
/// and `a-z` ranges, exactly as the reference's pointer walk.
fn match_bracket_class(c: u8, pat: &[u8], p_brk: usize, ec: usize) -> bool {
    let mut sig = true;
    let mut p = p_brk; // at '['
    if pat.get(p + 1) == Some(&b'^') {
        sig = false;
        p += 1; // skip the '^'
    }
    loop {
        p += 1; // C's `while (++p < ec)`
        if p >= ec {
            break;
        }
        if pat[p] == L_ESC {
            p += 1;
            if p < pat.len() && match_class(c, pat[p]) {
                return sig;
            }
        } else if pat.get(p + 1) == Some(&b'-') && p + 2 < ec {
            // A range `lo-hi`; the closing-adjacent `-` (p+2 == ec) is literal.
            let lo = pat[p];
            let hi = pat[p + 2];
            p += 2;
            if lo <= c && c <= hi {
                return sig;
            }
        } else if pat[p] == c {
            return sig;
        }
    }
    !sig
}

/// True iff the pattern contains no magic character (`nospecials`): the caller
/// can then use a plain substring search. NUL bytes in the pattern are fine —
/// unlike the reference's `strpbrk` scan we just look at every byte.
pub fn nospecials(pat: &[u8]) -> bool {
    !pat.iter().any(|b| SPECIALS.contains(b))
}

/// Plain substring search (`lmemfind` / `memchr` loop): byte offset of the
/// first occurrence of `needle` in `hay`, or `None`. Empty needle matches at 0.
pub fn plain_find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    let first = needle[0];
    let last_start = hay.len() - needle.len();
    let mut i = 0;
    while i <= last_start {
        if hay[i] == first && hay[i..i + needle.len()] == *needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// The pattern matcher state. Borrows the subject and pattern; the pattern
/// passed in has already had any leading `^` stripped by the caller (where
/// anchoring applies).
pub struct MatchState<'a> {
    src: &'a [u8],
    pat: &'a [u8],
    level: usize,
    matchdepth: i32,
    capture: [Capture; MAX_CAPTURES],
}

impl<'a> MatchState<'a> {
    pub fn new(src: &'a [u8], pat: &'a [u8]) -> Self {
        MatchState {
            src,
            pat,
            level: 0,
            matchdepth: MAX_CCALLS,
            capture: [Capture {
                init: 0,
                len: CapLen::Unfinished,
            }; MAX_CAPTURES],
        }
    }

    /// Attempt a match anchored at subject byte `s` (`reprepstate` + `match`).
    /// Returns the byte index one past the match end, or `None` if no match
    /// starts here.
    pub fn match_at(&mut self, s: usize) -> PatResult<Option<usize>> {
        self.level = 0;
        self.matchdepth = MAX_CCALLS;
        self.do_match(s, 0)
    }

    /// Number of captures `push_captures` would yield (`has_whole` is true when
    /// the whole match should stand in for the zero-capture case, as in
    /// `match`/`gmatch` but not `find`).
    pub fn num_captures(&self, has_whole: bool) -> usize {
        if self.level == 0 && has_whole {
            1
        } else {
            self.level
        }
    }

    /// Resolve the `i`-th capture against the whole-match range `[ws, we)`
    /// (`get_onecapture`). Index `0` with no captures yields the whole match.
    pub fn get_onecapture(&self, i: usize, ws: usize, we: usize) -> PatResult<CapValue> {
        if i >= self.level {
            if i != 0 {
                return Err(PatError::InvalidCaptureIndex(i as i32 + 1));
            }
            Ok(CapValue::Str { start: ws, end: we })
        } else {
            match self.capture[i].len {
                CapLen::Unfinished => Err(PatError::UnfinishedCapture),
                CapLen::Position => Ok(CapValue::Pos(self.capture[i].init as i64 + 1)),
                CapLen::Len(n) => Ok(CapValue::Str {
                    start: self.capture[i].init,
                    end: self.capture[i].init + n,
                }),
            }
        }
    }

    /// `classend`: index just past the pattern item starting at `p` (a single
    /// char, a `%x` escape, or a `[set]`).
    fn classend(&self, p: usize) -> PatResult<usize> {
        let c = self.pat[p];
        let mut p = p + 1;
        match c {
            L_ESC => {
                if p >= self.pat.len() {
                    return Err(PatError::EndsWithPercent);
                }
                Ok(p + 1)
            }
            b'[' => {
                if self.pat.get(p) == Some(&b'^') {
                    p += 1;
                }
                // do { ... } while (*p != ']')
                loop {
                    if p >= self.pat.len() {
                        return Err(PatError::MissingBracket);
                    }
                    let ch = self.pat[p];
                    p += 1;
                    if ch == L_ESC && p < self.pat.len() {
                        p += 1; // skip an escaped char (e.g. '%]')
                    }
                    if self.pat.get(p) == Some(&b']') {
                        break;
                    }
                }
                Ok(p + 1)
            }
            _ => Ok(p),
        }
    }

    /// `singlematch`: does `src[s]` match the single pattern item at `p`
    /// (whose end is `ep`)?
    fn single_match(&self, s: usize, p: usize, ep: usize) -> bool {
        if s >= self.src.len() {
            return false;
        }
        let c = self.src[s];
        match self.pat[p] {
            b'.' => true,
            L_ESC => match_class(c, self.pat[p + 1]),
            b'[' => match_bracket_class(c, self.pat, p, ep - 1),
            other => other == c,
        }
    }

    /// `matchbalance` (`%b`): match a balanced run delimited by `pat[p]` /
    /// `pat[p+1]` starting at `src[s]`.
    fn match_balance(&self, s: usize, p: usize) -> PatResult<Option<usize>> {
        if p + 1 >= self.pat.len() {
            return Err(PatError::BalanceArgs);
        }
        let b = self.pat[p];
        let e = self.pat[p + 1];
        if s >= self.src.len() || self.src[s] != b {
            return Ok(None);
        }
        let mut cont = 1i32;
        let mut s = s + 1;
        while s < self.src.len() {
            if self.src[s] == e {
                cont -= 1;
                if cont == 0 {
                    return Ok(Some(s + 1));
                }
            } else if self.src[s] == b {
                cont += 1;
            }
            s += 1;
        }
        Ok(None) // string ends out of balance
    }

    /// `max_expand`: greedy `*`/`+` — match as many as possible, then back off.
    fn max_expand(&mut self, s: usize, p: usize, ep: usize) -> PatResult<Option<usize>> {
        let mut i = 0;
        while self.single_match(s + i, p, ep) {
            i += 1;
        }
        loop {
            if let Some(res) = self.do_match(s + i, ep + 1)? {
                return Ok(Some(res));
            }
            if i == 0 {
                return Ok(None);
            }
            i -= 1;
        }
    }

    /// `min_expand`: lazy `-` — match as few as possible, growing on failure.
    fn min_expand(&mut self, mut s: usize, p: usize, ep: usize) -> PatResult<Option<usize>> {
        loop {
            if let Some(res) = self.do_match(s, ep + 1)? {
                return Ok(Some(res));
            }
            if self.single_match(s, p, ep) {
                s += 1;
            } else {
                return Ok(None);
            }
        }
    }

    fn capture_to_close(&self) -> PatResult<usize> {
        for level in (0..self.level).rev() {
            if matches!(self.capture[level].len, CapLen::Unfinished) {
                return Ok(level);
            }
        }
        Err(PatError::InvalidPatternCapture)
    }

    fn start_capture(&mut self, s: usize, p: usize, what: CapLen) -> PatResult<Option<usize>> {
        let level = self.level;
        if level >= MAX_CAPTURES {
            return Err(PatError::TooManyCaptures);
        }
        self.capture[level] = Capture { init: s, len: what };
        self.level = level + 1;
        let res = self.do_match(s, p)?;
        if res.is_none() {
            self.level -= 1; // undo capture
        }
        Ok(res)
    }

    fn end_capture(&mut self, s: usize, p: usize) -> PatResult<Option<usize>> {
        let l = self.capture_to_close()?;
        self.capture[l].len = CapLen::Len(s - self.capture[l].init);
        let res = self.do_match(s, p)?;
        if res.is_none() {
            self.capture[l].len = CapLen::Unfinished; // undo
        }
        Ok(res)
    }

    /// `check_capture`: resolve a back-reference digit to a closed capture index.
    fn check_capture(&self, digit: u8) -> PatResult<usize> {
        let l = digit as i32 - b'1' as i32;
        if l < 0
            || l as usize >= self.level
            || matches!(self.capture[l as usize].len, CapLen::Unfinished)
        {
            return Err(PatError::InvalidCaptureIndex(l + 1));
        }
        Ok(l as usize)
    }

    /// `match_capture` (`%1`–`%9`): match the literal text of capture `digit`.
    fn match_capture(&mut self, s: usize, digit: u8) -> PatResult<Option<usize>> {
        let l = self.check_capture(digit)?;
        let len = match self.capture[l].len {
            CapLen::Len(n) => n,
            // check_capture rejects unfinished/position captures via the index
            // check; a closed string capture is the only remaining kind.
            _ => return Ok(None),
        };
        let init = self.capture[l].init;
        if self.src.len() - s >= len && self.src[init..init + len] == self.src[s..s + len] {
            Ok(Some(s + len))
        } else {
            Ok(None)
        }
    }

    /// The `dflt:` block of `match`: a pattern item plus an optional suffix
    /// (`* + ? -`). Returns whether to loop (`Init`) or finish (`Done`).
    fn match_default(&mut self, s: usize, p: usize) -> PatResult<Step> {
        let ep = self.classend(p)?;
        if !self.single_match(s, p, ep) {
            match self.pat.get(ep) {
                // `*` `?` `-` accept zero repetitions: skip the item.
                Some(b'*') | Some(b'?') | Some(b'-') => Ok(Step::Init(s, ep + 1)),
                // `+` or no suffix: failure.
                _ => Ok(Step::Done(None)),
            }
        } else {
            match self.pat.get(ep) {
                Some(b'?') => {
                    if let Some(res) = self.do_match(s + 1, ep + 1)? {
                        Ok(Step::Done(Some(res)))
                    } else {
                        Ok(Step::Init(s, ep + 1))
                    }
                }
                Some(b'+') => Ok(Step::Done(self.max_expand(s + 1, p, ep)?)),
                Some(b'*') => Ok(Step::Done(self.max_expand(s, p, ep)?)),
                Some(b'-') => Ok(Step::Done(self.min_expand(s, p, ep)?)),
                _ => Ok(Step::Init(s + 1, ep)), // no suffix
            }
        }
    }

    /// `match`: the core recursive matcher. The `init:` tail-call loop is a
    /// `loop` with `Step::Init` re-entry; genuinely recursive cases (`?`,
    /// `*`/`+`/`-` expansion, captures) call `do_match` directly.
    fn do_match(&mut self, mut s: usize, mut p: usize) -> PatResult<Option<usize>> {
        if self.matchdepth == 0 {
            return Err(PatError::PatternTooComplex);
        }
        self.matchdepth -= 1;
        let res = loop {
            if p >= self.pat.len() {
                break Some(s); // end of pattern
            }
            let step = match self.pat[p] {
                b'(' => {
                    if self.pat.get(p + 1) == Some(&b')') {
                        Step::Done(self.start_capture(s, p + 2, CapLen::Position)?)
                    } else {
                        Step::Done(self.start_capture(s, p + 1, CapLen::Unfinished)?)
                    }
                }
                b')' => Step::Done(self.end_capture(s, p + 1)?),
                b'$' if p + 1 == self.pat.len() => Step::Done((s == self.src.len()).then_some(s)),
                L_ESC => match self.pat.get(p + 1).copied() {
                    Some(b'b') => match self.match_balance(s, p + 2)? {
                        Some(ns) => Step::Init(ns, p + 4),
                        None => Step::Done(None),
                    },
                    Some(b'f') => {
                        let fp = p + 2;
                        if self.pat.get(fp) != Some(&b'[') {
                            self.matchdepth += 1;
                            return Err(PatError::MissingFrontierBracket);
                        }
                        let ep = self.classend(fp)?;
                        let prev = if s == 0 { 0 } else { self.src[s - 1] };
                        let cur = if s < self.src.len() { self.src[s] } else { 0 };
                        if !match_bracket_class(prev, self.pat, fp, ep - 1)
                            && match_bracket_class(cur, self.pat, fp, ep - 1)
                        {
                            Step::Init(s, ep)
                        } else {
                            Step::Done(None)
                        }
                    }
                    Some(d @ b'0'..=b'9') => match self.match_capture(s, d)? {
                        Some(ns) => Step::Init(ns, p + 2),
                        None => Step::Done(None),
                    },
                    // `%` + non-class / EOF: a literal escape — fall to default.
                    _ => self.match_default(s, p)?,
                },
                _ => self.match_default(s, p)?,
            };
            match step {
                Step::Init(ns, np) => {
                    s = ns;
                    p = np;
                }
                Step::Done(r) => break r,
            }
        };
        self.matchdepth += 1;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a full anchored-from-each-position scan like `string.match` would,
    /// returning the (start, end) of the first match (byte indices).
    fn first_match(src: &[u8], pat: &[u8]) -> Option<(usize, usize)> {
        let anchor = pat.first() == Some(&b'^');
        let pat = if anchor { &pat[1..] } else { pat };
        let mut ms = MatchState::new(src, pat);
        let mut s = 0;
        loop {
            if let Some(e) = ms.match_at(s).unwrap() {
                return Some((s, e));
            }
            if anchor || s >= src.len() {
                return None;
            }
            s += 1;
        }
    }

    fn whole(src: &str, pat: &str) -> Option<String> {
        first_match(src.as_bytes(), pat.as_bytes())
            .map(|(s, e)| String::from_utf8_lossy(&src.as_bytes()[s..e]).into_owned())
    }

    #[test]
    fn literals_and_dot() {
        assert_eq!(first_match(b"hello", b"ell"), Some((1, 4)));
        assert_eq!(first_match(b"hello", b"xyz"), None);
        assert_eq!(first_match(b"abc", b"a.c"), Some((0, 3)));
    }

    #[test]
    fn anchors() {
        assert_eq!(first_match(b"abc", b"^a"), Some((0, 1)));
        assert_eq!(first_match(b"abc", b"^b"), None);
        assert_eq!(first_match(b"abc", b"c$"), Some((2, 3)));
        assert_eq!(first_match(b"abc", b"b$"), None);
        assert_eq!(first_match(b"", b"^$"), Some((0, 0)));
    }

    #[test]
    fn quantifiers() {
        assert_eq!(whole("aaa", "a*"), Some("aaa".into()));
        assert_eq!(whole("baaa", "a*"), Some("".into())); // greedy-but-empty at pos 0
        assert_eq!(whole("aaab", "a+"), Some("aaa".into()));
        assert_eq!(whole("aaab", "a-b"), Some("aaab".into())); // lazy then 'b'
        assert_eq!(whole("color", "colou?r"), Some("color".into()));
        assert_eq!(whole("colour", "colou?r"), Some("colour".into()));
    }

    #[test]
    fn classes() {
        assert_eq!(whole("  abc", "%a+"), Some("abc".into()));
        assert_eq!(whole("abc123", "%d+"), Some("123".into()));
        assert_eq!(whole("abc123", "%D+"), Some("abc".into()));
        assert_eq!(whole("a b\tc", "%s"), Some(" ".into()));
        // \v is whitespace in the C locale (and here), unlike Rust's default.
        assert_eq!(whole("x\x0by", "%s"), Some("\x0b".into()));
    }

    #[test]
    fn sets() {
        assert_eq!(whole("hello", "[el]+"), Some("ell".into()));
        assert_eq!(whole("hello", "[^el]+"), Some("h".into()));
        assert_eq!(whole("a-b]c", "[]%-]"), Some("-".into())); // ']' literal first, '-' escaped
        assert_eq!(whole("Z", "[A-Z]"), Some("Z".into()));
        assert_eq!(whole("5", "[0-9a-f]"), Some("5".into()));
    }

    #[test]
    fn captures_and_positions() {
        let src = b"hello world";
        let pat = b"(%w+)%s+(%w+)";
        let (s, e) = first_match(src, pat).unwrap();
        let mut ms = MatchState::new(src, pat);
        assert_eq!(ms.match_at(s).unwrap(), Some(e));
        match ms.get_onecapture(0, s, e).unwrap() {
            CapValue::Str { start, end } => assert_eq!(&src[start..end], b"hello"),
            _ => panic!(),
        }
        match ms.get_onecapture(1, s, e).unwrap() {
            CapValue::Str { start, end } => assert_eq!(&src[start..end], b"world"),
            _ => panic!(),
        }
        assert_eq!(ms.num_captures(true), 2);
    }

    #[test]
    fn position_capture() {
        let src = b"abc";
        let pat = b"()b()";
        let (s, e) = first_match(src, pat).unwrap();
        let mut ms = MatchState::new(src, pat);
        ms.match_at(s).unwrap();
        assert!(matches!(
            ms.get_onecapture(0, s, e).unwrap(),
            CapValue::Pos(2)
        ));
        assert!(matches!(
            ms.get_onecapture(1, s, e).unwrap(),
            CapValue::Pos(3)
        ));
    }

    #[test]
    fn balanced_and_backref() {
        assert_eq!(whole("(a(b)c)x", "%b()"), Some("(a(b)c)".into()));
        assert_eq!(whole("hello", "(l)%1"), Some("ll".into()));
        assert_eq!(whole("abab", "(ab)%1"), Some("abab".into()));
    }

    #[test]
    fn frontier() {
        // %f[%a] matches the empty string before the first alpha run.
        assert_eq!(first_match(b"  THE", b"%f[%a]"), Some((2, 2)));
        assert_eq!(first_match(b"123", b"%f[%a]"), None);
    }

    #[test]
    fn errors() {
        // Mirror how the real callbacks surface errors: an error may come from
        // `match_at` (malformed pattern) or from capture extraction afterward
        // (`unfinished capture`), exactly as `find`/`match` do.
        fn err(pat: &[u8]) -> PatError {
            let anchor = pat.first() == Some(&b'^');
            let p = if anchor { &pat[1..] } else { pat };
            let src = b"x";
            let mut ms = MatchState::new(src, p);
            let mut s = 0;
            loop {
                match ms.match_at(s) {
                    Err(e) => return e,
                    Ok(Some(e)) => {
                        for i in 0..ms.num_captures(true) {
                            if let Err(er) = ms.get_onecapture(i, s, e) {
                                return er;
                            }
                        }
                        unreachable!("expected error, matched cleanly");
                    }
                    Ok(None) => {}
                }
                if s >= src.len() {
                    unreachable!("expected error, no match");
                }
                s += 1;
            }
        }
        assert_eq!(err(b"("), PatError::UnfinishedCapture);
        assert_eq!(err(b"%"), PatError::EndsWithPercent);
        assert_eq!(err(b"["), PatError::MissingBracket);
        assert_eq!(err(b"%1"), PatError::InvalidCaptureIndex(1));
        assert_eq!(err(b"%f"), PatError::MissingFrontierBracket);
        assert_eq!(err(b"%b"), PatError::BalanceArgs);
    }

    #[test]
    fn nospecials_check() {
        assert!(nospecials(b"hello"));
        assert!(nospecials(b"a/b/c"));
        assert!(!nospecials(b"a.b"));
        assert!(!nospecials(b"a%d"));
        assert!(nospecials(b"")); // empty pattern is plain
    }

    #[test]
    fn plain_find_check() {
        assert_eq!(plain_find(b"a.b.c", b"."), Some(1));
        assert_eq!(plain_find(b"abc", b""), Some(0));
        assert_eq!(plain_find(b"abc", b"bc"), Some(1));
        assert_eq!(plain_find(b"abc", b"d"), None);
        assert_eq!(plain_find(b"abc", b"abcd"), None);
    }
}
