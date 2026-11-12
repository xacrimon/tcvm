//! What a program's runs are compared on.
//!
//! - Every JIT configuration of a binary against the same binary with the JIT off: the whole
//!   output and the exit status, exactly.
//! - A binary with the JIT off against reference Lua: the output of the strict units, after
//!   the accepted differences below are taken out of both.
//! - Every run: no crash, no hang the JIT-off run does not have, and no compile that failed on
//!   a verifier, checker or backend error (those fall back to the interpreter, so their output
//!   is right).

use crate::exec::Run;
use crate::rng::Rng;

/// Differences from reference Lua's error messages that tcvm accepts for now. Each is taken
/// out of both outputs unless it is enforced; drop a rule here once tcvm matches.
#[derive(Clone, Copy)]
pub struct Accepted {
    /// tcvm names no variable: ` (local 'x')`, ` (field 'y')`, … are removed.
    pub variable: bool,
    /// tcvm says `got nil` for a missing argument, Lua `got no value`.
    pub no_value: bool,
    /// tcvm names a library function without its library: `'string.rep'` reads `'rep'`.
    pub qualified: bool,
    /// Lua names the function of a `bad argument` by how the call site reached it (`'a'`,
    /// `'integer index'`), tcvm by its own name: the name is removed.
    pub callee: bool,
    /// Lua numbers a method call's arguments without the receiver, tcvm with it: the number
    /// of a `bad argument` is removed.
    pub arg_number: bool,
}

pub const RULES: &[&str] = &["variable", "no-value", "qualified", "callee", "arg-number"];

impl Accepted {
    pub fn all_but(enforced: &[String]) -> Accepted {
        let on = |name: &str| !enforced.iter().any(|e| e == name);
        Accepted {
            variable: on("variable"),
            no_value: on("no-value"),
            qualified: on("qualified"),
            callee: on("callee"),
            arg_number: on("arg-number"),
        }
    }

    pub fn normalize(&self, line: &str) -> String {
        let mut s = line.to_string();
        if self.variable {
            const WHAT: &[&str] = &[
                "local",
                "global",
                "field",
                "upvalue",
                "constant",
                "method",
                "for iterator",
                "metamethod",
                "function",
            ];
            for w in WHAT {
                let open = format!(" ({w} '");
                while let Some(i) = s.find(&open) {
                    let Some(j) = s[i + open.len()..].find("')") else {
                        break;
                    };
                    s.replace_range(i..i + open.len() + j + 2, "");
                }
            }
        }
        if self.no_value {
            s = s.replace("got no value", "got nil");
        }
        if self.qualified {
            let mut from = 0;
            while let Some(i) = s[from..].find("to '") {
                let at = from + i + 4;
                let word = s[at..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
                match word {
                    Some(n) if n > 0 && s[at + n..].starts_with('.') => {
                        s.replace_range(at..at + n + 1, "")
                    }
                    _ => {}
                }
                from = at;
            }
        }
        let mut from = 0;
        while let Some(i) = s[from..].find("bad argument #") {
            let at = from + i + "bad argument ".len();
            let digits = s[at + 1..].find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
            let mut end = at + 1 + digits;
            if self.arg_number {
                s.replace_range(at..end, "#");
                end = at + 1;
            }
            if self.callee
                && s[end..].starts_with(" to '")
                && let Some(q) = s[end + 5..].find('\'')
            {
                s.replace_range(end + 5..end + 5 + q, "");
            }
            from = end;
        }
        s
    }
}

pub struct Mode {
    pub name: String,
    pub env: Vec<(String, String)>,
    /// How many times the time limit it gets.
    pub slow: u32,
}

fn env(pairs: &[(&str, String)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

pub fn off() -> Mode {
    Mode {
        name: "off".into(),
        env: env(&[("TCVM_JIT", "off".into())]),
        slow: 1,
    }
}

/// The configurations a program runs under besides JIT-off: the default thresholds, everything
/// compiled at its first count, and a few drawn from the knobs.
pub fn modes(r: &mut Rng) -> Vec<Mode> {
    let one = "1".to_string();
    let mut out = vec![
        Mode {
            name: "default".into(),
            env: vec![],
            slow: 1,
        },
        Mode {
            name: "hot".into(),
            env: env(&[
                ("TCVM_JIT_HOT_CALL", one.clone()),
                ("TCVM_JIT_HOT_LOOP", one.clone()),
            ]),
            slow: 1,
        },
    ];
    for _ in 0..3 {
        let hc = *r.pick(&[1, 1, 2, 4, 16, 64]);
        let hl = *r.pick(&[1, 1, 2, 3, 5, 20, 57]);
        let mut name = format!("hc{hc},hl{hl}");
        let mut e = env(&[
            ("TCVM_JIT_HOT_CALL", hc.to_string()),
            ("TCVM_JIT_HOT_LOOP", hl.to_string()),
        ]);
        let mut slow = 1;
        let mut knob = |p: f64, tag: &str, k: &str, v: String, r: &mut Rng| {
            if r.chance(p) {
                name.push_str(&format!(",{tag}"));
                e.push((k.to_string(), v));
                true
            } else {
                false
            }
        };
        knob(0.3, "noinline", "TCVM_JIT_INLINE", "0".into(), r);
        knob(0.3, "deopt", "TCVM_JIT_DEOPT_ALL", one.clone(), r);
        knob(0.4, "check", "TCVM_JIT_CHECK", one.clone(), r);
        knob(0.1, "far", "TCVM_JIT_FAR", one.clone(), r);
        let n = *r.pick(&[1, 7, 64, 64, 1000, 1000]);
        if knob(0.2, &format!("gc{n}"), "TCVM_GC_STRESS", n.to_string(), r) {
            slow = 3;
        }
        out.push(Mode { name, env: e, slow });
    }
    if r.chance(0.3) {
        let n = *r.pick(&[1, 16, 16, 256, 256]);
        out.push(Mode {
            name: format!("off,gc{n}"),
            env: env(&[
                ("TCVM_JIT", "off".into()),
                ("TCVM_GC_STRESS", n.to_string()),
            ]),
            slow: 3,
        });
    }
    out
}

/// The mode `name` names (`Mode::name`): how a finding's configuration is run again.
pub fn mode_named(name: &str) -> Option<Mode> {
    let mut e = Vec::new();
    let mut slow = 1;
    for part in name.split(',') {
        let (k, v): (&str, String) = match part {
            "default" => continue,
            "hot" => {
                e.push(("TCVM_JIT_HOT_CALL".to_string(), "1".to_string()));
                ("TCVM_JIT_HOT_LOOP", "1".into())
            }
            "off" => ("TCVM_JIT", "off".into()),
            "noinline" => ("TCVM_JIT_INLINE", "0".into()),
            "deopt" => ("TCVM_JIT_DEOPT_ALL", "1".into()),
            "check" => ("TCVM_JIT_CHECK", "1".into()),
            "far" => ("TCVM_JIT_FAR", "1".into()),
            p if p.starts_with("hc") => ("TCVM_JIT_HOT_CALL", p[2..].into()),
            p if p.starts_with("hl") => ("TCVM_JIT_HOT_LOOP", p[2..].into()),
            p if p.starts_with("gc") => {
                slow = 3;
                ("TCVM_GC_STRESS", p[2..].into())
            }
            _ => return None,
        };
        e.push((k.to_string(), v));
    }
    Some(Mode {
        name: name.into(),
        env: e,
        slow,
    })
}

/// The unit an output line belongs to: lines start with `U<n>`.
pub fn unit_of(line: &str) -> Option<&str> {
    let rest = line.strip_prefix('U')?;
    let n = rest.find(|c: char| !c.is_ascii_digit())?;
    (n > 0).then(|| &line[..n + 1])
}

/// The first line two outputs differ on: (wanted, got).
pub fn first_diff(want: &str, got: &str) -> Option<(String, String)> {
    let (mut a, mut b) = (want.lines(), got.lines());
    loop {
        match (a.next(), b.next()) {
            (None, None) => return None,
            (x, y) if x == y => {}
            (x, y) => return Some((x.unwrap_or("<eof>").into(), y.unwrap_or("<eof>").into())),
        }
    }
}

/// The lines of the units in `strict`, normalized: what is compared with reference Lua.
pub fn strict_lines(out: &str, strict: &[String], acc: &Accepted) -> String {
    let mut s = String::new();
    for l in out.lines() {
        if unit_of(l).is_some_and(|u| strict.iter().any(|x| x == u)) {
            s.push_str(&merge_hist(&acc.normalize(l)));
            s.push('\n');
        }
    }
    s
}

/// Lines of a JIT log that report a failed compile the compiler is to blame for.
pub fn internal_errors(run: &Run) -> Vec<String> {
    (run.stderr.lines())
        .filter(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("failed: verifier")
                || l.contains("failed: backend")
                || l.contains("panicked")
                || l.contains("allocation check")
                || l.contains("checker")
        })
        .map(|l| l.chars().take(300).collect())
        .collect()
}

/// The results of a warm-up line (`U1w H key=count key=count …`) with their counts; keys hold
/// spaces.
fn entries(line: &str) -> Option<Vec<(&str, &str)>> {
    let rest = line.split_once(" H ")?.1;
    let b = rest.as_bytes();
    let (mut out, mut start, mut i) = (Vec::new(), 0, 0);
    while i < b.len() {
        if b[i] == b'=' {
            let digits = b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            let end = i + 1 + digits;
            if digits > 0 && (end == b.len() || b[end] == b' ') {
                out.push((&rest[start..i], &rest[i + 1..end]));
                start = (end + 1).min(b.len());
                i = end;
            }
        }
        i += 1;
    }
    Some(out)
}

fn keys(line: &str) -> Option<Vec<&str>> {
    Some(entries(line)?.into_iter().map(|(k, _)| k).collect())
}

/// A warm-up line with the counts of keys that normalizing made equal added up, in the order
/// `hist` sorts them.
fn merge_hist(line: &str) -> String {
    let Some(es) = entries(line) else {
        return line.to_string();
    };
    let mut merged: std::collections::BTreeMap<&str, u64> = Default::default();
    for (k, n) in es {
        *merged.entry(k).or_default() += n.parse::<u64>().unwrap_or(0);
    }
    let head = line.split_once(" H ").map_or(line, |x| x.0);
    let body: Vec<String> = merged.iter().map(|(k, n)| format!("{k}={n}")).collect();
    format!("{head} H {}", body.join(" "))
}

/// A result without its values: `#2,i,f|s` for two results, an integer and a float, and a
/// probed string. An error keeps its message, without the numbers.
fn shape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    let mut digits = false;
    while i < b.len() {
        let start = i == 0 || matches!(b[i - 1], b',' | b'|' | b' ' | b'<');
        let rest = &s[i..];
        let hex = rest.strip_prefix("f-0x").or(rest.strip_prefix("f0x"));
        if let (true, Some(h)) = (start, hex) {
            let n = h
                .find(|c: char| !(c.is_ascii_hexdigit() || matches!(c, '.' | 'p' | '+' | '-')))
                .unwrap_or(h.len());
            out.push('f');
            i += rest.len() - h.len() + n;
        } else if start
            && rest.starts_with('i')
            && rest[1..].starts_with(|c: char| c.is_ascii_digit() || c == '-')
        {
            let n = rest[2..]
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len() - 2);
            out.push('i');
            i += 2 + n;
        } else if start && rest.starts_with("s'") {
            // To the quote that ends the value: the next one before a separator.
            let mut end = rest.len();
            let mut from = 2;
            while let Some(q) = rest[from..].find('\'') {
                let after = rest[from + q + 1..].chars().next();
                if after.is_none_or(|c| matches!(c, ',' | '|' | '=' | '>' | ' ')) {
                    end = from + q + 1;
                    break;
                }
                from += q + 1;
            }
            out.push('s');
            i += end;
        } else {
            let c = b[i] as char;
            match c.is_ascii_digit() {
                true if digits => {}
                true => out.push('N'),
                false => out.push(c),
            }
            digits = c.is_ascii_digit();
            i += 1;
            continue;
        }
        digits = false;
    }
    out.chars().take(100).collect()
}

/// What tells two outputs' first difference apart from another cause's: the shapes of the
/// first result each has that the other lacks.
pub fn signature(want: &str, got: &str) -> String {
    let payload = |l: &str| l.split_once(' ').map_or(String::new(), |x| x.1.to_string());
    match (keys(want), keys(got)) {
        (Some(a), Some(b)) => {
            let only = |x: &[&str], y: &[&str]| {
                x.iter()
                    .find(|k| !y.contains(k))
                    .map_or("-".to_string(), |k| shape(k))
            };
            format!("{} | {}", only(&a, &b), only(&b, &a))
        }
        _ => format!("{} | {}", shape(&payload(want)), shape(&payload(got))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_differences_are_taken_out() {
        let acc = Accepted::all_but(&[]);
        assert_eq!(
            acc.normalize("U1q0 E<L4: attempt to index a nil value (field 'x')>"),
            "U1q0 E<L4: attempt to index a nil value>"
        );
        assert_eq!(
            acc.normalize("E<bad argument #2 to 'string.rep' (number expected, got no value)>"),
            acc.normalize("E<bad argument #1 to 'rep' (number expected, got nil)>")
        );
        let strict = Accepted::all_but(&["arg-number".into(), "callee".into()]);
        assert_eq!(
            strict.normalize("E<bad argument #2 to 'string.rep' (number expected, got no value)>"),
            "E<bad argument #2 to 'rep' (number expected, got nil)>"
        );
    }

    #[test]
    fn normalized_warm_up_keys_are_merged() {
        let acc = Accepted::all_but(&[]);
        let line = "U1w H E<L3: number (local 'a') has no integer representation>=2 E<L3: number (local 'b') has no integer representation>=3 #1,i1=4";
        assert_eq!(
            merge_hist(&acc.normalize(line)),
            "U1w H #1,i1=4 E<L3: number has no integer representation>=5"
        );
    }

    #[test]
    fn a_signature_keeps_the_shape_and_drops_the_values() {
        let want = "U2w H #2,i5,f0x1.8p+1=10 #1,s'a b'=3 E<L12: attempt to add a 'number' with a 'string'>=7";
        let got = "U2w H #2,i5,f0x1.8p+1=10 #1,s'a b'=3 E<L12: attempt to add a 'string' with a 'number'>=7";
        assert_eq!(
            signature(want, got),
            "E<LN: attempt to add a 'number' with a 'string'> | E<LN: attempt to add a 'string' with a 'number'>"
        );
        assert_eq!(
            signature("U1q3 #2,i-7,s'x'|nil", "U1q3 #1,f-0x1p+0"),
            "#N,i,s|nil | #N,f"
        );
        assert_eq!(
            signature("U1w H #1,i1=5 #1,i2=5", "U1w H #1,i1=5 #2,i2,nil=5"),
            "#N,i | #N,i,nil"
        );
    }

    #[test]
    fn a_mode_is_found_again_by_its_name() {
        let mut r = Rng::new(7);
        for m in modes(&mut r) {
            let again = mode_named(&m.name).expect("a mode");
            let sorted = |e: &[(String, String)]| {
                let mut e = e.to_vec();
                e.sort();
                e
            };
            assert_eq!(sorted(&again.env), sorted(&m.env), "{}", m.name);
        }
    }
}
