//! tcvm-fuzz: generated Lua programs, run under tcvm with the JIT off and in several JIT
//! configurations and under reference Lua, and compared (`oracle`). `tcvm-fuzz run 0 1000`
//! fuzzes the seeds 0..1000; needs `tcvm-dev` builds of tcvm-cli.

mod ast;
mod exec;
mod generate;
mod oracle;
mod reduce;
mod rng;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;

use clap::{Parser, Subcommand};
use exec::{Run, Status, run};
use oracle::Accepted;

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

#[derive(Parser)]
#[command(about = "Differential fuzzer for tcvm")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct Common {
    /// tcvm-cli binaries to fuzz, each compared with itself only [default: target/release/tcvm-cli]
    #[arg(long = "bin", value_name = "[NAME=]PATH")]
    bins: Vec<String>,
    /// Where programs and results go [default: target/fuzz]
    #[arg(long)]
    out: Option<PathBuf>,
    /// Programs fuzzed at a time [default: one less than the cores]
    #[arg(short, long)]
    jobs: Option<usize>,
    /// Seconds a run may take
    #[arg(long, default_value_t = 10)]
    timeout: u32,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the program of a seed; with KIND, every unit of that kind (`free`, `limits`, or a
    /// template's name)
    Gen { seed: u64, kind: Option<String> },
    /// Fuzz seed after seed until stopped (Ctrl-C), going on where the last run into the same
    /// output directory stopped; `summary.tsv` there lists what was found
    Run {
        /// Start at this seed instead
        #[arg(long)]
        from: Option<u64>,
        /// Stop after this many seeds
        #[arg(long)]
        count: Option<u64>,
        /// Stop after this long
        #[arg(long)]
        hours: Option<f64>,
        /// Programs to keep per cause
        #[arg(long, default_value_t = 5)]
        keep: usize,
        #[command(flatten)]
        common: Common,
        /// Reference Lua 5.5
        #[arg(long, default_value = "lua")]
        lua: String,
        /// Skip the comparison with reference Lua
        #[arg(long)]
        no_ref: bool,
        /// Accepted differences from reference Lua to report after all
        #[arg(long, value_delimiter = ',', value_parser = clap::builder::PossibleValuesParser::new(oracle::RULES))]
        enforce: Vec<String>,
    },
    /// Shrink a program that shows a finding, to FILE.min.lua or p<SEED>.min.lua
    Reduce {
        /// A seed, or a program's file
        program: String,
        /// `ref`: JIT-off differs from reference Lua. `diverges`: MODE differs from JIT-off.
        /// `internal`: MODE's JIT log has a compiler error. `crash`: MODE dies of a signal
        #[arg(long, value_parser = ["ref", "diverges", "internal", "crash"])]
        what: String,
        /// The configuration, as a finding names it (`hot`, `hc4,hl20,check`, …)
        #[arg(long, default_value = "hot")]
        mode: String,
        /// Text the differing line, the JIT log's error, or a crash's stderr must contain
        #[arg(long)]
        grep: Option<String>,
        #[arg(long, default_value = "lua")]
        lua: String,
        #[command(flatten)]
        common: Common,
    },
    /// Print a run's causes, the most seen first
    Triage {
        /// The run's output directory [default: target/fuzz]
        dir: Option<PathBuf>,
    },
    /// What the programs of the seeds LO..HI reach: opcodes the interpreter never dispatched
    /// and the JIT's events (`TCVM_STATS`)
    Cover {
        lo: u64,
        hi: u64,
        #[command(flatten)]
        common: Common,
    },
}

struct Config {
    bins: Vec<(String, String)>,
    out: PathBuf,
    jobs: usize,
    timeout: u32,
    lua: Option<String>,
    accepted: Accepted,
}

impl Config {
    fn new(c: &Common) -> Config {
        let out = c
            .out
            .clone()
            .unwrap_or_else(|| Path::new(REPO).join("target/fuzz"));
        std::fs::create_dir_all(out.join("prog")).expect("the output directory");
        let out = out.canonicalize().expect("the output directory");
        let default = format!("{REPO}/target/release/tcvm-cli");
        let specs = if c.bins.is_empty() {
            vec![default]
        } else {
            c.bins.clone()
        };
        let bins: Vec<(String, String)> = (specs.iter())
            .map(|b| {
                let (name, path) = b.split_once('=').unwrap_or(("tcvm", b));
                let path = Path::new(path)
                    .canonicalize()
                    .unwrap_or_else(|e| die(&format!("{path}: {e}")));
                (name.to_string(), path.to_string_lossy().into_owned())
            })
            .collect();
        for (i, (name, path)) in bins.iter().enumerate() {
            if bins[..i].iter().any(|(n, _)| n == name) {
                die("give the binaries distinct names: --bin NAME=PATH");
            }
            let probe = out.join("prog/probe.lua");
            std::fs::write(&probe, "local s = 0 for i = 1, 100 do s = s + i end\n")
                .expect("a probe");
            let hot = [
                ("TCVM_JIT_HOT_LOOP".to_string(), "1".to_string()),
                ("TCVM_JIT_LOG".to_string(), "1".to_string()),
            ];
            let r = run(&[path, "-f", "probe.lua"], &out.join("prog"), &hot, 10);
            if !r.ok() || !r.stderr.contains("jit:") {
                die(&format!(
                    "{path}: does not run, or is not a tcvm-dev build (no TCVM_JIT_LOG output)"
                ));
            }
        }
        Config {
            bins,
            out,
            jobs: c.jobs.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1))
            }),
            timeout: c.timeout,
            lua: None,
            accepted: Accepted::all_but(&[]),
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("tcvm-fuzz: {msg}");
    std::process::exit(2)
}

struct Finding {
    /// `reference` (reference Lua failed: a generator bug), `interp` (the JIT-off run failed),
    /// `ref` (JIT-off differs from reference Lua), `diverges`, `crash`, `hang` (a JIT
    /// configuration against JIT-off), `internal` (a compile failed on a compiler error).
    what: &'static str,
    bin: String,
    mode: String,
    unit: String,
    /// What findings of one cause share.
    sig: String,
    detail: String,
}

fn tail(s: &str, n: usize) -> String {
    let t: Vec<char> = s.chars().collect();
    t[t.len().saturating_sub(n)..].iter().collect()
}

fn clip(s: &str) -> String {
    s.chars().take(240).collect()
}

/// Of two outputs' first difference: (the unit, the signature, the lines).
fn diff_detail(want: &str, got: &str, r: &Run) -> (String, String, String) {
    let (w, g) = oracle::first_diff(want, got).unwrap_or_default();
    let unit = oracle::unit_of(&w)
        .or(oracle::unit_of(&g))
        .unwrap_or("")
        .to_string();
    let mut d = format!("want {} | got {}", clip(&w), clip(&g));
    let mut sig = oracle::signature(&w, &g);
    if !r.ok() {
        d.push_str(&format!(" | {} | {}", r.status, tail(&r.stderr, 300)));
        sig = format!(
            "{} | {}",
            r.status,
            squash(
                tail(&r.stderr, 300)
                    .lines()
                    .find(|l| l.contains("panicked") || l.contains("error"))
                    .unwrap_or("")
            )
        );
    }
    (unit, sig, d)
}

/// `s` with every number as `N`, cut short.
fn squash(s: &str) -> String {
    let mut out = String::new();
    let mut digits = false;
    for c in s.chars() {
        match c.is_ascii_digit() {
            true if digits => {}
            true => out.push('N'),
            false => out.push(c),
        }
        digits = c.is_ascii_digit();
    }
    out.chars().take(100).collect()
}

fn fuzz(seed: u64, cfg: &Config) -> (generate::Program, Vec<Finding>) {
    let prog = generate::Gen::new(seed).program(None);
    let dir = cfg.out.join("prog");
    let file = format!("p{seed}.lua");
    std::fs::write(dir.join(&file), prog.text()).expect("the program's file");
    let strict: Vec<String> = prog
        .units
        .iter()
        .filter(|u| u.strict)
        .map(|u| u.tag.clone())
        .collect();
    let mut found = Vec::new();
    let mut add = |what, bin: &str, mode: &str, unit: String, sig: String, detail: String| {
        found.push(Finding {
            what,
            bin: bin.into(),
            mode: mode.into(),
            unit,
            sig,
            detail,
        })
    };
    let failed = |r: &Run| {
        (
            format!(
                "{} | {}",
                r.status,
                squash(
                    tail(&r.stderr, 300)
                        .lines()
                        .find(|l| l.contains("panicked") || l.contains("error"))
                        .unwrap_or("")
                )
            ),
            format!("{} | {}", r.status, tail(&r.stderr, 300)),
        )
    };
    let reference = cfg.lua.as_ref().and_then(|lua| {
        let r = run(&[lua, &file], &dir, &[], cfg.timeout);
        if r.ok() && r.stdout.ends_with("END\n") {
            return Some(oracle::strict_lines(&r.stdout, &strict, &cfg.accepted));
        }
        let (sig, d) = failed(&r);
        add("reference", "lua", "", String::new(), sig, d);
        None
    });
    let mut r = rng::Rng::new(seed ^ 0x6d6f_6465);
    let modes = oracle::modes(&mut r);
    let base = [
        ("TCVM_JIT_LOG".to_string(), "1".to_string()),
        hash_seed(seed),
    ];
    for (name, bin) in &cfg.bins {
        let go = |m: &oracle::Mode| {
            let env: Vec<(String, String)> = base.iter().chain(&m.env).cloned().collect();
            let r = run(&[bin, "-f", &file], &dir, &env, cfg.timeout * m.slow);
            // Once more with more time: a loaded machine is no hang.
            match r.status {
                Status::Timeout => run(&[bin, "-f", &file], &dir, &env, cfg.timeout * m.slow * 3),
                _ => r,
            }
        };
        let off = go(&oracle::off());
        if !off.ok() || !off.stdout.ends_with("END\n") {
            let (sig, d) = failed(&off);
            add("interp", name, "off", String::new(), sig, d);
            continue;
        }
        if let Some(want) = &reference {
            let got = oracle::strict_lines(&off.stdout, &strict, &cfg.accepted);
            if *want != got {
                let (unit, sig, d) = diff_detail(want, &got, &off);
                add("ref", name, "off", unit, sig, d);
            }
        }
        for m in &modes {
            let r = go(m);
            for l in oracle::internal_errors(&r) {
                let msg = l.split_once("failed: ").map_or(l.as_str(), |x| x.1);
                add(
                    "internal",
                    name,
                    &m.name,
                    String::new(),
                    squash(msg),
                    l.clone(),
                );
            }
            if r.ok() && r.stdout == off.stdout {
                continue;
            }
            let what = match r.status {
                Status::Signal(_) => "crash",
                // Collecting at every step is slow, not stuck.
                Status::Timeout if m.name.contains("gc") => "slow",
                Status::Timeout => "hang",
                Status::Exit(_) => "diverges",
            };
            let (unit, sig, d) = diff_detail(&off.stdout, &r.stdout, &r);
            add(what, name, &m.name, unit, sig, d);
        }
    }
    (prog, found)
}

fn hash_seed(seed: u64) -> (String, String) {
    ("TCVM_HASH_SEED".to_string(), (seed % 1000 + 1).to_string())
}

fn cmd_reduce(program: &str, what: &str, mode: &str, grep: Option<&str>, lua: &str, cfg: &Config) {
    let mode = oracle::mode_named(mode).unwrap_or_else(|| die(&format!("no mode {mode}")));
    let seed: Option<u64> = program.parse().ok();
    let dir = cfg.out.join("prog");
    let (_, bin) = &cfg.bins[0];
    let base = [
        ("TCVM_JIT_LOG".to_string(), "1".to_string()),
        hash_seed(seed.unwrap_or(0)),
    ];
    // A candidate that lost a loop's exit is not worth the full limit.
    let secs = cfg.timeout.min(2);
    let tests = std::cell::Cell::new(0u32);
    let has = |line: &str| grep.is_none_or(|g| line.contains(g));
    let fails = |text: &str| {
        tests.set(tests.get() + 1);
        let file = "reduce.lua";
        std::fs::write(dir.join(file), text).expect("the candidate's file");
        let tcvm = |m: &oracle::Mode| {
            let env: Vec<(String, String)> = base.iter().chain(&m.env).cloned().collect();
            run(&[bin, "-f", file], &dir, &env, secs * m.slow)
        };
        let off = tcvm(&oracle::off());
        if !off.ok() || !off.stdout.ends_with("END\n") {
            return false;
        }
        match what {
            "ref" => {
                let r = run(&[lua, file], &dir, &[], secs);
                if !r.ok() || !r.stdout.ends_with("END\n") {
                    return false;
                }
                let strict: Vec<String> = (text.lines())
                    .filter(|l| l.starts_with("do -- U") && l.ends_with(" strict"))
                    .filter_map(|l| l.split_whitespace().nth(2).map(str::to_string))
                    .collect();
                let (a, b) = (
                    oracle::strict_lines(&r.stdout, &strict, &cfg.accepted),
                    oracle::strict_lines(&off.stdout, &strict, &cfg.accepted),
                );
                oracle::first_diff(&a, &b).is_some_and(|(w, g)| has(&w) || has(&g))
            }
            "crash" => (0..2).any(|_| {
                let r = tcvm(&mode);
                matches!(r.status, Status::Signal(_)) && grep.is_none_or(|g| r.stderr.contains(g))
            }),
            "diverges" => {
                // Twice: some findings depend on timing.
                (0..2).any(|_| {
                    let r = tcvm(&mode);
                    match oracle::first_diff(&off.stdout, &r.stdout) {
                        Some((w, g)) => has(&w) || has(&g),
                        None => !r.ok() && grep.is_none(),
                    }
                })
            }
            _ => oracle::internal_errors(&tcvm(&mode)).iter().any(|l| has(l)),
        }
    };
    let log = |s: &str| eprintln!("{s} ({} runs)", tests.get());
    let text = match seed {
        Some(seed) => {
            let p = generate::Gen::new(seed).program(None);
            if !fails(&p.text()) {
                die("the program does not show the finding");
            }
            reduce::program(p, &fails, &log).text()
        }
        None => {
            let text = std::fs::read_to_string(program)
                .unwrap_or_else(|e| die(&format!("{program}: {e}")));
            if !fails(&text) {
                die("the program does not show the finding");
            }
            text
        }
    };
    let text = reduce::lines(&text, &fails, &log);
    let out = match seed {
        Some(seed) => cfg.out.join(format!("prog/p{seed}.min.lua")),
        None => PathBuf::from(format!("{}.min.lua", program.trim_end_matches(".lua"))),
    };
    std::fs::write(&out, text).expect("the reduced program");
    println!("{}", out.display());
}

/// Run `job` on every seed of `lo..hi` until `stop`, on `jobs` threads; `each` gets the
/// results as they come.
fn for_seeds<T: Send>(
    lo: u64,
    hi: u64,
    jobs: usize,
    stop: &(dyn Fn() -> bool + Sync),
    job: impl Fn(u64) -> T + Sync,
    mut each: impl FnMut(u64, T),
) {
    let next = AtomicU64::new(lo);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|s| {
        for _ in 0..jobs {
            let (tx, next, job) = (tx.clone(), &next, &job);
            s.spawn(move || {
                while !stop() {
                    let seed = next.fetch_add(1, Ordering::Relaxed);
                    if seed >= hi || tx.send((seed, job(seed))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (seed, t) in rx {
            each(seed, t);
        }
    });
}

/// Set by SIGINT and SIGTERM: a run stops taking seeds and writes its summary.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

/// The findings of one cause, over a whole run.
#[derive(Default)]
struct Group {
    seeds: u64,
    /// Seeds whose programs are kept in `prog/`: the first `--keep`.
    kept: Vec<u64>,
    modes: Vec<String>,
}

type Groups = BTreeMap<(String, String), Group>;

/// `summary.tsv`: a line per cause, the most seen first within each kind.
fn write_summary(dir: &Path, groups: &Groups) {
    let mut rows: Vec<_> = groups.iter().collect();
    rows.sort_by_key(|(k, g)| (k.0.clone(), std::cmp::Reverse(g.seeds)));
    let mut text = String::from("what\tseeds\tkept\tmodes\tsignature\n");
    for ((what, sig), g) in rows {
        let kept: Vec<String> = g.kept.iter().map(|s| s.to_string()).collect();
        text.push_str(&format!(
            "{what}\t{}\t{}\t{}\t{sig}\n",
            g.seeds,
            kept.join(" "),
            g.modes.join(" ")
        ));
    }
    let tmp = dir.join("summary.tsv.tmp");
    std::fs::write(&tmp, text).expect("the summary");
    std::fs::rename(tmp, dir.join("summary.tsv")).expect("the summary");
}

fn read_summary(dir: &Path) -> Groups {
    let mut groups = Groups::new();
    let text = std::fs::read_to_string(dir.join("summary.tsv")).unwrap_or_default();
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.splitn(5, '\t').collect();
        let [what, seeds, kept, modes, sig] = c[..] else {
            continue;
        };
        let g = Group {
            seeds: seeds.parse().unwrap_or(0),
            kept: kept.split(' ').filter_map(|s| s.parse().ok()).collect(),
            modes: modes
                .split(' ')
                .filter(|m| !m.is_empty())
                .map(str::to_string)
                .collect(),
        };
        groups.insert((what.to_string(), sig.to_string()), g);
    }
    groups
}

/// Fuzz seeds from `from` on, until `count` are done, `hours` have passed or a signal stops
/// it. The output directory holds the run's state, so the next run goes on where this one
/// stopped:
///
/// - `summary.tsv`: every cause seen (`signature`), how many seeds showed it and under which
///   configurations, and the seeds whose programs are kept.
/// - `findings.tsv`: the findings of the kept programs, one per line.
/// - `prog/p<seed>.lua`: the kept programs, up to `keep` per cause.
/// - `state`: the next seed. `log.txt`: a line per session and per status report.
fn cmd_run(from: Option<u64>, count: Option<u64>, hours: Option<f64>, keep: usize, cfg: &Config) {
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(
            libc::SIGINT,
            on_stop as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            on_stop as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }
    let state = cfg.out.join("state");
    let resume = std::fs::read_to_string(&state)
        .ok()
        .and_then(|s| s.trim().parse().ok());
    let lo: u64 = from.or(resume).unwrap_or(0);
    let hi = count.map_or(u64::MAX, |n| lo.saturating_add(n));
    let mut groups = read_summary(&cfg.out);
    let mut findings = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("findings.tsv"))
        .expect("findings.tsv");
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("log.txt"))
        .expect("log.txt");
    let rev = std::process::Command::new("git")
        .args(["-C", REPO, "rev-parse", "--short", "HEAD"])
        .output();
    let rev = rev
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let bins: Vec<String> = cfg.bins.iter().map(|(n, p)| format!("{n}={p}")).collect();
    let mut say = |line: String| {
        eprintln!("{line}");
        let _ = writeln!(log, "{line}");
    };
    say(format!(
        "run from seed {lo}, {} jobs, tcvm at {rev}, {}, reference {}",
        cfg.jobs,
        bins.join(" "),
        cfg.lua.as_deref().unwrap_or("none")
    ));
    let started = std::time::Instant::now();
    let mut last = started;
    let (mut done, mut hit, mut top) = (0u64, 0u64, lo);
    let deadline = hours.map(|h| started + std::time::Duration::from_secs_f64(h * 3600.0));
    let stop =
        || STOP.load(Ordering::Relaxed) || deadline.is_some_and(|d| std::time::Instant::now() >= d);
    for_seeds(
        lo,
        hi,
        cfg.jobs,
        &stop,
        |seed| fuzz(seed, cfg),
        |seed, (prog, found)| {
            done += 1;
            top = top.max(seed + 1);
            hit += !found.is_empty() as u64;
            let mut keep_program = false;
            let mut seen: Vec<(String, String)> = Vec::new();
            for f in &found {
                let key = (f.what.to_string(), f.sig.clone());
                let g = groups.entry(key.clone()).or_default();
                if !seen.contains(&key) {
                    seen.push(key);
                    g.seeds += 1;
                    if g.kept.len() < keep {
                        g.kept.push(seed);
                    }
                }
                // With several binaries, which one showed it.
                let mode = match cfg.bins.len() {
                    1 => f.mode.clone(),
                    _ => format!("{}:{}", f.bin, f.mode),
                };
                if !g.modes.contains(&mode) && g.modes.len() < 12 {
                    g.modes.push(mode);
                }
                if g.kept.contains(&seed) {
                    keep_program = true;
                    let unit = prog.units.iter().find(|u| u.tag == f.unit);
                    let kind = unit.map_or("", |u| u.kind.as_str());
                    let detail = f.detail.replace(['\t', '\n'], " ");
                    writeln!(
                        findings,
                        "{seed}\t{}\t{}\t{}\t{}\t{kind}\t{detail}",
                        f.what, f.bin, f.mode, f.unit
                    )
                    .expect("a finding");
                }
            }
            if !keep_program {
                let _ = std::fs::remove_file(cfg.out.join(format!("prog/p{seed}.lua")));
            }
            if last.elapsed().as_secs() >= 30 {
                last = std::time::Instant::now();
                write_summary(&cfg.out, &groups);
                // Seeds below this are done or lost with the workers: a few run twice after a kill.
                let _ = std::fs::write(
                    &state,
                    top.saturating_sub(4 * cfg.jobs as u64).max(lo).to_string(),
                );
                let secs = started.elapsed().as_secs_f64();
                say(format!(
                    "{:.1}h: {done} seeds ({:.1}/s), {hit} with findings, {} causes, at seed {top}",
                    secs / 3600.0,
                    done as f64 / secs,
                    groups.len()
                ));
            }
        },
    );
    write_summary(&cfg.out, &groups);
    std::fs::write(&state, top.to_string()).expect("the state");
    let secs = started.elapsed().as_secs_f64();
    say(format!(
        "stopped at seed {top}: {done} seeds in {secs:.0}s ({:.1}/s), {hit} with findings, {} causes; {}",
        done as f64 / secs.max(0.001),
        groups.len(),
        cfg.out.join("summary.tsv").display()
    ));
}

fn cmd_triage(dir: Option<&Path>) {
    let dir = dir.map_or_else(|| Path::new(REPO).join("target/fuzz"), Path::to_path_buf);
    let groups = read_summary(&dir);
    let mut rows: Vec<_> = groups.iter().collect();
    rows.sort_by_key(|(k, g)| (k.0.clone(), std::cmp::Reverse(g.seeds)));
    for ((what, sig), g) in rows {
        let kept: Vec<String> = g.kept.iter().map(|s| s.to_string()).collect();
        println!(
            "== {what}: {} seeds (kept {}) in {}",
            g.seeds,
            kept.join(" "),
            g.modes.join(" ")
        );
        println!("   {sig}");
    }
}

/// tcvm's opcodes, from the table in its source.
fn opcodes() -> Vec<&'static str> {
    (include_str!("../../src/instruction.rs").lines())
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            let (code, name) = (w.next()?, w.next()?);
            let hex = code.strip_prefix("0x")?;
            (hex.len() == 2
                && u8::from_str_radix(hex, 16).is_ok()
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .then_some(name)
        })
        .collect()
}

fn cmd_cover(lo: u64, hi: u64, cfg: &Config) {
    let (name, bin) = &cfg.bins[0];
    let _ = name;
    let dir = cfg.out.join("prog");
    let job = |seed: u64| {
        let file = format!("c{seed}.lua");
        std::fs::write(
            dir.join(&file),
            generate::Gen::new(seed).program(None).text(),
        )
        .expect("the program's file");
        let mut stats = Vec::new();
        for (tag, mode) in [
            ("off", oracle::off()),
            ("hot", oracle::modes(&mut rng::Rng::new(0)).swap_remove(1)),
        ] {
            let st = format!("c{seed}.{tag}.stats");
            let mut env = mode.env.clone();
            env.push(("TCVM_STATS".into(), st.clone()));
            let r = run(&[bin, "-f", &file], &dir, &env, cfg.timeout);
            stats.push((
                r.ok(),
                std::fs::read_to_string(dir.join(&st)).unwrap_or_default(),
            ));
            let _ = std::fs::remove_file(dir.join(&st));
        }
        let _ = std::fs::remove_file(dir.join(&file));
        stats
    };
    let mut ops: BTreeMap<String, u64> = BTreeMap::new();
    let mut events: BTreeMap<String, u64> = BTreeMap::new();
    let mut failed = 0;
    for_seeds(lo, hi, cfg.jobs, &|| false, job, |_, stats| {
        for (i, (ok, text)) in stats.iter().enumerate() {
            failed += !ok as u32;
            for l in text.lines() {
                let Some((key, n)) = l.rsplit_once(' ') else {
                    continue;
                };
                let n: u64 = n.parse().unwrap_or(0);
                match key.strip_prefix("op ") {
                    // Dispatches with the JIT off: what the interpreter itself ran.
                    Some(op) if i == 0 => *ops.entry(op.into()).or_default() += n,
                    Some(_) => {}
                    None => *events.entry(key.into()).or_default() += n,
                }
            }
        }
    });
    let all = opcodes();
    let missing: Vec<&str> = all
        .iter()
        .copied()
        .filter(|o| !ops.contains_key(*o))
        .collect();
    println!(
        "{} of {} opcodes dispatched with the JIT off; never: {}",
        all.len() - missing.len(),
        all.len(),
        missing.join(" ")
    );
    println!("with everything compiled at its first count:");
    for (k, n) in &events {
        println!("  {k} {n}");
    }
    if failed > 0 {
        println!("{failed} runs failed");
    }
}

fn main() {
    match Args::parse().cmd {
        Cmd::Gen { seed, kind } => print!(
            "{}",
            generate::Gen::new(seed).program(kind.as_deref()).text()
        ),
        Cmd::Run {
            from,
            count,
            hours,
            keep,
            common,
            lua,
            no_ref,
            enforce,
        } => {
            let mut cfg = Config::new(&common);
            cfg.lua = (!no_ref).then_some(lua);
            cfg.accepted = Accepted::all_but(&enforce);
            cmd_run(from, count, hours, keep, &cfg);
        }
        Cmd::Reduce {
            program,
            what,
            mode,
            grep,
            lua,
            common,
        } => cmd_reduce(
            &program,
            &what,
            &mode,
            grep.as_deref(),
            &lua,
            &Config::new(&common),
        ),
        Cmd::Triage { dir } => cmd_triage(dir.as_deref()),
        Cmd::Cover { lo, hi, common } => cmd_cover(lo, hi, &Config::new(&common)),
    }
}
