//! tcvm-fuzz: generated Lua programs, run under tcvm with the JIT off and in several JIT
//! configurations and under reference Lua, and compared (`oracle`). `tcvm-fuzz run` fuzzes seed
//! after seed; needs `tcvm-dev` builds of tcvm-cli.

mod ast;
mod exec;
mod generate;
mod oracle;
mod par;
mod reduce;
mod rng;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use exec::{Run, Status, run};
use oracle::{Accepted, Mode};

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

#[derive(Parser)]
#[command(about = "Differential fuzzer for tcvm")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct Common {
    /// tcvm-cli binaries to fuzz, each compared with itself only [default: rel=target/release/tcvm-cli,
    /// and dbg=target/dbg/release/tcvm-cli where it is built]
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
    /// Reference Lua 5.5
    #[arg(long, default_value = "lua")]
    lua: String,
    /// Accepted differences from reference Lua to report after all
    #[arg(long, value_delimiter = ',', value_parser = clap::builder::PossibleValuesParser::new(oracle::RULES))]
    enforce: Vec<String>,
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
        /// Skip the comparison with reference Lua
        #[arg(long)]
        no_ref: bool,
    },
    /// Shrink a program that shows a finding, to FILE.min.lua or p<SEED>.min.lua
    Reduce {
        /// A seed, or a program's file
        program: String,
        /// The finding a candidate must show
        #[arg(long)]
        what: What,
        /// The configuration, as a finding names it (`hot`, `hc4,hl20,check`, `dbg:hot`, …)
        #[arg(long, default_value = "hot")]
        mode: String,
        /// Text the differing line, the JIT log's error, or a crash's stderr must contain,
        /// as it is or with its numbers as `N` (a signature of `triage`)
        #[arg(long)]
        grep: Option<String>,
        /// Stop after this many seconds with the smallest program so far (the output file
        /// always holds it)
        #[arg(long, value_name = "SECS")]
        max_time: Option<u64>,
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

#[derive(Clone, Copy, clap::ValueEnum)]
enum What {
    /// JIT-off differs from reference Lua
    Ref,
    /// MODE differs from JIT-off
    Diverges,
    /// MODE's JIT log has a compiler error
    Internal,
    /// MODE dies of a signal and JIT-off does not
    Crash,
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
        let specs = if c.bins.is_empty() {
            let mut d = vec![format!("rel={REPO}/target/release/tcvm-cli")];
            // Debug assertions check invariants a release build compiles out.
            let dbg = format!("{REPO}/target/dbg/release/tcvm-cli");
            if Path::new(&dbg).exists() {
                d.push(format!("dbg={dbg}"));
            } else {
                eprintln!(
                    "tcvm-fuzz: no {dbg}, fuzzing the release build only (README: Several binaries)"
                );
            }
            d
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
                ("TCVM_JIT_HOT_LOOP", "1".to_string()),
                ("TCVM_JIT_LOG", "1".to_string()),
            ];
            let r = run(&[path, "-f", "probe.lua"], &out.join("prog"), &hot, 10);
            let _ = std::fs::remove_file(&probe);
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
            lua: Some(c.lua.clone()),
            accepted: Accepted::all_but(&c.enforce),
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("tcvm-fuzz: {msg}");
    std::process::exit(2)
}

struct Finding {
    /// `reference` (reference Lua failed: a generator bug), `interp` (the JIT-off run failed),
    /// `ref` (JIT-off differs from reference Lua), `diverges`, `crash`, `hang`, `slow` (a
    /// configuration against JIT-off; `slow` is a timeout under GC stress), `internal` (a
    /// compile failed on a compiler error).
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
        let (s, e) = failure(r);
        d.push_str(&format!(" | {e}"));
        sig = s;
    }
    (unit, sig, d)
}

/// Of a failed run: (the signature, the status with the end of its stderr).
fn failure(r: &Run) -> (String, String) {
    let err = tail(&r.stderr, 300);
    let line = (err.lines())
        .find(|l| l.contains("panicked") || l.contains("error"))
        .unwrap_or("");
    (
        format!("{} | {}", r.status, squash(line)),
        format!("{} | {err}", r.status),
    )
}

/// `s` with every number as `N`, cut short.
fn squash(s: &str) -> String {
    numberless(s).chars().take(100).collect()
}

fn numberless(s: &str) -> String {
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
    out
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
    let reference = cfg.lua.as_ref().and_then(|lua| {
        let r = run(&[lua, &file], &dir, &[], cfg.timeout);
        if r.ok() && r.stdout.ends_with("END\n") {
            return Some(oracle::strict_lines(&r.stdout, &strict, &cfg.accepted));
        }
        let (sig, d) = failure(&r);
        add("reference", "lua", "", String::new(), sig, d);
        None
    });
    let mut r = rng::Rng::new(seed ^ 0x6d6f_6465);
    let modes = oracle::modes(&mut r, prog.units.iter().any(|u| u.big));
    // The debug-assertions build is there for its assertions: JIT-off, everything compiled
    // at once and one drawn configuration under the verifier reach them at a third of the cost.
    let light = {
        let (hot, drawn) = (&modes[1], &modes[2]);
        let checked = match drawn.name.contains("check") {
            true => drawn.clone(),
            false => Mode::named(&format!("{},check", drawn.name)).expect("a mode"),
        };
        vec![hot.clone(), checked]
    };
    let base = [("TCVM_JIT_LOG", "1".to_string()), hash_seed(seed)];
    for (name, bin) in &cfg.bins {
        // A hang already confirmed on this binary: the program hangs, so another timeout is
        // not worth the longer retry.
        let hung = std::cell::Cell::new(false);
        let go = |m: &Mode| {
            let env = [&base[..], &m.env].concat();
            let r = run(&[bin, "-f", &file], &dir, &env, cfg.timeout * m.slow());
            // Once more with more time: a loaded machine is no hang. Not under GC stress, where
            // a timeout is no hang either way.
            if r.status != Status::Timeout || m.gc || hung.get() {
                return r;
            }
            let r = run(&[bin, "-f", &file], &dir, &env, cfg.timeout * m.slow() * 3);
            hung.set(r.status == Status::Timeout);
            r
        };
        let off = go(&Mode::off());
        if !off.ok() || !off.stdout.ends_with("END\n") {
            let (sig, d) = failure(&off);
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
        for m in if name == "dbg" { &light } else { &modes } {
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
                Status::Timeout if m.gc => "slow",
                Status::Timeout => "hang",
                Status::Exit(_) => "diverges",
            };
            let (unit, sig, d) = diff_detail(&off.stdout, &r.stdout, &r);
            add(what, name, &m.name, unit, sig, d);
        }
    }
    (prog, found)
}

fn hash_seed(seed: u64) -> (&'static str, String) {
    ("TCVM_HASH_SEED", (seed % 1000 + 1).to_string())
}

fn cmd_reduce(
    program: &str,
    what: What,
    mode: &str,
    grep: Option<&str>,
    max_time: Option<u64>,
    cfg: &Config,
) {
    let start = Instant::now();
    // `dbg:hot`: the configuration of one binary, as a finding names it.
    let (bin, mode) = match mode.split_once(':') {
        Some((n, m)) => match cfg.bins.iter().find(|(b, _)| b == n) {
            Some((_, p)) => (p, m),
            None => die(&format!("no binary {n}")),
        },
        None => (&cfg.bins[0].1, mode),
    };
    let mode = Mode::named(mode).unwrap_or_else(|| die(&format!("no mode {mode}")));
    let seed: Option<u64> = program.parse().ok();
    // A seed's program file, `p<SEED>.lua` or a reduction of it, keeps the seed's hash seed.
    let file_seed = (Path::new(program).file_stem()).and_then(|s| {
        s.to_str()?
            .strip_prefix('p')?
            .split('.')
            .next()?
            .parse()
            .ok()
    });
    let base = [
        ("TCVM_JIT_LOG", "1".to_string()),
        hash_seed(seed.or(file_seed).unwrap_or(0)),
    ];
    let dirs: Vec<PathBuf> = (0..cfg.jobs)
        .map(|w| {
            let d = cfg.out.join(format!("reduce/w{w}"));
            std::fs::create_dir_all(&d).expect("a worker's directory");
            d
        })
        .collect();
    let lua = cfg.lua.as_deref().expect("a reference");
    // The numbers of a message change as the program shrinks: the signature does not.
    let has = |text: &str| grep.is_none_or(|g| text.contains(g) || numberless(text).contains(g));
    let test = |text: &str, w: usize, limit: Duration, cancel: &exec::Cancel| {
        let (dir, file) = (&dirs[w], "reduce.lua");
        std::fs::write(dir.join(file), text).expect("the candidate's file");
        let timed_out = std::cell::Cell::new(false);
        let tcvm = |m: &Mode| {
            let env = [&base[..], &m.env].concat();
            let r = exec::run_for(
                &[bin, "-f", file],
                dir,
                &env,
                limit * m.slow(),
                Some(cancel),
            );
            timed_out.set(timed_out.get() || r.status == Status::Timeout);
            r
        };
        let off = tcvm(&Mode::off());
        let interesting = off.ok()
            && off.stdout.ends_with("END\n")
            && match what {
                What::Ref => {
                    let r = exec::run_for(&[lua, file], dir, &[], limit, Some(cancel));
                    timed_out.set(timed_out.get() || r.status == Status::Timeout);
                    let strict: Vec<String> = (text.lines())
                        .filter(|l| l.starts_with("do -- U") && l.ends_with(" strict"))
                        .filter_map(|l| l.split_whitespace().nth(2).map(str::to_string))
                        .collect();
                    r.ok() && r.stdout.ends_with("END\n") && {
                        let (a, b) = (
                            oracle::strict_lines(&r.stdout, &strict, &cfg.accepted),
                            oracle::strict_lines(&off.stdout, &strict, &cfg.accepted),
                        );
                        oracle::first_diff(&a, &b).is_some_and(|(w, g)| has(&w) || has(&g))
                    }
                }
                What::Crash => {
                    let r = tcvm(&mode);
                    matches!(r.status, Status::Signal(_)) && has(&r.stderr)
                }
                What::Diverges => {
                    let r = tcvm(&mode);
                    match oracle::first_diff(&off.stdout, &r.stdout) {
                        Some((w, g)) => has(&w) || has(&g),
                        None => !r.ok() && grep.is_none(),
                    }
                }
                What::Internal => oracle::internal_errors(&tcvm(&mode)).iter().any(|l| has(l)),
            };
        reduce::Outcome {
            interesting,
            timed_out: timed_out.get(),
        }
    };
    let text = match seed {
        Some(seed) => generate::Gen::new(seed).program(None).text(),
        None => {
            std::fs::read_to_string(program).unwrap_or_else(|e| die(&format!("{program}: {e}")))
        }
    };
    let out = match seed {
        Some(seed) => cfg.out.join(format!("prog/p{seed}.min.lua")),
        None => PathBuf::from(format!("{}.min.lua", program.trim_end_matches(".lua"))),
    };
    let log = |s: &str| eprintln!("{s} [{:.0}s]", start.elapsed().as_secs_f64());
    let save = |text: &str| std::fs::write(&out, text).expect("the reduced program");
    let options = reduce::Options {
        workers: cfg.jobs,
        timeout_cap: cfg.timeout as f64,
        deadline: max_time.map(|s| start + Duration::from_secs(s)),
        log: &log,
        save: &save,
    };
    let (text, evidence) = reduce::reduce(&text, &test, &options).unwrap_or_else(|e| die(&e));
    if evidence.runs > 0 {
        eprintln!(
            "the result reproduced on {} of {} final replays",
            evidence.interesting, evidence.runs
        );
    }
    save(&text);
    println!("{}", out.display());
}

/// Set by SIGINT and SIGTERM: a run kills the runs in flight, drops their seeds and writes its
/// summary.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

/// From now on a SIGINT or SIGTERM sets `STOP` and kills the runs in flight: a seed that is half
/// run would hold the stop up for as long as all its configurations take.
fn stop_on_signals() {
    let handler = on_stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }
    std::thread::spawn(|| {
        while !STOP.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        eprintln!("tcvm-fuzz: stopping, the seeds in flight are dropped");
        // Again and again: a run may have started since.
        loop {
            exec::ALL.cancel();
            std::thread::sleep(Duration::from_millis(50));
        }
    });
}

/// The findings of one cause, over a whole run.
#[derive(Default)]
struct Group {
    seeds: u64,
    /// Seeds whose programs are kept in `prog/`: the first `--keep`.
    kept: Vec<u64>,
    modes: Vec<String>,
}

impl Group {
    fn kept(&self) -> String {
        let kept: Vec<String> = self.kept.iter().map(|s| s.to_string()).collect();
        kept.join(" ")
    }
}

/// By (kind, signature).
type Groups = BTreeMap<(String, String), Group>;

/// The causes with their kind and signature, the most seen first within each kind.
fn by_count(groups: &Groups) -> Vec<(&str, &str, &Group)> {
    let mut rows: Vec<_> = (groups.iter())
        .map(|((what, sig), g)| (what.as_str(), sig.as_str(), g))
        .collect();
    rows.sort_by_key(|&(what, _, g)| (what, std::cmp::Reverse(g.seeds)));
    rows
}

/// `summary.tsv`: a line per cause.
fn write_summary(dir: &Path, groups: &Groups) {
    let mut text = String::from("what\tseeds\tkept\tmodes\tsignature\n");
    for (what, sig, g) in by_count(groups) {
        text.push_str(&format!(
            "{what}\t{}\t{}\t{}\t{sig}\n",
            g.seeds,
            g.kept(),
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

/// A run's records in its output directory, which the next run into it goes on from:
///
/// - `summary.tsv`: every cause seen (`signature`), how many seeds showed it and under which
///   configurations, and the seeds whose programs are kept.
/// - `findings.tsv`: the findings of the kept programs, one per line.
/// - `prog/p<seed>.lua`: the kept programs, up to `keep` per cause.
/// - `state`: the next seed. `log.txt`: a line per session and per status report.
struct Session<'a> {
    cfg: &'a Config,
    keep: usize,
    groups: Groups,
    findings: std::fs::File,
    log: std::fs::File,
    started: Instant,
    /// The first seed of this session, and the one after the highest done.
    lo: u64,
    top: u64,
    done: u64,
    /// Seeds with findings.
    hit: u64,
}

impl<'a> Session<'a> {
    fn open(cfg: &'a Config, keep: usize, from: Option<u64>) -> Session<'a> {
        let append = |name: &str| {
            let mut file = std::fs::OpenOptions::new();
            let file = file.create(true).append(true).open(cfg.out.join(name));
            file.expect("a file of the run")
        };
        let state = std::fs::read_to_string(cfg.out.join("state")).ok();
        let lo = from
            .or(state.and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0);
        Session {
            cfg,
            keep,
            groups: read_summary(&cfg.out),
            findings: append("findings.tsv"),
            log: append("log.txt"),
            started: Instant::now(),
            lo,
            top: lo,
            done: 0,
            hit: 0,
        }
    }

    fn say(&mut self, line: String) {
        eprintln!("{line}");
        let _ = writeln!(self.log, "{line}");
    }

    fn program(&self, seed: u64) -> PathBuf {
        self.cfg.out.join(format!("prog/p{seed}.lua"))
    }

    /// Count a seed's findings by cause; its program stays if it is one of the first of a cause.
    fn record(&mut self, seed: u64, prog: &generate::Program, found: &[Finding]) {
        self.done += 1;
        self.top = self.top.max(seed + 1);
        self.hit += !found.is_empty() as u64;
        let mut kept = false;
        let mut seen: Vec<(String, String)> = Vec::new();
        for f in found {
            let key = (f.what.to_string(), f.sig.clone());
            let g = self.groups.entry(key.clone()).or_default();
            if !seen.contains(&key) {
                seen.push(key);
                g.seeds += 1;
                if g.kept.len() < self.keep {
                    g.kept.push(seed);
                }
            }
            // With several binaries, which one showed it.
            let mode = match self.cfg.bins.len() {
                1 => f.mode.clone(),
                _ => format!("{}:{}", f.bin, f.mode),
            };
            if !g.modes.contains(&mode) && g.modes.len() < MAX_MODES {
                g.modes.push(mode);
            }
            if g.kept.contains(&seed) {
                kept = true;
                let unit = prog.units.iter().find(|u| u.tag == f.unit);
                let kind = unit.map_or("", |u| u.kind.as_str());
                let detail = f.detail.replace(['\t', '\n'], " ");
                writeln!(
                    self.findings,
                    "{seed}\t{}\t{}\t{}\t{}\t{kind}\t{detail}",
                    f.what, f.bin, f.mode, f.unit
                )
                .expect("a finding");
            }
        }
        if !kept {
            let _ = std::fs::remove_file(self.program(seed));
        }
    }

    /// Write the summary and `next` as the seed the next run starts at.
    fn save(&self, next: u64) {
        write_summary(&self.cfg.out, &self.groups);
        std::fs::write(self.cfg.out.join("state"), next.to_string()).expect("the state");
    }

    fn rate(&self) -> f64 {
        self.done as f64 / self.started.elapsed().as_secs_f64().max(0.001)
    }
}

/// The configurations a cause lists at most.
const MAX_MODES: usize = 12;
/// Seconds between status lines.
const STATUS_SECS: u64 = 30;

/// Fuzz seeds from `from` on (or where the last run into the output directory stopped), until
/// `count` are done, `hours` have passed or a signal stops it.
fn cmd_run(from: Option<u64>, count: Option<u64>, hours: Option<f64>, keep: usize, cfg: &Config) {
    stop_on_signals();
    let mut s = Session::open(cfg, keep, from);
    let rev = std::process::Command::new("git")
        .args(["-C", REPO, "rev-parse", "--short", "HEAD"])
        .output();
    let rev = rev.map_or(String::new(), |o| {
        String::from_utf8_lossy(&o.stdout).trim().into()
    });
    let bins: Vec<String> = cfg.bins.iter().map(|(n, p)| format!("{n}={p}")).collect();
    s.say(format!(
        "run from seed {}, {} jobs, tcvm at {rev}, {}, reference {}",
        s.lo,
        cfg.jobs,
        bins.join(" "),
        cfg.lua.as_deref().unwrap_or("none")
    ));
    let deadline = hours.map(|h| s.started + Duration::from_secs_f64(h * 3600.0));
    let stop = || STOP.load(Ordering::Relaxed) || deadline.is_some_and(|d| Instant::now() >= d);
    let seeds = s.lo..count.map_or(u64::MAX, |n| s.lo.saturating_add(n));
    let mut last = Instant::now();
    let job = |seed| fuzz(seed, cfg);
    par::stream(seeds, cfg.jobs, &stop, job, |seed, (prog, found)| {
        // A killed seed: what its runs returned is no finding, and its program is not kept.
        if exec::ALL.cancelled() {
            let _ = std::fs::remove_file(s.program(seed));
            return;
        }
        s.record(seed, &prog, &found);
        if last.elapsed().as_secs() >= STATUS_SECS {
            last = Instant::now();
            // Seeds below this are done or lost with the workers: a few run twice after a kill.
            s.save(s.top.saturating_sub(4 * cfg.jobs as u64).max(s.lo));
            s.say(format!(
                "{:.1}h: {} seeds ({:.1}/s), {} with findings, {} causes, at seed {}",
                s.started.elapsed().as_secs_f64() / 3600.0,
                s.done,
                s.rate(),
                s.hit,
                s.groups.len(),
                s.top
            ));
        }
    });
    s.save(s.top);
    s.say(format!(
        "stopped at seed {}: {} seeds in {:.0}s ({:.1}/s), {} with findings, {} causes; {}",
        s.top,
        s.done,
        s.started.elapsed().as_secs_f64(),
        s.rate(),
        s.hit,
        s.groups.len(),
        cfg.out.join("summary.tsv").display()
    ));
}

fn cmd_triage(dir: Option<&Path>) {
    let dir = dir.map_or_else(|| Path::new(REPO).join("target/fuzz"), Path::to_path_buf);
    for (what, sig, g) in by_count(&read_summary(&dir)) {
        println!(
            "== {what}: {} seeds (kept {}) in {}",
            g.seeds,
            g.kept(),
            g.modes.join(" ")
        );
        println!("   {sig}");
    }
}

fn cmd_cover(lo: u64, hi: u64, cfg: &Config) {
    let bin = &cfg.bins[0].1;
    let dir = cfg.out.join("prog");
    let job = |seed: u64| {
        let file = format!("c{seed}.lua");
        std::fs::write(
            dir.join(&file),
            generate::Gen::new(seed).program(None).text(),
        )
        .expect("the program's file");
        let mut stats = Vec::new();
        for tag in ["off", "hot"] {
            let st = format!("c{seed}.{tag}.stats");
            let mut env = Mode::named(tag).expect("a mode").env;
            env.push(("TCVM_STATS", st.clone()));
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
    // Every opcode of the binary: its stats list the ones it never dispatched too.
    let mut ops: BTreeMap<String, u64> = BTreeMap::new();
    let mut events: BTreeMap<String, u64> = BTreeMap::new();
    let mut failed = 0;
    par::stream(lo..hi, cfg.jobs, &|| false, job, |_, stats| {
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
    let missing: Vec<&str> = (ops.iter())
        .filter(|&(_, &n)| n == 0)
        .map(|(op, _)| op.as_str())
        .collect();
    println!(
        "{} of {} opcodes dispatched with the JIT off; never: {}",
        ops.len() - missing.len(),
        ops.len(),
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
        Cmd::Gen { seed, kind } => {
            if let Some(k) = kind.as_deref().filter(|k| !generate::is_kind(k)) {
                die(&format!("no unit kind {k}"));
            }
            let prog = generate::Gen::new(seed).program(kind.as_deref());
            print!("{}", prog.text())
        }
        Cmd::Run {
            from,
            count,
            hours,
            keep,
            common,
            no_ref,
        } => {
            let mut cfg = Config::new(&common);
            if no_ref {
                cfg.lua = None;
            }
            cmd_run(from, count, hours, keep, &cfg);
        }
        Cmd::Reduce {
            program,
            what,
            mode,
            grep,
            max_time,
            common,
        } => cmd_reduce(
            &program,
            what,
            &mode,
            grep.as_deref(),
            max_time,
            &Config::new(&common),
        ),
        Cmd::Triage { dir } => cmd_triage(dir.as_deref()),
        Cmd::Cover { lo, hi, common } => cmd_cover(lo, hi, &Config::new(&common)),
    }
}
