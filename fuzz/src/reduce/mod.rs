//! Shrinking a program that shows a finding: shrinkray's reducer (`reducer.py`) and the passes of
//! it that apply to Lua, the grammar ones on tcvm's own parser. Candidates are tested in
//! parallel and their results taken in candidate order, so a reduction is a function of its input
//! alone. Smaller is fewer tokens, then fewer bytes; deletions keep line breaks, so the result
//! stays one statement per line.
//!
//! The schedule: initial cuts until a round gains under 1%, then the great passes (re-running the
//! ones that worked until they stop), the ok passes once the great ones stall, the last-ditch
//! ones, polish, and the passes a probation budget cut short; at the fixpoint the test gets a
//! chance to unstick (a raised time limit, a confirmation sweep), and then the reduction restarts
//! from the original, accepting only results below the fixpoint, while that improves it.

mod nondet;
mod passes;
mod patch;
mod problem;
mod timeout;
mod tree;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use problem::{Control, Oracle, Problem, Watch};
pub use problem::{Outcome, Test};

use crate::rng::Rng;

type Run = fn(&mut Problem, &mut Rng);

#[derive(Clone, Copy)]
struct Pass {
    name: &'static str,
    run: Run,
}

macro_rules! pass {
    ($name:literal, $f:expr) => {
        Pass {
            name: $name,
            run: $f,
        }
    };
}

fn grammar() -> [Pass; 4] {
    [
        pass!("tree/delete_children", tree::delete_children),
        pass!(
            "tree/delete_orphaned_declarations",
            tree::delete_orphaned_declarations
        ),
        pass!("tree/lift_nodes", tree::lift_nodes),
        pass!("tree/substitute_nodes", tree::substitute_nodes),
    ]
}

fn initial_cuts() -> Vec<Pass> {
    let mut v = vec![
        pass!("cut_comments", passes::cut_comments),
        pass!("hollow", passes::hollow),
        pass!("lines/delete_duplicates", passes::delete_duplicate_lines),
        pass!("lines/block_deletion(10,100)", |p, r| {
            passes::block_deletion(p, r, "\n", 10, 100)
        }),
        pass!("lift_braces", passes::lift_braces),
        pass!("remove_indents", passes::remove_indents),
        pass!("remove_whitespace", passes::remove_whitespace),
    ];
    v.extend(grammar());
    v
}

fn great_passes() -> Vec<Pass> {
    let mut v = vec![
        pass!("lines/delete_duplicates", passes::delete_duplicate_lines),
        pass!("lines/block_deletion(1,10)", |p, r| passes::block_deletion(
            p, r, "\n", 1, 10
        )),
        pass!("semicolons/block_deletion(1,10)", |p, r| {
            passes::block_deletion(p, r, ";", 1, 10)
        }),
        pass!("remove_indents", passes::remove_indents),
        pass!("hollow", passes::hollow),
        pass!("lift_braces", passes::lift_braces),
        pass!("debracket", passes::debracket),
    ];
    v.extend(grammar());
    v
}

fn ok_passes() -> Vec<Pass> {
    vec![
        pass!("token_spans", passes::token_spans),
        pass!(
            "lines/block_deletion(11,20)",
            |p, r| passes::block_deletion(p, r, "\n", 11, 20)
        ),
        pass!("remove_indents", passes::remove_indents),
        pass!("remove_whitespace", passes::remove_whitespace),
        pass!("reduce_integer_literals", passes::reduce_integer_literals),
        pass!("replace_falsey_with_zero", passes::replace_falsey_with_zero),
        pass!(
            "replace_identifiers_with_zero",
            passes::replace_identifiers_with_zero
        ),
        pass!("combine_expressions", passes::combine_expressions),
        pass!("merge_adjacent_strings", passes::merge_adjacent_strings),
        pass!("lexeme_based_deletions", passes::lexeme_based_deletions),
        pass!("normalize_identifiers", passes::normalize_identifiers),
        pass!("line_sorter", passes::line_sorter),
    ]
}

fn last_ditch_passes() -> Vec<Pass> {
    vec![
        // Many candidates, few hits: cheaper late, on reduced test cases.
        pass!("tokens/block_deletion(1,20)", |p, r| {
            passes::token_block_deletion(p, r, 1, 20)
        }),
        pass!("lines/block_deletion(21,100)", |p, r| {
            passes::block_deletion(p, r, "\n", 21, 100)
        }),
        pass!("token_spans", passes::token_spans),
        pass!("simplify_brackets", passes::simplify_brackets),
        pass!("cut_comments", passes::cut_comments),
    ]
}

fn polish_passes() -> Vec<Pass> {
    vec![pass!("short_deletions", passes::short_deletions)]
}

/// Consecutive tests without progress after which a pass on probation (its last complete run
/// made none) is set aside; set-aside passes run to completion before the reduction ends.
const PROBATION_BUDGET: u64 = 25;

/// The initial cuts' watcher looks at a pass every this many tests (shrinkray: every 5 seconds),
/// so it depends neither on timing nor on the number of workers.
const WATCH_TESTS: u64 = 1000;

struct Reducer {
    rng: Rng,
    unlocked_ok: bool,
    probation: HashMap<&'static str, bool>,
    incomplete: BTreeMap<&'static str, Pass>,
    /// For a pass that ran to completion without progress, the version of the test case it ran
    /// on.
    fingerprints: HashMap<&'static str, u64>,
}

impl Reducer {
    fn new(rng: Rng) -> Reducer {
        Reducer {
            rng,
            unlocked_ok: false,
            probation: HashMap::new(),
            incomplete: BTreeMap::new(),
            fingerprints: HashMap::new(),
        }
    }

    fn log(&self, p: &Problem, name: &str) {
        let (tokens, bytes) = problem::sort_key(p.current());
        p.oracle.log(&format!(
            "{name}: {tokens} tokens, {bytes} bytes ({} runs)",
            p.oracle.calls
        ));
    }

    /// Run a pass; `watch` is the initial cuts' watcher.
    fn run_pass(&mut self, p: &mut Problem, pass: Pass, budgeted: bool, watch: bool) {
        // A deterministic pass that made no progress on this test case makes none again. Under a
        // nondeterministic test a rejected candidate may pass on a retry.
        let deterministic = !p.nondeterministic();
        let started = p.version();
        if deterministic && self.fingerprints.get(pass.name) == Some(&started) {
            return;
        }
        let use_budget = budgeted && self.probation.get(pass.name).copied().unwrap_or(false);
        p.control = Control {
            budget: use_budget.then_some(PROBATION_BUDGET),
            watch: watch.then(|| Watch::new(WATCH_TESTS, p)),
            ..Control::default()
        };
        (pass.run)(p, &mut self.rng);
        let control = std::mem::take(&mut p.control);
        let progress = control.made_progress || p.version() != started;
        if progress {
            self.log(p, pass.name);
        }
        // A pass the time limit stopped tells nothing about itself.
        if p.oracle.expired() {
            return;
        }
        if control.exhausted {
            self.incomplete.insert(pass.name, pass);
            if progress {
                self.probation.insert(pass.name, false);
            }
        } else if !control.cancelled {
            self.incomplete.remove(pass.name);
            self.probation.insert(pass.name, !progress);
            if progress || !deterministic {
                self.fingerprints.remove(pass.name);
            } else {
                self.fingerprints.insert(pass.name, p.version());
            }
        }
    }

    fn run_great_passes(&mut self, p: &mut Problem) {
        let all = great_passes();
        let mut current = all.clone();
        while !p.cancelled() {
            let prev = p.version();
            let mut successful = Vec::new();
            for pass in &current {
                let size = p.current().len();
                self.run_pass(p, *pass, true, false);
                if p.current().len() < size {
                    successful.push(*pass);
                }
            }
            if p.version() == prev {
                if current.len() == all.len() {
                    break;
                }
                current = all.clone();
            } else if successful.is_empty() {
                current = all.clone();
            } else {
                current = successful;
            }
        }
    }

    fn run_some_passes(&mut self, p: &mut Problem) {
        let prev = p.version();
        self.run_great_passes(p);
        if prev != p.version() && !self.unlocked_ok {
            return;
        }
        self.unlocked_ok = true;
        for pass in ok_passes() {
            self.run_pass(p, pass, true, false);
        }
        if prev != p.version() {
            return;
        }
        for pass in last_ditch_passes() {
            self.run_pass(p, pass, true, false);
        }
    }

    fn initial_cut(&mut self, p: &mut Problem) {
        while !p.oracle.expired() {
            let prev = p.current().len();
            for pass in initial_cuts() {
                self.run_pass(p, pass, true, true);
            }
            if p.current().len() as f64 >= 0.99 * prev as f64 {
                return;
            }
        }
    }

    fn reduce(&mut self, p: &mut Problem) {
        // An empty program shows nothing, but the check is one test.
        if p.is_interesting(String::new()) && p.current().is_empty() {
            return;
        }
        self.initial_cut(p);
        while !p.oracle.expired() {
            let prev = p.version();
            self.run_some_passes(p);
            if p.version() != prev {
                continue;
            }
            for pass in polish_passes() {
                self.run_pass(p, pass, true, false);
            }
            if p.version() != prev {
                continue;
            }
            if !self.incomplete.is_empty() {
                let incomplete: Vec<Pass> = self.incomplete.values().copied().collect();
                for pass in incomplete {
                    self.run_pass(p, pass, false, false);
                }
                if p.version() != prev {
                    continue;
                }
            }
            if p.oracle.expired() || !p.attempt_unstick() {
                break;
            }
            // Something changed, so a pass without progress here may now make some.
            self.fingerprints.clear();
        }
    }
}

const SEED: u64 = 0x7368_7269_6e6b;

pub struct Options<'a> {
    pub workers: usize,
    /// The most a test's process may take, in seconds.
    pub timeout_cap: f64,
    /// When the reduction stops with what it has.
    pub deadline: Option<Instant>,
    pub log: &'a dyn Fn(&str),
    /// Called with each improvement.
    pub save: &'a dyn Fn(&str),
}

/// Shrink `initial`, which `test` must find interesting; with the hit rate of the result's final
/// replays (none when the time limit stopped it).
pub fn reduce<'a>(
    initial: &str,
    test: &'a Test<'a>,
    options: &Options<'a>,
) -> Result<(String, nondet::Evidence), String> {
    let pool = problem::Pool::default();
    // The workers stop when the reduction does, panicking or not.
    struct Stop<'p>(&'p problem::Pool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.shutdown();
        }
    }
    std::thread::scope(|s| {
        for w in 0..options.workers.max(1) {
            let pool = &pool;
            s.spawn(move || pool.work(w, test));
        }
        let _stop = Stop(&pool);
        run(initial, &pool, options)
    })
}

fn run<'a>(
    initial: &str,
    pool: &'a problem::Pool,
    options: &Options<'a>,
) -> Result<(String, nondet::Evidence), String> {
    let mut oracle = Oracle::new(initial.to_string(), pool, options);
    oracle.setup()?;
    let rng = Rng::new(SEED);
    Reducer::new(rng.clone()).reduce(&mut Problem::new(&mut oracle));
    // Greedy reduction can corner itself: re-reduce from the original, constrained to results
    // below the fixpoint, while that improves it. Replaying the first run's random state makes
    // the restart retrace it from the cache up to its first improvement.
    while !oracle.expired() {
        let fixpoint = oracle.incumbent.clone();
        if fixpoint == initial {
            break;
        }
        oracle.log("restart from the original");
        oracle.timeout.reset();
        Reducer::new(rng.clone()).reduce(&mut Problem::restart(&mut oracle, initial.to_string()));
        if oracle.incumbent == fixpoint {
            break;
        }
    }
    if oracle.expired() {
        oracle.log("stopped at the time limit");
    }
    Ok((oracle.incumbent.clone(), oracle.final_evidence))
}
