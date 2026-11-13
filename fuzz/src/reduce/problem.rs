//! What a reduction tests candidates against: shrinkray's `BasicReductionProblem`. `Oracle` holds
//! what every view of a reduction shares (the test, its results by candidate, the
//! nondeterminism policy, the time limit, and the incumbent with its history); a `Problem` is
//! one view, the reduction itself or a restart from the original below the first fixpoint.
//! Candidates are tested in batches on worker threads, and a batch's results are taken in
//! candidate order, so a reduction does not depend on the number of workers or on timing.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::nondet::{
    self, ANCHOR_SEED_RUNS, DETECTION_REPLAYS, Evidence, GATE_RUNS, Policy, VERIFY_INTERVAL,
    Verdict,
};
use super::timeout::Timeout;

/// One run of the test.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub interesting: bool,
    pub timed_out: bool,
}

/// The test: whether `text` shows the finding, run as worker `w` with each process limited to
/// the duration.
pub type Test<'a> = dyn Fn(&str, usize, Duration) -> Outcome + Sync + 'a;

/// Tokens, then bytes, then the text: fewer tokens is smaller, comments and whitespace count
/// only as bytes.
pub fn sort_key(text: &str) -> (usize, usize) {
    (tcvm::parser::lex(text).len(), text.len())
}

pub fn less(a: &str, b: &str) -> bool {
    (sort_key(a), a) < (sort_key(b), b)
}

fn cache_key(text: &str) -> (usize, u64) {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    (text.len(), h.finish())
}

/// What is known about one candidate. Deterministic: one run decides. Otherwise evidence
/// accumulates over every run of it, retries by later passes included, and the verdict latches
/// once the gauntlet decides.
#[derive(Clone, Debug, Default)]
struct Ledger {
    evidence: Evidence,
    verdict: Option<bool>,
    /// A verdict from a timed-out run holds only while the limit is at most this.
    timed_out_at: Option<f64>,
    min_hits: Option<u32>,
    /// The evidence when the gauntlet accepted, so the runs after it can be told apart.
    accepted_at: Option<Evidence>,
}

impl Ledger {
    fn unselected(&self) -> Evidence {
        self.evidence
            .since(self.accepted_at.expect("an accepted ledger"))
    }
}

pub struct Oracle<'a> {
    test: &'a Test<'a>,
    pub workers: usize,
    ledgers: HashMap<(usize, u64), Ledger>,
    pub policy: Policy,
    pub timeout: Timeout,
    /// The reduction's current test case and every one it adopted, oldest first.
    pub incumbent: String,
    history: Vec<String>,
    pub calls: u64,
    calls_at_last_verify: u64,
    /// Evidence about the final result, for the report.
    pub final_evidence: Evidence,
    log: &'a dyn Fn(&str),
    /// Called with each new incumbent.
    save: &'a dyn Fn(&str),
    deadline: Option<Instant>,
}

/// Run `f(i, worker)` for `i` in `0..n` on `workers` threads; results in order of `i`.
fn parallel<T: Send>(workers: usize, n: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for w in 0..workers.min(n) {
            let (next, out, f) = (&next, &out, &f);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = f(i, w);
                    *out[i].lock().expect("a result") = Some(r);
                }
            });
        }
    });
    (out.into_iter())
        .map(|m| m.into_inner().expect("a result").expect("every job ran"))
        .collect()
}

/// One run with its wall time.
struct Timed {
    outcome: Outcome,
    secs: f64,
}

impl<'a> Oracle<'a> {
    pub fn new(initial: String, test: &'a Test<'a>, o: &super::Options<'a>) -> Oracle<'a> {
        Oracle {
            test,
            workers: o.workers.max(1),
            ledgers: HashMap::new(),
            policy: Policy::default(),
            timeout: Timeout::new(o.timeout_cap),
            history: vec![initial.clone()],
            incumbent: initial,
            calls: 0,
            calls_at_last_verify: 0,
            final_evidence: Evidence::default(),
            log: o.log,
            save: o.save,
            deadline: o.deadline,
        }
    }

    /// Past the reduction's time limit: everything stops at the end of its batch.
    pub fn expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    pub fn log(&self, s: &str) {
        (self.log)(s)
    }

    fn secs(&mut self) -> (Duration, f64) {
        let t = self.timeout.current();
        (Duration::from_secs_f64(t), t)
    }

    fn record(&mut self, runs: &[Timed], used: f64) {
        for r in runs {
            self.calls += 1;
            if r.outcome.timed_out {
                self.timeout.record_timeout(used);
            } else {
                self.timeout
                    .record_completion(r.secs, r.outcome.interesting);
            }
        }
    }

    /// Run `texts` once each, in parallel.
    fn run_all(&mut self, texts: &[&str]) -> Vec<Outcome> {
        let (secs, used) = self.secs();
        let test = self.test;
        let runs = parallel(self.workers, texts.len(), |i, w| {
            let start = Instant::now();
            let outcome = test(texts[i], w, secs);
            Timed {
                outcome,
                secs: start.elapsed().as_secs_f64(),
            }
        });
        self.record(&runs, used);
        runs.iter().map(|r| r.outcome).collect()
    }

    fn run_one(&mut self, text: &str) -> Outcome {
        self.run_all(&[text])[0]
    }

    /// Replay the incumbent `runs` times in parallel into `evidence`, timeouts excluded (they say
    /// nothing about determinism); whether every completed run reproduced.
    fn replays_all_reproduce(&mut self, evidence: &mut Evidence, runs: u32) -> bool {
        let text = self.incumbent.clone();
        let texts = vec![text.as_str(); runs as usize];
        let mut all = true;
        for o in self.run_all(&texts) {
            if !o.timed_out {
                evidence.record(o.interesting);
                all &= o.interesting;
            }
        }
        self.policy.replay_calls += runs as u64;
        all
    }

    /// Check that the initial test case shows the finding, looking for nondeterminism: its first
    /// run is evidence like any replay, and one that has not reproduced yet is invalid only once
    /// it missed `GATE_RUNS` times.
    pub fn setup(&mut self) -> Result<(), String> {
        let mut evidence = Evidence::default();
        let all = self.replays_all_reproduce(&mut evidence, DETECTION_REPLAYS);
        let mut attempts = DETECTION_REPLAYS;
        while evidence.interesting == 0 && attempts < GATE_RUNS {
            attempts += 1;
            self.replays_all_reproduce(&mut evidence, 1);
        }
        if evidence.interesting == 0 {
            return Err("the program does not show the finding".into());
        }
        if !all {
            self.flip(evidence);
        }
        Ok(())
    }

    fn flip(&mut self, evidence: Evidence) {
        if self.policy.active {
            return;
        }
        self.policy.active = true;
        self.log(&format!(
            "nondeterministic: the incumbent reproduced on {} of {} replays; candidates are \
             confirmed by repeated runs from now on",
            evidence.interesting, evidence.runs
        ));
        // Every run so far was a sample, not a verdict.
        for l in self.ledgers.values_mut() {
            l.verdict = None;
            l.timed_out_at = None;
        }
        self.policy.confirming = false;
        self.confirm_or_backtrack(evidence);
    }

    /// The incumbent was adopted on single runs: confirm it, or back up through the history to
    /// the newest test case that does reproduce.
    fn confirm_or_backtrack(&mut self, mut evidence: Evidence) {
        let text = self.incumbent.clone();
        if self.confirmation_batch(&text, &mut evidence) {
            return;
        }
        match self.scan_history(Self::confirmation_batch) {
            None => self.log(&format!(
                "the incumbent reproduces rarely ({} of {} replays) and no earlier test case did \
                 better; going on from it",
                evidence.interesting, evidence.runs
            )),
            Some((text, e)) => {
                self.log(&format!(
                    "backtracking to an earlier test case that reproduces ({} of {} replays)",
                    e.interesting, e.runs
                ));
                self.set_incumbent(text);
            }
        }
    }

    /// Drive `evidence` about `text` to a confirmation verdict; an accept is extended to the
    /// anchor seed size, and the extension offered to the anchor.
    fn confirmation_batch(&mut self, text: &str, evidence: &mut Evidence) -> bool {
        loop {
            match nondet::confirmation_bar(*evidence) {
                Verdict::Reject => return false,
                Verdict::Accept => {
                    let at = *evidence;
                    while evidence.runs < ANCHOR_SEED_RUNS {
                        evidence.record(self.run_one(text).interesting);
                    }
                    self.policy.replay_calls += (evidence.runs - at.runs) as u64;
                    self.policy.raise_anchor(evidence.since(at));
                    return true;
                }
                Verdict::Continue => {
                    evidence.record(self.run_one(text).interesting);
                    self.policy.replay_calls += 1;
                }
            }
        }
    }

    /// Drive `evidence` about `text` to a gauntlet verdict at the hit minimum in force.
    fn gauntlet_batch(&mut self, text: &str, evidence: &mut Evidence) -> bool {
        loop {
            let p = &self.policy;
            match nondet::gauntlet(*evidence, p.anchor, p.min_hits, p.reject_bar()) {
                Verdict::Continue => {
                    evidence.record(self.run_one(text).interesting);
                    self.policy.replay_calls += 1;
                }
                v => return v == Verdict::Accept,
            }
        }
    }

    /// The newest adopted test case `judge` accepts: newest first at growing distances, then
    /// bisection between the last reject and the accept, biased to older (larger) entries.
    fn scan_history(
        &mut self,
        judge: fn(&mut Self, &str, &mut Evidence) -> bool,
    ) -> Option<(String, Evidence)> {
        let positions: Vec<usize> = (0..self.history.len())
            .rev()
            .filter(|&i| self.history[i] != self.incumbent)
            .collect();
        if positions.is_empty() {
            return None;
        }
        let test = |o: &mut Self, k: usize| {
            let text = o.history[positions[k]].clone();
            let mut e = Evidence::default();
            judge(o, &text, &mut e).then_some((text, e))
        };
        let mut rejected: isize = -1;
        let mut accepted = None;
        let mut k = 0;
        while k < positions.len() {
            if let Some(r) = test(self, k) {
                accepted = Some((k, r));
                break;
            }
            rejected = k as isize;
            k = 2 * k + 1;
        }
        let mut accepted = match accepted {
            Some(a) => a,
            None => {
                let oldest = positions.len() - 1;
                if oldest as isize <= rejected {
                    return None;
                }
                (oldest, test(self, oldest)?)
            }
        };
        while accepted.0 as isize - rejected > 1 {
            let mid = ((accepted.0 as isize + rejected) / 2) as usize;
            match test(self, mid) {
                Some(r) => accepted = (mid, r),
                None => rejected = mid as isize,
            }
        }
        Some(accepted.1)
    }

    /// Adopt `text` by backtracking rather than by reduction.
    fn set_incumbent(&mut self, text: String) {
        self.timeout.note_reduction();
        self.history.push(text.clone());
        self.incumbent = text;
        (self.save)(&self.incumbent);
    }

    /// Periodically replay the incumbent: while it looks deterministic this catches a test that
    /// reproduces most of the time; under handling it feeds the incumbent monitor.
    fn maybe_verify(&mut self) {
        if self.calls - self.calls_at_last_verify < VERIFY_INTERVAL {
            return;
        }
        self.calls_at_last_verify = self.calls;
        let mut evidence = Evidence::default();
        if !self.policy.active {
            if !self.replays_all_reproduce(&mut evidence, 1) {
                self.flip(evidence);
            }
            return;
        }
        self.replays_all_reproduce(&mut evidence, 1);
        if evidence.runs > 0 {
            self.policy.record_incumbent_run(evidence.interesting == 1);
        }
        if self.policy.incumbent_failing() {
            self.recover();
        }
    }

    /// The incumbent reproduces less often than the gauntlet required: a false accept. Demand
    /// more of later candidates and back up to the newest test case that clears the gauntlet.
    fn recover(&mut self) {
        self.policy.record_false_accept();
        let i = self.policy.incumbent;
        self.log(&format!(
            "the incumbent reproduces less often than expected ({} of {} replays); candidates \
             now need {} interesting runs",
            i.interesting, i.runs, self.policy.min_hits
        ));
        match self.scan_history(Self::gauntlet_batch) {
            None => self.log("no earlier test case reproduces better; going on"),
            Some((text, e)) => {
                self.log(&format!(
                    "backtracking to an earlier test case that reproduces ({} of {} replays)",
                    e.interesting, e.runs
                ));
                self.set_incumbent(text);
            }
        }
        self.policy.adopt(Evidence::default());
    }

    /// A candidate's runs under nondeterministic handling: the first run recruits it (in a fast
    /// sweep a miss rejects it), a hit starts the gauntlet, and an accept tops the ledger up
    /// while the extra runs could still raise the anchor.
    fn gauntlet_runs(
        test: &Test,
        policy: &Policy,
        ledger: &mut Ledger,
        text: &str,
        w: usize,
        secs: Duration,
        runs: &mut Vec<Timed>,
    ) -> bool {
        let min_hits = *ledger.min_hits.get_or_insert(policy.min_hits);
        let run = |runs: &mut Vec<Timed>| {
            let start = Instant::now();
            let outcome = test(text, w, secs);
            runs.push(Timed {
                outcome,
                secs: start.elapsed().as_secs_f64(),
            });
            outcome.interesting
        };
        let hit = run(runs);
        ledger.evidence.record(hit);
        if !hit && !policy.confirming {
            return false;
        }
        loop {
            match nondet::gauntlet(
                ledger.evidence,
                policy.anchor,
                min_hits,
                policy.reject_bar(),
            ) {
                Verdict::Reject => {
                    ledger.verdict = Some(false);
                    return false;
                }
                Verdict::Accept => {
                    ledger.accepted_at = Some(ledger.evidence);
                    while ledger.evidence.runs < ANCHOR_SEED_RUNS
                        && policy.raise_reachable(
                            ledger.unselected(),
                            ANCHOR_SEED_RUNS - ledger.evidence.runs,
                        )
                    {
                        let hit = run(runs);
                        ledger.evidence.record(hit);
                    }
                    ledger.verdict = Some(true);
                    return true;
                }
                Verdict::Continue => {
                    let hit = run(runs);
                    ledger.evidence.record(hit);
                }
            }
        }
    }

    fn latched(&self, l: &Ledger) -> Option<bool> {
        match l.timed_out_at {
            Some(t) if self.timeout.peek() > t => None,
            _ => l.verdict,
        }
    }

    /// Whether each of `texts` is interesting, tested in parallel. `filter` answers some without
    /// a run (`Some`); `adopt` is called in order for each interesting one.
    fn interesting(
        &mut self,
        texts: &[String],
        filter: impl Fn(&str) -> Option<bool>,
        mut adopt: impl FnMut(&mut Self, &str, Option<Evidence>),
    ) -> (Vec<bool>, usize) {
        self.maybe_verify();
        let answered: Vec<bool> = texts.iter().map(|t| filter(t).is_some()).collect();
        let mut out: Vec<Option<bool>> = texts.iter().map(|t| filter(t)).collect();
        for (i, t) in texts.iter().enumerate() {
            if out[i].is_none()
                && let Some(l) = self.ledgers.get(&cache_key(t))
            {
                out[i] = self.latched(l);
            }
        }
        // Duplicates in one batch run once.
        let mut jobs: Vec<usize> = Vec::new();
        let mut seen = HashMap::new();
        let mut job_of = vec![usize::MAX; texts.len()];
        for i in 0..texts.len() {
            if out[i].is_none() {
                job_of[i] = *seen.entry(cache_key(&texts[i])).or_insert_with(|| {
                    jobs.push(i);
                    jobs.len() - 1
                });
            }
        }
        let (secs, used) = self.secs();
        let ledgers: Vec<Ledger> = (jobs.iter())
            .map(|&i| {
                self.ledgers
                    .get(&cache_key(&texts[i]))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        let (test, policy) = (self.test, self.policy.clone());
        let results = parallel(self.workers, jobs.len(), |j, w| {
            let text = &texts[jobs[j]];
            let mut ledger = ledgers[j].clone();
            let mut runs = Vec::new();
            let r = if policy.active {
                Self::gauntlet_runs(test, &policy, &mut ledger, text, w, secs, &mut runs)
            } else {
                let start = Instant::now();
                let outcome = test(text, w, secs);
                runs.push(Timed {
                    outcome,
                    secs: start.elapsed().as_secs_f64(),
                });
                ledger.evidence.record(outcome.interesting);
                ledger.verdict = Some(outcome.interesting);
                ledger.timed_out_at = outcome.timed_out.then_some(used);
                outcome.interesting
            };
            (r, ledger, runs)
        });
        let tested = jobs.len();
        let mut verdicts = Vec::with_capacity(jobs.len());
        for (j, (r, ledger, runs)) in results.into_iter().enumerate() {
            self.record(&runs, used);
            self.policy.replay_calls += runs.len().saturating_sub(1) as u64;
            self.ledgers.insert(cache_key(&texts[jobs[j]]), ledger);
            verdicts.push(r);
        }
        let mut result = Vec::with_capacity(texts.len());
        for i in 0..texts.len() {
            let r = out[i].unwrap_or_else(|| verdicts[job_of[i]]);
            if r && !answered[i] {
                let unselected = (self.policy.active)
                    .then(|| self.ledgers.get(&cache_key(&texts[i])))
                    .flatten()
                    .and_then(|l| l.accepted_at.map(|_| l.unselected()));
                adopt(self, &texts[i], unselected);
            }
            result.push(r);
        }
        (result, tested)
    }

    /// Adopt `text` into the incumbent if it is smaller.
    fn improve(&mut self, text: &str, unselected: Option<Evidence>) -> bool {
        if !less(text, &self.incumbent) {
            return false;
        }
        self.timeout.note_reduction();
        self.incumbent = text.to_string();
        self.history.push(self.incumbent.clone());
        (self.save)(&self.incumbent);
        if self.policy.active {
            // The one validated event that may move the anchor; it also ends a confirmation
            // sweep, whose fixpoint is gone.
            let e = unselected.unwrap_or_default();
            self.policy.raise_anchor(e);
            self.policy.adopt(e);
            self.policy.confirming = false;
        }
        true
    }

    /// The reducer found nothing more to do: whether another round may make progress.
    pub fn attempt_unstick(&mut self) -> bool {
        if !self.policy.active {
            // The final check for a test that looked deterministic all the way through.
            let mut e = Evidence::default();
            let all = self.replays_all_reproduce(&mut e, DETECTION_REPLAYS);
            self.final_evidence = e;
            if !all {
                self.flip(e);
                return true;
            }
        } else if !self.policy.confirming {
            // A fixpoint reached with fast sweeps is no certificate.
            self.policy.confirming = true;
            self.log("confirmation sweep");
            return true;
        } else {
            self.policy.confirming = false;
            let mut e = Evidence::default();
            let text = self.incumbent.clone();
            for o in self.run_all(&vec![text.as_str(); ANCHOR_SEED_RUNS as usize]) {
                e.record(o.interesting);
            }
            self.final_evidence = e;
            for _ in 0..e.interesting {
                self.policy.record_incumbent_run(true);
            }
            for _ in 0..e.runs - e.interesting {
                self.policy.record_incumbent_run(false);
            }
            if self.policy.incumbent_failing() {
                self.recover();
                return true;
            }
        }
        self.timeout.attempt_unstick()
    }
}

/// A view of the reduction: the reduction itself (`restart` none), whose current test case is
/// the incumbent, or a restart from the original, which counts as interesting, below the first
/// fixpoint.
pub struct Problem<'o, 'a> {
    pub oracle: &'o mut Oracle<'a>,
    restart: Option<Restart>,
    pub control: Control,
}

/// The running pass's monitor: the probation budget of consecutive tests without progress, and
/// the initial cuts' watcher, which stops a pass once a tick of tests made no progress or the
/// pass's rate fell under half its best.
#[derive(Default)]
pub struct Control {
    pub budget: Option<u64>,
    pub consecutive: u64,
    pub exhausted: bool,
    pub made_progress: bool,
    pub watch: Option<Watch>,
    pub cancelled: bool,
}

pub struct Watch {
    pub tick: u64,
    since: u64,
    initial_size: usize,
    iters: u64,
    best: f64,
    at_tick: String,
}

impl Watch {
    pub fn new(tick: u64, current: &str) -> Watch {
        Watch {
            tick,
            since: 0,
            initial_size: current.len(),
            iters: 0,
            best: 0.0,
            at_tick: current.to_string(),
        }
    }
}

impl Control {
    fn observe(&mut self, tested: usize, changed: bool, current: &str) {
        if changed {
            self.consecutive = 0;
            self.made_progress = true;
        } else {
            self.consecutive += tested as u64;
            if self.budget.is_some_and(|b| self.consecutive > b) {
                self.exhausted = true;
                self.cancelled = true;
            }
        }
        if let Some(w) = &mut self.watch {
            w.since += tested as u64;
            while w.since >= w.tick {
                w.since -= w.tick;
                w.iters += 1;
                let rate = w.initial_size.saturating_sub(current.len()) as f64 / w.iters as f64;
                w.best = w.best.max(rate);
                if rate < 0.5 * w.best || current == w.at_tick {
                    self.cancelled = true;
                }
                w.at_tick = current.to_string();
            }
        }
    }
}

struct Restart {
    current: String,
    original: String,
    fixpoint: String,
}

impl<'o, 'a> Problem<'o, 'a> {
    pub fn new(oracle: &'o mut Oracle<'a>) -> Problem<'o, 'a> {
        Problem {
            oracle,
            restart: None,
            control: Control::default(),
        }
    }

    pub fn restart(oracle: &'o mut Oracle<'a>, original: String) -> Problem<'o, 'a> {
        let fixpoint = oracle.incumbent.clone();
        Problem {
            oracle,
            restart: Some(Restart {
                current: original.clone(),
                original,
                fixpoint,
            }),
            control: Control::default(),
        }
    }

    pub fn current(&self) -> &str {
        match &self.restart {
            Some(r) => &r.current,
            None => &self.oracle.incumbent,
        }
    }

    pub fn nondeterministic(&self) -> bool {
        self.oracle.policy.active
    }

    pub fn workers(&self) -> usize {
        self.oracle.workers
    }

    /// Whether each of `texts` is interesting, adopting in order each one smaller than the
    /// current test case; with the number of tests run.
    pub fn test(&mut self, texts: &[String]) -> (Vec<bool>, usize) {
        let current = self.current().to_string();
        let restart = self
            .restart
            .as_ref()
            .map(|r| (r.original.clone(), r.fixpoint.clone()));
        let filter = |t: &str| {
            if t == current {
                return Some(true);
            }
            match &restart {
                Some((original, _)) if t == original => Some(true),
                Some((_, fixpoint)) if !less(t, fixpoint) => Some(false),
                _ => None,
            }
        };
        let mut view = self.restart.as_mut().map(|r| &mut r.current);
        let (r, tested) = self.oracle.interesting(texts, filter, |o, t, unselected| {
            o.improve(t, unselected);
            if let Some(c) = view.as_deref_mut()
                && less(t, c)
            {
                *c = t.to_string();
            }
        });
        let now = self.current().to_string();
        self.control.observe(tested, now != current, &now);
        (r, tested)
    }

    pub fn cancelled(&self) -> bool {
        self.control.cancelled || self.oracle.expired()
    }

    pub fn is_interesting(&mut self, text: String) -> bool {
        self.test(&[text]).0[0]
    }

    pub fn attempt_unstick(&mut self) -> bool {
        // A restart reports to the reduction, which does its own unsticking.
        self.restart.is_none() && self.oracle.attempt_unstick()
    }
}
