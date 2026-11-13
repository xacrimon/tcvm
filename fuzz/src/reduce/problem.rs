//! What a reduction tests candidates against: shrinkray's `BasicReductionProblem`. `Oracle` holds
//! what every view of a reduction shares (the test, its results by candidate, the
//! nondeterminism policy, the time limit, and the incumbent with its history); a `Problem` is
//! one view, the reduction itself or a restart from the original below the first fixpoint.
//! Candidates are tested in batches on a pool of worker threads, and a batch's results are taken
//! in candidate order, so a reduction does not depend on the number of workers or on timing. A
//! batch can be submitted ahead of the one before it finishing (speculatively, behind it in the
//! queue) and cancelled, its processes killed, when it turns out to be built on a stale state.

use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use super::nondet::{
    self, ANCHOR_SEED_RUNS, DETECTION_REPLAYS, Evidence, GATE_RUNS, Policy, VERIFY_INTERVAL,
    Verdict,
};
use super::timeout::Timeout;
pub use crate::exec::Cancel;
use crate::par;

/// One run of the test.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub interesting: bool,
    pub timed_out: bool,
}

/// The test: whether `text` shows the finding, run as worker `w` with each process limited to
/// the duration and stopped by the `Cancel`.
pub type Test<'a> = dyn Fn(&str, usize, Duration, &Cancel) -> Outcome + Sync + 'a;

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
    pool: &'a Pool,
    pub workers: usize,
    /// Batches submitted, for their place in the queue.
    seq: u64,
    /// Bumped when results of earlier batches stop meaning what they did (the policy changed).
    epoch: u64,
    ledgers: HashMap<(usize, u64), Ledger>,
    pub policy: Policy,
    pub timeout: Timeout,
    /// The reduction's current test case and every one it adopted, oldest first.
    pub incumbent: String,
    history: Vec<String>,
    /// Counts the changes of `incumbent`.
    version: u64,
    pub calls: u64,
    calls_at_last_verify: u64,
    /// Evidence about the final result, for the report.
    pub final_evidence: Evidence,
    log: &'a dyn Fn(&str),
    /// Called with each new incumbent.
    save: &'a dyn Fn(&str),
    deadline: Option<Instant>,
}

/// One run with its wall time.
struct Timed {
    outcome: Outcome,
    secs: f64,
}

enum Work {
    /// One run, as a replay.
    Plain,
    /// A candidate's runs: one, or the gauntlet under the policy.
    Eval {
        ledger: Ledger,
        policy: Option<Arc<Policy>>,
    },
}

struct Job {
    text: Arc<String>,
    work: Work,
    limit: Duration,
    /// The limit as the time policy chose it, for cached timeouts.
    used: f64,
    cancel: Arc<Cancel>,
    index: usize,
    reply: mpsc::Sender<Done>,
}

struct Done {
    index: usize,
    verdict: bool,
    ledger: Ledger,
    runs: Vec<Timed>,
}

/// The workers' queue: synchronous batches first, then speculative ones, each in the order they
/// were submitted.
#[derive(Default)]
pub struct Pool {
    state: Mutex<Queue>,
    ready: Condvar,
}

#[derive(Default)]
struct Queue {
    /// By (speculative, batch, index).
    jobs: BTreeMap<(bool, u64, usize), Job>,
    shutdown: bool,
}

impl Pool {
    fn push(&self, speculative: bool, seq: u64, jobs: Vec<Job>) {
        let mut st = self.state.lock().expect("the queue");
        for j in jobs {
            st.jobs.insert((speculative, seq, j.index), j);
        }
        self.ready.notify_all();
    }

    pub fn shutdown(&self) {
        self.state.lock().expect("the queue").shutdown = true;
        self.ready.notify_all();
    }

    /// A worker's loop, until `shutdown`.
    pub fn work(&self, w: usize, test: &Test) {
        loop {
            let job = {
                let mut st = self.state.lock().expect("the queue");
                loop {
                    if st.shutdown {
                        return;
                    }
                    if let Some((_, j)) = st.jobs.pop_first() {
                        break j;
                    }
                    st = self.ready.wait(st).expect("the queue");
                }
            };
            if job.cancel.cancelled() {
                continue;
            }
            let mut runs = Vec::new();
            let run = |runs: &mut Vec<Timed>| {
                let start = Instant::now();
                let outcome = test(&job.text, w, job.limit, &job.cancel);
                runs.push(Timed {
                    outcome,
                    secs: start.elapsed().as_secs_f64(),
                });
                outcome
            };
            let (verdict, ledger) = match job.work {
                Work::Plain => (run(&mut runs).interesting, Ledger::default()),
                Work::Eval {
                    mut ledger,
                    policy: Some(policy),
                } => {
                    let r = Oracle::gauntlet_runs(&policy, &mut ledger, &mut runs, |r| {
                        run(r).interesting
                    });
                    (r, ledger)
                }
                Work::Eval {
                    mut ledger,
                    policy: None,
                } => {
                    let o = run(&mut runs);
                    ledger.evidence.record(o.interesting);
                    ledger.verdict = Some(o.interesting);
                    ledger.timed_out_at = o.timed_out.then_some(job.used);
                    (o.interesting, ledger)
                }
            };
            if job.cancel.cancelled() {
                continue;
            }
            // A cancelled batch's receiver may be gone.
            let _ = job.reply.send(Done {
                index: job.index,
                verdict,
                ledger,
                runs,
            });
        }
    }
}

/// How a candidate of a batch gets its answer.
enum Answer {
    /// Known to the submitter without a run; never adopted.
    Given(bool),
    /// The latched verdict of an earlier run.
    Cached(bool),
    /// The verdict of the batch's job of this index.
    Job(usize),
}

struct Candidate {
    text: String,
    key: (usize, u64),
    answer: Answer,
}

/// A submitted batch.
pub struct Ticket {
    candidates: Vec<Candidate>,
    /// The cache key of each job's candidate.
    jobs: Vec<(usize, u64)>,
    rx: mpsc::Receiver<Done>,
    cancel: Arc<Cancel>,
    used: f64,
    epoch: u64,
}

impl Ticket {
    /// Kill the batch's runs; workers skip its queued jobs when they come to them.
    fn cancel(self) {
        self.cancel.cancel();
    }
}

impl<'a> Oracle<'a> {
    pub fn new(initial: String, pool: &'a Pool, o: &super::Options<'a>) -> Oracle<'a> {
        Oracle {
            pool,
            workers: o.workers.max(1),
            seq: 0,
            epoch: 0,
            ledgers: HashMap::new(),
            policy: Policy::default(),
            timeout: Timeout::new(o.timeout_cap),
            history: vec![initial.clone()],
            incumbent: initial,
            version: 0,
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

    /// Run `texts` once each, in parallel, ahead of every queued batch.
    fn run_all(&mut self, texts: &[&str]) -> Vec<Outcome> {
        let (limit, used) = self.secs();
        let (reply, rx) = mpsc::channel();
        let cancel = Arc::new(Cancel::default());
        let jobs = (texts.iter().enumerate())
            .map(|(index, t)| Job {
                text: Arc::new(t.to_string()),
                work: Work::Plain,
                limit,
                used,
                cancel: cancel.clone(),
                index,
                reply: reply.clone(),
            })
            .collect();
        self.seq += 1;
        self.pool.push(false, self.seq, jobs);
        let mut runs: Vec<Option<Timed>> = (0..texts.len()).map(|_| None).collect();
        for _ in 0..texts.len() {
            let mut d = rx.recv().expect("a worker's result");
            runs[d.index] = d.runs.pop();
        }
        let runs: Vec<Timed> = runs.into_iter().map(|r| r.expect("every run")).collect();
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
        self.epoch += 1;
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
        self.version += 1;
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
        self.epoch += 1;
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
        policy: &Policy,
        ledger: &mut Ledger,
        runs: &mut Vec<Timed>,
        mut run: impl FnMut(&mut Vec<Timed>) -> bool,
    ) -> bool {
        let min_hits = *ledger.min_hits.get_or_insert(policy.min_hits);
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

    /// Submit `texts` for testing; `answers` holds those known without a run.
    fn submit(
        &mut self,
        texts: Vec<String>,
        answers: Vec<Option<bool>>,
        speculative: bool,
    ) -> Ticket {
        let keys = par::map(self.workers, texts.len(), |i| match answers[i] {
            Some(_) => (0, 0),
            None => cache_key(&texts[i]),
        });
        let (limit, used) = self.secs();
        let policy = self.policy.active.then(|| Arc::new(self.policy.clone()));
        let (reply, rx) = mpsc::channel();
        let cancel = Arc::new(Cancel::default());
        let mut queued: Vec<Job> = Vec::new();
        let mut jobs = Vec::new();
        // Duplicates in one batch run once.
        let mut seen = HashMap::new();
        let candidates = (texts.into_iter().zip(answers).zip(keys))
            .map(|((text, given), key)| {
                let ledger = self.ledgers.get(&key).filter(|_| given.is_none());
                let answer = match (given, ledger.and_then(|l| self.latched(l))) {
                    (Some(r), _) => Answer::Given(r),
                    (None, Some(r)) => Answer::Cached(r),
                    (None, None) => Answer::Job(*seen.entry(key).or_insert_with(|| {
                        jobs.push(key);
                        queued.push(Job {
                            text: Arc::new(text.clone()),
                            work: Work::Eval {
                                ledger: ledger.cloned().unwrap_or_default(),
                                policy: policy.clone(),
                            },
                            limit,
                            used,
                            cancel: cancel.clone(),
                            index: queued.len(),
                            reply: reply.clone(),
                        });
                        queued.len() - 1
                    })),
                };
                Candidate { text, key, answer }
            })
            .collect();
        self.seq += 1;
        self.pool.push(speculative, self.seq, queued);
        Ticket {
            candidates,
            jobs,
            rx,
            cancel,
            used,
            epoch: self.epoch,
        }
    }

    /// Wait for a batch and take its results in order, calling `adopt` for each interesting one;
    /// `None` when the policy changed since it was submitted.
    fn finish(
        &mut self,
        t: Ticket,
        mut adopt: impl FnMut(&mut Self, &str, Option<Evidence>),
    ) -> Option<(Vec<bool>, usize)> {
        self.maybe_verify();
        if t.epoch != self.epoch {
            t.cancel();
            return None;
        }
        let mut done: Vec<Option<Done>> = (0..t.jobs.len()).map(|_| None).collect();
        for _ in 0..t.jobs.len() {
            let d = t.rx.recv().expect("a worker's result");
            let i = d.index;
            done[i] = Some(d);
        }
        let tested = t.jobs.len();
        let mut verdicts = Vec::with_capacity(tested);
        for (key, d) in t.jobs.iter().zip(done) {
            let d = d.expect("every job ran");
            self.record(&d.runs, t.used);
            self.policy.replay_calls += d.runs.len().saturating_sub(1) as u64;
            self.ledgers.insert(*key, d.ledger);
            verdicts.push(d.verdict);
        }
        let mut result = Vec::with_capacity(t.candidates.len());
        for c in &t.candidates {
            let (r, tested) = match c.answer {
                Answer::Given(r) => (r, false),
                Answer::Cached(r) => (r, true),
                Answer::Job(j) => (verdicts[j], true),
            };
            if r && tested {
                let unselected = (self.policy.active)
                    .then(|| self.ledgers.get(&c.key))
                    .flatten()
                    .and_then(|l| l.accepted_at.map(|_| l.unselected()));
                adopt(self, &c.text, unselected);
            }
            result.push(r);
        }
        Some((result, tested))
    }

    /// Adopt `text` into the incumbent if it is smaller.
    fn improve(&mut self, text: &str, unselected: Option<Evidence>) -> bool {
        if !less(text, &self.incumbent) {
            return false;
        }
        self.timeout.note_reduction();
        self.incumbent = text.to_string();
        self.version += 1;
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
            self.epoch += 1;
            self.log("confirmation sweep");
            return true;
        } else {
            self.policy.confirming = false;
            self.epoch += 1;
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
    /// The test case's version at the last tick.
    at_tick: u64,
}

impl Watch {
    pub fn new(tick: u64, p: &Problem) -> Watch {
        Watch {
            tick,
            since: 0,
            initial_size: p.current().len(),
            iters: 0,
            best: 0.0,
            at_tick: p.version(),
        }
    }
}

impl Control {
    /// `tested` more tests ran, which `changed` the test case or not; it is now of `size`
    /// bytes, at `version`.
    fn observe(&mut self, tested: usize, changed: bool, size: usize, version: u64) {
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
                let rate = w.initial_size.saturating_sub(size) as f64 / w.iters as f64;
                w.best = w.best.max(rate);
                if rate < 0.5 * w.best || version == w.at_tick {
                    self.cancelled = true;
                }
                w.at_tick = version;
            }
        }
    }
}

struct Restart {
    current: String,
    /// Counts the changes of `current`.
    version: u64,
    original: String,
    fixpoint: String,
    /// The fixpoint's `sort_key`.
    key: (usize, usize),
}

/// A batch a view submitted, with the version of the test case it was built against.
pub struct Pending {
    ticket: Ticket,
    version: u64,
    rejected: usize,
}

/// The candidates a view answers without a run: the current test case, and in a restart the
/// original (interesting) and anything not below the fixpoint (not).
pub struct Filter<'p> {
    current: &'p str,
    restart: Option<&'p Restart>,
}

impl Filter<'_> {
    pub fn check(&self, t: &str) -> Option<bool> {
        if t == self.current {
            return Some(true);
        }
        match self.restart {
            Some(r) if t == r.original => Some(true),
            Some(r) if (sort_key(t), t) >= (r.key, r.fixpoint.as_str()) => Some(false),
            _ => None,
        }
    }
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
                version: 0,
                original,
                key: sort_key(&fixpoint),
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

    /// Changes whenever `current` does.
    pub fn version(&self) -> u64 {
        match &self.restart {
            Some(r) => r.version,
            None => self.oracle.version,
        }
    }

    pub fn nondeterministic(&self) -> bool {
        self.oracle.policy.active
    }

    pub fn workers(&self) -> usize {
        self.oracle.workers
    }

    /// What answers a candidate without a run.
    pub fn filter(&self) -> Filter<'_> {
        Filter {
            current: self.current(),
            restart: self.restart.as_ref(),
        }
    }

    fn observe(&mut self, tested: usize, changed: bool) {
        let (size, version) = (self.current().len(), self.version());
        self.control.observe(tested, changed, size, version);
    }

    /// Candidates the filter rejected before they were tested count against a pass's budget.
    pub fn note_rejected(&mut self, n: usize) {
        if n > 0 {
            self.observe(n, false);
        }
    }

    /// Submit `texts` for testing; `speculative` ones queue behind the others.
    pub fn submit(&mut self, texts: Vec<String>, speculative: bool) -> Pending {
        let filter = self.filter();
        let answers = par::map(self.oracle.workers, texts.len(), |i| {
            filter.check(&texts[i])
        });
        let rejected = answers.iter().filter(|a| **a == Some(false)).count();
        Pending {
            version: self.version(),
            ticket: self.oracle.submit(texts, answers, speculative),
            rejected,
        }
    }

    pub fn cancel(&mut self, p: Pending) {
        p.ticket.cancel();
    }

    /// Whether each submitted text is interesting, adopting in order each one smaller than the
    /// current test case; with the number of tests run. `None` when the current test case or
    /// the policy changed since the batch was submitted.
    pub fn finish(&mut self, p: Pending) -> Option<(Vec<bool>, usize)> {
        if p.version != self.version() {
            self.cancel(p);
            return None;
        }
        let mut view = self.restart.as_mut();
        let (r, tested) = self.oracle.finish(p.ticket, |o, t, unselected| {
            o.improve(t, unselected);
            if let Some(view) = view.as_deref_mut()
                && less(t, &view.current)
            {
                view.current = t.to_string();
                view.version += 1;
            }
        })?;
        self.observe(tested + p.rejected, self.version() != p.version);
        Some((r, tested))
    }

    pub fn test(&mut self, texts: &[String]) -> (Vec<bool>, usize) {
        loop {
            let p = self.submit(texts.to_vec(), false);
            if let Some(r) = self.finish(p) {
                return r;
            }
        }
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
