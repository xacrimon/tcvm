//! The test's time limit, adapted to measured run times: shrinkray's `adaptive_timeout.py`. The
//! base is the larger of 10× the p90 of recent run times and 5× the slowest recent interesting
//! run; when reduction stalls with timeouts, it doubles per level to see whether they hid
//! progress, and drops back when that does not pay off.

use std::collections::VecDeque;
use std::time::Instant;

// Shrinkray: 1 second; tests here take milliseconds, and a batch waits for its slowest.
const MIN_TIMEOUT: f64 = 0.1;
const RUNTIME_MULTIPLIER: f64 = 10.0;
const INTERESTING_MULTIPLIER: f64 = 5.0;
const RUNTIME_QUANTILE: f64 = 0.9;
const RUNTIME_WINDOW: usize = 200;
const INTERESTING_WINDOW: usize = 20;
const OUTCOME_WINDOW: usize = 100;
const MIN_OUTCOMES_FOR_EXPLORATION: usize = 20;
const TIMEOUT_RATE_THRESHOLD: f64 = 0.05;
const STALL_MIN_SECONDS: f64 = 20.0;
const STALL_TIMEOUT_MULTIPLIER: f64 = 3.0;
const RUNG_RUN_BUDGET: u32 = 50;

pub struct Timeout {
    cap: f64,
    runtimes: VecDeque<f64>,
    interesting_runtimes: VecDeque<f64>,
    timed_out: VecDeque<bool>,
    level: i32,
    runs_at_level: u32,
    timeouts_at_level: u32,
    /// Exploration was tried and did not help; cleared by the next reduction.
    exhausted: bool,
    timeouts_below_cap: u32,
    /// `attempt_unstick` raised the timeout and no reduction followed yet.
    raised_without_reduction: bool,
    last_progress: Instant,
}

fn push<T>(q: &mut VecDeque<T>, v: T, cap: usize) {
    if q.len() == cap {
        q.pop_front();
    }
    q.push_back(v);
}

impl Timeout {
    pub fn new(cap: f64) -> Timeout {
        Timeout {
            cap,
            runtimes: VecDeque::new(),
            interesting_runtimes: VecDeque::new(),
            timed_out: VecDeque::new(),
            level: 0,
            runs_at_level: 0,
            timeouts_at_level: 0,
            exhausted: false,
            timeouts_below_cap: 0,
            raised_without_reduction: false,
            last_progress: Instant::now(),
        }
    }

    fn timeout_rate(&self) -> f64 {
        match self.timed_out.len() {
            0 => 0.0,
            n => self.timed_out.iter().filter(|&&t| t).count() as f64 / n as f64,
        }
    }

    /// The limit for the next run; advances the exploration state.
    pub fn current(&mut self) -> f64 {
        self.evaluate_exploration();
        self.rung(self.level)
    }

    /// The limit without advancing anything, for checking cached timeouts.
    pub fn peek(&self) -> f64 {
        self.rung(self.level)
    }

    pub fn record_completion(&mut self, runtime: f64, interesting: bool) {
        push(&mut self.runtimes, runtime, RUNTIME_WINDOW);
        if interesting {
            push(&mut self.interesting_runtimes, runtime, INTERESTING_WINDOW);
        }
        push(&mut self.timed_out, false, OUTCOME_WINDOW);
        if self.level > 0 {
            self.runs_at_level += 1;
        }
    }

    pub fn record_timeout(&mut self, used: f64) {
        push(&mut self.timed_out, true, OUTCOME_WINDOW);
        if used < self.cap {
            self.timeouts_below_cap += 1;
        }
        if self.level > 0 {
            self.runs_at_level += 1;
            self.timeouts_at_level += 1;
        }
    }

    pub fn note_reduction(&mut self) {
        self.last_progress = Instant::now();
        self.timeouts_below_cap = 0;
        self.exhausted = false;
        self.raised_without_reduction = false;
        self.reset_exploration();
    }

    /// The reducer ran out of things to try: whether the limit was raised, so another round may
    /// make progress.
    pub fn attempt_unstick(&mut self) -> bool {
        if self.exhausted {
            return false;
        }
        let uncapped = self.cap.is_infinite();
        if !(uncapped && self.raised_without_reduction)
            && self.timeouts_below_cap > 0
            && self.rung(self.level + 1) > self.rung(self.level)
        {
            self.level += 1;
            self.runs_at_level = 0;
            self.timeouts_at_level = 0;
            self.raised_without_reduction = true;
            return true;
        }
        if self.level > 0 {
            self.reset_exploration();
            self.exhausted = true;
        }
        false
    }

    pub fn reset(&mut self) {
        self.runtimes.clear();
        self.interesting_runtimes.clear();
        self.timed_out.clear();
        self.timeouts_below_cap = 0;
        self.exhausted = false;
        self.raised_without_reduction = false;
        self.reset_exploration();
        self.last_progress = Instant::now();
    }

    fn reset_exploration(&mut self) {
        self.level = 0;
        self.runs_at_level = 0;
        self.timeouts_at_level = 0;
    }

    fn clamp(&self, v: f64) -> f64 {
        // The cap wins over the minimum.
        self.cap.min(MIN_TIMEOUT.max(v))
    }

    fn base(&self) -> f64 {
        if self.runtimes.is_empty() {
            return self.cap;
        }
        let mut sorted: Vec<f64> = self.runtimes.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let index =
            (sorted.len() - 1).min((RUNTIME_QUANTILE * sorted.len() as f64).ceil() as usize);
        let mut candidate = RUNTIME_MULTIPLIER * sorted[index];
        if let Some(m) = self.interesting_runtimes.iter().copied().reduce(f64::max) {
            candidate = candidate.max(INTERESTING_MULTIPLIER * m);
        }
        self.clamp(candidate)
    }

    fn rung(&self, level: i32) -> f64 {
        self.clamp(self.base() * 2f64.powi(level))
    }

    fn evaluate_exploration(&mut self) {
        if self.exhausted {
            return;
        }
        if self.level == 0 {
            let stall = STALL_MIN_SECONDS.max(STALL_TIMEOUT_MULTIPLIER * self.rung(0));
            if self.last_progress.elapsed().as_secs_f64() > stall
                && self.timed_out.len() >= MIN_OUTCOMES_FOR_EXPLORATION
                && self.timeout_rate() >= TIMEOUT_RATE_THRESHOLD
                && self.rung(1) > self.rung(0)
            {
                self.level = 1;
                self.runs_at_level = 0;
                self.timeouts_at_level = 0;
            }
        } else if self.runs_at_level >= RUNG_RUN_BUDGET {
            if self.timeouts_at_level > 0 && self.rung(self.level + 1) > self.rung(self.level) {
                self.level += 1;
                self.runs_at_level = 0;
                self.timeouts_at_level = 0;
            } else {
                self.reset_exploration();
                self.exhausted = true;
            }
        }
    }
}
