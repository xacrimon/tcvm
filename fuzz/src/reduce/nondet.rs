//! Statistics for reducing under a test that does not always reproduce: shrinkray's
//! `nondeterminism.py`, whose module docs explain the scheme. A candidate is adopted only when
//! its estimated reproduction rate clears a threshold priced off the incumbent's (the
//! gauntlet); the incumbent's rate is a monotone lower bound (the anchor); evidence about every
//! candidate accumulates across retries; and an incumbent that turns out to reproduce less often
//! than required is a detected false accept, answered by backtracking and a higher hit minimum.

/// Wilson score z for a two-sided 95% interval.
pub const Z: f64 = 1.96;
/// One-sided level for anchor raises over all of a run's attempts together.
const ANCHOR_ALPHA: f64 = 0.025;
/// Replays of the initial test case at startup, and of the result at the end.
pub const DETECTION_REPLAYS: u32 = 4;
/// While believed deterministic, the current test case is replayed every this many tests.
pub const VERIFY_INTERVAL: u64 = 100;
/// The confirmation bar for a test case adopted on single runs: reject with no hits at the
/// gate, accept on the fourth hit, give up at the cap.
pub const GATE_RUNS: u32 = 10;
const CONFIRM_CAP: u32 = 40;
const CONFIRM_MIN_HITS: u32 = 4;
pub const GAUNTLET_CAP: u32 = 30;
const GAUNTLET_GAMMA: f64 = 0.8;
const GAUNTLET_FLOOR: f64 = 0.05;
/// Past this anchor the incumbent counts as deterministic and gamma becomes 1.
const RETENTION_HIGH_WATER: f64 = 0.8;
/// Batches that may seed or raise the anchor are extended to this many runs.
pub const ANCHOR_SEED_RUNS: u32 = 20;
const INITIAL_MIN_HITS: u32 = 2;
const MIN_HITS_CEILING: u32 = 12;
/// The incumbent monitor judges only with this many fresh replays.
const MONITOR_MIN_RUNS: u32 = 5;

pub fn wilson_bound(interesting: u32, runs: u32, upper: bool, z: f64) -> f64 {
    if runs == 0 {
        return if upper { 1.0 } else { 0.0 };
    }
    if interesting == 0 && !upper {
        return 0.0;
    }
    if interesting == runs && upper {
        return 1.0;
    }
    let (n, p) = (runs as f64, interesting as f64 / runs as f64);
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt();
    let bound = if upper {
        centre + margin
    } else {
        centre - margin
    } / denominator;
    bound.clamp(0.0, 1.0)
}

/// The inverse of the standard normal CDF (Acklam's rational approximation, relative error
/// below 1.2e-9), for `anchor_z`.
fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383_577_518_672_69e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// The z for the `attempts`-th attempt to raise the anchor: Bonferroni over the attempts so
/// far, so the running maximum of many bounds does not overshoot the rate.
fn anchor_z(attempts: u32) -> f64 {
    Z.max(inv_normal_cdf(1.0 - ANCHOR_ALPHA / attempts.max(1) as f64))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Evidence {
    pub interesting: u32,
    pub runs: u32,
}

impl Evidence {
    pub fn record(&mut self, interesting: bool) {
        self.runs += 1;
        self.interesting += interesting as u32;
    }

    pub fn rate(&self) -> f64 {
        match self.runs {
            0 => 0.0,
            n => self.interesting as f64 / n as f64,
        }
    }

    pub fn lower_bound(&self, z: f64) -> f64 {
        wilson_bound(self.interesting, self.runs, false, z)
    }

    pub fn upper_bound(&self) -> f64 {
        wilson_bound(self.interesting, self.runs, true, Z)
    }

    /// The runs recorded after `earlier`, a snapshot of this evidence.
    pub fn since(&self, earlier: Evidence) -> Evidence {
        Evidence {
            interesting: self.interesting - earlier.interesting,
            runs: self.runs - earlier.runs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Reject,
    Continue,
}

fn gauntlet_threshold(anchor: f64) -> f64 {
    let gamma = if anchor >= RETENTION_HIGH_WATER {
        1.0
    } else {
        GAUNTLET_GAMMA
    };
    (gamma * anchor).max(GAUNTLET_FLOOR)
}

/// Whether a candidate's evidence clears the incumbent's anchor. `reject_bar` gives up once the
/// upper bound is below the rate the reduction's incumbents have reproduced at.
pub fn gauntlet(evidence: Evidence, anchor: f64, min_hits: u32, reject_bar: f64) -> Verdict {
    let threshold = gauntlet_threshold(anchor);
    if evidence.interesting >= min_hits && evidence.lower_bound(Z) >= threshold {
        return Verdict::Accept;
    }
    if evidence.upper_bound() < threshold.max(reject_bar) || evidence.runs >= GAUNTLET_CAP {
        return Verdict::Reject;
    }
    // Even if every remaining run hit, the bound would not clear the threshold.
    let best = Evidence {
        interesting: evidence.interesting + (GAUNTLET_CAP - evidence.runs),
        runs: GAUNTLET_CAP,
    };
    if best.lower_bound(Z) < threshold {
        return Verdict::Reject;
    }
    Verdict::Continue
}

/// Whether an incumbent adopted on single runs really reproduces.
pub fn confirmation_bar(evidence: Evidence) -> Verdict {
    if evidence.interesting >= CONFIRM_MIN_HITS {
        Verdict::Accept
    } else if (evidence.interesting == 0 && evidence.runs >= GATE_RUNS)
        || evidence.interesting + CONFIRM_CAP.saturating_sub(evidence.runs) < CONFIRM_MIN_HITS
    {
        Verdict::Reject
    } else {
        Verdict::Continue
    }
}

/// The run-level state. `active` is sticky; `anchor` is the monotone lower bound on the
/// incumbent's rate; `confirming` selects the sweep: in a fast sweep a candidate whose first
/// run misses is rejected at that one run's cost, in a confirmation sweep every candidate is
/// driven to a bound verdict.
#[derive(Clone, Debug)]
pub struct Policy {
    pub active: bool,
    pub anchor: f64,
    pub confirming: bool,
    pub min_hits: u32,
    pub false_accepts: u32,
    /// Fresh evidence about the current incumbent, from the runs after its accept decision and
    /// the monitor.
    pub incumbent: Evidence,
    /// The same pooled over every incumbent so far.
    pub pool: Evidence,
    anchor_attempts: u32,
    pub replay_calls: u64,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy {
            active: false,
            anchor: 0.0,
            confirming: false,
            min_hits: INITIAL_MIN_HITS,
            false_accepts: 0,
            incumbent: Evidence::default(),
            pool: Evidence::default(),
            anchor_attempts: 0,
            replay_calls: 0,
        }
    }
}

impl Policy {
    pub fn threshold(&self) -> f64 {
        gauntlet_threshold(self.anchor)
    }

    pub fn reject_bar(&self) -> f64 {
        self.pool.lower_bound(Z)
    }

    fn raise_bar(&self) -> f64 {
        self.anchor.max(self.pool.rate())
    }

    /// Whether `remaining` more runs after `unselected` could raise the anchor next attempt.
    pub fn raise_reachable(&self, unselected: Evidence, remaining: u32) -> bool {
        let best = Evidence {
            interesting: unselected.interesting + remaining,
            runs: unselected.runs + remaining,
        };
        best.lower_bound(anchor_z(self.anchor_attempts + 1)) > self.raise_bar()
    }

    /// Raise the anchor to `evidence`'s bound if it beats the raise bar; `evidence` must not
    /// have been selected on (the runs after an accept decision).
    pub fn raise_anchor(&mut self, evidence: Evidence) {
        self.anchor_attempts += 1;
        let bound = evidence.lower_bound(anchor_z(self.anchor_attempts));
        if bound > self.raise_bar() {
            self.anchor = bound;
        }
    }

    pub fn record_incumbent_run(&mut self, interesting: bool) {
        self.incumbent.record(interesting);
        self.pool.record(interesting);
    }

    pub fn adopt(&mut self, unselected: Evidence) {
        self.incumbent = unselected;
        self.pool.interesting += unselected.interesting;
        self.pool.runs += unselected.runs;
    }

    pub fn incumbent_failing(&self) -> bool {
        self.incumbent.runs >= MONITOR_MIN_RUNS && self.incumbent.upper_bound() < self.threshold()
    }

    pub fn record_false_accept(&mut self) {
        self.false_accepts += 1;
        self.min_hits = MIN_HITS_CEILING.min(self.min_hits + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_match_the_closed_form() {
        // One hit in one run: the gauntlet's comment gives 0.2065 for the lower bound.
        assert!((wilson_bound(1, 1, false, Z) - 0.2065).abs() < 1e-4);
        // LCB(4/30) is above the floor, as GAUNTLET_FLOOR's derivation needs.
        assert!(wilson_bound(4, 30, false, Z) > GAUNTLET_FLOOR);
        assert!((inv_normal_cdf(0.975) - 1.959964).abs() < 1e-5);
        assert_eq!(anchor_z(1), Z.max(inv_normal_cdf(0.975)));
    }
}
