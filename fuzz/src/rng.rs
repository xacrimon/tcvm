//! The random source of everything that must repeat: a program is a function of its seed alone,
//! and so is a reduction. `rand` over PCG-64 (`rand_pcg`, which tcvm's `math.random` uses too): a
//! named algorithm seeded from the seed, not the thread's generator, which no seed brings back.
//! `Cargo.lock` pins `rand`, whose sampling may change between versions: an update changes the
//! program of every seed.

use rand::distr::weighted::WeightedIndex;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt as _, SeedableRng};
use rand_pcg::Pcg64;

#[derive(Clone)]
pub struct Rng(Pcg64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(Pcg64::seed_from_u64(seed))
    }

    /// Uniform in `0..n`; `n` > 0.
    pub fn below(&mut self, n: usize) -> usize {
        self.0.random_range(0..n)
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        self.0.random_range(lo..=hi)
    }

    /// Uniform in `0.0..1.0`.
    pub fn f64(&mut self) -> f64 {
        self.0.random()
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.0.random_bool(p)
    }

    pub fn pick<'a, T>(&mut self, l: &'a [T]) -> &'a T {
        l.choose(&mut self.0).expect("something to pick")
    }

    pub fn shuffle<T>(&mut self, l: &mut [T]) {
        l.shuffle(&mut self.0);
    }

    /// An element drawn by `weight`; the weights are not all zero.
    pub fn pick_by<'a, T>(&mut self, l: &'a [T], weight: impl Fn(&T) -> u32) -> &'a T {
        (l.choose_weighted(&mut self.0, weight)).expect("a weight above zero")
    }

    /// An index drawn by weight; the weights are not all zero.
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let index = WeightedIndex::new(weights).expect("a weight above zero");
        self.0.sample(index)
    }
}
