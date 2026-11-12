//! xoshiro256**, seeded through splitmix64: a program is a function of its seed alone.

pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    pub fn u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in `0..n`; `n` > 0.
    pub fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.u64() % (hi - lo + 1) as u64) as i64
    }

    pub fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    pub fn pick<'a, T>(&mut self, l: &'a [T]) -> &'a T {
        &l[self.below(l.len())]
    }

    /// An index drawn by weight; the weights are not all zero.
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u64 = weights.iter().map(|&w| w as u64).sum();
        let mut x = self.u64() % total;
        for (i, &w) in weights.iter().enumerate() {
            if x < w as u64 {
                return i;
            }
            x -= w as u64;
        }
        unreachable!()
    }
}
