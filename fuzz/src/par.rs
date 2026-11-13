//! Work on several threads.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

/// Run `job` on each of `items` until `stop`, on `workers` threads; `each` gets the results on
/// the calling thread as they come, in no order.
pub fn stream<T: Send>(
    items: Range<u64>,
    workers: usize,
    stop: &(dyn Fn() -> bool + Sync),
    job: impl Fn(u64) -> T + Sync,
    mut each: impl FnMut(u64, T),
) {
    let next = AtomicU64::new(items.start);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|s| {
        for _ in 0..workers {
            let (tx, next, job) = (tx.clone(), &next, &job);
            s.spawn(move || {
                while !stop() {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.end || tx.send((i, job(i))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (i, t) in rx {
            each(i, t);
        }
    });
}

/// Below this many items, `map` runs them on the calling thread: spawning costs more.
const INLINE: usize = 64;

/// `f(i)` for `i` in `0..n` on `workers` threads; the results in order of `i`.
pub fn map<T: Send>(workers: usize, n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n <= INLINE {
        return (0..n).map(f).collect();
    }
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    let each = |i: u64, t| out[i as usize] = Some(t);
    stream(0..n as u64, workers, &|| false, |i| f(i as usize), each);
    (out.into_iter())
        .map(|t| t.expect("every job ran"))
        .collect()
}
