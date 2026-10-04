//! The Swiss table behind the table hash parts, trimmed from hashbrown's `RawTable`
//! (<https://github.com/rust-lang/hashbrown>, commit 7109e3a6388b0a6a0e843befc8876864e1d44d2e,
//! MIT OR Apache-2.0, Copyright (c) 2016 Amanieu d'Antras). Diff against that commit when
//! porting upstream fixes.
//!
//! Beyond the trimming (`Copy` entries, infallible allocation, no erase), a deleted entry
//! becomes a *dead* bucket with its own control tag, so `next` can resume from its key. A set
//! finds its key's entry live or dead and revives it; other keys may take a dead bucket like
//! a tombstone. Probes load group-aligned windows, so a control byte has no mirror and turning
//! a bucket dead or back is one store of its group. The generic group's tag match is exact,
//! since dead keys are compared by bits alone.

mod control;
mod raw;

#[cfg(test)]
pub(super) use control::Group;
pub(super) use raw::RawTable;
