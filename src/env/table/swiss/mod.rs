//! The Swiss table behind the table hash parts, trimmed from hashbrown's `RawTable`
//! (<https://github.com/rust-lang/hashbrown>, commit 7109e3a6388b0a6a0e843befc8876864e1d44d2e,
//! MIT OR Apache-2.0, Copyright (c) 2016 Amanieu d'Antras). Diff against that commit when
//! porting upstream fixes.
//!
//! Beyond the trimming (`Copy` entries, infallible allocation, no erase, no `Drop`), a deleted
//! entry becomes a *dead* bucket with its own control tag, so `next` can resume from its key. A
//! set finds its key's entry live or dead and revives it; other keys may take a dead bucket like
//! a tombstone. Probes load group-aligned windows, so a control byte has no mirror and turning
//! a bucket dead or back is one store of its group. The generic group's tag match is exact,
//! since dead keys are compared by bits alone.
//!
//! Tables allocate from a [`GcAlloc`](crate::dmm::allocator_api::GcAlloc), so there is nothing
//! to drop: the holder marks [`RawTable::allocation`] when traced, and the sweep frees it after.

mod control;
mod raw;

#[cfg(test)]
pub(super) use control::Group;
pub(super) use raw::RawTable;
