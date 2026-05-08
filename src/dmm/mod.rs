#![cfg_attr(miri, feature(strict_provenance))]

pub mod arena;
pub mod barrier;
pub mod checked_cell;
pub mod collect;
mod collect_impl;
mod context;
pub mod dynamic_roots;
mod gc;
mod gc_weak;
pub mod lock;
pub mod metrics;
mod no_drop;
mod static_collect;
mod types;
mod unsize;

pub mod allocator_api;

mod hashbrown;

#[doc(hidden)]
pub use tcvm_derive::__unelide_lifetimes;

#[doc(hidden)]
pub use self::{arena::__DynRootable, no_drop::__MustNotImplDrop, unsize::__CoercePtrInternal};

pub use self::{
    allocator_api::MetricsAlloc,
    arena::{Arena, Rootable},
    checked_cell::{BorrowError, BorrowMutError, Ref, RefMut},
    collect::{Collect, Trace},
    context::{Finalization, Mutation},
    dynamic_roots::{DynamicRoot, DynamicRootSet},
    gc::Gc,
    gc_weak::GcWeak,
    lock::{GcLock, GcRefLock, Lock, RefLock},
    static_collect::Static,
};
