//! `Stashable`/`Fetchable` traits for carrying GC values across `enter`
//! boundaries, plus stash handles for the types the MVP hands out.

use std::fmt;

use crate::Rootable;
use crate::dmm::{DynamicRoot, DynamicRootSet, Gc, Mutation, RefLock};
use crate::env::error::Error;
use crate::env::function::FunctionKind;
use crate::env::table::TableState;
use crate::env::thread::ThreadState;
use crate::env::{Function, Table, Thread, Value};
use crate::lua::executor::{Executor, ExecutorInner};

pub trait Stashable<'gc> {
    type Stashed;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> Self::Stashed;
}

pub trait Fetchable {
    type Fetched<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Self::Fetched<'gc>;
}

// ---------------------------------------------------------------------------

pub struct StashedFunction(pub(crate) DynamicRoot<Rootable![FunctionKind<'_>]>);

impl<'gc> Stashable<'gc> for Function<'gc> {
    type Stashed = StashedFunction;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedFunction {
        StashedFunction(roots.stash::<Rootable![FunctionKind<'_>]>(mc, self.inner()))
    }
}

impl Fetchable for StashedFunction {
    type Fetched<'gc> = Function<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Function<'gc> {
        Function::from_inner(roots.fetch::<Rootable![FunctionKind<'_>]>(&self.0))
    }
}

// ---------------------------------------------------------------------------

pub struct StashedTable(pub(crate) DynamicRoot<Rootable![RefLock<TableState<'_>>]>);

impl<'gc> Stashable<'gc> for Table<'gc> {
    type Stashed = StashedTable;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedTable {
        StashedTable(roots.stash::<Rootable![RefLock<TableState<'_>>]>(mc, self.inner()))
    }
}

impl Fetchable for StashedTable {
    type Fetched<'gc> = Table<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Table<'gc> {
        Table::from_inner(roots.fetch::<Rootable![RefLock<TableState<'_>>]>(&self.0))
    }
}

// ---------------------------------------------------------------------------

pub struct StashedThread(pub(crate) DynamicRoot<Rootable![RefLock<ThreadState<'_>>]>);

impl<'gc> Stashable<'gc> for Thread<'gc> {
    type Stashed = StashedThread;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedThread {
        StashedThread(roots.stash::<Rootable![RefLock<ThreadState<'_>>]>(mc, self.inner()))
    }
}

impl Fetchable for StashedThread {
    type Fetched<'gc> = Thread<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Thread<'gc> {
        Thread::from_inner(roots.fetch::<Rootable![RefLock<ThreadState<'_>>]>(&self.0))
    }
}

// ---------------------------------------------------------------------------

pub struct StashedExecutor(pub(crate) DynamicRoot<Rootable![RefLock<ExecutorInner<'_>>]>);

impl<'gc> Stashable<'gc> for Executor<'gc> {
    type Stashed = StashedExecutor;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedExecutor {
        StashedExecutor(roots.stash::<Rootable![RefLock<ExecutorInner<'_>>]>(mc, self.inner()))
    }
}

impl Fetchable for StashedExecutor {
    type Fetched<'gc> = Executor<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Executor<'gc> {
        Executor::from_inner(roots.fetch::<Rootable![RefLock<ExecutorInner<'_>>]>(&self.0))
    }
}

// ---------------------------------------------------------------------------

/// `'static`-erased handle to a Lua [`Value`]. The underlying value may be a
/// primitive (Integer/Float/Boolean/Nil) or a `Gc`-pointer flavor; we always
/// wrap in a one-slot `Gc<Value>` so the `DynamicRootSet` has a uniform
/// `Gc<...>` to pin. Allocation cost is negligible — stashing happens on
/// host boundaries, not in inner loops.
pub struct StashedValue(pub(crate) DynamicRoot<Rootable![Value<'_>]>);

impl<'gc> Stashable<'gc> for Value<'gc> {
    type Stashed = StashedValue;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedValue {
        let g = Gc::new(mc, self);
        StashedValue(roots.stash::<Rootable![Value<'_>]>(mc, g))
    }
}

impl Fetchable for StashedValue {
    type Fetched<'gc> = Value<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Value<'gc> {
        *roots.fetch::<Rootable![Value<'_>]>(&self.0)
    }
}

impl fmt::Debug for StashedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StashedValue")
    }
}

/// `'static`-erased handle to an in-flight Lua error. Bridges `Error<'gc>`
/// (a `Value<'gc>` carrier) into the host-facing [`super::RuntimeError`]
/// without losing the original payload.
pub struct StashedError(pub(crate) StashedValue);

impl<'gc> Stashable<'gc> for Error<'gc> {
    type Stashed = StashedError;
    fn stash(self, mc: &Mutation<'gc>, roots: DynamicRootSet<'gc>) -> StashedError {
        StashedError(self.value().stash(mc, roots))
    }
}

impl Fetchable for StashedError {
    type Fetched<'gc> = Error<'gc>;
    fn fetch<'gc>(&self, roots: DynamicRootSet<'gc>) -> Error<'gc> {
        Error::new(self.0.fetch(roots))
    }
}

impl fmt::Debug for StashedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StashedError")
    }
}
