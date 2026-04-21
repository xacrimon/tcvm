//! `Stashable`/`Fetchable` traits for carrying GC values across `enter`
//! boundaries, plus stash handles for the types the MVP hands out.

use crate::Rootable;
use crate::dmm::{DynamicRoot, DynamicRootSet, Mutation, RefLock};
use crate::env::function::FunctionKind;
use crate::env::table::TableState;
use crate::env::thread::ThreadState;
use crate::env::{Function, Table, Thread};
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
