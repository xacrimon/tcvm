use std::marker::PhantomData;

use crate::dmm::{lock::RefLock, Collect, Gc, Mutation};

#[derive(Clone, Copy)]
pub struct Table<'gc>(Gc<'gc, RefLock<TableState<'gc>>>);

pub struct TableState<'gc> {
    raw: RawTable<'gc>,
    metadatable: Option<Table<'gc>>,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct RawTable<'gc> {
    marker: PhantomData<&'gc ()>,
}
