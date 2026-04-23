#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
#![feature(macro_metavar_expr)]
#![feature(likely_unlikely)]
#![feature(allocator_api)]
#![feature(rust_preserve_none_cc)]
#![feature(variant_count)]

mod builtin;
pub(crate) mod compiler;
pub mod dmm;
pub mod env;
pub(crate) mod instruction;
pub mod lua;
pub(crate) mod parser;
pub(crate) mod vm;

pub use lua::{
    Context, Executor, ExecutorMode, Fetchable, FromMultiValue, FromValue, IntoMultiValue,
    IntoValue, LoadError, Lua, RuntimeError, Stashable, StashedExecutor, StashedFunction,
    StashedTable, StashedThread, TypeError,
};
