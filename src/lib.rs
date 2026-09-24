#![allow(incomplete_features)]
// Without the checked-cell guards, borrow guards are plain references; the `drop`s that release
// them in checked builds are still needed there.
#![cfg_attr(not(any(debug_assertions, test)), allow(clippy::drop_non_drop))]
#![feature(fn_align)]
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
pub mod vm;

pub use compiler::format::format_prototype;
pub use lua::{
    Context, Executor, ExecutorMode, Fetchable, FromMultiValue, FromValue, IntoMultiValue,
    IntoValue, LoadError, Lua, RuntimeError, Stashable, StashedError, StashedExecutor,
    StashedFunction, StashedTable, StashedThread, StashedValue, StepResult, TypeError,
};
