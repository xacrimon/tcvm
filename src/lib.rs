#![allow(incomplete_features)]
// Without the checked-cell guards, borrow guards are plain references; the `drop`s that release
// them in checked builds are still needed there.
#![cfg_attr(not(any(debug_assertions, test)), allow(clippy::drop_non_drop))]
// Without a library, the VM's and other libraries' code for it is left unused.
#![cfg_attr(not(feature = "lib-all"), allow(dead_code, unused_imports))]
#![cfg_attr(
    nightly,
    feature(
        fn_align,
        explicit_tail_calls,
        likely_unlikely,
        rust_preserve_none_cc,
        naked_functions_rustic_abi,
        offset_of_enum
    )
)]

mod builtin;
pub(crate) mod compiler;
pub mod dmm;
pub mod env;
pub(crate) mod instruction;
pub(crate) mod jit;
pub mod lua;
#[cfg(feature = "parser-api")]
pub mod parser;
#[cfg(not(feature = "parser-api"))]
pub(crate) mod parser;
pub(crate) mod utils;
pub mod vm;

pub use compiler::format::format_prototype;
pub use lua::{
    Context, Executor, ExecutorMode, Fetchable, FromMultiValue, FromValue, IntoMultiValue,
    IntoValue, LoadError, Lua, RuntimeError, SendLua, Stashable, StashedError, StashedExecutor,
    StashedFunction, StashedTable, StashedThread, StashedValue, StepResult, SyntaxError, TypeError,
};
