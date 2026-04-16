#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
#![feature(macro_metavar_expr)]
#![feature(likely_unlikely)]
#![feature(allocator_api)]
#![feature(rust_preserve_none_cc)]
#![feature(variant_count)]

mod compiler;
pub mod dmm;
pub mod env;
pub(crate) mod instruction;
mod parser;
mod vm;
