//! Tests of the continuation protocol, which is crate-internal: their
//! continuations are the `test:` entries of `CONT_TABLE`.

pub(crate) mod calls_lua;
pub(crate) mod pcall_through;
pub(crate) mod protocol;
