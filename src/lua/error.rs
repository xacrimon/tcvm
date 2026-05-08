use ariadne::Report;
use thiserror::Error;

use crate::compiler::CompileError;
use crate::lua::stash::StashedError;
use crate::parser::machinery::Span;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("parse error")]
    Parse(Vec<Report<'static, Span>>),
    #[error(transparent)]
    Compile(#[from] CompileError),
    #[error("internal: {0}")]
    Internal(&'static str),
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Internal VM error (type mismatch, missing metamethod, etc.). Carries
    /// the offending PC for debugging.
    #[error("vm error at pc {pc}")]
    Opcode { pc: usize },
    #[error("bad executor mode")]
    BadMode,
    /// The main thread yielded to the host before completing. `finish`/
    /// `execute` have no path to surface the yielded values, so they
    /// turn this into an error rather than reporting bogus completion.
    /// A host-side resume API isn't implemented yet.
    #[error("main thread yielded; resume not yet supported")]
    MainYielded,
    /// User-thrown Lua error (`error(value)`); payload is the stashed value.
    /// Inspect / display via `Lua::enter` + `Fetchable::fetch`.
    #[error("lua error")]
    Lua(StashedError),
    #[error(transparent)]
    Type(#[from] TypeError),
}

#[derive(Debug, Error)]
pub enum TypeError {
    #[error("expected {expected}, got {got}")]
    Mismatch {
        expected: &'static str,
        got: &'static str,
    },
    #[error("expected {expected} value(s), got {got}")]
    Arity { expected: usize, got: usize },
}
