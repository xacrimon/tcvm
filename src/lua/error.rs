use ariadne::Report;
use thiserror::Error;

use crate::compiler::CompileError;
use crate::env::NativeError;
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
    /// the offending PC for debugging. Phase 6 will fold these into Lua-level
    /// errors carrying a structured payload.
    #[error("vm error at pc {pc}")]
    Opcode { pc: usize },
    #[error("bad executor mode")]
    BadMode,
    /// Legacy native-callback error path (P0/P6 transition). Once all
    /// callbacks return `Error<'gc>` (P6+), this collapses into `Lua`.
    #[error("native callback error: {message}")]
    Native { message: String },
    /// User-thrown Lua error (`error(value)`); payload is the stashed value.
    /// Inspect / display via `Lua::enter` + `Fetchable::fetch`.
    #[error("lua error")]
    Lua(StashedError),
    #[error(transparent)]
    Type(#[from] TypeError),
}

impl From<NativeError> for RuntimeError {
    fn from(e: NativeError) -> Self {
        RuntimeError::Native { message: e.message }
    }
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
