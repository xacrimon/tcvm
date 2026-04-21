use ariadne::Report;
use thiserror::Error;

use crate::compiler::CompileError;
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
    #[error("vm error at pc {pc}")]
    Opcode { pc: usize },
    #[error("bad executor mode")]
    BadMode,
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
