pub(crate) mod defs;
pub(crate) mod format;
mod rules;
#[cfg(test)]
mod snapshot_tests;

use std::fmt;

use cstree::interning::TokenInterner;
use thiserror::Error;

use crate::dmm::{Collect, Gc, Mutation};
use crate::env::Prototype;
use crate::parser::syntax;

pub fn compile_chunk<'gc>(
    mc: &Mutation<'gc>,
    root: &syntax::Root,
    interner: &TokenInterner,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    rules::compile(mc, root, interner)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Collect)]
#[collect(internal, require_static)]
pub struct LineNumber(pub u64);

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", u128::from(self.0) + 1)
    }
}

#[derive(Debug, Error)]
pub enum ParseErrorKind {
    #[error("found {unexpected:?}, expected {expected:?}")]
    Unexpected {
        unexpected: String,
        expected: String,
    },
    #[error(
        "unexpected end of token stream{}",
        .expected.as_ref().map(|e| format!(", expected {e}")).unwrap_or_default()
    )]
    EndOfStream { expected: Option<String> },
    #[error("cannot assign to expression")]
    AssignToExpression,
    #[error("expression is not a statement")]
    ExpressionNotStatement,
    #[error("recursion limit reached")]
    RecursionLimit,
    #[error("lexer error")]
    LexError(#[from] LexError),
}

#[derive(Debug, Error)]
#[error("parse error at line {line_number}: {kind}")]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub line_number: LineNumber,
}

#[derive(Debug, Error)]
#[error("todo")]
pub struct LexError {}

#[derive(Debug, Clone, Error)]
pub enum CompileErrorKind {
    #[error("internal compiler error: {0}")]
    Internal(&'static str),
    #[error("insufficient available registers")]
    Registers,
    #[error("too many upvalues")]
    UpValues,
    #[error("too many fixed parameters")]
    FixedParameters,
    #[error("too many inner functions")]
    Functions,
    #[error("too many constants")]
    Constants,
    #[error("label defined multiple times")]
    DuplicateLabel,
    #[error("goto target label not found")]
    GotoInvalid,
    #[error("jump into scope of new local variable")]
    JumpLocal,
    #[error("jump offset overflow")]
    JumpOverflow,
}

#[derive(Debug, Clone, Error)]
#[error("compiler error at line {line_number}: {kind}")]
pub struct CompileError {
    pub kind: CompileErrorKind,
    pub line_number: LineNumber,
}

impl CompileError {
    pub(crate) fn internal(msg: &'static str) -> Self {
        CompileError {
            kind: CompileErrorKind::Internal(msg),
            line_number: LineNumber(0),
        }
    }
}
