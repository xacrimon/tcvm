pub(crate) mod defs;
pub(crate) mod format;
mod rules;
#[cfg(test)]
mod snapshot_tests;

use std::fmt;

use cstree::interning::TokenInterner;
use thiserror::Error;

use crate::dmm::{Collect, Gc};
use crate::env::{LuaString, Prototype};
use crate::lua;
use crate::parser::{LineMap, syntax};

pub fn compile_chunk<'gc>(
    ctx: lua::Context<'gc>,
    root: &syntax::Root,
    lines: &LineMap,
    interner: &TokenInterner,
    source: LuaString<'gc>,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    rules::compile(ctx, root, lines, interner, source)
}

/// 1-based source line; 0 when the error has no position (internal errors).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Collect)]
#[collect(internal, require_static)]
pub struct LineNumber(pub u32);

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The function a limit error names, by its `line_defined` (0 for the main chunk).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncLine(pub u32);

impl fmt::Display for FuncLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0 => f.write_str("main function"),
            n => write!(f, "function at line {n}"),
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum CompileErrorKind {
    #[error("internal compiler error: {0}")]
    Internal(&'static str),
    #[error("variable '{0}' not declared")]
    UndeclaredGlobal(String),
    #[error("_ENV is global when accessing variable '{0}'")]
    GlobalEnv(String),
    #[error("attempt to assign to const variable '{0}'")]
    ConstAssign(String),
    /// Lua's `errorlimit`.
    #[error("too many {what} (limit is {limit}) in {func}")]
    Limit {
        what: &'static str,
        limit: usize,
        func: FuncLine,
    },
    #[error("too many fixed parameters")]
    FixedParameters,
    #[error("too many inner functions")]
    Functions,
    #[error("too many constants")]
    Constants,
    #[error("too many table constructors")]
    Constructors,
    #[error("label '{0}' already defined on line {1}")]
    DuplicateLabel(String, u32),
    #[error("no visible label '{0}' for <goto> at line {1}")]
    GotoInvalid(String, u32),
    #[error("<goto {0}> at line {1} jumps into the scope of '{2}'")]
    JumpLocal(String, u32, String),
    #[error("jump offset overflow")]
    JumpOverflow,
    #[error("multiple to-be-closed variables in local list")]
    MultipleClose,
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
