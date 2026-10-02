use thiserror::Error;

use crate::compiler::CompileError;
use crate::lua::stash::StashedError;
use crate::parser::{self, SyntaxReport};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("{0}")]
    Parse(SyntaxError),
    #[error(transparent)]
    Compile(#[from] CompileError),
    #[error("internal: {0}")]
    Internal(&'static str),
}

/// A chunk's syntax errors. Displays as the plain-text report, the form a
/// Lua error message carries.
#[derive(Debug)]
pub struct SyntaxError {
    pub(crate) chunk: String,
    pub(crate) source: String,
    pub(crate) reports: Vec<SyntaxReport>,
}

impl SyntaxError {
    /// The reports drawn against the source, with ANSI colors when `color`.
    pub fn render(&self, color: bool) -> String {
        parser::render_reports(&self.reports, &self.source, &self.chunk, color)
    }
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render(false))
    }
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("bad executor mode")]
    BadMode,
    /// The main thread yielded to the host before completing. `finish`/
    /// `execute` can't surface yielded values across their `'gc`
    /// boundary, so they raise this rather than reporting bogus
    /// completion. To consume the yielded values and feed resume args
    /// back, drive the executor manually with `Executor::step` /
    /// `Lua::resume` instead.
    #[error("main thread yielded; use Lua::resume to continue")]
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
