#![allow(incomplete_features)]
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
#[cfg(jit_enabled)]
pub mod jit;
pub mod lua;
pub(crate) mod parser;
pub mod vm;

pub use compiler::format::format_prototype;
pub use lua::{
    Context, Executor, ExecutorMode, Fetchable, FromMultiValue, FromValue, IntoMultiValue,
    IntoValue, LoadError, Lua, RuntimeError, Stashable, StashedError, StashedExecutor,
    StashedFunction, StashedTable, StashedThread, StashedValue, StepResult, TypeError,
};

/// Individual front-end pipeline stages, exposed so benchmarks can time parsing
/// and bytecode generation in isolation — both live in `pub(crate)` modules
/// otherwise. Not part of the stable surface; `#[doc(hidden)]` keeps it out of
/// the docs. The later JIT stages (`lower`, `select`, `allocate`, `encode`) are
/// already public under `jit::`, so they need no wrapper here.
#[doc(hidden)]
pub mod bench_support {
    use cstree::build::NodeCache;

    use crate::compiler::{CompileError, compile_chunk};
    use crate::dmm::Gc;
    use crate::env::Prototype;
    use crate::lua::Context;
    use crate::parser::{self, syntax::Root};

    /// A parsed chunk: owns the syntax tree and the interner the compile stage
    /// still needs, so a caller can hand it straight to [`compile`].
    pub struct Parsed {
        root: Root,
        cache: NodeCache<'static>,
    }

    /// Stage 1 — source text to syntax tree. Panics on a malformed chunk; bench
    /// inputs are expected to be well-formed.
    pub fn parse(source: &str) -> Parsed {
        let mut cache = NodeCache::new();
        let (syntax, reports) = parser::parse(&mut cache, source);
        assert!(reports.is_empty(), "bench source failed to parse");
        let root = Root::new(syntax).expect("parser produced a Root node");
        Parsed { root, cache }
    }

    /// Stage 2 — syntax tree to a bytecode [`Prototype`]. Allocates in the arena,
    /// so it must run inside a [`Context`]; a benchmark that loops it should
    /// collect periodically to bound the garbage.
    pub fn compile<'gc>(
        ctx: Context<'gc>,
        parsed: &Parsed,
    ) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
        compile_chunk(ctx, &parsed.root, parsed.cache.interner())
    }
}
