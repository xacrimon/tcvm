use cstree::build::NodeCache;

use crate::compiler::compile_chunk;
use crate::dmm::{DynamicRootSet, Gc, Mutation, RefLock};
use crate::env::function::{Function, UpvalueState};
use crate::env::shape::Shape;
use crate::env::string::Interner;
use crate::env::{Table, Thread, Value};
use crate::lua::stash::{Fetchable, Stashable};
use crate::lua::{LoadError, State};
use crate::parser;

/// Cheap, copy handle into the arena mutation context.
#[derive(Copy, Clone)]
pub struct Context<'gc> {
    mutation: &'gc Mutation<'gc>,
    state: &'gc State<'gc>,
}

impl<'gc> Context<'gc> {
    pub(crate) fn new(mutation: &'gc Mutation<'gc>, state: &'gc State<'gc>) -> Self {
        Context { mutation, state }
    }

    pub fn mutation(self) -> &'gc Mutation<'gc> {
        self.mutation
    }

    pub fn globals(self) -> Table<'gc> {
        self.state.globals
    }

    /// Shared empty / root shape — every newly-allocated table starts
    /// here. Stable for the lifetime of the runtime.
    pub fn empty_shape(self) -> Shape<'gc> {
        self.state.empty_shape
    }

    /// Pre-interned `(name, bit)` pairs for every Lua metamethod.
    /// Used by `Shape::recompute_mm_cache` to walk a metatable in
    /// pointer-identity lookups instead of allocating per-name
    /// LuaStrings.
    pub(crate) fn metamethod_names(
        self,
    ) -> &'gc [(crate::env::LuaString<'gc>, crate::env::MetamethodBits)] {
        &self.state.metamethod_names
    }

    pub fn main_thread(self) -> Thread<'gc> {
        self.state.main_thread
    }

    pub fn roots(self) -> DynamicRootSet<'gc> {
        self.state.roots
    }

    pub(crate) fn interner(&self) -> &Interner<'gc> {
        &self.state.interner
    }

    pub fn stash<S: Stashable<'gc>>(self, s: S) -> S::Stashed {
        s.stash(self.mutation, self.state.roots)
    }

    pub fn fetch<F: Fetchable>(self, f: &F) -> F::Fetched<'gc> {
        f.fetch(self.state.roots)
    }

    /// Parse and compile `source` into a `Function`, with `_ENV` bound to the
    /// runtime's globals table.
    pub fn load(self, source: &str, _name: Option<&str>) -> Result<Function<'gc>, LoadError> {
        let mut cache = NodeCache::new();
        let (syntax, reports) = parser::parse(&mut cache, source);
        if !reports.is_empty() {
            return Err(LoadError::Parse(reports));
        }
        let root = parser::syntax::Root::new(syntax)
            .ok_or(LoadError::Internal("parser did not produce a Root node"))?;
        let proto = compile_chunk(self, &root, cache.interner())?;

        // Main chunk's upvalue 0 is _ENV. Pre-close it onto globals.
        let env_uv = Gc::new(
            self.mutation,
            RefLock::new(UpvalueState::Closed(Value::table(self.state.globals))),
        );
        Ok(Function::new_lua(self.mutation, proto, Box::from([env_uv])))
    }
}
