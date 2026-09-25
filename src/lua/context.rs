use cstree::build::NodeCache;

use crate::compiler::compile_chunk;
use crate::dmm::{DynamicRootSet, Gc, Mutation, RefLock};
use crate::env::function::{Function, UpvalueState};
use crate::env::shape::Shape;
use crate::env::string::Interner;
use crate::env::{LuaString, Symbols, Table, Value};
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

    /// Dict-mode sentinel for tables with no metatable. Tables that
    /// migrate to dict mode while carrying a metatable use the
    /// per-`MtCache` sentinel instead.
    #[inline]
    pub fn empty_dict_sentinel(self) -> Shape<'gc> {
        self.state.empty_dict_sentinel
    }

    /// Globally-interned ambient `LuaString` symbols (metamethod
    /// names and friends). Slow paths read this to skip per-call
    /// interning; `Table::ensure_mt_cache` reads it to walk a
    /// metatable's slots at adoption time.
    #[inline]
    pub fn symbols(self) -> &'gc Symbols<'gc> {
        &self.state.symbols
    }

    /// The metatable `v`'s metamethods come from: its own for tables and
    /// userdata, otherwise the one shared by its type.
    #[inline]
    pub fn metatable_of(self, v: Value<'gc>) -> Option<Table<'gc>> {
        if let Some(t) = v.get_table() {
            t.metatable()
        } else if let Some(u) = v.get_userdata() {
            u.metatable()
        } else {
            self.state.type_metatable(v.kind()).get()
        }
    }

    /// Metamethod `name` of `v`, nil when absent (`luaT_gettmbyobj`).
    #[inline]
    pub fn metamethod_of(self, v: Value<'gc>, name: LuaString<'gc>) -> Value<'gc> {
        self.metatable_of(v)
            .map_or(Value::nil(), |mt| mt.raw_get(Value::string(name)))
    }

    /// Set `v`'s metatable, which for types other than table and userdata
    /// is shared by every value of that type (`lua_setmetatable`).
    pub fn set_metatable_of(self, v: Value<'gc>, mt: Option<Table<'gc>>) {
        if let Some(t) = v.get_table() {
            t.set_metatable(self, mt);
        } else if let Some(u) = v.get_userdata() {
            u.set_metatable(self.mutation, mt);
        } else {
            self.state.type_metatable(v.kind()).set(self.mutation, mt);
        }
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
        f.fetch(self.mutation, self.state.roots)
    }

    /// Parse and compile `source` into a `Function`, with `_ENV` bound to the
    /// runtime's globals table.
    pub fn load(self, source: &str, name: Option<&str>) -> Result<Function<'gc>, LoadError> {
        let mut cache = NodeCache::new();
        let parse = parser::parse(&mut cache, source);
        // The parser only reports errors (see `State::report`), so any report
        // means the tree is unusable.
        if !parse.reports.is_empty() {
            return Err(LoadError::Parse(parse.reports));
        }
        let root = parser::syntax::Root::new(parse.root)
            .ok_or(LoadError::Internal("parser did not produce a Root node"))?;
        // Like Lua's `load`, an unnamed chunk is named by its own text
        // (rendered as `[string "..."]` in messages).
        let name = LuaString::new(self, name.unwrap_or(source).as_bytes());
        let proto = compile_chunk(self, &root, &parse.lines, cache.interner(), name)?;

        // Main chunk's upvalue 0 is _ENV. Pre-close it onto globals.
        let env_uv = Gc::new(
            self.mutation,
            RefLock::new(UpvalueState::Closed(Value::table(self.state.globals))),
        );
        let mut upvalues = Vec::with_capacity_in(
            1,
            crate::dmm::allocator_api::MetricsAlloc::new(self.mutation),
        );
        upvalues.push(env_uv);
        Ok(Function::new_lua(
            self.mutation,
            proto,
            upvalues.into_boxed_slice(),
        ))
    }
}
