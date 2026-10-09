use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use cstree::build::NodeCache;

use crate::compiler::compile_chunk;
use crate::dmm::{DynamicRootSet, Mutation};
use crate::env::function::{Function, UpvalueCell, UpvalueSlot};
use crate::env::shape::{MetamethodBits, Shape, inline_bucket};
use crate::env::string::Interner;
use crate::env::thread::ThreadState;
use crate::env::{LuaString, Symbols, Table, Value};
use crate::lua::stash::{Fetchable, Stashable};
use crate::lua::{LoadError, State, SyntaxError, bare_io_msg};
use crate::parser;
use crate::vm::abi::Handler;
use crate::vm::debug::chunk_id;
use crate::vm::unwind::OpError;

/// Cheap, copy handle into the runtime: one word, the `State`, whose first
/// bytes are the dispatch tables (so the interpreter's `rt` slot is also the
/// table base) and which caches the arena's mutation context.
#[derive(Copy, Clone)]
pub struct Context<'gc> {
    state: &'gc State<'gc>,
}

impl<'gc> Context<'gc> {
    pub(crate) fn new(mutation: &'gc Mutation<'gc>, state: &'gc State<'gc>) -> Self {
        state.rt.mutation.set(mutation);
        Context { state }
    }

    /// The context of the `rt` register compiled code holds.
    ///
    /// # Safety
    /// `state` is the running `State`, inside its `Lua::enter`.
    #[inline(always)]
    pub(crate) unsafe fn from_state(state: *const State<'static>) -> Self {
        Context {
            state: unsafe { &*(state as *const State<'gc>) },
        }
    }

    #[inline(always)]
    pub fn mutation(self) -> &'gc Mutation<'gc> {
        // SAFETY: set by `Context::new` inside `Lua::enter`, whose closure
        // every `Context` lives in.
        unsafe { &*self.state.rt.mutation.get() }
    }

    /// The handler of opcode byte `op`.
    #[inline(always)]
    pub(crate) fn handler(self, op: u8) -> Handler {
        // SAFETY: `State` is `repr(C)` with the 256-entry table first.
        unsafe {
            *(self.state as *const State<'gc>)
                .cast::<Handler>()
                .add(op as usize)
        }
    }

    /// The CALL continuation for `c` results wanted (0 = MULTRET).
    #[inline(always)]
    pub(crate) fn ret(self, c: u8) -> Handler {
        // SAFETY: the `rets` table follows `dispatch`.
        unsafe {
            *(self.state as *const State<'gc>)
                .cast::<Handler>()
                .add(256 + c as usize)
        }
    }

    #[inline(always)]
    pub(crate) fn jit(self) -> &'gc crate::jit::state::JitRuntime {
        &self.state.jit
    }

    /// Hot counter `i` (< `HOT_COUNTERS`).
    #[inline(always)]
    pub(crate) fn hot_counter(self, i: usize) -> &'gc std::cell::Cell<u16> {
        // SAFETY: callers pass `Instruction::hot_counter`, which is in range.
        unsafe { self.state.jit.hot.get_unchecked(i) }
    }

    /// What entry-table slot `d` tails into.
    #[inline(always)]
    pub(crate) fn jit_entry(self, d: u16) -> Handler {
        // SAFETY: a JIT word's slot is allocated while the word is in `Code`.
        unsafe { crate::jit::as_handler((*self.state.jit.entries.get().add(d as usize)).get()) }
    }

    #[inline(always)]
    pub(crate) fn thread_ptr(self) -> *mut ThreadState<'gc> {
        self.state.rt.thread.get()
    }

    #[inline(always)]
    pub(crate) fn set_thread(self, ts: *mut ThreadState<'gc>) {
        self.state.rt.thread.set(ts);
    }

    /// Whether allocation since the last check may owe the collector work.
    #[inline(always)]
    pub(crate) fn gc_due(self) -> bool {
        // SAFETY: the metrics live as long as the arena.
        unsafe { (*self.state.rt.metrics).gc_check_due() }
    }

    pub(crate) fn set_fault(self, fault: OpError<'gc>) {
        self.state.rt.fault.set(Some(fault));
    }

    pub(crate) fn take_fault(self) -> OpError<'gc> {
        self.state
            .rt
            .fault
            .take()
            .expect("impl_error without a pending fault")
    }

    pub fn globals(self) -> Table<'gc> {
        self.state.globals
    }

    /// The root shape of tables with no inline slots, where `{}` starts.
    /// Stable for the lifetime of the runtime.
    pub fn empty_shape(self) -> Shape<'gc> {
        self.state.root_shapes[0]
    }

    /// The root shape of tables with room for `n` inline slots, or for as
    /// many as tables hold inline.
    pub fn root_shape(self, n: usize) -> Shape<'gc> {
        self.state.root_shapes[inline_bucket(n)]
    }

    /// The `next` that `pairs` returns (see `State::next`).
    #[inline]
    pub(crate) fn next_fn(self) -> Function<'gc> {
        self.state.next
    }

    /// The iterator `ipairs` returns (see `State::next`).
    #[inline]
    pub(crate) fn ipairs_iter(self) -> Function<'gc> {
        self.state.ipairs_iter
    }

    /// See `State::unwind`.
    pub(crate) fn unwind_fn(self) -> Function<'gc> {
        self.state.unwind
    }

    /// A fresh epoch for an async native's locals, never 0.
    pub(crate) fn next_epoch(self) -> u32 {
        let e = self.state.epoch.get().wrapping_add(1).max(1);
        self.state.epoch.set(e);
        e
    }

    /// Make `waker` the step's (see `waker`), returning the one it replaces.
    pub(crate) fn set_waker(self, waker: *const std::task::Waker) -> *const std::task::Waker {
        self.state.waker.replace(waker)
    }

    /// The waker of the step in progress.
    pub(crate) fn waker(self) -> &'gc std::task::Waker {
        let w = self.state.waker.get();
        if w.is_null() {
            std::task::Waker::noop()
        } else {
            // SAFETY: set by `Executor::step` for its duration.
            unsafe { &*w }
        }
    }

    /// Run `f` on a byte buffer for building a string, empty, and keep its
    /// storage for the next one, so building a string allocates only the
    /// string. A nested use gets a fresh buffer.
    #[inline]
    pub(crate) fn with_buf<R>(self, f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
        let mut buf = self.state.buf.take();
        buf.clear();
        let r = f(&mut buf);
        // Not to hold on to the memory of one huge string.
        if buf.capacity() <= 1 << 16 {
            self.state.buf.set(buf);
        }
        r
    }

    /// Intern the string `f` writes, built in the buffer of [`with_buf`](Self::with_buf).
    #[inline]
    pub(crate) fn build_string(self, f: impl FnOnce(&mut Vec<u8>)) -> LuaString<'gc> {
        self.with_buf(|buf| {
            f(buf);
            LuaString::new(self, buf)
        })
    }

    /// Dict-mode sentinel for tables with no metatable. Tables that
    /// migrate to dict mode while carrying a metatable use the
    /// per-`MtCache` sentinel instead.
    #[inline]
    pub fn empty_dict_sentinel(self) -> Shape<'gc> {
        self.state.empty_dict_sentinel
    }

    #[inline]
    pub(crate) fn mt_classes(self) -> crate::env::shape::MtClasses<'gc> {
        self.state.mt_classes
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

    /// The metatable numbers share, usually none.
    #[inline]
    pub(crate) fn number_metatable(self) -> Option<Table<'gc>> {
        self.state
            .type_metatable(crate::lua::ValueKind::Float)
            .get()
    }

    /// Metamethod `name` of `v`, nil when absent (`luaT_gettmbyobj`).
    #[inline]
    pub fn metamethod_of(self, v: Value<'gc>, name: LuaString<'gc>) -> Value<'gc> {
        self.metatable_of(v)
            .map_or(Value::nil(), |mt| mt.raw_get(Value::string(name)))
    }

    /// Metamethod `bit` of `v`, nil when absent, read from the metatable's
    /// cache rather than looked up by name.
    #[inline]
    pub fn mm_of(self, v: Value<'gc>, bit: MetamethodBits) -> Value<'gc> {
        let cache = if let Some(t) = v.get_table() {
            t.meta(self)
        } else {
            let Some(mt) = self.metatable_of(v) else {
                return Value::nil();
            };
            Some(mt.ensure_mt_cache(self))
        };
        let mm = cache.map_or(Value::nil(), |c| c.mm(bit));
        #[cfg(debug_assertions)]
        if let Some(mt) = self.metatable_of(v) {
            let name = self.symbols().metamethods()[bit.bits().trailing_zeros() as usize].0;
            let raw = mt.raw_get(Value::string(name));
            debug_assert!(
                mm.same_bits(&raw),
                "stale metamethod cache for {}",
                String::from_utf8_lossy(name.as_bytes())
            );
        }
        mm
    }

    /// Set `v`'s metatable, which for types other than table and userdata
    /// is shared by every value of that type (`lua_setmetatable`).
    pub fn set_metatable_of(self, v: Value<'gc>, mt: Option<Table<'gc>>) {
        if let Some(t) = v.get_table() {
            t.set_metatable(self, mt);
        } else if let Some(u) = v.get_userdata() {
            u.set_metatable(self.mutation(), mt);
        } else {
            self.state.type_metatable(v.kind()).set(self.mutation(), mt);
        }
    }

    pub fn roots(self) -> DynamicRootSet<'gc> {
        self.state.roots
    }

    pub(crate) fn interner(&self) -> &Interner<'gc> {
        &self.state.interner
    }

    pub fn stash<S: Stashable<'gc>>(self, s: S) -> S::Stashed {
        s.stash(self.mutation(), self.state.roots)
    }

    pub fn fetch<F: Fetchable>(self, f: &F) -> F::Fetched<'gc> {
        f.fetch(self.mutation(), self.state.roots)
    }

    /// Parse and compile `source` into a `Function`, with `_ENV` bound to the
    /// runtime's globals table.
    pub fn load(self, source: &str, name: Option<&str>) -> Result<Function<'gc>, LoadError> {
        // Like Lua's `load`, an unnamed chunk is named by its own text
        // (rendered as `[string "..."]` in messages).
        let name = LuaString::new(self, name.unwrap_or(source).as_bytes());
        self.load_text(source, name, Value::table(self.state.globals))
    }

    /// Load the file at `path` as Lua's `loadfile` does: named `@path`, with
    /// a UTF-8 BOM and a first line starting with `#` skipped.
    pub fn load_file(self, path: &Path) -> Result<Function<'gc>, LoadError> {
        let path = path.as_os_str().as_bytes();
        self.load_file_with(Some(path), Value::table(self.state.globals))
    }

    /// `luaL_loadfilex`, reading stdin without a `path`, then
    /// [`load_bytes`](Self::load_bytes). A skipped `#` line leaves its
    /// newline, so line numbers hold.
    pub(crate) fn load_file_with(
        self,
        path: Option<&[u8]>,
        env: Value<'gc>,
    ) -> Result<Function<'gc>, LoadError> {
        let mut contents = Vec::new();
        let name = match path {
            Some(p) => {
                let shown = String::from_utf8_lossy(p);
                let mut file = File::open(Path::new(OsStr::from_bytes(p))).map_err(|e| {
                    LoadError::File(format!("cannot open {shown}: {}", bare_io_msg(&e)))
                })?;
                // `luaL_loadfilex` clears `errno` before reporting a read error.
                file.read_to_end(&mut contents)
                    .map_err(|_| LoadError::File(format!("cannot read {shown}")))?;
                [b"@", p].concat()
            }
            None => {
                std::io::stdin()
                    .read_to_end(&mut contents)
                    .map_err(|_| LoadError::File("cannot read stdin".into()))?;
                b"=stdin".to_vec()
            }
        };
        let body = contents.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&contents);
        let source = match body.strip_prefix(b"#") {
            Some(line) => line
                .iter()
                .position(|&b| b == b'\n')
                .map_or(&[][..], |i| &line[i..]),
            None => body,
        };
        self.load_bytes(source, LuaString::new(self, &name), env)
    }

    /// `luaL_loadbuffer` for a text chunk, which must be UTF-8.
    pub(crate) fn load_bytes(
        self,
        source: &[u8],
        name: LuaString<'gc>,
        env: Value<'gc>,
    ) -> Result<Function<'gc>, LoadError> {
        let source = std::str::from_utf8(source).map_err(|_| LoadError::NotUtf8 {
            chunk: String::from_utf8_lossy(&chunk_id(name.as_bytes())).into_owned(),
        })?;
        self.load_text(source, name, env)
    }

    fn load_text(
        self,
        source: &str,
        name: LuaString<'gc>,
        env: Value<'gc>,
    ) -> Result<Function<'gc>, LoadError> {
        let mut cache = NodeCache::new();
        let parse = parser::parse(&mut cache, source);
        let chunk = || String::from_utf8_lossy(&chunk_id(name.as_bytes())).into_owned();
        // The parser only reports errors (see `State::error`), so any report
        // means the tree is unusable.
        if !parse.reports.is_empty() {
            return Err(LoadError::Parse(SyntaxError {
                chunk: chunk(),
                source: source.to_owned(),
                reports: parse.reports,
            }));
        }
        let root = parser::syntax::Root::new(parse.root)
            .ok_or(LoadError::Internal("parser did not produce a Root node"))?;
        let proto =
            compile_chunk(self, &root, &parse.lines, cache.interner(), name).map_err(|error| {
                LoadError::Compile {
                    chunk: chunk(),
                    error,
                }
            })?;

        // Main chunk's upvalue 0 is _ENV, its only one.
        debug_assert_eq!(proto.num_upvalues, 1);
        let env = if proto.upvalue_desc[0].by_value {
            UpvalueSlot { value: env }
        } else {
            UpvalueSlot {
                cell: UpvalueCell::new_closed(self.mutation(), env),
            }
        };
        Ok(Function::new_lua(self.mutation(), proto, |dst| unsafe {
            dst.write(env)
        }))
    }
}
