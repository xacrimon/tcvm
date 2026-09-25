use std::cell::Cell;

use crate::dmm::{Collect, Gc};
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::lua::Context;

/// A Lua error in flight inside the VM. Lua's `error(v)` accepts any value, so
/// we model the carrier as a wrapped `Value<'gc>`. The host-facing
/// `RuntimeError` (in `lua/error.rs`) is the `'static`-ified version handed to
/// embedders; see `StashedError` for the bridge.
///
/// One GC pointer: with `CallbackAction` also pointer-sized, a native's
/// `Result<CallbackAction, Error>` is a (tag, pointer) pair and comes back in
/// registers. An error allocates, but so does the message it usually carries.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Error<'gc>(Gc<'gc, ErrorInner<'gc>>);

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ErrorInner<'gc> {
    value: Value<'gc>,
    /// Call level whose `source:line:` should prefix a string message, with
    /// the raising native at level 0 — Lua's `error(msg, level)` /
    /// `luaL_where` convention. Applied once by `ThreadState::raise`, which
    /// resets it to 0 so re-raising along the unwind path never prefixes
    /// twice. Saturated from `usize`: any level past the stack bottom names
    /// no frame.
    #[collect(require_static)]
    level: Cell<u32>,
    /// Set once an `xpcall` message handler has run for this error, so the
    /// unwinder doesn't run it again as the (possibly transformed) error
    /// continues to the catcher.
    #[collect(require_static)]
    handled: Cell<bool>,
    #[collect(require_static)]
    exit: Cell<Exit>,
}

/// Whether an error is an exit, which unwinds to the thread's base level past
/// every other catcher (`luaD_throwbaselevel`); see [`Error::exit`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    No,
    /// Ends the thread without an error.
    Clean,
    /// Ends the thread with the error's value.
    Failed,
}

impl<'gc> Error<'gc> {
    /// Raise `value` verbatim (`lua_error`).
    pub fn new(ctx: Context<'gc>, value: Value<'gc>) -> Self {
        Error(Gc::new(
            ctx.mutation(),
            ErrorInner {
                value,
                level: Cell::new(0),
                handled: Cell::new(false),
                exit: Cell::new(Exit::No),
            },
        ))
    }

    pub(crate) fn from_inner(inner: Gc<'gc, ErrorInner<'gc>>) -> Self {
        Error(inner)
    }

    pub(crate) fn inner(self) -> Gc<'gc, ErrorInner<'gc>> {
        self.0
    }

    /// Raise a message prefixed with the caller's position (`luaL_error`).
    ///
    /// Out of line: it is the error tail of nearly every builtin, and inlined
    /// it drags string interning and allocation into their hot bodies.
    #[cold]
    #[inline(never)]
    pub fn from_str(ctx: Context<'gc>, msg: &str) -> Self {
        let s = LuaString::new(ctx, msg.as_bytes());
        Error::new(ctx, Value::string(s)).with_level(1)
    }

    /// Errors are linear (raised once, then unwound), so this updates the
    /// shared object rather than allocating a copy.
    pub fn with_level(self, level: usize) -> Self {
        self.0.level.set(u32::try_from(level).unwrap_or(u32::MAX));
        self
    }

    pub fn value(self) -> Value<'gc> {
        self.0.value
    }

    pub fn level(self) -> usize {
        self.0.level.get() as usize
    }

    /// The error value as `lua_tostring` sees it: a string, a number as a
    /// string, or `None` for anything else.
    pub fn as_text(self, ctx: Context<'gc>) -> Option<LuaString<'gc>> {
        let v = self.value();
        (v.get_string().is_some() || v.get_integer().is_some() || v.get_float().is_some())
            .then(|| crate::builtin::util::basic_tostring(ctx, v))
    }

    /// The error as a host-printable message, following `lua.c`'s
    /// `msghandler` short of calling `__tostring`: [`as_text`](Self::as_text),
    /// else the value's type.
    pub fn message(self, ctx: Context<'gc>) -> LuaString<'gc> {
        self.as_text(ctx).unwrap_or_else(|| {
            let text = format!("(error object is a {} value)", self.value().type_name());
            LuaString::new(ctx, text.as_bytes())
        })
    }

    /// Swap the payload, keeping the handled flag; resets the level. The
    /// payload is not a cell (storing a `Gc` would need a barrier), so this
    /// allocates a fresh carrier.
    pub(crate) fn with_value(self, ctx: Context<'gc>, value: Value<'gc>) -> Self {
        let err = Error::new(ctx, value);
        err.0.handled.set(self.is_handled());
        err
    }

    pub(crate) fn mark_handled(self) -> Self {
        self.0.handled.set(true);
        self
    }

    pub(crate) fn is_handled(self) -> bool {
        self.0.handled.get()
    }

    /// Unwind the running thread to its base level past every other catcher
    /// and message handler (`luaD_throwbaselevel`), ending it with `err`, or
    /// cleanly without one.
    pub(crate) fn exit(ctx: Context<'gc>, err: Option<Error<'gc>>) -> Self {
        let exit = Error::new(ctx, err.map_or(Value::nil(), Error::value)).mark_handled();
        exit.0.exit.set(match err {
            Some(_) => Exit::Failed,
            None => Exit::Clean,
        });
        exit
    }

    pub(crate) fn exit_kind(self) -> Exit {
        self.0.exit.get()
    }
}
