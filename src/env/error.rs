use crate::dmm::Collect;
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::lua::Context;

/// A Lua error in flight inside the VM. Lua's `error(v)` accepts any value, so
/// we model the carrier as a wrapped `Value<'gc>`. The host-facing
/// `RuntimeError` (in `lua/error.rs`) is the `'static`-ified version handed to
/// embedders; see `StashedError` for the bridge.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Error<'gc> {
    value: Value<'gc>,
    /// Call level whose `source:line:` should prefix a string message, with
    /// the raising native at level 0 — Lua's `error(msg, level)` /
    /// `luaL_where` convention. Applied once by `ThreadState::raise`, which
    /// resets it to 0 so re-raising along the unwind path never prefixes
    /// twice.
    level: usize,
}

impl<'gc> Error<'gc> {
    /// Raise `value` verbatim (`lua_error`).
    pub fn new(value: Value<'gc>) -> Self {
        Error { value, level: 0 }
    }

    /// Raise a message prefixed with the caller's position (`luaL_error`).
    pub fn from_str(ctx: Context<'gc>, msg: &str) -> Self {
        let s = LuaString::new(ctx, msg.as_bytes());
        Error::new(Value::string(s)).with_level(1)
    }

    pub fn with_level(self, level: usize) -> Self {
        Error { level, ..self }
    }

    pub fn value(self) -> Value<'gc> {
        self.value
    }

    pub fn level(self) -> usize {
        self.level
    }

    /// The error as a host-printable message, following `lua.c`'s
    /// `msghandler`: strings and numbers as-is, anything else by type.
    // TODO: honour `__tostring` once metamethods exist.
    pub fn message(self, ctx: Context<'gc>) -> LuaString<'gc> {
        let v = self.value;
        if v.get_string().is_some() || v.get_integer().is_some() || v.get_float().is_some() {
            crate::builtin::util::basic_tostring(ctx, v)
        } else {
            let text = format!("(error object is a {} value)", v.type_name());
            LuaString::new(ctx, text.as_bytes())
        }
    }
}
