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
pub struct Error<'gc>(pub Value<'gc>);

impl<'gc> Error<'gc> {
    pub fn new(value: Value<'gc>) -> Self {
        Error(value)
    }

    pub fn value(self) -> Value<'gc> {
        self.0
    }

    /// Construct an error whose payload is a freshly-interned Lua string.
    /// This matches `error("msg")` in Lua source.
    pub fn from_str(ctx: Context<'gc>, msg: &str) -> Self {
        let s = LuaString::new(ctx, msg.as_bytes());
        Error(Value::string(s))
    }
}
