use crate::Context;
use crate::builtin::util;
use crate::env::{LuaString, Table, Value};

// See #27: constants/tables — config, cpath, loaded, path, preload, searchers

pub fn load<'gc>(ctx: Context<'gc>) {
    let lib = Table::new(ctx);
    util::set_not_implemented(ctx, lib, "package", &["loadlib", "searchpath"]);

    let lib_name = Value::string(LuaString::new(ctx, b"package"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));

    let require = util::not_implemented(ctx, "require");
    let require_key = Value::string(LuaString::new(ctx, b"require"));
    ctx.globals()
        .raw_set(ctx, require_key, Value::function(require));
}
