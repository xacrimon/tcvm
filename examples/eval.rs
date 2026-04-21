//! Minimal end-to-end example.
//!
//! Usage: `cargo run --example eval -- path/to/script.lua function_name`
//!
//! The script should define a global function `function_name(a, b)` that
//! takes two integers and returns one. The example calls it with (2, 3)
//! and prints the result.

use std::env;
use std::fs;

use tcvm::env::{LuaString, Value};
use tcvm::{Executor, LoadError, Lua, RuntimeError, TypeError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("missing script path")?;
    let fn_name = args.next().unwrap_or_else(|| "main".to_string());
    let source = fs::read_to_string(&path)?;

    let mut lua = Lua::new();

    let chunk = lua.try_enter(|ctx| -> Result<_, LoadError> {
        let chunk = ctx.load(&source, Some(&path))?;
        Ok(ctx.stash(Executor::start(ctx, chunk, ())))
    })?;
    lua.execute::<()>(&chunk)?;

    let call = lua.try_enter(|ctx| -> Result<_, RuntimeError> {
        let key = Value::String(LuaString::new(ctx.mutation(), fn_name.as_bytes()));
        let func = ctx
            .globals()
            .raw_get(key)
            .get_function()
            .ok_or(TypeError::Mismatch {
                expected: "function",
                got: "nil",
            })?;
        Ok(ctx.stash(Executor::start(ctx, func, (2i64, 3i64))))
    })?;

    let result: i64 = lua.execute(&call)?;
    println!("{fn_name}(2, 3) = {result}");
    Ok(())
}
