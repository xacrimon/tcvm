//! Chunk runners shared by the suite. Each loads every library and runs its
//! source as the chunk `=c`.

use tcvm::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Context, Executor, FromMultiValue, LoadError, Lua, RuntimeError, StashedExecutor};

/// Defines `cat(...)`, its arguments' `tostring`s joined by spaces. One line,
/// so the error positions of the code after it hold.
pub const CAT: &str = "local function cat(...) local t = table.pack(...) \
    for i = 1, t.n do t[i] = tostring(t[i]) end return table.concat(t, ' ') end ";

/// `src`'s results as `R`; panics if it raises.
pub fn eval<R: for<'gc> FromMultiValue<'gc>>(src: &str) -> R {
    let (mut lua, ex) = start(src);
    lua.execute(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"))
}

/// [`CAT`] then `src`: its string result, or the error it raised as text.
pub fn run(src: &str) -> Result<String, String> {
    let src = format!("{CAT}{src}");
    let (mut lua, ex) = start(&src);
    match lua.finish(&ex) {
        Ok(()) => Ok(lua.enter(|ctx| {
            text(
                ctx.fetch(&ex)
                    .take_result::<Value>(ctx)
                    .expect("one result"),
            )
        })),
        Err(RuntimeError::Lua(e)) => Err(lua.enter(|ctx| text(ctx.fetch(&e).value()))),
        Err(e) => panic!("unexpected failure for {src:?}: {e:?}"),
    }
}

/// [`run`]'s result; panics if `src` raises.
pub fn ok(src: &str) -> String {
    run(src).unwrap_or_else(|e| panic!("{src:?} raised {e:?}"))
}

/// [`run`]'s error; panics if `src` doesn't raise.
pub fn err(src: &str) -> String {
    run(src).expect_err(src)
}

/// Native that yields to its resumer: from the main thread, to the host.
pub fn yielder<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::yield_(None))
}

/// A `Lua` with every library and the global `yielder` bound to [`yielder`].
pub fn yielding_lua() -> Lua {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let f = Function::new_action(ctx.mutation(), yielder, &[]);
        let key = Value::string(LuaString::new(ctx, b"yielder"));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    });
    lua
}

/// `src` started on `lua` as the chunk `=c`.
pub fn start_on(lua: &mut Lua, src: &str) -> StashedExecutor {
    lua.try_enter(|ctx| -> Result<_, LoadError> {
        let chunk = ctx.load(src, Some("=c"))?;
        Ok(ctx.stash(Executor::start(ctx, chunk, ())))
    })
    .unwrap_or_else(|e| panic!("{src:?} failed to load: {e}"))
}

/// Each `--gc`-separated chunk of `src` run on `lua` to completion with a full
/// collection after it; returns the last one's string result. `yielder()`
/// collects mid-chunk.
pub fn run_chunks(lua: &mut Lua, src: &str) -> String {
    let mut result = String::new();
    for chunk in src.split("\n--gc\n") {
        let ex = start_on(lua, chunk);
        let mut step = lua.finish(&ex);
        while let Err(RuntimeError::MainYielded) = step {
            lua.collect_all();
            step = lua.resume(&ex, ());
        }
        step.expect("run");
        result = lua
            .try_enter(|ctx| {
                let r = ctx.fetch(&ex).take_result::<Option<LuaString>>(ctx)?;
                Ok::<_, RuntimeError>(r.map_or(String::new(), |s| {
                    String::from_utf8_lossy(s.as_bytes()).into_owned()
                }))
            })
            .expect("take result");
        drop(ex);
        lua.collect_all();
    }
    result
}

fn start(src: &str) -> (Lua, StashedExecutor) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(&mut lua, src);
    (lua, ex)
}

fn text(v: Value<'_>) -> String {
    let s = v.get_string().expect("a string value");
    String::from_utf8_lossy(s.as_bytes()).into_owned()
}
