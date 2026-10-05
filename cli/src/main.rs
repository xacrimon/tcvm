use std::io::IsTerminal;
use std::path::PathBuf;

use clap::Parser;
use tcvm::env::{Error, Function, LuaString, NativeClosure, Stack, Table, Value};
use tcvm::vm::async_native::Spawned;
use tcvm::{Executor, LoadError, Lua, RuntimeError, StashedExecutor, format_prototype};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: PathBuf,

    #[arg(short = 'l', long)]
    list: bool,

    #[arg(trailing_var_arg = true)]
    script_args: Vec<String>,
}

/// Print a clean diagnostic to stderr and exit with failure — used for
/// load/compile errors so a Lua source error doesn't surface as a panic.
fn die(e: &dyn std::fmt::Display) -> ! {
    eprintln!("tcvm: {e}");
    std::process::exit(1);
}

/// [`die`] for a load error; syntax errors are drawn in color on a terminal.
fn die_load(e: &LoadError) -> ! {
    match e {
        LoadError::Parse(e) => die(&e.render(std::io::stderr().is_terminal())),
        e => die(e),
    }
}

fn main() {
    let args = Args::parse();

    let mut lua = Lua::new();
    lua.load_all();

    if args.list {
        let listing = lua.enter(|ctx| {
            let chunk = ctx.load_file(&args.file)?;
            let closure = chunk.as_lua().expect("loaded chunk must be a Lua closure");
            Ok::<_, LoadError>(format_prototype(&closure.proto))
        });
        match listing {
            Ok(listing) => print!("{listing}"),
            Err(e) => die_load(&e),
        }
        return;
    }

    let file_path = args.file.as_os_str().as_encoded_bytes().to_vec();
    let script_args = args.script_args.clone();

    lua.enter(|ctx| {
        let arg_tbl = Table::new(ctx);
        let path_str = LuaString::new(ctx, &file_path);
        arg_tbl.raw_set(
            ctx,
            Value::integer(ctx.mutation(), 0),
            Value::string(path_str),
        );
        for (i, s) in script_args.iter().enumerate() {
            let v = LuaString::new(ctx, s.as_bytes());
            arg_tbl.raw_set(
                ctx,
                Value::integer(ctx.mutation(), (i + 1) as i64),
                Value::string(v),
            );
        }
        let key = LuaString::new(ctx, b"arg");
        ctx.globals()
            .raw_set(ctx, Value::string(key), Value::table(arg_tbl));
    });

    // `lua.c`'s `docall`: the chunk runs under `xpcall` with `msghandler`.
    // TODO(#228): an `Executor`-level message handler instead of the global.
    let ex = lua.enter(|ctx| {
        let chunk = ctx.load_file(&args.file)?;
        let xpcall = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, b"xpcall")));
        let handler = Function::new_async(ctx.mutation(), msghandler, &[]);
        let executor = Executor::start(ctx, xpcall, (chunk, handler));
        Ok::<_, LoadError>(ctx.stash(executor))
    });
    let ex = match ex {
        Ok(ex) => ex,
        Err(e) => die_load(&e),
    };

    let outcome = lua.finish(&ex).and_then(|()| {
        lua.try_enter(|ctx| {
            let (ok, err): (bool, Value) = ctx.fetch(&ex).take_result(ctx)?;
            Ok((!ok).then(|| ctx.stash(Error::new(ctx, err))))
        })
    });
    let err = match outcome {
        Ok(None) => return,
        Err(RuntimeError::Exit(code)) => exit(lua, ex, code),
        Ok(Some(err)) | Err(RuntimeError::Lua(err)) => lua.enter(|ctx| {
            String::from_utf8_lossy(ctx.fetch(&err).message(ctx).as_bytes()).into_owned()
        }),
        Err(e) => e.to_string(),
    };
    eprintln!("tcvm: {err}");
    exit(lua, ex, 1);
}

/// End the process once `lua` is dropped, which flushes and closes its files
/// as C's `exit` does for every `FILE*`.
fn exit(lua: Lua, ex: StashedExecutor, code: i32) -> ! {
    drop(ex);
    drop(lua);
    std::process::exit(code);
}

/// `lua.c`'s `msghandler`, short of the traceback (#228): the error object
/// as text, through its `__tostring` when that returns a string. It runs
/// before the chunk unwinds, so ahead of any pending `__close`. An error in
/// `__tostring` re-enters it instead.
fn msghandler<'gc>(
    _ctx: tcvm::Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        let call = cx.enter(|ctx, mut stack| {
            let v = stack.get(0);
            if Error::new(ctx, v).as_text(ctx).is_none() {
                let mm = ctx.metamethod_of(v, ctx.symbols().mm_tostring);
                if !mm.is_nil() {
                    stack.truncate(1);
                    stack.extend([mm, v]);
                    return true;
                }
            }
            stack.ret1(Value::string(Error::new(ctx, v).message(ctx)));
            false
        });
        if call {
            cx.call(1).await;
            cx.enter(|ctx, mut stack| {
                let r = stack.get(1);
                let s = match r.get_string() {
                    Some(_) => r,
                    None => Value::string(Error::new(ctx, stack.get(0)).message(ctx)),
                };
                stack.ret1(s);
            });
        }
        Ok(())
    }))
}
