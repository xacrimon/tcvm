use std::fs;
use std::path::PathBuf;

use clap::Parser;
use tcvm::env::{LuaString, Table, Value};
use tcvm::{Executor, LoadError, Lua, RuntimeError, StashedError, format_prototype};

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

fn main() {
    let args = Args::parse();

    let source = fs::read_to_string(&args.file).unwrap();

    // Lua's `@` prefix marks a chunk name as a file path (`luaO_chunkid`).
    let chunk_name = format!("@{}", args.file.display());

    let mut lua = Lua::new();
    lua.load_all();

    if args.list {
        let listing = lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some(&chunk_name))?;
            let closure = chunk.as_lua().expect("loaded chunk must be a Lua closure");
            Ok::<_, LoadError>(format_prototype(&closure.proto))
        });
        match listing {
            Ok(listing) => print!("{listing}"),
            Err(e) => die(&e),
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

    let ex = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some(&chunk_name))?;
        let executor = Executor::start(ctx, chunk, ());
        Ok::<_, LoadError>(ctx.stash(executor))
    });
    let ex = match ex {
        Ok(ex) => ex,
        Err(e) => die(&e),
    };

    if let Err(e) = lua.execute::<()>(&ex) {
        match e {
            RuntimeError::Lua(stashed) => {
                let msg = error_message(&mut lua, &stashed);
                eprintln!("tcvm: {msg}");
            }
            other => eprintln!("tcvm: {other}"),
        }
        std::process::exit(1);
    }
}

/// `lua.c`'s `msghandler`: an error object that isn't a string or number is
/// reported through its `__tostring` when that returns a string, and an error
/// raised by `__tostring` replaces the original.
fn error_message(lua: &mut Lua, err: &StashedError) -> String {
    let lossy = |s: LuaString<'_>| String::from_utf8_lossy(s.as_bytes()).into_owned();
    let call = lua.enter(|ctx| {
        let v = ctx.fetch(err).value();
        if v.get_string().is_some() || v.get_integer().is_some() || v.get_float().is_some() {
            return None;
        }
        let mm = ctx.metamethod_of(v, ctx.symbols().mm_tostring);
        if mm.is_nil() {
            return None;
        }
        Some(ctx.stash(Executor::start(ctx, mm, (v,))))
    });
    if let Some(ex) = call {
        match lua.finish(&ex) {
            Ok(()) => {
                let msg = lua.enter(|ctx| {
                    let r = ctx.fetch(&ex).take_result::<Value>(ctx).ok()?;
                    r.get_string().map(lossy)
                });
                if let Some(msg) = msg {
                    return msg;
                }
            }
            Err(RuntimeError::Lua(inner)) => {
                return lua.enter(|ctx| lossy(ctx.fetch(&inner).message(ctx)));
            }
            Err(_) => {}
        }
    }
    lua.enter(|ctx| lossy(ctx.fetch(err).message(ctx)))
}
