use clap::Parser;
use std::fs;
use std::path::PathBuf;
use tcvm::env::{LuaString, Table, Value};
use tcvm::{Executor, Lua, format_prototype};

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

fn main() {
    let args = Args::parse();

    let source = fs::read_to_string(&args.file).unwrap();

    let mut lua = Lua::new();
    lua.load_all();

    if args.list {
        let listing = lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("test")).unwrap();
            let closure = chunk.as_lua().expect("loaded chunk must be a Lua closure");
            format_prototype(&closure.proto)
        });
        print!("{listing}");
        return;
    }

    let file_path = args.file.as_os_str().as_encoded_bytes().to_vec();
    let script_args = args.script_args.clone();

    lua.enter(|ctx| {
        let arg_tbl = Table::new(ctx);
        let path_str = LuaString::new(ctx, &file_path);
        arg_tbl.raw_set(ctx, Value::integer(0), Value::string(path_str));
        for (i, s) in script_args.iter().enumerate() {
            let v = LuaString::new(ctx, s.as_bytes());
            arg_tbl.raw_set(ctx, Value::integer((i + 1) as i64), Value::string(v));
        }
        let key = LuaString::new(ctx, b"arg");
        ctx.globals()
            .raw_set(ctx, Value::string(key), Value::table(arg_tbl));
    });

    let ex = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some("test")).unwrap();
        let executor = Executor::start(ctx, chunk, ());
        ctx.stash(executor)
    });

    lua.execute::<()>(&ex).unwrap();
}
