use clap::Parser;
use std::fs;
use std::path::PathBuf;
use tcvm::{Executor, Lua, format_prototype};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: PathBuf,

    #[arg(short = 'l', long)]
    list: bool,
}

fn main() {
    let args = Args::parse();

    let source = fs::read_to_string(&args.file).unwrap();

    let mut lua = Lua::new();
    lua.load_all();

    if args.list {
        let listing = lua.enter(|ctx| {
            let chunk = ctx.load(&source, Some("test")).unwrap();
            let closure = chunk
                .as_lua()
                .expect("loaded chunk must be a Lua closure");
            format_prototype(&closure.proto)
        });
        print!("{listing}");
        return;
    }

    let ex = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some("test")).unwrap();
        let executor = Executor::start(ctx, chunk, ());
        ctx.stash(executor)
    });

    lua.execute::<()>(&ex).unwrap();
}
