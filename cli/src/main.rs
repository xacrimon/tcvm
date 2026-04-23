use clap::Parser;
use std::path::PathBuf;
use std::fs;
use tcvm::{Lua, Executor};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: PathBuf,
}

fn main() {
    let args = Args::parse();

    let source = fs::read_to_string(&args.file).unwrap();

    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua.enter(|ctx| {
        let chunk = ctx.load(&source, Some("test")).unwrap();
        let executor = Executor::start(ctx, chunk, ());
        ctx.stash(executor)
    });

    lua.execute::<()>(&ex).unwrap();
}
