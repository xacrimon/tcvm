use std::io::IsTerminal;
use std::path::PathBuf;
use std::pin::Pin;

use clap::Parser;
use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use tcvm::vm::sequence::{
    BoxSequence, CallbackAction, Execution, Sequence, SequencePoll, seq_trace_pointers,
};
use tcvm::{Executor, LoadError, Lua, RuntimeError, format_prototype};

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
        let handler = Function::new_native(ctx.mutation(), msghandler as NativeFn, &[]);
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
        Ok(Some(err)) | Err(RuntimeError::Lua(err)) => lua.enter(|ctx| {
            String::from_utf8_lossy(ctx.fetch(&err).message(ctx).as_bytes()).into_owned()
        }),
        Err(e) => e.to_string(),
    };
    eprintln!("tcvm: {err}");
    std::process::exit(1);
}

/// `lua.c`'s `msghandler`, short of the traceback (#228): the error object
/// as text, through its `__tostring` when that returns a string. It runs
/// before the chunk unwinds, so ahead of any pending `__close`.
fn msghandler<'gc>(
    ctx: tcvm::Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let err = Error::new(ctx, v);
    if err.as_text(ctx).is_none() {
        let mm = ctx.metamethod_of(v, ctx.symbols().mm_tostring);
        if !mm.is_nil() {
            stack.replace(&[mm, v]);
            let then = BoxSequence::new(ctx.mutation(), StringOrMessage(v));
            return Ok(CallbackAction::call(Some(then)));
        }
    }
    stack.ret1(Value::string(err.message(ctx)));
    Ok(CallbackAction::Return)
}

/// [`msghandler`] after `__tostring`: its result if that is a string, else
/// the error object's [`Error::message`]. An error it raises re-enters
/// `msghandler` instead.
#[derive(Collect)]
#[collect(no_drop)]
struct StringOrMessage<'gc>(Value<'gc>);

impl<'gc> Sequence<'gc> for StringOrMessage<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        ctx: tcvm::Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let r = stack.get(0);
        if r.get_string().is_some() {
            stack.ret1(r);
        } else {
            stack.ret1(Value::string(Error::new(ctx, self.0).message(ctx)));
        }
        Ok(SequencePoll::Return)
    }
}
