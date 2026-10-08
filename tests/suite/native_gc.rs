//! A native that allocates leaves for the collector when it lands, as the
//! fast entries do; its results wait in the frame and land afterwards. Each
//! loop allocates only inside natives, on one of the landing paths: a plain
//! call, a tail call, a continuation's return, and the boxed-integer
//! arithmetic slow path.

use tcvm::{Executor, Lua, StashedExecutor, StepResult};

fn start(lua: &mut Lua, src: &str) -> StashedExecutor {
    lua.try_enter(|ctx| -> Result<_, tcvm::LoadError> {
        let chunk = ctx.load(src, Some("=c"))?;
        Ok(ctx.stash(Executor::start(ctx, chunk, ())))
    })
    .unwrap_or_else(|e| panic!("{src:?} failed to load: {e}"))
}

/// Run `prologue` to completion and collect, then `src` step by step: its
/// integer result and how often dispatch left for the collector.
fn run_counting(prologue: &str, src: &str) -> (i64, usize) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start(&mut lua, prologue);
    lua.finish(&ex).expect("prologue");
    lua.collect_all();
    let ex = start(&mut lua, src);
    let mut exits = 0;
    loop {
        let done = lua.enter(|ctx| match ctx.fetch(&ex).step(ctx).expect("step") {
            StepResult::Done => true,
            StepResult::Pending => false,
            StepResult::Yielded(_) => panic!("yielded"),
        });
        if done {
            break;
        }
        exits += 1;
    }
    let r = lua.enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx).expect("result"));
    (r, exits)
}

fn check(name: &str, prologue: &str, src: &str, want: i64) {
    let (r, exits) = run_counting(prologue, src);
    assert_eq!(r, want, "{name}: wrong result");
    assert!(exits > 0, "{name}: never left for the collector");
}

#[test]
fn plain_native_call() {
    check(
        "tostring",
        "",
        "local n = 0
         for i = 1, 300000 do n = n + #tostring(i) end
         return n",
        1_688_895,
    );
    check(
        "table.pack",
        "",
        "local pack, n = table.pack, 0
         for i = 1, 300000 do n = n + pack(i, i).n end
         return n",
        600_000,
    );
}

#[test]
fn native_tail_call() {
    check(
        "tail tostring",
        "function f(i) return tostring(i) end",
        "local f, n = f, 0
         for i = 1, 300000 do n = n + #f(i) end
         return n",
        1_688_895,
    );
}

#[test]
fn continuation_return() {
    // Only gsub's result, built after the replacement function's calls,
    // allocates: the callback hands back strings made in the prologue.
    check(
        "gsub",
        "names = {}
         for i = 1, 100000 do names[i] = tostring(i) end
         cur = 0
         function rep(c) if c == 'a' then return names[cur] end return 'z' end",
        "local gsub, rep, n = string.gsub, rep, 0
         for i = 1, 100000 do cur = i; n = n + #gsub('a-b', '%a', rep) end
         return n",
        688_895,
    );
}

#[test]
fn boxed_arithmetic() {
    check(
        "boxed add",
        "",
        "local x = 1 << 40
         for i = 1, 300000 do x = x + 1 end
         return x - (1 << 40)",
        300000,
    );
}
