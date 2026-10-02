//! A builtin that calls Lua through a follow-up sequence (`pcall`) works as a
//! metamethod or generic-for iterator: the sequence's results reach the
//! caller's continuation. Expected strings come from `lua` 5.5.1 on the same
//! chunk, which hands its result to the host with `error(v, 0)`.

use tcvm::{Executor, LoadError, Lua, RuntimeError};

fn raised(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Err(RuntimeError::Lua(stashed)) => lua.enter(|ctx| {
            let s = ctx.fetch(&stashed).value().get_string().expect("string");
            String::from_utf8_lossy(s.as_bytes()).into_owned()
        }),
        other => panic!("expected a Lua error, got {other:?}"),
    }
}

/// One case per continuation kind: `__index` stores the first result,
/// `__newindex` drops it, the iterator fills the loop variables and `__lt`
/// branches on it.
#[test]
fn pcall_as_metamethod_and_iterator() {
    let src = r#"
        local out, log = {}, nil
        local t = setmetatable({}, {
          __index = pcall,
          __newindex = pcall,
          __call = function(self, k, v) if v then log = k .. "=" .. v end return k .. "!" end,
        })
        out[#out + 1] = tostring(t.x)
        t.y = 2
        out[#out + 1] = log
        for a, b in pcall, function() return 1 end do out[#out + 1] = tostring(a) .. " " .. tostring(b) break end
        local u = setmetatable({}, {__lt = pcall})
        out[#out + 1] = tostring(u < u)
        error(table.concat(out, ", "), 0)
    "#;
    assert_eq!(raised(src), "true, y=2, true 1, false");
}
