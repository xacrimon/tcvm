//! A `Lua` moved to another thread (`Lua::into_send`) goes on where it was,
//! with a coroutine suspended inside compiled code.

use tcvm::Lua;
use tcvm::env::Value;

use crate::common::start_on;

fn run(lua: &mut Lua, src: &str) -> String {
    let ex = start_on(lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    })
}

#[test]
fn moves_with_a_suspended_coroutine() {
    let mut lua = Lua::new();
    lua.load_all();
    let first = run(
        &mut lua,
        r#"
        local function gen()
          local s = 0
          for i = 1, 1e6 do
            s = s + i % 7
            if i % 1000 == 0 then coroutine.yield(s) end
          end
          return -1
        end
        co = coroutine.wrap(gen)
        local r
        for _ = 1, 400 do r = co() end
        return tostring(r)
        "#,
    );
    assert_eq!(first, "1200003");
    // SAFETY: no stash handle, userdata or async native outlives the move.
    let moved = unsafe { lua.into_send() };
    let (second, errors) = std::thread::spawn(move || {
        let mut lua = moved.into_inner();
        let r = run(
            &mut lua,
            "local r; for _ = 1, 400 do r = co() end; return tostring(r)",
        );
        (r, lua.jit_internal_errors())
    })
    .join()
    .expect("the moved Lua ran");
    assert_eq!(second, "2400000");
    assert_eq!(errors, 0);
}
