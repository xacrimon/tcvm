//! `__close` on to-be-closed variables (#45): scope exits, errors, message
//! handlers and yields. Expected strings come from `lua` 5.5.1 running the
//! same chunk (`=c`) after the one-line prelude.

use tcvm::env::Value;
use tcvm::{Executor, LoadError, Lua};

/// `out` logs its arguments; `mk(name, fail)` makes a value whose `__close`
/// logs its name, argument count and error, then raises `fail` if given.
const PRELUDE: &str = "local log = {} local function out(...) local t = table.pack(...) for i = 1, t.n do t[i] = tostring(t[i]) end log[#log + 1] = table.concat(t, ' ') end local function mk(name, fail) return setmetatable({}, {__close = function(...) local _, e = ... out(name, select('#', ...), e) if fail then error(fail, 0) end end}) end ";

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let src = format!("{PRELUDE}{src}");
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} raised {e:?}"));
    lua.enter(|ctx| {
        let v = ctx.fetch(&ex).take_result::<Value>(ctx).expect("result");
        String::from_utf8_lossy(v.get_string().expect("string").as_bytes()).into_owned()
    })
}

#[test]
fn block_exit_reverse_order() {
    let src = "do\n  local a <close> = mk('a')\n  local b <close> = mk('b')\n  out('body')\nend\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "body|b 1 nil|a 1 nil");
}

#[test]
fn nil_and_false_skipped() {
    let src = "do local a <close> = nil local b <close> = false end\nreturn 'ok'";
    assert_eq!(run(src), "ok");
}

#[test]
fn return_values_survive() {
    let src = "local function f() local a <close> = mk('a') return 1, 2, 3 end\nout(f())\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "a 1 nil|1 2 3");
}

#[test]
fn multret_return_survives() {
    let src = "local function g() return 'x', 'y', 'z' end\nlocal function f(...) local a <close> = mk('a') return ... end\nlocal function h() local a <close> = mk('h') return g() end\nout(f(1, 2, 3))\nout(h())\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "a 1 nil|1 2 3|h 1 nil|x y z");
}

#[test]
fn break_and_goto() {
    let src = "for i = 1, 3 do\n  local a <close> = mk('loop' .. i)\n  if i == 2 then break end\nend\ndo\n  local a <close> = mk('goto')\n  goto out\nend\n::out::\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "loop1 1 nil|loop2 1 nil|goto 1 nil");
}

#[test]
fn close_values_ignored_and_swapped() {
    let src = "local mt = {__close = function() out('orig') return 'ignored' end}\nlocal function f()\n  local x <close> = setmetatable({}, mt)\n  mt.__close = function() out('swapped') end\n  return 'real'\nend\nout(f())\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "swapped|real");
}

#[test]
fn callable_close() {
    let src = "do\n  local callable = setmetatable({}, {__call = function(self, o, ...) out('callable', select('#', ...)) end})\n  local x <close> = setmetatable({}, {__close = callable})\nend\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "callable 0");
}

#[test]
fn non_closable_names_variable() {
    let src = "out(pcall(function() local x <close> = {} end))\nout(pcall(function()\n  local a <close> = mk('a')\n  local b <close> = 1\nend))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "false c:1: variable 'x' got a non-closable value|a 2 c:4: variable 'b' got a non-closable value|false c:4: variable 'b' got a non-closable value"
    );
}

#[test]
fn uncallable_close() {
    let src = "out(pcall(function()\n  local a <close> = mk('a')\n  local mt = {__close = function() end}\n  local x <close> = setmetatable({}, mt)\n  mt.__close = nil\nend))\nout(pcall(function()\n  local a <close> = mk('a')\n  local mt = {__close = function() end}\n  local x <close> = setmetatable({}, mt)\n  mt.__close = 42\n  error('E', 0)\nend))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "a 2 c:6: attempt to call a nil value (metamethod 'close')|false c:6: attempt to call a nil value (metamethod 'close')|a 2 attempt to call a number value|false attempt to call a number value"
    );
}

#[test]
fn error_passes_error_object() {
    let src = "out(pcall(function() local a <close> = mk('a') error('E1', 0) end))\nout(pcall(function() local a <close> = mk('nil') error(nil) end))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "a 2 E1|false E1|nil 2 <no error object>|false <no error object>"
    );
}

#[test]
fn error_in_close_on_normal_exit() {
    let src = "out(pcall(function()\n  local a <close> = mk('a')\n  local b <close> = mk('b', 'EB')\n  local c <close> = mk('c')\n  return 'unreached'\nend))\nout(pcall(function()\n  do local a <close> = mk('a', 'EA') end\n  out('unreached')\nend))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "c 1 nil|b 1 nil|a 2 EB|false EB|a 1 nil|false EA");
}

#[test]
fn errors_chain_while_unwinding() {
    let src = "out(pcall(function()\n  local a <close> = mk('a', 'EA')\n  local b <close> = mk('b', 'EB')\n  error('ORIG', 0)\nend))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "b 2 ORIG|a 2 EB|false EA");
}

#[test]
fn unwind_closes_every_frame_innermost_first() {
    let src = "local function inner() local i <close> = mk('inner') error('DEEP', 0) end\nlocal function outer() local o <close> = mk('outer') inner() end\nout(pcall(outer))\nlocal function h()\n  local o <close> = mk('kept')\n  out(pcall(function() local i <close> = mk('caught') error('IN', 0) end))\n  out('h continues')\nend\nh()\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "inner 2 DEEP|outer 2 DEEP|false DEEP|caught 2 IN|false IN|h continues|kept 1 nil"
    );
}

#[test]
fn xpcall_handler_runs_first() {
    let src = "local function handler(m) out('handler', m) return 'handled:' .. tostring(m) end\nout(xpcall(function() local a <close> = mk('a') error('X', 0) end, handler))\nout(xpcall(function() local a <close> = mk('a', 'FROMCLOSE') error('X', 0) end, handler))\nout(xpcall(function() local a <close> = mk('a', 'FROMCLOSE') end, handler))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "handler X|a 2 handled:X|false handled:X|handler X|a 2 handled:X|handler FROMCLOSE|false handled:FROMCLOSE|a 1 nil|handler FROMCLOSE|false handled:FROMCLOSE"
    );
}

#[test]
fn return_call_in_close_scope() {
    let src = "local function g(x) out('in g', x) return 'r1', 'r2' end\nlocal function f() local c <close> = mk('c') return g('arg') end\nout(f())\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "in g arg|c 1 nil|r1 r2");
}

#[test]
fn yield_inside_close() {
    let src = "local function ymk(name) return setmetatable({}, {__close = function() out('closing', name) out('got', coroutine.yield(name)) end}) end\nlocal co = coroutine.wrap(function()\n  do local a <close> = ymk('block') end\n  out('after block')\n  out('pcall', pcall(function() local b <close> = ymk('unwind') error('E', 0) end))\n  local c <close> = ymk('return')\n  return 'R1', 'R2'\nend)\nout(co())\nout(co('v1'))\nout(co('v2'))\nout(co('v3'))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "closing block|block|got v1|after block|closing unwind|unwind|got v2|pcall false E|closing return|return|got v3|R1 R2"
    );
}

#[test]
fn coroutine_close_suspended() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a')\n  local b <close> = mk('b')\n  coroutine.yield()\nend)\ncoroutine.resume(co)\nout(coroutine.close(co))\nout(coroutine.status(co))\nco = coroutine.create(function()\n  local a <close> = mk('a', 'EA')\n  local b <close> = mk('b', 'EB')\n  coroutine.yield()\nend)\ncoroutine.resume(co)\nout(coroutine.close(co))\nout(coroutine.close(co))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "b 1 nil|a 1 nil|true|dead|b 1 nil|a 2 EB|false EA|true"
    );
}

#[test]
fn coroutine_close_after_error() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a')\n  local t <close> = setmetatable({tag = 'alive'}, {__close = function(o, e) out(o.tag, e) end})\n  error('DIE', 0)\nend)\nout(coroutine.resume(co))\nout(coroutine.status(co))\ncollectgarbage()\nout(coroutine.close(co))\nout(coroutine.close(co))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "false DIE|dead|alive DIE|a 2 DIE|false DIE|true");
}

#[test]
fn coroutine_close_runs_on_target() {
    let src = "local co\nco = coroutine.create(function()\n  local a <close> = setmetatable({}, {__close = function()\n    out(coroutine.running() == co, coroutine.isyieldable(), coroutine.isyieldable(co), coroutine.status(co))\n    coroutine.wrap(function() out('nested', coroutine.isyieldable(), coroutine.isyieldable(co)) end)()\n    coroutine.yield()\n  end})\n  coroutine.yield()\nend)\ncoroutine.resume(co)\nout(coroutine.close(co))\nout(coroutine.status(co))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "true false false running|nested true false|false attempt to yield across a C-call boundary|dead"
    );
}

#[test]
fn coroutine_close_inside_close() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a')\n  do\n    local b <close> = setmetatable({}, {__close = function() out('b yields') coroutine.yield() out('unreached') end})\n  end\nend)\ncoroutine.resume(co)\nout(coroutine.status(co))\nout(coroutine.close(co))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "b yields|suspended|a 1 nil|true");
}

#[test]
fn wrap_error_closes() {
    let src = "local f = coroutine.wrap(function()\n  local a <close> = mk('a')\n  local b <close> = mk('b', 'EB')\n  error('WRAPERR')\nend)\nout(pcall(f))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "b 2 c:4: WRAPERR|a 2 EB|false EB");
}

#[test]
fn coroutine_close_closes_upvalues() {
    let src = "local f\nlocal co = coroutine.create(function() local x = 42 f = function() return x end coroutine.yield() end)\ncoroutine.resume(co)\nout(coroutine.close(co))\ncollectgarbage()\nout(f())\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "true|42");
}

#[test]
fn generic_for_closing_value() {
    let src = "local function iter(t, name)\n  return function(_, i) i = i + 1 if t[i] then return i, t[i] end end, nil, 0, mk(name)\nend\nfor i, v in iter({10, 20, 30}, 'break') do out(i, v) if i == 2 then break end end\nfor i, v in iter({10}, 'end') do out(i, v) end\nout(pcall(function() for i in iter({1}, 'error') do error('boom', 0) end end))\nlocal function f(g) for i in iter({1}, 'return') do return g(i) end end\nout(f(function(i) out('in g', i) return 'r' end))\nfor i in iter({1, 2}, 'goto') do goto done end\n::done::\nfor k, v in pairs({a = 1}) do out(k, v) end\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "1 10|2 20|break 1 nil|1 10|end 1 nil|error 2 boom|false boom|in g 1|return 1 nil|r|goto 1 nil|a 1"
    );
}

#[test]
fn generic_for_non_closable() {
    let src = "out(pcall(function() for i in function() end, nil, nil, {} do end end))\nout(pcall(function() for i in function() end, nil, nil, false do out('skipped') end end))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "false c:1: variable '(for state)' got a non-closable value|true"
    );
}

#[test]
fn coroutine_close_self() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a')\n  local b <close> = mk('b')\n  out('yieldable', coroutine.isyieldable())\n  coroutine.close(coroutine.running())\n  out('unreached')\nend)\nout(coroutine.resume(co))\nout(coroutine.status(co), coroutine.close(co))\nlocal w = coroutine.wrap(function()\n  local a <close> = mk('outer')\n  pcall(function()\n    local b <close> = mk('inner')\n    coroutine.close(coroutine.running())\n  end)\n  out('unreached')\nend)\nout(select('#', w()))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "yieldable true|b 1 nil|a 1 nil|true|dead true|inner 1 nil|outer 1 nil|0"
    );
}

#[test]
fn coroutine_close_self_error() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a', 'EA')\n  local b <close> = mk('b')\n  out('pcall', pcall(coroutine.close, coroutine.running()))\nend)\nout(coroutine.resume(co))\nout(coroutine.status(co), coroutine.close(co))\nco = coroutine.create(function()\n  local a <close> = setmetatable({}, {__close = function() coroutine.yield('y') end})\n  coroutine.close(coroutine.running())\nend)\nout(coroutine.resume(co))\nout(pcall(coroutine.close, coroutine.running()))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "b 1 nil|a 1 nil|false EA|dead false EA|false attempt to yield across a C-call boundary|false cannot close main thread"
    );
}

#[test]
fn coroutine_close_self_during_error_close() {
    let src = "local co = coroutine.create(function()\n  local a <close> = mk('a')\n  pcall(function()\n    local b <close> = mk('b')\n    local c <close> = setmetatable({}, {__close = function() out('c') coroutine.close(coroutine.running()) end})\n    error('x', 0)\n  end)\n  out('unreached')\nend)\nout(coroutine.resume(co))\nout(coroutine.status(co))\nreturn table.concat(log, '|')";
    assert_eq!(run(src), "c|b 1 nil|a 1 nil|true|dead");
}

#[test]
fn coroutine_close_self_inside_close() {
    let src = "local function self_close() return setmetatable({}, {__close = function() out('b') coroutine.close(coroutine.running()) out('unreached') end}) end\nlocal co = coroutine.create(function() local a <close> = mk('a') local b <close> = self_close() coroutine.yield() end)\ncoroutine.resume(co)\nout('suspended', coroutine.close(co))\nco = coroutine.create(function() local a <close> = mk('a') local b <close> = self_close() error('E', 0) end)\ncoroutine.resume(co)\nout('dead', coroutine.close(co))\nco = coroutine.create(function() local a <close> = mk('a', 'EA') local b <close> = self_close() coroutine.yield() end)\ncoroutine.resume(co)\nout('failing', coroutine.close(co))\nlocal w = coroutine.wrap(function() local a <close> = mk('a') local b <close> = self_close() error('W', 0) end)\nout('wrap', pcall(w))\nreturn table.concat(log, '|')";
    assert_eq!(
        run(src),
        "b|a 1 nil|suspended true|b|a 1 nil|dead false nil|b|a 1 nil|failing false EA|b|a 1 nil|wrap false <no error object>"
    );
}
