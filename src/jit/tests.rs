//! Tests of the compiler stages on functions warmed up in the interpreter.

use crate::env::{LuaString, Value};
use crate::{Executor, LoadError, Lua};

/// Run `src`, then build the IR of global function `name` from `pc`.
pub(crate) fn ir_of(src: &str, name: &str, pc: u32) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let f = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, name.as_bytes())));
        let lf = f
            .get_function()
            .and_then(|f| f.as_lua())
            .expect("a Lua function");
        let opts = crate::jit::compile::Options {
            check: true,
            deopt_all: false,
        };
        match crate::jit::compile::build_ir(
            lf,
            pc,
            &opts,
            std::ptr::null(),
            &[],
            &mut Default::default(),
        ) {
            Ok((func, _)) => func.print(),
            Err(e) => panic!("{e:?}"),
        }
    })
}

#[test]
fn builds_is_prime() {
    let src = "function is_prime(n)
        if n < 2 then return false end
        if n % 2 == 0 then return n == 2 end
        local i = 3
        while i * i <= n do
            if n % i == 0 then return false end
            i = i + 2
        end
        return true
    end
    local c = 0
    for n = 1, 2000 do if is_prime(n) then c = c + 1 end end";
    let ir = ir_of(src, "is_prime", 0);
    println!("{ir}");
}

#[test]
fn builds_loop_entry_in_nest() {
    let src = "function mandel()
        local t = 0
        for y = 0, 20 do
            for x = 0, 20 do
                t = t + x * y
            end
        end
        return t
    end
    for i = 1, 3 do mandel() end";
    // The inner FORLOOP.
    let pc = {
        let mut lua = Lua::new();
        lua.load_all();
        let ex = lua
            .try_enter(|ctx| -> Result<_, LoadError> {
                let chunk = ctx.load(src, Some("=t"))?;
                Ok(ctx.stash(Executor::start(ctx, chunk, ())))
            })
            .expect("load");
        lua.finish(&ex).expect("run");
        lua.enter(|ctx| {
            let f = ctx
                .globals()
                .raw_get(Value::string(LuaString::new(ctx, b"mandel")));
            let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
            let code: Vec<_> = lf.proto.code.iter().collect();
            let loops: Vec<u32> = code
                .iter()
                .enumerate()
                .filter(|(_, i)| i.generic_op() == crate::instruction::Op::FORLOOP)
                .map(|(pc, _)| pc as u32)
                .collect();
            loops[0]
        })
    };
    let ir = ir_of(src, "mandel", pc);
    println!("{ir}");
}

/// The opcodes of global function `name` after running `src`.
fn ops_of(src: &str, name: &str) -> Vec<crate::instruction::Op> {
    let mut lua = Lua::new();
    lua.set_jit(false);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let f = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, name.as_bytes())));
        let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
        lf.proto.code.iter().map(|i| i.op()).collect()
    })
}

/// A form drops when its site sees kinds no form covers, so the JIT can
/// compile a form as what it says (6.5).
#[test]
fn forms_drop_on_foreign_kinds() {
    use crate::instruction::Op;
    let ops = ops_of(
        "function f(a, b) return a + b end
        for i = 1, 10 do f(i, i) end",
        "f",
    );
    assert!(ops.contains(&Op::ADD_II));
    let ops = ops_of(
        "function f(a, b) return a + b end
        for i = 1, 10 do f(i, i) end
        f('1', 2)",
        "f",
    );
    assert!(ops.contains(&Op::ADD) && !ops.contains(&Op::ADD_II));
    let ops = ops_of(
        "function f(a, b) return a < b end
        for i = 1, 10 do f(i, i) end",
        "f",
    );
    assert!(ops.contains(&Op::JLT_II));
    let ops = ops_of(
        "function f(a, b) return a < b end
        for i = 1, 10 do f(i, i) end
        f('a', 'b')",
        "f",
    );
    assert!(ops.contains(&Op::JLT) && !ops.contains(&Op::JLT_II));
}
