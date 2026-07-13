//! TEMPORARY: dump is_prime's compiled region.

use crate::jit::backend::aarch64::encode;
use crate::jit::backend::isel::select;
use crate::jit::backend::mach::print_mfunc;
use crate::jit::backend::regalloc::linear_scan;
use crate::jit::frontend::lower::lower;
use crate::jit::ir::print::print_func;
use crate::jit::ir::ty::{Rep, Ty, TypeSet};
use crate::{Executor, Lua};

const INT: Ty = Ty::new(Rep::Val, TypeSet::INT);

const SRC: &str = r#"
local function is_prime(x)
    for i=2, x-1 do
        if x % i == 0 then
            return false
        end
    end
    return true
end
for i = 2, 100 do is_prime(i) end
return is_prime
"#;

#[test]
#[ignore]
fn dump_is_prime() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| {
            let chunk = ctx.load(SRC, Some("p")).expect("compile");
            Ok::<_, crate::RuntimeError>(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("start");
    lua.finish(&ex).expect("run");
    let f = lua.enter(|ctx| {
        let v = ctx
            .fetch(&ex)
            .take_result::<crate::env::value::Value>(ctx)
            .expect("result");
        ctx.stash(v.get_function().expect("a function"))
    });

    lua.enter(|ctx| {
        let closure = ctx.fetch(&f).as_lua().expect("Lua closure");
        let func = lower(closure.proto, 0, vec![INT]).expect("lower");
        let m = select(&func).expect("isel");
        let ra = linear_scan(&m);
        let code = encode(&m, &func.pool, &ra).expect("encode");

        println!("=== BYTECODE");
        for (i, ins) in closure.proto.code.iter().enumerate() {
            println!("{i:04}  {ins:?}");
        }
        println!("=== IR\n{}", print_func(&func));
        println!("=== MIR\n{}", print_mfunc(&m));
        println!("=== {} bytes of code", code.len());
        std::fs::write("/tmp/is_prime.bin", code.bytes()).unwrap();
    });
}
