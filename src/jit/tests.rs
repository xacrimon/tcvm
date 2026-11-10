//! Tests of the compiler stages on functions warmed up in the interpreter.

use crate::env::{LuaString, Value};
use crate::{Executor, LoadError, Lua};

/// Run `src` in the interpreter, then hand `with` global function `name`
/// with the feedback it left.
fn warmed<R>(
    src: &str,
    name: &str,
    with: impl for<'gc> FnOnce(crate::lua::Context<'gc>, crate::env::function::LuaFn<'gc>) -> R,
) -> R {
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
        let lf = f
            .get_function()
            .and_then(|f| f.as_lua())
            .expect("a Lua function");
        with(ctx, lf)
    })
}

/// Run `src` in the interpreter, then build the IR of global function
/// `name` from `pc` with the feedback it left, whatever the JIT mode.
pub(crate) fn ir_of(src: &str, name: &str, pc: u32) -> String {
    warmed(src, name, |ctx, lf| {
        let opts = crate::jit::compile::Options {
            check: true,
            deopt_all: false,
        };
        match crate::jit::compile::build_ir(
            ctx,
            lf,
            pc,
            &opts,
            std::ptr::null(),
            &[],
            &mut crate::jit::compile::Times::new(),
        ) {
            Ok((func, _)) => func.print(),
            Err(e) => panic!("{e:?}"),
        }
    })
}

/// The IR the builder makes of global function `name` from `pc`, before
/// any pass: what it knows as it builds.
fn built_ir_of(src: &str, name: &str, pc: u32) -> String {
    use crate::jit::compile::{canonical_back_edges, resolve_original};
    warmed(src, name, |ctx, lf| {
        let mut code: Vec<_> = lf
            .proto
            .code
            .iter()
            .map(|i| resolve_original(&lf.proto, i))
            .collect();
        canonical_back_edges(&lf.proto, &mut code);
        let loop_entry = code[pc as usize].op() != crate::instruction::Op::FUNC;
        let tails = if loop_entry {
            Vec::new()
        } else {
            crate::jit::compile::self_tail_calls(lf, &code)
        };
        let cfg = crate::jit::build::cfg::Cfg::build(&lf.proto, code, pc, tails).expect("a CFG");
        let b = crate::jit::build::Builder::new(ctx, lf, &cfg, pc, loop_entry);
        b.build().expect("built").print()
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
            // A compiled loop's word is its entry's.
            let op = |i: &crate::instruction::Instruction| {
                if i.is_jit_word() {
                    super::original_of(lf, *i).generic_op()
                } else {
                    i.generic_op()
                }
            };
            let loops: Vec<u32> = code
                .iter()
                .enumerate()
                .filter(|(_, i)| op(i) == crate::instruction::Op::FORLOOP)
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
    // `3 * x` has the immediate on the left; the form follows `x`.
    let ops = ops_of(
        "function f(x) return 3 * x end
        for i = 1, 10 do f(i) end
        f(0.5)",
        "f",
    );
    assert!(ops.contains(&Op::MULI_F) && !ops.contains(&Op::MULI_I));
}

/// A `while` loop's entry points its back edge at the `JIT_LOOP` over its
/// `LOOP`, and retiring the entry points it past the `LOOP` again.
#[cfg(target_arch = "aarch64")]
#[test]
fn back_edges_follow_loop_entries() {
    use crate::instruction::Op;
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "function f(n) local i = 0 while i < n do i = i + 1 end return i end
                f(1000)",
                Some("=t"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let f = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, b"f")));
        let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
        let code = |pc: usize| lf.proto.code.iter().nth(pc).unwrap();
        let n = lf.proto.code.len();
        let header = (0..n)
            .find(|&pc| code(pc).op() == Op::JIT_LOOP)
            .expect("a loop entry");
        let back = (0..n).find(|&pc| code(pc).op() == Op::JMP_BACK).unwrap();
        let target = |pc: usize| (pc as i64 + 1 + code(pc).branch_offset() as i64) as usize;
        assert_eq!(target(back), header);
        let state = lf.proto.jit.get().unwrap();
        let mut st = state.borrow_mut(ctx.mutation());
        // Under `TCVM_JIT_HOT_CALL=1` the function's entry comes first.
        let k = (st.entries.iter())
            .position(|e| e.original.op() == Op::LOOP)
            .expect("the loop's entry");
        let e = st.entries.remove(k);
        assert_eq!(&*e.back_edges, &[back as u32]);
        super::retire_entry(ctx, lf, &mut st, &e);
        drop(st);
        assert_eq!(code(header).op(), Op::LOOP);
        assert_eq!(target(back), header + 1);
    });
}

/// Running out of code memory turns the JIT off, rather than failing each
/// later prototype's compile in turn.
#[cfg(target_arch = "aarch64")]
#[test]
fn exhausted_code_memory_turns_the_jit_off() {
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    lua.enter(|ctx| ctx.jit().code_alloc().limit_segments(1));
    let n = 400;
    let mut src = String::new();
    for k in 1..=n {
        src += &format!(
            "function f{k}(n) local s = 0 for i = 1, n do s = s + i * {k} end return s end\n"
        );
    }
    src += &format!("local t = 0 for k = 1, {n} do t = t + _G['f' .. k](1000) end return t");
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let t: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        assert_eq!(t.get_integer(), Some(500_500 * n * (n + 1) / 2));
        let jit = ctx.jit();
        assert!(jit.exhausted.get());
        assert_eq!(jit.internal_errors.get(), 0);
        let entries = |k: i64| {
            let f = ctx.globals().raw_get(Value::string(LuaString::new(
                ctx,
                format!("f{k}").as_bytes(),
            )));
            let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
            lf.proto.jit.get().map_or(0, |s| s.borrow().entries.len())
        };
        assert!(entries(1) > 0);
        assert_eq!(entries(n), 0);
    });
}

/// A loop whose FORPREP ran in a region compiled for the loop before it
/// takes the integer form when it runs, which that FORPREP could not write:
/// a generic FORLOOP left no feedback, so every build saw a loop that never
/// ran, and refused it.
#[cfg(target_arch = "aarch64")]
#[test]
fn loops_entered_from_compiled_code_compile() {
    use crate::instruction::Op;
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "function f(n)
                  local t = {}
                  for i = 1, n do t[i] = i end
                  local s = 0
                  for i = 1, n do s = s + t[i] end
                  return s
                end
                local r = 0
                for _ = 1, 3 do r = r + f(1000) end
                return r",
                Some("=t"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let r: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        assert_eq!(r.get_integer(), Some(3 * 500_500));
        let f = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, b"f")));
        let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
        let st = lf.proto.jit.get().expect("compiled").borrow();
        assert!(st.refused.is_empty(), "refused {:?}", st.refused);
        drop(st);
        let forms: Vec<Op> = (lf.proto.code.iter())
            .map(|w| {
                if w.is_jit_word() {
                    super::original_of(lf, w).op()
                } else {
                    w.op()
                }
            })
            .filter(|op| matches!(op, Op::FORLOOP | Op::FORLOOP_I | Op::FORLOOP_F))
            .collect();
        assert_eq!(forms, [Op::FORLOOP_I, Op::FORLOOP_I]);
    });
}

/// Under `TCVM_JIT_FAR` (a test mode, `tmodes.sh`'s `far`) the code range
/// lies out of the text's reach, and compiled code calls into the text
/// through islands.
#[cfg(target_arch = "aarch64")]
#[test]
fn far_code_reaches_the_text_through_islands() {
    if std::env::var_os("TCVM_JIT_FAR").is_none() {
        return;
    }
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local t = {} for i = 1, 1000 do t[i] = {i} end return #t",
                Some("=t"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let n: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        assert_eq!(n.get_integer(), Some(1000));
        assert_eq!(ctx.jit().internal_errors.get(), 0);
    });
    let (near, islands) = crate::jit::backend::alloc::range_state().expect("a code range");
    assert!(!near && islands > 0, "near {near}, {islands} islands");
}

/// A frame whose to-be-closed variables all closed runs its compiled CLOSEs
/// and RETURN, which exit while `HAS_TBC` is set.
#[cfg(target_arch = "aarch64")]
#[test]
fn closed_scopes_clear_has_tbc() {
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local mt = {__close = function() end}
                function f(n)
                  do local x <close> = setmetatable({}, mt) end
                  local s = 0
                  for i = 1, n do
                    local g = function() return i end
                    s = s + g()
                  end
                  return s
                end
                return f(1000)",
                Some("=t"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let r: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        assert_eq!(r.get_integer(), Some(500_500));
        let f = ctx
            .globals()
            .raw_get(Value::string(LuaString::new(ctx, b"f")));
        let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
        let st = lf.proto.jit.get().expect("compiled").borrow();
        assert!(!st.regions.is_empty());
        // At most once each: `TCVM_JIT_DEOPT_ALL` takes every exit once.
        let most = (st.regions.iter())
            .flat_map(|r| r.exits.iter())
            .filter(|e| e.tag == crate::jit::ir::ops::ExitTag::Unsupported)
            .map(|e| e.count.get())
            .max();
        assert!(most.is_some_and(|n| n <= 1), "{most:?}");
    });
}

/// A bounds check moved out of an inlined accessor into its caller's loop
/// (E2) records its miss at the accessor's site: its exit names the
/// accessor's prototype, the snapshot being the caller's.
#[cfg(target_arch = "aarch64")]
#[test]
fn hoisted_inlined_bounds_miss_at_the_callee() {
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "function at(t, i) return t[i] end
                function f(t, n) local s = 0 for i = 1, n do s = s + (at(t, i) or 0) end return s end
                local t = {}
                for i = 1, 100 do t[i] = i end
                for r = 1, 300 do f(t, 100) end",
                Some("=t"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let get = |name: &[u8]| {
            ctx.globals()
                .raw_get(Value::string(LuaString::new(ctx, name)))
                .get_function()
                .and_then(|f| f.as_lua())
                .unwrap()
        };
        let (at, f) = (get(b"at"), get(b"f"));
        let state = f.proto.jit.get().expect("f compiled");
        let st = state.borrow();
        let hoisted: Vec<_> = st
            .entries
            .iter()
            .filter_map(|e| e.region)
            .flat_map(|r| {
                r.exits
                    .iter()
                    .filter(|x| x.site.is_some() && x.site_proto != u16::MAX)
                    .map(|x| r.site_protos[x.site_proto as usize])
                    .collect::<Vec<_>>()
            })
            .collect();
        // Compiled at once (`TCVM_JIT_HOT_*`), no feedback makes the check.
        let config = &ctx.jit().config;
        assert!(config.dev.hot_call <= 1 || config.dev.hot_loop <= 1 || !hoisted.is_empty());
        assert!(hoisted.iter().all(|&p| crate::dmm::Gc::ptr_eq(p, at.proto)));
    });
}

/// One shape guard per receiver while nothing changes shapes, a field read
/// again is the first read, and a read after a store is the stored value.
#[test]
fn field_reads_numbered_by_memory_version() {
    let src = "function step(p)
        local a = p.x + p.x
        p.y = a
        return p.y + p.x
    end
    local p = {x = 1.5, y = 0}
    for i = 1, 300 do step(p) end";
    let ir = ir_of(src, "step", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("guardshape"), 1, "{ir}");
    assert_eq!(count("tabload"), 1, "{ir}");
    assert_eq!(count("tabstore"), 1, "{ir}");
}

/// What a loop does not change leaves it: the callee guard of an intrinsic
/// and a field the loop does not store; a field it stores stays.
#[test]
fn loop_invariants_leave_the_loop() {
    let src = "local sqrt = math.sqrt
    function norm(p, n)
        local s = 0
        for i = 1, n do
            s = s + sqrt(p.x * i)
            p.y = s
        end
        return s
    end
    for i = 1, 300 do norm({x = 2.0, y = 0}, 20) end";
    let ir = ir_of(src, "norm", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    // The loop is peeled: what stays in it appears in both copies.
    assert_eq!(count("guard.same"), 1, "{ir}");
    assert_eq!(count("guardshape"), 1, "{ir}");
    assert_eq!(count("tabload"), 1, "{ir}");
    assert_eq!(count("tabstore"), 2, "{ir}");
}

/// A loop-carried value whose last iteration's is nil is not guarded where
/// it is loaded: the path out of the loop never uses it as a table.
#[test]
fn loop_carried_definitions_guard_only_down_safe() {
    let src = "function append(packet, head)
        packet.link = nil
        if head == nil then return packet end
        local mouse = head
        local link = mouse.link
        while link ~= nil do
            mouse = link
            link = mouse.link
        end
        mouse.link = packet
        return head
    end
    for i = 1, 300 do
        local head
        for j = 1, i % 5 do head = append({}, head) end
    end";
    let ir = ir_of(src, "append", 0);
    assert!(!ir.contains("Type after"), "{ir}");
}

/// An exit does not restore a register whose slot still holds the value:
/// the ninth and tenth parameters, which callers leave in slots (W4 §2:
/// the first eight come in registers), are only ever loaded, though inside
/// the loop the snapshots name their guarded copies.
#[test]
fn snapshots_leave_out_unchanged_slots() {
    let src = "function h(_1, _2, _3, _4, _5, _6, _7, _8, a, b)
        local s = 0
        for i = 1, 10 do s = s + a.x * i + b end
        return s + a.y
    end
    for i = 1, 300 do h(1, 2, 3, 4, 5, 6, 7, 8, {x = 1.5, y = 2}, i) end";
    let ir = ir_of(src, "h", 0);
    let snaps: Vec<&str> = ir
        .lines()
        .filter_map(|l| l.split_once('[').map(|x| x.1))
        .collect();
    assert!(!snaps.is_empty(), "{ir}");
    assert!(
        snaps
            .iter()
            .all(|s| !s.contains("r8=") && !s.contains("r9=")),
        "{ir}"
    );
}

/// Registers a call does not write keep their values and their types after
/// it: the entry guards cover the uses after each call. The parameters came
/// in registers (W4 §2) and are never loaded from their slots.
#[test]
fn calls_keep_the_types_of_unwritten_registers() {
    let src = "local g
    g = function(x) local a = {} for j = 1, 3 do a[j] = x end return a[1] end
    function f(t, i, o)
        local s = t[i] + o.d
        g(i)
        s = s + t[i] + o.d
        g(i)
        return s + t[i] + i
    end
    local t = {}
    for i = 1, 10 do t[i] = i end
    for k = 1, 300 do f(t, 5, {d = 3}) end";
    let ir = ir_of(src, "f", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("guard val:tab"), 2, "{ir}");
    assert_eq!(count("load r0"), 0, "{ir}");
    assert_eq!(count("= param 0"), 1, "{ir}");
}

/// A branch on an equality with a constant makes the register that
/// constant in the arm where it holds.
#[test]
fn branches_refine_what_they_test() {
    let src = "function f(x)
        if x == 'key' then return #x end
        return 0
    end
    for i = 1, 300 do f(i % 2 == 0 and 'key' or i) end";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("istype"), "{ir}");
}

/// An identity guard on a callee makes its register that function: the
/// second call needs no guard.
#[test]
fn identity_guards_make_constants() {
    let src = "local g
    g = function(x) local a = {} for j = 1, 3 do a[j] = x end return a[1] end
    function f(h, x)
        h(x)
        h(x)
        return x
    end
    for i = 1, 300 do f(g, i) end";
    let ir = ir_of(src, "f", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("guard.same"), 1, "{ir}");
}

/// A table guarded to one shape on both arms of a branch has it after the
/// join, where GVN (by dominance) does not see it: no third guard.
#[test]
fn shape_guards_carry_through_joins() {
    let src = "function f(o, c)
        local a
        if c then a = o.x else a = o.y end
        return a + o.z
    end
    for i = 1, 300 do keep = {x = 1, y = 2, z = 3}; f(keep, i % 2 == 0) end";
    // `keep` holds the shape for the build after the run, past collections.
    let ir = ir_of(src, "f", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("guardshape"), 2, "{ir}");
}

/// Tables of per-object metatables of one content share one layout: the
/// owner's mono shape is deprecated once a second member joins, new tables
/// of the owner go poly, and an old one moves at its next field slow path.
#[test]
fn second_member_deprecates_mono_shapes() {
    let src = "V = {}
    V.__index = V
    a = setmetatable({x = 1}, {__index = V})
    a_before = a
    b = setmetatable({x = 2}, {__index = V})
    c = setmetatable({x = 3}, getmetatable(a))
    local s = a.x + b.x + c.x
    W = {}
    W.__index = W
    local function new(x, y)
        local o = setmetatable({}, {__index = W})
        o.x = x
        o.y = y
        return o
    end
    p = new(1, 2)
    p_mt = getmetatable(p)
    q = new(3, 4)
    sum = p.x + q.x + p.y + q.y
    same_mt = getmetatable(p) == p_mt";
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
        let g = |n: &str| {
            ctx.globals()
                .raw_get(Value::string(LuaString::new(ctx, n.as_bytes())))
                .get_table()
                .expect("a table")
        };
        let (a, b, c) = (g("a"), g("b"), g("c"));
        for (n, t) in [("a", a), ("b", b), ("c", c)] {
            let s = t.shape();
            assert!(
                s.is_poly() && !s.is_deprecated(),
                "{n}: poly {} deprecated {} class {:?}",
                s.is_poly(),
                s.is_deprecated(),
                s.mt_cache().map(|c| c.is_shared())
            );
        }
        assert!(crate::env::Shape::ptr_eq(a.shape(), b.shape()));
        assert!(crate::env::Shape::ptr_eq(c.shape(), b.shape()));
        // Fields set after `setmetatable`: the old table's slots move.
        let (p, q) = (g("p"), g("q"));
        assert!(crate::env::Shape::ptr_eq(p.shape(), q.shape()));
        let v = |n: &str| {
            ctx.globals()
                .raw_get(Value::string(LuaString::new(ctx, n.as_bytes())))
        };
        assert_eq!(v("sum").get_integer(), Some(10));
        assert_eq!(v("same_mt").get_boolean(), Some(true));
    });
}

/// Shapes that keep the key in one slot share one guard and one load; the
/// interpreter's refills told the compiler of all three.
#[test]
fn polymorphic_reads_merge() {
    let src = "function getx(p) return p.x + 1 end
    for i = 1, 300 do getx({x = i, y = 2}); getx({x = i, z = 3}); getx({x = i, w = 4}) end";
    let ir = ir_of(src, "getx", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("guardshapein"), 1, "{ir}");
    assert_eq!(count("tabload"), 1, "{ir}");
    assert_eq!(count("deopt"), 0, "{ir}");
}

/// Two layouts dispatch once: the last needs a guard, not a test and an
/// exit, and later reads of the same table know its shape is one of two.
#[test]
fn polymorphic_dispatch_carries_its_shapes() {
    let src = "function len2(p) return p.x * p.x + p.y * p.y end
    for i = 1, 300 do len2({x = i, y = 2}); len2({y = 2, x = i}) end";
    let ir = ir_of(src, "len2", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    assert_eq!(count("shapeis"), 1, "{ir}");
    assert_eq!(count("guardshape("), 1, "{ir}");
    assert_eq!(count("deopt"), 0, "{ir}");
}

/// Only a region with dependencies keeps lazy exits (ws2 §5.4): one past
/// each call once a class's bits are folded.
#[test]
fn lazy_exits_need_dependencies() {
    // No field loads either: their slots' kinds are dependencies.
    let plain = "local format = string.format
    function f()
        local s = 0
        for i = 1, 300 do s = s + #format('%d', i) end
        return s
    end
    f()";
    assert!(!ir_of(plain, "f", 0).contains("lazyexit"));
    let folded = "local mt = {}
    local o = setmetatable({}, mt)
    function f()
        local s = 0
        for i = 1, 300 do
            s = s + #string.format('%d', i)
            if o.y then s = s + 1 end
        end
        return s
    end
    f()";
    let ir = ir_of(folded, "f", 0);
    assert!(ir.contains("lazyexit"), "{ir}");
}

/// A write changing a class's bits invalidates the region that folded
/// them: its frame leaves at the lazy exit past the call it is in, and its
/// entry is retired. `hook` stays a call: inlining is off.
#[test]
fn class_writes_invalidate_dependent_regions() {
    let src = "mt = {}
    local o = setmetatable({x = 1}, mt)
    go = false
    local function hook(i) if go and i == 150 then mt.__index = {y = 5} end end
    function f()
        local s = 0
        for i = 1, 300 do
            hook(i)
            local y = o.y
            if y then s = s + y end
        end
        return s
    end
    for k = 1, 20 do f() end
    go = true
    r = f()";
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.set_jit_inline(false);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let g = |n: &str| {
            ctx.globals()
                .raw_get(Value::string(LuaString::new(ctx, n.as_bytes())))
        };
        assert_eq!(g("r").get_integer(), Some(755));
        let f = g("f").get_function().and_then(|f| f.as_lua()).unwrap();
        let st = f.proto.jit.get().expect("compiled").borrow();
        let dead: Vec<_> = st.regions.iter().filter(|r| r.invalidated.get()).collect();
        assert!(!dead.is_empty());
        let mut left = 0;
        for &&r in &dead {
            assert!(r.retired.get() && !r.lazy.is_empty());
            assert!(
                !st.entries
                    .iter()
                    .any(|e| e.region.is_some_and(|x| crate::dmm::Gc::ptr_eq(x, r)))
            );
            left += r
                .exits
                .iter()
                .filter(|e| e.tag == crate::jit::ir::ops::ExitTag::Lazy)
                .map(|e| e.count.get())
                .sum::<u32>();
        }
        assert!(left >= 1);
    });
}

/// Reads through `__index` tables fold to what the watched holders hold
/// (ws2 §5.5): a method two holders up is a constant callee, an own field
/// of a class whose walk ends in nil needs no nil test, and a key no holder
/// has is nil; nothing reads a class or a holder.
#[test]
fn holder_reads_fold() {
    let src = "local Base = {}
    function Base:kind() return 1 end
    local Mid = setmetatable({}, {__index = Base})
    local C = setmetatable({}, {__index = Mid})
    local o = setmetatable({v = 3}, {__index = C})
    function f()
        local s = 0
        for i = 1, 300 do
            s = s + o:kind() + o.v
            if o.missing then s = s + 1 end
        end
        return s
    end
    f()";
    let ir = ir_of(src, "f", 0);
    for op in [
        "classindex",
        "classbits",
        "call.index",
        "istype val:nil",
        "call r",
    ] {
        assert!(!ir.contains(op), "{op} in {ir}");
    }
}

/// A refined parameter is kept across a call in a loop as the repair
/// parameter standing for it, and its unboxed use after the next call is
/// made again from that (permute's `permute`, once its field reads lost
/// their nil arms).
#[test]
fn unboxed_refinements_survive_repeated_calls() {
    let src = "local P = setmetatable({count = 0, v = {0, 0, 0, 0, 0, 0}}, {__index = {}})
    function P:rec(n)
        self.count = self.count + 1
        if n ~= 0 then
            local n1 = n - 1
            self:rec(n1)
            for i = n, 1, -1 do
                self:swap(n, i)
                self:rec(n1)
                self:swap(n, i)
            end
        end
    end
    function P:swap(i, j)
        local t = self.v[i]
        self.v[i] = self.v[j]
        self.v[j] = t
    end
    for k = 1, 3 do P:rec(5) end
    f = P.rec";
    let ir = ir_of(src, "f", 0);
    assert!(ir.contains("call"), "{ir}");
}

/// An inlined body's builder reads the region's upvalues through the
/// region's closure, not its callee's: `h` is `f`'s varying upvalue, not
/// `apply`'s second (`z`, whose body would be inlined under a guard that
/// always fails).
#[test]
fn bodies_read_the_regions_upvalues() {
    let src = "local y, z = false, function(x) return x * 1000 end
    local function apply(h, v) if y then return z end local r = h(v); return r end
    local function mk(g)
        return function(n) local s = 0 for i = 1, n do s = s + apply(g, i) end return s end
    end
    local f1 = mk(function(x) return x + 1 end)
    local f2 = mk(function(x) return x + 2 end)
    for k = 1, 3 do f1(100); f2(100) end
    f = f1";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("ki32 1000"), "{ir}");
}

#[test]
fn guarded_callees_stay_known_across_calls() {
    let src = "function f(resume, co, n)
        local ok, r
        for _ = 1, n do ok, r = resume(co) end
        return ok, r
    end
    local co = coroutine.create(function() while true do coroutine.yield(1) end end)
    for k = 1, 3 do f(coroutine.resume, co, 1000) end";
    let ir = ir_of(src, "f", 0);
    // After the peeled iteration's call the loop gets the callee as a
    // copy of the constant, which needs no guard.
    assert_eq!(
        ir.lines().filter(|l| l.contains("guard.same")).count(),
        1,
        "{ir}"
    );
}

#[test]
fn field_loads_take_their_slot_kinds() {
    let src = "function f(p, n)
        local s = 0.0
        for i = 1, n do s = s + p.x * p.y end
        if p.next then s = s + p.next.x end
        return s
    end
    local q = {x = 0.5, y = 1.5}
    for k = 1, 300 do keep = {x = 1.5, y = 2.5, next = q}; f(keep, 10) end";
    let ir = ir_of(src, "f", 0);
    let count = |s: &str| ir.lines().filter(|l| l.contains(s)).count();
    // The floats need no checks, nor the table loaded from `next`.
    assert_eq!(count("guard val:float"), 0, "{ir}");
    assert_eq!(count("istype"), 0, "{ir}");
    assert!(ir.contains("val:float = tabload"), "{ir}");
    assert!(ir.contains("val:tab@s"), "{ir}");
}

/// A store keeps its barrier only while the value may be a heap object: a
/// parameter, untyped where the store is built, is a small integer once
/// its arithmetic use's guard moves to its definition (`speculate`).
#[test]
fn barriers_leave_stores_of_numbers() {
    let src = "function f(t, n, v)
        local s = 0
        for i = 1, n do t[i] = v; s = s + v end
        return s
    end
    keep = {}
    for i = 1, 40 do keep[i] = 0 end
    for r = 1, 300 do f(keep, 30, r) end";
    let ir = ir_of(src, "f", 0);
    assert!(ir.contains("arrstore"), "{ir}");
    assert!(!ir.contains("barrier"), "{ir}");
}

/// String keys of a table access read and write its string part directly
/// (W2.5), the generic walk only in a cold arm.
#[test]
fn string_keys_take_the_string_helpers() {
    let src = "function f(t, s, n)
        for i = 1, #s - n + 1 do
            local k = string.sub(s, i, i + n - 1)
            t[k] = (t[k] or 0) + 1
        end
    end
    keep = {}
    for r = 1, 300 do f(keep, 'ACGTTGCAAGCT', 3) end";
    let ir = ir_of(src, "f", 0);
    let mut cold = false;
    let hot: Vec<&str> = ir
        .lines()
        .filter(|l| {
            if !l.starts_with(' ') {
                cold = l.contains(" cold");
            }
            !cold
        })
        .collect();
    let has = |s: &str| hot.iter().any(|l| l.contains(s));
    assert!(has("call.getstr") && has("call.setstr"), "{ir}");
    assert!(!has("call.index") && !has("call.newindex"), "{ir}");
}

/// Float keys with integral values index the array part (W2.6): no generic
/// helper where the site saw only numbers.
#[test]
fn float_keys_index_arrays() {
    let src = "function f(t, n)
        local s, k = 0, 0.5
        for i = 1, n do k = k + 1.0; t[k + 0.5] = s; s = s + t[i * 1.0] end
        return s
    end
    keep = {}
    for i = 1, 40 do keep[i] = i end
    for r = 1, 300 do f(keep, 30) end";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("call.index"), "{ir}");
    assert!(!ir.contains("call.newindex"), "{ir}");
    assert!(ir.contains("ftoiexact"), "{ir}");
}

/// A float -0.0 converts to integer 0 as Lua's conversion does (a key,
/// `ceil`'s and `floor`'s results, a bitwise operand), without an exit.
#[test]
fn negative_zero_converts_in_compiled_code() {
    let src = "local ceil, floor = math.ceil, math.floor
    function f(n)
        local s, t = 0, {}
        for i = 1, n do
            local z = -0.0 * i
            t[z] = i
            s = s + ceil(-0.25 - (i % 2) * 0.5) + floor(z) + t[0] + (z | 0)
        end
        return s
    end
    r = 0
    for k = 1, 20 do r = r + f(300) end";
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.enter(|ctx| {
        let g = |n: &str| {
            ctx.globals()
                .raw_get(Value::string(LuaString::new(ctx, n.as_bytes())))
        };
        // Reference Lua 5.5.1.
        assert_eq!(g("r").get_integer(), Some(903000));
        let f = g("f").get_function().and_then(|f| f.as_lua()).unwrap();
        let st = f.proto.jit.get().expect("compiled").borrow();
        let exits: u32 = (st.regions.iter())
            .flat_map(|r| r.exits.iter())
            .map(|e| e.count.get())
            .sum();
        assert!(exits < 50, "{exits} exits");
    });
}

/// Closures and cell upvalues inline (W3 step 4): `make`'s closures come
/// from `closure.of` with its closure, its cell store goes through that
/// closure's cell, and its return closes what they opened.
#[test]
fn closures_inline() {
    let src = "local count = 0
    local function make(k)
        local n = k
        count = count + 1
        return function() n = n + 1 return n end
    end
    function f(n)
        local s = 0
        for i = 1, n do s = s + make(i)() end
        return s
    end
    for k = 1, 300 do f(10) end";
    let ir = ir_of(src, "f", 0);
    for op in ["closure.of", "upcell.of", "cellstore", "close r"] {
        assert!(ir.contains(op), "{op}: {ir}");
    }
}

/// No loop parameter carries a constant around the back edge, where it was
/// spilled: the call-boundary pass's copies after each call joined the
/// preheader's at the header, and `K`'s table pointer (not argument-free)
/// stayed joined; the header makes it again.
#[test]
fn constants_stay_in_their_blocks() {
    let src = "local K = {a = 1, b = 2}
    local function g(m)
        local i = 0
        if m > 3 then goto b end
        ::a:: i = i + 1
        ::b:: i = i + 2
        if i < 5 then goto a end
        return i
    end
    function f(n)
        local s = 0
        for i = 1, n do
            s = s + K.a + K.b
            g(i)
        end
        return s
    end
    for k = 1, 300 do f(10) end";
    let ir = ir_of(src, "f", 0);
    let params: Vec<&str> = ir
        .lines()
        .filter(|l| l.starts_with('b') && l.contains('(') && l.contains("=k"))
        .collect();
    assert!(params.is_empty(), "{params:?}\n{ir}");
}

/// Recursion inlines one copy (W3 §8): `fib`'s region inlines both of its
/// self-calls once, and each copy's two calls of `fib` stay real calls.
#[test]
fn recursion_inlines_one_copy() {
    let src = "function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end
    for k = 1, 300 do fib(10) end";
    let ir = ir_of(src, "fib", 0);
    assert_eq!(ir.matches("call r").count(), 4, "{ir}");
}

/// The plan inlines by frequency over size (W3 §12): `h`, longer than the
/// old 48-bytecode limit, runs on every call of `f` and inlines; `g` runs
/// on one call in a hundred, under the frequency floor, and is called.
#[test]
fn inlining_follows_the_plan() {
    let src = "local function h(x)
        local a0 = x
        local a1 = x * 1 + a0
        local a2 = x * 2 + a1
        local a3 = x * 3 + a2
        local a4 = x * 4 + a3
        local a5 = x * 5 + a4
        local a6 = x * 6 + a5
        local a7 = x * 7 + a6
        local a8 = x * 8 + a7
        local a9 = x * 9 + a8
        local a10 = x * 10 + a9
        local a11 = x * 11 + a10
        local a12 = x * 12 + a11
        local a13 = x * 13 + a12
        local a14 = x * 14 + a13
        local a15 = x * 15 + a14
        local a16 = x * 16 + a15
        local a17 = x * 17 + a16
        local a18 = x * 18 + a17
        local a19 = x * 19 + a18
        local a20 = x * 20 + a19
        local a21 = x * 21 + a20
        local a22 = x * 22 + a21
        local a23 = x * 23 + a22
        local a24 = x * 24 + a23
        local a25 = x * 25 + a24
        return a25
    end
    local function g(x)
        local a0 = x
        local a1 = x * 1 + a0
        local a2 = x * 2 + a1
        local a3 = x * 3 + a2
        local a4 = x * 4 + a3
        local a5 = x * 5 + a4
        local a6 = x * 6 + a5
        local a7 = x * 7 + a6
        local a8 = x * 8 + a7
        local a9 = x * 9 + a8
        local a10 = x * 10 + a9
        local a11 = x * 11 + a10
        local a12 = x * 12 + a11
        local a13 = x * 13 + a12
        local a14 = x * 14 + a13
        local a15 = x * 15 + a14
        local a16 = x * 16 + a15
        local a17 = x * 17 + a16
        local a18 = x * 18 + a17
        local a19 = x * 19 + a18
        local a20 = x * 20 + a19
        local a21 = x * 21 + a20
        local a22 = x * 22 + a21
        local a23 = x * 23 + a22
        local a24 = x * 24 + a23
        local a25 = x * 25 + a24
        return a25 + x
    end
    function f(n)
        local r = h(n)
        if n % 100 == 0 then r = r + g(n) end
        return r
    end
    for k = 1, 3000 do f(k) end";
    let ir = ir_of(src, "f", 0);
    assert_eq!(ir.matches("call r").count(), 1, "{ir}");
}

/// A call whose callee joins two constant functions dispatches on them
/// (W3 §11): both bodies inline, one real call stays for any other value.
#[test]
fn polymorphic_calls_dispatch() {
    let src = "local A = {} A.__index = A
    function A:get() return self.v + 1 end
    local B = {} B.__index = B
    function B:get() return self.v * 2 end
    local objs = {setmetatable({v = 1}, A), setmetatable({v = 2}, B)}
    function f(o) local r = o:get() return r end
    for k = 1, 300 do f(objs[k % 2 + 1]) end";
    let ir = ir_of(src, "f", 0);
    assert_eq!(ir.matches("call r").count(), 1, "{ir}");
    assert!(
        ir.contains("imul") || ir.contains("ishl") || ir.contains("iadd"),
        "{ir}"
    );
}

/// A `setmetatable` result with a constant metatable has the class its
/// last move gave the table's shape, guarded after the call: its
/// metamethods are constants that inline (W3 §9). (A second `setmetatable`
/// would drop the fact: it may reshape any table.)
#[test]
fn setmetatable_results_have_classes() {
    let src = "local V = {}
    V.__index = V
    V.__add = function(a, b) return a.x + b.x end
    function f(i)
        local a = setmetatable({x = i}, V)
        return a + a
    end
    for k = 1, 300 do f(k) end";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("callmm"), "{ir}");
}

/// A metamethod the operand's class gives as a constant inlines (W3 §9).
#[test]
fn class_metamethods_inline() {
    let src = "local V = {}
    V.__index = V
    local function vec(x, y) return setmetatable({x = x, y = y}, V) end
    V.__add = function(a, b) return vec(a.x + b.x, a.y + b.y) end
    function f(i)
        local a, b = vec(i, 1), vec(1, 2)
        local t = a.x + b.y
        local r = a + b
        return r.x + t
    end
    for k = 1, 300 do f(k) end";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("callmm"), "{ir}");
}

/// A region frame's self tail call up to a real call's `top` is its loop:
/// `ack`'s region does not leave for the interpreter's `TAILCALL`.
#[test]
fn self_tail_calls_up_to_top_loop() {
    let src = "function ack(m, n)
        if m == 0 then return n + 1 end
        if n == 0 then return ack(m - 1, 1) end
        return ack(m - 1, ack(m, n - 1))
    end
    for k = 1, 30 do ack(2, 3) end";
    let ir = ir_of(src, "ack", 0);
    assert!(!ir.contains("dispatch"), "{ir}");
}

/// A self-call's result is what the region's returns give (W4 step 6):
/// its guards are refinements, and one landing guard per call tests it
/// where the ways back other than the fast return land.
#[test]
fn self_call_results_are_trusted() {
    let src = "function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end
    for k = 1, 300 do fib(10) end";
    let ir = ir_of(src, "fib", 0);
    assert_eq!(ir.matches("(landing)").count(), 4, "{ir}");
    assert_eq!(ir.matches("guard val:small").count(), 5, "{ir}");
}

/// Callees known by their prototype alone inline (W3 step 8): closures of
/// one prototype taking turns at a site, guarded by the prototype, and one
/// the loop makes, without a guard.
#[test]
fn prototype_known_callees_inline() {
    let src = "local function mk(k) return function(x) return x * k end end
    local fs = {mk(2), mk(3)}
    function f(n)
        local s = 0
        for i = 1, n do
            s = s + fs[i % 2 + 1](i)
            local add = function(x) return x + i end
            s = s + add(1)
        end
        return s
    end
    for k = 1, 300 do f(10) end";
    let ir = ir_of(src, "f", 0);
    assert!(!ir.contains("call r"), "{ir}");
    // One prototype guard per copy of the loop body (`fs`'s site), as many
    // as the closures made, which need none.
    assert_eq!(
        ir.matches("guard.proto").count(),
        ir.matches("call.closure").count(),
        "{ir}"
    );
}

/// Constructors and concatenation inline (W3 step 3): `NEWTABLE` takes its
/// template by address, `SETLIST` stores from the body's slots, and the
/// only call left is `table.unpack`, made from `copy`'s body.
#[test]
fn constructors_inline() {
    let src = "local function point(x, y) return {x = x, y = y} end
    local function triple(a) return {a, a + 1, a + 2} end
    local function copy(t) return {0, table.unpack(t)} end
    local function name(i) return 'n' .. i end
    local src = {7, 8, 9}
    function f(n)
        local s = 0
        for i = 1, n do
            s = s + point(i, i).x + #triple(i) + #copy(src) + #name(i)
        end
        return s
    end
    for k = 1, 300 do f(10) end";
    let ir = ir_of(src, "f", 0);
    for op in ["newtable", "setlist", "concat"] {
        assert!(ir.contains(op), "{op}: {ir}");
    }
    assert!(
        ir.lines()
            .filter(|l| l.contains("call r"))
            .all(|l| l.contains(" after f")),
        "{ir}"
    );
}

/// Callees storing through the table helpers inline (W2.7): `rawset`,
/// `table.insert`, `table.remove`, `setmetatable` and a generic store touch
/// table memory alone; the builtins' fallbacks are calls from the bodies.
/// (A tail call still keeps a callee out, `inline::needs_frame`.)
#[test]
fn table_helpers_inline() {
    let src = "local mt = {}
    local keys = {'p', 'q', 'r'}
    local function put(t, k, v) rawset(t, k, v) end
    local function push(t, v) table.insert(t, v) end
    local function pop(t) local v = table.remove(t) return v end
    local function mk(o) local r = setmetatable(o, mt) return r end
    local function set(t, k, v) t[k] = v end
    function f(t, n)
        local s = 0
        for i = 1, n do
            put(t, 'a', i); push(t, i); s = s + pop(t)
            local o = mk({x = i}); set(o, keys[i % 3 + 1], i); s = s + o.x
        end
        return s
    end
    for k = 1, 300 do f({}, 10) end";
    let ir = ir_of(src, "f", 0);
    assert!(
        ir.lines()
            .filter(|l| l.contains("call r"))
            .all(|l| l.contains(" after f")),
        "{ir}"
    );
    for h in [
        "call.rawset",
        "call.tinsert",
        "call.tremove",
        "call.setmetatable",
        "call.newindex",
    ] {
        assert!(ir.contains(h), "{h} in {ir}");
    }
}

/// Array accesses read the class only for a nil element (W2.3): a read in
/// the failing arm of its use's type guard, a varying table's store after
/// testing the old value (none over an element read and typed already); an
/// invariant table's store guards it once.
#[test]
fn array_classes_wait_for_nils() {
    let src = "function f(m, n, j)
        local s = 0
        for i = 1, n do local row = m[i]; s = s + row[j]; row[j] = s; row[j + 1] = s end
        return s
    end
    function g(t, n, v) for i = 1, n do t[i] = v end end
    keep = {}
    for i = 1, 20 do keep[i] = {1, 2, 3} end
    for k = 1, 300 do f(keep, 20, 2); g(keep[1], 3, k) end";
    // Lines with `s` outside the cold arms.
    let hot = |ir: &str, s: &str| {
        let mut cold = false;
        ir.lines()
            .filter(|l| {
                if !l.starts_with(' ') {
                    cold = l.contains(" cold");
                }
                !cold && l.contains(s)
            })
            .count()
    };
    let ir = ir_of(src, "f", 0);
    assert_eq!(hot(&ir, "tabhasmm"), 0, "{ir}");
    // `row[j + 1]`'s old value only.
    assert_eq!(2 * hot(&ir, "istype val:nil"), hot(&ir, "arrstore"), "{ir}");
    assert!(!ir.contains("guard.miss"), "{ir}");
    let ir = ir_of(src, "g", 0);
    assert_eq!(hot(&ir, "arrload"), 0, "{ir}");
    assert_eq!(hot(&ir, "tabhasmm"), 1, "{ir}");
}

/// A table loaded from a slot that tracks its tables' stable shape needs no
/// shape guard (ws2 §6.1-6.2), in a loop over a list too: only the list's
/// head, a parameter, is guarded.
#[test]
fn loaded_tables_keep_their_tracked_shape() {
    let src = "for i = 1, 100 do keep = {v = i, next = keep} end
    function f(l) local s = 0 while l do s = s + l.v l = l.next end return s end
    for k = 1, 30 do f(keep) end";
    let ir = ir_of(src, "f", 0);
    let guards = ir.lines().filter(|l| l.contains("guardshape")).count();
    assert_eq!(guards, 1, "{ir}");
}

/// A shape guarded before a call holds after it while stable: the call's
/// lazy exit catches a table leaving it.
#[test]
fn stable_shapes_outlive_calls() {
    let src = "local n = 0
    local function g() n = n + 1 return 1 end
    function f(o) local s = o.x + g() s = s + o.y + g() return s + o.x end
    keep = {x = 1, y = 2}
    for k = 1, 300 do f(keep) end";
    let ir = ir_of(src, "f", 0);
    let guards = ir.lines().filter(|l| l.contains("guardshape")).count();
    assert_eq!(guards, 1, "{ir}");
}

#[test]
fn nilable_slots_load_with_nil() {
    let src = "a, b = {o = {}}, {o = {}}
    function f(t) if t.o then return 1 end return 0 end
    for k = 1, 300 do f(a); f(b) end
    b.o = nil
    for k = 1, 300 do f(a) end";
    let ir = ir_of(src, "f", 0);
    // A nil store to one table's slot is in every table's kind of it.
    assert!(
        ir.contains("val:nil|false|true|tab|fun|thr = tabload"),
        "{ir}"
    );
}

#[test]
fn number_slots_take_what_the_site_saw() {
    // The slot holds small integers and floats; this site saw floats only.
    let src = "zero = {x = 0}
    function f(p, n) local s = 0.0 for i = 1, n do s = s + p.x * 2.0 end return s end
    for k = 1, 300 do keep = {x = 1.5}; f(keep, 10) end";
    let ir = ir_of(src, "f", 0);
    assert!(ir.contains("val:small|float = tabload"), "{ir}");
    assert!(!ir.contains("tof64"), "{ir}");
}

/// A loop entered with its first store in a cold arm, whose hot blocks after
/// the arm are reached only through it, compiles with every check on in
/// debug builds: the allocator's checker visits blocks in their order, so
/// the allocator gets them in reverse postorder, cold arms in place.
/// Inlining is off: `new`'s own regions are the subject.
#[test]
fn cold_arm_successors_pass_the_allocator_checker() {
    let src = "Src = {a = 1, b = 2, c = 3, d = 4}
    function new()
      local obj = {x = 1}
      for k, v in pairs(Src) do obj[k] = v end
      return obj
    end
    for i = 1, 3000 do new() end";
    let mut lua = Lua::new();
    lua.set_jit(true);
    lua.set_jit_inline(false);
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
            .raw_get(Value::string(LuaString::new(ctx, b"new")));
        let lf = f.get_function().and_then(|f| f.as_lua()).unwrap();
        let st = lf.proto.jit.get().expect("compiled").borrow();
        assert!(
            st.strikes == 0 && st.regions.len() >= 2,
            "{} strikes",
            st.strikes
        );
    });
}

/// A test of a register held unboxed (a float `and`/`or` picks) tests its
/// box, which the type folds; the verifier checks it right after the build.
#[test]
fn unboxed_registers_test_as_truthy() {
    let src = "function pick(a, b)
      local t1 = a - b
      return t1 > 0 and t1 or b
    end
    for i = 1, 300 do pick(i + 0.5, 3.5) end";
    let ir = ir_of(src, "pick", 0);
    assert!(!ir.contains("isfalsy"), "{ir}");
}

/// How many `br`s branch on a value an instruction containing `op` defines
/// from the constant `k`, i32 or f64 (the program's compare, not a loop's).
fn branches_on(ir: &str, op: &str, k: i32) -> usize {
    let konst = [format!("= ki32 {k}"), format!("= kf64 {k}.0")];
    let ks: Vec<&str> = ir
        .lines()
        .filter(|l| konst.iter().any(|c| l.trim_end().ends_with(c.as_str())))
        .filter_map(|l| l.trim().split(':').next())
        .collect();
    let defs: Vec<&str> = ir
        .lines()
        .filter(|l| l.contains(op))
        .filter(|l| {
            let args = l.split(op).nth(1).unwrap_or("");
            args.split([',', ' ']).any(|a| ks.contains(&a.trim()))
        })
        .filter_map(|l| l.trim().split(':').next())
        .collect();
    ir.lines()
        .filter_map(|l| l.trim().strip_prefix("br "))
        .filter(|l| defs.iter().any(|d| l.split(' ').next() == Some(*d)))
        .count()
}

/// A branch an edge of an equal one dominates folds (W1 5.3): one branch
/// on `x < 3` per copy of the loop body (the peeled one, the loop's).
#[test]
fn dominated_branches_fold() {
    let src = "function f(x, n)
        local s = 0
        for i = 1, n do
            if x < 3 then
                s = s + 1
                if x < 3 then s = s + 2 end
            end
        end
        return s
    end
    for r = 1, 300 do f(r % 7, 10) end";
    let ir = ir_of(src, "f", 0);
    assert_eq!(branches_on(&ir, "cmp.lt", 3), 2, "{ir}");
}

/// A merge whose branch each incoming edge decides is copied into the
/// edges (W1 5.4): the second `x < 3` and the test of `y` go.
#[test]
fn merges_an_edge_decides_are_copied() {
    let src = "function f(x, t, n)
        local s = 0
        for i = 1, n do
            local y
            if x < 3 then y = t[1] else y = 2 end
            if x < 3 then s = s + y else s = s - 1 end
        end
        return s
    end
    function g(x, n)
        local s = 0
        for i = 1, n do
            local y = x < 3 and i or false
            if y then s = s + y end
        end
        return s
    end
    local t = {5, 6, 7, 8}
    for r = 1, 300 do f(r % 7, t, 10); g(r % 7, 10) end";
    let ir = ir_of(src, "f", 0);
    assert_eq!(branches_on(&ir, "cmp.lt", 3), 2, "{ir}");
    let ir = ir_of(src, "g", 0);
    assert!(!ir.contains("isfalsy"), "{ir}");
}

/// How many branches outside cold blocks reach the same block with the same
/// arguments on both edges, through blocks that only jump on.
fn idle_branches(ir: &str) -> usize {
    let mut blocks: Vec<(String, bool, Vec<String>)> = Vec::new();
    for l in ir.lines() {
        if let Some(h) = l.strip_prefix('b').filter(|_| l.ends_with(':')) {
            let name = format!("b{}", &h[..h.find('(').unwrap()]);
            blocks.push((name, l.contains(" cold"), Vec::new()));
        } else if l.starts_with("  ")
            && let Some(b) = blocks.last_mut()
        {
            b.2.push(l.trim().to_string());
        }
    }
    let find = |n: &str| blocks.iter().find(|b| b.0 == n);
    let end = |mut t: String, mut args: String| {
        for _ in 0..8 {
            match find(&t) {
                Some((_, _, body)) if args.is_empty() && body.len() == 1 => {
                    let Some(j) = body[0].strip_prefix("jump -> ") else {
                        break;
                    };
                    let (n, a) = j.split_once('(').unwrap();
                    (t, args) = (n.to_string(), a.trim_end_matches(')').to_string());
                }
                _ => break,
            }
        }
        (t, args)
    };
    let edge = |e: &str| {
        let (n, a) = e.trim().split_once('(').unwrap();
        (n.to_string(), a.trim_end_matches(')').to_string())
    };
    blocks
        .iter()
        .filter(|(_, cold, body)| {
            let Some(br) = body.last().and_then(|l| l.strip_prefix("br ")) else {
                return false;
            };
            let mut e = br.split(" -> ").skip(1).map(edge);
            let (a, b) = (e.next().unwrap(), e.next().unwrap());
            !cold && end(a.0, a.1) == end(b.0, b.1)
        })
        .count()
}

/// A branch whose edges meet before anything runs goes: here once the
/// body of the `if` is dead.
#[test]
fn branches_deciding_nothing_fold() {
    let src = "function f(x, n)
        local s = 0
        for i = 1, n do
            if x < 3 then local y = x & 1 end
            s = s + i
        end
        return s
    end
    for r = 1, 300 do f(r % 7, 10) end";
    let ir = ir_of(src, "f", 0);
    assert_eq!(idle_branches(&ir), 0, "{ir}");
    assert_eq!(branches_on(&ir, "cmp.lt", 3), 0, "{ir}");
}

/// Elements of two shapes, read in the caller then in an inlined body: the
/// body starts with the caller's facts (W3 §3) and tests no shape it knows.
#[test]
fn bodies_start_with_their_callers_facts() {
    let src = "local A = {}
    for i = 1, 50 do A[#A + 1] = {v = i, w = 1} A[#A + 1] = {u = 0, v = i, w = 2} end
    local function w(n) return n.w end
    function f(n) local s = 0 for i = 1, n do local o = A[i] s = s + o.v + w(o) end return s end
    for k = 1, 30 do f(100) end";
    let ir = ir_of(src, "f", 0);
    let body = ir
        .lines()
        .filter(|l| l.contains("guardshape") && l.contains("before f0"));
    assert_eq!(body.count(), 0, "{ir}");
}

/// The same read first in an inlined body: the caller continues with the
/// facts of the body's returns and tests no shape again.
#[test]
fn callers_continue_with_their_bodies_facts() {
    let src = "local A = {}
    for i = 1, 50 do A[#A + 1] = {v = i, w = 1} A[#A + 1] = {u = 0, v = i, w = 2} end
    local function w(n) return n.w end
    function f(n) local s = 0 for i = 1, n do local o = A[i] s = s + w(o) + o.v end return s end
    for k = 1, 30 do f(100) end";
    let ir = ir_of(src, "f", 0);
    let caller = ir
        .lines()
        .filter(|l| l.contains("guardshape") && !l.contains("before f0"));
    assert_eq!(caller.count(), 0, "{ir}");
}

/// An inlined body's results keep the types its returns give them: the
/// builder's join parameters are untyped but this one, an `i32` the caller
/// adds without a guard.
#[test]
fn inlined_results_keep_their_types() {
    let src = "local function inc(x) if x > 3 then return x + 1 end return x - 1 end
    function f(n) local s = 0 for i = 1, n do s = s + inc(i) end return s end
    for r = 1, 300 do f(10) end";
    let ir = built_ir_of(src, "f", 0);
    let typed = ir
        .lines()
        .filter(|l| l.starts_with('b') && l.contains(": i32)"));
    assert_eq!(typed.count(), 1, "{ir}");
}
