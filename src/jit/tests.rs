//! Tests of the compiler stages on functions warmed up in the interpreter.

use crate::env::{LuaString, Value};
use crate::{Executor, LoadError, Lua};

/// Run `src` in the interpreter, then build the IR of global function
/// `name` from `pc` with the feedback it left, whatever the JIT mode.
pub(crate) fn ir_of(src: &str, name: &str, pc: u32) -> String {
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
        let e = st.entries.remove(0);
        assert_eq!(&*e.back_edges, &[back as u32]);
        super::retire_entry(ctx, lf, &mut st, &e);
        drop(st);
        assert_eq!(code(header).op(), Op::LOOP);
        assert_eq!(target(back), header + 1);
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
/// the parameters here are only ever loaded, though inside the loop the
/// snapshots name their guarded copies.
#[test]
fn snapshots_leave_out_unchanged_slots() {
    let src = "function h(a, b)
        local s = 0
        for i = 1, 10 do s = s + a.x * i + b end
        return s + a.y
    end
    for i = 1, 300 do h({x = 1.5, y = 2}, i) end";
    let ir = ir_of(src, "h", 0);
    let snaps: Vec<&str> = ir
        .lines()
        .filter_map(|l| l.split_once('[').map(|x| x.1))
        .collect();
    assert!(!snaps.is_empty(), "{ir}");
    assert!(
        snaps
            .iter()
            .all(|s| !s.contains("r0=") && !s.contains("r1=")),
        "{ir}"
    );
}

/// Registers a call does not write keep their values in their slots and
/// their types after it: the entry guards cover the uses after each call.
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
    assert_eq!(count("load r0"), 3, "{ir}");
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
    for i = 1, 300 do f({x = 1, y = 2, z = 3}, i % 2 == 0) end";
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
