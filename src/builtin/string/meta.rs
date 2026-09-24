//! The metatable shared by all strings: `__index = string`, plus arithmetic
//! metamethods through which numeric strings coerce (lstrlib.c's
//! `stringmetamethods`); the VM itself never converts strings to numbers.

use crate::Context;
use crate::builtin::util::{AdjustResults, str_to_number};
use crate::dmm::Mutation;
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use crate::vm::num::{self, SlowNum};
use crate::vm::sequence::{BoxSequence, CallbackAction};

pub(super) fn install<'gc>(ctx: Context<'gc>, lib: Table<'gc>) {
    let mt = Table::new(ctx);
    for (name, f) in arith_natives(ctx) {
        let f = Function::new_native(ctx.mutation(), f, Box::new([]));
        mt.raw_set(ctx, Value::string(name), Value::function(f));
    }
    mt.raw_set(
        ctx,
        Value::string(ctx.symbols().mm_index),
        Value::table(lib),
    );
    ctx.set_metatable_of(Value::string(LuaString::new(ctx, b"")), Some(mt));
}

type ArithFn<'gc> = fn(&Mutation<'gc>, Value<'gc>, Value<'gc>) -> SlowNum<'gc>;

macro_rules! arith_natives {
    ($($native:ident => $op:expr, $mm:ident, $name:literal;)*) => {
        $(
            fn $native<'gc>(
                ctx: Context<'gc>,
                _closure: &NativeClosure<'gc>,
                stack: Stack<'gc, '_>,
            ) -> Result<CallbackAction<'gc>, Error<'gc>> {
                arith(ctx, stack, $op, ctx.symbols().$mm, $name)
            }
        )*

        fn arith_natives<'gc>(ctx: Context<'gc>) -> [(LuaString<'gc>, NativeFn); ${count($native)}] {
            [$((ctx.symbols().$mm, $native as NativeFn),)*]
        }
    };
}

arith_natives! {
    arith_add => num::op_arith_slow::<num::Add>, mm_add, "add";
    arith_sub => num::op_arith_slow::<num::Sub>, mm_sub, "sub";
    arith_mul => num::op_arith_slow::<num::Mul>, mm_mul, "mul";
    arith_mod => num::op_arith_slow::<num::Mod>, mm_mod, "mod";
    arith_pow => num::op_arith_slow::<num::Pow>, mm_pow, "pow";
    arith_div => num::op_arith_slow::<num::Div>, mm_div, "div";
    arith_idiv => num::op_arith_slow::<num::IDiv>, mm_idiv, "idiv";
    arith_unm => unm, mm_unm, "unm";
}

/// `lua_arith(LUA_OPUNM)` negates the topmost operand, which is `b`.
fn unm<'gc>(mc: &Mutation<'gc>, _a: Value<'gc>, b: Value<'gc>) -> SlowNum<'gc> {
    match b.get_integer() {
        Some(i) => SlowNum::Value(Value::integer(mc, i.wrapping_neg())),
        None => SlowNum::Value(Value::float(-b.read_float())),
    }
}

/// `v` as a number, if it is one or a string that converts entirely.
fn tonum<'gc>(mc: &Mutation<'gc>, v: Value<'gc>) -> Option<Value<'gc>> {
    if v.get_integer().is_some() || v.is_float() {
        return Some(v);
    }
    str_to_number(v.get_string()?.as_bytes()).map(|n| n.into_value(mc))
}

fn arith<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    op: ArithFn<'gc>,
    mm: LuaString<'gc>,
    name: &'static str,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let mc = ctx.mutation();
    let a = tonum(mc, stack.get(0));
    // lstrlib's `tonum` pushes the converted first operand, which then stands
    // in for a missing second one.
    let b = match a {
        Some(a) if stack.len() < 2 => Some(a),
        _ => tonum(mc, stack.get(1)),
    };
    let (Some(a), Some(b)) = (a, b) else {
        return trymt(ctx, stack, mm, name);
    };
    // Raised from inside the native, so without a position (level 0).
    let result = match op(mc, a, b) {
        SlowNum::Value(v) => v,
        SlowNum::ModByZero => {
            return Err(Error::from_str(ctx, "attempt to perform 'n%0'").with_level(0));
        }
        SlowNum::DivByZero => {
            return Err(Error::from_str(ctx, "attempt to divide by zero").with_level(0));
        }
        SlowNum::NotNumbers => unreachable!("both operands converted"),
    };
    stack.ret1(result);
    Ok(CallbackAction::Return)
}

/// The string operand's metamethod ran first, so only the second operand's
/// can still apply (lstrlib's `trymt`).
fn trymt<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    mm_name: LuaString<'gc>,
    name: &'static str,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let (a, b) = (stack.get(0), stack.get(1));
    let mm = match b.get_string() {
        Some(_) => Value::nil(),
        None => ctx.metamethod_of(b, mm_name),
    };
    if mm.is_nil() {
        return Err(Error::from_str(
            ctx,
            &format!(
                "attempt to {name} a '{}' with a '{}'",
                a.type_name(),
                b.type_name()
            ),
        ));
    }
    // `lua_call(L, 2, 1)`; the executor resolves a `__call` chain on `mm`.
    stack.replace(&[mm, a, b]);
    let then = BoxSequence::new(ctx.mutation(), AdjustResults(1));
    Ok(CallbackAction::call(Some(then)))
}
