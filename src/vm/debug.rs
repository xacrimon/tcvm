//! Debug-info lookups shared by error reporting: chunk-name formatting and
//! the `source:line:` of a call level (ldebug.c / lauxlib.c counterparts).

use crate::env::error::Error;
use crate::env::function::LuaFn;
use crate::env::string::LuaString;
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::lua::Context;
use crate::vm::frame::{self, Frame};
use crate::vm::unwind::OpError;

/// `luaO_chunkid`: how a chunk name prints in messages, capped at
/// `LUA_IDSIZE` (60) bytes. `=name` is literal, `@path` keeps the tail of
/// long paths, anything else is source text shown as `[string "..."]`.
pub(crate) fn chunk_id(source: &[u8]) -> Vec<u8> {
    const IDSIZE: usize = 60;
    match source.first() {
        Some(b'=') => {
            let name = &source[1..];
            name[..name.len().min(IDSIZE - 1)].to_vec()
        }
        Some(b'@') => {
            let name = &source[1..];
            if source.len() <= IDSIZE {
                name.to_vec()
            } else {
                // Room for "..." plus the terminator luac reserves.
                let keep = IDSIZE - 3 - 1;
                [b"...", &name[name.len() - keep..]].concat()
            }
        }
        _ => {
            const PRE: &[u8] = b"[string \"";
            const POS: &[u8] = b"\"]";
            let room = IDSIZE - PRE.len() - 3 - POS.len() - 1;
            let first_line = source.split(|&b| b == b'\n').next().unwrap_or(b"");
            let mut out = PRE.to_vec();
            if source.len() < room && first_line.len() == source.len() {
                out.extend_from_slice(source);
            } else {
                out.extend_from_slice(&first_line[..first_line.len().min(room)]);
                out.extend_from_slice(b"...");
            }
            out.extend_from_slice(POS);
            out
        }
    }
}

/// The top frame, when it is a Lua frame.
fn top_lua<'gc>(ts: &ThreadState<'gc>) -> Option<Frame<'gc>> {
    frame::frames(ts).next().filter(|f| !f.is_native())
}

/// `luaL_where`: `"chunk:line: "` for the frame `level` below the top (the
/// top frame is level 0, so a raising native's caller is level 1). Empty
/// when that frame isn't a Lua function or doesn't exist, like the reference.
pub(crate) fn where_prefix(ts: &ThreadState<'_>, level: usize) -> Vec<u8> {
    let Some(f) = frame::frames(ts).nth(level) else {
        return Vec::new();
    };
    if f.is_native() {
        return Vec::new();
    }
    let Some(line) = f.line() else {
        return Vec::new();
    };
    let mut out = chunk_id(f.closure().proto.source.as_bytes());
    out.extend_from_slice(format!(":{line}: ").as_bytes());
    out
}

/// Apply an error's pending position level against the thread's frames: a
/// string message gets the `where_prefix`; anything else is left alone
/// (`luaB_error` only decorates strings). The result carries level 0.
pub(crate) fn locate<'gc>(ctx: Context<'gc>, ts: &ThreadState<'gc>, err: Error<'gc>) -> Error<'gc> {
    let level = err.level();
    let Some(msg) = err.value().get_string().filter(|_| level > 0) else {
        return err.with_level(0);
    };
    let prefix = where_prefix(ts, level);
    if prefix.is_empty() {
        return err.with_level(0);
    }
    let text = [prefix.as_slice(), msg.as_bytes()].concat();
    err.with_value(ctx, Value::string(LuaString::new(ctx, &text)))
}

/// [`locate`] for an error raised by a native with no frame of its own (a
/// tail-called one): its level 1 is the top frame.
pub(crate) fn locate_unframed<'gc>(
    ctx: Context<'gc>,
    ts: &ThreadState<'gc>,
    err: Error<'gc>,
) -> Error<'gc> {
    let level = err.level();
    let Some(msg) = err.value().get_string().filter(|_| level > 0) else {
        return err.with_level(0);
    };
    let prefix = where_prefix(ts, level - 1);
    if prefix.is_empty() {
        return err.with_level(0);
    }
    let text = [prefix.as_slice(), msg.as_bytes()].concat();
    err.with_value(ctx, Value::string(LuaString::new(ctx, &text)))
}

/// An error with `msg` prefixed by the position of frame `level` below the
/// top; level 0 is the running Lua frame, for opcode faults.
pub(crate) fn error_at<'gc>(
    ctx: Context<'gc>,
    ts: &ThreadState<'gc>,
    msg: &str,
    level: usize,
) -> Error<'gc> {
    let mut text = where_prefix(ts, level);
    text.extend_from_slice(msg.as_bytes());
    Error::new(ctx, Value::string(LuaString::new(ctx, &text)))
}

/// `luaT_objtypename`: a table or userdata whose metatable has a string
/// `__name` reports that instead of its basic type. Unlike `luaL_tolstring`
/// and argument errors, it ignores the per-type metatables.
pub(crate) fn object_type_name<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> String {
    let mt = v
        .get_table()
        .and_then(|t| t.metatable())
        .or_else(|| v.get_userdata().and_then(|u| u.metatable()));
    let name = mt.map(|mt| mt.raw_get(Value::string(ctx.symbols().name)));
    match name.and_then(|n| n.get_string()) {
        Some(s) => String::from_utf8_lossy(s.as_bytes()).into_owned(),
        None => v.type_name().to_owned(),
    }
}

/// The reference message for an opcode fault (ldebug.c's `luaG_*error`
/// family), minus the variable-name suffix (`(local 'x')` etc.), which
/// needs register-to-name attribution that isn't implemented yet.
pub(crate) fn op_error_message<'gc>(
    ctx: Context<'gc>,
    ts: &ThreadState<'gc>,
    kind: OpError<'gc>,
) -> String {
    let tn = |v| object_type_name(ctx, v);
    let is_number = |v: Value<'gc>| v.get_integer().is_some() || v.get_float().is_some();
    match kind {
        OpError::Index(v) => format!("attempt to index a {} value", tn(v)),
        OpError::Call(v) => format!("attempt to call a {} value", tn(v)),
        OpError::Len(v) => format!("attempt to get length of a {} value", tn(v)),
        // `luaG_opinterror`: blame the first operand unless it is a number.
        OpError::Arith(a, b) => {
            let culprit = if is_number(a) { b } else { a };
            format!("attempt to perform arithmetic on a {} value", tn(culprit))
        }
        // `luaT_trybinTM`: two numbers that reach here failed integer
        // conversion; otherwise blame the non-number.
        OpError::Bitwise(a, b) if is_number(a) && is_number(b) => {
            "number has no integer representation".to_owned()
        }
        OpError::Bitwise(a, b) => {
            let culprit = if is_number(a) { b } else { a };
            format!(
                "attempt to perform bitwise operation on a {} value",
                tn(culprit)
            )
        }
        // `luaG_concaterror`: strings and numbers concatenate, so blame the
        // other operand.
        OpError::Concat(a, b) => {
            let culprit = if a.get_string().is_some() || is_number(a) {
                b
            } else {
                a
            };
            format!("attempt to concatenate a {} value", tn(culprit))
        }
        OpError::Compare(a, b) => {
            let (t1, t2) = (tn(a), tn(b));
            if t1 == t2 {
                format!("attempt to compare two {t1} values")
            } else {
                format!("attempt to compare {t1} with {t2}")
            }
        }
        OpError::DivByZero => "attempt to divide by zero".to_owned(),
        OpError::ModByZero => "attempt to perform 'n%0'".to_owned(),
        OpError::IndexChainLoop => "'__index' chain too long; possible loop".to_owned(),
        OpError::NewIndexChainLoop => "'__newindex' chain too long; possible loop".to_owned(),
        OpError::CallChainTooLong => "'__call' chain too long".to_owned(),
        OpError::GlobalRedefined(k) => {
            let name = top_lua(ts)
                .and_then(|f| f.closure().proto.constants.get(k as usize).copied())
                .and_then(|v| v.get_string())
                .map_or_else(
                    || "?".to_owned(),
                    |s| String::from_utf8_lossy(s.as_bytes()).into_owned(),
                );
            format!("global '{name}' already defined")
        }
        OpError::ForStepZero => "'for' step is zero".to_owned(),
        OpError::ForNotNumber(what, v) => {
            format!("bad 'for' {what} (number expected, got {})", tn(v))
        }
        OpError::NilIndex => "table index is nil".to_owned(),
        OpError::NanIndex => "table index is NaN".to_owned(),
        OpError::StackOverflow => "stack overflow".to_owned(),
        OpError::VarargN => "vararg table has no proper 'n'".to_owned(),
        OpError::NonClosable(reg) => {
            let name = top_lua(ts)
                .and_then(|f| local_name(f.closure(), reg, f.pc_index().saturating_sub(1)))
                .map_or_else(
                    || "?".to_owned(),
                    |s| String::from_utf8_lossy(s.as_bytes()).into_owned(),
                );
            format!("variable '{name}' got a non-closable value")
        }
        OpError::Internal(what) => format!("internal VM error: {what}"),
        OpError::Thrown(_) => unreachable!("a thrown error carries its own value"),
    }
}

/// Name of the local in register `reg` at instruction `pc` (`luaF_getlocalname`).
pub(crate) fn local_name<'gc>(closure: LuaFn<'gc>, reg: u8, pc: usize) -> Option<LuaString<'gc>> {
    let pc = pc as u32;
    closure
        .proto
        .locvars
        .iter()
        .filter(|v| v.start_pc <= pc && pc < v.end_pc)
        .nth(reg as usize)
        .map(|v| v.name)
}

/// The error for a call that would cross the thread's stack limit: "stack
/// overflow" at the running frame's position, or, when a message handler
/// already runs in the headroom, "error in error handling", which skips the
/// handler (`luaD_errerr`).
pub(crate) fn stack_overflow<'gc>(
    ctx: Context<'gc>,
    ts: &ThreadState<'gc>,
    positioned: bool,
) -> Error<'gc> {
    if ts.in_error_headroom() {
        error_in_error_handling(ctx)
    } else {
        let msg = op_error_message(ctx, ts, OpError::StackOverflow);
        // `luaG_runerror`: positioned only when the running function is Lua.
        if positioned && !ts.top_is_native() {
            error_at(ctx, ts, &msg, 0)
        } else {
            Error::new(ctx, Value::string(LuaString::new(ctx, msg.as_bytes())))
        }
    }
}

/// Raised in place of an error the message handler can't deal with; marked
/// handled so it goes straight to the catcher.
pub(crate) fn error_in_error_handling(ctx: Context<'_>) -> Error<'_> {
    let msg = LuaString::new(ctx, b"error in error handling");
    Error::new(ctx, Value::string(msg)).mark_handled()
}

#[cfg(test)]
mod tests {
    use super::chunk_id;

    // Expected strings come from `lua` 5.5.1 via `load(src, name)` +
    // `pcall(error)`.
    #[test]
    fn chunk_id_matches_reference() {
        let id = |s: &str| String::from_utf8(chunk_id(s.as_bytes())).unwrap();
        assert_eq!(id("=short"), "short");
        assert_eq!(id(&format!("={}", "n".repeat(70))), "n".repeat(59));
        assert_eq!(id("@file.lua"), "file.lua");
        assert_eq!(
            id(&format!("@{}.lua", "p".repeat(70))),
            format!("...{}.lua", "p".repeat(52))
        );
        assert_eq!(id("error('m')"), "[string \"error('m')\"]");
        assert_eq!(id("x = 1\nerror('m')"), "[string \"x = 1...\"]");
        let long = "error('m') -- 12345678901234567890123456789012345678901234567890";
        assert_eq!(
            id(long),
            "[string \"error('m') -- 1234567890123456789012345678901...\"]"
        );
    }
}
