//! Debug-info lookups shared by error reporting: chunk-name formatting and
//! the `source:line:` of a call level (ldebug.c / lauxlib.c counterparts).

use crate::env::error::Error;
use crate::env::string::LuaString;
use crate::env::thread::{Frame, ThreadState};
use crate::env::value::Value;
use crate::lua::Context;
use crate::vm::interp::OpError;

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

/// Source line the Lua frame is currently executing; `pc` points past the
/// current instruction (see `LuaFrame::pc`), and 0 means not yet entered.
pub(crate) fn frame_line(frame: &Frame<'_>) -> Option<u32> {
    let Frame::Lua(lf) = frame else { return None };
    lf.closure.proto.line_for_pc(lf.pc_index().checked_sub(1)?)
}

/// `luaL_where`: `"chunk:line: "` for call `level`, counting the raising
/// native as 0 and `ts.frames.last()` as 1. Empty when that level isn't a
/// Lua function (or doesn't exist), exactly like the reference.
pub(crate) fn where_prefix(ts: &ThreadState<'_>, level: usize) -> Vec<u8> {
    let Some(frame) = level
        .checked_sub(1)
        .and_then(|depth| ts.frames.iter().rev().nth(depth))
    else {
        return Vec::new();
    };
    let (Frame::Lua(lf), Some(line)) = (frame, frame_line(frame)) else {
        return Vec::new();
    };
    let mut out = chunk_id(lf.closure.proto.source.as_bytes());
    out.extend_from_slice(format!(":{line}: ").as_bytes());
    out
}

/// Apply an error's pending position level against the raising thread's
/// frames: a string message gets the `where_prefix`; anything else is left
/// alone (`luaB_error` only decorates strings). The result carries level 0.
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
    Error::new(ctx, Value::string(LuaString::new(ctx, &text)))
}

/// `luaT_objtypename`: a table or userdata whose metatable has a string
/// `__name` reports that instead of its basic type.
pub(crate) fn object_type_name<'gc>(ctx: Context<'gc>, v: Value<'gc>) -> String {
    let mt = v
        .get_table()
        .and_then(|t| t.metatable())
        .or_else(|| v.get_userdata().and_then(|u| u.metatable()));
    let name = mt.map(|mt| mt.raw_get(Value::string(LuaString::new(ctx, b"__name"))));
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
        OpError::GlobalRedefined(k) => {
            let name = ts
                .top_lua()
                .and_then(|lf| lf.closure.proto.constants.get(k as usize).copied())
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
        OpError::Internal(what) => format!("internal VM error: {what}"),
    }
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
