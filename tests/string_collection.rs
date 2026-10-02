//! The interner doesn't keep strings alive (#179): one that nothing else
//! references is collected, and interning its bytes again later yields a fresh
//! string that still compares equal to every other copy made from then on.

use tcvm::{Executor, LoadError, Lua};

/// Run `src` to completion, then report live bytes after full collections.
///
/// Two cycles: a dead string is freed in the first, but a dead shape's box
/// stays until the next trace of its parent prunes the edge to it.
fn live_bytes_after(src: &str) -> usize {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("strings"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    for _ in 0..2 {
        lua.collect_all();
    }
    lua.live_bytes()
}

/// `baseline` and `dead` must end up within a tenth of the payload that `live`
/// keeps, or the strings were retained.
fn assert_reclaimed(baseline: &str, dead: &str, live: &str) {
    let baseline = live_bytes_after(baseline);
    let dead = live_bytes_after(dead);
    let live = live_bytes_after(live);
    let payload = live.saturating_sub(baseline);
    assert!(
        payload > 1_000_000,
        "payload too small to be conclusive: baseline={baseline} live={live}"
    );
    let retained = dead.saturating_sub(baseline);
    assert!(
        retained < payload / 10,
        "strings not reclaimed: baseline={baseline} dead={dead} live={live}"
    );
}

#[test]
fn unreferenced_strings_are_reclaimed() {
    assert_reclaimed(
        "local s for i = 1, 50000 do s = i end",
        "local s for i = 1, 50000 do s = 'k' .. i end",
        "keep = {} for i = 1, 50000 do keep[i] = 'k' .. i end",
    );
}

#[test]
fn dynamic_field_names_are_reclaimed() {
    // Each table adds one property from the root shape, so the root's
    // transition table sees every key.
    assert_reclaimed(
        "for i = 1, 50000 do local t = {} t[1] = i end",
        "for i = 1, 50000 do local t = {} t['f' .. i] = i end",
        "keep = {} for i = 1, 50000 do local t = {} t['f' .. i] = i keep[i] = t end",
    );
}

#[test]
fn live_strings_stay_unique_while_collecting() {
    // The garbage keeps the collector cycling through every phase; each
    // re-made string must intern to the copy still held in `keep`.
    let src = "local keep = {} \
               for i = 1, 100 do keep[i] = 'v' .. i end \
               for round = 1, 2000 do \
                 for i = 1, 100 do \
                   local s = 'v' .. i \
                   if s ~= keep[i] or rawequal(s, keep[i]) == false then error('duplicate ' .. s) end \
                   local junk = 'junk' .. round .. '_' .. i \
                 end \
               end";
    live_bytes_after(src);
}

#[test]
fn reinterned_strings_after_death_are_equal() {
    // `w0`..`w49` live only in locals, so they die and are made again while
    // the collector runs; each pair made together must still be one string.
    let src = "for round = 1, 20000 do \
                 local a, b = 'w' .. (round % 50), 'w' .. (round % 50) \
                 if a ~= b then error('unequal ' .. a) end \
                 local t = {} t[a] = round \
                 if t[b] ~= round then error('lookup ' .. a) end \
                 local junk = {} \
               end";
    live_bytes_after(src);
}

#[test]
fn string_bytes_round_trip_at_every_length() {
    // Lengths around the box's alignment, an embedded NUL, and a large one.
    let src = "for _, n in ipairs({0, 1, 7, 8, 9, 15, 16, 17, 4096}) do \
                 local parts = {} \
                 for i = 1, n do parts[i] = string.char((i * 37) % 256) end \
                 local s = table.concat(parts) \
                 if #s ~= n then error('length ' .. n) end \
                 for i = 1, n do \
                   if s:byte(i) ~= (i * 37) % 256 then error('byte ' .. i .. ' of ' .. n) end \
                 end \
               end";
    live_bytes_after(src);
}

#[test]
fn string_bytes_count_as_gc_memory() {
    // The bytes live in the string's own allocation, so the collector sees them. (The scripts'
    // other strings differ a little, hence not exactly 1 MiB.)
    let small = live_bytes_after("keep = 'x'");
    let large = live_bytes_after("keep = string.rep('x', 1 << 20)");
    assert!(
        large - small > 1_000_000,
        "1 MiB string not counted: small={small} large={large}"
    );
}
