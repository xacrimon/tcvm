//! A fast entry that allocates may leave for the collector only once the
//! call has landed: its results, nil padding and `top` must be in place when
//! dispatch resumes after it.

use crate::common::eval;

#[test]
fn allocating_fast_entries_land_before_a_collection() {
    let src = "local keep = {}
        local function tail(mt) return setmetatable({}, mt) end
        local function tails(s) return s:sub(2, 3) end
        for i = 1, 300000 do
          local t, extra = setmetatable({i}, {})
          assert(getmetatable(t) ~= nil and extra == nil)
          local u = tail({__index = keep})
          assert(getmetatable(u).__index == keep)
          local x, y = ('abcdef'):sub(2, 3)
          assert(x == 'bc' and y == nil)
          assert(select('#', tails('abcdef')) == 1)
          keep[#keep + 1] = t
          if #keep > 500 then keep = {} end
        end
        return 1";
    assert_eq!(eval::<i64>(src), 1);
}
