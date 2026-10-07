//! Stores into objects a collection has already traced: the store handlers
//! test the gray bit and retry after the barrier, so a new object stored into
//! an old one survives the next collection. `yielder()` collects fully
//! mid-chunk.

use crate::common::{run_chunks, yielding_lua};

#[test]
fn stores_into_old_objects_are_barriered() {
    let mut lua = yielding_lua();
    let src = "old = {}
        for i = 1, 100 do old[i] = {} end
        local cell = {}
        local function set_cell(v) cell = v end
        yielder()
        for i = 1, 100 do
          old[i].v = {i}            -- SETFIELD, a transition then own slots
          old[i][1] = {i}           -- SETTABLE
          old[i].w = {i}
        end
        set_cell({7})                -- SETUPVAL into a closed cell
        glob = {old[1]}              -- SETTABUP
        old[101] = {1, 2, 3}         -- SETLIST into a fresh table
        yielder()
        local s = 0
        for i = 1, 100 do s = s + old[i].v[1] + old[i].w[1] + (old[i][1] or {0})[1] end
        return tostring(s + cell[1] + #glob + old[101][3])";
    assert_eq!(run_chunks(&mut lua, src), "15161");
}
