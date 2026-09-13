-- A register-pressure case for the JIT backend.
--
-- `mix` has twelve accumulators, which fits aarch64's twenty-register integer
-- pool with room to spare: it spills nothing there, and only spills on x86-64's
-- thirteen. That makes it useless for judging any change to spilling or
-- coalescing on the host architecture — the allocator never has to make a hard
-- decision.
--
-- `mix2` is built to force those decisions:
--
--   * **Twenty-eight accumulators**, all live across the whole loop, so the
--     working set exceeds every register file the backend targets.
--   * **A nested loop**, so a reload placed inside it is paid on every inner
--     iteration. Hoisting one out is worth four times what it looks like
--     statically, which is exactly the effect Belady-style spilling claims and a
--     static spill count cannot see.
--   * **A two-armed branch** whose arms touch disjoint sets of accumulators, so
--     the join has to reconcile two different live sets. Straight-line loop
--     bodies let a local heuristic and an optimal one agree; control-flow joins
--     are where they diverge.
--
-- Everything here stays inside the op set the backend compiles: integer
-- arithmetic, bitwise ops, comparisons, and numeric `for`. No calls, no tables,
-- nothing that can allocate or run a metamethod, since a region containing one
-- is declined rather than compiled.

local function mix2(n)
    local a, b, c, d = 1 + n, 2, 3, 4
    local e, f, g, h = 5, 6 + n, 7, 8
    local p, q, r, s = 9, 10, 11 + n, 12
    local t, u, v, w = 13, 14, 15 + n, 16
    local x, y, z, aa = 17, 18, 19 + n, 20
    local bb, cc, dd, ee = 21, 22, 23 + n, 24
    local ff, gg, hh, ii = 25, 26, 27 + n, 28

    for i = 1, n do
        -- Inner loop: short, hot, and reading values defined outside it.
        for j = 1, 4 do
            a = a + i * j
            b = b ~ (a << 1)
            c = c + b - j
        end

        -- The arms update disjoint accumulators, so neither set is live on both
        -- paths and the join must reconcile them.
        if (a ~ i) % 2 == 0 then
            d = d * 5 + a
            e = (e + c) % 1000000007
            f = f ~ (d >> 2)
            g = g + e + f
            t = t + d - g
            u = u ~ (t << 1)
            ff = ff + g - t
        else
            h = (h + c) % 1000003
            p = p + a - h
            q = q ~ (h << 2)
            r = r + p + i
            v = v + q - r
            w = w ~ (v >> 1)
            gg = gg + r - v
        end

        -- Read from both arms' results, so everything above stays live through
        -- the branch rather than dying on the path that did not write it.
        s = (s + q + r) % 998244353
        x = x + s - t
        y = y ~ (x << 2)
        z = z + y + u
        aa = (aa + z) % 999999937
        bb = bb + aa - v
        cc = cc ~ (bb << 1)
        dd = dd + cc + w
        ee = (ee + dd) % 998244353
        hh = hh + ee - ff
        ii = ii ~ (hh << 1)
        a = a + ii + gg
    end

    return a + b + c + d + e + f + g + h
        + p + q + r + s + t + u + v + w
        + x + y + z + aa + bb + cc + dd + ee
        + ff + gg + hh + ii
end

local acc = 0
for n = 1, 2000 do
    acc = acc + mix2(n)
end
print(acc)
