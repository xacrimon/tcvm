local function mix(n)
    local a, b, c, d = 1+n, 2, 3, 4
    local e, f, g, h = 5, 6+n, 7, 8
    local p, q, r, s = 9, 10, 11+n, 12
    for i = 1, n do
        a = a + i * 3
        b = b ~ (a << 1)
        c = c + b - i
        d = d * 5 + a
        e = (e + c) % 1000000007
        f = f ~ (d >> 2)
        g = g + e + f
        h = (h + g) % 1000003
        p = p + a - g
        q = q ~ (h << 2)
        r = r + p + i
        s = (s + q + r) % 998244353
        a = a + s
    end
    return a + b + c + d + e + f + g + h + p + q + r + s
end

local acc = 0
for n = 1, 2000 do
    acc = acc + mix(n)
end
print(acc)
