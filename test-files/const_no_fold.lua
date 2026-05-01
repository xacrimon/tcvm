-- Each of these must NOT fold per Lua's `validop` rules. The bytecode
-- should retain the runtime arith op (DIV/IDIV/MOD/BAND/SUB).

-- Division by zero — runtime would error or produce inf/NaN, so fold bails.
a = 1 / 0
b = 1 // 0
c = 5 % 0

-- Bitwise on non-integer-convertible float — runtime conversion would error.
d = 1.5 & 2

-- Float result that is NaN or zero — fold bails to avoid -0.0 collapse.
e = 0.0 - 0.0

-- Non-literal operand — no fold possible.
f = x + 1
g = -x

-- # (length) is never folded, even on literal strings.
h = #"abc"
