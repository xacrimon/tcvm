-- Each assignment should compile to a single LOAD instruction.
-- The constant table contains the folded values, not the operands.

-- Recursive arithmetic fold
a = (1 + 2) * (3 + 4)        -- 21
b = 5 + 6 * 3 - 6 / 3        -- 21.0 (float because of /)
c = 2 ^ 10                   -- 1024.0
d = 17 % 5                   -- 2
e = 20 // 6                  -- 3

-- Mixed int/float
f = 2 + 3.5                  -- 5.5
g = 10.0 / 4                 -- 2.5

-- Bitwise (operands integer-convertible)
h = 0xF0 | 0x0F              -- 255
i = 0xFF & 0x0F              -- 15
j = 0xFF ~ 0x0F              -- 240 (xor)
k = 1 << 8                   -- 256
l = 256 >> 4                 -- 16

-- Recursive unary
m = -(2 + 3)                 -- -5
n = -(-5)                    -- 5
o = ~~0xFF                   -- 255

-- not on consts (Bool/Nil/Numeral)
p = not true                 -- false
q = not nil                  -- true
r = not 1                    -- false
s = not false                -- true
