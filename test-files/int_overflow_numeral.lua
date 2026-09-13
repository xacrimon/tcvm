-- Integer literals at and beyond the i64 range (issue #109). Each previously
-- failed to load with 'internal compiler error: literal without value'.
-- Decimal overflow: reparsed as a float (Lua's str2int -> str2num fall-through).
a = 9223372036854775808           -- 2^63: float
b = 99999999999999999999999       -- float (1e+23)
c = 18446744073709551616          -- 2^64: float
-- The i64 boundary stays an integer.
d = 9223372036854775807           -- i64::MAX
-- Hex integer literals wrap mod 2^64 (read unsigned, reinterpreted as i64).
e = 0x7fffffffffffffff            -- i64::MAX
f = 0x8000000000000000            -- wraps to i64::MIN
g = 0xffffffffffffffff            -- wraps to -1
h = 0x10000000000000000           -- 2^64 -> 0
i = 0xffffffffffffffffff          -- >16 digits, low 64 bits -> -1
-- Regression guards: small/normal literals and hex floats are unaffected.
j = 0
k = 42
l = 0xff
m = 0x1p4
