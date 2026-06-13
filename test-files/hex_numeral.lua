-- Hex numeral forms from issue #99. Each previously failed to lex.
a = 0x1p4        -- unsigned binary exponent
b = 0x1.8p0      -- frac + unsigned exponent
c = 0X10         -- uppercase 0X prefix (integer)
d = 0X1.8        -- uppercase 0X prefix (float)
e = 0x.8         -- leading radix point
f = 0x.8p+1      -- leading radix point + signed exponent
g = 0x1.         -- trailing radix point
h = 0X.8P-2      -- uppercase prefix, leading dot, uppercase exponent
-- Regression guards: forms that already lexed must keep working.
i = 0x1p+4       -- signed exponent
j = 0x1.9p-3     -- frac + signed exponent
k = 0xFF         -- plain hex integer
