-- Decimal numeral forms with a leading or trailing radix point. The leading
-- and trailing `.` variants previously failed to lex as a single Float.
a = .5          -- leading radix point
b = 1.          -- trailing radix point
c = .5e2        -- leading radix point + exponent
d = 1.e5        -- trailing radix point + exponent
e = .0          -- leading radix point, zero frac
f = 5.          -- trailing radix point
-- Regression guards: forms that already lexed must keep working.
g = 1.5         -- digits on both sides
h = 1e5         -- exponent, no radix point
i = 0.5e5       -- digits + exponent
j = 3.14        -- plain float
