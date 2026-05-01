-- `if true then ...` should not emit a truthiness check; `if false then ...`
-- still compiles the body but with an unconditional skip via TEST/JMP.

if true then
    a = 1
end

if false then
    b = 2
end

-- `true and X` / `false or X` keep RHS but drop the LHS truthiness check.
c = true and 5
d = false or 7
