-- Operator-precedence verification with non-foldable operands. With the
-- expdesc-based fold this is the test that exercises emission order; the
-- old `op_prec.lua` test now folds end-to-end and exercises fold
-- correctness instead.
num = a + b * c - b / c
num = -a
