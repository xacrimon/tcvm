-- Empty then-blocks: the condition's TEST/JMP elides to a no-op tail.
-- Nested and sibling empty ifs left dead end_labels that tripped the
-- over-broad jump-elision assert in patch_to_here (rules.rs:614).
local a, b, c
if a then if b then if c then end end end
if a then end
if b then end
