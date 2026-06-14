-- A live label (forward goto) set to exactly the elision boundary must
-- survive: ::skip:: slides forward onto the next emitted instruction.
local a
goto skip
::skip::
if a then end
