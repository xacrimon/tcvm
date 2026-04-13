left off backend work apr 13

session: restructure-callframe-prototype-access

Ready to implement (design decisions needed):
  1. Native function calls — CALL/TAILCALL native paths currently raise!(). Need to decide on the native function signature (what
  args does a Rust callback receive? &Mutation, &mut ThreadState, a slice of args?)
  2. TFORCALL — needs to invoke R[base](R[base+1], R[base+2]) and store results. Blocked on native call support if the iterator is
  native, but could work for Lua iterators now.
  3. Upvalue capture in CLOSURE — needs upvalue descriptor data in Prototype (which upvalues to capture from enclosing scope vs. from
   the enclosing closure's upvalue list). Currently creates closures with empty upvalue arrays.
  4. VARARG/VARARGPREP — need to track actual argument count so varargs can be extracted. Need to decide where to store this (in
  CallFrame? adjust stack layout?)

  Lower priority (correctness refinements):
  5. Metamethod dispatch (MMBIN, __index, __newindex, __lt, __le, __unm, __bnot, __len, __concat, __close)
  7. TBC (to-be-closed variable marking)
