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

stack layout discrepancies
               
  1. count == 0 in RETURN should mean variable returns, not 0 (significant)
                                              
  From lopcodes.h: B=0 → variable (up to top), B=1 → 0 results, B=n+1 → n results. TCVM's if count == 0 { 0 } else { count - 1 }
  treats 0 as zero results. Same problem in CALL: args == 0 means args = top - A, and returns == 0 means MULTRET. All three require a
   top pointer — this is TODO #4.                                                                                                    

  3. TAILCALL: function slot at base - 1 not updated (cosmetic)                                                                    
                                                                                                                                     
  Reference moves the function object to base - 1 as part of luaD_pretailcall. TCVM only moves the args. No functional impact since
  TCVM reads the closure from frame.closure, not the stack slot.                                                                     
   
  ---                                                                                                                                
  Bottom line: the layout is structurally correct. The only immediately fixable gap is nil-filling missing arguments in CALL. The  
  variable-return/variable-arg semantics all depend on top pointer tracking, already listed as TODO #4.
