todo:
- varargs
- register index width issues
- nargs overflow in resolve_call_chain

Explicitly out of scope (documented TODOs)
 - Native target reached via a __call metamethod chain.
 - Native target as a metamethod (__index, __add, …) or as a TFORCALL
 iterator. These flow through schedule_meta_call / the continuation
 machinery, which assumes a Lua frame gets pushed. A native target there
 needs either a Frame::Callback variant (piccolo's approach) or a
 per-continuation native dispatch — either is a larger refactor.
 - Sequence-style continuations that let a native callback call back
 into Lua.
 - Coroutine yield from a native callback (no coroutine runtime yet).
 - Error-message propagation. NativeError carries a String for the
 sake of the callback author, but the interpreter drops it at the VM
 boundary: native Err triggers the existing raise!() path, so
 RuntimeError::Opcode { pc } is all the host sees, same as any other
 runtime error today. Plumbing the message is a separate change.
 - GC-valued errors.
