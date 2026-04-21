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

1. run_thread holds a long-lived RefMut on the thread — same-thread open upvalue access will panic.                
                                                                                
  run_thread (src/vm/interp.rs:215-234) takes ts = thread.borrow_mut(mc) and keeps it alive across the entire        
  tail-call dispatch chain. Handlers like op_getupval (line 324) and op_setupval (line 346) look up upvalues and, for
   the UpvalueState::Open branch, call t.borrow() on the upvalue's owning thread. op_closure (line 1398) populates   
  these upvalues with thread_handle — the same thread that's currently running. Any Lua script that reads a captured 
  local while the enclosing frame is still live (textbook closures like function outer() local n = 10; local inner = 
  function() return n end; return inner() end) will hit RefCell::borrow while a borrow_mut is active → panic.
                                                                                                                     
  This is pre-existing — the old run() had the same shape — but the MVP makes it reachable for the first time because
   we can now actually drive the VM. Fixing it probably requires either (a) releasing the RefMut and borrowing per   
  handler, (b) bypassing the RefCell via raw pointers in the hot path (what piccolo does — it uses Gc::as_ptr through
   the stack handle), or (c) restructuring so open upvalues don't go through RefCell at all.                         
                                                                                
  2. Executor::start panics on native functions.   
                 
  src/lua/executor.rs:55-57 uses expect("Executor::start: native functions not yet supported"). A host process
  shouldn't abort on this — should be Result<Self, RuntimeError> with a NotALuaFunction variant (or similar). Same   
  smell: the expect("run_thread requires a seeded frame") in run_thread.
                                                                                                                     
  3. op_return top-level truncation assumes cur_base >= 1.                      
                                                     
  src/vm/interp.rs:1140 — thread.stack.truncate(dst_start + nret) where dst_start = cur_base - 1. If a caller ever
  set up a frame with base = 0, this wraps to usize::MAX and the truncate panics. Executor::start always uses base = 
  1 so this is latent, but the invariant should be either asserted or documented on run_thread.                      
                                                   
  4. Main-thread aliasing makes multiple live executors unsound.                                                     
                                                                                
  Executor::start always uses ctx.main_thread() and clobbers its stack/frames. If a stashed executor is mid-run (or  
  in Result mode with live return values), calling start for a different executor silently trashes the first one's
  state. StashedExecutor handles give a false sense of isolation. Either Executor must own its own Thread (spawn a   
  fresh one per start), or start must fail when the main thread isn't clean.    
                                                                                                                     
  5. Executor::step flips mode to Result without checking thread terminal state.
                                                                                                                     
  src/lua/executor.rs:108-113 — if run_thread returns Ok(()) but the thread isn't actually Dead (hypothetical VM bug
  or future preemption semantics), we claim Result anyway. Should assert/check thread.borrow().status ==             
  ThreadStatus::Dead before flipping.                                           
                                                                                                                     
  Bugs that aren't safety but are wrong                                         
                                                     
  6. Context::load silently drops the name argument.                                                                 
                                                     
  src/lua/context.rs:49 — the _name: Option<&str> parameter is never used. It's supposed to feed the prototype's     
  source field so error messages, debug.getinfo, tracebacks, etc. can report where compiled code came from. Right now
   loading two chunks gives both the same anonymous source. Either thread it through to compile_chunk (which needs   
  its signature extended), or rename to () and delete the parameter until we're ready.
                                                                                                                     
  7. Context::load treats any ariadne report as a fatal parse error.
                                                                                                                     
  src/lua/context.rs:52 — if !reports.is_empty() bails on any report, but ariadne::Report can carry warnings/notes
  too. Need to filter by ReportKind::Error (or trust our parser to only emit errors and document that contract).     
                                                                                
  8. Executor::take_result clones the entire stack.                                                                  
                                                                                
  src/lua/executor.rs:127-130 — let values: Vec<Value<'gc>> = ts.stack.clone(). We immediately feed it to
  from_multi_value(&values). Could just pass &ts.stack directly and skip the allocation. For MVP fine, but it's      
  allocating every call.                                       
                                                                                                                     
  9. Lua::execute does two enter calls.                                         
                                                                                                                     
  src/lua/mod.rs:82-91 — one for finish (step), then another to re-fetch the executor and call take_result. Between  
  these, a collection cycle can run. It's probably sound (stashed executor keeps the thread alive), but it's two
  arena mutations where one would do. Consolidate.                                                                   
                                                                                
  Design / API issues                                                                                                
                                                                                
  10. Stashed handles can't be cloned or compared.                                                                   
                                                     
  StashedFunction/StashedTable/StashedThread/StashedExecutor are newtypes over DynamicRoot<R>. DynamicRoot implements
   Clone but we don't propagate it, and we don't implement PartialEq. If a user wants to hold a stashed function in a
   HashMap or pass it around multiple times, they can't.                                                             
                                                                                
  11. No StashedValue.                                         
                                                     
  The plan acknowledged this — Value<'gc> is an enum not a Gc, so you can't naively put it in a DynamicRoot. But this
   is a real gap for realistic use: a user who stores a number or bool in a Lua variable and wants to bring it back
  to Rust has no way to stash it without enter-ing. Needs either a wrapper enum that stashes the GC variants and     
  carries the primitives inline, or a policy that stashed values are fetched eagerly into owned Rust types.
                                                                                                                     
  12. No Lua::execute<R> where R can carry 'gc.                                 
                                                                                                                     
  The R: for<'gc> FromMultiValue<'gc> HRTB forces R to be 'static. Users wanting Table<'gc> back have to drop to
  enter + step + take_result by hand. Worth documenting; may want a second method that returns a stashed result.     
                                                                                
  13. StashedExecutor is public but ExecutorInner is pub(crate).                                                     
                                                                                
  Fine mechanically — the stash handle is opaque. But the Rootable![RefLock<ExecutorInner<'_>>] in its type shows
  through via clippy's type-complexity warnings. Worth a type StashedExecutorRoot = Rootable![...] or similar for    
  readability.                     
                                                                                                                     
  14. Executor::new never got implemented.                                      
                                                                                                                     
  The plan had pub fn new(ctx) -> Self creating an empty/stopped executor. Only start exists. Minor plan drift but
  callers can't create an executor without immediately seeding a call.
                                                                                                                     
  15. No ergonomic helpers for globals.            
                                                                                                                     
  ctx.globals().raw_get(Value::String(LuaString::new(mc, b"foo"))) is four layers of ceremony for a global lookup.
  Context::set_global<V: IntoValue>(&str, V) and get_global<V: FromValue>(&str) -> Result<V, _> would remove this    
  boilerplate. Piccolo has these.  
                                                                                                                     
  16. No IntoValue for &str / String.                                           
                                                                                                                     
  String construction needs &Mutation<'gc> (to allocate the LuaString), so it can't be a plain IntoValue<'gc>. This
  is the piccolo-style friction: needs a separate conversion trait like IntoValueCtx<'gc> that takes a &Mutation or
  Context. Currently users must manually create the LuaString.                                                       
                                                   
  17. LoadError::Parse(Vec<Report>) is user-hostile.                                                                 
                                                                                
  Report renders through ariadne's writer; our error's Display just says "parse error". Users have to reach into the 
  Vec and render each report themselves. At minimum, Display should summarize the count and first report; ideally it
  renders the whole thing to the caller's writer.                                                                    
                                                                                
  Code quality                                                                                                       
                                                                                
  18. Clippy — &mut *ts → &mut ts.                                                                                   
                                                     
  src/vm/interp.rs:234 — RefMut auto-derefs; the explicit * is noise.                                                
                                                                                
  19. Clippy — stashed newtype complexity.                                                                           
                                                                                
  Three Rootable![RefLock<...>] inlined type expressions flagged. Extract type aliases: type FunctionRoot = 
  Rootable![FunctionKind<'_>]; etc.                  
                                                                                                                     
  20. Executor::step's borrow_mut(mc) on the Executor's inner RefLock is fine but it's the third borrow per step.
                                                                                                                     
  self.0.borrow() to read mode+thread, then run_thread(mc, thread), then self.0.borrow_mut(mc) to write mode.
  Acceptable, but could be one borrow_mut(mc) wrapping the VM call (except then run_thread can't reborrow the        
  Executor — it doesn't need to, so this is actually fine).                     
                                                                                                                     
  21. from_multi_single! macro enumeration instead of blanket. 
                                                                                                                     
  The macro explicitly lists bool, i64, etc. Can't be a blanket impl<T: FromValue> FromMultiValue for T because      
  Vec<Value>, (), tuples, and Option<T> have their own impls and would conflict. Leaving it as enumeration is fine
  but every new FromValue type means remembering to add a line here. Consider documenting or consolidating.          
                                                                                
  Missing to make this a usable library                                                                              
                                                                                
  22. Native function calling. The fn() placeholder blocks every stdlib item (print, pairs, tostring, math.*, etc.). 
  This is the single biggest gap and the memory already flagged it as unresolved design.
                                                                                                                     
  23. Lua-level error propagation. Runtime errors in opcodes raise!() to impl_error which returns Err(Box<Error { pc:
   0 }>) — pc is hardcoded to 0 (src/vm/interp.rs:240, existing TODO). We surface a useless RuntimeError::Opcode {   
  pc: 0 }. Need: real error values (string / table / Lua-raised object), a traceback, integration with pcall/xpcall.
                                 
  24. Source names and debug info. Prototype.source exists but the compiler never populates it for the top-level
  chunk (looking at compile, source: None is passed into compile_function_to_chunk), and load() drops the name arg.  
  Without source info, any traceback we eventually build is blind.
                                                                                                                     
  25. String interning. LuaString::new allocates fresh per call. Looking up a global by name                         
  (ctx.globals().raw_get(Value::String(LuaString::new(mc, b"add")))) allocates and hashes a brand-new string every
  time even though the global keys are identical. State should hold an interner keyed on bytes → LuaString<'gc>.     
  Piccolo does this.                                                            
                                                                                                                     
  26. Tests are minimal. We cover the happy path (add(2,3) = 5). Missing:
  - Multi-return (tuple FromMultiValue)                                                                              
  - Closures with captures (would surface the borrow bug above)                 
  - Type mismatch on FromValue / FromMultiValue::Arity                                                               
  - Lua::finish → execute::<i64> on a chunk that runs a native-loaded function (blocked by #22)
  - Re-entrancy: calling lua.enter nested (piccolo forbids this)                                                     
  - Reusing the main thread across multiple calls                               
  - Parse error path through LoadError::Parse        
  - Long-running scripts and stack growth                                                                            
                                                               
  27. No way to pre-populate globals from Rust at construction. Lua::new() creates an empty globals table. For       
  scripting an app we'd want Lua::builder() or similar to register host functions before loading user code.
