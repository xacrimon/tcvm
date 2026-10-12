//! Offsets of the runtime fields compiled code reads, derived with
//! `offset_of!` (`Runtime` and `JitRuntime` are not `repr(C)`, so they are per
//! build), with tests against live objects.

use std::mem::offset_of;

use crate::dmm::Gc;
use crate::dmm::lock::RefLock;
use crate::env::table::TableState;
use crate::jit::state::JitRuntime;
use crate::lua::State;
use crate::vm::dispatch::Runtime;

/// `State.jit` from the `rt` register.
const JIT: usize = offset_of!(State<'static>, jit);
/// The pointer to the exit register image.
pub(crate) const EXIT_REGS: usize = JIT + offset_of!(JitRuntime, exit_regs);
/// `ThreadState::native`, from the `thread` register: the dispatch entry's
/// stack pointer, frame pointer and return address, the fast entries' stack
/// limit and the entry shim's target.
const NATIVE: usize = offset_of!(crate::env::thread::ThreadState<'static>, native);
pub(crate) const NATIVE_SP: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, sp);
pub(crate) const NATIVE_FP: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, fp);
pub(crate) const NATIVE_LR: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, lr);
pub(crate) const NATIVE_LIMIT: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, limit);
pub(crate) const NATIVE_DEEP: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, deep);
pub(crate) const NATIVE_TARGET: usize = NATIVE + offset_of!(crate::jit::state::NativeBase, target);
/// The pointer to the arena's `Metrics`.
pub(crate) const METRICS: usize =
    offset_of!(State<'static>, rt) + offset_of!(Runtime<'static>, metrics);
/// The running thread's state, from `rt`.
#[cfg(target_arch = "x86_64")]
pub(crate) const RT_THREAD: usize =
    offset_of!(State<'static>, rt) + offset_of!(Runtime<'static>, thread);
/// The allocation counter and threshold pair inside `Metrics`.
pub(crate) const GC_CHECK: usize = crate::dmm::metrics::Metrics::GC_CHECK_OFFSET;

/// The cell of a type's metatable (`State::type_metatable`, by its slot
/// there) from `rt`, and the metatable's word (null when none) within it.
pub(crate) const fn type_mt_cell(slot: u8) -> usize {
    offset_of!(State<'static>, type_metatables) + 8 * slot as usize
}
pub(crate) const NUMBER_MT_CELL: usize = type_mt_cell(2);
pub(crate) const STRING_MT_SLOT: u8 = 3;
pub(crate) const NUMBER_MT: usize =
    Gc::<'static, crate::dmm::lock::Lock<Option<crate::env::table::Table<'static>>>>::VALUE_OFFSET;

/// The end of the running thread's stack, from the `thread` register.
pub(crate) const THREAD_STACK_END: usize =
    offset_of!(crate::env::thread::ThreadState<'static>, stack_end);

/// A table's `TableState` from its lock (a table value's payload).
const T_STATE: usize = RefLock::<TableState<'static>>::TRAILING_FROM_LOCK
    - RefLock::<TableState<'static>>::TRAILING_FROM_INNER;
pub(crate) const T_SHAPE: usize = T_STATE + TableState::SHAPE_OFFSET;
pub(crate) const T_SPILL: usize = T_STATE + TableState::SPILL_OFFSET;
pub(crate) const T_SPILL_CAP: usize = T_STATE + TableState::SPILL_CAP_OFFSET;
pub(crate) const T_ARRAY: usize = T_STATE + TableState::ARRAY_OFFSET;
pub(crate) const T_ASIZE: usize = T_STATE + TableState::ASIZE_OFFSET;
/// A table's inline named slots, the values trailing its state.
pub(crate) const T_INLINE: usize = RefLock::<TableState<'static>>::TRAILING_FROM_LOCK;
/// A table's GC header (bit 0: gray, no barrier needed) before its lock.
pub(crate) const T_HEADER: i32 =
    -(Gc::<'static, RefLock<TableState<'static>>>::VALUE_OFFSET as i32);

/// A Lua closure's fields from its function's payload. `FN_IS_VARARG` also
/// tells a native function, whose variant ends before it, by the niche the
/// enum's tag takes there (`fn_kind_niche` checks it).
const FN_LUA: usize = offset_of!(crate::env::function::FunctionKind<'static>, Lua.0);
pub(crate) const FN_CODE: usize =
    FN_LUA + offset_of!(crate::env::function::LuaClosure<'static>, code);
pub(crate) const FN_NUM_PARAMS: usize =
    FN_LUA + offset_of!(crate::env::function::LuaClosure<'static>, num_params);
pub(crate) const FN_MAX_STACK: usize =
    FN_LUA + offset_of!(crate::env::function::LuaClosure<'static>, max_stack_size);
pub(crate) const FN_IS_VARARG: usize =
    FN_LUA + offset_of!(crate::env::function::LuaClosure<'static>, is_vararg);
/// A function value's top 16 bits (`QNAN_NEG | TAG_FUNCTION << 48`).
pub(crate) const FUNCTION_TAG: u16 = 0xFFF8 | 5;
/// A Lua closure's prototype pointer (a `Gc`, so the box), and from that
/// box the prototype's fast entry.
pub(crate) const FN_PROTO: usize =
    FN_LUA + offset_of!(crate::env::function::LuaClosure<'static>, proto);
pub(crate) const PROTO_FAST: usize =
    Gc::<'static, crate::env::function::Prototype<'static>>::VALUE_OFFSET
        + offset_of!(crate::env::function::Prototype<'static>, jit_fast);
const _: () = assert!(
    size_of::<crate::env::function::NativeClosure<'static>>()
        <= offset_of!(crate::env::function::FunctionKind<'static>, Native.0) + FN_IS_VARARG
);

// A header word is a `u64` slot of the value stack.
const _: () = assert!(size_of::<crate::env::value::Value<'static>>() == 8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lua;

    #[test]
    fn offsets_match_reality() {
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let base = ctx.jit() as *const JitRuntime as usize;
            let state = (ctx.jit() as *const JitRuntime as usize) - JIT;
            assert_eq!(base - state, JIT);
            let regs = unsafe { ((state + EXIT_REGS) as *const *mut u64).read() };
            assert_eq!(regs, ctx.jit().exit_regs);
            let metrics = unsafe { ((state + METRICS) as *const usize).read() };
            assert_ne!(metrics, 0);
            let number_mt = || unsafe {
                let cell = ((state + NUMBER_MT_CELL) as *const usize).read();
                ((cell + NUMBER_MT) as *const usize).read()
            };
            assert_eq!(number_mt(), 0);
            let mt = crate::env::table::Table::new(ctx);
            ctx.set_metatable_of(crate::env::value::Value::small(1), Some(mt));
            assert_ne!(number_mt(), 0);
        });
    }

    #[test]
    fn table_fields() {
        use crate::env::value::Value;
        use crate::env::{MetamethodBits, MtCache};
        let mut lua = Lua::new();
        lua.load_all();
        let ex = lua
            .try_enter(|ctx| {
                let src = "local t = setmetatable({10, 20, 30, x = 1}, {__index = {}, __add = print}) return t";
                Ok::<_, crate::LoadError>(ctx.stash(crate::Executor::start(
                    ctx,
                    ctx.load(src, None)?,
                    (),
                )))
            })
            .unwrap();
        lua.finish(&ex).unwrap();
        lua.enter(|ctx| {
            let v: Value = ctx.fetch(&ex).take_result(ctx).unwrap();
            let t = v.get_table().unwrap();
            let lock = (v.to_raw() & ((1 << 48) - 1)) as usize;
            let word = |off: usize| unsafe { ((lock + off) as *const u64).read() };
            let shape = t.shape();
            assert_eq!(word(T_SHAPE), shape.as_word());
            let asize = unsafe { ((lock + T_ASIZE) as *const u32).read() } as usize;
            assert!(asize >= 4);
            let arr = word(T_ARRAY) as *const Value;
            assert_eq!(unsafe { arr.add(2).read() }.get_small(), Some(20));
            let class: MtCache = shape.mt_cache().unwrap();
            let cw = class.as_word() as usize;
            let bits = unsafe { ((cw + MtCache::BITS_OFFSET) as *const u32).read() };
            assert!(bits & MetamethodBits::INDEX.bits() != 0);
            let index = unsafe { ((cw + MtCache::INDEX_OFFSET) as *const Value).read() };
            assert!(index.get_table().is_some());
            // The other metamethods: shape -> class -> rest -> slot.
            let sw = word(T_SHAPE) as usize;
            let class_word =
                unsafe { ((sw + crate::env::Shape::MT_CACHE_OFFSET) as *const usize).read() };
            assert_eq!(class_word, cw);
            let rest = unsafe { ((cw + MtCache::REST_OFFSET) as *const usize).read() };
            assert_ne!(rest, 0);
            let i = MetamethodBits::ADD.bits().trailing_zeros() as usize;
            let add =
                unsafe { ((rest + MtCache::REST_VALUES_OFFSET + 8 * i) as *const Value).read() };
            assert_eq!(add.to_raw(), class.mm(MetamethodBits::ADD).to_raw());
            assert!(add.get_function().is_some());
            let header = unsafe { ((lock as isize + T_HEADER as isize) as *const u64).read() };
            assert_eq!(header & 1 != 0, Gc::is_gray(t.inner()));
        });
    }

    #[test]
    fn upvalue_cell_value_pointer() {
        use crate::dmm::Gc;
        use crate::env::function::UpvalueCell;
        use crate::env::value::Value;
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let cell = UpvalueCell::new_closed(ctx.mutation(), Value::small(7));
            // What an upvalue slot holds: the cell's box.
            let boxed = Gc::erased_ptr(cell) as usize;
            let v = unsafe { ((boxed + UpvalueCell::V_OFFSET) as *const *const Value).read() };
            assert_eq!(unsafe { v.read() }.get_small(), Some(7));
        });
    }

    #[test]
    fn fn_kind_niche() {
        use crate::env::value::Value;
        let mut lua = Lua::new();
        lua.load_all();
        let ex = lua
            .try_enter(|ctx| {
                let src = "return {print, function(a, b) return a end, function(...) end}";
                Ok::<_, crate::LoadError>(ctx.stash(crate::Executor::start(
                    ctx,
                    ctx.load(src, None)?,
                    (),
                )))
            })
            .unwrap();
        lua.finish(&ex).unwrap();
        lua.enter(|ctx| {
            let v: Value = ctx.fetch(&ex).take_result(ctx).unwrap();
            let t = v.get_table().unwrap();
            // What compiled code reads: a function value's payload.
            let fun = |i: i32| {
                let f = t.raw_get(Value::small(i));
                assert_eq!((f.to_raw() >> 48) as u16, FUNCTION_TAG);
                let p = (f.to_raw() & ((1 << 48) - 1)) as usize;
                assert_eq!(p, Gc::as_ptr(f.get_function().unwrap().inner()) as usize);
                p
            };
            let byte = |p: usize, off: usize| unsafe { ((p + off) as *const u8).read() };
            assert!(byte(fun(1), FN_IS_VARARG) > 1);
            let fixed = fun(2);
            assert_eq!(byte(fixed, FN_IS_VARARG), 0);
            assert_eq!(byte(fixed, FN_NUM_PARAMS), 2);
            assert_eq!(byte(fun(3), FN_IS_VARARG), 1);
            let lf = t
                .raw_get(Value::small(2))
                .get_function()
                .unwrap()
                .as_lua()
                .unwrap();
            let code = unsafe { ((fixed + FN_CODE) as *const usize).read() };
            assert_eq!(code, lf.code as usize);
            let proto = unsafe { ((fixed + FN_PROTO) as *const usize).read() };
            lf.proto.jit_fast.set(0x1234);
            assert_eq!(unsafe { ((proto + PROTO_FAST) as *const usize).read() }, 0x1234);
            lf.proto.jit_fast.set(0);
            assert_eq!(byte(fixed, FN_MAX_STACK), lf.max_stack_size);
        });
    }

    #[test]
    fn string_length() {
        use crate::env::string::LuaString;
        use crate::env::value::Value;
        let mut lua = Lua::new();
        lua.enter(|ctx| {
            let s = LuaString::new(ctx, b"hello, world");
            let p = (Value::string(s).to_raw() & ((1 << 48) - 1)) as usize;
            let len = unsafe { ((p + LuaString::LEN_OFFSET) as *const usize).read() };
            assert_eq!(len, 12);
        });
    }

    #[test]
    fn named_slots_and_keys() {
        use crate::env::Shape;
        use crate::env::string::LuaString;
        use crate::env::value::{STRING_BITS, Value};
        let mut lua = Lua::new();
        lua.load_all();
        let ex = lua
            .try_enter(|ctx| {
                // More fields than any inline capacity, so some spill.
                let mut src = String::from("local t = {}");
                for i in 0..40 {
                    src += &format!(" t.k{i} = {i}");
                }
                src += " return t";
                Ok::<_, crate::LoadError>(ctx.stash(crate::Executor::start(
                    ctx,
                    ctx.load(src.as_str(), None)?,
                    (),
                )))
            })
            .unwrap();
        lua.finish(&ex).unwrap();
        lua.enter(|ctx| {
            let v: Value = ctx.fetch(&ex).take_result(ctx).unwrap();
            let t = v.get_table().unwrap();
            let lock = (v.to_raw() & ((1 << 48) - 1)) as usize;
            let rd = |a: usize| unsafe { (a as *const u64).read() } as usize;
            let rd32 = |a: usize| unsafe { (a as *const u32).read() } as usize;
            let shape = t.shape();
            let sw = rd(lock + T_SHAPE);
            assert_eq!(sw as u64, shape.as_word());
            let n = rd32(sw + Shape::SLOT_COUNT_OFFSET);
            assert_eq!(n, 40);
            let cap = unsafe { ((sw + Shape::INLINE_CAP_OFFSET) as *const u8).read() } as usize;
            assert!(cap < n);
            assert_eq!(rd32(sw + Shape::MT_SLOT_OFFSET), u32::MAX as usize);
            let keys = rd(rd(sw + Shape::KEYS_OFFSET) + Shape::KEYS_SLOTS_OFFSET);
            for i in 0..n {
                let kbox = rd(keys + 8 * i);
                let key = ((kbox + LuaString::VALUE_OFFSET) as u64) | STRING_BITS;
                let kv = unsafe { Value::from_raw(key) };
                let k = kv.get_string().unwrap();
                assert_eq!(k.as_bytes(), format!("k{i}").as_bytes());
                let at = if i < cap {
                    lock + T_INLINE + 8 * i
                } else {
                    rd(lock + T_SPILL) + 8 * (i - cap)
                };
                let v = unsafe { (at as *const Value).read() };
                assert_eq!(v.get_small(), Some(i as i32));
            }
        });
    }
}
