use crate::Context;
use crate::dmm::{Collect, Gc, Mutation, RefLock, allocator_api::MetricsAlloc};
use crate::env::string::LuaString;
use crate::env::value::Value;
use bitflags::bitflags;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Table<'gc>(Gc<'gc, RefLock<TableState<'gc>>>);

impl<'gc> Table<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        Table(Gc::new(mc, RefLock::new(TableState::new(mc))))
    }

    pub fn raw_get(self, key: Value<'gc>) -> Value<'gc> {
        self.0.borrow().raw_get(key)
    }

    pub fn raw_set(self, mc: &Mutation<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.0.borrow_mut(mc).raw_set(key, value);
    }

    pub fn raw_len(self) -> usize {
        self.0.borrow().raw_len()
    }

    pub fn metatable(self) -> Option<Table<'gc>> {
        self.0.borrow().metatable
    }

    pub fn set_metatable(self, mc: &Mutation<'gc>, mt: Option<Table<'gc>>) {
        self.0.borrow_mut(mc).metatable = mt;
    }

    pub fn get_metamethod(self, ctx: Context<'gc>, name: &[u8]) -> Value<'gc> {
        let Some(mt) = self.metatable() else {
            return Value::nil();
        };
        let key = Value::string(LuaString::new(ctx, name));
        mt.raw_get(key)
    }

    pub fn inner(self) -> Gc<'gc, RefLock<TableState<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<TableState<'gc>>>) -> Self {
        Table(g)
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct TableState<'gc> {
    array: Vec<Value<'gc>, MetricsAlloc<'gc>>,
    hash:
        hashbrown::HashMap<Value<'gc>, Value<'gc>, foldhash::fast::RandomState, MetricsAlloc<'gc>>,
    metatable: Option<Table<'gc>>,
    #[collect(require_static)]
    metamethods: Metamethod,
}

impl<'gc> TableState<'gc> {
    fn new(mc: &Mutation<'gc>) -> Self {
        Self {
            array: Vec::new_in(MetricsAlloc::new(mc)),
            hash: hashbrown::HashMap::with_hasher_in(
                foldhash::fast::RandomState::default(),
                MetricsAlloc::new(mc),
            ),
            metatable: None,
            metamethods: Metamethod::empty(),
        }
    }

    #[inline(always)]
    pub fn has_metamethod(&self, method: Metamethod) -> bool {
        self.metamethods.contains(method)
    }

    #[inline(always)]
    pub fn raw_get(&self, key: Value<'gc>) -> Value<'gc> {
        if let Some(index) = array_index(key) {
            return match self.array.get(index - 1) {
                Some(value) => *value,
                None => Value::nil(),
            };
        }

        match self.hash.get(&key) {
            Some(value) => *value,
            None => Value::nil(),
        }
    }

    #[inline(always)]
    pub fn raw_set(&mut self, key: Value<'gc>, value: Value<'gc>) {
        if let Some(index) = array_index(key) {
            if index > self.array.len() {
                self.array.resize(index, Value::nil());
            }

            self.array[index - 1] = value;
        }

        if !key.is_nil() {
            self.hash.insert(key, value);
        } else {
            self.hash.remove(&key);
        }
    }

    #[inline(always)]
    pub fn raw_len(&self) -> usize {
        self.array.len()
    }
}

/// Extract a valid array index from a Value (1-based positive integer).
fn array_index(key: Value) -> Option<usize> {
    if let Some(i) = key.get_integer() {
        if i >= 1 {
            return Some(i as usize);
        }
        return None;
    }

    if let Some(f) = key.get_float() {
        let i = f as i64;
        if i >= 1 && (i as f64) == f {
            return Some(i as usize);
        }
        return None;
    }

    None
}

bitflags! {
    pub struct Metamethod: u32 {
        const INDEX =    0b00000000000000000000000000000001;
        const NEWINDEX = 0b00000000000000000000000000000010;
    }
}
