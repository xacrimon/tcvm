use crate::dmm::{Collect, Gc, Mutation, RefLock, allocator_api::MetricsAlloc};
use crate::env::value::Value;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Table<'gc>(Gc<'gc, RefLock<TableState<'gc>>>);

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct TableState<'gc> {
    raw: RawTable<'gc>,
    metatable: Option<Table<'gc>>,
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct RawTable<'gc> {
    array: Vec<Value<'gc>>,
    hash: hashbrown::HashMap<Value<'gc>, Value<'gc>, foldhash::fast::RandomState, MetricsAlloc<'gc>>,
}

impl<'gc> Table<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let raw = RawTable {
            array: Vec::new(),
            hash: hashbrown::HashMap::with_hasher_in(
                foldhash::fast::RandomState::default(),
                MetricsAlloc::new(mc),
            ),
        };
        let state = TableState {
            raw,
            metatable: None,
        };
        Table(Gc::new(mc, RefLock::new(state)))
    }

    /// Raw get — no metamethod invocation.
    pub fn raw_get(self, key: Value<'gc>) -> Value<'gc> {
        let state = self.0.borrow();
        if let Some(i) = array_index(&key) {
            if let Some(v) = state.raw.array.get(i - 1) {
                return *v;
            }
        }
        state.raw.hash.get(&key).copied().unwrap_or(Value::Nil)
    }

    /// Raw set — no metamethod invocation.
    pub fn raw_set(self, mc: &Mutation<'gc>, key: Value<'gc>, value: Value<'gc>) {
        let mut state = self.0.borrow_mut(mc);
        if let Some(i) = array_index(&key) {
            let idx = i - 1;
            if idx < state.raw.array.len() {
                state.raw.array[idx] = value;
                return;
            }
            if idx == state.raw.array.len() && !value.is_nil() {
                state.raw.array.push(value);
                // Migrate consecutive integer keys from hash part
                loop {
                    let next = Value::Integer((state.raw.array.len() + 1) as i64);
                    if let Some(v) = state.raw.hash.remove(&next) {
                        state.raw.array.push(v);
                    } else {
                        break;
                    }
                }
                return;
            }
        }
        if value.is_nil() {
            state.raw.hash.remove(&key);
        } else {
            state.raw.hash.insert(key, value);
        }
    }

    /// Length of the array part.
    pub fn raw_len(self) -> usize {
        self.0.borrow().raw.array.len()
    }

    pub fn metatable(self) -> Option<Table<'gc>> {
        self.0.borrow().metatable
    }

    pub fn set_metatable(self, mc: &Mutation<'gc>, mt: Option<Table<'gc>>) {
        self.0.borrow_mut(mc).metatable = mt;
    }

    pub fn inner(self) -> Gc<'gc, RefLock<TableState<'gc>>> {
        self.0
    }
}

/// Extract a valid array index from a Value (1-based positive integer).
fn array_index(key: &Value) -> Option<usize> {
    match key {
        Value::Integer(i) if *i >= 1 => Some(*i as usize),
        Value::Float(f) => {
            let i = *f as i64;
            if i >= 1 && (i as f64) == *f {
                Some(i as usize)
            } else {
                None
            }
        }
        _ => None,
    }
}
