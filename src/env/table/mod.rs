use core::hash::BuildHasher;

use crate::Context;
use crate::dmm::{Collect, Gc, Mutation, RefLock, allocator_api::MetricsAlloc};
use crate::env::shape::{self, MAX_PROPERTIES_FAST, Shape};
use crate::env::string::LuaString;
use crate::env::value::{Value, ValueKind, value_hash};
use hashbrown::{HashTable, hash_table};

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Table<'gc>(Gc<'gc, RefLock<TableState<'gc>>>);

impl<'gc> Table<'gc> {
    /// Create a new empty table that starts at the runtime's shared empty
    /// shape. Prefer this constructor whenever a `Context` is in scope.
    pub fn new(ctx: Context<'gc>) -> Self {
        Self::new_with_shape(ctx.mutation(), ctx.empty_shape())
    }

    /// Create a new empty table starting at the given shape. Used by the
    /// `Lua::new` bootstrap before a `Context` exists, and by paths that
    /// already have the shape in hand.
    pub fn new_with_shape(mc: &Mutation<'gc>, shape: Shape<'gc>) -> Self {
        Table(Gc::new(mc, RefLock::new(TableState::new(mc, shape))))
    }

    pub fn raw_get(self, key: Value<'gc>) -> Value<'gc> {
        self.0.borrow().raw_get(key)
    }

    pub fn raw_set(self, mc: &Mutation<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.0.borrow_mut(mc).raw_set(mc, key, value);
    }

    pub fn raw_get_with_hash(self, key: Value<'gc>, hash: u64) -> Value<'gc> {
        self.0.borrow().raw_get_with_hash(key, hash)
    }

    pub fn raw_set_with_hash(
        self,
        mc: &Mutation<'gc>,
        key: Value<'gc>,
        value: Value<'gc>,
        hash: u64,
    ) {
        self.0
            .borrow_mut(mc)
            .raw_set_with_hash(mc, key, value, hash);
    }

    pub fn raw_len(self) -> usize {
        self.0.borrow().raw_len()
    }

    pub fn metatable(self) -> Option<Table<'gc>> {
        self.0.borrow().metatable
    }

    /// Replace the metatable. Re-shapes the table along the
    /// `set_metatable` transition edge so subsequent metamethod queries
    /// observe the new metatable's identity. Subsequent in-place
    /// mutations of the metatable update its shared `MtCache` bitset
    /// in place.
    pub fn set_metatable(self, ctx: Context<'gc>, mt: Option<Table<'gc>>) {
        let new_cache = match mt {
            Some(t) => Some(t.ensure_mt_cache(ctx)),
            None => None,
        };
        let mut state = self.0.borrow_mut(ctx.mutation());
        state.shape = shape::transition_set_metatable(ctx.mutation(), state.shape, new_cache);
        state.metatable = mt;
    }

    /// Look up a metamethod by pre-interned name. The runtime keeps
    /// every metamethod-name `LuaString` interned in `Context::symbols()`,
    /// so callers always have the identity in hand and we never
    /// re-intern at the call site.
    #[inline]
    pub fn get_metamethod(self, name: LuaString<'gc>) -> Value<'gc> {
        let Some(mt) = self.metatable() else {
            return Value::nil();
        };
        mt.raw_get(Value::string(name))
    }

    pub fn shape(self) -> Shape<'gc> {
        self.0.borrow().shape
    }

    pub fn inner(self) -> Gc<'gc, RefLock<TableState<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<TableState<'gc>>>) -> Self {
        Table(g)
    }

    /// Lazily allocate this table's `MtCache` and return it. The
    /// cache's identity is invariant across mutations of *this table*;
    /// metamethod-named writes mutate the cache's bitset in place.
    /// First-adoption walks the table once to compute initial bits.
    pub(crate) fn ensure_mt_cache(self, ctx: Context<'gc>) -> shape::MtCache<'gc> {
        {
            let state = self.0.borrow();
            if let Some(c) = state.mt_cache {
                return c;
            }
        }
        // First adoption: walk the table once to compute initial bits.
        let mut bits = shape::MetamethodBits::empty();
        {
            let state = self.0.borrow();
            for (name, bit) in ctx.symbols().metamethods() {
                if !state.raw_get(Value::string(name)).is_nil() {
                    bits |= bit;
                }
            }
        }
        let cache = shape::MtCache::new(ctx.mutation(), bits);
        self.0.borrow_mut(ctx.mutation()).mt_cache = Some(cache);
        cache
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct TableState<'gc> {
    /// Hidden class describing string-keyed property layout + metatable
    /// identity. Replaces the dead `metamethods` bitset. In dict mode,
    /// this is a per-table sentinel shape; ICs naturally bypass.
    pub(crate) shape: Shape<'gc>,
    /// String-keyed property values, indexed by `shape.find_slot(key)`.
    /// `properties.len() == shape.slot_count()` post-set. Empty in
    /// dict mode (storage moves to `dict`).
    pub(crate) properties: Vec<Value<'gc>, MetricsAlloc<'gc>>,
    /// Array part for positive integer keys 1..n. Self-contained — no
    /// shape involvement.
    array: Vec<Value<'gc>, MetricsAlloc<'gc>>,
    /// Fallback hash for non-string, non-array-integer keys (booleans,
    /// floats, table/function/thread-as-keys, negative integers).
    misc_hash: HashTable<(Value<'gc>, Value<'gc>), MetricsAlloc<'gc>>,
    /// Set when this table has dropped to dictionary mode for its
    /// string-keyed properties. Triggered by deletion of an existing
    /// slot or by exceeding `MAX_PROPERTIES_FAST` slots — both cases
    /// where shape-tree maintenance becomes hostile to ICs.
    dict: Option<DictState<'gc>>,
    /// Live metatable handle (for `getmetatable` and metamethod
    /// invocation). Identity is mirrored in `shape.mt_cache`.
    metatable: Option<Table<'gc>>,
    /// Metamethod-presence cache for *this* table when it's used as a
    /// metatable (lazily allocated on first adoption). The cache's
    /// `bits` field is updated in place by every metamethod-named
    /// write to this table; downstream shapes share this same `Gc`
    /// pointer and observe the updates without a freshness check.
    mt_cache: Option<shape::MtCache<'gc>>,
}

/// Slow / dictionary-mode storage for string-keyed properties. Replaces
/// the `(shape, properties)` pair in dict mode. ICs bypass naturally
/// because the table's shape becomes a unique dict-sentinel that no IC
/// will have cached.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct DictState<'gc> {
    pub(crate) table: HashTable<(LuaString<'gc>, Value<'gc>), MetricsAlloc<'gc>>,
}

#[inline]
fn lua_string_hash(key: LuaString<'_>) -> u64 {
    foldhash::fast::FixedState::default().hash_one(key)
}

impl<'gc> TableState<'gc> {
    fn new(mc: &Mutation<'gc>, shape: Shape<'gc>) -> Self {
        Self {
            shape,
            properties: Vec::new_in(MetricsAlloc::new(mc)),
            array: Vec::new_in(MetricsAlloc::new(mc)),
            misc_hash: HashTable::new_in(MetricsAlloc::new(mc)),
            dict: None,
            metatable: None,
            mt_cache: None,
        }
    }

    #[inline]
    pub fn shape(&self) -> Shape<'gc> {
        self.shape
    }

    #[inline]
    pub fn metatable(&self) -> Option<Table<'gc>> {
        self.metatable
    }

    #[inline]
    pub fn mt_cache(&self) -> Option<shape::MtCache<'gc>> {
        self.mt_cache
    }

    /// Read the slot directly. Caller is responsible for ensuring `slot`
    /// is in range — used by the IC fast path on a verified shape match.
    #[inline]
    pub unsafe fn property_at(&self, slot: u32) -> Value<'gc> {
        unsafe { *self.properties.get_unchecked(slot as usize) }
    }

    #[inline]
    pub fn raw_get(&self, key: Value<'gc>) -> Value<'gc> {
        if let Some(s) = key.get_string() {
            return self.get_string_key(s);
        }
        if let Some(index) = array_index(key) {
            return match self.array.get(index - 1) {
                Some(value) => *value,
                None => Value::nil(),
            };
        }
        self.misc_hash_get(key, value_hash(key))
    }

    #[inline]
    pub fn raw_get_with_hash(&self, key: Value<'gc>, hash: u64) -> Value<'gc> {
        if let Some(s) = key.get_string() {
            return self.get_string_key(s);
        }
        if let Some(index) = array_index(key) {
            return match self.array.get(index - 1) {
                Some(value) => *value,
                None => Value::nil(),
            };
        }
        self.misc_hash_get(key, hash)
    }

    #[inline]
    fn get_string_key(&self, key: LuaString<'gc>) -> Value<'gc> {
        if let Some(d) = &self.dict {
            let h = lua_string_hash(key);
            return d
                .table
                .find(h, |(k, _)| *k == key)
                .map_or(Value::nil(), |(_, v)| *v);
        }
        match self.shape.find_slot(key) {
            Some(slot) => self.properties[slot as usize],
            None => Value::nil(),
        }
    }

    #[inline]
    fn misc_hash_get(&self, key: Value<'gc>, hash: u64) -> Value<'gc> {
        debug_assert_eq!(hash, value_hash(key));
        debug_assert!(
            key.kind() != ValueKind::String,
            "string keys go through the shape, not misc_hash"
        );
        match self.misc_hash.find(hash, |(k, _)| *k == key) {
            Some((_, v)) => *v,
            None => Value::nil(),
        }
    }

    #[inline]
    pub fn raw_set(&mut self, mc: &Mutation<'gc>, key: Value<'gc>, value: Value<'gc>) {
        if let Some(s) = key.get_string() {
            self.set_string_key(mc, s, value);
            return;
        }
        if let Some(index) = array_index(key) {
            if index > self.array.len() {
                self.array.resize(index, Value::nil());
            }
            self.array[index - 1] = value;
            return;
        }
        if key.is_nil() {
            todo!();
        }
        self.misc_hash_set(key, value, value_hash(key));
    }

    #[inline]
    pub fn raw_set_with_hash(
        &mut self,
        mc: &Mutation<'gc>,
        key: Value<'gc>,
        value: Value<'gc>,
        hash: u64,
    ) {
        if let Some(s) = key.get_string() {
            self.set_string_key(mc, s, value);
            return;
        }
        if let Some(index) = array_index(key) {
            if index > self.array.len() {
                self.array.resize(index, Value::nil());
            }
            self.array[index - 1] = value;
            return;
        }
        if key.is_nil() {
            todo!();
        }
        self.misc_hash_set(key, value, hash);
    }

    fn set_string_key(&mut self, mc: &Mutation<'gc>, key: LuaString<'gc>, value: Value<'gc>) {
        if self.dict.is_some() {
            self.set_string_key_dict(mc, key, value);
            return;
        }

        if let Some(slot) = self.shape.find_slot(key) {
            // Existing slot.
            if value.is_nil() {
                // Deletion of an existing string-keyed slot: migrate to
                // dict mode so the shape tree doesn't carry a dead slot
                // forever. ICs that cached this shape will miss next
                // access (different shape pointer post-migration).
                self.migrate_to_dict(mc);
                self.set_string_key_dict(mc, key, value);
                return;
            }
            self.properties[slot as usize] = value;
        } else {
            // New slot. Cap shape growth to bound the transition tree;
            // beyond MAX_PROPERTIES_FAST, fall back to dict mode.
            if self.shape.slot_count() >= MAX_PROPERTIES_FAST {
                self.migrate_to_dict(mc);
                self.set_string_key_dict(mc, key, value);
                return;
            }
            let new_shape = shape::transition_add_prop(mc, self.shape, key);
            debug_assert_eq!(new_shape.slot_count() as usize, self.properties.len() + 1);
            self.shape = new_shape;
            self.properties.push(value);
        }
        // If *this* table has been adopted as a metatable, mirror the
        // write into the shared `MtCache` bitset so downstream shapes
        // observe the metamethod's new presence (or absence on
        // deletion) without a freshness check.
        if let Some(cache) = self.mt_cache
            && let Some(bit) = shape::metamethod_bit_of_bytes(key.as_bytes())
        {
            cache.update(bit, value);
        }
    }

    fn set_string_key_dict(&mut self, _mc: &Mutation<'gc>, key: LuaString<'gc>, value: Value<'gc>) {
        let dict = self
            .dict
            .as_mut()
            .expect("set_string_key_dict requires dict mode");
        let h = lua_string_hash(key);
        match dict
            .table
            .entry(h, |(k, _)| *k == key, |(k, _)| lua_string_hash(*k))
        {
            hash_table::Entry::Occupied(mut e) => {
                if value.is_nil() {
                    e.remove();
                } else {
                    e.get_mut().1 = value;
                }
            }
            hash_table::Entry::Vacant(e) => {
                if !value.is_nil() {
                    e.insert((key, value));
                }
            }
        }
        if let Some(cache) = self.mt_cache
            && let Some(bit) = shape::metamethod_bit_of_bytes(key.as_bytes())
        {
            cache.update(bit, value);
        }
    }

    /// Move from fast (shape-indexed `properties`) to dict mode. Copy
    /// existing slot values into a new `DictState`, discard the
    /// properties Vec, swap `shape` for a dict sentinel anchored on
    /// the same `mt_cache`. One-way for v1.
    fn migrate_to_dict(&mut self, mc: &Mutation<'gc>) {
        debug_assert!(
            self.dict.is_none(),
            "migrate_to_dict called on already-dict table"
        );
        let descs = self.shape.descriptors();
        let mut table = HashTable::with_capacity_in(descs.len(), MetricsAlloc::new(mc));
        for d in descs {
            let v = self.properties[d.slot as usize];
            if v.is_nil() {
                continue;
            }
            let h = lua_string_hash(d.key);
            table.insert_unique(h, (d.key, v), |(k, _)| lua_string_hash(*k));
        }
        self.properties.clear();
        self.shape = Shape::dict_sentinel(mc, self.shape.mt_cache());
        self.dict = Some(DictState { table });
    }

    fn misc_hash_set(&mut self, key: Value<'gc>, value: Value<'gc>, hash: u64) {
        debug_assert_eq!(hash, value_hash(key));
        debug_assert!(
            key.kind() != ValueKind::String,
            "string keys go through the shape, not misc_hash"
        );
        match self
            .misc_hash
            .entry(hash, |(k, _)| *k == key, |(k, _)| value_hash(*k))
        {
            hash_table::Entry::Occupied(mut e) => {
                if value.is_nil() {
                    e.remove();
                } else {
                    e.get_mut().1 = value;
                }
            }
            hash_table::Entry::Vacant(e) => {
                e.insert((key, value));
            }
        }
    }

    #[inline]
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
