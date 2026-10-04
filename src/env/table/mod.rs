mod hash_part;
mod swiss;

use core::cell::Cell;

use hash_part::{int_hash, lua_string_hash};

use crate::Context;
use crate::dmm::{
    Collect, Finalization, Gc, Mutation, RefLock, Trace, allocator_api::MetricsAlloc,
};
use crate::env::function::Template;
use crate::env::shape::{self, MAX_KEYED_PROPERTIES, MAX_PROPERTIES_FAST, Shape, WeakMode};
use crate::env::string::LuaString;
use crate::env::value::{Value, ValueKind, value_hash};

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Table<'gc>(Gc<'gc, RefLock<TableState<'gc>>>);

impl<'gc> Table<'gc> {
    /// Create a new empty table that starts at the runtime's shared empty
    /// shape. Prefer this constructor whenever a `Context` is in scope.
    pub fn new(ctx: Context<'gc>) -> Self {
        Self::new_with_shape(ctx.mutation(), ctx.empty_shape())
    }

    /// Create a new empty table in `shape`, which has no keys. Used by the
    /// `Lua::new` bootstrap before a `Context` exists.
    pub fn new_with_shape(mc: &Mutation<'gc>, shape: Shape<'gc>) -> Self {
        Table(Gc::new(
            mc,
            RefLock::new(TableState::new(mc, shape, &[], 0)),
        ))
    }

    /// A constructor's table, as `NEWTABLE` makes it from `t`.
    #[inline]
    pub fn from_template(mc: &Mutation<'gc>, t: &Template<'gc>) -> Self {
        Table(Gc::new(
            mc,
            RefLock::new(TableState::new(mc, t.shape, &t.values, t.items as usize)),
        ))
    }

    pub fn raw_get(self, key: Value<'gc>) -> Value<'gc> {
        self.0.borrow().raw_get(key)
    }

    pub fn raw_set(self, ctx: Context<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.0.borrow_mut(ctx.mutation()).raw_set(ctx, key, value);
    }

    /// See [`TableState::raw_set_keyed`].
    pub fn raw_set_keyed(self, ctx: Context<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.0
            .borrow_mut(ctx.mutation())
            .raw_set_keyed(ctx, key, value);
    }

    pub fn raw_len(self) -> usize {
        self.0.borrow().raw_len()
    }

    /// See [`TableState::next`].
    pub fn next(
        self,
        mc: &Mutation<'gc>,
        key: Value<'gc>,
    ) -> Result<Option<(Value<'gc>, Value<'gc>)>, InvalidKey> {
        self.0.borrow().next(mc, key)
    }

    pub fn metatable(self) -> Option<Table<'gc>> {
        self.0.borrow().metatable
    }

    /// Replace the metatable. Re-shapes the table along the
    /// `set_metatable` transition edge so subsequent metamethod queries
    /// observe the new metatable's identity. Subsequent in-place
    /// mutations of the metatable update its shared `MtCache` in place.
    pub fn set_metatable(self, ctx: Context<'gc>, mt: Option<Table<'gc>>) {
        let new_cache = mt.map(|t| t.ensure_mt_cache(ctx));
        let mut state = self.0.borrow_mut(ctx.mutation());
        state.shape = shape::transition_set_metatable(
            ctx.mutation(),
            state.shape,
            new_cache,
            ctx.empty_dict_sentinel(),
        );
        state.metatable = mt;
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
    /// metamethod-named and `__mode` writes update the cache in place.
    /// First adoption computes the initial bits and weak mode.
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
        let weak = WeakMode::of(self.raw_get(Value::string(ctx.symbols().mode)));
        let index = self.raw_get(Value::string(ctx.symbols().mm_index));
        let cache = shape::MtCache::new(ctx.mutation(), bits, weak, index);
        self.0.borrow_mut(ctx.mutation()).mt_cache = Some(cache);
        cache
    }

    /// One ephemeron pass over the deferred weak tables (see [`TableState::converge`]); returns
    /// whether marking must resume and the pass rerun.
    pub(crate) fn converge_weak(fc: &Finalization<'gc>) -> bool {
        let mut resurrected = false;
        for t in fc.deferred() {
            resurrected |= unsafe { Self::from_deferred(t) }.0.borrow().converge(fc);
        }
        resurrected
    }

    /// Clear the deferred weak tables' dead entries (see [`TableState::clear_dead`]).
    pub(crate) fn clear_weak(fc: &Finalization<'gc>) {
        for t in fc.deferred() {
            // Storing nil adopts no pointer, so no barrier is needed. One would re-gray the
            // table, and its retrace would defer it again after `clear_deferred`, tripping the
            // sweep assert.
            unsafe { Self::from_deferred(t).0.as_ref().as_checked_cell() }
                .borrow_mut()
                .clear_dead(fc);
        }
    }

    /// # Safety
    /// `gc` must come from [`Finalization::deferred`], and `TableState::trace` must be the only
    /// caller of `Trace::defer`.
    unsafe fn from_deferred(gc: Gc<'gc, ()>) -> Self {
        Table(unsafe { Gc::cast(gc) })
    }
}

pub struct TableState<'gc> {
    /// Hidden class describing string-keyed property layout + metatable
    /// identity. In dict mode, this is a per-table sentinel shape; ICs
    /// naturally bypass.
    pub(crate) shape: Shape<'gc>,
    /// String-keyed property values, indexed by `shape.find_slot(key)`.
    /// `properties.len() == shape.slot_count()` post-set. Empty in
    /// dict mode (storage moves to `dict`).
    pub(crate) properties: Vec<Value<'gc>, MetricsAlloc<'gc>>,
    /// Integer keys `0..array.len()`, as in LuaJIT: may hold nils, and is
    /// only resized by `rehash_ints`.
    array: Vec<Value<'gc>, MetricsAlloc<'gc>>,
    /// Every other integer key, and floats with an integral value.
    int_hash: hash_part::Part<'gc, i64, MetricsAlloc<'gc>>,
    /// Last border `raw_len` found, as Lua 5.5's `lenhint`.
    len_hint: Cell<usize>,
    /// Keys that are neither strings nor numbers with an integral value.
    misc_hash: hash_part::Part<'gc, Value<'gc>, MetricsAlloc<'gc>>,
    /// Set when this table has dropped to dictionary mode for its
    /// string-keyed properties, triggered by a key past `MAX_PROPERTIES_FAST`
    /// or `MAX_KEYED_PROPERTIES`. Deletion does not migrate: it must not
    /// reorder keys, or a `pairs` loop that clears entries would skip some.
    dict: Option<DictState<'gc>>,
    /// Live metatable handle (for `getmetatable` and metamethod
    /// invocation). Identity is mirrored in `shape.mt_cache`.
    metatable: Option<Table<'gc>>,
    /// Metamethod-presence cache for *this* table when it's used as a
    /// metatable (lazily allocated on first adoption). Its bits and weak
    /// mode are updated in place by every metamethod-named or `__mode`
    /// write to this table; downstream shapes share this same `Gc`
    /// pointer and observe the updates without a freshness check.
    mt_cache: Option<shape::MtCache<'gc>>,
}

// SAFETY: a weak table defers itself, and every edge it skips is resurrected by `converge` or
// dropped by `clear_dead` before sweeping (see `Lua::finalize_and_sweep`).
unsafe impl<'gc> Collect<'gc> for TableState<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.shape);
        cc.trace(&self.metatable);
        cc.trace(&self.mt_cache);
        let mode = self.weak_mode();
        if mode.is_empty() {
            cc.trace(&self.properties);
            cc.trace(&self.array);
            cc.trace(&self.int_hash);
            cc.trace(&self.misc_hash);
            cc.trace(&self.dict);
            return;
        }
        cc.defer();
        let weak_values = mode.contains(WeakMode::VALUES);
        let value = |cc: &mut T, v: &Value<'gc>| {
            if !(weak_values && v.is_weak_object()) {
                cc.trace(v);
            }
        };
        for v in self.properties.iter().chain(&self.array) {
            value(cc, v);
        }
        for e in self.int_hash.iter() {
            value(cc, &e.value);
        }
        if let Some(d) = &self.dict {
            for e in d.table.iter() {
                cc.trace(&e.key);
                value(cc, &e.value);
            }
        }
        // Only this part has object keys. An ephemeron's value waits for `converge` to find out
        // whether its key survives.
        for e in self.misc_hash.iter() {
            if !(mode.contains(WeakMode::KEYS) && e.key.is_weak_object()) {
                cc.trace(&e.key);
                value(cc, &e.value);
            }
        }
    }
}

/// Slow / dictionary-mode storage for string-keyed properties. Replaces
/// the `(shape, properties)` pair in dict mode. ICs bypass naturally
/// because the table's shape becomes a unique dict-sentinel that no IC
/// will have cached.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct DictState<'gc> {
    table: hash_part::Part<'gc, LuaString<'gc>, MetricsAlloc<'gc>>,
}

/// `next` was given a key that is not in the table.
#[derive(Debug, Clone, Copy)]
pub struct InvalidKey;

impl<'gc> TableState<'gc> {
    /// A table in `shape` holding `values` in its slots, with nil in keys
    /// `1..=items` of its array part.
    #[inline(always)]
    fn new(mc: &Mutation<'gc>, shape: Shape<'gc>, values: &[Value<'gc>], items: usize) -> Self {
        debug_assert_eq!(values.len(), shape.slot_count() as usize);
        Self {
            shape,
            properties: filled(mc, values.len(), |i| values[i]),
            // `nils` stays out of line: inlined, its spills cost every table.
            array: match items {
                0 => Vec::new_in(MetricsAlloc::new(mc)),
                n => nils(mc, n + 1),
            },
            int_hash: hash_part::Part::new_in(MetricsAlloc::new(mc)),
            len_hint: Cell::new(0),
            misc_hash: hash_part::Part::new_in(MetricsAlloc::new(mc)),
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

    /// If this table has been adopted as a metatable, mirror a string-keyed
    /// write into the shared `MtCache` (see [`shape::MtCache::mirror`]) so
    /// downstream shapes observe it without a freshness check. Cheap on the
    /// common path: short-circuits when `mt_cache` is None.
    #[inline]
    pub fn maybe_update_mt_bit(&self, key: Value<'gc>, value: Value<'gc>) {
        if let Some(s) = key.get_string()
            && let Some(cache) = self.mt_cache
        {
            cache.mirror(s, value);
        }
    }

    /// `__mode` of this table's metatable.
    #[inline]
    fn weak_mode(&self) -> WeakMode {
        if self.metatable.is_none() {
            return WeakMode::empty();
        }
        self.shape
            .mt_cache()
            .map_or(WeakMode::empty(), |c| c.weak())
    }

    /// One ephemeron pass after marking: resurrect the values of entries whose weak key
    /// survived. Returns whether any was, so marking must resume and the pass rerun.
    fn converge(&self, fc: &Finalization<'gc>) -> bool {
        let weak_values = self.weak_mode().contains(WeakMode::VALUES);
        let mut resurrected = false;
        for e in self.misc_hash.iter() {
            if e.key.is_weak_object()
                && !e.key.is_dead(fc)
                && e.value.is_dead(fc)
                && !(weak_values && e.value.is_weak_object())
            {
                e.value.resurrect(fc);
                resurrected = true;
            }
        }
        resurrected
    }

    /// Remove the entries whose key or value marking left dead, keeping their keys for `next`
    /// as a deletion does. Mode-independent: only an edge `trace` skipped can be dead, so a
    /// `__mode` changed mid-cycle can only make this drop entries early, which §2.5.4 allows.
    fn clear_dead(&mut self, fc: &Finalization<'gc>) {
        let mt_cache = self.mt_cache;
        let cleared = |key| {
            if let Some(c) = mt_cache {
                c.mirror(key, Value::nil());
            }
        };
        for (v, &key) in self.properties.iter_mut().zip(self.shape.keys()) {
            if v.is_dead(fc) {
                *v = Value::nil();
                cleared(key);
            }
        }
        for v in self.array.iter_mut().filter(|v| v.is_dead(fc)) {
            *v = Value::nil();
        }
        self.int_hash.kill_where(|e| e.value.is_dead(fc));
        if let Some(d) = &mut self.dict {
            d.table.kill_where(|e| {
                let dead = e.value.is_dead(fc);
                if dead {
                    cleared(e.key);
                }
                dead
            });
        }
        self.misc_hash
            .kill_where(|e| e.key.is_dead(fc) || e.value.is_dead(fc));
    }

    /// Read the slot directly; used by the IC fast path on a verified shape match.
    ///
    /// # Safety
    ///
    /// `slot` must be in range for this table's shape.
    #[inline]
    pub unsafe fn property_at(&self, slot: u32) -> Value<'gc> {
        unsafe { *self.properties.get_unchecked(slot as usize) }
    }

    /// Write the slot directly; the store-side counterpart of [`Self::property_at`].
    ///
    /// # Safety
    ///
    /// `slot` must be in range for this table's shape.
    #[inline]
    pub unsafe fn set_property_at(&mut self, slot: u32, v: Value<'gc>) {
        unsafe { *self.properties.get_unchecked_mut(slot as usize) = v }
    }

    #[inline]
    pub fn raw_get(&self, key: Value<'gc>) -> Value<'gc> {
        if let Some(s) = key.get_string() {
            return self.get_string_key(s);
        }
        if let Some(i) = int_key(key) {
            return self.get_int(i);
        }
        self.misc_hash_get(key, value_hash(key))
    }

    #[inline]
    fn get_int(&self, key: i64) -> Value<'gc> {
        match usize::try_from(key).ok().and_then(|s| self.array.get(s)) {
            Some(v) => *v,
            None => hash_part::get(&self.int_hash, int_hash(key), key),
        }
    }

    #[inline]
    fn array_slot(&self, key: i64) -> Option<usize> {
        usize::try_from(key).ok().filter(|&s| s < self.array.len())
    }

    #[inline]
    fn get_string_key(&self, key: LuaString<'gc>) -> Value<'gc> {
        if let Some(d) = &self.dict {
            return hash_part::get(&d.table, lua_string_hash(key), key);
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
            key.kind() != ValueKind::String && int_key(key).is_none(),
            "string and integer keys have their own parts"
        );
        hash_part::get(&self.misc_hash, hash, key)
    }

    #[inline]
    pub fn raw_set(&mut self, ctx: Context<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.raw_set_capped(ctx, key, value, MAX_PROPERTIES_FAST);
    }

    /// [`raw_set`](Self::raw_set) for `t[k] = v` (see [`MAX_KEYED_PROPERTIES`]).
    #[inline]
    pub fn raw_set_keyed(&mut self, ctx: Context<'gc>, key: Value<'gc>, value: Value<'gc>) {
        self.raw_set_capped(ctx, key, value, MAX_KEYED_PROPERTIES);
    }

    #[inline]
    fn raw_set_capped(&mut self, ctx: Context<'gc>, key: Value<'gc>, value: Value<'gc>, cap: u32) {
        if let Some(s) = key.get_string() {
            self.set_string_key(ctx, s, value, cap);
            return;
        }
        if let Some(i) = int_key(key) {
            self.set_int_key(i, value);
            return;
        }
        assert!(
            !key.is_nil(),
            "nil table key must be rejected before raw_set"
        );
        self.misc_hash_set(key, value, value_hash(key));
    }

    fn set_string_key(
        &mut self,
        ctx: Context<'gc>,
        key: LuaString<'gc>,
        value: Value<'gc>,
        cap: u32,
    ) {
        if self.dict.is_some() {
            self.set_string_key_dict(key, value);
            return;
        }

        match self.shape.find_slot(key) {
            Some(slot) => {
                // Deletion keeps the slot (nil-valued) so the shape stays stable
                // and `next` can resume from the deleted key.
                self.properties[slot as usize] = value;
                self.maybe_update_mt_bit(Value::string(key), value);
            }
            None => self.add_string_key(ctx, key, value, cap),
        }
    }

    /// Store `key`, which this fast-mode table's shape lacks, moving to dict
    /// mode if the shape has `cap` keys already.
    pub(crate) fn add_string_key(
        &mut self,
        ctx: Context<'gc>,
        key: LuaString<'gc>,
        value: Value<'gc>,
        cap: u32,
    ) {
        debug_assert!(self.dict.is_none() && self.shape.find_slot(key).is_none());
        // Deleting an absent key is a no-op; a slot for it would burn a
        // shape transition, and at the cap the dict migration would drop
        // the nil-valued entries a `pairs` loop still resumes from.
        if value.is_nil() {
            return;
        }
        if self.shape.slot_count() >= cap {
            self.migrate_to_dict(ctx);
            self.set_string_key_dict(key, value);
            return;
        }
        let new_shape = shape::transition_add_prop(ctx.mutation(), self.shape, key);
        debug_assert_eq!(new_shape.slot_count() as usize, self.properties.len() + 1);
        self.shape = new_shape;
        self.properties.push(value);
        self.maybe_update_mt_bit(Value::string(key), value);
    }

    fn set_string_key_dict(&mut self, key: LuaString<'gc>, value: Value<'gc>) {
        let dict = self
            .dict
            .as_mut()
            .expect("set_string_key_dict requires dict mode");
        hash_part::set(&mut dict.table, lua_string_hash(key), key, value);
        self.maybe_update_mt_bit(Value::string(key), value);
    }

    /// Move from fast (shape-indexed `properties`) to dict mode. Copy
    /// existing slot values into a new `DictState`, discard the
    /// properties Vec, swap `shape` for the unique dict sentinel
    /// anchored on the same `mt_cache` (per-`MtCache` registry, or
    /// `State::empty_dict_sentinel` when there's no metatable).
    /// One-way for v1.
    fn migrate_to_dict(&mut self, ctx: Context<'gc>) {
        debug_assert!(
            self.dict.is_none(),
            "migrate_to_dict called on already-dict table"
        );
        let keys = self.shape.keys();
        let mut table =
            hash_part::Part::with_capacity_in(keys.len(), MetricsAlloc::new(ctx.mutation()));
        for (&k, &v) in keys.iter().zip(&self.properties) {
            if v.is_nil() {
                continue;
            }
            hash_part::insert_unique(&mut table, lua_string_hash(k), k, v);
        }
        self.properties.clear();
        self.shape = match self.shape.mt_cache() {
            Some(c) => c.ensure_dict_sentinel(ctx.mutation()),
            None => ctx.empty_dict_sentinel(),
        };
        self.dict = Some(DictState { table });
    }

    /// `t[offset + i] = items[i - 1]`, as `SETLIST` stores a constructor's
    /// positional items.
    pub fn set_list(&mut self, offset: usize, items: &[Value<'gc>]) {
        match self.array.get_mut(offset + 1..offset + 1 + items.len()) {
            Some(slots) => slots.copy_from_slice(items),
            None => {
                for (i, &v) in items.iter().enumerate() {
                    self.set_int_key((offset + 1 + i) as i64, v);
                }
            }
        }
    }

    fn set_int_key(&mut self, key: i64, value: Value<'gc>) {
        if let Some(slot) = self.array_slot(key) {
            self.array[slot] = value;
            return;
        }
        let hash = int_hash(key);
        if value.is_nil() {
            hash_part::set(&mut self.int_hash, hash, key, value);
        } else if hash_part::set_no_grow(&mut self.int_hash, hash, key, value).is_err() {
            self.rehash_ints(key);
            match self.array_slot(key) {
                Some(slot) => self.array[slot] = value,
                None => hash_part::set(&mut self.int_hash, hash, key, value),
            }
        }
    }

    /// LuaJIT's `rehashtab`, run when a new key would grow `int_hash`: size the
    /// array to the largest `2^k + 1` that stays more than half full, counting
    /// `extra`, and rebuild `int_hash` from the live keys that don't fit.
    fn rehash_ints(&mut self, extra: i64) {
        let mut bins = [0u32; MAX_ABITS];
        let mut n = 0;
        for (k, v) in self.array.iter().enumerate() {
            if !v.is_nil() {
                n += count_int(k as i64, &mut bins);
            }
        }
        for e in self.int_hash.iter() {
            n += count_int(e.key, &mut bins);
        }
        n += count_int(extra, &mut bins);
        let asize = best_asize(&bins, n);
        if asize == self.array.len() {
            // Nothing moves; let `hash_part::set` rehash or grow as usual.
            return;
        }
        self.len_hint.set(asize / 2);

        let mut rest: Vec<(i64, Value<'gc>)> = Vec::new();
        if asize < self.array.len() {
            rest.extend(
                (asize..)
                    .zip(self.array.drain(asize..))
                    .filter(|(_, v)| !v.is_nil())
                    .map(|(k, v)| (k as i64, v)),
            );
        }
        self.array.resize(asize, Value::nil());
        let alloc = *self.int_hash.allocator();
        for e in self.int_hash.iter() {
            match usize::try_from(e.key).ok().filter(|&s| s < asize) {
                Some(slot) => self.array[slot] = e.value,
                None => rest.push((e.key, e.value)),
            }
        }
        // No slack, as in LuaJIT: a key that could extend the array must find
        // `int_hash` full and come back here rather than settle in the hash.
        self.int_hash = hash_part::Part::with_capacity_in(rest.len(), alloc);
        for (k, v) in rest {
            hash_part::insert_unique(&mut self.int_hash, int_hash(k), k, v);
        }
    }

    fn misc_hash_set(&mut self, key: Value<'gc>, value: Value<'gc>, hash: u64) {
        debug_assert_eq!(hash, value_hash(key));
        debug_assert!(
            key.kind() != ValueKind::String && int_key(key).is_none(),
            "string and integer keys have their own parts"
        );
        hash_part::set(&mut self.misc_hash, hash, key, value);
    }

    /// A border, found as Lua 5.5's `luaH_getn` does: near the last one
    /// first, so `t[#t + 1] = v` and `t[#t] = nil` stay O(1).
    pub fn raw_len(&self) -> usize {
        let last = self.array.len().saturating_sub(1);
        if last > 0 {
            let empty = |k: usize| self.array[k].is_nil();
            let found = |k: usize| {
                self.len_hint.set(k);
                k
            };
            let binsearch = |mut i: usize, mut j: usize| {
                while j - i > 1 {
                    let m = (i + j) / 2;
                    if empty(m) { j = m } else { i = m }
                }
                found(i)
            };
            let mut limit = self.len_hint.get().clamp(1, last);
            if empty(limit) {
                for _ in 0..4 {
                    if limit <= 1 {
                        break;
                    }
                    limit -= 1;
                    if !empty(limit) {
                        return found(limit);
                    }
                }
                return binsearch(0, limit);
            }
            for _ in 0..4 {
                if limit >= last {
                    break;
                }
                limit += 1;
                if empty(limit) {
                    return found(limit - 1);
                }
            }
            if empty(last) {
                return binsearch(limit, last);
            }
            self.len_hint.set(last);
        }
        if self.int_hash.is_empty() || self.get_int(last as i64 + 1).is_nil() {
            return last;
        }
        // Widen past the array into `int_hash`, then binary search.
        let (mut lo, mut hi) = (last as u64, last as u64 + 1);
        while !self.get_int(hi as i64).is_nil() {
            lo = hi;
            hi *= 2;
            if hi > i64::MAX as u64 / 2 {
                let mut i = 1;
                while !self.get_int(i).is_nil() {
                    i += 1;
                }
                return (i - 1) as usize;
            }
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.get_int(mid as i64).is_nil() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo as usize
    }

    /// Stateless successor for Lua's `next`: the first live entry after
    /// `key` in traversal order (array, integer hash, string keys, misc
    /// hash), `None` once exhausted, `nil` starts from the beginning. Hash
    /// parts are walked by bucket index, so a key deleted mid-traversal
    /// (left in its bucket, dead) still anchors the scan.
    pub fn next(
        &self,
        mc: &Mutation<'gc>,
        key: Value<'gc>,
    ) -> Result<Option<(Value<'gc>, Value<'gc>)>, InvalidKey> {
        let (part, from) = if key.is_nil() {
            (Part::Array, 0)
        } else if let Some(i) = int_key(key) {
            match self.array_slot(i) {
                Some(slot) => (Part::Array, slot + 1),
                None => {
                    let pos = hash_part::position(&self.int_hash, int_hash(i), i);
                    (Part::Ints, pos.ok_or(InvalidKey)? + 1)
                }
            }
        } else if let Some(s) = key.get_string() {
            let pos = match &self.dict {
                Some(d) => hash_part::position(&d.table, lua_string_hash(s), s),
                None => self.shape.find_slot(s).map(|slot| slot as usize),
            };
            (Part::Strings, pos.ok_or(InvalidKey)? + 1)
        } else {
            let pos = hash_part::position(&self.misc_hash, value_hash(key), key);
            (Part::Misc, pos.ok_or(InvalidKey)? + 1)
        };

        if part == Part::Array {
            for (i, v) in self.array.iter().enumerate().skip(from) {
                if !v.is_nil() {
                    return Ok(Some((Value::integer(mc, i as i64), *v)));
                }
            }
        }
        if part <= Part::Ints {
            let from = if part == Part::Ints { from } else { 0 };
            if let Some(e) = hash_part::next_live(&self.int_hash, from) {
                return Ok(Some((Value::integer(mc, e.key), e.value)));
            }
        }
        if part <= Part::Strings {
            let from = if part == Part::Strings { from } else { 0 };
            let found = match &self.dict {
                Some(d) => {
                    hash_part::next_live(&d.table, from).map(|e| (Value::string(e.key), e.value))
                }
                None => self.shape.keys()[from..]
                    .iter()
                    .zip(&self.properties[from..])
                    .map(|(&k, &v)| (Value::string(k), v))
                    .find(|(_, v)| !v.is_nil()),
            };
            if found.is_some() {
                return Ok(found);
            }
        }
        let from = if part == Part::Misc { from } else { 0 };
        Ok(hash_part::next_live(&self.misc_hash, from).map(|e| (e.key, e.value)))
    }
}

/// Which storage part a `next` cursor points into, in traversal order.
#[derive(PartialEq, PartialOrd)]
enum Part {
    Array,
    Ints,
    Strings,
    Misc,
}

/// A vector of `n` values, `f(i)` at `i`. Not `resize` or `extend_from_slice`,
/// whose fill or copy is a library call even for a few values.
#[inline(always)]
fn filled<'gc>(
    mc: &Mutation<'gc>,
    n: usize,
    f: impl Fn(usize) -> Value<'gc>,
) -> Vec<Value<'gc>, MetricsAlloc<'gc>> {
    let mut v = Vec::with_capacity_in(n, MetricsAlloc::new(mc));
    for (i, slot) in v.spare_capacity_mut()[..n].iter_mut().enumerate() {
        slot.write(f(i));
    }
    // SAFETY: the first `n` slots were just written.
    unsafe { v.set_len(n) };
    v
}

#[inline(never)]
fn nils<'gc>(mc: &Mutation<'gc>, n: usize) -> Vec<Value<'gc>, MetricsAlloc<'gc>> {
    filled(mc, n, |_| Value::nil())
}

/// The integer a key normalizes to: integers, and floats with an exact
/// integer value.
fn int_key(key: Value) -> Option<i64> {
    key.get_integer()
        .or_else(|| key.get_float().and_then(crate::vm::num::exact_float_to_int))
}

const MAX_ABITS: usize = 28;
/// LuaJIT's `LJ_MAX_ASIZE`.
const MAX_ASIZE: i64 = (1 << (MAX_ABITS - 1)) + 1;

/// LuaJIT's `countint`: bin `b` holds keys in `(2^b, 2^(b+1)]`, with 0..=2 in bin 0.
fn count_int(key: i64, bins: &mut [u32; MAX_ABITS]) -> u32 {
    if !(0..MAX_ASIZE).contains(&key) {
        return 0;
    }
    let k = key as u32;
    bins[if k > 2 { (k - 1).ilog2() as usize } else { 0 }] += 1;
    1
}

/// LuaJIT's `bestasize`: `n` is the number of keys counted into `bins`.
fn best_asize(bins: &[u32; MAX_ABITS], n: u32) -> usize {
    let (mut sum, mut size) = (0u32, 0);
    let mut b = 0;
    while b < MAX_ABITS && 2 * n > 1 << b && sum != n {
        sum += bins[b];
        if bins[b] > 0 && 2 * sum > 1 << b {
            size = (2usize << b) + 1;
        }
        b += 1;
    }
    size
}
