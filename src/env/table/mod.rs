mod hash_part;
mod slots;
mod swiss;

use core::cell::Cell;
use core::ptr::NonNull;

use hash_part::{int_hash, lua_string_hash};
use slots::Owned;

use crate::Context;
use crate::dmm::{
    Collect, Finalization, Gc, Mutation, RefLock, Trace, TrailingBytes, allocator_api::GcAlloc,
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
        Self::alloc(mc, shape, &[], 0)
    }

    /// A constructor's table, as `NEWTABLE` makes it from `t`.
    #[inline]
    pub fn from_template(mc: &Mutation<'gc>, t: &Template<'gc>) -> Self {
        Self::alloc(mc, t.shape, &t.values, t.items as usize)
    }

    /// A table in `shape` holding `values` in its slots, with nil in keys
    /// `1..=items` of an array part kept in the table's own cell.
    #[inline(always)]
    fn alloc(mc: &Mutation<'gc>, shape: Shape<'gc>, values: &[Value<'gc>], items: usize) -> Self {
        debug_assert_eq!(values.len(), shape.slot_count() as usize);
        // Only `set_metatable` gives a table a metatable, which it then marks.
        debug_assert!(shape.mt_cache().is_none());
        let cap = shape.inline_cap() as usize;
        let (inline, spilled) = values.split_at(values.len().min(cap));
        let asize = if items == 0 { 0 } else { items + 1 };
        let state = TableState {
            shape,
            spill: match spilled.len() {
                0 => NonNull::dangling(),
                n => slots::alloc(mc, n, spilled),
            },
            spill_cap: spilled.len() as u32,
            inline_len: (cap + asize) as u32,
            array: NonNull::dangling(),
            asize: asize as u32,
            len_hint: Cell::new(0),
            aux: None,
        };
        // SAFETY: initializes all `cap + asize` values.
        let gc = unsafe {
            Gc::new_with_trailing(mc, RefLock::new(state), |dst| {
                let dst = dst.cast::<Value<'gc>>();
                for (i, &v) in inline.iter().enumerate() {
                    dst.add(i).write(v);
                }
                for i in inline.len()..cap + asize {
                    dst.add(i).write(Value::nil());
                }
            })
        };
        if asize > 0 {
            // SAFETY: nothing else refers to the new table yet, and a pointer into its own
            // cell adopts no `Gc`.
            unsafe { (*gc.as_ptr()).array = Gc::trailing_ptr(gc).cast().add(cap) };
        }
        Table(gc)
    }

    /// The named slot at `loc` (see [`SlotLoc`]) of this table, whose state
    /// is `state`.
    ///
    /// # Safety
    ///
    /// `loc` must be resolved against `state`'s shape, for one of its slots.
    #[inline(always)]
    pub unsafe fn load(self, state: &TableState<'gc>, loc: SlotLoc) -> Value<'gc> {
        unsafe { *self.slot_ptr(state, loc) }
    }

    /// Store to the named slot at `loc`; the caller handles the barrier.
    ///
    /// # Safety
    ///
    /// As for [`Self::load`].
    #[inline(always)]
    pub unsafe fn store(self, state: &TableState<'gc>, loc: SlotLoc, v: Value<'gc>) {
        unsafe { *self.slot_ptr(state, loc) = v }
    }

    /// Store the property `to` adds to `state`'s shape at `loc`, its new
    /// slot, and move to `to`; the caller handles the barrier.
    ///
    /// # Safety
    ///
    /// `to` must be a property transition from `state`'s shape, `loc`
    /// resolved against `to` for its last slot, and [`TableState::has_room`]
    /// true for it.
    #[inline(always)]
    pub unsafe fn push(
        self,
        state: &mut TableState<'gc>,
        to: Shape<'gc>,
        loc: SlotLoc,
        v: Value<'gc>,
    ) {
        debug_assert!(state.has_room(loc) && to.slot_count() == state.shape.slot_count() + 1);
        unsafe { *self.slot_ptr(state, loc) = v };
        state.shape = to;
    }

    #[inline(always)]
    unsafe fn slot_ptr(self, state: &TableState<'gc>, loc: SlotLoc) -> *mut Value<'gc> {
        let base = match loc.spilled() {
            false => Gc::as_ptr(self.0).cast::<u8>().cast_mut(),
            true => state.spill.as_ptr().cast::<u8>(),
        };
        unsafe { base.add(loc.offset()).cast() }
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
        self.0.borrow().metatable()
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
        if let Some(c) = self.0.borrow().mt_cache() {
            return c;
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
        let mc = ctx.mutation();
        let cache = shape::MtCache::new(mc, self, bits, weak, index);
        self.0.borrow_mut(mc).aux_or_new(mc).mt_cache = Some(cache);
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
    /// String-keyed property values by slot, `shape.slot_count()` of them:
    /// the first `shape.inline_cap()` in this table's own cell (see
    /// `inline`), the rest in `spill`, a [`slots`] cell of `spill_cap`
    /// values or dangling. None in dict mode (storage moves to `dict`).
    spill: NonNull<Value<'gc>>,
    spill_cap: u32,
    /// Values in this table's own cell, right after it: the inline slots of
    /// the shape it was made in, then the array part a constructor sized.
    inline_len: u32,
    /// Integer keys `0..asize`, as in LuaJIT: may hold nils, and is only
    /// resized by `rehash_ints`. In this table's own cell, a [`slots`]
    /// cell, or dangling when empty.
    array: NonNull<Value<'gc>>,
    asize: u32,
    /// Last border `raw_len` found, as Lua 5.5's `lenhint`.
    len_hint: Cell<u32>,
    /// Every key the array part and the named slots don't hold, made by the
    /// first such key, or by adoption as a metatable.
    aux: Option<Gc<'gc, Owned<Aux<'gc>>>>,
}

// Everything a table holds is GC memory (see `slots` and `GcAlloc`), so sweeping one runs nothing.
const _: () = assert!(!core::mem::needs_drop::<TableState<'static>>());
const _: () = assert!(size_of::<TableState<'static>>() == 48);

// SAFETY: a `TableState` is only allocated by `Table::alloc`, through `Gc::new_with_trailing` with
// `inline_len` values, which never changes, and it has no drop glue.
unsafe impl TrailingBytes for TableState<'_> {
    #[inline(always)]
    fn trailing_len(&self) -> usize {
        self.inline_len as usize * size_of::<Value>()
    }
}

/// Where tables of one shape keep a named slot, resolved once so that a
/// cached access reads neither the shape nor its capacity: a byte offset
/// from the table's lock (`Gc::as_ptr`) for an inline slot, else
/// `SPILLED` plus a byte offset into its spill cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotLoc(u32);

impl SlotLoc {
    const SPILLED: u32 = 1 << 31;

    /// Where tables of `shape` keep `slot`.
    #[inline]
    pub fn new(shape: Shape<'_>, slot: u32) -> Self {
        let cap = shape.inline_cap();
        let size = size_of::<Value>() as u32;
        match slot.checked_sub(cap) {
            None => SlotLoc(RefLock::<TableState>::TRAILING_FROM_LOCK as u32 + slot * size),
            Some(i) => SlotLoc(Self::SPILLED | (i * size)),
        }
    }

    #[inline(always)]
    fn spilled(self) -> bool {
        self.0 & Self::SPILLED != 0
    }

    #[inline(always)]
    fn offset(self) -> usize {
        (self.0 & !Self::SPILLED) as usize
    }
}

// SAFETY: a weak table defers itself, and every edge it skips is resurrected by `converge` or
// dropped by `clear_dead` before sweeping (see `Lua::finalize_and_sweep`).
unsafe impl<'gc> Collect<'gc> for TableState<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.shape);
        // The shape's cache leaves its metatable to the tables that have it.
        if let Some(mt) = self.metatable() {
            cc.trace(&mt);
        }
        let mode = self.weak_mode();
        // SAFETY: non-empty parts outside the table's cell are live `slots` cells.
        if self.spill_cap > 0 {
            unsafe { slots::mark(cc, self.spill) };
        }
        if self.asize > 0 && !self.array_inline() {
            unsafe { slots::mark(cc, self.array) };
        }
        if let Some(h) = self.aux {
            cc.trace(&h);
        }
        let aux = self.aux();
        let (inline, spilled) = self.named();
        if mode.is_empty() {
            cc.trace(inline);
            cc.trace(spilled);
            cc.trace(self.array());
            // Each part's trace marks its memory too.
            if let Some(h) = aux {
                cc.trace(&h.ints);
                cc.trace(&h.misc);
                cc.trace(&h.strs);
                cc.trace(&h.mt_cache);
            }
            return;
        }
        if let Some(h) = aux {
            hash_part::mark(cc, &h.ints);
            hash_part::mark(cc, &h.misc);
            hash_part::mark(cc, &h.strs);
            cc.trace(&h.mt_cache);
        }
        cc.defer();
        let weak_values = mode.contains(WeakMode::VALUES);
        let value = |cc: &mut T, v: &Value<'gc>| {
            if !(weak_values && v.is_weak_object()) {
                cc.trace(v);
            }
        };
        for v in inline.iter().chain(spilled).chain(self.array()) {
            value(cc, v);
        }
        let Some(h) = aux else { return };
        for e in h.ints.iter() {
            value(cc, &e.value);
        }
        for e in h.strs.iter() {
            cc.trace(&e.key);
            value(cc, &e.value);
        }
        // Only this part has object keys. An ephemeron's value waits for `converge` to find out
        // whether its key survives.
        for e in h.misc.iter() {
            if !(mode.contains(WeakMode::KEYS) && e.key.is_weak_object()) {
                cc.trace(&e.key);
                value(cc, &e.value);
            }
        }
    }
}

/// What a table makes only once it needs it, in an [`Owned`] cell: its
/// hash parts, and its `MtCache` once it is a metatable.
struct Aux<'gc> {
    /// Integer keys outside the array part, and floats with an integral
    /// value.
    ints: hash_part::Part<'gc, i64, GcAlloc<'gc>>,
    /// Keys that are neither strings nor numbers with an integral value.
    misc: hash_part::Part<'gc, Value<'gc>, GcAlloc<'gc>>,
    /// String keys, in dictionary mode only: past `MAX_PROPERTIES_FAST` or
    /// `MAX_KEYED_PROPERTIES` keys, a table's shape becomes a dict sentinel
    /// no IC caches. Deletion does not migrate: it must not reorder keys, or
    /// a `pairs` loop that clears entries would skip some.
    strs: hash_part::Part<'gc, LuaString<'gc>, GcAlloc<'gc>>,
    /// See [`Table::ensure_mt_cache`].
    mt_cache: Option<shape::MtCache<'gc>>,
}

/// `next` was given a key that is not in the table.
#[derive(Debug, Clone, Copy)]
pub struct InvalidKey;

impl<'gc> TableState<'gc> {
    #[inline]
    pub fn shape(&self) -> Shape<'gc> {
        self.shape
    }

    #[inline]
    pub fn metatable(&self) -> Option<Table<'gc>> {
        // SAFETY: this table is reachable, and the cache comes from its shape.
        self.shape.mt_cache().map(|c| unsafe { c.table() })
    }

    /// This table's cache as a metatable, made by [`Table::ensure_mt_cache`].
    #[inline]
    pub fn mt_cache(&self) -> Option<shape::MtCache<'gc>> {
        self.aux().and_then(|a| a.mt_cache)
    }

    /// If this table has been adopted as a metatable, mirror a string-keyed
    /// write into the shared `MtCache` (see [`shape::MtCache::mirror`]) so
    /// downstream shapes observe it without a freshness check.
    #[inline]
    pub fn maybe_update_mt_bit(&self, key: Value<'gc>, value: Value<'gc>) {
        if let Some(s) = key.get_string()
            && let Some(cache) = self.mt_cache()
        {
            cache.mirror(s, value);
        }
    }

    /// `__mode` of this table's metatable.
    #[inline]
    fn weak_mode(&self) -> WeakMode {
        self.shape
            .mt_cache()
            .map_or(WeakMode::empty(), |c| c.weak())
    }

    /// One ephemeron pass after marking: resurrect the values of entries whose weak key
    /// survived. Returns whether any was, so marking must resume and the pass rerun.
    fn converge(&self, fc: &Finalization<'gc>) -> bool {
        let weak_values = self.weak_mode().contains(WeakMode::VALUES);
        let mut resurrected = false;
        let Some(h) = self.aux() else {
            return false;
        };
        for e in h.misc.iter() {
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
        let mt_cache = self.mt_cache();
        let cleared = |key| {
            if let Some(c) = mt_cache {
                c.mirror(key, Value::nil());
            }
        };
        let keys = self.shape.keys();
        let (inline, spilled) = self.named_mut();
        for (v, &key) in inline.iter_mut().chain(spilled).zip(keys) {
            if v.is_dead(fc) {
                *v = Value::nil();
                cleared(key);
            }
        }
        for v in self.array_mut().iter_mut().filter(|v| v.is_dead(fc)) {
            *v = Value::nil();
        }
        let Some(h) = self.aux_mut() else { return };
        h.ints.kill_where(|e| e.value.is_dead(fc));
        h.strs.kill_where(|e| {
            let dead = e.value.is_dead(fc);
            if dead {
                cleared(e.key);
            }
            dead
        });
        h.misc
            .kill_where(|e| e.key.is_dead(fc) || e.value.is_dead(fc));
    }

    #[inline(always)]
    fn aux(&self) -> Option<&Aux<'gc>> {
        // SAFETY: only this table refers to its cell, so its borrow stands in for the cell's.
        self.aux.map(|h| unsafe { Owned::get(h) })
    }

    #[inline(always)]
    fn aux_mut(&mut self) -> Option<&mut Aux<'gc>> {
        // SAFETY: as in `aux`.
        self.aux.map(|h| unsafe { Owned::get_mut(h) })
    }

    /// The aux cell, made if this table has none yet.
    #[inline]
    fn aux_or_new(&mut self, mc: &Mutation<'gc>) -> &mut Aux<'gc> {
        let h = *self.aux.get_or_insert_with(|| {
            Owned::new(
                mc,
                Aux {
                    ints: hash_part::Part::new_in(GcAlloc::new(mc)),
                    misc: hash_part::Part::new_in(GcAlloc::new(mc)),
                    strs: hash_part::Part::new_in(GcAlloc::new(mc)),
                    mt_cache: None,
                },
            )
        });
        // SAFETY: as in `aux`.
        unsafe { Owned::get_mut(h) }
    }

    /// This table's own cell's values (see `inline_len`).
    #[inline(always)]
    fn inline(&self) -> NonNull<Value<'gc>> {
        // SAFETY: every `TableState` lives in a `Table`'s cell.
        unsafe { RefLock::trailing_ptr_of(self).cast() }
    }

    /// Whether the array part is in this table's own cell.
    #[inline]
    fn array_inline(&self) -> bool {
        let start = self.inline().as_ptr().addr();
        let at = self.array.as_ptr().addr();
        at.wrapping_sub(start) < self.inline_len as usize * size_of::<Value>()
    }

    /// The named slots: the inline ones, then the spilled ones.
    #[inline(always)]
    fn named(&self) -> (&[Value<'gc>], &[Value<'gc>]) {
        let (n, cap) = (
            self.shape.slot_count() as usize,
            self.shape.inline_cap() as usize,
        );
        // SAFETY: the cell holds the shape's `cap` inline slots, and `spill` the rest.
        unsafe {
            (
                core::slice::from_raw_parts(self.inline().as_ptr(), n.min(cap)),
                core::slice::from_raw_parts(self.spill.as_ptr(), n.saturating_sub(cap)),
            )
        }
    }

    #[inline(always)]
    fn named_mut(&mut self) -> (&mut [Value<'gc>], &mut [Value<'gc>]) {
        let (n, cap) = (
            self.shape.slot_count() as usize,
            self.shape.inline_cap() as usize,
        );
        // SAFETY: as in `named`; the two never overlap.
        unsafe {
            (
                core::slice::from_raw_parts_mut(self.inline().as_ptr(), n.min(cap)),
                core::slice::from_raw_parts_mut(self.spill.as_ptr(), n.saturating_sub(cap)),
            )
        }
    }

    #[inline]
    fn named_ptr(&self, slot: u32) -> *mut Value<'gc> {
        debug_assert!(slot < self.shape.slot_count());
        let cap = self.shape.inline_cap();
        // SAFETY: in bounds, as in `named`.
        unsafe {
            match slot.checked_sub(cap) {
                None => self.inline().as_ptr().add(slot as usize),
                Some(i) => self.spill.as_ptr().add(i as usize),
            }
        }
    }

    /// The value in `slot` of this table's shape.
    #[inline]
    pub fn named_get(&self, slot: u32) -> Value<'gc> {
        unsafe { *self.named_ptr(slot) }
    }

    /// Store to `slot` of this table's shape.
    #[inline]
    pub fn named_set(&mut self, slot: u32, v: Value<'gc>) {
        unsafe { *self.named_ptr(slot) = v }
    }

    /// Whether the slot at `loc`, the next one this table's shape would
    /// add, has storage: inline slots always do, spilled ones while the
    /// spill cell has room.
    #[inline(always)]
    pub fn has_room(&self, loc: SlotLoc) -> bool {
        !loc.spilled() || loc.offset() < self.spill_cap as usize * size_of::<Value>()
    }

    #[inline(always)]
    fn array(&self) -> &[Value<'gc>] {
        // SAFETY: `array` holds `asize` values, in a cell nothing else refers to.
        unsafe { core::slice::from_raw_parts(self.array.as_ptr(), self.asize as usize) }
    }

    #[inline(always)]
    fn array_mut(&mut self) -> &mut [Value<'gc>] {
        // SAFETY: as in `array`.
        unsafe { core::slice::from_raw_parts_mut(self.array.as_ptr(), self.asize as usize) }
    }

    /// The array-part slot for key `i`, if the array part covers it.
    #[inline(always)]
    pub fn array_get(&self, i: usize) -> Option<Value<'gc>> {
        // SAFETY: in bounds.
        (i < self.asize as usize).then(|| unsafe { *self.array.as_ptr().add(i) })
    }

    /// Store to an array-part slot.
    ///
    /// # Safety
    ///
    /// `i` must be inside the array part (see [`Self::array_get`]).
    #[inline(always)]
    pub unsafe fn set_array_at(&mut self, i: usize, v: Value<'gc>) {
        debug_assert!(i < self.asize as usize);
        unsafe { *self.array.as_ptr().add(i) = v }
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
        match usize::try_from(key).ok().and_then(|s| self.array_get(s)) {
            Some(v) => v,
            None => match self.aux() {
                Some(h) => hash_part::get(&h.ints, int_hash(key), key),
                None => Value::nil(),
            },
        }
    }

    #[inline]
    fn array_slot(&self, key: i64) -> Option<usize> {
        usize::try_from(key)
            .ok()
            .filter(|&s| s < self.asize as usize)
    }

    #[inline]
    fn get_string_key(&self, key: LuaString<'gc>) -> Value<'gc> {
        if self.shape.is_dict() {
            return self.aux().map_or(Value::nil(), |h| {
                hash_part::get(&h.strs, lua_string_hash(key), key)
            });
        }
        match self.shape.find_slot(key) {
            Some(slot) => self.named_get(slot),
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
        match self.aux() {
            Some(h) => hash_part::get(&h.misc, hash, key),
            None => Value::nil(),
        }
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
            self.set_int_key(ctx.mutation(), i, value);
            return;
        }
        assert!(
            !key.is_nil(),
            "nil table key must be rejected before raw_set"
        );
        self.misc_hash_set(ctx.mutation(), key, value, value_hash(key));
    }

    fn set_string_key(
        &mut self,
        ctx: Context<'gc>,
        key: LuaString<'gc>,
        value: Value<'gc>,
        cap: u32,
    ) {
        if self.shape.is_dict() {
            self.set_string_key_dict(ctx.mutation(), key, value);
            return;
        }

        match self.shape.find_slot(key) {
            Some(slot) => {
                // Deletion keeps the slot (nil-valued) so the shape stays stable
                // and `next` can resume from the deleted key.
                self.named_set(slot, value);
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
        debug_assert!(!self.shape.is_dict() && self.shape.find_slot(key).is_none());
        // Deleting an absent key is a no-op; a slot for it would burn a
        // shape transition, and at the cap the dict migration would drop
        // the nil-valued entries a `pairs` loop still resumes from.
        if value.is_nil() {
            return;
        }
        if self.shape.slot_count() >= cap {
            self.migrate_to_dict(ctx);
            self.set_string_key_dict(ctx.mutation(), key, value);
            return;
        }
        let new_shape = shape::transition_add_prop(ctx.mutation(), self.shape, key);
        let slot = self.shape.slot_count();
        if !self.has_room(SlotLoc::new(new_shape, slot)) {
            self.grow_spill(ctx.mutation());
        }
        self.shape = new_shape;
        self.named_set(slot, value);
        self.maybe_update_mt_bit(Value::string(key), value);
    }

    /// Move the spilled slots to a cell with room for more; the old one is
    /// left for the sweep.
    #[cold]
    fn grow_spill(&mut self, mc: &Mutation<'gc>) {
        let old = self.named().1;
        let cap = (old.len() * 2).max(4);
        self.spill = slots::alloc(mc, cap, old);
        self.spill_cap = cap as u32;
    }

    fn set_string_key_dict(&mut self, mc: &Mutation<'gc>, key: LuaString<'gc>, value: Value<'gc>) {
        debug_assert!(self.shape.is_dict());
        let strs = &mut self.aux_or_new(mc).strs;
        hash_part::set(strs, lua_string_hash(key), key, value);
        self.maybe_update_mt_bit(Value::string(key), value);
    }

    /// Move the named slots to the string part and the shape to the dict
    /// sentinel of its metatable (or `State::empty_dict_sentinel`). One way.
    fn migrate_to_dict(&mut self, ctx: Context<'gc>) {
        debug_assert!(!self.shape.is_dict());
        let mc = ctx.mutation();
        let keys = self.shape.keys();
        let mut table = hash_part::Part::with_capacity_in(keys.len(), GcAlloc::new(mc));
        let (inline, spilled) = self.named();
        for (&k, &v) in keys.iter().zip(inline.iter().chain(spilled)) {
            if v.is_nil() {
                continue;
            }
            hash_part::insert_unique(&mut table, lua_string_hash(k), k, v);
        }
        self.aux_or_new(mc).strs = table;
        // The inline slots stay in the cell, unread: the dict sentinel has none.
        self.spill = NonNull::dangling();
        self.spill_cap = 0;
        self.shape = match self.shape.mt_cache() {
            Some(c) => c.ensure_dict_sentinel(mc),
            None => ctx.empty_dict_sentinel(),
        };
    }

    /// `t[offset + i] = items[i - 1]`, as `SETLIST` stores a constructor's
    /// positional items.
    pub fn set_list(&mut self, mc: &Mutation<'gc>, offset: usize, items: &[Value<'gc>]) {
        match self
            .array_mut()
            .get_mut(offset + 1..offset + 1 + items.len())
        {
            Some(slots) => slots.copy_from_slice(items),
            None => {
                for (i, &v) in items.iter().enumerate() {
                    self.set_int_key(mc, (offset + 1 + i) as i64, v);
                }
            }
        }
    }

    fn set_int_key(&mut self, mc: &Mutation<'gc>, key: i64, value: Value<'gc>) {
        if let Some(slot) = self.array_slot(key) {
            self.array_mut()[slot] = value;
            return;
        }
        let hash = int_hash(key);
        if value.is_nil() {
            if let Some(h) = self.aux_mut() {
                hash_part::set(&mut h.ints, hash, key, value);
            }
            return;
        }
        // No parts yet is a full `ints`: the key may belong in the array.
        let stored = self
            .aux_mut()
            .is_some_and(|h| hash_part::set_no_grow(&mut h.ints, hash, key, value).is_ok());
        if !stored {
            self.rehash_ints(mc, key);
            match self.array_slot(key) {
                Some(slot) => self.array_mut()[slot] = value,
                None => hash_part::set(&mut self.aux_or_new(mc).ints, hash, key, value),
            }
        }
    }

    /// LuaJIT's `rehashtab`, run when a new key would grow `int_hash`: size the
    /// array to the largest `2^k + 1` that stays more than half full, counting
    /// `extra`, and rebuild `int_hash` from the live keys that don't fit.
    fn rehash_ints(&mut self, mc: &Mutation<'gc>, extra: i64) {
        let mut bins = [0u32; MAX_ABITS];
        let mut n = count_array(self.array(), &mut bins);
        let ints = self.aux().map(|h| &h.ints);
        for e in ints.into_iter().flat_map(|t| t.iter()) {
            n += count_int(e.key, &mut bins);
        }
        n += count_int(extra, &mut bins);
        let asize = best_asize(&bins, n);
        if asize == self.asize as usize {
            // Nothing moves; let `hash_part::set` rehash or grow as usual.
            return;
        }
        self.len_hint.set(asize as u32 / 2);

        let old = self.array();
        let mut rest: Vec<(i64, Value<'gc>)> = Vec::new();
        if asize < old.len() {
            rest.extend(
                (asize..)
                    .zip(&old[asize..])
                    .filter(|(_, v)| !v.is_nil())
                    .map(|(k, &v)| (k as i64, v)),
            );
        }
        // The old cell is left for the sweep.
        self.array = match asize {
            0 => NonNull::dangling(),
            _ => slots::alloc(mc, asize, old),
        };
        self.asize = asize as u32;
        let array = self.array;
        let ints = self.aux().map(|h| &h.ints);
        for e in ints.into_iter().flat_map(|t| t.iter()) {
            match usize::try_from(e.key).ok().filter(|&s| s < asize) {
                // SAFETY: the new cell holds `asize` values.
                Some(slot) => unsafe { *array.as_ptr().add(slot) = e.value },
                None => rest.push((e.key, e.value)),
            }
        }
        if self.aux.is_none() && rest.is_empty() {
            return;
        }
        // No slack, as in LuaJIT: a key that could extend the array must find
        // `ints` full and come back here rather than settle in the hash.
        let h = self.aux_or_new(mc);
        h.ints = hash_part::Part::with_capacity_in(rest.len(), GcAlloc::new(mc));
        for (k, v) in rest {
            hash_part::insert_unique(&mut h.ints, int_hash(k), k, v);
        }
    }

    fn misc_hash_set(&mut self, mc: &Mutation<'gc>, key: Value<'gc>, value: Value<'gc>, hash: u64) {
        debug_assert_eq!(hash, value_hash(key));
        debug_assert!(
            key.kind() != ValueKind::String && int_key(key).is_none(),
            "string and integer keys have their own parts"
        );
        match self.aux_mut() {
            Some(h) => hash_part::set(&mut h.misc, hash, key, value),
            None if value.is_nil() => {}
            None => hash_part::set(&mut self.aux_or_new(mc).misc, hash, key, value),
        }
    }

    /// A border, found as Lua 5.5's `luaH_getn` does: near the last one
    /// first, so `t[#t + 1] = v` and `t[#t] = nil` stay O(1).
    pub fn raw_len(&self) -> usize {
        let array = self.array();
        let last = array.len().saturating_sub(1);
        if last > 0 {
            let empty = |k: usize| array[k].is_nil();
            let found = |k: usize| {
                self.len_hint.set(k as u32);
                k
            };
            let binsearch = |mut i: usize, mut j: usize| {
                while j - i > 1 {
                    let m = (i + j) / 2;
                    if empty(m) { j = m } else { i = m }
                }
                found(i)
            };
            let mut limit = (self.len_hint.get() as usize).clamp(1, last);
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
            self.len_hint.set(last as u32);
        }
        if self.aux().is_none_or(|h| h.ints.is_empty()) || self.get_int(last as i64 + 1).is_nil() {
            return last;
        }
        // Widen past the array into `ints`, then binary search.
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
        let hash = self.aux();
        let (part, from) = if key.is_nil() {
            (Part::Array, 0)
        } else if let Some(i) = int_key(key) {
            match self.array_slot(i) {
                Some(slot) => (Part::Array, slot + 1),
                None => {
                    let pos = hash.and_then(|h| hash_part::position(&h.ints, int_hash(i), i));
                    (Part::Ints, pos.ok_or(InvalidKey)? + 1)
                }
            }
        } else if let Some(s) = key.get_string() {
            let pos = match self.shape.is_dict() {
                true => hash.and_then(|h| hash_part::position(&h.strs, lua_string_hash(s), s)),
                false => self.shape.find_slot(s).map(|slot| slot as usize),
            };
            (Part::Strings, pos.ok_or(InvalidKey)? + 1)
        } else {
            let pos = hash.and_then(|h| hash_part::position(&h.misc, value_hash(key), key));
            (Part::Misc, pos.ok_or(InvalidKey)? + 1)
        };

        if part == Part::Array {
            for (i, v) in self.array().iter().enumerate().skip(from) {
                if !v.is_nil() {
                    return Ok(Some((Value::integer(mc, i as i64), *v)));
                }
            }
        }
        if part <= Part::Ints {
            let from = if part == Part::Ints { from } else { 0 };
            if let Some(e) = hash.and_then(|h| hash_part::next_live(&h.ints, from)) {
                return Ok(Some((Value::integer(mc, e.key), e.value)));
            }
        }
        if part <= Part::Strings {
            let from = if part == Part::Strings { from } else { 0 };
            let found = match self.shape.is_dict() {
                true => hash
                    .and_then(|h| hash_part::next_live(&h.strs, from))
                    .map(|e| (Value::string(e.key), e.value)),
                false => {
                    let (inline, spilled) = self.named();
                    self.shape.keys()[from..]
                        .iter()
                        .zip(inline.iter().chain(spilled).skip(from))
                        .map(|(&k, &v)| (Value::string(k), v))
                        .find(|(_, v)| !v.is_nil())
                }
            };
            if found.is_some() {
                return Ok(found);
            }
        }
        let from = if part == Part::Misc { from } else { 0 };
        Ok(hash
            .and_then(|h| hash_part::next_live(&h.misc, from))
            .map(|e| (e.key, e.value)))
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

/// LuaJIT's `countarray`: `count_int` for the key of every non-nil value in
/// `array`, a bin at a time, so the loop needs no per-key bin.
fn count_array(array: &[Value], bins: &mut [u32; MAX_ABITS]) -> u32 {
    let array = &array[..array.len().min(MAX_ASIZE as usize)];
    let (mut n, mut lo) = (0, 0);
    for (b, bin) in bins.iter_mut().enumerate() {
        let hi = ((2usize << b) + 1).min(array.len());
        if lo >= hi {
            break;
        }
        let c = array[lo..hi].iter().filter(|v| !v.is_nil()).count() as u32;
        *bin += c;
        n += c;
        lo = hi;
    }
    n
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_array_bins_like_count_int() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for len in (0..40).chain([63, 64, 65, 129, 1000, 4097]) {
            for density in [0, 1, 2, 4] {
                let array: Vec<Value> = (0..len)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        match (seed >> 33) % 4 < density {
                            true => Value::boolean(true),
                            false => Value::nil(),
                        }
                    })
                    .collect();
                let (mut by_key, mut by_bin) = ([0; MAX_ABITS], [0; MAX_ABITS]);
                let n: u32 = (0..len)
                    .filter(|&k| !array[k].is_nil())
                    .map(|k| count_int(k as i64, &mut by_key))
                    .sum();
                assert_eq!(count_array(&array, &mut by_bin), n, "len {len}");
                assert_eq!(by_bin, by_key, "len {len}");
            }
        }
    }
}
