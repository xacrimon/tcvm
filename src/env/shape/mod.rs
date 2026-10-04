//! Hidden classes (a.k.a. shapes) for Lua tables — tcvm's analog of V8
//! HiddenClasses / JSC Structures.
//!
//! A `Shape` is an immutable, GC-allocated descriptor identifying the
//! structural class of a `Table`: the ordered list of string-keyed
//! properties currently stored on the table, plus the identity of the
//! table's metatable. Two tables with the same shape have the same
//! storage layout (same string keys live at the same `properties`
//! slots) and observe the same metamethod-presence bitset (via the
//! shared `MtCache` pointer).
//!
//! Shapes form a transition tree: starting from `EMPTY_SHAPE`,
//! adding a string key transitions to a child shape; assigning a new
//! metatable transitions along a different edge. Shapes are deduped
//! via per-shape transition tables, so two tables that grow through
//! the same key sequence converge on the same shape pointer.

use core::cell::{Cell, UnsafeCell};
use core::mem::MaybeUninit;

use bitflags::bitflags;
use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::barrier::unlock;
use crate::dmm::{Collect, Gc, GcWeak, Lock, Mutation, RefLock, Trace};
use crate::env::for_each_metamethod;
use crate::env::string::LuaString;
use crate::env::value::Value;

/// V8-style "Map" / hidden class. Copy wrapper over a single Gc pointer
/// for cheap pass-by-value and pointer-equality identity checks.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Shape<'gc>(Gc<'gc, ShapeData<'gc>>);

/// Per-metatable metamethod-presence bitset. Allocated lazily when a
/// table is first adopted as a metatable; updated eagerly by every
/// metamethod-named write to the metatable. Multiple shapes (one per
/// distinct (key-list, metatable) pair) share a single `MtCache`
/// pointer for the same metatable. Identity (the `Gc` address) is
/// what `MtEdge` keys transitions on; the inner `bits` field changes
/// in place across mutations of `__index` / `__newindex` / etc.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct MtCache<'gc>(Gc<'gc, MtCacheData<'gc>>);

/// String keys by slot, shared by the shapes along a chain of property
/// transitions; each reads the prefix of its own `slot_count`.
///
/// Untraced: a live shape's prefix is its `last_key` plus its parent's prefix,
/// so the parent chain keeps it alive. A key past every live shape's prefix may
/// be dead, so only its address is used, and nothing appends after it either:
/// only a shape whose prefix is the whole array may, and it keeps that key alive.
struct Keys<'gc> {
    slots: Box<[UnsafeCell<MaybeUninit<LuaString<'gc>>>], MetricsAlloc<'gc>>,
    len: Cell<u32>,
    /// `(address, slot)` of the first `len` keys, built by the first lookup
    /// past `LINEAR_LOOKUP` keys. A dead key's address may be reused, but not
    /// within one array, since nothing appends once a key in it has died. So an
    /// address matches at most one entry, and one past a shape's prefix is not
    /// that shape's key.
    index: UnsafeCell<Option<HashTable<(usize, u32), MetricsAlloc<'gc>>>>,
}

/// Longest prefix [`Shape::find_slot`] scans rather than hashes.
const LINEAR_LOOKUP: u32 = 16;

// SAFETY: deliberately untraced (see `Keys`); it holds no other pointer.
unsafe impl<'gc> Collect<'gc> for Keys<'gc> {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Keys<'gc> {
    fn new(mc: &Mutation<'gc>, prefix: &[LuaString<'gc>], cap: usize) -> Gc<'gc, Self> {
        let mut slots = Vec::with_capacity_in(cap, MetricsAlloc::new(mc));
        slots.extend(prefix.iter().map(|&k| UnsafeCell::new(MaybeUninit::new(k))));
        slots.resize_with(cap, || UnsafeCell::new(MaybeUninit::uninit()));
        Gc::new(
            mc,
            Keys {
                slots: slots.into_boxed_slice(),
                len: Cell::new(prefix.len() as u32),
                index: UnsafeCell::new(None),
            },
        )
    }

    /// `key`'s slot, if it is among the first `n`. Out of line: inlined into
    /// the IC miss paths, it slowed them more than the call costs.
    #[inline(never)]
    fn find(&self, key: LuaString<'gc>, n: u32) -> Option<u32> {
        // SAFETY: no reference into the index outlives a call.
        let index = match unsafe { &*self.index.get() } {
            Some(index) => index,
            None => self.build_index(),
        };
        let a = addr(key);
        let &(_, i) = index.find(addr_hash(a), |&(b, _)| b == a)?;
        (i < n).then_some(i)
    }

    #[cold]
    #[inline(never)]
    fn build_index(&self) -> &HashTable<(usize, u32), MetricsAlloc<'gc>> {
        let mut t = HashTable::with_capacity_in(self.slots.len(), *Box::allocator(&self.slots));
        for (i, &k) in self.prefix(self.len.get()).iter().enumerate() {
            t.insert_unique(addr_hash(addr(k)), (addr(k), i as u32), |e| addr_hash(e.0));
        }
        // SAFETY: as in `find`.
        unsafe { (*self.index.get()).insert(t) }
    }

    fn prefix(&self, n: u32) -> &[LuaString<'gc>] {
        debug_assert!(n <= self.len.get());
        // SAFETY: slots below `len` are initialized and never written again, and
        // the cells are `repr(transparent)`.
        unsafe { core::slice::from_raw_parts(self.slots.as_ptr().cast(), n as usize) }
    }

    /// Append `key` for a shape whose prefix is `n` long, if that is the
    /// whole array and there is room.
    fn try_push(&self, n: u32, key: LuaString<'gc>) -> bool {
        if self.len.get() != n || n as usize == self.slots.len() {
            return false;
        }
        // SAFETY: no prefix reaches slot `n` yet.
        unsafe { (*self.slots[n as usize].get()).write(key) };
        self.len.set(n + 1);
        // SAFETY: as in `find`.
        if let Some(index) = unsafe { &mut *self.index.get() } {
            index.insert_unique(addr_hash(addr(key)), (addr(key), n), |e| addr_hash(e.0));
        }
        true
    }
}

/// A key's address, all an untraced key may be used for.
fn addr(key: LuaString<'_>) -> usize {
    Gc::as_ptr(key.inner()) as usize
}

/// A hash of an object's address, which has no entropy in its low bits.
fn addr_hash(addr: usize) -> u64 {
    let h = (addr as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ (h >> 32)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Collect)]
#[collect(internal, require_static)]
pub struct MetamethodBits(u32);

macro_rules! emit_bitflags {
    ($(($upper:ident, $bit:literal, $bytes:literal, $field:ident);)*) => {
        bitflags! {
            impl MetamethodBits: u32 {
                $(const $upper = 1 << $bit;)*
            }
        }
    };
}
for_each_metamethod!(emit_bitflags);

/// Most string keys a table holds in shape mode; one more moves it to dict
/// mode. Constant-key stores fill a table the same way each time, so its
/// shapes are shared.
pub const MAX_PROPERTIES_FAST: u32 = 512;

/// The same, for a key added by `t[k] = v`: such a table is usually a map,
/// whose keys and their order differ each time, so it would build a chain of
/// shapes nothing else uses.
pub const MAX_KEYED_PROPERTIES: u32 = 64;

/// Pairing of metamethod byte-name and the bit it occupies in
/// `MetamethodBits`. Used by:
///   - `metamethod_bit_of_bytes` (write-side: detect metatable
///     mutation that affects metamethod presence; updates the
///     metatable's `MtCache` bitset in place).
///   - `Table::ensure_mt_cache` (read-side at first-adoption: walk
///     the metatable's slots and OR together the bits for each
///     present metamethod to seed the cache).
macro_rules! emit_byte_table {
    ($(($upper:ident, $bit:literal, $bytes:literal, $field:ident);)*) => {
        pub const METAMETHOD_TABLE: &[(&[u8], MetamethodBits)] = &[
            $(($bytes, MetamethodBits::$upper),)*
        ];
    };
}
for_each_metamethod!(emit_byte_table);

/// Map a key's bytes to its metamethod bit, if any. Cheap match-on-bytes
/// lookup — no `Context`/`State` access needed, callable from any
/// `Table::raw_set` site.
#[inline]
pub fn metamethod_bit_of_bytes(name: &[u8]) -> Option<MetamethodBits> {
    if !name.starts_with(b"__") {
        return None;
    }
    for (n, bit) in METAMETHOD_TABLE {
        if *n == name {
            return Some(*bit);
        }
    }
    None
}

/// A shape's outgoing transitions, held inline in `ShapeData`. Mutation goes
/// through `Gc::write` on the owning shape to emit the backward barrier.
/// Children are weak, so a transient sub-shape can be reclaimed while its
/// parent lives, and an edge whose child is dropped is erased at the next trace.
pub struct TransitionTable<'gc>(UnsafeCell<Edges<'gc>>);

/// Most shapes have at most one child, and that edge needs no key of its own:
/// it is the child's (see [`EdgeKey::of`]).
enum Edges<'gc> {
    None,
    One(GcWeak<'gc, ShapeData<'gc>>),
    Many(Box<HashTable<Edge<'gc>, MetricsAlloc<'gc>>>),
}

#[derive(Clone, Copy)]
struct Edge<'gc> {
    key: EdgeKey<'gc>,
    child: GcWeak<'gc, ShapeData<'gc>>,
}

/// What a transition adds: a string key, or a metatable (`None` removes it).
/// Untraced in an [`Edge`], as a live child keeps it alive through `last_key`
/// or `mt_cache`; so it may be dead, and is compared and hashed by address.
#[derive(Clone, Copy)]
enum EdgeKey<'gc> {
    Prop(LuaString<'gc>),
    Mt(Option<MtCache<'gc>>),
}

impl<'gc> EdgeKey<'gc> {
    /// The key of the edge leading to `child`.
    fn of(child: &ShapeData<'gc>) -> Self {
        match child.last_key {
            Some(k) => EdgeKey::Prop(k),
            None => EdgeKey::Mt(child.mt_cache),
        }
    }

    fn addr(self) -> usize {
        match self {
            EdgeKey::Prop(k) => addr(k),
            EdgeKey::Mt(c) => c.map_or(0, |c| Gc::as_ptr(c.inner()) as usize),
        }
    }

    fn same(self, other: Self) -> bool {
        let kinds = matches!(
            (self, other),
            (EdgeKey::Prop(_), EdgeKey::Prop(_)) | (EdgeKey::Mt(_), EdgeKey::Mt(_))
        );
        kinds && self.addr() == other.addr()
    }

    fn hash(self) -> u64 {
        addr_hash(self.addr())
    }
}

// SAFETY: traces every child weakly; edge keys are deliberately untraced (see
// `EdgeKey`). Erasing from `trace(&self)` is sound because the collector never
// runs while the mutator borrows the table.
unsafe impl<'gc> Collect<'gc> for TransitionTable<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        let edges = unsafe { &mut *self.0.get() };
        match edges {
            Edges::None => {}
            Edges::One(child) if child.is_dropped() => *edges = Edges::None,
            Edges::One(child) => cc.trace(child),
            Edges::Many(t) => {
                t.retain(|e| !e.child.is_dropped());
                for e in t.iter() {
                    cc.trace(&e.child);
                }
            }
        }
    }
}

impl<'gc> TransitionTable<'gc> {
    fn new() -> Self {
        TransitionTable(UnsafeCell::new(Edges::None))
    }

    /// The live child along `key`.
    fn get(&self, mc: &Mutation<'gc>, key: EdgeKey<'gc>) -> Option<Shape<'gc>> {
        let child = match unsafe { &*self.0.get() } {
            Edges::None => return None,
            Edges::One(child) => *child,
            Edges::Many(t) => t.find(key.hash(), |e| e.key.same(key))?.child,
        };
        let child = child.upgrade(mc)?;
        EdgeKey::of(&child).same(key).then_some(Shape(child))
    }

    /// Make `child` the edge for its key, which has no live child.
    fn insert(&mut self, mc: &Mutation<'gc>, child: Gc<'gc, ShapeData<'gc>>) {
        let edges = self.0.get_mut();
        let new = Edge {
            key: EdgeKey::of(&child),
            child: Gc::downgrade(child),
        };
        match edges {
            Edges::None => *edges = Edges::One(new.child),
            Edges::One(old) => match old.upgrade(mc) {
                None => *edges = Edges::One(new.child),
                Some(old_child) => {
                    let old = Edge {
                        key: EdgeKey::of(&old_child),
                        child: *old,
                    };
                    let mut t = HashTable::new_in(MetricsAlloc::new(mc));
                    for e in [old, new] {
                        t.insert_unique(e.key.hash(), e, |e| e.key.hash());
                    }
                    *edges = Edges::Many(Box::new(t));
                }
            },
            Edges::Many(t) => {
                match t.entry(new.key.hash(), |e| e.key.same(new.key), |e| e.key.hash()) {
                    hash_table::Entry::Occupied(mut o) => *o.get_mut() = new,
                    hash_table::Entry::Vacant(v) => {
                        v.insert(new);
                    }
                }
            }
        }
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct ShapeData<'gc> {
    /// Parent shape this one was derived from. `None` only at the root.
    pub parent: Option<Shape<'gc>>,

    /// The string key added at this transition. `None` at the root and
    /// for shapes reached via a metatable transition (see
    /// `last_mt_change`).
    pub last_key: Option<LuaString<'gc>>,

    /// Number of string-keyed slots this shape covers. Slot N lives at
    /// `TableState::properties[N]`. Append-only along property
    /// transitions; preserved across metatable transitions.
    pub slot_count: u32,

    /// Metatable identity + live metamethod bits and weak mode. `None` = no
    /// metatable. Different metatables → different shapes; transitions
    /// go through `transition_set_metatable`.
    pub mt_cache: Option<MtCache<'gc>>,

    /// True if this shape represents a table that's gone slow
    /// (dictionary mode). Only one dictionary shape per `mt_cache` —
    /// see the per-`State` registry. Dictionary shapes have
    /// `slot_count = 0` and no keys.
    #[collect(require_static)]
    pub is_dict: bool,

    /// Outgoing transition edges, held inline (no separate `Gc`
    /// allocation). Mutation goes through `Gc::write` on the parent
    /// `Gc<ShapeData>` to emit the barrier.
    pub transitions: RefLock<TransitionTable<'gc>>,

    /// This shape's keys are the first `slot_count`, key `i` in slot `i`.
    keys: Gc<'gc, Keys<'gc>>,
}

bitflags! {
    /// A metatable's `__mode`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct WeakMode: u8 {
        const KEYS = 1 << 0;
        const VALUES = 1 << 1;
    }
}

impl WeakMode {
    /// A string containing `k` and/or `v`; anything else is strong. Unlike PUC's `getmode`, bytes
    /// after a NUL count.
    pub fn of(mode: Value<'_>) -> Self {
        let Some(s) = mode.get_string() else {
            return Self::empty();
        };
        let mut m = Self::empty();
        m.set(Self::KEYS, s.as_bytes().contains(&b'k'));
        m.set(Self::VALUES, s.as_bytes().contains(&b'v'));
        m
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct MtCacheData<'gc> {
    #[collect(require_static)]
    pub bits: Cell<MetamethodBits>,
    /// Read by the collector when it traces a table with this metatable.
    #[collect(require_static)]
    pub weak: Cell<WeakMode>,
    /// Bumped by every write of `__index`, so a cache entry that recorded it
    /// knows `__index` still names the same value.
    #[collect(require_static)]
    pub index_epoch: Cell<u32>,
    /// Lazily-allocated dict-mode sentinel for tables that drop into
    /// dict mode while carrying this metatable. Populated on the first
    /// call to `MtCache::ensure_dict_sentinel`; subsequent calls return
    /// the same `Shape` pointer so dict-mode tables sharing a metatable
    /// also share a shape.
    pub dict_sentinel: Lock<Option<Shape<'gc>>>,
}

impl<'gc> MtCacheData<'gc> {
    #[inline]
    pub fn get(&self) -> MetamethodBits {
        self.bits.get()
    }

    /// Set `bit` (if not already set). Safe regardless of mutation
    /// context — the underlying type is `Cell<MetamethodBits>` (a
    /// `u32` repr) with no Gc adoption, so no write barrier required.
    #[inline]
    pub fn set_bit(&self, bit: MetamethodBits) {
        self.bits.set(self.bits.get() | bit);
    }

    /// Clear `bit` (if not already clear). Same barrier-free
    /// rationale as `set_bit`.
    #[inline]
    pub fn clear_bit(&self, bit: MetamethodBits) {
        self.bits.set(self.bits.get() & !bit);
    }

    /// Set or clear `bit` based on whether `value` is nil. Used by
    /// metatable-mutation paths that observe a metamethod-named
    /// `raw_set`.
    #[inline]
    pub fn update(&self, bit: MetamethodBits, value: crate::env::value::Value<'_>) {
        if value.is_nil() {
            self.clear_bit(bit);
        } else {
            self.set_bit(bit);
        }
    }
}

impl<'gc> MtCache<'gc> {
    pub fn new(mc: &Mutation<'gc>, bits: MetamethodBits, weak: WeakMode) -> Self {
        MtCache(Gc::new(
            mc,
            MtCacheData {
                bits: Cell::new(bits),
                weak: Cell::new(weak),
                index_epoch: Cell::new(0),
                dict_sentinel: Lock::new(None),
            },
        ))
    }

    /// Lazily allocate the dict-mode sentinel shape that all tables
    /// carrying this metatable share once they migrate to dict mode.
    /// Hit-side cost is one Gc deref + one option load.
    #[inline]
    pub fn ensure_dict_sentinel(self, mc: &Mutation<'gc>) -> Shape<'gc> {
        if let Some(s) = self.0.dict_sentinel.get() {
            return s;
        }
        let new_shape = Shape::dict_sentinel(mc, Some(self));
        // We're adopting a fresh `Shape` Gc through the `Lock<Option<Shape>>`
        // field, so emit the backward barrier on the parent MtCacheData
        // before writing through `as_cell()`.
        mc.backward_barrier(Gc::erase(self.0), None);
        unsafe { self.0.dict_sentinel.as_cell() }.set(Some(new_shape));
        new_shape
    }

    #[inline]
    pub fn get(self) -> MetamethodBits {
        self.0.get()
    }

    #[inline]
    pub fn set_bit(self, bit: MetamethodBits) {
        self.0.set_bit(bit);
    }

    #[inline]
    pub fn clear_bit(self, bit: MetamethodBits) {
        self.0.clear_bit(bit);
    }

    #[inline]
    pub fn update(self, bit: MetamethodBits, value: crate::env::value::Value<'gc>) {
        self.0.update(bit, value);
    }

    #[inline]
    pub fn weak(self) -> WeakMode {
        self.0.weak.get()
    }

    #[inline]
    pub fn index_epoch(self) -> u32 {
        self.0.index_epoch.get()
    }

    /// Mirror a write of `key` to this cache's metatable into its bits and weak mode.
    #[inline]
    pub fn mirror(self, key: LuaString<'gc>, value: Value<'gc>) {
        match metamethod_bit_of_bytes(key.as_bytes()) {
            Some(bit) => {
                if bit == MetamethodBits::INDEX {
                    let e = &self.0.index_epoch;
                    e.set(e.get().wrapping_add(1));
                }
                self.update(bit, value)
            }
            None if key.as_bytes() == b"__mode" => self.0.weak.set(WeakMode::of(value)),
            None => {}
        }
    }

    #[inline]
    pub fn inner(self) -> Gc<'gc, MtCacheData<'gc>> {
        self.0
    }

    #[inline]
    pub fn ptr_eq(a: Self, b: Self) -> bool {
        Gc::ptr_eq(a.0, b.0)
    }
}

impl<'gc> Shape<'gc> {
    /// Allocate the global empty / root shape. There is exactly one
    /// per `State` — see `State::empty_shape`.
    pub fn root_empty(mc: &Mutation<'gc>) -> Self {
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                slot_count: 0,
                mt_cache: None,
                is_dict: false,
                transitions: RefLock::new(TransitionTable::new()),
                keys: Keys::new(mc, &[], 0),
            },
        ))
    }

    /// Allocate the dictionary-mode sentinel shape for a given metatable
    /// cache. Held in `State`'s per-cache registry.
    pub fn dict_sentinel(mc: &Mutation<'gc>, mt_cache: Option<MtCache<'gc>>) -> Self {
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                slot_count: 0,
                mt_cache,
                is_dict: true,
                transitions: RefLock::new(TransitionTable::new()),
                keys: Keys::new(mc, &[], 0),
            },
        ))
    }

    #[inline]
    pub fn data(self) -> &'gc ShapeData<'gc> {
        Gc::as_ref(self.0)
    }

    #[inline]
    pub fn inner(self) -> Gc<'gc, ShapeData<'gc>> {
        self.0
    }

    #[inline]
    pub fn ptr_eq(a: Self, b: Self) -> bool {
        Gc::ptr_eq(a.0, b.0)
    }

    #[inline]
    pub fn slot_count(self) -> u32 {
        self.data().slot_count
    }

    #[inline]
    pub fn is_dict(self) -> bool {
        self.data().is_dict
    }

    #[inline]
    pub fn mt_cache(self) -> Option<MtCache<'gc>> {
        self.data().mt_cache
    }

    /// String keys by slot.
    #[inline]
    pub fn keys(self) -> &'gc [LuaString<'gc>] {
        Gc::as_ref(self.data().keys).prefix(self.data().slot_count)
    }

    /// Look up a string key in this shape. The IC fast path bypasses this
    /// entirely; only slow paths reach here.
    #[inline]
    pub fn find_slot(self, key: LuaString<'gc>) -> Option<u32> {
        let n = self.slot_count();
        if n > LINEAR_LOOKUP {
            return Gc::as_ref(self.data().keys).find(key, n);
        }
        self.keys().iter().position(|&k| k == key).map(|i| i as u32)
    }

    /// Returns `true` if the metatable behind this shape currently has
    /// `bit` set in its live metamethod bitset. With no metatable,
    /// always `false`. The bitset is updated eagerly by writes to the
    /// metatable, so this read is always live (no freshness check).
    #[inline]
    pub fn has_mm(self, bit: MetamethodBits) -> bool {
        match self.mt_cache() {
            None => false,
            Some(c) => c.get().contains(bit),
        }
    }

    /// Like [`has_mm`](Self::has_mm), but true if *any* of `bits` is set.
    #[inline]
    pub fn has_any_mm(self, bits: MetamethodBits) -> bool {
        match self.mt_cache() {
            None => false,
            Some(c) => c.get().intersects(bits),
        }
    }
}

/// Add a string-keyed property `key` to `parent`, returning the child
/// shape. If a transition already exists for this key, reuse it;
/// otherwise allocate a new shape and install the edge.
pub fn transition_add_prop<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    key: LuaString<'gc>,
) -> Shape<'gc> {
    debug_assert!(
        !parent.is_dict(),
        "shape transitions on dict-mode shapes are forbidden"
    );

    if let Some(child) = parent
        .data()
        .transitions
        .borrow()
        .get(mc, EdgeKey::Prop(key))
    {
        return child;
    }

    // Slow path: allocate a child and install/replace the edge.
    let new_slot = parent.data().slot_count;
    let mut keys = parent.data().keys;
    if !keys.try_push(new_slot, key) {
        keys = Keys::new(
            mc,
            parent.keys(),
            (new_slot as usize + 1).next_power_of_two().max(4),
        );
        keys.try_push(new_slot, key);
    }

    let child_data = Gc::new(
        mc,
        ShapeData {
            parent: Some(parent),
            last_key: Some(key),
            slot_count: new_slot + 1,
            mt_cache: parent.data().mt_cache,
            is_dict: false,
            transitions: RefLock::new(TransitionTable::new()),
            keys,
        },
    );
    let parent_write = Gc::write(mc, parent.0);
    unlock!(parent_write, ShapeData, transitions)
        .borrow_mut()
        .insert(mc, child_data);
    Shape(child_data)
}

/// Switch the metatable on `parent`, returning a shape with the same
/// ordered key list but the new `mt_cache`. Caches transitions on
/// `parent` so repeated `setmetatable(t, mt)` calls share shapes.
///
/// For dict-mode parents the call routes to the per-`mt_cache` dict
/// sentinel (`MtCache::ensure_dict_sentinel`) or, when stripping the
/// metatable, to `State::empty_dict_sentinel` provided by the caller.
pub fn transition_set_metatable<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    new_mt: Option<MtCache<'gc>>,
    no_mt_dict_sentinel: Shape<'gc>,
) -> Shape<'gc> {
    // Dict-mode parent: never go through the prop-transition tree.
    // Route to the unique dict sentinel for the new metatable.
    if parent.is_dict() {
        return match new_mt {
            Some(c) => c.ensure_dict_sentinel(mc),
            None => no_mt_dict_sentinel,
        };
    }

    if let Some(child) = parent
        .data()
        .transitions
        .borrow()
        .get(mc, EdgeKey::Mt(new_mt))
    {
        return child;
    }

    // Slow path: a sibling with `parent`'s keys and the new `mt_cache`. Future
    // prop additions on the result mint their own edges normally.
    let child_data = Gc::new(
        mc,
        ShapeData {
            // Anchor the chain on `parent` itself: walking
            // (None last_key, Some parent) from the new shape reaches
            // `parent.last_key` -> `parent.parent.last_key` -> ...,
            // recovering the same key sequence.
            parent: Some(parent),
            last_key: None,
            slot_count: parent.slot_count(),
            mt_cache: new_mt,
            is_dict: false,
            transitions: RefLock::new(TransitionTable::new()),
            keys: parent.data().keys,
        },
    );
    let parent_write = Gc::write(mc, parent.0);
    unlock!(parent_write, ShapeData, transitions)
        .borrow_mut()
        .insert(mc, child_data);
    Shape(child_data)
}
