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

use core::cell::Cell;

use bitflags::bitflags;
use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::barrier::unlock;
use crate::dmm::{Collect, Gc, GcWeak, Mutation, RefLock};
use crate::env::string::LuaString;

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
pub struct MtCache<'gc>(Gc<'gc, MtCacheData>);

/// One descriptor entry: a string key and the slot it occupies in
/// `TableState::properties`. Stored in `ShapeData::descriptors` sorted
/// by `LuaString` pointer identity for O(log n) binary search on slow
/// paths.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Descriptor<'gc> {
    pub key: LuaString<'gc>,
    pub slot: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Collect)]
#[collect(internal, require_static)]
pub struct MetamethodBits(u32);

bitflags! {
    impl MetamethodBits: u32 {
        const INDEX     = 1 << 0;
        const NEWINDEX  = 1 << 1;
        const ADD       = 1 << 2;
        const SUB       = 1 << 3;
        const MUL       = 1 << 4;
        const DIV       = 1 << 5;
        const MOD       = 1 << 6;
        const POW       = 1 << 7;
        const IDIV      = 1 << 8;
        const BAND      = 1 << 9;
        const BOR       = 1 << 10;
        const BXOR      = 1 << 11;
        const BNOT      = 1 << 12;
        const SHL       = 1 << 13;
        const SHR       = 1 << 14;
        const UNM       = 1 << 15;
        const EQ        = 1 << 16;
        const LT        = 1 << 17;
        const LE        = 1 << 18;
        const CONCAT    = 1 << 19;
        const LEN       = 1 << 20;
        const CALL      = 1 << 21;
        const TOSTRING  = 1 << 22;
    }
}

/// Maximum string-keyed properties a table can hold in fast mode.
/// Beyond this, set_string_key migrates the table to dictionary mode
/// to bound memory growth in the transition tree.
pub const MAX_PROPERTIES_FAST: u32 = 64;

/// Pairing of metamethod byte-name and the bit it occupies in
/// `MetamethodBits`. Used by:
///   - `metamethod_bit_of_bytes` (write-side: detect metatable
///     mutation that affects metamethod presence; updates the
///     metatable's `MtCache` bitset in place).
///   - `Table::ensure_mt_cache` (read-side at first-adoption: walk
///     the metatable's slots and OR together the bits for each
///     present metamethod to seed the cache).
pub const METAMETHOD_TABLE: &[(&[u8], MetamethodBits)] = &[
    (b"__index", MetamethodBits::INDEX),
    (b"__newindex", MetamethodBits::NEWINDEX),
    (b"__add", MetamethodBits::ADD),
    (b"__sub", MetamethodBits::SUB),
    (b"__mul", MetamethodBits::MUL),
    (b"__div", MetamethodBits::DIV),
    (b"__mod", MetamethodBits::MOD),
    (b"__pow", MetamethodBits::POW),
    (b"__idiv", MetamethodBits::IDIV),
    (b"__band", MetamethodBits::BAND),
    (b"__bor", MetamethodBits::BOR),
    (b"__bxor", MetamethodBits::BXOR),
    (b"__bnot", MetamethodBits::BNOT),
    (b"__shl", MetamethodBits::SHL),
    (b"__shr", MetamethodBits::SHR),
    (b"__unm", MetamethodBits::UNM),
    (b"__eq", MetamethodBits::EQ),
    (b"__lt", MetamethodBits::LT),
    (b"__le", MetamethodBits::LE),
    (b"__concat", MetamethodBits::CONCAT),
    (b"__len", MetamethodBits::LEN),
    (b"__call", MetamethodBits::CALL),
    (b"__tostring", MetamethodBits::TOSTRING),
];

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

/// The transition table is held inline inside `ShapeData` (mirrors
/// `Prototype.ic_table`). Mutation goes through `Gc::write` on the
/// owning `ShapeData` to emit the backward barrier — children adopted
/// as `GcWeak` won't retain their targets, so a transient sub-shape
/// can be reclaimed by GC even while its parent is alive.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct TransitionTable<'gc> {
    /// Property-add edges keyed by the added LuaString.
    pub by_prop: HashTable<PropEdge<'gc>, MetricsAlloc<'gc>>,
    /// Set-metatable edges keyed by the new MtCache identity (None = no MT).
    pub by_mt: HashTable<MtEdge<'gc>, MetricsAlloc<'gc>>,
}

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct PropEdge<'gc> {
    pub key: LuaString<'gc>,
    pub child: GcWeak<'gc, ShapeData<'gc>>,
}

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct MtEdge<'gc> {
    pub mt_cache: Option<MtCache<'gc>>,
    pub child: GcWeak<'gc, ShapeData<'gc>>,
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

    /// Metatable identity + live metamethod bits. `None` = no
    /// metatable. Different metatables → different shapes; transitions
    /// go through `transition_set_metatable`.
    pub mt_cache: Option<MtCache<'gc>>,

    /// True if this shape represents a table that's gone slow
    /// (dictionary mode). Only one dictionary shape per `mt_cache` —
    /// see the per-`State` registry. Dictionary shapes have
    /// `slot_count = 0` and an empty `descriptors`.
    #[collect(require_static)]
    pub is_dict: bool,

    /// Outgoing transition edges, held inline (no separate `Gc`
    /// allocation). Mutation goes through `Gc::write` on the parent
    /// `Gc<ShapeData>` to emit the barrier.
    pub transitions: RefLock<TransitionTable<'gc>>,

    /// Full ordered descriptor list (parent's prefix + this shape's
    /// last_key, if any). Eager rather than lazy — keeps slow paths
    /// branchless.
    pub descriptors: Box<[Descriptor<'gc>]>,
}

#[derive(Collect)]
#[collect(internal, require_static)]
pub struct MtCacheData {
    pub bits: Cell<MetamethodBits>,
}

impl MtCacheData {
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
    pub fn new(mc: &Mutation<'gc>, bits: MetamethodBits) -> Self {
        MtCache(Gc::new(
            mc,
            MtCacheData {
                bits: Cell::new(bits),
            },
        ))
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
    pub fn inner(self) -> Gc<'gc, MtCacheData> {
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
                transitions: RefLock::new(TransitionTable {
                    by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                    by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
                }),
                descriptors: Box::from([]),
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
                transitions: RefLock::new(TransitionTable {
                    by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                    by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
                }),
                descriptors: Box::from([]),
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

    #[inline]
    pub fn descriptors(self) -> &'gc [Descriptor<'gc>] {
        &self.data().descriptors
    }

    /// Look up a string key in this shape. Linear scan for very small
    /// shapes; binary search by `LuaString` pointer identity above the
    /// cutoff. Returns the slot index if present.
    pub fn find_slot(self, key: LuaString<'gc>) -> Option<u32> {
        let descs = self.descriptors();
        if descs.len() <= 8 {
            for d in descs {
                if d.key == key {
                    return Some(d.slot);
                }
            }
            return None;
        }
        let key_ptr = Gc::as_ptr(key.inner()) as usize;
        descs
            .binary_search_by_key(&key_ptr, |d| Gc::as_ptr(d.key.inner()) as usize)
            .ok()
            .map(|i| descs[i].slot)
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

    /// Returns `true` if the metatable has *any* metamethod set. Used
    /// to short-circuit fast paths that don't care which one.
    #[inline]
    pub fn has_any_mm(self) -> bool {
        match self.mt_cache() {
            None => false,
            Some(c) => !c.get().is_empty(),
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

    // Fast path: existing edge.
    {
        let table = parent.data().transitions.borrow();
        let key_ptr = Gc::as_ptr(key.inner()) as usize;
        let h = key_ptr as u64;
        if let Some(edge) = table.by_prop.find(h, |e| e.key == key) {
            if let Some(child) = edge.child.upgrade(mc) {
                return Shape(child);
            }
            // Stale weak edge: drop it on the slow path. We can't
            // remove during the immutable borrow; instead, fall through
            // and let the insert below replace it.
        }
    }

    // Slow path: allocate a child and install/replace the edge.
    let new_slot = parent.data().slot_count;
    let mut new_descs: Vec<Descriptor<'gc>> = parent.descriptors().to_vec();
    new_descs.push(Descriptor {
        key,
        slot: new_slot,
    });
    // Keep descriptors sorted by LuaString pointer identity so
    // `Shape::find_slot` can binary-search the larger ones.
    new_descs.sort_by_key(|d| Gc::as_ptr(d.key.inner()) as usize);

    let child_data = Gc::new(
        mc,
        ShapeData {
            parent: Some(parent),
            last_key: Some(key),
            slot_count: new_slot + 1,
            mt_cache: parent.data().mt_cache,
            is_dict: false,
            transitions: RefLock::new(TransitionTable {
                by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
            }),
            descriptors: new_descs.into_boxed_slice(),
        },
    );
    let child = Shape(child_data);

    {
        let parent_write = Gc::write(mc, parent.0);
        let mut table = unlock!(parent_write, ShapeData, transitions).borrow_mut();
        let key_ptr = Gc::as_ptr(key.inner()) as usize;
        let h = key_ptr as u64;
        let entry = table.by_prop.entry(
            h,
            |e| e.key == key,
            |e| Gc::as_ptr(e.key.inner()) as usize as u64,
        );
        match entry {
            hash_table::Entry::Occupied(mut o) => {
                // Replace stale weak.
                *o.get_mut() = PropEdge {
                    key,
                    child: Gc::downgrade(child_data),
                };
            }
            hash_table::Entry::Vacant(v) => {
                v.insert(PropEdge {
                    key,
                    child: Gc::downgrade(child_data),
                });
            }
        }
    }

    child
}

/// Switch the metatable on `parent`, returning a shape with the same
/// ordered key list but the new `mt_cache`. Caches transitions on
/// `parent` so repeated `setmetatable(t, mt)` calls share shapes.
pub fn transition_set_metatable<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    new_mt: Option<MtCache<'gc>>,
) -> Shape<'gc> {
    // Fast path: existing edge.
    {
        let table = parent.data().transitions.borrow();
        let h = mt_edge_hash(new_mt);
        if let Some(edge) = table.by_mt.find(h, |e| mt_edge_eq(e.mt_cache, new_mt)) {
            if let Some(child) = edge.child.upgrade(mc) {
                return Shape(child);
            }
        }
    }

    // Slow path: rebuild the same key list under the new mt_cache. We
    // walk the parent chain to collect ordered keys, then re-resolve
    // from the empty shape via `transition_add_prop`. This is O(slots)
    // and rare (setmetatable is not a hot path).
    let keys = collect_keys_in_order(parent);
    let empty = empty_shape_for(mc, new_mt);
    let mut shape = empty;
    for k in keys {
        shape = transition_add_prop(mc, shape, k);
    }

    // Install the edge on `parent` (not `empty`) so future
    // setmetatable calls from this same starting shape share.
    {
        let parent_write = Gc::write(mc, parent.0);
        let mut table = unlock!(parent_write, ShapeData, transitions).borrow_mut();
        let h = mt_edge_hash(new_mt);
        let entry = table.by_mt.entry(
            h,
            |e| mt_edge_eq(e.mt_cache, new_mt),
            |e| mt_edge_hash(e.mt_cache),
        );
        match entry {
            hash_table::Entry::Occupied(mut o) => {
                *o.get_mut() = MtEdge {
                    mt_cache: new_mt,
                    child: Gc::downgrade(shape.0),
                };
            }
            hash_table::Entry::Vacant(v) => {
                v.insert(MtEdge {
                    mt_cache: new_mt,
                    child: Gc::downgrade(shape.0),
                });
            }
        }
    }

    shape
}

fn collect_keys_in_order<'gc>(shape: Shape<'gc>) -> Vec<LuaString<'gc>> {
    // Walk up the parent chain to gather keys in insertion order.
    let mut keys: Vec<LuaString<'gc>> = Vec::with_capacity(shape.slot_count() as usize);
    let mut s = shape;
    loop {
        if let Some(k) = s.data().last_key {
            keys.push(k);
        }
        match s.data().parent {
            Some(p) => s = p,
            None => break,
        }
    }
    keys.reverse();
    keys
}

fn empty_shape_for<'gc>(mc: &Mutation<'gc>, mt_cache: Option<MtCache<'gc>>) -> Shape<'gc> {
    // For now we allocate a fresh empty-with-mt root each call; the
    // caller (typically `set_metatable`) caches the result via the
    // transition edge above. A future optimization can pre-register
    // these in `State`.
    Shape(Gc::new(
        mc,
        ShapeData {
            parent: None,
            last_key: None,
            slot_count: 0,
            mt_cache,
            is_dict: false,
            transitions: RefLock::new(TransitionTable {
                by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
            }),
            descriptors: Box::from([]),
        },
    ))
}

#[inline]
fn mt_edge_hash<'gc>(cache: Option<MtCache<'gc>>) -> u64 {
    match cache {
        Some(c) => Gc::as_ptr(c.inner()) as usize as u64,
        None => 0,
    }
}

#[inline]
fn mt_edge_eq<'gc>(a: Option<MtCache<'gc>>, b: Option<MtCache<'gc>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => MtCache::ptr_eq(x, y),
        _ => false,
    }
}
