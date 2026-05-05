//! Hidden classes (a.k.a. shapes) for Lua tables — tcvm's analog of V8
//! HiddenClasses / JSC Structures.
//!
//! A `Shape` is an immutable, GC-allocated descriptor identifying the
//! structural class of a `Table`: the ordered list of string-keyed
//! properties currently stored on the table, plus the identity of the
//! table's metatable. Two tables with the same shape have the same
//! storage layout (same string keys live at the same `properties`
//! slots) and share metamethod-presence information.
//!
//! Shapes form a transition tree: starting from `EMPTY_SHAPE`,
//! adding a string key transitions to a child shape; assigning a new
//! metatable transitions along a different edge. Shapes are deduped
//! via per-shape transition tables, so two tables that grow through
//! the same key sequence converge on the same shape pointer.
//!
//! See `/Users/joelwejdenstal/.claude/plans/in-a-previous-pr-validated-wind.md`
//! for the full plan; this module implements its Phase 1 + 5 skeleton.

use core::cell::Cell;

use bitflags::bitflags;
use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::{Collect, Gc, GcWeak, Lock, Mutation, RefLock};
use crate::env::string::LuaString;

/// V8-style "Map" / hidden class. Copy wrapper over a single Gc pointer
/// for cheap pass-by-value and pointer-equality identity checks.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Shape<'gc>(Gc<'gc, ShapeData<'gc>>);

/// Identity token for a metatable instance. Generation counter bumps
/// when the metatable is mutated on a metamethod-named key — the cached
/// `mm_cache` on shapes that observe this token must then be re-derived.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct MtToken<'gc>(Gc<'gc, MtTokenData>);

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

bitflags! {
    /// Per-metamethod presence cache. Computed lazily on slow paths
    /// from the metatable's contents at the shape's `mt_token`'s
    /// generation snapshot.
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct MetamethodBits: u32 {
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
///     mutation that affects metamethod presence; bumps `MtToken`
///     generation).
///   - `Shape::recompute_mm_cache` (read-side: walk the metatable
///     and OR together the bits for each present metamethod).
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

/// Cached metamethod bits + the `mt_token` generation those bits were
/// derived under. Slow paths refresh when the live generation has
/// advanced past `gen_at_compute`.
#[derive(Clone, Copy, Default)]
pub struct MmCache {
    pub bits: u32,
    /// `u32::MAX` sentinel = never computed; recompute on first read.
    pub gen_at_compute: u32,
}

const MM_CACHE_UNCOMPUTED: u32 = u32::MAX;

/// The transition table is a side-allocated Gc-managed map living off
/// each shape. Insertion goes through `borrow_mut(mc)` which emits the
/// backward barrier — children adopted as `GcWeak` won't retain their
/// targets, so a transient sub-shape can be reclaimed by GC even while
/// its parent is alive.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct TransitionTable<'gc> {
    /// Property-add edges keyed by the added LuaString.
    pub by_prop: HashTable<PropEdge<'gc>, MetricsAlloc<'gc>>,
    /// Set-metatable edges keyed by the new metatable token (None = no MT).
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
    pub mt_token: Option<MtToken<'gc>>,
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

    /// Identity of the metatable for tables of this shape. `None` = no
    /// metatable. Different metatables → different shapes.
    pub mt_token: Option<MtToken<'gc>>,

    /// True if this shape represents a table that's gone slow
    /// (dictionary mode). Only one dictionary shape per `mt_token` —
    /// see the per-`State` registry. Dictionary shapes have
    /// `slot_count = 0` and an empty `descriptors`.
    #[collect(require_static)]
    pub is_dict: bool,

    /// Cached metamethod bits + generation-at-compute. Mutable; reads
    /// without barrier, writes via `Gc::write` over the parent shape.
    pub mm_cache: Lock<MmCacheRepr>,

    /// Outgoing transition edges. Side-allocated `Gc<RefLock>` so we
    /// can mutate without rewriting the immutable shape body.
    pub transitions: Gc<'gc, RefLock<TransitionTable<'gc>>>,

    /// Full ordered descriptor list (parent's prefix + this shape's
    /// last_key, if any). Eager rather than lazy — keeps slow paths
    /// branchless.
    pub descriptors: Box<[Descriptor<'gc>]>,
}

/// Newtype wrapper for `Lock<T>` cooperation: `Lock<T>` requires
/// `T: Copy + Collect`. `MmCache` is two u32s; this repr makes that
/// trivially Copy + 'static.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MmCacheRepr {
    pub bits: u32,
    pub gen_at_compute: u32,
}

unsafe impl<'gc> Collect<'gc> for MmCacheRepr {
    const NEEDS_TRACE: bool = false;
    fn trace<T: crate::dmm::collect::Trace<'gc>>(&self, _cc: &mut T) {}
}

unsafe impl<'gc> Collect<'gc> for MetamethodBits {
    const NEEDS_TRACE: bool = false;
    fn trace<T: crate::dmm::collect::Trace<'gc>>(&self, _cc: &mut T) {}
}

impl MmCacheRepr {
    pub const fn uncomputed() -> Self {
        Self {
            bits: 0,
            gen_at_compute: MM_CACHE_UNCOMPUTED,
        }
    }
}

#[derive(Collect)]
#[collect(internal, require_static)]
pub struct MtTokenData {
    pub generation: Cell<u32>,
}

impl MtTokenData {
    #[inline]
    pub fn current_gen(&self) -> u32 {
        self.generation.get()
    }

    /// Bump the generation. Safe regardless of mutation context — the
    /// underlying type is `Cell<u32>` with no Gc adoption, so no write
    /// barrier is required (mutation only touches non-Gc state).
    #[inline]
    pub fn bump(&self) {
        let next = self.generation.get().wrapping_add(1);
        self.generation.set(next);
    }
}

impl<'gc> MtToken<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        MtToken(Gc::new(
            mc,
            MtTokenData {
                generation: Cell::new(0),
            },
        ))
    }

    #[inline]
    pub fn current_gen(self) -> u32 {
        self.0.current_gen()
    }

    #[inline]
    pub fn bump(self) {
        self.0.bump();
    }

    #[inline]
    pub fn inner(self) -> Gc<'gc, MtTokenData> {
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
        let transitions = Gc::new(
            mc,
            RefLock::new(TransitionTable {
                by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
            }),
        );
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                slot_count: 0,
                mt_token: None,
                is_dict: false,
                mm_cache: Lock::new(MmCacheRepr::uncomputed()),
                transitions,
                descriptors: Box::from([]),
            },
        ))
    }

    /// Allocate the dictionary-mode sentinel shape for a given metatable
    /// token. Held in `State`'s per-token registry.
    pub fn dict_sentinel(mc: &Mutation<'gc>, mt_token: Option<MtToken<'gc>>) -> Self {
        let transitions = Gc::new(
            mc,
            RefLock::new(TransitionTable {
                by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
                by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
            }),
        );
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                slot_count: 0,
                mt_token,
                is_dict: true,
                mm_cache: Lock::new(MmCacheRepr::uncomputed()),
                transitions,
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
    pub fn mt_token(self) -> Option<MtToken<'gc>> {
        self.data().mt_token
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

    /// Read the cached `MetamethodBits`. The caller must check
    /// `mm_cache_stale` against the current `mt_token` generation
    /// before trusting the result.
    #[inline]
    pub fn mm_cache(self) -> MmCacheRepr {
        self.data().mm_cache.get()
    }

    /// True if the cached `mm_cache` has never been filled or is older
    /// than the current `mt_token` generation. With no metatable the
    /// cache is trivially fresh (empty bits).
    #[inline]
    pub fn mm_cache_stale(self) -> bool {
        match self.mt_token() {
            None => false,
            Some(t) => {
                let cache = self.mm_cache();
                cache.gen_at_compute == MM_CACHE_UNCOMPUTED
                    || t.current_gen() != cache.gen_at_compute
            }
        }
    }

    /// Pessimistic metamethod-presence check used in interpreter fast
    /// paths: returns `true` if the shape *might* expose `bit`. Reads
    /// the cached bits; returns `true` if the cache is stale (forces
    /// the slow path to refresh). With no metatable, always `false`.
    #[inline]
    pub fn maybe_has_mm(self, bit: MetamethodBits) -> bool {
        if self.mt_token().is_none() {
            return false;
        }
        let cache = self.mm_cache();
        if cache.gen_at_compute == MM_CACHE_UNCOMPUTED {
            return true;
        }
        if let Some(t) = self.mt_token()
            && t.current_gen() != cache.gen_at_compute
        {
            return true;
        }
        (cache.bits & bit.bits()) != 0
    }

    /// Pessimistic check for *any* metamethod. Slightly cheaper than
    /// `maybe_has_mm(INDEX) || maybe_has_mm(NEWINDEX) || …`.
    #[inline]
    pub fn maybe_has_any_mm(self) -> bool {
        if self.mt_token().is_none() {
            return false;
        }
        let cache = self.mm_cache();
        if cache.gen_at_compute == MM_CACHE_UNCOMPUTED {
            return true;
        }
        if let Some(t) = self.mt_token()
            && t.current_gen() != cache.gen_at_compute
        {
            return true;
        }
        cache.bits != 0
    }

    /// Store freshly computed metamethod bits. Caller is responsible
    /// for having computed `bits` against `mt_gen` from the current
    /// metatable contents.
    #[inline]
    pub fn store_mm_cache(self, mc: &Mutation<'gc>, bits: MetamethodBits, mt_gen: u32) {
        // Lock<T> as a field needs to go through Gc::write to emit the
        // barrier. The contained Cell<MmCacheRepr> has no Gc, so the
        // barrier is precautionary.
        let write = Gc::write(mc, self.0);
        crate::dmm::barrier::unlock!(write, ShapeData, mm_cache).set(MmCacheRepr {
            bits: bits.bits(),
            gen_at_compute: mt_gen,
        });
    }

    /// Refresh `mm_cache` from the live metatable's contents. Reads
    /// each known metamethod name out of `metatable` (using identity
    /// equality on pre-interned `LuaString`s) and ORs the corresponding
    /// bits into the result, then stores at the live `mt_token`
    /// generation. Idempotent if already fresh.
    pub fn refresh_mm_cache(
        self,
        mc: &Mutation<'gc>,
        metatable: Option<crate::env::table::Table<'gc>>,
        metamethod_names: &[(crate::env::LuaString<'gc>, MetamethodBits)],
    ) {
        let token_gen = match self.mt_token() {
            None => 0,
            Some(t) => t.current_gen(),
        };
        let mut bits = MetamethodBits::empty();
        if let Some(mt) = metatable {
            for (name, bit) in metamethod_names {
                let v = mt.raw_get(crate::env::Value::string(*name));
                if !v.is_nil() {
                    bits |= *bit;
                }
            }
        }
        self.store_mm_cache(mc, bits, token_gen);
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

    let transitions = Gc::new(
        mc,
        RefLock::new(TransitionTable {
            by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
            by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
        }),
    );

    let child_data = Gc::new(
        mc,
        ShapeData {
            parent: Some(parent),
            last_key: Some(key),
            slot_count: new_slot + 1,
            mt_token: parent.data().mt_token,
            is_dict: false,
            mm_cache: Lock::new(MmCacheRepr::uncomputed()),
            transitions,
            descriptors: new_descs.into_boxed_slice(),
        },
    );
    let child = Shape(child_data);

    {
        let mut table = parent.data().transitions.borrow_mut(mc);
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
/// ordered key list but the new `mt_token`. Caches transitions on
/// `parent` so repeated `setmetatable(t, mt)` calls share shapes.
pub fn transition_set_metatable<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    new_mt: Option<MtToken<'gc>>,
) -> Shape<'gc> {
    // Fast path: existing edge.
    {
        let table = parent.data().transitions.borrow();
        let h = mt_edge_hash(new_mt);
        if let Some(edge) = table.by_mt.find(h, |e| mt_edge_eq(e.mt_token, new_mt)) {
            if let Some(child) = edge.child.upgrade(mc) {
                return Shape(child);
            }
        }
    }

    // Slow path: rebuild the same key list under the new mt_token. We
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
        let mut table = parent.data().transitions.borrow_mut(mc);
        let h = mt_edge_hash(new_mt);
        let entry = table.by_mt.entry(
            h,
            |e| mt_edge_eq(e.mt_token, new_mt),
            |e| mt_edge_hash(e.mt_token),
        );
        match entry {
            hash_table::Entry::Occupied(mut o) => {
                *o.get_mut() = MtEdge {
                    mt_token: new_mt,
                    child: Gc::downgrade(shape.0),
                };
            }
            hash_table::Entry::Vacant(v) => {
                v.insert(MtEdge {
                    mt_token: new_mt,
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

fn empty_shape_for<'gc>(mc: &Mutation<'gc>, mt_token: Option<MtToken<'gc>>) -> Shape<'gc> {
    // For now we allocate a fresh empty-with-mt root each call; the
    // caller (typically `set_metatable`) caches the result via the
    // transition edge above. A future optimization can pre-register
    // these in `State`.
    let transitions = Gc::new(
        mc,
        RefLock::new(TransitionTable {
            by_prop: HashTable::new_in(MetricsAlloc::new(mc)),
            by_mt: HashTable::new_in(MetricsAlloc::new(mc)),
        }),
    );
    Shape(Gc::new(
        mc,
        ShapeData {
            parent: None,
            last_key: None,
            slot_count: 0,
            mt_token,
            is_dict: false,
            mm_cache: Lock::new(MmCacheRepr::uncomputed()),
            transitions,
            descriptors: Box::from([]),
        },
    ))
}

#[inline]
fn mt_edge_hash<'gc>(token: Option<MtToken<'gc>>) -> u64 {
    match token {
        Some(t) => Gc::as_ptr(t.inner()) as usize as u64,
        None => 0,
    }
}

#[inline]
fn mt_edge_eq<'gc>(a: Option<MtToken<'gc>>, b: Option<MtToken<'gc>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => MtToken::ptr_eq(x, y),
        _ => false,
    }
}
