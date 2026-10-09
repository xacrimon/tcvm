//! Hidden classes (a.k.a. shapes) for Lua tables — tcvm's analog of V8
//! HiddenClasses / JSC Structures.
//!
//! A `Shape` is an immutable, GC-allocated descriptor identifying the
//! structural class of a `Table`: the ordered list of string-keyed
//! properties currently stored on the table, plus the class of the
//! table's metatable. Two tables with the same shape have the same
//! storage layout (same string keys live at the same `properties`
//! slots) and observe the same metamethods (via the shared `MtCache`).
//! Metatables with the same metamethods share one class, so a table keeps
//! its metatable's identity itself when the class has several (a poly
//! shape, see [`ShapeData::mt_slot`]).
//!
//! Shapes form a transition tree: starting from `EMPTY_SHAPE`,
//! adding a string key transitions to a child shape; assigning a new
//! metatable transitions along a different edge. Shapes are deduped
//! via per-shape transition tables, so two tables that grow through
//! the same key sequence converge on the same shape pointer.

use core::cell::{Cell, UnsafeCell};
use core::mem::MaybeUninit;
use core::ops::Not;
use core::ptr::NonNull;

use bitflags::bitflags;
use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::barrier::unlock;
use crate::dmm::{Collect, Gc, GcWeak, Lock, Mutation, RefLock, Trace};
use crate::env::for_each_metamethod;
use crate::env::string::LuaString;
use crate::env::symbols::METAMETHOD_COUNT;
use crate::env::table::Table;
use crate::env::value::Value;

/// V8-style "Map" / hidden class. Copy wrapper over a single Gc pointer
/// for cheap pass-by-value and pointer-equality identity checks.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Shape<'gc>(Gc<'gc, ShapeData<'gc>>);

/// A metatable class: the metamethods (and `__mode`) its member metatables
/// all have. A table adopted as a metatable joins the class with its
/// metamethods from [`MtClasses`], so `setmetatable(o, {__index = C})` per
/// object shares one class and one shape per key set. Identity (the `Gc`
/// address) is what metatable edges key transitions on. A write to a
/// member's metamethods updates the class in place when it has one member,
/// and otherwise retires it ([`MtCache::make_stale`]).
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

/// The address of `v`'s table, or 0.
fn table_addr(v: Value<'_>) -> usize {
    v.get_table().map_or(0, |t| Gc::as_ptr(t.inner()) as usize)
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

/// The inline capacities a table can be allocated with, one root shape each:
/// a table's first `inline_cap` slots live in its own cell, the rest in a
/// spill cell. Few sizes, so tables built with nearby hints share shapes.
pub const INLINE_CAPS: [u8; 6] = [0, 2, 4, 8, 16, 32];

/// Index into [`INLINE_CAPS`] of the smallest capacity holding `n` slots, or
/// of the largest.
pub fn inline_bucket(n: usize) -> usize {
    INLINE_CAPS
        .iter()
        .position(|&c| c as usize >= n)
        .unwrap_or(INLINE_CAPS.len() - 1)
}

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

/// Whether a store of `key` to a metatable changes its [`MtCache`].
pub fn mirrored(key: LuaString<'_>) -> bool {
    let name = key.as_bytes();
    name == b"__mode" || metamethod_bit_of_bytes(name).is_some()
}

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

/// What a transition adds: a string key, a metatable class (`None`
/// removes it) and whether the child is poly, or the class tables of the
/// child are members of as metatables. Untraced in an [`Edge`], as a live
/// child keeps it alive through `last_key`, `mt_cache` or `as_mt`; so it may
/// be dead, and is compared and hashed by address.
#[derive(Clone, Copy)]
enum EdgeKey<'gc> {
    Prop(LuaString<'gc>),
    Mt(Option<MtCache<'gc>>, bool),
    Adopt(MtCache<'gc>),
}

impl<'gc> EdgeKey<'gc> {
    /// The key of the edge leading to `child`.
    fn of(child: &ShapeData<'gc>) -> Self {
        match (child.last_key, child.as_mt) {
            (Some(k), _) => EdgeKey::Prop(k),
            (None, Some(c)) if child.adopt_edge => EdgeKey::Adopt(c),
            (None, _) => EdgeKey::Mt(child.mt_cache, child.mt_slot != MT_IN_CLASS),
        }
    }

    fn addr(self) -> usize {
        match self {
            EdgeKey::Prop(k) => addr(k),
            // A class is word-aligned, so the low bit is free for poly.
            EdgeKey::Mt(c, poly) => c.map_or(0, |c| Gc::as_ptr(c.inner()) as usize) | poly as usize,
            EdgeKey::Adopt(c) => Gc::as_ptr(c.inner()) as usize,
        }
    }

    fn same(self, other: Self) -> bool {
        let kinds = matches!(
            (self, other),
            (EdgeKey::Prop(_), EdgeKey::Prop(_))
                | (EdgeKey::Mt(..), EdgeKey::Mt(..))
                | (EdgeKey::Adopt(_), EdgeKey::Adopt(_))
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
    /// for shapes reached via a metatable or adoption transition.
    pub last_key: Option<LuaString<'gc>>,

    /// Whether this shape was reached by [`transition_adopt`].
    #[collect(require_static)]
    adopt_edge: bool,

    /// Number of string-keyed slots this shape covers. Slot N lives at
    /// `TableState::properties[N]`. Append-only along property
    /// transitions; preserved across metatable transitions.
    pub slot_count: u32,

    /// The metatable's class: its metamethods and weak mode. `None` = no
    /// metatable. Transitions go through `transition_set_metatable`.
    pub mt_cache: Option<MtCache<'gc>>,

    /// Where a table of this shape keeps its metatable: [`MT_IN_CLASS`] for
    /// the class's owner, [`MT_IN_AUX`] for a dict-mode table's aux cell, else
    /// the named slot (poly), whose key is `Symbols::mt_key`. A shape stays
    /// poly along every later transition.
    #[collect(require_static)]
    pub mt_slot: u32,

    /// The class tables of this shape are members of as metatables, `None`
    /// until one is adopted (a dict-mode one keeps it in its aux cell). Kept
    /// along every later transition, so ICs on a metatable's shape know it
    /// is one.
    pub as_mt: Option<MtCache<'gc>>,

    /// The metamethods among this shape's keys, and whether `__mode` is one:
    /// the keys a metatable's class mirrors.
    #[collect(require_static)]
    mirrored: MetamethodBits,
    #[collect(require_static)]
    has_mode: bool,

    /// The adopted shape a table of this one moved to last, as a member of
    /// a shared class, which a table with the same metamethods joins.
    adopt_memo: Lock<Option<Shape<'gc>>>,

    /// For a metatable of this (adopted) shape: the last move `setmetatable`
    /// made of a table without a metatable, `(from, to)`, which a table in
    /// `from` repeats by storing `to` (and the metatable, `to` being poly).
    mt_move: Lock<Option<(Shape<'gc>, Shape<'gc>)>>,

    /// True if this shape represents a table that's gone slow
    /// (dictionary mode). Only one dictionary shape per `mt_cache` —
    /// see the per-`State` registry. Dictionary shapes have
    /// `slot_count = 0` and no keys.
    #[collect(require_static)]
    pub is_dict: bool,

    /// How many of the slots live in the table's own cell (see
    /// [`INLINE_CAPS`]); fixed for a whole tree, from its root.
    #[collect(require_static)]
    pub inline_cap: u8,

    /// Outgoing transition edges, held inline (no separate `Gc`
    /// allocation). Mutation goes through `Gc::write` on the parent
    /// `Gc<ShapeData>` to emit the barrier.
    pub transitions: RefLock<TransitionTable<'gc>>,

    /// This shape's keys are the first `slot_count`, key `i` in slot `i`.
    keys: Gc<'gc, Keys<'gc>>,
}

/// [`ShapeData::mt_slot`] of a mono shape: the metatable is the class's owner.
pub const MT_IN_CLASS: u32 = u32::MAX;
/// [`ShapeData::mt_slot`] of a poly dict sentinel: the metatable is in the
/// table's aux cell.
pub const MT_IN_AUX: u32 = u32::MAX - 1;

/// Where a table keeps its metatable (see [`ShapeData::mt_slot`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MtHome {
    Class,
    Slot(u32),
    Aux,
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
    /// The metatable that made this class, the metatable of every table in a
    /// mono shape of it. Untraced so that a shape held by an IC doesn't keep
    /// it alive: only those tables read it, and each of them marks it. Once
    /// dead only its address is compared, and a metatable reusing the address
    /// with these metamethods may as well be the owner.
    #[collect(require_static)]
    owner: NonNull<()>,
    /// Metatables that joined, the owner included; never decremented, so
    /// more than one means members other than the owner may be alive.
    #[collect(require_static)]
    members: Cell<u32>,
    /// Retired by a member's write (see [`MtCache::make_stale`]).
    #[collect(require_static)]
    stale: Cell<bool>,
    /// The address of the `TableState` of the member whose write retired
    /// this class, 0 before. Every member was alive at that write, so no
    /// other member shares the address.
    #[collect(require_static)]
    retired_by: Cell<usize>,
    /// Not in `MtClasses`: no other metatable joins it.
    #[collect(require_static)]
    private: bool,
    #[collect(require_static)]
    pub bits: Cell<MetamethodBits>,
    /// Read by the collector when it traces a table with this metatable.
    #[collect(require_static)]
    pub weak: Cell<WeakMode>,
    /// `__index` and `__newindex`, kept like `bits` (weak clears included),
    /// inline as every table access may need them.
    index: MmValue<'gc>,
    newindex: MmValue<'gc>,
    /// The other metamethods by bit index, allocated once the metatable has
    /// one: a metatable made per object (`{__index = C}`) is adopted per
    /// object, and a cache with all of them inline was five times the size.
    rest: Lock<Option<Gc<'gc, MmValues<'gc>>>>,
    /// Lazily-allocated dict-mode sentinels for tables that drop into
    /// dict mode while carrying this class, mono and poly. Populated on the
    /// first call to `MtCache::ensure_dict_sentinel`; subsequent calls return
    /// the same `Shape` pointer so dict-mode tables sharing a class also
    /// share a shape.
    pub dict_sentinel: Lock<Option<Shape<'gc>>>,
    pub dict_sentinel_poly: Lock<Option<Shape<'gc>>>,
}

/// A metamethod in [`MtCacheData`]. Untraced: every member metatable holds
/// it, and a shape carrying the class is only read through a table that
/// keeps its own member alive. A stale class holds nil.
pub struct MmValue<'gc>(Cell<Value<'gc>>);

unsafe impl<'gc> Collect<'gc> for MmValue<'gc> {
    const NEEDS_TRACE: bool = false;
}

/// A metamethod's bit index, below `METAMETHOD_COUNT`.
#[derive(Clone, Copy)]
pub struct MmIndex(u8);

impl MmIndex {
    pub const fn of(bit: MetamethodBits) -> Self {
        assert!(bit.bits().count_ones() == 1);
        MmIndex(bit.bits().trailing_zeros() as u8)
    }
}

/// [`MtCacheData::rest`], untraced like [`MmValue`].
pub struct MmValues<'gc>([Cell<Value<'gc>>; METAMETHOD_COUNT]);

unsafe impl<'gc> Collect<'gc> for MmValues<'gc> {
    const NEEDS_TRACE: bool = false;
}

const INDEX_IDX: usize = MetamethodBits::INDEX.bits().trailing_zeros() as usize;
const NEWINDEX_IDX: usize = MetamethodBits::NEWINDEX.bits().trailing_zeros() as usize;

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

/// The metamethod bits `values` (by bit index) make.
fn bits_of(values: &[Value<'_>; METAMETHOD_COUNT]) -> MetamethodBits {
    let mut bits = MetamethodBits::empty();
    for (i, v) in values.iter().enumerate() {
        if !v.is_nil() {
            bits |= MetamethodBits::from_bits_retain(1 << i);
        }
    }
    bits
}

/// A hash of a class's content, its values by their bits.
fn content_hash(
    bits: MetamethodBits,
    weak: WeakMode,
    values: &[Value<'_>; METAMETHOD_COUNT],
) -> u64 {
    let mut h = addr_hash(bits.bits() as usize ^ (weak.bits() as usize) << 32);
    for v in values.iter().filter(|v| !v.is_nil()) {
        h = addr_hash(h as usize ^ v.to_raw() as usize);
    }
    h
}

impl<'gc> MtCache<'gc> {
    /// A class of one member, `owner`, whose metamethods by bit index are
    /// `values`.
    pub fn new(
        mc: &Mutation<'gc>,
        owner: Table<'gc>,
        values: [Value<'gc>; METAMETHOD_COUNT],
        weak: WeakMode,
        private: bool,
    ) -> Self {
        let bits = bits_of(&values);
        let rest = (bits & !(MetamethodBits::INDEX | MetamethodBits::NEWINDEX))
            .is_empty()
            .not()
            .then(|| Gc::new(mc, MmValues(values.map(Cell::new))));
        MtCache(Gc::new(
            mc,
            MtCacheData {
                // SAFETY: a `Gc`'s pointer is never null.
                owner: unsafe { NonNull::new_unchecked(Gc::as_ptr(owner.inner()).cast_mut()) }
                    .cast(),
                members: Cell::new(1),
                stale: Cell::new(false),
                retired_by: Cell::new(0),
                private,
                bits: Cell::new(bits),
                weak: Cell::new(weak),
                index: MmValue(Cell::new(values[INDEX_IDX])),
                newindex: MmValue(Cell::new(values[NEWINDEX_IDX])),
                rest: Lock::new(rest),
                dict_sentinel: Lock::new(None),
                dict_sentinel_poly: Lock::new(None),
            },
        ))
    }

    /// Lazily allocate the dict-mode sentinel shape that all tables
    /// carrying this class share once they migrate to dict mode, the poly
    /// one keeping the metatable in the aux cell.
    #[inline]
    pub fn ensure_dict_sentinel(self, mc: &Mutation<'gc>, poly: bool) -> Shape<'gc> {
        let slot = match poly {
            false => &self.0.dict_sentinel,
            true => &self.0.dict_sentinel_poly,
        };
        if let Some(s) = slot.get() {
            return s;
        }
        let new_shape = Shape::dict_sentinel(mc, Some(self), poly);
        // We're adopting a fresh `Shape` Gc through the `Lock<Option<Shape>>`
        // field, so emit the backward barrier on the parent MtCacheData
        // before writing through `as_cell()`.
        mc.backward_barrier(Gc::erase(self.0), None);
        unsafe { slot.as_cell() }.set(Some(new_shape));
        new_shape
    }

    /// Whether `t` made this class (see [`MtCacheData::owner`]).
    #[inline]
    pub fn owned_by(self, t: Table<'gc>) -> bool {
        self.0.owner.as_ptr().cast_const() == Gc::as_ptr(t.inner()).cast()
    }

    #[inline]
    pub fn is_stale(self) -> bool {
        self.0.stale.get()
    }

    #[inline]
    pub fn is_private(self) -> bool {
        self.0.private
    }

    /// Whether the member whose `TableState` is at `member` retired this
    /// class (see [`MtCacheData::retired_by`]).
    #[inline]
    pub fn retired_by(self, member: usize) -> bool {
        self.0.retired_by.get() == member
    }

    /// Count another metatable with this content as a member.
    pub fn join(self) {
        self.0.members.set(self.0.members.get().saturating_add(1));
    }

    /// Retire this class, a member's metamethods having left it: every guard
    /// reading it fails (all bits, no `__index` or `__newindex`, no others),
    /// so a table of it reaches a slow path, which moves the table to its
    /// metatable's current class ([`Table::meta`]). The weak mode stays, as
    /// changing `__mode` in use is undefined (§2.5.4). `by` is the writer's
    /// `TableState` address, which gets classes of its own from then on, so
    /// a metatable that keeps changing can't retire one class after another
    /// under the tables of the other members.
    fn make_stale(self, by: usize) {
        self.0.stale.set(true);
        self.0.retired_by.set(by);
        self.0.bits.set(MetamethodBits::all());
        self.0.index.0.set(Value::nil());
        self.0.newindex.0.set(Value::nil());
        // SAFETY: storing `None` adopts nothing.
        unsafe { self.0.rest.as_cell() }.set(None);
    }

    /// Apply the write of `key` by the member whose `TableState` is at
    /// `member` to the class, which the member's metamethods define: in
    /// place for its only member, else the class is retired. `mc` is `None`
    /// for a weak clear.
    pub fn member_write(
        self,
        mc: Option<&Mutation<'gc>>,
        key: LuaString<'gc>,
        value: Value<'gc>,
        member: usize,
    ) {
        if self.is_stale() {
            return;
        }
        let unchanged = match metamethod_bit_of_bytes(key.as_bytes()) {
            Some(bit) => self.mm(bit).same_bits(&value),
            None if key.as_bytes() == b"__mode" => self.weak() == WeakMode::of(value),
            None => return,
        };
        if unchanged {
            return;
        }
        if self.0.members.get() > 1 {
            self.make_stale(member);
            return;
        }
        self.mirror(mc, key, value);
    }

    /// Whether a metatable may join this class whose mode is `weak` and
    /// whose metamethods, absent outside `named`, `value` gives by bit index
    /// (nil for absent).
    #[inline(always)]
    pub fn admits(
        self,
        named: MetamethodBits,
        value: impl Fn(u32) -> Value<'gc>,
        weak: WeakMode,
    ) -> bool {
        if self.is_stale() || self.is_private() || self.weak() != weak {
            return false;
        }
        let mut rest = named.bits();
        let mut bits = MetamethodBits::empty();
        while rest != 0 {
            let i = rest.trailing_zeros();
            rest &= rest - 1;
            let v = value(i);
            if v.is_nil() {
                continue;
            }
            let bit = MetamethodBits::from_bits_retain(1 << i);
            if !self.mm(bit).same_bits(&v) {
                return false;
            }
            bits |= bit;
        }
        bits == self.get()
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

    /// The metamethod `bit` names, nil when absent.
    #[inline(always)]
    pub fn mm(self, bit: MetamethodBits) -> Value<'gc> {
        if bit == MetamethodBits::INDEX {
            self.0.index.0.get()
        } else if bit == MetamethodBits::NEWINDEX {
            self.0.newindex.0.get()
        } else {
            match self.0.rest.get() {
                Some(rest) => rest.0[bit.bits().trailing_zeros() as usize].get(),
                None => Value::nil(),
            }
        }
    }

    /// [`Self::mm`] by the metamethod's bit index, for one other than
    /// `__index` and `__newindex`.
    #[inline(always)]
    pub fn mm_at(self, idx: MmIndex) -> Value<'gc> {
        debug_assert!(idx.0 as usize != INDEX_IDX && idx.0 as usize != NEWINDEX_IDX);
        match self.0.rest.get() {
            // SAFETY: an `MmIndex` is in bounds.
            Some(rest) => unsafe { rest.0.get_unchecked(idx.0 as usize) }.get(),
            None => Value::nil(),
        }
    }

    /// Address of the table `__index` names, else 0: it only identifies.
    #[inline]
    pub fn index_table(self) -> usize {
        table_addr(self.0.index.0.get())
    }

    /// The metatable that made this class.
    ///
    /// # Safety
    ///
    /// Only for the class of a reachable table in a mono shape, as that table
    /// keeps it alive (see `MtCacheData::owner`).
    #[inline]
    pub unsafe fn owner(self) -> Table<'gc> {
        Table::from_inner(unsafe { Gc::from_ptr(self.0.owner.cast().as_ptr()) })
    }

    /// Mirror a write of `key` to this cache's metatable into its bits and
    /// weak mode; only keys [`mirrored`] names change anything. `mc` may be
    /// `None` for a store of nil (a weak clear), which allocates nothing.
    #[inline]
    pub fn mirror(self, mc: Option<&Mutation<'gc>>, key: LuaString<'gc>, value: Value<'gc>) {
        match metamethod_bit_of_bytes(key.as_bytes()) {
            Some(bit) => {
                if bit == MetamethodBits::INDEX {
                    self.0.index.0.set(value);
                } else if bit == MetamethodBits::NEWINDEX {
                    self.0.newindex.0.set(value);
                } else {
                    let rest = match self.0.rest.get() {
                        Some(rest) => Some(rest),
                        None if value.is_nil() => None,
                        None => {
                            let mc = mc.expect("a metamethod stored without a mutation");
                            let rest = Gc::new(
                                mc,
                                MmValues(std::array::from_fn(|_| Cell::new(Value::nil()))),
                            );
                            // Adopting a fresh `Gc` through the lock: barrier
                            // first, as `ensure_dict_sentinel` does.
                            mc.backward_barrier(Gc::erase(self.0), None);
                            unsafe { self.0.rest.as_cell() }.set(Some(rest));
                            Some(rest)
                        }
                    };
                    if let Some(rest) = rest {
                        rest.0[bit.bits().trailing_zeros() as usize].set(value);
                    }
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
    /// Allocate an empty root shape for tables with `inline_cap` inline
    /// slots. There is one per capacity per `State`, see `State::root_shapes`.
    pub fn root_empty(mc: &Mutation<'gc>, inline_cap: u8) -> Self {
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                adopt_edge: false,
                slot_count: 0,
                mt_cache: None,
                mt_slot: MT_IN_CLASS,
                as_mt: None,
                mirrored: MetamethodBits::empty(),
                has_mode: false,
                adopt_memo: Lock::new(None),
                mt_move: Lock::new(None),
                is_dict: false,
                inline_cap,
                transitions: RefLock::new(TransitionTable::new()),
                keys: Keys::new(mc, &[], 0),
            },
        ))
    }

    /// Allocate the dictionary-mode sentinel shape for a given metatable
    /// class, held by the class (or `State`, for none); a poly one keeps
    /// the metatable in the aux cell.
    pub fn dict_sentinel(mc: &Mutation<'gc>, mt_cache: Option<MtCache<'gc>>, poly: bool) -> Self {
        Shape(Gc::new(
            mc,
            ShapeData {
                parent: None,
                last_key: None,
                adopt_edge: false,
                slot_count: 0,
                mt_cache,
                mt_slot: if poly { MT_IN_AUX } else { MT_IN_CLASS },
                as_mt: None,
                mirrored: MetamethodBits::empty(),
                has_mode: false,
                adopt_memo: Lock::new(None),
                mt_move: Lock::new(None),
                is_dict: true,
                inline_cap: 0,
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
    pub fn inline_cap(self) -> u32 {
        self.data().inline_cap as u32
    }

    #[inline]
    pub fn mt_cache(self) -> Option<MtCache<'gc>> {
        self.data().mt_cache
    }

    /// Where a table of this shape keeps its metatable.
    #[inline]
    pub fn mt_home(self) -> MtHome {
        match self.data().mt_slot {
            MT_IN_CLASS => MtHome::Class,
            MT_IN_AUX => MtHome::Aux,
            slot => MtHome::Slot(slot),
        }
    }

    /// Whether a table of this shape keeps its metatable itself.
    #[inline]
    pub fn is_poly(self) -> bool {
        self.data().mt_slot != MT_IN_CLASS
    }

    /// See [`ShapeData::as_mt`].
    #[inline]
    pub fn as_mt(self) -> Option<MtCache<'gc>> {
        self.data().as_mt
    }

    /// The metamethods among this shape's keys, and whether `__mode` is one.
    #[inline]
    pub fn mirrored(self) -> (MetamethodBits, bool) {
        (self.data().mirrored, self.data().has_mode)
    }

    /// See [`ShapeData::adopt_memo`].
    #[inline]
    pub fn adopt_memo(self) -> Option<Shape<'gc>> {
        self.data().adopt_memo.get()
    }

    pub fn set_adopt_memo(self, mc: &Mutation<'gc>, to: Shape<'gc>) {
        let memo = &self.data().adopt_memo;
        if memo.get().is_some_and(|m| Shape::ptr_eq(m, to)) {
            return;
        }
        // Adopting a `Gc` through the lock, as `MtCache::ensure_dict_sentinel` does.
        mc.backward_barrier(Gc::erase(self.0), None);
        unsafe { memo.as_cell() }.set(Some(to));
    }

    /// See [`ShapeData::mt_move`].
    #[inline]
    pub fn mt_move(self) -> Option<(Shape<'gc>, Shape<'gc>)> {
        self.data().mt_move.get()
    }

    pub fn set_mt_move(self, mc: &Mutation<'gc>, from: Shape<'gc>, to: Shape<'gc>) {
        let memo = &self.data().mt_move;
        if memo
            .get()
            .is_some_and(|(f, t)| Shape::ptr_eq(f, from) && Shape::ptr_eq(t, to))
        {
            return;
        }
        mc.backward_barrier(Gc::erase(self.0), None);
        unsafe { memo.as_cell() }.set(Some((from, to)));
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
        // By address, as `Keys::find` does: `Symbols::mt_key` shares its bytes
        // with an interned string a program can make.
        let a = addr(key);
        self.keys()
            .iter()
            .position(|&k| addr(k) == a)
            .map(|i| i as u32)
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

/// `parent`'s keys with `key` in the next slot: its own array when that
/// ends at `parent`'s prefix and has room, else a copy.
fn keys_with<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    key: LuaString<'gc>,
) -> Gc<'gc, Keys<'gc>> {
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
    keys
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
    let p = parent.data();
    let (mut mirrored, mut has_mode) = (p.mirrored, p.has_mode);
    match metamethod_bit_of_bytes(key.as_bytes()) {
        Some(bit) => mirrored |= bit,
        None => has_mode |= key.as_bytes() == b"__mode",
    }
    let child_data = Gc::new(
        mc,
        ShapeData {
            parent: Some(parent),
            last_key: Some(key),
            adopt_edge: false,
            slot_count: p.slot_count + 1,
            mt_cache: p.mt_cache,
            mt_slot: p.mt_slot,
            as_mt: p.as_mt,
            mirrored,
            has_mode,
            adopt_memo: Lock::new(None),
            mt_move: Lock::new(None),
            is_dict: false,
            inline_cap: p.inline_cap,
            transitions: RefLock::new(TransitionTable::new()),
            keys: keys_with(mc, parent, key),
        },
    );
    let parent_write = Gc::write(mc, parent.0);
    unlock!(parent_write, ShapeData, transitions)
        .borrow_mut()
        .insert(mc, child_data);
    Shape(child_data)
}

/// Switch the metatable class on `parent`, returning a shape with the same
/// ordered key list but `new_mt`. Caches transitions on `parent` so repeated
/// `setmetatable(t, mt)` calls share shapes. A `poly` child of a mono parent
/// adds the slot that keeps the metatable, keyed `mt_key`; a poly parent's
/// children stay poly in the same slot.
///
/// For dict-mode parents the call routes to the class's dict sentinel
/// (`MtCache::ensure_dict_sentinel`) or, when stripping the metatable, to
/// `State::empty_dict_sentinel` provided by the caller.
pub fn transition_set_metatable<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    new_mt: Option<MtCache<'gc>>,
    poly: bool,
    mt_key: LuaString<'gc>,
    no_mt_dict_sentinel: Shape<'gc>,
) -> Shape<'gc> {
    debug_assert!(poly || !parent.is_poly());
    // Dict-mode parent: never go through the prop-transition tree.
    // Route to the unique dict sentinel for the new class.
    if parent.is_dict() {
        return match new_mt {
            Some(c) => c.ensure_dict_sentinel(mc, poly),
            None => no_mt_dict_sentinel,
        };
    }

    if let Some(child) = parent
        .data()
        .transitions
        .borrow()
        .get(mc, EdgeKey::Mt(new_mt, poly))
    {
        return child;
    }

    // Slow path: a sibling with `parent`'s keys and the new class. Future
    // prop additions on the result mint their own edges normally.
    let p = parent.data();
    let (slot_count, mt_slot, keys) = match poly && !parent.is_poly() {
        true => (
            p.slot_count + 1,
            p.slot_count,
            keys_with(mc, parent, mt_key),
        ),
        false => (p.slot_count, p.mt_slot, p.keys),
    };
    let child_data = Gc::new(
        mc,
        ShapeData {
            parent: Some(parent),
            last_key: None,
            adopt_edge: false,
            slot_count,
            mt_cache: new_mt,
            mt_slot,
            as_mt: p.as_mt,
            mirrored: p.mirrored,
            has_mode: p.has_mode,
            adopt_memo: Lock::new(None),
            mt_move: Lock::new(None),
            is_dict: false,
            inline_cap: p.inline_cap,
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

/// `parent` for tables that are members of `class` as metatables: the same
/// keys and metatable. A shared class is remembered as `parent`'s
/// [`ShapeData::adopt_memo`].
pub fn transition_adopt<'gc>(
    mc: &Mutation<'gc>,
    parent: Shape<'gc>,
    class: MtCache<'gc>,
) -> Shape<'gc> {
    debug_assert!(!parent.is_dict());
    let edge = parent
        .data()
        .transitions
        .borrow()
        .get(mc, EdgeKey::Adopt(class));
    let child = match edge {
        Some(child) => child,
        None => {
            let p = parent.data();
            let child_data = Gc::new(
                mc,
                ShapeData {
                    parent: Some(parent),
                    last_key: None,
                    adopt_edge: true,
                    slot_count: p.slot_count,
                    mt_cache: p.mt_cache,
                    mt_slot: p.mt_slot,
                    as_mt: Some(class),
                    mirrored: p.mirrored,
                    has_mode: p.has_mode,
                    adopt_memo: Lock::new(None),
                    mt_move: Lock::new(None),
                    is_dict: false,
                    inline_cap: p.inline_cap,
                    transitions: RefLock::new(TransitionTable::new()),
                    keys: p.keys,
                },
            );
            let parent_write = Gc::write(mc, parent.0);
            unlock!(parent_write, ShapeData, transitions)
                .borrow_mut()
                .insert(mc, child_data);
            Shape(child_data)
        }
    };
    if !class.is_private() {
        parent.set_adopt_memo(mc, child);
    }
    child
}

/// The metatable classes of one Lua instance by content, so a table adopted
/// as a metatable joins the class with its metamethods. Weak, and only a
/// lookup aid: an entry whose class died, went stale, or changed in place
/// just stops matching.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct MtClasses<'gc>(Gc<'gc, RefLock<ClassTable<'gc>>>);

struct ClassTable<'gc>(UnsafeCell<HashTable<ClassEntry<'gc>, MetricsAlloc<'gc>>>);

#[derive(Clone, Copy)]
struct ClassEntry<'gc> {
    hash: u64,
    class: GcWeak<'gc, MtCacheData<'gc>>,
}

// SAFETY: traces every class weakly and drops the dead ones, as
// `TransitionTable` does.
unsafe impl<'gc> Collect<'gc> for ClassTable<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        let t = unsafe { &mut *self.0.get() };
        t.retain(|e| !e.class.is_dropped());
        for e in t.iter() {
            cc.trace(&e.class);
        }
    }
}

impl<'gc> MtClasses<'gc> {
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let t = HashTable::new_in(MetricsAlloc::new(mc));
        MtClasses(Gc::new(mc, RefLock::new(ClassTable(UnsafeCell::new(t)))))
    }

    /// The class for `mt`, whose metamethods by bit index are `values`: the
    /// registered one with that content, joined, or a new one `mt` owns. A
    /// metatable with none, or `private` (its writes have retired a class
    /// before), gets a class of its own that nothing else joins.
    pub fn adopt(
        self,
        mc: &Mutation<'gc>,
        mt: Table<'gc>,
        values: [Value<'gc>; METAMETHOD_COUNT],
        weak: WeakMode,
        private: bool,
    ) -> MtCache<'gc> {
        let bits = bits_of(&values);
        if private || (bits.is_empty() && weak.is_empty()) {
            return MtCache::new(mc, mt, values, weak, true);
        }
        let hash = content_hash(bits, weak, &values);
        let mut cell = self.0.borrow_mut(mc);
        let table = cell.0.get_mut();
        let found = table
            .find(hash, |e| {
                e.hash == hash
                    && e.class
                        .upgrade(mc)
                        .is_some_and(|c| MtCache(c).admits(bits, |i| values[i as usize], weak))
            })
            .and_then(|e| e.class.upgrade(mc));
        if let Some(c) = found {
            let c = MtCache(c);
            c.join();
            return c;
        }
        let c = MtCache::new(mc, mt, values, weak, false);
        let entry = ClassEntry {
            hash,
            class: Gc::downgrade(c.0),
        };
        table.insert_unique(hash, entry, |e| e.hash);
        c
    }
}
