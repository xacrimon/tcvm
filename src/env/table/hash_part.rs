//! The two hash parts of a table (misc keys and dict-mode strings) share
//! one entry layout and one set of primitives built around Lua's `next`
//! contract: deleting an entry leaves it in place with a nil value so a
//! traversal can resume from its key.

use core::alloc::Allocator;
use core::hash::BuildHasher;

use hashbrown::{HashTable, hash_table};

use crate::dmm::{Collect, Gc, Trace};
use crate::env::string::LuaString;
use crate::env::value::{Value, value_hash};

/// One hash-part entry. A dead entry (nil `value`) does not trace its
/// key, so the key may dangle once the collector frees the object; `next`
/// resolving a dead key (the only thing that looks one up) goes through
/// `Key::same_dead`, which never dereferences it. Dead entries are reaped
/// before any rehash so their hash is never recomputed. A set never revives
/// a dead entry either (a new object at a freed address would land in the
/// wrong bucket).
#[derive(Clone, Copy)]
pub(super) struct Entry<'gc, K> {
    pub key: K,
    pub value: Value<'gc>,
}

unsafe impl<'gc, K: Collect<'gc>> Collect<'gc> for Entry<'gc, K> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if self.is_live() {
            cc.trace(&self.key);
            cc.trace(&self.value);
        }
    }
}

impl<'gc, K> Entry<'gc, K> {
    #[inline]
    pub fn is_live(&self) -> bool {
        !self.value.is_nil()
    }
}

/// Hash-part key: identity comparison, and the hash used to place it.
pub(super) trait Key: Copy {
    /// Full equality, used against a live entry. Safe to dereference — a live
    /// entry's key is traced.
    fn same(self, other: Self) -> bool;
    fn hash(self) -> u64;
    /// Equality against a *dead* entry's key, which may dangle. Must never
    /// dereference either side. Defaults to `same`, which already holds for
    /// keys with no heap-boxed representation (e.g. `LuaString`, compared by
    /// pointer either way).
    #[inline]
    fn same_dead(self, other: Self) -> bool {
        self.same(other)
    }
}

impl Key for Value<'_> {
    #[inline]
    fn same(self, other: Self) -> bool {
        self == other
    }

    #[inline]
    fn hash(self) -> u64 {
        value_hash(self)
    }

    #[inline]
    fn same_dead(self, other: Self) -> bool {
        self.same_bits(&other)
    }
}

impl Key for LuaString<'_> {
    #[inline]
    fn same(self, other: Self) -> bool {
        Gc::ptr_eq(self.inner(), other.inner())
    }

    #[inline]
    fn hash(self) -> u64 {
        lua_string_hash(self)
    }
}

#[inline]
pub(super) fn lua_string_hash(key: LuaString<'_>) -> u64 {
    foldhash::fast::FixedState::default().hash_one(key)
}

pub(super) type Part<'gc, K, A> = HashTable<Entry<'gc, K>, A>;

#[inline]
pub(super) fn get<'gc, K: Key, A: Allocator>(
    table: &Part<'gc, K, A>,
    hash: u64,
    key: K,
) -> Value<'gc> {
    table
        .find(hash, |e| e.is_live() && e.key.same(key))
        .map_or(Value::nil(), |e| e.value)
}

/// Bucket index of `key`, dead or alive. A dead entry's key only ever matches by
/// `same_dead`: it may be a dangling heap-boxed integer, so full value equality
/// (which would dereference it) is limited to live entries.
#[inline]
pub(super) fn position<K: Key, A: Allocator>(
    table: &Part<'_, K, A>,
    hash: u64,
    key: K,
) -> Option<usize> {
    table.find_bucket_index(hash, |e| {
        if e.is_live() {
            e.key.same(key)
        } else {
            e.key.same_dead(key)
        }
    })
}

/// First live entry at bucket index `from` or later.
pub(super) fn next_live<'a, 'gc, K, A: Allocator>(
    table: &'a Part<'gc, K, A>,
    from: usize,
) -> Option<&'a Entry<'gc, K>> {
    (from..table.num_buckets())
        .filter_map(|i| table.get_bucket(i))
        .find(|e| e.is_live())
}

/// Deletion goes through `find_mut` rather than `entry`: `entry` reserves
/// a slot before probing, and a rehash here would reorder a `pairs` loop
/// that is clearing the table. An insert reaps dead entries when it would
/// otherwise grow the table — hashbrown rehashes exactly when
/// `len == capacity`, so this is what keeps a dead key out of the hasher.
pub(super) fn set<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) {
    if value.is_nil() {
        if let Some(e) = table.find_mut(hash, |e| e.is_live() && e.key.same(key)) {
            e.value = value;
        }
        return;
    }
    if table.len() == table.capacity() {
        table.retain(|e| e.is_live());
    }
    match table.entry(hash, |e| e.is_live() && e.key.same(key), rehash) {
        hash_table::Entry::Occupied(mut e) => e.get_mut().value = value,
        hash_table::Entry::Vacant(e) => {
            e.insert(Entry { key, value });
        }
    }
}

/// Insert a key known to be absent (dict migration).
#[inline]
pub(super) fn insert_unique<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) {
    table.insert_unique(hash, Entry { key, value }, rehash);
}

fn rehash<K: Key>(e: &Entry<'_, K>) -> u64 {
    debug_assert!(e.is_live(), "rehash reached a dead entry");
    e.key.hash()
}
