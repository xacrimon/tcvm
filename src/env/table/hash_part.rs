//! The hash parts of a table (integer, misc and dict-mode string keys) share
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
/// key, so the key may dangle once the collector frees the object, as in
/// the reference: nothing dereferences a dead key. Identity is compared
/// bitwise, and dead entries are reaped before any rehash so their hash
/// is never recomputed. A set revives a dead entry only when `Key::REVIVE`
/// allows it; otherwise the key's second entry can make `next` resume from
/// the dead one and revisit entries.
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

/// Hash-part key: identity comparison that never dereferences either
/// side, and the hash used to place it.
pub(super) trait Key: Copy {
    /// Whether the hash is a function of what `same` compares, so a dead
    /// entry that matches is already in the right bucket.
    const REVIVE: bool;

    fn same(self, other: Self) -> bool;
    fn hash(self) -> u64;
}

// Integers never reach a `Value`-keyed part, so bit identity is value equality here and
// never dereferences a boxed int.
impl Key for Value<'_> {
    const REVIVE: bool = true;

    #[inline]
    fn same(self, other: Self) -> bool {
        self.same_bits(&other)
    }

    #[inline]
    fn hash(self) -> u64 {
        value_hash(self)
    }
}

// Content-hashed but compared by pointer: a new string at a freed key's
// address would land in the old string's bucket.
impl Key for LuaString<'_> {
    const REVIVE: bool = false;

    #[inline]
    fn same(self, other: Self) -> bool {
        Gc::ptr_eq(self.inner(), other.inner())
    }

    #[inline]
    fn hash(self) -> u64 {
        lua_string_hash(self)
    }
}

impl Key for i64 {
    const REVIVE: bool = true;

    #[inline]
    fn same(self, other: Self) -> bool {
        self == other
    }

    #[inline]
    fn hash(self) -> u64 {
        int_hash(self)
    }
}

#[inline]
pub(super) fn int_hash(key: i64) -> u64 {
    foldhash::fast::FixedState::default().hash_one(key)
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

/// Bucket index of `key`, dead or alive.
#[inline]
pub(super) fn position<K: Key, A: Allocator>(
    table: &Part<'_, K, A>,
    hash: u64,
    key: K,
) -> Option<usize> {
    table.find_bucket_index(hash, |e| e.key.same(key))
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
    let matches = |e: &Entry<'gc, K>| (K::REVIVE || e.is_live()) && e.key.same(key);
    match table.entry(hash, matches, rehash) {
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
