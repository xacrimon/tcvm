//! The two hash parts of a table (misc keys and dict-mode strings) share
//! one entry layout and one set of primitives built around Lua's `next`
//! contract: deleting an entry leaves it in place with a nil value so a
//! traversal can resume from its key.

use core::alloc::Allocator;

use hashbrown::{HashTable, hash_table};

use crate::dmm::{Collect, Gc, Trace};
use crate::env::string::LuaString;
use crate::env::value::Value;

/// One hash-part entry. A dead entry (nil `value`) does not trace its
/// key, so the key may dangle once the collector frees the object: nothing
/// here dereferences a key. Identity is compared bitwise and `hash` is
/// stored so rehashing never recomputes it. A dead key whose address is
/// reused by a new object matches on identity only if the hash matches
/// too, which is exactly when reviving the entry in place is correct.
#[derive(Clone, Copy)]
pub(super) struct Entry<'gc, K> {
    pub key: K,
    pub hash: u64,
    pub value: Value<'gc>,
}

unsafe impl<'gc, K: Collect<'gc>> Collect<'gc> for Entry<'gc, K> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if !self.value.is_nil() {
            cc.trace(&self.key);
            cc.trace(&self.value);
        }
    }
}

impl<'gc, K> Entry<'gc, K> {
    #[inline]
    fn matches(&self, hash: u64, key: K) -> bool
    where
        K: Key,
    {
        self.hash == hash && self.key.same(key)
    }

    #[inline]
    pub fn is_live(&self) -> bool {
        !self.value.is_nil()
    }
}

/// Identity comparison that never dereferences either side.
pub(super) trait Key: Copy {
    fn same(self, other: Self) -> bool;
}

impl Key for Value<'_> {
    #[inline]
    fn same(self, other: Self) -> bool {
        self == other
    }
}

impl Key for LuaString<'_> {
    #[inline]
    fn same(self, other: Self) -> bool {
        Gc::ptr_eq(self.inner(), other.inner())
    }
}

pub(super) type Part<'gc, K, A> = HashTable<Entry<'gc, K>, A>;

#[inline]
pub(super) fn get<'gc, K: Key, A: Allocator>(
    table: &Part<'gc, K, A>,
    hash: u64,
    key: K,
) -> Value<'gc> {
    table
        .find(hash, |e| e.matches(hash, key))
        .map_or(Value::nil(), |e| e.value)
}

/// Bucket index of `key`, dead or alive.
#[inline]
pub(super) fn position<K: Key, A: Allocator>(
    table: &Part<'_, K, A>,
    hash: u64,
    key: K,
) -> Option<usize> {
    table.find_bucket_index(hash, |e| e.matches(hash, key))
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
/// that is clearing the table. Dead entries are reaped only when an insert
/// would otherwise grow the table (growth rehashes anyway, and inserting
/// mid-traversal is undefined).
pub(super) fn set<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) {
    if value.is_nil() {
        if let Some(e) = table.find_mut(hash, |e| e.matches(hash, key)) {
            e.value = value;
        }
        return;
    }
    if table.len() == table.capacity() {
        table.retain(|e| e.is_live());
    }
    match table.entry(hash, |e| e.matches(hash, key), |e| e.hash) {
        hash_table::Entry::Occupied(mut e) => e.get_mut().value = value,
        hash_table::Entry::Vacant(e) => {
            e.insert(Entry { key, hash, value });
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
    table.insert_unique(hash, Entry { key, hash, value }, |e| e.hash);
}
