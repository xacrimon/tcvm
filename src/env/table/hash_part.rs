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
/// is never recomputed. A set revives a matching dead entry: a second entry
/// for the key could make `next` resume from the dead one and revisit entries.
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
    fn same(self, other: Self) -> bool;
    fn hash(self) -> u64;
}

// Integers never reach a `Value`-keyed part, so bit identity is value equality here and
// never dereferences a boxed int.
impl Key for Value<'_> {
    #[inline]
    fn same(self, other: Self) -> bool {
        self.same_bits(&other)
    }

    #[inline]
    fn hash(self) -> u64 {
        value_hash(self)
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

impl Key for i64 {
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
    key.content_hash()
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
    // A dead key's address may now hold another object with a different hash, but `entry`
    // only tests buckets on `hash`'s probe path with its tag, so any match is a valid home.
    match table.entry(hash, |e| e.key.same(key), rehash) {
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

#[cfg(test)]
mod tests {
    use std::alloc::Global;
    use std::cell::RefCell;
    use std::collections::HashSet;

    use super::*;

    thread_local! {
        static HEAP: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    }

    /// A `LuaString` stand-in: compared by address, hashed by whatever
    /// content `HEAP` holds there now.
    #[derive(Clone, Copy)]
    struct Addr(usize);

    impl Key for Addr {
        fn same(self, other: Self) -> bool {
            self.0 == other.0
        }

        fn hash(self) -> u64 {
            HEAP.with_borrow(|h| h[self.0])
        }
    }

    /// Freed addresses are reused at once with new content while dead entries
    /// still hold them. Only two tags and 64 low hash bits are drawn, so probe
    /// paths and tags collide constantly.
    #[test]
    fn revive_at_reused_address() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % n
        };
        let mut t: Part<'_, Addr, Global> = HashTable::new_in(Global);
        let (mut live, mut free) = (Vec::new(), Vec::new());
        let v = Value::boolean(true);
        for step in 0..10_000 {
            match rand(16) {
                0..=9 if live.len() < 300 => {
                    let h = (1 + rand(2) as u64) << 57 | rand(64) as u64;
                    let a = HEAP.with_borrow_mut(|m| match free.pop() {
                        Some(a) => {
                            m[a] = h;
                            a
                        }
                        None => {
                            m.push(h);
                            m.len() - 1
                        }
                    });
                    set(&mut t, h, Addr(a), v);
                    live.push(a);
                }
                10 if !live.is_empty() => {
                    let a = live.swap_remove(rand(live.len()));
                    set(&mut t, Addr(a).hash(), Addr(a), Value::nil());
                    free.push(a);
                }
                11 if !live.is_empty() => {
                    let a = Addr(live[rand(live.len())]);
                    set(&mut t, a.hash(), a, Value::nil());
                    set(&mut t, a.hash(), a, v);
                }
                12 => {
                    // `next` semantics, deleting some keys as they are visited. A
                    // cursor's address can't be reused until the traversal ends.
                    let expect: HashSet<usize> = live.iter().copied().collect();
                    let (mut seen, mut cursor) = (HashSet::new(), None);
                    loop {
                        let from = cursor.map_or(0, |k: Addr| {
                            position(&t, k.hash(), k).expect("cursor lost") + 1
                        });
                        let Some(&e) = next_live(&t, from) else { break };
                        assert!(seen.insert(e.key.0), "step {step}: revisited a key");
                        if rand(32) == 0 {
                            set(&mut t, e.key.hash(), e.key, Value::nil());
                        }
                        cursor = Some(e.key);
                    }
                    assert_eq!(seen, expect, "step {step}: traversal missed keys");
                    for a in live.extract_if(.., |a| get(&t, Addr(*a).hash(), Addr(*a)).is_nil()) {
                        free.push(a);
                    }
                }
                _ => {}
            }
            if step % 16 != 0 {
                continue;
            }
            for &a in &live {
                let k = Addr(a);
                let p = position(&t, k.hash(), k).expect("live key lost");
                assert!(
                    t.get_bucket(p).unwrap().is_live(),
                    "step {step}: position is a dead entry"
                );
            }
        }
    }
}
