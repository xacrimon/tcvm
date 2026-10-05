//! The hash parts of a table (integer, misc and dict-mode string keys) share
//! one entry layout and one set of primitives built around Lua's `next`
//! contract: deleting an entry leaves it in its bucket, dead, so a traversal
//! can resume from its key.

use core::alloc::Allocator;
use core::hash::BuildHasher;

use super::swiss::RawTable;
use crate::dmm::allocator_api::GcAlloc;
use crate::dmm::{Collect, Gc, Trace};
use crate::env::string::LuaString;
use crate::env::value::{Value, value_hash};

/// One hash-part entry. Only live entries are traced, so a dead entry's key
/// may dangle once the collector frees the object, as in the reference:
/// dead keys are only ever compared bitwise, never dereferenced or rehashed.
/// A set revives its key's dead entry, so `next` never meets a stale second
/// entry for a key.
#[derive(Clone, Copy)]
pub(super) struct Entry<'gc, K> {
    pub key: K,
    pub value: Value<'gc>,
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

pub(super) type Part<'gc, K, A> = RawTable<Entry<'gc, K>, A>;

unsafe impl<'gc, K: Copy + Collect<'gc>> Collect<'gc> for Part<'gc, K, GcAlloc<'gc>> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        mark(cc, self);
        for e in self.iter() {
            cc.trace(&e.key);
            cc.trace(&e.value);
        }
    }
}

/// Mark `table`'s memory, which its holder must do whenever it is traced.
#[inline]
pub(super) fn mark<'gc, K: Copy, T: Trace<'gc>>(cc: &mut T, table: &Part<'gc, K, GcAlloc<'gc>>) {
    if let Some(ptr) = table.allocation() {
        // SAFETY: the table's allocation, live while the table is.
        unsafe { GcAlloc::mark(cc, ptr) };
    }
}

#[inline]
pub(super) fn get<'gc, K: Key, A: Allocator>(
    table: &Part<'gc, K, A>,
    hash: u64,
    key: K,
) -> Value<'gc> {
    table
        .get(hash, |e| e.key.same(key))
        .map_or(Value::nil(), |e| e.value)
}

/// Bucket index of `key`'s entry, live or dead.
#[inline]
pub(super) fn position<K: Key, A: Allocator>(
    table: &Part<'_, K, A>,
    hash: u64,
    key: K,
) -> Option<usize> {
    table.position(hash, |e| e.key.same(key))
}

/// First live entry at bucket index `from` or later, and its bucket.
#[inline]
pub(super) fn next_live<'a, 'gc, K: Key, A: Allocator>(
    table: &'a Part<'gc, K, A>,
    from: usize,
) -> Option<(usize, &'a Entry<'gc, K>)> {
    table.next_full(from)
}

/// Only a new key can move entries (by rehashing): an overwrite, delete or
/// revive stays in its bucket, so none of them disturbs a `pairs` loop.
#[inline]
pub(super) fn set<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) {
    if value.is_nil() {
        table.kill(hash, |e| e.key.same(key));
    } else if let Err(NeedsGrowth) = set_no_grow(table, hash, key, value) {
        table.insert(hash, Entry { key, value }, rehash);
    }
}

/// The table must grow before it can take a new key.
pub(super) struct NeedsGrowth;

/// [`set`] for a non-nil `value`, unless `key` is new and the table is full.
#[inline(always)]
pub(super) fn set_no_grow<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) -> Result<(), NeedsGrowth> {
    debug_assert!(!value.is_nil());
    // SAFETY: each result is used straight from `find_or_find_insert_index` for `hash`.
    unsafe {
        match table.find_or_find_insert_index(hash, |e| e.key.same(key)) {
            Ok((pos, bit)) => table.revive(pos, bit, hash).value = value,
            Err(index) if table.needs_growth(index) => return Err(NeedsGrowth),
            Err(index) => table.insert_at_index(index, hash, Entry { key, value }),
        }
    }
    Ok(())
}

/// Insert a key known to be absent (dict migration, array rehash).
#[inline]
pub(super) fn insert_unique<'gc, K: Key, A: Allocator>(
    table: &mut Part<'gc, K, A>,
    hash: u64,
    key: K,
    value: Value<'gc>,
) {
    table.insert(hash, Entry { key, value }, rehash);
}

fn rehash<K: Key>(e: &Entry<'_, K>) -> u64 {
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

    fn is_live<K: Key>(t: &Part<'_, K, Global>, key: K) -> bool {
        position(t, key.hash(), key)
            .and_then(|p| next_live(t, p))
            .is_some_and(|(_, e)| e.key.same(key))
    }

    /// Freed addresses are reused at once with new content while dead entries
    /// still hold them. Only 64 low hash values and four tags are drawn, in
    /// pairs that differ in the lowest bit as the generic group's false matches
    /// do and pairs that share a dead tag, so probe sequences and tags collide
    /// constantly.
    #[test]
    fn reused_address() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % n
        };
        let mut t: Part<'_, Addr, Global> = RawTable::new_in(Global);
        let (mut live, mut free) = (Vec::new(), Vec::new());
        let v = Value::boolean(true);
        for step in 0..10_000 {
            match rand(16) {
                0..=9 if live.len() < 300 => {
                    let h = [2u64, 3, 66, 67][rand(4)] << 57 | rand(64) as u64;
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
                    let at = position(&t, a.hash(), a);
                    set(&mut t, a.hash(), a, Value::nil());
                    set(&mut t, a.hash(), a, v);
                    assert_eq!(position(&t, a.hash(), a), at, "step {step}: revive moved");
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
                        let Some((_, &e)) = next_live(&t, from) else {
                            break;
                        };
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
            assert_eq!(t.len(), live.len(), "step {step}: live count");
            if step % 16 != 0 {
                continue;
            }
            for &a in &live {
                assert!(is_live(&t, Addr(a)), "step {step}: live key lost");
            }
        }
    }

    /// A table with no room left for a new key: overwriting, deleting and
    /// re-setting keys while traversing it must neither move an entry nor skip or
    /// revisit one.
    #[test]
    fn traverse_full_table_while_setting() {
        let mut t: Part<'_, i64, Global> = RawTable::new_in(Global);
        let v = |n: i64| Value::boolean(n % 2 == 0);
        for n in 0..1000 {
            // Fresh keys each round, so they also land in other keys' dead buckets. Grow to
            // a size that varies by round, then fill the table without growing it.
            let start = n * 1_000_000;
            let mut k = start;
            while k <= start + n % 500 {
                set(&mut t, k.hash(), k, v(k));
                k += 1;
            }
            while set_no_grow(&mut t, k.hash(), k, v(k)).is_ok() {
                k += 1;
            }
            let keys = start..k;
            // Delete and re-set half the keys first, so dead buckets are reused at capacity too.
            for k in keys.clone().step_by(2) {
                set(&mut t, k.hash(), k, Value::nil());
                set(&mut t, k.hash(), k, v(k + n));
            }
            let (mut seen, mut cursor) = (HashSet::new(), None);
            loop {
                let from = cursor.map_or(0, |k: i64| position(&t, k.hash(), k).unwrap() + 1);
                let Some((_, &e)) = next_live(&t, from) else {
                    break;
                };
                assert!(seen.insert(e.key), "{n}: revisited {}", e.key);
                let at = position(&t, e.key.hash(), e.key);
                match e.key % 3 {
                    0 => set(&mut t, e.key.hash(), e.key, v(e.key + 1)),
                    1 => set(&mut t, e.key.hash(), e.key, Value::nil()),
                    _ => {}
                }
                assert_eq!(
                    position(&t, e.key.hash(), e.key),
                    at,
                    "{n}: {} moved",
                    e.key
                );
                cursor = Some(e.key);
            }
            assert_eq!(
                seen.len(),
                keys.clone().count(),
                "{n}: traversal missed keys"
            );
            // Leave only dead entries behind for the next round's keys, or start over.
            for k in keys {
                set(&mut t, k.hash(), k, Value::nil());
            }
            if n % 7 == 0 {
                t = RawTable::new_in(Global);
            }
        }
    }

    /// Three keys share an address in turn: W dies in group 0, X dies in group 1, and Y
    /// revives X's bucket past W's. A delete of Y must find Y's entry, not W's, even with a
    /// true match for Y's dead tag right below W's, where upstream's generic group falsely
    /// matched W's tag (Y's with the low bit flipped).
    #[test]
    fn reused_address_behind_a_dead_twin() {
        let width = super::super::swiss::Group::WIDTH;
        let h = |tag: u64, low: usize| tag << 57 | low as u64;
        let new = |hash: u64| {
            HEAP.with_borrow_mut(|m| {
                m.push(hash);
                Addr(m.len() - 1)
            })
        };
        let reuse = |a: Addr, hash: u64| HEAP.with_borrow_mut(|m| m[a.0] = hash);
        let mut t: Part<'_, Addr, Global> = RawTable::with_capacity_in(20, Global);
        let v = Value::boolean(true);
        // Fill group 0 so probes from it go on to group 1.
        for tag in 1..=width as u64 - 2 {
            let f = new(h(tag, 0));
            set(&mut t, f.hash(), f, v);
        }
        let z = new(h(0x50, 0)); // dead tag 0x90, Y's
        set(&mut t, z.hash(), z, v);
        let a = new(h(0x11, 0)); // W, dead tag 0x91
        set(&mut t, a.hash(), a, v);
        set(&mut t, a.hash(), a, Value::nil());
        reuse(a, h(0x50, width)); // X, dead tag 0x90
        set(&mut t, a.hash(), a, v);
        set(&mut t, a.hash(), a, Value::nil());
        reuse(a, h(0x10, 0)); // Y, dead tag 0x90
        set(&mut t, a.hash(), a, v);
        assert_eq!(
            position(&t, a.hash(), a),
            Some(width),
            "Y revives X's bucket"
        );
        set(&mut t, z.hash(), z, Value::nil());
        set(&mut t, a.hash(), a, Value::nil());
        assert!(
            get(&t, a.hash(), a).is_nil(),
            "delete of Y missed its entry"
        );
        assert_eq!(position(&t, a.hash(), a), Some(width));
        assert_eq!(t.len(), width - 2);
    }
}
