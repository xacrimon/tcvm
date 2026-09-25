use core::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::hash::BuildHasher;

use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::{Collect, Finalization, Gc, Mutation, RefLock, TrailingBytes};
use crate::lua::Context;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaString<'gc>(Gc<'gc, StringData>);

/// A string's header; its bytes follow it in the same allocation.
#[derive(Collect)]
#[collect(internal, require_static)]
pub struct StringData {
    /// The interner's hash of the bytes.
    hash: u64,
    len: usize,
}

// SAFETY: only `Interner::intern` makes a `StringData`, through `Gc::new_with_bytes` with `len`
// bytes, and it has no drop glue.
unsafe impl TrailingBytes for StringData {
    #[inline(always)]
    fn trailing_len(&self) -> usize {
        self.len
    }
}

impl<'gc> LuaString<'gc> {
    pub fn new(context: Context<'gc>, bytes: &[u8]) -> Self {
        context.interner().intern(context.mutation(), bytes)
    }

    pub fn as_bytes(self) -> &'gc [u8] {
        Gc::trailing_bytes(self.0)
    }

    pub fn len(self) -> usize {
        self.0.len
    }

    /// A hash of the bytes, computed once when the string was interned.
    #[inline]
    pub(crate) fn content_hash(self) -> u64 {
        self.0.hash
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn inner(&self) -> Gc<'gc, StringData> {
        self.0
    }

    pub(crate) fn from_inner(inner: Gc<'gc, StringData>) -> Self {
        Self(inner)
    }
}

impl<'gc> PartialEq for LuaString<'gc> {
    fn eq(&self, other: &Self) -> bool {
        if Gc::ptr_eq(self.0, other.0) {
            return true;
        }
        debug_assert_ne!(self.as_bytes(), other.as_bytes());
        false
    }
}

impl<'gc> Eq for LuaString<'gc> {}

impl<'gc> PartialOrd for LuaString<'gc> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'gc> Ord for LuaString<'gc> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl<'gc> Hash for LuaString<'gc> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.content_hash());
    }
}

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct Interner<'gc>(Gc<'gc, RefLock<InternerState<'gc>>>);

/// Interned strings, deliberately untraced so a string dies once nothing else
/// references it. Sound only because [`Interner::prune`] drops dead strings after
/// marking and before every sweep (see `Lua::collect_debt`), so no entry ever
/// points at freed memory.
struct InternerState<'gc> {
    table: HashTable<LuaString<'gc>, MetricsAlloc<'gc>>,
    // Fixed so string-keyed tables, which reuse this hash, iterate in the same order every run.
    hasher: foldhash::fast::FixedState,
}

// SAFETY: the entries are not traced on purpose (see `InternerState`).
unsafe impl<'gc> Collect<'gc> for InternerState<'gc> {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Interner<'gc> {
    pub(crate) fn new(mc: &Mutation<'gc>) -> Self {
        let state = InternerState {
            table: HashTable::new_in(MetricsAlloc::new(mc)),
            hasher: foldhash::fast::FixedState::default(),
        };

        Self(Gc::new(mc, RefLock::new(state)))
    }

    /// Forget the strings that die this cycle. Must run once marking is complete
    /// and before sweeping starts, with no mutation in between.
    pub(crate) fn prune(&self, fc: &Finalization<'gc>) {
        self.0
            .borrow_mut(fc)
            .table
            .retain(|s| !Gc::is_dead(fc, s.0));
    }

    pub(crate) fn intern(&self, mc: &Mutation<'gc>, bytes: &[u8]) -> LuaString<'gc> {
        let mut state = self.0.borrow_mut(mc);
        let InternerState { table, hasher } = &mut *state;

        let hash = {
            let mut hasher = hasher.build_hasher();
            hasher.write(bytes);
            hasher.finish()
        };

        let eq = |s: &LuaString<'gc>| s.content_hash() == hash && s.as_bytes() == bytes;
        match table.entry(hash, eq, |s| s.content_hash()) {
            hash_table::Entry::Occupied(o) => *o.get(),
            hash_table::Entry::Vacant(v) => {
                let len = bytes.len();
                let string = LuaString(Gc::new_with_bytes(mc, StringData { hash, len }, bytes));
                v.insert(string);
                string
            }
        }
    }
}
