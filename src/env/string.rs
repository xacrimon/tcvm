use core::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::hash::BuildHasher;

use hashbrown::{HashTable, hash_table};

use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::{Collect, Finalization, Gc, Mutation, RefLock};
use crate::lua::Context;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaString<'gc>(Gc<'gc, StringData>);

#[derive(Collect)]
#[collect(internal, require_static)]
pub struct StringData {
    // `'static` because `StringData` is; the arena outlives every string.
    bytes: Box<[u8], MetricsAlloc<'static>>,
}

impl<'gc> LuaString<'gc> {
    pub fn new(context: Context<'gc>, bytes: &[u8]) -> Self {
        context.interner().intern(context.mutation(), bytes)
    }

    pub fn as_bytes(self) -> &'gc [u8] {
        &Gc::as_ref(self.0).bytes
    }

    pub fn len(self) -> usize {
        Gc::as_ref(self.0).bytes.len()
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
        state.write(&self.0.bytes);
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
    table: HashTable<InternEntry<'gc>, MetricsAlloc<'gc>>,
    hasher: foldhash::fast::RandomState,
}

#[derive(Clone, Copy)]
struct InternEntry<'gc> {
    hash: u64,
    string: Gc<'gc, StringData>,
}

// SAFETY: the entries are not traced on purpose (see `InternerState`).
unsafe impl<'gc> Collect<'gc> for InternerState<'gc> {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Interner<'gc> {
    pub(crate) fn new(mc: &Mutation<'gc>) -> Self {
        let state = InternerState {
            table: HashTable::new_in(MetricsAlloc::new(mc)),
            hasher: foldhash::fast::RandomState::default(),
        };

        Self(Gc::new(mc, RefLock::new(state)))
    }

    /// Forget the strings that die this cycle. Must run once marking is complete
    /// and before sweeping starts, with no mutation in between.
    pub(crate) fn prune(&self, fc: &Finalization<'gc>) {
        self.0
            .borrow_mut(fc)
            .table
            .retain(|e| !Gc::is_dead(fc, e.string));
    }

    pub(crate) fn intern(&self, mc: &Mutation<'gc>, bytes: &[u8]) -> LuaString<'gc> {
        let mut state = self.0.borrow_mut(mc);
        let InternerState { table, hasher } = &mut *state;

        let hash = {
            let mut hasher = hasher.build_hasher();
            hasher.write(bytes);
            hasher.finish()
        };

        let eq = |e: &InternEntry<'gc>| e.hash == hash && *e.string.bytes == *bytes;
        match table.entry(hash, eq, |e| e.hash) {
            hash_table::Entry::Occupied(o) => LuaString(o.get().string),
            hash_table::Entry::Vacant(v) => {
                let mut buf = Vec::with_capacity_in(
                    bytes.len(),
                    MetricsAlloc::from_metrics(mc.metrics().clone()),
                );
                buf.extend_from_slice(bytes);
                let string = Gc::new(
                    mc,
                    StringData {
                        bytes: buf.into_boxed_slice(),
                    },
                );
                v.insert(InternEntry { hash, string });
                LuaString(string)
            }
        }
    }
}
