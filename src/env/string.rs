use crate::dmm::allocator_api::MetricsAlloc;
use crate::dmm::{Collect, Gc, Mutation, RefLock};
use crate::lua::Context;
use core::hash::{Hash, Hasher};
use hashbrown::{HashTable, hash_table};
use std::cmp::Ordering;
use std::hash::BuildHasher;

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaString<'gc>(Gc<'gc, StringData>);

#[derive(Collect)]
#[collect(internal, require_static)]
pub struct StringData {
    bytes: Box<[u8]>,
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

    pub fn inner(&self) -> Gc<'gc, StringData> {
        self.0
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

#[derive(Collect)]
#[collect(internal, no_drop)]
struct InternerState<'gc> {
    table: HashTable<LuaString<'gc>, MetricsAlloc<'gc>>,
    #[collect(require_static)]
    hasher: foldhash::fast::RandomState,
}

impl<'gc> Interner<'gc> {
    pub(crate) fn new(mc: &Mutation<'gc>) -> Self {
        let state = InternerState {
            table: HashTable::new_in(MetricsAlloc::new(mc)),
            hasher: foldhash::fast::RandomState::default(),
        };

        Self(Gc::new(mc, RefLock::new(state)))
    }

    pub(crate) fn intern(&self, mc: &Mutation<'gc>, bytes: &[u8]) -> LuaString<'gc> {
        let mut state = self.0.borrow_mut(mc);
        let InternerState { table, hasher } = &mut *state;

        let eq = |string: &LuaString| &*string.0.bytes == bytes;
        let hash = |string: &LuaString| {
            let mut hasher: foldhash::fast::FoldHasher<'_> = hasher.build_hasher();
            hasher.write(&string.0.bytes);
            hasher.finish()
        };

        let target_hash = {
            let mut hasher = hasher.build_hasher();
            hasher.write(&bytes);
            hasher.finish()
        };

        let entry = table.entry(target_hash, eq, hash);

        match entry {
            hash_table::Entry::Occupied(entry) => *entry.get(),
            hash_table::Entry::Vacant(entry) => {
                let data = StringData {
                    bytes: bytes.into(),
                };

                let string = LuaString(Gc::new(mc, data));
                entry.insert(string);
                string
            }
        }
    }
}
