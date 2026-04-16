use core::hash::{BuildHasher, Hash, Hasher};
use std::cmp::Ordering;

use crate::dmm::{Collect, Gc, Mutation};

#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct LuaString<'gc>(Gc<'gc, StringData>);

#[derive(Collect)]
#[collect(internal, require_static)]
pub struct StringData {
    hash: u64,
    bytes: Box<[u8]>,
}

fn compute_hash(bytes: &[u8]) -> u64 {
    let mut hasher = foldhash::fast::FixedState::with_seed(0).build_hasher();
    bytes.hash(&mut hasher);
    hasher.finish()
}

impl<'gc> LuaString<'gc> {
    pub fn new(mc: &Mutation<'gc>, bytes: &[u8]) -> Self {
        let hash = compute_hash(bytes);
        let data = StringData {
            hash,
            bytes: bytes.into(),
        };
        LuaString(Gc::new(mc, data))
    }

    pub fn as_bytes(self) -> &'gc [u8] {
        &Gc::as_ref(self.0).bytes
    }

    pub fn len(self) -> usize {
        Gc::as_ref(self.0).bytes.len()
    }

    pub fn cached_hash(self) -> u64 {
        Gc::as_ref(self.0).hash
    }
}

impl<'gc> PartialEq for LuaString<'gc> {
    fn eq(&self, other: &Self) -> bool {
        if Gc::ptr_eq(self.0, other.0) {
            return true;
        }
        self.as_bytes() == other.as_bytes()
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
        state.write_u64(self.cached_hash());
    }
}
