mod inner {
    use core::hash::{BuildHasher, Hash};

    use std::alloc::Allocator;

    use crate::dmm::barrier::IndexWrite;
    use crate::dmm::collect::{Collect, Trace};

    unsafe impl<'gc, K, V, S, A> Collect<'gc> for hashbrown::HashMap<K, V, S, A>
    where
        K: Collect<'gc>,
        V: Collect<'gc>,
        S: 'static,
        A: Allocator + Clone + Collect<'gc>,
    {
        const NEEDS_TRACE: bool = K::NEEDS_TRACE || V::NEEDS_TRACE || A::NEEDS_TRACE;

        #[inline]
        fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
            for (k, v) in self {
                cc.trace(k);
                cc.trace(v);
            }
            cc.trace(self.allocator());
        }
    }

    unsafe impl<'gc, T, S, A> Collect<'gc> for hashbrown::HashSet<T, S, A>
    where
        T: Collect<'gc>,
        S: 'static,
        A: Allocator + Clone + Collect<'gc>,
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;

        #[inline]
        fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
            for v in self {
                cc.trace(v);
            }
            cc.trace(self.allocator());
        }
    }

    unsafe impl<'gc, T, A> Collect<'gc> for hashbrown::HashTable<T, A>
    where
        T: Collect<'gc>,
        A: Allocator + Clone + Collect<'gc>,
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;

        #[inline]
        fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
            for v in self {
                cc.trace(v);
            }
            cc.trace(self.allocator());
        }
    }

    unsafe impl<K, V, S, A, Q> IndexWrite<&Q> for hashbrown::HashMap<K, V, S, A>
    where
        K: Eq + Hash,
        Q: Hash + hashbrown::Equivalent<K> + ?Sized,
        S: BuildHasher,
        A: Allocator,
    {
    }
}
