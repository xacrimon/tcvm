mod bitmask;
mod group;
mod tag;

use self::bitmask::BitMask;
pub(crate) use self::{
    bitmask::BitMaskIter,
    group::Group,
    tag::{Tag, TagSliceExt},
};

#[cfg(test)]
mod tests {
    use super::{BitMask, Group, Tag};

    /// The group operations treat dead tags as the table relies on, over every
    /// control byte a table can hold in many neighbourhoods: a dead tag matches
    /// itself and at most other dead tags (never a bucket without an entry),
    /// is neither empty nor full, and a rehash clears it.
    #[test]
    fn dead_tags() {
        #[repr(C, align(16))]
        struct Aligned([Tag; 16]);
        let dead: Vec<Tag> = (0..0x80).map(|t| Tag(t).dead()).collect();
        for t in 0..=255 {
            assert_eq!(Tag(t).is_dead(), dead.contains(&Tag(t)), "{t:#x}");
        }
        for &d in &dead {
            assert!(d.is_special() && !d.special_is_empty(), "{d:?}");
        }
        let mut tags: Vec<Tag> = dead.clone();
        tags.sort_by_key(|t| t.0);
        tags.dedup();
        tags.extend((0..0x80).map(Tag).chain([Tag::EMPTY, Tag::DELETED]));
        let n = tags.len();
        for start in 0..n {
            let mut group = Aligned([Tag::EMPTY; 16]);
            for (i, b) in group.0.iter_mut().take(Group::WIDTH).enumerate() {
                *b = tags[(start + i * 37) % n];
            }
            let bytes = &group.0[..Group::WIDTH];
            let loaded = unsafe { Group::load_aligned(group.0.as_ptr()) };
            let bits = |m: BitMask| m.into_iter().collect::<Vec<_>>();
            let which = |f: &dyn Fn(Tag) -> bool| -> Vec<usize> {
                (0..Group::WIDTH).filter(|&i| f(bytes[i])).collect()
            };
            for &d in &dead {
                let got = bits(loaded.match_tag(d));
                assert!(
                    which(&|t| t == d).iter().all(|i| got.contains(i)),
                    "{bytes:?}"
                );
                assert!(got.iter().all(|&i| bytes[i].is_dead()), "{bytes:?}");
            }
            assert_eq!(bits(loaded.match_empty()), which(&|t| t == Tag::EMPTY));
            assert_eq!(bits(loaded.match_full()), which(&Tag::is_full));
            assert_eq!(
                bits(loaded.match_empty_or_deleted()),
                which(&Tag::is_special)
            );
            let mut out = Aligned([Tag::EMPTY; 16]);
            unsafe {
                loaded
                    .convert_special_to_empty_and_full_to_deleted()
                    .store_aligned(out.0.as_mut_ptr());
            }
            for i in which(&Tag::is_dead) {
                assert_eq!(out.0[i], Tag::EMPTY);
            }
        }
    }
}
