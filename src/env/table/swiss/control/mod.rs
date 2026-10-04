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
    /// control byte a table can hold in many neighbourhoods: a full or dead tag
    /// matches exactly itself, a dead tag is neither empty nor full, and a rehash
    /// clears it.
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
        let matchable: Vec<Tag> = tags
            .iter()
            .copied()
            .filter(|t| t.is_full() || t.is_dead())
            .collect();
        let n = tags.len();
        let mut groups: Vec<Vec<Tag>> = (0..n)
            .map(|start| {
                (0..Group::WIDTH)
                    .map(|i| tags[(start + i * 37) % n])
                    .collect()
            })
            .collect();
        // Upstream's generic `match_tag` falsely matched `t ^ 1` right above a true `t`.
        groups.extend(matchable.iter().map(|&t| {
            (0..Group::WIDTH)
                .map(|i| if i % 2 == 0 { t } else { Tag(t.0 ^ 1) })
                .collect()
        }));
        for g in &groups {
            let mut group = Aligned([Tag::EMPTY; 16]);
            group.0[..Group::WIDTH].copy_from_slice(g);
            let bytes = &group.0[..Group::WIDTH];
            let loaded = unsafe { Group::load_aligned(group.0.as_ptr()) };
            let bits = |m: BitMask| m.into_iter().collect::<Vec<_>>();
            let which = |f: &dyn Fn(Tag) -> bool| -> Vec<usize> {
                (0..Group::WIDTH).filter(|&i| f(bytes[i])).collect()
            };
            for &t in &matchable {
                assert_eq!(
                    bits(loaded.match_tag(t)),
                    which(&|b| b == t),
                    "{t:?} in {bytes:?}"
                );
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
