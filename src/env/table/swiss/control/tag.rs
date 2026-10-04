use core::{fmt, mem};

/// Single tag in a control group.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Tag(pub(in super::super) u8);
impl Tag {
    /// Control tag value for an empty bucket.
    pub(crate) const EMPTY: Tag = Tag(0b1111_1111);

    /// Control tag value for a deleted bucket.
    pub(crate) const DELETED: Tag = Tag(0b1000_0000);

    /// Checks whether a control tag represents a full bucket (top bit is clear).
    #[inline]
    pub(crate) const fn is_full(self) -> bool {
        self.0 & 0x80 == 0
    }

    /// Checks whether a control tag represents a special value (top bit is set).
    #[inline]
    pub(crate) const fn is_special(self) -> bool {
        self.0 & 0x80 != 0
    }

    /// Checks whether a special control value is EMPTY. Unlike upstream this can't test one
    /// bit, since dead tags use the low bit.
    #[inline]
    pub(crate) const fn special_is_empty(self) -> bool {
        debug_assert!(self.is_special());
        self.0 == Self::EMPTY.0
    }

    /// Creates a control tag representing a full bucket with the given hash.
    #[inline]
    pub(crate) const fn full(hash: u64) -> Tag {
        // Constant for function that grabs the top 7 bits of the hash.
        const MIN_HASH_LEN: usize = if size_of::<usize>() < size_of::<u64>() {
            size_of::<usize>()
        } else {
            size_of::<u64>()
        };

        // Grab the top 7 bits of the hash. While the hash is normally a full 64-bit
        // value, some hash functions (such as FxHash) produce a usize result
        // instead, which means that the top 32 bits are 0 on 32-bit platforms.
        // So we use MIN_HASH_LEN constant to handle this.
        let top7 = hash >> (MIN_HASH_LEN * 8 - 7);
        Tag((top7 & 0x7f) as u8) // truncation
    }

    /// The tag of a dead bucket that held this full tag: `0b10tt_tttt`, which the generic
    /// group's `match_empty` doesn't take for EMPTY. It may equal `DELETED`, which only exists
    /// inside a rehash, when no bucket is dead.
    #[inline]
    pub(crate) const fn dead(self) -> Tag {
        debug_assert!(self.is_full());
        Tag(0x80 | self.0 & 0x3f)
    }

    #[inline]
    pub(crate) const fn is_dead(self) -> bool {
        self.0 & 0xc0 == 0x80
    }
}
impl fmt::Debug for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dead() {
            f.debug_tuple("dead").field(&(self.0 & 0x3f)).finish()
        } else if self.is_special() {
            if self.special_is_empty() {
                f.pad("EMPTY")
            } else {
                f.pad("DELETED")
            }
        } else {
            f.debug_tuple("full").field(&(self.0 & 0x7F)).finish()
        }
    }
}

/// Extension trait for slices of tags.
pub(crate) trait TagSliceExt {
    /// Fills the control with the given tag.
    fn fill_tag(&mut self, tag: Tag);

    /// Clears out the control.
    #[inline]
    fn fill_empty(&mut self) {
        self.fill_tag(Tag::EMPTY);
    }
}
impl TagSliceExt for [mem::MaybeUninit<Tag>] {
    #[inline]
    fn fill_tag(&mut self, tag: Tag) {
        // SAFETY: We have access to the entire slice, so, we can write to the entire slice.
        unsafe { self.as_mut_ptr().write_bytes(tag.0, self.len()) }
    }
}
