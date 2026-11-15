//! Site records: what the builder digested of each site's feedback, which
//! a generic op names by index (`aux`) and its rewrite reads in place of
//! the feedback and the heap. One table per compile, lent to the emitter
//! that runs.

use crate::env::shape::Kind;
use crate::env::table::SlotLoc;
use crate::jit::feedback::MAX_SHAPES;

/// What a field access does for receivers of one shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Form {
    /// The shape holds the key at `loc`.
    Own,
    /// The shape lacks the key: a load's nil, unless `__index` answers.
    Absent,
    /// A store adds the key at `loc`, moving the table to shape `to`.
    Transition,
    /// The helper's: a dictionary, or a store the interpreter keeps to its
    /// slow path.
    Helper,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Case {
    /// The shape, as a table's shape word.
    pub(crate) shape: u64,
    pub(crate) form: Form,
    /// Whether the shape has a class: its metamethod bits are read where a
    /// nil lets a metamethod answer.
    pub(crate) classed: bool,
    pub(crate) loc: SlotLoc,
    /// A store's: the kind the slot's values keep to, and the shape its
    /// tables have while it tracks one (0 for none). Both read at compile
    /// time: a kind only widens and a tracked shape only goes away.
    pub(crate) kind: Kind,
    pub(crate) expect: u64,
    /// A transition's new shape.
    pub(crate) to: u64,
}

/// A constant-key access: `n` cases in the order tested, none for a site
/// with no usable feedback or more shapes than a site takes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FieldSite {
    /// The key's index in the constant pool: a string.
    pub(crate) key: u32,
    pub(crate) n: u8,
    pub(crate) cases: [Case; MAX_SHAPES],
}

#[derive(Default)]
pub(crate) struct Sites {
    pub(crate) fields: Vec<FieldSite>,
    /// The shapes the emitted code names, which the region keeps alive.
    pub(crate) kept: Vec<u64>,
}

impl Sites {
    pub(crate) fn clear(&mut self) {
        self.fields.clear();
        self.kept.clear();
    }

    /// The index a generic op names `site` by; none past what `aux` holds.
    pub(crate) fn field(&mut self, site: FieldSite) -> Option<u16> {
        let at = u16::try_from(self.fields.len()).ok()?;
        self.fields.push(site);
        Some(at)
    }

    pub(crate) fn keep(&mut self, shape: u64) {
        if !self.kept.contains(&shape) {
            self.kept.push(shape);
        }
    }
}
