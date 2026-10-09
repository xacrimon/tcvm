//! A compiled region: its code, the constants it embeds, and its exits.

use std::cell::Cell;

use crate::dmm::{Collect, Gc};
use crate::env::function::Prototype;

/// A region of compiled code, owned by its prototype's `JitState`.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Region<'gc> {
    pub(crate) proto: Gc<'gc, Prototype<'gc>>,
    pub(crate) entry_pc: u32,
    #[collect(require_static)]
    pub(crate) retired: Cell<bool>,
}
