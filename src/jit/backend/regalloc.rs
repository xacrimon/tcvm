//! Where each virtual register lives.
//!
//! The encoder consumes an [`Allocation`] and nothing else, so the choice of
//! allocator is a swappable policy. Two exist:
//!
//!   - [`spill_everything`]: every value gets a stack slot; the encoder reloads
//!     operands into scratch registers around each instruction. Terrible code —
//!     and exactly what is wanted first, because it makes the *encoder* the only
//!     thing under test. When linear scan lands, any answer it changes is a
//!     register allocation bug, bisected against this.
//!   - linear scan: not yet written.

use crate::jit::backend::mach::{MFunc, VReg};

/// A physical register, numbered within its class: `x0`–`x30`, or `d0`–`d31`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PReg(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alloc {
    Reg(PReg),
    /// A slot in the native frame, indexed in words.
    ///
    /// Spill slots are *not* Lua stack slots. The Lua stack is the interpreter's
    /// and is 16-byte tagged values; these are one machine word and the collector
    /// never sees them — which is fine only because the values they hold are also
    /// reachable from the Lua stack at every point a collection can happen (a
    /// guard's exit stub writes them all back).
    Spill(u32),
}

pub struct Allocation {
    map: Vec<Alloc>,
    pub num_spills: u32,
}

impl Allocation {
    pub fn of(&self, v: VReg) -> Alloc {
        self.map[v.0 as usize]
    }
}

/// Give every virtual register its own stack slot.
pub fn spill_everything(f: &MFunc) -> Allocation {
    Allocation {
        map: (0..f.num_vregs() as u32).map(Alloc::Spill).collect(),
        num_spills: f.num_vregs() as u32,
    }
}
