/*
 * The following license applies to this file, which derives many
 * details (register and constraint definitions, for example) from the
 * files `BacktrackingAllocator.h`, `BacktrackingAllocator.cpp`,
 * `LIR.h`, and possibly definitions in other related files in
 * `js/src/jit/`:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

//! The allocator's vocabulary: registers, values, operands with their
//! constraints, locations and program points. Vendored from regalloc2
//! 0.15.2 (`src/lib.rs`, `src/index.rs`): the type definitions verbatim, without
//! `MachineEnv` (and `PRegSet: From<&MachineEnv>`), the `Function` trait (`VCode`
//! is the one function type), serde attributes and `define_index!`'s storage arm.

#![allow(dead_code, clippy::all)]

/// Register classes.
///
/// Every value has a "register class", which is like a type at the
/// register-allocator level. Every register must belong to only one
/// class; i.e., they are disjoint.
///
/// For tight bit-packing throughout our data structures, we support
/// only three classes, "int", "float" and "vector". Usually two will
/// be enough on modern machines, as they have one class of general-purpose
/// integer registers of machine width (e.g. 64 bits), and another
/// class of float/vector registers used both for FP and for vector
/// operations. Additionally for machines with totally separate vector
/// registers a third class is provided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegClass {
    Int = 0,
    Float = 1,
    Vector = 2,
}

/// A physical register. Contains a physical register number and a class.
///
/// The `hw_enc` field contains the physical register number and is in
/// a logically separate index space per class; in other words, Int
/// register 0 is different than Float register 0.
///
/// Because of bit-packed encodings throughout the implementation,
/// `hw_enc` must fit in 6 bits, i.e., at most 64 registers per class.
///
/// The value returned by `index()`, in contrast, is in a single index
/// space shared by all classes, in order to enable uniform reasoning
/// about physical registers. This is done by putting the class bit at
/// the MSB, or equivalently, declaring that indices 0..=63 are the 64
/// integer registers and indices 64..=127 are the 64 float registers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PReg {
    bits: u8,
}

impl PReg {
    pub const MAX_BITS: usize = 6;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;
    pub const NUM_INDEX: usize = 1 << (Self::MAX_BITS + 2); // including RegClass bits
    pub const INVALID: u8 = ((RegClass::Int as u8) << Self::MAX_BITS) | (Self::MAX as u8);

    /// Create a new PReg. The `hw_enc` range is 6 bits.
    #[inline(always)]
    pub const fn new(hw_enc: usize, class: RegClass) -> Self {
        debug_assert!(hw_enc <= PReg::MAX);
        PReg {
            bits: ((class as u8) << Self::MAX_BITS) | (hw_enc as u8),
        }
    }

    /// The physical register number, as encoded by the ISA for the particular register class.
    #[inline(always)]
    pub const fn hw_enc(self) -> usize {
        self.bits as usize & Self::MAX
    }

    /// The register class.
    #[inline(always)]
    pub const fn class(self) -> RegClass {
        match (self.bits >> Self::MAX_BITS) & 0b11 {
            0 => RegClass::Int,
            1 => RegClass::Float,
            2 => RegClass::Vector,
            _ => unreachable!(),
        }
    }

    /// Get an index into the (not necessarily contiguous) index space of
    /// all physical registers. Allows one to maintain an array of data for
    /// all PRegs and index it efficiently.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.bits as usize
    }

    /// Construct a PReg from the value returned from `.index()`.
    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        PReg {
            bits: (index & (Self::NUM_INDEX - 1)) as u8,
        }
    }

    /// Return the "invalid PReg", which can be used to initialize
    /// data structures.
    #[inline(always)]
    pub const fn invalid() -> Self {
        PReg {
            bits: Self::INVALID,
        }
    }

    /// Return a valid [`PReg`] or [`None`] if it is invalid.
    #[inline(always)]
    pub const fn as_valid(self) -> Option<Self> {
        if self.bits == Self::INVALID {
            None
        } else {
            Some(self)
        }
    }
}

impl Default for PReg {
    fn default() -> Self {
        Self::invalid()
    }
}

impl core::fmt::Debug for PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "PReg(hw = {}, class = {:?}, index = {})",
            self.hw_enc(),
            self.class(),
            self.index()
        )
    }
}

impl core::fmt::Display for PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let class = match self.class() {
            RegClass::Int => "i",
            RegClass::Float => "f",
            RegClass::Vector => "v",
        };
        write!(f, "p{}{}", self.hw_enc(), class)
    }
}

/// A type for internal bit arrays.
type Bits = u64;

/// A physical register set. Used to represent clobbers
/// efficiently.
///
/// The set is `Copy` and is guaranteed to have constant, and small,
/// size, as it is based on a bitset internally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PRegSet {
    bits: [Bits; Self::LEN],
}

impl PRegSet {
    /// The number of bits per element in the internal bit array.
    const BITS: usize = core::mem::size_of::<Bits>() * 8;

    /// Length of the internal bit array.
    const LEN: usize = (PReg::NUM_INDEX + Self::BITS - 1) / Self::BITS;

    /// Create an empty set.
    pub const fn empty() -> Self {
        Self {
            bits: [0; Self::LEN],
        }
    }

    /// Splits the given register index into parts to access the internal bit array.
    const fn split_index(reg: PReg) -> (usize, usize) {
        let index = reg.index();
        (index >> Self::BITS.ilog2(), index & (Self::BITS - 1))
    }

    /// Returns whether the given register is part of the set.
    pub fn contains(&self, reg: PReg) -> bool {
        let (index, bit) = Self::split_index(reg);
        self.bits[index] & (1 << bit) != 0
    }

    /// Add a physical register (PReg) to the set, returning the new value.
    pub const fn with(self, reg: PReg) -> Self {
        let (index, bit) = Self::split_index(reg);
        let mut out = self;
        out.bits[index] |= 1 << bit;
        out
    }

    /// Add a physical register (PReg) to the set.
    pub const fn add(&mut self, reg: PReg) {
        let (index, bit) = Self::split_index(reg);
        self.bits[index] |= 1 << bit;
    }

    /// Remove a physical register (PReg) from the set.
    pub fn remove(&mut self, reg: PReg) {
        let (index, bit) = Self::split_index(reg);
        self.bits[index] &= !(1 << bit);
    }

    /// Add all of the registers in one set to this one, mutating in
    /// place.
    pub fn union_from(&mut self, other: PRegSet) {
        for i in 0..self.bits.len() {
            self.bits[i] |= other.bits[i];
        }
    }

    pub fn intersect_from(&mut self, other: PRegSet) {
        for i in 0..self.bits.len() {
            self.bits[i] &= other.bits[i];
        }
    }

    pub fn invert(&self) -> PRegSet {
        let mut set = self.bits;
        for i in 0..self.bits.len() {
            set[i] = !self.bits[i];
        }
        PRegSet { bits: set }
    }

    pub fn is_empty(&self, regclass: RegClass) -> bool {
        self.bits[regclass as usize] == 0
    }

    /// Returns the number of register in this set.
    pub fn len(&self) -> u32 {
        self.bits.iter().map(|s| s.count_ones()).sum()
    }

    /// Returns the maximum register in this set, with the highest hw_enc value.
    pub fn max_preg(&self) -> Option<PReg> {
        self.into_iter().last()
    }

    /// Add all registers from `0..reg` to this set, not including `reg` itself.
    pub fn add_up_to(&mut self, reg: PReg) {
        let (index, bit) = Self::split_index(reg);
        for i in 0..index {
            self.bits[i] = !0;
        }
        self.bits[index] = (1 << bit) - 1;
    }
}

impl core::ops::BitAnd<PRegSet> for PRegSet {
    type Output = PRegSet;

    fn bitand(self, rhs: PRegSet) -> Self::Output {
        let mut out = self;
        out.intersect_from(rhs);
        out
    }
}

impl core::ops::BitOr<PRegSet> for PRegSet {
    type Output = PRegSet;

    fn bitor(self, rhs: PRegSet) -> Self::Output {
        let mut out = self;
        out.union_from(rhs);
        out
    }
}

impl IntoIterator for &PRegSet {
    type Item = PReg;
    type IntoIter = PRegSetIter;
    fn into_iter(self) -> PRegSetIter {
        (*self).into_iter()
    }
}

impl IntoIterator for PRegSet {
    type Item = PReg;
    type IntoIter = PRegSetIter;
    fn into_iter(self) -> PRegSetIter {
        PRegSetIter {
            bits: self.bits,
            cur: 0,
        }
    }
}

pub struct PRegSetIter {
    bits: [Bits; PRegSet::LEN],
    cur: usize,
}

impl Iterator for PRegSetIter {
    type Item = PReg;
    fn next(&mut self) -> Option<PReg> {
        loop {
            let bits = self.bits.get_mut(self.cur)?;
            if *bits != 0 {
                let bit = bits.trailing_zeros();
                *bits &= !(1 << bit);
                let index = bit as usize + self.cur * PRegSet::BITS;
                return Some(PReg::from_index(index));
            }
            self.cur += 1;
        }
    }
}

impl FromIterator<PReg> for PRegSet {
    fn from_iter<T: IntoIterator<Item = PReg>>(iter: T) -> Self {
        let mut set = Self::default();
        for preg in iter {
            set.add(preg);
        }
        set
    }
}

impl core::fmt::Display for PRegSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{{")?;
        for preg in self.into_iter() {
            write!(f, "{preg}, ")?;
        }
        write!(f, "}}")
    }
}

/// A virtual register. Contains a virtual register number and a
/// class.
///
/// A virtual register ("vreg") corresponds to an SSA value. All
/// dataflow in the input program is specified via flow through a
/// virtual register; even uses of specially-constrained locations,
/// such as fixed physical registers, are done by using vregs, because
/// we need the vreg's live range in order to track the use of that
/// location.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VReg {
    bits: u32,
}

impl VReg {
    pub const MAX_BITS: usize = 30;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;

    #[inline(always)]
    pub const fn new(virt_reg: usize, class: RegClass) -> Self {
        debug_assert!(virt_reg <= VReg::MAX);
        VReg {
            bits: ((virt_reg as u32) << 2) | (class as u8 as u32),
        }
    }

    #[inline(always)]
    pub const fn vreg(self) -> usize {
        let vreg = (self.bits >> 2) as usize;
        vreg
    }

    #[inline(always)]
    pub const fn class(self) -> RegClass {
        match self.bits & 0b11 {
            0 => RegClass::Int,
            1 => RegClass::Float,
            2 => RegClass::Vector,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub const fn invalid() -> Self {
        VReg::new(Self::MAX, RegClass::Int)
    }

    #[inline(always)]
    pub const fn bits(self) -> usize {
        self.bits as usize
    }

    #[inline(always)]
    pub const fn from_bits(bits: u32) -> VReg {
        Self { bits }
    }
}

impl From<u32> for VReg {
    fn from(value: u32) -> Self {
        Self { bits: value }
    }
}

impl core::fmt::Debug for VReg {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "VReg(vreg = {}, class = {:?})",
            self.vreg(),
            self.class()
        )
    }
}

impl core::fmt::Display for VReg {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "v{}", self.vreg())
    }
}

/// A spillslot is a space in the stackframe used by the allocator to
/// temporarily store a value.
///
/// The allocator is responsible for allocating indices in this space,
/// and will specify how many spillslots have been used when the
/// allocation is completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpillSlot {
    bits: u32,
}

impl SpillSlot {
    /// The maximum spillslot index.
    pub const MAX: usize = (1 << 24) - 1;

    /// Create a new SpillSlot.
    #[inline(always)]
    pub fn new(slot: usize) -> Self {
        debug_assert!(slot <= Self::MAX);
        SpillSlot { bits: slot as u32 }
    }

    /// Get the spillslot index for this spillslot.
    #[inline(always)]
    pub fn index(self) -> usize {
        (self.bits & 0x00ffffff) as usize
    }

    /// Get the spillslot `offset` slots away.
    #[inline(always)]
    pub fn plus(self, offset: usize) -> Self {
        SpillSlot::new(self.index() + offset)
    }

    /// Get the invalid spillslot, used for initializing data structures.
    #[inline(always)]
    pub fn invalid() -> Self {
        SpillSlot { bits: 0xffff_ffff }
    }

    /// Is this the invalid spillslot?
    #[inline(always)]
    pub fn is_invalid(self) -> bool {
        self == Self::invalid()
    }

    /// Is this a valid spillslot (not `SpillSlot::invalid()`)?
    #[inline(always)]
    pub fn is_valid(self) -> bool {
        self != Self::invalid()
    }
}

impl core::fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "stack{}", self.index())
    }
}

/// An `OperandConstraint` specifies where a vreg's value must be
/// placed at a particular reference to that vreg via an
/// `Operand`. The constraint may be loose -- "any register of a given
/// class", for example -- or very specific, such as "this particular
/// physical register". The allocator's result will always satisfy all
/// given constraints; however, if the input has a combination of
/// constraints that are impossible to satisfy, then allocation may
/// fail or the allocator may panic (providing impossible constraints
/// is usually a programming error in the client, rather than a
/// function of bad input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandConstraint {
    /// Any location is fine (register or stack slot).
    Any,
    /// Operand must be in a register. Register is read-only for Uses.
    Reg,
    /// Operand must be on the stack.
    Stack,
    /// Operand must be in a fixed register.
    FixedReg(PReg),
    /// On defs only: reuse a use's register.
    Reuse(usize),
    /// Operand must be in a specific range of registers.
    ///
    /// The contained `usize` indicates the (exclusive) upper limit of a
    /// register range, `n`. An operand with this constraint may allocate a
    /// register between `0 ..= n-1`. Due to encoding constraints, `n` must be a
    /// power of two and below 2^16.
    Limit(usize),
}

impl core::fmt::Display for OperandConstraint {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::Any => write!(f, "any"),
            Self::Reg => write!(f, "reg"),
            Self::Stack => write!(f, "stack"),
            Self::FixedReg(preg) => write!(f, "fixed({preg})"),
            Self::Reuse(idx) => write!(f, "reuse({idx})"),
            Self::Limit(max) => write!(f, "limit(0..={})", max - 1),
        }
    }
}

/// The "kind" of the operand: whether it reads a vreg (Use) or writes
/// a vreg (Def).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    Def = 0,
    Use = 1,
}

/// The "position" of the operand: where it has its read/write
/// effects. These are positions "in" the instruction, and "early" and
/// "late" are relative to the instruction's main effect or
/// computation. In other words, the allocator assumes that the
/// instruction (i) performs all reads and writes of "early" operands,
/// (ii) does its work, and (iii) performs all reads and writes of its
/// "late" operands.
///
/// A "write" (def) at "early" or a "read" (use) at "late" may be
/// slightly nonsensical, given the above, if the read is necessary
/// for the computation or the write is a result of it. A way to think
/// of it is that the value (even if a result of execution) *could*
/// have been read or written at the given location without causing
/// any register-usage conflicts. In other words, these write-early or
/// use-late operands ensure that the particular allocations are valid
/// for longer than usual and that a register is not reused between
/// the use (normally complete at "Early") and the def (normally
/// starting at "Late"). See `Operand` for more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandPos {
    Early = 0,
    Late = 1,
}

/// An `Operand` encodes everything about a mention of a register in
/// an instruction: virtual register number, and any constraint that
/// applies to the register at this program point.
///
/// An Operand may be a use or def (this corresponds to `LUse` and
/// `LAllocation` in Ion).
///
/// Generally, regalloc2 considers operands to have their effects at
/// one of two points that exist in an instruction: "Early" or
/// "Late". All operands at a given program-point are assigned
/// non-conflicting locations based on their constraints. Each operand
/// has a "kind", one of use/def/mod, corresponding to
/// read/write/read-write, respectively.
///
/// Usually, an instruction's inputs will be "early uses" and outputs
/// will be "late defs", though there are valid use-cases for other
/// combinations too. For example, a single "instruction" seen by the
/// regalloc that lowers into multiple machine instructions and reads
/// some of its inputs after it starts to write outputs must either
/// make those input(s) "late uses" or those output(s) "early defs" so
/// that the conflict (overlap) is properly accounted for. See
/// comments on the constructors below for more.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Operand {
    /// Bit-pack into 64 bits.
    ///
    /// unused:21 constraint:7 kind:1 pos:1 class:2 vreg:32
    ///
    /// where `constraint` is an `OperandConstraint`, `kind` is an
    /// `OperandKind`, `pos` is an `OperandPos`, `class` is a
    /// `RegClass`, and `vreg` is a vreg index.
    ///
    /// The constraints are encoded as follows:
    /// - 1xxxxxx => FixedReg(preg)
    /// - 01xxxxx => Reuse(index)
    /// - 001xxxx => Limit(max)
    /// - 0000000 => Any
    /// - 0000001 => Reg
    /// - 0000010 => Stack
    /// - _ => Unused for now
    bits: u64,
}

impl Operand {
    const VREG_BITS: usize = 32;
    const VREG_SHIFT: usize = 0;
    const VREG_MASK: u64 = (1 << Self::VREG_BITS) - 1;

    const CLASS_BITS: usize = 2;
    const CLASS_SHIFT: usize = Self::VREG_SHIFT + Self::VREG_BITS;
    const CLASS_MASK: u64 = (1 << Self::CLASS_BITS) - 1;

    const POS_BITS: usize = 1;
    const POS_SHIFT: usize = Self::CLASS_SHIFT + Self::CLASS_BITS;
    const POS_MASK: u64 = (1 << Self::POS_BITS) - 1;

    const KIND_BITS: usize = 1;
    const KIND_SHIFT: usize = Self::POS_SHIFT + Self::POS_BITS;
    const KIND_MASK: u64 = (1 << Self::KIND_BITS) - 1;

    const CONSTRAINT_BITS: usize = 7;
    const CONSTRAINT_SHIFT: usize = Self::KIND_SHIFT + Self::KIND_BITS;
    const CONSTRAINT_MASK: u64 = (1 << Self::CONSTRAINT_BITS) - 1;

    const TOTAL_BITS: usize = Self::CONSTRAINT_SHIFT + Self::CONSTRAINT_BITS;

    /// Construct a new operand.
    #[inline(always)]
    pub fn new(
        vreg: VReg,
        constraint: OperandConstraint,
        kind: OperandKind,
        pos: OperandPos,
    ) -> Self {
        let constraint_field = match constraint {
            OperandConstraint::Any => 0,
            OperandConstraint::Reg => 1,
            OperandConstraint::Stack => 2,
            OperandConstraint::FixedReg(preg) => {
                debug_assert_eq!(preg.class(), vreg.class());
                0b1000000 | preg.hw_enc() as u64
            }
            OperandConstraint::Reuse(which) => {
                debug_assert!(which <= 0b11111);
                0b0100000 | which as u64
            }
            OperandConstraint::Limit(max) => {
                assert!(max.is_power_of_two());
                assert!(
                    max <= PReg::MAX + 1,
                    "limit is larger than the allowed register encoding"
                );
                let log2 = max.ilog2();
                debug_assert!(log2 <= 0b1111);
                0b0010000 | log2 as u64
            }
        };
        let class_field = vreg.class() as u8 as u64;
        let pos_field = pos as u8 as u64;
        let kind_field = kind as u8 as u64;
        Operand {
            bits: ((vreg.vreg() as u64) << Self::VREG_SHIFT)
                | (class_field << Self::CLASS_SHIFT)
                | (pos_field << Self::POS_SHIFT)
                | (kind_field << Self::KIND_SHIFT)
                | (constraint_field << Self::CONSTRAINT_SHIFT),
        }
    }

    /// Create an `Operand` that designates a use of a VReg that must
    /// be in a register, and that is used at the "before" point,
    /// i.e., can be overwritten by a result.
    #[inline(always)]
    pub fn reg_use(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Use,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a use of a VReg that must
    /// be in a register, and that is used up until the "after" point,
    /// i.e., must not conflict with any results.
    #[inline(always)]
    pub fn reg_use_at_end(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Use,
            OperandPos::Late,
        )
    }

    /// Create an `Operand` that designates a definition of a VReg
    /// that must be in a register, and that occurs at the "after"
    /// point, i.e. may reuse a register that carried a use into this
    /// instruction.
    #[inline(always)]
    pub fn reg_def(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::Late,
        )
    }

    /// Create an `Operand` that designates a definition of a VReg
    /// that must be in a register, and that occurs early at the
    /// "before" point, i.e., must not conflict with any input to the
    /// instruction.
    ///
    /// Note that the register allocator will ensure that such an
    /// early-def operand is live throughout the instruction, i.e., also
    /// at the after-point. Hence it will also avoid conflicts with all
    /// outputs to the instruction. As such, early defs are appropriate
    /// for use as "temporary registers" that an instruction can use
    /// throughout its execution separately from the inputs and outputs.
    #[inline(always)]
    pub fn reg_def_at_start(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a def (and use) of a
    /// temporary *within* the instruction. This register is assumed
    /// to be written by the instruction, and will not conflict with
    /// any input or output, but should not be used after the
    /// instruction completes.
    ///
    /// Note that within a single instruction, the dedicated scratch
    /// register (as specified in the `MachineEnv`) is also always
    /// available for use. The register allocator may use the register
    /// *between* instructions in order to implement certain sequences
    /// of moves, but will never hold a value live in the scratch
    /// register across an instruction.
    #[inline(always)]
    pub fn reg_temp(vreg: VReg) -> Self {
        // For now a temp is equivalent to a def-at-start operand,
        // which gives the desired semantics but does not enforce the
        // "not reused later" constraint.
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a def of a vreg that must
    /// reuse the register assigned to an input to the
    /// instruction. The input is identified by `idx` (is the `idx`th
    /// `Operand` for the instruction) and must be constraint to a
    /// register, i.e., be the result of `Operand::reg_use(vreg)`.
    #[inline(always)]
    pub fn reg_reuse_def(vreg: VReg, idx: usize) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reuse(idx),
            OperandKind::Def,
            OperandPos::Late,
        )
    }

    /// Create an `Operand` that designates a use of a vreg and
    /// ensures that it is placed in the given, fixed PReg at the
    /// use. It is guaranteed that the `Allocation` resulting for this
    /// operand will be `preg`.
    #[inline(always)]
    pub fn reg_fixed_use(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Use,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a def of a vreg and
    /// ensures that it is placed in the given, fixed PReg at the
    /// def. It is guaranteed that the `Allocation` resulting for this
    /// operand will be `preg`.
    #[inline(always)]
    pub fn reg_fixed_def(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Def,
            OperandPos::Late,
        )
    }

    /// Same as `reg_fixed_use` but at `OperandPos::Late`.
    #[inline(always)]
    pub fn reg_fixed_use_at_end(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Use,
            OperandPos::Late,
        )
    }

    /// Same as `reg_fixed_def` but at `OperandPos::Early`.
    #[inline(always)]
    pub fn reg_fixed_def_at_start(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Def,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a use of a vreg and places
    /// no constraints on its location (i.e., it can be allocated into
    /// either a register or on the stack).
    #[inline(always)]
    pub fn any_use(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Any,
            OperandKind::Use,
            OperandPos::Early,
        )
    }

    /// Create an `Operand` that designates a def of a vreg and places
    /// no constraints on its location (i.e., it can be allocated into
    /// either a register or on the stack).
    #[inline(always)]
    pub fn any_def(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Any,
            OperandKind::Def,
            OperandPos::Late,
        )
    }

    /// Create an `Operand` that always results in an assignment to the
    /// given fixed `preg`, *without* tracking liveranges in that
    /// `preg`. Must only be used for non-allocatable registers.
    #[inline(always)]
    pub fn fixed_nonallocatable(preg: PReg) -> Self {
        Operand::new(
            VReg::new(VReg::MAX, preg.class()),
            OperandConstraint::FixedReg(preg),
            OperandKind::Use,
            OperandPos::Early,
        )
    }

    /// Get the virtual register designated by an operand. Every
    /// operand must name some virtual register, even if it constrains
    /// the operand to a fixed physical register as well; the vregs
    /// are used to track dataflow.
    #[inline(always)]
    pub fn vreg(self) -> VReg {
        let vreg_idx = ((self.bits >> Self::VREG_SHIFT) & Self::VREG_MASK) as usize;
        VReg::new(vreg_idx, self.class())
    }

    /// Get the register class used by this operand.
    #[inline(always)]
    pub fn class(self) -> RegClass {
        let class_field = (self.bits >> Self::CLASS_SHIFT) & Self::CLASS_MASK;
        match class_field {
            0 => RegClass::Int,
            1 => RegClass::Float,
            2 => RegClass::Vector,
            _ => unreachable!(),
        }
    }

    /// Get the "kind" of this operand: a definition (write) or a use
    /// (read).
    #[inline(always)]
    pub fn kind(self) -> OperandKind {
        let kind_field = (self.bits >> Self::KIND_SHIFT) & Self::KIND_MASK;
        match kind_field {
            0 => OperandKind::Def,
            1 => OperandKind::Use,
            _ => unreachable!(),
        }
    }

    /// Get the "position" of this operand, i.e., where its read
    /// and/or write occurs: either before the instruction executes,
    /// or after it does. Ordinarily, uses occur at "before" and defs
    /// at "after", though there are cases where this is not true.
    #[inline(always)]
    pub fn pos(self) -> OperandPos {
        let pos_field = (self.bits >> Self::POS_SHIFT) & Self::POS_MASK;
        match pos_field {
            0 => OperandPos::Early,
            1 => OperandPos::Late,
            _ => unreachable!(),
        }
    }

    /// Get the "constraint" of this operand, i.e., what requirements
    /// its allocation must fulfill.
    #[inline(always)]
    pub fn constraint(self) -> OperandConstraint {
        let constraint_field =
            ((self.bits >> Self::CONSTRAINT_SHIFT) & Self::CONSTRAINT_MASK) as usize;
        if constraint_field & 0b1000000 != 0 {
            OperandConstraint::FixedReg(PReg::new(constraint_field & 0b0111111, self.class()))
        } else if constraint_field & 0b0100000 != 0 {
            OperandConstraint::Reuse(constraint_field & 0b0011111)
        } else if constraint_field & 0b0010000 != 0 {
            OperandConstraint::Limit(1 << (constraint_field & 0b0001111))
        } else {
            match constraint_field {
                0 => OperandConstraint::Any,
                1 => OperandConstraint::Reg,
                2 => OperandConstraint::Stack,
                _ => unreachable!(),
            }
        }
    }

    /// If this operand is for a fixed non-allocatable register (see
    /// [`Operand::fixed_nonallocatable`]), then returns the physical register that it will
    /// be assigned to.
    #[inline(always)]
    pub fn as_fixed_nonallocatable(self) -> Option<PReg> {
        match self.constraint() {
            OperandConstraint::FixedReg(preg) if self.vreg().vreg() == VReg::MAX => Some(preg),
            _ => None,
        }
    }

    /// Get the raw 64-bit encoding of this operand's fields.
    #[inline(always)]
    pub fn bits(self) -> u64 {
        self.bits
    }

    /// Construct an `Operand` from the raw 64-bit encoding returned
    /// from `bits()`.
    #[inline(always)]
    pub fn from_bits(bits: u64) -> Self {
        debug_assert_eq!(bits >> Self::TOTAL_BITS, 0);
        Operand { bits }
    }
}

impl core::fmt::Debug for Operand {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

impl core::fmt::Display for Operand {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if let Some(preg) = self.as_fixed_nonallocatable() {
            return write!(f, "Fixed: {preg}");
        }
        match (self.kind(), self.pos()) {
            (OperandKind::Def, OperandPos::Late) | (OperandKind::Use, OperandPos::Early) => {
                write!(f, "{:?}", self.kind())?;
            }
            _ => {
                write!(f, "{:?}@{:?}", self.kind(), self.pos())?;
            }
        }
        write!(
            f,
            ": {}{} {}",
            self.vreg(),
            match self.class() {
                RegClass::Int => "i",
                RegClass::Float => "f",
                RegClass::Vector => "v",
            },
            self.constraint()
        )
    }
}

/// An Allocation represents the end result of regalloc for an
/// Operand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Allocation {
    /// Bit-pack in 32 bits.
    ///
    /// kind:3 unused:1 index:28
    bits: u32,
}

impl core::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

impl core::fmt::Display for Allocation {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self.kind() {
            AllocationKind::None => write!(f, "none"),
            AllocationKind::Reg => write!(f, "{}", self.as_reg().unwrap()),
            AllocationKind::Stack => write!(f, "{}", self.as_stack().unwrap()),
        }
    }
}

impl Allocation {
    /// Construct a new Allocation.
    #[inline(always)]
    pub(crate) fn new(kind: AllocationKind, index: usize) -> Self {
        debug_assert!(index < (1 << 28));
        Self {
            bits: ((kind as u8 as u32) << 29) | (index as u32),
        }
    }

    /// Get the "none" allocation, which is distinct from the other
    /// possibilities and is used to initialize data structures.
    #[inline(always)]
    pub fn none() -> Allocation {
        Allocation::new(AllocationKind::None, 0)
    }

    /// Create an allocation into a register.
    #[inline(always)]
    pub fn reg(preg: PReg) -> Allocation {
        Allocation::new(AllocationKind::Reg, preg.index())
    }

    /// Create an allocation into a spillslot.
    #[inline(always)]
    pub fn stack(slot: SpillSlot) -> Allocation {
        Allocation::new(AllocationKind::Stack, slot.bits as usize)
    }

    /// Get the allocation's "kind": none, register, or stack (spillslot).
    #[inline(always)]
    pub fn kind(self) -> AllocationKind {
        match (self.bits >> 29) & 7 {
            0 => AllocationKind::None,
            1 => AllocationKind::Reg,
            2 => AllocationKind::Stack,
            _ => unreachable!(),
        }
    }

    /// Is the allocation "none"?
    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.kind() == AllocationKind::None
    }

    /// Is the allocation not "none"?
    #[inline(always)]
    pub fn is_some(self) -> bool {
        self.kind() != AllocationKind::None
    }

    /// Is the allocation a register?
    #[inline(always)]
    pub fn is_reg(self) -> bool {
        self.kind() == AllocationKind::Reg
    }

    /// Is the allocation on the stack (a spillslot)?
    #[inline(always)]
    pub fn is_stack(self) -> bool {
        self.kind() == AllocationKind::Stack
    }

    /// Get the index of the spillslot or register. If register, this
    /// is an index that can be used by `PReg::from_index()`.
    #[inline(always)]
    pub fn index(self) -> usize {
        (self.bits & ((1 << 28) - 1)) as usize
    }

    /// Get the allocation as a physical register, if any.
    #[inline(always)]
    pub fn as_reg(self) -> Option<PReg> {
        if self.kind() == AllocationKind::Reg {
            Some(PReg::from_index(self.index()))
        } else {
            None
        }
    }

    /// Get the allocation as a spillslot, if any.
    #[inline(always)]
    pub fn as_stack(self) -> Option<SpillSlot> {
        if self.kind() == AllocationKind::Stack {
            Some(SpillSlot {
                bits: self.index() as u32,
            })
        } else {
            None
        }
    }

    /// Get the raw bits for the packed encoding of this allocation.
    #[inline(always)]
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Construct an allocation from its packed encoding.
    #[inline(always)]
    pub fn from_bits(bits: u32) -> Self {
        debug_assert!(bits >> 29 >= 5);
        Self { bits }
    }
}

/// An allocation is one of two "kinds" (or "none"): register or
/// spillslot/stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AllocationKind {
    None = 0,
    Reg = 1,
    Stack = 2,
}

/// A position before or after an instruction at which we can make an
/// edit.
///
/// Note that this differs from `OperandPos` in that the former
/// describes specifically a constraint on an operand, while this
/// describes a program point. `OperandPos` could grow more options in
/// the future, for example if we decide that an "early write" or
/// "late read" phase makes sense, while `InstPosition` will always
/// describe these two insertion points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum InstPosition {
    Before = 0,
    After = 1,
}

/// A program point: a single point before or after a given instruction.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgPoint {
    bits: u32,
}

impl core::fmt::Debug for ProgPoint {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "progpoint{}{}",
            self.inst().index(),
            match self.pos() {
                InstPosition::Before => "-pre",
                InstPosition::After => "-post",
            }
        )
    }
}

impl ProgPoint {
    /// Create a new ProgPoint before or after the given instruction.
    #[inline(always)]
    pub fn new(inst: Inst, pos: InstPosition) -> Self {
        let bits = ((inst.0 as u32) << 1) | (pos as u8 as u32);
        Self { bits }
    }

    /// Create a new ProgPoint before the given instruction.
    #[inline(always)]
    pub fn before(inst: Inst) -> Self {
        Self::new(inst, InstPosition::Before)
    }

    /// Create a new ProgPoint after the given instruction.
    #[inline(always)]
    pub fn after(inst: Inst) -> Self {
        Self::new(inst, InstPosition::After)
    }

    /// Get the instruction that this ProgPoint is before or after.
    #[inline(always)]
    pub fn inst(self) -> Inst {
        // Cast to i32 to do an arithmetic right-shift, which will
        // preserve an `Inst::invalid()` (which is -1, or all-ones).
        Inst::new(((self.bits as i32) >> 1) as usize)
    }

    /// Get the "position" (Before or After) relative to the
    /// instruction.
    #[inline(always)]
    pub fn pos(self) -> InstPosition {
        match self.bits & 1 {
            0 => InstPosition::Before,
            1 => InstPosition::After,
            _ => unreachable!(),
        }
    }

    /// Get the "next" program point: for After, this is the Before of
    /// the next instruction, while for Before, this is After of the
    /// same instruction.
    #[inline(always)]
    pub fn next(self) -> ProgPoint {
        Self {
            bits: self.bits + 1,
        }
    }

    /// Get the "previous" program point, the inverse of `.next()`
    /// above.
    #[inline(always)]
    pub fn prev(self) -> ProgPoint {
        Self {
            bits: self.bits - 1,
        }
    }

    /// Convert to a raw encoding in 32 bits.
    #[inline(always)]
    pub fn to_index(self) -> u32 {
        self.bits
    }

    /// Construct from the raw 32-bit encoding.
    #[inline(always)]
    pub fn from_index(index: u32) -> Self {
        Self { bits: index }
    }

    #[inline(always)]
    pub fn invalid() -> Self {
        Self::before(Inst::new(usize::MAX))
    }
}

macro_rules! define_index {
    ($ix:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $ix(pub u32);
        impl $ix {
            #[inline(always)]
            pub fn new(i: usize) -> Self {
                Self(i as u32)
            }
            #[inline(always)]
            pub fn index(self) -> usize {
                debug_assert!(self.is_valid());
                self.0 as usize
            }
            #[inline(always)]
            pub fn invalid() -> Self {
                Self(u32::MAX)
            }
            #[inline(always)]
            pub fn is_invalid(self) -> bool {
                self == Self::invalid()
            }
            #[inline(always)]
            pub fn is_valid(self) -> bool {
                self != Self::invalid()
            }
            #[inline(always)]
            pub fn next(self) -> $ix {
                debug_assert!(self.is_valid());
                Self(self.0 + 1)
            }
            #[inline(always)]
            pub fn prev(self) -> $ix {
                debug_assert!(self.is_valid());
                Self(self.0 - 1)
            }

            #[inline(always)]
            pub fn raw_u32(self) -> u32 {
                self.0
            }
        }

        impl $crate::jit::backend::reg::ContainerIndex for $ix {}
    };
}

pub trait ContainerIndex: Clone + Copy + core::fmt::Debug + PartialEq + Eq {}

pub trait ContainerComparator {
    type Ix: ContainerIndex;
    fn compare(&self, a: Self::Ix, b: Self::Ix) -> core::cmp::Ordering;
}

define_index!(Inst);
define_index!(Block);

#[derive(Clone, Copy, Debug)]
pub struct InstRange(Inst, Inst);

impl InstRange {
    #[inline(always)]
    pub fn new(from: Inst, to: Inst) -> Self {
        debug_assert!(from.index() <= to.index());
        InstRange(from, to)
    }

    #[inline(always)]
    pub fn first(self) -> Inst {
        debug_assert!(self.len() > 0);
        self.0
    }

    #[inline(always)]
    pub fn last(self) -> Inst {
        debug_assert!(self.len() > 0);
        self.1.prev()
    }

    #[inline(always)]
    pub fn rest(self) -> InstRange {
        debug_assert!(self.len() > 0);
        InstRange::new(self.0.next(), self.1)
    }

    #[inline(always)]
    pub fn len(self) -> usize {
        self.1.index() - self.0.index()
    }

    #[inline(always)]
    pub fn iter(self) -> impl DoubleEndedIterator<Item = Inst> {
        (self.0.index()..self.1.index()).map(|i| Inst::new(i))
    }
}
