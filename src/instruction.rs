//! The bytecode instruction word.
//!
//! An instruction is a 64-bit word with a fixed field layout, *not* a Rust
//! enum. The layout is the same for every opcode:
//!
//! ```text
//! byte:  0    1    2    3    4    5    6    7
//!       op    a    b    c   [------ ext ------]
//!                 [-- h --]  ext is either d:u16 @4 + e:u16 @6, or imm:i32 @4
//! ```
//!
//! `h` is `b` and `c` read as one `u16`.
//!
//! Every opcode's operands are a prefix of `a, b, c` plus at most one use of
//! the extension word, so a handler reaches any field with one shift and mask
//! off a value it already holds in a register. A Rust enum cannot do that: it
//! has padding bytes (so the word can't be loaded or copied as an integer),
//! and passing one by value to a tail-called handler makes LLVM re-materialize
//! the whole word from the destructured fields — 6-9 wasted instructions on
//! every slow-path transition.
//!
//! Opcodes are declared once, in the [`instructions!`] table at the bottom of
//! this file, which generates [`Op`], the typed constructors, and the shape
//! each opcode uses. The interpreter's handler array is indexed by `Op`, so
//! the table is also what fixes the dispatch numbering.

use std::fmt;

use crate::dmm::Mutation;
use crate::env::value::Value;

/// A register index. Canonical home for what the compiler calls
/// `RegisterIndex`, so emitter code can pass one straight to a constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

/// An index into the enclosing closure's upvalue array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UpIdx(pub u8);

/// An index into the prototype's constant pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KIdx(pub u16);

/// An index into the prototype's `ic_table`. One slot is allocated per emitted
/// GETTABUP/SETTABUP/GETFIELD/SETFIELD/SELF; sites are not deduped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IcIdx(pub u16);

/// An index into the prototype's child-prototype list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProtoIdx(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateIdx(pub u16);

/// A number packed into an immediate opcode's 32-bit slot. Bit 0 set: a 31-bit
/// integer stored as `n << 1 | 1`. Bit 0 clear: an `f32` bit pattern, so only
/// floats whose low mantissa bit is clear qualify (`0.5`, `2.0`; not `0.1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Imm(u32);

impl Imm {
    pub const INT_MIN: i64 = -(1 << 30);
    pub const INT_MAX: i64 = (1 << 30) - 1;

    pub fn from_int(n: i64) -> Option<Imm> {
        if (Self::INT_MIN..=Self::INT_MAX).contains(&n) {
            Some(Imm(((n as i32) << 1 | 1) as u32))
        } else {
            None
        }
    }

    /// Exactly representable as an `f32` whose low mantissa bit is clear.
    /// Compared by bits so `-0.0` keeps its sign and NaN is rejected.
    pub fn from_float(f: f64) -> Option<Imm> {
        let bits = (f as f32).to_bits();
        if bits & 1 == 0 && f.is_finite() && (f32::from_bits(bits) as f64).to_bits() == f.to_bits()
        {
            Some(Imm(bits))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn is_int(self) -> bool {
        self.0 & 1 != 0
    }

    #[inline(always)]
    pub fn int(self) -> i64 {
        ((self.0 as i32) >> 1) as i64
    }

    #[inline(always)]
    pub fn float(self) -> f64 {
        f32::from_bits(self.0) as f64
    }
}

/// A number literal packed into an immediate compare's 16-bit slot as
/// `n << 1 | is_float`, `n` a 15-bit integer. A float qualifies when it is
/// integral and not `-0.0`; the flag only tells metamethods and errors which
/// type the literal was (Lua's `isSCnumber`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CmpImm(u16);

/// A branch offset: 24 bits, sign-extended, held in bits 40..63 of every
/// branch opcode whatever its shape, so a reader never asks the
/// shape. Constructed in range; the compiler patches offsets through
/// `set_branch_offset`, which reports one that does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offset(pub i32);

impl CmpImm {
    pub const MIN: i64 = -(1 << 14);
    pub const MAX: i64 = (1 << 14) - 1;

    pub fn from_int(n: i64) -> Option<CmpImm> {
        (Self::MIN..=Self::MAX)
            .contains(&n)
            .then_some(CmpImm(((n as i16) << 1) as u16))
    }

    pub fn from_float(f: f64) -> Option<CmpImm> {
        if f.fract() != 0.0 || (f == 0.0 && f.is_sign_negative()) {
            return None;
        }
        let n = f as i64;
        (Self::MIN..=Self::MAX)
            .contains(&n)
            .then_some(CmpImm(((n as i16) << 1 | 1) as u16))
    }

    #[inline(always)]
    pub fn int(self) -> i64 {
        (self.0 as i16 >> 1) as i64
    }

    #[inline(always)]
    pub fn is_float(self) -> bool {
        self.0 & 1 != 0
    }
}

/// Something that can occupy an operand slot. Implemented for the index
/// newtypes and for the raw types used by count/flag operands; the *declared*
/// type of each field is what a constructor takes, so two same-width operands
/// (`IcIdx` and `KIdx`, say) can't be passed in the wrong order.
pub trait Operand: Copy {
    fn bits(self) -> u64;
}

macro_rules! impl_operand {
    ($($ty:ty => |$s:ident| $e:expr),* $(,)?) => {
        $(impl Operand for $ty {
            #[inline(always)]
            fn bits($s) -> u64 { $e }
        })*
    };
}

impl_operand! {
    Reg      => |self| self.0 as u64,
    UpIdx    => |self| self.0 as u64,
    KIdx     => |self| self.0 as u64,
    IcIdx    => |self| self.0 as u64,
    ProtoIdx => |self| self.0 as u64,
    TemplateIdx => |self| self.0 as u64,
    u8       => |self| self as u64,
    u16      => |self| self as u64,
    bool     => |self| self as u64,
    i32      => |self| self as u32 as u64,
    Imm      => |self| self.0 as u64,
    CmpImm   => |self| self.0 as u64,
    // Placed in the top 24 bits of the 32-bit immediate slot.
    Offset   => |self| {
        debug_assert!(Instruction::fits_imm24(self.0));
        ((self.0 as u32 as u64) & 0xff_ffff) << 8
    },
}

/// Which operand slots an opcode uses. Named after the slots themselves:
/// `Abde` is `a`, `b`, `d`, `e`; `AImm` is `a` plus the 32-bit extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Nil,
    A,
    Ab,
    Abc,
    Ad,
    Abd,
    Abde,
    AImm,
    AbImm,
    AhImm,
    AbcImm,
    Imm,
}

const A_SHIFT: u32 = 8;
const B_SHIFT: u32 = 16;
const H_SHIFT: u32 = 16;
const C_SHIFT: u32 = 24;
const D_SHIFT: u32 = 32;
const E_SHIFT: u32 = 48;
const IMM_SHIFT: u32 = 32;
/// The 24-bit branch offset of an `AhImm` word.
const IMM24_SHIFT: u32 = 40;

/// Packers, one per [`Shape`]. Named to match the shape so the table can
/// select one by pasting the shape token.
pub mod shape {
    use super::{
        A_SHIFT, B_SHIFT, C_SHIFT, D_SHIFT, E_SHIFT, H_SHIFT, IMM_SHIFT, Instruction, Op, Operand,
    };

    #[inline(always)]
    fn slot8(v: u64) -> u64 {
        debug_assert!(v <= u8::MAX as u64, "operand does not fit an 8-bit slot");
        v & 0xff
    }

    #[inline(always)]
    fn slot16(v: u64) -> u64 {
        debug_assert!(v <= u16::MAX as u64, "operand does not fit a 16-bit slot");
        v & 0xffff
    }

    pub struct Nil;
    pub struct A;
    pub struct Ab;
    pub struct Abc;
    pub struct Ad;
    pub struct Abd;
    pub struct Abde;
    pub struct AImm;
    pub struct AbImm;
    pub struct AhImm;
    pub struct AbcImm;
    pub struct Imm;

    impl Nil {
        #[inline(always)]
        pub fn pack(op: Op) -> Instruction {
            Instruction(op as u64)
        }
    }

    impl A {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand) -> Instruction {
            Instruction(op as u64 | slot8(a.bits()) << A_SHIFT)
        }
    }

    impl Ab {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, b: impl Operand) -> Instruction {
            Instruction(op as u64 | slot8(a.bits()) << A_SHIFT | slot8(b.bits()) << B_SHIFT)
        }
    }

    impl Abc {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, b: impl Operand, c: impl Operand) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot8(b.bits()) << B_SHIFT
                    | slot8(c.bits()) << C_SHIFT,
            )
        }
    }

    impl Ad {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, d: impl Operand) -> Instruction {
            Instruction(op as u64 | slot8(a.bits()) << A_SHIFT | slot16(d.bits()) << D_SHIFT)
        }
    }

    impl Abd {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, b: impl Operand, d: impl Operand) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot8(b.bits()) << B_SHIFT
                    | slot16(d.bits()) << D_SHIFT,
            )
        }
    }

    impl Abde {
        #[inline(always)]
        pub fn pack(
            op: Op,
            a: impl Operand,
            b: impl Operand,
            d: impl Operand,
            e: impl Operand,
        ) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot8(b.bits()) << B_SHIFT
                    | slot16(d.bits()) << D_SHIFT
                    | slot16(e.bits()) << E_SHIFT,
            )
        }
    }

    impl AImm {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, imm: impl Operand) -> Instruction {
            Instruction(
                op as u64 | slot8(a.bits()) << A_SHIFT | (imm.bits() & 0xffff_ffff) << IMM_SHIFT,
            )
        }
    }

    impl AbImm {
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, b: impl Operand, imm: impl Operand) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot8(b.bits()) << B_SHIFT
                    | (imm.bits() & 0xffff_ffff) << IMM_SHIFT,
            )
        }
    }

    impl AhImm {
        /// The immediate slot holds an `Offset` (bits 40..63); bits 32..34
        /// are the site's adaptive bits.
        #[inline(always)]
        pub fn pack(op: Op, a: impl Operand, h: impl Operand, imm: impl Operand) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot16(h.bits()) << H_SHIFT
                    | (imm.bits() & 0xffff_ffff) << IMM_SHIFT,
            )
        }
    }

    impl AbcImm {
        #[inline(always)]
        pub fn pack(
            op: Op,
            a: impl Operand,
            b: impl Operand,
            c: impl Operand,
            imm: impl Operand,
        ) -> Instruction {
            Instruction(
                op as u64
                    | slot8(a.bits()) << A_SHIFT
                    | slot8(b.bits()) << B_SHIFT
                    | slot8(c.bits()) << C_SHIFT
                    | (imm.bits() & 0xffff_ffff) << IMM_SHIFT,
            )
        }
    }

    impl Imm {
        #[inline(always)]
        pub fn pack(op: Op, imm: impl Operand) -> Instruction {
            Instruction(op as u64 | (imm.bits() & 0xffff_ffff) << IMM_SHIFT)
        }
    }
}

/// One bytecode instruction: an opcode byte plus its operand slots.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Instruction(u64);

impl Instruction {
    /// The raw opcode byte — what the interpreter indexes its handler array
    /// with, and the only field read on the dispatch path.
    #[inline(always)]
    pub fn opcode(self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    pub fn op(self) -> Op {
        debug_assert!(
            (self.opcode() as usize) < Op::COUNT,
            "opcode byte out of range"
        );
        // Every word originates from a constructor below, so the byte is always
        // a declared discriminant; `Op` is `repr(u8)` and contiguous from 0.
        unsafe { std::mem::transmute::<u8, Op>(self.opcode()) }
    }

    #[inline(always)]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// A word that is no instruction: the interpreter passes a return's
    /// value count to a continuation in the instruction register.
    #[inline(always)]
    pub(crate) const fn from_raw(raw: u64) -> Self {
        Instruction(raw)
    }

    // --- slot reads -------------------------------------------------------
    //
    // Deliberately unchecked against the opcode: a handler reached through
    // dispatch already knows which opcode it is. The tuple accessors below
    // carry a debug-only shape check for readers that match on `op()`.

    #[inline(always)]
    pub fn a(self) -> u8 {
        (self.0 >> A_SHIFT) as u8
    }

    #[inline(always)]
    pub fn b(self) -> u8 {
        (self.0 >> B_SHIFT) as u8
    }

    #[inline(always)]
    pub fn c(self) -> u8 {
        (self.0 >> C_SHIFT) as u8
    }

    #[inline(always)]
    pub fn d(self) -> u16 {
        (self.0 >> D_SHIFT) as u16
    }

    #[inline(always)]
    pub fn e(self) -> u16 {
        (self.0 >> E_SHIFT) as u16
    }

    #[inline(always)]
    pub fn imm(self) -> i32 {
        (self.0 >> IMM_SHIFT) as u32 as i32
    }

    /// The 24-bit field at bits 40..63, sign-extended: every branch offset.
    #[inline(always)]
    pub fn imm24(self) -> i32 {
        ((self.0 as i64) >> IMM24_SHIFT) as i32
    }

    /// Whether `v` fits a branch offset.
    pub const fn fits_imm24(v: i32) -> bool {
        v >= -(1 << 23) && v < (1 << 23)
    }

    /// The branch offset of this word: the same field in every shape.
    #[inline(always)]
    pub fn branch_offset(self) -> i32 {
        self.imm24()
    }

    /// The packed [`Imm`] of an immediate-operand opcode.
    #[inline(always)]
    pub fn imm_k(self) -> Imm {
        Imm((self.0 >> IMM_SHIFT) as u32)
    }

    // Decoded off the whole word: one `tbnz` / one `asr`, no 32-bit extract first.

    #[inline(always)]
    pub fn imm_is_int(self) -> bool {
        self.0 & (1 << IMM_SHIFT) != 0
    }

    #[inline(always)]
    pub fn imm_int(self) -> i64 {
        (self.0 as i64) >> (IMM_SHIFT + 1)
    }

    #[inline(always)]
    pub fn imm_float(self) -> f64 {
        f32::from_bits((self.0 >> IMM_SHIFT) as u32) as f64
    }

    /// The `Value` an immediate stands for; slow paths hand it to
    /// metamethods and error messages.
    #[inline]
    pub fn imm_value<'gc>(self, mc: &Mutation<'gc>) -> Value<'gc> {
        if self.imm_is_int() {
            Value::integer(mc, self.imm_int())
        } else {
            Value::float(self.imm_float())
        }
    }

    #[inline(always)]
    pub fn h(self) -> u16 {
        (self.0 >> H_SHIFT) as u16
    }

    /// The [`CmpImm`] of an immediate compare.
    #[inline(always)]
    pub fn cmp_imm(self) -> CmpImm {
        CmpImm(self.h())
    }

    /// `cmp_imm().int()` off the whole word, one `sbfx`.
    #[inline(always)]
    pub fn cmp_imm_int(self) -> i64 {
        ((self.0 << (64 - H_SHIFT - 16)) as i64) >> (64 - 15)
    }

    /// The `Value` an immediate compare's literal stands for.
    #[inline]
    pub fn cmp_imm_value<'gc>(self, mc: &Mutation<'gc>) -> Value<'gc> {
        let k = self.cmp_imm();
        if k.is_float() {
            Value::float(k.int() as f64)
        } else {
            Value::integer(mc, k.int())
        }
    }

    #[inline(always)]
    fn expect(self, shape: Shape) {
        debug_assert_eq!(
            self.op().shape(),
            shape,
            "{:?} is not {shape:?}-shaped",
            self.op()
        );
    }

    #[inline(always)]
    pub fn ab(self) -> (u8, u8) {
        self.expect(Shape::Ab);
        (self.a(), self.b())
    }

    /// `a`, `b`, `c`; also of an `AbcImm` word, whose first three slots
    /// are the same (a CALLS read as a CALL).
    #[inline(always)]
    pub fn abc(self) -> (u8, u8, u8) {
        debug_assert!(
            matches!(self.op().shape(), Shape::Abc | Shape::AbcImm),
            "{:?} is not Abc-shaped",
            self.op()
        );
        (self.a(), self.b(), self.c())
    }

    #[inline(always)]
    pub fn ad(self) -> (u8, u16) {
        self.expect(Shape::Ad);
        (self.a(), self.d())
    }

    #[inline(always)]
    pub fn abd(self) -> (u8, u8, u16) {
        self.expect(Shape::Abd);
        (self.a(), self.b(), self.d())
    }

    #[inline(always)]
    pub fn abde(self) -> (u8, u8, u16, u16) {
        self.expect(Shape::Abde);
        (self.a(), self.b(), self.d(), self.e())
    }

    #[inline(always)]
    pub fn a_imm(self) -> (u8, i32) {
        self.expect(Shape::AImm);
        (self.a(), self.imm())
    }

    /// `AbcImm` slots of an immediate arithmetic op: `dst`, `src`, and the
    /// source-order-flipped flag.
    #[inline(always)]
    pub fn abc_imm(self) -> (u8, u8, bool) {
        self.expect(Shape::AbcImm);
        (self.a(), self.b(), self.c() & 1 != 0)
    }

    #[inline(always)]
    pub fn ab_imm(self) -> (u8, u8, i32) {
        self.expect(Shape::AbImm);
        (self.a(), self.b(), self.imm())
    }

    /// `a` and the branch offset of an `AImm` branch.
    #[inline(always)]
    pub fn a_offset(self) -> (u8, i32) {
        self.expect(Shape::AImm);
        (self.a(), self.imm24())
    }

    /// `a`, `b` and the branch offset of an `AbImm` branch.
    #[inline(always)]
    pub fn ab_offset(self) -> (u8, u8, i32) {
        self.expect(Shape::AbImm);
        (self.a(), self.b(), self.imm24())
    }

    #[inline(always)]
    pub fn ah_imm(self) -> (u8, u16, i32) {
        self.expect(Shape::AhImm);
        (self.a(), self.h(), self.imm24())
    }

    // --- slot writes ------------------------------------------------------
    //
    // The emitter patches instructions already on the tape: jump offsets once
    // the target is known, and the destination register of an arithmetic
    // instruction once the operand temps have been freed.

    #[inline(always)]
    fn set_slot(&mut self, shift: u32, mask: u64, v: u64) {
        self.0 = (self.0 & !(mask << shift)) | ((v & mask) << shift);
    }

    #[inline(always)]
    pub fn set_a(&mut self, v: u8) {
        self.set_slot(A_SHIFT, 0xff, v as u64);
    }

    #[inline(always)]
    pub fn set_b(&mut self, v: u8) {
        self.set_slot(B_SHIFT, 0xff, v as u64);
    }

    #[inline(always)]
    pub fn set_c(&mut self, v: u8) {
        self.set_slot(C_SHIFT, 0xff, v as u64);
    }

    #[inline(always)]
    pub fn set_d(&mut self, v: u16) {
        self.set_slot(D_SHIFT, 0xffff, v as u64);
    }

    #[inline(always)]
    pub fn set_e(&mut self, v: u16) {
        self.set_slot(E_SHIFT, 0xffff, v as u64);
    }

    #[inline(always)]
    pub fn set_imm(&mut self, v: i32) {
        self.set_slot(IMM_SHIFT, 0xffff_ffff, v as u32 as u64);
    }

    /// Set the 24-bit branch offset.
    #[inline(always)]
    pub fn set_imm24(&mut self, v: i32) {
        debug_assert!(Self::fits_imm24(v));
        self.set_slot(IMM24_SHIFT, 0xff_ffff, v as u32 as u64);
    }

    /// Set this branch's offset; `false` when it does not fit 24 bits.
    pub fn set_branch_offset(&mut self, v: i32) -> bool {
        if !Self::fits_imm24(v) {
            return false;
        }
        self.set_imm24(v);
        true
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = self.op();
        write!(f, "{}", op.name())?;
        match op.shape() {
            Shape::Nil => Ok(()),
            Shape::A => write!(f, "(a={})", self.a()),
            Shape::Ab => write!(f, "(a={}, b={})", self.a(), self.b()),
            Shape::Abc => write!(f, "(a={}, b={}, c={})", self.a(), self.b(), self.c()),
            Shape::Ad => write!(f, "(a={}, d={})", self.a(), self.d()),
            Shape::Abd => write!(f, "(a={}, b={}, d={})", self.a(), self.b(), self.d()),
            Shape::Abde => write!(
                f,
                "(a={}, b={}, d={}, e={})",
                self.a(),
                self.b(),
                self.d(),
                self.e()
            ),
            Shape::AImm => write!(f, "(a={}, imm={})", self.a(), self.imm()),
            Shape::AbImm => write!(f, "(a={}, b={}, imm={})", self.a(), self.b(), self.imm()),
            Shape::AhImm => write!(f, "(a={}, h={}, imm={})", self.a(), self.h(), self.imm24()),
            Shape::AbcImm => write!(
                f,
                "(a={}, b={}, c={}, imm={:?})",
                self.a(),
                self.b(),
                self.c(),
                self.imm_k()
            ),
            Shape::Imm => write!(f, "(imm={})", self.imm()),
        }
    }
}

/// The ISA table: one row per opcode, giving its dispatch number, name,
/// constructor, operand shape, and the name and type of each operand.
///
/// Adding an opcode here gives you the `Op` variant, the typed constructor and
/// the shape metadata; the interpreter builds its handler array with
/// [`Op::table`], which requires exactly one row per opcode, so a missing
/// handler is a build error.
macro_rules! instructions {
    ($(
        $(#[$meta:meta])*
        $num:literal $op:ident $ctor:ident $shape:ident { $($field:ident : $fty:ty),* $(,)? }
    )*) => {
        /// Opcode numbers. Contiguous from zero — the interpreter indexes its
        /// handler array with `op as usize`.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        pub enum Op {
            $($(#[$meta])* $op = $num,)*
        }

        impl Op {
            pub const ALL: &'static [Op] = &[$(Op::$op),*];
            pub const COUNT: usize = Self::ALL.len();

            pub fn name(self) -> &'static str {
                match self { $(Op::$op => stringify!($op),)* }
            }

            /// Which operand slots this opcode uses.
            pub fn shape(self) -> Shape {
                match self { $(Op::$op => Shape::$shape,)* }
            }

            /// A table from groups of rows (the families' and the singles'),
            /// every opcode listed exactly once; `fill` is overwritten.
            pub const fn table_of<T: Copy>(fill: T, groups: &[&[(Op, T)]]) -> [T; Op::COUNT] {
                let mut t = [fill; Op::COUNT];
                let mut seen = [false; Op::COUNT];
                let mut n = 0;
                let mut g = 0;
                while g < groups.len() {
                    let rows = groups[g];
                    let mut i = 0;
                    while i < rows.len() {
                        let (op, v) = rows[i];
                        assert!(!seen[op as usize], "opcode listed twice");
                        seen[op as usize] = true;
                        t[op as usize] = v;
                        n += 1;
                        i += 1;
                    }
                    g += 1;
                }
                assert!(n == Op::COUNT, "an opcode has no handler");
                t
            }

            pub const fn table<T: Copy>(rows: [(Op, T); Op::COUNT]) -> [T; Op::COUNT] {
                let mut t = [rows[0].1; Op::COUNT];
                let mut seen = [false; Op::COUNT];
                let mut i = 0;
                while i < rows.len() {
                    let (op, v) = rows[i];
                    assert!(!seen[op as usize], "opcode listed twice");
                    seen[op as usize] = true;
                    t[op as usize] = v;
                    i += 1;
                }
                t
            }
        }

        /// `Op::ALL[i] as u8 == i`, which `Instruction::op` and the handler
        /// array both rely on. A row inserted with a stale number breaks the
        /// build here rather than dispatching to the wrong handler.
        const _: () = {
            let mut i = 0;
            while i < Op::COUNT {
                assert!(Op::ALL[i] as usize == i, "opcode numbers must be contiguous from 0");
                i += 1;
            }
        };

        impl Instruction {
            $(
                $(#[$meta])*
                #[inline(always)]
                pub fn $ctor($($field: $fty),*) -> Self {
                    shape::$shape::pack(Op::$op $(, $field)*)
                }
            )*
        }
    };
}

instructions! {
    0x00 MOVE       mov         Ab    { dst: Reg, src: Reg }
    0x01 LOAD       load        Ad    { dst: Reg, idx: KIdx }
    0x02 LFALSESKIP lfalseskip  A     { src: Reg }

    /// Upvalue reads take a by-value upvalue (`UpValueDescriptor::by_value`);
    /// the assembler rewrites them to the `_REF` forms for the others.
    /// SETUPVAL's upvalue is never by value.
    0x03 GETUPVAL   getupval    Ab    { dst: Reg, idx: UpIdx }
    0x04 SETUPVAL   setupval    Ab    { src: Reg, idx: UpIdx }
    0x05 GETTABUP   gettabup    Abde  { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x06 SETTABUP   settabup    Abde  { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x07 GETTABLE   gettable    Abc   { dst: Reg, table: Reg, key: Reg }
    0x08 SETTABLE   settable    Abc   { src: Reg, table: Reg, key: Reg }
    0x09 GETFIELD   getfield    Abde  { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x0a SETFIELD   setfield    Abde  { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }

    /// Method-call setup: `R[dst] = R[object][K[key_idx]]; R[dst+1] = R[object]`.
    /// Backs `obj:m(...)` codegen.
    0x0b SELF       self_       Abde  { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }

    /// `R[dst]` = a new table made from `templates[template]`.
    0x0c NEWTABLE   newtable    Ad    { dst: Reg, template: TemplateIdx }
    0x0d ADD        add         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x0e SUB        sub         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x0f MUL        mul         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x10 MOD        mod_        Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x11 POW        pow         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x12 DIV        div         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x13 IDIV       idiv        Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x14 BAND       band        Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x15 BOR        bor         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x16 BXOR       bxor        Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x17 SHL        shl         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x18 SHR        shr         Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x19 UNM        unm         Ab    { dst: Reg, src: Reg }
    0x1a BNOT       bnot        Ab    { dst: Reg, src: Reg }
    0x1b NOT        not         Ab    { dst: Reg, src: Reg }
    0x1c LEN        len         Ab    { dst: Reg, src: Reg }
    0x1d CONCAT     concat      Abc   { dst: Reg, lhs: Reg, rhs: Reg }
    0x1e CLOSE      close       A     { start: Reg }
    0x1f TBC        tbc         A     { val: Reg }
    0x20 JMP        jmp         Imm   { offset: Offset }

    // --- conditional branches -------------------------------------------------
    //
    // Each jumps `offset` past itself when its test holds (`J..`, `JT`,
    // `JTSET`) or when it doesn't (`JN..`, `JF`, `JFSET`); see
    // `Op::branch_sense`. `JNLT a b` is not `JLE b a`: they differ on NaN and
    // in the metamethod they call.

    0x21 JEQ        jeq         AbImm { lhs: Reg, rhs: Reg, offset: Offset }
    0x22 JNEQ       jneq        AbImm { lhs: Reg, rhs: Reg, offset: Offset }
    0x23 JLT        jlt         AbImm { lhs: Reg, rhs: Reg, offset: Offset }
    0x24 JNLT       jnlt        AbImm { lhs: Reg, rhs: Reg, offset: Offset }
    0x25 JLE        jle         AbImm { lhs: Reg, rhs: Reg, offset: Offset }
    0x26 JNLE       jnle        AbImm { lhs: Reg, rhs: Reg, offset: Offset }

    // `R[src] <cmp> imm`. `GT`/`GE` are the swapped `LT`/`LE`, so a literal on
    // either side compiles to one of these. Equality never consults `__eq`.

    0x27 JEQI       jeqi        AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x28 JNEQI      jneqi       AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x29 JLTI       jlti        AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2a JNLTI      jnlti       AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2b JLEI       jlei        AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2c JNLEI      jnlei       AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2d JGTI       jgti        AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2e JNGTI      jngti       AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x2f JGEI       jgei        AhImm { src: Reg, imm: CmpImm, offset: Offset }
    0x30 JNGEI      jngei       AhImm { src: Reg, imm: CmpImm, offset: Offset }

    // `R[src] == K[key]`, a string: identity, as strings are interned.

    0x31 JEQS       jeqs        AhImm { src: Reg, key: KIdx, offset: Offset }
    0x32 JNEQS      jneqs       AhImm { src: Reg, key: KIdx, offset: Offset }

    // Truthiness of `R[src]`; the `SET` forms copy it to `R[dst]` when they
    // jump (`a or b`).

    0x33 JT         jt          AImm  { src: Reg, offset: Offset }
    0x34 JF         jf          AImm  { src: Reg, offset: Offset }
    0x35 JTSET      jtset       AbImm { dst: Reg, src: Reg, offset: Offset }
    0x36 JFSET      jfset       AbImm { dst: Reg, src: Reg, offset: Offset }

    0x37 CALL       call        Abc   { func: Reg, args: u8, returns: u8 }
    0x38 TAILCALL   tailcall    Ab    { func: Reg, args: u8 }
    0x39 RETURN     ret         Ab    { values: Reg, count: u8 }
    0x3a FORLOOP    forloop     AImm  { base: Reg, offset: Offset }
    0x3b FORPREP    forprep     AImm  { base: Reg, offset: Offset }

    /// Generic `for`, over [`TFOR_VARS`] hidden slots at `base` (iterator,
    /// state, closing value, traversal position) and then its variables.
    0x3c TFORPREP   tforprep    AImm  { base: Reg, offset: Offset }
    0x3d TFORCALL   tforcall    Ab    { base: Reg, count: u8 }
    0x3e TFORLOOP   tforloop    AImm  { base: Reg, offset: Offset }

    0x3f SETLIST    setlist     Abd   { table: Reg, count: u8, offset: u16 }
    0x40 CLOSURE    closure     Ad    { dst: Reg, proto: ProtoIdx }
    0x41 VARARG     vararg      Ab    { dst: Reg, count: u8 }

    /// Optimized below-base read of an un-escaped named vararg: integer key
    /// `1..=num_extras`, `"n"` for the count, else nil. `base` is unused at
    /// run time but is the table operand when the epilogue rewrites this to
    /// `GETTABLE` for an escaped vararg. Lua 5.5 `OP_GETVARG`.
    0x42 VARARGGET  varargget   Abc   { dst: Reg, base: Reg, key: Reg }

    0x43 VARARGPREP varargprep  A     { num_fixed: u8 }
    0x44 ERRNNIL    errnnil     Ad    { src: Reg, name_key: KIdx }
    0x45 NOP        nop         Nil   { }
    0x46 STOP       stop        Nil   { }

    // --- immediate-operand forms --------------------------------------
    //
    // `R[dst] = R[src] <op> imm`; the `R`-prefixed forms compute `imm <op> R[src]`
    // (cf. ARM `rsb`) so no handler selects operands on its fast path. `flipped`
    // means the constant was on the left in the source (`1 + x`); only the slow
    // path reads it, to pass metamethods and errors the operands in source order.
    // Invariants the handlers rely on: `MODI`/`IDIVI` never carry a zero integer
    // immediate; the bitwise forms only carry integer immediates.

    0x47 ADDI       addi        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x48 SUBI       subi        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x49 MULI       muli        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4a MODI       modi        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4b POWI       powi        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4c DIVI       divi        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4d IDIVI      idivi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4e BANDI      bandi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x4f BORI       bori        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x50 BXORI      bxori       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x51 SHLI       shli        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x52 SHRI       shri        AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x53 RSUBI      rsubi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x54 RMODI      rmodi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x55 RPOWI      rpowi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x56 RDIVI      rdivi       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x57 RIDIVI     ridivi      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x58 RSHLI      rshli       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x59 RSHRI      rshri       AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }

    0x5a RETURN0    ret0        Nil   { }
    0x5b RETURN1    ret1        A     { value: Reg }

    // --- constant-key table forms --------------------------------------------
    //
    // Never emitted: the generic handler fills the site's inline cache and
    // rewrites the site to the form for what the entry holds: an own slot in
    // the table's cell (`_INL`) or its spill cell (`_AUX`), a key the shape
    // lacks (`_ABSENT`), a slot in the `__index` table (`_PROTO`), an added
    // key (`_TRANS`). Same operands as the generic form; `c` counts refills.

    0x5c GETFIELD_INL     getfield_inl    Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5d GETFIELD_AUX     getfield_aux    Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5e GETFIELD_ABSENT  getfield_absent Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5f GETFIELD_PROTO   getfield_proto  Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x60 GETTABUP_INL     gettabup_inl    Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x61 GETTABUP_AUX     gettabup_aux    Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x62 GETTABUP_ABSENT  gettabup_absent Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x63 GETTABUP_PROTO   gettabup_proto  Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x64 SELF_INL         self_inl        Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x65 SELF_AUX         self_aux        Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x66 SELF_ABSENT      self_absent     Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x67 SELF_PROTO       self_proto      Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x68 SETFIELD_INL     setfield_inl    Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x69 SETFIELD_AUX     setfield_aux    Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x6a SETFIELD_TRANS   setfield_trans  Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x6b SETFIELD_ABSENT  setfield_absent Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x6c SETTABUP_INL     settabup_inl    Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x6d SETTABUP_AUX     settabup_aux    Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x6e SETTABUP_TRANS   settabup_trans  Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x6f SETTABUP_ABSENT  settabup_absent Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }

    // --- shared-cell upvalue forms ------------------------------------------
    //
    // Never emitted: the assembler rewrites GETUPVAL, GETTABUP and SETTABUP
    // to these for an upvalue that is not by value. Not quickened.

    0x70 GETUPVAL_REF getupval_ref Ab   { dst: Reg, idx: UpIdx }
    0x71 GETTABUP_REF gettabup_ref Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x72 SETTABUP_REF settabup_ref Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }

    // --- CALL by result count -----------------------------------------------
    //
    // A CALL wanting no result (`returns` 1) or one (`returns` 2), whose
    // continuation is a constant. `returns` stays for the generic paths.

    0x73 CALL_R0    call_r0     Abc   { func: Reg, args: u8, returns: u8 }
    0x74 CALL_R1    call_r1     Abc   { func: Reg, args: u8, returns: u8 }

    // --- adaptive arithmetic -------------------------------------------------
    //
    // Never emitted: the generic handler of a binary arithmetic or bitwise
    // site rewrites it to the form for the operand kinds it saw, and back to
    // the generic opcode after three misses (locked). The forms keep the
    // generic operand layout; the adaptive bits (`misses`, `locked`) sit in
    // bits 32..34 of a register form and in `c` bits 1..3 of an immediate
    // form. See `OP_INFO`.

    /// A register-form binary op whose `lhs` is a table with the metamethod;
    /// the original opcode is in `e`'s low byte.
    0x75 ARITH_MM     arith_mm    Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    /// As `ARITH_MM`, the metamethod from `rhs`, a table, and `lhs` a number.
    0x76 ARITH_MM_R   arith_mm_r  Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    /// An immediate-form op whose register operand is a table with the
    /// metamethod: `c` bits 4..7 index `IMM_ARITH_OPS` for the original opcode.
    0x77 ARITH_MMI    arith_mmi   AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }

    /// `R[dst] = imm`, a small integer, and `R[dst .. dst+count) = nil`.
    0x78 LOADI        loadi       AImm   { dst: Reg, imm: i32 }
    0x79 LOADNIL      loadnil     Ab     { dst: Reg, count: u8 }

    // Register arithmetic: both small, both float, small/float, float/small.
    0x7a ADD_II       add_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x7b SUB_II       sub_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x7c MUL_II       mul_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x7d MOD_II       mod_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x7e IDIV_II      idiv_ii     Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x7f DIV_II       div_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x80 ADD_FF       add_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x81 SUB_FF       sub_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x82 MUL_FF       mul_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x83 MOD_FF       mod_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x84 POW_FF       pow_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x85 DIV_FF       div_ff      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x86 IDIV_FF      idiv_ff     Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x87 ADD_IF       add_if      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x88 SUB_IF       sub_if      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x89 MUL_IF       mul_if      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x8a DIV_IF       div_if      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x8b ADD_FI       add_fi      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x8c SUB_FI       sub_fi      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x8d MUL_FI       mul_fi      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x8e DIV_FI       div_fi      Abc    { dst: Reg, lhs: Reg, rhs: Reg }

    // Register bitwise: both small.
    0x8f BAND_II      band_ii     Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x90 BOR_II       bor_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x91 BXOR_II      bxor_ii     Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x92 SHL_II       shl_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0x93 SHR_II       shr_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }

    // Immediate arithmetic: small register with an integer immediate (`_I`),
    // float register (`_F`), small register with a float result (`_IF`).
    0x94 ADDI_I       addi_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x95 SUBI_I       subi_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x96 MULI_I       muli_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x97 MODI_I       modi_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x98 IDIVI_I      idivi_i     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x99 RSUBI_I      rsubi_i     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9a ADDI_F       addi_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9b SUBI_F       subi_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9c MULI_F       muli_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9d MODI_F       modi_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9e IDIVI_F      idivi_f     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0x9f RSUBI_F      rsubi_f     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa0 POWI_F       powi_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa1 DIVI_F       divi_f      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa2 RDIVI_F      rdivi_f     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa3 ADDI_IF      addi_if     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa4 SUBI_IF      subi_if     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa5 MULI_IF      muli_if     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa6 DIVI_IF      divi_if     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa7 RDIVI_IF     rdivi_if    AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }

    // Immediate bitwise: small register.
    0xa8 BANDI_I      bandi_i     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xa9 BORI_I       bori_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xaa BXORI_I      bxori_i     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xab SHLI_I       shli_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xac SHRI_I       shri_i      AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }

    // Power with small integers has a float result: forms beyond section 12.3.
    0xad POW_II       pow_ii      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xae POWI_IF      powi_if     AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }
    0xaf RPOWI_IF     rpowi_if    AbcImm { dst: Reg, src: Reg, flipped: bool, imm: Imm }

    // Any inline numbers, int-int with an integer result: what a register
    // site whose operand kinds keep changing takes instead of locking.
    0xb0 ADD_NN       add_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb1 SUB_NN       sub_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb2 MUL_NN       mul_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb3 MOD_NN       mod_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb4 POW_NN       pow_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb5 DIV_NN       div_nn      Abc    { dst: Reg, lhs: Reg, rhs: Reg }
    0xb6 IDIV_NN      idiv_nn     Abc    { dst: Reg, lhs: Reg, rhs: Reg }

    // Compares: both small for the register forms, a float register
    // for the immediate ones.
    0xb7 JLT_II       jlt_ii      AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xb8 JNLT_II      jnlt_ii     AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xb9 JLE_II       jle_ii      AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xba JNLE_II      jnle_ii     AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xbb JEQ_II       jeq_ii      AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xbc JNEQ_II      jneq_ii     AbImm  { lhs: Reg, rhs: Reg, offset: Offset }
    0xbd JLTI_F       jlti_f      AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xbe JNLTI_F      jnlti_f     AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xbf JLEI_F       jlei_f      AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xc0 JNLEI_F      jnlei_f     AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xc1 JGTI_F       jgti_f      AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xc2 JNGTI_F      jngti_f     AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xc3 JGEI_F       jgei_f      AhImm  { src: Reg, imm: CmpImm, offset: Offset }
    0xc4 JNGEI_F      jngei_f     AhImm  { src: Reg, imm: CmpImm, offset: Offset }

    // Loops: written by the prep instruction, guarded, no counter.
    0xc5 FORLOOP_I    forloop_i   AImm   { base: Reg, offset: Offset }
    0xc6 FORLOOP_F    forloop_f   AImm   { base: Reg, offset: Offset }
    0xc7 TFORCALL_NEXT   tforcall_next   Ab { base: Reg, count: u8 }
    0xc8 TFORCALL_IPAIRS tforcall_ipairs Ab { base: Reg, count: u8 }

    // CALL with the callee in `src` rather than `func`: the
    // compiler's fusion of the MOVE that fills the function slot of a call
    // of a local. The callee is read when the call runs, after the
    // arguments (the manual fixes no order between the two). `R[func]`
    // gets the callee first, so the slow paths see a plain CALL.
    0xc9 CALLS        calls       AbcImm { func: Reg, args: u8, returns: u8, src: Reg }
    0xca CALLS_R0     calls_r0    AbcImm { func: Reg, args: u8, returns: u8, src: Reg }
    0xcb CALLS_R1     calls_r1    AbcImm { func: Reg, args: u8, returns: u8, src: Reg }
}

/// The polymorphic family an opcode belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    None,
    RegArith,
    RegBit,
    ImmArith,
    ImmBit,
    /// Constant-key table access, specialized by the inline-cache fill.
    Field,
    /// Register compares and equality (`JLT`...), forms `[II, -, -, -, -]`.
    CmpReg,
    /// Immediate compares (`JLTI`...), forms `[-, F, -, -, -]`.
    CmpImm,
    /// Loop steps, specialized by their prep instruction.
    Loop,
}

/// The operation of an arithmetic or bitwise opcode, generic or specialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithKind {
    None,
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IDiv,
    BAnd,
    BOr,
    BXor,
    Shl,
    Shr,
}

/// What the generic handlers and the listing know about an opcode.
#[derive(Debug, Clone, Copy)]
pub struct OpInfo {
    /// The opcode the compiler emitted for this site.
    pub generic: Op,
    pub family: Family,
    pub kind: ArithKind,
    /// The immediate is the left operand (the `R` forms).
    pub reversed: bool,
    /// Bit position of the adaptive bits (`misses: u2`, `locked: u1`); 0 for
    /// an opcode without them.
    pub adaptive_shift: u8,
    /// The specialized forms by operand kinds: `[II, FF, IF, FI, NN]` for a
    /// register family (`NN`: any inline numbers, the form a site whose
    /// kinds keep changing takes instead of locking), `[I, F, IF, -, -]` for
    /// an immediate one.
    pub forms: [Option<Op>; 5],
}

/// Adaptive bits of an `Abc` site: bits 32..34 (the low bits of `d`).
pub const ADAPTIVE_ABC: u8 = 32;
/// Adaptive bits of an `AbcImm` site: `c` bits 1..3, beside `flipped`.
pub const ADAPTIVE_ABC_IMM: u8 = 25;
/// Adaptive bits of an `AbImm` compare site: the `c` byte.
pub const ADAPTIVE_AB_IMM: u8 = 24;
/// Adaptive bits of an `AhImm` compare site: bits 32..34, below the 24-bit
/// offset.
pub const ADAPTIVE_AH_IMM: u8 = 32;

/// Misses at which a site locks to its generic opcode.
pub const MISSES_TO_LOCK: u8 = 3;

/// The immediate arithmetic opcodes `ARITH_MMI` can stand for, indexed by
/// `c` bits 4..7 (padded to 16 so the index needs no bounds check).
pub const IMM_ARITH_OPS: [Op; 16] = [
    Op::ADDI,
    Op::SUBI,
    Op::MULI,
    Op::MODI,
    Op::POWI,
    Op::DIVI,
    Op::IDIVI,
    Op::RSUBI,
    Op::RMODI,
    Op::RPOWI,
    Op::RDIVI,
    Op::RIDIVI,
    Op::ADDI,
    Op::ADDI,
    Op::ADDI,
    Op::ADDI,
];
/// How many of [`IMM_ARITH_OPS`] are real.
pub const IMM_ARITH_COUNT: usize = 12;

const fn op_info(op: Op) -> OpInfo {
    match op {
        Op::ADD => OpInfo {
            generic: Op::ADD,
            family: Family::RegArith,
            kind: ArithKind::Add,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::ADD_II),
                Some(Op::ADD_FF),
                Some(Op::ADD_IF),
                Some(Op::ADD_FI),
                Some(Op::ADD_NN),
            ],
        },
        Op::SUB => OpInfo {
            generic: Op::SUB,
            family: Family::RegArith,
            kind: ArithKind::Sub,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::SUB_II),
                Some(Op::SUB_FF),
                Some(Op::SUB_IF),
                Some(Op::SUB_FI),
                Some(Op::SUB_NN),
            ],
        },
        Op::MUL => OpInfo {
            generic: Op::MUL,
            family: Family::RegArith,
            kind: ArithKind::Mul,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::MUL_II),
                Some(Op::MUL_FF),
                Some(Op::MUL_IF),
                Some(Op::MUL_FI),
                Some(Op::MUL_NN),
            ],
        },
        Op::MOD => OpInfo {
            generic: Op::MOD,
            family: Family::RegArith,
            kind: ArithKind::Mod,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::MOD_II),
                Some(Op::MOD_FF),
                None,
                None,
                Some(Op::MOD_NN),
            ],
        },
        Op::POW => OpInfo {
            generic: Op::POW,
            family: Family::RegArith,
            kind: ArithKind::Pow,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::POW_II),
                Some(Op::POW_FF),
                None,
                None,
                Some(Op::POW_NN),
            ],
        },
        Op::DIV => OpInfo {
            generic: Op::DIV,
            family: Family::RegArith,
            kind: ArithKind::Div,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::DIV_II),
                Some(Op::DIV_FF),
                Some(Op::DIV_IF),
                Some(Op::DIV_FI),
                Some(Op::DIV_NN),
            ],
        },
        Op::IDIV => OpInfo {
            generic: Op::IDIV,
            family: Family::RegArith,
            kind: ArithKind::IDiv,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [
                Some(Op::IDIV_II),
                Some(Op::IDIV_FF),
                None,
                None,
                Some(Op::IDIV_NN),
            ],
        },
        Op::BAND => OpInfo {
            generic: Op::BAND,
            family: Family::RegBit,
            kind: ArithKind::BAnd,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [Some(Op::BAND_II), None, None, None, None],
        },
        Op::BOR => OpInfo {
            generic: Op::BOR,
            family: Family::RegBit,
            kind: ArithKind::BOr,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [Some(Op::BOR_II), None, None, None, None],
        },
        Op::BXOR => OpInfo {
            generic: Op::BXOR,
            family: Family::RegBit,
            kind: ArithKind::BXor,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [Some(Op::BXOR_II), None, None, None, None],
        },
        Op::SHL => OpInfo {
            generic: Op::SHL,
            family: Family::RegBit,
            kind: ArithKind::Shl,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [Some(Op::SHL_II), None, None, None, None],
        },
        Op::SHR => OpInfo {
            generic: Op::SHR,
            family: Family::RegBit,
            kind: ArithKind::Shr,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [Some(Op::SHR_II), None, None, None, None],
        },
        Op::ADDI => OpInfo {
            generic: Op::ADDI,
            family: Family::ImmArith,
            kind: ArithKind::Add,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [
                Some(Op::ADDI_I),
                Some(Op::ADDI_F),
                Some(Op::ADDI_IF),
                None,
                None,
            ],
        },
        Op::SUBI => OpInfo {
            generic: Op::SUBI,
            family: Family::ImmArith,
            kind: ArithKind::Sub,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [
                Some(Op::SUBI_I),
                Some(Op::SUBI_F),
                Some(Op::SUBI_IF),
                None,
                None,
            ],
        },
        Op::MULI => OpInfo {
            generic: Op::MULI,
            family: Family::ImmArith,
            kind: ArithKind::Mul,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [
                Some(Op::MULI_I),
                Some(Op::MULI_F),
                Some(Op::MULI_IF),
                None,
                None,
            ],
        },
        Op::MODI => OpInfo {
            generic: Op::MODI,
            family: Family::ImmArith,
            kind: ArithKind::Mod,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::MODI_I), Some(Op::MODI_F), None, None, None],
        },
        Op::POWI => OpInfo {
            generic: Op::POWI,
            family: Family::ImmArith,
            kind: ArithKind::Pow,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, Some(Op::POWI_F), Some(Op::POWI_IF), None, None],
        },
        Op::DIVI => OpInfo {
            generic: Op::DIVI,
            family: Family::ImmArith,
            kind: ArithKind::Div,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, Some(Op::DIVI_F), Some(Op::DIVI_IF), None, None],
        },
        Op::IDIVI => OpInfo {
            generic: Op::IDIVI,
            family: Family::ImmArith,
            kind: ArithKind::IDiv,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::IDIVI_I), Some(Op::IDIVI_F), None, None, None],
        },
        Op::RSUBI => OpInfo {
            generic: Op::RSUBI,
            family: Family::ImmArith,
            kind: ArithKind::Sub,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::RSUBI_I), Some(Op::RSUBI_F), None, None, None],
        },
        Op::RMODI => OpInfo {
            generic: Op::RMODI,
            family: Family::ImmArith,
            kind: ArithKind::Mod,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, None, None, None, None],
        },
        Op::RPOWI => OpInfo {
            generic: Op::RPOWI,
            family: Family::ImmArith,
            kind: ArithKind::Pow,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, None, Some(Op::RPOWI_IF), None, None],
        },
        Op::RDIVI => OpInfo {
            generic: Op::RDIVI,
            family: Family::ImmArith,
            kind: ArithKind::Div,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, Some(Op::RDIVI_F), Some(Op::RDIVI_IF), None, None],
        },
        Op::RIDIVI => OpInfo {
            generic: Op::RIDIVI,
            family: Family::ImmArith,
            kind: ArithKind::IDiv,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, None, None, None, None],
        },
        Op::BANDI => OpInfo {
            generic: Op::BANDI,
            family: Family::ImmBit,
            kind: ArithKind::BAnd,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::BANDI_I), None, None, None, None],
        },
        Op::BORI => OpInfo {
            generic: Op::BORI,
            family: Family::ImmBit,
            kind: ArithKind::BOr,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::BORI_I), None, None, None, None],
        },
        Op::BXORI => OpInfo {
            generic: Op::BXORI,
            family: Family::ImmBit,
            kind: ArithKind::BXor,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::BXORI_I), None, None, None, None],
        },
        Op::SHLI => OpInfo {
            generic: Op::SHLI,
            family: Family::ImmBit,
            kind: ArithKind::Shl,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::SHLI_I), None, None, None, None],
        },
        Op::SHRI => OpInfo {
            generic: Op::SHRI,
            family: Family::ImmBit,
            kind: ArithKind::Shr,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [Some(Op::SHRI_I), None, None, None, None],
        },
        Op::RSHLI => OpInfo {
            generic: Op::RSHLI,
            family: Family::ImmBit,
            kind: ArithKind::Shl,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, None, None, None, None],
        },
        Op::RSHRI => OpInfo {
            generic: Op::RSHRI,
            family: Family::ImmBit,
            kind: ArithKind::Shr,
            reversed: true,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None, None, None, None, None],
        },
        Op::ADD_II => OpInfo {
            generic: Op::ADD,
            ..op_info(Op::ADD)
        },
        Op::SUB_II => OpInfo {
            generic: Op::SUB,
            ..op_info(Op::SUB)
        },
        Op::MUL_II => OpInfo {
            generic: Op::MUL,
            ..op_info(Op::MUL)
        },
        Op::MOD_II => OpInfo {
            generic: Op::MOD,
            ..op_info(Op::MOD)
        },
        Op::IDIV_II => OpInfo {
            generic: Op::IDIV,
            ..op_info(Op::IDIV)
        },
        Op::DIV_II => OpInfo {
            generic: Op::DIV,
            ..op_info(Op::DIV)
        },
        Op::ADD_FF => OpInfo {
            generic: Op::ADD,
            ..op_info(Op::ADD)
        },
        Op::SUB_FF => OpInfo {
            generic: Op::SUB,
            ..op_info(Op::SUB)
        },
        Op::MUL_FF => OpInfo {
            generic: Op::MUL,
            ..op_info(Op::MUL)
        },
        Op::MOD_FF => OpInfo {
            generic: Op::MOD,
            ..op_info(Op::MOD)
        },
        Op::POW_FF => OpInfo {
            generic: Op::POW,
            ..op_info(Op::POW)
        },
        Op::DIV_FF => OpInfo {
            generic: Op::DIV,
            ..op_info(Op::DIV)
        },
        Op::IDIV_FF => OpInfo {
            generic: Op::IDIV,
            ..op_info(Op::IDIV)
        },
        Op::ADD_IF => OpInfo {
            generic: Op::ADD,
            ..op_info(Op::ADD)
        },
        Op::SUB_IF => OpInfo {
            generic: Op::SUB,
            ..op_info(Op::SUB)
        },
        Op::MUL_IF => OpInfo {
            generic: Op::MUL,
            ..op_info(Op::MUL)
        },
        Op::DIV_IF => OpInfo {
            generic: Op::DIV,
            ..op_info(Op::DIV)
        },
        Op::ADD_FI => OpInfo {
            generic: Op::ADD,
            ..op_info(Op::ADD)
        },
        Op::SUB_FI => OpInfo {
            generic: Op::SUB,
            ..op_info(Op::SUB)
        },
        Op::MUL_FI => OpInfo {
            generic: Op::MUL,
            ..op_info(Op::MUL)
        },
        Op::DIV_FI => OpInfo {
            generic: Op::DIV,
            ..op_info(Op::DIV)
        },
        Op::BAND_II => OpInfo {
            generic: Op::BAND,
            ..op_info(Op::BAND)
        },
        Op::BOR_II => OpInfo {
            generic: Op::BOR,
            ..op_info(Op::BOR)
        },
        Op::BXOR_II => OpInfo {
            generic: Op::BXOR,
            ..op_info(Op::BXOR)
        },
        Op::SHL_II => OpInfo {
            generic: Op::SHL,
            ..op_info(Op::SHL)
        },
        Op::SHR_II => OpInfo {
            generic: Op::SHR,
            ..op_info(Op::SHR)
        },
        Op::ADDI_I => OpInfo {
            generic: Op::ADDI,
            ..op_info(Op::ADDI)
        },
        Op::SUBI_I => OpInfo {
            generic: Op::SUBI,
            ..op_info(Op::SUBI)
        },
        Op::MULI_I => OpInfo {
            generic: Op::MULI,
            ..op_info(Op::MULI)
        },
        Op::MODI_I => OpInfo {
            generic: Op::MODI,
            ..op_info(Op::MODI)
        },
        Op::IDIVI_I => OpInfo {
            generic: Op::IDIVI,
            ..op_info(Op::IDIVI)
        },
        Op::RSUBI_I => OpInfo {
            generic: Op::RSUBI,
            ..op_info(Op::RSUBI)
        },
        Op::ADDI_F => OpInfo {
            generic: Op::ADDI,
            ..op_info(Op::ADDI)
        },
        Op::SUBI_F => OpInfo {
            generic: Op::SUBI,
            ..op_info(Op::SUBI)
        },
        Op::MULI_F => OpInfo {
            generic: Op::MULI,
            ..op_info(Op::MULI)
        },
        Op::MODI_F => OpInfo {
            generic: Op::MODI,
            ..op_info(Op::MODI)
        },
        Op::IDIVI_F => OpInfo {
            generic: Op::IDIVI,
            ..op_info(Op::IDIVI)
        },
        Op::RSUBI_F => OpInfo {
            generic: Op::RSUBI,
            ..op_info(Op::RSUBI)
        },
        Op::POWI_F => OpInfo {
            generic: Op::POWI,
            ..op_info(Op::POWI)
        },
        Op::DIVI_F => OpInfo {
            generic: Op::DIVI,
            ..op_info(Op::DIVI)
        },
        Op::RDIVI_F => OpInfo {
            generic: Op::RDIVI,
            ..op_info(Op::RDIVI)
        },
        Op::ADDI_IF => OpInfo {
            generic: Op::ADDI,
            ..op_info(Op::ADDI)
        },
        Op::SUBI_IF => OpInfo {
            generic: Op::SUBI,
            ..op_info(Op::SUBI)
        },
        Op::MULI_IF => OpInfo {
            generic: Op::MULI,
            ..op_info(Op::MULI)
        },
        Op::DIVI_IF => OpInfo {
            generic: Op::DIVI,
            ..op_info(Op::DIVI)
        },
        Op::RDIVI_IF => OpInfo {
            generic: Op::RDIVI,
            ..op_info(Op::RDIVI)
        },
        Op::BANDI_I => OpInfo {
            generic: Op::BANDI,
            ..op_info(Op::BANDI)
        },
        Op::BORI_I => OpInfo {
            generic: Op::BORI,
            ..op_info(Op::BORI)
        },
        Op::BXORI_I => OpInfo {
            generic: Op::BXORI,
            ..op_info(Op::BXORI)
        },
        Op::SHLI_I => OpInfo {
            generic: Op::SHLI,
            ..op_info(Op::SHLI)
        },
        Op::SHRI_I => OpInfo {
            generic: Op::SHRI,
            ..op_info(Op::SHRI)
        },
        Op::POW_II => OpInfo {
            generic: Op::POW,
            ..op_info(Op::POW)
        },
        Op::POWI_IF => OpInfo {
            generic: Op::POWI,
            ..op_info(Op::POWI)
        },
        Op::RPOWI_IF => OpInfo {
            generic: Op::RPOWI,
            ..op_info(Op::RPOWI)
        },
        Op::ADD_NN => OpInfo {
            generic: Op::ADD,
            ..op_info(Op::ADD)
        },
        Op::SUB_NN => OpInfo {
            generic: Op::SUB,
            ..op_info(Op::SUB)
        },
        Op::MUL_NN => OpInfo {
            generic: Op::MUL,
            ..op_info(Op::MUL)
        },
        Op::MOD_NN => OpInfo {
            generic: Op::MOD,
            ..op_info(Op::MOD)
        },
        Op::POW_NN => OpInfo {
            generic: Op::POW,
            ..op_info(Op::POW)
        },
        Op::DIV_NN => OpInfo {
            generic: Op::DIV,
            ..op_info(Op::DIV)
        },
        Op::IDIV_NN => OpInfo {
            generic: Op::IDIV,
            ..op_info(Op::IDIV)
        },
        Op::JLT => OpInfo {
            generic: Op::JLT,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JLT_II), None, None, None, None],
        },
        Op::JLT_II => OpInfo {
            generic: Op::JLT,
            ..op_info(Op::JLT)
        },
        Op::JNLT => OpInfo {
            generic: Op::JNLT,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JNLT_II), None, None, None, None],
        },
        Op::JNLT_II => OpInfo {
            generic: Op::JNLT,
            ..op_info(Op::JNLT)
        },
        Op::JLE => OpInfo {
            generic: Op::JLE,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JLE_II), None, None, None, None],
        },
        Op::JLE_II => OpInfo {
            generic: Op::JLE,
            ..op_info(Op::JLE)
        },
        Op::JNLE => OpInfo {
            generic: Op::JNLE,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JNLE_II), None, None, None, None],
        },
        Op::JNLE_II => OpInfo {
            generic: Op::JNLE,
            ..op_info(Op::JNLE)
        },
        Op::JEQ => OpInfo {
            generic: Op::JEQ,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JEQ_II), None, None, None, None],
        },
        Op::JEQ_II => OpInfo {
            generic: Op::JEQ,
            ..op_info(Op::JEQ)
        },
        Op::JNEQ => OpInfo {
            generic: Op::JNEQ,
            family: Family::CmpReg,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AB_IMM,
            forms: [Some(Op::JNEQ_II), None, None, None, None],
        },
        Op::JNEQ_II => OpInfo {
            generic: Op::JNEQ,
            ..op_info(Op::JNEQ)
        },
        Op::JLTI => OpInfo {
            generic: Op::JLTI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JLTI_F), None, None, None],
        },
        Op::JLTI_F => OpInfo {
            generic: Op::JLTI,
            ..op_info(Op::JLTI)
        },
        Op::JNLTI => OpInfo {
            generic: Op::JNLTI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JNLTI_F), None, None, None],
        },
        Op::JNLTI_F => OpInfo {
            generic: Op::JNLTI,
            ..op_info(Op::JNLTI)
        },
        Op::JLEI => OpInfo {
            generic: Op::JLEI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JLEI_F), None, None, None],
        },
        Op::JLEI_F => OpInfo {
            generic: Op::JLEI,
            ..op_info(Op::JLEI)
        },
        Op::JNLEI => OpInfo {
            generic: Op::JNLEI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JNLEI_F), None, None, None],
        },
        Op::JNLEI_F => OpInfo {
            generic: Op::JNLEI,
            ..op_info(Op::JNLEI)
        },
        Op::JGTI => OpInfo {
            generic: Op::JGTI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JGTI_F), None, None, None],
        },
        Op::JGTI_F => OpInfo {
            generic: Op::JGTI,
            ..op_info(Op::JGTI)
        },
        Op::JNGTI => OpInfo {
            generic: Op::JNGTI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JNGTI_F), None, None, None],
        },
        Op::JNGTI_F => OpInfo {
            generic: Op::JNGTI,
            ..op_info(Op::JNGTI)
        },
        Op::JGEI => OpInfo {
            generic: Op::JGEI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JGEI_F), None, None, None],
        },
        Op::JGEI_F => OpInfo {
            generic: Op::JGEI,
            ..op_info(Op::JGEI)
        },
        Op::JNGEI => OpInfo {
            generic: Op::JNGEI,
            family: Family::CmpImm,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_AH_IMM,
            forms: [None, Some(Op::JNGEI_F), None, None, None],
        },
        Op::JNGEI_F => OpInfo {
            generic: Op::JNGEI,
            ..op_info(Op::JNGEI)
        },
        Op::FORLOOP_I => OpInfo {
            generic: Op::FORLOOP,
            family: Family::Loop,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::FORLOOP_F => OpInfo {
            generic: Op::FORLOOP,
            family: Family::Loop,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::TFORCALL_NEXT => OpInfo {
            generic: Op::TFORCALL,
            family: Family::Loop,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::TFORCALL_IPAIRS => OpInfo {
            generic: Op::TFORCALL,
            family: Family::Loop,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::ARITH_MM | Op::ARITH_MM_R => OpInfo {
            generic: Op::ARITH_MM,
            family: Family::RegArith,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC,
            forms: [None; 5],
        },
        Op::ARITH_MMI => OpInfo {
            generic: Op::ARITH_MMI,
            family: Family::ImmArith,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: ADAPTIVE_ABC_IMM,
            forms: [None; 5],
        },
        Op::GETFIELD_INL => OpInfo {
            generic: Op::GETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETFIELD_AUX => OpInfo {
            generic: Op::GETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETFIELD_ABSENT => OpInfo {
            generic: Op::GETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETFIELD_PROTO => OpInfo {
            generic: Op::GETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETTABUP_INL => OpInfo {
            generic: Op::GETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETTABUP_AUX => OpInfo {
            generic: Op::GETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETTABUP_ABSENT => OpInfo {
            generic: Op::GETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::GETTABUP_PROTO => OpInfo {
            generic: Op::GETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SELF_INL => OpInfo {
            generic: Op::SELF,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SELF_AUX => OpInfo {
            generic: Op::SELF,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SELF_ABSENT => OpInfo {
            generic: Op::SELF,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SELF_PROTO => OpInfo {
            generic: Op::SELF,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETFIELD_INL => OpInfo {
            generic: Op::SETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETFIELD_AUX => OpInfo {
            generic: Op::SETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETFIELD_TRANS => OpInfo {
            generic: Op::SETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETFIELD_ABSENT => OpInfo {
            generic: Op::SETFIELD,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETTABUP_INL => OpInfo {
            generic: Op::SETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETTABUP_AUX => OpInfo {
            generic: Op::SETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETTABUP_TRANS => OpInfo {
            generic: Op::SETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        Op::SETTABUP_ABSENT => OpInfo {
            generic: Op::SETTABUP,
            family: Family::Field,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
        op => OpInfo {
            generic: op,
            family: Family::None,
            kind: ArithKind::None,
            reversed: false,
            adaptive_shift: 0,
            forms: [None; 5],
        },
    }
}

pub static OP_INFO: [OpInfo; 256] = {
    let none = OpInfo {
        generic: Op::MOVE,
        family: Family::None,
        kind: ArithKind::None,
        reversed: false,
        adaptive_shift: 0,
        forms: [None; 5],
    };
    let mut t = [none; 256];
    let mut i = 0;
    while i < Op::COUNT {
        t[i] = op_info(Op::ALL[i]);
        i += 1;
    }
    t
};

impl Op {
    /// The generic opcode a specialized form stands in for, else itself. A
    /// metamethod form's original opcode is in its word (`Instruction::mm_orig`).
    #[inline]
    pub fn unquickened(self) -> Op {
        OP_INFO[self as usize].generic
    }

    #[inline(always)]
    pub fn info(self) -> &'static OpInfo {
        &OP_INFO[self as usize]
    }

    /// An immediate-form opcode that computes `imm <op> R[src]` (the `R` forms).
    #[inline]
    pub fn is_reversed(self) -> bool {
        OP_INFO[self as usize].reversed
    }
}

/// Each conditional branch paired with the one that jumps exactly when it
/// doesn't.
macro_rules! branch_pairs {
    ($($t:ident $f:ident),* $(,)?) => {
        impl Op {
            /// Whether a conditional branch jumps when its test holds or when
            /// it doesn't; `None` for other opcodes.
            #[inline]
            pub fn branch_sense(self) -> Option<bool> {
                match self {
                    $(Op::$t => Some(true), Op::$f => Some(false),)*
                    _ => None,
                }
            }

            /// The conditional branch that jumps exactly when this one doesn't.
            pub fn negated(self) -> Op {
                match self {
                    $(Op::$t => Op::$f, Op::$f => Op::$t,)*
                    _ => panic!("{self:?} is not a conditional branch"),
                }
            }
        }
    };
}

branch_pairs! {
    JEQ JNEQ, JLT JNLT, JLE JNLE, JEQI JNEQI, JLTI JNLTI, JLEI JNLEI, JGTI JNGTI,
    JGEI JNGEI, JEQS JNEQS, JT JF, JTSET JFSET,
}

impl Instruction {
    /// This instruction with its opcode replaced, operands kept.
    #[inline]
    pub(crate) fn with_op(self, op: Op) -> Self {
        Instruction((self.0 & !0xff) | op as u64)
    }

    /// The adaptive bits of a specializable site: `(misses, locked)`.
    #[inline(always)]
    pub(crate) fn adaptive(self) -> (u8, bool) {
        let bits = (self.0 >> OP_INFO[self.opcode() as usize].adaptive_shift) as u8;
        (bits & 3, bits & 4 != 0)
    }

    /// This instruction with its adaptive bits set.
    #[inline(always)]
    pub(crate) fn with_adaptive(self, misses: u8, locked: bool) -> Self {
        let shift = OP_INFO[self.opcode() as usize].adaptive_shift;
        debug_assert!(shift != 0 && misses < 4);
        let v = (misses as u64) | (locked as u64) << 2;
        Instruction((self.0 & !(7 << shift)) | v << shift)
    }

    /// The metamethod form `form` (`ARITH_MM`, `ARITH_MM_R` or `ARITH_MMI`) of
    /// this binary op, whose generic opcode `orig` the form records.
    #[inline]
    pub(crate) fn with_mm_form(self, form: Op, orig: Op) -> Self {
        if form == Op::ARITH_MMI {
            let mut idx = 0;
            while idx < IMM_ARITH_COUNT && IMM_ARITH_OPS[idx] as u8 != orig as u8 {
                idx += 1;
            }
            debug_assert!(idx < IMM_ARITH_COUNT, "no ARITH_MMI form for {orig:?}");
            let mut i = self.with_op(form);
            i.set_c((self.c() & 0x0f) | (idx as u8) << 4);
            i
        } else {
            let mut i = self.with_op(form);
            i.set_e(orig as u8 as u16);
            i
        }
    }

    /// The opcode an `ARITH_MMI` stands in for.
    #[inline(always)]
    pub(crate) fn mm_orig_imm(self) -> Op {
        IMM_ARITH_OPS[(self.c() >> 4) as usize & 15]
    }

    /// The opcode an `ARITH_MM` or `ARITH_MM_R` stands in for.
    #[inline(always)]
    pub(crate) fn mm_orig_reg(self) -> Op {
        Instruction((self.e() & 0xff) as u64).op()
    }

    /// The opcode a metamethod form stands in for.
    #[inline(always)]
    pub(crate) fn mm_orig(self) -> Op {
        if self.op() == Op::ARITH_MMI {
            self.mm_orig_imm()
        } else {
            self.mm_orig_reg()
        }
    }

    /// The generic opcode of this site, whatever form it is in.
    #[inline(always)]
    pub(crate) fn generic_op(self) -> Op {
        match self.op() {
            Op::ARITH_MM | Op::ARITH_MM_R | Op::ARITH_MMI => self.mm_orig(),
            op => op.unquickened(),
        }
    }

    /// This instruction as the compiler emitted it: the generic opcode, no
    /// adaptive bits, no recorded original.
    #[inline(always)]
    pub fn unquickened(self) -> Self {
        let generic = self.generic_op();
        let mut i = match self.op() {
            Op::ARITH_MM | Op::ARITH_MM_R => {
                let mut i = self.with_op(generic);
                i.set_e(0);
                i
            }
            Op::ARITH_MMI => {
                let mut i = self.with_op(generic);
                i.set_c(self.c() & 0x0f);
                i
            }
            _ => self.with_op(generic),
        };
        if OP_INFO[generic as usize].adaptive_shift != 0 {
            i = i.with_adaptive(0, false);
        }
        i
    }
}

/// Offset of a generic `for`'s first variable from its base. Past Lua 5.5's
/// three hidden slots is a fourth, where `TFORCALL` keeps its place in a
/// table it walks without calling `next`, as LuaJIT keeps it in its control
/// slot.
pub const TFOR_VARS: u8 = 4;

/// Where CLOSURE takes an upvalue from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpvalSource {
    /// The enclosing function's local in this register.
    ParentLocal(u8),
    /// The enclosing function's upvalue at this index.
    ParentUpvalue(u8),
}

/// How CLOSURE fills one upvalue.
#[derive(Debug, Clone, Copy)]
pub struct UpValueDescriptor {
    pub source: UpvalSource,
    /// Its variable is never assigned after its initialization, so closures
    /// hold its value rather than share a cell (LuaJIT Remake's immutable
    /// upvalues). `debug.setupvalue` on one would change only that closure's
    /// copy.
    pub by_value: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_is_a_padding_free_8_bytes() {
        assert_eq!(size_of::<Instruction>(), 8);
        assert_eq!(align_of::<Instruction>(), 8);
    }

    #[test]
    fn round_trips_every_shape() {
        let i = Instruction::add(Reg(1), Reg(2), Reg(3));
        assert_eq!(i.op(), Op::ADD);
        assert_eq!(i.abc(), (1, 2, 3));

        let i = Instruction::getfield(Reg(4), Reg(5), IcIdx(600), KIdx(700));
        assert_eq!(i.op(), Op::GETFIELD);
        assert_eq!(i.abde(), (4, 5, 600, 700));

        let i = Instruction::jmp(Offset(-9));
        assert_eq!(i.op(), Op::JMP);
        assert_eq!(i.branch_offset(), -9);

        let i = Instruction::forloop(Reg(2), Offset(-(1 << 23)));
        assert_eq!(i.a_offset(), (2, -(1 << 23)));

        let i = Instruction::jeq(Reg(7), Reg(8), Offset(-3));
        assert_eq!(i.ab_offset(), (7, 8, -3));
        let mut i = Instruction::jeq(Reg(7), Reg(8), Offset(0));
        assert!(i.set_branch_offset((1 << 23) - 1));
        assert_eq!(i.ab_offset(), (7, 8, (1 << 23) - 1));
        assert!(!i.set_branch_offset(-(1 << 23) - 1));
        assert_eq!(i.with_adaptive(1, false).ab_offset(), (7, 8, (1 << 23) - 1));

        let i = Instruction::self_(Reg(1), Reg(2), IcIdx(4), KIdx(3));
        assert_eq!(i.abde(), (1, 2, 4, 3));

        let i = Instruction::setlist(Reg(1), 2, 3);
        assert_eq!(i.abd(), (1, 2, 3));

        assert_eq!(Instruction::nop().op(), Op::NOP);

        let k = Imm::from_int(-7).unwrap();
        let i = Instruction::addi(Reg(1), Reg(2), true, k);
        assert_eq!(i.abc_imm(), (1, 2, true));
        assert!(i.imm_is_int());
        assert_eq!(i.imm_int(), -7);

        let k = CmpImm::from_int(-7).unwrap();
        // An `AhImm` offset is 24 bits.
        let i = Instruction::jlti(Reg(3), k, Offset(-(1 << 23)));
        assert_eq!(i.ah_imm(), (3, k.0, -(1 << 23)));
        let mut i = Instruction::jlti(Reg(3), k, Offset(5));
        assert!(i.set_branch_offset((1 << 23) - 1));
        assert_eq!(i.ah_imm(), (3, k.0, (1 << 23) - 1));
        assert!(!i.set_branch_offset(1 << 23));
        assert_eq!(i.with_adaptive(2, true).adaptive(), (2, true));
        assert_eq!(i.with_adaptive(2, true).ah_imm(), (3, k.0, (1 << 23) - 1));
        assert_eq!(i.cmp_imm(), k);
        assert_eq!(i.cmp_imm_int(), -7);

        let i = Instruction::jeqs(Reg(3), KIdx(u16::MAX), Offset(5));
        assert_eq!(i.ah_imm(), (3, u16::MAX, 5));
    }

    #[test]
    fn cmp_imm_packing_limits() {
        for n in [CmpImm::MIN, -1, 0, 1, CmpImm::MAX] {
            let k = CmpImm::from_int(n).unwrap();
            assert_eq!((k.int(), k.is_float()), (n, false));
            assert_eq!(Instruction::jeqi(Reg(0), k, Offset(-1)).cmp_imm_int(), n);
            let k = CmpImm::from_float(n as f64).unwrap();
            assert_eq!((k.int(), k.is_float()), (n, true));
            assert_eq!(Instruction::jeqi(Reg(0), k, Offset(-1)).cmp_imm_int(), n);
        }
        assert!(CmpImm::from_int(CmpImm::MAX + 1).is_none());
        assert!(CmpImm::from_int(CmpImm::MIN - 1).is_none());
        assert!(CmpImm::from_float(0.5).is_none());
        assert!(CmpImm::from_float(-0.0).is_none());
        assert!(CmpImm::from_float(f64::NAN).is_none());
        assert!(CmpImm::from_float(f64::INFINITY).is_none());
        assert!(CmpImm::from_float((CmpImm::MAX + 1) as f64).is_none());
    }

    #[test]
    fn imm_packing_limits() {
        assert_eq!(Imm::from_int(Imm::INT_MAX).unwrap().int(), Imm::INT_MAX);
        assert_eq!(Imm::from_int(Imm::INT_MIN).unwrap().int(), Imm::INT_MIN);
        assert!(Imm::from_int(Imm::INT_MAX + 1).is_none());
        assert!(Imm::from_int(Imm::INT_MIN - 1).is_none());
        assert_eq!(Imm::from_int(0).unwrap().int(), 0);

        assert_eq!(Imm::from_float(0.5).unwrap().float(), 0.5);
        assert_eq!(
            Imm::from_float(-0.0).unwrap().float().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(Imm::from_float(1e6).unwrap().float(), 1e6);
        assert!(Imm::from_float(0.1).is_none());
        assert!(Imm::from_float(f64::NAN).is_none());
        assert!(Imm::from_float(f64::INFINITY).is_none());
        // 24 significant bits: exact as f32, but the tag bit is taken.
        assert!(Imm::from_float(16777215.0).is_none());
        assert!(Imm::from_float(16777214.0).is_some());
    }

    #[test]
    fn setters_touch_only_their_slot() {
        let mut i = Instruction::add(Reg(1), Reg(2), Reg(3));
        i.set_a(9);
        assert_eq!(i.abc(), (9, 2, 3));
        assert_eq!(i.op(), Op::ADD);

        let mut i = Instruction::jmp(Offset(5));
        assert!(i.set_branch_offset(-1));
        assert_eq!(i.branch_offset(), -1);
        assert_eq!(i.op(), Op::JMP);

        let mut i = Instruction::getfield(Reg(1), Reg(2), IcIdx(3), KIdx(4));
        i.set_d(u16::MAX);
        assert_eq!(i.abde(), (1, 2, u16::MAX, 4));
    }
}
