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

    #[inline(always)]
    pub fn abc(self) -> (u8, u8, u8) {
        self.expect(Shape::Abc);
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
        (self.a(), self.b(), self.c() != 0)
    }

    #[inline(always)]
    pub fn ab_imm(self) -> (u8, u8, i32) {
        self.expect(Shape::AbImm);
        (self.a(), self.b(), self.imm())
    }

    #[inline(always)]
    pub fn ah_imm(self) -> (u8, u16, i32) {
        self.expect(Shape::AhImm);
        (self.a(), self.h(), self.imm())
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
            Shape::AhImm => write!(f, "(a={}, h={}, imm={})", self.a(), self.h(), self.imm()),
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
    0x20 JMP        jmp         Imm   { offset: i32 }

    // --- conditional branches -------------------------------------------------
    //
    // Each jumps `offset` past itself when its test holds (`J..`, `JT`,
    // `JTSET`) or when it doesn't (`JN..`, `JF`, `JFSET`); see
    // `Op::branch_sense`. `JNLT a b` is not `JLE b a`: they differ on NaN and
    // in the metamethod they call.

    0x21 JEQ        jeq         AbImm { lhs: Reg, rhs: Reg, offset: i32 }
    0x22 JNEQ       jneq        AbImm { lhs: Reg, rhs: Reg, offset: i32 }
    0x23 JLT        jlt         AbImm { lhs: Reg, rhs: Reg, offset: i32 }
    0x24 JNLT       jnlt        AbImm { lhs: Reg, rhs: Reg, offset: i32 }
    0x25 JLE        jle         AbImm { lhs: Reg, rhs: Reg, offset: i32 }
    0x26 JNLE       jnle        AbImm { lhs: Reg, rhs: Reg, offset: i32 }

    // `R[src] <cmp> imm`. `GT`/`GE` are the swapped `LT`/`LE`, so a literal on
    // either side compiles to one of these. Equality never consults `__eq`.

    0x27 JEQI       jeqi        AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x28 JNEQI      jneqi       AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x29 JLTI       jlti        AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2a JNLTI      jnlti       AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2b JLEI       jlei        AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2c JNLEI      jnlei       AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2d JGTI       jgti        AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2e JNGTI      jngti       AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x2f JGEI       jgei        AhImm { src: Reg, imm: CmpImm, offset: i32 }
    0x30 JNGEI      jngei       AhImm { src: Reg, imm: CmpImm, offset: i32 }

    // `R[src] == K[key]`, a string: identity, as strings are interned.

    0x31 JEQS       jeqs        AhImm { src: Reg, key: KIdx, offset: i32 }
    0x32 JNEQS      jneqs       AhImm { src: Reg, key: KIdx, offset: i32 }

    // Truthiness of `R[src]`; the `SET` forms copy it to `R[dst]` when they
    // jump (`a or b`).

    0x33 JT         jt          AImm  { src: Reg, offset: i32 }
    0x34 JF         jf          AImm  { src: Reg, offset: i32 }
    0x35 JTSET      jtset       AbImm { dst: Reg, src: Reg, offset: i32 }
    0x36 JFSET      jfset       AbImm { dst: Reg, src: Reg, offset: i32 }

    0x37 CALL       call        Abc   { func: Reg, args: u8, returns: u8 }
    0x38 TAILCALL   tailcall    Ab    { func: Reg, args: u8 }
    0x39 RETURN     ret         Ab    { values: Reg, count: u8 }
    0x3a FORLOOP    forloop     AImm  { base: Reg, offset: i32 }
    0x3b FORPREP    forprep     AImm  { base: Reg, offset: i32 }

    /// Generic `for`, over [`TFOR_VARS`] hidden slots at `base` (iterator,
    /// state, closing value, traversal position) and then its variables.
    0x3c TFORPREP   tforprep    AImm  { base: Reg, offset: i32 }
    0x3d TFORCALL   tforcall    Ab    { base: Reg, count: u8 }
    0x3e TFORLOOP   tforloop    AImm  { base: Reg, offset: i32 }

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

    // --- quickened forms ------------------------------------------------
    //
    // Never emitted: the interpreter rewrites a table access to the form for
    // the kind of entry its inline cache holds once it fills it (`_OWN` an own
    // slot, `_ABSENT` a key the shape lacks, `_PROTO` a slot in the
    // `__index` table, `_TRANS` an added key), and back on a miss. Same
    // operands as the generic form.

    0x5c GETFIELD_OWN    getfield_own    Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5d GETFIELD_ABSENT getfield_absent Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5e GETFIELD_PROTO  getfield_proto  Abde { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x5f GETTABUP_OWN    gettabup_own    Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x60 GETTABUP_ABSENT gettabup_absent Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x61 GETTABUP_PROTO  gettabup_proto  Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x62 SELF_OWN        self_own        Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x63 SELF_ABSENT     self_absent     Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x64 SELF_PROTO      self_proto      Abde { dst: Reg, object: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x65 SETFIELD_OWN    setfield_own    Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x66 SETFIELD_TRANS  setfield_trans  Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x67 SETTABUP_OWN    settabup_own    Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x68 SETTABUP_TRANS  settabup_trans  Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x69 SETFIELD_ABSENT setfield_absent Abde { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x6a SETTABUP_ABSENT settabup_absent Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }

    // --- shared-cell upvalue forms ------------------------------------------
    //
    // Never emitted: the assembler rewrites GETUPVAL, GETTABUP and SETTABUP
    // to these for an upvalue that is not by value. Not quickened.

    0x6b GETUPVAL_REF getupval_ref Ab   { dst: Reg, idx: UpIdx }
    0x6c GETTABUP_REF gettabup_ref Abde { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x6d SETTABUP_REF settabup_ref Abde { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }

    // --- CALL by result count -----------------------------------------------
    //
    // A CALL wanting no result (`returns` 1) or one (`returns` 2), whose
    // continuation is a constant. `returns` stays for the generic paths.

    0x6e CALL_R0    call_r0     Abc   { func: Reg, args: u8, returns: u8 }
    0x6f CALL_R1    call_r1     Abc   { func: Reg, args: u8, returns: u8 }
}

impl Op {
    /// The generic opcode a quickened one stands in for, else itself.
    #[inline]
    pub fn unquickened(self) -> Op {
        match self {
            Op::GETFIELD_OWN | Op::GETFIELD_ABSENT | Op::GETFIELD_PROTO => Op::GETFIELD,
            Op::GETTABUP_OWN | Op::GETTABUP_ABSENT | Op::GETTABUP_PROTO => Op::GETTABUP,
            Op::SELF_OWN | Op::SELF_ABSENT | Op::SELF_PROTO => Op::SELF,
            Op::SETFIELD_OWN | Op::SETFIELD_TRANS | Op::SETFIELD_ABSENT => Op::SETFIELD,
            Op::SETTABUP_OWN | Op::SETTABUP_TRANS | Op::SETTABUP_ABSENT => Op::SETTABUP,
            op => op,
        }
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

        let i = Instruction::jmp(-9);
        assert_eq!(i.op(), Op::JMP);
        assert_eq!(i.imm(), -9);

        let i = Instruction::forloop(Reg(2), i32::MIN);
        assert_eq!(i.a_imm(), (2, i32::MIN));

        let i = Instruction::jeq(Reg(7), Reg(8), -3);
        assert_eq!(i.ab_imm(), (7, 8, -3));

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
        let i = Instruction::jlti(Reg(3), k, i32::MIN);
        assert_eq!(i.ah_imm(), (3, k.0, i32::MIN));
        assert_eq!(i.cmp_imm(), k);
        assert_eq!(i.cmp_imm_int(), -7);

        let i = Instruction::jeqs(Reg(3), KIdx(u16::MAX), 5);
        assert_eq!(i.ah_imm(), (3, u16::MAX, 5));
    }

    #[test]
    fn cmp_imm_packing_limits() {
        for n in [CmpImm::MIN, -1, 0, 1, CmpImm::MAX] {
            let k = CmpImm::from_int(n).unwrap();
            assert_eq!((k.int(), k.is_float()), (n, false));
            assert_eq!(Instruction::jeqi(Reg(0), k, -1).cmp_imm_int(), n);
            let k = CmpImm::from_float(n as f64).unwrap();
            assert_eq!((k.int(), k.is_float()), (n, true));
            assert_eq!(Instruction::jeqi(Reg(0), k, -1).cmp_imm_int(), n);
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

        let mut i = Instruction::jmp(5);
        i.set_imm(-1);
        assert_eq!(i.imm(), -1);
        assert_eq!(i.op(), Op::JMP);

        let mut i = Instruction::getfield(Reg(1), Reg(2), IcIdx(3), KIdx(4));
        i.set_d(u16::MAX);
        assert_eq!(i.abde(), (1, 2, u16::MAX, 4));
    }
}
