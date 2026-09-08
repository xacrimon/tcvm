//! The bytecode instruction word.
//!
//! An instruction is a 64-bit word with a fixed field layout, *not* a Rust
//! enum. The layout is the same for every opcode:
//!
//! ```text
//! byte:  0    1    2    3    4    5    6    7
//!       op    a    b    c   [------ ext ------]
//!                           ext is either d:u16 @4 + e:u16 @6, or imm:i32 @4
//! ```
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
/// GETTABUP/SETTABUP/GETFIELD/SETFIELD; sites are not deduped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IcIdx(pub u16);

/// An index into the prototype's child-prototype list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProtoIdx(pub u16);

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
    u8       => |self| self as u64,
    u16      => |self| self as u64,
    bool     => |self| self as u64,
    i32      => |self| self as u32 as u64,
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
    Imm,
}

const A_SHIFT: u32 = 8;
const B_SHIFT: u32 = 16;
const C_SHIFT: u32 = 24;
const D_SHIFT: u32 = 32;
const E_SHIFT: u32 = 48;
const IMM_SHIFT: u32 = 32;

/// Packers, one per [`Shape`]. Named to match the shape so the table can
/// select one by pasting the shape token.
pub mod shape {
    use super::{A_SHIFT, B_SHIFT, C_SHIFT, D_SHIFT, E_SHIFT, IMM_SHIFT, Instruction, Op, Operand};

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

    /// `Ab` whose `b` slot is a flag (`TEST`).
    #[inline(always)]
    pub fn ab_flag(self) -> (u8, bool) {
        self.expect(Shape::Ab);
        (self.a(), self.b() != 0)
    }

    #[inline(always)]
    pub fn abc(self) -> (u8, u8, u8) {
        self.expect(Shape::Abc);
        (self.a(), self.b(), self.c())
    }

    /// `Abc` whose `c` slot is a flag (`EQ`/`LT`/`LE`/`TESTSET`).
    #[inline(always)]
    pub fn abc_flag(self) -> (u8, u8, bool) {
        self.expect(Shape::Abc);
        (self.a(), self.b(), self.c() != 0)
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

    // --- control-flow helpers ---------------------------------------------

    /// True for the conditional opcodes a `JMP` can follow as its predecessor.
    #[inline]
    pub fn is_control(self) -> bool {
        matches!(self.op(), Op::EQ | Op::LT | Op::LE | Op::TEST | Op::TESTSET)
    }

    /// The polarity flag of a control opcode. `TEST` has no third register
    /// operand, so its flag sits in `b`; every other control opcode carries it
    /// in `c`.
    #[inline]
    pub fn inverted(self) -> bool {
        debug_assert!(self.is_control(), "{self:?} carries no polarity flag");
        if self.op() == Op::TEST {
            self.b() != 0
        } else {
            self.c() != 0
        }
    }

    #[inline]
    pub fn set_inverted(&mut self, v: bool) {
        debug_assert!(self.is_control(), "{self:?} carries no polarity flag");
        if self.op() == Op::TEST {
            self.set_b(v as u8);
        } else {
            self.set_c(v as u8);
        }
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
            Shape::Imm => write!(f, "(imm={})", self.imm()),
        }
    }
}

/// The ISA table: one row per opcode, giving its dispatch number, name,
/// constructor, operand shape, and the name and type of each operand.
///
/// Adding an opcode here gives you the `Op` variant, the typed constructor and
/// the shape metadata; the interpreter's handler array is keyed by `Op`, so
/// the compiler will point at the missing handler.
macro_rules! instructions {
    ($(
        $(#[$meta:meta])*
        $num:literal $op:ident $ctor:ident $shape:ident { $($field:ident : $fty:ty),* $(,)? }
    )*) => {
        /// Opcode numbers. Contiguous from zero — the interpreter indexes its
        /// handler array with `op as usize`.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        #[allow(clippy::upper_case_acronyms)]
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
    0x03 GETUPVAL   getupval    Ab    { dst: Reg, idx: UpIdx }
    0x04 SETUPVAL   setupval    Ab    { src: Reg, idx: UpIdx }
    0x05 GETTABUP   gettabup    Abde  { dst: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x06 SETTABUP   settabup    Abde  { src: Reg, idx: UpIdx, ic_idx: IcIdx, key: KIdx }
    0x07 GETTABLE   gettable    Abc   { dst: Reg, table: Reg, key: Reg }
    0x08 SETTABLE   settable    Abc   { src: Reg, table: Reg, key: Reg }
    0x09 GETFIELD   getfield    Abde  { dst: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }
    0x0a SETFIELD   setfield    Abde  { src: Reg, table: Reg, ic_idx: IcIdx, key_idx: KIdx }

    /// Method-call setup: `R[dst] = R[object][K[key_idx]]; R[dst+1] = R[object]`.
    /// Backs `obj:m(...)` codegen. Falls back to the slow path on `__index`
    /// when the direct lookup is nil. No inline cache yet — the `e` slot is
    /// free for one.
    0x0b SELF       self_       Abd   { dst: Reg, object: Reg, key_idx: KIdx }

    0x0c NEWTABLE   newtable    A     { dst: Reg }
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
    0x21 EQ         eq          Abc   { lhs: Reg, rhs: Reg, inverted: bool }
    0x22 LT         lt          Abc   { lhs: Reg, rhs: Reg, inverted: bool }
    0x23 LE         le          Abc   { lhs: Reg, rhs: Reg, inverted: bool }
    0x24 TEST       test        Ab    { src: Reg, inverted: bool }
    0x25 TESTSET    testset     Abc   { dst: Reg, src: Reg, inverted: bool }
    0x26 CALL       call        Abc   { func: Reg, args: u8, returns: u8 }
    0x27 TAILCALL   tailcall    Ab    { func: Reg, args: u8 }
    0x28 RETURN     ret         Ab    { values: Reg, count: u8 }
    0x29 FORLOOP    forloop     AImm  { base: Reg, offset: i32 }
    0x2a FORPREP    forprep     AImm  { base: Reg, offset: i32 }
    0x2b TFORPREP   tforprep    AImm  { base: Reg, offset: i32 }
    0x2c TFORCALL   tforcall    Ab    { base: Reg, count: u8 }
    0x2d TFORLOOP   tforloop    AImm  { base: Reg, offset: i32 }
    0x2e SETLIST    setlist     Abd   { table: Reg, count: u8, offset: u16 }
    0x2f CLOSURE    closure     Ad    { dst: Reg, proto: ProtoIdx }
    0x30 VARARG     vararg      Ab    { dst: Reg, count: u8 }

    /// Optimized below-base read of an un-escaped named vararg: integer key
    /// `1..=num_extras`, `"n"` for the count, else nil. `base` is unused at
    /// run time but is the table operand when the epilogue rewrites this to
    /// `GETTABLE` for an escaped vararg. Lua 5.5 `OP_GETVARG`.
    0x31 VARARGGET  varargget   Abc   { dst: Reg, base: Reg, key: Reg }

    0x32 VARARGPREP varargprep  A     { num_fixed: u8 }
    0x33 ERRNNIL    errnnil     Ad    { src: Reg, name_key: KIdx }
    0x34 NOP        nop         Nil   { }
    0x35 STOP       stop        Nil   { }
}

/// Describes how to capture an upvalue when creating a closure.
#[derive(Debug, Clone, Copy)]
pub enum UpValueDescriptor {
    /// Capture from the enclosing function's local register at the given index.
    ParentLocal(u8),
    /// Copy from the enclosing function's upvalue at the given index.
    ParentUpvalue(u8),
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

        let i = Instruction::eq(Reg(7), Reg(8), true);
        assert_eq!(i.abc_flag(), (7, 8, true));

        let i = Instruction::self_(Reg(1), Reg(2), KIdx(3));
        assert_eq!(i.abd(), (1, 2, 3));

        assert_eq!(Instruction::nop().op(), Op::NOP);
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
