//! An x86-64 assembler: legacy and VEX encodings, labels, a pool of 64-bit
//! literals addressed from `rip`, and branches to fixed addresses outside
//! the buffer, resolved once the code's address is known. The baseline is
//! x86-64-v3: AVX, BMI2 and FMA instructions are used freely.

/// A general-purpose register by its encoding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) struct Gpr(pub(crate) u8);

/// An `xmm` register.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) struct Xmm(pub(crate) u8);

pub(crate) const RAX: Gpr = Gpr(0);
pub(crate) const RCX: Gpr = Gpr(1);
pub(crate) const RDX: Gpr = Gpr(2);
pub(crate) const RBX: Gpr = Gpr(3);
pub(crate) const RSP: Gpr = Gpr(4);
pub(crate) const RBP: Gpr = Gpr(5);
pub(crate) const RSI: Gpr = Gpr(6);
pub(crate) const RDI: Gpr = Gpr(7);
pub(crate) const R8: Gpr = Gpr(8);
pub(crate) const R9: Gpr = Gpr(9);
pub(crate) const R10: Gpr = Gpr(10);
pub(crate) const R11: Gpr = Gpr(11);
pub(crate) const R12: Gpr = Gpr(12);
pub(crate) const R13: Gpr = Gpr(13);
pub(crate) const R14: Gpr = Gpr(14);
pub(crate) const R15: Gpr = Gpr(15);

/// Operand width of an integer instruction: 32 bits (the upper half of the
/// destination zeroed) or 64.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Sz {
    D,
    Q,
}

impl Sz {
    pub(crate) fn bits(self) -> u32 {
        match self {
            Sz::D => 32,
            Sz::Q => 64,
        }
    }
}

/// A condition, by its encoding; `invert` makes the ones nothing names.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Cond {
    O = 0,
    No = 1,
    B = 2,
    Ae = 3,
    E = 4,
    Ne = 5,
    Be = 6,
    A = 7,
    S = 8,
    Ns = 9,
    P = 10,
    Np = 11,
    L = 12,
    Ge = 13,
    Le = 14,
    G = 15,
}

impl Cond {
    /// The condition that holds exactly when this one does not.
    pub(crate) fn invert(self) -> Cond {
        // SAFETY: every value 0..16 is a condition.
        unsafe { std::mem::transmute(self as u8 ^ 1) }
    }
}

/// The two-operand integer operations sharing one encoding scheme.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Arith {
    Add = 0,
    Or = 1,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Shift {
    Shl,
    Shr,
    Sar,
}

/// The scalar double operations `v<op>sd d, n, m`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum FBin {
    Add = 0x58,
    Mul = 0x59,
    Sub = 0x5c,
    Div = 0x5e,
    Sqrt = 0x51,
}

/// The packed bitwise operations `v<op>pd d, n, m`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum FBit {
    And = 0x54,
    Xor = 0x57,
}

/// A position in the buffer, bound once.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Label(pub(crate) u32);

/// A memory operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mem {
    /// `[base + disp]`.
    B(Gpr, i32),
    /// `[base + index << scale + disp]`.
    X(Gpr, Gpr, u8, i32),
    /// `[rip + label]`.
    L(Label),
    /// The pooled 64-bit literal.
    K(u64),
}

/// The `r/m` operand of an instruction.
#[derive(Clone, Copy)]
pub(crate) enum Rm {
    R(u8),
    M(Mem),
}

impl From<Gpr> for Rm {
    fn from(r: Gpr) -> Rm {
        Rm::R(r.0)
    }
}

impl From<Xmm> for Rm {
    fn from(r: Xmm) -> Rm {
        Rm::R(r.0)
    }
}

impl From<Mem> for Rm {
    fn from(m: Mem) -> Rm {
        Rm::M(m)
    }
}

/// A 32-bit field holding `label - (at + 4 + tail)`: a branch's, or a
/// `rip`-relative operand's with `tail` immediate bytes after it.
#[derive(Clone, Copy, Debug)]
struct Fixup {
    at: u32,
    label: Label,
    tail: u8,
}

/// The encoding failed: a branch to a fixed address out of reach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AsmError {
    ExternRange,
}

#[derive(Default)]
pub(crate) struct Asm {
    code: Vec<u8>,
    labels: Vec<u32>,
    fixups: Vec<Fixup>,
    /// `call`/`jmp rel32` to a fixed address: (the instruction, target).
    externs: Vec<(u32, usize)>,
    /// 64-bit literals by label, emitted after the code.
    literals: Vec<(Label, u64)>,
    lit_index: std::collections::HashMap<u64, Label>,
    /// Where the last unconditional jump ended.
    jump_end: u32,
}

const UNBOUND: u32 = u32::MAX;

/// VEX prefixes' `pp` field.
const P66: u8 = 1;
const PF3: u8 = 2;
const PF2: u8 = 3;
/// VEX opcode maps.
const M0F: u8 = 1;
const M0F38: u8 = 2;
const M0F3A: u8 = 3;

impl Asm {
    pub(crate) fn new() -> Self {
        Asm {
            jump_end: UNBOUND,
            ..Asm::default()
        }
    }

    /// Bytes emitted so far.
    pub(crate) fn offset(&self) -> usize {
        self.code.len()
    }

    pub(crate) fn new_label(&mut self) -> Label {
        let l = Label(self.labels.len() as u32);
        self.labels.push(UNBOUND);
        l
    }

    pub(crate) fn bind(&mut self, l: Label) {
        debug_assert_eq!(self.labels[l.0 as usize], UNBOUND, "label bound twice");
        self.labels[l.0 as usize] = self.code.len() as u32;
    }

    /// Bind `l` where code may be overwritten (`Region::invalidate`): a
    /// `nop` of `code::JUMP` bytes, which nothing branches into.
    pub(crate) fn patch_point(&mut self, l: Label) {
        self.bind(l);
        self.code.extend([0x0f, 0x1f, 0x44, 0x00, 0x00]);
    }

    /// The bound label's byte offset.
    pub(crate) fn label_offset(&self, l: Label) -> usize {
        let at = self.labels[l.0 as usize];
        debug_assert_ne!(at, UNBOUND);
        at as usize
    }

    /// Whether the last instruction is an unconditional jump: what follows
    /// is reached only by a branch.
    pub(crate) fn after_jump(&self) -> bool {
        self.jump_end == self.code.len() as u32
    }

    /// Pad with `int3` to a multiple of `align` bytes.
    pub(crate) fn align(&mut self, align: usize) {
        while !self.code.len().is_multiple_of(align) {
            self.code.push(0xcc);
        }
    }

    /// A 64-bit word in line.
    pub(crate) fn quad(&mut self, v: u64) {
        self.code.extend(v.to_le_bytes());
    }

    /// The label of a pooled 64-bit literal.
    pub(crate) fn literal(&mut self, v: u64) -> Label {
        if let Some(&l) = self.lit_index.get(&v) {
            return l;
        }
        let l = self.new_label();
        self.literals.push((l, v));
        self.lit_index.insert(v, l);
        l
    }

    /// An upper bound of the finished code's length.
    pub(crate) fn max_len(&self) -> usize {
        self.code.len() + 8 + 8 * self.literals.len() + 8
    }

    /// Emit the literal pool, resolve every branch for code placed at
    /// `base`, and hand back the code.
    pub(crate) fn finish(mut self, base: usize) -> Result<Vec<u8>, AsmError> {
        if !self.literals.is_empty() {
            self.align(8);
            for (l, v) in std::mem::take(&mut self.literals) {
                self.bind(l);
                self.quad(v);
            }
            // A packed operation reads 16 bytes of its literal.
            self.quad(0);
        }
        for fx in &self.fixups {
            let target = self.labels[fx.label.0 as usize];
            assert_ne!(target, UNBOUND, "branch to an unbound label");
            let rel = target as i64 - (fx.at as i64 + 4 + fx.tail as i64);
            self.code[fx.at as usize..fx.at as usize + 4]
                .copy_from_slice(&(rel as i32).to_le_bytes());
        }
        for &(at, target) in &self.externs {
            let here = base + at as usize;
            let target = crate::jit::backend::alloc::branch_target(here, target)
                .ok_or(AsmError::ExternRange)?;
            let rel = target as i64 - (here as i64 + 5);
            self.code[at as usize + 1..at as usize + 5]
                .copy_from_slice(&(rel as i32).to_le_bytes());
        }
        Ok(self.code)
    }

    // --- encoding ----------------------------------------------------------

    fn b(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn mem(&mut self, m: Mem) -> Mem {
        match m {
            Mem::K(v) => Mem::L(self.literal(v)),
            m => m,
        }
    }

    /// `REX`, when any of its bits is set or `force` (a byte register past
    /// `bl`).
    fn rex(&mut self, w: bool, reg: u8, rm: Rm, force: bool) {
        let (x, b) = match rm {
            Rm::R(r) => (0, r >> 3),
            Rm::M(Mem::B(base, _)) => (0, base.0 >> 3),
            Rm::M(Mem::X(base, index, _, _)) => (index.0 >> 3, base.0 >> 3),
            Rm::M(_) => (0, 0),
        };
        let v = (w as u8) << 3 | (reg >> 3) << 2 | x << 1 | b;
        if v != 0 || force {
            self.b(0x40 | v);
        }
    }

    /// The ModRM byte with its SIB and displacement; `tail` immediate bytes
    /// follow a `rip`-relative operand.
    fn modrm(&mut self, reg: u8, rm: Rm, tail: u8) {
        let reg = (reg & 7) << 3;
        match rm {
            Rm::R(r) => self.b(0xc0 | reg | r & 7),
            Rm::M(Mem::B(base, disp)) => {
                let md = Self::disp_mode(base, disp);
                self.b(md << 6 | reg | base.0 & 7);
                if base.0 & 7 == 4 {
                    self.b(0x24);
                }
                self.disp(md, disp);
            }
            Rm::M(Mem::X(base, index, scale, disp)) => {
                debug_assert!(index.0 != 4, "rsp as an index");
                let md = Self::disp_mode(base, disp);
                self.b(md << 6 | reg | 4);
                self.b(scale << 6 | (index.0 & 7) << 3 | base.0 & 7);
                self.disp(md, disp);
            }
            Rm::M(Mem::L(label)) => {
                self.b(reg | 5);
                self.fixups.push(Fixup {
                    at: self.code.len() as u32,
                    label,
                    tail,
                });
                self.code.extend([0; 4]);
            }
            Rm::M(Mem::K(_)) => unreachable!("an unpooled literal"),
        }
    }

    fn disp_mode(base: Gpr, disp: i32) -> u8 {
        // `rbp` and `r13` as a base always take a displacement.
        if disp == 0 && base.0 & 7 != 5 {
            0
        } else if i8::try_from(disp).is_ok() {
            1
        } else {
            2
        }
    }

    fn disp(&mut self, md: u8, disp: i32) {
        match md {
            1 => self.b(disp as u8),
            2 => self.code.extend(disp.to_le_bytes()),
            _ => {}
        }
    }

    /// A legacy instruction: `REX`, the opcode bytes, ModRM.
    fn op(&mut self, sz: Sz, opcode: &[u8], reg: u8, rm: impl Into<Rm>, tail: u8) {
        let rm = match rm.into() {
            Rm::M(m) => Rm::M(self.mem(m)),
            rm => rm,
        };
        self.rex(sz == Sz::Q, reg, rm, false);
        self.code.extend(opcode);
        self.modrm(reg, rm, tail);
    }

    /// A VEX instruction with operands `reg`, `vvvv` and `rm`.
    #[allow(clippy::too_many_arguments)]
    fn vex(
        &mut self,
        pp: u8,
        map: u8,
        w: bool,
        opcode: u8,
        reg: u8,
        vvvv: u8,
        rm: impl Into<Rm>,
        tail: u8,
    ) {
        let rm = match rm.into() {
            Rm::M(m) => Rm::M(self.mem(m)),
            rm => rm,
        };
        let (x, b) = match rm {
            Rm::R(r) => (0, r >> 3),
            Rm::M(Mem::B(base, _)) => (0, base.0 >> 3),
            Rm::M(Mem::X(base, index, _, _)) => (index.0 >> 3, base.0 >> 3),
            Rm::M(_) => (0, 0),
        };
        let r = reg >> 3;
        let v = (!vvvv & 15) << 3 | pp;
        if x == 0 && b == 0 && !w && map == M0F {
            self.b(0xc5);
            self.b((r ^ 1) << 7 | v);
        } else {
            self.b(0xc4);
            self.b((r ^ 1) << 7 | (x ^ 1) << 6 | (b ^ 1) << 5 | map);
            self.b((w as u8) << 7 | v);
        }
        self.b(opcode);
        self.modrm(reg, rm, tail);
    }

    /// An immediate as one sign-extended byte or four.
    fn imm8_or_32(&mut self, imm: i32) {
        match i8::try_from(imm) {
            Ok(b) => self.b(b as u8),
            Err(_) => self.code.extend(imm.to_le_bytes()),
        }
    }

    // --- moves -------------------------------------------------------------

    /// `mov d, s`; nothing when equal.
    pub(crate) fn mov(&mut self, d: Gpr, s: Gpr) {
        if d != s {
            self.op(Sz::Q, &[0x8b], d.0, s, 0);
        }
    }

    /// `mov d32, s32`: the low half, zero-extended (emitted when equal too).
    pub(crate) fn mov32(&mut self, d: Gpr, s: Gpr) {
        self.op(Sz::D, &[0x8b], d.0, s, 0);
    }

    /// `d = imm`, in the shortest of the three encodings; the flags stay.
    pub(crate) fn mov_imm(&mut self, d: Gpr, imm: u64) {
        if imm <= u32::MAX as u64 {
            self.rex(false, 0, Rm::R(d.0), false);
            self.b(0xb8 | d.0 & 7);
            self.code.extend((imm as u32).to_le_bytes());
        } else if i32::try_from(imm as i64).is_ok() {
            self.op(Sz::Q, &[0xc7], 0, d, 4);
            self.code.extend((imm as u32).to_le_bytes());
        } else {
            self.rex(true, 0, Rm::R(d.0), false);
            self.b(0xb8 | d.0 & 7);
            self.code.extend(imm.to_le_bytes());
        }
    }

    pub(crate) fn load(&mut self, sz: Sz, d: Gpr, m: Mem) {
        self.op(sz, &[0x8b], d.0, m, 0);
    }

    pub(crate) fn store(&mut self, sz: Sz, m: Mem, s: Gpr) {
        self.op(sz, &[0x89], s.0, m, 0);
    }

    /// `mov [m], imm32`, sign-extended to 64 bits for `Sz::Q`.
    pub(crate) fn store_imm(&mut self, sz: Sz, m: Mem, imm: i32) {
        self.op(sz, &[0xc7], 0, m, 4);
        self.code.extend(imm.to_le_bytes());
    }

    /// `movzx d32, byte [m]`.
    pub(crate) fn load8(&mut self, d: Gpr, m: Mem) {
        self.op(Sz::D, &[0x0f, 0xb6], d.0, m, 0);
    }

    /// `movsxd d, s32`.
    pub(crate) fn movsxd(&mut self, d: Gpr, s: impl Into<Rm>) {
        self.op(Sz::Q, &[0x63], d.0, s, 0);
    }

    pub(crate) fn lea(&mut self, sz: Sz, d: Gpr, m: Mem) {
        self.op(sz, &[0x8d], d.0, m, 0);
    }

    pub(crate) fn push(&mut self, r: Gpr) {
        self.rex(false, 0, Rm::R(r.0), false);
        self.b(0x50 | r.0 & 7);
    }

    pub(crate) fn pop(&mut self, r: Gpr) {
        self.rex(false, 0, Rm::R(r.0), false);
        self.b(0x58 | r.0 & 7);
    }

    // --- integer arithmetic ------------------------------------------------

    /// `op d, s` with `s` a register or memory.
    pub(crate) fn alu(&mut self, op: Arith, sz: Sz, d: Gpr, s: impl Into<Rm>) {
        self.op(sz, &[(op as u8) << 3 | 3], d.0, s, 0);
    }

    /// `op [m], s`.
    pub(crate) fn alu_mr(&mut self, op: Arith, sz: Sz, m: Mem, s: Gpr) {
        self.op(sz, &[(op as u8) << 3 | 1], s.0, m, 0);
    }

    /// `op d, imm32` with `d` a register or memory, the immediate
    /// sign-extended.
    pub(crate) fn alu_imm(&mut self, op: Arith, sz: Sz, d: impl Into<Rm>, imm: i32) {
        let short = i8::try_from(imm).is_ok();
        self.op(
            sz,
            &[if short { 0x83 } else { 0x81 }],
            op as u8,
            d,
            if short { 1 } else { 4 },
        );
        self.imm8_or_32(imm);
    }

    pub(crate) fn cmp(&mut self, sz: Sz, a: Gpr, b: impl Into<Rm>) {
        self.alu(Arith::Cmp, sz, a, b);
    }

    pub(crate) fn cmp_imm(&mut self, sz: Sz, a: impl Into<Rm>, imm: i32) {
        self.alu_imm(Arith::Cmp, sz, a, imm);
    }

    pub(crate) fn test(&mut self, sz: Sz, a: Gpr, b: Gpr) {
        self.op(sz, &[0x85], b.0, a, 0);
    }

    pub(crate) fn test_imm(&mut self, sz: Sz, a: impl Into<Rm>, imm: i32) {
        self.op(sz, &[0xf7], 0, a, 4);
        self.code.extend(imm.to_le_bytes());
    }

    /// `test byte [m], imm8`.
    pub(crate) fn test8_imm(&mut self, m: Mem, imm: u8) {
        self.op(Sz::D, &[0xf6], 0, m, 1);
        self.b(imm);
    }

    /// `cmp byte [m], imm8`.
    pub(crate) fn cmp8_imm(&mut self, m: Mem, imm: u8) {
        self.op(Sz::D, &[0x80], 7, m, 1);
        self.b(imm);
    }

    pub(crate) fn imul(&mut self, sz: Sz, d: Gpr, s: impl Into<Rm>) {
        self.op(sz, &[0x0f, 0xaf], d.0, s, 0);
    }

    pub(crate) fn neg(&mut self, sz: Sz, d: Gpr) {
        self.op(sz, &[0xf7], 3, d, 0);
    }

    pub(crate) fn not(&mut self, sz: Sz, d: Gpr) {
        self.op(sz, &[0xf7], 2, d, 0);
    }

    /// `rdx:rax / s` signed, the quotient to `rax` and the remainder to `rdx`.
    pub(crate) fn idiv(&mut self, sz: Sz, s: Gpr) {
        self.op(sz, &[0xf7], 7, s, 0);
    }

    /// The same unsigned.
    pub(crate) fn div(&mut self, sz: Sz, s: Gpr) {
        self.op(sz, &[0xf7], 6, s, 0);
    }

    /// `cdq` or `cqo`: `rax`'s sign into `rdx`.
    pub(crate) fn sign_rdx(&mut self, sz: Sz) {
        if sz == Sz::Q {
            self.b(0x48);
        }
        self.b(0x99);
    }

    pub(crate) fn shift_imm(&mut self, sh: Shift, sz: Sz, d: Gpr, k: u8) {
        let ext = match sh {
            Shift::Shl => 4,
            Shift::Shr => 5,
            Shift::Sar => 7,
        };
        self.op(sz, &[0xc1], ext, d, 1);
        self.b(k);
    }

    /// `shlx`/`shrx`/`sarx d, s, count`: by `count` mod the width, the flags
    /// kept.
    pub(crate) fn shiftx(&mut self, sh: Shift, sz: Sz, d: Gpr, s: impl Into<Rm>, count: Gpr) {
        let pp = match sh {
            Shift::Shl => P66,
            Shift::Shr => PF2,
            Shift::Sar => PF3,
        };
        self.vex(pp, M0F38, sz == Sz::Q, 0xf7, d.0, count.0, s, 0);
    }

    /// `setcc d8; movzx d32, d8`.
    pub(crate) fn setcc(&mut self, c: Cond, d: Gpr) {
        self.rex(false, 0, Rm::R(d.0), d.0 >= 4);
        self.code.extend([0x0f, 0x90 | c as u8]);
        self.modrm(0, Rm::R(d.0), 0);
        self.rex(false, d.0, Rm::R(d.0), d.0 >= 4);
        self.code.extend([0x0f, 0xb6]);
        self.modrm(d.0, Rm::R(d.0), 0);
    }

    pub(crate) fn cmov(&mut self, c: Cond, sz: Sz, d: Gpr, s: impl Into<Rm>) {
        self.op(sz, &[0x0f, 0x40 | c as u8], d.0, s, 0);
    }

    // --- floats ------------------------------------------------------------

    /// `vmovaps d, s`; nothing when equal.
    pub(crate) fn fmov(&mut self, d: Xmm, s: Xmm) {
        if d != s {
            self.vex(0, M0F, false, 0x28, d.0, 0, s, 0);
        }
    }

    /// `vmovsd d, [m]`, the upper half zeroed.
    pub(crate) fn fload(&mut self, d: Xmm, m: Mem) {
        self.vex(PF2, M0F, false, 0x10, d.0, 0, m, 0);
    }

    pub(crate) fn fstore(&mut self, m: Mem, s: Xmm) {
        self.vex(PF2, M0F, false, 0x11, s.0, 0, m, 0);
    }

    /// `vmovq d, s`: the bits.
    pub(crate) fn fmov_from_gpr(&mut self, d: Xmm, s: Gpr) {
        self.vex(P66, M0F, true, 0x6e, d.0, 0, s, 0);
    }

    pub(crate) fn fmov_to_gpr(&mut self, d: Gpr, s: Xmm) {
        self.vex(P66, M0F, true, 0x7e, s.0, 0, d, 0);
    }

    pub(crate) fn fbin(&mut self, op: FBin, d: Xmm, n: Xmm, m: impl Into<Rm>) {
        self.vex(PF2, M0F, false, op as u8, d.0, n.0, m, 0);
    }

    pub(crate) fn fbit(&mut self, op: FBit, d: Xmm, n: Xmm, m: impl Into<Rm>) {
        self.vex(P66, M0F, false, op as u8, d.0, n.0, m, 0);
    }

    /// `d = 0.0`.
    pub(crate) fn fzero(&mut self, d: Xmm) {
        self.fbit(FBit::Xor, d, d, d);
    }

    /// `vroundsd d, n, n, mode`, inexact not raised: 1 floors, 2 ceils.
    pub(crate) fn fround(&mut self, d: Xmm, n: Xmm, mode: u8) {
        self.vex(P66, M0F3A, false, 0x0b, d.0, n.0, n, 1);
        self.b(8 | mode);
    }

    /// `vucomisd a, b`: unordered sets `ZF`, `PF` and `CF`.
    pub(crate) fn fcmp(&mut self, a: Xmm, b: impl Into<Rm>) {
        self.vex(P66, M0F, false, 0x2e, a.0, 0, b, 0);
    }

    /// `d` all ones where `n == m` (ordered), else zero.
    pub(crate) fn fcmp_eq(&mut self, d: Xmm, n: Xmm, m: Xmm) {
        self.vex(PF2, M0F, false, 0xc2, d.0, n.0, m, 1);
        self.b(0);
    }

    /// `vptest a, a`: `ZF` when `a` is zero.
    pub(crate) fn ftest(&mut self, a: Xmm) {
        self.vex(P66, M0F38, false, 0x17, a.0, 0, a, 0);
    }

    /// `d = (f64) s`, with no dependency on what `d` held.
    pub(crate) fn cvt_from_int(&mut self, sz: Sz, d: Xmm, s: Gpr) {
        self.fzero(d);
        self.vex(PF2, M0F, sz == Sz::Q, 0x2a, d.0, d.0, s, 0);
    }

    /// `d = (int) s`, truncating; out of range and NaN give the minimum.
    pub(crate) fn cvt_to_int(&mut self, sz: Sz, d: Gpr, s: Xmm) {
        self.vex(PF2, M0F, sz == Sz::Q, 0x2c, d.0, 0, s, 0);
    }

    /// `d = d - n * m`, fused.
    pub(crate) fn fnmadd(&mut self, d: Xmm, n: Xmm, m: Xmm) {
        self.vex(P66, M0F38, true, 0xbd, d.0, n.0, m, 0);
    }

    // --- control -----------------------------------------------------------

    fn rel32(&mut self, label: Label) {
        self.fixups.push(Fixup {
            at: self.code.len() as u32,
            label,
            tail: 0,
        });
        self.code.extend([0; 4]);
    }

    /// The distance back to a bound label from the end of a two-byte
    /// branch here, when one byte holds it.
    fn short(&self, l: Label) -> Option<i8> {
        let at = self.labels[l.0 as usize];
        if at == UNBOUND {
            return None;
        }
        i8::try_from(at as i64 - (self.code.len() as i64 + 2)).ok()
    }

    pub(crate) fn jmp(&mut self, l: Label) {
        match self.short(l) {
            Some(d) => self.code.extend([0xeb, d as u8]),
            None => {
                self.b(0xe9);
                self.rel32(l);
            }
        }
        self.jump_end = self.code.len() as u32;
    }

    pub(crate) fn jcc(&mut self, c: Cond, l: Label) {
        match self.short(l) {
            Some(d) => self.code.extend([0x70 | c as u8, d as u8]),
            None => {
                self.code.extend([0x0f, 0x80 | c as u8]);
                self.rel32(l);
            }
        }
    }

    pub(crate) fn call(&mut self, l: Label) {
        self.b(0xe8);
        self.rel32(l);
    }

    /// `jmp` to a fixed address, through an island out of reach.
    pub(crate) fn jmp_far(&mut self, target: usize) {
        self.externs.push((self.code.len() as u32, target));
        self.code.extend([0xe9, 0, 0, 0, 0]);
        self.jump_end = self.code.len() as u32;
    }

    pub(crate) fn call_far(&mut self, target: usize) {
        self.externs.push((self.code.len() as u32, target));
        self.code.extend([0xe8, 0, 0, 0, 0]);
    }

    pub(crate) fn jmp_reg(&mut self, r: impl Into<Rm>) {
        self.op(Sz::D, &[0xff], 4, r, 0);
        self.jump_end = self.code.len() as u32;
    }

    pub(crate) fn call_reg(&mut self, r: impl Into<Rm>) {
        self.op(Sz::D, &[0xff], 2, r, 0);
    }

    pub(crate) fn ret(&mut self) {
        self.b(0xc3);
        self.jump_end = self.code.len() as u32;
    }

    pub(crate) fn nop(&mut self) {
        self.b(0x90);
    }

    pub(crate) fn ud2(&mut self) {
        self.code.extend([0x0f, 0x0b]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(f: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut a = Asm::new();
        f(&mut a);
        a.finish(0).unwrap()
    }

    /// Encodings as binutils' `objdump` reads them back.
    #[test]
    fn encodings() {
        assert_eq!(bytes(|a| a.mov(RAX, R15)), [0x49, 0x8b, 0xc7]);
        assert_eq!(bytes(|a| a.mov_imm(R9, 7)), [0x41, 0xb9, 7, 0, 0, 0]);
        assert_eq!(
            bytes(|a| a.mov_imm(RAX, !0)),
            [0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            bytes(|a| a.mov_imm(R11, 0x1122_3344_5566_7788)),
            [0x49, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        assert_eq!(
            bytes(|a| a.load(Sz::Q, RCX, Mem::B(R14, 8))),
            [0x49, 0x8b, 0x4e, 0x08]
        );
        assert_eq!(
            bytes(|a| a.load(Sz::Q, RCX, Mem::B(RSP, 0))),
            [0x48, 0x8b, 0x0c, 0x24]
        );
        assert_eq!(
            bytes(|a| a.load(Sz::Q, RCX, Mem::B(R13, 0))),
            [0x49, 0x8b, 0x4d, 0x00]
        );
        assert_eq!(
            bytes(|a| a.store(Sz::D, Mem::B(RBP, -256), R8)),
            [0x44, 0x89, 0x85, 0x00, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            bytes(|a| a.load(Sz::Q, RAX, Mem::X(RSI, R11, 3, 16))),
            [0x4a, 0x8b, 0x44, 0xde, 0x10]
        );
        assert_eq!(
            bytes(|a| a.lea(Sz::D, RDX, Mem::X(RDI, RSI, 0, 0))),
            [0x8d, 0x14, 0x37]
        );
        assert_eq!(bytes(|a| a.alu(Arith::Add, Sz::D, RAX, RCX)), [0x03, 0xc1]);
        assert_eq!(
            bytes(|a| a.alu_imm(Arith::Sub, Sz::Q, RSP, 32)),
            [0x48, 0x83, 0xec, 0x20]
        );
        assert_eq!(
            bytes(|a| a.cmp_imm(Sz::D, R12, 1000)),
            [0x41, 0x81, 0xfc, 0xe8, 0x03, 0, 0]
        );
        assert_eq!(
            bytes(|a| a.test_imm(Sz::D, RDX, 4)),
            [0xf7, 0xc2, 4, 0, 0, 0]
        );
        assert_eq!(bytes(|a| a.imul(Sz::Q, R8, RDX)), [0x4c, 0x0f, 0xaf, 0xc2]);
        assert_eq!(
            bytes(|a| a.shift_imm(Shift::Sar, Sz::Q, R11, 48)),
            [0x49, 0xc1, 0xfb, 0x30]
        );
        assert_eq!(
            bytes(|a| a.shiftx(Shift::Shl, Sz::Q, RAX, RCX, RDX)),
            [0xc4, 0xe2, 0xe9, 0xf7, 0xc1]
        );
        assert_eq!(
            bytes(|a| a.shiftx(Shift::Sar, Sz::D, R8, R9, R10)),
            [0xc4, 0x42, 0x2a, 0xf7, 0xc1]
        );
        assert_eq!(
            bytes(|a| a.setcc(Cond::E, RSI)),
            [0x40, 0x0f, 0x94, 0xc6, 0x40, 0x0f, 0xb6, 0xf6]
        );
        assert_eq!(
            bytes(|a| a.cmov(Cond::L, Sz::Q, RAX, R9)),
            [0x49, 0x0f, 0x4c, 0xc1]
        );
        assert_eq!(bytes(|a| a.push(R12)), [0x41, 0x54]);
        assert_eq!(
            bytes(|a| a.fmov(Xmm(1), Xmm(9))),
            [0xc4, 0xc1, 0x78, 0x28, 0xc9]
        );
        assert_eq!(
            bytes(|a| a.fload(Xmm(0), Mem::B(R14, 16))),
            [0xc4, 0xc1, 0x7b, 0x10, 0x46, 0x10]
        );
        assert_eq!(
            bytes(|a| a.fstore(Mem::B(RBP, -8), Xmm(3))),
            [0xc5, 0xfb, 0x11, 0x5d, 0xf8]
        );
        assert_eq!(
            bytes(|a| a.fmov_from_gpr(Xmm(2), RAX)),
            [0xc4, 0xe1, 0xf9, 0x6e, 0xd0]
        );
        assert_eq!(
            bytes(|a| a.fmov_to_gpr(R10, Xmm(15))),
            [0xc4, 0x41, 0xf9, 0x7e, 0xfa]
        );
        assert_eq!(
            bytes(|a| a.fbin(FBin::Add, Xmm(0), Xmm(1), Xmm(2))),
            [0xc5, 0xf3, 0x58, 0xc2]
        );
        assert_eq!(
            bytes(|a| a.fbin(FBin::Div, Xmm(8), Xmm(9), Xmm(10))),
            [0xc4, 0x41, 0x33, 0x5e, 0xc2]
        );
        assert_eq!(
            bytes(|a| a.fround(Xmm(0), Xmm(1), 1)),
            [0xc4, 0xe3, 0x71, 0x0b, 0xc1, 0x09]
        );
        assert_eq!(bytes(|a| a.fcmp(Xmm(0), Xmm(1))), [0xc5, 0xf9, 0x2e, 0xc1]);
        assert_eq!(
            bytes(|a| a.fcmp_eq(Xmm(15), Xmm(0), Xmm(1))),
            [0xc5, 0x7b, 0xc2, 0xf9, 0x00]
        );
        assert_eq!(bytes(|a| a.ftest(Xmm(15))), [0xc4, 0x42, 0x79, 0x17, 0xff]);
        assert_eq!(
            bytes(|a| a.cvt_to_int(Sz::Q, RAX, Xmm(1))),
            [0xc4, 0xe1, 0xfb, 0x2c, 0xc1]
        );
        assert_eq!(
            bytes(|a| a.fnmadd(Xmm(0), Xmm(1), Xmm(2))),
            [0xc4, 0xe2, 0xf1, 0xbd, 0xc2]
        );
        assert_eq!(bytes(|a| a.jmp_reg(R11)), [0x41, 0xff, 0xe3]);
        assert_eq!(bytes(|a| a.call_reg(Mem::B(RAX, 8))), [0xff, 0x50, 0x08]);
    }

    #[test]
    fn labels_and_literals() {
        // A short backward jump, a long forward one, and a literal past
        // an immediate.
        let code = bytes(|a| {
            let (top, end) = (a.new_label(), a.new_label());
            a.bind(top);
            a.jcc(Cond::Ne, end);
            a.jmp(top);
            a.bind(end);
            a.cmp(Sz::Q, RAX, Mem::K(0x1122_3344_5566_7788));
            a.ret();
        });
        assert_eq!(code[..8], [0x0f, 0x85, 2, 0, 0, 0, 0xeb, 0xf8]);
        // `cmp rax, [rip + 1]`: the pool is aligned past the `ret`.
        assert_eq!(code[8..16], [0x48, 0x3b, 0x05, 1, 0, 0, 0, 0xc3]);
        assert_eq!(code[16..24], 0x1122_3344_5566_7788u64.to_le_bytes());
    }
}
