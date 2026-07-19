//! An x86-64 assembler: instruction encodings and label fixups.
//!
//! Deliberately ignorant of the machine IR, exactly as its aarch64 sibling
//! (`aarch64_asm`) is. It knows how to encode instructions and how to patch a
//! branch once its target is known, and nothing else. Every encoding here is
//! checked two ways: the execution tests at the bottom map the bytes into memory
//! and *run* them, and [`tests::matches_system_assembler`] diffs the bytes against
//! what the GNU assembler emits for the same mnemonics.
//!
//! # What is different from a fixed-width ISA
//!
//! x86-64 instructions are variable length, so the buffer is a `Vec<u8>`, not a
//! `Vec<u32>`, and a branch displacement is a byte count rather than a word count.
//! [`Asm::finish`] pads the tail to a 4-byte boundary and repacks into `u32`s only
//! because the code allocator's interface speaks words; the padding is `int3`
//! (`0xCC`) and sits past the final `ret`, so it never executes.
//!
//! # Two-address, and what that forces on the caller
//!
//! Most arithmetic here is two-address: `add d, s` computes `d = d + s`. The
//! machine IR is three-address, so the *encoder* (`x64`) — not this file —
//! materializes `d = a op b` as a move plus an in-place op. This file just offers
//! the primitives.
//!
//! Immediate forms are offered as fallible constructors (`try_*`) wherever the
//! encoding has a limited range, matching the aarch64 assembler's contract: the
//! caller decides whether to materialize into a scratch register or decline.

/// A general-purpose register, `rax`..`r15`, numbered in the hardware's order
/// (`rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7, r8=8, ..., r15=15`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Gpr(pub u8);

/// An SSE register, `xmm0`..`xmm15`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Xmm(pub u8);

pub const RAX: Gpr = Gpr(0);
pub const RCX: Gpr = Gpr(1);
pub const RDX: Gpr = Gpr(2);
pub const RSP: Gpr = Gpr(4);
#[allow(dead_code)]
pub const RBP: Gpr = Gpr(5);

/// An x86 condition code, as the low nibble the `jcc`/`setcc` opcodes take.
///
/// Not the IR's `Cc`: the mapping from an IR compare to one of these depends on
/// whether the operands were integers or floats (NaN, and the fact that `comisd`
/// only ever sets the *unsigned* flags), so the translation belongs to the caller.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Cond {
    E = 0x4,
    Ne = 0x5,
    /// Unsigned below / above, and the forms `comisd` produces: an unordered
    /// compare sets CF, so `Ae`/`A` (CF clear) are exactly the tests that read
    /// false against a NaN.
    B = 0x2,
    Ae = 0x3,
    Be = 0x6,
    A = 0x7,
    /// Signed.
    L = 0xC,
    Ge = 0xD,
    Le = 0xE,
    G = 0xF,
    /// Sign flag: set iff the result is negative. Used for the sign-agreement test
    /// in the floor divides.
    S = 0x8,
    Ns = 0x9,
    /// Parity: set on an unordered float compare.
    P = 0xA,
    Np = 0xB,
}

impl Cond {
    /// The condition that holds exactly when this one does not. Every x86
    /// condition pair differs only in bit 0, so the inverse is a single flip.
    pub fn invert(self) -> Cond {
        // Safe: every value below is one of the twelve variants, and flipping bit
        // 0 maps each to its complement, which is also a variant.
        match self {
            Cond::E => Cond::Ne,
            Cond::Ne => Cond::E,
            Cond::B => Cond::Ae,
            Cond::Ae => Cond::B,
            Cond::Be => Cond::A,
            Cond::A => Cond::Be,
            Cond::L => Cond::Ge,
            Cond::Ge => Cond::L,
            Cond::Le => Cond::G,
            Cond::G => Cond::Le,
            Cond::S => Cond::Ns,
            Cond::Ns => Cond::S,
            Cond::P => Cond::Np,
            Cond::Np => Cond::P,
        }
    }
}

/// A branch target, resolved by [`Asm::bind`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Label(u32);

struct Fixup {
    /// Byte offset of the 4-byte rel32 field to patch.
    at: usize,
    /// Byte offset of the *end* of the branch instruction — rel32 is measured from
    /// here.
    end: usize,
    label: Label,
}

#[derive(Default)]
pub struct Asm {
    bytes: Vec<u8>,
    /// Byte offset of each bound label; `u32::MAX` until bound.
    labels: Vec<u32>,
    fixups: Vec<Fixup>,
}

const UNBOUND: u32 = u32::MAX;

impl Asm {
    pub fn new() -> Self {
        Asm::default()
    }

    /// Current position, in bytes from the start of the buffer.
    pub fn offset(&self) -> usize {
        self.bytes.len()
    }

    pub fn new_label(&mut self) -> Label {
        let l = Label(self.labels.len() as u32);
        self.labels.push(UNBOUND);
        l
    }

    /// Bind a label here, dropping a `jmp` that was about to jump to it.
    ///
    /// An unconditional `jmp rel32` that is the last instruction before its own
    /// target is a no-op. Catching it here rather than in the encoder keeps the
    /// rule general: it is a property of the branch and the label, so it holds
    /// whatever emitted the branch — a block's jump, the fallthrough arm of a
    /// two-way branch, or the last exit stub's return to the epilogue. Only the
    /// *last* instruction, and only the unconditional `jmp` (5 bytes: `E9` + rel32):
    /// a `jcc` to the next instruction still has to test, so it must stay.
    pub fn bind(&mut self, l: Label) {
        debug_assert_eq!(self.labels[l.0 as usize], UNBOUND, "label bound twice");

        if let Some(f) = self.fixups.last()
            && f.end == self.bytes.len()
            && f.label == l
            && f.at + 4 == f.end
            && self.bytes[f.at - 1] == 0xE9
        {
            self.fixups.pop();
            self.bytes.truncate(self.bytes.len() - 5);
        }

        self.labels[l.0 as usize] = self.bytes.len() as u32;
    }

    /// Resolve every branch and hand back the encoded words.
    ///
    /// The byte stream is padded to a multiple of four with `int3` and repacked
    /// little-endian, because the code allocator's interface is word-oriented (see
    /// the module docs).
    pub fn finish(mut self) -> Vec<u32> {
        for f in &self.fixups {
            let target = self.labels[f.label.0 as usize];
            assert_ne!(target, UNBOUND, "branch to an unbound label");
            let delta = target as i64 - f.end as i64;
            assert!(
                (i32::MIN as i64..=i32::MAX as i64).contains(&delta),
                "branch out of rel32 range"
            );
            self.bytes[f.at..f.at + 4].copy_from_slice(&(delta as i32).to_le_bytes());
        }

        while self.bytes.len() % 4 != 0 {
            self.bytes.push(0xCC);
        }
        self.bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn emit(&mut self, b: u8) {
        self.bytes.push(b);
    }

    fn emit_all(&mut self, bs: &[u8]) {
        self.bytes.extend_from_slice(bs);
    }

    // --- encoding cores -----------------------------------------------------
    //
    // Legacy prefixes first, then REX, then opcode, then ModRM/SIB/disp/imm.

    /// Register-to-register: ModRM `mod=11`, `reg` and `rm` naming the two
    /// registers. `prefixes` are legacy bytes (`0x66`, `0xF2`, ...); `opcode` is
    /// one or two bytes (a leading `0x0F` for the two-byte maps).
    fn rr(&mut self, prefixes: &[u8], rex_w: bool, opcode: &[u8], reg: u8, rm: u8) {
        self.emit_all(prefixes);
        let rex = 0x40 | ((rex_w as u8) << 3) | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1);
        // A bare `0x40` REX is only needed to reach the low-byte registers
        // (spl/bpl/sil/dil); callers that need it pass `rex_w=false` with a high
        // register and get it, and the 8-bit ops force it explicitly.
        if rex != 0x40 || rex_w {
            self.emit(rex);
        }
        self.emit_all(opcode);
        self.emit(0b11_000_000 | ((reg & 7) << 3) | (rm & 7));
    }

    /// Like [`Self::rr`], but for an 8-bit register operand: a REX prefix is
    /// emitted whenever one is needed — either to reach `r8b`–`r15b` (REX.B/R) or
    /// to select `spl`/`bpl`/`sil`/`dil` over the legacy high-byte registers (a
    /// bare `0x40` for `rm`/`reg` in `4..=7`). It is omitted for `al`/`cl`/`dl`/`bl`,
    /// which is what the GNU assembler does.
    fn rr_rex8(&mut self, prefixes: &[u8], opcode: &[u8], reg: u8, rm: u8) {
        self.emit_all(prefixes);
        if needs_byte_rex(reg, rm) {
            self.emit(0x40 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1));
        }
        self.emit_all(opcode);
        self.emit(0b11_000_000 | ((reg & 7) << 3) | (rm & 7));
    }

    /// Register-to-memory `[base + disp]`. `reg` is the ModRM.reg field (a
    /// register number or an opcode extension); `base` is the address register.
    fn rm(&mut self, prefixes: &[u8], rex_w: bool, opcode: &[u8], reg: u8, base: u8, disp: i32) {
        self.emit_all(prefixes);
        let rex = 0x40 | ((rex_w as u8) << 3) | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 || rex_w {
            self.emit(rex);
        }
        self.emit_all(opcode);
        self.emit_mem_operand(reg, base, disp);
    }

    /// The ModRM (+ optional SIB + disp) for a `[base + disp]` operand.
    fn emit_mem_operand(&mut self, reg: u8, base: u8, disp: i32) {
        let low = base & 7;
        // rsp/r12 (`rm == 100`) always need a SIB byte to express "no index".
        let needs_sib = low == 4;
        // rbp/r13 (`rm == 101`) cannot use `mod=00` — that encoding means
        // rip-relative — so a zero displacement there still takes an explicit
        // `mod=01` disp8.
        let (mod_bits, emit_disp): (u8, u8) = if disp == 0 && low != 5 {
            (0b00, 0)
        } else if (-128..=127).contains(&disp) {
            (0b01, 1)
        } else {
            (0b10, 4)
        };

        self.emit(mod_bits << 6 | ((reg & 7) << 3) | (if needs_sib { 0b100 } else { low }));
        if needs_sib {
            // scale=0, index=100 (none), base=low.
            self.emit(0b00_100_000 | low);
        }
        match emit_disp {
            1 => self.emit(disp as u8),
            4 => self.emit_all(&(disp).to_le_bytes()),
            _ => {}
        }
    }

    // --- moves --------------------------------------------------------------

    /// `mov d, s` (64-bit) — `mov r/m64, r64` (`89 /r`), so `reg=s, rm=d`. The
    /// store form is the one the GNU assembler emits for a register-register move.
    pub fn mov(&mut self, d: Gpr, s: Gpr) {
        if d != s {
            self.rr(&[], true, &[0x89], s.0, d.0);
        }
    }

    /// `movsd d, s` — `F2 0F 10 /r`.
    pub fn movsd(&mut self, d: Xmm, s: Xmm) {
        if d != s {
            self.rr(&[0xF2], false, &[0x0F, 0x10], d.0, s.0);
        }
    }

    /// `mov d, #imm`, in the shortest of the three forms.
    pub fn mov_imm(&mut self, d: Gpr, imm: i64) {
        if (0..=0xFFFF_FFFF).contains(&imm) {
            // `mov r32, imm32` zero-extends into the full 64-bit register.
            self.emit_all(&[]);
            if d.0 >= 8 {
                self.emit(0x41); // REX.B
            }
            self.emit(0xB8 | (d.0 & 7));
            self.emit_all(&(imm as u32).to_le_bytes());
        } else if (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
            // `mov r/m64, imm32`, sign-extended: `REX.W C7 /0 id`.
            self.rr(&[], true, &[0xC7], 0, d.0);
            self.emit_all(&(imm as i32).to_le_bytes());
        } else {
            // `movabs r64, imm64`: `REX.W B8+rd io`.
            self.emit(0x48 | ((d.0 >> 3) & 1));
            self.emit(0xB8 | (d.0 & 7));
            self.emit_all(&imm.to_le_bytes());
        }
    }

    // --- integer arithmetic (two-address: `d = d op s`) ---------------------

    /// `add d, s` — `01 /r` (`add r/m64, r64`), `reg=s, rm=d`.
    pub fn add(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x01], s.0, d.0);
    }

    pub fn sub(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x29], s.0, d.0);
    }

    /// `imul d, s` — `0F AF /r` (`imul r64, r/m64`), `reg=d, rm=s`.
    pub fn imul(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x0F, 0xAF], d.0, s.0);
    }

    pub fn and(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x21], s.0, d.0);
    }

    pub fn or(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x09], s.0, d.0);
    }

    pub fn xor(&mut self, d: Gpr, s: Gpr) {
        self.rr(&[], true, &[0x31], s.0, d.0);
    }

    /// `neg d` — `F7 /3`.
    pub fn neg(&mut self, d: Gpr) {
        self.rr(&[], true, &[0xF7], 3, d.0);
    }

    /// `not d` — `F7 /2`.
    pub fn not(&mut self, d: Gpr) {
        self.rr(&[], true, &[0xF7], 2, d.0);
    }

    /// `idiv s` — `F7 /7`. Divides `rdx:rax` by `s`; quotient to `rax`, remainder
    /// to `rdx`. Pair with [`Self::cqo`] to sign-extend the dividend first.
    pub fn idiv(&mut self, s: Gpr) {
        self.rr(&[], true, &[0xF7], 7, s.0);
    }

    /// `cqo` — sign-extend `rax` into `rdx:rax`. `REX.W 99`.
    pub fn cqo(&mut self) {
        self.emit(0x48);
        self.emit(0x99);
    }

    /// `shl d, cl` — `D3 /4`. Only the `cl` (variable) forms are provided; the IR
    /// never selects a shift, so a constant form would be dead code.
    pub fn shl_cl(&mut self, d: Gpr) {
        self.rr(&[], true, &[0xD3], 4, d.0);
    }

    pub fn shr_cl(&mut self, d: Gpr) {
        self.rr(&[], true, &[0xD3], 5, d.0);
    }

    pub fn sar_cl(&mut self, d: Gpr) {
        self.rr(&[], true, &[0xD3], 7, d.0);
    }

    /// `op d, #imm`, sign-extended. Uses the compact 8-bit form (`83 /ext ib`)
    /// when the immediate fits a signed byte, else the 32-bit form (`81 /ext id`).
    /// `ext` selects the operation.
    fn alu_imm(&mut self, ext: u8, d: Gpr, imm: i32) {
        if let Ok(imm8) = i8::try_from(imm) {
            self.rr(&[], true, &[0x83], ext, d.0);
            self.emit(imm8 as u8);
        } else {
            self.rr(&[], true, &[0x81], ext, d.0);
            self.emit_all(&imm.to_le_bytes());
        }
    }

    /// `add d, #imm` if `imm` fits a sign-extended 32-bit field. The immediate ALU
    /// ops the IR can produce are add/sub only.
    pub fn try_add_imm(&mut self, d: Gpr, imm: i64) -> bool {
        let Ok(imm) = i32::try_from(imm) else {
            return false;
        };
        self.alu_imm(0, d, imm);
        true
    }

    pub fn try_sub_imm(&mut self, d: Gpr, imm: i64) -> bool {
        let Ok(imm) = i32::try_from(imm) else {
            return false;
        };
        self.alu_imm(5, d, imm);
        true
    }

    // --- comparison ---------------------------------------------------------

    /// `cmp a, b` — `39 /r` (`cmp r/m64, r64`), `reg=b, rm=a`; sets flags for
    /// `a - b`.
    pub fn cmp(&mut self, a: Gpr, b: Gpr) {
        self.rr(&[], true, &[0x39], b.0, a.0);
    }

    /// `cmp a, #imm` if `imm` fits a sign-extended 32-bit field — `81 /7 id`.
    pub fn try_cmp_imm(&mut self, a: Gpr, imm: i64) -> bool {
        let Ok(imm) = i32::try_from(imm) else {
            return false;
        };
        self.alu_imm(7, a, imm);
        true
    }

    /// `test a, b` — `85 /r`; sets flags for `a & b`.
    pub fn test(&mut self, a: Gpr, b: Gpr) {
        self.rr(&[], true, &[0x85], b.0, a.0);
    }

    /// `setcc d8` — `0F 90+cc`. Writes the low byte of `d`; the encoder pairs it
    /// with [`Self::movzx8`] to clear the rest.
    pub fn setcc(&mut self, cond: Cond, d: Gpr) {
        // `/0` extension, 8-bit operand: force REX so `spl`/`r11b` etc. resolve.
        self.rr_rex8(&[], &[0x0F, 0x90 | cond as u8], 0, d.0);
    }

    /// `movzx d, s8` — `0F B6 /r`, zero-extend a byte register into `d`.
    pub fn movzx8(&mut self, d: Gpr, s: Gpr) {
        // REX.W for the 64-bit destination; the byte source still needs the prefix
        // present to name the low-byte registers, which REX.W supplies.
        self.rr(&[], true, &[0x0F, 0xB6], d.0, s.0);
    }

    // --- floating point (two-address: `d = d op s`) -------------------------

    pub fn addsd(&mut self, d: Xmm, s: Xmm) {
        self.rr(&[0xF2], false, &[0x0F, 0x58], d.0, s.0);
    }

    pub fn subsd(&mut self, d: Xmm, s: Xmm) {
        self.rr(&[0xF2], false, &[0x0F, 0x5C], d.0, s.0);
    }

    pub fn mulsd(&mut self, d: Xmm, s: Xmm) {
        self.rr(&[0xF2], false, &[0x0F, 0x59], d.0, s.0);
    }

    pub fn divsd(&mut self, d: Xmm, s: Xmm) {
        self.rr(&[0xF2], false, &[0x0F, 0x5E], d.0, s.0);
    }

    /// `xorpd d, s` — `66 0F 57 /r`. Used with a sign-bit mask to negate.
    pub fn xorpd(&mut self, d: Xmm, s: Xmm) {
        self.rr(&[0x66], false, &[0x0F, 0x57], d.0, s.0);
    }

    /// `comisd a, b` — `66 0F 2F /r`. Ordered compare; an unordered result (either
    /// operand NaN) sets `ZF=PF=CF=1`, which is why the float condition mapping is
    /// not the integer one.
    pub fn comisd(&mut self, a: Xmm, b: Xmm) {
        self.rr(&[0x66], false, &[0x0F, 0x2F], a.0, b.0);
    }

    /// `cvtsi2sd d, s` — `F2 REX.W 0F 2A /r`, signed 64-bit integer to double. A
    /// real conversion.
    pub fn cvtsi2sd(&mut self, d: Xmm, s: Gpr) {
        self.rr(&[0xF2], true, &[0x0F, 0x2A], d.0, s.0);
    }

    /// `movq d, s` — `66 REX.W 0F 6E /r`, general register bits into an xmm. A
    /// reinterpret, no conversion.
    pub fn movq_to_xmm(&mut self, d: Xmm, s: Gpr) {
        self.rr(&[0x66], true, &[0x0F, 0x6E], d.0, s.0);
    }

    /// `movq d, s` — `66 REX.W 0F 7E /r`, xmm bits into a general register. The
    /// `7E` store form names the xmm in ModRM.reg and the gpr in ModRM.rm.
    pub fn movq_to_gpr(&mut self, d: Gpr, s: Xmm) {
        self.rr(&[0x66], true, &[0x0F, 0x7E], s.0, d.0);
    }

    // --- memory -------------------------------------------------------------
    //
    // Displacements are always in range: x86 addresses with a 32-bit signed
    // displacement, which covers every offset the backend produces, so these are
    // infallible where the aarch64 ones are `try_`.

    /// `mov d, [base + off]` — `8B /r`.
    pub fn load(&mut self, d: Gpr, base: Gpr, off: i32) {
        self.rm(&[], true, &[0x8B], d.0, base.0, off);
    }

    /// `mov [base + off], s` — `89 /r`.
    pub fn store(&mut self, base: Gpr, off: i32, s: Gpr) {
        self.rm(&[], true, &[0x89], s.0, base.0, off);
    }

    /// `movzx d, byte [base + off]` — `0F B6 /r`, zero-extending.
    pub fn load8(&mut self, d: Gpr, base: Gpr, off: i32) {
        self.rm(&[], true, &[0x0F, 0xB6], d.0, base.0, off);
    }

    /// `mov byte [base + off], s8` — `88 /r`. A REX prefix appears when the byte
    /// source needs one (`r8b`–`r15b`, or `spl`–`dil`) or the base is an extended
    /// register; `al`/`bl` with a low base need none, matching GNU.
    pub fn store8(&mut self, base: Gpr, off: i32, s: Gpr) {
        let rex_b = (base.0 >> 3) & 1;
        let rex_r = (s.0 >> 3) & 1;
        if rex_b != 0 || rex_r != 0 || (4..=7).contains(&s.0) {
            self.emit(0x40 | (rex_r << 2) | rex_b);
        }
        self.emit(0x88);
        self.emit_mem_operand(s.0, base.0, off);
    }

    /// `movsd d, [base + off]` — `F2 0F 10 /r`.
    pub fn load_f(&mut self, d: Xmm, base: Gpr, off: i32) {
        self.rm(&[0xF2], false, &[0x0F, 0x10], d.0, base.0, off);
    }

    /// `movsd [base + off], s` — `F2 0F 11 /r`.
    pub fn store_f(&mut self, base: Gpr, off: i32, s: Xmm) {
        self.rm(&[0xF2], false, &[0x0F, 0x11], s.0, base.0, off);
    }

    // --- stack --------------------------------------------------------------

    pub fn push(&mut self, r: Gpr) {
        if r.0 >= 8 {
            self.emit(0x41); // REX.B
        }
        self.emit(0x50 | (r.0 & 7));
    }

    pub fn pop(&mut self, r: Gpr) {
        if r.0 >= 8 {
            self.emit(0x41);
        }
        self.emit(0x58 | (r.0 & 7));
    }

    // --- control flow -------------------------------------------------------

    /// `jmp rel32` — `E9 id`.
    pub fn jmp(&mut self, l: Label) {
        self.emit(0xE9);
        self.push_rel32(l);
    }

    /// `jcc rel32` — `0F 80+cc id`.
    pub fn jcc(&mut self, cond: Cond, l: Label) {
        self.emit(0x0F);
        self.emit(0x80 | cond as u8);
        self.push_rel32(l);
    }

    fn push_rel32(&mut self, l: Label) {
        let at = self.bytes.len();
        self.emit_all(&[0, 0, 0, 0]);
        self.fixups.push(Fixup {
            at,
            end: self.bytes.len(),
            label: l,
        });
    }

    pub fn ret(&mut self) {
        self.emit(0xC3);
    }
}

/// Whether an 8-bit register operand needs a REX prefix: to reach `r8b`–`r15b`
/// (bit 3 set), or to select `spl`/`bpl`/`sil`/`dil` (`4..=7`) over the legacy
/// high-byte registers. `al`/`cl`/`dl`/`bl` need none.
fn needs_byte_rex(reg: u8, rm: u8) -> bool {
    reg >= 8 || rm >= 8 || (4..=7).contains(&reg) || (4..=7).contains(&rm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::code::CodeBuf;

    fn run_args(words: &[u32], a: u64, b: u64) -> u64 {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut buf = CodeBuf::new(bytes.len()).expect("mmap");
        buf.write(|code| code.copy_from_slice(&bytes));
        let code = buf.finalize();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code.entry()) };
        f(a, b)
    }

    const RDI: Gpr = Gpr(7);
    const RSI: Gpr = Gpr(6);
    const R10: Gpr = Gpr(10);

    /// Move the first argument (rdi) into rax and return it, so the harness has a
    /// way to observe a register's value.
    fn ret_rax(a: &mut Asm) {
        a.mov(RAX, R10);
        a.ret();
    }

    #[test]
    fn mov_imm_roundtrips() {
        for &v in &[
            0i64,
            1,
            -1,
            42,
            -42,
            0xffff,
            0x1_0000,
            -0x1_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0x1234_5678_9abc_def0u64 as i64,
            i64::MIN,
            i64::MAX,
        ] {
            let mut a = Asm::new();
            a.mov_imm(R10, v);
            ret_rax(&mut a);
            let got = run_args(&a.finish(), 0, 0);
            assert_eq!(got, v as u64, "mov_imm {v:#x}");
        }
    }

    #[test]
    fn arithmetic() {
        // rax = rdi + rsi
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.add(RAX, RSI);
        a.ret();
        assert_eq!(run_args(&a.finish(), 3, 39), 42);

        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.sub(RAX, RSI);
        a.ret();
        assert_eq!(run_args(&a.finish(), 50, 8), 42);

        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.imul(RAX, RSI);
        a.ret();
        assert_eq!(run_args(&a.finish(), 6, 7), 42);

        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.neg(RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 0) as i64, -42);

        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.not(RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), 41, 0) as i64, !41i64);

        // High registers, to exercise the REX.B path.
        let mut a = Asm::new();
        a.mov(Gpr(11), RDI);
        a.mov(Gpr(12), RSI);
        a.add(Gpr(11), Gpr(12));
        a.mov(RAX, Gpr(11));
        a.ret();
        assert_eq!(run_args(&a.finish(), 20, 22), 42);
    }

    #[test]
    fn add_imm_positive_and_negative() {
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        assert!(a.try_add_imm(RAX, 2));
        assert!(a.try_sub_imm(RAX, 44));
        a.ret();
        assert_eq!(run_args(&a.finish(), 84, 0), 42);
    }

    #[test]
    fn shifts() {
        // rax = rdi >>a rsi  (arithmetic)
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.mov(RCX, RSI);
        a.sar_cl(RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), (-8i64) as u64, 1) as i64, -4);

        // rax = rdi >>l rsi  (logical)
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.mov(RCX, RSI);
        a.shr_cl(RAX);
        a.ret();
        assert_eq!(
            run_args(&a.finish(), (-8i64) as u64, 1),
            0x7fff_ffff_ffff_fffc
        );

        // rax = rdi << rsi
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.mov(RCX, RSI);
        a.shl_cl(RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), 3, 4), 48);
    }

    #[test]
    fn division() {
        // rax = rdi / rsi (truncating), via cqo + idiv
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.cqo();
        a.idiv(RSI);
        a.ret();
        let div = a.finish();
        assert_eq!(run_args(&div, 42, 5) as i64, 8);
        assert_eq!(run_args(&div, (-7i64) as u64, 2) as i64, -3); // trunc

        // Remainder is in rdx.
        let mut a = Asm::new();
        a.mov(RAX, RDI);
        a.cqo();
        a.idiv(RSI);
        a.mov(RAX, RDX);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 5) as i64, 2);
    }

    #[test]
    fn setcc_conditions() {
        let cases = [
            (Cond::E, 5i64, 5i64, 1u64),
            (Cond::E, 5, 6, 0),
            (Cond::Ne, 5, 6, 1),
            (Cond::L, -1, 1, 1),
            (Cond::L, 1, -1, 0),
            (Cond::Le, 5, 5, 1),
            (Cond::G, 1, -1, 1),
            (Cond::Ge, 5, 5, 1),
            (Cond::Ge, 4, 5, 0),
        ];
        for (cond, x, y, want) in cases {
            let mut a = Asm::new();
            a.cmp(RDI, RSI);
            a.setcc(cond, RAX);
            a.movzx8(RAX, RAX);
            a.ret();
            let got = run_args(&a.finish(), x as u64, y as u64);
            assert_eq!(got, want, "setcc {cond:?} with {x} vs {y}");
        }
    }

    #[test]
    fn cmp_imm() {
        let mut a = Asm::new();
        assert!(a.try_cmp_imm(RDI, 42));
        a.setcc(Cond::E, RAX);
        a.movzx8(RAX, RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 0), 1);

        let mut a = Asm::new();
        assert!(a.try_cmp_imm(RDI, -7));
        a.setcc(Cond::E, RAX);
        a.movzx8(RAX, RAX);
        a.ret();
        assert_eq!(run_args(&a.finish(), (-7i64) as u64, 0), 1);
    }

    #[test]
    fn float_ops() {
        // (rdi as f64 + rsi as f64) * 2.0, via bit moves in both directions.
        let mut a = Asm::new();
        a.cvtsi2sd(Xmm(0), RDI);
        a.cvtsi2sd(Xmm(1), RSI);
        a.addsd(Xmm(0), Xmm(1));
        a.mov_imm(R10, 2.0f64.to_bits() as i64);
        a.movq_to_xmm(Xmm(1), R10);
        a.mulsd(Xmm(0), Xmm(1));
        a.movq_to_gpr(RAX, Xmm(0));
        a.ret();
        let bits = run_args(&a.finish(), 3, 4);
        assert_eq!(f64::from_bits(bits), 14.0);
    }

    /// The float compare mapping exists because of NaN. `comisd a,b` sets CF on an
    /// unordered result, so `Ae`/`A` (CF clear) are the tests that read false
    /// against a NaN — which is Lua's rule for every comparison but `~=`.
    #[test]
    fn float_compare_is_nan_safe() {
        let nan = f64::NAN.to_bits();
        let one = 1.0f64.to_bits();

        // lt(a,b) as `comisd b, a; seta`: true iff b > a and ordered.
        let lt = |x: u64, y: u64| {
            let mut a = Asm::new();
            a.movq_to_xmm(Xmm(0), RDI);
            a.movq_to_xmm(Xmm(1), RSI);
            a.comisd(Xmm(1), Xmm(0)); // b vs a
            a.setcc(Cond::A, RAX);
            a.movzx8(RAX, RAX);
            a.ret();
            run_args(&a.finish(), x, y)
        };
        assert_eq!(lt(one, nan), 0, "1 < NaN is false");
        assert_eq!(lt(nan, one), 0, "NaN < 1 is false");
        assert_eq!(lt(0.5f64.to_bits(), one), 1, "0.5 < 1");
        assert_eq!(lt(2.0f64.to_bits(), one), 0, "2 < 1 is false");
    }

    #[test]
    fn memory_round_trip() {
        let mut mem = [0u64; 4];
        let ptr = mem.as_mut_ptr() as u64;

        // [rdi + 16] <- rsi; rax <- [rdi + 16]
        let mut a = Asm::new();
        a.store(RDI, 16, RSI);
        a.load(RAX, RDI, 16);
        a.ret();
        assert_eq!(run_args(&a.finish(), ptr, 0xdead_beef), 0xdead_beef);
        assert_eq!(mem[2], 0xdead_beef);

        // Byte access at a non-scaled offset.
        let mut a = Asm::new();
        a.store8(RDI, 9, RSI);
        a.load8(RAX, RDI, 9);
        a.ret();
        assert_eq!(run_args(&a.finish(), ptr, 0xff), 0xff);

        // A base whose low three bits are 100 (r12) forces a SIB byte.
        let mut a = Asm::new();
        a.mov(Gpr(12), RDI);
        a.store(Gpr(12), 24, RSI);
        a.load(RAX, Gpr(12), 24);
        a.ret();
        assert_eq!(run_args(&a.finish(), ptr, 0x1234), 0x1234);
    }

    #[test]
    fn branches_and_labels() {
        // if rdi != 0 { 42 } else { 7 }
        let mut a = Asm::new();
        let taken = a.new_label();
        let done = a.new_label();
        a.test(RDI, RDI);
        a.jcc(Cond::Ne, taken);
        a.mov_imm(RAX, 7);
        a.jmp(done);
        a.bind(taken);
        a.mov_imm(RAX, 42);
        a.bind(done);
        a.ret();
        let words = a.finish();
        assert_eq!(run_args(&words, 1, 0), 42);
        assert_eq!(run_args(&words, 0, 0), 7);
    }

    /// A backward branch, i.e. an actual loop: sum 1..=rdi.
    #[test]
    fn backward_branch() {
        let mut a = Asm::new();
        let top = a.new_label();
        let done = a.new_label();
        a.mov_imm(RAX, 0); // acc
        a.bind(top);
        a.test(RDI, RDI);
        a.jcc(Cond::E, done);
        a.add(RAX, RDI);
        assert!(a.try_sub_imm(RDI, 1));
        a.jmp(top);
        a.bind(done);
        a.ret();
        let words = a.finish();
        assert_eq!(run_args(&words, 10, 0), 55);
        assert_eq!(run_args(&words, 0, 0), 0);
    }

    /// A `jmp` to the very next instruction is dropped; a conditional one is not.
    #[test]
    fn jmp_to_the_next_instruction_is_dropped() {
        let mut a = Asm::new();
        let l = a.new_label();
        a.mov_imm(RAX, 7);
        a.jmp(l);
        a.bind(l);
        a.ret();
        // mov (5 or 7 bytes) + ret (1), no jmp; padded to a word multiple.
        let n = a.finish().len();
        let mut a2 = Asm::new();
        a2.mov_imm(RAX, 7);
        a2.ret();
        assert_eq!(n, a2.finish().len(), "the jmp should be gone");
    }

    /// Every encoding, against the GNU assembler. Execution proves the semantics;
    /// this catches a byte that happens to run while differing from the canonical
    /// form (a stray prefix, say), which disassembles as garbage under objdump.
    ///
    /// Linux-only: it is a differential test against GNU `as`/`objdump`. On macOS
    /// (where the x86-64 backend runs under Rosetta) the system `as` is LLVM's and
    /// rejects the GAS-syntax reference, so there is no canonical form to compare to.
    #[cfg(target_os = "linux")]
    #[test]
    fn matches_system_assembler() {
        use std::io::Write;
        use std::process::Command;

        let mut a = Asm::new();
        // A representative spread: reg-reg, high regs, imm forms, mem with and
        // without SIB, float, and the setcc/movzx pair.
        a.mov(Gpr(3), Gpr(1)); // mov rbx, rcx
        a.mov(Gpr(11), Gpr(2)); // mov r11, rdx
        a.mov_imm(Gpr(0), 42);
        a.mov_imm(Gpr(1), -42);
        a.mov_imm(Gpr(2), 0xffff_ffff);
        a.mov_imm(Gpr(0), 0x1_0000_0000);
        a.add(Gpr(0), Gpr(1));
        a.sub(Gpr(0), Gpr(1));
        a.imul(Gpr(0), Gpr(1));
        a.and(Gpr(0), Gpr(1));
        a.or(Gpr(0), Gpr(1));
        a.xor(Gpr(0), Gpr(1));
        a.neg(Gpr(0));
        a.not(Gpr(0));
        a.cqo();
        a.idiv(Gpr(1));
        a.sar_cl(Gpr(0));
        a.shr_cl(Gpr(0));
        a.shl_cl(Gpr(0));
        assert!(a.try_add_imm(Gpr(0), 42));
        assert!(a.try_sub_imm(Gpr(0), 42));
        a.cmp(Gpr(1), Gpr(2));
        assert!(a.try_cmp_imm(Gpr(1), 42));
        a.test(Gpr(1), Gpr(2));
        a.setcc(Cond::E, Gpr(0));
        a.setcc(Cond::L, Gpr(11));
        a.movzx8(Gpr(0), Gpr(0));
        a.addsd(Xmm(0), Xmm(1));
        a.subsd(Xmm(0), Xmm(1));
        a.mulsd(Xmm(0), Xmm(1));
        a.divsd(Xmm(0), Xmm(1));
        a.xorpd(Xmm(0), Xmm(1));
        a.comisd(Xmm(1), Xmm(2));
        a.cvtsi2sd(Xmm(0), Gpr(1));
        a.movq_to_xmm(Xmm(0), Gpr(1));
        a.movq_to_gpr(Gpr(0), Xmm(1));
        a.load(Gpr(0), Gpr(1), 16);
        a.store(Gpr(1), 16, Gpr(0));
        a.load8(Gpr(0), Gpr(1), 9);
        a.store8(Gpr(1), 9, Gpr(0));
        a.load_f(Xmm(0), Gpr(1), 16);
        a.store_f(Gpr(1), 16, Xmm(0));
        a.load(Gpr(0), Gpr(4), 8); // rsp base: SIB
        a.load(Gpr(0), Gpr(12), 8); // r12 base: SIB + REX.B
        a.load(Gpr(0), Gpr(5), 0); // rbp base: disp8 forced
        a.push(Gpr(5));
        a.push(Gpr(12));
        a.pop(Gpr(12));
        a.pop(Gpr(5));
        a.ret();

        let got = a.finish();
        let got_bytes: Vec<u8> = got.iter().flat_map(|w| w.to_le_bytes()).collect();

        // Reassemble the same mnemonics with GNU `as` and diff the bytes. Skip if
        // the toolchain is absent rather than failing the suite.
        let asm_src = ASM_SRC;
        let dir = std::env::temp_dir();
        let src = dir.join("tcvm_x64_asm_test.s");
        let obj = dir.join("tcvm_x64_asm_test.o");
        if std::fs::File::create(&src)
            .and_then(|mut f| f.write_all(asm_src.as_bytes()))
            .is_err()
        {
            return;
        }
        let status = Command::new("as").arg("-o").arg(&obj).arg(&src).status();
        let Ok(status) = status else { return };
        if !status.success() {
            panic!("`as` failed to assemble the reference program");
        }
        let out = Command::new("objdump")
            .args(["-d", "--insn-width=16"])
            .arg(&obj)
            .output()
            .expect("objdump");
        let text = String::from_utf8_lossy(&out.stdout);
        let want: Vec<u8> = parse_objdump_bytes(&text);

        // `finish` pads the tail to a word boundary with `int3`; the reference has
        // no padding, so compare the instruction bytes and check the rest is fill.
        assert!(
            got_bytes.len() >= want.len() && got_bytes[want.len()..].iter().all(|&b| b == 0xCC),
            "trailing bytes are not int3 padding:\nours: {}",
            hex(&got_bytes)
        );
        assert_eq!(
            &got_bytes[..want.len()],
            want.as_slice(),
            "\nours:  {}\nwant:  {}",
            hex(&got_bytes[..want.len()]),
            hex(&want)
        );
    }

    fn hex(bs: &[u8]) -> String {
        bs.iter().map(|b| format!("{b:02x} ")).collect()
    }

    /// Pull the instruction-byte column out of `objdump -d` output.
    fn parse_objdump_bytes(text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut in_text = false;
        for line in text.lines() {
            if line.contains("<.text>:") {
                in_text = true;
                continue;
            }
            if !in_text {
                continue;
            }
            // Lines look like: "   0:\t48 89 cb             \tmov ...".
            let Some((_, rest)) = line.split_once('\t') else {
                continue;
            };
            let bytes_col = rest.split('\t').next().unwrap_or("");
            for tok in bytes_col.split_whitespace() {
                if let Ok(b) = u8::from_str_radix(tok, 16) {
                    out.push(b);
                }
            }
        }
        out
    }

    /// The GNU-assembler source mirroring the program built above, in AT&T syntax
    /// (`op src, dst`). Kept in lockstep with the `Asm` calls by hand.
    const ASM_SRC: &str = "\
.intel_syntax noprefix
.text
    mov rbx, rcx
    mov r11, rdx
    mov eax, 42
    mov rcx, -42
    mov edx, 0xffffffff
    movabs rax, 0x100000000
    add rax, rcx
    sub rax, rcx
    imul rax, rcx
    and rax, rcx
    or rax, rcx
    xor rax, rcx
    neg rax
    not rax
    cqo
    idiv rcx
    sar rax, cl
    shr rax, cl
    shl rax, cl
    add rax, 42
    sub rax, 42
    cmp rcx, rdx
    cmp rcx, 42
    test rcx, rdx
    sete al
    setl r11b
    movzx rax, al
    addsd xmm0, xmm1
    subsd xmm0, xmm1
    mulsd xmm0, xmm1
    divsd xmm0, xmm1
    xorpd xmm0, xmm1
    comisd xmm1, xmm2
    cvtsi2sd xmm0, rcx
    movq xmm0, rcx
    movq rax, xmm1
    mov rax, [rcx + 16]
    mov [rcx + 16], rax
    movzx rax, byte ptr [rcx + 9]
    mov byte ptr [rcx + 9], al
    movsd xmm0, [rcx + 16]
    movsd [rcx + 16], xmm0
    mov rax, [rsp + 8]
    mov rax, [r12 + 8]
    mov rax, [rbp + 0]
    push rbp
    push r12
    pop r12
    pop rbp
    ret
";
}
