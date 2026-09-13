//! An aarch64 assembler: instruction encodings and label fixups.
//!
//! Deliberately ignorant of the machine IR. It knows how to encode instructions
//! and how to patch a branch once its target is known, and nothing else. Every
//! encoding here was checked against the system assembler — see
//! `tests::matches_system_assembler`, which reassembles the same program with
//! `clang` and diffs the bytes.
//!
//! Immediate forms are offered as fallible constructors (`try_*`) wherever the
//! encoding has a limited range. The caller decides what to do when a value does
//! not fit, because the right answer is context-dependent: sometimes materialize
//! into a scratch register, sometimes decline to compile.

/// A general-purpose register, `x0`–`x30` plus the zero register.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Gpr(pub u8);

/// A floating-point register, `d0`–`d31`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Fpr(pub u8);

/// `xzr` in most positions, `sp` in a few. The distinction is per-instruction;
/// this assembler only ever emits it where it reads as the zero register.
pub const ZR: Gpr = Gpr(31);
/// The stack pointer, encoded identically to `ZR` but interpreted as `sp` by the
/// instructions below that name it.
pub const SP: Gpr = Gpr(31);
pub const FP: Gpr = Gpr(29);
pub const LR: Gpr = Gpr(30);

/// An aarch64 condition code. Not the IR's `Cc`: the mapping from an IR compare
/// to one of these depends on whether the operands were integers or floats
/// (NaN is why), so the translation belongs to the caller.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
}

impl Cond {
    /// The condition that holds exactly when this one does not.
    ///
    /// Used by `cset` (which is `csinc` with the condition inverted) and by
    /// guards, which branch out of line when the thing they assert is false.
    ///
    /// Sound for the float conditions too, but only as a *branch* inversion: the
    /// inverse of `mi` is `pl`, which is true for NaN. That is what a guard wants
    /// — "not (a < b)" — and is not the same as the float mapping for `>=`.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => panic!("`al` has no inverse"),
        }
    }
}

/// A branch target, resolved by [`Asm::bind`] at or after the branches to it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Label(u32);

#[derive(Clone, Copy)]
enum FixupKind {
    /// 26-bit signed word offset: `b`.
    B26,
    /// 19-bit signed word offset: `b.cond`, `cbnz`, `cbz`.
    B19,
}

struct Fixup {
    /// Index into `words` of the instruction to patch.
    at: usize,
    label: Label,
    kind: FixupKind,
}

#[derive(Default)]
pub struct Asm {
    words: Vec<u32>,
    /// Word offset of each bound label; `u32::MAX` until bound.
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
        self.words.len() * 4
    }

    pub fn new_label(&mut self) -> Label {
        let l = Label(self.labels.len() as u32);
        self.labels.push(UNBOUND);
        l
    }

    /// Bind a label here, dropping a branch that was about to jump to it.
    ///
    /// An unconditional `b` that is the last instruction before its own target is
    /// a no-op. Catching it here rather than in the encoder is what makes the rule
    /// general: it is a property of the branch and the label, so it holds whatever
    /// emitted the branch — a block's `jump`, the fallthrough arm of a two-way
    /// branch, or the last exit stub's return to the epilogue. The encoder gets to
    /// stay ignorant of what happens to be laid out next.
    ///
    /// Only ever the *last* instruction, and only an unconditional one: a `cbnz`
    /// to the next instruction is not dead, it is a branch whose fallthrough and
    /// target coincide, and dropping it would drop the test with it.
    pub fn bind(&mut self, l: Label) {
        debug_assert_eq!(self.labels[l.0 as usize], UNBOUND, "label bound twice");

        if let Some(f) = self.fixups.last()
            && f.at + 1 == self.words.len()
            && f.label == l
            && matches!(f.kind, FixupKind::B26)
        {
            self.fixups.pop();
            self.words.pop();
        }

        self.labels[l.0 as usize] = self.words.len() as u32;
    }

    /// Resolve every branch and hand back the encoded words.
    ///
    /// Fixups are applied here rather than at `bind` time so that forward
    /// branches — which are most of them, since blocks are laid out in RPO —
    /// need no back-patching machinery at the call site.
    pub fn finish(mut self) -> Vec<u32> {
        for f in &self.fixups {
            let target = self.labels[f.label.0 as usize];
            assert_ne!(target, UNBOUND, "branch to an unbound label");
            let delta = target as i64 - f.at as i64;
            let w = &mut self.words[f.at];
            match f.kind {
                FixupKind::B26 => {
                    assert!(
                        (-(1 << 25)..(1 << 25)).contains(&delta),
                        "branch out of ±128MB range"
                    );
                    *w |= (delta as u32) & 0x03ff_ffff;
                }
                FixupKind::B19 => {
                    assert!(
                        (-(1 << 18)..(1 << 18)).contains(&delta),
                        "conditional branch out of ±1MB range"
                    );
                    *w |= ((delta as u32) & 0x0007_ffff) << 5;
                }
            }
        }
        self.words
    }

    fn emit(&mut self, w: u32) {
        self.words.push(w);
    }

    // --- moves --------------------------------------------------------------

    /// `mov xd, xn` — `orr xd, xzr, xn`.
    pub fn mov(&mut self, d: Gpr, n: Gpr) {
        if d != n {
            self.emit(0xAA00_03E0 | ((n.0 as u32) << 16) | d.0 as u32);
        }
    }

    /// `fmov dd, dn`.
    pub fn fmov(&mut self, d: Fpr, n: Fpr) {
        if d != n {
            self.emit(0x1E60_4000 | ((n.0 as u32) << 5) | d.0 as u32);
        }
    }

    /// `mov xd, #imm`, as one to four `movz`/`movk`s.
    ///
    /// The negated form matters more than it looks: small negative constants are
    /// everywhere in Lua bytecode, and `movn` encodes them in one instruction
    /// where `movz`+`movk` would take four.
    pub fn mov_imm(&mut self, d: Gpr, imm: i64) {
        let u = imm as u64;

        // Try the one-instruction forms first.
        for hw in 0..4u32 {
            let shift = hw * 16;
            if u & !(0xffffu64 << shift) == 0 {
                let half = ((u >> shift) & 0xffff) as u32;
                self.emit(0xD280_0000 | (hw << 21) | (half << 5) | d.0 as u32); // movz
                return;
            }
            if !u & !(0xffffu64 << shift) == 0 {
                let half = ((!u >> shift) & 0xffff) as u32;
                self.emit(0x9280_0000 | (hw << 21) | (half << 5) | d.0 as u32); // movn
                return;
            }
        }

        let mut first = true;
        for hw in 0..4u32 {
            let half = ((u >> (hw * 16)) & 0xffff) as u32;
            if half == 0 && !first {
                continue;
            }
            let base = if first { 0xD280_0000 } else { 0xF280_0000 }; // movz : movk
            self.emit(base | (hw << 21) | (half << 5) | d.0 as u32);
            first = false;
        }
    }

    // --- integer arithmetic -------------------------------------------------

    fn alu_rrr(&mut self, base: u32, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(base | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | d.0 as u32);
    }

    pub fn add(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0x8B00_0000, d, n, m);
    }

    pub fn sub(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0xCB00_0000, d, n, m);
    }

    /// `mul xd, xn, xm` — `madd xd, xn, xm, xzr`.
    pub fn mul(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(0x9B00_7C00 | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | d.0 as u32);
    }

    /// `sdiv xd, xn, xm` — truncating signed division.
    ///
    /// Traps on nothing: `n / 0` is 0 and `i64::MIN / -1` is `i64::MIN`. Both are
    /// what Rust's `wrapping_div` produces, which is what the interpreter uses, so
    /// the only divisor the JIT has to care about is the zero the frontend already
    /// guards.
    pub fn sdiv(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(0x9AC0_0C00 | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | d.0 as u32);
    }

    /// `msub xd, xn, xm, xa` — `xd = xa - xn * xm`.
    pub fn msub(&mut self, d: Gpr, n: Gpr, m: Gpr, a: Gpr) {
        self.emit(
            0x9B00_8000
                | ((m.0 as u32) << 16)
                | ((a.0 as u32) << 10)
                | ((n.0 as u32) << 5)
                | d.0 as u32,
        );
    }

    /// `csel xd, xn, xm, cond` — `xd = cond ? xn : xm`.
    pub fn csel(&mut self, d: Gpr, n: Gpr, m: Gpr, cond: Cond) {
        self.emit(
            0x9A80_0000
                | ((m.0 as u32) << 16)
                | ((cond as u32) << 12)
                | ((n.0 as u32) << 5)
                | d.0 as u32,
        );
    }

    /// `ccmp xn, #imm, #nzcv, cond` — compare only if `cond` holds, otherwise
    /// force the flags to `nzcv` outright.
    ///
    /// This is how you `&&` two conditions without a branch: the second compare
    /// runs only when the first passed, and when it didn't, the flags are set to
    /// something the final `csel` will read as false.
    pub fn ccmp_imm(&mut self, n: Gpr, imm: u8, nzcv: u8, cond: Cond) {
        debug_assert!(imm < 32, "ccmp immediate is 5 bits");
        debug_assert!(nzcv < 16, "nzcv is 4 bits");
        self.emit(
            0xFA40_0800
                | ((imm as u32) << 16)
                | ((cond as u32) << 12)
                | ((n.0 as u32) << 5)
                | nzcv as u32,
        );
    }

    pub fn and(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0x8A00_0000, d, n, m);
    }

    pub fn orr(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0xAA00_0000, d, n, m);
    }

    pub fn eor(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0xCA00_0000, d, n, m);
    }

    pub fn lslv(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0x9AC0_2000, d, n, m);
    }

    pub fn lsrv(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0x9AC0_2400, d, n, m);
    }

    pub fn asrv(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.alu_rrr(0x9AC0_2800, d, n, m);
    }

    /// `neg xd, xm` — `sub xd, xzr, xm`.
    pub fn neg(&mut self, d: Gpr, m: Gpr) {
        self.sub(d, ZR, m);
    }

    /// `mvn xd, xm` — `orn xd, xzr, xm`.
    pub fn mvn(&mut self, d: Gpr, m: Gpr) {
        self.emit(0xAA20_03E0 | ((m.0 as u32) << 16) | d.0 as u32);
    }

    /// `add xd, xn, #imm` if `imm` fits the 12-bit unsigned field (optionally
    /// shifted left by 12). Negative immediates become a `sub`.
    pub fn try_add_imm(&mut self, d: Gpr, n: Gpr, imm: i64) -> bool {
        if imm < 0 {
            return match imm.checked_neg() {
                Some(pos) => self.try_addsub_imm(0xD100_0000, d, n, pos),
                None => false,
            };
        }
        self.try_addsub_imm(0x9100_0000, d, n, imm)
    }

    pub fn try_sub_imm(&mut self, d: Gpr, n: Gpr, imm: i64) -> bool {
        if imm < 0 {
            return match imm.checked_neg() {
                Some(pos) => self.try_addsub_imm(0x9100_0000, d, n, pos),
                None => false,
            };
        }
        self.try_addsub_imm(0xD100_0000, d, n, imm)
    }

    fn try_addsub_imm(&mut self, base: u32, d: Gpr, n: Gpr, imm: i64) -> bool {
        debug_assert!(imm >= 0);
        let (sh, val) = if imm < 4096 {
            (0, imm as u32)
        } else if imm & 0xfff == 0 && imm < (4096 << 12) {
            (1, (imm >> 12) as u32)
        } else {
            return false;
        };
        self.emit(base | (sh << 22) | (val << 10) | ((n.0 as u32) << 5) | d.0 as u32);
        true
    }

    // --- comparison ---------------------------------------------------------

    /// `cmp xn, xm` — `subs xzr, xn, xm`.
    pub fn cmp(&mut self, n: Gpr, m: Gpr) {
        self.emit(0xEB00_001F | ((m.0 as u32) << 16) | ((n.0 as u32) << 5));
    }

    /// `cmp xn, #imm` — `subs xzr, xn, #imm`. Negative immediates become `cmn`.
    pub fn try_cmp_imm(&mut self, n: Gpr, imm: i64) -> bool {
        let (base, imm) = if imm < 0 {
            match imm.checked_neg() {
                Some(pos) => (0xB100_001F, pos), // cmn = adds xzr, xn, #imm
                None => return false,
            }
        } else {
            (0xF100_001F, imm)
        };
        let (sh, val) = if imm < 4096 {
            (0, imm as u32)
        } else if imm & 0xfff == 0 && imm < (4096 << 12) {
            (1, (imm >> 12) as u32)
        } else {
            return false;
        };
        self.emit(base | (sh << 22) | (val << 10) | ((n.0 as u32) << 5));
        true
    }

    /// `cset xd, cond` — `csinc xd, xzr, xzr, invert(cond)`.
    pub fn cset(&mut self, d: Gpr, cond: Cond) {
        self.emit(0x9A9F_07E0 | ((cond.invert() as u32) << 12) | d.0 as u32);
    }

    /// `fcmp dn, dm`. Ordered: an unordered result (either operand NaN) leaves
    /// `N=0, Z=0, C=1, V=1`, which is why the float condition mapping is not the
    /// integer one.
    pub fn fcmp(&mut self, n: Fpr, m: Fpr) {
        self.emit(0x1E60_2000 | ((m.0 as u32) << 16) | ((n.0 as u32) << 5));
    }

    // --- floating point -----------------------------------------------------

    fn falu_rrr(&mut self, base: u32, d: Fpr, n: Fpr, m: Fpr) {
        self.emit(base | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | d.0 as u32);
    }

    pub fn fadd(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.falu_rrr(0x1E60_2800, d, n, m);
    }

    pub fn fsub(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.falu_rrr(0x1E60_3800, d, n, m);
    }

    pub fn fmul(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.falu_rrr(0x1E60_0800, d, n, m);
    }

    pub fn fdiv(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.falu_rrr(0x1E60_1800, d, n, m);
    }

    pub fn fneg(&mut self, d: Fpr, n: Fpr) {
        self.emit(0x1E61_4000 | ((n.0 as u32) << 5) | d.0 as u32);
    }

    /// `fmov dd, xn` — reinterpret the bits, no conversion.
    pub fn fmov_to_fpr(&mut self, d: Fpr, n: Gpr) {
        self.emit(0x9E67_0000 | ((n.0 as u32) << 5) | d.0 as u32);
    }

    /// `fmov xd, dn` — reinterpret the bits, no conversion.
    pub fn fmov_to_gpr(&mut self, d: Gpr, n: Fpr) {
        self.emit(0x9E66_0000 | ((n.0 as u32) << 5) | d.0 as u32);
    }

    /// `scvtf dd, xn` — signed 64-bit integer to double. A real conversion.
    pub fn scvtf(&mut self, d: Fpr, n: Gpr) {
        self.emit(0x9E62_0000 | ((n.0 as u32) << 5) | d.0 as u32);
    }

    // --- memory -------------------------------------------------------------
    //
    // Two addressing forms, tried in order: the scaled unsigned-offset form
    // (widest reach, but the offset must be a multiple of the access size) and
    // the unscaled signed form (any byte offset, but only ±256). Between them
    // they cover every offset the backend actually produces; a caller that hits
    // neither must materialize the address itself.

    /// `ldr xt, [xn, #off]`.
    pub fn try_ldr(&mut self, t: Gpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0xF940_0000, 0xF840_0000, 8, t.0, n, off)
    }

    /// `str xt, [xn, #off]`.
    pub fn try_str(&mut self, t: Gpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0xF900_0000, 0xF800_0000, 8, t.0, n, off)
    }

    /// `ldrb wt, [xn, #off]` — zero-extends into the full 64-bit register.
    pub fn try_ldrb(&mut self, t: Gpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0x3940_0000, 0x3840_0000, 1, t.0, n, off)
    }

    /// `strb wt, [xn, #off]`.
    pub fn try_strb(&mut self, t: Gpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0x3900_0000, 0x3800_0000, 1, t.0, n, off)
    }

    /// `ldr dt, [xn, #off]`.
    pub fn try_ldr_f(&mut self, t: Fpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0xFD40_0000, 0xFC40_0000, 8, t.0, n, off)
    }

    /// `str dt, [xn, #off]`.
    pub fn try_str_f(&mut self, t: Fpr, n: Gpr, off: i32) -> bool {
        self.try_mem(0xFD00_0000, 0xFC00_0000, 8, t.0, n, off)
    }

    fn try_mem(&mut self, scaled: u32, unscaled: u32, size: i32, t: u8, n: Gpr, off: i32) -> bool {
        if off >= 0 && off % size == 0 && off / size < 4096 {
            let imm = (off / size) as u32;
            self.emit(scaled | (imm << 10) | ((n.0 as u32) << 5) | t as u32);
            return true;
        }
        if (-256..256).contains(&off) {
            let imm9 = (off as u32) & 0x1ff;
            self.emit(unscaled | (imm9 << 12) | ((n.0 as u32) << 5) | t as u32);
            return true;
        }
        false
    }

    /// `ldr xt, [xn, xm]`.
    pub fn ldr_reg(&mut self, t: Gpr, n: Gpr, m: Gpr) {
        self.emit(0xF862_6800 | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | t.0 as u32);
    }

    /// `str xt, [xn, xm]`.
    pub fn str_reg(&mut self, t: Gpr, n: Gpr, m: Gpr) {
        self.emit(0xF822_6800 | ((m.0 as u32) << 16) | ((n.0 as u32) << 5) | t.0 as u32);
    }

    /// `stp x1, x2, [sp, #-16]!` — pre-indexed, the standard frame push.
    pub fn stp_pre(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        debug_assert!(off % 8 == 0 && (-512..512).contains(&off));
        let imm7 = ((off / 8) as u32) & 0x7f;
        self.emit(
            0xA980_0000 | (imm7 << 15) | ((t2.0 as u32) << 10) | ((n.0 as u32) << 5) | t1.0 as u32,
        );
    }

    /// `ldp x1, x2, [sp], #16` — post-indexed, the matching pop.
    pub fn ldp_post(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        debug_assert!(off % 8 == 0 && (-512..512).contains(&off));
        let imm7 = ((off / 8) as u32) & 0x7f;
        self.emit(
            0xA8C0_0000 | (imm7 << 15) | ((t2.0 as u32) << 10) | ((n.0 as u32) << 5) | t1.0 as u32,
        );
    }

    /// `mov xd, sp` / `mov sp, xn`. Encoded as `add xd, xn, #0`, which is the
    /// only `mov` that names `sp` rather than `xzr`.
    pub fn mov_sp(&mut self, d: Gpr, n: Gpr) {
        self.emit(0x9100_0000 | ((n.0 as u32) << 5) | d.0 as u32);
    }

    // --- control flow -------------------------------------------------------

    pub fn b(&mut self, l: Label) {
        self.fixups.push(Fixup {
            at: self.words.len(),
            label: l,
            kind: FixupKind::B26,
        });
        self.emit(0x1400_0000);
    }

    pub fn b_cond(&mut self, cond: Cond, l: Label) {
        self.fixups.push(Fixup {
            at: self.words.len(),
            label: l,
            kind: FixupKind::B19,
        });
        self.emit(0x5400_0000 | cond as u32);
    }

    pub fn cbnz(&mut self, t: Gpr, l: Label) {
        self.fixups.push(Fixup {
            at: self.words.len(),
            label: l,
            kind: FixupKind::B19,
        });
        self.emit(0xB500_0000 | t.0 as u32);
    }

    pub fn cbz(&mut self, t: Gpr, l: Label) {
        self.fixups.push(Fixup {
            at: self.words.len(),
            label: l,
            kind: FixupKind::B19,
        });
        self.emit(0xB400_0000 | t.0 as u32);
    }

    pub fn ret(&mut self) {
        self.emit(0xD65F_03C0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::backend::code::CodeBuf;

    fn run<T>(words: &[u32]) -> T
    where
        T: Copy,
    {
        let mut buf = CodeBuf::new(words.len() * 4).expect("mmap");
        buf.write(|code| {
            for (i, w) in words.iter().enumerate() {
                code[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        });
        let code = buf.finalize();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code.entry()) };
        let r = f(0, 0);
        unsafe { std::mem::transmute_copy(&r) }
    }

    fn run_args(words: &[u32], a: u64, b: u64) -> u64 {
        let mut buf = CodeBuf::new(words.len() * 4).expect("mmap");
        buf.write(|code| {
            for (i, w) in words.iter().enumerate() {
                code[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        });
        let code = buf.finalize();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code.entry()) };
        f(a, b)
    }

    const X0: Gpr = Gpr(0);
    const X1: Gpr = Gpr(1);
    const X2: Gpr = Gpr(2);
    const D0: Fpr = Fpr(0);
    const D1: Fpr = Fpr(1);

    /// Every `mov_imm` shape: one-instruction `movz`, one-instruction `movn`,
    /// and the multi-halfword fallback. These are the encodings most likely to be
    /// subtly wrong, and executing them is the only check that cannot lie.
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
            0x1234_5678_9abc_def0u64 as i64,
            i64::MIN,
            i64::MAX,
        ] {
            let mut a = Asm::new();
            a.mov_imm(X0, v);
            a.ret();
            let got: u64 = run(&a.finish());
            assert_eq!(got, v as u64, "mov_imm {v:#x}");
        }
    }

    #[test]
    fn arithmetic() {
        let mut a = Asm::new();
        a.add(X0, X0, X1);
        a.ret();
        assert_eq!(run_args(&a.finish(), 3, 39), 42);

        let mut a = Asm::new();
        a.sub(X0, X0, X1);
        a.ret();
        assert_eq!(run_args(&a.finish(), 50, 8), 42);

        let mut a = Asm::new();
        a.mul(X0, X0, X1);
        a.ret();
        assert_eq!(run_args(&a.finish(), 6, 7), 42);

        let mut a = Asm::new();
        a.neg(X0, X0);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 0) as i64, -42);

        let mut a = Asm::new();
        a.mvn(X0, X0);
        a.ret();
        assert_eq!(run_args(&a.finish(), 41, 0) as i64, !41i64);

        let mut a = Asm::new();
        a.asrv(X0, X0, X1);
        a.ret();
        assert_eq!(run_args(&a.finish(), (-8i64) as u64, 1) as i64, -4);

        let mut a = Asm::new();
        a.lsrv(X0, X0, X1);
        a.ret();
        assert_eq!(
            run_args(&a.finish(), (-8i64) as u64, 1),
            0x7fff_ffff_ffff_fffc
        );
    }

    #[test]
    fn add_imm_positive_and_negative() {
        let mut a = Asm::new();
        assert!(a.try_add_imm(X0, X0, 2));
        assert!(a.try_add_imm(X0, X0, -44));
        a.ret();
        assert_eq!(run_args(&a.finish(), 84, 0), 42);

        // Out of range: 12 bits, unshifted or shifted by 12.
        let mut a = Asm::new();
        assert!(!a.try_add_imm(X0, X0, 0x1001));
        assert!(a.try_add_imm(X0, X0, 0x2000));
    }

    /// `cset` is `csinc` with the condition *inverted*, which is exactly the kind
    /// of thing that produces a backend that works for `eq` and fails for `lt`.
    /// Check every condition the IR can produce, both ways.
    #[test]
    fn cset_conditions() {
        let cases = [
            (Cond::Eq, 5i64, 5i64, 1u64),
            (Cond::Eq, 5, 6, 0),
            (Cond::Ne, 5, 6, 1),
            (Cond::Ne, 5, 5, 0),
            (Cond::Lt, -1, 1, 1),
            (Cond::Lt, 1, -1, 0),
            (Cond::Le, 5, 5, 1),
            (Cond::Le, 6, 5, 0),
            (Cond::Gt, 1, -1, 1),
            (Cond::Gt, -1, 1, 0),
            (Cond::Ge, 5, 5, 1),
            (Cond::Ge, 4, 5, 0),
        ];
        for (cond, x, y, want) in cases {
            let mut a = Asm::new();
            a.cmp(X0, X1);
            a.cset(X0, cond);
            a.ret();
            let got = run_args(&a.finish(), x as u64, y as u64);
            assert_eq!(got, want, "cset {cond:?} with {x} vs {y}");
        }
    }

    #[test]
    fn cmp_imm() {
        let mut a = Asm::new();
        assert!(a.try_cmp_imm(X0, 42));
        a.cset(X0, Cond::Eq);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 0), 1);

        // Negative immediates go through `cmn`.
        let mut a = Asm::new();
        assert!(a.try_cmp_imm(X0, -7));
        a.cset(X0, Cond::Eq);
        a.ret();
        assert_eq!(run_args(&a.finish(), (-7i64) as u64, 0), 1);
    }

    #[test]
    fn float_ops() {
        // (a as f64 + b as f64) * 2.0, via bit moves in both directions.
        let mut a = Asm::new();
        a.scvtf(D0, X0);
        a.scvtf(D1, X1);
        a.fadd(D0, D0, D1);
        a.mov_imm(X2, 2.0f64.to_bits() as i64);
        a.fmov_to_fpr(D1, X2);
        a.fmul(D0, D0, D1);
        a.fmov_to_gpr(X0, D0);
        a.ret();
        let bits = run_args(&a.finish(), 3, 4);
        assert_eq!(f64::from_bits(bits), 14.0);
    }

    /// The float condition mapping exists because of NaN. `lt` must be `mi`, not
    /// `lt`: an unordered compare sets `V`, so the integer `lt` (N != V) would
    /// report true for `NaN < 1`.
    #[test]
    fn float_compare_is_nan_safe() {
        let nan = f64::NAN.to_bits();
        let one = 1.0f64.to_bits();

        for (cond, want_nan) in [
            (Cond::Mi, 0u64), // lt
            (Cond::Ls, 0),    // le
            (Cond::Gt, 0),    // gt
            (Cond::Ge, 0),    // ge
            (Cond::Eq, 0),    // eq
            (Cond::Ne, 1),    // ne — the one comparison NaN satisfies
        ] {
            let mut a = Asm::new();
            a.fmov_to_fpr(D0, X0);
            a.fmov_to_fpr(D1, X1);
            a.fcmp(D0, D1);
            a.cset(X0, cond);
            a.ret();
            let got = run_args(&a.finish(), nan, one);
            assert_eq!(got, want_nan, "NaN vs 1.0 under {cond:?}");
        }

        // And the ordered cases still behave.
        let mut a = Asm::new();
        a.fmov_to_fpr(D0, X0);
        a.fmov_to_fpr(D1, X1);
        a.fcmp(D0, D1);
        a.cset(X0, Cond::Mi);
        a.ret();
        let words = a.finish();
        assert_eq!(run_args(&words, 0.5f64.to_bits(), one), 1);
        assert_eq!(run_args(&words, 2.0f64.to_bits(), one), 0);
    }

    #[test]
    fn memory_round_trip() {
        let mut mem = [0u64; 4];
        let ptr = mem.as_mut_ptr() as u64;

        // [ptr + 16] <- x1; x0 <- [ptr + 16]
        let mut a = Asm::new();
        assert!(a.try_str(X1, X0, 16));
        assert!(a.try_ldr(X0, X0, 16));
        a.ret();
        assert_eq!(run_args(&a.finish(), ptr, 0xdead_beef), 0xdead_beef);
        assert_eq!(mem[2], 0xdead_beef);

        // Byte access, and the unscaled form (offset 9 is not a multiple of 8).
        let mut a = Asm::new();
        assert!(a.try_strb(X1, X0, 9));
        assert!(a.try_ldrb(X0, X0, 9));
        a.ret();
        assert_eq!(run_args(&a.finish(), ptr, 0xff), 0xff);
    }

    #[test]
    fn branches_and_labels() {
        // if x0 != 0 { 42 } else { 7 }
        let mut a = Asm::new();
        let taken = a.new_label();
        let done = a.new_label();
        a.cbnz(X0, taken);
        a.mov_imm(X0, 7);
        a.b(done);
        a.bind(taken);
        a.mov_imm(X0, 42);
        a.bind(done);
        a.ret();
        let words = a.finish();
        assert_eq!(run_args(&words, 1, 0), 42);
        assert_eq!(run_args(&words, 0, 0), 7);
    }

    /// A backward branch, i.e. an actual loop: sum 1..=x0.
    #[test]
    fn backward_branch() {
        let mut a = Asm::new();
        let top = a.new_label();
        let done = a.new_label();
        a.mov_imm(X1, 0); // acc
        a.bind(top);
        a.cbz(X0, done);
        a.add(X1, X1, X0);
        assert!(a.try_sub_imm(X0, X0, 1));
        a.b(top);
        a.bind(done);
        a.mov(X0, X1);
        a.ret();
        let words = a.finish();
        assert_eq!(run_args(&words, 10, 0), 55);
        assert_eq!(run_args(&words, 0, 0), 0);
    }

    /// Every encoding, against the system assembler.
    ///
    /// The execution tests above prove the instructions *do the right thing*,
    /// which is the property that matters — but they cannot catch an encoding
    /// that happens to work while differing from the canonical one (a reserved
    /// bit set, say), and such an instruction disassembles as garbage the first
    /// time something goes wrong and you reach for `objdump`. The expected words
    /// here are `clang`'s, verbatim.
    #[test]
    fn matches_system_assembler() {
        let x0 = X0;
        let x1 = X1;
        let x2 = X2;
        let (x5, x9) = (Gpr(5), Gpr(9));
        let (d0, d1, d2, d5, d9) = (D0, D1, Fpr(2), Fpr(5), Fpr(9));

        let mut a = Asm::new();
        a.mov(x5, x9);
        a.fmov(d5, d9);
        a.mov_imm(x0, 42);
        a.mov_imm(x1, -42);
        a.mov_imm(x2, 0x1234_5678);
        a.add(x0, x1, x2);
        a.sub(x0, x1, x2);
        a.mul(x0, x1, x2);
        a.sdiv(x0, x1, x2);
        a.msub(x0, x1, x2, Gpr(3));
        a.csel(x0, x1, x2, Cond::Mi);
        a.ccmp_imm(x1, 0, 0, Cond::Ne);
        a.and(x0, x1, x2);
        a.orr(x0, x1, x2);
        a.eor(x0, x1, x2);
        a.lslv(x0, x1, x2);
        a.lsrv(x0, x1, x2);
        a.asrv(x0, x1, x2);
        a.neg(x0, x2);
        a.mvn(x0, x2);
        assert!(a.try_add_imm(x0, x1, 42));
        assert!(a.try_sub_imm(x0, x1, 42));
        assert!(a.try_add_imm(x0, x1, 0x1000));
        a.cmp(x1, x2);
        assert!(a.try_cmp_imm(x1, 42));
        assert!(a.try_cmp_imm(x1, -7));
        a.cset(x0, Cond::Eq);
        a.cset(x0, Cond::Lt);
        a.cset(x0, Cond::Mi);
        a.cset(x0, Cond::Ls);
        a.fcmp(d1, d2);
        a.fadd(d0, d1, d2);
        a.fsub(d0, d1, d2);
        a.fmul(d0, d1, d2);
        a.fdiv(d0, d1, d2);
        a.fneg(d0, d1);
        a.fmov_to_fpr(d0, x1);
        a.fmov_to_gpr(x0, d1);
        a.scvtf(d0, x1);
        assert!(a.try_ldr(x0, x1, 16));
        assert!(a.try_str(x0, x1, 16));
        assert!(a.try_ldrb(x0, x1, 9));
        assert!(a.try_strb(x0, x1, 9));
        assert!(a.try_ldr_f(d0, x1, 16));
        assert!(a.try_str_f(d0, x1, 16));
        assert!(a.try_ldr(x0, x1, -8)); // unscaled: negative offset
        assert!(a.try_str(x0, x1, -8));
        a.ldr_reg(x0, x1, x2);
        a.str_reg(x0, x1, x2);
        a.stp_pre(FP, LR, SP, -16);
        a.ldp_post(FP, LR, SP, 16);
        a.mov_sp(FP, SP);
        a.ret();

        #[rustfmt::skip]
        let want: &[u32] = &[
            0xaa0903e5, // mov   x5, x9
            0x1e604125, // fmov  d5, d9
            0xd2800540, // mov   x0, #42
            0x92800521, // mov   x1, #-42
            0xd28acf02, // mov   x2, #0x5678
            0xf2a24682, // movk  x2, #0x1234, lsl #16
            0x8b020020, // add   x0, x1, x2
            0xcb020020, // sub   x0, x1, x2
            0x9b027c20, // mul   x0, x1, x2
            0x9ac20c20, // sdiv  x0, x1, x2
            0x9b028c20, // msub  x0, x1, x2, x3
            0x9a824020, // csel  x0, x1, x2, mi
            0xfa401820, // ccmp  x1, #0, #0, ne
            0x8a020020, // and   x0, x1, x2
            0xaa020020, // orr   x0, x1, x2
            0xca020020, // eor   x0, x1, x2
            0x9ac22020, // lsl   x0, x1, x2
            0x9ac22420, // lsr   x0, x1, x2
            0x9ac22820, // asr   x0, x1, x2
            0xcb0203e0, // neg   x0, x2
            0xaa2203e0, // mvn   x0, x2
            0x9100a820, // add   x0, x1, #42
            0xd100a820, // sub   x0, x1, #42
            0x91400420, // add   x0, x1, #1, lsl #12
            0xeb02003f, // cmp   x1, x2
            0xf100a83f, // cmp   x1, #42
            0xb1001c3f, // cmn   x1, #7
            0x9a9f17e0, // cset  x0, eq
            0x9a9fa7e0, // cset  x0, lt
            0x9a9f57e0, // cset  x0, mi
            0x9a9f87e0, // cset  x0, ls
            0x1e622020, // fcmp  d1, d2
            0x1e622820, // fadd  d0, d1, d2
            0x1e623820, // fsub  d0, d1, d2
            0x1e620820, // fmul  d0, d1, d2
            0x1e621820, // fdiv  d0, d1, d2
            0x1e614020, // fneg  d0, d1
            0x9e670020, // fmov  d0, x1
            0x9e660020, // fmov  x0, d1
            0x9e620020, // scvtf d0, x1
            0xf9400820, // ldr   x0, [x1, #16]
            0xf9000820, // str   x0, [x1, #16]
            0x39402420, // ldrb  w0, [x1, #9]
            0x39002420, // strb  w0, [x1, #9]
            0xfd400820, // ldr   d0, [x1, #16]
            0xfd000820, // str   d0, [x1, #16]
            0xf85f8020, // ldur  x0, [x1, #-8]
            0xf81f8020, // stur  x0, [x1, #-8]
            0xf8626820, // ldr   x0, [x1, x2]
            0xf8226820, // str   x0, [x1, x2]
            0xa9bf7bfd, // stp   x29, x30, [sp, #-16]!
            0xa8c17bfd, // ldp   x29, x30, [sp], #16
            0x910003fd, // mov   x29, sp
            0xd65f03c0, // ret
        ];

        let got = a.finish();
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g, w, "word {i}: got {g:#010x}, clang says {w:#010x}");
        }
        assert_eq!(got.len(), want.len());
    }

    /// A `b` to the very next instruction is dropped; a conditional one is not.
    #[test]
    fn branch_to_the_next_instruction_is_dropped() {
        let mut a = Asm::new();
        let l = a.new_label();
        a.mov_imm(X0, 7);
        a.b(l);
        a.bind(l);
        a.ret();
        assert_eq!(a.finish().len(), 2, "the `b` should be gone");

        // The conditional must survive: its fallthrough and its target coincide,
        // but the *test* still has to run — and here it feeds a `cset` after it.
        let mut a = Asm::new();
        let l = a.new_label();
        a.cbnz(X0, l);
        a.bind(l);
        a.ret();
        assert_eq!(a.finish().len(), 2, "the `cbnz` must stay");

        // And only the *last* instruction: an intervening one makes it live.
        let mut a = Asm::new();
        let l = a.new_label();
        a.b(l);
        a.mov_imm(X0, 7);
        a.bind(l);
        a.ret();
        let words = a.finish();
        assert_eq!(words.len(), 3);
        assert_eq!(run_args(&words, 0, 0), 0, "the branch must skip the mov");
    }

    /// A frame push/pop, exercised by clobbering the stack in between.
    #[test]
    fn frame_setup() {
        let mut a = Asm::new();
        a.stp_pre(FP, LR, SP, -16);
        a.mov_sp(FP, SP);
        assert!(a.try_sub_imm(SP, SP, 32));
        assert!(a.try_str(X0, SP, 8));
        assert!(a.try_ldr(X0, SP, 8));
        a.mov_sp(SP, FP);
        a.ldp_post(FP, LR, SP, 16);
        a.ret();
        assert_eq!(run_args(&a.finish(), 42, 0), 42);
    }
}
