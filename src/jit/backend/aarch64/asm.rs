//! An aarch64 assembler: encodings, labels, a literal pool, and branches to
//! fixed addresses outside the buffer, resolved once the code's address is
//! known. Encodings are checked against the system assembler in the tests.

// The encoder also covers instructions later milestones emit.
#![allow(dead_code)]

/// A general-purpose register, `x0`-`x30`; 31 is `xzr` or `sp` by position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) struct Gpr(pub(crate) u8);

/// A floating-point register, `d0`-`d31`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) struct Fpr(pub(crate) u8);

pub(crate) const ZR: Gpr = Gpr(31);
pub(crate) const SP: Gpr = Gpr(31);
pub(crate) const FP: Gpr = Gpr(29);
pub(crate) const LR: Gpr = Gpr(30);

/// Operand width of an integer instruction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Sz {
    W,
    X,
}

impl Sz {
    #[inline]
    fn sf(self) -> u32 {
        match self {
            Sz::W => 0,
            Sz::X => 1 << 31,
        }
    }

    pub(crate) fn bits(self) -> u32 {
        match self {
            Sz::W => 32,
            Sz::X => 64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Cond {
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
    pub(crate) fn invert(self) -> Cond {
        use Cond::*;
        match self {
            Eq => Ne,
            Ne => Eq,
            Hs => Lo,
            Lo => Hs,
            Mi => Pl,
            Pl => Mi,
            Vs => Vc,
            Vc => Vs,
            Hi => Ls,
            Ls => Hi,
            Ge => Lt,
            Lt => Ge,
            Gt => Le,
            Le => Gt,
            Al => panic!("`al` has no inverse"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        use Cond::*;
        match self {
            Eq => "eq",
            Ne => "ne",
            Hs => "hs",
            Lo => "lo",
            Mi => "mi",
            Pl => "pl",
            Vs => "vs",
            Vc => "vc",
            Hi => "hi",
            Ls => "ls",
            Ge => "ge",
            Lt => "lt",
            Gt => "gt",
            Le => "le",
            Al => "al",
        }
    }
}

/// Shift of a shifted-register operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
}

/// Extend of an extended-register operand or a register index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Extend {
    Uxtw = 2,
    Lsl = 3,
    Sxtw = 6,
}

/// A position in the buffer, bound once.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Label(pub(crate) u32);

#[derive(Clone, Copy, Debug)]
enum FixupKind {
    /// `b`, `bl`: 26-bit word offset.
    B26,
    /// `b.cond`, `cbz`, `cbnz`, `ldr` literal: 19-bit word offset.
    B19,
    /// `tbz`, `tbnz`: 14-bit word offset.
    B14,
    /// `adr`: 21-bit byte offset.
    Adr21,
}

#[derive(Clone, Copy, Debug)]
struct Fixup {
    at: u32,
    label: Label,
    kind: FixupKind,
}

/// The encoding failed: a branch out of range, or an immediate that does not
/// fit where the caller asserted it would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AsmError {
    BranchRange,
    ExternRange,
}

#[derive(Default)]
pub(crate) struct Asm {
    words: Vec<u32>,
    labels: Vec<u32>,
    fixups: Vec<Fixup>,
    /// `b`/`bl` to a fixed address: (word, target).
    externs: Vec<(u32, usize)>,
    /// 64-bit literals by label, emitted after the code.
    literals: Vec<(Label, u64)>,
    lit_index: std::collections::HashMap<u64, Label>,
}

const UNBOUND: u32 = u32::MAX;

#[inline]
fn r(g: Gpr) -> u32 {
    g.0 as u32
}

#[inline]
fn f(g: Fpr) -> u32 {
    g.0 as u32
}

impl Asm {
    pub(crate) fn new() -> Self {
        Asm::default()
    }

    /// Bytes emitted so far.
    pub(crate) fn offset(&self) -> usize {
        self.words.len() * 4
    }

    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    pub(crate) fn new_label(&mut self) -> Label {
        let l = Label(self.labels.len() as u32);
        self.labels.push(UNBOUND);
        l
    }

    pub(crate) fn bind(&mut self, l: Label) {
        debug_assert_eq!(self.labels[l.0 as usize], UNBOUND, "label bound twice");
        self.labels[l.0 as usize] = self.words.len() as u32;
    }

    pub(crate) fn is_bound(&self, l: Label) -> bool {
        self.labels[l.0 as usize] != UNBOUND
    }

    /// The bound label's byte offset.
    pub(crate) fn label_offset(&self, l: Label) -> usize {
        let w = self.labels[l.0 as usize];
        debug_assert_ne!(w, UNBOUND);
        w as usize * 4
    }

    /// Pad with `brk` to a multiple of `align` bytes.
    pub(crate) fn align(&mut self, align: usize) {
        while !self.offset().is_multiple_of(align) {
            self.brk(0);
        }
    }

    #[inline]
    pub(crate) fn emit(&mut self, w: u32) {
        self.words.push(w);
    }

    fn fixup(&mut self, label: Label, kind: FixupKind) {
        self.fixups.push(Fixup {
            at: self.words.len() as u32,
            label,
            kind,
        });
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

    /// Emit the literal pool, resolve every branch for code placed at
    /// `base`, and hand back the words.
    pub(crate) fn finish(mut self, base: usize) -> Result<Vec<u32>, AsmError> {
        if !self.literals.is_empty() {
            self.align(8);
            for (l, v) in std::mem::take(&mut self.literals) {
                self.bind(l);
                self.emit(v as u32);
                self.emit((v >> 32) as u32);
            }
        }
        for fx in &self.fixups {
            let target = self.labels[fx.label.0 as usize];
            assert_ne!(target, UNBOUND, "branch to an unbound label");
            let delta = target as i64 - fx.at as i64;
            let w = &mut self.words[fx.at as usize];
            match fx.kind {
                FixupKind::B26 => {
                    if !(-(1 << 25)..(1 << 25)).contains(&delta) {
                        return Err(AsmError::BranchRange);
                    }
                    *w |= (delta as u32) & 0x03ff_ffff;
                }
                FixupKind::B19 => {
                    if !(-(1 << 18)..(1 << 18)).contains(&delta) {
                        return Err(AsmError::BranchRange);
                    }
                    *w |= ((delta as u32) & 0x7ffff) << 5;
                }
                FixupKind::B14 => {
                    if !(-(1 << 13)..(1 << 13)).contains(&delta) {
                        return Err(AsmError::BranchRange);
                    }
                    *w |= ((delta as u32) & 0x3fff) << 5;
                }
                FixupKind::Adr21 => {
                    let bytes = delta * 4;
                    if !(-(1 << 20)..(1 << 20)).contains(&bytes) {
                        return Err(AsmError::BranchRange);
                    }
                    let b = bytes as u32;
                    *w |= (b & 3) << 29 | ((b >> 2) & 0x7ffff) << 5;
                }
            }
        }
        for &(at, target) in &self.externs {
            let here = base as i64 + at as i64 * 4;
            let delta = (target as i64 - here) / 4;
            if !(-(1 << 25)..(1 << 25)).contains(&delta) || (target as i64 - here) % 4 != 0 {
                return Err(AsmError::ExternRange);
            }
            self.words[at as usize] |= (delta as u32) & 0x03ff_ffff;
        }
        Ok(self.words)
    }

    // --- moves and immediates ---------------------------------------------

    /// `mov xd, xn` (`orr xd, xzr, xn`); nothing when equal.
    pub(crate) fn mov(&mut self, d: Gpr, n: Gpr) {
        if d != n {
            self.emit(0xAA00_03E0 | r(n) << 16 | r(d));
        }
    }

    /// `mov wd, wn`: zero-extends.
    pub(crate) fn mov_w(&mut self, d: Gpr, n: Gpr) {
        self.emit(0x2A00_03E0 | r(n) << 16 | r(d));
    }

    /// `mov xd, sp` / `mov sp, xn` (`add #0`).
    pub(crate) fn mov_sp(&mut self, d: Gpr, n: Gpr) {
        self.emit(0x9100_0000 | r(n) << 5 | r(d));
    }

    pub(crate) fn movz(&mut self, sz: Sz, d: Gpr, imm16: u16, hw: u32) {
        self.emit(sz.sf() | 0x5280_0000 | hw << 21 | (imm16 as u32) << 5 | r(d));
    }

    pub(crate) fn movn(&mut self, sz: Sz, d: Gpr, imm16: u16, hw: u32) {
        self.emit(sz.sf() | 0x1280_0000 | hw << 21 | (imm16 as u32) << 5 | r(d));
    }

    pub(crate) fn movk(&mut self, sz: Sz, d: Gpr, imm16: u16, hw: u32) {
        self.emit(sz.sf() | 0x7280_0000 | hw << 21 | (imm16 as u32) << 5 | r(d));
    }

    /// `xd = imm` in the fewest of `movz`/`movn`/`movk` or one `orr` of a
    /// logical immediate.
    pub(crate) fn mov_imm(&mut self, d: Gpr, imm: u64) {
        let halves = |v: u64| (0..4).filter(|&h| (v >> (h * 16)) & 0xffff != 0).count();
        let zeros = 4 - halves(imm);
        let ones = 4 - halves(!imm);
        if imm == 0 {
            self.movz(Sz::X, d, 0, 0);
            return;
        }
        if zeros < 3
            && ones < 3
            && let Some(enc) = logical_imm(imm, 64)
        {
            self.emit(0xB200_03E0 | enc << 10 | r(d));
            return;
        }
        if ones > zeros {
            let mut first = true;
            for hw in 0..4u32 {
                let half = ((imm >> (hw * 16)) & 0xffff) as u16;
                if half == 0xffff {
                    continue;
                }
                if first {
                    self.movn(Sz::X, d, !half, hw);
                    first = false;
                } else {
                    self.movk(Sz::X, d, half, hw);
                }
            }
            if first {
                self.movn(Sz::X, d, 0, 0);
            }
        } else {
            let mut first = true;
            for hw in 0..4u32 {
                let half = ((imm >> (hw * 16)) & 0xffff) as u16;
                if half == 0 {
                    continue;
                }
                if first {
                    self.movz(Sz::X, d, half, hw);
                    first = false;
                } else {
                    self.movk(Sz::X, d, half, hw);
                }
            }
        }
    }

    /// `wd = imm` (32-bit).
    pub(crate) fn mov_imm_w(&mut self, d: Gpr, imm: u32) {
        let lo = imm as u16;
        let hi = (imm >> 16) as u16;
        if hi == 0 {
            self.movz(Sz::W, d, lo, 0);
        } else if lo == 0 {
            self.movz(Sz::W, d, hi, 1);
        } else if hi == 0xffff {
            self.movn(Sz::W, d, !lo, 0);
        } else if lo == 0xffff {
            self.movn(Sz::W, d, !hi, 1);
        } else if let Some(enc) = logical_imm(imm as u64, 32) {
            self.emit(0x3200_03E0 | enc << 10 | r(d));
        } else {
            self.movz(Sz::W, d, lo, 0);
            self.movk(Sz::W, d, hi, 1);
        }
    }

    /// `ldr xd, =v` from the literal pool.
    pub(crate) fn ldr_lit(&mut self, d: Gpr, v: u64) {
        let l = self.literal(v);
        self.fixup(l, FixupKind::B19);
        self.emit(0x5800_0000 | r(d));
    }

    /// `ldr xd, label`: an 8-byte word the caller binds and emits.
    pub(crate) fn ldr_label(&mut self, d: Gpr, l: Label) {
        self.fixup(l, FixupKind::B19);
        self.emit(0x5800_0000 | r(d));
    }

    /// `ldr dd, =v` from the literal pool.
    pub(crate) fn ldr_lit_f(&mut self, d: Fpr, v: u64) {
        let l = self.literal(v);
        self.fixup(l, FixupKind::B19);
        self.emit(0x5C00_0000 | f(d));
    }

    /// `adr xd, label`.
    pub(crate) fn adr(&mut self, d: Gpr, l: Label) {
        self.fixup(l, FixupKind::Adr21);
        self.emit(0x1000_0000 | r(d));
    }

    // --- integer arithmetic ---------------------------------------------

    fn rrr(&mut self, base: u32, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(sz.sf() | base | r(m) << 16 | r(n) << 5 | r(d));
    }

    fn rrr_shift(&mut self, base: u32, sz: Sz, d: Gpr, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        debug_assert!(amt < sz.bits());
        self.emit(sz.sf() | base | (sh as u32) << 22 | r(m) << 16 | amt << 10 | r(n) << 5 | r(d));
    }

    pub(crate) fn add(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x0B00_0000, sz, d, n, m);
    }

    pub(crate) fn adds(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x2B00_0000, sz, d, n, m);
    }

    pub(crate) fn sub(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x4B00_0000, sz, d, n, m);
    }

    pub(crate) fn subs(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x6B00_0000, sz, d, n, m);
    }

    pub(crate) fn add_shift(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        self.rrr_shift(0x0B00_0000, sz, d, n, m, sh, amt);
    }

    pub(crate) fn sub_shift(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        self.rrr_shift(0x4B00_0000, sz, d, n, m, sh, amt);
    }

    pub(crate) fn cmp(&mut self, sz: Sz, n: Gpr, m: Gpr) {
        self.subs(sz, ZR, n, m);
    }

    pub(crate) fn cmn(&mut self, sz: Sz, n: Gpr, m: Gpr) {
        self.adds(sz, ZR, n, m);
    }

    pub(crate) fn cmp_shift(&mut self, sz: Sz, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        self.rrr_shift(0x6B00_0000, sz, ZR, n, m, sh, amt);
    }

    /// `cmp xn, wm, <ext>` (64-bit compare against an extended register).
    pub(crate) fn cmp_ext(&mut self, n: Gpr, m: Gpr, ext: Extend) {
        self.emit(0xEB20_0000 | r(m) << 16 | (ext as u32) << 13 | r(n) << 5 | 31);
    }

    /// `add xd, xn, wm, <ext> #amt`.
    pub(crate) fn add_ext(&mut self, d: Gpr, n: Gpr, m: Gpr, ext: Extend, amt: u32) {
        debug_assert!(amt <= 4);
        self.emit(0x8B20_0000 | r(m) << 16 | (ext as u32) << 13 | amt << 10 | r(n) << 5 | r(d));
    }

    /// The `imm12` field of an add/sub immediate, optionally `lsl #12`.
    pub(crate) fn addsub_imm(imm: u64) -> Option<u32> {
        if imm < 4096 {
            Some(imm as u32)
        } else if imm & 0xfff == 0 && imm < (4096 << 12) {
            Some(1 << 12 | (imm >> 12) as u32)
        } else {
            None
        }
    }

    fn rri(&mut self, base: u32, sz: Sz, d: Gpr, n: Gpr, enc: u32) {
        self.emit(sz.sf() | base | enc << 10 | r(n) << 5 | r(d));
    }

    /// `add d, n, #imm`; `imm` must fit (`addsub_imm`).
    pub(crate) fn add_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        let enc = Self::addsub_imm(imm).expect("add immediate");
        self.rri(0x1100_0000, sz, d, n, enc);
    }

    pub(crate) fn adds_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        let enc = Self::addsub_imm(imm).expect("adds immediate");
        self.rri(0x3100_0000, sz, d, n, enc);
    }

    pub(crate) fn sub_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        let enc = Self::addsub_imm(imm).expect("sub immediate");
        self.rri(0x5100_0000, sz, d, n, enc);
    }

    pub(crate) fn subs_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        let enc = Self::addsub_imm(imm).expect("subs immediate");
        self.rri(0x7100_0000, sz, d, n, enc);
    }

    pub(crate) fn cmp_imm(&mut self, sz: Sz, n: Gpr, imm: u64) {
        self.subs_imm(sz, ZR, n, imm);
    }

    pub(crate) fn cmn_imm(&mut self, sz: Sz, n: Gpr, imm: u64) {
        self.adds_imm(sz, ZR, n, imm);
    }

    pub(crate) fn and(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x0A00_0000, sz, d, n, m);
    }

    pub(crate) fn orr(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x2A00_0000, sz, d, n, m);
    }

    pub(crate) fn eor(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x4A00_0000, sz, d, n, m);
    }

    pub(crate) fn ands(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x6A00_0000, sz, d, n, m);
    }

    pub(crate) fn tst(&mut self, sz: Sz, n: Gpr, m: Gpr) {
        self.ands(sz, ZR, n, m);
    }

    pub(crate) fn orr_shift(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        self.rrr_shift(0x2A00_0000, sz, d, n, m, sh, amt);
    }

    pub(crate) fn eor_shift(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, sh: Shift, amt: u32) {
        self.rrr_shift(0x4A00_0000, sz, d, n, m, sh, amt);
    }

    /// `mvn d, m` (`orn d, zr, m`).
    pub(crate) fn mvn(&mut self, sz: Sz, d: Gpr, m: Gpr) {
        self.emit(sz.sf() | 0x2A20_03E0 | r(m) << 16 | r(d));
    }

    /// `neg d, m` (`sub d, zr, m`).
    pub(crate) fn neg(&mut self, sz: Sz, d: Gpr, m: Gpr) {
        self.sub(sz, d, ZR, m);
    }

    /// `negs d, m`: sets V on `MIN`.
    pub(crate) fn negs(&mut self, sz: Sz, d: Gpr, m: Gpr) {
        self.subs(sz, d, ZR, m);
    }

    fn logical_i(&mut self, base: u32, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        let enc = logical_imm(imm, sz.bits()).expect("logical immediate");
        self.emit(sz.sf() | base | enc << 10 | r(n) << 5 | r(d));
    }

    pub(crate) fn and_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        self.logical_i(0x1200_0000, sz, d, n, imm);
    }

    pub(crate) fn orr_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        self.logical_i(0x3200_0000, sz, d, n, imm);
    }

    pub(crate) fn eor_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        self.logical_i(0x5200_0000, sz, d, n, imm);
    }

    pub(crate) fn ands_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, imm: u64) {
        self.logical_i(0x7200_0000, sz, d, n, imm);
    }

    pub(crate) fn tst_imm(&mut self, sz: Sz, n: Gpr, imm: u64) {
        self.ands_imm(sz, ZR, n, imm);
    }

    pub(crate) fn lslv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x1AC0_2000, sz, d, n, m);
    }

    pub(crate) fn lsrv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x1AC0_2400, sz, d, n, m);
    }

    pub(crate) fn asrv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x1AC0_2800, sz, d, n, m);
    }

    pub(crate) fn sdiv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x1AC0_0C00, sz, d, n, m);
    }

    pub(crate) fn udiv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.rrr(0x1AC0_0800, sz, d, n, m);
    }

    /// `madd d, n, m, a`: `a + n * m`.
    pub(crate) fn madd(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, a: Gpr) {
        self.emit(sz.sf() | 0x1B00_0000 | r(m) << 16 | r(a) << 10 | r(n) << 5 | r(d));
    }

    /// `msub d, n, m, a`: `a - n * m`.
    pub(crate) fn msub(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, a: Gpr) {
        self.emit(sz.sf() | 0x1B00_8000 | r(m) << 16 | r(a) << 10 | r(n) << 5 | r(d));
    }

    pub(crate) fn mul(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr) {
        self.madd(sz, d, n, m, ZR);
    }

    /// `smull xd, wn, wm`: the full 64-bit product of two `w` registers.
    pub(crate) fn smull(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(0x9B20_7C00 | r(m) << 16 | r(n) << 5 | r(d));
    }

    /// `smulh xd, xn, xm`: the high 64 bits of the 128-bit product.
    pub(crate) fn smulh(&mut self, d: Gpr, n: Gpr, m: Gpr) {
        self.emit(0x9B40_7C00 | r(m) << 16 | r(n) << 5 | r(d));
    }

    pub(crate) fn csel(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, c: Cond) {
        self.emit(sz.sf() | 0x1A80_0000 | r(m) << 16 | (c as u32) << 12 | r(n) << 5 | r(d));
    }

    pub(crate) fn csinc(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, c: Cond) {
        self.emit(sz.sf() | 0x1A80_0400 | r(m) << 16 | (c as u32) << 12 | r(n) << 5 | r(d));
    }

    pub(crate) fn csinv(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, c: Cond) {
        self.emit(sz.sf() | 0x5A80_0000 | r(m) << 16 | (c as u32) << 12 | r(n) << 5 | r(d));
    }

    pub(crate) fn csneg(&mut self, sz: Sz, d: Gpr, n: Gpr, m: Gpr, c: Cond) {
        self.emit(sz.sf() | 0x5A80_0400 | r(m) << 16 | (c as u32) << 12 | r(n) << 5 | r(d));
    }

    /// `cset d, c`.
    pub(crate) fn cset(&mut self, sz: Sz, d: Gpr, c: Cond) {
        self.csinc(sz, d, ZR, ZR, c.invert());
    }

    /// `ccmp n, m, #nzcv, c`.
    pub(crate) fn ccmp(&mut self, sz: Sz, n: Gpr, m: Gpr, nzcv: u8, c: Cond) {
        debug_assert!(nzcv < 16);
        self.emit(sz.sf() | 0x7A40_0000 | r(m) << 16 | (c as u32) << 12 | r(n) << 5 | nzcv as u32);
    }

    /// `ccmp n, #imm5, #nzcv, c`.
    pub(crate) fn ccmp_imm(&mut self, sz: Sz, n: Gpr, imm: u8, nzcv: u8, c: Cond) {
        debug_assert!(imm < 32 && nzcv < 16);
        self.emit(
            sz.sf() | 0x7A40_0800 | (imm as u32) << 16 | (c as u32) << 12 | r(n) << 5 | nzcv as u32,
        );
    }

    /// `ccmn n, #imm5, #nzcv, c`.
    pub(crate) fn ccmn_imm(&mut self, sz: Sz, n: Gpr, imm: u8, nzcv: u8, c: Cond) {
        debug_assert!(imm < 32 && nzcv < 16);
        self.emit(
            sz.sf() | 0x3A40_0800 | (imm as u32) << 16 | (c as u32) << 12 | r(n) << 5 | nzcv as u32,
        );
    }

    fn bfm(&mut self, base: u32, sz: Sz, d: Gpr, n: Gpr, immr: u32, imms: u32) {
        let nbit = match sz {
            Sz::W => 0,
            Sz::X => 1 << 22,
        };
        self.emit(sz.sf() | base | nbit | immr << 16 | imms << 10 | r(n) << 5 | r(d));
    }

    pub(crate) fn ubfm(&mut self, sz: Sz, d: Gpr, n: Gpr, immr: u32, imms: u32) {
        self.bfm(0x5300_0000, sz, d, n, immr, imms);
    }

    pub(crate) fn sbfm(&mut self, sz: Sz, d: Gpr, n: Gpr, immr: u32, imms: u32) {
        self.bfm(0x1300_0000, sz, d, n, immr, imms);
    }

    pub(crate) fn lsl_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, s: u32) {
        let b = sz.bits();
        debug_assert!(s < b);
        self.ubfm(sz, d, n, (b - s) % b, b - 1 - s);
    }

    pub(crate) fn lsr_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, s: u32) {
        debug_assert!(s < sz.bits());
        self.ubfm(sz, d, n, s, sz.bits() - 1);
    }

    pub(crate) fn asr_imm(&mut self, sz: Sz, d: Gpr, n: Gpr, s: u32) {
        debug_assert!(s < sz.bits());
        self.sbfm(sz, d, n, s, sz.bits() - 1);
    }

    pub(crate) fn ubfx(&mut self, sz: Sz, d: Gpr, n: Gpr, lsb: u32, width: u32) {
        self.ubfm(sz, d, n, lsb, lsb + width - 1);
    }

    pub(crate) fn sbfx(&mut self, sz: Sz, d: Gpr, n: Gpr, lsb: u32, width: u32) {
        self.sbfm(sz, d, n, lsb, lsb + width - 1);
    }

    /// `sxtw xd, wn`.
    pub(crate) fn sxtw(&mut self, d: Gpr, n: Gpr) {
        self.sbfm(Sz::X, d, n, 0, 31);
    }

    // --- memory -----------------------------------------------------------

    /// A load or store at `[n, #off]`: the scaled unsigned form, else the
    /// unscaled signed one. `None` when neither reaches.
    fn mem(scaled: u32, unscaled: u32, size: i32, t: u32, n: Gpr, off: i32) -> Option<u32> {
        if off >= 0 && off % size == 0 && off / size < 4096 {
            Some(scaled | ((off / size) as u32) << 10 | r(n) << 5 | t)
        } else if (-256..256).contains(&off) {
            Some(unscaled | ((off as u32) & 0x1ff) << 12 | r(n) << 5 | t)
        } else {
            None
        }
    }

    pub(crate) fn mem_fits(size: i32, off: i32) -> bool {
        (off >= 0 && off % size == 0 && off / size < 4096) || (-256..256).contains(&off)
    }

    fn mem_op(&mut self, scaled: u32, unscaled: u32, size: i32, t: u32, n: Gpr, off: i32) {
        let w = Self::mem(scaled, unscaled, size, t, n, off).expect("memory offset out of range");
        self.emit(w);
    }

    pub(crate) fn ldr(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0xF940_0000, 0xF840_0000, 8, r(t), n, off);
    }

    pub(crate) fn str(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0xF900_0000, 0xF800_0000, 8, r(t), n, off);
    }

    pub(crate) fn ldr_w(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0xB940_0000, 0xB840_0000, 4, r(t), n, off);
    }

    pub(crate) fn str_w(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0xB900_0000, 0xB800_0000, 4, r(t), n, off);
    }

    pub(crate) fn ldrsw(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0xB980_0000, 0xB880_0000, 4, r(t), n, off);
    }

    pub(crate) fn ldrh(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0x7940_0000, 0x7840_0000, 2, r(t), n, off);
    }

    pub(crate) fn strh(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0x7900_0000, 0x7800_0000, 2, r(t), n, off);
    }

    pub(crate) fn ldrb(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0x3940_0000, 0x3840_0000, 1, r(t), n, off);
    }

    pub(crate) fn strb(&mut self, t: Gpr, n: Gpr, off: i32) {
        self.mem_op(0x3900_0000, 0x3800_0000, 1, r(t), n, off);
    }

    pub(crate) fn ldr_d(&mut self, t: Fpr, n: Gpr, off: i32) {
        self.mem_op(0xFD40_0000, 0xFC40_0000, 8, f(t), n, off);
    }

    pub(crate) fn str_d(&mut self, t: Fpr, n: Gpr, off: i32) {
        self.mem_op(0xFD00_0000, 0xFC00_0000, 8, f(t), n, off);
    }

    /// `ldr xt, [n, m, <ext> #(3 if shift)]`.
    pub(crate) fn ldr_idx(&mut self, t: Gpr, n: Gpr, m: Gpr, ext: Extend, shift: bool) {
        self.emit(
            0xF860_0800 | r(m) << 16 | (ext as u32) << 13 | (shift as u32) << 12 | r(n) << 5 | r(t),
        );
    }

    pub(crate) fn str_idx(&mut self, t: Gpr, n: Gpr, m: Gpr, ext: Extend, shift: bool) {
        self.emit(
            0xF820_0800 | r(m) << 16 | (ext as u32) << 13 | (shift as u32) << 12 | r(n) << 5 | r(t),
        );
    }

    pub(crate) fn ldr_d_idx(&mut self, t: Fpr, n: Gpr, m: Gpr, ext: Extend, shift: bool) {
        self.emit(
            0xFC60_0800 | r(m) << 16 | (ext as u32) << 13 | (shift as u32) << 12 | r(n) << 5 | f(t),
        );
    }

    pub(crate) fn str_d_idx(&mut self, t: Fpr, n: Gpr, m: Gpr, ext: Extend, shift: bool) {
        self.emit(
            0xFC20_0800 | r(m) << 16 | (ext as u32) << 13 | (shift as u32) << 12 | r(n) << 5 | f(t),
        );
    }

    fn pair(&mut self, base: u32, t1: u32, t2: u32, n: Gpr, off: i32) {
        debug_assert!(off % 8 == 0 && (-512..512).contains(&off));
        let imm7 = ((off / 8) as u32) & 0x7f;
        self.emit(base | imm7 << 15 | t2 << 10 | r(n) << 5 | t1);
    }

    pub(crate) fn stp(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        self.pair(0xA900_0000, r(t1), r(t2), n, off);
    }

    pub(crate) fn ldp(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        self.pair(0xA940_0000, r(t1), r(t2), n, off);
    }

    /// `stp t1, t2, [n, #off]!`.
    pub(crate) fn stp_pre(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        self.pair(0xA980_0000, r(t1), r(t2), n, off);
    }

    /// `ldp t1, t2, [n], #off`.
    pub(crate) fn ldp_post(&mut self, t1: Gpr, t2: Gpr, n: Gpr, off: i32) {
        self.pair(0xA8C0_0000, r(t1), r(t2), n, off);
    }

    pub(crate) fn stp_d(&mut self, t1: Fpr, t2: Fpr, n: Gpr, off: i32) {
        self.pair(0x6D00_0000, f(t1), f(t2), n, off);
    }

    pub(crate) fn ldp_d(&mut self, t1: Fpr, t2: Fpr, n: Gpr, off: i32) {
        self.pair(0x6D40_0000, f(t1), f(t2), n, off);
    }

    // --- floating point -----------------------------------------------------

    fn fff(&mut self, base: u32, d: Fpr, n: Fpr, m: Fpr) {
        self.emit(base | f(m) << 16 | f(n) << 5 | f(d));
    }

    fn ff(&mut self, base: u32, d: Fpr, n: Fpr) {
        self.emit(base | f(n) << 5 | f(d));
    }

    pub(crate) fn fadd(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_2800, d, n, m);
    }

    pub(crate) fn fsub(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_3800, d, n, m);
    }

    pub(crate) fn fmul(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_0800, d, n, m);
    }

    pub(crate) fn fdiv(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_1800, d, n, m);
    }

    pub(crate) fn fmax(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_4800, d, n, m);
    }

    pub(crate) fn fmin(&mut self, d: Fpr, n: Fpr, m: Fpr) {
        self.fff(0x1E60_5800, d, n, m);
    }

    pub(crate) fn fmov(&mut self, d: Fpr, n: Fpr) {
        if d != n {
            self.ff(0x1E60_4000, d, n);
        }
    }

    pub(crate) fn fneg(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E61_4000, d, n);
    }

    pub(crate) fn fabs(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E60_C000, d, n);
    }

    pub(crate) fn fsqrt(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E61_C000, d, n);
    }

    /// Round toward minus infinity.
    pub(crate) fn frintm(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E65_4000, d, n);
    }

    /// Round toward plus infinity.
    pub(crate) fn frintp(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E64_C000, d, n);
    }

    /// Round toward zero.
    pub(crate) fn frintz(&mut self, d: Fpr, n: Fpr) {
        self.ff(0x1E65_C000, d, n);
    }

    pub(crate) fn fcmp(&mut self, n: Fpr, m: Fpr) {
        self.emit(0x1E60_2000 | f(m) << 16 | f(n) << 5);
    }

    pub(crate) fn fcmp_zero(&mut self, n: Fpr) {
        self.emit(0x1E60_2008 | f(n) << 5);
    }

    pub(crate) fn fcsel(&mut self, d: Fpr, n: Fpr, m: Fpr, c: Cond) {
        self.emit(0x1E60_0C00 | f(m) << 16 | (c as u32) << 12 | f(n) << 5 | f(d));
    }

    /// `fmov dd, xn`: the bits, no conversion.
    pub(crate) fn fmov_from_gpr(&mut self, d: Fpr, n: Gpr) {
        self.emit(0x9E67_0000 | r(n) << 5 | f(d));
    }

    /// `fmov xd, dn`.
    pub(crate) fn fmov_to_gpr(&mut self, d: Gpr, n: Fpr) {
        self.emit(0x9E66_0000 | f(n) << 5 | r(d));
    }

    /// `scvtf dd, (w|x)n`.
    pub(crate) fn scvtf(&mut self, sz: Sz, d: Fpr, n: Gpr) {
        self.emit(sz.sf() | 0x1E62_0000 | r(n) << 5 | f(d));
    }

    /// `fcvtzs (w|x)d, dn`.
    pub(crate) fn fcvtzs(&mut self, sz: Sz, d: Gpr, n: Fpr) {
        self.emit(sz.sf() | 0x1E78_0000 | f(n) << 5 | r(d));
    }

    /// The 8-bit `fmov` immediate of `v`, when it has one.
    pub(crate) fn fmov_imm8(v: f64) -> Option<u32> {
        let bits = v.to_bits();
        if bits & 0x0000_ffff_ffff_ffff != 0 {
            return None;
        }
        let sign = (bits >> 63) as u32;
        let be = ((bits >> 52) & 0x7ff) as u32;
        let frac = ((bits >> 48) & 0xf) as u32;
        // The exponent is NOT(b):b x8:cd (VFPExpandImm).
        let (b, cd) = match be {
            0x3fc..=0x3ff => (1, be - 0x3fc),
            0x400..=0x403 => (0, be - 0x400),
            _ => return None,
        };
        Some(sign << 7 | b << 6 | cd << 4 | frac)
    }

    pub(crate) fn fmov_imm(&mut self, d: Fpr, imm8: u32) {
        self.emit(0x1E60_1000 | imm8 << 13 | f(d));
    }

    // --- control flow -------------------------------------------------------

    pub(crate) fn b(&mut self, l: Label) {
        self.fixup(l, FixupKind::B26);
        self.emit(0x1400_0000);
    }

    pub(crate) fn bl(&mut self, l: Label) {
        self.fixup(l, FixupKind::B26);
        self.emit(0x9400_0000);
    }

    /// `b` to a fixed address outside the buffer.
    pub(crate) fn b_far(&mut self, target: usize) {
        self.externs.push((self.words.len() as u32, target));
        self.emit(0x1400_0000);
    }

    /// Bytes `finish` adds: the literal pool, aligned.
    pub(crate) fn pool_bytes(&self) -> usize {
        if self.literals.is_empty() {
            0
        } else {
            8 + self.literals.len() * 8
        }
    }

    /// `bl` to a fixed address outside the buffer.
    pub(crate) fn bl_far(&mut self, target: usize) {
        self.externs.push((self.words.len() as u32, target));
        self.emit(0x9400_0000);
    }

    pub(crate) fn b_cond(&mut self, c: Cond, l: Label) {
        self.fixup(l, FixupKind::B19);
        self.emit(0x5400_0000 | c as u32);
    }

    pub(crate) fn cbz(&mut self, sz: Sz, t: Gpr, l: Label) {
        self.fixup(l, FixupKind::B19);
        self.emit(sz.sf() | 0x3400_0000 | r(t));
    }

    pub(crate) fn cbnz(&mut self, sz: Sz, t: Gpr, l: Label) {
        self.fixup(l, FixupKind::B19);
        self.emit(sz.sf() | 0x3500_0000 | r(t));
    }

    pub(crate) fn tbz(&mut self, t: Gpr, bit: u32, l: Label) {
        debug_assert!(bit < 64);
        self.fixup(l, FixupKind::B14);
        self.emit(0x3600_0000 | (bit >> 5) << 31 | (bit & 31) << 19 | r(t));
    }

    pub(crate) fn tbnz(&mut self, t: Gpr, bit: u32, l: Label) {
        debug_assert!(bit < 64);
        self.fixup(l, FixupKind::B14);
        self.emit(0x3700_0000 | (bit >> 5) << 31 | (bit & 31) << 19 | r(t));
    }

    pub(crate) fn br(&mut self, n: Gpr) {
        self.emit(0xD61F_0000 | r(n) << 5);
    }

    pub(crate) fn blr(&mut self, n: Gpr) {
        self.emit(0xD63F_0000 | r(n) << 5);
    }

    pub(crate) fn ret(&mut self) {
        self.emit(0xD65F_03C0);
    }

    pub(crate) fn brk(&mut self, imm: u16) {
        self.emit(0xD420_0000 | (imm as u32) << 5);
    }

    pub(crate) fn nop(&mut self) {
        self.emit(0xD503_201F);
    }
}

/// The `N:immr:imms` field of a logical immediate of `width` bits, if `v`
/// is one: a rotated run of ones replicated across the register.
pub(crate) fn logical_imm(v: u64, width: u32) -> Option<u32> {
    let v = if width == 32 {
        let v = v & 0xffff_ffff;
        v | v << 32
    } else {
        v
    };
    if v == 0 || v == u64::MAX {
        return None;
    }
    // Smallest element size whose pattern repeats.
    let mut size = 64u32;
    while size > 2 {
        let half = size / 2;
        let mask = (1u64 << half) - 1;
        if (v & mask) != ((v >> half) & mask) {
            break;
        }
        size = half;
    }
    let mask = if size == 64 {
        u64::MAX
    } else {
        (1u64 << size) - 1
    };
    let elem = v & mask;
    // A rotation of a contiguous run of ones: count the ones, find the rotation.
    let ones = elem.count_ones();
    if ones == 0 || ones == size {
        return None;
    }
    // Rotate right until the run sits at the bottom.
    let rot_right = |x: u64, s: u32| -> u64 {
        if s == 0 {
            x
        } else {
            ((x >> s) | (x << (size - s))) & mask
        }
    };
    let run = (1u64 << ones) - 1;
    let mut rot = None;
    for s in 0..size {
        if rot_right(elem, s) == run {
            rot = Some(s);
            break;
        }
    }
    let s = rot?;
    // `immr` rotates the run right; we rotated `elem` right by `s` to get the
    // run, so the run is rotated right by `size - s` to get `elem`.
    let immr = (size - s) % size;
    let imms = ((!(size * 2 - 1)) & 0x3f) | (ones - 1);
    let n = if size == 64 { 1 } else { 0 };
    if width == 32 && n == 1 {
        return None;
    }
    Some(n << 12 | immr << 6 | (imms & 0x3f))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble `src` with the system assembler and return its words.
    fn system_asm(src: &str) -> Option<Vec<u32>> {
        let dir =
            std::env::temp_dir().join(format!("tcvm-asm-{}-{}", std::process::id(), src.len()));
        std::fs::create_dir_all(&dir).ok()?;
        let s = dir.join("t.s");
        let o = dir.join("t.o");
        std::fs::write(&s, format!(".text\n.globl _t\n_t:\n{src}\n")).ok()?;
        let st = std::process::Command::new("clang")
            .args(["-c", "-arch", "arm64", "-o"])
            .arg(&o)
            .arg(&s)
            .status()
            .ok()?;
        if !st.success() {
            return None;
        }
        let out = std::process::Command::new("otool")
            .args(["-t", "-X"])
            .arg(&o)
            .output()
            .ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        let mut words = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let _addr = it.next();
            for w in it {
                words.push(u32::from_str_radix(w, 16).ok()?);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        Some(words)
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn matches_system_assembler() {
        let mut a = Asm::new();
        let mut src = String::new();
        macro_rules! t {
            ($text:expr, $e:expr) => {{
                $e;
                src.push_str($text);
                src.push('\n');
            }};
        }
        let (x0, x1, x2, x3, x9) = (Gpr(0), Gpr(1), Gpr(2), Gpr(3), Gpr(9));
        let (d0, d1, d2) = (Fpr(0), Fpr(1), Fpr(2));
        t!("mov x0, x1", a.mov(x0, x1));
        t!("mov w0, w1", a.mov_w(x0, x1));
        t!("mov x0, sp", a.mov_sp(x0, SP));
        t!("mov sp, x29", a.mov_sp(SP, FP));
        t!("movz x0, #0x1234, lsl #16", a.movz(Sz::X, x0, 0x1234, 1));
        t!("movn w1, #5", a.movn(Sz::W, x1, 5, 0));
        t!("movk x2, #0xbeef, lsl #48", a.movk(Sz::X, x2, 0xbeef, 3));
        t!("add x0, x1, x2", a.add(Sz::X, x0, x1, x2));
        t!("add w0, w1, w2", a.add(Sz::W, x0, x1, x2));
        t!("adds w0, w1, w2", a.adds(Sz::W, x0, x1, x2));
        t!("sub x0, x1, x2", a.sub(Sz::X, x0, x1, x2));
        t!("subs w3, w1, w2", a.subs(Sz::W, x3, x1, x2));
        t!(
            "add x0, x1, x2, lsl #3",
            a.add_shift(Sz::X, x0, x1, x2, Shift::Lsl, 3)
        );
        t!(
            "sub x0, x1, x2, asr #63",
            a.sub_shift(Sz::X, x0, x1, x2, Shift::Asr, 63)
        );
        t!("cmp x1, x2", a.cmp(Sz::X, x1, x2));
        t!("cmp w1, w2", a.cmp(Sz::W, x1, x2));
        t!("cmn w1, w2", a.cmn(Sz::W, x1, x2));
        t!(
            "cmp x1, x2, lsr #32",
            a.cmp_shift(Sz::X, x1, x2, Shift::Lsr, 32)
        );
        t!("cmp x1, w2, sxtw", a.cmp_ext(x1, x2, Extend::Sxtw));
        t!(
            "add x0, x1, w2, uxtw #3",
            a.add_ext(x0, x1, x2, Extend::Uxtw, 3)
        );
        t!("add x0, x1, #4095", a.add_imm(Sz::X, x0, x1, 4095));
        t!("add x0, sp, #16", a.add_imm(Sz::X, x0, SP, 16));
        t!("add w0, w1, #0x1000", a.add_imm(Sz::W, x0, x1, 0x1000));
        t!("adds w0, w1, #7", a.adds_imm(Sz::W, x0, x1, 7));
        t!("sub x0, x1, #8", a.sub_imm(Sz::X, x0, x1, 8));
        t!("subs w0, w1, #9", a.subs_imm(Sz::W, x0, x1, 9));
        t!("cmp x1, #2", a.cmp_imm(Sz::X, x1, 2));
        t!("cmn w1, #1", a.cmn_imm(Sz::W, x1, 1));
        t!("and x0, x1, x2", a.and(Sz::X, x0, x1, x2));
        t!("orr w0, w1, w2", a.orr(Sz::W, x0, x1, x2));
        t!("eor x0, x1, x2", a.eor(Sz::X, x0, x1, x2));
        t!("tst x1, x2", a.tst(Sz::X, x1, x2));
        t!(
            "orr x0, x1, x2, lsl #48",
            a.orr_shift(Sz::X, x0, x1, x2, Shift::Lsl, 48)
        );
        t!(
            "eor w0, w1, w2, asr #31",
            a.eor_shift(Sz::W, x0, x1, x2, Shift::Asr, 31)
        );
        t!("mvn x0, x1", a.mvn(Sz::X, x0, x1));
        t!("mvn w0, w1", a.mvn(Sz::W, x0, x1));
        t!("neg w0, w1", a.neg(Sz::W, x0, x1));
        t!("negs w0, w1", a.negs(Sz::W, x0, x1));
        t!(
            "and x0, x1, #0xffffffffffff",
            a.and_imm(Sz::X, x0, x1, 0xffff_ffff_ffff)
        );
        t!(
            "orr x0, x1, #0xffffffff00000000",
            a.orr_imm(Sz::X, x0, x1, 0xffff_ffff_0000_0000)
        );
        t!(
            "orr x0, x1, #0xfffc000000000000",
            a.orr_imm(Sz::X, x0, x1, 0xfffc_0000_0000_0000)
        );
        t!(
            "orr x0, x1, #0xffff000000000000",
            a.orr_imm(Sz::X, x0, x1, 0xffff_0000_0000_0000)
        );
        t!("eor w0, w1, #1", a.eor_imm(Sz::W, x0, x1, 1));
        t!("and w0, w1, #0x3f", a.and_imm(Sz::W, x0, x1, 0x3f));
        t!(
            "tst x1, #0x5555555555555555",
            a.tst_imm(Sz::X, x1, 0x5555_5555_5555_5555)
        );
        t!("tst w1, #0xff00", a.tst_imm(Sz::W, x1, 0xff00));
        t!("lsl x0, x1, x2", a.lslv(Sz::X, x0, x1, x2));
        t!("lsr w0, w1, w2", a.lsrv(Sz::W, x0, x1, x2));
        t!("asr x0, x1, x2", a.asrv(Sz::X, x0, x1, x2));
        t!("sdiv w0, w1, w2", a.sdiv(Sz::W, x0, x1, x2));
        t!("sdiv x0, x1, x2", a.sdiv(Sz::X, x0, x1, x2));
        t!("udiv x0, x1, x2", a.udiv(Sz::X, x0, x1, x2));
        t!("madd x0, x1, x2, x3", a.madd(Sz::X, x0, x1, x2, x3));
        t!("msub w0, w1, w2, w3", a.msub(Sz::W, x0, x1, x2, x3));
        t!("mul x0, x1, x2", a.mul(Sz::X, x0, x1, x2));
        t!("smull x0, w1, w2", a.smull(x0, x1, x2));
        t!("smulh x0, x1, x2", a.smulh(x0, x1, x2));
        t!("csel x0, x1, x2, lt", a.csel(Sz::X, x0, x1, x2, Cond::Lt));
        t!("csel w0, w1, w2, hs", a.csel(Sz::W, x0, x1, x2, Cond::Hs));
        t!("csinc x0, x1, x2, ne", a.csinc(Sz::X, x0, x1, x2, Cond::Ne));
        t!("csinv x0, x1, x2, eq", a.csinv(Sz::X, x0, x1, x2, Cond::Eq));
        t!("csneg w0, w1, w2, gt", a.csneg(Sz::W, x0, x1, x2, Cond::Gt));
        t!("cset w0, eq", a.cset(Sz::W, x0, Cond::Eq));
        t!("ccmp x1, x2, #4, ne", a.ccmp(Sz::X, x1, x2, 4, Cond::Ne));
        t!("ccmp w1, #0, #8, eq", a.ccmp_imm(Sz::W, x1, 0, 8, Cond::Eq));
        t!("ccmn w1, #1, #0, ne", a.ccmn_imm(Sz::W, x1, 1, 0, Cond::Ne));
        t!("lsl x0, x1, #3", a.lsl_imm(Sz::X, x0, x1, 3));
        t!("lsl w0, w1, #31", a.lsl_imm(Sz::W, x0, x1, 31));
        t!("lsr x0, x1, #32", a.lsr_imm(Sz::X, x0, x1, 32));
        t!("lsr x9, x1, #48", a.lsr_imm(Sz::X, x9, x1, 48));
        t!("asr w0, w1, #31", a.asr_imm(Sz::W, x0, x1, 31));
        t!("ubfx x0, x1, #16, #8", a.ubfx(Sz::X, x0, x1, 16, 8));
        t!("sbfx x0, x1, #17, #15", a.sbfx(Sz::X, x0, x1, 17, 15));
        t!("sxtw x0, w1", a.sxtw(x0, x1));
        t!("ldr x0, [x1, #8]", a.ldr(x0, x1, 8));
        t!("ldr x0, [x1, #-16]", a.ldr(x0, x1, -16));
        t!("ldur x0, [x1, #3]", a.ldr(x0, x1, 3));
        t!("str x0, [x22, #2040]", a.str(x0, Gpr(22), 2040));
        t!("ldr w0, [x1, #4]", a.ldr_w(x0, x1, 4));
        t!("str w0, [x1, #8]", a.str_w(x0, x1, 8));
        t!("ldrsw x0, [x1, #8]", a.ldrsw(x0, x1, 8));
        t!("ldrh w0, [x23, #4100]", a.ldrh(x0, Gpr(23), 4100));
        t!("strh w0, [x1, #2]", a.strh(x0, x1, 2));
        t!("ldrb w0, [x1, #1]", a.ldrb(x0, x1, 1));
        t!("strb w0, [x1, #7]", a.strb(x0, x1, 7));
        t!("ldr d0, [x1, #8]", a.ldr_d(d0, x1, 8));
        t!("str d1, [x1, #-8]", a.str_d(d1, x1, -8));
        t!(
            "ldr x0, [x1, x2, lsl #3]",
            a.ldr_idx(x0, x1, x2, Extend::Lsl, true)
        );
        t!(
            "ldr x0, [x1, w2, uxtw #3]",
            a.ldr_idx(x0, x1, x2, Extend::Uxtw, true)
        );
        t!(
            "str x0, [x1, x2, lsl #3]",
            a.str_idx(x0, x1, x2, Extend::Lsl, true)
        );
        t!(
            "ldr d0, [x1, x2, lsl #3]",
            a.ldr_d_idx(d0, x1, x2, Extend::Lsl, true)
        );
        t!(
            "str d0, [x1, w2, sxtw #3]",
            a.str_d_idx(d0, x1, x2, Extend::Sxtw, true)
        );
        t!("stp x0, x1, [x2, #16]", a.stp(x0, x1, x2, 16));
        t!("ldp x0, x1, [x2, #-16]", a.ldp(x0, x1, x2, -16));
        t!("stp x29, x30, [sp, #-48]!", a.stp_pre(FP, LR, SP, -48));
        t!("ldp x29, x30, [sp], #48", a.ldp_post(FP, LR, SP, 48));
        t!("stp d0, d1, [x17, #176]", a.stp_d(d0, d1, Gpr(17), 176));
        t!("ldp d0, d1, [x17, #176]", a.ldp_d(d0, d1, Gpr(17), 176));
        t!("fadd d0, d1, d2", a.fadd(d0, d1, d2));
        t!("fsub d0, d1, d2", a.fsub(d0, d1, d2));
        t!("fmul d0, d1, d2", a.fmul(d0, d1, d2));
        t!("fdiv d0, d1, d2", a.fdiv(d0, d1, d2));
        t!("fmax d0, d1, d2", a.fmax(d0, d1, d2));
        t!("fmin d0, d1, d2", a.fmin(d0, d1, d2));
        t!("fmov d0, d1", a.fmov(d0, d1));
        t!("fneg d0, d1", a.fneg(d0, d1));
        t!("fabs d0, d1", a.fabs(d0, d1));
        t!("fsqrt d0, d1", a.fsqrt(d0, d1));
        t!("frintm d0, d1", a.frintm(d0, d1));
        t!("frintp d0, d1", a.frintp(d0, d1));
        t!("frintz d0, d1", a.frintz(d0, d1));
        t!("fcmp d0, d1", a.fcmp(d0, d1));
        t!("fcmp d1, #0.0", a.fcmp_zero(d1));
        t!("fcsel d0, d1, d2, mi", a.fcsel(d0, d1, d2, Cond::Mi));
        t!("fmov d0, x1", a.fmov_from_gpr(d0, x1));
        t!("fmov x0, d1", a.fmov_to_gpr(x0, d1));
        t!("scvtf d0, w1", a.scvtf(Sz::W, d0, x1));
        t!("scvtf d0, x1", a.scvtf(Sz::X, d0, x1));
        t!("fcvtzs w0, d1", a.fcvtzs(Sz::W, x0, d1));
        t!("fcvtzs x0, d1", a.fcvtzs(Sz::X, x0, d1));
        for v in [1.0f64, 2.0, 0.5, -1.0, 0.125, 31.0, -3.5, 10.0] {
            let imm = Asm::fmov_imm8(v).expect("fmov immediate");
            src.push_str(&format!("fmov d2, #{v:?}\n"));
            a.fmov_imm(d2, imm);
        }
        assert!(Asm::fmov_imm8(0.1).is_none());
        assert!(Asm::fmov_imm8(0.0).is_none());
        assert!(Asm::fmov_imm8(64.0).is_none());
        t!("br x16", a.br(Gpr(16)));
        t!("blr x9", a.blr(x9));
        t!("ret", a.ret());
        t!("brk #1", a.brk(1));
        t!("nop", a.nop());
        let l = a.new_label();
        let l2 = a.new_label();
        t!("b 1f", a.b(l));
        t!("b.ne 1f", a.b_cond(Cond::Ne, l));
        t!("cbz w1, 1f", a.cbz(Sz::W, x1, l));
        t!("cbnz x1, 1f", a.cbnz(Sz::X, x1, l));
        t!("tbz x1, #0, 1f", a.tbz(x1, 0, l));
        t!("tbnz x1, #47, 1f", a.tbnz(x1, 47, l));
        t!("adr x0, 1f", a.adr(x0, l));
        t!("bl 1f", a.bl(l));
        a.bind(l);
        src.push_str("1:\n");
        t!("b.eq 1b", a.b_cond(Cond::Eq, l));
        t!("b 2f", a.b(l2));
        a.bind(l2);
        src.push_str("2:\n");
        for v in [
            0u64,
            1,
            0xffff,
            0x1_0000,
            0xffff_ffff_0000_0000,
            0xffff_fffe_0000_0000,
            0xfff9_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x1234_5678_9abc_def0,
            u64::MAX,
            u64::MAX - 5,
            0xffff_0000_ffff_ffff,
        ] {
            let before = a.words().len();
            a.mov_imm(x0, v);
            let n = a.words().len() - before;
            let mut b = Asm::new();
            b.mov_imm(x0, v);
            assert_eq!(b.words().len(), n);
            // Check by emulation instead of text: the value each sequence makes.
            let mut x = 0u64;
            for &w in b.words() {
                x = emulate_mov(x, w).unwrap_or_else(|| panic!("not a mov word {w:#x}"));
            }
            assert_eq!(x, v, "mov_imm {v:#x}");
            // Keep the system-assembler comparison aligned: assemble the same words.
            for &w in b.words() {
                src.push_str(&format!(".inst {w:#x}\n"));
            }
        }
        let words = a.finish(0).unwrap();
        let Some(sys) = system_asm(&src) else {
            eprintln!("no system assembler; skipping");
            return;
        };
        assert_eq!(words.len(), sys.len(), "length mismatch");
        let lines: Vec<&str> = src.lines().filter(|l| !l.ends_with(':')).collect();
        for (i, (&w, &s)) in words.iter().zip(&sys).enumerate() {
            assert_eq!(
                w,
                s,
                "word {i} ({}): ours {w:#010x}, system {s:#010x}",
                lines.get(i).unwrap_or(&"?")
            );
        }
    }

    fn emulate_mov(x: u64, w: u32) -> Option<u64> {
        let d = w & 31;
        assert_eq!(d, 0);
        let hw = (w >> 21) & 3;
        let imm = ((w >> 5) & 0xffff) as u64;
        match w & 0xff80_0000 {
            0xD280_0000 => Some(imm << (hw * 16)),
            0x9280_0000 => Some(!(imm << (hw * 16))),
            0xF280_0000 => Some((x & !(0xffff << (hw * 16))) | imm << (hw * 16)),
            _ if w & 0xff80_03e0 == 0xB200_03e0 => {
                // orr x0, xzr, #imm: decode the logical immediate.
                let n = (w >> 22) & 1;
                let immr = (w >> 16) & 63;
                let imms = (w >> 10) & 63;
                Some(decode_logical(n, immr, imms))
            }
            _ => None,
        }
    }

    fn decode_logical(n: u32, immr: u32, imms: u32) -> u64 {
        let len = 31 - ((n << 6) | (!imms & 0x3f)).leading_zeros();
        let size = 1u32 << len;
        let s = imms & (size - 1);
        let rr = immr & (size - 1);
        let mask = if size == 64 {
            u64::MAX
        } else {
            (1u64 << size) - 1
        };
        let welem = (1u64 << (s + 1)) - 1;
        let elem = if rr == 0 {
            welem
        } else {
            ((welem >> rr) | (welem << (size - rr))) & mask
        };
        let mut v = 0u64;
        let mut i = 0;
        while i < 64 {
            v |= elem << i;
            i += size;
        }
        v
    }

    #[test]
    fn logical_immediates_round_trip() {
        for size in [2u32, 4, 8, 16, 32, 64] {
            for ones in 1..size {
                for rot in 0..size {
                    let mask = if size == 64 {
                        u64::MAX
                    } else {
                        (1u64 << size) - 1
                    };
                    let run = (1u64 << ones) - 1;
                    let elem = if rot == 0 {
                        run
                    } else {
                        ((run >> rot) | (run << (size - rot))) & mask
                    };
                    let mut v = 0u64;
                    let mut i = 0;
                    while i < 64 {
                        v |= elem << i;
                        i += size;
                    }
                    let enc = logical_imm(v, 64).unwrap_or_else(|| panic!("{v:#x} not encoded"));
                    let (n, immr, imms) = (enc >> 12, (enc >> 6) & 63, enc & 63);
                    assert_eq!(decode_logical(n, immr, imms), v, "{v:#x}");
                }
            }
        }
        assert!(logical_imm(0x1234, 64).is_none());
    }
}
