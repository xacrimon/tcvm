//! Each instruction's sequence, with the tests branches, guards, selects
//! and sets share.

use crate::env::table::SlotLoc;
use crate::env::value::{BOX, FALSE, NIL, PAYLOAD_MASK as PTR_MASK, SMALL_INT as SMALL_BITS};
use crate::env::{MtCache, Shape};
use crate::jit::backend::aarch64::abi::{BASE, CLOSURE, D31, THREAD, X16, X17};
use crate::jit::backend::aarch64::asm::{Cond, Extend, Fpr, Gpr, Label, SP, Shift, Sz, ZR};
use crate::jit::backend::aarch64::emit::{Em, fpr, gpr};
use crate::jit::backend::aarch64::inst::{AluOp, FOp, FUnOp, MInst, Test};
use crate::jit::backend::aarch64::lower::{float_cond, int_cond};
use crate::jit::backend::aarch64::types::{is_single_test, type_branch, type_fail, type_flags};
use crate::jit::backend::reg::{PReg, RegClass};
use crate::jit::ir::ops::{Cc, Ctx};
use crate::jit::ir::types::TypeSet;
use crate::jit::layout;
use crate::jit::layout::FUNCTION_TAG;

/// The operand size of a test of one register against zero for equality,
/// which `cbz`/`cbnz` branch on without flags; a B1 is a W.
fn zero_test(test: Test, cond: Cond) -> Option<Sz> {
    if !matches!(cond, Cond::Eq | Cond::Ne) {
        return None;
    }
    match test {
        Test::B1 | Test::I32 { imm: Some(0) } => Some(Sz::W),
        Test::X { imm: Some(0) } => Some(Sz::X),
        _ => None,
    }
}

impl<'a> Em<'a> {
    /// The branch `i` to `lt` where its test holds, else `lf`.
    pub(super) fn br(
        &mut self,
        i: usize,
        test: Test,
        cond: Cond,
        lt: Label,
        lf: Label,
        next: Option<Label>,
    ) {
        let multi = matches!(test, Test::Type(set, _) if !is_single_test(set));
        if test == Test::B1 || multi {
            // Where `cond` holds, and where it does not.
            let (yes, no) = if cond == Cond::Eq || (test == Test::B1 && cond == Cond::Ne) {
                (lt, lf)
            } else {
                (lf, lt)
            };
            let v = self.r(i, 0);
            let w = match test {
                Test::Type(set, words) => self.words(i, 0, set, words),
                _ => Vec::new(),
            };
            if Some(yes) == next {
                match test {
                    Test::Type(set, _) => type_fail(&mut self.a, v, set, no, &w),
                    _ => self.a.cbz(Sz::W, v, no),
                }
                return;
            }
            match test {
                Test::Type(set, _) => type_branch(&mut self.a, v, set, yes, &w),
                _ => self.a.cbnz(Sz::W, v, yes),
            }
            if Some(no) != next {
                self.a.b(no);
            }
            return;
        }
        if let Test::Bit(b) | Test::MmBit(b) = test
            && self.bit_branches
        {
            let (set, clear) = if cond == Cond::Ne { (lt, lf) } else { (lf, lt) };
            let after = self.a.new_label();
            let x = match test {
                Test::MmBit(_) => {
                    let t = self.r(i, 0);
                    // `next` is reached by falling through, its edge's moves
                    // in between.
                    let none = if Some(clear) == next { after } else { clear };
                    self.class_bits(X16, t, none);
                    X16
                }
                _ => self.r(i, 0),
            };
            if Some(set) == next {
                self.a.tbz(x, b as u32, clear);
            } else {
                self.a.tbnz(x, b as u32, set);
                if Some(clear) != next {
                    self.a.b(clear);
                }
            }
            self.a.bind(after);
            return;
        }
        if let Some(sz) = zero_test(test, cond) {
            // `cbz` where the operand is zero (`Eq` holding).
            let v = self.r(i, 0);
            let (on_zero, on_other) = if cond == Cond::Eq { (lt, lf) } else { (lf, lt) };
            if Some(on_zero) == next {
                self.a.cbnz(sz, v, on_other);
            } else {
                self.a.cbz(sz, v, on_zero);
                if Some(on_other) != next {
                    self.a.b(on_other);
                }
            }
            return;
        }
        let c = self.test(i, 0, test, cond);
        if Some(lt) == next {
            self.a.b_cond(c.invert(), lf);
        } else {
            self.a.b_cond(c, lt);
            if Some(lf) != next {
                self.a.b(lf);
            }
        }
    }

    /// `d` = the metamethod bits of table `t`'s class, or on to `none`
    /// without a class, `d` zero there.
    pub(super) fn class_bits(&mut self, d: Gpr, t: Gpr, none: Label) {
        self.a.ldr(d, t, layout::T_SHAPE as i32);
        self.a.ldr(d, d, Shape::MT_CACHE_OFFSET as i32);
        self.a.cbz(Sz::X, d, none);
        self.a.ldr_w(d, d, MtCache::BITS_OFFSET as i32);
    }

    /// Flags `Eq` when table `t`'s shape is one of `shape_sets[set]`: one
    /// load, then a compare per shape, each after the first only while none
    /// matched (`ccmp`, `Z` set otherwise).
    pub(super) fn shape_in(&mut self, t: Gpr, set: u32) {
        self.a.ldr(X16, t, layout::T_SHAPE as i32);
        for (n, &word) in self.l.shape_sets[set as usize].iter().enumerate() {
            self.a.ldr_lit(X17, word);
            if n == 0 {
                self.a.cmp(Sz::X, X16, X17);
            } else {
                self.a.ccmp(Sz::X, X16, X17, 0b0100, Cond::Ne);
            }
        }
    }

    /// Set the flags for `test` on instruction `i`'s operands from `k`;
    /// returns the condition that means the test holds, given `cond` (the
    /// IR's condition for compares, ignored for type tests).
    pub(super) fn test(&mut self, i: usize, k: usize, test: Test, cond: Cond) -> Cond {
        match test {
            Test::I32 { imm: Some(n) } => {
                let x = self.r(i, k);
                if n < 0 {
                    self.a.cmn_imm(Sz::W, x, (-n) as u64);
                } else {
                    self.a.cmp_imm(Sz::W, x, n as u64);
                }
                cond
            }
            Test::I32 { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::W, x, y);
                cond
            }
            Test::X { imm: Some(n) } => {
                let x = self.r(i, k);
                if n < 0 {
                    self.a.cmn_imm(Sz::X, x, (-n) as u64);
                } else {
                    self.a.cmp_imm(Sz::X, x, n as u64);
                }
                cond
            }
            Test::X { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::X, x, y);
                cond
            }
            Test::F64 { zero: true } => {
                let x = self.fr(i, k);
                self.a.fcmp_zero(x);
                cond
            }
            Test::F64 { zero: false } => {
                let (x, y) = (self.fr(i, k), self.fr(i, k + 1));
                self.a.fcmp(x, y);
                cond
            }
            // `cond` is `Lo` for falsy, `Hs` for truthy.
            Test::Falsy(words) => {
                let v = self.r(i, k);
                let w = self.words(i, k, TypeSet::FALSY, words);
                type_flags(&mut self.a, v, TypeSet::FALSY, &w);
                cond
            }
            Test::B1 => {
                let c = self.r(i, k);
                self.a.cmp_imm(Sz::W, c, 0);
                cond
            }
            Test::Bit(b) => {
                let x = self.r(i, k);
                self.a.tst_imm(Sz::W, x, 1 << b);
                cond
            }
            Test::MmBit(b) => {
                let t = self.r(i, k);
                let done = self.a.new_label();
                self.class_bits(X16, t, done);
                self.a.bind(done);
                self.a.tst_imm(Sz::W, X16, 1 << b);
                cond
            }
            Test::HasFlag(mask) => {
                self.a.ldr(X16, BASE, -24);
                self.a.tst_imm(Sz::X, X16, mask as u64);
                cond
            }
            Test::NumMtSet => {
                self.a.ldr_lit(
                    X16,
                    self.links.type_mt_cells[layout::NUMBER_MT_SLOT as usize],
                );
                self.a.ldr(X16, X16, layout::NUMBER_MT as i32);
                self.a.cmp_imm(Sz::X, X16, 0);
                cond
            }
            Test::TypeMtIs { slot, word } => {
                self.a.ldr_lit(X16, self.links.type_mt_cells[slot as usize]);
                self.a.ldr(X16, X16, layout::NUMBER_MT as i32);
                self.a.ldr_lit(X17, word);
                self.a.cmp(Sz::X, X16, X17);
                cond
            }
            // A native function has the niche value of `is_vararg` there.
            Test::FnFixedLua | Test::FnNative => {
                // 0 and 1 for a Lua closure's flag, past them a native's niche.
                let f = self.r(i, k);
                self.a.ldrb(X16, f, layout::FN_IS_VARARG as i32);
                self.a.cmp_imm(Sz::W, X16, (test == Test::FnNative) as u64);
                cond
            }
            Test::FnFits => {
                let f = self.r(i, k);
                self.a.ldrb(X16, f, layout::FN_MAX_STACK as i32);
                self.a.add_shift(Sz::X, X16, BASE, X16, Shift::Lsl, 3);
                self.a.ldr(X17, THREAD, layout::THREAD_STACK_END as i32);
                self.a.cmp(Sz::X, X16, X17);
                cond
            }
            Test::TabHasRoom(loc) => {
                let t = self.r(i, k);
                let slot = SlotLoc::from_raw(loc).offset() / 8;
                self.a.ldr_w(X16, t, layout::T_SPILL_CAP as i32);
                if slot < 4096 {
                    self.a.cmp_imm(Sz::W, X16, slot as u64);
                } else {
                    self.a.mov_imm(X17, slot as u64);
                    self.a.cmp(Sz::W, X16, X17);
                }
                cond
            }
            // No cell, or the parts' counts or'd: zero. The cell is loaded
            // again for each part past the second, x16 taking the counts.
            Test::TabNoKeys(parts) => {
                let t = self.r(i, k);
                let done = self.a.new_label();
                self.a.ldr(X16, t, layout::T_AUX as i32);
                self.a.cbz(Sz::X, X16, done);
                let mut offs = layout::aux_counts(parts).peekable();
                let first = offs.next().expect("a part");
                if offs.peek().is_none() {
                    self.a.ldr(X16, X16, first as i32);
                } else {
                    self.a.ldr(X17, X16, first as i32);
                    let mut again = false;
                    while let Some(off) = offs.next() {
                        if again {
                            self.a.ldr(X16, t, layout::T_AUX as i32);
                        }
                        self.a.ldr(X16, X16, off as i32);
                        let d = if offs.peek().is_some() { X17 } else { X16 };
                        self.a.orr(Sz::X, d, X17, X16);
                        again = true;
                    }
                }
                self.a.bind(done);
                self.a.cmp_imm(Sz::X, X16, 0);
                cond
            }
            Test::Shape(word) => {
                let t = self.r(i, k);
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_lit(X17, word);
                self.a.cmp(Sz::X, X16, X17);
                cond
            }
            Test::ShapeIn(set) => {
                let t = self.r(i, k);
                self.shape_in(t, set);
                cond
            }
            // `cond` is `Eq` for in the set, `Ne` for not.
            Test::Type(set, words) => {
                let v = self.r(i, k);
                let w = self.words(i, k, set, words);
                let c = type_flags(&mut self.a, v, set, &w);
                if cond == Cond::Eq { c } else { c.invert() }
            }
        }
    }

    /// Exit unless instruction `i`'s operand 0 is in `set`.
    pub(super) fn type_guard(&mut self, i: usize, set: TypeSet, exit: u32) {
        let (l, v) = (self.exit(exit), self.r(i, 0));
        type_fail(&mut self.a, v, set, l, &[]);
    }

    pub(super) fn inst(
        &mut self,
        i: usize,
        block: usize,
        next: Option<Label>,
    ) -> Result<(), String> {
        let inst = self.v.insts[i].clone();
        #[cfg(feature = "tcvm-dev")]
        if let Some(fired) = self.links.fired
            && let Some(k) = inst.exit_id()
            && self.forced[k as usize]
        {
            // An exit that never fired fires now.
            let l = self.exit(k);
            self.a.ldr_lit(X16, (fired + 4 * k as usize) as u64);
            self.a.ldr_w(X16, X16, 0);
            self.a.cbz(Sz::W, X16, l);
        }
        use MInst::*;
        match inst {
            Prologue => self.open_frame(),
            Params => {}
            // A value the allocator computes again where used.
            MovImm(_) | FImm(_) | AluImm(..) | LoadSlot(_) if self.allocs(i)[0].is_none() => {}
            MovImm(v) => {
                let d = self.r(i, 0);
                self.a.mov_imm(d, v);
            }
            FImm(bits) => {
                let d = self.fr(i, 0);
                self.fimm(d, bits);
            }
            LoadSlot(r) => {
                let d = self.reg(i, 0);
                self.word_at(false, d, BASE, r as i32 * 8);
            }
            StoreSlot(r) => {
                let v = self.reg(i, 0);
                self.word_at(true, v, BASE, r as i32 * 8);
            }
            FrameClosure => {
                let d = self.r(i, 0);
                self.a.ldr(d, BASE, -32);
                self.a.and_imm(Sz::X, d, d, PTR_MASK);
            }
            LoadUpvalOf(k) => {
                let (d, c) = (self.reg(i, 0), self.r(i, 1));
                let off = crate::jit::compile::upvalues_offset() + k as usize * 8;
                self.word_at(false, d, c, off as i32);
            }
            Alu(op, sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match op {
                    AluOp::Add => self.a.add(sz, d, n, m),
                    AluOp::Sub => self.a.sub(sz, d, n, m),
                    AluOp::And => self.a.and(sz, d, n, m),
                    AluOp::Orr => self.a.orr(sz, d, n, m),
                    AluOp::Eor => self.a.eor(sz, d, n, m),
                    AluOp::Mul => self.a.mul(sz, d, n, m),
                    AluOp::Udiv => self.a.udiv(sz, d, n, m),
                }
            }
            AluImm(op, sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                match op {
                    AluOp::Add => self.a.add_imm(sz, d, n, k),
                    AluOp::Sub => self.a.sub_imm(sz, d, n, k),
                    AluOp::And => self.a.and_imm(sz, d, n, k),
                    AluOp::Orr => self.a.orr_imm(sz, d, n, k),
                    AluOp::Eor => self.a.eor_imm(sz, d, n, k),
                    _ => return Err(format!("{op:?} with an immediate")),
                }
            }
            ShiftImm(sh, sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                match sh {
                    Shift::Lsl => self.a.lsl_imm(sz, d, n, k as u32),
                    Shift::Lsr => self.a.lsr_imm(sz, d, n, k as u32),
                    Shift::Asr => self.a.asr_imm(sz, d, n, k as u32),
                }
            }
            ShiftReg(sh, sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                match sh {
                    Shift::Lsl => self.a.lslv(sz, d, n, m),
                    Shift::Lsr => self.a.lsrv(sz, d, n, m),
                    Shift::Asr => self.a.asrv(sz, d, n, m),
                }
            }
            MulAdd { sub, sz } => {
                let (d, n, m, a) = (self.r(i, 0), self.r(i, 1), self.r(i, 2), self.r(i, 3));
                if sub {
                    self.a.msub(sz, d, n, m, a);
                } else {
                    self.a.madd(sz, d, n, m, a);
                }
            }
            UMulShr(k) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.umull(d, n, m);
                self.a.lsr_imm(Sz::X, d, d, k as u32);
            }
            Neg(sz) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.neg(sz, d, n);
            }
            Mvn(sz) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.mvn(sz, d, n);
            }
            AddOvf { sub, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                if sub {
                    self.a.subs(Sz::W, d, n, m);
                } else {
                    self.a.adds(Sz::W, d, n, m);
                }
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            AddImmOvf { sub, imm, exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                if sub {
                    self.a.subs_imm(Sz::W, d, n, imm as u64);
                } else {
                    self.a.adds_imm(Sz::W, d, n, imm as u64);
                }
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            MulOvf { exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.smull(X16, n, m);
                self.a.cmp_ext(X16, X16, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov_w(d, X16);
            }
            NegOvf { exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.negs(Sz::W, d, n);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Vs, l);
            }
            // `cneg` is `csneg` of the operand with itself, inverted.
            AbsOvf { exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.cmp_imm(Sz::W, n, 0);
                self.a.csneg(Sz::W, d, n, n, Cond::Ge);
                // Negative only for `MIN`, which has no i32 absolute value.
                let l = self.exit(exit);
                if self.bit_branches {
                    self.a.tbnz(d, 31, l);
                } else {
                    self.a.cmp_imm(Sz::W, d, 0);
                    self.a.b_cond(Cond::Lt, l);
                }
            }
            Abs => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.cmp_imm(Sz::X, n, 0);
                self.a.csneg(Sz::X, d, n, n, Cond::Ge);
            }
            DivMod {
                div,
                sz,
                nonzero,
                minus1,
                exit,
            } => self.divmod(i, div, sz, nonzero, minus1, exit),
            ShiftI32 { left, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.sxtw(X16, n);
                self.a.sxtw(X17, m);
                self.lua_shift(X16, X16, X17, left);
                self.a.cmp_ext(X16, X16, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov_w(d, X16);
            }
            ShiftI64 { left } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.lua_shift(X16, n, m, left);
                self.a.mov(d, X16);
            }
            Sxtw => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.sxtw(d, n);
            }
            Fcvtzs(sz) => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fcvtzs(sz, d, n);
            }
            Scvtf(sz) => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                self.a.scvtf(sz, d, n);
            }
            FmovToGpr => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fmov_to_gpr(d, n);
            }
            FmovFromGpr => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                self.a.fmov_from_gpr(d, n);
            }
            LToI { exit } => {
                let n = self.r(i, 0);
                self.a.cmp_ext(n, n, Extend::Sxtw);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
            }
            FToIExact { exit } => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                self.a.fcvtzs(Sz::W, X16, n);
                self.a.scvtf(Sz::W, D31, X16);
                self.a.fcmp(n, D31);
                let l = self.exit(exit);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov_w(d, X16);
            }
            FToLExact { exit } => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                let l = self.exit(exit);
                // `fcvtzs` saturates 2^63 to i64::MAX, which converts back to 2^63.
                self.a.fcvtzs(Sz::X, X16, n);
                self.a.cmn_imm(Sz::X, X16, 1);
                self.a.b_cond(Cond::Vs, l);
                self.a.scvtf(Sz::X, D31, X16);
                self.a.fcmp(n, D31);
                self.a.b_cond(Cond::Ne, l);
                self.a.mov(d, X16);
            }
            // `sdiv` gives `MIN / -1 = MIN` without trapping, so `MIN % -1` is 0.
            Rem { sz, nonzero, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                if !nonzero {
                    let l = self.exit(exit);
                    self.a.cbz(sz, m, l);
                }
                self.a.sdiv(sz, X16, n, m);
                self.a.msub(sz, d, X16, m, n);
            }
            StrCmp { le } => {
                use crate::env::string::LuaString;
                let (d, a, b) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, k, lb, ca, cb) = (
                    self.r(i, 3),
                    self.r(i, 4),
                    self.r(i, 5),
                    self.r(i, 6),
                    self.r(i, 7),
                );
                let (top, tail, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                self.a.ldr(d, a, LuaString::LEN_OFFSET as i32);
                self.a.ldr(lb, b, LuaString::LEN_OFFSET as i32);
                self.a.cmp(Sz::X, d, lb);
                self.a.csel(Sz::X, n, d, lb, Cond::Lo);
                self.a.add_imm(Sz::X, X16, a, LuaString::DATA_OFFSET as u64);
                self.a.add_imm(Sz::X, X17, b, LuaString::DATA_OFFSET as u64);
                self.a.movz(Sz::X, k, 0, 0);
                self.a.bind(top);
                self.a.cmp(Sz::X, k, n);
                self.a.b_cond(Cond::Hs, tail);
                self.a.ldrb_reg(ca, X16, k);
                self.a.ldrb_reg(cb, X17, k);
                self.a.add_imm(Sz::X, k, k, 1);
                self.a.cmp(Sz::W, ca, cb);
                self.a.b_cond(Cond::Eq, top);
                // The first differing byte decides.
                self.a.cset(Sz::W, d, Cond::Lo);
                self.a.b(done);
                // A common prefix: the shorter orders first.
                self.a.bind(tail);
                self.a.cmp(Sz::X, d, lb);
                self.a.cset(Sz::W, d, if le { Cond::Ls } else { Cond::Lo });
                self.a.bind(done);
            }
            // A float inline, the other kinds out of line.
            ToF64 { exit, kbox, kinds } => {
                let (d, v) = (self.fr(i, 0), self.r(i, 1));
                let (other, done) = (self.a.new_label(), self.a.new_label());
                if !kinds.contains(TypeSet::FLOAT) {
                    self.a.b(other);
                } else if kbox {
                    let k = self.r(i, 2);
                    self.a.cmp(Sz::X, v, k);
                    self.a.b_cond(Cond::Hs, other);
                    self.a.fmov_from_gpr(d, v);
                } else {
                    self.a.mov_imm(X16, BOX);
                    self.a.cmp(Sz::X, v, X16);
                    self.a.b_cond(Cond::Hs, other);
                    self.a.fmov_from_gpr(d, v);
                }
                self.a.bind(done);
                let l = self.exit(exit);
                let tramp = (kinds.contains(TypeSet::STR)).then(|| {
                    self.links
                        .trampoline_of(crate::jit::helpers::jit_str_to_f64 as *const () as usize)
                });
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(other);
                    if kinds.contains(TypeSet::SMALL) {
                        let next = em.a.new_label();
                        em.a.lsr_imm(Sz::X, X16, v, 32);
                        em.a.cmn_imm(Sz::W, X16, 1);
                        em.a.b_cond(Cond::Ne, next);
                        em.a.scvtf(Sz::W, d, v);
                        em.a.b(done);
                        em.a.bind(next);
                    }
                    // The tag, sign-extended: a big integer's is -6, a string's -5.
                    if kinds.contains(TypeSet::BIGINT) {
                        let next = em.a.new_label();
                        em.a.asr_imm(Sz::X, X16, v, 48);
                        em.a.cmn_imm(Sz::X, X16, 6);
                        em.a.b_cond(Cond::Ne, next);
                        em.a.and_imm(Sz::X, X16, v, PTR_MASK);
                        em.a.ldr(X16, X16, 0);
                        em.a.scvtf(Sz::X, d, X16);
                        em.a.b(done);
                        em.a.bind(next);
                    }
                    if let Some(t) = tramp {
                        em.a.asr_imm(Sz::X, X16, v, 48);
                        em.a.cmn_imm(Sz::X, X16, 5);
                        em.a.b_cond(Cond::Ne, l);
                        // The trampoline keeps every register but x0, its
                        // result: the bits of the f64, or the failure
                        // (`HELPER_FAIL_VAL`, a NaN no conversion makes).
                        em.a.sub_imm(Sz::X, SP, SP, 16);
                        em.a.str(Gpr(0), SP, 0);
                        em.a.mov(Gpr(0), v);
                        em.a.bl_far(t);
                        em.a.mov(X16, Gpr(0));
                        em.a.ldr(Gpr(0), SP, 0);
                        em.a.add_imm(Sz::X, SP, SP, 16);
                        let fail = crate::jit::ir::ops::HELPER_FAIL_VAL;
                        em.a.movz(Sz::X, X17, (fail >> 48) as u16, 3);
                        em.a.cmp(Sz::X, X16, X17);
                        em.a.b_cond(Cond::Eq, l);
                        em.a.fmov_from_gpr(d, X16);
                        em.a.b(done);
                    } else {
                        em.a.b(l);
                    }
                }));
            }
            BoxI32 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.orr_imm(Sz::X, d, n, SMALL_BITS);
            }
            BoxB1 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                // false + c
                self.a.mov_imm(X16, FALSE);
                self.a.add_ext(d, X16, n, Extend::Uxtw, 0);
            }
            BoxI64 { exit } => {
                // x1 holds the value; the result goes to x0.
                let done = self.a.new_label();
                let slow = self.a.new_label();
                self.a.cmp_ext(Gpr(1), Gpr(1), Extend::Sxtw);
                self.a.b_cond(Cond::Ne, slow);
                self.a.orr_imm(Sz::X, Gpr(0), Gpr(1), SMALL_BITS);
                self.a.b(done);
                self.a.bind(slow);
                self.a.ldr_lit(Gpr(0), self.links.rt);
                self.a.bl_far(
                    self.links
                        .trampoline_of(addr!(crate::jit::helpers::jit_box_i64)),
                );
                if exit != u32::MAX {
                    self.gc_check(exit);
                }
                self.a.bind(done);
            }
            UnboxI64 => {
                let (d, v) = (self.r(i, 0), self.r(i, 1));
                self.unbox_int(d, v);
            }
            FMod => {
                // r = a - floor(a / b) * b. A rounded quotient whose floor is
                // off leaves r outside (0, b), as do a NaN, an infinity, a
                // zero or negative divisor and a zero remainder.
                let (d, q, x, y) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2), self.fr(i, 3));
                let (slow, back) = (self.a.new_label(), self.a.new_label());
                self.a.fdiv(q, x, y);
                self.a.frintm(q, q);
                self.a.fmsub(d, q, y, x);
                self.a.fcmp_zero(d);
                self.a.fccmp(d, y, 0, Cond::Gt);
                self.a.b_cond(Cond::Ge, slow);
                self.a.bind(back);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.fmod_slow(slow, back, d, x, y);
                }));
            }
            FAlu(op) => {
                let (d, n, m) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2));
                match op {
                    FOp::Add => self.a.fadd(d, n, m),
                    FOp::Sub => self.a.fsub(d, n, m),
                    FOp::Mul => self.a.fmul(d, n, m),
                    FOp::Div => self.a.fdiv(d, n, m),
                }
            }
            FUn(op) => {
                let (d, n) = (self.fr(i, 0), self.fr(i, 1));
                match op {
                    FUnOp::Neg => self.a.fneg(d, n),
                    FUnOp::Abs => self.a.fabs(d, n),
                    FUnOp::Sqrt => self.a.fsqrt(d, n),
                    FUnOp::Floor => self.a.frintm(d, n),
                    FUnOp::Ceil => self.a.frintp(d, n),
                }
            }
            Set { test, cond } => {
                let d = self.r(i, 0);
                if let Test::Type(set, words) = test
                    && !is_single_test(set)
                {
                    // `d` may share the tested value's register: written last.
                    let v = self.r(i, 1);
                    let w = self.words(i, 1, set, words);
                    let (yes, done) = (self.a.new_label(), self.a.new_label());
                    type_branch(&mut self.a, v, set, yes, &w);
                    self.a.movz(Sz::W, d, (cond != Cond::Eq) as u16, 0);
                    self.a.b(done);
                    self.a.bind(yes);
                    self.a.movz(Sz::W, d, (cond == Cond::Eq) as u16, 0);
                    self.a.bind(done);
                } else {
                    let c = self.test(i, 1, test, cond);
                    self.a.cset(Sz::W, d, c);
                }
            }
            Select { test, cond, float } => {
                let c = self.test(i, 3, test, cond);
                if float {
                    let (d, x, y) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2));
                    self.a.fcsel(d, x, y, c);
                } else {
                    let (d, x, y) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                    self.a.csel(Sz::X, d, x, y, c);
                }
            }
            Guard { test, cond, exit } => {
                let l = self.exit(exit);
                if let Test::Bit(b) | Test::MmBit(b) = test
                    && self.bit_branches
                {
                    // Passes where `cond` holds (`Ne`: the bit set).
                    let pass = self.a.new_label();
                    let x = match test {
                        Test::MmBit(_) => {
                            let t = self.r(i, 0);
                            self.class_bits(X16, t, if cond == Cond::Ne { l } else { pass });
                            X16
                        }
                        _ => self.r(i, 0),
                    };
                    if cond == Cond::Ne {
                        self.a.tbz(x, b as u32, l);
                    } else {
                        self.a.tbnz(x, b as u32, l);
                    }
                    self.a.bind(pass);
                } else if let Some(sz) = zero_test(test, cond) {
                    // Passes where `cond` holds: out where it does not.
                    let c = self.r(i, 0);
                    if cond == Cond::Ne {
                        self.a.cbz(sz, c, l);
                    } else {
                        self.a.cbnz(sz, c, l);
                    }
                } else if let Test::Type(set, words) = test
                    && !is_single_test(set)
                {
                    let v = self.r(i, 0);
                    let w = self.words(i, 0, set, words);
                    if cond == Cond::Eq {
                        type_fail(&mut self.a, v, set, l, &w);
                    } else {
                        type_branch(&mut self.a, v, set, l, &w);
                    }
                } else {
                    let c = self.test(i, 0, test, cond);
                    self.a.b_cond(c.invert(), l);
                }
            }
            GuardLanding { set, exit } => match self.pending_based.take() {
                // The fast return skips it: a self-call returns what its
                // returns give. A level returning to the call (its callee's
                // `BASE`, `jit_exit`) and the slow path land on it.
                Some((r, hdr)) if r == block => {
                    let (cont, boxed) = (self.a.new_label(), self.a.new_label());
                    self.resume[block].boxed = Some(boxed);
                    self.ool.push(Box::new(move |em: &mut Em<'_>| {
                        em.a.bind(boxed);
                        em.a.sub_imm(Sz::X, BASE, BASE, hdr as u64 * 8 + 32);
                        em.a.bind(em.resume[block].based);
                        em.type_guard(i, set, exit);
                        em.a.b(cont);
                    }));
                    self.a.bind(cont);
                }
                _ => self.type_guard(i, set, exit),
            },
            ClassMm(k) => {
                // Shape, class, the class's other metamethods: each may be
                // missing (null), and then so is the metamethod.
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let (none, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a
                    .ldr(X16, X16, crate::env::Shape::MT_CACHE_OFFSET as i32);
                self.a.cbz(Sz::X, X16, none);
                self.a.ldr(X16, X16, MtCache::REST_OFFSET as i32);
                self.a.cbz(Sz::X, X16, none);
                self.a.ldr(
                    d,
                    X16,
                    (MtCache::REST_VALUES_OFFSET + 8 * k as usize) as i32,
                );
                self.a.b(done);
                self.a.bind(none);
                self.a.mov_imm(d, NIL);
                self.a.bind(done);
            }
            StrLenOf => {
                let (d, s) = (self.r(i, 0), self.r(i, 1));
                self.a
                    .ldr(d, s, crate::env::string::LuaString::LEN_OFFSET as i32);
            }
            // A small integer inline, the other kinds out of line, clamped.
            ToPos { exit, kinds } => {
                let (d, v, t) = (self.r(i, 0), self.r(i, 1), self.fr(i, 2));
                let (other, done) = (self.a.new_label(), self.a.new_label());
                if kinds.contains(TypeSet::SMALL) {
                    self.a.lsr_imm(Sz::X, X16, v, 32);
                    self.a.cmn_imm(Sz::W, X16, 1);
                    self.a.b_cond(Cond::Ne, other);
                    self.a.mov_w(d, v);
                } else {
                    self.a.b(other);
                }
                self.a.bind(done);
                let l = self.exit(exit);
                let tramp = (kinds.contains(TypeSet::STR)).then(|| {
                    self.links
                        .trampoline_of(crate::jit::helpers::jit_str_to_i64 as *const () as usize)
                });
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(other);
                    // Each arm leaves the i64 in x16 for the clamp.
                    let clamp = em.a.new_label();
                    // The tag, sign-extended: a big integer's is -6, a string's -5.
                    if kinds.contains(TypeSet::BIGINT) {
                        let next = em.a.new_label();
                        em.a.asr_imm(Sz::X, X16, v, 48);
                        em.a.cmn_imm(Sz::X, X16, 6);
                        em.a.b_cond(Cond::Ne, next);
                        em.a.and_imm(Sz::X, X16, v, PTR_MASK);
                        em.a.ldr(X16, X16, 0);
                        em.a.b(clamp);
                        em.a.bind(next);
                    }
                    // `fcvtzs` saturates out of range to MAX, which no f64 is, or to
                    // a MIN that does not convert back (-2^63 itself does).
                    if kinds.contains(TypeSet::FLOAT) {
                        let next = em.a.new_label();
                        em.a.mov_imm(X16, BOX);
                        em.a.cmp(Sz::X, v, X16);
                        em.a.b_cond(Cond::Hs, next);
                        em.a.fmov_from_gpr(t, v);
                        em.a.fcvtzs(Sz::X, X16, t);
                        em.a.cmn_imm(Sz::X, X16, 1);
                        em.a.b_cond(Cond::Vs, l);
                        em.a.scvtf(Sz::X, D31, X16);
                        em.a.fcmp(D31, t);
                        em.a.b_cond(Cond::Ne, l);
                        em.a.b(clamp);
                        em.a.bind(next);
                    }
                    if let Some(tramp) = tramp {
                        em.a.asr_imm(Sz::X, X16, v, 48);
                        em.a.cmn_imm(Sz::X, X16, 5);
                        em.a.b_cond(Cond::Ne, l);
                        // The trampoline keeps every register but x0, its result
                        // (converted or not); the integer goes to a slot above it.
                        em.a.sub_imm(Sz::X, SP, SP, 32);
                        em.a.stp(Gpr(0), Gpr(1), SP, 0);
                        em.a.mov(Gpr(0), v);
                        em.a.add_imm(Sz::X, Gpr(1), SP, 16);
                        em.a.bl_far(tramp);
                        em.a.mov(X17, Gpr(0));
                        em.a.ldr(X16, SP, 16);
                        em.a.ldp(Gpr(0), Gpr(1), SP, 0);
                        em.a.add_imm(Sz::X, SP, SP, 32);
                        em.a.cbz(Sz::X, X17, l);
                    } else {
                        em.a.b(l);
                    }
                    // x16 to i32, saturating: past it, the sign's extreme.
                    em.a.bind(clamp);
                    let fits = em.a.new_label();
                    em.a.cmp_ext(X16, X16, Extend::Sxtw);
                    em.a.b_cond(Cond::Eq, fits);
                    em.a.asr_imm(Sz::X, X17, X16, 63);
                    em.a.eor_imm(Sz::W, X16, X17, 0x7fff_ffff);
                    em.a.bind(fits);
                    em.a.mov_w(d, X16);
                    em.a.b(done);
                }));
            }
            IFCmp { cc } => {
                let (d, i, f, t) = (self.r(i, 0), self.r(i, 1), self.fr(i, 2), self.fr(i, 3));
                let (slow, done) = (self.a.new_label(), self.a.new_label());
                self.if_fast(d, i, f, t, cc, slow);
                self.a.bind(done);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(slow);
                    em.if_big(d, i, f, t, cc, done);
                    em.a.b(done);
                }));
            }
            NumCmp { cc, kx, ky } => {
                let (d, x, y) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (fx, fy, ix, iy) = (self.fr(i, 3), self.fr(i, 4), self.r(i, 5), self.r(i, 6));
                let float = |k: TypeSet| k.contains(TypeSet::FLOAT);
                let int = |k: TypeSet| k.intersects(TypeSet::INT);
                let (x_int, xf_y_int, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                // Two floats inline; `x` an integer, or `y` one, out of line.
                if !float(kx) {
                    self.a.b(x_int);
                } else {
                    if int(kx) {
                        self.is_float_or(x, x_int);
                    }
                    if !float(ky) {
                        self.a.b(xf_y_int);
                    } else {
                        if int(ky) {
                            self.is_float_or(y, xf_y_int);
                        }
                        self.a.fmov_from_gpr(fx, x);
                        self.a.fmov_from_gpr(fy, y);
                        self.a.fcmp(fx, fy);
                        self.a.cset(Sz::W, d, float_cond(cc));
                    }
                }
                self.a.bind(done);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    if int(kx) {
                        em.a.bind(x_int);
                        em.unbox_int(ix, x);
                        let xi_yf = em.a.new_label();
                        if !int(ky) {
                            em.a.b(xi_yf);
                        } else {
                            if float(ky) {
                                em.is_float_or_not(y, xi_yf);
                            }
                            em.unbox_int(iy, y);
                            em.a.cmp(Sz::X, ix, iy);
                            em.a.cset(Sz::W, d, int_cond(cc));
                            em.a.b(done);
                        }
                        em.a.bind(xi_yf);
                        if float(ky) {
                            em.a.fmov_from_gpr(fy, y);
                            em.if_both(d, ix, fy, fx, cc);
                        }
                        em.a.b(done);
                    }
                    if float(kx) && int(ky) {
                        em.a.bind(xf_y_int);
                        em.unbox_int(iy, y);
                        em.a.fmov_from_gpr(fx, x);
                        em.if_both(d, iy, fx, fy, cc.swap());
                        em.a.b(done);
                    }
                }));
            }
            StrByteAt { exit } => {
                use crate::env::string::LuaString;
                let (d, s, p) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let l = self.exit(exit);
                // k = p, or len + p + 1 for a negative p; in range when
                // k - 1 is below len, unsigned (both below 2^31: i32 suffices).
                self.a.ldr(X16, s, LuaString::LEN_OFFSET as i32);
                self.a.add(Sz::W, X17, X16, p);
                self.a.cmp_imm(Sz::W, p, 0);
                self.a.csinc(Sz::W, X17, p, X17, Cond::Ge);
                self.a.sub_imm(Sz::W, X17, X17, 1);
                self.a.cmp(Sz::W, X17, X16);
                self.a.b_cond(Cond::Hs, l);
                self.a.add(Sz::X, X17, s, X17);
                self.a.ldrb(d, X17, LuaString::DATA_OFFSET as i32);
            }
            LoadCell => {
                let (d, c) = (self.reg(i, 0), self.r(i, 1));
                self.a
                    .ldr(X16, c, crate::env::function::UpvalueCell::V_OFFSET as i32);
                self.word_at(false, d, X16, 0);
            }
            GuardProto { proto, fun, exit } => {
                let v = self.r(i, 0);
                let l = self.exit(exit);
                if !fun {
                    self.a.lsr_imm(Sz::X, X16, v, 48);
                    self.a.movz(Sz::W, X17, FUNCTION_TAG, 0);
                    self.a.cmp(Sz::W, X16, X17);
                    self.a.b_cond(Cond::Ne, l);
                }
                // A native's bytes past its variant may be anything: the
                // tag byte first (`FN_IS_VARARG`, 0 for a fixed Lua closure).
                self.a.and_imm(Sz::X, X16, v, PTR_MASK);
                self.a.ldrb(X17, X16, layout::FN_IS_VARARG as i32);
                self.a.cbnz(Sz::W, X17, l);
                self.a.ldr(X16, X16, layout::FN_PROTO as i32);
                self.a.ldr_lit(X17, proto);
                self.a.cmp(Sz::X, X16, X17);
                self.a.b_cond(Cond::Ne, l);
            }
            ClassBits(class) => {
                let d = self.r(i, 0);
                self.a.ldr_lit(X16, class);
                self.a.ldr_w(d, X16, MtCache::BITS_OFFSET as i32);
            }
            ClassIndex(class) => {
                let d = self.r(i, 0);
                self.a.ldr_lit(X16, class);
                self.a.ldr(d, X16, MtCache::INDEX_OFFSET as i32);
            }
            TabLoad(loc) | TabStore(loc) => {
                let store = matches!(inst, TabStore(_));
                let (t, v) = (self.r(i, !store as usize), self.reg(i, store as usize));
                let loc = SlotLoc::from_raw(loc);
                if loc.spilled() {
                    self.a.ldr(X16, t, layout::T_SPILL as i32);
                    self.word_at(store, v, X16, loc.offset() as i32);
                } else {
                    self.word_at(store, v, t, loc.offset() as i32);
                }
            }
            TabSetShape(shape) => {
                let t = self.r(i, 0);
                self.a.ldr_lit(X16, shape);
                self.a.str(X16, t, layout::T_SHAPE as i32);
            }
            TabASize => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.ldr_w(d, t, layout::T_ASIZE as i32);
            }
            TabArray => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.ldr(d, t, layout::T_ARRAY as i32);
            }
            ArrLoad => {
                let (d, a, k) = (self.reg(i, 0), self.r(i, 1), self.r(i, 2));
                self.word_idx(false, d, a, k);
            }
            ArrStore => {
                let (a, k, v) = (self.r(i, 0), self.r(i, 1), self.reg(i, 2));
                self.word_idx(true, v, a, k);
            }
            ArrLoadAt(k) => {
                let (d, a) = (self.reg(i, 0), self.r(i, 1));
                self.word_at(false, d, a, k as i32 * 8);
            }
            ArrStoreAt(k) => {
                let (a, v) = (self.r(i, 0), self.reg(i, 1));
                self.word_at(true, v, a, k as i32 * 8);
            }
            NamedNext => {
                let (d, t, from) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, cap, mt) = (self.r(i, 3), self.r(i, 4), self.r(i, 5));
                let [top, spilled, check, step, none, done] = [(); 6].map(|_| self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr_w(n, X16, Shape::SLOT_COUNT_OFFSET as i32);
                self.a.ldrb(cap, X16, Shape::INLINE_CAP_OFFSET as i32);
                self.a.ldr_w(mt, X16, Shape::MT_SLOT_OFFSET as i32);
                self.a.mov_w(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::W, d, n);
                self.a.b_cond(Cond::Hs, none);
                self.a.cmp(Sz::W, d, mt);
                self.a.b_cond(Cond::Eq, step);
                self.a.cmp(Sz::W, d, cap);
                self.a.b_cond(Cond::Hs, spilled);
                self.a.add_imm(Sz::X, X17, t, layout::T_INLINE as u64);
                self.a.ldr_idx(X16, X17, d, Extend::Uxtw, true);
                self.a.b(check);
                self.a.bind(spilled);
                self.a.ldr(X17, t, layout::T_SPILL as i32);
                self.a.sub(Sz::W, X16, d, cap);
                self.a.ldr_idx(X16, X17, X16, Extend::Uxtw, true);
                self.a.bind(check);
                self.a.eor_imm(Sz::X, X16, X16, NIL);
                self.a.cbnz(Sz::X, X16, done);
                self.a.bind(step);
                self.a.add_imm(Sz::W, d, d, 1);
                self.a.b(top);
                self.a.bind(none);
                self.a.movn(Sz::W, d, 0, 0);
                self.a.bind(done);
            }
            NamedVal => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (spilled, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldrb(X16, X16, Shape::INLINE_CAP_OFFSET as i32);
                self.a.cmp(Sz::W, k, X16);
                self.a.b_cond(Cond::Hs, spilled);
                self.a.add_imm(Sz::X, X17, t, layout::T_INLINE as u64);
                self.a.ldr_idx(d, X17, k, Extend::Uxtw, true);
                self.a.b(done);
                self.a.bind(spilled);
                self.a.sub(Sz::W, X16, k, X16);
                self.a.ldr(X17, t, layout::T_SPILL as i32);
                self.a.ldr_idx(d, X17, X16, Extend::Uxtw, true);
                self.a.bind(done);
            }
            NamedKey => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr(X16, X16, Shape::KEYS_OFFSET as i32);
                self.a.ldr(X16, X16, Shape::KEYS_SLOTS_OFFSET as i32);
                self.a.ldr_idx(X16, X16, k, Extend::Uxtw, true);
                self.a.add_imm(
                    Sz::X,
                    X16,
                    X16,
                    crate::env::string::LuaString::VALUE_OFFSET as u64,
                );
                self.a.ldr_lit(X17, crate::env::value::STRING_BITS);
                self.a.orr(Sz::X, d, X16, X17);
            }
            ArrNext => {
                let (d, t, from, v) = (self.r(i, 0), self.r(i, 1), self.r(i, 2), self.r(i, 3));
                let (top, done) = (self.a.new_label(), self.a.new_label());
                self.a.ldr_w(X16, t, layout::T_ASIZE as i32);
                self.a.ldr(X17, t, layout::T_ARRAY as i32);
                self.a.mov_w(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::W, d, X16);
                self.a.b_cond(Cond::Hs, done);
                self.a.ldr_idx(v, X17, d, Extend::Uxtw, true);
                // Nil is one run of ones, a logical immediate.
                self.a.eor_imm(Sz::X, v, v, NIL);
                self.a.cbnz(Sz::X, v, done);
                self.a.add_imm(Sz::W, d, d, 1);
                self.a.b(top);
                self.a.bind(done);
            }
            // The slow path, out of line, keeps every register itself: x0
            // and x1 across its own argument setup, the rest in the
            // trampoline.
            Barrier => {
                let t = self.r(i, 0);
                let (slow, back) = (self.a.new_label(), self.a.new_label());
                self.a.ldr(X16, t, layout::T_HEADER);
                self.a.tbz(X16, 0, slow);
                self.a.bind(back);
                let tramp = self
                    .links
                    .trampoline_of(crate::jit::helpers::jit_barrier as *const () as usize);
                let rt = self.links.rt;
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(slow);
                    em.a.stp_pre(Gpr(0), Gpr(1), SP, -16);
                    if t != Gpr(1) {
                        em.a.mov(Gpr(1), t);
                    }
                    em.a.ldr_lit(Gpr(0), rt);
                    em.a.bl_far(tramp);
                    em.a.ldp_post(Gpr(0), Gpr(1), SP, 16);
                    em.a.b(back);
                }));
            }
            NumVarargs => {
                let d = self.r(i, 0);
                self.a.ldr(X16, BASE, -32);
                self.a.lsr_imm(Sz::X, d, X16, 48);
            }
            VarargAt(k) => {
                // The extras end at the header: extra k is at
                // base - 32 - 8 * nv + 8 * k, read only when k < nv.
                let (d, nv) = (self.r(i, 0), self.r(i, 1));
                self.a.cmp_imm(Sz::W, nv, k as u64);
                self.a.lsl_imm(Sz::W, X16, nv, 3);
                self.a.sub(Sz::X, X16, BASE, X16);
                self.a.ldr(d, X16, 8 * k as i32 - 32);
                self.a.mov_imm(X17, NIL);
                self.a.csel(Sz::X, d, d, X17, Cond::Hi);
            }
            VarargIdx => {
                // The index clamped to 0 out of range, so the load stays in
                // the frame.
                let (d, nv, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.sub_imm(Sz::W, X16, k, 1);
                self.a.cmp(Sz::W, X16, nv);
                self.a.csel(Sz::W, X16, X16, ZR, Cond::Lo);
                self.a.lsl_imm(Sz::W, X17, nv, 3);
                self.a.sub(Sz::X, X17, BASE, X17);
                self.a.add_shift(Sz::X, X17, X17, X16, Shift::Lsl, 3);
                self.a.ldr(d, X17, -32);
                self.a.mov_imm(X16, NIL);
                self.a.csel(Sz::X, d, d, X16, Cond::Lo);
            }
            StoreCell => {
                let (c, v) = (self.r(i, 0), self.reg(i, 1));
                self.a
                    .ldr(X16, c, crate::env::function::UpvalueCell::V_OFFSET as i32);
                self.word_at(true, v, X16, 0);
            }
            GcCheck { exit } => self.gc_check(exit),
            Helper(h) | HelperCold(h) => {
                let sig = h.sig();
                for (k, c) in sig.ctx.iter().enumerate() {
                    match c {
                        Ctx::Rt => self.a.ldr_lit(Gpr(k as u8), self.links.rt),
                        Ctx::Thread => self.a.mov(Gpr(k as u8), THREAD),
                        Ctx::Base => self.a.mov(Gpr(k as u8), BASE),
                    }
                }
                if let Some(imm) = sig.imm {
                    self.a.mov_imm(Gpr(sig.ctx.len() as u8), imm);
                }
                let target = crate::jit::helpers::address(h);
                if matches!(inst, HelperCold(_)) {
                    self.a.bl_far(self.links.trampoline_of(target));
                } else {
                    self.a.bl_far(target);
                }
            }
            Jump => {
                let t = self.dest[self.v.succs(block)[0].index()];
                if Some(t) != next {
                    self.a.b(t);
                }
            }
            Br { test, cond } => {
                let ([lt, lf], inline) = self.edge_labels(i, block, next);
                self.br(i, test, cond, lt, lf, next);
                for (_, _, e) in inline {
                    self.edit(e);
                }
            }
            Call {
                a,
                nargs,
                pc_after,
                known,
                typed,
            } => self.call(i, block, next, a, nargs, pc_after, known, typed),
            Dispatch { op, word, pc_after } => self.dispatch(op, word, pc_after),
            TailEnter { nargs, known } => self.tail_enter(i, nargs, known),
            SetTop { at } => {
                let n = self.r(i, 0);
                self.set_top(n, at as u64 * 8);
            }
            Resume { wanted, c, a, hdr } => self.on_resume(i, block, wanted, c, a, hdr),
            CallMm {
                hdr,
                nargs,
                pc_after,
            } => self.call_mm(i, block, next, hdr, nargs, pc_after),
            Return { a, n, x0 } => self.ret(a, n, x0),
            Deopt { exit } => {
                let l = self.exit(exit);
                self.a.b(l);
            }
            LazyExit { exit } => {
                let (at, l) = (self.a.new_label(), self.exit(exit));
                self.a.patch_point(at);
                self.lazy.push((at, l));
            }
        }
        Ok(())
    }

    /// `FMod`'s out-of-line path from `slow` back to `back`, `d` holding
    /// the inline remainder of `x` by `y`.
    fn fmod_slow(&mut self, slow: Label, back: Label, d: Fpr, x: Fpr, y: Fpr) {
        // A zero remainder is exact: fmod's zero has the dividend's sign.
        let call = self.a.new_label();
        self.a.bind(slow);
        self.a.fcmp_zero(d);
        self.a.b_cond(Cond::Ne, call);
        self.a.fmov_to_gpr(X16, x);
        self.a.and_imm(Sz::X, X16, X16, 1 << 63);
        self.a.fmov_from_gpr(d, X16);
        self.a.b(back);
        self.a.bind(call);
        let t = self.links.trampoline_of(crate::jit::helpers::address(
            crate::jit::ir::ops::HelperId::FMod,
        ));
        self.a.fmov_to_gpr(X16, x);
        self.a.fmov_to_gpr(X17, y);
        self.a.sub_imm(Sz::X, SP, SP, 16);
        self.a.stp_d(Fpr(0), Fpr(1), SP, 0);
        self.a.fmov_from_gpr(Fpr(0), X16);
        self.a.fmov_from_gpr(Fpr(1), X17);
        self.a.bl_far(t);
        self.a.fmov_to_gpr(X16, Fpr(0));
        self.a.ldp_d(Fpr(0), Fpr(1), SP, 0);
        self.a.add_imm(Sz::X, SP, SP, 16);
        self.a.fmov_from_gpr(d, X16);
        self.a.b(back);
    }

    /// The thread's `top` past `n` values from `off` bytes above `BASE`.
    pub(super) fn set_top(&mut self, n: Gpr, off: u64) {
        self.a.ldr(X16, THREAD, layout::THREAD_STACK as i32);
        self.a.add_imm(Sz::X, X17, BASE, off);
        self.a.sub(Sz::X, X17, X17, X16);
        self.a.add_shift(Sz::X, X17, n, X17, Shift::Lsr, 3);
        self.a.str(X17, THREAD, layout::THREAD_TOP as i32);
    }

    /// The frame's header takes `CLOSURE` for its closure, the vararg
    /// count kept.
    pub(super) fn set_frame_closure(&mut self) {
        self.a.ldr(X16, BASE, -32);
        self.a.and_imm(Sz::X, X16, X16, !crate::vm::frame::PTR_MASK);
        self.a.orr(Sz::X, X16, X16, CLOSURE);
        self.a.str(X16, BASE, -32);
    }

    pub(super) fn gc_check(&mut self, exit: u32) {
        self.a.ldr_lit(X16, self.links.metrics);
        self.a
            .ldp(X16, X17, X16, crate::jit::layout::GC_CHECK as i32);
        self.a.cmp(Sz::X, X16, X17);
        let l = self.exit(exit);
        self.a.b_cond(Cond::Hs, l);
    }

    /// Lua's shift: `n << m` (or `>>`) on 64 bits, logical, 0 past 63.
    pub(super) fn lua_shift(&mut self, d: Gpr, n: Gpr, m: Gpr, left: bool) {
        // A right shift is a left shift by `-m`.
        let cnt = X17;
        if left {
            self.a.mov(cnt, m);
        } else {
            self.a.neg(Sz::X, cnt, m);
        }
        let neg = self.a.new_label();
        let zero = self.a.new_label();
        let done = self.a.new_label();
        self.a.cmp_imm(Sz::X, cnt, 0);
        self.a.b_cond(Cond::Lt, neg);
        self.a.cmp_imm(Sz::X, cnt, 64);
        self.a.b_cond(Cond::Ge, zero);
        self.a.lslv(Sz::X, d, n, cnt);
        self.a.b(done);
        self.a.bind(neg);
        // Unsigned: `-i64::MIN` is still negative.
        self.a.neg(Sz::X, cnt, cnt);
        self.a.cmp_imm(Sz::X, cnt, 64);
        self.a.b_cond(Cond::Hs, zero);
        self.a.lsrv(Sz::X, d, n, cnt);
        self.a.b(done);
        self.a.bind(zero);
        self.a.mov(d, ZR);
        self.a.bind(done);
    }

    /// `IFCmp`'s inline part: `d = i cc f` as floats while `|i| <= 2^53`
    /// converts exactly (`i + 2^53` below 2^54, unsigned; 2^53 itself goes
    /// to `slow`, which is exact too), else on to `slow`.
    pub(super) fn if_fast(&mut self, d: Gpr, i: Gpr, f: Fpr, t: Fpr, cc: Cc, slow: Label) {
        self.a.movz(Sz::X, X16, 0x20, 3);
        self.a.add(Sz::X, X16, i, X16);
        self.a.lsr_imm(Sz::X, X16, X16, 54);
        self.a.cbnz(Sz::X, X16, slow);
        self.a.scvtf(Sz::X, t, i);
        self.a.fcmp(t, f);
        self.a.cset(Sz::W, d, float_cond(cc));
    }

    /// `IFCmp` past ±2^53, as Lua's `LTintfloat` and kin: a float whose
    /// floor or ceiling is an i64 compares by it (`i < ceil f`, `i <= floor
    /// f`, `floor f < i` for `Gt`, `ceil f <= i` for `Ge`, `i == floor f`,
    /// exact as a non-integral float is below 2^53 in magnitude); a NaN is
    /// false; any other float is above or below every integer by its sign.
    pub(super) fn if_big(&mut self, d: Gpr, i: Gpr, f: Fpr, t: Fpr, cc: Cc, done: Label) {
        let (no, huge) = (self.a.new_label(), self.a.new_label());
        self.a.fcmp(f, f);
        self.a.b_cond(Cond::Vs, no);
        match cc {
            Cc::Lt | Cc::Ge => self.a.frintp(t, f),
            _ => self.a.frintm(t, f),
        }
        // Out of i64 when `fcvtzs` saturates: to MAX, which no f64 is, or to
        // a MIN that does not convert back (-2^63 itself does).
        self.a.fcvtzs(Sz::X, X16, t);
        self.a.cmn_imm(Sz::X, X16, 1);
        self.a.b_cond(Cond::Vs, huge);
        self.a.scvtf(Sz::X, D31, X16);
        self.a.fcmp(D31, t);
        self.a.b_cond(Cond::Ne, huge);
        self.a.cmp(Sz::X, i, X16);
        self.a.cset(Sz::W, d, int_cond(cc));
        self.a.b(done);
        self.a.bind(huge);
        match cc {
            Cc::Lt | Cc::Le => {
                self.a.fcmp_zero(f);
                self.a.cset(Sz::W, d, Cond::Gt);
                self.a.b(done);
            }
            Cc::Gt | Cc::Ge => {
                self.a.fcmp_zero(f);
                self.a.cset(Sz::W, d, Cond::Mi);
                self.a.b(done);
            }
            Cc::Eq => {}
            _ => unreachable!("an int-float compare is ordered"),
        }
        self.a.bind(no);
        self.a.movz(Sz::W, d, 0, 0);
    }

    /// `if_fast` and `if_big` in place, for code already out of line.
    pub(super) fn if_both(&mut self, d: Gpr, i: Gpr, f: Fpr, t: Fpr, cc: Cc) {
        let (slow, done) = (self.a.new_label(), self.a.new_label());
        self.if_fast(d, i, f, t, cc, slow);
        self.a.b(done);
        self.a.bind(slow);
        self.if_big(d, i, f, t, cc, done);
        self.a.bind(done);
    }

    /// Branch to `other` unless number `v` is a float.
    pub(super) fn is_float_or(&mut self, v: Gpr, other: Label) {
        self.a.mov_imm(X16, BOX);
        self.a.cmp(Sz::X, v, X16);
        self.a.b_cond(Cond::Hs, other);
    }

    /// Branch to `float` if number `v` is a float.
    pub(super) fn is_float_or_not(&mut self, v: Gpr, float: Label) {
        self.a.mov_imm(X16, BOX);
        self.a.cmp(Sz::X, v, X16);
        self.a.b_cond(Cond::Lo, float);
    }

    /// `d` = the i64 of integer `v`, small or boxed.
    pub(super) fn unbox_int(&mut self, d: Gpr, v: Gpr) {
        let (big, done) = (self.a.new_label(), self.a.new_label());
        self.a.lsr_imm(Sz::X, X16, v, 32);
        self.a.cmn_imm(Sz::W, X16, 1);
        self.a.b_cond(Cond::Ne, big);
        self.a.sxtw(d, v);
        self.a.b(done);
        self.a.bind(big);
        self.a.and_imm(Sz::X, X16, v, PTR_MASK);
        self.a.ldr(d, X16, 0);
        self.a.bind(done);
    }

    /// Floor division or modulo of operand 1 by operand 2.
    pub(super) fn divmod(
        &mut self,
        i: usize,
        div: bool,
        sz: Sz,
        nonzero: bool,
        minus1: bool,
        exit: u32,
    ) {
        let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
        let l = self.exit(exit);
        if !nonzero {
            self.a.cbz(sz, m, l);
        }
        if div && sz == Sz::W && minus1 {
            // MIN // -1 leaves i32.
            let ok = self.a.new_label();
            self.a.cmn_imm(Sz::W, m, 1);
            self.a.b_cond(Cond::Ne, ok);
            self.a.movz(Sz::W, X16, 0x8000, 1);
            self.a.cmp(Sz::W, n, X16);
            self.a.b_cond(Cond::Eq, l);
            self.a.bind(ok);
        }
        let (q, r) = (X16, X17);
        self.a.sdiv(sz, q, n, m);
        self.a.msub(sz, r, q, m, n);
        if div {
            // q - 1 when the remainder is nonzero and the signs differ.
            let done = self.a.new_label();
            self.a.cbz(sz, r, done);
            self.a.eor(sz, r, n, m);
            self.a.tbz(r, sz.bits() - 1, done);
            self.a.sub_imm(sz, q, q, 1);
            self.a.bind(done);
            self.a.mov(d, q);
        } else {
            // r + m when the remainder is nonzero and its sign differs from m's.
            let done = self.a.new_label();
            self.a.cbz(sz, r, done);
            self.a.eor(sz, q, r, m);
            self.a.tbz(q, sz.bits() - 1, done);
            self.a.add(sz, r, r, m);
            self.a.bind(done);
            self.a.mov(d, r);
        }
    }

    /// A value word loaded or stored at `[n, #off]`: from an x register, or
    /// from a d register holding a float's (`Lower::in_d`).
    pub(super) fn word_at(&mut self, store: bool, t: PReg, n: Gpr, off: i32) {
        match (t.class(), store) {
            (RegClass::Int, false) => self.a.ldr(gpr(t), n, off),
            (RegClass::Int, true) => self.a.str(gpr(t), n, off),
            (_, false) => self.a.ldr_d(fpr(t), n, off),
            (_, true) => self.a.str_d(fpr(t), n, off),
        }
    }

    /// The same at `[n, wm, uxtw #3]`.
    pub(super) fn word_idx(&mut self, store: bool, t: PReg, n: Gpr, m: Gpr) {
        match (t.class(), store) {
            (RegClass::Int, false) => self.a.ldr_idx(gpr(t), n, m, Extend::Uxtw, true),
            (RegClass::Int, true) => self.a.str_idx(gpr(t), n, m, Extend::Uxtw, true),
            (_, false) => self.a.ldr_d_idx(fpr(t), n, m, Extend::Uxtw, true),
            (_, true) => self.a.str_d_idx(fpr(t), n, m, Extend::Uxtw, true),
        }
    }
}
