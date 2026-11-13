//! Each instruction's sequence, with the tests branches, guards, selects
//! and sets share.

use crate::env::table::SlotLoc;
use crate::env::value::{BOX, FALSE, NIL, PAYLOAD_MASK as PTR_MASK, SMALL_INT as SMALL_BITS};
use crate::env::{MtCache, Shape};
use crate::jit::backend::x86_64::abi::{ARGS, BASE, CLOSURE, FS, S0, S1, THREAD};
use crate::jit::backend::x86_64::asm::{
    Arith, Cond, FBin, FBit, Gpr, Label, Mem, RAX, RDI, RDX, RSI, RSP, Shift, Sz, Xmm,
};
use crate::jit::backend::x86_64::emit::Em;
use crate::jit::backend::x86_64::inst::{AluOp, FCmp, FOp, FUnOp, MInst, Test};
use crate::jit::backend::x86_64::lower::{float_test, int_cond};
use crate::jit::backend::x86_64::types::{is_single_test, type_branch, type_fail, type_flags};
use crate::jit::ir::ops::{Cc, Ctx};
use crate::jit::ir::types::TypeSet;
use crate::jit::layout;
use crate::jit::layout::FUNCTION_TAG;

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
        if let Test::Type(set) = test
            && !is_single_test(set)
        {
            // Where `cond` holds, and where it does not.
            let (yes, no) = if cond == Cond::E { (lt, lf) } else { (lf, lt) };
            let v = self.r(i, 0);
            if Some(yes) == next {
                type_fail(&mut self.a, v, set, no);
                return;
            }
            type_branch(&mut self.a, v, set, yes);
            if Some(no) != next {
                self.a.jmp(no);
            }
            return;
        }
        let c = self.test(i, 0, test, cond);
        if Some(lt) == next {
            self.a.jcc(c.invert(), lf);
        } else {
            self.a.jcc(c, lt);
            if Some(lf) != next {
                self.a.jmp(lf);
            }
        }
    }

    /// `d` = the metamethod bits of table `t`'s class, or on to `none`
    /// without a class, `d` zero there.
    pub(super) fn class_bits(&mut self, d: Gpr, t: Gpr, none: Label) {
        self.a.load(Sz::Q, d, Mem::B(t, layout::T_SHAPE as i32));
        self.a
            .load(Sz::Q, d, Mem::B(d, Shape::MT_CACHE_OFFSET as i32));
        self.a.test(Sz::Q, d, d);
        self.a.jcc(Cond::E, none);
        self.a
            .load(Sz::D, d, Mem::B(d, MtCache::BITS_OFFSET as i32));
    }

    /// Flags `E` when table `t`'s shape is one of `shape_sets[set]`: one
    /// load, then a compare per shape, each after the first only while none
    /// matched.
    pub(super) fn shape_in(&mut self, t: Gpr, set: u32) {
        let hit = self.a.new_label();
        self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
        let words = &self.l.shape_sets[set as usize];
        for (n, &word) in words.iter().enumerate() {
            self.a.cmp(Sz::Q, S0, Mem::K(word));
            if n + 1 < words.len() {
                self.a.jcc(Cond::E, hit);
            }
        }
        self.a.bind(hit);
    }

    /// Flags for a compare of `x` with `y` made as `how` says.
    fn float_flags(&mut self, how: FCmp, x: Xmm, y: Xmm) {
        match how {
            FCmp::Ord => self.a.fcmp(x, y),
            FCmp::Rev => self.a.fcmp(y, x),
            // The mask is a NaN or zero, so `PF` is the result. Scalar: a
            // register's upper lane holds anything (a packed literal's
            // neighbour, a call's result).
            FCmp::Eq => {
                self.a.fcmp_eq(FS, x, y);
                self.a.fcmp(FS, FS);
            }
        }
    }

    /// `d = x cc y` of floats.
    fn float_set(&mut self, d: Gpr, cc: Cc, x: Xmm, y: Xmm) {
        let (how, cond) = float_test(cc);
        self.float_flags(how, x, y);
        self.a.setcc(cond, d);
    }

    /// Set the flags for `test` on instruction `i`'s operands from `k`;
    /// returns the condition that means the test holds, given `cond` (the
    /// IR's condition for compares, `E` or `Ne` for type tests).
    pub(super) fn test(&mut self, i: usize, k: usize, test: Test, cond: Cond) -> Cond {
        match test {
            Test::I32 { imm: Some(0) } | Test::B1 => {
                let x = self.r(i, k);
                self.a.test(Sz::D, x, x);
            }
            Test::I32 { imm: Some(n) } => {
                let x = self.r(i, k);
                self.a.cmp_imm(Sz::D, x, n);
            }
            Test::I32 { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::D, x, y);
            }
            Test::X { imm: Some(0) } => {
                let x = self.r(i, k);
                self.a.test(Sz::Q, x, x);
            }
            Test::X { imm: Some(n) } => {
                let x = self.r(i, k);
                self.a.cmp_imm(Sz::Q, x, n);
            }
            Test::X { imm: None } => {
                let (x, y) = (self.r(i, k), self.r(i, k + 1));
                self.a.cmp(Sz::Q, x, y);
            }
            Test::F64 { zero, how } => {
                let x = self.fr(i, k);
                let y = if zero {
                    self.a.fzero(FS);
                    FS
                } else {
                    self.fr(i, k + 1)
                };
                self.float_flags(how, x, y);
            }
            Test::Bit(b) => {
                let x = self.r(i, k);
                self.a.test_imm(Sz::D, x, 1 << b);
            }
            Test::MmBit(b) => {
                let t = self.r(i, k);
                let done = self.a.new_label();
                self.class_bits(S0, t, done);
                self.a.bind(done);
                self.a.test_imm(Sz::D, S0, 1 << b);
            }
            Test::HasFlag(mask) => self.a.test8_imm(Mem::B(BASE, -24), mask),
            Test::NumMtSet => {
                self.a.mov_imm(
                    S0,
                    self.links.type_mt_cells[layout::NUMBER_MT_SLOT as usize],
                );
                self.a
                    .cmp_imm(Sz::Q, Mem::B(S0, layout::NUMBER_MT as i32), 0);
            }
            Test::TypeMtIs { slot, word } => {
                self.a.mov_imm(S0, self.links.type_mt_cells[slot as usize]);
                self.a.load(Sz::Q, S0, Mem::B(S0, layout::NUMBER_MT as i32));
                self.a.cmp(Sz::Q, S0, Mem::K(word));
            }
            // 0 and 1 for a Lua closure's flag, past them a native's niche.
            Test::FnFixedLua | Test::FnNative => {
                let f = self.r(i, k);
                self.a.cmp8_imm(
                    Mem::B(f, layout::FN_IS_VARARG as i32),
                    (test == Test::FnNative) as u8,
                );
            }
            Test::FnFits => {
                let f = self.r(i, k);
                self.a.load8(S0, Mem::B(f, layout::FN_MAX_STACK as i32));
                self.a.lea(Sz::Q, S0, Mem::X(BASE, S0, 3, 0));
                self.a
                    .cmp(Sz::Q, S0, Mem::B(THREAD, layout::THREAD_STACK_END as i32));
            }
            Test::TabHasRoom(loc) => {
                let t = self.r(i, k);
                let slot = SlotLoc::from_raw(loc).offset() / 8;
                self.a
                    .cmp_imm(Sz::D, Mem::B(t, layout::T_SPILL_CAP as i32), slot as i32);
            }
            // No cell, or the parts' counts or'd: zero.
            Test::TabNoKeys(parts) => {
                let t = self.r(i, k);
                let done = self.a.new_label();
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_AUX as i32));
                self.a.test(Sz::Q, S0, S0);
                self.a.jcc(Cond::E, done);
                let mut offs = layout::aux_counts(parts);
                let first = offs.next().expect("a part");
                self.a.load(Sz::Q, S1, Mem::B(S0, first as i32));
                for off in offs {
                    self.a.alu(Arith::Or, Sz::Q, S1, Mem::B(S0, off as i32));
                }
                self.a.mov(S0, S1);
                self.a.bind(done);
                self.a.test(Sz::Q, S0, S0);
            }
            Test::Shape(word) => {
                let t = self.r(i, k);
                self.a.mov_imm(S0, word);
                self.a.cmp(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
            }
            Test::ShapeIn(set) => {
                let t = self.r(i, k);
                self.shape_in(t, set);
            }
            Test::Type(set) => {
                let v = self.r(i, k);
                let c = type_flags(&mut self.a, v, set);
                return if cond == Cond::E { c } else { c.invert() };
            }
        }
        cond
    }

    /// Exit unless instruction `i`'s operand 0 is in `set`.
    pub(super) fn type_guard(&mut self, i: usize, set: TypeSet, exit: u32) {
        let (l, v) = (self.exit(exit), self.r(i, 0));
        type_fail(&mut self.a, v, set, l);
    }

    /// The tag of `v`'s word, sign-extended (a big integer's is -6, a
    /// string's -5), against `tag`: flags `E` when it is that.
    fn tag_is(&mut self, v: Gpr, tag: i32) {
        self.a.mov(S0, v);
        self.a.shift_imm(Shift::Sar, Sz::Q, S0, 48);
        self.a.cmp_imm(Sz::D, S0, tag);
    }

    /// Flags `E` when `v` is a small integer.
    fn is_small(&mut self, v: Gpr) {
        self.a.mov(S0, v);
        self.a.shift_imm(Shift::Shr, Sz::Q, S0, 32);
        self.a.cmp_imm(Sz::D, S0, -1);
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
            self.a.mov_imm(S0, (fired + 4 * k as usize) as u64);
            self.a.cmp_imm(Sz::D, Mem::B(S0, 0), 0);
            self.a.jcc(Cond::E, l);
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
                self.word_at(false, d, Mem::B(BASE, r as i32 * 8));
            }
            StoreSlot(r) => {
                let v = self.reg(i, 0);
                self.word_at(true, v, Mem::B(BASE, r as i32 * 8));
            }
            FrameClosure => {
                let d = self.r(i, 0);
                self.a.load(Sz::Q, d, Mem::B(BASE, -32));
                self.a.alu(Arith::And, Sz::Q, d, Mem::K(PTR_MASK));
            }
            LoadUpvalOf(k) => {
                let (d, c) = (self.reg(i, 0), self.r(i, 1));
                let off = crate::jit::compile::upvalues_offset() + k as usize * 8;
                self.word_at(false, d, Mem::B(c, off as i32));
            }
            Add(sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                if d == n {
                    self.a.alu(Arith::Add, sz, d, m);
                } else if d == m {
                    self.a.alu(Arith::Add, sz, d, n);
                } else {
                    self.a.lea(sz, d, Mem::X(n, m, 0, 0));
                }
            }
            AddImm(sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                if d == n {
                    self.a.alu_imm(Arith::Add, sz, d, k);
                } else {
                    self.a.lea(sz, d, Mem::B(n, k));
                }
            }
            Alu(op, sz) => {
                let (d, m) = (self.r(i, 0), self.r(i, 2));
                debug_assert_eq!(d, self.r(i, 1));
                match op {
                    AluOp::Sub => self.a.alu(Arith::Sub, sz, d, m),
                    AluOp::And => self.a.alu(Arith::And, sz, d, m),
                    AluOp::Or => self.a.alu(Arith::Or, sz, d, m),
                    AluOp::Xor => self.a.alu(Arith::Xor, sz, d, m),
                    AluOp::Mul => self.a.imul(sz, d, m),
                    AluOp::Add => return Err("an add as a two-address instruction".into()),
                }
            }
            AluImm(op, sz, k) => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                let op = match op {
                    AluOp::And => Arith::And,
                    AluOp::Or => Arith::Or,
                    AluOp::Xor => Arith::Xor,
                    _ => return Err(format!("{op:?} with an immediate")),
                };
                self.a.mov(d, n);
                match (sz, i32::try_from(k as i64)) {
                    (Sz::D, _) => self.a.alu_imm(op, sz, d, k as u32 as i32),
                    (Sz::Q, Ok(k)) => self.a.alu_imm(op, sz, d, k),
                    (Sz::Q, Err(_)) => self.a.alu(op, sz, d, Mem::K(k)),
                }
            }
            ShiftImm(sh, sz, k) => {
                let d = self.r(i, 0);
                debug_assert_eq!(d, self.r(i, 1));
                self.a.shift_imm(sh, sz, d, k);
            }
            ShiftReg(sh, sz) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.shiftx(sh, sz, d, n, m);
            }
            UMulShr(k) => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.mov32(S0, m);
                self.a.mov32(S1, n);
                self.a.imul(Sz::Q, S1, S0);
                self.a.shift_imm(Shift::Shr, Sz::Q, S1, k);
                self.a.mov(d, S1);
            }
            Neg(sz) => {
                let d = self.r(i, 0);
                self.a.neg(sz, d);
            }
            Not(sz) => {
                let d = self.r(i, 0);
                self.a.not(sz, d);
            }
            AddOvf { sub, exit } => {
                let (d, m) = (self.r(i, 0), self.r(i, 2));
                debug_assert_eq!(d, self.r(i, 1));
                self.a
                    .alu(if sub { Arith::Sub } else { Arith::Add }, Sz::D, d, m);
                let l = self.exit(exit);
                self.a.jcc(Cond::O, l);
            }
            AddImmOvf { imm, exit } => {
                let d = self.r(i, 0);
                debug_assert_eq!(d, self.r(i, 1));
                self.a.alu_imm(Arith::Add, Sz::D, d, imm);
                let l = self.exit(exit);
                self.a.jcc(Cond::O, l);
            }
            MulOvf { exit } => {
                let (d, m) = (self.r(i, 0), self.r(i, 2));
                debug_assert_eq!(d, self.r(i, 1));
                self.a.imul(Sz::D, d, m);
                let l = self.exit(exit);
                self.a.jcc(Cond::O, l);
            }
            NegOvf { exit } => {
                let d = self.r(i, 0);
                self.a.neg(Sz::D, d);
                let l = self.exit(exit);
                self.a.jcc(Cond::O, l);
            }
            // The negation where it is not negative: of `MIN` it is, as
            // `MIN` then is, which has no i32 absolute value.
            AbsOvf { exit } => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.mov32(S0, n);
                self.a.neg(Sz::D, S0);
                self.a.cmov(Cond::S, Sz::D, S0, n);
                self.a.test(Sz::D, S0, S0);
                let l = self.exit(exit);
                self.a.jcc(Cond::S, l);
                self.a.mov32(d, S0);
            }
            Abs => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.mov(S0, n);
                self.a.neg(Sz::Q, S0);
                self.a.cmov(Cond::S, Sz::Q, S0, n);
                self.a.mov(d, S0);
            }
            DivMod {
                div,
                sz,
                nonzero,
                minus1,
                exit,
            } => self.divmod(i, div, sz, nonzero, minus1, exit),
            // `MIN % -1` is 0, around `idiv`.
            Rem { sz, nonzero, exit } => {
                let m = self.r(i, 2);
                if !nonzero {
                    let l = self.exit(exit);
                    self.a.test(sz, m, m);
                    self.a.jcc(Cond::E, l);
                }
                let (m1, back) = (self.a.new_label(), self.a.new_label());
                self.a.cmp_imm(sz, m, -1);
                self.a.jcc(Cond::E, m1);
                self.a.sign_rdx(sz);
                self.a.idiv(sz, m);
                self.a.bind(back);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(m1);
                    em.a.mov_imm(RDX, 0);
                    em.a.jmp(back);
                }));
            }
            UDiv => {
                let m = self.r(i, 2);
                self.a.alu(Arith::Xor, Sz::D, RDX, RDX);
                self.a.div(Sz::Q, m);
            }
            ShiftI32 { left, exit } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.movsxd(S0, n);
                self.a.movsxd(S1, m);
                self.lua_shift(S0, S1, left);
                self.a.movsxd(S1, S0);
                self.a.cmp(Sz::Q, S1, S0);
                let l = self.exit(exit);
                self.a.jcc(Cond::Ne, l);
                self.a.mov32(d, S0);
            }
            ShiftI64 { left } => {
                let (d, n, m) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.lua_shift(n, m, left);
                self.a.mov(d, S0);
            }
            Sext => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.movsxd(d, n);
            }
            IntToF(sz) => {
                let (d, n) = (self.fr(i, 0), self.r(i, 1));
                self.a.cvt_from_int(sz, d, n);
            }
            // The conversion gives the minimum out of range and for a NaN,
            // the one value whose decrement overflows.
            FToInt => {
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                let (fix, back) = (self.a.new_label(), self.a.new_label());
                self.a.cvt_to_int(Sz::Q, d, n);
                self.a.cmp_imm(Sz::Q, d, 1);
                self.a.jcc(Cond::O, fix);
                self.a.bind(back);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    let nan = em.a.new_label();
                    em.a.bind(fix);
                    em.a.fcmp(n, n);
                    em.a.jcc(Cond::P, nan);
                    em.a.fzero(FS);
                    em.a.fcmp(n, FS);
                    em.a.jcc(Cond::Be, back);
                    em.a.mov_imm(d, i64::MAX as u64);
                    em.a.jmp(back);
                    em.a.bind(nan);
                    em.a.mov_imm(d, 0);
                    em.a.jmp(back);
                }));
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
                self.a.movsxd(S0, n);
                self.a.cmp(Sz::Q, S0, n);
                let l = self.exit(exit);
                self.a.jcc(Cond::Ne, l);
            }
            // Out of range converts to the minimum, which converts back
            // to itself alone.
            FToIExact { exit } | FToLExact { exit } => {
                let sz = if matches!(inst, FToIExact { .. }) {
                    Sz::D
                } else {
                    Sz::Q
                };
                let (d, n) = (self.r(i, 0), self.fr(i, 1));
                let l = self.exit(exit);
                self.exact_int(sz, n, l);
                if sz == Sz::D {
                    self.a.mov32(d, S0);
                } else {
                    self.a.mov(d, S0);
                }
            }
            StrCmp { le } => {
                use crate::env::string::LuaString;
                let (d, a, b) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, k, lb) = (self.r(i, 3), self.r(i, 4), self.r(i, 5));
                let (top, tail, done) =
                    (self.a.new_label(), self.a.new_label(), self.a.new_label());
                let data = LuaString::DATA_OFFSET as i32;
                self.a
                    .load(Sz::Q, d, Mem::B(a, LuaString::LEN_OFFSET as i32));
                self.a
                    .load(Sz::Q, lb, Mem::B(b, LuaString::LEN_OFFSET as i32));
                self.a.mov(n, d);
                self.a.cmp(Sz::Q, d, lb);
                self.a.cmov(Cond::A, Sz::Q, n, lb);
                self.a.mov_imm(k, 0);
                self.a.bind(top);
                self.a.cmp(Sz::Q, k, n);
                self.a.jcc(Cond::Ae, tail);
                self.a.load8(S0, Mem::X(a, k, 0, data));
                self.a.load8(S1, Mem::X(b, k, 0, data));
                self.a.alu_imm(Arith::Add, Sz::Q, k, 1);
                self.a.cmp(Sz::D, S0, S1);
                self.a.jcc(Cond::E, top);
                // The first differing byte decides.
                self.a.setcc(Cond::B, d);
                self.a.jmp(done);
                // A common prefix: the shorter orders first.
                self.a.bind(tail);
                self.a.cmp(Sz::Q, d, lb);
                self.a.setcc(if le { Cond::Be } else { Cond::B }, d);
                self.a.bind(done);
            }
            // A float inline, the other kinds out of line.
            ToF64 { exit, kinds } => {
                let (d, v) = (self.fr(i, 0), self.r(i, 1));
                let (other, done) = (self.a.new_label(), self.a.new_label());
                if !kinds.contains(TypeSet::FLOAT) {
                    self.a.jmp(other);
                } else {
                    self.a.cmp(Sz::Q, v, Mem::K(BOX));
                    self.a.jcc(Cond::Ae, other);
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
                        em.is_small(v);
                        em.a.jcc(Cond::Ne, next);
                        em.a.cvt_from_int(Sz::D, d, v);
                        em.a.jmp(done);
                        em.a.bind(next);
                    }
                    if kinds.contains(TypeSet::BIGINT) {
                        let next = em.a.new_label();
                        em.tag_is(v, -6);
                        em.a.jcc(Cond::Ne, next);
                        em.a.mov(S0, v);
                        em.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
                        em.a.load(Sz::Q, S0, Mem::B(S0, 0));
                        em.a.cvt_from_int(Sz::Q, d, S0);
                        em.a.jmp(done);
                        em.a.bind(next);
                    }
                    if let Some(t) = tramp {
                        em.tag_is(v, -5);
                        em.a.jcc(Cond::Ne, l);
                        // The trampoline keeps every register but rax, its
                        // result: the bits of the f64, or the failure
                        // (`HELPER_FAIL_VAL`, a NaN no conversion makes).
                        em.a.alu_imm(Arith::Sub, Sz::Q, RSP, 16);
                        em.a.store(Sz::Q, Mem::B(RSP, 0), RAX);
                        em.a.store(Sz::Q, Mem::B(RSP, 8), RDI);
                        em.a.mov(RDI, v);
                        em.a.call_far(t);
                        em.a.mov(S0, RAX);
                        em.a.load(Sz::Q, RAX, Mem::B(RSP, 0));
                        em.a.load(Sz::Q, RDI, Mem::B(RSP, 8));
                        em.a.alu_imm(Arith::Add, Sz::Q, RSP, 16);
                        em.a.cmp(Sz::Q, S0, Mem::K(crate::jit::ir::ops::HELPER_FAIL_VAL));
                        em.a.jcc(Cond::E, l);
                        em.a.fmov_from_gpr(d, S0);
                        em.a.jmp(done);
                    } else {
                        em.a.jmp(l);
                    }
                }));
            }
            // The upper half may hold anything: all of it is set.
            BoxI32 => {
                let d = self.r(i, 0);
                self.a.alu(Arith::Or, Sz::Q, d, Mem::K(SMALL_BITS));
            }
            // false + c
            BoxB1 => {
                let (d, n) = (self.r(i, 0), self.r(i, 1));
                self.a.mov32(S0, n);
                self.a.mov_imm(d, FALSE);
                self.a.alu(Arith::Add, Sz::Q, d, S0);
            }
            BoxI64 { exit } => {
                // rsi holds the value; the result goes to rax.
                let (done, slow) = (self.a.new_label(), self.a.new_label());
                self.a.movsxd(S0, RSI);
                self.a.cmp(Sz::Q, S0, RSI);
                self.a.jcc(Cond::Ne, slow);
                self.a.mov(RAX, RSI);
                self.a.alu(Arith::Or, Sz::Q, RAX, Mem::K(SMALL_BITS));
                self.a.jmp(done);
                // The trampoline keeps every register but rax.
                self.a.bind(slow);
                self.a.alu_imm(Arith::Sub, Sz::Q, RSP, 16);
                self.a.store(Sz::Q, Mem::B(RSP, 0), RDI);
                self.a.mov_imm(RDI, self.links.rt);
                self.a.call_far(
                    self.links
                        .trampoline_of(addr!(crate::jit::helpers::jit_box_i64)),
                );
                self.a.load(Sz::Q, RDI, Mem::B(RSP, 0));
                self.a.alu_imm(Arith::Add, Sz::Q, RSP, 16);
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
                self.a.fbin(FBin::Div, q, x, y);
                self.a.fround(q, q, 1);
                self.a.fmov(d, x);
                self.a.fnmadd(d, q, y);
                self.a.fzero(FS);
                self.a.fcmp(d, FS);
                self.a.jcc(Cond::Be, slow);
                self.a.fcmp(d, y);
                self.a.jcc(Cond::Ae, slow);
                self.a.bind(back);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.fmod_slow(slow, back, d, x, y);
                }));
            }
            FAlu(op) => {
                let (d, n, m) = (self.fr(i, 0), self.fr(i, 1), self.fr(i, 2));
                let op = match op {
                    FOp::Add => FBin::Add,
                    FOp::Sub => FBin::Sub,
                    FOp::Mul => FBin::Mul,
                    FOp::Div => FBin::Div,
                };
                self.a.fbin(op, d, n, m);
            }
            FUn(op) => {
                let (d, n) = (self.fr(i, 0), self.fr(i, 1));
                match op {
                    FUnOp::Neg => self.a.fbit(FBit::Xor, d, n, Mem::K2(1 << 63)),
                    FUnOp::Abs => self.a.fbit(FBit::And, d, n, Mem::K2(!(1 << 63))),
                    FUnOp::Sqrt => self.a.fbin(FBin::Sqrt, d, n, n),
                    FUnOp::Floor => self.a.fround(d, n, 1),
                    FUnOp::Ceil => self.a.fround(d, n, 2),
                }
            }
            Set { test, cond } => {
                let d = self.r(i, 0);
                if let Test::Type(set) = test
                    && !is_single_test(set)
                {
                    // `d` may share the tested value's register: written last.
                    let v = self.r(i, 1);
                    let (yes, done) = (self.a.new_label(), self.a.new_label());
                    type_branch(&mut self.a, v, set, yes);
                    self.a.mov_imm(d, (cond != Cond::E) as u64);
                    self.a.jmp(done);
                    self.a.bind(yes);
                    self.a.mov_imm(d, (cond == Cond::E) as u64);
                    self.a.bind(done);
                } else {
                    let c = self.test(i, 1, test, cond);
                    self.a.setcc(c, d);
                }
            }
            Select { test, cond, float } => {
                let c = self.test(i, 3, test, cond);
                if float {
                    let (d, x) = (self.fr(i, 0), self.fr(i, 1));
                    debug_assert_eq!(d, self.fr(i, 2));
                    let keep = self.a.new_label();
                    self.a.jcc(c.invert(), keep);
                    self.a.fmov(d, x);
                    self.a.bind(keep);
                } else {
                    let (d, x) = (self.r(i, 0), self.r(i, 1));
                    debug_assert_eq!(d, self.r(i, 2));
                    self.a.cmov(c, Sz::Q, d, x);
                }
            }
            Guard { test, cond, exit } => {
                let l = self.exit(exit);
                if let Test::Type(set) = test
                    && !is_single_test(set)
                {
                    let v = self.r(i, 0);
                    if cond == Cond::E {
                        type_fail(&mut self.a, v, set, l);
                    } else {
                        type_branch(&mut self.a, v, set, l);
                    }
                } else {
                    let c = self.test(i, 0, test, cond);
                    self.a.jcc(c.invert(), l);
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
                        em.a.alu_imm(Arith::Sub, Sz::Q, BASE, hdr as i32 * 8 + 32);
                        em.a.bind(em.resume[block].based);
                        em.type_guard(i, set, exit);
                        em.a.jmp(cont);
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
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
                self.a
                    .load(Sz::Q, S0, Mem::B(S0, Shape::MT_CACHE_OFFSET as i32));
                self.a.test(Sz::Q, S0, S0);
                self.a.jcc(Cond::E, none);
                self.a
                    .load(Sz::Q, S0, Mem::B(S0, MtCache::REST_OFFSET as i32));
                self.a.test(Sz::Q, S0, S0);
                self.a.jcc(Cond::E, none);
                let off = MtCache::REST_VALUES_OFFSET + 8 * k as usize;
                self.a.load(Sz::Q, d, Mem::B(S0, off as i32));
                self.a.jmp(done);
                self.a.bind(none);
                self.a.mov_imm(d, NIL);
                self.a.bind(done);
            }
            StrLenOf => {
                let (d, s) = (self.r(i, 0), self.r(i, 1));
                let off = crate::env::string::LuaString::LEN_OFFSET as i32;
                self.a.load(Sz::Q, d, Mem::B(s, off));
            }
            // A small integer inline, the other kinds out of line, clamped.
            ToPos { exit, kinds } => {
                let (d, v, t) = (self.r(i, 0), self.r(i, 1), self.fr(i, 2));
                let (other, done) = (self.a.new_label(), self.a.new_label());
                if kinds.contains(TypeSet::SMALL) {
                    self.is_small(v);
                    self.a.jcc(Cond::Ne, other);
                    self.a.mov32(d, v);
                } else {
                    self.a.jmp(other);
                }
                self.a.bind(done);
                let l = self.exit(exit);
                let tramp = (kinds.contains(TypeSet::STR)).then(|| {
                    self.links
                        .trampoline_of(crate::jit::helpers::jit_str_to_i64 as *const () as usize)
                });
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(other);
                    // Each arm leaves the i64 in `S0` for the clamp.
                    let clamp = em.a.new_label();
                    if kinds.contains(TypeSet::BIGINT) {
                        let next = em.a.new_label();
                        em.tag_is(v, -6);
                        em.a.jcc(Cond::Ne, next);
                        em.a.mov(S0, v);
                        em.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
                        em.a.load(Sz::Q, S0, Mem::B(S0, 0));
                        em.a.jmp(clamp);
                        em.a.bind(next);
                    }
                    if kinds.contains(TypeSet::FLOAT) {
                        let next = em.a.new_label();
                        em.a.cmp(Sz::Q, v, Mem::K(BOX));
                        em.a.jcc(Cond::Ae, next);
                        em.a.fmov_from_gpr(t, v);
                        em.exact_int(Sz::Q, t, l);
                        em.a.jmp(clamp);
                        em.a.bind(next);
                    }
                    if let Some(tramp) = tramp {
                        em.tag_is(v, -5);
                        em.a.jcc(Cond::Ne, l);
                        // The trampoline keeps every register but rax, its
                        // result (converted or not); the integer goes to a
                        // slot above the registers kept here.
                        em.a.alu_imm(Arith::Sub, Sz::Q, RSP, 32);
                        em.a.store(Sz::Q, Mem::B(RSP, 0), RAX);
                        em.a.store(Sz::Q, Mem::B(RSP, 8), RDI);
                        em.a.store(Sz::Q, Mem::B(RSP, 16), RSI);
                        em.a.mov(RDI, v);
                        em.a.lea(Sz::Q, RSI, Mem::B(RSP, 24));
                        em.a.call_far(tramp);
                        em.a.mov(S1, RAX);
                        em.a.load(Sz::Q, S0, Mem::B(RSP, 24));
                        em.a.load(Sz::Q, RAX, Mem::B(RSP, 0));
                        em.a.load(Sz::Q, RDI, Mem::B(RSP, 8));
                        em.a.load(Sz::Q, RSI, Mem::B(RSP, 16));
                        em.a.alu_imm(Arith::Add, Sz::Q, RSP, 32);
                        em.a.test(Sz::Q, S1, S1);
                        em.a.jcc(Cond::E, l);
                    } else {
                        em.a.jmp(l);
                    }
                    // `S0` to i32, saturating: past it, the sign's extreme.
                    em.a.bind(clamp);
                    let fits = em.a.new_label();
                    em.a.movsxd(S1, S0);
                    em.a.cmp(Sz::Q, S1, S0);
                    em.a.jcc(Cond::E, fits);
                    em.a.shift_imm(Shift::Sar, Sz::Q, S0, 63);
                    em.a.alu_imm(Arith::Xor, Sz::D, S0, 0x7fff_ffff);
                    em.a.bind(fits);
                    em.a.mov32(d, S0);
                    em.a.jmp(done);
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
                    em.a.jmp(done);
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
                    self.a.jmp(x_int);
                } else {
                    if int(kx) {
                        self.a.cmp(Sz::Q, x, Mem::K(BOX));
                        self.a.jcc(Cond::Ae, x_int);
                    }
                    if !float(ky) {
                        self.a.jmp(xf_y_int);
                    } else {
                        if int(ky) {
                            self.a.cmp(Sz::Q, y, Mem::K(BOX));
                            self.a.jcc(Cond::Ae, xf_y_int);
                        }
                        self.a.fmov_from_gpr(fx, x);
                        self.a.fmov_from_gpr(fy, y);
                        self.float_set(d, cc, fx, fy);
                    }
                }
                self.a.bind(done);
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    if int(kx) {
                        em.a.bind(x_int);
                        em.unbox_int(ix, x);
                        let xi_yf = em.a.new_label();
                        if !int(ky) {
                            em.a.jmp(xi_yf);
                        } else {
                            if float(ky) {
                                em.a.cmp(Sz::Q, y, Mem::K(BOX));
                                em.a.jcc(Cond::B, xi_yf);
                            }
                            em.unbox_int(iy, y);
                            em.a.cmp(Sz::Q, ix, iy);
                            em.a.setcc(int_cond(cc), d);
                            em.a.jmp(done);
                        }
                        em.a.bind(xi_yf);
                        if float(ky) {
                            em.a.fmov_from_gpr(fy, y);
                            em.if_both(d, ix, fy, fx, cc);
                        }
                        em.a.jmp(done);
                    }
                    if float(kx) && int(ky) {
                        em.a.bind(xf_y_int);
                        em.unbox_int(iy, y);
                        em.a.fmov_from_gpr(fx, x);
                        em.if_both(d, iy, fx, fy, cc.swap());
                        em.a.jmp(done);
                    }
                }));
            }
            StrByteAt { exit } => {
                use crate::env::string::LuaString;
                let (d, s, p) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let l = self.exit(exit);
                // k = p, or len + p + 1 for a negative p; in range when
                // k - 1 is below len, unsigned (both below 2^31: i32 suffices).
                self.a
                    .load(Sz::Q, S0, Mem::B(s, LuaString::LEN_OFFSET as i32));
                self.a.lea(Sz::D, S1, Mem::X(S0, p, 0, 1));
                self.a.test(Sz::D, p, p);
                self.a.cmov(Cond::Ns, Sz::D, S1, p);
                self.a.alu_imm(Arith::Sub, Sz::D, S1, 1);
                self.a.cmp(Sz::D, S1, S0);
                self.a.jcc(Cond::Ae, l);
                self.a
                    .load8(d, Mem::X(s, S1, 0, LuaString::DATA_OFFSET as i32));
            }
            LoadCell => {
                let (d, c) = (self.reg(i, 0), self.r(i, 1));
                let off = crate::env::function::UpvalueCell::V_OFFSET as i32;
                self.a.load(Sz::Q, S0, Mem::B(c, off));
                self.word_at(false, d, Mem::B(S0, 0));
            }
            GuardProto { proto, fun, exit } => {
                let v = self.r(i, 0);
                let l = self.exit(exit);
                if !fun {
                    self.a.mov(S0, v);
                    self.a.shift_imm(Shift::Shr, Sz::Q, S0, 48);
                    self.a.cmp_imm(Sz::D, S0, FUNCTION_TAG as i32);
                    self.a.jcc(Cond::Ne, l);
                }
                // A native's bytes past its variant may be anything: the
                // tag byte first (`FN_IS_VARARG`, 0 for a fixed Lua closure).
                self.a.mov(S0, v);
                self.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
                self.a.cmp8_imm(Mem::B(S0, layout::FN_IS_VARARG as i32), 0);
                self.a.jcc(Cond::Ne, l);
                self.a.mov_imm(S1, proto);
                self.a.cmp(Sz::Q, S1, Mem::B(S0, layout::FN_PROTO as i32));
                self.a.jcc(Cond::Ne, l);
            }
            ClassBits(class) => {
                let d = self.r(i, 0);
                self.a.mov_imm(S0, class);
                self.a
                    .load(Sz::D, d, Mem::B(S0, MtCache::BITS_OFFSET as i32));
            }
            ClassIndex(class) => {
                let d = self.r(i, 0);
                self.a.mov_imm(S0, class);
                self.a
                    .load(Sz::Q, d, Mem::B(S0, MtCache::INDEX_OFFSET as i32));
            }
            TabLoad(loc) | TabStore(loc) => {
                let store = matches!(inst, TabStore(_));
                let (t, v) = (self.r(i, !store as usize), self.reg(i, store as usize));
                let loc = SlotLoc::from_raw(loc);
                if loc.spilled() {
                    self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SPILL as i32));
                    self.word_at(store, v, Mem::B(S0, loc.offset() as i32));
                } else {
                    self.word_at(store, v, Mem::B(t, loc.offset() as i32));
                }
            }
            TabSetShape(shape) => {
                let t = self.r(i, 0);
                self.a.mov_imm(S0, shape);
                self.a.store(Sz::Q, Mem::B(t, layout::T_SHAPE as i32), S0);
            }
            TabASize => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.load(Sz::D, d, Mem::B(t, layout::T_ASIZE as i32));
            }
            TabArray => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                self.a.load(Sz::Q, d, Mem::B(t, layout::T_ARRAY as i32));
            }
            ArrLoad => {
                let (d, a, k) = (self.reg(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.mov32(S0, k);
                self.word_at(false, d, Mem::X(a, S0, 3, 0));
            }
            ArrStore => {
                let (a, k, v) = (self.r(i, 0), self.r(i, 1), self.reg(i, 2));
                self.a.mov32(S0, k);
                self.word_at(true, v, Mem::X(a, S0, 3, 0));
            }
            ArrLoadAt(k) => {
                let (d, a) = (self.reg(i, 0), self.r(i, 1));
                self.word_at(false, d, Mem::B(a, k as i32 * 8));
            }
            ArrStoreAt(k) => {
                let (a, v) = (self.r(i, 0), self.reg(i, 1));
                self.word_at(true, v, Mem::B(a, k as i32 * 8));
            }
            NamedNext => {
                let (d, t, from) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (n, cap, mt) = (self.r(i, 3), self.r(i, 4), self.r(i, 5));
                let [top, spilled, check, step, none, done] = [(); 6].map(|_| self.a.new_label());
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
                self.a
                    .load(Sz::D, n, Mem::B(S0, Shape::SLOT_COUNT_OFFSET as i32));
                self.a
                    .load8(cap, Mem::B(S0, Shape::INLINE_CAP_OFFSET as i32));
                self.a
                    .load(Sz::D, mt, Mem::B(S0, Shape::MT_SLOT_OFFSET as i32));
                self.a.mov32(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::D, d, n);
                self.a.jcc(Cond::Ae, none);
                self.a.cmp(Sz::D, d, mt);
                self.a.jcc(Cond::E, step);
                self.a.cmp(Sz::D, d, cap);
                self.a.jcc(Cond::Ae, spilled);
                self.a
                    .load(Sz::Q, S0, Mem::X(t, d, 3, layout::T_INLINE as i32));
                self.a.jmp(check);
                self.a.bind(spilled);
                self.a.load(Sz::Q, S1, Mem::B(t, layout::T_SPILL as i32));
                self.a.mov32(S0, d);
                self.a.alu(Arith::Sub, Sz::D, S0, cap);
                self.a.load(Sz::Q, S0, Mem::X(S1, S0, 3, 0));
                self.a.bind(check);
                self.a.cmp(Sz::Q, S0, Mem::K(NIL));
                self.a.jcc(Cond::Ne, done);
                self.a.bind(step);
                self.a.alu_imm(Arith::Add, Sz::D, d, 1);
                self.a.jmp(top);
                self.a.bind(none);
                self.a.mov_imm(d, u32::MAX as u64);
                self.a.bind(done);
            }
            NamedVal => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                let (spilled, done) = (self.a.new_label(), self.a.new_label());
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
                self.a
                    .load8(S0, Mem::B(S0, Shape::INLINE_CAP_OFFSET as i32));
                self.a.mov32(S1, k);
                self.a.cmp(Sz::D, S1, S0);
                self.a.jcc(Cond::Ae, spilled);
                self.a
                    .load(Sz::Q, d, Mem::X(t, S1, 3, layout::T_INLINE as i32));
                self.a.jmp(done);
                self.a.bind(spilled);
                self.a.alu(Arith::Sub, Sz::D, S1, S0);
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SPILL as i32));
                self.a.load(Sz::Q, d, Mem::X(S0, S1, 3, 0));
                self.a.bind(done);
            }
            NamedKey => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.mov32(S1, k);
                self.a.load(Sz::Q, S0, Mem::B(t, layout::T_SHAPE as i32));
                self.a
                    .load(Sz::Q, S0, Mem::B(S0, Shape::KEYS_OFFSET as i32));
                self.a
                    .load(Sz::Q, S0, Mem::B(S0, Shape::KEYS_SLOTS_OFFSET as i32));
                self.a.load(Sz::Q, S0, Mem::X(S0, S1, 3, 0));
                let off = crate::env::string::LuaString::VALUE_OFFSET as i32;
                self.a.lea(Sz::Q, d, Mem::B(S0, off));
                self.a
                    .alu(Arith::Or, Sz::Q, d, Mem::K(crate::env::value::STRING_BITS));
            }
            ArrNext => {
                let (d, t, from, v) = (self.r(i, 0), self.r(i, 1), self.r(i, 2), self.r(i, 3));
                let (top, done) = (self.a.new_label(), self.a.new_label());
                self.a.load(Sz::D, S0, Mem::B(t, layout::T_ASIZE as i32));
                self.a.load(Sz::Q, S1, Mem::B(t, layout::T_ARRAY as i32));
                self.a.mov_imm(v, NIL);
                self.a.mov32(d, from);
                self.a.bind(top);
                self.a.cmp(Sz::D, d, S0);
                self.a.jcc(Cond::Ae, done);
                self.a.cmp(Sz::Q, v, Mem::X(S1, d, 3, 0));
                self.a.jcc(Cond::Ne, done);
                self.a.alu_imm(Arith::Add, Sz::D, d, 1);
                self.a.jmp(top);
                self.a.bind(done);
            }
            // The slow path, out of line, keeps every register itself: rdi
            // and rsi across its own argument setup, the rest in the
            // trampoline.
            Barrier => {
                let t = self.r(i, 0);
                let (slow, back) = (self.a.new_label(), self.a.new_label());
                self.a.test8_imm(Mem::B(t, layout::T_HEADER), 1);
                self.a.jcc(Cond::E, slow);
                self.a.bind(back);
                let tramp = self
                    .links
                    .trampoline_of(crate::jit::helpers::jit_barrier as *const () as usize);
                let rt = self.links.rt;
                self.ool.push(Box::new(move |em: &mut Em<'_>| {
                    em.a.bind(slow);
                    em.a.alu_imm(Arith::Sub, Sz::Q, RSP, 16);
                    em.a.store(Sz::Q, Mem::B(RSP, 0), RDI);
                    em.a.store(Sz::Q, Mem::B(RSP, 8), RSI);
                    em.a.mov(RSI, t);
                    em.a.mov_imm(RDI, rt);
                    em.a.call_far(tramp);
                    em.a.load(Sz::Q, RDI, Mem::B(RSP, 0));
                    em.a.load(Sz::Q, RSI, Mem::B(RSP, 8));
                    em.a.alu_imm(Arith::Add, Sz::Q, RSP, 16);
                    em.a.jmp(back);
                }));
            }
            NumVarargs => {
                let d = self.r(i, 0);
                self.a.load(Sz::Q, d, Mem::B(BASE, -32));
                self.a.shift_imm(Shift::Shr, Sz::Q, d, 48);
            }
            VarargAt(k) => {
                // The extras end at the header: extra k is at
                // base - 32 - 8 * nv + 8 * k, read only when k < nv.
                let (d, nv) = (self.r(i, 0), self.r(i, 1));
                self.a.mov32(S0, nv);
                self.a.neg(Sz::Q, S0);
                self.a
                    .load(Sz::Q, S0, Mem::X(BASE, S0, 3, 8 * k as i32 - 32));
                self.a.mov_imm(S1, NIL);
                self.a.cmp_imm(Sz::D, nv, k as i32);
                self.a.cmov(Cond::Be, Sz::Q, S0, S1);
                self.a.mov(d, S0);
            }
            VarargIdx => {
                // The index clamped to 0 out of range, so the load stays in
                // the frame. `d` may share a register with either operand:
                // it is written past their last read.
                let (d, nv, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.lea(Sz::D, S0, Mem::B(k, -1));
                self.a.mov32(S1, nv);
                self.a.neg(Sz::Q, S1);
                self.a.lea(Sz::Q, S1, Mem::X(BASE, S1, 3, 0));
                self.a.cmp(Sz::D, S0, nv);
                self.a.mov_imm(d, 0);
                self.a.cmov(Cond::B, Sz::Q, d, S0);
                self.a.load(Sz::Q, S0, Mem::X(S1, d, 3, -32));
                self.a.mov_imm(d, NIL);
                self.a.cmov(Cond::B, Sz::Q, d, S0);
            }
            StoreCell => {
                let (c, v) = (self.r(i, 0), self.reg(i, 1));
                let off = crate::env::function::UpvalueCell::V_OFFSET as i32;
                self.a.load(Sz::Q, S0, Mem::B(c, off));
                self.word_at(true, v, Mem::B(S0, 0));
            }
            GcCheck { exit } => self.gc_check(exit),
            Helper(h) | HelperCold(h) => {
                let sig = h.sig();
                for (k, c) in sig.ctx.iter().enumerate() {
                    match c {
                        Ctx::Rt => self.a.mov_imm(ARGS[k], self.links.rt),
                        Ctx::Thread => self.a.mov(ARGS[k], THREAD),
                        Ctx::Base => self.a.mov(ARGS[k], BASE),
                    }
                }
                if let Some(imm) = sig.imm {
                    self.a.mov_imm(ARGS[sig.ctx.len()], imm);
                }
                let target = crate::jit::helpers::address(h);
                if matches!(inst, HelperCold(_)) {
                    self.a.call_far(self.links.trampoline_of(target));
                } else {
                    self.a.call_far(target);
                }
            }
            Jump => {
                let t = self.dest[self.v.succs(block)[0].index()];
                if Some(t) != next {
                    self.a.jmp(t);
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
                self.set_top(n, at as i32 * 8);
            }
            Resume { wanted, c, a, hdr } => self.on_resume(i, block, wanted, c, a, hdr),
            CallMm {
                hdr,
                nargs,
                pc_after,
            } => self.call_mm(i, block, next, hdr, nargs, pc_after),
            Return { a, n, reg } => self.ret(a, n, reg),
            Deopt { exit } => {
                let l = self.exit(exit);
                self.a.jmp(l);
            }
            LazyExit { exit } => {
                let (at, l) = (self.a.new_label(), self.exit(exit));
                self.a.patch_point(at);
                self.lazy.push((at, l));
            }
        }
        Ok(())
    }

    /// `S0` = the integer of `sz` that float `n` holds exactly, else on to
    /// `l`.
    fn exact_int(&mut self, sz: Sz, n: Xmm, l: Label) {
        self.a.cvt_to_int(sz, S0, n);
        self.a.cvt_from_int(sz, FS, S0);
        self.a.fcmp(n, FS);
        self.a.jcc(Cond::Ne, l);
        self.a.jcc(Cond::P, l);
    }

    /// `FMod`'s out-of-line path from `slow` back to `back`, `d` holding
    /// the inline remainder of `x` by `y`.
    fn fmod_slow(&mut self, slow: Label, back: Label, d: Xmm, x: Xmm, y: Xmm) {
        // A zero remainder is exact: fmod's zero has the dividend's sign.
        let call = self.a.new_label();
        self.a.bind(slow);
        self.a.fzero(FS);
        self.a.fcmp(d, FS);
        self.a.jcc(Cond::Ne, call);
        self.a.jcc(Cond::P, call);
        self.a.fbit(FBit::And, d, x, Mem::K2(1 << 63));
        self.a.jmp(back);
        self.a.bind(call);
        let t = self.links.trampoline_of(crate::jit::helpers::address(
            crate::jit::ir::ops::HelperId::FMod,
        ));
        // The trampoline keeps every register but xmm0, its result.
        self.a.fmov_to_gpr(S0, x);
        self.a.fmov_to_gpr(S1, y);
        self.a.alu_imm(Arith::Sub, Sz::Q, RSP, 16);
        self.a.fstore(Mem::B(RSP, 0), Xmm(0));
        self.a.fstore(Mem::B(RSP, 8), Xmm(1));
        self.a.fmov_from_gpr(Xmm(0), S0);
        self.a.fmov_from_gpr(Xmm(1), S1);
        self.a.call_far(t);
        self.a.fmov_to_gpr(S0, Xmm(0));
        self.a.fload(Xmm(0), Mem::B(RSP, 0));
        self.a.fload(Xmm(1), Mem::B(RSP, 8));
        self.a.alu_imm(Arith::Add, Sz::Q, RSP, 16);
        self.a.fmov_from_gpr(d, S0);
        self.a.jmp(back);
    }

    /// The thread's `top` past `n` values (a count in the low half) from
    /// `off` bytes above `BASE`.
    pub(super) fn set_top(&mut self, n: Gpr, off: i32) {
        self.a.mov32(S1, n);
        self.a.lea(Sz::Q, S0, Mem::B(BASE, off));
        self.a.alu(
            Arith::Sub,
            Sz::Q,
            S0,
            Mem::B(THREAD, layout::THREAD_STACK as i32),
        );
        self.a.shift_imm(Shift::Shr, Sz::Q, S0, 3);
        self.a.alu(Arith::Add, Sz::Q, S0, S1);
        self.a
            .store(Sz::Q, Mem::B(THREAD, layout::THREAD_TOP as i32), S0);
    }

    /// The frame's header takes `CLOSURE` for its closure, the vararg
    /// count kept.
    pub(super) fn set_frame_closure(&mut self) {
        self.a.load(Sz::Q, S0, Mem::B(BASE, -32));
        self.a.shift_imm(Shift::Shr, Sz::Q, S0, 48);
        self.a.shift_imm(Shift::Shl, Sz::Q, S0, 48);
        self.a.alu(Arith::Or, Sz::Q, S0, CLOSURE);
        self.a.store(Sz::Q, Mem::B(BASE, -32), S0);
    }

    pub(super) fn gc_check(&mut self, exit: u32) {
        self.a.mov_imm(S0, self.links.metrics);
        self.a.load(Sz::Q, S1, Mem::B(S0, layout::GC_CHECK as i32));
        self.a
            .cmp(Sz::Q, S1, Mem::B(S0, layout::GC_CHECK as i32 + 8));
        let l = self.exit(exit);
        self.a.jcc(Cond::Ae, l);
    }

    /// Lua's shift, into `S0`: `n << m` (or `>>`) on 64 bits, logical, 0
    /// past 63. `n` may be `S0` and `m` `S1`, which it uses.
    pub(super) fn lua_shift(&mut self, n: Gpr, m: Gpr, left: bool) {
        // A right shift is a left shift by `-m`.
        self.a.mov(S1, m);
        if !left {
            self.a.neg(Sz::Q, S1);
        }
        let [neg, zero, done] = [(); 3].map(|_| self.a.new_label());
        self.a.test(Sz::Q, S1, S1);
        self.a.jcc(Cond::S, neg);
        self.a.cmp_imm(Sz::Q, S1, 64);
        self.a.jcc(Cond::Ge, zero);
        self.a.shiftx(Shift::Shl, Sz::Q, S0, n, S1);
        self.a.jmp(done);
        self.a.bind(neg);
        // Unsigned: `-i64::MIN` is still negative.
        self.a.neg(Sz::Q, S1);
        self.a.cmp_imm(Sz::Q, S1, 64);
        self.a.jcc(Cond::Ae, zero);
        self.a.shiftx(Shift::Shr, Sz::Q, S0, n, S1);
        self.a.jmp(done);
        self.a.bind(zero);
        self.a.mov_imm(S0, 0);
        self.a.bind(done);
    }

    /// `IFCmp`'s inline part: `d = i cc f` as floats while `|i| <= 2^53`
    /// converts exactly (`i + 2^53` below 2^54, unsigned; 2^53 itself goes
    /// to `slow`, which is exact too), else on to `slow`.
    pub(super) fn if_fast(&mut self, d: Gpr, i: Gpr, f: Xmm, t: Xmm, cc: Cc, slow: Label) {
        self.a.mov_imm(S0, 1 << 53);
        self.a.alu(Arith::Add, Sz::Q, S0, i);
        self.a.shift_imm(Shift::Shr, Sz::Q, S0, 54);
        self.a.jcc(Cond::Ne, slow);
        self.a.cvt_from_int(Sz::Q, t, i);
        self.float_set(d, cc, t, f);
    }

    /// `IFCmp` past ±2^53, as Lua's `LTintfloat` and kin: a float whose
    /// floor or ceiling is an i64 compares by it (`i < ceil f`, `i <= floor
    /// f`, `floor f < i` for `Gt`, `ceil f <= i` for `Ge`, `i == floor f`,
    /// exact as a non-integral float is below 2^53 in magnitude); a NaN is
    /// false; any other float is above or below every integer by its sign.
    pub(super) fn if_big(&mut self, d: Gpr, i: Gpr, f: Xmm, t: Xmm, cc: Cc, done: Label) {
        let (no, huge) = (self.a.new_label(), self.a.new_label());
        self.a.fcmp(f, f);
        self.a.jcc(Cond::P, no);
        match cc {
            Cc::Lt | Cc::Ge => self.a.fround(t, f, 2),
            _ => self.a.fround(t, f, 1),
        }
        self.exact_int(Sz::Q, t, huge);
        self.a.cmp(Sz::Q, i, S0);
        self.a.setcc(int_cond(cc), d);
        self.a.jmp(done);
        self.a.bind(huge);
        self.a.fzero(FS);
        match cc {
            Cc::Lt | Cc::Le => {
                self.a.fcmp(f, FS);
                self.a.setcc(Cond::A, d);
                self.a.jmp(done);
            }
            Cc::Gt | Cc::Ge => {
                self.a.fcmp(FS, f);
                self.a.setcc(Cond::A, d);
                self.a.jmp(done);
            }
            Cc::Eq => {}
            _ => unreachable!("an int-float compare is ordered"),
        }
        self.a.bind(no);
        self.a.mov_imm(d, 0);
    }

    /// `if_fast` and `if_big` in place, for code already out of line.
    pub(super) fn if_both(&mut self, d: Gpr, i: Gpr, f: Xmm, t: Xmm, cc: Cc) {
        let (slow, done) = (self.a.new_label(), self.a.new_label());
        self.if_fast(d, i, f, t, cc, slow);
        self.a.jmp(done);
        self.a.bind(slow);
        self.if_big(d, i, f, t, cc, done);
        self.a.bind(done);
    }

    /// `d` = the i64 of integer `v`, small or boxed.
    pub(super) fn unbox_int(&mut self, d: Gpr, v: Gpr) {
        let (big, done) = (self.a.new_label(), self.a.new_label());
        self.is_small(v);
        self.a.jcc(Cond::Ne, big);
        self.a.movsxd(d, v);
        self.a.jmp(done);
        self.a.bind(big);
        self.a.mov(S0, v);
        self.a.alu(Arith::And, Sz::Q, S0, Mem::K(PTR_MASK));
        self.a.load(Sz::Q, d, Mem::B(S0, 0));
        self.a.bind(done);
    }

    /// Floor division or modulo of rax by operand 2, the quotient left in
    /// rax or the remainder in rdx.
    pub(super) fn divmod(
        &mut self,
        i: usize,
        div: bool,
        sz: Sz,
        nonzero: bool,
        minus1: bool,
        exit: u32,
    ) {
        let m = self.r(i, 2);
        let l = self.exit(exit);
        if !nonzero {
            self.a.test(sz, m, m);
            self.a.jcc(Cond::E, l);
        }
        let (m1, back, done) = (self.a.new_label(), self.a.new_label(), self.a.new_label());
        if minus1 {
            self.a.cmp_imm(sz, m, -1);
            self.a.jcc(Cond::E, m1);
            self.ool.push(Box::new(move |em: &mut Em<'_>| {
                em.a.bind(m1);
                if !div {
                    em.a.mov_imm(RDX, 0);
                } else {
                    // MIN // -1 leaves i32.
                    if sz == Sz::D {
                        em.a.cmp_imm(Sz::D, RAX, i32::MIN);
                        em.a.jcc(Cond::E, l);
                    }
                    em.a.neg(sz, RAX);
                }
                em.a.jmp(back);
            }));
        }
        if div {
            // q - 1 when the remainder is nonzero and the signs differ.
            self.a.mov(S0, RAX);
            self.a.alu(Arith::Xor, sz, S0, m);
            self.a.sign_rdx(sz);
            self.a.idiv(sz, m);
            self.a.test(sz, RDX, RDX);
            self.a.jcc(Cond::E, done);
            self.a.test(sz, S0, S0);
            self.a.jcc(Cond::Ns, done);
            self.a.alu_imm(Arith::Sub, sz, RAX, 1);
        } else {
            // r + m when the remainder is nonzero and its sign differs from m's.
            self.a.sign_rdx(sz);
            self.a.idiv(sz, m);
            self.a.test(sz, RDX, RDX);
            self.a.jcc(Cond::E, done);
            self.a.mov(S0, RDX);
            self.a.alu(Arith::Xor, sz, S0, m);
            self.a.jcc(Cond::Ns, done);
            self.a.alu(Arith::Add, sz, RDX, m);
        }
        self.a.bind(done);
        self.a.bind(back);
    }
}
