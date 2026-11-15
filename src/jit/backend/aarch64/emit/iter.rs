//! A generic `for`'s steps over a table: the scans for the next slot that
//! holds a value, and what a named slot's index gives.

use crate::env::Shape;
use crate::env::value::NIL;
use crate::jit::backend::aarch64::abi::{X16, X17};
use crate::jit::backend::aarch64::asm::{Cond, Extend, Sz};
use crate::jit::backend::aarch64::emit::Em;
use crate::jit::backend::aarch64::inst::{MInst, MOp};
use crate::jit::layout;

impl Em<'_> {
    pub(super) fn step(&mut self, i: usize, m: MInst) {
        match m.op {
            MOp::ArrNext => {
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
            MOp::NamedNext => {
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
            MOp::NamedVal => {
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
            MOp::NamedKey => {
                let (d, t, k) = (self.r(i, 0), self.r(i, 1), self.r(i, 2));
                self.a.ldr(X16, t, layout::T_SHAPE as i32);
                self.a.ldr(X16, X16, Shape::KEYS_OFFSET as i32);
                self.a.ldr(X16, X16, Shape::KEYS_SLOTS_OFFSET as i32);
                self.a.ldr_idx(X16, X16, k, Extend::Uxtw, true);
                let value = crate::env::string::LuaString::VALUE_OFFSET as u64;
                self.a.add_imm(Sz::X, X16, X16, value);
                self.a.ldr_lit(X17, crate::env::value::STRING_BITS);
                self.a.orr(Sz::X, d, X16, X17);
            }
            // No cell: zero. Else the parts' counts or'd.
            _ => {
                let (d, t) = (self.r(i, 0), self.r(i, 1));
                let done = self.a.new_label();
                self.a.ldr(X16, t, layout::T_AUX as i32);
                self.a.mov_imm(X17, 0);
                self.a.cbz(Sz::X, X16, done);
                for off in layout::aux_counts(m.imm as u8) {
                    self.a.ldr(d, X16, off as i32);
                    self.a.orr(Sz::X, X17, X17, d);
                }
                self.a.bind(done);
                self.a.mov(d, X17);
            }
        }
    }
}
