//! The opcode handlers, by family. Every dispatch target is declared with
//! `handler!` (`vm::abi`); shared logic is an `#[inline(always)]` function.

pub(crate) mod arith;
pub(crate) mod call;
pub(crate) mod compare;
pub(crate) mod control;
pub(crate) mod data;
pub(crate) mod field;
pub(crate) mod meta;
pub(crate) mod table;

/// Specialized handlers from rows. A row names the opcode, its
/// handler, the guard that binds the operands as the form's kinds, and the
/// operation; each arm is the shell every form of one template shares, and
/// a family's rows also list its handlers for the dispatch table as `$rows`.
/// The templates name the files' items by path; the rows live beside the
/// guards and operations they use.
macro_rules! family {
    // `R[a] = R[b] <op> R[c]`: `$guard(&R[b], &R[c])` binds the operands and
    // `$body` is an `arith::Out`; its miss and the guard's failure go to the
    // generic handler.
    (arith $rows:ident, generic = $g:ident;
     $($OP:ident = $name:ident ($guard:expr) |$l:ident, $r:ident| $body:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, lhs, rhs) = (insn.a(), insn.b(), insn.c());
                if let Some(($l, $r)) = $guard(&reg![lhs], &reg![rhs]) {
                    $crate::vm::ops::family!(@store dst, $body)
                }
                tail!($g)
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    // `R[a] = R[b] <op> imm`: `$guard(&R[b], insn, $reversed)` binds the
    // operands in source order (`$reversed` puts the immediate first).
    (imm $rows:ident, generic = $g:ident;
     $($OP:ident = $name:ident ($guard:ident, $reversed:expr) |$l:ident, $r:ident| $body:expr),* $(,)?) => {
        $($crate::vm::abi::handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (dst, src) = (insn.a(), insn.b());
                if let Some(($l, $r)) = $guard(&reg![src], insn, $reversed) {
                    $crate::vm::ops::family!(@store dst, $body)
                }
                tail!($g)
            }
        })*
        pub(crate) const $rows: &[($crate::instruction::Op, $crate::vm::abi::Handler)] =
            &[$(($crate::instruction::Op::$OP, $name)),*];
    };
    (@store $dst:ident, $body:expr) => {
        match $body {
            $crate::vm::ops::arith::Out::Value(v) => {
                reg![$dst] = v;
                next!()
            }
            $crate::vm::ops::arith::Out::Float(f) => {
                reg![$dst].write_float_unchecked(f);
                next!()
            }
            $crate::vm::ops::arith::Out::FloatChecked(f) => {
                reg![$dst].write_float(f);
                next!()
            }
            $crate::vm::ops::arith::Out::Miss => {}
        }
    };
}
pub(crate) use family;
