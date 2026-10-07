//! Moves, loads and upvalue access.

use crate::env::function::UpvalueCell;
use crate::env::value::Value;
use crate::vm::abi::handler;
use crate::vm::frame::fill_nil;

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `R[a] = R[b]`
    op fn op_move {
        let (dst, src) = insn.ab();
        reg![dst] = reg![src];
        next!()
    }

    /// `R[a] = K[d]`
    op fn op_load {
        let (dst, idx) = insn.ad();
        reg![dst] = k![idx];
        next!()
    }

    /// `R[a] = imm`, a small integer.
    op fn op_loadi {
        let (dst, imm) = insn.a_imm();
        reg![dst] = Value::small(imm);
        next!()
    }

    /// `R[a .. a+b) = nil`
    op fn op_loadnil {
        let (dst, count) = insn.ab();
        unsafe { fill_nil(base.add(dst as usize), count as usize) };
        next!()
    }

    /// `R[a] = false; pc++`
    op fn op_lfalseskip {
        reg![insn.a()] = Value::boolean(false);
        jump_by!(1);
        next!()
    }

    /// `R[a] = UpValue[b]`, a by-value upvalue.
    op fn op_getupval {
        let (dst, idx) = insn.ab();
        reg![dst] = upval!(value idx);
        next!()
    }

    /// `R[a] = UpValue[b]`, a shared cell.
    op fn op_getupval_ref {
        let (dst, idx) = insn.ab();
        reg![dst] = upval!(cell idx).get();
        next!()
    }

    /// `UpValue[b] = R[a]`, always a shared cell.
    op fn op_setupval {
        let (src, idx) = insn.ab();
        let v = reg![src];
        let cell = upval!(cell idx);
        if let Some(target) = UpvalueCell::barrier_target(cell, thread!().handle()) {
            barrier!(target);
        }
        unsafe { UpvalueCell::set_barriered(cell, v) };
        next!()
    }
}
