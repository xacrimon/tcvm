//! The compile pipeline: bytecode to the high-level graph, the copies,
//! selection, register allocation and encoding.

use crate::env::function::LuaFn;
use crate::env::value::Value;
use crate::jit::build::cfg::CfgError;
use crate::jit::ir::types::TypeSet;
use crate::lua::Context;

/// A compile that legitimately gives up. A compiler bug is a panic
/// (`jit::bug!`), never one of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bail {
    /// The region uses what the compiler does not handle.
    Unsupported(&'static str),
    TooLarge,
    /// Code memory ran out.
    OutOfCode,
    /// No path does work before an exit nothing widens.
    Useless,
}

/// What tiering does about a bail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BailAction {
    /// Refuse the entry and count a strike against its prototype.
    Strike,
    /// Refuse the entry alone.
    Refuse,
    /// Turn the JIT off.
    Exhaust,
}

impl Bail {
    pub(crate) fn action(self) -> BailAction {
        match self {
            Bail::Unsupported(_) | Bail::TooLarge => BailAction::Strike,
            Bail::Useless => BailAction::Refuse,
            Bail::OutOfCode => BailAction::Exhaust,
        }
    }

    /// The bail's kind, without its reason.
    pub(crate) fn class(self) -> &'static str {
        match self {
            Bail::Unsupported(_) => "unsupported",
            Bail::TooLarge => "too-large",
            Bail::OutOfCode => "out-of-code",
            Bail::Useless => "useless",
        }
    }
}

impl std::fmt::Display for Bail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bail::Unsupported(why) => write!(f, "unsupported: {why}"),
            Bail::TooLarge => write!(f, "too large"),
            Bail::OutOfCode => write!(f, "out of code memory"),
            Bail::Useless => write!(f, "no path does work before a deopt"),
        }
    }
}

impl From<CfgError> for Bail {
    fn from(e: CfgError) -> Self {
        match e {
            CfgError::Irreducible => Bail::Unsupported("irreducible control flow"),
        }
    }
}

impl From<crate::jit::backend::alloc::ReserveError> for Bail {
    fn from(e: crate::jit::backend::alloc::ReserveError) -> Self {
        match e {
            crate::jit::backend::alloc::ReserveError::TooLarge => Bail::TooLarge,
            crate::jit::backend::alloc::ReserveError::Exhausted(_) => Bail::OutOfCode,
        }
    }
}

/// The word a JIT entry replaced, for an instruction that is one.
pub(crate) fn resolve_original(
    proto: &crate::env::function::Prototype<'_>,
    i: crate::instruction::Instruction,
) -> crate::instruction::Instruction {
    if !i.is_jit_word() {
        return i;
    }
    let state = proto.jit.get().expect("a JIT word without a JIT state");
    let st = state.borrow();
    st.entries
        .iter()
        .find(|e| e.slot == i.d())
        .map(|e| e.original)
        .expect("a JIT word without its entry")
}

/// The `TAILCALL`s of a function region whose call sites saw only its own
/// prototype's closures, which become jumps to its start, the callee guarded to
/// be the frame's: not for a vararg function, nor one whose closures capture
/// its registers (the frame's open upvalues would have to close first).
pub(crate) fn self_tail_calls(
    closure: LuaFn<'_>,
    code: &[crate::instruction::Instruction],
) -> Vec<u32> {
    use crate::env::feedback::slot_proto;
    let proto = closure.proto;
    if proto.is_vararg
        || crate::jit::build::cfg::captured(&proto)
            .iter()
            .next()
            .is_some()
    {
        return Vec::new();
    }
    let me = crate::env::value::Value::function(closure.function()).to_raw();
    (0..code.len() as u32)
        .filter(|&t| {
            let i = code[t as usize];
            // Up to `top` only after a call setting it.
            let after_call = t > 0 && {
                let p = code[t as usize - 1];
                p.c() == 0
                    && matches!(
                        p.op().unquickened(),
                        crate::instruction::Op::CALL
                            | crate::instruction::Op::CALL_R0
                            | crate::instruction::Op::CALL_R1
                            | crate::instruction::Op::CALLS
                            | crate::instruction::Op::CALLS_R0
                            | crate::instruction::Op::CALLS_R1
                    )
            };
            if i.op() != crate::instruction::Op::TAILCALL
                || (i.b() == 0 && !after_call)
                || proto.feedback.bits(t as usize) & crate::jit::feedback::CONST_MISS != 0
            {
                return false;
            }
            let w = proto.feedback.callee(t as usize).map_or(0, |c| c.get());
            w == me || slot_proto(w).is_some_and(|p| crate::dmm::Gc::ptr_eq(p, proto))
        })
        .collect()
}

/// Point each counted back edge that jumps past a `LOOP` at it, as the CFG
/// takes the `LOOP` for the loop's header; an entry there pointed its own
/// already (`Entry::back_edges`).
pub(crate) fn canonical_back_edges(
    proto: &crate::env::function::Prototype<'_>,
    code: &mut [crate::instruction::Instruction],
) {
    use crate::instruction::Op;
    let pointed: Vec<u32> = proto.jit.get().map_or(Vec::new(), |s| {
        s.borrow()
            .entries
            .iter()
            .flat_map(|e| e.back_edges.iter().copied())
            .collect()
    });
    for pc in 0..code.len() {
        let mut i = code[pc];
        if !matches!(i.op(), Op::JMP_BACK | Op::TFORLOOP) || pointed.contains(&(pc as u32)) {
            continue;
        }
        let t = (pc as i64 + 1 + i.branch_offset() as i64) as usize;
        if code[t - 1].op() == Op::LOOP {
            assert!(i.set_branch_offset(i.branch_offset() - 1));
            code[pc] = i;
        }
    }
}

static UPVALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Bytes from a Lua closure's cell to its first upvalue slot: the same for
/// every closure, noted at each compile.
pub(crate) fn upvalues_offset() -> usize {
    UPVALS.load(std::sync::atomic::Ordering::Relaxed)
}

fn note_upvalues_offset(closure: LuaFn<'_>) {
    let o = closure.upvalue_ptr() as usize - closure.as_ptr() as usize;
    UPVALS.store(o, std::sync::atomic::Ordering::Relaxed);
}

/// Compile the region of `closure` entered at `pc`, place its code, and
/// return it ready to install.
pub(crate) fn compile<'gc>(
    _ctx: Context<'gc>,
    closure: LuaFn<'gc>,
    _pc: u32,
    _frame: *const Value<'gc>,
    _seen: &[(u8, TypeSet)],
) -> Result<crate::dmm::Gc<'gc, crate::jit::region::Region<'gc>>, Bail> {
    note_upvalues_offset(closure);
    Err(Bail::Unsupported("no builder yet"))
}

pub(crate) fn chunk_name(closure: LuaFn<'_>) -> String {
    format!(
        "{}:{}",
        String::from_utf8_lossy(closure.proto.source.as_bytes()),
        closure.proto.line_defined
    )
}
