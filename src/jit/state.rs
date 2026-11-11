//! The JIT's runtime state: the per-`State` tables compiled code and the
//! entry handlers read, and the per-prototype record of its entries.

use std::cell::{Cell, RefCell};

use crate::dmm::{Collect, Gc, Mutation, RefLock, Trace};
use crate::env::function::Prototype;
use crate::instruction::{HOT_COUNTERS, Instruction};
use crate::jit::backend::target::abi::EXIT_REGS;
use crate::jit::region::Region;

/// Regions one prototype may hold at once: a function entry and three loops.
pub(crate) const MAX_ENTRIES: usize = 4;
/// Regions a prototype may ever compile.
pub(crate) const MAX_COMPILES: u8 = 16;
/// Recompiles of one entry that learned nothing before it is blacklisted;
/// each doubles the exits and executions the next one waits for.
pub(crate) const MAX_RETRIES: u8 = 6;
/// Compiler failures on a prototype before it is refused.
pub(crate) const MAX_STRIKES: u8 = 3;
/// Takings of one widenable exit that mark its entry for recompilation,
/// before backoff (`Region::exit_hot`).
pub(crate) const EXIT_HOT: u32 = 10;

/// Settings read from the environment when a `Lua` is created: `TCVM_JIT=off`,
/// and with `tcvm-dev` the development knobs.
#[derive(Clone, Debug)]
pub(crate) struct JitConfig {
    pub(crate) enabled: bool,
    #[cfg(feature = "tcvm-dev")]
    pub(crate) dev: Dev,
}

/// Invocations of a function (a power of two: `FUNC` tests the count's low
/// bits) and iterations of a loop before compiling.
pub(crate) const HOT_CALL: u16 = 256;
pub(crate) const HOT_LOOP: u16 = 200;

/// The development knobs (CLAUDE.md).
#[cfg(feature = "tcvm-dev")]
#[derive(Clone, Debug)]
pub(crate) struct Dev {
    pub(crate) hot_call: u16,
    pub(crate) hot_loop: u16,
    pub(crate) log: bool,
    pub(crate) dump: DumpFlags,
    pub(crate) check: bool,
    /// Log each compile's time per stage.
    pub(crate) time: bool,
    pub(crate) only: Option<String>,
    pub(crate) deopt_all: bool,
    /// Inline known small callees (E1); `TCVM_JIT_INLINE=0` turns it off.
    pub(crate) inline: bool,
    /// Record each compile's allocation in this directory (`ra::corpus`).
    pub(crate) ra_capture: Option<String>,
}

#[cfg(feature = "tcvm-dev")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DumpFlags {
    pub(crate) ir: bool,
    pub(crate) opt: bool,
    pub(crate) vcode: bool,
    pub(crate) ra: bool,
    pub(crate) asm: bool,
}

impl JitConfig {
    pub(crate) fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        JitConfig {
            enabled: var("TCVM_JIT").is_none_or(|v| v != "off" && v != "0"),
            #[cfg(feature = "tcvm-dev")]
            dev: Dev::from_env(),
        }
    }

    #[inline]
    pub(crate) fn hot_call(&self) -> u16 {
        #[cfg(feature = "tcvm-dev")]
        return self.dev.hot_call;
        #[cfg(not(feature = "tcvm-dev"))]
        HOT_CALL
    }

    #[inline]
    pub(crate) fn hot_loop(&self) -> u16 {
        #[cfg(feature = "tcvm-dev")]
        return self.dev.hot_loop;
        #[cfg(not(feature = "tcvm-dev"))]
        HOT_LOOP
    }

    #[inline]
    pub(crate) fn inline(&self) -> bool {
        #[cfg(feature = "tcvm-dev")]
        return self.dev.inline;
        #[cfg(not(feature = "tcvm-dev"))]
        true
    }
}

#[cfg(feature = "tcvm-dev")]
impl Dev {
    fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        let num = |name: &str, default: u16| {
            var(name)
                .and_then(|v| v.parse::<u16>().ok())
                .map(|n| n.max(1))
                .unwrap_or(default)
        };
        let flag = |name: &str| var(name).is_some_and(|v| v != "0" && !v.is_empty());
        let mut dump = DumpFlags::default();
        if let Some(d) = var("TCVM_JIT_DUMP") {
            for part in d.split(',') {
                match part.trim() {
                    "ir" => dump.ir = true,
                    "opt" => dump.opt = true,
                    "vcode" => dump.vcode = true,
                    "ra" => dump.ra = true,
                    "asm" => dump.asm = true,
                    "all" => {
                        dump = DumpFlags {
                            ir: true,
                            opt: true,
                            vcode: true,
                            ra: true,
                            asm: true,
                        }
                    }
                    _ => {}
                }
            }
        }
        Dev {
            hot_call: num("TCVM_JIT_HOT_CALL", HOT_CALL)
                .checked_next_power_of_two()
                .unwrap_or(1 << 15),
            hot_loop: num("TCVM_JIT_HOT_LOOP", HOT_LOOP),
            log: flag("TCVM_JIT_LOG"),
            dump,
            check: flag("TCVM_JIT_CHECK") || cfg!(debug_assertions),
            time: flag("TCVM_JIT_TIME"),
            only: var("TCVM_JIT_ONLY"),
            deopt_all: flag("TCVM_JIT_DEOPT_ALL"),
            inline: var("TCVM_JIT_INLINE").is_none_or(|v| v != "0" && v != "off"),
            ra_capture: var("TCVM_JIT_RA_CAPTURE"),
        }
    }
}

/// The JIT's part of `State`. Compiled code reaches its fields through the
/// `layout` offsets off the `rt` register.
pub(crate) struct JitRuntime {
    /// Counters `LOOP` and the `FORLOOP` forms decrement; zero tails
    /// `jit_hot`.
    pub(crate) hot: [Cell<u16>; HOT_COUNTERS],
    /// `FUNC` tails `jit_hot` when a function's invocation count has these
    /// bits clear: every `hot_call` (a power of two) invocations; all ones
    /// while the JIT is off.
    pub(crate) call_mask: Cell<u64>,
    /// Base of the entry table: slot `d` holds what `JIT_ENTRY d` and
    /// `JIT_LOOP d` tail into. Republished when the table grows.
    pub(crate) entries: Cell<*const Cell<*const u8>>,
    table: RefCell<EntryTable>,
    /// Where `exit_common` writes the register image (`EXIT_REGS` words).
    pub(crate) exit_regs: *mut u64,
    exit_regs_buf: Box<[Cell<u64>; EXIT_REGS]>,
    pub(crate) config: JitConfig,
    /// The cold-arm trampolines made so far: (helper, its entry). Their code
    /// is freed before the allocator below.
    trampolines: RefCell<Vec<(usize, usize)>>,
    trampoline_code: RefCell<Vec<crate::jit::backend::alloc::CodeBlock>>,
    code: RefCell<Option<std::rc::Rc<crate::jit::backend::alloc::CodeAllocator>>>,
    /// The register allocator, its buffers kept from compile to compile.
    pub(crate) ra: RefCell<crate::jit::backend::ra::Ctx>,
    /// Compile failures from running out of code memory turn the JIT off.
    pub(crate) exhausted: Cell<bool>,
    /// Where handlers return in a level (`abi::level_return`).
    pub(crate) level_ret: usize,
    /// The return address of a region dispatch entered (`abi::level_land`):
    /// its `ret` lands there and goes on in the level.
    pub(crate) level_land: usize,
    /// What `abi::level_jump` branches to.
    pub(crate) jump_target: Cell<usize>,
    /// Compiles that failed on a verifier or backend error, not a refusal
    /// (`Lua::jit_internal_errors`).
    #[cfg(feature = "tcvm-dev")]
    pub(crate) internal_errors: Cell<u32>,
}

#[derive(Default)]
struct EntryTable {
    slots: Vec<Cell<*const u8>>,
    free: Vec<u16>,
}

impl JitRuntime {
    pub(crate) fn new() -> Self {
        let config = JitConfig::from_env();
        let start = config.hot_call().min(config.hot_loop());
        let exit_regs_buf: Box<[Cell<u64>; EXIT_REGS]> =
            Box::new(std::array::from_fn(|_| Cell::new(0)));
        let exit_regs = exit_regs_buf.as_ptr() as *mut u64;
        let rt = JitRuntime {
            hot: std::array::from_fn(|_| Cell::new(start)),
            call_mask: Cell::new(config.hot_call() as u64 - 1),
            entries: Cell::new(std::ptr::null()),
            table: RefCell::new(EntryTable::default()),
            exit_regs,
            exit_regs_buf,
            config,
            trampolines: RefCell::new(Vec::new()),
            trampoline_code: RefCell::new(Vec::new()),
            code: RefCell::new(None),
            ra: RefCell::new(crate::jit::backend::ra::Ctx::new(
                crate::jit::backend::target::abi::ra_env(),
            )),
            exhausted: Cell::new(false),
            level_ret: crate::jit::backend::target::abi::level_return_addr(),
            level_land: crate::vm::stack::level_land(),
            jump_target: Cell::new(0),
            #[cfg(feature = "tcvm-dev")]
            internal_errors: Cell::new(0),
        };
        rt.grow_table(16);
        rt
    }

    fn grow_table(&self, n: usize) {
        let mut t = self.table.borrow_mut();
        let old = t.slots.len();
        let mut slots: Vec<Cell<*const u8>> = Vec::with_capacity(n);
        for s in &t.slots {
            slots.push(Cell::new(s.get()));
        }
        slots.resize_with(n, || Cell::new(std::ptr::null()));
        for i in (old..n).rev() {
            t.free.push(i as u16);
        }
        t.slots = slots;
        self.entries.set(t.slots.as_ptr());
    }

    /// A free entry-table slot holding `code`.
    pub(crate) fn alloc_slot(&self, code: *const u8) -> Option<u16> {
        if self.table.borrow().free.is_empty() {
            let n = self.table.borrow().slots.len();
            if n >= u16::MAX as usize {
                return None;
            }
            self.grow_table((n * 2).min(u16::MAX as usize));
        }
        let mut t = self.table.borrow_mut();
        let d = t.free.pop()?;
        t.slots[d as usize].set(code);
        Some(d)
    }

    pub(crate) fn set_slot(&self, d: u16, code: *const u8) {
        self.table.borrow().slots[d as usize].set(code);
    }

    pub(crate) fn free_slot(&self, d: u16) {
        let mut t = self.table.borrow_mut();
        t.slots[d as usize].set(std::ptr::null());
        t.free.push(d);
    }

    /// The code allocator, created with the first compile.
    pub(crate) fn code_alloc(&self) -> std::rc::Rc<crate::jit::backend::alloc::CodeAllocator> {
        self.code
            .borrow_mut()
            .get_or_insert_with(|| {
                std::rc::Rc::new(crate::jit::backend::alloc::CodeAllocator::new())
            })
            .clone()
    }

    /// The trampoline that calls `helper` keeping the registers (A5), made
    /// on first use.
    pub(crate) fn trampoline(
        &self,
        helper: usize,
        kind: crate::jit::backend::lower::Tramp,
    ) -> Result<usize, crate::jit::compile::CompileError> {
        if let Some(&(_, t)) = self.trampolines.borrow().iter().find(|(h, _)| *h == helper) {
            return Ok(t);
        }
        let (asm, frame) = crate::jit::backend::target::emit::trampoline(helper, kind);
        let alloc = self.code_alloc();
        let mut block = alloc.reserve(asm.max_len())?;
        let code = asm
            .finish(block.rx() as usize)
            .map_err(|e| crate::jit::compile::CompileError::Backend(format!("assembler: {e:?}")))?;
        block.write(&code);
        block.describe_frame(frame);
        let t = block.rx() as usize;
        self.trampoline_code.borrow_mut().push(block);
        self.trampolines.borrow_mut().push((helper, t));
        Ok(t)
    }

    /// The register image of the last exit.
    pub(crate) fn exit_regs(&self) -> &[Cell<u64>; EXIT_REGS] {
        &self.exit_regs_buf
    }

    /// The counter's restart value for a site of `word`'s kind.
    pub(crate) fn hot_start(&self, word: Instruction) -> u16 {
        if word.op() == crate::instruction::Op::FUNC {
            self.config.hot_call()
        } else {
            self.config.hot_loop()
        }
    }
}

// SAFETY: holds no `Gc` pointer: regions are owned by their prototypes.
unsafe impl<'gc> Collect<'gc> for JitRuntime {
    const NEEDS_TRACE: bool = false;
    fn trace<T: Trace<'gc>>(&self, _cc: &mut T) {}
}

/// Where an entry stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EntryState {
    Compiled,
    /// Its slot points at `jit_settle`, which runs the original word until
    /// `Entry::settle` runs out, then recompiles.
    Settling,
}

/// A compiled entry of a prototype: the counting word it replaced and the
/// region its slot enters.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct Entry<'gc> {
    pub(crate) pc: u32,
    #[collect(require_static)]
    pub(crate) original: Instruction,
    pub(crate) slot: u16,
    pub(crate) region: Option<Gc<'gc, Region<'gc>>>,
    /// Recompiles that learned nothing (`MAX_RETRIES`).
    pub(crate) retries: u8,
    /// The recompile asked for learned feedback: outside the budget.
    pub(crate) free_recompile: bool,
    /// Executions left before a settling entry recompiles.
    pub(crate) settle: u32,
    #[collect(require_static)]
    pub(crate) state: EntryState,
    /// The headers (first pcs) of a loop entry's loop and the loops around
    /// it, innermost first; empty for a function entry.
    #[collect(require_static)]
    pub(crate) nest: Box<[u32]>,
    pub(crate) age: u32,
    /// The back edges of a `LOOP` entry's loop, which jump past the `LOOP`
    /// until the install points them at it.
    #[collect(require_static)]
    pub(crate) back_edges: Box<[u32]>,
    /// Kinds of the entry values failed entry guards saw, by register.
    #[collect(require_static)]
    pub(crate) seen: Vec<(u8, crate::jit::ir::types::TypeSet)>,
}

/// The per-prototype JIT record (`Prototype::jit`).
#[derive(Collect, Default)]
#[collect(internal, no_drop)]
pub(crate) struct JitState<'gc> {
    pub(crate) entries: Vec<Entry<'gc>>,
    /// Every region compiled for the prototype, live or retired: a frame may
    /// still return into a retired one.
    pub(crate) regions: Vec<Gc<'gc, Region<'gc>>>,
    pub(crate) strikes: u8,
    pub(crate) compiles: u8,
    /// pcs whose entries were blacklisted or struck out.
    pub(crate) refused: Vec<u32>,
    pub(crate) age: u32,
    /// The direct branches of compiled code to the prototype's known entry
    /// (D2): each region's, by its code offset, with a call's stub
    /// (`Emitted::direct`), which `jit::set_fast` repoints as the fast entry
    /// comes and goes.
    pub(crate) callers: Vec<(crate::dmm::GcWeak<'gc, Region<'gc>>, u32, Option<u32>)>,
}

pub(crate) type JitStateRef<'gc> = Gc<'gc, RefLock<JitState<'gc>>>;

/// The prototype's JIT record, created on first use.
pub(crate) fn jit_state<'gc>(
    mc: &Mutation<'gc>,
    proto: Gc<'gc, Prototype<'gc>>,
) -> JitStateRef<'gc> {
    if let Some(s) = proto.jit.get() {
        return s;
    }
    let s = Gc::new(mc, RefLock::new(JitState::default()));
    mc.backward_barrier(Gc::erase(proto), None);
    // SAFETY: the barrier above covers the adopted pointer.
    unsafe { proto.jit.as_cell() }.set(Some(s));
    s
}
