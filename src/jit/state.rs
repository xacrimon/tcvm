//! The JIT's runtime state: the per-`State` tables compiled code and the
//! entry handlers read, and the per-prototype record of its entries.

use std::cell::{Cell, RefCell};

use crate::dmm::{Collect, Gc, Mutation, RefLock, Trace};
use crate::env::function::Prototype;
use crate::instruction::{HOT_COUNTERS, Instruction};
use crate::jit::region::Region;

/// Regions one prototype may hold at once: a function entry and three loops.
pub(crate) const MAX_ENTRIES: usize = 4;
/// Regions a prototype may ever compile.
pub(crate) const MAX_COMPILES: u8 = 16;
/// Recompiles of one entry before it is blacklisted.
pub(crate) const MAX_RECOMPILES: u8 = 4;
/// Compiler failures on a prototype before it is refused.
pub(crate) const MAX_STRIKES: u8 = 3;
/// Takings of one widenable exit that mark its entry for recompilation.
pub(crate) const EXIT_HOT: u32 = 10;
/// Words of the register image `exit_common` writes: x0-x15, x19-x21,
/// x26-x28, d0-d31, then the region's spill slots from word 64.
pub(crate) const EXIT_REGS: usize = 128;

/// Settings read from the environment when a `Lua` is created.
#[derive(Clone, Debug)]
pub(crate) struct JitConfig {
    pub(crate) enabled: bool,
    pub(crate) hot_call: u16,
    pub(crate) hot_loop: u16,
    pub(crate) log: bool,
    pub(crate) dump: DumpFlags,
    pub(crate) check: bool,
    pub(crate) fastalloc: bool,
    pub(crate) only: Option<String>,
    pub(crate) deopt_all: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DumpFlags {
    pub(crate) ir: bool,
    pub(crate) opt: bool,
    pub(crate) vcode: bool,
    pub(crate) asm: bool,
}

impl JitConfig {
    pub(crate) fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        let num = |name: &str| {
            var(name)
                .and_then(|v| v.parse::<u16>().ok())
                .map(|n| n.max(1))
                .unwrap_or(200)
        };
        let flag = |name: &str| var(name).is_some_and(|v| v != "0" && !v.is_empty());
        let mut dump = DumpFlags::default();
        if let Some(d) = var("TCVM_JIT_DUMP") {
            for part in d.split(',') {
                match part.trim() {
                    "ir" => dump.ir = true,
                    "opt" => dump.opt = true,
                    "vcode" => dump.vcode = true,
                    "asm" => dump.asm = true,
                    "all" => {
                        dump = DumpFlags {
                            ir: true,
                            opt: true,
                            vcode: true,
                            asm: true,
                        }
                    }
                    _ => {}
                }
            }
        }
        JitConfig {
            enabled: var("TCVM_JIT").is_none_or(|v| v != "off" && v != "0")
                && cfg!(target_arch = "aarch64"),
            hot_call: num("TCVM_JIT_HOT_CALL"),
            hot_loop: num("TCVM_JIT_HOT_LOOP"),
            log: flag("TCVM_JIT_LOG"),
            dump,
            check: flag("TCVM_JIT_CHECK") || cfg!(debug_assertions),
            fastalloc: flag("TCVM_JIT_FASTALLOC"),
            only: var("TCVM_JIT_ONLY"),
            deopt_all: flag("TCVM_JIT_DEOPT_ALL"),
        }
    }
}

/// The JIT's part of `State`. Compiled code reaches its fields through the
/// `layout` offsets off the `rt` register.
pub(crate) struct JitRuntime {
    /// Counters `FUNC`, `LOOP` and the `FORLOOP` forms decrement; zero
    /// tails `jit_hot`.
    pub(crate) hot: [Cell<u16>; HOT_COUNTERS],
    /// Base of the entry table: slot `d` holds what `JIT_ENTRY d` and
    /// `JIT_LOOP d` tail into. Republished when the table grows.
    pub(crate) entries: Cell<*const Cell<*const u8>>,
    table: RefCell<EntryTable>,
    /// Where `exit_common` writes the register image (`EXIT_REGS` words).
    pub(crate) exit_regs: *mut u64,
    exit_regs_buf: Box<[Cell<u64>; EXIT_REGS]>,
    pub(crate) config: JitConfig,
    code: RefCell<Option<std::rc::Rc<crate::jit::backend::alloc::CodeAllocator>>>,
    /// Compile failures from running out of code memory turn the JIT off.
    pub(crate) exhausted: Cell<bool>,
}

#[derive(Default)]
struct EntryTable {
    slots: Vec<Cell<*const u8>>,
    free: Vec<u16>,
}

impl JitRuntime {
    pub(crate) fn new() -> Self {
        let config = JitConfig::from_env();
        let start = config.hot_call.min(config.hot_loop);
        let exit_regs_buf: Box<[Cell<u64>; EXIT_REGS]> =
            Box::new(std::array::from_fn(|_| Cell::new(0)));
        let exit_regs = exit_regs_buf.as_ptr() as *mut u64;
        let rt = JitRuntime {
            hot: std::array::from_fn(|_| Cell::new(start)),
            entries: Cell::new(std::ptr::null()),
            table: RefCell::new(EntryTable::default()),
            exit_regs,
            exit_regs_buf,
            config,
            code: RefCell::new(None),
            exhausted: Cell::new(false),
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
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn code_alloc(&self) -> std::rc::Rc<crate::jit::backend::alloc::CodeAllocator> {
        self.code
            .borrow_mut()
            .get_or_insert_with(|| {
                std::rc::Rc::new(crate::jit::backend::alloc::CodeAllocator::new(
                    crate::jit::backend::aarch64::abi::exit_common,
                ))
            })
            .clone()
    }

    /// The register image of the last exit.
    pub(crate) fn exit_regs(&self) -> &[Cell<u64>; EXIT_REGS] {
        &self.exit_regs_buf
    }

    /// The counter's restart value for a site of `word`'s kind.
    pub(crate) fn hot_start(&self, word: Instruction) -> u16 {
        if word.op() == crate::instruction::Op::FUNC {
            self.config.hot_call
        } else {
            self.config.hot_loop
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
    /// Its slot points at `jit_recompile`.
    Recompile,
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
    pub(crate) recompiles: u8,
    /// A never-executed exit asked for a recompile outside the budget.
    pub(crate) free_recompile: bool,
    #[collect(require_static)]
    pub(crate) state: EntryState,
    /// Depth of the loop header in the loop forest (0 for a function entry),
    /// for eviction.
    pub(crate) depth: u16,
    pub(crate) age: u32,
    /// Kinds of the entry values failed entry guards saw, by register.
    #[collect(require_static)]
    pub(crate) seen: Vec<(u8, crate::jit::ir::types::TypeSet)>,
}

/// The per-prototype JIT record (`Prototype::jit`).
#[derive(Collect, Default)]
#[collect(internal, no_drop)]
pub(crate) struct JitState<'gc> {
    pub(crate) entries: Vec<Entry<'gc>>,
    /// Receiver shapes hot shape exits saw, by pc.
    pub(crate) shapes_seen: Vec<(u32, crate::env::shape::Shape<'gc>)>,
    /// Every region compiled for the prototype, live or retired: a frame may
    /// still return into a retired one.
    pub(crate) regions: Vec<Gc<'gc, Region<'gc>>>,
    pub(crate) strikes: u8,
    pub(crate) compiles: u8,
    /// pcs whose entries were blacklisted or struck out.
    pub(crate) refused: Vec<u32>,
    pub(crate) age: u32,
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
