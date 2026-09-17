use core::hash::{Hash, Hasher};
use std::hint;
use std::marker::PhantomData;

use crate::dmm::{Collect, Gc, Mutation, collect::Trace};
use crate::env::function::Function;
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::Thread;
use crate::env::userdata::Userdata;

#[derive(Clone, Copy, Collect, PartialEq, Eq, Debug)]
#[collect(internal, require_static)]
#[repr(u8)]
pub enum ValueKind {
    Nil = 0,
    Boolean = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Table = 5,
    Function = 6,
    Thread = 7,
    Userdata = 8,
}

// NaN-boxed. Floats are stored as their raw bits; every other value lives in the negative quiet
// NaN space, `QNAN_NEG | tag << 48 | payload`. Tag 0 is deliberately unused so the hardware
// default NaN and its negation (`0x7FF8…`/`0xFFF8…`, the only NaNs arithmetic produces) stay plain
// floats; only a NaN with a crafted payload can reach the box space, and `float` remaps it.
const QNAN_NEG: u64 = 0xFFF8_0000_0000_0000;
const TAG_SHIFT: u32 = 48;
const BOX: u64 = QNAN_NEG | 1 << TAG_SHIFT;
const PAYLOAD_MASK: u64 = (1 << TAG_SHIFT) - 1;
const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;

// Tag 7 holds the immediates. Small ints fill the payload's top half with ones so the whole top
// word is 0xFFFF_FFFF: boxing a zero-extended i32 is one `orr` with a logical immediate and the
// low word can be used as a `w` register without sign extension. nil/false/true sit just below.
const TAG_USERDATA: u64 = 1;
const TAG_BOXED_INT: u64 = 2;
const TAG_STRING: u64 = 3;
const TAG_TABLE: u64 = 4;
const TAG_FUNCTION: u64 = 5;
const TAG_THREAD: u64 = 6;
const TAG_IMMEDIATE: u64 = 7;

const SMALL_INT: u64 = 0xFFFF_FFFF_0000_0000;
const NIL: u64 = 0xFFFF_FFFE_0000_0000;
const FALSE: u64 = NIL | 1;
const TRUE: u64 = NIL | 2;

// Guarantees `bits` sits at offset 0 and spans the whole 8 bytes: read_float/
// write_float cast `&Value`/`&mut Value` straight to `*const f64`/`*mut f64` and
// rely on that, not just on the size assert below.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Value<'gc> {
    bits: u64,
    _marker: PhantomData<&'gc ()>,
}

const _: () = assert!(size_of::<Value<'_>>() == 8);

impl<'gc> Value<'gc> {
    #[inline(always)]
    const fn from_bits(bits: u64) -> Self {
        Self {
            bits,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn boxed(tag: u64, payload: u64) -> Self {
        debug_assert!(tag != 0 && payload & !PAYLOAD_MASK == 0);
        Self::from_bits(QNAN_NEG | tag << TAG_SHIFT | payload)
    }

    #[inline(always)]
    fn is_boxed(&self) -> bool {
        self.bits >= BOX
    }

    /// Only meaningful when `is_boxed`.
    #[inline(always)]
    fn tag(&self) -> u64 {
        (self.bits >> TAG_SHIFT) & 7
    }

    #[inline(always)]
    fn payload(&self) -> u64 {
        self.bits & PAYLOAD_MASK
    }

    #[inline(always)]
    fn is_tag(&self, tag: u64) -> bool {
        self.bits >> TAG_SHIFT == (QNAN_NEG >> TAG_SHIFT) | tag
    }

    #[inline(always)]
    fn ptr<T>(&self) -> Gc<'gc, T> {
        unsafe { Gc::from_ptr(self.payload() as usize as *const T) }
    }

    #[inline(always)]
    fn from_ptr<T>(tag: u64, v: Gc<'gc, T>) -> Self {
        Self::boxed(tag, Gc::as_ptr(v) as usize as u64)
    }

    #[inline(always)]
    pub fn nil() -> Self {
        Self::from_bits(NIL)
    }

    #[inline(always)]
    pub fn is_nil(&self) -> bool {
        self.bits == NIL
    }

    #[inline(always)]
    pub fn boolean(v: bool) -> Self {
        Self::from_bits(if v { TRUE } else { FALSE })
    }

    #[inline(always)]
    pub fn get_boolean(&self) -> Option<bool> {
        match self.bits {
            FALSE => Some(false),
            TRUE => Some(true),
            _ => None,
        }
    }

    /// Integers outside `i32` are heap-allocated, so this needs the mutation context.
    #[inline(always)]
    pub fn integer(mc: &Mutation<'gc>, v: i64) -> Self {
        if let Ok(small) = i32::try_from(v) {
            Self::small(small)
        } else {
            Self::boxed_integer(mc, v)
        }
    }

    // Kept out of line so the allocator doesn't get inlined into every arithmetic handler.
    //
    // `v` must not fit `i32`: `Hash` hashes a boxed int by its dereferenced value but a
    // small int by its raw bits, so a boxed and a small `Value` holding the same in-range
    // `i64` would be `Eq` (which does dereference) but hash unequal, corrupting any table
    // keyed on them. `integer` is the only caller and already guarantees this.
    #[cold]
    #[inline(never)]
    fn boxed_integer(mc: &Mutation<'gc>, v: i64) -> Self {
        debug_assert!(
            i32::try_from(v).is_err(),
            "boxed_integer called with an i32-range value"
        );
        Self::from_ptr(TAG_BOXED_INT, Gc::new(mc, v))
    }

    /// An integer that is known to fit inline; never allocates.
    #[inline(always)]
    pub fn small(v: i32) -> Self {
        Self::from_bits(SMALL_INT | v as u32 as u64)
    }

    /// The inline integer, if this is one. Heap-boxed integers report `None`; this is the
    /// fast-path check, `get_integer` is the complete one.
    #[inline(always)]
    pub fn get_small(&self) -> Option<i32> {
        if hint::likely(self.bits >> 32 == SMALL_INT >> 32) {
            Some(self.bits as i32)
        } else {
            None
        }
    }

    /// Both inline integers with one compare: the top words are all ones in both operands
    /// exactly when they are in their AND.
    #[inline(always)]
    pub fn both_small(a: &Self, b: &Self) -> Option<(i32, i32)> {
        if a.bits & b.bits >= SMALL_INT {
            Some((a.bits as i32, b.bits as i32))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_integer(&self) -> Option<i64> {
        if let Some(i) = self.get_small() {
            Some(i as i64)
        } else if self.is_tag(TAG_BOXED_INT) {
            Some(*self.ptr::<i64>())
        } else {
            None
        }
    }

    /// Handlers that store straight to a slot use `write_float` instead, which keeps the box
    /// check off the result path.
    #[inline(always)]
    pub fn float(v: f64) -> Self {
        let bits = v.to_bits();
        Self::from_bits(if bits >= BOX { CANONICAL_NAN } else { bits })
    }

    #[inline(always)]
    pub fn is_float(&self) -> bool {
        !self.is_boxed()
    }

    /// The float in this slot, read straight from memory. Volatile so LLVM keeps it a separate
    /// FP-register load instead of reusing the integer load of the tag check, which would cost a
    /// GPR->FPR move on the hot path.
    ///
    /// Garbage (some other bit pattern read as a float) if the slot isn't actually a float —
    /// the caller's job to have checked `is_float` first — but never unsound: every 64-bit
    /// pattern is a valid `f64`, so this can't misinterpret memory the way a wrong-`T` pointer
    /// cast could.
    #[inline(always)]
    pub fn read_float(&self) -> f64 {
        unsafe { core::ptr::read_volatile(self as *const Self as *const f64) }
    }

    /// `*self = Value::float(f)`, storing from the FP register directly; the box-space
    /// check then runs off the result's critical path. The fixup is a second volatile store
    /// so it stays a never-taken branch rather than a select or a call with a frame.
    #[inline(always)]
    pub fn write_float(&mut self, f: f64) {
        unsafe {
            core::ptr::write_volatile(self as *mut Self as *mut f64, f);
            if hint::unlikely(f.to_bits() >= BOX) {
                core::ptr::write_volatile(self as *mut Self as *mut u64, CANONICAL_NAN);
            }
        }
    }

    #[inline(always)]
    pub fn get_float(&self) -> Option<f64> {
        if self.is_boxed() {
            return None;
        }

        Some(f64::from_bits(self.bits))
    }

    #[inline(always)]
    pub fn string(v: LuaString<'gc>) -> Self {
        Self::from_ptr(TAG_STRING, v.inner())
    }

    #[inline(always)]
    pub fn get_string(&self) -> Option<LuaString<'gc>> {
        if !self.is_tag(TAG_STRING) {
            return None;
        }

        Some(LuaString::from_inner(self.ptr()))
    }

    #[inline(always)]
    pub fn table(v: Table<'gc>) -> Self {
        Self::from_ptr(TAG_TABLE, v.inner())
    }

    #[inline(always)]
    pub fn get_table(&self) -> Option<Table<'gc>> {
        if !self.is_tag(TAG_TABLE) {
            return None;
        }

        Some(Table::from_inner(self.ptr()))
    }

    #[inline(always)]
    pub fn function(v: Function<'gc>) -> Self {
        Self::from_ptr(TAG_FUNCTION, v.inner())
    }

    #[inline(always)]
    pub fn get_function(&self) -> Option<Function<'gc>> {
        if !self.is_tag(TAG_FUNCTION) {
            return None;
        }

        Some(Function::from_inner(self.ptr()))
    }

    #[inline(always)]
    pub fn thread(v: Thread<'gc>) -> Self {
        Self::from_ptr(TAG_THREAD, v.inner())
    }

    #[inline(always)]
    pub fn get_thread(&self) -> Option<Thread<'gc>> {
        if !self.is_tag(TAG_THREAD) {
            return None;
        }

        Some(Thread::from_inner(self.ptr()))
    }

    #[inline(always)]
    pub fn userdata(v: Userdata<'gc>) -> Self {
        Self::from_ptr(TAG_USERDATA, v.inner())
    }

    #[inline(always)]
    pub fn get_userdata(&self) -> Option<Userdata<'gc>> {
        if !self.is_tag(TAG_USERDATA) {
            return None;
        }

        Some(Userdata::from_inner(self.ptr()))
    }

    #[inline(always)]
    pub fn is_falsy(&self) -> bool {
        self.bits.wrapping_sub(NIL) < 2
    }

    #[inline(always)]
    pub fn kind(self) -> ValueKind {
        if !self.is_boxed() {
            return ValueKind::Float;
        }

        match self.tag() {
            TAG_IMMEDIATE => match self.bits {
                NIL => ValueKind::Nil,
                FALSE | TRUE => ValueKind::Boolean,
                _ => ValueKind::Integer,
            },
            TAG_BOXED_INT => ValueKind::Integer,
            TAG_STRING => ValueKind::String,
            TAG_TABLE => ValueKind::Table,
            TAG_FUNCTION => ValueKind::Function,
            TAG_THREAD => ValueKind::Thread,
            TAG_USERDATA => ValueKind::Userdata,
            _ => unsafe { hint::unreachable_unchecked() },
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self.kind() {
            ValueKind::Nil => "nil",
            ValueKind::Boolean => "boolean",
            ValueKind::Integer | ValueKind::Float => "number",
            ValueKind::String => "string",
            ValueKind::Table => "table",
            ValueKind::Function => "function",
            ValueKind::Thread => "thread",
            ValueKind::Userdata => "userdata",
        }
    }
}

// Boxed integers compare and hash by value so that equal integers are interchangeable regardless
// of which side got heap-allocated.
impl<'gc> PartialEq for Value<'gc> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.bits == other.bits {
            return true;
        }

        if self.is_tag(TAG_BOXED_INT) || other.is_tag(TAG_BOXED_INT) {
            self.get_integer() == other.get_integer()
        } else {
            false
        }
    }
}

impl<'gc> Eq for Value<'gc> {}

impl<'gc> Value<'gc> {
    /// Bit-identity only, unlike `PartialEq`: never dereferences a boxed integer, so
    /// this is the comparison a dead table entry's key must use — its box may already
    /// be swept. Two differently-boxed but numerically equal integers compare unequal.
    #[inline(always)]
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<'gc> Hash for Value<'gc> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_tag(TAG_BOXED_INT) {
            state.write_u64(*self.ptr::<i64>() as u64);
        } else {
            state.write_u64(self.bits);
        }
    }
}

#[inline]
pub(crate) fn value_hash(v: Value<'_>) -> u64 {
    use std::hash::BuildHasher;
    foldhash::fast::FixedState::default().hash_one(v)
}

unsafe impl<'gc> Collect<'gc> for Value<'gc> {
    #[inline]
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if !self.is_boxed() {
            return;
        }

        // `from_ptr` recovers the header from the value's layout, so each type has to be traced
        // as itself rather than through an erased pointer.
        unsafe {
            match self.tag() {
                TAG_BOXED_INT => self.ptr::<i64>().trace(cc),
                TAG_STRING => self.get_string().unwrap_unchecked().trace(cc),
                TAG_TABLE => self.get_table().unwrap_unchecked().trace(cc),
                TAG_FUNCTION => self.get_function().unwrap_unchecked().trace(cc),
                TAG_THREAD => self.get_thread().unwrap_unchecked().trace(cc),
                TAG_USERDATA => self.get_userdata().unwrap_unchecked().trace(cc),
                _ => (),
            }
        }
    }
}
