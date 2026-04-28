use core::hash::{Hash, Hasher};
use std::hint;
use std::marker::PhantomData;

use crate::dmm::{Collect, Gc, collect::Trace};
use crate::env::function::Function;
use crate::env::string::LuaString;
use crate::env::table::Table;
use crate::env::thread::Thread;
use crate::env::userdata::Userdata;

#[derive(Clone, Copy, Collect, PartialEq, Eq)]
#[collect(internal, require_static)]
#[repr(u8)]
pub enum ValueKind {
    Nil,
    Boolean,
    Integer,
    Float,
    String,
    Table,
    Function,
    Thread,
    Userdata,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Value<'gc> {
    kind: ValueKind,
    data: u64,
    _marker: PhantomData<&'gc ()>,
}

impl<'gc> Value<'gc> {
    pub fn nil() -> Self {
        Self {
            kind: ValueKind::Nil,
            data: 0,
            _marker: PhantomData,
        }
    }

    pub fn is_nil(&self) -> bool {
        self.kind == ValueKind::Nil
    }

    pub fn boolean(v: bool) -> Self {
        Self {
            kind: ValueKind::Boolean,
            data: v as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_boolean(self) -> Option<bool> {
        if self.kind != ValueKind::Boolean {
            return None;
        }

        Some(match self.data {
            0 => false,
            1 => true,
            _ => unsafe { hint::unreachable_unchecked() },
        })
    }

    pub fn integer(v: i64) -> Self {
        Self {
            kind: ValueKind::Integer,
            data: v as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_integer(self) -> Option<i64> {
        if self.kind != ValueKind::Integer {
            return None;
        }

        Some(self.data as i64)
    }

    pub fn float(v: f64) -> Self {
        Self {
            kind: ValueKind::Float,
            data: v.to_bits(),
            _marker: PhantomData,
        }
    }

    pub fn get_float(self) -> Option<f64> {
        if self.kind != ValueKind::Float {
            return None;
        }

        Some(f64::from_bits(self.data))
    }

    pub fn string(v: LuaString<'gc>) -> Self {
        Self {
            kind: ValueKind::String,
            data: Gc::as_ptr(v.inner()) as usize as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_string(self) -> Option<LuaString<'gc>> {
        if self.kind != ValueKind::String {
            return None;
        }

        let ptr = unsafe { Gc::from_ptr(self.data as usize as *const _) };
        Some(LuaString::from_inner(ptr))
    }

    pub fn table(v: Table<'gc>) -> Self {
        Self {
            kind: ValueKind::Table,
            data: Gc::as_ptr(v.inner()) as usize as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_table(self) -> Option<Table<'gc>> {
        if self.kind != ValueKind::Table {
            return None;
        }

        let ptr = unsafe { Gc::from_ptr(self.data as usize as *const _) };
        Some(Table::from_inner(ptr))
    }

    pub fn function(v: Function<'gc>) -> Self {
        Self {
            kind: ValueKind::Function,
            data: Gc::as_ptr(v.inner()) as usize as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_function(self) -> Option<Function<'gc>> {
        if self.kind != ValueKind::Function {
            return None;
        }

        let ptr = unsafe { Gc::from_ptr(self.data as usize as *const _) };
        Some(Function::from_inner(ptr))
    }

    pub fn thread(v: Thread<'gc>) -> Self {
        Self {
            kind: ValueKind::Thread,
            data: Gc::as_ptr(v.inner()) as usize as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_thread(self) -> Option<Thread<'gc>> {
        if self.kind != ValueKind::Thread {
            return None;
        }

        let ptr = unsafe { Gc::from_ptr(self.data as usize as *const _) };
        Some(Thread::from_inner(ptr))
    }

    pub fn userdata(v: Userdata<'gc>) -> Self {
        Self {
            kind: ValueKind::Userdata,
            data: Gc::as_ptr(v.inner()) as usize as u64,
            _marker: PhantomData,
        }
    }

    pub fn get_userdata(self) -> Option<Userdata<'gc>> {
        if self.kind != ValueKind::Userdata {
            return None;
        }

        let ptr = unsafe { Gc::from_ptr(self.data as usize as *const _) };
        Some(Userdata::from_inner(ptr))
    }

    pub fn is_falsy(&self) -> bool {
        self.kind == ValueKind::Nil || self.get_boolean() == Some(false)
    }

    pub fn kind(self) -> ValueKind {
        self.kind
    }

    pub fn type_name(&self) -> &'static str {
        match self.kind {
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

impl<'gc> Eq for Value<'gc> {}

impl<'gc> Hash for Value<'gc> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.data);
    }
}

unsafe impl<'gc> Collect<'gc> for Value<'gc> {
    #[inline]
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        unsafe {
            match self.kind {
                ValueKind::String => self.get_string().unwrap_unchecked().trace(cc),
                ValueKind::Table => self.get_table().unwrap_unchecked().trace(cc),
                ValueKind::Function => self.get_function().unwrap_unchecked().trace(cc),
                ValueKind::Thread => self.get_thread().unwrap_unchecked().trace(cc),
                ValueKind::Userdata => self.get_userdata().unwrap_unchecked().trace(cc),
                _ => (),
            }
        }
    }
}
