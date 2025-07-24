use std::mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueType {
    Nil = 0,
    Boolean = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Userdata = 5,
    Function = 6,
    Thread = 7,
    Table = 8,
}

#[derive(Clone, Copy)]
enum Fat {
    Nil,
    Boolean(bool),
    Integer(i32),
    Float(f64),
    String(*const ()),
    Userdata(*const ()),
    Function(*const ()),
    Thread(*const ()),
    Table(*const ()),
}
