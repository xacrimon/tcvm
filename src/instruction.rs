type Register = u8;
type UpvalueIndex = u8;
type ConstantIndex = u16;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
#[repr(align(8))]
pub enum Instruction {
    MOVE {
        dst: Register,
        src: Register,
    },

    LOAD {
        dst: Register,
        idx: ConstantIndex,
    },

    LFALSESKIP {
        src: Register,
    },

    GETUPVAL {
        dst: Register,
        idx: UpvalueIndex,
    },

    SETUPVAL {
        src: Register,
        idx: UpvalueIndex,
    },

    GETTABUP {
        dst: Register,
        idx: UpvalueIndex,
        key: ConstantIndex,
    },

    SETTABUP {
        src: Register,
        idx: UpvalueIndex,
        key: ConstantIndex,
    },

    GETTABLE {
        dst: Register,
        table: Register,
        key: Register,
    },

    SETTABLE {
        src: Register,
        table: Register,
        key: Register,
    },

    NEWTABLE {
        dst: Register,
    },

    ADD {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    SUB {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    MUL {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    MOD {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    POW {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    DIV {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    IDIV {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    BAND {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    BOR {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    BXOR {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    SHL {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    SHR {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    MMBIN {
        lhs: Register,
        rhs: Register,
        metamethod: MetaMethod,
    },

    UNM {
        dst: Register,
        src: Register,
    },

    BNOT {
        dst: Register,
        src: Register,
    },

    NOT {
        dst: Register,
        src: Register,
    },

    LEN {
        dst: Register,
        src: Register,
    },

    CONCAT {
        dst: Register,
        lhs: Register,
        rhs: Register,
    },

    CLOSE {
        start: Register,
    },

    TBC {
        val: Register,
    },

    JMP {
        offset: i32,
    },

    EQ {
        lhs: Register,
        rhs: Register,
        inverted: bool,
    },

    LT {
        lhs: Register,
        rhs: Register,
        inverted: bool,
    },

    LE {
        lhs: Register,
        rhs: Register,
        inverted: bool,
    },

    TEST {
        src: Register,
        inverted: bool,
    },

    CALL {
        func: Register,
        args: u8,
        returns: u8,
    },

    TAILCALL {
        func: Register,
        args: u8,
    },

    RETURN {
        values: Register,
        count: u8,
    },

    FORLOOP {},

    FORPREP {},

    TFORPREP {},

    TFORCALL {},

    TFORLOOP {},

    SETLIST {},

    CLOSURE {},

    VARARG {},

    VARARGPREP {},

    NOP,

    STOP,
}

impl Instruction {
    pub fn discriminant(self) -> u8 {
        unsafe { *<*const _>::from(&self).cast::<u8>() }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MetaMethod {
    INDEX,
    NEWINDEX,
    GC,
    MODE,
    LEN,
    EQ,
    ADD,
    SUB,
    MUL,
    MOD,
    POW,
    DIV,
    IDIV,
    BAND,
    BOR,
    BXOR,
    SHL,
    SHR,
    UNM,
    BNOT,
    LT,
    LE,
    CONCAT,
    CALL,
    CLOSE,
}
