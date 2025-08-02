type Register = u8;
type UpvalueIndex = u8;
type ConstantIndex = u16;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Instruction {
    MOVE {
        dst: Register,
        src: Register,
    },
    LOADK {
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

    GETTABUP,
    GETTABLE,
    GETI,
    GETFIELD,

    SETTABUP,
    SETTABLE,
    SETI,
    SETFIELD,

    NEWTABLE,

    SELF,

    ADDI,

    ADDK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    SUBK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    MULK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    MODK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    POWK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    DIVK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    IDIVK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },

    BANDK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    BORK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },
    BXORK {
        dst: Register,
        lhs: Register,
        rhs: ConstantIndex,
    },

    SHRI,
    SHLI,

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

    MMBIN,
    MMBINI,
    MMBINK,

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
        start: Register,
        end: Register,
    },

    CLOSE {
        start: Register,
    },
    TBC {
        val: Register,
    },
    JMP,
    EQ,
    LT,
    LE,

    EQK,
    EQI,
    LTI,
    LEI,
    GTI,
    GEI,

    TEST,
    TESTSET,

    CALL,
    TAILCALL,

    RETURN,
    RETURN0,
    RETURN1,

    FORLOOP,
    FORPREP,

    TFORPREP,
    TFORCALL,
    TFORLOOP,

    SETLIST,

    CLOSURE,

    VARARG,

    VARARGPREP,

    NOP,

    STOP,
}

impl Instruction {
    pub fn discriminant(self) -> u8 {
        unsafe { *<*const _>::from(&self).cast::<u8>() }
    }
}
