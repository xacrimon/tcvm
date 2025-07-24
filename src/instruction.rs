type Register = u8;
type ConstantIndex = u16;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Instruction {
    Stop,
    Nop,
    Add {
        out: Register,
        lhs: Register,
        rhs: Register,
    },
    Load {
        out: Register,
        idx: ConstantIndex,
    },
}

impl Instruction {
    pub fn discriminant(self) -> u8 {
        unsafe { *<*const _>::from(&self).cast::<u8>() }
    }
}
