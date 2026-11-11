//! Unwind information for code blocks: a DWARF CIE and FDE per block,
//! registered with the system unwinder, so a panic in a native or helper
//! called from compiled code unwinds through its frames to the thread's
//! start (`vm::stack::thread_start`). The registers and rules are the
//! target's (`target::unwind`).

use crate::jit::backend::target::unwind as rules;

unsafe extern "C" {
    fn __register_frame(p: *const u8);
    fn __deregister_frame(p: *const u8);
}

pub(crate) const DW_CFA_DEF_CFA: u8 = 0x0c;
pub(crate) const DW_CFA_OFFSET: u8 = 0x80;

/// A block's frame description, registered while it lives.
pub(crate) struct Fde {
    bytes: Box<[u64]>,
    fde: usize,
}

impl Fde {
    /// Describe `[start, start + len)` as code that calls out only with a
    /// frame of `frame` bytes open (`frame_rules`), as regions and
    /// trampolines do. Prologues and epilogues are described wrong: nothing
    /// unwinds from them.
    pub(crate) fn register(start: usize, len: usize, frame: u32) -> Fde {
        let mut b = Vec::new();
        b.extend([0; 4]);
        b.extend(0u32.to_le_bytes());
        b.push(1);
        b.extend(b"zR\0");
        uleb(&mut b, rules::CODE_ALIGN);
        b.push(0x78); // data alignment -8
        b.push(rules::RETURN_COLUMN);
        uleb(&mut b, 1);
        b.push(0); // DW_EH_PE_absptr
        rules::entry_rules(&mut b);
        close(&mut b, 0);
        let fde = b.len();
        b.extend([0; 4]);
        b.extend((fde as u32 + 4).to_le_bytes());
        b.extend((start as u64).to_le_bytes());
        b.extend((len as u64).to_le_bytes());
        uleb(&mut b, 0);
        rules::frame_rules(&mut b, frame);
        close(&mut b, fde);
        // libgcc's `__register_frame` reads entries up to a zero length.
        b.extend([0; 8]);
        let bytes: Box<[u64]> = b
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let f = Fde { bytes, fde };
        // SAFETY: a well-formed CIE and FDE, kept until `drop`.
        unsafe { __register_frame(f.entry()) };
        f
    }

    /// What the unwinder takes: Apple's libunwind one FDE, libgcc the table.
    fn entry(&self) -> *const u8 {
        let p = self.bytes.as_ptr().cast::<u8>();
        if cfg!(target_vendor = "apple") {
            p.wrapping_add(self.fde)
        } else {
            p
        }
    }
}

impl Drop for Fde {
    fn drop(&mut self) {
        // SAFETY: registered in `register`.
        unsafe { __deregister_frame(self.entry()) };
    }
}

pub(crate) fn uleb(b: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            b.push(byte);
            return;
        }
        b.push(byte | 0x80);
    }
}

/// Pad the entry at `start` to 8 bytes with `DW_CFA_nop`s and fill in its
/// length.
fn close(b: &mut Vec<u8>, start: usize) {
    while !(b.len() - start).is_multiple_of(8) {
        b.push(0);
    }
    let len = (b.len() - start - 4) as u32;
    b[start..start + 4].copy_from_slice(&len.to_le_bytes());
}
