//! Guard rails of the redesign (`.claude/docs/jit-redesign.md`, section 13).
//! The lists here are the closed lists: an op, a machine opcode or a helper
//! is added by editing its list, and a known violation only ever leaves.

use crate::jit::ir::ops::{Dialect, HELPERS, OPS};

fn words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

const OPS_BOTH: &str = "\
    K32 K64 KF64 KObj Load Store LoadIdx StoreIdx Base Thread SlotLoad SlotStore \
    Add Sub Mul And Or Xor Not Neg Shl Shr Sar SDiv UDiv ZExt32 SExt32 \
    Trunc32 Cmp Select AddOvf SubOvf MulOvf NegOvf FAdd FSub FMul FDiv FNeg \
    FAbs FSqrt FFloor FCeil FCmp IToF LToF FToI FToL BoxI32 BoxF64 BoxPtr \
    UnboxF64 UnboxPtr Guard GuardNot Proved Deopt LazyExit FrameState \
    Param Jump Br Ret CallHelper";
const OPS_HIGH: &str = "\
    CheckType IsType BoxI64 UnboxI64 ToF64 DivFloor ModFloor LuaShl LuaShr \
    FToIExact FToLExact LToI AbsOvf FIDiv FMod Barrier NewTable GcCheck Arith Unm Compare Truthy \
    GetIndex SetIndex GetField SetField Concat Len ForStart ForStep ForLimit ForSkip \
    ForMore ForNext";
const OPS_LOW: &str = "";

#[test]
fn ir_ops_are_the_listed_ones() {
    for (d, list) in [
        (Dialect::Both, OPS_BOTH),
        (Dialect::H, OPS_HIGH),
        (Dialect::L, OPS_LOW),
    ] {
        let table: Vec<&str> = OPS
            .iter()
            .filter(|o| o.dialect == d)
            .map(|o| o.name)
            .collect();
        assert_eq!(table, words(list), "{d:?}");
    }
}

const HELPER_LIST: &str = "\
    FMod Sin Cos Tan Asin Acos Exp Log Log2 Log10 Pow Atan2 LibmFmod \
    ArithCoerce Unm BNot Len Concat UpvalBarrier Barrier Closure ClosureOf \
    Close BoxInt NewTable SetList VarargGet VarargTable TforNext TforIpairs TopArgs \
    VarargAll Mm IndexFn NewIndexFn Index GetStr SetStr NewIndex Type \
    GetMetatable SetMetatable RawGet RawSet RawEqual RawLen ToString \
    ToNumber StrSub StrByte TInsert TRemove";

#[test]
fn helpers_are_the_listed_ones() {
    let table: Vec<&str> = HELPERS.iter().map(|h| h.name).collect();
    assert_eq!(table, words(HELPER_LIST));
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::words;
    use crate::jit::backend::aarch64::inst::ROWS;

    /// The frame's open and close; everything else is one instruction, or
    /// one of selection's and the encoder's agreed cases (section 8.3).
    const PSEUDO: &str = "Entry Ret";
    const MOP_LIST: &str = "\
        Entry MovImm MovWide FImm LdrSlot StrSlot Ldr Str LdrIdx StrIdx MovBase \
        MovThread Uxtw Cset Csel AndRR EorRR SdivRR Sxtw NegR MvnR NegsR FNeg Frintm \
        OrrRR AndImm AndWide OrrImm OrrWide TstImm TstWide AddRR SubRR MulRR \
        AddImm SubImm AddsRR SubsRR AddsImm SubsImm CmpRR CmpImm CmnImm CmpSxtw \
        Fcmp LslRR LsrRR AsrRR LslImm LsrImm AsrImm Smull Scvtf Fcvtzs FmovToGpr FmovFromGpr FAdd FSub \
        FMul FDiv B BCond BCondExit BExit Call Ret";

    #[test]
    fn machine_opcodes_are_the_listed_ones() {
        let table: Vec<&str> = ROWS.iter().map(|m| m.name).collect();
        assert_eq!(table, words(MOP_LIST));
        let pseudo: Vec<&str> = (ROWS.iter().filter(|m| m.pseudo)).map(|m| m.name).collect();
        assert_eq!(pseudo, words(PSEUDO));
    }

    /// Named in any arm: the pinned registers.
    const PINNED: [&str; 2] = ["BASE", "THREAD"];
    /// Named in the frame pseudo-instructions' arms.
    const FRAME: [&str; 3] = ["SP", "FP", "LR"];
    /// The registers the encoder names at the base, by arm, each with the
    /// step that removes it. Nothing is added here.
    const KNOWN: [(&str, &str); 3] = [
        // Step 4: the first result, the count and the results pointer.
        ("Ret", "Gpr("),
        ("Ret", "INSN"),
        ("Ret", "PC"),
    ];

    fn is_register(w: &str) -> bool {
        let numbered = |p: char| {
            w.strip_prefix(p)
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        };
        matches!(
            w,
            "ZR" | "SP" | "FP" | "LR" | "BASE" | "THREAD" | "RT" | "CLOSURE" | "INSN" | "PC"
        ) || numbered('X')
            || numbered('D')
    }

    #[test]
    fn encoder_names_no_register() {
        let src = include_str!("backend/aarch64/emit/inst.rs");
        let mut arm = "";
        let mut found: Vec<(&str, &str)> = Vec::new();
        for line in src.lines() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("use ") {
                continue;
            }
            if let Some(rest) = t.strip_prefix("MOp::") {
                let end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric())
                    .unwrap_or(rest.len());
                arm = &rest[..end];
            }
            let code = t.split("//").next().unwrap();
            let made = ["Gpr(", "Fpr("].into_iter().filter(|c| code.contains(c));
            let named = code
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .filter(|w| is_register(w));
            for w in made.chain(named) {
                let frame = matches!(arm, "Entry" | "Ret") && FRAME.contains(&w);
                if !PINNED.contains(&w) && !frame && !found.contains(&(arm, w)) {
                    found.push((arm, w));
                }
            }
        }
        found.sort();
        let mut known = KNOWN.to_vec();
        known.sort();
        assert_eq!(found, known);
    }
}
