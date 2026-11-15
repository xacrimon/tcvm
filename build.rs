fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    cfg_aliases::cfg_aliases! {
        nightly: { feature = "nightly" },
        // The JIT enters and leaves interpreter handlers in their nightly-only ABI.
        // aarch64 alone while the compiler is rebuilt: x86-64 selection and
        // its encoder come back once the lowered IR is stable.
        jit: { all(feature = "jit", nightly, target_arch = "aarch64") },
    }
}
