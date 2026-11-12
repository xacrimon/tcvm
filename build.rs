fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    cfg_aliases::cfg_aliases! {
        nightly: { feature = "nightly" },
        // The JIT enters and leaves interpreter handlers in their nightly-only ABI.
        jit: { all(feature = "jit", nightly, any(target_arch = "aarch64", target_arch = "x86_64")) },
    }
}
