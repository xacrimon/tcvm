fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    cfg_aliases::cfg_aliases! {
        jit: { all(feature = "jit", target_arch = "aarch64") },
    }
}
