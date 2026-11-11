fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    cfg_aliases::cfg_aliases! {
        jit: { all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")) },
    }
}
