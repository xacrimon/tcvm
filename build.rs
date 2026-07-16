//! Emits the `jit_enabled` cfg, which is what the crate actually gates the JIT
//! on. It is set only when the `jit` feature is requested *and* the target has a
//! backend for it (aarch64 macOS or Linux). Everywhere else the JIT is compiled
//! out entirely rather than falling back to a stub, so unsupported targets carry
//! no JIT code at all.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(jit_enabled)");

    let requested = std::env::var_os("CARGO_FEATURE_JIT").is_some();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let supported = arch == "aarch64" && matches!(os.as_str(), "macos" | "linux");

    if requested && supported {
        println!("cargo::rustc-cfg=jit_enabled");
    } else if requested {
        println!(
            "cargo::warning=jit feature is enabled but there is no JIT backend for \
             {arch}-{os}; building without the JIT"
        );
    }
}
