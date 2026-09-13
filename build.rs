//! Emits the `jit_enabled` cfg, which is what the crate actually gates the JIT
//! on. It is set only when the `jit` feature is requested *and* the target has a
//! backend for it (aarch64 macOS or Linux, or x86-64 macOS or Linux). Everywhere
//! else the JIT is compiled out entirely rather than falling back to a stub, so
//! unsupported targets carry no JIT code at all.
//!
//! x86-64 macOS and x86-64 Linux share the System V AMD64 ABI, so one backend
//! serves both; the macOS build also lets the x86-64 backend be exercised under
//! Rosetta 2 on an Apple Silicon host.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(jit_enabled)");

    let requested = std::env::var_os("CARGO_FEATURE_JIT").is_some();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let supported = match (arch.as_str(), os.as_str()) {
        ("aarch64", "macos" | "linux") => true,
        ("x86_64", "macos" | "linux") => true,
        _ => false,
    };

    if requested && supported {
        println!("cargo::rustc-cfg=jit_enabled");
    } else if requested {
        println!(
            "cargo::warning=jit feature is enabled but there is no JIT backend for \
             {arch}-{os}; building without the JIT"
        );
    }
}
