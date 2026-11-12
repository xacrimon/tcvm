//! Running one program under one configuration, with a time limit.

use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Exit(i32),
    Signal(i32),
    Timeout,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Exit(c) => write!(f, "exit {c}"),
            Status::Signal(s) => write!(f, "signal {s}"),
            Status::Timeout => write!(f, "timeout"),
        }
    }
}

/// The process groups of the runs in flight, so a stop can kill them.
static RUNNING: Mutex<Vec<i32>> = Mutex::new(Vec::new());
/// Set once a stop killed the runs in flight: no run starts after it, and what the killed ones
/// returned means nothing.
static ABORTED: AtomicBool = AtomicBool::new(false);

/// Kill every run in flight and start no other.
pub fn abort() {
    ABORTED.store(true, Ordering::Relaxed);
    for &pg in RUNNING.lock().expect("the runs").iter() {
        // SAFETY: killpg only sends a signal.
        unsafe { libc::killpg(pg, libc::SIGKILL) };
    }
}

pub fn aborted() -> bool {
    ABORTED.load(Ordering::Relaxed)
}

pub struct Run {
    pub status: Status,
    pub stdout: String,
    pub stderr: String,
}

impl Run {
    pub fn ok(&self) -> bool {
        self.status == Status::Exit(0)
    }
}

/// Run `argv` in `dir` with `env` added and every inherited `TCVM_*` removed; it dies of
/// SIGALRM past `secs`.
pub fn run(argv: &[&str], dir: &Path, env: &[(String, String)], secs: u32) -> Run {
    let mut cmd = Command::new(argv[0]);
    // Own process group: a terminal Ctrl-C reaches only the fuzzer, so the runs in flight
    // finish instead of being recorded as SIGINT crashes.
    cmd.args(&argv[1..])
        .current_dir(dir)
        .stdin(Stdio::null())
        .process_group(0);
    for (k, _) in std::env::vars_os() {
        if k.to_string_lossy().starts_with("TCVM_") {
            cmd.env_remove(k);
        }
    }
    cmd.envs(env.iter().map(|(k, v)| (k, v)));
    // SAFETY: `alarm` is async-signal-safe, and an alarm survives `exec`.
    unsafe {
        cmd.pre_exec(move || {
            libc::alarm(secs);
            Ok(())
        });
    }
    let failed = |e: String| Run {
        status: Status::Exit(-1),
        stdout: String::new(),
        stderr: e,
    };
    if aborted() {
        return failed("aborted".into());
    }
    let child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return failed(format!("cannot run {}: {e}", argv[0])),
    };
    // The child leads its own process group: its pid is the group's.
    let pg = child.id() as i32;
    RUNNING.lock().expect("the runs").push(pg);
    if aborted() {
        // SAFETY: killpg only sends a signal.
        unsafe { libc::killpg(pg, libc::SIGKILL) };
    }
    let out = child.wait_with_output();
    RUNNING.lock().expect("the runs").retain(|&p| p != pg);
    let out = match out {
        Ok(out) => out,
        Err(e) => return failed(format!("cannot run {}: {e}", argv[0])),
    };
    let status = match (out.status.code(), out.status.signal()) {
        (Some(c), _) => Status::Exit(c),
        (None, Some(libc::SIGALRM)) => Status::Timeout,
        (None, s) => Status::Signal(s.unwrap_or(0)),
    };
    Run {
        status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}
