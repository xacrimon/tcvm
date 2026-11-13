//! Running one program under one configuration, with a time limit.

use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar};
use std::time::Duration;

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

/// Run `argv` in `dir` with `env` added and every inherited `TCVM_*` removed; its process group
/// is killed past `secs`.
pub fn run(argv: &[&str], dir: &Path, env: &[(String, String)], secs: u32) -> Run {
    run_for(argv, dir, env, Duration::from_secs(secs as u64), None)
}

/// Stops a group of runs: the ones in flight are killed, and none starts after it.
#[derive(Default)]
pub struct Cancel {
    stopped: AtomicBool,
    groups: Mutex<Vec<i32>>,
}

impl Cancel {
    pub fn cancel(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        for &pg in self.groups.lock().expect("the runs").iter() {
            // SAFETY: killpg only sends a signal.
            unsafe { libc::killpg(pg, libc::SIGKILL) };
        }
    }

    pub fn cancelled(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }
}

/// `run` with a limit finer than seconds, and a `Cancel` that stops it.
pub fn run_for(
    argv: &[&str],
    dir: &Path,
    env: &[(String, String)],
    limit: Duration,
    cancel: Option<&Cancel>,
) -> Run {
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
    let failed = |e: String| Run {
        status: Status::Exit(-1),
        stdout: String::new(),
        stderr: e,
    };
    if aborted() || cancel.is_some_and(Cancel::cancelled) {
        return failed("aborted".into());
    }
    let child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return failed(format!("cannot run {}: {e}", argv[0])),
    };
    // The child leads its own process group: its pid is the group's.
    let pg = child.id() as i32;
    RUNNING.lock().expect("the runs").push(pg);
    if let Some(c) = cancel {
        c.groups.lock().expect("the runs").push(pg);
    }
    if aborted() || cancel.is_some_and(Cancel::cancelled) {
        // SAFETY: killpg only sends a signal.
        unsafe { libc::killpg(pg, libc::SIGKILL) };
    }
    // The limit is kept from here, not by a timer armed in the child: any `pre_exec` makes
    // `Command` fork instead of `posix_spawn`, and on macOS a fork takes the allocator's locks
    // for every thread of the process.
    let watch = Arc::new((Mutex::new(false), Condvar::new()));
    let fired = Arc::new(AtomicBool::new(false));
    let dog = {
        let (watch, fired) = (watch.clone(), fired.clone());
        std::thread::spawn(move || {
            let (done, cv) = &*watch;
            let done = done.lock().expect("the watchdog");
            let (done, _) = cv
                .wait_timeout_while(done, limit, |d| !*d)
                .expect("the watchdog");
            if !*done {
                fired.store(true, Ordering::Relaxed);
                // SAFETY: killpg only sends a signal.
                unsafe { libc::killpg(pg, libc::SIGKILL) };
            }
        })
    };
    let out = child.wait_with_output();
    *watch.0.lock().expect("the watchdog") = true;
    watch.1.notify_one();
    dog.join().expect("the watchdog");
    RUNNING.lock().expect("the runs").retain(|&p| p != pg);
    if let Some(c) = cancel {
        c.groups.lock().expect("the runs").retain(|&p| p != pg);
    }
    let out = match out {
        Ok(out) => out,
        Err(e) => return failed(format!("cannot run {}: {e}", argv[0])),
    };
    let status = match (out.status.code(), out.status.signal()) {
        (Some(c), _) => Status::Exit(c),
        _ if fired.load(Ordering::Relaxed) => Status::Timeout,
        (None, s) => Status::Signal(s.unwrap_or(0)),
    };
    Run {
        status,
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}
