//! Running one program under one configuration, with a time limit.

use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Command, Stdio};

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
    let out = match cmd.output() {
        Ok(out) => out,
        Err(e) => {
            return Run {
                status: Status::Exit(-1),
                stdout: String::new(),
                stderr: format!("cannot run {}: {e}", argv[0]),
            };
        }
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
