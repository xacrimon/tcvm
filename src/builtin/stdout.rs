//! The process's stdout buffer, shared by `print` and the `io` library.

use std::io::Write;
use std::sync::{Mutex, MutexGuard, Once, PoisonError};

/// Bytes held back from a writer per a `setvbuf` mode.
pub(crate) struct WBuf {
    buf: Vec<u8>,
    pub(crate) mode: BufMode,
}

/// `setvbuf`'s modes; the sizes are the buffer's capacity.
#[derive(Clone, Copy)]
pub(crate) enum BufMode {
    No,
    Line(usize),
    Full(usize),
}

/// A new file's buffer, like stdio's default full buffering.
pub(crate) const DEFAULT_BUF_SIZE: usize = 8192;

impl WBuf {
    pub(crate) const fn new(mode: BufMode) -> Self {
        WBuf {
            buf: Vec::new(),
            mode,
        }
    }

    pub(crate) fn flush(&mut self, w: &mut impl Write) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let (res, _) = count_write(w, &self.buf);
        // Dropped even on failure, so one error isn't reported by every
        // later operation.
        self.buf.clear();
        res
    }

    /// Buffer `buf`, or write it through to `w` once it no longer fits. The
    /// count is the bytes of `buf` accepted before an error.
    #[inline]
    pub(crate) fn write(&mut self, w: &mut impl Write, buf: &[u8]) -> (std::io::Result<()>, u64) {
        if let BufMode::Full(cap) = self.mode
            && self.buf.len() + buf.len() <= cap
        {
            self.buf.extend_from_slice(buf);
            return (Ok(()), buf.len() as u64);
        }
        self.write_through(w, buf)
    }

    #[inline(never)]
    fn write_through(&mut self, w: &mut impl Write, buf: &[u8]) -> (std::io::Result<()>, u64) {
        let (cap, line) = match self.mode {
            BufMode::No => (0, false),
            BufMode::Line(cap) => (cap, true),
            BufMode::Full(cap) => (cap, false),
        };
        // Line mode sends everything through the last newline and holds back
        // the rest, as macOS stdio does.
        let head_len = match line {
            true => buf.iter().rposition(|&b| b == b'\n').map_or(0, |nl| nl + 1),
            false => 0,
        };
        let (head, tail) = buf.split_at(head_len);
        if self.buf.len() + buf.len() > cap
            && let Err(e) = self.flush(w)
        {
            return (Err(e), 0);
        }
        if !head.is_empty() {
            if self.buf.is_empty() {
                let (res, n) = count_write(w, head);
                if res.is_err() {
                    return (res, n);
                }
            } else {
                // Not flushed above, so `buf` fits: one write for both.
                self.buf.extend_from_slice(head);
                if let Err(e) = self.flush(w) {
                    return (Err(e), 0);
                }
            }
        }
        if self.buf.len() + tail.len() > cap {
            let (res, n) = count_write(w, tail);
            return (res, head_len as u64 + n);
        }
        self.buf.extend_from_slice(tail);
        (Ok(()), buf.len() as u64)
    }
}

/// The process's one stdout buffer, which `print`, `io.write` and every
/// `io.stdout` handle share, as C's `stdout`: line-buffered on a terminal,
/// fully buffered otherwise, and drained at exit.
static STDOUT: Mutex<WBuf> = Mutex::new(WBuf::new(BufMode::No));

/// Fd 1, unbuffered.
struct Fd1;

impl Write for Fd1 {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY: `buf` is valid for `buf.len()` bytes.
        let n = unsafe { libc::write(1, buf.as_ptr().cast(), buf.len()) };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(n as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) struct StdoutLock(MutexGuard<'static, WBuf>);

/// Stdout's buffer, picking its mode on first use.
pub(crate) fn stdout() -> StdoutLock {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: plain libc calls; `drain_stdout` only touches `STDOUT`.
        let tty = unsafe { libc::isatty(1) } == 1;
        unsafe { libc::atexit(drain_stdout) };
        STDOUT.lock().unwrap_or_else(PoisonError::into_inner).mode = match tty {
            true => BufMode::Line(DEFAULT_BUF_SIZE),
            false => BufMode::Full(DEFAULT_BUF_SIZE),
        };
    });
    StdoutLock(STDOUT.lock().unwrap_or_else(PoisonError::into_inner))
}

/// Covers `exit` from anywhere, `os.exit` included. A lock held at exit
/// (another thread mid-write) is left alone rather than waited on.
extern "C" fn drain_stdout() {
    let mut buf = match STDOUT.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    let _ = buf.flush(&mut Fd1);
}

impl StdoutLock {
    pub(crate) fn write_counted(&mut self, buf: &[u8]) -> (std::io::Result<()>, u64) {
        self.0.write(&mut Fd1, buf)
    }

    pub(crate) fn set_mode(&mut self, mode: BufMode) -> std::io::Result<()> {
        self.0.mode = mode;
        self.0.flush(&mut Fd1)
    }

    /// Before reading stdin, as stdio does, so a prompt shows on a terminal.
    pub(crate) fn flush_if_interactive(&mut self) -> std::io::Result<()> {
        match self.0.mode {
            BufMode::Full(_) => Ok(()),
            _ => self.0.flush(&mut Fd1),
        }
    }
}

impl Write for StdoutLock {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_counted(buf).0.map(|()| buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush(&mut Fd1)
    }
}

/// `write_all`, but reporting the byte count reached so a failure can surface it
/// (the count is 0 for an immediate `EBADF`, the common forced-failure case).
pub(crate) fn count_write<W: Write>(w: &mut W, buf: &[u8]) -> (std::io::Result<()>, u64) {
    let mut written = 0u64;
    while (written as usize) < buf.len() {
        match w.write(&buf[written as usize..]) {
            Ok(0) => {
                return (
                    Err(std::io::Error::from(std::io::ErrorKind::WriteZero)),
                    written,
                );
            }
            Ok(n) => written += n as u64,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return (Err(e), written),
        }
    }
    (Ok(()), written)
}
