#[cfg(any(debug_assertions, test))]
use core::cell::Cell;
use core::{
    cell::UnsafeCell,
    cmp::Ordering,
    fmt,
    ops::{Deref, DerefMut},
};

#[cfg(any(debug_assertions, test))]
type BorrowFlag = isize;

/// Stripped-down `RefCell<T>` used as the cell behind `RefLock`.
///
/// Behaves like `RefCell` under debug builds and under `cfg(test)`:
/// `borrow` / `borrow_mut` track outstanding borrows in a `Cell<isize>`
/// flag and panic on aliasing. Under release-without-tests, both
/// methods skip the check entirely and rely on the caller to uphold
/// Rust's aliasing rules manually.
///
/// **Safety contract for release builds:** at any moment, the
/// outstanding borrows on a single `CheckedCell` must satisfy Rust's
/// `&` / `&mut` rules. Concretely: never call `borrow_mut` while a
/// `Ref` is live, and never call `borrow` while a `RefMut` is live.
/// VM hot paths follow a strict drop-then-reborrow discipline; tests
/// (run in debug or with `cfg(test)`) catch violations.
pub struct CheckedCell<T: ?Sized> {
    #[cfg(any(debug_assertions, test))]
    flag: Cell<BorrowFlag>,
    value: UnsafeCell<T>,
}

impl<T> CheckedCell<T> {
    #[inline]
    pub const fn new(t: T) -> Self {
        Self {
            #[cfg(any(debug_assertions, test))]
            flag: Cell::new(0),
            value: UnsafeCell::new(t),
        }
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }

    #[inline]
    pub fn take(&self) -> T
    where
        T: Default,
    {
        core::mem::take(&mut *self.borrow_mut())
    }
}

impl<T: ?Sized> CheckedCell<T> {
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.value.get()
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: `&mut self` provides exclusive access.
        unsafe { &mut *self.value.get() }
    }

    #[inline]
    #[track_caller]
    pub fn borrow(&self) -> Ref<'_, T> {
        match self.try_borrow() {
            Ok(r) => r,
            Err(_) => panic!("already mutably borrowed"),
        }
    }

    #[inline]
    pub fn try_borrow(&self) -> Result<Ref<'_, T>, BorrowError> {
        #[cfg(any(debug_assertions, test))]
        {
            let f = self.flag.get();
            if f < 0 {
                return Err(BorrowError { _private: () });
            }
            self.flag.set(f + 1);
            Ok(Ref {
                value: unsafe { &*self.value.get() },
                flag: &self.flag,
            })
        }
        #[cfg(not(any(debug_assertions, test)))]
        {
            // SAFETY: borrow checking is disabled in release-without-tests;
            // callers must ensure no `RefMut` is outstanding. Debug and
            // test builds verify this via the flag.
            Ok(Ref {
                value: unsafe { &*self.value.get() },
            })
        }
    }

    #[inline]
    #[track_caller]
    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        match self.try_borrow_mut() {
            Ok(r) => r,
            Err(_) => panic!("already borrowed"),
        }
    }

    #[inline]
    pub fn try_borrow_mut(&self) -> Result<RefMut<'_, T>, BorrowMutError> {
        #[cfg(any(debug_assertions, test))]
        {
            let f = self.flag.get();
            if f != 0 {
                return Err(BorrowMutError { _private: () });
            }
            self.flag.set(-1);
            Ok(RefMut {
                value: unsafe { &mut *self.value.get() },
                flag: &self.flag,
            })
        }
        #[cfg(not(any(debug_assertions, test)))]
        {
            // SAFETY: borrow checking is disabled in release-without-tests;
            // callers must ensure no other borrows exist. Debug and test
            // builds verify this via the flag.
            Ok(RefMut {
                value: unsafe { &mut *self.value.get() },
            })
        }
    }
}

impl<T: Default> Default for CheckedCell<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone> Clone for CheckedCell<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.borrow().clone())
    }
}

impl<T: ?Sized + PartialEq> PartialEq for CheckedCell<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self.borrow() == *other.borrow()
    }
}

impl<T: ?Sized + Eq> Eq for CheckedCell<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for CheckedCell<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.borrow().partial_cmp(&*other.borrow())
    }
}

impl<T: ?Sized + Ord> Ord for CheckedCell<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.borrow().cmp(&*other.borrow())
    }
}

impl<T> From<T> for CheckedCell<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

pub struct BorrowError {
    _private: (),
}

impl fmt::Debug for BorrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowError").finish()
    }
}

impl fmt::Display for BorrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("already mutably borrowed")
    }
}

pub struct BorrowMutError {
    _private: (),
}

impl fmt::Debug for BorrowMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowMutError").finish()
    }
}

impl fmt::Display for BorrowMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("already borrowed")
    }
}

pub struct Ref<'b, T: ?Sized + 'b> {
    value: &'b T,
    #[cfg(any(debug_assertions, test))]
    flag: &'b Cell<BorrowFlag>,
}

impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

#[cfg(any(debug_assertions, test))]
impl<T: ?Sized> Drop for Ref<'_, T> {
    #[inline]
    fn drop(&mut self) {
        let f = self.flag.get();
        debug_assert!(f > 0);
        self.flag.set(f - 1);
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.value, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.value, f)
    }
}

pub struct RefMut<'b, T: ?Sized + 'b> {
    value: &'b mut T,
    #[cfg(any(debug_assertions, test))]
    flag: &'b Cell<BorrowFlag>,
}

impl<T: ?Sized> Deref for RefMut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: ?Sized> DerefMut for RefMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

#[cfg(any(debug_assertions, test))]
impl<T: ?Sized> Drop for RefMut<'_, T> {
    #[inline]
    fn drop(&mut self) {
        debug_assert_eq!(self.flag.get(), -1);
        self.flag.set(0);
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RefMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.value, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for RefMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.value, f)
    }
}
