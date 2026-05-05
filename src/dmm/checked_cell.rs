use core::{
    cell::{Cell, UnsafeCell},
    cmp::Ordering,
    fmt,
    ops::{Deref, DerefMut},
};

#[cfg(debug_assertions)]
type BorrowFlag = isize;

pub struct CheckedCell<T: ?Sized> {
    #[cfg(debug_assertions)]
    flag: Cell<BorrowFlag>,
    value: UnsafeCell<T>,
}

impl<T> CheckedCell<T> {
    #[inline]
    pub const fn new(t: T) -> Self {
        Self {
            #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
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
        #[cfg(not(debug_assertions))]
        {
            // SAFETY: borrow checking is disabled in release; callers must
            // ensure no `RefMut` is outstanding. Debug builds verify this.
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
        #[cfg(debug_assertions)]
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
        #[cfg(not(debug_assertions))]
        {
            // SAFETY: borrow checking is disabled in release; callers must
            // ensure no other borrows exist. Debug builds verify this.
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
    #[cfg(debug_assertions)]
    flag: &'b Cell<BorrowFlag>,
}

impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

#[cfg(debug_assertions)]
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
    #[cfg(debug_assertions)]
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

#[cfg(debug_assertions)]
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
