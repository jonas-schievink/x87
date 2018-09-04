/// Executes assembly instructions on the host and returns the host FPU's state
/// after the operations.
///
/// The FPU's state is restored to its original state after the instructions
/// have been executed.
#[macro_export]
//#[cfg(test)]  // FIXME apparently doesn't work with integration tests
macro_rules! run_host_asm {
    ( $($args:tt)+ ) => {{
        union X87StateUnion {
            raw: [u8; 108],
            structured: $crate::X87State,
        }

        let mut backup = X87StateUnion { raw: [0; 108] };
        let mut state = X87StateUnion { raw: [0; 108] };
        unsafe {
            asm!("fnsave $0" : "=*m"(&mut backup.raw) : : "memory" : "volatile");
            asm!($($args)+ :: "memory" : "volatile");
            asm!("fnsave $1\nfrstor $0"
                : "=*m"(&mut backup.raw), "=*m"(&mut state.raw) : : "memory" : "volatile");

            state.structured.scrub()
        }
    }};
}

/// An exact or rounded result of a computation.
#[derive(Debug)]
pub enum ExactOrRounded<T> {
    Exact(T),
    Rounded(T),
}

impl<T> ExactOrRounded<T> {
    /// Creates an exact result if `exact` is `true`, a rounded result if not.
    pub fn exact_if(value: T, exact: bool) -> Self {
        if exact {
            ExactOrRounded::Exact(value)
        } else {
            ExactOrRounded::Rounded(value)
        }
    }

    /// Returns a boolean indicating whether `self` represents an exact result.
    pub fn is_exact(&self) -> bool {
        match self {
            ExactOrRounded::Exact(_) => true,
            ExactOrRounded::Rounded(_) => false,
        }
    }

    /// Calls a closure `f` with the contained value, and returns `Rounded` if
    /// either `self` or the result returned by `f` is `Rounded`.
    ///
    /// Only returns `Exact` if all steps of a computation report that the
    /// result is exact.
    pub fn chain<F, U>(self, f: F) -> ExactOrRounded<U>
    where F: FnOnce(T) -> ExactOrRounded<U> {
        match self {
            ExactOrRounded::Exact(t) => f(t),
            ExactOrRounded::Rounded(t) => {
                ExactOrRounded::Rounded(f(t).into_inner())
            }
        }
    }

    /// Applies a closure to the inner value.
    ///
    /// The exactness of the result will be the same as for `self`.
    pub fn map<F, U>(self, f: F) -> ExactOrRounded<U>
    where F: FnOnce(T) -> U {
        match self {
            ExactOrRounded::Exact(t) => ExactOrRounded::Exact(f(t)),
            ExactOrRounded::Rounded(t) => ExactOrRounded::Rounded(f(t)),
        }
    }

    /// Extracts the inner value, disposing exactness information.
    pub fn into_inner(self) -> T {
        match self {
            ExactOrRounded::Exact(t) => t,
            ExactOrRounded::Rounded(t) => t,
        }
    }
}
