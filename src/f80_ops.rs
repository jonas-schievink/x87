//! Arithmetic operation implementations for `f80`.

use ::f80;

use std::ops;

impl ops::Neg for f80 {
    type Output = f80;

    fn neg(self) -> f80 {
        f80(self.0 ^ 0x8000_0000_0000_0000_0000)
    }
}

impl<'a> ops::Neg for &'a f80 {
    type Output = f80;

    fn neg(self) -> f80 {
        -(*self)
    }
}
// (tested via proptest)
