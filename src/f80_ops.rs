//! Implementations of `std::ops` for `f80`.
//!
//! Note that this only contains trivial algorithms (such as flipping the sign)
//! and forwards to the impls in the `f80_algo` module.
//!
//! All implementations of standard Rust operators (in `std::ops`) use
//! round-to-nearest-ties-to-even mode and never cause exceptions.

use {f80, Classified, RoundingMode};

use std::ops;

impl ops::Neg for f80 {
    type Output = f80;

    fn neg(self) -> f80 {
        f80::from_bits(self.to_bits() ^ 0x8000_0000_0000_0000_0000)
    }
}

impl<'a> ops::Neg for &'a f80 {
    type Output = f80;

    fn neg(self) -> f80 {
        -(*self)
    }
}
// (tested via proptest)

impl PartialEq for f80 {
    fn eq(&self, other: &Self) -> bool {
        match (self.classify(), other.classify()) {
            (Classified::NaN {..}, _) => false,
            (_, Classified::NaN {..}) => false,

            // -0.0 == -0.0 == 0.0 == 0.0
            (Classified::Zero { .. }, Classified::Zero { .. }) => true,
            (Classified::Zero { .. }, _) => false,
            (_, Classified::Zero { .. }) => false,
            (Classified::Inf {sign: lsign}, Classified::Inf {sign: rsign}) => lsign == rsign,

            (lhs, rhs) => {
                // FIXME: normalization needed?
                lhs.decompose().unwrap() == rhs.decompose().unwrap()
            }
        }
    }
}

impl ops::Add for f80 {
    type Output = f80;

    fn add(self, rhs: f80) -> f80 {
        self.add_checked(rhs, RoundingMode::default()).into_inner()
    }
}

impl ops::Sub for f80 {
    type Output = f80;

    fn sub(self, rhs: f80) -> f80 {
        self.sub_checked(rhs, RoundingMode::default()).into_inner()
    }
}

impl ops::Mul for f80 {
    type Output = f80;

    fn mul(self, rhs: f80) -> f80 {
        self.mul_checked(rhs, RoundingMode::default()).into_inner()
    }
}
