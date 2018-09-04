use std::ops::{Add, Sub, Neg, Mul};
use std::cmp::Ordering;
use std::fmt;

/// An integer represented as sign and magnitude.
///
/// This can represent signed zero and is useful for implementing floating-point
/// operations.
#[derive(Copy, Clone)]
pub struct SignMagnitude {
    sign: bool,
    magnitude: u128,
}

impl SignMagnitude {
    pub fn new(sign: bool, magnitude: u128) -> Self {
        Self { sign, magnitude }
    }

    pub fn sign(&self) -> bool {
        self.sign
    }

    pub fn magnitude(&self) -> u128 {
        self.magnitude
    }
}

/// Compares two `SignMagnitude` values for equality.
///
/// This considers `-0` and `+0` to be equal.
impl PartialEq for SignMagnitude {
    fn eq(&self, other: &Self) -> bool {
        if self.magnitude == 0 && other.magnitude == 0 {
            true
        } else {
            self.sign == other.sign && self.magnitude == other.magnitude
        }
    }
}

impl Eq for SignMagnitude {}

impl PartialOrd for SignMagnitude {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SignMagnitude {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.magnitude == 0 && other.magnitude == 0 {
            return Ordering::Equal;
        }

        match (self.sign, other.sign) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => self.magnitude.cmp(&other.magnitude).reverse(),
            (false, false) => self.magnitude.cmp(&other.magnitude),
        }
    }
}

impl Add for SignMagnitude {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if self.sign == rhs.sign {
            Self {
                sign: self.sign,
                magnitude: self.magnitude + rhs.magnitude,
            }
        } else {
            let (sign, mag) = match (self, rhs) {
                (l, r) if l.magnitude < r.magnitude => {
                    (r.sign, r.magnitude - l.magnitude)
                }
                (l, r) if l.magnitude > r.magnitude => {
                    (l.sign, l.magnitude - r.magnitude)
                }
                _ => {  // equal magnitudes
                    // make sure the sign isn't getting set (we already know
                    // l and r have opposite signs)
                    (false, 0)
                }
            };

            Self {
                sign,
                magnitude: mag,
            }
        }
    }
}

impl Sub for SignMagnitude {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl Mul for SignMagnitude {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let sign = self.sign != rhs.sign;
        Self {
            sign,
            magnitude: self.magnitude * rhs.magnitude,
        }
    }
}

impl Neg for SignMagnitude {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            sign: !self.sign,
            magnitude: self.magnitude,
        }
    }
}

macro_rules! from_unsigned {
    ($t:ty) => {
        impl From<$t> for SignMagnitude {
            fn from(v: $t) -> Self {
                Self {
                    sign: false,
                    magnitude: u128::from(v),
                }
            }
        }
    };
}

macro_rules! from_signed {
    ($t:ty) => {
        impl From<$t> for SignMagnitude {
            fn from(v: $t) -> Self {
                Self {
                    sign: v < 0,
                    magnitude: v.abs() as u128,
                }
            }
        }
    };
}

from_unsigned!(u8);
from_unsigned!(u16);
from_unsigned!(u32);
from_unsigned!(u64);
from_unsigned!(u128);
from_signed!(i8);
from_signed!(i16);
from_signed!(i32);
from_signed!(i64);

impl fmt::Display for SignMagnitude {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.sign {
            f.write_str("-")?;
        }

        self.magnitude.fmt(f)
    }
}

impl fmt::Debug for SignMagnitude {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}
