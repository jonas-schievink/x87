//! Provides a decomposed representation of finite (non-NaN, non-Inf)
//! floating-point numbers.

use sign_mag::SignMagnitude;
use std::fmt;
use f80_mod::FloatResult;

/// A normalized or denormal `f80` decomposed into its components (may be zero).
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Decomposed {
    pub sign: bool,
    exponent: i16,
    /// The fixed-point significand with additional guard, round and sticky
    /// bits.
    ///
    /// The layout of this value looks like this:
    ///
    /// ```notrust
    ///       +---------+----------+-------+-------+--------+
    /// Bits: | 127-66  |   65-3   |   2   |   1   |    0   |
    /// #Bits:|   62    |    63    |   1   |   1   |    1   |
    /// What: | Integer | Fraction | guard | round | sticky |
    ///       +---------+----------+-------+-------+--------+
    /// ```
    significand: u128,
}

impl Decomposed {
    const LEADING_ZEROS_WHEN_NORMALIZED: u32 = 64 - 3;  // 3 for the GRS bits

    pub fn zero() -> Self {
        Self {
            sign: false,
            exponent: 0,
            significand: 0,
        }
    }

    /// Creates a new `Decomposed` float, representing a finite number.
    ///
    /// The number represented is:
    ///
    /// ```notrust
    /// (-sign) * significand * 2^exponent
    /// ```
    ///
    /// The most significant bit of the `significand` is the integer bit, the
    /// other 63 bits are fractional bits.
    pub fn new(sign: bool, exponent: i16, significand: u64) -> Self {
        Self {
            sign,
            exponent,
            significand: u128::from(significand) << 3,
        }
    }

    /// Creates a decomposed float from a sign-magnitude fixed-point value and
    /// an exponent.
    ///
    /// The `sign_mag` value defines the sign and the significand of the float.
    /// The significand is the magnitude of the value and also contains
    /// additional bits used for correct rounding: The least significant 3 bits
    /// are the GRS bits (guard, round, sticky), the 63 bits above those are the
    /// fraction bits, and the bits above those are the integer portion of the
    /// number.
    pub fn with_sign_magnitude_exponent(sign_mag: SignMagnitude, exponent: i16) -> Self {
        // FIXME using all bits not possible when we have the 3 rounding bits
        // (what happens when multiplying large values?)
        Self {
            sign: sign_mag.sign(),
            exponent,
            significand: sign_mag.magnitude(),  // FIXME is GRS bit handling right?
        }
    }

    pub fn exponent(&self) -> i16 {
        self.exponent
    }

    /// Returns `true` if `self` encodes exactly `0.0` or `-0.0`.
    ///
    /// If the significand would be rounded to 0, but isn't exactly 0 when
    /// taking the excess bits into account, this will return `false`.
    pub fn is_zero(&self) -> bool {
        self.significand == 0
    }

    /// Returns the sign and significand as a `SignMagnitude` value.
    pub fn to_sign_magnitude(&self) -> SignMagnitude {
        // FIXME should GRS bits be included here?
        SignMagnitude::new(self.sign, self.significand)
    }

    /// Normalizes the value, adjusting the significand so that bit #63 (the
    /// canonical integer bit) is set.
    ///
    /// This might shift 1-bits out of the significand, in which case a rounded
    /// result is returned. It will also adjust the exponent to compensate for
    /// the shifted significand.
    ///
    /// If `self` is zero this does nothing as no normalization is necessary.
    pub fn normalize(&self) -> FloatResult<Self> {
        let mut normalized = *self;

        if self.significand == 0 {
            // Value is zero. Make the exponent sane (since it doesn't matter)
            // and return.
            normalized.exponent = 0;
            return FloatResult::Exact(normalized);
        }

        // If normalized, we want bit #63 to be set, and all higher-valued bits
        // to be unset. So we want there to be exactly 64 leading zeros.
        let leading_zeros = self.significand.leading_zeros();
        if leading_zeros > Self::LEADING_ZEROS_WHEN_NORMALIZED {
            // Too many zeros on the left, shift first 1-bit to the integer pos.
            // This doesn't lose any bits (ie. doesn't round).
            let shift = leading_zeros - Self::LEADING_ZEROS_WHEN_NORMALIZED;
            normalized.exponent -= shift as i16;
            normalized.significand = self.significand << shift; // FIXME shifts all GRS bits
            assert!(normalized.is_normalized());

            FloatResult::Exact(normalized)
        } else if leading_zeros < Self::LEADING_ZEROS_WHEN_NORMALIZED {
            // There's a 1-bit too far to the left, shift it into the integer
            // bit position.
            let shift = Self::LEADING_ZEROS_WHEN_NORMALIZED - leading_zeros;
            normalized.exponent += shift as i16;
            normalized.significand = self.significand >> shift; // FIXME sticky
            assert!(normalized.is_normalized());

            if self.significand == normalized.significand << shift { // FIXME uuuh un-sticky or sth?
                FloatResult::Exact(normalized)
            } else {
                // Lost bits in the process
                // FIXME: proper rounding?
                FloatResult::Rounded(normalized)
            }
        } else {
            // == 64 -> already normalized
            FloatResult::Exact(normalized)
        }
    }

    /// Adjust exponent and significant so that the exponent equals the given
    /// `exponent` while `self` still refers to the same number.
    pub fn adjust_exponent_to(&self, exponent: i16) -> FloatResult<Self> {
        let mut adj = *self;
        adj.exponent = exponent;

        if exponent < self.exponent {
            // smaller exponent, need to shift significand left to adjust
            let shift = self.exponent - exponent;
            adj.significand = if shift > 127 { 0 } else { self.significand << shift };
            let back = if shift > 127 { 0 } else { adj.significand >> shift };
            trace!(
                "adj_exponent: prev exp={}; new exp={}; left by << {}; {:?} -> {:?}",
                self.exponent, exponent, shift, self, adj
            );

            if self.significand == back {
                FloatResult::Exact(adj)
            } else {
                // FIXME should this be `TooLarge`, logically?
                FloatResult::Rounded(adj)
            }
        } else if exponent > self.exponent {
            // larger exponent, need to shift significand right to adjust
            let shift = exponent - self.exponent;
            adj.significand = if shift > 127 { 0 } else { self.significand >> shift };
            let back = if shift > 127 { 0 } else { adj.significand << shift };
            trace!(
                "adj_exponent: prev exp={}; new exp={}; right by >> {}; {:?} -> {:?}",
                self.exponent, exponent, shift, self, adj
            );

            if self.significand == back {
                FloatResult::Exact(adj)
            } else {
                FloatResult::Rounded(adj)
            }
        } else {
            // already at the target exponent
            FloatResult::Exact(adj)
        }
    }

    fn is_normalized(&self) -> bool {
        self.significand.leading_zeros() == Self::LEADING_ZEROS_WHEN_NORMALIZED
    }

    pub fn as_f32_significand(&self) -> FloatResult<u32> {
        // 23 fraction bits
        let result = self.reduced_fraction(23);
        let exact = result.is_exact();
        let result = result.unwrap_exact_or_rounded() as u32;
        if exact {
            FloatResult::Exact(result)
        } else {
            FloatResult::Rounded(result)
        }
    }

    fn as_f64_significand(&self) -> FloatResult<u64> {
        // 52 fraction bits
        let result = self.reduced_fraction(52);
        let exact = result.is_exact();
        let result = result.unwrap_exact_or_rounded();
        if exact {
            FloatResult::Exact(result)
        } else {
            FloatResult::Rounded(result)
        }
    }

    pub fn as_f80_fraction(&self) -> FloatResult<u64> {
        // 63 fraction bits
        let result = self.reduced_fraction(63);
        let exact = result.is_exact();
        let result = result.unwrap_exact_or_rounded();
        if exact {
            FloatResult::Exact(result)
        } else {
            FloatResult::Rounded(result)
        }
    }

    pub fn as_f80_significant(&self) -> FloatResult<u64> {
        let fraction = self.as_f80_fraction();
        let exact = fraction.is_exact();
        let fraction = fraction.unwrap_exact_or_rounded();
        if self.integer_bits() > 1 {
            if self.sign {
                return FloatResult::TooSmall;
            } else {
                return FloatResult::TooLarge;
            }
        }
        let int = self.integer_bits() == 1;
        let significand = if int { (1 << 63) } else { 0 };
        if exact {
            FloatResult::Exact(significand | fraction)
        } else {
            FloatResult::Rounded(significand | fraction)
        }
    }

    /// Rounds the fraction bits to get the given number of fraction bits.
    ///
    /// This is used for converting the extended significand (with guard, round
    /// and sticky bits) to the significand to use in the resulting `f80`. It
    /// can also produce smaller outputs for use in `f32` or `f64`.
    ///
    /// Note that this will not return any integer bits.
    fn reduced_fraction(&self, bits: u8) -> FloatResult<u64> {
        assert!(bits <= 64, "too many bits for a u64");
        // f80 has 63 fraction bits, we have more for the overflow calculations

        let raw_fraction = (self.significand & (0x7fff_ffff_ffff_ffff << 3)) >> 3;  // clears GRS
        let truncated = raw_fraction >> (63 - bits);

        // TODO rounding
        if truncated << (63 - bits) == raw_fraction {
            FloatResult::Exact(truncated as u64)
        } else {
            FloatResult::Rounded(truncated as u64)
        }
    }

    fn integer_bits(&self) -> u128 {
        self.significand >> (63 + 3)    // move GRS bits and fraction away
    }
}

impl fmt::Debug for Decomposed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.sign {
            write!(f, "-")?;
        }

        let int_part = self.integer_bits();
        let raw_fraction = (self.significand & (0x7fff_ffff_ffff_ffff << 3)) >> 3;  // clears GRS
        let grs = self.significand & 0b111;
        let frac = format!("{:063b}", raw_fraction);
        let frac = frac.trim_right_matches('0');
        let frac = if frac.is_empty() { "0" } else { &frac };
        let frac_grs = if grs != 0 {
            format!("{}|{:03b}", frac, grs)
        } else {
            frac.to_string()
        };
        write!(f, "{:#b}.{}*2^{}", int_part, frac_grs, self.exponent)
    }
}
