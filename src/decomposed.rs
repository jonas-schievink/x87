//! Provides a decomposed representation of finite (non-NaN, non-Inf)
//! floating-point numbers.

use sign_mag::SignMagnitude;
use f80_mod::FloatResult;
use RoundingMode;
use std::{fmt, ops};

/// A normalized or denormal `f80` decomposed into its components (may be zero).
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Decomposed {
    pub sign: bool,
    exponent: i16,
    significand: Significand,
}

impl Decomposed {
    const LEADING_ZEROS_WHEN_NORMALIZED: u32 = 64 - 3;  // 3 for the GRS bits

    pub fn zero() -> Self {
        Self {
            sign: false,
            exponent: 0,
            significand: Significand::zero(),
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
            significand: Significand::from(significand),
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
            significand: Significand::from_raw(sign_mag.magnitude()),
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
        self.significand.is_exactly_zero()
    }

    /// Returns the sign and significand as a `SignMagnitude` value.
    pub fn to_sign_magnitude(&self) -> SignMagnitude {
        // FIXME should GRS bits be included here?
        SignMagnitude::new(self.sign, self.significand.raw())
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

        if self.significand.is_exactly_zero() {
            // Value is zero. Make the exponent sane (since it doesn't matter)
            // and return.
            normalized.exponent = 0;
            return FloatResult::Exact(normalized);
        }

        // If normalized, we want bit #63 to be set, and all higher-valued bits
        // to be unset. So we want there to be exactly 64 leading zeros.
        let leading_zeros = self.significand.raw().leading_zeros();
        if leading_zeros > Self::LEADING_ZEROS_WHEN_NORMALIZED {
            // Too many zeros on the left, shift first 1-bit to the integer pos
            // by adjusting exponent downwards.
            // This doesn't lose any bits (ie. doesn't round).
            let diff = (leading_zeros - Self::LEADING_ZEROS_WHEN_NORMALIZED) as i16;
            let result = normalized.adjust_exponent_to(self.exponent - diff);
            assert!(result.is_exact(), "unexpectedly lost bits during normalize");
            normalized = result.unwrap_exact_or_rounded();
            assert!(normalized.is_normalized());

            FloatResult::Exact(normalized)
        } else if leading_zeros < Self::LEADING_ZEROS_WHEN_NORMALIZED {
            // There's a 1-bit too far to the left, shift it into the integer
            // bit position by adjusting exponent upwards.
            let diff = (Self::LEADING_ZEROS_WHEN_NORMALIZED - leading_zeros) as i16;
            let result = normalized.adjust_exponent_to(self.exponent + diff);
            let exact = result.is_exact();
            normalized = result.unwrap_exact_or_rounded();
            assert!(normalized.is_normalized());

            if exact {
                FloatResult::Exact(normalized)
            } else {
                // Lost bits in the process
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
            let shift = (self.exponent - exponent) as u16;
            adj.significand = self.significand << shift;
            let back = adj.significand >> shift;
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
            let shift = (exponent - self.exponent) as u16;
            adj.significand = self.significand >> shift;
            let back = adj.significand << shift;
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
        self.significand.raw().leading_zeros() == Self::LEADING_ZEROS_WHEN_NORMALIZED
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

    pub fn as_f80_significand(&self) -> FloatResult<u64> {
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
        // FIXME store and use real rounding mode
        self.significand.reduced_fraction(bits, self.sign, RoundingMode::Nearest)
    }

    fn integer_bits(&self) -> u128 {
        self.significand.integer_bits()
    }
}

impl fmt::Debug for Decomposed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.sign {
            write!(f, "-")?;
        }

        write!(f, "{:?}*2^{}", self.significand, self.exponent)
    }
}

// TODO: Make new trimmed down `Significand` type private to this module

#[derive(PartialEq, Eq, Clone, Copy)]
struct Significand {
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

impl Significand {
    fn zero() -> Self {
        Self {
            significand: 0,
        }
    }

    fn from_raw(significand: u128) -> Self {
        Self { significand }
    }

    fn raw(&self) -> u128 {
        self.significand
    }

    fn is_exactly_zero(&self) -> bool {
        self.significand == 0
    }

    fn integer_bits(&self) -> u128 {
        self.significand >> (63 + 3)    // move GRS bits and fraction away
    }

    fn sticky_bit(&self) -> u128 {
        self.significand & 1
    }

    /// Rounds the fraction bits to get the given number of fraction bits.
    ///
    /// This is used for converting the extended significand (with guard, round
    /// and sticky bits) to the significand to use in the resulting `f80`. It
    /// can also produce smaller outputs for use in `f32` or `f64`.
    ///
    /// Note that this will not return any integer bits.
    fn reduced_fraction(&self, bits: u8, _sign: bool, _rounding: RoundingMode) -> FloatResult<u64> {
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
}

impl From<u64> for Significand {
    fn from(v: u64) -> Self {
        Self {
            significand: u128::from(v) << 3,    // GRS bits cleared
        }
    }
}

impl fmt::Debug for Significand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
        write!(f, "{:#b}.{}", int_part, frac_grs)
    }
}

/// Left-shifts preserve the sticky bit and shift it around.
// FIXME is this right?
impl ops::Shl<u16> for Significand {
    type Output = Self;

    fn shl(self, rhs: u16) -> Self {
        let sticky = self.sticky_bit();
        let result = if rhs > 127 { 0 } else { self.significand << rhs };
        Significand {
            significand: result | sticky,
        }
    }
}

/// Right-shifts preserve and update the sticky bits with the `OR` of all bits
/// shifted out.
impl ops::Shr<u16> for Significand {
    type Output = Self;

    fn shr(self, rhs: u16) -> Self {
        let sticky = self.sticky_bit() != 0;
        let mask = if rhs > 127 { !0 } else { (1 << rhs) - 1 };
        let lost_bits = self.significand & mask;
        let sticky = sticky || lost_bits != 0;
        let sticky = if sticky { 1 } else { 0 };
        let result = if rhs > 127 { 0 } else { self.significand >> rhs };
        Significand {
            significand: result | sticky,
        }
    }
}
