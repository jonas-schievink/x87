//! All of this was a huge mistake.
//!
//! But those who dare challenge the gods must pay for what they've done.

#![allow(unused)]

use {RoundingMode, FloatResult};

/// The extracted fixed-point significand of an `f80`, with rounding semantics.
///
/// This implements the different rounding modes and provides operations
/// respecting those to the rest of the library.
///
/// A `Significand` alone does not represent a number. It needs to be
/// interpreted with an exponent (and a sign) to be a complete number.
pub struct Significand {
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
    val: u128,
    /// The rounding semantics in use for this value.
    rounding: RoundingMode,
}

impl Significand {
    /// # Parameters
    ///
    /// * `f80_significand`: The significand (integer and fraction bits) of an
    ///   `f80` (64 bits: 1 integer bit and 63 fraction bits).
    /// * `rounding`: The selected rounding mode.
    pub fn new(f80_significand: u64, rounding: RoundingMode) -> Self {
        Self {
            val: u128::from(f80_significand) << 3,
            rounding,
        }
    }

    /// Converts `self` to the corresponding significand field to use in an
    /// `f80`.
    ///
    /// If the integer part is too large to fit in the target, this will panic.
    /// Call `normalize` to ensure this doesn't happen.
    pub fn to_f80_significand(&self) -> FloatResult<u64> {
        if self.integer_part() > 1 {
            panic!("`Significand` too large for f80 (call normalize first)");
        }

        let int = if self.integer_part() == 0 { 0 } else { 1 << 63 };

        let fraction = self.fraction_rounded();
        let exact = fraction.is_exact();
        let fraction = fraction.unwrap_exact_or_rounded();
        let result = int | fraction;
        if exact {
            FloatResult::Exact(result)
        } else {
            FloatResult::Rounded(result)
        }
    }

    fn integer_part(&self) -> u64 {
        (self.val >> 66) as u64
    }

    /// The fraction bits, rounded to yield the 63 `f80` fraction bits.
    fn fraction_rounded(&self) -> FloatResult<u64> {
        let _f80_part = ((self.val >> 3) & 0x7fff_ffff_ffff_ffff) as u64;
        unimplemented!()
    }

    /// Normalizes `self` by shifting it so that only the least significant
    /// integer bit is set. Returns the required exponent adjustment.
    pub fn normalize(&mut self) -> i16 {
        unimplemented!()
    }
}

// FIXME needs the sign to implement round to +/-inf correctly
