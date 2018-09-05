//! Implementation of floating point algorithms (arithmetic, etc.).
//!
//! Useful resources:
//! * http://web.cs.ucla.edu/digital_arithmetic/files/ch8.pdf
//! * http://pages.cs.wisc.edu/~markhill/cs354/Fall2008/notes/flpt.apprec.html

use {f80, Classified, RoundingMode, FloatResult};
use decomposed::Decomposed;
use sign_mag::SignMagnitude;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

use std::{cmp, u128};

impl f80 {
    pub fn add_checked(self, rhs: Self, _rounding: RoundingMode) -> FloatResult<Self> {
        let lhs = self;
        let (lhs_c, rhs_c) = match lhs.propagate_nans(rhs) {
            Ok((lhs_c, rhs_c)) => (lhs_c, rhs_c),
            Err(res) => return res,
        };

        match (&lhs_c, &rhs_c) {
            (Classified::Inf { sign: lsign }, Classified::Inf{ sign: rsign }) => {
                if lsign == rsign { // lhs == rhs
                    return FloatResult::Exact(lhs);
                } else {
                    return FloatResult::InvalidOperand { sign: true };
                }
            }
            (Classified::Inf {..}, _) => return FloatResult::Exact(lhs),
            (_, Classified::Inf {..}) => return FloatResult::Exact(rhs),
            _ => {}
        }

        let (l, r) = (lhs_c.decompose().unwrap(), rhs_c.decompose().unwrap());
        trace!("l cls: {:?}; decomp: {:?}", lhs_c, l);
        trace!("r cls: {:?}; decomp: {:?}", rhs_c, r);

        // Align exponents to the highest one
        let exp = cmp::max(l.exponent(), r.exponent());
        let (l, r) = (l.adjust_exponent_to(exp), r.adjust_exponent_to(exp));
        let exact = l.is_exact() && r.is_exact();
        let (l, r) = (l.into_inner(), r.into_inner());
        trace!("lhs={:?}", l);
        trace!("rhs={:?}", r);

        let sum = l.to_sign_magnitude() + r.to_sign_magnitude();
        let sum_decomp = Decomposed::with_sign_magnitude_exponent(sum, exp);
        trace!("sum={:?}", sum_decomp);
        let sum = f80::from_decomposed(sum_decomp);
        trace!("sum decomp: {:?}", sum.into_inner().decompose());

        let exact = exact && sum.is_exact();
        match sum {
            FloatResult::Exact(f) if !exact => FloatResult::Rounded(f),
            _ => sum,
        }
    }

    pub fn sub_checked(self, rhs: Self, rounding: RoundingMode) -> FloatResult<Self> {
        // We need to do NaN propagation on the non-negated rhs to get the right sign
        if let Err(res) = self.propagate_nans(rhs) {
            return res;
        }

        // Now this can't be that simple, can it?
        self.add_checked(-rhs, rounding)
    }

    pub fn mul_checked(self, rhs: Self, _rounding: RoundingMode) -> FloatResult<Self> {
        let lhs = self;
        let (lhs_c, rhs_c) = match lhs.propagate_nans(rhs) {
            Ok((lhs_c, rhs_c)) => (lhs_c, rhs_c),
            Err(res) => return res,
        };

        match (&lhs_c, &rhs_c) {
            (Classified::Inf { sign: lsign }, Classified::Inf{ sign: rsign }) => {
                if lsign == rsign { // lhs == rhs
                    return FloatResult::Exact(lhs);
                } else {
                    return FloatResult::InvalidOperand { sign: true };
                }
            }
            (Classified::Inf {..}, _) => return FloatResult::Exact(lhs),
            (_, Classified::Inf {..}) => return FloatResult::Exact(rhs),
            _ => {}
        }

        let (l, r) = (lhs_c.decompose().unwrap(), rhs_c.decompose().unwrap());
        trace!("mul l cls: {:?}; decomp: {:?}", lhs_c, l);
        trace!("mul r cls: {:?}; decomp: {:?}", rhs_c, r);

        let exponent = l.exponent() + r.exponent();
        let (l, r) = (l.to_sign_magnitude(), r.to_sign_magnitude());
        let sign = l.sign() != r.sign();
        let mag = BigUint::from(l.magnitude()) * BigUint::from(r.magnitude());
        trace!("mul: {:#X} * {:#X} = {:#X}", l.magnitude(), r.magnitude(), mag);

        // The product has twice the bits of the significand. Shift it back to
        // adjust and make sure the shifted bits end up in the sticky bit.
        let bits: usize = 63 + 3;   // 63 fraction bits + 3 extra bits (GRS)
        let mask: u128 = (1 << bits) - 1;
        let dropped_bits = &mag & BigUint::from(mask);
        let sticky = dropped_bits != BigUint::from(0u8);
        let mag = mag >> bits;
        trace!("mul: dropped bits={:#X}, left={:#X}, sticky={}", dropped_bits, mag, sticky);
        assert!(mag < u128::MAX.into());

        let product = SignMagnitude::new(sign, mag.to_u128()
            .expect("multiplication exceeds u128 range"));
        let mut product_decomp = Decomposed::with_sign_magnitude_exponent(product, exponent);
        if sticky {
            product_decomp.set_sticky();
        }
        trace!("product={:?}", product_decomp);
        let product = f80::from_decomposed(product_decomp);
        trace!("product decomp: {:?}", product.into_inner().decompose());

        product
    }

    /// Classifies `self` and `rhs`, returning an `Err` when one of them is NaN.
    fn propagate_nans(self, rhs: Self) -> Result<(Classified, Classified), FloatResult<Self>> {
        // short-circuit on invalid operands (which might turn into QNaNs later)
        let lhs = self;
        let (lhs_c, rhs_c) = match (lhs.classify_checked(), rhs.classify_checked()) {
            (Some(lhs), Some(rhs)) => (lhs, rhs),
            _ => return Err(FloatResult::InvalidOperand {sign: false}),  // FIXME check against HW
        };

        match (lhs_c, rhs_c) {
            // Propagate NaNs turning signaling ones into quiet ones.
            (
                Classified::NaN {sign: lsign, payload: lpl, signaling: _},
                Classified::NaN {sign: rsign, payload: rpl, signaling: _},
            ) => {
                // When both operands are NaNs, the x87 (at least in my Haswell
                // CPU) propagates the one with the larger payload.
                let (sign, payload) = if lpl > rpl {
                    (lsign, lpl)
                } else {
                    (rsign, rpl)
                };

                let cls = Classified::NaN {
                    sign,
                    signaling: false,
                    payload,
                };
                Err(FloatResult::Exact(cls.pack()))
            },
            (Classified::NaN {sign, payload, signaling: _}, _) |
            (_, Classified::NaN {sign, payload, signaling: _}) => {
                let cls = Classified::NaN {
                    sign,
                    signaling: false,
                    payload,
                };
                Err(FloatResult::Exact(cls.pack()))
            },
            (lhs_c, rhs_c) => Ok((lhs_c, rhs_c)),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate env_logger;

    use f80;
    use ieee754::Ieee754;

    #[test]
    fn add_simple() {
        let two = f80::from(1.0f32) + f80::from(1.0f32);
        assert_eq!(two, f80::from(2.0f32));
    }

    #[test]
    fn mul_simple() {
        env_logger::try_init().ok();
        assert_eq!(f80::from(1.0f32) * f80::from(1.0f32), f80::from(1.0f32));
        assert_eq!(f80::from(1.0f32) * f80::from(2.0f32), f80::from(2.0f32));
        assert_eq!(f80::from(0.0f32) * f80::from(1.0f32), f80::from(0.0f32));
        assert_eq!(f80::from(-1.0f32) * f80::from(1.0f32), f80::from(-1.0f32));
        assert_eq!(f80::from(-1.0f32) * f80::from(-1.0f32), f80::from(1.0f32));
        assert_eq!(f80::from(0.5f32) * f80::from(1.0f32), f80::from(0.5f32));
        assert_eq!(f80::from(0.5f32) * f80::from(1.5f32), f80::from(0.75f32));
    }

    fn addition(lhs_bits: u32, rhs_bits: u32) {
        env_logger::try_init().ok();

        let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));
        let f32sum = lhs + rhs;
        let f32bits = f32sum.to_bits();

        let (l80, r80) = (f80::from(lhs), f80::from(rhs));
        let f80sum = l80 + r80;
        let f80bits = f80sum.to_f32().to_bits();

        debug!("expected: {:#010X}+{:#010X}={:#010X} ({}+{}={})", lhs_bits, rhs_bits, f32sum.to_bits(), lhs, rhs, f32sum);
        debug!("     f80: {:?}+{:?}={:?} ({:?}+{:?}={:?})", l80, r80, f80sum, l80.classify(), r80.classify(), f80sum.classify());
        debug!("    back: {:#010X}+{:#010X}={:#010X}", l80.to_f32().to_bits(), r80.to_f32().to_bits(), f80sum.to_f32().to_bits());
        assert_eq!(f32bits, f80bits, "{:X}; {:X}", f32bits, f80bits);
    }

    #[test]
    fn add_denormal() {
        env_logger::try_init().ok();

        let (lhs, rhs) = (f32::from_bits(1), f32::from_bits(8388607));
        let f32sum = lhs + rhs;
        let f32f80 = f80::from(f32sum);
        let f32bits = f32f80.to_bits();

        let (l80, r80) = (f80::from(lhs), f80::from(rhs));
        let f80sum = l80 + r80;
        let f80bits = f80sum.to_bits();

        debug!("l f32 decomp: {:?}", lhs.decompose());
        debug!("r f32 decomp: {:?}", rhs.decompose());
        debug!("l f80 decomp: {:?}", l80.decompose());
        debug!("r f80 decomp: {:?}", l80.decompose());
        debug!("{}+{}={}", lhs, rhs, f32sum);
        debug!("({:?}+{:?}={:?})", lhs.classify(), rhs.classify(), f32sum.classify());
        debug!("f80: {:?}+{:?}={:?}", l80.classify(), r80.classify(), f80sum.classify());

        debug!("f32 result: {:?}", f32f80.classify());
        debug!("f80 result: {:?}", f80sum.classify());
        assert_eq!(f32f80, f80sum, "{:#X}<->{:#X}", f32bits, f80bits);
    }

    #[test]
    fn add_smoke_1() {
        addition(1, 16777216);
    }

    /// Exponent alignment would lose MSbs here. Make sure it neither trips an
    /// assertion nor returns the wrong result.
    #[test]
    fn add_large() {
        addition(206028801, 738197504);
    }

    #[test]
    fn shift_overflow() {
        addition(206028801, 1400897536);
    }

    /// This only passes when round-to-nearest is implemented.
    #[test]
    fn add_rounding_error() {
        addition(1358958057, 2977955840);
    }

    /// This one popped up randomly while implementing round-to-nearest, but is
    /// independent of that.
    #[test]
    fn add_wrong_nan_handling() {
        addition(2139095041, 0);
    }

    /// This failed because the ties-to-even case used to just do
    /// `truncated & !1`, which always rounds DOWN to an even significand. But
    /// in case we get a `xxx1.100`, we want to round UP, not down.
    #[test]
    fn add_wrong_even_rounding() {
        addition(552910884, 572284665);
    }

    /// This test failed because not all bits were incorporated into the sticky
    /// bit calculation, erroneously rounding down instead of up.
    #[test]
    fn add_wrong_sticky_computation() {
        addition(561687281, 574749429);
    }

    // Most of these were found by proptest and extracted to ease debugging.
    // Thank you proptest!
}
