//! Implementation of floating point algorithms (arithmetic, etc.).
//!
//! Useful resources:
//! * http://web.cs.ucla.edu/digital_arithmetic/files/ch8.pdf
//! * http://pages.cs.wisc.edu/~markhill/cs354/Fall2008/notes/flpt.apprec.html

use {f80, Classified, RoundingMode, FloatResult};
use decomposed::Decomposed;
use std::cmp;

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
                    return FloatResult::InvalidOperand;
                }
            }
            (Classified::Inf {..}, _) => return FloatResult::Exact(lhs),
            (_, Classified::Inf {..}) => return FloatResult::Exact(rhs),
            _ => {}
        }

        let (l, r) = (lhs_c.decompose().unwrap(), rhs_c.decompose().unwrap());
        trace!("l cls: {:?}; decomp: {:?}", lhs_c, l);
        trace!("r cls: {:?}; decomp: {:?}", rhs_c, r);

        // To add two numbers, their exponents must be equal. We adjust the
        // numbers to the lower exponent, which possibly shifts one of the
        // numbers up (to the left).
        // This becomes a problem when the adjustment causes the significand to
        // become very large, potentially losing highly significant bits. The
        // "solution" chosen here limits the amount of downscaling artificially.
        // FIXME How is this usually done?
        let exp = cmp::max(l.exponent(), r.exponent());
        //let exp = cmp::max(max_exp-15, cmp::min(l.exponent, r.exponent));
        let (l, r) = (l.adjust_exponent_to(exp), r.adjust_exponent_to(exp));
        let exact = l.is_exact() && r.is_exact();
        let (l, r) = (l.unwrap_exact_or_rounded(), r.unwrap_exact_or_rounded());
        trace!("adj exp={}; lhs={:?}; rhs={:?}", exp, l, r);

        let sum = l.to_sign_magnitude() + r.to_sign_magnitude();
        let sum = f80::from_decomposed(Decomposed::with_sign_magnitude_exponent(sum, exp));
        trace!("sum decomp: {:?}", sum.into_inner().decompose());

        let exact = exact && sum.is_exact();
        match sum {
            FloatResult::Exact(f) if !exact => FloatResult::Rounded(f),
            _ => sum,
        }
    }

    /// Classifies `self` and `rhs`, returning an `Err` when one of them is NaN.
    fn propagate_nans(self, rhs: Self) -> Result<(Classified, Classified), FloatResult<Self>> {
        // short-circuit on invalid operands (which might turn into QNaNs later)
        let lhs = self;
        let (lhs_c, rhs_c) = match (lhs.classify_checked(), rhs.classify_checked()) {
            (Some(lhs), Some(rhs)) => (lhs, rhs),
            _ => return Err(FloatResult::InvalidOperand),
        };

        match (lhs_c, rhs_c) {
            // Propagate NaNs, preferably the left one, turning signaling ones
            // into quiet ones.
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
}
