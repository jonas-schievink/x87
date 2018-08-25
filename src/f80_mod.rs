use ieee754::Ieee754;
use std::{f32, f64, fmt};

/// An 80-bit extended precision floating point number.
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default)]
pub struct f80(pub(crate) u128);

impl f80 {
    pub const ZERO: Self         = f80(0);
    pub const INFINITY: Self     = f80(0x7fff_8000_0000_0000_0000);
    pub const NEG_INFINITY: Self = f80(0xffff_8000_0000_0000_0000);

    /// Not a Number (`NaN`).
    ///
    /// This is a quiet NaN (qNaN) with the same payload as returned by invalid
    /// operantions (such as `0.0 / 0.0`). The sign bit is not set.
    pub const NAN: Self          = f80(0x7fff_C000_0000_0000_0000);

    const EXPONENT_BIAS: i16 = 16383;

    /// Create an f80 from its raw byte representation.
    pub fn from_bytes(bytes: [u8; 10]) -> Self {
        let mut raw_u128 = [0; 16];
        raw_u128[..10].copy_from_slice(&bytes);

        // FIXME use from_bytes_le once stable (https://github.com/rust-lang/rust/issues/52963)
        union Convert {
            val: u128,
            bytes: [u8; 16],
        }
        let c = Convert { bytes: raw_u128 };

        f80(unsafe { c.val })
    }

    /// Returns the raw bytes making up this `f80`.
    pub fn to_bytes(&self) -> [u8; 10] {
        // FIXME use to_bytes_le once stable (https://github.com/rust-lang/rust/issues/52963)
        union Convert {
            val: u128,
            bytes: [u8; 16],
        }
        let c = Convert { val: self.0 };

        let mut bytes = [0; 10];
        bytes.copy_from_slice(unsafe { &c.bytes[..10] });
        bytes
    }

    /// Converts an `f32` to an `f80`. This is lossless and no rounding is
    /// performed.
    pub fn from_f32(f: f32) -> Self {
        let (sign, exp, significand) = f.decompose();
        let raw_exp = f.decompose_raw().1;
        //println!("from {} ({:#010X}) sign={} exp={} raw_exp={:#04X} significand={:#08X}", f, f.to_bits(), sign, exp, raw_exp, significand);

        let f80_fraction = u64::from(significand) << 40;    // 23-bit -> 63-bit
        let cls = match raw_exp {
            0x00 => match significand {
                0 => Classified::Zero { sign },
                _ => Classified::Denormal { sign, integer_bit: false, fraction: f80_fraction },
            }
            0xFF => match significand {
                0 => Classified::Inf { sign },
                _ => match significand & (1 << 22) {
                    0 => Classified::SNaN { sign, payload: u64::from(significand) },
                    _ => Classified::QNaN { sign, payload: u64::from(significand & !(1 << 22)) },
                }
            }
            _ => Classified::Normal { sign, exponent: exp, fraction: f80_fraction },
        };
        cls.pack()
    }

    /// Converts an `f64` to an `f80`. This is lossless and no rounding is
    /// performed.
    pub fn from_f64(f: f64) -> Self {
        let (sign, exp, significand) = f.decompose();
        let raw_exp = f.decompose_raw().1;
        //println!("from {} ({:#010X}) sign={} exp={} raw_exp={:#04X} significand={:#08X}", f, f.to_bits(), sign, exp, raw_exp, significand);

        let f80_fraction = significand << 11;    // 52-bit -> 63-bit
        let cls = match raw_exp {
            0x000 => match significand {
                0 => Classified::Zero { sign },
                _ => Classified::Denormal { sign, integer_bit: false, fraction: f80_fraction },
            }
            0x7FF => match significand {
                0 => Classified::Inf { sign },
                _ => match significand & (1 << 51) {
                    0 => Classified::SNaN { sign, payload: u64::from(significand) },
                    _ => Classified::QNaN { sign, payload: u64::from(significand & !(1 << 51)) },
                }
            }
            _ => Classified::Normal { sign, exponent: exp, fraction: f80_fraction },
        };
        cls.pack()
    }

    /// Converts `self` to an `f32`, possibly rounding in the process.
    pub fn to_f32(&self) -> f32 {
        self.to_f32_checked().into_inner()
    }

    /// Converts `self` to an `f64`, possibly rounding in the process.
    pub fn to_f64(&self) -> f64 {
        self.to_f64_checked().into_inner()
    }

    /// Converts `self` to an `f32`, reporting any loss of information (eg. by
    /// rounding).
    pub fn to_f32_checked(&self) -> FloatResult<f32> {
        // use the more "developer-friendly" output of `classify()` and
        // translate each case to the corresponding IEEE representation.
        let classified = self.classify();
        match classified {
            Classified::Zero { sign } => {
                FloatResult::Exact(if sign { -0.0 } else { 0.0 })
            },
            Classified::Inf { sign } => {
                FloatResult::Exact(if sign { -1.0/0.0 } else { 1.0/0.0 })
            },
            Classified::Denormal { sign, integer_bit: _, fraction } => {
                let raw_exp = 0;
                let frac = (fraction >> 40) & 0x007f_ffff; // truncate fraction to the upper 23 bits
                let result = f32::recompose_raw(sign, raw_exp, frac as u32);
                if (frac << 40) == fraction {
                    FloatResult::Exact(result)
                } else {
                    // lost bits in the process
                    FloatResult::Rounded(result)
                }
            }
            Classified::Normal { sign, exponent, fraction } => {
                let frac = (fraction >> 40) & 0x007f_ffff; // truncate fraction to the upper 23 bits
                let exp = exponent as i8 as i16;
                let result = f32::recompose(sign, exp, frac as u32);
                //println!("Normal->f32: exp={}->{} frac={:#X}->{:#010X}", exponent, exp, fraction, frac);
                match (exp == exponent, (frac << 40) == fraction) {
                    (true, true) => FloatResult::Exact(result),
                    (true, false) => FloatResult::Rounded(result), // lost fraction bits in the process
                    (false, _) => if exponent < 0 { // exponent is off = more than 2x off
                        FloatResult::TooSmall
                    } else {
                        FloatResult::TooLarge
                    },
                }
            }
            // We assume the host is IEEE 754-2008 compliant and uses the MSb of
            // the significant as an "is_quiet" flag. x87 does this and it
            // matches up with using a zero-payload SNaN for Infinities.
            Classified::SNaN { sign, payload } |
            Classified::QNaN { sign, payload } => {
                let is_quiet = if let Classified::QNaN { .. } = classified { true } else { false };
                let raw_exp = !0 & 0xff;
                let pl = (payload & 0x3f_ffff) as u32; // 22 remaining fraction bits

                // set quiet bit
                let fraction = if is_quiet {
                    pl | (1 << 22)
                } else {
                    pl
                };

                let result = f32::recompose_raw(sign, raw_exp, fraction);
                if payload == pl.into() {
                    FloatResult::Exact(result)
                } else {
                    FloatResult::LostNaN(result)
                }
            }
        }
    }

    /// Converts `self` to an `f64`, reporting any loss of information (eg. by
    /// rounding).
    pub fn to_f64_checked(&self) -> FloatResult<f64> {
        let classified = self.classify();
        match classified {
            Classified::Zero { sign } => {
                FloatResult::Exact(if sign { -0.0 } else { 0.0 })
            },
            Classified::Inf { sign } => {
                FloatResult::Exact(if sign { -1.0/0.0 } else { 1.0/0.0 })
            },
            Classified::Denormal { sign, integer_bit: _, fraction } => {
                let raw_exp = 0;
                let frac = (fraction >> 11) & 0xf_ffff_ffff_ffff; // truncate fraction to the upper 52 bits
                let result = f64::recompose_raw(sign, raw_exp, frac);
                if (frac << 11) == fraction {
                    FloatResult::Exact(result)
                } else {
                    // lost bits in the process
                    FloatResult::Rounded(result)
                }
            }
            Classified::Normal { sign, exponent, fraction } => {
                let frac = (fraction >> 11) & 0xf_ffff_ffff_ffff; // truncate fraction to the upper 52 bits
                let mut exp = exponent as u16 & 0b000_0111_1111_1111; // drop upper 4 bits of exponent (15->11)
                exp |= if exp & 0b100_0000_0000 != 0 {0b1111_1000_0000_0000} else {0};   // sign-extend
                let exp = exp as i16;

                let result = f64::recompose(sign, exp, frac);
                //println!("Normal->f64: exp={}->{} frac={:#X}->{:#010X}", exponent, exp, fraction, frac);
                match (exp == exponent, (frac << 11) == fraction) {
                    (true, true) => FloatResult::Exact(result),
                    (true, false) => FloatResult::Rounded(result), // lost fraction bits in the process
                    (false, _) => if exponent < 0 { // exponent is off = more than 2x off
                        FloatResult::TooSmall
                    } else {
                        FloatResult::TooLarge
                    },
                }
            }
            // We assume the host is IEEE 754-2008 compliant and uses the MSb of
            // the significant as an "is_quiet" flag. x87 does this and it
            // matches up with using a zero-payload SNaN for Infinities.
            Classified::SNaN { sign, payload } |
            Classified::QNaN { sign, payload } => {
                let is_quiet = if let Classified::QNaN { .. } = classified { true } else { false };
                let raw_exp = 0x7ff;
                let pl = payload & 0x7_ffff_ffff_ffff; // 51 remaining fraction bits

                // set quiet bit
                let fraction = if is_quiet {
                    pl | (1 << 51)
                } else {
                    pl
                };

                let result = f64::recompose_raw(sign, raw_exp, fraction);
                if payload == pl.into() {
                    FloatResult::Exact(result)
                } else {
                    FloatResult::LostNaN(result)
                }
            }
        }
    }

    /// Returns the value of the sign bit (the most significant bit in the raw
    /// 80-bit value).
    fn sign_bit(&self) -> bool {
        self.0 & (1 << 79) != 0
    }

    pub fn is_nan(&self) -> bool {
        match self.classify() {
            Classified::QNaN {..} | Classified::SNaN {..} => true,
            _ => false,
        }
    }

    pub fn is_infinite(&self) -> bool {
        match self.classify() {
            Classified::Inf {..} => true,
            _ => false,
        }
    }

    pub fn is_finite(&self) -> bool {
        !(self.is_infinite() || self.is_nan())
    }

    pub fn is_normal(&self) -> bool {
        match self.classify() {
            Classified::Normal {..} => true,
            _ => false,
        }
    }

    pub fn is_sign_negative(&self) -> bool {
        self.sign_bit()
    }

    pub fn is_sign_positive(&self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns the "real" (unbiased) exponent.
    pub fn exponent(&self) -> i16 {
        (self.biased_exponent() as i16) - Self::EXPONENT_BIAS
    }

    /// The biased or "raw" 15-bit exponent stored directly in the `f80` bits.
    fn biased_exponent(&self) -> u16 {
        ((self.0 >> 64) & 0x7fff) as u16
    }

    pub fn integer_part(&self) -> bool {
        (self.0 >> 63) & 0b1 != 0
    }

    pub fn fraction(&self) -> u64 {
        (self.0 & 0x7fff_ffff_ffff_ffff) as u64
    }

    /// Returns the 64-bit significant, including the explicit integer bit.
    ///
    /// The MSb of the returned value is the integer part (1 for normalized
    /// numbers), the remaining 63 bits are the fractional part.
    pub fn significand(&self) -> u64 {
        (self.0 & 0xffff_ffff_ffff_ffff) as u64
    }

    /// Classifies `self`, returning the kind of floating-point number its value
    /// indicates.
    ///
    /// If `self` is considered an invalid operand, an indefinite result is
    /// returned, which is a quiet NaN (QNaN) with a payload of 0.
    pub fn classify(&self) -> Classified {
        self.classify_checked().unwrap_or(Classified::INDEFINITE)
    }

    /// Classifies `self`, returning the kind of floating-point number its value
    /// indicates.
    ///
    /// Returns `None` when `self` is an invalid operand. In that case,
    /// operations involving `self` either cause an invalid operand exception or
    /// interpret `self` as a an "Indefinite" QNaN (with a payload of 0).
    ///
    /// To mimic the default behaviour of using an indefinite QNaN, call
    /// `.unwrap_or(Classified::INDEFINITE)` on the return value.
    pub fn classify_checked(&self) -> Option<Classified> {
        let sign = self.sign_bit();
        let integer_bit = self.integer_part();
        let fraction = self.fraction();
        let msb2 = (self.significand() >> 62) & 0b11;
        let rest62 = fraction & 0x3fff_ffff_ffff_ffff;
        let raw_exponent = self.biased_exponent();
        //println!("raw={:?} sign={} exp={} integer_part={} fraction={:#X} msb2={:#b} rest62:{:#b}", self, sign, raw_exponent, integer_bit, fraction, msb2, rest62);

        let cls = match raw_exponent {
            0 => match (integer_bit, fraction) {
                (false, 0) => Classified::Zero { sign },
                (false, _) => Classified::Denormal { sign, integer_bit, fraction },
                (true, _) => Classified::Denormal { sign, integer_bit, fraction },    // "Pseudo-Denormal"
            },
            0x7fff /* all bits set */ => match (msb2, rest62) {
                (0b00, 0) => return None,   // Pseudo-Infinity
                (0b00, _) => return None,   // Pseudo-NaN
                (0b01, _) => return None,   // Pseudo-NaN
                (0b10, 0) => Classified::Inf { sign },
                (0b10, _) => Classified::SNaN { sign, payload: rest62 },
                (0b11, _) => Classified::QNaN { sign, payload: rest62 },
                _ => unreachable!(),
            },
            _ => match integer_bit {
                false => return None,   // Unnormal
                true => Classified::Normal {
                    sign,
                    exponent: self.exponent(),
                    fraction,
                },
            }
        };
        //println!("classify: {:?} -> {:?}", self, cls);
        Some(cls)
    }
}
// TODO: implement as much as is useful from `f32` and `f64`

impl fmt::Debug for f80 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#022X}", self.0)
    }
}

impl From<f32> for f80 {
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

impl From<f64> for f80 {
    fn from(f: f64) -> Self {
        Self::from_f64(f)
    }
}

impl From<u8> for f80 {
    fn from(v: u8) -> Self {
        Self::from(f32::from(v))
    }
}

impl From<i8> for f80 {
    fn from(v: i8) -> Self {
        Self::from_f32(f32::from(v))
    }
}

impl From<u16> for f80 {
    fn from(v: u16) -> Self {
        Self::from(f32::from(v))
    }
}

impl From<i16> for f80 {
    fn from(v: i16) -> Self {
        Self::from(f32::from(v))
    }
}

impl From<u32> for f80 {
    fn from(v: u32) -> Self {
        Self::from(f64::from(v))
    }
}

impl From<i32> for f80 {
    fn from(v: i32) -> Self {
        Self::from(f64::from(v))
    }
}

/// The result of a floating-point operation/conversion with result `T`.
#[derive(Debug, Copy, Clone)]
pub enum FloatResult<T> {
    /// No rounding or range error, result is exact.
    Exact(T),
    /// The result was rounded by no more than one ULP.
    Rounded(T),
    /// The result is too large for the type `T` and would be "rounded" towards
    /// `+Inf`.
    TooLarge,
    /// The result if too small for the type `T` and would be "rounded" towards
    /// `-Inf`.
    TooSmall,
    /// Parts of a NaN payload were lost.
    ///
    /// Either the payload was truncated (1-bits were lost), or the payload was
    /// replaced by a different value because the payload had a special meaning
    /// for the target type.
    LostNaN(T),
}

impl<T: fmt::Debug> FloatResult<T> {
    /// Returns the exact result, panicking if the result isn't a
    /// `FloatResult::Exact`.
    pub fn unwrap_exact(self) -> T {
        if let FloatResult::Exact(f) = self {
            f
        } else {
            panic!("called `unwrap_exact` on a {:?}", self);
        }
    }
}

impl FloatResult<f32> {
    /// Extract the (possibly rounded) result.
    pub fn into_inner(self) -> f32 {
        match self {
            FloatResult::Exact(t) => t,
            FloatResult::Rounded(t) => t,
            FloatResult::TooLarge => f32::INFINITY,
            FloatResult::TooSmall => f32::NEG_INFINITY,
            FloatResult::LostNaN(t) => t,
        }
    }
}

impl FloatResult<f64> {
    /// Extract the (possibly rounded) result.
    pub fn into_inner(self) -> f64 {
        match self {
            FloatResult::Exact(t) => t,
            FloatResult::Rounded(t) => t,
            FloatResult::TooLarge => f64::INFINITY,
            FloatResult::TooSmall => f64::NEG_INFINITY,
            FloatResult::LostNaN(t) => t,
        }
    }
}

/// An `f80` separated by the kind of value it represents.
#[derive(Debug)]
pub enum Classified {
    /// All-zero exponent and significand.
    Zero {
        sign: bool,
    },
    /// All-zero exponent, significand `<1` (integer bit clear).
    ///
    /// The value is `s * m * 2^(-16382)` (where `s` is the sign and `m` the
    /// fractional part of the significant).
    Denormal {
        sign: bool,
        /// The integer bit (bit #63). If set, this is a pseudo-denormal, which
        /// isn't generated by the coprocessor. It should generally be `false`.
        integer_bit: bool,
        /// The fractional part of the significant (63 bits).
        fraction: u64,
    },
    /// All-one exponent, MSb of significand is 1, rest 0.
    Inf {
        sign: bool,
    },
    /// Signalling NaN. All-one exponent, most significant 2 bits of significand
    /// are `10`, rest is anything but zero.
    SNaN {
        sign: bool,
        /// The non-zero payload carried in the lower 62 bits of the
        /// significand.
        payload: u64,
    },
    /// Quiet NaN or "Indefinite" result. All-one exponent, most significant 2
    /// bits of significand are `11`.
    QNaN {
        sign: bool,
        /// The lower 62 bits of the significand (may be 0).
        ///
        /// If the payload is 0, this is an "Indefinite" result created by
        /// undefined calculations (root/logarithm of negative numbers, 0/0,
        /// Inf/Inf, ...) or using an otherwise invalid operand.
        payload: u64,
    },
    /// Normalized value. Any non-zero and not-all-bits-set exponent, MSb (bit
    /// #63) of the significand must be set.
    Normal {
        sign: bool,
        /// The unbiased 15-bit exponent (exponent field with bias already
        /// subtracted).
        exponent: i16,
        /// Fraction (63 bits). The integer part of the significand is 1.
        fraction: u64,
    },

    // All other representations ("pseudo"-NaN/-Infinity/-Denormal) are invalid
    // operands starting with the 80387.
}

impl Classified {
    /// The value returned by invalid or undefined operations.
    pub const INDEFINITE: Self = Classified::QNaN { sign: false, payload: 0 };

    /// Converts this classified representation back into an equivalent `f80`.
    pub fn pack(&self) -> f80 {
        let (sign, raw_exponent, significant): (bool, u16, u64) = match self {
            Classified::Zero { sign } => {
                (*sign, 0, 0)
            }
            Classified::SNaN { sign, payload } => {
                (*sign, !0, (0b10 << 62) | payload)
            }
            Classified::QNaN { sign, payload } => {
                (*sign, !0, (0b11 << 62) | payload)
            }
            Classified::Inf { sign } => {
                (*sign, !0, (0b10 << 62))
            }
            Classified::Normal { sign, exponent, fraction } => {
                let int = 1 << 63;    // integer bit always set
                (*sign, (exponent + f80::EXPONENT_BIAS) as u16, int | *fraction)
            }
            Classified::Denormal { sign, integer_bit, fraction } => {
                let int = (if *integer_bit { 1 } else { 0 }) << 63;
                (*sign, 0, int | *fraction)
            }
        };

        let sign: u128 = (if sign { 1 } else { 0 }) << 79;
        let raw_exp: u128 = u128::from(raw_exponent & 0x7fff) << 64;
        let significant: u128 = u128::from(significant);

        let raw = sign | raw_exp | significant;
        //println!("pack: {:?} -> {:#020X}", self, raw);
        f80(raw)
    }
}

// `std::ops` behaves like all exceptions are masked and with default rounding.
// Forward to more flexible ops for use by the x87 emulator.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        match f80::ZERO.classify() {
            Classified::Zero { sign: false } => {},
            e => panic!("f80::ZERO ({:?}) is {:?}", f80::ZERO, e),
        }
    }

    #[test]
    fn infinity() {
        match f80::INFINITY.classify() {
            Classified::Inf { sign: false } => {},
            e => panic!("f80::INFINITY ({:?}) is {:?}", f80::INFINITY, e),
        }

        match f80::NEG_INFINITY.classify() {
            Classified::Inf { sign: true } => {},
            e => panic!("f80::NEG_INFINITY ({:?}) is {:?}", f80::NEG_INFINITY, e),
        }
    }

    #[test]
    fn nan() {
        match f80::NAN.classify() {
            Classified::QNaN { sign: false, payload: 0 } => {},
            e => panic!("f80::NAN ({:?}) is {:?}", f80::NAN, e),
        }
    }
}
