use decomposed::Decomposed;
use ieee754::Ieee754;
use std::{f32, f64, u64, fmt};
use utils::ExactOrRounded;

/// An 80-bit extended precision floating point number.
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default)]
pub struct f80(u128);

impl f80 {
    pub const ZERO: Self         = f80(0);
    pub const INFINITY: Self     = f80(0x7fff_8000_0000_0000_0000);
    pub const NEG_INFINITY: Self = f80(0xffff_8000_0000_0000_0000);

    /// Not a Number (`NaN`).
    ///
    /// This is a quiet NaN (qNaN) with the same payload as returned by invalid
    /// operations (such as `0.0 / 0.0`). The sign bit is not set.
    pub const NAN: Self          = f80(0x7fff_C000_0000_0000_0000);

    /// The value to subtract from the raw exponent field to get the actual
    /// exponent.
    const EXPONENT_BIAS: i16 = 16383;
    const MIN_EXPONENT: i16 = 1 - Self::EXPONENT_BIAS;
    const MAX_EXPONENT: i16 = 0b0111_1111_1111_1110 - Self::EXPONENT_BIAS;
    /// Exponent used for all denormals.
    const DENORMAL_EXPONENT: i16 = -16382;

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

    pub fn from_bits(raw: u128) -> Self {
        assert_eq!(raw & 0xffff_ffff_ffff_ffff_ffff, raw, "invalid bits set");
        f80(raw)
    }

    pub fn to_bits(&self) -> u128 {
        self.0
    }

    /// Converts an `f32` to an `f80`. This is lossless and no rounding is
    /// performed.
    pub fn from_f32(f: f32) -> Self {
        let (sign, exp, significand) = f.decompose();
        let raw_exp = f.decompose_raw().1;
        trace!("from_f32: {} ({:?}, {:#010X}) sign={} exp={} raw_exp={:#04X} significand={:#08X}", f, f.classify(), f.to_bits(), sign, exp, raw_exp, significand);

        let f80_fraction = u64::from(significand) << 40;    // 23-bit -> 63-bit
        let decomp = match raw_exp {
            0x00 => match significand {
                0 => Decomposed::zero(),
                _ => {
                    // f32 denormal, exponent is -126, integer bit unset.
                    Decomposed::new(sign, -126, f80_fraction)
                },
            }
            0xFF => match significand {
                0 => return if sign { Self::NEG_INFINITY } else { Self::INFINITY },
                _ => return Classified::NaN {
                    sign,
                    signaling: significand & (1 << 22) == 0,
                    // The smaller f32 payload is moved into the most significant bits
                    payload: u64::from(significand & !(1 << 22)) << 40,
                }.pack(),
            }
            _ => Decomposed::new(sign, exp, (1 << 63) | f80_fraction),  // normal
        };
        Self::from_decomposed(decomp).unwrap_exact()
    }

    /// Converts an `f64` to an `f80`. This is lossless and no rounding is
    /// performed.
    pub fn from_f64(f: f64) -> Self {
        let (sign, exp, significand) = f.decompose();
        let raw_exp = f.decompose_raw().1;
        trace!("from_f64: {} ({:?}, {:#010X}) sign={} exp={} raw_exp={:#04X} significand={:#08X}", f, f.classify(), f.to_bits(), sign, exp, raw_exp, significand);

        let f80_fraction = significand << 11;    // 52-bit -> 63-bit
        let decomp = match raw_exp {
            0x00 => match significand {
                0 => Decomposed::zero(),
                _ => {
                    // f64 denormal, exponent is -1022, integer bit unset.
                    Decomposed::new(sign, -1022, f80_fraction)
                },
            }
            0b111_11111111 => match significand {
                0 => return if sign { Self::NEG_INFINITY } else { Self::INFINITY },
                _ => return Classified::NaN {
                    sign,
                    signaling: significand & (1 << 51) == 0,
                    payload: u64::from(significand & !(1 << 51)),
                }.pack(),
            }
            _ => Decomposed::new(sign, exp, (1 << 63) | f80_fraction),  // normal
        };
        Self::from_decomposed(decomp).unwrap_exact()
    }

    /// Creates a normal, denormal, or zero `f80` from its decomposed
    /// components.
    ///
    /// `decomp` will be normalized and rounded to fit in the `f80` encoding
    /// space.
    pub fn from_decomposed(decomp: Decomposed) -> FloatResult<Self> {
        let orig_decomp = decomp;
        trace!("from_decomposed:     orig: {:?}", orig_decomp);
        let decomp = decomp.normalize()
            .chain(|norm| {
                trace!("from_decomposed: normaliz: {:?}", norm);
                norm.round()
            })
            .chain(|rounded| {
                trace!("from_decomposed:  rounded: {:?}", rounded);
                rounded.normalize()
            })
            .map(|decomp| {
                trace!("from_decomposed: postnorm: {:?}", decomp);
                decomp
            });
        let exact = decomp.is_exact();
        let decomp = decomp.into_inner();

        let sign = if decomp.sign { 1 << 79 } else { 0 };

        if decomp.is_zero() {
            // Encodes `+0.0` or `-0.0`
            return FloatResult::Exact(f80(sign));
        }

        // If the exponent is in the "normal" range, we can represent this
        // number as a normal f80 (the integer bit is set anyway since this is
        // normalized).
        if decomp.exponent() >= f80::MIN_EXPONENT && decomp.exponent() <= f80::MAX_EXPONENT {
            let exp = (decomp.exponent() + Self::EXPONENT_BIAS) as u128;
            assert_eq!(exp & 0x8000, 0, "MSb set");
            let significand = decomp.as_f80_significand();
            let exact = exact && significand.is_exact();
            let significand = u128::from(significand.unwrap_exact_or_rounded());
            let exp = exp << 64;
            let result = f80(sign | exp | significand);
            trace!("-> normal {:?} (sign={}, raw_exp={:X}, exact={})", result, sign, exp, exact);
            if exact {
                FloatResult::Exact(result)
            } else {
                FloatResult::Rounded(result)
            }
        } else if decomp.exponent() > f80::MAX_EXPONENT {
            // Cannot be represented as normal or denormal since it's far too
            // large
            if sign != 0 {
                FloatResult::TooSmall
            } else {
                FloatResult::TooLarge
            }
        } else {
            // Exponent is smaller than what we'd like. This will end up being a
            // denormal or, if the significand doesn't have enough bits, zero.
            // One cool thing is that a denormal with an all-zero significand
            // ends up being the encoding for zero, so we don't have to special-
            // case this.
            let denorm = decomp.adjust_exponent_to(f80::DENORMAL_EXPONENT);
            let exact = exact && denorm.is_exact();
            let significand = denorm.into_inner().as_f80_fraction();
            let exact = exact && significand.is_exact();
            let significand = u128::from(significand.into_inner());

            // Encode denormal.
            let result = f80(sign | significand);
            if exact {
                FloatResult::Exact(result)
            } else {
                FloatResult::Rounded(result)
            }
        }
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
        trace!("to_f32_checked: self={:?}={:?}={:?}", self, classified, classified.decompose());
        match classified {
            Classified::Zero { sign } => {
                return FloatResult::Exact(if sign { -0.0 } else { 0.0 });
            },
            Classified::Inf { sign } => {
                return FloatResult::Exact(if sign { -1.0 / 0.0 } else { 1.0 / 0.0 });
            },
            // We assume the host is IEEE 754-2008 compliant and uses the MSb of
            // the significand as an "is_quiet" flag. x87 does this and it
            // matches up with using a zero-payload SNaN for Infinities.
            Classified::NaN { sign, signaling, payload } => {
                let raw_exp = !0 & 0xff;
                // Extract the payload from the upper 22 bits
                let pl = ((payload & (0x3f_ffff << 40)) >> 40) as u32; // 22 remaining fraction bits

                // set quiet bit
                let fraction = if signaling {
                    pl
                } else {
                    pl | (1 << 22)
                };

                let result = f32::recompose_raw(sign, raw_exp, fraction);
                if payload == u64::from(pl) << 40 {
                    return FloatResult::Exact(result);
                } else {
                    return FloatResult::LostNaN(result);
                }
            }
            _ => {}
        }

        // Finite (normal or denormal)
        let decomp = self.decompose().unwrap();
        trace!("to_f32_checked:     orig: {:?}", decomp);
        let decomp = decomp.normalize()
            .chain(|norm| {
                trace!("to_f32_checked: normaliz: {:?}", norm);
                norm.round_to(23)
            })
            .chain(|rounded| {
                trace!("to_f32_checked:  rounded: {:?}", rounded);
                rounded.normalize()
            })
            .map(|decomp| {
                trace!("to_f32_checked: postnorm: {:?}", decomp);
                decomp
            });
        let exact = decomp.is_exact();
        let decomp = decomp.into_inner();

        // If the exponent is too small for f32, try encoding as a
        // denormal, then fall back to rounding to 0. If it's too large,
        // we've hit +/-Inf.
        if decomp.exponent() >= -126 && decomp.exponent() <= 127 {
            // Fits in a normal f32, but significand might need
            // rounding. Go from 63 fraction bits to 23:
            decomp.as_f32_significand().map(|fraction| {
                let result = f32::recompose(decomp.sign, decomp.exponent(), fraction);
                trace!("normal; exp={}; frac={:X}; exact={}; result={}", decomp.exponent(), fraction, exact, result);
                result
            }).into()
        } else if decomp.exponent() < -126 {
            // Too close to 0.0 to be a normal float. Try denormal,
            // rounding to zero if that also doesn't fit.
            // Denormals need an exponent of -126:
            decomp.adjust_exponent_to(-126)
                .keep_exact_if(exact)
                .chain(|decomp| {
                    // Extract the fraction bits we have left:
                    decomp.as_f32_significand()
                }).map(|fraction| {
                    trace!("needs denormal. adj={:?}; exact={}", decomp, exact);
                    // The `fraction` bits might end up being all zero. In that
                    // case, the f64 will encode zero, which is correct here.
                    let sign = if decomp.sign { 0x8000_0000 } else { 0 };
                    f32::from_bits(sign | fraction)
            }).into()
        } else {
            // Exponent too large. Number too large or small for f32
            // range. "Round" to +/-Inf.
            if decomp.sign {
                FloatResult::TooSmall
            } else {
                FloatResult::TooLarge
            }
        }
    }

    // FIXME We should **REALLY** deduplicate these to_f32/to_f64!!
    /// Converts `self` to an `f64`, reporting any loss of information (eg. by
    /// rounding).
    pub fn to_f64_checked(&self) -> FloatResult<f64> {
        let classified = self.classify();
        match classified {
            Classified::Zero { sign } => {
                return FloatResult::Exact(if sign { -0.0 } else { 0.0 });
            },
            Classified::Inf { sign } => {
                return FloatResult::Exact(if sign { -1.0 / 0.0 } else { 1.0 / 0.0 });
            },
            // We assume the host is IEEE 754-2008 compliant and uses the MSb of
            // the significand as an "is_quiet" flag. x87 does this and it
            // matches up with using a zero-payload SNaN for Infinities.
            Classified::NaN { sign, signaling, payload } => {
                let raw_exp = 0x7ff;
                let pl = payload & 0x7_ffff_ffff_ffff; // 51 remaining fraction bits

                // set quiet bit
                let fraction = if signaling {
                    pl
                } else {
                    pl | (1 << 51)
                };

                let result = f64::recompose_raw(sign, raw_exp, fraction);
                if payload == pl.into() {
                    return FloatResult::Exact(result);
                } else {
                    return FloatResult::LostNaN(result);
                }
            }
            _ => {}
        }

        // Finite (normal or denormal)
        let decomp = self.decompose().unwrap();
        trace!("to_f64_checked:     orig: {:?}", decomp);
        let decomp = decomp.normalize()
            .chain(|norm| {
                trace!("to_f64_checked: normaliz: {:?}", norm);
                norm.round_to(52)
            })
            .chain(|rounded| {
                trace!("to_f64_checked:  rounded: {:?}", rounded);
                rounded.normalize()
            })
            .map(|decomp| {
                trace!("to_f64_checked: postnorm: {:?}", decomp);
                decomp
            });
        let exact = decomp.is_exact();
        let decomp = decomp.into_inner();

        // If the exponent is too small for f64, try encoding as a
        // denormal, then fall back to rounding to 0. If it's too large,
        // we've hit +/-Inf.
        if decomp.exponent() >= -1022 && decomp.exponent() <= 1023 {
            // Fits in a normal f64, but significand might need
            // rounding. Go from 63 fraction bits to 52:
            decomp.as_f64_significand().map(|fraction| {
                let result = f64::recompose(decomp.sign, decomp.exponent(), fraction);
                trace!("normal; exp={}; frac={:X}; exact={}; result={}", decomp.exponent(), fraction, exact, result);
                result
            }).into()
        } else if decomp.exponent() < -1022 {
            // Too close to 0.0 to be a normal float. Try denormal,
            // rounding to zero if that also doesn't fit.
            // Denormals need an exponent of -1022:
            decomp.adjust_exponent_to(-1022)
                .keep_exact_if(exact)
                .chain(|decomp| {
                    // Extract the 52 fraction bits we have left:
                    decomp.as_f64_significand()
                }).map(|fraction| {
                    trace!("needs denormal. adj={:?}; exact={}", decomp, exact);
                    // The `fraction` bits might end up being all zero. In that
                    // case, the f64 will encode zero, which is correct here.
                    let sign = if decomp.sign { 0x8000_0000_0000_0000 } else { 0 };
                    f64::from_bits(sign | fraction)
                }).into()
        } else {
            // Exponent too large. Number too large or small for f64
            // range. "Round" to +/-Inf.
            if decomp.sign {
                FloatResult::TooSmall
            } else {
                FloatResult::TooLarge
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
            Classified::NaN {..} => true,
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

    /// Returns the 64-bit significand, including the explicit integer bit.
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
                (0b10, _) => Classified::NaN { sign, signaling: true, payload: rest62 },
                (0b11, _) => Classified::NaN { sign, signaling: false, payload: rest62 },
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

    pub fn decompose(&self) -> Option<Decomposed> {
        self.classify_checked().and_then(|cls| cls.decompose())
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
    /// An operand of the performed operation was invalid.
    ///
    /// If the invalid operand exception is masked, the result of the operation
    /// is an indefinite result, a QNaN with payload 0, which is also returned
    /// by `into_inner`.
    InvalidOperand {
        /// When turned into a QNaN, the sign depends on the performed
        /// operation. Presumably the designers just left the signal paths for
        /// the sign bit intact.
        sign: bool,
    },
}

impl<T> From<ExactOrRounded<T>> for FloatResult<T> {
    fn from(t: ExactOrRounded<T>) -> Self {
        match t {
            ExactOrRounded::Exact(t) => FloatResult::Exact(t),
            ExactOrRounded::Rounded(t) => FloatResult::Rounded(t),
        }
    }
}

impl<T> FloatResult<T> {
    pub fn is_exact(&self) -> bool {
        match self {
            FloatResult::Exact(_) => true,
            _ => false,
        }
    }
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

    pub fn unwrap_exact_or_rounded(self) -> T {
        match self {
            FloatResult::Exact(t) => t,
            FloatResult::Rounded(t) => t,
            _ => panic!("called `unwrap_exact_or_rounded` on a {:?}", self),
        }
    }
    // FIXME: Try separating "exact-or-rounded" into its own enum
}

impl FloatResult<f32> {
    /// Extract the (possibly rounded or NaN-propagated) result.
    pub fn into_inner(self) -> f32 {
        match self {
            FloatResult::Exact(t) => t,
            FloatResult::Rounded(t) => t,
            FloatResult::TooLarge => f32::INFINITY,
            FloatResult::TooSmall => f32::NEG_INFINITY,
            FloatResult::LostNaN(t) => t,
            FloatResult::InvalidOperand { sign: false } => f32::NAN,
            FloatResult::InvalidOperand { sign: true } => -f32::NAN,
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
            FloatResult::InvalidOperand { sign: false } => f64::NAN,
            FloatResult::InvalidOperand { sign: true } => -f64::NAN,
        }
    }
}

impl FloatResult<f80> {
    /// Extract the (possibly rounded) result.
    pub fn into_inner(self) -> f80 {
        match self {
            FloatResult::Exact(t) => t,
            FloatResult::Rounded(t) => t,
            FloatResult::TooLarge => f80::INFINITY,
            FloatResult::TooSmall => f80::NEG_INFINITY,
            FloatResult::LostNaN(t) => t,
            FloatResult::InvalidOperand { sign: false } => f80::NAN,
            FloatResult::InvalidOperand { sign: true } => -f80::NAN,
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
    /// fractional part of the significand).
    Denormal {
        sign: bool,
        /// The integer bit (bit #63). If set, this is a pseudo-denormal, which
        /// isn't generated by the coprocessor. It should generally be `false`,
        /// but is respected in calculations when `true`.
        integer_bit: bool,
        /// The fractional part of the significand (63 bits).
        fraction: u64,
    },
    /// All-one exponent, MSb of significand is 1, rest 0.
    Inf {
        sign: bool,
    },
    /// A quiet or signaling NaN. All-one exponent, MSb of significand is set,
    /// rest is anything but zero.
    NaN {
        sign: bool,
        signaling: bool,
        /// The payload carried in the lower 62 bits of the significand.
        ///
        /// If the payload of a quiet NaN is 0, this is an "Indefinite" result
        /// created by undefined calculations (root/logarithm of negative
        /// numbers, 0/0, Inf/Inf, ...) or using an otherwise invalid operand.
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
    pub const INDEFINITE: Self = Classified::NaN { sign: false, signaling: false, payload: 0 };

    /// Converts this classified representation back into an equivalent `f80`.
    pub fn pack(&self) -> f80 {
        let (sign, raw_exponent, significand): (bool, u16, u64) = match self {
            Classified::Zero { sign } => {
                (*sign, 0, 0)
            }
            Classified::NaN { sign, signaling: true, payload } => {
                (*sign, !0, (0b10 << 62) | payload)
            }
            Classified::NaN { sign, signaling: false, payload } => {
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
        let significand: u128 = u128::from(significand);

        let raw = sign | raw_exp | significand;
        //println!("pack: {:?} -> {:#020X}", self, raw);
        f80(raw)
    }

    /// If `self` is neither a NaN, Infinity and Zero value, decomposes it into
    /// its components.
    pub fn decompose(&self) -> Option<Decomposed> {
        let (sign, exponent, significand) = match self {
            Classified::Normal { sign, exponent, fraction } => {
                // Normalized floats always have the integer bit (#63) set
                (*sign, *exponent, fraction | (1 << 63))
            }
            Classified::Denormal { sign, integer_bit, fraction } => {
                // Denormals have a fixed exponent of -16382
                let int = if *integer_bit { 1 << 63 } else { 0 };
                (*sign, -16382, int | fraction)
            }
            Classified::Zero { sign } => {
                // Select the minimum exponent to ensure that exponent alignment
                // chooses the other operand's exponent.
                (*sign, -16382, 0)
            }
            _ => return None,
        };

        Some(Decomposed::new(sign, exponent, significand.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate env_logger;

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
            Classified::NaN { sign: false, signaling: false, payload: 0 } => {},
            e => panic!("f80::NAN ({:?}) is {:?}", f80::NAN, e),
        }
    }

    #[test]
    fn min_max_exp() {
        assert_eq!(f80::MIN_EXPONENT, -16382);
        assert_eq!(f80::MAX_EXPONENT, 16383);
    }

    fn test_from_f32(bits: u32) {
        env_logger::try_init().ok();

        let f32 = f32::from_bits(bits);
        let f8 = f80::from(f32);
        let same = f8.to_f32_checked().unwrap_exact();

        assert_eq!(
            f32.to_bits(), same.to_bits(),
            "{}->{} ({:#010X}->{:#010X}) ({:?}={:?}={:?})",
            f32, same, f32.to_bits(), same.to_bits(), f8, f8.decompose(),
            f8.classify()
        );
    }

    #[test]
    fn from_zero_f32() {
        test_from_f32(0);
    }

    #[test]
    fn from_tiny_f32() {
        test_from_f32(1);
    }

    /// Ensure a normal f32 can be roundtripped via `from_f32` and `to_f32`.
    #[test]
    fn f32_roundtrip_normal() {
        test_from_f32(8388608);
    }

    fn test_from_f64(bits: u64) {
        env_logger::try_init().ok();

        let f64 = f64::from_bits(bits);
        let f8 = f80::from(f64);
        let same = f8.to_f64_checked().unwrap_exact();

        assert_eq!(
            f64.to_bits(), same.to_bits(),
            "{}->{} ({:#010X}->{:#010X}) ({:?}={:?}={:?})",
            f64, same, f64.to_bits(), same.to_bits(), f8, f8.decompose(),
            f8.classify()
        );
    }

    #[test]
    fn f64_roundtrip_bug_1() {
        test_from_f64(9223372036854775809);
    }

    #[test]
    fn f64_roundtrip_bug_2() {
        test_from_f64(9218868437227405312);
    }
}
