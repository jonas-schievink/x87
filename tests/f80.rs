extern crate x87;
#[macro_use] extern crate proptest;

use x87::*;

use std::u32;

proptest! {
    #[test]
    fn to_from_bytes_roundtrip(bytes: [u8; 10]) {
        let f = f80::from_bytes(bytes);
        assert_eq!(f.to_bytes(), bytes);
    }
}

proptest! {
    /// Tests that calling `.classify().pack()` on a value results in the same
    /// value.
    #[test]
    fn f80_classify_roundtrip(bytes: [u8; 10]) {
        let f = f80::from_bytes(bytes);
        let classified = f.classify_checked();
        if let Some(classified) = classified {
            let back = classified.pack();
            // bitwise comparison
            if f.to_bytes() != back.to_bytes() {
                let back_cls = back.classify();
                panic!("{:?} != {:?} ({:?} vs. {:?})", f, back, classified, back_cls);
            }
        }
    }
}

proptest! {
    /// Tests that `.decompose()` followed by `from_decomposed` yields the
    /// original value.
    #[test]
    fn f80_decompose_roundtrip(raw: [u8; 10]) {
        let f8 = f80::from_bytes(raw);
        if let Some(decomp) = f8.decompose() {
            f80::from_decomposed(decomp).unwrap_exact();
        }
    }
}

proptest! {
    #[test]
    fn f80_f32_roundtrip(f: u32) {
        let f = f32::from_bits(f);
        let f80 = f80::from_f32(f);
        let same = f80.to_f32_checked().unwrap_exact();
        assert_eq!(f.to_bits(), same.to_bits(), "{}->{} ({:#010X}->{:#010X}) {:?}={:?}", f, same, f.to_bits(), same.to_bits(), f80, f80.classify());
    }
}

proptest! {
    #[test]
    fn f80_f64_roundtrip(f: u64) {
        let f = f64::from_bits(f);
        let f80 = f80::from_f64(f);
        let same = f80.to_f64_checked().unwrap_exact();
        assert_eq!(f.to_bits(), same.to_bits(), "{}->{} ({:#010X}->{:#010X}) {:?}={:?}", f, same, f.to_bits(), same.to_bits(), f80, f80.classify());
    }
}

proptest! {
    /// Ensures that unary negation always flips the sign of a value.
    #[test]
    fn f80_neg(bytes: [u8; 10]) {
        let f = f80::from_bytes(bytes);
        let sign = f.is_sign_negative();
        assert_eq!((-f).is_sign_negative(), !sign);
    }
}

proptest! {
    #[test]
    fn f80_eq(lhs_bits: u32, rhs_bits: u32) {
        let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));

        let l80 = f80::from(lhs);
        let r80 = f80::from(rhs);
        if !l80.is_nan() {
            assert_eq!(l80, l80);
        }
        if !r80.is_nan() {
            assert_eq!(r80, r80);
        }
        let f32eq = lhs == rhs;
        assert_eq!(l80 == r80, f32eq, "{:#010X}=={:#010X}:{}", lhs_bits, rhs_bits, f32eq);
        let f32eq_reversed = rhs == lhs;
        assert_eq!(r80 == l80, f32eq_reversed);
    }
}

proptest! {
    #[test]
    fn add_f32(lhs_bits: u32, rhs_bits: u32) {
        let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));
        let f32sum = lhs + rhs;
        let f32bits = f32sum.to_bits();

        let (l80, r80) = (f80::from(lhs), f80::from(rhs));
        let f80sum = l80 + r80;
        let f80bits = f80sum.to_f32().to_bits();

        println!("{:#010X}+{:#010X}={:#010X} ({}+{}={})", lhs_bits, rhs_bits, f32sum.to_bits(), lhs, rhs, f32sum);
        println!("f80: {:?}+{:?}={:?}", l80.classify(), r80.classify(), f80sum.classify());
        if f32sum.is_nan() {
            // NaN propagation might not work the same way on x87 and the host's
            // implementation of IEEE, so just require that both impls return
            // NaN here.
            assert!(f80sum.is_nan());
        } else {
            // For non-NaN results, require bitwise equality.
            prop_assert_eq!(f32bits, f80bits);
        }
    }
}

proptest! {
    #[test]
    fn add_f64(lhs_bits: u64, rhs_bits: u64) {
        let (lhs, rhs) = (f64::from_bits(lhs_bits), f64::from_bits(rhs_bits));
        let f64sum = lhs + rhs;
        let f64bits = f64sum.to_bits();

        let (l80, r80) = (f80::from(lhs), f80::from(rhs));
        let f80sum = l80 + r80;
        let f80bits = f80sum.to_f64().to_bits();

        println!("{:#010X}+{:#010X}={:#010X} ({}+{}={})", lhs_bits, rhs_bits, f64sum.to_bits(), lhs, rhs, f64sum);
        println!("f80: {:?}+{:?}={:?}", l80.classify(), r80.classify(), f80sum.classify());

        // Mimicking f64 operations via conversion to f80 might yield the wrong
        // result due to double rounding. The difference should not be more than
        // 1 ULP however, so accept any result that's "close enough".
        prop_assert!(f64bits - 1 <= f80bits && f80bits <= f64bits + 1);
    }
}

/// Exhaustively test that conversion from `f32` to `f80` and back is lossless.
///
/// Takes about a minute to run with optimizations, so it's ignored by default.
/// You can run this via `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn f32_f80_roundtrip_exhaustive() {
    for i in 0..=u32::MAX {
        let f32 = f32::from_bits(i);
        let f80 = f80::from(f32);
        let same = f80.to_f32();

        assert_eq!(
            f32.to_bits(), same.to_bits(),
            "{}->{} ({:#010X}->{:#010X}) ({:?}={:?}={:?})",
            f32, same, f32.to_bits(), same.to_bits(), f80, f80.decompose(),
            f80.classify()
        );
    }
}

#[test]
fn inf() {
    let raw = 2139095040;

    let f = f32::from_bits(raw);
    let f80 = f80::from(f);
    let same = f80.to_f32();
    assert_eq!(f.to_bits(), same.to_bits(), "{}->{} ({:#010X}->{:#010X}) {:?}={:?}", f, same, f.to_bits(), same.to_bits(), f80, f80.classify());
}
