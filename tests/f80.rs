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
        let same = f80.to_f64_checked();
        println!("{}->{:?} ({:#010X}->{:#010X}) {:?}={:?}", f, same, f.to_bits(), same.into_inner().to_bits(), f80, f80.classify());
        let same = same.unwrap_exact();
        assert_eq!(f.to_bits(), same.to_bits());
    }
}

proptest! {
    #[test]
    fn f80_neg(bytes: [u8; 10]) {
        let f = f80::from_bytes(bytes);
        let sign = f.is_sign_negative();
        assert_eq!((-f).is_sign_negative(), !sign);
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

        assert_eq!(f32.to_bits(), same.to_bits());
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
