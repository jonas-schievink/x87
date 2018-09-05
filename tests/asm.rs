//! This file contains tests that verify correctness against the host's FPU.
//!
//! Naturally, this only works when the host is x86 or x86-64. Due to the use of
//! inline assembly, it also requires nightly Rust.

#![feature(asm, untagged_unions)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[macro_use] extern crate x87;
#[macro_use] extern crate proptest;
extern crate env_logger;
extern crate ieee754;

use x87::{X87State, f80};
use ieee754::Ieee754;

union X87StateUnion {
    raw: [u8; 108],
    structured: X87State,
}

/// Tests that our test framework does something reasonable.
#[test]
fn meta() {
    let mut backup = X87StateUnion { raw: [0; 108] };
    let mut state = X87StateUnion { raw: [0; 108] };
    unsafe {
        asm!(r"
        fnsave $0
        fld1
        fnsave $1
        frstor $0
        "
        // =* -> indirect output (inout - the output is a pointer that's also an input)
        : "=*m"(&mut backup.raw), "=*m"(&mut state.raw)
        :: "memory");
    }

    let state2 = run_host_asm!("fld1" :);

    assert_eq!(unsafe { state.structured.scrub() }, state2);
}

/// This is a case where f80 addition results in a different result than f64
/// addition.
///
/// Due to double rounding, the f80 result is 1 ULP below the f64 result.
///
/// This test checks that the host FPU's result is the same as our result and
/// differs in the expected way from the native `f64+f64` result.
#[test]
fn add_f64_double_round_wrong() {
    let lhs = f64::from_bits(964674174654497230);
    let rhs = f64::from_bits(10131472521302454270);
    let mut result80 = 0.0f64;

    run_host_asm!(r"
        fldl $1
        fldl $2
        faddp
        fstpl $0
    " : "=*m"(&mut result80), "=*m"(&lhs), "=*m"(&rhs));
    println!("{}+{}={}", lhs, rhs, result80);

    // Ensure that the wrong result is exactly what we expect.
    let result64 = lhs + rhs;
    assert_eq!(result64.to_bits(), result80.to_bits() + 1, "host FPU returned wrong result");

    let (l80, r80) = (f80::from(lhs), f80::from(rhs));
    let f80sum = l80 + r80;
    let f80bits = f80sum.to_f64().to_bits();
    assert_eq!(result80.to_bits(), f80bits, "host FPU != emulation result");
}

fn add32(lhs_bits: u32, rhs_bits: u32) {
    let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));

    let mut native_f32_sum = 0.0f32;
    let mut native_f80_sum = [0u8; 10];
    // Note that operands are pushed in reversed order
    run_host_asm!(r"
        flds $1
        flds $0
        faddp
        fsts $2
        fstpt $3
    " : "=*m"(&lhs), "=*m"(&rhs), "=*m"(&mut native_f32_sum), "=*m"(&mut native_f80_sum));

    let (l80, r80) = (f80::from(lhs), f80::from(rhs));
    let f80sum = l80 + r80;
    let f80_f32bits = f80sum.to_f32().to_bits();

    let f80native = f80::from_bytes(native_f80_sum);
    assert_eq!(
        f80_f32bits, native_f32_sum.to_bits(),
        "f32 sum mismatch: x87:{}={:?}={:#010X}={:?}, native:{}={:?}={:#010X}={:?}",
        f80sum.to_f32(), f80sum.to_f32().classify(), f80_f32bits, f80sum.to_f32().decompose(),
        native_f32_sum, native_f32_sum.classify(), native_f32_sum.to_bits(), native_f32_sum.decompose(),
    );
    assert_eq!(
        f80sum.to_bytes(), native_f80_sum,
        "f80 sum mismatch: x87:{:?}={:?}, native:{:?}={:?}",
        f80sum, f80sum.classify(), f80native, f80native.classify(),
    );
}

fn sub32(lhs_bits: u32, rhs_bits: u32) {
    let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));

    let mut native_f32_diff = 0.0f32;
    let mut native_f80_diff = [0u8; 10];
    // Note that operands are pushed in reversed order
    run_host_asm!(r"
        flds $1
        flds $0
        fsubp
        fsts $2
        fstpt $3
    " : "=*m"(&lhs), "=*m"(&rhs), "=*m"(&mut native_f32_diff), "=*m"(&mut native_f80_diff));

    let (l80, r80) = (f80::from(lhs), f80::from(rhs));
    let f80diff = l80 - r80;
    let f80_f32bits = f80diff.to_f32().to_bits();

    let f80native = f80::from_bytes(native_f80_diff);
    assert_eq!(
        f80_f32bits, native_f32_diff.to_bits(),
        "f32 sum mismatch: x87:{}={:?}={:#010X}={:?}, native:{}={:?}={:#010X}={:?}",
        f80diff.to_f32(), f80diff.to_f32().classify(), f80_f32bits, f80diff.to_f32().decompose(),
        native_f32_diff, native_f32_diff.classify(), native_f32_diff.to_bits(), native_f32_diff.decompose(),
    );
    assert_eq!(
        f80diff.to_bytes(), native_f80_diff,
        "f80 sum mismatch: x87:{:?}={:?}, native:{:?}={:?}",
        f80diff, f80diff.classify(), f80native, f80native.classify(),
    );
}

fn mul32(lhs_bits: u32, rhs_bits: u32) {
    let (lhs, rhs) = (f32::from_bits(lhs_bits), f32::from_bits(rhs_bits));

    let mut native_f32_prod = 0.0f32;
    let mut native_f80_prod = [0u8; 10];
    // Note that operands are pushed in reversed order
    run_host_asm!(r"
        flds $1
        flds $0
        fmulp
        fsts $2
        fstpt $3
    " : "=*m"(&lhs), "=*m"(&rhs), "=*m"(&mut native_f32_prod), "=*m"(&mut native_f80_prod));

    let (l80, r80) = (f80::from(lhs), f80::from(rhs));
    let f80prod = l80 * r80;
    let f80_f32bits = f80prod.to_f32().to_bits();

    let f80native = f80::from_bytes(native_f80_prod);
    assert_eq!(
        f80_f32bits, native_f32_prod.to_bits(),
        "f32 product mismatch: x87:{}={:?}={:#010X}={:?}, native:{}={:?}={:#010X}={:?}",
        f80prod.to_f32(), f80prod.to_f32().classify(), f80_f32bits, f80prod.to_f32().decompose(),
        native_f32_prod, native_f32_prod.classify(), native_f32_prod.to_bits(), native_f32_prod.decompose(),
    );
    assert_eq!(
        f80prod.to_bytes(), native_f80_prod,
        "f80 product mismatch: x87:{:?}={:?}, native:{:?}={:?}",
        f80prod, f80prod.classify(), f80native, f80native.classify(),
    );
}

/// Discrepancy in NaN payload propagation between the crate and host FPU.
///
/// Converting an f32 NaN to f80 should place its payload in the upper bits of
/// the f80. We used the lower bits.
#[test]
fn f32_add_nan_payload() {
    env_logger::try_init().ok();
    add32(2139095041, 0);
}

#[test]
fn nan_propagation() {
    env_logger::try_init().ok();
    add32(0xff800002, 0x7f800001);
    add32(0x7f800002, 0xff800001);
    add32(0xff800002, 0xff800001);
    add32(0x7f800002, 0x7f800001);

    add32(0xff800001, 0x7f800002);
    add32(0x7f800001, 0xff800002);
    add32(0xff800001, 0xff800002);
    add32(0x7f800001, 0x7f800002);
}

/// Rounding a fraction of `.111111` would not change the integer bits, but has
/// to.
#[test]
fn rounding_affects_integer_bits() {
    env_logger::try_init().ok();
    add32(1, 3976200192);
}

/// Missing the correct rounding and postnormalization steps in `to_f32_checked`
/// result in this failing.
#[test]
fn to_f32_postnormalizes() {
    env_logger::try_init().ok();
    add32(3120562177, 1518338048);
}

/// This fails if addition of equal-magnitude but opposing-sign values results
/// in `-0.0`.
#[test]
fn addition_doesnt_create_signed_zero() {
    env_logger::try_init().ok();
    add32(54623649, 2202107297);
}

/// Checks addition of infinities against x87 HW.
#[test]
fn infinities() {
    let pinf: f32 = 1.0/0.0;    // +Inf
    let minf: f32 = -1.0/0.0;   // -Inf
    add32(pinf.to_bits(), pinf.to_bits());
    add32(minf.to_bits(), minf.to_bits());
    // These return a signed NaN on a real x87:
    add32(minf.to_bits(), pinf.to_bits());
    add32(pinf.to_bits(), minf.to_bits());
}

/// Test that a zero operand won't cause the exponent alignment code to drop
/// bits of the other operand.
///
/// Could also solve this by special-casing 0.0 when adding.
#[test]
fn zero_exponent() {
    env_logger::try_init().ok();
    add32(2147483649, 0);
}

/// `0.0 - NaN = NaN`.
///
/// We returned `-NaN` instead.
#[test]
fn zero_minus_nan() {
    env_logger::try_init().ok();
    sub32(0, 2139095041);
}

#[test]
fn mul_denormal_zero() {
    env_logger::try_init().ok();
    mul32(0, 1);
    mul32(1, 0);
}

#[test]
fn mul_f32_denormals() {
    env_logger::try_init().ok();
    mul32(1, 3);
}

/// There appears to be a bug when encoding a denormal f32 result that causes a
/// deviation of 1 ULP against both native x87 and IEEE f32 arithmetic.
///
/// (this also causes `mul_f32` to fail)
#[test]
#[ignore]   // FIXME find and fix this bug
fn mul_rounding_denormal_result() {
    env_logger::try_init().ok();
    mul32(2496593444, 706412423);
}

// Note that many of the proptests are duplicated in `f80.rs` - the versions in
// there do not need asm! or an x86 host as they test against the operations on
// `f32`/`f64`. The ones in here test against the host FPU.

proptest! {
    #[test]
    fn add_f32(lhs_bits: u32, rhs_bits: u32) {
        add32(lhs_bits, rhs_bits);
    }
}

proptest! {
    #[test]
    fn sub_f32(lhs_bits: u32, rhs_bits: u32) {
        sub32(lhs_bits, rhs_bits);
    }
}

proptest! {
    #[test]
    fn mul_f32(lhs_bits: u32, rhs_bits: u32) {
        mul32(lhs_bits, rhs_bits);
    }
}
