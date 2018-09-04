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
    run_host_asm!(r"
        flds $0
        flds $1
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

// Note that many of the proptests are duplicated in `f80.rs` - the versions in
// there do not need asm! or an x86 host as they test against the operations on
// `f32`/`f64`. The ones in here test against the host FPU.

proptest! {
    #[test]
    fn add_f32(lhs_bits: u32, rhs_bits: u32) {
        add32(lhs_bits, rhs_bits);
    }
}
