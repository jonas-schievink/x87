//! This file contains tests that verify correctness against the host's FPU.
//!
//! Naturally, this only works when the host is x86 or x86-64.

#![feature(asm, untagged_unions)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[macro_use] extern crate x87;

use x87::{X87State, f80};

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
