#![feature(asm, untagged_unions)]
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

extern crate x87;
//#[macro_use] extern crate quickcheck;

use x87::X87State;

union X87StateUnion {
    raw: [u8; 108],
    structured: X87State,
}

/// Executes assembly instructions and returns the host FPU's state after the
/// operations.
macro_rules! host_state_after {
    ( $($op:tt)* ) => {{
        let mut backup = X87StateUnion { raw: [0; 108] };
        let mut state = X87StateUnion { raw: [0; 108] };
        unsafe {
            asm!(concat!(r"
            fnsave $0
            ",
            $(stringify!($op),)*
            r"
            fnsave $1
            frstor $0
            ")
            // =* -> indirect output (inout - the output is a pointer that's also an input)
            : "=*m"(&mut backup.raw), "=*m"(&mut state.raw)
            :: "memory");
            state.structured.scrub()
        }
    }};
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

    let state2 = host_state_after!(fld1);

    assert_eq!(unsafe { state.structured.scrub() }, state2);

}
