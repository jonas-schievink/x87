/// Executes assembly instructions on the host and returns the host FPU's state
/// after the operations.
///
/// The FPU's state is restored to its original state after the instructions
/// have been executed.
#[macro_export]
//#[cfg(test)]  // FIXME apparently doesn't work with integration tests
macro_rules! run_host_asm {
    ( $($args:tt)+ ) => {{
        union X87StateUnion {
            raw: [u8; 108],
            structured: $crate::X87State,
        }

        let mut backup = X87StateUnion { raw: [0; 108] };
        let mut state = X87StateUnion { raw: [0; 108] };
        unsafe {
            asm!("fnsave $0" : "=*m"(&mut backup.raw) : : "memory" : "volatile");
            asm!($($args)+ :: "memory" : "volatile");
            asm!("fnsave $1\nfrstor $0"
                : "=*m"(&mut backup.raw), "=*m"(&mut state.raw) : : "memory" : "volatile");

            state.structured.scrub()
        }
    }};
}
