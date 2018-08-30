//! x87 floating point processor emulator.
//!
//! This is an emulator for "modern-day" x87 floating point operations,
//! "modern-day" meaning any implementation in the Pentium era or beyond.
//! Emulating the original 8087 is not the goal of this project.
//!
//! Protected mode and linear addressing (no segmentation) is assumed. The host
//! is assumed to use little-endian.

#![doc(html_root_url = "https://docs.rs/x87/0.1.0")]
#![warn(missing_debug_implementations)]
//#![warn(missing_docs)]    // FIXME

#[macro_use] extern crate bitflags;
#[macro_use] extern crate log;
extern crate ieee754;
extern crate core;

mod decomposed;
mod f80_algo;
mod f80_mod;
mod f80_ops;
mod significand;
mod sign_mag;

pub use f80_mod::*;

/// The possible x87 exceptions that can be raised.
#[derive(Debug)]
pub enum Exception {
    /// Inexact result exception.
    Precision,
    /// Numeric underflow.
    Underflow,
    /// Numeric overflow.
    Overflow,
    /// A finite non-zero operand was divided by zero.
    ZeroDivide,
    /// A denormal operand was used or loaded from memory as a float or double.
    Denormalized,
    /// Invalid arithmetic operation or register stack over/underflow.
    InvalidOperation,
}

/// The different rounding modes supported by the x87.
#[derive(Debug)]
pub enum RoundingMode {
    /// Round results to the nearest representable number. If both surrounding
    /// numbers have the same distance, round to the even number ("ties to
    /// even").
    Nearest,
    /// Round towards `-Inf`.
    Down,
    /// Round towards `+Inf`.
    Up,
    /// Round towards 0 (truncate).
    Zero,
}

impl Default for RoundingMode {
    fn default() -> Self {
        RoundingMode::Nearest
    }
}

bitflags! {
    pub struct StatusRegister: u16 {
        /// FPU busy.
        const BUSY = 1 << 15;
        /// `TOP` value mask.
        const TOP  = 0b00111000_00000000;
        /// Condition code / Status flag mask.
        const CC   = 0b01000111_00000000;
        /// Exception summary.
        ///
        /// This is set to 1 iff one or more unmasked exceptions have been
        /// generated.
        const ES   = 1 << 7;
        /// Stack fault.
        ///
        /// When set to 1, an invalid operation exception means that a stack
        /// over/underflow happened.
        const SF   = 1 << 6;
        /// Precision exception.
        const PE   = 1 << 5;
        /// Numeric underflow exception.
        const UE   = 1 << 4;
        /// Numeric overflow exception.
        const OE   = 1 << 3;
        /// Zero divide exception.
        const ZE   = 1 << 2;
        /// Denormalized operand exception.
        const DE   = 1 << 1;
        /// Invalid operation exception.
        const IE   = 1 << 0;
    }
}

impl Default for StatusRegister {
    fn default() -> Self {
        StatusRegister::from_bits(0).unwrap()
    }
}

bitflags! {
    pub struct ControlRegister: u16 {
        /// Infinity control.
        ///
        /// This is ignored by the emulator and any x87 implementation after the
        /// 80287 (it was dropped with the 80387).
        const X  = 1 << 12;
        /// Rounding control.
        ///
        /// * `0b00`: Round to nearest (even) - default value.
        /// * `0b01`: Round down (towards `-Inf`).
        /// * `0b10`: Round up (towards `+Inf`).
        /// * `0b11`: Round toward zero (truncate).
        const RC = 0b00001100_00000000;
        /// Precision control.
        ///
        /// * `0b00`: Single precision (24 bit significand, 32 bit values)
        /// * `0b01`: reserved
        /// * `0b10`: Double precision (53 bit significand, 64 bit values)
        /// * `0b11`: Extended precision (64 bit significand, 80 bit values) -
        ///   default value
        const PC = 0b00000011_00000000;

        /// Bit 6 is reserved, but gets initialized to 1, so this constant needs
        /// to be defined.
        const RESERVED_6 = 1 << 6;

        /// Precision exception mask.
        ///
        /// When set, the precision exception will not be generated when the
        /// `PE` bit in the status register is set.
        const PM = 1 << 5;
        /// Numeric underflow exception mask.
        const UM   = 1 << 4;
        /// Numeric overflow exception mask.
        const OM   = 1 << 3;
        /// Zero divide exception mask.
        const ZM   = 1 << 2;
        /// Denormalized operand exception mask.
        const DM   = 1 << 1;
        /// Invalid operation exception mask.
        const IM   = 1 << 0;

        /// Bitmask containing all exception mask bits.
        const EXCEPTION_MASKS = 0b00000000_00111111;
    }
}

/// The reset state of the control register masks all exceptions, sets the
/// rounding mode to "round to nearest (even)" and the precision to 64 bits
/// (yielding the full 80 bit values).
impl Default for ControlRegister {
    fn default() -> Self {
        ControlRegister::from_bits(0x037F).unwrap()
    }
}

/// 2-bit tag stored in the tag word; marks the corresponding physical register.
/*enum Tag {
    Valid   = 0b00,
    Zero    = 0b01,
    Special = 0b10,
    Empty   = 0b11,
}

impl Tag {
    /// Must only be called with a valid tag (`0b00` - `0b11`).
    fn from_raw(raw: u8) -> Self {
        match raw {
            0b00 => Tag::Valid,
            0b01 => Tag::Zero,
            0b10 => Tag::Special,
            0b11 => Tag::Empty,
            _ => panic!("invalid tag {:#b}", raw),
        }
    }
}*/

/*/// A physical FPU register (0-7).
///
/// This is independent of the `TOP` of the stack.
#[derive(Debug)]
struct PhysReg(u8);
*/
/// A register in the FPU register stack.
#[derive(Debug)]
pub struct StackReg(u8);
/*
/// Stores tag bits for the registers.
#[derive(Debug)]
struct TagWord(u16);

impl TagWord {
    pub fn tag(&self, reg: PhysReg) -> Tag {
        let raw = (self.0 >> (reg.0 * 2)) & 0b11;
        Tag::from_raw(raw as u8)
    }
}*/

/// The x87 coprocessor.
// x87 BCD coprocessor
#[derive(Debug)]
pub struct X87 {
    status: StatusRegister,
    control: ControlRegister,
    tags: u16,
    /// Linear address of the last instruction executed.
    last_instr_ptr: u32,
    last_data_ptr: u32,
    last_opcode: u16,
    registers: [f80; 8],
}

impl X87 {
    /// Creates a new x87 processor.
    ///
    /// All registers will be initialized to their reset values.
    pub fn new() -> X87 {
        Self {
            status: StatusRegister::default(),
            control: ControlRegister::default(),
            tags: 0xFFFF,
            last_instr_ptr: 0,
            last_data_ptr: 0,
            last_opcode: 0,
            registers: [f80::from_bytes([0; 10]); 8],
        }
    }

    /// Execute a decoded instruction.
    ///
    /// # Parameters
    ///
    /// * `instr`: The instruction to execute (including its operands).
    /// * `opcode`: 10-bit x87 opcode that encodes the instruction.
    /// * `addr`: Linear 32-bit address of the first instruction or prefix byte.
    ///
    /// ## Opcode
    ///
    /// The `opcode` parameter is stored in the "last instruction opcode"
    /// register. It is not used to direct execution of the instruction. The
    /// encoding of x87 instructions looks like this:
    ///
    /// ```notrust
    /// +-----------+-----------+
    /// | 1101 1xxx | mmxx xrrr |
    /// +-----------+-----------+
    /// ```
    ///
    /// Where the second byte is a Mod-Reg-R/M byte whose `Reg` field encodes
    /// part of the opcode. The `opcode` value must consist of the
    /// `xxx mmxx xrrr` sequence, the fixed bits in the first byte are ignored.
    pub fn execute(&mut self, instr: &Instr, opcode: u16, addr: u32) -> OpResult {
        trace!("x87::execute: instr={:?}, opcode={:#b}, addr={:#X}", instr, opcode, addr);

        if let Some(mem) = instr.memory_operand() {
            self.last_data_ptr = mem.addr;
        }
        self.last_instr_ptr = addr;
        self.last_opcode = opcode & 0b0000_0111_1111_1111;

        match instr {
            _ => unimplemented!("instruction {:?}", instr),
        }
    }
}

/// The result of a floating point operation.
#[must_use = "the caller might need to raise an exception"]
#[derive(Debug)]
pub enum OpResult {
    /// The operation has caused an exception.
    Exception(Exception),
    /// Operation was executed without raising an exception.
    Success,
}

/// A decoded x87 instruction with operands already loaded from memory.
///
/// This is a slightly simplified representation compared to equivalent assembly
/// and always explicitly specifies all operands. While this would allow `Instr`
/// to encode instructions that are not encodable in x87, this is not supported
/// by this library (doing it anyway might lead to bogus values in FPU registers
/// or straight up panics, so please don't).
///
/// `Instr` already contains the loaded operands. The loading has to be done by
/// the user. This library makes no attempt at modeling the (complex) x86 memory
/// system.
#[derive(Debug)]
pub enum Instr {
    /// `fadd` / `faddp` / `fiadd`
    ///
    /// Adds `dest` to `src` and stores the result in `dest`, optionally popping
    /// the register stack.
    ///
    /// To encode `fiadd`, `src` may be an integer operand that will be
    /// converted to an 80-bit float before the computation.
    Fadd {
        dest: Operand,
        src: Operand,
        /// Whether this is a `faddp` instruction. This will pop the register
        /// stack after performing the computation.
        pop: bool,
    },

    /// Push `+1.0` onto the register stack.
    Fld1,
    /// Push `log2(10)` onto the register stack.
    FldL2T,
    /// Push `log2(e)` onto the register stack.
    FldL2E,
    /// Push the x87 processor's approximation of Pi onto the register stack.
    FldPi,
    /// Push `log10(2)` onto the register stack.
    FldLg2,
    /// Push `ln(2)` onto the register stack.
    FldLn2,
    /// Push `+0.0` onto the register stack.
    FldZ,

    // TODO incomplete
}

impl Instr {
    /// Returns the memory operand, if the instruction has one.
    fn memory_operand(&self) -> Option<&MemoryOperand> {
        use self::Instr::*;

        match self {
            Fadd { src, .. } => src.as_memory_operand(),
            Fld1 | FldL2T | FldL2E | FldPi | FldLg2 | FldLn2 | FldZ => None,
        }
    }
}

/// An operand referring a register or loaded from memory.
#[derive(Debug)]
pub enum Operand {
    /// FPU register stack reference.
    St(StackReg),
    /// Operand loaded from memory.
    Mem(MemoryOperand),
}

impl Operand {
    fn as_memory_operand(&self) -> Option<&MemoryOperand> {
        if let Operand::Mem(mem) = self {
            Some(mem)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct MemoryOperand {
    pub addr: u32,
    pub kind: MemoryOperandKind,
}

#[derive(Debug)]
pub enum MemoryOperandKind {
    /// `m16int`
    Int16(i16),
    /// `m32int`
    Int32(i32),
    /// `m32fp`
    Float32(f32),
    /// `m64fp`
    Float64(f64),
}

/// In-memory x87 state (assuming 32-bit protected mode or 64-bit long mode).
///
/// This is compatible with the memory layout created by a "real" x87 using an
/// `fsave` instruction.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct X87State {
    control: u32,
    status: u32,
    /// Low 16 bits only. Stores 2 tag bits for each register.
    tag: u32,
    instr_ptr_offset: u32,
    instr_ptr_selector: u16,
    last_opcode: u16,
    data_ptr_offset: u32,
    data_ptr_selector: u32,
    /// The register stack.
    st: [[u8; 10]; 8],
}

impl X87State {
    /// Scrubs the x87 state, removing all references to instruction and operand
    /// positions in memory.
    ///
    /// This allows comparing x87 states created by executing the same
    /// operations, but at different memory locations. Otherwise, they would
    /// refer to different instruction and operand addresses.
    pub fn scrub(self) -> Self {
        Self {
            instr_ptr_selector: !0,
            instr_ptr_offset: !0,
            data_ptr_selector: !0,
            data_ptr_offset: !0,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::size_of;

    #[test]
    fn state_size() {
        assert_eq!(size_of::<X87State>(), 108);
    }

    #[test]
    fn construct() {
        let _x87 = X87::new();
    }
}
