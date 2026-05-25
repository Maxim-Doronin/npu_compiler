# Bytecode Format

This document describes the bytecode format used by the NPU compiler for representing the orchestration of kernel execution (i.e. NPU blobs). This binary format is intended to be interpreted by a Virtual Machine (VM), which performs the actual execution during inference. As this format is able to describe the control flow, it is able to support multiple complex use-cases, such as:

- dynamic models, whose data shape is only known during inference
- repeated parametrized function calls, such as for functions encountered in the architecture of LLMs (also known as repeating blocks execution)
- dynamically-dispatched kernels, based on the NPU platform, when diverse platform-specific kernels are contained within the same bytecode file
- weights separation, where the schedule is split between an initialization stage and execution stage

## Table of Contents
* [High-Level Overview](#1-high-level-overview)
* [Bytecode Format Prerequisites](#2-bytecode-format-prerequisites)
* [Bytecode File Structure](#3-bytecode-file-structure)
* [Opcodes](#4-opcodes)
* [Bytecode Dialect](#5-bytecode-dialect)
* [Virtual Machine](#6-virtual-machine)

## 1. High-Level Overview

![Bytecode dialect -> Bytecode file -> NPU VM Runtime](../assets/bytecode_overview.svg "Bytecode High-Level Overview")

## 2. Bytecode Format Prerequisites

### 2.1. Endianness

All multi-byte elements inside the format are stored using the little-endian byte order. This includes primitive types, such as integers, as well as the instructions themselves (opcode + operands).

## 3. Bytecode File Structure

The bytecode format is comprised of a file header, followed by multiple sections. The file header contains the following entries:

```
magic_number: "NPUByte\x00",
version: major.minor.patch (uint16_t.uint16_t.uint16_t),
section_header_table: section_header_table,
sections: uint8_t[]
```

### 3.1. Magic Number

The header begins with a magic number, which is a unique binary identifier used for this particular bytecode format. The chosen magic number is represented by the following 8 bytes:

```
"NPUByte\x00"
```

### 3.2. Version

The bytecode format follows the [Semantic Versioning](https://semver.org/) scheme. The `major` number is used to represent changes that break backward compatibility, the `minor` number represents backward compatible, but not forward compatible changes, and the `patch` number represents backward compatible changes.

#### Compatibility

The format aims to provide the following compatibility guarantees:
- backward compatibility: binaries serialized by older versions of the compiler are compatible with newer runtimes
- forward compatibility: binaries serialized by newer versions of the compiler are compatible with older runtimes

Currently, there is no fixed compatibility window for this format. The intention is to provide compatibility as long as possible. If this changes, it will be documented here.

In order to ensure compatibility, some constraints have to be followed. On the compiler side:
- Every operation in the bytecode contains the version where it has been introduced, which should not be changed. Once an operation has been added, it cannot be removed or changed, in a way that would break compatibility (e.g. no semantic changes, no operator mutations, no attribute changes etc). Attributes and types are also versioned. An operation can only use attributes and types that have a version lower or equal to its own.
- Adding a new operation implies increasing the minor number, as the new operation must have this newer version set as the version where it was introduced.
- The compiler has a bytecode target version for serialization. It must only use operations, attributes and types that have the version lower or equal to the target version.
- Every supported NPU platform has a default target version which is used for compilation. Increasing this target version means that forward compatibility is broken. During compilation, a user may toggle specific features which can affect the target bytecode version. If this happens, the compiler reports a warning to inform the user about the minimum runtime version required.

On the runtime side:
- The virtual machine knows what is the latest bytecode version it supports. It will check if the version in the file is compatible and stop the execution otherwise.
- The virtual machine must maintain the support for all previous versions of the bytecode format, as long as backward compatibility is intended to be maintained.

##### Compatibility Testing

On the compiler side, every operation has its serialization tested against all supported target versions. For an operation introduced in version N, the following must be tested:

- The serialization from bytecode IR with targets N, N+1, N+2 etc.
    - these tests should produce the equivalent opcode in binary form
- The serialization from bytecode IR with targets N-1, N-2 etc.
    - these tests will check for expected failures

On the runtime side, pre-compiled bytecode binaries are used to test the compatibility of the VM. These binaries cover all of the existing bytecode versions, which ensure that the interpreter remains compatible with previous versions. All operations are covered by these binaries (e.g. all instructions).

### 3.3. Section Header Table

The sections header table describes every section found within the file. It contains the following:

```
section_header_table {
    num_sections: uint64_t,            // The number of sections inside the file
    section_headers: section_header[]  // The header of every section. The number of headers corresponds with the number of sections
}

section_header {
    type: uint8_t,         // The type of the section
    name_index: uint64_t,  // The name of the section, specified using an index inside the string section
    offset: uint64_t,      // The offset of the section's data within the file
    size: uint64_t,        // The size in bytes of the section
    info: uint8_t[]        // Extra information about the section, which helps interpret the section's data. The information contained depends on the section type
}
```

Each section type has a unique identifier. The format supports the following section types:

#### Function Section

This section contains all of the functions that are executed by the VM. It has the type identifier `0x00`. The info part of the section header has the following structure:

```
info {
    num_functions: uint64_t,              // The number of functions present in the section
    entrypoint_function_index: uint64_t,  // The index of the entrypoint function for execution
    function_info: function_info[]        // Per-function information fields. The number of information fields corresponds with the number of functions
}

function_info {
    name_index: uint64_t,             // The unique name of the function, specified using an index inside the string section
    function_type_index: uint64_t,    // The signature type of the function, specified using an index inside the type section
    num_general_registers: uint64_t,  // The number of general registers used by the function
    body_offset: uint64_t,            // The starting offset of the function's body within the section
    body_size: uint64_t               // The size in bytes of the function's body
}
```

#### Constant Section

This section contains constants that are used by functions during execution. It has the type identifier `0x01`. The info part of the section header has the following structure:

```
info {
    num_constants: uint64_t,        // The number of constants present in the section
    constant_info: constant_info[]  // Per-constant information fields. The number of information fields corresponds with the number of constants
}

constant_info {
    constant_offset: uint64_t,  // The starting offset of the constant within the section
    constant_size: uint64_t,    // The size in bytes of the constant
}
```

#### String Section

This section contains strings referenced by the other sections, such as the function section. It contains strings used during execution (e.g. assert messages), as well as the names of the functions. The strings are null-terminated. Having the strings placed in this dedicated section, instead of being inlined, can help us avoid duplicating their value and therefore reduce the potential size of the bytecode format.

It has the type identifier `0x02`. The info part of the section header has the following structure:

```
info {
    num_strings: uint64_t,      // The number of strings present in the section
    string_info: string_info[]  // Per-string information fields. The number of information fields corresponds with the number of strings
}

string_info {
    string_offset: uint64_t,  // The starting offset of the string within the section
    string_size: uint64_t,    // The size in bytes of the string
}
```

#### Kernel Section

This section contains the binary values of the kernels that will be executed on the NPU device. These kernels are referenced by the function bodies, when they are called for execution.

It has the type identifier `0x03`. The info part of the section header has the following structure:

```
info {
    num_kernels: uint64_t,      // The number of kernels present in the section
    kernel_info: kernel_info[]  // Per-kernel information fields. The number of information fields corresponds with the number of kernels
}

kernel_info {
    kernel_offset: uint64_t,  // The starting offset of the kernel within the section
    kernel_size: uint64_t,    // The size in bytes of the kernel
}
```

#### Type Section

This section contains the type definitions used throughout the file. This includes data types (e.g. integer, floating-point), buffer types, function signature types etc.

It has the type identifier `0x04`. The info part of the section header has the following structure:

```
info {
    num_types: uint64_t,    // The number of types present in the section
    type_info: type_info[]  // Per-type information fields. The number of information fields corresponds with the number of types
}

type_info {
    type_offset: uint64_t,  // The starting offset of the type within the section
    type_size: uint64_t,    // The size in bytes of the type
}
```

The type data itself is encoded differently for each supported type definition. Each type definition has an identifier that allows the bytecode parser to interpret the rest of the fields. There can be multiple entries inside the section with the same identifier, but with different fields (e.g. one entry for 32-bit integers, one entry for 64-bit integers). The types used in the rest of the bytecode file are referenced by the type's index within this section.

Below are enumerated the supported types, as well as their binary encoding within the type section:

##### Integer Type

Any integer type, whose width is specified explicitly.

```
integer_type {
    id: uint8_t,    // 0x01
    width: uint8_t  // The number of bits inside the type
}
```

##### Floating-Point Type

Any supported floating-point type, whose width is specified explicitly.

```
float_type {
    id: uint8_t,     // 0x02
    width: uint8_t,  // The number of bits inside the type
    format: uint8_t  // The specific float format:
                     // - float (IEEE 754):              0x00
                     // - bfloat (Brain Floating Point): 0x01
                     // - tfloat (TensorFloat):          0x02
                     // - E4M3 (float8):                 0x03
                     // - E5M2 (float8):                 0x04
                     // - E2M1 (float4):                 0x05
}
```

##### Opaque Type

An opaque type, whose semantics are unknown. The width is specified explicitly. This type can be used when the other existing types cannot express it; e.g. opaque data that is passed as input to a kernel call.

```
opaque_type {
    id: uint8_t,    // 0x03
    width: uint8_t  // The number of bits inside the type
}
```

##### Buffer Type

```
buffer_type {
    id: uint8_t,                // 0x04
    data_type_index: uint64_t,  // The index of the data type inside the type section
    rank: int64_t,              // The number of dimensions of the buffer
    shape: int64_t[],           // The shape of the buffer (-1 represents a dynamic dimension)
    strides: int64_t[]          // The strides of the buffer (-1 represents a dynamic dimension)
}
```

##### Function Type

The function type describes the signature of a function whose code is interpreted by the VM. It describes the number of arguments and return values, as well as their types.

```
function_type {
    id: uint8_t,                      // 0x05
    num_args: uint16_t,               // The number of arguments passed to the function
    arg_type_indices: uint64_t[],     // Array of type indices for each argument
    num_results: uint16_t,            // The number of results returned by the function
    result_type_indices: uint64_t[],  // Array of type indices for each result
}
```

## 4. Opcodes

The bytecode format supports a predefined set of instructions, each identified by a unique opcode. Every instruction consists of the opcode, the addressing mode, and zero or more operands supplying registers or data that are used by the operation. The number of the operands is determined by the opcode.

All of the registers that are used by the instructions have 64 bits. This allows us to reuse the same registers for all instructions, even if only part of the register is used for storing the data. Every opcode has a clear specification on how many bits are used out of the operand registers (e.g. treat the value inside the register as a 32-bit or 64-bit floating-point number). In case an instruction uses only part of the register's data, the data is expected to be placed in the least-significant part of the register.

Each register is identified by a unique register number, which is represented by a signed 16-bit integer.

In the rest of the section, the following terminology is used:
- `rd`: the destination register, i.e. the register that this operation writes into
- `rs`, `rsN`: one or more source registers, i.e. a register that is used as input for the instruction; `rs` is used in case the operation utilizes a single source register, `rsN` (e.g. `rs1`, `rs2` etc.) is used in case the operation utilizes more than one source register
- `x[r...]`: to represent the value inside the register, whether it is read from or written into
- `imm`: to represent an immediate value, i.e. a direct value passed to the instruction instead of a register

### Binary Representation

The following items represent the binary representation of the instructions, based on the number of operands utilized. Every instruction utilizes the following format, from the least-significant bits to the most-significant bits:
- the opcode: the unique identifier of the instruction, stored using 14 bits
- the addressing mode: the next 2 bits after the opcode; it determines the significance of the operands (for example, it specifies whether operands should be treated as registers or direct values)
- operands: zero or more operands, where each operand is stored using 16 bits
    - Note: the `set.imm` instruction is an exception to this, as it has a 64 bit operand

#### Zero Operands

- As there are no operands, the addressing mode does not make a difference. For consistency with the other instructions, 14 bits will still be used for storing the opcode, which avoids special instruction parsing logic for this case in the interpreter.

    15-14 | 13-0
    -- | --
    00 | OPCODE

#### One Operand

1. Register-based addressing mode: the value inside the register is used by the instruction

    31-16 | 15-14 | 13-0
    -- | -- | --
    r | 00 | OPCODE

2. Direct addressing mode: the immediate value of the operand is used by the instruction

    31-16 | 15-14 | 13-0
    -- | -- | --
    imm | 01 | OPCODE

#### Two Operands

1. Register-based addressing mode: both operands are registers, and the values inside them are used by the instruction

    47-32 | 31-16 | 15-14 | 13-0
    -- | -- | -- | --
    rs | rd | 00 | OPCODE

2. Direct addressing mode: the first operand is a register and the second operand is an immediate value

    47-32 | 31-16 | 15-14 | 13-0
    -- | -- | -- | --
    imm | rd | 01 | OPCODE

#### Three Operands

1. Register-based addressing mode: all three operands are registers, and the values inside them are used by the instruction

    63-48 | 47-32 | 31-16 | 15-14 | 13-0
    -- | -- | -- | -- | --
    rs2 | rs1 | rd | 00 | OPCODE

2. Direct addressing mode: the first and second operands are registers, and the third operand is an immediate value

    63-48 | 47-32 | 31-16 | 15-14 | 13-0
    -- | -- | -- | -- | --
    imm | rs1 | rd | 01 | OPCODE

#### Four+ Operands

1. Register-based addressing mode: all operands are registers, and the values inside them are used by the instruction

    ... | 63-48 | 47-32 | 31-16 | 15-14 | 13-0
    -- | -- | -- | -- | -- | --
    ... | r3 | r2 | r1 | 00 | OPCODE

In case of instructions with a variadic number of operands, immediate values are used to specify how many operands are used by the instruction. For example, the `call` instruction has the following signature: `call rs, N, rN..., M, rM...`. In this case, `N` and `M` are immediate values that dictate how many operands follow next (e.g. `rN...` means `N` registers). Each variadic instruction's specification dictates which operands are registers and which are immediate values.

> Note: The `imm` values in the instruction specifications below are not considered as operands **for the addressing mode**. For example, for the `round.f32` / `round.f64` instructions mention operands `rd`, `rs` and `imm`. The addressing mode has no impact over `imm`, meaning that the rules for the two-operand format above is used; this means that for addressing mode two (i.e. direct addressing mode), the second operand `rs` becomes an immediate value.

### Arithmetic - Integers

There are dedicated instructions for 32 and 64-bit integer arithmetic computation. All integer values are expected to be stored in two's complement, which allows us to use the same instruction for signed and unsigned data.

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
abs.i32 | rd, rs | x[rd] = \|x[rs]\| | Absolute value of 32-bit integer | 1
abs.i64 | rd, rs | x[rd] = \|x[rs]\| | Absolute value of 64-bit integer | 1
add.i32 | rd, rs1, rs2 | x[rd] = x[rs1] + x[rs2] | Addition of two 32-bit integers | 1,2
add.i64 | rd, rs1, rs2 | x[rd] = x[rs1] + x[rs2] | Addition of two 64-bit integers | 1,2
div.i32 | rd, rs1, rs2 | x[rd] = x[rs1] / x[rs2] | Division of two 32-bit integers | 1,2
div.i64 | rd, rs1, rs2 | x[rd] = x[rs1] / x[rs2] | Division of two 64-bit integers | 1,2
max.i32 | rd, rs1, rs2 | x[rd] = max(x[rs1], x[rs2]) | Maximum value between two 32-bit integers | 1,2
max.i64 | rd, rs1, rs2 | x[rd] = max(x[rs1], x[rs2]) | Maximum value between two 64-bit integers | 1,2
min.i32 | rd, rs1, rs2 | x[rd] = min(x[rs1], x[rs2]) | Minimum value between two 32-bit integers | 1,2
min.i64 | rd, rs1, rs2 | x[rd] = min(x[rs1], x[rs2]) | Minimum value between two 64-bit integers | 1,2
mul.i32 | rd, rs1, rs2 | x[rd] = x[rs1] * x[rs2] | Multiplication of two 32-bit integers | 1,2
mul.i64 | rd, rs1, rs2 | x[rd] = x[rs1] * x[rs2] | Multiplication of two 64-bit integers | 1,2
rem.i32 | rd, rs1, rs2 | x[rd] = x[rs1] % x[rs2] | Remainder of the division between two 32-bit integers | 1,2
rem.i64 | rd, rs1, rs2 | x[rd] = x[rs1] % x[rs2] | Remainder of the division between two 64-bit integers | 1,2
sub.i32 | rd, rs1, rs2 | x[rd] = x[rs1] - x[rs2] | Subtraction of two 32-bit integers | 1,2
sub.i64 | rd, rs1, rs2 | x[rd] = x[rs1] - x[rs2] | Subtraction of two 64-bit integers | 1,2

### Arithmetic - Floating-Point Numbers

There are dedicated instructions for 32 and 64-bit floating-point arithmetic computation. The floating-point values are expected to be stored in the [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) format; in other words, the single and double-precision IEEE 754 formats are supported.

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
abs.f32 | rd, rs | x[rd] = \|x[rs]\| | Absolute value of 32-bit floating point number | 1
abs.f64 | rd, rs | x[rd] = \|x[rs]\| | Absolute value of 64-bit floating point number | 1
add.f32 | rd, rs1, rs2 | x[rd] = x[rs1] + x[rs2] | Addition of two 32-bit floating point numbers | 1,2
add.f64 | rd, rs1, rs2 | x[rd] = x[rs1] + x[rs2] | Addition of two 64-bit floating point numbers | 1,2
ceil.f32 | rd, rs | x[rd] = ceil(x[rs]) | Rounds up a 32-bit floating point number to the nearest integer | 1
ceil.f64 | rd, rs | x[rd] = ceil(x[rs]) | Rounds up a 64-bit floating point number to the nearest integer | 1
div.f32 | rd, rs1, rs2 | x[rd] = x[rs1] / x[rs2] | Division of two 32-bit floating point numbers | 1,2
div.f64 | rd, rs1, rs2 | x[rd] = x[rs1] / x[rs2] | Division of two 64-bit floating point numbers | 1,2
floor.f32 | rd, rs | x[rd] = floor(x[rs]) | Rounds down a 32-bit floating point number to the nearest integer | 1
floor.f64 | rd, rs | x[rd] = floor(x[rs]) | Rounds down a 64-bit floating point number to the nearest integer | 1
max.f32 | rd, rs1, rs2 | x[rd] = max(x[rs1], x[rs2]) | Maximum value between two 32-bit floating point numbers | 1,2
max.f64 | rd, rs1, rs2 | x[rd] = max(x[rs1], x[rs2]) | Maximum value between two 64-bit floating point numbers | 1,2
min.f32 | rd, rs1, rs2 | x[rd] = min(x[rs1], x[rs2]) | Minimum value between two 32-bit floating point numbers | 1,2
min.f64 | rd, rs1, rs2 | x[rd] = min(x[rs1], x[rs2]) | Minimum value between two 64-bit floating point numbers | 1,2
mul.f32 | rd, rs1, rs2 | x[rd] = x[rs1] * x[rs2] | Multiplication of two 32-bit floating point numbers | 1,2
mul.f64 | rd, rs1, rs2 | x[rd] = x[rs1] * x[rs2] | Multiplication of two 64-bit floating point numbers | 1,2
neg.f32 | rd, rs | x[rd] = -x[rs] | Negation of 32-bit floating point number | 1
neg.f64 | rd, rs | x[rd] = -x[rs] | Negation of 64-bit floating point number | 1
rem.f32 | rd, rs1, rs2 | x[rd] = x[rs1] % x[rs2] | Remainder of the division between two 32-bit floating point numbers | 1,2
rem.f64 | rd, rs1, rs2 | x[rd] = x[rs1] % x[rs2] | Remainder of the division between two 64-bit floating point numbers | 1,2
round.f32 | rd, rs, imm | x[rd] = round(x[rs]) | Rounds a 32-bit floating point number to the nearest integer; The flag inside imm specifies the rounding mode | 1
round.f64 | rd, rs, imm | x[rd] = round(x[rs]) | Rounds a 64-bit floating point number to the nearest integer; The flag inside imm specifies the rounding mode | 1
sub.f32 | rd, rs1, rs2 | x[rd] = x[rs1] - x[rs2] | Subtraction of two 32-bit floating point numbers | 1,2
sub.f64 | rd, rs1, rs2 | x[rd] = x[rs1] - x[rs2] | Subtraction of two 64-bit floating point numbers | 1,2

For the `round.f32` and `round.f64` instructions, the rounding mode is specified via the flag passed as an immediate value (`imm`) and has the following meaning:
- RNE (round to nearest): value `0x0`
- RDN (round down): value `0x1`
- RUP (round up): value `0x2`
- RTZ (round toward zero): value `0x3`

### Math - Floating-Point Numbers

There are dedicated instructions for 32 and 64-bit floating-point mathematical functions. The floating-point values are expected to be stored in the [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) format; in other words, the single and double-precision IEEE 754 formats are supported.

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
atan.f32 | rd, rs | x[rd] = atan(x[rs]) | | 1
atan.f64 | rd, rs | x[rd] = atan(x[rs]) | | 1
cos.f32 | rd, rs | x[rd] = cos(x[rs]) | | 1
cos.f64 | rd, rs | x[rd] = cos(x[rs]) | | 1
cosh.f32 | rd, rs | x[rd] = cosh(x[rs]) | | 1
cosh.f64 | rd, rs | x[rd] = cosh(x[rs]) | | 1
exp.f32 | rd, rs | x[rd] = exp(x[rs]) | | 1
exp.f64 | rd, rs | x[rd] = exp(x[rs]) | | 1
log.f32 | rd, rs, imm | x[rd] = log(x[rs], imm) | The base is specified via the imm flag | 1
log.f64 | rd, rs, imm | x[rd] = log(x[rs], imm) | The base is specified via the imm flag | 1
pow.f32 | rd, rs1, rs2 | x[rd] = pow(x[rs1], x[rs2]) | | 1,2
pow.f64 | rd, rs1, rs2 | x[rd] = pow(x[rs1], x[rs2]) | | 1,2
rsqrt.f32 | rd, rs | x[rd] = rsqrt(x[rs]) | | 1
rsqrt.f64 | rd, rs | x[rd] = rsqrt(x[rs]) | | 1
sin.f32 | rd, rs | x[rd] = sin(x[rs]) | | 1
sin.f64 | rd, rs | x[rd] = sin(x[rs]) | | 1
sinh.f32 | rd, rs | x[rd] = sinh(x[rs]) | | 1
sinh.f64 | rd, rs | x[rd] = sinh(x[rs]) | | 1
sqrt.f32 | rd, rs | x[rd] = sqrt(x[rs]) | | 1
sqrt.f64 | rd, rs | x[rd] = sqrt(x[rs]) | | 1
tanh.f32 | rd, rs | x[rd] = tanh(x[rs]) | | 1
tanh.f64 | rd, rs | x[rd] = tanh(x[rs]) | | 1
tan.f32 | rd, rs | x[rd] = tan(x[rs]) | | 1
tan.f64 | rd, rs | x[rd] = tan(x[rs]) | | 1

For the `log.f32` and `log.f64` instructions, the base is specified via the flag passed as an immediate value (`imm`) and has the following meaning:
- base e (natural logarithm): value `0x0`
- base 10: value `0x1`
- base 2: value `0x2`

### Bitwise

There are dedicated instructions for bitwise manipulation of 32 and 64-bit values.

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
and.32 | rd, rs1, rs2 | x[rd] = x[rs1] & x[rs2] | Bitwise AND between two 32-bit values | 1,2
and.64 | rd, rs1, rs2 | x[rd] = x[rs1] & x[rs2] | Bitwise AND between two 64-bit values | 1,2
not.32 | rd, rs | x[rd] = ~x[rs] | Bitwise negation of 32-bit value | 1
not.64 | rd, rs | x[rd] = ~x[rs] | Bitwise negation of 64-bit value | 1
or.32 | rd, rs1, rs2 | x[rd] = x[rs1] \| x[rs2] | Bitwise OR between two 32-bit values | 1,2
or.64 | rd, rs1, rs2 | x[rd] = x[rs1] \| x[rs2] | Bitwise OR between two 64-bit values | 1,2
xor.32 | rd, rs1, rs2 | x[rd] = x[rs1] ^ x[rs2] | Bitwise XOR between two 32-bit values | 1,2
xor.64 | rd, rs1, rs2 | x[rd] = x[rs1] ^ x[rs2] | Bitwise XOR between two 64-bit values | 1,2
sll.32 | rd, rs1, rs2 | x[rd] = x[rs1] << x[rs2] | Logical shift left of 32-bit value by the number of bits specified | 1,2
sll.64 | rd, rs1, rs2 | x[rd] = x[rs1] << x[rs2] | Logical shift left of 64-bit value by the number of bits specified | 1,2
srl.32 | rd, rs1, rs2 | x[rd] = x[rs1] >> x[rs2] | Logical shift right of 32-bit value by the number of bits specified | 1,2
srl.64 | rd, rs1, rs2 | x[rd] = x[rs1] >> x[rs2] | Logical shift right of 64-bit value by the number of bits specified | 1,2
sra.32 | rd, rs1, rs2 | x[rd] = x[rs1] >> x[rs2] | Arithmetic shift right of 32-bit value by the number of bits specified. Preserves the sign bit | 1,2
sra.64 | rd, rs1, rs2 | x[rd] = x[rs1] >> x[rs2] | Arithmetic shift right of 64-bit value by the number of bits specified. Preserves the sign bit | 1,2

### Comparison

There are dedicated instructions for comparing 32 / 64-bit integers or floating-point numbers. All integer values are expected to be stored in two's complement, which allows us to use the same instruction for signed and unsigned data. The instructions make use of a flag to denote whether to treat the values as signed or unsigned, as well as the type of comparison that should be done. The flag is passed as an immediate value (`imm`) which has the following binary representation:

15-9 | 8 | 7-0
-- | -- | --
reserved | SIGN | CMP

The meaning of the fields is the following:

- `CMP` specifies the comparison function:
    - EQ (equal): value `0x00`
    - NE (not equal): value `0x01`
    - GT (greater than): value `0x02`
    - GTE (greater than or equal): value `0x03`
    - LT (less than): value `0x04`
    - LTE (less than or equal): value `0x05`
- the `SIGN` bit determines whether to treat the operands as signed (value `1`) or unsigned (value `0`); this sign bit only has meaning for comparisons between integers

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
cmp.i32 | rd, rs1, rs2, imm | x[rd] = cmp(x[rs1], x[rs2]) | Comparison between two 32-bit integers | 1,2
cmp.i64 | rd, rs1, rs2, imm | x[rd] = cmp(x[rs1], x[rs2]) | Comparison between two 64-bit integers | 1,2
cmp.f32 | rd, rs1, rs2, imm | x[rd] = cmp(x[rs1], x[rs2]) | Comparison between two 32-bit floating-point numbers | 1,2
cmp.f64 | rd, rs1, rs2, imm | x[rd] = cmp(x[rs1], x[rs2]) | Comparison between two 64-bit floating-point numbers | 1,2

### Conversion

There are dedicated instructions for converting between all supported primitive types.

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
convert.i8toi16 | rd, rs | x[rd] = convert<int16_t>(x[rs]) | Converts 8-bit integer value to 16-bit integer value | 1
convert.i8toi32 | rd, rs | x[rd] = convert<int32_t>(x[rs]) | Converts 8-bit integer value to 32-bit integer value | 1
convert.i8toi64 | rd, rs | x[rd] = convert<int64_t>(x[rs]) | Converts 8-bit integer value to 64-bit integer value | 1
convert.i8tof32 | rd, rs | x[rd] = convert<float>(x[rs]) | Converts 8-bit integer value to 32-bit floating-point value | 1
convert.i8tof64 | rd, rs | x[rd] = convert<double>(x[rs]) | Converts 8-bit integer value to 64-bit floating-point value | 1
convert.i16toi8 | rd, rs | x[rd] = convert<int8_t>(x[rs]) | Converts 16-bit integer value to 8-bit integer value | 1
convert.i16toi32 | rd, rs | x[rd] = convert<int32_t>(x[rs]) | Converts 16-bit integer value to 32-bit integer value | 1
convert.i16toi64 | rd, rs | x[rd] = convert<int64_t>(x[rs]) | Converts 16-bit integer value to 64-bit integer value | 1
convert.i16tof32 | rd, rs | x[rd] = convert<float>(x[rs]) | Converts 16-bit integer value to 32-bit floating-point value | 1
convert.i16tof64 | rd, rs | x[rd] = convert<double>(x[rs]) | Converts 16-bit integer value to 64-bit floating-point value | 1
convert.i32toi8 | rd, rs | x[rd] = convert<int8_t>(x[rs]) | Converts 32-bit integer value to 8-bit integer value | 1
convert.i32toi16 | rd, rs | x[rd] = convert<int16_t>(x[rs]) | Converts 32-bit integer value to 16-bit integer value | 1
convert.i32toi64 | rd, rs | x[rd] = convert<int64_t>(x[rs]) | Converts 32-bit integer value to 64-bit integer value | 1
convert.i32tof32 | rd, rs | x[rd] = convert<float>(x[rs]) | Converts 32-bit integer value to 32-bit floating-point value | 1
convert.i32tof64 | rd, rs | x[rd] = convert<double>(x[rs]) | Converts 32-bit integer value to 64-bit floating-point value | 1
convert.i64toi8 | rd, rs | x[rd] = convert<int8_t>(x[rs]) | Converts 64-bit integer value to 8-bit integer value | 1
convert.i64toi16 | rd, rs | x[rd] = convert<int16_t>(x[rs]) | Converts 64-bit integer value to 16-bit integer value | 1
convert.i64toi32 | rd, rs | x[rd] = convert<int32_t>(x[rs]) | Converts 64-bit integer value to 32-bit integer value | 1
convert.i64tof32 | rd, rs | x[rd] = convert<float>(x[rs]) | Converts 64-bit integer value to 32-bit floating-point value | 1
convert.i64tof64 | rd, rs | x[rd] = convert<double>(x[rs]) | Converts 64-bit integer value to 64-bit floating-point value | 1
convert.f32toi8 | rd, rs | x[rd] = convert<int8_t>(x[rs]) | Converts 32-bit floating-point value to 8-bit integer value | 1
convert.f32toi16 | rd, rs | x[rd] = convert<int16_t>(x[rs]) | Converts 32-bit floating-point value to 16-bit integer value | 1
convert.f32toi32 | rd, rs | x[rd] = convert<int32_t>(x[rs]) | Converts 32-bit floating-point value to 32-bit integer value | 1
convert.f32toi64 | rd, rs | x[rd] = convert<int64_t>(x[rs]) | Converts 32-bit floating-point value to 64-bit integer value | 1
convert.f32tof64 | rd, rs | x[rd] = convert<double>(x[rs]) | Converts 32-bit floating-point value to 64-bit floating-point value | 1
convert.f64toi8 | rd, rs | x[rd] = convert<int8_t>(x[rs]) | Converts 64-bit floating-point value to 8-bit integer value | 1
convert.f64toi16 | rd, rs | x[rd] = convert<int16_t>(x[rs]) | Converts 64-bit floating-point value to 16-bit integer value | 1
convert.f64toi32 | rd, rs | x[rd] = convert<int32_t>(x[rs]) | Converts 64-bit floating-point value to 32-bit integer value | 1
convert.f64toi64 | rd, rs | x[rd] = convert<int64_t>(x[rs]) | Converts 64-bit floating-point value to 64-bit integer value | 1
convert.f64tof32 | rd, rs | x[rd] = convert<float>(x[rs]) | Converts 64-bit floating-point value to 32-bit floating-point value | 1

### Conditional assignment

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
select | rd, rs1, rs2, rs3 | x[rd] = x[rs1] ? x[rs2] : x[rs3] | Conditionally assigns the value of rs2 (true value) or rs3 (false value) registers into the rd register, based on the value of rs1. The condition is considered true if x[rs1] is non-zero | 1

### Control Flow

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
call | rs, N, rN..., M, rM... |  | Call the function whose identifier is passed via rs. The identifier represents the index of the function within the function section. `N` is an immediate value that represents the number of values returned by the function and is followed by `N` destination register operands; these registers will store the return values. `M` is an immediate value that represents the number of parameters passed to the function and is followed by `M` register operands; the values of these operands will be passed to the function as arguments | -
ret | | | Return to the caller function | -
retv | N, rN... | | Return to the caller function and return the values given as operands. `N` is an immediate value that represents the number of values returned by the function and is followed by `N` register operands, whose values will be passed to the destination registers used by the associated `call` instruction. The number of destination registers from the `call` instruction must be equal to the number of values returned by the `retv` instruction | -
jmp | rs | pc += x[rs] | Unconditionally jump to PC-relative offset found inside rs. The offset is added to the current `pc` value. The target address must fall within the current function's body range | 1,2
je | rs, rs1, rs2 | if (x[rs1] == x[rs2]) pc += x[rs] | Jump to PC-relative offset found in rs if the value inside rs1 is equal to the value inside rs2. The offset is added to the current `pc` value. The target address must fall within the current function's body range | 1,2
jne | rs, rs1, rs2 | if (x[rs1] != x[rs2]) pc += x[rs] | Jump to PC-relative offset found in rs if the value inside rs1 is not equal to the value inside rs2. The offset is added to the current `pc` value. The target address must fall within the current function's body range | 1,2
assert | rs1, rs2 | | Check that the value of register rs1 is non-zero. If it is zero, the VM traps and prints a message selected by the value of rs2, interpreted as an index into the string table. If the value of register rs1 is non-zero, no trap is taken and no message is printed | 1
nop | | | Do nothing | -

### Kernel Submission

The compute submission mechanism is modeled after the [oneAPI Level Zero](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html) specification. As a result, concepts such as kernels, events and command lists are expressed via dedicated instructions. Details about the kernel submission concepts can be found in the [VM kernel submission](#65-kernel-submission) chapter.

Instruction name | Arguments | Details | Addressing Modes
-- | -- | -- | --
kernel.create | rd, rs, N, rN..., M, rM... | Create a kernel handle and store it into the rd register. The kernel is identified by rs, which contains the index of the kernel inside the kernel section. `N` is an immediate value that represents the number of input buffers and is followed by the same number of registers that contain buffer handles for the inputs. Similarly, `M` is an immediate value that specifies the number of output buffers and is followed by the same number of registers that contain buffer handles for the outputs | -
kernel.delete | rs | Delete the kernel whose identifier is found in the rs register | 1
event.create | rd | Create an event and store its identifier into the rd register | 1
event.delete | rs | Delete the event whose identifier is found in the rs register | 1
event.wait | rs | Wait for the event from rs to be signalled | 1
event.signal | rs | Signal the event from rs | 1
cmd_list.create | rd | Create a command list and store its identifier in rd | 1
cmd_list.delete | rs | Delete the command list whose identifier is found in the rs register | 1
cmd_list.add_kernel | rs1, rs2, N, rN..., M, rM... | Add the kernel handle from rs2 to the command list from rs1. `N` is an immediate value that represents the number of signal events used by the kernel (at most one signal event is accepted) and is followed by the same number of registers that contain event identifiers; in case more than one signal event is passed, the VM traps. `M` is an immediate value that represents the number of wait events used by the kernel and is followed by the same number of registers that contain event identifiers | -
cmd_list.add_barrier | rs, N, rN..., M, rM... | Add the barrier to the command list from rs. `N` is an immediate value that represents the number of signal events used by the barrier (at most one signal event is accepted) and is followed by the same number of registers that contain event identifiers; in case more than one signal event is passed, the VM traps. `M` is an immediate value that represents the number of wait events used by the barrier and is followed by the same number of registers that contain event identifiers | -
cmd_list.reset | rs | Reset the command list from rs to the initial (empty) state. Makes it ready for appending commands | 1
cmd_list.close | rs | Closes the command list from rs. Makes it ready to be executed by a command queue | 1
cmd_list.exec | rs | Execute the command list from rs. This instruction is not waiting for the execution to finish | 1

### Buffers

Buffers are a common primitive used for representing and manipulating large chunks of memory. A buffer's metadata is comprised of the following:
- the rank of the buffer
- the shape of the buffer, which must be static during execution (i.e. not dynamic)
- the strides of the buffer, which must be static during execution (i.e. not dynamic)
- the element type of the buffer

The buffer also contains information about the underlying data, which is not exposed to instructions:
- the memory allocation of the data (i.e. its address and size)
- whether the data is owned by the buffer or not
- the read / write permissions over the data

Buffers are identified by unique handles, which represent opaque 64-bit integers that are managed by the VM. These handles can be obtained upon the creation of a buffer, for example when using `buffer.create`. In this case, the buffer owns the underlying memory and can be deleted (e.g. using `buffer.delete`). Handles can also be passed by the VM to the entrypoint function, via parameter registers which correspond to the function arguments. In this case, the buffers are created internally by the VM and it references external memory which cannot be deleted; the permission of these buffers can also be limited by the VM upon their creation. More information about the way buffers are handled by the VM can be found in the [Memory Sandboxing](#62-memory-sandboxing) chapter.

Instruction name | Arguments | Details | Addressing Modes
-- | -- | -- | --
buffer.create | rd, r0, N, rN..., rN... | Create a buffer and store its handle in the `rd` register. The element type is specified via the `r0` register and corresponds to the index of the type inside the type section. The rank of the buffer is specified via the `N` immediate value, which is followed by `N` operands that correspond to the shape of the buffer and `N` operands that correspond to the strides of the buffer. The resulting buffer owns the underlying memory allocation and has read+write permission | -
buffer.from_const | rd, r0, r1, N, rN..., rN... | Create a buffer using the constant whose index is found in r0 and store the resulting buffer's handle in the `rd` register. The element type is specified via the `r1` register and corresponds to the index of the type inside the type section. The rank of the buffer is specified using the `N` immediate value, which is followed by `N` operands that correspond to the shape of the buffer and `N` operands that correspond to the strides of the buffer. The resulting buffer does not own the underlying memory, and only has read permission | -
buffer.delete | rs | Delete a buffer whose handle is found in the rs register. The buffer must own the underlying memory, in order for it to be deleted. If the memory is unowned, the VM will trap. All derived buffers (e.g. those obtained from `buffer.subview` or `buffer.view`) will also have their handles invalidated | 1
buffer.get_rank | rd, rs | Get the rank of the buffer whose handle is found in the rs register, and store it into rd | 1
buffer.get_dim | rd, rs1, rs2 | Extracts the requested dimension from the shape of the buffer. rs1 contains the handle of the target buffer, rs2 contains the targeted dimension | 1,2
buffer.get_stride | rd, rs1, rs2 | Extracts the requested dimension from the stride of the buffer. rs1 contains the handle of the target buffer, rs2 contains the targeted dimension | 1,2
buffer.get_elem_type | rd, rs | Get the element type of the buffer whose handle is found in the rs register, and store it into rd. The element type is identified by its index within the type section | 1
buffer.set_dim | rs, rs1, rs2 | Set the rs2 shape dimension of the buffer whose handle is found in the rs register, to the value from rs1. The VM validates that the new shape does not exceed the buffer's memory range | 1,2
buffer.set_stride | rs, rs1, rs2 | Set the rs2 stride dimension of the buffer whose handle is found in the rs register, to the value from rs1. The VM validates that the new stride does not exceed the buffer's memory range | 1,2
buffer.set_elem_type | rs1, rs2 | Set the element type of the buffer whose handle is found in the rs1 register, to the value from rs2. The element type is identified by its index within the type section | 1
buffer.subview | rd, rs, N, rN..., rN..., rN... | Create a subview of the buffer whose handle is found in the rs register, by applying the subview metadata specified by the rest of the operands, and store the resulting buffer's handle in the rd register. `N` is an immediate value that represents the rank of the buffer, and is followed by `3 x N` values which represent the offsets, sizes and strides of the subview. The resulting buffer shares the memory with the original buffer. The VM validates that the computed view of the buffer does not exceed the buffer's memory range. The lifecycle of this buffer is tied to the parent buffer; once the parent buffer is deleted, the handle of this derived buffer will be invalidated | -
buffer.view | rd, rs, r0, N, rN..., rN... | Create a buffer with a new element type, shape and strides for the buffer whose handle is found in the rs register, and store the resulting buffer's handle in rd. r0 specifies the index of the new element type within the type section. `N` is an immediate value that represents the new rank, followed by `2 x N` registers: the first `N` values represent the new shape dimensions, and the next `N` values represent the new strides. The underlying memory is shared with the source buffer. The VM validates that the computed view of the buffer does not exceed the buffer's memory range. The lifecycle of this buffer is tied to the parent buffer; once the parent buffer is deleted, the handle of this derived buffer will be invalidated | -
buffer.clone | rd, rs | Create a clone of the buffer whose handle is found in the rs register and store the new buffer's handle into rd. The resulting buffer owns underlying memory and has read+write access over it | 1
buffer.copy | rs1, rs2 | Copy the contents of the buffer whose handle is found in the rs2 register into the buffer whose handle is found in the rs1 register. The two buffers should have the same element type, shape and strides | 1
buffer.load | rd, rs, N, rN... | Loads an element from the buffer whose handle is found in the rs register and store it into rd. `N` is an immediate value that represents the rank of the buffer, and is followed by `N` values which represent the indices where the value is found. The VM validates that the computed byte offset falls within the buffer's memory range | -
buffer.store | rs1, rs2, N, rN... | Stores the value from rs2 into the buffer whose handle is found in the rs1 register. `N` is an immediate value that represents the rank of the buffer, and is followed by `N` values which represent the indices where the value should be written inside the buffer. The VM validates that the computed byte offset falls within the buffer's memory range and that the memory is writable | -
buffer.fill | rs1, rs2, imm | Fill the buffer whose handle is found in the rs1 register with the value from rs2. imm bytes from rs2 will be used. The size of the buffer must be a multiple of the byte-size of the value. The VM validates that the buffer's memory is writable | 1,2
buffer.concat | rd, rs1, rs2, rs3 | Concatenate two buffers whose handles are found in the rs1 and rs2 registers along the axis provided by rs3, and store the resulting buffer's handle in rd. The two source buffers must have compatible shapes for concatenation: the element types must be identical and all dimensions except the concatenation axis must match. The resulting buffer owns underlying memory and has read+write access over it | 1

#### Working with Buffers

##### Creation

```mlir
// Create a buffer using the `buffer.create` instruction. It receives the following operands:
// - the destination register, where the handle of the new buffer will be stored (here `rd`)
// - the element type of the buffer, passed via a register that contains the index of the type inside the type section (here `r0`)
// - the rank of the buffer, passed as an immediate value (here `3`)
// - `rank` operands that represent the shape of the buffer (here `r1`, `r2`, `r3`)
// - `rank` operands that represent the strides of the buffer (here `r4`, `r5`, `r6`)
// A buffer will be allocated by this instruction, and its handle will be stored in the destination register (here `rd`).
buffer.create rd, r0, 3, r1, r2, r3, r4, r5, r6
```

##### SubView

```mlir
// The `buffer.subview` instruction takes the following arguments:
// - the destination register, where the handle of the new buffer will be stored (here `rd`)
// - the source register, which contains the handle of the original buffer (here `rs`)
// - the rank of the buffer, passed as an immediate value (here `3`)
// - a variable number of operands, containing the offsets, sizes and strides that describe the subview;
//   there are `3 x rank` operands expected; for this example, rank three is used with the following subview description:
//     offsets: [r1, r2, r3]
//     sizes:   [r4, r5, r6]
//     strides: [r7, r8, r9]
buffer.subview rd, rs, 3, r1, r2, r3, r4, r5, r6, r7, r8, r9
```

### Set

Instruction name | Arguments | Implementation | Details | Addressing Modes
-- | -- | -- | -- | --
set | rd, rs | x[rd] = x[rs] | Set the value from the rs register into the rd register | 1
set.imm | rd, imm | x[rd] = imm | Set a 64 bit immediate value into rd. Note: this instructions has a 64 bit operand, compared to the other instructions which have 16 bit operands | -

## 5. Bytecode Dialect

To simplify the representation, implementation and the compatibility testing, a Bytecode dialect is introduced. This dialect is intended to represent a 1-to-1 mapping with the bytecode format described above. This means that every opcode has an equivalent operation inside the dialect, and that every section is represented in the IR, which makes the serialization straight-forward.

### 5.1. Operations

There are two main types of operations represented in the dialect:

1. Section operations, which are meant to be containers for other operations. For example, the constant section contains a list of constant operations, each identified by its index within the section.
2. Instruction operations, which are meant to represent the instruction set of the bytecode format. Each operation is based on the specification of the instruction and contains the unique opcode, as well as the operands. The operand types depend on the available addressing modes of each instruction; for example, instructions that support immediate addressing can have primitive types as operands, such as integers, while instructions that support register addressing have register types as operands.

Every serializable operation and type contains a field which specifies the version in which it was introduced. This field is used to check whether the operation is compatible with the target version for the compilation. If it is not, the operation must either be converted to functionally-equivalent operations that are compatible, or the compilation must fail with a clear message.

### 5.2. Types

#### Register Type

A register type is introduced to represent the registers used by the instructions. This type is an alias to a 16-bit signed integer, which is used to represent the register number. It is used to represent both global and general registers.

## 6. Virtual Machine

The Virtual Machine (VM) is able to parse the bytecode format and interpret its instructions. An implementation is expected to perform the following steps during execution:

1. Parse the file header, to ensure compatibility with the file version and identify the sections using the section header table.
2. Set the `platform` register to the NPU platform used for the execution.
3. Identify the entrypoint function from the function section, parse the function type associated with it, and initialize its call frame.
4. Allocate the general and parameter registers used by the function. Create external buffers for the input and output data, and set their handles into the frame's parameter registers. The permissions for each external buffer are configured based on the function signature (e.g. read-only for inputs, read-write for outputs).
5. Add the "exit" return address to the call frame. As this is the entrypoint function, the return address that is set has a special meaning to stop execution. The VM implementation can decide what this special return address should be (e.g. the return address could be an optional which has no value).
6. Set the `pc` register to the starting address of the entrypoint function's body.
7. Begin the fetch-decode-execute cycle.

As the bytecode format stores its content in little-endian format, the VM is expected to be executed on a little-endian host. This allows data to be directly interpreted as values, without byte reordering. The VM implementation is expected to check for the byte ordering of the host machine and stop execution if it is not little-endian.

### 6.1. Registers

The VM uses 64-bit registers for all instructions. There are two types of registers used:

#### Global Registers

These registers are used across the entire execution. These are the following:

- `pc` program counter: stores the address of the instruction currently executing; this register is not exposed to instructions and cannot be manually modified; only dedicated instructions (such as `jmp`), can modify its state, via a relative offset
- `platform`: specifies the NPU platform used for the inference;  its register number is `-1`

The `platform` register is intended to provide information about the NPU platform, which can influence the behavior of the bytecode execution; for example, in case the file contains per-platform kernels, the appropriate kernel can be dispatched based on the value of this register.

#### General Registers

Beside the global registers, a variable-number of general registers are used during execution. They are identified as `r[0-N]` in this document (e.g. `r0`, `r1`).

The general registers are function-specific, such that every function has its own set of registers. This was chosen as it simplifies the design due to the following:
- it removes the risk of a callee function manipulating the state of a caller function
- in case the compilation optimizes the register utilization via register allocation, such that the number of registers utilized is reduced, there is no need to save the register state when calling a function
- registers could be created and destroyed by the VM implementation, as function calls occur, thus reducing the memory utilized during execution

It is also not necessary to have registers shared across function calls, as the chosen [calling convention](#63-calling-convention) transfers data between calls via the call frames that have their own set of registers.

Every function specifies the number of general registers it utilizes.

### 6.2. Memory Sandboxing

The VM enforces memory safety through buffers. All memory accessed during execution is managed via buffer handles, which encapsulate the underlying memory allocation along with metadata such as ownership and permissions. The VM validates every memory access by checking the buffer's bounds and permissions, ensuring that no out-of-bounds or unauthorized access can occur.

#### Buffer Memory Model

Each buffer managed by the VM tracks the following internal state:

Field | Type | Description
-- | -- | --
host_ptr | void* | The host memory pointer for the buffer's data (internal to the VM, never exposed to bytecode)
size | uint64_t | The total size of the buffer's memory allocation in bytes
ownership | enum | `owned` or `unowned`
permissions | enum | `R` (read-only) or `RW` (read-write)

These fields are maintained internally by the VM and are not directly accessible to bytecode instructions. Instructions interact with buffers exclusively through handles stored in registers.

Buffers are categorized based on their origin:

Category | Ownership | Permissions | Notes
-- | -- | -- | --
Managed buffers | `owned` | `RW` | Buffers allocated via `buffer.create`, `buffer.clone` or `buffer.concat`. The VM allocates the underlying memory. These buffers can be freed via `buffer.delete`.
External buffers | `unowned` | `R` / `RW` | Buffers passed to the entrypoint function as arguments. These reference memory external to the VM (e.g. model input/output data provided by the NPU plugin). The permissions are configured by the VM upon creation based on the function signature (e.g. read-only for inputs, read-write for outputs). These buffers cannot be deleted by bytecode instructions.
Derived buffers | `unowned` | Inherited | Buffers created via `buffer.subview` or `buffer.view`. These share the underlying memory with the source buffer and inherit the source buffer's permissions. Deletion is not permitted since the memory is not owned.
Meta buffers | `unowned` | `R` | Buffers that reference data inside the bytecode file sections (e.g. large constants from the constant section). These are read-only and cannot be deleted.

#### Memory Access Validation

When a buffer-accessing instruction is executed (e.g. `buffer.load`, `buffer.store`, `buffer.fill`), the VM performs the following validation steps:

1. **Handle validation**: Verify that the buffer handle refers to a valid, live buffer. If the handle is invalid or references a buffer that has been deleted, the VM traps with `"invalid buffer handle"`.
2. **Bounds check**: Verify that the entire access range falls within the buffer's memory allocation. The access offset and size are computed from the buffer's shape, strides and element type. If the access would exceed the buffer's bounds, the VM traps with `"out-of-bounds buffer access"`.
3. **Permission check**: Verify that the buffer's permissions allow the requested operation. Write operations (e.g. `buffer.store`, `buffer.fill`) require `RW` permission. If the permission is insufficient, the VM traps with `"buffer access permission denied"`.

#### Ownership Validation

The VM also validates ownership when a buffer deletion is requested:

- `buffer.delete` requires that the target buffer has `owned` ownership. Attempting to delete an unowned buffer (external, derived, or meta) causes the VM to trap with `"cannot delete unowned buffer"`.

When a buffer is deleted, all of its derived buffers are also deleted. This includes buffers that are transitively derived (e.g. a subview of a subview).

#### Memory Allocation Limits

To prevent exhaustion of host memory, the VM enforces a configurable maximum total allocation size for owned buffers. When instructions such as `buffer.create`, `buffer.clone` or `buffer.concat` are executed, the VM checks whether the new allocation would exceed this limit. If so, the VM traps with `"memory allocation limit exceeded"`.

### 6.3. Calling Convention

The calling convention used by the bytecode format makes use of the concept of a **call frame**. Each function invocation has its own call frame, which contains the following:
- general registers, whose number is specified in the section header for each function
- parameter registers, determined by the number of parameters passed to the `call` instruction
- the return address, which contains the address of the next instruction after the `call` instruction; it is inaccessible to instructions

When a function is called, a call frame is created. Internally, the return address is set and enough registers are allocated to store both the general and parameter registers needed by the function. These registers are zero-initialized, to prevent leaking information between call frames (e.g. if the VM implementation reuses call frame allocations). The registers are identified by unique numbers. If a function has `G` general registers and `P` parameter registers, the call frame will contain `G+P` registers, each identified by the following numbers:

```
                     general registers         parameter registers
                   |                    |  |                        |
Register numbers: [0, 1, 2, ..., G-2, G-1, G+0, G+1, G+2, ... , G+P-1]

Example:
- 10 general registers
- 5 parameter registers
                   general registers   parameter registers
                   |                |  |                 |
Register numbers: [0, 1, 2, ..., 8, 9, 10, 11, 12, ..., 14]
```

The order of the parameter registers corresponds with the order of parameters passed to the `call` instructions. In other words, the first parameter will correspond with the register number `G+0`, the last parameter will correspond with the register number `G+P-1`.

Beside parameters, the `call` instruction can also receive zero or more destination registers, in which the return values will be stored. When the `retv` instruction is called, the VM will set the return values to the specified destination registers. The order of the destination register operands for the `call` instruction corresponds with the order of operands for the `retv` instruction.

Before executing the entrypoint function, the VM performs the following steps:
1. Initialize a call frame for the entrypoint function, where the requested number of general and parameter registers are allocated.
2. Set the "exit" return address in the call frame.
3. Set the values inside the parameter registers to the entrypoint function's arguments. If these arguments are input and output buffers, the VM will create buffer objects that point to these external addresses, and place the buffer handles in the parameter registers.

The following is an example which shows the calling convention in practice, starting with the entrypoint function which internally calls another function.

```mlir
function_section: [
    // The entrypoint function's call frame is created with the following configuration:
    //   - general registers:   r0, r1 (register numbers 0, 1)
    //   - parameter registers: rp0    (register number 2)
    //   - return address: END
    0: @main, num_args=1, num_results=0, {
        // ---
        // Function body...
        // ---

        // Create a buffer to be used by the inner function.
        // `rp0` holds the element type index, and the remaining operands (that are not shown) describe the rank, shape and strides.
        buffer.create r0, rp0, ...

        // The `call` instruction creates a new call frame and jumps to the target function.
        // It receives one destination register which will store the result value (`r1`), and one parameter registers (`r0`).
        call @inner_fn, 1, r1, 1, r0

        // `r1` now contains the result value from `@inner_fn` (i.e. the value 5)

        // ---
        // Function body...
        // ---

        // The `ret` instruction jumps to the return address, returning control to the caller function.
        // As there is no caller function and the return address has a special meaning, the execution will stop.
        ret
    }

    // The inner function's call frame is created with the following configuration:
    //   - general registers:   r0  (register number 0)
    //   - parameter registers: rp0 (register numbers 1)
    //   - return address: address of next instruction after `call` from @main
    1: @inner_fn, num_args=1, num_results=1, {
        // ---
        // Function body which uses rp0...
        // ---

        set.imm r0, 5  // Prepare the value that will be returned by the function; in this example, return the value 5

        // The `retv` instruction jumps to the return address, returning control to the caller function.
        // The variadic operands specify the returned values. In this case, there is a single returned value contained inside register `r0`
        retv 1, r0
    }
]
```

#### Call Frame Limits

To prevent memory overflow in case of infinite recursions, a limit has been set to the maximum number of call frames that can exist. This limit has been set to 1000. This means that there can be at most 1000 nested function calls at a given time. Additionally, instructions use 16 bits to address register numbers, which is interpreted as a signed integer where the positive numbers correspond to general registers; this means that there is a total of 2^15 addressable registers. As a call frame can also contain parameter registers alongside the general registers, this addressable register space is shared between the two register types. As a result, the total number of registers allocated per call frame (general + parameter) is 2^15.

This results in the following theoretical maximum memory utilization for call frames:

```
max_frames     = 1000
max_regs       = 2^15
reg_size_bytes = 8
Maximum memory utilization for call frames = max_frames * max_regs * reg_size_bytes = 250MB
```

### 6.4. Weights Separation Flow

In the weights separation use-case, the original model's constants are passed as inputs during execution. These constants are transformed once for NPU execution, then the inferences make use of these transformed constants. To achieve this, there is an initialization phase that should be done before the main execution.

The initialization phase consists of one or more init functions which receive the following buffers as arguments:
- an input buffer, which contains all of the original constants
- an output buffer, which will be populated by the transformed constants computed by the init function

For every constant, the init function extracts the relevant part of the input buffer, transforms the content, then places the result in the correct position of the output buffer. The output buffer is allocated by the NPU plugin after querying the necessary output buffer size for the init function.

After the initialization phase, the output buffers of the init functions are passed as inputs to the main function, which internally extracts the relevant part of the buffers for every constant.

```mlir
functions_section: [
    // This example contains two init functions, whose goal is to transform the constants from the original model, such that they can be used by the main function.
    // init0 will process two constants, while init1 will process three constants.
    // These functions will only be called once, after which main can be called any number of times to execute the actual inferences.
    0: @init0, num_args=2, num_results=0, {
        // Register rp0 contains the original constants
        // Register rp1 should be populated with the transformed constants by this init function

        // Option 1. Call NPU kernel associated with the first init function, which receives the same input / output buffers
        kernel.create ...

        // Option 2. Manually process each constant via dedicated NPU kernels / CPU transformations
        add.i64 r0, rp0, 0     // First constant is found at offset 0 and has 1024 bytes
        add.i64 r1, rp0, 1024  // Second constant is found at offset 1024
        add.i64 r2, rp1, 0     // First transformed constant should be placed at offset 0 and should have 2048 bytes
        add.i64 r3, rp1, 2048  // Second constant should be placed at offset 2048
        // Process constants inside r0, r1 and store the results in r2, r3

        ret
    }
    1: @init1, num_args=2, num_results=0, {
    }

    // This example assumes that the original model has one input value and one output value. The main function receives the two buffers as arguments,
    // while also receiving the two buffers with transformed constants, that are populated by the init functions.
    2: @main, num_args=4, num_results=0, {
    }
]
```

### 6.5. Kernel Submission

The compute submission mechanism is modeled after the [oneAPI Level Zero](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html) specification, which makes use of concepts such as kernels, command lists, command queues, events and barriers:

- **Kernels** are units of compute that are executed on the NPU. They are composed of an ELF blob binary, as well as the input and output buffers used for the execution. The ELF blob binaries are found in the kernel section of the bytecode format.
- **Command lists** represent a sequence of commands to execute. They are populated with kernels for performing computations, or by barriers for synchronization.
- **Command queues** are used for submitting command lists for execution. They allow the programming stage (i.e. populating the command list) to be done separately from the execution stage (i.e. populating the command queue with command lists).
- **Events** are synchronization primitives which can be used by kernels to signal completion, or to delay execution until a signal is received.
- **Barriers** are synchronization primitives that are submitted to the command lists, similar to kernels. They ensure that execution waits until the previous commands in the list are completed.

The bytecode format is able to create and manage kernels, events, barriers and command lists via dedicated instructions. This gives the code flexibility in generating computation workloads, defining synchronization points between them, submitting work for execution, and waiting for execution to complete. The command queues however are not visible to the bytecode. Instead, an external component (in this case, the NPU plugin) provides the VM a single command queue that can be used for execution. When work is submitted for execution (e.g. via `cmd_list.exec`), the VM will internally make use of the command queue. The command queue must be owned and managed by an external component from the VM, in order to support multiple inferences at the same time, as this external queue could be populated by multiple sources of inferences (e.g. multiple VM instances).

## Opens

> Open: Perhaps in the future we'll need to add support for features dedicated to debugging or profiling the bytecode. These should not fail execution on VMs that do not support such features. Would it make sense to represent optional features that VMs can skip executing?

> Open: Jump instructions could be crafted such that they fall in the middle of an instruction. Is there a way to ensure that the jumps fall at instruction boundaries, considering that there are instances of variadic operands?

> Open: Every operation should mention how they handle cases of overflow, division by zero etc. Do we need a status flag register, which records whether a previous operation had an overflow etc?

> Open: Clarify how the existing contract between the plugin and compiler looks like for the current weights separation solution. This will affect how the init functions get called (e.g. do we need multiple entrypoints, such as for calling the init functions separately for pipelining, or should we try to pipeline the init functions in the bytecode while calling init from main only for the first inference), and how information is passed between the init and main functions.
