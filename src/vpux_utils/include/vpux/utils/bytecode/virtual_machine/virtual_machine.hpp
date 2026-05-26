//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "vpux/utils/bytecode/instructions.hpp"

namespace vpux::bytecode {

// Represents the execution lifecycle of the VirtualMachine.
// Transitions: Initialized -> Running -> Finalized (normal) or Halted (error).
enum class State : uint8_t { Initialized, Running, Halted, Finalized };

class Function {
    std::string _name;              // Human-readable function name
    uint64_t _numGeneralRegisters;  // Number of general-purpose registers used by the function
    bool _isEntrypoint;             // True if this function is the bytecode entry point
    std::vector<uint8_t> _body;     // Raw instruction bytes representing the function body

public:
    Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint, std::vector<uint8_t> body);

    std::string getName() const;
    uint64_t getNumGeneralRegisters() const;
    bool isEntrypoint() const;
    const std::vector<uint8_t>& getBody() const;
};

class CallFrame {
    std::vector<int64_t> _registers;  // The register set for the current function call, indexed by register number
    const uint8_t* _returnAddress;    // The return address to jump to after this function call completes

public:
    CallFrame(uint64_t numRegisters, const uint8_t* returnAddress);

    // Returns a mutable reference to the register at the given index.
    // Throws std::out_of_range if the index exceeds the register count.
    int64_t& getReg(int16_t index);

    // Sets the register at the given index to the specified value.
    // Throws std::out_of_range if the index exceeds the register count.
    void setReg(int16_t index, int64_t value);

    const uint8_t* getReturnAddress() const;
};

class VirtualMachine {
    std::vector<Function> _functions;  // All functions loaded from the bytecode function section(s)
    State _state{State::Initialized};  // Current execution state of the VM
    const uint8_t* _pc{
            nullptr};  // Program counter that points to the executing instruction within the active function body

    // Advances the program counter by the given opcode's instruction size
    // No-op if the VM is not in the Running state
    void incrementPC(OpCode opcode);

    // Interprets the instruction stream of the given function until a RET instruction is encountered or an unknown
    // opcode halts execution
    void execute(const Function& function);

public:
    // Deserializes a bytecode binary, extracts all functions from the function section(s), and stores them for later
    // execution. Returns false and prints an error if parsing fails.
    bool parse(const std::vector<uint8_t>& bytecode);

    /// Print the bytecode binary in a human-readable format
    /// @param bytecode The raw bytecode to print
    /// @param printFull If true, also prints the content of binary sections such constants or kernels
    /// @param indentLevel The indentation level for pretty-printing nested structures
    /// @return true if printing succeeded, false if parsing the bytecode failed
    bool print(const std::vector<uint8_t>& bytecode, bool printFull = true, size_t indentLevel = 0) const;

    // Locates the entry-point function and begins execution.
    // Prints an error and returns immediately if no entry point is found.
    void run();
};

}  // namespace vpux::bytecode
