//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_BYTECODE_OPS_BASE
#define VPUX_COMPILER_DIALECT_BYTECODE_OPS_BASE

include "vpux/compiler/dialect/bytecode/dialect.td"
include "vpux/compiler/dialect/bytecode/ops_interfaces.td"
include "vpux/compiler/dialect/bytecode/types.td"
include "mlir/IR/OpBase.td"

class Bytecode_Op<string mnemonic, list<Trait> traits = []> :
        Op<Bytecode_Dialect, mnemonic, traits> {
}

class Bytecode_SerializableOp<string mnemonic, list<Trait> traits = [], string minVersion = ""> :
        Bytecode_Op<mnemonic, [
                DeclareOpInterfaceMethods<Bytecode_VersionedOpInterface>,
                DeclareOpInterfaceMethods<Bytecode_SerializableOpInterface>
            ] # traits> {
    code serializableExtraClassDefinition = [{
        vpux::bytecode::Version $cppClass::getMinVersion() {
            return vpux::bytecode::Version(}] # !subst(".", ", ", minVersion) # [{);
        }
    }];
    let extraClassDefinition = serializableExtraClassDefinition;
}

class Bytecode_InstructionOp<string mnemonic, list<Trait> traits = [], string minVersion = "", string opcode = ""> :
        Bytecode_SerializableOp<mnemonic, [
                DeclareOpInterfaceMethods<Bytecode_InstructionOpInterface>
            ] # traits,
            minVersion> {
    let extraClassDefinition = [{
        vpux::bytecode::OpCode $cppClass::getOpcode() {
            return }] # opcode # [{;
        }
        size_t $cppClass::getBinarySize() {
            return vpux::bytecode::INSTRUCTION_SIZES.at(getOpcode());
        }
    }] # serializableExtraClassDefinition;
}

class Bytecode_ZeroOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription;

    let assemblyFormat = [{
        attr-dict
    }];
}

class Bytecode_OneOperandInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `reg` is a register
    }];

    let arguments = (ins
        Bytecode_RegisterType:$reg
    );

    let assemblyFormat = [{
        $reg attr-dict
    }];
}


class Bytecode_TwoOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `dst` and `src` are registers
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$src
    );

    let assemblyFormat = [{
        $dst `,` $src attr-dict
    }];
}

class Bytecode_ThreeOperandsInstructionBase<string mnemonic, list<Trait> traits = [], string minVersion, string opcode>
        : Bytecode_InstructionOp<mnemonic, traits, minVersion, opcode> {
    string opDescription = "";
    let description = opDescription # [{
        The operation supports one addressing mode:
        1. Register mode: `lhs` and `rhs` are registers
    }];

    let arguments = (ins
        Bytecode_RegisterType:$dst,
        Bytecode_RegisterType:$lhs,
        Bytecode_RegisterType:$rhs
    );

    let assemblyFormat = [{
        $dst `,` $lhs `,` $rhs attr-dict
    }];
}

#endif
