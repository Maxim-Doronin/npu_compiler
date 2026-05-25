//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <iterator>

using namespace vpux;

int16_t bytecode::getRegisterNumber(mlir::Value operand) {
    // This should be extended to cover more complex cases (e.g. general register is not a direct parent)
    if (auto regOp = operand.getDefiningOp<bytecode::GeneralRegisterOp>()) {
        return static_cast<int16_t>(regOp.getRegNum());
    }
    VPUX_THROW("Could not find defining GeneralRegisterOp for operand {0}", operand);
}

llvm::StringMap<uint64_t> bytecode::buildTypeIndexMap(bytecode::TypeSectionOp typeSection) {
    llvm::StringMap<uint64_t> map;
    uint64_t index = 0;
    for (auto typeOp : typeSection.getContent().getOps<bytecode::TypeOp>()) {
        auto [it, inserted] = map.try_emplace(typeOp.getSymName(), index);
        VPUX_THROW_UNLESS(inserted, "Duplicate type symbol name '{0}' found in the type section", typeOp.getSymName());
        ++index;
    }
    return map;
}

int16_t bytecode::getStringIndex(StringRef symName, mlir::ModuleOp moduleOp) {
    auto stringSectionOps = moduleOp.getOps<bytecode::StringSectionOp>();
    auto numStringSections = std::distance(stringSectionOps.begin(), stringSectionOps.end());
    VPUX_THROW_UNLESS(numStringSections == 1, "Expected exactly one StringSectionOp in the module, but found {0}",
                      numStringSections);

    auto stringSection = *stringSectionOps.begin();
    auto stringOps = stringSection.getContent().getOps<bytecode::StringOp>();
    auto stringOpIt = llvm::find_if(stringOps, [&](bytecode::StringOp stringOp) {
        return stringOp.getSymName() == symName;
    });
    VPUX_THROW_UNLESS(stringOpIt != stringOps.end(), "Could not find string with symbol name {0} in the string section",
                      symName);
    return static_cast<int16_t>(std::distance(stringOps.begin(), stringOpIt));
}

bytecode::FloatFormat bytecode::getFloatFormat(mlir::FloatType floatType) {
    if (floatType.isBF16()) {
        return bytecode::FloatFormat::BFloat;
    }
    if (mlir::isa<mlir::FloatTF32Type>(floatType)) {
        return bytecode::FloatFormat::TFloat;
    }
    if (mlir::isa<mlir::Float8E4M3Type, mlir::Float8E4M3FNType, mlir::Float8E4M3FNUZType, mlir::Float8E4M3B11FNUZType>(
                floatType)) {
        return bytecode::FloatFormat::E4M3;
    }
    if (mlir::isa<mlir::Float8E5M2Type, mlir::Float8E5M2FNUZType>(floatType)) {
        return bytecode::FloatFormat::E5M2;
    }
    if (mlir::isa<mlir::Float4E2M1FNType>(floatType)) {
        return bytecode::FloatFormat::E2M1;
    }
    return bytecode::FloatFormat::IEEE;
}
