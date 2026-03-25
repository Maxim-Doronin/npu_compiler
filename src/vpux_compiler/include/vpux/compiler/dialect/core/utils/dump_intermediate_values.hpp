//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/YAMLTraits.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <optional>
#include <string>
#include <vector>

namespace vpux {

struct OpFilter {
    std::string name;
    std::vector<std::string> locations;
};

struct DumpIntermediateValuesConfig {
    std::string passName;
    std::vector<OpFilter> filters;
};

}  // namespace vpux

LLVM_YAML_IS_SEQUENCE_VECTOR(vpux::OpFilter)

namespace llvm ::yaml {

template <>
struct MappingTraits<vpux::OpFilter> {
    static void mapping(IO& io, vpux::OpFilter& info) {
        io.mapRequired("name", info.name);
        io.mapRequired("locations", info.locations);
    }
};

template <>
struct MappingTraits<vpux::DumpIntermediateValuesConfig> {
    static void mapping(IO& io, vpux::DumpIntermediateValuesConfig& info) {
        io.mapRequired("pass", info.passName);
        io.mapRequired("op_filters", info.filters);
    }
};

}  // namespace llvm::yaml

namespace vpux {

mlir::FailureOr<DumpIntermediateValuesConfig> parseYaml(StringRef fileName);
void addIntermediateValueDumper(mlir::PassManager& pm, mlir::StringRef configFilePath, const Logger& log);
mlir::LogicalResult dumpIntermediateValues(mlir::ModuleOp moduleOp, ArrayRef<OpFilter> filters, const Logger& log);

}  // namespace vpux
