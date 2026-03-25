//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/utils/dump_intermediate_values.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/YAMLTraits.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <memory>
#include <optional>
#include <vector>

namespace vpux::Core {
#define GEN_PASS_DECL_DUMPINTERMEDIATEVALUES
#define GEN_PASS_DEF_DUMPINTERMEDIATEVALUES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

class DumpIntermediateValuesPass final : public Core::impl::DumpIntermediateValuesBase<DumpIntermediateValuesPass> {
    std::vector<OpFilter> _filters;

    void safeRunOnModule() final {
        if (mlir::failed(dumpIntermediateValues(getOperation(), _filters, _log))) {
            signalPassFailure();
        }
    }

public:
    explicit DumpIntermediateValuesPass(ArrayRef<OpFilter> filters, const Logger& log): _filters(filters) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) override {
        if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
            return mlir::failure();
        }
        llvm::yaml::Input input(opFilters);
        _filters.clear();
        input >> _filters;
        if (input.error()) {
            return errorHandler("Failed to parse op-filters option");
        }

        return mlir::success();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createDumpIntermediateValuesPass(const Logger& log) {
    return std::make_unique<DumpIntermediateValuesPass>(std::vector<OpFilter>{}, log);
}

std::unique_ptr<mlir::Pass> vpux::Core::createDumpIntermediateValuesPass(StringRef configFileName, const Logger& log) {
    const auto config = parseYaml(configFileName);
    VPUX_THROW_WHEN(mlir::failed(config), "Failed to parse YAML file: {0}", configFileName);
    return std::make_unique<DumpIntermediateValuesPass>(config->filters, log);
}
