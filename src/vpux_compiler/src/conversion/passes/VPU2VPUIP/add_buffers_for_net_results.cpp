//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/allocate_buffers_for_net_results.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>

namespace vpux {
#define GEN_PASS_DECL_ADDBUFFERSFORNETRESULTS
#define GEN_PASS_DEF_ADDBUFFERSFORNETRESULTS
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// AddBuffersForNetResults
//

class AddBuffersForNetResults final : public impl::AddBuffersForNetResultsBase<AddBuffersForNetResults> {
public:
    explicit AddBuffersForNetResults(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnFunc
//

void AddBuffersForNetResults::safeRunOnModule() {
    auto module = getOperation();

    SmallVector<mlir::func::CallOp> callOps;
    module.walk([&](mlir::func::CallOp callOp) {
        callOps.push_back(callOp);
    });
    SmallVector<mlir::func::FuncOp> funcOps;
    module.walk([&](mlir::func::FuncOp funcOp) {
        auto closestModuleParentOp = funcOp->getParentOfType<mlir::ModuleOp>();
        auto moduleName = closestModuleParentOp.getSymName().value_or("");
        if (moduleName == "VPU.SW" || funcOp.isExternal()) {
            /*
            Example of external functions:
            module @VPU.SW {
                func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
                    attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
            }
            */
            _log.trace("Function '@{0}' at {1} is skipped as it's either external or part of @VPU.SW Module",
                       funcOp.getSymName(), funcOp.getLoc());
            return mlir::WalkResult::skip();
        }
        funcOps.push_back(funcOp);
        return mlir::WalkResult::advance();
    });

    vpux::allocateBuffersForNetResults(callOps, funcOps, _log);
}

}  // namespace

//
// createAddBuffersForNetResults
//

std::unique_ptr<mlir::Pass> vpux::createAddBuffersForNetResults(Logger log) {
    return std::make_unique<AddBuffersForNetResults>(log);
}
