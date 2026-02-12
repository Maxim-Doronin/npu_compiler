//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFVERTICALFUSION
#define GEN_PASS_DEF_SCFVERTICALFUSION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// SCFVerticalFusionPass
//

class SCFVerticalFusionPass final : public VPU::impl::SCFVerticalFusionBase<SCFVerticalFusionPass> {
public:
    SCFVerticalFusionPass(bool enableDynamicDimAlignment, Logger log)
            : _enableDynamicDimAlignment(enableDynamicDimAlignment) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    bool _enableDynamicDimAlignment = false;
};

mlir::LogicalResult SCFVerticalFusionPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableDynamicDimAlignment.hasValue()) {
        _log.trace("Overloading SCFVerticalFusionPass argument by MLIR variable");
        _enableDynamicDimAlignment = enableDynamicDimAlignment.getValue();
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void SCFVerticalFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);
    mlir::IRRewriter irBuilder(builder);

    llvm::SetVector<mlir::Operation*> fusedOps;

    if (_enableDynamicDimAlignment) {
        VPU::setDynamicDimAlignment(func);
    }

    func->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::TilingInterface operation) {
        auto* op = operation.getOperation();

        if (fusedOps.contains(op)) {
            _log.nest().trace("Operation has already been fused");
            return;
        }

        if (!op->hasAttr(tilingStrategy)) {
            _log.nest().trace("No tiling strategy or it has already been applied.");
            return;
        }

        const auto options = VPU::TilingContextOptions(VPU::TilingContextOptions::ContextType::TILING,
                                                       /* enableSCFTiling = */ true);
        auto tilingContext = VPU::createTilingContext(op, options);
        const auto fused = tilingContext.applySCFTilingAndFusion(irBuilder, _log);

        if (!fused.empty()) {
            fusedOps.insert(fused.begin(), fused.end());
        }
    });

    // remove dynamic alignment
    VPU::removeDynamicDimAlignment(func);
}

}  // namespace

//
// createSCFVerticalFusionPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFVerticalFusionPass(bool enableDynamicDimAlignment, Logger log) {
    return std::make_unique<SCFVerticalFusionPass>(enableDynamicDimAlignment, log);
}
