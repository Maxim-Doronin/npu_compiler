//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

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
    explicit SCFVerticalFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SCFVerticalFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);
    mlir::IRRewriter irBuilder(builder);

    llvm::SetVector<mlir::Operation*> fusedOps;

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

        auto tilingContext = VPU::createTilingContext(op, /* enableSCFTiling = */ true);
        auto fused = tilingContext.applyVerticalFusion(irBuilder, _log);

        if (!mlir::failed(fused)) {
            fusedOps.insert(fused.value().begin(), fused.value().end());
        }
    });
}

}  // namespace

//
// createSCFVerticalFusionPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFVerticalFusionPass(Logger log) {
    return std::make_unique<SCFVerticalFusionPass>(log);
}
