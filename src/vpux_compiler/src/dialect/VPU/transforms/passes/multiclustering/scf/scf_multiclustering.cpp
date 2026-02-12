//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFMULTICLUSTERING
#define GEN_PASS_DEF_SCFMULTICLUSTERING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// SCFMulticlusteringPass
//

class SCFMulticlusteringPass final : public VPU::impl::SCFMulticlusteringBase<SCFMulticlusteringPass> {
public:
    explicit SCFMulticlusteringPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void SCFMulticlusteringPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);
    mlir::IRRewriter irBuilder(builder);

    llvm::SetVector<mlir::Operation*> fusedOps;

    func->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::TilingInterface operation) {
        auto* op = operation.getOperation();
        _log.trace("Attempt to multicluster op at loc {0}", op->getLoc());

        if (fusedOps.contains(op)) {
            _log.nest().trace("Operation has already been fused");
            return;
        }

        if (!op->hasAttr(multiClusterStrategy)) {
            _log.nest().trace("No multicluster strategy or it has already been applied.");
            return;
        }
        const auto mcStrategy = op->getAttr(multiClusterStrategy);
        const auto options = VPU::TilingContextOptions(VPU::TilingContextOptions::ContextType::MULTICLUSTERING,
                                                       /* enableSCFTiling = */ true);
        auto tilingContext = VPU::createTilingContext(op, options);
        const auto fused = tilingContext.applySCFTilingAndFusion(irBuilder, _log);

        if (!fused.empty()) {
            // Op and its producers are compatible, therefore there is the
            // possibility of no spilling between them.
            fusedOps.insert(fused.begin(), fused.end());
            _log.nest().trace("Op was vertically fused in multiclustering loop.");
        } else {
            // Transition between op and its producers cannot be done without
            // a spill.
            auto res = tilingContext.applyTiling(irBuilder, _log);
            VPUX_THROW_WHEN(mlir::failed(res), "Multiclustering failed for op at loc {0}, with strategy {1}",
                            op->getLoc(), mcStrategy);
            fusedOps.insert(op);
            _log.nest().trace("Op was wrapped in multiclustering loop.");
        }
    });
}

}  // namespace

//
// createSCFMulticlusteringPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFMulticlusteringPass(Logger log) {
    return std::make_unique<SCFMulticlusteringPass>(log);
}
