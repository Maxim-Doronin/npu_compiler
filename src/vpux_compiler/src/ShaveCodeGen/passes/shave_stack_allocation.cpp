//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SHAVESTACKALLOCATION
#define GEN_PASS_DEF_SHAVESTACKALLOCATION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

static constexpr unsigned MAX_ALLOC_BYTES = 64;

class ShaveStackAllocationPass final : public ShaveCodeGen::impl::ShaveStackAllocationBase<ShaveStackAllocationPass> {
public:
    using SwKernelUses = SmallVector<vpux::VPU::GenericSwLayerOp>;
    explicit ShaveStackAllocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    mlir::LogicalResult addScratchBuffers(mlir::func::FuncOp func, SwKernelUses& uses);

    void safeRunOnModule() final {
        auto moduleOp = getOperation();
        auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

        mlir::PassManager pm(&getContext());
        pm.addPass(mlir::bufferization::createBufferHoistingPass());
        pm.addPass(mlir::bufferization::createBufferLoopHoistingPass());

        mlir::bufferization::PromoteBuffersToStackPassOptions opts;
        opts.maxAllocSizeInBytes = MAX_ALLOC_BYTES;
        opts.maxRankOfAllocatedMemRef = 1;
        pm.addPass(mlir::bufferization::createPromoteBuffersToStackPass(opts));

        // Create a cache of uses of every FuncOp by SwKernelOps.
        llvm::DenseMap<mlir::func::FuncOp, SwKernelUses> funcUseMap;
        moduleOp.walk([&](vpux::VPU::GenericSwLayerOp swLayerOp) {
            auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(swLayerOp.getCallee());
            funcUseMap[kernelFunc].push_back(swLayerOp);
        });

        auto funcOps = swModule.getOps<mlir::func::FuncOp>();

        for (auto func : funcOps) {
            if (mlir::failed(pm.run(func))) {
                _log.trace("Failed to promote buffers to stack");
                return signalPassFailure();
            }

            if (mlir::failed(addScratchBuffers(func, funcUseMap[func]))) {
                _log.trace("Failed to add scratch buffers");
                return signalPassFailure();
            }
        }
    }
};

mlir::LogicalResult ShaveStackAllocationPass::addScratchBuffers(mlir::func::FuncOp func, SwKernelUses& uses) {
    auto allocOps = vpux::to_small_vector(func.getOps<mlir::memref::AllocOp>());
    if (allocOps.empty()) {
        return mlir::success();
    }

    auto& ctx = getContext();
    mlir::OpBuilder builder(&ctx);

    // Collect memref.alloc ops and replace them with kernel arguments.
    auto inputTys = vpux::to_small_vector(func.getArgumentTypes());
    auto outputTys = vpux::to_small_vector(func.getResultTypes());
    auto& funcBlock = func.getFunctionBody();
    SmallVector<mlir::Type> scratchTypes;
    bool hasInvalidAllocOps = false;
    for (auto allocOp : allocOps) {
        auto allocTy = allocOp.getType();
        if (!allocTy.hasStaticShape() || !allocTy.areTrailingDimsContiguous(allocTy.getRank())) {
            // Reject dynamic and non-contiguous memref allocs since these are
            // not supported at the moment.
            // TODO: E#192657 convert dynamic and strided allocations to scratch buffers
            mlir::emitError(allocOp.getLoc()) << "Unexpected allocation in shave kernel.";
            hasInvalidAllocOps = true;
            continue;
        }
        auto arg = funcBlock.addArgument(allocOp.getType(), allocOp.getLoc());
        allocOp->getResult(0).replaceAllUsesWith(arg);
        allocOp->erase();
        inputTys.push_back(arg.getType());
        scratchTypes.push_back(mlir::memref::getTensorTypeFromMemRefType(arg.getType()));
    }

    auto funcType = mlir::FunctionType::get(&ctx, inputTys, outputTys);
    func.setFunctionType(funcType);

    // Add the new scratch buffers to every call of the kernel.
    for (auto swLayerOp : uses) {
        SmallVector<mlir::Type> resultTys;
        resultTys.reserve(swLayerOp->getResultTypes().size() + scratchTypes.size());
        resultTys.insert(resultTys.end(), swLayerOp->getResultTypes().begin(), swLayerOp->getResultTypes().end());
        resultTys.insert(resultTys.end(), scratchTypes.begin(), scratchTypes.end());
        builder.setInsertionPoint(swLayerOp);
        auto newSwLayerOp = builder.create<VPU::GenericSwLayerOp>(swLayerOp->getLoc(), resultTys, swLayerOp.getCallee(),
                                                                  swLayerOp->getOperands());
        size_t numResults = swLayerOp->getResultTypes().size();
        SmallVector<mlir::Value> results(newSwLayerOp->getResults().begin(),
                                         newSwLayerOp->getResults().begin() + numResults);
        swLayerOp->replaceAllUsesWith(results);
        swLayerOp->erase();
    }

    if (hasInvalidAllocOps) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createShaveStackAllocationPass(Logger log) {
    return std::make_unique<ShaveStackAllocationPass>(log);
}
