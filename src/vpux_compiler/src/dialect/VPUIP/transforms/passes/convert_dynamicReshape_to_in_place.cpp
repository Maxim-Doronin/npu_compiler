//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/logger/logger.hpp"
namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTDYNAMICRESHAPETOINPLACE
#define GEN_PASS_DEF_CONVERTDYNAMICRESHAPETOINPLACE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ConvertDynamicReshapeToInPlacePass
//

class ConvertDynamicReshapeToInPlacePass final :
        public VPUIP::impl::ConvertDynamicReshapeToInPlaceBase<ConvertDynamicReshapeToInPlacePass> {
public:
    explicit ConvertDynamicReshapeToInPlacePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult makeInPlaceDynamicReshape(VPUIP::SwKernelOp dynamicReshape, Logger _log);
};

mlir::LogicalResult ConvertDynamicReshapeToInPlacePass::makeInPlaceDynamicReshape(VPUIP::SwKernelOp dynamicReshape,
                                                                                  Logger _log) {
    auto input = dynamicReshape.getInputs()[0];
    auto output = dynamicReshape.getOutputs()[0];
    mlir::OpBuilder builder(dynamicReshape);

    auto inputRank = mlir::cast<vpux::NDTypeInterface>(input.getType()).getRank();
    auto outputRank = mlir::cast<vpux::NDTypeInterface>(output.getType()).getRank();

    bool inputIsDynamic = hasUngroupedInputBoundedBuffers(dynamicReshape);
    auto operands = dynamicReshape.getOperands();

    if (inputRank != outputRank) {
        return mlir::failure();
    }

    auto outputMemSpace = mlir::cast<vpux::NDTypeInterface>(output.getType()).getMemoryKind();
    auto inputMemSpace = mlir::cast<vpux::NDTypeInterface>(input.getType()).getMemoryKind();

    if (outputMemSpace != inputMemSpace) {
        return mlir::failure();
    }

    auto updatedDataBuffer = builder.create<VPUIP::ViewOp>(takeOpLoc(dynamicReshape, "view"), output.getType(), input);

    SmallVector<mlir::Value> inputs{dynamicReshape.getOperand(0), dynamicReshape.getOperand(1)};
    SmallVector<mlir::Value> outputs(
            operands.begin() + dynamicReshape.getInputs().size() + dynamicReshape.getDynamicInputShapes().size(),
            operands.end());

    inputs[0] = updatedDataBuffer.getResult();
    outputs[0] = updatedDataBuffer.getResult();

    if (inputIsDynamic) {
        auto inputBounds = dynamicReshape.getOperand(2);
        auto inputShapesMap = dynamicReshape.getDynamicInputShapesMapAttr();

        auto tileIndex = dynamicReshape.getTileIndexAttr();

        auto newSwKernelOp = builder.create<VPUIP::SwKernelOp>(
                takeOpLoc(dynamicReshape, "swKernel"), inputs, outputs[0], inputBounds, inputShapesMap,
                dynamicReshape.getDynamicOutputShapeBuffs(), dynamicReshape.getDynamicOutputShapesMapAttr(),
                dynamicReshape.getKernelFunction(), tileIndex);

        auto swKernelRun = *dynamicReshape.getBody().getOps<VPUIP::SwKernelRun>().begin();
        VPUIP::initSwKernel(newSwKernelOp, swKernelRun, _log);

        dynamicReshape.replaceAllUsesWith(newSwKernelOp);
        dynamicReshape.erase();
    } else {
        dynamicReshape->setOperand(0, updatedDataBuffer.getResult());
        dynamicReshape->setOperand(2, updatedDataBuffer.getResult());
    }

    return mlir::success();
}

void ConvertDynamicReshapeToInPlacePass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPUIP::SwKernelOp op) {
        auto kernelEntryName = getSwKernelEntryName(op);

        if (kernelEntryName == "dynamic_reshape") {
            if (mlir::succeeded(makeInPlaceDynamicReshape(op, _log))) {
                _log.trace("DynamicReshape {0} converted to in place", op->getLoc());
            } else {
                _log.trace("Can't convert DynamicReshape {0} to in place", op->getLoc());
            }
        }
    });
}

}  // namespace

//
// createConvertDynamicReshapeToInPlacePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertDynamicReshapeToInPlacePass(Logger log) {
    return std::make_unique<ConvertDynamicReshapeToInPlacePass>(log);
}
