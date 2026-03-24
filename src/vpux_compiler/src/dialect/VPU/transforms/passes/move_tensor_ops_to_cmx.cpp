//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVETENSOROPSTOCMX
#define GEN_PASS_DEF_MOVETENSOROPSTOCMX
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
namespace {

//
// MoveTensorOpsToCMXPass
//

class MoveTensorOpsToCMXPass final : public VPU::impl::MoveTensorOpsToCMXBase<MoveTensorOpsToCMXPass> {
public:
    explicit MoveTensorOpsToCMXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveTensorOpsToCMXPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::NCEOpInterface nceOp) {
        if (!VPU::isNceOpWithPadAttr(nceOp.getOperation())) {
            return;
        }

        auto consumerOp = nceOp.getOperation();
        auto input = nceOp.getOperation()->getOperand(0);

        auto inputCopy = mlir::dyn_cast_or_null<VPU::CopyOp>(input.getDefiningOp());
        if (inputCopy == nullptr) {
            return;
        }

        auto padOp = mlir::dyn_cast_or_null<mlir::tensor::PadOp>(inputCopy.getInput().getDefiningOp());
        if (padOp == nullptr) {
            return;
        }

        inputCopy.getInputMutable().set(padOp.getSource());
        vpux::inferReturnTypes(inputCopy, vpux::InferShapedTypeMode::SHAPE);

        inputCopy->moveBefore(padOp.getOperation());

        padOp.getSourceMutable().set(inputCopy.getResult());
        auto newResultType =
                mlir::cast<vpux::NDTypeInterface>(padOp.getResultType()).changeMemSpace(inputCopy.getOutMemSpaceAttr());
        padOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(newResultType));

        consumerOp->setOperand(0, padOp.getResult());
    });

    func.walk([&](mlir::tensor::InsertSliceOp insertSliceOp) {
        auto input = insertSliceOp.getSource();
        auto castOp = mlir::dyn_cast_or_null<mlir::tensor::CastOp>(input.getDefiningOp());
        if (castOp == nullptr) {
            return;
        }

        auto copyOp = mlir::dyn_cast_or_null<VPU::CopyOp>(castOp.getSource().getDefiningOp());
        if (copyOp == nullptr || !copyOp->hasOneUse()) {
            return;
        }

        auto copyInputType = mlir::cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
        auto origDstType = mlir::cast<NDTypeInterface>(castOp.getResult().getType());
        auto castDstType = origDstType.changeMemSpace(copyInputType.getMemSpace());

        castOp.getSourceMutable().set(copyOp.getInput());
        castOp.getDest().setType(mlir::cast<mlir::RankedTensorType>(castDstType));

        castOp->moveBefore(copyOp.getOperation());

        copyOp.getInputMutable().set(castOp.getResult());
        copyOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(origDstType));

        insertSliceOp.getSourceMutable().set(copyOp.getResult());
    });
}
}  // namespace

//
// createMoveTensorOpsToCMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createMoveTensorOpsToCMXPass(Logger log) {
    return std::make_unique<MoveTensorOpsToCMXPass>(log);
}
