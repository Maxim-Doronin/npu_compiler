//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp) {
    auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
    auto isFakeQuantizeOpOutput = true;
    for (auto user : postOp->getUsers()) {
        if (!mlir::isa<IE::FakeQuantizeOp>(user)) {
            isFakeQuantizeOpOutput = false;
            break;
        }
    }

    // since FusePostOps is called also after LowPrecisionPipeline
    const auto operandType = mlir::cast<vpux::NDTypeInterface>(postOp->getOperand(0).getType());
    const auto isQuantizedElemType = mlir::isa<mlir::quant::QuantizedType>(operandType.getElementType());

    return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
};

bool hasPerChannelQuantizedOutput(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        auto fq = mlir::dyn_cast<IE::FakeQuantizeOp>(user);
        if (fq == nullptr) {
            continue;
        }

        auto inLow = fq.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHigh = fq.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLow = fq.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHigh = fq.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(inLow == nullptr || inHigh == nullptr || outLow == nullptr || outHigh == nullptr,
                        "Got FakeQuantize with non-constant parameters, loc: {0}", fq->getLoc());

        if (!inLow.getContentAttr().isSplat() || !inHigh.getContentAttr().isSplat() ||
            !outLow.getContentAttr().isSplat() || !outHigh.getContentAttr().isSplat()) {
            return true;
        }
    }

    return false;
};

}  // namespace VPU
}  // namespace vpux
