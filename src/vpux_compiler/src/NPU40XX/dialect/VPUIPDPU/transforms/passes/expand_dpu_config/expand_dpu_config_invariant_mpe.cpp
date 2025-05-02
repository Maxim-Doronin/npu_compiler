//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace {

// Helper function to calculate zero point offset for input activations and weights
uint8_t getZeroPointBias(mlir::Value operand) {
    // Get also ZP
    SmallVector<uint8_t> quantZeroPoints;

    auto type = mlir::cast<vpux::NDTypeInterface>(operand.getType());

    auto elementType = type.getElementType();
    if (const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType =
                       mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    // Return only the first element as the zero point bias
    return quantZeroPoints[0];
}

}  // namespace

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantMPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos)) {
        auto inActType = getBaseType(mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType());
        if (inActType.isInteger(CHAR_BIT)) {
            builder.create<MPEActivationBiasOp>(origInvOp.getLoc(), getZeroPointBias(inAct));
        }
    }

    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        auto wtType = getBaseType(mlir::cast<mlir::MemRefType>(weights.getType()).getElementType());

        if (wtType.isInteger(CHAR_BIT)) {
            builder.create<MPEWeightsBiasOp>(origInvOp.getLoc(), getZeroPointBias(weights));
        }
    }

    // mpe_daz not set/used in graph_file nce_lib so then
    // MPEDenormalOperandsFTZOp will not be instantiated here.

    return mlir::success();
}
