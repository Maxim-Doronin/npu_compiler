//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_invariant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUInvariantMPE(
        VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, mlir::Block* invBlock,
        const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    if (auto inAct = getInvBlockArg(BlockArg::ACT_IN, invBlock, invBlockArgsPos)) {
        auto inActType = getBaseType(mlir::cast<mlir::MemRefType>(inAct.getType()).getElementType());
        if (inActType.isInteger(CHAR_BIT)) {
            builder.create<MPEActivationBiasOp>(origInvOp.getLoc(), VPUIPDPU::getZeroPoint(inAct.getType()));
        }
    }

    if (auto weights = getInvBlockArg(BlockArg::WEIGHTS, invBlock, invBlockArgsPos)) {
        auto wtType = getBaseType(mlir::cast<mlir::MemRefType>(weights.getType()).getElementType());
        if (wtType.isUnsignedInteger(CHAR_BIT)) {
            builder.create<MPEWeightsBiasOp>(origInvOp.getLoc(), VPUIPDPU::getZeroPoint(weights.getType()));
        }
    }

    // mpe_daz not set/used in graph_file nce_lib so then
    // MPEDenormalOperandsFTZOp will not be instantiated here.

    return mlir::success();
}
