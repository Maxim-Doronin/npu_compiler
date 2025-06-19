//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/convert_quantize_ops_to_nce_ops_strategy.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

namespace vpux::IE::arch37xx {

void ConvertQuantizeOpsToNceOpsStrategy::prepareAvgPool(mlir::ConversionTarget& toAvgPoolTarget,
                                                        mlir::RewritePatternSet& toAvgPoolPatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) const {
    // HW Eltwise and AvgPool supports only per-tensor bias/scale parameters
    // perTensor quantize/dequantize convert to avgpool
    // E#98802 avgpool is faster than add for big input size.
    // avgpool support rank >= 3, and currently convert shape to 4D does not support avgpool, so here limit to rank = 4

    // Dummy AvgPool's numerical stability is preferred for low-precision floating-point quantization (FP8),
    // eltwise Add with self could shift values further away from 0.0 causing precision loss.

    toAvgPoolTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(quantizeOp.getInput().getType());
        auto inRank = inType.getRank();

        return isLegalQuantizeOp(quantizeOp, _canUseCMajor) || inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(quantizeOp) &&
                !vpux::isFloat8Quantized(quantizeOp.getDstElemType()));
    });
    toAvgPoolTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        auto inType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType());
        auto inRank = inType.getRank();

        return isLegalDequantizeOp(dequantizeOp) || inRank != 4 ||
               (inType.getTotalAllocSize() <= vpux::VPU::getTotalCMXSize(dequantizeOp) &&
                !vpux::isFloat8Quantized(inType.getElementType())) ||
               hasConstProducer(dequantizeOp);
    });
    toAvgPoolTarget.addLegalOp<IE::AvgPoolOp>();

    toAvgPoolPatterns.add<IE::QuantizeDequantizeToAvgPool<IE::QuantizeOp>>(&ctx, log);
    toAvgPoolPatterns.add<IE::QuantizeDequantizeToAvgPool<IE::DequantizeOp>>(&ctx, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareEltwise(mlir::ConversionTarget& toEltwiseTarget,
                                                        mlir::RewritePatternSet& toEltwisePatterns,
                                                        mlir::MLIRContext& ctx, Logger& log) const {
    toEltwiseTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        return IE::isLegalQuantizeOp(quantizeOp, _canUseCMajor);
    });
    toEltwiseTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        return IE::isLegalDequantizeOp(dequantizeOp) || hasConstProducer(dequantizeOp);
    });
    toEltwiseTarget.addLegalOp<IE::AddOp>();
    toEltwiseTarget.addLegalOp<IE::QuantizeCastOp>();

    toEltwisePatterns.add<IE::DequantizeToAddRewriter>(&ctx, log);
    toEltwisePatterns.add<IE::QuantizeToAddRewriter>(&ctx, log);
}

void ConvertQuantizeOpsToNceOpsStrategy::prepareQuantToConv(mlir::ConversionTarget& quantToConvTarget,
                                                            mlir::RewritePatternSet& quantToConvPatterns,
                                                            mlir::MLIRContext& ctx, Logger& log) const {
    quantToConvTarget.addDynamicallyLegalOp<IE::QuantizeOp>([&](IE::QuantizeOp quantizeOp) {
        const auto isPerChannelQuantized = isPerChannelQuantizedType(quantizeOp.getOutput());
        auto outputLayerUsers = quantizeOp.getOutput().getUsers();
        auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
            return mlir::isa<IE::ConvolutionOp>(user);
        });

        return (anyUserIsConv && _canUseCMajor) || !isPerChannelQuantized;
    });

    quantToConvTarget.addDynamicallyLegalOp<IE::DequantizeOp>([&](IE::DequantizeOp dequantizeOp) {
        const auto isPerChannelQuantized = isPerChannelQuantizedType(dequantizeOp.getInput());
        auto outElemmentType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getOutput().getType()).getElementType();
        return !isPerChannelQuantized || !outElemmentType.isF16() || hasConstProducer(dequantizeOp);
    });

    quantToConvTarget.addLegalOp<Const::DeclareOp>();
    quantToConvTarget.addLegalOp<IE::GroupConvolutionOp>();

    quantToConvPatterns.add<IE::QuantizeToDwRewriter>(&ctx, log);
    quantToConvPatterns.add<IE::DequantizeToDwRewriter>(&ctx, log);
}

}  // namespace vpux::IE::arch37xx
