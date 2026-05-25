//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_quantize_ops_to_nce_ops.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTQUANTIZEOPSTONCEOPS
#define GEN_PASS_DEF_CONVERTQUANTIZEOPSTONCEOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace vpux::IE {

bool isQuantizedPerAxis(mlir::Value val) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
    return mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
}

bool shouldConvertQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor) {
    auto outputLayerUsers = quantizeOp.getOutput().getUsers();
    auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
        return mlir::isa<IE::ConvolutionOp>(user);
    });

    return anyUserIsConv && canUseCMajor;
};

bool isPerChannelQuantizedType(mlir::Value val) {
    auto elemType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
    auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    if (perAxisType == nullptr) {
        return false;
    }

    auto rank = mlir::cast<vpux::NDTypeInterface>(val.getType()).getRank();
    if (rank != 4) {
        return false;
    }

    auto quantizeDim = perAxisType.getQuantizedDimension();
    return quantizeDim == Dims4D::Act::C.ind();
}

bool is16BitStorageType(vpux::NDTypeInterface type) {
    auto quantType = mlir::cast<mlir::quant::QuantizedType>(type.getElementType());
    return quantType.getStorageTypeIntegralWidth() == 16;
}

bool is8BitStorageType(vpux::NDTypeInterface type) {
    auto quantType = mlir::cast<mlir::quant::QuantizedType>(type.getElementType());
    return quantType.getStorageTypeIntegralWidth() == 8;
}

}  // namespace vpux::IE

namespace {

//
// ConvertQuantizeOpsToNceOpsPass
//

class ConvertQuantizeOpsToNceOpsPass final :
        public IE::impl::ConvertQuantizeOpsToNceOpsBase<ConvertQuantizeOpsToNceOpsPass> {
public:
    explicit ConvertQuantizeOpsToNceOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertQuantizeOpsToNceOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getConvertQuantizeOpsToNceOpsStrategy();

    mlir::RewritePatternSet toAvgPoolPatterns(&ctx);
    strategy->prepareAvgPool(toAvgPoolPatterns, ctx, _log);
    walkAndApplyPatterns(func, std::move(toAvgPoolPatterns));

    // perTensor quantize/dequantize convert to add or and
    mlir::RewritePatternSet toEltwisePatterns(&ctx);
    strategy->prepareEltwise(toEltwisePatterns, ctx, _log);
    walkAndApplyPatterns(func, std::move(toEltwisePatterns));

    // per-axis scales and per-tensor zero points quantize/dequantize convert to DW conv
    mlir::RewritePatternSet quantToConvPatterns(&ctx);
    strategy->prepareQuantToConv(quantToConvPatterns, ctx, _log);
    walkAndApplyPatterns(func, std::move(quantToConvPatterns));
}

}  // namespace

//
// createConvertQuantizeOpsToNceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertQuantizeOpsToNceOpsPass(Logger log) {
    return std::make_unique<ConvertQuantizeOpsToNceOpsPass>(log);
}
