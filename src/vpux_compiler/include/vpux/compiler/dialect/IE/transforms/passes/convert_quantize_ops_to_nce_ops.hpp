//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

namespace vpux::IE {

mlir::Value buildDwWeights(const mlir::Location& loc, const int64_t OC, const mlir::Type& elementType,
                           mlir::PatternRewriter& rewriter);
bool isQuantizedPerAxis(mlir::Value val);
// cst -> dequant case.
// When a dequantize operation has a constant producer, execute dequantization on activation shaves.
// ConvertQuantizeOpsToNCEOps must skip such dequantize operations.
// Mark them legal.
inline bool hasConstProducer(IE::DequantizeOp dequantOp) {
    return mlir::isa_and_nonnull<Const::DeclareOp>(dequantOp.getInput().getDefiningOp());
}
bool isLegalQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor);
bool isLegalDequantizeOp(IE::DequantizeOp dequantizeOp);
bool isPerChannelQuantizedType(mlir::Value val);

}  // namespace vpux::IE
