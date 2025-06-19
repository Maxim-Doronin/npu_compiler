//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/cast_utils.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {

mlir::LogicalResult isQuantizeCastValid(mlir::Location loc, mlir::Type srcType, mlir::Type dstType) {
    const auto srcBitSize = getElemTypeSize(srcType).count();
    const auto dstBitSize = getElemTypeSize(dstType).count();

    if (srcBitSize != dstBitSize) {
        return errorAt(loc, "Src type bit-width: {0} is different from dst type bit-width: {1}", srcBitSize,
                       dstBitSize);
    }

    if (mlir::isa<FloatType>(srcType) || mlir::isa<FloatType>(dstType)) {
        if (srcBitSize > 8 || dstBitSize > 8) {
            return errorAt(
                    loc, "Maximum low precision bit width is 8 for both src and dst types, but got src: {0}, dst: {0}",
                    srcBitSize, dstBitSize);
        }
    } else if (mlir::isa<IntegerType>(srcType) || mlir::isa<IntegerType>(dstType)) {
        if (srcBitSize > 16 || dstBitSize > 16) {
            return errorAt(loc,
                           "Maximum integer bit width is 16 for both src and dst types, but got src: {0}, dst: {1}",
                           srcBitSize, dstBitSize);
        }
    }
    // Admitting all cases except I1, as currently we're treating it as pure I1 data type
    // not requiring any quant cast
    if (srcBitSize < 2 || !isPowerOfTwo(srcBitSize)) {
        return errorAt(loc, "Src type bit size: {0} is not a power of two", srcBitSize);
    }

    if ((isFloat8(srcType) && srcType != mlir::cast<mlir::quant::QuantizedType>(dstType).getStorageType()) ||
        (isFloat8(dstType) && dstType != mlir::cast<mlir::quant::QuantizedType>(srcType).getStorageType())) {
        return errorAt(loc, "Low precision float types can only be casted to types of the same bit format");
    }

    return mlir::success();
}

}  // namespace vpux
