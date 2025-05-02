//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/dma_transaction.hpp"

namespace vpux {
namespace VPUASM {

DMATransactionConfig getDMATransactionConfigFromDescriptorAttr(VPUIP::DMADescriptorAttr& descriptor,
                                                               bool isConversionEnabled,
                                                               bool isActivationDecompression) {
    DMATransactionConfig transactionConfig{};

    auto numPlanes = descriptor.getNumPlanes().getInt();
    auto length = descriptor.getLen().getInt();
    auto srcWidth = descriptor.getSrcWidth().getInt();
    auto srcStride = descriptor.getSrcStride().getInt();
    auto srcPlaneStride = descriptor.getSrcPlaneStride().getInt();
    auto dstWidth = descriptor.getDstWidth().getInt();
    auto dstStride = descriptor.getDstStride().getInt();
    auto dstPlaneStride = descriptor.getDstPlaneStride().getInt();

    int64_t srcDimSize1 = 0;
    int64_t dstDimSize1 = 0;

    // For decompression srcDimSize1 & dstDimSize1 should not be relevant since they are based on
    // srcWidth/dstWidth (that are not programmed)
    if (!isActivationDecompression) {
        srcDimSize1 = srcWidth ? static_cast<int64_t>((length / srcWidth) - 1) : 0;
        dstDimSize1 = (dstWidth && (length > dstWidth)) ? static_cast<int64_t>((length / dstWidth) - 1) : 0;
    }

    // DMA only does FP32 -> FP16/BF16 conversions,
    // Because of this, dstDimSize1 will always be half of the original value
    if (isConversionEnabled && dstDimSize1) {
        dstDimSize1 = ((dstDimSize1 + 1) / 2) - 1;
    }

    int64_t numDims = 0;
    if (numPlanes > 1) {
        numDims = 2;
    } else if (srcWidth == srcStride && dstWidth == dstStride) {
        numDims = 0;
    } else {
        numDims = 1;
    }
    transactionConfig.numDims = numDims;

    switch (numDims) {
    case 2:
        VPUX_THROW_WHEN(numPlanes == 0, "numPlanes cannot be 0 for a 3D transaction");
        transactionConfig.srcDimSizes[2] = numPlanes - 1;
        transactionConfig.dstDimSizes[2] = numPlanes - 1;

        transactionConfig.srcStrides[2] = srcPlaneStride;
        transactionConfig.dstStrides[2] = dstPlaneStride;

        [[fallthrough]];
    case 1:
        transactionConfig.srcDimSizes[1] = srcDimSize1;
        transactionConfig.dstDimSizes[1] = dstDimSize1;

        transactionConfig.srcStrides[1] = srcStride;
        transactionConfig.dstStrides[1] = dstStride;

        [[fallthrough]];
    case 0:
        [[fallthrough]];
    default:
        transactionConfig.srcDimSizes[0] = srcWidth;
        transactionConfig.dstDimSizes[0] = dstWidth;
        break;
    }

    return transactionConfig;
}

DMATransactionConfig getDMATransactionConfigFromTransaction(DMATransaction& transaction) {
    DMATransactionConfig transactionConfig{};

    // These strict checks may need relaxing for certain types of special DMA transactions
    VPUX_THROW_WHEN(transaction.inputs.size() != 1, "DMA transaction with unsupported number of input patterns");
    VPUX_THROW_WHEN(transaction.outputs.size() != 1, "DMA transaction with unsupported number of output patterns");

    auto& inputPattern = transaction.inputs.front();
    auto& outputPattern = transaction.outputs.front();

    auto checkPatternComponent = [&](auto& input, auto& result) {
        VPUX_THROW_WHEN(input.size() == 0, "DMA pattern conversion check failure");
        VPUX_THROW_WHEN(input.size() > result.size(), "DMA pattern conversion check failure");
    };

    checkPatternComponent(inputPattern.dims, transactionConfig.srcDimSizes);
    checkPatternComponent(inputPattern.strides, transactionConfig.srcStrides);
    checkPatternComponent(outputPattern.dims, transactionConfig.dstDimSizes);
    checkPatternComponent(outputPattern.strides, transactionConfig.dstStrides);

    VPUX_THROW_WHEN(inputPattern.dims.size() != inputPattern.strides.size(),
                    "Mismatch between pattern dim count and stride count");
    VPUX_THROW_WHEN(outputPattern.dims.size() != outputPattern.strides.size(),
                    "Mismatch between pattern dim count and stride count");

    // Pattern layout
    // ________________________________
    // Index      || 0  | 1  | 2  | 3  |
    // Dim        || d3 | d2 | d1 | d0 |
    // Stride     || s3 | s2 | s1 | s0 |
    //                ^                |
    //          highest rank           |
    // ________________________________|
    //

    // DMA layout
    // ________________________________
    // Index      || 0  | 1  | 2  | 3  |
    // Dim        || d0 | d1 | d2 | d3 |
    // Stride     || 0  | s0 | s1 | s2 |
    // ________________________________|

    // Reverse dims and strides from memref order to DMA order
    std::copy(inputPattern.dims.rbegin(), inputPattern.dims.rend(), transactionConfig.srcDimSizes.begin());
    std::copy(inputPattern.strides.rbegin(), inputPattern.strides.rend() - 1, transactionConfig.srcStrides.begin() + 1);
    std::copy(outputPattern.dims.rbegin(), outputPattern.dims.rend(), transactionConfig.dstDimSizes.begin());
    std::copy(outputPattern.strides.rbegin(), outputPattern.strides.rend() - 1,
              transactionConfig.dstStrides.begin() + 1);

    const auto minOne = [](auto& val) {
        val > 1 ? val -= 1 : val = 0;
    };

    // Subtract 1 from all dims except the first one
    // This is HW-specific, but since all arches respect this logic, keep it in the common VPUASM dialect
    std::for_each(transactionConfig.srcDimSizes.begin() + 1, transactionConfig.srcDimSizes.end(), minOne);
    std::for_each(transactionConfig.dstDimSizes.begin() + 1, transactionConfig.dstDimSizes.end(), minOne);

    transactionConfig.numDims = std::max(inputPattern.dims.size(), outputPattern.dims.size()) - 1;

    return transactionConfig;
}

DMATransactionConfig getDMATransactionConfig(VPUASM::NNDMAOp dmaOp, bool isConversionEnabled,
                                             bool isActivationDecompression) {
    VPUASM::DMATransactionConfig transactionConfig;
    if (auto transactionAttr = dmaOp.getDmaTransactionAttr()) {
        auto transaction = transactionAttr.getDMATransaction();
        transactionConfig = VPUASM::getDMATransactionConfigFromTransaction(transaction);
    } else {
        if (auto descriptorAttr = dmaOp.getDmaDescriptorAttr()) {
            transactionConfig = VPUASM::getDMATransactionConfigFromDescriptorAttr(descriptorAttr, isConversionEnabled,
                                                                                  isActivationDecompression);
        } else {
            VPUX_THROW("Transaction cannot be composed without both transaction and descriptor attributes");
        }
    }

    return transactionConfig;
}

}  // namespace VPUASM
}  // namespace vpux
