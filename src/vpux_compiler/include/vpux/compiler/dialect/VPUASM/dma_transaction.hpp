//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#pragma once

namespace vpux {
namespace VPUASM {

// Arch-agnostic limit to DMA number of dimensions
// Should be set to the highest number of dimensions from all supported arches
constexpr size_t DMA_MAX_NUM_DIMS = 6;

struct DMATransactionConfig {
    std::array<uint64_t, DMA_MAX_NUM_DIMS> srcDimSizes = {};
    std::array<int64_t, DMA_MAX_NUM_DIMS> srcStrides = {};
    std::array<uint64_t, DMA_MAX_NUM_DIMS> dstDimSizes = {};
    std::array<int64_t, DMA_MAX_NUM_DIMS> dstStrides = {};

    uint64_t numDims = {};
};

DMATransactionConfig getDMATransactionConfigFromDescriptorAttr(VPUIP::DMADescriptorAttr& descriptor,
                                                               bool isConversionEnabled,
                                                               bool isActivationDecompression);

DMATransactionConfig getDMATransactionConfigFromTransaction(DMATransaction& transaction);

DMATransactionConfig getDMATransactionConfig(VPUASM::NNDMAOp dmaOp, bool isConversionEnabled = false,
                                             bool isActivationDecompression = false);

}  // namespace VPUASM
}  // namespace vpux
