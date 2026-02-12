//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/core/layers.hpp"

namespace vpux {

int64_t updateSizeForCompression(int64_t origTensorSize, llvm::ArrayRef<int64_t> origShape, int64_t sparsityMapSize) {
    // In worst case scenario depending on the content of activation, its final size after
    // compression might be bigger than original size. Compiler before performing DDR
    // allocation needs to adjust required size by this buffer
    // Formula from HAS for Dense BITC is following:
    //   DTS = X * Y * Z * (element size in bytes)
    //   denseSize = (DTS * (65/64)) + 1
    //   DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    auto worstCaseSize = static_cast<int64_t>(origTensorSize * 65 / 64) + 1;

    // Formula from HAS for Sparse BITC is following:
    //   BBS - bitmap buffer size in bytes
    //   DTS = X * Y * Z * (element size in bytes)
    //   sparseSize = ((DTS + BBS + (2 * X * Y)) * (65/64) ) + 1
    //   DDR Allocation (32B Aligned) = sparseSize + ( (sparseSize % 32) ? (32 – (sparseSize % 32) : 0)
    if (sparsityMapSize != 0) {
        worstCaseSize = static_cast<int64_t>(
                origTensorSize + sparsityMapSize +
                (2 * origShape[vpux::Dims4D::Act::W.ind()] * origShape[vpux::Dims4D::Act::H.ind()]) * (65 / 64) + 1);
    }

    if (worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT) {
        worstCaseSize += ACT_COMPRESSION_BUF_SIZE_ALIGNMENT - worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT;
    }
    return worstCaseSize;
}

}  // namespace vpux
