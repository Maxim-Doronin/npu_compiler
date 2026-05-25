//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringRef.h>

namespace vpux {

// Below attributes will be moved to corresponding operations/type definitions as part of #E194772

// offsets is added to DeclareBufferOp to indicate view offset in each dimension of the
// original tensor.
constexpr llvm::StringLiteral viewOffsetsAttrName = "offsets";
// stridedInput is added to a DMA op to indicate that input argument to
// DMA op has dynamic strides
constexpr llvm::StringLiteral stridedInputAttrName = "stridedInput";
// stridedOutput is added to a DMA op to indicate that output argument to
// DMA op has dynamic strides
constexpr llvm::StringLiteral stridedOutputAttrName = "stridedOutput";
// Added to net.DataInfo op to indicate that user has marked this argument as
// having dynamic strides
constexpr llvm::StringLiteral dynamicStridesAttrName = "dynamicStrides";
// perClusterBufferOffset is added to DeclareBufferOp to indicate per-cluster buffer
// offsets (dim-level shape offsets) when the subview axis overlaps with the distributed
// tiling axis. Used during UnrollDistributedOps to compute correct per-cluster byte offsets.
constexpr llvm::StringLiteral perClusterBufferOffsetAttrName = "perClusterBufferOffset";

}  // namespace vpux
