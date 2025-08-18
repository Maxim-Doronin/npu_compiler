//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {
namespace NCESparsity {

const VPU::SparsitySupport FULLY_SUPPORTED_SPARSITY_MODE =
        SparsitySupport::SPARSE_INPUTS | SparsitySupport::SPARSE_OUTPUTS | SparsitySupport::SPARSE_WEIGHTS;

inline VPU::SparsitySupport bitwiseNot(const VPU::SparsitySupport bits) {
    static_assert(sizeof(bits) == sizeof(uint32_t), "VPU::SparsitySupport has unexpected size");
    return static_cast<VPU::SparsitySupport>(~static_cast<uint32_t>(bits));
}

}  // namespace NCESparsity
}  // namespace VPU
}  // namespace vpux
