//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::VPU::arch50xx {

constexpr bool shaveControlsDpuValue = true;
constexpr bool shaveDpuNeedWeightTable = true;

constexpr bool getShaveControlsDpuConstraint() {
    return shaveControlsDpuValue;
}

constexpr bool getShaveDpuNeedWeightTable() {
    return shaveDpuNeedWeightTable;
}

}  // namespace vpux::VPU::arch50xx
