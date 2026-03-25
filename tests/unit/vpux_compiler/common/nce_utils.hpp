//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

// see src/vpux_translate_utils/src/hwtest/hwtest_utils.cpp for a more detailed implementation of this function
// outStart and outEnd represent the {W, H, C} dimensions
inline vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder& builder, ArrayRef<int64_t> outStart,
                                              ArrayRef<int64_t> outEnd) {
    auto pad = VPU::getPaddingAttr(builder.getContext(), 0, 0, 0, 0);

    return builder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), getIntArrayAttr(builder, outStart),
                                            getIntArrayAttr(builder, outEnd), pad, VPU::MPEMode::CUBOID_16x16);
}
