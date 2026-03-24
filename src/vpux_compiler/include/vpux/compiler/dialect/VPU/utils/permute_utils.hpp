//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

namespace vpux::VPU {

mlir::FailureOr<VPU::DistributionInfo> applyPermutationOnDistributionInfo(vpux::NDTypeInterface inType,
                                                                          const VPU::DistributionInfo& inDistribution,
                                                                          mlir::AffineMap memPerm, DimsOrder srcOrder,
                                                                          DimsOrder dstOrder, ShapeRef srcShape,
                                                                          ShapeRef dstShape);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::FailureOr<VPU::DistributionInfoAttr> applyPermutationOnDistributionInfoAttr(T inDistributedType,
                                                                                  mlir::AffineMap memPerm,
                                                                                  DimsOrder srcOrder,
                                                                                  DimsOrder dstOrder, ShapeRef srcShape,
                                                                                  ShapeRef dstShape) {
    const auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());

    auto distributionInfoOrFailure = applyPermutationOnDistributionInfo(inDistributedType, inDistribution, memPerm,
                                                                        srcOrder, dstOrder, srcShape, dstShape);
    if (mlir::failed(distributionInfoOrFailure)) {
        return mlir::failure();
    }

    return VPU::DistributionInfo::getAttrFromClass(inDistributedType.getContext(), distributionInfoOrFailure.value());
}

}  // namespace vpux::VPU
