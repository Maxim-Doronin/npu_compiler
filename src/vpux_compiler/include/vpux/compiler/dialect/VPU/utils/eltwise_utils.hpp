//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Types.h>

namespace vpux::VPU {
enum class EltwiseType : uint64_t;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {
bool isNCEEltwiseSupported(mlir::Operation* op, vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                           vpux::NDTypeInterface outputType, bool allowDifferentScales, bool allowDifferentZp,
                           bool checkLayout, bool checkChannelAlignment, LogCb logCb);

template <class ConcreteOp>
bool isEltwiseLhsActivation(ConcreteOp op) {
    const auto lhsType = mlir::cast<mlir::ShapedType>(op.getInput1().getType());
    const auto outShapeRes = mlir::cast<mlir::ShapedType>(op.getOutput().getType());

    return (lhsType == outShapeRes);
}

vpux::VPU::EltwiseType decodeNceEltwiseType(mlir::Operation* operation);

}  // namespace VPU
}  // namespace vpux
