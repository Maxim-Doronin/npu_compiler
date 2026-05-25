//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>

namespace vpux::VPU {

mlir::RankedTensorType inferNCEMatmulOutputType(vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                                                vpux::NDTypeInterface origOutputType);

bool isNCEMatMulSupported(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                          vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, vpux::LogCb logCb,
                          bool checkLayout, bool checkChannelAlignment);

}  // namespace vpux::VPU
