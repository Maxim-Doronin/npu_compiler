//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef AUTO_PADDING_ODU = "VPU.AutoPaddingODU";
constexpr StringRef AUTO_PADDING_IDU = "VPU.AutoPaddingIDU";

constexpr std::string_view INPUT_PADDING_ATTR_NAME = "input_padding";
constexpr std::string_view OUTPUT_PADDING_ATTR_NAME = "output_padding";

constexpr int64_t WIDTH16_CHANNEL_LIMIT = 10;
constexpr int64_t FP16_WIDTH = 16;

bool hasAutoPadding(mlir::ModuleOp);
bool hasAutoPaddingODU(mlir::ModuleOp);
bool hasAutoPaddingIDU(mlir::ModuleOp);
bool inputCompatibleWithAutoPad(vpux::NDTypeInterface);
bool outputCompatibleWithAutoPad(vpux::NDTypeInterface);
bool canAutopadOutput(mlir::Operation*, std::optional<vpux::NDTypeInterface> type = std::nullopt);

}  // namespace VPU
}  // namespace vpux
