//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
namespace vpux {
namespace VPU {

int64_t getMaxLstmSequenceHiddenSizeConstant(config::ArchKind arch);
int64_t getMaxLstmCellHiddenSizeConstant(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
