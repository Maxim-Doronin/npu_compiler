//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/string_ref.hpp"

namespace vpux::VPU {

constexpr StringRef WORKLOAD_MANAGEMENT_STATUS = "VPU.WorkloadManagementStatus";

WorkloadManagementStatus getWorkloadManagementStatus(mlir::ModuleOp module);
void setWorkloadManagementStatus(mlir::ModuleOp module, WorkloadManagementStatus value);

}  // namespace vpux::VPU
