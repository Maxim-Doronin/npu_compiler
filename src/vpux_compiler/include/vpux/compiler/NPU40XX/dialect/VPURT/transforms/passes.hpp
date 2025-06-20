//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

namespace vpux::VPURT::arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createInsertSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFindWlmEnqueueBarrierPass(
        WorkloadManagementMode workloadManagementMode = WorkloadManagementMode::PWLM_V0_LCA,
        bool disableDmaSwFifo = false, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOrderBarriersForWlmPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmSplitGraphToPagesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmLegalizeSplitGraphToPagesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmLegalizePagesForBarrierDmasPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWlmInsertDummyDmasInPagesPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace vpux::VPURT::arch40xx
