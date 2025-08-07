//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {

// Encode tile index and index of SHAVE unit inside the tile into a single integer for a convenient use in scheduling
// passes. The queue indexing is done in listIndex major order. If list index attribute of a shave task is not provided
// then it is assumed 0 and the encoding does not distinguish between different SHAVE FIFOs and the encoding follows
// the tileIndex.
int64_t getShaveQueueIdEncoding(int64_t numTiles, int64_t tileIndex, int64_t listIndex);
namespace VPU {
constexpr StringRef USE_DEDICATED_FIFO_PER_SHAVE_ENGINE = "VPU.UseDedicatedFifoPerShaveEngine";
bool isFifoPerShaveEngineEnabled(mlir::Operation* op);
bool hasSupportForFifoPerShaveEngine(VPU::ArchKind arch, bool enableWorkloadManagement);
}  // namespace VPU
}  // namespace vpux
