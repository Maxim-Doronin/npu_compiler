//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace vpux {

using TaskQueue = std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>;
enum class MinMaxOption { Min, Max };

template <typename T>
bool compareVPURTOpPosition(const T& lhs, const T& rhs, const BarrierInfo& barrierInfo, bool useIROrder = false) {
    static_assert(std::is_same_v<T, VPURT::TaskOp> || std::is_same_v<T, VPURT::DeclareVirtualBarrierOp> ||
                          std::is_same_v<T, size_t>,
                  "Unsupported type for comparison");

    if constexpr (std::is_same_v<T, VPURT::DeclareVirtualBarrierOp>) {
        return barrierInfo.getIndex(lhs) < barrierInfo.getIndex(rhs);
    } else if constexpr (std::is_same_v<T, size_t>) {
        return lhs < rhs;
    } else if constexpr (std::is_same_v<T, VPURT::TaskOp>) {
        if (useIROrder) {
            // Use IR order for comparison
            return lhs->isBeforeInBlock(rhs);
        } else {
            return barrierInfo.getIndex(lhs) < barrierInfo.getIndex(rhs);
        }
    }
}

ExecutionGroupListMap createSWTaskExecutionGroups(mlir::func::FuncOp netFunc);
ExecutionGroupListMap createDPUTaskExecutionGroups(mlir::func::FuncOp netFunc);

void updateBarriersForDma(const SmallVector<size_t>& consumes, const SmallVector<size_t>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo);
void updateBarriersForDma(const SmallVector<mlir::Value>& consumes, const SmallVector<mlir::Value>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo);

// Fetch tasks are only attached to DMAs on port 0 and list 0 in later dialect
// In this context supportedDMA is a DMA which has channel DDR and port 0
bool isDMAOnSupportedPortAndChannel(VPURT::TaskOp dmaTaskOp);

// Function to find min or max position in a vector of TaskOps
VPURT::TaskOp findMinMaxPosition(const SmallVector<size_t>& dmas, BarrierInfo& barrierInfo, MinMaxOption option);

// Check if two list of barriers have any barrier in common and return earliest of them
std::optional<size_t> getEarliestCommonBarrier(const BarrierInfo::TaskSet& taskSetOne,
                                               const BarrierInfo::TaskSet& taskSetTwo, BarrierInfo& barrierInfo);

// Return last task that updates series of barriers
// As we check for the last DMA, sort the vector and use the last one
mlir::Operation* findLastTaskToUpdate(const BarrierInfo::TaskSet& barriers, BarrierInfo& barrierInfo);

// Create new barrier and add producer and consumer
VPURT::DeclareVirtualBarrierOp createNewBarrier(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                mlir::Operation* insertionPoint, VPURT::TaskOp producer,
                                                VPURT::TaskOp consumer);

void addElementsToSet(BarrierInfo::TaskSet& targetSet, const BarrierInfo::TaskSet& sourceSet);
bool lastTaskInGroupHasMandatoryUpdateBarrier(const ExecutionGroup& executionGroup, BarrierInfo& barrierInfo);
bool inSameTaskBlock(size_t task1, size_t task2, const BlockRange& blockRange);

}  // namespace vpux
