//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <deque>

namespace vpux {

unsigned getClustersNumber(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

using NCETaskSignature = TaskSignature<VPUIP::NCEClusterTaskOp>;

// Base class for profiling buffer schedulers
// Algorithm is same for any amount of cluster, difference only in used types/ops
// For numClusters != 1 will be used DistributedBufferType, for single cluster ops memref. See details in
// SingleClusterScheduler
class ClusterBufferScheduler {
private:
    mlir::Type getTimestampType(unsigned dpuTasksAmount);

public:
    ClusterBufferScheduler(unsigned clusterNum, unsigned profilingWorkloadSize, mlir::OpBuilder& builder,
                           mlir::MLIRContext* ctx, vpux::IndexedSymbolAttr memKindAttr, mlir::func::FuncOp netFunc,
                           std::shared_ptr<NameUniqifier> uniqifier);

    // Return needed for storing profiling results in DDR memory size in bytes
    unsigned getRequiredDdrMemory() const;

    // Schedule next NCE Task
    void scheduleNceTask(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

    // Add needed for profiling buffers/views/copies
    void addProfilingOps(unsigned& currentDDROffset, SmallVector<mlir::Value>& clusterResults,
                         mlir::BlockArgument& profilingResult, unsigned& tasksCounter);

    static unsigned getNextBufferId();

    // In case of tests same class may be called several times, so counter will be reused. Not a problem for parser, but
    // for clarity better to reset
    static void resetBufferIdCounter();

protected:
    // Region of logic, which depends on amount of clusters. By default operates on distributed types

    NCETaskSignature getTaskSignature(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

    mlir::Operation* createAllocationOp(unsigned totalSizeCMXElements, const std::string& location);

    mlir::Value getViewToBuffer(mlir::Operation* currentProfilingBuffer, unsigned, ArrayRef<int64_t>);

    mlir::Value copyToDDR(mlir::BlockArgument& profilingResult, mlir::Operation*,
                          SmallVector<mlir::Value>& dpuProfilingOutputs, unsigned numElements, unsigned offset,
                          StringRef name);

private:
    unsigned _clustersNum;
    unsigned _profilingWorkloadSize;
    unsigned _profilingElementSize;
    std::deque<unsigned> _profilingBufferSizes;
    SmallVector<NCETaskSignature> _nceTaskSignatures;
    mlir::OpBuilder& _builder;
    mlir::MLIRContext* _ctx;
    mlir::func::FuncOp _netFunc;
    vpux::IndexedSymbolAttr _memKindAttr;
    std::shared_ptr<NameUniqifier> _uniqifier;
    static inline unsigned uniqBufferId = 0;
};

}  // namespace vpux
