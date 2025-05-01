//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <set>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
namespace vpux::VPU {

class ClusteredOpInterface;

// Analysis which finds clustered op siblings and consumers set.
// Siblings and consumers are computed lazily and cached in _siblingGroups.
// Be careful not to introduce or remove clustered ops into IR when using this class, if
// siblings were already cached it might lead to missed ops or to invalid ops being returned.
class SiblingOpsAnalysis {
public:
    explicit SiblingOpsAnalysis(mlir::Operation*);

    // Get clustered ops that are siblings to the passed clustered op.
    virtual std::set<ClusteredOpInterface> getSiblings(ClusteredOpInterface);

    // Get clustered ops that consume result of the passed clustered op.
    // Clustered op is considered to be a consumer if it consumes result directly
    // or if it consumes a result of the view-like op which in turn consumes result
    // of the passed op like in following pattern:
    // ClusteredOp(producer) -> View-like op -> ClusteredOp(consumer)
    virtual std::set<ClusteredOpInterface> getConsumers(ClusteredOpInterface);

protected:
    std::set<ClusteredOpInterface> lookupOpSiblings(mlir::Operation* op);
    mlir::Operation* getConsumerOp(ClusteredOpInterface);
    std::vector<std::set<ClusteredOpInterface>> _siblingGroups{};
    llvm::DenseMap<ClusteredOpInterface, std::set<ClusteredOpInterface>> _consumers{};
};

// Eager version of SiblingOpsAnalysis which doesn't modify internal state
// on calls to getConsumers/getSiblings. Suitable for usage with multithreading.
class EagerSiblingOpsAnalysis : public SiblingOpsAnalysis {
public:
    explicit EagerSiblingOpsAnalysis(mlir::func::FuncOp func);
    std::set<ClusteredOpInterface> getSiblings(ClusteredOpInterface) override;
    std::set<ClusteredOpInterface> getConsumers(ClusteredOpInterface) override;
};
}  // namespace vpux::VPU
