//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"

namespace vpux::VPU {

// Empty constructor to satisfy getAnalysis API requirements.
SiblingOpsAnalysis::SiblingOpsAnalysis(mlir::Operation*) {
}

std::set<ClusteredOpInterface> SiblingOpsAnalysis::lookupOpSiblings(mlir::Operation* op) {
    if (op == nullptr) {
        return {};
    }
    if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        for (const auto& siblingGroup : _siblingGroups) {
            if (siblingGroup.find(clusteredOp) != siblingGroup.end()) {
                return siblingGroup;
            }
        }
    }

    return {};
}

mlir::Operation* SiblingOpsAnalysis::getConsumerOp(ClusteredOpInterface clusteredOp) {
    if (isPassthroughOp(clusteredOp.getOperation())) {
        // For passthrough ops, ensure input and output tensors use the same pool of ops to
        // determine the distribution
        return clusteredOp.getOperation();
    } else {
        for (const auto& consumer : clusteredOp->getUsers()) {
            // find first valid consumer and use it to get all its clustered siblings
            if (mlir::isa<ClusteredOpInterface>(consumer) || isPassthroughOp(consumer)) {
                return consumer;
            }
        }
    }
    return nullptr;
}

std::set<ClusteredOpInterface> SiblingOpsAnalysis::getSiblings(ClusteredOpInterface clusteredOp) {
    auto siblings = lookupOpSiblings(clusteredOp);
    if (siblings.empty()) {
        siblings = getSiblingOps(clusteredOp);
        _siblingGroups.emplace_back(siblings);
    }
    return siblings;
}

std::set<ClusteredOpInterface> SiblingOpsAnalysis::getConsumers(ClusteredOpInterface clusteredOp) {
    if (_consumers.contains(clusteredOp)) {
        return _consumers.at(clusteredOp);
    }
    auto consumerOp = getConsumerOp(clusteredOp);
    if (consumerOp == nullptr) {
        return {};
    }
    auto consumers = lookupOpSiblings(consumerOp);
    if (consumers.empty()) {
        consumers = getSiblingOps(consumerOp);
        _siblingGroups.emplace_back(consumers);
    }
    _consumers.insert(std::make_pair(clusteredOp, consumers));
    return consumers;
}

EagerSiblingOpsAnalysis::EagerSiblingOpsAnalysis(mlir::func::FuncOp func): SiblingOpsAnalysis(func) {
    // Skip siblings for architectures that do not support halo overlap to save some compile time
    if (!outputOverlappedParamsIsHaloSupported(func)) {
        return;
    }
    func->walk([&](VPU::ClusteredOpInterface clusteredOp) {
        if (lookupOpSiblings(clusteredOp.getOperation()).empty()) {
            auto opSiblings = getSiblingOps(clusteredOp.getOperation());
            _siblingGroups.emplace_back(opSiblings);
        }
        auto consumerOp = getConsumerOp(clusteredOp);
        if (consumerOp == nullptr) {
            return;
        }
        auto consumerSiblings = lookupOpSiblings(consumerOp);
        if (consumerSiblings.empty()) {
            consumerSiblings = getSiblingOps(consumerOp);
            _siblingGroups.emplace_back(consumerSiblings);
        }
        _consumers.insert(std::make_pair(clusteredOp, consumerSiblings));
    });
}

std::set<ClusteredOpInterface> EagerSiblingOpsAnalysis::getSiblings(ClusteredOpInterface clusteredOp) {
    return lookupOpSiblings(clusteredOp);
}

std::set<ClusteredOpInterface> EagerSiblingOpsAnalysis::getConsumers(ClusteredOpInterface clusteredOp) {
    if (_consumers.contains(clusteredOp)) {
        return _consumers.at(clusteredOp);
    }
    return {};
}
}  // namespace vpux::VPU
