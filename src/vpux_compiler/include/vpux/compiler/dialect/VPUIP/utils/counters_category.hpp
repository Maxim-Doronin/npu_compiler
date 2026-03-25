//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/statistics_collection.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <string>
#include <vector>

namespace vpux::VPUIP {

using CountersNode = vpux::utils::OpCounterTree::Node;
using CountersVec = std::vector<CountersNode>;

class SpecificCategoryCounter final : public vpux::utils::OpCounter {
public:
    using vpux::utils::OpCounter::OpCounter;

    // Records an operation by incrementing its specific counters. Returns true if the operation was successfully
    // recorded, false otherwise
    bool record(mlir::Operation* op) override;

    // Logs collected statistics in different formats depending on the op category
    void printStatistics(const vpux::Logger& log) const override;

private:
    uint64_t _size{0};
    uint64_t _opCount{0};
};

// Determines if the category represents a DMA operation with non-zero size and is used to decide whether to print size
// information in statistics. Returns true if the category is related to DMA operations with non-zero size, false
// otherwise.
bool printDMASizes(const std::string& category, const uint64_t& size);

// Converts byte count to human-readable format with 2 decimal places and appropriate units (KB, MB, GB).
std::string convertBytesToReadableSize(uint64_t bytes);

// Creates a counter tree node with the specified category, predicate for determining if an operation is part of that
// category, optional nested counters which are subcategories of a broader category (NCEClusterTaskOp [broad category]
// --> CONV [child]), and an optional handler for operations that match this category but don't match any nested
// counter.
CountersNode makeCounterNode(const std::string& category, vpux::utils::OpCounter::IsOperationSuitable predicate,
                             CountersVec&& nestedCounters = {},
                             vpux::utils::OpCounter::HandleUnrecognizedCounter handler = {});

// Checks if an NCE operation is supported for counting based on its task type.
// Supported task types include CONV, DWCONV, ELTWISE, MAXPOOL, and AVEPOOL. Returns true if the operation is supported,
// false otherwise.
bool isOpCounterSupported(mlir::Operation* op);

}  // namespace vpux::VPUIP
