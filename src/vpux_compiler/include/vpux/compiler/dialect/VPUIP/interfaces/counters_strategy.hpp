//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/statistics_collection.hpp"

#include <vector>

namespace vpux::VPUIP {

class ICountersStrategy {
public:
    virtual void appendCounters(std::vector<vpux::utils::OpCounterTree::Node>& counters) = 0;
    virtual ~ICountersStrategy() = default;

protected:
    bool isOpCounterSupported(mlir::Operation* op);
};

}  // namespace vpux::VPUIP
