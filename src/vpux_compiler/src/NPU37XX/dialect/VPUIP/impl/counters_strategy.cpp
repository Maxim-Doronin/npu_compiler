//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/counters_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/counters_category.hpp"

namespace vpux::VPUIP::arch37xx {

void CountersStrategy::appendCounters(std::vector<vpux::utils::OpCounterTree::Node>& counters) {
    CountersVec nestedCountersSCL;
    nestedCountersSCL.push_back(VPUIP::makeCounterNode("opCount", [&](mlir::Operation* op) {
        return VPUIP::isOpCounterSupported(op);
    }));
    auto sclEnableChecker = [](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast_if_present<VPUIP::NCEClusterTaskOp>(op)) {
            if (auto mpeEngine = nceOp.getMpeEngine()) {
                if (auto attr = mlir::dyn_cast<VPU::MPEEngine37XXAttr>(*mpeEngine)) {
                    return attr.getMode().getValue() == VPU::MPEEngine37XXMode::SCL;
                }
            }
        }
        return false;
    };
    counters.push_back(VPUIP::makeCounterNode("SCL Task Op", sclEnableChecker, std::move(nestedCountersSCL)));
}
}  // namespace vpux::VPUIP::arch37xx
