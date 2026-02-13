//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/ungroup_buffer_section_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void VPUIP::UngroupBufferSectionStrategy::registerRewriters(RewriterRegistry& registry, Logger& log) const {
    if (_enableReorderSubViewOp) {
        vpux::VPUIP::registerMoveSubViewBeforeSparseBufferRewriters(registry, log);
    }
    vpux::VPUIP::registerUngroupSparseBufferRewriters(registry, log);
}

std::unique_ptr<IDynamicRewriterStrategy> VPUIP::createUngroupBufferSectionStrategy(bool enableReorderSubViewOp) {
    return std::make_unique<VPUIP::UngroupBufferSectionStrategy>(enableReorderSubViewOp);
}
