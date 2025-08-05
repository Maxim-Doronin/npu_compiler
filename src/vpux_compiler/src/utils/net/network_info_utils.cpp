//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/net/network_info_utils.hpp"

namespace vpux::net {

namespace {
void addBlockToEmptyRegion(mlir::Region& region) {
    if (region.empty()) {
        region.emplaceBlock();
    }
}
}  // namespace

void setupSections(net::NetworkInfoOp netInfo, bool enableProfiling) {
    addBlockToEmptyRegion(netInfo.getInputsInfo());
    addBlockToEmptyRegion(netInfo.getOutputsInfo());
    if (enableProfiling) {
        VPUX_THROW_WHEN(netInfo.getProfilingOutputsInfo().empty(), "profiling_outputs_info is not created");
        addBlockToEmptyRegion(netInfo.getProfilingOutputsInfo().front());
    }
}

void eraseSectionEntries(mlir::Region& section, size_t begin) {
    VPUX_THROW_WHEN(section.getBlocks().size() != 1, "A section must have exactly 1 block");

    auto& block = section.front();
    VPUX_THROW_WHEN(block.getOperations().size() < begin, "Too many entries asked to be erased");
    for (size_t n = block.getOperations().size() - begin; n > 0; --n) {
        block.back().erase();
    }
}

}  // namespace vpux::net
