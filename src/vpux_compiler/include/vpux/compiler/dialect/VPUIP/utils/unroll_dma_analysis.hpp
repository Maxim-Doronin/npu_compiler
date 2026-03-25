//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include <array>

namespace vpux {
namespace VPUIP {

enum class UnrollDMAAnalysisNeeded {
    UnrollDepthToSpaceDMAPass,
    UnrollSpaceToDepthDMAPass,
    UnrollUpsamplingDMAPass,
    UnrollPermuteDMAPass,
    UnrollExpandDMAPass,
    UnrollPerAxisTileDMAPass,
    UnrollGatherDMAPass,
    NumberOfAnalyzedPasses
};

class UnrollDMAAnalysis {
public:
    using StorageType = std::array<uint8_t, static_cast<size_t>(UnrollDMAAnalysisNeeded::NumberOfAnalyzedPasses)>;
    UnrollDMAAnalysis(mlir::Operation* operation);

    bool passNeeded(UnrollDMAAnalysisNeeded passTag);

private:
    mlir::Operation* _operation;
    StorageType _lookupArray{};
};

}  // namespace VPUIP
}  // namespace vpux
