//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/weights_prefetch_vf_scheduling.hpp"

namespace vpux::VPU::VF::v2 {

/*
  Scheduling scenario with full prefetching input and output dmas
*/
class FullPrefetchingVFScheduling : public WeightsPrefetchingVFScheduling {
public:
    FullPrefetchingVFScheduling(Logger log, bool prefetching);

    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    VFScenario getType() const override;

protected:
    void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const int64_t index,
                                const int64_t tilesNumber) const override;

    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     const size_t index) const override;
};
}  // namespace vpux::VPU::VF::v2
