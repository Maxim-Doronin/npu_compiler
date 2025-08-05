//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v1 {
/*
  Scheduling scenario with weights prefetching
*/
class WeightsPrefetchingVFScheduling : public VFScheduling {
public:
    WeightsPrefetchingVFScheduling(Logger log, bool prefetching);

    /*
      Validate memory requirements
    */
    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    /*
      Type of scenario
    */
    VFScenario getType() const override;

protected:
    /*
      Correct prefetching cost
    */
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     const size_t index) const override;
};
}  // namespace vpux::VPU::VF::v1
