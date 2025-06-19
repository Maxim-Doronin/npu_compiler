//
// Copyright (C) 2025-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v1 {

/*
  Factory which creates VF scheduling scenario
*/

class VFSchedulingFactory {
public:
    VFSchedulingFactory(bool prefetching);
    /*
      create scheduling scenario
    */
    std::shared_ptr<IVFScheduling<VFConfig>> createVFScenario(VFScenario scenarioCode, Logger log) const;

private:
    bool _prefetching = true;
};

}  // namespace vpux::VPU::VF::v1
