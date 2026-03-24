//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/prefetch_lastop_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/weights_prefetch_vf_scheduling.hpp"

namespace vpux::VPU::VF::v1 {
VFSchedulingFactory::VFSchedulingFactory(bool prefetching): _prefetching(prefetching) {
}

std::shared_ptr<IVFScheduling<VFConfig>> VFSchedulingFactory::createVFScenario(VFScenario scenarioCode,
                                                                               Logger log) const {
    switch (scenarioCode) {
    case VFScenario::MINIMAL: {
        return std::make_shared<VPU::VF::v1::MinimalRequirementsVFScheduling>(log, _prefetching);
    }
    case VFScenario::LASTOP_PREFETCHING: {
        return std::make_shared<VPU::VF::v1::PrefetchingLastOpVFScheduling>(log, _prefetching);
    }
    case VFScenario::WEIGHTS_PREFETCHING: {
        return std::make_shared<VPU::VF::v1::WeightsPrefetchingVFScheduling>(log, _prefetching);
    }
    case VFScenario::FULL_PREFETCHING: {
        return std::make_shared<VPU::VF::v1::FullPrefetchingVFScheduling>(log, _prefetching);
    }
    case VFScenario::VF_PIPELINING: {
        return std::make_shared<VPU::VF::v1::PipeliningVFScheduling>(log, _prefetching);
    }
    default: {
        VPUX_THROW("No scheduling implemented for {0}", scenarioCode);
    }
    }
}
}  // namespace vpux::VPU::VF::v1
