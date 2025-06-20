//
// Copyright (C) 2025-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v1 {

/*
  Candidate for VF
*/
class VFCase final {
public:
    using VFConfigType = VFConfig;
    /*
      Constructor of VF case
     */
    explicit VFCase(VFConfigType& config, Dim axis);

    /*
     Destructor of VF case
    */
    ~VFCase();

    /*
     Move constructor
    */
    VFCase(VFCase&& vfCase);

    /*
     Move assignment operator
    */
    VFCase& operator=(VFCase&& other);

    /*
     Copy constructor
    */
    VFCase(const VFCase& vfCase);

    /*
     Copy assignment operator
    */
    VFCase& operator=(const VFCase& other);

    /*
     Set number of tiles
    */
    void setTilingNumber(int64_t number);

    /*
     Set VF scheduling
    */
    void setScheduling(std::shared_ptr<IVFScheduling<VFConfigType>> vfScheduling);

    /*
     Set VF tiling storage
    */
    void setTilingStorage(std::unique_ptr<TilingOperationStorage> vfStorage);

    /*
     Get VF cost
    */
    StrategyCost getCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
     Check if VF case has been initialized with scheduling
    */
    bool isInitialized();

    /*
     Get VF config
    */
    VFConfigType& getConfig();

    /*
     Generate VF tiling
    */
    mlir::ArrayAttr getTiling() const;

    /*
     Set Scheduling and tiling to VF
    */
    void approveScheduling();

    /*
     Get current tiling number
    */
    int64_t getTilingNumber() const;

private:
    /*
    Add CMX write spills
    */
    void addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log);

    /*
     Clear cached data
    */
    void clearCache();

    /*
     VF data
    */
    VFConfigType _config;

    /*
     Axis for tiling
    */
    Dim _axis;

    /*
     Number of tiles
    */
    int64_t _tilingNumber = 1;

    /*
     VF Scheduling
    */
    std::shared_ptr<IVFScheduling<VFConfigType>> _vfScheduling;

    /*
     VF TilingOperationStorage
    */
    std::unique_ptr<TilingOperationStorage> _vfTilingStorage;

    /*
     Cached VF cost
    */
    std::optional<StrategyCost> _cachedCost;
};
}  // namespace vpux::VPU::VF::v1
