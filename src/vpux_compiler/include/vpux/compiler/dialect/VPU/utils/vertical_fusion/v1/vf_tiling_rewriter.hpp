//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_tiling_base_rewriter.hpp"

namespace vpux::VPU::VF::v1 {

//
// VerticalFusionTilingRewriter
//

class VerticalFusionTilingRewriter : public VerticalFusionTilingRewriterBase<VFConfig, VFSchedulingFactory> {
public:
    VerticalFusionTilingRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : VerticalFusionTilingRewriterBase<VFConfig, VFSchedulingFactory>(ctx, enableVerticalFusionPipelining,
                                                                              costFunction, log) {
    }

protected:
    std::pair<DimArr, int64_t> getDimsData(ArrayRef<int64_t> strategy) const override {
        auto dim = getVFTilingDim(strategy);
        VPUX_THROW_WHEN(!dim.has_value(), "There is no tiling for VF");
        DimArr dims = {dim.value()};
        return std::make_pair(dims, strategy[dim.value().ind()]);
    }
};
}  // namespace vpux::VPU::VF::v1
