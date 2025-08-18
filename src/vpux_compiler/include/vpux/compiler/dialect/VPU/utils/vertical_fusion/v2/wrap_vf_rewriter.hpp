//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/wrap_vf_base_rewriter.hpp"

namespace vpux::VPU::VF::v2 {

//
// WrapVFRewriter
//

class WrapVFRewriter : public VF::WrapVFRewriterBase {
public:
    WrapVFRewriter(mlir::MLIRContext* ctx, Logger log): VF::WrapVFRewriterBase(ctx, log) {
    }

    bool opNeedsTobeWrapped(VPU::VerticalFusionOpInterface op) const override;
};
}  // namespace vpux::VPU::VF::v2
