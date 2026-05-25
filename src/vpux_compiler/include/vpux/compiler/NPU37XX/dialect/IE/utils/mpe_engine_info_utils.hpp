//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

namespace vpux::IE::arch37xx {
template <typename OpType>
class MPEEngineInfoOpModel : public IE::MPEEngineInfoOpInterface::ExternalModel<MPEEngineInfoOpModel<OpType>, OpType> {
public:
    mlir::Attribute getMPEEngineMode(mlir::Operation* op) const {
        return VPU::MPEEngine37XXAttr::get(
                op->getContext(), VPU::MPEEngine37XXModeAttr::get(op->getContext(), VPU::MPEEngine37XXMode::SCL));
    }
};
}  // namespace vpux::IE::arch37xx
