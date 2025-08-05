//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef PIPELINE_OPTIONS = "Options";

vpux::config::PipelineOptionsOp getPipelineOptionsOp(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp);

// This function returns the value of the attrName constraint from PipelineOptions
template <typename T = size_t>
T getConstraint(mlir::Operation* op, StringRef attrName) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch constraint");

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(attrName);
    VPUX_THROW_WHEN(attrValue == nullptr, "Failed to find config.OptionOp attribute: {0}", attrName);
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attrValue.getOptionValue())) {
        return static_cast<T>(intAttr.getValue().getSExtValue());
    } else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attrValue.getOptionValue())) {
        return static_cast<T>(floatAttr.getValueAsDouble());
    } else if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue())) {
        return static_cast<T>(boolAttr.getValue());
    }
    VPUX_THROW("Unsupported type for constraint {0}", attrName);
}

}  // namespace VPU
}  // namespace vpux
