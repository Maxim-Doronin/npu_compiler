//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/workload_management_status_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;
using namespace vpux::VPU;

WorkloadManagementStatus vpux::VPU::getWorkloadManagementStatus(mlir::ModuleOp moduleOp) {
    auto pipelineOptionOp = moduleOp.lookupSymbol<config::PipelineOptionsOp>(PIPELINE_OPTIONS);
    VPUX_THROW_WHEN(pipelineOptionOp == nullptr, "Failed to find PipelineOptions to fetch workload management status");

    auto wlmStatusConfigOp = pipelineOptionOp.lookupSymbol<config::OptionOp>(WORKLOAD_MANAGEMENT_STATUS);
    VPUX_THROW_WHEN(wlmStatusConfigOp == nullptr, "Failed to find config.OptionOp to fetch workload management status");

    auto wlmStatusString = mlir::dyn_cast<mlir::StringAttr>(wlmStatusConfigOp.getOptionValue());
    VPUX_THROW_WHEN(wlmStatusString == nullptr, "{0} config.OptionOp is expected to be a string, got {1}",
                    WORKLOAD_MANAGEMENT_STATUS, wlmStatusConfigOp);

    auto wlmStatus = symbolizeWorkloadManagementStatus(wlmStatusString.getValue());
    VPUX_THROW_WHEN(!wlmStatus.has_value(), "Failed to symbolize workload management status from string '{0}'",
                    wlmStatusString.getValue());

    return wlmStatus.value();
}

void vpux::VPU::setWorkloadManagementStatus(mlir::ModuleOp moduleOp, WorkloadManagementStatus value) {
    auto context = moduleOp.getContext();
    auto pipelineOptionsOp = VPU::getPipelineOptionsOp(*context, moduleOp);
    const auto attrName = mlir::StringAttr::get(context, WORKLOAD_MANAGEMENT_STATUS);
    auto attrValue = mlir::StringAttr::get(context, stringifyEnum(value));

    if (auto wlmStatusConfigOp = pipelineOptionsOp.lookupSymbol<config::OptionOp>(attrName)) {
        wlmStatusConfigOp.setOptionValueAttr(attrValue);
    } else {
        auto optionsBuilder = mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front());
        optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), attrName, attrValue);
    }
}
