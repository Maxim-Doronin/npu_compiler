//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/work_item_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult WorkItemRewriter::matchAndRewrite(VPUASM::WorkItemOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    enum TaskType : uint8_t { DPU = 0, DMA, KERNEL, SYSTEM_MANAGEMENT, UNKNOWN = 255 };

    auto realTaskIndex = origOp.getRealTaskIndex();
    auto nextWorkItemIdx = origOp.getNextWorkitemIdx();

    uint32_t nextWorkItemIdxValue = 0;

    if (nextWorkItemIdx.has_value()) {
        nextWorkItemIdxValue = nextWorkItemIdx.value().getValue();
    }

    uint64_t descPtrOffset = 0;
    TaskType workItemType;

    switch (origOp.getTaskType()) {
    case VPURegMapped::TaskType::DPUVariant:
        workItemType = TaskType::DPU;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        break;
    case VPURegMapped::TaskType::DMA:
        workItemType = TaskType::DMA;
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        workItemType = TaskType::KERNEL;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        break;
    default:
        return origOp.emitOpError("Invalid workItem task type");
    }

    WorkItem workItemDescriptor;
    workItemDescriptor.write<Fields::desc_ptr>(descPtrOffset);
    workItemDescriptor.write<Fields::wi_type>(workItemType);
    workItemDescriptor.write<Fields::wi_unit>(realTaskIndex.getTileIdx());
    workItemDescriptor.write<Fields::wi_sub_unit>(realTaskIndex.getListIdx());
    workItemDescriptor.write<Fields::next_workitem_idx>(nextWorkItemIdxValue);

    auto workItemDescriptorAttr = WorkItemAttr::get(rewriter.getContext(), std::move(workItemDescriptor));
    rewriter.create<NPUReg40XX::WorkItemOp>(origOp.getLoc(), origOp.getSymNameAttr(), origOp.getTaskTypeAttr(),
                                            origOp.getFirstTaskAttr(), workItemDescriptorAttr);
    rewriter.eraseOp(origOp);
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
