//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <functional>
#include <map>
namespace vpux::VPUIP {
#define GEN_PASS_DECL_SYNCSHVDPU
#define GEN_PASS_DEF_SYNCSHVDPU
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class SyncShvDpuPass final : public VPUIP::impl::SyncShvDpuBase<SyncShvDpuPass> {
public:
    explicit SyncShvDpuPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
//
// SyncShvDpuPass
//

void SyncShvDpuPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    auto bufferOps = func.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &func.getBody().front().front();
    auto barriersOps = func.getOps<VPURT::DeclareVirtualBarrierOp>();
    auto barrierInsertionPoint = !barriersOps.empty() ? *barriersOps.begin() : &func.getBody().front().front();
    mlir::Value inBuffer = nullptr;
    mlir::Value outBuffer = nullptr;
    func->walk([&](VPURT::TaskOp taskOpS) {
        if (isDpuShaveKernelType(taskOpS)) {
            auto swUpBarrier = taskOpS.getUpdateBarriers();
            _log.trace("SyncShvDpuPass: Shave op found: {0}", taskOpS);
            // Create dummy input and output buffer that will hold value of size 0
            if ((inBuffer == nullptr) || (outBuffer == nullptr)) {
                inBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
                outBuffer = VPUIP::createDummyBuffer(builder, bufferInsertionPoint);
            }
            builder.setInsertionPoint(barrierInsertionPoint);
            auto newSyncBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOpS.getLoc());
            auto syncTaskInsertionPoint = taskOpS;
            builder.setInsertionPointAfter(syncTaskInsertionPoint);
            VPUIP::createSyncDMA(builder, inBuffer, outBuffer, 0, {newSyncBarrier}, {swUpBarrier});
            // Add sync task
            // TODO : E#-164934 Selective insert if consumer is DPU task.
            taskOpS.getUpdateBarriersMutable().assign(newSyncBarrier);
        }
    });
}

}  // namespace

//
// createSyncShvDpuPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSyncShvDpuPass(Logger log) {
    return std::make_unique<SyncShvDpuPass>(log);
}
