//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CORRECTNCEWORKLOADS
#define GEN_PASS_DEF_CORRECTNCEWORKLOADS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// CorrectNCEWorkloads
//

class CorrectNCEWorkloadsPass final : public VPU::impl::CorrectNCEWorkloadsBase<CorrectNCEWorkloadsPass> {
public:
    explicit CorrectNCEWorkloadsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CorrectNCEWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
    auto sparsityConstraint = strategyFactory->getSparsityConstraint();
    auto supportedChannels = strategyFactory->getSupportedChannelsDW();

    WorkloadSplitter splitter(func, supportedChannels, _log);
    splitter.correctInvalidWorkload(sparsityConstraint);
}

}  // namespace

//
// createCorrectNCEWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createCorrectNCEWorkloadsPass(Logger log) {
    return std::make_unique<CorrectNCEWorkloadsPass>(log);
}
