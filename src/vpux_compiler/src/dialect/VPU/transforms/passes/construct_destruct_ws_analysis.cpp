//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CONSTRUCTWSANALYSIS
#define GEN_PASS_DEF_CONSTRUCTWSANALYSIS
#define GEN_PASS_DECL_DESTRUCTWSANALYSIS
#define GEN_PASS_DEF_DESTRUCTWSANALYSIS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

struct ConstructWsAnalysis final : public VPU::impl::ConstructWsAnalysisBase<ConstructWsAnalysis> {
    ConstructWsAnalysis(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnModule() final {
        std::ignore = getAnalysis<VPU::WeightsSeparationInfo>();
    }
};

struct DestructWsAnalysis final : public VPU::impl::DestructWsAnalysisBase<DestructWsAnalysis> {
    DestructWsAnalysis(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnModule() final {
        auto object = getCachedAnalysis<VPU::WeightsSeparationInfo>();
        VPUX_THROW_WHEN(!object.has_value(), "WS analysis is not cached");
        object->get().invalidate();
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConstructWsAnalysisPass(const Logger& log) {
    return std::make_unique<ConstructWsAnalysis>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPU::createDestructWsAnalysisPass(const Logger& log) {
    return std::make_unique<DestructWsAnalysis>(log);
}
