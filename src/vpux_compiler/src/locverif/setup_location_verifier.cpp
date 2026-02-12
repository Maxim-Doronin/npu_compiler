//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/locverif/locations_verifier.hpp"
#include "vpux/compiler/locverif/passes.hpp"

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/error.hpp"

#include <memory>

using namespace vpux;

namespace {

enum class LocationsVerifierMarker { BEGIN, END };

//
// SetupLocationVerifierPass
//

class SetupLocationVerifierPass final : public ModulePass {
public:
    SetupLocationVerifierPass(const Logger& log, LocationsVerifierMarker marker, LocationsVerificationMode mode)
            : ModulePass(mlir::TypeID::get<SetupLocationVerifierPass>()), _mode(mode), _marker(marker) {
        ModulePass::initLogger(log, "setup-location-verifier");
    }
    SetupLocationVerifierPass(const SetupLocationVerifierPass& other)
            : ModulePass(other), _mode(other._mode), _marker(other._marker) {
    }
    SetupLocationVerifierPass(SetupLocationVerifierPass&&) = delete;
    SetupLocationVerifierPass& operator=(const SetupLocationVerifierPass&) = delete;
    SetupLocationVerifierPass& operator=(SetupLocationVerifierPass&&) = delete;
    ~SetupLocationVerifierPass() {
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override {
        if (mlir::failed(ModulePass::initialize(ctx))) {
            return mlir::failure();
        }
        if (modeOption.hasValue()) {
            _mode = getLocationsVerificationMode(modeOption);
        }
        return mlir::success();
    }

    StringRef getDescription() const override {
        return "Setup Location Verifier Pass";
    }

    StringRef getName() const override {
        if (_marker == LocationsVerifierMarker::BEGIN) {
            return "StartLocationVerifierPass";
        } else {
            return "StopLocationVerifierPass";
        }
    }

    StringRef getArgument() const override {
        return "setup-location-verifier";
    }

    std::unique_ptr<mlir::Pass> clonePass() const override {
        return std::make_unique<SetupLocationVerifierPass>(*this);
    }

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetupLocationVerifierPass)
private:
    void safeRunOnModule() override;

    LocationsVerificationMode _mode;
    LocationsVerifierMarker _marker;

    mlir::Pass::Option<std::string> modeOption{*this, "mode", ::llvm::cl::desc("Location verifier mode")};
};

void SetupLocationVerifierPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    const auto currentMode = getLocationsVerificationMode(moduleOp);
    // Running full verification if verification was enabled before
    if (_marker == LocationsVerifierMarker::END && currentMode != LocationsVerificationMode::OFF) {
        const auto verificationResult = verifyLocationsUniquenessFull(moduleOp, getName());
        if (mlir::failed(verificationResult)) {
            signalPassFailure();
        }
    }
    setLocationsVerificationMode(moduleOp, _mode);
}

std::unique_ptr<mlir::Pass> createSetupLocationVerifierPass(const Logger& log) {
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::BEGIN,
                                                       LocationsVerificationMode::OFF);
}

}  // namespace

namespace vpux::locverif {

std::unique_ptr<mlir::Pass> createStartLocationVerifierPass(
        const Logger& log, const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode) {
    const auto mode = getLocationsVerificationMode(locationsVerificationMode);
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::BEGIN, mode);
}

std::unique_ptr<mlir::Pass> createStopLocationVerifierPass(const Logger& log) {
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::END,
                                                       LocationsVerificationMode::OFF);
}

void registerPasses() {
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createSetupLocationVerifierPass(Logger::global());
    });
}

}  // namespace vpux::locverif
