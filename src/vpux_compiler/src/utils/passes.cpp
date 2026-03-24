//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// PatternBenefit
//

const mlir::PatternBenefit vpux::benefitLow(1);
const mlir::PatternBenefit vpux::benefitMid(2);
const mlir::PatternBenefit vpux::benefitHigh(3);

// Return a pattern benefit vector from large to small
SmallVector<mlir::PatternBenefit> vpux::getBenefitLevels(uint32_t levels) {
    SmallVector<mlir::PatternBenefit> benefitLevels;
    for (const auto level : irange(levels) | reversed) {
        benefitLevels.push_back(mlir::PatternBenefit(level));
    }
    return benefitLevels;
}

// Extract benefit levels starting from startIndex, extracting numLevels elements
// For example: extractBenefitLevels([4,3,2,1,0], 1, 3) returns [3,2,1]
llvm::ArrayRef<mlir::PatternBenefit> vpux::extractBenefitLevels(llvm::ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                size_t startIndex, size_t numLevels) {
    VPUX_THROW_UNLESS(startIndex + numLevels <= benefitLevels.size(),
                      "extractBenefitLevels: startIndex {0} + numLevels {1} exceeds benefitLevels size {2}", startIndex,
                      numLevels, benefitLevels.size());
    return benefitLevels.slice(startIndex, numLevels);
}

// Extract benefit levels from a benefit vector
llvm::ArrayRef<mlir::PatternBenefit> vpux::extractBenefitLevels(llvm::ArrayRef<mlir::PatternBenefit> benefitLevels,
                                                                size_t numLevels) {
    VPUX_THROW_UNLESS(numLevels <= benefitLevels.size(),
                      "extractBenefitLevels: numLevels {0} exceeds benefitLevels size {1}", numLevels,
                      benefitLevels.size());
    return extractBenefitLevels(benefitLevels, 0, numLevels);
}

//
// FunctionPass
//

void vpux::FunctionPass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::FunctionPass::runOnOperation() {
    auto currentOp = getOperation();
    if (currentOp.isExternal()) {
        return;
    }

    auto passName = getName();
    // Enable IE_NPU_LOG_FILTER to match both formats: "PassName" (getName) and "pass-name" (getArgument)
    _log.setAlternateName(passName);
    try {
        vpux::Logger::global().trace("Started {0} pass on function '{1}'", passName, currentOp.getName());

        _log = _log.nest();
        safeRunOnFunc();
        _log = _log.unnest();
    } catch (const std::exception& e) {
        (void)errorAt(currentOp, "{0} Pass failed : {1}", passName, e.what());
        signalPassFailure();
    }
}

//
// ModulePass
//

void vpux::ModulePass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::ModulePass::runOnOperation() {
    auto currentOp = getOperation();
    auto passName = getName();
    // Enable IE_NPU_LOG_FILTER to match both formats: "PassName" (getName) and "pass-name" (getArgument)
    _log.setAlternateName(passName);
    try {
        vpux::Logger::global().trace("Started {0} pass on module '{1}'", passName, currentOp.getName());
        safeRunOnModule();
    } catch (const std::exception& e) {
        (void)errorAt(currentOp, "{0} failed : {1}", passName, e.what());
        signalPassFailure();
    }
}
