//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/AnalysisManager.h>
#include <vector>

namespace vpux::ShaveCodeGen {

class FusionChainAnalysis {
public:
    enum class State : uint8_t { Uninitialized, ComputeOpChains, CodeGenCapsuleChains, Invalidated };

public:
    FusionChainAnalysis(mlir::Operation* op);

    void appendCodeGenCapsuleChain(std::vector<mlir::Operation*>& newChain);

    std::vector<std::vector<mlir::Operation*>> getComputeOpChains() const;
    std::vector<std::vector<mlir::Operation*>> getCodeGenCapsulesChains() const;

    // State manipulation
    void invalidate();
    void setState(State newState);
    State getState() const;

    // Preserve validity unless explicitly invalidated
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses&);

private:
    std::vector<std::vector<mlir::Operation*>> _computeOpChains;
    std::vector<std::vector<mlir::Operation*>> _codeGenCapsulesChains;

    State _state = State::Uninitialized;
};
}  // namespace vpux::ShaveCodeGen
