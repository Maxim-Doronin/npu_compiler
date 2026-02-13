//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_call_ops_interface.hpp"

using namespace vpux;

//
// One-shot-bufferization utilities for NestedCallOp bufferization
//

//
// getFuncOneShotAnalysisState
//

const mlir::bufferization::func_ext::FuncAnalysisState& getFuncOneShotAnalysisState(
        const mlir::bufferization::AnalysisState& state) {
    VPUX_THROW_WHEN(!mlir::isa<mlir::bufferization::OneShotAnalysisState>(state), "Expected OneShotAnalysisState");

    auto* result = static_cast<const mlir::bufferization::OneShotAnalysisState&>(state)
                           .getExtension<mlir::bufferization::func_ext::FuncAnalysisState>();
    VPUX_THROW_WHEN(result == nullptr, "FuncAnalysisState does not exist");

    return *result;
}

//
// getFuncOpAnalysisState
//

mlir::bufferization::func_ext::FuncOpAnalysisState getFuncOpAnalysisState(
        const mlir::bufferization::AnalysisState& state, mlir::func::FuncOp funcOp) {
    if (!mlir::isa<mlir::bufferization::OneShotAnalysisState>(state)) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    auto* funcState = static_cast<const mlir::bufferization::OneShotAnalysisState&>(state)
                              .getExtension<mlir::bufferization::func_ext::FuncAnalysisState>();
    if (!funcState) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    const auto& analyzedFuncOps = funcState->analyzedFuncOps;
    auto it = analyzedFuncOps.find(funcOp);
    if (it == analyzedFuncOps.end()) {
        return mlir::bufferization::func_ext::FuncOpAnalysisState::NotAnalyzed;
    }
    return it->second;
}

//
//  getEquivalentFuncArgIdx
//

std::optional<int64_t> getEquivalentFuncArgIdx(mlir::func::FuncOp funcOp,
                                               const mlir::bufferization::func_ext::FuncAnalysisState& state,
                                               int64_t returnValIdx) {
    auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
    if (funcOpIt == state.equivalentFuncArgs.end()) {
        // No equivalence info stores for funcOp.
        return std::nullopt;
    }

    auto retValIt = funcOpIt->getSecond().find(returnValIdx);
    if (retValIt == funcOpIt->getSecond().end()) {
        // Return value has no equivalent bbArg.
        return std::nullopt;
    }

    return retValIt->getSecond();
}
