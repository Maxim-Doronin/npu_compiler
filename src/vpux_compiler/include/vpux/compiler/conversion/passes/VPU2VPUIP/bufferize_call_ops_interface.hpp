//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>

//
// getFuncOneShotAnalysisState
//

const mlir::bufferization::func_ext::FuncAnalysisState& getFuncOneShotAnalysisState(
        const mlir::bufferization::AnalysisState& state);

//
// getFuncOpAnalysisState
//

mlir::bufferization::func_ext::FuncOpAnalysisState getFuncOpAnalysisState(
        const mlir::bufferization::AnalysisState& state, mlir::func::FuncOp funcOp);

//
//  getEquivalentFuncArgIdx
//

std::optional<int64_t> getEquivalentFuncArgIdx(mlir::func::FuncOp funcOp,
                                               const mlir::bufferization::func_ext::FuncAnalysisState& state,
                                               int64_t returnValIdx);

namespace vpux {

class NestedCallOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<NestedCallOpBufferizeModel, Core::NestedCallOp> {
public:
    bool bufferizesToMemoryReadImpl(Core::NestedCallOp op, mlir::OpOperand& opOperand,
                                    const mlir::bufferization::AnalysisState& state) const;
    bool bufferizesToMemoryWriteImpl(Core::NestedCallOp op, mlir::OpOperand& opOperand,
                                     const mlir::bufferization::AnalysisState& state) const;
    mlir::bufferization::AliasingValueList getAliasingValuesImpl(Core::NestedCallOp op, mlir::OpOperand& opOperand,
                                                                 const mlir::bufferization::AnalysisState& state) const;
    mlir::LogicalResult bufferizeImpl(Core::NestedCallOp op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      mlir::bufferization::BufferizationState& state,
                                      Core::NestedCallOp::Adaptor& adaptor) const;
};

}  // namespace vpux
