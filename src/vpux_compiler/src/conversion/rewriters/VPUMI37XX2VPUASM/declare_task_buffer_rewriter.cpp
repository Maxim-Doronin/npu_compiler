//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareTaskBufferRewriter::symbolize(
        VPUMI37XX::DeclareTaskBufferOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op.getResult()).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::DeclareTaskBufferOp>(op.getLoc(), symName, taskIdx, op.getTaskTypeAttr(),
                                                              op.getOffsetAttr());
    rewriter.eraseOp(op);
    return SymbolizationResult(newOp);
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> DeclareTaskBufferRewriter::getSymbolicNames(
        VPUMI37XX::DeclareTaskBufferOp op, size_t) {
    return createSymbolicName(op, VPURegMapped::stringifyTaskType(op.getTaskType()).str(), /* counter */ std::nullopt);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
