//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>

#include <vpux/utils/core/format.hpp>

#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/core/ops.cpp.inc>

mlir::LogicalResult vpux::Core::NestedCallOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto calleeAttr = getProperties().getCallee();
    if (!calleeAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }

    if (calleeAttr.getNestedReferences().empty()) {
        return emitOpError("'callee' must be a nested symbol");
    }

    // Check that the callee points to a mlir::func::FuncOp.
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, calleeAttr);
    if (funcOp == nullptr) {
        return emitOpError(formatv("{0} does not point to a valid 'func.func' op", calleeAttr));
    }

    // Let's keep this a bit simpler than the original mlir::func::FuncOp implementation because most developers
    // are familiar with the contract between caller and callee.
    const auto funcOpType = funcOp.getFunctionType();
    if (getOperandTypes() != funcOpType.getInputs()) {
        return emitOpError(formatv("{0} operand types do not match", calleeAttr));
    }
    if (getResultTypes() != funcOpType.getResults()) {
        return emitOpError(formatv("{0} result types do not match", calleeAttr));
    }

    return mlir::success();
}

void vpux::Core::NestedCallOp::build(mlir::OpBuilder& /*builder*/, mlir::OperationState& odsState,
                                     mlir::SymbolRefAttr callee, mlir::TypeRange results, mlir::ValueRange operands) {
    odsState.addOperands(operands);
    odsState.addTypes(results);
    auto& props = odsState.getOrAddProperties<Properties>();
    props.callee = callee;
}
