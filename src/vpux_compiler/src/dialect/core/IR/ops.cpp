//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp>
#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/interfaces/type_interfaces.hpp>

#include <vpux/compiler/utils/error.hpp>
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

mlir::LogicalResult vpux::Core::ReinterpretCastOp::verify() {
    auto inputType = getInput().getType();
    auto outputType = getOutput().getType();
    if (inputType.getTypeID() != outputType.getTypeID()) {
        return errorAt(*this, "Cannot change type id: '{0}' -> '{1}'", inputType, outputType);
    }

    const auto inNdType = mlir::cast<NDTypeInterface>(inputType);
    const auto outNdType = mlir::cast<NDTypeInterface>(outputType);
    // Note: preserve rank to satisfy potential IR requirements (e.g. 4d shapes)
    if (inNdType.getRank() != outNdType.getRank()) {
        return errorAt(*this, "Cannot cast to different rank: '{0}' -> '{1}'", inputType, outputType);
    }
    if (inNdType.getTotalAllocSize() != outNdType.getTotalAllocSize()) {
        return errorAt(*this, "Cannot cast to different allocation size: '{0}' -> '{1}'", inputType, outputType);
    }
    return mlir::success();
}

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Core::ReinterpretCastOp origOp,
                                      Core::ReinterpretCastOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-CoreReinterpretCastOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto newOutType = vpux::getBufferType(origOp.getResult().getType());
    auto newOp = rewriter.create<Core::ReinterpretCastOp>(origOp->getLoc(), newOutType, newArgs.getInput());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

void vpux::registerCoreBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Core::CoreDialect*) {
        Core::ReinterpretCastOp::attachInterface<VpuGenericOneShotBufferizeModel<Core::ReinterpretCastOp>>(*ctx);
    });
}
